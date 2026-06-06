import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import datetime
from cs336_bmine.langmodel import TransformerLM
from cs336_bmine.train_util import save_model, cross_entropy, AdamW
import numpy as np
from typing import Type, Any, Optional, Callable
from functools import partial

class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        '''
        Given an instantiated PyTorch nn.Module to be parallelized, construct an
        FSDP module that will handle weight all-gathers and gradient
        reduce-scatters. Make sure that your hooks or your module wrappers
        all-gather the weights in time for the forward pass. To limit memory use,
        only start gathering after the layer two before the current one has
        completed its forward pass.  In the backward pass, your hooks or module
        wrappers should all-gather to have the weights available for the
        computation. When the gradients are available, they should be
        reducescattered to the appropriate ranks. Make sure to free the gathered
        weights after use. When compute_dtype is provided, cast the weights to that
        dtype before communicating or using them for compute, while keeping master
        weights and optimizer updates in FP32.
        '''
        super().__init__()
        self.device = device

        self.module = module
        self.named_modules_dict = {}
        self.first_fw_i = 0
        self.first_bw_i = 0
        self.active_contexts = {}

        self.num_sharded_modules = 0

        self.fw_i_to_name = {}
        self.bw_i_to_name = {}
        self.fw_name_to_i = {}
        self.bw_name_to_i = {}

        self.prefetch_in_process = {}
        self.prefetch_handles = {}
        self.prefetch_params = {}
        self.param_pads = {}
        self.param_shapes = {}
        
        self.grad_scatter_handles = []

        self.master_weights = {}
        self.sharded_param_ids = set()
        self.training_state = 'IDLE'
        self.world_size = dist.get_world_size()
        self.compute_dtype = compute_dtype

        for name, submodule in module.named_modules():
            self.named_modules_dict[name] = submodule
            owned_params = list(submodule.parameters(recurse=False))
            if len(owned_params) == 0:
                continue

            has_sharded_param = False
            has_meta_param = False
            ws = dist.get_world_size()
            for pname, param in submodule.named_parameters(recurse=False):

                if self.device is None:
                    self.device = param.device

                if len(param.shape) > 1 and param.shape[0] >= ws:
                    has_sharded_param = True
                
                if param.device.type == 'meta':
                    has_meta_param = True

            physical_mod = submodule
            self.master_weights[name] = {}
            if has_meta_param:
                physical_mod = submodule.to_empty(device=self.device)
                if dist.get_rank() == 0:
                    physical_mod.reset_parameters()

                for pname, param in physical_mod.named_parameters(recurse=False):
                    dist.broadcast(param.data, src=0)
                    if len(param.shape) == 1 or param.shape[0] < ws:
                        submodule._parameters[pname] = param
                        self.master_weights[name][pname] = param
                
            if not has_sharded_param:
                continue

            self.num_sharded_modules += 1
            self.prefetch_handles[name] = {}
            self.prefetch_params[name] = {}
            self.param_pads[name] = {}
            self.param_shapes[name] = {}
            self.prefetch_in_process[name] = False

            for pname, param in physical_mod.named_parameters(recurse=False):
                if len(param.shape) == 1 or param.shape[0] < ws:
                    physical_mod._parameters[pname] = submodule._parameters[pname]
                    continue
                dist.broadcast(param.data, src=0)
                dim1 = param.shape[0]
                shard_remainder = dim1 % ws
                self.param_pads[name][pname] = 0
                self.param_shapes[name][pname] = param.shape
                if shard_remainder != 0:
                    shard_size = (dim1 // ws) + 1
                    param_pad = ws * shard_size - dim1
                    self.param_pads[name][pname] = param_pad
                    self.param_shapes[name][pname] = [param.shape[0] + param_pad] + \
                        list(param.shape[1:])
                else:
                    shard_size = dim1 // ws
                shard_shape = [shard_size] + list(param.shape[1:])
                requires_grad = param.requires_grad
                ndim_param = torch.nn.Parameter(
                    torch.empty(size=shard_shape, dtype=param.dtype, device=self.device),
                    requires_grad=requires_grad
                )
                shard_param = torch.nn.Parameter(
                    torch.empty(size=(ndim_param.numel(),), dtype=param.dtype, device=self.device),
                    requires_grad=requires_grad
                )
                del ndim_param
                scatter_list = None
                if dist.get_rank() == 0:
                    scatter_list = [
                        param.data[i*shard_size:(i+1)*shard_size, ...]
                        for i in range(ws)
                    ]
                    ndims = len(param.shape)
                    if shard_remainder != 0:
                        pad_list = [0] * (ndims - 1) * 2 + [0, self.param_pads[name][pname]]
                        scatter_list[-1] = torch.nn.functional.pad(scatter_list[-1], pad=pad_list)
                    scatter_list = [sle.flatten() for sle in scatter_list]
                dist.scatter(shard_param.data, scatter_list, src=0)
                self.master_weights[name][pname] = shard_param
                self.sharded_param_ids.add(id(shard_param))
                self.prefetch_handles[name][pname] = None

                submodule._parameters[pname] = shard_param

                fw_pre_hook = partial(self.forward_pre_hook, name)
                fw_post_hook = partial(self.forward_post_hook, name)
                bw_pre_hook = partial(self.backward_pre_hook, name)
                bw_post_hook = partial(self.backward_post_hook, name)

                submodule.register_forward_pre_hook(fw_pre_hook)
                submodule.register_forward_hook(fw_post_hook)
                submodule.register_full_backward_pre_hook(bw_pre_hook)
                submodule.register_full_backward_hook(bw_post_hook)

        # Ensure all replicated (non-sharded) parameters are synced
        for param in self.module.parameters():
            if id(param) not in self.sharded_param_ids:
                # 1. Broadcast to guarantee identical starting weights across ranks
                if param.device.type != 'meta':
                    dist.broadcast(param.data, src=0)
                
                # 2. Attach an all_reduce hook to average their local gradients
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._sync_grad)

    def _sync_grad(self, param_tensor: torch.Tensor):
        param_tensor.grad.data.div_(self.world_size)
        handle = dist.all_reduce(
            param_tensor.grad.data, op=dist.ReduceOp.SUM, async_op=True
        )
        self.grad_scatter_handles.append(handle)
        return

    def _fetch_param(self, name, pname):
        mw = self.master_weights[name][pname]
        with torch.no_grad():
            if self.compute_dtype is not None:
                dtype = self.compute_dtype
                mw_casted = mw.to(self.compute_dtype)
            else:
                dtype = mw.dtype
                mw_casted = mw
            padded_numels = self.world_size * mw_casted.numel()
            self.prefetch_params[name][pname] = torch.empty((padded_numels,), dtype=dtype, device=self.device)
            handle = dist.all_gather_into_tensor(self.prefetch_params[name][pname], mw_casted, async_op=True)
            self.prefetch_handles[name][pname] = handle
        return

    def _wait_build_param(self, name, pname, is_backward=False):
        handle = self.prefetch_handles[name][pname]
        with torch.no_grad():
            handle.wait()
        self.prefetch_handles[name][pname] = None
        base_tensor = self.prefetch_params[name][pname]
        mw = self.master_weights[name][pname]
        
        base_tensor.requires_grad_(mw.requires_grad)
        
        # FIX: Attach the hook to the leaf tensor during the FORWARD pass.
        # This guarantees it catches the gradient from the autograd graph.
        if base_tensor.requires_grad and not is_backward:
            def _post_backward_grad_hook(grad):
                with torch.no_grad():
                    if mw.grad is None:
                        mw.grad = torch.zeros_like(mw)
                    
                    # 'grad' here is the gradient of base_tensor. 
                    # It is inherently flat and padded exactly as we need!
                    grad_to_scatter = grad.contiguous().to(mw.grad.dtype) / self.world_size
                    handle = dist.reduce_scatter_tensor(
                        mw.grad, grad_to_scatter,
                        op=dist.ReduceOp.SUM, async_op=True,
                    )
                    self.grad_scatter_handles.append(handle)

                    # Swap back to master weight so the optimizer finds it
                    self.named_modules_dict[name]._parameters[pname] = mw
                return grad
                
            base_tensor.register_hook(_post_backward_grad_hook)

        padded_param = base_tensor.reshape(self.param_shapes[name][pname])
        if self.param_pads[name][pname] != 0:
            true_param = padded_param[:-self.param_pads[name][pname], ...]
        else:
            true_param = padded_param
            
        if true_param.requires_grad:
            true_param.retain_grad()
                
        return true_param
   
    def _check_prefetch(self, name, tag_fsdp=False):
        for pname in self.master_weights[name]:
            fetch_handle = self.prefetch_handles[name].get(pname, None)
            if fetch_handle is None:
                self._fetch_param(name, pname)
        
        for pname, _ in self.prefetch_handles[name].items():
            true_param = self._wait_build_param(name, pname)
            if tag_fsdp:
                true_param._fsdp_tag = (name, pname)
            self.named_modules_dict[name]._parameters[pname] = true_param
        return

    def pack_hook(self, tensor):
        if hasattr(tensor, '_fsdp_tag'):
            # This is our massive weight. 
            # Return a lightweight tuple so autograd drops the tensor.
            return ("FSDP_WEIGHT", tensor._fsdp_tag)
        
        # This is a normal activation. Return it as-is so autograd keeps it.
        return ("ACTIVATION", tensor)

    def unpack_hook(self, packed_obj):
        obj_type, payload = packed_obj
        if obj_type == "ACTIVATION":
            return payload
            
        if obj_type == "FSDP_WEIGHT":
            name, pname = payload

            handle = self.prefetch_handles[name].get(pname, None)
            if handle is None:
                self._fetch_param(name, pname)
                
            # FIX: Pass is_backward=True
            rebuilt_full_weight = self._wait_build_param(name, pname, is_backward=True)

            self.named_modules_dict[name]._parameters[pname] = rebuilt_full_weight
            return rebuilt_full_weight

    def forward_pre_hook(self, name, module, args):
        if self.training_state == 'FORWARD':
            if name not in self.fw_name_to_i:
                self.fw_i_to_name[self.first_fw_i] = name
                self.fw_name_to_i[name] = self.first_fw_i
                self.first_fw_i += 1
        elif self.training_state == 'BACKWARD':
            if name not in self.bw_name_to_i:
                self.bw_i_to_name[self.first_bw_i] = name
                self.bw_name_to_i[name] = self.first_bw_i
                self.first_bw_i += 1
                
        ctx = torch.autograd.graph.saved_tensors_hooks(self.pack_hook, self.unpack_hook)
        ctx.__enter__()
        
        # Store it so we can exit later
        self.active_contexts[name] = ctx
       
        self._check_prefetch(name, tag_fsdp=True)

        self.prefetch_in_process[name] = False
        return
    
    def forward_post_hook(self, name, module, args, output):
        fetch_names = []
        self.prefetch_in_process[name] = False
        if self.training_state == 'FORWARD':
            name_to_i = self.fw_name_to_i
            i_to_name = self.fw_i_to_name
        elif self.training_state == 'BACKWARD':
            name_to_i = self.bw_name_to_i
            i_to_name = self.bw_i_to_name

        i_exec = name_to_i[name]
        for i in (i_exec+1, i_exec+2):
            if i in i_to_name:
                cand = i_to_name[i]
                if not self.prefetch_in_process[cand]:
                    fetch_names.append(cand)

        for fname in fetch_names:
            for pname in self.master_weights[fname]:
                fetch_handle = self.prefetch_handles[fname].get(pname, None)
                if fetch_handle is None:
                    self._fetch_param(fname, pname)
            self.prefetch_in_process[fname] = True

        if name in self.active_contexts:
            self.active_contexts[name].__exit__(None, None, None)
            del self.active_contexts[name]
        
        # put back the master weights, destroy the collected param
        for pname in self.prefetch_params[name]:
            module._parameters[pname] = self.master_weights[name][pname]
            param = self.prefetch_params[name][pname]
            self.prefetch_params[name][pname] = None
            del param
        
        return
    
    def backward_pre_hook(self, name, module, grad_output):
        if self.training_state == 'FORWARD':
            self.training_state = 'BACKWARD'

        if name not in self.bw_name_to_i:
            self.bw_i_to_name[self.first_bw_i] = name
            self.bw_name_to_i[name] = self.first_bw_i
            self.first_bw_i += 1
        
        self.prefetch_in_process[name] = False
        return

    def backward_post_hook(self, name, module, grad_input, grad_output):
        fetch_names = []
        self.prefetch_in_process[name] = False
        name_to_i = self.bw_name_to_i
        i_to_name = self.bw_i_to_name

        i_exec = name_to_i[name]
        for i in (i_exec+1, i_exec+2):
            if i in i_to_name:
                cand = i_to_name[i]
                if not self.prefetch_in_process[cand]:
                    fetch_names.append(cand)

        for fname in fetch_names:
            for pname in self.master_weights[fname]:
                fetch_handle = self.prefetch_handles[fname].get(pname, None)
                if fetch_handle is None:
                    self._fetch_param(fname, pname)
            self.prefetch_in_process[fname] = True
        
        return

    def forward(self, *inputs, **kwargs):
        self.training_state = "FORWARD"
        # 1. Run the forward pass
        outputs = self.module(*inputs, **kwargs)
        # 2. Attach a trigger to the output tensor
        if isinstance(outputs, torch.Tensor) and outputs.requires_grad:
            def _backward_start_trigger(grad):
                self.training_state = "BACKWARD"
                return grad # Tensor hooks must return the gradient
            outputs.register_hook(_backward_start_trigger)
        return outputs

    def finish_gradient_synchronization(self):
        '''
        When called, wait for asynchronous communication calls to finish on the GPU.
        '''
        for handle in self.grad_scatter_handles:
            handle.wait()
        self.grad_scatter_handles.clear()
        self.training_state = 'FORWARD'
    
    def gather_full_params(self):
        full_state_dict = self.module.state_dict()
        holders = {}
        handles = {}
        shapes = {}
        pads = {}
        with torch.no_grad():
            for name in self.master_weights:
                for pname, param in self.master_weights[name].items():
                    if pname not in self.param_shapes[name]: # has sharding
                        continue
                    numels = self.world_size * param.numel()
                    holder = torch.empty((numels,), dtype=param.dtype, device=self.device)
                    handle = dist.all_gather_into_tensor(holder, param, async_op=True)
                    holders[f'{name}.{pname}'] = holder
                    handles[f'{name}.{pname}'] = handle
                    shapes[f'{name}.{pname}'] = self.param_shapes[name][pname]
                    pads[f'{name}.{pname}'] = self.param_pads[name][pname]
            
            for fpname, handle in handles.items():
                handle.wait()
                holder = holders[fpname]
                reshaped_tensor = holder.reshape(shapes[fpname])
                if pads[fpname] != 0:
                    reshaped_tensor = reshaped_tensor[:-pads[fpname], ...]
                full_state_dict[fpname] = reshaped_tensor

        handles.clear()
        return full_state_dict


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        '''
        Initializes the sharded state optimizer.
        params is a collection of parameters to be optimized
        (or parameter groups, in case the user wants to use different hyperparameters,
        such as learning rates, for different parts of the model);
        these parameters will be sharded across all the ranks.
        The optimizer_cls parameter specifies the type of
        optimizer to be wrapped (e.g., optim.AdamW).
        Finally, any remaining keyword arguments are forwarded to the
        constructor of the optimizer_cls. Make sure to
        call the torch.optim.Optimizer super-class constructor in this method.
        '''
        self.num_params = 0
        self.rank_param_groups = []
        self.all_param_groups = []
        super().__init__(params, {})
        self.optimizer = optimizer_cls(self.rank_param_groups, **kwargs)

    def step(self, closure: Optional[Callable] = None, **kwargs):
        '''
        Calls the wrapped optimizer's step() method with the provided closure
        and keyword arguments. After updating the parameters, synchronize with
        the other ranks.
        '''
        self.optimizer.step(closure, **kwargs)
        for param_group in self.all_param_groups:
            for param, prank in zip(param_group['params'], param_group['ranks']):
                dist.broadcast(param.data, src=prank)
        return

    def add_param_group(self, param_group: dict[str, Any]):
        '''
        This method should add a parameter group to the sharded optimizer.
        This is called during construction of the sharded optimizer by
        the super-class constructor and may also be called during training
        (e.g., for gradually unfreezing layers in a model).
        As a result, this method should handle assigning the model's
        parameters among the ranks.
        '''
        num_params = self.num_params
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        rank_pg = {k: v for k, v in param_group.items() if k != 'params'}
        rank_pg['params'] = []
        full_param_group = {k: v for k, v in param_group.items()}
        full_param_group['ranks'] = []
        for param in param_group['params']:
            prank = num_params % world_size
            full_param_group['ranks'].append(prank)
            if prank == rank:
                rank_pg['params'].append(param)
            num_params += 1

        self.all_param_groups.append(full_param_group)
        if len(rank_pg['params']) != 0:
            self.rank_param_groups.append(rank_pg)

        super().add_param_group(param_group)
        self.num_params = num_params
        return


def ddp_train_worker(
    rank,
    world_size,
    seed: int,
    num_steps: int,
    batch_size: int,
    model_dims: dict,
    opt_params: dict,
    save_model_path: str,
    device: str | torch.device,
):
    setup(rank, world_size)

    torch.manual_seed(seed)
    np.random.seed(seed)

    raw_model = TransformerLM(**model_dims, device=device)
    model = DDP(raw_model)

    # 1. Initialization
    optimizer = AdamW(model.parameters(), **opt_params)
    
    seq_len = model_dims['context_length']

    vocab_size = model_dims['vocab_size']

    # 3. The "Iterative Logic" is inside the worker
    flat_grad_holder = None
    for step in range(num_steps):
        # Step 1: Data Sharding
        # You must ensure each rank gets a DIFFERENT, non-overlapping slice 
        # of the current batch (n/d examples).
        global_batch = get_global_batch(batch_size, seq_len, vocab_size, device)
        x, y = get_sharded_batch(global_batch, rank, world_size)

        # Step 2: Forward & Backward pass

        logits = model(x)
        ce = cross_entropy(logits, y)

        ce.backward()
        model.finish_gradient_synchronization()

        # Step 4: Optimizer update
        # Since all ranks have identical weights and identical averaged gradients,
        # this step will deterministically produce the exact same updated weights on all ranks.
        optimizer.step()
        optimizer.zero_grad()
    
    if rank == 0:
        save_model(model.module, model_dims, save_model_path)


class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        '''
        Given an instantiated PyTorch nn.Module to be parallelized, 
        construct a DDP container that will handle gradient synchronization
        across ranks.
        '''
        super().__init__()
        self.module = module
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.sync_grads)
        self.handles = []

    def sync_grads(self, param: torch.Tensor):
        handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG, async_op=True)
        self.handles.append(handle)
        return

    def forward(self, *inputs, **kwargs):
        '''
        Calls the wrapped module’s forward() method with the
        provided positional and keyword arguments.
        '''
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        '''
        When called, wait for asynchronous communication
        '''
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        return


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def get_global_batch(batch_size, seq_len, vocab_size, device):
    return (
        torch.randint(0, vocab_size, size=(batch_size, seq_len), device=device),
        torch.randint(0, vocab_size, size=(batch_size, seq_len), device=device)
    )

def get_sharded_batch(global_batch, rank, world_size):
    x, y = global_batch
    batch_size = x.shape[0]
    assert batch_size % world_size == 0, f"batch_size {batch_size} is not divisible by world_size {world_size}"
    shard_size = batch_size // world_size
    return (
        x[rank*shard_size:(rank+1)*shard_size, ...],
        y[rank*shard_size:(rank+1)*shard_size, ...],
    )

def single_proc_train(
    seed: int,
    num_steps: int,
    batch_size: int,
    model_dims: dict,
    opt_params: dict,
    save_model_path: str,
    device: str | torch.device,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Initialization
    model = TransformerLM(**model_dims, device=device)
    optimizer = AdamW(model.parameters(), **opt_params)
    
    seq_len = model_dims['context_length']
    vocab_size = model_dims['vocab_size']

    # 2. Broadcast initial parameters (Step 0)
    # This ensures all ranks start with the exact same weights

    # 3. The "Iterative Logic" is inside the worker
    for step in range(num_steps):
        # Step 1: Data Sharding
        # You must ensure each rank gets a DIFFERENT, non-overlapping slice 
        # of the current batch (n/d examples).
        x, y = get_global_batch(batch_size, seq_len, vocab_size, device)

        # Step 2: Forward & Backward pass
        logits = model(x)
        ce = cross_entropy(logits, y)

        ce.backward()

        # Step 4: Optimizer update
        # Since all ranks have identical weights and identical averaged gradients,
        # this step will deterministically produce the exact same updated weights on all ranks.
        optimizer.step()
        optimizer.zero_grad()
    
    save_model(model, model_dims, save_model_path)
    return

def naive_ddp_train_worker(
    rank,
    world_size,
    seed: int,
    num_steps: int,
    batch_size: int,
    model_dims: dict,
    opt_params: dict,
    save_model_path: str,
    device: str | torch.device,
    flat_gradients: bool = False,
):
    setup(rank, world_size)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Initialization
    model = TransformerLM(**model_dims, device=device)
    optimizer = AdamW(model.parameters(), **opt_params)
    
    seq_len = model_dims['context_length']

    vocab_size = model_dims['vocab_size']

    # 2. Broadcast initial parameters (Step 0)
    # This ensures all ranks start with the exact same weights
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # 3. The "Iterative Logic" is inside the worker
    flat_grad_holder = None
    for step in range(num_steps):
        # Step 1: Data Sharding
        # You must ensure each rank gets a DIFFERENT, non-overlapping slice 
        # of the current batch (n/d examples).
        global_batch = get_global_batch(batch_size, seq_len, vocab_size, device)
        x, y = get_sharded_batch(global_batch, rank, world_size)

        # Step 2: Forward & Backward pass

        logits = model(x)
        ce = cross_entropy(logits, y)

        ce.backward()

        # Step 3: All-Reduce Gradients
        # Manually iterate over parameters and sum their gradients across ranks
        dtype = None
        total_len = 0
        if flat_gradients and flat_grad_holder is None:
            for param in model.parameters():
                if param.grad is not None:
                    if dtype is None:
                        dtype = param.grad.data.dtype
                    total_len += param.grad.data.numel()
            flat_grad_holder = torch.empty(size=(total_len,), dtype=dtype, device=device)
        
        if not flat_gradients:
            for param in model.parameters():
                if param.grad is not None:
                    # all_reduce sums the gradients in-place by default
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
        else:
            i = 0
            for param in model.parameters():
                if param.grad is not None:
                    numels = param.grad.data.numel()
                    flat_grad_holder[i:i+numels] = param.grad.data.view(-1)
                    i += numels
            dist.all_reduce(flat_grad_holder, op=dist.ReduceOp.AVG)

            i = 0
            for param in model.parameters():
                if param.grad is not None:
                    numels = param.grad.data.numel()
                    param.grad.data.copy_(flat_grad_holder[i:i+numels].view_as(param.grad.data))
                    i += numels

        # Step 4: Optimizer update
        # Since all ranks have identical weights and identical averaged gradients,
        # this step will deterministically produce the exact same updated weights on all ranks.
        optimizer.step()
        optimizer.zero_grad()
    
    if rank == 0:
        save_model(model, model_dims, save_model_path)


def read_model_setup(mdims) -> dict:
    keys = ('d_model', 'd_ff',  'num_layers', 'num_heads', 'context_length', 'vocab_size')
    model_dims = dict(zip(keys, mdims))
    return model_dims

def verify_models_match(baseline_path: str, ddp_path: str, ddp_flat_path: str, tol: float = 1e-4):
    """
    Loads saved model states and compares their parameters.
    Assumes save_model() saves a dict containing the model's state_dict, 
    or the state_dict directly.
    """
    print(f"Loading models for verification...")
    
    # Load checkpoints
    base_ckpt = torch.load(baseline_path, map_location='cpu', weights_only=False)
    ddp_ckpt = torch.load(ddp_path, map_location='cpu', weights_only=False)
    flat_ckpt = torch.load(ddp_flat_path, map_location='cpu', weights_only=False)
    
    # Extract state_dicts (adjust if your save_model nests the state_dict under a key like 'model')
    sd_base = base_ckpt.get('model', base_ckpt) if isinstance(base_ckpt, dict) else base_ckpt
    sd_ddp = ddp_ckpt.get('model', ddp_ckpt) if isinstance(ddp_ckpt, dict) else ddp_ckpt
    sd_flat = flat_ckpt.get('model', flat_ckpt) if isinstance(flat_ckpt, dict) else flat_ckpt

    def compare_dicts(sd1, sd2, name1, name2):
        for key in sd1.keys():
            if key not in sd2:
                print(f"Key {key} found in {name1} but not in {name2}")
                return False
            
            tensor1, tensor2 = sd1[key], sd2[key]
            
            # Check if they are exactly identical (unlikely due to FP math)
            if torch.equal(tensor1, tensor2):
                continue
                
            # Check if they are mathematically close
            if not torch.allclose(tensor1, tensor2, atol=tol, rtol=tol):
                max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
                print(f"Mismatch in {key} between {name1} and {name2}. Max diff: {max_diff}")
                return False
        return True

    print("Comparing Single Process vs Naive DDP...")
    match_1 = compare_dicts(sd_base, sd_ddp, "Single Process", "Naive DDP")
    
    print("Comparing Single Process vs Flat DDP...")
    match_2 = compare_dicts(sd_base, sd_flat, "Single Process", "Flat DDP")

    match_3 = compare_dicts(sd_ddp, sd_flat, "Naive DDP", "Flat DDP")
    
    if match_1 and match_2:
        print("SUCCESS: All models are mathematically identical!")
    else:
        print("FAILED: Model weights diverged.")


def test_naive_ddp(seed: int = 0, num_steps: int = 10, on_cluster: bool = False):
    device_1 = 'cpu' if not on_cluster else 'cuda'
    device_2 = 'cpu' if not on_cluster else 'cuda'

    mdims = (32, 32, 2, 4, 32, 10000) 
    if on_cluster:
        mdims = (2560, 10240, 32, 32, 10000)
    model_dims = read_model_setup(mdims)
    
    batch_size = 8
    world_size = 4

    save_model_path_1 = '../data/results/dp/model_test_single.pkl'
    save_model_path_2 = '../data/results/dp/model_test_naive_ddp.pkl'
    save_model_path_3 = '../data/results/dp/model_test_naive_ddp_flat.pkl'
    save_model_path_4 = '../data/results/dp/model_test_ddp.pkl'

    opt_params = {}
    print('num_steps', num_steps)
    
    single_proc_train(seed, num_steps, batch_size, model_dims, opt_params, save_model_path_1, device_1)
    
    mp.spawn(fn=naive_ddp_train_worker,
             args=(world_size, seed, num_steps, batch_size, model_dims, opt_params, save_model_path_2, device_2, False),
             nprocs=world_size, join=True)

    mp.spawn(fn=naive_ddp_train_worker,
             args=(world_size, seed, num_steps, batch_size, model_dims, opt_params, save_model_path_3, device_2, True),
             nprocs=world_size, join=True)

    mp.spawn(fn=ddp_train_worker,
             args=(world_size, seed, num_steps, batch_size, model_dims, opt_params, save_model_path_4, device_2),
             nprocs=world_size, join=True)
    
    # Add this to the end of your test_naive_ddp() function:
    verify_models_match(save_model_path_1, save_model_path_2, save_model_path_3)
    verify_models_match(save_model_path_1, save_model_path_3, save_model_path_4)
    return

if __name__ == '__main__':
    import fire
    fire.Fire(test_naive_ddp)
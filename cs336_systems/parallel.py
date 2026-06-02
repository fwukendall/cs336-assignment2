import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import datetime
from cs336_bmine.langmodel import TransformerLM
from cs336_bmine.train_util import save_model, cross_entropy, AdamW
import numpy as np
from typing import Type, Any, Optional, Callable


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
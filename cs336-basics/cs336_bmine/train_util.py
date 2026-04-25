from collections.abc import Callable, Iterable
from typing import Optional
import math
import torch
import os
from torch import nn, sigmoid
from einops import rearrange, einsum
from jaxtyping import Int, Float, Bool
from typing import IO, BinaryIO
import numpy.typing as npt
import numpy as np


def save_model(
    model: torch.nn.Module,
    config: dict,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    msd = model.state_dict()
    torch.save(
        {
            'model': msd,
            'config': config,
        },
        out,
    )
    return


def load_model_for_inference(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    constructor: type,
    device: str | torch.device = 'cuda',
) -> torch.nn.Module:
    saved = torch.load(src, weights_only=False)
    config = saved['config']
    if 'model_dimension' in config:
        config = config['model_dimension']
    model = constructor(device=device, **config)
    model.load_state_dict(saved['model'])
    return model


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    config: dict,
) -> None:
    msd = model.state_dict()
    osd = optimizer.state_dict()
    torch.save(
        {
            'model': msd,
            'optimizer': osd,
            'iteration': iteration,
            'config': config,
        },
        out,
    )
    return


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    checkpt = torch.load(src, weights_only=False)
    model.load_state_dict(checkpt['model'])
    if optimizer:
        optimizer.load_state_dict(checkpt['optimizer'])
    return checkpt['iteration']


def get_batch(
    tokens: npt.NDArray,
    batch_size: int,
    seq_len: int,
    device: str,
    seed: int|None = None,
) -> tuple[Int[torch.Tensor, 'batch_size seq_len'], Int[torch.Tensor, 'batch_size seq_len']]:
    if seed is not None:
        torch.manual_seed(seed)
    start_idx = torch.randint(low=0, high=tokens.shape[0]-seq_len, size=(batch_size, 1))
    idx1 = start_idx + torch.arange(0, seq_len)
    idx2 = start_idx + torch.arange(1, seq_len+1)

    t1 = torch.from_numpy(tokens[idx1.numpy()].astype(np.int64)).to(device=device)
    t2 = torch.from_numpy(tokens[idx2.numpy()].astype(np.int64)).to(device=device)
    return (t1, t2)


def clip_gradients(params: Iterable[nn.Parameter], max_norm: float, eps=1e-6):
    grad_l2_sum = 0.0
    for p in params:
        if p.grad is None:
            continue
        g2 = (p.grad.data ** 2).flatten()
        grad_l2_sum += float(g2.sum())
    l2_norm = np.sqrt(grad_l2_sum)
    if l2_norm <= max_norm:
        return
    clip_factor = max_norm / (l2_norm + eps)
    for p in params:
        if p.grad is None:
            continue
        p.grad.data *= clip_factor
    return


def get_cosine_learning_rate_sched(
    t: int,
    a_max: float,
    a_min: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:

    if t < warmup_iters:
        return t * a_max / warmup_iters
    elif t > cosine_cycle_iters:
        return a_min
    
    angle = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters) * np.pi
    return float(a_min + 0.5 * (1 + np.cos(angle)) * (a_max - a_min))

dims_GPT2XL = {
    'context_length': 1_024,
    'num_layers': 48,
    'vocab_size': 50_257,
    'd_model': 1_600,
    'num_heads': 25,
}

def adamw_accounting(
    batch_size,
    vocab_size,
    context_length,
    num_layers,
    d_model,
    num_heads,
    data_bytes=4,
):
    # d_ff = 4 * d_model
    # num_params
    p_embed = d_model * vocab_size
    
    p_per_layer_norm = d_model * 2
    p_per_layer_qkv = 3 * d_model ** 2
    p_per_layer_o = d_model ** 2
    p_per_layer_ffn = 2 * 4 * d_model ** 2
    p_per_layer = p_per_layer_ffn + p_per_layer_norm + p_per_layer_qkv + p_per_layer_o
    p_attn = p_per_layer * num_layers

    p_final_norm = d_model
    p_output = d_model * vocab_size
    
    p_lmparams = p_embed + p_attn + p_final_norm + p_output
    p_adamstate = 2 * p_lmparams

    g_total = p_lmparams

    # activations per batch_size
    a_attn_norms = context_length * d_model * num_layers * 2
    a_attn_qkvproj = context_length * d_model * 3 * num_layers
    a_attn_qtk = context_length ** 2 * num_layers * num_heads
    a_attn_softmax = context_length ** 2 * num_layers * num_heads
    a_attn_vals = context_length * d_model * num_layers
    a_attn_outs = context_length * d_model * num_layers

    a_ffn_w1 = 4 * d_model * num_layers * context_length
    a_ffn_sig = 4 * d_model * num_layers * context_length
    a_ffn_w3 = 4 * d_model * num_layers * context_length
    a_ffn_out = d_model * num_layers * context_length

    a_embed = d_model * context_length
    a_final_norm = context_length * d_model

    a_out_logits = vocab_size * context_length

    param_total = p_lmparams * data_bytes
    opt_total = p_adamstate * data_bytes
    grad_total = g_total * data_bytes
    act_perbs = a_attn_norms + a_attn_qkvproj + a_attn_qtk + a_attn_softmax + \
                a_attn_vals + a_attn_outs + a_ffn_w1 + a_ffn_sig + \
                a_ffn_out + a_embed + a_final_norm + a_out_logits
    act_total = act_perbs * batch_size * data_bytes
    return {
        'mem': {
            'param': param_total,
            'opt': opt_total,
            'grad': grad_total,
            'act': act_total
        },
    }


def cross_entropy(
    logits: Float[torch.Tensor, ' ... vocab_size'],
    targets: Int[torch.Tensor, ' ...'],
) -> Float[torch.Tensor, '']:
    logit_max, _ = logits.max(dim=-1, keepdim=True)
    logits = logits - logit_max
    logits_2d = logits.reshape(-1, logits.shape[-1])
    targets_1d = targets.reshape(-1)
    out_logits = logits_2d[torch.arange(targets_1d.shape[0], device=logits.device), targets_1d]
    denom_sum = torch.log(torch.exp(logits_2d).sum(dim=-1))
    ce_loss = (-out_logits + denom_sum).flatten().mean()
    return ce_loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter], 
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        for k, v in defaults.items():
            if hasattr(v, '__iter__'):
                for i, subv in enumerate(v):
                    if subv < 0:
                        raise ValueError(f"Invalid {k}_{i}: {subv}")
            else:
                if v < 0:
                    raise ValueError(f"Invalid {k}: {v}")
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] 
            wd = group["weight_decay"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, for adam it starts at 1
                m = state.get("m", None)
                v = state.get("v", None)
                if m is None:
                    m = torch.zeros_like(p.data)
                    v = torch.zeros_like(p.data)

                grad = p.grad.data # Get the gradient of loss with respect to p.

                # m = b1 * m + (1 - b1) * grad
                # v = b2 * v + (1 - b2) * grad**2

                # perform in-place updates to m and v
                m.mul_(b1).add_(grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                lr_t = lr * np.sqrt(1 - b2**t) / (1 - b1**t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps) # Update weight tensor in-place.
                p.data -= lr * wd * p.data
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss


class SGD(torch.optim.Optimizer):
    '''
    Example SGD implementation provided by the assignment
    '''
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] 
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
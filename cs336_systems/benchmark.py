import cs336_basics
import cs336_bmine
import torch
from typing import Literal, Callable
import timeit
from functools import partial
import numpy as np
import pandas as pd
import fire
import torch.cuda.nvtx as nvtx

def load_model(model_dims: dict, device='cuda') -> torch.nn.Module:
    model = cs336_bmine.langmodel.TransformerLM(**model_dims, device=device)
    return model


def load_optimizer(
    opt_params: dict,
    model: torch.nn.Module,
    device='cuda',
) -> torch.optim.Optimizer:
    opt = cs336_bmine.train_util.AdamW(model.parameters(), **opt_params)
    return opt

@torch.inference_mode(True)
def benchmark_fw(
    model: torch.nn.Module,
    get_batch: Callable,
    warmup: int,
    n_steps: int,
) -> list[float]:
    device = model.device
    model.eval()
    for _ in range(warmup):
        x, _ = get_batch()
        _ = model(x)

    times = []
    for _ in range(n_steps):
        x, _ = get_batch()
        with nvtx.range('bmmode_fw'):
            torch.cuda.synchronize(device)
            start_t = timeit.default_timer()
            _ = model(x)
            torch.cuda.synchronize(device)
            end_t = timeit.default_timer()
        times.append(end_t - start_t)
    return times

def benchmark_full(
    model: torch.nn.Module,
    get_batch: Callable,
    warmup: int,
    n_steps: int,
    opt: torch.optim.Optimizer | None = None,
):
    device = model.device
    model.train()
    for _ in range(warmup):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        x, y = get_batch()
        logits = model(x)
        ce = cs336_bmine.train_util.cross_entropy(logits, y)
        ce.backward()
        if opt is not None:
            opt.step()
    
    times = []
    for _ in range(n_steps):
        curtime = 0.
        x, y = get_batch()
        if opt is not None:
            opt.zero_grad(set_to_none=True)

        with nvtx.range('bmmode_fw_withgrad'):
            torch.cuda.synchronize(device)
            start_t = timeit.default_timer()
            logits = model(x)

            torch.cuda.synchronize(device)
            pt = timeit.default_timer()
        curtime += pt - start_t

        ce = cs336_bmine.train_util.cross_entropy(logits, y)
        with nvtx.range('bmmode_bw'):
            torch.cuda.synchronize(device)
            start_t = timeit.default_timer()
            ce.backward()
            torch.cuda.synchronize(device)
            pt = timeit.default_timer()
        curtime += pt - start_t

        if opt is not None:
            with nvtx.range('bmmode_opt'):
                torch.cuda.synchronize(device)
                start_t = timeit.default_timer()
                opt.step()
                torch.cuda.synchronize(device)
                pt = timeit.default_timer()
            curtime += pt - start_t
        times.append(curtime)
    return times


def run_benchmark(
    warmup: int = 5,
    n_steps: int = 10,
    batch_size: int = 4,
    mode: Literal['fw', 'fw-bw', 'fw-bw-opt'] = 'fw',
    model_dims: dict = {},
    opt_params: dict | None = None,
    device: torch.device | None = None,
) -> list[float]:

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mode == 'fw-bw-opt' and opt_params is None:
        opt_params = {}
    
    model = load_model(model_dims, device=device)
    opt = None
    if mode == 'fw-bw-opt':
        opt = load_optimizer(opt_params, model, device=device)
    
    seq_len = 512
    if 'context_length' in model_dims:
        seq_len = model_dims['context_length']

    vocab_size = model_dims['vocab_size']

    get_batch = lambda: (
        torch.randint(0, vocab_size, size=(batch_size, seq_len), device=device),
        torch.randint(0, vocab_size, size=(batch_size, seq_len), device=device)
    )

    if mode == 'fw':
        times = benchmark_fw(model, get_batch, warmup, n_steps)
    else:
        times = benchmark_full(
            model, get_batch, warmup, n_steps, opt,
        )
    return times


def run_preset(
    warmup: int = 5,
    n_steps: int = 10,
    on_laptop: bool = True,
    context_length: int = 512,
    vocab_size: int = 10000,
    batch_size: int = 4,
):
    setups = {
        # name: d_model d_ff num_layers num_heads
        'small': (768, 3072, 12, 12),
        'medium': (1024, 4096, 24, 16),
        'large': (1280, 5120, 36, 20),
        'xl': (2560, 10240, 32, 32),
        # '10B': (4608, 12288, 50, 36),
    }

    keys = ('d_model', 'd_ff',  'num_layers', 'num_heads')
    model_dims_dict = {}
    for name, mdims in setups.items():
        if on_laptop and name in ('10B', 'xl', 'large', 'medium'):
            print(f"Won't be able to run {name} on your measly 3050")
            continue
        model_dims = dict(zip(keys, mdims))
        model_dims['context_length'] = context_length
        model_dims['vocab_size'] = vocab_size
        model_dims_dict[name] = model_dims
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stats_all = {}
    # no need to actually pass in
    for mode in ['fw', 'fw-bw', 'fw-bw-opt']:
        stats_all[mode] = {}
        for name, model_dims in model_dims_dict.items():
            print(f'benchmarking {mode} on {name}')
            times = run_benchmark(
                warmup=warmup,
                n_steps=n_steps,
                batch_size=batch_size,
                mode=mode,
                model_dims=model_dims,
                device=device,
            )
            m, s = np.mean(times), np.std(times)
            stats_all[mode][name+'_mean'] = m
            stats_all[mode][name+'_std'] = s
    out_df = pd.DataFrame(stats_all)
    print(out_df)
    return

if __name__ == '__main__':
    fire.Fire()

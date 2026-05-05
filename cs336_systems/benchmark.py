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
from contextlib import nullcontext
import pickle

MODEL_SETUPS = {
    # name: d_model d_ff num_layers num_heads
    'small': (768, 3072, 12, 12),
    'medium': (1024, 4096, 24, 16),
    'large': (1280, 5120, 36, 20),
    'xl': (2560, 10240, 32, 32),
    # '10B': (4608, 12288, 50, 36),
}

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
    cast_bf16: bool = False,
    mark_bmmode: bool = True,
    record_mem: bool = False,
    run_name: str = '',
) -> list[float]:

    device = model.device
    model.eval()
    context = nullcontext()
    if cast_bf16:
        context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    for _ in range(warmup):
        x, _ = get_batch()
        with context:
            _ = model(x)

    times = []
    torch.cuda.cudart().cudaProfilerStart()

    if record_mem:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    for _ in range(n_steps):
        x, _ = get_batch()
        with (nullcontext() if not mark_bmmode else nvtx.range('bmmode_fw')):
            torch.cuda.synchronize(device)
            start_t = timeit.default_timer()
            with context:
                _ = model(x)
            torch.cuda.synchronize(device)
            end_t = timeit.default_timer()
        times.append(end_t - start_t)

    if record_mem:
        # Save a pickle file to be loaded by PyTorch's online tool.
        run_prefix = 'fw_' + (f'{run_name}_' if run_name != '' else '')
        torch.cuda.memory._dump_snapshot(f"cs336_systems/{run_prefix}mem_snap.pickle")
        # Stop recording history.
        torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.cudart().cudaProfilerStop()
    return times

def benchmark_full(
    model: torch.nn.Module,
    get_batch: Callable,
    warmup: int,
    n_steps: int,
    opt: torch.optim.Optimizer | None = None,
    cast_bf16: bool = False,
    record_mem: bool = False,
    run_name: str = '',
):
    device = model.device
    context = nullcontext()
    if cast_bf16:
        context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    model.train()
    for _ in range(warmup):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        x, y = get_batch()
        with context:
            logits = model(x)
            ce = cs336_bmine.train_util.cross_entropy(logits, y)
        ce.backward()
        if opt is not None:
            opt.step()
    
    times = []
    if opt is not None:
        run_prefix = 'opt_' + (f'{run_name}_' if run_name != '' else '')
    else:
        run_prefix = 'bw_' + (f'{run_name}_' if run_name != '' else '')
    if record_mem:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    for _ in range(n_steps):
        curtime = 0.
        x, y = get_batch()
        if opt is not None:
            opt.zero_grad(set_to_none=True)

        with nvtx.range('bmmode_fw_withgrad'):
            torch.cuda.synchronize(device)
            start_t = timeit.default_timer()
            with context:
                logits = model(x)
            torch.cuda.synchronize(device)
            pt = timeit.default_timer()
        curtime += pt - start_t

        with context:
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
    if record_mem:
        # Save a pickle file to be loaded by PyTorch's online tool.
        torch.cuda.memory._dump_snapshot(f"cs336_systems/{run_prefix}mem_snap.pickle")
        # Stop recording history.
        torch.cuda.memory._record_memory_history(enabled=None)
    return times


def run_benchmark(
    warmup: int = 5,
    n_steps: int = 10,
    batch_size: int = 4,
    mode: Literal['fw', 'fw-bw', 'fw-bw-opt'] = 'fw',
    model_dims: dict | str = 'small',
    opt_params: dict | None = None,
    device: torch.device | None = None,
    cast_bf16: bool = False,
    record_mem: bool = False,
    run_name: str = '',
) -> list[float]:
    if isinstance(model_dims, str):
        model_dims = read_model_setup(MODEL_SETUPS[model_dims])

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mode == 'fw-bw-opt' and opt_params is None:
        opt_params = {}
    
    model = torch.compile(load_model(model_dims, device=device), fullgraph=True)
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
        times = benchmark_fw(model, get_batch, warmup, n_steps, cast_bf16,
                             record_mem=record_mem, run_name=run_name)
    else:
        times = benchmark_full(
            model, get_batch, warmup, n_steps, opt, cast_bf16,
            record_mem=record_mem, run_name=run_name
        )
    return times

def read_model_setup(mdims) -> dict:
    keys = ('d_model', 'd_ff',  'num_layers', 'num_heads')
    model_dims = dict(zip(keys, mdims))
    model_dims['context_length'] = 512
    model_dims['vocab_size'] = 10000
    return model_dims

def run_preset(
    warmup: int = 5,
    n_steps: int = 10,
    on_laptop: bool = True,
    context_length: int = 512,
    vocab_size: int = 10000,
    batch_size: int = 4,
    cast_bf16: bool = False,
    record_mem: bool = False,
):
    setups = MODEL_SETUPS
    model_dims_dict = {}
    for name, mdims in setups.items():
        if on_laptop and name in ('10B', 'xl', 'large', 'medium'):
            print(f"Won't be able to run {name} on your measly 3050")
            continue
        model_dims = read_model_setup(mdims)
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
            run_name = f'{"fp32" if not cast_bf16 else "bf16"}_CL{context_length}_{name}'
            times = run_benchmark(
                warmup=warmup,
                n_steps=n_steps,
                batch_size=batch_size,
                mode=mode,
                model_dims=model_dims,
                device=device,
                cast_bf16=cast_bf16,
                record_mem=record_mem,
                run_name=run_name,
            )
            m, s = np.mean(times), np.std(times)
            stats_all[mode][name+'_mean'] = m
            stats_all[mode][name+'_std'] = s
    out_df = pd.DataFrame(stats_all)
    print(out_df)
    return


def run_attention(
    d_k: int = 16,
    seq_len: int = 256,
    warmup: int = 20,
    n_steps: int = 100,
    bm_mode: Literal['fw', 'bw'] = 'fw',
    cast_bf16: bool = True,
    mem_filename: str | None = None,
    attn_impl: Literal['bmine', 'basic', 'torch', 'flash'] = 'bmine',
    do_compile: bool = False,
):
    batch_size = 8
    device = 'cuda'
    get_batch = lambda: (
        torch.rand((batch_size, seq_len, d_k,), device=device, requires_grad=True),
        torch.rand((batch_size, seq_len, d_k,), device=device, requires_grad=True),
        torch.rand((batch_size, seq_len, d_k,), device=device, requires_grad=True),
        torch.rand((batch_size, seq_len, d_k,), device=device,),
    )
    mask = (torch.tril(torch.ones(seq_len, seq_len)) == 1).to(device)
    mask = mask.view(1, seq_len, seq_len)
    if attn_impl == 'bmine':
        attn_func = cs336_bmine.langmodel.scaled_dot_product_attention
    elif attn_impl == 'basic':
        from cs336_basics.model import scaled_dot_product_attention as basic_attn
        attn_func = basic_attn
    elif attn_impl == 'torch':
        from torch.nn.functional import scaled_dot_product_attention as torch_attn
        attn_func = torch_attn
    else:
        raise NotImplementedError(attn_impl)
    
    if do_compile:
        attn_func = torch.compile(attn_func)

    # warmup
    context = nullcontext()
    if cast_bf16:
        context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)

    for _ in range(warmup):
        Q, K, V, Y = get_batch()
        with context:
            X = attn_func(Q, K, V, mask)
            if bm_mode == 'bw':
                mse = torch.nn.functional.mse_loss(X, Y)
        if bm_mode == 'bw':
            mse.backward()
        

    times = []
    if bm_mode == 'fw':
        if mem_filename is not None:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        for _ in range(n_steps):
            Q, K, V, _ = get_batch()
            torch.cuda.synchronize(device)
            start_t = timeit.default_timer()
            with context:
                _ = attn_func(Q, K, V, mask)
            torch.cuda.synchronize(device)
            end_t = timeit.default_timer()
            times.append(end_t - start_t)

        if mem_filename is not None:
            torch.cuda.memory._dump_snapshot(f"cs336_systems/{mem_filename}_mem_snap.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
    
    else:
        for _ in range(n_steps):
            Q, K, V, Y = get_batch()
            with context:
                X = attn_func(Q, K, V, mask)
                mse = torch.nn.functional.mse_loss(X, Y)
            torch.cuda.synchronize(device)
            start_t = timeit.default_timer()
            mse.backward()
            torch.cuda.synchronize(device)
            end_t = timeit.default_timer()
            times.append(end_t - start_t)

    return times

def run_attn_preset(
    warmup: int = 20,
    n_steps: int = 100,
    cast_bf16: bool = True,
    mem_prefix: str | None = None,
    attn_impl: Literal['bmine', 'basic', 'torch', 'flash'] = 'bmine',
    do_compile: bool = False,
    out_prefix: str | None = None,
):
    if out_prefix is None:
        out_prefix = 'laptop'
    if mem_prefix is None:
        mem_prefix = 'laptop'

    d_k_list = [16, 32, 64, 128]
    seq_len_list = [256, 1024, 4096, 8192, 16384]
    # d_k_list = [16, 32] # , 64, 128]
    # seq_len_list = [256, 1024] # , 4096, 8192, 16384]
    mem_err_list = []
    type_kw = 'bf16' if cast_bf16 else 'fp32'
    comp_kw = 'docomp' if do_compile else 'nocomp'
    base_name = f'afunc-{attn_impl}_{comp_kw}_{type_kw}'
    times_dict = {
        'fw': {},
        'bw': {},
    }
    OOM_start = {'fw': [], 'bw': []}
    for d_k in d_k_list:
        for seq_len in seq_len_list:
            run_name = f'dk{d_k}_cl{seq_len}_{base_name}'
            for bm_mode in ['fw', 'bw']:
                OOM_list = OOM_start[bm_mode]
                should_run = True
                for oom_dk, oom_seqlen in OOM_list:
                    if d_k >= oom_dk and seq_len >= oom_seqlen:
                        should_run = False
                        break
                if not should_run:
                    print('Skipping', bm_mode, run_name)
                    times_dict[bm_mode][run_name] = 'OOM'
                    continue
                try:
                    if bm_mode == 'fw':
                        mem_filename = f'{mem_prefix}_{run_name}'
                    else:
                        mem_filename = None
                    times = run_attention(
                        d_k=d_k,
                        seq_len=seq_len,
                        warmup=warmup,
                        n_steps=n_steps,
                        bm_mode=bm_mode,
                        cast_bf16=cast_bf16,
                        mem_filename=mem_filename,
                        attn_impl=attn_impl,
                        do_compile=do_compile,
                    )
                except torch.cuda.OutOfMemoryError:
                    OOM_start[bm_mode].append((d_k, seq_len))
                    print(f'OOM on {bm_mode} {run_name}') 
                    mem_err_list.append(f'{bm_mode}_{run_name}')
                    times_dict[bm_mode][run_name] = 'OOM'
                    continue
                times_dict[bm_mode][run_name] = times
    out_filename = f'{out_prefix}_{base_name}_times.pkl' 
    with open(out_filename, 'wb') as out:
        pickle.dump(times_dict, out)
        print('Written', out_filename)
    print(base_name)
    summ_dict = {f'{base_name}_fw': {}, f'{base_name}_bw': {}}
    for bm_mode, bm_times in times_dict.items():
        for run_name, times in bm_times.items():
            run_abbr = run_name.split(f'_{base_name}')[0]
            if times == 'OOM':
                summ_dict[f'{base_name}_{bm_mode}'][run_abbr] = np.nan
            else:
                summ_dict[f'{base_name}_{bm_mode}'][run_abbr] = np.mean(times)
    summ_df = pd.DataFrame(summ_dict)
    print(summ_df)
    out_filename = f'{out_prefix}_{base_name}_summ.csv' 
    summ_df.to_csv(out_filename, index=None)
    print('Written', out_filename)
    return

if __name__ == '__main__':
    fire.Fire()


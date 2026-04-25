from collections.abc import Callable, Iterable
import yaml
from typing import Optional
from functools import partial
import math
import torch
import os
from torch import nn, sigmoid
from einops import rearrange, einsum
from jaxtyping import Int, Float, Bool
from typing import IO, BinaryIO
import numpy.typing as npt
import numpy as np

from train_util import AdamW, get_cosine_learning_rate_sched, save_model, \
clip_gradients, get_batch, cross_entropy, save_checkpoint, load_checkpoint
import fire
from langmodel import TransformerLM
from torch.utils.tensorboard import SummaryWriter
import time


def flatten_config(d, parent_key='', sep='/'):
    """Flattens a nested dictionary for TensorBoard logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # TensorBoard doesn't accept lists, so cast beta values to strings
            items.append((new_key, str(v))) 
        else:
            items.append((new_key, v))
    return dict(items)


@torch.inference_mode(True)
def eval_loss(
    model: torch.nn.Module,
    get_eval_batch: Callable,
    eval_iters: int,
) -> float:
    ce = 0
    model.eval()
    for _ in range(eval_iters):
        x, y = get_eval_batch()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            ce += cross_entropy(logits, y).item() / eval_iters
    model.train()
    return ce


def run_train(
    train_data_path: str,
    valid_data_path: str,
    out_save_path: str,
    checkpt_path: str,
    hyperparams: dict,
    single_batch=False,
    save_steps=50,
    eval_iters=20,
    log_steps=5,
    load_existing=False,
):
    board_name = hyperparams['name']
    writer = SummaryWriter(log_dir=f'runs/{board_name}')

    flat_hparams = flatten_config(hyperparams)
    has_added_hparams = False

    token_mmap = np.memmap(train_data_path, dtype=np.uint16)
    eval_mmap = np.memmap(valid_data_path, dtype=np.uint16)
    model_dims = hyperparams['model_dimension']
    adam_params = hyperparams['train_params']['adam_params']
    lr_sched_params = hyperparams['train_params']['lr_schedule']
    device = torch.device('cuda')

    model_dims['dtype'] = torch.float32

    lm = TransformerLM(device=device, **model_dims)
    opt = AdamW(lm.parameters(), lr=lr_sched_params['a_min'], **adam_params)

    batch_size = hyperparams['train_params']['batch_size']
    grad_accum_steps = hyperparams['train_params']['grad_accum_steps']
    max_steps = hyperparams['train_params']['max_steps']
    grad_max = hyperparams['train_params']['grad_max']

    sched = partial(get_cosine_learning_rate_sched, **lr_sched_params,
                    cosine_cycle_iters=max_steps-lr_sched_params['warmup_iters'])

    start_epoch = 1
    if load_existing:
        start_epoch = load_checkpoint(checkpt_path, lm, opt) + 1
        has_added_hparams = True
        print(f'Loaded saved state from {checkpt_path}, iter {start_epoch}')

    get_train_batch = partial(
        get_batch, 
        tokens=token_mmap,
        batch_size=batch_size,
        seq_len=model_dims['context_length'],
        device=device
    )
    get_valid_batch = partial(
        get_batch, 
        tokens=eval_mmap,
        batch_size=batch_size,
        seq_len=model_dims['context_length'],
        device=device
    )

    grad_clip = partial(clip_gradients, params=lm.parameters(), max_norm=grad_max)
    seed = None
    if single_batch:
        seed = max_steps

    cum_train_loss_sum = 0
    last_logging_t = start_epoch - 1
    start_time = time.time()
    for t in range(start_epoch, max_steps+1):
        opt.zero_grad(set_to_none=True)

        for _ in range(grad_accum_steps):
            x, y = get_train_batch(seed=seed)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = lm(x)
                ce = cross_entropy(logits, y) / grad_accum_steps
                cum_train_loss_sum += ce.item()
            
            ce.backward() # accumulates gradient

        grad_clip()
        lr = sched(t=t)
        for group in opt.param_groups:
            group['lr'] = lr
        opt.step()
        if t % log_steps == 0:
            loss = ce.item() * grad_accum_steps
            if not has_added_hparams:
                writer.add_hparams(flat_hparams, metric_dict={
                    'Loss/train': loss,
                    'Loss/valid': loss,
                    'Time/train': 0.0,
                }, run_name='hparam_metrics')
                has_added_hparams = True
            writer.add_scalar('Loss/train', ce.item() * grad_accum_steps, t)
        
        del x, y, logits, ce

        if t % save_steps == 0:
            save_checkpoint(lm, opt, t, checkpt_path, hyperparams)
            valid_loss = eval_loss(model=lm, get_eval_batch=get_valid_batch, eval_iters=eval_iters)
            train_loss = cum_train_loss_sum / (t - last_logging_t)
            print(t, np.round(train_loss, 3), np.round(valid_loss, 3),
                  'saved to', checkpt_path)
            cur_time = time.time()
            writer.add_scalar('Loss/valid', valid_loss, t)
            writer.add_scalar('Time/train', cur_time - start_time, t)
            last_logging_t = t
            cum_train_loss_sum = 0
    writer.close()
    save_model(lm, model_dims, out_save_path)
    print('Saved model to', out_save_path)
    return

def run(
    train_data_path: str,
    run_name: str,
    valid_data_path: str | None = None,
    save_steps: int = 50,
    eval_iters: int = 20,
    load_existing: bool = False,
    **kwargs,
):
    with open('./train_configs/experiments.yaml', 'r') as cf:
        configs = yaml.safe_load(cf)
    
    if valid_data_path is None:
        valid_data_path = train_data_path
    
    hyperparams = configs[run_name]

    checkpt_path = f'../data/results/{run_name}_checkpoint.pkl'
    out_save_path = f'../data/results/{run_name}_out.pkl'
    run_train(
        train_data_path=train_data_path,
        valid_data_path=valid_data_path,
        out_save_path=out_save_path,
        checkpt_path=checkpt_path,
        hyperparams=hyperparams,
        save_steps=save_steps,
        eval_iters=eval_iters,
        load_existing=load_existing,
        **kwargs,
    )
    return

if __name__ == '__main__':
    import fire
    fire.Fire(run)

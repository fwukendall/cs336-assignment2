from typing import IO, BinaryIO
from functools import partial
import os
import torch
from torch import nn, sigmoid
from einops import rearrange, einsum
from jaxtyping import Int, Float, Bool
import numpy as np
from langmodel import TransformerLM, softmax
from bpe import BytePairEncoder
from train_util import load_model_for_inference, get_batch
from train import eval_loss
import fire


def run(
    model_src: str | os.PathLike | BinaryIO | IO[bytes],
    valid_data_path: str | None = None,
    num_sequences = 8192,
    batch_size = 16,
    seed: int | None = None,
):
    lm = load_model_for_inference(model_src, TransformerLM)
    eval_mmap = np.memmap(valid_data_path, dtype=np.uint16)
    get_valid_batch = partial(
        get_batch, 
        tokens=eval_mmap,
        batch_size=batch_size,
        seq_len=lm.context_length,
        device=lm.device,
        seed=seed,
    )
    loss = eval_loss(lm, get_valid_batch, eval_iters=num_sequences//batch_size)
    print('-'*64)
    print(model_src)
    print(f'{loss:.4f}')
    print()


if __name__ == '__main__':
    fire.Fire(run)

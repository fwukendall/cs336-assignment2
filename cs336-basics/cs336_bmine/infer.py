from typing import IO, BinaryIO
import os
import torch
from torch import nn, sigmoid
from einops import rearrange, einsum
from jaxtyping import Int, Float, Bool
import numpy as np
from langmodel import TransformerLM, softmax
from bpe import BytePairEncoder
from train_util import load_model_for_inference
import fire


def sample_new_token(
    logit_vec: Float[torch.Tensor, 'vocab_size'],
    temp: float,
    p: float,
) -> int:
    with torch.device(logit_vec.device):
        assert temp >= 0
        assert p <= 1.0
        if temp == 0:
            _, max_i = logit_vec.max(dim=0)
            return int(max_i.item())
        prob_vec = softmax(logit_vec / temp, -1)
        v_sorted, i_sorted = prob_vec.sort(dim=0, descending=True)
        irange = torch.arange(i_sorted.shape[0])
        cum_v = v_sorted.cumsum(dim=0)
        if p < 1.0:
            cutoff = irange[cum_v >= p][0]
            cum_v, i_sorted = cum_v[:cutoff+1], i_sorted[:cutoff+1]
            cum_v.div_(cum_v[-1].item())
        randfloat = torch.rand(1)[0]
        choice = i_sorted[cum_v >= randfloat][0]
    return int(choice.item())


@torch.inference_mode(True)
def decode_basic(
    lm: TransformerLM,
    encoder: BytePairEncoder,
    input_seq: str,
    temp: float = 0.,
    p: float = 1.0,
    max_out: int | None = None,
    split_text: str | None = None,
) -> str:
    num_gen = 0
    terminate = False
    token_list = encoder.encode(input_seq)
    out_tokens = []

    split_token = None
    if split_text:
        split_bytes = split_text.encode('utf-8')
        if split_bytes in encoder.rocab:
            split_token = encoder.rocab[split_bytes]
    while (max_out is None or num_gen < max_out) and not terminate:
        tokens = torch.tensor(token_list).to(lm.device)
        all_logits = lm(tokens)
        next_logits = all_logits[-1]
        next_token = sample_new_token(logit_vec=next_logits, temp=temp, p=p)
        out_tokens.append(next_token)
        token_list = (token_list + [next_token])[-lm.context_length:]
        num_gen += 1
        if split_token and next_token == split_token:
            terminate = True
        del all_logits
    out = encoder.decode(out_tokens)
    return input_seq + out

def run(
    model_src: str | os.PathLike | BinaryIO | IO[bytes],
    bpe_vocab_path: str,
    bpe_merges_path: str,
    temp: float = 0.,
    p: float = 1.0,
    max_out: int | None = None,
    seed: int | None = None,
    split_text: str | None = None,
    num_tries: int = 1,
):
    if split_text is None:
        split_text = '<|endoftext|>'
    lm = load_model_for_inference(model_src, TransformerLM)
    encoder = BytePairEncoder.from_files(
        bpe_vocab_path, bpe_merges_path, special_tokens=[split_text],
    )
    if seed:
        print(f'Setting seed to {seed}')
        np.random.seed(seed)
        torch.manual_seed(seed)
    boiler_prompts = [
        '<|endoftext|>',
        # 'Once upon a time, there',
        # 'One day, a brave little dog named',
        # 'In the middle of the forest, there',
        # 'Lily and Tom went to the park and ran into a magic'
    ]
    for _ in range(num_tries):
        prompt = str(np.random.choice(boiler_prompts))
        out = decode_basic(
            lm=lm,
            encoder=encoder,
            input_seq=prompt,
            temp=temp,
            p=p,
            max_out=max_out,
            split_text=split_text,
        )
        print(out)
    return


if __name__ == '__main__':
    fire.Fire(run)

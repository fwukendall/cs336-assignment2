import torch
from einops import rearrange, einsum
from typing import Literal, Callable
import numpy as np

def causal_mask_helper(r0, r1, c0, c1, mask) -> Literal[-1, 0, 1]:
    '''
    the block rectangle is defined by [(r0, c0), (r1, c0), (r1, c1), (r1, c1)]
    all coordinates are inclusive
    '''
    if r0 >= c1: # inclusive because diag is 1
        # entire block on lower left 
        return 1
    elif r1 < c0:
        # entire block on upper right
        return -1
    else:
        mrow = torch.arange(r0, r1+1, 1, device=mask.device).unsqueeze(-1)
        mcol = torch.arange(c0, c1+1, 1, device=mask.device).unsqueeze(0)
        mtmp = mrow.broadcast_to((mrow.shape[0], mcol.shape[1])) >= mcol
        mask[...] = mtmp
    return 0


def flash_v2_single_head_helper(ctx, Q, K, V, O, L, is_causal):
    device = Q.device
    Bq = ctx.Q_TILE_SIZE
    Bk = ctx.K_TILE_SIZE
    for t in [Q, K, V, O, L]:
        assert t.is_contiguous(), 'Need a contigous tensor!'
    Tq = Q.shape[-2] // Bq
    Tk = K.shape[-2] // Bk
    B = Q.shape[0]
    Oi = torch.empty(B, Bq, Q.shape[-1], dtype=torch.float32, device=device)
    li = torch.empty(B, Bq, device=device, dtype=torch.float32)
    mi_curr = torch.empty(B, Bq, device=device, dtype=torch.float32)
    mi_last = torch.empty(B, Bq, device=device, dtype=torch.float32)
    mask = torch.empty(Bq, Bk, device=device, dtype=bool)
    denom_norm = K.shape[-1] ** (-0.5)
    for i in range(Tq):
        Qi = Q[:, i*Bq:(i+1)*Bq, :]
        Oi.zero_()
        li.zero_()
        mi_last[...] = -float('inf')
        for j in range(Tk):
            Kj = K[:, j*Bk:(j+1)*Bk, :]
            Vj = V[:, j*Bk:(j+1)*Bk, :]
            Sij = einsum(Qi, Kj, 'b b_q d_k, b b_k d_k -> b b_q b_k') * denom_norm
            if is_causal:
                mode = causal_mask_helper(i*Bq, (i+1)*Bq-1, j*Bk, (j+1)*Bk-1, mask)
                if mode == -1:
                    # entire block on upper right, skip
                    continue
                elif mode == 0:
                    Sij.masked_fill_(~mask.broadcast_to(B, *mask.shape), -float('inf'))
                # if mode == 1 then no-op
            mi_curr = mi_last.clip(min=Sij.max(dim=-1)[0]) 
            Pij = torch.exp(Sij - mi_curr.unsqueeze(-1))
            adj_coef = torch.exp(mi_last - mi_curr)
            li.mul_(adj_coef).add_(Pij.sum(dim=-1))
            Oi.mul_(adj_coef.unsqueeze(-1)).add_(einsum(Pij, Vj, 'b b_q b_k, b b_k d_k -> b b_q d_k'))
            mi_last[...] = mi_curr
        O[:, i*Bq:(i+1)*Bq, :] = Oi.div(li.unsqueeze(-1))
        L[:, i*Bq:(i+1)*Bq] = mi_curr + torch.log(li)
    return


class FlashAttentionV2NoTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        _, seq_len, _ = Q.shape[-1], Q.shape[-2], Q.shape[:-2]
        d_k, _, _ = K.shape[-1], K.shape[-2], K.shape[:-2]
        pow2 = 2 ** int(np.floor(np.log2(16 * 1024 // (4 * d_k))))

        ctx.Q_TILE_SIZE = min(seq_len, 256, pow2)
        ctx.K_TILE_SIZE = min(seq_len, 128, pow2)

        Qf = Q.flatten(end_dim=-3)
        Kf = K.flatten(end_dim=-3)
        Vf = V.flatten(end_dim=-3)
        
        O = torch.empty(Q.shape, dtype=torch.float32, device=Q.device)
        L = torch.empty(*Q.shape[:-1], dtype=torch.float32, device=Q.device)
        Of = O.flatten(end_dim=-3)
        Lf = L.flatten(end_dim=-2)

        flash_v2_single_head_helper(ctx, Qf, Kf, Vf, Of, Lf, is_causal)

        ctx.Q = Q
        ctx.K = K
        ctx.V = V
        ctx.O = O
        ctx.L = L
        return O
    
    @staticmethod
    def backward(ctx, Q, K, V, O, dO, L):
        raise NotImplementedError

def falsh_attention_v2_no_triton(Q, K, V, is_causal=False):
    return FlashAttentionV2NoTriton.apply(Q, K, V, is_causal)

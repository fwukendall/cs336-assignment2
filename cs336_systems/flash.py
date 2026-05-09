import torch
from einops import rearrange, einsum
from typing import Literal, Callable
import numpy as np
import triton
import triton.language as tl

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
                    Sij.masked_fill_(~mask, -float('inf'))
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


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # load K directly transposed
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    if is_causal:
        nomask_till = tl.cdiv(query_tile_index * Q_TILE_SIZE, K_TILE_SIZE)
        mask_till = tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE)
    else:
        nomask_till = tl.cdiv(N_KEYS, K_TILE_SIZE)
        mask_till = nomask_till

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi_curr = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi_last = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)

    for j in range(nomask_till):
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')
        Sij = tl.dot(Q_tile, K_tile) * scale

        mi_curr = tl.maximum(mi_last, tl.max(Sij, axis=1))
        Pij = tl.exp(Sij - mi_curr[:, None])
        adj_coef = tl.exp(mi_last - mi_curr)

        li *= adj_coef
        li += tl.sum(Pij, axis=1)

        Oi *= adj_coef[:, None]
        Oi += tl.dot(Pij.to(V_tile.dtype), V_tile)

        mi_last = mi_curr

        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))


    for j in range(nomask_till, mask_till):
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')
        Sij = tl.dot(Q_tile, K_tile) * scale

        offsets_q = (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE))[:, None]
        offsets_k = (j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE))[None, :]
        mask = offsets_q >= offsets_k

        Sij = tl.where(mask, Sij, -1e6)

        mi_curr = tl.maximum(mi_last, tl.max(Sij, axis=1))
        Pij = tl.exp(Sij - mi_curr[:, None])
        adj_coef = tl.exp(mi_last - mi_curr)

        li *= adj_coef
        li += tl.sum(Pij, axis=1)

        Oi *= adj_coef[:, None]
        Oi += tl.dot(Pij, V_tile)

        mi_last = mi_curr

        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    tl.store(O_block_ptr, (Oi / li[:, None]).to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, mi_curr + tl.log(li))
    return
 

class FlashAttentionV2NoTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        _, seq_len, _ = Q.shape[-1], Q.shape[-2], Q.shape[:-2]
        d_k, _, _ = K.shape[-1], K.shape[-2], K.shape[:-2]
        pow2 = 2 ** int(np.floor(np.log2(16 * 1024 // (4 * d_k))))

        ctx.Q_TILE_SIZE = min(seq_len, 512)# , pow2)
        ctx.K_TILE_SIZE = min(seq_len, 512)# , pow2)

        Qf = Q.flatten(end_dim=-3)
        Kf = K.flatten(end_dim=-3)
        Vf = V.flatten(end_dim=-3)
        
        O = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device)
        L = torch.empty(*Q.shape[:-1], dtype=torch.float32, device=Q.device)
        Of = O.flatten(end_dim=-3)
        Lf = L.flatten(end_dim=-2)

        flash_v2_single_head_helper(ctx, Qf, Kf, Vf, Of, Lf, is_causal)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        raise NotImplementedError

def falsh_attention_v2_no_triton(Q, K, V, is_causal=False):
    return FlashAttentionV2NoTriton.apply(Q, K, V, is_causal)

class FlashAttentionV2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        for t in [Q, K, V]:
            assert t.is_cuda, "Expected CUDA tensors"
            assert t.is_contiguous(), "Expected contiguous tensors"


        _, seq_len, _ = Q.shape[-1], Q.shape[-2], Q.shape[:-2]
        d_k, _, _ = K.shape[-1], K.shape[-2], K.shape[:-2]
        pow2 = 2 ** int(np.floor(np.log2(16 * 1024 // (4 * d_k))))

        ctx.Q_TILE_SIZE = min(seq_len, 64)# , pow2)
        ctx.K_TILE_SIZE = min(seq_len, 64)# , pow2)

        Qf = Q.flatten(end_dim=-3)
        Kf = K.flatten(end_dim=-3)
        Vf = V.flatten(end_dim=-3)
        
        O = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device)
        L = torch.empty(*Q.shape[:-1], dtype=torch.float32, device=Q.device)
        Of = O.flatten(end_dim=-3)
        Lf = L.flatten(end_dim=-2)

        Tq = triton.cdiv(seq_len, ctx.Q_TILE_SIZE)
        B = Qf.shape[0]

        scale = d_k ** (-0.5)

        flash_fwd_kernel[(Tq, B)](
            Qf, Kf, Vf, Of, Lf,
            Qf.stride(0), Qf.stride(1), Qf.stride(2),
            Kf.stride(0), Kf.stride(1), Kf.stride(2),
            Vf.stride(0), Vf.stride(1), Vf.stride(2),
            Of.stride(0), Of.stride(1), Of.stride(2),
            Lf.stride(0), Lf.stride(1),
            seq_len, seq_len,
            scale,
            is_causal=is_causal,
            D=d_k,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        raise NotImplementedError

def falsh_attention_v2_triton(Q, K, V, is_causal=False):
    return FlashAttentionV2Triton.apply(Q, K, V, is_causal)

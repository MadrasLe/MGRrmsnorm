"""
🔄 Triton RoPE Kernel — Fused Rotary Position Embeddings
---------------------------------------------------------
Replaces 8+ PyTorch ops with a single Triton kernel.
Supports both interleaved (LLaMA) and half-rotate (Qwen/Gemma) modes.

v2: Per-head grid — each program handles ONE (batch, seq_pos, head)
    instead of looping over all heads. Fixes register spill on all GPUs.

Author: Gabriel Yogi / MegaGemm
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


@triton.jit
def _rope_fwd_kernel(
    # Input/output pointers
    X_ptr, X_OUT_ptr,          # [batch, heads, seq_len, head_dim]
    COS_ptr, SIN_ptr,          # [max_seq_len, half_dim] or [batch, seq_len, half_dim]
    # Strides for X (same for X_OUT)
    stride_xb, stride_xh, stride_xs, stride_xd,
    # Strides for cos/sin
    stride_cb, stride_cs, stride_cd,
    # Dims
    seq_len,
    HALF_DIM: tl.constexpr,
    HALF_ROTATE: tl.constexpr,  # 0=interleaved, 1=half-rotate (Qwen)
    HAS_BATCH_COS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Each program handles ONE (batch, seq_pos, head) triple.
    No in-kernel head loop → minimal register pressure.
    """
    # 2D grid: (batch * seq_len, num_heads)
    pos_id = tl.program_id(0)    # flattened (batch, seq_pos)
    head_id = tl.program_id(1)   # head index

    batch_idx = pos_id // seq_len
    seq_idx = pos_id % seq_len

    # Load cos/sin for this position
    if HAS_BATCH_COS:
        cos_base = COS_ptr + batch_idx * stride_cb + seq_idx * stride_cs
        sin_base = SIN_ptr + batch_idx * stride_cb + seq_idx * stride_cs
    else:
        cos_base = COS_ptr + seq_idx * stride_cs
        sin_base = SIN_ptr + seq_idx * stride_cs

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < HALF_DIM

    cos_vals = tl.load(cos_base + d_offsets * stride_cd, mask=d_mask, other=1.0)
    sin_vals = tl.load(sin_base + d_offsets * stride_cd, mask=d_mask, other=0.0)

    # Base pointer for this (batch, head, seq_pos)
    x_base = X_ptr + batch_idx * stride_xb + head_id * stride_xh + seq_idx * stride_xs
    xo_base = X_OUT_ptr + batch_idx * stride_xb + head_id * stride_xh + seq_idx * stride_xs

    if HALF_ROTATE:
        # Half-rotate: first half paired with second half
        first = tl.load(x_base + d_offsets * stride_xd, mask=d_mask, other=0.0)
        second = tl.load(x_base + (d_offsets + HALF_DIM) * stride_xd, mask=d_mask, other=0.0)

        first_rot = first * cos_vals - second * sin_vals
        second_rot = second * cos_vals + first * sin_vals

        tl.store(xo_base + d_offsets * stride_xd, first_rot, mask=d_mask)
        tl.store(xo_base + (d_offsets + HALF_DIM) * stride_xd, second_rot, mask=d_mask)
    else:
        # Interleaved: even/odd pairs
        even = tl.load(x_base + (2 * d_offsets) * stride_xd, mask=d_mask, other=0.0)
        odd = tl.load(x_base + (2 * d_offsets + 1) * stride_xd, mask=d_mask, other=0.0)

        even_rot = even * cos_vals - odd * sin_vals
        odd_rot = even * sin_vals + odd * cos_vals

        tl.store(xo_base + (2 * d_offsets) * stride_xd, even_rot, mask=d_mask)
        tl.store(xo_base + (2 * d_offsets + 1) * stride_xd, odd_rot, mask=d_mask)


def triton_apply_rotary_emb(
    q: torch.Tensor,          # [batch, q_heads, seq_len, head_dim]
    k: torch.Tensor,          # [batch, kv_heads, seq_len, head_dim]
    cos: torch.Tensor,        # [max_seq_len, head_dim//2] or [batch, seq_len, head_dim//2]
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    half_rotate: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton RoPE: replaces 8+ PyTorch ops with 2 kernel launches (Q + K).
    Each launch uses a 2D grid: (batch * seq_len, num_heads).
    """
    bsz, num_q_heads, seq_len, head_dim = q.shape
    num_kv_heads = k.shape[1]
    half_dim = head_dim // 2

    # Gather cos/sin by position_ids if needed
    if position_ids is not None:
        cos = cos[position_ids]  # [batch, seq_len, half_dim]
        sin = sin[position_ids]
        has_batch_cos = True
    else:
        cos = cos[:seq_len]      # [seq_len, half_dim]
        sin = sin[:seq_len]
        has_batch_cos = False

    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)

    # Output tensors
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    BLOCK_D = triton.next_power_of_2(half_dim)

    # Cos/sin strides
    if has_batch_cos:
        cs_b, cs_s, cs_d = cos.stride(0), cos.stride(1), cos.stride(2)
    else:
        cs_b, cs_s, cs_d = 0, cos.stride(0), cos.stride(1)

    # Launch 1: Apply RoPE to Q — grid (bsz * seq_len, num_q_heads)
    _rope_fwd_kernel[(bsz * seq_len, num_q_heads)](
        q, q_out, cos, sin,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        cs_b, cs_s, cs_d,
        seq_len=seq_len,
        HALF_DIM=half_dim,
        HALF_ROTATE=1 if half_rotate else 0,
        HAS_BATCH_COS=1 if has_batch_cos else 0,
        BLOCK_D=BLOCK_D,
    )

    # Launch 2: Apply RoPE to K — grid (bsz * seq_len, num_kv_heads)
    _rope_fwd_kernel[(bsz * seq_len, num_kv_heads)](
        k, k_out, cos, sin,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cs_b, cs_s, cs_d,
        seq_len=seq_len,
        HALF_DIM=half_dim,
        HALF_ROTATE=1 if half_rotate else 0,
        HAS_BATCH_COS=1 if has_batch_cos else 0,
        BLOCK_D=BLOCK_D,
    )

    return q_out, k_out

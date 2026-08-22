"""
Decode-only fused RMSNormGated + Linear for Qwen 3.5 linear attention.

Computes, for one decode token:
  y = linear(reshape(rmsnorm(x_per_head) * silu(gate) * norm_weight))

The RMSNorm is per value head, matching RMSNormGated in models/llama.py.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


_CFG_BLOCK_N = max(1, _env_int("MEGAGEMM_RMSNORM_GATED_LINEAR_BLOCK_N", 16))


if _HAS_TRITON:
    @triton.jit
    def _rmsnorm_gated_linear_kernel(
        x_ptr,       # [B, H, D]
        gate_ptr,    # [B, H, D]
        norm_ptr,    # [D]
        w_ptr,       # [N, H * D]
        b_ptr,       # [N] or dummy
        out_ptr,     # [B, N]
        stride_xb, stride_xh, stride_xd,
        stride_gb, stride_gh, stride_gd,
        stride_wn, stride_wk,
        stride_ob, stride_on,
        out_dim,
        eps,
        HAS_BIAS: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < out_dim
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for head in tl.static_range(0, NUM_HEADS):
            x = tl.load(
                x_ptr + batch * stride_xb + head * stride_xh + offs_d * stride_xd,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            gate = tl.load(
                gate_ptr + batch * stride_gb + head * stride_gh + offs_d * stride_gd,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            norm_w = tl.load(norm_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)

            inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
            gated = x * inv_rms * norm_w * (gate * tl.sigmoid(gate))

            offs_k = head * HEAD_DIM + offs_d
            w = tl.load(
                w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * gated[None, :], axis=1)

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias

        tl.store(
            out_ptr + batch * stride_ob + offs_n * stride_on,
            acc.to(out_ptr.dtype.element_ty),
            mask=n_mask,
        )


def _fallback(
    x: torch.Tensor,
    gate: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    linear_weight: torch.Tensor,
    linear_bias: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    batch, num_heads, head_dim = x.shape
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = x.float() * torch.rsqrt(variance + eps)
    normed = normed * norm_weight.view(1, 1, head_dim).float()
    normed = normed * torch.nn.functional.silu(gate.float())
    flat = normed.to(x.dtype).reshape(batch, num_heads * head_dim)
    result = F.linear(flat, linear_weight, linear_bias)
    if out is None:
        return result
    out.copy_(result)
    return out


def rmsnorm_gated_linear_decode(
    x: torch.Tensor,
    gate: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    linear_weight: torch.Tensor,
    linear_bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.shape != gate.shape or x.ndim != 3:
        raise ValueError("x/gate must both be [batch, num_heads, head_dim]")
    batch, num_heads, head_dim = x.shape
    if norm_weight.numel() != head_dim:
        raise ValueError("norm_weight size must match head_dim")
    if linear_weight.shape[1] != num_heads * head_dim:
        raise ValueError("linear_weight input dim must match num_heads * head_dim")

    out_dim = int(linear_weight.shape[0])
    if out is None:
        out = torch.empty((batch, out_dim), device=x.device, dtype=x.dtype)
    elif tuple(out.shape) != (batch, out_dim) or out.device != x.device or out.dtype != x.dtype:
        raise ValueError("out must be [batch, out_dim] with same device/dtype as x")

    if not (
        _HAS_TRITON
        and x.is_cuda
        and gate.is_cuda
        and norm_weight.is_cuda
        and linear_weight.is_cuda
        and not torch.is_grad_enabled()
        and x.stride(-1) == 1
        and gate.stride(-1) == 1
        and head_dim > 0
        and head_dim <= 256
    ):
        return _fallback(x, gate, norm_weight, eps, linear_weight, linear_bias, out)

    w = linear_weight if linear_weight.is_contiguous() else linear_weight.contiguous()
    norm_w = norm_weight if norm_weight.is_contiguous() else norm_weight.contiguous()
    bias_ptr = linear_bias if linear_bias is not None else out
    block_d = triton.next_power_of_2(head_dim)
    block_n = min(_CFG_BLOCK_N, triton.next_power_of_2(out_dim))
    grid = (batch, triton.cdiv(out_dim, block_n))
    _rmsnorm_gated_linear_kernel[grid](
        x,
        gate,
        norm_w,
        w,
        bias_ptr,
        out,
        x.stride(0), x.stride(1), x.stride(2),
        gate.stride(0), gate.stride(1), gate.stride(2),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        out_dim,
        float(eps),
        HAS_BIAS=1 if linear_bias is not None else 0,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out


HAS_RMSNORM_GATED_LINEAR = _HAS_TRITON


def rmsnorm_gated_linear_runtime_config() -> dict:
    return {
        "has_triton": bool(_HAS_TRITON),
        "block_n": int(_CFG_BLOCK_N),
    }


__all__ = [
    "rmsnorm_gated_linear_decode",
    "rmsnorm_gated_linear_runtime_config",
    "HAS_RMSNORM_GATED_LINEAR",
]

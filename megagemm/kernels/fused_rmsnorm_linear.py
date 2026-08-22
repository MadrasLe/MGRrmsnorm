"""
Decode-oriented fused RMSNorm + Linear kernel.

Computes:
  y = linear(rmsnorm(x, norm_weight, eps, offset))

This avoids materializing the normalized activation tensor in HBM for decode.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


_CFG_SHAPE_GUARD = _env_bool("MEGAGEMM_FUSED_RMSNORM_LINEAR_SHAPE_GUARD", True)
_CFG_FORCE_TRITON = _env_bool("MEGAGEMM_FUSED_RMSNORM_LINEAR_FORCE_TRITON", False)
_CFG_MAX_ROWS = max(1, _env_int("MEGAGEMM_FUSED_RMSNORM_LINEAR_MAX_ROWS", 4))
_CFG_MIN_N = max(1, _env_int("MEGAGEMM_FUSED_RMSNORM_LINEAR_MIN_N", 1024))
_CFG_MIN_K = max(1, _env_int("MEGAGEMM_FUSED_RMSNORM_LINEAR_MIN_K", 1024))
_CFG_FORCED_BN = _env_int("MEGAGEMM_FUSED_RMSNORM_LINEAR_BLOCK_N", 0)
_CFG_FORCED_BK = _env_int("MEGAGEMM_FUSED_RMSNORM_LINEAR_BLOCK_K", 0)
_CFG_FORCED_WARPS = _env_int("MEGAGEMM_FUSED_RMSNORM_LINEAR_NUM_WARPS", 0)
_CFG_FORCED_STAGES = _env_int("MEGAGEMM_FUSED_RMSNORM_LINEAR_NUM_STAGES", 0)
_CFG_PREFILL_ENABLED = _env_bool("MEGAGEMM_FUSED_RMSNORM_LINEAR_PREFILL", False)
_CFG_PREFILL_FORCE_TRITON = _env_bool("MEGAGEMM_FUSED_RMSNORM_LINEAR_PREFILL_FORCE_TRITON", False)
_CFG_PREFILL_MIN_ROWS = max(1, _env_int("MEGAGEMM_FUSED_RMSNORM_LINEAR_PREFILL_MIN_ROWS", 128))
_CFG_TWO_PASS = _env_bool("MEGAGEMM_FUSED_RMSNORM_LINEAR_TWO_PASS", True)
_CFG_TWO_PASS_MIN_N = max(1, _env_int("MEGAGEMM_FUSED_RMSNORM_LINEAR_TWO_PASS_MIN_N", 8192))
_CFG_TWO_PASS_MIN_K = max(1, _env_int("MEGAGEMM_FUSED_RMSNORM_LINEAR_TWO_PASS_MIN_K", 2048))


if _HAS_TRITON:
    @triton.jit
    def _fused_rmsnorm_linear_kernel(
        x_ptr,        # [M, K]
        nw_ptr,       # [K]
        w_ptr,        # [N, K]
        b_ptr,        # [N] or dummy
        y_ptr,        # [M, N]
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        K,
        N,
        eps,
        HAS_BIAS: tl.constexpr,
        OFFSET: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        base_x = x_ptr + pid_m * stride_xm

        sum_sq = tl.zeros((), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            x = tl.load(
                base_x + offs_k * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=0)

        inv_rms = tl.rsqrt(sum_sq / K + eps)
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x = tl.load(
                base_x + offs_k * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            nw = tl.load(
                nw_ptr + offs_k,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            if OFFSET:
                nw = nw + 1.0
            nx = x * inv_rms * nw

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w_mask = n_mask[:, None] & k_mask[None, :]
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
            acc += tl.sum(w * nx[None, :], axis=1)

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias

        y_ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=n_mask)

    @triton.jit
    def _rmsnorm_inv_kernel(
        x_ptr,
        inv_ptr,
        stride_xm, stride_xk,
        K,
        eps,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_k = tl.arange(0, BLOCK_K)
        mask = offs_k < K
        x = tl.load(
            x_ptr + pid_m * stride_xm + offs_k * stride_xk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sum_sq = tl.sum(x * x, axis=0)
        tl.store(inv_ptr + pid_m, tl.rsqrt(sum_sq / K + eps))

    @triton.jit
    def _rmsnorm_linear_from_inv_kernel(
        x_ptr,
        nw_ptr,
        inv_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        K,
        N,
        HAS_BIAS: tl.constexpr,
        OFFSET: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        inv_rms = tl.load(inv_ptr + pid_m).to(tl.float32)
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            x = tl.load(
                x_ptr + pid_m * stride_xm + offs_k * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            nw = tl.load(
                nw_ptr + offs_k,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            if OFFSET:
                nw = nw + 1.0
            nx = x * inv_rms * nw
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w = tl.load(
                w_ptrs,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * nx[None, :], axis=1)

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias

        y_ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


def _pick_cfg(k_dim: int, n_dim: int, rows: int):
    if _CFG_FORCED_BN > 0 and _CFG_FORCED_BK > 0 and _CFG_FORCED_WARPS > 0:
        return _CFG_FORCED_BN, _CFG_FORCED_BK, _CFG_FORCED_WARPS, max(1, _CFG_FORCED_STAGES or 2)

    if n_dim >= 16384:
        block_n = 16
    elif n_dim >= 4096:
        block_n = 16
    elif n_dim >= 2048:
        block_n = 32
    else:
        block_n = 64

    if k_dim >= 2048:
        block_k = 128
    elif k_dim >= 1024:
        block_k = 128
    else:
        block_k = 64

    if rows > 1 and block_n < 64:
        block_n = min(64, block_n * 2)

    num_warps = 4
    num_stages = 2
    return block_n, block_k, num_warps, num_stages


def _flatten_rows(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x if x.is_contiguous() else x.contiguous()
    x_2d = x.flatten(0, -2)
    return x_2d if x_2d.is_contiguous() else x_2d.contiguous()


def _fallback_rmsnorm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    linear_weight: torch.Tensor,
    linear_bias: Optional[torch.Tensor],
    norm_offset: bool,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    w = (norm_weight + 1.0) if norm_offset else norm_weight
    normed = (normed * w).to(x.dtype)
    if out is None:
        return F.linear(normed, linear_weight, linear_bias)
    out.copy_(F.linear(normed, linear_weight, linear_bias))
    return out


def fused_rmsnorm_linear_prefers_triton_shape(
    in_dim: int,
    out_dim: int,
    rows: int,
    mode: str = "decode",
) -> bool:
    if mode == "prefill":
        if _CFG_PREFILL_FORCE_TRITON:
            return True
        if not _CFG_PREFILL_ENABLED:
            return False
        if rows < _CFG_PREFILL_MIN_ROWS:
            return False
        if in_dim < _CFG_MIN_K:
            return False
        if out_dim < _CFG_MIN_N:
            return False
        return True
    if _CFG_FORCE_TRITON:
        return True
    if not _CFG_SHAPE_GUARD:
        return True
    if rows > _CFG_MAX_ROWS:
        return False
    if in_dim < _CFG_MIN_K:
        return False
    if out_dim < _CFG_MIN_N:
        return False
    return True


def fused_rmsnorm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    linear_weight: torch.Tensor,
    linear_bias: Optional[torch.Tensor] = None,
    norm_offset: bool = False,
    out: Optional[torch.Tensor] = None,
    mode: str = "decode",
    inv_rms: Optional[torch.Tensor] = None,
    two_pass: Optional[bool] = None,
) -> torch.Tensor:
    """
    Decode-oriented fused RMSNorm + Linear.
    """
    if x.shape[-1] != linear_weight.shape[-1]:
        raise ValueError(
            f"in_features mismatch: x={x.shape[-1]} linear_weight={linear_weight.shape[-1]}"
        )
    if norm_weight.shape[0] != x.shape[-1]:
        raise ValueError(
            f"norm_weight mismatch: got {norm_weight.shape[0]} expected {x.shape[-1]}"
        )

    orig_shape = x.shape
    x_2d = _flatten_rows(x)
    m_rows = int(x_2d.shape[0])
    k_dim = int(x_2d.shape[1])
    n_dim = int(linear_weight.shape[0])
    use_triton = (
        _HAS_TRITON
        and x.is_cuda
        and linear_weight.is_cuda
        and norm_weight.is_cuda
        and not torch.is_grad_enabled()
        and fused_rmsnorm_linear_prefers_triton_shape(k_dim, n_dim, m_rows, mode=mode)
    )

    if out is None:
        out_2d = torch.empty((m_rows, n_dim), device=x.device, dtype=x.dtype)
    else:
        expected = (*orig_shape[:-1], n_dim)
        if tuple(out.shape) != tuple(expected):
            raise ValueError(f"out shape mismatch: got {tuple(out.shape)} expected {tuple(expected)}")
        if out.device != x.device or out.dtype != x.dtype:
            raise ValueError("out must match x device and dtype")
        out_2d = out if out.ndim == 2 else out.flatten(0, -2)

    if not use_triton:
        return _fallback_rmsnorm_linear(
            x,
            norm_weight,
            eps,
            linear_weight,
            linear_bias,
            norm_offset,
            out,
        )

    w = linear_weight if linear_weight.is_contiguous() else linear_weight.contiguous()
    norm_w = norm_weight if norm_weight.is_contiguous() else norm_weight.contiguous()
    bias_ptr = linear_bias if linear_bias is not None else x_2d
    block_n, block_k, num_warps, num_stages = _pick_cfg(k_dim, n_dim, m_rows)
    use_two_pass = (
        bool(_CFG_TWO_PASS if two_pass is None else two_pass)
        and mode == "decode"
        and m_rows <= _CFG_MAX_ROWS
        and n_dim >= _CFG_TWO_PASS_MIN_N
        and k_dim >= _CFG_TWO_PASS_MIN_K
    )
    if use_two_pass:
        if (
            inv_rms is None
            or tuple(inv_rms.shape) != (m_rows,)
            or inv_rms.device != x.device
            or inv_rms.dtype != torch.float32
        ):
            inv_rms = torch.empty((m_rows,), device=x.device, dtype=torch.float32)
        inv_block = triton.next_power_of_2(k_dim)
        _rmsnorm_inv_kernel[(m_rows,)](
            x_2d,
            inv_rms,
            x_2d.stride(0), x_2d.stride(1),
            k_dim,
            float(eps),
            BLOCK_K=inv_block,
            num_warps=min(8, max(1, inv_block // 512)),
            num_stages=2,
        )
        grid = (m_rows, triton.cdiv(n_dim, block_n))
        _rmsnorm_linear_from_inv_kernel[grid](
            x_2d,
            norm_w,
            inv_rms,
            w,
            bias_ptr,
            out_2d,
            x_2d.stride(0), x_2d.stride(1),
            w.stride(0), w.stride(1),
            out_2d.stride(0), out_2d.stride(1),
            k_dim,
            n_dim,
            HAS_BIAS=1 if linear_bias is not None else 0,
            OFFSET=1 if norm_offset else 0,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        if out is not None:
            return out
        return out_2d.view(*orig_shape[:-1], n_dim)

    grid = (m_rows, triton.cdiv(n_dim, block_n))
    _fused_rmsnorm_linear_kernel[grid](
        x_2d,
        norm_w,
        w,
        bias_ptr,
        out_2d,
        x_2d.stride(0), x_2d.stride(1),
        w.stride(0), w.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        k_dim,
        n_dim,
        float(eps),
        HAS_BIAS=1 if linear_bias is not None else 0,
        OFFSET=1 if norm_offset else 0,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if out is not None:
        return out
    return out_2d.view(*orig_shape[:-1], n_dim)


HAS_FUSED_RMSNORM_LINEAR = _HAS_TRITON


def fused_rmsnorm_linear_runtime_config() -> dict:
    return {
        "has_triton": bool(_HAS_TRITON),
        "shape_guard": bool(_CFG_SHAPE_GUARD),
        "force_triton": bool(_CFG_FORCE_TRITON),
        "prefill_enabled": bool(_CFG_PREFILL_ENABLED),
        "prefill_force_triton": bool(_CFG_PREFILL_FORCE_TRITON),
        "prefill_min_rows": int(_CFG_PREFILL_MIN_ROWS),
        "two_pass": bool(_CFG_TWO_PASS),
        "two_pass_min_n": int(_CFG_TWO_PASS_MIN_N),
        "two_pass_min_k": int(_CFG_TWO_PASS_MIN_K),
        "max_rows": int(_CFG_MAX_ROWS),
        "min_n": int(_CFG_MIN_N),
        "min_k": int(_CFG_MIN_K),
        "forced_block_n": int(_CFG_FORCED_BN),
        "forced_block_k": int(_CFG_FORCED_BK),
        "forced_num_warps": int(_CFG_FORCED_WARPS),
        "forced_num_stages": int(_CFG_FORCED_STAGES),
    }


__all__ = [
    "fused_rmsnorm_linear",
    "fused_rmsnorm_linear_prefers_triton_shape",
    "fused_rmsnorm_linear_runtime_config",
    "HAS_FUSED_RMSNORM_LINEAR",
]

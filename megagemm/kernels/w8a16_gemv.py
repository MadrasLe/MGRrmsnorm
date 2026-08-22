"""
⚡ W8A16 Fused GEMV — Triton Kernel (v3 — zero overhead)
---------------------------------------------------------
Fused INT8 weight dequantization GEMV for decode (M=1..small).

v3: NO autotune (saves ~15μs lookup per call × 112 = 1.7ms/token),
    fixed BLOCK sizes, designed for DIRECT invocation from hot loop.

Author: Gabriel Yogi
"""

import os

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

__all__ = ['w8a16_gemv_direct', 'HAS_W8A16_GEMV', 'precompute_w8a16_grid']


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ═══════════════════════════════════════════════════════
# Triton Kernel — NO autotune, fixed BLOCK for decode
# ═══════════════════════════════════════════════════════

# Fixed block sizes tuned for L4/A100 M=1 decode:
# BLOCK_N=32: enough programs (e.g. 17920/32=560) to saturate SMs
# BLOCK_K=256: long K tiles to amortize launch cost, fits in registers
_W8A16_BLOCK_N = _env_int("MEGAGEMM_W8A16_BLOCK_N", 32)
_W8A16_BLOCK_K = _env_int("MEGAGEMM_W8A16_BLOCK_K", 256)
_W8A16_NUM_WARPS = _env_int("MEGAGEMM_W8A16_WARPS", 4)
_W8A16_NUM_STAGES = _env_int("MEGAGEMM_W8A16_STAGES", 2)
_W8A16_GEMM_BLOCK_M = _env_int("MEGAGEMM_W8A16_GEMM_BLOCK_M", 8)
_W8A16_GEMM_BLOCK_N = _env_int("MEGAGEMM_W8A16_GEMM_BLOCK_N", 64)
_W8A16_GEMM_BLOCK_K = _env_int("MEGAGEMM_W8A16_GEMM_BLOCK_K", 64)
_W8A16_GEMM_NUM_WARPS = _env_int("MEGAGEMM_W8A16_GEMM_WARPS", 4)
_W8A16_GEMM_NUM_STAGES = _env_int("MEGAGEMM_W8A16_GEMM_STAGES", 2)
_USE_W8A16_SMALL_M_GEMM = os.environ.get(
    "MEGAGEMM_W8A16_SMALL_M_GEMM", "0"
).strip().lower() in {"1", "true", "yes", "on"}

if _HAS_TRITON:
    @triton.jit
    def _w8a16_gemv_kernel(
        x_ptr, w_ptr, scale_ptr, out_ptr, bias_ptr,
        N, K,
        stride_xm, stride_wn, stride_om,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        w_base = w_ptr + offs_n * stride_wn

        for k_off in range(0, K, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x = tl.load(
                x_ptr + pid_m * stride_xm + offs_k,
                mask=mask_k, other=0.0,
            ).to(tl.float32)

            w_i8 = tl.load(
                w_base[:, None] + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0,
            )

            acc += tl.sum(w_i8.to(tl.float32) * x[None, :], axis=1)

        scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
        acc *= scale

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias

        tl.store(
            out_ptr + pid_m * stride_om + offs_n,
            acc.to(tl.float16),
            mask=mask_n,
        )

    @triton.jit
    def _w8a16_small_m_gemm_kernel(
        x_ptr, w_ptr, scale_ptr, out_ptr, bias_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_off in range(0, K, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )

            w_i8 = tl.load(
                w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            ).to(tl.float32)
            w_scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
            w = (w_i8 * w_scale[None, :]).to(tl.float16)
            acc += tl.dot(x, w, out_dtype=tl.float32)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[None, :]

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc.to(tl.float16),
            mask=mask_m[:, None] & mask_n[None, :],
        )


# ═══════════════════════════════════════════════════════
# Pre-computed grid (computed ONCE at setup, reused 28×4 per token)
# ═══════════════════════════════════════════════════════

def precompute_w8a16_grid(N: int, M: int = 1) -> tuple:
    """Pre-compute the kernel grid for a given (N, M).
    Call once during _prepare_flat_decode, store the result."""
    if not _HAS_TRITON:
        return (1, 1)
    if M > 1:
        block_n = _W8A16_GEMM_BLOCK_N if _USE_W8A16_SMALL_M_GEMM else _W8A16_BLOCK_N
        return (triton.cdiv(N, block_n), triton.cdiv(M, _W8A16_GEMM_BLOCK_M))
    return (triton.cdiv(N, _W8A16_BLOCK_N), M)


# ═══════════════════════════════════════════════════════
# Direct invocation — minimal Python overhead
# ═══════════════════════════════════════════════════════

def w8a16_gemv_direct(
    x,              # [M, K] fp16, contiguous
    weight_int8,    # [N, K] int8, contiguous
    scale,          # [N] fp16
    bias,           # [N] or None
    out,            # [M, N] fp16, pre-allocated
    grid,           # pre-computed (cdiv(N, BLOCK_N), M)
):
    """
    Zero-overhead W8A16 GEMV: call the Triton kernel directly.

    No autotune, no allocation, no squeeze/unsqueeze, no contiguous check.
    All those are done ONCE during setup — this function is pure kernel dispatch.
    """
    if not _HAS_TRITON:
        w = weight_int8.to(x.dtype) * scale.unsqueeze(1).to(x.dtype)
        out.copy_(torch.nn.functional.linear(x, w, bias))
        return out

    if x.shape[0] > 1 and _USE_W8A16_SMALL_M_GEMM:
        try:
            _w8a16_small_m_gemm_kernel[grid](
                x, weight_int8, scale, out,
                bias if bias is not None else scale,
                x.shape[0],
                weight_int8.shape[0],   # N
                weight_int8.shape[1],   # K
                x.stride(0),
                x.stride(1),
                weight_int8.stride(0),
                weight_int8.stride(1),
                out.stride(0),
                out.stride(1),
                HAS_BIAS=(bias is not None),
                BLOCK_M=_W8A16_GEMM_BLOCK_M,
                BLOCK_N=_W8A16_GEMM_BLOCK_N,
                BLOCK_K=_W8A16_GEMM_BLOCK_K,
                num_warps=_W8A16_GEMM_NUM_WARPS,
                num_stages=_W8A16_GEMM_NUM_STAGES,
            )
            return out
        except Exception:
            pass

    gemv_grid = (triton.cdiv(weight_int8.shape[0], _W8A16_BLOCK_N), x.shape[0])
    _w8a16_gemv_kernel[gemv_grid](
        x, weight_int8, scale, out,
        bias if bias is not None else scale,
        weight_int8.shape[0],   # N
        weight_int8.shape[1],   # K
        x.stride(0),
        weight_int8.stride(0),
        out.stride(0),
        HAS_BIAS=(bias is not None),
        BLOCK_N=_W8A16_BLOCK_N,
        BLOCK_K=_W8A16_BLOCK_K,
        num_warps=_W8A16_NUM_WARPS,
        num_stages=_W8A16_NUM_STAGES,
    )
    return out


# Backward compatible wrapper (used by non-hot-loop code)
def w8a16_gemv(x, weight_int8, scale, bias=None, out=None):
    """Convenience wrapper for non-performance-critical paths."""
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)
    M, K = x.shape
    N = weight_int8.shape[0]
    if out is None:
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    grid = precompute_w8a16_grid(N, M)
    w8a16_gemv_direct(x, weight_int8, scale, bias, out, grid)
    return out.squeeze(0) if squeeze else out


HAS_W8A16_GEMV = _HAS_TRITON

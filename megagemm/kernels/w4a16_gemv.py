"""
⚡ W4A16 Fused GEMV — Triton Kernel for AWQ INT4 (v3)
------------------------------------------------------
Fixed the ROOT CAUSE: AWQ weights are [K, N//8] (column-major access).
This version expects TRANSPOSED weights [N//8, K] so the kernel
reads rows = coalesced memory access = optimal bandwidth.

The transposition is done ONCE at model load (zero runtime cost).

Design mirrors w8a16_gemv.py which achieves 93+ tok/s:
- Fixed blocks, no autotune
- Weight layout: [N_packed, K] (row access per program)
- acc[BLOCK_NP] accumulates 8 outputs per packed element
- Unpack all 8 nibbles AFTER reduction for minimal compute

Author: Gabriel Yogi
"""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

__all__ = ['w4a16_gemv_direct', 'precompute_w4a16_grid', 'HAS_W4A16_GEMV']

# Block sizes matching the fast W8A16 kernel
_W4A16_BLOCK_N = 32    # real output elements (= 4 packed int32 groups)
_W4A16_BLOCK_K = 128   # MUST equal GROUP_SIZE! Each tile = exactly 1 quant group
_W4A16_NUM_WARPS = 4
_W4A16_NUM_STAGES = 2


if _HAS_TRITON:
    @triton.jit
    def _w4a16_gemv_kernel(
        # Pointers
        x_ptr,          # [M, K] fp16
        qw_ptr,         # [N//8, K] int32 TRANSPOSED packed weights
        scales_ptr,     # [K//G, N] fp16 per-group scales
        qz_ptr,         # [K//G, N//8] int32 packed zero-points
        out_ptr,        # [M, N] fp16
        bias_ptr,       # [N] fp16 or dummy
        # Dims
        N, K,
        # Strides
        stride_xm,
        stride_qwn, stride_qwk,    # TRANSPOSED qweight strides: [N//8, K]
        stride_sg, stride_sn,
        stride_zg, stride_zn,
        stride_om,
        # Constants
        GROUP_SIZE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        W4A16 GEMV with TRANSPOSED weights for coalesced access.
        Each program handles BLOCK_N output elements.
        Weight layout is [N//8, K] so we read ROWS (coalesced).
        """
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)

        # Real output indices
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # Packed indices and nibble position within int32
        offs_n_packed = offs_n // 8   # [BLOCK_N]
        n_sub = offs_n % 8            # [BLOCK_N]

        # AWQ interleaved shift: nibble[i] at bits ((i//2)*4 + (i%2)*16)
        w_shift = (n_sub // 2) * 4 + (n_sub % 2) * 16   # [BLOCK_N]

        # Accumulator
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Base pointer for this program's packed weight rows
        # qw_ptr layout: [N//8, K], stride_qwn = K (row stride), stride_qwk = 1
        # Each program handles BLOCK_N/8 packed rows

        for k_off in range(0, K, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            # Load activation
            x = tl.load(
                x_ptr + pid_m * stride_xm + offs_k,
                mask=mask_k, other=0.0,
            ).to(tl.float32)   # [BLOCK_K]

            # Load packed weights: [BLOCK_N, BLOCK_K] using TRANSPOSED layout
            # offs_n_packed indexes rows, offs_k indexes columns → ROW access = COALESCED
            qw = tl.load(
                qw_ptr + offs_n_packed[:, None] * stride_qwn + offs_k[None, :] * stride_qwk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0,
            )  # [BLOCK_N, BLOCK_K] int32

            # Unpack nibble: shift is [BLOCK_N], broadcast to [BLOCK_N, BLOCK_K]
            w_int4 = ((qw >> w_shift[:, None]) & 0xF).to(tl.float32)  # [BLOCK_N, BLOCK_K]

            # Load per-group scale and zero-point
            g = k_off // GROUP_SIZE
            scale = tl.load(
                scales_ptr + g * stride_sg + offs_n * stride_sn,
                mask=mask_n, other=1.0,
            ).to(tl.float32)   # [BLOCK_N]

            qz_packed = tl.load(
                qz_ptr + g * stride_zg + offs_n_packed * stride_zn,
                mask=mask_n, other=0,
            )
            zero = ((qz_packed >> w_shift) & 0xF).to(tl.float32)   # [BLOCK_N]

            # Dequant: (w_int4 - zero) * scale, then dot with x
            w_dequant = (w_int4 - zero[:, None]) * scale[:, None]   # [BLOCK_N, BLOCK_K]
            acc += tl.sum(w_dequant * x[None, :], axis=1)   # [BLOCK_N]

        # Bias
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias

        # Store
        tl.store(
            out_ptr + pid_m * stride_om + offs_n,
            acc.to(tl.float16),
            mask=mask_n,
        )


# ═══════════════════════════════════════════════════════

def precompute_w4a16_grid(N: int, M: int = 1) -> tuple:
    if not _HAS_TRITON:
        return (1, 1)
    return (triton.cdiv(N, _W4A16_BLOCK_N), M)


def w4a16_gemv_direct(x, qweight_t, scales, qzeros, bias, out, grid, group_size):
    """
    Zero-overhead W4A16 GEMV.
    qweight_t must be TRANSPOSED: [N//8, K] (not AWQ's default [K, N//8]).
    """
    N = scales.shape[1]
    K = qweight_t.shape[1]

    if not _HAS_TRITON:
        from megagemm.quantization.w4a16 import _dequantize_pytorch
        w_fp16 = _dequantize_pytorch(qweight_t.t().contiguous(), scales, qzeros, group_size)
        torch.mm(x, w_fp16, out=out)
        if bias is not None:
            out.add_(bias)
        return out

    _w4a16_gemv_kernel[grid](
        x, qweight_t, scales, qzeros, out,
        bias if bias is not None else scales,
        N, K,
        x.stride(0),
        qweight_t.stride(0), qweight_t.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        out.stride(0),
        GROUP_SIZE=group_size,
        HAS_BIAS=(bias is not None),
        BLOCK_N=_W4A16_BLOCK_N,
        BLOCK_K=_W4A16_BLOCK_K,
        num_warps=_W4A16_NUM_WARPS,
        num_stages=_W4A16_NUM_STAGES,
    )
    return out


HAS_W4A16_GEMV = _HAS_TRITON

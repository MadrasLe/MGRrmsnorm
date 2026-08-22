"""
⚡ MegaGemm AWQ GEMM — Triton W4A16 Kernel
---------------------------------------------
Custom Triton kernel for AWQ INT4 dequant + FP16 matmul,
optimized for decode (M=1 to small M).

Key optimizations vs AutoAWQ's kernel:
  - No split_k for decode → eliminates intermediate tensor + reduction
  - Triton autotuning → finds optimal block sizes per shape
  - Simpler code path → less register pressure

AWQ weight format:
  - qweight: [K, N//8] int32 — 8 INT4 weights packed per int32
  - qzeros:  [K//G, N//8] int32 — packed zero-points
  - scales:  [K//G, N] float16 — per-group channel scales
  - group_size: typically 128

Author: Gabriel Yogi
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

__all__ = ['awq_gemm_megagemm', 'HAS_AWQ_MEGAGEMM']

# AWQ packing order (reverse interleave)
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

# ═══════════════════════════════════════════════════════
# Triton Kernel
# ═══════════════════════════════════════════════════════

if _HAS_TRITON:
    @triton.autotune(
        configs=[
            # Decode-optimized (M=1 to 4)
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_stages=4, num_warps=2),
            # Prefill-optimized (M=32+)
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _awq_gemm_kernel(
        # Pointers
        x_ptr,           # [M, K] float16 activations
        qweight_ptr,     # [K, N//8] int32 packed weights
        qzeros_ptr,      # [K//G, N//8] int32 packed zeros
        scales_ptr,      # [K//G, N] float16 scales
        out_ptr,         # [M, N] float16 output
        # Dims
        M, N, K,
        # Group size
        group_size,
        # Strides
        stride_xm, stride_xk,
        stride_qwk, stride_qwn,
        stride_qzg, stride_qzn,
        stride_sg, stride_sn,
        stride_om, stride_on,
        # Tile sizes (autotuned)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused AWQ dequant + FP16 GEMM.

        For each K-tile:
          1. Load packed INT4 weights
          2. Unpack 8 values per int32 (bit shift + mask)
          3. Apply AWQ reorder
          4. Dequantize: (w_int4 - zeros) * scales → FP16
          5. tl.dot(x_fp16, w_fp16) → accumulate in FP16/FP32
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Offsets for this tile
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # N offsets in packed space (N//8 columns in qweight)
        offs_n_packed = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
        # N offsets in unpacked space (full N columns)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Precompute AWQ reorder shifts
        # AWQ stores values in order [0,4,1,5,2,6,3,7] within each packed int32
        # We need shifts: [0, 16, 4, 20, 8, 24, 12, 28] (× 4 bits each)
        reverse_order = (
            (tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]
        ).reshape(8)
        shifts = reverse_order * 4  # [8] — bit shift amounts

        # Broadcast shifts to [BLOCK_K, BLOCK_N]
        # Each packed column unpacks to 8 values
        shifts_2d = tl.broadcast_to(
            shifts[None, :],
            (BLOCK_K * (BLOCK_N // 8), 8)
        )
        shifts_2d = tl.reshape(shifts_2d, (BLOCK_K, BLOCK_N))

        # Main loop over K dimension
        for k_off in range(0, K, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)

            # ──── Load activations [BLOCK_M, BLOCK_K] ────
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

            # ──── Load packed weights [BLOCK_K, BLOCK_N//8] ────
            qw_ptrs = qweight_ptr + offs_k[:, None] * stride_qwk + offs_n_packed[None, :] * stride_qwn
            mask_qw = (offs_k[:, None] < K) & (offs_n_packed[None, :] < N // 8)
            qw = tl.load(qw_ptrs, mask=mask_qw, other=0)

            # ──── Unpack INT4: [BLOCK_K, BLOCK_N//8] → [BLOCK_K, BLOCK_N] ────
            # Interleave 3x to expand each int32 to 8 positions
            qw = tl.interleave(qw, qw)
            qw = tl.interleave(qw, qw)
            qw = tl.interleave(qw, qw)
            # Now qw is [BLOCK_K, BLOCK_N] with duplicated packed values

            # Extract 4-bit values with AWQ reorder
            w_int4 = (qw >> shifts_2d) & 0xF

            # ──── Load zeros for this K-group [1, BLOCK_N//8] ────
            group_idx = (k_off // group_size)
            qz_ptrs = qzeros_ptr + group_idx * stride_qzg + offs_n_packed[None, :] * stride_qzn
            mask_qz = offs_n_packed[None, :] < N // 8
            qz = tl.load(qz_ptrs, mask=mask_qz, other=0)

            # Unpack zeros: [1, BLOCK_N//8] → [1, BLOCK_N]
            qz = tl.interleave(qz, qz)
            qz = tl.interleave(qz, qz)
            qz = tl.interleave(qz, qz)
            # Build shifts for [1, BLOCK_N] shape (no slice needed)
            shifts_zeros = tl.broadcast_to(
                shifts[None, :],
                (BLOCK_N // 8, 8)
            )
            shifts_zeros = tl.reshape(shifts_zeros, (1, BLOCK_N))
            zeros = (qz >> shifts_zeros) & 0xF
            zeros = tl.broadcast_to(zeros, (BLOCK_K, BLOCK_N))

            # ──── Load scales [1, BLOCK_N] ────
            s_ptrs = scales_ptr + group_idx * stride_sg + offs_n[None, :] * stride_sn
            mask_s = offs_n[None, :] < N
            scales = tl.load(s_ptrs, mask=mask_s, other=0.0)
            scales = tl.broadcast_to(scales, (BLOCK_K, BLOCK_N))

            # ──── Dequantize: (int4 - zeros) * scales → FP16 ────
            w_fp16 = (w_int4 - zeros).to(tl.float16) * scales.to(tl.float16)

            # ──── Matmul: x_fp16 @ w_fp16 → accumulate ────
            acc += tl.dot(x_tile, w_fp16)

        # ──── Store output ────
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc.to(tl.float16), mask=mask_out)


# ═══════════════════════════════════════════════════════
# Python API
# ═══════════════════════════════════════════════════════

def awq_gemm_megagemm(
    x: torch.Tensor,           # [..., K] float16
    qweight: torch.Tensor,     # [K, N//8] int32
    scales: torch.Tensor,      # [K//G, N] float16
    qzeros: torch.Tensor,      # [K//G, N//8] int32
    group_size: int = 128,
) -> torch.Tensor:
    """
    Custom AWQ fused dequant+GEMM kernel, optimized for decode.

    Returns: [..., N] float16
    """
    if not _HAS_TRITON or not x.is_cuda:
        return None  # caller uses fallback

    orig_shape = x.shape
    K = orig_shape[-1]
    N = qweight.shape[1] * 8  # 8 INT4 values packed per int32

    # Flatten to 2D
    x_2d = x.reshape(-1, K).contiguous()
    M = x_2d.shape[0]

    # Pad M to BLOCK_M minimum (16)
    pad_m = 0
    if M < 16:
        pad_m = 16 - M
        x_2d = F.pad(x_2d, (0, 0, 0, pad_m))
        M_padded = 16
    else:
        M_padded = M

    # Output
    out = torch.empty((M_padded, N), dtype=x.dtype, device=x.device)

    # Grid
    grid = lambda META: (
        triton.cdiv(M_padded, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    _awq_gemm_kernel[grid](
        x_2d, qweight, qzeros, scales, out,
        M_padded, N, K,
        group_size,
        x_2d.stride(0), x_2d.stride(1),
        qweight.stride(0), qweight.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
    )

    # Remove padding
    if pad_m > 0:
        out = out[:M, :]

    return out.reshape(*orig_shape[:-1], N)


HAS_AWQ_MEGAGEMM = _HAS_TRITON

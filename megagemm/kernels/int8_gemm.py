"""
⚡ INT8 Fused GEMM — Triton Kernel
------------------------------------
Fuses dynamic activation quantization + INT8 matmul + dequantization
into a single Triton kernel using INT8 Tensor Cores.

Instead of:
  x_scale = x.abs().amax() / 127           # kernel 1: absmax
  x_int8 = (x / x_scale).round()           # kernel 2: quantize
  out_i32 = torch._int_mm(x_int8, w_int8)  # kernel 3: matmul
  out = out_i32 * x_scale * w_scale         # kernel 4: dequant

We do:
  out = int8_fused_gemm(x, w_int8, w_scale)
  # 1 kernel: quantize in registers → INT8 TC matmul → dequant

Requires: sm_80+ (A100/L4/H100), Triton with INT8 tl.dot support.
Falls back to torch._int_mm if Triton INT8 dot is not available.

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

__all__ = ['int8_fused_gemm', 'int8_small_m_gemm', 'HAS_INT8_FUSED_GEMM']


# ═══════════════════════════════════════════════════════
# Check if Triton supports INT8 tl.dot on this GPU
# ═══════════════════════════════════════════════════════

def _check_triton_int8_dot() -> bool:
    """Test if tl.dot with INT8 inputs works on current GPU."""
    if not _HAS_TRITON:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability()
        if cap[0] < 8:
            return False

        # Quick compile+run test with a minimal kernel
        @triton.jit
        def _test_kernel(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
            a = tl.load(a_ptr + tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :])
            b = tl.load(b_ptr + tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :])
            c = tl.dot(a, b)
            tl.store(c_ptr + tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :], c)

        a = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device='cuda')
        b = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device='cuda')
        c = torch.empty((32, 32), dtype=torch.int32, device='cuda')
        _test_kernel[(1,)](a, b, c, BLOCK=32)
        return True
    except Exception:
        return False


_TRITON_INT8_CHECKED = False
_TRITON_INT8_SUPPORTED = False


def _get_triton_int8_support() -> bool:
    global _TRITON_INT8_CHECKED, _TRITON_INT8_SUPPORTED
    if not _TRITON_INT8_CHECKED:
        _TRITON_INT8_SUPPORTED = _check_triton_int8_dot()
        _TRITON_INT8_CHECKED = True
    return _TRITON_INT8_SUPPORTED


# ═══════════════════════════════════════════════════════
# Triton Kernel
# ═══════════════════════════════════════════════════════

if _HAS_TRITON:
    @triton.jit
    def _int8_fused_gemm_kernel(
        # Pointers
        x_ptr,          # [M, K] float16 input activations
        w_ptr,          # [K, N] int8 transposed weights (column-major for output)
        w_scale_ptr,    # [N] float16 per-channel weight scales
        out_ptr,        # [M, N] float16 output
        bias_ptr,       # [N] float16 bias (or null)
        # Dims
        M, N, K,
        # Strides
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_om, stride_on,
        # Flags
        HAS_BIAS: tl.constexpr,
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused INT8 GEMM kernel:
        1. Load x tile (FP16) → compute per-tile absmax → quantize to INT8
        2. Load w tile (INT8)
        3. tl.dot(x_int8, w_int8) → INT32 accumulator (on Tensor Cores)
        4. Dequantize: acc * x_scale * w_scale → FP16
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Offsets for this tile
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Accumulator in INT32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        # Per-row activation scale accumulators
        # We'll compute ONE scale per row across all K tiles
        # Strategy: compute absmax across all K tiles first, then quantize+matmul
        # This is a two-pass approach for correctness

        # Pass 1: Compute per-row absmax across all K tiles
        row_amax = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for k_off in range(0, K, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)

            # Load x tile [BLOCK_M, BLOCK_K]
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

            # Update per-row absmax
            tile_amax = tl.max(tl.abs(x_tile), axis=1)  # [BLOCK_M]
            row_amax = tl.maximum(row_amax, tile_amax.to(tl.float32))

        # Compute scale: amax / 127
        row_amax = tl.maximum(row_amax, 1e-12)
        x_scale = row_amax / 127.0  # [BLOCK_M] float32
        inv_x_scale = 127.0 / row_amax  # [BLOCK_M] for faster quant

        # Pass 2: Quantize x tiles and accumulate matmul
        for k_off in range(0, K, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)

            # Reload x tile [BLOCK_M, BLOCK_K]
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.float32)

            # Quantize x to INT8: x_int8 = round(x * inv_scale)
            x_scaled = x_tile * inv_x_scale[:, None]
            # Clamp to [-128, 127] and cast
            x_int8 = tl.maximum(tl.minimum(x_scaled + 0.5, 127.0), -128.0).to(tl.int8)

            # Load w tile [BLOCK_K, BLOCK_N] int8
            w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
            mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            w_tile = tl.load(w_ptrs, mask=mask_w, other=0)

            # INT8 × INT8 → INT32 on Tensor Cores
            acc += tl.dot(x_int8, w_tile)

        # Dequantize: result = acc_int32 * x_scale * w_scale
        # x_scale: [BLOCK_M], w_scale: [BLOCK_N]
        w_scale = tl.load(w_scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

        out = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :]

        # Optional bias
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            out += bias[None, :]

        # Store output
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, out.to(tl.float16), mask=mask_out)

    @triton.jit
    def _int8_small_m_gemm_kernel(
        # Pointers
        x_ptr,          # [M, K] int8 activations
        x_scale_ptr,    # [M, 1] float32 activation scales
        w_ptr,          # [N, K] int8 weights (row-major)
        w_scale_ptr,    # [N] float16 per-channel weight scales
        out_ptr,        # [M, N] float16 output
        bias_ptr,       # [N] float16 bias (or dummy)
        # Dims
        M, N, K,
        # Strides
        stride_xm, stride_xk,
        stride_xsm,
        stride_wn, stride_wk,
        stride_om, stride_on,
        # Flags
        HAS_BIAS: tl.constexpr,
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Small-M INT8 GEMM specialized for decode (M=1..small):
        - no M padding
        - INT8 Tensor Core dot path via tl.dot
        """
        pid_m = tl.program_id(0)  # tile over M
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k_off in range(0, K, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x_vals = tl.load(x_ptrs, mask=mask_x, other=0)

            # Load transposed view [BLOCK_K, BLOCK_N] from row-major [N, K]
            w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
            mask_w = (offs_n[None, :] < N) & (offs_k[:, None] < K)
            w_vals = tl.load(w_ptrs, mask=mask_w, other=0)

            acc += tl.dot(x_vals, w_vals)

        x_scale = tl.load(x_scale_ptr + offs_m * stride_xsm, mask=offs_m < M, other=0.0).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        out = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :]

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            out += bias[None, :]

        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, out.to(tl.float16), mask=mask_out)


# ═══════════════════════════════════════════════════════
# Python API
# ═══════════════════════════════════════════════════════

def int8_fused_gemm(
    x: torch.Tensor,           # [*, K] float16 activations
    weight_int8: torch.Tensor,  # [N, K] int8 weights (row-major)
    w_scale: torch.Tensor,      # [N] float16 per-channel scale
    bias: torch.Tensor = None,  # [N] float16 optional bias
) -> torch.Tensor:
    """
    Fused INT8 GEMM: quantize activations + INT8 matmul + dequant.

    Equivalent to:
        x_scale = x.abs().amax(-1, keepdim=True) / 127
        x_int8 = (x / x_scale).round().to(int8)
        out = (x_int8 @ weight.T).float() * x_scale * w_scale

    But done in a single Triton kernel with zero intermediate tensors.

    Args:
        x: Input activations [..., K] in FP16
        weight_int8: INT8 weights [N, K] (output_features × input_features)
        w_scale: Per-channel weight scales [N]
        bias: Optional bias [N]

    Returns:
        Output tensor [..., N] in FP16
    """
    if not (_HAS_TRITON and x.is_cuda and _get_triton_int8_support()):
        return None  # Caller should use fallback

    orig_shape = x.shape
    K = orig_shape[-1]
    N = weight_int8.shape[0]

    # Flatten to 2D
    x_2d = x.reshape(-1, K).contiguous()
    M = x_2d.shape[0]

    # Pad M to multiple of BLOCK_M for Tensor Core alignment
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32

    pad_m = 0
    if M % BLOCK_M != 0:
        pad_m = BLOCK_M - (M % BLOCK_M)
        x_2d = F.pad(x_2d, (0, 0, 0, pad_m))
    M_padded = M + pad_m

    # Transpose weight to [K, N] for the kernel (column-major access)
    w_t = weight_int8.t().contiguous()  # [K, N]

    # Output
    out = torch.empty((M_padded, N), dtype=x.dtype, device=x.device)

    # Grid
    grid = (triton.cdiv(M_padded, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _int8_fused_gemm_kernel[grid](
        x_2d, w_t, w_scale, out,
        bias if bias is not None else x_2d,  # dummy ptr if no bias
        M_padded, N, K,
        x_2d.stride(0), x_2d.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=(bias is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Remove padding and reshape
    if pad_m > 0:
        out = out[:M, :]

    return out.reshape(*orig_shape[:-1], N)


def int8_small_m_gemm(
    x_int8: torch.Tensor,      # [M, K] int8 activations (already quantized)
    x_scale: torch.Tensor,     # [M, 1] float32 activation scales
    weight_int8: torch.Tensor, # [N, K] int8 weights
    w_scale: torch.Tensor,     # [N] float16 per-channel scales
    bias: torch.Tensor = None, # [N] optional bias
) -> torch.Tensor:
    """
    Small-M INT8 GEMM specialized for decode, avoiding M-padding overhead.
    Returns None when Triton path is unavailable so caller can fallback.
    """
    if not (_HAS_TRITON and x_int8.is_cuda and _get_triton_int8_support()):
        return None
    if x_int8.dtype != torch.int8 or weight_int8.dtype != torch.int8:
        return None
    if x_int8.dim() != 2 or weight_int8.dim() != 2:
        return None
    if x_int8.shape[1] != weight_int8.shape[1]:
        return None

    m, k = x_int8.shape
    n = weight_int8.shape[0]
    if m <= 0 or k <= 0 or n <= 0:
        return None

    x_q = x_int8.contiguous()
    xs = x_scale.contiguous()
    w_q = weight_int8.contiguous()
    ws = w_scale.contiguous()

    out = torch.empty((m, n), dtype=torch.float16, device=x_q.device)

    block_m = 16
    block_n = 64
    block_k = 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))

    _int8_small_m_gemm_kernel[grid](
        x_q,
        xs,
        w_q,
        ws,
        out,
        bias if bias is not None else ws,  # dummy ptr if no bias
        m,
        n,
        k,
        x_q.stride(0),
        x_q.stride(1),
        xs.stride(0),
        w_q.stride(0),
        w_q.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=(bias is not None),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    return out


HAS_INT8_FUSED_GEMM = _HAS_TRITON

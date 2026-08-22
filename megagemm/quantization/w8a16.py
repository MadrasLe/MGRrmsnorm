"""
⚡ W8A16 / W8A8 Quantized Linear — INT8 Quantization
------------------------------------------------------
On-the-fly FP16 → INT8 quantization with per-channel scaling.

Format:
  - weight_int8: [out_features, in_features] as int8
  - scale:       [out_features] as float16

Two modes:
  - W8A16 (fallback): Dequantize INT8→FP16, then FP16 matmul. Slow but universal.
  - W8A8  (fast):     Dynamically quantize activations to INT8, use torch._int_mm
                      for native INT8 Tensor Core GEMM. ~2-3x faster on A100/L4.

The mode is auto-detected based on hardware support (sm_80+, PyTorch >= 2.1).

Author: Gabriel Yogi
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Int8Linear', 'quantize_to_int8', 'INT8_AVAILABLE', 'INT8_TENSORCORE']

INT8_AVAILABLE = True  # int8 always available


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# Speed/VRAM tradeoff knobs for W8A16 fallback mode (non-TensorCore path).
# - CACHE: dequantize once and reuse (fastest, higher steady VRAM).
# - REUSE: keep a reusable FP16 buffer to avoid per-token allocations.
_USE_INT8_DEQUANT_CACHE = _env_enabled("MEGAGEMM_INT8_DEQUANT_CACHE", default=False)
_USE_INT8_DEQUANT_REUSE = _env_enabled("MEGAGEMM_INT8_DEQUANT_REUSE", default=True)
_USE_INT8_TRITON_FUSED = _env_enabled("MEGAGEMM_INT8_TRITON_FUSED", default=False)
_USE_INT8_WEIGHT_T_CACHE = _env_enabled("MEGAGEMM_INT8_WEIGHT_T_CACHE", default=False)
_USE_INT8_SMALL_M_KERNEL = _env_enabled("MEGAGEMM_INT8_SMALL_M_KERNEL", default=True)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


_INT8_TC_MIN_M = max(1, _env_int("MEGAGEMM_INT8_TC_MIN_M", 16))
_INT8_TC_ALIGN_M = max(1, _env_int("MEGAGEMM_INT8_TC_ALIGN_M", 16))
_INT8_SMALL_M_MAX = max(1, _env_int("MEGAGEMM_INT8_SMALL_M_MAX", 4))

# Try Triton fused INT8 GEMM (highest priority)
try:
    from ..kernels.int8_gemm import (
        int8_fused_gemm,
        int8_small_m_gemm,
        _get_triton_int8_support,
    )
    _HAS_TRITON_INT8 = True
except Exception:
    _HAS_TRITON_INT8 = False
    int8_small_m_gemm = None

# Detect INT8 Tensor Core support (torch._int_mm on sm_80+)
def _check_int_mm_support() -> bool:
    """Check if torch._int_mm is available and working on current GPU."""
    if not hasattr(torch, '_int_mm'):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability()
        if cap[0] < 8:  # Need sm_80+ (Ampere/Ada/Hopper)
            return False
        # Quick functional test (M must be > 16)
        a = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device='cuda')
        b = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device='cuda')
        torch._int_mm(a, b)
        return True
    except Exception:
        return False

INT8_TENSORCORE = None  # Lazy init (CUDA may not be ready at import time)


def _get_int8_tensorcore() -> bool:
    """Lazy-init INT8 Tensor Core detection."""
    global INT8_TENSORCORE
    if INT8_TENSORCORE is None:
        INT8_TENSORCORE = _check_int_mm_support()
    return INT8_TENSORCORE


def quantize_to_int8(weight_fp16: torch.Tensor):
    """
    Quantize FP16 weight tensor to INT8 symmetric per-channel.

    Symmetric: zero-point is always 0, range is [-127, 127].
    Per-channel: each output row gets its own scale factor.

    Args:
        weight_fp16: [out_features, in_features] FP16 tensor

    Returns:
        (weight_int8, scale) where:
        - weight_int8: [out_features, in_features] int8
        - scale: [out_features] float16
    """
    # Per-channel absmax
    amax = weight_fp16.abs().amax(dim=1).clamp(min=1e-12)  # [out]

    # Scale to fit in [-127, 127]
    scale = amax / 127.0  # [out]

    # Quantize: divide by scale, round, clamp, cast
    w_scaled = weight_fp16 / scale.unsqueeze(1)
    w_int8 = w_scaled.round().clamp(-128, 127).to(torch.int8)

    return w_int8, scale.to(torch.float16)


def _dynamic_quantize_activation(x: torch.Tensor):
    """
    Dynamically quantize activations to INT8 per-token symmetric.

    Args:
        x: [..., K] float16/bfloat16 tensor

    Returns:
        (x_int8, x_scale) where:
        - x_int8: [..., K] int8 (same shape)
        - x_scale: [..., 1] float32 (per-token scale)
    """
    # Per-token absmax (last dim = features)
    x_amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12).float()
    x_scale = x_amax / 127.0

    # Quantize
    x_int8 = (x.float() / x_scale).round().clamp(-128, 127).to(torch.int8)

    return x_int8, x_scale


class Int8Linear(nn.Module):
    """
    Drop-in nn.Linear replacement using INT8 weights.

    Stores weights in int8 (8-bit) with per-channel scale.

    Forward path (auto-selected):
      - W8A8 (fast):     x_int8 @ w_int8 via INT8 Tensor Cores (A100/L4/H100)
      - W8A16 (fallback): dequant weights to FP16, then FP16 matmul

    2x compression vs FP16 with near-lossless quality.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Placeholder buffers — filled by from_linear()
        self.register_buffer('weight_int8',
            torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('scale',
            torch.zeros(out_features, dtype=torch.float16))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        # Runtime dequant state for W8A16 fallback.
        # Not registered as buffers on purpose (ephemeral perf cache).
        self._cached_weight = None
        self._cached_weight_key = None
        self._cached_weight_versions = (-1, -1)
        self._reused_weight = None
        self._reused_weight_key = None
        self._cached_weight_t = None
        self._cached_weight_t_key = None
        self._cached_weight_t_version = -1
        self._tc_pad_x_int8 = None
        self._tc_pad_x_int8_key = None

    def _clear_runtime_caches(self):
        self._cached_weight = None
        self._cached_weight_key = None
        self._cached_weight_versions = (-1, -1)
        self._reused_weight = None
        self._reused_weight_key = None
        self._cached_weight_t = None
        self._cached_weight_t_key = None
        self._cached_weight_t_version = -1
        self._tc_pad_x_int8 = None
        self._tc_pad_x_int8_key = None

    def _apply(self, fn):
        """
        Override to prevent model.to(dtype=float16) from converting INT8 weights.

        PyTorch's .to(dtype) would convert int8 buffers, destroying compression.
        We intercept: move INT8 weight to the correct device but keep its dtype.
        """
        # Save INT8 weight before parent _apply touches it
        int8_weight = self.weight_int8.data.clone()

        # Let parent handle scale, bias, and general module bookkeeping
        super()._apply(fn)

        # Restore INT8 weight on the new device (scale was moved by parent)
        target_device = self.scale.device
        self.weight_int8 = int8_weight.to(device=target_device)
        self._clear_runtime_caches()

        return self

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> 'Int8Linear':
        """
        Create Int8Linear from an existing nn.Linear module.
        Quantizes weights on-the-fly.
        """
        has_bias = linear.bias is not None
        int8_layer = cls(linear.in_features, linear.out_features, bias=has_bias)

        # Quantize weights
        w_fp16 = linear.weight.data.to(torch.float16)
        w_int8, scale = quantize_to_int8(w_fp16)

        # Register as proper buffers
        int8_layer.register_buffer('weight_int8', w_int8)
        int8_layer.register_buffer('scale', scale)

        if has_bias:
            int8_layer.bias = nn.Parameter(linear.bias.data.to(torch.float16))

        return int8_layer

    def _get_weight_t(self) -> torch.Tensor:
        """Get contiguous transposed weight for _int_mm with optional caching."""
        if not _USE_INT8_WEIGHT_T_CACHE:
            return self.weight_int8.t().contiguous()

        device = self.weight_int8.device
        dev_idx = int(device.index) if device.index is not None else -1
        key = (device.type, dev_idx)
        version = int(self.weight_int8._version)

        if (
            self._cached_weight_t is not None
            and self._cached_weight_t_key == key
            and self._cached_weight_t_version == version
        ):
            return self._cached_weight_t

        wt = self.weight_int8.t().contiguous()
        self._cached_weight_t = wt
        self._cached_weight_t_key = key
        self._cached_weight_t_version = version
        return wt

    def _get_tc_pad_x_int8(self, rows: int, cols: int, device: torch.device) -> torch.Tensor:
        """Reusable INT8 activation pad buffer for _int_mm small-M calls."""
        dev_idx = int(device.index) if device.index is not None else -1
        key = (rows, cols, device.type, dev_idx)
        if self._tc_pad_x_int8 is None or self._tc_pad_x_int8_key != key:
            self._tc_pad_x_int8 = torch.empty(
                (rows, cols),
                dtype=torch.int8,
                device=device,
            )
            self._tc_pad_x_int8_key = key
        return self._tc_pad_x_int8

    def _dequant_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Return dequantized weight for W8A16 fallback with optional caching/reuse."""
        device = self.weight_int8.device
        dev_idx = int(device.index) if device.index is not None else -1
        key = (str(dtype), device.type, dev_idx)
        versions = (int(self.weight_int8._version), int(self.scale._version))

        if _USE_INT8_DEQUANT_CACHE:
            if (
                self._cached_weight is not None
                and self._cached_weight_key == key
                and self._cached_weight_versions == versions
            ):
                return self._cached_weight

            w = self.weight_int8.to(dtype) * self.scale.unsqueeze(1).to(dtype)
            self._cached_weight = w.contiguous()
            self._cached_weight_key = key
            self._cached_weight_versions = versions
            return self._cached_weight

        if _USE_INT8_DEQUANT_REUSE:
            if (
                self._reused_weight is None
                or self._reused_weight_key != key
            ):
                self._reused_weight = torch.empty(
                    self.weight_int8.shape,
                    device=device,
                    dtype=dtype,
                )
                self._reused_weight_key = key
            # Keep compute explicit in-place to avoid temporary large allocations.
            self._reused_weight.copy_(self.weight_int8)
            self._reused_weight.mul_(self.scale.unsqueeze(1).to(dtype))
            return self._reused_weight

        return self.weight_int8.to(dtype) * self.scale.unsqueeze(1).to(dtype)

    @property
    def weight(self) -> torch.Tensor:
        """
        Compatibility shim for code paths that expect nn.Linear.weight.

        Returns a dequantized FP16 tensor and should be treated as fallback.
        """
        return self._dequant_weight(torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Priority 1: Triton fused INT8 GEMM (single kernel quantize+matmul+dequant).
        if (
            _USE_INT8_TRITON_FUSED
            and _HAS_TRITON_INT8
            and x.is_cuda
            and not torch.is_grad_enabled()
        ):
            try:
                out = int8_fused_gemm(
                    x,
                    self.weight_int8,
                    self.scale,
                    self.bias,
                )
                if out is not None:
                    return out if out.dtype == x.dtype else out.to(x.dtype)
            except Exception:
                pass

        # Priority 2: torch._int_mm Tensor Core path.
        if _get_int8_tensorcore():
            return self._forward_int8_tensorcore(x)

        # Priority 3: W8A16 fallback (with optional dequant cache/reuse).
        return self._forward_dequant(x)

    def _forward_dequant(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback: dequantize INT8 → FP16, then FP16 matmul."""
        w = self._dequant_weight(x.dtype)
        return F.linear(x, w, self.bias)

    def _forward_int8_tensorcore(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast path: INT8 Tensor Core GEMM via torch._int_mm.

        1. Dynamically quantize activations x -> x_int8 (per-token)
        2. INT8 matmul: x_int8 @ weight_int8.T -> INT32 result
        3. Dequantize: result * x_scale * w_scale -> FP16

        This uses native INT8 Tensor Cores (~2x faster than FP16 TC).
        """
        orig_dtype = x.dtype
        orig_shape = x.shape  # [..., K]

        # Flatten to 2D for _int_mm: [M, K]
        x_2d = x.reshape(-1, x.shape[-1])
        m = x_2d.shape[0]

        # Step 1: dynamic activation quantization on REAL rows only.
        # Avoid quantizing padded rows in decode (M=1), which was pure overhead.
        x_int8, x_scale = _dynamic_quantize_activation(x_2d)  # [m, K], [m, 1]

        # Small-M Triton path (decode-focused): avoid _int_mm row padding entirely.
        if (
            _USE_INT8_SMALL_M_KERNEL
            and _HAS_TRITON_INT8
            and int8_small_m_gemm is not None
            and x_int8.is_cuda
            and m <= _INT8_SMALL_M_MAX
            and not torch.is_grad_enabled()
        ):
            try:
                out_small = int8_small_m_gemm(
                    x_int8,
                    x_scale,
                    self.weight_int8,
                    self.scale,
                    self.bias,
                )
                if out_small is not None:
                    out_small = out_small.reshape(*orig_shape[:-1], -1)
                    return out_small if out_small.dtype == orig_dtype else out_small.to(orig_dtype)
            except Exception:
                pass

        # Step 2: INT8 x INT8 -> INT32 on Tensor Cores
        w_t = self._get_weight_t()  # [K, N] int8, cached

        def _align_rows(rows: int, align: int) -> int:
            if align <= 1:
                return rows
            rem = rows % align
            return rows if rem == 0 else rows + (align - rem)

        def _run_int_mm(target_rows: int):
            pad_rows = target_rows - m
            if pad_rows > 0:
                x_mm = self._get_tc_pad_x_int8(target_rows, x_int8.shape[1], x_int8.device)
                x_mm[:m].copy_(x_int8)
                x_mm[m:].zero_()
            else:
                x_mm = x_int8
            out_i32 = torch._int_mm(x_mm, w_t)
            if pad_rows > 0:
                out_i32 = out_i32[:m, :]
            return out_i32

        # Aggressive small-M target first (decode path); retry conservatively if needed.
        target_rows = _align_rows(max(m, _INT8_TC_MIN_M), _INT8_TC_ALIGN_M)
        try:
            out_int32 = _run_int_mm(target_rows)
        except RuntimeError:
            safe_rows = _align_rows(max(m, 32), 32)
            if safe_rows == target_rows:
                raise
            out_int32 = _run_int_mm(safe_rows)

        # Step 3: dequantize -> float
        # x_scale: [m, 1], w_scale: [1, N]
        w_scale = self.scale.float().unsqueeze(0)
        out = out_int32.float() * x_scale * w_scale

        # Reshape back + cast
        out = out.reshape(*orig_shape[:-1], -1).to(orig_dtype)

        if self.bias is not None:
            out = out + self.bias

        return out

    @property
    def weight_memory_mb(self):
        """Actual memory used by INT8 weights + scales."""
        total = self.weight_int8.nelement() * 1   # INT8 = 1 byte
        total += self.scale.nelement() * 2         # FP16 scale = 2 bytes
        return total / (1024 ** 2)

    @property
    def fp16_equivalent_mb(self):
        """Memory that FP16 weights would use."""
        return self.in_features * self.out_features * 2 / (1024 ** 2)

    def __repr__(self):
        ratio = self.fp16_equivalent_mb / self.weight_memory_mb if self.weight_memory_mb > 0 else 0
        if (
            _USE_INT8_TRITON_FUSED
            and _HAS_TRITON_INT8
            and _get_triton_int8_support()
        ):
            mode = "W8A8-triton-fused"
        elif (
            _USE_INT8_SMALL_M_KERNEL
            and _HAS_TRITON_INT8
            and _get_triton_int8_support()
        ):
            mode = "W8A8-smallM+TC"
        elif _get_int8_tensorcore():
            mode = "W8A8-TC"
        elif _USE_INT8_DEQUANT_CACHE:
            mode = "W8A16-cache"
        elif _USE_INT8_DEQUANT_REUSE:
            mode = "W8A16-reuse"
        else:
            mode = "W8A16-dequant"
        return (
            f"Int8Linear(in={self.in_features}, out={self.out_features}, "
            f"{ratio:.1f}x compression, {mode})"
        )

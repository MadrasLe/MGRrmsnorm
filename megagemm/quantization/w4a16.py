"""
⚡ W4A16 Quantized Linear — AWQ INT4 Weight Quantization
---------------------------------------------------------
Fused INT4 dequant+GEMM using AutoAWQ's optimized kernels.

Kernel priority:
  1. awq_ext (CUDA kernel) — fastest, from pip install autoawq
  2. Triton kernel         — good, from AutoAWQ's triton module
  3. PyTorch fallback      — slow but always works

AWQ format (per-group, packed along output dim):
  - qweight: [in_features, out_features // 8] INT32
  - qzeros:  [in_features // group_size, out_features // 8] INT32
  - scales:  [in_features // group_size, out_features] FP16

Author: Gabriel Yogi
"""

import torch
import torch.nn as nn

__all__ = ['QuantizedLinear', 'W4A16_AVAILABLE']

W4A16_AVAILABLE = True

PACK_FACTOR = 8
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

# ─────────────────────────────────────────────────────────
# Kernel detection: try awq_ext > triton > pure PyTorch
# ─────────────────────────────────────────────────────────
_AWQ_EXT = None
_AWQ_TRITON_GEMM = None
_AWQ_TRITON_DEQUANT = None
_BACKEND = "pytorch"  # will be updated below

try:
    import awq_ext
    _AWQ_EXT = awq_ext
    _BACKEND = "cuda"
except ImportError:
    pass

if _AWQ_EXT is None:
    try:
        from awq.modules.triton.gemm import awq_gemm_triton, awq_dequantize_triton
        _AWQ_TRITON_GEMM = awq_gemm_triton
        _AWQ_TRITON_DEQUANT = awq_dequantize_triton
        _BACKEND = "triton"
    except ImportError:
        pass

if _BACKEND == "pytorch":
    # Last resort: try importing triton directly for potential torch.compile
    try:
        import triton
        _BACKEND = "pytorch"  # still pytorch, but triton available for compile
    except ImportError:
        pass

_BACKEND_ANNOUNCED = False


def _announce_backend_once():
    """Avoid advertising an AWQ fallback when another INT4 backend is in use."""
    global _BACKEND_ANNOUNCED
    if _BACKEND_ANNOUNCED:
        return
    print(f"  AWQ backend: {_BACKEND}" +
          (" (fused CUDA kernel)" if _BACKEND == "cuda" else
           " (Triton kernel)" if _BACKEND == "triton" else
           " (PyTorch fallback — install autoawq for 10-50x speedup)"))
    _BACKEND_ANNOUNCED = True


# ─────────────────────────────────────────────────────────
# PyTorch fallback dequantization (slow but always works)
# ─────────────────────────────────────────────────────────
_CACHED_SHIFTS = {}
_CACHED_REVERSE_IDX = {}

def _get_shifts(device):
    if device not in _CACHED_SHIFTS:
        _CACHED_SHIFTS[device] = torch.tensor(
            [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device
        )
    return _CACHED_SHIFTS[device]

def _get_reverse_idx(N, device):
    key = (device, N)
    if key not in _CACHED_REVERSE_IDX:
        idx = torch.arange(N, dtype=torch.int32, device=device)
        _CACHED_REVERSE_IDX[key] = idx.view(-1, PACK_FACTOR)[:, AWQ_REVERSE_ORDER].reshape(-1)
    return _CACHED_REVERSE_IDX[key]

def _dequantize_pytorch(qweight, scales, qzeros, group_size):
    """PyTorch-only dequant. Slow but correct. Returns [K, N] FP16."""
    K, N_packed = qweight.shape
    N = N_packed * PACK_FACTOR
    device = qweight.device
    shifts = _get_shifts(device)
    reverse_idx = _get_reverse_idx(N, device)

    iweight = ((qweight.unsqueeze(-1) >> shifts) & 0xF).reshape(K, N)
    izeros = ((qzeros.unsqueeze(-1) >> shifts) & 0xF).reshape(-1, N)

    iweight = iweight[:, reverse_idx] & 0xF
    izeros = izeros[:, reverse_idx] & 0xF

    scales_exp = scales.repeat_interleave(group_size, dim=0)
    izeros_exp = izeros.repeat_interleave(group_size, dim=0)

    return (iweight.to(torch.float16) - izeros_exp.to(torch.float16)) * scales_exp


# ─────────────────────────────────────────────────────────
# Unified forward: picks best available kernel
# ─────────────────────────────────────────────────────────
def _awq_forward(x, qweight, scales, qzeros, group_size, bias, out_features, transposed=False):
    """
    Runs AWQ quantized linear using the best available backend.

    If transposed=True, qweight is [N//8, K] (our fast decode layout).
    We transpose back to [K, N//8] for AutoAWQ kernels (only during prefill).
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])  # [M, K]
    M, K = x_2d.shape
    x_2d = x_2d.to(torch.float16)

    # Get [K, N//8] layout for AutoAWQ kernels
    qw = qweight.t().contiguous() if transposed else qweight

    if _AWQ_EXT is not None:
        if M * K >= 1024:
            w_fp16 = _AWQ_EXT.dequantize_weights_cuda(
                qw, scales, qzeros, 0, 0, 0, False
            )
            out = torch.matmul(x_2d, w_fp16)
        else:
            out = _AWQ_EXT.gemm_forward_cuda(
                x_2d, qw, scales, qzeros, 8
            )

    elif _AWQ_TRITON_GEMM is not None:
        if M * K >= 1024:
            w_fp16 = _AWQ_TRITON_DEQUANT(qw, scales, qzeros)
            out = torch.matmul(x_2d, w_fp16.to(torch.float16))
        else:
            out = _AWQ_TRITON_GEMM(
                x_2d, qw, scales, qzeros, split_k_iters=8
            )

    else:
        w_fp16 = _dequantize_pytorch(qw, scales, qzeros, group_size)
        out = torch.matmul(x_2d, w_fp16)

    if bias is not None:
        out = out + bias

    return out.reshape(*orig_shape[:-1], out_features)


# ─────────────────────────────────────────────────────────
# QuantizedLinear module
# ─────────────────────────────────────────────────────────
class QuantizedLinear(nn.Module):
    """
    Drop-in nn.Linear replacement using AWQ INT4 weights.

    Uses best available kernel automatically:
      awq_ext (CUDA) > Triton > PyTorch fallback
    """

    def __init__(self, in_features, out_features, group_size=128, bias=False):
        super().__init__()
        _announce_backend_once()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self._transposed = False  # Set True after transpose for decode

        assert out_features % PACK_FACTOR == 0
        assert in_features % group_size == 0

        num_groups = in_features // group_size

        self.register_buffer('qweight',
            torch.zeros(in_features, out_features // PACK_FACTOR, dtype=torch.int32))
        self.register_buffer('scales',
            torch.zeros(num_groups, out_features, dtype=torch.float16))
        self.register_buffer('qzeros',
            torch.zeros(num_groups, out_features // PACK_FACTOR, dtype=torch.int32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def transpose_for_decode(self):
        """Transpose qweight [K, N//8] → [N//8, K] for coalesced decode. One-time cost."""
        if not self._transposed:
            self.qweight.data = self.qweight.data.t().contiguous()
            self._transposed = True

    @property
    def weight_memory_mb(self):
        total = self.qweight.nelement() * 4
        total += self.scales.nelement() * 2
        total += self.qzeros.nelement() * 4
        return total / (1024 ** 2)

    @property
    def fp16_equivalent_mb(self):
        return self.in_features * self.out_features * 2 / (1024 ** 2)

    def forward(self, x):
        return _awq_forward(
            x, self.qweight, self.scales, self.qzeros,
            self.group_size, self.bias, self.out_features,
            transposed=self._transposed,
        )

    def __repr__(self):
        ratio = self.fp16_equivalent_mb / self.weight_memory_mb if self.weight_memory_mb > 0 else 0
        return (
            f"QuantizedLinear(in={self.in_features}, out={self.out_features}, "
            f"group={self.group_size}, {ratio:.1f}x, backend={_BACKEND})"
        )

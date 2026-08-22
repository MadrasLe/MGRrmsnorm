"""
RMSNorm CUDA Module
-------------------
High-performance RMSNorm with FP32/FP16/BF16 support.
"""

import os

import torch
import torch.nn as nn

# Try to import the compiled CUDA ops
try:
    import rmsnorm_cuda_ops
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False
if os.environ.get("MEGAGEMM_DISABLE_CUDA_RMSNORM", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    _CUDA_AVAILABLE = False
_CUDA_RMSNORM_FAILURE_REPORTED = False
_ALLOW_UNTESTED_CUDA_RMSNORM_ARCH = os.environ.get(
    "MEGAGEMM_ALLOW_UNTESTED_CUDA_RMSNORM_ARCH", ""
).strip().lower() in {"1", "true", "yes", "on"}


def _torch_rmsnorm(input: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    variance = input.float().pow(2).mean(-1, keepdim=True)
    return (input * torch.rsqrt(variance + epsilon)).type_as(input) * weight


def _disable_cuda_rmsnorm_after_failure(exc: Exception) -> None:
    """Disable the extension after a launch failure so fallback stays quiet."""
    global _CUDA_AVAILABLE, _CUDA_RMSNORM_FAILURE_REPORTED
    _CUDA_AVAILABLE = False
    if not _CUDA_RMSNORM_FAILURE_REPORTED:
        _CUDA_RMSNORM_FAILURE_REPORTED = True
        print(
            "[MegaGemm][warn] CUDA RMSNorm extension failed; "
            f"falling back to PyTorch/Triton RMSNorm ({exc})"
        )


def can_use_cuda_rmsnorm_for(input: torch.Tensor, offset: bool = False) -> bool:
    if not _CUDA_AVAILABLE:
        return False
    if offset or not input.is_cuda:
        return False
    if input.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        return False
    if input.dtype == torch.bfloat16:
        try:
            major, _ = torch.cuda.get_device_capability(input.device)
        except Exception:
            return False
        if major < 8:
            return False
        if major >= 10 and not _ALLOW_UNTESTED_CUDA_RMSNORM_ARCH:
            return False
    return True


class RMSNormFunction(torch.autograd.Function):
    """Autograd function for RMSNorm CUDA kernel."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-5):
        if not can_use_cuda_rmsnorm_for(input):
            raise RuntimeError(
                "CUDA RMSNorm is unavailable for this tensor/device; "
                "SM75/T4 is supported through the FP16 path, while BF16 requires SM80+."
            )

        # Flatten input for kernel (expects 2D). Some decode paths hand us
        # non-contiguous head views, so reshape here instead of view.
        orig_shape = input.shape
        input_2d = input.reshape(-1, input.size(-1)).contiguous()

        output, inv_rms = rmsnorm_cuda_ops.rmsnorm_forward(input_2d, weight, epsilon)

        ctx.save_for_backward(input_2d, weight, inv_rms)
        ctx.epsilon = epsilon
        ctx.orig_shape = orig_shape

        return output.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_2d, weight, inv_rms = ctx.saved_tensors

        grad_output_2d = grad_output.contiguous().view(-1, grad_output.size(-1))

        grad_input, grad_weight = rmsnorm_cuda_ops.rmsnorm_backward(
            grad_output_2d, input_2d, weight, inv_rms
        )

        return grad_input.view(ctx.orig_shape), grad_weight, None


def rmsnorm_forward(input: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Inference fast path (no autograd wrapper).
    """
    if not can_use_cuda_rmsnorm_for(input):
        return _torch_rmsnorm(input, weight, epsilon)

    orig_shape = input.shape
    input_2d = input.reshape(-1, input.size(-1)).contiguous()
    try:
        output, _ = rmsnorm_cuda_ops.rmsnorm_forward(input_2d, weight, epsilon)
        return output.reshape(orig_shape)
    except Exception as exc:
        _disable_cuda_rmsnorm_after_failure(exc)
        return _torch_rmsnorm(input, weight, epsilon)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization - CUDA Accelerated.

    Drop-in replacement for torch.nn.RMSNorm with ~67% speedup.
    Supports FP32, FP16, and BF16 automatically based on input dtype.

    Args:
        hidden_size (int): The size of the last dimension of the input.
        epsilon (float): Small constant for numerical stability. Default: 1e-5.

    Example:
        >>> norm = RMSNorm(4096).cuda()
        >>> x = torch.randn(32, 128, 4096, device='cuda')
        >>> y = norm(x)  # [32, 128, 4096]
    """

    def __init__(self, hidden_size: int, epsilon: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and can_use_cuda_rmsnorm_for(input):
            try:
                return RMSNormFunction.apply(input, self.weight, self.epsilon)
            except Exception as exc:
                _disable_cuda_rmsnorm_after_failure(exc)
        return rmsnorm_forward(input, self.weight, self.epsilon)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.epsilon}"

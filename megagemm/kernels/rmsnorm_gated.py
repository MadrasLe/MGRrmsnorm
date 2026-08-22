"""
Fused RMSNorm + SiLU gate for Qwen 3.5 linear attention.

Fast path targets the actual Qwen usage:
  x    : [..., hidden]
  gate : [..., hidden]
  out  = rmsnorm(x) * weight * silu(gate)

Falls back to PyTorch when Triton is unavailable or the layout is not a
cheap row-major launch.
"""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

__all__ = ["rmsnorm_gated", "HAS_RMSNORM_GATED"]


def _pytorch_rmsnorm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    orig_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return (x_norm * weight * torch.nn.functional.silu(gate)).to(orig_dtype)


if _HAS_TRITON:
    @triton.jit
    def _rmsnorm_gated_kernel(
        x_ptr,
        gate_ptr,
        weight_ptr,
        out_ptr,
        n_cols,
        stride_row,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_off = row * stride_row
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        x = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        inv_rms = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
        out = x * inv_rms * weight * (gate * tl.sigmoid(gate))
        tl.store(out_ptr + row_off + cols, out, mask=mask)


def rmsnorm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    if x.shape != gate.shape:
        raise ValueError("rmsnorm_gated expects x and gate to have the same shape")
    if x.shape[-1] != weight.numel():
        raise ValueError("rmsnorm_gated weight size must match the last dimension")

    if not (
        _HAS_TRITON
        and x.is_cuda
        and gate.is_cuda
        and weight.is_cuda
        and x.stride(-1) == 1
        and gate.stride(-1) == 1
    ):
        return _pytorch_rmsnorm_gated(x, gate, weight, eps)

    n_cols = x.shape[-1]
    if n_cols == 0 or n_cols > 4096:
        return _pytorch_rmsnorm_gated(x, gate, weight, eps)

    x_2d = x.reshape(-1, n_cols)
    gate_2d = gate.reshape(-1, n_cols)
    out = torch.empty_like(x_2d)

    block_size = triton.next_power_of_2(n_cols)
    num_warps = 4 if block_size <= 256 else 8

    _rmsnorm_gated_kernel[(x_2d.shape[0],)](
        x_2d,
        gate_2d,
        weight,
        out,
        n_cols,
        x_2d.stride(0),
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.reshape_as(x)


HAS_RMSNORM_GATED = _HAS_TRITON

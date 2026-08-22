"""
Standalone RMSNorm Triton kernel with optional Gemma/Qwen offset.

Compatible with all GPUs (T4, L4, A100, H100).
Uses tl.minimum to clamp column indices within tensor bounds,
preventing out-of-bounds pointer generation even when BLOCK_SIZE > n_cols.

This complements the existing CUDA extension, which only handles the
non-offset case.
"""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

__all__ = [
    "rmsnorm_triton",
    "rmsnorm_triton_add_dual",
    "rmsnorm_triton_add_dual_router",
    "rmsnorm_triton_attn_residual_dual",
    "rmsnorm_triton_attn_residual_router_bridge",
    "rmsnorm_triton_attn_residual_router_bridge_single",
    "rmsnorm_triton_dual",
    "rmsnorm_triton_add",
    "rmsnorm_triton_pair_add_final",
    "rmsnorm_triton_pair_add_final_residual",
    "rmsnorm_triton_weighted_scaled_no_weight_dual",
    "rmsnorm_triton_no_weight",
    "rmsnorm_triton_scaled_no_weight",
    "HAS_TRITON_RMSNORM",
]


def _pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, offset: bool) -> torch.Tensor:
    orig_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    scale = weight + 1.0 if offset else weight
    return (x_norm * scale).to(orig_dtype)


def _pytorch_rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(orig_dtype)


if _HAS_TRITON:
    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        n_cols,
        x_stride_row,
        out_stride_row,
        eps,
        OFFSET: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_row_off = row * x_stride_row
        out_row_off = row * out_stride_row
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        # Clamp indices to prevent OOB pointer generation on last row.
        # Even with mask=False, some Triton backends still compute the
        # address and fault if it points past allocated memory.
        safe_cols = tl.minimum(cols, n_cols - 1)

        x = tl.load(x_ptr + x_row_off + safe_cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        inv_rms = 1.0 / tl.sqrt(var + eps)

        if HAS_WEIGHT:
            weight = tl.load(
                weight_ptr + safe_cols, mask=mask, other=0.0
            ).to(tl.float32)
            if OFFSET:
                out = x * inv_rms * (weight + 1.0)
            else:
                out = x * inv_rms * weight
        else:
            out = x * inv_rms
        tl.store(out_ptr + out_row_off + safe_cols, out, mask=mask)

    @triton.jit
    def _rmsnorm_scaled_no_weight_kernel(
        x_ptr,
        scale_ptr,
        out_ptr,
        n_cols,
        stride_row,
        eps,
        output_scale,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_off = row * stride_row
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        safe_cols = tl.minimum(cols, n_cols - 1)

        x = tl.load(x_ptr + row_off + safe_cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        inv_rms = 1.0 / tl.sqrt(var + eps)
        normalized = (x * inv_rms).to(out_ptr.dtype.element_ty)
        scale = tl.load(scale_ptr + safe_cols, mask=mask, other=0.0).to(tl.float32)
        scaled = (normalized * scale).to(out_ptr.dtype.element_ty)
        out = scaled * output_scale
        tl.store(out_ptr + row_off + safe_cols, out, mask=mask)

    @triton.jit
    def _rmsnorm_dual_kernel(
        x_ptr,
        weight_a_ptr,
        weight_b_ptr,
        out_a_ptr,
        out_b_ptr,
        n_cols,
        stride_row,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_off = row * stride_row
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        safe_cols = tl.minimum(cols, n_cols - 1)

        x = tl.load(x_ptr + row_off + safe_cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        normalized = x * (1.0 / tl.sqrt(var + eps))
        weight_a = tl.load(weight_a_ptr + safe_cols, mask=mask, other=0.0).to(tl.float32)
        weight_b = tl.load(weight_b_ptr + safe_cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_a_ptr + row_off + safe_cols, normalized * weight_a, mask=mask)
        tl.store(out_b_ptr + row_off + safe_cols, normalized * weight_b, mask=mask)

    @triton.jit
    def _rmsnorm_add_dual_kernel(
        lhs_ptr,
        rhs_ptr,
        shared_weight_ptr,
        expert_weight_ptr,
        hidden_ptr,
        shared_out_ptr,
        expert_out_ptr,
        n_cols,
        lhs_stride_row,
        rhs_stride_row,
        hidden_stride_row,
        shared_stride_row,
        expert_stride_row,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        safe_cols = tl.minimum(cols, n_cols - 1)

        lhs = tl.load(
            lhs_ptr + row * lhs_stride_row + safe_cols,
            mask=mask,
            other=0.0,
        )
        rhs = tl.load(
            rhs_ptr + row * rhs_stride_row + safe_cols,
            mask=mask,
            other=0.0,
        )
        # Preserve torch.add(..., out=BF16/FP16)'s materialized result before
        # either pre-FFN normalization consumes it.
        hidden = (lhs + rhs).to(hidden_ptr.dtype.element_ty)
        tl.store(
            hidden_ptr + row * hidden_stride_row + safe_cols,
            hidden,
            mask=mask,
        )

        hidden_f32 = hidden.to(tl.float32)
        hidden_var = tl.sum(hidden_f32 * hidden_f32, axis=0) / n_cols
        inv_hidden_rms = 1.0 / tl.sqrt(hidden_var + eps)
        shared_weight = tl.load(
            shared_weight_ptr + safe_cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        expert_weight = tl.load(
            expert_weight_ptr + safe_cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            shared_out_ptr + row * shared_stride_row + safe_cols,
            hidden_f32 * inv_hidden_rms * shared_weight,
            mask=mask,
        )
        tl.store(
            expert_out_ptr + row * expert_stride_row + safe_cols,
            hidden_f32 * inv_hidden_rms * expert_weight,
            mask=mask,
        )

    @triton.jit
    def _rmsnorm_add_dual_router_kernel(
        lhs_ptr,
        rhs_ptr,
        shared_weight_ptr,
        expert_weight_ptr,
        router_scale_ptr,
        hidden_ptr,
        shared_out_ptr,
        expert_out_ptr,
        router_out_ptr,
        n_cols,
        lhs_stride_row,
        rhs_stride_row,
        hidden_stride_row,
        shared_stride_row,
        expert_stride_row,
        router_stride_row,
        eps,
        router_output_scale,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        safe_cols = tl.minimum(cols, n_cols - 1)

        lhs = tl.load(
            lhs_ptr + row * lhs_stride_row + safe_cols,
            mask=mask,
            other=0.0,
        )
        rhs = tl.load(
            rhs_ptr + row * rhs_stride_row + safe_cols,
            mask=mask,
            other=0.0,
        )
        hidden = (lhs + rhs).to(hidden_ptr.dtype.element_ty)
        tl.store(
            hidden_ptr + row * hidden_stride_row + safe_cols,
            hidden,
            mask=mask,
        )

        hidden_f32 = hidden.to(tl.float32)
        hidden_var = tl.sum(hidden_f32 * hidden_f32, axis=0) / n_cols
        inv_hidden_rms = 1.0 / tl.sqrt(hidden_var + eps)
        normalized = hidden_f32 * inv_hidden_rms
        shared_weight = tl.load(
            shared_weight_ptr + safe_cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        expert_weight = tl.load(
            expert_weight_ptr + safe_cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            shared_out_ptr + row * shared_stride_row + safe_cols,
            normalized * shared_weight,
            mask=mask,
        )
        tl.store(
            expert_out_ptr + row * expert_stride_row + safe_cols,
            normalized * expert_weight,
            mask=mask,
        )

        # Match scaled-no-weight RMSNorm's two BF16/FP16 roundings exactly.
        normalized_rounded = normalized.to(router_out_ptr.dtype.element_ty)
        router_scale = tl.load(
            router_scale_ptr + safe_cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scaled = (normalized_rounded * router_scale).to(
            router_out_ptr.dtype.element_ty
        )
        router_out = scaled.to(tl.float32) * router_output_scale
        tl.store(
            router_out_ptr + row * router_stride_row + safe_cols,
            router_out,
            mask=mask,
        )

    @triton.jit
    def _rmsnorm_attn_residual_router_bridge_kernel(
        attn_ptr,
        residual_ptr,
        post_weight_ptr,
        shared_weight_ptr,
        expert_weight_ptr,
        router_scale_ptr,
        post_norm_out_ptr,
        hidden_ptr,
        shared_out_ptr,
        expert_out_ptr,
        router_out_ptr,
        n_cols,
        attn_stride_row,
        residual_stride_row,
        post_norm_stride_row,
        hidden_stride_row,
        shared_stride_row,
        expert_stride_row,
        router_stride_row,
        eps,
        router_output_scale,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        safe_cols = tl.minimum(cols, n_cols - 1)

        attn = tl.load(
            attn_ptr + row * attn_stride_row + safe_cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        attn_var = tl.sum(attn * attn, axis=0) / n_cols
        post_weight = tl.load(
            post_weight_ptr + safe_cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        # Materialize the first RMSNorm in the destination dtype before the
        # residual add, exactly as the former first kernel's global store did.
        post_norm = (
            attn * (1.0 / tl.sqrt(attn_var + eps)) * post_weight
        ).to(post_norm_out_ptr.dtype.element_ty)
        tl.store(
            post_norm_out_ptr + row * post_norm_stride_row + safe_cols,
            post_norm,
            mask=mask,
        )

        residual = tl.load(
            residual_ptr + row * residual_stride_row + safe_cols,
            mask=mask,
            other=0.0,
        )
        hidden = (residual + post_norm).to(hidden_ptr.dtype.element_ty)
        tl.store(
            hidden_ptr + row * hidden_stride_row + safe_cols,
            hidden,
            mask=mask,
        )

        hidden_f32 = hidden.to(tl.float32)
        hidden_var = tl.sum(hidden_f32 * hidden_f32, axis=0) / n_cols
        normalized = hidden_f32 * (1.0 / tl.sqrt(hidden_var + eps))
        shared_weight = tl.load(
            shared_weight_ptr + safe_cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        expert_weight = tl.load(
            expert_weight_ptr + safe_cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            shared_out_ptr + row * shared_stride_row + safe_cols,
            normalized * shared_weight,
            mask=mask,
        )
        tl.store(
            expert_out_ptr + row * expert_stride_row + safe_cols,
            normalized * expert_weight,
            mask=mask,
        )

        # Preserve both materialized low-precision roundings used by the
        # scaled no-weight router normalization.
        normalized_rounded = normalized.to(router_out_ptr.dtype.element_ty)
        router_scale = tl.load(
            router_scale_ptr + safe_cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scaled = (normalized_rounded * router_scale).to(
            router_out_ptr.dtype.element_ty
        )
        router_out = scaled.to(tl.float32) * router_output_scale
        tl.store(
            router_out_ptr + row * router_stride_row + safe_cols,
            router_out,
            mask=mask,
        )

    @triton.jit
    def _rmsnorm_weighted_scaled_no_weight_dual_kernel(
        x_ptr,
        weight_ptr,
        scale_ptr,
        weighted_out_ptr,
        scaled_out_ptr,
        n_cols,
        x_stride_row,
        weighted_stride_row,
        scaled_stride_row,
        eps,
        output_scale,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        safe_cols = tl.minimum(cols, n_cols - 1)
        x = tl.load(
            x_ptr + row * x_stride_row + safe_cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        normalized = x * (1.0 / tl.sqrt(var + eps))

        weight = tl.load(weight_ptr + safe_cols, mask=mask, other=0.0).to(tl.float32)
        weighted = normalized * weight
        tl.store(
            weighted_out_ptr + row * weighted_stride_row + safe_cols,
            weighted,
            mask=mask,
        )

        # Match scaled-no-weight RMSNorm's two intermediate BF16/FP16 stores.
        normalized_rounded = normalized.to(scaled_out_ptr.dtype.element_ty)
        scale = tl.load(scale_ptr + safe_cols, mask=mask, other=0.0).to(tl.float32)
        scaled = (normalized_rounded * scale).to(scaled_out_ptr.dtype.element_ty)
        router_input = scaled.to(tl.float32) * output_scale
        tl.store(
            scaled_out_ptr + row * scaled_stride_row + safe_cols,
            router_input,
            mask=mask,
        )

    @triton.jit
    def _rmsnorm_add_kernel(
        lhs_ptr,
        rhs_ptr,
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
        safe_cols = tl.minimum(cols, n_cols - 1)

        lhs = tl.load(lhs_ptr + row_off + safe_cols, mask=mask, other=0.0)
        rhs = tl.load(rhs_ptr + row_off + safe_cols, mask=mask, other=0.0)
        # Preserve the eager `lhs + rhs` BF16/FP16 rounding before RMSNorm.
        summed = (lhs + rhs).to(out_ptr.dtype.element_ty).to(tl.float32)
        var = tl.sum(summed * summed, axis=0) / n_cols
        weight = tl.load(weight_ptr + safe_cols, mask=mask, other=0.0).to(tl.float32)
        out = summed * (1.0 / tl.sqrt(var + eps)) * weight
        tl.store(out_ptr + row_off + safe_cols, out, mask=mask)

    @triton.jit
    def _rmsnorm_pair_add_final_kernel(
        shared_ptr,
        expert_ptr,
        shared_weight_ptr,
        expert_weight_ptr,
        final_weight_ptr,
        residual_ptr,
        out_ptr,
        n_cols,
        shared_stride_row,
        expert_stride_row,
        residual_stride_row,
        out_stride_row,
        eps,
        ADD_RESIDUAL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        safe_cols = tl.minimum(cols, n_cols - 1)

        shared = tl.load(
            shared_ptr + row * shared_stride_row + safe_cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        expert = tl.load(
            expert_ptr + row * expert_stride_row + safe_cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        shared_var = tl.sum(shared * shared, axis=0) / n_cols
        expert_var = tl.sum(expert * expert, axis=0) / n_cols
        shared_weight = tl.load(
            shared_weight_ptr + safe_cols, mask=mask, other=0.0
        ).to(tl.float32)
        expert_weight = tl.load(
            expert_weight_ptr + safe_cols, mask=mask, other=0.0
        ).to(tl.float32)

        # Match the two standalone BF16/FP16 branch RMSNorm stores before add.
        shared_norm = (
            shared * (1.0 / tl.sqrt(shared_var + eps)) * shared_weight
        ).to(out_ptr.dtype.element_ty)
        expert_norm = (
            expert * (1.0 / tl.sqrt(expert_var + eps)) * expert_weight
        ).to(out_ptr.dtype.element_ty)
        summed = (shared_norm + expert_norm).to(out_ptr.dtype.element_ty).to(tl.float32)
        final_var = tl.sum(summed * summed, axis=0) / n_cols
        final_weight = tl.load(
            final_weight_ptr + safe_cols, mask=mask, other=0.0
        ).to(tl.float32)
        out = summed * (1.0 / tl.sqrt(final_var + eps)) * final_weight
        if ADD_RESIDUAL:
            # Preserve the standalone final RMSNorm store before residual.add_.
            out = out.to(out_ptr.dtype.element_ty).to(tl.float32)
            residual = tl.load(
                residual_ptr + row * residual_stride_row + safe_cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            out += residual
        tl.store(
            out_ptr + row * out_stride_row + safe_cols,
            out,
            mask=mask,
        )


def rmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    offset: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.shape[-1] != weight.numel():
        raise ValueError("rmsnorm_triton weight size must match the last dimension")

    if out is not None:
        if out.shape != x.shape:
            raise ValueError("rmsnorm_triton out shape must match input")
        if out.device != x.device or out.dtype != x.dtype:
            raise ValueError("rmsnorm_triton out device/dtype must match input")
        if out.stride(-1) != 1:
            raise ValueError("rmsnorm_triton out must be contiguous in the last dimension")

    if not (_HAS_TRITON and x.is_cuda and weight.is_cuda and x.stride(-1) == 1):
        result = _pytorch_rmsnorm(x, weight, eps, offset)
        if out is not None:
            out.copy_(result)
            return out
        return result

    n_cols = x.shape[-1]
    if n_cols == 0 or n_cols > 8192:
        result = _pytorch_rmsnorm(x, weight, eps, offset)
        if out is not None:
            out.copy_(result)
            return out
        return result

    x_2d = x.reshape(-1, n_cols)
    out_2d = torch.empty_like(x_2d) if out is None else out.reshape(-1, n_cols)

    block_size = triton.next_power_of_2(n_cols)

    # Cap num_warps at 4 for safety across all architectures.
    # RMSNorm is memory-bound — more warps don't help.
    num_warps = min(4, max(1, block_size // 256))

    _rmsnorm_kernel[(x_2d.shape[0],)](
        x_2d,
        weight,
        out_2d,
        n_cols,
        x_2d.stride(0),
        out_2d.stride(0),
        eps,
        OFFSET=offset,
        HAS_WEIGHT=True,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out_2d.reshape_as(x)


def rmsnorm_triton_no_weight(
    x: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    if not (_HAS_TRITON and x.is_cuda and x.stride(-1) == 1):
        return _pytorch_rmsnorm_no_weight(x, eps)

    n_cols = int(x.shape[-1])
    if n_cols == 0 or n_cols > 8192:
        return _pytorch_rmsnorm_no_weight(x, eps)

    x_2d = x.reshape(-1, n_cols)
    out = torch.empty_like(x_2d)
    block_size = triton.next_power_of_2(n_cols)
    num_warps = min(4, max(1, block_size // 256))
    _rmsnorm_kernel[(x_2d.shape[0],)](
        x_2d,
        x_2d,
        out,
        n_cols,
        x_2d.stride(0),
        out.stride(0),
        eps,
        OFFSET=False,
        HAS_WEIGHT=False,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.reshape_as(x)


def rmsnorm_triton_dual(
    x: torch.Tensor,
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] != weight_a.numel() or x.shape[-1] != weight_b.numel():
        raise ValueError("dual RMSNorm weights must match the last dimension")
    if not (
        _HAS_TRITON
        and x.is_cuda
        and weight_a.is_cuda
        and weight_b.is_cuda
        and x.stride(-1) == 1
        and weight_a.is_contiguous()
        and weight_b.is_contiguous()
    ):
        return (
            _pytorch_rmsnorm(x, weight_a, eps, False),
            _pytorch_rmsnorm(x, weight_b, eps, False),
        )

    n_cols = int(x.shape[-1])
    if n_cols == 0 or n_cols > 8192:
        return (
            _pytorch_rmsnorm(x, weight_a, eps, False),
            _pytorch_rmsnorm(x, weight_b, eps, False),
        )
    x_2d = x.reshape(-1, n_cols)
    out_a = torch.empty_like(x_2d)
    out_b = torch.empty_like(x_2d)
    block_size = triton.next_power_of_2(n_cols)
    num_warps = min(4, max(1, block_size // 256))
    _rmsnorm_dual_kernel[(x_2d.shape[0],)](
        x_2d,
        weight_a,
        weight_b,
        out_a,
        out_b,
        n_cols,
        x_2d.stride(0),
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out_a.reshape_as(x), out_b.reshape_as(x)


def rmsnorm_triton_add_dual(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    shared_weight: torch.Tensor,
    expert_weight: torch.Tensor,
    eps: float = 1e-5,
    *,
    out_hidden: Optional[torch.Tensor] = None,
    shared_out: Optional[torch.Tensor] = None,
    expert_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse a staged add with two RMSNorms while returning the materialized sum."""
    if lhs.shape != rhs.shape:
        raise ValueError("add-dual RMSNorm inputs must have the same shape")
    n_cols = int(lhs.shape[-1])
    if n_cols != shared_weight.numel() or n_cols != expert_weight.numel():
        raise ValueError("add-dual RMSNorm weights must match the last dimension")
    if out_hidden is not None:
        if out_hidden.shape != lhs.shape:
            raise ValueError("add-dual RMSNorm out_hidden shape must match inputs")
        if out_hidden.device != lhs.device or out_hidden.dtype != lhs.dtype:
            raise ValueError(
                "add-dual RMSNorm out_hidden device/dtype must match inputs"
            )
        if not out_hidden.is_contiguous():
            raise ValueError(
                "add-dual RMSNorm out_hidden must be contiguous"
            )
    for name, output in (
        ("shared_out", shared_out),
        ("expert_out", expert_out),
    ):
        if output is None:
            continue
        if output.shape != lhs.shape:
            raise ValueError(f"add-dual RMSNorm {name} shape must match inputs")
        if output.device != lhs.device or output.dtype != lhs.dtype:
            raise ValueError(
                f"add-dual RMSNorm {name} device/dtype must match inputs"
            )
        if not output.is_contiguous():
            raise ValueError(f"add-dual RMSNorm {name} must be contiguous")

    can_use_triton = bool(
        _HAS_TRITON
        and lhs.is_cuda
        and rhs.is_cuda
        and all(
            weight.is_cuda
            and weight.device == lhs.device
            and weight.is_contiguous()
            for weight in (shared_weight, expert_weight)
        )
        and rhs.device == lhs.device
        and rhs.dtype == lhs.dtype
        and lhs.is_contiguous()
        and rhs.is_contiguous()
        and 0 < n_cols <= 8192
    )
    if not can_use_triton:
        hidden = (lhs + rhs).to(lhs.dtype)
        if out_hidden is not None:
            out_hidden.copy_(hidden)
            hidden = out_hidden
        shared_result = _pytorch_rmsnorm(hidden, shared_weight, eps, False)
        expert_result = _pytorch_rmsnorm(hidden, expert_weight, eps, False)
        if shared_out is not None:
            shared_out.copy_(shared_result)
            shared_result = shared_out
        if expert_out is not None:
            expert_out.copy_(expert_result)
            expert_result = expert_out
        return (
            hidden,
            shared_result,
            expert_result,
        )

    lhs_2d = lhs.reshape(-1, n_cols)
    rhs_2d = rhs.reshape(-1, n_cols)
    hidden_2d = (
        torch.empty_like(lhs_2d)
        if out_hidden is None
        else out_hidden.reshape(-1, n_cols)
    )
    shared_out_2d = (
        torch.empty_like(lhs_2d)
        if shared_out is None
        else shared_out.reshape(-1, n_cols)
    )
    expert_out_2d = (
        torch.empty_like(lhs_2d)
        if expert_out is None
        else expert_out.reshape(-1, n_cols)
    )
    block_size = triton.next_power_of_2(n_cols)
    num_warps = min(4, max(1, block_size // 256))
    _rmsnorm_add_dual_kernel[(lhs_2d.shape[0],)](
        lhs_2d,
        rhs_2d,
        shared_weight,
        expert_weight,
        hidden_2d,
        shared_out_2d,
        expert_out_2d,
        n_cols,
        lhs_2d.stride(0),
        rhs_2d.stride(0),
        hidden_2d.stride(0),
        shared_out_2d.stride(0),
        expert_out_2d.stride(0),
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return (
        hidden_2d.reshape_as(lhs),
        shared_out_2d.reshape_as(lhs),
        expert_out_2d.reshape_as(lhs),
    )


def rmsnorm_triton_add_dual_router(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    shared_weight: torch.Tensor,
    expert_weight: torch.Tensor,
    router_scale: torch.Tensor,
    eps: float,
    router_output_scale: float,
    *,
    out_hidden: Optional[torch.Tensor] = None,
    shared_out: Optional[torch.Tensor] = None,
    expert_out: Optional[torch.Tensor] = None,
    router_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Share the post-attention RMS reduction with both MoE branches and router."""
    if lhs.shape != rhs.shape:
        raise ValueError("add-dual-router RMSNorm inputs must have the same shape")
    n_cols = int(lhs.shape[-1])
    vectors = (shared_weight, expert_weight, router_scale)
    if any(int(vector.numel()) != n_cols for vector in vectors):
        raise ValueError(
            "add-dual-router RMSNorm vectors must match the last dimension"
        )
    for name, output in (
        ("out_hidden", out_hidden),
        ("shared_out", shared_out),
        ("expert_out", expert_out),
        ("router_out", router_out),
    ):
        if output is None:
            continue
        if output.shape != lhs.shape:
            raise ValueError(f"add-dual-router RMSNorm {name} shape must match inputs")
        if output.device != lhs.device or output.dtype != lhs.dtype:
            raise ValueError(
                f"add-dual-router RMSNorm {name} device/dtype must match inputs"
            )
        if not output.is_contiguous():
            raise ValueError(f"add-dual-router RMSNorm {name} must be contiguous")

    can_use_triton = bool(
        _HAS_TRITON
        and lhs.is_cuda
        and rhs.is_cuda
        and rhs.device == lhs.device
        and rhs.dtype == lhs.dtype
        and lhs.is_contiguous()
        and rhs.is_contiguous()
        and all(
            vector.is_cuda
            and vector.device == lhs.device
            and vector.is_contiguous()
            for vector in vectors
        )
        and 0 < n_cols <= 8192
    )
    if not can_use_triton:
        hidden, shared_result, expert_result = rmsnorm_triton_add_dual(
            lhs,
            rhs,
            shared_weight,
            expert_weight,
            eps,
            out_hidden=out_hidden,
            shared_out=shared_out,
            expert_out=expert_out,
        )
        router_result = rmsnorm_triton_scaled_no_weight(
            hidden,
            router_scale,
            eps,
            router_output_scale,
            out=router_out,
        )
        return hidden, shared_result, expert_result, router_result

    lhs_2d = lhs.reshape(-1, n_cols)
    rhs_2d = rhs.reshape(-1, n_cols)
    outputs = []
    for output in (out_hidden, shared_out, expert_out, router_out):
        outputs.append(
            torch.empty_like(lhs_2d)
            if output is None
            else output.reshape(-1, n_cols)
        )
    hidden_2d, shared_2d, expert_2d, router_2d = outputs
    block_size = triton.next_power_of_2(n_cols)
    num_warps = min(4, max(1, block_size // 256))
    _rmsnorm_add_dual_router_kernel[(lhs_2d.shape[0],)](
        lhs_2d,
        rhs_2d,
        shared_weight,
        expert_weight,
        router_scale,
        hidden_2d,
        shared_2d,
        expert_2d,
        router_2d,
        n_cols,
        lhs_2d.stride(0),
        rhs_2d.stride(0),
        hidden_2d.stride(0),
        shared_2d.stride(0),
        expert_2d.stride(0),
        router_2d.stride(0),
        eps,
        float(router_output_scale),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return tuple(output.reshape_as(lhs) for output in outputs)


def rmsnorm_triton_attn_residual_dual(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_attn_weight: torch.Tensor,
    shared_weight: torch.Tensor,
    expert_weight: torch.Tensor,
    eps: float = 1e-5,
    *,
    out_hidden: Optional[torch.Tensor] = None,
    post_norm_out: Optional[torch.Tensor] = None,
    shared_out: Optional[torch.Tensor] = None,
    expert_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the exact two-kernel Gemma4 attention-to-MoE bridge."""
    if attn_out.shape != residual.shape:
        raise ValueError("attention output and residual must have the same shape")
    if int(attn_out.shape[-1]) != post_attn_weight.numel():
        raise ValueError(
            "attention bridge post-attention weight must match the last dimension"
        )
    post_norm = rmsnorm_triton(
        attn_out,
        post_attn_weight,
        eps,
        False,
        out=post_norm_out,
    )
    return rmsnorm_triton_add_dual(
        residual,
        post_norm,
        shared_weight,
        expert_weight,
        eps,
        out_hidden=out_hidden,
        shared_out=shared_out,
        expert_out=expert_out,
    )


def rmsnorm_triton_attn_residual_router_bridge(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_attn_weight: torch.Tensor,
    shared_weight: torch.Tensor,
    expert_weight: torch.Tensor,
    router_scale: torch.Tensor,
    eps: float,
    router_output_scale: float,
    *,
    out_hidden: Optional[torch.Tensor] = None,
    post_norm_out: Optional[torch.Tensor] = None,
    shared_out: Optional[torch.Tensor] = None,
    expert_out: Optional[torch.Tensor] = None,
    router_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Gemma4's exact two-kernel attention-to-MoE/router bridge."""
    if attn_out.shape != residual.shape:
        raise ValueError("attention output and residual must have the same shape")
    if int(attn_out.shape[-1]) != post_attn_weight.numel():
        raise ValueError(
            "attention router bridge post-attention weight must match the last dimension"
        )
    post_norm = rmsnorm_triton(
        attn_out,
        post_attn_weight,
        eps,
        False,
        out=post_norm_out,
    )
    return rmsnorm_triton_add_dual_router(
        residual,
        post_norm,
        shared_weight,
        expert_weight,
        router_scale,
        eps,
        router_output_scale,
        out_hidden=out_hidden,
        shared_out=shared_out,
        expert_out=expert_out,
        router_out=router_out,
    )


def rmsnorm_triton_attn_residual_router_bridge_single(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_attn_weight: torch.Tensor,
    shared_weight: torch.Tensor,
    expert_weight: torch.Tensor,
    router_scale: torch.Tensor,
    eps: float,
    router_output_scale: float,
    *,
    out_hidden: Optional[torch.Tensor] = None,
    post_norm_out: Optional[torch.Tensor] = None,
    shared_out: Optional[torch.Tensor] = None,
    expert_out: Optional[torch.Tensor] = None,
    router_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Gemma4's attention-to-MoE/router bridge in one Triton kernel."""
    if attn_out.shape != residual.shape:
        raise ValueError("attention output and residual must have the same shape")
    n_cols = int(attn_out.shape[-1])
    vectors = (
        post_attn_weight,
        shared_weight,
        expert_weight,
        router_scale,
    )
    if any(int(vector.numel()) != n_cols for vector in vectors):
        raise ValueError(
            "single-kernel attention router bridge vectors must match the last dimension"
        )
    for name, output in (
        ("out_hidden", out_hidden),
        ("post_norm_out", post_norm_out),
        ("shared_out", shared_out),
        ("expert_out", expert_out),
        ("router_out", router_out),
    ):
        if output is None:
            continue
        if output.shape != attn_out.shape:
            raise ValueError(
                f"single-kernel attention router bridge {name} shape must match inputs"
            )
        if output.device != attn_out.device or output.dtype != attn_out.dtype:
            raise ValueError(
                f"single-kernel attention router bridge {name} device/dtype must match inputs"
            )
        if not output.is_contiguous():
            raise ValueError(
                f"single-kernel attention router bridge {name} must be contiguous"
            )

    can_use_triton = bool(
        _HAS_TRITON
        and attn_out.is_cuda
        and residual.is_cuda
        and residual.device == attn_out.device
        and residual.dtype == attn_out.dtype
        and attn_out.is_contiguous()
        and residual.is_contiguous()
        and all(
            vector.is_cuda
            and vector.device == attn_out.device
            and vector.is_contiguous()
            for vector in vectors
        )
        and 0 < n_cols <= 8192
    )
    if not can_use_triton:
        return rmsnorm_triton_attn_residual_router_bridge(
            attn_out,
            residual,
            post_attn_weight,
            shared_weight,
            expert_weight,
            router_scale,
            eps,
            router_output_scale,
            out_hidden=out_hidden,
            post_norm_out=post_norm_out,
            shared_out=shared_out,
            expert_out=expert_out,
            router_out=router_out,
        )

    attn_2d = attn_out.reshape(-1, n_cols)
    residual_2d = residual.reshape(-1, n_cols)
    post_norm_2d = (
        torch.empty_like(attn_2d)
        if post_norm_out is None
        else post_norm_out.reshape(-1, n_cols)
    )
    output_values = []
    for output in (out_hidden, shared_out, expert_out, router_out):
        output_values.append(
            torch.empty_like(attn_2d)
            if output is None
            else output.reshape(-1, n_cols)
        )
    hidden_2d, shared_2d, expert_2d, router_2d = output_values
    block_size = triton.next_power_of_2(n_cols)
    num_warps = min(4, max(1, block_size // 256))
    _rmsnorm_attn_residual_router_bridge_kernel[(attn_2d.shape[0],)](
        attn_2d,
        residual_2d,
        post_attn_weight,
        shared_weight,
        expert_weight,
        router_scale,
        post_norm_2d,
        hidden_2d,
        shared_2d,
        expert_2d,
        router_2d,
        n_cols,
        attn_2d.stride(0),
        residual_2d.stride(0),
        post_norm_2d.stride(0),
        hidden_2d.stride(0),
        shared_2d.stride(0),
        expert_2d.stride(0),
        router_2d.stride(0),
        eps,
        float(router_output_scale),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return tuple(output.reshape_as(attn_out) for output in output_values)


def rmsnorm_triton_add(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    if lhs.shape != rhs.shape:
        raise ValueError("add RMSNorm inputs must have the same shape")
    if lhs.shape[-1] != weight.numel():
        raise ValueError("add RMSNorm weight must match the last dimension")
    if not (
        _HAS_TRITON
        and lhs.is_cuda
        and rhs.is_cuda
        and weight.is_cuda
        and lhs.stride(-1) == 1
        and rhs.stride(-1) == 1
        and weight.is_contiguous()
    ):
        summed = (lhs + rhs).to(lhs.dtype)
        return _pytorch_rmsnorm(summed, weight, eps, False)

    n_cols = int(lhs.shape[-1])
    if n_cols == 0 or n_cols > 8192:
        summed = (lhs + rhs).to(lhs.dtype)
        return _pytorch_rmsnorm(summed, weight, eps, False)
    lhs_2d = lhs.reshape(-1, n_cols)
    rhs_2d = rhs.reshape(-1, n_cols)
    out = torch.empty_like(lhs_2d)
    block_size = triton.next_power_of_2(n_cols)
    num_warps = min(4, max(1, block_size // 256))
    _rmsnorm_add_kernel[(lhs_2d.shape[0],)](
        lhs_2d,
        rhs_2d,
        weight,
        out,
        n_cols,
        lhs_2d.stride(0),
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.reshape_as(lhs)


def rmsnorm_triton_pair_add_final(
    shared: torch.Tensor,
    expert: torch.Tensor,
    shared_weight: torch.Tensor,
    expert_weight: torch.Tensor,
    final_weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fuse two independent branch RMSNorms, their add, and final RMSNorm."""
    if shared.shape != expert.shape:
        raise ValueError("paired RMSNorm inputs must have the same shape")
    n_cols = int(shared.shape[-1])
    weights = (shared_weight, expert_weight, final_weight)
    if any(int(weight.numel()) != n_cols for weight in weights):
        raise ValueError("paired RMSNorm weights must match the last dimension")

    eligible = bool(
        _HAS_TRITON
        and shared.is_cuda
        and expert.is_cuda
        and all(weight.is_cuda for weight in weights)
        and shared.stride(-1) == 1
        and expert.stride(-1) == 1
        and all(weight.is_contiguous() for weight in weights)
        and 0 < n_cols <= 8192
    )
    if not eligible:
        shared_norm = _pytorch_rmsnorm(shared, shared_weight, eps, False)
        expert_norm = _pytorch_rmsnorm(expert, expert_weight, eps, False)
        summed = (shared_norm + expert_norm).to(shared.dtype)
        return _pytorch_rmsnorm(summed, final_weight, eps, False)

    shared_2d = shared.reshape(-1, n_cols)
    expert_2d = expert.reshape(-1, n_cols)
    out = torch.empty_like(shared_2d)
    block_size = triton.next_power_of_2(n_cols)
    num_warps = min(4, max(1, block_size // 256))
    _rmsnorm_pair_add_final_kernel[(shared_2d.shape[0],)](
        shared_2d,
        expert_2d,
        shared_weight,
        expert_weight,
        final_weight,
        shared_2d,
        out,
        n_cols,
        shared_2d.stride(0),
        expert_2d.stride(0),
        shared_2d.stride(0),
        out.stride(0),
        eps,
        ADD_RESIDUAL=False,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.reshape_as(shared)


def rmsnorm_triton_pair_add_final_residual(
    shared: torch.Tensor,
    expert: torch.Tensor,
    shared_weight: torch.Tensor,
    expert_weight: torch.Tensor,
    final_weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fuse post-MoE branch norms, branch add, final norm, and residual add."""
    if shared.shape != expert.shape or shared.shape != residual.shape:
        raise ValueError("fused post-MoE inputs must have the same shape")
    n_cols = int(shared.shape[-1])
    weights = (shared_weight, expert_weight, final_weight)
    if any(int(weight.numel()) != n_cols for weight in weights):
        raise ValueError("fused post-MoE weights must match the last dimension")
    if expert.device != shared.device or residual.device != shared.device:
        raise ValueError("fused post-MoE inputs must be on the same device")
    if expert.dtype != shared.dtype or residual.dtype != shared.dtype:
        raise ValueError("fused post-MoE inputs must have the same dtype")
    if out is not None:
        if out.shape != shared.shape:
            raise ValueError("fused post-MoE out shape must match input")
        if out.device != shared.device or out.dtype != shared.dtype:
            raise ValueError("fused post-MoE out device/dtype must match input")
        if out.stride(-1) != 1:
            raise ValueError("fused post-MoE out must be contiguous in the last dimension")

    eligible = bool(
        _HAS_TRITON
        and shared.is_cuda
        and expert.is_cuda
        and residual.is_cuda
        and all(weight.is_cuda for weight in weights)
        and shared.stride(-1) == 1
        and expert.stride(-1) == 1
        and residual.stride(-1) == 1
        and all(weight.is_contiguous() for weight in weights)
        and 0 < n_cols <= 8192
    )
    if not eligible:
        shared_norm = _pytorch_rmsnorm(shared, shared_weight, eps, False)
        expert_norm = _pytorch_rmsnorm(expert, expert_weight, eps, False)
        summed = (shared_norm + expert_norm).to(shared.dtype)
        final_norm = _pytorch_rmsnorm(summed, final_weight, eps, False)
        result = (residual + final_norm).to(shared.dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    shared_2d = shared.reshape(-1, n_cols)
    expert_2d = expert.reshape(-1, n_cols)
    residual_2d = residual.reshape(-1, n_cols)
    out_2d = torch.empty_like(shared_2d) if out is None else out.reshape(-1, n_cols)
    block_size = triton.next_power_of_2(n_cols)
    num_warps = min(4, max(1, block_size // 256))
    _rmsnorm_pair_add_final_kernel[(shared_2d.shape[0],)](
        shared_2d,
        expert_2d,
        shared_weight,
        expert_weight,
        final_weight,
        residual_2d,
        out_2d,
        n_cols,
        shared_2d.stride(0),
        expert_2d.stride(0),
        residual_2d.stride(0),
        out_2d.stride(0),
        eps,
        ADD_RESIDUAL=True,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out_2d.reshape_as(shared)


def rmsnorm_triton_scaled_no_weight(
    x: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    output_scale: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.shape[-1] != scale.numel():
        raise ValueError("scale size must match the last dimension")
    if out is not None:
        if out.shape != x.shape:
            raise ValueError("scaled no-weight RMSNorm out shape must match input")
        if out.device != x.device or out.dtype != x.dtype:
            raise ValueError(
                "scaled no-weight RMSNorm out device/dtype must match input"
            )
        if out.stride(-1) != 1:
            raise ValueError(
                "scaled no-weight RMSNorm out must be contiguous in the last dimension"
            )
    if not (
        _HAS_TRITON
        and x.is_cuda
        and scale.is_cuda
        and x.stride(-1) == 1
        and scale.is_contiguous()
    ):
        normalized = _pytorch_rmsnorm_no_weight(x, eps)
        result = normalized.mul(scale).mul(float(output_scale))
        if out is not None:
            out.copy_(result)
            return out
        return result

    n_cols = int(x.shape[-1])
    if n_cols == 0 or n_cols > 8192:
        normalized = _pytorch_rmsnorm_no_weight(x, eps)
        result = normalized.mul(scale).mul(float(output_scale))
        if out is not None:
            out.copy_(result)
            return out
        return result

    x_2d = x.reshape(-1, n_cols)
    out_2d = torch.empty_like(x_2d) if out is None else out.reshape(-1, n_cols)
    block_size = triton.next_power_of_2(n_cols)
    num_warps = min(4, max(1, block_size // 256))
    _rmsnorm_scaled_no_weight_kernel[(x_2d.shape[0],)](
        x_2d,
        scale,
        out_2d,
        n_cols,
        x_2d.stride(0),
        eps,
        float(output_scale),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out_2d.reshape_as(x)


def rmsnorm_triton_weighted_scaled_no_weight_dual(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    output_scale: float,
    weighted_out: Optional[torch.Tensor] = None,
    scaled_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Share one RMS reduction between expert-input and router-input outputs."""
    n_cols = int(x.shape[-1])
    if int(weight.numel()) != n_cols or int(scale.numel()) != n_cols:
        raise ValueError("dual scaled RMSNorm vectors must match the last dimension")
    for name, out in (("weighted_out", weighted_out), ("scaled_out", scaled_out)):
        if out is None:
            continue
        if out.shape != x.shape:
            raise ValueError(f"{name} shape must match input")
        if out.device != x.device or out.dtype != x.dtype:
            raise ValueError(f"{name} device/dtype must match input")
        if out.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous in the last dimension")

    eligible = bool(
        _HAS_TRITON
        and x.is_cuda
        and weight.is_cuda
        and scale.is_cuda
        and x.stride(-1) == 1
        and weight.is_contiguous()
        and scale.is_contiguous()
        and 0 < n_cols <= 8192
    )
    if not eligible:
        weighted = _pytorch_rmsnorm(x, weight, eps, False)
        scaled = (
            _pytorch_rmsnorm_no_weight(x, eps)
            .mul(scale)
            .mul(float(output_scale))
        )
        if weighted_out is not None:
            weighted_out.copy_(weighted)
            weighted = weighted_out
        if scaled_out is not None:
            scaled_out.copy_(scaled)
            scaled = scaled_out
        return weighted, scaled

    x_2d = x.reshape(-1, n_cols)
    weighted_2d = (
        torch.empty_like(x_2d)
        if weighted_out is None
        else weighted_out.reshape(-1, n_cols)
    )
    scaled_2d = (
        torch.empty_like(x_2d)
        if scaled_out is None
        else scaled_out.reshape(-1, n_cols)
    )
    block_size = triton.next_power_of_2(n_cols)
    num_warps = min(4, max(1, block_size // 256))
    _rmsnorm_weighted_scaled_no_weight_dual_kernel[(x_2d.shape[0],)](
        x_2d,
        weight,
        scale,
        weighted_2d,
        scaled_2d,
        n_cols,
        x_2d.stride(0),
        weighted_2d.stride(0),
        scaled_2d.stride(0),
        eps,
        float(output_scale),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return weighted_2d.reshape_as(x), scaled_2d.reshape_as(x)


HAS_TRITON_RMSNORM = _HAS_TRITON

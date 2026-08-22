"""Specialized Gemma 4 A4B matrix router for prefill and batched decode."""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False


__all__ = [
    "HAS_GEMMA4_MOE_ROUTER",
    "gemma4_moe_prefill_router_topk",
    "gemma4_moe_prefill_router_prefers_shape",
]


HAS_GEMMA4_MOE_ROUTER = _HAS_TRITON


def gemma4_moe_prefill_router_prefers_shape(
    rows: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
) -> bool:
    return bool(
        _HAS_TRITON
        and (0 < int(rows) <= 32 or int(rows) == 400)
        and int(hidden_dim) == 2816
        and int(num_experts) == 128
        and int(top_k) == 8
        and dtype == torch.bfloat16
    )


def _workspace_tensor(
    workspace: Optional[dict[str, torch.Tensor]],
    key: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = None if workspace is None else workspace.get(key)
    if (
        tensor is None
        or tuple(tensor.shape) != tuple(shape)
        or tensor.device != device
        or tensor.dtype != dtype
    ):
        tensor = torch.empty(shape, device=device, dtype=dtype)
        if workspace is not None:
            workspace[key] = tensor
    return tensor


if _HAS_TRITON:
    @triton.jit
    def _gemma4_moe_prefill_router_topk_kernel(
        hidden_ptr,
        router_weight_ptr,
        expert_scale_ptr,
        route_weight_ptr,
        expert_id_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rt: tl.constexpr,
        stride_em: tl.constexpr,
        stride_et: tl.constexpr,
        ROWS: tl.constexpr,
        HIDDEN: tl.constexpr,
        EXPERTS: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        block_m = tl.program_id(0)
        offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_e = tl.arange(0, BLOCK_E)
        m_mask = offs_m < ROWS
        e_mask = offs_e < EXPERTS

        logits = tl.zeros([BLOCK_M, BLOCK_E], dtype=tl.float32)
        for k_start in range(0, HIDDEN, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < HIDDEN
            x = tl.load(
                hidden_ptr
                + offs_m[:, None] * stride_hm
                + offs_k[None, :] * stride_hk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            weight = tl.load(
                router_weight_ptr
                + offs_k[:, None] * stride_wk
                + offs_e[None, :] * stride_we,
                mask=k_mask[:, None] & e_mask[None, :],
                other=0.0,
            )
            logits += tl.dot(x, weight, out_dtype=tl.float32)

        # torch.mm writes BF16 logits before top-k in the existing path.
        vals = logits.to(route_weight_ptr.dtype.element_ty).to(tl.float32)
        vals = tl.where(m_mask[:, None] & e_mask[None, :], vals, -float("inf"))
        top_offsets = tl.arange(0, TOP_K)
        top_vals = tl.full([BLOCK_M, TOP_K], -float("inf"), dtype=tl.float32)
        top_ids = tl.full([BLOCK_M, TOP_K], 0, dtype=tl.int64)

        for top_idx in tl.static_range(0, TOP_K):
            max_val = tl.max(vals, axis=1)
            is_max = vals == max_val[:, None]
            max_idx = tl.min(
                tl.where(is_max, offs_e[None, :], BLOCK_E),
                axis=1,
            )
            slot = top_offsets[None, :] == top_idx
            top_vals = tl.where(slot, max_val[:, None], top_vals)
            top_ids = tl.where(slot, max_idx[:, None].to(tl.int64), top_ids)
            vals = tl.where(offs_e[None, :] == max_idx[:, None], -float("inf"), vals)

        stable_max = tl.max(top_vals, axis=1)
        exp_vals = tl.exp(top_vals - stable_max[:, None])
        probs = exp_vals / tl.sum(exp_vals, axis=1)[:, None]
        selected_scale = tl.load(
            expert_scale_ptr + top_ids,
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        route = probs * selected_scale

        tl.store(
            route_weight_ptr
            + offs_m[:, None] * stride_rm
            + top_offsets[None, :] * stride_rt,
            route,
            mask=m_mask[:, None],
        )
        tl.store(
            expert_id_ptr
            + offs_m[:, None] * stride_em
            + top_offsets[None, :] * stride_et,
            top_ids,
            mask=m_mask[:, None],
        )


def gemma4_moe_prefill_router_topk(
    normalized_hidden: torch.Tensor,
    router_weight: torch.Tensor,
    expert_scale: torch.Tensor,
    top_k: int,
    *,
    workspace: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if normalized_hidden.ndim != 2 or router_weight.ndim != 2:
        raise ValueError("normalized_hidden and router_weight must be 2D")
    rows, hidden_dim = normalized_hidden.shape
    num_experts, weight_hidden = router_weight.shape
    if int(weight_hidden) != int(hidden_dim):
        raise ValueError("router hidden dimensions do not match")
    if int(expert_scale.numel()) != int(num_experts):
        raise ValueError("expert_scale must have one value per expert")
    eligibility = {
        "preferred_shape": gemma4_moe_prefill_router_prefers_shape(
            int(rows), int(hidden_dim), int(num_experts), int(top_k), normalized_hidden.dtype
        ),
        "hidden_cuda": normalized_hidden.is_cuda,
        "weight_cuda": router_weight.is_cuda,
        "expert_scale_cuda": expert_scale.is_cuda,
        "hidden_contiguous": normalized_hidden.is_contiguous(),
        "weight_contiguous": router_weight.is_contiguous(),
        "expert_scale_contiguous": expert_scale.is_contiguous(),
        "weight_dtype": router_weight.dtype == normalized_hidden.dtype,
        "expert_scale_dtype": expert_scale.dtype == normalized_hidden.dtype,
        "grad_disabled": not torch.is_grad_enabled(),
    }
    if not all(eligibility.values()):
        failed = ", ".join(key for key, value in eligibility.items() if not value)
        raise RuntimeError(f"Gemma4 fused matrix router is not eligible: {failed}")

    route_weights = _workspace_tensor(
        workspace,
        "gemma4_prefill_route_weights",
        (int(rows), int(top_k)),
        device=normalized_hidden.device,
        dtype=normalized_hidden.dtype,
    )
    expert_ids = _workspace_tensor(
        workspace,
        "gemma4_prefill_expert_ids",
        (int(rows), int(top_k)),
        device=normalized_hidden.device,
        dtype=torch.int64,
    )
    block_m = 16
    block_e = 128
    block_k = 64
    _gemma4_moe_prefill_router_topk_kernel[
        (triton.cdiv(int(rows), block_m),)
    ](
        normalized_hidden,
        router_weight,
        expert_scale,
        route_weights,
        expert_ids,
        normalized_hidden.stride(0),
        normalized_hidden.stride(1),
        router_weight.stride(0),
        router_weight.stride(1),
        route_weights.stride(0),
        route_weights.stride(1),
        expert_ids.stride(0),
        expert_ids.stride(1),
        ROWS=int(rows),
        HIDDEN=int(hidden_dim),
        EXPERTS=int(num_experts),
        TOP_K=int(top_k),
        BLOCK_M=block_m,
        BLOCK_E=block_e,
        BLOCK_K=block_k,
        num_warps=8,
        num_stages=3,
    )
    return route_weights, expert_ids

"""Grouped-MM expert prefill for the Gemma 4 A4B batch-16 shape.

The regular segmented path is the correctness fallback. This candidate uses
PyTorch's native grouped GEMM twice and keeps only route packing, GeGLU, and
the deterministic top-k reduction in small Triton kernels.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None


_TARGET_ROWS = 400
_TARGET_HIDDEN = 2816
_TARGET_INTERMEDIATE = 704
_TARGET_EXPERTS = 128
_TARGET_TOP_K = 8


def gemma4_grouped_mm_prefill_available() -> bool:
    return bool(_HAS_TRITON and callable(getattr(torch, "_grouped_mm", None)))


def _cuda_graph_capture_active(tensor: torch.Tensor) -> bool:
    if not tensor.is_cuda:
        return False
    checker = getattr(torch.cuda, "is_current_stream_capturing", None)
    if not callable(checker):
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def gemma4_grouped_mm_prefill_prefers_shape(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> bool:
    if not gemma4_grouped_mm_prefill_available():
        return False
    if torch.is_grad_enabled() or _cuda_graph_capture_active(hidden_states):
        return False
    if hidden_states.ndim != 2 or selected_experts.ndim != 2:
        return False
    if tuple(selected_experts.shape) != tuple(routing_weights.shape):
        return False
    rows, hidden_dim = (int(value) for value in hidden_states.shape)
    top_k = int(selected_experts.shape[1])
    if (
        rows != _TARGET_ROWS
        or hidden_dim != _TARGET_HIDDEN
        or top_k != _TARGET_TOP_K
    ):
        return False
    if tuple(gate_up_proj.shape) != (
        _TARGET_EXPERTS,
        2 * _TARGET_INTERMEDIATE,
        _TARGET_HIDDEN,
    ):
        return False
    if tuple(down_proj.shape) != (
        _TARGET_EXPERTS,
        _TARGET_HIDDEN,
        _TARGET_INTERMEDIATE,
    ):
        return False
    tensors = (
        hidden_states,
        gate_up_proj,
        down_proj,
        selected_experts,
        routing_weights,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    if hidden_states.dtype != torch.bfloat16:
        return False
    if gate_up_proj.dtype != hidden_states.dtype or down_proj.dtype != hidden_states.dtype:
        return False
    if routing_weights.dtype != hidden_states.dtype:
        return False
    return True


def _workspace_tensor(
    workspace: Optional[dict[str, Any]],
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = None if workspace is None else workspace.get(name)
    if (
        tensor is None
        or tuple(tensor.shape) != tuple(shape)
        or tensor.device != device
        or tensor.dtype != dtype
    ):
        tensor = torch.empty(shape, device=device, dtype=dtype)
        if workspace is not None:
            workspace[name] = tensor
    return tensor


if _HAS_TRITON:

    @triton.jit
    def _gemma4_grouped_mm_route_pack_kernel(
        experts_ptr,
        sorted_tokens_ptr,
        inverse_slots_ptr,
        offsets_ptr,
        ASSIGNMENTS: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_ASSIGNMENTS: tl.constexpr,
    ):
        expert = tl.program_id(0)
        slots = tl.arange(0, BLOCK_ASSIGNMENTS)
        valid = slots < ASSIGNMENTS
        routed_expert = tl.load(
            experts_ptr + slots,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        matches = valid & (routed_expert == expert)
        local_slot = tl.cumsum(matches.to(tl.int32), axis=0) - 1
        count = tl.sum(matches.to(tl.int32), axis=0)
        start = tl.sum(
            (valid & (routed_expert < expert)).to(tl.int32),
            axis=0,
        )
        sorted_slot = start + local_slot
        tl.store(
            sorted_tokens_ptr + sorted_slot,
            slots // TOP_K,
            mask=matches,
        )
        tl.store(
            inverse_slots_ptr + slots,
            sorted_slot,
            mask=matches,
        )
        tl.store(offsets_ptr + expert, start + count)

    @triton.jit
    def _gemma4_grouped_mm_geglu_kernel(
        gate_up_ptr,
        act_ptr,
        stride_gm: tl.constexpr,
        stride_gn: tl.constexpr,
        stride_am: tl.constexpr,
        stride_an: tl.constexpr,
        ROWS: tl.constexpr,
        I: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < ROWS) & (offs_n[None, :] < I)
        gate = tl.load(
            gate_up_ptr
            + offs_m[:, None] * stride_gm
            + offs_n[None, :] * stride_gn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            gate_up_ptr
            + offs_m[:, None] * stride_gm
            + (offs_n[None, :] + I) * stride_gn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        inner = 0.7978845608028654 * (
            gate + 0.044715 * gate * gate * gate
        )
        activated = gate * tl.sigmoid(2.0 * inner) * up
        tl.store(
            act_ptr
            + offs_m[:, None] * stride_am
            + offs_n[None, :] * stride_an,
            activated.to(act_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _gemma4_grouped_mm_topk_reduce_kernel(
        projected_ptr,
        inverse_slots_ptr,
        routing_ptr,
        residual_ptr,
        out_ptr,
        stride_pm: tl.constexpr,
        stride_ph: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rk: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        ADD_RESIDUAL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H
        top_idx = tl.arange(0, TOP_K)
        original_slots = row * TOP_K + top_idx
        sorted_slots = tl.load(
            inverse_slots_ptr + original_slots
        ).to(tl.int64)
        route = tl.load(
            routing_ptr + row * stride_rm + top_idx * stride_rk
        ).to(tl.float32)
        values = tl.load(
            projected_ptr
            + sorted_slots[:, None] * stride_pm
            + offs_h[None, :] * stride_ph,
            mask=h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum(values * route[:, None], axis=0)
        if ADD_RESIDUAL:
            acc += tl.load(
                residual_ptr + row * stride_xm + offs_h * stride_xh,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            out_ptr + row * stride_om + offs_h * stride_oh,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )


def gemma4_grouped_mm_prefill(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    workspace: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    """Run the exact Gemma 4 A4B B16 expert prefill candidate."""
    if workspace is not None:
        workspace["grouped_mm_prefill_active"] = 0
        workspace["grouped_mm_prefill_route_pack"] = 0
        workspace["grouped_mm_prefill_deterministic_reduce"] = 0
    if not gemma4_grouped_mm_prefill_prefers_shape(
        hidden_states,
        gate_up_proj,
        down_proj,
        selected_experts,
        routing_weights,
    ):
        raise RuntimeError("Gemma4 grouped-MM prefill is not eligible for this shape")
    if residual is not None and tuple(residual.shape) != tuple(hidden_states.shape):
        raise ValueError("residual must have the same shape as hidden_states")
    if out is not None and tuple(out.shape) != tuple(hidden_states.shape):
        raise ValueError("out must have the same shape as hidden_states")

    rows = int(hidden_states.shape[0])
    hidden_dim = int(hidden_states.shape[1])
    top_k = int(selected_experts.shape[1])
    assignments = rows * top_k
    intermediate_dim = int(gate_up_proj.shape[1] // 2)
    num_experts = int(gate_up_proj.shape[0])
    device = hidden_states.device

    hidden = (
        hidden_states
        if hidden_states.is_contiguous()
        else hidden_states.contiguous()
    )
    experts = selected_experts.reshape(-1)
    if experts.dtype != torch.int64:
        experts = experts.to(torch.int64)
    if not experts.is_contiguous():
        experts = experts.contiguous()
    route = (
        routing_weights
        if routing_weights.is_contiguous()
        else routing_weights.contiguous()
    )

    sorted_tokens = _workspace_tensor(
        workspace,
        "grouped_mm_sorted_tokens",
        (assignments,),
        device=device,
        dtype=torch.int64,
    )
    inverse_slots = _workspace_tensor(
        workspace,
        "grouped_mm_inverse_slots",
        (assignments,),
        device=device,
        dtype=torch.int64,
    )
    offsets = _workspace_tensor(
        workspace,
        "grouped_mm_offsets",
        (num_experts,),
        device=device,
        dtype=torch.int32,
    )
    block_assignments = int(triton.next_power_of_2(assignments))
    _gemma4_grouped_mm_route_pack_kernel[(num_experts,)](
        experts,
        sorted_tokens,
        inverse_slots,
        offsets,
        ASSIGNMENTS=assignments,
        TOP_K=top_k,
        BLOCK_ASSIGNMENTS=block_assignments,
        num_warps=8,
        num_stages=1,
    )

    sorted_hidden = _workspace_tensor(
        workspace,
        "grouped_mm_sorted_hidden",
        (assignments, hidden_dim),
        device=device,
        dtype=hidden.dtype,
    )
    torch.index_select(hidden, 0, sorted_tokens, out=sorted_hidden)

    grouped_mm = getattr(torch, "_grouped_mm")
    gate_up = grouped_mm(
        sorted_hidden,
        gate_up_proj.transpose(1, 2),
        offs=offsets,
    )
    if tuple(gate_up.shape) != (assignments, 2 * intermediate_dim):
        raise RuntimeError(
            "Gemma4 grouped-MM gate/up returned an unexpected shape: "
            f"{tuple(gate_up.shape)}"
        )

    activated = _workspace_tensor(
        workspace,
        "grouped_mm_activated",
        (assignments, intermediate_dim),
        device=device,
        dtype=hidden.dtype,
    )
    block_m = 32
    block_n = 128
    _gemma4_grouped_mm_geglu_kernel[
        (triton.cdiv(assignments, block_m), triton.cdiv(intermediate_dim, block_n))
    ](
        gate_up,
        activated,
        gate_up.stride(0),
        gate_up.stride(1),
        activated.stride(0),
        activated.stride(1),
        ROWS=assignments,
        I=intermediate_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )

    projected = grouped_mm(
        activated,
        down_proj.transpose(1, 2),
        offs=offsets,
    )
    if tuple(projected.shape) != (assignments, hidden_dim):
        raise RuntimeError(
            "Gemma4 grouped-MM down projection returned an unexpected shape: "
            f"{tuple(projected.shape)}"
        )

    final = out if out is not None else torch.empty_like(hidden_states)
    residual_ptr = residual if residual is not None else hidden
    reduce_block_n = 256
    _gemma4_grouped_mm_topk_reduce_kernel[
        (rows, triton.cdiv(hidden_dim, reduce_block_n))
    ](
        projected,
        inverse_slots,
        route,
        residual_ptr,
        final,
        projected.stride(0),
        projected.stride(1),
        route.stride(0),
        route.stride(1),
        residual_ptr.stride(0),
        residual_ptr.stride(1),
        final.stride(0),
        final.stride(1),
        H=hidden_dim,
        TOP_K=top_k,
        ADD_RESIDUAL=bool(residual is not None),
        BLOCK_N=reduce_block_n,
        num_warps=4,
        num_stages=1,
    )
    if workspace is not None:
        workspace["grouped_mm_prefill_active"] = 1
        workspace["grouped_mm_prefill_route_pack"] = 1
        workspace["grouped_mm_prefill_deterministic_reduce"] = 1
        workspace["grouped_mm_prefill_assignments"] = assignments
        workspace["grouped_mm_prefill_num_experts"] = num_experts
        workspace["grouped_mm_prefill_activation"] = "gelu_pytorch_tanh"
    return final


__all__ = [
    "gemma4_grouped_mm_prefill",
    "gemma4_grouped_mm_prefill_available",
    "gemma4_grouped_mm_prefill_prefers_shape",
]

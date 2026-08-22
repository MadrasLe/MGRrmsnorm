"""
Qwen3 MoE grouped decode kernels.

This path is for decode/small-M inference. It batches all routed expert
assignments for one layer into two Triton launches:
  1. grouped expert gate/up projection
  2. fused SwiGLU + down projection + weighted accumulation

Large Qwen3 prefill can optionally use the segmented Triton path below. It
groups assignments by expert, builds a compact tile list, and launches one
gate/up grouped GEMM plus one down/accum grouped GEMM per layer without the
padding used by the PyTorch batched-bucket fallback. The hot path uses a
counting scatter instead of a full argsort when Triton is available.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _is_power_of_2(value: int) -> bool:
    value = int(value)
    return value > 0 and (value & (value - 1)) == 0


_CFG_MAX_ASSIGNMENTS = max(1, _env_int("MEGAGEMM_QWEN3_MOE_GROUPED_MAX_ASSIGNMENTS", 64))
_CFG_FORCE_TRITON = _env_bool("MEGAGEMM_QWEN3_MOE_GROUPED_FORCE_TRITON", False)
_CFG_BLOCK_N = max(1, _env_int("MEGAGEMM_QWEN3_MOE_GROUPED_BLOCK_N", 64))
_CFG_BLOCK_K = max(1, _env_int("MEGAGEMM_QWEN3_MOE_GROUPED_BLOCK_K", 128))
_CFG_NUM_WARPS = max(1, _env_int("MEGAGEMM_QWEN3_MOE_GROUPED_NUM_WARPS", 4))
_CFG_NUM_STAGES = max(1, _env_int("MEGAGEMM_QWEN3_MOE_GROUPED_NUM_STAGES", 2))
_CFG_FUSED_ROUTER = _env_bool("MEGAGEMM_QWEN3_MOE_FUSED_ROUTER", False)
_CFG_ROUTER_BLOCK_K = max(16, _env_int("MEGAGEMM_QWEN3_MOE_ROUTER_BLOCK_K", 64))
_CFG_ROUTER_K_SPLITS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_ROUTER_K_SPLITS", 1),
)
if not _is_power_of_2(_CFG_ROUTER_K_SPLITS):
    _CFG_ROUTER_K_SPLITS = 1
_CFG_FUSED_ROUTER_MAX_ROWS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_FUSED_ROUTER_MAX_ROWS", 1),
)
_CFG_TOKEN_ACCUM = _env_bool("MEGAGEMM_QWEN3_MOE_TOKEN_ACCUM", False)
_CFG_TOKEN_ACCUM_MIN_ROWS = max(1, _env_int("MEGAGEMM_QWEN3_MOE_TOKEN_ACCUM_MIN_ROWS", 1))
_CFG_GROUPED_FUSED_GATE = _env_bool("MEGAGEMM_QWEN3_MOE_GROUPED_FUSED_GATE", False)
_CFG_GROUPED_DOT = _env_bool("MEGAGEMM_QWEN3_MOE_GROUPED_DOT", False)
_CFG_GROUPED_DOT_ALLOW_CUDA_GRAPHS = _env_bool(
    "MEGAGEMM_QWEN3_MOE_GROUPED_DOT_ALLOW_CUDA_GRAPHS",
    False,
)
_CFG_DECODE_CUDA_GRAPHS = _env_bool("MEGAGEMM_DECODE_CUDA_GRAPHS", False)
_CFG_EXPERT_GROUPED_DECODE = _env_bool("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_DECODE", False)
_CFG_SHARED_ROUTE_DECODE = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_DECODE",
    False,
)
_CFG_SHARED_ROUTE_BATCH_MAX_ROWS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_BATCH_MAX_ROWS", 1),
)
_CFG_SHARED_ROUTE_ASSUME_IDENTICAL = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_ASSUME_IDENTICAL",
    False,
)
_CFG_SINGLE_ROW_GEMV = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SINGLE_ROW_GEMV",
    False,
)
_CFG_ROUTE_MATRIX_DECODE = _env_bool(
    "MEGAGEMM_QWEN3_MOE_ROUTE_MATRIX_DECODE",
    False,
)
_CFG_ROUTE_MATRIX_MAX_ROWS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_ROUTE_MATRIX_MAX_ROWS", 16),
)
# Shared-route batch values above 1 are only valid when every row selects the
# same experts in the same top-k order. The route-matrix path is a correctness
# fallback for heterogeneous decode batches; compact grouped decode is faster.
_CFG_SHARED_ROUTE_TOKEN_ACCUM = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_TOKEN_ACCUM",
    False,
)
_CFG_SHARED_ROUTE_PARTIAL_REDUCE = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_PARTIAL_REDUCE",
    False,
)
_CFG_SHARED_ROUTE_COALESCED_WEIGHTS = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_COALESCED_WEIGHTS",
    False,
)
_CFG_SHARED_ROUTE_TOKEN_ACCUM_NUM_WARPS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_TOKEN_ACCUM_NUM_WARPS", _CFG_NUM_WARPS),
)
_CFG_SHARED_ROUTE_BLOCK_M = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_BLOCK_M", 1),
)
_CFG_SHARED_ROUTE_GATE_BLOCK_N = max(
    16,
    _env_int("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_GATE_BLOCK_N", _CFG_BLOCK_N),
)
_CFG_SHARED_ROUTE_GATE_K_SPLITS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_GATE_K_SPLITS", 4),
)
if not _is_power_of_2(_CFG_SHARED_ROUTE_GATE_K_SPLITS):
    _CFG_SHARED_ROUTE_GATE_K_SPLITS = 1
_CFG_SHARED_ROUTE_DOWN_BLOCK_N = max(
    16,
    _env_int("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_DOWN_BLOCK_N", _CFG_BLOCK_N),
)
_CFG_SHARED_ROUTE_SPLIT_GATE = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_SPLIT_GATE",
    False,
)
_CFG_SHARED_ROUTE_SPLIT_GATE_BLOCK_M = max(
    16,
    _env_int("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_SPLIT_GATE_BLOCK_M", 16),
)
_CFG_SHARED_ROUTE_SPLIT_GATE_NUM_STAGES = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SHARED_ROUTE_SPLIT_GATE_NUM_STAGES", 4),
)
_CFG_EXPERT_GROUPED_GENERAL_DECODE = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_GENERAL_DECODE",
    False,
)
_CFG_EXPERT_GROUPED_DENSE_DECODE = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_DENSE_DECODE",
    False,
)
_CFG_EXPERT_GROUPED_COMPACT_DECODE = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DECODE",
    False,
)
_CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_FUSED_PACK",
    True,
)
_CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE",
    False,
)
_CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST",
    False,
)
_CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT",
    False,
)
_CFG_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK",
    True,
)
_CFG_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS",
    False,
)
_CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM",
    False,
)
_CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N", _CFG_BLOCK_N),
)
_CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N", max(_CFG_BLOCK_N, 128)),
)
_CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS = max(
    1,
    _env_int(
        "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_NUM_WARPS",
        _CFG_NUM_WARPS,
    ),
)
_CFG_EXPERT_GROUPED_COMPACT_NUM_STAGES = max(
    1,
    _env_int(
        "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_NUM_STAGES",
        3,
    ),
)
_CFG_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES = max(
    1,
    _env_int(
        "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES",
        _CFG_EXPERT_GROUPED_COMPACT_NUM_STAGES,
    ),
)
_CFG_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES = max(
    1,
    _env_int(
        "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES",
        _CFG_EXPERT_GROUPED_COMPACT_NUM_STAGES,
    ),
)
_CFG_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM = max(
    1,
    _env_int(
        "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM",
        1,
    ),
)
_CFG_EXPERT_GROUPED_COMPACT_PAIRED_GATE_UP_DOT = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_PAIRED_GATE_UP_DOT",
    False,
)
_CFG_EXPERT_GROUPED_COMPACT_SPLIT_GATE_UP = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_SPLIT_GATE_UP",
    False,
)
_CFG_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT",
    False,
)
_CFG_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID",
    False,
)
_CFG_EXPERT_GROUPED_COMPACT_L2_GROUP_SIZE = max(
    1,
    _env_int(
        "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_L2_GROUP_SIZE",
        8,
    ),
)
_CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT = _env_bool(
    "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DIRECT_OUT",
    False,
)
_CFG_EXPERT_GROUPED_MIN_ROWS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_MIN_ROWS", 1),
)
_CFG_EXPERT_GROUPED_MAX_ROWS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_MAX_ROWS", 16),
)
_CFG_EXPERT_GROUPED_BLOCK_M = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_BLOCK_M", 16),
)
_CFG_EXPERT_GROUPED_ROUTE_BLOCK = max(
    32,
    _env_int("MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_ROUTE_BLOCK", 128),
)
_CFG_INT8_DECODE = _env_bool("MEGAGEMM_QWEN3_MOE_INT8_DECODE", True)
_CFG_SEGMENTED_PREFILL = _env_bool("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL", False)
_CFG_SEGMENTED_PREFILL_DENSE_GRID = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_DENSE_GRID",
    False,
)
_CFG_SEGMENTED_PREFILL_FUSED_GATE = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_FUSED_GATE",
    False,
)
_CFG_SEGMENTED_PREFILL_ASYNC_TILES = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_ASYNC_TILES",
    True,
)
_CFG_SEGMENTED_PREFILL_ASYNC_TILES_MAX_ASSIGNMENTS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_ASYNC_TILES_MAX_ASSIGNMENTS", 4096),
)
_CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_PARTIAL_REDUCE",
    True,
)
_CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS = max(
    1,
    _env_int(
        "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS",
        4096,
    ),
)
_CFG_SEGMENTED_PREFILL_PARTIAL_CACHE_MAX_ASSIGNMENTS = max(
    0,
    _env_int(
        "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_PARTIAL_CACHE_MAX_ASSIGNMENTS",
        512,
    ),
)
_CFG_SEGMENTED_PREFILL_ROUTE_SCATTER = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_ROUTE_SCATTER",
    True,
)
_CFG_SEGMENTED_PREFILL_ROUTE_BLOCK = max(
    128,
    _env_int("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_ROUTE_BLOCK", 256),
)
_CFG_SEGMENTED_PREFILL_MIN_ASSIGNMENTS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_MIN_ASSIGNMENTS", 4096),
)
_CFG_SEGMENTED_PREFILL_BLOCK_M = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_BLOCK_M", 32),
)
_CFG_SEGMENTED_PREFILL_BLOCK_N = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_BLOCK_N", 128),
)
_CFG_SEGMENTED_PREFILL_BLOCK_K = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_BLOCK_K", 64),
)
_CFG_SEGMENTED_PREFILL_FUSED_GATE_BLOCK_N = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_FUSED_GATE_BLOCK_N", 64),
)
_CFG_SEGMENTED_PREFILL_FIXED_ROUTE_PACK = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_FIXED_ROUTE_PACK",
    False,
)
_CFG_SEGMENTED_PREFILL_COMPACT_ROUTE_PACK = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_COMPACT_ROUTE_PACK",
    False,
)
_CFG_SEGMENTED_PREFILL_SINGLE_ACCUMULATOR = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_SINGLE_ACCUMULATOR",
    False,
)
_CFG_SEGMENTED_PREFILL_SORTED_PARTIAL = _env_bool(
    "MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_SORTED_PARTIAL",
    False,
)
_CFG_SEGMENTED_PREFILL_GROUP_SIZE_M = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_GROUP_SIZE_M", 8),
)
_CFG_SEGMENTED_PREFILL_NUM_WARPS = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_NUM_WARPS", 4),
)
_CFG_SEGMENTED_PREFILL_NUM_STAGES = max(
    1,
    _env_int("MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_NUM_STAGES", 3),
)

_ACT_SILU = 0
_ACT_GELU_TANH = 1


def _activation_id(activation: str) -> int:
    act = str(activation).strip().lower()
    if act in {"silu", "swiglu"}:
        return _ACT_SILU
    if act in {"gelu", "gelu_tanh", "gelu_pytorch_tanh", "geglu"}:
        return _ACT_GELU_TANH
    raise ValueError(f"Unsupported Qwen3 MoE activation: {activation}")


def qwen3_moe_grouped_prefers_triton_shape(
    rows: int,
    top_k: int,
    hidden_dim: int,
    intermediate_dim: int,
    *,
    max_assignments: Optional[int] = None,
) -> bool:
    assignments = int(rows) * int(top_k)
    assignment_limit = (
        int(_CFG_MAX_ASSIGNMENTS)
        if max_assignments is None
        else max(1, int(max_assignments))
    )
    if assignments <= 0:
        return False
    if _CFG_FORCE_TRITON:
        return True
    if assignments > assignment_limit:
        return False
    if hidden_dim <= 0 or intermediate_dim <= 0:
        return False
    return True


def qwen3_moe_segmented_prefers_triton_shape(
    rows: int,
    top_k: int,
    hidden_dim: int,
    intermediate_dim: int,
    *,
    force: bool = False,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
) -> bool:
    assignments = int(rows) * int(top_k)
    selected_block_m = int(_CFG_SEGMENTED_PREFILL_BLOCK_M if block_m is None else block_m)
    selected_block_n = int(_CFG_SEGMENTED_PREFILL_BLOCK_N if block_n is None else block_n)
    selected_block_k = int(_CFG_SEGMENTED_PREFILL_BLOCK_K if block_k is None else block_k)
    decode_general = bool(
        _CFG_EXPERT_GROUPED_GENERAL_DECODE
        and int(rows) >= _CFG_EXPERT_GROUPED_MIN_ROWS
        and int(rows) <= _CFG_EXPERT_GROUPED_MAX_ROWS
    )
    if not _HAS_TRITON:
        return False
    if not (force or _CFG_SEGMENTED_PREFILL or decode_general):
        return False
    if assignments < _CFG_SEGMENTED_PREFILL_MIN_ASSIGNMENTS and not (force or decode_general):
        return False
    if hidden_dim <= 0 or intermediate_dim <= 0:
        return False
    return (
        _is_power_of_2(selected_block_m)
        and _is_power_of_2(selected_block_n)
        and _is_power_of_2(selected_block_k)
    )


def _fallback_grouped_moe(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    activation: str,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    final = torch.zeros_like(hidden_states) if out is None else out.zero_()
    rows, top_k = selected_experts.shape
    for token_idx in range(rows):
        token = hidden_states[token_idx : token_idx + 1]
        for top_idx in range(top_k):
            expert_idx = int(selected_experts[token_idx, top_idx])
            gate_up = F.linear(token, gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            if _activation_id(activation) == _ACT_GELU_TANH:
                activated = F.gelu(gate, approximate="tanh") * up
            else:
                activated = F.silu(gate) * up
            projected = F.linear(activated, down_proj[expert_idx])
            final[token_idx].add_(
                projected.squeeze(0).to(final.dtype) * routing_weights[token_idx, top_idx].to(final.dtype)
            )
    return final


def _dequant_expert(weight_int8: torch.Tensor, scale: torch.Tensor, expert_idx: int, dtype: torch.dtype) -> torch.Tensor:
    return weight_int8[expert_idx].to(dtype) * scale[expert_idx].unsqueeze(1).to(dtype)


def _fallback_grouped_moe_int8(
    hidden_states: torch.Tensor,
    gate_up_int8: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_int8: torch.Tensor,
    down_scale: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    activation: str,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    final = torch.zeros_like(hidden_states) if out is None else out.zero_()
    rows, top_k = selected_experts.shape
    dtype = hidden_states.dtype
    act_id = _activation_id(activation)
    for token_idx in range(rows):
        token = hidden_states[token_idx : token_idx + 1]
        for top_idx in range(top_k):
            expert_idx = int(selected_experts[token_idx, top_idx])
            gate_up_w = _dequant_expert(gate_up_int8, gate_up_scale, expert_idx, dtype)
            gate_up = F.linear(token, gate_up_w)
            gate, up = gate_up.chunk(2, dim=-1)
            if act_id == _ACT_GELU_TANH:
                activated = F.gelu(gate, approximate="tanh") * up
            else:
                activated = F.silu(gate) * up
            down_w = _dequant_expert(down_int8, down_scale, expert_idx, dtype)
            projected = F.linear(activated, down_w)
            final[token_idx].add_(
                projected.squeeze(0).to(final.dtype) * routing_weights[token_idx, top_idx].to(final.dtype)
            )
    return final


def _workspace_tensor(
    workspace: Optional[dict[str, torch.Tensor]],
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    zero: bool = False,
) -> torch.Tensor:
    if workspace is None:
        tensor = torch.empty(shape, device=device, dtype=dtype)
    else:
        tensor = workspace.get(name)
        if (
            tensor is None
            or tuple(tensor.shape) != tuple(shape)
            or tensor.device != device
            or tensor.dtype != dtype
        ):
            tensor = torch.empty(shape, device=device, dtype=dtype)
            workspace[name] = tensor
    if zero:
        tensor.zero_()
    return tensor


def _segmented_prefill_graph_partial_name(
    assignments: int,
    hidden_dim: int,
) -> str:
    return f"segmented_graph_partial_out_{int(assignments)}_{int(hidden_dim)}"


def qwen3_moe_prepare_segmented_prefill_graph_workspace(
    workspace: dict[str, torch.Tensor],
    *,
    assignments: int,
    hidden_dim: int,
    device: torch.device,
    num_experts: int = 0,
    block_m: int = 0,
    route_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Allocate shape-stable reduction and route buffers used by graph replay."""
    if workspace is None:
        raise ValueError("A persistent workspace is required for CUDA graph prefill")
    name = _segmented_prefill_graph_partial_name(assignments, hidden_dim)
    partial = _workspace_tensor(
        workspace,
        name,
        (int(assignments), int(hidden_dim)),
        device=device,
        dtype=torch.float32,
    )
    workspace["segmented_prefill_graph_partial_name"] = name
    workspace["segmented_prefill_graph_partial_bytes"] = int(
        partial.numel() * partial.element_size()
    )
    workspace["segmented_prefill_graph_partial_dtype"] = str(partial.dtype)
    if int(num_experts) > 0 and int(block_m) > 0 and route_dtype is not None:
        max_tiles = _segmented_tile_upper_bound(
            int(assignments),
            int(num_experts),
            int(block_m),
        )
        route_specs = (
            ("segmented_compact_sorted_tokens", (int(assignments),), torch.int64),
            ("segmented_compact_sorted_route", (int(assignments),), route_dtype),
            ("segmented_compact_sorted_slots", (int(assignments),), torch.int64),
            ("segmented_compact_tile_experts", (max_tiles,), torch.int64),
            ("segmented_compact_tile_starts", (max_tiles,), torch.int64),
            ("segmented_compact_tile_lengths", (max_tiles,), torch.int64),
            ("segmented_compact_num_tiles", (1,), torch.int32),
            ("segmented_graph_route_counts", (int(num_experts),), torch.int32),
            ("segmented_graph_route_starts", (int(num_experts),), torch.int32),
            (
                "segmented_graph_route_tiles_per_expert",
                (int(num_experts),),
                torch.int32,
            ),
            ("segmented_graph_route_tile_offsets", (int(num_experts),), torch.int32),
        )
        for name, shape, dtype in route_specs:
            _workspace_tensor(
                workspace,
                name,
                shape,
                device=device,
                dtype=dtype,
            )
        workspace["segmented_prefill_graph_route_workspace_bytes"] = sum(
            int(workspace[name].numel() * workspace[name].element_size())
            for name, _, _ in route_specs
        )
    return partial


def _cuda_graph_capture_active(tensor: torch.Tensor) -> bool:
    if not tensor.is_cuda:
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _workspace_token_ids(
    workspace: Optional[dict[str, torch.Tensor]],
    rows: int,
    top_k: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if workspace is None:
        return torch.arange(rows, device=device, dtype=torch.int64).repeat_interleave(top_k)

    tensor = workspace.get("token_ids")
    shape = (rows * top_k,)
    if (
        tensor is None
        or tuple(tensor.shape) != shape
        or tensor.device != device
        or tensor.dtype != torch.int64
    ):
            tensor = torch.arange(rows, device=device, dtype=torch.int64).repeat_interleave(top_k)
            workspace["token_ids"] = tensor
    return tensor


def _build_segmented_tile_tensors(
    counts: torch.Tensor,
    *,
    block_m: int,
    workspace: Optional[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    counts_cpu = counts.detach().cpu().tolist()
    tile_experts_cpu: list[int] = []
    tile_starts_cpu: list[int] = []
    tile_lengths_cpu: list[int] = []

    cursor = 0
    for expert_idx, raw_count in enumerate(counts_cpu):
        count = int(raw_count)
        if count <= 0:
            continue
        local = 0
        while local < count:
            tile_len = min(int(block_m), count - local)
            tile_experts_cpu.append(int(expert_idx))
            tile_starts_cpu.append(int(cursor + local))
            tile_lengths_cpu.append(int(tile_len))
            local += int(block_m)
        cursor += count

    device = counts.device
    if not tile_experts_cpu:
        empty = torch.empty((0,), device=device, dtype=torch.int64)
        return empty, empty, empty

    tile_shape = (len(tile_experts_cpu),)
    if workspace is None:
        tile_experts = torch.tensor(tile_experts_cpu, device=device, dtype=torch.int64)
        tile_starts = torch.tensor(tile_starts_cpu, device=device, dtype=torch.int64)
        tile_lengths = torch.tensor(tile_lengths_cpu, device=device, dtype=torch.int64)
    else:
        tile_experts = _workspace_tensor(
            workspace,
            "segmented_tile_experts",
            tile_shape,
            device=device,
            dtype=torch.int64,
        )
        tile_starts = _workspace_tensor(
            workspace,
            "segmented_tile_starts",
            tile_shape,
            device=device,
            dtype=torch.int64,
        )
        tile_lengths = _workspace_tensor(
            workspace,
            "segmented_tile_lengths",
            tile_shape,
            device=device,
            dtype=torch.int64,
        )
        tile_experts.copy_(torch.tensor(tile_experts_cpu, device=device, dtype=torch.int64))
        tile_starts.copy_(torch.tensor(tile_starts_cpu, device=device, dtype=torch.int64))
        tile_lengths.copy_(torch.tensor(tile_lengths_cpu, device=device, dtype=torch.int64))
        workspace["segmented_prefill_last_tiles"] = int(len(tile_experts_cpu))
        workspace["segmented_prefill_last_assignments"] = int(cursor)

    return tile_experts, tile_starts, tile_lengths


def _build_segmented_tile_tensors_gpu(
    counts: torch.Tensor,
    *,
    block_m: int,
    workspace: Optional[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build compact expert tile metadata with GPU tensor ops.

    This keeps the segmented prefill grid compact: sum_e ceil(count[e]/BLOCK_M)
    tiles instead of num_experts * ceil(max(count)/BLOCK_M). The only scalar
    sync is the final tile count required by the Triton launch grid.
    """
    device = counts.device
    tiles_per_expert = torch.div(
        counts + int(block_m) - 1,
        int(block_m),
        rounding_mode="floor",
    ).to(torch.int64)
    num_tiles = int(tiles_per_expert.sum().item())
    if workspace is not None:
        workspace["segmented_prefill_last_tiles"] = int(num_tiles)
    if num_tiles <= 0:
        empty = torch.empty((0,), device=device, dtype=torch.int64)
        return empty, empty, empty

    expert_ids = torch.arange(int(counts.numel()), device=device, dtype=torch.int64)
    tile_experts = torch.repeat_interleave(
        expert_ids,
        tiles_per_expert,
        output_size=num_tiles,
    )
    expert_tile_starts = torch.cumsum(tiles_per_expert, dim=0) - tiles_per_expert
    tile_ord = torch.arange(num_tiles, device=device, dtype=torch.int64)
    local_tile = tile_ord - expert_tile_starts.index_select(0, tile_experts)
    starts = torch.cumsum(counts, dim=0) - counts
    tile_starts = starts.index_select(0, tile_experts) + local_tile * int(block_m)
    remaining = counts.index_select(0, tile_experts) - local_tile * int(block_m)
    tile_lengths = torch.clamp(remaining, min=0, max=int(block_m)).to(torch.int64)

    return tile_experts, tile_starts, tile_lengths


def _segmented_tile_upper_bound(assignments: int, num_experts: int, block_m: int) -> int:
    """Tight upper bound for sum_e ceil(count[e] / block_m)."""
    assignments = max(0, int(assignments))
    num_experts = max(0, int(num_experts))
    block_m = max(1, int(block_m))
    active = min(assignments, num_experts)
    if active == 0:
        return 0
    return active + (assignments - active) // block_m


def _build_segmented_tile_tensors_gpu_async(
    counts: torch.Tensor,
    starts: torch.Tensor,
    *,
    assignments: int,
    block_m: int,
    workspace: Optional[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build compact tile metadata without synchronizing a GPU scalar to Python."""
    num_experts = int(counts.numel())
    max_tiles = _segmented_tile_upper_bound(assignments, num_experts, block_m)
    if max_tiles <= 0:
        empty = torch.empty((0,), device=counts.device, dtype=torch.int64)
        zero = torch.zeros((1,), device=counts.device, dtype=torch.int64)
        return empty, empty, empty, zero, 0

    tiles_per_expert = torch.div(
        counts + int(block_m) - 1,
        int(block_m),
        rounding_mode="floor",
    ).to(torch.int64)
    tile_offsets = torch.cumsum(tiles_per_expert, dim=0) - tiles_per_expert

    if workspace is None:
        tile_experts = torch.empty((max_tiles,), device=counts.device, dtype=torch.int64)
        tile_starts = torch.empty((max_tiles,), device=counts.device, dtype=torch.int64)
        tile_lengths = torch.empty((max_tiles,), device=counts.device, dtype=torch.int64)
        num_tiles = torch.empty((1,), device=counts.device, dtype=torch.int64)
    else:
        tile_experts = _workspace_tensor(
            workspace,
            "segmented_async_tile_experts",
            (max_tiles,),
            device=counts.device,
            dtype=torch.int64,
        )
        tile_starts = _workspace_tensor(
            workspace,
            "segmented_async_tile_starts",
            (max_tiles,),
            device=counts.device,
            dtype=torch.int64,
        )
        tile_lengths = _workspace_tensor(
            workspace,
            "segmented_async_tile_lengths",
            (max_tiles,),
            device=counts.device,
            dtype=torch.int64,
        )
        num_tiles = _workspace_tensor(
            workspace,
            "segmented_async_num_tiles",
            (1,),
            device=counts.device,
            dtype=torch.int64,
        )

    search_steps = max(1, int(num_experts).bit_length())
    _qwen3_moe_build_compact_tiles_kernel[(max_tiles,)](
        counts,
        starts,
        tiles_per_expert,
        tile_offsets,
        tile_experts,
        tile_starts,
        tile_lengths,
        num_tiles,
        int(num_experts),
        BLOCK_M=int(block_m),
        SEARCH_STEPS=int(search_steps),
        num_warps=1,
        num_stages=1,
    )
    if workspace is not None:
        workspace["segmented_prefill_async_tiles"] = 1
        workspace["segmented_prefill_max_tiles"] = int(max_tiles)
    return tile_experts, tile_starts, tile_lengths, num_tiles, int(max_tiles)


if _HAS_TRITON:
    @triton.jit
    def _qwen3_moe_build_compact_tiles_kernel(
        counts_ptr,
        starts_ptr,
        tiles_per_expert_ptr,
        tile_offsets_ptr,
        tile_experts_ptr,
        tile_starts_ptr,
        tile_lengths_ptr,
        num_tiles_ptr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
    ):
        tile = tl.program_id(0).to(tl.int64)
        last_expert = NUM_EXPERTS - 1
        last_offset = tl.load(tile_offsets_ptr + last_expert).to(tl.int64)
        last_tiles = tl.load(tiles_per_expert_ptr + last_expert).to(tl.int64)
        total_tiles = last_offset + last_tiles
        valid = tile < total_tiles
        tl.store(num_tiles_ptr, total_tiles, mask=tile == 0)

        # Find the first expert whose inclusive tile prefix is greater than the
        # compact tile id. This keeps the launch at O(actual tiles) and avoids
        # synchronizing total_tiles back to Python for the grid size.
        low = tl.full((), 0, tl.int64)
        high = tl.full((), NUM_EXPERTS, tl.int64)
        for _ in tl.static_range(0, SEARCH_STEPS):
            searching = low < high
            middle = tl.minimum((low + high) // 2, NUM_EXPERTS - 1)
            middle_offset = tl.load(tile_offsets_ptr + middle).to(tl.int64)
            middle_tiles = tl.load(tiles_per_expert_ptr + middle).to(tl.int64)
            before_or_at_tile = middle_offset + middle_tiles <= tile
            next_low = tl.where(before_or_at_tile, middle + 1, low)
            next_high = tl.where(before_or_at_tile, high, middle)
            low = tl.where(searching, next_low, low)
            high = tl.where(searching, next_high, high)

        expert = tl.minimum(low, last_expert)
        tile_offset = tl.load(tile_offsets_ptr + expert).to(tl.int64)
        local_tile = tile - tile_offset
        count = tl.load(counts_ptr + expert).to(tl.int64)
        start = tl.load(starts_ptr + expert).to(tl.int64)
        remaining = count - local_tile * BLOCK_M
        tile_len = tl.minimum(remaining, BLOCK_M)
        tl.store(tile_experts_ptr + tile, expert, mask=valid)
        tl.store(
            tile_starts_ptr + tile,
            start + local_tile * BLOCK_M,
            mask=valid,
        )
        tl.store(tile_lengths_ptr + tile, tile_len, mask=valid)

    @triton.jit
    def _qwen3_moe_route_scatter_by_expert_kernel(
        experts_ptr,
        route_ptr,
        starts_ptr,
        counters_ptr,
        sorted_tokens_ptr,
        sorted_route_ptr,
        sorted_slots_ptr,
        assignments: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < assignments
        expert = tl.load(experts_ptr + offs, mask=mask, other=0).to(tl.int64)
        route = tl.load(route_ptr + offs, mask=mask, other=0.0)
        rank = tl.atomic_add(
            counters_ptr + expert,
            1,
            sem="relaxed",
            mask=mask,
        ).to(tl.int64)
        out_pos = tl.load(starts_ptr + expert, mask=mask, other=0).to(tl.int64) + rank
        token = offs // TOP_K
        tl.store(sorted_tokens_ptr + out_pos, token, mask=mask)
        tl.store(sorted_route_ptr + out_pos, route, mask=mask)
        tl.store(sorted_slots_ptr + out_pos, offs, mask=mask)

    @triton.jit
    def _qwen3_moe_invert_sorted_slots_kernel(
        sorted_slots_ptr,
        slot_to_sorted_ptr,
        ASSIGNMENTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        sorted_pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = sorted_pos < ASSIGNMENTS
        original_slot = tl.load(
            sorted_slots_ptr + sorted_pos,
            mask=mask,
            other=0,
        ).to(tl.int64)
        tl.store(slot_to_sorted_ptr + original_slot, sorted_pos, mask=mask)

    @triton.jit
    def _qwen3_moe_fixed_route_pack_kernel(
        experts_ptr,
        route_ptr,
        sorted_tokens_ptr,
        sorted_route_ptr,
        sorted_slots_ptr,
        tile_experts_ptr,
        tile_starts_ptr,
        tile_lengths_ptr,
        ASSIGNMENTS: tl.constexpr,
        ROWS: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_ASSIGNMENTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        M_TILES: tl.constexpr,
    ):
        expert = tl.program_id(0)
        offs = tl.arange(0, BLOCK_ASSIGNMENTS)
        mask = offs < ASSIGNMENTS
        routed_expert = tl.load(experts_ptr + offs, mask=mask, other=-1).to(tl.int64)
        matches = mask & (routed_expert == expert)
        prefix = tl.cumsum(matches.to(tl.int32), axis=0)
        local_slot = prefix - 1
        count = tl.sum(matches.to(tl.int32), axis=0)
        out_pos = expert * ROWS + local_slot
        route = tl.load(route_ptr + offs, mask=matches, other=0.0)

        tl.store(sorted_tokens_ptr + out_pos, offs // TOP_K, mask=matches)
        tl.store(sorted_route_ptr + out_pos, route, mask=matches)
        tl.store(sorted_slots_ptr + out_pos, offs, mask=matches)

        for local_tile in tl.static_range(0, M_TILES):
            tile_idx = expert * M_TILES + local_tile
            local_start = local_tile * BLOCK_M
            tile_len = tl.maximum(0, tl.minimum(count - local_start, BLOCK_M))
            tl.store(tile_experts_ptr + tile_idx, expert)
            tl.store(tile_starts_ptr + tile_idx, expert * ROWS + local_start)
            tl.store(tile_lengths_ptr + tile_idx, tile_len)

    @triton.jit
    def _qwen3_moe_compact_route_pack_kernel(
        experts_ptr,
        route_ptr,
        num_tiles_ptr,
        sorted_tokens_ptr,
        sorted_route_ptr,
        sorted_slots_ptr,
        tile_experts_ptr,
        tile_starts_ptr,
        tile_lengths_ptr,
        ASSIGNMENTS: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_ASSIGNMENTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        MAX_TILES_PER_EXPERT: tl.constexpr,
    ):
        expert = tl.program_id(0)
        offs = tl.arange(0, BLOCK_ASSIGNMENTS)
        mask = offs < ASSIGNMENTS
        routed_expert = tl.load(
            experts_ptr + offs,
            mask=mask,
            other=-1,
        ).to(tl.int64)
        matches = mask & (routed_expert == expert)
        prefix = tl.cumsum(matches.to(tl.int32), axis=0)
        local_slot = prefix - 1
        count = tl.sum(matches.to(tl.int32), axis=0)
        start = tl.sum(
            (mask & (routed_expert < expert)).to(tl.int32),
            axis=0,
        )
        tiles = (count + BLOCK_M - 1) // BLOCK_M
        tile_offset = tl.atomic_add(
            num_tiles_ptr,
            tiles,
            sem="relaxed",
        ).to(tl.int32)
        out_pos = start + local_slot
        route = tl.load(route_ptr + offs, mask=matches, other=0.0)

        tl.store(sorted_tokens_ptr + out_pos, offs // TOP_K, mask=matches)
        tl.store(sorted_route_ptr + out_pos, route, mask=matches)
        tl.store(sorted_slots_ptr + out_pos, offs, mask=matches)

        for local_tile in tl.static_range(0, MAX_TILES_PER_EXPERT):
            local_start = local_tile * BLOCK_M
            valid = local_tile < tiles
            tile_idx = tile_offset + local_tile
            tile_len = tl.maximum(0, tl.minimum(count - local_start, BLOCK_M))
            tl.store(tile_experts_ptr + tile_idx, expert, mask=valid)
            tl.store(tile_starts_ptr + tile_idx, start + local_start, mask=valid)
            tl.store(tile_lengths_ptr + tile_idx, tile_len, mask=valid)

    @triton.jit
    def _qwen3_moe_compact_route_counts_kernel(
        experts_ptr,
        counts_ptr,
        starts_ptr,
        tiles_per_expert_ptr,
        ASSIGNMENTS: tl.constexpr,
        BLOCK_ASSIGNMENTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Build fixed-size route metadata without a global atomic counter."""
        expert = tl.program_id(0)
        offs = tl.arange(0, BLOCK_ASSIGNMENTS)
        mask = offs < ASSIGNMENTS
        routed_expert = tl.load(
            experts_ptr + offs,
            mask=mask,
            other=-1,
        ).to(tl.int64)
        matches = mask & (routed_expert == expert)
        count = tl.sum(matches.to(tl.int32), axis=0)
        start = tl.sum(
            (mask & (routed_expert < expert)).to(tl.int32),
            axis=0,
        )
        tiles = (count + BLOCK_M - 1) // BLOCK_M
        tl.store(counts_ptr + expert, count)
        tl.store(starts_ptr + expert, start)
        tl.store(tiles_per_expert_ptr + expert, tiles)

    @triton.jit
    def _qwen3_moe_compact_route_tile_prefix_kernel(
        tiles_per_expert_ptr,
        tile_offsets_ptr,
        num_tiles_ptr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_EXPERTS: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK_EXPERTS)
        mask = offs < NUM_EXPERTS
        tiles = tl.load(
            tiles_per_expert_ptr + offs,
            mask=mask,
            other=0,
        ).to(tl.int32)
        inclusive = tl.cumsum(tiles, axis=0)
        tl.store(tile_offsets_ptr + offs, inclusive - tiles, mask=mask)
        tl.store(num_tiles_ptr, tl.sum(tiles, axis=0))

    @triton.jit
    def _qwen3_moe_compact_route_scatter_kernel(
        experts_ptr,
        route_ptr,
        counts_ptr,
        starts_ptr,
        tiles_per_expert_ptr,
        tile_offsets_ptr,
        sorted_tokens_ptr,
        sorted_route_ptr,
        sorted_slots_ptr,
        tile_experts_ptr,
        tile_starts_ptr,
        tile_lengths_ptr,
        ASSIGNMENTS: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_ASSIGNMENTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        MAX_TILES: tl.constexpr,
        MAX_TILES_PER_EXPERT: tl.constexpr,
    ):
        expert = tl.program_id(0)
        offs = tl.arange(0, BLOCK_ASSIGNMENTS)
        mask = offs < ASSIGNMENTS
        routed_expert = tl.load(
            experts_ptr + offs,
            mask=mask,
            other=-1,
        ).to(tl.int64)
        matches = mask & (routed_expert == expert)
        prefix = tl.cumsum(matches.to(tl.int32), axis=0)
        local_slot = prefix - 1

        count = tl.load(counts_ptr + expert).to(tl.int32)
        start = tl.load(starts_ptr + expert).to(tl.int32)
        tiles = tl.load(tiles_per_expert_ptr + expert).to(tl.int32)
        tile_offset = tl.load(tile_offsets_ptr + expert).to(tl.int32)
        out_pos = start + local_slot
        route = tl.load(route_ptr + offs, mask=matches, other=0.0)
        out_mask = matches & (out_pos >= 0) & (out_pos < ASSIGNMENTS)

        tl.store(sorted_tokens_ptr + out_pos, offs // TOP_K, mask=out_mask)
        tl.store(sorted_route_ptr + out_pos, route, mask=out_mask)
        tl.store(sorted_slots_ptr + out_pos, offs, mask=out_mask)

        for local_tile in tl.static_range(0, MAX_TILES_PER_EXPERT):
            local_start = local_tile * BLOCK_M
            tile_idx = tile_offset + local_tile
            valid = (local_tile < tiles) & (tile_idx >= 0) & (tile_idx < MAX_TILES)
            tile_len = tl.maximum(0, tl.minimum(count - local_start, BLOCK_M))
            tl.store(tile_experts_ptr + tile_idx, expert, mask=valid)
            tl.store(tile_starts_ptr + tile_idx, start + local_start, mask=valid)
            tl.store(tile_lengths_ptr + tile_idx, tile_len, mask=valid)

    @triton.jit
    def _qwen3_moe_router_topk_softmax_kernel(
        hidden_ptr,
        weight_ptr,
        weights_ptr,
        expert_ids_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_wm: tl.constexpr,
        stride_wt: tl.constexpr,
        stride_em: tl.constexpr,
        stride_et: tl.constexpr,
        H: tl.constexpr,
        E: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs_e = tl.arange(0, BLOCK_E)
        e_mask = offs_e < E
        acc = tl.zeros([BLOCK_E], dtype=tl.float32)

        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < H
            x = tl.load(
                hidden_ptr + row * stride_hm + offs_k * stride_hk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                weight_ptr + offs_e[:, None] * stride_we + offs_k[None, :] * stride_wk,
                mask=e_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        vals = tl.where(e_mask, acc, -float("inf"))
        top_offsets = tl.arange(0, TOP_K)
        top_vals = tl.full([TOP_K], -float("inf"), dtype=tl.float32)
        top_ids = tl.full([TOP_K], 0, dtype=tl.int64)

        for k in tl.static_range(0, TOP_K):
            max_val = tl.max(vals, axis=0)
            is_max = vals == max_val
            max_idx = tl.min(tl.where(is_max, offs_e, BLOCK_E), axis=0)
            top_vals = tl.where(top_offsets == k, max_val, top_vals)
            top_ids = tl.where(top_offsets == k, max_idx.to(tl.int64), top_ids)
            vals = tl.where(offs_e == max_idx, -float("inf"), vals)

        stable_max = tl.max(top_vals, axis=0)
        exp_vals = tl.exp(top_vals - stable_max)
        denom = tl.sum(exp_vals, axis=0)
        probs = exp_vals / denom

        tl.store(
            weights_ptr + row * stride_wm + top_offsets * stride_wt,
            probs.to(weights_ptr.dtype.element_ty),
        )
        tl.store(
            expert_ids_ptr + row * stride_em + top_offsets * stride_et,
            top_ids,
        )

    @triton.jit
    def _qwen3_moe_router_k_split_kernel(
        hidden_ptr,
        weight_ptr,
        partial_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_pm: tl.constexpr,
        stride_ps: tl.constexpr,
        stride_pe: tl.constexpr,
        H: tl.constexpr,
        E: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        split_id = tl.program_id(1)
        offs_e = tl.arange(0, BLOCK_E)
        e_mask = offs_e < E
        split_start = split_id * SPLIT_K
        acc = tl.zeros([BLOCK_E], dtype=tl.float32)

        for k_offset in range(0, SPLIT_K, BLOCK_K):
            offs_k = split_start + k_offset + tl.arange(0, BLOCK_K)
            k_mask = (offs_k < H) & (offs_k < split_start + SPLIT_K)
            x = tl.load(
                hidden_ptr + row * stride_hm + offs_k * stride_hk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                weight_ptr + offs_e[:, None] * stride_we + offs_k[None, :] * stride_wk,
                mask=e_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        tl.store(
            partial_ptr
            + row * stride_pm
            + split_id * stride_ps
            + offs_e * stride_pe,
            acc,
            mask=e_mask,
        )

    @triton.jit
    def _qwen3_moe_router_k_reduce_topk_softmax_kernel(
        partial_ptr,
        weights_ptr,
        expert_ids_ptr,
        stride_pm: tl.constexpr,
        stride_ps: tl.constexpr,
        stride_pe: tl.constexpr,
        stride_wm: tl.constexpr,
        stride_wt: tl.constexpr,
        stride_em: tl.constexpr,
        stride_et: tl.constexpr,
        E: tl.constexpr,
        TOP_K: tl.constexpr,
        K_SPLITS: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs_e = tl.arange(0, BLOCK_E)
        offs_s = tl.arange(0, K_SPLITS)
        e_mask = offs_e < E
        partials = tl.load(
            partial_ptr
            + row * stride_pm
            + offs_s[:, None] * stride_ps
            + offs_e[None, :] * stride_pe,
            mask=e_mask[None, :],
            other=0.0,
        )
        vals = tl.where(e_mask, tl.sum(partials, axis=0), -float("inf"))
        top_offsets = tl.arange(0, TOP_K)
        top_vals = tl.full([TOP_K], -float("inf"), dtype=tl.float32)
        top_ids = tl.full([TOP_K], 0, dtype=tl.int64)

        for k in tl.static_range(0, TOP_K):
            max_val = tl.max(vals, axis=0)
            is_max = vals == max_val
            max_idx = tl.min(tl.where(is_max, offs_e, BLOCK_E), axis=0)
            top_vals = tl.where(top_offsets == k, max_val, top_vals)
            top_ids = tl.where(top_offsets == k, max_idx.to(tl.int64), top_ids)
            vals = tl.where(offs_e == max_idx, -float("inf"), vals)

        stable_max = tl.max(top_vals, axis=0)
        exp_vals = tl.exp(top_vals - stable_max)
        probs = exp_vals / tl.sum(exp_vals, axis=0)
        tl.store(
            weights_ptr + row * stride_wm + top_offsets * stride_wt,
            probs.to(weights_ptr.dtype.element_ty),
        )
        tl.store(
            expert_ids_ptr + row * stride_em + top_offsets * stride_et,
            top_ids,
        )

    @triton.jit
    def _qwen3_moe_topk_softmax_kernel(
        logits_ptr,
        weights_ptr,
        expert_ids_ptr,
        expert_scale_ptr,
        stride_lm: tl.constexpr,
        stride_le: tl.constexpr,
        stride_wm: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_em: tl.constexpr,
        stride_ek: tl.constexpr,
        E: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_E: tl.constexpr,
        APPLY_EXPERT_SCALE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs_e = tl.arange(0, BLOCK_E)
        e_mask = offs_e < E
        vals = tl.load(
            logits_ptr + row * stride_lm + offs_e * stride_le,
            mask=e_mask,
            other=-float("inf"),
        ).to(tl.float32)

        top_offsets = tl.arange(0, TOP_K)
        top_vals = tl.full([TOP_K], -float("inf"), dtype=tl.float32)
        top_ids = tl.full([TOP_K], 0, dtype=tl.int64)

        for k in tl.static_range(0, TOP_K):
            max_val = tl.max(vals, axis=0)
            is_max = vals == max_val
            max_idx = tl.min(tl.where(is_max, offs_e, BLOCK_E), axis=0)
            top_vals = tl.where(top_offsets == k, max_val, top_vals)
            top_ids = tl.where(top_offsets == k, max_idx.to(tl.int64), top_ids)
            vals = tl.where(offs_e == max_idx, -float("inf"), vals)

        stable_max = tl.max(top_vals, axis=0)
        exp_vals = tl.exp(top_vals - stable_max)
        denom = tl.sum(exp_vals, axis=0)
        probs = exp_vals / denom
        if APPLY_EXPERT_SCALE:
            expert_scale = tl.load(expert_scale_ptr + top_ids).to(tl.float32)
            probs *= expert_scale

        tl.store(
            weights_ptr + row * stride_wm + top_offsets * stride_wk,
            probs.to(weights_ptr.dtype.element_ty),
        )
        tl.store(
            expert_ids_ptr + row * stride_em + top_offsets * stride_ek,
            top_ids,
        )

    @triton.jit
    def _qwen3_moe_topk_softmax_compact_pack_kernel(
        logits_ptr,
        weights_ptr,
        expert_ids_ptr,
        expert_scale_ptr,
        counts_ptr,
        dense_tokens_ptr,
        dense_route_ptr,
        dense_assign_ptr,
        stride_lm: tl.constexpr,
        stride_le: tl.constexpr,
        stride_wm: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_em: tl.constexpr,
        stride_ek: tl.constexpr,
        stride_dte: tl.constexpr,
        stride_dtm: tl.constexpr,
        stride_dre: tl.constexpr,
        stride_drm: tl.constexpr,
        stride_dae: tl.constexpr,
        stride_dam: tl.constexpr,
        E: tl.constexpr,
        ROWS: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Select routes and build the deterministic expert-major B16 pack."""
        offs_e = tl.arange(0, BLOCK_E)
        e_mask = offs_e < E
        top_offsets = tl.arange(0, TOP_K)
        counts = tl.zeros((BLOCK_E,), dtype=tl.int32)

        # One persistent CTA preserves row-major assignment order. That makes
        # each expert's slots byte-identical to the old 128-CTA pack kernel.
        for row in tl.static_range(0, ROWS):
            vals = tl.load(
                logits_ptr + row * stride_lm + offs_e * stride_le,
                mask=e_mask,
                other=-float("inf"),
            ).to(tl.float32)
            top_vals = tl.full([TOP_K], -float("inf"), dtype=tl.float32)
            top_ids = tl.full([TOP_K], 0, dtype=tl.int64)

            for k in tl.static_range(0, TOP_K):
                max_val = tl.max(vals, axis=0)
                is_max = vals == max_val
                max_idx = tl.min(
                    tl.where(is_max, offs_e, BLOCK_E),
                    axis=0,
                )
                top_vals = tl.where(top_offsets == k, max_val, top_vals)
                top_ids = tl.where(
                    top_offsets == k,
                    max_idx.to(tl.int64),
                    top_ids,
                )
                vals = tl.where(offs_e == max_idx, -float("inf"), vals)

            stable_max = tl.max(top_vals, axis=0)
            exp_vals = tl.exp(top_vals - stable_max)
            probs = exp_vals / tl.sum(exp_vals, axis=0)
            probs *= tl.load(expert_scale_ptr + top_ids).to(tl.float32)

            tl.store(
                weights_ptr + row * stride_wm + top_offsets * stride_wk,
                probs.to(weights_ptr.dtype.element_ty),
            )
            tl.store(
                expert_ids_ptr + row * stride_em + top_offsets * stride_ek,
                top_ids,
            )

            for k in tl.static_range(0, TOP_K):
                selected = tl.sum(
                    tl.where(top_offsets == k, top_ids, 0),
                    axis=0,
                ).to(tl.int64)
                probability = tl.sum(
                    tl.where(top_offsets == k, probs, 0.0),
                    axis=0,
                )
                selected_mask = e_mask & (offs_e == selected)
                slot = tl.sum(
                    tl.where(selected_mask, counts, 0),
                    axis=0,
                ).to(tl.int64)
                assignment = row * TOP_K + k
                tl.store(
                    dense_tokens_ptr
                    + selected * stride_dte
                    + slot * stride_dtm,
                    row,
                )
                tl.store(
                    dense_route_ptr
                    + selected * stride_dre
                    + slot * stride_drm,
                    probability.to(dense_route_ptr.dtype.element_ty),
                )
                tl.store(
                    dense_assign_ptr
                    + selected * stride_dae
                    + slot * stride_dam,
                    assignment,
                )
                counts += tl.where(selected_mask, 1, 0)

        tl.store(counts_ptr + offs_e, counts, mask=e_mask)

    @triton.jit
    def _qwen3_moe_gate_up_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        expert_ids_ptr,
        token_ids_ptr,
        gate_up_out_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        H: tl.constexpr,
        TWO_I: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_a = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < TWO_I

        expert = tl.load(expert_ids_ptr + pid_a).to(tl.int64)
        token = tl.load(token_ids_ptr + pid_a).to(tl.int64)

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < H

            x = tl.load(
                hidden_ptr + token * stride_hm + offs_k * stride_hk,
                mask=k_mask,
                other=0.0,
            )
            w = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_n[:, None] * stride_wo
                + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        tl.store(
            gate_up_out_ptr + pid_a * stride_om + offs_n * stride_on,
            acc.to(gate_up_out_ptr.dtype.element_ty),
            mask=n_mask,
        )

    @triton.jit
    def _qwen3_moe_gate_up_int8_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        gate_up_scale_ptr,
        expert_ids_ptr,
        token_ids_ptr,
        gate_up_out_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_se: tl.constexpr,
        stride_so: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        H: tl.constexpr,
        TWO_I: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_a = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < TWO_I

        expert = tl.load(expert_ids_ptr + pid_a).to(tl.int64)
        token = tl.load(token_ids_ptr + pid_a).to(tl.int64)
        w_scale = tl.load(
            gate_up_scale_ptr + expert * stride_se + offs_n * stride_so,
            mask=n_mask,
            other=1.0,
        ).to(tl.float32)

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < H
            x = tl.load(
                hidden_ptr + token * stride_hm + offs_k * stride_hk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            w_i8 = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_n[:, None] * stride_wo
                + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.float32)
            acc += tl.sum((w_i8 * w_scale[:, None]) * x[None, :], axis=1)

        tl.store(
            gate_up_out_ptr + pid_a * stride_om + offs_n * stride_on,
            acc.to(gate_up_out_ptr.dtype.element_ty),
            mask=n_mask,
        )

    @triton.jit
    def _qwen3_moe_gate_swiglu_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        expert_ids_ptr,
        token_ids_ptr,
        act_out_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        H: tl.constexpr,
        I: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        USE_DOT: tl.constexpr,
    ):
        pid_a = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < I

        expert = tl.load(expert_ids_ptr + pid_a).to(tl.int64)
        token = tl.load(token_ids_ptr + pid_a).to(tl.int64)

        gate_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        up_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < H

            x = tl.load(
                hidden_ptr + token * stride_hm + offs_k * stride_hk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            gate_w = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_n[:, None] * stride_wo
                + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            up_w = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + (offs_n[:, None] + I) * stride_wo
                + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            if USE_DOT:
                gate_acc += tl.sum(
                    tl.dot(
                        x[None, :],
                        tl.trans(gate_w),
                        out_dtype=tl.float32,
                    ),
                    axis=0,
                )
                up_acc += tl.sum(
                    tl.dot(
                        x[None, :],
                        tl.trans(up_w),
                        out_dtype=tl.float32,
                    ),
                    axis=0,
                )
            else:
                gate_acc += tl.sum(gate_w.to(tl.float32) * x[None, :], axis=1)
                up_acc += tl.sum(up_w.to(tl.float32) * x[None, :], axis=1)

        if ACT == 1:
            c = 0.7978845608028654
            inner = c * (gate_acc + 0.044715 * gate_acc * gate_acc * gate_acc)
            gate_act = gate_acc * tl.sigmoid(2.0 * inner)
        else:
            gate_act = gate_acc * tl.sigmoid(gate_acc)
        activated = gate_act * up_acc

        tl.store(
            act_out_ptr + pid_a * stride_om + offs_n * stride_on,
            activated.to(act_out_ptr.dtype.element_ty),
            mask=n_mask,
        )

    @triton.jit
    def _qwen3_moe_gate_swiglu_int8_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        gate_up_scale_ptr,
        expert_ids_ptr,
        token_ids_ptr,
        act_out_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_se: tl.constexpr,
        stride_so: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        H: tl.constexpr,
        I: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_a = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < I

        expert = tl.load(expert_ids_ptr + pid_a).to(tl.int64)
        token = tl.load(token_ids_ptr + pid_a).to(tl.int64)
        gate_scale = tl.load(
            gate_up_scale_ptr + expert * stride_se + offs_n * stride_so,
            mask=n_mask,
            other=1.0,
        ).to(tl.float32)
        up_scale = tl.load(
            gate_up_scale_ptr + expert * stride_se + (offs_n + I) * stride_so,
            mask=n_mask,
            other=1.0,
        ).to(tl.float32)

        gate_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        up_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < H
            x = tl.load(
                hidden_ptr + token * stride_hm + offs_k * stride_hk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            gate_i8 = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_n[:, None] * stride_wo
                + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.float32)
            up_i8 = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + (offs_n[:, None] + I) * stride_wo
                + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.float32)
            gate_acc += tl.sum((gate_i8 * gate_scale[:, None]) * x[None, :], axis=1)
            up_acc += tl.sum((up_i8 * up_scale[:, None]) * x[None, :], axis=1)

        if ACT == 1:
            c = 0.7978845608028654
            inner = c * (gate_acc + 0.044715 * gate_acc * gate_acc * gate_acc)
            gate_act = gate_acc * tl.sigmoid(2.0 * inner)
        else:
            gate_act = gate_acc * tl.sigmoid(gate_acc)
        activated = gate_act * up_acc
        tl.store(
            act_out_ptr + pid_a * stride_om + offs_n * stride_on,
            activated.to(act_out_ptr.dtype.element_ty),
            mask=n_mask,
        )

    @triton.jit
    def _qwen3_moe_swiglu_down_accum_kernel(
        gate_up_ptr,
        down_w_ptr,
        expert_ids_ptr,
        token_ids_ptr,
        routing_ptr,
        out_ptr,
        stride_gm: tl.constexpr,
        stride_gk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_a = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H

        expert = tl.load(expert_ids_ptr + pid_a).to(tl.int64)
        token = tl.load(token_ids_ptr + pid_a).to(tl.int64)
        route = tl.load(routing_ptr + pid_a).to(tl.float32)

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I

            gate = tl.load(
                gate_up_ptr + pid_a * stride_gm + offs_k * stride_gk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                gate_up_ptr + pid_a * stride_gm + (offs_k + I) * stride_gk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            if ACT == 1:
                c = 0.7978845608028654
                inner = c * (gate + 0.044715 * gate * gate * gate)
                gate_act = gate * tl.sigmoid(2.0 * inner)
            else:
                gate_act = gate * tl.sigmoid(gate)
            activated = gate_act * up

            w = tl.load(
                down_w_ptr
                + expert * stride_we
                + offs_h[:, None] * stride_wh
                + offs_k[None, :] * stride_wi,
                mask=h_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * activated[None, :], axis=1)

        out_offsets = out_ptr + token * stride_om + offs_h * stride_oh
        tl.atomic_add(out_offsets, acc * route, sem="relaxed", mask=h_mask)

    @triton.jit
    def _qwen3_moe_swiglu_down_partial_kernel(
        gate_up_ptr,
        down_w_ptr,
        expert_ids_ptr,
        routing_ptr,
        partial_ptr,
        stride_gm: tl.constexpr,
        stride_gk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_pa: tl.constexpr,
        stride_ph: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_a = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H

        expert = tl.load(expert_ids_ptr + pid_a).to(tl.int64)
        route = tl.load(routing_ptr + pid_a).to(tl.float32)

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I

            gate = tl.load(
                gate_up_ptr + pid_a * stride_gm + offs_k * stride_gk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                gate_up_ptr + pid_a * stride_gm + (offs_k + I) * stride_gk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            if ACT == 1:
                c = 0.7978845608028654
                inner = c * (gate + 0.044715 * gate * gate * gate)
                gate_act = gate * tl.sigmoid(2.0 * inner)
            else:
                gate_act = gate * tl.sigmoid(gate)
            activated = gate_act * up

            w = tl.load(
                down_w_ptr
                + expert * stride_we
                + offs_h[:, None] * stride_wh
                + offs_k[None, :] * stride_wi,
                mask=h_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * activated[None, :], axis=1)

        tl.store(
            partial_ptr + pid_a * stride_pa + offs_h * stride_ph,
            acc * route,
            mask=h_mask,
        )

    @triton.jit
    def _qwen3_moe_swiglu_down_accum_int8_kernel(
        gate_up_ptr,
        down_w_ptr,
        down_scale_ptr,
        expert_ids_ptr,
        token_ids_ptr,
        routing_ptr,
        out_ptr,
        stride_gm: tl.constexpr,
        stride_gk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_se: tl.constexpr,
        stride_sh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_a = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H

        expert = tl.load(expert_ids_ptr + pid_a).to(tl.int64)
        token = tl.load(token_ids_ptr + pid_a).to(tl.int64)
        route = tl.load(routing_ptr + pid_a).to(tl.float32)
        w_scale = tl.load(
            down_scale_ptr + expert * stride_se + offs_h * stride_sh,
            mask=h_mask,
            other=1.0,
        ).to(tl.float32)

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I
            gate = tl.load(
                gate_up_ptr + pid_a * stride_gm + offs_k * stride_gk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                gate_up_ptr + pid_a * stride_gm + (offs_k + I) * stride_gk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            if ACT == 1:
                c = 0.7978845608028654
                inner = c * (gate + 0.044715 * gate * gate * gate)
                gate_act = gate * tl.sigmoid(2.0 * inner)
            else:
                gate_act = gate * tl.sigmoid(gate)
            activated = gate_act * up
            w_i8 = tl.load(
                down_w_ptr
                + expert * stride_we
                + offs_h[:, None] * stride_wh
                + offs_k[None, :] * stride_wi,
                mask=h_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.float32)
            acc += tl.sum((w_i8 * w_scale[:, None]) * activated[None, :], axis=1)

        out_offsets = out_ptr + token * stride_om + offs_h * stride_oh
        tl.atomic_add(out_offsets, acc * route, sem="relaxed", mask=h_mask)

    @triton.jit
    def _qwen3_moe_swiglu_down_token_accum_kernel(
        gate_up_ptr,
        down_w_ptr,
        expert_ids_ptr,
        routing_ptr,
        out_ptr,
        stride_gm: tl.constexpr,
        stride_gk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        token = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for top_idx in tl.static_range(0, TOP_K):
            pid_a = token * TOP_K + top_idx
            expert = tl.load(expert_ids_ptr + pid_a).to(tl.int64)
            route = tl.load(routing_ptr + pid_a).to(tl.float32)

            expert_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
            for k_start in range(0, I, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < I

                gate = tl.load(
                    gate_up_ptr + pid_a * stride_gm + offs_k * stride_gk,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                up = tl.load(
                    gate_up_ptr + pid_a * stride_gm + (offs_k + I) * stride_gk,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                if ACT == 1:
                    c = 0.7978845608028654
                    inner = c * (gate + 0.044715 * gate * gate * gate)
                    gate_act = gate * tl.sigmoid(2.0 * inner)
                else:
                    gate_act = gate * tl.sigmoid(gate)
                activated = gate_act * up

                w = tl.load(
                    down_w_ptr
                    + expert * stride_we
                    + offs_h[:, None] * stride_wh
                    + offs_k[None, :] * stride_wi,
                    mask=h_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                expert_acc += tl.sum(w * activated[None, :], axis=1)

            acc += expert_acc * route

        tl.store(
            out_ptr + token * stride_om + offs_h * stride_oh,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )

    @triton.jit
    def _qwen3_moe_swiglu_down_token_accum_int8_kernel(
        gate_up_ptr,
        down_w_ptr,
        down_scale_ptr,
        expert_ids_ptr,
        routing_ptr,
        out_ptr,
        stride_gm: tl.constexpr,
        stride_gk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_se: tl.constexpr,
        stride_sh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        token = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for top_idx in tl.static_range(0, TOP_K):
            pid_a = token * TOP_K + top_idx
            expert = tl.load(expert_ids_ptr + pid_a).to(tl.int64)
            route = tl.load(routing_ptr + pid_a).to(tl.float32)
            w_scale = tl.load(
                down_scale_ptr + expert * stride_se + offs_h * stride_sh,
                mask=h_mask,
                other=1.0,
            ).to(tl.float32)

            expert_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
            for k_start in range(0, I, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < I
                gate = tl.load(
                    gate_up_ptr + pid_a * stride_gm + offs_k * stride_gk,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                up = tl.load(
                    gate_up_ptr + pid_a * stride_gm + (offs_k + I) * stride_gk,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                if ACT == 1:
                    c = 0.7978845608028654
                    inner = c * (gate + 0.044715 * gate * gate * gate)
                    gate_act = gate * tl.sigmoid(2.0 * inner)
                else:
                    gate_act = gate * tl.sigmoid(gate)
                activated = gate_act * up
                w_i8 = tl.load(
                    down_w_ptr
                    + expert * stride_we
                    + offs_h[:, None] * stride_wh
                    + offs_k[None, :] * stride_wi,
                    mask=h_mask[:, None] & k_mask[None, :],
                    other=0,
                ).to(tl.float32)
                expert_acc += tl.sum((w_i8 * w_scale[:, None]) * activated[None, :], axis=1)
            acc += expert_acc * route

        tl.store(
            out_ptr + token * stride_om + offs_h * stride_oh,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )

    @triton.jit
    def _qwen3_moe_down_from_act_token_accum_kernel(
        act_ptr,
        down_w_ptr,
        expert_ids_ptr,
        routing_ptr,
        out_ptr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        USE_DOT: tl.constexpr,
    ):
        token = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for top_idx in tl.static_range(0, TOP_K):
            pid_a = token * TOP_K + top_idx
            expert = tl.load(expert_ids_ptr + pid_a).to(tl.int64)
            route = tl.load(routing_ptr + pid_a).to(tl.float32)

            expert_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
            for k_start in range(0, I, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < I

                activated = tl.load(
                    act_ptr + pid_a * stride_am + offs_k * stride_ai,
                    mask=k_mask,
                    other=0.0,
                )
                w = tl.load(
                    down_w_ptr
                    + expert * stride_we
                    + offs_h[:, None] * stride_wh
                    + offs_k[None, :] * stride_wi,
                    mask=h_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                if USE_DOT:
                    expert_acc += tl.sum(
                        tl.dot(
                            activated[None, :],
                            tl.trans(w),
                            out_dtype=tl.float32,
                        ),
                        axis=0,
                    )
                else:
                    expert_acc += tl.sum(w.to(tl.float32) * activated[None, :], axis=1)

            acc += expert_acc * route

        tl.store(
            out_ptr + token * stride_om + offs_h * stride_oh,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )

    @triton.jit
    def _qwen3_moe_down_from_act_token_accum_int8_kernel(
        act_ptr,
        down_w_ptr,
        down_scale_ptr,
        expert_ids_ptr,
        routing_ptr,
        out_ptr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_se: tl.constexpr,
        stride_sh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        token = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for top_idx in tl.static_range(0, TOP_K):
            pid_a = token * TOP_K + top_idx
            expert = tl.load(expert_ids_ptr + pid_a).to(tl.int64)
            route = tl.load(routing_ptr + pid_a).to(tl.float32)
            w_scale = tl.load(
                down_scale_ptr + expert * stride_se + offs_h * stride_sh,
                mask=h_mask,
                other=1.0,
            ).to(tl.float32)

            expert_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
            for k_start in range(0, I, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < I
                activated = tl.load(
                    act_ptr + pid_a * stride_am + offs_k * stride_ai,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                w_i8 = tl.load(
                    down_w_ptr
                    + expert * stride_we
                    + offs_h[:, None] * stride_wh
                    + offs_k[None, :] * stride_wi,
                    mask=h_mask[:, None] & k_mask[None, :],
                    other=0,
                ).to(tl.float32)
                expert_acc += tl.sum((w_i8 * w_scale[:, None]) * activated[None, :], axis=1)
            acc += expert_acc * route

        tl.store(
            out_ptr + token * stride_om + offs_h * stride_oh,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )

    @triton.jit
    def _qwen3_moe_expert_grouped_scatter_kernel(
        expert_ids_ptr,
        routing_ptr,
        counts_ptr,
        dense_tokens_ptr,
        dense_route_ptr,
        assignments: tl.constexpr,
        ROWS: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < assignments
        expert = tl.load(expert_ids_ptr + offs, mask=mask, other=0).to(tl.int64)
        route = tl.load(routing_ptr + offs, mask=mask, other=0.0)
        token = offs // TOP_K
        slot = tl.atomic_add(counts_ptr + expert, 1, sem="relaxed", mask=mask)
        slot_mask = mask & (slot < ROWS)
        tl.store(
            dense_tokens_ptr + expert * ROWS + slot,
            token,
            mask=slot_mask,
        )
        tl.store(
            dense_route_ptr + expert * ROWS + slot,
            route,
            mask=slot_mask,
        )

    @triton.jit
    def _qwen3_moe_expert_grouped_active_scatter_kernel(
        expert_ids_ptr,
        routing_ptr,
        counts_ptr,
        dense_tokens_ptr,
        dense_route_ptr,
        active_experts_ptr,
        active_count_ptr,
        expert_to_candidate_ptr,
        assignments: tl.constexpr,
        ROWS: tl.constexpr,
        TOP_K: tl.constexpr,
        BUILD_MAP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid == 0:
            tl.store(active_count_ptr, 0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < assignments
        expert = tl.load(expert_ids_ptr + offs, mask=mask, other=0).to(tl.int64)
        route = tl.load(routing_ptr + offs, mask=mask, other=0.0)
        token = offs // TOP_K
        slot = tl.atomic_add(counts_ptr + expert, 1, sem="relaxed", mask=mask)
        slot_mask = mask & (slot < ROWS)
        tl.store(
            dense_tokens_ptr + expert * ROWS + slot,
            token,
            mask=slot_mask,
        )
        tl.store(
            dense_route_ptr + expert * ROWS + slot,
            route,
            mask=slot_mask,
        )
        is_first = mask & (slot == 0)
        candidate = tl.atomic_add(active_count_ptr, 1, sem="relaxed", mask=is_first)
        tl.store(active_experts_ptr + candidate, expert, mask=is_first)
        if BUILD_MAP:
            tl.store(expert_to_candidate_ptr + expert, candidate.to(tl.int64), mask=is_first)

    @triton.jit
    def _qwen3_moe_expert_grouped_unique_kernel(
        expert_ids_ptr,
        unique_ptr,
        expert_to_candidate_ptr,
        assignments: tl.constexpr,
        WRITE_MAP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < assignments
        expert = tl.load(expert_ids_ptr + offs, mask=mask, other=-1).to(tl.int64)
        is_first = mask
        for prev in tl.static_range(0, 128):
            if prev < assignments:
                prev_expert = tl.load(expert_ids_ptr + prev).to(tl.int64)
                is_first = is_first & ((prev >= offs) | (prev_expert != expert))
        tl.store(unique_ptr + offs, tl.where(is_first, 1, 0), mask=mask)
        if WRITE_MAP:
            tl.store(expert_to_candidate_ptr + expert, offs, mask=is_first)

    @triton.jit
    def _qwen3_moe_expert_grouped_compact_pack_kernel(
        expert_ids_ptr,
        routing_ptr,
        counts_ptr,
        dense_tokens_ptr,
        dense_route_ptr,
        dense_assign_ptr,
        unique_ptr,
        active_experts_ptr,
        active_count_ptr,
        expert_to_candidate_ptr,
        assignments: tl.constexpr,
        ROWS: tl.constexpr,
        TOP_K: tl.constexpr,
        STORE_ASSIGN: tl.constexpr,
        STORE_ACTIVE: tl.constexpr,
        BUILD_MAP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK)
        mask = offs < assignments
        expert = tl.load(expert_ids_ptr + offs, mask=mask, other=-1).to(tl.int64)
        route = tl.load(routing_ptr + offs, mask=mask, other=0.0)
        token = offs // TOP_K

        slot = tl.zeros((BLOCK,), dtype=tl.int32)
        count = tl.zeros((BLOCK,), dtype=tl.int32)
        for idx in tl.static_range(0, 128):
            if idx < assignments:
                other_expert = tl.load(expert_ids_ptr + idx).to(tl.int64)
                same = mask & (other_expert == expert)
                count += tl.where(same, 1, 0)
                slot += tl.where(same & (idx < offs), 1, 0)

        is_first = mask & (slot == 0)
        slot_mask = mask & (slot < ROWS)
        tl.store(counts_ptr + expert, count, mask=is_first)
        tl.store(unique_ptr + offs, tl.where(is_first, 1, 0), mask=mask)
        rank = tl.zeros((BLOCK,), dtype=tl.int32)
        active_total = tl.full((), 0, tl.int32)
        if STORE_ACTIVE:
            for idx in tl.static_range(0, 128):
                if idx < assignments:
                    first_at_idx = tl.sum(tl.where(is_first & (offs == idx), 1, 0), axis=0)
                    rank += tl.where(offs > idx, first_at_idx, 0)
                    active_total += first_at_idx
            tl.store(active_count_ptr, active_total)
            tl.store(active_experts_ptr + rank, expert, mask=is_first)
            if BUILD_MAP:
                tl.store(expert_to_candidate_ptr + expert, rank.to(tl.int64), mask=is_first)
        tl.store(
            dense_tokens_ptr + expert * ROWS + slot,
            token,
            mask=slot_mask,
        )
        tl.store(
            dense_route_ptr + expert * ROWS + slot,
            route,
            mask=slot_mask,
        )
        if STORE_ASSIGN:
            tl.store(
                dense_assign_ptr + expert * ROWS + slot,
                offs,
                mask=slot_mask,
            )

    @triton.jit
    def _qwen3_moe_expert_grouped_compact_expert_pack_kernel(
        expert_ids_ptr,
        routing_ptr,
        counts_ptr,
        dense_tokens_ptr,
        dense_route_ptr,
        dense_assign_ptr,
        assignments: tl.constexpr,
        ROWS: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        expert = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < assignments
        routed_expert = tl.load(
            expert_ids_ptr + offs,
            mask=mask,
            other=-1,
        ).to(tl.int64)
        matches = mask & (routed_expert == expert)
        slot = tl.cumsum(matches.to(tl.int32), axis=0) - 1
        count = tl.sum(matches.to(tl.int32), axis=0)
        route = tl.load(routing_ptr + offs, mask=matches, other=0.0)
        out_pos = expert * ROWS + slot

        tl.store(counts_ptr + expert, count)
        tl.store(dense_tokens_ptr + out_pos, offs // TOP_K, mask=matches)
        tl.store(dense_route_ptr + out_pos, route, mask=matches)
        tl.store(dense_assign_ptr + out_pos, offs, mask=matches)

    @triton.jit
    def _qwen3_moe_expert_grouped_compact_active_blocks_kernel(
        counts_ptr,
        active_experts_ptr,
        active_count_ptr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        expert = tl.arange(0, BLOCK)
        mask = expert < NUM_EXPERTS
        active = tl.load(counts_ptr + expert, mask=mask, other=0) > 0
        rank = tl.cumsum(active.to(tl.int32), axis=0) - 1
        tl.store(active_experts_ptr + rank, expert, mask=active)
        tl.store(active_count_ptr, tl.sum(active.to(tl.int32), axis=0))

    @triton.jit
    def _qwen3_moe_expert_grouped_gate_swiglu_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        counts_ptr,
        dense_tokens_ptr,
        act_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_ae: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        H: tl.constexpr,
        I: tl.constexpr,
        ROWS: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        expert = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        count = tl.load(counts_ptr + expert).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < count
        n_mask = offs_n < I
        token_ids = tl.load(
            dense_tokens_ptr + expert * ROWS + offs_m,
            mask=m_mask,
            other=0,
        ).to(tl.int64)

        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < H
            x = tl.load(
                hidden_ptr + token_ids[:, None] * stride_hm + offs_k[None, :] * stride_hk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            gate_w = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wk
                + offs_n[None, :] * stride_wo,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            up_w = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wk
                + (offs_n[None, :] + I) * stride_wo,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            gate_acc += tl.dot(x, gate_w)
            up_acc += tl.dot(x, up_w)

        if ACT == 1:
            c = 0.7978845608028654
            inner = c * (gate_acc + 0.044715 * gate_acc * gate_acc * gate_acc)
            gate_act = gate_acc * tl.sigmoid(2.0 * inner)
        else:
            gate_act = gate_acc * tl.sigmoid(gate_acc)
        activated = gate_act * up_acc

        tl.store(
            act_ptr
            + expert * stride_ae
            + offs_m[:, None] * stride_am
            + offs_n[None, :] * stride_ai,
            activated.to(act_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_expert_grouped_down_accum_kernel(
        act_ptr,
        down_w_ptr,
        counts_ptr,
        dense_tokens_ptr,
        dense_route_ptr,
        accum_ptr,
        stride_ae: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        expert = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_h = tl.program_id(2)
        count = tl.load(counts_ptr + expert).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < count
        h_mask = offs_h < H
        token_ids = tl.load(
            dense_tokens_ptr + expert * ROWS + offs_m,
            mask=m_mask,
            other=0,
        ).to(tl.int64)
        route = tl.load(
            dense_route_ptr + expert * ROWS + offs_m,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I
            activated = tl.load(
                act_ptr
                + expert * stride_ae
                + offs_m[:, None] * stride_am
                + offs_k[None, :] * stride_ai,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                down_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wi
                + offs_h[None, :] * stride_wh,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(activated, w)

        tl.atomic_add(
            accum_ptr + token_ids[:, None] * stride_om + offs_h[None, :] * stride_oh,
            acc * route[:, None],
            sem="relaxed",
            mask=m_mask[:, None] & h_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_expert_grouped_compact_gate_swiglu_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        expert_ids_ptr,
        unique_or_count_ptr,
        counts_ptr,
        dense_tokens_ptr,
        act_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_ae: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        H: tl.constexpr,
        I: tl.constexpr,
        ROWS: tl.constexpr,
        ASSIGNMENTS: tl.constexpr,
        CANDIDATE_SLOTS: tl.constexpr,
        ACTIVE_LIST: tl.constexpr,
        EXPERT_GRID: tl.constexpr,
        COALESCED_WEIGHTS: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        EXPERTS_PER_PROGRAM: tl.constexpr,
        PAIRED_GATE_UP_DOT: tl.constexpr,
        EMPTY_EXPERT_EARLY_EXIT: tl.constexpr,
        ACTIVE_LIST_EARLY_EXIT: tl.constexpr,
        L2_GROUPED_GRID: tl.constexpr,
        NUM_CANDIDATE_GROUPS: tl.constexpr,
        NUM_PID_N: tl.constexpr,
        L2_GROUP_SIZE: tl.constexpr,
    ):
        if L2_GROUPED_GRID:
            pid = tl.program_id(0)
            num_pid_in_group = L2_GROUP_SIZE * NUM_PID_N
            group_id = pid // num_pid_in_group
            first_candidate = group_id * L2_GROUP_SIZE
            group_size = tl.minimum(
                NUM_CANDIDATE_GROUPS - first_candidate,
                L2_GROUP_SIZE,
            )
            pid_in_group = pid % num_pid_in_group
            candidate_group = first_candidate + (pid_in_group % group_size)
            pid_m = 0
            pid_n = pid_in_group // group_size
        else:
            candidate_group = tl.program_id(0)
            pid_m = tl.program_id(1)
            pid_n = tl.program_id(2)
        if ACTIVE_LIST_EARLY_EXIT and ACTIVE_LIST and EXPERTS_PER_PROGRAM == 1:
            active_count = tl.load(unique_or_count_ptr).to(tl.int64)
            if candidate_group >= active_count:
                return
        if EMPTY_EXPERT_EARLY_EXIT and EXPERT_GRID and EXPERTS_PER_PROGRAM == 1:
            if tl.load(counts_ptr + candidate_group) <= 0:
                return
        for candidate_offset in tl.static_range(0, EXPERTS_PER_PROGRAM):
            candidate = candidate_group * EXPERTS_PER_PROGRAM + candidate_offset
            candidate_valid = candidate < CANDIDATE_SLOTS
            if EXPERT_GRID:
                expert = candidate
                is_first = candidate_valid
            elif ACTIVE_LIST:
                active_count = tl.load(unique_or_count_ptr).to(tl.int64)
                is_first = candidate_valid & (candidate < active_count)
                expert = tl.load(
                    expert_ids_ptr + candidate,
                    mask=is_first,
                    other=0,
                ).to(tl.int64)
            else:
                expert = tl.load(
                    expert_ids_ptr + candidate,
                    mask=candidate_valid,
                    other=0,
                ).to(tl.int64)
                is_first = candidate_valid & (
                    tl.load(unique_or_count_ptr + candidate, mask=candidate_valid, other=0)
                    != 0
                )
            count = tl.load(counts_ptr + expert, mask=is_first, other=0).to(tl.int64)
            count = tl.where(is_first, count, 0)
            work = is_first & (count > 0)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            m_mask = work & (offs_m < count)
            n_mask = offs_n < I
            token_ids = tl.load(
                dense_tokens_ptr + expert * ROWS + offs_m,
                mask=m_mask,
                other=0,
            ).to(tl.int64)

            if PAIRED_GATE_UP_DOT:
                pair_acc = tl.zeros((BLOCK_M, BLOCK_N * 2), dtype=tl.float32)
                offs_pair = tl.arange(0, BLOCK_N * 2)
                pair_local = offs_pair % BLOCK_N
                pair_plane = offs_pair // BLOCK_N
                pair_output = pid_n * BLOCK_N + pair_local + pair_plane * I
                pair_mask = pid_n * BLOCK_N + pair_local < I
            else:
                gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, H, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < H
                x = tl.load(
                    hidden_ptr
                    + token_ids[:, None] * stride_hm
                    + offs_k[None, :] * stride_hk,
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                if PAIRED_GATE_UP_DOT and COALESCED_WEIGHTS:
                    pair_w = tl.load(
                        gate_up_w_ptr
                        + expert * stride_we
                        + pair_output[:, None] * stride_wo
                        + offs_k[None, :] * stride_wk,
                        mask=work & pair_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    pair_acc += tl.dot(
                        x,
                        tl.trans(pair_w),
                        out_dtype=tl.float32,
                    )
                elif PAIRED_GATE_UP_DOT:
                    pair_w = tl.load(
                        gate_up_w_ptr
                        + expert * stride_we
                        + pair_output[None, :] * stride_wo
                        + offs_k[:, None] * stride_wk,
                        mask=work & k_mask[:, None] & pair_mask[None, :],
                        other=0.0,
                    )
                    pair_acc += tl.dot(x, pair_w, out_dtype=tl.float32)
                elif COALESCED_WEIGHTS:
                    gate_w = tl.load(
                        gate_up_w_ptr
                        + expert * stride_we
                        + offs_n[:, None] * stride_wo
                        + offs_k[None, :] * stride_wk,
                        mask=work & n_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    up_w = tl.load(
                        gate_up_w_ptr
                        + expert * stride_we
                        + (offs_n[:, None] + I) * stride_wo
                        + offs_k[None, :] * stride_wk,
                        mask=work & n_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    gate_acc += tl.dot(x, tl.trans(gate_w), out_dtype=tl.float32)
                    up_acc += tl.dot(x, tl.trans(up_w), out_dtype=tl.float32)
                else:
                    gate_w = tl.load(
                        gate_up_w_ptr
                        + expert * stride_we
                        + offs_n[None, :] * stride_wo
                        + offs_k[:, None] * stride_wk,
                        mask=work & k_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    )
                    up_w = tl.load(
                        gate_up_w_ptr
                        + expert * stride_we
                        + (offs_n[None, :] + I) * stride_wo
                        + offs_k[:, None] * stride_wk,
                        mask=work & k_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    )
                    gate_acc += tl.dot(x, gate_w)
                    up_acc += tl.dot(x, up_w)

            if PAIRED_GATE_UP_DOT:
                pair_acc = tl.reshape(pair_acc, (BLOCK_M, 2, BLOCK_N))
                pair_acc = tl.permute(pair_acc, 0, 2, 1)
                gate_acc, up_acc = tl.split(pair_acc)

            if ACT == 1:
                c = 0.7978845608028654
                inner = c * (gate_acc + 0.044715 * gate_acc * gate_acc * gate_acc)
                gate_act = gate_acc * tl.sigmoid(2.0 * inner)
            else:
                gate_act = gate_acc * tl.sigmoid(gate_acc)
            activated = gate_act * up_acc

            tl.store(
                act_ptr
                + candidate * stride_ae
                + offs_m[:, None] * stride_am
                + offs_n[None, :] * stride_ai,
                activated.to(act_ptr.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )

    @triton.jit
    def _qwen3_moe_expert_grouped_compact_gate_up_split_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        expert_ids_ptr,
        unique_or_count_ptr,
        counts_ptr,
        dense_tokens_ptr,
        gate_up_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_ge: tl.constexpr,
        stride_gm: tl.constexpr,
        stride_gn: tl.constexpr,
        H: tl.constexpr,
        I: tl.constexpr,
        ROWS: tl.constexpr,
        ASSIGNMENTS: tl.constexpr,
        CANDIDATE_SLOTS: tl.constexpr,
        ACTIVE_LIST: tl.constexpr,
        EXPERT_GRID: tl.constexpr,
        COALESCED_WEIGHTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        EXPERTS_PER_PROGRAM: tl.constexpr,
        EMPTY_EXPERT_EARLY_EXIT: tl.constexpr,
        ACTIVE_LIST_EARLY_EXIT: tl.constexpr,
    ):
        candidate_group = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        if ACTIVE_LIST_EARLY_EXIT and ACTIVE_LIST and EXPERTS_PER_PROGRAM == 1:
            active_count = tl.load(unique_or_count_ptr).to(tl.int64)
            if candidate_group >= active_count:
                return
        if EMPTY_EXPERT_EARLY_EXIT and EXPERT_GRID and EXPERTS_PER_PROGRAM == 1:
            if tl.load(counts_ptr + candidate_group) <= 0:
                return
        for candidate_offset in tl.static_range(0, EXPERTS_PER_PROGRAM):
            candidate = candidate_group * EXPERTS_PER_PROGRAM + candidate_offset
            candidate_valid = candidate < CANDIDATE_SLOTS
            if EXPERT_GRID:
                expert = candidate
                is_first = candidate_valid
            elif ACTIVE_LIST:
                active_count = tl.load(unique_or_count_ptr).to(tl.int64)
                is_first = candidate_valid & (candidate < active_count)
                expert = tl.load(
                    expert_ids_ptr + candidate,
                    mask=is_first,
                    other=0,
                ).to(tl.int64)
            else:
                expert = tl.load(
                    expert_ids_ptr + candidate,
                    mask=candidate_valid,
                    other=0,
                ).to(tl.int64)
                is_first = candidate_valid & (
                    tl.load(unique_or_count_ptr + candidate, mask=candidate_valid, other=0)
                    != 0
                )
            count = tl.load(counts_ptr + expert, mask=is_first, other=0).to(tl.int64)
            count = tl.where(is_first, count, 0)
            work = is_first & (count > 0)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            m_mask = work & (offs_m < count)
            n_mask = offs_n < 2 * I
            token_ids = tl.load(
                dense_tokens_ptr + expert * ROWS + offs_m,
                mask=m_mask,
                other=0,
            ).to(tl.int64)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, H, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < H
                x = tl.load(
                    hidden_ptr
                    + token_ids[:, None] * stride_hm
                    + offs_k[None, :] * stride_hk,
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                if COALESCED_WEIGHTS:
                    w = tl.load(
                        gate_up_w_ptr
                        + expert * stride_we
                        + offs_n[:, None] * stride_wo
                        + offs_k[None, :] * stride_wk,
                        mask=work & n_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
                else:
                    w = tl.load(
                        gate_up_w_ptr
                        + expert * stride_we
                        + offs_n[None, :] * stride_wo
                        + offs_k[:, None] * stride_wk,
                        mask=work & k_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(x, w)

            tl.store(
                gate_up_ptr
                + candidate * stride_ge
                + offs_m[:, None] * stride_gm
                + offs_n[None, :] * stride_gn,
                acc.to(gate_up_ptr.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )

    @triton.jit
    def _qwen3_moe_expert_grouped_compact_swiglu_kernel(
        expert_ids_ptr,
        unique_or_count_ptr,
        counts_ptr,
        gate_up_ptr,
        act_ptr,
        stride_ge: tl.constexpr,
        stride_gm: tl.constexpr,
        stride_gn: tl.constexpr,
        stride_ae: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        I: tl.constexpr,
        CANDIDATE_SLOTS: tl.constexpr,
        ACTIVE_LIST: tl.constexpr,
        EXPERT_GRID: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EXPERTS_PER_PROGRAM: tl.constexpr,
        ACTIVE_LIST_EARLY_EXIT: tl.constexpr,
    ):
        candidate_group = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        if ACTIVE_LIST_EARLY_EXIT and ACTIVE_LIST and EXPERTS_PER_PROGRAM == 1:
            active_count = tl.load(unique_or_count_ptr).to(tl.int64)
            if candidate_group >= active_count:
                return
        for candidate_offset in tl.static_range(0, EXPERTS_PER_PROGRAM):
            candidate = candidate_group * EXPERTS_PER_PROGRAM + candidate_offset
            candidate_valid = candidate < CANDIDATE_SLOTS
            if EXPERT_GRID:
                expert = candidate
                is_first = candidate_valid
            elif ACTIVE_LIST:
                active_count = tl.load(unique_or_count_ptr).to(tl.int64)
                is_first = candidate_valid & (candidate < active_count)
                expert = tl.load(
                    expert_ids_ptr + candidate,
                    mask=is_first,
                    other=0,
                ).to(tl.int64)
            else:
                expert = tl.load(
                    expert_ids_ptr + candidate,
                    mask=candidate_valid,
                    other=0,
                ).to(tl.int64)
                is_first = candidate_valid & (
                    tl.load(unique_or_count_ptr + candidate, mask=candidate_valid, other=0)
                    != 0
                )
            count = tl.load(counts_ptr + expert, mask=is_first, other=0).to(tl.int64)
            count = tl.where(is_first, count, 0)
            work = is_first & (count > 0)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            m_mask = work & (offs_m < count)
            n_mask = offs_n < I
            gate = tl.load(
                gate_up_ptr
                + candidate * stride_ge
                + offs_m[:, None] * stride_gm
                + offs_n[None, :] * stride_gn,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                gate_up_ptr
                + candidate * stride_ge
                + offs_m[:, None] * stride_gm
                + (offs_n[None, :] + I) * stride_gn,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if ACT == 1:
                c = 0.7978845608028654
                inner = c * (gate + 0.044715 * gate * gate * gate)
                gate_act = gate * tl.sigmoid(2.0 * inner)
            else:
                gate_act = gate * tl.sigmoid(gate)
            tl.store(
                act_ptr
                + candidate * stride_ae
                + offs_m[:, None] * stride_am
                + offs_n[None, :] * stride_ai,
                (gate_act * up).to(act_ptr.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )

    @triton.jit
    def _qwen3_moe_expert_grouped_compact_down_accum_kernel(
        act_ptr,
        down_w_ptr,
        expert_ids_ptr,
        unique_or_count_ptr,
        counts_ptr,
        dense_tokens_ptr,
        dense_route_ptr,
        accum_ptr,
        stride_ae: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        ROWS: tl.constexpr,
        ASSIGNMENTS: tl.constexpr,
        ACTIVE_LIST: tl.constexpr,
        EXPERT_GRID: tl.constexpr,
        COALESCED_WEIGHTS: tl.constexpr,
        ACTIVE_LIST_EARLY_EXIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        candidate = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_h = tl.program_id(2)
        if ACTIVE_LIST_EARLY_EXIT and ACTIVE_LIST:
            active_count = tl.load(unique_or_count_ptr).to(tl.int64)
            if candidate >= active_count:
                return
        if EXPERT_GRID:
            expert = candidate
            is_first = True
        elif ACTIVE_LIST:
            active_count = tl.load(unique_or_count_ptr).to(tl.int64)
            is_first = candidate < active_count
            expert = tl.load(expert_ids_ptr + candidate, mask=is_first, other=0).to(tl.int64)
        else:
            expert = tl.load(expert_ids_ptr + candidate).to(tl.int64)
            is_first = tl.load(unique_or_count_ptr + candidate) != 0
        count = tl.load(counts_ptr + expert).to(tl.int64)
        count = tl.where(is_first, count, 0)
        work = is_first & (count > 0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = work & (offs_m < count)
        h_mask = offs_h < H
        token_ids = tl.load(
            dense_tokens_ptr + expert * ROWS + offs_m,
            mask=m_mask,
            other=0,
        ).to(tl.int64)
        route = tl.load(
            dense_route_ptr + expert * ROWS + offs_m,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I
            activated = tl.load(
                act_ptr
                + candidate * stride_ae
                + offs_m[:, None] * stride_am
                + offs_k[None, :] * stride_ai,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            if COALESCED_WEIGHTS:
                w = tl.load(
                    down_w_ptr
                    + expert * stride_we
                    + offs_h[:, None] * stride_wh
                    + offs_k[None, :] * stride_wi,
                    mask=work & h_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                acc += tl.dot(activated, tl.trans(w), out_dtype=tl.float32)
            else:
                w = tl.load(
                    down_w_ptr
                    + expert * stride_we
                    + offs_k[:, None] * stride_wi
                    + offs_h[None, :] * stride_wh,
                    mask=work & k_mask[:, None] & h_mask[None, :],
                    other=0.0,
                )
                acc += tl.dot(activated, w)

        tl.atomic_add(
            accum_ptr + token_ids[:, None] * stride_om + offs_h[None, :] * stride_oh,
            acc * route[:, None],
            sem="relaxed",
            mask=m_mask[:, None] & h_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_expert_grouped_compact_down_partial_kernel(
        act_ptr,
        down_w_ptr,
        expert_ids_ptr,
        unique_or_count_ptr,
        counts_ptr,
        dense_assign_ptr,
        dense_route_ptr,
        partial_ptr,
        stride_ae: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_pa: tl.constexpr,
        stride_ph: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        ROWS: tl.constexpr,
        ASSIGNMENTS: tl.constexpr,
        CANDIDATE_SLOTS: tl.constexpr,
        ACTIVE_LIST: tl.constexpr,
        EXPERT_GRID: tl.constexpr,
        COALESCED_WEIGHTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        EXPERTS_PER_PROGRAM: tl.constexpr,
        EMPTY_EXPERT_EARLY_EXIT: tl.constexpr,
        ACTIVE_LIST_EARLY_EXIT: tl.constexpr,
        L2_GROUPED_GRID: tl.constexpr,
        NUM_CANDIDATE_GROUPS: tl.constexpr,
        NUM_PID_N: tl.constexpr,
        L2_GROUP_SIZE: tl.constexpr,
    ):
        if L2_GROUPED_GRID:
            pid = tl.program_id(0)
            num_pid_in_group = L2_GROUP_SIZE * NUM_PID_N
            group_id = pid // num_pid_in_group
            first_candidate = group_id * L2_GROUP_SIZE
            group_size = tl.minimum(
                NUM_CANDIDATE_GROUPS - first_candidate,
                L2_GROUP_SIZE,
            )
            pid_in_group = pid % num_pid_in_group
            candidate_group = first_candidate + (pid_in_group % group_size)
            pid_m = 0
            pid_h = pid_in_group // group_size
        else:
            candidate_group = tl.program_id(0)
            pid_m = tl.program_id(1)
            pid_h = tl.program_id(2)
        if ACTIVE_LIST_EARLY_EXIT and ACTIVE_LIST and EXPERTS_PER_PROGRAM == 1:
            active_count = tl.load(unique_or_count_ptr).to(tl.int64)
            if candidate_group >= active_count:
                return
        if EMPTY_EXPERT_EARLY_EXIT and EXPERT_GRID and EXPERTS_PER_PROGRAM == 1:
            if tl.load(counts_ptr + candidate_group) <= 0:
                return
        for candidate_offset in tl.static_range(0, EXPERTS_PER_PROGRAM):
            candidate = candidate_group * EXPERTS_PER_PROGRAM + candidate_offset
            candidate_valid = candidate < CANDIDATE_SLOTS
            if EXPERT_GRID:
                expert = candidate
                is_first = candidate_valid
            elif ACTIVE_LIST:
                active_count = tl.load(unique_or_count_ptr).to(tl.int64)
                is_first = candidate_valid & (candidate < active_count)
                expert = tl.load(
                    expert_ids_ptr + candidate,
                    mask=is_first,
                    other=0,
                ).to(tl.int64)
            else:
                expert = tl.load(
                    expert_ids_ptr + candidate,
                    mask=candidate_valid,
                    other=0,
                ).to(tl.int64)
                is_first = candidate_valid & (
                    tl.load(unique_or_count_ptr + candidate, mask=candidate_valid, other=0)
                    != 0
                )
            count = tl.load(counts_ptr + expert, mask=is_first, other=0).to(tl.int64)
            count = tl.where(is_first, count, 0)
            work = is_first & (count > 0)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
            m_mask = work & (offs_m < count)
            h_mask = offs_h < H
            assign_ids = tl.load(
                dense_assign_ptr + expert * ROWS + offs_m,
                mask=m_mask,
                other=0,
            ).to(tl.int64)
            route = tl.load(
                dense_route_ptr + expert * ROWS + offs_m,
                mask=m_mask,
                other=0.0,
            ).to(tl.float32)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, I, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < I
                activated = tl.load(
                    act_ptr
                    + candidate * stride_ae
                    + offs_m[:, None] * stride_am
                    + offs_k[None, :] * stride_ai,
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                if COALESCED_WEIGHTS:
                    w = tl.load(
                        down_w_ptr
                        + expert * stride_we
                        + offs_h[:, None] * stride_wh
                        + offs_k[None, :] * stride_wi,
                        mask=work & h_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(activated, tl.trans(w), out_dtype=tl.float32)
                else:
                    w = tl.load(
                        down_w_ptr
                        + expert * stride_we
                        + offs_k[:, None] * stride_wi
                        + offs_h[None, :] * stride_wh,
                        mask=work & k_mask[:, None] & h_mask[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(activated, w)

            tl.store(
                partial_ptr
                + assign_ids[:, None] * stride_pa
                + offs_h[None, :] * stride_ph,
                acc * route[:, None],
                mask=m_mask[:, None] & h_mask[None, :],
            )

    @triton.jit
    def _qwen3_moe_assignment_reduce_kernel(
        partial_ptr,
        out_ptr,
        stride_pa: tl.constexpr,
        stride_ph: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        token = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        base = token * TOP_K
        for top_idx in tl.static_range(0, 16):
            if top_idx < TOP_K:
                part = tl.load(
                    partial_ptr + (base + top_idx) * stride_pa + offs_h * stride_ph,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                acc += part

        tl.store(
            out_ptr + token * stride_om + offs_h * stride_oh,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )

    @triton.jit
    def _qwen3_moe_assignment_reduce_gemma4_post_kernel(
        partial_ptr,
        shared_ptr,
        shared_weight_ptr,
        expert_weight_ptr,
        final_weight_ptr,
        residual_ptr,
        out_ptr,
        layer_scalar_ptr,
        next_norm_weight_ptr,
        next_norm_out_ptr,
        stride_pa: tl.constexpr,
        stride_ph: tl.constexpr,
        stride_sm: tl.constexpr,
        stride_sh: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_nm: tl.constexpr,
        stride_nh: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        EPS: tl.constexpr,
        FUSE_LAYER_SCALAR: tl.constexpr,
        WRITE_NEXT_NORM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        token = tl.program_id(0)
        offs_h = tl.arange(0, BLOCK_SIZE)
        h_mask = offs_h < H
        safe_h = tl.minimum(offs_h, H - 1)

        expert = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        assignment_base = token * TOP_K
        for top_idx in tl.static_range(0, TOP_K):
            expert += tl.load(
                partial_ptr
                + (assignment_base + top_idx) * stride_pa
                + safe_h * stride_ph,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)

        # Match the standalone assignment reduction's BF16/FP16 output store.
        expert = expert.to(out_ptr.dtype.element_ty).to(tl.float32)
        shared = tl.load(
            shared_ptr + token * stride_sm + safe_h * stride_sh,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        shared_var = tl.sum(shared * shared, axis=0) / H
        expert_var = tl.sum(expert * expert, axis=0) / H
        shared_weight = tl.load(
            shared_weight_ptr + safe_h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        expert_weight = tl.load(
            expert_weight_ptr + safe_h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)

        # Preserve both branch RMSNorm stores and the in-place BF16 branch add.
        shared_norm = (
            shared * (1.0 / tl.sqrt(shared_var + EPS)) * shared_weight
        ).to(out_ptr.dtype.element_ty)
        expert_norm = (
            expert * (1.0 / tl.sqrt(expert_var + EPS)) * expert_weight
        ).to(out_ptr.dtype.element_ty)
        merged = (shared_norm + expert_norm).to(out_ptr.dtype.element_ty).to(tl.float32)
        merged_var = tl.sum(merged * merged, axis=0) / H
        final_weight = tl.load(
            final_weight_ptr + safe_h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        final_norm = (
            merged * (1.0 / tl.sqrt(merged_var + EPS)) * final_weight
        ).to(out_ptr.dtype.element_ty).to(tl.float32)
        residual = tl.load(
            residual_ptr + token * stride_rm + safe_h * stride_rh,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        final_hidden = residual + final_norm
        if FUSE_LAYER_SCALAR:
            # Preserve the two BF16 staging points in the eager chain:
            # post-MoE residual store, then in-place layer-scalar multiply.
            staged_hidden = final_hidden.to(out_ptr.dtype.element_ty).to(tl.float32)
            layer_scalar = tl.load(layer_scalar_ptr).to(tl.float32)
            scaled_hidden = (
                staged_hidden * layer_scalar
            ).to(out_ptr.dtype.element_ty)
            tl.store(
                out_ptr + token * stride_om + safe_h * stride_oh,
                scaled_hidden,
                mask=h_mask,
            )

            if WRITE_NEXT_NORM:
                scaled_fp32 = scaled_hidden.to(tl.float32)
                next_var = tl.sum(scaled_fp32 * scaled_fp32, axis=0) / H
                next_weight = tl.load(
                    next_norm_weight_ptr + safe_h,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                next_norm = (
                    scaled_fp32
                    * (1.0 / tl.sqrt(next_var + EPS))
                    * next_weight
                )
                tl.store(
                    next_norm_out_ptr
                    + token * stride_nm
                    + safe_h * stride_nh,
                    next_norm,
                    mask=h_mask,
                )
        else:
            tl.store(
                out_ptr + token * stride_om + safe_h * stride_oh,
                final_hidden,
                mask=h_mask,
            )

    @triton.jit
    def _qwen3_moe_expert_grouped_compact_down_token_accum_kernel(
        act_ptr,
        down_w_ptr,
        selected_experts_ptr,
        routing_ptr,
        expert_to_candidate_ptr,
        out_ptr,
        stride_ae: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_se_m: tl.constexpr,
        stride_se_k: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        token = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for top_idx in tl.static_range(0, TOP_K):
            expert = tl.load(
                selected_experts_ptr + token * stride_se_m + top_idx * stride_se_k
            ).to(tl.int64)
            candidate = tl.load(expert_to_candidate_ptr + expert).to(tl.int64)
            route = tl.load(
                routing_ptr + token * stride_rm + top_idx * stride_rk
            ).to(tl.float32)

            expert_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
            for k_start in range(0, I, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < I
                activated = tl.load(
                    act_ptr
                    + candidate * stride_ae
                    + token * stride_am
                    + offs_k * stride_ai,
                    mask=k_mask,
                    other=0.0,
                )
                w = tl.load(
                    down_w_ptr
                    + expert * stride_we
                    + offs_k[:, None] * stride_wi
                    + offs_h[None, :] * stride_wh,
                    mask=k_mask[:, None] & h_mask[None, :],
                    other=0.0,
                )
                expert_acc += tl.sum(activated[:, None].to(tl.float32) * w.to(tl.float32), axis=0)
            acc += expert_acc * route

        tl.store(
            out_ptr + token * stride_om + offs_h * stride_oh,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )

    @triton.jit
    def _qwen3_moe_shared_route_gate_swiglu_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        selected_experts_ptr,
        act_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_se_m: tl.constexpr,
        stride_se_k: tl.constexpr,
        stride_ae: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        H: tl.constexpr,
        I: tl.constexpr,
        ROWS: tl.constexpr,
        TOP_K: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        COALESCED_WEIGHTS: tl.constexpr,
    ):
        top_idx = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        expert = tl.load(selected_experts_ptr + top_idx * stride_se_k).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < ROWS
        n_mask = offs_n < I

        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < H
            x = tl.load(
                hidden_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            if COALESCED_WEIGHTS:
                gate_w = tl.load(
                    gate_up_w_ptr
                    + expert * stride_we
                    + offs_n[:, None] * stride_wo
                    + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                up_w = tl.load(
                    gate_up_w_ptr
                    + expert * stride_we
                    + (offs_n[:, None] + I) * stride_wo
                    + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                gate_acc += tl.dot(x, tl.trans(gate_w), out_dtype=tl.float32)
                up_acc += tl.dot(x, tl.trans(up_w), out_dtype=tl.float32)
            else:
                gate_w = tl.load(
                    gate_up_w_ptr
                    + expert * stride_we
                    + offs_k[:, None] * stride_wk
                    + offs_n[None, :] * stride_wo,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
                up_w = tl.load(
                    gate_up_w_ptr
                    + expert * stride_we
                    + offs_k[:, None] * stride_wk
                    + (offs_n[None, :] + I) * stride_wo,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
                gate_acc += tl.dot(x, gate_w)
                up_acc += tl.dot(x, up_w)

        if ACT == 1:
            c = 0.7978845608028654
            inner = c * (gate_acc + 0.044715 * gate_acc * gate_acc * gate_acc)
            gate_act = gate_acc * tl.sigmoid(2.0 * inner)
        else:
            gate_act = gate_acc * tl.sigmoid(gate_acc)
        activated = gate_act * up_acc

        tl.store(
            act_ptr
            + top_idx * stride_ae
            + offs_m[:, None] * stride_am
            + offs_n[None, :] * stride_ai,
            activated.to(act_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_shared_route_gate_k_split_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        selected_experts_ptr,
        partial_ptr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_se_k: tl.constexpr,
        stride_ps: tl.constexpr,
        stride_pe: tl.constexpr,
        stride_pn: tl.constexpr,
        H: tl.constexpr,
        I: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        top_idx = tl.program_id(0)
        pid_n = tl.program_id(1)
        split_id = tl.program_id(2)
        expert = tl.load(selected_experts_ptr + top_idx * stride_se_k).to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < I
        split_start = split_id * SPLIT_K

        gate_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        up_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k_offset in range(0, SPLIT_K, BLOCK_K):
            offs_k = split_start + k_offset + tl.arange(0, BLOCK_K)
            k_mask = (offs_k < H) & (offs_k < split_start + SPLIT_K)
            x = tl.load(
                hidden_ptr + offs_k * stride_hk,
                mask=k_mask,
                other=0.0,
            )
            gate_w = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wk
                + offs_n[None, :] * stride_wo,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            up_w = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wk
                + (offs_n[None, :] + I) * stride_wo,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            gate_acc += tl.sum(
                tl.dot(x[None, :], gate_w, out_dtype=tl.float32),
                axis=0,
            )
            up_acc += tl.sum(
                tl.dot(x[None, :], up_w, out_dtype=tl.float32),
                axis=0,
            )

        partial_base = (
            partial_ptr
            + split_id * stride_ps
            + top_idx * stride_pe
        )
        tl.store(
            partial_base + offs_n * stride_pn,
            gate_acc,
            mask=n_mask,
        )
        tl.store(
            partial_base + (offs_n + I) * stride_pn,
            up_acc,
            mask=n_mask,
        )

    @triton.jit
    def _qwen3_moe_shared_route_gate_k_reduce_swiglu_kernel(
        partial_ptr,
        act_ptr,
        stride_ps: tl.constexpr,
        stride_pe: tl.constexpr,
        stride_pn: tl.constexpr,
        stride_ae: tl.constexpr,
        stride_ai: tl.constexpr,
        I: tl.constexpr,
        ACT: tl.constexpr,
        K_SPLITS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        top_idx = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_s = tl.arange(0, K_SPLITS)
        n_mask = offs_n < I
        partial_base = partial_ptr + top_idx * stride_pe

        gate_parts = tl.load(
            partial_base
            + offs_s[:, None] * stride_ps
            + offs_n[None, :] * stride_pn,
            mask=n_mask[None, :],
            other=0.0,
        )
        up_parts = tl.load(
            partial_base
            + offs_s[:, None] * stride_ps
            + (offs_n[None, :] + I) * stride_pn,
            mask=n_mask[None, :],
            other=0.0,
        )
        gate_acc = tl.sum(gate_parts, axis=0)
        up_acc = tl.sum(up_parts, axis=0)
        if ACT == 1:
            c = 0.7978845608028654
            inner = c * (gate_acc + 0.044715 * gate_acc * gate_acc * gate_acc)
            gate_act = gate_acc * tl.sigmoid(2.0 * inner)
        else:
            gate_act = gate_acc * tl.sigmoid(gate_acc)

        tl.store(
            act_ptr + top_idx * stride_ae + offs_n * stride_ai,
            (gate_act * up_acc).to(act_ptr.dtype.element_ty),
            mask=n_mask,
        )

    @triton.jit
    def _qwen3_moe_shared_route_gate_up_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        selected_experts_ptr,
        gate_up_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_se_k: tl.constexpr,
        stride_ge: tl.constexpr,
        stride_gm: tl.constexpr,
        stride_gn: tl.constexpr,
        H: tl.constexpr,
        TWO_I: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        top_idx = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        expert = tl.load(selected_experts_ptr + top_idx * stride_se_k).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < ROWS
        n_mask = offs_n < TWO_I

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < H
            x = tl.load(
                hidden_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wk
                + offs_n[None, :] * stride_wo,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(x, w)

        tl.store(
            gate_up_ptr
            + top_idx * stride_ge
            + offs_m[:, None] * stride_gm
            + offs_n[None, :] * stride_gn,
            acc.to(gate_up_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_shared_route_swiglu_down_accum_kernel(
        gate_up_ptr,
        down_w_ptr,
        selected_experts_ptr,
        routing_ptr,
        accum_ptr,
        stride_ge: tl.constexpr,
        stride_gm: tl.constexpr,
        stride_gn: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        ROWS: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        top_idx = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_h = tl.program_id(2)
        expert = tl.load(selected_experts_ptr + top_idx).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < ROWS
        h_mask = offs_h < H
        route = tl.load(
            routing_ptr + offs_m * stride_rm + top_idx * stride_rk,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I
            gate = tl.load(
                gate_up_ptr
                + top_idx * stride_ge
                + offs_m[:, None] * stride_gm
                + offs_k[None, :] * stride_gn,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                gate_up_ptr
                + top_idx * stride_ge
                + offs_m[:, None] * stride_gm
                + (offs_k[None, :] + I) * stride_gn,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if ACT == 1:
                c = 0.7978845608028654
                inner = c * (gate + 0.044715 * gate * gate * gate)
                gate_act = gate * tl.sigmoid(2.0 * inner)
            else:
                gate_act = gate * tl.sigmoid(gate)
            activated = gate_act * up

            w = tl.load(
                down_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wi
                + offs_h[None, :] * stride_wh,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(activated.to(gate_up_ptr.dtype.element_ty), w)

        tl.atomic_add(
            accum_ptr + offs_m[:, None] * stride_om + offs_h[None, :] * stride_oh,
            acc * route[:, None],
            sem="relaxed",
            mask=m_mask[:, None] & h_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_shared_route_down_accum_kernel(
        act_ptr,
        down_w_ptr,
        selected_experts_ptr,
        routing_ptr,
        accum_ptr,
        stride_ae: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        ROWS: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        COALESCED_WEIGHTS: tl.constexpr,
    ):
        top_idx = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_h = tl.program_id(2)
        expert = tl.load(selected_experts_ptr + top_idx).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < ROWS
        h_mask = offs_h < H
        route = tl.load(
            routing_ptr + offs_m * stride_rm + top_idx * stride_rk,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I
            activated = tl.load(
                act_ptr
                + top_idx * stride_ae
                + offs_m[:, None] * stride_am
                + offs_k[None, :] * stride_ai,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            if COALESCED_WEIGHTS:
                w = tl.load(
                    down_w_ptr
                    + expert * stride_we
                    + offs_h[:, None] * stride_wh
                    + offs_k[None, :] * stride_wi,
                    mask=h_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                acc += tl.dot(activated, tl.trans(w), out_dtype=tl.float32)
            else:
                w = tl.load(
                    down_w_ptr
                    + expert * stride_we
                    + offs_k[:, None] * stride_wi
                    + offs_h[None, :] * stride_wh,
                    mask=k_mask[:, None] & h_mask[None, :],
                    other=0.0,
                )
                acc += tl.dot(activated, w)

        tl.atomic_add(
            accum_ptr + offs_m[:, None] * stride_om + offs_h[None, :] * stride_oh,
            acc * route[:, None],
            sem="relaxed",
            mask=m_mask[:, None] & h_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_accum_add_residual_kernel(
        accum_ptr,
        residual_ptr,
        out_ptr,
        stride_am: tl.constexpr,
        stride_ah: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        H: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (row < ROWS) & (offs_h < H)
        acc = tl.load(
            accum_ptr + row * stride_am + offs_h * stride_ah,
            mask=mask,
            other=0.0,
        )
        res = tl.load(
            residual_ptr + row * stride_rm + offs_h * stride_rh,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            out_ptr + row * stride_om + offs_h * stride_oh,
            (acc + res).to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _qwen3_moe_shared_route_down_token_accum_kernel(
        act_ptr,
        down_w_ptr,
        selected_experts_ptr,
        routing_ptr,
        out_ptr,
        stride_ae: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_se_m: tl.constexpr,
        stride_se_k: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        token = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for top_idx in tl.static_range(0, TOP_K):
            expert = tl.load(
                selected_experts_ptr + token * stride_se_m + top_idx * stride_se_k
            ).to(tl.int64)
            route = tl.load(
                routing_ptr + token * stride_rm + top_idx * stride_rk
            ).to(tl.float32)

            expert_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
            for k_start in range(0, I, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < I
                activated = tl.load(
                    act_ptr
                    + top_idx * stride_ae
                    + token * stride_am
                    + offs_k * stride_ai,
                    mask=k_mask,
                    other=0.0,
                )
                w = tl.load(
                    down_w_ptr
                    + expert * stride_we
                    + offs_k[:, None] * stride_wi
                    + offs_h[None, :] * stride_wh,
                    mask=k_mask[:, None] & h_mask[None, :],
                    other=0.0,
                )
                expert_acc += tl.sum(activated[:, None].to(tl.float32) * w.to(tl.float32), axis=0)
            acc += expert_acc * route

        tl.store(
            out_ptr + token * stride_om + offs_h * stride_oh,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )

    @triton.jit
    def _qwen3_moe_shared_route_down_partial_kernel(
        act_ptr,
        down_w_ptr,
        selected_experts_ptr,
        routing_ptr,
        partial_ptr,
        stride_ae: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rk: tl.constexpr,
        stride_pe: tl.constexpr,
        stride_pm: tl.constexpr,
        stride_ph: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        COALESCED_WEIGHTS: tl.constexpr,
    ):
        top_idx = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_h = tl.program_id(2)
        expert = tl.load(selected_experts_ptr + top_idx).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < ROWS
        h_mask = offs_h < H
        route = tl.load(
            routing_ptr + offs_m * stride_rm + top_idx * stride_rk,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I
            activated = tl.load(
                act_ptr
                + top_idx * stride_ae
                + offs_m[:, None] * stride_am
                + offs_k[None, :] * stride_ai,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            if COALESCED_WEIGHTS:
                w = tl.load(
                    down_w_ptr
                    + expert * stride_we
                    + offs_h[:, None] * stride_wh
                    + offs_k[None, :] * stride_wi,
                    mask=h_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                acc += tl.dot(activated, tl.trans(w), out_dtype=tl.float32)
            else:
                w = tl.load(
                    down_w_ptr
                    + expert * stride_we
                    + offs_k[:, None] * stride_wi
                    + offs_h[None, :] * stride_wh,
                    mask=k_mask[:, None] & h_mask[None, :],
                    other=0.0,
                )
                acc += tl.dot(activated, w)

        tl.store(
            partial_ptr
            + top_idx * stride_pe
            + offs_m[:, None] * stride_pm
            + offs_h[None, :] * stride_ph,
            acc * route[:, None],
            mask=m_mask[:, None] & h_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_shared_route_reduce_kernel(
        partial_ptr,
        out_ptr,
        stride_pe: tl.constexpr,
        stride_pm: tl.constexpr,
        stride_ph: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        H: tl.constexpr,
        ROWS: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < ROWS
        h_mask = offs_h < H

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for top_idx in tl.static_range(0, TOP_K):
            part = tl.load(
                partial_ptr
                + top_idx * stride_pe
                + offs_m[:, None] * stride_pm
                + offs_h[None, :] * stride_ph,
                mask=m_mask[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += part

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_h[None, :] * stride_oh,
            acc.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & h_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_route_matrix_gate_swiglu_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        selected_experts_ptr,
        act_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_se_m: tl.constexpr,
        stride_se_k: tl.constexpr,
        stride_ae: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        H: tl.constexpr,
        I: tl.constexpr,
        ROWS: tl.constexpr,
        TOP_K: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        top_idx = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        base_m = pid_m * BLOCK_M
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < I

        for row_off in tl.static_range(0, 16):
            if row_off < BLOCK_M:
                row = base_m + row_off
                row_valid = row < ROWS
                expert = tl.load(
                    selected_experts_ptr + row * stride_se_m + top_idx * stride_se_k,
                    mask=row_valid,
                    other=0,
                ).to(tl.int64)

                gate_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                up_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                for k_start in range(0, H, BLOCK_K):
                    offs_k = k_start + tl.arange(0, BLOCK_K)
                    k_mask = offs_k < H
                    x = tl.load(
                        hidden_ptr + row * stride_hm + offs_k * stride_hk,
                        mask=row_valid & k_mask,
                        other=0.0,
                    )
                    gate_w = tl.load(
                        gate_up_w_ptr
                        + expert * stride_we
                        + offs_k[:, None] * stride_wk
                        + offs_n[None, :] * stride_wo,
                        mask=k_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    )
                    up_w = tl.load(
                        gate_up_w_ptr
                        + expert * stride_we
                        + offs_k[:, None] * stride_wk
                        + (offs_n[None, :] + I) * stride_wo,
                        mask=k_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    )
                    x_f = x.to(tl.float32)
                    gate_acc += tl.sum(x_f[:, None] * gate_w.to(tl.float32), axis=0)
                    up_acc += tl.sum(x_f[:, None] * up_w.to(tl.float32), axis=0)

                if ACT == 1:
                    c = 0.7978845608028654
                    inner = c * (gate_acc + 0.044715 * gate_acc * gate_acc * gate_acc)
                    gate_act = gate_acc * tl.sigmoid(2.0 * inner)
                else:
                    gate_act = gate_acc * tl.sigmoid(gate_acc)
                activated = gate_act * up_acc

                tl.store(
                    act_ptr
                    + top_idx * stride_ae
                    + row * stride_am
                    + offs_n * stride_ai,
                    activated.to(act_ptr.dtype.element_ty),
                    mask=row_valid & n_mask,
                )

    @triton.jit
    def _qwen3_moe_route_matrix_down_accum_kernel(
        act_ptr,
        down_w_ptr,
        selected_experts_ptr,
        routing_ptr,
        accum_ptr,
        stride_ae: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_se_m: tl.constexpr,
        stride_se_k: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        ROWS: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        top_idx = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_h = tl.program_id(2)
        base_m = pid_m * BLOCK_M
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H

        for row_off in tl.static_range(0, 16):
            if row_off < BLOCK_M:
                row = base_m + row_off
                row_valid = row < ROWS
                expert = tl.load(
                    selected_experts_ptr + row * stride_se_m + top_idx * stride_se_k,
                    mask=row_valid,
                    other=0,
                ).to(tl.int64)
                route = tl.load(
                    routing_ptr + row * stride_rm + top_idx * stride_rk,
                    mask=row_valid,
                    other=0.0,
                ).to(tl.float32)

                acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                for k_start in range(0, I, BLOCK_K):
                    offs_k = k_start + tl.arange(0, BLOCK_K)
                    k_mask = offs_k < I
                    activated = tl.load(
                        act_ptr
                        + top_idx * stride_ae
                        + row * stride_am
                        + offs_k * stride_ai,
                        mask=row_valid & k_mask,
                        other=0.0,
                    )
                    w = tl.load(
                        down_w_ptr
                        + expert * stride_we
                        + offs_k[:, None] * stride_wi
                        + offs_h[None, :] * stride_wh,
                        mask=k_mask[:, None] & h_mask[None, :],
                        other=0.0,
                    )
                    acc += tl.sum(activated[:, None].to(tl.float32) * w.to(tl.float32), axis=0)

                tl.atomic_add(
                    accum_ptr + row * stride_om + offs_h * stride_oh,
                    acc * route,
                    sem="relaxed",
                    mask=row_valid & h_mask,
                )

    @triton.jit
    def _qwen3_moe_segmented_grid_gate_up_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        sorted_tokens_ptr,
        counts_ptr,
        starts_ptr,
        gate_up_out_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        H: tl.constexpr,
        TWO_I: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        expert = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        count = tl.load(counts_ptr + expert).to(tl.int64)
        start = tl.load(starts_ptr + expert).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < count
        n_mask = offs_n < TWO_I
        row_ids = start + offs_m
        token_ids = tl.load(
            sorted_tokens_ptr + row_ids,
            mask=row_mask,
            other=0,
        ).to(tl.int64)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < H
            x = tl.load(
                hidden_ptr + token_ids[:, None] * stride_hm + offs_k[None, :] * stride_hk,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wk
                + offs_n[None, :] * stride_wo,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(x, w)

        tl.store(
            gate_up_out_ptr + row_ids[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc.to(gate_up_out_ptr.dtype.element_ty),
            mask=row_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_segmented_grid_down_accum_kernel(
        gate_up_ptr,
        down_w_ptr,
        sorted_tokens_ptr,
        sorted_route_ptr,
        counts_ptr,
        starts_ptr,
        accum_ptr,
        stride_gm: tl.constexpr,
        stride_gk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        expert = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_h = tl.program_id(2)

        count = tl.load(counts_ptr + expert).to(tl.int64)
        start = tl.load(starts_ptr + expert).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < count
        h_mask = offs_h < H
        row_ids = start + offs_m
        token_ids = tl.load(
            sorted_tokens_ptr + row_ids,
            mask=row_mask,
            other=0,
        ).to(tl.int64)
        route = tl.load(
            sorted_route_ptr + row_ids,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I
            gate = tl.load(
                gate_up_ptr + row_ids[:, None] * stride_gm + offs_k[None, :] * stride_gk,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                gate_up_ptr
                + row_ids[:, None] * stride_gm
                + (offs_k[None, :] + I) * stride_gk,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if ACT == 1:
                c = 0.7978845608028654
                inner = c * (gate + 0.044715 * gate * gate * gate)
                gate_act = gate * tl.sigmoid(2.0 * inner)
            else:
                gate_act = gate * tl.sigmoid(gate)
            activated = (gate_act * up).to(gate_up_ptr.dtype.element_ty)

            w = tl.load(
                down_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wi
                + offs_h[None, :] * stride_wh,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(activated, w)

        tl.atomic_add(
            accum_ptr + token_ids[:, None] * stride_om + offs_h[None, :] * stride_oh,
            acc * route[:, None],
            sem="relaxed",
            mask=row_mask[:, None] & h_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_segmented_gate_swiglu_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        sorted_tokens_ptr,
        tile_experts_ptr,
        tile_starts_ptr,
        tile_lengths_ptr,
        num_tiles_ptr,
        act_out_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        H: tl.constexpr,
        I: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        MAX_TILES: tl.constexpr,
    ):
        pid_tile = tl.program_id(0)
        pid_n = tl.program_id(1)

        num_tiles = tl.load(num_tiles_ptr).to(tl.int64)
        num_tiles = tl.maximum(0, tl.minimum(num_tiles, MAX_TILES))
        if pid_tile >= num_tiles:
            return

        tile_start = tl.load(tile_starts_ptr + pid_tile).to(tl.int64)
        tile_len = tl.load(tile_lengths_ptr + pid_tile).to(tl.int64)
        if tile_len <= 0:
            return
        expert = tl.load(tile_experts_ptr + pid_tile).to(tl.int64)

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < tile_len
        n_mask = offs_n < I
        row_ids = tile_start + offs_m
        token_ids = tl.load(
            sorted_tokens_ptr + row_ids,
            mask=row_mask,
            other=0,
        ).to(tl.int64)

        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < H
            x = tl.load(
                hidden_ptr + token_ids[:, None] * stride_hm + offs_k[None, :] * stride_hk,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            gate_w = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wk
                + offs_n[None, :] * stride_wo,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            up_w = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wk
                + (offs_n[None, :] + I) * stride_wo,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            gate_acc += tl.dot(x, gate_w)
            up_acc += tl.dot(x, up_w)

        if ACT == 1:
            c = 0.7978845608028654
            inner = c * (gate_acc + 0.044715 * gate_acc * gate_acc * gate_acc)
            gate_act = gate_acc * tl.sigmoid(2.0 * inner)
        else:
            gate_act = gate_acc * tl.sigmoid(gate_acc)
        activated = gate_act * up_acc

        tl.store(
            act_out_ptr + row_ids[:, None] * stride_om + offs_n[None, :] * stride_on,
            activated.to(act_out_ptr.dtype.element_ty),
            mask=row_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_segmented_down_from_act_accum_kernel(
        act_ptr,
        down_w_ptr,
        sorted_tokens_ptr,
        sorted_route_ptr,
        tile_experts_ptr,
        tile_starts_ptr,
        tile_lengths_ptr,
        num_tiles_ptr,
        accum_ptr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        MAX_TILES: tl.constexpr,
    ):
        pid_tile = tl.program_id(0)
        pid_h = tl.program_id(1)

        num_tiles = tl.load(num_tiles_ptr).to(tl.int64)
        num_tiles = tl.maximum(0, tl.minimum(num_tiles, MAX_TILES))
        if pid_tile >= num_tiles:
            return

        tile_start = tl.load(tile_starts_ptr + pid_tile).to(tl.int64)
        tile_len = tl.load(tile_lengths_ptr + pid_tile).to(tl.int64)
        if tile_len <= 0:
            return
        expert = tl.load(tile_experts_ptr + pid_tile).to(tl.int64)

        offs_m = tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < tile_len
        h_mask = offs_h < H
        row_ids = tile_start + offs_m
        token_ids = tl.load(
            sorted_tokens_ptr + row_ids,
            mask=row_mask,
            other=0,
        ).to(tl.int64)
        route = tl.load(
            sorted_route_ptr + row_ids,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I
            activated = tl.load(
                act_ptr + row_ids[:, None] * stride_am + offs_k[None, :] * stride_ai,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                down_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wi
                + offs_h[None, :] * stride_wh,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(activated, w)

        tl.atomic_add(
            accum_ptr + token_ids[:, None] * stride_om + offs_h[None, :] * stride_oh,
            acc * route[:, None],
            sem="relaxed",
            mask=row_mask[:, None] & h_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_segmented_down_from_act_partial_kernel(
        act_ptr,
        down_w_ptr,
        sorted_route_ptr,
        sorted_slots_ptr,
        tile_experts_ptr,
        tile_starts_ptr,
        tile_lengths_ptr,
        num_tiles_ptr,
        partial_ptr,
        stride_am: tl.constexpr,
        stride_ai: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_pm: tl.constexpr,
        stride_ph: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        MAX_TILES: tl.constexpr,
        SORTED_PARTIAL: tl.constexpr,
    ):
        pid_tile = tl.program_id(0)
        pid_h = tl.program_id(1)

        num_tiles = tl.load(num_tiles_ptr).to(tl.int64)
        num_tiles = tl.maximum(0, tl.minimum(num_tiles, MAX_TILES))
        if pid_tile >= num_tiles:
            return

        tile_start = tl.load(tile_starts_ptr + pid_tile).to(tl.int64)
        tile_len = tl.load(tile_lengths_ptr + pid_tile).to(tl.int64)
        if tile_len <= 0:
            return
        expert = tl.load(tile_experts_ptr + pid_tile).to(tl.int64)

        offs_m = tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < tile_len
        h_mask = offs_h < H
        row_ids = tile_start + offs_m
        if SORTED_PARTIAL:
            partial_rows = row_ids
        else:
            partial_rows = tl.load(
                sorted_slots_ptr + row_ids,
                mask=row_mask,
                other=0,
            ).to(tl.int64)
        route = tl.load(
            sorted_route_ptr + row_ids,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I
            activated = tl.load(
                act_ptr + row_ids[:, None] * stride_am + offs_k[None, :] * stride_ai,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                down_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wi
                + offs_h[None, :] * stride_wh,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(activated, w)

        tl.store(
            partial_ptr
            + partial_rows[:, None] * stride_pm
            + offs_h[None, :] * stride_ph,
            acc * route[:, None],
            mask=row_mask[:, None] & h_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_segmented_gate_up_single_accum_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        sorted_tokens_ptr,
        tile_experts_ptr,
        tile_starts_ptr,
        tile_lengths_ptr,
        num_tiles_ptr,
        gate_up_out_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        H: tl.constexpr,
        TWO_I: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        MAX_TILES: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        # Match the grouped 1-D scheduling used by mature fused-MoE GEMMs:
        # keep several M tiles close while walking N so expert weights remain hot.
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(TWO_I, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(MAX_TILES - first_pid_m, GROUP_SIZE_M)
        pid_in_group = pid % num_pid_in_group
        pid_tile = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        num_tiles = tl.load(num_tiles_ptr).to(tl.int64)
        num_tiles = tl.maximum(0, tl.minimum(num_tiles, MAX_TILES))
        if pid_tile >= num_tiles:
            return

        tile_start = tl.load(tile_starts_ptr + pid_tile).to(tl.int64)
        tile_len = tl.load(tile_lengths_ptr + pid_tile).to(tl.int64)
        if tile_len <= 0:
            return
        expert = tl.load(tile_experts_ptr + pid_tile).to(tl.int64)

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < tile_len
        n_mask = offs_n < TWO_I
        row_ids = tile_start + offs_m
        token_ids = tl.load(
            sorted_tokens_ptr + row_ids,
            mask=row_mask,
            other=0,
        ).to(tl.int64)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < H
            x = tl.load(
                hidden_ptr
                + token_ids[:, None] * stride_hm
                + offs_k[None, :] * stride_hk,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_n[None, :] * stride_wo
                + offs_k[:, None] * stride_wk,
                mask=n_mask[None, :] & k_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(x, w)

        tl.store(
            gate_up_out_ptr
            + row_ids[:, None] * stride_om
            + offs_n[None, :] * stride_on,
            acc.to(gate_up_out_ptr.dtype.element_ty),
            mask=row_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_segmented_down_single_accum_partial_kernel(
        gate_up_ptr,
        down_w_ptr,
        sorted_route_ptr,
        sorted_slots_ptr,
        tile_experts_ptr,
        tile_starts_ptr,
        tile_lengths_ptr,
        num_tiles_ptr,
        partial_ptr,
        stride_gm: tl.constexpr,
        stride_gk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_pm: tl.constexpr,
        stride_ph: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        MAX_TILES: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        SORTED_PARTIAL: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(H, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(MAX_TILES - first_pid_m, GROUP_SIZE_M)
        pid_in_group = pid % num_pid_in_group
        pid_tile = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        num_tiles = tl.load(num_tiles_ptr).to(tl.int64)
        num_tiles = tl.maximum(0, tl.minimum(num_tiles, MAX_TILES))
        if pid_tile >= num_tiles:
            return

        tile_start = tl.load(tile_starts_ptr + pid_tile).to(tl.int64)
        tile_len = tl.load(tile_lengths_ptr + pid_tile).to(tl.int64)
        if tile_len <= 0:
            return
        expert = tl.load(tile_experts_ptr + pid_tile).to(tl.int64)

        offs_m = tl.arange(0, BLOCK_M)
        offs_h = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < tile_len
        h_mask = offs_h < H
        row_ids = tile_start + offs_m
        if SORTED_PARTIAL:
            partial_rows = row_ids
        else:
            partial_rows = tl.load(
                sorted_slots_ptr + row_ids,
                mask=row_mask,
                other=0,
            ).to(tl.int64)
        route = tl.load(
            sorted_route_ptr + row_ids,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I
            gate = tl.load(
                gate_up_ptr
                + row_ids[:, None] * stride_gm
                + offs_k[None, :] * stride_gk,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                gate_up_ptr
                + row_ids[:, None] * stride_gm
                + (offs_k[None, :] + I) * stride_gk,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if ACT == 1:
                c = 0.7978845608028654
                inner = c * (gate + 0.044715 * gate * gate * gate)
                gate_act = gate * tl.sigmoid(2.0 * inner)
            else:
                gate_act = gate * tl.sigmoid(gate)
            activated = (gate_act * up).to(gate_up_ptr.dtype.element_ty)
            w = tl.load(
                down_w_ptr
                + expert * stride_we
                + offs_h[None, :] * stride_wh
                + offs_k[:, None] * stride_wi,
                mask=h_mask[None, :] & k_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(activated, w)

        tl.store(
            partial_ptr
            + partial_rows[:, None] * stride_pm
            + offs_h[None, :] * stride_ph,
            acc * route[:, None],
            mask=row_mask[:, None] & h_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_segmented_partial_reduce_kernel(
        partial_ptr,
        slot_to_sorted_ptr,
        residual_ptr,
        out_ptr,
        stride_pm: tl.constexpr,
        stride_ph: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        ADD_RESIDUAL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SORTED_PARTIAL: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H
        slots = row * TOP_K + tl.arange(0, TOP_K)
        if SORTED_PARTIAL:
            partial_rows = tl.load(slot_to_sorted_ptr + slots).to(tl.int64)
        else:
            partial_rows = slots
        values = tl.load(
            partial_ptr
            + partial_rows[:, None] * stride_pm
            + offs_h[None, :] * stride_ph,
            mask=h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum(values, axis=0)
        if ADD_RESIDUAL:
            acc += tl.load(
                residual_ptr + row * stride_rm + offs_h * stride_rh,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            out_ptr + row * stride_om + offs_h * stride_oh,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )

    @triton.jit
    def _qwen3_moe_padded_bmm_atomic_route_pack_kernel(
        flat_experts_ptr,
        counters_ptr,
        token_pad_ptr,
        slot_to_padded_ptr,
        PADDED_COUNT: tl.constexpr,
        TOP_K: tl.constexpr,
        ASSIGNMENTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = slots < ASSIGNMENTS
        experts = tl.load(flat_experts_ptr + slots, mask=mask, other=0).to(tl.int64)
        ranks = tl.atomic_add(counters_ptr + experts, 1, mask=mask).to(tl.int64)
        padded_offsets = experts * PADDED_COUNT + ranks
        tl.store(
            token_pad_ptr + padded_offsets,
            slots // TOP_K,
            mask=mask,
        )
        tl.store(slot_to_padded_ptr + slots, padded_offsets, mask=mask)

    @triton.jit
    def _qwen3_moe_dominant_padded_bmm_route_pack_kernel(
        flat_experts_ptr,
        light_counters_ptr,
        heavy_counter_ptr,
        light_token_pad_ptr,
        heavy_token_ids_ptr,
        slot_to_light_ptr,
        slot_to_heavy_ptr,
        heavy_expert,
        LIGHT_PADDED_COUNT: tl.constexpr,
        TOP_K: tl.constexpr,
        ASSIGNMENTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = slots < ASSIGNMENTS
        experts = tl.load(flat_experts_ptr + slots, mask=mask, other=0).to(tl.int64)
        is_heavy = experts == heavy_expert
        counter_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        increments = tl.full((BLOCK,), 1, dtype=tl.int32)

        heavy_ranks = tl.atomic_add(
            heavy_counter_ptr + counter_offsets,
            increments,
            sem="relaxed",
            mask=mask & is_heavy,
        ).to(tl.int64)
        tl.store(
            heavy_token_ids_ptr + heavy_ranks,
            slots // TOP_K,
            mask=mask & is_heavy,
        )
        tl.store(
            slot_to_heavy_ptr + slots,
            heavy_ranks,
            mask=mask & is_heavy,
        )

        light_ranks = tl.atomic_add(
            light_counters_ptr + experts,
            increments,
            sem="relaxed",
            mask=mask & ~is_heavy,
        ).to(tl.int64)
        light_offsets = experts * LIGHT_PADDED_COUNT + light_ranks
        tl.store(
            light_token_pad_ptr + light_offsets,
            slots // TOP_K,
            mask=mask & ~is_heavy,
        )
        tl.store(
            slot_to_light_ptr + slots,
            light_offsets,
            mask=mask & ~is_heavy,
        )

    @triton.jit
    def _qwen3_moe_padded_bmm_activation_kernel(
        gate_up_ptr,
        activated_ptr,
        ELEMENTS: tl.constexpr,
        I: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < ELEMENTS
        rows = offsets // I
        cols = offsets - rows * I
        gate = tl.load(
            gate_up_ptr + rows * (2 * I) + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            gate_up_ptr + rows * (2 * I) + I + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if ACT == 1:
            inner = 0.7978845608028654 * (
                gate + 0.044715 * gate * gate * gate
            )
            gate_act = gate * tl.sigmoid(2.0 * inner)
        else:
            gate_act = gate * tl.sigmoid(gate)
        tl.store(activated_ptr + offsets, gate_act * up, mask=mask)

    @triton.jit
    def _qwen3_moe_padded_bmm_reduce_kernel(
        projected_ptr,
        slot_to_padded_ptr,
        route_ptr,
        residual_ptr,
        out_ptr,
        stride_pm: tl.constexpr,
        stride_ph: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_TOP_K: tl.constexpr,
        ADD_RESIDUAL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H
        top_offsets = tl.arange(0, BLOCK_TOP_K)
        top_mask = top_offsets < TOP_K
        slots = row * TOP_K + top_offsets
        padded_rows = tl.load(
            slot_to_padded_ptr + slots,
            mask=top_mask,
            other=0,
        ).to(tl.int64)
        route = tl.load(
            route_ptr + slots,
            mask=top_mask,
            other=0.0,
        ).to(tl.float32)
        values = tl.load(
            projected_ptr
            + padded_rows[:, None] * stride_pm
            + offs_h[None, :] * stride_ph,
            mask=top_mask[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum(values * route[:, None], axis=0)
        if ADD_RESIDUAL:
            acc += tl.load(
                residual_ptr + row * stride_rm + offs_h * stride_rh,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            out_ptr + row * stride_om + offs_h * stride_oh,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )

    @triton.jit
    def _qwen3_moe_dominant_padded_bmm_reduce_kernel(
        light_projected_ptr,
        heavy_projected_ptr,
        flat_experts_ptr,
        slot_to_light_ptr,
        slot_to_heavy_ptr,
        route_ptr,
        residual_ptr,
        out_ptr,
        heavy_expert,
        stride_lm: tl.constexpr,
        stride_lh: tl.constexpr,
        stride_hm: tl.constexpr,
        stride_hh: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_TOP_K: tl.constexpr,
        ADD_RESIDUAL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        h_mask = offs_h < H
        top_offsets = tl.arange(0, BLOCK_TOP_K)
        top_mask = top_offsets < TOP_K
        slots = row * TOP_K + top_offsets
        experts = tl.load(
            flat_experts_ptr + slots,
            mask=top_mask,
            other=0,
        ).to(tl.int64)
        is_heavy = experts == heavy_expert
        light_rows = tl.load(
            slot_to_light_ptr + slots,
            mask=top_mask,
            other=0,
        ).to(tl.int64)
        heavy_rows = tl.load(
            slot_to_heavy_ptr + slots,
            mask=top_mask,
            other=0,
        ).to(tl.int64)
        route = tl.load(
            route_ptr + slots,
            mask=top_mask,
            other=0.0,
        ).to(tl.float32)
        light_values = tl.load(
            light_projected_ptr
            + light_rows[:, None] * stride_lm
            + offs_h[None, :] * stride_lh,
            mask=top_mask[:, None] & ~is_heavy[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        heavy_values = tl.load(
            heavy_projected_ptr
            + heavy_rows[:, None] * stride_hm
            + offs_h[None, :] * stride_hh,
            mask=top_mask[:, None] & is_heavy[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum((light_values + heavy_values) * route[:, None], axis=0)
        if ADD_RESIDUAL:
            acc += tl.load(
                residual_ptr + row * stride_rm + offs_h * stride_rh,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            out_ptr + row * stride_om + offs_h * stride_oh,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )

    @triton.jit
    def _qwen3_moe_segmented_gate_up_kernel(
        hidden_ptr,
        gate_up_w_ptr,
        sorted_tokens_ptr,
        tile_experts_ptr,
        tile_starts_ptr,
        tile_lengths_ptr,
        gate_up_out_ptr,
        stride_hm: tl.constexpr,
        stride_hk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        H: tl.constexpr,
        TWO_I: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_tile = tl.program_id(0)
        pid_n = tl.program_id(1)

        tile_start = tl.load(tile_starts_ptr + pid_tile).to(tl.int64)
        tile_len = tl.load(tile_lengths_ptr + pid_tile).to(tl.int64)
        expert = tl.load(tile_experts_ptr + pid_tile).to(tl.int64)

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < tile_len
        n_mask = offs_n < TWO_I
        row_ids = tile_start + offs_m
        token_ids = tl.load(
            sorted_tokens_ptr + row_ids,
            mask=row_mask,
            other=0,
        ).to(tl.int64)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < H
            x = tl.load(
                hidden_ptr + token_ids[:, None] * stride_hm + offs_k[None, :] * stride_hk,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                gate_up_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wk
                + offs_n[None, :] * stride_wo,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(x, w)

        tl.store(
            gate_up_out_ptr + row_ids[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc.to(gate_up_out_ptr.dtype.element_ty),
            mask=row_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _qwen3_moe_segmented_down_accum_kernel(
        gate_up_ptr,
        down_w_ptr,
        sorted_tokens_ptr,
        sorted_route_ptr,
        tile_experts_ptr,
        tile_starts_ptr,
        tile_lengths_ptr,
        accum_ptr,
        stride_gm: tl.constexpr,
        stride_gk: tl.constexpr,
        stride_we: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        I: tl.constexpr,
        H: tl.constexpr,
        ACT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_tile = tl.program_id(0)
        pid_h = tl.program_id(1)

        tile_start = tl.load(tile_starts_ptr + pid_tile).to(tl.int64)
        tile_len = tl.load(tile_lengths_ptr + pid_tile).to(tl.int64)
        expert = tl.load(tile_experts_ptr + pid_tile).to(tl.int64)

        offs_m = tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < tile_len
        h_mask = offs_h < H
        row_ids = tile_start + offs_m
        token_ids = tl.load(
            sorted_tokens_ptr + row_ids,
            mask=row_mask,
            other=0,
        ).to(tl.int64)
        route = tl.load(
            sorted_route_ptr + row_ids,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, I, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < I
            gate = tl.load(
                gate_up_ptr + row_ids[:, None] * stride_gm + offs_k[None, :] * stride_gk,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                gate_up_ptr
                + row_ids[:, None] * stride_gm
                + (offs_k[None, :] + I) * stride_gk,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if ACT == 1:
                c = 0.7978845608028654
                inner = c * (gate + 0.044715 * gate * gate * gate)
                gate_act = gate * tl.sigmoid(2.0 * inner)
            else:
                gate_act = gate * tl.sigmoid(gate)
            activated = (gate_act * up).to(gate_up_ptr.dtype.element_ty)

            w = tl.load(
                down_w_ptr
                + expert * stride_we
                + offs_k[:, None] * stride_wi
                + offs_h[None, :] * stride_wh,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(activated, w)

        tl.atomic_add(
            accum_ptr + token_ids[:, None] * stride_om + offs_h[None, :] * stride_oh,
            acc * route[:, None],
            sem="relaxed",
            mask=row_mask[:, None] & h_mask[None, :],
        )


def qwen3_moe_topk_softmax(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    workspace: Optional[dict[str, torch.Tensor]] = None,
    expert_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return normalized top-k router weights from logits.

    This is equivalent to:
      top_logits, selected = torch.topk(logits, k)
      weights = softmax(top_logits)

    For Qwen3 MoE with norm_topk_prob=True this is also equivalent to
    topk(softmax(logits)) followed by top-k renormalization, but avoids the
    full-expert softmax and fuses the tiny top-k softmax path for decode.
    """
    if router_logits.ndim != 2:
        raise ValueError("router_logits must be [M, num_experts]")
    rows, num_experts = router_logits.shape
    top_k = min(int(top_k), int(num_experts))
    if expert_scale is not None and int(expert_scale.numel()) != int(num_experts):
        raise ValueError("expert_scale size must match num_experts")
    if rows == 0 or top_k <= 0:
        weights = router_logits.new_empty((rows, top_k))
        experts = torch.empty((rows, top_k), device=router_logits.device, dtype=torch.int64)
        return weights, experts

    use_triton = (
        _HAS_TRITON
        and router_logits.is_cuda
        and router_logits.is_contiguous()
        and not torch.is_grad_enabled()
        and top_k <= 16
        and _is_power_of_2(top_k)
        and num_experts <= 1024
        and (
            expert_scale is None
            or (
                expert_scale.is_cuda
                and expert_scale.is_contiguous()
                and expert_scale.device == router_logits.device
            )
        )
    )
    if not use_triton:
        top_logits, selected_experts = torch.topk(router_logits, top_k, dim=-1)
        routing_weights = torch.nn.functional.softmax(
            top_logits,
            dtype=torch.float32,
            dim=-1,
        ).to(router_logits.dtype)
        if expert_scale is not None:
            routing_weights = routing_weights * expert_scale.to(
                device=router_logits.device,
                dtype=routing_weights.dtype,
            ).reshape(-1)[selected_experts]
        return routing_weights, selected_experts

    weights = _workspace_tensor(
        workspace,
        "router_weights",
        (rows, top_k),
        device=router_logits.device,
        dtype=router_logits.dtype,
    )
    experts = _workspace_tensor(
        workspace,
        "router_experts",
        (rows, top_k),
        device=router_logits.device,
        dtype=torch.int64,
    )
    block_e = triton.next_power_of_2(num_experts)
    _qwen3_moe_topk_softmax_kernel[(rows,)](
        router_logits,
        weights,
        experts,
        router_logits if expert_scale is None else expert_scale,
        router_logits.stride(0),
        router_logits.stride(1),
        weights.stride(0),
        weights.stride(1),
        experts.stride(0),
        experts.stride(1),
        num_experts,
        TOP_K=top_k,
        BLOCK_E=block_e,
        APPLY_EXPERT_SCALE=expert_scale is not None,
        num_warps=4,
        num_stages=2,
    )
    return weights, experts


def qwen3_moe_compact_route_pack(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    num_experts: int,
    workspace: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the expert-major compact route used by B16 decode."""
    if workspace is None:
        raise ValueError("compact route pack requires a persistent workspace")
    if selected_experts.ndim != 2 or selected_experts.shape != routing_weights.shape:
        raise ValueError("selected experts and routing weights must be matching 2D tensors")
    rows, top_k = selected_experts.shape
    assignments = int(rows) * int(top_k)
    if (
        not _HAS_TRITON
        or not selected_experts.is_cuda
        or not routing_weights.is_cuda
        or selected_experts.dtype != torch.int64
        or not selected_experts.is_contiguous()
        or not routing_weights.is_contiguous()
        or torch.is_grad_enabled()
        or assignments > 128
    ):
        raise RuntimeError("compact route pack is not eligible for this input")

    counts = _workspace_tensor(
        workspace,
        "expert_grouped_compact_counts",
        (int(num_experts),),
        device=selected_experts.device,
        dtype=torch.int32,
    )
    dense_tokens = _workspace_tensor(
        workspace,
        "expert_grouped_compact_tokens",
        (int(num_experts), int(rows)),
        device=selected_experts.device,
        dtype=torch.int64,
    )
    dense_route = _workspace_tensor(
        workspace,
        "expert_grouped_compact_route",
        (int(num_experts), int(rows)),
        device=routing_weights.device,
        dtype=routing_weights.dtype,
    )
    dense_assign = _workspace_tensor(
        workspace,
        "expert_grouped_compact_assign",
        (int(num_experts), int(rows)),
        device=selected_experts.device,
        dtype=torch.int64,
    )
    pack_block = min(128, triton.next_power_of_2(max(assignments, 1)))
    _qwen3_moe_expert_grouped_compact_expert_pack_kernel[(int(num_experts),)](
        selected_experts,
        routing_weights,
        counts,
        dense_tokens,
        dense_route,
        dense_assign,
        assignments,
        ROWS=int(rows),
        TOP_K=int(top_k),
        BLOCK=pack_block,
        num_warps=4,
        num_stages=1,
    )
    return counts, dense_tokens, dense_route, dense_assign


def qwen3_moe_topk_softmax_compact_pack(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    workspace: dict[str, torch.Tensor],
    compact_workspace: dict[str, torch.Tensor],
    expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse exact B16 top-k selection with the expert-major route pack."""
    if router_logits.ndim != 2:
        raise ValueError("router logits must be [rows, experts]")
    rows, num_experts = router_logits.shape
    top_k = min(int(top_k), int(num_experts))
    eligible = bool(
        _HAS_TRITON
        and router_logits.is_cuda
        and router_logits.is_contiguous()
        and router_logits.dtype == torch.bfloat16
        and not torch.is_grad_enabled()
        and int(rows) == 16
        and int(num_experts) == 128
        and int(top_k) == 8
        and workspace is not None
        and compact_workspace is not None
        and expert_scale is not None
        and expert_scale.is_cuda
        and expert_scale.is_contiguous()
        and expert_scale.device == router_logits.device
        and expert_scale.dtype == router_logits.dtype
        and int(expert_scale.numel()) == int(num_experts)
    )
    if not eligible:
        raise RuntimeError("fused B16 top-k compact route pack is not eligible")

    weights = _workspace_tensor(
        workspace,
        "router_weights",
        (int(rows), int(top_k)),
        device=router_logits.device,
        dtype=router_logits.dtype,
    )
    experts = _workspace_tensor(
        workspace,
        "router_experts",
        (int(rows), int(top_k)),
        device=router_logits.device,
        dtype=torch.int64,
    )
    counts = _workspace_tensor(
        compact_workspace,
        "expert_grouped_compact_counts",
        (int(num_experts),),
        device=router_logits.device,
        dtype=torch.int32,
    )
    dense_tokens = _workspace_tensor(
        compact_workspace,
        "expert_grouped_compact_tokens",
        (int(num_experts), int(rows)),
        device=router_logits.device,
        dtype=torch.int64,
    )
    dense_route = _workspace_tensor(
        compact_workspace,
        "expert_grouped_compact_route",
        (int(num_experts), int(rows)),
        device=router_logits.device,
        dtype=router_logits.dtype,
    )
    dense_assign = _workspace_tensor(
        compact_workspace,
        "expert_grouped_compact_assign",
        (int(num_experts), int(rows)),
        device=router_logits.device,
        dtype=torch.int64,
    )
    block_e = triton.next_power_of_2(int(num_experts))
    _qwen3_moe_topk_softmax_compact_pack_kernel[(1,)](
        router_logits,
        weights,
        experts,
        expert_scale,
        counts,
        dense_tokens,
        dense_route,
        dense_assign,
        router_logits.stride(0),
        router_logits.stride(1),
        weights.stride(0),
        weights.stride(1),
        experts.stride(0),
        experts.stride(1),
        dense_tokens.stride(0),
        dense_tokens.stride(1),
        dense_route.stride(0),
        dense_route.stride(1),
        dense_assign.stride(0),
        dense_assign.stride(1),
        int(num_experts),
        ROWS=int(rows),
        TOP_K=int(top_k),
        BLOCK_E=block_e,
        num_warps=4,
        num_stages=2,
    )
    compact_workspace["expert_grouped_compact_route_prepacked_rows"] = int(rows)
    compact_workspace["expert_grouped_compact_route_prepacked_assignments"] = int(
        rows * top_k
    )
    compact_workspace["expert_grouped_compact_route_prepacked_experts"] = int(
        num_experts
    )
    return weights, experts


def qwen3_moe_router_topk_softmax(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
    *,
    workspace: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused router projection + normalized top-k for decode/small-M Qwen3 MoE.

    Equivalent to:
      logits = hidden_states @ router_weight.T
      top_logits, selected = torch.topk(logits, top_k)
      weights = softmax(top_logits)

    This avoids materializing the full [rows, experts] logits tensor and cuts
    the separate top-k launch in the decode path. It is intentionally narrow:
    if the shape/device is not a good fit, it falls back to the unfused path.
    """
    if hidden_states.ndim != 2 or router_weight.ndim != 2:
        raise ValueError("hidden_states and router_weight must be 2D tensors")
    rows, hidden_dim = hidden_states.shape
    num_experts, weight_hidden = router_weight.shape
    if int(weight_hidden) != int(hidden_dim):
        raise ValueError("router_weight hidden dimension must match hidden_states")
    top_k = min(int(top_k), int(num_experts))
    if rows == 0 or top_k <= 0:
        weights = hidden_states.new_empty((rows, top_k))
        experts = torch.empty((rows, top_k), device=hidden_states.device, dtype=torch.int64)
        return weights, experts

    use_triton = (
        _CFG_FUSED_ROUTER
        and _HAS_TRITON
        and hidden_states.is_cuda
        and router_weight.is_cuda
        and hidden_states.is_contiguous()
        and router_weight.is_contiguous()
        and not torch.is_grad_enabled()
        and rows <= _CFG_FUSED_ROUTER_MAX_ROWS
        and hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and router_weight.dtype == hidden_states.dtype
        and top_k <= 16
        and _is_power_of_2(top_k)
        and num_experts <= 256
    )
    if not use_triton:
        logits = torch.nn.functional.linear(hidden_states, router_weight)
        return qwen3_moe_topk_softmax(logits, top_k, workspace=workspace)

    weights = _workspace_tensor(
        workspace,
        "router_weights",
        (rows, top_k),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    experts = _workspace_tensor(
        workspace,
        "router_experts",
        (rows, top_k),
        device=hidden_states.device,
        dtype=torch.int64,
    )
    block_e = triton.next_power_of_2(num_experts)
    block_k = min(_CFG_ROUTER_BLOCK_K, triton.next_power_of_2(hidden_dim))
    router_k_splits = int(_CFG_ROUTER_K_SPLITS) if rows <= _CFG_FUSED_ROUTER_MAX_ROWS else 1
    if router_k_splits > 1:
        split_k = triton.cdiv(hidden_dim, router_k_splits)
        partials = _workspace_tensor(
            workspace,
            "router_k_partials",
            (rows, router_k_splits, num_experts),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        _qwen3_moe_router_k_split_kernel[(rows, router_k_splits)](
            hidden_states,
            router_weight,
            partials,
            hidden_states.stride(0),
            hidden_states.stride(1),
            router_weight.stride(0),
            router_weight.stride(1),
            partials.stride(0),
            partials.stride(1),
            partials.stride(2),
            hidden_dim,
            num_experts,
            SPLIT_K=split_k,
            BLOCK_E=block_e,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=2,
        )
        _qwen3_moe_router_k_reduce_topk_softmax_kernel[(rows,)](
            partials,
            weights,
            experts,
            partials.stride(0),
            partials.stride(1),
            partials.stride(2),
            weights.stride(0),
            weights.stride(1),
            experts.stride(0),
            experts.stride(1),
            num_experts,
            TOP_K=top_k,
            K_SPLITS=router_k_splits,
            BLOCK_E=block_e,
            num_warps=4,
            num_stages=1,
        )
    else:
        _qwen3_moe_router_topk_softmax_kernel[(rows,)](
            hidden_states,
            router_weight,
            weights,
            experts,
            hidden_states.stride(0),
            hidden_states.stride(1),
            router_weight.stride(0),
            router_weight.stride(1),
            weights.stride(0),
            weights.stride(1),
            experts.stride(0),
            experts.stride(1),
            hidden_dim,
            num_experts,
            TOP_K=top_k,
            BLOCK_E=block_e,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=2,
        )
    if workspace is not None:
        workspace["router_last_k_splits"] = int(router_k_splits)
    return weights, experts


def qwen3_moe_grouped_decode_int8(
    hidden_states: torch.Tensor,
    gate_up_int8: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_int8: torch.Tensor,
    down_scale: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    activation: str = "silu",
    out: Optional[torch.Tensor] = None,
    workspace: Optional[dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Grouped Qwen3 MoE decode over W8A16 expert weights."""
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be [M, H]")
    if selected_experts.shape != routing_weights.shape:
        raise ValueError("selected_experts and routing_weights must have the same shape")
    if selected_experts.ndim != 2:
        raise ValueError("selected_experts must be [M, top_k]")

    rows, hidden_dim = hidden_states.shape
    if rows == 0:
        return torch.zeros_like(hidden_states) if out is None else out.zero_()
    top_k = int(selected_experts.shape[1])
    intermediate_dim = int(gate_up_int8.shape[1] // 2)
    act_id = _activation_id(activation)

    use_triton = (
        _CFG_INT8_DECODE
        and _HAS_TRITON
        and hidden_states.is_cuda
        and gate_up_int8.is_cuda
        and gate_up_scale.is_cuda
        and down_int8.is_cuda
        and down_scale.is_cuda
        and selected_experts.is_cuda
        and routing_weights.is_cuda
        and not torch.is_grad_enabled()
        and qwen3_moe_grouped_prefers_triton_shape(rows, top_k, hidden_dim, intermediate_dim)
    )
    if not use_triton:
        return _fallback_grouped_moe_int8(
            hidden_states,
            gate_up_int8,
            gate_up_scale,
            down_int8,
            down_scale,
            selected_experts,
            routing_weights,
            activation,
            out,
        )

    hidden = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    gate_up_w = gate_up_int8 if gate_up_int8.is_contiguous() else gate_up_int8.contiguous()
    gate_up_s = gate_up_scale if gate_up_scale.is_contiguous() else gate_up_scale.contiguous()
    down_w = down_int8 if down_int8.is_contiguous() else down_int8.contiguous()
    down_s = down_scale if down_scale.is_contiguous() else down_scale.contiguous()
    experts = selected_experts.reshape(-1).to(torch.int64).contiguous()
    route = routing_weights.reshape(-1).contiguous()
    token_ids = _workspace_token_ids(workspace, rows, top_k, device=hidden.device)
    assignments = int(experts.numel())
    if workspace is not None:
        workspace["grouped_decode_last_path"] = "assignment"
        workspace["grouped_decode_last_rows"] = int(rows)
        workspace["grouped_decode_last_assignments"] = int(assignments)

    block_k = min(_CFG_BLOCK_K, triton.next_power_of_2(hidden_dim))
    block_n_down = min(_CFG_BLOCK_N, triton.next_power_of_2(hidden_dim))
    block_k_down = min(_CFG_BLOCK_K, triton.next_power_of_2(intermediate_dim))
    use_token_accum = bool(_CFG_TOKEN_ACCUM and rows >= _CFG_TOKEN_ACCUM_MIN_ROWS)
    use_grouped_fused_gate = bool(_CFG_GROUPED_FUSED_GATE and use_token_accum)

    if use_grouped_fused_gate:
        activated = _workspace_tensor(
            workspace,
            "activated",
            (assignments, intermediate_dim),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        block_n_act = min(_CFG_BLOCK_N, triton.next_power_of_2(intermediate_dim))
        _qwen3_moe_gate_swiglu_int8_kernel[
            (assignments, triton.cdiv(intermediate_dim, block_n_act))
        ](
            hidden,
            gate_up_w,
            gate_up_s,
            experts,
            token_ids,
            activated,
            hidden.stride(0),
            hidden.stride(1),
            gate_up_w.stride(0),
            gate_up_w.stride(1),
            gate_up_w.stride(2),
            gate_up_s.stride(0),
            gate_up_s.stride(1),
            activated.stride(0),
            activated.stride(1),
            hidden_dim,
            intermediate_dim,
            ACT=act_id,
            BLOCK_N=block_n_act,
            BLOCK_K=block_k,
            num_warps=_CFG_NUM_WARPS,
            num_stages=_CFG_NUM_STAGES,
        )
        final = out if out is not None else torch.empty_like(hidden_states)
        _qwen3_moe_down_from_act_token_accum_int8_kernel[
            (rows, triton.cdiv(hidden_dim, block_n_down))
        ](
            activated,
            down_w,
            down_s,
            experts,
            route,
            final,
            activated.stride(0),
            activated.stride(1),
            down_w.stride(0),
            down_w.stride(1),
            down_w.stride(2),
            down_s.stride(0),
            down_s.stride(1),
            final.stride(0),
            final.stride(1),
            intermediate_dim,
            hidden_dim,
            TOP_K=top_k,
            BLOCK_N=block_n_down,
            BLOCK_K=block_k_down,
            num_warps=_CFG_NUM_WARPS,
            num_stages=_CFG_NUM_STAGES,
        )
        return final

    gate_up = _workspace_tensor(
        workspace,
        "gate_up",
        (assignments, 2 * intermediate_dim),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    block_n = min(_CFG_BLOCK_N, triton.next_power_of_2(2 * intermediate_dim))
    _qwen3_moe_gate_up_int8_kernel[
        (assignments, triton.cdiv(2 * intermediate_dim, block_n))
    ](
        hidden,
        gate_up_w,
        gate_up_s,
        experts,
        token_ids,
        gate_up,
        hidden.stride(0),
        hidden.stride(1),
        gate_up_w.stride(0),
        gate_up_w.stride(1),
        gate_up_w.stride(2),
        gate_up_s.stride(0),
        gate_up_s.stride(1),
        gate_up.stride(0),
        gate_up.stride(1),
        hidden_dim,
        2 * intermediate_dim,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=_CFG_NUM_WARPS,
        num_stages=_CFG_NUM_STAGES,
    )

    if use_token_accum:
        final = out if out is not None else torch.empty_like(hidden_states)
        _qwen3_moe_swiglu_down_token_accum_int8_kernel[
            (rows, triton.cdiv(hidden_dim, block_n_down))
        ](
            gate_up,
            down_w,
            down_s,
            experts,
            route,
            final,
            gate_up.stride(0),
            gate_up.stride(1),
            down_w.stride(0),
            down_w.stride(1),
            down_w.stride(2),
            down_s.stride(0),
            down_s.stride(1),
            final.stride(0),
            final.stride(1),
            intermediate_dim,
            hidden_dim,
            TOP_K=top_k,
            ACT=act_id,
            BLOCK_N=block_n_down,
            BLOCK_K=block_k_down,
            num_warps=_CFG_NUM_WARPS,
            num_stages=_CFG_NUM_STAGES,
        )
        return final

    accum = _workspace_tensor(
        workspace,
        "accum",
        (rows, hidden_dim),
        device=hidden.device,
        dtype=torch.float32,
        zero=True,
    )
    _qwen3_moe_swiglu_down_accum_int8_kernel[
        (assignments, triton.cdiv(hidden_dim, block_n_down))
    ](
        gate_up,
        down_w,
        down_s,
        experts,
        token_ids,
        route,
        accum,
        gate_up.stride(0),
        gate_up.stride(1),
        down_w.stride(0),
        down_w.stride(1),
        down_w.stride(2),
        down_s.stride(0),
        down_s.stride(1),
        accum.stride(0),
        accum.stride(1),
        intermediate_dim,
        hidden_dim,
        ACT=act_id,
        BLOCK_N=block_n_down,
        BLOCK_K=block_k_down,
        num_warps=_CFG_NUM_WARPS,
        num_stages=_CFG_NUM_STAGES,
    )
    if out is not None:
        out.copy_(accum)
        return out
    return accum.to(hidden_states.dtype)


def _qwen3_moe_expert_grouped_decode(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    activation: str,
    out: Optional[torch.Tensor],
    workspace: Optional[dict[str, torch.Tensor]],
) -> torch.Tensor:
    rows, hidden_dim = hidden_states.shape
    top_k = int(selected_experts.shape[1])
    assignments = int(rows) * int(top_k)
    num_experts = int(gate_up_proj.shape[0])
    intermediate_dim = int(gate_up_proj.shape[1] // 2)
    act_id = _activation_id(activation)

    hidden = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    gate_up_w = gate_up_proj if gate_up_proj.is_contiguous() else gate_up_proj.contiguous()
    down_w = down_proj if down_proj.is_contiguous() else down_proj.contiguous()
    experts = selected_experts.reshape(-1).to(torch.int64).contiguous()
    route = routing_weights.reshape(-1).contiguous()

    counts = _workspace_tensor(
        workspace,
        "expert_grouped_counts",
        (num_experts,),
        device=hidden.device,
        dtype=torch.int32,
        zero=True,
    )
    dense_tokens = _workspace_tensor(
        workspace,
        "expert_grouped_tokens",
        (num_experts, int(rows)),
        device=hidden.device,
        dtype=torch.int64,
    )
    dense_route = _workspace_tensor(
        workspace,
        "expert_grouped_route",
        (num_experts, int(rows)),
        device=hidden.device,
        dtype=route.dtype,
    )

    route_block = int(_CFG_EXPERT_GROUPED_ROUTE_BLOCK)
    _qwen3_moe_expert_grouped_scatter_kernel[
        (triton.cdiv(assignments, route_block),)
    ](
        experts,
        route,
        counts,
        dense_tokens,
        dense_route,
        int(assignments),
        ROWS=int(rows),
        TOP_K=int(top_k),
        BLOCK=route_block,
    )

    block_m = min(
        int(_CFG_EXPERT_GROUPED_BLOCK_M),
        16,
        int(triton.next_power_of_2(max(int(rows), 1))),
    )
    block_n_act = min(_CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N, triton.next_power_of_2(intermediate_dim))
    block_k_hidden = min(_CFG_BLOCK_K, triton.next_power_of_2(hidden_dim))
    block_n_down = min(_CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N, triton.next_power_of_2(hidden_dim))
    block_k_down = min(_CFG_BLOCK_K, triton.next_power_of_2(intermediate_dim))

    activated = _workspace_tensor(
        workspace,
        "expert_grouped_activated",
        (num_experts, int(rows), intermediate_dim),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    _qwen3_moe_expert_grouped_gate_swiglu_kernel[
        (
            num_experts,
            triton.cdiv(int(rows), block_m),
            triton.cdiv(intermediate_dim, block_n_act),
        )
    ](
        hidden,
        gate_up_w,
        counts,
        dense_tokens,
        activated,
        hidden.stride(0),
        hidden.stride(1),
        gate_up_w.stride(0),
        gate_up_w.stride(1),
        gate_up_w.stride(2),
        activated.stride(0),
        activated.stride(1),
        activated.stride(2),
        hidden_dim,
        intermediate_dim,
        ROWS=int(rows),
        ACT=act_id,
        BLOCK_M=block_m,
        BLOCK_N=block_n_act,
        BLOCK_K=block_k_hidden,
        num_warps=_CFG_NUM_WARPS,
        num_stages=_CFG_NUM_STAGES,
    )

    accum = _workspace_tensor(
        workspace,
        "expert_grouped_accum",
        (int(rows), hidden_dim),
        device=hidden.device,
        dtype=torch.float32,
        zero=True,
    )
    _qwen3_moe_expert_grouped_down_accum_kernel[
        (
            num_experts,
            triton.cdiv(int(rows), block_m),
            triton.cdiv(hidden_dim, block_n_down),
        )
    ](
        activated,
        down_w,
        counts,
        dense_tokens,
        dense_route,
        accum,
        activated.stride(0),
        activated.stride(1),
        activated.stride(2),
        down_w.stride(0),
        down_w.stride(1),
        down_w.stride(2),
        accum.stride(0),
        accum.stride(1),
        intermediate_dim,
        hidden_dim,
        ROWS=int(rows),
        BLOCK_M=block_m,
        BLOCK_N=block_n_down,
        BLOCK_K=block_k_down,
        num_warps=_CFG_NUM_WARPS,
        num_stages=_CFG_NUM_STAGES,
    )

    if workspace is not None:
        workspace["expert_grouped_decode_last_rows"] = int(rows)
        workspace["expert_grouped_decode_last_assignments"] = int(assignments)
        workspace["expert_grouped_decode_last_num_experts"] = int(num_experts)
        workspace["expert_grouped_decode_last_block_m"] = int(block_m)

    final = out if out is not None else torch.empty_like(hidden_states)
    final.copy_(accum)
    return final


def _qwen3_moe_expert_grouped_compact_decode(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    activation: str,
    out: Optional[torch.Tensor],
    workspace: Optional[dict[str, torch.Tensor]],
    partial_reduce: Optional[bool] = None,
    route_prepacked: bool = False,
    post_moe_shared: Optional[torch.Tensor] = None,
    post_moe_shared_weight: Optional[torch.Tensor] = None,
    post_moe_expert_weight: Optional[torch.Tensor] = None,
    post_moe_final_weight: Optional[torch.Tensor] = None,
    post_moe_residual: Optional[torch.Tensor] = None,
    post_moe_wait_event: Optional[object] = None,
    post_moe_layer_scalar: Optional[torch.Tensor] = None,
    post_moe_next_norm_weight: Optional[torch.Tensor] = None,
    post_moe_next_norm_out: Optional[torch.Tensor] = None,
    post_moe_write_next_norm: bool = False,
    post_moe_eps: float = 1e-6,
) -> torch.Tensor:
    rows, hidden_dim = hidden_states.shape
    top_k = int(selected_experts.shape[1])
    assignments = int(rows) * int(top_k)
    num_experts = int(gate_up_proj.shape[0])
    intermediate_dim = int(gate_up_proj.shape[1] // 2)
    act_id = _activation_id(activation)

    hidden = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    gate_up_w = gate_up_proj if gate_up_proj.is_contiguous() else gate_up_proj.contiguous()
    down_w = down_proj if down_proj.is_contiguous() else down_proj.contiguous()
    experts = selected_experts.reshape(-1).to(torch.int64).contiguous()
    route = routing_weights.reshape(-1).contiguous()
    use_active_list = bool(_CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST)
    selected_partial_reduce = (
        bool(_CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE)
        if partial_reduce is None
        else bool(partial_reduce)
    )
    use_compact_token_accum = bool(
        _CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM and not selected_partial_reduce
    )
    use_fused_pack = bool(
        (_CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK or selected_partial_reduce)
        and not use_compact_token_accum
        and assignments <= 128
    )
    use_partial_reduce = bool(selected_partial_reduce and use_fused_pack)
    post_moe_values = (
        post_moe_shared,
        post_moe_shared_weight,
        post_moe_expert_weight,
        post_moe_final_weight,
        post_moe_residual,
    )
    use_fused_post_moe = all(value is not None for value in post_moe_values)
    if any(value is not None for value in post_moe_values) and not use_fused_post_moe:
        raise ValueError("fused Gemma4 post-MoE inputs must be provided together")
    next_norm_values = (
        post_moe_layer_scalar,
        post_moe_next_norm_weight,
        post_moe_next_norm_out,
    )
    fuse_layer_scalar = all(value is not None for value in next_norm_values)
    if any(value is not None for value in next_norm_values) and not fuse_layer_scalar:
        raise ValueError(
            "fused Gemma4 layer-scalar/next-norm inputs must be provided together"
        )
    if fuse_layer_scalar and not use_fused_post_moe:
        raise ValueError(
            "fused Gemma4 layer-scalar/next-norm requires fused post-MoE"
        )
    if post_moe_write_next_norm and not fuse_layer_scalar:
        raise ValueError("next-attention RMSNorm output requires layer-scalar fusion")
    if use_fused_post_moe:
        if not use_partial_reduce:
            raise RuntimeError("fused Gemma4 post-MoE requires compact partial reduction")
        if out is None:
            raise ValueError("fused Gemma4 post-MoE requires a persistent output buffer")
        if (
            int(rows) != 16
            or int(hidden_dim) != 2816
            or int(intermediate_dim) != 704
            or int(num_experts) != 128
            or int(top_k) != 8
            or hidden_states.dtype != torch.bfloat16
        ):
            raise RuntimeError("fused Gemma4 post-MoE is restricted to A4B BF16 B16")
        for name, tensor in (
            ("shared", post_moe_shared),
            ("residual", post_moe_residual),
            ("out", out),
        ):
            if tuple(tensor.shape) != tuple(hidden_states.shape):
                raise ValueError(f"fused Gemma4 post-MoE {name} shape mismatch")
            if tensor.device != hidden_states.device or tensor.dtype != hidden_states.dtype:
                raise ValueError(f"fused Gemma4 post-MoE {name} device/dtype mismatch")
            if tensor.stride(-1) != 1:
                raise ValueError(f"fused Gemma4 post-MoE {name} must be contiguous")
        if out.data_ptr() in {
            post_moe_shared.data_ptr(),
            post_moe_residual.data_ptr(),
            hidden_states.data_ptr(),
        }:
            raise ValueError("fused Gemma4 post-MoE output must not alias its inputs")
        for name, tensor in (
            ("shared_weight", post_moe_shared_weight),
            ("expert_weight", post_moe_expert_weight),
            ("final_weight", post_moe_final_weight),
        ):
            if int(tensor.numel()) != int(hidden_dim):
                raise ValueError(f"fused Gemma4 post-MoE {name} size mismatch")
            if (
                tensor.device != hidden_states.device
                or tensor.dtype != hidden_states.dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"fused Gemma4 post-MoE {name} layout mismatch")
        if fuse_layer_scalar:
            if (
                int(post_moe_layer_scalar.numel()) != 1
                or post_moe_layer_scalar.device != hidden_states.device
                or post_moe_layer_scalar.dtype != hidden_states.dtype
            ):
                raise ValueError("fused Gemma4 layer scalar must be one BF16 GPU value")
            if (
                int(post_moe_next_norm_weight.numel()) != int(hidden_dim)
                or post_moe_next_norm_weight.device != hidden_states.device
                or post_moe_next_norm_weight.dtype != hidden_states.dtype
                or not post_moe_next_norm_weight.is_contiguous()
            ):
                raise ValueError("fused Gemma4 next RMSNorm weight layout mismatch")
            if (
                tuple(post_moe_next_norm_out.shape) != tuple(hidden_states.shape)
                or post_moe_next_norm_out.device != hidden_states.device
                or post_moe_next_norm_out.dtype != hidden_states.dtype
                or post_moe_next_norm_out.stride(-1) != 1
            ):
                raise ValueError("fused Gemma4 next RMSNorm output layout mismatch")
            if post_moe_next_norm_out.data_ptr() == out.data_ptr():
                raise ValueError(
                    "fused Gemma4 next RMSNorm output must not alias hidden output"
                )
    use_expert_grid_pack = bool(
        _CFG_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK
        and use_partial_reduce
        and assignments == num_experts
        and assignments <= 128
    )
    if route_prepacked:
        if workspace is None:
            raise ValueError("prepacked compact route requires a persistent workspace")
        if not use_expert_grid_pack:
            raise RuntimeError(
                "prepacked compact route requires deterministic expert-grid decode"
            )
        expected_pack = (
            int(rows),
            int(assignments),
            int(num_experts),
        )
        actual_pack = (
            int(workspace.get("expert_grouped_compact_route_prepacked_rows", 0) or 0),
            int(
                workspace.get(
                    "expert_grouped_compact_route_prepacked_assignments",
                    0,
                )
                or 0
            ),
            int(
                workspace.get(
                    "expert_grouped_compact_route_prepacked_experts",
                    0,
                )
                or 0
            ),
        )
        if actual_pack != expected_pack:
            raise RuntimeError(
                f"prepacked compact route metadata mismatch: {actual_pack} != {expected_pack}"
            )
    process_expert_grid = bool(use_expert_grid_pack and not use_active_list)

    counts = _workspace_tensor(
        workspace,
        "expert_grouped_compact_counts",
        (num_experts,),
        device=hidden.device,
        dtype=torch.int32,
        zero=not use_fused_pack,
    )
    dense_tokens = _workspace_tensor(
        workspace,
        "expert_grouped_compact_tokens",
        (num_experts, int(rows)),
        device=hidden.device,
        dtype=torch.int64,
    )
    dense_route = _workspace_tensor(
        workspace,
        "expert_grouped_compact_route",
        (num_experts, int(rows)),
        device=hidden.device,
        dtype=route.dtype,
    )
    if use_partial_reduce:
        dense_assign = _workspace_tensor(
            workspace,
            "expert_grouped_compact_assign",
            (num_experts, int(rows)),
            device=hidden.device,
            dtype=torch.int64,
        )
    else:
        dense_assign = dense_tokens
    if use_active_list:
        unique_flags = _workspace_tensor(
            workspace,
            "expert_grouped_compact_unique",
            (assignments,),
            device=hidden.device,
            dtype=torch.int8,
        )
        active_experts = _workspace_tensor(
            workspace,
            "expert_grouped_compact_active_experts",
            (assignments,),
            device=hidden.device,
            dtype=torch.int64,
        )
        active_count = _workspace_tensor(
            workspace,
            "expert_grouped_compact_active_count",
            (1,),
            device=hidden.device,
            dtype=torch.int32,
        )
        if use_compact_token_accum:
            expert_to_candidate = _workspace_tensor(
                workspace,
                "expert_grouped_compact_expert_to_candidate",
                (num_experts,),
                device=hidden.device,
                dtype=torch.int64,
            )
        else:
            expert_to_candidate = active_experts
        grid_experts = active_experts
        unique_or_count = active_count
    else:
        unique_flags = _workspace_tensor(
            workspace,
            "expert_grouped_compact_unique",
            (assignments,),
            device=hidden.device,
            dtype=torch.int8,
        )
        if use_compact_token_accum:
            expert_to_candidate = _workspace_tensor(
                workspace,
                "expert_grouped_compact_expert_to_candidate",
                (num_experts,),
                device=hidden.device,
                dtype=torch.int64,
            )
        else:
            expert_to_candidate = unique_flags
        active_experts = unique_flags
        active_count = unique_flags
        grid_experts = experts
        unique_or_count = unique_flags

    route_block = int(_CFG_EXPERT_GROUPED_ROUTE_BLOCK)
    if route_prepacked:
        if workspace is not None:
            workspace["expert_grouped_compact_route_prepacked_hits"] = int(
                workspace.get("expert_grouped_compact_route_prepacked_hits", 0) or 0
            ) + 1
    elif use_expert_grid_pack:
        pack_block = min(128, triton.next_power_of_2(max(int(assignments), 1)))
        _qwen3_moe_expert_grouped_compact_expert_pack_kernel[(num_experts,)](
            experts,
            route,
            counts,
            dense_tokens,
            dense_route,
            dense_assign,
            int(assignments),
            ROWS=int(rows),
            TOP_K=int(top_k),
            BLOCK=pack_block,
            num_warps=4,
            num_stages=1,
        )
        if use_active_list:
            active_block = triton.next_power_of_2(max(int(num_experts), 1))
            _qwen3_moe_expert_grouped_compact_active_blocks_kernel[(1,)](
                counts,
                active_experts,
                active_count,
                NUM_EXPERTS=int(num_experts),
                BLOCK=active_block,
                num_warps=4,
                num_stages=1,
            )
    elif use_fused_pack:
        pack_block = min(128, triton.next_power_of_2(max(int(assignments), 1)))
        _qwen3_moe_expert_grouped_compact_pack_kernel[(1,)](
            experts,
            route,
            counts,
            dense_tokens,
            dense_route,
            dense_assign,
            unique_flags,
            active_experts,
            active_count,
            expert_to_candidate,
            int(assignments),
            ROWS=int(rows),
            TOP_K=int(top_k),
            STORE_ASSIGN=use_partial_reduce,
            STORE_ACTIVE=use_active_list,
            BUILD_MAP=use_compact_token_accum,
            BLOCK=pack_block,
        )
    elif use_active_list:
        _qwen3_moe_expert_grouped_active_scatter_kernel[
            (triton.cdiv(assignments, route_block),)
        ](
            experts,
            route,
            counts,
            dense_tokens,
            dense_route,
            active_experts,
            active_count,
            expert_to_candidate,
            int(assignments),
            ROWS=int(rows),
            TOP_K=int(top_k),
            BUILD_MAP=use_compact_token_accum,
            BLOCK=route_block,
        )
    else:
        _qwen3_moe_expert_grouped_scatter_kernel[
            (triton.cdiv(assignments, route_block),)
        ](
            experts,
            route,
            counts,
            dense_tokens,
            dense_route,
            int(assignments),
            ROWS=int(rows),
            TOP_K=int(top_k),
            BLOCK=route_block,
        )
        unique_block = min(128, triton.next_power_of_2(max(int(assignments), 1)))
        _qwen3_moe_expert_grouped_unique_kernel[
            (triton.cdiv(assignments, unique_block),)
        ](
            experts,
            unique_flags,
            expert_to_candidate,
            int(assignments),
            WRITE_MAP=use_compact_token_accum,
            BLOCK=unique_block,
        )

    block_m = min(
        int(_CFG_EXPERT_GROUPED_BLOCK_M),
        16,
        int(triton.next_power_of_2(max(int(rows), 1))),
    )
    block_n_act = min(
        _CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N,
        triton.next_power_of_2(intermediate_dim),
    )
    block_k_hidden = min(_CFG_BLOCK_K, triton.next_power_of_2(hidden_dim))
    block_n_down = min(
        _CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N,
        triton.next_power_of_2(hidden_dim),
    )
    block_k_down = min(_CFG_BLOCK_K, triton.next_power_of_2(intermediate_dim))

    candidate_slots = int(num_experts if use_expert_grid_pack else assignments)
    experts_per_program = int(
        _CFG_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM
        if use_expert_grid_pack
        else 1
    )
    if experts_per_program not in (1, 2, 4):
        experts_per_program = 1
    experts_per_program = min(experts_per_program, candidate_slots)
    candidate_groups = int(triton.cdiv(candidate_slots, experts_per_program))
    use_active_list_early_exit = bool(
        _CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT
        and use_active_list
        and experts_per_program == 1
    )
    use_split_gate_up = bool(_CFG_EXPERT_GROUPED_COMPACT_SPLIT_GATE_UP)
    use_l2_grouped_grid = bool(
        _CFG_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID
        and (use_expert_grid_pack or use_active_list)
        and experts_per_program == 1
        and int(rows) <= block_m
        and not use_split_gate_up
    )
    l2_group_size = min(
        int(_CFG_EXPERT_GROUPED_COMPACT_L2_GROUP_SIZE),
        candidate_groups,
    )
    activated = _workspace_tensor(
        workspace,
        "expert_grouped_compact_activated",
        (candidate_slots, int(rows), intermediate_dim),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    if use_split_gate_up:
        gate_up_tmp = _workspace_tensor(
            workspace,
            "expert_grouped_compact_gate_up",
            (candidate_slots, int(rows), 2 * intermediate_dim),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        _qwen3_moe_expert_grouped_compact_gate_up_split_kernel[
            (
                candidate_groups,
                triton.cdiv(int(rows), block_m),
                triton.cdiv(2 * intermediate_dim, block_n_act),
            )
        ](
            hidden,
            gate_up_w,
            grid_experts,
            unique_or_count,
            counts,
            dense_tokens,
            gate_up_tmp,
            hidden.stride(0),
            hidden.stride(1),
            gate_up_w.stride(0),
            gate_up_w.stride(1),
            gate_up_w.stride(2),
            gate_up_tmp.stride(0),
            gate_up_tmp.stride(1),
            gate_up_tmp.stride(2),
            hidden_dim,
            intermediate_dim,
            ROWS=int(rows),
            ASSIGNMENTS=int(assignments),
            CANDIDATE_SLOTS=candidate_slots,
            ACTIVE_LIST=use_active_list,
            EXPERT_GRID=process_expert_grid,
            COALESCED_WEIGHTS=bool(
                _CFG_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS
            ),
            BLOCK_M=block_m,
            BLOCK_N=block_n_act,
            BLOCK_K=block_k_hidden,
            EXPERTS_PER_PROGRAM=experts_per_program,
            EMPTY_EXPERT_EARLY_EXIT=bool(
                _CFG_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT
            ),
            ACTIVE_LIST_EARLY_EXIT=use_active_list_early_exit,
            num_warps=_CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS,
            num_stages=_CFG_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES,
        )
        _qwen3_moe_expert_grouped_compact_swiglu_kernel[
            (
                candidate_groups,
                triton.cdiv(int(rows), block_m),
                triton.cdiv(intermediate_dim, block_n_act),
            )
        ](
            grid_experts,
            unique_or_count,
            counts,
            gate_up_tmp,
            activated,
            gate_up_tmp.stride(0),
            gate_up_tmp.stride(1),
            gate_up_tmp.stride(2),
            activated.stride(0),
            activated.stride(1),
            activated.stride(2),
            intermediate_dim,
            CANDIDATE_SLOTS=candidate_slots,
            ACTIVE_LIST=use_active_list,
            EXPERT_GRID=process_expert_grid,
            ACT=act_id,
            BLOCK_M=block_m,
            BLOCK_N=block_n_act,
            EXPERTS_PER_PROGRAM=experts_per_program,
            ACTIVE_LIST_EARLY_EXIT=use_active_list_early_exit,
            num_warps=_CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS,
            num_stages=1,
        )
    else:
        gate_num_pid_n = int(triton.cdiv(intermediate_dim, block_n_act))
        gate_grid = (
            (candidate_groups * gate_num_pid_n,)
            if use_l2_grouped_grid
            else (
                candidate_groups,
                triton.cdiv(int(rows), block_m),
                gate_num_pid_n,
            )
        )
        _qwen3_moe_expert_grouped_compact_gate_swiglu_kernel[
            gate_grid
        ](
            hidden,
            gate_up_w,
            grid_experts,
            unique_or_count,
            counts,
            dense_tokens,
            activated,
            hidden.stride(0),
            hidden.stride(1),
            gate_up_w.stride(0),
            gate_up_w.stride(1),
            gate_up_w.stride(2),
            activated.stride(0),
            activated.stride(1),
            activated.stride(2),
            hidden_dim,
            intermediate_dim,
            ROWS=int(rows),
            ASSIGNMENTS=int(assignments),
            CANDIDATE_SLOTS=candidate_slots,
            ACTIVE_LIST=use_active_list,
            EXPERT_GRID=process_expert_grid,
            COALESCED_WEIGHTS=bool(
                _CFG_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS
            ),
            ACT=act_id,
            BLOCK_M=block_m,
            BLOCK_N=block_n_act,
            BLOCK_K=block_k_hidden,
            EXPERTS_PER_PROGRAM=experts_per_program,
            PAIRED_GATE_UP_DOT=bool(
                _CFG_EXPERT_GROUPED_COMPACT_PAIRED_GATE_UP_DOT
            ),
            EMPTY_EXPERT_EARLY_EXIT=bool(
                _CFG_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT
            ),
            ACTIVE_LIST_EARLY_EXIT=use_active_list_early_exit,
            L2_GROUPED_GRID=use_l2_grouped_grid,
            NUM_CANDIDATE_GROUPS=candidate_groups,
            NUM_PID_N=gate_num_pid_n,
            L2_GROUP_SIZE=l2_group_size,
            num_warps=_CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS,
            num_stages=_CFG_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES,
        )

    if use_compact_token_accum:
        selected = selected_experts if selected_experts.is_contiguous() else selected_experts.contiguous()
        route_matrix = routing_weights if routing_weights.is_contiguous() else routing_weights.contiguous()
        final = out if out is not None else torch.empty_like(hidden_states)
        _qwen3_moe_expert_grouped_compact_down_token_accum_kernel[
            (
                int(rows),
                triton.cdiv(hidden_dim, block_n_down),
            )
        ](
            activated,
            down_w,
            selected,
            route_matrix,
            expert_to_candidate,
            final,
            activated.stride(0),
            activated.stride(1),
            activated.stride(2),
            down_w.stride(0),
            down_w.stride(1),
            down_w.stride(2),
            selected.stride(0),
            selected.stride(1),
            route_matrix.stride(0),
            route_matrix.stride(1),
            final.stride(0),
            final.stride(1),
            intermediate_dim,
            hidden_dim,
            TOP_K=top_k,
            BLOCK_N=block_n_down,
            BLOCK_K=block_k_down,
            num_warps=_CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS,
            num_stages=_CFG_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES,
        )
        if workspace is not None:
            workspace["expert_grouped_compact_decode_last_rows"] = int(rows)
            workspace["expert_grouped_compact_decode_last_assignments"] = int(assignments)
            workspace["expert_grouped_compact_decode_last_num_experts"] = int(num_experts)
            workspace["expert_grouped_compact_decode_last_block_m"] = int(block_m)
            workspace["expert_grouped_compact_decode_last_token_accum"] = 1
            workspace["expert_grouped_compact_decode_last_active_list"] = int(use_active_list)
            workspace["expert_grouped_compact_decode_last_active_list_early_exit"] = int(
                use_active_list_early_exit
            )
            workspace["expert_grouped_compact_decode_last_fused_pack"] = int(use_fused_pack)
            workspace["expert_grouped_compact_decode_last_partial_reduce"] = 0
            workspace["expert_grouped_compact_decode_last_split_gate_up"] = int(
                use_split_gate_up
            )
            workspace["expert_grouped_compact_decode_last_empty_expert_early_exit"] = int(
                _CFG_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT
            )
            workspace["expert_grouped_compact_decode_last_l2_grouped_grid"] = int(
                use_l2_grouped_grid
            )
            workspace["expert_grouped_compact_decode_last_l2_group_size"] = int(
                l2_group_size
            )
        return final

    if use_partial_reduce:
        partial = _workspace_tensor(
            workspace,
            "expert_grouped_compact_partial",
            (int(assignments), hidden_dim),
            device=hidden.device,
            dtype=torch.float32,
        )
        down_num_pid_n = int(triton.cdiv(hidden_dim, block_n_down))
        down_grid = (
            (candidate_groups * down_num_pid_n,)
            if use_l2_grouped_grid
            else (
                candidate_groups,
                triton.cdiv(int(rows), block_m),
                down_num_pid_n,
            )
        )
        _qwen3_moe_expert_grouped_compact_down_partial_kernel[down_grid](
            activated,
            down_w,
            grid_experts,
            unique_or_count,
            counts,
            dense_assign,
            dense_route,
            partial,
            activated.stride(0),
            activated.stride(1),
            activated.stride(2),
            down_w.stride(0),
            down_w.stride(1),
            down_w.stride(2),
            partial.stride(0),
            partial.stride(1),
            intermediate_dim,
            hidden_dim,
            ROWS=int(rows),
            ASSIGNMENTS=int(assignments),
            CANDIDATE_SLOTS=candidate_slots,
            ACTIVE_LIST=use_active_list,
            EXPERT_GRID=process_expert_grid,
            COALESCED_WEIGHTS=bool(
                _CFG_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS
            ),
            BLOCK_M=block_m,
            BLOCK_N=block_n_down,
            BLOCK_K=block_k_down,
            EXPERTS_PER_PROGRAM=experts_per_program,
            EMPTY_EXPERT_EARLY_EXIT=bool(
                _CFG_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT
            ),
            ACTIVE_LIST_EARLY_EXIT=use_active_list_early_exit,
            L2_GROUPED_GRID=use_l2_grouped_grid,
            NUM_CANDIDATE_GROUPS=candidate_groups,
            NUM_PID_N=down_num_pid_n,
            L2_GROUP_SIZE=l2_group_size,
            num_warps=_CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS,
            num_stages=_CFG_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES,
        )
        final = out if out is not None else torch.empty_like(hidden_states)
        if use_fused_post_moe:
            if post_moe_wait_event is not None:
                torch.cuda.current_stream(hidden.device).wait_event(post_moe_wait_event)
            post_block = triton.next_power_of_2(hidden_dim)
            _qwen3_moe_assignment_reduce_gemma4_post_kernel[(int(rows),)](
                partial,
                post_moe_shared,
                post_moe_shared_weight,
                post_moe_expert_weight,
                post_moe_final_weight,
                post_moe_residual,
                final,
                post_moe_layer_scalar if fuse_layer_scalar else final,
                post_moe_next_norm_weight if fuse_layer_scalar else post_moe_final_weight,
                post_moe_next_norm_out if fuse_layer_scalar else final,
                partial.stride(0),
                partial.stride(1),
                post_moe_shared.stride(0),
                post_moe_shared.stride(1),
                post_moe_residual.stride(0),
                post_moe_residual.stride(1),
                final.stride(0),
                final.stride(1),
                (
                    post_moe_next_norm_out.stride(0)
                    if fuse_layer_scalar
                    else final.stride(0)
                ),
                (
                    post_moe_next_norm_out.stride(1)
                    if fuse_layer_scalar
                    else final.stride(1)
                ),
                H=hidden_dim,
                TOP_K=top_k,
                EPS=float(post_moe_eps),
                FUSE_LAYER_SCALAR=bool(fuse_layer_scalar),
                WRITE_NEXT_NORM=bool(post_moe_write_next_norm),
                BLOCK_SIZE=post_block,
                num_warps=4,
                num_stages=1,
            )
        else:
            _qwen3_moe_assignment_reduce_kernel[
                (
                    int(rows),
                    triton.cdiv(hidden_dim, block_n_down),
                )
            ](
                partial,
                final,
                partial.stride(0),
                partial.stride(1),
                final.stride(0),
                final.stride(1),
                hidden_dim,
                TOP_K=top_k,
                BLOCK_N=block_n_down,
                num_warps=_CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS,
                num_stages=_CFG_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES,
            )
        if workspace is not None:
            workspace["expert_grouped_compact_decode_last_rows"] = int(rows)
            workspace["expert_grouped_compact_decode_last_assignments"] = int(assignments)
            workspace["expert_grouped_compact_decode_last_num_experts"] = int(num_experts)
            workspace["expert_grouped_compact_decode_last_block_m"] = int(block_m)
            workspace["expert_grouped_compact_decode_last_token_accum"] = 0
            workspace["expert_grouped_compact_decode_last_active_list"] = int(use_active_list)
            workspace["expert_grouped_compact_decode_last_active_list_early_exit"] = int(
                use_active_list_early_exit
            )
            workspace["expert_grouped_compact_decode_last_fused_pack"] = int(use_fused_pack)
            workspace["expert_grouped_compact_decode_last_expert_grid_pack"] = int(
                use_expert_grid_pack
            )
            workspace["expert_grouped_compact_decode_last_coalesced_weights"] = int(
                _CFG_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS
            )
            workspace["expert_grouped_compact_decode_last_partial_reduce"] = 1
            workspace["expert_grouped_compact_decode_last_fused_post_moe"] = int(
                use_fused_post_moe
            )
            workspace["expert_grouped_compact_decode_last_gate_block_n"] = int(
                block_n_act
            )
            workspace["expert_grouped_compact_decode_last_down_block_n"] = int(
                block_n_down
            )
            workspace["expert_grouped_compact_decode_last_num_warps"] = int(
                _CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS
            )
            workspace["expert_grouped_compact_decode_last_num_stages"] = int(
                _CFG_EXPERT_GROUPED_COMPACT_NUM_STAGES
            )
            workspace["expert_grouped_compact_decode_last_gate_num_stages"] = int(
                _CFG_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES
            )
            workspace["expert_grouped_compact_decode_last_down_num_stages"] = int(
                _CFG_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES
            )
            workspace["expert_grouped_compact_decode_last_experts_per_program"] = int(
                experts_per_program
            )
            workspace["expert_grouped_compact_decode_last_paired_gate_up_dot"] = int(
                _CFG_EXPERT_GROUPED_COMPACT_PAIRED_GATE_UP_DOT
            )
            workspace["expert_grouped_compact_decode_last_split_gate_up"] = int(
                use_split_gate_up
            )
            workspace["expert_grouped_compact_decode_last_empty_expert_early_exit"] = int(
                _CFG_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT
            )
            workspace["expert_grouped_compact_decode_last_l2_grouped_grid"] = int(
                use_l2_grouped_grid
            )
            workspace["expert_grouped_compact_decode_last_l2_group_size"] = int(
                l2_group_size
            )
        return final

    direct_out = bool(
        _CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
    )
    if direct_out:
        accum = (out if out is not None else torch.empty_like(hidden_states)).zero_()
    else:
        accum = _workspace_tensor(
            workspace,
            "expert_grouped_compact_accum",
            (int(rows), hidden_dim),
            device=hidden.device,
            dtype=torch.float32,
            zero=True,
        )
    _qwen3_moe_expert_grouped_compact_down_accum_kernel[
        (
            candidate_slots,
            triton.cdiv(int(rows), block_m),
            triton.cdiv(hidden_dim, block_n_down),
        )
    ](
        activated,
        down_w,
        grid_experts,
        unique_or_count,
        counts,
        dense_tokens,
        dense_route,
        accum,
        activated.stride(0),
        activated.stride(1),
        activated.stride(2),
        down_w.stride(0),
        down_w.stride(1),
        down_w.stride(2),
        accum.stride(0),
        accum.stride(1),
        intermediate_dim,
        hidden_dim,
        ROWS=int(rows),
        ASSIGNMENTS=int(assignments),
        ACTIVE_LIST=use_active_list,
        EXPERT_GRID=process_expert_grid,
        COALESCED_WEIGHTS=bool(_CFG_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS),
        ACTIVE_LIST_EARLY_EXIT=use_active_list_early_exit,
        BLOCK_M=block_m,
        BLOCK_N=block_n_down,
        BLOCK_K=block_k_down,
        num_warps=_CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS,
        num_stages=_CFG_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES,
    )

    if workspace is not None:
        workspace["expert_grouped_compact_decode_last_rows"] = int(rows)
        workspace["expert_grouped_compact_decode_last_assignments"] = int(assignments)
        workspace["expert_grouped_compact_decode_last_num_experts"] = int(num_experts)
        workspace["expert_grouped_compact_decode_last_block_m"] = int(block_m)
        workspace["expert_grouped_compact_decode_last_token_accum"] = 0
        workspace["expert_grouped_compact_decode_last_active_list"] = int(use_active_list)
        workspace["expert_grouped_compact_decode_last_active_list_early_exit"] = int(
            use_active_list_early_exit
        )
        workspace["expert_grouped_compact_decode_last_fused_pack"] = int(use_fused_pack)
        workspace["expert_grouped_compact_decode_last_expert_grid_pack"] = int(
            use_expert_grid_pack
        )
        workspace["expert_grouped_compact_decode_last_coalesced_weights"] = int(
            _CFG_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS
        )
        workspace["expert_grouped_compact_decode_last_partial_reduce"] = 0
        workspace["expert_grouped_compact_decode_last_direct_out"] = int(direct_out)
        workspace["expert_grouped_compact_decode_last_gate_block_n"] = int(block_n_act)
        workspace["expert_grouped_compact_decode_last_down_block_n"] = int(block_n_down)
        workspace["expert_grouped_compact_decode_last_num_warps"] = int(
            _CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS
        )
        workspace["expert_grouped_compact_decode_last_num_stages"] = int(
            _CFG_EXPERT_GROUPED_COMPACT_NUM_STAGES
        )
        workspace["expert_grouped_compact_decode_last_gate_num_stages"] = int(
            _CFG_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES
        )
        workspace["expert_grouped_compact_decode_last_down_num_stages"] = int(
            _CFG_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES
        )
        workspace["expert_grouped_compact_decode_last_split_gate_up"] = int(
            use_split_gate_up
        )
        workspace["expert_grouped_compact_decode_last_empty_expert_early_exit"] = int(
            _CFG_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT
        )
        workspace["expert_grouped_compact_decode_last_l2_grouped_grid"] = int(
            use_l2_grouped_grid
        )
        workspace["expert_grouped_compact_decode_last_l2_group_size"] = int(
            l2_group_size
        )

    if direct_out:
        return accum

    final = out if out is not None else torch.empty_like(hidden_states)
    final.copy_(accum)
    return final


def _copy_accum_with_optional_residual(
    accum: torch.Tensor,
    hidden_states: torch.Tensor,
    out: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    workspace: Optional[dict[str, torch.Tensor]],
    *,
    workspace_prefix: str,
) -> torch.Tensor:
    final = out if out is not None else torch.empty_like(hidden_states)
    if residual is None:
        final.copy_(accum)
        return final

    if (
        _HAS_TRITON
        and accum.is_cuda
        and residual.is_cuda
        and final.is_cuda
        and accum.ndim == 2
        and residual.ndim == 2
        and final.ndim == 2
        and tuple(accum.shape) == tuple(residual.shape) == tuple(final.shape)
    ):
        rows, hidden_dim = residual.shape
        block_n = min(256, triton.next_power_of_2(int(hidden_dim)))
        _qwen3_moe_accum_add_residual_kernel[
            (int(rows), triton.cdiv(int(hidden_dim), block_n))
        ](
            accum,
            residual,
            final,
            accum.stride(0),
            accum.stride(1),
            residual.stride(0),
            residual.stride(1),
            final.stride(0),
            final.stride(1),
            int(hidden_dim),
            ROWS=int(rows),
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=1,
        )
        if workspace is not None:
            workspace[f"{workspace_prefix}_residual_fused"] = 1
        return final

    # Rare fallback path: keep semantics correct even if Triton is unavailable.
    torch.add(residual, accum.to(residual.dtype), out=final)
    if workspace is not None:
        workspace[f"{workspace_prefix}_residual_fused"] = 0
    return final


def _qwen3_moe_shared_route_decode(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    activation: str,
    out: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    workspace: Optional[dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    rows, hidden_dim = hidden_states.shape
    top_k = int(selected_experts.shape[1])
    intermediate_dim = int(gate_up_proj.shape[1] // 2)
    act_id = _activation_id(activation)

    hidden = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    gate_up_w = gate_up_proj if gate_up_proj.is_contiguous() else gate_up_proj.contiguous()
    down_w = down_proj if down_proj.is_contiguous() else down_proj.contiguous()
    selected = selected_experts if selected_experts.is_contiguous() else selected_experts.contiguous()
    route = routing_weights if routing_weights.is_contiguous() else routing_weights.contiguous()

    block_m = min(int(_CFG_SHARED_ROUTE_BLOCK_M), 16)
    block_n_act = min(_CFG_SHARED_ROUTE_GATE_BLOCK_N, triton.next_power_of_2(intermediate_dim))
    block_k_hidden = min(_CFG_BLOCK_K, triton.next_power_of_2(hidden_dim))
    block_n_down = min(_CFG_SHARED_ROUTE_DOWN_BLOCK_N, triton.next_power_of_2(hidden_dim))
    block_k_down = min(_CFG_BLOCK_K, triton.next_power_of_2(intermediate_dim))

    if _CFG_SHARED_ROUTE_SPLIT_GATE:
        split_block_m = min(int(_CFG_SHARED_ROUTE_SPLIT_GATE_BLOCK_M), 16)
        gate_up = _workspace_tensor(
            workspace,
            "shared_route_gate_up",
            (top_k, int(rows), 2 * intermediate_dim),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        _qwen3_moe_shared_route_gate_up_kernel[
            (
                top_k,
                triton.cdiv(int(rows), split_block_m),
                triton.cdiv(2 * intermediate_dim, block_n_act),
            )
        ](
            hidden,
            gate_up_w,
            selected,
            gate_up,
            hidden.stride(0),
            hidden.stride(1),
            gate_up_w.stride(0),
            gate_up_w.stride(1),
            gate_up_w.stride(2),
            selected.stride(1),
            gate_up.stride(0),
            gate_up.stride(1),
            gate_up.stride(2),
            hidden_dim,
            2 * intermediate_dim,
            ROWS=int(rows),
            BLOCK_M=split_block_m,
            BLOCK_N=block_n_act,
            BLOCK_K=block_k_hidden,
            num_warps=_CFG_NUM_WARPS,
            num_stages=_CFG_SHARED_ROUTE_SPLIT_GATE_NUM_STAGES,
        )

        accum = _workspace_tensor(
            workspace,
            "shared_route_accum",
            (int(rows), hidden_dim),
            device=hidden.device,
            dtype=torch.float32,
            zero=True,
        )
        _qwen3_moe_shared_route_swiglu_down_accum_kernel[
            (
                top_k,
                triton.cdiv(int(rows), split_block_m),
                triton.cdiv(hidden_dim, block_n_down),
            )
        ](
            gate_up,
            down_w,
            selected,
            route,
            accum,
            gate_up.stride(0),
            gate_up.stride(1),
            gate_up.stride(2),
            down_w.stride(0),
            down_w.stride(1),
            down_w.stride(2),
            route.stride(0),
            route.stride(1),
            accum.stride(0),
            accum.stride(1),
            intermediate_dim,
            hidden_dim,
            ROWS=int(rows),
            ACT=act_id,
            BLOCK_M=split_block_m,
            BLOCK_N=block_n_down,
            BLOCK_K=block_k_down,
            num_warps=_CFG_NUM_WARPS,
            num_stages=_CFG_SHARED_ROUTE_SPLIT_GATE_NUM_STAGES,
        )
        if workspace is not None:
            workspace["shared_route_decode_last_rows"] = int(rows)
            workspace["shared_route_decode_last_top_k"] = int(top_k)
            workspace["shared_route_decode_last_block_m"] = int(split_block_m)
            workspace["shared_route_decode_last_gate_k_splits"] = 1
            workspace["shared_route_decode_last_split_gate"] = 1
            workspace["shared_route_decode_last_token_accum"] = 0
            workspace["shared_route_decode_last_partial_reduce"] = 0
        return _copy_accum_with_optional_residual(
            accum,
            hidden_states,
            out,
            residual,
            workspace,
            workspace_prefix="shared_route_decode",
        )

    activated = _workspace_tensor(
        workspace,
        "shared_route_activated",
        (top_k, int(rows), intermediate_dim),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    gate_k_splits = int(_CFG_SHARED_ROUTE_GATE_K_SPLITS) if int(rows) == 1 else 1
    if gate_k_splits > 1:
        split_k = triton.cdiv(hidden_dim, gate_k_splits)
        partial_gate_up = _workspace_tensor(
            workspace,
            "shared_route_gate_k_partials",
            (gate_k_splits, top_k, 2 * intermediate_dim),
            device=hidden.device,
            dtype=torch.float32,
        )
        _qwen3_moe_shared_route_gate_k_split_kernel[
            (
                top_k,
                triton.cdiv(intermediate_dim, block_n_act),
                gate_k_splits,
            )
        ](
            hidden,
            gate_up_w,
            selected,
            partial_gate_up,
            hidden.stride(1),
            gate_up_w.stride(0),
            gate_up_w.stride(1),
            gate_up_w.stride(2),
            selected.stride(1),
            partial_gate_up.stride(0),
            partial_gate_up.stride(1),
            partial_gate_up.stride(2),
            hidden_dim,
            intermediate_dim,
            SPLIT_K=split_k,
            BLOCK_N=block_n_act,
            BLOCK_K=block_k_hidden,
            num_warps=_CFG_NUM_WARPS,
            num_stages=_CFG_NUM_STAGES,
        )
        _qwen3_moe_shared_route_gate_k_reduce_swiglu_kernel[
            (
                top_k,
                triton.cdiv(intermediate_dim, block_n_act),
            )
        ](
            partial_gate_up,
            activated,
            partial_gate_up.stride(0),
            partial_gate_up.stride(1),
            partial_gate_up.stride(2),
            activated.stride(0),
            activated.stride(2),
            intermediate_dim,
            ACT=act_id,
            K_SPLITS=gate_k_splits,
            BLOCK_N=block_n_act,
            num_warps=1,
            num_stages=1,
        )
    else:
        _qwen3_moe_shared_route_gate_swiglu_kernel[
            (
                top_k,
                triton.cdiv(int(rows), block_m),
                triton.cdiv(intermediate_dim, block_n_act),
            )
        ](
            hidden,
            gate_up_w,
            selected,
            activated,
            hidden.stride(0),
            hidden.stride(1),
            gate_up_w.stride(0),
            gate_up_w.stride(1),
            gate_up_w.stride(2),
            selected.stride(0),
            selected.stride(1),
            activated.stride(0),
            activated.stride(1),
            activated.stride(2),
            hidden_dim,
            intermediate_dim,
            ROWS=int(rows),
            TOP_K=top_k,
            ACT=act_id,
            BLOCK_M=block_m,
            BLOCK_N=block_n_act,
            BLOCK_K=block_k_hidden,
            COALESCED_WEIGHTS=bool(_CFG_SHARED_ROUTE_COALESCED_WEIGHTS),
            num_warps=_CFG_NUM_WARPS,
            num_stages=_CFG_NUM_STAGES,
        )

    if _CFG_SHARED_ROUTE_PARTIAL_REDUCE:
        partial = _workspace_tensor(
            workspace,
            "shared_route_partial",
            (top_k, int(rows), hidden_dim),
            device=hidden.device,
            dtype=torch.float32,
        )
        _qwen3_moe_shared_route_down_partial_kernel[
            (
                top_k,
                triton.cdiv(int(rows), block_m),
                triton.cdiv(hidden_dim, block_n_down),
            )
        ](
            activated,
            down_w,
            selected,
            route,
            partial,
            activated.stride(0),
            activated.stride(1),
            activated.stride(2),
            down_w.stride(0),
            down_w.stride(1),
            down_w.stride(2),
            route.stride(0),
            route.stride(1),
            partial.stride(0),
            partial.stride(1),
            partial.stride(2),
            intermediate_dim,
            hidden_dim,
            ROWS=int(rows),
            BLOCK_M=block_m,
            BLOCK_N=block_n_down,
            BLOCK_K=block_k_down,
            COALESCED_WEIGHTS=bool(_CFG_SHARED_ROUTE_COALESCED_WEIGHTS),
            num_warps=_CFG_NUM_WARPS,
            num_stages=_CFG_NUM_STAGES,
        )
        final = out if out is not None else torch.empty_like(hidden_states)
        _qwen3_moe_shared_route_reduce_kernel[
            (
                triton.cdiv(int(rows), block_m),
                triton.cdiv(hidden_dim, block_n_down),
            )
        ](
            partial,
            final,
            partial.stride(0),
            partial.stride(1),
            partial.stride(2),
            final.stride(0),
            final.stride(1),
            hidden_dim,
            ROWS=int(rows),
            TOP_K=top_k,
            BLOCK_M=block_m,
            BLOCK_N=block_n_down,
            num_warps=1,
            num_stages=1,
        )
        if workspace is not None:
            workspace["shared_route_decode_last_rows"] = int(rows)
            workspace["shared_route_decode_last_top_k"] = int(top_k)
            workspace["shared_route_decode_last_block_m"] = int(block_m)
            workspace["shared_route_decode_last_gate_k_splits"] = int(gate_k_splits)
            workspace["shared_route_decode_last_split_gate"] = 0
            workspace["shared_route_decode_last_token_accum"] = 0
            workspace["shared_route_decode_last_partial_reduce"] = 1
        return final

    if _CFG_SHARED_ROUTE_TOKEN_ACCUM:
        final = out if out is not None else torch.empty_like(hidden_states)
        _qwen3_moe_shared_route_down_token_accum_kernel[
            (
                int(rows),
                triton.cdiv(hidden_dim, block_n_down),
            )
        ](
            activated,
            down_w,
            selected,
            route,
            final,
            activated.stride(0),
            activated.stride(1),
            activated.stride(2),
            down_w.stride(0),
            down_w.stride(1),
            down_w.stride(2),
            selected.stride(0),
            selected.stride(1),
            route.stride(0),
            route.stride(1),
            final.stride(0),
            final.stride(1),
            intermediate_dim,
            hidden_dim,
            TOP_K=top_k,
            BLOCK_N=block_n_down,
            BLOCK_K=block_k_down,
            num_warps=_CFG_SHARED_ROUTE_TOKEN_ACCUM_NUM_WARPS,
            num_stages=_CFG_NUM_STAGES,
        )
        if workspace is not None:
            workspace["shared_route_decode_last_rows"] = int(rows)
            workspace["shared_route_decode_last_top_k"] = int(top_k)
            workspace["shared_route_decode_last_block_m"] = int(block_m)
            workspace["shared_route_decode_last_gate_k_splits"] = int(gate_k_splits)
            workspace["shared_route_decode_last_split_gate"] = 0
            workspace["shared_route_decode_last_token_accum"] = 1
            workspace["shared_route_decode_last_partial_reduce"] = 0
        return final

    accum = _workspace_tensor(
        workspace,
        "shared_route_accum",
        (int(rows), hidden_dim),
        device=hidden.device,
        dtype=torch.float32,
        zero=True,
    )
    _qwen3_moe_shared_route_down_accum_kernel[
        (
            top_k,
            triton.cdiv(int(rows), block_m),
            triton.cdiv(hidden_dim, block_n_down),
        )
    ](
        activated,
        down_w,
        selected,
        route,
        accum,
        activated.stride(0),
        activated.stride(1),
        activated.stride(2),
        down_w.stride(0),
        down_w.stride(1),
        down_w.stride(2),
        route.stride(0),
        route.stride(1),
        accum.stride(0),
        accum.stride(1),
        intermediate_dim,
        hidden_dim,
        ROWS=int(rows),
        TOP_K=top_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n_down,
        BLOCK_K=block_k_down,
        COALESCED_WEIGHTS=bool(_CFG_SHARED_ROUTE_COALESCED_WEIGHTS),
        num_warps=_CFG_NUM_WARPS,
        num_stages=_CFG_NUM_STAGES,
    )

    if workspace is not None:
        workspace["shared_route_decode_last_rows"] = int(rows)
        workspace["shared_route_decode_last_top_k"] = int(top_k)
        workspace["shared_route_decode_last_block_m"] = int(block_m)
        workspace["shared_route_decode_last_gate_k_splits"] = int(gate_k_splits)
        workspace["shared_route_decode_last_split_gate"] = 0
        workspace["shared_route_decode_last_token_accum"] = 0
        workspace["shared_route_decode_last_partial_reduce"] = 0

    return _copy_accum_with_optional_residual(
        accum,
        hidden_states,
        out,
        residual,
        workspace,
        workspace_prefix="shared_route_decode",
    )


def _qwen3_moe_route_matrix_decode(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    activation: str,
    out: Optional[torch.Tensor],
    workspace: Optional[dict[str, torch.Tensor]],
) -> torch.Tensor:
    rows, hidden_dim = hidden_states.shape
    top_k = int(selected_experts.shape[1])
    intermediate_dim = int(gate_up_proj.shape[1] // 2)
    act_id = _activation_id(activation)

    hidden = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    gate_up_w = gate_up_proj if gate_up_proj.is_contiguous() else gate_up_proj.contiguous()
    down_w = down_proj if down_proj.is_contiguous() else down_proj.contiguous()
    selected = selected_experts if selected_experts.is_contiguous() else selected_experts.contiguous()
    route = routing_weights if routing_weights.is_contiguous() else routing_weights.contiguous()

    block_m = min(
        int(_CFG_EXPERT_GROUPED_BLOCK_M),
        16,
        int(triton.next_power_of_2(max(int(rows), 1))),
    )
    block_n_act = min(_CFG_BLOCK_N, triton.next_power_of_2(intermediate_dim))
    block_k_hidden = min(_CFG_BLOCK_K, triton.next_power_of_2(hidden_dim))
    block_n_down = min(_CFG_BLOCK_N, triton.next_power_of_2(hidden_dim))
    block_k_down = min(_CFG_BLOCK_K, triton.next_power_of_2(intermediate_dim))

    activated = _workspace_tensor(
        workspace,
        "route_matrix_activated",
        (top_k, int(rows), intermediate_dim),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    _qwen3_moe_route_matrix_gate_swiglu_kernel[
        (
            top_k,
            triton.cdiv(int(rows), block_m),
            triton.cdiv(intermediate_dim, block_n_act),
        )
    ](
        hidden,
        gate_up_w,
        selected,
        activated,
        hidden.stride(0),
        hidden.stride(1),
        gate_up_w.stride(0),
        gate_up_w.stride(1),
        gate_up_w.stride(2),
        selected.stride(0),
        selected.stride(1),
        activated.stride(0),
        activated.stride(1),
        activated.stride(2),
        hidden_dim,
        intermediate_dim,
        ROWS=int(rows),
        TOP_K=top_k,
        ACT=act_id,
        BLOCK_M=block_m,
        BLOCK_N=block_n_act,
        BLOCK_K=block_k_hidden,
        num_warps=_CFG_NUM_WARPS,
        num_stages=_CFG_NUM_STAGES,
    )

    accum = _workspace_tensor(
        workspace,
        "route_matrix_accum",
        (int(rows), hidden_dim),
        device=hidden.device,
        dtype=torch.float32,
        zero=True,
    )
    _qwen3_moe_route_matrix_down_accum_kernel[
        (
            top_k,
            triton.cdiv(int(rows), block_m),
            triton.cdiv(hidden_dim, block_n_down),
        )
    ](
        activated,
        down_w,
        selected,
        route,
        accum,
        activated.stride(0),
        activated.stride(1),
        activated.stride(2),
        down_w.stride(0),
        down_w.stride(1),
        down_w.stride(2),
        selected.stride(0),
        selected.stride(1),
        route.stride(0),
        route.stride(1),
        accum.stride(0),
        accum.stride(1),
        intermediate_dim,
        hidden_dim,
        ROWS=int(rows),
        TOP_K=top_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n_down,
        BLOCK_K=block_k_down,
        num_warps=_CFG_NUM_WARPS,
        num_stages=_CFG_NUM_STAGES,
    )

    if workspace is not None:
        workspace["route_matrix_decode_last_rows"] = int(rows)
        workspace["route_matrix_decode_last_top_k"] = int(top_k)
        workspace["route_matrix_decode_last_block_m"] = int(block_m)

    final = out if out is not None else torch.empty_like(hidden_states)
    final.copy_(accum)
    return final


def qwen3_moe_grouped_decode(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    activation: str = "silu",
    out: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    workspace: Optional[dict[str, torch.Tensor]] = None,
    max_assignments: Optional[int] = None,
    expert_grouped_compact: Optional[bool] = None,
    expert_grouped_min_rows: Optional[int] = None,
    expert_grouped_max_rows: Optional[int] = None,
    assignment_partial_reduce: bool = False,
    expert_grouped_compact_partial_reduce: Optional[bool] = None,
    compact_route_prepacked: bool = False,
    post_moe_shared: Optional[torch.Tensor] = None,
    post_moe_shared_weight: Optional[torch.Tensor] = None,
    post_moe_expert_weight: Optional[torch.Tensor] = None,
    post_moe_final_weight: Optional[torch.Tensor] = None,
    post_moe_residual: Optional[torch.Tensor] = None,
    post_moe_wait_event: Optional[object] = None,
    post_moe_layer_scalar: Optional[torch.Tensor] = None,
    post_moe_next_norm_weight: Optional[torch.Tensor] = None,
    post_moe_next_norm_out: Optional[torch.Tensor] = None,
    post_moe_write_next_norm: bool = False,
    post_moe_eps: float = 1e-6,
) -> torch.Tensor:
    """
    Grouped Qwen3 MoE decode/small-M forward.

    Shapes:
      hidden_states:    [M, H]
      gate_up_proj:     [E, 2I, H]
      down_proj:        [E, H, I]
      selected_experts: [M, K]
      routing_weights:  [M, K]
    """
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be [M, H]")
    if selected_experts.shape != routing_weights.shape:
        raise ValueError("selected_experts and routing_weights must have the same shape")
    if selected_experts.ndim != 2:
        raise ValueError("selected_experts must be [M, top_k]")
    if residual is not None and tuple(residual.shape) != tuple(hidden_states.shape):
        raise ValueError("residual must have the same shape as hidden_states")
    post_moe_values = (
        post_moe_shared,
        post_moe_shared_weight,
        post_moe_expert_weight,
        post_moe_final_weight,
        post_moe_residual,
    )
    fused_post_moe_requested = all(value is not None for value in post_moe_values)
    if any(value is not None for value in post_moe_values) and not fused_post_moe_requested:
        raise ValueError("fused Gemma4 post-MoE inputs must be provided together")
    next_norm_values = (
        post_moe_layer_scalar,
        post_moe_next_norm_weight,
        post_moe_next_norm_out,
    )
    fused_next_norm_requested = any(value is not None for value in next_norm_values)
    if fused_next_norm_requested and not all(
        value is not None for value in next_norm_values
    ):
        raise ValueError(
            "fused Gemma4 layer-scalar/next-norm inputs must be provided together"
        )
    if fused_next_norm_requested and not fused_post_moe_requested:
        raise ValueError(
            "fused Gemma4 layer-scalar/next-norm requires fused post-MoE"
        )

    rows, hidden_dim = hidden_states.shape
    if rows == 0:
        if residual is not None:
            return residual
        return torch.zeros_like(hidden_states) if out is None else out.zero_()
    top_k = int(selected_experts.shape[1])
    intermediate_dim = int(gate_up_proj.shape[1] // 2)
    act_id = _activation_id(activation)
    selected_compact_grouped = (
        bool(_CFG_EXPERT_GROUPED_COMPACT_DECODE)
        if expert_grouped_compact is None
        else bool(expert_grouped_compact)
    )
    selected_grouped_min_rows = (
        int(_CFG_EXPERT_GROUPED_MIN_ROWS)
        if expert_grouped_min_rows is None
        else max(1, int(expert_grouped_min_rows))
    )
    selected_grouped_max_rows = (
        int(_CFG_EXPERT_GROUPED_MAX_ROWS)
        if expert_grouped_max_rows is None
        else max(1, int(expert_grouped_max_rows))
    )
    deterministic_reduce_requested = bool(
        assignment_partial_reduce
        or expert_grouped_compact_partial_reduce is True
    )

    use_triton = (
        _HAS_TRITON
        and hidden_states.is_cuda
        and gate_up_proj.is_cuda
        and down_proj.is_cuda
        and selected_experts.is_cuda
        and routing_weights.is_cuda
        and not torch.is_grad_enabled()
        and qwen3_moe_grouped_prefers_triton_shape(
            rows,
            top_k,
            hidden_dim,
            intermediate_dim,
            max_assignments=max_assignments,
        )
    )
    if not use_triton:
        if residual is not None or fused_post_moe_requested or compact_route_prepacked:
            raise RuntimeError("residual fusion requires the Triton shared-route decode path")
        return _fallback_grouped_moe(
            hidden_states,
            gate_up_proj,
            down_proj,
            selected_experts,
            routing_weights,
            activation,
            out,
        )

    hidden = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    gate_up_w = gate_up_proj if gate_up_proj.is_contiguous() else gate_up_proj.contiguous()
    down_w = down_proj if down_proj.is_contiguous() else down_proj.contiguous()
    experts = selected_experts.reshape(-1).to(torch.int64).contiguous()
    route = routing_weights.reshape(-1).contiguous()
    token_ids = _workspace_token_ids(workspace, rows, top_k, device=hidden.device)
    assignments = int(experts.numel())
    if workspace is not None:
        workspace["grouped_decode_last_path"] = "assignment"
        workspace["grouped_decode_last_rows"] = int(rows)
        workspace["grouped_decode_last_assignments"] = int(assignments)

    use_shared_route = (
        _CFG_SHARED_ROUTE_DECODE
        and not deterministic_reduce_requested
        and not (_CFG_SINGLE_ROW_GEMV and rows == 1)
        and (
            rows == 1
            or (
                _CFG_SHARED_ROUTE_ASSUME_IDENTICAL
                and rows <= _CFG_SHARED_ROUTE_BATCH_MAX_ROWS
            )
        )
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and not int((workspace or {}).get("shared_route_decode_disabled", 0) or 0)
    )
    if use_shared_route:
        if residual is not None and (
            _CFG_SHARED_ROUTE_PARTIAL_REDUCE or _CFG_SHARED_ROUTE_TOKEN_ACCUM
        ):
            raise RuntimeError(
                "residual fusion is supported for shared-route accum decode only"
            )
        try:
            result = _qwen3_moe_shared_route_decode(
                hidden_states,
                gate_up_proj,
                down_proj,
                selected_experts,
                routing_weights,
                activation=activation,
                out=residual if residual is not None else out,
                residual=residual,
                workspace=workspace,
            )
            if workspace is not None:
                workspace["grouped_decode_last_path"] = "shared_route"
            return result
        except Exception as exc:
            if residual is not None:
                raise
            if workspace is not None:
                workspace["shared_route_decode_disabled"] = 1
                workspace["shared_route_decode_fail_reason"] = str(exc)
    elif residual is not None:
        raise RuntimeError("residual fusion requires the shared-route decode path")

    use_general_grouped = (
        _CFG_EXPERT_GROUPED_GENERAL_DECODE
        and not deterministic_reduce_requested
        and rows >= _CFG_EXPERT_GROUPED_MIN_ROWS
        and rows <= _CFG_EXPERT_GROUPED_MAX_ROWS
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and not int((workspace or {}).get("expert_grouped_general_decode_disabled", 0) or 0)
    )
    if use_general_grouped:
        try:
            result = qwen3_moe_segmented_prefill(
                hidden_states,
                gate_up_proj,
                down_proj,
                selected_experts,
                routing_weights,
                activation=activation,
                out=out,
                workspace=workspace,
            )
            if workspace is not None:
                workspace["expert_grouped_general_decode_last_rows"] = int(rows)
                workspace["expert_grouped_general_decode_last_assignments"] = int(assignments)
                workspace["grouped_decode_last_path"] = "expert_grouped_general"
            return result
        except Exception as exc:
            if workspace is not None:
                workspace["expert_grouped_general_decode_disabled"] = 1
                workspace["expert_grouped_general_decode_fail_reason"] = str(exc)

    use_compact_grouped = (
        selected_compact_grouped
        and not (_CFG_SINGLE_ROW_GEMV and rows == 1)
        and rows >= selected_grouped_min_rows
        and rows <= selected_grouped_max_rows
        and assignments <= 128
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and not int((workspace or {}).get("expert_grouped_compact_decode_disabled", 0) or 0)
    )
    if use_compact_grouped:
        try:
            result = _qwen3_moe_expert_grouped_compact_decode(
                hidden_states,
                gate_up_proj,
                down_proj,
                selected_experts,
                routing_weights,
                activation=activation,
                out=out,
                workspace=workspace,
                partial_reduce=expert_grouped_compact_partial_reduce,
                route_prepacked=compact_route_prepacked,
                post_moe_shared=post_moe_shared,
                post_moe_shared_weight=post_moe_shared_weight,
                post_moe_expert_weight=post_moe_expert_weight,
                post_moe_final_weight=post_moe_final_weight,
                post_moe_residual=post_moe_residual,
                post_moe_wait_event=post_moe_wait_event,
                post_moe_layer_scalar=post_moe_layer_scalar,
                post_moe_next_norm_weight=post_moe_next_norm_weight,
                post_moe_next_norm_out=post_moe_next_norm_out,
                post_moe_write_next_norm=post_moe_write_next_norm,
                post_moe_eps=post_moe_eps,
            )
            if workspace is not None:
                workspace["expert_grouped_compact_decode_hits"] = int(
                    workspace.get("expert_grouped_compact_decode_hits", 0) or 0
                ) + 1
                workspace["grouped_decode_last_path"] = "expert_grouped_compact"
            return result
        except Exception as exc:
            if compact_route_prepacked:
                if workspace is not None:
                    workspace["expert_grouped_compact_route_prepacked_disabled"] = 1
                    workspace["expert_grouped_compact_route_prepacked_fail_reason"] = str(
                        exc
                    )
                try:
                    result = _qwen3_moe_expert_grouped_compact_decode(
                        hidden_states,
                        gate_up_proj,
                        down_proj,
                        selected_experts,
                        routing_weights,
                        activation=activation,
                        out=out,
                        workspace=workspace,
                        partial_reduce=expert_grouped_compact_partial_reduce,
                        route_prepacked=False,
                        post_moe_shared=post_moe_shared,
                        post_moe_shared_weight=post_moe_shared_weight,
                        post_moe_expert_weight=post_moe_expert_weight,
                        post_moe_final_weight=post_moe_final_weight,
                        post_moe_residual=post_moe_residual,
                        post_moe_wait_event=post_moe_wait_event,
                        post_moe_layer_scalar=post_moe_layer_scalar,
                        post_moe_next_norm_weight=post_moe_next_norm_weight,
                        post_moe_next_norm_out=post_moe_next_norm_out,
                        post_moe_write_next_norm=post_moe_write_next_norm,
                        post_moe_eps=post_moe_eps,
                    )
                    if workspace is not None:
                        workspace["expert_grouped_compact_decode_hits"] = int(
                            workspace.get("expert_grouped_compact_decode_hits", 0)
                            or 0
                        ) + 1
                        workspace["grouped_decode_last_path"] = (
                            "expert_grouped_compact_pack_fallback"
                        )
                    return result
                except Exception as fallback_exc:
                    exc = fallback_exc
            if workspace is not None:
                workspace["expert_grouped_compact_decode_disabled"] = 1
                workspace["expert_grouped_compact_decode_fail_reason"] = str(exc)
            if fused_post_moe_requested:
                raise

    if compact_route_prepacked:
        raise RuntimeError("prepacked compact route did not reach compact grouped decode")

    if fused_post_moe_requested:
        raise RuntimeError("fused Gemma4 post-MoE requires compact grouped decode")

    use_expert_grouped = (
        _CFG_EXPERT_GROUPED_DECODE
        and _CFG_EXPERT_GROUPED_DENSE_DECODE
        and not deterministic_reduce_requested
        and rows >= _CFG_EXPERT_GROUPED_MIN_ROWS
        and rows <= _CFG_EXPERT_GROUPED_MAX_ROWS
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and not int((workspace or {}).get("expert_grouped_decode_disabled", 0) or 0)
    )
    if use_expert_grouped:
        try:
            return _qwen3_moe_expert_grouped_decode(
                hidden_states,
                gate_up_proj,
                down_proj,
                selected_experts,
                routing_weights,
                activation=activation,
                out=out,
                workspace=workspace,
            )
        except Exception as exc:
            if workspace is not None:
                workspace["expert_grouped_decode_disabled"] = 1
                workspace["expert_grouped_decode_fail_reason"] = str(exc)

    use_route_matrix = (
        _CFG_ROUTE_MATRIX_DECODE
        and not deterministic_reduce_requested
        and rows >= _CFG_EXPERT_GROUPED_MIN_ROWS
        and rows <= _CFG_ROUTE_MATRIX_MAX_ROWS
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and not int((workspace or {}).get("route_matrix_decode_disabled", 0) or 0)
    )
    if use_route_matrix:
        try:
            return _qwen3_moe_route_matrix_decode(
                hidden_states,
                gate_up_proj,
                down_proj,
                selected_experts,
                routing_weights,
                activation=activation,
                out=out,
                workspace=workspace,
            )
        except Exception as exc:
            if workspace is not None:
                workspace["route_matrix_decode_disabled"] = 1
                workspace["route_matrix_decode_fail_reason"] = str(exc)

    if workspace is not None:
        workspace["grouped_decode_last_path"] = "assignment"

    block_k = min(_CFG_BLOCK_K, triton.next_power_of_2(hidden_dim))
    block_n_down = min(_CFG_BLOCK_N, triton.next_power_of_2(hidden_dim))
    block_k_down = min(_CFG_BLOCK_K, triton.next_power_of_2(intermediate_dim))
    use_token_accum = bool(_CFG_TOKEN_ACCUM and rows >= _CFG_TOKEN_ACCUM_MIN_ROWS)
    use_grouped_fused_gate = bool(
        _CFG_GROUPED_FUSED_GATE
        and use_token_accum
        and not assignment_partial_reduce
    )
    if use_grouped_fused_gate:
        dot_graph_safe = (not _CFG_DECODE_CUDA_GRAPHS) or _CFG_GROUPED_DOT_ALLOW_CUDA_GRAPHS
        use_dot = bool(
            _CFG_GROUPED_DOT
            and dot_graph_safe
            and not int((workspace or {}).get("grouped_dot_disabled", 0) or 0)
        )
        if _CFG_GROUPED_DOT and not dot_graph_safe and workspace is not None:
            workspace["grouped_dot_graph_disabled"] = 1
        activated = _workspace_tensor(
            workspace,
            "activated",
            (assignments, intermediate_dim),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        block_n_act = min(_CFG_BLOCK_N, triton.next_power_of_2(intermediate_dim))
        try:
            _qwen3_moe_gate_swiglu_kernel[
                (assignments, triton.cdiv(intermediate_dim, block_n_act))
            ](
                hidden,
                gate_up_w,
                experts,
                token_ids,
                activated,
                hidden.stride(0),
                hidden.stride(1),
                gate_up_w.stride(0),
                gate_up_w.stride(1),
                gate_up_w.stride(2),
                activated.stride(0),
                activated.stride(1),
                hidden_dim,
                intermediate_dim,
                ACT=act_id,
                BLOCK_N=block_n_act,
                BLOCK_K=block_k,
                USE_DOT=use_dot,
                num_warps=_CFG_NUM_WARPS,
                num_stages=_CFG_NUM_STAGES,
            )
            final = out if out is not None else torch.empty_like(hidden_states)
            _qwen3_moe_down_from_act_token_accum_kernel[
                (rows, triton.cdiv(hidden_dim, block_n_down))
            ](
                activated,
                down_w,
                experts,
                route,
                final,
                activated.stride(0),
                activated.stride(1),
                down_w.stride(0),
                down_w.stride(1),
                down_w.stride(2),
                final.stride(0),
                final.stride(1),
                intermediate_dim,
                hidden_dim,
                TOP_K=top_k,
                BLOCK_N=block_n_down,
                BLOCK_K=block_k_down,
                USE_DOT=use_dot,
                num_warps=_CFG_NUM_WARPS,
                num_stages=_CFG_NUM_STAGES,
            )
            return final
        except Exception:
            if use_dot and workspace is not None:
                workspace["grouped_dot_disabled"] = 1
                return qwen3_moe_grouped_decode(
                    hidden_states,
                    gate_up_proj,
                    down_proj,
                    selected_experts,
                    routing_weights,
                    activation=activation,
                    out=out,
                    workspace=workspace,
                    assignment_partial_reduce=assignment_partial_reduce,
                    expert_grouped_compact_partial_reduce=(
                        expert_grouped_compact_partial_reduce
                    ),
                )
            raise

    gate_up = _workspace_tensor(
        workspace,
        "gate_up",
        (assignments, 2 * intermediate_dim),
        device=hidden.device,
        dtype=hidden.dtype,
    )

    block_n = min(_CFG_BLOCK_N, triton.next_power_of_2(2 * intermediate_dim))
    _qwen3_moe_gate_up_kernel[(assignments, triton.cdiv(2 * intermediate_dim, block_n))](
        hidden,
        gate_up_w,
        experts,
        token_ids,
        gate_up,
        hidden.stride(0),
        hidden.stride(1),
        gate_up_w.stride(0),
        gate_up_w.stride(1),
        gate_up_w.stride(2),
        gate_up.stride(0),
        gate_up.stride(1),
        hidden_dim,
        2 * intermediate_dim,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=_CFG_NUM_WARPS,
        num_stages=_CFG_NUM_STAGES,
    )

    if assignment_partial_reduce:
        partial = _workspace_tensor(
            workspace,
            "assignment_partial",
            (assignments, hidden_dim),
            device=hidden.device,
            dtype=torch.float32,
        )
        _qwen3_moe_swiglu_down_partial_kernel[
            (assignments, triton.cdiv(hidden_dim, block_n_down))
        ](
            gate_up,
            down_w,
            experts,
            route,
            partial,
            gate_up.stride(0),
            gate_up.stride(1),
            down_w.stride(0),
            down_w.stride(1),
            down_w.stride(2),
            partial.stride(0),
            partial.stride(1),
            intermediate_dim,
            hidden_dim,
            ACT=act_id,
            BLOCK_N=block_n_down,
            BLOCK_K=block_k_down,
            num_warps=_CFG_NUM_WARPS,
            num_stages=_CFG_NUM_STAGES,
        )
        final = out if out is not None else torch.empty_like(hidden_states)
        _qwen3_moe_assignment_reduce_kernel[
            (rows, triton.cdiv(hidden_dim, block_n_down))
        ](
            partial,
            final,
            partial.stride(0),
            partial.stride(1),
            final.stride(0),
            final.stride(1),
            hidden_dim,
            TOP_K=top_k,
            BLOCK_N=block_n_down,
            num_warps=_CFG_NUM_WARPS,
            num_stages=_CFG_NUM_STAGES,
        )
        if workspace is not None:
            workspace["grouped_decode_last_partial_reduce"] = 1
        return final

    if workspace is not None:
        workspace["grouped_decode_last_partial_reduce"] = 0

    if use_token_accum:
        final = out if out is not None else torch.empty_like(hidden_states)
        _qwen3_moe_swiglu_down_token_accum_kernel[(rows, triton.cdiv(hidden_dim, block_n_down))](
            gate_up,
            down_w,
            experts,
            route,
            final,
            gate_up.stride(0),
            gate_up.stride(1),
            down_w.stride(0),
            down_w.stride(1),
            down_w.stride(2),
            final.stride(0),
            final.stride(1),
            intermediate_dim,
            hidden_dim,
            TOP_K=top_k,
            ACT=act_id,
            BLOCK_N=block_n_down,
            BLOCK_K=block_k_down,
            num_warps=_CFG_NUM_WARPS,
            num_stages=_CFG_NUM_STAGES,
        )
        return final

    accum = _workspace_tensor(
        workspace,
        "accum",
        (rows, hidden_dim),
        device=hidden.device,
        dtype=torch.float32,
        zero=True,
    )
    _qwen3_moe_swiglu_down_accum_kernel[(assignments, triton.cdiv(hidden_dim, block_n_down))](
        gate_up,
        down_w,
        experts,
        token_ids,
        route,
        accum,
        gate_up.stride(0),
        gate_up.stride(1),
        down_w.stride(0),
        down_w.stride(1),
        down_w.stride(2),
        accum.stride(0),
        accum.stride(1),
        intermediate_dim,
        hidden_dim,
        ACT=act_id,
        BLOCK_N=block_n_down,
        BLOCK_K=block_k_down,
        num_warps=_CFG_NUM_WARPS,
        num_stages=_CFG_NUM_STAGES,
    )

    if out is not None:
        out.copy_(accum)
        return out
    result = accum.to(hidden_states.dtype)
    return result


def _route_assignments_by_expert_scatter(
    flat_experts: torch.Tensor,
    flat_route: torch.Tensor,
    starts: torch.Tensor,
    *,
    rows: int,
    top_k: int,
    num_experts: int,
    workspace: Optional[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assignments = int(flat_experts.numel())
    device = flat_experts.device
    sorted_tokens = _workspace_tensor(
        workspace,
        "segmented_sorted_tokens",
        (assignments,),
        device=device,
        dtype=torch.int64,
    )
    sorted_route = _workspace_tensor(
        workspace,
        "segmented_sorted_route",
        (assignments,),
        device=device,
        dtype=flat_route.dtype,
    )
    sorted_slots = _workspace_tensor(
        workspace,
        "segmented_sorted_slots",
        (assignments,),
        device=device,
        dtype=torch.int64,
    )
    counters = _workspace_tensor(
        workspace,
        "segmented_route_counters",
        (int(num_experts),),
        device=device,
        dtype=torch.int32,
        zero=True,
    )
    block = int(_CFG_SEGMENTED_PREFILL_ROUTE_BLOCK)
    _qwen3_moe_route_scatter_by_expert_kernel[
        (triton.cdiv(assignments, block),)
    ](
        flat_experts,
        flat_route,
        starts,
        counters,
        sorted_tokens,
        sorted_route,
        sorted_slots,
        int(assignments),
        TOP_K=int(top_k),
        BLOCK=block,
    )
    if workspace is not None:
        workspace["segmented_prefill_route_scatter"] = 1
        workspace["segmented_prefill_route_scatter_fail_reason"] = ""
    return sorted_tokens, sorted_route, sorted_slots


def _route_assignments_by_expert_fixed_pack(
    flat_experts: torch.Tensor,
    flat_route: torch.Tensor,
    *,
    rows: int,
    top_k: int,
    num_experts: int,
    block_m: int,
    workspace: Optional[dict[str, torch.Tensor]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    assignments = int(flat_experts.numel())
    capacity = int(num_experts) * int(rows)
    m_tiles = triton.cdiv(int(rows), int(block_m))
    num_tiles = int(num_experts) * int(m_tiles)
    device = flat_experts.device

    sorted_tokens = _workspace_tensor(
        workspace,
        "segmented_fixed_sorted_tokens",
        (capacity,),
        device=device,
        dtype=torch.int64,
    )
    sorted_route = _workspace_tensor(
        workspace,
        "segmented_fixed_sorted_route",
        (capacity,),
        device=device,
        dtype=flat_route.dtype,
    )
    sorted_slots = _workspace_tensor(
        workspace,
        "segmented_fixed_sorted_slots",
        (capacity,),
        device=device,
        dtype=torch.int64,
    )
    tile_experts = _workspace_tensor(
        workspace,
        "segmented_fixed_tile_experts",
        (num_tiles,),
        device=device,
        dtype=torch.int64,
    )
    tile_starts = _workspace_tensor(
        workspace,
        "segmented_fixed_tile_starts",
        (num_tiles,),
        device=device,
        dtype=torch.int64,
    )
    tile_lengths = _workspace_tensor(
        workspace,
        "segmented_fixed_tile_lengths",
        (num_tiles,),
        device=device,
        dtype=torch.int64,
    )
    num_tiles_gpu = None if workspace is None else workspace.get(
        "segmented_fixed_num_tiles"
    )
    if (
        num_tiles_gpu is None
        or tuple(num_tiles_gpu.shape) != (1,)
        or num_tiles_gpu.device != device
        or num_tiles_gpu.dtype != torch.int64
    ):
        num_tiles_gpu = torch.full(
            (1,),
            num_tiles,
            device=device,
            dtype=torch.int64,
        )
        if workspace is not None:
            workspace["segmented_fixed_num_tiles"] = num_tiles_gpu
            workspace["segmented_fixed_num_tiles_value"] = int(num_tiles)
    elif workspace is not None and int(
        workspace.get("segmented_fixed_num_tiles_value", -1)
    ) != int(num_tiles):
        num_tiles_gpu.fill_(int(num_tiles))
        workspace["segmented_fixed_num_tiles_value"] = int(num_tiles)

    block_assignments = int(triton.next_power_of_2(max(1, assignments)))
    _qwen3_moe_fixed_route_pack_kernel[(int(num_experts),)](
        flat_experts,
        flat_route,
        sorted_tokens,
        sorted_route,
        sorted_slots,
        tile_experts,
        tile_starts,
        tile_lengths,
        ASSIGNMENTS=assignments,
        ROWS=int(rows),
        TOP_K=int(top_k),
        BLOCK_ASSIGNMENTS=block_assignments,
        BLOCK_M=int(block_m),
        M_TILES=int(m_tiles),
        num_warps=4,
        num_stages=1,
    )
    if workspace is not None:
        workspace["segmented_prefill_route_scatter"] = 0
        workspace["segmented_prefill_fixed_route_pack"] = 1
        workspace["segmented_prefill_fixed_route_capacity"] = int(capacity)
    return (
        sorted_tokens,
        sorted_route,
        sorted_slots,
        tile_experts,
        tile_starts,
        tile_lengths,
        num_tiles_gpu,
        int(num_tiles),
    )


def _route_assignments_by_expert_compact_pack(
    flat_experts: torch.Tensor,
    flat_route: torch.Tensor,
    *,
    rows: int,
    top_k: int,
    num_experts: int,
    block_m: int,
    workspace: Optional[dict[str, torch.Tensor]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Pack medium prefill routes without host syncs or assignment atomics."""
    assignments = int(flat_experts.numel())
    max_tiles = _segmented_tile_upper_bound(
        assignments,
        int(num_experts),
        int(block_m),
    )
    device = flat_experts.device

    sorted_tokens = _workspace_tensor(
        workspace,
        "segmented_compact_sorted_tokens",
        (assignments,),
        device=device,
        dtype=torch.int64,
    )
    sorted_route = _workspace_tensor(
        workspace,
        "segmented_compact_sorted_route",
        (assignments,),
        device=device,
        dtype=flat_route.dtype,
    )
    sorted_slots = _workspace_tensor(
        workspace,
        "segmented_compact_sorted_slots",
        (assignments,),
        device=device,
        dtype=torch.int64,
    )
    tile_experts = _workspace_tensor(
        workspace,
        "segmented_compact_tile_experts",
        (max_tiles,),
        device=device,
        dtype=torch.int64,
    )
    tile_starts = _workspace_tensor(
        workspace,
        "segmented_compact_tile_starts",
        (max_tiles,),
        device=device,
        dtype=torch.int64,
    )
    tile_lengths = _workspace_tensor(
        workspace,
        "segmented_compact_tile_lengths",
        (max_tiles,),
        device=device,
        dtype=torch.int64,
    )
    num_tiles_gpu = _workspace_tensor(
        workspace,
        "segmented_compact_num_tiles",
        (1,),
        device=device,
        dtype=torch.int32,
        zero=True,
    )

    block_assignments = int(triton.next_power_of_2(max(1, assignments)))
    assignment_warps = 8 if block_assignments >= 2048 else 4
    _qwen3_moe_compact_route_pack_kernel[(int(num_experts),)](
        flat_experts,
        flat_route,
        num_tiles_gpu,
        sorted_tokens,
        sorted_route,
        sorted_slots,
        tile_experts,
        tile_starts,
        tile_lengths,
        ASSIGNMENTS=int(assignments),
        TOP_K=int(top_k),
        BLOCK_ASSIGNMENTS=block_assignments,
        BLOCK_M=int(block_m),
        MAX_TILES_PER_EXPERT=int(triton.cdiv(int(rows), int(block_m))),
        num_warps=assignment_warps,
        num_stages=1,
    )
    if workspace is not None:
        workspace["segmented_prefill_route_scatter"] = 0
        workspace["segmented_prefill_async_tiles"] = 0
        workspace["segmented_prefill_fixed_route_pack"] = 0
        workspace["segmented_prefill_compact_route_pack"] = 1
        workspace["segmented_prefill_compact_route_pack_passes"] = 1
        workspace["segmented_prefill_compact_route_capacity"] = int(assignments)
        workspace["segmented_prefill_max_tiles"] = int(max_tiles)
    return (
        sorted_tokens,
        sorted_route,
        sorted_slots,
        tile_experts,
        tile_starts,
        tile_lengths,
        num_tiles_gpu,
        int(max_tiles),
    )


def _route_assignments_by_expert_compact_pack_graph(
    flat_experts: torch.Tensor,
    flat_route: torch.Tensor,
    *,
    rows: int,
    top_k: int,
    num_experts: int,
    block_m: int,
    workspace: Optional[dict[str, torch.Tensor]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Graph-safe compact packing with deterministic tile offsets."""
    if workspace is None:
        raise ValueError("Graph compact route packing requires persistent workspace")

    assignments = int(flat_experts.numel())
    max_tiles = _segmented_tile_upper_bound(assignments, num_experts, block_m)
    device = flat_experts.device
    sorted_tokens = _workspace_tensor(
        workspace,
        "segmented_compact_sorted_tokens",
        (assignments,),
        device=device,
        dtype=torch.int64,
    )
    sorted_route = _workspace_tensor(
        workspace,
        "segmented_compact_sorted_route",
        (assignments,),
        device=device,
        dtype=flat_route.dtype,
    )
    sorted_slots = _workspace_tensor(
        workspace,
        "segmented_compact_sorted_slots",
        (assignments,),
        device=device,
        dtype=torch.int64,
    )
    tile_experts = _workspace_tensor(
        workspace,
        "segmented_compact_tile_experts",
        (max_tiles,),
        device=device,
        dtype=torch.int64,
    )
    tile_starts = _workspace_tensor(
        workspace,
        "segmented_compact_tile_starts",
        (max_tiles,),
        device=device,
        dtype=torch.int64,
    )
    tile_lengths = _workspace_tensor(
        workspace,
        "segmented_compact_tile_lengths",
        (max_tiles,),
        device=device,
        dtype=torch.int64,
    )
    counts = _workspace_tensor(
        workspace,
        "segmented_graph_route_counts",
        (num_experts,),
        device=device,
        dtype=torch.int32,
    )
    starts = _workspace_tensor(
        workspace,
        "segmented_graph_route_starts",
        (num_experts,),
        device=device,
        dtype=torch.int32,
    )
    tiles_per_expert = _workspace_tensor(
        workspace,
        "segmented_graph_route_tiles_per_expert",
        (num_experts,),
        device=device,
        dtype=torch.int32,
    )
    tile_offsets = _workspace_tensor(
        workspace,
        "segmented_graph_route_tile_offsets",
        (num_experts,),
        device=device,
        dtype=torch.int32,
    )
    num_tiles_gpu = _workspace_tensor(
        workspace,
        "segmented_compact_num_tiles",
        (1,),
        device=device,
        dtype=torch.int32,
    )

    block_assignments = int(triton.next_power_of_2(max(1, assignments)))
    assignment_warps = 8 if block_assignments >= 2048 else 4
    _qwen3_moe_compact_route_counts_kernel[(num_experts,)](
        flat_experts,
        counts,
        starts,
        tiles_per_expert,
        ASSIGNMENTS=assignments,
        BLOCK_ASSIGNMENTS=block_assignments,
        BLOCK_M=block_m,
        num_warps=assignment_warps,
        num_stages=1,
    )
    block_experts = int(triton.next_power_of_2(max(1, num_experts)))
    _qwen3_moe_compact_route_tile_prefix_kernel[(1,)](
        tiles_per_expert,
        tile_offsets,
        num_tiles_gpu,
        NUM_EXPERTS=num_experts,
        BLOCK_EXPERTS=block_experts,
        num_warps=4,
        num_stages=1,
    )
    _qwen3_moe_compact_route_scatter_kernel[(num_experts,)](
        flat_experts,
        flat_route,
        counts,
        starts,
        tiles_per_expert,
        tile_offsets,
        sorted_tokens,
        sorted_route,
        sorted_slots,
        tile_experts,
        tile_starts,
        tile_lengths,
        ASSIGNMENTS=assignments,
        TOP_K=top_k,
        BLOCK_ASSIGNMENTS=block_assignments,
        BLOCK_M=block_m,
        MAX_TILES=max_tiles,
        MAX_TILES_PER_EXPERT=int(triton.cdiv(rows, block_m)),
        num_warps=assignment_warps,
        num_stages=1,
    )
    workspace["segmented_prefill_route_scatter"] = 0
    workspace["segmented_prefill_async_tiles"] = 0
    workspace["segmented_prefill_fixed_route_pack"] = 0
    workspace["segmented_prefill_compact_route_pack"] = 1
    workspace["segmented_prefill_compact_route_pack_passes"] = 2
    workspace["segmented_prefill_graph_route_pack"] = 1
    workspace["segmented_prefill_compact_route_capacity"] = assignments
    workspace["segmented_prefill_max_tiles"] = max_tiles
    return (
        sorted_tokens,
        sorted_route,
        sorted_slots,
        tile_experts,
        tile_starts,
        tile_lengths,
        num_tiles_gpu,
        max_tiles,
    )


def _route_assignments_by_expert_argsort(
    flat_experts: torch.Tensor,
    flat_route: torch.Tensor,
    *,
    rows: int,
    top_k: int,
    workspace: Optional[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids = _workspace_token_ids(workspace, int(rows), int(top_k), device=flat_experts.device)
    order = torch.argsort(flat_experts)
    sorted_tokens = token_ids.index_select(0, order).contiguous()
    sorted_route = flat_route.index_select(0, order).contiguous()
    sorted_slots = order.to(torch.int64).contiguous()
    if workspace is not None:
        workspace["segmented_prefill_route_scatter"] = 0
    return sorted_tokens, sorted_route, sorted_slots


def _segmented_prefill_uses_partial_reduce(
    *,
    use_fused_gate: bool,
    assignments: int,
    deterministic_reduce: bool,
) -> bool:
    if not use_fused_gate:
        return False
    if deterministic_reduce:
        return True
    return bool(
        _CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE
        and int(assignments)
        <= _CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS
    )


def qwen3_moe_segmented_prefill(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    activation: str = "silu",
    out: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    workspace: Optional[dict[str, torch.Tensor]] = None,
    force: bool = False,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    fused_gate_block_n: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    fused_gate: Optional[bool] = None,
    dense_grid: Optional[bool] = None,
    route_scatter: Optional[bool] = None,
    fixed_route_pack: Optional[bool] = None,
    compact_route_pack: Optional[bool] = None,
    async_tiles_max_assignments: Optional[int] = None,
    single_accumulator: Optional[bool] = None,
    sorted_partial: Optional[bool] = None,
    group_size_m: Optional[int] = None,
    graph_safe: bool = False,
    deterministic_reduce: bool = False,
) -> torch.Tensor:
    """
    Segmented Qwen3 MoE prefill over sorted expert assignments.

    This avoids the padded expert buckets used by the PyTorch bmm fallback:
      hidden_states:    [M, H]
      gate_up_proj:     [E, 2I, H]
      down_proj:        [E, H, I]
      selected_experts: [M, K]
      routing_weights:  [M, K]
    """
    if workspace is not None:
        workspace["segmented_prefill_residual_fused"] = 0
        workspace["segmented_prefill_compact_route_pack"] = 0
        workspace["segmented_prefill_compact_route_pack_passes"] = 0
        workspace["segmented_prefill_graph_route_pack"] = 0
        workspace["segmented_prefill_graph_partial_cached"] = 0
        workspace["segmented_prefill_single_accumulator"] = 0
        workspace["segmented_prefill_sorted_partial"] = 0
        workspace["segmented_prefill_slot_inverse_bytes"] = 0
        workspace["segmented_prefill_group_size_m"] = 0
        workspace["segmented_prefill_partial_storage_requested"] = "torch.float32"
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be [M, H]")
    if selected_experts.shape != routing_weights.shape:
        raise ValueError("selected_experts and routing_weights must have the same shape")
    if selected_experts.ndim != 2:
        raise ValueError("selected_experts must be [M, top_k]")
    if residual is not None and tuple(residual.shape) != tuple(hidden_states.shape):
        raise ValueError("residual must have the same shape as hidden_states")

    rows, hidden_dim = hidden_states.shape
    if rows == 0:
        if residual is not None:
            final = out if out is not None else torch.empty_like(hidden_states)
            if final is not residual:
                final.copy_(residual)
            return final
        return torch.zeros_like(hidden_states) if out is None else out.zero_()
    top_k = int(selected_experts.shape[1])
    assignments = int(rows) * int(top_k)
    num_experts = int(gate_up_proj.shape[0])
    intermediate_dim = int(gate_up_proj.shape[1] // 2)
    act_id = _activation_id(activation)
    selected_block_m = int(_CFG_SEGMENTED_PREFILL_BLOCK_M if block_m is None else block_m)
    selected_block_n = int(_CFG_SEGMENTED_PREFILL_BLOCK_N if block_n is None else block_n)
    selected_block_k = int(_CFG_SEGMENTED_PREFILL_BLOCK_K if block_k is None else block_k)
    selected_fused_gate_block_n = int(
        _CFG_SEGMENTED_PREFILL_FUSED_GATE_BLOCK_N
        if fused_gate_block_n is None
        else fused_gate_block_n
    )
    selected_num_warps = int(_CFG_SEGMENTED_PREFILL_NUM_WARPS if num_warps is None else num_warps)
    selected_num_stages = int(_CFG_SEGMENTED_PREFILL_NUM_STAGES if num_stages is None else num_stages)
    selected_fused_gate = bool(
        _CFG_SEGMENTED_PREFILL_FUSED_GATE if fused_gate is None else fused_gate
    )
    selected_deterministic_reduce = bool(deterministic_reduce)
    if selected_deterministic_reduce:
        # The non-fused down path accumulates expert assignments with atomic_add.
        # Deterministic inference must use the per-assignment buffer followed by
        # the fixed top-k reduction, even above the normal memory/perf cutoff.
        selected_fused_gate = True
    selected_dense_grid = bool(
        _CFG_SEGMENTED_PREFILL_DENSE_GRID if dense_grid is None else dense_grid
    )
    selected_route_scatter = bool(
        _CFG_SEGMENTED_PREFILL_ROUTE_SCATTER if route_scatter is None else route_scatter
    )
    selected_fixed_route_pack = bool(
        _CFG_SEGMENTED_PREFILL_FIXED_ROUTE_PACK
        if fixed_route_pack is None
        else fixed_route_pack
    )
    selected_compact_route_pack = bool(
        _CFG_SEGMENTED_PREFILL_COMPACT_ROUTE_PACK
        if compact_route_pack is None
        else compact_route_pack
    )
    selected_single_accumulator = bool(
        _CFG_SEGMENTED_PREFILL_SINGLE_ACCUMULATOR
        if single_accumulator is None
        else single_accumulator
    )
    selected_sorted_partial = bool(
        _CFG_SEGMENTED_PREFILL_SORTED_PARTIAL
        if sorted_partial is None
        else sorted_partial
    )
    selected_group_size_m = max(
        1,
        int(
            _CFG_SEGMENTED_PREFILL_GROUP_SIZE_M
            if group_size_m is None
            else group_size_m
        ),
    )
    selected_async_tiles_max_assignments = (
        int(_CFG_SEGMENTED_PREFILL_ASYNC_TILES_MAX_ASSIGNMENTS)
        if async_tiles_max_assignments is None
        else max(1, int(async_tiles_max_assignments))
    )
    use_triton = (
        _HAS_TRITON
        and hidden_states.is_cuda
        and gate_up_proj.is_cuda
        and down_proj.is_cuda
        and selected_experts.is_cuda
        and routing_weights.is_cuda
        and not torch.is_grad_enabled()
        and hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and gate_up_proj.dtype == hidden_states.dtype
        and down_proj.dtype == hidden_states.dtype
        and qwen3_moe_segmented_prefers_triton_shape(
            int(rows),
            int(top_k),
            int(hidden_dim),
            int(intermediate_dim),
            force=force,
            block_m=selected_block_m,
            block_n=selected_block_n,
            block_k=selected_block_k,
        )
    )
    if not use_triton:
        raise RuntimeError("Qwen3 MoE segmented prefill is not available for this shape/device")

    hidden = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    gate_up_w = gate_up_proj if gate_up_proj.is_contiguous() else gate_up_proj.contiguous()
    down_w = down_proj if down_proj.is_contiguous() else down_proj.contiguous()
    flat_experts = selected_experts.reshape(-1).to(torch.int64).contiguous()
    flat_route = routing_weights.reshape(-1).contiguous()
    # Warmup and capture must select the same route pack and partial buffer.
    # Ambient CUDA capture state alone made the exact warmup run a different
    # kernel path from the graph it was supposed to validate.
    graph_capture_active = bool(
        graph_safe or _cuda_graph_capture_active(hidden)
    )
    if workspace is not None:
        workspace["segmented_prefill_graph_mode_requested"] = int(graph_safe)
    block_m = selected_block_m
    fixed_route_pack_active = bool(
        selected_fixed_route_pack
        and selected_fused_gate
        and not selected_dense_grid
        and int(rows) == 25
        and int(top_k) == 8
        and int(num_experts) == 128
        and int(hidden_dim) == 2816
        and int(intermediate_dim) == 704
        and hidden.dtype == torch.bfloat16
    )
    compact_route_pack_active = bool(
        selected_compact_route_pack
        and selected_fused_gate
        and not selected_dense_grid
        and int(rows) in (50, 100, 200, 400)
        and int(top_k) == 8
        and int(num_experts) == 128
        and int(hidden_dim) == 2816
        and int(intermediate_dim) == 704
        and hidden.dtype == torch.bfloat16
    )
    sorted_capacity = int(assignments)
    counts = None
    starts = None
    num_tiles_gpu = None
    max_num_tiles = 0
    if fixed_route_pack_active:
        (
            sorted_tokens,
            sorted_route,
            sorted_slots,
            tile_experts,
            tile_starts,
            tile_lengths,
            num_tiles_gpu,
            num_tiles,
        ) = _route_assignments_by_expert_fixed_pack(
            flat_experts,
            flat_route,
            rows=int(rows),
            top_k=int(top_k),
            num_experts=int(num_experts),
            block_m=int(block_m),
            workspace=workspace,
        )
        max_num_tiles = int(num_tiles)
        sorted_capacity = int(num_experts) * int(rows)
    elif compact_route_pack_active:
        compact_pack = (
            _route_assignments_by_expert_compact_pack_graph
            if graph_capture_active
            else _route_assignments_by_expert_compact_pack
        )
        (
            sorted_tokens,
            sorted_route,
            sorted_slots,
            tile_experts,
            tile_starts,
            tile_lengths,
            num_tiles_gpu,
            max_num_tiles,
        ) = compact_pack(
            flat_experts,
            flat_route,
            rows=int(rows),
            top_k=int(top_k),
            num_experts=int(num_experts),
            block_m=int(block_m),
            workspace=workspace,
        )
        num_tiles = int(max_num_tiles)
    else:
        if workspace is not None:
            workspace["segmented_prefill_fixed_route_pack"] = 0
            workspace["segmented_prefill_compact_route_pack"] = 0
        counts = torch.bincount(flat_experts, minlength=num_experts)
        starts = torch.cumsum(counts, dim=0) - counts
        if selected_route_scatter:
            try:
                sorted_tokens, sorted_route, sorted_slots = _route_assignments_by_expert_scatter(
                    flat_experts,
                    flat_route,
                    starts,
                    rows=int(rows),
                    top_k=int(top_k),
                    num_experts=int(num_experts),
                    workspace=workspace,
                )
            except Exception as exc:
                if workspace is not None:
                    workspace["segmented_prefill_route_scatter_fail_reason"] = str(exc)
                sorted_tokens, sorted_route, sorted_slots = _route_assignments_by_expert_argsort(
                    flat_experts,
                    flat_route,
                    rows=int(rows),
                    top_k=int(top_k),
                    workspace=workspace,
                )
        else:
            sorted_tokens, sorted_route, sorted_slots = _route_assignments_by_expert_argsort(
                flat_experts,
                flat_route,
                rows=int(rows),
                top_k=int(top_k),
                workspace=workspace,
            )

    block_n = min(selected_block_n, int(triton.next_power_of_2(max(hidden_dim, 1))))
    block_k = min(selected_block_k, int(triton.next_power_of_2(max(hidden_dim, intermediate_dim, 1))))
    async_compact_tiles = bool(
        _CFG_SEGMENTED_PREFILL_ASYNC_TILES
        and selected_fused_gate
        and not selected_dense_grid
        and not fixed_route_pack_active
        and not compact_route_pack_active
        and assignments <= selected_async_tiles_max_assignments
    )
    if fixed_route_pack_active or compact_route_pack_active:
        pass
    elif selected_dense_grid:
        max_count = int(counts.max().item())
        if max_count <= 0:
            return torch.zeros_like(hidden_states) if out is None else out.zero_()
        num_m_tiles = triton.cdiv(int(max_count), int(block_m))
        num_tiles = int(num_experts) * int(num_m_tiles)
    elif async_compact_tiles:
        (
            tile_experts,
            tile_starts,
            tile_lengths,
            num_tiles_gpu,
            max_num_tiles,
        ) = _build_segmented_tile_tensors_gpu_async(
            counts,
            starts,
            assignments=int(assignments),
            block_m=block_m,
            workspace=workspace,
        )
        num_tiles = int(max_num_tiles)
    else:
        tile_experts, tile_starts, tile_lengths = _build_segmented_tile_tensors_gpu(
            counts,
            block_m=block_m,
            workspace=workspace,
        )
        num_tiles = int(tile_experts.numel())
        if num_tiles <= 0:
            if residual is not None:
                final = out if out is not None else torch.empty_like(hidden_states)
                if final is not residual:
                    final.copy_(residual)
                return final
            return torch.zeros_like(hidden_states) if out is None else out.zero_()

    if workspace is not None:
        workspace["segmented_prefill_last_tiles"] = (
            0 if async_compact_tiles else int(num_tiles)
        )
        workspace["segmented_prefill_last_assignments"] = int(assignments)
        workspace["segmented_prefill_dense_grid"] = int(selected_dense_grid)
        workspace["segmented_prefill_fused_gate"] = int(
            selected_fused_gate and not selected_dense_grid
        )
        workspace["segmented_prefill_fixed_route_pack"] = int(
            fixed_route_pack_active
        )
        workspace["segmented_prefill_compact_route_pack"] = int(
            compact_route_pack_active
        )
        workspace["segmented_prefill_async_tiles_max_assignments"] = int(
            selected_async_tiles_max_assignments
        )
        workspace["segmented_prefill_selected_block_m"] = int(selected_block_m)
        workspace["segmented_prefill_selected_block_n"] = int(selected_block_n)
        workspace["segmented_prefill_selected_block_k"] = int(selected_block_k)
        workspace["segmented_prefill_selected_fused_gate_block_n"] = int(
            selected_fused_gate_block_n
        )
        workspace["segmented_prefill_selected_num_warps"] = int(selected_num_warps)
        workspace["segmented_prefill_selected_num_stages"] = int(selected_num_stages)

    use_fused_gate = selected_fused_gate and not selected_dense_grid
    use_partial_reduce = _segmented_prefill_uses_partial_reduce(
        use_fused_gate=use_fused_gate,
        assignments=assignments,
        deterministic_reduce=selected_deterministic_reduce,
    )
    if workspace is not None:
        workspace["segmented_prefill_partial_reduce"] = int(use_partial_reduce)
        workspace["segmented_prefill_deterministic_reduce"] = int(
            selected_deterministic_reduce and use_partial_reduce
        )
        workspace["segmented_prefill_atomic_reduce"] = int(not use_partial_reduce)
        workspace["segmented_prefill_partial_reduce_max_assignments"] = int(
            _CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS
        )
    if use_fused_gate and num_tiles_gpu is None:
        if workspace is None:
            num_tiles_gpu = torch.tensor(
                [int(num_tiles)],
                device=hidden.device,
                dtype=torch.int64,
            )
        else:
            num_tiles_gpu = _workspace_tensor(
                workspace,
                "segmented_num_tiles",
                (1,),
                device=hidden.device,
                dtype=torch.int64,
            )
            num_tiles_gpu.fill_(int(num_tiles))
    accum = None
    if not use_partial_reduce:
        accum = torch.zeros(
            (rows, hidden_dim),
            device=hidden.device,
            dtype=torch.float32,
        )

    down_block_n = min(block_n, int(triton.next_power_of_2(hidden_dim)))
    inter_block_k = min(block_k, int(triton.next_power_of_2(intermediate_dim)))

    hidden_block_k = min(block_k, int(triton.next_power_of_2(hidden_dim)))
    use_single_accumulator = bool(
        selected_single_accumulator
        and use_fused_gate
        and use_partial_reduce
        and not selected_dense_grid
    )
    sorted_partial_active = bool(
        selected_sorted_partial
        and use_fused_gate
        and use_partial_reduce
        and not selected_dense_grid
        and int(sorted_capacity) == int(assignments)
    )
    slot_to_sorted = sorted_slots
    if sorted_partial_active:
        slot_to_sorted = _workspace_tensor(
            workspace,
            "segmented_slot_to_sorted",
            (assignments,),
            device=hidden.device,
            dtype=torch.int64,
        )
        inverse_block = 256
        _qwen3_moe_invert_sorted_slots_kernel[
            (triton.cdiv(assignments, inverse_block),)
        ](
            sorted_slots,
            slot_to_sorted,
            ASSIGNMENTS=int(assignments),
            BLOCK=inverse_block,
            num_warps=4,
            num_stages=1,
        )
    if workspace is not None:
        workspace["segmented_prefill_single_accumulator"] = int(
            use_single_accumulator
        )
        workspace["segmented_prefill_sorted_partial"] = int(
            sorted_partial_active
        )
        workspace["segmented_prefill_slot_inverse_bytes"] = int(
            slot_to_sorted.numel() * slot_to_sorted.element_size()
            if sorted_partial_active
            else 0
        )
        workspace["segmented_prefill_group_size_m"] = int(
            selected_group_size_m
        )

    if use_single_accumulator:
        gate_up = torch.empty(
            (sorted_capacity, 2 * intermediate_dim),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        single_gate_block_n = min(
            selected_fused_gate_block_n,
            int(triton.next_power_of_2(2 * intermediate_dim)),
        )
        gate_grid = int(num_tiles) * int(
            triton.cdiv(2 * intermediate_dim, single_gate_block_n)
        )
        grouped_size_m = min(int(selected_group_size_m), int(num_tiles))
        _qwen3_moe_segmented_gate_up_single_accum_kernel[(gate_grid,)](
            hidden,
            gate_up_w,
            sorted_tokens,
            tile_experts,
            tile_starts,
            tile_lengths,
            num_tiles_gpu,
            gate_up,
            hidden.stride(0),
            hidden.stride(1),
            gate_up_w.stride(0),
            gate_up_w.stride(1),
            gate_up_w.stride(2),
            gate_up.stride(0),
            gate_up.stride(1),
            int(hidden_dim),
            int(2 * intermediate_dim),
            BLOCK_M=block_m,
            BLOCK_N=single_gate_block_n,
            BLOCK_K=hidden_block_k,
            MAX_TILES=int(num_tiles),
            GROUP_SIZE_M=int(grouped_size_m),
            num_warps=selected_num_warps,
            num_stages=selected_num_stages,
        )

        if graph_capture_active:
            partial = qwen3_moe_prepare_segmented_prefill_graph_workspace(
                workspace,
                assignments=assignments,
                hidden_dim=hidden_dim,
                device=hidden.device,
            )
            cache_partial = True
        else:
            cache_partial = bool(
                workspace is not None
                and assignments
                <= _CFG_SEGMENTED_PREFILL_PARTIAL_CACHE_MAX_ASSIGNMENTS
            )
            partial = _workspace_tensor(
                workspace if cache_partial else None,
                "segmented_partial_out",
                (assignments, hidden_dim),
                device=hidden.device,
                dtype=torch.float32,
            )
        if workspace is not None:
            workspace["segmented_prefill_partial_cached"] = int(cache_partial)
            workspace["segmented_prefill_graph_partial_cached"] = int(
                graph_capture_active
            )
            workspace["segmented_prefill_partial_dtype"] = str(partial.dtype)
            workspace["segmented_prefill_partial_bytes"] = int(
                partial.numel() * partial.element_size()
            )

        down_grid = int(num_tiles) * int(
            triton.cdiv(hidden_dim, down_block_n)
        )
        _qwen3_moe_segmented_down_single_accum_partial_kernel[(down_grid,)](
            gate_up,
            down_w,
            sorted_route,
            sorted_slots,
            tile_experts,
            tile_starts,
            tile_lengths,
            num_tiles_gpu,
            partial,
            gate_up.stride(0),
            gate_up.stride(1),
            down_w.stride(0),
            down_w.stride(1),
            down_w.stride(2),
            partial.stride(0),
            partial.stride(1),
            int(intermediate_dim),
            int(hidden_dim),
            ACT=act_id,
            BLOCK_M=block_m,
            BLOCK_N=down_block_n,
            BLOCK_K=inter_block_k,
            MAX_TILES=int(num_tiles),
            GROUP_SIZE_M=int(grouped_size_m),
            SORTED_PARTIAL=sorted_partial_active,
            num_warps=selected_num_warps,
            num_stages=selected_num_stages,
        )

        final = out if out is not None else torch.empty_like(hidden_states)
        reduce_block_n = min(256, int(triton.next_power_of_2(hidden_dim)))
        residual_ptr = residual if residual is not None else hidden
        _qwen3_moe_segmented_partial_reduce_kernel[
            (int(rows), triton.cdiv(hidden_dim, reduce_block_n))
        ](
            partial,
            slot_to_sorted,
            residual_ptr,
            final,
            partial.stride(0),
            partial.stride(1),
            residual_ptr.stride(0),
            residual_ptr.stride(1),
            final.stride(0),
            final.stride(1),
            int(hidden_dim),
            TOP_K=int(top_k),
            ADD_RESIDUAL=bool(residual is not None),
            BLOCK_N=reduce_block_n,
            SORTED_PARTIAL=sorted_partial_active,
            num_warps=4,
            num_stages=1,
        )
        if workspace is not None:
            workspace["segmented_prefill_residual_fused"] = int(
                residual is not None
            )
        return final

    if use_fused_gate:
        act = torch.empty(
            (sorted_capacity, intermediate_dim),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        fused_gate_block_n = min(
            selected_fused_gate_block_n,
            int(triton.next_power_of_2(intermediate_dim)),
        )
        gate_grid_n = int(triton.cdiv(intermediate_dim, fused_gate_block_n))
        _qwen3_moe_segmented_gate_swiglu_kernel[(int(num_tiles), gate_grid_n)](
            hidden,
            gate_up_w,
            sorted_tokens,
            tile_experts,
            tile_starts,
            tile_lengths,
            num_tiles_gpu,
            act,
            hidden.stride(0),
            hidden.stride(1),
            gate_up_w.stride(0),
            gate_up_w.stride(1),
            gate_up_w.stride(2),
            act.stride(0),
            act.stride(1),
            int(hidden_dim),
            int(intermediate_dim),
            ACT=act_id,
            BLOCK_M=block_m,
            BLOCK_N=fused_gate_block_n,
            BLOCK_K=hidden_block_k,
            MAX_TILES=int(num_tiles),
            num_warps=selected_num_warps,
            num_stages=selected_num_stages,
        )
        if use_partial_reduce:
            if graph_capture_active:
                partial = qwen3_moe_prepare_segmented_prefill_graph_workspace(
                    workspace,
                    assignments=assignments,
                    hidden_dim=hidden_dim,
                    device=hidden.device,
                )
                cache_partial = True
            else:
                cache_partial = bool(
                    workspace is not None
                    and assignments
                    <= _CFG_SEGMENTED_PREFILL_PARTIAL_CACHE_MAX_ASSIGNMENTS
                )
                partial = _workspace_tensor(
                    workspace if cache_partial else None,
                    "segmented_partial_out",
                    (assignments, hidden_dim),
                    device=hidden.device,
                    dtype=torch.float32,
                )
            if workspace is not None:
                workspace["segmented_prefill_partial_cached"] = int(cache_partial)
                workspace["segmented_prefill_graph_partial_cached"] = int(
                    graph_capture_active
                )
                workspace["segmented_prefill_partial_dtype"] = str(partial.dtype)
                workspace["segmented_prefill_partial_bytes"] = int(
                    partial.numel() * partial.element_size()
                )
            down_grid_n = int(triton.cdiv(hidden_dim, down_block_n))
            _qwen3_moe_segmented_down_from_act_partial_kernel[
                (int(num_tiles), down_grid_n)
            ](
                act,
                down_w,
                sorted_route,
                sorted_slots,
                tile_experts,
                tile_starts,
                tile_lengths,
                num_tiles_gpu,
                partial,
                act.stride(0),
                act.stride(1),
                down_w.stride(0),
                down_w.stride(1),
                down_w.stride(2),
                partial.stride(0),
                partial.stride(1),
                int(intermediate_dim),
                int(hidden_dim),
                BLOCK_M=block_m,
                BLOCK_N=down_block_n,
                BLOCK_K=inter_block_k,
                MAX_TILES=int(num_tiles),
                SORTED_PARTIAL=sorted_partial_active,
                num_warps=selected_num_warps,
                num_stages=selected_num_stages,
            )
            final = out if out is not None else torch.empty_like(hidden_states)
            reduce_block_n = min(256, int(triton.next_power_of_2(hidden_dim)))
            residual_ptr = residual if residual is not None else hidden
            _qwen3_moe_segmented_partial_reduce_kernel[
                (int(rows), triton.cdiv(hidden_dim, reduce_block_n))
            ](
                partial,
                slot_to_sorted,
                residual_ptr,
                final,
                partial.stride(0),
                partial.stride(1),
                residual_ptr.stride(0),
                residual_ptr.stride(1),
                final.stride(0),
                final.stride(1),
                int(hidden_dim),
                TOP_K=int(top_k),
                ADD_RESIDUAL=bool(residual is not None),
                BLOCK_N=reduce_block_n,
                SORTED_PARTIAL=sorted_partial_active,
                num_warps=4,
                num_stages=1,
            )
            if workspace is not None:
                workspace["segmented_prefill_residual_fused"] = int(residual is not None)
            return final
        down_grid_n = int(triton.cdiv(hidden_dim, down_block_n))
        _qwen3_moe_segmented_down_from_act_accum_kernel[
            (int(num_tiles), down_grid_n)
        ](
            act,
            down_w,
            sorted_tokens,
            sorted_route,
            tile_experts,
            tile_starts,
            tile_lengths,
            num_tiles_gpu,
            accum,
            act.stride(0),
            act.stride(1),
            down_w.stride(0),
            down_w.stride(1),
            down_w.stride(2),
            accum.stride(0),
            accum.stride(1),
            int(intermediate_dim),
            int(hidden_dim),
            BLOCK_M=block_m,
            BLOCK_N=down_block_n,
            BLOCK_K=inter_block_k,
            MAX_TILES=int(num_tiles),
            num_warps=selected_num_warps,
            num_stages=selected_num_stages,
        )
    else:
        gate_up = torch.empty(
            (assignments, 2 * intermediate_dim),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        gate_block_n = min(block_n, int(triton.next_power_of_2(2 * intermediate_dim)))
        if selected_dense_grid:
            _qwen3_moe_segmented_grid_gate_up_kernel[
                (int(num_experts), int(num_m_tiles), triton.cdiv(2 * intermediate_dim, gate_block_n))
            ](
                hidden,
                gate_up_w,
                sorted_tokens,
                counts,
                starts,
                gate_up,
                hidden.stride(0),
                hidden.stride(1),
                gate_up_w.stride(0),
                gate_up_w.stride(1),
                gate_up_w.stride(2),
                gate_up.stride(0),
                gate_up.stride(1),
                int(hidden_dim),
                int(2 * intermediate_dim),
                BLOCK_M=block_m,
                BLOCK_N=gate_block_n,
                BLOCK_K=hidden_block_k,
                num_warps=selected_num_warps,
                num_stages=selected_num_stages,
            )
            _qwen3_moe_segmented_grid_down_accum_kernel[
                (int(num_experts), int(num_m_tiles), triton.cdiv(hidden_dim, down_block_n))
            ](
                gate_up,
                down_w,
                sorted_tokens,
                sorted_route,
                counts,
                starts,
                accum,
                gate_up.stride(0),
                gate_up.stride(1),
                down_w.stride(0),
                down_w.stride(1),
                down_w.stride(2),
                accum.stride(0),
                accum.stride(1),
                int(intermediate_dim),
                int(hidden_dim),
                ACT=act_id,
                BLOCK_M=block_m,
                BLOCK_N=down_block_n,
                BLOCK_K=inter_block_k,
                num_warps=selected_num_warps,
                num_stages=selected_num_stages,
            )
        else:
            _qwen3_moe_segmented_gate_up_kernel[
                (num_tiles, triton.cdiv(2 * intermediate_dim, gate_block_n))
            ](
                hidden,
                gate_up_w,
                sorted_tokens,
                tile_experts,
                tile_starts,
                tile_lengths,
                gate_up,
                hidden.stride(0),
                hidden.stride(1),
                gate_up_w.stride(0),
                gate_up_w.stride(1),
                gate_up_w.stride(2),
                gate_up.stride(0),
                gate_up.stride(1),
                int(hidden_dim),
                int(2 * intermediate_dim),
                BLOCK_M=block_m,
                BLOCK_N=gate_block_n,
                BLOCK_K=hidden_block_k,
                num_warps=selected_num_warps,
                num_stages=selected_num_stages,
            )
            _qwen3_moe_segmented_down_accum_kernel[
                (num_tiles, triton.cdiv(hidden_dim, down_block_n))
            ](
                gate_up,
                down_w,
                sorted_tokens,
                sorted_route,
                tile_experts,
                tile_starts,
                tile_lengths,
                accum,
                gate_up.stride(0),
                gate_up.stride(1),
                down_w.stride(0),
                down_w.stride(1),
                down_w.stride(2),
                accum.stride(0),
                accum.stride(1),
                int(intermediate_dim),
                int(hidden_dim),
                ACT=act_id,
                BLOCK_M=block_m,
                BLOCK_N=down_block_n,
                BLOCK_K=inter_block_k,
                num_warps=selected_num_warps,
                num_stages=selected_num_stages,
            )

    return _copy_accum_with_optional_residual(
        accum,
        hidden_states,
        out,
        residual,
        workspace,
        workspace_prefix="segmented_prefill",
    )


def qwen3_moe_padded_bmm_prefill(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    activation: str = "silu",
    out: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    workspace: Optional[dict[str, torch.Tensor]] = None,
    down_output_dtype: str = "bf16",
    align_m: int = 16,
    route_pack: str = "argsort",
    route_pack_block: int = 256,
    fused_activation: bool = False,
    activation_block: int = 256,
    reduce_block_n: int = 256,
    reduce_num_warps: int = 4,
    max_padding_ratio: Optional[float] = None,
) -> torch.Tensor:
    """Long-prefill candidate using two padded strided-batched GEMMs.

    The expert-major batch is padded only to the largest routed expert. The
    final Triton kernel gathers each original top-k slot and reduces it in a
    fixed order, avoiding CUDA ``index_add_`` and a full assignment-by-hidden
    partial buffer.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Padded-BMM MoE prefill requires Triton")
    if torch.is_grad_enabled():
        raise RuntimeError("Padded-BMM MoE prefill is inference-only")
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be [M, H]")
    if selected_experts.ndim != 2:
        raise ValueError("selected_experts must be [M, top_k]")
    if selected_experts.shape != routing_weights.shape:
        raise ValueError("selected_experts and routing_weights must have the same shape")
    if int(selected_experts.shape[0]) != int(hidden_states.shape[0]):
        raise ValueError("selected_experts rows must match hidden_states")
    if residual is not None and tuple(residual.shape) != tuple(hidden_states.shape):
        raise ValueError("residual must have the same shape as hidden_states")
    if out is not None and tuple(out.shape) != tuple(hidden_states.shape):
        raise ValueError("out must have the same shape as hidden_states")
    if not all(
        tensor.is_cuda
        for tensor in (
            hidden_states,
            gate_up_proj,
            down_proj,
            selected_experts,
            routing_weights,
        )
    ):
        raise RuntimeError("Padded-BMM MoE prefill requires CUDA tensors")
    if _cuda_graph_capture_active(hidden_states):
        raise RuntimeError("Padded-BMM MoE prefill is not CUDA-graph safe")

    rows, hidden_dim = hidden_states.shape
    top_k = int(selected_experts.shape[1])
    num_experts = int(gate_up_proj.shape[0])
    intermediate_dim = int(gate_up_proj.shape[1] // 2)
    assignments = int(rows) * int(top_k)
    if int(rows) == 0:
        final = out if out is not None else torch.empty_like(hidden_states)
        if residual is None:
            return final.zero_()
        if final is not residual:
            final.copy_(residual)
        return final
    if top_k <= 0 or num_experts <= 0 or intermediate_dim <= 0:
        raise ValueError("top_k, num_experts, and intermediate_dim must be positive")
    if tuple(gate_up_proj.shape[1:]) != (2 * intermediate_dim, int(hidden_dim)):
        raise ValueError("gate_up_proj must be [E, 2I, H]")
    if tuple(down_proj.shape) != (num_experts, int(hidden_dim), intermediate_dim):
        raise ValueError("down_proj must be [E, H, I]")
    if gate_up_proj.dtype != hidden_states.dtype or down_proj.dtype != hidden_states.dtype:
        raise ValueError("expert and hidden dtypes must match")
    if down_output_dtype not in ("bf16", "fp32"):
        raise ValueError("down_output_dtype must be 'bf16' or 'fp32'")
    if route_pack not in ("argsort", "atomic"):
        raise ValueError("route_pack must be 'argsort' or 'atomic'")
    if int(route_pack_block) not in (64, 128, 256, 512, 1024):
        raise ValueError("route_pack_block must be one of 64, 128, 256, 512, 1024")
    if int(activation_block) not in (128, 256, 512, 1024):
        raise ValueError("activation_block must be one of 128, 256, 512, 1024")
    if int(reduce_block_n) not in (64, 128, 256):
        raise ValueError("reduce_block_n must be one of 64, 128, 256")
    if int(reduce_num_warps) not in (4, 8):
        raise ValueError("reduce_num_warps must be 4 or 8")

    hidden = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    gate_up_w = gate_up_proj if gate_up_proj.is_contiguous() else gate_up_proj.contiguous()
    down_w = down_proj if down_proj.is_contiguous() else down_proj.contiguous()
    flat_experts = selected_experts.reshape(-1).to(torch.int64).contiguous()
    flat_route = routing_weights.reshape(-1).contiguous()

    counts = torch.bincount(flat_experts, minlength=num_experts)
    max_count = int(counts.max().item())
    if max_count <= 0:
        final = out if out is not None else torch.empty_like(hidden_states)
        if residual is None:
            return final.zero_()
        if final is not residual:
            final.copy_(residual)
        return final
    align_m = max(1, int(align_m))
    padded_count = ((max_count + align_m - 1) // align_m) * align_m
    padded_size = int(num_experts) * int(padded_count)
    padding_ratio = float(padded_size) / float(assignments)
    if max_padding_ratio is not None and padding_ratio > float(max_padding_ratio):
        raise RuntimeError(
            "Padded-BMM capacity guard: expert skew would expand "
            f"{assignments} assignments to {padded_size} padded rows "
            f"({padding_ratio:.3f}x > {float(max_padding_ratio):.3f}x)"
        )

    token_pad = torch.zeros(
        padded_size,
        device=hidden.device,
        dtype=torch.int64,
    )
    slot_to_padded = torch.empty(
        assignments,
        device=hidden.device,
        dtype=torch.int64,
    )
    if route_pack == "atomic":
        counters = torch.zeros(
            num_experts,
            device=hidden.device,
            dtype=torch.int32,
        )
        block = int(route_pack_block)
        _qwen3_moe_padded_bmm_atomic_route_pack_kernel[
            (triton.cdiv(assignments, block),)
        ](
            flat_experts,
            counters,
            token_pad,
            slot_to_padded,
            PADDED_COUNT=int(padded_count),
            TOP_K=int(top_k),
            ASSIGNMENTS=int(assignments),
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )
    else:
        order = torch.argsort(flat_experts, stable=True)
        sorted_experts = flat_experts.index_select(0, order)
        starts = torch.cumsum(counts, dim=0) - counts
        sorted_positions = torch.arange(
            assignments,
            device=hidden.device,
            dtype=torch.int64,
        )
        ranks = sorted_positions - starts.index_select(0, sorted_experts)
        padded_offsets = sorted_experts * int(padded_count) + ranks
        token_pad.scatter_(0, padded_offsets, order // int(top_k))
        slot_to_padded.scatter_(0, order, padded_offsets)

    padded_hidden = hidden.index_select(0, token_pad).reshape(
        num_experts,
        padded_count,
        int(hidden_dim),
    )
    gate_up = torch.bmm(padded_hidden, gate_up_w.transpose(1, 2))
    del padded_hidden
    if fused_activation:
        activated = torch.empty(
            (num_experts, padded_count, intermediate_dim),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        activation_elements = num_experts * padded_count * intermediate_dim
        block = int(activation_block)
        _qwen3_moe_padded_bmm_activation_kernel[
            (triton.cdiv(activation_elements, block),)
        ](
            gate_up,
            activated,
            ELEMENTS=activation_elements,
            I=intermediate_dim,
            ACT=_activation_id(activation),
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )
    else:
        gate, up = gate_up.chunk(2, dim=-1)
        if _activation_id(activation) == _ACT_GELU_TANH:
            activated = F.gelu(gate, approximate="tanh") * up
        else:
            activated = F.silu(gate) * up
        del gate, up
    del gate_up

    if down_output_dtype == "fp32":
        try:
            projected = torch.bmm(
                activated,
                down_w.transpose(1, 2),
                out_dtype=torch.float32,
            )
        except TypeError as exc:
            raise RuntimeError(
                "This PyTorch build does not support BF16 bmm with FP32 output"
            ) from exc
    else:
        projected = torch.bmm(activated, down_w.transpose(1, 2))
    del activated

    projected_2d = projected.reshape(padded_size, int(hidden_dim))
    final = out if out is not None else torch.empty_like(hidden_states)
    residual_ptr = residual if residual is not None else hidden
    block_n = min(
        int(reduce_block_n),
        int(triton.next_power_of_2(max(1, int(hidden_dim)))),
    )
    block_top_k = int(triton.next_power_of_2(max(1, int(top_k))))
    _qwen3_moe_padded_bmm_reduce_kernel[
        (int(rows), triton.cdiv(int(hidden_dim), block_n))
    ](
        projected_2d,
        slot_to_padded,
        flat_route,
        residual_ptr,
        final,
        projected_2d.stride(0),
        projected_2d.stride(1),
        residual_ptr.stride(0),
        residual_ptr.stride(1),
        final.stride(0),
        final.stride(1),
        int(hidden_dim),
        TOP_K=int(top_k),
        BLOCK_TOP_K=block_top_k,
        ADD_RESIDUAL=bool(residual is not None),
        BLOCK_N=block_n,
        num_warps=int(reduce_num_warps),
        num_stages=1,
    )
    if workspace is not None:
        workspace["padded_bmm_prefill"] = 1
        workspace["padded_bmm_deterministic_reduce"] = 1
        workspace["padded_bmm_max_count"] = int(max_count)
        workspace["padded_bmm_padded_count"] = int(padded_count)
        workspace["padded_bmm_padded_assignments"] = int(padded_size)
        workspace["padded_bmm_down_output_dtype"] = str(projected.dtype)
        workspace["padded_bmm_route_pack"] = str(route_pack)
        workspace["padded_bmm_route_pack_block"] = int(route_pack_block)
        workspace["padded_bmm_fused_activation"] = int(bool(fused_activation))
        workspace["padded_bmm_activation_block"] = int(activation_block)
        workspace["padded_bmm_reduce_block_n"] = int(reduce_block_n)
        workspace["padded_bmm_reduce_num_warps"] = int(reduce_num_warps)
        workspace["padded_bmm_padding_ratio"] = float(padding_ratio)
        workspace["padded_bmm_max_padding_ratio"] = (
            None if max_padding_ratio is None else float(max_padding_ratio)
        )
    return final


def qwen3_moe_dominant_expert_padded_bmm_prefill(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    activation: str = "silu",
    out: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    workspace: Optional[dict[str, torch.Tensor]] = None,
    align_m: int = 16,
    route_pack_block: int = 256,
    activation_block: int = 512,
    reduce_block_n: int = 256,
    reduce_num_warps: int = 4,
    minimum_dominant_skew: float = 4.0,
    max_light_padding_ratio: float = 1.25,
) -> torch.Tensor:
    """Route one dominant expert through dense GEMMs and pad only the rest.

    Global padded BMM is wasteful when one expert is much hotter than the
    remaining experts. This candidate removes that expert from the padded
    capacity calculation, computes it with two dense GEMMs, and keeps the
    fixed-order top-k reduction used by the established padded-BMM path.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Dominant-expert padded-BMM prefill requires Triton")
    if torch.is_grad_enabled():
        raise RuntimeError("Dominant-expert padded-BMM prefill is inference-only")
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be [M, H]")
    if selected_experts.ndim != 2:
        raise ValueError("selected_experts must be [M, top_k]")
    if selected_experts.shape != routing_weights.shape:
        raise ValueError("selected_experts and routing_weights must have the same shape")
    if int(selected_experts.shape[0]) != int(hidden_states.shape[0]):
        raise ValueError("selected_experts rows must match hidden_states")
    if residual is not None and tuple(residual.shape) != tuple(hidden_states.shape):
        raise ValueError("residual must have the same shape as hidden_states")
    if out is not None and tuple(out.shape) != tuple(hidden_states.shape):
        raise ValueError("out must have the same shape as hidden_states")
    if not all(
        tensor.is_cuda
        for tensor in (
            hidden_states,
            gate_up_proj,
            down_proj,
            selected_experts,
            routing_weights,
        )
    ):
        raise RuntimeError("Dominant-expert padded-BMM prefill requires CUDA tensors")
    if _cuda_graph_capture_active(hidden_states):
        raise RuntimeError(
            "Dominant-expert padded-BMM prefill is not CUDA-graph safe"
        )

    rows, hidden_dim = hidden_states.shape
    top_k = int(selected_experts.shape[1])
    num_experts = int(gate_up_proj.shape[0])
    intermediate_dim = int(gate_up_proj.shape[1] // 2)
    assignments = int(rows) * int(top_k)
    if int(rows) == 0:
        final = out if out is not None else torch.empty_like(hidden_states)
        if residual is None:
            return final.zero_()
        if final is not residual:
            final.copy_(residual)
        return final
    if top_k <= 0 or num_experts <= 1 or intermediate_dim <= 0:
        raise ValueError(
            "top_k and intermediate_dim must be positive and num_experts must exceed one"
        )
    if tuple(gate_up_proj.shape[1:]) != (2 * intermediate_dim, int(hidden_dim)):
        raise ValueError("gate_up_proj must be [E, 2I, H]")
    if tuple(down_proj.shape) != (num_experts, int(hidden_dim), intermediate_dim):
        raise ValueError("down_proj must be [E, H, I]")
    if gate_up_proj.dtype != hidden_states.dtype or down_proj.dtype != hidden_states.dtype:
        raise ValueError("expert and hidden dtypes must match")
    if int(route_pack_block) not in (64, 128, 256, 512, 1024):
        raise ValueError("route_pack_block must be one of 64, 128, 256, 512, 1024")
    if int(activation_block) not in (128, 256, 512, 1024):
        raise ValueError("activation_block must be one of 128, 256, 512, 1024")
    if int(reduce_block_n) not in (64, 128, 256):
        raise ValueError("reduce_block_n must be one of 64, 128, 256")
    if int(reduce_num_warps) not in (4, 8):
        raise ValueError("reduce_num_warps must be 4 or 8")
    if float(minimum_dominant_skew) <= 1.0:
        raise ValueError("minimum_dominant_skew must exceed 1.0")
    if float(max_light_padding_ratio) < 1.0:
        raise ValueError("max_light_padding_ratio must be at least 1.0")

    hidden = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    gate_up_w = gate_up_proj if gate_up_proj.is_contiguous() else gate_up_proj.contiguous()
    down_w = down_proj if down_proj.is_contiguous() else down_proj.contiguous()
    flat_experts = selected_experts.reshape(-1).to(torch.int64).contiguous()
    flat_route = routing_weights.reshape(-1).contiguous()

    counts = torch.bincount(flat_experts, minlength=num_experts)
    heavy_count_tensor, heavy_expert_tensor = counts.max(dim=0)
    heavy_count = int(heavy_count_tensor.item())
    heavy_expert = int(heavy_expert_tensor.item())
    average_count = float(assignments) / float(num_experts)
    dominant_skew = float(heavy_count) / average_count
    if dominant_skew < float(minimum_dominant_skew):
        raise RuntimeError(
            "Dominant-expert guard: hottest expert is only "
            f"{dominant_skew:.3f}x the mean, below "
            f"{float(minimum_dominant_skew):.3f}x"
        )

    light_counts = counts.clone()
    light_counts[heavy_expert] = 0
    light_assignments = int(assignments) - int(heavy_count)
    max_light_count = int(light_counts.max().item())
    if light_assignments <= 0 or max_light_count <= 0:
        raise RuntimeError("Dominant-expert path requires non-heavy assignments")
    align_m = max(1, int(align_m))
    light_padded_count = (
        (max_light_count + align_m - 1) // align_m
    ) * align_m
    light_padded_size = int(num_experts) * int(light_padded_count)
    light_padding_ratio = float(light_padded_size) / float(light_assignments)
    if light_padding_ratio > float(max_light_padding_ratio):
        raise RuntimeError(
            "Dominant-expert light-capacity guard: remaining experts expand "
            f"{light_assignments} assignments to {light_padded_size} rows "
            f"({light_padding_ratio:.3f}x > "
            f"{float(max_light_padding_ratio):.3f}x)"
        )

    light_token_pad = torch.zeros(
        light_padded_size,
        device=hidden.device,
        dtype=torch.int64,
    )
    heavy_token_ids = torch.empty(
        heavy_count,
        device=hidden.device,
        dtype=torch.int64,
    )
    slot_to_light = torch.zeros(
        assignments,
        device=hidden.device,
        dtype=torch.int64,
    )
    slot_to_heavy = torch.zeros_like(slot_to_light)
    light_counters = torch.zeros(
        num_experts,
        device=hidden.device,
        dtype=torch.int32,
    )
    heavy_counter = torch.zeros(1, device=hidden.device, dtype=torch.int32)
    pack_block = int(route_pack_block)
    _qwen3_moe_dominant_padded_bmm_route_pack_kernel[
        (triton.cdiv(assignments, pack_block),)
    ](
        flat_experts,
        light_counters,
        heavy_counter,
        light_token_pad,
        heavy_token_ids,
        slot_to_light,
        slot_to_heavy,
        heavy_expert,
        LIGHT_PADDED_COUNT=int(light_padded_count),
        TOP_K=int(top_k),
        ASSIGNMENTS=int(assignments),
        BLOCK=pack_block,
        num_warps=4,
        num_stages=1,
    )

    light_hidden = hidden.index_select(0, light_token_pad).reshape(
        num_experts,
        light_padded_count,
        int(hidden_dim),
    )
    light_gate_up = torch.bmm(light_hidden, gate_up_w.transpose(1, 2))
    del light_hidden
    light_activated = torch.empty(
        (num_experts, light_padded_count, intermediate_dim),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    light_activation_elements = (
        num_experts * light_padded_count * intermediate_dim
    )
    act_block = int(activation_block)
    _qwen3_moe_padded_bmm_activation_kernel[
        (triton.cdiv(light_activation_elements, act_block),)
    ](
        light_gate_up,
        light_activated,
        ELEMENTS=light_activation_elements,
        I=intermediate_dim,
        ACT=_activation_id(activation),
        BLOCK=act_block,
        num_warps=4,
        num_stages=1,
    )
    del light_gate_up
    try:
        light_projected = torch.bmm(
            light_activated,
            down_w.transpose(1, 2),
            out_dtype=torch.float32,
        )
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch build does not support BF16 bmm with FP32 output"
        ) from exc
    del light_activated

    heavy_hidden = hidden.index_select(0, heavy_token_ids)
    heavy_gate_up = torch.mm(
        heavy_hidden,
        gate_up_w[heavy_expert].transpose(0, 1),
    )
    del heavy_hidden
    heavy_activated = torch.empty(
        (heavy_count, intermediate_dim),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    heavy_activation_elements = heavy_count * intermediate_dim
    _qwen3_moe_padded_bmm_activation_kernel[
        (triton.cdiv(heavy_activation_elements, act_block),)
    ](
        heavy_gate_up,
        heavy_activated,
        ELEMENTS=heavy_activation_elements,
        I=intermediate_dim,
        ACT=_activation_id(activation),
        BLOCK=act_block,
        num_warps=4,
        num_stages=1,
    )
    del heavy_gate_up
    try:
        heavy_projected = torch.bmm(
            heavy_activated.unsqueeze(0),
            down_w[heavy_expert : heavy_expert + 1].transpose(1, 2),
            out_dtype=torch.float32,
        ).squeeze(0)
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch build does not support BF16 bmm with FP32 output"
        ) from exc
    del heavy_activated

    light_projected_2d = light_projected.reshape(
        light_padded_size,
        int(hidden_dim),
    )
    final = out if out is not None else torch.empty_like(hidden_states)
    residual_ptr = residual if residual is not None else hidden
    block_n = min(
        int(reduce_block_n),
        int(triton.next_power_of_2(max(1, int(hidden_dim)))),
    )
    block_top_k = int(triton.next_power_of_2(max(1, int(top_k))))
    _qwen3_moe_dominant_padded_bmm_reduce_kernel[
        (int(rows), triton.cdiv(int(hidden_dim), block_n))
    ](
        light_projected_2d,
        heavy_projected,
        flat_experts,
        slot_to_light,
        slot_to_heavy,
        flat_route,
        residual_ptr,
        final,
        heavy_expert,
        light_projected_2d.stride(0),
        light_projected_2d.stride(1),
        heavy_projected.stride(0),
        heavy_projected.stride(1),
        residual_ptr.stride(0),
        residual_ptr.stride(1),
        final.stride(0),
        final.stride(1),
        int(hidden_dim),
        TOP_K=int(top_k),
        BLOCK_TOP_K=block_top_k,
        ADD_RESIDUAL=bool(residual is not None),
        BLOCK_N=block_n,
        num_warps=int(reduce_num_warps),
        num_stages=1,
    )
    if workspace is not None:
        workspace["dominant_padded_bmm_prefill"] = 1
        workspace["dominant_padded_bmm_deterministic_reduce"] = 1
        workspace["dominant_padded_bmm_heavy_expert"] = int(heavy_expert)
        workspace["dominant_padded_bmm_heavy_count"] = int(heavy_count)
        workspace["dominant_padded_bmm_skew"] = float(dominant_skew)
        workspace["dominant_padded_bmm_light_assignments"] = int(
            light_assignments
        )
        workspace["dominant_padded_bmm_light_max_count"] = int(
            max_light_count
        )
        workspace["dominant_padded_bmm_light_padded_count"] = int(
            light_padded_count
        )
        workspace["dominant_padded_bmm_light_padded_assignments"] = int(
            light_padded_size
        )
        workspace["dominant_padded_bmm_light_padding_ratio"] = float(
            light_padding_ratio
        )
        workspace["dominant_padded_bmm_capacity_ratio"] = float(
            (light_padded_size + heavy_count) / assignments
        )
        workspace["dominant_padded_bmm_down_output_dtype"] = str(
            light_projected.dtype
        )
        workspace["dominant_padded_bmm_route_pack"] = "atomic_split"
        workspace["dominant_padded_bmm_route_pack_block"] = int(
            route_pack_block
        )
        workspace["dominant_padded_bmm_activation_block"] = int(
            activation_block
        )
        workspace["dominant_padded_bmm_reduce_block_n"] = int(
            reduce_block_n
        )
        workspace["dominant_padded_bmm_reduce_num_warps"] = int(
            reduce_num_warps
        )
    return final


def qwen3_moe_grouped_runtime_config() -> dict:
    return {
        "has_triton": bool(_HAS_TRITON),
        "topk_softmax": bool(_HAS_TRITON),
        "fused_router": bool(_HAS_TRITON and _CFG_FUSED_ROUTER),
        "fused_router_max_rows": int(_CFG_FUSED_ROUTER_MAX_ROWS),
        "router_k_splits": int(_CFG_ROUTER_K_SPLITS),
        "accum_store_first": False,
        "token_accum": bool(_HAS_TRITON and _CFG_TOKEN_ACCUM),
        "token_accum_min_rows": int(_CFG_TOKEN_ACCUM_MIN_ROWS),
        "grouped_fused_gate": bool(_HAS_TRITON and _CFG_GROUPED_FUSED_GATE),
        "grouped_dot": bool(
            _HAS_TRITON
            and _CFG_GROUPED_DOT
            and ((not _CFG_DECODE_CUDA_GRAPHS) or _CFG_GROUPED_DOT_ALLOW_CUDA_GRAPHS)
        ),
        "grouped_dot_requested": bool(_HAS_TRITON and _CFG_GROUPED_DOT),
        "grouped_dot_graph_disabled": bool(
            _HAS_TRITON
            and _CFG_GROUPED_DOT
            and _CFG_DECODE_CUDA_GRAPHS
            and not _CFG_GROUPED_DOT_ALLOW_CUDA_GRAPHS
        ),
        "expert_grouped_decode": bool(_HAS_TRITON and _CFG_EXPERT_GROUPED_DECODE),
        "shared_route_decode": bool(_HAS_TRITON and _CFG_SHARED_ROUTE_DECODE),
        "shared_route_batch_max_rows": int(_CFG_SHARED_ROUTE_BATCH_MAX_ROWS),
        "shared_route_assume_identical": bool(_CFG_SHARED_ROUTE_ASSUME_IDENTICAL),
        "single_row_gemv": bool(_CFG_SINGLE_ROW_GEMV),
        "shared_route_partial_reduce": bool(
            _HAS_TRITON and _CFG_SHARED_ROUTE_DECODE and _CFG_SHARED_ROUTE_PARTIAL_REDUCE
        ),
        "shared_route_coalesced_weights": bool(
            _HAS_TRITON and _CFG_SHARED_ROUTE_DECODE and _CFG_SHARED_ROUTE_COALESCED_WEIGHTS
        ),
        "shared_route_token_accum": bool(
            _HAS_TRITON and _CFG_SHARED_ROUTE_DECODE and _CFG_SHARED_ROUTE_TOKEN_ACCUM
        ),
        "shared_route_token_accum_num_warps": int(_CFG_SHARED_ROUTE_TOKEN_ACCUM_NUM_WARPS),
        "shared_route_block_m": int(_CFG_SHARED_ROUTE_BLOCK_M),
        "shared_route_gate_block_n": int(_CFG_SHARED_ROUTE_GATE_BLOCK_N),
        "shared_route_gate_k_splits": int(_CFG_SHARED_ROUTE_GATE_K_SPLITS),
        "shared_route_down_block_n": int(_CFG_SHARED_ROUTE_DOWN_BLOCK_N),
        "shared_route_split_gate": bool(
            _HAS_TRITON and _CFG_SHARED_ROUTE_DECODE and _CFG_SHARED_ROUTE_SPLIT_GATE
        ),
        "shared_route_split_gate_block_m": int(_CFG_SHARED_ROUTE_SPLIT_GATE_BLOCK_M),
        "shared_route_split_gate_num_stages": int(_CFG_SHARED_ROUTE_SPLIT_GATE_NUM_STAGES),
        "route_matrix_decode": bool(_HAS_TRITON and _CFG_ROUTE_MATRIX_DECODE),
        "route_matrix_max_rows": int(_CFG_ROUTE_MATRIX_MAX_ROWS),
        "expert_grouped_general_decode": bool(
            _HAS_TRITON and _CFG_EXPERT_GROUPED_GENERAL_DECODE
        ),
        "expert_grouped_dense_decode": bool(_HAS_TRITON and _CFG_EXPERT_GROUPED_DENSE_DECODE),
        "expert_grouped_compact_decode": bool(
            _HAS_TRITON and _CFG_EXPERT_GROUPED_COMPACT_DECODE
        ),
        "expert_grouped_compact_fused_pack": bool(
            _HAS_TRITON and _CFG_EXPERT_GROUPED_COMPACT_DECODE and _CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK
        ),
        "expert_grouped_compact_partial_reduce": bool(
            _HAS_TRITON
            and _CFG_EXPERT_GROUPED_COMPACT_DECODE
            and _CFG_EXPERT_GROUPED_COMPACT_FUSED_PACK
            and _CFG_EXPERT_GROUPED_COMPACT_PARTIAL_REDUCE
        ),
        "expert_grouped_compact_active_list": bool(
            _HAS_TRITON
            and _CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST
        ),
        "expert_grouped_compact_active_list_early_exit": bool(
            _HAS_TRITON
            and _CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST
            and _CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT
        ),
        "expert_grouped_compact_expert_grid_pack": bool(
            _HAS_TRITON
            and _CFG_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK
        ),
        "expert_grouped_compact_coalesced_weights": bool(
            _HAS_TRITON
            and _CFG_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS
        ),
        "expert_grouped_compact_token_accum": bool(
            _HAS_TRITON
            and _CFG_EXPERT_GROUPED_COMPACT_DECODE
            and _CFG_EXPERT_GROUPED_COMPACT_TOKEN_ACCUM
        ),
        "expert_grouped_compact_gate_block_n": int(_CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N),
        "expert_grouped_compact_down_block_n": int(_CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N),
        "expert_grouped_compact_num_warps": int(
            _CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS
        ),
        "expert_grouped_compact_num_stages": int(
            _CFG_EXPERT_GROUPED_COMPACT_NUM_STAGES
        ),
        "expert_grouped_compact_gate_num_stages": int(
            _CFG_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES
        ),
        "expert_grouped_compact_down_num_stages": int(
            _CFG_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES
        ),
        "expert_grouped_compact_experts_per_program": int(
            _CFG_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM
        ),
        "expert_grouped_compact_paired_gate_up_dot": bool(
            _HAS_TRITON
            and _CFG_EXPERT_GROUPED_COMPACT_PAIRED_GATE_UP_DOT
        ),
        "expert_grouped_compact_split_gate_up": bool(
            _HAS_TRITON
            and _CFG_EXPERT_GROUPED_COMPACT_SPLIT_GATE_UP
        ),
        "expert_grouped_compact_empty_expert_early_exit": bool(
            _HAS_TRITON
            and _CFG_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT
        ),
        "expert_grouped_compact_l2_grouped_grid": bool(
            _HAS_TRITON and _CFG_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID
        ),
        "expert_grouped_compact_l2_group_size": int(
            _CFG_EXPERT_GROUPED_COMPACT_L2_GROUP_SIZE
        ),
        "expert_grouped_compact_direct_out": bool(
            _HAS_TRITON
            and _CFG_EXPERT_GROUPED_COMPACT_DECODE
            and _CFG_EXPERT_GROUPED_COMPACT_DIRECT_OUT
        ),
        "expert_grouped_min_rows": int(_CFG_EXPERT_GROUPED_MIN_ROWS),
        "expert_grouped_max_rows": int(_CFG_EXPERT_GROUPED_MAX_ROWS),
        "expert_grouped_block_m": int(_CFG_EXPERT_GROUPED_BLOCK_M),
        "int8_decode": bool(_HAS_TRITON and _CFG_INT8_DECODE),
        "max_assignments": int(_CFG_MAX_ASSIGNMENTS),
        "force_triton": bool(_CFG_FORCE_TRITON),
        "block_n": int(_CFG_BLOCK_N),
        "block_k": int(_CFG_BLOCK_K),
        "router_block_k": int(_CFG_ROUTER_BLOCK_K),
        "num_warps": int(_CFG_NUM_WARPS),
        "num_stages": int(_CFG_NUM_STAGES),
        "segmented_prefill": bool(_HAS_TRITON and _CFG_SEGMENTED_PREFILL),
        "segmented_prefill_dense_grid": bool(
            _HAS_TRITON and _CFG_SEGMENTED_PREFILL_DENSE_GRID
        ),
        "segmented_prefill_fused_gate": bool(
            _HAS_TRITON and _CFG_SEGMENTED_PREFILL_FUSED_GATE
        ),
        "segmented_prefill_async_tiles": bool(
            _HAS_TRITON and _CFG_SEGMENTED_PREFILL_ASYNC_TILES
        ),
        "segmented_prefill_async_tiles_max_assignments": int(
            _CFG_SEGMENTED_PREFILL_ASYNC_TILES_MAX_ASSIGNMENTS
        ),
        "segmented_prefill_partial_reduce": bool(
            _HAS_TRITON and _CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE
        ),
        "segmented_prefill_partial_reduce_max_assignments": int(
            _CFG_SEGMENTED_PREFILL_PARTIAL_REDUCE_MAX_ASSIGNMENTS
        ),
        "segmented_prefill_partial_cache_max_assignments": int(
            _CFG_SEGMENTED_PREFILL_PARTIAL_CACHE_MAX_ASSIGNMENTS
        ),
        "segmented_prefill_route_scatter": bool(
            _HAS_TRITON and _CFG_SEGMENTED_PREFILL_ROUTE_SCATTER
        ),
        "segmented_prefill_fixed_route_pack": bool(
            _HAS_TRITON and _CFG_SEGMENTED_PREFILL_FIXED_ROUTE_PACK
        ),
        "segmented_prefill_compact_route_pack": bool(
            _HAS_TRITON and _CFG_SEGMENTED_PREFILL_COMPACT_ROUTE_PACK
        ),
        "segmented_prefill_single_accumulator": bool(
            _HAS_TRITON and _CFG_SEGMENTED_PREFILL_SINGLE_ACCUMULATOR
        ),
        "segmented_prefill_sorted_partial": bool(
            _HAS_TRITON and _CFG_SEGMENTED_PREFILL_SORTED_PARTIAL
        ),
        "segmented_prefill_group_size_m": int(
            _CFG_SEGMENTED_PREFILL_GROUP_SIZE_M
        ),
        "segmented_prefill_route_block": int(_CFG_SEGMENTED_PREFILL_ROUTE_BLOCK),
        "segmented_prefill_min_assignments": int(_CFG_SEGMENTED_PREFILL_MIN_ASSIGNMENTS),
        "segmented_prefill_block_m": int(_CFG_SEGMENTED_PREFILL_BLOCK_M),
        "segmented_prefill_block_n": int(_CFG_SEGMENTED_PREFILL_BLOCK_N),
        "segmented_prefill_block_k": int(_CFG_SEGMENTED_PREFILL_BLOCK_K),
        "segmented_prefill_fused_gate_block_n": int(
            _CFG_SEGMENTED_PREFILL_FUSED_GATE_BLOCK_N
        ),
        "segmented_prefill_num_warps": int(_CFG_SEGMENTED_PREFILL_NUM_WARPS),
        "segmented_prefill_num_stages": int(_CFG_SEGMENTED_PREFILL_NUM_STAGES),
    }


HAS_QWEN3_MOE_GROUPED = _HAS_TRITON


__all__ = [
    "HAS_QWEN3_MOE_GROUPED",
    "qwen3_moe_grouped_decode",
    "qwen3_moe_grouped_decode_int8",
    "qwen3_moe_grouped_prefers_triton_shape",
    "qwen3_moe_grouped_runtime_config",
    "qwen3_moe_compact_route_pack",
    "qwen3_moe_dominant_expert_padded_bmm_prefill",
    "qwen3_moe_router_topk_softmax",
    "qwen3_moe_padded_bmm_prefill",
    "qwen3_moe_segmented_prefers_triton_shape",
    "qwen3_moe_segmented_prefill",
    "qwen3_moe_topk_softmax",
    "qwen3_moe_topk_softmax_compact_pack",
]

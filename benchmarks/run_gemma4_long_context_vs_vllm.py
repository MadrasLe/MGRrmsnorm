"""Fair Gemma 4 MoE long-context benchmark for MegaGemm and vLLM.

Each backend runs in a separate process. A shared token-ID manifest keeps every
prompt byte-for-byte identical while one model load covers the full context and
batch matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from run_gemma4_moe_batch_vs_vllm import (  # noqa: E402
    env_flag,
    prompt_token_contract,
    run_excluded_megagemm_prefill_profile,
    run_megagemm_request,
    run_vllm_request,
    summarize,
    token_matrix_comparison,
)
from run_gemma4_moe_vs_vllm import (  # noqa: E402
    dtype_from_name,
    gpu_snapshot,
    make_vllm,
)


DEFAULT_PROMPT = (
    "Write a compact Python Fibonacci function and explain its time complexity."
)
FILLER_TEXT = """
This benchmark uses a deterministic technical passage about numerical computing,
GPU kernels, memory bandwidth, synchronization, matrix multiplication, attention,
key value caches, sparse expert routing, software testing, reproducible timing,
compiler behavior, data structures, algorithms, operating systems, distributed
services, scientific measurement, error analysis, and reliable engineering.
The passage is deliberately varied so expert routing sees ordinary language rather
than one repeated token. Both inference engines receive exactly the same token IDs.
""".strip()
SUPPORTED_BATCHES = (1, 2, 4, 8, 16)
MEGAGEMM_MIN_WARMUPS = 3
MEGAGEMM_MAX_WARMUPS = 8
MEGAGEMM_REQUIRED_STABLE_WARMUP_PAIRS = 2
MEGAGEMM_WARMUP_MAX_LAST_PAIR_RATIO = 1.10
MEGAGEMM_DECODE_MODE_GRAPH_STEP = "request_local_graph_step"
MEGAGEMM_DECODE_MODE_GRAPH_BURST = "request_local_graph_burst8_gpu_feedback"
MEGAGEMM_DECODE_MODE_EAGER = "eager"
MEGAGEMM_PREFILL_PROFILE_HYBRID = "guarded_dominant_expert"
MEGAGEMM_PREFILL_PROFILE_SEGMENTED = "segmented_deterministic"
MEGAGEMM_PADDED_PREFILL_MINIMUM_SPEEDUP = 1.02
MEGAGEMM_PREFILL_CHUNK_BASELINE_TOKENS = 16_384
MEGAGEMM_PREFILL_CHUNK_CANDIDATE_TOKENS = 32_768
MEGAGEMM_PREFILL_CHUNK_MINIMUM_SPEEDUP = 1.01
MEGAGEMM_PREFILL_CHUNK_MAXIMUM_SPREAD_RATIO = 1.05
BENCHMARK_FORCED_TOKEN_ENV = "MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID"
ROUTE_NORMALIZED_POLICY = "full_lm_head_single_allowed_continuation"
ROUTE_NORMALIZED_DEFAULT_REPEATS = 3
DECODE_STAGE_PROFILE_STEPS = 8


def parse_int_list(raw: str, *, name: str) -> list[int]:
    values: list[int] = []
    for item in str(raw).split(","):
        value = int(item.strip())
        if value <= 0:
            raise ValueError(f"{name} values must be positive")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"at least one {name} value is required")
    return values


def build_long_prompt_token_rows(
    base_ids: list[int],
    filler_ids: list[int],
    special_ids: set[int],
    target_tokens: int,
    required_rows: int,
) -> list[list[int]]:
    """Insert deterministic normal-text tokens before the original prompt tail."""
    base = [int(token) for token in base_ids]
    filler = [int(token) for token in filler_ids if int(token) not in special_ids]
    if not base:
        raise ValueError("base prompt token list is empty")
    if target_tokens < len(base):
        raise ValueError(
            f"target context {target_tokens} is shorter than base prompt {len(base)}"
        )
    if required_rows < 1:
        raise ValueError("required_rows must be positive")

    filler_count = target_tokens - len(base)
    if filler_count and not filler:
        raise ValueError("normal-text filler token list is empty")
    if required_rows > 1 and filler_count == 0:
        raise ValueError("distinct rows require at least one filler token")

    prefix_len = 1 if base[0] in special_ids else 0
    prefix = base[:prefix_len]
    suffix = base[prefix_len:]
    marker_ids = list(dict.fromkeys(filler))
    if required_rows > len(marker_ids):
        raise ValueError(
            f"only {len(marker_ids)} distinct filler tokens for {required_rows} rows"
        )

    rows: list[list[int]] = []
    for row_index in range(required_rows):
        inserted: list[int] = []
        if filler_count:
            shift = (row_index * 37) % len(filler)
            inserted = [
                filler[(shift + position) % len(filler)]
                for position in range(filler_count)
            ]
            inserted[0] = marker_ids[row_index]
        row = prefix + inserted + suffix
        if len(row) != target_tokens:
            raise AssertionError(
                f"constructed {len(row)} tokens; expected {target_tokens}"
            )
        rows.append(row)

    if len({tuple(row) for row in rows}) != required_rows:
        raise RuntimeError("long-context prompt rows are not distinct")
    return rows


def _tokenize_base_and_filler(model: str, prompt: str) -> tuple[Any, list[int], list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    formatted = prompt
    if getattr(tokenizer, "chat_template", None):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    base_ids = [
        int(token)
        for token in tokenizer.encode(formatted, add_special_tokens=False)
    ]
    filler_ids = [
        int(token)
        for token in tokenizer.encode(FILLER_TEXT, add_special_tokens=False)
    ]
    return tokenizer, base_ids, filler_ids


def load_or_create_prompt_manifest(
    model: str,
    prompt: str,
    contexts: list[int],
    required_rows: int,
    path: Path,
) -> tuple[dict[int, list[list[int]]], dict[str, Any]]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1:
            raise RuntimeError(f"unsupported prompt manifest schema: {path}")
    else:
        tokenizer, base_ids, filler_ids = _tokenize_base_and_filler(model, prompt)
        special_ids = {int(token) for token in tokenizer.all_special_ids}
        cases: dict[str, Any] = {}
        for context in contexts:
            rows = build_long_prompt_token_rows(
                base_ids,
                filler_ids,
                special_ids,
                context,
                required_rows,
            )
            cases[str(context)] = {
                "contract": prompt_token_contract(rows),
                "token_ids": rows,
            }
        payload = {
            "schema_version": 1,
            "generator": "gemma4_long_context_v1",
            "base_prompt_tokens": len(base_ids),
            "cases": cases,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    token_rows: dict[int, list[list[int]]] = {}
    for context in contexts:
        case = (payload.get("cases") or {}).get(str(context))
        if not case:
            raise RuntimeError(f"prompt manifest lacks context {context}: {path}")
        rows = [
            [int(token) for token in row]
            for row in case.get("token_ids", [])
        ]
        if len(rows) < required_rows:
            raise RuntimeError(
                f"context {context} has {len(rows)} rows; {required_rows} required"
            )
        stored_contract = prompt_token_contract(rows)
        if stored_contract != case.get("contract"):
            raise RuntimeError(f"prompt manifest contract mismatch at context {context}")
        if int(stored_contract["tokens_per_row"]) != context:
            raise RuntimeError(
                f"manifest context {context} has "
                f"{stored_contract['tokens_per_row']} tokens"
            )
        token_rows[context] = rows[:required_rows]
    return token_rows, payload


def case_key(batch_size: int, context: int) -> str:
    return f"b{int(batch_size)}_c{int(context)}"


def route_normalized_token_id(prompts: list[list[int]]) -> int:
    """Pick one ordinary in-prompt token for identical continuation routing."""
    if not prompts or not prompts[0]:
        raise ValueError("route-normalized diagnostic requires non-empty prompts")
    row = prompts[0]
    position = len(row) // 2 if len(row) > 2 else 0
    token_id = int(row[position])
    if token_id < 0:
        raise ValueError("route-normalized token ID must be non-negative")
    return token_id


def forced_token_matrix_contract(
    matrix: list[list[int]],
    *,
    token_id: int,
    expected_rows: int,
    expected_tokens: int,
) -> dict[str, Any]:
    expected = [
        [int(token_id)] * int(expected_tokens)
        for _ in range(int(expected_rows))
    ]
    comparison = token_matrix_comparison(expected, matrix)
    return {
        **comparison,
        "forced_token_id": int(token_id),
        "expected_rows": int(expected_rows),
        "expected_tokens_per_row": int(expected_tokens),
    }


@contextmanager
def megagemm_benchmark_forced_token(token_id: int):
    """Force feedback tokens while retaining the model's real LM-head work."""
    import megagemm.models.llama as llama_model

    token_id = int(token_id)
    if token_id < 0:
        raise ValueError("forced token ID must be non-negative")
    previous_env = os.environ.get(BENCHMARK_FORCED_TOKEN_ENV)
    previous_module_value = int(
        getattr(llama_model, "_BENCHMARK_FORCED_TOKEN_ID", -1)
    )
    os.environ[BENCHMARK_FORCED_TOKEN_ENV] = str(token_id)
    llama_model._BENCHMARK_FORCED_TOKEN_ID = token_id
    try:
        yield
    finally:
        llama_model._BENCHMARK_FORCED_TOKEN_ID = previous_module_value
        if previous_env is None:
            os.environ.pop(BENCHMARK_FORCED_TOKEN_ENV, None)
        else:
            os.environ[BENCHMARK_FORCED_TOKEN_ENV] = previous_env


def summarize_long(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result = summarize(samples)
    for key in ("prefill_tok_s", "input_plus_output_tok_s"):
        values = [float(row[key]) for row in samples]
        result[f"{key}_median"] = float(statistics.median(values))
    return result


def megagemm_deterministic_moe_contract(runtime: dict[str, Any]) -> dict[str, Any]:
    prefill_layers = int(runtime.get("gemma4_a4b_segmented_prefill_layers", 0) or 0)
    prefill_deterministic = int(
        runtime.get(
            "qwen3_moe_segmented_prefill_deterministic_reduce_layers",
            0,
        )
        or 0
    )
    prefill_atomic = int(
        runtime.get("qwen3_moe_segmented_prefill_atomic_reduce_layers", 0) or 0
    )
    padded_bmm_layers = int(
        runtime.get("gemma4_long_padded_bmm_prefill_last_active_layers", 0) or 0
    )
    padded_bmm_disabled = int(
        runtime.get("gemma4_long_padded_bmm_prefill_disabled_layers", 0) or 0
    )
    padded_bmm_enabled = bool(
        runtime.get("gemma4_long_padded_bmm_prefill_enabled", False)
    )
    padded_bmm_failure = str(
        runtime.get("gemma4_long_padded_bmm_prefill_first_failure", "") or ""
    )
    padded_bmm_failures = list(
        runtime.get("gemma4_long_padded_bmm_prefill_failures") or []
    )
    dominant_layers = int(
        runtime.get(
            "gemma4_long_dominant_expert_prefill_last_active_layers",
            0,
        )
        or 0
    )
    dominant_guard_layers = int(
        runtime.get(
            "gemma4_long_dominant_expert_prefill_guard_miss_layers",
            0,
        )
        or 0
    )
    dominant_disabled = int(
        runtime.get(
            "gemma4_long_dominant_expert_prefill_disabled_layers",
            0,
        )
        or 0
    )
    dominant_enabled = bool(
        runtime.get("gemma4_long_dominant_expert_prefill_enabled", False)
    )
    dominant_failure = str(
        runtime.get(
            "gemma4_long_dominant_expert_prefill_first_failure",
            "",
        )
        or ""
    )
    dominant_failures = list(
        runtime.get("gemma4_long_dominant_expert_prefill_failures") or []
    )
    dominant_fallback_layers = dominant_guard_layers + dominant_disabled
    segmented_prefill_exact = bool(
        prefill_deterministic == prefill_layers
        and prefill_atomic == 0
        and padded_bmm_layers == 0
        and dominant_layers == 0
    )
    padded_bmm_prefill_exact = bool(
        padded_bmm_enabled
        and padded_bmm_layers == prefill_layers
        and padded_bmm_disabled == 0
        and prefill_deterministic == 0
        and prefill_atomic == 0
    )
    hybrid_prefill_exact = bool(
        padded_bmm_enabled
        and padded_bmm_layers > 0
        and padded_bmm_disabled > 0
        and padded_bmm_layers + padded_bmm_disabled == prefill_layers
        and prefill_deterministic == padded_bmm_disabled
        and prefill_atomic == 0
    )
    dominant_prefill_exact = bool(
        dominant_enabled
        and dominant_layers == prefill_layers
        and dominant_fallback_layers == 0
        and prefill_deterministic == 0
        and prefill_atomic == 0
    )
    dominant_hybrid_prefill_exact = bool(
        dominant_enabled
        and dominant_layers > 0
        and dominant_fallback_layers > 0
        and dominant_layers + dominant_fallback_layers == prefill_layers
        and prefill_deterministic == dominant_fallback_layers
        and prefill_atomic == 0
    )
    decode_policy = dict(runtime.get("gemma4_batch_moe_decode_policy") or {})
    decode_layers = int(decode_policy.get("enabled_layers", 0) or 0)
    decode_deterministic = int(
        runtime.get("gemma4_batch_moe_decode_deterministic_reduce_layers", 0)
        or 0
    )
    exact = bool(
        prefill_layers > 0
        and decode_layers > 0
        and (
            segmented_prefill_exact
            or padded_bmm_prefill_exact
            or hybrid_prefill_exact
            or dominant_prefill_exact
            or dominant_hybrid_prefill_exact
        )
        and decode_deterministic == decode_layers
    )
    if dominant_prefill_exact:
        prefill_backend = "dominant_expert_hybrid_fp32"
    elif dominant_hybrid_prefill_exact:
        prefill_backend = (
            "dominant_expert_hybrid_fp32_with_segmented_deterministic_fallback"
        )
    elif padded_bmm_prefill_exact:
        prefill_backend = "padded_bmm_fp32"
    elif hybrid_prefill_exact:
        prefill_backend = "padded_bmm_fp32_with_segmented_deterministic_fallback"
    elif segmented_prefill_exact:
        prefill_backend = "segmented_deterministic"
    else:
        prefill_backend = "invalid"
    return {
        "exact": exact,
        "prefill_layers": prefill_layers,
        "prefill_deterministic_reduce_layers": prefill_deterministic,
        "prefill_atomic_reduce_layers": prefill_atomic,
        "prefill_backend": prefill_backend,
        "prefill_covered_layers": (
            dominant_layers + padded_bmm_layers + prefill_deterministic
        ),
        "dominant_expert_prefill_layers": dominant_layers,
        "dominant_expert_prefill_guard_miss_layers": dominant_guard_layers,
        "dominant_expert_prefill_disabled_layers": dominant_disabled,
        "dominant_expert_prefill_first_failure": dominant_failure,
        "dominant_expert_prefill_failures": dominant_failures,
        "padded_bmm_prefill_layers": padded_bmm_layers,
        "padded_bmm_prefill_disabled_layers": padded_bmm_disabled,
        "padded_bmm_prefill_first_failure": padded_bmm_failure,
        "padded_bmm_prefill_failures": padded_bmm_failures,
        "decode_layers": decode_layers,
        "decode_deterministic_reduce_layers": decode_deterministic,
    }


def megagemm_compact_active_list_contract(
    runtime: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    requested = env_flag(
        "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST",
        False,
    )
    early_exit_requested = env_flag(
        "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT",
        False,
    )
    runtime_enabled = bool(
        runtime.get("qwen3_moe_expert_grouped_compact_active_list", False)
    )
    runtime_early_exit_enabled = bool(
        runtime.get(
            "qwen3_moe_expert_grouped_compact_active_list_early_exit",
            False,
        )
    )
    policy = dict(runtime.get("gemma4_batch_moe_decode_policy") or {})
    decode_layers = int(policy.get("enabled_layers", 0) or 0)
    active_list_layers = int(
        runtime.get("gemma4_batch_moe_decode_active_list_layers", 0) or 0
    )
    early_exit_layers = int(
        runtime.get(
            "gemma4_batch_moe_decode_active_list_early_exit_layers",
            0,
        )
        or 0
    )
    required_decode_layers = 30 if requested and batch_size == 16 else decode_layers
    expected_layers = required_decode_layers if requested and batch_size == 16 else 0
    expected_early_exit_layers = (
        decode_layers
        if requested and early_exit_requested and batch_size == 16
        else 0
    )
    exact = bool(
        runtime_enabled == requested
        and runtime_early_exit_enabled == early_exit_requested
        and (not requested or early_exit_requested)
        and decode_layers == required_decode_layers
        and active_list_layers == expected_layers
        and early_exit_layers == expected_early_exit_layers
    )
    return {
        "exact": exact,
        "requested": requested,
        "early_exit_requested": early_exit_requested,
        "runtime_enabled": runtime_enabled,
        "runtime_early_exit_enabled": runtime_early_exit_enabled,
        "decode_layers": decode_layers,
        "required_decode_layers": required_decode_layers,
        "active_list_layers": active_list_layers,
        "active_list_early_exit_layers": early_exit_layers,
        "expected_active_list_layers": expected_layers,
        "expected_active_list_early_exit_layers": expected_early_exit_layers,
    }


def megagemm_compact_active_expert_snapshot(model: Any) -> dict[str, Any]:
    active_by_layer: list[int] = []
    for layer in getattr(model, "layers", ()):
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        workspace = getattr(experts, "_grouped_decode_workspace", None)
        if not isinstance(workspace, dict):
            continue
        counts = workspace.get("expert_grouped_compact_counts")
        if not isinstance(counts, torch.Tensor) or counts.numel() == 0:
            continue
        active_by_layer.append(int(torch.count_nonzero(counts).item()))

    if not active_by_layer:
        return {
            "available": False,
            "layers": 0,
            "active_experts_by_layer": [],
        }
    return {
        "available": True,
        "layers": len(active_by_layer),
        "active_experts_by_layer": active_by_layer,
        "minimum": min(active_by_layer),
        "median": float(statistics.median(active_by_layer)),
        "maximum": max(active_by_layer),
        "mean": float(statistics.fmean(active_by_layer)),
    }


def summarize_compact_active_expert_snapshots(
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    available = [row for row in snapshots if bool(row.get("available", False))]
    active_values = [
        int(value)
        for row in available
        for value in row.get("active_experts_by_layer", [])
    ]
    if not active_values:
        return {
            "available": False,
            "requests": 0,
            "layer_observations": 0,
        }
    return {
        "available": True,
        "requests": len(available),
        "layers_per_request": [int(row.get("layers", 0)) for row in available],
        "layer_observations": len(active_values),
        "minimum": min(active_values),
        "median": float(statistics.median(active_values)),
        "maximum": max(active_values),
        "mean": float(statistics.fmean(active_values)),
        "request_medians": [float(row["median"]) for row in available],
    }


def megagemm_long_sliding_prefill_contract(
    runtime: dict[str, Any],
    *,
    batch_size: int,
    context: int,
) -> dict[str, Any]:
    expected = bool(context == 2048 and batch_size in (8, 16))
    enabled = bool(runtime.get("gemma4_long_sliding_prefill_enabled", False))
    hits = int(runtime.get("gemma4_long_sliding_prefill_hits", 0) or 0)
    exact = bool(not expected or (enabled and hits > 0))
    return {
        "exact": exact,
        "expected": expected,
        "enabled": enabled,
        "hits": hits,
        "shape": {"batch_size": int(batch_size), "context": int(context)},
    }


def megagemm_long_decode_attention_contract(
    runtime: dict[str, Any],
    *,
    batch_size: int,
    context: int,
) -> dict[str, Any]:
    expected = bool(batch_size == 16 and context == 2048)
    paged = dict(runtime.get("paged_decode_runtime") or {})
    selected = dict(paged.get("grouped_segmented_selected_segments") or {})
    selected_tiles = dict(
        paged.get("grouped_segmented_selected_tile_sizes") or {}
    )
    sliding_segments = int(selected.get("sliding_h256_gqa2", 0) or 0)
    full_segments = int(selected.get("full_h512_gqa8", 0) or 0)
    sliding_tile_size = int(
        selected_tiles.get("sliding_h256_gqa2", 0) or 0
    )
    full_tile_size = int(selected_tiles.get("full_h512_gqa8", 0) or 0)
    exact = bool(
        not expected
        or (
            not bool(paged.get("grouped_segmented_disabled", False))
            and not str(paged.get("grouped_segmented_failure", "") or "")
            and sliding_segments == 32
            and full_segments == 8
            and sliding_tile_size == 64
            and full_tile_size == 16
        )
    )
    return {
        "exact": exact,
        "expected": expected,
        "sliding_segments": sliding_segments,
        "full_segments": full_segments,
        "sliding_tile_size": sliding_tile_size,
        "full_tile_size": full_tile_size,
        "grouped_segmented_disabled": bool(
            paged.get("grouped_segmented_disabled", False)
        ),
        "grouped_segmented_failure": str(
            paged.get("grouped_segmented_failure", "") or ""
        ),
        "shape": {"batch_size": int(batch_size), "context": int(context)},
    }


def megagemm_long_full_prefill_contract(
    runtime: dict[str, Any],
    *,
    batch_size: int,
    context: int,
) -> dict[str, Any]:
    expected = bool(context == 2048 and batch_size in (8, 16))
    enabled = bool(runtime.get("gemma4_long_full_prefill_enabled", False))
    hits = int(runtime.get("gemma4_long_full_prefill_hits", 0) or 0)
    exact = bool(not expected or (enabled and hits > 0))
    return {
        "exact": exact,
        "expected": expected,
        "enabled": enabled,
        "hits": hits,
        "shape": {"batch_size": int(batch_size), "context": int(context)},
    }


def megagemm_long_attention_prepare_contract(
    runtime: dict[str, Any],
    *,
    batch_size: int,
    context: int,
) -> dict[str, Any]:
    expected = bool(context == 2048 and batch_size in (8, 16))
    enabled = bool(runtime.get("gemma4_fused_attn_prepare_enabled", False))
    hits = int(runtime.get("gemma4_fused_attn_prepare_hits", 0) or 0)
    disabled_layers = int(
        runtime.get("gemma4_fused_attn_prepare_disabled_layers", 0) or 0
    )
    skip_reason = str(
        runtime.get("gemma4_fused_attn_prepare_skip_reason", "") or ""
    )
    exact = bool(
        not expected
        or (enabled and hits > 0 and disabled_layers == 0 and not skip_reason)
    )
    return {
        "exact": exact,
        "expected": expected,
        "enabled": enabled,
        "hits": hits,
        "disabled_layers": disabled_layers,
        "skip_reason": skip_reason,
        "shape": {"batch_size": int(batch_size), "context": int(context)},
    }


def megagemm_long_dominant_expert_prefill_contract(
    runtime: dict[str, Any],
    *,
    batch_size: int,
    context: int,
) -> dict[str, Any]:
    expected = bool(batch_size == 16 and context == 2048)
    enabled = bool(
        runtime.get("gemma4_long_dominant_expert_prefill_enabled", False)
    )
    active_layers = int(
        runtime.get(
            "gemma4_long_dominant_expert_prefill_last_active_layers",
            0,
        )
        or 0
    )
    guard_miss_layers = int(
        runtime.get(
            "gemma4_long_dominant_expert_prefill_guard_miss_layers",
            0,
        )
        or 0
    )
    disabled_layers = int(
        runtime.get(
            "gemma4_long_dominant_expert_prefill_disabled_layers",
            0,
        )
        or 0
    )
    fallback_layers = guard_miss_layers + disabled_layers
    expected_layers = int(
        runtime.get("gemma4_a4b_segmented_prefill_layers", 0) or 0
    )
    deterministic_fallback_layers = int(
        runtime.get(
            "qwen3_moe_segmented_prefill_deterministic_reduce_layers",
            0,
        )
        or 0
    )
    atomic_fallback_layers = int(
        runtime.get("qwen3_moe_segmented_prefill_atomic_reduce_layers", 0) or 0
    )
    hits = int(
        runtime.get("gemma4_long_dominant_expert_prefill_hits", 0) or 0
    )
    minimum_skew = float(
        runtime.get(
            "gemma4_long_dominant_expert_prefill_minimum_skew",
            0.0,
        )
        or 0.0
    )
    max_light_padding_ratio = float(
        runtime.get(
            "gemma4_long_dominant_expert_prefill_max_light_padding_ratio",
            0.0,
        )
        or 0.0
    )
    profiles = list(
        runtime.get("gemma4_long_dominant_expert_prefill_profiles") or []
    )
    guard_rejections = list(
        runtime.get("gemma4_long_dominant_expert_prefill_guard_rejections") or []
    )
    failures = list(
        runtime.get("gemma4_long_dominant_expert_prefill_failures") or []
    )
    first_failure = str(
        runtime.get(
            "gemma4_long_dominant_expert_prefill_first_failure",
            "",
        )
        or ""
    )
    candidate_config_exact = bool(
        int(runtime.get("gemma4_long_dominant_expert_prefill_rows", 0) or 0)
        == 32_768
        and str(
            runtime.get(
                "gemma4_long_dominant_expert_prefill_down_output_dtype",
                "",
            )
            or ""
        )
        == "fp32"
        and str(
            runtime.get(
                "gemma4_long_dominant_expert_prefill_route_pack",
                "",
            )
            or ""
        )
        == "atomic_split"
        and bool(
            runtime.get(
                "gemma4_long_dominant_expert_prefill_deterministic_reduce",
                False,
            )
        )
        and int(
            runtime.get(
                "gemma4_long_dominant_expert_prefill_route_pack_block",
                0,
            )
            or 0
        )
        == 256
        and int(
            runtime.get(
                "gemma4_long_dominant_expert_prefill_activation_block",
                0,
            )
            or 0
        )
        == 512
        and int(
            runtime.get(
                "gemma4_long_dominant_expert_prefill_reduce_block_n",
                0,
            )
            or 0
        )
        == 256
        and int(
            runtime.get(
                "gemma4_long_dominant_expert_prefill_reduce_num_warps",
                0,
            )
            or 0
        )
        == 4
        and int(
            runtime.get(
                "gemma4_long_dominant_expert_prefill_align_m",
                0,
            )
            or 0
        )
        == 16
        and math.isclose(minimum_skew, 7.5, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(
            max_light_padding_ratio,
            1.25,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    profiles_exact = bool(
        len(profiles) == active_layers
        and all(
            float(profile.get("dominant_skew", 0.0) or 0.0) >= minimum_skew
            and 1.0
            <= float(profile.get("light_padding_ratio", 0.0) or 0.0)
            <= max_light_padding_ratio
            for profile in profiles
        )
    )
    guard_details_exact = bool(len(guard_rejections) == guard_miss_layers)
    failure_details_exact = bool(
        len(failures) == disabled_layers
        and ((disabled_layers == 0 and not first_failure) or first_failure)
    )
    dominant_only_exact = bool(
        expected_layers > 0
        and active_layers == expected_layers
        and fallback_layers == 0
        and deterministic_fallback_layers == 0
        and atomic_fallback_layers == 0
        and hits > 0
    )
    hybrid_fallback_exact = bool(
        expected_layers > 0
        and active_layers > 0
        and fallback_layers > 0
        and active_layers + fallback_layers == expected_layers
        and deterministic_fallback_layers == fallback_layers
        and atomic_fallback_layers == 0
        and hits > 0
    )
    segmented_only_exact = bool(
        expected_layers > 0
        and active_layers == 0
        and deterministic_fallback_layers == expected_layers
        and atomic_fallback_layers == 0
        and (
            (not enabled and fallback_layers == 0)
            or (enabled and fallback_layers == expected_layers)
        )
    )
    exact = bool(
        not expected
        or (
            candidate_config_exact
            and profiles_exact
            and guard_details_exact
            and failure_details_exact
            and (
                dominant_only_exact
                or hybrid_fallback_exact
                or segmented_only_exact
            )
        )
    )
    if dominant_only_exact:
        coverage_mode = "dominant_expert_hybrid_fp32"
    elif hybrid_fallback_exact:
        coverage_mode = (
            "dominant_expert_hybrid_fp32_with_segmented_deterministic_fallback"
        )
    elif segmented_only_exact:
        coverage_mode = "segmented_deterministic_fallback"
    else:
        coverage_mode = "invalid"
    return {
        "exact": exact,
        "expected": expected,
        "enabled": enabled,
        "candidate": "dense_dominant_plus_light_padded_bmm",
        "candidate_config_exact": candidate_config_exact,
        "active_layers": active_layers,
        "guard_miss_layers": guard_miss_layers,
        "disabled_layers": disabled_layers,
        "fallback_layers": fallback_layers,
        "expected_layers": expected_layers,
        "deterministic_fallback_layers": deterministic_fallback_layers,
        "atomic_fallback_layers": atomic_fallback_layers,
        "hits": hits,
        "minimum_skew": minimum_skew,
        "max_light_padding_ratio": max_light_padding_ratio,
        "profiles": profiles,
        "profiles_exact": profiles_exact,
        "guard_rejections": guard_rejections,
        "guard_details_exact": guard_details_exact,
        "failures": failures,
        "first_failure": first_failure,
        "failure_details_exact": failure_details_exact,
        "dominant_only_exact": dominant_only_exact,
        "hybrid_fallback_exact": hybrid_fallback_exact,
        "segmented_only_exact": segmented_only_exact,
        "coverage_mode": coverage_mode,
        "shape": {"batch_size": int(batch_size), "context": int(context)},
    }


def megagemm_long_routed_expert_prefill_contract(
    runtime: dict[str, Any],
    *,
    batch_size: int,
    context: int,
) -> dict[str, Any]:
    if (
        batch_size == 16
        and context == 2048
        and "gemma4_long_dominant_expert_prefill_enabled" in runtime
    ):
        return megagemm_long_dominant_expert_prefill_contract(
            runtime,
            batch_size=batch_size,
            context=context,
        )
    expected = bool(batch_size == 16 and context in (1024, 2048))
    enabled = bool(runtime.get("gemma4_long_padded_bmm_prefill_enabled", False))
    long_rows = int(runtime.get("gemma4_long_padded_bmm_prefill_rows", 0) or 0)
    down_output_dtype = str(
        runtime.get("gemma4_long_padded_bmm_prefill_down_output_dtype", "") or ""
    )
    route_pack = str(
        runtime.get("gemma4_long_padded_bmm_prefill_route_pack", "") or ""
    )
    route_pack_block = int(
        runtime.get("gemma4_long_padded_bmm_prefill_route_pack_block", 0) or 0
    )
    max_padding_ratio = float(
        runtime.get(
            "gemma4_long_padded_bmm_prefill_max_padding_ratio",
            0.0,
        )
        or 0.0
    )
    fused_activation = bool(
        runtime.get("gemma4_long_padded_bmm_prefill_fused_activation", False)
    )
    activation_block = int(
        runtime.get("gemma4_long_padded_bmm_prefill_activation_block", 0) or 0
    )
    reduce_block_n = int(
        runtime.get("gemma4_long_padded_bmm_prefill_reduce_block_n", 0) or 0
    )
    reduce_num_warps = int(
        runtime.get("gemma4_long_padded_bmm_prefill_reduce_num_warps", 0) or 0
    )
    align_m = int(runtime.get("gemma4_long_padded_bmm_prefill_align_m", 0) or 0)
    hits = int(runtime.get("gemma4_long_padded_bmm_prefill_hits", 0) or 0)
    active_layers = int(
        runtime.get("gemma4_long_padded_bmm_prefill_last_active_layers", 0) or 0
    )
    disabled_layers = int(
        runtime.get("gemma4_long_padded_bmm_prefill_disabled_layers", 0) or 0
    )
    failure = str(
        runtime.get("gemma4_long_padded_bmm_prefill_first_failure", "") or ""
    )
    failures = list(runtime.get("gemma4_long_padded_bmm_prefill_failures") or [])
    expected_layers = int(runtime.get("gemma4_a4b_segmented_prefill_layers", 0) or 0)
    deterministic_fallback_layers = int(
        runtime.get(
            "qwen3_moe_segmented_prefill_deterministic_reduce_layers",
            0,
        )
        or 0
    )
    atomic_fallback_layers = int(
        runtime.get("qwen3_moe_segmented_prefill_atomic_reduce_layers", 0) or 0
    )
    segmented_config = dict(
        runtime.get("gemma4_a4b_segmented_prefill_config") or {}
    )
    segmented_long_rows = int(segmented_config.get("long_rows", 0) or 0)
    segmented_long_config = dict(segmented_config.get("long") or {})
    async_tile_limit = int(
        segmented_long_config.get("async_tiles_max_assignments", 0) or 0
    )
    async_tile_expected_assignments = int(batch_size) * int(context) * 8
    async_tile_requested = bool(
        expected and async_tile_limit >= async_tile_expected_assignments
    )
    async_tile_hits = int(
        runtime.get("qwen3_moe_segmented_prefill_async_tile_hits", 0) or 0
    )
    async_tile_contract_exact = bool(
        not async_tile_requested or async_tile_hits > 0
    )
    sorted_partial_requested = bool(
        expected and segmented_long_config.get("sorted_partial", False)
    )
    sorted_partial_hits = int(
        runtime.get("qwen3_moe_segmented_prefill_sorted_partial_hits", 0) or 0
    )
    sorted_partial_layers = int(
        runtime.get("qwen3_moe_segmented_prefill_sorted_partial_layers", 0) or 0
    )
    sorted_partial_contract_exact = bool(
        not sorted_partial_requested
        or (
            sorted_partial_hits > 0
            and expected_layers > 0
            and sorted_partial_layers == expected_layers
        )
    )
    segmented_long_config_exact = bool(
        segmented_long_rows == 16_384
        and int(segmented_long_config.get("block_m", 0) or 0) in (64, 128)
        and int(segmented_long_config.get("block_n", 0) or 0) == 256
        and int(segmented_long_config.get("block_k", 0) or 0) == 64
        and int(segmented_long_config.get("fused_gate_block_n", 0) or 0) == 128
        and int(segmented_long_config.get("num_warps", 0) or 0) in (4, 8)
        and int(segmented_long_config.get("num_stages", 0) or 0) in (3, 4)
        and not bool(segmented_long_config.get("compact_route_pack", False))
    )
    deterministic_coverage_exact = bool(
        expected_layers > 0
        and active_layers > 0
        and active_layers + disabled_layers == expected_layers
        and deterministic_fallback_layers == disabled_layers
        and atomic_fallback_layers == 0
    )
    segmented_only_exact = bool(
        expected_layers > 0
        and active_layers == 0
        and deterministic_fallback_layers == expected_layers
        and atomic_fallback_layers == 0
        and (
            (not enabled and disabled_layers == 0)
            or (enabled and disabled_layers == expected_layers)
        )
    )
    failure_diagnostics_exact = bool(
        (disabled_layers == 0 and not failure)
        or (disabled_layers > 0 and bool(failure))
    )
    failure_details_exact = bool(
        "gemma4_long_padded_bmm_prefill_failures" not in runtime
        or len(failures) == disabled_layers
    )
    exact = bool(
        not expected
        or (
            (
                not enabled
                and segmented_only_exact
                and segmented_long_config_exact
                and async_tile_contract_exact
                and sorted_partial_contract_exact
            )
            or (
                enabled
                and long_rows == 16_384
                and down_output_dtype == "fp32"
                and route_pack == "argsort"
                and route_pack_block == 256
                and max_padding_ratio == 2.0
                and fused_activation
                and activation_block == 512
                and reduce_block_n == 256
                and reduce_num_warps == 4
                and align_m == 16
                and (
                    (hits > 0 and deterministic_coverage_exact)
                    or segmented_only_exact
                )
                and failure_diagnostics_exact
                and failure_details_exact
                and async_tile_contract_exact
                and sorted_partial_contract_exact
            )
        )
    )
    if segmented_only_exact and not enabled:
        coverage_mode = "segmented_deterministic_skew_gated"
    elif segmented_only_exact:
        coverage_mode = "segmented_deterministic_fallback"
    elif active_layers == expected_layers and disabled_layers == 0:
        coverage_mode = "padded_bmm_fp32"
    elif deterministic_coverage_exact:
        coverage_mode = "padded_bmm_fp32_with_segmented_deterministic_fallback"
    else:
        coverage_mode = "invalid"
    return {
        "exact": exact,
        "expected": expected,
        "enabled": enabled,
        "long_rows": long_rows,
        "down_output_dtype": down_output_dtype,
        "route_pack": route_pack,
        "route_pack_block": route_pack_block,
        "max_padding_ratio": max_padding_ratio,
        "fused_activation": fused_activation,
        "activation_block": activation_block,
        "reduce_block_n": reduce_block_n,
        "reduce_num_warps": reduce_num_warps,
        "align_m": align_m,
        "hits": hits,
        "active_layers": active_layers,
        "expected_layers": expected_layers,
        "disabled_layers": disabled_layers,
        "deterministic_fallback_layers": deterministic_fallback_layers,
        "atomic_fallback_layers": atomic_fallback_layers,
        "segmented_long_rows": segmented_long_rows,
        "segmented_long_config": segmented_long_config,
        "segmented_long_config_exact": segmented_long_config_exact,
        "async_tile_limit": async_tile_limit,
        "async_tile_expected_assignments": async_tile_expected_assignments,
        "async_tile_requested": async_tile_requested,
        "async_tile_hits": async_tile_hits,
        "async_tile_contract_exact": async_tile_contract_exact,
        "sorted_partial_requested": sorted_partial_requested,
        "sorted_partial_hits": sorted_partial_hits,
        "sorted_partial_layers": sorted_partial_layers,
        "sorted_partial_contract_exact": sorted_partial_contract_exact,
        "deterministic_coverage_exact": deterministic_coverage_exact,
        "segmented_only_exact": segmented_only_exact,
        "coverage_mode": coverage_mode,
        "failure": failure,
        "failures": failures,
        "failure_diagnostics_exact": failure_diagnostics_exact,
        "failure_details_exact": failure_details_exact,
        "shape": {"batch_size": int(batch_size), "context": int(context)},
    }


def megagemm_long_decode_burst_contract(
    graph: dict[str, Any],
    runtime: dict[str, Any],
    *,
    max_tokens: int,
    burst_size: int = 8,
) -> dict[str, Any]:
    decode_steps = max(0, int(max_tokens) - 1)
    expected_bursts = math.ceil(decode_steps / int(burst_size))
    expected_feedback_copies = max(0, decode_steps - expected_bursts)
    graph_exact = bool(
        graph.get("enabled", False)
        and graph.get("prefer_step", False)
        and graph.get("shape_cache", False)
        and not graph.get("shared_shape_cache", False)
        and int(graph.get("failures", 0) or 0) == 0
        and int(graph.get("replays", 0) or 0) > 0
        and int(graph.get("physical_rebinds", 0) or 0) == 0
        and int(graph.get("greedy_token_shape_graphs", 0) or 0) > 0
        and graph.get("token_burst_enabled", False)
        and int(graph.get("token_burst_size", 0) or 0) == int(burst_size)
        and int(graph.get("token_burst_steps", 0) or 0) == decode_steps
        and int(graph.get("token_bursts", 0) or 0) == expected_bursts
        and int(graph.get("greedy_token_steps", 0) or 0) == decode_steps
        and int(graph.get("batched_token_host_copies", 0) or 0)
        == expected_bursts
        and not graph.get("persistent_token_feedback_enabled", False)
        and int(graph.get("persistent_token_feedback_steps", 0) or 0) == 0
        and int(graph.get("token_feedback_copies", 0) or 0)
        == expected_feedback_copies
        and int(graph.get("vectorized_input_updates", 0) or 0)
        == expected_bursts
        and int(graph.get("chain_input_updates_skipped", 0) or 0) == 0
    )
    softcap_exact = bool(
        runtime.get("gemma4_batch_cublas_lm_head_enabled", False)
        and int(runtime.get("gemma4_batch_cublas_lm_head_hits", 0) or 0) > 0
        and runtime.get("gemma4_batch_fused_softcap_argmax_enabled", False)
        and runtime.get("gemma4_batch_fused_softcap_argmax_available", False)
        and int(runtime.get("gemma4_batch_fused_softcap_argmax_hits", 0) or 0) > 0
        and not runtime.get("gemma4_batch_fused_softcap_argmax_disabled", False)
        and not str(
            runtime.get("gemma4_batch_fused_softcap_argmax_error", "") or ""
        )
    )
    return {
        "exact": bool(graph_exact and softcap_exact),
        "graph_exact": graph_exact,
        "softcap_exact": softcap_exact,
        "burst_size": int(burst_size),
        "expected_decode_steps": decode_steps,
        "expected_bursts": expected_bursts,
        "expected_feedback_copies": expected_feedback_copies,
        "expected_vectorized_input_updates": expected_bursts,
    }


def normalize_megagemm_row(
    row: dict[str, Any],
    *,
    batch_size: int,
    context: int,
    max_tokens: int,
    decode_mode: str = MEGAGEMM_DECODE_MODE_GRAPH_STEP,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prefill_ms = float(row.get("scheduler_prefill_ms") or 0.0)
    decode_ms = float(row.get("scheduler_decode_ms") or 0.0)
    if prefill_ms <= 0.0 or decode_ms <= 0.0:
        raise RuntimeError(
            f"invalid MegaGemm phase timing: prefill={prefill_ms} decode={decode_ms}"
        )
    selected_prefill_cap = int(
        os.environ.get(
            "MEGAGEMM_GEMMA4_DETERMINISTIC_PREFILL_MAX_BATCHED_TOKENS",
            "0",
        )
        or 0
    )
    if batch_size == 16 and context == 2048 and selected_prefill_cap > 0:
        chunk_contract = megagemm_prefill_chunk_plan_contract(
            row.get("prefill_chunk_plan"),
            batch_size=batch_size,
            context=context,
            token_cap=selected_prefill_cap,
        )
        row["prefill_chunk_plan_contract"] = chunk_contract
        if not chunk_contract["exact"]:
            raise RuntimeError(
                "MegaGemm measured prefill chunk contract failed: "
                + json.dumps(chunk_contract, sort_keys=True)
            )
    graph = dict(row.get("decode_cuda_graphs") or {})
    runtime = dict(row.pop("decode_runtime", {}) or {})
    scheduler_reuse_enabled = bool(
        graph.get("request_scheduler_reuse_enabled", False)
    )
    scheduler_reused = bool(graph.get("request_scheduler_reused", False))
    scheduler_reuse_contract = {
        "advertised": scheduler_reuse_enabled,
        "enabled": scheduler_reuse_enabled,
        "reused": scheduler_reused,
        "reuse_count": int(graph.get("request_scheduler_reuse_count", 0) or 0),
        "captures_this_request": int(graph.get("captures", 0) or 0),
        "warmups_this_request": int(graph.get("warmups", 0) or 0),
        "physical_rebinds_this_request": int(
            graph.get("physical_rebinds", 0) or 0
        ),
    }
    scheduler_reuse_contract["exact"] = bool(
        not scheduler_reused
        or (
            scheduler_reuse_enabled
            and int(graph.get("captures", 0) or 0) == 0
            and int(graph.get("warmups", 0) or 0) == 0
            and int(graph.get("physical_rebinds", 0) or 0) == 0
        )
    )
    if not scheduler_reuse_contract["exact"]:
        raise RuntimeError(
            "MegaGemm persistent request-Scheduler contract failed: "
            + json.dumps(scheduler_reuse_contract, sort_keys=True)
        )
    long_decode_burst: dict[str, Any] | None = None
    if decode_mode == MEGAGEMM_DECODE_MODE_GRAPH_STEP:
        if (
            not graph.get("enabled")
            or bool(graph.get("token_burst_enabled", False))
            or bool(graph.get("shared_shape_cache", False))
            or int(graph.get("failures", 0) or 0) != 0
            or int(graph.get("replays", 0) or 0) <= 0
            or int(graph.get("physical_rebinds", 0) or 0) != 0
        ):
            raise RuntimeError(f"MegaGemm decode graph-step contract failed: {graph}")
        decode_graph_scope = (
            "engine_persistent_scheduler_replay"
            if scheduler_reused
            else "engine_persistent_scheduler_capture"
            if scheduler_reuse_enabled
            else "request_local"
        )
    elif decode_mode == MEGAGEMM_DECODE_MODE_GRAPH_BURST:
        long_decode_burst = megagemm_long_decode_burst_contract(
            graph,
            runtime,
            max_tokens=max_tokens,
        )
        if not long_decode_burst["exact"]:
            raise RuntimeError(
                "MegaGemm decode graph-burst contract failed: "
                + json.dumps(
                    {"contract": long_decode_burst, "graph": graph},
                    sort_keys=True,
                )
            )
        decode_graph_scope = (
            "engine_persistent_scheduler_burst8_replay"
            if scheduler_reused
            else "engine_persistent_scheduler_burst8_capture"
            if scheduler_reuse_enabled
            else "request_local_burst8_gpu_feedback"
        )
    elif decode_mode == MEGAGEMM_DECODE_MODE_EAGER:
        if (
            bool(graph.get("enabled", False))
            or int(graph.get("captures", 0) or 0) != 0
            or int(graph.get("replays", 0) or 0) != 0
            or int(graph.get("physical_rebinds", 0) or 0) != 0
        ):
            raise RuntimeError(f"MegaGemm eager decode contract failed: {graph}")
        decode_graph_scope = "disabled"
    else:
        raise ValueError(f"unsupported MegaGemm decode mode: {decode_mode}")
    deterministic_moe = megagemm_deterministic_moe_contract(runtime)
    if not deterministic_moe["exact"]:
        raise RuntimeError(
            "MegaGemm deterministic MoE runtime contract failed: "
            + json.dumps(deterministic_moe, sort_keys=True)
        )
    compact_active_list = megagemm_compact_active_list_contract(
        runtime,
        batch_size=batch_size,
    )
    if not compact_active_list["exact"]:
        raise RuntimeError(
            "MegaGemm compact active-list runtime contract failed: "
            + json.dumps(compact_active_list, sort_keys=True)
        )
    long_sliding_prefill = megagemm_long_sliding_prefill_contract(
        runtime,
        batch_size=batch_size,
        context=context,
    )
    if not long_sliding_prefill["exact"]:
        raise RuntimeError(
            "MegaGemm long sliding-prefill runtime contract failed: "
            + json.dumps(long_sliding_prefill, sort_keys=True)
        )
    long_decode_attention = megagemm_long_decode_attention_contract(
        runtime,
        batch_size=batch_size,
        context=context,
    )
    if not long_decode_attention["exact"]:
        raise RuntimeError(
            "MegaGemm long decode-attention runtime contract failed: "
            + json.dumps(long_decode_attention, sort_keys=True)
        )
    long_full_prefill = megagemm_long_full_prefill_contract(
        runtime,
        batch_size=batch_size,
        context=context,
    )
    if not long_full_prefill["exact"]:
        raise RuntimeError(
            "MegaGemm long full-prefill runtime contract failed: "
            + json.dumps(long_full_prefill, sort_keys=True)
        )
    long_attention_prepare = megagemm_long_attention_prepare_contract(
        runtime,
        batch_size=batch_size,
        context=context,
    )
    if not long_attention_prepare["exact"]:
        raise RuntimeError(
            "MegaGemm long attention-prepare runtime contract failed: "
            + json.dumps(long_attention_prepare, sort_keys=True)
        )
    long_routed_expert_prefill = megagemm_long_routed_expert_prefill_contract(
        runtime,
        batch_size=batch_size,
        context=context,
    )
    if not long_routed_expert_prefill["exact"]:
        raise RuntimeError(
            "MegaGemm long routed-expert prefill runtime contract failed: "
            + json.dumps(long_routed_expert_prefill, sort_keys=True)
        )
    prefill_tokens = batch_size * context
    decode_tokens = batch_size * (max_tokens - 1)
    row.update(
        {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "decode_tokens": decode_tokens,
            "decode_tok_s": decode_tokens / (decode_ms / 1000.0),
            "decode_measurement_method": "scheduler_phase_wall_time",
            "decode_execution_mode": decode_mode,
            "decode_graph_scope": decode_graph_scope,
            "long_decode_burst_contract": long_decode_burst,
            "phase_metrics_status": "valid",
            "deterministic_moe_contract": deterministic_moe,
            "compact_active_list_contract": compact_active_list,
            "long_sliding_prefill_contract": long_sliding_prefill,
            "long_decode_attention_contract": long_decode_attention,
            "long_full_prefill_contract": long_full_prefill,
            "long_attention_prepare_contract": long_attention_prepare,
            "long_routed_expert_prefill_contract": long_routed_expert_prefill,
            "request_scheduler_reuse_contract": scheduler_reuse_contract,
            "prefill_tokens": prefill_tokens,
            "prefill_tok_s": prefill_tokens / (prefill_ms / 1000.0),
            "input_plus_output_tok_s": (
                prefill_tokens + batch_size * max_tokens
            )
            / (float(row["total_ms"]) / 1000.0),
        }
    )
    return row, runtime


def normalize_vllm_row(
    row: dict[str, Any],
    *,
    batch_size: int,
    context: int,
    max_tokens: int,
) -> dict[str, Any]:
    prefill_ms = row.get("prefill_ms")
    decode_ms = row.get("decode_ms")
    if prefill_ms is None or float(prefill_ms) <= 0.0:
        raise RuntimeError(
            "vLLM request metrics did not provide a valid long-context TTFT: "
            f"{row.get('phase_metrics_status')} {row.get('phase_metrics_reason')}"
        )
    if decode_ms is None or float(decode_ms) <= 0.0:
        raise RuntimeError("vLLM request metrics did not provide valid decode timing")
    prefill_tokens = batch_size * context
    row.update(
        {
            "prefill_tokens": prefill_tokens,
            "prefill_tok_s": prefill_tokens / (float(prefill_ms) / 1000.0),
            "input_plus_output_tok_s": (
                prefill_tokens + batch_size * max_tokens
            )
            / (float(row["total_ms"]) / 1000.0),
        }
    )
    return row


def run_megagemm_route_normalized_diagnostic(
    engine,
    prompts: list[list[int]],
    *,
    batch_size: int,
    context: int,
    max_tokens: int,
    repeats: int,
    decode_mode: str,
    require_scheduler_reuse: bool = False,
) -> dict[str, Any]:
    token_id = route_normalized_token_id(prompts)
    configure_megagemm_decode_mode(decode_mode, getattr(engine, "model", None))
    with megagemm_benchmark_forced_token(token_id):
        warmup_request = run_megagemm_request(engine, prompts, max_tokens)
        warmup_active_experts = megagemm_compact_active_expert_snapshot(
            getattr(engine, "model", None)
        )
        warmup, warmup_runtime = normalize_megagemm_row(
            warmup_request,
            batch_size=batch_size,
            context=context,
            max_tokens=max_tokens,
            decode_mode=decode_mode,
        )
        warmup_contract = forced_token_matrix_contract(
            warmup["token_ids"],
            token_id=token_id,
            expected_rows=batch_size,
            expected_tokens=max_tokens,
        )
        if not warmup_contract["exact"]:
            raise RuntimeError(
                "MegaGemm route-normalized warmup token contract failed: "
                + json.dumps(warmup_contract, sort_keys=True)
            )
        if int(warmup.get("benchmark_forced_token_id", -1)) != token_id:
            raise RuntimeError("MegaGemm Scheduler did not activate forced-token mode")
        warmup["forced_token_contract"] = warmup_contract
        warmup["compact_active_experts"] = warmup_active_experts
        warmup_topology = megagemm_runtime_topology(warmup_runtime)

        samples: list[dict[str, Any]] = []
        for index in range(int(repeats)):
            measured_request = run_megagemm_request(engine, prompts, max_tokens)
            measured_active_experts = megagemm_compact_active_expert_snapshot(
                getattr(engine, "model", None)
            )
            row, runtime = normalize_megagemm_row(
                measured_request,
                batch_size=batch_size,
                context=context,
                max_tokens=max_tokens,
                decode_mode=decode_mode,
            )
            reuse_contract = row["request_scheduler_reuse_contract"]
            if require_scheduler_reuse and not (
                reuse_contract["enabled"]
                and reuse_contract["reused"]
                and reuse_contract["exact"]
            ):
                raise RuntimeError(
                    "MegaGemm route-normalized request recaptured decode instead "
                    "of reusing its Scheduler: "
                    + json.dumps(reuse_contract, sort_keys=True)
                )
            row["repeat"] = index + 1
            row["compact_active_experts"] = measured_active_experts
            row["forced_token_contract"] = forced_token_matrix_contract(
                row["token_ids"],
                token_id=token_id,
                expected_rows=batch_size,
                expected_tokens=max_tokens,
            )
            row["runtime_topology"] = megagemm_runtime_topology(runtime)
            row["runtime_topology_contract"] = runtime_topology_comparison(
                warmup_topology,
                row["runtime_topology"],
            )
            if not row["forced_token_contract"]["exact"]:
                raise RuntimeError(
                    "MegaGemm route-normalized measured token contract failed: "
                    + json.dumps(row["forced_token_contract"], sort_keys=True)
                )
            if not row["runtime_topology_contract"]["exact"]:
                raise RuntimeError(
                    "MegaGemm route-normalized runtime topology changed: "
                    + json.dumps(row["runtime_topology_contract"], sort_keys=True)
                )
            samples.append(row)

    active_expert_summary = summarize_compact_active_expert_snapshots(
        [warmup_active_experts]
        + [dict(row["compact_active_experts"]) for row in samples]
    )
    return {
        "enabled": True,
        "excluded_from_natural_measurement": True,
        "policy": ROUTE_NORMALIZED_POLICY,
        "implementation": "natural_argmax_then_in_graph_token_fill",
        "forced_token_id": token_id,
        "warmup": warmup,
        "warmup_runtime_topology": warmup_topology,
        "samples": samples,
        "summary": summarize_long(samples),
        "compact_active_expert_summary": active_expert_summary,
    }


def run_vllm_route_normalized_diagnostic(
    llm,
    prompts: list[list[int]],
    *,
    batch_size: int,
    context: int,
    max_tokens: int,
    repeats: int,
) -> dict[str, Any]:
    token_id = route_normalized_token_id(prompts)
    warmup = normalize_vllm_row(
        run_vllm_request(
            llm,
            prompts,
            max_tokens,
            allowed_token_id=token_id,
        ),
        batch_size=batch_size,
        context=context,
        max_tokens=max_tokens,
    )
    warmup["forced_token_contract"] = forced_token_matrix_contract(
        warmup["token_ids"],
        token_id=token_id,
        expected_rows=batch_size,
        expected_tokens=max_tokens,
    )
    if not warmup["forced_token_contract"]["exact"]:
        raise RuntimeError("vLLM route-normalized warmup token contract failed")

    samples: list[dict[str, Any]] = []
    for index in range(int(repeats)):
        row = normalize_vllm_row(
            run_vllm_request(
                llm,
                prompts,
                max_tokens,
                allowed_token_id=token_id,
            ),
            batch_size=batch_size,
            context=context,
            max_tokens=max_tokens,
        )
        row["repeat"] = index + 1
        row["forced_token_contract"] = forced_token_matrix_contract(
            row["token_ids"],
            token_id=token_id,
            expected_rows=batch_size,
            expected_tokens=max_tokens,
        )
        if not row["forced_token_contract"]["exact"]:
            raise RuntimeError(
                "vLLM route-normalized measured token contract failed: "
                + json.dumps(row["forced_token_contract"], sort_keys=True)
            )
        samples.append(row)

    return {
        "enabled": True,
        "excluded_from_natural_measurement": True,
        "policy": ROUTE_NORMALIZED_POLICY,
        "implementation": "vllm_sampling_params_allowed_token_ids",
        "forced_token_id": token_id,
        "warmup": warmup,
        "samples": samples,
        "summary": summarize_long(samples),
    }


def decode_stage_breakdown(timing: dict[str, Any]) -> dict[str, Any]:
    path = str(timing.get("decode_path") or "unknown")
    if path == "flat":
        leaf_keys = (
            "embed_ms",
            "flat_norm1_ms",
            "flat_qkv_ms",
            "flat_rope_kv_ms",
            "flat_attn_core_ms",
            "flat_o_proj_ms",
            "flat_resid_norm_ms",
            "flat_moe_ms",
            "flat_residual_ms",
            "lm_head_ms",
            "sample_ms",
        )
    else:
        leaf_keys = (
            "embed_ms",
            "attn_input_norm_ms",
            "attn_qkv_ms",
            "attn_norm_rope_ms",
            "attn_kv_write_ms",
            "attn_core_ms",
            "attn_o_proj_ms",
            "moe_router_ms",
            "moe_experts_ms",
            "mlp_output_norm_ms",
            "ple_ms",
            "lm_head_ms",
            "sample_ms",
        )
    steps = max(1, int(timing.get("steps", 1) or 1))
    total_ms = float(timing.get("total_ms", 0.0) or 0.0)
    ranking = []
    for key in leaf_keys:
        value = float(timing.get(key, 0.0) or 0.0)
        if value <= 0.0:
            continue
        ranking.append(
            {
                "stage": key.removesuffix("_ms"),
                "ms": value,
                "ms_per_step": value / steps,
                "share_of_total": value / total_ms if total_ms > 0.0 else None,
            }
        )
    ranking.sort(key=lambda item: float(item["ms"]), reverse=True)
    leaf_total_ms = sum(float(item["ms"]) for item in ranking)
    return {
        "decode_path": path,
        "steps": steps,
        "batch_size": int(timing.get("batch_size", 0) or 0),
        "top_level_ms": {
            key: float(timing.get(key, 0.0) or 0.0)
            for key in ("embed_ms", "decode_body_ms", "lm_head_ms", "sample_ms")
        },
        "total_ms": total_ms,
        "ms_per_step": total_ms / steps,
        "leaf_stage_ranking": ranking,
        "leaf_accounted_ms": leaf_total_ms,
        "leaf_accounted_ratio": (
            leaf_total_ms / total_ms if total_ms > 0.0 else None
        ),
    }


def run_excluded_megagemm_decode_stage_profile(
    engine,
    prompts: list[list[int]],
    *,
    token_id: int,
    selected_decode_mode: str,
) -> dict[str, Any]:
    import megagemm.models.llama as llama_model

    previous_globals = {
        "_DECODE_TIMING": bool(getattr(llama_model, "_DECODE_TIMING", False)),
        "_DECODE_TIMING_PRINT": bool(
            getattr(llama_model, "_DECODE_TIMING_PRINT", True)
        ),
        "_DECODE_TIMING_DETAIL": bool(
            getattr(llama_model, "_DECODE_TIMING_DETAIL", False)
        ),
    }
    timing_env_names = (
        "MEGAGEMM_DECODE_TIMING",
        "MEGAGEMM_DECODE_TIMING_PRINT",
        "MEGAGEMM_DECODE_TIMING_DETAIL",
    )
    previous_env = {name: os.environ.get(name) for name in timing_env_names}
    profile_tokens = DECODE_STAGE_PROFILE_STEPS + 1
    timing: dict[str, Any] = {}
    row: dict[str, Any] = {}
    try:
        llama_model._DECODE_TIMING = True
        llama_model._DECODE_TIMING_PRINT = False
        llama_model._DECODE_TIMING_DETAIL = True
        os.environ["MEGAGEMM_DECODE_TIMING"] = "1"
        os.environ["MEGAGEMM_DECODE_TIMING_PRINT"] = "0"
        os.environ["MEGAGEMM_DECODE_TIMING_DETAIL"] = "1"
        configure_megagemm_decode_mode(
            MEGAGEMM_DECODE_MODE_EAGER,
            getattr(engine, "model", None),
        )
        with megagemm_benchmark_forced_token(token_id):
            row = run_megagemm_request(engine, prompts, profile_tokens)
        timing = dict(engine.model.get_last_decode_timing() or {})
    finally:
        for name, value in previous_globals.items():
            setattr(llama_model, name, value)
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        configure_megagemm_decode_mode(
            selected_decode_mode,
            getattr(engine, "model", None),
        )

    token_contract = forced_token_matrix_contract(
        row.get("token_ids") or [],
        token_id=token_id,
        expected_rows=len(prompts),
        expected_tokens=profile_tokens,
    )
    if not token_contract["exact"]:
        raise RuntimeError(
            "excluded MegaGemm decode profile token contract failed: "
            + json.dumps(token_contract, sort_keys=True)
        )
    if (
        int(timing.get("steps", 0) or 0) != DECODE_STAGE_PROFILE_STEPS
        or int(timing.get("batch_size", 0) or 0) != len(prompts)
    ):
        raise RuntimeError(
            "excluded MegaGemm decode profile did not execute one exact burst: "
            + json.dumps(timing, sort_keys=True)
        )
    return {
        "excluded": True,
        "measurement": "cuda_events_eager_multistep_after_paid_samples",
        "forced_token_id": int(token_id),
        "profile_output_tokens": profile_tokens,
        "token_contract": token_contract,
        "scheduler_decode_ms": float(row.get("scheduler_decode_ms") or 0.0),
        "timing": timing,
        "breakdown": decode_stage_breakdown(timing),
    }


def _first_generated_tokens(token_ids: list[list[int]]) -> list[list[int]]:
    if not token_ids or any(not row for row in token_ids):
        raise RuntimeError("token matrix has no first generated token")
    return [[int(row[0])] for row in token_ids]


def megagemm_runtime_topology(runtime: dict[str, Any]) -> dict[str, Any]:
    """Return the path-selection state that must remain fixed after warmup."""
    markers = (
        "_disabled",
        "_failure",
        "_fail_reason",
        "_skip_reason",
        "_last_active_layers",
        "_deterministic_reduce_layers",
        "_atomic_reduce_layers",
        "_policy",
    )
    fields = {
        str(key): value
        for key, value in runtime.items()
        if str(key).endswith("_enabled")
        or any(marker in str(key) for marker in markers)
        or str(key) == "gemma4_grouped_mm_prefill_selected_layers"
    }
    encoded = json.dumps(
        fields,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "fields": json.loads(encoded.decode("utf-8")),
    }


def runtime_topology_comparison(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    reference_fields = dict(reference.get("fields") or {})
    candidate_fields = dict(candidate.get("fields") or {})
    changed = {}
    for key in sorted(set(reference_fields) | set(candidate_fields)):
        left = reference_fields.get(key)
        right = candidate_fields.get(key)
        if left != right:
            changed[key] = {"reference": left, "candidate": right}
    return {
        "exact": not changed,
        "reference_sha256": str(reference.get("sha256") or ""),
        "candidate_sha256": str(candidate.get("sha256") or ""),
        "changed_fields": changed,
    }


def evaluate_megagemm_warmup_stability(
    warmups: list[dict[str, Any]],
    *,
    minimum_warmups: int = MEGAGEMM_MIN_WARMUPS,
    maximum_warmups: int = MEGAGEMM_MAX_WARMUPS,
    required_stable_pairs: int = MEGAGEMM_REQUIRED_STABLE_WARMUP_PAIRS,
    maximum_last_pair_ratio: float = MEGAGEMM_WARMUP_MAX_LAST_PAIR_RATIO,
) -> dict[str, Any]:
    """Require repeated token, timing, and runtime-path stability before timing."""
    completed = len(warmups)
    transitions: list[dict[str, Any]] = []
    for previous, current in zip(warmups, warmups[1:]):
        previous_ms = float(previous["total_ms"])
        current_ms = float(current["total_ms"])
        timing_ratio = max(previous_ms, current_ms) / max(
            min(previous_ms, current_ms),
            1e-9,
        )
        tokens = token_matrix_comparison(
            previous["token_ids"],
            current["token_ids"],
        )
        topology = runtime_topology_comparison(
            previous["runtime_topology"],
            current["runtime_topology"],
        )
        transitions.append(
            {
                "stable": bool(
                    tokens["exact"]
                    and topology["exact"]
                    and timing_ratio <= float(maximum_last_pair_ratio)
                ),
                "timing_ratio": float(timing_ratio),
                "tokens": tokens,
                "runtime_topology": topology,
            }
        )

    consecutive_stable_pairs = 0
    for transition in reversed(transitions):
        if not transition["stable"]:
            break
        consecutive_stable_pairs += 1

    enough_samples = completed >= int(minimum_warmups)
    stable = bool(
        enough_samples
        and consecutive_stable_pairs >= int(required_stable_pairs)
    )
    budget_exhausted = completed >= int(maximum_warmups)
    last = transitions[-1] if transitions else None
    if completed < 2:
        reason = "need_two_samples"
    elif not enough_samples:
        reason = "minimum_warmups_not_reached"
    elif last is not None and not last["tokens"]["exact"]:
        reason = "last_pair_tokens_changed"
    elif last is not None and not last["runtime_topology"]["exact"]:
        reason = "last_pair_runtime_topology_changed"
    elif (
        last is not None
        and float(last["timing_ratio"]) > float(maximum_last_pair_ratio)
    ):
        reason = "last_pair_timing_unstable"
    elif not stable:
        reason = "more_stable_pairs_required"
    else:
        reason = "stable"
    return {
        "stable": stable,
        "accepted": stable,
        "reason": reason,
        "completed_warmups": completed,
        "minimum_warmups": int(minimum_warmups),
        "maximum_warmups": int(maximum_warmups),
        "required_stable_pairs": int(required_stable_pairs),
        "consecutive_stable_pairs": int(consecutive_stable_pairs),
        "budget_exhausted": bool(budget_exhausted),
        "maximum_last_pair_ratio": float(maximum_last_pair_ratio),
        "last_pair_total_ratio": (
            None if last is None else float(last["timing_ratio"])
        ),
        "last_pair_tokens": None if last is None else last["tokens"],
        "last_pair_runtime_topology": (
            None if last is None else last["runtime_topology"]
        ),
    }


def measured_token_contract(
    warmup_reference: list[list[int]],
    measured_reference: list[list[int]] | None,
    candidate: list[list[int]],
) -> tuple[dict[str, Any], list[list[int]]]:
    """Validate steady-state tokens without treating graph capture as measured output."""
    full_vs_warmup = token_matrix_comparison(warmup_reference, candidate)
    first_vs_warmup = token_matrix_comparison(
        _first_generated_tokens(warmup_reference),
        _first_generated_tokens(candidate),
    )
    reference = candidate if measured_reference is None else measured_reference
    steady_state = token_matrix_comparison(reference, candidate)
    return (
        {
            "full_tokens_vs_excluded_warmup": full_vs_warmup,
            "first_token_vs_excluded_warmup": first_vs_warmup,
            "steady_state_vs_first_measured": steady_state,
        },
        reference,
    )


def _raise_measured_token_failure(
    contract: dict[str, Any],
    *,
    label: str,
    require_full_warmup_match: bool = False,
) -> None:
    full_warmup = contract["full_tokens_vs_excluded_warmup"]
    if require_full_warmup_match and not full_warmup["exact"]:
        raise RuntimeError(
            f"{label} greedy tokens changed after stabilized warmup: "
            + json.dumps(full_warmup, sort_keys=True)
        )
    first_token = contract["first_token_vs_excluded_warmup"]
    if not first_token["exact"]:
        raise RuntimeError(
            f"{label} first generated token changed after excluded warmup: "
            + json.dumps(first_token, sort_keys=True)
        )
    steady_state = contract["steady_state_vs_first_measured"]
    if not steady_state["exact"]:
        raise RuntimeError(
            f"{label} greedy tokens changed across measured repeats: "
            + json.dumps(steady_state, sort_keys=True)
        )


def _megagemm_warmup_settings(
    args: argparse.Namespace,
) -> tuple[int, int, int, float]:
    minimum = max(
        int(args.warmups),
        int(getattr(args, "megagemm_min_warmups", MEGAGEMM_MIN_WARMUPS)),
    )
    maximum = int(
        getattr(args, "megagemm_max_warmups", MEGAGEMM_MAX_WARMUPS)
    )
    required_pairs = int(
        getattr(
            args,
            "megagemm_required_stable_warmup_pairs",
            MEGAGEMM_REQUIRED_STABLE_WARMUP_PAIRS,
        )
    )
    maximum_ratio = float(
        getattr(
            args,
            "megagemm_warmup_max_last_pair_ratio",
            MEGAGEMM_WARMUP_MAX_LAST_PAIR_RATIO,
        )
    )
    return minimum, maximum, required_pairs, maximum_ratio


def configure_megagemm_decode_mode(mode: str, model: Any | None = None) -> None:
    """Configure decode while retaining only a compatible idle graph owner."""
    use_burst = mode == MEGAGEMM_DECODE_MODE_GRAPH_BURST
    if mode in {
        MEGAGEMM_DECODE_MODE_GRAPH_STEP,
        MEGAGEMM_DECODE_MODE_GRAPH_BURST,
    }:
        os.environ["MEGAGEMM_DECODE_CUDA_GRAPHS"] = "1"
    elif mode == MEGAGEMM_DECODE_MODE_EAGER:
        os.environ["MEGAGEMM_DECODE_CUDA_GRAPHS"] = "0"
    else:
        raise ValueError(f"unsupported MegaGemm decode mode: {mode}")
    os.environ["MEGAGEMM_DECODE_GRAPH_TOKEN_BURST"] = "1" if use_burst else "0"
    os.environ["MEGAGEMM_MULTI_STEP_BURST_BATCH"] = "8"
    os.environ["MEGAGEMM_GEMMA4_BATCH_CUBLAS_LM_HEAD"] = "1"
    os.environ["MEGAGEMM_GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX"] = (
        "1" if use_burst else "0"
    )
    os.environ["MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK"] = "0"
    os.environ["MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE"] = "0"
    os.environ["MEGAGEMM_REUSE_REQUEST_SCHEDULER"] = "1"

    import megagemm.models.llama as llama_model

    previous_softcap_mode = bool(
        llama_model._GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX
    )
    llama_model._GEMMA4_BATCH_CUBLAS_LM_HEAD = True
    llama_model._GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX = use_burst
    # Python counters advance while a graph is captured, not when it is replayed.
    # Preserve that capture evidence when the same burst graph remains active.
    if model is not None and previous_softcap_mode != use_burst:
        model._gemma4_batch_cublas_lm_head_hits = 0
        model._gemma4_batch_fused_softcap_argmax_hits = 0
        model._gemma4_batch_fused_softcap_argmax_disable = False
        model._gemma4_batch_fused_softcap_argmax_error = ""


def megagemm_decode_mode_candidates(
    *,
    batch_size: int,
    context: int,
    max_tokens: int = 64,
    long_decode_burst_proven: bool = False,
) -> list[str]:
    if (
        long_decode_burst_proven
        and int(batch_size) == 16
        and int(context) == 2048
        and int(max_tokens) == 64
    ):
        return [
            MEGAGEMM_DECODE_MODE_GRAPH_BURST,
            MEGAGEMM_DECODE_MODE_GRAPH_STEP,
            MEGAGEMM_DECODE_MODE_EAGER,
        ]
    modes = [MEGAGEMM_DECODE_MODE_GRAPH_STEP]
    if batch_size == 16 and context >= 2048:
        modes.append(MEGAGEMM_DECODE_MODE_EAGER)
    return modes


def megagemm_execution_candidates(
    *,
    batch_size: int,
    context: int,
    determinism_auto_fallback: bool,
    max_tokens: int = 64,
    long_decode_burst_proven: bool = False,
) -> list[dict[str, str]]:
    modes = megagemm_decode_mode_candidates(
        batch_size=batch_size,
        context=context,
        max_tokens=max_tokens,
        long_decode_burst_proven=long_decode_burst_proven,
    )
    del determinism_auto_fallback
    segmented = [
        {
            "decode_mode": mode,
            "prefill_profile": MEGAGEMM_PREFILL_PROFILE_SEGMENTED,
        }
        for mode in modes
    ]
    if (
        modes
        and modes[0] == MEGAGEMM_DECODE_MODE_GRAPH_BURST
        and int(batch_size) == 16
        and int(context) == 2048
        and int(max_tokens) == 64
        and env_flag(
            "MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_PREFILL",
            False,
        )
    ):
        return [
            {
                "decode_mode": MEGAGEMM_DECODE_MODE_GRAPH_BURST,
                "prefill_profile": MEGAGEMM_PREFILL_PROFILE_HYBRID,
            },
            *segmented,
        ]
    return segmented


def force_segmented_long_prefill(model, reason: str) -> dict[str, Any]:
    """Disable guarded long-prefill candidates without rebuilding the model."""
    disabled = 0
    seen: set[int] = set()
    for layer in list(getattr(model, "layers", []) or []):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if experts is None or id(experts) in seen:
            continue
        has_dominant = hasattr(
            experts,
            "_gemma4_long_dominant_expert_prefill_disabled",
        )
        has_legacy_padded = hasattr(
            experts,
            "_gemma4_long_padded_bmm_prefill_disabled",
        )
        if not (has_dominant or has_legacy_padded):
            continue
        seen.add(id(experts))
        if has_dominant:
            experts._gemma4_long_dominant_expert_prefill_disabled = True
            experts._gemma4_long_dominant_expert_prefill_fail_reason = str(reason)
            experts._gemma4_long_dominant_expert_prefill_last_active = False
            experts._gemma4_long_dominant_expert_prefill_last_guard_reason = ""
        if has_legacy_padded:
            experts._gemma4_long_padded_bmm_prefill_disabled = True
            experts._gemma4_long_padded_bmm_prefill_fail_reason = str(reason)
            experts._gemma4_long_padded_bmm_prefill_last_active = False
        for workspace_name in (
            "_gemma4_long_dominant_expert_prefill_workspace",
            "_gemma4_long_padded_bmm_prefill_workspace",
            "_segmented_prefill_workspace",
        ):
            workspace = getattr(experts, workspace_name, None)
            if isinstance(workspace, dict):
                workspace.clear()
        disabled += 1
    if disabled <= 0:
        raise RuntimeError(
            "deterministic segmented fallback found no Gemma4 expert layers"
        )
    return {
        "prefill_profile": MEGAGEMM_PREFILL_PROFILE_SEGMENTED,
        "disabled_layers": disabled,
        "reason": str(reason),
    }


def enable_guarded_padded_long_prefill(model) -> dict[str, Any]:
    """Restore the guarded dominant-expert path after an excluded reference."""
    enabled = 0
    seen: set[int] = set()
    for layer in list(getattr(model, "layers", []) or []):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if experts is None or id(experts) in seen:
            continue
        has_dominant = hasattr(
            experts,
            "_gemma4_long_dominant_expert_prefill_disabled",
        )
        has_legacy_padded = hasattr(
            experts,
            "_gemma4_long_padded_bmm_prefill_disabled",
        )
        if not (has_dominant or has_legacy_padded):
            continue
        seen.add(id(experts))
        if has_dominant:
            experts._gemma4_long_dominant_expert_prefill_disabled = False
            experts._gemma4_long_dominant_expert_prefill_fail_reason = ""
            experts._gemma4_long_dominant_expert_prefill_last_active = False
            experts._gemma4_long_dominant_expert_prefill_last_guard_reason = ""
            experts._gemma4_long_dominant_expert_prefill_hits = 0
            experts._gemma4_long_dominant_expert_prefill_assignments = 0
            experts._gemma4_long_dominant_expert_prefill_guard_misses = 0
        if has_legacy_padded:
            experts._gemma4_long_padded_bmm_prefill_disabled = False
            experts._gemma4_long_padded_bmm_prefill_fail_reason = ""
            experts._gemma4_long_padded_bmm_prefill_last_active = False
            experts._gemma4_long_padded_bmm_prefill_hits = 0
            experts._gemma4_long_padded_bmm_prefill_assignments = 0
        for workspace_name in (
            "_gemma4_long_dominant_expert_prefill_workspace",
            "_gemma4_long_padded_bmm_prefill_workspace",
            "_segmented_prefill_workspace",
        ):
            workspace = getattr(experts, workspace_name, None)
            if isinstance(workspace, dict):
                workspace.clear()
        enabled += 1
    if enabled <= 0:
        raise RuntimeError(
            "guarded dominant-expert restore found no Gemma4 expert layers"
        )
    return {
        "prefill_profile": MEGAGEMM_PREFILL_PROFILE_HYBRID,
        "enabled_layers": enabled,
    }


def evaluate_guarded_padded_prefill_promotion(
    segmented_reference_rows: list[dict[str, Any]],
    candidate_warmups: list[dict[str, Any]],
    *,
    minimum_speedup: float = MEGAGEMM_PADDED_PREFILL_MINIMUM_SPEEDUP,
) -> dict[str, Any]:
    """Gate the guarded long-prefill candidate on exact tokens and TTFT."""
    if len(segmented_reference_rows) < 2:
        raise ValueError("two segmented reference rows are required")
    if not candidate_warmups:
        raise ValueError("at least one guarded prefill candidate warmup is required")

    reference_tokens = segmented_reference_rows[-1]["token_ids"]
    reference_stability = token_matrix_comparison(
        segmented_reference_rows[-2]["token_ids"],
        reference_tokens,
    )
    candidate_token_contracts = [
        token_matrix_comparison(reference_tokens, row["token_ids"])
        for row in candidate_warmups
    ]
    candidate_tokens_exact = bool(
        reference_stability["exact"]
        and all(contract["exact"] for contract in candidate_token_contracts)
    )

    last_moe_contract = dict(
        candidate_warmups[-1].get("deterministic_moe_contract") or {}
    )
    padded_layers = int(last_moe_contract.get("padded_bmm_prefill_layers", 0) or 0)
    dominant_layers = int(
        last_moe_contract.get("dominant_expert_prefill_layers", 0) or 0
    )
    dominant_disabled = int(
        last_moe_contract.get("dominant_expert_prefill_disabled_layers", 0) or 0
    )
    dominant_active = dominant_layers > 0
    candidate_layers = dominant_layers + padded_layers
    prefill_layers = int(last_moe_contract.get("prefill_layers", 0) or 0)
    covered_layers = int(last_moe_contract.get("prefill_covered_layers", 0) or 0)
    candidate_path_exact = bool(
        last_moe_contract.get("exact", False)
        and candidate_layers > 0
        and prefill_layers > 0
        and covered_layers == prefill_layers
        and dominant_disabled == 0
    )

    stable_window = candidate_warmups[-min(3, len(candidate_warmups)) :]
    reference_window = segmented_reference_rows[
        -min(3, len(segmented_reference_rows)) :
    ]
    reference_prefill_samples_ms = [
        float(row["prefill_ms"]) for row in reference_window
    ]
    reference_prefill_ms = float(statistics.median(reference_prefill_samples_ms))
    candidate_prefill_ms = float(
        statistics.median(float(row["prefill_ms"]) for row in stable_window)
    )
    speedup = reference_prefill_ms / max(candidate_prefill_ms, 1e-9)
    apply_change = bool(
        candidate_tokens_exact
        and candidate_path_exact
        and speedup >= float(minimum_speedup)
    )
    if not reference_stability["exact"]:
        reason = "segmented_reference_tokens_changed"
    elif not candidate_tokens_exact:
        reason = (
            "dominant_expert_tokens_differ_from_segmented_reference"
            if dominant_active
            else "padded_bmm_tokens_differ_from_segmented_reference"
        )
    elif not candidate_path_exact:
        reason = (
            "dominant_expert_did_not_cover_the_deterministic_moe_contract"
            if dominant_active
            else "padded_bmm_did_not_cover_the_deterministic_moe_contract"
        )
    elif speedup < float(minimum_speedup):
        reason = (
            "dominant_expert_prefill_speedup_below_threshold"
            if dominant_active
            else "padded_bmm_prefill_speedup_below_threshold"
        )
    else:
        reason = "exact_and_faster"
    return {
        "accepted": apply_change,
        "decision": (
            "APPLY_DOMINANT_EXPERT_HYBRID"
            if apply_change and dominant_active
            else "APPLY_PADDED_BMM"
            if apply_change
            else "KEEP_SEGMENTED"
        ),
        "candidate": (
            "dense_dominant_plus_light_padded_bmm"
            if dominant_active
            else "global_padded_bmm"
        ),
        "reason": reason,
        "minimum_speedup": float(minimum_speedup),
        "reference_prefill_ms": reference_prefill_ms,
        "reference_prefill_samples_ms": reference_prefill_samples_ms,
        "candidate_prefill_ms": candidate_prefill_ms,
        "candidate_prefill_samples_ms": [
            float(row["prefill_ms"]) for row in stable_window
        ],
        "speedup": float(speedup),
        "segmented_reference_tokens": reference_stability,
        "candidate_tokens_exact": candidate_tokens_exact,
        "candidate_token_contracts": candidate_token_contracts,
        "candidate_path_exact": candidate_path_exact,
        "dominant_expert_prefill_layers": dominant_layers,
        "dominant_expert_prefill_disabled_layers": dominant_disabled,
        "padded_bmm_prefill_layers": padded_layers,
        "prefill_layers": prefill_layers,
        "prefill_covered_layers": covered_layers,
    }


def should_reject_stable_topology_early(stability: dict[str, Any]) -> bool:
    """Stop repeating a mode once stable topology still changes tokens."""
    topology = dict(stability.get("last_pair_runtime_topology") or {})
    return bool(
        int(stability.get("completed_warmups", 0) or 0)
        >= int(stability.get("minimum_warmups", MEGAGEMM_MIN_WARMUPS) or 0)
        and stability.get("reason") == "last_pair_tokens_changed"
        and topology.get("exact") is True
    )


def should_reject_graph_step_early(stability: dict[str, Any]) -> bool:
    """Backward-compatible name for the stable-topology rejection gate."""
    return should_reject_stable_topology_early(stability)


def megagemm_prefill_chunk_plan_contract(
    plan: dict[str, Any] | None,
    *,
    batch_size: int,
    context: int,
    token_cap: int,
) -> dict[str, Any]:
    """Validate the exact request grouping used by the long-prefill scheduler."""
    batch = int(batch_size)
    seq_len = int(context)
    cap = int(token_cap)
    if batch <= 0 or seq_len <= 0 or cap < seq_len:
        raise ValueError("invalid prefill chunk contract shape or token cap")

    requests_per_chunk = max(1, min(batch, cap // seq_len))
    expected_request_counts: list[int] = []
    remaining = batch
    while remaining > 0:
        count = min(requests_per_chunk, remaining)
        expected_request_counts.append(count)
        remaining -= count
    expected_prompt_tokens = [count * seq_len for count in expected_request_counts]
    actual = dict(plan or {})
    checks = {
        "present": bool(plan),
        "strategy": str(actual.get("strategy") or "") == "batched_tokens",
        "total_prompt_tokens": int(actual.get("total_prompt_tokens", 0) or 0)
        == batch * seq_len,
        "num_chunks": int(actual.get("num_chunks", 0) or 0)
        == len(expected_request_counts),
        "max_requests": int(actual.get("max_requests", 0) or 0)
        >= max(expected_request_counts),
        "max_batched_tokens": int(actual.get("max_batched_tokens", 0) or 0)
        == cap,
        "deterministic_moe_token_cap": int(
            actual.get("deterministic_moe_token_cap", 0) or 0
        )
        == cap,
        "chunk_prompt_tokens": list(actual.get("chunk_prompt_tokens") or [])
        == expected_prompt_tokens,
        "chunk_request_counts": list(actual.get("chunk_request_counts") or [])
        == expected_request_counts,
    }
    return {
        "exact": all(checks.values()),
        "checks": checks,
        "token_cap": cap,
        "expected": {
            "total_prompt_tokens": batch * seq_len,
            "num_chunks": len(expected_request_counts),
            "chunk_prompt_tokens": expected_prompt_tokens,
            "chunk_request_counts": expected_request_counts,
        },
        "actual": actual,
    }


def _set_megagemm_prefill_chunk_token_cap(token_cap: int) -> None:
    cap = str(int(token_cap))
    os.environ["MEGAGEMM_PREFILL_MAX_BATCHED_TOKENS"] = cap
    os.environ[
        "MEGAGEMM_GEMMA4_DETERMINISTIC_PREFILL_MAX_BATCHED_TOKENS"
    ] = cap


@contextmanager
def megagemm_prefill_chunk_gate_mode(token_cap: int):
    keys = (
        "MEGAGEMM_PREFILL_MAX_BATCHED_TOKENS",
        "MEGAGEMM_GEMMA4_DETERMINISTIC_PREFILL_MAX_BATCHED_TOKENS",
        "MEGAGEMM_DECODE_CUDA_GRAPHS",
        "MEGAGEMM_DECODE_GRAPH_TOKEN_BURST",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        _set_megagemm_prefill_chunk_token_cap(token_cap)
        os.environ["MEGAGEMM_DECODE_CUDA_GRAPHS"] = "0"
        os.environ["MEGAGEMM_DECODE_GRAPH_TOKEN_BURST"] = "0"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _discard_megagemm_request_scheduler(engine) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    engine._last_scheduler = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_megagemm_prefill_chunk_gate_workspaces(engine) -> None:
    for layer in getattr(engine.model, "layers", ()):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        workspace = getattr(experts, "_segmented_prefill_workspace", None)
        if isinstance(workspace, dict):
            workspace.clear()
        gate = getattr(mlp, "gate", None)
        topk_workspaces = getattr(gate, "_prefill_topk_workspaces", None)
        if isinstance(topk_workspaces, dict):
            topk_workspaces.clear()


def _snapshot_megagemm_prefill_chunk_gate_state(engine) -> list[tuple[Any, dict[str, Any]]]:
    tracked: list[tuple[Any, dict[str, Any]]] = []

    def capture(owner: Any, names: tuple[str, ...]) -> None:
        if owner is None:
            return
        values = {
            name: getattr(owner, name)
            for name in names
            if hasattr(owner, name)
        }
        if values:
            tracked.append((owner, values))

    for layer in getattr(engine.model, "layers", ()):
        attention = getattr(layer, "self_attn", None)
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        capture(
            attention,
            (
                "_gemma4_fused_attn_prepare_disabled",
                "_gemma4_fused_attn_prepare_skip_reason",
            ),
        )
        capture(
            experts,
            (
                "_segmented_prefill_disabled",
                "_segmented_prefill_fail_reason",
            ),
        )

    paged_attention_module = sys.modules.get("megagemm.kernels.paged_attention")
    capture(
        paged_attention_module,
        (
            "_GEMMA4_LONG_SLIDING_PREFILL_DISABLED",
            "_GEMMA4_LONG_SLIDING_PREFILL_FAILURE",
            "_GEMMA4_LONG_SLIDING_PREFILL_LOGGED",
            "_GEMMA4_LONG_FULL_PREFILL_DISABLED",
            "_GEMMA4_LONG_FULL_PREFILL_FAILURE",
            "_GEMMA4_LONG_FULL_PREFILL_LOGGED",
        ),
    )
    return tracked


def _restore_megagemm_prefill_chunk_gate_state(
    engine,
    tracked: list[tuple[Any, dict[str, Any]]],
) -> None:
    for owner, values in tracked:
        for name, value in values.items():
            setattr(owner, name, value)
    _clear_megagemm_prefill_chunk_gate_workspaces(engine)


_PREFILL_CHUNK_GATE_RUNTIME_COUNTERS = (
    "gemma4_long_sliding_prefill_hits",
    "gemma4_long_full_prefill_hits",
    "gemma4_fused_qkv_prefill_hits",
    "gemma4_fused_attn_prepare_hits",
    "qwen3_moe_segmented_prefill_total_hits",
)


def _prefill_chunk_gate_runtime_snapshot(engine) -> dict[str, Any]:
    return dict(engine.model.decode_runtime_stats() or {})


def _prefill_chunk_gate_runtime_contract(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    deltas = {
        key: int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        for key in _PREFILL_CHUNK_GATE_RUNTIME_COUNTERS
    }
    checks = {
        "long_sliding": bool(after.get("gemma4_long_sliding_prefill_enabled", False))
        and deltas["gemma4_long_sliding_prefill_hits"] > 0,
        "long_full": bool(after.get("gemma4_long_full_prefill_enabled", False))
        and deltas["gemma4_long_full_prefill_hits"] > 0,
        "fused_qkv": bool(after.get("gemma4_fused_qkv_prefill_enabled", False))
        and deltas["gemma4_fused_qkv_prefill_hits"] > 0,
        "fused_attention_prepare": bool(
            after.get("gemma4_fused_attn_prepare_enabled", False)
        )
        and deltas["gemma4_fused_attn_prepare_hits"] > 0
        and int(after.get("gemma4_fused_attn_prepare_disabled_layers", 0) or 0)
        == 0,
        "segmented_moe": deltas["qwen3_moe_segmented_prefill_total_hits"] > 0
        and int(after.get("qwen3_moe_segmented_prefill_disabled_layers", 0) or 0)
        == 0
        and not str(
            after.get("qwen3_moe_segmented_prefill_first_failure", "") or ""
        ),
    }
    return {
        "exact": all(checks.values()),
        "checks": checks,
        "hit_deltas": deltas,
    }


def _run_megagemm_prefill_chunk_gate_group(
    engine,
    prompts: list[list[int]],
    *,
    token_cap: int,
    repeats: int,
) -> dict[str, Any]:
    batch_size = len(prompts)
    context = len(prompts[0]) if prompts else 0
    if not prompts or any(len(row) != context for row in prompts):
        raise ValueError("prefill chunk gate requires a non-empty rectangular prompt matrix")

    runtime_before = _prefill_chunk_gate_runtime_snapshot(engine)
    warmup = run_megagemm_request(engine, prompts, 2)
    warmup_plan = megagemm_prefill_chunk_plan_contract(
        warmup.get("prefill_chunk_plan"),
        batch_size=batch_size,
        context=context,
        token_cap=token_cap,
    )
    if not warmup_plan["exact"]:
        raise RuntimeError(
            "MegaGemm prefill chunk warmup contract failed: "
            + json.dumps(warmup_plan, sort_keys=True)
        )

    rows: list[dict[str, Any]] = []
    plan_contracts: list[dict[str, Any]] = []
    for _ in range(int(repeats)):
        row = run_megagemm_request(engine, prompts, 2)
        plan_contract = megagemm_prefill_chunk_plan_contract(
            row.get("prefill_chunk_plan"),
            batch_size=batch_size,
            context=context,
            token_cap=token_cap,
        )
        if not plan_contract["exact"]:
            raise RuntimeError(
                "MegaGemm prefill chunk measured contract failed: "
                + json.dumps(plan_contract, sort_keys=True)
            )
        rows.append(row)
        plan_contracts.append(plan_contract)

    runtime_after = _prefill_chunk_gate_runtime_snapshot(engine)
    runtime_contract = _prefill_chunk_gate_runtime_contract(
        runtime_before,
        runtime_after,
    )
    prefill_samples = [float(row.get("scheduler_prefill_ms") or 0.0) for row in rows]
    if any(sample <= 0.0 for sample in prefill_samples):
        raise RuntimeError(f"invalid prefill timing samples: {prefill_samples}")
    token_reference = list(rows[0].get("token_ids") or [])
    token_contracts = [
        token_matrix_comparison(token_reference, list(row.get("token_ids") or []))
        for row in rows
    ]
    spread_ratio = max(prefill_samples) / min(prefill_samples)
    tokens_exact = all(contract["exact"] for contract in token_contracts)
    stable = bool(
        tokens_exact
        and runtime_contract["exact"]
        and spread_ratio <= MEGAGEMM_PREFILL_CHUNK_MAXIMUM_SPREAD_RATIO
    )
    return {
        "error": None,
        "token_cap": int(token_cap),
        "warmup_excluded": True,
        "warmup_prefill_ms": float(warmup.get("scheduler_prefill_ms") or 0.0),
        "warmup_plan_contract": warmup_plan,
        "samples_ms": prefill_samples,
        "median_ms": float(statistics.median(prefill_samples)),
        "spread_ratio": float(spread_ratio),
        "maximum_spread_ratio": MEGAGEMM_PREFILL_CHUNK_MAXIMUM_SPREAD_RATIO,
        "token_ids": token_reference,
        "token_contracts": token_contracts,
        "tokens_exact": tokens_exact,
        "plan_contracts": plan_contracts,
        "runtime_contract": runtime_contract,
        "stable": stable,
    }


def evaluate_megagemm_prefill_chunk_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_recheck: dict[str, Any],
    *,
    minimum_speedup: float = MEGAGEMM_PREFILL_CHUNK_MINIMUM_SPEEDUP,
) -> dict[str, Any]:
    baseline_ms = float(baseline.get("median_ms") or 0.0)
    candidate_ms = float(candidate.get("median_ms") or 0.0)
    recheck_ms = float(baseline_recheck.get("median_ms") or 0.0)
    baseline_stability_ratio = (
        max(baseline_ms, recheck_ms) / min(baseline_ms, recheck_ms)
        if baseline_ms > 0.0 and recheck_ms > 0.0
        else math.inf
    )
    baseline_tokens = token_matrix_comparison(
        list(baseline.get("token_ids") or []),
        list(baseline_recheck.get("token_ids") or []),
    )
    candidate_tokens = token_matrix_comparison(
        list(baseline.get("token_ids") or []),
        list(candidate.get("token_ids") or []),
    )
    speedup = baseline_ms / candidate_ms if candidate_ms > 0.0 else 0.0
    apply_change = bool(
        baseline.get("stable") is True
        and baseline_recheck.get("stable") is True
        and candidate.get("stable") is True
        and baseline_stability_ratio
        <= MEGAGEMM_PREFILL_CHUNK_MAXIMUM_SPREAD_RATIO
        and baseline_tokens["exact"]
        and candidate_tokens["exact"]
        and speedup >= float(minimum_speedup)
    )
    if candidate.get("error"):
        reason = "candidate_error"
    elif not baseline.get("stable") or not baseline_recheck.get("stable"):
        reason = "baseline_not_stable"
    elif baseline_stability_ratio > MEGAGEMM_PREFILL_CHUNK_MAXIMUM_SPREAD_RATIO:
        reason = "baseline_recheck_timing_changed"
    elif not baseline_tokens["exact"]:
        reason = "baseline_recheck_tokens_changed"
    elif not candidate.get("stable"):
        reason = "candidate_not_stable_or_runtime_path_inexact"
    elif not candidate_tokens["exact"]:
        reason = "candidate_tokens_changed"
    elif speedup < float(minimum_speedup):
        reason = "candidate_speedup_below_threshold"
    else:
        reason = "exact_stable_and_faster"
    selected_cap = (
        MEGAGEMM_PREFILL_CHUNK_CANDIDATE_TOKENS
        if apply_change
        else MEGAGEMM_PREFILL_CHUNK_BASELINE_TOKENS
    )
    return {
        "apply_change": apply_change,
        "decision": "APPLY_32768" if apply_change else "KEEP_16384",
        "reason": reason,
        "selected_token_cap": selected_cap,
        "minimum_speedup": float(minimum_speedup),
        "speedup": float(speedup),
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms if candidate_ms > 0.0 else None,
        "baseline_recheck_ms": recheck_ms,
        "baseline_stability_ratio": float(baseline_stability_ratio),
        "maximum_baseline_stability_ratio": (
            MEGAGEMM_PREFILL_CHUNK_MAXIMUM_SPREAD_RATIO
        ),
        "estimated_savings_ms_per_request": (
            float(baseline_ms - candidate_ms) if candidate_ms > 0.0 else 0.0
        ),
        "baseline_recheck_tokens": baseline_tokens,
        "candidate_tokens": candidate_tokens,
    }


def _cuda_error_invalidates_context(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "illegal memory access",
            "device-side assert",
            "unspecified launch failure",
            "streamcaptureinvalidated",
        )
    )


def run_megagemm_prefill_chunk_autotune(
    engine,
    prompts: list[list[int]],
    *,
    repeats: int,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    candidate_state: list[tuple[Any, dict[str, Any]]] | None = None
    sequence = (
        ("baseline_16384", MEGAGEMM_PREFILL_CHUNK_BASELINE_TOKENS),
        ("candidate_32768", MEGAGEMM_PREFILL_CHUNK_CANDIDATE_TOKENS),
        ("baseline_recheck_16384", MEGAGEMM_PREFILL_CHUNK_BASELINE_TOKENS),
    )
    for name, token_cap in sequence:
        _clear_megagemm_prefill_chunk_gate_workspaces(engine)
        _discard_megagemm_request_scheduler(engine)
        if name == "candidate_32768":
            candidate_state = _snapshot_megagemm_prefill_chunk_gate_state(engine)
        try:
            with megagemm_prefill_chunk_gate_mode(token_cap):
                groups[name] = _run_megagemm_prefill_chunk_gate_group(
                    engine,
                    prompts,
                    token_cap=token_cap,
                    repeats=repeats,
                )
        except Exception as exc:
            if name != "candidate_32768" or _cuda_error_invalidates_context(exc):
                raise
            groups[name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "token_cap": int(token_cap),
                "stable": False,
                "samples_ms": [],
                "median_ms": None,
                "token_ids": [],
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            if name == "candidate_32768" and candidate_state is not None:
                _restore_megagemm_prefill_chunk_gate_state(
                    engine,
                    candidate_state,
                )

    decision = evaluate_megagemm_prefill_chunk_gate(
        groups["baseline_16384"],
        groups["candidate_32768"],
        groups["baseline_recheck_16384"],
    )
    if not groups["baseline_16384"].get("stable") or not groups[
        "baseline_recheck_16384"
    ].get("stable"):
        raise RuntimeError(
            "MegaGemm 16k prefill baseline failed its loaded-checkpoint gate"
        )
    _set_megagemm_prefill_chunk_token_cap(decision["selected_token_cap"])
    _clear_megagemm_prefill_chunk_gate_workspaces(engine)
    _discard_megagemm_request_scheduler(engine)
    return {
        "enabled": True,
        "shape": {
            "batch_size": len(prompts),
            "context": len(prompts[0]) if prompts else 0,
        },
        "gate_output_tokens": 2,
        "repeats": int(repeats),
        "groups": groups,
        **decision,
    }


def _base_result(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    minimum, maximum, required_pairs, maximum_ratio = _megagemm_warmup_settings(
        args
    )
    return {
        "schema_version": 1,
        "status": "partial",
        "backend": args.backend,
        "model": args.model,
        "dtype": args.dtype,
        "contexts": args.contexts,
        "batch_sizes": args.batch_sizes,
        "max_seq_len": args.max_seq_len,
        "max_tokens": args.max_tokens,
        "vllm_max_num_batched_tokens": int(
            getattr(args, "max_num_batched_tokens", args.max_seq_len)
        ),
        "warmups": args.warmups,
        "megagemm_warmup_gate": {
            "minimum_warmups": minimum,
            "maximum_warmups": maximum,
            "required_stable_pairs": required_pairs,
            "maximum_last_pair_ratio": maximum_ratio,
        },
        "megagemm_determinism_auto_fallback": bool(
            getattr(args, "megagemm_determinism_auto_fallback", False)
        ),
        "megagemm_prefill_chunk_autotune_requested": bool(
            getattr(args, "megagemm_prefill_chunk_autotune", False)
        ),
        "megagemm_prefill_chunk_gate_repeats": int(
            getattr(args, "megagemm_prefill_chunk_gate_repeats", 3)
        ),
        "megagemm_prefill_stage_profile_requested": bool(
            getattr(args, "profile_prefill_stages", False)
        ),
        "repeats": args.repeats,
        "route_normalized_diagnostic": bool(
            getattr(args, "route_normalized_diagnostic", False)
        ),
        "route_normalized_repeats": int(
            getattr(
                args,
                "route_normalized_repeats",
                ROUTE_NORMALIZED_DEFAULT_REPEATS,
            )
        ),
        "require_request_scheduler_reuse": bool(
            getattr(args, "require_request_scheduler_reuse", False)
        ),
        "route_normalized_policy": ROUTE_NORMALIZED_POLICY,
        "token_reference_policy": "first_measured_repeat",
        "warmup_token_policy": (
            "adaptive_full_matrix_runtime_topology_and_timing_stability"
            if args.backend == "megagemm"
            else "excluded_full_matrix_with_first_token_guard"
        ),
        "prompt_manifest_generator": manifest.get("generator"),
        "prompt_contracts": {
            key: value["contract"]
            for key, value in (manifest.get("cases") or {}).items()
            if int(key) in args.contexts
        },
        "gpu": gpu_snapshot(),
        "cases": {},
    }


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def _load_megagemm_resume_result(
    args: argparse.Namespace,
    prompts: dict[int, list[list[int]]],
    manifest: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    if not out_path.is_file():
        raise FileNotFoundError(
            f"--resume requires an existing MegaGemm result: {out_path}"
        )

    result = json.loads(out_path.read_text(encoding="utf-8"))
    expected_prompt_contracts = {
        key: value["contract"]
        for key, value in (manifest.get("cases") or {}).items()
        if int(key) in args.contexts
    }
    minimum, maximum, required_pairs, maximum_ratio = _megagemm_warmup_settings(
        args
    )
    expected = {
        "schema_version": 1,
        "backend": "megagemm",
        "model": args.model,
        "dtype": args.dtype,
        "contexts": args.contexts,
        "batch_sizes": args.batch_sizes,
        "max_seq_len": args.max_seq_len,
        "max_tokens": args.max_tokens,
        "vllm_max_num_batched_tokens": int(args.max_num_batched_tokens),
        "warmups": args.warmups,
        "megagemm_warmup_gate": {
            "minimum_warmups": minimum,
            "maximum_warmups": maximum,
            "required_stable_pairs": required_pairs,
            "maximum_last_pair_ratio": maximum_ratio,
        },
        "megagemm_determinism_auto_fallback": bool(
            getattr(args, "megagemm_determinism_auto_fallback", False)
        ),
        "repeats": args.repeats,
        "route_normalized_diagnostic": bool(
            getattr(args, "route_normalized_diagnostic", False)
        ),
        "route_normalized_repeats": int(
            getattr(
                args,
                "route_normalized_repeats",
                ROUTE_NORMALIZED_DEFAULT_REPEATS,
            )
        ),
        "route_normalized_policy": ROUTE_NORMALIZED_POLICY,
        "token_reference_policy": "first_measured_repeat",
        "warmup_token_policy": (
            "adaptive_full_matrix_runtime_topology_and_timing_stability"
        ),
        "prompt_manifest_generator": manifest.get("generator"),
        "prompt_contracts": expected_prompt_contracts,
    }
    mismatches = [
        field for field, value in expected.items() if result.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            "refusing to resume an incompatible MegaGemm result; mismatched "
            "fields: " + ", ".join(sorted(mismatches))
        )

    cases = result.get("cases")
    if not isinstance(cases, dict):
        raise ValueError("refusing to resume: existing result has no cases object")
    for context in args.contexts:
        for batch_size in args.batch_sizes:
            key = case_key(batch_size, context)
            case = cases.get(key)
            if not isinstance(case, dict) or case.get("status") != "complete":
                continue
            expected_contract = prompt_token_contract(prompts[context][:batch_size])
            if (
                int(case.get("batch_size", -1)) != batch_size
                or int(case.get("context", -1)) != context
                or case.get("prompt_contract") != expected_contract
            ):
                raise ValueError(
                    f"refusing to resume: completed case {key} does not match "
                    "the current prompt/shape contract"
                )
    return result


def run_megagemm_sweep(
    args: argparse.Namespace,
    prompts: dict[int, list[list[int]]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    os.environ.setdefault("MEGAGEMM_FP16_STREAMING", "1")
    os.environ.setdefault("MEGAGEMM_FLAT_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_DISABLE_CUDA_RMSNORM", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_SHAPE_CACHE", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE", "0")
    os.environ.setdefault("MEGAGEMM_REUSE_REQUEST_SCHEDULER", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_STABLE_MAX_BLOCKS", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_GRAPH_TOKEN_BURST", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    out_path = Path(args.out_json)
    resume = bool(getattr(args, "resume", False))
    result = (
        _load_megagemm_resume_result(args, prompts, manifest, out_path)
        if resume
        else _base_result(args, manifest)
    )
    requested_keys = [
        case_key(batch_size, context)
        for context in args.contexts
        for batch_size in args.batch_sizes
    ]
    skipped_keys = [
        key
        for key in requested_keys
        if (result.get("cases", {}).get(key) or {}).get("status") == "complete"
    ]
    pending_keys = [key for key in requested_keys if key not in skipped_keys]
    runtime_gpu = dict(
        (result.get("resume") or {}).get("current_gpu")
        or result.get("gpu")
        or {}
    )
    long_decode_burst_requested = env_flag(
        "MEGAGEMM_GEMMA4_B16_LONG_GRAPH_TOKEN_BURST_PROVEN"
    )
    long_decode_burst_proven = bool(
        long_decode_burst_requested
        and str(args.dtype).lower() == "bf16"
        and int(args.max_tokens) == 64
        and "A100" in str(runtime_gpu.get("name") or "").upper()
    )
    result["long_decode_burst_promotion"] = {
        "requested": long_decode_burst_requested,
        "runtime_proven": long_decode_burst_proven,
        "gpu": str(runtime_gpu.get("name") or ""),
        "dtype": str(args.dtype),
        "max_tokens": int(args.max_tokens),
        "eligible_shape": {"batch_size": 16, "context": 2048},
        "burst_size": 8,
        "feedback": "gpu_to_gpu_between_graph_replays",
        "persistent_token_feedback": False,
    }
    result["status"] = "partial" if pending_keys else "complete"
    if resume:
        result["resume"] = {
            "enabled": True,
            "skipped_complete_cases": skipped_keys,
            "pending_cases": pending_keys,
            "current_gpu": gpu_snapshot(),
        }
    _write_result(out_path, result)
    if resume:
        print(
            "MEGAGEMM_RESUME "
            + json.dumps(result["resume"], sort_keys=True)
        )
    if not pending_keys:
        print("MEGAGEMM_RESUME_COMPLETE: no GPU model load or benchmark required")
        return result

    from megagemm.engine import InferenceEngine

    engine = InferenceEngine(
        args.model,
        device="cuda",
        dtype=dtype_from_name(args.dtype),
        max_seq_len=args.max_seq_len,
        max_batch_size=max(args.batch_sizes),
        deterministic=True,
    )

    chunk_gate_requested = bool(
        getattr(args, "megagemm_prefill_chunk_autotune", False)
    )
    chunk_gate_eligible = bool(
        16 in args.batch_sizes
        and 2048 in args.contexts
        and case_key(16, 2048) in pending_keys
    )
    if chunk_gate_requested and chunk_gate_eligible:
        print(
            "\n== MegaGemm B=16 context=2048 loaded-checkpoint "
            "prefill chunk gate (excluded) =="
        )
        chunk_gate = run_megagemm_prefill_chunk_autotune(
            engine,
            prompts[2048][:16],
            repeats=int(args.megagemm_prefill_chunk_gate_repeats),
        )
        result["megagemm_prefill_chunk_autotune"] = chunk_gate
        print(
            "MEGAGEMM_PREFILL_CHUNK_GATE "
            + json.dumps(
                {
                    key: chunk_gate[key]
                    for key in (
                        "decision",
                        "reason",
                        "selected_token_cap",
                        "speedup",
                        "baseline_ms",
                        "candidate_ms",
                        "baseline_recheck_ms",
                        "estimated_savings_ms_per_request",
                    )
                },
                sort_keys=True,
            )
        )
        _write_result(out_path, result)
    elif chunk_gate_requested:
        result["megagemm_prefill_chunk_autotune"] = {
            "enabled": False,
            "reason": "B16/C2048 is not a pending benchmark case",
        }
        _write_result(out_path, result)

    for context in args.contexts:
        for batch_size in args.batch_sizes:
            key = case_key(batch_size, context)
            if key in skipped_keys:
                print(f"MEGAGEMM_RESUME_SKIP {key}: existing case is complete")
                continue
            case_prompts = prompts[context][:batch_size]
            print(
                f"\n== MegaGemm B={batch_size} context={context} "
                "adaptive stabilization warmup (excluded) =="
            )
            minimum_warmups, maximum_warmups, required_pairs, maximum_ratio = (
                _megagemm_warmup_settings(args)
            )
            execution_candidates = megagemm_execution_candidates(
                batch_size=batch_size,
                context=context,
                max_tokens=args.max_tokens,
                long_decode_burst_proven=long_decode_burst_proven,
                determinism_auto_fallback=bool(
                    getattr(args, "megagemm_determinism_auto_fallback", False)
                ),
            )
            determinism_cost_guard = bool(
                getattr(args, "megagemm_determinism_auto_fallback", False)
                and int(batch_size) == 16
                and int(context) == 2048
            )
            decode_mode_candidates = [
                candidate["decode_mode"] for candidate in execution_candidates
            ]
            decode_mode_attempts: list[dict[str, Any]] = []
            selected_decode_mode: str | None = None
            selected_prefill_profile: str | None = None
            prefill_fallback: dict[str, Any] | None = None
            segmented_prefill_reference_rows: list[dict[str, Any]] = []
            segmented_prefill_reference_tokens: list[list[int]] | None = None
            prefill_promotion_gate: dict[str, Any] | None = None
            warmup_rows: list[dict[str, Any]] = []
            runtime: dict[str, Any] = {}
            warmup_stability: dict[str, Any] = {}
            case_result = {
                "status": "warming",
                "batch_size": batch_size,
                "context": context,
                "prompt_contract": prompt_token_contract(case_prompts),
                "capture_run_excluded": True,
                "decode_mode_candidates": decode_mode_candidates,
                "execution_candidates": execution_candidates,
                "decode_mode_attempts": decode_mode_attempts,
                "decode_execution_mode": None,
                "long_decode_burst_promoted": bool(
                    long_decode_burst_proven
                    and int(batch_size) == 16
                    and int(context) == 2048
                    and int(args.max_tokens) == 64
                ),
                "prefill_profile": None,
                "prefill_fallback": None,
                "segmented_prefill_reference": None,
                "prefill_promotion_gate": None,
                "warmups": warmup_rows,
                "warmup_stability": warmup_stability,
                "samples": [],
                "summary": None,
                "runtime": runtime,
                "route_normalized_diagnostic": None,
                "excluded_prefill_stage_profile": None,
                "excluded_decode_stage_profile": None,
            }
            result["cases"][key] = case_result
            _write_result(out_path, result)
            requires_segmented_prefill_reference = bool(
                execution_candidates
                and execution_candidates[0]["decode_mode"]
                == MEGAGEMM_DECODE_MODE_GRAPH_BURST
                and execution_candidates[0]["prefill_profile"]
                == MEGAGEMM_PREFILL_PROFILE_HYBRID
                and int(batch_size) == 16
                and int(context) == 2048
                and int(args.max_tokens) == 64
            )
            if requires_segmented_prefill_reference:
                configure_megagemm_decode_mode(
                    MEGAGEMM_DECODE_MODE_GRAPH_BURST,
                    getattr(engine, "model", None),
                )
                reference_profile = force_segmented_long_prefill(
                    engine.model,
                    "excluded exact reference for guarded dominant-expert promotion",
                )
                print(
                    "MEGAGEMM_SEGMENTED_PREFILL_REFERENCE_START "
                    + json.dumps(reference_profile, sort_keys=True)
                )
                reference_stability: dict[str, Any] = {}
                case_result["segmented_prefill_reference"] = {
                    "excluded": True,
                    "status": "warming",
                    "runs": segmented_prefill_reference_rows,
                    "stability": reference_stability,
                }
                _write_result(out_path, result)
                for reference_index in range(maximum_warmups):
                    reference_row, reference_runtime = normalize_megagemm_row(
                        run_megagemm_request(
                            engine,
                            case_prompts,
                            args.max_tokens,
                        ),
                        batch_size=batch_size,
                        context=context,
                        max_tokens=args.max_tokens,
                        decode_mode=MEGAGEMM_DECODE_MODE_GRAPH_BURST,
                    )
                    reference_row["reference"] = reference_index + 1
                    reference_row["runtime_topology"] = megagemm_runtime_topology(
                        reference_runtime
                    )
                    segmented_prefill_reference_rows.append(reference_row)
                    reference_stability = evaluate_megagemm_warmup_stability(
                        segmented_prefill_reference_rows,
                        minimum_warmups=minimum_warmups,
                        maximum_warmups=maximum_warmups,
                        required_stable_pairs=required_pairs,
                        maximum_last_pair_ratio=maximum_ratio,
                    )
                    case_result["segmented_prefill_reference"]["stability"] = (
                        reference_stability
                    )
                    _write_result(out_path, result)
                    print(
                        "MEGAGEMM_SEGMENTED_PREFILL_REFERENCE "
                        + json.dumps(
                            {
                                "reference": reference_index + 1,
                                "total_ms": reference_row["total_ms"],
                                "prefill_ms": reference_row["prefill_ms"],
                                "decode_ms": reference_row["decode_ms"],
                                "prefill_backend": reference_row[
                                    "deterministic_moe_contract"
                                ]["prefill_backend"],
                            },
                            sort_keys=True,
                        )
                    )
                    print(
                        "MEGAGEMM_SEGMENTED_PREFILL_REFERENCE_STABILITY "
                        + json.dumps(reference_stability, sort_keys=True)
                    )
                    if reference_stability["stable"]:
                        break
                if not reference_stability.get("stable", False):
                    raise RuntimeError(
                        "segmented prefill reference did not stabilize within "
                        f"{maximum_warmups} excluded runs: "
                        + json.dumps(reference_stability, sort_keys=True)
                    )
                reference_token_contract = reference_stability["last_pair_tokens"]
                reference_topology_contract = reference_stability[
                    "last_pair_runtime_topology"
                ]
                segmented_prefill_reference_tokens = (
                    segmented_prefill_reference_rows[-1]["token_ids"]
                )
                restore_status = enable_guarded_padded_long_prefill(engine.model)
                case_result["segmented_prefill_reference"] = {
                    "excluded": True,
                    "status": "stable",
                    "runs": segmented_prefill_reference_rows,
                    "stability": reference_stability,
                    "token_contract": reference_token_contract,
                    "runtime_topology_contract": reference_topology_contract,
                    "restore": restore_status,
                }
                _write_result(out_path, result)
                print(
                    "MEGAGEMM_GUARDED_DOMINANT_PREFILL_RESTORED "
                    + json.dumps(restore_status, sort_keys=True)
                )
            for candidate_index, candidate in enumerate(execution_candidates):
                decode_mode = candidate["decode_mode"]
                prefill_profile = candidate["prefill_profile"]
                has_guarded_prefill_candidate = any(
                    item["prefill_profile"] == MEGAGEMM_PREFILL_PROFILE_HYBRID
                    for item in execution_candidates
                )
                if (
                    prefill_profile == MEGAGEMM_PREFILL_PROFILE_SEGMENTED
                    and prefill_fallback is None
                    and has_guarded_prefill_candidate
                ):
                    if prefill_promotion_gate is not None:
                        fallback_reason = (
                            "guarded dominant-expert promotion rejected: "
                            + str(prefill_promotion_gate.get("reason") or "unknown")
                        )
                    else:
                        fallback_reason = (
                            "automatic deterministic fallback after "
                            "stable-topology token divergence"
                        )
                    prefill_fallback = force_segmented_long_prefill(
                        engine.model,
                        fallback_reason,
                    )
                    case_result["prefill_fallback"] = prefill_fallback
                    print(
                        "MEGAGEMM_PREFILL_PROFILE_FALLBACK "
                        + json.dumps(prefill_fallback, sort_keys=True)
                    )
                configure_megagemm_decode_mode(
                    decode_mode,
                    getattr(engine, "model", None),
                )
                print(
                    f"MEGAGEMM_EXECUTION_PROFILE decode={decode_mode} "
                    f"prefill={prefill_profile}"
                )
                warmup_rows = []
                runtime = {}
                warmup_stability = {}
                case_result.update(
                    {
                        "status": "warming",
                        "decode_execution_mode": decode_mode,
                        "prefill_profile": prefill_profile,
                        "warmups": warmup_rows,
                        "warmup_stability": warmup_stability,
                        "samples": [],
                        "summary": None,
                        "runtime": runtime,
                    }
                )
                _write_result(out_path, result)
                fallback_available = candidate_index + 1 < len(execution_candidates)
                for index in range(maximum_warmups):
                    row, runtime = normalize_megagemm_row(
                        run_megagemm_request(engine, case_prompts, args.max_tokens),
                        batch_size=batch_size,
                        context=context,
                        max_tokens=args.max_tokens,
                        decode_mode=decode_mode,
                    )
                    row["warmup"] = index + 1
                    row["runtime_topology"] = megagemm_runtime_topology(runtime)
                    if segmented_prefill_reference_tokens is not None:
                        row[
                            "token_diagnostic_vs_segmented_prefill_reference"
                        ] = token_matrix_comparison(
                            segmented_prefill_reference_tokens,
                            row["token_ids"],
                        )
                    warmup_rows.append(row)
                    warmup_stability = evaluate_megagemm_warmup_stability(
                        warmup_rows,
                        minimum_warmups=minimum_warmups,
                        maximum_warmups=maximum_warmups,
                        required_stable_pairs=required_pairs,
                        maximum_last_pair_ratio=maximum_ratio,
                    )
                    case_result["warmup_stability"] = warmup_stability
                    case_result["runtime"] = runtime
                    _write_result(out_path, result)
                    print(
                        f"warmup={index + 1}/{maximum_warmups} "
                        f"mode={decode_mode} "
                        f"total={row['total_ms']:.1f}ms "
                        f"prefill={row['prefill_ms']:.1f}ms "
                        f"decode={row['decode_ms']:.1f}ms"
                    )
                    moe_contract = row["deterministic_moe_contract"]
                    print(
                        "MEGAGEMM_WARMUP_STABILITY "
                        + json.dumps(
                            {
                                **warmup_stability,
                                "decode_execution_mode": decode_mode,
                                "runtime_topology_sha256": row["runtime_topology"][
                                    "sha256"
                                ],
                                "padded_bmm_prefill_layers": moe_contract[
                                    "padded_bmm_prefill_layers"
                                ],
                                "padded_bmm_prefill_disabled_layers": moe_contract[
                                    "padded_bmm_prefill_disabled_layers"
                                ],
                                "padded_bmm_prefill_failures": moe_contract[
                                    "padded_bmm_prefill_failures"
                                ],
                                "dominant_expert_prefill_layers": moe_contract.get(
                                    "dominant_expert_prefill_layers",
                                    0,
                                ),
                                "dominant_expert_prefill_guard_miss_layers": (
                                    moe_contract.get(
                                        "dominant_expert_prefill_guard_miss_layers",
                                        0,
                                    )
                                ),
                                "dominant_expert_prefill_disabled_layers": (
                                    moe_contract.get(
                                        "dominant_expert_prefill_disabled_layers",
                                        0,
                                    )
                                ),
                            },
                            sort_keys=True,
                        )
                    )
                    if (
                        prefill_profile == MEGAGEMM_PREFILL_PROFILE_HYBRID
                        and segmented_prefill_reference_rows
                    ):
                        reference_contract = row[
                            "token_diagnostic_vs_segmented_prefill_reference"
                        ]
                        moe_contract = row["deterministic_moe_contract"]
                        if (
                            not reference_contract["exact"]
                            or (
                                int(
                                    moe_contract.get(
                                        "dominant_expert_prefill_layers",
                                        0,
                                    )
                                    or 0
                                )
                                + int(
                                    moe_contract.get(
                                        "padded_bmm_prefill_layers",
                                        0,
                                    )
                                    or 0
                                )
                            )
                            <= 0
                        ):
                            prefill_promotion_gate = (
                                evaluate_guarded_padded_prefill_promotion(
                                    segmented_prefill_reference_rows,
                                    warmup_rows,
                                )
                            )
                            warmup_stability = {
                                **warmup_stability,
                                "accepted": False,
                                "stable": False,
                                "mode_rejected_early": True,
                                "reason": prefill_promotion_gate["reason"],
                                "prefill_promotion_gate": prefill_promotion_gate,
                            }
                            case_result["warmup_stability"] = warmup_stability
                            case_result["prefill_promotion_gate"] = (
                                prefill_promotion_gate
                            )
                            _write_result(out_path, result)
                            print(
                                "MEGAGEMM_DOMINANT_PREFILL_PROMOTION "
                                + json.dumps(prefill_promotion_gate, sort_keys=True)
                            )
                            break
                    if warmup_stability["stable"]:
                        if (
                            prefill_profile == MEGAGEMM_PREFILL_PROFILE_HYBRID
                            and segmented_prefill_reference_rows
                        ):
                            prefill_promotion_gate = (
                                evaluate_guarded_padded_prefill_promotion(
                                    segmented_prefill_reference_rows,
                                    warmup_rows,
                                )
                            )
                            case_result["prefill_promotion_gate"] = (
                                prefill_promotion_gate
                            )
                            print(
                                "MEGAGEMM_DOMINANT_PREFILL_PROMOTION "
                                + json.dumps(prefill_promotion_gate, sort_keys=True)
                            )
                            if not prefill_promotion_gate["accepted"]:
                                warmup_stability = {
                                    **warmup_stability,
                                    "accepted": False,
                                    "stable": False,
                                    "reason": prefill_promotion_gate["reason"],
                                    "prefill_promotion_gate": prefill_promotion_gate,
                                }
                                case_result["warmup_stability"] = warmup_stability
                                _write_result(out_path, result)
                                break
                        selected_decode_mode = decode_mode
                        selected_prefill_profile = prefill_profile
                        break
                    reject_for_fallback = bool(
                        fallback_available
                        and decode_mode
                        in {
                            MEGAGEMM_DECODE_MODE_GRAPH_STEP,
                            MEGAGEMM_DECODE_MODE_GRAPH_BURST,
                        }
                    )
                    if (
                        (determinism_cost_guard or reject_for_fallback)
                        and should_reject_stable_topology_early(warmup_stability)
                    ):
                        rejection_reason = (
                            "tokens changed with identical runtime topology"
                        )
                        warmup_stability = {
                            **warmup_stability,
                            "mode_rejected_early": True,
                            "fallback_available": fallback_available,
                            "rejection_reason": rejection_reason,
                        }
                        if fallback_available:
                            warmup_stability["fallback_reason"] = rejection_reason
                        case_result["warmup_stability"] = warmup_stability
                        _write_result(out_path, result)
                        break
                if selected_decode_mode is not None:
                    break
                decode_mode_attempts.append(
                    {
                        "decode_execution_mode": decode_mode,
                        "prefill_profile": prefill_profile,
                        "status": "rejected",
                        "warmups": warmup_rows,
                        "warmup_stability": warmup_stability,
                        "last_runtime_topology": (
                            warmup_rows[-1]["runtime_topology"]
                            if warmup_rows
                            else None
                        ),
                    }
                )
                _write_result(out_path, result)
                if fallback_available:
                    print(
                        "MEGAGEMM_DECODE_MODE_FALLBACK "
                        + json.dumps(
                            {
                                "from": decode_mode,
                                "from_prefill": prefill_profile,
                                "to": execution_candidates[candidate_index + 1][
                                    "decode_mode"
                                ],
                                "to_prefill": execution_candidates[
                                    candidate_index + 1
                                ]["prefill_profile"],
                                "reason": warmup_stability.get("reason"),
                            },
                            sort_keys=True,
                        )
                    )

            if selected_decode_mode is None:
                case_result["status"] = "failed"
                case_result["warmup_stability_failure"] = {
                    "error": (
                        "MegaGemm runtime did not stabilize before the bounded "
                        "warmup budget was exhausted"
                    ),
                    "contract": warmup_stability,
                    "last_runtime_topology": (
                        warmup_rows[-1]["runtime_topology"] if warmup_rows else None
                    ),
                }
                _write_result(out_path, result)
                raise RuntimeError(
                    f"MegaGemm B={batch_size} C={context} did not stabilize "
                    "in any permitted decode mode: "
                    + json.dumps(warmup_stability, sort_keys=True)
                )
            case_result["decode_execution_mode"] = selected_decode_mode
            case_result["prefill_profile"] = selected_prefill_profile
            warmup_reference = warmup_rows[-1]["token_ids"]
            for row in warmup_rows:
                row["token_diagnostic_vs_final_warmup"] = token_matrix_comparison(
                    warmup_reference,
                    row["token_ids"],
                )

            samples: list[dict[str, Any]] = []
            measured_reference: list[list[int]] | None = None
            stabilized_runtime_topology = warmup_rows[-1]["runtime_topology"]
            case_result.update(
                {
                    "status": "measuring",
                    "warmups": warmup_rows,
                    "warmup_stability": warmup_stability,
                    "stabilized_runtime_topology": stabilized_runtime_topology,
                    "samples": samples,
                    "summary": None,
                    "runtime": runtime,
                }
            )
            _write_result(out_path, result)
            for index in range(args.repeats):
                row, runtime = normalize_megagemm_row(
                    run_megagemm_request(engine, case_prompts, args.max_tokens),
                    batch_size=batch_size,
                    context=context,
                    max_tokens=args.max_tokens,
                    decode_mode=selected_decode_mode,
                )
                reuse_contract = row.get("request_scheduler_reuse_contract")
                require_scheduler_reuse = bool(
                    getattr(args, "require_request_scheduler_reuse", False)
                )
                if require_scheduler_reuse and not (
                    reuse_contract
                    and reuse_contract.get("enabled")
                    and reuse_contract.get("reused")
                    and reuse_contract.get("exact")
                ):
                    raise RuntimeError(
                        f"MegaGemm B={batch_size} C={context} measured request "
                        "recaptured decode instead of reusing its Scheduler: "
                        + json.dumps(reuse_contract, sort_keys=True)
                    )
                row["repeat"] = index + 1
                row["runtime_topology"] = megagemm_runtime_topology(runtime)
                row["runtime_topology_contract"] = runtime_topology_comparison(
                    stabilized_runtime_topology,
                    row["runtime_topology"],
                )
                token_contract, measured_reference = measured_token_contract(
                    warmup_reference,
                    measured_reference,
                    row["token_ids"],
                )
                row["token_contract"] = token_contract
                samples.append(row)
                case_result["summary"] = summarize_long(samples)
                case_result["runtime"] = runtime
                if index == 0:
                    case_result["warmup_to_first_measured"] = token_contract[
                        "full_tokens_vs_excluded_warmup"
                    ]
                    print(
                        "MEGAGEMM_STABILIZED_WARMUP_TOKEN_CONTRACT "
                        + json.dumps(
                            token_contract["full_tokens_vs_excluded_warmup"],
                            sort_keys=True,
                        )
                    )
                _write_result(out_path, result)
                print(
                    f"repeat={index + 1}/{args.repeats} "
                    f"total={row['total_ms']:.1f}ms prefill={row['prefill_ms']:.1f}ms "
                    f"decode={row['decode_ms']:.1f}ms"
                )
                try:
                    if not row["runtime_topology_contract"]["exact"]:
                        raise RuntimeError(
                            f"MegaGemm B={batch_size} C={context} runtime topology "
                            "changed after stabilized warmup: "
                            + json.dumps(
                                row["runtime_topology_contract"],
                                sort_keys=True,
                            )
                        )
                    _raise_measured_token_failure(
                        token_contract,
                        label=f"MegaGemm B={batch_size} C={context}",
                        require_full_warmup_match=True,
                    )
                except RuntimeError as exc:
                    case_result["status"] = "failed"
                    case_result["token_stability_failure"] = {
                        "repeat": index + 1,
                        "error": str(exc),
                        "contract": token_contract,
                        "runtime_topology_contract": row[
                            "runtime_topology_contract"
                        ],
                    }
                    _write_result(out_path, result)
                    raise

            if batch_size == 16 and (
                not runtime.get(
                    "gemma4_fused_attn_moe_router_single_kernel_decode_enabled"
                )
                or int(
                    runtime.get(
                        "gemma4_fused_attn_moe_router_single_kernel_decode_hits", 0
                    )
                    or 0
                )
                <= 0
            ):
                raise RuntimeError(
                    "promoted B16 single-kernel router bridge was not exercised"
                )
            route_diagnostic = None
            if bool(getattr(args, "route_normalized_diagnostic", False)):
                print(
                    f"\n== MegaGemm B={batch_size} context={context} "
                    "route-normalized diagnostic (excluded) =="
                )
                route_diagnostic = run_megagemm_route_normalized_diagnostic(
                    engine,
                    case_prompts,
                    batch_size=batch_size,
                    context=context,
                    max_tokens=args.max_tokens,
                    repeats=int(
                        getattr(
                            args,
                            "route_normalized_repeats",
                            ROUTE_NORMALIZED_DEFAULT_REPEATS,
                        )
                    ),
                    decode_mode=selected_decode_mode,
                    require_scheduler_reuse=bool(
                        getattr(args, "require_request_scheduler_reuse", False)
                    ),
                )
                case_result["route_normalized_diagnostic"] = route_diagnostic
                _write_result(out_path, result)
                print(
                    "MEGAGEMM_ROUTE_NORMALIZED "
                    + json.dumps(
                        {
                            "forced_token_id": route_diagnostic[
                                "forced_token_id"
                            ],
                            "summary": route_diagnostic["summary"],
                            "compact_active_experts": route_diagnostic[
                                "compact_active_expert_summary"
                            ],
                        },
                        sort_keys=True,
                    )
                )

            if bool(getattr(args, "profile_prefill_stages", False)) and (
                int(batch_size) == 16 and int(context) == 2048
            ):
                print(
                    "\n== MegaGemm B=16 context=2048 excluded post-measurement "
                    "prefill-stage profile =="
                )
                prefill_profile = run_excluded_megagemm_prefill_profile(
                    engine,
                    case_prompts,
                    measured_reference or [],
                )
                case_result["excluded_prefill_stage_profile"] = prefill_profile
                _write_result(out_path, result)
                print(
                    "MEGAGEMM_EXCLUDED_PREFILL_STAGE_PROFILE "
                    + json.dumps(prefill_profile, sort_keys=True)
                )

            if bool(getattr(args, "profile_decode_stages", False)) and (
                int(batch_size) == 16 and int(context) == 2048
            ):
                token_id = (
                    int(route_diagnostic["forced_token_id"])
                    if route_diagnostic is not None
                    else route_normalized_token_id(case_prompts)
                )
                print(
                    "\n== MegaGemm B=16 context=2048 excluded decode-stage "
                    "profile =="
                )
                stage_profile = run_excluded_megagemm_decode_stage_profile(
                    engine,
                    case_prompts,
                    token_id=token_id,
                    selected_decode_mode=selected_decode_mode,
                )
                case_result["excluded_decode_stage_profile"] = stage_profile
                _write_result(out_path, result)
                print(
                    "MEGAGEMM_EXCLUDED_DECODE_STAGE_PROFILE "
                    + json.dumps(stage_profile["breakdown"], sort_keys=True)
                )
            case_result["status"] = "complete"
            case_result["summary"] = summarize_long(samples)
            case_result["runtime"] = runtime
            _write_result(out_path, result)

    result["status"] = "complete"
    if resume:
        result["resume"]["completed_during_resume"] = pending_keys
        result["resume"]["pending_cases"] = []
    _write_result(out_path, result)
    return result


def run_vllm_sweep(
    args: argparse.Namespace,
    prompts: dict[int, list[list[int]]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    args.max_batch_size = max(args.batch_sizes)
    llm, runtime, version, kwargs = make_vllm(args)
    result = _base_result(args, manifest)
    result.update(
        {
            "version": version,
            "cuda_runtime": runtime,
            "llm_kwargs": kwargs,
        }
    )
    out_path = Path(args.out_json)
    _write_result(out_path, result)
    print(f"vLLM version: {version}")
    print(f"vLLM kwargs: {kwargs}")

    for context in args.contexts:
        for batch_size in args.batch_sizes:
            key = case_key(batch_size, context)
            case_prompts = prompts[context][:batch_size]
            print(
                f"\n== vLLM B={batch_size} context={context} "
                "compile warmup (excluded) =="
            )
            warmup_rows: list[dict[str, Any]] = []
            for index in range(args.warmups):
                row = normalize_vllm_row(
                    run_vllm_request(llm, case_prompts, args.max_tokens),
                    batch_size=batch_size,
                    context=context,
                    max_tokens=args.max_tokens,
                )
                row["warmup"] = index + 1
                warmup_rows.append(row)
                print(
                    f"warmup={index + 1}/{args.warmups} "
                    f"total={row['total_ms']:.1f}ms prefill={row['prefill_ms']:.1f}ms "
                    f"decode={row['decode_ms']:.1f}ms"
                )
            warmup_reference = warmup_rows[-1]["token_ids"]
            for row in warmup_rows:
                row["token_diagnostic_vs_final_warmup"] = token_matrix_comparison(
                    warmup_reference,
                    row["token_ids"],
                )

            samples: list[dict[str, Any]] = []
            measured_reference: list[list[int]] | None = None
            case_result = {
                "status": "measuring",
                "batch_size": batch_size,
                "context": context,
                "prompt_contract": prompt_token_contract(case_prompts),
                "capture_run_excluded": True,
                "warmups": warmup_rows,
                "samples": samples,
                "summary": None,
                "route_normalized_diagnostic": None,
            }
            result["cases"][key] = case_result
            _write_result(out_path, result)
            for index in range(args.repeats):
                row = normalize_vllm_row(
                    run_vllm_request(llm, case_prompts, args.max_tokens),
                    batch_size=batch_size,
                    context=context,
                    max_tokens=args.max_tokens,
                )
                row["repeat"] = index + 1
                token_contract, measured_reference = measured_token_contract(
                    warmup_reference,
                    measured_reference,
                    row["token_ids"],
                )
                row["token_contract"] = token_contract
                samples.append(row)
                case_result["summary"] = summarize_long(samples)
                if index == 0:
                    case_result["warmup_to_first_measured"] = token_contract[
                        "full_tokens_vs_excluded_warmup"
                    ]
                    print(
                        "VLLM_EXCLUDED_WARMUP_TOKEN_DIAGNOSTIC "
                        + json.dumps(
                            token_contract["full_tokens_vs_excluded_warmup"],
                            sort_keys=True,
                        )
                    )
                _write_result(out_path, result)
                print(
                    f"repeat={index + 1}/{args.repeats} "
                    f"total={row['total_ms']:.1f}ms prefill={row['prefill_ms']:.1f}ms "
                    f"decode={row['decode_ms']:.1f}ms"
                )
                try:
                    _raise_measured_token_failure(
                        token_contract,
                        label=f"vLLM B={batch_size} C={context}",
                    )
                except RuntimeError as exc:
                    case_result["status"] = "failed"
                    case_result["token_stability_failure"] = {
                        "repeat": index + 1,
                        "error": str(exc),
                        "contract": token_contract,
                    }
                    _write_result(out_path, result)
                    raise
            if bool(getattr(args, "route_normalized_diagnostic", False)):
                print(
                    f"\n== vLLM B={batch_size} context={context} "
                    "route-normalized diagnostic (excluded) =="
                )
                route_diagnostic = run_vllm_route_normalized_diagnostic(
                    llm,
                    case_prompts,
                    batch_size=batch_size,
                    context=context,
                    max_tokens=args.max_tokens,
                    repeats=int(
                        getattr(
                            args,
                            "route_normalized_repeats",
                            ROUTE_NORMALIZED_DEFAULT_REPEATS,
                        )
                    ),
                )
                case_result["route_normalized_diagnostic"] = route_diagnostic
                _write_result(out_path, result)
                print(
                    "VLLM_ROUTE_NORMALIZED "
                    + json.dumps(
                        {
                            "forced_token_id": route_diagnostic[
                                "forced_token_id"
                            ],
                            "summary": route_diagnostic["summary"],
                        },
                        sort_keys=True,
                    )
                )
            case_result["status"] = "complete"
            case_result["summary"] = summarize_long(samples)
            _write_result(out_path, result)

    result["status"] = "complete"
    _write_result(out_path, result)
    return result


def compare_results(
    megagemm_path: Path,
    vllm_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    mg = json.loads(megagemm_path.read_text(encoding="utf-8"))
    vl = json.loads(vllm_path.read_text(encoding="utf-8"))
    contract_fields = (
        "dtype",
        "contexts",
        "batch_sizes",
        "max_seq_len",
        "max_tokens",
        "vllm_max_num_batched_tokens",
        "prompt_contracts",
        "route_normalized_diagnostic",
        "route_normalized_repeats",
        "route_normalized_policy",
    )
    mismatched = [field for field in contract_fields if mg.get(field) != vl.get(field)]
    if mismatched:
        raise RuntimeError(f"cross-backend contract mismatch: {mismatched}")
    if mg.get("status") != "complete" or vl.get("status") != "complete":
        raise RuntimeError("both backend results must be complete")

    resume_info = mg.get("resume") if isinstance(mg.get("resume"), dict) else {}
    resumed_cases = list(resume_info.get("skipped_complete_cases") or [])
    mg_source_gpu = str((mg.get("gpu") or {}).get("name", "unknown"))
    mg_resume_gpu = str(
        (resume_info.get("current_gpu") or {}).get("name", "unknown")
    )
    vllm_gpu = str((vl.get("gpu") or {}).get("name", "unknown"))
    if resume_info.get("enabled"):
        same_gpu_model = len({mg_source_gpu, mg_resume_gpu, vllm_gpu}) == 1
        execution_scope = (
            "resumed_same_gpu_model"
            if same_gpu_model
            else "resumed_cross_environment"
        )
        print("\n== FAIR LONG-CONTEXT RESUMED RESULT ==")
        print(f"execution_scope: {execution_scope}")
        print(f"MegaGemm source GPU: {mg_source_gpu}")
        print(f"resume/vLLM GPU: {mg_resume_gpu} / {vllm_gpu}")
        print(f"resumed MegaGemm cases: {resumed_cases}")
    else:
        execution_scope = "same_vm"
        print("\n== FAIR LONG-CONTEXT SAME-VM RESULT ==")
        print(f"GPU: {mg_source_gpu}")
    print(f"model: {mg['model']}")
    print(f"dtype: {mg['dtype']} (both, unquantized)")
    print(f"contexts: {mg['contexts']} output={mg['max_tokens']}")
    print(f"batch_sizes: {mg['batch_sizes']}")
    print(f"vLLM version: {vl.get('version', 'unknown')}")
    print("vLLM prefix cache: OFF")
    print()
    print(
        f"{'B':>3} {'ctx':>6} {'MG prefill':>11} {'vLLM prefill':>12} "
        f"{'MG/vL':>8} {'MG decode':>10} {'vLLM decode':>11} {'MG/vL':>8} "
        f"{'total':>8}"
    )

    comparisons: dict[str, Any] = {}
    all_tokens_exact = True
    route_normalized_enabled = bool(mg.get("route_normalized_diagnostic", False))
    route_normalized_all_tokens_exact = True
    for context in mg["contexts"]:
        for batch_size in mg["batch_sizes"]:
            key = case_key(batch_size, context)
            mg_case = mg["cases"][key]
            vl_case = vl["cases"][key]
            if mg_case.get("prompt_contract") != vl_case.get("prompt_contract"):
                raise RuntimeError(f"prompt contract mismatch in case {key}")
            ms = mg_case["summary"]
            vs = vl_case["summary"]
            prefill_ratio = float(vs["prefill_ms_median"]) / float(
                ms["prefill_ms_median"]
            )
            decode_ratio = float(ms["decode_tok_s_median"]) / float(
                vs["decode_tok_s_median"]
            )
            total_ratio = float(ms["output_tok_s_total_median"]) / float(
                vs["output_tok_s_total_median"]
            )
            token_check = token_matrix_comparison(
                mg_case["samples"][0]["token_ids"],
                vl_case["samples"][0]["token_ids"],
            )
            all_tokens_exact = bool(all_tokens_exact and token_check["exact"])
            route_comparison = None
            if route_normalized_enabled:
                mg_route = dict(mg_case.get("route_normalized_diagnostic") or {})
                vl_route = dict(vl_case.get("route_normalized_diagnostic") or {})
                if not mg_route.get("enabled") or not vl_route.get("enabled"):
                    raise RuntimeError(
                        f"route-normalized diagnostic is missing in case {key}"
                    )
                if int(mg_route.get("forced_token_id", -1)) != int(
                    vl_route.get("forced_token_id", -2)
                ):
                    raise RuntimeError(
                        f"route-normalized token ID mismatch in case {key}"
                    )
                mg_route_summary = mg_route["summary"]
                vl_route_summary = vl_route["summary"]
                route_token_check = token_matrix_comparison(
                    mg_route["samples"][0]["token_ids"],
                    vl_route["samples"][0]["token_ids"],
                )
                route_normalized_all_tokens_exact = bool(
                    route_normalized_all_tokens_exact
                    and route_token_check["exact"]
                )
                mg_route_decode_ms = float(
                    mg_route_summary["decode_ms_median"]
                )
                vl_route_decode_ms = float(
                    vl_route_summary["decode_ms_median"]
                )
                decode_gap_ms = mg_route_decode_ms - vl_route_decode_ms
                route_comparison = {
                    "forced_token_id": int(mg_route["forced_token_id"]),
                    "token_comparison": route_token_check,
                    "prefill_speedup": float(
                        vl_route_summary["prefill_ms_median"]
                    )
                    / float(mg_route_summary["prefill_ms_median"]),
                    "decode_throughput_ratio": float(
                        mg_route_summary["decode_tok_s_median"]
                    )
                    / float(vl_route_summary["decode_tok_s_median"]),
                    "total_output_throughput_ratio": float(
                        mg_route_summary["output_tok_s_total_median"]
                    )
                    / float(vl_route_summary["output_tok_s_total_median"]),
                    "megagemm_decode_ms": mg_route_decode_ms,
                    "vllm_decode_ms": vl_route_decode_ms,
                    "megagemm_minus_vllm_decode_ms": decode_gap_ms,
                    "megagemm_minus_vllm_decode_us_per_step": (
                        decode_gap_ms * 1000.0 / (int(mg["max_tokens"]) - 1)
                    ),
                    "megagemm_forced_vs_natural_decode_ratio": (
                        mg_route_decode_ms / float(ms["decode_ms_median"])
                    ),
                    "vllm_forced_vs_natural_decode_ratio": (
                        vl_route_decode_ms / float(vs["decode_ms_median"])
                    ),
                }
            comparisons[key] = {
                "batch_size": batch_size,
                "context": context,
                "megagemm_resumed": key in resumed_cases,
                "prefill_speedup": prefill_ratio,
                "decode_throughput_ratio": decode_ratio,
                "total_output_throughput_ratio": total_ratio,
                "token_comparison": token_check,
                "route_normalized": route_comparison,
            }
            print(
                f"{batch_size:>3} {context:>6} "
                f"{float(ms['prefill_ms_median']):>11.2f} "
                f"{float(vs['prefill_ms_median']):>12.2f} "
                f"{prefill_ratio:>7.3f}x "
                f"{float(ms['decode_tok_s_median']):>10.2f} "
                f"{float(vs['decode_tok_s_median']):>11.2f} "
                f"{decode_ratio:>7.3f}x {total_ratio:>7.3f}x"
            )
            print(
                f"    tokens_exact={token_check['exact']} "
                f"first_mismatch={json.dumps(token_check['first_mismatch'], sort_keys=True)}"
            )
            if route_comparison is not None:
                print(
                    "    route_normalized "
                    f"tokens_exact={route_comparison['token_comparison']['exact']} "
                    "MG/vLLM "
                    "prefill="
                    f"{route_comparison['prefill_speedup']:.3f}x "
                    "decode="
                    f"{route_comparison['decode_throughput_ratio']:.3f}x "
                    "total="
                    f"{route_comparison['total_output_throughput_ratio']:.3f}x "
                    "gap_per_step="
                    f"{route_comparison['megagemm_minus_vllm_decode_us_per_step']:.1f}us"
                )

    if route_normalized_enabled:
        result_class = (
            "PERFORMANCE_AND_TOKEN_PARITY_PASS"
            if all_tokens_exact and route_normalized_all_tokens_exact
            else (
                "ROUTE_NORMALIZED_PERFORMANCE_VALID"
                if route_normalized_all_tokens_exact
                else "ROUTE_NORMALIZATION_FAILED"
            )
        )
    else:
        result_class = (
            "PERFORMANCE_AND_TOKEN_PARITY_PASS"
            if all_tokens_exact
            else "SHAPE_MATCHED_PERFORMANCE_ONLY"
        )
    result = {
        "schema_version": 1,
        "status": "complete",
        "megagemm_json": str(megagemm_path),
        "vllm_json": str(vllm_path),
        "execution_scope": execution_scope,
        "resumed_megagemm_cases": resumed_cases,
        "all_tokens_exact": all_tokens_exact,
        "route_normalized_enabled": route_normalized_enabled,
        "route_normalized_all_tokens_exact": (
            route_normalized_all_tokens_exact
            if route_normalized_enabled
            else None
        ),
        "result_class": result_class,
        "cases": comparisons,
    }
    print(f"Cross-backend result class: {result['result_class']}")
    _write_result(out_path, result)
    print(f"wrote {out_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("megagemm", "vllm", "compare"), required=True
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--contexts", default="1024,2048")
    parser.add_argument("--batch-sizes", default="1,16")
    parser.add_argument("--max-seq-len", type=int, default=2112)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--megagemm-min-warmups",
        type=int,
        default=MEGAGEMM_MIN_WARMUPS,
    )
    parser.add_argument(
        "--megagemm-max-warmups",
        type=int,
        default=MEGAGEMM_MAX_WARMUPS,
    )
    parser.add_argument(
        "--megagemm-required-stable-warmup-pairs",
        type=int,
        default=MEGAGEMM_REQUIRED_STABLE_WARMUP_PAIRS,
    )
    parser.add_argument(
        "--megagemm-warmup-max-last-pair-ratio",
        type=float,
        default=MEGAGEMM_WARMUP_MAX_LAST_PAIR_RATIO,
    )
    parser.add_argument(
        "--megagemm-determinism-auto-fallback",
        action="store_true",
        help=(
            "retry B16/C2048 with segmented deterministic MoE prefill before "
            "falling back to eager decode"
        ),
    )
    parser.add_argument(
        "--megagemm-prefill-chunk-autotune",
        action="store_true",
        help=(
            "run one loaded-checkpoint B16/C2048 16k-versus-32k prefill gate "
            "and retain the exact stable winner"
        ),
    )
    parser.add_argument(
        "--megagemm-prefill-chunk-gate-repeats",
        type=int,
        default=3,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--route-normalized-diagnostic",
        action="store_true",
        help=(
            "run an excluded identical-continuation diagnostic after the natural "
            "samples while retaining full LM-head work"
        ),
    )
    parser.add_argument(
        "--route-normalized-repeats",
        type=int,
        default=ROUTE_NORMALIZED_DEFAULT_REPEATS,
    )
    parser.add_argument(
        "--require-request-scheduler-reuse",
        action="store_true",
        help=(
            "require every measured MegaGemm request to reuse its compatible "
            "idle Scheduler without CUDA Graph warmup or recapture"
        ),
    )
    parser.add_argument(
        "--profile-prefill-stages",
        action="store_true",
        help=(
            "capture one excluded post-measurement MegaGemm prefill CUDA-event "
            "breakdown and verify its tokens against the measured prefix"
        ),
    )
    parser.add_argument(
        "--profile-decode-stages",
        action="store_true",
        help="capture one excluded eight-step MegaGemm CUDA-event breakdown",
    )
    parser.add_argument("--max-total-prefill-tokens", type=int, default=32768)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-token-ids-json", default="")
    parser.add_argument("--megagemm-json", default="")
    parser.add_argument("--vllm-json", default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse validated complete cases from an existing MegaGemm JSON",
    )
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    if args.backend == "compare":
        if not args.megagemm_json or not args.vllm_json:
            raise ValueError("compare requires --megagemm-json and --vllm-json")
        compare_results(
            Path(args.megagemm_json),
            Path(args.vllm_json),
            Path(args.out_json),
        )
        return 0

    if not args.model:
        raise ValueError("--model is required")
    args.contexts = parse_int_list(args.contexts, name="context")
    args.batch_sizes = parse_int_list(args.batch_sizes, name="batch")
    unsupported = [batch for batch in args.batch_sizes if batch not in SUPPORTED_BATCHES]
    if unsupported:
        raise ValueError(
            f"unsupported batches {unsupported}; choose from {SUPPORTED_BATCHES}"
        )
    if args.max_tokens < 2:
        raise ValueError("max_tokens must be at least 2")
    if args.warmups < 1 or args.repeats < 1:
        raise ValueError("warmups and repeats must be positive")
    if args.route_normalized_repeats < 1:
        raise ValueError("route_normalized_repeats must be positive")
    if args.megagemm_prefill_chunk_gate_repeats < 2:
        raise ValueError("megagemm_prefill_chunk_gate_repeats must be at least 2")
    if args.megagemm_min_warmups < 3:
        raise ValueError("megagemm_min_warmups must be at least 3")
    if args.megagemm_max_warmups < max(
        args.warmups,
        args.megagemm_min_warmups,
    ):
        raise ValueError(
            "megagemm_max_warmups must cover the requested minimum warmups"
        )
    if args.megagemm_required_stable_warmup_pairs < 2:
        raise ValueError(
            "megagemm_required_stable_warmup_pairs must be at least 2"
        )
    if (
        args.megagemm_required_stable_warmup_pairs
        >= args.megagemm_max_warmups
    ):
        raise ValueError(
            "megagemm_max_warmups must exceed required stable warmup pairs"
        )
    if args.megagemm_warmup_max_last_pair_ratio < 1.0:
        raise ValueError(
            "megagemm_warmup_max_last_pair_ratio must be at least 1.0"
        )
    if max(args.contexts) + args.max_tokens > args.max_seq_len:
        raise ValueError(
            "largest context plus output exceeds max_seq_len: "
            f"{max(args.contexts)} + {args.max_tokens} > {args.max_seq_len}"
        )
    largest_prefill = max(args.contexts) * max(args.batch_sizes)
    if largest_prefill > args.max_total_prefill_tokens:
        raise ValueError(
            f"largest prefill has {largest_prefill} tokens; safety cap is "
            f"{args.max_total_prefill_tokens}"
        )
    args.max_num_batched_tokens = (
        int(args.vllm_max_num_batched_tokens)
        if int(args.vllm_max_num_batched_tokens) > 0
        else largest_prefill
    )
    if args.max_num_batched_tokens < max(args.contexts):
        raise ValueError(
            "vLLM max_num_batched_tokens must fit at least one longest prompt"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    manifest_path = Path(args.prompt_token_ids_json)
    if not args.prompt_token_ids_json:
        raise ValueError("--prompt-token-ids-json is required")
    prompts, manifest = load_or_create_prompt_manifest(
        args.model,
        args.prompt,
        args.contexts,
        max(args.batch_sizes),
        manifest_path,
    )
    print("Gemma 4 MoE long-context benchmark")
    print(f"  backend: {args.backend}")
    print(f"  model: {args.model}")
    print(f"  contexts: {args.contexts}")
    print(f"  batch_sizes: {args.batch_sizes}")
    print(f"  output_tokens: {args.max_tokens}")
    print(f"  max_seq_len: {args.max_seq_len}")
    print(f"  warmups: {args.warmups} repeats: {args.repeats}")
    print(
        "  route_normalized_diagnostic: "
        f"{bool(args.route_normalized_diagnostic)} "
        f"repeats={args.route_normalized_repeats}"
    )
    if args.backend == "megagemm":
        minimum, maximum, required_pairs, maximum_ratio = (
            _megagemm_warmup_settings(args)
        )
        print(
            "  adaptive_warmup_gate: "
            f"min={minimum} max={maximum} stable_pairs={required_pairs} "
            f"max_pair_ratio={maximum_ratio:.3f}"
        )
        print(
            "  determinism_auto_fallback: "
            f"{bool(args.megagemm_determinism_auto_fallback)}"
        )
        print(
            "  prefill_chunk_autotune: "
            f"{bool(args.megagemm_prefill_chunk_autotune)} "
            f"repeats={args.megagemm_prefill_chunk_gate_repeats}"
        )
        print(
            "  prefill_stage_profile: "
            f"{bool(args.profile_prefill_stages)} (excluded after measured samples)"
        )
    print(f"  gpu: {gpu_snapshot()}")

    result = (
        run_megagemm_sweep(args, prompts, manifest)
        if args.backend == "megagemm"
        else run_vllm_sweep(args, prompts, manifest)
    )
    print("\n== SUMMARY ==")
    for key, case in result["cases"].items():
        print(f"{key} " + json.dumps(case["summary"], sort_keys=True))
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

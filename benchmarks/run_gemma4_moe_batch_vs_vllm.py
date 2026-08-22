"""One-process-per-backend Gemma 4 MoE batch throughput matrix.

The companion shell harness downloads once, runs MegaGemm first, and installs
vLLM only after MegaGemm's graph and kernel correctness gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_gemma4_moe_vs_vllm import (  # noqa: E402
    dtype_from_name,
    extract_vllm_request_metrics,
    gpu_snapshot,
    load_exact_prompt,
    make_vllm,
    sync_cuda,
    validated_vllm_phase_span,
)


SUPPORTED_BATCHES = (2, 4, 8, 16)
VLLM_MIN_WARMUPS = 3
VLLM_MAX_WARMUPS = 8
VLLM_WARMUP_MAX_LAST_PAIR_RATIO = 1.05


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_batch_sizes(raw: str) -> list[int]:
    values: list[int] = []
    for item in str(raw).split(","):
        value = int(item.strip())
        if value not in SUPPORTED_BATCHES:
            raise ValueError(
                f"unsupported batch {value}; choose from {SUPPORTED_BATCHES}"
            )
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one batch size is required")
    return values


def build_distinct_equal_length_prompt_inputs(
    model: str,
    exact_prompt: str,
    required: int,
    expected_tokens: int,
) -> tuple[list[str], list[list[int]]]:
    """Create deterministic heterogeneous prompts without changing the shape."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    base_ids = [
        int(token)
        for token in tokenizer.encode(exact_prompt, add_special_tokens=False)
    ]
    if len(base_ids) != expected_tokens:
        raise RuntimeError(
            f"base prompt has {len(base_ids)} tokens; expected {expected_tokens}"
        )

    special_ids = {int(token) for token in tokenizer.all_special_ids}
    mutable_positions = [
        index
        for index, token in enumerate(base_ids)
        if token not in special_ids and 1 < index < len(base_ids) - 2
    ]
    if not mutable_positions:
        raise RuntimeError("could not find a mutable content token in the prompt")

    prompts = [exact_prompt]
    prompt_token_ids = [list(base_ids)]
    seen_prompts = {exact_prompt}
    bos_prefix = str(getattr(tokenizer, "bos_token", "") or "")
    require_bos_prefix = bool(bos_prefix and exact_prompt.startswith(bos_prefix))
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    candidate = 0
    attempts = 0
    max_attempts = min(max(vocab_size * 2, 4096), 200000)
    while len(prompts) < required and attempts < max_attempts:
        position = mutable_positions[(len(prompts) - 1) % len(mutable_positions)]
        token_id = candidate % vocab_size
        candidate += 1
        attempts += 1
        if token_id in special_ids or token_id == base_ids[position]:
            continue
        token_piece = tokenizer.convert_ids_to_tokens(token_id)
        if not token_piece or (
            str(token_piece).startswith("<") and str(token_piece).endswith(">")
        ):
            continue

        variant_ids = list(base_ids)
        variant_ids[position] = token_id
        try:
            variant = tokenizer.decode(
                variant_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            roundtrip = tokenizer.encode(variant, add_special_tokens=False)
        except Exception:
            continue
        if variant in seen_prompts:
            continue
        if require_bos_prefix and not variant.startswith(bos_prefix):
            continue
        if [int(token) for token in roundtrip] != variant_ids:
            continue
        prompts.append(variant)
        prompt_token_ids.append(list(variant_ids))
        seen_prompts.add(variant)

    if len(prompts) != required:
        raise RuntimeError(
            f"built only {len(prompts)}/{required} distinct {expected_tokens}-token prompts"
        )
    return prompts, prompt_token_ids


def build_distinct_equal_length_prompts(
    model: str,
    exact_prompt: str,
    required: int,
    expected_tokens: int,
) -> list[str]:
    """Compatibility wrapper returning the round-trippable prompt texts."""
    prompts, _ = build_distinct_equal_length_prompt_inputs(
        model,
        exact_prompt,
        required,
        expected_tokens,
    )
    return prompts


def prompt_token_contract(prompt_token_ids: list[list[int]]) -> dict[str, Any]:
    if not prompt_token_ids or any(not row for row in prompt_token_ids):
        raise ValueError("prompt token matrix must contain non-empty rows")
    normalized = [
        [int(token) for token in row]
        for row in prompt_token_ids
    ]
    row_lengths = {len(row) for row in normalized}
    if len(row_lengths) != 1:
        raise ValueError("prompt token matrix must be rectangular")
    distinct_rows = len({tuple(row) for row in normalized})
    if distinct_rows != len(normalized):
        raise ValueError("prompt token rows must be distinct")
    encoded = json.dumps(
        normalized,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return {
        "format": "prompt_token_ids",
        "rows": len(normalized),
        "tokens_per_row": next(iter(row_lengths)),
        "distinct_rows": distinct_rows,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def load_or_create_prompt_token_ids(
    model: str,
    prompt: str,
    required: int,
    expected_tokens: int,
    manifest_path: Path | None,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Persist one token matrix so both backend processes consume identical IDs."""
    token_rows: list[list[int]]
    if manifest_path is not None and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        token_rows = [
            [int(token) for token in row]
            for row in payload.get("token_ids", [])
        ]
        stored_contract = payload.get("contract")
        computed_contract = prompt_token_contract(token_rows)
        if stored_contract != computed_contract:
            raise RuntimeError(
                f"prompt token manifest contract mismatch: {manifest_path}"
            )
    else:
        exact_prompt, prompt_tokens = load_exact_prompt(model, prompt)
        if prompt_tokens != expected_tokens:
            raise RuntimeError(
                f"batch policy expects exactly {expected_tokens} prompt tokens, "
                f"got {prompt_tokens}"
            )
        _, token_rows = build_distinct_equal_length_prompt_inputs(
            model,
            exact_prompt,
            required,
            expected_tokens,
        )
        computed_contract = prompt_token_contract(token_rows)
        if manifest_path is not None:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "contract": computed_contract,
                "token_ids": token_rows,
            }
            temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(manifest_path)

    if len(token_rows) < required:
        raise RuntimeError(
            f"prompt token manifest has {len(token_rows)} rows; {required} required"
        )
    token_rows = token_rows[:required]
    contract = prompt_token_contract(token_rows)
    if int(contract["tokens_per_row"]) != expected_tokens:
        raise RuntimeError(
            f"prompt token rows have {contract['tokens_per_row']} tokens; "
            f"expected {expected_tokens}"
        )
    return token_rows, contract


def token_matrix_comparison(
    reference: list[list[int]],
    candidate: list[list[int]],
) -> dict[str, Any]:
    row_exact = [left == right for left, right in zip(reference, candidate)]
    common_prefixes = []
    first_mismatch = None
    for left, right in zip(reference, candidate):
        common = 0
        for left_token, right_token in zip(left, right):
            if int(left_token) != int(right_token):
                break
            common += 1
        common_prefixes.append(common)
    for row_index in range(max(len(reference), len(candidate))):
        if row_index >= len(reference) or row_index >= len(candidate):
            first_mismatch = {
                "row": row_index,
                "token": 0,
                "reference": None if row_index >= len(reference) else "row_present",
                "candidate": None if row_index >= len(candidate) else "row_present",
            }
            break
        left = reference[row_index]
        right = candidate[row_index]
        common = common_prefixes[row_index]
        if common < len(left) or common < len(right):
            first_mismatch = {
                "row": row_index,
                "token": common,
                "reference": int(left[common]) if common < len(left) else None,
                "candidate": int(right[common]) if common < len(right) else None,
            }
            break
    return {
        "exact": len(reference) == len(candidate) and all(row_exact),
        "rows": len(candidate),
        "row_exact": row_exact,
        "common_prefix_tokens": common_prefixes,
        "min_common_prefix_tokens": min(common_prefixes, default=0),
        "mismatched_rows": sum(not exact for exact in row_exact)
        + abs(len(reference) - len(candidate)),
        "first_mismatch": first_mismatch,
    }


def evaluate_vllm_warmup_stability(
    warmups: list[dict[str, Any]],
    *,
    minimum_warmups: int = VLLM_MIN_WARMUPS,
    maximum_warmups: int = VLLM_MAX_WARMUPS,
    maximum_last_pair_ratio: float = VLLM_WARMUP_MAX_LAST_PAIR_RATIO,
) -> dict[str, Any]:
    """Evaluate strict timing stability and safe post-budget acceptance."""
    completed = len(warmups)
    if completed < 2:
        return {
            "stable": False,
            "accepted": False,
            "reason": "need_two_samples",
            "acceptance_reason": "need_two_samples",
            "completed_warmups": completed,
            "minimum_warmups": int(minimum_warmups),
            "maximum_warmups": int(maximum_warmups),
            "maximum_last_pair_ratio": float(maximum_last_pair_ratio),
            "last_pair_total_ratio": None,
            "last_pair_tokens": None,
            "all_warmup_tokens": None,
        }

    previous = warmups[-2]
    current = warmups[-1]
    previous_ms = float(previous["total_ms"])
    current_ms = float(current["total_ms"])
    timing_ratio = max(previous_ms, current_ms) / max(
        min(previous_ms, current_ms), 1e-9
    )
    token_comparison = token_matrix_comparison(
        previous["token_ids"], current["token_ids"]
    )
    token_reference = warmups[0]["token_ids"]
    all_token_comparisons = [
        token_matrix_comparison(token_reference, row["token_ids"])
        for row in warmups[1:]
    ]
    all_tokens_stable = all(
        bool(comparison["exact"]) for comparison in all_token_comparisons
    )
    enough_samples = completed >= int(minimum_warmups)
    budget_exhausted = completed >= int(maximum_warmups)
    timing_stable = timing_ratio <= float(maximum_last_pair_ratio)
    tokens_stable = bool(token_comparison["exact"])
    stable = bool(
        enough_samples
        and timing_stable
        and tokens_stable
        and all_tokens_stable
    )
    accepted = bool(
        stable
        or (enough_samples and budget_exhausted and all_tokens_stable)
    )
    if not enough_samples:
        reason = "minimum_warmups_not_reached"
    elif not all_tokens_stable or not tokens_stable:
        reason = "last_pair_tokens_changed"
    elif not timing_stable:
        reason = "last_pair_timing_unstable"
    else:
        reason = "stable"
    if stable:
        acceptance_reason = "strict_timing_and_tokens_stable"
    elif accepted:
        acceptance_reason = "warmup_budget_exhausted_all_tokens_exact"
    elif not all_tokens_stable:
        acceptance_reason = "warmup_tokens_changed"
    else:
        acceptance_reason = reason
    return {
        "stable": stable,
        "accepted": accepted,
        "reason": reason,
        "acceptance_reason": acceptance_reason,
        "completed_warmups": completed,
        "minimum_warmups": int(minimum_warmups),
        "maximum_warmups": int(maximum_warmups),
        "budget_exhausted": bool(budget_exhausted),
        "maximum_last_pair_ratio": float(maximum_last_pair_ratio),
        "last_pair_total_ratio": float(timing_ratio),
        "last_pair_tokens": token_comparison,
        "all_warmup_tokens": {
            "exact": bool(all_tokens_stable),
            "checked_pairs": len(all_token_comparisons),
            "comparisons": all_token_comparisons,
        },
    }


def align_vllm_outputs_to_prompts(
    outputs: list[Any],
    prompts: list[list[int]],
) -> tuple[list[list[int]], dict[str, Any]]:
    """Return vLLM token rows in the exact input-prompt order."""
    output_prompt_ids = [
        getattr(request, "prompt_token_ids", None) for request in outputs
    ]
    if all(prompt_ids is None for prompt_ids in output_prompt_ids):
        matrix = [
            [int(token) for token in request.outputs[0].token_ids]
            for request in outputs
        ]
        return matrix, {
            "method": "returned_order_no_prompt_ids",
            "reordered": False,
            "output_to_prompt_index": list(range(len(matrix))),
        }
    if any(prompt_ids is None for prompt_ids in output_prompt_ids):
        raise RuntimeError(
            "vLLM returned prompt_token_ids for only part of the batch"
        )

    prompt_slots: dict[tuple[int, ...], list[int]] = {}
    for prompt_index, prompt in enumerate(prompts):
        prompt_slots.setdefault(tuple(int(token) for token in prompt), []).append(
            prompt_index
        )

    matrix: list[list[int] | None] = [None] * len(prompts)
    output_to_prompt_index: list[int] = []
    for request, prompt_ids in zip(outputs, output_prompt_ids):
        prompt_key = tuple(int(token) for token in prompt_ids)
        slots = prompt_slots.get(prompt_key)
        if not slots:
            raise RuntimeError(
                "vLLM returned an output whose prompt_token_ids do not match "
                "the shared prompt manifest"
            )
        prompt_index = slots.pop(0)
        output_to_prompt_index.append(prompt_index)
        matrix[prompt_index] = [
            int(token) for token in request.outputs[0].token_ids
        ]

    missing = [index for index, row in enumerate(matrix) if row is None]
    leftovers = sum(len(slots) for slots in prompt_slots.values())
    if missing or leftovers:
        raise RuntimeError(
            "vLLM output-to-prompt alignment is incomplete: "
            f"missing={missing} unmatched_prompts={leftovers}"
        )
    return [row for row in matrix if row is not None], {
        "method": "prompt_token_ids",
        "reordered": output_to_prompt_index != list(range(len(outputs))),
        "output_to_prompt_index": output_to_prompt_index,
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def median(key: str, *, require_all: bool = True) -> float | None:
        values = [float(row[key]) for row in samples if row.get(key) is not None]
        if not values or (require_all and len(values) != len(samples)):
            return None
        return float(statistics.median(values))

    return {
        "samples": len(samples),
        "total_ms_median": median("total_ms"),
        "output_tok_s_total_median": median("output_tok_s_total"),
        "prefill_ms_median": median("prefill_ms"),
        "decode_ms_median": median("decode_ms"),
        "decode_tok_s_median": median("decode_tok_s"),
        "decode_measurement_methods": sorted(
            {str(row["decode_measurement_method"]) for row in samples}
        ),
        "phase_metrics_valid_samples": sum(
            row.get("phase_metrics_status") == "valid" for row in samples
        ),
    }


def write_partial_checkpoint(
    args: argparse.Namespace,
    backend: str,
    cases: dict[str, Any],
    **extra: Any,
) -> None:
    """Preserve completed paid-GPU cases before moving to the next batch."""
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": backend,
        "status": "partial",
        "model": args.model,
        "dtype": args.dtype,
        "batch_sizes": args.batch_sizes,
        "max_tokens": args.max_tokens,
        "cases": cases,
        **extra,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def completed_token_matrix(engine, expected_batch: int, max_tokens: int) -> list[list[int]]:
    scheduler = engine._last_scheduler
    completed = sorted(scheduler._completed, key=lambda req: int(req.request_id))
    matrix = [[int(token) for token in req.generated_ids] for req in completed]
    if len(matrix) != expected_batch:
        raise RuntimeError(
            f"MegaGemm completed {len(matrix)} rows; expected {expected_batch}"
        )
    bad = [len(row) for row in matrix if len(row) != max_tokens]
    if bad:
        raise RuntimeError(
            f"MegaGemm token count mismatch: expected {max_tokens}, got {bad[:8]}"
        )
    return matrix


def run_megagemm_request(
    engine,
    prompts: list[list[int]],
    max_tokens: int,
) -> dict[str, Any]:
    batch_size = len(prompts)
    sync_cuda()
    started = time.perf_counter()
    engine.generate_batch(
        prompts,
        max_new_tokens=max_tokens,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        ignore_eos=True,
        verbose=False,
        decode_outputs=False,
        materialize_generated_tokens=True,
    )
    sync_cuda()
    total_ms = (time.perf_counter() - started) * 1000.0
    matrix = completed_token_matrix(engine, batch_size, max_tokens)
    scheduler_stats = engine._last_scheduler.get_stats()
    return {
        "total_ms": total_ms,
        "output_tokens": batch_size * max_tokens,
        "output_tok_s_total": batch_size * max_tokens / (total_ms / 1000.0),
        "token_ids": matrix,
        "decode_cuda_graphs": scheduler_stats.get("decode_cuda_graphs", {}),
        "prefill_cuda_graphs": scheduler_stats.get("prefill_cuda_graphs", {}),
        "scheduler_prefill_ms": scheduler_stats.get("prefill_time_ms"),
        "scheduler_decode_ms": scheduler_stats.get("decode_time_ms"),
        "benchmark_forced_token_id": scheduler_stats.get(
            "benchmark_forced_token_id", -1
        ),
        "prefill_stage_timing": scheduler_stats.get("prefill_stage_timing", {}),
        "prefill_stage_chunks": scheduler_stats.get("prefill_stage_chunks", 0),
        "prefill_stage_total_tokens": scheduler_stats.get(
            "prefill_stage_total_tokens", 0
        ),
        "prefill_stage_total_seqs": scheduler_stats.get(
            "prefill_stage_total_seqs", 0
        ),
        "prefill_stage_max_len": scheduler_stats.get("prefill_stage_max_len", 0),
        "prefill_chunk_plan": scheduler_stats.get("prefill_chunk_plan"),
        "decode_runtime": engine.model.decode_runtime_stats(),
    }


def run_excluded_megagemm_prefill_profile(
    engine,
    prompts: list[list[int]],
    reference_tokens: list[list[int]],
) -> dict[str, Any]:
    """Profile one short eager request after latency samples have completed."""
    import importlib

    model_module = importlib.import_module(engine.model.__class__.__module__)
    previous_timing = bool(getattr(model_module, "_PREFILL_TIMING", False))
    previous_print = bool(getattr(model_module, "_PREFILL_TIMING_PRINT", True))
    previous_env = os.environ.get("MEGAGEMM_PREFILL_TIMING")
    previous_graph_env = os.environ.get("MEGAGEMM_PREFILL_CUDA_GRAPHS")
    profile_tokens = 2
    try:
        setattr(model_module, "_PREFILL_TIMING", True)
        setattr(model_module, "_PREFILL_TIMING_PRINT", False)
        os.environ["MEGAGEMM_PREFILL_TIMING"] = "1"
        os.environ["MEGAGEMM_PREFILL_CUDA_GRAPHS"] = "0"
        row = run_megagemm_request(engine, prompts, profile_tokens)
    finally:
        setattr(model_module, "_PREFILL_TIMING", previous_timing)
        setattr(model_module, "_PREFILL_TIMING_PRINT", previous_print)
        if previous_env is None:
            os.environ.pop("MEGAGEMM_PREFILL_TIMING", None)
        else:
            os.environ["MEGAGEMM_PREFILL_TIMING"] = previous_env
        if previous_graph_env is None:
            os.environ.pop("MEGAGEMM_PREFILL_CUDA_GRAPHS", None)
        else:
            os.environ["MEGAGEMM_PREFILL_CUDA_GRAPHS"] = previous_graph_env

    timing = dict(row.get("prefill_stage_timing") or {})
    if not timing or int(row.get("prefill_stage_chunks") or 0) <= 0:
        raise RuntimeError("excluded MegaGemm prefill profile produced no stage timing")
    expected = [tokens[:profile_tokens] for tokens in reference_tokens]
    comparison = token_matrix_comparison(expected, row["token_ids"])
    if not comparison["exact"]:
        raise RuntimeError(
            "excluded MegaGemm prefill profile changed greedy tokens: "
            + json.dumps(comparison, sort_keys=True)
        )
    ranked_stages = sorted(
        (
            {"stage": name, "ms": float(value)}
            for name, value in timing.items()
            if name.endswith("_ms") and name != "total_ms"
        ),
        key=lambda item: item["ms"],
        reverse=True,
    )
    return {
        "excluded_from_latency_summary": True,
        "measurement": "summed_cuda_events_after_measured_requests",
        "profile_output_tokens_per_request": profile_tokens,
        "request_total_ms": float(row["total_ms"]),
        "scheduler_prefill_ms": float(row.get("scheduler_prefill_ms") or 0.0),
        "stage_timing": timing,
        "ranked_stages": ranked_stages,
        "chunks": int(row.get("prefill_stage_chunks") or 0),
        "total_tokens": int(row.get("prefill_stage_total_tokens") or 0),
        "total_seqs": int(row.get("prefill_stage_total_seqs") or 0),
        "max_len": int(row.get("prefill_stage_max_len") or 0),
        "tokens_vs_measured_prefix": comparison,
    }


def run_megagemm_first_token_contract(
    engine,
    prompts: list[list[int]],
    *,
    reference_tokens: list[int] | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Verify that prefill logits are finite and generate their exact greedy token."""
    if reference_tokens is not None and len(reference_tokens) != len(prompts):
        raise ValueError(
            "first-token reference length must match the number of prompts"
        )
    captured: dict[int, torch.Tensor] = {}

    def capture(req, logits: torch.Tensor) -> None:
        captured[int(req.request_id)] = logits.detach().clone()

    engine.generate_batch(
        prompts,
        max_new_tokens=1,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        ignore_eos=True,
        verbose=False,
        prefill_capture_hook=capture,
        decode_outputs=False,
        materialize_generated_tokens=True,
    )
    sync_cuda()
    completed = sorted(
        engine._last_scheduler._completed,
        key=lambda req: int(req.request_id),
    )
    generated = completed_token_matrix(engine, len(prompts), 1)
    if len(captured) != len(prompts):
        raise RuntimeError(
            f"prefill contract captured {len(captured)} rows; expected {len(prompts)}"
        )

    rows = []
    all_finite = True
    all_exact = True
    all_reference_exact = True
    for prompt_index, (req, generated_row) in enumerate(zip(completed, generated)):
        request_id = int(req.request_id)
        logits = captured.get(request_id)
        if logits is None:
            raise RuntimeError(
                "prefill contract did not capture completed request "
                f"{request_id}; captured IDs={sorted(captured)}"
            )
        finite = bool(torch.isfinite(logits).all().item())
        greedy_token = int(logits.argmax(dim=-1).item()) if finite else -1
        generated_token = int(generated_row[0])
        exact = bool(finite and greedy_token == generated_token)
        reference_token = (
            None
            if reference_tokens is None
            else int(reference_tokens[prompt_index])
        )
        reference_exact = bool(
            reference_token is None
            or (finite and generated_token == reference_token)
        )
        all_finite = bool(all_finite and finite)
        all_exact = bool(all_exact and exact)
        all_reference_exact = bool(all_reference_exact and reference_exact)
        rows.append({
            "prompt_index": prompt_index,
            "request_id": request_id,
            "finite": finite,
            "greedy_token": greedy_token,
            "generated_token": generated_token,
            "exact": exact,
            "reference_token": reference_token,
            "reference_exact": reference_exact,
        })

    result = {
        "rows": len(rows),
        "all_finite": all_finite,
        "all_exact": all_exact,
        "all_reference_exact": all_reference_exact,
        "generated_tokens": [int(row[0]) for row in generated],
        "details": rows,
    }
    if raise_on_failure and (
        not all_finite or not all_exact or not all_reference_exact
    ):
        raise RuntimeError(
            "MegaGemm first-token contract failed: "
            + json.dumps(result, sort_keys=True)
        )
    return result


_GEMMA4_PREFILL_RUNTIME_GLOBALS = {
    "fused_qkv": "_GEMMA4_FUSED_QKV_PREFILL",
    "fused_attention_prepare": "_GEMMA4_FUSED_ATTN_PREP_PREFILL",
    "implicit_causal": "_GEMMA4_IMPLICIT_CAUSAL_PREFILL",
    "vectorized_kv": "_GEMMA4_VECTORIZED_PREFILL_KV",
    "parallel_moe": "_GEMMA4_PARALLEL_MOE_PREFILL",
    "fused_dual_ffn_norm": "_GEMMA4_FUSED_DUAL_FFN_NORM_PREFILL",
    "fused_add_ffn_norm": "_GEMMA4_FUSED_ADD_FFN_NORM_PREFILL",
    "fused_post_ffn_norms": "_GEMMA4_FUSED_POST_FFN_NORMS_PREFILL",
}


def _gemma4_prefill_runtime_state(engine) -> dict[str, bool]:
    import megagemm.models.llama as llama_model

    state = {
        key: bool(getattr(llama_model, attr))
        for key, attr in _GEMMA4_PREFILL_RUNTIME_GLOBALS.items()
    }
    experts = []
    v_norms = []
    for layer in getattr(engine.model, "layers", ()):
        mlp = getattr(layer, "mlp", None)
        layer_experts = getattr(mlp, "experts", None)
        if layer_experts is not None and hasattr(
            layer_experts, "_gemma4_a4b_segmented_prefill"
        ):
            experts.append(layer_experts)
        attn = getattr(layer, "self_attn", None)
        v_norm = getattr(attn, "v_norm", None)
        if v_norm is not None and hasattr(v_norm, "_force_pytorch_no_weight"):
            v_norms.append(v_norm)
    state["segmented_moe"] = bool(
        experts and all(expert._gemma4_a4b_segmented_prefill for expert in experts)
    )
    state["triton_v_norm"] = bool(
        not v_norms
        or all(not norm._force_pytorch_no_weight for norm in v_norms)
    )
    state["sequential_prefill"] = bool(
        getattr(engine.model, "_force_sequential_prefill", False)
    )
    return state


def _set_gemma4_prefill_runtime_state(
    engine,
    state: dict[str, bool],
) -> None:
    import megagemm.models.llama as llama_model

    for key, attr in _GEMMA4_PREFILL_RUNTIME_GLOBALS.items():
        setattr(llama_model, attr, bool(state[key]))
    for layer in getattr(engine.model, "layers", ()):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if experts is not None and hasattr(
            experts, "_gemma4_a4b_segmented_prefill"
        ):
            experts._gemma4_a4b_segmented_prefill = bool(
                state["segmented_moe"]
            )
        attn = getattr(layer, "self_attn", None)
        v_norm = getattr(attn, "v_norm", None)
        if v_norm is not None and hasattr(v_norm, "_force_pytorch_no_weight"):
            v_norm._force_pytorch_no_weight = not bool(state["triton_v_norm"])
    engine.model._force_sequential_prefill = bool(state["sequential_prefill"])


def _prefill_contract_case_summary(
    name: str,
    state: dict[str, bool],
    contract: dict[str, Any] | None,
    elapsed_ms: float,
    error: str = "",
) -> dict[str, Any]:
    details = [] if contract is None else list(contract.get("details") or [])
    return {
        "case": name,
        "runtime": dict(state),
        "elapsed_ms": elapsed_ms,
        "error": error,
        "finite_rows": sum(bool(row.get("finite")) for row in details),
        "self_exact_rows": sum(bool(row.get("exact")) for row in details),
        "reference_exact_rows": sum(
            bool(row.get("reference_exact")) for row in details
        ),
        "all_finite": bool(contract and contract.get("all_finite")),
        "all_exact": bool(contract and contract.get("all_exact")),
        "all_reference_exact": bool(
            contract and contract.get("all_reference_exact")
        ),
        "generated_tokens": (
            [] if contract is None else list(contract.get("generated_tokens") or [])
        ),
    }


def run_gemma4_prefill_correctness_gate(
    engine,
    prompts: list[list[int]],
    *,
    contract_runner=run_megagemm_first_token_contract,
) -> dict[str, Any]:
    """Select a stable B16 prefill path against an exact sequential oracle."""
    baseline = _gemma4_prefill_runtime_state(engine)

    oracle_state = dict(baseline)
    oracle_state["sequential_prefill"] = True
    _set_gemma4_prefill_runtime_state(engine, oracle_state)
    oracle_started = time.perf_counter()
    try:
        oracle = contract_runner(engine, prompts)
    except Exception:
        _set_gemma4_prefill_runtime_state(engine, baseline)
        raise
    oracle_ms = (time.perf_counter() - oracle_started) * 1000.0
    reference_tokens = list(oracle["generated_tokens"])

    all_fast_off = {
        "fused_qkv": False,
        "fused_attention_prepare": False,
        "implicit_causal": False,
        "vectorized_kv": False,
        "parallel_moe": False,
        "fused_dual_ffn_norm": False,
        "fused_add_ffn_norm": False,
        "fused_post_ffn_norms": False,
        "segmented_moe": False,
        "triton_v_norm": False,
    }
    candidate_overrides = [
        ("current", {}),
        ("without_parallel_moe", {"parallel_moe": False}),
        ("without_vectorized_kv", {"vectorized_kv": False}),
        ("without_triton_v_norm", {"triton_v_norm": False}),
        (
            "without_fused_ffn_norms",
            {
                "fused_dual_ffn_norm": False,
                "fused_add_ffn_norm": False,
                "fused_post_ffn_norms": False,
            },
        ),
        ("without_fused_qkv", {"fused_qkv": False}),
        (
            "without_fused_attention_prepare",
            {"fused_attention_prepare": False},
        ),
        ("without_implicit_causal", {"implicit_causal": False}),
        ("without_segmented_moe", {"segmented_moe": False}),
        (
            "safe_attention",
            {
                "fused_qkv": False,
                "fused_attention_prepare": False,
                "implicit_causal": False,
                "triton_v_norm": False,
            },
        ),
        (
            "safe_moe",
            {
                "parallel_moe": False,
                "segmented_moe": False,
                "fused_dual_ffn_norm": False,
                "fused_add_ffn_norm": False,
                "fused_post_ffn_norms": False,
            },
        ),
        ("safe_batched", all_fast_off),
        ("sequential_prefill", {"sequential_prefill": True}),
    ]

    cases = []
    selected_name = ""
    selected_state = None
    selected_contract = None
    for name, overrides in candidate_overrides:
        state = dict(baseline)
        state.update(overrides)
        _set_gemma4_prefill_runtime_state(engine, state)
        started = time.perf_counter()
        try:
            contract = contract_runner(
                engine,
                prompts,
                reference_tokens=reference_tokens,
                raise_on_failure=False,
            )
            error = ""
        except Exception as exc:
            contract = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        case = _prefill_contract_case_summary(
            name, state, contract, elapsed_ms, error
        )
        cases.append(case)
        passed = bool(
            contract
            and contract.get("all_finite")
            and contract.get("all_exact")
            and contract.get("all_reference_exact")
        )
        if not passed:
            continue

        recheck_started = time.perf_counter()
        try:
            recheck = contract_runner(
                engine,
                prompts,
                reference_tokens=reference_tokens,
                raise_on_failure=False,
            )
            recheck_error = ""
        except Exception as exc:
            recheck = None
            recheck_error = f"{type(exc).__name__}: {exc}"
        recheck_ms = (time.perf_counter() - recheck_started) * 1000.0
        recheck_case = _prefill_contract_case_summary(
            f"{name}_recheck",
            state,
            recheck,
            recheck_ms,
            recheck_error,
        )
        cases.append(recheck_case)
        stable = bool(
            recheck
            and recheck.get("all_finite")
            and recheck.get("all_exact")
            and recheck.get("all_reference_exact")
        )
        if stable:
            selected_name = name
            selected_state = state
            selected_contract = recheck
            break

    if selected_state is None or selected_contract is None:
        _set_gemma4_prefill_runtime_state(engine, baseline)
        raise RuntimeError(
            "Gemma4 B16 prefill correctness gate found no stable exact path: "
            + json.dumps(cases, sort_keys=True)
        )

    _set_gemma4_prefill_runtime_state(engine, selected_state)
    if selected_name == "current":
        decision = "KEEP_CURRENT"
    elif selected_name == "sequential_prefill":
        decision = "APPLY_SEQUENTIAL_FALLBACK"
    else:
        decision = "APPLY_RUNTIME_ROLLBACK"
    return {
        "decision": decision,
        "selected": selected_name,
        "selected_runtime": dict(selected_state),
        "oracle": {
            "source": "same_loaded_model_sequential_prefill",
            "elapsed_ms": oracle_ms,
            "tokens": reference_tokens,
        },
        "selected_contract": selected_contract,
        "cases": cases,
    }


def run_gemma4_prefill_finite_trace(
    engine,
    prompts: list[list[int]],
) -> dict[str, Any]:
    """Trace one safe B>1 prefill and stop at its first nonfinite stage."""
    model = engine.model
    begin = getattr(model, "begin_gemma4_prefill_finite_trace", None)
    end = getattr(model, "end_gemma4_prefill_finite_trace", None)
    if not callable(begin) or not callable(end):
        raise RuntimeError("loaded model does not expose Gemma4 finite tracing")

    baseline = _gemma4_prefill_runtime_state(engine)
    safe_state = dict(baseline)
    safe_state.update({
        "fused_qkv": False,
        "fused_attention_prepare": False,
        "implicit_causal": False,
        "vectorized_kv": False,
        "parallel_moe": False,
        "fused_dual_ffn_norm": False,
        "fused_add_ffn_norm": False,
        "fused_post_ffn_norms": False,
        "segmented_moe": False,
        "triton_v_norm": False,
        "sequential_prefill": False,
    })
    _set_gemma4_prefill_runtime_state(engine, safe_state)

    begin(stop_on_nonfinite=True)
    contract = None
    error = ""
    started = time.perf_counter()
    try:
        contract = run_megagemm_first_token_contract(
            engine,
            prompts,
            raise_on_failure=False,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        trace = end()
        _set_gemma4_prefill_runtime_state(engine, baseline)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    status = str(trace.get("status") or "UNKNOWN")
    if status == "PASS" and contract is not None and not contract.get("all_finite"):
        status = "CONTRACT_NONFINITE"
    return {
        "mode": "safe_batched_first_nonfinite",
        "status": status,
        "elapsed_ms": elapsed_ms,
        "runtime": safe_state,
        "first_bad": trace.get("first_bad"),
        "events": list(trace.get("events") or []),
        "contract": contract,
        "error": error,
    }


def run_batch_lm_head_kernel_gate(engine, rows: int) -> dict[str, Any]:
    import megagemm.models.llama as llama_model

    model = engine.model
    hidden_dim = int(model.config.hidden_size)
    vocab_size = int(model.config.vocab_size)
    model_dtype = next(model.parameters()).dtype
    device_name = torch.cuda.get_device_name(next(model.parameters()).device)
    if (
        rows != 16
        or hidden_dim != 2816
        or vocab_size != 262144
        or model_dtype != torch.bfloat16
        or "A100" not in device_name.upper()
    ):
        llama_model._GEMMA4_BATCH_CUBLAS_LM_HEAD = False
        return {
            "rows": rows,
            "hidden": hidden_dim,
            "vocab": vocab_size,
            "dtype": str(model_dtype),
            "device_name": device_name,
            "cases": [],
            "decision": "SKIP_UNSUPPORTED_SHAPE",
            "selected": "current_logits_cap_argmax",
        }

    if env_flag("MEGAGEMM_GEMMA4_B16_GRAPH_TOKEN_BURST_PROVEN"):
        llama_model._GEMMA4_BATCH_CUBLAS_LM_HEAD = True
        model._gemma4_batch_cublas_lm_head_hits = 0
        return {
            "rows": rows,
            "hidden": hidden_dim,
            "vocab": vocab_size,
            "dtype": "bf16",
            "device_name": device_name,
            "cases": [],
            "decision": "USE_PROVEN_EXACT_TOKEN_PATH",
            "selected": "cublas_greedy_token",
            "evidence": {
                "run_id": "gemma4_moe_ab_20260721_002328",
                "tokens_exact": True,
                "rows": 16,
                "tokens_per_row": 64,
            },
        }

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260719)
    hidden = torch.randn(
        (rows, 1, hidden_dim),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    current_out = torch.empty((rows,), device=hidden.device, dtype=torch.long)

    def current_call() -> torch.Tensor:
        capped_logits = model._decode_logits_from_hidden(hidden)
        torch.argmax(capped_logits[:, -1, :], dim=-1, out=current_out)
        return current_out

    def greedy_token_call() -> torch.Tensor:
        return model._decode_next_token_greedy(hidden)

    with torch.inference_mode():
        # The established logits+softcap path is the correctness oracle. The
        # direct-token candidate must match it, never define its own reference.
        llama_model._GEMMA4_BATCH_CUBLAS_LM_HEAD = False
        current_call()
        token_reference = current_out.clone()
        sync_cuda()
        llama_model._GEMMA4_BATCH_CUBLAS_LM_HEAD = True
        token_out = greedy_token_call()
        sync_cuda()

        specs = [
            ("current_logits_cap_argmax", current_call, current_out),
            ("cublas_greedy_token", greedy_token_call, token_out),
            ("current_logits_cap_argmax_recheck", current_call, current_out),
        ]
        cases: list[dict[str, Any]] = []
        for case, call, output in specs:
            row: dict[str, Any] = {
                "case": case,
                "measurement": "cuda_graph",
                "median_us": None,
                "samples_us": [],
                "tokens_equal": False,
                "eligible": False,
                "error": None,
            }
            try:
                for _ in range(3):
                    call()
                sync_cuda()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call()
                for _ in range(3):
                    graph.replay()
                sync_cuda()
                tokens_equal = bool(torch.equal(output, token_reference))
                samples_us = []
                for _ in range(5):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(25):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples_us.append(float(start.elapsed_time(end)) * 40.0)
                row.update({
                    "median_us": float(statistics.median(samples_us)),
                    "samples_us": samples_us,
                    "tokens_equal": tokens_equal,
                    "eligible": tokens_equal,
                })
                del graph
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            cases.append(row)

    eligible = [
        row
        for row in cases
        if row["eligible"] and row["median_us"] is not None
    ]
    current = [
        row
        for row in eligible
        if str(row["case"]).startswith("current_logits_cap_argmax")
    ]
    greedy_token = next(
        (row for row in eligible if row["case"] == "cublas_greedy_token"),
        None,
    )
    current_reference_us = (
        min(float(row["median_us"]) for row in current)
        if current
        else None
    )
    current_stability_ratio = (
        max(float(row["median_us"]) for row in current) / current_reference_us
        if len(current) == 2 and current_reference_us
        else None
    )
    speedup = (
        current_reference_us / float(greedy_token["median_us"])
        if current_reference_us is not None and greedy_token is not None
        else 0.0
    )
    candidate_latency_ratio = (
        float(greedy_token["median_us"]) / current_reference_us
        if current_reference_us is not None and greedy_token is not None
        else None
    )
    apply_greedy_token = bool(
        greedy_token is not None
        and current_stability_ratio is not None
        and current_stability_ratio <= 1.03
        and candidate_latency_ratio is not None
        and candidate_latency_ratio <= 1.03
    )
    llama_model._GEMMA4_BATCH_CUBLAS_LM_HEAD = apply_greedy_token
    model._gemma4_batch_cublas_lm_head_hits = 0
    return {
        "rows": rows,
        "hidden": hidden_dim,
        "vocab": vocab_size,
        "dtype": "bf16",
        "device_name": device_name,
        "cases": cases,
        "decision": (
            "ENABLE_EXACT_TOKEN_FOR_SCHEDULER_GATE"
            if apply_greedy_token
            else "KEEP_CURRENT"
        ),
        "selected": (
            "cublas_greedy_token"
            if apply_greedy_token
            else "current_logits_cap_argmax"
        ),
        "minimum_promotion_speedup": 1.01,
        "maximum_current_stability_ratio": 1.03,
        "maximum_candidate_latency_ratio": 1.03,
        "current_reference_us": current_reference_us,
        "current_stability_ratio": current_stability_ratio,
        "greedy_token_speedup_vs_current": speedup,
        "greedy_token_latency_ratio_vs_current": candidate_latency_ratio,
    }


def run_scheduler_token_burst_gate(
    engine,
    prompts: list[list[int]],
    max_tokens: int,
    lm_head_gate: dict[str, Any],
) -> dict[str, Any]:
    """Promote GPU token feedback only after an exact same-VM end-to-end A/B."""
    import megagemm.models.llama as llama_model

    candidate_available = (
        lm_head_gate.get("selected") == "cublas_greedy_token"
    )

    def configure(
        enabled: bool,
        persistent_feedback: bool = False,
        fused_softcap: bool = False,
    ) -> None:
        active = bool(enabled and candidate_available)
        llama_model._GEMMA4_BATCH_CUBLAS_LM_HEAD = active
        llama_model._GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX = bool(
            active and fused_softcap
        )
        os.environ["MEGAGEMM_DECODE_GRAPH_TOKEN_BURST"] = "1" if active else "0"
        os.environ["MEGAGEMM_GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX"] = (
            "1" if active and fused_softcap else "0"
        )
        os.environ["MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK"] = (
            "1" if active and persistent_feedback else "0"
        )

    if not candidate_available:
        configure(False)
        return {
            "decision": "SKIP_NO_EXACT_TOKEN_PATH",
            "selected": "step_logits",
            "cases": [],
        }

    if (
        env_flag("MEGAGEMM_GEMMA4_B16_GRAPH_TOKEN_BURST_PROVEN")
        and env_flag("MEGAGEMM_GEMMA4_B16_FUSED_SOFTCAP_ARGMAX_PROVEN")
    ):
        persistent_proven = env_flag(
            "MEGAGEMM_GEMMA4_B16_PERSISTENT_TOKEN_FEEDBACK_PROVEN"
        )
        configure(
            True,
            persistent_feedback=persistent_proven,
            fused_softcap=True,
        )
        engine.model._gemma4_batch_cublas_lm_head_hits = 0
        engine.model._gemma4_batch_fused_softcap_argmax_hits = 0
        return {
            "decision": (
                "USE_PROVEN_FUSED_SOFTCAP_PERSISTENT_FEEDBACK"
                if persistent_proven
                else "USE_PROVEN_FUSED_SOFTCAP_ARGMAX"
            ),
            "selected": (
                "fused_softcap_argmax_persistent_feedback"
                if persistent_proven
                else "fused_softcap_argmax"
            ),
            "minimum_promotion_speedup": 1.01,
            "token_reference_source": "paid_exact_softcap_gate",
            "cases": [],
        }

    if env_flag("MEGAGEMM_GEMMA4_B16_GRAPH_TOKEN_BURST_PROVEN"):
        cases: dict[str, dict[str, Any]] = {
            "graph_token_burst": {
                "decode_ms": [],
                "tokens_exact": True,
                "token_checks": [],
            },
            "fused_softcap_argmax": {
                "decode_ms": [],
                "tokens_exact": True,
                "token_checks": [],
                "softcap_hits": [],
                "softcap_disabled": [],
                "softcap_errors": [],
            },
            "fused_softcap_argmax_persistent_feedback": {
                "decode_ms": [],
                "tokens_exact": True,
                "token_checks": [],
                "softcap_hits": [],
                "softcap_disabled": [],
                "softcap_errors": [],
                "persistent_feedback_steps": [],
                "feedback_copies": [],
                "vectorized_input_updates": [],
                "chain_input_updates_skipped": [],
                "graph_failures": [],
            },
        }

        def reset_softcap_runtime() -> None:
            engine.model._gemma4_batch_fused_softcap_argmax_hits = 0
            engine.model._gemma4_batch_fused_softcap_argmax_disable = False
            engine.model._gemma4_batch_fused_softcap_argmax_error = ""

        configure(True, persistent_feedback=False, fused_softcap=False)
        baseline_prime = run_megagemm_request(engine, prompts, max_tokens)
        reset_softcap_runtime()
        configure(True, persistent_feedback=False, fused_softcap=True)
        candidate_prime = run_megagemm_request(engine, prompts, max_tokens)
        candidate_prime_runtime = dict(candidate_prime.get("decode_runtime") or {})
        priming_comparison = token_matrix_comparison(
            baseline_prime["token_ids"],
            candidate_prime["token_ids"],
        )
        # Capture the chained-input graph before measuring it. Capture-time
        # execution is already excluded for the other graph variants and must
        # not contaminate this candidate's latency or token contract either.
        reset_softcap_runtime()
        configure(True, persistent_feedback=True, fused_softcap=True)
        persistent_prime = run_megagemm_request(engine, prompts, max_tokens)
        persistent_prime_runtime = dict(persistent_prime.get("decode_runtime") or {})
        persistent_priming_comparison = token_matrix_comparison(
            baseline_prime["token_ids"],
            persistent_prime["token_ids"],
        )

        token_reference: list[list[int]] | None = None
        order = (
            ("graph_token_burst", False, False),
            ("fused_softcap_argmax", False, True),
            ("fused_softcap_argmax_persistent_feedback", True, True),
            ("fused_softcap_argmax_persistent_feedback", True, True),
            ("fused_softcap_argmax", False, True),
            ("graph_token_burst", False, False),
        )
        for case_name, persistent_feedback, fused_softcap in order:
            reset_softcap_runtime()
            configure(
                True,
                persistent_feedback=persistent_feedback,
                fused_softcap=fused_softcap,
            )
            row = run_megagemm_request(engine, prompts, max_tokens)
            if token_reference is None:
                token_reference = row["token_ids"]
            comparison = token_matrix_comparison(token_reference, row["token_ids"])
            graph_stats = dict(row.get("decode_cuda_graphs") or {})
            runtime = dict(row.get("decode_runtime") or {})
            case = cases[case_name]
            case["decode_ms"].append(float(row.get("scheduler_decode_ms") or 0.0))
            case["tokens_exact"] = bool(case["tokens_exact"] and comparison["exact"])
            case["token_checks"].append({
                "exact": bool(comparison["exact"]),
                "mismatched_rows": int(comparison["mismatched_rows"]),
                "min_common_prefix_tokens": int(
                    comparison["min_common_prefix_tokens"]
                ),
                "first_mismatch": comparison["first_mismatch"],
            })
            if fused_softcap:
                case["softcap_hits"].append(int(
                    runtime.get("gemma4_batch_fused_softcap_argmax_hits", 0) or 0
                ))
                case["softcap_disabled"].append(bool(
                    runtime.get("gemma4_batch_fused_softcap_argmax_disabled", False)
                ))
                case["softcap_errors"].append(str(
                    runtime.get("gemma4_batch_fused_softcap_argmax_error", "") or ""
                ))
            if persistent_feedback:
                case["persistent_feedback_steps"].append(int(
                    graph_stats.get("persistent_token_feedback_steps", 0) or 0
                ))
                case["feedback_copies"].append(int(
                    graph_stats.get("token_feedback_copies", 0) or 0
                ))
                case["vectorized_input_updates"].append(int(
                    graph_stats.get("vectorized_input_updates", 0) or 0
                ))
                case["chain_input_updates_skipped"].append(int(
                    graph_stats.get("chain_input_updates_skipped", 0) or 0
                ))
                case["graph_failures"].append(int(
                    graph_stats.get("failures", 0) or 0
                ))

        for case in cases.values():
            values = [float(value) for value in case["decode_ms"]]
            case["decode_ms_median"] = float(statistics.median(values))
            case["stability_ratio"] = (
                max(values) / min(values) if values and min(values) > 0.0 else None
            )

        baseline = cases["graph_token_burst"]
        softcap = cases["fused_softcap_argmax"]
        combined = cases["fused_softcap_argmax_persistent_feedback"]
        expected_decode_steps = max(0, int(max_tokens) - 1)
        expected_persistent_steps = expected_decode_steps
        prime_runtime_by_case = {
            "fused_softcap_argmax": candidate_prime_runtime,
            "fused_softcap_argmax_persistent_feedback": persistent_prime_runtime,
        }
        softcap_capture_evidence = {
            case_name: {
                "hits": int(
                    runtime.get("gemma4_batch_fused_softcap_argmax_hits", 0) or 0
                ),
                "disabled": bool(
                    runtime.get(
                        "gemma4_batch_fused_softcap_argmax_disabled", False
                    )
                ),
                "error": str(
                    runtime.get("gemma4_batch_fused_softcap_argmax_error", "")
                    or ""
                ),
            }
            for case_name, runtime in prime_runtime_by_case.items()
        }
        softcap_contracts = {}
        for case_name, evidence in softcap_capture_evidence.items():
            case = cases[case_name]
            softcap_contracts[case_name] = bool(
                evidence["hits"] > 0
                and not evidence["disabled"]
                and not evidence["error"]
                and not any(bool(value) for value in case["softcap_disabled"])
                and not any(str(value) for value in case["softcap_errors"])
            )
        persistent_contract = bool(
            combined["persistent_feedback_steps"]
            and all(
                int(value) == expected_persistent_steps
                for value in combined["persistent_feedback_steps"]
            )
            and all(int(value) <= 1 for value in combined["feedback_copies"])
            and all(
                int(value) <= 1 for value in combined["vectorized_input_updates"]
            )
            and all(
                int(value) > 0 for value in combined["chain_input_updates_skipped"]
            )
            and all(int(value) == 0 for value in combined["graph_failures"])
        )
        baseline_stable = float(baseline["stability_ratio"] or 999.0) <= 1.03
        eligible: list[tuple[str, float]] = []
        minimum_speedup_by_case = {
            "fused_softcap_argmax": 1.002,
            "fused_softcap_argmax_persistent_feedback": 1.002,
        }
        minimum_savings_ms_by_case = {
            "fused_softcap_argmax": 1.0,
            "fused_softcap_argmax_persistent_feedback": 2.0,
        }
        for case_name, extra_contract in (
            ("fused_softcap_argmax", True),
            ("fused_softcap_argmax_persistent_feedback", persistent_contract),
        ):
            case = cases[case_name]
            candidate_ms = float(case["decode_ms_median"])
            savings_ms = float(baseline["decode_ms_median"]) - candidate_ms
            speedup = (
                float(baseline["decode_ms_median"]) / candidate_ms
                if candidate_ms > 0.0
                else 0.0
            )
            case["decode_speedup_vs_baseline"] = speedup
            case["decode_savings_ms_vs_baseline"] = savings_ms
            case["minimum_promotion_speedup"] = minimum_speedup_by_case[case_name]
            case["minimum_savings_ms"] = minimum_savings_ms_by_case[case_name]
            if (
                token_reference is not None
                and baseline["tokens_exact"]
                and case["tokens_exact"]
                and softcap_contracts[case_name]
                and extra_contract
                and baseline_stable
                and float(case["stability_ratio"] or 999.0) <= 1.03
                and speedup >= minimum_speedup_by_case[case_name]
                and savings_ms >= minimum_savings_ms_by_case[case_name]
            ):
                eligible.append((case_name, candidate_ms))

        selected = (
            min(eligible, key=lambda item: item[1])[0]
            if eligible
            else "graph_token_burst"
        )
        apply_softcap = selected != "graph_token_burst"
        apply_persistent = selected.endswith("persistent_feedback")
        configure(
            True,
            persistent_feedback=apply_persistent,
            fused_softcap=apply_softcap,
        )
        engine.model._gemma4_batch_cublas_lm_head_hits = 0
        reset_softcap_runtime()
        return {
            "decision": (
                "APPLY_FUSED_SOFTCAP_ARGMAX_PERSISTENT_FEEDBACK"
                if selected == "fused_softcap_argmax_persistent_feedback"
                else (
                    "APPLY_FUSED_SOFTCAP_ARGMAX"
                    if selected == "fused_softcap_argmax"
                    else "KEEP_GRAPH_TOKEN_BURST"
                )
            ),
            "selected": selected,
            "minimum_promotion_speedup": 1.002,
            "minimum_promotion_savings_ms": 1.0,
            "maximum_stability_ratio": 1.03,
            "decode_speedup": float(
                cases[selected].get("decode_speedup_vs_baseline", 1.0)
            ),
            "softcap_capture_evidence": softcap_capture_evidence,
            "softcap_contracts": softcap_contracts,
            "persistent_contract": persistent_contract,
            "expected_decode_steps": expected_decode_steps,
            "expected_persistent_steps": expected_persistent_steps,
            "token_reference_source": "steady_state_graph_token_burst",
            "priming_tokens_exact_diagnostic_only": bool(
                priming_comparison["exact"]
            ),
            "priming_first_mismatch": priming_comparison["first_mismatch"],
            "persistent_priming_tokens_exact_diagnostic_only": bool(
                persistent_priming_comparison["exact"]
            ),
            "persistent_priming_first_mismatch": (
                persistent_priming_comparison["first_mismatch"]
            ),
            "cases": cases,
        }

    cases: dict[str, dict[str, Any]] = {
        "step_logits": {
            "decode_ms": [],
            "total_ms": [],
            "tokens_exact": True,
            "token_checks": [],
        },
        "graph_token_burst": {
            "decode_ms": [],
            "total_ms": [],
            "tokens_exact": True,
            "token_checks": [],
        },
    }

    # These two requests only populate the two shared CUDA-graph variants.
    # Capture-time execution is intentionally not a correctness oracle because
    # its first eager/capture steps differ from steady-state graph replay.
    configure(False)
    baseline_prime = run_megagemm_request(engine, prompts, max_tokens)
    configure(True)
    candidate_prime = run_megagemm_request(engine, prompts, max_tokens)
    priming_comparison = token_matrix_comparison(
        baseline_prime["token_ids"],
        candidate_prime["token_ids"],
    )

    token_reference: list[list[int]] | None = None
    order = (
        ("step_logits", False),
        ("graph_token_burst", True),
        ("graph_token_burst", True),
        ("step_logits", False),
    )
    for case_name, enabled in order:
        configure(enabled)
        row = run_megagemm_request(engine, prompts, max_tokens)
        if token_reference is None:
            if case_name != "step_logits":
                raise RuntimeError("steady-state token oracle must come from step_logits")
            token_reference = row["token_ids"]
        comparison = token_matrix_comparison(token_reference, row["token_ids"])
        case = cases[case_name]
        case["decode_ms"].append(float(row.get("scheduler_decode_ms") or 0.0))
        case["total_ms"].append(float(row["total_ms"]))
        case["tokens_exact"] = bool(case["tokens_exact"] and comparison["exact"])
        case["token_checks"].append({
            "exact": bool(comparison["exact"]),
            "mismatched_rows": int(comparison["mismatched_rows"]),
            "min_common_prefix_tokens": int(
                comparison["min_common_prefix_tokens"]
            ),
            "first_mismatch": comparison["first_mismatch"],
        })

    for case in cases.values():
        values = [float(value) for value in case["decode_ms"]]
        case["decode_ms_median"] = float(statistics.median(values))
        case["stability_ratio"] = (
            max(values) / min(values) if values and min(values) > 0.0 else None
        )

    baseline = cases["step_logits"]
    candidate = cases["graph_token_burst"]
    speedup = (
        float(baseline["decode_ms_median"])
        / float(candidate["decode_ms_median"])
        if float(candidate["decode_ms_median"]) > 0.0
        else 0.0
    )
    apply_burst = bool(
        token_reference is not None
        and baseline["tokens_exact"]
        and candidate["tokens_exact"]
        and float(baseline["stability_ratio"] or 999.0) <= 1.03
        and float(candidate["stability_ratio"] or 999.0) <= 1.03
        and speedup >= 1.01
    )
    configure(apply_burst)
    engine.model._gemma4_batch_cublas_lm_head_hits = 0
    return {
        "decision": "APPLY_GRAPH_TOKEN_BURST" if apply_burst else "KEEP_STEP_LOGITS",
        "selected": "graph_token_burst" if apply_burst else "step_logits",
        "minimum_promotion_speedup": 1.01,
        "maximum_stability_ratio": 1.03,
        "decode_speedup": speedup,
        "token_reference_source": "first_post_capture_step_logits",
        "priming_tokens_exact_diagnostic_only": bool(priming_comparison["exact"]),
        "priming_first_mismatch": priming_comparison["first_mismatch"],
        "cases": cases,
    }


def run_attention_decode_kernel_gate(
    engine,
    prompts: list[str],
    max_tokens: int,
) -> dict[str, Any]:
    """Apply the already-proven exact Gemma4 attention configuration."""
    import megagemm.kernels.paged_attention as paged_attention

    model = engine.model
    model_dtype = next(model.parameters()).dtype
    device_name = torch.cuda.get_device_name(next(model.parameters()).device)
    layer_shapes = []
    for layer_index, layer in enumerate(model.layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        layer_shapes.append({
            "layer": layer_index,
            "type": str(getattr(attention, "layer_type", "")),
            "q_heads": int(getattr(attention, "num_q_heads", 0) or 0),
            "kv_heads": int(getattr(attention, "num_kv_heads", 0) or 0),
            "head_dim": int(getattr(attention, "head_dim", 0) or 0),
        })

    topology = {
        "layers": len(layer_shapes),
        "sliding_gqa2_h256_layers": sum(
            shape["type"] == "sliding_attention"
            and shape["q_heads"] == 16
            and shape["kv_heads"] == 8
            and shape["head_dim"] == 256
            for shape in layer_shapes
        ),
        "full_gqa8_h512_layers": sum(
            shape["type"] == "full_attention"
            and shape["q_heads"] == 16
            and shape["kv_heads"] == 2
            and shape["head_dim"] == 512
            for shape in layer_shapes
        ),
    }

    baseline_config = {
        "gqa2_direct": False,
        "grouped_segmented": True,
        "warps_h256": 8,
        "warps_h512": 4,
    }

    def configure(config: dict[str, Any]) -> None:
        os.environ["MEGAGEMM_PAGED_DECODE_GQA2"] = "0"
        os.environ["MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_DECODE"] = (
            "1" if bool(config["grouped_segmented"]) else "0"
        )
        os.environ["MEGAGEMM_PAGED_DECODE_WARPS"] = "0"
        os.environ["MEGAGEMM_PAGED_DECODE_WARPS_H256"] = str(
            int(config["warps_h256"])
        )
        os.environ["MEGAGEMM_PAGED_DECODE_WARPS_H512"] = str(
            int(config["warps_h512"])
        )
        # A rejected experimental compile must not poison the selected fallback.
        paged_attention._GQA2_DECODE_DISABLED = False

    supported = bool(
        len(prompts) == 16
        and max_tokens >= 2
        and str(getattr(model.config, "model_type", "")) == "gemma4_text"
        and model_dtype == torch.bfloat16
        and "A100" in device_name.upper()
        and topology["layers"] == 30
        and topology["sliding_gqa2_h256_layers"] == 25
        and topology["full_gqa8_h512_layers"] == 5
    )
    if not supported:
        configure(baseline_config)
        return {
            "decision": "SKIP_UNSUPPORTED_SHAPE",
            "selected": "current",
            "selected_config": baseline_config,
            "device_name": device_name,
            "dtype": str(model_dtype),
            "rows": len(prompts),
            "topology": topology,
            "cases": [],
        }
    configure(baseline_config)
    return {
        "decision": "USE_PROVEN_EXACT_BASELINE",
        "selected": "current",
        "selected_config": baseline_config,
        "device_name": device_name,
        "dtype": "bf16",
        "rows": len(prompts),
        "topology": topology,
        "cases": [],
        "promoted_baseline": {
            "case": "full_h512_w4",
            "run_id": "gemma4_moe_ab_20260725_003420",
            "tokens_exact": True,
            "speedup": 1.0139743365538179,
        },
        "retired_candidates": [
            {
                "case": "sliding_h256_w4_fixed_qk32",
                "run_id": "gemma4_moe_ab_20260725_010328",
                "baseline_decode_ms": 978.4286900003281,
                "candidate_decode_ms": 930.540703999668,
                "reason": "changed 3 of 16 greedy token rows",
            },
        ],
    }

    # Match the paid B16 workload exactly (up to the harness's standard 64
    # tokens), so promotion cannot hide drift that appears late in generation.
    gate_tokens = min(int(max_tokens), 64)
    candidate_specs = [
        {"case": "current", **baseline_config},
        {"case": "current_recheck", **baseline_config},
    ]

    token_reference: list[list[int]] | None = None
    cases: list[dict[str, Any]] = []
    for spec in candidate_specs:
        config = {
            "gqa2_direct": bool(spec["gqa2_direct"]),
            "grouped_segmented": bool(spec["grouped_segmented"]),
            "warps_h256": int(spec["warps_h256"]),
            "warps_h512": int(spec["warps_h512"]),
        }
        row: dict[str, Any] = {
            "case": str(spec["case"]),
            "config": config,
            "decode_ms": [],
            "total_ms": [],
            "graph_checks": [],
            "token_checks": [],
            "tokens_exact": True,
            "eligible": False,
            "error": None,
        }
        try:
            configure(config)
            stats_before = paged_attention.paged_decode_runtime_stats()
            prime = run_megagemm_request(engine, prompts, gate_tokens)
            prime_graph = dict(prime.get("decode_cuda_graphs") or {})
            for _ in range(3):
                measured = run_megagemm_request(engine, prompts, gate_tokens)
                graph = dict(measured.get("decode_cuda_graphs") or {})
                graph_check = {
                    "enabled": bool(graph.get("enabled")),
                    "failures": int(graph.get("failures", 0) or 0),
                    "replays": int(graph.get("replays", 0) or 0),
                    "captured_graphs": int(
                        graph.get("captured_graphs", 0) or 0
                    ),
                }
                row["graph_checks"].append(graph_check)
                if (
                    not graph_check["enabled"]
                    or graph_check["failures"] != 0
                    or graph_check["replays"] <= 0
                ):
                    raise RuntimeError(
                        f"decode CUDA graph did not replay: {graph_check}"
                    )

                decode_ms = float(measured.get("scheduler_decode_ms") or 0.0)
                if decode_ms <= 0.0:
                    raise RuntimeError(
                        f"invalid scheduler decode timing: {decode_ms}ms"
                    )
                if token_reference is None:
                    if spec["case"] != "current":
                        raise RuntimeError(
                            "attention token oracle must come from current"
                        )
                    token_reference = measured["token_ids"]
                comparison = token_matrix_comparison(
                    token_reference,
                    measured["token_ids"],
                )
                row["token_checks"].append({
                    "exact": bool(comparison["exact"]),
                    "mismatched_rows": int(comparison["mismatched_rows"]),
                    "min_common_prefix_tokens": int(
                        comparison["min_common_prefix_tokens"]
                    ),
                    "first_mismatch": comparison["first_mismatch"],
                })
                row["tokens_exact"] = bool(
                    row["tokens_exact"] and comparison["exact"]
                )
                row["decode_ms"].append(decode_ms)
                row["total_ms"].append(float(measured["total_ms"]))

            stats_after = paged_attention.paged_decode_runtime_stats()
            generic_hit_delta = int(
                stats_after["generic_direct_hits"]
                - stats_before["generic_direct_hits"]
            )
            decode_values = [float(value) for value in row["decode_ms"]]
            median_decode_ms = float(statistics.median(decode_values))
            stability_ratio = max(decode_values) / min(decode_values)
            shape_policy_exercised = bool(
                spec["case"] in {"current", "current_recheck"}
                or generic_hit_delta > 0
            )
            row.update({
                "prime_graph": {
                    "enabled": bool(prime_graph.get("enabled")),
                    "failures": int(prime_graph.get("failures", 0) or 0),
                    "replays": int(prime_graph.get("replays", 0) or 0),
                },
                "median_decode_ms": median_decode_ms,
                "median_decode_tok_s": (
                    len(prompts) * (gate_tokens - 1)
                    / (median_decode_ms / 1000.0)
                ),
                "stability_ratio": stability_ratio,
                "generic_direct_hit_delta": generic_hit_delta,
                "gqa2_disabled_after": bool(stats_after["gqa2_disabled"]),
                "shape_policy_exercised": shape_policy_exercised,
                "eligible": bool(
                    row["tokens_exact"]
                    and shape_policy_exercised
                    and stability_ratio <= 1.03
                ),
            })
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if spec["case"] == "current":
                configure(baseline_config)
                raise
        cases.append(row)

    eligible = [
        row
        for row in cases
        if row["eligible"] and row.get("median_decode_ms") is not None
    ]
    current_rows = [
        row
        for row in eligible
        if row["case"] in {"current", "current_recheck"}
    ]
    candidate_rows = [
        row
        for row in eligible
        if row["case"] not in {"current", "current_recheck"}
    ]
    current_reference_ms = (
        min(float(row["median_decode_ms"]) for row in current_rows)
        if current_rows
        else None
    )
    current_stability_ratio = (
        max(float(row["median_decode_ms"]) for row in current_rows)
        / current_reference_ms
        if len(current_rows) == 2 and current_reference_ms
        else None
    )
    measured_winner = (
        min(candidate_rows, key=lambda row: float(row["median_decode_ms"]))
        if candidate_rows
        else None
    )
    measured_speedup = (
        current_reference_ms / float(measured_winner["median_decode_ms"])
        if current_reference_ms is not None and measured_winner is not None
        else None
    )
    apply_candidate = bool(
        current_reference_ms is not None
        and current_stability_ratio is not None
        and current_stability_ratio <= 1.03
        and measured_winner is not None
        and measured_speedup is not None
        and measured_speedup >= 1.02
    )
    selected_name = (
        str(measured_winner["case"]) if apply_candidate else "current"
    )
    selected_config = (
        dict(measured_winner["config"])
        if apply_candidate
        else dict(baseline_config)
    )
    configure(selected_config)

    selected_decode_ms = (
        float(measured_winner["median_decode_ms"])
        if apply_candidate and measured_winner is not None
        else current_reference_ms
    )
    savings_ms = (
        float(current_reference_ms - selected_decode_ms)
        if (
            apply_candidate
            and current_reference_ms is not None
            and selected_decode_ms is not None
        )
        else 0.0
    )
    return {
        "decision": "APPLY" if apply_candidate else "KEEP_CURRENT",
        "selected": selected_name,
        "selected_config": selected_config,
        "device_name": device_name,
        "dtype": "bf16",
        "rows": len(prompts),
        "gate_tokens_per_row": gate_tokens,
        "topology": topology,
        "minimum_promotion_speedup": 1.02,
        "maximum_stability_ratio": 1.03,
        "current_reference_decode_ms": current_reference_ms,
        "current_stability_ratio": current_stability_ratio,
        "measured_winner": (
            None if measured_winner is None else measured_winner["case"]
        ),
        "measured_speedup_vs_current": measured_speedup,
        "estimated_savings_ms_for_gate_request": savings_ms,
        "estimated_savings_ms_per_decode_step": (
            savings_ms / (gate_tokens - 1)
        ),
        "promoted_baseline": {
            "case": "full_h512_w4",
            "run_id": "gemma4_moe_ab_20260725_003420",
            "tokens_exact": True,
            "speedup": 1.0139743365538179,
            "reason": "v79 exact and stable; retained despite sub-2-percent rerun gate",
        },
        "retired_candidates": [
            {
                "case": "sliding_h256_w4",
                "run_id": "gemma4_moe_ab_20260725_003420",
                "baseline_decode_ms": 992.167480000262,
                "candidate_decode_ms": 941.2031780007055,
                "reason": "v79 changed 4 of 16 greedy token rows",
            },
            {
                "case": "gqa2_direct_w8",
                "run_id": "gemma4_moe_ab_20260725_001448",
                "baseline_decode_ms": 990.6452070000569,
                "candidate_decode_ms": 1022.5569540000379,
                "reason": "v78 was 3.2 percent slower with exact tokens",
            },
            {
                "case": "gqa2_direct_w4",
                "run_id": "gemma4_moe_ab_20260725_001448",
                "baseline_decode_ms": 990.6452070000569,
                "candidate_decode_ms": 949.2625489999114,
                "reason": "v78 changed 3 of 16 greedy token rows",
            },
            {
                "case": "generic_global_w4",
                "run_id": "gemma4_moe_ab_20260725_001448",
                "baseline_decode_ms": 990.6452070000569,
                "candidate_decode_ms": 929.586896000103,
                "reason": "v78 changed 3 of 16 greedy token rows",
            },
        ],
        "cases": cases,
    }


def _run_legacy_fused_next_attn_norm_decode_gate(
    engine,
    prompts: list[str],
    max_tokens: int,
) -> dict[str, Any]:
    """Gate the post-MoE -> next-layer RMSNorm chain on full B16 replay."""
    model = engine.model
    model_dtype = next(model.parameters()).dtype
    device_name = torch.cuda.get_device_name(next(model.parameters()).device)
    supported_shape = bool(
        len(prompts) == 16
        and max_tokens >= 2
        and str(getattr(model.config, "model_type", "")) == "gemma4_text"
        and int(getattr(model.config, "hidden_size", 0) or 0) == 2816
        and len(model.layers) == 30
        and model_dtype == torch.bfloat16
        and "A100" in device_name.upper()
        and all(
            bool(getattr(layer, "is_moe_layer", False))
            and int(getattr(layer, "hidden_size_per_layer_input", 0) or 0) == 0
            for layer in model.layers
        )
    )

    def reset_candidate_failure_state() -> None:
        for layer in model.layers:
            experts = getattr(getattr(layer, "mlp", None), "experts", None)
            if experts is None:
                continue
            experts._grouped_decode_disabled = False
            experts._grouped_decode_fail_reason = ""
            workspace = getattr(experts, "_grouped_decode_workspace", None)
            if isinstance(workspace, dict):
                workspace.pop("expert_grouped_compact_decode_disabled", None)
                workspace.pop("expert_grouped_compact_decode_fail_reason", None)

    def configure(enabled: bool) -> None:
        reset_candidate_failure_state()
        os.environ["MEGAGEMM_GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE"] = (
            "1" if enabled else "0"
        )
        runtime_supported = bool(
            getattr(
                model,
                "_gemma4_flat_fused_next_attn_norm_supported",
                False,
            )
        )
        model._gemma4_flat_fused_next_attn_norm_enabled = bool(
            enabled and runtime_supported
        )
        policy = getattr(model, "_gemma4_flat_parallel_moe_policy", None)
        if isinstance(policy, dict):
            policy["fused_next_attn_norm_requested"] = bool(enabled)
            policy["fused_next_attn_norm_enabled"] = bool(
                model._gemma4_flat_fused_next_attn_norm_enabled
            )

    if not supported_shape:
        configure(False)
        return {
            "decision": "SKIP_UNSUPPORTED_SHAPE",
            "selected": "current",
            "enabled": False,
            "device_name": device_name,
            "dtype": str(model_dtype),
            "rows": len(prompts),
            "cases": [],
        }

    gate_tokens = min(int(max_tokens), 64)
    specs = (
        ("current", False),
        ("fused_post_moe_next_attn_norm", True),
        ("current_recheck", False),
    )
    token_reference: list[list[int]] | None = None
    cases: list[dict[str, Any]] = []
    for case_name, enabled in specs:
        row: dict[str, Any] = {
            "case": case_name,
            "enabled": bool(enabled),
            "decode_ms": [],
            "total_ms": [],
            "graph_checks": [],
            "token_checks": [],
            "tokens_exact": True,
            "eligible": False,
            "error": None,
        }
        try:
            configure(enabled)
            hits_before = int(
                getattr(model, "_gemma4_flat_fused_next_attn_norm_hits", 0)
            )
            scalar_hits_before = int(
                getattr(model, "_gemma4_flat_fused_layer_scalar_hits", 0)
            )
            prime = run_megagemm_request(engine, prompts, gate_tokens)
            prime_graph = dict(prime.get("decode_cuda_graphs") or {})
            # The first baseline request allocates flat buffers. Re-apply the
            # candidate after allocation before measuring/capturing it.
            configure(enabled)
            for _ in range(3):
                measured = run_megagemm_request(engine, prompts, gate_tokens)
                graph = dict(measured.get("decode_cuda_graphs") or {})
                graph_check = {
                    "enabled": bool(graph.get("enabled")),
                    "failures": int(graph.get("failures", 0) or 0),
                    "replays": int(graph.get("replays", 0) or 0),
                    "captured_graphs": int(
                        graph.get("captured_graphs", 0) or 0
                    ),
                }
                row["graph_checks"].append(graph_check)
                if (
                    not graph_check["enabled"]
                    or graph_check["failures"] != 0
                    or graph_check["replays"] <= 0
                ):
                    raise RuntimeError(
                        f"decode CUDA graph did not replay: {graph_check}"
                    )

                decode_ms = float(measured.get("scheduler_decode_ms") or 0.0)
                if decode_ms <= 0.0:
                    raise RuntimeError(
                        f"invalid scheduler decode timing: {decode_ms}ms"
                    )
                if token_reference is None:
                    if case_name != "current":
                        raise RuntimeError(
                            "next-attention RMSNorm token oracle must come from current"
                        )
                    token_reference = measured["token_ids"]
                comparison = token_matrix_comparison(
                    token_reference,
                    measured["token_ids"],
                )
                row["token_checks"].append({
                    "exact": bool(comparison["exact"]),
                    "mismatched_rows": int(comparison["mismatched_rows"]),
                    "min_common_prefix_tokens": int(
                        comparison["min_common_prefix_tokens"]
                    ),
                    "first_mismatch": comparison["first_mismatch"],
                })
                row["tokens_exact"] = bool(
                    row["tokens_exact"] and comparison["exact"]
                )
                row["decode_ms"].append(decode_ms)
                row["total_ms"].append(float(measured["total_ms"]))

            hits_after = int(
                getattr(model, "_gemma4_flat_fused_next_attn_norm_hits", 0)
            )
            scalar_hits_after = int(
                getattr(model, "_gemma4_flat_fused_layer_scalar_hits", 0)
            )
            decode_values = [float(value) for value in row["decode_ms"]]
            median_decode_ms = float(statistics.median(decode_values))
            stability_ratio = max(decode_values) / min(decode_values)
            hit_delta = hits_after - hits_before
            scalar_hit_delta = scalar_hits_after - scalar_hits_before
            path_exercised = bool(
                (enabled and hit_delta > 0 and scalar_hit_delta > 0)
                or (not enabled and hit_delta == 0 and scalar_hit_delta == 0)
            )
            row.update({
                "prime_graph": {
                    "enabled": bool(prime_graph.get("enabled")),
                    "failures": int(prime_graph.get("failures", 0) or 0),
                    "replays": int(prime_graph.get("replays", 0) or 0),
                },
                "runtime_supported": bool(
                    getattr(
                        model,
                        "_gemma4_flat_fused_next_attn_norm_supported",
                        False,
                    )
                ),
                "next_norm_hit_delta": hit_delta,
                "layer_scalar_hit_delta": scalar_hit_delta,
                "path_exercised": path_exercised,
                "median_decode_ms": median_decode_ms,
                "median_decode_tok_s": (
                    len(prompts) * (gate_tokens - 1)
                    / (median_decode_ms / 1000.0)
                ),
                "stability_ratio": stability_ratio,
                "eligible": bool(
                    row["tokens_exact"]
                    and path_exercised
                    and stability_ratio <= 1.03
                ),
            })
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            configure(False)
            if case_name == "current":
                raise
        cases.append(row)

    eligible = [
        row
        for row in cases
        if row["eligible"] and row.get("median_decode_ms") is not None
    ]
    current_rows = [
        row
        for row in eligible
        if row["case"] in {"current", "current_recheck"}
    ]
    candidate = next(
        (
            row
            for row in eligible
            if row["case"] == "fused_post_moe_next_attn_norm"
        ),
        None,
    )
    current_reference_ms = (
        min(float(row["median_decode_ms"]) for row in current_rows)
        if current_rows
        else None
    )
    current_stability_ratio = (
        max(float(row["median_decode_ms"]) for row in current_rows)
        / current_reference_ms
        if len(current_rows) == 2 and current_reference_ms
        else None
    )
    speedup = (
        current_reference_ms / float(candidate["median_decode_ms"])
        if current_reference_ms is not None and candidate is not None
        else None
    )
    apply_candidate = bool(
        current_stability_ratio is not None
        and current_stability_ratio <= 1.03
        and speedup is not None
        and speedup >= 1.01
    )
    configure(apply_candidate)
    if not apply_candidate:
        model._gemma4_flat_fused_next_attn_norm_hits = 0
        model._gemma4_flat_fused_layer_scalar_hits = 0
    return {
        "decision": "APPLY" if apply_candidate else "KEEP_CURRENT",
        "selected": (
            "fused_post_moe_next_attn_norm" if apply_candidate else "current"
        ),
        "enabled": apply_candidate,
        "device_name": device_name,
        "dtype": "bf16",
        "rows": len(prompts),
        "gate_tokens_per_row": gate_tokens,
        "minimum_promotion_speedup": 1.01,
        "maximum_stability_ratio": 1.03,
        "current_reference_decode_ms": current_reference_ms,
        "current_stability_ratio": current_stability_ratio,
        "measured_speedup_vs_current": speedup,
        "estimated_savings_ms_for_gate_request": (
            float(current_reference_ms - float(candidate["median_decode_ms"]))
            if apply_candidate
            and current_reference_ms is not None
            and candidate is not None
            else 0.0
        ),
        "cases": cases,
    }


def run_fused_next_attn_norm_decode_gate(
    engine,
    prompts: list[str],
    max_tokens: int,
) -> dict[str, Any]:
    """Enable the exact v81 winner without repeating its paid full-request A/B."""
    model = engine.model
    model_dtype = next(model.parameters()).dtype
    device_name = torch.cuda.get_device_name(next(model.parameters()).device)
    supported = bool(
        len(prompts) == 16
        and max_tokens >= 2
        and str(getattr(model.config, "model_type", "")) == "gemma4_text"
        and int(getattr(model.config, "hidden_size", 0) or 0) == 2816
        and len(model.layers) == 30
        and model_dtype == torch.bfloat16
        and "A100" in device_name.upper()
        and all(
            bool(getattr(layer, "is_moe_layer", False))
            and int(getattr(layer, "hidden_size_per_layer_input", 0) or 0) == 0
            for layer in model.layers
        )
    )
    os.environ["MEGAGEMM_GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE"] = (
        "1" if supported else "0"
    )
    # Flat-decode support is materialized during the first capture, after this
    # policy gate runs. Select the proven topology here; post-capture runtime
    # validation below still requires actual support, enablement, and hits.
    enabled = bool(supported)
    model._gemma4_flat_fused_next_attn_norm_enabled = enabled
    policy = getattr(model, "_gemma4_flat_parallel_moe_policy", None)
    if isinstance(policy, dict):
        policy["fused_next_attn_norm_requested"] = bool(supported)
        policy["fused_next_attn_norm_enabled"] = enabled
    return {
        "decision": (
            "USE_PROVEN_EXACT_BASELINE" if enabled else "SKIP_UNSUPPORTED_SHAPE"
        ),
        "selected": (
            "fused_post_moe_next_attn_norm" if enabled else "current"
        ),
        "enabled": enabled,
        "device_name": device_name,
        "dtype": str(model_dtype),
        "rows": len(prompts),
        "cases": [],
        "evidence": {
            "run_id": "gemma4_moe_ab_20260727_004118",
            "harness_rev": "gemma4-ab-qwen-snapshot-v81-next-attn-norm-chain",
            "tokens_exact": True,
            "rows": 16,
            "tokens_per_row": 64,
            "baseline_decode_ms": 978.7455109999428,
            "candidate_decode_ms": 966.9392049997896,
            "speedup": 1.0122099775654,
            "estimated_savings_ms": 11.8063060001532,
            "graph_replay_stable": True,
        },
    }


def run_compact_kernel_gate(
    engine,
    prompts: list[str],
    max_tokens: int,
) -> dict[str, Any]:
    import megagemm.kernels.qwen3_moe as qwen3_moe_kernel
    from megagemm.kernels.qwen3_moe import (
        _fallback_grouped_moe,
        qwen3_moe_grouped_decode,
    )

    layer = engine.model.layers[0]
    moe = layer.mlp
    experts = moe.experts
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260715)
    hidden = torch.randn(
        (16, experts.hidden_dim),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    with torch.inference_mode():
        _, decode_weights_before, decode_experts_before = moe.gate(
            hidden[:8],
            is_prefill=False,
        )
        decode_weights_before = decode_weights_before.clone()
        decode_experts_before = decode_experts_before.clone()
        decode_logits_before = moe.gate._decode_router_logits_by_rows[8]
        decode_logits_ptr = int(decode_logits_before.data_ptr())
        decode_workspace_before = moe.gate._decode_topk_workspaces[8]

        routing_weights, selected_experts = moe.gate.route(hidden)
        num_experts = int(experts.gate_up_proj.shape[0])
        route_counts = torch.bincount(
            selected_experts.reshape(-1),
            minlength=num_experts,
        )
        active_experts = int((route_counts > 0).sum().item())
        singleton_experts = int((route_counts == 1).sum().item())
        moe.gate(hidden[:4], is_prefill=False)

        _, decode_weights_after, decode_experts_after = moe.gate(
            hidden[:8],
            is_prefill=False,
        )
        decode_weights_after = decode_weights_after.clone()
        decode_experts_after = decode_experts_after.clone()

        assignment_workspace: dict[str, Any] = {}
        baseline = qwen3_moe_grouped_decode(
            hidden,
            experts.gate_up_proj,
            experts.down_proj,
            selected_experts,
            routing_weights,
            activation=experts.hidden_act,
            out=torch.empty_like(hidden),
            workspace=assignment_workspace,
            max_assignments=128,
            expert_grouped_compact=False,
            assignment_partial_reduce=True,
        ).clone()
        assignment_repeat = qwen3_moe_grouped_decode(
            hidden,
            experts.gate_up_proj,
            experts.down_proj,
            selected_experts,
            routing_weights,
            activation=experts.hidden_act,
            out=torch.empty_like(hidden),
            workspace={},
            max_assignments=128,
            expert_grouped_compact=False,
            assignment_partial_reduce=True,
        ).clone()
        reference = _fallback_grouped_moe(
            hidden,
            experts.gate_up_proj,
            experts.down_proj,
            selected_experts,
            routing_weights,
            experts.hidden_act,
            torch.empty_like(hidden),
        )

        def apply_compact_config(config: dict[str, Any]) -> None:
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST = bool(
                config["active_list"]
            )
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT = bool(
                config["active_list_early_exit"]
            )
            os.environ[
                "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST"
            ] = "1" if config["active_list"] else "0"
            os.environ[
                "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT"
            ] = "1" if config["active_list_early_exit"] else "0"
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK = bool(
                config["expert_grid_pack"]
            )
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS = bool(
                config["coalesced_weights"]
            )
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N = int(
                config["gate_block_n"]
            )
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N = int(
                config["down_block_n"]
            )
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS = int(
                config["num_warps"]
            )
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_NUM_STAGES = int(
                config["num_stages"]
            )
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES = int(
                config["gate_num_stages"]
            )
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES = int(
                config["down_num_stages"]
            )
            os.environ[
                "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES"
            ] = str(int(config["gate_num_stages"]))
            os.environ[
                "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES"
            ] = str(int(config["down_num_stages"]))
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM = int(
                config["experts_per_program"]
            )
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_PAIRED_GATE_UP_DOT = bool(
                config["paired_gate_up_dot"]
            )
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_SPLIT_GATE_UP = bool(
                config["split_gate_up"]
            )
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT = bool(
                config["empty_expert_early_exit"]
            )
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID = bool(
                config["l2_grouped_grid"]
            )
            qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_L2_GROUP_SIZE = int(
                config["l2_group_size"]
            )

        def compact_call(
            workspace: dict[str, Any],
            out: torch.Tensor,
        ) -> torch.Tensor:
            return qwen3_moe_grouped_decode(
                hidden,
                experts.gate_up_proj,
                experts.down_proj,
                selected_experts,
                routing_weights,
                activation=experts.hidden_act,
                out=out,
                workspace=workspace,
                max_assignments=128,
                expert_grouped_compact=True,
                expert_grouped_min_rows=9,
                expert_grouped_max_rows=16,
                expert_grouped_compact_partial_reduce=True,
            )

        def assignment_call(
            workspace: dict[str, Any],
            out: torch.Tensor,
        ) -> torch.Tensor:
            return qwen3_moe_grouped_decode(
                hidden,
                experts.gate_up_proj,
                experts.down_proj,
                selected_experts,
                routing_weights,
                activation=experts.hidden_act,
                out=out,
                workspace=workspace,
                max_assignments=128,
                expert_grouped_compact=False,
                assignment_partial_reduce=True,
            )

        current_config = {
            "active_list": bool(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST
            ),
            "active_list_early_exit": bool(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT
            ),
            "expert_grid_pack": bool(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK
            ),
            "coalesced_weights": bool(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS
            ),
            "gate_block_n": int(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_BLOCK_N
            ),
            "down_block_n": int(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_BLOCK_N
            ),
            "num_warps": int(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_NUM_WARPS
            ),
            "num_stages": int(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_NUM_STAGES
            ),
            "gate_num_stages": int(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES
            ),
            "down_num_stages": int(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES
            ),
            "experts_per_program": int(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM
            ),
            "paired_gate_up_dot": bool(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_PAIRED_GATE_UP_DOT
            ),
            "split_gate_up": bool(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_SPLIT_GATE_UP
            ),
            "empty_expert_early_exit": bool(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT
            ),
            "l2_grouped_grid": bool(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID
            ),
            "l2_group_size": int(
                qwen3_moe_kernel._CFG_EXPERT_GROUPED_COMPACT_L2_GROUP_SIZE
            ),
        }
        candidate_specs = [
            {
                "case": "current",
                "path": "expert_grouped_compact",
                **current_config,
            },
            {
                "case": "current_recheck",
                "path": "expert_grouped_compact",
                **current_config,
            },
        ]

        tune_cases: list[dict[str, Any]] = []
        for spec in candidate_specs:
            config = {
                key: spec[key]
                for key in (
                    "active_list",
                    "active_list_early_exit",
                    "expert_grid_pack",
                    "coalesced_weights",
                    "gate_block_n",
                    "down_block_n",
                    "num_warps",
                    "num_stages",
                    "gate_num_stages",
                    "down_num_stages",
                    "experts_per_program",
                    "paired_gate_up_dot",
                    "split_gate_up",
                    "empty_expert_early_exit",
                    "l2_grouped_grid",
                    "l2_group_size",
                )
            }
            path = str(spec["path"])
            row: dict[str, Any] = {
                "case": str(spec["case"]),
                "path": path,
                "config": dict(config),
                "measurement": "cuda_graph",
                "median_us": None,
                "samples_us": [],
                "max_abs_error": None,
                "cosine": None,
                "repeat_max_abs_error": None,
                "eligible": False,
                "error": None,
            }
            try:
                apply_compact_config(config)
                candidate_workspace: dict[str, Any] = {}
                candidate_out = torch.empty_like(hidden)
                kernel_call = (
                    compact_call
                    if path == "expert_grouped_compact"
                    else assignment_call
                )
                for _ in range(3):
                    kernel_call(candidate_workspace, candidate_out)
                sync_cuda()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    kernel_call(candidate_workspace, candidate_out)
                for _ in range(3):
                    graph.replay()
                sync_cuda()
                first = candidate_out.clone()
                graph.replay()
                sync_cuda()
                second = candidate_out.clone()
                sync_cuda()

                actual_path = str(
                    candidate_workspace.get("grouped_decode_last_path", "") or ""
                )
                if actual_path != path:
                    failure = str(
                        candidate_workspace.get(
                            "expert_grouped_compact_decode_fail_reason", ""
                        )
                        or ""
                    )
                    raise RuntimeError(
                        "candidate fell through to a different decode path: "
                        f"expected={path} actual={actual_path} failure={failure!r}"
                    )
                max_abs_error = float(
                    (first.float() - baseline.float()).abs().max().item()
                )
                cosine = float(
                    torch.nn.functional.cosine_similarity(
                        first.flatten().float(),
                        baseline.flatten().float(),
                        dim=0,
                    ).item()
                )
                repeat_max_abs_error = float(
                    (second.float() - first.float()).abs().max().item()
                )
                samples_us = []
                for _ in range(5):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(25):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples_us.append(float(start.elapsed_time(end)) * 40.0)
                median_us = float(statistics.median(samples_us))
                eligible = bool(
                    max_abs_error <= 0.02
                    and cosine >= 0.999
                    and repeat_max_abs_error == 0.0
                )
                row.update({
                    "median_us": median_us,
                    "samples_us": samples_us,
                    "max_abs_error": max_abs_error,
                    "cosine": cosine,
                    "repeat_max_abs_error": repeat_max_abs_error,
                    "eligible": eligible,
                })
                del graph
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            tune_cases.append(row)

        eligible_cases = [
            row
            for row in tune_cases
            if row["eligible"] and row["median_us"] is not None
        ]
        current_rows = [
            row
            for row in eligible_cases
            if row["case"] in ("current", "current_recheck")
        ]
        measured_winner = (
            min(eligible_cases, key=lambda row: float(row["median_us"]))
            if eligible_cases
            else None
        )
        current_reference_us = (
            min(float(row["median_us"]) for row in current_rows)
            if current_rows
            else None
        )
        current_stability_ratio = (
            max(float(row["median_us"]) for row in current_rows)
            / current_reference_us
            if len(current_rows) == 2 and current_reference_us
            else None
        )
        selected_name = "current"
        selected_path = "expert_grouped_compact"
        selected_config = dict(current_config)
        decision = "SKIP_KEEP_CURRENT"
        measured_speedup = 1.0
        if (
            current_reference_us is not None
            and measured_winner is not None
        ):
            measured_speedup = current_reference_us / float(
                measured_winner["median_us"]
            )
            if current_stability_ratio is None or current_stability_ratio > 1.03:
                decision = "SKIP_UNSTABLE_KEEP_CURRENT"
            elif (
                measured_winner["case"] != "current"
                and measured_winner["case"] != "current_recheck"
                and measured_speedup >= 1.02
            ):
                selected_name = str(measured_winner["case"])
                selected_path = str(measured_winner["path"])
                selected_config = dict(measured_winner["config"])
                decision = "APPLY"
            else:
                decision = "KEEP_CURRENT"
        apply_compact_config(selected_config)
        decode_kernel_tune = {
            "decision": decision,
            "minimum_promotion_speedup": 1.02,
            "maximum_current_stability_ratio": 1.03,
            "current_reference_us": current_reference_us,
            "current_stability_ratio": current_stability_ratio,
            "measured_winner": (
                None if measured_winner is None else measured_winner["case"]
            ),
            "measured_speedup_vs_current": measured_speedup,
            "selected": selected_name,
            "selected_path": selected_path,
            "selected_config": selected_config,
            "cases": tune_cases,
            "retired_candidates": [
                {
                    "case": "active_list_early_exit",
                    "run_id": "gemma4_moe_ab_20260727_031037",
                    "baseline_us": 626.8928146362305,
                    "candidate_us": 634.9107360839844,
                    "speedup": 0.9873715768342382,
                    "reason": (
                        "v84 was exact but 1.26% slower; active-list dispatch "
                        "is retired from paid runs"
                    ),
                },
                {
                    "case": "gate_s4_down_s3",
                    "run_id": "gemma4_moe_ab_20260727_020604",
                    "baseline_us": 632.1356964111328,
                    "candidate_us": 631.1526489257812,
                    "speedup": 1.001557543150654,
                    "reason": "v82 gained only 0.156%, below the 2% promotion floor",
                },
                {
                    "case": "expert_major_g1",
                    "run_id": "gemma4_moe_ab_20260724_205618",
                    "baseline_us": 632.3814392089844,
                    "candidate_us": 636.7846298217773,
                    "speedup": 0.9930852749790375,
                    "reason": "v77 measured slower than the stable compact baseline",
                },
                {
                    "case": "vllm_m16_e128_bf16",
                    "run_id": "gemma4_moe_ab_20260724_205618",
                    "baseline_us": 632.3814392089844,
                    "candidate_us": 660.8076477050781,
                    "speedup": 0.9569826278572664,
                    "reason": "v77 measured slower than the stable compact baseline",
                },
            ],
        }

        runtime_config = qwen3_moe_kernel.qwen3_moe_grouped_runtime_config()
        runtime_selected_config = {
            "active_list": bool(
                runtime_config["expert_grouped_compact_active_list"]
            ),
            "active_list_early_exit": bool(
                runtime_config[
                    "expert_grouped_compact_active_list_early_exit"
                ]
            ),
            "expert_grid_pack": bool(
                runtime_config["expert_grouped_compact_expert_grid_pack"]
            ),
            "coalesced_weights": bool(
                runtime_config["expert_grouped_compact_coalesced_weights"]
            ),
            "gate_block_n": int(
                runtime_config["expert_grouped_compact_gate_block_n"]
            ),
            "down_block_n": int(
                runtime_config["expert_grouped_compact_down_block_n"]
            ),
            "num_warps": int(
                runtime_config["expert_grouped_compact_num_warps"]
            ),
            "num_stages": int(
                runtime_config["expert_grouped_compact_num_stages"]
            ),
            "gate_num_stages": int(
                runtime_config["expert_grouped_compact_gate_num_stages"]
            ),
            "down_num_stages": int(
                runtime_config["expert_grouped_compact_down_num_stages"]
            ),
            "experts_per_program": int(
                runtime_config["expert_grouped_compact_experts_per_program"]
            ),
            "paired_gate_up_dot": bool(
                runtime_config["expert_grouped_compact_paired_gate_up_dot"]
            ),
            "split_gate_up": bool(
                runtime_config["expert_grouped_compact_split_gate_up"]
            ),
            "empty_expert_early_exit": bool(
                runtime_config["expert_grouped_compact_empty_expert_early_exit"]
            ),
            "l2_grouped_grid": bool(
                runtime_config["expert_grouped_compact_l2_grouped_grid"]
            ),
            "l2_group_size": int(
                runtime_config["expert_grouped_compact_l2_group_size"]
            ),
        }
        decode_kernel_tune["runtime_config_after_selection"] = (
            runtime_selected_config
        )
        if runtime_selected_config != selected_config:
            raise RuntimeError(
                "compact autotune config drifted immediately after selection: "
                f"selected={selected_config} runtime={runtime_selected_config}"
            )

        selected_model_layers = 0
        for model_layer in engine.model.layers:
            model_mlp = getattr(model_layer, "mlp", None)
            model_experts = getattr(model_mlp, "experts", None)
            if model_experts is None or not hasattr(
                model_experts,
                "_gemma4_batch_decode_use_compact",
            ):
                continue
            model_experts._gemma4_batch_decode_use_compact = bool(
                selected_path == "expert_grouped_compact"
            )
            selected_model_layers += 1
        decode_kernel_tune["selected_model_layers"] = selected_model_layers

        selected_workspace: dict[str, Any] = {}
        selected_call = (
            compact_call
            if selected_path == "expert_grouped_compact"
            else assignment_call
        )
        compact = selected_call(
            selected_workspace,
            torch.empty_like(hidden),
        ).clone()
        compact_repeat = selected_call(
            {},
            torch.empty_like(hidden),
        ).clone()
    sync_cuda()
    max_abs = float((compact.float() - baseline.float()).abs().max().item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            compact.flatten().float(), baseline.flatten().float(), dim=0
        ).item()
    )
    assignment_repeat_max_abs = float(
        (assignment_repeat.float() - baseline.float()).abs().max().item()
    )
    compact_repeat_max_abs = float(
        (compact_repeat.float() - compact.float()).abs().max().item()
    )
    assignment_reference_max_abs = float(
        (baseline.float() - reference.float()).abs().max().item()
    )
    assignment_reference_cosine = float(
        torch.nn.functional.cosine_similarity(
            baseline.flatten().float(), reference.flatten().float(), dim=0
        ).item()
    )
    router_decode_repeat_max_abs = float(
        (decode_weights_after.float() - decode_weights_before.float()).abs().max().item()
    )
    router_decode_experts_equal = bool(
        torch.equal(decode_experts_before, decode_experts_after)
    )
    result = {
        "rows": 16,
        "assignments": 128,
        "active_experts": active_experts,
        "singleton_experts": singleton_experts,
        "empty_experts": num_experts - active_experts,
        "path": selected_workspace.get("grouped_decode_last_path"),
        "max_abs_error": max_abs,
        "cosine": cosine,
        "assignment_repeat_max_abs_error": assignment_repeat_max_abs,
        "compact_repeat_max_abs_error": compact_repeat_max_abs,
        "assignment_reference_max_abs_error": assignment_reference_max_abs,
        "assignment_reference_cosine": assignment_reference_cosine,
        "router_decode_repeat_max_abs_error": router_decode_repeat_max_abs,
        "router_decode_experts_equal": router_decode_experts_equal,
        "decode_kernel_tune": decode_kernel_tune,
        "assignment_partial_reduce": int(
            assignment_workspace.get("grouped_decode_last_partial_reduce", 0) or 0
        ),
        "compact_partial_reduce": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_partial_reduce",
                0,
            )
            or 0
        ),
        "selected_partial_reduce": int(
            selected_workspace.get(
                (
                    "expert_grouped_compact_decode_last_partial_reduce"
                    if selected_path == "expert_grouped_compact"
                    else "grouped_decode_last_partial_reduce"
                ),
                0,
            )
            or 0
        ),
        "compact_active_list": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_active_list",
                0,
            )
            or 0
        ),
        "compact_active_list_early_exit": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_active_list_early_exit",
                0,
            )
            or 0
        ),
        "compact_expert_grid_pack": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_expert_grid_pack",
                0,
            )
            or 0
        ),
        "compact_coalesced_weights": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_coalesced_weights",
                0,
            )
            or 0
        ),
        "compact_gate_block_n": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_gate_block_n",
                0,
            )
            or 0
        ),
        "compact_down_block_n": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_down_block_n",
                0,
            )
            or 0
        ),
        "compact_num_warps": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_num_warps",
                0,
            )
            or 0
        ),
        "compact_num_stages": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_num_stages",
                0,
            )
            or 0
        ),
        "compact_gate_num_stages": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_gate_num_stages",
                0,
            )
            or 0
        ),
        "compact_down_num_stages": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_down_num_stages",
                0,
            )
            or 0
        ),
        "compact_experts_per_program": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_experts_per_program",
                0,
            )
            or 0
        ),
        "compact_paired_gate_up_dot": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_paired_gate_up_dot",
                0,
            )
            or 0
        ),
        "compact_split_gate_up": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_split_gate_up",
                0,
            )
            or 0
        ),
        "compact_empty_expert_early_exit": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_empty_expert_early_exit",
                0,
            )
            or 0
        ),
        "compact_l2_grouped_grid": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_l2_grouped_grid",
                0,
            )
            or 0
        ),
        "compact_l2_group_size": int(
            selected_workspace.get(
                "expert_grouped_compact_decode_last_l2_group_size",
                0,
            )
            or 0
        ),
        "router_workspace_isolated": bool(
            moe.gate._decode_topk_workspaces is not moe.gate._prefill_topk_workspaces
            and moe.gate._decode_router_logits_by_rows
            is not moe.gate._prefill_router_logits_by_rows
            and moe.gate._decode_topk_workspaces[8] is decode_workspace_before
            and moe.gate._decode_topk_workspaces[8]
            is not moe.gate._prefill_topk_workspaces[16]
            and moe.gate._decode_router_logits_by_rows[8] is decode_logits_before
            and int(moe.gate._decode_router_logits_by_rows[8].data_ptr())
            == decode_logits_ptr
            and 4 in moe.gate._decode_router_logits_by_rows
            and moe.gate._decode_router_logits_by_rows[4]
            is not moe.gate._decode_router_logits_by_rows[8]
        ),
    }
    if result["path"] != selected_path:
        raise RuntimeError(f"decode kernel gate selected the wrong path: {result}")
    if max_abs > 0.02 or cosine < 0.999:
        raise RuntimeError(f"selected decode kernel equivalence failed: {result}")
    if assignment_reference_max_abs > 0.05 or assignment_reference_cosine < 0.999:
        raise RuntimeError(f"assignment kernel equivalence failed: {result}")
    if assignment_repeat_max_abs != 0.0 or compact_repeat_max_abs != 0.0:
        raise RuntimeError(f"deterministic MoE reduction repeated differently: {result}")
    if not result["assignment_partial_reduce"] or not result["selected_partial_reduce"]:
        raise RuntimeError(f"deterministic MoE reduction was not selected: {result}")
    if selected_model_layers != 30:
        raise RuntimeError(f"selected decode path was not propagated to all layers: {result}")
    if selected_path == "expert_grouped_compact" and (
        bool(result["compact_active_list"])
        != bool(selected_config["active_list"])
        or bool(result["compact_active_list_early_exit"])
        != bool(selected_config["active_list_early_exit"])
        or bool(result["compact_expert_grid_pack"])
        != bool(selected_config["expert_grid_pack"])
        or bool(result["compact_coalesced_weights"])
        != bool(selected_config["coalesced_weights"])
        or result["compact_gate_block_n"] != int(selected_config["gate_block_n"])
        or result["compact_down_block_n"] != int(selected_config["down_block_n"])
        or result["compact_num_warps"] != int(selected_config["num_warps"])
        or result["compact_num_stages"] != int(selected_config["num_stages"])
        or result["compact_gate_num_stages"]
        != int(selected_config["gate_num_stages"])
        or result["compact_down_num_stages"]
        != int(selected_config["down_num_stages"])
        or result["compact_experts_per_program"]
        != int(selected_config["experts_per_program"])
        or bool(result["compact_paired_gate_up_dot"])
        != bool(selected_config["paired_gate_up_dot"])
        or bool(result["compact_split_gate_up"])
        != bool(selected_config["split_gate_up"])
        or bool(result["compact_empty_expert_early_exit"])
        != bool(selected_config["empty_expert_early_exit"])
        or bool(result["compact_l2_grouped_grid"])
        != bool(selected_config["l2_grouped_grid"])
        or result["compact_l2_group_size"]
        != int(selected_config["l2_group_size"])
    ):
        raise RuntimeError(f"selected compact decode config was not exercised: {result}")
    if not router_decode_experts_equal or router_decode_repeat_max_abs > 1e-6:
        raise RuntimeError(f"router decode changed across prefill: {result}")
    if not result["router_workspace_isolated"]:
        raise RuntimeError(f"router workspace isolation failed: {result}")

    micro_decision = str(decode_kernel_tune.get("decision") or "")
    full_gate: dict[str, Any] = {
        "decision": "SKIP_MICROBENCH_NO_WINNER",
        "selected": "current",
        "minimum_promotion_speedup": 1.01,
        "maximum_stability_ratio": 1.03,
        "cases": [],
    }
    if (
        micro_decision == "APPLY"
        and len(prompts) == 16
        and max_tokens >= 2
        and selected_path == "expert_grouped_compact"
    ):
        gate_tokens = min(int(max_tokens), 64)
        token_reference: list[list[int]] | None = None
        full_cases: list[dict[str, Any]] = []

        def configure_full_case(config: dict[str, Any]) -> None:
            apply_compact_config(config)
            for model_layer in engine.model.layers:
                model_experts = getattr(
                    getattr(model_layer, "mlp", None),
                    "experts",
                    None,
                )
                if model_experts is None:
                    continue
                model_experts._grouped_decode_disabled = False
                model_experts._grouped_decode_fail_reason = ""
                workspace = getattr(
                    model_experts,
                    "_grouped_decode_workspace",
                    None,
                )
                if isinstance(workspace, dict):
                    workspace.pop("expert_grouped_compact_decode_disabled", None)
                    workspace.pop("expert_grouped_compact_decode_fail_reason", None)

        full_specs = (
            ("current", current_config),
            (str(selected_name), selected_config),
            ("current_recheck", current_config),
        )
        for case_name, case_config in full_specs:
            row: dict[str, Any] = {
                "case": case_name,
                "config": dict(case_config),
                "decode_ms": [],
                "total_ms": [],
                "graph_checks": [],
                "token_checks": [],
                "tokens_exact": True,
                "eligible": False,
                "error": None,
            }
            try:
                configure_full_case(case_config)
                prime = run_megagemm_request(engine, prompts, gate_tokens)
                row["prime_graph"] = dict(prime.get("decode_cuda_graphs") or {})
                configure_full_case(case_config)
                for _ in range(2):
                    measured = run_megagemm_request(
                        engine,
                        prompts,
                        gate_tokens,
                    )
                    graph = dict(measured.get("decode_cuda_graphs") or {})
                    graph_check = {
                        "enabled": bool(graph.get("enabled")),
                        "failures": int(graph.get("failures", 0) or 0),
                        "replays": int(graph.get("replays", 0) or 0),
                        "shape_graphs": int(graph.get("shape_graphs", 0) or 0),
                    }
                    row["graph_checks"].append(graph_check)
                    if (
                        not graph_check["enabled"]
                        or graph_check["failures"] != 0
                        or graph_check["replays"] <= 0
                    ):
                        raise RuntimeError(
                            f"decode CUDA graph did not replay: {graph_check}"
                        )

                    decode_ms = float(
                        measured.get("scheduler_decode_ms") or 0.0
                    )
                    if decode_ms <= 0.0:
                        raise RuntimeError(
                            f"invalid scheduler decode timing: {decode_ms}ms"
                        )
                    if token_reference is None:
                        if case_name != "current":
                            raise RuntimeError(
                                "compact pipeline token oracle must come from current"
                            )
                        token_reference = measured["token_ids"]
                    comparison = token_matrix_comparison(
                        token_reference,
                        measured["token_ids"],
                    )
                    row["token_checks"].append(comparison)
                    row["tokens_exact"] = bool(
                        row["tokens_exact"] and comparison["exact"]
                    )
                    row["decode_ms"].append(decode_ms)
                    row["total_ms"].append(float(measured["total_ms"]))

                decode_values = [float(value) for value in row["decode_ms"]]
                row["median_decode_ms"] = float(
                    statistics.median(decode_values)
                )
                row["median_decode_tok_s"] = (
                    len(prompts)
                    * (gate_tokens - 1)
                    / (float(row["median_decode_ms"]) / 1000.0)
                )
                row["stability_ratio"] = max(decode_values) / min(
                    decode_values
                )
                row["eligible"] = bool(
                    row["tokens_exact"]
                    and row["stability_ratio"] <= 1.03
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            full_cases.append(row)

        eligible_full = [
            row
            for row in full_cases
            if row["eligible"] and row.get("median_decode_ms") is not None
        ]
        full_current = [
            row
            for row in eligible_full
            if row["case"] in {"current", "current_recheck"}
        ]
        full_candidate = next(
            (
                row
                for row in eligible_full
                if row["case"] == selected_name
            ),
            None,
        )
        full_current_ms = (
            min(float(row["median_decode_ms"]) for row in full_current)
            if full_current
            else None
        )
        full_current_stability = (
            max(float(row["median_decode_ms"]) for row in full_current)
            / full_current_ms
            if len(full_current) == 2 and full_current_ms
            else None
        )
        full_speedup = (
            full_current_ms / float(full_candidate["median_decode_ms"])
            if full_current_ms is not None and full_candidate is not None
            else None
        )
        full_apply = bool(
            full_current_stability is not None
            and full_current_stability <= 1.03
            and full_speedup is not None
            and full_speedup >= 1.01
        )
        final_config = selected_config if full_apply else current_config
        configure_full_case(final_config)
        full_gate = {
            "decision": "APPLY" if full_apply else "KEEP_CURRENT",
            "selected": selected_name if full_apply else "current",
            "gate_tokens_per_row": gate_tokens,
            "minimum_promotion_speedup": 1.01,
            "maximum_stability_ratio": 1.03,
            "current_reference_decode_ms": full_current_ms,
            "current_stability_ratio": full_current_stability,
            "measured_speedup_vs_current": full_speedup,
            "estimated_savings_ms_for_gate_request": (
                float(
                    full_current_ms
                    - float(full_candidate["median_decode_ms"])
                )
                if full_apply
                and full_current_ms is not None
                and full_candidate is not None
                else 0.0
            ),
            "cases": full_cases,
        }
        decode_kernel_tune["microbenchmark_decision"] = micro_decision
        decode_kernel_tune["microbenchmark_selected"] = selected_name
        decode_kernel_tune["decision"] = full_gate["decision"]
        decode_kernel_tune["selected"] = full_gate["selected"]
        decode_kernel_tune["selected_config"] = dict(final_config)
        decode_kernel_tune["runtime_config_after_selection"] = dict(
            final_config
        )
        result["compact_gate_num_stages"] = int(
            final_config["gate_num_stages"]
        )
        result["compact_down_num_stages"] = int(
            final_config["down_num_stages"]
        )
        result["compact_active_list"] = int(final_config["active_list"])
        result["compact_active_list_early_exit"] = int(
            final_config["active_list_early_exit"]
        )
    else:
        apply_compact_config(current_config)
        decode_kernel_tune["decision"] = "KEEP_CURRENT"
        decode_kernel_tune["selected"] = "current"
        decode_kernel_tune["selected_config"] = dict(current_config)
        decode_kernel_tune["runtime_config_after_selection"] = dict(
            current_config
        )
    decode_kernel_tune["full_request_gate"] = full_gate
    return result


def run_segmented_prefill_kernel_gate(
    engine,
    batch_sizes: list[int],
) -> dict[str, Any]:
    import megagemm.kernels.qwen3_moe as moe_kernel

    layer = engine.model.layers[0]
    moe = layer.mlp
    experts = moe.experts
    gate_batches = sorted({4} | {int(batch) for batch in batch_sizes})
    cases: dict[str, Any] = {}
    selected_by_rows: dict[str, Any] = {}
    shape_tuning_applied = False
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260716)
    tunable_keys = (
        "block_m",
        "block_n",
        "block_k",
        "fused_gate_block_n",
        "num_warps",
        "num_stages",
        "single_accumulator",
        "group_size_m",
    )

    for batch_size in gate_batches:
        rows = int(batch_size) * 25
        hidden = torch.randn(
            (rows, experts.hidden_dim),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(0.02)
        with torch.inference_mode():
            routing_weights, selected_experts = moe.gate.route(hidden)
            routing_weights = routing_weights.clone()
            selected_experts = selected_experts.clone()
            base_options = experts._segmented_prefill_kernel_options(rows)
            reference_options = dict(base_options)
            reference_options["compact_route_pack"] = False
            reference = moe_kernel.qwen3_moe_segmented_prefill(
                hidden,
                experts.gate_up_proj,
                experts.down_proj,
                selected_experts,
                routing_weights,
                activation=experts.hidden_act,
                workspace={},
                **reference_options,
            ).clone()
        sync_cuda()

        # v76 measured the one-accumulator candidate at 1617.9 us/layer
        # versus 1359.9 us/layer for this baseline. Keep it out of paid runs.
        candidate_changes: list[tuple[str, dict[str, int]]] = [("current", {})]

        tuning_rows: list[dict[str, Any]] = []

        def run_case(
            case_name: str,
            changes: dict[str, int],
        ) -> dict[str, Any]:
            options = dict(base_options)
            options.update(changes)
            workspace: dict[str, Any] = {}

            def invoke() -> torch.Tensor:
                with torch.inference_mode():
                    return moe_kernel.qwen3_moe_segmented_prefill(
                        hidden,
                        experts.gate_up_proj,
                        experts.down_proj,
                        selected_experts,
                        routing_weights,
                        activation=experts.hidden_act,
                        workspace=workspace,
                        **options,
                    )

            try:
                invoke()
                invoke()
                sync_cuda()
            except Exception as exc:
                row = {
                    "case": case_name,
                    "path": "segmented",
                    "eligible": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "config": {
                        key: int(options[key])
                        for key in tunable_keys
                        if key in options
                    },
                    "median_us": None,
                    "samples_us": [],
                }
                tuning_rows.append(row)
                return row

            samples_us: list[float] = []
            output = None
            for _ in range(5):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = invoke()
                end.record()
                end.synchronize()
                samples_us.append(float(start.elapsed_time(end) * 1000.0))
            assert output is not None
            first = output.clone()
            second = invoke().clone()
            sync_cuda()

            max_abs = float((second.float() - first.float()).abs().max().item())
            reference_max_abs = float(
                (first.float() - reference.float()).abs().max().item()
            )
            reference_cosine = float(
                torch.nn.functional.cosine_similarity(
                    first.float().reshape(-1),
                    reference.float().reshape(-1),
                    dim=0,
                ).item()
            )
            partial_reduce = int(
                workspace.get("segmented_prefill_partial_reduce", 0) or 0
            )
            compact_route_pack = int(
                workspace.get("segmented_prefill_compact_route_pack", 0) or 0
            )
            compact_route_pack_passes = int(
                workspace.get("segmented_prefill_compact_route_pack_passes", 0)
                or 0
            )
            partial_cached = int(
                workspace.get("segmented_prefill_partial_cached", 0) or 0
            )
            persistent_partial_buffer = bool(
                workspace.get("segmented_partial_out") is not None
            )
            partial_dtype = str(
                workspace.get("segmented_prefill_partial_dtype", "") or ""
            )
            single_accumulator = int(
                workspace.get(
                    "segmented_prefill_single_accumulator",
                    0,
                )
                or 0
            )
            group_size_m = int(
                workspace.get("segmented_prefill_group_size_m", 0) or 0
            )
            eligible = bool(
                torch.equal(first, second)
                and max_abs == 0.0
                and reference_cosine >= 0.9999
                and reference_max_abs <= 2e-3
                and partial_reduce
                and compact_route_pack
                and compact_route_pack_passes == 1
                and not int(
                    workspace.get("segmented_prefill_async_tiles", 0) or 0
                )
                and not int(
                    workspace.get("segmented_prefill_route_scatter", 0) or 0
                )
                and partial_dtype == "torch.float32"
                and single_accumulator
                == int(bool(options.get("single_accumulator", False)))
                and group_size_m == int(options.get("group_size_m", 8))
                and not (
                    rows * int(selected_experts.shape[1]) > 512
                    and (partial_cached or persistent_partial_buffer)
                )
            )
            row = {
                "case": case_name,
                "path": "segmented",
                "eligible": eligible,
                "error": None,
                "config": {
                    key: int(options[key])
                    for key in tunable_keys
                    if key in options
                },
                "median_us": float(statistics.median(samples_us)),
                "samples_us": samples_us,
                "repeat_max_abs_error": max_abs,
                "repeat_exact": bool(torch.equal(first, second)),
                "reference_max_abs_error": reference_max_abs,
                "reference_cosine": reference_cosine,
                "partial_reduce": partial_reduce,
                "partial_cached": partial_cached,
                "partial_bytes": int(
                    workspace.get("segmented_prefill_partial_bytes", 0) or 0
                ),
                "partial_dtype": partial_dtype,
                "single_accumulator": single_accumulator,
                "group_size_m": group_size_m,
                "persistent_partial_buffer": persistent_partial_buffer,
                "compact_route_pack": compact_route_pack,
                "compact_route_pack_passes": compact_route_pack_passes,
                "selected_tiles": int(
                    workspace.get("segmented_prefill_last_tiles", 0) or 0
                ),
            }
            tuning_rows.append(row)
            return row

        def run_grouped_mm_case() -> dict[str, Any]:
            case_name = "torch_grouped_mm_prefill"
            workspace: dict[str, Any] = {}
            baseline_config = {
                key: int(base_options[key])
                for key in tunable_keys
                if key in base_options
            }

            def invoke() -> torch.Tensor:
                with torch.inference_mode():
                    return grouped_prefill_kernel.gemma4_grouped_mm_prefill(
                        hidden,
                        experts.gate_up_proj,
                        experts.down_proj,
                        selected_experts,
                        routing_weights,
                        workspace=workspace,
                    )

            try:
                invoke()
                invoke()
                sync_cuda()
            except Exception as exc:
                row = {
                    "case": case_name,
                    "path": "grouped_mm",
                    "eligible": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "config": baseline_config,
                    "median_us": None,
                    "samples_us": [],
                }
                tuning_rows.append(row)
                return row

            samples_us: list[float] = []
            output = None
            for _ in range(5):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = invoke()
                end.record()
                end.synchronize()
                samples_us.append(float(start.elapsed_time(end) * 1000.0))
            assert output is not None
            first = output.clone()
            second = invoke().clone()
            sync_cuda()

            alias_error = None
            alias_repeat_exact = False
            alias_repeat_max_abs = float("inf")
            alias_reference_max_abs = float("inf")
            alias_reference_cosine = 0.0
            alias_returns_out = False
            try:
                residual_seed = torch.randn(
                    hidden.shape,
                    device=hidden.device,
                    dtype=hidden.dtype,
                    generator=generator,
                ).mul_(0.02)
                reference_alias = residual_seed.clone()
                with torch.inference_mode():
                    reference_alias_out = moe_kernel.qwen3_moe_segmented_prefill(
                        hidden,
                        experts.gate_up_proj,
                        experts.down_proj,
                        selected_experts,
                        routing_weights,
                        activation=experts.hidden_act,
                        out=reference_alias,
                        residual=reference_alias,
                        workspace={},
                        **reference_options,
                    ).clone()

                first_alias_buffer = residual_seed.clone()
                first_alias_workspace: dict[str, Any] = {}
                with torch.inference_mode():
                    first_alias_result = (
                        grouped_prefill_kernel.gemma4_grouped_mm_prefill(
                            hidden,
                            experts.gate_up_proj,
                            experts.down_proj,
                            selected_experts,
                            routing_weights,
                            out=first_alias_buffer,
                            residual=first_alias_buffer,
                            workspace=first_alias_workspace,
                        )
                    )
                alias_returns_out = (
                    first_alias_result.data_ptr()
                    == first_alias_buffer.data_ptr()
                )
                first_alias = first_alias_result.clone()

                second_alias_buffer = residual_seed.clone()
                with torch.inference_mode():
                    second_alias = (
                        grouped_prefill_kernel.gemma4_grouped_mm_prefill(
                            hidden,
                            experts.gate_up_proj,
                            experts.down_proj,
                            selected_experts,
                            routing_weights,
                            out=second_alias_buffer,
                            residual=second_alias_buffer,
                            workspace={},
                        ).clone()
                    )
                sync_cuda()
                alias_repeat_exact = bool(
                    torch.equal(first_alias, second_alias)
                )
                alias_repeat_max_abs = float(
                    (first_alias.float() - second_alias.float())
                    .abs()
                    .max()
                    .item()
                )
                alias_reference_max_abs = float(
                    (first_alias.float() - reference_alias_out.float())
                    .abs()
                    .max()
                    .item()
                )
                alias_reference_cosine = float(
                    torch.nn.functional.cosine_similarity(
                        first_alias.float().reshape(-1),
                        reference_alias_out.float().reshape(-1),
                        dim=0,
                    ).item()
                )
            except Exception as exc:
                alias_error = f"{type(exc).__name__}: {exc}"
                try:
                    sync_cuda()
                except Exception:
                    pass

            repeat_max_abs = float(
                (second.float() - first.float()).abs().max().item()
            )
            reference_max_abs = float(
                (first.float() - reference.float()).abs().max().item()
            )
            reference_cosine = float(
                torch.nn.functional.cosine_similarity(
                    first.float().reshape(-1),
                    reference.float().reshape(-1),
                    dim=0,
                ).item()
            )
            offsets = workspace.get("grouped_mm_offsets")
            final_offset = (
                int(offsets[-1].item())
                if isinstance(offsets, torch.Tensor) and offsets.numel()
                else -1
            )
            route_pack = int(
                workspace.get("grouped_mm_prefill_route_pack", 0) or 0
            )
            deterministic_reduce = int(
                workspace.get(
                    "grouped_mm_prefill_deterministic_reduce",
                    0,
                )
                or 0
            )
            active = int(
                workspace.get("grouped_mm_prefill_active", 0) or 0
            )
            eligible = bool(
                torch.equal(first, second)
                and repeat_max_abs == 0.0
                and reference_cosine >= 0.9999
                and reference_max_abs <= 2e-3
                and active
                and route_pack
                and deterministic_reduce
                and final_offset == rows * int(selected_experts.shape[1])
                and alias_error is None
                and alias_repeat_exact
                and alias_repeat_max_abs == 0.0
                and alias_reference_cosine >= 0.9999
                and alias_reference_max_abs <= 2e-3
                and alias_returns_out
            )
            row = {
                "case": case_name,
                "path": "grouped_mm",
                "eligible": eligible,
                "error": None,
                "config": baseline_config,
                "median_us": float(statistics.median(samples_us)),
                "samples_us": samples_us,
                "repeat_exact": bool(torch.equal(first, second)),
                "repeat_max_abs_error": repeat_max_abs,
                "reference_max_abs_error": reference_max_abs,
                "reference_cosine": reference_cosine,
                "route_pack": route_pack,
                "deterministic_reduce": deterministic_reduce,
                "final_offset": final_offset,
                "assignments": int(rows * selected_experts.shape[1]),
                "alias_error": alias_error,
                "alias_repeat_exact": alias_repeat_exact,
                "alias_repeat_max_abs_error": alias_repeat_max_abs,
                "alias_reference_max_abs_error": alias_reference_max_abs,
                "alias_reference_cosine": alias_reference_cosine,
                "alias_returns_out": alias_returns_out,
            }
            tuning_rows.append(row)
            return row

        baseline = run_case("current", {})
        candidates = [
            run_case(name, changes)
            for name, changes in candidate_changes
            if name != "current"
        ]
        baseline_recheck = run_case("current_recheck", {})
        if (
            not baseline.get("repeat_exact", False)
            or float(baseline.get("repeat_max_abs_error", float("inf"))) != 0.0
            or not baseline_recheck.get("repeat_exact", False)
            or float(
                baseline_recheck.get("repeat_max_abs_error", float("inf"))
            )
            != 0.0
        ):
            raise RuntimeError(
                f"segmented prefill repeated differently: {tuning_rows}"
            )
        if not baseline["eligible"] or not baseline_recheck["eligible"]:
            raise RuntimeError(
                f"segmented prefill baseline contract failed: {tuning_rows}"
            )
        baseline_first_us = float(baseline["median_us"])
        baseline_recheck_us = float(baseline_recheck["median_us"])
        baseline_us = min(baseline_first_us, baseline_recheck_us)
        baseline_stability_ratio = (
            max(baseline_first_us, baseline_recheck_us) / baseline_us
            if baseline_us > 0.0
            else float("inf")
        )
        eligible_candidates = [row for row in candidates if row["eligible"]]
        measured_winner = min(
            [baseline_recheck, *eligible_candidates],
            key=lambda row: float(row["median_us"]),
        )
        measured_winner_us = float(measured_winner["median_us"])
        measured_speedup = baseline_us / measured_winner_us
        beats_both_baselines = bool(
            measured_winner["case"] != "current_recheck"
            and baseline_first_us / measured_winner_us >= 1.02
            and baseline_recheck_us / measured_winner_us >= 1.02
        )
        apply_tuning = bool(
            measured_winner["case"] != "current_recheck"
            and beats_both_baselines
        )
        selected = measured_winner if apply_tuning else baseline_recheck
        selected_config = dict(selected["config"])
        selected_layers = 0
        for model_layer in engine.model.layers:
            layer_experts = getattr(
                getattr(model_layer, "mlp", None),
                "experts",
                None,
            )
            setter = getattr(
                layer_experts,
                "set_segmented_prefill_runtime_options",
                None,
            )
            if callable(setter):
                setter(rows, selected_config)
                selected_layers += 1
        if selected_layers != len(engine.model.layers):
            raise RuntimeError(
                "segmented prefill runtime config did not reach every model layer: "
                f"{selected_layers}/{len(engine.model.layers)}"
            )
        shape_tuning_applied = shape_tuning_applied or apply_tuning
        if apply_tuning and bool(
            selected_config.get("single_accumulator", False)
        ):
            decision = "APPLY_SINGLE_ACCUMULATOR"
        elif apply_tuning:
            decision = "APPLY_SEGMENTED_CONFIG"
        else:
            decision = "KEEP_CURRENT_SEGMENTED"
        selected_by_rows[str(rows)] = {
            "decision": decision,
            "selected": selected["case"],
            "selected_path": str(selected.get("path") or "segmented"),
            "selected_config": selected_config,
            "selected_layers": selected_layers,
            "baseline_us": baseline_us,
            "baseline_stability_ratio": baseline_stability_ratio,
            "baseline_stability_is_diagnostic": True,
            "measured_winner": measured_winner["case"],
            "measured_speedup": measured_speedup,
            "beats_both_baselines": beats_both_baselines,
            "minimum_promotion_speedup": 1.02,
        }
        cases[str(batch_size)] = {
            "batch_size": int(batch_size),
            "rows": int(rows),
            "assignments": int(rows * selected_experts.shape[1]),
            "decision": decision,
            "selected": selected["case"],
            "selected_path": str(selected.get("path") or "segmented"),
            "selected_config": selected_config,
            "selected_layers": selected_layers,
            "baseline_us": baseline_us,
            "baseline_stability_ratio": baseline_stability_ratio,
            "baseline_stability_is_diagnostic": True,
            "measured_speedup": measured_speedup,
            "measured_winner": measured_winner["case"],
            "beats_both_baselines": beats_both_baselines,
            "minimum_promotion_speedup": 1.02,
            "tuning": tuning_rows,
        }

    return {
        "decision": "KEEP_FP32_PARTIAL",
        "selected": "fp32_partial",
        "cases": cases,
        "gate_batches": gate_batches,
        "shape_tuning_applied": shape_tuning_applied,
        "retired_candidates": {
            "single_accumulator_l2_g8": {
                "retired_after": "v76",
                "baseline_us_per_layer": 1359.8719835281372,
                "candidate_us_per_layer": 1617.9200410842896,
                "reason": "19 percent slower on A100 B16 prefill",
            },
        },
        "single_accumulator_applied": any(
            selection.get("decision") == "APPLY_SINGLE_ACCUMULATOR"
            for selection in selected_by_rows.values()
        ),
        "selected_by_rows": selected_by_rows,
    }


def run_gemma4_prefill_attn_moe_bridge_gate(
    engine,
    batch_sizes: list[int],
) -> dict[str, Any]:
    """Select the exact two-kernel attention-to-MoE bridge before paid timing."""
    batch_size = max(int(batch) for batch in batch_sizes)
    rows = batch_size * 25
    if not env_flag("MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_PREFILL"):
        return {
            "decision": "DISABLED_BY_ENV",
            "selected": "sequential",
            "batch_size": batch_size,
            "rows": rows,
            "selected_layers": 0,
            "cases": [],
        }

    from megagemm.kernels.rmsnorm_triton import (
        rmsnorm_triton_attn_residual_router_bridge,
        rmsnorm_triton_dual,
        rmsnorm_triton_scaled_no_weight,
    )

    if batch_size != 16 or rows != 400:
        return {
            "decision": "NOT_ELIGIBLE",
            "selected": "sequential",
            "batch_size": batch_size,
            "rows": rows,
            "selected_layers": 0,
            "cases": [],
        }

    layer = engine.model.layers[0]
    hidden_dim = int(layer.mlp.hidden_dim)
    setter_name = "set_gemma4_prefill_attn_moe_bridge_runtime"
    if not callable(getattr(layer, setter_name, None)):
        raise RuntimeError(
            "Gemma4 prefill attention-to-MoE bridge selector is unavailable"
        )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260723)
    attn_out = torch.randn(
        (rows, hidden_dim),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    residual = torch.randn(
        (rows, hidden_dim),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    baseline_hidden = torch.empty_like(residual)
    candidate_hidden = torch.empty_like(residual)
    candidate_post_norm = torch.empty_like(residual)
    candidate_shared = torch.empty_like(residual)
    candidate_expert = torch.empty_like(residual)
    candidate_router = torch.empty_like(residual)
    router = layer.mlp.gate
    router_scale = router.scale.to(
        device=attn_out.device,
        dtype=attn_out.dtype,
    ).reshape(-1)

    def run_sequential() -> tuple[torch.Tensor, ...]:
        with torch.inference_mode():
            post_attn = layer.post_attention_layernorm(attn_out)
            torch.add(residual, post_attn, out=baseline_hidden)
            router_in = rmsnorm_triton_scaled_no_weight(
                baseline_hidden,
                router_scale,
                router.input_norm.eps,
                router.scalar_root,
            )
            shared_in, expert_in = rmsnorm_triton_dual(
                baseline_hidden,
                layer.pre_feedforward_layernorm.weight,
                layer.pre_feedforward_layernorm_2.weight,
                layer.pre_feedforward_layernorm.eps,
            )
        return baseline_hidden, shared_in, expert_in, router_in

    def run_exact_bridge() -> tuple[torch.Tensor, ...]:
        with torch.inference_mode():
            return rmsnorm_triton_attn_residual_router_bridge(
                attn_out,
                residual,
                layer.post_attention_layernorm.weight,
                layer.pre_feedforward_layernorm.weight,
                layer.pre_feedforward_layernorm_2.weight,
                router_scale,
                layer.post_attention_layernorm.eps,
                router.scalar_root,
                out_hidden=candidate_hidden,
                post_norm_out=candidate_post_norm,
                shared_out=candidate_shared,
                expert_out=candidate_expert,
                router_out=candidate_router,
            )

    def clone_outputs(
        outputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        return tuple(output.clone() for output in outputs)

    def compare_outputs(
        actual: tuple[torch.Tensor, ...],
        expected: tuple[torch.Tensor, ...],
    ) -> tuple[bool, float]:
        exact = all(
            torch.equal(actual_tensor, expected_tensor)
            for actual_tensor, expected_tensor in zip(actual, expected)
        )
        max_abs = max(
            float(
                (actual_tensor.float() - expected_tensor.float())
                .abs()
                .max()
                .item()
            )
            for actual_tensor, expected_tensor in zip(actual, expected)
        )
        return bool(exact), float(max_abs)

    def output_diagnostics(
        actual: tuple[torch.Tensor, ...],
        expected: tuple[torch.Tensor, ...],
    ) -> dict[str, dict[str, Any]]:
        diagnostics: dict[str, dict[str, Any]] = {}
        for name, actual_tensor, expected_tensor in zip(
            ("hidden", "shared", "expert", "router"),
            actual,
            expected,
        ):
            diagnostics[name] = {
                "exact": bool(torch.equal(actual_tensor, expected_tensor)),
                "max_abs_error": float(
                    (actual_tensor.float() - expected_tensor.float())
                    .abs()
                    .max()
                    .item()
                ),
            }
        return diagnostics

    def measure(
        name: str,
        invoke,
        reference: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[
        dict[str, Any],
        tuple[torch.Tensor, ...] | None,
    ]:
        iterations_per_sample = 8
        try:
            invoke()
            invoke()
            sync_cuda()
            samples_us: list[float] = []
            output = None
            for _ in range(5):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations_per_sample):
                    output = invoke()
                end.record()
                end.synchronize()
                samples_us.append(
                    float(
                        start.elapsed_time(end)
                        * 1000.0
                        / iterations_per_sample
                    )
                )
            assert output is not None
            first = clone_outputs(output)
            second = clone_outputs(invoke())
            sync_cuda()
            repeat_exact, repeat_max_abs = compare_outputs(first, second)
            reference_exact, reference_max_abs = (
                (True, 0.0)
                if reference is None
                else compare_outputs(first, reference)
            )
            row = {
                "case": name,
                "eligible": bool(
                    repeat_exact
                    and repeat_max_abs == 0.0
                    and reference_exact
                    and reference_max_abs == 0.0
                ),
                "error": None,
                "median_us": float(statistics.median(samples_us)),
                "samples_us": samples_us,
                "iterations_per_sample": iterations_per_sample,
                "repeat_exact": repeat_exact,
                "repeat_max_abs_error": repeat_max_abs,
                "reference_exact": reference_exact,
                "reference_max_abs_error": reference_max_abs,
            }
            return row, first
        except Exception as exc:
            try:
                sync_cuda()
            except Exception:
                pass
            return (
                {
                    "case": name,
                    "eligible": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "median_us": None,
                    "samples_us": [],
                    "iterations_per_sample": iterations_per_sample,
                    "repeat_exact": False,
                    "repeat_max_abs_error": None,
                    "reference_exact": False,
                    "reference_max_abs_error": None,
                },
                None,
            )

    baseline, reference = measure("sequential", run_sequential)
    if not baseline["eligible"] or reference is None:
        raise RuntimeError(
            f"Gemma4 prefill attention-to-MoE baseline failed: {baseline}"
        )
    baseline["kernel_count"] = 4
    candidate, candidate_output = measure(
        "exact_two_kernel_attn_moe_router_bridge",
        run_exact_bridge,
        reference,
    )
    candidate["kernel_count"] = 2
    candidate["reference_outputs"] = (
        output_diagnostics(candidate_output, reference)
        if candidate_output is not None
        else {}
    )
    baseline_recheck, _ = measure("sequential_recheck", run_sequential, reference)
    baseline_recheck["kernel_count"] = 4
    if not baseline_recheck["eligible"]:
        raise RuntimeError(
            "Gemma4 prefill attention-to-MoE baseline recheck failed: "
            f"{baseline_recheck}"
        )

    alias_exact = False
    alias_max_abs = None
    alias_error = None
    alias_output_diagnostics: dict[str, dict[str, Any]] = {}
    try:
        alias_residual = residual.clone()
        with torch.inference_mode():
            alias_outputs = rmsnorm_triton_attn_residual_router_bridge(
                attn_out,
                alias_residual,
                layer.post_attention_layernorm.weight,
                layer.pre_feedforward_layernorm.weight,
                layer.pre_feedforward_layernorm_2.weight,
                router_scale,
                layer.post_attention_layernorm.eps,
                router.scalar_root,
                out_hidden=alias_residual,
            )
        alias_outputs = clone_outputs(alias_outputs)
        sync_cuda()
        alias_exact, alias_max_abs = compare_outputs(alias_outputs, reference)
        alias_output_diagnostics = output_diagnostics(alias_outputs, reference)
    except Exception as exc:
        alias_error = f"{type(exc).__name__}: {exc}"
        try:
            sync_cuda()
        except Exception:
            pass
    candidate["alias_exact"] = alias_exact
    candidate["alias_max_abs_error"] = alias_max_abs
    candidate["alias_error"] = alias_error
    candidate["alias_outputs"] = alias_output_diagnostics
    candidate["eligible"] = bool(
        candidate["eligible"] and alias_exact and alias_max_abs == 0.0
    )

    baseline_first_us = float(baseline["median_us"])
    baseline_recheck_us = float(baseline_recheck["median_us"])
    conservative_baseline_us = min(baseline_first_us, baseline_recheck_us)
    candidate_us = (
        float(candidate["median_us"])
        if candidate["eligible"] and candidate["median_us"] is not None
        else float("inf")
    )
    speedup = (
        conservative_baseline_us / candidate_us
        if candidate_us < float("inf")
        else 0.0
    )
    beats_both_baselines = bool(
        candidate["eligible"]
        and baseline_first_us / candidate_us >= 1.02
        and baseline_recheck_us / candidate_us >= 1.02
    )
    apply_candidate = bool(beats_both_baselines and speedup >= 1.02)
    selected_layers = 0
    for model_layer in engine.model.layers:
        setter = getattr(model_layer, setter_name, None)
        if callable(setter):
            setter(rows, apply_candidate)
            selected_layers += int(apply_candidate)
    expected_layers = len(engine.model.layers) if apply_candidate else 0
    if selected_layers != expected_layers:
        raise RuntimeError(
            "Gemma4 prefill attention-to-MoE bridge selection did not reach "
            "every layer: "
            f"{selected_layers}/{expected_layers}"
        )

    return {
        "decision": (
            "APPLY_FUSED_ATTN_MOE_BRIDGE"
            if apply_candidate
            else "KEEP_SEQUENTIAL_ATTN_MOE_BRIDGE"
        ),
        "selected": (
            "exact_two_kernel_attn_moe_router_bridge"
            if apply_candidate
            else "sequential"
        ),
        "batch_size": batch_size,
        "rows": rows,
        "selected_layers": selected_layers,
        "baseline_us": conservative_baseline_us,
        "baseline_stability_ratio": (
            max(baseline_first_us, baseline_recheck_us)
            / conservative_baseline_us
        ),
        "baseline_stability_is_diagnostic": True,
        "candidate_us": None if candidate_us == float("inf") else candidate_us,
        "speedup": speedup,
        "beats_both_baselines": beats_both_baselines,
        "minimum_promotion_speedup": 1.02,
        "router_output_fused": True,
        "estimated_savings_ms_per_prefill": (
            (conservative_baseline_us - candidate_us) * 30.0 / 1000.0
            if apply_candidate
            else 0.0
        ),
        "cases": [baseline, candidate, baseline_recheck],
    }


def run_gemma4_prefill_router_kernel_gate(
    engine,
    batch_sizes: list[int],
) -> dict[str, Any]:
    """Select the 400-row matrix router only when it beats cuBLAS exactly."""
    batch_size = max(int(batch) for batch in batch_sizes)
    rows = batch_size * 25
    if not env_flag("MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_PREFILL"):
        return {
            "decision": "DISABLED_BY_ENV",
            "selected": "cublas_plus_topk",
            "batch_size": batch_size,
            "rows": rows,
            "selected_layers": 0,
            "cases": [],
        }
    if batch_size != 16 or rows != 400:
        return {
            "decision": "NOT_ELIGIBLE",
            "selected": "cublas_plus_topk",
            "batch_size": batch_size,
            "rows": rows,
            "selected_layers": 0,
            "cases": [],
        }

    from megagemm.kernels.gemma4_moe_router import (
        gemma4_moe_prefill_router_topk,
    )
    from megagemm.kernels.qwen3_moe import qwen3_moe_topk_softmax

    router = engine.model.layers[0].mlp.gate
    setter_name = "set_fused_prefill_runtime"
    if not callable(getattr(router, setter_name, None)):
        raise RuntimeError("Gemma4 fused prefill router selector is unavailable")

    weight = getattr(router.proj, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError("Gemma4 BF16 router weight is unavailable")
    weight = weight.contiguous()
    expert_scale = router.per_expert_scale.to(
        device=weight.device,
        dtype=weight.dtype,
    ).reshape(-1).contiguous()
    hidden_dim = int(weight.shape[1])
    top_k = int(router.top_k)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260803)
    normalized = torch.randn(
        (rows, hidden_dim),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(hidden_dim ** -0.5)
    logits = torch.empty(
        (rows, int(weight.shape[0])),
        device=normalized.device,
        dtype=normalized.dtype,
    )
    baseline_workspace: dict[str, torch.Tensor] = {}
    candidate_workspace: dict[str, torch.Tensor] = {}

    def run_cublas() -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            torch.mm(normalized, weight.t(), out=logits)
            return qwen3_moe_topk_softmax(
                logits,
                top_k,
                workspace=baseline_workspace,
                expert_scale=expert_scale,
            )

    def run_fused() -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            return gemma4_moe_prefill_router_topk(
                normalized,
                weight,
                expert_scale,
                top_k,
                workspace=candidate_workspace,
            )

    def clone_outputs(
        outputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return outputs[0].clone(), outputs[1].clone()

    def compare_outputs(
        actual: tuple[torch.Tensor, torch.Tensor],
        expected: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[bool, float, bool]:
        weights_exact = bool(torch.equal(actual[0], expected[0]))
        experts_exact = bool(torch.equal(actual[1], expected[1]))
        max_weight_error = float(
            (actual[0].float() - expected[0].float()).abs().max().item()
        )
        return weights_exact, max_weight_error, experts_exact

    def measure(
        name: str,
        invoke,
        reference: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[dict[str, Any], tuple[torch.Tensor, torch.Tensor] | None]:
        iterations_per_sample = 16
        try:
            invoke()
            invoke()
            sync_cuda()
            samples_us: list[float] = []
            output = None
            for _ in range(5):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations_per_sample):
                    output = invoke()
                end.record()
                end.synchronize()
                samples_us.append(
                    float(
                        start.elapsed_time(end)
                        * 1000.0
                        / iterations_per_sample
                    )
                )
            assert output is not None
            first = clone_outputs(output)
            second = clone_outputs(invoke())
            sync_cuda()
            repeat_weights, repeat_error, repeat_experts = compare_outputs(
                first,
                second,
            )
            if reference is None:
                reference_weights, reference_error, reference_experts = (
                    True,
                    0.0,
                    True,
                )
            else:
                reference_weights, reference_error, reference_experts = (
                    compare_outputs(first, reference)
                )
            row = {
                "case": name,
                "eligible": bool(
                    repeat_weights
                    and repeat_experts
                    and repeat_error == 0.0
                    and reference_weights
                    and reference_experts
                    and reference_error == 0.0
                ),
                "error": None,
                "median_us": float(statistics.median(samples_us)),
                "samples_us": samples_us,
                "iterations_per_sample": iterations_per_sample,
                "repeat_weights_exact": repeat_weights,
                "repeat_experts_exact": repeat_experts,
                "repeat_max_weight_error": repeat_error,
                "reference_weights_exact": reference_weights,
                "reference_experts_exact": reference_experts,
                "reference_max_weight_error": reference_error,
            }
            return row, first
        except Exception as exc:
            try:
                sync_cuda()
            except Exception:
                pass
            return (
                {
                    "case": name,
                    "eligible": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "median_us": None,
                    "samples_us": [],
                    "iterations_per_sample": iterations_per_sample,
                },
                None,
            )

    baseline, reference = measure("cublas_plus_topk", run_cublas)
    if not baseline["eligible"] or reference is None:
        raise RuntimeError(f"Gemma4 prefill router baseline failed: {baseline}")
    candidate, _ = measure("fused_matrix_router", run_fused, reference)
    baseline_recheck, _ = measure(
        "cublas_plus_topk_recheck",
        run_cublas,
        reference,
    )
    if not baseline_recheck["eligible"]:
        raise RuntimeError(
            f"Gemma4 prefill router baseline recheck failed: {baseline_recheck}"
        )

    baseline_first_us = float(baseline["median_us"])
    baseline_recheck_us = float(baseline_recheck["median_us"])
    conservative_baseline_us = min(baseline_first_us, baseline_recheck_us)
    candidate_us = (
        float(candidate["median_us"])
        if candidate["eligible"] and candidate["median_us"] is not None
        else float("inf")
    )
    speedup = (
        conservative_baseline_us / candidate_us
        if candidate_us < float("inf")
        else 0.0
    )
    beats_both_baselines = bool(
        candidate["eligible"]
        and baseline_first_us / candidate_us >= 1.02
        and baseline_recheck_us / candidate_us >= 1.02
    )
    apply_candidate = bool(beats_both_baselines and speedup >= 1.02)

    configured_layers = 0
    selected_layers = 0
    for layer in engine.model.layers:
        setter = getattr(layer.mlp.gate, setter_name, None)
        if callable(setter):
            setter(rows, apply_candidate)
            configured_layers += 1
            selected_layers += int(apply_candidate)
    if configured_layers != len(engine.model.layers):
        raise RuntimeError(
            "Gemma4 fused prefill router selection did not reach every layer: "
            f"{configured_layers}/{len(engine.model.layers)}"
        )

    return {
        "decision": (
            "APPLY_FUSED_MATRIX_ROUTER"
            if apply_candidate
            else "KEEP_CUBLAS_ROUTER"
        ),
        "selected": (
            "fused_matrix_router" if apply_candidate else "cublas_plus_topk"
        ),
        "batch_size": batch_size,
        "rows": rows,
        "configured_layers": configured_layers,
        "selected_layers": selected_layers,
        "baseline_us": conservative_baseline_us,
        "baseline_stability_ratio": (
            max(baseline_first_us, baseline_recheck_us)
            / conservative_baseline_us
        ),
        "candidate_us": None if candidate_us == float("inf") else candidate_us,
        "speedup": speedup,
        "beats_both_baselines": beats_both_baselines,
        "minimum_promotion_speedup": 1.02,
        "estimated_savings_ms_per_prefill": (
            (conservative_baseline_us - candidate_us) * 30.0 / 1000.0
            if apply_candidate
            else 0.0
        ),
        "cases": [baseline, candidate, baseline_recheck],
    }


def validate_megagemm_runtime(
    engine,
    batch_size: int,
    compact_gate: dict[str, Any],
    lm_head_gate: dict[str, Any],
    scheduler_burst_gate: dict[str, Any],
    attention_kernel_gate: dict[str, Any],
    next_attn_norm_gate: dict[str, Any],
    prefill_gate: dict[str, Any],
) -> dict[str, Any]:
    import megagemm.kernels.paged_attention as paged_attention

    status = engine.model.decode_runtime_stats()
    scheduler_stats = engine._last_scheduler.get_stats()
    decode_graph_stats = dict(scheduler_stats.get("decode_cuda_graphs") or {})
    prefill_graph_stats = dict(scheduler_stats.get("prefill_cuda_graphs") or {})
    policy = status.get("gemma4_batch_moe_decode_policy") or {}
    paths = status.get("gemma4_batch_moe_decode_last_paths") or {}
    decode_kernel_tune = compact_gate.get("decode_kernel_tune") or {}
    selected_path = str(
        decode_kernel_tune.get("selected_path") or "expert_grouped_compact"
    )
    expected_path = selected_path if batch_size >= 9 else "assignment"
    selected_config = dict(decode_kernel_tune.get("selected_config") or {})
    selected_attention_config = dict(
        attention_kernel_gate.get("selected_config") or {}
    )
    runtime_attention_config = {
        "gqa2_direct": env_flag("MEGAGEMM_PAGED_DECODE_GQA2"),
        "grouped_segmented": env_flag(
            "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_DECODE"
        ),
        "warps_h256": int(
            os.environ.get("MEGAGEMM_PAGED_DECODE_WARPS_H256", "0") or 0
        ),
        "warps_h512": int(
            os.environ.get("MEGAGEMM_PAGED_DECODE_WARPS_H512", "0") or 0
        ),
    }
    prefill_selection = dict(
        (prefill_gate.get("selected_by_rows") or {}).get(
            str(int(batch_size) * 25),
            {},
        )
    )
    prefill_bridge_selection = dict(
        prefill_gate.get("attn_moe_bridge") or {}
    )
    prefill_router_selection = dict(prefill_gate.get("router") or {})
    expected_prefill_config = dict(
        prefill_selection.get("selected_config") or {}
    )
    first_experts = engine.model.layers[0].mlp.experts
    effective_prefill_options = first_experts._segmented_prefill_kernel_options(
        int(batch_size) * 25
    )
    effective_prefill_config = {
        key: int(effective_prefill_options[key])
        for key in expected_prefill_config
    }
    scatter_stats_fn = getattr(
        engine.block_manager,
        "prefill_kv_scatter_stats",
        None,
    )
    prefill_kv_scatter = (
        dict(scatter_stats_fn()) if callable(scatter_stats_fn) else {}
    )
    result = {
        "policy": policy,
        "last_paths": paths,
        "expected_path": expected_path,
        "selected_path": selected_path,
        "prefill_tile_decision": str(prefill_selection.get("decision") or ""),
        "prefill_tile_config": effective_prefill_config,
        "prefill_kv_scatter": prefill_kv_scatter,
        "prefill_graph": prefill_graph_stats,
        "prefill_attn_moe_bridge_decision": str(
            prefill_bridge_selection.get("decision") or ""
        ),
        "prefill_attn_moe_bridge_rows": int(
            prefill_bridge_selection.get("rows", 0) or 0
        ),
        "fused_attn_moe_bridge_prefill_enabled": bool(
            status.get(
                "gemma4_fused_attn_moe_bridge_prefill_enabled",
                False,
            )
        ),
        "fused_attn_moe_bridge_prefill_hits": int(
            status.get(
                "gemma4_fused_attn_moe_bridge_prefill_hits",
                0,
            )
            or 0
        ),
        "fused_attn_moe_router_bridge_prefill_hits": int(
            status.get(
                "gemma4_fused_attn_moe_router_bridge_prefill_hits",
                0,
            )
            or 0
        ),
        "fused_attn_moe_bridge_prefill_enabled_layers": int(
            status.get(
                "gemma4_fused_attn_moe_bridge_prefill_enabled_layers",
                0,
            )
            or 0
        ),
        "prefill_attn_moe_bridge_error": str(
            status.get("gemma4_prefill_attn_moe_bridge_error", "") or ""
        ),
        "prefill_router_decision": str(
            prefill_router_selection.get("decision") or ""
        ),
        "prefill_router_rows": int(
            prefill_router_selection.get("rows", 0) or 0
        ),
        "fused_router_prefill_enabled": bool(
            status.get("gemma4_router_fused_prefill_enabled", False)
        ),
        "fused_router_prefill_hits": int(
            status.get("gemma4_router_fused_prefill_hits", 0) or 0
        ),
        "fused_router_prefill_selected_layers": int(
            status.get("gemma4_router_fused_prefill_selected_layers", 0) or 0
        ),
        "fused_router_prefill_disabled_layers": int(
            status.get("gemma4_router_fused_prefill_disabled_layers", 0) or 0
        ),
        "fused_router_prefill_error": str(
            status.get("gemma4_router_fused_prefill_error", "") or ""
        ),
        "compact_hits": int(
            status.get("gemma4_batch_moe_decode_compact_hits", 0) or 0
        ),
        "grouped_disabled_layers": int(
            status.get("qwen3_moe_grouped_decode_disabled_layers", 0) or 0
        ),
        "grouped_failure": str(
            status.get("qwen3_moe_grouped_decode_first_failure", "") or ""
        ),
        "compact_disabled_layers": int(
            status.get("gemma4_batch_moe_decode_compact_disabled_layers", 0) or 0
        ),
        "compact_failure": str(
            status.get("gemma4_batch_moe_decode_compact_first_failure", "") or ""
        ),
        "deterministic_reduce_layers": int(
            status.get(
                "gemma4_batch_moe_decode_deterministic_reduce_layers",
                0,
            )
            or 0
        ),
        "expert_grid_pack_layers": int(
            status.get(
                "gemma4_batch_moe_decode_expert_grid_pack_layers",
                0,
            )
            or 0
        ),
        "coalesced_weight_layers": int(
            status.get(
                "gemma4_batch_moe_decode_coalesced_weight_layers",
                0,
            )
            or 0
        ),
        "selected_compact_config": selected_config,
        "runtime_compact_config": {
            "active_list": bool(
                status.get(
                    "qwen3_moe_expert_grouped_compact_active_list",
                    False,
                )
            ),
            "active_list_early_exit": bool(
                status.get(
                    "qwen3_moe_expert_grouped_compact_active_list_early_exit",
                    False,
                )
            ),
            "expert_grid_pack": bool(
                status.get(
                    "qwen3_moe_expert_grouped_compact_expert_grid_pack",
                    False,
                )
            ),
            "coalesced_weights": bool(
                status.get(
                    "qwen3_moe_expert_grouped_compact_coalesced_weights",
                    False,
                )
            ),
            "gate_block_n": int(
                status.get(
                    "qwen3_moe_expert_grouped_compact_gate_block_n",
                    0,
                )
                or 0
            ),
            "down_block_n": int(
                status.get(
                    "qwen3_moe_expert_grouped_compact_down_block_n",
                    0,
                )
                or 0
            ),
            "num_warps": int(
                status.get(
                    "qwen3_moe_expert_grouped_compact_num_warps",
                    0,
                )
                or 0
            ),
            "num_stages": int(
                status.get(
                    "qwen3_moe_expert_grouped_compact_num_stages",
                    0,
                )
                or 0
            ),
            "gate_num_stages": int(
                status.get(
                    "qwen3_moe_expert_grouped_compact_gate_num_stages",
                    0,
                )
                or 0
            ),
            "down_num_stages": int(
                status.get(
                    "qwen3_moe_expert_grouped_compact_down_num_stages",
                    0,
                )
                or 0
            ),
            "experts_per_program": int(
                status.get(
                    "qwen3_moe_expert_grouped_compact_experts_per_program",
                    0,
                )
                or 0
            ),
            "paired_gate_up_dot": bool(
                status.get(
                    "qwen3_moe_expert_grouped_compact_paired_gate_up_dot",
                    False,
                )
            ),
            "split_gate_up": bool(
                status.get(
                    "qwen3_moe_expert_grouped_compact_split_gate_up",
                    False,
                )
            ),
            "empty_expert_early_exit": bool(
                status.get(
                    "qwen3_moe_expert_grouped_compact_empty_expert_early_exit",
                    False,
                )
            ),
            "l2_grouped_grid": bool(
                status.get(
                    "qwen3_moe_expert_grouped_compact_l2_grouped_grid",
                    False,
                )
            ),
            "l2_group_size": int(
                status.get(
                    "qwen3_moe_expert_grouped_compact_l2_group_size",
                    0,
                )
                or 0
            ),
        },
        "prefill_partial_reduce_layers": int(
            status.get(
                "qwen3_moe_segmented_prefill_partial_reduce_layers",
                0,
            )
            or 0
        ),
        "prefill_partial_reduce_hits": int(
            status.get(
                "qwen3_moe_segmented_prefill_partial_reduce_hits",
                0,
            )
            or 0
        ),
        "prefill_single_accumulator_layers": int(
            status.get(
                "qwen3_moe_segmented_prefill_single_accumulator_layers",
                0,
            )
            or 0
        ),
        "prefill_single_accumulator_hits": int(
            status.get(
                "qwen3_moe_segmented_prefill_single_accumulator_hits",
                0,
            )
            or 0
        ),
        "prefill_compact_route_pack_layers": int(
            status.get(
                "qwen3_moe_segmented_prefill_compact_route_pack_layers",
                0,
            )
            or 0
        ),
        "prefill_compact_route_pack_hits": int(
            status.get(
                "qwen3_moe_segmented_prefill_compact_route_pack_hits",
                0,
            )
            or 0
        ),
        "prefill_compact_route_pack_single_scan_layers": int(
            status.get(
                "qwen3_moe_segmented_prefill_compact_route_pack_single_scan_layers",
                0,
            )
            or 0
        ),
        "vectorized_prefill_kv_hits": int(
            status.get("gemma4_batch_prefill_vectorized_kv_hits", 0) or 0
        ),
        "implicit_causal_prefill_enabled": bool(
            status.get("gemma4_implicit_causal_prefill_enabled", False)
        ),
        "implicit_causal_prefill_batches": int(
            status.get("gemma4_implicit_causal_prefill_batches", 0) or 0
        ),
        "implicit_causal_prefill_hits": int(
            status.get("gemma4_implicit_causal_prefill_hits", 0) or 0
        ),
        "parallel_moe_prefill_enabled": bool(
            status.get("gemma4_parallel_moe_prefill_enabled", False)
        ),
        "parallel_moe_prefill_hits": int(
            status.get("gemma4_parallel_moe_prefill_hits", 0) or 0
        ),
        "parallel_moe_prefill_policy": dict(
            status.get("gemma4_parallel_moe_prefill_policy") or {}
        ),
        "last_token_only_hits": int(
            status.get("prefill_last_token_only_hits", 0) or 0
        ),
        "parallel_moe_enabled": bool(
            status.get("gemma4_parallel_moe_decode_enabled", False)
        ),
        "parallel_moe_hits": int(
            status.get("gemma4_parallel_moe_decode_hits", 0) or 0
        ),
        "parallel_moe_policy": dict(
            status.get("gemma4_parallel_moe_decode_policy") or {}
        ),
        "fused_attn_moe_bridge_decode_enabled": bool(
            status.get(
                "gemma4_fused_attn_moe_bridge_decode_enabled",
                False,
            )
        ),
        "fused_attn_moe_bridge_decode_hits": int(
            status.get(
                "gemma4_fused_attn_moe_bridge_decode_hits",
                0,
            )
            or 0
        ),
        "fused_attn_moe_router_bridge_decode_enabled": bool(
            status.get(
                "gemma4_fused_attn_moe_router_bridge_decode_enabled",
                False,
            )
        ),
        "fused_attn_moe_router_bridge_decode_hits": int(
            status.get(
                "gemma4_fused_attn_moe_router_bridge_decode_hits",
                0,
            )
            or 0
        ),
        "fused_attn_moe_router_single_kernel_decode_enabled": bool(
            status.get(
                "gemma4_fused_attn_moe_router_single_kernel_decode_enabled",
                False,
            )
        ),
        "fused_attn_moe_router_single_kernel_decode_hits": int(
            status.get(
                "gemma4_fused_attn_moe_router_single_kernel_decode_hits",
                0,
            )
            or 0
        ),
        "fused_router_compact_pack_decode_enabled": bool(
            status.get(
                "gemma4_fused_router_compact_pack_decode_enabled",
                False,
            )
        ),
        "fused_router_compact_pack_decode_hits": int(
            status.get(
                "gemma4_fused_router_compact_pack_decode_hits",
                0,
            )
            or 0
        ),
        "fused_router_compact_pack_disabled_layers": int(
            status.get("gemma4_router_compact_pack_disabled_layers", 0) or 0
        ),
        "fused_router_compact_pack_error": str(
            status.get("gemma4_router_compact_pack_error", "") or ""
        ),
        "fused_router_compact_pack_workspace_disabled_layers": int(
            status.get(
                "gemma4_router_compact_pack_workspace_disabled_layers",
                0,
            )
            or 0
        ),
        "fused_router_compact_pack_workspace_error": str(
            status.get("gemma4_router_compact_pack_workspace_error", "") or ""
        ),
        "fused_post_moe_norm_residual_enabled": bool(
            status.get(
                "gemma4_fused_post_moe_norm_residual_decode_enabled",
                False,
            )
        ),
        "fused_post_moe_norm_residual_hits": int(
            status.get(
                "gemma4_fused_post_moe_norm_residual_decode_hits",
                0,
            )
            or 0
        ),
        "fused_expert_reduce_post_moe_enabled": bool(
            status.get(
                "gemma4_fused_expert_reduce_post_moe_decode_enabled",
                False,
            )
        ),
        "fused_expert_reduce_post_moe_hits": int(
            status.get(
                "gemma4_fused_expert_reduce_post_moe_decode_hits",
                0,
            )
            or 0
        ),
        "fused_expert_reduce_post_moe_layers": int(
            status.get(
                "gemma4_batch_moe_decode_fused_post_moe_layers",
                0,
            )
            or 0
        ),
        "fused_next_attn_norm_supported": bool(
            status.get(
                "gemma4_fused_next_attn_norm_decode_supported",
                False,
            )
        ),
        "fused_next_attn_norm_enabled": bool(
            status.get(
                "gemma4_fused_next_attn_norm_decode_enabled",
                False,
            )
        ),
        "fused_next_attn_norm_hits": int(
            status.get("gemma4_fused_next_attn_norm_decode_hits", 0) or 0
        ),
        "fused_layer_scalar_hits": int(
            status.get("gemma4_fused_layer_scalar_decode_hits", 0) or 0
        ),
        "fused_router_expert_input_norm_enabled": bool(
            status.get(
                "gemma4_fused_router_expert_input_norm_decode_enabled",
                False,
            )
        ),
        "fused_router_expert_input_norm_hits": int(
            status.get(
                "gemma4_fused_router_expert_input_norm_decode_hits",
                0,
            )
            or 0
        ),
        "fused_qkv_prefill_hits": int(
            status.get("gemma4_fused_qkv_prefill_hits", 0) or 0
        ),
        "fused_qkv_prefill_skip_reason": str(
            status.get("gemma4_fused_qkv_prefill_skip_reason", "") or ""
        ),
        "fused_attn_prepare_hits": int(
            status.get("gemma4_fused_attn_prepare_hits", 0) or 0
        ),
        "fused_attn_prepare_disabled_layers": int(
            status.get("gemma4_fused_attn_prepare_disabled_layers", 0) or 0
        ),
        "fused_attn_prepare_skip_reason": str(
            status.get("gemma4_fused_attn_prepare_skip_reason", "") or ""
        ),
        "fused_router_decode_enabled": bool(
            status.get("gemma4_router_fused_decode_enabled", False)
        ),
        "fused_router_decode_hits": int(
            status.get("gemma4_router_fused_decode_hits", 0) or 0
        ),
        "fused_router_decode_selected_layers": int(
            status.get("gemma4_router_fused_decode_selected_layers", 0) or 0
        ),
        "fused_router_decode_disabled_layers": int(
            status.get("gemma4_router_fused_decode_disabled_layers", 0) or 0
        ),
        "fused_router_decode_last_paths": dict(
            status.get("gemma4_router_fused_decode_last_paths") or {}
        ),
        "fused_router_decode_error": str(
            status.get("gemma4_router_fused_decode_error", "") or ""
        ),
        "batch_cublas_lm_head_enabled": bool(
            status.get("gemma4_batch_cublas_lm_head_enabled", False)
        ),
        "batch_cublas_lm_head_hits": int(
            status.get("gemma4_batch_cublas_lm_head_hits", 0) or 0
        ),
        "batch_fused_softcap_argmax_enabled": bool(
            status.get("gemma4_batch_fused_softcap_argmax_enabled", False)
        ),
        "batch_fused_softcap_argmax_available": bool(
            status.get("gemma4_batch_fused_softcap_argmax_available", False)
        ),
        "batch_fused_softcap_argmax_hits": int(
            status.get("gemma4_batch_fused_softcap_argmax_hits", 0) or 0
        ),
        "batch_fused_softcap_argmax_disabled": bool(
            status.get("gemma4_batch_fused_softcap_argmax_disabled", False)
        ),
        "batch_fused_softcap_argmax_error": str(
            status.get("gemma4_batch_fused_softcap_argmax_error", "") or ""
        ),
        "scheduler_greedy_token_steps": int(
            decode_graph_stats.get("greedy_token_steps", 0) or 0
        ),
        "scheduler_batched_token_host_copies": int(
            decode_graph_stats.get("batched_token_host_copies", 0) or 0
        ),
        "scheduler_vectorized_input_updates": int(
            decode_graph_stats.get("vectorized_input_updates", 0) or 0
        ),
        "scheduler_greedy_token_shape_graphs": int(
            decode_graph_stats.get("greedy_token_shape_graphs", 0) or 0
        ),
        "scheduler_token_bursts": int(
            decode_graph_stats.get("token_bursts", 0) or 0
        ),
        "scheduler_token_burst_steps": int(
            decode_graph_stats.get("token_burst_steps", 0) or 0
        ),
        "scheduler_token_feedback_copies": int(
            decode_graph_stats.get("token_feedback_copies", 0) or 0
        ),
        "scheduler_persistent_token_feedback_enabled": bool(
            decode_graph_stats.get("persistent_token_feedback_enabled", False)
        ),
        "scheduler_persistent_token_feedback_steps": int(
            decode_graph_stats.get("persistent_token_feedback_steps", 0) or 0
        ),
        "scheduler_chain_input_updates_skipped": int(
            decode_graph_stats.get("chain_input_updates_skipped", 0) or 0
        ),
        "selected_attention_config": selected_attention_config,
        "runtime_attention_config": runtime_attention_config,
        "paged_decode_runtime": paged_attention.paged_decode_runtime_stats(),
    }
    embed_scale = getattr(engine.model, "gemma4_embed_scale", None)
    embed_weight = engine.model.embed_tokens.weight
    expected_embed_scale = torch.tensor(
        float(engine.model.embed_scale),
        dtype=embed_weight.dtype,
    ).item()
    embed_scale_contract = {
        "present": isinstance(embed_scale, torch.Tensor),
        "dtype": str(embed_scale.dtype) if isinstance(embed_scale, torch.Tensor) else "",
        "expected_dtype": str(embed_weight.dtype),
        "device": str(embed_scale.device) if isinstance(embed_scale, torch.Tensor) else "",
        "expected_device": str(embed_weight.device),
        "value": float(embed_scale.item()) if isinstance(embed_scale, torch.Tensor) else None,
        "expected_value": float(expected_embed_scale),
    }
    embed_scale_contract["exact"] = bool(
        embed_scale_contract["present"]
        and embed_scale.dtype == embed_weight.dtype
        and embed_scale.device == embed_weight.device
        and embed_scale_contract["value"] == embed_scale_contract["expected_value"]
    )
    result["embedding_scale_contract"] = embed_scale_contract
    if not embed_scale_contract["exact"]:
        raise RuntimeError(
            "Gemma4 embedding scale is not materialized in checkpoint dtype: "
            + json.dumps(embed_scale_contract, sort_keys=True)
        )
    result["batch_cublas_lm_head_graph_replay"] = bool(
        result["scheduler_greedy_token_steps"] > 0
        and result["scheduler_greedy_token_shape_graphs"] > 0
    )
    if batch_size == 16 and runtime_attention_config["grouped_segmented"]:
        paged_runtime = result["paged_decode_runtime"]
        if (
            int(paged_runtime.get("grouped_segmented_hits", 0) or 0) <= 0
            or bool(paged_runtime.get("grouped_segmented_disabled", False))
        ):
            raise RuntimeError(
                "Gemma4 grouped segmented attention did not reach the "
                f"captured B16 runtime: {paged_runtime}"
            )
    if int(policy.get("enabled_layers", 0) or 0) != 30:
        raise RuntimeError(f"Gemma4 batch policy is not active on all layers: {result}")
    expected_compact_path_layers = (
        30 if selected_path == "expert_grouped_compact" else 0
    )
    if int(policy.get("compact_path_layers", 0) or 0) != expected_compact_path_layers:
        raise RuntimeError(
            f"Gemma4 selected MoE path was not propagated to all layers: {result}"
        )
    if result["grouped_disabled_layers"] or result["compact_disabled_layers"]:
        raise RuntimeError(f"Gemma4 grouped decode disabled itself: {result}")
    if int(paths.get(expected_path, 0) or 0) != 30:
        raise RuntimeError(f"Gemma4 batch used the wrong MoE path: {result}")
    if result["deterministic_reduce_layers"] != 30:
        raise RuntimeError(f"Gemma4 batch used atomic MoE reduction: {result}")
    if batch_size == 16:
        selected_compact = selected_path == "expert_grouped_compact"
        expected_grid_layers = (
            30
            if selected_compact and selected_config.get("expert_grid_pack")
            else 0
        )
        expected_coalesced_layers = (
            30
            if selected_compact and selected_config.get("coalesced_weights")
            else 0
        )
        if result["expert_grid_pack_layers"] != expected_grid_layers:
            raise RuntimeError(
                f"Gemma4 B16 selected route pack was not exercised: {result}"
            )
        if result["coalesced_weight_layers"] != expected_coalesced_layers:
            raise RuntimeError(
                f"Gemma4 B16 selected weight layout was not exercised: {result}"
            )
        if result["runtime_compact_config"] != selected_config:
            raise RuntimeError(
                f"Gemma4 B16 compact autotune config drifted at runtime: {result}"
            )
    single_accumulator_applied = (
        prefill_selection.get("decision") == "APPLY_SINGLE_ACCUMULATOR"
    )
    expected_single_accumulator_layers = 30 if single_accumulator_applied else 0
    if (
        result["prefill_single_accumulator_layers"]
        != expected_single_accumulator_layers
    ):
        raise RuntimeError(
            "Gemma4 single-accumulator prefill selection drifted at runtime: "
            f"{result}"
        )
    if (
        single_accumulator_applied
        and result["prefill_single_accumulator_hits"] <= 0
    ):
        raise RuntimeError(
            "Gemma4 single-accumulator prefill was selected but not exercised: "
            f"{result}"
        )
    if result["prefill_partial_reduce_layers"] != 30:
        raise RuntimeError(f"Gemma4 prefill used atomic MoE reduction: {result}")
    if result["prefill_partial_reduce_hits"] <= 0:
        raise RuntimeError(
            f"Gemma4 prefill reported no partial reductions: {result}"
        )
    if prefill_gate.get("selected") != "fp32_partial":
        raise RuntimeError(f"Gemma4 prefill partial contract drifted: {prefill_gate}")
    if (
        int(prefill_selection.get("selected_layers", 0) or 0) != 30
        or effective_prefill_config != expected_prefill_config
    ):
        raise RuntimeError(
            "Gemma4 prefill tile autotune was not applied to the runtime: "
            f"expected={expected_prefill_config}, effective={effective_prefill_config}, "
            f"selection={prefill_selection}"
        )
    bridge_applied = (
        prefill_bridge_selection.get("decision")
        == "APPLY_FUSED_ATTN_MOE_BRIDGE"
    )
    bridge_is_current_shape = (
        int(batch_size) * 25
        == int(prefill_bridge_selection.get("rows", 0) or 0)
    )
    bridge_rolled_back = bool(
        prefill_bridge_selection.get("post_contract_rollback")
    )
    if bridge_applied and bridge_is_current_shape and (
        not result["fused_attn_moe_bridge_prefill_enabled"]
        or result["fused_attn_moe_bridge_prefill_enabled_layers"] != 30
        or result["fused_attn_moe_bridge_prefill_hits"] <= 0
        or result["fused_attn_moe_router_bridge_prefill_hits"] <= 0
        or result["prefill_attn_moe_bridge_error"]
    ):
        raise RuntimeError(
            "Gemma4 fused prefill attention-to-MoE bridge was not exercised: "
            f"{result}"
        )
    if not bridge_applied and bridge_is_current_shape and (
        result["fused_attn_moe_bridge_prefill_enabled"]
        or result["fused_attn_moe_bridge_prefill_enabled_layers"] != 0
        or (
            not bridge_rolled_back
            and result["fused_attn_moe_bridge_prefill_hits"] != 0
        )
        or (
            not bridge_rolled_back
            and result["fused_attn_moe_router_bridge_prefill_hits"] != 0
        )
    ):
        raise RuntimeError(
            "Rejected Gemma4 prefill attention-to-MoE bridge became active: "
            f"{result}"
        )
    router_applied = (
        prefill_router_selection.get("decision")
        == "APPLY_FUSED_MATRIX_ROUTER"
    )
    router_is_current_shape = (
        int(batch_size) * 25
        == int(prefill_router_selection.get("rows", 0) or 0)
    )
    if router_applied and router_is_current_shape and (
        not result["fused_router_prefill_enabled"]
        or result["fused_router_prefill_selected_layers"] != 30
        or result["fused_router_prefill_hits"] <= 0
        or result["fused_router_prefill_disabled_layers"] != 0
        or result["fused_router_prefill_error"]
    ):
        raise RuntimeError(
            "Gemma4 fused 400-row prefill router was not exercised: "
            f"{result}"
        )
    if not router_applied and router_is_current_shape and (
        result["fused_router_prefill_selected_layers"] != 0
    ):
        raise RuntimeError(
            "Rejected Gemma4 fused 400-row prefill router became active: "
            f"{result}"
        )
    if (
        result["prefill_compact_route_pack_layers"] != 30
        or result["prefill_compact_route_pack_hits"] <= 0
        or result["prefill_compact_route_pack_single_scan_layers"] != 30
    ):
        raise RuntimeError(f"Gemma4 compact prefill route pack was not exercised: {result}")
    if (
        batch_size >= 9
        and selected_path == "expert_grouped_compact"
        and result["compact_hits"] <= 0
    ):
        raise RuntimeError(f"Gemma4 compact path reported no hits: {result}")
    if result["vectorized_prefill_kv_hits"] <= 0:
        raise RuntimeError(f"Gemma4 batch KV writes were not vectorized: {result}")
    if result["last_token_only_hits"] <= 0:
        raise RuntimeError(f"Gemma4 batch prefill projected all tokens: {result}")
    if batch_size in (8, 16) and (
        not result["parallel_moe_enabled"]
        or result["parallel_moe_hits"] <= 0
        or int(result["parallel_moe_policy"].get("rows", 0) or 0) != batch_size
        or not bool(
            result["parallel_moe_policy"].get("isolated_shared_norm_buffers", False)
        )
        or not bool(
            result["parallel_moe_policy"].get("fork_before_router", False)
        )
    ):
        raise RuntimeError(f"Gemma4 parallel MoE decode was not exercised: {result}")
    attn_moe_decode_bridge_selected = bool(
        result["parallel_moe_policy"].get(
            "fused_attn_moe_bridge_requested",
            False,
        )
    )
    if batch_size == 16 and (
        result["fused_attn_moe_bridge_decode_enabled"]
        != attn_moe_decode_bridge_selected
        or (
            attn_moe_decode_bridge_selected
            and (
                result["fused_attn_moe_bridge_decode_hits"] <= 0
                or not bool(
                    result["parallel_moe_policy"].get(
                        "isolated_attn_moe_bridge_buffers",
                        False,
                    )
                )
            )
        )
        or (
            not attn_moe_decode_bridge_selected
            and result["fused_attn_moe_bridge_decode_hits"] != 0
        )
    ):
        raise RuntimeError(
            "Gemma4 attention-to-MoE decode bridge selection drifted: "
            f"{result}"
        )
    if batch_size != 16 and (
        result["fused_attn_moe_bridge_decode_enabled"]
        or result["fused_attn_moe_bridge_decode_hits"] != 0
    ):
        raise RuntimeError(
            "Gemma4 B16-only attention-to-MoE decode bridge leaked to "
            f"another batch: {result}"
        )
    attn_moe_router_bridge_selected = bool(
        result["parallel_moe_policy"].get(
            "fused_attn_moe_router_bridge_requested",
            False,
        )
    )
    if batch_size == 16 and (
        result["fused_attn_moe_router_bridge_decode_enabled"]
        != attn_moe_router_bridge_selected
        or (
            attn_moe_router_bridge_selected
            and (
                result["fused_attn_moe_router_bridge_decode_hits"] <= 0
                or not bool(
                    result["parallel_moe_policy"].get(
                        "isolated_attn_moe_router_buffers",
                        False,
                    )
                )
            )
        )
        or (
            not attn_moe_router_bridge_selected
            and result["fused_attn_moe_router_bridge_decode_hits"] != 0
        )
    ):
        raise RuntimeError(
            "Gemma4 attention-to-MoE/router decode bridge selection drifted: "
            f"{result}"
        )
    if batch_size != 16 and (
        result["fused_attn_moe_router_bridge_decode_enabled"]
        or result["fused_attn_moe_router_bridge_decode_hits"] != 0
    ):
        raise RuntimeError(
            "Gemma4 B16-only attention-to-MoE/router decode bridge leaked to "
            f"another batch: {result}"
        )
    attn_moe_router_single_kernel_selected = bool(
        result["parallel_moe_policy"].get(
            "fused_attn_moe_router_single_kernel_requested",
            False,
        )
    )
    if batch_size == 16 and (
        result["fused_attn_moe_router_single_kernel_decode_enabled"]
        != attn_moe_router_single_kernel_selected
        or (
            attn_moe_router_single_kernel_selected
            and result["fused_attn_moe_router_single_kernel_decode_hits"] <= 0
        )
        or (
            not attn_moe_router_single_kernel_selected
            and result["fused_attn_moe_router_single_kernel_decode_hits"] != 0
        )
    ):
        raise RuntimeError(
            "Gemma4 single-kernel attention-to-MoE/router bridge selection "
            f"drifted: {result}"
        )
    if batch_size != 16 and (
        result["fused_attn_moe_router_single_kernel_decode_enabled"]
        or result["fused_attn_moe_router_single_kernel_decode_hits"] != 0
    ):
        raise RuntimeError(
            "Gemma4 B16-only single-kernel attention-to-MoE/router bridge "
            f"leaked to another batch: {result}"
        )
    router_compact_pack_selected = bool(
        result["parallel_moe_policy"].get(
            "fused_router_compact_pack_requested",
            False,
        )
    )
    if batch_size == 16 and (
        result["fused_router_compact_pack_decode_enabled"]
        != router_compact_pack_selected
        or (
            router_compact_pack_selected
            and (
                result["fused_router_compact_pack_decode_hits"] <= 0
                or result["fused_router_compact_pack_disabled_layers"] != 0
                or bool(result["fused_router_compact_pack_error"])
                or result[
                    "fused_router_compact_pack_workspace_disabled_layers"
                ]
                != 0
                or bool(result["fused_router_compact_pack_workspace_error"])
            )
        )
        or (
            not router_compact_pack_selected
            and result["fused_router_compact_pack_decode_hits"] != 0
        )
    ):
        raise RuntimeError(
            "Gemma4 fused router/compact-pack selection drifted: "
            f"{result}"
        )
    if batch_size != 16 and (
        result["fused_router_compact_pack_decode_enabled"]
        or result["fused_router_compact_pack_decode_hits"] != 0
    ):
        raise RuntimeError(
            "Gemma4 B16-only fused router/compact-pack leaked to another batch: "
            f"{result}"
        )
    if batch_size == 16 and (
        result["fused_post_moe_norm_residual_enabled"]
        or result["fused_post_moe_norm_residual_hits"] != 0
    ):
        raise RuntimeError(
            f"Unproven Gemma4 fused post-MoE decode chain became active: {result}"
        )
    if batch_size == 16 and (
        not result["fused_expert_reduce_post_moe_enabled"]
        or result["fused_expert_reduce_post_moe_hits"] <= 0
        or result["fused_expert_reduce_post_moe_layers"] != 30
        or not bool(
            result["parallel_moe_policy"].get(
                "isolated_post_moe_output_buffers",
                False,
            )
        )
    ):
        raise RuntimeError(
            f"Gemma4 fused expert reduction/post-MoE chain was not exercised: {result}"
        )
    next_norm_selected = bool(next_attn_norm_gate.get("enabled"))
    if batch_size == 16 and (
        not result["fused_next_attn_norm_supported"]
        or result["fused_next_attn_norm_enabled"] != next_norm_selected
        or (
            next_norm_selected
            and (
                result["fused_next_attn_norm_hits"] <= 0
                or result["fused_layer_scalar_hits"] <= 0
            )
        )
        or (
            not next_norm_selected
            and (
                result["fused_next_attn_norm_hits"] != 0
                or result["fused_layer_scalar_hits"] != 0
            )
        )
    ):
        raise RuntimeError(
            "Gemma4 post-MoE/next-attention RMSNorm selection drifted: "
            f"{result}"
        )
    if batch_size != 16 and (
        result["fused_expert_reduce_post_moe_enabled"]
        or result["fused_expert_reduce_post_moe_hits"] != 0
        or result["fused_expert_reduce_post_moe_layers"] != 0
    ):
        raise RuntimeError(
            f"Gemma4 B16-only fused expert/post-MoE chain leaked to another batch: {result}"
        )
    if batch_size == 16 and (
        result["fused_router_expert_input_norm_enabled"]
        or result["fused_router_expert_input_norm_hits"] != 0
    ):
        raise RuntimeError(
            f"Unproven Gemma4 fused router/expert input norm became active: {result}"
        )
    if batch_size in (8, 16) and (
        result["fused_qkv_prefill_hits"] < 30
        or result["fused_attn_prepare_hits"] < 30
        or result["fused_attn_prepare_disabled_layers"] != 0
    ):
        raise RuntimeError(
            f"Gemma4 batch fused attention prefill was not exercised: {result}"
        )
    if (
        batch_size == 16
        and env_flag("MEGAGEMM_GEMMA4_IMPLICIT_CAUSAL_PREFILL")
        and (
            not result["implicit_causal_prefill_enabled"]
            or result["implicit_causal_prefill_batches"] <= 0
            or result["implicit_causal_prefill_hits"] < 25
        )
    ):
        raise RuntimeError(
            f"Gemma4 B16 implicit-causal prefill was not exercised: {result}"
        )
    if (
        batch_size == 16
        and env_flag("MEGAGEMM_GEMMA4_PARALLEL_MOE_PREFILL")
        and not env_flag("MEGAGEMM_PREFILL_CUDA_GRAPHS")
        and (
            not result["parallel_moe_prefill_enabled"]
            or result["parallel_moe_prefill_hits"] < 30
            or int(
                result["parallel_moe_prefill_policy"].get("batch_size", 0) or 0
            )
            != 16
            or int(
                result["parallel_moe_prefill_policy"].get("seq_len", 0) or 0
            )
            != 25
        )
    ):
        raise RuntimeError(
            f"Gemma4 B16 parallel MoE prefill was not exercised: {result}"
        )
    if (
        result["fused_router_decode_enabled"]
        or result["fused_router_decode_hits"] != 0
        or result["fused_router_decode_selected_layers"] != 0
        or result["fused_router_decode_disabled_layers"] != 0
        or int(
            result["fused_router_decode_last_paths"].get(
                "legacy",
                0,
            )
            or 0
        )
        != 30
    ):
        raise RuntimeError(f"Gemma4 legacy decode router lock was not exercised: {result}")
    scheduler_selected = str(scheduler_burst_gate.get("selected") or "")
    persistent_feedback_selected = bool(
        scheduler_selected == "fused_softcap_argmax_persistent_feedback"
    )
    fused_softcap_selected = scheduler_selected in {
        "fused_softcap_argmax",
        "fused_softcap_argmax_persistent_feedback",
    }
    greedy_token_backend_selected = bool(
        lm_head_gate.get("selected") == "cublas_greedy_token"
        and scheduler_selected in {
            "graph_token_burst",
            "fused_softcap_argmax",
            "fused_softcap_argmax_persistent_feedback",
        }
    )
    expected_cublas_lm_head = bool(
        batch_size == 16 and greedy_token_backend_selected
    )
    softcap_contracts = dict(
        scheduler_burst_gate.get("softcap_contracts") or {}
    )
    softcap_capture_evidence = dict(
        (
            scheduler_burst_gate.get("softcap_capture_evidence")
            or {}
        ).get(scheduler_selected)
        or {}
    )
    same_vm_softcap_contract = bool(
        softcap_contracts.get(scheduler_selected, False)
    )
    proven_softcap_contract = bool(
        str(scheduler_burst_gate.get("decision") or "").startswith(
            "USE_PROVEN_FUSED_SOFTCAP"
        )
        and scheduler_burst_gate.get("token_reference_source")
        == "paid_exact_softcap_gate"
    )
    result["batch_fused_softcap_argmax_gate_contract"] = bool(
        same_vm_softcap_contract or proven_softcap_contract
    )
    result["batch_fused_softcap_argmax_gate_contract_source"] = (
        "same_vm_capture"
        if same_vm_softcap_contract
        else ("paid_exact_proven" if proven_softcap_contract else "")
    )
    result["batch_fused_softcap_argmax_capture_evidence"] = (
        softcap_capture_evidence
    )
    result["batch_fused_softcap_argmax_replay_exercised"] = bool(
        result["batch_cublas_lm_head_graph_replay"]
        and result["scheduler_greedy_token_steps"] > 0
        and result["scheduler_token_burst_steps"]
        == result["scheduler_greedy_token_steps"]
    )
    if (
        result["batch_cublas_lm_head_enabled"] != greedy_token_backend_selected
        or (
            expected_cublas_lm_head
            and result["batch_cublas_lm_head_hits"] <= 0
            and not result["batch_cublas_lm_head_graph_replay"]
        )
    ):
        raise RuntimeError(
            f"Gemma4 selected batch LM-head backend was not exercised: {result}"
        )
    if batch_size == 16 and (
        result["batch_fused_softcap_argmax_enabled"] != fused_softcap_selected
        or (
            fused_softcap_selected
            and (
                not result["batch_fused_softcap_argmax_available"]
                or result["batch_fused_softcap_argmax_disabled"]
                or bool(result["batch_fused_softcap_argmax_error"])
                or not result["batch_fused_softcap_argmax_gate_contract"]
                or not result[
                    "batch_fused_softcap_argmax_replay_exercised"
                ]
            )
        )
    ):
        raise RuntimeError(
            f"Gemma4 fused softcap/argmax selection drifted: {result}"
        )
    if result["scheduler_vectorized_input_updates"] <= 0:
        raise RuntimeError(
            f"Gemma4 decode inputs were not updated as batched copies: {result}"
        )
    if result["scheduler_batched_token_host_copies"] <= 0:
        raise RuntimeError(
            f"Gemma4 greedy tokens were copied to host one-by-one: {result}"
        )
    if expected_cublas_lm_head and (
        result["scheduler_greedy_token_steps"] <= 0
        or result["scheduler_greedy_token_shape_graphs"] <= 0
        or result["scheduler_token_bursts"] <= 0
        or result["scheduler_token_burst_steps"]
        != result["scheduler_greedy_token_steps"]
        or result["scheduler_batched_token_host_copies"]
        >= result["scheduler_greedy_token_steps"]
    ):
        raise RuntimeError(
            f"Gemma4 greedy-token CUDA graph was not exercised: {result}"
        )
    if batch_size == 16 and persistent_feedback_selected and (
        not result["scheduler_persistent_token_feedback_enabled"]
        or result["scheduler_persistent_token_feedback_steps"]
        != result["scheduler_greedy_token_steps"]
        or result["scheduler_token_feedback_copies"] > 1
        or result["scheduler_chain_input_updates_skipped"] <= 0
    ):
        raise RuntimeError(
            "Gemma4 persistent graph token feedback contract drifted: "
            f"{result}"
        )
    if runtime_attention_config != selected_attention_config:
        raise RuntimeError(
            "Gemma4 attention kernel gate configuration drifted at runtime: "
            f"{result}"
        )
    if attention_kernel_gate.get("decision") == "APPLY":
        selected_attention_case = next(
            (
                row
                for row in attention_kernel_gate.get("cases", [])
                if row.get("case") == attention_kernel_gate.get("selected")
            ),
            {},
        )
        if (
            not bool(selected_attention_case.get("tokens_exact"))
            or not bool(selected_attention_case.get("shape_policy_exercised"))
            or selected_attention_case.get("error")
        ):
            raise RuntimeError(
                "Gemma4 selected attention warp policy was not exercised: "
                f"{result}"
            )
    if batch_size == 16 and env_flag("MEGAGEMM_PREFILL_CUDA_GRAPHS") and (
        not bool(prefill_graph_stats.get("enabled"))
        or int(prefill_graph_stats.get("warmups", 0) or 0) != 1
        or int(prefill_graph_stats.get("captures", 0) or 0) != 1
        or int(prefill_graph_stats.get("capture_body_warmups", 0) or 0) != 2
        or int(prefill_graph_stats.get("capture_replays", 0) or 0) != 1
        or int(prefill_graph_stats.get("replays", 0) or 0) < 2
        or int(prefill_graph_stats.get("failures", 0) or 0) != 0
        or int(prefill_graph_stats.get("buckets", 0) or 0) != 1
        or list(prefill_graph_stats.get("bucket_kinds") or []) != ["padded"]
        or list(prefill_graph_stats.get("kv_write_modes") or [])
        != ["external_after_replay"]
        or int(prefill_graph_stats.get("deferred_kv_layers", 0) or 0) != 30
        or int(prefill_graph_stats.get("external_kv_write_replays", 0) or 0)
        != int(prefill_graph_stats.get("capture_replays", 0) or 0)
        + int(prefill_graph_stats.get("replays", 0) or 0)
    ):
        raise RuntimeError(
            f"Gemma4 B16 prefill CUDA graph contract failed: {result}"
        )
    return result


def run_megagemm(
    args: argparse.Namespace,
    prompt_pool: list[list[int]],
) -> dict[str, Any]:
    os.environ.setdefault("MEGAGEMM_FP16_STREAMING", "1")
    os.environ.setdefault("MEGAGEMM_FLAT_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_DISABLE_CUDA_RMSNORM", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_SHAPE_CACHE", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE", "1")
    os.environ.setdefault("MEGAGEMM_DECODE_CUDA_GRAPHS_STABLE_MAX_BLOCKS", "1")

    from megagemm.engine import InferenceEngine
    from megagemm.engine.deterministic import is_deterministic

    engine = InferenceEngine(
        args.model,
        device="cuda",
        dtype=dtype_from_name(args.dtype),
        max_seq_len=args.max_seq_len,
        max_batch_size=max(args.batch_sizes),
        deterministic=args.deterministic,
    )
    determinism_contract = {
        "requested": bool(args.deterministic),
        "engine_enabled": bool(getattr(engine, "_deterministic", False)),
        "runtime_enabled": bool(is_deterministic()),
        "torch_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }
    if args.deterministic and (
        not determinism_contract["engine_enabled"]
        or not determinism_contract["runtime_enabled"]
        or not determinism_contract["torch_algorithms_enabled"]
        or determinism_contract["cublas_workspace_config"] != ":4096:8"
    ):
        raise RuntimeError(
            "MegaGemm deterministic contract was not activated: "
            + json.dumps(determinism_contract, sort_keys=True)
        )
    print(
        "MEGAGEMM_DETERMINISM_CONTRACT "
        + json.dumps(determinism_contract, sort_keys=True)
    )
    if args.prefill_finite_trace_only:
        trace = run_gemma4_prefill_finite_trace(
            engine,
            prompt_pool[:max(int(batch) for batch in args.batch_sizes)],
        )
        print(
            "MEGAGEMM_PREFILL_FINITE_TRACE "
            + json.dumps(trace, sort_keys=True)
        )
        return {
            "backend": "megagemm",
            "trace_only": True,
            "prefill_finite_trace": trace,
            "determinism_contract": determinism_contract,
        }
    prefill_correctness_gate = run_gemma4_prefill_correctness_gate(
        engine,
        prompt_pool[:max(int(batch) for batch in args.batch_sizes)],
    )
    print(
        "MEGAGEMM_PREFILL_CORRECTNESS_GATE "
        + json.dumps(prefill_correctness_gate, sort_keys=True)
    )
    first_token_contract = prefill_correctness_gate["selected_contract"]
    print(
        "MEGAGEMM_FIRST_TOKEN_CONTRACT "
        + json.dumps(first_token_contract, sort_keys=True)
    )
    lm_head_gate = run_batch_lm_head_kernel_gate(
        engine,
        max(int(batch) for batch in args.batch_sizes),
    )
    print("MEGAGEMM_LM_HEAD_KERNEL_GATE " + json.dumps(lm_head_gate, sort_keys=True))
    compact_gate = run_compact_kernel_gate(
        engine,
        prompt_pool[:max(int(batch) for batch in args.batch_sizes)],
        args.max_tokens,
    )
    print("MEGAGEMM_COMPACT_KERNEL_GATE " + json.dumps(compact_gate, sort_keys=True))
    compact_decision = str(
        compact_gate.get("decode_kernel_tune", {}).get("decision") or ""
    )
    prefill_graph_setting = os.environ.get("MEGAGEMM_PREFILL_CUDA_GRAPHS")
    os.environ["MEGAGEMM_PREFILL_CUDA_GRAPHS"] = "0"
    try:
        scheduler_burst_gate = run_scheduler_token_burst_gate(
            engine,
            prompt_pool[:max(int(batch) for batch in args.batch_sizes)],
            args.max_tokens,
            lm_head_gate,
        )
    finally:
        if prefill_graph_setting is None:
            os.environ.pop("MEGAGEMM_PREFILL_CUDA_GRAPHS", None)
        else:
            os.environ["MEGAGEMM_PREFILL_CUDA_GRAPHS"] = prefill_graph_setting
    print(
        "MEGAGEMM_SCHEDULER_BURST_GATE "
        + json.dumps(scheduler_burst_gate, sort_keys=True)
    )

    scheduler_decision = str(scheduler_burst_gate.get("decision") or "")
    decode_promoted = bool(
        compact_decision == "APPLY"
        or scheduler_decision.startswith("APPLY_")
    )
    if args.stop_after_no_decode_promotion and not decode_promoted:
        early_stop = {
            "reason": "no exact decode candidate cleared the full-request gate",
            "compact_decision": compact_decision,
            "scheduler_decision": scheduler_decision,
            "stable_baseline_preserved": True,
            "skipped": [
                "prefill_retuning",
                "measured_megagemm_matrix",
                "vllm_install_and_run",
            ],
        }
        print(
            "MEGAGEMM_EARLY_STOP_NO_DECODE_PROMOTION "
            + json.dumps(early_stop, sort_keys=True)
        )
        return {
            "status": "gate_rejected",
            "backend": "megagemm",
            "determinism_contract": determinism_contract,
            "prefill_correctness_gate": prefill_correctness_gate,
            "first_token_contract": first_token_contract,
            "lm_head_kernel_gate": lm_head_gate,
            "compact_kernel_gate": compact_gate,
            "scheduler_burst_gate": scheduler_burst_gate,
            "early_stop_no_decode_promotion": early_stop,
            "cases": {},
        }

    prefill_gate = run_segmented_prefill_kernel_gate(engine, args.batch_sizes)
    prefill_bridge_gate = run_gemma4_prefill_attn_moe_bridge_gate(
        engine,
        args.batch_sizes,
    )
    prefill_router_gate = run_gemma4_prefill_router_kernel_gate(
        engine,
        args.batch_sizes,
    )
    prefill_gate["attn_moe_bridge"] = prefill_bridge_gate
    prefill_gate["router"] = prefill_router_gate
    print("MEGAGEMM_PREFILL_KERNEL_GATE " + json.dumps(prefill_gate, sort_keys=True))
    print(
        "MEGAGEMM_PREFILL_ATTN_MOE_BRIDGE_GATE "
        + json.dumps(prefill_bridge_gate, sort_keys=True)
    )
    print(
        "MEGAGEMM_PREFILL_ROUTER_KERNEL_GATE "
        + json.dumps(prefill_router_gate, sort_keys=True)
    )

    prefill_graph_setting = os.environ.get("MEGAGEMM_PREFILL_CUDA_GRAPHS")
    os.environ["MEGAGEMM_PREFILL_CUDA_GRAPHS"] = "0"
    try:
        attention_kernel_gate = run_attention_decode_kernel_gate(
            engine,
            prompt_pool[:max(int(batch) for batch in args.batch_sizes)],
            args.max_tokens,
        )
        next_attn_norm_gate = run_fused_next_attn_norm_decode_gate(
            engine,
            prompt_pool[:max(int(batch) for batch in args.batch_sizes)],
            args.max_tokens,
        )
    finally:
        if prefill_graph_setting is None:
            os.environ.pop("MEGAGEMM_PREFILL_CUDA_GRAPHS", None)
        else:
            os.environ["MEGAGEMM_PREFILL_CUDA_GRAPHS"] = prefill_graph_setting
    print(
        "MEGAGEMM_ATTENTION_KERNEL_GATE "
        + json.dumps(attention_kernel_gate, sort_keys=True)
    )
    print(
        "MEGAGEMM_NEXT_ATTN_NORM_KERNEL_GATE "
        + json.dumps(next_attn_norm_gate, sort_keys=True)
    )

    gate_prompts = prompt_pool[:max(int(batch) for batch in args.batch_sizes)]
    oracle_tokens = [
        int(token) for token in prefill_correctness_gate["oracle"]["tokens"]
    ]
    post_kernel_contract = run_megagemm_first_token_contract(
        engine,
        gate_prompts,
        reference_tokens=oracle_tokens,
        raise_on_failure=False,
    )
    post_kernel_passed = bool(
        post_kernel_contract.get("all_finite")
        and post_kernel_contract.get("all_exact")
        and post_kernel_contract.get("all_reference_exact")
    )
    bridge_requested = bool(
        prefill_bridge_gate.get("decision")
        == "APPLY_FUSED_ATTN_MOE_BRIDGE"
    )
    router_requested = bool(
        prefill_router_gate.get("decision")
        == "APPLY_FUSED_MATRIX_ROUTER"
    )
    post_kernel_stats = engine.model.decode_runtime_stats()
    candidate_runtime_failures: list[str] = []
    if bridge_requested and (
        int(
            post_kernel_stats.get(
                "gemma4_fused_attn_moe_bridge_prefill_enabled_layers",
                0,
            )
            or 0
        )
        != 30
        or int(
            post_kernel_stats.get(
                "gemma4_fused_attn_moe_router_bridge_prefill_hits",
                0,
            )
            or 0
        )
        <= 0
        or str(
            post_kernel_stats.get("gemma4_prefill_attn_moe_bridge_error", "")
            or ""
        )
    ):
        candidate_runtime_failures.append("attn_moe_router_bridge")
    if router_requested and (
        int(
            post_kernel_stats.get(
                "gemma4_router_fused_prefill_selected_layers",
                0,
            )
            or 0
        )
        != 30
        or int(
            post_kernel_stats.get("gemma4_router_fused_prefill_hits", 0)
            or 0
        )
        <= 0
        or int(
            post_kernel_stats.get(
                "gemma4_router_fused_prefill_disabled_layers",
                0,
            )
            or 0
        )
        != 0
        or str(
            post_kernel_stats.get("gemma4_router_fused_prefill_error", "")
            or ""
        )
    ):
        candidate_runtime_failures.append("fused_matrix_router")
    if candidate_runtime_failures:
        post_kernel_passed = False
        post_kernel_contract["candidate_runtime_failures"] = list(
            candidate_runtime_failures
        )
    prefill_correctness_gate["post_kernel_gate_recheck"] = post_kernel_contract
    if not post_kernel_passed and (bridge_requested or router_requested):
        rows = max(int(batch) for batch in args.batch_sizes) * 25
        for layer in engine.model.layers:
            bridge_setter = getattr(
                layer,
                "set_gemma4_prefill_attn_moe_bridge_runtime",
                None,
            )
            if callable(bridge_setter):
                bridge_setter(rows, False)
            router_setter = getattr(
                layer.mlp.gate,
                "set_fused_prefill_runtime",
                None,
            )
            if callable(router_setter):
                router_setter(rows, False)

        rollback_reason = {
            "token_contract": {
                "all_finite": bool(post_kernel_contract.get("all_finite")),
                "all_exact": bool(post_kernel_contract.get("all_exact")),
                "all_reference_exact": bool(
                    post_kernel_contract.get("all_reference_exact")
                ),
            },
            "runtime_failures": candidate_runtime_failures,
        }
        if bridge_requested:
            prefill_bridge_gate["pre_rollback_decision"] = (
                "APPLY_FUSED_ATTN_MOE_BRIDGE"
            )
            prefill_bridge_gate["decision"] = "ROLLBACK_POST_KERNEL_CONTRACT"
            prefill_bridge_gate["selected"] = "sequential"
            prefill_bridge_gate["selected_layers"] = 0
            prefill_bridge_gate["post_contract_rollback"] = rollback_reason
        if router_requested:
            prefill_router_gate["pre_rollback_decision"] = (
                "APPLY_FUSED_MATRIX_ROUTER"
            )
            prefill_router_gate["decision"] = "ROLLBACK_POST_KERNEL_CONTRACT"
            prefill_router_gate["selected"] = "cublas_plus_topk"
            prefill_router_gate["selected_layers"] = 0
            prefill_router_gate["post_contract_rollback"] = rollback_reason

        rollback_contract = run_megagemm_first_token_contract(
            engine,
            gate_prompts,
            reference_tokens=oracle_tokens,
            raise_on_failure=False,
        )
        prefill_correctness_gate["post_kernel_candidate_rollback"] = (
            rollback_contract
        )
        post_kernel_passed = bool(
            rollback_contract.get("all_finite")
            and rollback_contract.get("all_exact")
            and rollback_contract.get("all_reference_exact")
        )
        print(
            "MEGAGEMM_PREFILL_CANDIDATE_ROLLBACK "
            + json.dumps(
                {
                    "bridge_requested": bridge_requested,
                    "router_requested": router_requested,
                    "runtime_failures": candidate_runtime_failures,
                    "passed": post_kernel_passed,
                },
                sort_keys=True,
            )
        )
        if post_kernel_passed:
            post_kernel_contract = rollback_contract
    if not post_kernel_passed:
        recovery = run_gemma4_prefill_correctness_gate(engine, gate_prompts)
        if list(recovery["oracle"]["tokens"]) != oracle_tokens:
            raise RuntimeError(
                "Gemma4 sequential prefill oracle changed during one model load: "
                f"initial={oracle_tokens}, recovery={recovery['oracle']['tokens']}"
            )
        prefill_correctness_gate["post_kernel_gate_recovery"] = recovery
        for key in (
            "decision",
            "selected",
            "selected_runtime",
            "selected_contract",
        ):
            prefill_correctness_gate[key] = recovery[key]
        first_token_contract = recovery["selected_contract"]
    else:
        prefill_correctness_gate["selected_contract"] = post_kernel_contract
        first_token_contract = post_kernel_contract
    print(
        "MEGAGEMM_POST_KERNEL_PREFILL_CONTRACT "
        + json.dumps(first_token_contract, sort_keys=True)
    )

    prefill_graph_enabled = env_flag("MEGAGEMM_PREFILL_CUDA_GRAPHS")
    cases: dict[str, Any] = {}
    for batch_size in args.batch_sizes:
        print(f"\n== MegaGemm batch={batch_size} compile/capture (excluded) ==")
        capture_run = run_megagemm_request(
            engine, prompt_pool[:batch_size], args.max_tokens
        )
        capture_first_token_comparison = token_matrix_comparison(
            [[token] for token in oracle_tokens[:batch_size]],
            [[int(row[0])] for row in capture_run["token_ids"]],
        )
        capture_run["first_token_vs_prefill_oracle"] = (
            capture_first_token_comparison
        )
        if not capture_first_token_comparison["exact"]:
            raise RuntimeError(
                "MegaGemm measured-path prefill changed the oracle first token: "
                + json.dumps(capture_first_token_comparison, sort_keys=True)
            )
        print(
            f"capture total={capture_run['total_ms']:.1f}ms "
            f"total={capture_run['output_tok_s_total']:.1f} tok/s"
        )

        prefill_graph_gate: dict[str, Any] = {"required": False}
        measured_prefill_replays = 0
        if batch_size == 16 and prefill_graph_enabled:
            warmup_graph = dict(capture_run.get("prefill_cuda_graphs") or {})
            if (
                not bool(warmup_graph.get("enabled"))
                or int(warmup_graph.get("warmups", 0) or 0) != 1
                or int(warmup_graph.get("captures", 0) or 0) != 0
                or int(warmup_graph.get("capture_body_warmups", 0) or 0) != 0
                or int(warmup_graph.get("failures", 0) or 0) != 0
            ):
                raise RuntimeError(
                    "Gemma4 B16 prefill graph eager warmup failed; "
                    f"stopping before vLLM install: {warmup_graph}"
                )

            # The v99 gate only compared the first generated token. That token
            # can remain unchanged even when prefill K/V differs; the bad state
            # first affected greedy selection at token 2. Exercise enough of
            # decode here to validate that the captured prefill state is usable.
            prefill_graph_probe_tokens = min(3, max(1, int(args.max_tokens)))
            eager_probe_tokens = [
                row[:prefill_graph_probe_tokens]
                for row in capture_run["token_ids"]
            ]
            print(
                "== MegaGemm batch=16 prefill graph capture "
                f"(excluded, {prefill_graph_probe_tokens} output tokens) =="
            )
            capture_probe = run_megagemm_request(
                engine,
                prompt_pool[:batch_size],
                prefill_graph_probe_tokens,
            )
            capture_comparison = token_matrix_comparison(
                eager_probe_tokens,
                capture_probe["token_ids"],
            )
            capture_graph = dict(capture_probe.get("prefill_cuda_graphs") or {})
            capture_decode_graph = dict(
                capture_probe.get("decode_cuda_graphs") or {}
            )
            capture_runtime = engine.model.decode_runtime_stats()
            graph_route_pack_layers = int(
                capture_runtime.get(
                    "qwen3_moe_segmented_prefill_graph_route_pack_layers",
                    0,
                )
                or 0
            )
            if not capture_comparison["exact"]:
                raise RuntimeError(
                    "Gemma4 B16 prefill graph capture changed greedy output; "
                    "stopping before vLLM install: "
                    + json.dumps(capture_comparison, sort_keys=True)
                )
            if (
                int(capture_graph.get("captures", 0) or 0) != 1
                or int(capture_graph.get("capture_body_warmups", 0) or 0) != 2
                or int(capture_graph.get("capture_replays", 0) or 0) != 1
                or int(capture_graph.get("replays", 0) or 0) != 0
                or int(capture_graph.get("failures", 0) or 0) != 0
                or int(capture_graph.get("buckets", 0) or 0) != 1
                or list(capture_graph.get("bucket_kinds") or []) != ["padded"]
                or list(capture_graph.get("kv_write_modes") or [])
                != ["external_after_replay"]
                or int(capture_graph.get("deferred_kv_layers", 0) or 0) != 30
                or int(capture_graph.get("external_kv_write_replays", 0) or 0) != 1
                or int(capture_graph.get("workspace_tensors", 0) or 0) != 30
                or int(capture_graph.get("workspace_bytes", 0) or 0)
                != 30 * 3200 * 2816 * 4
                or graph_route_pack_layers != 30
            ):
                raise RuntimeError(
                    "Gemma4 B16 prefill graph capture/replay failed; "
                    "stopping before vLLM install: "
                    f"graph={capture_graph}, "
                    f"deterministic_route_layers={graph_route_pack_layers}/30"
                )

            print(
                "== MegaGemm batch=16 prefill graph replay gate "
                f"(excluded, {prefill_graph_probe_tokens} output tokens) =="
            )
            replay_probe = run_megagemm_request(
                engine,
                prompt_pool[:batch_size],
                prefill_graph_probe_tokens,
            )
            replay_comparison = token_matrix_comparison(
                eager_probe_tokens,
                replay_probe["token_ids"],
            )
            replay_graph = dict(replay_probe.get("prefill_cuda_graphs") or {})
            replay_decode_graph = dict(
                replay_probe.get("decode_cuda_graphs") or {}
            )
            if not replay_comparison["exact"]:
                raise RuntimeError(
                    "Gemma4 B16 prefill graph replay changed greedy output; "
                    "stopping before vLLM install: "
                    + json.dumps(replay_comparison, sort_keys=True)
                )
            if (
                int(replay_graph.get("captures", 0) or 0) != 1
                or int(replay_graph.get("capture_body_warmups", 0) or 0) != 2
                or int(replay_graph.get("capture_replays", 0) or 0) != 1
                or int(replay_graph.get("replays", 0) or 0) < 1
                or int(replay_graph.get("failures", 0) or 0) != 0
                or int(replay_graph.get("buckets", 0) or 0) != 1
                or list(replay_graph.get("bucket_kinds") or []) != ["padded"]
                or list(replay_graph.get("kv_write_modes") or [])
                != ["external_after_replay"]
                or int(replay_graph.get("deferred_kv_layers", 0) or 0) != 30
                or int(replay_graph.get("external_kv_write_replays", 0) or 0)
                != int(replay_graph.get("capture_replays", 0) or 0)
                + int(replay_graph.get("replays", 0) or 0)
                or int(replay_graph.get("workspace_tensors", 0) or 0) != 30
                or int(replay_graph.get("workspace_bytes", 0) or 0)
                != 30 * 3200 * 2816 * 4
            ):
                raise RuntimeError(
                    "Gemma4 B16 prefill graph did not replay; "
                    f"stopping before vLLM install: {replay_graph}"
                )
            if (
                int(replay_decode_graph.get("replays", 0) or 0) < 1
                or int(replay_decode_graph.get("failures", 0) or 0) != 0
                or int(replay_decode_graph.get("physical_rebinds", 0) or 0) != 0
            ):
                raise RuntimeError(
                    "Gemma4 B16 shared decode graph did not reuse its physical KV "
                    "binding; stopping before vLLM install: "
                    f"capture={capture_decode_graph}, replay={replay_decode_graph}"
                )
            measured_prefill_replays = int(replay_graph.get("replays", 0) or 0)
            prefill_graph_gate = {
                "required": True,
                "warmup": warmup_graph,
                "capture": capture_graph,
                "replay": replay_graph,
                "capture_decode": capture_decode_graph,
                "replay_decode": replay_decode_graph,
                "deterministic_route_layers": graph_route_pack_layers,
                "probe_output_tokens": prefill_graph_probe_tokens,
                "capture_tokens_vs_eager": capture_comparison,
                "replay_tokens_vs_eager": replay_comparison,
                "measured_tokens_vs_eager": [],
            }
            print(
                "MEGAGEMM_B16_PREFILL_GRAPH_GATE "
                + json.dumps(prefill_graph_gate, sort_keys=True)
            )

        samples = []
        token_reference: list[list[int]] | None = None
        for repeat in range(args.repeats):
            row = run_megagemm_request(
                engine, prompt_pool[:batch_size], args.max_tokens
            )
            graph = row.get("decode_cuda_graphs") or {}
            if (
                not graph.get("enabled")
                or int(graph.get("failures", 0) or 0) != 0
                or int(graph.get("replays", 0) or 0) <= 0
                or int(graph.get("physical_rebinds", 0) or 0) != 0
            ):
                raise RuntimeError(
                    f"MegaGemm decode graph failed at batch={batch_size}: {graph}"
                )
            if batch_size == 16 and prefill_graph_enabled:
                prefill_graph = row.get("prefill_cuda_graphs") or {}
                current_prefill_replays = int(
                    prefill_graph.get("replays", 0) or 0
                )
                if (
                    int(prefill_graph.get("captures", 0) or 0) != 1
                    or int(
                        prefill_graph.get("capture_body_warmups", 0) or 0
                    )
                    != 2
                    or int(prefill_graph.get("capture_replays", 0) or 0) != 1
                    or int(prefill_graph.get("failures", 0) or 0) != 0
                    or list(prefill_graph.get("bucket_kinds") or []) != ["padded"]
                    or list(prefill_graph.get("kv_write_modes") or [])
                    != ["external_after_replay"]
                    or int(prefill_graph.get("deferred_kv_layers", 0) or 0) != 30
                    or int(prefill_graph.get("external_kv_write_replays", 0) or 0)
                    != int(prefill_graph.get("capture_replays", 0) or 0)
                    + current_prefill_replays
                    or int(prefill_graph.get("workspace_tensors", 0) or 0) != 30
                    or current_prefill_replays <= measured_prefill_replays
                ):
                    raise RuntimeError(
                        "Gemma4 B16 measured prefill did not use CUDA graph replay; "
                        f"stopping before vLLM install: {prefill_graph}"
                    )
                measured_prefill_replays = current_prefill_replays
                eager_comparison = token_matrix_comparison(
                    capture_run["token_ids"], row["token_ids"]
                )
                prefill_graph_gate["measured_tokens_vs_eager"].append(
                    eager_comparison
                )
                if not eager_comparison["exact"]:
                    raise RuntimeError(
                        "Gemma4 B16 measured prefill graph changed the full "
                        "greedy token matrix; stopping before vLLM install: "
                        + json.dumps(eager_comparison, sort_keys=True)
                    )
            if token_reference is None:
                token_reference = row["token_ids"]
            comparison = token_matrix_comparison(token_reference, row["token_ids"])
            if not comparison["exact"]:
                failure = {
                    "batch_size": batch_size,
                    "repeat": repeat + 1,
                    "comparison": comparison,
                }
                write_partial_checkpoint(
                    args,
                    "megagemm",
                    cases,
                    prefill_correctness_gate=prefill_correctness_gate,
                    compact_kernel_gate=compact_gate,
                    lm_head_kernel_gate=lm_head_gate,
                    scheduler_burst_gate=scheduler_burst_gate,
                    attention_kernel_gate=attention_kernel_gate,
                    next_attn_norm_kernel_gate=next_attn_norm_gate,
                    prefill_kernel_gate=prefill_gate,
                    determinism_contract=determinism_contract,
                    first_token_contract=first_token_contract,
                    token_stability_failure=failure,
                )
                raise RuntimeError(
                    "MegaGemm greedy tokens changed across identical repeats: "
                    + json.dumps(failure, sort_keys=True)
                )
            prefill_ms = float(row.get("scheduler_prefill_ms") or 0.0)
            decode_ms = float(row.get("scheduler_decode_ms") or 0.0)
            if prefill_ms <= 0.0 or decode_ms <= 0.0:
                raise RuntimeError(
                    f"MegaGemm scheduler phase timing is invalid at batch={batch_size}: "
                    f"prefill={prefill_ms}ms decode={decode_ms}ms"
                )
            decode_tokens = batch_size * (args.max_tokens - 1)
            row.update({
                "repeat": repeat + 1,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "decode_tokens": decode_tokens,
                "decode_tok_s": decode_tokens / (decode_ms / 1000.0),
                "decode_measurement_method": "scheduler_phase_wall_time",
                "token_stability_vs_first_repeat": comparison,
            })
            samples.append(row)
            print(
                f"MegaGemm B={batch_size} repeat={repeat + 1}/{args.repeats} "
                f"total={row['total_ms']:.1f}ms decode={decode_ms:.1f}ms "
                f"decode={row['decode_tok_s']:.1f} tok/s"
            )
        cases[str(batch_size)] = {
            "batch_size": batch_size,
            "capture_run_excluded": capture_run,
            "prefill_graph_gate": prefill_graph_gate,
            "samples": samples,
            "summary": summarize(samples),
        }
        write_partial_checkpoint(
            args,
            "megagemm",
            cases,
            prefill_correctness_gate=prefill_correctness_gate,
            compact_kernel_gate=compact_gate,
            lm_head_kernel_gate=lm_head_gate,
            scheduler_burst_gate=scheduler_burst_gate,
            attention_kernel_gate=attention_kernel_gate,
            next_attn_norm_kernel_gate=next_attn_norm_gate,
            prefill_kernel_gate=prefill_gate,
            determinism_contract=determinism_contract,
            first_token_contract=first_token_contract,
            runtime_validation={"status": "pending"},
        )
        runtime = validate_megagemm_runtime(
            engine,
            batch_size,
            compact_gate,
            lm_head_gate,
            scheduler_burst_gate,
            attention_kernel_gate,
            next_attn_norm_gate,
            prefill_gate,
        )
        cases[str(batch_size)]["runtime_gate"] = runtime
        print("MEGAGEMM_BATCH_GATE " + json.dumps({
            "batch_size": batch_size,
            "decode_graph": samples[-1].get("decode_cuda_graphs") or {},
            "prefill_graph": samples[-1].get("prefill_cuda_graphs") or {},
            "prefill_graph_gate": prefill_graph_gate,
            "runtime": runtime,
            "capture_run_excluded": True,
            "token_drift_is_diagnostic": False,
        }, sort_keys=True))
        if batch_size == 16 and not prefill_graph_enabled:
            print("== MegaGemm B=16 excluded post-measurement prefill profile ==")
            prefill_profile = run_excluded_megagemm_prefill_profile(
                engine,
                prompt_pool[:batch_size],
                token_reference or [],
            )
            cases[str(batch_size)]["prefill_profile_excluded"] = prefill_profile
            print(
                "MEGAGEMM_PREFILL_PROFILE_EXCLUDED "
                + json.dumps(prefill_profile, sort_keys=True)
            )
        write_partial_checkpoint(
            args,
            "megagemm",
            cases,
            prefill_correctness_gate=prefill_correctness_gate,
            compact_kernel_gate=compact_gate,
            lm_head_kernel_gate=lm_head_gate,
            scheduler_burst_gate=scheduler_burst_gate,
            attention_kernel_gate=attention_kernel_gate,
            next_attn_norm_kernel_gate=next_attn_norm_gate,
            prefill_kernel_gate=prefill_gate,
            determinism_contract=determinism_contract,
            first_token_contract=first_token_contract,
        )
    profile_breakdown: dict[str, Any] | None = None
    if args.profile_breakdown:
        profile_batch = max(int(batch) for batch in args.batch_sizes)
        print(f"\n== MegaGemm B={profile_batch} post-measurement profile ==")
        profile_breakdown = engine.profile_decode_breakdown(
            prompt_pool[:profile_batch],
            max_new_tokens=args.max_tokens,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            ignore_eos=True,
        )

    return {
        "backend": "megagemm",
        "determinism_contract": determinism_contract,
        "prefill_correctness_gate": prefill_correctness_gate,
        "first_token_contract": first_token_contract,
        "lm_head_kernel_gate": lm_head_gate,
        "scheduler_burst_gate": scheduler_burst_gate,
        "attention_kernel_gate": attention_kernel_gate,
        "next_attn_norm_kernel_gate": next_attn_norm_gate,
        "compact_kernel_gate": compact_gate,
        "prefill_kernel_gate": prefill_gate,
        "profile_breakdown": profile_breakdown,
        "cases": cases,
    }


def run_vllm_request(
    llm,
    prompts: list[list[int]],
    max_tokens: int,
    *,
    allowed_token_id: int | None = None,
) -> dict[str, Any]:
    batch_size = len(prompts)
    from vllm import SamplingParams

    sampling_kwargs: dict[str, Any] = dict(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    if allowed_token_id is not None:
        if int(allowed_token_id) < 0:
            raise ValueError("allowed_token_id must be non-negative")
        sampling_kwargs["allowed_token_ids"] = [int(allowed_token_id)]
    params = SamplingParams(**sampling_kwargs)
    token_prompts = [
        {"prompt_token_ids": [int(token) for token in row]}
        for row in prompts
    ]
    sync_cuda()
    started = time.perf_counter()
    try:
        outputs = llm.generate(token_prompts, params, use_tqdm=False)
    except TypeError:
        outputs = llm.generate(token_prompts, params)
    sync_cuda()
    total_ms = (time.perf_counter() - started) * 1000.0
    matrix, output_alignment = align_vllm_outputs_to_prompts(outputs, prompts)
    if len(matrix) != batch_size or any(len(row) != max_tokens for row in matrix):
        raise RuntimeError(
            f"vLLM output shape mismatch: rows={len(matrix)} "
            f"lengths={[len(row) for row in matrix]}"
        )
    if allowed_token_id is not None and any(
        token != int(allowed_token_id) for row in matrix for token in row
    ):
        raise RuntimeError(
            "vLLM allowed-token continuation contract failed: output contained "
            f"a token other than {int(allowed_token_id)}"
        )
    request_metrics = [extract_vllm_request_metrics(request) for request in outputs]
    required = ("arrival_time", "first_token_time", "finished_time")
    if request_metrics and all(
        all(name in metrics for name in required) for metrics in request_metrics
    ):
        aggregate_metrics = {
            "arrival_time": min(row["arrival_time"] for row in request_metrics),
            "first_token_time": max(row["first_token_time"] for row in request_metrics),
            "finished_time": max(row["finished_time"] for row in request_metrics),
        }
    else:
        aggregate_metrics = {}
    phase = validated_vllm_phase_span(aggregate_metrics, total_ms)
    decode_ms = (
        phase["decode_ms"]
        if phase["valid"] and max_tokens > 1 and phase["decode_ms"] > 0.0
        else None
    )
    decode_tokens = batch_size * (max_tokens - 1) if decode_ms is not None else None
    return {
        "total_ms": total_ms,
        "output_tokens": batch_size * max_tokens,
        "output_tok_s_total": batch_size * max_tokens / (total_ms / 1000.0),
        "prefill_ms": phase["prefill_ms"] if phase["valid"] else None,
        "prefill_measurement_method": (
            "vllm_batch_request_metrics_ttft"
            if phase["valid"]
            else str(phase["status"])
        ),
        "decode_ms": decode_ms,
        "decode_tokens": decode_tokens,
        "decode_tok_s": (
            decode_tokens / (decode_ms / 1000.0)
            if decode_tokens is not None and decode_ms is not None
            else None
        ),
        "decode_measurement_method": (
            "vllm_batch_request_metrics_first_to_finished"
            if decode_ms is not None
            else str(phase["status"])
        ),
        "phase_metrics_status": phase["status"],
        "phase_metrics_reason": phase["reason"],
        "request_metrics": request_metrics,
        "request_metrics_aggregate": aggregate_metrics,
        "request_metric_total_ms": phase["metric_total_ms"],
        "request_metric_wall_error_ms": phase["wall_error_ms"],
        "request_metric_wall_error_ratio": phase["wall_error_ratio"],
        "output_alignment": output_alignment,
        "token_ids": matrix,
        "allowed_token_id": (
            None if allowed_token_id is None else int(allowed_token_id)
        ),
    }


def run_vllm(
    args: argparse.Namespace,
    prompt_pool: list[list[int]],
) -> dict[str, Any]:
    args.max_batch_size = max(args.batch_sizes)
    llm, runtime, version, kwargs = make_vllm(args)
    print(f"vLLM version: {version}")
    print(f"vLLM kwargs: {kwargs}")
    cases: dict[str, Any] = {}
    for batch_size in args.batch_sizes:
        print(
            f"\n== vLLM batch={batch_size} adaptive warmup "
            f"({VLLM_MIN_WARMUPS}-{VLLM_MAX_WARMUPS} identical requests) =="
        )
        warmups: list[dict[str, Any]] = []
        warmup_stability: dict[str, Any] = {}
        for warmup_index in range(VLLM_MAX_WARMUPS):
            warmup = run_vllm_request(
                llm, prompt_pool[:batch_size], args.max_tokens
            )
            warmup["warmup"] = warmup_index + 1
            warmups.append(warmup)
            warmup_stability = evaluate_vllm_warmup_stability(warmups)
            ratio = warmup_stability.get("last_pair_total_ratio")
            ratio_text = "n/a" if ratio is None else f"{float(ratio):.3f}x"
            print(
                f"vLLM B={batch_size} warmup={warmup_index + 1}/"
                f"{VLLM_MAX_WARMUPS} total={warmup['total_ms']:.1f}ms "
                f"last_pair={ratio_text} "
                f"status={warmup_stability['acceptance_reason']}"
            )
            if warmup_stability["stable"]:
                break

        case_key = str(batch_size)
        cases[case_key] = {
            "batch_size": batch_size,
            "warmup": warmups[-1],
            "warmups": warmups,
            "warmup_stability": warmup_stability,
            "samples": [],
            "summary": summarize([]),
        }
        write_partial_checkpoint(
            args,
            "vllm",
            cases,
            version=version,
            cuda_runtime=runtime,
            llm_kwargs=kwargs,
        )
        print(
            "VLLM_WARMUP_ACCEPTANCE "
            + json.dumps(warmup_stability, sort_keys=True)
        )
        if not warmup_stability.get("accepted"):
            raise RuntimeError(
                f"vLLM batch={batch_size} warmup was not safe to measure after "
                f"{VLLM_MAX_WARMUPS} identical requests: "
                + json.dumps(warmup_stability, sort_keys=True)
            )

        # Do not subtract a max_tokens=1 request from this workload: vLLM may
        # choose a different execution path. Use this same request's metrics.
        samples: list[dict[str, Any]] = []
        token_reference = warmups[-1]["token_ids"]
        for repeat in range(args.repeats):
            row = run_vllm_request(llm, prompt_pool[:batch_size], args.max_tokens)
            row["repeat"] = repeat + 1
            comparison = token_matrix_comparison(token_reference, row["token_ids"])
            row["token_stability_vs_final_warmup"] = comparison
            samples.append(row)
            cases[case_key]["samples"] = samples
            if not comparison["exact"]:
                failure = {
                    "batch_size": batch_size,
                    "repeat": repeat + 1,
                    "reference": "final_accepted_warmup",
                    "comparison": comparison,
                }
                cases[case_key]["token_stability_failure"] = failure
                write_partial_checkpoint(
                    args,
                    "vllm",
                    cases,
                    version=version,
                    cuda_runtime=runtime,
                    llm_kwargs=kwargs,
                    token_stability_failure=failure,
                )
                raise RuntimeError(
                    "vLLM greedy tokens changed after accepted warmup: "
                    + json.dumps(failure, sort_keys=True)
                )
            if row["decode_ms"] is None:
                print(
                    f"vLLM B={batch_size} repeat={repeat + 1}/{args.repeats} "
                    f"total={row['total_ms']:.1f}ms phase=n/a "
                    f"status={row['phase_metrics_status']}"
                )
            else:
                print(
                    f"vLLM B={batch_size} repeat={repeat + 1}/{args.repeats} "
                    f"total={row['total_ms']:.1f}ms "
                    f"prefill={row['prefill_ms']:.1f}ms "
                    f"decode={row['decode_ms']:.1f}ms "
                    f"decode={row['decode_tok_s']:.1f} tok/s"
                )
            print(
                "VLLM_BATCH_REQUEST_METRICS "
                + json.dumps(
                    {
                        "aggregate": row["request_metrics_aggregate"],
                        "status": row["phase_metrics_status"],
                        "reason": row["phase_metrics_reason"],
                        "metric_total_ms": row["request_metric_total_ms"],
                        "wall_total_ms": row["total_ms"],
                        "wall_error_ratio": row["request_metric_wall_error_ratio"],
                    },
                    sort_keys=True,
                )
            )
        cases[case_key]["samples"] = samples
        cases[case_key]["summary"] = summarize(samples)
        cases[case_key]["token_stability"] = {
            "exact": True,
            "reference": "final_accepted_warmup",
            "checked_measured_samples": len(samples),
        }
        write_partial_checkpoint(
            args,
            "vllm",
            cases,
            version=version,
            cuda_runtime=runtime,
            llm_kwargs=kwargs,
        )
    return {
        "backend": "vllm",
        "version": version,
        "cuda_runtime": runtime,
        "llm_kwargs": kwargs,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("megagemm", "vllm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-sizes", default="2,4,8,16")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the proven stable deterministic baseline (default: enabled)",
    )
    parser.add_argument(
        "--stop-after-no-decode-promotion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "write the kernel-gate result and stop before full measurements "
            "when no new decode candidate is promoted"
        ),
    )
    parser.add_argument(
        "--prefill-finite-trace-only",
        action="store_true",
        help=(
            "load MegaGemm, trace one safe batched prefill to its first "
            "nonfinite tensor, write JSON, and skip every benchmark sweep"
        ),
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--profile-breakdown",
        action="store_true",
        help="profile one MegaGemm request after all latency samples",
    )
    parser.add_argument(
        "--prompt-token-ids-json",
        default="",
        help="shared token-ID manifest used unchanged by both backend processes",
    )
    parser.add_argument(
        "--prompt",
        default="Write a compact Python Fibonacci function and explain its time complexity.",
    )
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    args.batch_sizes = parse_batch_sizes(args.batch_sizes)

    if args.backend == "megagemm" and args.deterministic:
        # Set before the first CUDA runtime query and before any cuBLAS handle exists.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.dtype != "bf16":
        raise ValueError("the promoted Gemma4 batch policy is BF16-only")
    if args.max_tokens < 2:
        raise ValueError("max_tokens must be >= 2")
    if args.repeats < 1:
        raise ValueError("repeats must be >= 1")

    manifest_path = (
        Path(args.prompt_token_ids_json)
        if args.prompt_token_ids_json
        else None
    )
    prompt_pool, prompt_contract = load_or_create_prompt_token_ids(
        args.model,
        args.prompt,
        max(args.batch_sizes),
        expected_tokens=25,
        manifest_path=manifest_path,
    )
    prompt_tokens = int(prompt_contract["tokens_per_row"])
    if prompt_tokens + args.max_tokens > args.max_seq_len:
        raise ValueError("prompt plus output exceeds max_seq_len")
    print("Gemma 4 MoE batch throughput matrix")
    print(f"  backend: {args.backend}")
    print(f"  model: {args.model}")
    print(f"  dtype: {args.dtype}")
    print(f"  batch_sizes: {args.batch_sizes}")
    print(f"  prompt_tokens_per_request: {prompt_tokens}")
    print(f"  distinct_equal_length_prompts: {len(prompt_pool)}")
    print(f"  prompt_token_contract: {json.dumps(prompt_contract, sort_keys=True)}")
    print(f"  output_tokens_per_request: {args.max_tokens}")
    print(f"  repeats: {args.repeats}")
    print(f"  deterministic: {args.deterministic if args.backend == 'megagemm' else 'n/a'}")
    print(f"  gpu: {gpu_snapshot()}")

    result = (
        run_megagemm(args, prompt_pool)
        if args.backend == "megagemm"
        else run_vllm(args, prompt_pool)
    )
    result.update({
        "model": args.model,
        "dtype": args.dtype,
        "batch_sizes": args.batch_sizes,
        "workload": "pretokenized_equal_length_distinct_prompts",
        "prompt_contract": prompt_contract,
        "prompt_tokens": prompt_tokens,
        "max_tokens": args.max_tokens,
        "gpu": gpu_snapshot(),
    })
    result.setdefault("status", "complete")
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("\n== MATRIX SUMMARY ==")
    if result.get("trace_only"):
        print(
            "prefill finite trace: "
            + json.dumps(result["prefill_finite_trace"], sort_keys=True)
        )
    elif result.get("early_stop_no_decode_promotion"):
        print(
            "gate rejected; full MegaGemm matrix intentionally skipped: "
            + json.dumps(
                result["early_stop_no_decode_promotion"],
                sort_keys=True,
            )
        )
    else:
        for batch_size in args.batch_sizes:
            summary = result["cases"][str(batch_size)]["summary"]
            print(f"B={batch_size} " + json.dumps(summary, sort_keys=True))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

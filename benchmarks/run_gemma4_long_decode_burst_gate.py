"""Loaded-checkpoint gate for Gemma4 B16 long-context decode token bursts.

The gate keeps prefill deterministic, loads the checkpoint once, and alternates
the current one-token graph step with an eight-token GPU-feedback burst.  Token
feedback stays outside the captured graph because the persistent chain failed
the exact-token contract at this long-context shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from run_gemma4_long_context_vs_vllm import (  # noqa: E402
    DEFAULT_PROMPT,
    force_segmented_long_prefill,
    load_or_create_prompt_manifest,
    megagemm_deterministic_moe_contract,
)
from run_gemma4_moe_batch_vs_vllm import (  # noqa: E402
    run_megagemm_request,
    token_matrix_comparison,
)
from run_gemma4_moe_vs_vllm import dtype_from_name, gpu_snapshot  # noqa: E402


HARNESS_REV = "gemma4-long-decode-burst-gate-v2-gpu-feedback"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def token_matrix_contract(rows: list[list[int]]) -> dict[str, Any]:
    normalized = [[int(token) for token in row] for row in rows]
    lengths = sorted({len(row) for row in normalized})
    encoded = json.dumps(
        normalized,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return {
        "rows": len(normalized),
        "row_lengths": lengths,
        "total_tokens": sum(len(row) for row in normalized),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def compact_token_check(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact": bool(comparison.get("exact", False)),
        "rows": int(comparison.get("rows", 0) or 0),
        "mismatched_rows": int(comparison.get("mismatched_rows", 0) or 0),
        "min_common_prefix_tokens": int(
            comparison.get("min_common_prefix_tokens", 0) or 0
        ),
        "first_mismatch": comparison.get("first_mismatch"),
    }


def graph_contract(
    stats: dict[str, Any],
    *,
    candidate: bool,
    max_tokens: int,
    burst_size: int,
) -> dict[str, Any]:
    decode_steps = max(0, int(max_tokens) - 1)
    expected_bursts = math.ceil(decode_steps / int(burst_size))
    expected_feedback_copies = max(0, decode_steps - expected_bursts)
    common = bool(
        stats.get("enabled", False)
        and stats.get("prefer_step", False)
        and stats.get("shape_cache", False)
        and not stats.get("shared_shape_cache", False)
        and int(stats.get("failures", 0) or 0) == 0
        and int(stats.get("replays", 0) or 0) > 0
        and int(stats.get("greedy_token_shape_graphs", 0) or 0) > 0
    )
    if candidate:
        mode = bool(
            stats.get("token_burst_enabled", False)
            and int(stats.get("token_burst_size", 0) or 0) == int(burst_size)
            and int(stats.get("token_burst_steps", 0) or 0) == decode_steps
            and int(stats.get("token_bursts", 0) or 0) == expected_bursts
            and int(stats.get("greedy_token_steps", 0) or 0) == decode_steps
            and int(stats.get("batched_token_host_copies", 0) or 0)
            == expected_bursts
            and not stats.get("persistent_token_feedback_enabled", False)
            and int(stats.get("persistent_token_feedback_steps", 0) or 0) == 0
            and int(stats.get("token_feedback_copies", 0) or 0)
            == expected_feedback_copies
            and int(stats.get("vectorized_input_updates", 0) or 0)
            == expected_bursts
            and int(stats.get("chain_input_updates_skipped", 0) or 0) == 0
        )
    else:
        mode = bool(
            not stats.get("token_burst_enabled", False)
            and not stats.get("persistent_token_feedback_enabled", False)
            and int(stats.get("token_burst_steps", 0) or 0) == 0
            and int(stats.get("token_bursts", 0) or 0) == 0
        )
    return {
        "exact": bool(common and mode),
        "expected_decode_steps": decode_steps,
        "expected_bursts": expected_bursts if candidate else 0,
        "expected_feedback_copies": expected_feedback_copies if candidate else 0,
        "expected_vectorized_input_updates": expected_bursts if candidate else 0,
        "stats": stats,
    }


def softcap_contract(runtime: dict[str, Any], *, candidate: bool) -> dict[str, Any]:
    enabled = bool(runtime.get("gemma4_batch_fused_softcap_argmax_enabled", False))
    available = bool(
        runtime.get("gemma4_batch_fused_softcap_argmax_available", False)
    )
    hits = int(runtime.get("gemma4_batch_fused_softcap_argmax_hits", 0) or 0)
    disabled = bool(
        runtime.get("gemma4_batch_fused_softcap_argmax_disabled", False)
    )
    error = str(runtime.get("gemma4_batch_fused_softcap_argmax_error", "") or "")
    cublas_enabled = bool(runtime.get("gemma4_batch_cublas_lm_head_enabled", False))
    cublas_hits = int(runtime.get("gemma4_batch_cublas_lm_head_hits", 0) or 0)
    exact = bool(
        cublas_enabled
        and cublas_hits > 0
        and (
            enabled and available and hits > 0 and not disabled and not error
            if candidate
            else not enabled and hits == 0 and not disabled and not error
        )
    )
    return {
        "exact": exact,
        "enabled": enabled,
        "available": available,
        "hits": hits,
        "disabled": disabled,
        "error": error,
        "cublas_enabled": cublas_enabled,
        "cublas_hits": cublas_hits,
    }


def summarize_case(samples: list[dict[str, Any]]) -> dict[str, Any]:
    decode_ms = [float(sample["decode_ms"]) for sample in samples]
    total_ms = [float(sample["total_ms"]) for sample in samples]
    return {
        "decode_ms": decode_ms,
        "decode_ms_median": float(statistics.median(decode_ms)),
        "total_ms": total_ms,
        "total_ms_median": float(statistics.median(total_ms)),
        "stability_ratio": (
            max(decode_ms) / min(decode_ms)
            if decode_ms and min(decode_ms) > 0.0
            else None
        ),
        "tokens_exact": all(sample["token_check"]["exact"] for sample in samples),
        "graph_contract_exact": all(
            sample["graph_contract"]["exact"] for sample in samples
        ),
        "softcap_contract_exact": all(
            sample["softcap_contract"]["exact"] for sample in samples
        ),
        "deterministic_moe_contract_exact": all(
            sample["deterministic_moe_contract"]["exact"] for sample in samples
        ),
        "samples": samples,
    }


def decide_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_speedup: float,
    minimum_savings_ms: float,
    maximum_stability_ratio: float,
) -> dict[str, Any]:
    baseline_ms = float(baseline["decode_ms_median"])
    candidate_ms = float(candidate["decode_ms_median"])
    speedup = baseline_ms / candidate_ms if candidate_ms > 0.0 else 0.0
    savings_ms = baseline_ms - candidate_ms
    baseline_valid = bool(
        baseline["tokens_exact"]
        and baseline["graph_contract_exact"]
        and baseline["softcap_contract_exact"]
        and baseline["deterministic_moe_contract_exact"]
        and float(baseline.get("stability_ratio") or 999.0)
        <= maximum_stability_ratio
    )
    candidate_valid = bool(
        candidate["tokens_exact"]
        and candidate["graph_contract_exact"]
        and candidate["softcap_contract_exact"]
        and candidate["deterministic_moe_contract_exact"]
        and float(candidate.get("stability_ratio") or 999.0)
        <= maximum_stability_ratio
    )
    apply_change = bool(
        baseline_valid
        and candidate_valid
        and speedup >= minimum_speedup
        and savings_ms >= minimum_savings_ms
    )
    return {
        "decision": (
            "APPLY_LONG_DECODE_GPU_FEEDBACK_BURST"
            if apply_change
            else "KEEP_ONE_STEP"
        ),
        "apply_change": apply_change,
        "baseline_valid": baseline_valid,
        "candidate_valid": candidate_valid,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "speedup": speedup,
        "savings_ms": savings_ms,
        "minimum_speedup": float(minimum_speedup),
        "minimum_savings_ms": float(minimum_savings_ms),
        "maximum_stability_ratio": float(maximum_stability_ratio),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="bf16", choices=("bf16",))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=2112)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--minimum-savings-ms", type=float, default=20.0)
    parser.add_argument("--maximum-stability-ratio", type=float, default=1.03)
    parser.add_argument("--prompt-token-ids-json", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if (
        args.batch_size != 16
        or args.context != 2048
        or args.max_tokens != 64
        or args.burst_size != 8
    ):
        raise SystemExit("This gate is fixed to B16, C2048, 64 outputs, and burst 8")
    if args.context + args.max_tokens > args.max_seq_len:
        raise SystemExit("context plus output exceeds max_seq_len")
    if args.repeats < 2:
        raise SystemExit("repeats must be at least 2")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    os.environ["MEGAGEMM_DECODE_CUDA_GRAPHS"] = "1"
    os.environ["MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP"] = "1"
    os.environ["MEGAGEMM_DECODE_CUDA_GRAPHS_SHAPE_CACHE"] = "1"
    os.environ["MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE"] = "0"
    os.environ["MEGAGEMM_DECODE_CUDA_GRAPHS_STABLE_MAX_BLOCKS"] = "1"
    os.environ["MEGAGEMM_MULTI_STEP_BURST_BATCH"] = str(args.burst_size)
    os.environ["MEGAGEMM_GEMMA4_BATCH_CUBLAS_LM_HEAD"] = "1"
    os.environ["MEGAGEMM_GEMMA4_MOE_LONG_PADDED_BMM_PREFILL"] = "0"

    prompts, manifest = load_or_create_prompt_manifest(
        args.model,
        DEFAULT_PROMPT,
        [args.context],
        args.batch_size,
        Path(args.prompt_token_ids_json),
    )
    prompt_rows = prompts[args.context][: args.batch_size]

    print("Gemma4 B16 long decode GPU-feedback burst gate")
    print("  harness_rev:", HARNESS_REV)
    print("  gpu:", torch.cuda.get_device_name(0))
    print(
        "  shape:",
        f"batch={args.batch_size} context={args.context} output={args.max_tokens}",
        f"burst={args.burst_size} dtype={args.dtype}",
    )
    print("  checkpoint_loads: 1")
    print("  vllm_install: disabled")

    from megagemm.engine import InferenceEngine
    import megagemm.models.llama as llama_model

    engine = InferenceEngine(
        args.model,
        device="cuda",
        dtype=dtype_from_name(args.dtype),
        max_seq_len=args.max_seq_len,
        max_batch_size=args.batch_size,
        deterministic=True,
    )
    fallback = force_segmented_long_prefill(
        engine.model,
        "long decode burst gate requires deterministic segmented prefill",
    )
    if int(fallback.get("disabled_layers", 0) or 0) != 30:
        raise RuntimeError(f"expected 30 deterministic prefill layers: {fallback}")

    baseline_name = "one_step"
    softcap_name = "one_step_fused_softcap"
    candidate_name = "burst8_gpu_feedback"
    case_names = (baseline_name, softcap_name, candidate_name)

    def configure(case_name: str) -> None:
        use_burst = case_name == candidate_name
        use_softcap = case_name != baseline_name
        llama_model._GEMMA4_BATCH_CUBLAS_LM_HEAD = True
        llama_model._GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX = use_softcap
        os.environ["MEGAGEMM_DECODE_GRAPH_TOKEN_BURST"] = (
            "1" if use_burst else "0"
        )
        os.environ["MEGAGEMM_GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX"] = (
            "1" if use_softcap else "0"
        )
        # The v1 long-context gate produced stable but wrong tokens with graph-owned
        # feedback.  Keep the candidate on explicit GPU-to-GPU feedback.
        os.environ["MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK"] = "0"

    def reset_decode_runtime() -> None:
        engine.model._gemma4_batch_cublas_lm_head_hits = 0
        engine.model._gemma4_batch_fused_softcap_argmax_hits = 0
        engine.model._gemma4_batch_fused_softcap_argmax_disable = False
        engine.model._gemma4_batch_fused_softcap_argmax_error = ""

    def run_case(case_name: str, reference: list[list[int]]) -> dict[str, Any]:
        use_burst = case_name == candidate_name
        use_softcap = case_name != baseline_name
        configure(case_name)
        reset_decode_runtime()
        row = run_megagemm_request(engine, prompt_rows, args.max_tokens)
        decode_ms = float(row.get("scheduler_decode_ms") or 0.0)
        if decode_ms <= 0.0:
            raise RuntimeError(f"scheduler decode timing is unavailable: {row}")
        runtime = dict(row.get("decode_runtime") or {})
        graph_stats = dict(row.get("decode_cuda_graphs") or {})
        return {
            "decode_ms": decode_ms,
            "total_ms": float(row["total_ms"]),
            "token_check": compact_token_check(
                token_matrix_comparison(reference, row["token_ids"])
            ),
            "output_contract": token_matrix_contract(row["token_ids"]),
            "graph_contract": graph_contract(
                graph_stats,
                candidate=use_burst,
                max_tokens=args.max_tokens,
                burst_size=args.burst_size,
            ),
            "softcap_contract": softcap_contract(runtime, candidate=use_softcap),
            "deterministic_moe_contract": megagemm_deterministic_moe_contract(
                runtime
            ),
        }

    prime_rows: dict[str, dict[str, Any]] = {}
    for case_name in case_names:
        configure(case_name)
        reset_decode_runtime()
        prime_rows[case_name] = run_megagemm_request(
            engine, prompt_rows, args.max_tokens
        )

    reference = prime_rows[baseline_name]["token_ids"]
    prime_checks = {
        case_name: compact_token_check(
            token_matrix_comparison(reference, prime_rows[case_name]["token_ids"])
        )
        for case_name in case_names
    }
    samples: dict[str, list[dict[str, Any]]] = {
        case_name: [] for case_name in case_names
    }
    order: list[str] = []
    for repeat in range(args.repeats):
        order.extend(case_names if repeat % 2 == 0 else reversed(case_names))
    for index, case_name in enumerate(order, start=1):
        sample = run_case(case_name, reference)
        samples[case_name].append(sample)
        print(
            json.dumps(
                {
                    "sample": index,
                    "case": case_name,
                    "decode_ms": sample["decode_ms"],
                    "tokens": sample["token_check"],
                    "graph_contract_exact": sample["graph_contract"]["exact"],
                    "softcap_contract_exact": sample["softcap_contract"]["exact"],
                    "deterministic_moe_contract_exact": sample[
                        "deterministic_moe_contract"
                    ]["exact"],
                },
                sort_keys=True,
            )
        )

    baseline = summarize_case(samples[baseline_name])
    softcap_only = summarize_case(samples[softcap_name])
    candidate = summarize_case(samples[candidate_name])
    decision = decide_gate(
        baseline,
        candidate,
        minimum_speedup=args.minimum_speedup,
        minimum_savings_ms=args.minimum_savings_ms,
        maximum_stability_ratio=args.maximum_stability_ratio,
    )
    if not softcap_only["tokens_exact"]:
        isolated_result = "FUSED_SOFTCAP_TOKEN_DIVERGENCE"
    elif not candidate["tokens_exact"]:
        isolated_result = "GPU_FEEDBACK_BURST_TOKEN_DIVERGENCE"
    elif not candidate["graph_contract_exact"]:
        isolated_result = "GPU_FEEDBACK_BURST_CONTRACT_FAILURE"
    else:
        isolated_result = "EXACT"
    configure(baseline_name)
    reset_decode_runtime()

    result = {
        "harness_rev": HARNESS_REV,
        "gpu": gpu_snapshot(),
        "model": args.model,
        "shape": {
            "batch_size": args.batch_size,
            "context": args.context,
            "max_tokens": args.max_tokens,
            "max_seq_len": args.max_seq_len,
            "burst_size": args.burst_size,
            "dtype": args.dtype,
        },
        "checkpoint_loads": 1,
        "vllm_install": False,
        "candidate_mode": {
            "token_burst": True,
            "burst_size": args.burst_size,
            "persistent_token_feedback": False,
            "feedback": "gpu_to_gpu_between_graph_replays",
            "persistent_rejection": (
                "v1 was repeat-stable but diverged from one-step tokens at B16/C2048"
            ),
            "isolation_case": softcap_name,
        },
        "prompt_manifest": {
            "schema_version": int(manifest.get("schema_version", 0) or 0),
            "generator": str(manifest.get("generator", "") or ""),
            "base_prompt_tokens": int(
                manifest.get("base_prompt_tokens", 0) or 0
            ),
            "contract": dict(
                ((manifest.get("cases") or {}).get(str(args.context)) or {}).get(
                    "contract"
                )
                or {}
            ),
            "path": str(args.prompt_token_ids_json),
        },
        "prefill_fallback": fallback,
        "reference_output_contract": token_matrix_contract(reference),
        "priming_tokens": prime_checks,
        "isolation": {
            "result": isolated_result,
            "softcap_only_tokens_exact": softcap_only["tokens_exact"],
            "gpu_feedback_burst_tokens_exact": candidate["tokens_exact"],
        },
        "cases": {
            baseline_name: baseline,
            softcap_name: softcap_only,
            candidate_name: candidate,
        },
        "decision": decision,
        "runtime_restored_to_baseline": True,
    }
    _write_json(Path(args.out_json), result)
    print("DECISION " + json.dumps(decision, sort_keys=True))
    print("wrote", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

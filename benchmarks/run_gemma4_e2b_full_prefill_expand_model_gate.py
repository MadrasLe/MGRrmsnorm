#!/usr/bin/env python3
"""Paired loaded-model A/B for E2B/L4 expanded full-prefill attention."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_inference_matrix as matrix
from benchmarks.run_gemma4_e2b_phase_split import (
    DEFAULT_MODEL,
    _configure_megagemm_profile,
)


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def summarize(
    samples: list[dict[str, Any]],
    *,
    minimum_speedup: float,
    maximum_spread: float,
) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}
    for name in ("baseline_explicit", "candidate_expanded_implicit"):
        selected = [row for row in samples if row["case"] == name]
        prefill = [float(row["prefill_ms"]) for row in selected]
        wall = [float(row["wall_ms"]) for row in selected]
        cases[name] = {
            "samples": len(selected),
            "prefill_ms_median": _median(prefill),
            "prefill_ms_samples": prefill,
            "prefill_spread_ratio": max(prefill) / min(prefill) if prefill else None,
            "wall_ms_median": _median(wall),
            "runtime_hits": sum(int(row["runtime_hits_delta"]) for row in selected),
            "token_digests": sorted({str(row["token_digest"]) for row in selected}),
            "errors": sorted(
                {
                    str(row["runtime_error"])
                    for row in selected
                    if row.get("runtime_error")
                }
            ),
        }

    baseline = cases["baseline_explicit"]
    candidate = cases["candidate_expanded_implicit"]
    baseline_ms = float(baseline["prefill_ms_median"])
    candidate_ms = float(candidate["prefill_ms_median"])
    speedup = baseline_ms / candidate_ms if candidate_ms > 0.0 else 0.0
    saved_ms = baseline_ms - candidate_ms
    digests = {
        str(row["token_digest"])
        for row in samples
        if row.get("token_digest")
    }
    correct = len(digests) == 1
    exact_hits = all(
        int(row["runtime_hits_delta"]) == (7 if row["case"].startswith("candidate") else 0)
        for row in samples
    )
    stable = bool(
        float(baseline["prefill_spread_ratio"] or float("inf")) <= maximum_spread
        and float(candidate["prefill_spread_ratio"] or float("inf")) <= maximum_spread
    )
    no_errors = not baseline["errors"] and not candidate["errors"]
    apply_change = bool(
        correct
        and exact_hits
        and stable
        and no_errors
        and speedup >= minimum_speedup
    )
    return {
        "decision": (
            "PROMOTE_E2B_L4_FULL_PREFILL_EXPAND"
            if apply_change
            else "DO_NOT_PROMOTE_E2B_L4_FULL_PREFILL_EXPAND"
        ),
        "apply_change": apply_change,
        "cases": cases,
        "prefill_speedup": speedup,
        "prefill_saved_ms": saved_ms,
        "candidate_prefill_reduction_pct": (
            saved_ms / baseline_ms * 100.0 if baseline_ms > 0.0 else 0.0
        ),
        "token_digest_exact": correct,
        "runtime_hits_exact": exact_hits,
        "stable": stable,
        "no_runtime_errors": no_errors,
        "minimum_speedup": minimum_speedup,
        "maximum_spread": maximum_spread,
    }


def _attention_modules(engine) -> list[Any]:
    modules = []
    for layer in engine.model.layers:
        attention = getattr(layer, "self_attn", None)
        if (
            attention is not None
            and int(getattr(attention, "sliding_window", 0) or 0) <= 0
            and int(getattr(attention, "head_dim", 0) or 0) == 512
        ):
            modules.append(attention)
    return modules


def _set_candidate(modules: list[Any], enabled: bool) -> None:
    for attention in modules:
        attention._gemma4_e2b_l4_full_prefill_expand_enabled = bool(enabled)
        attention._gemma4_e2b_l4_full_prefill_expand_error = ""


def _hit_count(modules: list[Any]) -> int:
    return sum(
        int(getattr(attention, "_gemma4_e2b_l4_full_prefill_expand_hits", 0))
        for attention in modules
    )


def _runtime_error(modules: list[Any]) -> str:
    return next(
        (
            str(attention._gemma4_e2b_l4_full_prefill_expand_error)
            for attention in modules
            if getattr(attention, "_gemma4_e2b_l4_full_prefill_expand_error", "")
        ),
        "",
    )


def _token_digest(engine) -> str:
    scheduler = engine._last_scheduler
    completed = sorted(scheduler._completed, key=lambda request: int(request.request_id))
    rows = [
        [int(token_id) for token_id in request.generated_ids]
        for request in completed
    ]
    encoded = json.dumps(rows, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sample(
    engine,
    modules: list[Any],
    prompts: list[str],
    *,
    case: str,
    pair: int,
    position: int,
) -> dict[str, Any]:
    import torch

    candidate = case == "candidate_expanded_implicit"
    _set_candidate(modules, candidate)
    before = _hit_count(modules)
    torch.cuda.synchronize()
    started = time.perf_counter()
    engine.generate_batch(
        prompts,
        max_new_tokens=1,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        ignore_eos=True,
        decode_outputs=False,
    )
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    after = _hit_count(modules)
    scheduler = engine._last_scheduler
    stats = scheduler.get_stats()
    generated = int(stats.get("total_tokens") or 0)
    if generated != len(prompts):
        raise RuntimeError(f"generated {generated} tokens; expected {len(prompts)}")
    return {
        "case": case,
        "pair": pair,
        "position": position,
        "wall_ms": wall_ms,
        "prefill_ms": float(stats.get("prefill_time_ms") or 0.0),
        "decode_ms": float(stats.get("decode_time_ms") or 0.0),
        "runtime_hits_delta": after - before,
        "runtime_error": _runtime_error(modules),
        "token_digest": _token_digest(engine),
        "chunk_plan": stats.get("prefill_chunk_plan"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from megagemm.engine import InferenceEngine

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu = torch.cuda.get_device_name(0)
    if "L4" not in gpu.upper():
        raise RuntimeError(f"this gate requires NVIDIA L4, found {gpu}")

    profile = _configure_megagemm_profile(args.model)
    engine = InferenceEngine(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
        max_batch_size=8,
        max_seq_len=2304,
        num_blocks=0,
        block_size=16,
        kv_alloc="auto",
        cache_dir=args.cache_dir,
    )
    prompts, constructed_tokens = matrix.build_prompts(engine.tokenizer, 8, 2048)
    modules = _attention_modules(engine)
    if len(modules) != 7:
        raise RuntimeError(f"expected seven full H512 layers, found {len(modules)}")

    print("Gemma 4 E2B/L4 full-prefill expanded-SDPA loaded-model gate")
    print(f"  gpu: {gpu}")
    print(f"  model: {args.model}")
    print("  workload: B8/P2048/BF16/max_new_tokens=1")
    print(f"  full H512 layers: {len(modules)}")
    print(f"  constructed prompt tokens: {constructed_tokens}")
    print(f"  warmups: {args.warmups} per case")
    print(f"  paired repeats: {args.repeats}")

    for case in ("baseline_explicit", "candidate_expanded_implicit"):
        for index in range(1, args.warmups + 1):
            print(f"Warmup case={case} repeat={index}/{args.warmups}", flush=True)
            _sample(
                engine,
                modules,
                prompts,
                case=case,
                pair=-index,
                position=0,
            )

    samples: list[dict[str, Any]] = []
    for pair in range(1, args.repeats + 1):
        order = (
            ("baseline_explicit", "candidate_expanded_implicit")
            if pair % 2
            else ("candidate_expanded_implicit", "baseline_explicit")
        )
        for position, case in enumerate(order, start=1):
            print(
                f"Measure pair={pair}/{args.repeats} position={position}/2 case={case}",
                flush=True,
            )
            row = _sample(
                engine,
                modules,
                prompts,
                case=case,
                pair=pair,
                position=position,
            )
            samples.append(row)
            print(
                f"  prefill={row['prefill_ms']:.2f}ms wall={row['wall_ms']:.2f}ms "
                f"hits={row['runtime_hits_delta']} digest={row['token_digest'][:12]}",
                flush=True,
            )

    summary = summarize(
        samples,
        minimum_speedup=args.minimum_speedup,
        maximum_spread=args.maximum_spread,
    )
    payload = {
        "benchmark": "gemma4_e2b_full_prefill_expand_model_gate",
        "model": args.model,
        "gpu": gpu,
        "torch": torch.__version__,
        "profile_environment": profile,
        "system": {
            "git": matrix.git_snapshot(),
            "gpu": matrix.gpu_snapshot(),
            "packages": matrix.installed_package_versions(),
        },
        "workload": {
            "batch_size": 8,
            "prompt_tokens_requested": 2048,
            "constructed_prompt_tokens": constructed_tokens,
            "max_new_tokens": 1,
            "full_h512_layers": len(modules),
        },
        "samples": samples,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nGEMMA 4 E2B / L4 — FULL PREFILL EXPAND A/B")
    print("case                              prefill ms      wall ms       hits")
    for name, row in summary["cases"].items():
        print(
            f"{name:<34} {row['prefill_ms_median']:10.2f} "
            f"{row['wall_ms_median']:12.2f} {row['runtime_hits']:10d}"
        )
    print(
        f"speedup={summary['prefill_speedup']:.4f}x "
        f"saved={summary['prefill_saved_ms']:.2f}ms "
        f"reduction={summary['candidate_prefill_reduction_pct']:.2f}%"
    )
    print("DECISION " + json.dumps(summary, sort_keys=True))
    print(f"Wrote: {args.output}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--minimum-speedup", type=float, default=1.05)
    parser.add_argument("--maximum-spread", type=float, default=1.08)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmups < 1 or args.repeats < 3:
        raise SystemExit("use at least one warmup and three paired repeats")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

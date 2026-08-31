#!/usr/bin/env python3
"""Loaded-model A/B for the Gemma 4 E2B/L4 LM-head launch candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_inference_matrix as matrix
from benchmarks.run_gemma4_e2b_lm_head_sweep import (
    CURRENT,
    LaunchConfig,
    _set_config,
)
from benchmarks.run_gemma4_e2b_phase_split import (
    DEFAULT_MODEL,
    _configure_megagemm_profile,
)


CANDIDATE = LaunchConfig(64, 128, 4, 2)
CASES = {
    "production": CURRENT,
    "candidate_bn64": CANDIDATE,
}


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _token_digest(engine: Any) -> str:
    scheduler = engine._last_scheduler
    completed = sorted(
        scheduler._completed,
        key=lambda request: int(request.request_id),
    )
    rows = [
        [int(token_id) for token_id in request.generated_ids]
        for request in completed
    ]
    encoded = json.dumps(rows, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sample(
    engine: Any,
    kernel_module: Any,
    prompts: list[str],
    *,
    case: str,
    pair: int,
    position: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    config = CASES[case]
    _set_config(kernel_module, config)
    torch.cuda.synchronize()
    started = time.perf_counter()
    engine.generate_batch(
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        ignore_eos=True,
        decode_outputs=False,
    )
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0

    scheduler = engine._last_scheduler
    stats = scheduler.get_stats()
    generated = int(stats.get("total_tokens") or 0)
    expected = len(prompts) * max_new_tokens
    if generated != expected:
        raise RuntimeError(f"generated {generated} tokens; expected {expected}")
    runtime = engine.model.decode_runtime_stats()
    return {
        "case": case,
        "config": config.name,
        "pair": pair,
        "position": position,
        "wall_ms": wall_ms,
        "prefill_ms": float(stats.get("prefill_time_ms") or 0.0),
        "decode_ms": float(stats.get("decode_time_ms") or 0.0),
        "generated_tokens": generated,
        "token_digest": _token_digest(engine),
        "fused_rmsnorm_lm_head_use": bool(
            runtime.get("fused_rmsnorm_lm_head_argmax_use", False)
        ),
        "fused_rmsnorm_lm_head_disabled": bool(
            runtime.get("fused_rmsnorm_lm_head_argmax_disabled", False)
        ),
        "fused_rmsnorm_lm_head_error": str(
            runtime.get("fused_rmsnorm_lm_head_argmax_error") or ""
        ),
    }


def summarize_full_model_gate(
    samples: list[dict[str, Any]],
    *,
    batch_size: int,
    max_new_tokens: int,
    minimum_speedup: float,
    maximum_ratio_spread: float,
    minimum_faster_fraction: float,
) -> dict[str, Any]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(int(sample["pair"]), {})[str(sample["case"])] = sample

    pairs: list[dict[str, Any]] = []
    for pair_index in sorted(grouped):
        pair = grouped[pair_index]
        if set(pair) != set(CASES):
            continue
        current = pair["production"]
        candidate = pair["candidate_bn64"]
        current_ms = float(current["decode_ms"])
        candidate_ms = float(candidate["decode_ms"])
        if current_ms <= 0.0 or candidate_ms <= 0.0:
            continue
        pairs.append(
            {
                "pair": pair_index,
                "production_decode_ms": current_ms,
                "candidate_decode_ms": candidate_ms,
                "speedup": current_ms / candidate_ms,
                "saved_ms": current_ms - candidate_ms,
            }
        )

    speedups = [float(pair["speedup"]) for pair in pairs]
    saved = [float(pair["saved_ms"]) for pair in pairs]
    faster_pairs = sum(value > 1.0 for value in speedups)
    required_faster = math.ceil(len(speedups) * minimum_faster_fraction)
    ratio_spread = max(speedups) / min(speedups) if speedups else None
    median_speedup = _median(speedups)
    median_saved = _median(saved)
    all_digests = {
        str(sample["token_digest"])
        for sample in samples
        if sample.get("token_digest")
    }
    tokens_exact = len(all_digests) == 1
    fused_path_active = bool(
        samples
        and all(sample.get("fused_rmsnorm_lm_head_use") for sample in samples)
        and not any(sample.get("fused_rmsnorm_lm_head_disabled") for sample in samples)
        and not any(sample.get("fused_rmsnorm_lm_head_error") for sample in samples)
    )
    apply_change = bool(
        pairs
        and tokens_exact
        and fused_path_active
        and median_speedup >= minimum_speedup
        and ratio_spread is not None
        and ratio_spread <= maximum_ratio_spread
        and faster_pairs >= required_faster
    )

    incremental_tokens = batch_size * (max_new_tokens - 1)
    cases: dict[str, dict[str, Any]] = {}
    for case in CASES:
        selected = [sample for sample in samples if sample["case"] == case]
        decode_values = [float(sample["decode_ms"]) for sample in selected]
        median_decode = _median(decode_values)
        cases[case] = {
            "config": CASES[case].name,
            "samples": len(selected),
            "decode_ms_median": median_decode,
            "decode_ms_samples": decode_values,
            "incremental_decode_tps": (
                incremental_tokens / (median_decode / 1000.0)
                if median_decode > 0.0
                else 0.0
            ),
            "wall_ms_median": _median(
                [float(sample["wall_ms"]) for sample in selected]
            ),
            "prefill_ms_median": _median(
                [float(sample["prefill_ms"]) for sample in selected]
            ),
        }

    return {
        "decision": (
            "PROMOTE_GEMMA4_E2B_L4_LM_HEAD_BN64_BK128_W4_S2"
            if apply_change
            else "KEEP_GEMMA4_E2B_L4_LM_HEAD_PRODUCTION_CONFIG"
        ),
        "apply_change": apply_change,
        "cases": cases,
        "pairs": pairs,
        "paired_speedup_median": median_speedup,
        "paired_saved_ms_median": median_saved,
        "paired_speedup_spread_ratio": ratio_spread,
        "candidate_faster_pairs": faster_pairs,
        "required_faster_pairs": required_faster,
        "token_digest_exact": tokens_exact,
        "fused_path_active_all_samples": fused_path_active,
        "minimum_speedup": minimum_speedup,
        "maximum_ratio_spread": maximum_ratio_spread,
        "minimum_faster_fraction": minimum_faster_fraction,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import importlib
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu = torch.cuda.get_device_name(0)
    if "L4" not in gpu.upper():
        raise RuntimeError(f"this gate requires NVIDIA L4, found {gpu}")

    # Apply the profile before importing the engine/model modules because their
    # production policy flags are resolved at module import time.
    profile = _configure_megagemm_profile(args.model)
    from megagemm.engine import InferenceEngine

    kernel_module = importlib.import_module("megagemm.kernels.lm_head_argmax")
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

    print("Gemma 4 E2B/L4 LM-head loaded-model A/B")
    print(f"  gpu: {gpu}")
    print(f"  model: {args.model}")
    print("  workload: B8/P2048/BF16")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  constructed prompt tokens: {constructed_tokens}")
    print(f"  production: {CURRENT.name}")
    print(f"  candidate: {CANDIDATE.name}")
    print(f"  paired repeats: {args.repeats}")

    for case in CASES:
        for index in range(1, args.warmups + 1):
            print(f"Warmup case={case} repeat={index}/{args.warmups}", flush=True)
            _sample(
                engine,
                kernel_module,
                prompts,
                case=case,
                pair=-index,
                position=0,
                max_new_tokens=args.max_new_tokens,
            )

    samples: list[dict[str, Any]] = []
    for pair in range(1, args.repeats + 1):
        order = (
            ("production", "candidate_bn64")
            if pair % 2
            else ("candidate_bn64", "production")
        )
        for position, case in enumerate(order, start=1):
            print(
                f"Measure pair={pair}/{args.repeats} "
                f"position={position}/2 case={case}",
                flush=True,
            )
            row = _sample(
                engine,
                kernel_module,
                prompts,
                case=case,
                pair=pair,
                position=position,
                max_new_tokens=args.max_new_tokens,
            )
            samples.append(row)
            print(
                f"  decode={row['decode_ms']:.2f}ms "
                f"prefill={row['prefill_ms']:.2f}ms "
                f"wall={row['wall_ms']:.2f}ms "
                f"digest={row['token_digest'][:12]}",
                flush=True,
            )

    summary = summarize_full_model_gate(
        samples,
        batch_size=8,
        max_new_tokens=args.max_new_tokens,
        minimum_speedup=args.minimum_speedup,
        maximum_ratio_spread=args.maximum_ratio_spread,
        minimum_faster_fraction=args.minimum_faster_fraction,
    )
    payload = {
        "benchmark": "gemma4_e2b_lm_head_full_model_gate",
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
            "max_new_tokens": args.max_new_tokens,
        },
        "samples": samples,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nGEMMA 4 E2B / L4 — LM HEAD FULL-MODEL A/B")
    print("case                   decode ms    decode tok/s       wall ms")
    for case, row in summary["cases"].items():
        print(
            f"{case:<22} {row['decode_ms_median']:10.2f} "
            f"{row['incremental_decode_tps']:15.2f} "
            f"{row['wall_ms_median']:13.2f}"
        )
    print("DECISION " + json.dumps(summary, sort_keys=True))
    print(f"Wrote: {args.output}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.005)
    parser.add_argument("--maximum-ratio-spread", type=float, default=1.06)
    parser.add_argument("--minimum-faster-fraction", type=float, default=0.80)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmups < 1 or args.repeats < 3 or args.max_new_tokens < 2:
        raise SystemExit("use warmups>=1, repeats>=3, max-new-tokens>=2")
    if not 0.5 <= args.minimum_faster_fraction <= 1.0:
        raise SystemExit("minimum-faster-fraction must be in [0.5, 1.0]")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Loaded-model factorial A/B for E2B H512 attention and the dense bridge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

try:
    from run_gemma4_e2b_ple_conditioned_gelu import (
        _decode_tps,
        _successful_long_rows,
    )
except ImportError:
    from benchmarks.run_gemma4_e2b_ple_conditioned_gelu import (
        _decode_tps,
        _successful_long_rows,
    )


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "benchmarks" / "benchmark_inference_matrix.py"
CASES: tuple[tuple[str, bool, bool], ...] = (
    ("baseline", False, False),
    ("h512_attention", True, False),
    ("dense_bridge", False, True),
    ("combined", True, True),
    ("baseline_recheck", False, False),
)


def _case_environment(
    *,
    h512: bool,
    bridge: bool,
    forced_token_id: int,
) -> dict[str, str]:
    env = os.environ.copy()
    root_text = str(ROOT)
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        root_text
        if not current_pythonpath
        else root_text + os.pathsep + current_pythonpath
    )
    env.update(
        {
            "MEGAGEMM_FLAT_DECODE": "1",
            "MEGAGEMM_BENCHMARK_TOKEN_DIGEST": "1",
            # Keep every candidate on the exact same autoregressive route.
            # The real LM head and argmax still execute before feedback is
            # overwritten by this benchmark-only token.
            "MEGAGEMM_BENCHMARK_FORCED_TOKEN_ID": str(forced_token_id),
            "MEGAGEMM_DECODE_CUDA_GRAPHS": "0",
            "MEGAGEMM_DECODE_PREFER_STEP": "0",
            "MEGAGEMM_REUSE_REQUEST_SCHEDULER": "0",
            "MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE": "1",
            "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_PREFILL": "1",
            "MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE": "0",
            "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE": "0",
            "MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE": "0",
            "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE": "0",
            "MEGAGEMM_GEMMA4_PLE_CONDITIONED_GELU_DECODE": "0",
            "MEGAGEMM_GEMMA4_E2B_CUBLASLT_GATEUP_DECODE": "0",
            "MEGAGEMM_PAGED_DECODE_SPLITS": "1",
            "MEGAGEMM_PAGED_DECODE_GQA2": "1",
            "MEGAGEMM_PAGED_DECODE_WARPS_H256": "2",
            "MEGAGEMM_GEMMA4_E2B_L4_H512_GROUPED_ATTN_DECODE": (
                "1" if h512 else "0"
            ),
            "MEGAGEMM_GEMMA4_E2B_L4_H512_ATTN_SEGMENTS": "32",
            "MEGAGEMM_GEMMA4_E2B_L4_H512_ATTN_TILE": "16",
            "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_WARPS": "4",
            "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_STAGES": "3",
            "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_REDUCE_WARPS": "4",
            "MEGAGEMM_GEMMA4_DENSE_ATTN_MLP_BRIDGE_DECODE": (
                "1" if bridge else "0"
            ),
        }
    )
    return env


def _audit_row(
    row: dict[str, Any],
    *,
    expect_h512: bool,
    expect_bridge: bool,
    forced_token_id: int,
    index: int,
) -> list[str]:
    errors: list[str] = []
    stats = row.get("decode_runtime_stats")
    if not isinstance(stats, dict):
        return [f"row {index}: decode_runtime_stats missing"]
    scheduler = row.get("scheduler_stats")
    if not isinstance(scheduler, dict):
        errors.append(f"row {index}: scheduler_stats missing")
    elif int(scheduler.get("benchmark_forced_token_id", -1)) != forced_token_id:
        errors.append(
            f"row {index}: forced token route is not active "
            f"({scheduler.get('benchmark_forced_token_id')} != {forced_token_id})"
        )
    paged = stats.get("paged_decode_runtime")
    if not isinstance(paged, dict):
        errors.append(f"row {index}: paged_decode_runtime missing")
        paged = {}
    grouped_hits = int(paged.get("grouped_segmented_hits") or 0)
    grouped_disabled = bool(paged.get("grouped_segmented_disabled", False))
    grouped_failure = str(paged.get("grouped_segmented_failure") or "")
    segments = paged.get("grouped_segmented_selected_segments") or {}
    tiles = paged.get("grouped_segmented_selected_tile_sizes") or {}
    topology = "e2b_l4_full_h512_gqa8"
    if expect_h512:
        if grouped_hits <= 0:
            errors.append(f"row {index}: grouped H512 recorded no hits")
        if int(segments.get(topology, 0) or 0) != 32:
            errors.append(f"row {index}: H512 did not select 32 segments")
        if int(tiles.get(topology, 0) or 0) != 16:
            errors.append(f"row {index}: H512 did not select tile 16")
        if grouped_disabled or grouped_failure:
            errors.append(
                f"row {index}: grouped H512 disabled itself: {grouped_failure}"
            )
    elif grouped_hits != 0:
        errors.append(f"row {index}: baseline unexpectedly has {grouped_hits} grouped hits")

    bridge_enabled = bool(
        stats.get("gemma4_dense_attn_mlp_bridge_decode_enabled")
    )
    bridge_hits = int(
        stats.get("gemma4_dense_attn_mlp_bridge_decode_hits") or 0
    )
    bridge_disabled = bool(
        stats.get("gemma4_dense_attn_mlp_bridge_runtime_disabled")
    )
    bridge_failure = str(stats.get("gemma4_dense_attn_mlp_bridge_failure") or "")
    if bridge_enabled != expect_bridge:
        errors.append(
            f"row {index}: bridge enabled={bridge_enabled}, expected {expect_bridge}"
        )
    if expect_bridge and bridge_hits <= 0:
        errors.append(f"row {index}: dense bridge recorded no hits")
    if not expect_bridge and bridge_hits != 0:
        errors.append(f"row {index}: baseline unexpectedly has {bridge_hits} bridge hits")
    if bridge_disabled or bridge_failure:
        errors.append(f"row {index}: dense bridge disabled itself: {bridge_failure}")
    return errors


def _case_result(
    name: str,
    case_dir: Path,
    *,
    expect_h512: bool,
    expect_bridge: bool,
    forced_token_id: int,
) -> dict[str, Any]:
    raw_files = sorted(case_dir.glob("*.jsonl"))
    if len(raw_files) != 1:
        return {
            "name": name,
            "status": "failed",
            "errors": [f"expected one JSONL artifact, found {len(raw_files)}"],
        }
    rows = _successful_long_rows(raw_files[0])
    errors: list[str] = []
    for index, row in enumerate(rows):
        errors.extend(
            _audit_row(
                row,
                expect_h512=expect_h512,
                expect_bridge=expect_bridge,
                forced_token_id=forced_token_id,
                index=index,
            )
        )
    decode_samples = [_decode_tps(row) for row in rows]
    output_samples = [float(row.get("output_tps") or 0.0) for row in rows]
    digests = [str(row.get("generated_token_digest") or "") for row in rows]
    if not rows:
        errors.append("no successful P2048/B8 rows")
    if any(value <= 0.0 for value in decode_samples):
        errors.append("one or more decode wall-throughput samples are missing")
    if any(not digest for digest in digests):
        errors.append("one or more generated-token digests are missing")
    return {
        "name": name,
        "status": "ok" if not errors else "failed",
        "raw_jsonl": str(raw_files[0]),
        "decode_samples_tps": decode_samples,
        "output_samples_tps": output_samples,
        "median_decode_tps": (
            statistics.median(decode_samples) if decode_samples else 0.0
        ),
        "median_output_tps": (
            statistics.median(output_samples) if output_samples else 0.0
        ),
        "spread_ratio": (
            max(decode_samples) / min(decode_samples)
            if decode_samples and min(decode_samples) > 0.0
            else float("inf")
        ),
        "digests": digests,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--h512-minimum-speedup", type=float, default=1.005)
    parser.add_argument("--bridge-minimum-speedup", type=float, default=1.001)
    parser.add_argument("--combined-minimum-speedup", type=float, default=1.005)
    parser.add_argument("--maximum-spread", type=float, default=1.04)
    parser.add_argument("--forced-token-id", type=int, default=42)
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Return status 2 when the scientific gate is invalid.",
    )
    args = parser.parse_args()
    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3")
    if args.forced_token_id < 0:
        raise ValueError("--forced-token-id must be non-negative")
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for name, h512, bridge in CASES:
        case_dir = out_root / name
        case_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(MATRIX),
            "--backend",
            "megagemm",
            "--model",
            args.model,
            "--hardware-label",
            "1xl4",
            "--batch-sizes",
            "8",
            "--prompt-tokens",
            "2048",
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--repeats",
            str(args.repeats),
            "--warmup",
            str(args.warmup),
            "--out-dir",
            str(case_dir),
            "--run-id",
            f"gemma4_e2b_{name}",
            "--device",
            "cuda",
            "--dtype",
            "bf16",
            "--max-seq-len",
            "2304",
            "--max-batch-size",
            "8",
            "--ignore-eos",
        ]
        print(f"\n=== {name} ===", flush=True)
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=_case_environment(
                h512=h512,
                bridge=bridge,
                forced_token_id=args.forced_token_id,
            ),
            check=False,
        )
        if completed.returncode != 0:
            results.append(
                {
                    "name": name,
                    "status": "failed",
                    "errors": [f"benchmark exited with {completed.returncode}"],
                }
            )
            continue
        results.append(
            _case_result(
                name,
                case_dir,
                expect_h512=h512,
                expect_bridge=bridge,
                forced_token_id=args.forced_token_id,
            )
        )

    by_name = {row["name"]: row for row in results}
    errors = [
        f"{row['name']}: {error}"
        for row in results
        for error in row.get("errors", ())
    ]
    if set(by_name) != {name for name, _, _ in CASES}:
        errors.append("one or more factorial cases are missing")
    all_digests = {
        digest
        for row in results
        for digest in row.get("digests", ())
        if digest
    }
    digest_match = len(all_digests) == 1
    if not digest_match:
        errors.append("generated-token digests differ across factorial cases")
    case_stable = all(
        float(row.get("spread_ratio") or float("inf")) <= args.maximum_spread
        for row in results
    )
    baseline_medians = [
        float(by_name.get(name, {}).get("median_decode_tps") or 0.0)
        for name in ("baseline", "baseline_recheck")
    ]
    baseline_medians = [value for value in baseline_medians if value > 0.0]
    baseline_stability_ratio = (
        max(baseline_medians) / min(baseline_medians)
        if len(baseline_medians) == 2
        else float("inf")
    )
    stable = bool(
        case_stable and baseline_stability_ratio <= args.maximum_spread
    )
    if not case_stable:
        errors.append("one or more decode cases exceeded the allowed spread")
    if baseline_stability_ratio > args.maximum_spread:
        errors.append("baseline/recheck drift exceeded the allowed spread")

    baseline_tps = (
        statistics.median(baseline_medians)
        if len(baseline_medians) == 2
        else 0.0
    )
    case_positions = {name: index for index, (name, _, _) in enumerate(CASES)}
    final_position = case_positions["baseline_recheck"]
    baseline_first_tps = float(
        by_name.get("baseline", {}).get("median_decode_tps") or 0.0
    )
    baseline_last_tps = float(
        by_name.get("baseline_recheck", {}).get("median_decode_tps") or 0.0
    )
    interpolated_baselines = {}
    for name in ("h512_attention", "dense_bridge", "combined"):
        fraction = case_positions[name] / final_position
        interpolated_baselines[name] = (
            baseline_first_tps
            + ((baseline_last_tps - baseline_first_tps) * fraction)
        )
    speedups = {
        name: (
            float(by_name.get(name, {}).get("median_decode_tps") or 0.0)
            / interpolated_baselines[name]
            if interpolated_baselines[name] > 0.0
            else 0.0
        )
        for name in ("h512_attention", "dense_bridge", "combined")
    }
    all_valid = bool(
        not errors
        and len(results) == len(CASES)
        and all(row.get("status") == "ok" for row in results)
    )
    promote_h512 = bool(
        all_valid and speedups["h512_attention"] >= args.h512_minimum_speedup
    )
    promote_bridge = bool(
        all_valid and speedups["dense_bridge"] >= args.bridge_minimum_speedup
    )
    promote_pair = bool(
        all_valid and speedups["combined"] >= args.combined_minimum_speedup
    )
    if not all_valid:
        decision = "INVALID_GATE"
    elif promote_pair:
        decision = "PROMOTE_H512_AND_BRIDGE_AS_PAIR"
    elif promote_h512:
        decision = "PROMOTE_H512_ONLY"
    elif promote_bridge:
        decision = "PROMOTE_BRIDGE_ONLY"
    else:
        decision = "KEEP_BASELINE"
    payload = {
        "decision": decision,
        "model": args.model,
        "shape": "L4/BF16/P2048/B8",
        "route_normalized_forced_token_id": args.forced_token_id,
        "digest_match": digest_match,
        "stable": stable,
        "baseline_stability_ratio": baseline_stability_ratio,
        "baseline_reference_decode_tps": baseline_tps,
        "interpolated_baseline_decode_tps": interpolated_baselines,
        "speedups_vs_interpolated_baseline": speedups,
        "h512_minimum_speedup": args.h512_minimum_speedup,
        "bridge_minimum_speedup": args.bridge_minimum_speedup,
        "combined_minimum_speedup": args.combined_minimum_speedup,
        "maximum_spread": args.maximum_spread,
        "winner_h512_config": {
            "segments": 32,
            "tile_size": 16,
            "warps": 4,
            "stages": 3,
            "reduce_warps": 4,
        },
        "results": results,
        "errors": errors,
    }
    decision_path = out_root / "decision.json"
    decision_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + "=" * 96)
    print("GEMMA 4 E2B / L4 / BF16 / P2048 / B8 — H512 + DENSE BRIDGE FACTORIAL")
    print("=" * 96)
    for row in results:
        print(
            f"{row['name']:<18} decode={float(row.get('median_decode_tps') or 0.0):8.2f} "
            f"output={float(row.get('median_output_tps') or 0.0):8.2f} "
            f"spread={float(row.get('spread_ratio') or 0.0):.4f}"
        )
    print("DECISION " + json.dumps(payload, separators=(",", ":")))
    print(f"wrote {decision_path}")
    if args.strict_exit and decision == "INVALID_GATE":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Tune Gemma4 B16 grouped decode attention at the long-context endpoint."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

from run_gemma4_grouped_segmented_attention_microbench import (
    BLOCK_SIZE,
    MAX_ABS_ERROR,
    MAX_BASELINE_STABILITY_RATIO,
    MAX_RELATIVE_L2_ERROR,
    MIN_COSINE,
    ROWS,
    TOPOLOGIES,
    capture_case,
    make_inputs,
    torch_reference,
)

TARGET_DECODE_GAP_MS = 147.0


def _numeric_valid(row: dict[str, Any]) -> bool:
    return bool(
        row.get("error") is None
        and float(row["repeat_max_abs_error"]) == 0.0
        and float(row["max_abs_error"]) <= MAX_ABS_ERROR
        and float(row["relative_l2_error"]) <= MAX_RELATIVE_L2_ERROR
        and float(row["cosine"]) >= MIN_COSINE
    )


@torch.inference_mode()
def run_topology(
    topology: Any,
    *,
    context: int,
    table_blocks: int,
    segment_counts: tuple[int, ...],
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    from megagemm.kernels.paged_attention import (
        _triton_paged_decode_grouped_segmented_fused,
    )

    tensors = make_inputs(
        topology,
        context=context,
        table_blocks=table_blocks,
        norm_eps=1.0e-6,
    )
    reference = torch_reference(
        query=tensors["query_fp32"],
        kv_cache=tensors["kv_cache"],
        block_table=tensors["block_table"],
        context=context,
        scale=1.0,
        sliding_window=topology.sliding_window,
    )

    def make_case(segments: int, suffix: str = "") -> dict[str, Any]:
        out = torch.empty_like(tensors["raw_query"])

        def call() -> torch.Tensor:
            return _triton_paged_decode_grouped_segmented_fused(
                tensors["raw_query"],
                tensors["kv_cache"],
                tensors["block_table"],
                tensors["seq_lens"],
                1.0,
                tensors["cos"],
                tensors["sin"],
                tensors["positions"],
                half_rotate=True,
                rotary_dim=topology.head_dim,
                q_norm_weight=tensors["norm_weight"],
                norm_eps=1.0e-6,
                out=out,
                sliding_window=topology.sliding_window,
                force=True,
                num_segments_override=segments,
            )

        row = capture_case(
            name=f"segments_{segments}{suffix}",
            call=call,
            out=out,
            reference=reference,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        row["segments"] = segments
        return row

    rows = [make_case(16)]
    rows.extend(make_case(count) for count in segment_counts if count != 16)
    rows.append(make_case(16, "_recheck"))

    baseline_rows = (rows[0], rows[-1])
    baseline_valid = all(_numeric_valid(row) for row in baseline_rows)
    if baseline_valid:
        baseline_samples = [float(row["median_us"]) for row in baseline_rows]
        baseline_us = min(baseline_samples)
        baseline_stability = max(baseline_samples) / baseline_us
    else:
        baseline_us = None
        baseline_stability = None

    valid_candidates = [row for row in rows[:-1] if _numeric_valid(row)]
    if (
        baseline_us is not None
        and baseline_stability is not None
        and baseline_stability <= MAX_BASELINE_STABILITY_RATIO
        and valid_candidates
    ):
        winner = min(valid_candidates, key=lambda row: float(row["median_us"]))
        winner_us = float(winner["median_us"])
        speedup = baseline_us / winner_us
        valid = True
    else:
        winner = None
        winner_us = None
        speedup = None
        valid = False

    return {
        "topology": topology.name,
        "shape": {
            "rows": ROWS,
            "context": context,
            "table_blocks": table_blocks,
            "q_heads": topology.q_heads,
            "kv_heads": topology.kv_heads,
            "head_dim": topology.head_dim,
            "layers": topology.layers,
            "sliding_window": topology.sliding_window,
        },
        "valid": valid,
        "baseline_segments": 16,
        "baseline_us": baseline_us,
        "baseline_stability_ratio": baseline_stability,
        "winner": winner["case"] if winner is not None else None,
        "winner_segments": int(winner["segments"]) if winner is not None else None,
        "winner_us": winner_us,
        "speedup": speedup,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=2111)
    parser.add_argument("--table-blocks", type=int, default=132)
    parser.add_argument("--segment-counts", default="4,8,16,32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_long_decode_attention_tune_a100.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    gpu = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if "a100" not in gpu.lower() or vram_gb < 70.0:
        raise SystemExit(f"A100 80GB required, found {gpu} ({vram_gb:.2f}GB)")
    if args.context <= 0 or math.ceil(args.context / BLOCK_SIZE) > args.table_blocks:
        raise SystemExit("table-blocks is too small for the requested context")
    if args.iterations <= 0 or args.repeats < 3:
        raise SystemExit("iterations must be positive and repeats must be >= 3")

    try:
        segment_counts = tuple(
            dict.fromkeys(int(value) for value in args.segment_counts.split(","))
        )
    except ValueError as exc:
        raise SystemExit("segment-counts must be comma-separated integers") from exc
    if 16 not in segment_counts or any(value not in (4, 8, 16, 32) for value in segment_counts):
        raise SystemExit("segment-counts must include 16 and only use 4,8,16,32")

    print("Gemma4 B16 long-context decode attention tune")
    print("  harness_rev: gemma4-long-decode-attention-tune-v1")
    print(f"  gpu: {gpu}")
    print(f"  torch: {torch.__version__} cuda={torch.version.cuda}")
    print(
        "  workload: "
        f"rows={ROWS} context={args.context} table_blocks={args.table_blocks} "
        f"segments={segment_counts}"
    )
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  measurement: CUDA graph replay")

    results = [
        run_topology(
            topology,
            context=args.context,
            table_blocks=args.table_blocks,
            segment_counts=segment_counts,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        for topology in TOPOLOGIES
    ]
    for result in results:
        print(json.dumps(result, sort_keys=True))

    valid = all(bool(result["valid"]) for result in results)
    if valid:
        baseline_us_per_step = sum(
            float(result["baseline_us"]) * int(result["shape"]["layers"])
            for result in results
        )
        selected_us_per_step = sum(
            float(result["winner_us"]) * int(result["shape"]["layers"])
            for result in results
        )
        weighted_speedup = baseline_us_per_step / selected_us_per_step
        savings_ms_per_step = (baseline_us_per_step - selected_us_per_step) / 1000.0
    else:
        baseline_us_per_step = None
        selected_us_per_step = None
        weighted_speedup = None
        savings_ms_per_step = None

    apply_change = bool(
        valid
        and weighted_speedup is not None
        and weighted_speedup >= args.minimum_speedup
    )
    decision = {
        "decision": "APPLY_LONG_SEGMENTS" if apply_change else "KEEP_SEGMENTS_16",
        "apply_change": apply_change,
        "valid": valid,
        "minimum_speedup": float(args.minimum_speedup),
        "baseline_us_per_step_30_layers": baseline_us_per_step,
        "selected_us_per_step_30_layers": selected_us_per_step,
        "weighted_speedup": weighted_speedup,
        "estimated_savings_ms_per_step": savings_ms_per_step,
        "estimated_savings_ms_per_64_tokens": (
            savings_ms_per_step * 64.0 if savings_ms_per_step is not None else None
        ),
        "estimated_decode_gap_coverage": (
            savings_ms_per_step * 64.0 / TARGET_DECODE_GAP_MS
            if savings_ms_per_step is not None
            else None
        ),
        "target_decode_gap_ms": TARGET_DECODE_GAP_MS,
        "selected_segments": {
            result["topology"]: result["winner_segments"] for result in results
        },
        "topologies": results,
        "runtime": {
            "gpu": gpu,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    print("DECISION " + json.dumps(decision, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())

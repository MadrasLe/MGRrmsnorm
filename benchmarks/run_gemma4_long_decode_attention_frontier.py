#!/usr/bin/env python3
"""Retune long sliding decode segments after promoting the 64-token tile."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

try:
    from run_gemma4_long_decode_attention_shape_tune import (
        BLOCK_SIZE,
        MAX_BASELINE_STABILITY_RATIO,
        ROWS,
        TOPOLOGIES,
        LaunchConfig,
        _numeric_valid,
        _set_launch_env,
        capture_case,
        make_inputs,
        torch_reference,
    )
except ImportError:
    from benchmarks.run_gemma4_long_decode_attention_shape_tune import (
        BLOCK_SIZE,
        MAX_BASELINE_STABILITY_RATIO,
        ROWS,
        TOPOLOGIES,
        LaunchConfig,
        _numeric_valid,
        _set_launch_env,
        capture_case,
        make_inputs,
        torch_reference,
    )


HARNESS_REV = "gemma4-long-decode-attention-frontier-v1"
TOPOLOGY_NAME = "sliding_h256_gqa2"
SLIDING_LAYERS = 25
DECODE_TOKENS = 64
TARGET_REMAINING_DECODE_GAP_MS = 120.58
BASELINE = LaunchConfig(32, 64, 4, 3, 4)
CANDIDATES = (
    BASELINE,
    LaunchConfig(16, 64, 4, 3, 4),
    LaunchConfig(8, 64, 4, 3, 4),
    LaunchConfig(4, 64, 4, 3, 4),
    LaunchConfig(16, 64, 4, 2, 4),
    LaunchConfig(16, 64, 4, 4, 4),
    LaunchConfig(8, 64, 4, 2, 4),
)


def _topology() -> Any:
    return next(topology for topology in TOPOLOGIES if topology.name == TOPOLOGY_NAME)


@torch.inference_mode()
def run_gate(
    *,
    context: int,
    table_blocks: int,
    warmup: int,
    iterations: int,
    repeats: int,
    minimum_speedup: float,
) -> dict[str, Any]:
    from megagemm.kernels.paged_attention import (
        _triton_paged_decode_grouped_segmented_fused,
    )

    topology = _topology()
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

    def measure(config: LaunchConfig, suffix: str = "") -> dict[str, Any]:
        _set_launch_env(config)
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
                num_segments_override=config.segments,
                tile_size_override=config.tile_size,
            )

        row = capture_case(
            name=config.name + suffix,
            call=call,
            out=out,
            reference=reference,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        row["config"] = asdict(config)
        return row

    cases = [measure(config) for config in CANDIDATES]
    cases.append(measure(BASELINE, "_recheck"))
    baseline_rows = (cases[0], cases[-1])
    baseline_valid = all(_numeric_valid(row) for row in baseline_rows)
    if baseline_valid:
        baseline_values = [float(row["median_us"]) for row in baseline_rows]
        baseline_us = min(baseline_values)
        stability = max(baseline_values) / baseline_us
    else:
        baseline_us = None
        stability = None

    candidates = [row for row in cases[:-1] if _numeric_valid(row)]
    valid = bool(
        baseline_us is not None
        and stability is not None
        and stability <= MAX_BASELINE_STABILITY_RATIO
        and candidates
    )
    winner = min(candidates, key=lambda row: float(row["median_us"])) if valid else None
    winner_us = float(winner["median_us"]) if winner is not None else None
    speedup = baseline_us / winner_us if winner_us is not None else None
    apply_change = bool(speedup is not None and speedup >= minimum_speedup)
    selected_config = winner["config"] if apply_change else asdict(BASELINE)
    selected_us = winner_us if apply_change else baseline_us
    savings_ms = (
        (baseline_us - selected_us) * SLIDING_LAYERS * DECODE_TOKENS / 1000.0
        if baseline_us is not None and selected_us is not None
        else None
    )
    return {
        "decision": "APPLY_SLIDING_SEGMENTS" if apply_change else "KEEP_SEG32_TILE64",
        "apply_change": apply_change,
        "valid": valid,
        "minimum_speedup": minimum_speedup,
        "baseline": asdict(BASELINE),
        "baseline_us": baseline_us,
        "baseline_stability_ratio": stability,
        "measured_winner": winner["case"] if winner is not None else None,
        "measured_winner_config": winner["config"] if winner is not None else None,
        "measured_winner_us": winner_us,
        "speedup": speedup,
        "selected_config": selected_config,
        "estimated_savings_ms_per_64_tokens": savings_ms,
        "estimated_remaining_gap_coverage": (
            savings_ms / TARGET_REMAINING_DECODE_GAP_MS
            if savings_ms is not None
            else None
        ),
        "target_remaining_decode_gap_ms": TARGET_REMAINING_DECODE_GAP_MS,
        "shape": {
            "rows": ROWS,
            "context": context,
            "table_blocks": table_blocks,
            "q_heads": topology.q_heads,
            "kv_heads": topology.kv_heads,
            "head_dim": topology.head_dim,
            "sliding_window": topology.sliding_window,
            "layers": SLIDING_LAYERS,
            "decode_tokens": DECODE_TOKENS,
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=2111)
    parser.add_argument("--table-blocks", type=int, default=132)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_long_decode_attention_frontier_a100.json",
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

    print("Gemma4 B16 long sliding decode attention frontier")
    print(f"  harness_rev: {HARNESS_REV}")
    print(f"  gpu: {gpu}")
    print(f"  torch: {torch.__version__} cuda={torch.version.cuda}")
    print(f"  workload: rows={ROWS} context={args.context} table_blocks={args.table_blocks}")
    print("  baseline: segments=32 tile=64 warps=4 stages=3 reduce_warps=4")
    print("  candidates: segments=4/8/16/32 with tile=64; focused stages=2/4")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  measurement: CUDA graph replay")

    decision = run_gate(
        context=args.context,
        table_blocks=args.table_blocks,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        minimum_speedup=args.minimum_speedup,
    )
    decision["runtime"] = {
        "gpu": gpu,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    print("DECISION " + json.dumps(decision, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0 if decision["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

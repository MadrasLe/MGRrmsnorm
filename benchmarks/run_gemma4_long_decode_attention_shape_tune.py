#!/usr/bin/env python3
"""Tune tiles and launch occupancy after the long-context segment promotion."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

try:
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
except ImportError:
    from benchmarks.run_gemma4_grouped_segmented_attention_microbench import (
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

TARGET_DECODE_GAP_MS = 131.2


@dataclass(frozen=True)
class LaunchConfig:
    segments: int
    tile_size: int
    warps: int
    stages: int
    reduce_warps: int

    @property
    def name(self) -> str:
        return (
            f"seg{self.segments}_t{self.tile_size}_w{self.warps}_"
            f"s{self.stages}_r{self.reduce_warps}"
        )


def _promoted_baseline(topology: str) -> LaunchConfig:
    if topology == "sliding_h256_gqa2":
        return LaunchConfig(32, 32, 4, 3, 4)
    if topology == "full_h512_gqa8":
        return LaunchConfig(8, 16, 4, 3, 4)
    raise ValueError(f"unsupported topology: {topology}")


def _candidate_configs(topology: str) -> list[LaunchConfig]:
    baseline = _promoted_baseline(topology)
    candidates = [baseline]
    for tile_size in (16, 32, 64):
        for warps in (4, 8):
            candidates.append(
                LaunchConfig(
                    baseline.segments,
                    tile_size,
                    warps,
                    3,
                    4,
                )
            )
    candidates.extend(
        (
            LaunchConfig(
                baseline.segments,
                baseline.tile_size,
                4,
                2,
                4,
            ),
            LaunchConfig(
                baseline.segments,
                baseline.tile_size,
                4,
                4,
                4,
            ),
            LaunchConfig(
                baseline.segments,
                baseline.tile_size,
                4,
                3,
                8,
            ),
        )
    )
    return list(dict.fromkeys(candidates))


def _set_launch_env(config: LaunchConfig) -> None:
    os.environ["MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_WARPS"] = str(
        config.warps
    )
    os.environ["MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_STAGES"] = str(
        config.stages
    )
    os.environ["MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_REDUCE_WARPS"] = str(
        config.reduce_warps
    )


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

    baseline = _promoted_baseline(topology.name)
    cases = [measure(config) for config in _candidate_configs(topology.name)]
    cases.append(measure(baseline, "_recheck"))

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
        "baseline": asdict(baseline),
        "baseline_us": baseline_us,
        "baseline_stability_ratio": stability,
        "winner": winner["case"] if winner is not None else None,
        "winner_config": winner["config"] if winner is not None else None,
        "winner_us": winner_us,
        "speedup": speedup,
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
        default="bench_results/gemma4_long_decode_attention_shape_a100.json",
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

    print("Gemma4 B16 long-context decode attention shape tune")
    print("  harness_rev: gemma4-long-decode-attention-shape-v1")
    print(f"  gpu: {gpu}")
    print(f"  torch: {torch.__version__} cuda={torch.version.cuda}")
    print(f"  workload: rows={ROWS} context={args.context} table_blocks={args.table_blocks}")
    print("  promoted segments: sliding=32 full=8")
    print("  candidates: tile=16/32/64, warps=4/8, stages=2/3/4, reduce=4/8")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  measurement: CUDA graph replay")

    results = [
        run_topology(
            topology,
            context=args.context,
            table_blocks=args.table_blocks,
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
        selected = []
        for result in results:
            promote_topology = bool(
                result["speedup"] is not None
                and float(result["speedup"]) >= args.minimum_speedup
            )
            selected.append(
                {
                    "topology": result["topology"],
                    "promoted": promote_topology,
                    "config": (
                        result["winner_config"]
                        if promote_topology
                        else result["baseline"]
                    ),
                    "median_us": (
                        float(result["winner_us"])
                        if promote_topology
                        else float(result["baseline_us"])
                    ),
                    "local_speedup": float(result["speedup"]),
                }
            )
        baseline_us = sum(
            float(result["baseline_us"]) * int(result["shape"]["layers"])
            for result in results
        )
        winner_us = sum(
            float(selection["median_us"])
            * int(result["shape"]["layers"])
            for result, selection in zip(results, selected)
        )
        speedup = baseline_us / winner_us
        savings_per_step_ms = (baseline_us - winner_us) / 1000.0
    else:
        baseline_us = None
        winner_us = None
        speedup = None
        savings_per_step_ms = None
        selected = []

    apply_change = bool(
        valid and speedup is not None and speedup >= args.minimum_speedup
    )
    decision = {
        "decision": "APPLY_LONG_LAUNCH_CONFIG" if apply_change else "KEEP_PROMOTED_SEGMENTS",
        "apply_change": apply_change,
        "valid": valid,
        "minimum_speedup": float(args.minimum_speedup),
        "baseline_us_per_step_30_layers": baseline_us,
        "winner_us_per_step_30_layers": winner_us,
        "weighted_speedup": speedup,
        "estimated_savings_ms_per_step": savings_per_step_ms,
        "estimated_savings_ms_per_64_tokens": (
            savings_per_step_ms * 64.0 if savings_per_step_ms is not None else None
        ),
        "estimated_remaining_gap_coverage": (
            savings_per_step_ms * 64.0 / TARGET_DECODE_GAP_MS
            if savings_per_step_ms is not None
            else None
        ),
        "target_remaining_decode_gap_ms": TARGET_DECODE_GAP_MS,
        "selected_configs": {
            selection["topology"]: selection["config"]
            for selection in selected
        },
        "topology_promotions": selected,
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

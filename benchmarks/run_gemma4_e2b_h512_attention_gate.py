#!/usr/bin/env python3
"""Tune the exact Gemma4 E2B/L4 B8 full-attention H512/GQA8 shape."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

try:
    import run_gemma4_grouped_segmented_attention_microbench as shared
except ImportError:
    import benchmarks.run_gemma4_grouped_segmented_attention_microbench as shared


ROWS = 8
LAYERS = 7
BLOCK_SIZE = 16
MAX_BASELINE_STABILITY_RATIO = 1.04
MIN_SPEEDUP = 1.02


@dataclass(frozen=True)
class LaunchConfig:
    segments: int
    tile_size: int
    warps: int = 4
    stages: int = 3
    reduce_warps: int = 4

    @property
    def name(self) -> str:
        return (
            f"seg{self.segments}_t{self.tile_size}_w{self.warps}_"
            f"s{self.stages}_r{self.reduce_warps}"
        )


CONFIGS = tuple(
    LaunchConfig(segments, tile)
    for segments, tile in (
        (4, 16),
        (4, 32),
        (4, 64),
        (8, 16),
        (8, 32),
        (8, 64),
        (16, 16),
        (16, 32),
        (32, 16),
    )
)


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


def _valid(row: dict[str, Any]) -> bool:
    return bool(
        row.get("error") is None
        and float(row["repeat_max_abs_error"]) == 0.0
        and float(row["max_abs_error"]) <= shared.MAX_ABS_ERROR
        and float(row["relative_l2_error"])
        <= shared.MAX_RELATIVE_L2_ERROR
        and float(row["cosine"]) >= shared.MIN_COSINE
    )


@torch.inference_mode()
def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    from megagemm.kernels.paged_attention import (
        _triton_paged_decode_fused,
        _triton_paged_decode_grouped_segmented_fused,
    )

    # The shared reference builder is intentionally reused, but this gate has
    # E2B's B8 topology rather than the A100 harness's B16 topology.
    shared.ROWS = ROWS
    topology = shared.Topology(
        name="e2b_l4_full_h512_gqa8",
        q_heads=8,
        kv_heads=1,
        head_dim=512,
        layers=LAYERS,
        sliding_window=None,
        prior_vllm_us=0.0,
    )
    tensors = shared.make_inputs(
        topology,
        context=args.context,
        table_blocks=args.table_blocks,
        norm_eps=args.norm_eps,
    )
    scale = 1.0 / math.sqrt(topology.head_dim)
    reference = shared.torch_reference(
        query=tensors["query_fp32"],
        kv_cache=tensors["kv_cache"],
        block_table=tensors["block_table"],
        context=args.context,
        scale=scale,
        sliding_window=None,
    )

    def measure_baseline(name: str) -> dict[str, Any]:
        out = torch.empty_like(tensors["raw_query"])

        def call() -> torch.Tensor:
            return _triton_paged_decode_fused(
                tensors["raw_query"],
                tensors["kv_cache"],
                tensors["block_table"],
                tensors["seq_lens"],
                scale,
                tensors["cos"],
                tensors["sin"],
                tensors["positions"],
                half_rotate=True,
                rotary_dim=topology.head_dim,
                q_norm_weight=tensors["norm_weight"],
                norm_eps=args.norm_eps,
                out=out,
                split_policy_override=1,
            )

        return shared.capture_case(
            name=name,
            call=call,
            out=out,
            reference=reference,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )

    def measure_candidate(config: LaunchConfig) -> dict[str, Any]:
        _set_launch_env(config)
        out = torch.empty_like(tensors["raw_query"])

        def call() -> torch.Tensor:
            return _triton_paged_decode_grouped_segmented_fused(
                tensors["raw_query"],
                tensors["kv_cache"],
                tensors["block_table"],
                tensors["seq_lens"],
                scale,
                tensors["cos"],
                tensors["sin"],
                tensors["positions"],
                half_rotate=True,
                rotary_dim=topology.head_dim,
                q_norm_weight=tensors["norm_weight"],
                norm_eps=args.norm_eps,
                out=out,
                force=True,
                num_segments_override=config.segments,
                tile_size_override=config.tile_size,
            )

        row = shared.capture_case(
            name=config.name,
            call=call,
            out=out,
            reference=reference,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        row["config"] = asdict(config)
        return row

    baseline_first = measure_baseline("generic_h512_first")
    candidates = [measure_candidate(config) for config in CONFIGS]
    baseline_recheck = measure_baseline("generic_h512_recheck")
    baseline_rows = (baseline_first, baseline_recheck)
    baseline_valid = all(_valid(row) for row in baseline_rows)
    if baseline_valid:
        baseline_samples = [float(row["median_us"]) for row in baseline_rows]
        baseline_us = min(baseline_samples)
        stability = max(baseline_samples) / baseline_us
    else:
        baseline_us = None
        stability = None
    valid_candidates = [row for row in candidates if _valid(row)]
    valid = bool(
        baseline_us is not None
        and stability is not None
        and stability <= MAX_BASELINE_STABILITY_RATIO
        and valid_candidates
    )
    winner = (
        min(valid_candidates, key=lambda row: float(row["median_us"]))
        if valid
        else None
    )
    winner_us = float(winner["median_us"]) if winner is not None else None
    speedup = baseline_us / winner_us if winner_us is not None else None
    apply_change = bool(speedup is not None and speedup >= args.minimum_speedup)
    return {
        "decision": "TEST_FULL_MODEL" if apply_change else "KEEP_GENERIC_H512",
        "apply_change": apply_change,
        "valid": valid,
        "shape": {
            "rows": ROWS,
            "q_heads": 8,
            "kv_heads": 1,
            "head_dim": 512,
            "layers": LAYERS,
            "context": args.context,
            "table_blocks": args.table_blocks,
            "block_size": BLOCK_SIZE,
            "sliding_window": None,
            "dtype": "bf16",
        },
        "baseline_us": baseline_us,
        "baseline_stability_ratio": stability,
        "winner": winner["case"] if winner is not None else None,
        "winner_config": winner["config"] if winner is not None else None,
        "winner_us": winner_us,
        "speedup": speedup,
        "estimated_savings_ms_per_step_7_layers": (
            (baseline_us - winner_us) * LAYERS / 1000.0
            if baseline_us is not None and winner_us is not None
            else None
        ),
        "minimum_speedup": args.minimum_speedup,
        "baseline": list(baseline_rows),
        "candidates": candidates,
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=2175)
    parser.add_argument("--table-blocks", type=int, default=144)
    parser.add_argument("--norm-eps", type=float, default=1.0e-6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=MIN_SPEEDUP)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_e2b_h512_attention_l4.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    gpu = torch.cuda.get_device_name(0)
    if "l4" not in gpu.lower():
        raise SystemExit(f"NVIDIA L4 required, found {gpu}")
    if args.context <= 0 or math.ceil(args.context / BLOCK_SIZE) > args.table_blocks:
        raise SystemExit("table-blocks is too small for the requested context")
    if args.iterations <= 0 or args.repeats < 3:
        raise SystemExit("iterations must be positive and repeats must be >= 3")

    os.environ["MEGAGEMM_GEMMA4_E2B_L4_H512_GROUPED_ATTN_DECODE"] = "0"
    os.environ["MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_DECODE"] = "0"
    os.environ["MEGAGEMM_PAGED_DECODE_GQA2"] = "0"
    print("Gemma4 E2B/L4 full H512/GQA8 attention gate")
    print(f"  gpu: {gpu}")
    print("  shape: B8 Q8/KV1 H512 full attention, 7 layers")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    result = run_gate(args)
    print("DECISION " + json.dumps(result, sort_keys=True))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired LM-head launch sweep for the exact Gemma 4 E2B/L4 decode shape.

All configurations share one allocation and one Python process. Each
candidate is measured against the production configuration in alternating
order, which removes most L4 clock drift from the comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


@dataclass(frozen=True)
class LaunchConfig:
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int = 2

    @property
    def name(self) -> str:
        return (
            f"bn{self.block_n}_bk{self.block_k}_"
            f"w{self.num_warps}_s{self.num_stages}"
        )


CURRENT = LaunchConfig(256, 128, 4, 2)
CANDIDATES = (
    LaunchConfig(64, 64, 4),
    LaunchConfig(64, 128, 4),
    LaunchConfig(64, 128, 8),
    LaunchConfig(128, 64, 4),
    LaunchConfig(128, 64, 8),
    LaunchConfig(128, 128, 4),
    LaunchConfig(128, 128, 8),
    LaunchConfig(256, 64, 4),
    LaunchConfig(256, 64, 8),
    LaunchConfig(256, 128, 8),
    LaunchConfig(256, 128, 4, 1),
    LaunchConfig(256, 128, 4, 3),
    LaunchConfig(512, 64, 8),
)


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _valid_candidate(
    row: dict[str, Any],
    *,
    minimum_speedup: float,
    maximum_ratio_spread: float,
    minimum_faster_fraction: float,
) -> bool:
    pair_speedups = [float(value) for value in row.get("pair_speedups") or []]
    required_faster = math.ceil(len(pair_speedups) * minimum_faster_fraction)
    return bool(
        row.get("error") is None
        and row.get("tokens_equal") is True
        and pair_speedups
        and float(row.get("median_speedup") or 0.0) >= minimum_speedup
        and float(row.get("speedup_spread_ratio") or float("inf"))
        <= maximum_ratio_spread
        and int(row.get("candidate_faster_pairs") or 0) >= required_faster
    )


def decide_paired_lm_head_sweep(
    rows: list[dict[str, Any]],
    *,
    minimum_speedup: float,
    maximum_ratio_spread: float,
    minimum_faster_fraction: float,
) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if _valid_candidate(
            row,
            minimum_speedup=minimum_speedup,
            maximum_ratio_spread=maximum_ratio_spread,
            minimum_faster_fraction=minimum_faster_fraction,
        )
    ]
    valid.sort(key=lambda row: float(row["median_speedup"]), reverse=True)
    winner = valid[0] if valid else None
    speedup = float(winner["median_speedup"]) if winner is not None else None
    current_us = (
        float(winner["current_median_us"]) if winner is not None else None
    )
    candidate_us = (
        float(winner["candidate_median_us"]) if winner is not None else None
    )
    return {
        "decision": (
            f"TEST_FULL_MODEL_{str(winner['name']).upper()}"
            if winner is not None
            else "KEEP_CURRENT_LM_HEAD_CONFIG"
        ),
        "apply_change": winner is not None,
        "current": CURRENT.name,
        "winner": winner.get("name") if winner is not None else None,
        "current_us": current_us,
        "winner_us": candidate_us,
        "speedup": speedup,
        "minimum_speedup": minimum_speedup,
        "maximum_ratio_spread": maximum_ratio_spread,
        "minimum_faster_fraction": minimum_faster_fraction,
        "estimated_saved_ms_per_128_decode_steps": (
            (current_us - candidate_us) * 128.0 / 1000.0
            if current_us is not None and candidate_us is not None
            else None
        ),
    }


def _set_config(module: Any, config: LaunchConfig) -> None:
    module._CFG_FORCED_BN = int(config.block_n)
    module._CFG_FORCED_BK = int(config.block_k)
    module._CFG_FORCED_WARPS = int(config.num_warps)
    module._CFG_FORCED_STAGES = int(config.num_stages)
    module._CFG_TRITON_REDUCE = True


def _measure_once_us(
    module: Any,
    config: LaunchConfig,
    fn: Any,
    *,
    iterations: int,
) -> float:
    import torch

    _set_config(module, config)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / float(iterations)


def _warm_config(module: Any, config: LaunchConfig, fn: Any, warmups: int) -> None:
    import torch

    _set_config(module, config)
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()


def _measure_candidate(
    module: Any,
    config: LaunchConfig,
    fn: Any,
    reference: Any,
    *,
    warmups: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    import torch

    row: dict[str, Any] = {
        "name": config.name,
        "config": asdict(config),
        "tokens_equal": False,
        "current_samples_us": [],
        "candidate_samples_us": [],
        "pair_speedups": [],
        "candidate_faster_pairs": 0,
        "error": None,
    }
    try:
        _set_config(module, config)
        actual = fn().clone()
        torch.cuda.synchronize()
        row["tokens_equal"] = bool(torch.equal(actual, reference))

        _warm_config(module, CURRENT, fn, warmups)
        _warm_config(module, config, fn, warmups)
        for pair in range(repeats):
            order = (CURRENT, config) if pair % 2 == 0 else (config, CURRENT)
            measured: dict[str, float] = {}
            for selected in order:
                key = "current" if selected == CURRENT else "candidate"
                measured[key] = _measure_once_us(
                    module,
                    selected,
                    fn,
                    iterations=iterations,
                )
            current_us = measured["current"]
            candidate_us = measured["candidate"]
            row["current_samples_us"].append(current_us)
            row["candidate_samples_us"].append(candidate_us)
            row["pair_speedups"].append(current_us / candidate_us)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"

    current_samples = [float(value) for value in row["current_samples_us"]]
    candidate_samples = [float(value) for value in row["candidate_samples_us"]]
    speedups = [float(value) for value in row["pair_speedups"]]
    row.update(
        {
            "current_median_us": _median(current_samples),
            "candidate_median_us": _median(candidate_samples),
            "median_speedup": _median(speedups),
            "speedup_spread_ratio": (
                max(speedups) / min(speedups) if speedups else None
            ),
            "candidate_faster_pairs": sum(value > 1.0 for value in speedups),
        }
    )
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu = torch.cuda.get_device_name(0)
    if "L4" not in gpu.upper():
        raise RuntimeError(f"this gate requires NVIDIA L4, found {gpu}")
    if (args.rows, args.hidden_size, args.vocab_size, args.dtype) != (
        8,
        1536,
        262144,
        "bf16",
    ):
        raise RuntimeError(
            "this promotion gate requires rows=8 hidden=1536 "
            "vocab=262144 BF16"
        )

    module = importlib.import_module("megagemm.kernels.lm_head_argmax")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    hidden = torch.randn(
        args.rows,
        args.hidden_size,
        device="cuda",
        dtype=dtype,
    ) * 0.02
    # Gemma RMSNorm stores the delta from one and applies offset=True.
    norm_weight = torch.randn(
        args.hidden_size,
        device="cuda",
        dtype=dtype,
    ) * 0.02
    weight = torch.randn(
        args.vocab_size,
        args.hidden_size,
        device="cuda",
        dtype=dtype,
    ) * 0.02
    out_tokens = torch.empty(args.rows, device="cuda", dtype=torch.long)
    n_blocks_cap = (args.vocab_size + 15) // 16
    partial_vals = torch.empty(
        args.rows,
        n_blocks_cap,
        device="cuda",
        dtype=torch.float32,
    )
    partial_idxs = torch.empty(
        args.rows,
        n_blocks_cap,
        device="cuda",
        dtype=torch.int32,
    )

    def kernel_call():
        return module.lm_head_rmsnorm_argmax(
            hidden,
            norm_weight,
            args.eps,
            True,
            weight,
            out_tokens=out_tokens,
            partial_vals=partial_vals,
            partial_idxs=partial_idxs,
        )

    rms = torch.rsqrt(
        hidden.float().pow(2).mean(dim=-1, keepdim=True) + float(args.eps)
    )
    normed = (hidden * rms * (norm_weight.float() + 1.0)).to(dtype=dtype)
    reference = torch.nn.functional.linear(normed, weight).argmax(dim=-1)
    torch.cuda.synchronize()

    print("Gemma 4 E2B/L4 paired fused RMSNorm LM-head sweep")
    print(f"  gpu: {gpu}")
    print("  shape: rows=8 hidden=1536 vocab=262144 BF16 offset-RMSNorm")
    print(f"  current: {CURRENT.name}")
    print("  one allocation: yes")
    print("  alternating paired measurements: yes")
    print("  model_download: disabled")
    print("  competing_engine_install: disabled")

    rows: list[dict[str, Any]] = []
    for index, config in enumerate(CANDIDATES, start=1):
        print(f"\n=== {index}/{len(CANDIDATES)} {config.name} ===", flush=True)
        row = _measure_candidate(
            module,
            config,
            kernel_call,
            reference,
            warmups=args.warmups,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        rows.append(row)
        print("CASE", json.dumps(row, sort_keys=True), flush=True)

    decision = decide_paired_lm_head_sweep(
        rows,
        minimum_speedup=args.minimum_speedup,
        maximum_ratio_spread=args.maximum_ratio_spread,
        minimum_faster_fraction=args.minimum_faster_fraction,
    )
    payload = {
        "benchmark": "gemma4_e2b_paired_lm_head_sweep",
        "gpu": gpu,
        "torch": torch.__version__,
        "shape": {
            "rows": args.rows,
            "hidden_size": args.hidden_size,
            "vocab_size": args.vocab_size,
            "dtype": args.dtype,
            "norm_offset": True,
        },
        "rows": rows,
        **decision,
    }

    print("\nRANKING — PAIRED SPEEDUP")
    ranked = [row for row in rows if row.get("median_speedup") is not None]
    ranked.sort(key=lambda row: float(row["median_speedup"]), reverse=True)
    for index, row in enumerate(ranked, 1):
        print(
            f"{index:02d}. {row['name']}: "
            f"{float(row['median_speedup']):.4f}x "
            f"current={float(row['current_median_us']):.3f}us "
            f"candidate={float(row['candidate_median_us']):.3f}us "
            f"faster={int(row['candidate_faster_pairs'])}/{args.repeats} "
            f"ratio_spread={float(row['speedup_spread_ratio']):.4f}"
        )
    print("\nDECISION", json.dumps(decision, sort_keys=True))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1536)
    parser.add_argument("--vocab-size", type=int, default=262144)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--minimum-speedup", type=float, default=1.03)
    parser.add_argument("--maximum-ratio-spread", type=float, default=1.08)
    parser.add_argument("--minimum-faster-fraction", type=float, default=0.80)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmups < 1 or args.iterations < 1 or args.repeats < 3:
        raise SystemExit("use warmups>=1, iterations>=1, and repeats>=3")
    if not 0.5 <= args.minimum_faster_fraction <= 1.0:
        raise SystemExit("minimum-faster-fraction must be in [0.5, 1.0]")
    payload = run(args)
    return 0 if any(row.get("median_speedup") is not None for row in payload["rows"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())

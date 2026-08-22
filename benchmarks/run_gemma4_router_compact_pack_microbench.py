#!/usr/bin/env python3
"""Gate the exact Gemma4 B16 top-k plus compact route-pack fusion."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megagemm.kernels.qwen3_moe import (
    qwen3_moe_compact_route_pack,
    qwen3_moe_topk_softmax,
    qwen3_moe_topk_softmax_compact_pack,
)


def _measure(
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
    return samples


def _snapshot(
    weights: torch.Tensor,
    experts: torch.Tensor,
    pack: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    counts, dense_tokens, dense_route, dense_assign = pack
    rows = int(weights.shape[0])
    valid = torch.arange(rows, device=counts.device)[None, :] < counts[:, None]
    return {
        "weights": weights.clone(),
        "experts": experts.clone(),
        "counts": counts.clone(),
        "tokens": dense_tokens[valid].clone(),
        "route": dense_route[valid].clone(),
        "assign": dense_assign[valid].clone(),
    }


def _exact(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> bool:
    return lhs.keys() == rhs.keys() and all(
        torch.equal(lhs[name], rhs[name]) for name in lhs
    )


def _max_abs_error(
    lhs: dict[str, torch.Tensor],
    rhs: dict[str, torch.Tensor],
) -> float:
    errors = []
    for name in lhs:
        left = lhs[name]
        right = rhs[name]
        if left.shape != right.shape:
            return float("inf")
        if left.numel() == 0:
            continue
        errors.append(float((left.float() - right.float()).abs().max().item()))
    return max(errors, default=0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--target-gap-ms", type=float, default=0.85)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_router_compact_pack_a100.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.rows != 16:
        raise SystemExit("this gate is deliberately restricted to --rows 16")
    if "A100" not in torch.cuda.get_device_name(0).upper():
        raise SystemExit("this gate is deliberately restricted to NVIDIA A100")
    torch.set_grad_enabled(False)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    experts_count = 128
    top_k = 8
    layers = 30
    torch.manual_seed(111)

    logits = torch.randn(args.rows, experts_count, device=device, dtype=dtype)
    expert_scale = torch.randn(experts_count, device=device, dtype=dtype)
    baseline_topk_workspace: dict[str, torch.Tensor] = {}
    baseline_pack_workspace: dict[str, torch.Tensor] = {}
    fused_topk_workspace: dict[str, torch.Tensor] = {}
    fused_pack_workspace: dict[str, torch.Tensor] = {}

    def current():
        weights, selected = qwen3_moe_topk_softmax(
            logits,
            top_k,
            workspace=baseline_topk_workspace,
            expert_scale=expert_scale,
        )
        pack = qwen3_moe_compact_route_pack(
            selected,
            weights,
            num_experts=experts_count,
            workspace=baseline_pack_workspace,
        )
        return weights, selected, pack

    def fused():
        weights, selected = qwen3_moe_topk_softmax_compact_pack(
            logits,
            top_k,
            workspace=fused_topk_workspace,
            compact_workspace=fused_pack_workspace,
            expert_scale=expert_scale,
        )
        pack = (
            fused_pack_workspace["expert_grouped_compact_counts"],
            fused_pack_workspace["expert_grouped_compact_tokens"],
            fused_pack_workspace["expert_grouped_compact_route"],
            fused_pack_workspace["expert_grouped_compact_assign"],
        )
        return weights, selected, pack

    error = None
    exact = False
    repeat_exact = False
    max_abs_error = None
    try:
        reference = _snapshot(*current())
        candidate = _snapshot(*fused())
        candidate_repeat = _snapshot(*fused())
        torch.cuda.synchronize()
        exact = _exact(reference, candidate)
        repeat_exact = _exact(candidate, candidate_repeat)
        max_abs_error = _max_abs_error(reference, candidate)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    rows_out = []
    cases = (
        ("current_topk_plus_128cta_pack", current),
        ("fused_persistent_topk_pack", fused),
        ("current_recheck", current),
    )
    for name, fn in cases:
        case_error = error if name == "fused_persistent_topk_pack" else None
        samples: list[float] = []
        if case_error is None:
            try:
                samples = _measure(
                    fn,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
            except Exception as exc:
                case_error = f"{type(exc).__name__}: {exc}"
        median_us = float(statistics.median(samples)) if samples else None
        row = {
            "case": name,
            "median_us": median_us,
            "samples_us": samples,
            "error": case_error,
            "exact": exact,
            "repeat_exact": repeat_exact,
            "max_abs_error": max_abs_error,
            "ms_per_token_30_layers": (
                median_us * layers / 1000.0 if median_us is not None else None
            ),
        }
        rows_out.append(row)
        print(json.dumps(row, sort_keys=True))

    current_values = [
        row["median_us"]
        for row in rows_out
        if row["case"].startswith("current") and row["median_us"] is not None
    ]
    fused_row = next(
        row for row in rows_out if row["case"] == "fused_persistent_topk_pack"
    )
    baseline_us = min(current_values) if current_values else None
    stability_ratio = (
        max(current_values) / baseline_us
        if baseline_us is not None and baseline_us > 0.0
        else None
    )
    fused_us = fused_row["median_us"]
    speedup = (
        baseline_us / fused_us
        if baseline_us is not None and fused_us is not None and fused_us > 0.0
        else None
    )
    savings_ms = (
        (baseline_us - fused_us) * layers / 1000.0
        if baseline_us is not None and fused_us is not None
        else None
    )
    apply_change = bool(
        exact
        and repeat_exact
        and max_abs_error == 0.0
        and fused_row["error"] is None
        and stability_ratio is not None
        and stability_ratio <= 1.03
        and speedup is not None
        and speedup >= float(args.minimum_speedup)
    )
    summary = {
        "decision": "APPLY" if apply_change else "KEEP_BASELINE",
        "apply_change": apply_change,
        "gpu": torch.cuda.get_device_name(0),
        "shape": {
            "rows": args.rows,
            "experts": experts_count,
            "top_k": top_k,
            "layers": layers,
            "dtype": "bf16",
        },
        "baseline_us": baseline_us,
        "fused_us": fused_us,
        "speedup": speedup,
        "baseline_stability_ratio": stability_ratio,
        "minimum_speedup": float(args.minimum_speedup),
        "estimated_savings_ms_per_token_30_layers": savings_ms,
        "target_gap_ms": float(args.target_gap_ms),
        "estimated_gap_coverage": (
            savings_ms / float(args.target_gap_ms)
            if savings_ms is not None and float(args.target_gap_ms) > 0.0
            else None
        ),
        "exact": exact,
        "repeat_exact": repeat_exact,
        "max_abs_error": max_abs_error,
        "ranking": sorted(
            rows_out,
            key=lambda item: (
                item["median_us"] is None,
                item["median_us"] if item["median_us"] is not None else float("inf"),
            ),
        ),
    }
    print("DECISION " + json.dumps(summary, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Gate the shared-variance Gemma4 router/expert input normalization."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "megagemm" / "kernels" / "rmsnorm_triton.py"
SPEC = importlib.util.spec_from_file_location("megagemm_rmsnorm_dual_gate", KERNEL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load RMSNorm kernel from {KERNEL_PATH}")
KERNEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KERNEL)
rmsnorm_triton = KERNEL.rmsnorm_triton
rmsnorm_triton_scaled_no_weight = KERNEL.rmsnorm_triton_scaled_no_weight
rmsnorm_triton_weighted_scaled_no_weight_dual = (
    KERNEL.rmsnorm_triton_weighted_scaled_no_weight_dual
)


def _measure(fn, *, warmup: int, iterations: int, repeats: int) -> list[float]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_router_expert_input_norm_a100.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.rows != 16:
        raise SystemExit("this gate is deliberately restricted to --rows 16")
    torch.set_grad_enabled(False)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden = 2816
    eps = 1e-6
    output_scale = hidden ** -0.5
    torch.manual_seed(61)
    x = torch.randn(args.rows, hidden, device=device, dtype=dtype)
    weight = torch.randn(hidden, device=device, dtype=dtype)
    scale = torch.randn(hidden, device=device, dtype=dtype)
    fused_weighted_out = torch.empty_like(x)
    fused_scaled_out = torch.empty_like(x)

    def current():
        weighted = rmsnorm_triton(x, weight, eps, False)
        scaled = rmsnorm_triton_scaled_no_weight(x, scale, eps, output_scale)
        return weighted, scaled

    def fused():
        return rmsnorm_triton_weighted_scaled_no_weight_dual(
            x,
            weight,
            scale,
            eps,
            output_scale,
            weighted_out=fused_weighted_out,
            scaled_out=fused_scaled_out,
        )

    reference_weighted, reference_scaled = current()
    candidate_weighted, candidate_scaled = fused()
    torch.cuda.synchronize()
    weighted_exact = bool(torch.equal(reference_weighted, candidate_weighted))
    scaled_exact = bool(torch.equal(reference_scaled, candidate_scaled))
    weighted_max_error = float(
        (reference_weighted.float() - candidate_weighted.float()).abs().max().item()
    )
    scaled_max_error = float(
        (reference_scaled.float() - candidate_scaled.float()).abs().max().item()
    )

    rows_out = []
    for name, fn in (
        ("current_two_kernels", current),
        ("fused_shared_variance", fused),
        ("current_recheck", current),
    ):
        samples = _measure(
            fn,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        median_us = float(statistics.median(samples))
        row = {
            "case": name,
            "median_us": median_us,
            "samples_us": samples,
            "ms_per_token_30_layers": median_us * 30.0 / 1000.0,
            "weighted_exact": weighted_exact,
            "scaled_exact": scaled_exact,
            "weighted_max_error": weighted_max_error,
            "scaled_max_error": scaled_max_error,
        }
        rows_out.append(row)
        print(json.dumps(row, sort_keys=True))

    current_values = [
        row["median_us"] for row in rows_out if row["case"].startswith("current")
    ]
    baseline_us = min(current_values)
    stability_ratio = max(current_values) / baseline_us
    fused_us = next(
        row["median_us"]
        for row in rows_out
        if row["case"] == "fused_shared_variance"
    )
    speedup = baseline_us / fused_us
    correct = bool(
        weighted_exact
        and scaled_exact
        and weighted_max_error == 0.0
        and scaled_max_error == 0.0
    )
    apply_change = bool(
        correct
        and stability_ratio <= 1.03
        and speedup >= float(args.minimum_speedup)
    )
    summary = {
        "decision": "APPLY" if apply_change else "KEEP_BASELINE",
        "apply_change": apply_change,
        "gpu": torch.cuda.get_device_name(0),
        "shape": {"rows": args.rows, "hidden": hidden, "dtype": "bf16"},
        "baseline_us": baseline_us,
        "fused_us": fused_us,
        "speedup": speedup,
        "baseline_stability_ratio": stability_ratio,
        "minimum_speedup": float(args.minimum_speedup),
        "estimated_savings_ms_per_token_30_layers": (
            baseline_us - fused_us
        ) * 30.0 / 1000.0,
        "weighted_exact": weighted_exact,
        "scaled_exact": scaled_exact,
        "weighted_max_error": weighted_max_error,
        "scaled_max_error": scaled_max_error,
        "ranking": sorted(rows_out, key=lambda item: item["median_us"]),
    }
    print("DECISION " + json.dumps(summary, sort_keys=True))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

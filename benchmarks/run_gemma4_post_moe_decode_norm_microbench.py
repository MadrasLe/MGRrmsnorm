#!/usr/bin/env python3
"""Gate the fused Gemma4 post-MoE normalization and residual chain."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "megagemm" / "kernels" / "rmsnorm_triton.py"
SPEC = importlib.util.spec_from_file_location("megagemm_rmsnorm_gate", KERNEL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load RMSNorm kernel from {KERNEL_PATH}")
KERNEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KERNEL)
rmsnorm_triton = KERNEL.rmsnorm_triton
rmsnorm_triton_pair_add_final_residual = (
    KERNEL.rmsnorm_triton_pair_add_final_residual
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
        default="bench_results/gemma4_post_moe_decode_norm_a100.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.rows not in (1, 8, 16):
        raise SystemExit("--rows must be one of 1, 8, or 16")
    torch.set_grad_enabled(False)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden = 2816
    eps = 1e-6
    torch.manual_seed(59)
    shared = torch.randn(args.rows, hidden, device=device, dtype=dtype)
    expert = torch.randn_like(shared)
    residual = torch.randn_like(shared)
    shared_weight = torch.randn(hidden, device=device, dtype=dtype)
    expert_weight = torch.randn(hidden, device=device, dtype=dtype)
    final_weight = torch.randn(hidden, device=device, dtype=dtype)
    baseline_out = torch.empty_like(shared)
    fused_out = torch.empty_like(shared)

    def current():
        shared_norm = rmsnorm_triton(shared, shared_weight, eps, False)
        expert_norm = rmsnorm_triton(expert, expert_weight, eps, False)
        summed = (shared_norm + expert_norm).to(dtype)
        final_norm = rmsnorm_triton(summed, final_weight, eps, False)
        return torch.add(residual, final_norm, out=baseline_out)

    def fused():
        return rmsnorm_triton_pair_add_final_residual(
            shared,
            expert,
            shared_weight,
            expert_weight,
            final_weight,
            residual,
            eps,
            out=fused_out,
        )

    reference = current().clone()
    candidate = fused().clone()
    aliased_residual = residual.clone()
    aliased = rmsnorm_triton_pair_add_final_residual(
        shared,
        expert,
        shared_weight,
        expert_weight,
        final_weight,
        aliased_residual,
        eps,
        out=aliased_residual,
    ).clone()
    torch.cuda.synchronize()
    cosine = float(
        torch.nn.functional.cosine_similarity(
            reference.float().reshape(1, -1),
            candidate.float().reshape(1, -1),
        ).item()
    )
    max_abs_error = float((reference.float() - candidate.float()).abs().max().item())
    exact = bool(torch.equal(reference, candidate))
    alias_exact = bool(torch.equal(reference, aliased))

    rows_out = []
    for name, fn in (
        ("current_five_kernels", current),
        ("fused_one_kernel", fused),
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
            "exact": exact,
            "alias_exact": alias_exact,
            "cosine": cosine,
            "max_abs_error": max_abs_error,
        }
        rows_out.append(row)
        print(json.dumps(row, sort_keys=True))

    current_values = [
        row["median_us"] for row in rows_out if row["case"].startswith("current")
    ]
    baseline_us = min(current_values)
    stability_ratio = max(current_values) / baseline_us
    fused_us = next(
        row["median_us"] for row in rows_out if row["case"] == "fused_one_kernel"
    )
    speedup = baseline_us / fused_us
    correct = bool(exact and alias_exact and cosine >= 0.99999 and max_abs_error == 0.0)
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
        "exact": exact,
        "alias_exact": alias_exact,
        "cosine": cosine,
        "max_abs_error": max_abs_error,
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

#!/usr/bin/env python3
"""Gate the Gemma4 B16 attention-to-MoE/router decode bridge."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "megagemm" / "kernels" / "rmsnorm_triton.py"
SPEC = importlib.util.spec_from_file_location(
    "megagemm_rmsnorm_attn_moe_decode_gate",
    KERNEL_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load RMSNorm kernel from {KERNEL_PATH}")
KERNEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KERNEL)
rmsnorm_triton_attn_residual_router_bridge = (
    KERNEL.rmsnorm_triton_attn_residual_router_bridge
)
rmsnorm_triton_attn_residual_router_bridge_single = (
    KERNEL.rmsnorm_triton_attn_residual_router_bridge_single
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


def _max_abs_error(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--target-gap-ms", type=float, default=0.706)
    parser.add_argument(
        "--out-json",
        default=(
            "bench_results/"
            "gemma4_attn_moe_router_single_kernel_decode_a100.json"
        ),
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
    layers = 30
    eps = 1e-6
    torch.manual_seed(86)

    attn_out = torch.randn(args.rows, hidden, device=device, dtype=dtype)
    residual = torch.randn_like(attn_out)
    post_weight = torch.randn(hidden, device=device, dtype=dtype)
    shared_weight = torch.randn(hidden, device=device, dtype=dtype)
    expert_weight = torch.randn(hidden, device=device, dtype=dtype)
    router_scale = torch.randn(hidden, device=device, dtype=dtype)
    router_output_scale = hidden ** -0.5

    baseline_hidden = torch.empty_like(attn_out)
    baseline_post = torch.empty_like(attn_out)
    baseline_shared = torch.empty_like(attn_out)
    baseline_expert = torch.empty_like(attn_out)
    baseline_router = torch.empty_like(attn_out)
    candidate_post = torch.empty_like(attn_out)
    candidate_hidden = torch.empty_like(attn_out)
    candidate_shared = torch.empty_like(attn_out)
    candidate_expert = torch.empty_like(attn_out)
    candidate_router = torch.empty_like(attn_out)

    def baseline():
        return rmsnorm_triton_attn_residual_router_bridge(
            attn_out,
            residual,
            post_weight,
            shared_weight,
            expert_weight,
            router_scale,
            eps,
            router_output_scale,
            out_hidden=baseline_hidden,
            post_norm_out=baseline_post,
            shared_out=baseline_shared,
            expert_out=baseline_expert,
            router_out=baseline_router,
        )

    def candidate():
        return rmsnorm_triton_attn_residual_router_bridge_single(
            attn_out,
            residual,
            post_weight,
            shared_weight,
            expert_weight,
            router_scale,
            eps,
            router_output_scale,
            out_hidden=candidate_hidden,
            post_norm_out=candidate_post,
            shared_out=candidate_shared,
            expert_out=candidate_expert,
            router_out=candidate_router,
        )

    reference = tuple(value.clone() for value in baseline()) + (
        baseline_post.clone(),
    )
    candidate_values = tuple(value.clone() for value in candidate()) + (
        candidate_post.clone(),
    )
    aliased_residual = residual.clone()
    aliased_post = torch.empty_like(attn_out)
    aliased = tuple(
        value.clone()
        for value in rmsnorm_triton_attn_residual_router_bridge_single(
            attn_out,
            aliased_residual,
            post_weight,
            shared_weight,
            expert_weight,
            router_scale,
            eps,
            router_output_scale,
            out_hidden=aliased_residual,
            post_norm_out=aliased_post,
        )
    ) + (aliased_post.clone(),)
    torch.cuda.synchronize()

    exact = all(
        torch.equal(lhs, rhs) for lhs, rhs in zip(reference, candidate_values)
    )
    alias_exact = all(torch.equal(lhs, rhs) for lhs, rhs in zip(reference, aliased))
    max_abs_error = max(
        _max_abs_error(lhs, rhs) for lhs, rhs in zip(reference, candidate_values)
    )
    alias_max_abs_error = max(
        _max_abs_error(lhs, rhs) for lhs, rhs in zip(reference, aliased)
    )

    rows_out = []
    for name, fn in (
        ("fused_two_kernel_router_bridge", baseline),
        ("fused_one_kernel_router_bridge", candidate),
        ("fused_two_kernel_router_bridge_recheck", baseline),
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
            "ms_per_token_30_layers": median_us * layers / 1000.0,
            "exact": exact,
            "alias_exact": alias_exact,
            "max_abs_error": max_abs_error,
            "alias_max_abs_error": alias_max_abs_error,
        }
        rows_out.append(row)
        print(json.dumps(row, sort_keys=True))

    baseline_values = [
        row["median_us"]
        for row in rows_out
        if row["case"].startswith("fused_two_kernel_router_bridge")
    ]
    baseline_us = min(baseline_values)
    stability_ratio = max(baseline_values) / baseline_us
    fused_us = next(
        row["median_us"]
        for row in rows_out
        if row["case"] == "fused_one_kernel_router_bridge"
    )
    speedup = baseline_us / fused_us
    savings_ms = (baseline_us - fused_us) * layers / 1000.0
    correct = bool(
        exact
        and alias_exact
        and max_abs_error == 0.0
        and alias_max_abs_error == 0.0
    )
    apply_change = bool(
        correct
        and stability_ratio <= 1.03
        and speedup >= float(args.minimum_speedup)
    )
    summary = {
        "decision": (
            "APPLY_SINGLE_KERNEL" if apply_change else "KEEP_TWO_KERNEL"
        ),
        "apply_change": apply_change,
        "gpu": torch.cuda.get_device_name(0),
        "shape": {
            "rows": args.rows,
            "hidden": hidden,
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
            if float(args.target_gap_ms) > 0.0
            else None
        ),
        "exact": exact,
        "alias_exact": alias_exact,
        "max_abs_error": max_abs_error,
        "alias_max_abs_error": alias_max_abs_error,
        "ranking": sorted(rows_out, key=lambda item: item["median_us"]),
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

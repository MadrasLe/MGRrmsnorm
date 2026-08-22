#!/usr/bin/env python3
"""Measure the Gemma4 MoE post-FFN normalization chain without model weights."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megagemm.kernels.rmsnorm_triton import (  # noqa: E402
    rmsnorm_triton,
    rmsnorm_triton_add,
    rmsnorm_triton_pair_add_final,
)


def _measure(fn, *, warmup: int, iterations: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        graph = torch.cuda.CUDAGraph()
        static_out = None
        with torch.cuda.graph(graph):
            static_out = fn()
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
        del static_out
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_post_ffn_norm_a100.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.set_grad_enabled(False)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    rows, hidden = 25, 2816
    eps = 1e-6
    torch.manual_seed(7)
    shared = torch.randn(rows, hidden, device=device, dtype=dtype)
    expert = torch.randn(rows, hidden, device=device, dtype=dtype)
    shared_weight = torch.randn(hidden, device=device, dtype=dtype)
    expert_weight = torch.randn(hidden, device=device, dtype=dtype)
    final_weight = torch.randn(hidden, device=device, dtype=dtype)

    def current():
        shared_norm = rmsnorm_triton(shared, shared_weight, eps, False)
        expert_norm = rmsnorm_triton(expert, expert_weight, eps, False)
        return rmsnorm_triton_add(
            shared_norm,
            expert_norm,
            final_weight,
            eps,
        )

    def fused():
        return rmsnorm_triton_pair_add_final(
            shared,
            expert,
            shared_weight,
            expert_weight,
            final_weight,
            eps,
        )

    reference = current()
    candidate = fused()
    torch.cuda.synchronize()
    cosine = float(torch.nn.functional.cosine_similarity(
        reference.float().reshape(1, -1),
        candidate.float().reshape(1, -1),
    ).item())
    max_abs_error = float((reference.float() - candidate.float()).abs().max().item())

    rows_out = []
    for name, fn in (("current_three_kernels", current), ("fused_one_kernel", fused)):
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
            "ms_per_prefill_30_layers": median_us * 30.0 / 1000.0,
            "cosine": cosine,
            "max_abs_error": max_abs_error,
        }
        rows_out.append(row)
        print(json.dumps(row, sort_keys=True))

    ranking = sorted(rows_out, key=lambda item: item["median_us"])
    current_us = next(item["median_us"] for item in rows_out if item["case"] == "current_three_kernels")
    fused_us = next(item["median_us"] for item in rows_out if item["case"] == "fused_one_kernel")
    speedup = current_us / fused_us
    correct = bool(cosine >= 0.9999 and max_abs_error <= 0.03125)
    decision = "APPLY" if correct and speedup >= 1.05 else "REVERT"
    summary = {
        "decision": decision,
        "gpu": torch.cuda.get_device_name(0),
        "shape": {"rows": rows, "hidden": hidden, "dtype": "bf16"},
        "speedup": speedup,
        "estimated_savings_ms_per_prefill": (current_us - fused_us) * 30.0 / 1000.0,
        "cosine": cosine,
        "max_abs_error": max_abs_error,
        "ranking": ranking,
    }
    print("DECISION " + json.dumps(summary, sort_keys=True))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

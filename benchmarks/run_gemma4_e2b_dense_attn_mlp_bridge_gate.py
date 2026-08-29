#!/usr/bin/env python3
"""Gate the one-launch Gemma4 E2B dense attention-to-MLP bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Callable

import torch


ROWS = 8
HIDDEN = 1536
LAYERS = 35
MIN_SPEEDUP = 1.02
MAX_STABILITY_RATIO = 1.04


def _measure_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    iterations: int,
    repeats: int,
) -> list[float]:
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
    return samples


def _capture_case(
    name: str,
    call: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    reference_hidden: torch.Tensor,
    reference_pre_ff: torch.Tensor,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    try:
        outputs = None
        for _ in range(warmup):
            outputs = call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = call()
        assert outputs is not None
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        hidden_first, pre_ff_first = (tensor.clone() for tensor in outputs)
        graph.replay()
        torch.cuda.synchronize()
        hidden_second, pre_ff_second = (tensor.clone() for tensor in outputs)
        samples = _measure_graph(
            graph,
            iterations=iterations,
            repeats=repeats,
        )
        hidden_delta = hidden_first.float() - reference_hidden.float()
        pre_ff_delta = pre_ff_first.float() - reference_pre_ff.float()
        return {
            "case": name,
            "median_us": float(statistics.median(samples)),
            "samples_us": samples,
            "hidden_exact": bool(torch.equal(hidden_first, reference_hidden)),
            "pre_ff_exact": bool(torch.equal(pre_ff_first, reference_pre_ff)),
            "hidden_max_abs_error": float(hidden_delta.abs().max().item()),
            "pre_ff_max_abs_error": float(pre_ff_delta.abs().max().item()),
            "repeat_hidden_exact": bool(torch.equal(hidden_first, hidden_second)),
            "repeat_pre_ff_exact": bool(torch.equal(pre_ff_first, pre_ff_second)),
            "error": None,
        }
    except Exception as exc:
        return {
            "case": name,
            "median_us": None,
            "samples_us": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _valid(row: dict[str, Any]) -> bool:
    return bool(
        row.get("error") is None
        and row.get("hidden_exact") is True
        and row.get("pre_ff_exact") is True
        and row.get("repeat_hidden_exact") is True
        and row.get("repeat_pre_ff_exact") is True
    )


@torch.inference_mode()
def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    from megagemm.kernels.rmsnorm_triton import (
        rmsnorm_triton,
        rmsnorm_triton_attn_residual_dense,
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260829)
    options = {"device": "cuda", "dtype": torch.bfloat16}
    attn = torch.empty((ROWS, HIDDEN), **options).normal_(
        mean=0.0,
        std=0.2,
        generator=generator,
    )
    residual = torch.empty_like(attn).normal_(
        mean=0.0,
        std=0.3,
        generator=generator,
    )
    post_weight = torch.empty((HIDDEN,), **options).normal_(
        mean=1.0,
        std=0.05,
        generator=generator,
    )
    pre_ff_weight = torch.empty((HIDDEN,), **options).normal_(
        mean=1.0,
        std=0.05,
        generator=generator,
    )

    post_reference = rmsnorm_triton(
        attn,
        post_weight,
        args.norm_eps,
        False,
    )
    reference_hidden = (residual + post_reference).to(torch.bfloat16)
    reference_pre_ff = rmsnorm_triton(
        reference_hidden,
        pre_ff_weight,
        args.norm_eps,
        False,
    )
    torch.cuda.synchronize()

    baseline_post = torch.empty_like(attn)
    baseline_hidden = torch.empty_like(attn)
    baseline_pre_ff = torch.empty_like(attn)

    def baseline_call() -> tuple[torch.Tensor, torch.Tensor]:
        rmsnorm_triton(
            attn,
            post_weight,
            args.norm_eps,
            False,
            out=baseline_post,
        )
        torch.add(residual, baseline_post, out=baseline_hidden)
        rmsnorm_triton(
            baseline_hidden,
            pre_ff_weight,
            args.norm_eps,
            False,
            out=baseline_pre_ff,
        )
        return baseline_hidden, baseline_pre_ff

    candidate_hidden = torch.empty_like(attn)
    candidate_pre_ff = torch.empty_like(attn)

    def candidate_call() -> tuple[torch.Tensor, torch.Tensor]:
        return rmsnorm_triton_attn_residual_dense(
            attn,
            residual,
            post_weight,
            pre_ff_weight,
            args.norm_eps,
            out_hidden=candidate_hidden,
            pre_ff_out=candidate_pre_ff,
        )

    baseline_first = _capture_case(
        "three_launch_baseline_first",
        baseline_call,
        reference_hidden=reference_hidden,
        reference_pre_ff=reference_pre_ff,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    candidate = _capture_case(
        "single_launch_dense_bridge",
        candidate_call,
        reference_hidden=reference_hidden,
        reference_pre_ff=reference_pre_ff,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    baseline_recheck = _capture_case(
        "three_launch_baseline_recheck",
        baseline_call,
        reference_hidden=reference_hidden,
        reference_pre_ff=reference_pre_ff,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    valid = all(_valid(row) for row in (baseline_first, candidate, baseline_recheck))
    if valid:
        baseline_values = (
            float(baseline_first["median_us"]),
            float(baseline_recheck["median_us"]),
        )
        baseline_us = min(baseline_values)
        stability = max(baseline_values) / baseline_us
        candidate_us = float(candidate["median_us"])
        valid = stability <= MAX_STABILITY_RATIO
        speedup = baseline_us / candidate_us if valid else None
    else:
        baseline_us = None
        stability = None
        candidate_us = None
        speedup = None
    apply_change = bool(speedup is not None and speedup >= args.minimum_speedup)
    return {
        "decision": "TEST_FULL_MODEL" if apply_change else "KEEP_THREE_LAUNCH_BASELINE",
        "apply_change": apply_change,
        "valid": valid,
        "shape": {
            "rows": ROWS,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "dtype": "bf16",
        },
        "baseline_us": baseline_us,
        "candidate_us": candidate_us,
        "baseline_stability_ratio": stability,
        "speedup": speedup,
        "estimated_savings_ms_per_step_35_layers": (
            (baseline_us - candidate_us) * LAYERS / 1000.0
            if baseline_us is not None and candidate_us is not None
            else None
        ),
        "minimum_speedup": args.minimum_speedup,
        "cases": [baseline_first, candidate, baseline_recheck],
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm-eps", type=float, default=1.0e-6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--minimum-speedup", type=float, default=MIN_SPEEDUP)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_e2b_dense_attn_mlp_bridge_l4.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    gpu = torch.cuda.get_device_name(0)
    if "l4" not in gpu.lower():
        raise SystemExit(f"NVIDIA L4 required, found {gpu}")
    if args.iterations <= 0 or args.repeats < 3:
        raise SystemExit("iterations must be positive and repeats must be >= 3")
    print("Gemma4 E2B/L4 dense attention -> MLP bridge gate")
    print(f"  gpu: {gpu}")
    print("  shape: B8 H1536 BF16, 35 dense layers")
    print("  baseline: post-attn RMSNorm + residual add + pre-FFN RMSNorm")
    print("  candidate: one Triton launch with exact staged rounding")
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

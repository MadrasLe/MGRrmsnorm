#!/usr/bin/env python3
"""Gate Gemma4's shared MLP at one real B8xC2048 prefill chunk."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

# This process-local setting makes the existing DeepFusion implementation a
# real Triton control instead of its default PyTorch fallback.
os.environ.setdefault("MEGAGEMM_DEEPFUSION_PREFILL_FORCE_TRITON", "1")

from megagemm.kernels.deepfusion_mlp import deepfusion_swiglu_down
from megagemm.kernels.swiglu import gated_activation_forward


TensorFn = Callable[[], torch.Tensor]


def _measure_us(
    fn: TensorFn,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
    return {
        "median_us": statistics.median(samples),
        "samples_us": samples,
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float().reshape(1, -1),
            right.float().reshape(1, -1),
        ).item()
    )


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--near-tie-ratio", type=float, default=1.005)
    parser.add_argument("--maximum-baseline-spread", type=float, default=1.03)
    parser.add_argument("--maximum-candidate-spread", type=float, default=1.04)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_long_shared_mlp_prefill_a100.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 1 or args.iterations < 1 or args.repeats < 3:
        raise SystemExit("warmup/iterations must be positive and repeats >= 3")

    gpu = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if "a100" not in gpu.lower().replace("-", " ").split():
        raise SystemExit(f"This shape gate requires an A100, found: {gpu}")
    if vram_gb < 70.0:
        raise SystemExit(f"This gate requires an A100 80GB, found {vram_gb:.2f}GB")

    torch.manual_seed(20260811)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    rows = 16_384
    hidden_dim = 2_816
    intermediate_dim = 2_112
    layers = 30
    chunks = 2
    layer_invocations = layers * chunks
    target_gap_ms = 433.605810

    print("Gemma4 long shared-MLP prefill gate")
    print(f"  gpu: {gpu}")
    print(f"  capability: {capability}")
    print(f"  vram_gb: {vram_gb:.2f}")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  package_install: disabled")
    print(
        "  shape: "
        f"rows={rows} H={hidden_dim} shared_I={intermediate_dim} dtype=bf16"
    )
    print(f"  B16 estimate: {chunks} chunks x {layers} layers")

    def random_tensor(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=dtype).normal_(0.0, 0.02)

    hidden = random_tensor((rows, hidden_dim))
    gate_up_weight = random_tensor((2 * intermediate_dim, hidden_dim))
    down_weight = random_tensor((hidden_dim, intermediate_dim))
    gate_up_weight_t = gate_up_weight.transpose(0, 1)
    down_weight_t = down_weight.transpose(0, 1)

    def make_current() -> tuple[TensorFn, dict[str, Any]]:
        gate_up_out = torch.empty(
            (rows, 2 * intermediate_dim), device=device, dtype=dtype
        )
        down_out = torch.empty((rows, hidden_dim), device=device, dtype=dtype)

        def invoke() -> torch.Tensor:
            torch.mm(hidden, gate_up_weight_t, out=gate_up_out)
            gate = gate_up_out[:, :intermediate_dim]
            up = gate_up_out[:, intermediate_dim:]
            activated = F.gelu(gate, approximate="tanh")
            activated.mul_(up)
            torch.mm(activated, down_weight_t, out=down_out)
            return down_out

        return invoke, {"path": "cublas_gelu_mul_cublas"}

    def make_fused_activation(
        block_size: int,
    ) -> tuple[TensorFn, dict[str, Any]]:
        gate_up_out = torch.empty(
            (rows, 2 * intermediate_dim), device=device, dtype=dtype
        )
        activated_out = torch.empty(
            (rows, intermediate_dim), device=device, dtype=dtype
        )
        down_out = torch.empty((rows, hidden_dim), device=device, dtype=dtype)

        def invoke() -> torch.Tensor:
            torch.mm(hidden, gate_up_weight_t, out=gate_up_out)
            gated_activation_forward(
                gate_up_out,
                intermediate_dim,
                activation="gelu_tanh",
                out=activated_out,
                block_size=block_size,
            )
            torch.mm(activated_out, down_weight_t, out=down_out)
            return down_out

        return invoke, {
            "path": "cublas_fused_gelu_mul_cublas",
            "activation_block": int(block_size),
        }

    def make_deepfusion() -> tuple[TensorFn, dict[str, Any]]:
        gate_up_out = torch.empty(
            (rows, 2 * intermediate_dim), device=device, dtype=dtype
        )
        down_out = torch.empty((rows, hidden_dim), device=device, dtype=dtype)

        def invoke() -> torch.Tensor:
            torch.mm(hidden, gate_up_weight_t, out=gate_up_out)
            return deepfusion_swiglu_down(
                gate_up_out,
                down_weight,
                out=down_out,
                mode="prefill",
                activation="gelu_tanh",
            )

        return invoke, {"path": "cublas_deepfusion_gelu_down"}

    reference_fn, _ = make_current()
    reference = reference_fn().detach().clone()
    reference_repeat = reference_fn().detach().clone()
    torch.cuda.synchronize()
    if not torch.equal(reference, reference_repeat):
        raise SystemExit("Production baseline is not repeat-exact")
    del reference_fn, reference_repeat

    measured: list[dict[str, Any]] = []

    def profile_case(
        name: str,
        factory: Callable[[], tuple[TensorFn, dict[str, Any]]],
    ) -> None:
        invoke: TensorFn | None = None
        try:
            invoke, config = factory()
            first = invoke().detach().clone()
            second = invoke().detach().clone()
            torch.cuda.synchronize()
            delta = (first.float() - reference.float()).abs()
            repeat_delta = (first.float() - second.float()).abs()
            finite = bool(torch.isfinite(first).all().item())
            cosine = _cosine(first, reference)
            max_abs_error = float(delta.max().item())
            mean_abs_error = float(delta.mean().item())
            repeat_exact = bool(torch.equal(first, second))
            repeat_max_abs_error = float(repeat_delta.max().item())
            correct = bool(
                finite
                and repeat_exact
                and cosine >= 0.9999
                and max_abs_error <= 0.125
            )
            timing = _measure_us(
                invoke,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            row = {
                "case": name,
                "config": config,
                "error": None,
                "correct": correct,
                "finite": finite,
                "cosine": cosine,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "repeat_exact": repeat_exact,
                "repeat_max_abs_error": repeat_max_abs_error,
                **timing,
            }
            row["sample_spread_ratio"] = (
                max(timing["samples_us"]) / min(timing["samples_us"])
            )
            del first, second, delta, repeat_delta
        except Exception as exc:
            row = {
                "case": name,
                "error": f"{type(exc).__name__}: {exc}",
                "correct": False,
                "median_us": None,
                "samples_us": [],
                "sample_spread_ratio": None,
            }
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        measured.append(row)
        print(json.dumps(row, sort_keys=True))
        del invoke
        gc.collect()

    profile_case("current", make_current)
    for block_size in (128, 256, 512, 1024):
        profile_case(
            f"fused_activation_b{block_size}",
            lambda block_size=block_size: make_fused_activation(block_size),
        )
    profile_case("deepfusion_gelu_down", make_deepfusion)
    profile_case("current_recheck", make_current)

    by_name = {row["case"]: row for row in measured}
    cold_baseline = by_name["current"]
    stable_baseline = by_name["current_recheck"]
    baseline_us = (
        float(stable_baseline["median_us"])
        if stable_baseline.get("correct")
        and stable_baseline.get("median_us") is not None
        else None
    )
    cold_baseline_us = (
        float(cold_baseline["median_us"])
        if cold_baseline.get("correct") and cold_baseline.get("median_us") is not None
        else None
    )
    cold_baseline_ratio = (
        max(cold_baseline_us, baseline_us) / min(cold_baseline_us, baseline_us)
        if cold_baseline_us is not None
        and baseline_us is not None
        and min(cold_baseline_us, baseline_us) > 0.0
        else None
    )
    baseline_sample_spread_ratio = stable_baseline.get("sample_spread_ratio")

    candidates = [
        row
        for row in measured
        if row["case"] not in ("current", "current_recheck")
        and row.get("correct")
        and row.get("median_us") is not None
    ]
    raw_measured_winner = (
        min(candidates, key=lambda row: float(row["median_us"]))
        if candidates
        else None
    )
    near_tied_candidates = (
        [
            row
            for row in candidates
            if float(row["median_us"])
            <= float(raw_measured_winner["median_us"])
            * float(args.near_tie_ratio)
        ]
        if raw_measured_winner is not None
        else []
    )
    measured_winner = (
        min(
            near_tied_candidates,
            key=lambda row: (
                float(row["sample_spread_ratio"]),
                float(row["median_us"]),
            ),
        )
        if near_tied_candidates
        else None
    )
    candidate_sample_spread_ratio = (
        float(measured_winner["sample_spread_ratio"])
        if measured_winner is not None
        and measured_winner.get("sample_spread_ratio") is not None
        else None
    )
    stable = bool(
        baseline_sample_spread_ratio is not None
        and float(baseline_sample_spread_ratio)
        <= float(args.maximum_baseline_spread)
        and candidate_sample_spread_ratio is not None
        and candidate_sample_spread_ratio <= float(args.maximum_candidate_spread)
    )
    speedup = (
        baseline_us / float(measured_winner["median_us"])
        if baseline_us is not None and measured_winner is not None
        else None
    )
    apply_change = bool(
        stable
        and speedup is not None
        and speedup >= float(args.minimum_speedup)
    )
    selected_us = (
        float(measured_winner["median_us"])
        if apply_change and measured_winner is not None
        else baseline_us
    )
    estimated_baseline_ms = (
        baseline_us * layer_invocations / 1000.0
        if baseline_us is not None
        else None
    )
    estimated_selected_ms = (
        selected_us * layer_invocations / 1000.0
        if selected_us is not None
        else None
    )
    estimated_savings_ms = (
        estimated_baseline_ms - estimated_selected_ms
        if estimated_baseline_ms is not None
        and estimated_selected_ms is not None
        else None
    )
    winner_name = (
        str(measured_winner["case"])
        if apply_change and measured_winner is not None
        else "current"
    )
    summary = {
        "decision": "APPLY" if apply_change else "KEEP_CURRENT",
        "apply_change": apply_change,
        "winner": winner_name,
        "measured_winner": (
            measured_winner["case"] if measured_winner is not None else None
        ),
        "raw_measured_winner": (
            raw_measured_winner["case"]
            if raw_measured_winner is not None
            else None
        ),
        "gpu": gpu,
        "capability": list(capability),
        "shape": {
            "rows": rows,
            "hidden_dim": hidden_dim,
            "shared_intermediate_dim": intermediate_dim,
            "dtype": "bf16",
        },
        "layers": layers,
        "chunks": chunks,
        "layer_invocations": layer_invocations,
        "baseline_us": baseline_us,
        "selected_us": selected_us,
        "cold_baseline_us": cold_baseline_us,
        "cold_baseline_ratio": cold_baseline_ratio,
        "baseline_sample_spread_ratio": baseline_sample_spread_ratio,
        "candidate_sample_spread_ratio": candidate_sample_spread_ratio,
        "stable": stable,
        "maximum_baseline_spread": float(args.maximum_baseline_spread),
        "maximum_candidate_spread": float(args.maximum_candidate_spread),
        "minimum_speedup": float(args.minimum_speedup),
        "near_tie_ratio": float(args.near_tie_ratio),
        "speedup": speedup,
        "estimated_baseline_ms_b16_prefill": estimated_baseline_ms,
        "estimated_selected_ms_b16_prefill": estimated_selected_ms,
        "estimated_savings_ms_b16_prefill": estimated_savings_ms,
        "target_prefill_gap_ms": target_gap_ms,
        "estimated_gap_coverage": (
            estimated_savings_ms / target_gap_ms
            if estimated_savings_ms is not None
            else None
        ),
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "cases": measured,
    }
    print("DECISION " + json.dumps(summary, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

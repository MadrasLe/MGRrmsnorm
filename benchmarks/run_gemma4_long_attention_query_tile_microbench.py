#!/usr/bin/env python3
"""Tune query-tile granularity for both promoted Gemma4 long attentions."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from megagemm.kernels import paged_attention


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


def _required_output(output: Optional[torch.Tensor], name: str) -> torch.Tensor:
    if output is None:
        raise RuntimeError(f"{name} returned None for the required A100 shape")
    return output


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--near-tie-ratio", type=float, default=1.005)
    parser.add_argument("--maximum-baseline-spread", type=float, default=1.03)
    parser.add_argument("--maximum-candidate-spread", type=float, default=1.04)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_long_attention_query_tile_a100.json",
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
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 8
    seq_len = 2_048
    q_heads = 16
    b16_chunks = 2
    target_gap_ms = 271.552678

    print("Gemma4 long-attention query-tile gate")
    print(f"  gpu: {gpu}")
    print(f"  capability: {capability}")
    print(f"  vram_gb: {vram_gb:.2f}")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  package_install: disabled")
    print(f"  shared shape: batch={batch_size} context={seq_len} q_heads={q_heads}")
    print("  objective: reuse each K/V tile across more query positions")

    def run_attention_gate(
        *,
        attention_type: str,
        kv_heads: int,
        head_dim: int,
        layer_invocations: int,
        current_config: dict[str, int],
        candidate_configs: tuple[dict[str, int | str], ...],
    ) -> dict[str, Any]:
        q = torch.randn(
            batch_size,
            q_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        k = torch.randn(
            batch_size,
            kv_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        v = torch.randn_like(k)

        def make_fn(config: dict[str, int | str]) -> TensorFn:
            block_m = int(config["block_m"])
            num_warps = int(config["num_warps"])
            if attention_type == "sliding":
                def invoke() -> torch.Tensor:
                    return _required_output(
                        paged_attention.gemma4_long_sliding_prefill_attention(
                            q,
                            k,
                            v,
                            sliding_window=1_024,
                            scale=1.0,
                            block_m=block_m,
                            num_warps=num_warps,
                            num_stages=2,
                            force=True,
                        ),
                        "long sliding attention",
                    )
            else:
                def invoke() -> torch.Tensor:
                    return _required_output(
                        paged_attention.gemma4_long_full_prefill_attention(
                            q,
                            k,
                            v,
                            scale=1.0,
                            block_m=block_m,
                            block_n=32,
                            num_warps=num_warps,
                            num_stages=2,
                            force=True,
                        ),
                        "long full attention",
                    )
            return invoke

        current_fn = make_fn(current_config)
        reference = current_fn().detach().clone()
        reference_repeat = current_fn().detach().clone()
        torch.cuda.synchronize()
        if not torch.equal(reference, reference_repeat):
            raise SystemExit(f"{attention_type} production baseline is not repeat-exact")
        del reference_repeat

        measured: list[dict[str, Any]] = []

        def profile_case(name: str, config: dict[str, int | str]) -> None:
            invoke: TensorFn | None = None
            try:
                invoke = make_fn(config)
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
                    "attention_type": attention_type,
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
                    "attention_type": attention_type,
                    "case": name,
                    "config": config,
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

        profile_case("current", current_config)
        for config in candidate_configs:
            profile_case(str(config["case"]), config)
        profile_case("current_recheck", current_config)

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
            if cold_baseline.get("correct")
            and cold_baseline.get("median_us") is not None
            else None
        )
        baseline_spread = stable_baseline.get("sample_spread_ratio")
        candidates = [
            row
            for row in measured
            if row["case"] not in ("current", "current_recheck")
            and row.get("correct")
            and row.get("median_us") is not None
        ]
        raw_winner = (
            min(candidates, key=lambda row: float(row["median_us"]))
            if candidates
            else None
        )
        near_tied = (
            [
                row
                for row in candidates
                if float(row["median_us"])
                <= float(raw_winner["median_us"]) * float(args.near_tie_ratio)
            ]
            if raw_winner is not None
            else []
        )
        measured_winner = (
            min(
                near_tied,
                key=lambda row: (
                    float(row["sample_spread_ratio"]),
                    float(row["median_us"]),
                ),
            )
            if near_tied
            else None
        )
        candidate_spread = (
            float(measured_winner["sample_spread_ratio"])
            if measured_winner is not None
            and measured_winner.get("sample_spread_ratio") is not None
            else None
        )
        stable = bool(
            baseline_spread is not None
            and float(baseline_spread) <= float(args.maximum_baseline_spread)
            and candidate_spread is not None
            and candidate_spread <= float(args.maximum_candidate_spread)
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
        baseline_ms = (
            baseline_us * layer_invocations / 1000.0
            if baseline_us is not None
            else None
        )
        selected_ms = (
            selected_us * layer_invocations / 1000.0
            if selected_us is not None
            else None
        )
        summary = {
            "attention_type": attention_type,
            "decision": "APPLY" if apply_change else "KEEP_CURRENT",
            "apply_change": apply_change,
            "winner": (
                measured_winner["case"]
                if apply_change and measured_winner is not None
                else "current"
            ),
            "measured_winner": (
                measured_winner["case"] if measured_winner is not None else None
            ),
            "raw_measured_winner": (
                raw_winner["case"] if raw_winner is not None else None
            ),
            "baseline_us": baseline_us,
            "selected_us": selected_us,
            "cold_baseline_us": cold_baseline_us,
            "baseline_sample_spread_ratio": baseline_spread,
            "candidate_sample_spread_ratio": candidate_spread,
            "stable": stable,
            "speedup": speedup,
            "layer_invocations": layer_invocations,
            "estimated_baseline_ms_b16_prefill": baseline_ms,
            "estimated_selected_ms_b16_prefill": selected_ms,
            "estimated_savings_ms_b16_prefill": (
                baseline_ms - selected_ms
                if baseline_ms is not None and selected_ms is not None
                else None
            ),
            "cases": measured,
        }
        print("ATTENTION_DECISION " + json.dumps(summary, sort_keys=True))
        del q, k, v, reference, current_fn
        gc.collect()
        torch.cuda.empty_cache()
        return summary

    sliding = run_attention_gate(
        attention_type="sliding",
        kv_heads=8,
        head_dim=256,
        layer_invocations=25 * b16_chunks,
        current_config={"case": "current", "block_m": 64, "num_warps": 8},
        candidate_configs=(
            {"case": "sliding_bm64_w4", "block_m": 64, "num_warps": 4},
            {"case": "sliding_bm128_w8", "block_m": 128, "num_warps": 8},
            {"case": "sliding_bm128_w4", "block_m": 128, "num_warps": 4},
        ),
    )
    full = run_attention_gate(
        attention_type="full",
        kv_heads=2,
        head_dim=512,
        layer_invocations=5 * b16_chunks,
        current_config={"case": "current", "block_m": 32, "num_warps": 4},
        candidate_configs=(
            {"case": "full_bm64_w4", "block_m": 64, "num_warps": 4},
            {"case": "full_bm64_w8", "block_m": 64, "num_warps": 8},
        ),
    )

    baseline_parts = (
        sliding.get("estimated_baseline_ms_b16_prefill"),
        full.get("estimated_baseline_ms_b16_prefill"),
    )
    selected_parts = (
        sliding.get("estimated_selected_ms_b16_prefill"),
        full.get("estimated_selected_ms_b16_prefill"),
    )
    baseline_ms = (
        sum(float(value) for value in baseline_parts)
        if all(value is not None for value in baseline_parts)
        else None
    )
    selected_ms = (
        sum(float(value) for value in selected_parts)
        if all(value is not None for value in selected_parts)
        else None
    )
    savings_ms = (
        baseline_ms - selected_ms
        if baseline_ms is not None and selected_ms is not None
        else None
    )
    summary = {
        "decision": (
            "APPLY_SELECTED" if sliding["apply_change"] or full["apply_change"]
            else "KEEP_CURRENT"
        ),
        "apply_change": bool(sliding["apply_change"] or full["apply_change"]),
        "gpu": gpu,
        "capability": list(capability),
        "shape": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "q_heads": q_heads,
            "dtype": "bf16",
        },
        "b16_chunks": b16_chunks,
        "minimum_speedup": float(args.minimum_speedup),
        "near_tie_ratio": float(args.near_tie_ratio),
        "maximum_baseline_spread": float(args.maximum_baseline_spread),
        "maximum_candidate_spread": float(args.maximum_candidate_spread),
        "estimated_baseline_ms_b16_prefill": baseline_ms,
        "estimated_selected_ms_b16_prefill": selected_ms,
        "estimated_savings_ms_b16_prefill": savings_ms,
        "target_prefill_gap_ms": target_gap_ms,
        "estimated_gap_coverage": (
            savings_ms / target_gap_ms if savings_ms is not None else None
        ),
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "sliding": sliding,
        "full": full,
    }
    print("DECISION " + json.dumps(summary, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

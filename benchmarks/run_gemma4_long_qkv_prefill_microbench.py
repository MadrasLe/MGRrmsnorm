#!/usr/bin/env python3
"""Gate long-prefill QKV packing for the Gemma4 A4B A100 topology."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import torch


TensorTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
CaseFn = Callable[[], None]
OutputFn = Callable[[], TensorTuple]


def _measure_us(
    fn: CaseFn,
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
    return {"median_us": statistics.median(samples), "samples_us": samples}


def _tensor_tuple_error(reference: TensorTuple, candidate: TensorTuple) -> dict[str, Any]:
    dot = 0.0
    reference_norm = 0.0
    candidate_norm = 0.0
    max_abs_error = 0.0
    abs_error_sum = 0.0
    elements = 0
    finite = True
    for ref, cand in zip(reference, candidate):
        ref32 = ref.float()
        cand32 = cand.float()
        delta = (cand32 - ref32).abs()
        dot += float((ref32 * cand32).sum().item())
        reference_norm += float((ref32 * ref32).sum().item())
        candidate_norm += float((cand32 * cand32).sum().item())
        max_abs_error = max(max_abs_error, float(delta.max().item()))
        abs_error_sum += float(delta.sum().item())
        elements += int(delta.numel())
        finite = finite and bool(torch.isfinite(cand).all().item())
        del ref32, cand32, delta
    denominator = math.sqrt(reference_norm * candidate_norm)
    return {
        "finite": finite,
        "cosine": dot / denominator if denominator > 0.0 else 1.0,
        "max_abs_error": max_abs_error,
        "mean_abs_error": abs_error_sum / max(1, elements),
    }


def _repeat_error(first: TensorTuple, second: TensorTuple) -> tuple[bool, float]:
    exact = True
    max_abs_error = 0.0
    for left, right in zip(first, second):
        exact = exact and bool(torch.equal(left, right))
        max_abs_error = max(
            max_abs_error,
            float((left.float() - right.float()).abs().max().item()),
        )
    return exact, max_abs_error


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--near-tie-ratio", type=float, default=1.005)
    parser.add_argument("--maximum-baseline-spread", type=float, default=1.03)
    parser.add_argument("--maximum-candidate-spread", type=float, default=1.04)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_long_qkv_prefill_a100.json",
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
    hidden = 2_816
    target_gap_ms = 271.552678

    print("Gemma4 long QKV-prefill packing gate")
    print(f"  gpu: {gpu}")
    print(f"  capability: {capability}")
    print(f"  vram_gb: {vram_gb:.2f}")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  package_install: disabled")
    print(f"  shared shape: rows={rows} hidden={hidden} dtype=bf16")
    print("  candidates: three GEMMs vs Q+(KV) vs one QKV GEMM")

    def run_shape(
        *,
        attention_type: str,
        q_size: int,
        k_size: int,
        v_size: int,
        layer_invocations: int,
    ) -> dict[str, Any]:
        x = torch.empty((rows, hidden), device=device, dtype=dtype).normal_(0.0, 0.02)
        q_weight = torch.empty((q_size, hidden), device=device, dtype=dtype).normal_(0.0, 0.02)
        k_weight = torch.empty((k_size, hidden), device=device, dtype=dtype).normal_(0.0, 0.02)
        v_weight = torch.empty((v_size, hidden), device=device, dtype=dtype).normal_(0.0, 0.02)
        kv_weight = torch.cat((k_weight, v_weight), dim=0).contiguous()
        qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0).contiguous()

        q_out = torch.empty((rows, q_size), device=device, dtype=dtype)
        k_out = torch.empty((rows, k_size), device=device, dtype=dtype)
        v_out = torch.empty((rows, v_size), device=device, dtype=dtype)
        q_pair_out = torch.empty_like(q_out)
        kv_out = torch.empty((rows, k_size + v_size), device=device, dtype=dtype)
        qkv_out = torch.empty(
            (rows, q_size + k_size + v_size), device=device, dtype=dtype
        )

        def separate() -> None:
            torch.mm(x, q_weight.t(), out=q_out)
            torch.mm(x, k_weight.t(), out=k_out)
            torch.mm(x, v_weight.t(), out=v_out)

        def separate_outputs() -> TensorTuple:
            return q_out, k_out, v_out

        def q_plus_kv() -> None:
            torch.mm(x, q_weight.t(), out=q_pair_out)
            torch.mm(x, kv_weight.t(), out=kv_out)

        def q_plus_kv_outputs() -> TensorTuple:
            return (
                q_pair_out,
                kv_out[:, :k_size],
                kv_out[:, k_size:],
            )

        def fused_qkv() -> None:
            torch.mm(x, qkv_weight.t(), out=qkv_out)

        def fused_qkv_outputs() -> TensorTuple:
            return (
                qkv_out[:, :q_size],
                qkv_out[:, q_size : q_size + k_size],
                qkv_out[:, q_size + k_size :],
            )

        separate()
        torch.cuda.synchronize()
        reference = tuple(output.detach().clone() for output in separate_outputs())
        separate()
        torch.cuda.synchronize()
        baseline_exact, baseline_repeat_error = _repeat_error(
            reference,
            separate_outputs(),
        )
        if not baseline_exact:
            raise SystemExit(
                f"{attention_type} separate-QKV baseline is not repeat-exact: "
                f"{baseline_repeat_error}"
            )

        measured: list[dict[str, Any]] = []

        def profile_case(
            name: str,
            fn: CaseFn,
            outputs: OutputFn,
            kernels: int,
        ) -> None:
            try:
                fn()
                torch.cuda.synchronize()
                first = tuple(output.detach().clone() for output in outputs())
                fn()
                torch.cuda.synchronize()
                repeat_exact, repeat_max_abs_error = _repeat_error(first, outputs())
                error = _tensor_tuple_error(reference, outputs())
                correct = bool(
                    error["finite"]
                    and repeat_exact
                    and error["cosine"] >= 0.9999
                    and error["max_abs_error"] <= 0.03125
                )
                timing = _measure_us(
                    fn,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
                row = {
                    "attention_type": attention_type,
                    "case": name,
                    "kernels": kernels,
                    "error": None,
                    "correct": correct,
                    **error,
                    "repeat_exact": repeat_exact,
                    "repeat_max_abs_error": repeat_max_abs_error,
                    **timing,
                }
                row["sample_spread_ratio"] = (
                    max(timing["samples_us"]) / min(timing["samples_us"])
                )
                del first
            except Exception as exc:
                row = {
                    "attention_type": attention_type,
                    "case": name,
                    "kernels": kernels,
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
            gc.collect()

        profile_case("current_three_gemms", separate, separate_outputs, 3)
        profile_case("q_plus_fused_kv", q_plus_kv, q_plus_kv_outputs, 2)
        profile_case("fused_qkv", fused_qkv, fused_qkv_outputs, 1)
        profile_case("current_recheck", separate, separate_outputs, 3)

        by_name = {row["case"]: row for row in measured}
        cold_baseline = by_name["current_three_gemms"]
        baseline = by_name["current_recheck"]
        baseline_us = (
            float(baseline["median_us"])
            if baseline.get("correct") and baseline.get("median_us") is not None
            else None
        )
        candidates = [
            row
            for row in measured
            if row["case"] not in ("current_three_gemms", "current_recheck")
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
                    int(row["kernels"]),
                    float(row["sample_spread_ratio"]),
                    float(row["median_us"]),
                ),
            )
            if near_tied
            else None
        )
        baseline_spread = baseline.get("sample_spread_ratio")
        candidate_spread = (
            measured_winner.get("sample_spread_ratio")
            if measured_winner is not None
            else None
        )
        stable = bool(
            baseline_spread is not None
            and float(baseline_spread) <= float(args.maximum_baseline_spread)
            and candidate_spread is not None
            and float(candidate_spread) <= float(args.maximum_candidate_spread)
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
                else "current_three_gemms"
            ),
            "measured_winner": (
                measured_winner["case"] if measured_winner is not None else None
            ),
            "raw_measured_winner": (
                raw_winner["case"] if raw_winner is not None else None
            ),
            "baseline_us": baseline_us,
            "cold_baseline_us": (
                cold_baseline.get("median_us")
                if cold_baseline.get("correct")
                else None
            ),
            "selected_us": selected_us,
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
        print("QKV_DECISION " + json.dumps(summary, sort_keys=True))

        del (
            x,
            q_weight,
            k_weight,
            v_weight,
            kv_weight,
            qkv_weight,
            q_out,
            k_out,
            v_out,
            q_pair_out,
            kv_out,
            qkv_out,
            reference,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return summary

    sliding = run_shape(
        attention_type="sliding",
        q_size=4_096,
        k_size=2_048,
        v_size=2_048,
        layer_invocations=25 * 2,
    )
    full = run_shape(
        attention_type="full",
        q_size=8_192,
        k_size=1_024,
        v_size=1_024,
        layer_invocations=5 * 2,
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
        "shape": {"rows": rows, "hidden": hidden, "dtype": "bf16"},
        "minimum_speedup": float(args.minimum_speedup),
        "near_tie_ratio": float(args.near_tie_ratio),
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

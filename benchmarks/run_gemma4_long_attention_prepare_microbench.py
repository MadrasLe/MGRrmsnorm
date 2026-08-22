#!/usr/bin/env python3
"""Gate fused Q/K/V normalization, RoPE, and layouts for Gemma4 long prefill."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megagemm.kernels.gemma4_attention_prepare import (  # noqa: E402
    gemma4_prefill_attention_prepare,
)
from megagemm.kernels.rope import apply_rotary_emb  # noqa: E402
from megagemm.kernels.rmsnorm_triton import (  # noqa: E402
    rmsnorm_triton,
    rmsnorm_triton_no_weight,
)


TensorTuple = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
CaseFn = Callable[[], TensorTuple]


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


def _tuple_error(reference: TensorTuple, candidate: TensorTuple) -> dict[str, Any]:
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
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--precondition-pairs", type=int, default=5)
    parser.add_argument(
        "--only",
        choices=("all", "sliding", "full"),
        default="all",
    )
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--maximum-baseline-spread", type=float, default=1.03)
    parser.add_argument("--maximum-candidate-spread", type=float, default=1.04)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_long_attention_prepare_a100.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if (
        args.warmup < 1
        or args.iterations < 1
        or args.repeats < 3
        or args.precondition_pairs < 1
    ):
        raise SystemExit(
            "warmup/iterations/precondition-pairs must be positive and repeats >= 3"
        )

    gpu = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if "a100" not in gpu.lower().replace("-", " ").split():
        raise SystemExit(f"This shape gate requires an A100, found: {gpu}")
    if vram_gb < 70.0:
        raise SystemExit(f"This gate requires an A100 80GB, found {vram_gb:.2f}GB")

    torch.manual_seed(20260812)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 8
    seq_len = 2_048
    q_heads = 16
    b16_chunks = 2
    target_gap_ms = 266.321581

    print("Gemma4 long attention-prepare gate")
    print(f"  gpu: {gpu}")
    print(f"  capability: {capability}")
    print(f"  vram_gb: {vram_gb:.2f}")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  package_install: disabled")
    print(f"  shared shape: batch={batch_size} context={seq_len} q_heads={q_heads}")
    print("  candidate: one fused kernel for RMSNorm + RoPE + Q/K/V layouts")
    print(f"  measured attention types: {args.only}")
    print(f"  precondition pairs before timing: {args.precondition_pairs}")

    def run_shape(
        *,
        attention_type: str,
        kv_heads: int,
        head_dim: int,
        k_eq_v: bool,
        layer_invocations: int,
    ) -> dict[str, Any]:
        eps = 1e-6
        q_raw = torch.empty(
            (batch_size, seq_len, q_heads * head_dim),
            device=device,
            dtype=dtype,
        ).normal_(0.0, 0.02)
        k_raw = torch.empty(
            (batch_size, seq_len, kv_heads * head_dim),
            device=device,
            dtype=dtype,
        ).normal_(0.0, 0.02)
        v_raw = k_raw if k_eq_v else torch.empty_like(k_raw).normal_(0.0, 0.02)
        q_weight = torch.empty((head_dim,), device=device, dtype=dtype).normal_(
            1.0, 0.02
        )
        k_weight = torch.empty((head_dim,), device=device, dtype=dtype).normal_(
            1.0, 0.02
        )
        positions = torch.arange(seq_len, device=device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(batch_size, -1).contiguous()
        frequencies = torch.empty(
            (seq_len, head_dim // 2), device=device, dtype=torch.float32
        ).normal_(0.0, 0.01)
        cos = torch.cos(frequencies)
        sin = torch.sin(frequencies)

        q_fused = torch.empty(
            (batch_size, q_heads, seq_len, head_dim), device=device, dtype=dtype
        )
        k_fused = torch.empty(
            (batch_size, kv_heads, seq_len, head_dim), device=device, dtype=dtype
        )
        v_fused = torch.empty_like(k_fused)
        k_cache_fused = torch.empty(
            (batch_size, seq_len, kv_heads, head_dim), device=device, dtype=dtype
        )
        v_cache_fused = torch.empty_like(k_cache_fused)

        def current() -> TensorTuple:
            q = q_raw.view(
                batch_size, seq_len, q_heads, head_dim
            ).transpose(1, 2).contiguous()
            k = k_raw.view(
                batch_size, seq_len, kv_heads, head_dim
            ).transpose(1, 2).contiguous()
            v = v_raw.view(
                batch_size, seq_len, kv_heads, head_dim
            ).transpose(1, 2).contiguous()
            q = rmsnorm_triton(q, q_weight, eps, False)
            k = rmsnorm_triton(k, k_weight, eps, False)
            v = rmsnorm_triton_no_weight(v, eps)
            q, _ = apply_rotary_emb(
                q, q, cos, sin, position_ids=positions, half_rotate=True
            )
            k, _ = apply_rotary_emb(
                k, k, cos, sin, position_ids=positions, half_rotate=True
            )
            return (
                q,
                k,
                v,
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
            )

        def fused() -> TensorTuple:
            return gemma4_prefill_attention_prepare(
                q_raw,
                k_raw,
                v_raw,
                q_weight,
                k_weight,
                cos,
                sin,
                positions,
                num_q_heads=q_heads,
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                eps=eps,
                q_out=q_fused,
                k_out=k_fused,
                v_out=v_fused,
                k_cache=k_cache_fused,
                v_cache=v_cache_fused,
            )

        reference = tuple(output.detach().clone() for output in current())
        baseline_repeat = current()
        torch.cuda.synchronize()
        baseline_exact, baseline_repeat_error = _repeat_error(
            reference, baseline_repeat
        )
        if not baseline_exact:
            raise SystemExit(
                f"{attention_type} baseline is not repeat-exact: "
                f"{baseline_repeat_error}"
            )
        del baseline_repeat

        # Compile both implementations and settle allocator/clock state before
        # the first measured baseline. The v1 sliding result changed by 12.8%
        # only after the candidate had run, despite stable samples within each
        # pass; alternating warmup removes that ordering artifact.
        for _ in range(args.precondition_pairs):
            current()
            fused()
        torch.cuda.synchronize()

        measured: list[dict[str, Any]] = []

        def profile_case(case: str, fn: CaseFn) -> None:
            try:
                first = tuple(output.detach().clone() for output in fn())
                second = fn()
                torch.cuda.synchronize()
                repeat_exact, repeat_max_abs_error = _repeat_error(first, second)
                error = _tuple_error(reference, second)
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
                    "case": case,
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
                del first, second
            except Exception as exc:
                row = {
                    "attention_type": attention_type,
                    "case": case,
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

        profile_case("current_prepare", current)
        profile_case("fused_prepare", fused)
        profile_case("current_recheck", current)
        profile_case("fused_recheck", fused)

        by_name = {row["case"]: row for row in measured}
        cold_baseline = by_name["current_prepare"]
        baseline = by_name["current_recheck"]
        first_candidate = by_name["fused_prepare"]
        candidate = by_name["fused_recheck"]
        baseline_us = (
            float(baseline["median_us"])
            if baseline.get("correct") and baseline.get("median_us") is not None
            else None
        )
        candidate_us = (
            float(candidate["median_us"])
            if candidate.get("correct") and candidate.get("median_us") is not None
            else None
        )
        first_candidate_us = (
            float(first_candidate["median_us"])
            if first_candidate.get("correct")
            and first_candidate.get("median_us") is not None
            else None
        )
        cold_us = (
            float(cold_baseline["median_us"])
            if cold_baseline.get("correct")
            and cold_baseline.get("median_us") is not None
            else None
        )
        baseline_run_ratio = (
            max(cold_us, baseline_us) / min(cold_us, baseline_us)
            if cold_us is not None and baseline_us is not None
            else None
        )
        candidate_run_ratio = (
            max(first_candidate_us, candidate_us)
            / min(first_candidate_us, candidate_us)
            if first_candidate_us is not None and candidate_us is not None
            else None
        )
        baseline_spread = max(
            float(cold_baseline.get("sample_spread_ratio") or float("inf")),
            float(baseline.get("sample_spread_ratio") or float("inf")),
        )
        candidate_spread = max(
            float(first_candidate.get("sample_spread_ratio") or float("inf")),
            float(candidate.get("sample_spread_ratio") or float("inf")),
        )
        conservative_baseline_us = (
            min(cold_us, baseline_us)
            if cold_us is not None and baseline_us is not None
            else None
        )
        conservative_candidate_us = (
            max(first_candidate_us, candidate_us)
            if first_candidate_us is not None and candidate_us is not None
            else None
        )
        conservative_speedup = (
            conservative_baseline_us / conservative_candidate_us
            if conservative_baseline_us is not None
            and conservative_candidate_us is not None
            else None
        )
        stable = bool(
            baseline_spread <= float(args.maximum_baseline_spread)
            and candidate_spread <= float(args.maximum_candidate_spread)
            and candidate_run_ratio is not None
            and candidate_run_ratio <= float(args.maximum_candidate_spread)
        )
        speedup = (
            baseline_us / candidate_us
            if baseline_us is not None and candidate_us is not None
            else None
        )
        apply_change = bool(
            stable
            and conservative_speedup is not None
            and conservative_speedup >= float(args.minimum_speedup)
        )
        decision_baseline_us = conservative_baseline_us
        decision_candidate_us = conservative_candidate_us
        selected_us = decision_candidate_us if apply_change else decision_baseline_us
        baseline_ms = (
            decision_baseline_us * layer_invocations / 1000.0
            if decision_baseline_us is not None
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
            "winner": "fused_prepare" if apply_change else "current_prepare",
            "baseline_us": baseline_us,
            "cold_baseline_us": cold_us,
            "candidate_us": candidate_us,
            "first_candidate_us": first_candidate_us,
            "selected_us": selected_us,
            "baseline_run_ratio": baseline_run_ratio,
            "candidate_run_ratio": candidate_run_ratio,
            "decision_baseline_us": decision_baseline_us,
            "decision_candidate_us": decision_candidate_us,
            "baseline_sample_spread_ratio": baseline_spread,
            "candidate_sample_spread_ratio": candidate_spread,
            "stable": stable,
            "speedup": speedup,
            "conservative_speedup": conservative_speedup,
            "layer_invocations": layer_invocations,
            "estimated_baseline_ms_b16_prefill": baseline_ms,
            "estimated_selected_ms_b16_prefill": selected_ms,
            "estimated_savings_ms_b16_prefill": (
                baseline_ms - selected_ms
                if baseline_ms is not None and selected_ms is not None
                else None
            ),
            "shape": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "k_eq_v": k_eq_v,
            },
            "cases": measured,
        }
        print("PREPARE_DECISION " + json.dumps(summary, sort_keys=True))

        del (
            q_raw,
            k_raw,
            v_raw,
            q_weight,
            k_weight,
            positions,
            frequencies,
            cos,
            sin,
            q_fused,
            k_fused,
            v_fused,
            k_cache_fused,
            v_cache_fused,
            reference,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return summary

    sliding = (
        run_shape(
            attention_type="sliding",
            kv_heads=8,
            head_dim=256,
            k_eq_v=False,
            layer_invocations=25 * b16_chunks,
        )
        if args.only in ("all", "sliding")
        else None
    )
    full = (
        run_shape(
            attention_type="full",
            kv_heads=2,
            head_dim=512,
            k_eq_v=True,
            layer_invocations=5 * b16_chunks,
        )
        if args.only in ("all", "full")
        else None
    )

    active_results = [result for result in (sliding, full) if result is not None]

    baseline_parts = tuple(
        result.get("estimated_baseline_ms_b16_prefill")
        for result in active_results
    )
    selected_parts = tuple(
        result.get("estimated_selected_ms_b16_prefill")
        for result in active_results
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
            "APPLY_SELECTED"
            if any(result["apply_change"] for result in active_results)
            else "KEEP_CURRENT"
        ),
        "apply_change": any(result["apply_change"] for result in active_results),
        "gpu": gpu,
        "capability": list(capability),
        "shape": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "q_heads": q_heads,
            "dtype": "bf16",
        },
        "b16_chunks": b16_chunks,
        "measured_attention_types": args.only,
        "precondition_pairs": int(args.precondition_pairs),
        "minimum_speedup": float(args.minimum_speedup),
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

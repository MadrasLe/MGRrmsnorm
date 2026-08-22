#!/usr/bin/env python3
"""Measure Gemma 4 short-prefill SDPA with explicit versus implicit causality."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch

from megagemm.kernels.paged_attention import prefill_attention


def _measure_us(
    fn: Callable[[], torch.Tensor],
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
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
    return samples


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            left.float().reshape(1, -1),
            right.float().reshape(1, -1),
        ).item()
    )


def _run_shape(
    *,
    name: str,
    batch_size: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    layer_count: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict:
    device = torch.device("cuda")
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

    causal_bool = torch.ones(
        seq_len,
        seq_len,
        device=device,
        dtype=torch.bool,
    ).tril_()
    explicit_mask = torch.zeros(
        batch_size,
        1,
        seq_len,
        seq_len,
        device=device,
        dtype=dtype,
    )
    explicit_mask.masked_fill_(
        ~causal_bool.unsqueeze(0).unsqueeze(0),
        float("-inf"),
    )
    local_sliding_mask = torch.zeros(
        1,
        1,
        seq_len,
        seq_len,
        device=device,
        dtype=dtype,
    )
    local_sliding_mask.masked_fill_(
        ~causal_bool.unsqueeze(0).unsqueeze(0),
        float("-inf"),
    )

    def explicit() -> torch.Tensor:
        attn_mask = (
            local_sliding_mask + explicit_mask
            if name == "sliding"
            else explicit_mask
        )
        return prefill_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=attn_mask,
            scale=1.0,
        )

    def implicit() -> torch.Tensor:
        return prefill_attention(
            q,
            k,
            v,
            is_causal=True,
            attn_mask=None,
            scale=1.0,
        )

    explicit_ref = explicit()
    implicit_ref = implicit()
    torch.cuda.synchronize()
    max_abs_error = float(
        (explicit_ref.float() - implicit_ref.float()).abs().max().item()
    )
    cosine = _cosine(explicit_ref, implicit_ref)

    explicit_samples = _measure_us(
        explicit,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    implicit_samples = _measure_us(
        implicit,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    explicit_recheck = _measure_us(
        explicit,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )

    explicit_us = statistics.median(explicit_samples)
    implicit_us = statistics.median(implicit_samples)
    recheck_us = statistics.median(explicit_recheck)
    baseline_us = min(explicit_us, recheck_us)
    stability_ratio = max(explicit_us, recheck_us) / baseline_us
    speedup = baseline_us / implicit_us
    correct = cosine >= 0.9999 and max_abs_error <= 0.125
    use_implicit = (
        name == "sliding"
        and correct
        and stability_ratio <= 1.03
        and speedup >= 1.02
    )
    selected_us = implicit_us if use_implicit else baseline_us

    return {
        "attention_type": name,
        "shape": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "dtype": str(dtype).removeprefix("torch."),
        },
        "layers": layer_count,
        "correct": correct,
        "cosine": cosine,
        "max_abs_error": max_abs_error,
        "explicit_mask_us": explicit_us,
        "explicit_mask_recheck_us": recheck_us,
        "implicit_causal_us": implicit_us,
        "baseline_stability_ratio": stability_ratio,
        "speedup": speedup,
        "selected_path": (
            "implicit_causal" if use_implicit else "explicit_mask"
        ),
        "selected_us": selected_us,
        "estimated_explicit_ms": baseline_us * layer_count / 1000.0,
        "estimated_implicit_ms": implicit_us * layer_count / 1000.0,
        "estimated_selected_ms": selected_us * layer_count / 1000.0,
        "estimated_savings_ms": (
            (baseline_us - selected_us) * layer_count / 1000.0
        ),
        "samples_us": {
            "explicit_mask": explicit_samples,
            "implicit_causal": implicit_samples,
            "explicit_mask_recheck": explicit_recheck,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_prefill_attention_mask_a100.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(20260728)
    dtype = torch.bfloat16
    gpu = torch.cuda.get_device_name(0)
    print("Gemma4 short-prefill attention mask gate")
    print(f"  gpu: {gpu}")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  shape: batch=16 context=25 dtype=bf16")

    common = {
        "batch_size": 16,
        "seq_len": 25,
        "q_heads": 16,
        "dtype": dtype,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
    }
    results = [
        _run_shape(
            name="sliding",
            kv_heads=8,
            head_dim=256,
            layer_count=25,
            **common,
        ),
        _run_shape(
            name="full",
            kv_heads=2,
            head_dim=512,
            layer_count=5,
            **common,
        ),
    ]
    for result in results:
        print(json.dumps(result, sort_keys=True))

    total_explicit_ms = sum(row["estimated_explicit_ms"] for row in results)
    total_implicit_ms = sum(row["estimated_implicit_ms"] for row in results)
    total_selected_ms = sum(row["estimated_selected_ms"] for row in results)
    total_savings_ms = total_explicit_ms - total_selected_ms
    correct = all(row["correct"] for row in results)
    stable = all(row["baseline_stability_ratio"] <= 1.03 for row in results)
    speedup = (
        total_explicit_ms / total_selected_ms
        if total_selected_ms > 0.0
        else None
    )
    selected_paths = {
        row["attention_type"]: row["selected_path"] for row in results
    }
    decision = (
        "KEEP_SLIDING_ONLY"
        if (
            correct
            and stable
            and selected_paths == {
                "sliding": "implicit_causal",
                "full": "explicit_mask",
            }
            and speedup is not None
            and speedup >= 1.02
        )
        else "ROLL_BACK"
    )
    summary = {
        "decision": decision,
        "correct": correct,
        "stable": stable,
        "gpu": gpu,
        "shape": {"batch_size": 16, "seq_len": 25, "dtype": "bf16"},
        "estimated_explicit_ms_30_layers": total_explicit_ms,
        "estimated_implicit_ms_30_layers": total_implicit_ms,
        "estimated_selected_ms_30_layers": total_selected_ms,
        "estimated_savings_ms_30_layers": total_savings_ms,
        "selected_paths": selected_paths,
        "speedup": speedup,
        "results": results,
    }
    print("DECISION " + json.dumps(summary, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0 if correct else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Synthetic Gemma4 A4B attention-preparation benchmark (no model download)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

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


def measure(
    fn: Callable[[], None], *, warmup: int, iterations: int, repeats: int
) -> dict:
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
    return {
        "median_us": statistics.median(samples),
        "samples_us": samples,
        "measurement": "eager_cuda_events",
    }


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    ref = reference.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    return (
        float((ref - cand).abs().max().item()),
        float(F.cosine_similarity(ref, cand, dim=0).item()),
    )


def benchmark_shape(
    *,
    name: str,
    rows: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    k_eq_v: bool,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> list[dict]:
    eps = 1e-6
    q_raw = torch.randn(
        (1, rows, q_heads * head_dim), device="cuda", dtype=dtype
    )
    k_raw = torch.randn(
        (1, rows, kv_heads * head_dim), device="cuda", dtype=dtype
    )
    v_raw = k_raw if k_eq_v else torch.randn_like(k_raw)
    q_weight = torch.randn((head_dim,), device="cuda", dtype=dtype)
    k_weight = torch.randn((head_dim,), device="cuda", dtype=dtype)
    positions = torch.arange(rows, device="cuda", dtype=torch.long).unsqueeze(0)
    frequencies = torch.randn((rows, head_dim // 2), device="cuda") * 0.01
    cos = torch.cos(frequencies)
    sin = torch.sin(frequencies)

    q_fused = torch.empty(
        (1, q_heads, rows, head_dim), device="cuda", dtype=dtype
    )
    k_fused = torch.empty(
        (1, kv_heads, rows, head_dim), device="cuda", dtype=dtype
    )
    v_fused = torch.empty_like(k_fused)
    k_cache_fused = torch.empty(
        (1, rows, kv_heads, head_dim), device="cuda", dtype=dtype
    )
    v_cache_fused = torch.empty_like(k_cache_fused)

    def reference() -> tuple[torch.Tensor, ...]:
        q = q_raw.view(1, rows, q_heads, head_dim).transpose(1, 2).contiguous()
        k = k_raw.view(1, rows, kv_heads, head_dim).transpose(1, 2).contiguous()
        v = v_raw.view(1, rows, kv_heads, head_dim).transpose(1, 2).contiguous()
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

    def fused() -> None:
        gemma4_prefill_attention_prepare(
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

    reference_outputs = reference()
    fused()
    torch.cuda.synchronize()
    max_abs = 0.0
    min_cosine = 1.0
    for ref, candidate in zip(
        reference_outputs,
        (q_fused, k_fused, v_fused, k_cache_fused, v_cache_fused),
    ):
        error, cosine = tensor_error(ref, candidate)
        max_abs = max(max_abs, error)
        min_cosine = min(min_cosine, cosine)

    rows_out = []
    for case, fn in (("current_prepare", reference), ("fused_prepare", fused)):
        timing = measure(
            fn,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        row = {
            "attention_type": name,
            "case": case,
            "rows": rows,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "k_eq_v": k_eq_v,
            "max_abs_error": max_abs if case == "fused_prepare" else 0.0,
            "min_cosine": min_cosine if case == "fused_prepare" else 1.0,
            **timing,
        }
        rows_out.append(row)
        print(json.dumps(row, sort_keys=True))
    return rows_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_attention_prepare_a100.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if "A100" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("This exact-shape decision benchmark requires an A100")

    print("Gemma4 fused attention-prepare microbenchmark")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(f"  capability: {torch.cuda.get_device_capability(0)}")
    print(f"  rows: {args.rows}")
    print("  dtype: bf16")
    print("  model download: none")

    torch.manual_seed(20260712)
    with torch.inference_mode():
        results = benchmark_shape(
            name="sliding",
            rows=args.rows,
            q_heads=16,
            kv_heads=8,
            head_dim=256,
            k_eq_v=False,
            dtype=torch.bfloat16,
            args=args,
        )
        results.extend(
            benchmark_shape(
                name="full",
                rows=args.rows,
                q_heads=16,
                kv_heads=2,
                head_dim=512,
                k_eq_v=True,
                dtype=torch.bfloat16,
                args=args,
            )
        )

    by_key = {(row["attention_type"], row["case"]): row for row in results}
    current_us = (
        25.0 * by_key[("sliding", "current_prepare")]["median_us"]
        + 5.0 * by_key[("full", "current_prepare")]["median_us"]
    )
    fused_us = (
        25.0 * by_key[("sliding", "fused_prepare")]["median_us"]
        + 5.0 * by_key[("full", "fused_prepare")]["median_us"]
    )
    max_abs = max(
        row["max_abs_error"] for row in results if row["case"] == "fused_prepare"
    )
    min_cosine = min(
        row["min_cosine"] for row in results if row["case"] == "fused_prepare"
    )
    correct = max_abs <= 0.03125 and min_cosine >= 0.999
    speedup = current_us / fused_us
    decision = {
        "decision": "APPLY" if correct and speedup >= 1.05 else "REVERT",
        "correct": correct,
        "max_abs_error": max_abs,
        "min_cosine": min_cosine,
        "current_ms_per_prefill": current_us / 1000.0,
        "fused_ms_per_prefill": fused_us / 1000.0,
        "estimated_savings_ms": (current_us - fused_us) / 1000.0,
        "speedup": speedup,
    }
    print("\nDECISION " + json.dumps(decision, sort_keys=True))
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "rows": args.rows,
        "results": results,
        "decision": decision,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0 if decision["decision"] == "APPLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

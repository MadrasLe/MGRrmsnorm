"""
Benchmark MegaGemm scheduler throughput for before/after code changes.

Designed for Colab T4 runs:

    python benchmarks/benchmark_scheduler_tps.py \
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --batch-sizes 1,2,4,8 \
      --max-new-tokens 64 \
      --repeats 3 \
      --out baseline_scheduler_tps.json

Run the exact same command after a scheduler change and compare the JSON files.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch


def _parse_csv_ints(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _runtime_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _empty_cache(device: str) -> None:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "device_index": idx,
        "name": torch.cuda.get_device_name(idx),
        "capability": list(torch.cuda.get_device_capability(idx)),
        "total_gb": total / 1024**3,
        "free_gb": free / 1024**3,
        "multiprocessors": props.multi_processor_count,
    }


def _percent_delta(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return (new - old) / old * 100.0


def _run_short_completion_smoke(engine, prompt: str) -> dict[str, Any]:
    """Exercise the scheduler edge case without aborting the TPS benchmark."""
    try:
        outputs = engine.generate_batch(
            [prompt, prompt],
            max_new_tokens=1,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
        )
        return {
            "ok": True,
            "num_outputs": len(outputs),
            "output_lengths_chars": [len(text) for text in outputs],
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "num_outputs": 0,
            "output_lengths_chars": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_one_batch(
    engine,
    *,
    batch_size: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str,
) -> dict[str, Any]:
    prompts = [prompt] * batch_size
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    start = time.perf_counter()
    outputs = engine.generate_batch(
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    _sync(device)
    elapsed = time.perf_counter() - start

    stats = {}
    scheduler = getattr(engine, "_last_scheduler", None)
    if scheduler is not None:
        stats = scheduler.get_stats()
    generated_tokens = int(stats.get("total_tokens") or (batch_size * max_new_tokens))
    peak_gb = None
    if device == "cuda" and torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / elapsed if elapsed > 0 else 0.0,
        "tokens_per_second_per_request": (
            generated_tokens / elapsed / batch_size if elapsed > 0 else 0.0
        ),
        "peak_allocated_gb": peak_gb,
        "num_outputs": len(outputs),
        "scheduler_stats": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model id, local snapshot, or .mgx artifact.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quantize", choices=["int8", "fp8", "awq"])
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument("--kv-alloc", default="auto", choices=["auto", "greedy"])
    parser.add_argument(
        "--prompt",
        default="Explique em poucas frases o que e inferencia de modelos de linguagem.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--out", default="scheduler_tps.json")
    parser.add_argument(
        "--skip-short-smoke",
        action="store_true",
        help="Do not run the max_new_tokens=1 scheduler edge-case smoke test.",
    )
    args = parser.parse_args()

    batch_sizes = _parse_csv_ints(args.batch_sizes)
    max_batch_size = args.max_batch_size or max(batch_sizes)
    dtype = _runtime_dtype(args.dtype)

    print("MegaGemm scheduler TPS benchmark")
    print(f"  model:          {args.model}")
    print(f"  device:         {args.device}")
    print(f"  dtype:          {args.dtype}")
    print(f"  quantize:       {args.quantize or 'none'}")
    print(f"  batch sizes:    {batch_sizes}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  repeats:        {args.repeats}")
    print(f"  gpu:            {_gpu_snapshot()}")

    from megagemm.engine import InferenceEngine

    _empty_cache(args.device)
    engine = InferenceEngine(
        args.model,
        dtype=dtype,
        device=args.device,
        quantize=args.quantize,
        max_batch_size=max_batch_size,
        max_seq_len=args.max_seq_len,
        num_blocks=args.num_blocks,
        kv_alloc=args.kv_alloc,
    )

    print("\nWarmup")
    engine.generate_batch(
        [args.prompt],
        max_new_tokens=args.warmup_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
    )
    _sync(args.device)

    short_smoke = None
    if not args.skip_short_smoke:
        print("\nShort-completion smoke (max_new_tokens=1)")
        short_smoke = _run_short_completion_smoke(engine, args.prompt)
        print(f"  ok={short_smoke['ok']} error={short_smoke['error']}")

    rows = []
    print("\nTimed runs")
    for batch_size in batch_sizes:
        samples = []
        for repeat_idx in range(args.repeats):
            _empty_cache(args.device)
            sample = _run_one_batch(
                engine,
                batch_size=batch_size,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
            )
            samples.append(sample)
            print(
                f"  batch={batch_size:2d} repeat={repeat_idx + 1}/{args.repeats} "
                f"{sample['tokens_per_second']:8.2f} tok/s "
                f"({sample['tokens_per_second_per_request']:7.2f} tok/s/req, "
                f"{sample['elapsed_seconds']:.3f}s)"
            )

        tps_values = [sample["tokens_per_second"] for sample in samples]
        median_tps = statistics.median(tps_values)
        best_tps = max(tps_values)
        worst_tps = min(tps_values)
        row = {
            "batch_size": batch_size,
            "median_tokens_per_second": median_tps,
            "best_tokens_per_second": best_tps,
            "worst_tokens_per_second": worst_tps,
            "median_tokens_per_second_per_request": median_tps / batch_size,
            "spread_percent_best_vs_worst": _percent_delta(best_tps, worst_tps),
            "samples": samples,
        }
        rows.append(row)
        print(
            f"  batch={batch_size:2d} median={median_tps:8.2f} tok/s "
            f"best={best_tps:8.2f} worst={worst_tps:8.2f}"
        )

    result = {
        "benchmark": "scheduler_tps",
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "quantize": args.quantize or "none",
        "batch_sizes": batch_sizes,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "max_seq_len": args.max_seq_len,
        "max_batch_size": max_batch_size,
        "num_blocks": args.num_blocks,
        "kv_alloc": args.kv_alloc,
        "gpu": _gpu_snapshot(),
        "short_completion_smoke": short_smoke,
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("\nSummary")
    for row in rows:
        print(
            f"  batch={row['batch_size']:2d}: "
            f"median {row['median_tokens_per_second']:.2f} tok/s total, "
            f"{row['median_tokens_per_second_per_request']:.2f} tok/s/req"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark MegaMesh replica workers over HTTP."""

from __future__ import annotations

import argparse
import json
import time
from statistics import mean

from megagemm.mesh import MeshRouter


def parse_int_list(value) -> list[int]:
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            parts.extend(str(item).split(","))
    else:
        parts = str(value).split(",")
    return [int(part.strip()) for part in parts if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="MegaMesh distributed replica benchmark")
    parser.add_argument("--workers", required=True, help="Comma-separated worker URLs, optionally url@weight#name")
    parser.add_argument("--batch-sizes", nargs="+", default="1,2,4,8", help="Batch sizes to test")
    parser.add_argument("--prompt", default="Explain distributed inference in one paragraph.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    router = MeshRouter(args.workers, timeout=args.timeout)
    batch_sizes = parse_int_list(args.batch_sizes)

    print("MegaMesh Replica Benchmark")
    print(f"Workers: {args.workers}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("\nHealth:")
    for row in router.health():
        label = row.get("name") or row.get("endpoint")
        gpu = row.get("gpu") or {}
        gpu_name = gpu.get("name", "unknown")
        print(f"  {label}: ok={row.get('ok')} weight={row.get('configured_weight')} gpu={gpu_name}")

    rows = []
    for batch_size in batch_sizes:
        prompts = [f"{args.prompt} [{i}]" for i in range(batch_size)]
        for _ in range(args.warmup):
            router.generate_batch_with_stats(
                prompts,
                max_new_tokens=min(args.max_new_tokens, 16),
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

        times = []
        tokens = []
        worker_snapshots = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            _, stats = router.generate_batch_with_stats(
                prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            tokens.append(int(stats.get("generated_tokens") or 0))
            worker_snapshots.append(stats.get("workers", []))

        avg_time = mean(times)
        avg_tokens = mean(tokens)
        total_tps = avg_tokens / max(avg_time, 1e-9)
        row = {
            "batch_size": batch_size,
            "generated_tokens": round(avg_tokens, 2),
            "total_ms": round(avg_time * 1000.0, 2),
            "total_tps": round(total_tps, 2),
            "per_seq_tps": round(total_tps / batch_size, 2),
            "workers_last_run": worker_snapshots[-1] if worker_snapshots else [],
        }
        rows.append(row)
        print(
            f"  batch={batch_size:>3} | total={row['total_tps']:>7.2f} tok/s | "
            f"per-seq={row['per_seq_tps']:>6.2f} tok/s | "
            f"time={row['total_ms']:>8.2f} ms"
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": "megamesh_replica",
                    "workers": args.workers,
                    "max_new_tokens": args.max_new_tokens,
                    "rows": rows,
                },
                f,
                indent=2,
            )
        print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()

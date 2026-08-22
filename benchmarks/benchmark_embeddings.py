"""
Benchmark MegaGemm embedding throughput against backend variants.

Examples:
    python benchmarks/benchmark_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
    python benchmarks/benchmark_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2 --device cuda --dtype fp16 --batch-sizes 32,64,128
    python benchmarks/benchmark_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2 --device cuda --dtype fp16 --scenario mixed --batch-sizes 32,64,128 --max-batch-tokens-grid 0,2048,4096
    python benchmarks/benchmark_embeddings.py --model /path/to/local/model --local-files-only
"""

from __future__ import annotations

import argparse
import json
import time

from megagemm.embeddings import EmbeddingEngine
import torch


def build_uniform_texts(num_texts: int) -> list[str]:
    return ["MegaGemm embedding benchmark sentence." for _ in range(num_texts)]


def build_mixed_texts(num_texts: int) -> list[str]:
    lengths = [4, 8, 12, 20, 32, 48, 64, 96]
    texts = []
    for i in range(num_texts):
        n = lengths[i % len(lengths)]
        texts.append(" ".join([f"token{i % 7}"] * n))
    return texts


def run_case(
    model: str,
    label: str,
    backend: str,
    native_padding_free: bool,
    texts: list[str],
    device: str,
    dtype: str,
    batch_size: int,
    max_batch_tokens: int,
    runs: int,
    warmup: int,
    local_files_only: bool,
    task: str | None,
):
    engine = EmbeddingEngine(
        model,
        device=device,
        dtype=dtype,
        backend=backend,
        native_padding_free=native_padding_free,
        max_batch_tokens=max_batch_tokens,
        local_files_only=local_files_only,
    )
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        stats = engine.benchmark(texts, batch_size=batch_size, runs=runs, warmup=warmup, task=task)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        stats["peak_vram_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
        stats["wall_clock_ms"] = wall_ms
    else:
        stats = engine.benchmark(texts, batch_size=batch_size, runs=runs, warmup=warmup, task=task)
    stats["backend_label"] = label
    stats["backend_requested"] = backend
    stats["native_padding_free_enabled"] = bool(native_padding_free)
    return engine, stats


def _parse_csv_ints(raw: str) -> list[int]:
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def _format_speedup(baseline: float, current: float) -> str:
    if baseline <= 0 or current <= 0:
        return "-"
    return f"{current / baseline:.3f}x"


def _print_device_header(device: str, dtype: str) -> None:
    print("Embedding Benchmark")
    print(f"  Torch:   {torch.__version__}")
    print(f"  Device:  {device}")
    print(f"  Dtype:   {dtype}")
    if device == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:     {name}")
        print(f"  VRAM:    {props.total_memory / 1e9:.1f} GB")
        print(f"  SMs:     {props.multi_processor_count}")
    print()


def _print_case_table(name: str, rows: list[dict]) -> None:
    print(f"\n=== {name} ===")
    print(
        f"{'backend':<16}"
        f"{'batch':>8}"
        f"{'max_tok':>10}"
        f"{'texts/s':>12}"
        f"{'tok/s':>12}"
        f"{'lat_ms':>12}"
        f"{'speedup':>10}"
        f"{'vram_mb':>12}"
    )
    baseline = None
    for row in rows:
        if baseline is None and row["backend_label"] == "hf" and row["max_batch_tokens"] == 0:
            baseline = row["texts_per_second"]
        speedup = _format_speedup(baseline or 0.0, row["texts_per_second"])
        vram = row.get("peak_vram_mb")
        vram_str = f"{vram:.0f}" if isinstance(vram, (int, float)) else "-"
        print(
            f"{row['backend_label']:<16}"
            f"{int(row['batch_size']):>8}"
            f"{int(row['max_batch_tokens']):>10}"
            f"{row['texts_per_second']:>12.1f}"
            f"{row['tokens_per_second']:>12.1f}"
            f"{row['avg_latency_ms']:>12.1f}"
            f"{speedup:>10}"
            f"{vram_str:>12}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark MegaGemm embedding backends")
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local snapshot path")
    parser.add_argument("--scenario", choices=["uniform", "mixed", "both"], default="both")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--task", choices=["query", "document", "passage"], help="Optional retrieval task prompt")
    parser.add_argument("--batch-size", type=int, default=32, help="Single batch size for compatibility")
    parser.add_argument("--batch-sizes", default="", help="Comma-separated batch sizes; overrides --batch-size")
    parser.add_argument("--num-texts", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-batch-tokens", type=int, default=0, help="Token-budget batching cap for MegaGemm")
    parser.add_argument("--max-batch-tokens-grid", default="", help="Comma-separated token caps to test (e.g. 0,1024,2048)")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-native-dense", action="store_true")
    parser.add_argument("--skip-native-packed", action="store_true")
    parser.add_argument("--skip-auto", action="store_true")
    parser.add_argument("--json-out", help="Optional path to save raw benchmark results as JSON")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_sizes = _parse_csv_ints(args.batch_sizes) if args.batch_sizes.strip() else [args.batch_size]
    token_caps = _parse_csv_ints(args.max_batch_tokens_grid) if args.max_batch_tokens_grid.strip() else [args.max_batch_tokens]

    _print_device_header(device=device, dtype=args.dtype)

    scenarios = []
    if args.scenario in {"uniform", "both"}:
        scenarios.append(("uniform", build_uniform_texts(args.num_texts)))
    if args.scenario in {"mixed", "both"}:
        scenarios.append(("mixed", build_mixed_texts(args.num_texts)))

    all_results = {
        "model": args.model,
        "device": device,
        "dtype": args.dtype,
        "scenario": args.scenario,
        "batch_sizes": batch_sizes,
        "max_batch_tokens_grid": token_caps,
        "runs": args.runs,
        "warmup": args.warmup,
        "results": [],
    }

    for name, texts in scenarios:
        case_rows = []
        for batch_size in batch_sizes:
            if not args.skip_hf:
                _, hf_stats = run_case(
                    args.model,
                    label="hf",
                    backend="hf",
                    native_padding_free=False,
                    texts=texts,
                    device=device,
                    dtype=args.dtype,
                    batch_size=batch_size,
                    max_batch_tokens=0,
                    runs=args.runs,
                    warmup=args.warmup,
                    local_files_only=args.local_files_only,
                    task=args.task,
                )
                hf_stats["scenario"] = name
                case_rows.append(hf_stats)

            for token_cap in token_caps:
                if not (args.skip_native or args.skip_native_dense):
                    _, native_dense_stats = run_case(
                        args.model,
                        label="native_dense",
                        backend="native",
                        native_padding_free=False,
                        texts=texts,
                        device=device,
                        dtype=args.dtype,
                        batch_size=batch_size,
                        max_batch_tokens=token_cap,
                        runs=args.runs,
                        warmup=args.warmup,
                        local_files_only=args.local_files_only,
                        task=args.task,
                    )
                    native_dense_stats["scenario"] = name
                    case_rows.append(native_dense_stats)

                if not (args.skip_native or args.skip_native_packed):
                    _, native_packed_stats = run_case(
                        args.model,
                        label="native_packed",
                        backend="native",
                        native_padding_free=True,
                        texts=texts,
                        device=device,
                        dtype=args.dtype,
                        batch_size=batch_size,
                        max_batch_tokens=token_cap,
                        runs=args.runs,
                        warmup=args.warmup,
                        local_files_only=args.local_files_only,
                        task=args.task,
                    )
                    native_packed_stats["scenario"] = name
                    case_rows.append(native_packed_stats)

                if not args.skip_auto:
                    _, auto_stats = run_case(
                        args.model,
                        label="auto",
                        backend="auto",
                        native_padding_free=True,
                        texts=texts,
                        device=device,
                        dtype=args.dtype,
                        batch_size=batch_size,
                        max_batch_tokens=token_cap,
                        runs=args.runs,
                        warmup=args.warmup,
                        local_files_only=args.local_files_only,
                        task=args.task,
                    )
                    auto_stats["scenario"] = name
                    case_rows.append(auto_stats)

        _print_case_table(name, case_rows)
        all_results["results"].extend(case_rows)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON results to {args.json_out}")


if __name__ == "__main__":
    main()

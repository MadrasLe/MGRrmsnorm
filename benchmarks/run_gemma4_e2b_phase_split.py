"""Paired prefill/first-token and incremental-decode measurement for Gemma 4 E2B.

The benchmark deliberately measures one backend per process so the model is
loaded only once while the 1-token and 128-token cases share the same engine.
The paired difference removes model load and approximates the cost of the
remaining decode tokens without depending on backend-private timing hooks.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_inference_matrix as matrix


DEFAULT_MODEL = "google/gemma-4-E2B-it"


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _configure_megagemm_profile(model: str) -> dict[str, str]:
    """Apply the controlled publication profile, ignoring notebook leftovers."""
    from benchmarks.run_publication_gpu_suite import profile_environment

    for key in list(os.environ):
        if key.startswith("MEGAGEMM_"):
            os.environ.pop(key, None)
    profile = profile_environment("gemma4-e2b-fast", model)
    os.environ.update(profile)
    return profile


def _runner_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        backend=args.backend,
        model=args.model,
        tokenizer=args.tokenizer,
        dtype="bf16",
        quantize=None,
        device="cuda",
        max_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_blocks=0,
        block_size=16,
        kv_alloc="auto",
        kv_offload=False,
        num_cpu_blocks=0,
        gpu_window=64,
        cache_dir=args.cache_dir,
        mgx_prefer_payload_cache=False,
        mgx_payload_cache_dir=None,
        ignore_eos=True,
        local_files_only=args.local_files_only,
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.max_seq_len,
        vllm_max_num_seqs=args.batch_size,
        vllm_max_num_batched_tokens=0,
        vllm_enforce_eager=False,
        vllm_language_model_only=True,
        vllm_disable_prefix_caching=True,
        vllm_disable_cudagraph_memory_profiler=False,
    )


def _sample(
    runner,
    prompts: list[str],
    *,
    token_count: int,
    expected_generated_tokens: int,
    pair_index: int,
    order_position: int,
) -> dict[str, Any]:
    matrix.cleanup_cuda()
    result = runner(prompts, token_count)
    elapsed_s = float(result["elapsed_s"])
    generated_tokens = int(result["generated_tokens"])
    if generated_tokens != expected_generated_tokens:
        raise RuntimeError(
            f"generated {generated_tokens} tokens for max_new_tokens={token_count}; "
            f"expected exactly {expected_generated_tokens} with ignore_eos enabled"
        )
    extra = dict(result.get("extra") or {})
    scheduler = extra.get("scheduler_stats")
    internal: dict[str, float] = {}
    if isinstance(scheduler, dict):
        for source, target in (
            ("prefill_time_ms", "prefill_ms"),
            ("decode_time_ms", "decode_ms"),
            ("decode_throughput", "decode_tps"),
        ):
            value = scheduler.get(source)
            if value is not None:
                internal[target] = float(value)
    return {
        "pair_index": pair_index,
        "order_position": order_position,
        "max_new_tokens_per_request": token_count,
        "generated_tokens": generated_tokens,
        "elapsed_s": elapsed_s,
        "output_tps": generated_tokens / elapsed_s,
        "internal": internal,
    }


def summarize_samples(
    samples: list[dict[str, Any]],
    *,
    batch_size: int,
    short_tokens: int,
    long_tokens: int,
) -> dict[str, Any]:
    grouped: dict[int, dict[int, dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(int(sample["pair_index"]), {})[
            int(sample["max_new_tokens_per_request"])
        ] = sample

    pairs: list[dict[str, Any]] = []
    for pair_index in sorted(grouped):
        pair = grouped[pair_index]
        if short_tokens not in pair or long_tokens not in pair:
            continue
        short = pair[short_tokens]
        long = pair[long_tokens]
        delta_s = float(long["elapsed_s"]) - float(short["elapsed_s"])
        pairs.append(
            {
                "pair_index": pair_index,
                "short_elapsed_s": float(short["elapsed_s"]),
                "long_elapsed_s": float(long["elapsed_s"]),
                "incremental_decode_s": delta_s,
                "short_fraction_of_long": (
                    float(short["elapsed_s"]) / float(long["elapsed_s"])
                ),
                "incremental_fraction_of_long": delta_s / float(long["elapsed_s"]),
            }
        )

    if not pairs:
        raise RuntimeError("no complete short/long measurement pairs")
    bad_pairs = [pair for pair in pairs if pair["incremental_decode_s"] <= 0.0]
    if bad_pairs:
        raise RuntimeError(f"non-positive paired decode deltas: {bad_pairs}")

    short_elapsed = [float(pair["short_elapsed_s"]) for pair in pairs]
    long_elapsed = [float(pair["long_elapsed_s"]) for pair in pairs]
    decode_elapsed = [float(pair["incremental_decode_s"]) for pair in pairs]
    incremental_tokens = batch_size * (long_tokens - short_tokens)
    long_generated_tokens = batch_size * long_tokens

    long_internal_prefill = [
        float(sample["internal"]["prefill_ms"])
        for sample in samples
        if int(sample["max_new_tokens_per_request"]) == long_tokens
        and sample.get("internal", {}).get("prefill_ms") is not None
    ]
    long_internal_decode = [
        float(sample["internal"]["decode_ms"])
        for sample in samples
        if int(sample["max_new_tokens_per_request"]) == long_tokens
        and sample.get("internal", {}).get("decode_ms") is not None
    ]

    median_short = _median(short_elapsed)
    median_long = _median(long_elapsed)
    median_decode = _median(decode_elapsed)
    return {
        "complete_pairs": len(pairs),
        "pairs": pairs,
        "first_token_phase_ms": median_short * 1000.0,
        "incremental_decode_ms": median_decode * 1000.0,
        "incremental_decode_tokens": incremental_tokens,
        "incremental_decode_tps": incremental_tokens / median_decode,
        "long_total_ms": median_long * 1000.0,
        "long_output_tps": long_generated_tokens / median_long,
        "median_first_token_fraction": _median(
            [float(pair["short_fraction_of_long"]) for pair in pairs]
        ),
        "median_incremental_decode_fraction": _median(
            [float(pair["incremental_fraction_of_long"]) for pair in pairs]
        ),
        "internal_long_prefill_ms": _median(long_internal_prefill),
        "internal_long_decode_ms": _median(long_internal_decode),
        "method": {
            "first_token_phase": (
                f"wall time for {short_tokens} generated token per request; "
                "includes prefill, first-token compute, sampling, and API overhead"
            ),
            "incremental_decode": (
                f"paired wall-time difference between {long_tokens} and "
                f"{short_tokens} generated tokens per request"
            ),
        },
    }


def measure(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "megagemm":
        profile = _configure_megagemm_profile(args.model)
    else:
        profile = {}

    runtime_args = _runner_args(args)
    tokenizer_source = args.tokenizer or args.model
    tokenizer = matrix.load_tokenizer(
        tokenizer_source,
        local_files_only=args.local_files_only,
    )
    prompts, prompt_tokens_actual = matrix.build_prompts(
        tokenizer,
        args.batch_size,
        args.prompt_tokens,
    )

    print("Gemma 4 E2B paired phase split", flush=True)
    print(f"  backend:       {args.backend}", flush=True)
    print(f"  model:         {args.model}", flush=True)
    print(f"  batch:         {args.batch_size}", flush=True)
    print(f"  prompt:        {args.prompt_tokens} requested", flush=True)
    print(f"  prompt actual: {prompt_tokens_actual} total", flush=True)
    print(f"  token cases:   {args.short_tokens}, {args.long_tokens}", flush=True)
    print(f"  warmups:       {args.warmups} per token case", flush=True)
    print(f"  paired runs:   {args.repeats}", flush=True)

    runner = matrix.make_runner(runtime_args, tokenizer)
    for token_count in (args.short_tokens, args.long_tokens):
        expected = args.batch_size * token_count
        for warmup_index in range(args.warmups):
            print(
                f"Warmup tokens={token_count} "
                f"repeat={warmup_index + 1}/{args.warmups}",
                flush=True,
            )
            _sample(
                runner,
                prompts,
                token_count=token_count,
                expected_generated_tokens=expected,
                pair_index=-(warmup_index + 1),
                order_position=0,
            )

    samples: list[dict[str, Any]] = []
    for pair_index in range(1, args.repeats + 1):
        order = (
            (args.short_tokens, args.long_tokens)
            if pair_index % 2
            else (args.long_tokens, args.short_tokens)
        )
        for order_position, token_count in enumerate(order, start=1):
            print(
                f"Measure pair={pair_index}/{args.repeats} "
                f"position={order_position}/2 tokens={token_count}",
                flush=True,
            )
            sample = _sample(
                runner,
                prompts,
                token_count=token_count,
                expected_generated_tokens=args.batch_size * token_count,
                pair_index=pair_index,
                order_position=order_position,
            )
            samples.append(sample)
            print(
                f"  elapsed={sample['elapsed_s']:.6f}s "
                f"output={sample['output_tps']:.2f} tok/s",
                flush=True,
            )

    summary = summarize_samples(
        samples,
        batch_size=args.batch_size,
        short_tokens=args.short_tokens,
        long_tokens=args.long_tokens,
    )
    payload = {
        "benchmark": "gemma4_e2b_paired_phase_split",
        "backend": args.backend,
        "model": args.model,
        "dtype": "bf16",
        "hardware_label": "1xl4",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": _jsonable_args(args),
        "profile_environment": profile,
        "system": {
            "git": matrix.git_snapshot(),
            "gpu": matrix.gpu_snapshot(),
            "nvidia_smi": matrix.nvidia_smi_snapshot(),
            "packages": matrix.installed_package_versions(),
        },
        "prompt_tokens_actual_total": prompt_tokens_actual,
        "samples": samples,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nPHASE_SPLIT " + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"Wrote: {args.output}", flush=True)
    return payload


def compare_payloads(
    megagemm: dict[str, Any],
    vllm: dict[str, Any],
) -> dict[str, Any]:
    if megagemm.get("backend") != "megagemm":
        raise ValueError("--megagemm-json is not a MegaGemm measurement")
    if vllm.get("backend") != "vllm":
        raise ValueError("--vllm-json is not a vLLM measurement")
    for key in ("model", "dtype", "prompt_tokens_actual_total"):
        if megagemm.get(key) != vllm.get(key):
            raise ValueError(
                f"incomparable {key}: {megagemm.get(key)!r} != {vllm.get(key)!r}"
            )
    for key in ("batch_size", "prompt_tokens", "short_tokens", "long_tokens"):
        if megagemm["args"].get(key) != vllm["args"].get(key):
            raise ValueError(
                f"incomparable {key}: {megagemm['args'].get(key)!r} != "
                f"{vllm['args'].get(key)!r}"
            )

    mg = megagemm["summary"]
    vl = vllm["summary"]
    first_gap_ms = float(mg["first_token_phase_ms"]) - float(
        vl["first_token_phase_ms"]
    )
    decode_gap_ms = float(mg["incremental_decode_ms"]) - float(
        vl["incremental_decode_ms"]
    )
    total_gap_ms = float(mg["long_total_ms"]) - float(vl["long_total_ms"])
    accounted_gap_ms = first_gap_ms + decode_gap_ms
    positive_first = max(0.0, first_gap_ms)
    positive_decode = max(0.0, decode_gap_ms)
    positive_sum = positive_first + positive_decode
    if positive_sum > 0.0:
        first_share = positive_first / positive_sum
        decode_share = positive_decode / positive_sum
    else:
        first_share = 0.0
        decode_share = 0.0

    dominant = "first_token_phase" if first_gap_ms >= decode_gap_ms else "incremental_decode"
    return {
        "benchmark": "gemma4_e2b_paired_phase_comparison",
        "model": megagemm["model"],
        "dtype": megagemm["dtype"],
        "hardware_label": megagemm["hardware_label"],
        "workload": {
            "batch_size": megagemm["args"]["batch_size"],
            "prompt_tokens_requested": megagemm["args"]["prompt_tokens"],
            "prompt_tokens_actual_total": megagemm["prompt_tokens_actual_total"],
            "short_tokens": megagemm["args"]["short_tokens"],
            "long_tokens": megagemm["args"]["long_tokens"],
        },
        "megagemm": mg,
        "vllm": vl,
        "gaps": {
            "first_token_phase_ms": first_gap_ms,
            "incremental_decode_ms": decode_gap_ms,
            "long_total_ms": total_gap_ms,
            "accounted_gap_ms": accounted_gap_ms,
            "median_non_additivity_ms": total_gap_ms - accounted_gap_ms,
            "megagemm_vs_vllm_output_pct": (
                float(mg["long_output_tps"]) / float(vl["long_output_tps"]) - 1.0
            )
            * 100.0,
            "megagemm_vs_vllm_incremental_decode_pct": (
                float(mg["incremental_decode_tps"])
                / float(vl["incremental_decode_tps"])
                - 1.0
            )
            * 100.0,
        },
        "positive_gap_attribution": {
            "first_token_phase_fraction": first_share,
            "incremental_decode_fraction": decode_share,
            "dominant": dominant,
        },
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    megagemm = json.loads(args.megagemm_json.read_text(encoding="utf-8"))
    vllm = json.loads(args.vllm_json.read_text(encoding="utf-8"))
    result = compare_payloads(megagemm, vllm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    mg = result["megagemm"]
    vl = result["vllm"]
    gaps = result["gaps"]
    attribution = result["positive_gap_attribution"]
    print("\nGEMMA 4 E2B / L4 / BF16 — PAIRED PHASE SPLIT")
    print("metric                         MegaGemm        vLLM        MG-vLLM")
    print(
        f"first-token phase (ms)       {mg['first_token_phase_ms']:10.2f} "
        f"{vl['first_token_phase_ms']:10.2f} {gaps['first_token_phase_ms']:12.2f}"
    )
    print(
        f"incremental decode (ms)      {mg['incremental_decode_ms']:10.2f} "
        f"{vl['incremental_decode_ms']:10.2f} {gaps['incremental_decode_ms']:12.2f}"
    )
    print(
        f"incremental decode (tok/s)   {mg['incremental_decode_tps']:10.2f} "
        f"{vl['incremental_decode_tps']:10.2f}"
    )
    print(
        f"long total (ms)              {mg['long_total_ms']:10.2f} "
        f"{vl['long_total_ms']:10.2f} {gaps['long_total_ms']:12.2f}"
    )
    print(
        f"long output (tok/s)          {mg['long_output_tps']:10.2f} "
        f"{vl['long_output_tps']:10.2f}"
    )
    print(
        "gap attribution: "
        f"first-token={attribution['first_token_phase_fraction'] * 100.0:.1f}% "
        f"decode={attribution['incremental_decode_fraction'] * 100.0:.1f}% "
        f"dominant={attribution['dominant']}"
    )
    print("COMPARISON " + json.dumps(result, ensure_ascii=False))
    print(f"Wrote: {args.output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    measure_parser = commands.add_parser("measure")
    measure_parser.add_argument("--backend", required=True, choices=("megagemm", "vllm"))
    measure_parser.add_argument("--model", default=DEFAULT_MODEL)
    measure_parser.add_argument("--tokenizer")
    measure_parser.add_argument("--batch-size", type=int, default=8)
    measure_parser.add_argument("--prompt-tokens", type=int, default=2048)
    measure_parser.add_argument("--short-tokens", type=int, default=1)
    measure_parser.add_argument("--long-tokens", type=int, default=128)
    measure_parser.add_argument("--warmups", type=int, default=3)
    measure_parser.add_argument("--repeats", type=int, default=5)
    measure_parser.add_argument("--max-seq-len", type=int, default=2304)
    measure_parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    measure_parser.add_argument("--cache-dir")
    measure_parser.add_argument("--local-files-only", action="store_true")
    measure_parser.add_argument("--output", type=Path, required=True)

    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--megagemm-json", type=Path, required=True)
    compare_parser.add_argument("--vllm-json", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "measure":
        if args.short_tokens < 1:
            raise SystemExit("--short-tokens must be >= 1")
        if args.long_tokens <= args.short_tokens:
            raise SystemExit("--long-tokens must be greater than --short-tokens")
        if args.batch_size != 8 or args.prompt_tokens != 2048:
            raise SystemExit("this controlled E2B/L4 gate requires batch=8 and prompt=2048")
        measure(args)
    else:
        compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

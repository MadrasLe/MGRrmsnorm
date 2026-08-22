"""
Benchmark MGX Prophet speculative state reuse against baseline prompt prefill.

This harness focuses on the Prophet-specific question:
how much prompt preparation work can be avoided versus rebuilding context
from scratch?

It measures, per query:
  - baseline prefill latency
  - Prophet recovery latency
  - committed recovery route
  - recovery policy statistics
  - estimated reused prompt tokens
  - optional greedy continuation agreement

Examples:
    python benchmarks/benchmark_prophet.py ^
        --model Qwen/Qwen2.5-1.5B-Instruct ^
        --dtype fp16 ^
        --quantize int8 ^
        --device cuda ^
        --export-if-missing ^
        --reset-prophet-dir ^
        --json-out artifacts/benchmark_prophet_qwen15b_int8.json

    python benchmarks/benchmark_prophet.py ^
        --mgx artifacts/Qwen--Qwen2.5-1.5B-Instruct-int8.mgx ^
        --validation-mode none ^
        --runs 5 ^
        --continuation-tokens 8
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from megagemm import export_to_mgx
from megagemm.engine import InferenceEngine, MGXProphetLibrary
from megagemm.models import prime_mgx_payload_cache


def _runtime_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _safe_model_stem(model_ref: str) -> str:
    return model_ref.replace("\\", "--").replace("/", "--").replace(":", "--")


def _default_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    quant_suffix = args.quantize if args.quantize != "none" else args.dtype
    base = _safe_model_stem(args.model)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    mgx_path = Path(args.mgx) if args.mgx else artifacts_dir / f"{base}-{quant_suffix}.mgx"
    prophet_dir = (
        Path(args.prophet_dir)
        if args.prophet_dir
        else artifacts_dir / f"{mgx_path.stem}-prophet-benchmark"
    )
    return mgx_path, prophet_dir


def _cleanup_device(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _reset_cuda_peaks(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_snapshot(device: str) -> dict[str, float]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {}
    return {
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024 ** 2),
        "cuda_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "cuda_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
    }


def _sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _render_prompt_parts(parts: list[Any]) -> str:
    rendered: list[str] = []
    for idx, part in enumerate(parts):
        if isinstance(part, str):
            rendered.append(part)
            continue
        if not isinstance(part, dict):
            raise ValueError(f"Unsupported prompt part type at index {idx}: {type(part).__name__}")

        text = str(part.get("text", ""))
        repeat = int(part.get("repeat", 1) or 1)
        if repeat < 0:
            raise ValueError(f"prompt part repeat must be non-negative at index {idx}")
        rendered.append(text * repeat)
    return "".join(rendered)


def _render_prompt_item(item: dict[str, Any]) -> str:
    prompt = str(item.get("prompt", ""))
    parts = item.get("prompt_parts")
    if parts is not None:
        if not isinstance(parts, list):
            raise ValueError("'prompt_parts' must be a list when provided.")
        prompt += _render_prompt_parts(parts)
    return prompt.strip()


def _coerce_prompt_items(items: list[Any], *, prefix: str) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(items):
        if isinstance(item, str):
            prompt = item.strip()
            label = f"{prefix}-{idx + 1}"
        elif isinstance(item, dict):
            prompt = _render_prompt_item(item)
            label = str(item.get("label") or f"{prefix}-{idx + 1}")
        else:
            raise ValueError(f"Unsupported workload item type: {type(item).__name__}")
        if not prompt:
            continue
        normalized.append({"label": label, "prompt": prompt})
    return normalized


def _load_workload(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    captures: list[dict[str, str]] = []
    queries: list[dict[str, str]] = []

    if args.workload_file:
        payload = json.loads(Path(args.workload_file).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--workload-file must contain a JSON object with 'captures' and/or 'queries'.")
        captures.extend(_coerce_prompt_items(list(payload.get("captures", [])), prefix="capture"))
        queries.extend(_coerce_prompt_items(list(payload.get("queries", [])), prefix="query"))

    if args.capture_prompt:
        captures.extend(
            {"label": f"capture-{idx + 1}", "prompt": prompt.strip()}
            for idx, prompt in enumerate(args.capture_prompt)
            if prompt.strip()
        )

    if args.query:
        queries.extend(
            {"label": f"query-{idx + 1}", "prompt": prompt.strip()}
            for idx, prompt in enumerate(args.query)
            if prompt.strip()
        )

    if not captures:
        captures = [
            {
                "label": "latency-core",
                "prompt": "Explain why compiled model artifacts reduce cold-start latency.",
            },
            {
                "label": "batching-core",
                "prompt": "Summarize the main serving benefits of continuous batching for LLM inference.",
            },
            {
                "label": "kv-core",
                "prompt": "Explain how paged KV cache reduces memory fragmentation during LLM serving.",
            },
        ]

    if not queries:
        anchor = captures[0]["prompt"]
        queries = [
            {"label": "exact", "prompt": anchor},
            {"label": "prefix-extension", "prompt": anchor + " Give one practical deployment example."},
            {"label": "prefix-refocus", "prompt": anchor + " Focus on TTFT and GPU serving."},
            {
                "label": "semantic-nearby",
                "prompt": "Explain how compiled model artifacts help reduce TTFT in GPU serving.",
            },
            {
                "label": "off-domain",
                "prompt": "What makes sourdough bread more acidic than regular bread?",
            },
        ]

    return captures, queries


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _speedup_pct(baseline_seconds: float, candidate_seconds: float) -> float | None:
    if baseline_seconds <= 0:
        return None
    return ((baseline_seconds - candidate_seconds) / baseline_seconds) * 100.0


def _with_continuation_seconds(prepare_seconds: float, continuation: dict[str, Any] | None) -> float:
    continuation_seconds = 0.0
    if continuation is not None:
        continuation_seconds = float(continuation.get("seconds", 0.0) or 0.0)
    return float(prepare_seconds) + continuation_seconds


def _extract_reuse_metrics(result: dict[str, Any], prompt_tokens: int) -> dict[str, Any]:
    committed_source = result.get("committed_source")
    validation = result.get("validation") or {}
    policy = validation.get("policy") or {}

    reused_tokens_estimate = 0
    if committed_source == "prophet_exact":
        reused_tokens_estimate = int(prompt_tokens)
    elif committed_source == "prophet_prefix_reuse":
        reused_tokens_estimate = int(policy.get("common_prefix_tokens", 0) or 0)

    return {
        "reused_tokens_estimate": reused_tokens_estimate,
        "reused_token_ratio_estimate": (
            reused_tokens_estimate / max(1, int(prompt_tokens))
            if prompt_tokens > 0
            else None
        ),
        "policy": policy,
    }


def _compare_token_lists(lhs: list[int], rhs: list[int]) -> dict[str, Any]:
    compare_len = min(len(lhs), len(rhs))
    if compare_len <= 0:
        return {
            "available": False,
            "compare_len": 0,
            "agreement": None,
            "first_token_match": None,
        }

    matched = sum(1 for left, right in zip(lhs[:compare_len], rhs[:compare_len]) if int(left) == int(right))
    return {
        "available": True,
        "compare_len": compare_len,
        "agreement": matched / compare_len,
        "first_token_match": bool(int(lhs[0]) == int(rhs[0])),
    }


def _baseline_prepare(
    engine: InferenceEngine,
    prompt: str,
    *,
    seq_id: int,
    max_new_tokens: int,
    continuation_tokens: int,
) -> dict[str, Any]:
    _reset_cuda_peaks(str(engine.device))
    _sync_device(str(engine.device))
    start = time.perf_counter()
    prefill = engine.prefill_context(prompt, seq_id=seq_id, max_new_tokens=max_new_tokens)
    _sync_device(str(engine.device))
    end = time.perf_counter()

    continuation = None
    if continuation_tokens > 0:
        cont_start = time.perf_counter()
        continuation = engine.generate_from_context(
            seq_id,
            max_new_tokens=continuation_tokens,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        _sync_device(str(engine.device))
        cont_end = time.perf_counter()
        continuation = {
            "seconds": cont_end - cont_start,
            "token_ids": list(continuation.get("token_ids", [])),
            "text": continuation.get("text"),
            "stopped": bool(continuation.get("stopped", False)),
        }

    result = {
        "seconds": end - start,
        "seq_id": seq_id,
        "prompt_tokens": int(prefill["prompt_len"]),
        "seq_len": int(prefill["seq_len"]),
        "continuation": continuation,
        "memory": _cuda_peak_snapshot(str(engine.device)),
    }
    return result


def _can_compare_continuation(result: dict[str, Any], *, validation_mode: str, validation_tokens: int) -> bool:
    source = result.get("committed_source")
    if source in {"prophet_exact", "prophet_prefix_reuse"}:
        return True
    if validation_mode == "none" and source == "prophet_no_validation":
        return True
    if validation_mode == "full_prefill" and validation_tokens <= 0 and source in {"prophet_validated", "prefill_fallback"}:
        return True
    return False


def _prophet_prepare(
    engine: InferenceEngine,
    library_dir: str,
    prompt: str,
    *,
    seq_id: int,
    max_new_tokens: int,
    continuation_tokens: int,
    top_k: int,
    min_similarity: float,
    prefix_tokens: int,
    require_compatible: bool,
    validation_mode: str,
    validation_tokens: int,
    agreement_threshold: float,
    fallback_to_prefill: bool,
    min_prefix_reuse_score: float,
    min_prefix_coverage: float,
    max_prefix_rollback_ratio: float,
    max_prefix_tail_ratio: float,
    use_resident_cache: bool,
    resident_cache_max_entries: int,
) -> dict[str, Any]:
    _reset_cuda_peaks(str(engine.device))
    _sync_device(str(engine.device))
    start = time.perf_counter()
    result = engine.prophet_restore_speculative(
        library_dir,
        prompt,
        seq_id=seq_id,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        min_similarity=min_similarity,
        prefix_tokens=prefix_tokens,
        require_compatible=require_compatible,
        validation_mode=validation_mode,
        validation_tokens=validation_tokens,
        agreement_threshold=agreement_threshold,
        fallback_to_prefill=fallback_to_prefill,
        min_prefix_reuse_score=min_prefix_reuse_score,
        min_prefix_coverage=min_prefix_coverage,
        max_prefix_rollback_ratio=max_prefix_rollback_ratio,
        max_prefix_tail_ratio=max_prefix_tail_ratio,
        use_resident_cache=use_resident_cache,
        resident_cache_max_entries=resident_cache_max_entries,
    )
    _sync_device(str(engine.device))
    end = time.perf_counter()

    continuation = None
    if (
        continuation_tokens > 0
        and result.get("restored")
        and _can_compare_continuation(
            result,
            validation_mode=validation_mode,
            validation_tokens=validation_tokens,
        )
    ):
        cont_start = time.perf_counter()
        continuation_raw = engine.generate_from_context(
            result["seq_id"],
            max_new_tokens=continuation_tokens,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        _sync_device(str(engine.device))
        cont_end = time.perf_counter()
        continuation = {
            "seconds": cont_end - cont_start,
            "token_ids": list(continuation_raw.get("token_ids", [])),
            "text": continuation_raw.get("text"),
            "stopped": bool(continuation_raw.get("stopped", False)),
        }

    return {
        "seconds": end - start,
        "result": result,
        "continuation": continuation,
        "memory": _cuda_peak_snapshot(str(engine.device)),
    }


def _summarize_query_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_seconds = [sample["baseline"]["seconds"] for sample in samples]
    prophet_seconds = [sample["prophet"]["seconds"] for sample in samples]
    total_baseline_seconds = [sample["delta"]["total_baseline_seconds"] for sample in samples]
    total_prophet_seconds = [sample["delta"]["total_prophet_seconds"] for sample in samples]
    speedups = [
        sample["delta"]["speedup_pct"]
        for sample in samples
        if sample["delta"]["speedup_pct"] is not None
    ]
    total_speedups = [
        sample["delta"]["total_speedup_pct"]
        for sample in samples
        if sample["delta"]["total_speedup_pct"] is not None
    ]
    reused_ratios = [
        sample["prophet"]["reused_token_ratio_estimate"]
        for sample in samples
        if sample["prophet"]["reused_token_ratio_estimate"] is not None
    ]
    continuation_agreements = [
        sample["continuation"]["agreement"]
        for sample in samples
        if sample["continuation"].get("agreement") is not None
    ]

    committed_source_counts: dict[str, int] = {}
    for sample in samples:
        source = str(sample["prophet"]["committed_source"])
        committed_source_counts[source] = committed_source_counts.get(source, 0) + 1

    return {
        "runs": len(samples),
        "avg_baseline_seconds": _mean(baseline_seconds),
        "avg_prophet_seconds": _mean(prophet_seconds),
        "avg_speedup_pct": _mean(speedups),
        "avg_total_baseline_seconds": _mean(total_baseline_seconds),
        "avg_total_prophet_seconds": _mean(total_prophet_seconds),
        "avg_total_speedup_pct": _mean(total_speedups),
        "avg_reused_token_ratio_estimate": _mean(reused_ratios),
        "avg_continuation_agreement": _mean(continuation_agreements),
        "committed_source_counts": committed_source_counts,
        "accepted_runs": sum(1 for sample in samples if sample["prophet"]["speculative_accepted"]),
        "restored_runs": sum(1 for sample in samples if sample["prophet"]["restored"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MGX Prophet speculative state reuse.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF model id or local snapshot directory.")
    parser.add_argument("--mgx", help="Compiled MGX artifact path.")
    parser.add_argument("--prophet-dir", help="Directory used to store the Prophet library.")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--quantize", choices=("none", "int8", "int4"), default="none")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workload-file", help="Optional JSON file with {'captures': [...], 'queries': [...]} entries.")
    parser.add_argument("--capture-prompt", action="append", default=[], help="Prompt to capture into Prophet. May be passed multiple times.")
    parser.add_argument("--query", action="append", default=[], help="Prompt to benchmark. May be passed multiple times.")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    parser.add_argument("--prefix-tokens", type=int, default=16)
    parser.add_argument("--validation-mode", choices=("none", "full_prefill"), default="full_prefill")
    parser.add_argument("--validation-tokens", type=int, default=4)
    parser.add_argument("--agreement-threshold", type=float, default=1.0)
    parser.add_argument("--fallback-to-prefill", action="store_true", help="Keep the validator baseline when speculative validation fails.")
    parser.add_argument("--continuation-tokens", type=int, default=0, help="Optional greedy continuation tokens to compare after preparation.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--min-prefix-reuse-score", type=float, default=0.55)
    parser.add_argument("--min-prefix-coverage", type=float, default=0.50)
    parser.add_argument("--max-prefix-rollback-ratio", type=float, default=0.35)
    parser.add_argument("--max-prefix-tail-ratio", type=float, default=0.50)
    parser.add_argument(
        "--prophet-resident-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep restored Prophet snapshots live and fork future exact/prefix hits.",
    )
    parser.add_argument(
        "--prophet-resident-cache-max-entries",
        type=int,
        default=16,
        help="Maximum live Prophet resident source sequences per engine.",
    )
    parser.add_argument("--skip-hash-check", action="store_true", help="Skip MGX payload hash verification during load.")
    parser.add_argument(
        "--mgx-prefer-payload-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer a reusable extracted safetensors payload cache when loading the MGX artifact.",
    )
    parser.add_argument("--mgx-payload-cache-dir", help="Optional directory for reusable MGX payload cache files.")
    parser.add_argument(
        "--mgx-emit-payload-cache",
        action="store_true",
        help="Emit or prime the reusable MGX payload cache before running Prophet.",
    )
    parser.add_argument(
        "--mgx-export-mode",
        choices=("normal", "streaming"),
        default="streaming",
        help="MGX export implementation used when --export-if-missing or --force-export is active.",
    )
    parser.add_argument("--export-if-missing", action="store_true", help="Export MGX if the artifact does not exist yet.")
    parser.add_argument("--force-export", action="store_true", help="Always rebuild the MGX artifact before running the benchmark.")
    parser.add_argument("--reset-prophet-dir", action="store_true", help="Delete the Prophet directory before capturing the workload.")
    parser.add_argument("--json-out", help="Optional JSON output path.")
    args = parser.parse_args()

    mgx_path, prophet_dir = _default_paths(args)
    dtype = _runtime_dtype(args.dtype)
    quantize = None if args.quantize == "none" else args.quantize
    captures, queries = _load_workload(args)

    print()
    print("=" * 80)
    print("MGX PROPHET BENCHMARK")
    print("=" * 80)
    print(f"Model:            {args.model}")
    print(f"MGX:              {mgx_path}")
    print(f"Prophet dir:      {prophet_dir}")
    print(f"Device:           {args.device}")
    print(f"DType:            {args.dtype}")
    print(f"Quantize:         {args.quantize}")
    print(f"Validation mode:  {args.validation_mode}")
    print(f"Runs/Warmup:      {args.runs}/{args.warmup}")
    print(f"Continuation tok: {args.continuation_tokens}")
    print(f"MGX payload cache:{args.mgx_prefer_payload_cache}")

    if args.force_export or not mgx_path.exists():
        if not args.export_if_missing and not args.force_export:
            raise SystemExit(
                f"MGX artifact {mgx_path} does not exist. Pass --export-if-missing or provide --mgx."
            )
        print()
        print("=" * 80)
        print("EXPORTING MGX")
        print("=" * 80)
        export_to_mgx(
            args.model,
            mgx_path,
            dtype=args.dtype,
            quantize=args.quantize,
            emit_payload_cache=args.mgx_emit_payload_cache,
            payload_cache_dir=args.mgx_payload_cache_dir,
            export_mode=args.mgx_export_mode,
        )
    elif args.mgx_emit_payload_cache:
        print()
        print("=" * 80)
        print("PRIMING MGX PAYLOAD CACHE")
        print("=" * 80)
        prime_mgx_payload_cache(
            mgx_path,
            validate_payload_hash=not args.skip_hash_check,
            payload_cache_dir=args.mgx_payload_cache_dir,
        )

    if args.reset_prophet_dir and prophet_dir.exists():
        shutil.rmtree(prophet_dir)

    _cleanup_device(args.device)
    engine = InferenceEngine(
        str(mgx_path),
        dtype=dtype,
        device=args.device,
        quantize=quantize,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        mgx_verify_payload=not args.skip_hash_check,
        mgx_prefer_payload_cache=args.mgx_prefer_payload_cache,
        mgx_payload_cache_dir=args.mgx_payload_cache_dir,
    )

    print()
    print("=" * 80)
    print("CAPTURING WORKLOAD")
    print("=" * 80)
    capture_reports: list[dict[str, Any]] = []
    for idx, item in enumerate(captures):
        seq_id = 1000 + idx
        prefill = engine.prefill_context(
            item["prompt"],
            seq_id=seq_id,
            max_new_tokens=args.max_new_tokens,
        )
        entry = engine.prophet_capture(
            str(prophet_dir),
            seq_id,
            text=item["prompt"],
            label=item["label"],
            metadata={"benchmark": "prophet", "capture_index": idx},
        )
        engine.free_sequence(seq_id)
        report = {
            "label": item["label"],
            "prompt": item["prompt"],
            "prompt_tokens": int(prefill["prompt_len"]),
            "seq_len": int(prefill["seq_len"]),
            "entry_id": entry["entry_id"],
        }
        capture_reports.append(report)
        print(json.dumps(report, indent=2, ensure_ascii=False))

    library_stats = MGXProphetLibrary(prophet_dir).stats()

    print()
    print("=" * 80)
    print("RUNNING QUERIES")
    print("=" * 80)
    query_reports: list[dict[str, Any]] = []
    flat_results: list[dict[str, Any]] = []

    seq_counter = 5000
    for query in queries:
        print()
        print("-" * 80)
        print(f"{query['label']}: {query['prompt']}")
        print("-" * 80)

        for _ in range(max(0, args.warmup)):
            baseline_seq_id = seq_counter
            seq_counter += 1
            baseline_warmup = _baseline_prepare(
                engine,
                query["prompt"],
                seq_id=baseline_seq_id,
                max_new_tokens=args.max_new_tokens,
                continuation_tokens=0,
            )
            if baseline_warmup["seq_id"] in engine.block_manager.block_tables:
                engine.free_sequence(baseline_warmup["seq_id"])

            prophet_seq_id = seq_counter
            seq_counter += 1
            prophet_warmup = _prophet_prepare(
                engine,
                str(prophet_dir),
                query["prompt"],
                seq_id=prophet_seq_id,
                max_new_tokens=args.max_new_tokens,
                continuation_tokens=0,
                top_k=args.top_k,
                min_similarity=args.min_similarity,
                prefix_tokens=args.prefix_tokens,
                require_compatible=True,
                validation_mode=args.validation_mode,
                validation_tokens=args.validation_tokens,
                agreement_threshold=args.agreement_threshold,
                fallback_to_prefill=args.fallback_to_prefill,
                min_prefix_reuse_score=args.min_prefix_reuse_score,
                min_prefix_coverage=args.min_prefix_coverage,
                max_prefix_rollback_ratio=args.max_prefix_rollback_ratio,
                max_prefix_tail_ratio=args.max_prefix_tail_ratio,
                use_resident_cache=args.prophet_resident_cache,
                resident_cache_max_entries=args.prophet_resident_cache_max_entries,
            )
            warm_seq_id = prophet_warmup["result"].get("seq_id")
            if warm_seq_id is not None and warm_seq_id in engine.block_manager.block_tables:
                engine.free_sequence(warm_seq_id)

        samples: list[dict[str, Any]] = []
        for run_idx in range(max(1, args.runs)):
            baseline_seq_id = seq_counter
            seq_counter += 1
            baseline = _baseline_prepare(
                engine,
                query["prompt"],
                seq_id=baseline_seq_id,
                max_new_tokens=args.max_new_tokens,
                continuation_tokens=args.continuation_tokens,
            )
            if baseline["seq_id"] in engine.block_manager.block_tables:
                engine.free_sequence(baseline["seq_id"])

            prophet_seq_id = seq_counter
            seq_counter += 1
            prophet = _prophet_prepare(
                engine,
                str(prophet_dir),
                query["prompt"],
                seq_id=prophet_seq_id,
                max_new_tokens=args.max_new_tokens,
                continuation_tokens=args.continuation_tokens,
                top_k=args.top_k,
                min_similarity=args.min_similarity,
                prefix_tokens=args.prefix_tokens,
                require_compatible=True,
                validation_mode=args.validation_mode,
                validation_tokens=args.validation_tokens,
                agreement_threshold=args.agreement_threshold,
                fallback_to_prefill=args.fallback_to_prefill,
                min_prefix_reuse_score=args.min_prefix_reuse_score,
                min_prefix_coverage=args.min_prefix_coverage,
                max_prefix_rollback_ratio=args.max_prefix_rollback_ratio,
                max_prefix_tail_ratio=args.max_prefix_tail_ratio,
                use_resident_cache=args.prophet_resident_cache,
                resident_cache_max_entries=args.prophet_resident_cache_max_entries,
            )

            prophet_result = prophet["result"]
            reuse_metrics = _extract_reuse_metrics(prophet_result, baseline["prompt_tokens"])

            continuation_report = {
                "available": False,
                "agreement": None,
                "first_token_match": None,
            }
            baseline_cont = baseline.get("continuation")
            prophet_cont = prophet.get("continuation")
            if baseline_cont is not None and prophet_cont is not None:
                continuation_report = _compare_token_lists(
                    list(baseline_cont.get("token_ids", [])),
                    list(prophet_cont.get("token_ids", [])),
                )

            baseline_total_seconds = _with_continuation_seconds(baseline["seconds"], baseline_cont)
            prophet_total_seconds = _with_continuation_seconds(prophet["seconds"], prophet_cont)
            sample = {
                "run_index": run_idx,
                "baseline": baseline,
                "prophet": {
                    "seconds": prophet["seconds"],
                    "restored": bool(prophet_result.get("restored", False)),
                    "speculative_accepted": bool(prophet_result.get("speculative_accepted", False)),
                    "committed_source": prophet_result.get("committed_source"),
                    "reason": prophet_result.get("reason"),
                    "match": prophet_result.get("match"),
                    "decision_trace": prophet_result.get("decision_trace"),
                    "validation": prophet_result.get("validation"),
                    "continuation": prophet_cont,
                    "memory": prophet["memory"],
                    **reuse_metrics,
                },
                "delta": {
                    "seconds_saved": baseline["seconds"] - prophet["seconds"],
                    "speedup_pct": _speedup_pct(baseline["seconds"], prophet["seconds"]),
                    "total_baseline_seconds": baseline_total_seconds,
                    "total_prophet_seconds": prophet_total_seconds,
                    "total_seconds_saved": baseline_total_seconds - prophet_total_seconds,
                    "total_speedup_pct": _speedup_pct(baseline_total_seconds, prophet_total_seconds),
                },
                "continuation": continuation_report,
            }
            samples.append(sample)
            flat_results.append(
                {
                    "query_label": query["label"],
                    "run_index": run_idx,
                    "baseline_seconds": baseline["seconds"],
                    "prophet_seconds": prophet["seconds"],
                    "seconds_saved": sample["delta"]["seconds_saved"],
                    "speedup_pct": sample["delta"]["speedup_pct"],
                    "total_baseline_seconds": sample["delta"]["total_baseline_seconds"],
                    "total_prophet_seconds": sample["delta"]["total_prophet_seconds"],
                    "total_seconds_saved": sample["delta"]["total_seconds_saved"],
                    "total_speedup_pct": sample["delta"]["total_speedup_pct"],
                    "committed_source": sample["prophet"]["committed_source"],
                    "restored": sample["prophet"]["restored"],
                    "speculative_accepted": sample["prophet"]["speculative_accepted"],
                    "reason": sample["prophet"]["reason"],
                    "reused_tokens_estimate": sample["prophet"]["reused_tokens_estimate"],
                    "reused_token_ratio_estimate": sample["prophet"]["reused_token_ratio_estimate"],
                    "continuation_agreement": sample["continuation"]["agreement"],
                }
            )

            print(
                json.dumps(
                    {
                        "run_index": run_idx,
                        "baseline_seconds": baseline["seconds"],
                        "prophet_seconds": prophet["seconds"],
                        "total_baseline_seconds": sample["delta"]["total_baseline_seconds"],
                        "total_prophet_seconds": sample["delta"]["total_prophet_seconds"],
                        "committed_source": sample["prophet"]["committed_source"],
                        "speculative_accepted": sample["prophet"]["speculative_accepted"],
                        "speedup_pct": sample["delta"]["speedup_pct"],
                        "total_speedup_pct": sample["delta"]["total_speedup_pct"],
                        "reused_token_ratio_estimate": sample["prophet"]["reused_token_ratio_estimate"],
                        "continuation_agreement": sample["continuation"]["agreement"],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )

            prophet_final_seq_id = prophet_result.get("seq_id")
            if prophet_final_seq_id is not None and prophet_final_seq_id in engine.block_manager.block_tables:
                engine.free_sequence(prophet_final_seq_id)

        query_reports.append(
            {
                "label": query["label"],
                "prompt": query["prompt"],
                "samples": samples,
                "summary": _summarize_query_samples(samples),
            }
        )

    summary = {
        "config": {
            "model": args.model,
            "mgx": str(mgx_path),
            "prophet_dir": str(prophet_dir),
            "device": args.device,
            "dtype": args.dtype,
            "quantize": args.quantize,
            "max_seq_len": args.max_seq_len,
            "max_new_tokens": args.max_new_tokens,
            "max_batch_size": args.max_batch_size,
            "top_k": args.top_k,
            "min_similarity": args.min_similarity,
            "prefix_tokens": args.prefix_tokens,
            "validation_mode": args.validation_mode,
            "validation_tokens": args.validation_tokens,
            "agreement_threshold": args.agreement_threshold,
            "fallback_to_prefill": args.fallback_to_prefill,
            "continuation_tokens": args.continuation_tokens,
            "runs": args.runs,
            "warmup": args.warmup,
            "min_prefix_reuse_score": args.min_prefix_reuse_score,
            "min_prefix_coverage": args.min_prefix_coverage,
            "max_prefix_rollback_ratio": args.max_prefix_rollback_ratio,
            "max_prefix_tail_ratio": args.max_prefix_tail_ratio,
            "prophet_resident_cache": args.prophet_resident_cache,
            "prophet_resident_cache_max_entries": args.prophet_resident_cache_max_entries,
            "skip_hash_check": args.skip_hash_check,
            "mgx_prefer_payload_cache": args.mgx_prefer_payload_cache,
            "mgx_payload_cache_dir": args.mgx_payload_cache_dir,
            "mgx_emit_payload_cache": args.mgx_emit_payload_cache,
            "mgx_export_mode": args.mgx_export_mode,
        },
        "engine_init_timing": engine.get_init_timing(),
        "library_stats": library_stats,
        "resident_cache_stats": MGXProphetLibrary.resident_cache_stats(engine),
        "captures": capture_reports,
        "queries": query_reports,
        "flat_results": flat_results,
    }

    route_totals: dict[str, dict[str, float]] = {}
    for row in flat_results:
        source = str(row["committed_source"])
        bucket = route_totals.setdefault(
            source,
            {
                "count": 0.0,
                "baseline_seconds_sum": 0.0,
                "prophet_seconds_sum": 0.0,
                "speedup_pct_sum": 0.0,
                "speedup_pct_count": 0.0,
            },
        )
        bucket["count"] += 1.0
        bucket["baseline_seconds_sum"] += float(row["baseline_seconds"])
        bucket["prophet_seconds_sum"] += float(row["prophet_seconds"])
        if row["speedup_pct"] is not None:
            bucket["speedup_pct_sum"] += float(row["speedup_pct"])
            bucket["speedup_pct_count"] += 1.0

    summary["route_summary"] = {
        source: {
            "count": int(values["count"]),
            "avg_baseline_seconds": values["baseline_seconds_sum"] / max(values["count"], 1.0),
            "avg_prophet_seconds": values["prophet_seconds_sum"] / max(values["count"], 1.0),
            "avg_speedup_pct": (
                values["speedup_pct_sum"] / values["speedup_pct_count"]
                if values["speedup_pct_count"] > 0
                else None
            ),
        }
        for source, values in route_totals.items()
    }

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON summary to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
MGX Prophet threshold sweep for Colab/Kaggle.

This script captures one reference runtime state into MGX Prophet and then
evaluates multiple query prompts across several recovery thresholds.

It helps answer questions like:
- At which semantic threshold does Prophet still recover this family of prompts?
- Is a match happening because of exact text, prefix hash, or semantic similarity?
- Does a successful restore preserve the captured KV/session state correctly?

Example:

    python examples/mgx_prophet_threshold_sweep.py ^
        --model Qwen/Qwen2.5-1.5B-Instruct ^
        --dtype fp16 ^
        --quantize int8 ^
        --device cuda ^
        --reset-prophet-dir ^
        --json-out artifacts/mgx_prophet_threshold_sweep.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from megagemm import export_to_mgx, inspect_mgx
from megagemm.engine import InferenceEngine, MGXProphetLibrary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep MGX Prophet recovery thresholds against a captured reference state."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face model id or local snapshot directory.",
    )
    parser.add_argument(
        "--mgx",
        help="Output MGX artifact path. Defaults to artifacts/<model>-<dtype-or-quant>.mgx",
    )
    parser.add_argument(
        "--prophet-dir",
        help="Directory used to store the Prophet library.",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument(
        "--quantize",
        choices=("none", "int8", "int4"),
        default="none",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--capture-prompt",
        default="Explain why compiled model artifacts reduce cold-start latency.",
        help="Prompt used to build and capture the reference state.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Query prompt to test. Can be passed multiple times.",
    )
    parser.add_argument(
        "--queries-file",
        help="Optional .txt or .json file with query prompts. TXT uses one prompt per line; JSON expects a list of strings.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.00,0.20,0.35,0.50,0.70,0.85,0.95",
        help="Comma-separated semantic thresholds to test.",
    )
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=16,
        help="Prefix hashing window used by Prophet lookup/restore.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many Prophet matches to retain for each lookup.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip MGX payload hash verification during inspect/load to speed up experiments.",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Re-export the MGX artifact even if it already exists.",
    )
    parser.add_argument(
        "--reset-prophet-dir",
        action="store_true",
        help="Delete and recreate the Prophet library directory before capturing the reference state.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to save the final JSON report.",
    )
    return parser.parse_args()


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
        else artifacts_dir / f"{mgx_path.stem}-prophet-threshold-sweep"
    )
    return mgx_path, prophet_dir


def _parse_thresholds(raw: str) -> list[float]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("At least one threshold must be provided.")
    return sorted(dict.fromkeys(values))


def _load_queries(args: argparse.Namespace) -> list[dict[str, str]]:
    queries: list[str] = []
    if args.queries_file:
        path = Path(args.queries_file)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"{path} must contain a JSON list of prompt strings.")
            for item in payload:
                if not isinstance(item, str):
                    raise ValueError(f"{path} contains a non-string query entry.")
                item = item.strip()
                if item:
                    queries.append(item)
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    queries.append(line)
    for item in args.query:
        item = item.strip()
        if item:
            queries.append(item)

    if not queries:
        queries = [
            args.capture_prompt,
            args.capture_prompt + " Focus on TTFT, first-token latency, and serving startup.",
            "Summarize the serving implication of this request: " + args.capture_prompt,
            "Describe how sourdough fermentation changes bread texture and acidity.",
        ]

    labeled: list[dict[str, str]] = []
    for idx, prompt in enumerate(queries):
        if prompt == args.capture_prompt:
            label = "exact"
        elif idx == 1:
            label = "nearby-variant"
        elif idx == 2:
            label = "paraphrase"
        elif idx == 3:
            label = "off-domain"
        else:
            label = f"query-{idx + 1}"
        labeled.append({"label": label, "prompt": prompt})
    return labeled


def _prepare_prompt(engine: InferenceEngine, prompt: str) -> tuple[str, torch.Tensor]:
    formatted_prompt = prompt
    bos = engine.tokenizer.bos_token
    already_formatted = bool(bos and prompt.startswith(bos))

    if (
        not already_formatted
        and hasattr(engine.tokenizer, "chat_template")
        and engine.tokenizer.chat_template
    ):
        try:
            formatted_prompt = engine.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted_prompt = prompt

    add_special = not (bos and formatted_prompt.startswith(bos))
    input_ids = engine.tokenizer.encode(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=add_special,
    ).to(engine.device)
    return formatted_prompt, input_ids


def _prefill_prompt(
    engine: InferenceEngine,
    prompt: str,
    *,
    seq_id: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    prefill_info = engine.prefill_context(
        prompt,
        seq_id=seq_id,
        max_new_tokens=max_new_tokens,
    )
    snapshot = engine.save_context(seq_id, text=prompt)
    return {
        "formatted_prompt": prefill_info["formatted_prompt"],
        "prompt_len": prefill_info["prompt_len"],
        "snapshot": snapshot,
    }


def _snapshot_equivalence(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    kv_a = a.get("kv_data_by_layer", {})
    kv_b = b.get("kv_data_by_layer", {})
    shared_layers = sorted(set(kv_a.keys()) & set(kv_b.keys()))
    first_layer = shared_layers[0] if shared_layers else None
    first_layer_equal = None
    if first_layer is not None:
        first_layer_equal = bool(torch.equal(kv_a[first_layer], kv_b[first_layer]))

    return {
        "seq_len_equal": a.get("seq_len") == b.get("seq_len"),
        "source_model_hash_equal": a.get("source_model_hash") == b.get("source_model_hash"),
        "tokenizer_hash_equal": a.get("tokenizer_hash") == b.get("tokenizer_hash"),
        "chat_template_hash_equal": a.get("chat_template_hash") == b.get("chat_template_hash"),
        "first_shared_kv_layer": first_layer,
        "first_shared_kv_layer_equal": first_layer_equal,
    }


def _match_route(match: dict[str, Any] | None) -> str | None:
    if not match:
        return None
    if match.get("exact_text_match"):
        return "exact_text"
    if match.get("prefix_match"):
        return "prefix_hash"
    if match.get("semantic_similarity") is not None:
        return "semantic_similarity"
    return "unknown"


def _print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _summarize_match(match: dict[str, Any] | None) -> dict[str, Any] | None:
    if not match:
        return None
    return {
        "entry_id": match.get("entry_id"),
        "label": match.get("label"),
        "seq_len": match.get("seq_len"),
        "exact_text_match": match.get("exact_text_match"),
        "prefix_match": match.get("prefix_match"),
        "semantic_similarity": match.get("semantic_similarity"),
        "score": match.get("score"),
        "route": _match_route(match),
    }


def main() -> int:
    args = _parse_args()
    mgx_path, prophet_dir = _default_paths(args)
    thresholds = _parse_thresholds(args.thresholds)
    queries = _load_queries(args)
    dtype = _runtime_dtype(args.dtype)
    quantize = None if args.quantize == "none" else args.quantize

    _print_header("MGX PROPHET THRESHOLD SWEEP")
    print(f"Model:          {args.model}")
    print(f"MGX:            {mgx_path}")
    print(f"Prophet dir:    {prophet_dir}")
    print(f"Device:         {args.device}")
    print(f"DType:          {args.dtype}")
    print(f"Quantize:       {args.quantize}")
    print(f"Capture prompt: {args.capture_prompt}")
    print(f"Thresholds:     {', '.join(f'{value:.2f}' for value in thresholds)}")
    print("Prophet note: exact text and prefix-hash matches bypass the semantic threshold in v0.")

    export_started = time.perf_counter()
    if args.force_export or not mgx_path.exists():
        _print_header("EXPORTING MGX")
        export_info = export_to_mgx(
            args.model,
            mgx_path,
            dtype=args.dtype,
            quantize=args.quantize,
        )
        header = export_info.get("header", {})
        payload_bytes = (
            header.get("tensor_size")
            or header.get("payload_length")
            or export_info.get("payload_cache", {}).get("payload_cache_bytes")
        )
        print(json.dumps(
            {
                "artifact": str(mgx_path),
                "payload_bytes": payload_bytes,
                "quantization": export_info["manifest"]["quantization"],
                "architecture": export_info["manifest"]["architecture"],
            },
            indent=2,
        ))
    export_seconds = time.perf_counter() - export_started

    artifact_info = inspect_mgx(
        mgx_path,
        validate_payload_hash=not args.skip_hash_check,
    )
    _print_header("INSPECTING MGX")
    print(json.dumps(
        {
            "session_state_present": artifact_info["session_state_present"],
            "quantization": artifact_info["manifest"]["quantization"],
            "tokenizer_hash": artifact_info["manifest"]["tokenizer_hash"],
            "source_model_hash": artifact_info["manifest"]["source_model_hash"],
            "payload_cache_exists": artifact_info["payload_cache_exists"],
        },
        indent=2,
    ))

    if args.reset_prophet_dir and prophet_dir.exists():
        shutil.rmtree(prophet_dir)

    engine = InferenceEngine(
        str(mgx_path),
        dtype=dtype,
        device=args.device,
        quantize=quantize,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        mgx_verify_payload=not args.skip_hash_check,
        mgx_prefer_payload_cache=True,
    )

    library = MGXProphetLibrary(prophet_dir)
    existing_entries_before = library.stats()["entries"]

    _print_header("CAPTURING REFERENCE STATE")
    capture_seq_id = 9001
    capture = _prefill_prompt(
        engine,
        args.capture_prompt,
        seq_id=capture_seq_id,
        max_new_tokens=args.max_new_tokens,
    )
    reference_snapshot = capture["snapshot"]
    captured_entry = engine.prophet_capture(
        str(prophet_dir),
        capture_seq_id,
        text=args.capture_prompt,
        label="threshold-sweep-reference",
        metadata={
            "artifact": str(mgx_path),
            "capture_prompt": args.capture_prompt,
        },
    )
    engine.block_manager.free_sequence(capture_seq_id)
    print(json.dumps(
        {
            "captured_entry_id": captured_entry["entry_id"],
            "prompt_tokens": capture["prompt_len"],
            "snapshot_seq_len": reference_snapshot["seq_len"],
            "existing_entries_before": existing_entries_before,
            "entries_after_capture": MGXProphetLibrary(prophet_dir).stats()["entries"],
        },
        indent=2,
    ))

    reports: list[dict[str, Any]] = []
    flat_results: list[dict[str, Any]] = []

    for query_index, query in enumerate(queries):
        _print_header(f"QUERY {query_index + 1}: {query['label']}")
        print(query["prompt"])
        query_report = {
            "label": query["label"],
            "prompt": query["prompt"],
            "thresholds": [],
        }

        for threshold_index, threshold in enumerate(thresholds):
            lookup_started = time.perf_counter()
            matches = engine.prophet_lookup(
                str(prophet_dir),
                query["prompt"],
                top_k=args.top_k,
                min_similarity=threshold,
                prefix_tokens=args.prefix_tokens,
                require_compatible=True,
            )
            lookup_seconds = time.perf_counter() - lookup_started

            best = matches[0] if matches else None
            semantic_similarity = best.get("semantic_similarity") if best else None
            semantic_gate_passed = (
                semantic_similarity is not None and float(semantic_similarity) >= float(threshold)
            )

            restore_started = time.perf_counter()
            restore = engine.prophet_restore_best(
                str(prophet_dir),
                query["prompt"],
                seq_id=10000 + (query_index * 100) + threshold_index,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                min_similarity=threshold,
                prefix_tokens=args.prefix_tokens,
                require_compatible=True,
            )
            restore_seconds = time.perf_counter() - restore_started

            equivalence = None
            if restore.get("restored"):
                restored_snapshot = engine.save_context(restore["seq_id"], text=query["prompt"])
                equivalence = _snapshot_equivalence(reference_snapshot, restored_snapshot)
                engine.block_manager.free_sequence(restore["seq_id"])

            result = {
                "threshold": threshold,
                "matched": bool(matches),
                "restored": bool(restore.get("restored")),
                "best_match": _summarize_match(best),
                "semantic_gate_passed": semantic_gate_passed,
                "lookup_seconds": lookup_seconds,
                "restore_seconds": restore_seconds,
                "equivalence": equivalence,
            }
            query_report["thresholds"].append(result)
            flat_results.append(
                {
                    "query_label": query["label"],
                    "threshold": threshold,
                    "matched": result["matched"],
                    "restored": result["restored"],
                    "route": (result["best_match"] or {}).get("route"),
                    "semantic_similarity": (result["best_match"] or {}).get("semantic_similarity"),
                    "score": (result["best_match"] or {}).get("score"),
                    "lookup_seconds": lookup_seconds,
                    "restore_seconds": restore_seconds,
                    "integrity_ok": bool(
                        equivalence
                        and equivalence.get("seq_len_equal")
                        and equivalence.get("source_model_hash_equal")
                        and equivalence.get("tokenizer_hash_equal")
                        and equivalence.get("chat_template_hash_equal")
                        and equivalence.get("first_shared_kv_layer_equal")
                    ),
                }
            )

            summary = result["best_match"] or {}
            print(
                json.dumps(
                    {
                        "threshold": threshold,
                        "matched": result["matched"],
                        "restored": result["restored"],
                        "route": summary.get("route"),
                        "semantic_similarity": summary.get("semantic_similarity"),
                        "score": summary.get("score"),
                        "lookup_seconds": lookup_seconds,
                        "restore_seconds": restore_seconds,
                        "integrity_ok": bool(
                            equivalence
                            and equivalence.get("seq_len_equal")
                            and equivalence.get("source_model_hash_equal")
                            and equivalence.get("tokenizer_hash_equal")
                            and equivalence.get("chat_template_hash_equal")
                            and equivalence.get("first_shared_kv_layer_equal")
                        ),
                    },
                    indent=2,
                )
            )

        matched_thresholds = [
            item["threshold"]
            for item in query_report["thresholds"]
            if item["matched"]
        ]
        restored_thresholds = [
            item["threshold"]
            for item in query_report["thresholds"]
            if item["restored"]
        ]
        query_report["summary"] = {
            "matched_thresholds": matched_thresholds,
            "restored_thresholds": restored_thresholds,
            "highest_matched_threshold": max(matched_thresholds) if matched_thresholds else None,
            "highest_restored_threshold": max(restored_thresholds) if restored_thresholds else None,
        }
        reports.append(query_report)

    summary = {
        "config": {
            "model": args.model,
            "mgx": str(mgx_path),
            "prophet_dir": str(prophet_dir),
            "device": args.device,
            "dtype": args.dtype,
            "quantize": args.quantize,
            "capture_prompt": args.capture_prompt,
            "thresholds": thresholds,
            "prefix_tokens": args.prefix_tokens,
            "top_k": args.top_k,
            "skip_hash_check": args.skip_hash_check,
            "queries": queries,
        },
        "timing": {
            "export_seconds": export_seconds,
            "engine_init_seconds": engine.get_init_timing().get("total_seconds"),
        },
        "artifact": {
            "session_state_present": artifact_info["session_state_present"],
            "quantization": artifact_info["manifest"]["quantization"],
            "architecture": artifact_info["manifest"]["architecture"],
            "payload_cache_exists": artifact_info["payload_cache_exists"],
        },
        "capture": {
            "entry_id": captured_entry["entry_id"],
            "prompt_tokens": capture["prompt_len"],
            "seq_len": reference_snapshot["seq_len"],
            "entries_before": existing_entries_before,
            "entries_after": MGXProphetLibrary(prophet_dir).stats()["entries"],
        },
        "reports": reports,
        "flat_results": flat_results,
    }

    _print_header("FINAL SUMMARY")
    print(json.dumps(summary, indent=2))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved JSON summary to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

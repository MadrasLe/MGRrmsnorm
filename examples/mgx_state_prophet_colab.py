"""
Colab-friendly MGX state and Prophet demo.

This script exercises three end-to-end flows:

1. Export a Hugging Face model or local snapshot into `.mgx`
2. Prefill a live sequence and embed its runtime state back into `.mgx`
3. Capture, lookup, and restore the same sequence through MGX Prophet

Example:

    python examples/mgx_state_prophet_colab.py ^
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ^
        --dtype fp16 ^
        --quantize none ^
        --prompt "Explain why compiled model artifacts reduce cold-start latency."
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch

from megagemm import export_to_mgx, inspect_mgx
from megagemm.engine import InferenceEngine


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an MGX state + Prophet smoke test that is easy to use from Colab."
    )
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model id or local snapshot directory.",
    )
    parser.add_argument(
        "--mgx",
        help="Output MGX artifact path. Defaults to artifacts/<model>-<dtype-or-quant>.mgx",
    )
    parser.add_argument(
        "--stateful-mgx",
        help="Path for the MGX artifact with embedded session state.",
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
        "--prompt",
        default="Explain why compiled model artifacts reduce cold-start latency.",
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
        "--min-similarity",
        type=float,
        default=0.35,
        help="Semantic recovery threshold used by Prophet lookup/restore.",
    )
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=16,
        help="Prefix hashing window used by Prophet lookup/restore.",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip MGX payload hash verification during inspect/load to speed up experiments.",
    )
    parser.add_argument(
        "--min-prefix-reuse-score",
        type=float,
        default=0.55,
        help="Minimum recovery-policy score required before Prophet may accept token-prefix replay.",
    )
    parser.add_argument(
        "--min-prefix-coverage",
        type=float,
        default=0.50,
        help="Minimum fraction of query tokens that must already exist in the recovered prefix.",
    )
    parser.add_argument(
        "--max-prefix-rollback-ratio",
        type=float,
        default=0.35,
        help="Maximum tolerated rollback ratio relative to the recovered candidate sequence.",
    )
    parser.add_argument(
        "--max-prefix-tail-ratio",
        type=float,
        default=0.50,
        help="Maximum tolerated replay-tail ratio relative to the query token count.",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Re-export the MGX artifact even if it already exists.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to save the final JSON summary.",
    )
    return parser.parse_args()


def _runtime_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping[name]


def _safe_model_stem(model_ref: str) -> str:
    return model_ref.replace("\\", "--").replace("/", "--").replace(":", "--")


def _default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    quant_suffix = args.quantize if args.quantize != "none" else args.dtype
    base = _safe_model_stem(args.model)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    mgx_path = Path(args.mgx) if args.mgx else artifacts_dir / f"{base}-{quant_suffix}.mgx"
    stateful_path = (
        Path(args.stateful_mgx)
        if args.stateful_mgx
        else mgx_path.with_name(f"{mgx_path.stem}-stateful{mgx_path.suffix}")
    )
    prophet_dir = (
        Path(args.prophet_dir)
        if args.prophet_dir
        else artifacts_dir / f"{mgx_path.stem}-prophet"
    )
    return mgx_path, stateful_path, prophet_dir


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
) -> Dict[str, Any]:
    prefill_info = engine.prefill_context(
        prompt,
        seq_id=seq_id,
        max_new_tokens=max_new_tokens,
    )
    snapshot = engine.save_context(seq_id, text=prompt)
    return {
        "seq_id": seq_id,
        "formatted_prompt": prefill_info["formatted_prompt"],
        "prompt_len": prefill_info["prompt_len"],
        "snapshot": snapshot,
    }


def _snapshot_equivalence(a: dict, b: dict) -> dict[str, Any]:
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


def _print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> int:
    args = _parse_args()
    mgx_path, stateful_path, prophet_dir = _default_paths(args)
    dtype = _runtime_dtype(args.dtype)
    quantize = None if args.quantize == "none" else args.quantize

    _print_header("MGX STATE + PROPHET DEMO")
    print(f"Model:        {args.model}")
    print(f"MGX:          {mgx_path}")
    print(f"Stateful MGX: {stateful_path}")
    print(f"Prophet dir:  {prophet_dir}")
    print(f"Device:       {args.device}")
    print(f"DType:        {args.dtype}")
    print(f"Quantize:     {args.quantize}")

    export_started = time.perf_counter()
    export_info = None
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

    _print_header("INSPECTING BASE MGX")
    base_inspect = inspect_mgx(
        mgx_path,
        validate_payload_hash=not args.skip_hash_check,
    )
    print(json.dumps(
        {
            "session_state_present": base_inspect["session_state_present"],
            "quantization": base_inspect["manifest"]["quantization"],
            "tokenizer_hash": base_inspect["manifest"]["tokenizer_hash"],
            "source_model_hash": base_inspect["manifest"]["source_model_hash"],
        },
        indent=2,
    ))

    _print_header("PREFILLING LIVE SEQUENCE")
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
    seq_id = 101
    prefill = _prefill_prompt(
        engine,
        args.prompt,
        seq_id=seq_id,
        max_new_tokens=args.max_new_tokens,
    )
    reference_snapshot = prefill["snapshot"]
    print(json.dumps(
        {
            "formatted_prompt_len": len(prefill["formatted_prompt"]),
            "prompt_tokens": prefill["prompt_len"],
            "snapshot_seq_len": reference_snapshot["seq_len"],
        },
        indent=2,
    ))

    _print_header("EMBEDDING STATE INTO MGX")
    stateful_info = engine.save_context_to_mgx(
        seq_id,
        out_path=str(stateful_path),
        text=args.prompt,
    )
    print(json.dumps(
        {
            "session_state_present": stateful_info["session_state_present"],
            "session_manifest": stateful_info["session_state"]["manifest"]["snapshot"],
        },
        indent=2,
    ))

    _print_header("CAPTURING MGX PROPHET ENTRY")
    prophet_entry = engine.prophet_capture(
        str(prophet_dir),
        seq_id,
        text=args.prompt,
        label="colab-demo",
        metadata={
            "demo": "mgx-state-prophet",
            "artifact": str(mgx_path),
        },
    )
    prophet_matches = engine.prophet_lookup(
        str(prophet_dir),
        args.prompt,
        top_k=3,
        min_similarity=args.min_similarity,
        prefix_tokens=args.prefix_tokens,
        require_compatible=True,
    )
    print(json.dumps(
        {
            "captured_entry_id": prophet_entry["entry_id"],
            "match_count": len(prophet_matches),
            "best_match": prophet_matches[0] if prophet_matches else None,
        },
        indent=2,
    ))

    engine.block_manager.free_sequence(seq_id)

    _print_header("RESTORING EMBEDDED STATE FROM MGX")
    restored_engine = InferenceEngine(
        str(stateful_path),
        dtype=dtype,
        device=args.device,
        quantize=quantize,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        mgx_verify_payload=not args.skip_hash_check,
        mgx_prefer_payload_cache=True,
    )
    restored_seq_id = restored_engine.restore_context_from_mgx(
        seq_id=202,
        max_new_tokens=args.max_new_tokens,
    )
    restored_snapshot = restored_engine.save_context(restored_seq_id, text=args.prompt)
    restored_engine.block_manager.free_sequence(restored_seq_id)
    state_restore_equivalence = _snapshot_equivalence(reference_snapshot, restored_snapshot)
    print(json.dumps(state_restore_equivalence, indent=2))

    _print_header("RESTORING FROM MGX PROPHET")
    prophet_restore = engine.prophet_restore_best(
        str(prophet_dir),
        args.prompt,
        seq_id=303,
        max_new_tokens=args.max_new_tokens,
        top_k=3,
        min_similarity=args.min_similarity,
        prefix_tokens=args.prefix_tokens,
        require_compatible=True,
    )
    prophet_restore_equivalence = None
    if prophet_restore.get("restored"):
        prophet_snapshot = engine.save_context(prophet_restore["seq_id"], text=args.prompt)
        prophet_restore_equivalence = _snapshot_equivalence(reference_snapshot, prophet_snapshot)
        engine.block_manager.free_sequence(prophet_restore["seq_id"])
    print(json.dumps(
        {
            "restored": prophet_restore.get("restored", False),
            "reason": prophet_restore.get("reason"),
            "best_match": prophet_restore.get("match"),
            "equivalence": prophet_restore_equivalence,
        },
        indent=2,
    ))

    _print_header("SPECULATIVE PROPHET RESTORE")
    speculative_restore = engine.prophet_restore_speculative(
        str(prophet_dir),
        args.prompt,
        seq_id=404,
        max_new_tokens=args.max_new_tokens,
        top_k=3,
        min_similarity=args.min_similarity,
        prefix_tokens=args.prefix_tokens,
        require_compatible=True,
        validation_mode="full_prefill",
        validation_tokens=min(4, max(1, args.max_new_tokens)),
        agreement_threshold=1.0,
        fallback_to_prefill=True,
        min_prefix_reuse_score=args.min_prefix_reuse_score,
        min_prefix_coverage=args.min_prefix_coverage,
        max_prefix_rollback_ratio=args.max_prefix_rollback_ratio,
        max_prefix_tail_ratio=args.max_prefix_tail_ratio,
    )
    speculative_equivalence = None
    if speculative_restore.get("restored"):
        speculative_snapshot = engine.save_context(speculative_restore["seq_id"], text=args.prompt)
        speculative_equivalence = _snapshot_equivalence(reference_snapshot, speculative_snapshot)
        engine.free_sequence(speculative_restore["seq_id"])
    print(json.dumps(
        {
            "restored": speculative_restore.get("restored", False),
            "reason": speculative_restore.get("reason"),
            "committed_source": speculative_restore.get("committed_source"),
            "speculative_accepted": speculative_restore.get("speculative_accepted", False),
            "validation": speculative_restore.get("validation"),
            "equivalence": speculative_equivalence,
        },
        indent=2,
    ))

    _print_header("PREFIX REUSE PROPHET RESTORE")
    prefix_prompt = args.prompt + " Give one practical deployment example."
    prefix_baseline = _prefill_prompt(
        engine,
        prefix_prompt,
        seq_id=505,
        max_new_tokens=args.max_new_tokens,
    )
    prefix_baseline_snapshot = prefix_baseline["snapshot"]
    engine.free_sequence(prefix_baseline["seq_id"])

    prefix_restore = engine.prophet_restore_speculative(
        str(prophet_dir),
        prefix_prompt,
        seq_id=506,
        max_new_tokens=args.max_new_tokens,
        top_k=3,
        min_similarity=args.min_similarity,
        prefix_tokens=args.prefix_tokens,
        require_compatible=True,
        validation_mode="full_prefill",
        validation_tokens=min(4, max(1, args.max_new_tokens)),
        agreement_threshold=1.0,
        fallback_to_prefill=True,
        min_prefix_reuse_score=args.min_prefix_reuse_score,
        min_prefix_coverage=args.min_prefix_coverage,
        max_prefix_rollback_ratio=args.max_prefix_rollback_ratio,
        max_prefix_tail_ratio=args.max_prefix_tail_ratio,
    )
    prefix_equivalence = None
    if prefix_restore.get("restored"):
        prefix_snapshot = engine.save_context(prefix_restore["seq_id"], text=prefix_prompt)
        prefix_equivalence = _snapshot_equivalence(prefix_baseline_snapshot, prefix_snapshot)
        engine.free_sequence(prefix_restore["seq_id"])
    print(json.dumps(
        {
            "prompt": prefix_prompt,
            "restored": prefix_restore.get("restored", False),
            "reason": prefix_restore.get("reason"),
            "committed_source": prefix_restore.get("committed_source"),
            "speculative_accepted": prefix_restore.get("speculative_accepted", False),
            "validation": prefix_restore.get("validation"),
            "equivalence": prefix_equivalence,
        },
        indent=2,
    ))

    summary = {
        "config": {
            "model": args.model,
            "mgx": str(mgx_path),
            "stateful_mgx": str(stateful_path),
            "prophet_dir": str(prophet_dir),
            "device": args.device,
            "dtype": args.dtype,
            "quantize": args.quantize,
            "prompt": args.prompt,
            "max_seq_len": args.max_seq_len,
            "max_new_tokens": args.max_new_tokens,
            "min_similarity": args.min_similarity,
            "prefix_tokens": args.prefix_tokens,
            "min_prefix_reuse_score": args.min_prefix_reuse_score,
            "min_prefix_coverage": args.min_prefix_coverage,
            "max_prefix_rollback_ratio": args.max_prefix_rollback_ratio,
            "max_prefix_tail_ratio": args.max_prefix_tail_ratio,
            "skip_hash_check": args.skip_hash_check,
        },
        "timing": {
            "export_seconds": export_seconds,
            "engine_init_seconds": engine.get_init_timing().get("total_seconds"),
            "restored_engine_init_seconds": restored_engine.get_init_timing().get("total_seconds"),
        },
        "base_mgx": {
            "session_state_present": base_inspect["session_state_present"],
            "quantization": base_inspect["manifest"]["quantization"],
            "architecture": base_inspect["manifest"]["architecture"],
        },
        "stateful_mgx": {
            "session_state_present": stateful_info["session_state_present"],
            "equivalence": state_restore_equivalence,
        },
        "prophet": {
            "captured_entry_id": prophet_entry["entry_id"],
            "match_count": len(prophet_matches),
            "restore_result": prophet_restore,
            "equivalence": prophet_restore_equivalence,
            "speculative_restore_result": speculative_restore,
            "speculative_equivalence": speculative_equivalence,
            "prefix_restore_prompt": prefix_prompt,
            "prefix_restore_result": prefix_restore,
            "prefix_restore_equivalence": prefix_equivalence,
        },
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

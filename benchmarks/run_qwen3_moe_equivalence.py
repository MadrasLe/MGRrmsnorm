#!/usr/bin/env python3
"""Qwen3 MoE equivalence check: Hugging Face reference vs MegaGemm.

The full 30B-A3B model is too large to keep both runtimes resident on a single
80GB GPU. This script can run in phases:

  1. --backend hf       writes reference logits and forced decode token.
  2. --backend megagemm reads the reference and compares MegaGemm logits.

Use --backend both for a single fresh Colab process; it unloads HF before
loading MegaGemm.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_PROMPT = (
    "Write a complete Python function that computes Fibonacci numbers "
    "iteratively, with a short explanation."
)


def dtype_from_arg(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "vram_gb": round(props.total_memory / 1024**3, 2),
    }


def build_input_ids(model: str, prompt: str, *, chat_template: bool) -> torch.Tensor:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if chat_template and getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = prompt
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=not chat_template)
    return encoded["input_ids"].to(torch.long)


def topk_payload(logits: torch.Tensor, k: int) -> dict[str, Any]:
    values, indices = torch.topk(logits.float(), k=min(k, logits.numel()), dim=-1)
    return {
        "ids": [int(x) for x in indices.tolist()],
        "values": [float(x) for x in values.tolist()],
    }


def compare_logits(actual: torch.Tensor, expected: torch.Tensor, *, top_k: int) -> dict[str, Any]:
    actual_f = actual.detach().cpu().float().view(-1)
    expected_f = expected.detach().cpu().float().view(-1)
    diff = actual_f - expected_f
    actual_top = topk_payload(actual_f, top_k)
    expected_top = topk_payload(expected_f, top_k)
    actual_set = set(actual_top["ids"])
    expected_set = set(expected_top["ids"])
    denom = actual_f.norm() * expected_f.norm()
    cosine = float(torch.dot(actual_f, expected_f) / denom) if float(denom) != 0.0 else 0.0
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "cosine": cosine,
        "top1_match": int(actual_top["ids"][0]) == int(expected_top["ids"][0]),
        "top1_actual": int(actual_top["ids"][0]),
        "top1_expected": int(expected_top["ids"][0]),
        "topk_overlap": len(actual_set & expected_set),
        "topk": int(top_k),
        "actual_topk": actual_top,
        "expected_topk": expected_top,
    }


def run_hf_reference(args: argparse.Namespace, input_ids: torch.Tensor) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    dtype = dtype_from_arg(args.dtype)
    device = torch.device(args.device)
    print("== HF REFERENCE ==", flush=True)
    print(f"model: {args.model}", flush=True)
    print(f"dtype: {args.dtype}", flush=True)
    print(f"input_tokens: {int(input_ids.shape[1])}", flush=True)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    sync_cuda()
    load_ms = (time.perf_counter() - start) * 1000.0

    input_ids_dev = input_ids.to(device)
    with torch.inference_mode():
        start = time.perf_counter()
        prefill_logits = model(input_ids=input_ids_dev, use_cache=False).logits[:, -1, :]
        forced_token = int(prefill_logits.argmax(dim=-1).item())
        forced = torch.tensor([[forced_token]], device=device, dtype=torch.long)
        decode_input = torch.cat([input_ids_dev, forced], dim=1)
        decode_logits = model(input_ids=decode_input, use_cache=False).logits[:, -1, :]
        sync_cuda()
        forward_ms = (time.perf_counter() - start) * 1000.0

    result = {
        "backend": "hf",
        "model": args.model,
        "dtype": args.dtype,
        "gpu": gpu_snapshot(),
        "input_ids": input_ids.cpu(),
        "input_tokens": int(input_ids.shape[1]),
        "forced_token": forced_token,
        "prefill_logits": prefill_logits.detach().cpu().float(),
        "decode_logits": decode_logits.detach().cpu().float(),
        "load_ms": load_ms,
        "forward_ms": forward_ms,
    }
    print(
        json.dumps(
            {
                "load_ms": round(load_ms, 1),
                "forward_ms": round(forward_ms, 1),
                "forced_token": forced_token,
                "prefill_top5": topk_payload(result["prefill_logits"][0], 5),
                "decode_top5": topk_payload(result["decode_logits"][0], 5),
            },
            indent=2,
        ),
        flush=True,
    )
    del model
    cleanup_cuda()
    return result


def run_megagemm(args: argparse.Namespace, reference: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("MEGAGEMM_FP16_STREAMING", "1")
    os.environ.setdefault("MEGAGEMM_FLAT_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_DISABLE_CUDA_RMSNORM", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_BLOCK_N", "64")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_BLOCK_K", "64")
    os.environ.setdefault("MEGAGEMM_QWEN3_MOE_GROUPED_NUM_WARPS", "8")

    from megagemm.engine import InferenceEngine

    dtype = dtype_from_arg(args.dtype)
    device = torch.device(args.device)
    input_ids = reference["input_ids"].to(device=device, dtype=torch.long)
    forced_token = int(reference["forced_token"])
    seq_len = int(input_ids.shape[1])
    max_seq_len = max(int(args.max_seq_len), seq_len + 2)

    print("== MEGAGEMM CANDIDATE ==", flush=True)
    print(f"model: {args.model}", flush=True)
    print(f"dtype: {args.dtype}", flush=True)
    print(f"input_tokens: {seq_len}", flush=True)
    start = time.perf_counter()
    engine = InferenceEngine(
        args.model,
        dtype=dtype,
        device=str(device),
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )
    sync_cuda()
    load_ms = (time.perf_counter() - start) * 1000.0

    seq_id = engine._next_seq_id()
    engine.block_manager.allocate_sequence(seq_id, seq_len + 2)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    forced = torch.tensor([[forced_token]], device=device, dtype=torch.long)
    forced_pos = torch.tensor([[seq_len]], device=device, dtype=torch.long)

    with torch.inference_mode():
        start = time.perf_counter()
        prefill_logits = engine.model.prefill(input_ids, positions, engine.block_manager, seq_id)
        decode_logits = engine.model.decode_step(
            forced,
            forced_pos,
            engine.block_manager,
            [seq_id],
        )
        sync_cuda()
        forward_ms = (time.perf_counter() - start) * 1000.0

    prefill_last = prefill_logits[:, -1, :].detach().cpu().float()
    decode_last = decode_logits[:, -1, :].detach().cpu().float()
    result = {
        "backend": "megagemm",
        "model": args.model,
        "dtype": args.dtype,
        "gpu": gpu_snapshot(),
        "input_tokens": seq_len,
        "forced_token": forced_token,
        "prefill_logits": prefill_last,
        "decode_logits": decode_last,
        "load_ms": load_ms,
        "forward_ms": forward_ms,
    }
    comparisons = {
        "prefill": compare_logits(
            prefill_last[0],
            reference["prefill_logits"][0],
            top_k=args.top_k,
        ),
        "decode": compare_logits(
            decode_last[0],
            reference["decode_logits"][0],
            top_k=args.top_k,
        ),
    }
    result["comparisons"] = comparisons
    print(
        json.dumps(
            {
                "load_ms": round(load_ms, 1),
                "forward_ms": round(forward_ms, 1),
                "forced_token": forced_token,
                "comparisons": comparisons,
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def save_payload(path: str, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"wrote {out}", flush=True)


def load_payload(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def jsonable_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "backend": payload.get("backend"),
        "model": payload.get("model"),
        "dtype": payload.get("dtype"),
        "gpu": payload.get("gpu"),
        "input_tokens": payload.get("input_tokens"),
        "forced_token": payload.get("forced_token"),
        "load_ms": payload.get("load_ms"),
        "forward_ms": payload.get("forward_ms"),
    }
    if "comparisons" in payload:
        summary["comparisons"] = payload["comparisons"]
    return summary


def enforce_thresholds(args: argparse.Namespace, result: dict[str, Any]) -> None:
    comparisons = result.get("comparisons") or {}
    failures = []
    for name, row in comparisons.items():
        if float(row["cosine"]) < float(args.min_cosine):
            failures.append(f"{name}: cosine {row['cosine']:.8f} < {args.min_cosine}")
        if float(row["max_abs"]) > float(args.max_abs):
            failures.append(f"{name}: max_abs {row['max_abs']:.6f} > {args.max_abs}")
        if args.require_top1_match and not bool(row["top1_match"]):
            failures.append(
                f"{name}: top1 mismatch actual={row['top1_actual']} expected={row['top1_expected']}"
            )
    if failures:
        raise SystemExit("Equivalence check failed:\n  " + "\n  ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hf", "megagemm", "both"], default="both")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--reference", default="")
    parser.add_argument("--out", default="bench_results/qwen3_moe_equivalence.pt")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-abs", type=float, default=2.0)
    parser.add_argument("--require-top1-match", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    print("== QWEN3 MOE EQUIVALENCE ==", flush=True)
    print(f"backend: {args.backend}", flush=True)
    print(f"gpu: {gpu_snapshot()}", flush=True)

    result: dict[str, Any]
    reference: dict[str, Any]
    if args.backend in {"hf", "both"}:
        input_ids = build_input_ids(
            args.model,
            args.prompt,
            chat_template=not args.no_chat_template,
        )
        reference = run_hf_reference(args, input_ids)
        if args.backend == "hf":
            save_payload(args.out, reference)
            result = reference
        else:
            result = run_megagemm(args, reference)
            result["reference_backend"] = "hf"
            save_payload(args.out, result)
    else:
        if not args.reference:
            raise SystemExit("--backend megagemm requires --reference from a previous --backend hf run")
        reference = load_payload(args.reference)
        result = run_megagemm(args, reference)
        result["reference_path"] = args.reference
        save_payload(args.out, result)

    summary = jsonable_summary(result)
    print("== SUMMARY ==", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.summary_json:
        out_json = Path(args.summary_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {out_json}", flush=True)
    if args.backend != "hf":
        enforce_thresholds(args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

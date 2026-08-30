"""
Profile Gemma 4 decode on MegaGemm with torch.profiler + model timing buckets.

Designed for Kaggle / Colab GPU runs. The profiler keeps the validated
production decode fusions enabled while collecting the internal CUDA-event
breakdown.

    python benchmarks/profile_gemma4_decode.py \
      --model google/gemma-4-E2B-it \
      --dtype bf16 \
      --batch-size 8 \
      --prompt-tokens 2048 \
      --max-new-tokens 32 \
      --ignore-eos \
      --out /tmp/gemma4_e2b_l4_decode_profile.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


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


def _percent(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return part / total * 100.0


def _print_group(summary: dict[str, float], title: str, keys: list[tuple[str, str]], total_key: str) -> None:
    total = float(summary.get(total_key, 0.0) or 0.0)
    print(f"\n{title}")
    print(f"  total: {total:.2f} ms")
    for key, label in keys:
        if key not in summary:
            continue
        value = float(summary[key])
        print(f"  {label:<18} {value:8.2f} ms  ({_percent(value, total):5.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile Gemma 4 decode on MegaGemm")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quantize", choices=["int8", "fp8", "awq"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="Build an exact synthetic prompt per request; 0 uses --prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument("--kv-alloc", default="auto", choices=["auto", "greedy"])
    parser.add_argument(
        "--prompt",
        default="Explique KV cache em uma frase e depois diga por que sliding attention existe.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--out", default="gemma4_decode_profile.json")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    # Recreate the promoted E2B profile in every process instead of inheriting
    # stale notebook flags.  Decode timing must then be enabled before llama.py
    # is imported because its instrumentation flags are module-level.
    from benchmarks.run_gemma4_e2b_phase_split import (
        _configure_megagemm_profile,
    )

    profile_environment = _configure_megagemm_profile(args.model)
    os.environ["MEGAGEMM_DECODE_TIMING"] = "1"
    os.environ["MEGAGEMM_DECODE_TIMING_PRINT"] = "0"

    from megagemm.engine import InferenceEngine

    dtype = _runtime_dtype(args.dtype)
    max_batch_size = args.max_batch_size or args.batch_size
    print("MegaGemm Gemma4 decode profiler")
    print(f"  model:          {args.model}")
    print(f"  device:         {args.device}")
    print(f"  dtype:          {args.dtype}")
    print(f"  quantize:       {args.quantize or 'none'}")
    print(f"  batch_size:     {args.batch_size}")
    print(f"  prompt_tokens:  {args.prompt_tokens or 'text prompt'}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  gpu:            {_gpu_snapshot()}")
    print(f"  decode_timing:  {os.environ.get('MEGAGEMM_DECODE_TIMING')}")
    print(f"  profile:        {profile_environment}")
    if args.batch_size > 1 and args.repetition_penalty != 1.0:
        print("  note:           generate_batch() currently ignores repetition_penalty")

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
    if args.prompt_tokens > 0:
        from benchmarks.benchmark_inference_matrix import build_prompts

        prompts, prompt_tokens_actual = build_prompts(
            engine.tokenizer,
            args.batch_size,
            args.prompt_tokens,
        )
    else:
        prompts = [args.prompt] * args.batch_size
        prompt_tokens_actual = sum(
            len(engine.tokenizer.encode(prompt, add_special_tokens=False))
            for prompt in prompts
        )
    print(f"  prompt_actual:  {prompt_tokens_actual} total tokens")

    runtime_stats_fn = getattr(engine.model, "decode_runtime_stats", None)
    runtime_before = runtime_stats_fn() if callable(runtime_stats_fn) else {}

    warmup_tokens = min(args.warmup_tokens, args.max_new_tokens)
    if warmup_tokens > 0:
        print("\nWarmup")
        engine.generate_batch(
            prompts,
            max_new_tokens=warmup_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            ignore_eos=args.ignore_eos,
        )
        _sync(args.device)

    print("\nProfiling")
    _empty_cache(args.device)
    summary = engine.profile_decode_breakdown(
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        ignore_eos=args.ignore_eos,
    )
    _sync(args.device)

    scheduler_stats = {}
    scheduler = getattr(engine, "_last_scheduler", None)
    if scheduler is not None:
        scheduler_stats = scheduler.get_stats()
    runtime_after = runtime_stats_fn() if callable(runtime_stats_fn) else {}

    decode_total = float(summary.get("decode_total_ms", 0.0) or 0.0)
    print(f"\nScheduler stats: {scheduler_stats}")
    if decode_total > 0.0:
        _print_group(
            summary,
            "Decode Stage Breakdown",
            [
                ("decode_embed_ms", "embed"),
                ("decode_attn_ms", "attn"),
                ("decode_mlp_ms", "mlp"),
                ("decode_ple_ms", "ple"),
                ("decode_lm_head_ms", "lm_head"),
                ("decode_sample_ms", "sample"),
            ],
            "decode_total_ms",
        )
        _print_group(
            summary,
            "Decode Attention Details",
            [
                ("decode_attn_input_norm_ms", "input_norm"),
                ("decode_attn_qkv_ms", "qkv"),
                ("decode_attn_norm_rope_ms", "norm+rope"),
                ("decode_attn_kv_write_ms", "kv_write"),
                ("decode_attn_core_ms", "attn_core"),
                ("decode_attn_core_sliding_ms", "core_sliding"),
                ("decode_attn_core_full_ms", "core_full"),
                ("decode_attn_o_proj_ms", "o_proj"),
                ("decode_attn_output_norm_ms", "output_norm"),
            ],
            "decode_attn_ms",
        )
        _print_group(
            summary,
            "Decode MLP Details",
            [
                ("decode_mlp_input_norm_ms", "input_norm"),
                ("decode_mlp_gate_up_ms", "gate_up"),
                ("decode_mlp_act_ms", "activation"),
                ("decode_mlp_down_ms", "down_proj"),
                ("decode_mlp_output_norm_ms", "output_norm"),
                ("decode_dense_post_norm_chain_ms", "dense_tail_chain"),
            ],
            "decode_mlp_ms",
        )

    counter_keys = (
        "gemma4_flat_fused_qkv_layers",
        "gemma4_flat_fused_gateup_hits",
        "gemma4_flat_deepfusion_hits",
        "gemma4_ple_conditioned_gelu_decode_hits",
        "gemma4_cublaslt_gateup_decode_hits",
        "gemma4_dense_post_norm_chain_decode_hits",
        "gemma4_dense_next_attn_norm_decode_hits",
        "gemma4_dense_attn_mlp_bridge_decode_hits",
    )
    runtime_counter_delta = {}
    for key in counter_keys:
        before = runtime_before.get(key, 0)
        after = runtime_after.get(key, 0)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            runtime_counter_delta[key] = after - before
    paged_before = runtime_before.get("paged_decode_runtime") or {}
    paged_after = runtime_after.get("paged_decode_runtime") or {}
    for key in (
        "gqa2_direct_hits",
        "generic_direct_hits",
        "grouped_segmented_hits",
    ):
        before = paged_before.get(key, 0)
        after = paged_after.get(key, 0)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            runtime_counter_delta[f"paged_decode_{key}"] = after - before
    print("\nProduction fast-path counter deltas")
    print(json.dumps(runtime_counter_delta, indent=2))

    print("\nTorch Profiler Buckets")
    for key in (
        "cpu_launch_ms",
        "cpu_alloc_ms",
        "cpu_view_ms",
        "cuda_gemv_ms",
        "cuda_fused_norm_qkv_ms",
        "cuda_attn_ms",
        "cuda_norm_ms",
        "cuda_deepfusion_ms",
        "cuda_fused_lm_head_argmax_ms",
        "cuda_swiglu_ms",
        "launch_calls",
    ):
        if key in summary:
            value = summary[key]
            suffix = "" if key == "launch_calls" else " ms"
            if key == "launch_calls":
                print(f"  {key:<28} {int(value)}")
            else:
                print(f"  {key:<28} {float(value):.2f}{suffix}")

    payload = {
        "args": vars(args),
        "profile_environment": profile_environment,
        "gpu": _gpu_snapshot(),
        "prompt_tokens_actual_total": prompt_tokens_actual,
        "scheduler_stats": scheduler_stats,
        "decode_runtime_before": runtime_before,
        "decode_runtime_after": runtime_after,
        "decode_runtime_counter_delta": runtime_counter_delta,
        "summary": summary,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

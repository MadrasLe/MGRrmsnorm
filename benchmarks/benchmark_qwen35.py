"""
Benchmark suite for Qwen 3.5 on MegaGemm.

Modes:
  compare  - Compare MegaGemm native vs Hugging Face on greedy cached decode
  sweep    - Sweep native latency/throughput over prompt/output lengths
  batch    - Measure MegaGemm continuous batching throughput

Usage:
    python benchmarks/benchmark_qwen35.py compare --model Qwen/Qwen3.5-0.8B
    python benchmarks/benchmark_qwen35.py sweep --model Qwen/Qwen3.5-0.8B
    python benchmarks/benchmark_qwen35.py batch --model Qwen/Qwen3.5-0.8B
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from statistics import mean
from typing import List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from megagemm.engine import InferenceEngine


COMPARE_PROMPTS = [
    "Explique linear attention em transformers em 4 frases curtas.",
    "Compare linear attention e softmax attention em termos de complexidade e memoria.",
    "Descreva como o cache KV interage com decode autoregressivo.",
]


def get_compare_decode_burst() -> int:
    raw = os.environ.get("MEGAGEMM_MULTI_STEP_BURST_SINGLE", "").strip()
    if not raw:
        raw = os.environ.get("MEGAGEMM_MULTI_STEP_BURST", "").strip()
    if not raw:
        raw = "16"
    return max(1, int(raw))


def env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass
class RunMetrics:
    prompt_tokens: int
    output_tokens: int
    prefill_ms: float
    decode_ms: float
    total_ms: float
    ttft_ms: float
    decode_tps: float
    total_tps: float
    peak_vram_mb: float


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def sync_if_cuda(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def parse_int_list(value) -> list[int]:
    """
    Accept both CLI styles:
      --prompt-lengths 1024,2048,4096
      --prompt-lengths 1024 2048 4096
    """
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            parts.extend(str(item).split(","))
    else:
        parts = str(value).split(",")
    return [int(part) for part in parts if str(part).strip()]


def ensure_num_blocks_capacity(
    num_blocks: int,
    max_seq_len: int,
    *,
    max_batch_size: int = 1,
    block_size: int = 16,
) -> int:
    if num_blocks <= 0:
        return num_blocks
    needed = max_batch_size * max(1, (max_seq_len + block_size - 1) // block_size)
    if num_blocks < needed:
        print(
            f"[MegaGemm] Increasing num_blocks {num_blocks} -> {needed} "
            f"for max_seq_len={max_seq_len}, max_batch_size={max_batch_size}.",
            flush=True,
        )
        return needed
    return num_blocks


def format_prompt(tokenizer, prompt: str) -> str:
    try:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def hf_decode_step(hf_model, next_ids, prev_out, next_positions):
    kwargs = {
        "input_ids": next_ids,
        "past_key_values": prev_out.past_key_values,
        "use_cache": True,
        "return_dict": True,
    }
    try:
        return hf_model(
            **kwargs,
            position_ids=next_positions,
            cache_position=next_positions.view(-1),
        )
    except TypeError:
        try:
            return hf_model(**kwargs, position_ids=next_positions)
        except TypeError:
            return hf_model(**kwargs)


def run_megagemm_greedy(engine: InferenceEngine, prompt: str, max_new_tokens: int) -> RunMetrics:
    seq_id = engine._next_seq_id()
    bos = engine.tokenizer.bos_token
    already_formatted = bos and prompt.startswith(bos)
    formatted = prompt
    if not already_formatted and getattr(engine.tokenizer, "chat_template", None):
        formatted = format_prompt(engine.tokenizer, prompt)
    add_special = not (bos and formatted.startswith(bos))
    input_ids = engine.tokenizer.encode(
        formatted, return_tensors="pt", add_special_tokens=add_special,
    ).to(engine.device)
    prompt_len = input_ids.shape[1]
    needed_context = prompt_len + max(1, max_new_tokens)
    max_context = int(getattr(engine.model.config, "max_position_embeddings", needed_context))
    if needed_context > max_context:
        raise ValueError(
            f"Requested context needs {needed_context} tokens, but model max_position_embeddings={max_context}. "
            "Use shorter prompt/output lengths."
        )
    rope_len = int(getattr(engine.model, "_rope_cache_max_seq_len", 0) or 0)
    if needed_context > rope_len and hasattr(engine.model, "set_rope_cache_max_seq_len"):
        print(
            f"[MegaGemm] Resizing RoPE cache for benchmark context: "
            f"{rope_len} -> {needed_context} tokens"
        )
        engine.model.set_rope_cache_max_seq_len(needed_context, device=engine.device)

    engine.block_manager.allocate_sequence(seq_id, prompt_len + max_new_tokens)
    decode_input = torch.empty(1, 1, dtype=torch.long, device=engine.device)
    decode_pos = torch.empty(1, 1, dtype=torch.long, device=engine.device)

    try:
        positions = torch.arange(prompt_len, device=engine.device).unsqueeze(0)
        sync_if_cuda(engine.device)
        t0 = time.perf_counter()
        logits = engine.model.prefill(input_ids, positions, engine.block_manager, seq_id)
        sync_if_cuda(engine.device)
        t1 = time.perf_counter()

        next_token_id = int(logits[:, -1, :].argmax(dim=-1).item())
        generated_ids = [next_token_id]

        remaining = max_new_tokens - 1
        pos_counter = prompt_len
        decode_burst = get_compare_decode_burst()
        has_multi = (
            hasattr(engine.model, "decode_multi_step")
            and decode_burst > 1
            and not env_enabled("MEGAGEMM_BENCH_DISABLE_MULTI_STEP", False)
        )

        while remaining > 0:
            if has_multi:
                burst = min(decode_burst, remaining)
                decode_input.fill_(next_token_id)
                decode_pos.fill_(pos_counter)
                all_tokens, _ = engine.model.decode_multi_step(
                    decode_input,
                    decode_pos,
                    engine.block_manager,
                    [seq_id],
                    num_steps=burst,
                    return_final_logits=False,
                )
                burst_tokens = all_tokens[0].tolist()
                generated_ids.extend(burst_tokens)
                next_token_id = int(burst_tokens[-1])
                pos_counter += len(burst_tokens)
                remaining -= burst
            else:
                decode_input.fill_(next_token_id)
                decode_pos.fill_(pos_counter)
                logits = engine.model.decode_step(
                    decode_input, decode_pos, engine.block_manager, [seq_id],
                )
                next_token_id = int(logits[:, -1, :].argmax(dim=-1).item())
                generated_ids.append(next_token_id)
                pos_counter += 1
                remaining -= 1

        sync_if_cuda(engine.device)
        t2 = time.perf_counter()

        output_tokens = len(generated_ids)
        prefill_ms = (t1 - t0) * 1000
        decode_ms = (t2 - t1) * 1000
        total_ms = (t2 - t0) * 1000
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        return RunMetrics(
            prompt_tokens=prompt_len,
            output_tokens=output_tokens,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            total_ms=total_ms,
            ttft_ms=prefill_ms,
            decode_tps=output_tokens / max(t2 - t1, 1e-9),
            total_tps=output_tokens / max(t2 - t0, 1e-9),
            peak_vram_mb=peak_vram,
        )
    finally:
        engine.block_manager.free_sequence(seq_id)


def load_hf_model(model_name: str, dtype: torch.dtype, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    ).eval()
    return tokenizer, model


def run_hf_greedy(tokenizer, model, prompt: str, max_new_tokens: int, device: str):
    formatted = format_prompt(tokenizer, prompt)
    input_ids = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_len = input_ids.shape[1]
    sync_if_cuda(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
    sync_if_cuda(device)
    t1 = time.perf_counter()

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    prev_out = out
    generated_ids = [int(next_token.item())]

    for step in range(max_new_tokens - 1):
        next_pos = torch.tensor([[prompt_len + step]], device=device, dtype=torch.long)
        with torch.no_grad():
            prev_out = hf_decode_step(model, next_token, prev_out, next_pos)
        next_token = prev_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(int(next_token.item()))

    sync_if_cuda(device)
    t2 = time.perf_counter()

    output_tokens = len(generated_ids)
    prefill_ms = (t1 - t0) * 1000
    decode_ms = (t2 - t1) * 1000
    total_ms = (t2 - t0) * 1000
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    return RunMetrics(
        prompt_tokens=prompt_len,
        output_tokens=output_tokens,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        total_ms=total_ms,
        ttft_ms=prefill_ms,
        decode_tps=output_tokens / max(t2 - t1, 1e-9),
        total_tps=output_tokens / max(t2 - t0, 1e-9),
        peak_vram_mb=peak_vram,
    )


def summarize_metrics(metrics: List[RunMetrics]):
    return {
        "prompt_tokens": round(mean(m.prompt_tokens for m in metrics), 2),
        "output_tokens": round(mean(m.output_tokens for m in metrics), 2),
        "prefill_ms": round(mean(m.prefill_ms for m in metrics), 2),
        "decode_ms": round(mean(m.decode_ms for m in metrics), 2),
        "total_ms": round(mean(m.total_ms for m in metrics), 2),
        "ttft_ms": round(mean(m.ttft_ms for m in metrics), 2),
        "decode_tps": round(mean(m.decode_tps for m in metrics), 2),
        "total_tps": round(mean(m.total_tps for m in metrics), 2),
        "peak_vram_mb": round(max(m.peak_vram_mb for m in metrics), 2),
    }


def print_summary(label: str, summary: dict):
    print(f"\n{label}")
    print(f"  prompt_tokens: {summary['prompt_tokens']}")
    print(f"  output_tokens: {summary['output_tokens']}")
    print(f"  ttft_ms:       {summary['ttft_ms']}")
    print(f"  decode_ms:     {summary['decode_ms']}")
    print(f"  total_ms:      {summary['total_ms']}")
    print(f"  decode_tps:    {summary['decode_tps']} tok/s")
    print(f"  total_tps:     {summary['total_tps']} tok/s")
    print(f"  peak_vram_mb:  {summary['peak_vram_mb']}")
    if "model_vram_mb" in summary:
        print(f"  model_vram_mb: {summary['model_vram_mb']}")
    if "kv_reserved_mb" in summary:
        print(f"  kv_reserved_mb: {summary['kv_reserved_mb']}")
    if "weights_vram_mb" in summary:
        print(f"  weights_vram_mb: {summary['weights_vram_mb']}")


def print_decode_runtime(label: str, stats: dict):
    if not stats:
        return
    print(f"\n{label} decode fast paths")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def build_target_prompt(tokenizer, target_tokens: int) -> str:
    seed = "Linear attention improves asymptotic scaling in autoregressive transformers. "
    text = seed
    while True:
        formatted = format_prompt(tokenizer, text)
        tok_count = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
        if tok_count >= target_tokens:
            return text
        text += seed


def run_compare(args):
    dtype = parse_dtype(args.dtype)
    prompts = [args.prompt] if args.prompt else COMPARE_PROMPTS

    print("🏁 Qwen3.5 MegaGemm vs Hugging Face")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Runs: warmup={args.warmup}, bench={args.runs}, max_new_tokens={args.max_new_tokens}")
    if env_enabled("MEGAGEMM_DECODE_TIMING", False):
        print(
            "Warning: MEGAGEMM_DECODE_TIMING=1 adds CUDA event/sync overhead. "
            "Use this run for decode breakdown, not final TPS comparison."
        )

    mg_metrics = []
    mg_decode_runtime = {}
    clear_gpu()
    engine = InferenceEngine(
        args.model,
        dtype=dtype,
        device=args.device,
        num_blocks=args.num_blocks,
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
    )
    mg_model_vram_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    mg_kv_reserved_mb = engine.block_manager.memory_usage_mb()
    try:
        for prompt in prompts[:1]:
            for _ in range(args.warmup):
                _ = run_megagemm_greedy(engine, prompt, min(16, args.max_new_tokens))
        for _ in range(args.runs):
            for prompt in prompts:
                clear_gpu()
                mg_metrics.append(run_megagemm_greedy(engine, prompt, args.max_new_tokens))
        if hasattr(engine.model, "decode_runtime_stats"):
            mg_decode_runtime = engine.model.decode_runtime_stats()
    finally:
        del engine
        clear_gpu()

    mg_summary = summarize_metrics(mg_metrics)
    mg_summary["model_vram_mb"] = round(mg_model_vram_mb, 2)
    mg_summary["kv_reserved_mb"] = round(mg_kv_reserved_mb, 2)
    mg_summary["weights_vram_mb"] = round(max(mg_model_vram_mb - mg_kv_reserved_mb, 0.0), 2)
    print_summary("MegaGemm Native", mg_summary)
    print_decode_runtime("MegaGemm Native", mg_decode_runtime)

    hf_metrics = []
    hf_summary = None
    hf_error = None
    gap = None
    if getattr(args, "skip_hf", False):
        print("\nHugging Face")
        print("  skipped: --skip-hf/--native-only")
    else:
        clear_gpu()
        hf_model = None
        try:
            hf_tokenizer, hf_model = load_hf_model(args.model, dtype, args.device)
            hf_model_vram_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
            for prompt in prompts[:1]:
                for _ in range(args.warmup):
                    _ = run_hf_greedy(hf_tokenizer, hf_model, prompt, min(16, args.max_new_tokens), args.device)
            for _ in range(args.runs):
                for prompt in prompts:
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    hf_metrics.append(run_hf_greedy(hf_tokenizer, hf_model, prompt, args.max_new_tokens, args.device))

            hf_summary = summarize_metrics(hf_metrics)
            hf_summary["model_vram_mb"] = round(hf_model_vram_mb, 2)
            print_summary("Hugging Face", hf_summary)

            gap = {
                "decode_tps_ratio_mg_over_hf": round(mg_summary["decode_tps"] / max(hf_summary["decode_tps"], 1e-9), 4),
                "total_tps_ratio_mg_over_hf": round(mg_summary["total_tps"] / max(hf_summary["total_tps"], 1e-9), 4),
                "ttft_ratio_mg_over_hf": round(mg_summary["ttft_ms"] / max(hf_summary["ttft_ms"], 1e-9), 4),
            }
            print("\nGap")
            print(f"  decode_tps_ratio_mg_over_hf: {gap['decode_tps_ratio_mg_over_hf']}")
            print(f"  total_tps_ratio_mg_over_hf:  {gap['total_tps_ratio_mg_over_hf']}")
            print(f"  ttft_ratio_mg_over_hf:       {gap['ttft_ratio_mg_over_hf']}")
        except Exception as exc:
            hf_error = f"{type(exc).__name__}: {exc}"
            print("\nHugging Face")
            print(f"  unavailable: {hf_error}")
            print("  MegaGemm result above is still valid. Use --skip-hf to suppress this check.")
        finally:
            if hf_model is not None:
                del hf_model
            clear_gpu()

    if args.out:
        payload = {
            "mode": "compare",
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "runs": args.runs,
            "max_new_tokens": args.max_new_tokens,
            "prompts": prompts,
            "megagemm": {"summary": mg_summary, "runs": [asdict(m) for m in mg_metrics]},
            "megagemm_decode_runtime": mg_decode_runtime,
            "huggingface": (
                {"summary": hf_summary, "runs": [asdict(m) for m in hf_metrics]}
                if hf_summary is not None
                else {"summary": None, "runs": [], "error": hf_error, "skipped": bool(getattr(args, "skip_hf", False))}
            ),
            "gap": gap,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to {args.out}")


def run_sweep(args):
    dtype = parse_dtype(args.dtype)
    prompt_lengths = parse_int_list(args.prompt_lengths)
    output_lengths = parse_int_list(args.output_lengths)
    args.num_blocks = ensure_num_blocks_capacity(
        args.num_blocks,
        args.max_seq_len,
        max_batch_size=1,
    )
    total_cases = len(prompt_lengths) * len(output_lengths)

    print("📊 Qwen3.5 Native Throughput/Latency Sweep")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Prompt lengths: {prompt_lengths}")
    print(f"Output lengths: {output_lengths}")
    if args.warmup == 0 and args.runs == 1:
        print("Warning: warmup=0 and runs=1 measures cold-start/JIT and may under-report throughput.")

    clear_gpu()
    engine = InferenceEngine(
        args.model,
        dtype=dtype,
        device=args.device,
        num_blocks=args.num_blocks,
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
    )
    prompt_map = {prompt_len: build_target_prompt(engine.tokenizer, prompt_len) for prompt_len in prompt_lengths}

    rows = []
    try:
        case_idx = 0
        for prompt_len in prompt_lengths:
            prompt = prompt_map[prompt_len]
            for output_len in output_lengths:
                case_idx += 1
                print(
                    f"  [{case_idx}/{total_cases}] running prompt≈{prompt_len} tok | out={output_len} tok "
                    f"(warmup={args.warmup}, runs={args.runs})",
                    flush=True,
                )
                for _ in range(args.warmup):
                    _ = run_megagemm_greedy(engine, prompt, min(output_len, 16))

                metrics = []
                for _ in range(args.runs):
                    clear_gpu()
                    metrics.append(run_megagemm_greedy(engine, prompt, output_len))

                summary = summarize_metrics(metrics)
                row = {
                    "target_prompt_tokens": prompt_len,
                    "output_tokens": output_len,
                    **summary,
                }
                rows.append(row)
                print(
                    f"  prompt≈{prompt_len:>4} tok (actual {summary['prompt_tokens']:>5.1f}) | out={output_len:>4} tok | "
                    f"ttft={summary['ttft_ms']:>8.2f} ms | "
                    f"decode={summary['decode_tps']:>7.2f} tok/s | "
                    f"total={summary['total_tps']:>7.2f} tok/s | "
                    f"vram={summary['peak_vram_mb']:>7.2f} MB"
                )
    finally:
        del engine
        clear_gpu()

    if args.out:
        payload = {
            "mode": "sweep",
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "runs": args.runs,
            "rows": rows,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to {args.out}")


def run_batch(args):
    dtype = parse_dtype(args.dtype)
    batch_sizes = parse_int_list(args.batch_sizes)
    args.num_blocks = ensure_num_blocks_capacity(
        args.num_blocks,
        args.max_seq_len,
        max_batch_size=max(batch_sizes),
    )
    base_prompt = args.prompt or COMPARE_PROMPTS[0]

    print("📦 Qwen3.5 Continuous Batching Benchmark")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Max new tokens: {args.max_new_tokens}")

    clear_gpu()
    engine = InferenceEngine(
        args.model,
        dtype=dtype,
        device=args.device,
        num_blocks=args.num_blocks,
        max_batch_size=max(batch_sizes),
        max_seq_len=args.max_seq_len,
    )

    rows = []
    try:
        for batch_size in batch_sizes:
            prompts = [f"{base_prompt} [{i}]" for i in range(batch_size)]

            try:
                for _ in range(args.warmup):
                    _ = engine.generate_batch(
                        prompts,
                        max_new_tokens=min(args.max_new_tokens, 16),
                        temperature=0.0,
                        verbose=False,
                    )

                times = []
                total_tokens = []
                peak_vram = 0.0
                for _ in range(args.runs):
                    clear_gpu()
                    sync_if_cuda(args.device)
                    t0 = time.perf_counter()
                    outputs = engine.generate_batch(
                        prompts,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.0,
                        verbose=False,
                    )
                    sync_if_cuda(args.device)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
                    total_tokens.append(sum(len(engine.tokenizer.encode(o)) for o in outputs))
                    if torch.cuda.is_available():
                        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1024**2)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                msg = str(exc)
                if not isinstance(exc, torch.OutOfMemoryError) and "out of memory" not in msg.lower():
                    raise
                clear_gpu()
                row = {
                    "batch_size": batch_size,
                    "error": "cuda_oom",
                    "message": msg.splitlines()[0][:220],
                }
                rows.append(row)
                print(f"  batch={batch_size:>3} | OOM | {row['message']}")
                break

            avg_time = mean(times)
            avg_tokens = mean(total_tokens)
            total_tps = avg_tokens / max(avg_time, 1e-9)
            per_seq_tps = total_tps / batch_size
            row = {
                "batch_size": batch_size,
                "total_tokens": round(avg_tokens, 2),
                "total_ms": round(avg_time * 1000, 2),
                "total_tps": round(total_tps, 2),
                "per_seq_tps": round(per_seq_tps, 2),
                "peak_vram_mb": round(peak_vram, 2),
            }
            rows.append(row)
            print(
                f"  batch={batch_size:>3} | total={row['total_tps']:>7.2f} tok/s | "
                f"per-seq={row['per_seq_tps']:>6.2f} tok/s | "
                f"time={row['total_ms']:>8.2f} ms | vram={row['peak_vram_mb']:>7.2f} MB"
            )
    finally:
        del engine
        clear_gpu()

    if args.out:
        payload = {
            "mode": "batch",
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "runs": args.runs,
            "max_new_tokens": args.max_new_tokens,
            "rows": rows,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to {args.out}")


def main():
    parser = argparse.ArgumentParser(description="Qwen 3.5 benchmark suite for MegaGemm")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    compare = subparsers.add_parser("compare", help="Compare MegaGemm native vs Hugging Face")
    compare.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    compare.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    compare.add_argument("--device", default="cuda")
    compare.add_argument("--max-new-tokens", type=int, default=128)
    compare.add_argument("--warmup", type=int, default=1)
    compare.add_argument("--runs", type=int, default=3)
    compare.add_argument("--num-blocks", type=int, default=512)
    compare.add_argument("--max-seq-len", type=int, default=1024)
    compare.add_argument("--prompt", default=None)
    compare.add_argument("--out", default=None)
    compare.add_argument(
        "--skip-hf",
        "--native-only",
        dest="skip_hf",
        action="store_true",
        help="Run only MegaGemm native metrics; do not load Hugging Face baseline.",
    )

    sweep = subparsers.add_parser("sweep", help="Benchmark native throughput/latency sweep")
    sweep.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    sweep.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    sweep.add_argument("--device", default="cuda")
    sweep.add_argument("--prompt-lengths", nargs="+", default="32,128,512")
    sweep.add_argument("--output-lengths", nargs="+", default="32,128,256")
    sweep.add_argument("--warmup", type=int, default=1)
    sweep.add_argument("--runs", type=int, default=3)
    sweep.add_argument("--num-blocks", type=int, default=0)
    sweep.add_argument("--max-seq-len", type=int, default=2048)
    sweep.add_argument("--out", default=None)

    batch = subparsers.add_parser("batch", help="Benchmark MegaGemm continuous batching")
    batch.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    batch.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    batch.add_argument("--device", default="cuda")
    batch.add_argument("--batch-sizes", nargs="+", default="1,2,4,8")
    batch.add_argument("--max-new-tokens", type=int, default=128)
    batch.add_argument("--warmup", type=int, default=1)
    batch.add_argument("--runs", type=int, default=3)
    batch.add_argument("--num-blocks", type=int, default=1024)
    batch.add_argument("--max-seq-len", type=int, default=1024)
    batch.add_argument("--prompt", default=None)
    batch.add_argument("--out", default=None)

    args = parser.parse_args()
    if args.mode == "compare":
        run_compare(args)
    elif args.mode == "sweep":
        run_sweep(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()

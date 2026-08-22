#!/usr/bin/env python3
"""
Bench360-style throughput benchmark for vLLM.

Designed to mirror `examples/bench360_compare.py` as closely as possible:
  - same prompt set
  - same approximate 320-token input shaping
  - same batch sizes
  - same max output tokens
  - same output TPS metric = output_tokens / wall_time

Typical Colab usage:

    %cd /content/drive/MyDrive/MGRrmsnorm
    %pip install -U vllm
    !python examples/bench360_vllm_compare.py --max-batch 128
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer


def _load_workload() -> Tuple[List[str], int]:
    src = Path(__file__).with_name("bench360_compare.py")
    spec = importlib.util.spec_from_file_location("_bench360_workload", src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load workload from {src}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PROMPTS_320), int(module.TARGET_INPUT_TOKENS)


PROMPTS_320, TARGET_INPUT_TOKENS = _load_workload()


def _prepare_prompts(
    tokenizer,
    prompts: List[str],
    target_input_tokens: int,
) -> Tuple[List[str], List[int]]:
    pad_phrase = (
        " Provide thorough analysis with specific examples, data points, and citations "
        "from peer-reviewed literature where applicable. Consider multiple perspectives "
        "and discuss counterarguments."
    )
    prepared: List[str] = []
    prompt_lens: List[int] = []

    for prompt in prompts:
        text = prompt
        while True:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            toks = tokenizer.encode(formatted)
            cur_len = len(toks)
            if cur_len >= target_input_tokens:
                break
            text += pad_phrase

        if cur_len > target_input_tokens + 20:
            toks = toks[:target_input_tokens]
            cur_len = target_input_tokens

        prepared.append(text)
        prompt_lens.append(cur_len)

    return prepared, prompt_lens


def _build_conversations(prompts: List[str]) -> List[List[dict]]:
    return [[{"role": "user", "content": prompt}] for prompt in prompts]


def _count_output_tokens(tokenizer, outputs) -> int:
    total = 0
    for out in outputs:
        seq_out = out.outputs[0]
        token_ids = getattr(seq_out, "token_ids", None)
        if token_ids is not None:
            total += len(token_ids)
        else:
            total += len(tokenizer.encode(seq_out.text))
    return total


def run_bench360_vllm(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_tokens: int = 32,
    batch_sizes: List[int] | None = None,
    num_warmup: int = 3,
    shape_warmup_runs: int = 0,
    dtype: str = "float16",
    gpu_memory_utilization: float = 0.90,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    enable_prefix_caching: bool = False,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = False,
    trust_remote_code: bool = False,
) -> None:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover - runtime-only dependency
        raise SystemExit(
            "vLLM is required for this benchmark. Install it first, e.g. `%pip install -U vllm`.\n"
            f"Import error: {exc}"
        )

    if batch_sizes is None:
        batch_sizes = [16, 32, 64, 128]

    gpu_name = "CPU"
    vram_gb = 0.0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"\n{'='*70}")
    print("  📊 Bench360-Compatible Benchmark — vLLM")
    print(f"{'='*70}")
    print("  Paper:    arXiv:2511.16682 (Bench360)")
    print(f"  Model:    {model_name}")
    print(f"  Engine:   vLLM")
    print(f"  DType:    {dtype}")
    print("  Input:    ~320 tokens (Bench360 standard)")
    print(f"  Output:   {max_tokens} tokens (Bench360: 32)")
    print(f"  Batches:  {batch_sizes}")
    print(f"  GPU:      {gpu_name} ({vram_gb:.0f}GB)")
    print(f"  Warmup:   {num_warmup} runs")
    if shape_warmup_runs > 0:
        print(f"  Shape warmup: {shape_warmup_runs} untimed runs per batch")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    padded_prompts, prompt_lens = _prepare_prompts(
        tokenizer,
        PROMPTS_320,
        TARGET_INPUT_TOKENS,
    )
    avg_in = sum(prompt_lens) / len(prompt_lens)
    print(f"   Input tokens: min={min(prompt_lens)}, avg={avg_in:.0f}, max={max(prompt_lens)}")
    print(f"   Output tokens: {max_tokens} (target)")
    print(f"   Total context: ~{avg_in:.0f} + {max_tokens} = ~{avg_in + max_tokens:.0f}")

    bench_max_batch = max(batch_sizes) if batch_sizes else 128
    if max_model_len is None:
        max_model_len = TARGET_INPUT_TOKENS + max_tokens + 32

    llm_kwargs = dict(
        model=model_name,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=bench_max_batch,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        trust_remote_code=trust_remote_code,
        enable_prefix_caching=enable_prefix_caching,
    )
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

    print("\n📦 Loading vLLM engine...")
    t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_time = time.perf_counter() - t0
    print(f"   Loaded in {load_time:.1f}s")

    sampling_kwargs = dict(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    try:
        sampling_params = SamplingParams(min_tokens=max_tokens, **sampling_kwargs)
    except TypeError:
        sampling_params = SamplingParams(**sampling_kwargs)

    print(f"   vLLM args: max_model_len={max_model_len}, max_num_seqs={bench_max_batch}, "
          f"gpu_mem_util={gpu_memory_utilization:.2f}, "
          f"max_num_batched_tokens={max_num_batched_tokens if max_num_batched_tokens is not None else 'auto'}, "
          f"prefix_cache={int(enable_prefix_caching)}, eager={int(enforce_eager)}")

    print(f"   🔥 Warmup ({num_warmup} runs)...")
    warm_conv = _build_conversations(["warmup text for compilation"])
    for _ in range(num_warmup):
        _ = llm.chat(warm_conv, sampling_params=sampling_params, use_tqdm=False)
    print("   ✅ Ready!")

    results = {}
    for bs in batch_sizes:
        prompts = (padded_prompts * ((bs // len(padded_prompts)) + 1))[:bs]
        conversations = _build_conversations(prompts)
        total_in_tokens = sum(prompt_lens[i % len(prompt_lens)] for i in range(bs))

        print(f"\n{'─'*70}")
        print(f"  🚀 Batch={bs}")
        print(f"{'─'*70}")

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if shape_warmup_runs > 0:
                print(f"   🔁 Shape warmup ({shape_warmup_runs} runs)...")
                for _ in range(shape_warmup_runs):
                    _ = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_start

            total_out_tokens = _count_output_tokens(tokenizer, outputs)
            output_tps = total_out_tokens / elapsed
            prefill_tps = total_in_tokens / elapsed
            combined_tps = (total_in_tokens + total_out_tokens) / elapsed

            results[bs] = {
                "ok": True,
                "time": elapsed,
                "tps_output": output_tps,
                "tps_prefill": prefill_tps,
                "tps_combined": combined_tps,
                "total_in": total_in_tokens,
                "total_out": total_out_tokens,
            }

            print(f"   ⏱️  Time:        {elapsed:.2f}s")
            print(f"   📊 Output TPS:  {output_tps:.1f} tok/s  (Bench360 metric)")
            print(f"   📊 Prefill TPS: {prefill_tps:.1f} tok/s")
            print(f"   📊 Combined:    {combined_tps:.1f} tok/s")
            print(f"   📊 Tokens:      {total_in_tokens} in + {total_out_tokens} out")
            sample = outputs[0].outputs[0].text.strip().replace("\n", " ")
            print(f"   Sample: {sample[:120]}...")
        except Exception as exc:
            results[bs] = {"ok": False, "error": str(exc)}
            print(f"   ❌ FAILED: {exc}")

    print(f"\n{'='*70}")
    print("  📊 BENCH360-COMPATIBLE RESULTS — vLLM")
    print(f"{'='*70}")
    print(f"  Model: {model_name} | GPU: {gpu_name} | {dtype}")
    print(f"  Input: ~{avg_in:.0f} tokens | Output: {max_tokens} tokens")
    print(f"{'─'*70}")
    print(f"  {'Batch':>5} │ {'Time':>7} │ {'Output TPS':>11} │ {'Prefill TPS':>12} │ {'Combined':>10}")
    print(f"  {'─────'}┼{'─────────'}┼{'─────────────'}┼{'──────────────'}┼{'───────────'}")

    peak_tps = 0.0
    peak_bs = 0
    for bs in batch_sizes:
        r = results.get(bs, {})
        if not r.get("ok", False):
            print(f"  {bs:>5} │     ERR │           — │            — │          —")
            continue
        marker = ""
        if r["tps_output"] > peak_tps:
            peak_tps = r["tps_output"]
            peak_bs = bs
            marker = " 🏆"
        print(
            f"  {bs:>5} │ {r['time']:>6.1f}s │ {r['tps_output']:>9.1f}  │ "
            f"{r['tps_prefill']:>10.1f}  │ {r['tps_combined']:>8.1f}{marker}"
        )

    if peak_bs > 0:
        print(f"\n  🏆 Peak at batch={peak_bs}: {peak_tps:.1f} output TPS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bench360-Compatible vLLM benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--shape-warmup-runs", type=int, default=0)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = [s for s in [16, 32, 64, 128, 256] if s <= args.max_batch]
    run_bench360_vllm(
        model_name=args.model,
        max_tokens=args.max_tokens,
        batch_sizes=sizes,
        num_warmup=args.warmup,
        shape_warmup_runs=args.shape_warmup_runs,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.enable_prefix_caching,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()

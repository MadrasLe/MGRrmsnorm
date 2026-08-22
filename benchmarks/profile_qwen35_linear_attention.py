"""
Profile Qwen 3.5 prefill hot path on MegaGemm.

Focus:
  - linear attention prefill
  - conv1d / chunk-local attention / recurrent state updates

Usage:
    python benchmarks/profile_qwen35_linear_attention.py --model Qwen/Qwen3.5-0.8B --prompt-length 128
"""

import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from megagemm.engine import InferenceEngine


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


def format_prompt(tokenizer, prompt: str) -> str:
    try:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def build_target_prompt(tokenizer, target_tokens: int) -> str:
    seed = "Linear attention improves asymptotic scaling in autoregressive transformers. "
    text = seed
    while True:
        formatted = format_prompt(tokenizer, text)
        tok_count = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
        if tok_count >= target_tokens:
            return text
        text += seed


def main():
    parser = argparse.ArgumentParser(description="Profile Qwen 3.5 linear attention prefill on MegaGemm")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=30)
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    clear_gpu()

    print("Profiling Qwen3.5 prefill")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Prompt length target: {args.prompt_length}")

    engine = InferenceEngine(
        args.model,
        dtype=dtype,
        device=args.device,
        num_blocks=args.num_blocks,
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
    )

    prompt = build_target_prompt(engine.tokenizer, args.prompt_length)
    formatted = format_prompt(engine.tokenizer, prompt)
    input_ids = engine.tokenizer(
        formatted, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(engine.device)
    print(f"Actual prompt tokens: {input_ids.shape[1]}")
    positions = torch.arange(input_ids.shape[1], device=engine.device).unsqueeze(0)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.inference_mode():
        seq_id = engine._next_seq_id()
        engine.block_manager.allocate_sequence(seq_id, input_ids.shape[1] + 8)
        _ = engine.model.prefill(input_ids, positions, engine.block_manager, seq_id)
        sync_if_cuda(args.device)
        engine.block_manager.free_sequence(seq_id)

        clear_gpu()
        seq_id = engine._next_seq_id()
        engine.block_manager.allocate_sequence(seq_id, input_ids.shape[1] + 8)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            _ = engine.model.prefill(input_ids, positions, engine.block_manager, seq_id)
            sync_if_cuda(args.device)

    engine.block_manager.free_sequence(seq_id)

    print("\nTop ops by self CUDA time")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.topk))

    print("\nTop ops by self CPU time")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.topk))


if __name__ == "__main__":
    main()

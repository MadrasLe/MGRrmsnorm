"""
🏁 MegaGemm Throughput Benchmark — Batching vs Sequential
-----------------------------------------------------------
Shows the real advantage of continuous batching:
- MegaGemm: batch=1, 2, 4, 8 (concurrent requests via scheduler)
- HuggingFace: sequential (model.generate one at a time)

MegaGemm's paged attention + continuous batching processes multiple
sequences in a single decode step → near-linear throughput scaling.

Usage (Colab):
  !python benchmark_throughput.py
"""

import torch
import gc
import sys
import os
import time
from typing import List, Optional

sys.path.insert(0, os.getcwd())

# ── Config ──
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
MODEL_LABEL = "Qwen-7B AWQ"
MAX_NEW_TOKENS = 64
WARMUP = True

# Multiple prompts for batching (all different to simulate real serving)
ALL_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list.",
    "What are the main causes of climate change?",
    "Describe how a neural network learns.",
    "What is the difference between TCP and UDP?",
    "Explain the theory of relativity briefly.",
    "Write a haiku about programming.",
]


def format_prompt(prompt, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return prompt


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ════════════════════════════════════════
# MegaGemm — Sequential (single)
# ════════════════════════════════════════
def bench_megagemm_sequential(engine, tokenizer, prompts):
    """Run prompts one at a time (no batching)."""
    formatted = [format_prompt(p, tokenizer) for p in prompts]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    total_tokens = 0
    for fmt in formatted:
        output = engine.generate(
            fmt, max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7, top_k=50, top_p=0.9,
        )
        total_tokens += len(tokenizer.encode(output))

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tps = total_tokens / elapsed if elapsed > 0 else 0
    return total_tokens, elapsed, tps


# ════════════════════════════════════════
# MegaGemm — Batched (concurrent)
# ════════════════════════════════════════
def bench_megagemm_batched(engine, tokenizer, prompts):
    """Run all prompts concurrently with continuous batching."""
    formatted = [format_prompt(p, tokenizer) for p in prompts]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    outputs = engine.generate_batch(
        formatted, max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7, top_k=50, top_p=0.9,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(tokenizer.encode(o)) for o in outputs)
    tps = total_tokens / elapsed if elapsed > 0 else 0
    return total_tokens, elapsed, tps


# ════════════════════════════════════════
# HuggingFace — Sequential only
# ════════════════════════════════════════
def bench_hf_sequential(model, tokenizer, prompts):
    """HuggingFace: one prompt at a time (its only real mode)."""
    formatted = [format_prompt(p, tokenizer) for p in prompts]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    total_tokens = 0
    for fmt in formatted:
        inputs = tokenizer(fmt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=0.7,
                top_k=50, top_p=0.9,
            )
        total_tokens += len(output_ids[0]) - input_len

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tps = total_tokens / elapsed if elapsed > 0 else 0
    return total_tokens, elapsed, tps


# ════════════════════════════════════════
# Main
# ════════════════════════════════════════
if __name__ == "__main__":
    print("🏁" + "=" * 58)
    print("  Throughput Benchmark: Batching vs Sequential")
    print("=" * 60)

    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "No GPU"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu} ({vram_gb:.1f}GB)")
    print(f"Model: {MODEL_NAME}")
    print(f"Max tokens/prompt: {MAX_NEW_TOKENS}")
    print(f"Total prompts available: {len(ALL_PROMPTS)}")

    results = []  # (label, n_prompts, total_tok, time_s, tps)

    # ── Phase 1: MegaGemm ──
    print(f"\n{'='*60}")
    print("🔥 MegaGemm — Loading model...")
    print(f"{'='*60}")

    from megagemm.engine import InferenceEngine
    from transformers import AutoTokenizer

    engine = InferenceEngine(MODEL_NAME, num_blocks=512, block_size=16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    mg_vram = torch.cuda.memory_allocated() / 1024**2

    # Warmup
    if WARMUP:
        print("  Warming up...")
        fmt = format_prompt("Hi", tokenizer)
        engine.generate(fmt, max_new_tokens=10, temperature=0.7)

    # Test batch sizes: 1, 2, 4, 8
    batch_sizes = [1, 2, 4, 8]

    for bs in batch_sizes:
        prompts = ALL_PROMPTS[:bs]
        label = f"MegaGemm seq" if bs == 1 else f"MegaGemm batch={bs}"

        if bs == 1:
            tok, t, tps = bench_megagemm_sequential(engine, tokenizer, prompts)
        else:
            tok, t, tps = bench_megagemm_batched(engine, tokenizer, prompts)

        results.append((label, bs, tok, t, tps))
        print(f"  {label:<25} → {tps:>6.1f} tok/s  ({tok} tok in {t:.1f}s)")

    del engine
    cleanup()

    # ── Phase 2: HuggingFace ──
    print(f"\n{'='*60}")
    print("🤗 HuggingFace — Loading model...")
    print(f"{'='*60}")

    try:
        from awq import AutoAWQForCausalLM
        hf_model = AutoAWQForCausalLM.from_quantized(
            MODEL_NAME, fuse_layers=True, device_map="auto",
        ).model
        print("  (loaded via AutoAWQ)")
    except Exception:
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float16, device_map="auto",
        )
        print("  (loaded via transformers)")

    hf_vram = torch.cuda.memory_allocated() / 1024**2

    if WARMUP:
        print("  Warming up...")
        inputs = tokenizer("Hi", return_tensors="pt").to(hf_model.device)
        with torch.inference_mode():
            hf_model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.7)

    # HF can only do sequential — test with same prompt counts
    for bs in batch_sizes:
        prompts = ALL_PROMPTS[:bs]
        label = f"HuggingFace seq ×{bs}"

        tok, t, tps = bench_hf_sequential(hf_model, tokenizer, prompts)
        results.append((label, bs, tok, t, tps))
        print(f"  {label:<25} → {tps:>6.1f} tok/s  ({tok} tok in {t:.1f}s)")

    del hf_model
    cleanup()

    # ── Final Report ──
    print(f"\n{'='*60}")
    print("📊 THROUGHPUT RESULTS")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME} | GPU: {gpu}")
    print(f"Max tokens/prompt: {MAX_NEW_TOKENS}")
    print()
    print(f"{'Configuration':<25} {'Prompts':>7} {'Tokens':>7} {'Time':>7} {'Throughput':>10}")
    print("-" * 60)

    # Print MegaGemm results
    mg_results = [(l, n, tok, t, tps) for l, n, tok, t, tps in results if "MegaGemm" in l]
    hf_results = [(l, n, tok, t, tps) for l, n, tok, t, tps in results if "HuggingFace" in l]

    print("  🔥 MegaGemm (continuous batching)")
    for label, n, tok, t, tps in mg_results:
        print(f"  {label:<23} {n:>7} {tok:>7} {t:>6.1f}s {tps:>8.1f} t/s")

    print()
    print("  🤗 HuggingFace (sequential only)")
    for label, n, tok, t, tps in hf_results:
        print(f"  {label:<23} {n:>7} {tok:>7} {t:>6.1f}s {tps:>8.1f} t/s")

    # Comparison at each batch size
    print(f"\n{'='*60}")
    print("📈 SCALING COMPARISON")
    print(f"{'='*60}")
    print(f"{'Prompts':>7} {'MegaGemm':>12} {'HuggingFace':>12} {'Winner':>15}")
    print("-" * 50)

    for i, bs in enumerate(batch_sizes):
        mg_tps = mg_results[i][4] if i < len(mg_results) else 0
        hf_tps = hf_results[i][4] if i < len(hf_results) else 0

        if mg_tps > hf_tps:
            ratio = mg_tps / hf_tps if hf_tps > 0 else 0
            winner = f"🔥 MG {ratio:.1f}x"
        else:
            ratio = hf_tps / mg_tps if mg_tps > 0 else 0
            winner = f"🤗 HF {ratio:.1f}x"

        print(f"  {bs:>5} {mg_tps:>10.1f}t/s {hf_tps:>10.1f}t/s {winner:>13}")

    # Scaling factor
    if len(mg_results) >= 2:
        mg_1 = mg_results[0][4]
        mg_8 = mg_results[-1][4]
        scaling = mg_8 / mg_1 if mg_1 > 0 else 0
        print(f"\n  🔥 MegaGemm scaling: {mg_1:.1f} → {mg_8:.1f} tok/s "
              f"({scaling:.1f}x with {batch_sizes[-1]}x prompts)")

    if len(hf_results) >= 2:
        hf_1 = hf_results[0][4]
        hf_8 = hf_results[-1][4]
        scaling = hf_8 / hf_1 if hf_1 > 0 else 0
        print(f"  🤗 HuggingFace scaling: {hf_1:.1f} → {hf_8:.1f} tok/s "
              f"({scaling:.1f}x with {batch_sizes[-1]}x prompts)")

    print(f"\n✅ Benchmark complete!")

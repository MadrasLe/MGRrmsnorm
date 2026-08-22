"""
🔢 MegaGemm AWQ Quantized Benchmark
======================================
Tests AWQ INT4 quantized models vs FP16 baseline.

Benchmark:
  1. TinyLlama 1.1B FP16 (baseline)
  2. Qwen2.5-7B-Instruct-AWQ (quantized, with chat template)
  3. Sequential + Batched throughput comparison

Run on Colab L4:
    %cd /content/drive/MyDrive/MGRrmsnorm
    !python setup.py build_ext --inplace 2>&1 | tail -5
    !pip install huggingface_hub safetensors transformers autoawq
    !python benchmark_quantized.py
"""

import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# =============================================================================
# Config
# =============================================================================
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.0
BATCH_SIZE = 16

# User questions (will be wrapped in chat template for instruct models)
QUESTIONS = [
    "What is the capital of France?",
    "Explain backpropagation in one sentence.",
    "What does the theory of general relativity state?",
    "Why is Python popular for machine learning?",
    "What are the first 10 Fibonacci numbers?",
    "How are neural networks inspired by the brain?",
    "What is the speed of light in vacuum?",
    "How does quantum computing differ from classical computing?",
]

# Raw completion prompts (for base/chat models that handle raw text)
RAW_PROMPTS = [
    "The capital of France is",
    "Neural networks are inspired by",
    "The theory of general relativity states that",
    "Python is a programming language known for",
    "The Fibonacci sequence starts with",
    "Neural networks are inspired by",
    "The speed of light in vacuum is",
    "Quantum computing differs from classical computing because",
]

# AWQ models to test
AWQ_MODELS = [
    ("Qwen/Qwen2.5-7B-Instruct-AWQ", "Qwen2.5-7B AWQ"),
]

# FP16 baseline
FP16_BASELINE = ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B FP16")


# =============================================================================
# Chat templates
# =============================================================================
def qwen2_chat(question: str) -> str:
    """Qwen2.5 Instruct chat template (ChatML format)."""
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# =============================================================================
# Benchmark helpers
# =============================================================================
from megagemm.engine import InferenceEngine


def run_benchmark(engine, label, prompts, display_prompts=None):
    """Run sequential + batched benchmark, return results dict."""
    if display_prompts is None:
        display_prompts = prompts

    # Warmup
    _ = engine.generate("Hello", max_new_tokens=5, temperature=0.0)

    # Sequential
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    seq_r = []
    for p in prompts:
        seq_r.append(engine.generate(p, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE))
    torch.cuda.synchronize()
    seq_time = time.perf_counter() - t0
    seq_toks = sum(len(engine.tokenizer.encode(r)) for r in seq_r)
    seq_tps = seq_toks / seq_time

    # Batched
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batch_r = engine.generate_batch(prompts, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
    torch.cuda.synchronize()
    batch_time = time.perf_counter() - t0
    batch_toks = sum(len(engine.tokenizer.encode(r)) for r in batch_r)
    batch_tps = batch_toks / batch_time

    mem_gb = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\n  {label}:")
    print(f"    Sequential: {seq_tps:.1f} tok/s")
    print(f"    Batched:    {batch_tps:.1f} tok/s ({batch_tps/seq_tps:.1f}x)")
    print(f"    VRAM:       {mem_gb:.1f}GB")
    print(f"    Samples:")
    for i in range(min(2, len(batch_r))):
        q = display_prompts[i][:40]
        a = batch_r[i].strip()[:80]
        print(f"      Q: {q}")
        print(f"      A: {a}")
        print()

    return {
        'label': label,
        'seq_tps': seq_tps,
        'batch_tps': batch_tps,
        'mem_gb': mem_gb,
    }


# =============================================================================
# Run benchmarks
# =============================================================================
results = []

# FP16 baseline (raw prompts — TinyLlama handles them fine)
print("=" * 60)
print(f"📏 FP16 Baseline: {FP16_BASELINE[1]}")
print("=" * 60)
try:
    engine = InferenceEngine(FP16_BASELINE[0], dtype=torch.float16, max_batch_size=BATCH_SIZE)
    results.append(run_benchmark(engine, FP16_BASELINE[1], RAW_PROMPTS))
    del engine; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
except Exception as e:
    print(f"  ❌ Failed: {e}")

# AWQ models (with proper chat template)
for model_name, label in AWQ_MODELS:
    print(f"\n{'=' * 60}")
    print(f"🔢 AWQ Quantized: {label}")
    print("=" * 60)
    try:
        engine = InferenceEngine(
            model_name, dtype=torch.float16,
            max_batch_size=BATCH_SIZE,
            num_blocks=512,
        )
        # Format questions with Qwen2 chat template
        chat_prompts = [qwen2_chat(q) for q in QUESTIONS]
        results.append(run_benchmark(engine, label, chat_prompts, display_prompts=QUESTIONS))
        del engine; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback; traceback.print_exc()

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'=' * 60}")
print("📊 AWQ QUANTIZATION BENCHMARK")
print("=" * 60)
print(f"  {'Model':>25s} | {'Seq':>7s} | {'Batch':>7s} | {'VRAM':>5s}")
print(f"  {'-'*25}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}")
for r in results:
    print(f"  {r['label']:>25s} | {r['seq_tps']:>5.1f}/s | {r['batch_tps']:>5.1f}/s | {r['mem_gb']:>.1f}GB")

if len(results) >= 2:
    fp16 = results[0]
    awq = results[1]
    print(f"\n  💡 7B AWQ uses {awq['mem_gb']:.1f}GB VRAM")
    print(f"  💡 7B AWQ batch throughput: {awq['batch_tps']:.0f} tok/s")
    savings_pct = (1 - awq['mem_gb'] / 14.0) * 100
    print(f"  💡 VRAM savings vs FP16 7B: ~{savings_pct:.0f}%")

print("\nBenchmark complete! 🏁")

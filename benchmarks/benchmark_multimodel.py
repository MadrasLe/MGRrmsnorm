"""
🧠 MegaGemm Multi-Model Benchmark
====================================
Tests multiple model families side by side:
  - TinyLlama 1.1B (LLaMA baseline)
  - Qwen3-1.7B (QK-Norm)
  - Qwen2.5-1.5B (QKV-bias)
  - Gemma-2-2B (RMSNorm+1, GeGLU, logit cap) — needs HF_TOKEN

Run on Colab:
    %cd /content/drive/MyDrive/MGRrmsnorm
    !python setup.py build_ext --inplace 2>&1 | tail -5
    !pip install huggingface_hub safetensors transformers
    !python benchmark_multimodel.py
"""

import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# =============================================================================
# Config
# =============================================================================
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.0
BATCH_SIZE = 8

PROMPTS = [
    "The capital of France is",
    "In machine learning, backpropagation is",
    "The theory of general relativity states that",
    "Python is a programming language known for",
    "The Fibonacci sequence starts with",
    "Neural networks are inspired by",
    "The speed of light in vacuum is",
    "Quantum computing differs from classical computing because",
]

# Models to test (order: easiest → hardest)
MODELS = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B",  "llama"),
    ("Qwen/Qwen3-1.7B",                     "Qwen3 1.7B",      "qwen3"),
    ("Qwen/Qwen2.5-1.5B",                   "Qwen2.5 1.5B",    "qwen2"),
]

# Gemma needs HF_TOKEN — add if available
if os.environ.get("HF_TOKEN"):
    MODELS.append(
        ("google/gemma-2-2b", "Gemma2 2B", "gemma2")
    )
else:
    print("ℹ️  Set HF_TOKEN to also test Gemma 2 (gated model)\n")


# =============================================================================
# Benchmark
# =============================================================================
from megagemm.engine import InferenceEngine

results = []

for model_name, label, model_type in MODELS:
    print("=" * 60)
    print(f"🧪 Testing: {label} ({model_type})")
    print("=" * 60)

    try:
        # Load model
        engine = InferenceEngine(
            model_name,
            dtype=torch.float16,
            max_batch_size=BATCH_SIZE,
        )

        # Warmup
        _ = engine.generate("Hello", max_new_tokens=5, temperature=0.0)

        # --- Sequential (baseline) ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seq_results = []
        for p in PROMPTS:
            out = engine.generate(p, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
            seq_results.append(out)
        torch.cuda.synchronize()
        seq_time = time.perf_counter() - t0
        seq_tokens = sum(len(engine.tokenizer.encode(r)) for r in seq_results)
        seq_tps = seq_tokens / seq_time

        # --- Batched ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch_results = engine.generate_batch(
            PROMPTS, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
        )
        torch.cuda.synchronize()
        batch_time = time.perf_counter() - t0
        batch_tokens = sum(len(engine.tokenizer.encode(r)) for r in batch_results)
        batch_tps = batch_tokens / batch_time

        speedup = batch_tps / seq_tps
        param_mb = sum(p.nelement() * p.element_size() for p in engine.model.parameters()) / (1024**2)
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3

        results.append({
            'label': label,
            'type': model_type,
            'params_mb': param_mb,
            'seq_tps': seq_tps,
            'batch_tps': batch_tps,
            'speedup': speedup,
            'mem_gb': mem_gb,
        })

        # Show sample outputs
        print(f"\n  Sequential: {seq_tps:.1f} tok/s")
        print(f"  Batched:    {batch_tps:.1f} tok/s ({speedup:.1f}x)")
        print(f"  Memory:     {mem_gb:.1f}GB")
        print(f"\n  Sample outputs:")
        for i in range(min(3, len(batch_results))):
            print(f"    [{i}] '{PROMPTS[i][:25]}...' → '{batch_results[i][:60]}...'")

        # Cleanup
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print()

    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        results.append({
            'label': label, 'type': model_type,
            'params_mb': 0, 'seq_tps': 0, 'batch_tps': 0,
            'speedup': 0, 'mem_gb': 0, 'error': str(e),
        })
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# Summary
# =============================================================================
print("=" * 60)
print("📊 MULTI-MODEL COMPARISON")
print("=" * 60)
print(f"  {'Model':>20s} | {'Type':>6s} | {'Params':>7s} | {'Seq':>7s} | {'Batch':>7s} | {'Speed':>6s} | {'VRAM':>5s}")
print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}")

for r in results:
    if r.get('error'):
        print(f"  {r['label']:>20s} | {r['type']:>6s} | {'FAILED':>7s} | {'-':>7s} | {'-':>7s} | {'-':>6s} | {'-':>5s}")
    else:
        print(f"  {r['label']:>20s} | {r['type']:>6s} | {r['params_mb']:.0f}MB | "
              f"{r['seq_tps']:>5.1f}/s | {r['batch_tps']:>5.1f}/s | "
              f"{r['speedup']:>5.1f}x | {r['mem_gb']:>.1f}GB")

if results:
    best = max((r for r in results if not r.get('error')), key=lambda x: x['batch_tps'], default=None)
    if best:
        print(f"\n  🏆 Fastest: {best['label']} at {best['batch_tps']:.0f} tok/s (batch={BATCH_SIZE})")

print("\nBenchmark complete! 🏁")

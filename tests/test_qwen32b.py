"""
🧪 Qwen2.5-32B AWQ Test on L4
-------------------------------
Tests the largest model that should fit entirely in L4 VRAM (~16GB AWQ).
Also tests with partial offload for comparison.

Model: Qwen/Qwen2.5-32B-Instruct-AWQ
  - 64 layers, hidden=5120, 40 heads, GQA=8x
  - ~16GB AWQ INT4
  - L4 has 22GB → should fit with ~6GB for KV cache

Usage (Colab):
  !python test_qwen32b.py
"""

import torch
import gc
import sys
import os
import time
import traceback

sys.path.insert(0, os.getcwd())

MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function that finds prime numbers up to N.",
]

def run_test(n_gpu_layers=-1, num_blocks=128, max_tokens=100):
    """Test Qwen2.5-32B AWQ with given config."""
    mode = "full GPU" if n_gpu_layers < 0 else f"offload ({n_gpu_layers} GPU layers)"

    print(f"\n{'='*60}")
    print(f"Qwen2.5-32B AWQ — {mode}")
    print(f"{'='*60}")

    try:
        from megagemm.engine import InferenceEngine
        from transformers import AutoTokenizer

        t0 = time.perf_counter()
        engine = InferenceEngine(
            MODEL,
            num_blocks=num_blocks,
            block_size=16,
            n_gpu_layers=n_gpu_layers,
        )
        load_time = time.perf_counter() - t0

        cfg = engine.model.config
        print(f"  Layers: {cfg.num_hidden_layers}, Hidden: {cfg.hidden_size}")
        print(f"  Load time: {load_time:.1f}s")

        vram_mb = torch.cuda.memory_allocated() / 1024**2
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  VRAM: {vram_mb:.0f}MB / {vram_total:.0f}MB")

        offloader = engine.model._offloader
        if offloader:
            print(f"  Offloader: {offloader}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        total_tokens = 0
        total_time = 0

        for prompt in PROMPTS:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            t0 = time.perf_counter()
            output = engine.generate(
                formatted,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            gen_time = time.perf_counter() - t0

            n_tokens = len(tokenizer.encode(output))
            tok_s = n_tokens / gen_time if gen_time > 0 else 0
            total_tokens += n_tokens
            total_time += gen_time

            print(f"\n  Q: {prompt}")
            print(f"  A: {output[:300]}")
            print(f"  ⏱ {gen_time:.2f}s, ~{tok_s:.1f} tok/s, {n_tokens} tokens")

        avg_toks = total_tokens / total_time if total_time > 0 else 0
        print(f"\n  📊 Average: {avg_toks:.1f} tok/s over {total_tokens} tokens")

        if offloader:
            offloader.print_stats()

        del engine
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n  ✅ PASS")
        return True

    except Exception as e:
        print(f"\n  ❌ ERROR: {e}")
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen2.5-32B AWQ Test")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name()
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu} ({vram:.1f}GB)")
    else:
        print("No GPU!"); sys.exit(1)

    # Test 1: Full GPU (should fit in L4's 22GB)
    # KV cache: 64 layers × big → keep num_blocks small
    ok1 = run_test(n_gpu_layers=-1, num_blocks=128, max_tokens=100)

    gc.collect(); torch.cuda.empty_cache()

    # Test 2: Partial offload (32 of 64 layers on CPU) — compare speed
    ok2 = run_test(n_gpu_layers=32, num_blocks=128, max_tokens=100)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  {'✅' if ok1 else '❌'} Full GPU")
    print(f"  {'✅' if ok2 else '❌'} Partial offload (32 GPU / 32 CPU)")

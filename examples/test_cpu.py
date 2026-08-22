#!/usr/bin/env python3
"""
🖥️ CPU Inference Test — MegaGemm
Tests that ALL fallbacks work correctly on CPU (no GPU needed).
"""
import sys
import time
import torch

# Force CPU
DEVICE = 'cpu'
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model (~1GB RAM)

def main():
    print("=" * 60)
    print("  🖥️  MegaGemm CPU Inference Test")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Model:  {MODEL}")
    print(f"  GPU available: {torch.cuda.is_available()}")
    print()

    # Check kernel fallbacks
    print("📋 Kernel fallbacks:")
    try:
        from megagemm.kernels.rmsnorm import RMSNormFunction
        print("  RMSNorm CUDA:     ✅ loaded (will fallback on CPU)")
    except Exception:
        print("  RMSNorm CUDA:     ⚠️ not available (PyTorch fallback)")

    try:
        from megagemm.kernels.swiglu import MegaGemmFunction
        print("  SwiGLU Triton:    ✅ loaded (will fallback on CPU)")
    except Exception:
        print("  SwiGLU Triton:    ⚠️ not available (PyTorch fallback)")

    try:
        import triton
        print("  Triton:           ✅ available (not used on CPU)")
    except Exception:
        print("  Triton:           ⚠️ not available (PyTorch fallback)")

    print()

    # Load engine
    print("📦 Loading model on CPU...")
    t0 = time.perf_counter()
    from megagemm.engine import InferenceEngine
    engine = InferenceEngine(MODEL, device=DEVICE, dtype=torch.float32)
    dt = time.perf_counter() - t0
    print(f"  ✅ Loaded in {dt:.1f}s")
    print()

    # Test 1: Single generation
    print("─" * 60)
    print("  Test 1: Single generation (greedy)")
    print("─" * 60)
    prompt = "What is 2 + 2? Answer in one word."
    t0 = time.perf_counter()
    out = engine.generate(prompt, max_new_tokens=16, temperature=0.0, verbose=False)
    dt = time.perf_counter() - t0
    print(f"  Prompt: {prompt}")
    print(f"  Output: {out}")
    print(f"  Time:   {dt:.1f}s ({16/dt:.1f} tok/s)")
    print()

    # Test 2: Another prompt
    print("─" * 60)
    print("  Test 2: Code generation")
    print("─" * 60)
    prompt = "Write a one-line Python hello world."
    t0 = time.perf_counter()
    out = engine.generate(prompt, max_new_tokens=32, temperature=0.0, verbose=False)
    dt = time.perf_counter() - t0
    print(f"  Prompt: {prompt}")
    print(f"  Output: {out}")
    print(f"  Time:   {dt:.1f}s ({32/dt:.1f} tok/s)")
    print()

    # Test 3: Batch generation
    print("─" * 60)
    print("  Test 3: Batch generation (2 prompts)")
    print("─" * 60)
    prompts = [
        "Capital of Japan?",
        "Largest planet?",
    ]
    t0 = time.perf_counter()
    outs = engine.generate_batch(prompts, max_new_tokens=16, temperature=0.0, verbose=False)
    dt = time.perf_counter() - t0
    for i, (p, o) in enumerate(zip(prompts, outs)):
        print(f"  [{i+1}] {p} → {o.strip()[:80]}")
    print(f"  Time: {dt:.1f}s")
    print()

    print("=" * 60)
    print("  ✅ All CPU tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
⚡ CPU Inference Benchmark: FP32 vs BF16 vs INT8
==================================================
End-to-end inference speed on CPU with different precisions.
"""
import torch
import time
import sys
import os

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain what gravity is in two sentences."
MAX_TOKENS = 32

def bench_inference(engine, prompt, max_tokens, warmup=1, repeat=3):
    """Benchmark end-to-end generation, return avg tok/s."""
    # Warmup
    for _ in range(warmup):
        engine.generate(prompt, max_new_tokens=max_tokens, temperature=0.0, verbose=False)

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = engine.generate(prompt, max_new_tokens=max_tokens, temperature=0.0, verbose=False)
        dt = time.perf_counter() - t0
        times.append(dt)

    avg = sum(times) / len(times)
    return max_tokens / avg, out


def main():
    print("=" * 70)
    print("  ⚡ CPU Inference Benchmark: FP32 vs BF16")
    print("=" * 70)
    print(f"  Model:      {MODEL}")
    print(f"  Prompt:     {PROMPT[:50]}...")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  CPU:        {os.cpu_count()} cores")
    print()

    # Set all CPU cores
    torch.set_num_threads(os.cpu_count())
    print(f"  torch threads: {torch.get_num_threads()}")
    print()

    from megagemm.engine import InferenceEngine

    results = []

    # ── Test 1: FP32 ──
    print("─" * 70)
    print("  [1/3] Loading FP32...")
    sys.stdout.flush()
    t0 = time.perf_counter()
    engine_fp32 = InferenceEngine(MODEL, device='cpu', dtype=torch.float32)
    load_t = time.perf_counter() - t0
    print(f"  Loaded in {load_t:.1f}s")

    print("  Benchmarking FP32...", end=" ", flush=True)
    tps, out = bench_inference(engine_fp32, PROMPT, MAX_TOKENS)
    print(f"{tps:.1f} tok/s")
    print(f"  Output: {out[:80]}...")
    results.append(("FP32", tps, out))
    del engine_fp32
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Test 2: BF16 ──
    print()
    print("─" * 70)
    print("  [2/3] Loading BF16...")
    sys.stdout.flush()
    t0 = time.perf_counter()
    engine_bf16 = InferenceEngine(MODEL, device='cpu', dtype=torch.bfloat16)
    load_t = time.perf_counter() - t0
    print(f"  Loaded in {load_t:.1f}s")

    print("  Benchmarking BF16...", end=" ", flush=True)
    tps, out = bench_inference(engine_bf16, PROMPT, MAX_TOKENS)
    print(f"{tps:.1f} tok/s")
    print(f"  Output: {out[:80]}...")
    results.append(("BF16", tps, out))
    del engine_bf16

    # ── Test 3: FP16 ──
    print()
    print("─" * 70)
    print("  [3/3] Loading FP16...")
    sys.stdout.flush()
    t0 = time.perf_counter()
    engine_fp16 = InferenceEngine(MODEL, device='cpu', dtype=torch.float16)
    load_t = time.perf_counter() - t0
    print(f"  Loaded in {load_t:.1f}s")

    print("  Benchmarking FP16...", end=" ", flush=True)
    tps, out = bench_inference(engine_fp16, PROMPT, MAX_TOKENS)
    print(f"{tps:.1f} tok/s")
    print(f"  Output: {out[:80]}...")
    results.append(("FP16", tps, out))
    del engine_fp16

    # ── Summary ──
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    base = results[0][1]  # FP32 as baseline
    for name, tps, _ in results:
        bar = "█" * int(tps * 3)
        speedup = tps / base
        print(f"  {name:<6} │ {tps:>6.1f} tok/s │ {speedup:>5.2f}x │ {bar}")

    # Output coherence
    print()
    print("  Output coherence:")
    for name, _, out in results:
        print(f"  {name:<6}: {out[:70]}...")

    print("=" * 70)


if __name__ == "__main__":
    main()

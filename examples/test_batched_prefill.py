"""
⚡ Batched Prefill Benchmark for MegaGemm
==========================================
Compares sequential prefill (1 request per forward) vs batched prefill
(N requests per forward). Critical for text classification throughput.

Usage (Kaggle/Colab):
    python examples/test_batched_prefill.py

    # Larger batch
    python examples/test_batched_prefill.py --batch-size 16

    # With specific model
    python examples/test_batched_prefill.py --model Qwen/Qwen2.5-7B-Instruct

Author: Gabriel Yogi
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def run_benchmark(model_name: str, batch_size: int = 8, max_tokens: int = 5, quantize: str = None):
    """Benchmark sequential vs batched prefill."""
    import torch
    from megagemm.engine import InferenceEngine

    print(f"\n{'='*70}")
    print(f"  Batched Prefill Benchmark")
    print(f"  Model: {model_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max tokens: {max_tokens}")
    print(f"{'='*70}")

    # --- Build prompts (varying lengths to test padding) ---
    base_prompts = [
        "Classify the sentiment: I love this product!",
        "Classify the sentiment: This is terrible, I hate it.",
        "Classify the sentiment: The movie was okay, nothing special.",
        "Classify the sentiment: Best purchase I ever made, highly recommend!",
        "Classify the sentiment: Not worth the price, very disappointing.",
        "Classify the sentiment: It works fine.",
        "Classify the sentiment: Absolutely incredible experience, would do again.",
        "Classify the sentiment: Waste of money, broke after one day.",
        "Classify the sentiment: Pretty good overall, minor issues.",
        "Classify the sentiment: The quality exceeded my expectations.",
        "Classify the sentiment: Arrived damaged and customer service unhelpful.",
        "Classify the sentiment: Average product, does what it says.",
        "Classify the sentiment: Five stars, amazing quality and fast shipping!",
        "Classify the sentiment: One star, the worst thing I've ever bought.",
        "Classify the sentiment: Decent for the price, no complaints.",
        "Classify the sentiment: Perfect gift for my friend, she loved it!",
    ]
    prompts = (base_prompts * ((batch_size // len(base_prompts)) + 1))[:batch_size]

    # --- Load engine ---
    print("\n📦 Loading model...")
    t0 = time.perf_counter()
    engine = InferenceEngine(model_name, quantize=quantize, monitor=True)
    t_load = time.perf_counter() - t0
    print(f"   Loaded in {t_load:.1f}s")

    # --- TEST 1: Sequential prefill (baseline) ---
    print(f"\n{'─'*70}")
    print(f"  🐢 Sequential Prefill ({batch_size} requests, 1 per forward)")
    print(f"{'─'*70}")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.perf_counter()

    sequential_results = []
    for prompt in prompts:
        text, _ = engine.generate(
            prompt, max_new_tokens=max_tokens, temperature=0.1, xai=True, verbose=False,
        )
        sequential_results.append(text)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_sequential = time.perf_counter() - t_start

    print(f"   Time: {t_sequential:.3f}s")
    print(f"   Rate: {batch_size / t_sequential:.1f} prompts/s")
    print(f"   Per prompt: {t_sequential / batch_size * 1000:.0f}ms")
    for i, (p, r) in enumerate(zip(prompts[:3], sequential_results[:3])):
        print(f"   [{i}] {p[:50]}... → {r[:40]}")

    # Reset monitor for fair comparison
    engine.reset_monitor()

    # --- TEST 2: Batched prefill (via generate_batch) ---
    print(f"\n{'─'*70}")
    print(f"  🚀 Batched Prefill ({batch_size} requests, all in 1 forward)")
    print(f"{'─'*70}")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.perf_counter()

    batch_results = engine.generate_batch(
        prompts, max_new_tokens=max_tokens, temperature=0.1, verbose=False,
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_batched = time.perf_counter() - t_start

    print(f"   Time: {t_batched:.3f}s")
    print(f"   Rate: {batch_size / t_batched:.1f} prompts/s")
    print(f"   Per prompt: {t_batched / batch_size * 1000:.0f}ms")
    for i, (p, r) in enumerate(zip(prompts[:3], batch_results[:3])):
        print(f"   [{i}] {p[:50]}... → {r[:40]}")

    # --- Comparison ---
    speedup = t_sequential / t_batched if t_batched > 0 else float('inf')

    print(f"\n{'='*70}")
    print(f"  📊 Results")
    print(f"{'='*70}")
    print(f"  Sequential: {t_sequential:.3f}s  ({batch_size / t_sequential:.1f} prompts/s)")
    print(f"  Batched:    {t_batched:.3f}s  ({batch_size / t_batched:.1f} prompts/s)")
    print(f"  Speedup:    {speedup:.2f}x {'🔥' if speedup > 1.5 else '🟡' if speedup > 1.0 else '🔴'}")

    # Extrapolate to texts/hour
    seq_per_hour = (batch_size / t_sequential) * 3600
    bat_per_hour = (batch_size / t_batched) * 3600
    print(f"\n  Extrapolated throughput:")
    print(f"    Sequential: {seq_per_hour:,.0f} texts/hour")
    print(f"    Batched:    {bat_per_hour:,.0f} texts/hour")
    print(f"    Gain:       +{bat_per_hour - seq_per_hour:,.0f} texts/hour")
    print(f"{'='*70}")

    # --- TEST 3: Equivalence validation (greedy) ---
    print(f"\n{'─'*70}")
    print(f"  🔬 Equivalence Test (greedy, temperature=0.0)")
    print(f"{'─'*70}")

    # Sequential greedy (repetition_penalty=1.0 to match generate_batch)
    greedy_seq = []
    for prompt in prompts:
        text = engine.generate(
            prompt, max_new_tokens=max_tokens, temperature=0.0,
            repetition_penalty=1.0, verbose=False,
        )
        # generate returns (text, report) when xai=True, just text otherwise
        if isinstance(text, tuple):
            text = text[0]
        greedy_seq.append(text)

    engine.reset_monitor()

    # Batched greedy
    greedy_bat = engine.generate_batch(
        prompts, max_new_tokens=max_tokens, temperature=0.0, verbose=False,
    )

    # Compare
    matches = 0
    for i, (s, b) in enumerate(zip(greedy_seq, greedy_bat)):
        match = s.strip() == b.strip()
        matches += int(match)
        status = "✅" if match else "❌"
        print(f"   [{i}] {status}")
        if not match:
            print(f"       seq: {repr(s[:60])}")
            print(f"       bat: {repr(b[:60])}")

    print(f"\n   Result: {matches}/{batch_size} identical")
    if matches == batch_size:
        print(f"   ✅ PERFECT EQUIVALENCE — batched prefill is bit-exact!")
    elif matches >= batch_size * 0.8:
        print(f"   🟡 MOSTLY EQUIVALENT — minor differences (floating point order)")
    else:
        print(f"   🔴 DIVERGENT — attention mask or padding issue needs debugging")
    print(f"{'='*70}")

    return speedup


def main():
    parser = argparse.ArgumentParser(description="MegaGemm Batched Prefill Benchmark")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--quantize", choices=["int8"], default=None)
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()

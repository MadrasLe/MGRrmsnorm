#!/usr/bin/env python3
"""
📊 Decode-Only Benchmark — MegaGemm
=====================================
Isolates DECODE throughput by using minimal prompts + long output.
Prefill is ~10 tokens → negligible. All measured time is decode.

Metrics:
  - TPOT: Time Per Output Token (ms) — single-stream latency
  - Decode TPS: total output tokens / decode time — batch throughput
  - Per-seq TPS: tokens/sec each sequence gets

Comparison targets (L4 GPU, 7B FP16, single-stream):
  - LMDeploy: ~90ms/tok TPOT (best)
  - vLLM:     ~100-110ms/tok
  - SGLang:   ~100-120ms/tok
  - TGI:      ~120-140ms/tok

Usage:
    python examples/bench_decode.py
    python examples/bench_decode.py --output-tokens 256 --max-batch 64
"""

import time
import argparse
import torch


# Minimal prompts (~10-15 tokens) with specific topics to force long output
TINY_PROMPTS = [
    "Write a detailed essay about the French Revolution.",
    "Explain how neural networks learn from data.",
    "Describe the complete lifecycle of a star.",
    "List all major causes of World War One.",
    "Explain how the human digestive system works.",
    "Write a guide on building a compiler.",
    "Analyze the history of the Roman Empire.",
    "Explain how quantum computers work step by step.",
]


def run_decode_benchmark(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    output_tokens: int = 512,
    batch_sizes: list = None,
    quantize: str = None,
):
    from megagemm.engine import InferenceEngine

    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    gpu_name = "CPU"
    vram_gb = 0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"\n{'='*70}")
    print(f"  📊 Decode-Only Benchmark — MegaGemm")
    print(f"{'='*70}")
    print(f"  Model:    {model_name}")
    print(f"  Quant:    {quantize or 'FP16'}")
    print(f"  Prompt:   ~10 tokens (minimal — decode dominates)")
    print(f"  Output:   {output_tokens} tokens")
    print(f"  Batches:  {batch_sizes}")
    print(f"  GPU:      {gpu_name} ({vram_gb:.0f}GB)")
    print(f"{'='*70}")

    # Load engine
    print(f"\n📦 Loading {model_name}...")
    t0 = time.perf_counter()
    engine = InferenceEngine(
        model_name,
        quantize=quantize,
        kv_offload=True,
    )
    tokenizer = engine.tokenizer
    load_time = time.perf_counter() - t0
    print(f"   Loaded in {load_time:.1f}s")

    # Warmup
    print(f"   🔥 Warmup...")
    for _ in range(3):
        engine.generate("warmup", max_new_tokens=10, temperature=0.0,
                        repetition_penalty=1.0, verbose=False)
    engine.generate_batch(["warmup batch"], max_new_tokens=10,
                          temperature=0.0, verbose=False)
    engine.reset_monitor()
    torch.cuda.synchronize()
    print(f"   ✅ Ready!")

    # ── Single-stream first (TPOT comparison) ──
    print(f"\n{'─'*70}")
    print(f"  ⏱️  SINGLE-STREAM (TPOT) — 1 request, {output_tokens} tokens")
    print(f"{'─'*70}")

    prompt = TINY_PROMPTS[0]
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    out = engine.generate(
        prompt, max_new_tokens=output_tokens,
        temperature=0.7, repetition_penalty=1.1, verbose=False,
    )
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    out_toks = len(tokenizer.encode(out))
    tpot_ms = (elapsed / out_toks) * 1000

    print(f"   Output tokens: {out_toks}")
    print(f"   Time:          {elapsed:.2f}s")
    print(f"   TPOT:          {tpot_ms:.1f} ms/token")
    print(f"   TPS:           {out_toks / elapsed:.1f} tok/s")
    print(f"   Sample:        {out[:80]}...")

    single_tpot = tpot_ms
    single_tps = out_toks / elapsed

    # ── Batch decode ──
    results = {}

    for bs in batch_sizes:
        prompts = (TINY_PROMPTS * ((bs // len(TINY_PROMPTS)) + 1))[:bs]

        print(f"\n{'─'*70}")
        print(f"  🚀 Batch={bs} — {bs}×{output_tokens} = {bs * output_tokens} decode tokens")
        print(f"{'─'*70}")

        try:
            torch.cuda.synchronize()
            engine.reset_monitor()

            t_start = time.perf_counter()
            outputs = engine.generate_batch(
                prompts,
                max_new_tokens=output_tokens,
                temperature=0.7,
                verbose=False,
            )
            torch.cuda.synchronize()
            t_end = time.perf_counter()

            elapsed = t_end - t_start

            # Count output tokens
            total_out = sum(len(tokenizer.encode(o)) for o in outputs)
            total_tps = total_out / elapsed
            per_seq_tps = total_tps / bs
            tpot = (elapsed / (total_out / bs)) * 1000  # ms per token per sequence

            results[bs] = {
                'ok': True,
                'time': elapsed,
                'total_out': total_out,
                'total_tps': total_tps,
                'per_seq_tps': per_seq_tps,
                'tpot_ms': tpot,
            }

            print(f"   ⏱️  Time:         {elapsed:.2f}s")
            print(f"   📊 Decode TPS:   {total_tps:.1f} tok/s (all seqs)")
            print(f"   📊 Per-seq TPS:  {per_seq_tps:.1f} tok/s")
            print(f"   📊 TPOT:         {tpot:.1f} ms/token")
            print(f"   📊 Tokens:       {total_out} output tokens")

        except Exception as e:
            results[bs] = {'ok': False, 'error': str(e)}
            print(f"   ❌ FAILED: {e}")

    # ── Results table ──
    print(f"\n{'='*70}")
    print(f"  📊 DECODE BENCHMARK RESULTS — MegaGemm")
    print(f"{'='*70}")
    print(f"  Model: {model_name} | GPU: {gpu_name} | {quantize or 'FP16'}")
    print(f"  Output: {output_tokens} tokens/req | Prompt: ~10 tokens")
    print(f"{'─'*70}")
    print(f"  {'Batch':>5} │ {'Time':>7} │ {'Decode TPS':>11} │ {'Per-seq':>9} │ {'TPOT':>10}")
    print(f"  {'─'*5}─┼─{'─'*7}─┼─{'─'*11}─┼─{'─'*9}─┼─{'─'*10}")

    # Single-stream result
    print(f"  {'1*':>5} │ {(single_tpot * output_tokens / 1000):>6.1f}s │ "
          f"{single_tps:>9.1f}  │ {single_tps:>7.1f}  │ {single_tpot:>7.1f} ms")

    peak_tps = single_tps
    peak_bs = 1
    for bs in batch_sizes:
        r = results.get(bs, {})
        if not r.get('ok', False):
            print(f"  {bs:>5} │     ERR │           — │         — │          —")
            continue
        marker = ""
        if r['total_tps'] > peak_tps:
            peak_tps = r['total_tps']
            peak_bs = bs
            marker = " 🏆"
        print(f"  {bs:>5} │ {r['time']:>6.1f}s │ "
              f"{r['total_tps']:>9.1f}  │ {r['per_seq_tps']:>7.1f}  │ "
              f"{r['tpot_ms']:>7.1f} ms{marker}")

    print(f"\n  🏆 Peak decode: {peak_tps:.0f} TPS at batch={peak_bs}")

    # ── Comparison ──
    print(f"\n{'─'*70}")
    print(f"  📊 COMPARISON — Single-Stream TPOT (ms/token)")
    print(f"{'─'*70}")
    print(f"  Bench360 L4 (7B FP16, single-stream):")
    print(f"    LMDeploy: ~90 ms/tok   (best)")
    print(f"    vLLM:     ~100 ms/tok")
    print(f"    SGLang:   ~110 ms/tok")
    print(f"    TGI:      ~130 ms/tok")
    print(f"    MegaGemm: {single_tpot:.0f} ms/tok  ← you are here")
    if single_tpot < 90:
        print(f"    ✅ FASTER than all Bench360 engines!")
    elif single_tpot < 110:
        print(f"    ✅ Competitive with top engines!")
    elif single_tpot < 140:
        print(f"    ⚠️  Mid-range, room for improvement")
    else:
        print(f"    ⚠️  Below Bench360 engines")

    print(f"\n  📊 COMPARISON — Batch Decode TPS")
    print(f"{'─'*70}")
    if peak_bs > 1 and peak_bs in results:
        r = results[peak_bs]
        print(f"  MegaGemm peak: {peak_tps:.0f} TPS at batch={peak_bs}")
        print(f"  (vLLM H100 high-concurrency: ~3000+ TPS)")
        print(f"  (SGLang A100 batch=64: ~5000 TPS)")
        print(f"  (These are bigger GPUs — L4 is 5-10x less bandwidth)")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="MegaGemm Decode-Only Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-tokens", type=int, default=512,
                        help="Tokens to generate per request")
    parser.add_argument("--max-batch", type=int, default=64,
                        help="Max batch size to test")
    parser.add_argument("--quantize", choices=["int8", "fp8"], default=None)
    args = parser.parse_args()

    sizes = [s for s in [1, 2, 4, 8, 16, 32, 64, 128] if s <= args.max_batch]

    run_decode_benchmark(
        model_name=args.model,
        output_tokens=args.output_tokens,
        batch_sizes=sizes,
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()

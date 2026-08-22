#!/usr/bin/env python3
"""
🚀 CPU Performance Push — Close the gap with llama.cpp
=======================================================
Target: 75% of llama.cpp (56 tok/s from current 10 tok/s)

Optimizations tested:
1. torch.compile (fuse forward pass, eliminate Python overhead)
2. torch.set_float32_matmul_precision('high')
3. INT8 kernel + torch.compile combo
"""
import torch
import torch.nn as nn
import time
import sys
import os

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain what gravity is in two sentences."
MAX_TOKENS = 32
REPEAT = 3

torch.set_num_threads(os.cpu_count())


def decode_tok_per_sec(engine, prompt, max_tokens, warmup=1, repeat=3):
    """Measure decode-only tok/s (excluding prefill)."""
    for _ in range(warmup):
        engine.generate(prompt, max_new_tokens=max_tokens, temperature=0.0, verbose=False)

    times = []
    output = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = engine.generate(prompt, max_new_tokens=max_tokens, temperature=0.0, verbose=False)
        dt = time.perf_counter() - t0
        times.append(dt)
        if output is None:
            output = out

    avg = sum(times) / len(times)
    return max_tokens / avg, output


def convert_int8(model):
    from megagemm.kernels.cpu_int8 import CPUInt8Linear
    count = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, child_name, CPUInt8Linear.from_float(child))
                count += 1
    return count


def main():
    from megagemm.engine import InferenceEngine

    print("=" * 70)
    print("  🚀 CPU Performance Push")
    print("=" * 70)
    print(f"  Target: 56 tok/s (75% of llama.cpp Q8 = 75 tok/s)")
    print(f"  Threads: {torch.get_num_threads()}")
    print()

    results = []

    # ── 1. Baseline FP32 ──
    print("─" * 70)
    print("  [1] Baseline FP32 (eager)")
    sys.stdout.flush()
    engine = InferenceEngine(MODEL, device='cpu', dtype=torch.float32)
    tps, out = decode_tok_per_sec(engine, PROMPT, MAX_TOKENS)
    print(f"  → {tps:.1f} tok/s")
    results.append(("FP32 eager", tps))

    # ── 2. torch.compile on model ──
    print()
    print("─" * 70)
    print("  [2] FP32 + torch.compile")
    sys.stdout.flush()
    print("  Compiling model (first run will be slow)...", flush=True)
    engine.model.forward = torch.compile(
        engine.model.forward,
        mode="reduce-overhead",
        fullgraph=False,
    )
    # Extra warmup for compilation
    print("  Warmup (compilation)...", end=" ", flush=True)
    t0 = time.perf_counter()
    engine.generate(PROMPT, max_new_tokens=8, temperature=0.0, verbose=False)
    print(f"{time.perf_counter()-t0:.1f}s")

    tps, out = decode_tok_per_sec(engine, PROMPT, MAX_TOKENS)
    print(f"  → {tps:.1f} tok/s")
    results.append(("FP32 compiled", tps))
    del engine

    # ── 3. Set matmul precision ──
    print()
    print("─" * 70)
    print("  [3] FP32 + high precision matmul")
    sys.stdout.flush()
    torch.set_float32_matmul_precision('high')
    engine = InferenceEngine(MODEL, device='cpu', dtype=torch.float32)
    tps, out = decode_tok_per_sec(engine, PROMPT, MAX_TOKENS)
    print(f"  → {tps:.1f} tok/s")
    results.append(("FP32 high-prec", tps))

    # ── 4. FP32 + high prec + compile ──
    print()
    print("─" * 70)
    print("  [4] FP32 + high precision + torch.compile")
    sys.stdout.flush()
    print("  Compiling...", end=" ", flush=True)
    engine.model.forward = torch.compile(
        engine.model.forward,
        mode="reduce-overhead",
        fullgraph=False,
    )
    engine.generate(PROMPT, max_new_tokens=8, temperature=0.0, verbose=False)
    print("done")
    tps, out = decode_tok_per_sec(engine, PROMPT, MAX_TOKENS)
    print(f"  → {tps:.1f} tok/s")
    results.append(("FP32 hp+compile", tps))
    del engine

    # ── 5. INT8 + compile ──
    print()
    print("─" * 70)
    print("  [5] INT8 AVX2 + torch.compile")
    sys.stdout.flush()
    torch.set_float32_matmul_precision('high')
    engine = InferenceEngine(MODEL, device='cpu', dtype=torch.float32)
    n = convert_int8(engine.model)
    print(f"  Converted {n} layers to INT8")
    print("  Compiling...", end=" ", flush=True)
    engine.model.forward = torch.compile(
        engine.model.forward,
        mode="reduce-overhead",
        fullgraph=False,
    )
    engine.generate(PROMPT, max_new_tokens=8, temperature=0.0, verbose=False)
    print("done")
    tps, out = decode_tok_per_sec(engine, PROMPT, MAX_TOKENS)
    print(f"  → {tps:.1f} tok/s")
    results.append(("INT8 compiled", tps))

    # ── Summary ──
    print()
    print("=" * 70)
    print("  Summary (llama.cpp Q8 = 75.3 tok/s)")
    print("=" * 70)

    for name, tps in results:
        pct = tps / 75.3 * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:<18} │ {tps:>6.1f} tok/s │ {pct:>5.1f}% │ {bar}")

    print(f"  {'llama.cpp Q8':<18} │ {'75.3':>6} tok/s │ 100.0% │ {'█' * 50}")
    print("=" * 70)


if __name__ == "__main__":
    main()

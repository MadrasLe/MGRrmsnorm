#!/usr/bin/env python3
"""
⚡ CPU INT8 End-to-End Benchmark
==================================
FP32 baseline vs AVX2 INT8 (all nn.Linear → CPUInt8Linear)
"""
import torch
import torch.nn as nn
import time
import sys
import os

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain what gravity is in two sentences."
MAX_TOKENS = 32

torch.set_num_threads(os.cpu_count())


def convert_to_int8(model):
    """Replace all nn.Linear with CPUInt8Linear."""
    from megagemm.kernels.cpu_int8 import CPUInt8Linear
    count = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                int8_layer = CPUInt8Linear.from_float(child)
                setattr(module, child_name, int8_layer)
                count += 1
    return count


def bench(engine, prompt, max_tokens, warmup=1, repeat=3):
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
    from megagemm.engine import InferenceEngine

    print("=" * 70)
    print("  ⚡ CPU INT8 End-to-End Benchmark")
    print("=" * 70)
    print(f"  Model:   {MODEL}")
    print(f"  Threads: {torch.get_num_threads()}")
    print()

    # ── FP32 baseline ──
    print("─" * 70)
    print("  [1/2] FP32 baseline...")
    sys.stdout.flush()
    engine = InferenceEngine(MODEL, device='cpu', dtype=torch.float32)

    print("  Benchmarking FP32...", end=" ", flush=True)
    tps_fp32, out_fp32 = bench(engine, PROMPT, MAX_TOKENS)
    print(f"{tps_fp32:.1f} tok/s")
    print(f"  → {out_fp32[:70]}...")

    # ── Convert to INT8 ──
    print()
    print("─" * 70)
    print("  [2/2] Converting to AVX2 INT8...")
    sys.stdout.flush()

    t0 = time.perf_counter()
    n = convert_to_int8(engine.model)
    dt = time.perf_counter() - t0

    # Memory after conversion
    param_bytes = sum(
        p.nelement() * p.element_size() for p in engine.model.parameters()
    )
    buffer_bytes = sum(
        b.nelement() * b.element_size() for b in engine.model.buffers()
    )
    total_mb = (param_bytes + buffer_bytes) / 1024 / 1024

    print(f"  Converted {n} layers in {dt:.1f}s")
    print(f"  Model size: {total_mb:.0f}MB (was 2404MB FP32)")

    print("  Benchmarking INT8...", end=" ", flush=True)
    tps_int8, out_int8 = bench(engine, PROMPT, MAX_TOKENS)
    print(f"{tps_int8:.1f} tok/s")
    print(f"  → {out_int8[:70]}...")

    # ── Summary ──
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    speedup = tps_int8 / tps_fp32
    print(f"  FP32:  {tps_fp32:>6.1f} tok/s │ {'█' * int(tps_fp32 * 2)}")
    print(f"  INT8:  {tps_int8:>6.1f} tok/s │ {'█' * int(tps_int8 * 2)}  ({speedup:.2f}x)")
    print()
    print(f"  Output match:")
    print(f"    FP32: {out_fp32[:60]}...")
    print(f"    INT8: {out_int8[:60]}...")
    print("=" * 70)


if __name__ == "__main__":
    main()

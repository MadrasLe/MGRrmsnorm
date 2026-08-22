#!/usr/bin/env python3
"""
⚡ AVX2 INT8 Kernel Benchmark — Speed + Correctness
"""
import torch
import time
import sys

def bench(fn, warmup=3, repeat=20):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return sum(times[2:-2]) / (len(times) - 4)


def main():
    from megagemm.kernels.cpu_int8 import quantize_to_int8, cpu_gemv, cpu_gemm, _get_lib

    lib = _get_lib()

    # Check threads
    num_threads = 1
    if lib and hasattr(lib, 'megagemm_get_num_threads'):
        num_threads = lib.megagemm_get_num_threads()

    print("=" * 70)
    print("  ⚡ AVX2 INT8 Kernel Benchmark")
    print("=" * 70)
    print(f"  Kernel: {'✅ loaded' if lib else '❌ fallback'}")
    print(f"  Threads: {num_threads}")
    print()

    # ── Quick sanity check first ──
    print("  🔍 Sanity check (small 64×128)...", end=" ", flush=True)
    w = torch.randn(64, 128)
    x = torch.randn(128)
    wq, s = quantize_to_int8(w)
    out = cpu_gemv(wq, s, x)
    ref = w @ x
    cos = torch.nn.functional.cosine_similarity(out.unsqueeze(0), ref.unsqueeze(0)).item()
    print(f"cos={cos:.4f} {'✅' if cos > 0.99 else '❌'}")

    if cos < 0.9:
        print("  ❌ FAILED sanity check. Kernel output is wrong!")
        return

    # ── GEMV tests ──
    sizes = [
        ("Small",            256,  512),
        ("QKV (0.5B)",       896,  1152),
        ("Gate+Up (0.5B)",   896,  4864*2),
        ("QKV (7B)",         3584, 4608),
        ("Down (7B)",        18944, 3584),
        ("Gate+Up (7B)",     3584, 18944*2),
    ]

    print()
    print("─" * 70)
    print("  GEMV (decode — 1 token)")
    print("─" * 70)

    for name, K, M in sizes:
        sys.stdout.write(f"  {name:<18} {M:>6}×{K:>6} ... ")
        sys.stdout.flush()

        w_fp32 = torch.randn(M, K)
        x = torch.randn(K)
        wq, scales = quantize_to_int8(w_fp32)

        # Correctness
        ref = w_fp32 @ x
        out = cpu_gemv(wq, scales, x)
        cos = torch.nn.functional.cosine_similarity(
            out.unsqueeze(0), ref.unsqueeze(0)).item()

        # Speed
        t_pt = bench(lambda: w_fp32 @ x)
        t_k = bench(lambda: cpu_gemv(wq, scales, x))
        sp = t_pt / t_k

        print(f"PT={t_pt:.2f}ms  K={t_k:.2f}ms  {sp:.2f}x  cos={cos:.4f}")

    # ── GEMM tests (8 tokens) ──
    print()
    print("─" * 70)
    print("  GEMM (prefill — 8 tokens)")
    print("─" * 70)

    N = 8
    for name, K, M in sizes:
        sys.stdout.write(f"  {name:<18} {N}×{K:>5}×{M:>6} ... ")
        sys.stdout.flush()

        w_fp32 = torch.randn(M, K)
        x = torch.randn(N, K)
        wq, scales = quantize_to_int8(w_fp32)

        ref = x @ w_fp32.T
        out = cpu_gemm(wq, scales, x)
        cos = torch.nn.functional.cosine_similarity(
            out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()

        t_pt = bench(lambda: x @ w_fp32.T)
        t_k = bench(lambda: cpu_gemm(wq, scales, x))
        sp = t_pt / t_k

        print(f"PT={t_pt:.2f}ms  K={t_k:.2f}ms  {sp:.2f}x  cos={cos:.4f}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

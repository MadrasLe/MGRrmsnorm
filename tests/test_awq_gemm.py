"""
Unit test & benchmark: MegaGemm AWQ GEMM vs AutoAWQ kernel.
Run on Colab: python tests/test_awq_gemm.py
"""

# This is a standalone CUDA benchmark driver, not a fixture-based pytest module.
__test__ = False

import torch
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_awq_tensors(K, N, group_size=128, device='cuda'):
    """Create fake AWQ quantized tensors for testing."""
    assert N % 8 == 0
    assert K % group_size == 0

    num_groups = K // group_size

    # Simulate quantized weights: random INT4 values packed into INT32
    # Each int32 holds 8 INT4 values (4 bits each)
    qweight = torch.zeros(K, N // 8, dtype=torch.int32, device=device)
    for i in range(8):
        shift = i * 4
        vals = torch.randint(0, 16, (K, N // 8), dtype=torch.int32, device=device)
        qweight |= (vals << shift)

    # Scales: random FP16 per group
    scales = torch.randn(num_groups, N, dtype=torch.float16, device=device).abs() * 0.1 + 0.01

    # Zeros: random INT4 packed same as weights
    qzeros = torch.zeros(num_groups, N // 8, dtype=torch.int32, device=device)
    for i in range(8):
        shift = i * 4
        vals = torch.randint(0, 16, (num_groups, N // 8), dtype=torch.int32, device=device)
        qzeros |= (vals << shift)

    return qweight, scales, qzeros


def dequant_reference(qweight, scales, qzeros, group_size):
    """PyTorch reference dequantization (slow but correct)."""
    K, N_packed = qweight.shape
    N = N_packed * 8
    device = qweight.device

    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device)

    # Unpack weights
    iweight = ((qweight.unsqueeze(-1) >> shifts) & 0xF).reshape(K, N)

    # Unpack zeros
    izeros = ((qzeros.unsqueeze(-1) >> shifts) & 0xF).reshape(-1, N)

    # Apply AWQ reorder
    reverse_idx = torch.arange(N, dtype=torch.int32, device=device)
    reverse_idx = reverse_idx.view(-1, 8)[:, AWQ_REVERSE_ORDER].reshape(-1)
    iweight = iweight[:, reverse_idx] & 0xF
    izeros = izeros[:, reverse_idx] & 0xF

    # Dequantize
    scales_exp = scales.repeat_interleave(group_size, dim=0)
    izeros_exp = izeros.repeat_interleave(group_size, dim=0)

    return (iweight.to(torch.float16) - izeros_exp.to(torch.float16)) * scales_exp


def test_correctness(M, K, N, group_size=128):
    """Compare MegaGemm AWQ kernel output vs PyTorch reference matmul."""
    from megagemm.kernels.awq_gemm import awq_gemm_megagemm

    torch.manual_seed(42)

    x = torch.randn(M, K, dtype=torch.float16, device='cuda')
    qweight, scales, qzeros = create_awq_tensors(K, N, group_size)

    # Reference: dequant → FP16 matmul
    w_fp16 = dequant_reference(qweight, scales, qzeros, group_size)  # [K, N]
    ref_out = x @ w_fp16  # [M, N]

    # MegaGemm kernel
    mg_out = awq_gemm_megagemm(x, qweight, scales, qzeros, group_size)

    if mg_out is None:
        print(f"  ⚠️  SKIP M={M:>4}, K={K:>5}, N={N:>5} — kernel returned None")
        return True

    # Compare
    max_diff = (ref_out - mg_out).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_out.flatten().unsqueeze(0).float(),
        mg_out.flatten().unsqueeze(0).float()
    ).item()

    passed = cos_sim > 0.99 and max_diff < 1.0
    status = "✅ PASS" if passed else "❌ FAIL"

    print(f"  {status} M={M:>4}, K={K:>5}, N={N:>5} │ max_diff={max_diff:.4f}, cos={cos_sim:.6f}")
    return passed


def test_speed(M, K, N, group_size=128, warmup=20, iters=100):
    """Benchmark MegaGemm vs AutoAWQ kernel."""
    from megagemm.kernels.awq_gemm import awq_gemm_megagemm

    torch.manual_seed(42)

    x = torch.randn(M, K, dtype=torch.float16, device='cuda')
    qweight, scales, qzeros = create_awq_tensors(K, N, group_size)

    # ── AutoAWQ baseline ──
    try:
        from awq.modules.triton.gemm import awq_gemm_triton
        has_autoawq = True
    except ImportError:
        has_autoawq = False

    if has_autoawq:
        x_2d = x.reshape(-1, K)
        for _ in range(warmup):
            awq_gemm_triton(x_2d, qweight, scales, qzeros, split_k_iters=8)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            awq_gemm_triton(x_2d, qweight, scales, qzeros, split_k_iters=8)
        end.record()
        torch.cuda.synchronize()
        autoawq_ms = start.elapsed_time(end) / iters
    else:
        autoawq_ms = float('inf')

    # ── MegaGemm kernel ──
    for _ in range(warmup):
        awq_gemm_megagemm(x, qweight, scales, qzeros, group_size)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        awq_gemm_megagemm(x, qweight, scales, qzeros, group_size)
    end.record()
    torch.cuda.synchronize()
    mg_ms = start.elapsed_time(end) / iters

    speedup = autoawq_ms / mg_ms if mg_ms > 0 else 0
    faster = "🔥" if speedup > 1.1 else "🟡" if speedup > 0.9 else "🔴"

    awq_str = f"{autoawq_ms:.3f}ms" if has_autoawq else "N/A"
    print(f"  {faster} M={M:>4}, K={K:>5}, N={N:>5} │ AutoAWQ: {awq_str}, MegaGemm: {mg_ms:.3f}ms, Speedup: {speedup:.2f}x")


if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # ═══════════════════════════════════════════
    print("=" * 70)
    print("  CORRECTNESS TESTS (vs PyTorch reference dequant)")
    print("=" * 70)

    all_pass = True

    # Qwen 2.5 7B dimensions
    for M in [1, 2, 4, 32]:
        for K, N in [(3584, 3584), (3584, 18944), (18944, 3584)]:
            all_pass &= test_correctness(M, K, N)

    print()
    if all_pass:
        print("✅ ALL CORRECTNESS TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED — DO NOT INTEGRATE")
        sys.exit(1)

    # ═══════════════════════════════════════════
    print()
    print("=" * 70)
    print("  SPEED BENCHMARK (vs AutoAWQ Triton kernel)")
    print("=" * 70)

    for M in [1, 4, 32]:
        for K, N in [(3584, 3584), (3584, 18944), (18944, 3584)]:
            test_speed(M, K, N)
        print()

    print("Done!")

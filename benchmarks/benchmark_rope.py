"""
🔄 RoPE Benchmark - Compare PyTorch vs MegaGemm CUDA
"""

import torch
import time
import sys

# Add parent to path
sys.path.insert(0, '.')

from megagemm import RoPE, precompute_freqs_cis, apply_rotary_emb


def benchmark_fn(fn, warmup=10, iterations=100):
    """Benchmark a function"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / iterations * 1000  # ms


class PyTorchRoPE(torch.nn.Module):
    """Reference PyTorch RoPE for comparison"""
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.dim = dim
        self.max_seq_len = max_seq_len

    def forward(self, q, k):
        seq_len = q.shape[2]
        t = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)

        # Apply rotation
        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]

        q_rot = torch.stack([
            q_even * cos - q_odd * sin,
            q_even * sin + q_odd * cos
        ], dim=-1).flatten(-2)

        k_rot = torch.stack([
            k_even * cos - k_odd * sin,
            k_even * sin + k_odd * cos
        ], dim=-1).flatten(-2)

        return q_rot, k_rot


def run_benchmark(batch=32, heads=32, seq_len=512, head_dim=128, dtype=torch.float16):
    """Run RoPE benchmark"""
    device = 'cuda'

    print(f"\n{'='*60}")
    print(f"RoPE Benchmark: batch={batch}, heads={heads}, seq={seq_len}, dim={head_dim}")
    print(f"dtype={dtype}, device={device}")
    print(f"{'='*60}")

    # Create inputs
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    # MegaGemm RoPE
    mg_rope = RoPE(head_dim=head_dim, max_seq_len=seq_len * 2).to(device)

    # PyTorch RoPE
    pt_rope = PyTorchRoPE(head_dim, max_seq_len=seq_len * 2).to(device)

    # Warmup and correctness check
    q_mg, k_mg = mg_rope(q.float(), k.float())
    q_pt, k_pt = pt_rope(q.float(), k.float())

    max_diff = max((q_mg - q_pt).abs().max(), (k_mg - k_pt).abs().max())
    print(f"Correctness check: max diff = {max_diff:.2e}")

    # Benchmark
    mg_time = benchmark_fn(lambda: mg_rope(q, k))
    pt_time = benchmark_fn(lambda: pt_rope(q, k))

    speedup = pt_time / mg_time

    print(f"\nResults:")
    print(f"  PyTorch RoPE:  {pt_time:.3f} ms")
    print(f"  MegaGemm RoPE: {mg_time:.3f} ms")
    print(f"  Speedup:       {speedup:.2f}x {'🚀' if speedup > 1 else ''}")

    return {'pytorch': pt_time, 'megagemm': mg_time, 'speedup': speedup}


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")

    # Run various configs
    configs = [
        {'batch': 1, 'heads': 32, 'seq_len': 128, 'head_dim': 128},
        {'batch': 1, 'heads': 32, 'seq_len': 512, 'head_dim': 128},
        {'batch': 1, 'heads': 32, 'seq_len': 2048, 'head_dim': 128},
        {'batch': 8, 'heads': 32, 'seq_len': 512, 'head_dim': 128},
    ]

    results = []
    for config in configs:
        result = run_benchmark(**config)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"Average Speedup: {avg_speedup:.2f}x")

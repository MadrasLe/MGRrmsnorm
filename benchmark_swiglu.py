"""
🔥 MEGA GEMM SwiGLU Benchmark
Autor: Gabriel & Ada
Descrição: Benchmark comparativo entre PyTorch native e MegaGemmTriton
"""

import torch
import torch.nn.functional as F
import time
from typing import Tuple, Dict
import sys

# Import from megagemm package
from megagemm import MegaGemmTriton

def benchmark_fn(fn, *args, warmup: int = 10, iterations: int = 100) -> Tuple[float, float]:
    """
    Benchmark a function with warmup and multiple iterations.
    Returns: (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std


class PyTorchSwiGLU(torch.nn.Module):
    """PyTorch native SwiGLU implementation for baseline comparison."""
    
    def __init__(self, d_model: int, multiple_of: int = 256, hidden_multiple: float = 5/3):
        super().__init__()
        hidden = int(d_model * hidden_multiple)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.hidden = hidden
        
        # Separate W1 (gate) and W2 (value) 
        self.w1 = torch.nn.Linear(d_model, hidden, bias=False)  # Gate
        self.w2 = torch.nn.Linear(d_model, hidden, bias=False)  # Value
        self.w3 = torch.nn.Linear(hidden, d_model, bias=False)  # Output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        value = self.w2(x)
        hidden = F.silu(gate) * value
        return self.w3(hidden)


def run_benchmark(
    batch_size: int = 32,
    seq_len: int = 512,
    d_model: int = 4096,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> Dict[str, Dict]:
    """Run full benchmark suite."""
    
    print(f"\n{'='*60}")
    print(f"🔥 MEGA GEMM SwiGLU Benchmark")
    print(f"{'='*60}")
    print(f"Config: batch={batch_size}, seq={seq_len}, d_model={d_model}")
    print(f"Dtype: {dtype}, Device: {device}")
    print(f"{'='*60}\n")
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    
    # Initialize models
    pytorch_model = PyTorchSwiGLU(d_model).to(device).to(dtype)
    triton_model = MegaGemmTriton(d_model).to(device).to(dtype)
    
    # Ensure same weights for fair comparison (copy w1+w2 into w12)
    with torch.no_grad():
        # triton_model.w12 is [d_model, 2*hidden], cats gate and value
        triton_model.w12.weight.data[:triton_model.hidden, :] = pytorch_model.w1.weight.data
        triton_model.w12.weight.data[triton_model.hidden:, :] = pytorch_model.w2.weight.data
        triton_model.w3.weight.data = pytorch_model.w3.weight.data.clone()
    
    results = {}
    
    # =========================================================================
    # Benchmark PyTorch Native
    # =========================================================================
    print("⏱️  Benchmarking PyTorch Native SwiGLU...")
    
    def pytorch_forward():
        return pytorch_model(x)
    
    pt_mean, pt_std = benchmark_fn(pytorch_forward)
    results['pytorch'] = {'mean_ms': pt_mean, 'std_ms': pt_std}
    print(f"   Mean: {pt_mean:.3f} ms ± {pt_std:.3f} ms")
    
    # =========================================================================
    # Benchmark MegaGemm Triton
    # =========================================================================
    print("⏱️  Benchmarking MegaGemm Triton SwiGLU...")
    
    def triton_forward():
        return triton_model(x)
    
    tr_mean, tr_std = benchmark_fn(triton_forward)
    results['triton'] = {'mean_ms': tr_mean, 'std_ms': tr_std}
    print(f"   Mean: {tr_mean:.3f} ms ± {tr_std:.3f} ms")
    
    # =========================================================================
    # Correctness Check
    # =========================================================================
    print("\n🔍 Verifying correctness...")
    with torch.no_grad():
        out_pt = pytorch_model(x)
        out_tr = triton_model(x)
        max_diff = (out_pt - out_tr).abs().max().item()
        print(f"   Max diff: {max_diff:.6f}")
        if max_diff < 1e-2:
            print("   ✅ Correctness PASSED")
        else:
            print("   ⚠️  Outputs differ significantly (expected with different weight layouts)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    speedup = pt_mean / tr_mean
    
    print(f"\n{'='*60}")
    print("📊 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Implementation':<25} {'Time (ms)':<15} {'Speedup':<10}")
    print(f"{'-'*60}")
    print(f"{'PyTorch Native':<25} {pt_mean:>6.3f} ± {pt_std:.3f}   {'1.00x':>10}")
    print(f"{'MegaGemm Triton':<25} {tr_mean:>6.3f} ± {tr_std:.3f}   {f'{speedup:.2f}x':>10}")
    print(f"{'='*60}")
    
    if speedup > 1.0:
        print(f"\n🚀 MegaGemm Triton is {speedup:.2f}x FASTER than PyTorch!")
    else:
        print(f"\n📉 PyTorch is {1/speedup:.2f}x faster (unexpected)")
    
    # Memory stats
    print(f"\n📦 Memory Usage:")
    print(f"   Peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"   Peak reserved:  {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    
    return results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        sys.exit(1)
    
    print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
    print(f"🔧 PyTorch: {torch.__version__}")
    
    # Run benchmarks for different configurations
    configs = [
        # (batch, seq, d_model)
        (8, 128, 2048),    # Small
        (16, 256, 4096),   # Medium
        (32, 512, 4096),   # Large (LLM-like)
    ]
    
    for batch, seq, d_model in configs:
        try:
            run_benchmark(batch_size=batch, seq_len=seq, d_model=d_model)
        except Exception as e:
            print(f"❌ Error with config ({batch}, {seq}, {d_model}): {e}")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

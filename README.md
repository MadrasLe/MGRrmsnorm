#  Optimized Fused RMSNorm CUDA Kernel and MegaGemm Kernels
<div align="center">

[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

<!-- Badge Personalizado de Performance -->
[![Performance](https://img.shields.io/badge/Speedup-+67%25_vs_Native-brightgreen?style=for-the-badge&logo=rocket&logoColor=white)](https://github.com/MadrasLe/MGRrmsnorm)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>


This repository contains a highly optimized, production-ready CUDA implementation of **RMSNorm (Root Mean Square Normalization)**. It features a custom **Fused Kernel** with efficient **Backward Pass** support, designed to outperform standard PyTorch implementations and rival industry-standard kernels like Liger.

Built for high-throughput LLM training, this kernel leverages advanced CUDA techniques including **Vectorized Loads (float4)**, **Warp-Level Reductions**, and **Grid-Stride Loops** to minimize memory bandwidth contention.

---

## Performance Benchmark

Tests conducted on a **Tesla T4 GPU** training a MoE (Mixture of Experts) model (~60M params) with FP16 mixed precision.

### TPS Comparison (Higher is Better)

```text
PyTorch RMSNorm    ██████████████ 21,752
Gabriel Kernel     ████████████████████████ 36,447  🚀
Liger Kernel       ██████████████████████████ 37,476
```

| Implementation | Tokens Per Second (TPS) | Speedup vs PyTorch | Status |
| :--- | :--- | :--- | :--- |
| **Liger Kernel (LinkedIn)** | **~37,476** | **1.72x** | Industry Standard |
| **MegaGemm Fused Kernel** | **~36,447** | **1.67x** | **This Repo** 🚀 |
| **PyTorch Baseline** | ~21,752 | 1.0x | Baseline |

**Analysis:**
- **Vs PyTorch:** The custom kernel provides a massive **~67% throughput increase** compared to the native PyTorch implementation.
- **Vs Liger:** It performs nearly identically to the highly-optimized Liger kernel (within ~2.7% margin), validating the "production-grade" architecture.
- 
### Numerical Stability & Loss Analysis
Beyond speed, this kernel demonstrated a slight improvement in training loss convergence compared to both PyTorch native and Liger implementations during our experiments.

-Observed Effect: Lower validation loss curves in early training stages.

-Hypothesis: This is likely due to the implementation of the reduction step, which maintains higher precision accumulation in registers (float32) before casting back to float16 for memory writing, reducing underflow/overflow artifacts in the RMS calculation.

###  Test Environment

- **Hardware:** NVIDIA Tesla T4 (16GB)
- **Precision:** FP16 Mixed Precision
- **Model Architecture:** Mixture of Experts (MoE)
    - **Layers:** 6
    - **Experts:** 4 (Top-2 Active)
    - **Parameters:** ~60M Total / ~42M Active
- **Workload:** Full Training Loop (Forward + Backward)

---

### Why This Matters

In large-scale LLM training, Normalization layers (RMSNorm/LayerNorm) can account for **8–12% of total step time** due to their memory-bound nature. Reducing their overhead directly increases overall throughput and decreases training cost.

This kernel demonstrates that:
- **Custom CUDA implementations** can rival enterprise-grade solutions (like Liger).
- **Decoupling from external frameworks** improves reproducibility and control.
- **Kernel-level optimization** (memory coalescing, warp shuffles) remains a critical frontier in scaling LLMs.

---

##  Key Features & Optimizations

### 1. Memory Bandwidth Optimization
- **Float4 Vectorization:** Loads data in 128-bit chunks (4 floats) per instruction, maximizing global memory utilization.
- **Strict Alignment Checks:** Enforces 16-byte memory alignment to prevent segmentation faults and ensure safe vectorization.

### 2. Compute Efficiency
- **Warp Reduction:** Uses `__shfl_down_sync` primitive for ultra-fast intra-warp summation, avoiding slower shared memory for the initial reduction stage.
- **Fused Operation:** Performs root-mean-square calculation, normalization, and scaling in a single kernel launch, eliminating redundant global memory round-trips.
- **Fast Math:** Leverages `rsqrtf` and FMA (Fused Multiply-Add) instructions.

### 3. Scalable Backward Pass
- **Grid-Stride Loops:** The Backward kernel uses a grid-stride pattern, decoupling the number of CUDA blocks from the batch size.
- **Register Accumulation:** Weight gradients (`dw`) are accumulated in local registers (`float4 dw_acc[]`) instead of global memory.
- **Minimized Atomics:** Global `atomicAdd` operations are performed only **once per thread** (after the loop), reducing contention from $O(N)$ to $O(\text{GridSize})$.

---

##  Installation

### Prerequisites
- NVIDIA GPU (Compute Capability 7.5+ recommended)
- CUDA Toolkit
- PyTorch

### Build from Source
```bash
# Install explicitly
python setup.py install

# Or build in-place
python setup.py build_ext --inplace
```

---

## Usage

The kernel exposes a `torch.autograd.Function`, making it a drop-in replacement for `torch.nn.RMSNorm`.

```python
import torch
import torch.nn as nn
from rmsnorm_cuda import RMSNorm  # Assuming installed as package or local import

# Initialize
device = torch.device("cuda")
hidden_size = 4096
model = RMSNorm(hidden_size).to(device)

# Forward
x = torch.randn(32, 128, hidden_size, device=device, requires_grad=True)
y = model(x)

# Backward (Fully supported)
loss = y.sum()
loss.backward()

print("Input Gradients:", x.grad)
print("Weight Gradients:", model.weight.grad)
```

---

## 🧠 Technical Architecture

### Forward Pass
1. **Load:** Threads read input $x$ using `float4`.
2. **Reduce:** Compute $\sum x^2$ via warp shuffle + shared memory reduction.
3. **Scale:** Thread 0 computes $s = \frac{1}{\sqrt{\text{mean} + \epsilon}}$.
4. **Write:** All threads write $y = x \cdot s \cdot w$.
5. **Context:** Saves $s$ (inverse RMS) for the backward pass.

### Backward Pass
1. **Grid-Stride:** A fixed grid (e.g., 128 blocks) iterates over rows.
2. **Compute $dx$:** Uses the saved $s$ to compute input gradients efficiently.
3. **Accumulate $dw$:** Accumulates weight gradients in registers to avoid atomic contention.
4. **Finalize:** Atomically adds register values to the global weight gradient buffer.

---

## Credits

- **Gabriel:** Lead Engineer & Implementation.


# 🔥 MegaGemm - High Performance CUDA Kernels for LLMs

<div align="center">

[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-6C4DC4?style=for-the-badge&logo=openai&logoColor=white)](https://triton-lang.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

[![Performance](https://img.shields.io/badge/Speedup-3x_vs_Native-brightgreen?style=for-the-badge&logo=rocket&logoColor=white)](https://github.com/MadrasLe/MGRrmsnorm)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

Production-ready **RMSNorm** and **SwiGLU** kernels optimized for LLM training and inference. Drop-in replacements for PyTorch with **up to 3x speedup**.

## ✨ Features

- 🚀 **RMSNorm CUDA Kernel** - FP32, FP16, BF16 support with vectorized loads
- ⚡ **SwiGLU Triton Kernel** - Fused activation with memory-efficient design  
- 🔄 **Full Autograd Support** - Forward and backward passes
- 📦 **pip installable** - `pip install -e .`

---

## 📊 Performance Benchmarks

### RMSNorm Performance

| GPU | Architecture | PyTorch | MegaGemm | Speedup |
|-----|-------------|---------|----------|---------|
| **NVIDIA L4** | Ada Lovelace | 0.818 ms | 0.270 ms | **3.03x** 🔥 |
| **Tesla T4** | Turing | 21,752 TPS | 36,447 TPS | **1.67x** |

> Tested with: batch=32, seq=128, hidden=4096, dtype=float16

### SwiGLU Performance

| GPU | PyTorch | MegaGemm | Notes |
|-----|---------|----------|-------|
| **NVIDIA L4** | 58.78 ms | 56.64 ms | Memory-efficient (matmul-bound) |

> The SwiGLU kernel's main benefit is **memory efficiency** through fused W1+W2 matmul, not raw compute speed.

### Why RMSNorm is 3x Faster

RMSNorm is **memory-bound**, making it ideal for optimization:
- **half2 Vectorization** - 64-bit loads for FP16
- **float4 Vectorization** - 128-bit loads for FP32  
- **Warp Shuffles** - Fast reduction without shared memory
- **FP32 Accumulators** - Numerical stability in mixed precision

---

## 🚀 Installation

```bash
# Clone
git clone https://github.com/MadrasLe/MGRrmsnorm.git
cd MGRrmsnorm

# Install
pip install triton
pip install -e .
```

### Requirements
- NVIDIA GPU (Compute Capability 7.5+)
- CUDA Toolkit 11.8+
- PyTorch 2.0+
- Triton 2.0+

---

## 📖 Usage

### RMSNorm (CUDA)

```python
from megagemm import RMSNorm
import torch

# FP16
model = RMSNorm(4096).cuda().half()
x = torch.randn(32, 128, 4096, device='cuda', dtype=torch.float16)
y = model(x)

# BF16 (Ampere+ GPUs)
model_bf16 = RMSNorm(4096).cuda().to(torch.bfloat16)
y = model_bf16(x.to(torch.bfloat16))

# Backward pass fully supported
loss = y.sum()
loss.backward()
```

### SwiGLU (Triton)

```python
from megagemm import MegaGemmTriton
import torch

model = MegaGemmTriton(d_model=4096).cuda().half()
x = torch.randn(32, 128, 4096, device='cuda', dtype=torch.float16)
y = model(x)  # [32, 128, 4096]
```

---

## 🧠 Technical Details

### RMSNorm Kernel Architecture

**Forward Pass:**
1. Load input with `float4`/`half2` vectorization
2. Compute Σx² via warp shuffle reduction
3. Calculate inverse RMS: `s = rsqrt(mean + ε)`
4. Write normalized output: `y = x * s * w`

**Backward Pass:**
1. Grid-stride loop over rows
2. Register accumulation for weight gradients
3. Single atomic add per thread (minimized contention)

### SwiGLU Kernel Architecture

- Fused W1+W2 into single matmul for memory efficiency
- Triton kernel for SiLU(gate) × value activation
- No intermediate tensor allocation

---

## 📁 Project Structure

```
MGRrmsnorm/
├── megagemm/              # Python package
│   ├── __init__.py
│   ├── rmsnorm.py         # RMSNorm module
│   └── swiglu.py          # SwiGLU Triton module
├── src/
│   ├── rmsnorm_kernel.cu  # CUDA kernels (FP32/FP16/BF16)
│   └── rmsnorm_kernel.h   # Header declarations
├── pytorch_binding/
│   └── binding.cpp        # PyTorch C++ bindings
├── benchmark_swiglu.py    # Benchmark script
├── setup.py
└── pyproject.toml
```

---

## 🔬 Numerical Stability

The kernel maintains **FP32 accumulators** during reduction, even for FP16/BF16 inputs. This prevents underflow/overflow in the RMS calculation and has been observed to produce slightly better training loss curves compared to naive implementations.

---

## 📝 Citation

```bibtex
@software{megagemm2024,
  author = {Gabriel Yogi},
  title = {MegaGemm: High Performance CUDA Kernels for LLMs},
  year = {2024},
  url = {https://github.com/MadrasLe/MGRrmsnorm}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Gabriel Yogi** - Lead Engineer & Implementation
- Inspired by [Liger Kernel](https://github.com/linkedin/Liger-Kernel) architecture

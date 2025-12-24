# 🔥 MEGA GEMM - Colab Test Notebook
# Execute cada célula em sequência

# =============================================================================
# CÉLULA 1: Clone o repositório
# =============================================================================
# Se estiver usando seu repo no GitHub:
# !git clone https://github.com/MadrasLe/MGRrmsnorm.git
# %cd MGRrmsnorm

# Ou faça upload do zip e extraia:
# from google.colab import files
# uploaded = files.upload()  # Upload MGRrmsnorm.zip
# !unzip MGRrmsnorm.zip

# =============================================================================
# CÉLULA 2: Verificar GPU
# =============================================================================
"""
!nvidia-smi
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name()}")
"""

# =============================================================================
# CÉLULA 3: Instalar dependências
# =============================================================================
"""
!pip install triton
"""

# =============================================================================
# CÉLULA 4: Compilar e instalar o pacote
# =============================================================================
"""
%cd /content/MGRrmsnorm
!pip install -e . -v
"""

# =============================================================================
# CÉLULA 5: Testar RMSNorm (FP32)
# =============================================================================
"""
import torch
from megagemm import RMSNorm

device = torch.device("cuda")
hidden_size = 4096

# Criar modelo
model = RMSNorm(hidden_size).to(device)
print(f"✅ RMSNorm criado: {model}")

# Testar forward
x = torch.randn(32, 128, hidden_size, device=device, requires_grad=True)
y = model(x)
print(f"✅ Forward OK: {y.shape}")

# Testar backward
loss = y.sum()
loss.backward()
print(f"✅ Backward OK: grad shape = {x.grad.shape}")
"""

# =============================================================================
# CÉLULA 6: Testar RMSNorm (FP16)
# =============================================================================
"""
import torch
from megagemm import RMSNorm

device = torch.device("cuda")
hidden_size = 4096

# FP16
model_fp16 = RMSNorm(hidden_size).to(device).half()
x_fp16 = torch.randn(32, 128, hidden_size, device=device, dtype=torch.float16, requires_grad=True)

y_fp16 = model_fp16(x_fp16)
print(f"✅ FP16 Forward OK: {y_fp16.shape}, dtype={y_fp16.dtype}")

loss = y_fp16.sum()
loss.backward()
print(f"✅ FP16 Backward OK")
"""

# =============================================================================
# CÉLULA 7: Testar RMSNorm (BF16) - Apenas em Ampere+
# =============================================================================
"""
import torch
from megagemm import RMSNorm

device = torch.device("cuda")
hidden_size = 4096

# Verificar se GPU suporta BF16
if torch.cuda.get_device_capability()[0] >= 8:
    model_bf16 = RMSNorm(hidden_size).to(device).to(torch.bfloat16)
    x_bf16 = torch.randn(32, 128, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    
    y_bf16 = model_bf16(x_bf16)
    print(f"✅ BF16 Forward OK: {y_bf16.shape}")
else:
    print("⚠️ BF16 requer GPU Ampere+ (A100, etc)")
"""

# =============================================================================
# CÉLULA 8: Testar SwiGLU Triton
# =============================================================================
"""
import torch
from megagemm import MegaGemmTriton

device = torch.device("cuda")
d_model = 4096

model = MegaGemmTriton(d_model).to(device).half()
print(f"✅ MegaGemmTriton: {model}")

x = torch.randn(32, 128, d_model, device=device, dtype=torch.float16, requires_grad=True)
y = model(x)
print(f"✅ Forward OK: {y.shape}")

loss = y.sum()
loss.backward()
print(f"✅ Backward OK")
"""

# =============================================================================
# CÉLULA 9: Benchmark Comparativo
# =============================================================================
"""
%cd /content/MGRrmsnorm
!python benchmark_swiglu.py
"""

# =============================================================================
# CÉLULA 10: Comparar com PyTorch Nativo
# =============================================================================
"""
import torch
import time
from megagemm import RMSNorm

def benchmark(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms

device = "cuda"
batch, seq, hidden = 32, 512, 4096

# Input
x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float16)

# MegaGemm
mg_norm = RMSNorm(hidden).to(device).half()
mg_time = benchmark(lambda: mg_norm(x))

# PyTorch Nativo
class PyTorchRMSNorm(torch.nn.Module):
    def __init__(self, hidden, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

pt_norm = PyTorchRMSNorm(hidden).to(device).half()
pt_time = benchmark(lambda: pt_norm(x))

print(f"PyTorch Native:  {pt_time:.3f} ms")
print(f"MegaGemm CUDA:   {mg_time:.3f} ms")
print(f"Speedup:         {pt_time/mg_time:.2f}x 🚀")
"""

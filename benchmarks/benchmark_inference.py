"""
🔥 MegaGemm Inference Benchmark — with CUDA kernel compilation
===============================================================
Run on Colab:

Célula 1 — Compilar:
    %cd /content/drive/MyDrive/MGRrmsnorm
    !pip install -e . -v 2>&1 | tail -5
    !pip install huggingface_hub safetensors transformers

Célula 2 — Rodar este script:
    !python benchmark_inference.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# =============================================================================
# 1. Kernel Status Report
# =============================================================================
print("=" * 60)
print("🔍 KERNEL STATUS REPORT")
print("=" * 60)

# RMSNorm CUDA
try:
    import rmsnorm_cuda_ops
    print("  ✅ RMSNorm CUDA kernel: COMPILED")
    _has_rmsnorm = True
except ImportError:
    print("  ❌ RMSNorm CUDA kernel: NOT COMPILED (using PyTorch fallback)")
    print("     Fix: pip install -e .")
    _has_rmsnorm = False

# Triton
try:
    import triton
    print(f"  ✅ Triton: v{triton.__version__}")
    _has_triton = True
except ImportError:
    print("  ❌ Triton: NOT INSTALLED")
    _has_triton = False

# SwiGLU Triton
if _has_triton:
    try:
        from megagemm.swiglu import MegaGemmFunction
        print("  ✅ SwiGLU Triton kernel: AVAILABLE")
    except Exception as e:
        print(f"  ❌ SwiGLU Triton: {e}")

# Paged Attention Triton
if _has_triton:
    try:
        from megagemm.paged_attention import _HAS_TRITON
        if _HAS_TRITON:
            print("  ✅ PagedAttention Triton kernel: AVAILABLE")
        else:
            print("  ⚠️  PagedAttention: Using PyTorch fallback")
    except Exception as e:
        print(f"  ❌ PagedAttention: {e}")

# GPU info
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  🖥️  GPU: {gpu} ({mem:.1f}GB)")
    print(f"  🔧 CUDA: {torch.version.cuda}")
else:
    print("  ❌ No CUDA GPU!")

# =============================================================================
# 2. Load Model
# =============================================================================
print("\n" + "=" * 60)
print("📦 LOADING MODEL")
print("=" * 60)

from megagemm.engine import InferenceEngine

engine = InferenceEngine(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype=torch.float16,
    num_blocks=4096,
    block_size=16,
)

# Check which kernels the model is actually using
from megagemm.models.llama import _HAS_CUDA_RMSNORM, _HAS_TRITON_SWIGLU
print(f"\n  Model using RMSNorm CUDA: {_HAS_CUDA_RMSNORM}")
print(f"  Model using SwiGLU Triton: {_HAS_TRITON_SWIGLU}")

# =============================================================================
# 3. Warmup
# =============================================================================
print("\n" + "=" * 60)
print("🔥 WARMUP")
print("=" * 60)

# Warmup run (compile Triton kernels, warm caches)
_ = engine.generate("Hello", max_new_tokens=5, temperature=0.0)
print("  Warmup done!")

# =============================================================================
# 4. Benchmark
# =============================================================================
print("\n" + "=" * 60)
print("📊 BENCHMARK")
print("=" * 60)

prompts = [
    "The capital of France is",
    "In machine learning, backpropagation is",
    "The theory of general relativity states that",
]

for prompt in prompts:
    print(f"\n  Prompt: '{prompt}'")
    output = engine.generate(
        prompt,
        max_new_tokens=50,
        temperature=0.0,
        verbose=True,
    )
    print(f"  Output: '{output[:100]}...'")

# =============================================================================
# 5. Throughput Test (longer generation)
# =============================================================================
print("\n" + "=" * 60)
print("🏎️  THROUGHPUT TEST (128 tokens)")
print("=" * 60)

torch.cuda.synchronize()
t0 = time.perf_counter()

output = engine.generate(
    "Once upon a time, in a kingdom far away,",
    max_new_tokens=128,
    temperature=0.7,
    top_k=50,
    verbose=True,
)

torch.cuda.synchronize()
total = time.perf_counter() - t0

print(f"\n  Total wall time: {total*1000:.0f}ms")
print(f"  Generated: '{output[:150]}...'")

# =============================================================================
# 6. Compare with HuggingFace (if you want)
# =============================================================================
print("\n" + "=" * 60)
print("📈 COMPARISON vs HuggingFace generate()")
print("=" * 60)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    hf_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    prompt = "The capital of France is"
    inputs = hf_tokenizer(prompt, return_tensors="pt").to("cuda")

    # Warmup
    with torch.no_grad():
        _ = hf_model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        hf_out = hf_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    torch.cuda.synchronize()
    hf_time = time.perf_counter() - t0
    hf_tokens = hf_out.shape[1] - inputs.input_ids.shape[1]
    hf_tps = hf_tokens / hf_time

    # MegaGemm
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    mg_out = engine.generate(prompt, max_new_tokens=50, temperature=0.0)
    torch.cuda.synchronize()
    mg_time = time.perf_counter() - t0
    mg_tps = 50 / mg_time

    print(f"  HuggingFace: {hf_tps:.1f} tok/s ({hf_time*1000:.0f}ms)")
    print(f"  MegaGemm:    {mg_tps:.1f} tok/s ({mg_time*1000:.0f}ms)")
    print(f"  Ratio:       {mg_tps/hf_tps:.2f}x")

    # Cleanup
    del hf_model
    torch.cuda.empty_cache()

except Exception as e:
    print(f"  ⚠️  HF comparison skipped: {e}")

print("\n🏁 Benchmark complete!")

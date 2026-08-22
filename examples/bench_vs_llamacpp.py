#!/usr/bin/env python3
"""
🏁 MegaGemm vs llama.cpp — Head-to-Head CPU Benchmark
========================================================
Same CPU, same model, same prompt, same output length.
Fair comparison on the L4 server.
"""
import subprocess
import os
import sys
import time
import json
import re

MODEL_HF = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_GGUF_Q8 = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
GGUF_FILE_Q8 = "qwen2.5-0.5b-instruct-q8_0.gguf"
GGUF_FILE_Q4 = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

PROMPT = "Explain what gravity is in two sentences."
MAX_TOKENS = 32
REPEAT = 3

LLAMA_DIR = "/tmp/llama_cpp"
GGUF_DIR = "/tmp/gguf_models"


def install_llama_cpp():
    """Install llama.cpp from pre-built release."""
    if os.path.exists(f"{LLAMA_DIR}/llama-cli"):
        print("  llama.cpp already installed")
        return True

    os.makedirs(LLAMA_DIR, exist_ok=True)
    print("  📦 Installing llama.cpp...")

    # Try pip install (easiest)
    try:
        r = subprocess.run(
            ["pip", "install", "llama-cpp-python", "--quiet"],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:
        pass

    # Build from source
    print("  🔨 Building llama.cpp from source...")
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp.git", LLAMA_DIR],
            check=True, capture_output=True, text=True, timeout=60,
        )
        subprocess.run(
            ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
             "-DGGML_CUDA=OFF", "-DGGML_CPU_AARCH64=OFF"],
            cwd=LLAMA_DIR, check=True, capture_output=True, text=True, timeout=30,
        )
        subprocess.run(
            ["cmake", "--build", "build", "--config", "Release", "-j", str(os.cpu_count())],
            cwd=LLAMA_DIR, check=True, capture_output=True, text=True, timeout=300,
        )
        # Find the binary
        for path in [f"{LLAMA_DIR}/build/bin/llama-cli", f"{LLAMA_DIR}/build/llama-cli"]:
            if os.path.exists(path):
                os.symlink(path, f"{LLAMA_DIR}/llama-cli")
                break
        print("  ✅ Built successfully")
        return True
    except Exception as e:
        print(f"  ❌ Build failed: {e}")
        return False


def download_gguf(filename):
    """Download GGUF model from HuggingFace."""
    path = f"{GGUF_DIR}/{filename}"
    if os.path.exists(path):
        print(f"  {filename} already downloaded")
        return path

    os.makedirs(GGUF_DIR, exist_ok=True)
    print(f"  📦 Downloading {filename}...")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=MODEL_GGUF_Q8,
            filename=filename,
            local_dir=GGUF_DIR,
        )
        print(f"  ✅ Downloaded")
        return path
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return None


def bench_llama_cpp(model_path, label):
    """Benchmark llama.cpp CLI."""
    cli = None
    for path in [f"{LLAMA_DIR}/llama-cli",
                 f"{LLAMA_DIR}/build/bin/llama-cli"]:
        if os.path.exists(path):
            cli = path
            break

    if not cli:
        print(f"  ❌ llama-cli not found")
        return None, None

    print(f"  Running {label}...", end=" ", flush=True)

    times = []
    output = None

    for i in range(REPEAT):
        try:
            r = subprocess.run(
                [cli,
                 "-m", model_path,
                 "-p", PROMPT,
                 "-n", str(MAX_TOKENS),
                 "--temp", "0",
                 "-t", str(os.cpu_count()),
                 "--no-display-prompt",
                 "-ngl", "0",  # Force CPU only
                ],
                capture_output=True, text=True, timeout=120,
            )
            stderr = r.stderr

            # Parse timing from llama.cpp output
            # llama_perf_sampler_print:    sampling time =    ...
            # llama_perf_context_print:          eval time =    ...

            if output is None:
                output = r.stdout.strip()

            # Try to find tokens/s
            for line in stderr.split('\n'):
                if 'eval time' in line and 'token' in line:
                    # eval time = X ms / Y tokens (Z tokens per second)
                    m = re.search(r'([\d.]+)\s+tokens per second', line)
                    if m:
                        times.append(float(m.group(1)))
        except Exception as e:
            print(f"\n  ❌ Error: {e}")
            return None, None

    if times:
        avg = sum(times) / len(times)
        print(f"{avg:.1f} tok/s")
        return avg, output
    else:
        # Fallback: time it ourselves
        print("(timing from wall clock)")
        return None, output


def bench_megagemm(mode="fp32"):
    """Benchmark MegaGemm CPU."""
    import torch
    torch.set_num_threads(os.cpu_count())
    from megagemm.engine import InferenceEngine

    print(f"  Loading MegaGemm {mode.upper()}...", end=" ", flush=True)

    if mode == "int8":
        import torch.nn as nn
        from megagemm.kernels.cpu_int8 import CPUInt8Linear
        engine = InferenceEngine(MODEL_HF, device='cpu', dtype=torch.float32)
        for name, module in engine.model.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    setattr(module, child_name, CPUInt8Linear.from_float(child))
    else:
        engine = InferenceEngine(MODEL_HF, device='cpu', dtype=torch.float32)

    # Warmup
    engine.generate(PROMPT, max_new_tokens=MAX_TOKENS, temperature=0.0, verbose=False)

    times = []
    output = None
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        out = engine.generate(PROMPT, max_new_tokens=MAX_TOKENS, temperature=0.0, verbose=False)
        dt = time.perf_counter() - t0
        times.append(MAX_TOKENS / dt)
        if output is None:
            output = out

    avg = sum(times) / len(times)
    print(f"{avg:.1f} tok/s")
    del engine
    return avg, output


def main():
    print("=" * 70)
    print("  🏁 MegaGemm vs llama.cpp — CPU Benchmark")
    print("=" * 70)
    print(f"  Model:   Qwen2.5-0.5B-Instruct")
    print(f"  Prompt:  {PROMPT}")
    print(f"  Tokens:  {MAX_TOKENS}")
    print(f"  CPU:     {os.cpu_count()} cores")
    print()

    results = []

    # ── MegaGemm FP32 ──
    print("─" * 70)
    print("  MegaGemm")
    print("─" * 70)
    tps, out = bench_megagemm("fp32")
    results.append(("MegaGemm FP32", tps, out))

    tps, out = bench_megagemm("int8")
    results.append(("MegaGemm INT8", tps, out))

    # ── llama.cpp ──
    print()
    print("─" * 70)
    print("  llama.cpp")
    print("─" * 70)

    if install_llama_cpp():
        # Q8_0
        path = download_gguf(GGUF_FILE_Q8)
        if path:
            tps, out = bench_llama_cpp(path, "Q8_0")
            if tps:
                results.append(("llama.cpp Q8", tps, out))

        # Q4_K_M
        path = download_gguf(GGUF_FILE_Q4)
        if path:
            tps, out = bench_llama_cpp(path, "Q4_K_M")
            if tps:
                results.append(("llama.cpp Q4", tps, out))

    # ── Summary ──
    print()
    print("=" * 70)
    print("  🏁 Results")
    print("=" * 70)

    max_tps = max(r[1] for r in results if r[1]) if results else 1
    for name, tps, out in results:
        if tps:
            bar = "█" * int(tps / max_tps * 30)
            print(f"  {name:<18} │ {tps:>6.1f} tok/s │ {bar}")

    print()
    print("  Outputs:")
    for name, _, out in results:
        if out:
            print(f"  {name:<18}: {out[:55]}...")

    print("=" * 70)


if __name__ == "__main__":
    main()

"""
🔄 MegaGemm Layer Offload Test
-------------------------------
Tests the CPU/disk offloading feature with progressively larger models.

Test 1: LLaMA 3.2 1B — partial offload (regression check, should match full GPU)
Test 2: LLaMA 3.1 8B — partial offload (some layers on CPU)
Test 3: Qwen2.5-7B AWQ — partial offload with quantization

Usage (Colab):
  !python test_offload.py
"""

# Standalone multi-model/offload runner; execute this file directly on a GPU.
__test__ = False

import torch
import gc
import sys
import os
import time
import traceback

sys.path.insert(0, os.getcwd())

def test_offload(model_name, n_gpu_layers, prompts, max_tokens=40,
                 num_blocks=512, offload_dir=None):
    """Test a model with layer offloading."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"  n_gpu_layers={n_gpu_layers}, offload_dir={offload_dir}")
    print(f"{'='*60}")

    try:
        from megagemm.engine import InferenceEngine

        t0 = time.perf_counter()
        engine = InferenceEngine(
            model_name,
            num_blocks=num_blocks,
            block_size=16,
            n_gpu_layers=n_gpu_layers,
            offload_dir=offload_dir,
        )
        load_time = time.perf_counter() - t0

        # Print model config
        cfg = engine.model.config
        print(f"  Layers: {cfg.num_hidden_layers}, Hidden: {cfg.hidden_size}")
        print(f"  Load time: {load_time:.1f}s")

        # VRAM usage
        vram_mb = torch.cuda.memory_allocated() / 1024**2
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  VRAM: {vram_mb:.0f}MB / {vram_total:.0f}MB")

        # Offload info
        offloader = engine.model._offloader
        if offloader:
            print(f"  Offloader: {offloader}")
        else:
            print(f"  Offloader: None (all on GPU)")

        # Format prompts with chat template
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for prompt in prompts:
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted = prompt

            t0 = time.perf_counter()
            output = engine.generate(
                formatted,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            gen_time = time.perf_counter() - t0

            # Estimate tokens generated
            n_tokens = len(tokenizer.encode(output))
            tok_s = n_tokens / gen_time if gen_time > 0 else 0

            print(f"\n  Q: {prompt}")
            print(f"  A: {output[:200]}")
            print(f"  ⏱ {gen_time:.2f}s, ~{tok_s:.1f} tok/s")

        # Print offload stats
        if offloader:
            offloader.print_stats()

        # Cleanup
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n  Status: ✅ OK")
        return True

    except Exception as e:
        print(f"\n  ERROR: {e}")
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("MegaGemm Layer Offload Test")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name()
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu} ({vram_total:.1f}GB)")
    else:
        print("No GPU! Exiting.")
        sys.exit(1)

    prompts = ["What is the capital of France?"]
    results = {}

    # ── Test 1: LLaMA 3.2 1B — full GPU (baseline) ──
    results["1B-full-gpu"] = test_offload(
        "meta-llama/Llama-3.2-1B-Instruct",
        n_gpu_layers=-1,  # All on GPU
        prompts=prompts,
        num_blocks=512,
    )

    # ── Test 2: LLaMA 3.2 1B — partial offload (8 of 16 layers on CPU) ──
    gc.collect(); torch.cuda.empty_cache()
    results["1B-partial"] = test_offload(
        "meta-llama/Llama-3.2-1B-Instruct",
        n_gpu_layers=8,  # 8 on GPU, 8 on CPU
        prompts=prompts,
        num_blocks=512,
    )

    # ── Test 3: LLaMA 3.2 1B — full offload (all layers on CPU) ──
    gc.collect(); torch.cuda.empty_cache()
    results["1B-full-offload"] = test_offload(
        "meta-llama/Llama-3.2-1B-Instruct",
        n_gpu_layers=0,  # All on CPU
        prompts=prompts,
        num_blocks=512,
    )

    # ── Test 4: LLaMA 3.1 8B — partial offload ──
    gc.collect(); torch.cuda.empty_cache()
    results["8B-partial"] = test_offload(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        n_gpu_layers=16,  # 16 of 32 layers on GPU
        prompts=prompts,
        num_blocks=256,
    )

    # ── Test 5: Qwen2.5-7B AWQ — partial offload ──
    gc.collect(); torch.cuda.empty_cache()
    results["7B-AWQ-partial"] = test_offload(
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
        n_gpu_layers=14,  # 14 of 28 layers on GPU
        prompts=prompts,
        num_blocks=512,
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print("OFFLOAD TEST SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")
    print()

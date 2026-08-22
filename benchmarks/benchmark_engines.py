"""
🏁 MegaGemm vs HuggingFace Benchmark
--------------------------------------
Fair comparison on the SAME model, prompt, and hardware.
Tests both FP16 and AWQ models.

Usage (Colab):
  # Cell 1: Setup
  %cd /content/drive/MyDrive/MGRrmsnorm
  !python setup.py build_ext --inplace
  !pip install huggingface_hub safetensors transformers autoawq

  # Cell 2: Run benchmark
  !python benchmark_engines.py
"""

import torch
import gc
import sys
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.getcwd())

# ── Config ──
MAX_NEW_TOKENS = 128
WARMUP_RUNS = 2
BENCH_RUNS = 3

PROMPTS = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute the Fibonacci sequence.",
]

# Models to test (in order)
MODELS = [
    {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "label": "LLaMA-1B (FP16)",
        "is_awq": False,
    },
    {
        "name": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "label": "Qwen-7B (AWQ INT4)",
        "is_awq": True,
    },
]


@dataclass
class BenchResult:
    engine: str
    model_label: str
    prompt: str
    tokens_generated: int
    time_s: float
    tok_s: float


@dataclass
class EngineSummary:
    engine: str
    model_label: str
    avg_tok_s: float
    vram_mb: float
    results: List[BenchResult] = field(default_factory=list)


def format_prompt(prompt: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return prompt


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ════════════════════════════════════════════════
# MegaGemm
# ════════════════════════════════════════════════
def bench_megagemm(model_cfg: dict) -> Optional[EngineSummary]:
    model_name = model_cfg["name"]
    label = model_cfg["label"]

    print(f"\n  🔥 MegaGemm — {label}")
    print(f"  {'-'*50}")

    try:
        from megagemm.engine import InferenceEngine
        from transformers import AutoTokenizer

        engine = InferenceEngine(model_name, num_blocks=512, block_size=16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vram = torch.cuda.memory_allocated() / 1024**2

        results = []

        # Warmup
        for _ in range(WARMUP_RUNS):
            fmt = format_prompt("Hi", tokenizer)
            engine.generate(fmt, max_new_tokens=10, temperature=0.7)

        # Benchmark
        for run in range(BENCH_RUNS):
            for prompt in PROMPTS:
                formatted = format_prompt(prompt, tokenizer)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                output = engine.generate(
                    formatted, max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.7, top_k=50, top_p=0.9,
                    repetition_penalty=1.1,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                n_tok = len(tokenizer.encode(output))
                tok_s = n_tok / elapsed if elapsed > 0 else 0

                results.append(BenchResult(
                    engine="MegaGemm", model_label=label, prompt=prompt,
                    tokens_generated=n_tok, time_s=elapsed, tok_s=tok_s,
                ))

                if run == 0:
                    print(f"    {prompt[:45]:.<48} {tok_s:>6.1f} tok/s ({n_tok} tok)")

        total_tok = sum(r.tokens_generated for r in results)
        total_time = sum(r.time_s for r in results)
        avg = total_tok / total_time if total_time > 0 else 0

        print(f"    {'Average':.<48} {avg:>6.1f} tok/s  |  VRAM: {vram:.0f}MB")

        del engine
        cleanup()

        return EngineSummary(
            engine="MegaGemm", model_label=label,
            avg_tok_s=avg, vram_mb=vram, results=results,
        )

    except Exception as e:
        print(f"    ❌ Failed: {e}")
        import traceback; traceback.print_exc()
        cleanup()
        return None


# ════════════════════════════════════════════════
# HuggingFace transformers
# ════════════════════════════════════════════════
def bench_huggingface(model_cfg: dict) -> Optional[EngineSummary]:
    model_name = model_cfg["name"]
    label = model_cfg["label"]
    is_awq = model_cfg["is_awq"]

    print(f"\n  🤗 HuggingFace — {label}")
    print(f"  {'-'*50}")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model — handle AWQ vs FP16
        if is_awq:
            # Try autoawq's own loader (more reliable than transformers for AWQ)
            try:
                from awq import AutoAWQForCausalLM
                model = AutoAWQForCausalLM.from_quantized(
                    model_name,
                    fuse_layers=True,
                    device_map="auto",
                )
                # Get the actual HF model for generate()
                hf_model = model.model
                print(f"    (loaded via AutoAWQ with fused layers)")
            except Exception as e1:
                # Fallback: try transformers directly
                try:
                    from transformers import AutoModelForCausalLM
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        model_name, dtype=torch.float16, device_map="auto",
                    )
                    print(f"    (loaded via transformers)")
                except Exception as e2:
                    print(f"    ❌ Cannot load AWQ model:")
                    print(f"       AutoAWQ: {e1}")
                    print(f"       Transformers: {e2}")
                    return None
        else:
            from transformers import AutoModelForCausalLM
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float16, device_map="auto",
            )

        vram = torch.cuda.memory_allocated() / 1024**2

        results = []

        # Warmup
        for _ in range(WARMUP_RUNS):
            fmt = format_prompt("Hi", tokenizer)
            inputs = tokenizer(fmt, return_tensors="pt").to(hf_model.device)
            with torch.inference_mode():
                hf_model.generate(**inputs, max_new_tokens=10, do_sample=True,
                                  temperature=0.7)

        # Benchmark
        for run in range(BENCH_RUNS):
            for prompt in PROMPTS:
                formatted = format_prompt(prompt, tokenizer)
                inputs = tokenizer(formatted, return_tensors="pt").to(hf_model.device)
                input_len = inputs["input_ids"].shape[1]

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    output_ids = hf_model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True, temperature=0.7,
                        top_k=50, top_p=0.9,
                        repetition_penalty=1.1,
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                n_tok = len(output_ids[0]) - input_len
                tok_s = n_tok / elapsed if elapsed > 0 else 0

                results.append(BenchResult(
                    engine="HuggingFace", model_label=label, prompt=prompt,
                    tokens_generated=n_tok, time_s=elapsed, tok_s=tok_s,
                ))

                if run == 0:
                    print(f"    {prompt[:45]:.<48} {tok_s:>6.1f} tok/s ({n_tok} tok)")

        total_tok = sum(r.tokens_generated for r in results)
        total_time = sum(r.time_s for r in results)
        avg = total_tok / total_time if total_time > 0 else 0

        print(f"    {'Average':.<48} {avg:>6.1f} tok/s  |  VRAM: {vram:.0f}MB")

        del hf_model
        if is_awq and 'model' in dir():
            del model
        cleanup()

        return EngineSummary(
            engine="HuggingFace", model_label=label,
            avg_tok_s=avg, vram_mb=vram, results=results,
        )

    except Exception as e:
        print(f"    ❌ Failed: {e}")
        import traceback; traceback.print_exc()
        cleanup()
        return None


# ════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════
if __name__ == "__main__":
    print("🏁" + "="*58)
    print("  MegaGemm vs HuggingFace — Inference Benchmark")
    print("="*60)

    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "No GPU"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    print(f"GPU: {gpu} ({vram_gb:.1f}GB)")
    print(f"Max tokens: {MAX_NEW_TOKENS}, Warmup: {WARMUP_RUNS}, Runs: {BENCH_RUNS}")
    print(f"Prompts: {len(PROMPTS)}")

    all_summaries = []

    for model_cfg in MODELS:
        print(f"\n{'='*60}")
        print(f"📦 Model: {model_cfg['label']}")
        print(f"   {model_cfg['name']}")
        print(f"{'='*60}")

        # Run MegaGemm first
        mg = bench_megagemm(model_cfg)
        if mg:
            all_summaries.append(mg)
        cleanup()

        # Then HuggingFace
        hf = bench_huggingface(model_cfg)
        if hf:
            all_summaries.append(hf)
        cleanup()

        # Speed comparison for this model
        if mg and hf:
            ratio = mg.avg_tok_s / hf.avg_tok_s if hf.avg_tok_s > 0 else 0
            winner = "MegaGemm" if ratio >= 1 else "HuggingFace"
            factor = ratio if ratio >= 1 else (1/ratio if ratio > 0 else 0)
            vram_diff = hf.vram_mb - mg.vram_mb

            print(f"\n  ── {model_cfg['label']} Result ──")
            print(f"  MegaGemm:    {mg.avg_tok_s:>6.1f} tok/s  ({mg.vram_mb:.0f}MB)")
            print(f"  HuggingFace: {hf.avg_tok_s:>6.1f} tok/s  ({hf.vram_mb:.0f}MB)")
            if ratio >= 1:
                print(f"  🔥 MegaGemm is {factor:.1f}x FASTER")
            else:
                print(f"  🤗 HuggingFace is {factor:.1f}x faster")
            if abs(vram_diff) > 50:
                less = "MegaGemm" if vram_diff > 0 else "HuggingFace"
                print(f"  💾 {less} uses {abs(vram_diff):.0f}MB less VRAM")

    # ── Final Summary ──
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Engine':<15} {'tok/s':>8} {'VRAM':>8}")
    print("-" * 58)
    for s in all_summaries:
        print(f"{s.model_label:<25} {s.engine:<15} {s.avg_tok_s:>7.1f} {s.vram_mb:>7.0f}MB")

    print(f"\n✅ Benchmark complete!")

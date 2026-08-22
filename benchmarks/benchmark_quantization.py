"""
📊 MegaGemm Comprehensive Benchmark
=====================================
Three-phase benchmark comparing MegaGemm against HuggingFace Transformers:

  Phase 1 — HF vs MegaGemm: TPS, TTFT, VRAM (single request, same model)
  Phase 2 — Quantization:   FP16 vs INT8 vs AWQ (VRAM, TPS, cosine sim)
  Phase 3 — Continuous Batching: Multi-request throughput scaling

Usage (Colab / GPU):
    python benchmarks/benchmark_quantization.py --model Qwen/Qwen2.5-7B-Instruct
    python benchmarks/benchmark_quantization.py --model meta-llama/Llama-3.2-3B-Instruct --skip-awq

Author: Gabriel Yogi
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import time
import gc
import json

# ─── Constants ───
PROMPTS = [
    "What is machine learning?",
    "Explain the theory of relativity in simple terms.",
    "Write a Python function that sorts a list using quicksort.",
    "What are the main differences between TCP and UDP?",
    "Summarize the French Revolution in 3 sentences.",
]
MAX_NEW_TOKENS = 128
WARMUP_TOKENS = 32

# ─── Helpers ───

def get_vram_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2, torch.cuda.max_memory_allocated() / 1024**2

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0).float(), b.unsqueeze(0).float()
    ).item()


# ═══════════════════════════════════════════════════════════
#  PHASE 1: HF vs MegaGemm — Single Request Comparison
# ═══════════════════════════════════════════════════════════

def benchmark_hf(model_name: str, prompts: list, max_new_tokens: int):
    """Benchmark HuggingFace Transformers generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    clear_gpu()
    vram_before = torch.cuda.memory_allocated() / 1024**2

    print(f"\n  📦 Loading HuggingFace model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    load_time = time.time() - t0
    vram_model = torch.cuda.memory_allocated() / 1024**2 - vram_before

    print(f"  ✅ HF loaded in {load_time:.1f}s | Model VRAM: {vram_model:.0f} MB")

    # Warmup
    with torch.no_grad():
        warmup_ids = tokenizer("warmup", return_tensors="pt").to("cuda")
        model.generate(**warmup_ids, max_new_tokens=WARMUP_TOKENS, do_sample=False)
    torch.cuda.synchronize()

    # Benchmark generation
    total_tokens = 0
    total_time = 0
    ttft_list = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0,
            )

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        gen_len = out.shape[1] - prompt_len
        elapsed = t_end - t_start
        total_tokens += gen_len
        total_time += elapsed

        # Approximate TTFT (first forward pass time — no easy way to measure exactly in HF)
        # We'll estimate it from total time / tokens ratio
        ttft_est = elapsed / max(gen_len, 1)  # rough per-token time
        ttft_list.append(ttft_est * prompt_len)  # scale by prompt length

    avg_tps = total_tokens / total_time if total_time > 0 else 0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2

    # Also get reference logits for cosine sim
    ref_logits = {}
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(inputs["input_ids"])
            ref_logits[prompt] = out.logits[0, -1, :].float().cpu()

    # Cleanup
    del model
    clear_gpu()

    return {
        "tps": avg_tps,
        "model_vram_mb": vram_model,
        "peak_vram_mb": peak_vram,
        "load_time_s": load_time,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "ref_logits": ref_logits,
    }


def benchmark_megagemm_single(
    model_name: str, quantize: str, prompts: list,
    max_new_tokens: int, ref_logits: dict, label: str,
):
    """Benchmark MegaGemm single-request generation."""
    from megagemm.engine import InferenceEngine

    clear_gpu()
    vram_before = torch.cuda.memory_allocated() / 1024**2

    print(f"\n  📦 Loading MegaGemm ({label})...")
    t0 = time.time()
    engine = InferenceEngine(model_name, quantize=quantize, num_blocks=256)
    load_time = time.time() - t0

    vram_model = torch.cuda.memory_allocated() / 1024**2 - vram_before
    print(f"  ✅ MegaGemm loaded in {load_time:.1f}s | VRAM: {vram_model:.0f} MB")

    # Warmup
    engine.generate("warmup", max_new_tokens=WARMUP_TOKENS, temperature=0.0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark generation
    total_tokens = 0
    total_time = 0
    ttft_list = []

    for prompt in prompts:
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        output = engine.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=0.0,
        )

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        gen_len = len(engine.tokenizer.encode(output))
        elapsed = t_end - t_start
        total_tokens += gen_len
        total_time += elapsed

    avg_tps = total_tokens / total_time if total_time > 0 else 0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2

    # Cosine similarity vs HF reference
    cos_sims = []
    if ref_logits:
        bm = engine.block_manager
        mg_model = engine.model
        for i, prompt in enumerate(prompts):
            input_ids = engine.tokenizer.encode(prompt)
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids], device="cuda")
            else:
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor([input_ids], device="cuda")
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                input_ids = input_ids.to("cuda")

            seq_len = input_ids.shape[1]
            seq_id = 9000 + i
            bm.allocate_sequence(seq_id, seq_len)
            positions = torch.arange(seq_len, device="cuda").unsqueeze(0)

            with torch.no_grad():
                mg_logits = mg_model.prefill(input_ids, positions, bm, seq_id)
                mg_logits = mg_logits[0, -1, :].float().cpu()

            cos = cosine_similarity(ref_logits[prompt], mg_logits)
            cos_sims.append(cos)
            bm.free_sequence(seq_id)

    avg_cos = sum(cos_sims) / len(cos_sims) if cos_sims else None

    # Cleanup
    del engine
    clear_gpu()

    return {
        "label": label,
        "tps": avg_tps,
        "model_vram_mb": vram_model,
        "peak_vram_mb": peak_vram,
        "load_time_s": load_time,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "cosine_sim": avg_cos,
    }


# ═══════════════════════════════════════════════════════════
#  PHASE 3: Continuous Batching Throughput
# ═══════════════════════════════════════════════════════════

def benchmark_continuous_batching(model_name: str, quantize: str, max_new_tokens: int):
    """Benchmark continuous batching with increasing batch sizes."""
    from megagemm.engine import InferenceEngine

    clear_gpu()

    print(f"\n  📦 Loading MegaGemm for batch benchmark...")
    quant_label = quantize.upper() if quantize else "FP16"
    engine = InferenceEngine(model_name, quantize=quantize, num_blocks=512)

    # Warmup
    engine.generate("warmup", max_new_tokens=WARMUP_TOKENS, temperature=0.0)
    torch.cuda.synchronize()

    batch_results = []

    for batch_size in [1, 2, 4, 8]:
        batch_prompts = (PROMPTS * ((batch_size // len(PROMPTS)) + 1))[:batch_size]

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()

        if batch_size == 1:
            # Single request baseline
            output = engine.generate(
                batch_prompts[0], max_new_tokens=max_new_tokens, temperature=0.0,
            )
            outputs = [output]
        else:
            # Continuous batching
            outputs = engine.generate_batch(
                batch_prompts, max_new_tokens=max_new_tokens,
                temperature=0.0, verbose=False,
            )

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        total_tokens = sum(len(engine.tokenizer.encode(o)) for o in outputs)
        tps = total_tokens / elapsed if elapsed > 0 else 0
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2

        batch_results.append({
            "batch_size": batch_size,
            "total_tokens": total_tokens,
            "elapsed_s": elapsed,
            "throughput_tps": tps,
            "peak_vram_mb": peak_vram,
        })

        print(f"     batch={batch_size}: {tps:.1f} tok/s "
              f"({total_tokens} tokens in {elapsed:.1f}s) "
              f"| Peak VRAM: {peak_vram:.0f} MB")

    # Cleanup
    del engine
    clear_gpu()

    return batch_results


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MegaGemm Comprehensive Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--awq-model", type=str, default=None,
                        help="AWQ model name (auto-detected if not set)")
    parser.add_argument("--skip-awq", action="store_true",
                        help="Skip AWQ benchmark")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Max new tokens per prompt")
    args = parser.parse_args()

    awq_model = args.awq_model or args.model + "-AWQ"

    print("\n" + "█" * 70)
    print("  📊 MegaGemm Comprehensive Benchmark")
    print(f"  Model: {args.model}")
    if not args.skip_awq:
        print(f"  AWQ:   {awq_model}")
    print(f"  Prompts: {len(PROMPTS)} × {args.max_tokens} tokens")
    print("█" * 70)

    # ═══════════════════════════════════════════════════════
    #  PHASE 1: HF vs MegaGemm (FP16)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'▓' * 60}")
    print(f"  Phase 1: HuggingFace vs MegaGemm (FP16)")
    print(f"{'▓' * 60}")

    hf_result = benchmark_hf(args.model, PROMPTS, args.max_tokens)
    ref_logits = hf_result.pop("ref_logits")

    mg_fp16 = benchmark_megagemm_single(
        args.model, None, PROMPTS, args.max_tokens, ref_logits, "FP16"
    )

    print(f"\n  {'─' * 55}")
    print(f"  Phase 1 Results: HF vs MegaGemm (FP16 single-request)")
    print(f"  {'─' * 55}")
    print(f"  {'Engine':<15} {'Model VRAM':>12} {'TPS':>10} {'Cosine Sim':>12}")
    print(f"  {'─'*15} {'─'*12} {'─'*10} {'─'*12}")
    print(f"  {'HF Transformers':<15} {hf_result['model_vram_mb']:>10.0f} MB"
          f" {hf_result['tps']:>8.1f}/s {'(ref)':>12}")
    print(f"  {'MegaGemm FP16':<15} {mg_fp16['model_vram_mb']:>10.0f} MB"
          f" {mg_fp16['tps']:>8.1f}/s {mg_fp16['cosine_sim']:>12.6f}")

    speedup = mg_fp16['tps'] / hf_result['tps'] if hf_result['tps'] > 0 else 0
    vram_ratio = mg_fp16['model_vram_mb'] / hf_result['model_vram_mb'] if hf_result['model_vram_mb'] > 0 else 0
    print(f"\n  ⚡ MegaGemm speedup: {speedup:.2f}x vs HF")
    print(f"  💾 VRAM ratio: {vram_ratio:.2f}x vs HF")

    # ═══════════════════════════════════════════════════════
    #  PHASE 2: Quantization Comparison
    # ═══════════════════════════════════════════════════════
    print(f"\n{'▓' * 60}")
    print(f"  Phase 2: Quantization (FP16 vs INT8 vs AWQ)")
    print(f"{'▓' * 60}")

    mg_int8 = benchmark_megagemm_single(
        args.model, "int8", PROMPTS, args.max_tokens, ref_logits, "INT8"
    )

    mg_awq = None
    if not args.skip_awq:
        try:
            mg_awq = benchmark_megagemm_single(
                awq_model, None, PROMPTS, args.max_tokens, ref_logits, "AWQ"
            )
        except Exception as e:
            print(f"  ⚠️ AWQ failed: {e}")

    quant_results = [mg_fp16, mg_int8]
    if mg_awq:
        quant_results.append(mg_awq)

    print(f"\n  {'─' * 70}")
    print(f"  Phase 2 Results: Quantization Comparison")
    print(f"  {'─' * 70}")
    print(f"  {'Mode':<12} {'Model VRAM':>12} {'Compression':>13} {'TPS':>10} {'Cosine Sim':>12} {'Quality':>10}")
    print(f"  {'─'*12} {'─'*12} {'─'*13} {'─'*10} {'─'*12} {'─'*10}")

    fp16_vram = mg_fp16['model_vram_mb']
    for r in quant_results:
        compress = fp16_vram / r['model_vram_mb'] if r['model_vram_mb'] > 0 else 0
        cos_str = f"{r['cosine_sim']:.6f}" if r['cosine_sim'] else "N/A"
        quality = "🟢" if r['cosine_sim'] and r['cosine_sim'] > 0.999 else (
            "🟡" if r['cosine_sim'] and r['cosine_sim'] > 0.99 else (
            "🟠" if r['cosine_sim'] and r['cosine_sim'] > 0.95 else "🔴"))
        print(f"  {r['label']:<12} {r['model_vram_mb']:>10.0f} MB {compress:>11.1f}x"
              f" {r['tps']:>8.1f}/s {cos_str:>12} {quality:>10}")

    # ═══════════════════════════════════════════════════════
    #  PHASE 3: Continuous Batching
    # ═══════════════════════════════════════════════════════
    print(f"\n{'▓' * 60}")
    print(f"  Phase 3: Continuous Batching Throughput (FP16)")
    print(f"{'▓' * 60}")

    batch_results = benchmark_continuous_batching(
        args.model, None, args.max_tokens
    )

    print(f"\n  {'─' * 55}")
    print(f"  Phase 3 Results: Continuous Batching Scaling")
    print(f"  {'─' * 55}")
    base_tps = batch_results[0]['throughput_tps']
    print(f"  {'Batch':>7} {'Throughput':>12} {'Scaling':>10} {'Peak VRAM':>12}")
    print(f"  {'─'*7} {'─'*12} {'─'*10} {'─'*12}")
    for r in batch_results:
        scaling = r['throughput_tps'] / base_tps if base_tps > 0 else 0
        print(f"  {r['batch_size']:>7} {r['throughput_tps']:>10.1f}/s {scaling:>8.1f}x"
              f" {r['peak_vram_mb']:>10.0f} MB")

    # ═══════════════════════════════════════════════════════
    #  FINAL SUMMARY + README MARKDOWN
    # ═══════════════════════════════════════════════════════
    model_short = args.model.split("/")[-1]

    print(f"\n{'═' * 70}")
    print(f"  📋 README Markdown — Copy below:")
    print(f"{'═' * 70}\n")

    # Table 1: HF vs MegaGemm
    print("### HF vs MegaGemm — Single Request\n")
    print(f"| Engine | Model VRAM | TPS | Cosine Sim |")
    print(f"|--------|-----------|-----|------------|")
    print(f"| HF Transformers | {hf_result['model_vram_mb']:.0f} MB | "
          f"{hf_result['tps']:.1f} tok/s | (reference) |")
    print(f"| **MegaGemm FP16** | {mg_fp16['model_vram_mb']:.0f} MB | "
          f"**{mg_fp16['tps']:.1f} tok/s** | {mg_fp16['cosine_sim']:.6f} |")
    if speedup > 1:
        print(f"\n> ⚡ MegaGemm is **{speedup:.1f}x faster** than HuggingFace Transformers")
    else:
        print(f"\n> MegaGemm speedup: {speedup:.2f}x vs HF")

    # Table 2: Quantization
    print("\n### Quantization Comparison\n")
    print("| Mode | Model VRAM | Compression | TPS | Cosine Sim | Quality |")
    print("|------|-----------|-------------|-----|------------|---------|")
    for r in quant_results:
        compress = fp16_vram / r['model_vram_mb'] if r['model_vram_mb'] > 0 else 0
        cos_str = f"{r['cosine_sim']:.6f}" if r['cosine_sim'] else "N/A"
        quality = "🟢 Excellent" if r['cosine_sim'] and r['cosine_sim'] > 0.999 else (
            "🟢 Good" if r['cosine_sim'] and r['cosine_sim'] > 0.99 else (
            "🟠 Acceptable" if r['cosine_sim'] and r['cosine_sim'] > 0.95 else "🔴 Degraded"))
        print(f"| **{r['label']}** | {r['model_vram_mb']:.0f} MB | {compress:.1f}x | "
              f"{r['tps']:.1f} tok/s | {cos_str} | {quality} |")

    # Table 3: Continuous Batching
    print("\n### Continuous Batching Scaling\n")
    print("| Batch Size | Throughput | Scaling | Peak VRAM |")
    print("|-----------|-----------|---------|-----------|")
    for r in batch_results:
        scaling = r['throughput_tps'] / base_tps if base_tps > 0 else 0
        print(f"| {r['batch_size']} | {r['throughput_tps']:.1f} tok/s | "
              f"{scaling:.1f}x | {r['peak_vram_mb']:.0f} MB |")

    print(f"\n{'═' * 70}")

    # Save JSON
    results = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(),
        "hf": {k: v for k, v in hf_result.items()},
        "megagemm_fp16": mg_fp16,
        "megagemm_int8": mg_int8,
        "megagemm_awq": mg_awq,
        "continuous_batching": batch_results,
    }
    out_file = f"benchmark_{model_short}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  💾 Results saved to {out_file}")


if __name__ == "__main__":
    main()

"""
📊 MegaGemm — KV Offload Equivalence Benchmark
================================================
Compares MegaGemm baseline (GPU-only KV) vs TieredBlockManager (GPU+CPU KV)
at the logit and generation level to verify offloading doesn't degrade quality.

Metrics:
  - Logit cosine similarity (prefill phase)
  - Top-1 / Top-5 token agreement
  - Greedy generation token match (decode phase)
  - Per-prompt divergence analysis

Run on Colab:
    !python benchmarks/benchmark_equivalence_offload.py
    !python benchmarks/benchmark_equivalence_offload.py --model Qwen/Qwen2.5-7B-Instruct \\
        --gpu-blocks 128 --cpu-blocks 2048

Author: Gabriel Yogi
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import time
import gc
import json
from typing import Dict, List

# ─── Prompts ───
PROMPTS_SHORT = [
    "What is machine learning?",
    "Explain the theory of relativity in simple terms.",
    "Write a Python function that sorts a list using quicksort.",
    "What are the main differences between TCP and UDP?",
    "Summarize the French Revolution in 3 sentences.",
]

PROMPT_LONG = (
    "Below is a detailed history of artificial intelligence. "
    "The field of AI has a rich history spanning decades of research. "
    "In 1950, Alan Turing published 'Computing Machinery and Intelligence'. "
    "The Dartmouth Conference of 1956 founded AI as a field. Early work "
    "focused on symbolic reasoning and game playing. The 1960s-70s saw "
    "expert systems and early neural networks. The Perceptron showed promise "
    "but limitations led to the first AI winter. The 1980s brought MYCIN and "
    "XCON. Backpropagation was popularized in 1986. The 2000s brought SVMs "
    "and random forests. In 2006, deep belief networks reignited deep learning. "
    "AlexNet in 2012 proved deep learning could outperform traditional methods. "
    "Transformers in 2017 revolutionized NLP. BERT, GPT and successors showed "
    "scaling produces capable systems. Today GPT-4, Claude, Gemini and LLaMA "
    "push the boundaries. Based on this history, explain the three most "
    "important breakthroughs in AI and predict the next major breakthrough."
)

MAX_NEW_TOKENS = 100
BLOCK_SIZE = 16


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_benchmark(
    model_name: str,
    gpu_blocks: int = 256,
    cpu_blocks: int = 512,
    gpu_window: int = 32,
    include_long: bool = True,
):
    """Compare baseline MegaGemm vs offloaded MegaGemm."""

    from megagemm.engine import InferenceEngine

    prompts = PROMPTS_SHORT.copy()
    if include_long:
        prompts.append(PROMPT_LONG)

    total_blocks_baseline = gpu_blocks + cpu_blocks  # all on GPU for baseline

    print("\n" + "=" * 70)
    print("  📊 MegaGemm KV Offload — Equivalence Benchmark")
    print(f"  Model: {model_name}")
    print(f"  Baseline: {total_blocks_baseline} GPU blocks")
    print(f"  Offload:  {gpu_blocks} GPU + {cpu_blocks} CPU blocks (window={gpu_window})")
    print(f"  Prompts:  {len(prompts)} ({len(PROMPTS_SHORT)} short" +
          (f" + 1 long)" if include_long else ")"))
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    #  STEP 1: Baseline (GPU-only) — collect logits + generations
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"📦 Step 1: Loading BASELINE engine (GPU-only, {total_blocks_baseline} blocks)...")
    print(f"{'─'*60}")

    t0 = time.time()
    engine_base = InferenceEngine(
        model_name, dtype=torch.float16,
        num_blocks=total_blocks_baseline, block_size=BLOCK_SIZE,
    )
    print(f"  ✅ Baseline loaded in {time.time()-t0:.1f}s")

    # Warmup
    _ = engine_base.generate("Hello", max_new_tokens=3, temperature=0.0)

    # Collect logits via prefill
    tokenizer = engine_base.tokenizer
    base_model = engine_base.model
    bm_base = engine_base.block_manager
    base_logits = {}

    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt)
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], device="cuda")
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0).to("cuda")

        seq_len = input_ids.shape[1]
        seq_id = 9000 + i
        bm_base.allocate_sequence(seq_id, seq_len + MAX_NEW_TOKENS)
        positions = torch.arange(seq_len, device="cuda").unsqueeze(0)

        with torch.no_grad():
            logits = base_model.prefill(input_ids, positions, bm_base, seq_id)
            base_logits[i] = logits[0, -1, :].float().cpu()

        bm_base.free_sequence(seq_id)

    # Collect generations (greedy)
    base_generations = {}
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        text = engine_base.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
        base_generations[i] = {"text": text, "time": time.time() - t0}

    print(f"  ✅ Baseline data collected for {len(prompts)} prompts")

    alloc_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  GPU mem: {alloc_mb:.0f}MB")

    del engine_base, base_model, bm_base
    clear_gpu()
    print(f"  🗑️ Baseline unloaded")

    # ══════════════════════════════════════════════════════════════════
    #  STEP 2: Offloaded (GPU+CPU) — collect logits + generations
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"🔄 Step 2: Loading OFFLOAD engine ({gpu_blocks} GPU + {cpu_blocks} CPU)...")
    print(f"{'─'*60}")

    t0 = time.time()
    engine_offload = InferenceEngine(
        model_name, dtype=torch.float16,
        num_blocks=gpu_blocks, block_size=BLOCK_SIZE,
        kv_offload=True, num_cpu_blocks=cpu_blocks, gpu_window=gpu_window,
    )
    print(f"  ✅ Offload engine loaded in {time.time()-t0:.1f}s")

    # Warmup
    _ = engine_offload.generate("Hello", max_new_tokens=3, temperature=0.0)

    off_model = engine_offload.model
    bm_off = engine_offload.block_manager
    off_logits = {}

    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt)
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], device="cuda")
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0).to("cuda")

        seq_len = input_ids.shape[1]
        seq_id = 9000 + i
        bm_off.allocate_sequence(seq_id, seq_len + MAX_NEW_TOKENS)

        # Ensure blocks on GPU before prefill
        if hasattr(bm_off, 'ensure_blocks_on_gpu'):
            bm_off.ensure_blocks_on_gpu([seq_id])

        positions = torch.arange(seq_len, device="cuda").unsqueeze(0)

        with torch.no_grad():
            logits = off_model.prefill(input_ids, positions, bm_off, seq_id)
            off_logits[i] = logits[0, -1, :].float().cpu()

        bm_off.free_sequence(seq_id)

    # Collect generations (greedy)
    off_generations = {}
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        text = engine_offload.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
        off_generations[i] = {"text": text, "time": time.time() - t0}

    print(f"  ✅ Offload data collected for {len(prompts)} prompts")

    alloc_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  GPU mem: {alloc_mb:.0f}MB")

    # Print offload stats
    bm_off.print_stats()

    del engine_offload, off_model, bm_off
    clear_gpu()
    print(f"  🗑️ Offload engine unloaded")

    # ══════════════════════════════════════════════════════════════════
    #  STEP 3: Compare logits
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("📐 Phase 1: Logit Cosine Similarity (Prefill)")
    print(f"{'─'*60}")

    logit_results = []
    for i, prompt in enumerate(prompts):
        bl = base_logits[i]
        ol = off_logits[i]

        cos_sim = torch.nn.functional.cosine_similarity(
            bl.unsqueeze(0), ol.unsqueeze(0)
        ).item()

        base_top5 = bl.topk(5).indices.tolist()
        off_top5 = ol.topk(5).indices.tolist()
        top1_match = base_top5[0] == off_top5[0]
        top5_overlap = len(set(base_top5) & set(off_top5))

        # Max absolute difference
        max_diff = (bl - ol).abs().max().item()

        label = prompt[:55] + "..." if len(prompt) > 55 else prompt
        result = {
            "prompt": label,
            "cosine_similarity": cos_sim,
            "top1_match": top1_match,
            "top5_overlap": top5_overlap,
            "max_logit_diff": max_diff,
            "base_top1": tokenizer.decode(base_top5[0]),
            "off_top1": tokenizer.decode(off_top5[0]),
        }
        logit_results.append(result)

        sym = "✅" if cos_sim > 0.9999 else ("⚠️" if cos_sim > 0.999 else "❌")
        print(f"  {sym} cos={cos_sim:.6f} "
              f"top1={'✓' if top1_match else '✗'} "
              f"top5={top5_overlap}/5 "
              f"maxΔ={max_diff:.4f} "
              f"│ {label[:45]}")

    avg_cos = sum(r["cosine_similarity"] for r in logit_results) / len(logit_results)
    top1_rate = sum(r["top1_match"] for r in logit_results) / len(logit_results)
    avg_top5 = sum(r["top5_overlap"] for r in logit_results) / len(logit_results)

    print(f"\n  📊 Avg Cosine Similarity: {avg_cos:.6f}")
    print(f"  📊 Top-1 Agreement:      {top1_rate:.0%}")
    print(f"  📊 Avg Top-5 Overlap:    {avg_top5:.1f}/5")

    # ══════════════════════════════════════════════════════════════════
    #  STEP 4: Compare generations
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("📝 Phase 2: Greedy Generation Comparison")
    print(f"{'─'*60}")

    gen_results = []
    for i, prompt in enumerate(prompts):
        base_text = base_generations[i]["text"]
        off_text = off_generations[i]["text"]
        base_time = base_generations[i]["time"]
        off_time = off_generations[i]["time"]

        # Token-level comparison
        base_tok = tokenizer.encode(base_text)
        off_tok = tokenizer.encode(off_text)
        min_len = min(len(base_tok), len(off_tok))

        matches = sum(1 for j in range(min_len) if base_tok[j] == off_tok[j])
        agreement = matches / min_len if min_len > 0 else 0

        # Find first divergence
        diverge_at = min_len
        for j in range(min_len):
            if base_tok[j] != off_tok[j]:
                diverge_at = j
                break

        exact_match = base_text == off_text
        label = prompt[:50] + "..." if len(prompt) > 50 else prompt

        result = {
            "prompt": label,
            "exact_match": exact_match,
            "token_agreement": agreement,
            "diverge_at_token": diverge_at,
            "base_tokens": len(base_tok),
            "off_tokens": len(off_tok),
            "base_time_s": round(base_time, 3),
            "off_time_s": round(off_time, 3),
            "speedup": round(base_time / off_time, 2) if off_time > 0 else 0,
        }
        gen_results.append(result)

        sym = "✅" if exact_match else ("⚠️" if agreement > 0.9 else "❌")
        print(f"\n  {sym} {label}")
        print(f"     Agreement: {agreement:.1%} "
              f"(diverge@tok {diverge_at}) "
              f"{'EXACT MATCH' if exact_match else ''}")
        print(f"     Base ({base_time:.2f}s): {base_text[:90].replace(chr(10), '↵')}...")
        print(f"     Offl ({off_time:.2f}s): {off_text[:90].replace(chr(10), '↵')}...")

    avg_agreement = sum(r["token_agreement"] for r in gen_results) / len(gen_results)
    avg_diverge = sum(r["diverge_at_token"] for r in gen_results) / len(gen_results)
    exact_rate = sum(r["exact_match"] for r in gen_results) / len(gen_results)

    print(f"\n  📊 Exact Match Rate:     {exact_rate:.0%}")
    print(f"  📊 Avg Token Agreement:  {avg_agreement:.1%}")
    print(f"  📊 Avg Divergence Point: token {avg_diverge:.0f}")

    # ══════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  📋 SUMMARY — {model_name}")
    print(f"  Baseline: {total_blocks_baseline} GPU blocks")
    print(f"  Offload:  {gpu_blocks} GPU + {cpu_blocks} CPU (window={gpu_window})")
    print(f"{'='*70}")
    print(f"  Logit Cosine Similarity:    {avg_cos:.6f}")
    print(f"  Top-1 Token Agreement:      {top1_rate:.0%}")
    print(f"  Top-5 Overlap:              {avg_top5:.1f}/5")
    print(f"  Generation Exact Match:     {exact_rate:.0%}")
    print(f"  Generation Token Agreement: {avg_agreement:.1%}")
    print(f"  Avg Divergence Point:       token {avg_diverge:.0f}")

    if avg_cos > 0.9999 and exact_rate == 1.0:
        print(f"\n  🟢 PERFECT — Offload output is bit-identical to baseline")
    elif avg_cos > 0.999:
        print(f"\n  🟡 EXCELLENT — Near-identical (tiny floating-point differences)")
    elif avg_cos > 0.99:
        print(f"\n  🟠 GOOD — Minor differences (likely from block eviction during prefill)")
    else:
        print(f"\n  🔴 DEGRADED — Significant differences detected")

    print(f"{'='*70}\n")

    return {
        "model": model_name,
        "config": {
            "gpu_blocks": gpu_blocks,
            "cpu_blocks": cpu_blocks,
            "gpu_window": gpu_window,
            "baseline_blocks": total_blocks_baseline,
        },
        "logit_cosine_sim": round(avg_cos, 6),
        "top1_agreement": round(top1_rate, 3),
        "top5_overlap": round(avg_top5, 1),
        "gen_exact_match": round(exact_rate, 3),
        "gen_token_agreement": round(avg_agreement, 3),
        "avg_diverge_token": round(avg_diverge, 1),
        "per_prompt_logits": logit_results,
        "per_prompt_gen": gen_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MegaGemm KV Offload Equivalence Benchmark"
    )
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model name")
    parser.add_argument("--gpu-blocks", type=int, default=256,
                        help="GPU KV cache blocks for offload engine")
    parser.add_argument("--cpu-blocks", type=int, default=512,
                        help="CPU KV cache blocks for offload engine")
    parser.add_argument("--gpu-window", type=int, default=32,
                        help="Blocks to keep on GPU per sequence")
    parser.add_argument("--no-long", action="store_true",
                        help="Skip long prompt test")
    args = parser.parse_args()

    results = run_benchmark(
        args.model,
        gpu_blocks=args.gpu_blocks,
        cpu_blocks=args.cpu_blocks,
        gpu_window=args.gpu_window,
        include_long=not args.no_long,
    )

    # Save results
    model_short = args.model.split("/")[-1]
    out_file = f"equivalence_offload_{model_short}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📁 Results saved to {out_file}")

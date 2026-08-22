"""
📊 MegaGemm vs HuggingFace — Quality Equivalence Benchmark
============================================================
Compares MegaGemm inference against HuggingFace Transformers reference:
  - Logit cosine similarity (prefill)
  - Token-level agreement
  - Output text comparison

Usage (Colab / GPU):
    python benchmarks/benchmark_equivalence.py --model Qwen/Qwen2.5-7B-Instruct --quantize int8
    python benchmarks/benchmark_equivalence.py --model meta-llama/Llama-3.2-3B-Instruct

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
from typing import Dict, List

# ─── Config ───
PROMPTS = [
    "What is machine learning?",
    "Explain the theory of relativity in simple terms.",
    "Write a Python function that sorts a list using quicksort.",
    "What are the main differences between TCP and UDP?",
    "Summarize the French Revolution in 3 sentences.",
]

MAX_NEW_TOKENS = 100


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def maybe_format_chat_prompt(tokenizer, prompt: str) -> str:
    """
    Mirror MegaGemm's prompt preparation so HF and native generation compare
    the same effective context for chat-tuned models.
    """
    formatted_prompt = prompt
    bos = getattr(tokenizer, "bos_token", None)
    already_formatted = bool(bos and prompt.startswith(bos))

    if (
        not already_formatted
        and hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template
    ):
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted_prompt = prompt
    return formatted_prompt


def run_benchmark(model_name: str, quantize: str = None):
    """Run full equivalence benchmark — sequential loading to avoid OOM."""

    print("\n" + "=" * 70)
    print("  🔬 MegaGemm vs HuggingFace — Equivalence Benchmark")
    print(f"  Model: {model_name}")
    print(f"  Quantize: {quantize or 'FP16'}")
    print("=" * 70)

    # ══════════════════════════════════════════════
    #  STEP 1: Load HF, collect logits + generations, UNLOAD
    # ══════════════════════════════════════════════
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'─'*60}")
    print(f"📦 Step 1: Loading HF reference & collecting data...")
    print(f"{'─'*60}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted_prompts = {
        prompt: maybe_format_chat_prompt(tokenizer, prompt)
        for prompt in PROMPTS
    }
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="cuda",
    )
    hf_model.eval()
    print(f"  ✅ HF loaded in {time.time()-t0:.1f}s")

    # Collect HF logits
    hf_logits = {}
    for prompt in PROMPTS:
        inputs = tokenizer(formatted_prompts[prompt], return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = hf_model(inputs["input_ids"])
            hf_logits[prompt] = out.logits[0, -1, :].float().cpu()

    # Collect HF generations (greedy)
    hf_generations = {}
    for prompt in PROMPTS:
        inputs = tokenizer(formatted_prompts[prompt], return_tensors="pt", add_special_tokens=False).to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = hf_model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, temperature=1.0,
            )
        hf_time = time.time() - t0
        prompt_len = inputs["input_ids"].shape[1]
        hf_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        hf_generations[prompt] = {"text": hf_text, "time": hf_time}

    print(f"  ✅ HF data collected for {len(PROMPTS)} prompts")

    # UNLOAD HF model to free VRAM
    del hf_model
    clear_gpu()
    print(f"  🗑️ HF model unloaded, VRAM freed")

    # ══════════════════════════════════════════════
    #  STEP 2: Load MegaGemm, compare
    # ══════════════════════════════════════════════
    from megagemm.engine import InferenceEngine

    print(f"\n{'─'*60}")
    print(f"🔥 Step 2: Loading MegaGemm ({quantize or 'FP16'})...")
    print(f"{'─'*60}")

    t0 = time.time()
    mg_engine = InferenceEngine(model_name, quantize=quantize, num_blocks=256)
    print(f"  ✅ MegaGemm loaded in {time.time()-t0:.1f}s")

    # ── Phase 1: Logit Comparison ──
    print(f"\n{'─'*60}")
    print("📐 Phase 1: Logit Cosine Similarity (Prefill)")
    print(f"{'─'*60}")

    mg_model = mg_engine.model
    bm = mg_engine.block_manager
    logit_results = []

    for i, prompt in enumerate(PROMPTS):
        # MegaGemm forward pass
        input_ids = tokenizer.encode(formatted_prompts[prompt], add_special_tokens=False)
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], device="cuda")
        else:
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor([input_ids], device="cuda")
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to("cuda")

        seq_len = input_ids.shape[1]
        seq_id = 8000 + i
        bm.allocate_sequence(seq_id, seq_len)
        positions = torch.arange(seq_len, device="cuda").unsqueeze(0)

        with torch.no_grad():
            mg_logits_raw = mg_model.prefill(input_ids, positions, bm, seq_id)
            mg_logits = mg_logits_raw[0, -1, :].float().cpu()

        bm.free_sequence(seq_id)

        # Compare
        cos_sim = torch.nn.functional.cosine_similarity(
            hf_logits[prompt].unsqueeze(0), mg_logits.unsqueeze(0)
        ).item()

        hf_top5 = hf_logits[prompt].topk(5).indices.tolist()
        mg_top5 = mg_logits.topk(5).indices.tolist()
        top1_match = hf_top5[0] == mg_top5[0]
        top5_overlap = len(set(hf_top5) & set(mg_top5))

        result = {
            "prompt": prompt[:60] + "..." if len(prompt) > 60 else prompt,
            "cosine_similarity": cos_sim,
            "top1_match": top1_match,
            "top5_overlap": top5_overlap,
            "hf_top1": tokenizer.decode(hf_top5[0]),
            "mg_top1": tokenizer.decode(mg_top5[0]),
        }
        logit_results.append(result)

        match_sym = "✅" if top1_match else "⚠️"
        print(f"  {match_sym} cos={cos_sim:.6f} "
              f"top1={top1_match} "
              f"top5={top5_overlap}/5 "
              f"│ {result['prompt'][:50]}")

    avg_cos = sum(r["cosine_similarity"] for r in logit_results) / len(logit_results)
    top1_rate = sum(r["top1_match"] for r in logit_results) / len(logit_results)
    avg_top5 = sum(r["top5_overlap"] for r in logit_results) / len(logit_results)

    print(f"\n  📊 Average Cosine Similarity: {avg_cos:.6f}")
    print(f"  📊 Top-1 Agreement Rate:      {top1_rate:.0%}")
    print(f"  📊 Average Top-5 Overlap:      {avg_top5:.1f}/5")

    # ── Phase 2: Generation Comparison ──
    print(f"\n{'─'*60}")
    print("📝 Phase 2: Greedy Generation Comparison")
    print(f"{'─'*60}")

    # Warm up native decode once so one-time kernel setup / shape heuristics
    # do not pollute the first timed comparison.
    _ = mg_engine.generate(
        PROMPTS[0],
        max_new_tokens=min(8, MAX_NEW_TOKENS),
        temperature=0.0,
        repetition_penalty=1.0,
    )

    gen_results = []
    for prompt in PROMPTS:
        # MegaGemm generation
        t0 = time.time()
        mg_text = mg_engine.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            repetition_penalty=1.0,
        )
        mg_time = time.time() - t0

        hf_text = hf_generations[prompt]["text"]
        hf_time = hf_generations[prompt]["time"]

        # Token-level comparison
        hf_tokens = tokenizer.encode(hf_text)
        mg_tokens = tokenizer.encode(mg_text)
        min_len = min(len(hf_tokens), len(mg_tokens))

        matches = sum(1 for i in range(min_len) if hf_tokens[i] == mg_tokens[i])
        token_agreement = matches / min_len if min_len > 0 else 0

        diverge_at = min_len
        for i in range(min_len):
            if hf_tokens[i] != mg_tokens[i]:
                diverge_at = i
                break

        result = {
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "hf_text": hf_text[:150].replace("\n", "↵"),
            "mg_text": mg_text[:150].replace("\n", "↵"),
            "token_agreement": token_agreement,
            "diverge_at_token": diverge_at,
            "hf_tokens": len(hf_tokens),
            "mg_tokens": len(mg_tokens),
            "hf_time_s": hf_time,
            "mg_time_s": mg_time,
        }
        gen_results.append(result)

        print(f"\n  Prompt: {result['prompt']}")
        print(f"  Token agreement: {token_agreement:.1%} "
              f"(diverge at token {diverge_at})")
        print(f"  HF  ({hf_time:.2f}s): {result['hf_text'][:100]}")
        print(f"  MG  ({mg_time:.2f}s): {result['mg_text'][:100]}")

    avg_agreement = sum(r["token_agreement"] for r in gen_results) / len(gen_results)
    avg_diverge = sum(r["diverge_at_token"] for r in gen_results) / len(gen_results)

    print(f"\n  📊 Average Token Agreement: {avg_agreement:.1%}")
    print(f"  📊 Average Divergence Point: token {avg_diverge:.0f}")

    # ── Summary ──
    quant_label = quantize.upper() if quantize else "FP16"

    print(f"\n{'='*70}")
    print(f"  📋 SUMMARY — {model_name} ({quant_label})")
    print(f"{'='*70}")
    print(f"  Logit Cosine Similarity:   {avg_cos:.6f}")
    print(f"  Top-1 Token Agreement:     {top1_rate:.0%}")
    print(f"  Top-5 Overlap:             {avg_top5:.1f}/5")
    print(f"  Generation Token Match:    {avg_agreement:.1%}")
    print(f"  Avg Divergence Point:      token {avg_diverge:.0f}")

    if avg_cos > 0.999:
        print(f"\n  🟢 EXCELLENT — Near-identical to HuggingFace reference")
    elif avg_cos > 0.99:
        print(f"\n  🟡 GOOD — Very close to HuggingFace reference")
    elif avg_cos > 0.95:
        print(f"\n  🟠 ACCEPTABLE — Minor logit differences (expected with quantization)")
    else:
        print(f"\n  🔴 DEGRADED — Significant differences detected")

    print(f"{'='*70}\n")

    # Cleanup
    del mg_engine
    clear_gpu()

    return {
        "model": model_name,
        "quantize": quant_label,
        "logit_cosine_sim": avg_cos,
        "top1_agreement": top1_rate,
        "top5_overlap": avg_top5,
        "gen_token_agreement": avg_agreement,
        "avg_diverge_token": avg_diverge,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MegaGemm vs HuggingFace Equivalence Benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--quantize", type=str, default=None, choices=["int8", "fp8"],
                        help="Quantization mode (default: none/FP16)")
    args = parser.parse_args()

    results = run_benchmark(args.model, args.quantize)

    # Save results
    out_file = f"equivalence_{args.model.split('/')[-1]}_{results['quantize']}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

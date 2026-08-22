"""
🔬 Comprehensive Prefill Validation for MegaGemm
==================================================
Validates MegaGemm inference against HuggingFace ground truth.

Tests:
  1. Text comparison: MegaGemm sequential vs HuggingFace (greedy)
  2. Text comparison: MegaGemm batched vs HuggingFace (greedy)
  3. MegaGemm sequential vs batched (internal consistency)
  4. Varying output lengths (5, 20, 50 tokens)
  5. Varying prompt lengths (short, medium, long)

Usage (Kaggle/Colab):
    python examples/validate_prefill.py
    python examples/validate_prefill.py --model Qwen/Qwen2.5-1.5B-Instruct
    python examples/validate_prefill.py --model mistralai/Mistral-7B-Instruct-v0.3

Author: Gabriel Yogi
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def run_validation(model_name: str, quantize: str = None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from megagemm.engine import InferenceEngine

    print(f"\n{'='*70}")
    print(f"  🔬 Comprehensive Prefill Validation")
    print(f"  Model: {model_name}")
    print(f"{'='*70}")

    # ── Prompts with varying lengths ──
    prompts = {
        'short': "What is 2+2?",
        'medium': "Classify the sentiment of this review: I absolutely love this product, it exceeded all my expectations and I would recommend it to everyone!",
        'long': "Summarize the following text in one sentence: Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Unlike traditional programming where rules are explicitly coded, machine learning algorithms use statistical techniques to give computers the ability to learn from experience. The field has grown rapidly in recent years, driven by increases in computing power, availability of large datasets, and algorithmic improvements.",
    }

    max_tokens_list = [5, 20, 50]

    # ══════════════════════════════════════════════════════
    # PHASE 1: HuggingFace ground truth (then free memory)
    # ══════════════════════════════════════════════════════
    print("\n📦 Phase 1: Loading HuggingFace model...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    hf_model.eval()
    print(f"   HF loaded in {time.perf_counter() - t0:.1f}s")

    # Generate all HF outputs
    hf_outputs = {}  # (prompt_name, max_tokens) -> text
    hf_times = {}    # (prompt_name, max_tokens) -> seconds
    for max_tokens in max_tokens_list:
        for prompt_name, prompt in prompts.items():
            # Apply chat template
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    formatted = prompt
            else:
                formatted = prompt

            bos = tokenizer.bos_token
            add_special = not (bos and formatted.startswith(bos))
            hf_input = tokenizer.encode(
                formatted, return_tensors='pt', add_special_tokens=add_special
            ).to(hf_model.device)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_hf = time.perf_counter()
            with torch.inference_mode():
                hf_output = hf_model.generate(
                    hf_input,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    repetition_penalty=1.0,
                )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_hf = time.perf_counter() - t_hf

            hf_generated = hf_output[0, hf_input.shape[1]:].tolist()
            hf_text = tokenizer.decode(hf_generated, skip_special_tokens=True).strip()
            hf_outputs[(prompt_name, max_tokens)] = hf_text
            hf_times[(prompt_name, max_tokens)] = t_hf

    print(f"   Generated {len(hf_outputs)} HF ground truths")

    # Free HF model
    del hf_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("   🗑️  HF model freed from VRAM")

    # ══════════════════════════════════════════════════════
    # PHASE 2: MegaGemm (loads into freed VRAM)
    # ══════════════════════════════════════════════════════
    print("\n📦 Phase 2: Loading MegaGemm engine...")
    t0 = time.perf_counter()
    engine = InferenceEngine(model_name, quantize=quantize)
    print(f"   MegaGemm loaded in {time.perf_counter() - t0:.1f}s")

    # ══════════════════════════════════════════════════════
    # PHASE 3: Compare
    # ══════════════════════════════════════════════════════
    hf_vs_seq_pass = 0
    hf_vs_bat_pass = 0
    seq_vs_bat_pass = 0
    total_tests = 0

    for max_tokens in max_tokens_list:
        print(f"\n{'─'*70}")
        print(f"  📊 Testing with max_tokens={max_tokens}")
        print(f"{'─'*70}")

        prompt_texts = list(prompts.values())

        # MegaGemm batched (all at once) — timed
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_bat_start = time.perf_counter()
        mg_bat_results = engine.generate_batch(
            prompt_texts, max_new_tokens=max_tokens,
            temperature=0.0, verbose=False,
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_bat = time.perf_counter() - t_bat_start

        # MegaGemm sequential — timed
        mg_seq_texts = []
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_seq_start = time.perf_counter()
        for prompt in prompt_texts:
            r = engine.generate(
                prompt, max_new_tokens=max_tokens,
                temperature=0.0, repetition_penalty=1.0, verbose=False,
            )
            mg_seq_texts.append((r[0] if isinstance(r, tuple) else r).strip())
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_seq = time.perf_counter() - t_seq_start

        for idx, (prompt_name, prompt) in enumerate(prompts.items()):
            total_tests += 1

            hf_text = hf_outputs[(prompt_name, max_tokens)]
            hf_time = hf_times[(prompt_name, max_tokens)]
            mg_seq_text = mg_seq_texts[idx]
            mg_bat_text = mg_bat_results[idx].strip()

            # TEXT comparison
            hf_seq_match = hf_text == mg_seq_text
            hf_bat_match = hf_text == mg_bat_text
            seq_bat_match = mg_seq_text == mg_bat_text

            if hf_seq_match: hf_vs_seq_pass += 1
            if hf_bat_match: hf_vs_bat_pass += 1
            if seq_bat_match: seq_vs_bat_pass += 1

            s1 = "✅" if hf_seq_match else "❌"
            s2 = "✅" if hf_bat_match else "❌"
            s3 = "✅" if seq_bat_match else "❌"

            print(f"\n   [{prompt_name}] max_tokens={max_tokens}")
            print(f"   ┌─ HF:     {hf_text}")
            print(f"   ├─ MG_seq: {mg_seq_text}")
            print(f"   └─ MG_bat: {mg_bat_text}")
            print(f"   Match: HF↔seq {s1} | HF↔bat {s2} | seq↔bat {s3}")

            if not hf_seq_match:
                for i, (a, b) in enumerate(zip(hf_text, mg_seq_text)):
                    if a != b:
                        print(f"   Diverges at char {i}: HF=...{repr(hf_text[max(0,i-5):i+10])} vs MG=...{repr(mg_seq_text[max(0,i-5):i+10])}")
                        break

        # Speed summary for this max_tokens
        n = len(prompts)
        hf_total = sum(hf_times[(k, max_tokens)] for k in prompts)
        print(f"\n   ⚡ Speed: {n} prompts × {max_tokens} max_tokens (total time to generate all {n})")
        print(f"      HuggingFace:       {hf_total*1000:7.0f}ms  ({hf_total/n*1000:.0f}ms/prompt)")
        print(f"      MegaGemm seq:      {t_seq*1000:7.0f}ms  ({t_seq/n*1000:.0f}ms/prompt)")
        print(f"      MegaGemm batched:  {t_bat*1000:7.0f}ms  ({t_bat/n*1000:.0f}ms/prompt)")
        print(f"      ─────────────────────────────────────────")
        # Who's fastest?
        times = {'HF': hf_total, 'MG_seq': t_seq, 'MG_bat': t_bat}
        fastest = min(times, key=times.get)
        for name, t in times.items():
            vs_fastest = t / times[fastest]
            marker = " 🏆" if name == fastest else ""
            print(f"      {name:20s} {vs_fastest:.1f}x{marker}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  📊 VALIDATION SUMMARY ({total_tests} tests)")
    print(f"{'='*70}")
    print(f"  HF vs MG_seq:    {hf_vs_seq_pass}/{total_tests} text-exact {'✅' if hf_vs_seq_pass == total_tests else '🟡'}")
    print(f"  HF vs MG_bat:    {hf_vs_bat_pass}/{total_tests} text-exact {'✅' if hf_vs_bat_pass == total_tests else '🟡'}")
    print(f"  MG_seq vs bat:   {seq_vs_bat_pass}/{total_tests} text-exact {'✅' if seq_vs_bat_pass == total_tests else '🔴'}")

    if seq_vs_bat_pass == total_tests:
        print(f"\n  ✅ INTERNAL CONSISTENCY: sequential == batched (all {total_tests} tests)")
    else:
        print(f"\n  🔴 INTERNAL ISSUE: sequential != batched in {total_tests - seq_vs_bat_pass} tests")

    if hf_vs_seq_pass == total_tests:
        print(f"  ✅ HF EQUIVALENCE: MegaGemm matches HuggingFace exactly!")
    elif hf_vs_seq_pass >= total_tests * 0.8:
        print(f"  🟡 HF CLOSE: minor differences (FP16 compute order, normal)")
    else:
        print(f"  🔴 HF DIVERGENCE: needs investigation")

    print(f"{'='*70}")

    return hf_vs_seq_pass, seq_vs_bat_pass, total_tests


def main():
    parser = argparse.ArgumentParser(description="MegaGemm Prefill Validation vs HuggingFace")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--quantize", choices=["int8"], default=None)
    args = parser.parse_args()

    run_validation(model_name=args.model, quantize=args.quantize)


if __name__ == "__main__":
    main()

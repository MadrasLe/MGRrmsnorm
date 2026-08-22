"""
⚡ MegaGemm Throughput Benchmark — tok/s with Batch Scaling
============================================================
Measures generation throughput in tokens/second, scaling batch sizes.

Tests HF vs MegaGemm sequential vs MegaGemm batched with:
  - Batch sizes: 1, 4, 8, 16, 32
  - Real-world prompt lengths (short, medium, long)
  - Fixed output length (max_tokens=30)
  - Reports: tok/s, prefill tok/s, decode tok/s

Usage:
    python examples/benchmark_throughput.py
    python examples/benchmark_throughput.py --model Qwen/Qwen2.5-1.5B-Instruct
    python examples/benchmark_throughput.py --max-batch 16  # limit batch size

Author: Gabriel Yogi
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# ── Prompts pool (varying lengths) ──
PROMPTS = [
    # Short (~10 tokens)
    "What is 2+2?",
    "Explain gravity in one sentence.",
    "What causes rain?",
    "Define machine learning briefly.",
    # Medium (~30-40 tokens)
    "Classify the sentiment of this review: I absolutely love this product, it exceeded all my expectations and I would recommend it to everyone!",
    "Classify the sentiment: This was the worst purchase I've ever made. Complete waste of money, broke after one day. Never buying again.",
    "Classify the sentiment: The product is okay. Nothing special, does what it says. Decent quality for the price, no complaints really.",
    "Classify the sentiment: Pretty good overall! Some minor issues with shipping but the product itself is fantastic. Five stars for quality.",
    # Long (~80+ tokens)
    "Summarize the following in one sentence: Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Unlike traditional programming where rules are explicitly coded, machine learning algorithms use statistical techniques to give computers the ability to learn from experience. The field has grown rapidly in recent years, driven by increases in computing power and availability of large datasets.",
    "Summarize in one sentence: Natural language processing is a field of computer science and artificial intelligence concerned with the interactions between computers and human language. It involves programming computers to process and analyze large amounts of natural language data. The result is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them.",
    "Summarize in one sentence: Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural networks, and transformers have been applied to fields including natural language processing, computer vision, and speech recognition.",
    "Summarize in one sentence: Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. It differs from supervised learning in that labelled input-output pairs are not required.",
    # Extra long (~100+ tokens)
    "Analyze the following passage and identify the main argument, supporting evidence, and conclusion: The rapid advancement of artificial intelligence has sparked intense debate about its impact on employment. Proponents argue that AI will create more jobs than it destroys, pointing to historical precedents where technological revolutions ultimately increased total employment. However, critics counter that the pace of AI development is unprecedented and that many workers lack the skills needed for emerging roles. Studies from leading research institutions suggest that while up to 30 percent of current jobs may be automated by 2030, new categories of employment will emerge in AI development, maintenance, and oversight.",
    "Analyze this passage: Climate change represents one of the most significant challenges facing humanity in the twenty-first century. Rising global temperatures, driven primarily by greenhouse gas emissions from human activities, are causing widespread environmental disruption. This includes rising sea levels threatening coastal communities, more frequent and severe weather events, and shifts in ecosystems that affect biodiversity. International efforts to address climate change have produced agreements like the Paris Accord, which aims to limit global warming to well below two degrees Celsius above pre-industrial levels.",
    "Analyze this passage: The democratization of education through online learning platforms has transformed how people access knowledge worldwide. Massive Open Online Courses and similar platforms have made high-quality educational content available to anyone with an internet connection. This shift has particularly benefited learners in developing countries who previously had limited access to advanced educational resources. However, challenges remain including digital divide issues, maintaining engagement in virtual environments, and ensuring the quality and credibility of online credentials.",
    "Analyze this passage: Quantum computing represents a fundamental shift in computational paradigms that could revolutionize fields from cryptography to drug discovery. Unlike classical computers that process information in binary bits, quantum computers use quantum bits or qubits that can exist in multiple states simultaneously through superposition. This enables quantum computers to solve certain types of problems exponentially faster than their classical counterparts. Major technology companies and research institutions are investing billions in developing practical quantum computing systems.",
]


def run_benchmark(model_name: str, max_tokens: int = 30, batch_sizes=None, quantize=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from megagemm.engine import InferenceEngine

    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16]

    print(f"\n{'='*75}")
    print(f"  ⚡ MegaGemm Throughput Benchmark")
    print(f"  Model: {model_name}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Prompt pool: {len(PROMPTS)} prompts")
    print(f"{'='*75}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ════════════════════════════════════════
    # PHASE 1: HuggingFace baseline
    # ════════════════════════════════════════
    print("\n📦 Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    hf_model.eval()

    # Warmup: 1 throw-away generation to heat CUDA/cuBLAS
    print("   🔥 HF warmup...")
    _wu = tokenizer.encode("warmup", return_tensors='pt').to(hf_model.device)
    with torch.inference_mode():
        hf_model.generate(_wu, max_new_tokens=5, do_sample=False)
    del _wu
    torch.cuda.synchronize()
    print("   ✅ HF ready (steady-state)")

    hf_results = {}
    for bs in batch_sizes:
        prompts = (PROMPTS * ((bs // len(PROMPTS)) + 1))[:bs]

        # Tokenize with chat template
        all_inputs = []
        total_input_tokens = 0
        for p in prompts:
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                try:
                    msg = [{"role": "user", "content": p}]
                    p = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                except Exception:
                    pass
            bos = tokenizer.bos_token
            add_sp = not (bos and p.startswith(bos))
            ids = tokenizer.encode(p, return_tensors='pt', add_special_tokens=add_sp).to(hf_model.device)
            all_inputs.append(ids)
            total_input_tokens += ids.shape[1]

        # Generate one by one (HF doesn't batch easily with varying lengths)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_output_tokens = 0
        for ids in all_inputs:
            with torch.inference_mode():
                out = hf_model.generate(ids, max_new_tokens=max_tokens, do_sample=False)
            total_output_tokens += out.shape[1] - ids.shape[1]
        torch.cuda.synchronize()
        t_hf = time.perf_counter() - t0

        hf_results[bs] = {
            'time': t_hf,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'tok_s': (total_input_tokens + total_output_tokens) / t_hf,
            'output_tok_s': total_output_tokens / t_hf,
        }

    del hf_model
    import gc; gc.collect()
    torch.cuda.empty_cache()
    print("   🗑️  HF freed")

    # ════════════════════════════════════════
    # PHASE 2: MegaGemm
    # ════════════════════════════════════════
    print("📦 Loading MegaGemm engine...")
    engine = InferenceEngine(model_name, quantize=quantize)

    # Warmup: compile all Triton kernels (paged attention, RMSNorm, etc.)
    print("   🔥 MegaGemm warmup (compiling Triton kernels)...")
    engine.generate("warmup", max_new_tokens=5, temperature=0.0,
                    repetition_penalty=1.0, verbose=False)
    engine.generate_batch(["warmup"], max_new_tokens=5, temperature=0.0, verbose=False)
    engine.reset_monitor()
    torch.cuda.synchronize()
    print("   ✅ MegaGemm ready (steady-state)")

    mg_seq_results = {}
    mg_bat_results = {}

    for bs in batch_sizes:
        prompts = (PROMPTS * ((bs // len(PROMPTS)) + 1))[:bs]

        # Count input tokens
        total_input_tokens = 0
        for p in prompts:
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                try:
                    msg = [{"role": "user", "content": p}]
                    fmt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                except Exception:
                    fmt = p
            else:
                fmt = p
            bos = tokenizer.bos_token
            add_sp = not (bos and fmt.startswith(bos))
            total_input_tokens += len(tokenizer.encode(fmt, add_special_tokens=add_sp))

        # ── Sequential ──
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seq_texts = []
        for p in prompts:
            r = engine.generate(p, max_new_tokens=max_tokens, temperature=0.0,
                                repetition_penalty=1.0, verbose=False)
            seq_texts.append(r[0] if isinstance(r, tuple) else r)
        torch.cuda.synchronize()
        t_seq = time.perf_counter() - t0

        total_output_tokens = sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in seq_texts)
        mg_seq_results[bs] = {
            'time': t_seq,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'tok_s': (total_input_tokens + total_output_tokens) / t_seq,
            'output_tok_s': total_output_tokens / t_seq,
        }

        # ── Batched ──
        engine.reset_monitor()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        bat_texts = engine.generate_batch(
            prompts, max_new_tokens=max_tokens, temperature=0.0, verbose=False,
        )
        torch.cuda.synchronize()
        t_bat = time.perf_counter() - t0

        total_output_tokens_bat = sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in bat_texts)
        mg_bat_results[bs] = {
            'time': t_bat,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens_bat,
            'total_tokens': total_input_tokens + total_output_tokens_bat,
            'tok_s': (total_input_tokens + total_output_tokens_bat) / t_bat,
            'output_tok_s': total_output_tokens_bat / t_bat,
        }

    # ════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════
    print(f"\n{'='*75}")
    print(f"  📊 THROUGHPUT RESULTS (tok/s = total tokens processed / time)")
    print(f"{'='*75}")

    print(f"\n  {'Batch':>5} │ {'':^25} │ {'':^25} │ {'':^25}")
    print(f"  {'Size':>5} │ {'HuggingFace':^25} │ {'MegaGemm seq':^25} │ {'MegaGemm batched':^25}")
    print(f"  {'─'*5}─┼─{'─'*25}─┼─{'─'*25}─┼─{'─'*25}")

    for bs in batch_sizes:
        hf = hf_results[bs]
        seq = mg_seq_results[bs]
        bat = mg_bat_results[bs]

        hf_str = f"{hf['tok_s']:,.0f} tok/s ({hf['time']:.1f}s)"
        seq_str = f"{seq['tok_s']:,.0f} tok/s ({seq['time']:.1f}s)"
        bat_str = f"{bat['tok_s']:,.0f} tok/s ({bat['time']:.1f}s)"

        # Mark fastest
        fastest = min(hf['time'], seq['time'], bat['time'])
        if hf['time'] == fastest: hf_str += " 🏆"
        if seq['time'] == fastest: seq_str += " 🏆"
        if bat['time'] == fastest: bat_str += " 🏆"

        print(f"  {bs:>5} │ {hf_str:^25} │ {seq_str:^25} │ {bat_str:^25}")

    # Scaling summary
    print(f"\n  📈 Batch Scaling (MegaGemm batched vs sequential):")
    for bs in batch_sizes:
        seq = mg_seq_results[bs]
        bat = mg_bat_results[bs]
        speedup = seq['time'] / bat['time'] if bat['time'] > 0 else 0
        bar = "█" * int(speedup * 3)
        print(f"      batch={bs:>2}: {speedup:>5.2f}x  {bar}")

    print(f"\n  📈 MegaGemm batched vs HuggingFace:")
    for bs in batch_sizes:
        hf = hf_results[bs]
        bat = mg_bat_results[bs]
        speedup = hf['time'] / bat['time'] if bat['time'] > 0 else 0
        bar = "█" * int(speedup * 3)
        print(f"      batch={bs:>2}: {speedup:>5.2f}x  {bar}")

    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser(description="MegaGemm Throughput Benchmark")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--max-batch", type=int, default=16,
                        help="Max batch size to test (tests 1, 4, 8, ... up to this)")
    parser.add_argument("--quantize", choices=["int8"], default=None)
    args = parser.parse_args()

    sizes = [s for s in [1, 4, 8, 16, 32] if s <= args.max_batch]
    run_benchmark(args.model, args.max_tokens, sizes, args.quantize)


if __name__ == "__main__":
    main()

"""
🔄 MegaGemm — KV Cache CPU Offload Benchmark
==============================================
Tests correctness and performance of TieredBlockManager GPU↔CPU offloading.

Run on Colab:
    Célula 1 — Setup:
        %cd /content/drive/MyDrive/MGRrmsnorm
        !pip install -e . -v 2>&1 | tail -5
        !pip install huggingface_hub safetensors transformers

    Célula 2 — Rodar:
        !python benchmarks/benchmark_kv_offload.py

    Opções:
        !python benchmarks/benchmark_kv_offload.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
        !python benchmarks/benchmark_kv_offload.py --model Qwen/Qwen2.5-7B-Instruct --gpu-blocks 128 --cpu-blocks 2048

Author: Gabriel Yogi
"""

import sys, os, time, json, argparse, gc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch

# =============================================================================
# Config
# =============================================================================
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BLOCK_SIZE = 16

PROMPTS_SHORT = [
    "The capital of France is",
    "In machine learning, backpropagation is",
    "The theory of general relativity states that",
]

PROMPT_LONG = (
    "Below is a detailed history of artificial intelligence from its origins to the present day. "
    "The field of artificial intelligence (AI) has a rich history that spans decades of research, "
    "breakthroughs, setbacks, and transformative innovations. "
    "In 1950, Alan Turing published his seminal paper 'Computing Machinery and Intelligence', "
    "which introduced the concept of the Turing Test as a measure of machine intelligence. "
    "This marked the beginning of formal inquiry into whether machines could think. "
    "The Dartmouth Conference of 1956 is widely regarded as the founding event of AI as a field. "
    "John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon organized this workshop, "
    "where the term 'artificial intelligence' was first coined. Early AI work focused on symbolic "
    "reasoning, problem solving, and game playing. Programs like the Logic Theorist and the General "
    "Problem Solver demonstrated that machines could perform tasks requiring reasoning. "
    "The 1960s and 1970s saw the development of expert systems, natural language processing, "
    "and early neural network research. Frank Rosenblatt's Perceptron showed promise for pattern "
    "recognition, but Minsky and Papert's 1969 book highlighted its limitations, leading to the "
    "first 'AI winter'. Funding dried up and progress stalled. In the 1980s, expert systems like "
    "MYCIN and XCON revived commercial interest. Japan's Fifth Generation Computer project "
    "spurred an international AI arms race. However, the limitations of rule-based systems and "
    "the combinatorial explosion problem led to a second AI winter in the late 1980s and early 1990s. "
    "The resurgence of neural networks began with backpropagation, popularized by Rumelhart, "
    "Hinton, and Williams in 1986. This allowed training of multi-layer networks, but hardware "
    "limitations kept progress slow. The 2000s brought support vector machines, random forests, "
    "and ensemble methods. Statistical approaches began to dominate. In 2006, Geoffrey Hinton's "
    "work on deep belief networks reignited interest in deep learning. The ImageNet challenge of "
    "2012, where AlexNet achieved a breakthrough in image recognition, proved that deep learning "
    "could outperform traditional methods by a wide margin. This launched the modern deep learning "
    "revolution. Transformers, introduced in 2017 by Vaswani et al. with 'Attention Is All You Need', "
    "revolutionized NLP. BERT, GPT, and their successors demonstrated that large language models "
    "could understand and generate human-like text. GPT-3 in 2020 showed that scaling model size "
    "and training data could produce remarkably capable systems. Today, models like GPT-4, Claude, "
    "Gemini, and LLaMA push the boundaries of what AI can achieve. "
    "Based on this comprehensive history, explain the three most important breakthroughs in AI "
    "and predict what the next major breakthrough might be."
)

MAX_NEW_TOKENS_SHORT = 50
MAX_NEW_TOKENS_LONG = 256


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_gpu_mem():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  GPU mem: {alloc:.0f}MB alloc / {reserved:.0f}MB reserved / {total:.0f}MB total")


def print_cpu_mem():
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  CPU mem: {mem.used/1024**3:.1f}GB used / {mem.total/1024**3:.1f}GB total")
    except ImportError:
        pass


# =============================================================================
# Test 1: Correctness — outputs must match with and without offload
# =============================================================================
def test_correctness(model_name, gpu_blocks, cpu_blocks, gpu_window):
    print("\n" + "=" * 70)
    print("🧪 TEST 1: CORRECTNESS — Output equivalence with/without offload")
    print("=" * 70)

    from megagemm.engine import InferenceEngine

    # --- Baseline: no offload ---
    print("\n  [1/2] Loading model WITHOUT offload...")
    engine_base = InferenceEngine(
        model_name,
        dtype=torch.float16,
        num_blocks=gpu_blocks + cpu_blocks,  # all on GPU
        block_size=BLOCK_SIZE,
    )

    # Warmup
    _ = engine_base.generate("Hello", max_new_tokens=3, temperature=0.0)

    base_outputs = []
    for prompt in PROMPTS_SHORT:
        out = engine_base.generate(prompt, max_new_tokens=MAX_NEW_TOKENS_SHORT, temperature=0.0)
        base_outputs.append(out)
        print(f"    Base: '{out[:80]}...'")

    del engine_base
    clear_gpu()

    # --- Tiered: GPU+CPU offload ---
    print(f"\n  [2/2] Loading model WITH offload (gpu={gpu_blocks}, cpu={cpu_blocks}, window={gpu_window})...")
    engine_offload = InferenceEngine(
        model_name,
        dtype=torch.float16,
        num_blocks=gpu_blocks,
        block_size=BLOCK_SIZE,
        kv_offload=True,
        num_cpu_blocks=cpu_blocks,
        gpu_window=gpu_window,
    )

    # Warmup
    _ = engine_offload.generate("Hello", max_new_tokens=3, temperature=0.0)

    offload_outputs = []
    for prompt in PROMPTS_SHORT:
        out = engine_offload.generate(prompt, max_new_tokens=MAX_NEW_TOKENS_SHORT, temperature=0.0)
        offload_outputs.append(out)
        print(f"    Offload: '{out[:80]}...'")

    # Compare
    print("\n  📊 Results:")
    all_match = True
    for i, (base, offload) in enumerate(zip(base_outputs, offload_outputs)):
        match = base == offload
        emoji = "✅" if match else "❌"
        print(f"    Prompt {i+1}: {emoji} {'MATCH' if match else 'MISMATCH'}")
        if not match:
            all_match = False
            # Show token-level divergence
            base_toks = base.split()
            off_toks = offload.split()
            for j, (bt, ot) in enumerate(zip(base_toks, off_toks)):
                if bt != ot:
                    print(f"      First divergence at word {j}: '{bt}' vs '{ot}'")
                    break

    del engine_offload
    clear_gpu()

    return all_match


# =============================================================================
# Test 2: Performance — throughput comparison
# =============================================================================
def test_performance(model_name, gpu_blocks, cpu_blocks, gpu_window):
    print("\n" + "=" * 70)
    print("🏎️  TEST 2: PERFORMANCE — Throughput comparison")
    print("=" * 70)

    from megagemm.engine import InferenceEngine

    results = {}

    # --- Baseline: no offload ---
    print("\n  [1/2] Benchmark WITHOUT offload...")
    engine_base = InferenceEngine(
        model_name,
        dtype=torch.float16,
        num_blocks=gpu_blocks + cpu_blocks,
        block_size=BLOCK_SIZE,
    )
    _ = engine_base.generate("Hello", max_new_tokens=3, temperature=0.0)

    print_gpu_mem()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = engine_base.generate(
        PROMPTS_SHORT[0],
        max_new_tokens=MAX_NEW_TOKENS_SHORT,
        temperature=0.0,
    )
    torch.cuda.synchronize()
    base_time = time.perf_counter() - t0
    base_tps = MAX_NEW_TOKENS_SHORT / base_time

    print(f"    Baseline: {base_tps:.1f} tok/s ({base_time*1000:.0f}ms)")

    del engine_base
    clear_gpu()

    # --- Tiered: GPU+CPU offload ---
    print(f"\n  [2/2] Benchmark WITH offload (gpu={gpu_blocks}, cpu={cpu_blocks})...")
    engine_offload = InferenceEngine(
        model_name,
        dtype=torch.float16,
        num_blocks=gpu_blocks,
        block_size=BLOCK_SIZE,
        kv_offload=True,
        num_cpu_blocks=cpu_blocks,
        gpu_window=gpu_window,
    )
    _ = engine_offload.generate("Hello", max_new_tokens=3, temperature=0.0)

    print_gpu_mem()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = engine_offload.generate(
        PROMPTS_SHORT[0],
        max_new_tokens=MAX_NEW_TOKENS_SHORT,
        temperature=0.0,
        verbose=True,
    )
    torch.cuda.synchronize()
    offload_time = time.perf_counter() - t0
    offload_tps = MAX_NEW_TOKENS_SHORT / offload_time

    print(f"    Offload:  {offload_tps:.1f} tok/s ({offload_time*1000:.0f}ms)")

    del engine_offload
    clear_gpu()

    # Summary
    ratio = offload_tps / base_tps if base_tps > 0 else 0
    print(f"\n  📊 Summary:")
    print(f"    Baseline:    {base_tps:.1f} tok/s")
    print(f"    With Offload: {offload_tps:.1f} tok/s")
    print(f"    Ratio:       {ratio:.2f}x {'(overhead from offload)' if ratio < 1 else ''}")

    results = {
        'baseline_tps': round(base_tps, 1),
        'offload_tps': round(offload_tps, 1),
        'ratio': round(ratio, 3),
        'gpu_blocks': gpu_blocks,
        'cpu_blocks': cpu_blocks,
        'gpu_window': gpu_window,
    }
    return results


# =============================================================================
# Test 3: Long context — stress test with large KV cache
# =============================================================================
def test_long_context(model_name, gpu_blocks, cpu_blocks, gpu_window):
    print("\n" + "=" * 70)
    print("📏 TEST 3: LONG CONTEXT — KV cache exceeds GPU capacity")
    print("=" * 70)

    from megagemm.engine import InferenceEngine

    # Use moderate GPU pool to force offloading while having headroom.
    # Need enough GPU blocks to hold the prompt during prefill (write_kv),
    # but few enough that decode will trigger eviction to CPU.
    small_gpu = min(gpu_blocks, 64)
    small_window = min(gpu_window, small_gpu // 4)  # 25% window

    print(f"\n  Config: gpu_blocks={small_gpu}, cpu_blocks={cpu_blocks}, window={small_window}")
    print(f"  GPU can hold: {small_gpu * BLOCK_SIZE} tokens")
    print(f"  CPU can hold: {cpu_blocks * BLOCK_SIZE} tokens")
    print(f"  Total capacity: {(small_gpu + cpu_blocks) * BLOCK_SIZE} tokens")

    engine = InferenceEngine(
        model_name,
        dtype=torch.float16,
        num_blocks=small_gpu,
        block_size=BLOCK_SIZE,
        kv_offload=True,
        num_cpu_blocks=cpu_blocks,
        gpu_window=small_window,
    )

    # Warmup
    _ = engine.generate("Hello", max_new_tokens=3, temperature=0.0)
    print_gpu_mem()
    print_cpu_mem()

    print(f"\n  🔥 Generating with long prompt ({len(PROMPT_LONG)} chars)...")
    print(f"     + {MAX_NEW_TOKENS_LONG} new tokens")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    try:
        output = engine.generate(
            PROMPT_LONG,
            max_new_tokens=MAX_NEW_TOKENS_LONG,
            temperature=0.7,
            top_k=50,
            verbose=True,
        )
        torch.cuda.synchronize()
        t_total = time.perf_counter() - t0

        print(f"\n  ✅ Success! Generated {MAX_NEW_TOKENS_LONG} tokens in {t_total:.1f}s")
        print(f"  📝 Output preview: '{output[:200]}...'")

        # Print offload stats
        engine.block_manager.print_stats()
        print_gpu_mem()
        print_cpu_mem()

        result = {
            'success': True,
            'time_s': round(t_total, 2),
            'tps': round(MAX_NEW_TOKENS_LONG / t_total, 1),
            'gpu_blocks': small_gpu,
            'cpu_blocks': cpu_blocks,
        }

    except Exception as e:
        print(f"\n  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        result = {'success': False, 'error': str(e)}

    del engine
    clear_gpu()

    return result


# =============================================================================
# Test 4: Batch generation with offload
# =============================================================================
def test_batch(model_name, gpu_blocks, cpu_blocks, gpu_window):
    print("\n" + "=" * 70)
    print("📦 TEST 4: BATCH GENERATION — Continuous batching with offload")
    print("=" * 70)

    from megagemm.engine import InferenceEngine

    engine = InferenceEngine(
        model_name,
        dtype=torch.float16,
        num_blocks=gpu_blocks,
        block_size=BLOCK_SIZE,
        kv_offload=True,
        num_cpu_blocks=cpu_blocks,
        gpu_window=gpu_window,
    )

    # Warmup
    _ = engine.generate("Hello", max_new_tokens=3, temperature=0.0)

    print(f"\n  Running batch of {len(PROMPTS_SHORT)} prompts...")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    try:
        outputs = engine.generate_batch(
            PROMPTS_SHORT,
            max_new_tokens=MAX_NEW_TOKENS_SHORT,
            temperature=0.0,
            verbose=True,
        )
        torch.cuda.synchronize()
        t_total = time.perf_counter() - t0

        print(f"\n  ✅ Batch complete! {len(outputs)} outputs in {t_total:.1f}s")
        for i, out in enumerate(outputs):
            print(f"    [{i}]: '{out[:80]}...'")

        engine.block_manager.print_stats()

        result = {
            'success': True,
            'num_prompts': len(PROMPTS_SHORT),
            'time_s': round(t_total, 2),
        }

    except Exception as e:
        print(f"\n  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        result = {'success': False, 'error': str(e)}

    del engine
    clear_gpu()

    return result


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MegaGemm KV Cache Offload Benchmark")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model ID")
    parser.add_argument("--gpu-blocks", type=int, default=256,
                        help="Number of KV cache blocks on GPU")
    parser.add_argument("--cpu-blocks", type=int, default=512,
                        help="Number of KV cache blocks on CPU (pinned)")
    parser.add_argument("--gpu-window", type=int, default=32,
                        help="Blocks to keep on GPU per sequence (window)")
    parser.add_argument("--skip-correctness", action="store_true",
                        help="Skip correctness test (faster)")
    parser.add_argument("--skip-performance", action="store_true",
                        help="Skip performance test")
    parser.add_argument("--skip-long-context", action="store_true",
                        help="Skip long context test")
    parser.add_argument("--skip-batch", action="store_true",
                        help="Skip batch test")
    args = parser.parse_args()

    print("=" * 70)
    print("🔄 MegaGemm — KV Cache CPU Offload Benchmark")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  GPU blocks: {args.gpu_blocks}")
    print(f"  CPU blocks: {args.cpu_blocks}")
    print(f"  GPU window: {args.gpu_window}")
    print(f"  Block size: {BLOCK_SIZE}")
    print(f"  GPU token capacity: {args.gpu_blocks * BLOCK_SIZE}")
    print(f"  CPU token capacity: {args.cpu_blocks * BLOCK_SIZE}")
    print(f"  Total token capacity: {(args.gpu_blocks + args.cpu_blocks) * BLOCK_SIZE}")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu} ({mem:.1f}GB)")
    print_cpu_mem()

    all_results = {
        'model': args.model,
        'gpu_blocks': args.gpu_blocks,
        'cpu_blocks': args.cpu_blocks,
        'gpu_window': args.gpu_window,
    }

    # Test 1: Correctness
    if not args.skip_correctness:
        correct = test_correctness(args.model, args.gpu_blocks, args.cpu_blocks, args.gpu_window)
        all_results['correctness'] = correct
    else:
        print("\n⏭️  Skipping correctness test")

    # Test 2: Performance
    if not args.skip_performance:
        perf = test_performance(args.model, args.gpu_blocks, args.cpu_blocks, args.gpu_window)
        all_results['performance'] = perf
    else:
        print("\n⏭️  Skipping performance test")

    # Test 3: Long context
    if not args.skip_long_context:
        long_ctx = test_long_context(args.model, args.gpu_blocks, args.cpu_blocks, args.gpu_window)
        all_results['long_context'] = long_ctx
    else:
        print("\n⏭️  Skipping long context test")

    # Test 4: Batch
    if not args.skip_batch:
        batch = test_batch(args.model, args.gpu_blocks, args.cpu_blocks, args.gpu_window)
        all_results['batch'] = batch
    else:
        print("\n⏭️  Skipping batch test")

    # Save results
    print("\n" + "=" * 70)
    print("📋 FINAL RESULTS")
    print("=" * 70)
    print(json.dumps(all_results, indent=2))

    out_file = f"benchmark_kv_offload_{args.model.split('/')[-1]}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📁 Results saved to {out_file}")

    print("\n🏁 Benchmark complete!")

#!/usr/bin/env python3
"""
📊 Bench360-Compatible Benchmark for MegaGemm
================================================
Matches Bench360 methodology (arXiv:2511.16682) for direct
comparison with vLLM, SGLang, LMDeploy, TGI on the same workload.

Bench360 setup:
  - Models: Qwen2.5-7B-Instruct, LLaMA-3.1-8B-Instruct, Mistral-7B-v0.3
  - Batch sizes: {16, 32, 64, 128}
  - Prompts: ~320 tokens input
  - Output: ~32 tokens
  - Metric: TPS = total_output_tokens / total_time (prefill + decode)
  - GPU: L4 (24GB), A10 (24GB), A30 (24GB)
  - Precision: FP16

Usage:
    python examples/bench360_compare.py
    python examples/bench360_compare.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import time
import argparse
import torch

# ── Bench360-style prompts (~320 tokens each after chat template) ──
# These are reasoning/knowledge prompts similar to Bench360's MMLU/SQuAD tasks
PROMPTS_320 = [
    # ~320 tokens after chat template wrapping
    """Explain in detail the process of photosynthesis in C4 plants, including the Hatch-Slack pathway. Describe how the spatial separation of initial carbon fixation in mesophyll cells and the Calvin cycle in bundle sheath cells provides an advantage in hot, arid environments. Compare this mechanism with CAM photosynthesis found in succulent plants, noting the temporal separation strategy. Include discussion of the key enzymes PEP carboxylase and RuBisCO, their relative affinities for CO2, and how C4 plants minimize photorespiration. Also explain why C4 photosynthesis evolved independently in multiple plant lineages and what environmental pressures drove this convergent evolution. Finally, discuss the ongoing research efforts to engineer C4 photosynthetic pathways into C3 crop plants like rice, and the potential agricultural benefits.""",

    """Provide a comprehensive analysis of the Byzantine Empire's administrative system under Justinian I. Discuss the Corpus Juris Civilis and its lasting impact on Western legal traditions. Explain the theme system of military-administrative provinces that replaced the older Roman provincial structure. Detail the role of the Emperor as both political and religious leader, the concept of caesaropapism, and how this differed from the Western model of church-state relations. Analyze the economic foundations of the empire, including the solidus as a stable currency, the silk trade monopoly, and the importance of Constantinople as a commercial crossroads. Describe the military reforms, including the use of Greek fire in naval warfare and the evolution of heavy cavalry tactics. Conclude with an assessment of how these institutional innovations contributed to the empire's remarkable longevity.""",

    """Describe the mathematical foundations of quantum computing, starting with the representation of qubits as vectors in a two-dimensional complex Hilbert space. Explain how quantum gates operate as unitary transformations, and detail the properties of common gates including the Hadamard gate, Pauli gates, CNOT gate, and Toffoli gate. Discuss the concept of quantum entanglement and how it enables computational advantages through Bell states. Explain Shor's algorithm for integer factorization, including the quantum Fourier transform component, and analyze its implications for cryptographic security. Compare the circuit model of quantum computing with alternative paradigms such as adiabatic quantum computing and measurement-based quantum computing. Discuss current hardware approaches including superconducting qubits, trapped ions, and topological qubits, noting the challenges of decoherence and error correction.""",

    """Analyze the economic implications of artificial intelligence adoption across major industry sectors. Begin with manufacturing, discussing how predictive maintenance, quality control through computer vision, and autonomous robotics are transforming production efficiency. Move to healthcare, examining AI applications in medical imaging diagnosis, drug discovery pipeline acceleration, and personalized treatment planning through genomic analysis. Discuss the financial services sector, covering algorithmic trading strategies, fraud detection systems, credit scoring models, and regulatory compliance automation. Examine the transportation sector's transformation through autonomous vehicles, route optimization, and predictive logistics. For each sector, provide estimates of potential productivity gains, displacement of existing jobs, and creation of new roles. Conclude with a discussion of the broader macroeconomic effects including potential impacts on GDP growth, income inequality, and international competitiveness.""",

    """Explain the geological processes responsible for plate tectonics, beginning with the evidence that Alfred Wegener presented for continental drift and the subsequent development of the theory through seafloor spreading observations by Harry Hess. Detail the three types of plate boundaries: divergent boundaries where new oceanic crust forms at mid-ocean ridges, convergent boundaries where subduction zones create deep sea trenches and volcanic arcs, and transform boundaries where plates slide past each other creating fault systems. Discuss the driving mechanisms including mantle convection, ridge push, and slab pull forces. Explain how plate tectonics connects to the rock cycle, mountain building through orogeny, and the distribution of earthquakes and volcanic activity. Describe the Wilson Cycle of ocean basin opening and closing, and explain how paleomagnetism preserved in oceanic crust provides evidence for seafloor spreading and polar wandering.""",

    """Discuss the evolution of programming language paradigms from the earliest machine code through modern multi-paradigm languages. Begin with assembly language and the transition to high-level languages with FORTRAN and COBOL in the 1950s. Explain the development of structured programming through ALGOL and Pascal, and how this addressed the problems of unstructured goto-based code. Detail the emergence of object-oriented programming through Simula and Smalltalk, and its mainstream adoption through C++ and Java. Analyze the resurgence of functional programming concepts from lambda calculus through Lisp, ML, Haskell, and their influence on modern languages like Scala and Kotlin. Discuss concurrent and parallel programming paradigms including the actor model in Erlang and Go's goroutines. Examine how modern languages like Rust introduce ownership-based memory safety, and how Python's simplicity drove the machine learning revolution.""",

    """Analyze the causes, progression, and lasting consequences of the Industrial Revolution, beginning with its origins in 18th century Britain. Explain the pre-conditions that made Britain uniquely positioned: abundant coal and iron resources, a stable political system after the Glorious Revolution, enclosure movements that created a mobile labor force, and existing mercantile networks. Detail the key technological innovations including Newcomen and Watt's steam engines, Hargreaves' spinning jenny, Arkwright's water frame, and Cartwright's power loom. Discuss how the factory system replaced cottage industry, creating new patterns of urbanization, labor relations, and social class structures. Analyze the second Industrial Revolution of the late 19th century, focusing on steel production, electrical power, chemical industries, and assembly line manufacturing. Examine the environmental consequences including pollution, deforestation, and the beginning of anthropogenic climate change.""",

    """Provide a detailed explanation of the human immune system, distinguishing between innate and adaptive immunity. For innate immunity, describe the role of physical barriers like skin and mucous membranes, the complement system, natural killer cells, macrophages, neutrophils, and dendritic cells. Explain the inflammatory response and the role of cytokines in coordinating immune reactions. For adaptive immunity, detail the development and function of T cells including helper T cells, cytotoxic T cells, and regulatory T cells, explaining positive and negative selection in the thymus. Describe B cell maturation, antibody structure with heavy and light chains, class switching between IgM, IgG, IgA, IgE, and IgD isotypes, and affinity maturation through somatic hypermutation. Explain the formation of immunological memory through memory T and B cells, and how this principle underlies vaccination strategies including mRNA vaccines.""",
]

TARGET_INPUT_TOKENS = 320  # Bench360 standard


def run_bench360(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_tokens: int = 32,
    batch_sizes: list = None,
    quantize: str = None,
    num_warmup: int = 3,
    shape_warmup_runs: int = 0,
):
    """Run Bench360-compatible throughput benchmark."""
    from megagemm.engine import InferenceEngine

    if batch_sizes is None:
        batch_sizes = [16, 32, 64, 128]

    gpu_name = "CPU"
    vram_gb = 0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"\n{'='*70}")
    print(f"  📊 Bench360-Compatible Benchmark — MegaGemm")
    print(f"{'='*70}")
    print(f"  Paper:    arXiv:2511.16682 (Bench360)")
    print(f"  Model:    {model_name}")
    print(f"  Quant:    {quantize or 'FP16'}")
    print(f"  Input:    ~320 tokens (Bench360 standard)")
    print(f"  Output:   {max_tokens} tokens (Bench360: 32)")
    print(f"  Batches:  {batch_sizes}")
    print(f"  GPU:      {gpu_name} ({vram_gb:.0f}GB)")
    print(f"  Warmup:   {num_warmup} runs")
    if shape_warmup_runs > 0:
        print(f"  Shape warmup: {shape_warmup_runs} untimed runs per batch")
    print(f"{'='*70}")

    # Load engine (auto-config)
    actual_model = model_name
    engine_quantize = quantize
    if quantize == 'awq':
        # AWQ: load the AWQ variant, auto-detected by loader
        actual_model = model_name.replace('-Instruct', '-Instruct-AWQ')
        engine_quantize = None  # AWQ is auto-detected from model config
    print(f"\n📦 Loading {actual_model}...")
    t0 = time.perf_counter()
    bench_max_batch = max(batch_sizes) if batch_sizes else 128
    bench_max_seq_len = TARGET_INPUT_TOKENS + max_tokens + 32
    engine = InferenceEngine(
        actual_model,
        quantize=engine_quantize,
        kv_offload=False,  # pure GPU for throughput benchmark
        max_batch_size=bench_max_batch,
        max_seq_len=bench_max_seq_len,
    )
    tokenizer = engine.tokenizer
    load_time = time.perf_counter() - t0
    print(f"   Loaded in {load_time:.1f}s")

    # Pad or truncate prompts to TARGET_INPUT_TOKENS
    # Bench360 uses ~320 input tokens per prompt
    padded_prompts = []
    prompt_lens = []
    pad_phrase = " Provide thorough analysis with specific examples, data points, and citations from peer-reviewed literature where applicable. Consider multiple perspectives and discuss counterarguments."
    for p in PROMPTS_320:
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            try:
                msg = [{"role": "user", "content": p}]
                fmt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            except Exception:
                fmt = p
        else:
            fmt = p
        toks = tokenizer.encode(fmt)
        cur_len = len(toks)
        # Pad if too short
        text = p
        while cur_len < TARGET_INPUT_TOKENS:
            text += pad_phrase
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                try:
                    msg = [{"role": "user", "content": text}]
                    fmt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                except Exception:
                    fmt = text
            else:
                fmt = text
            toks = tokenizer.encode(fmt)
            cur_len = len(toks)
        # Truncate if over
        if cur_len > TARGET_INPUT_TOKENS + 20:
            toks = toks[:TARGET_INPUT_TOKENS]
            cur_len = TARGET_INPUT_TOKENS
        padded_prompts.append(text)
        prompt_lens.append(cur_len)

    avg_in = sum(prompt_lens) / len(prompt_lens)
    print(f"   Input tokens: min={min(prompt_lens)}, avg={avg_in:.0f}, max={max(prompt_lens)}")
    print(f"   Output tokens: {max_tokens} (fixed)")
    print(f"   Total context: ~{avg_in:.0f} + {max_tokens} = ~{avg_in + max_tokens:.0f}")

    # Warmup
    print(f"   🔥 Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        engine.generate("warmup text for compilation", max_new_tokens=5,
                        temperature=0.0, repetition_penalty=1.0, verbose=False)
    engine.generate_batch(["warmup batch"], max_new_tokens=5,
                          temperature=0.0, verbose=False)
    engine.reset_monitor()
    torch.cuda.synchronize()
    print(f"   ✅ Ready!")

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e9
        print(f"   VRAM: {vram_used:.1f}GB / {vram_gb:.0f}GB ({vram_used/vram_gb*100:.0f}%)")

    # ── Run benchmark ──
    results = {}

    for bs in batch_sizes:
        prompts = (PROMPTS_320 * ((bs // len(PROMPTS_320)) + 1))[:bs]

        print(f"\n{'─'*70}")
        print(f"  🚀 Batch={bs}")
        print(f"{'─'*70}")

        try:
            torch.cuda.synchronize()
            engine.reset_monitor()

            if shape_warmup_runs > 0:
                print(f"   🔁 Shape warmup ({shape_warmup_runs} runs)...")
                for _ in range(shape_warmup_runs):
                    _ = engine.generate_batch(
                        (padded_prompts * ((bs // len(padded_prompts)) + 1))[:bs],
                        max_new_tokens=max_tokens,
                        temperature=0.0,
                        verbose=False,
                    )
                    torch.cuda.synchronize()
                engine.reset_monitor()

            t_start = time.perf_counter()
            outputs = engine.generate_batch(
                (padded_prompts * ((bs // len(padded_prompts)) + 1))[:bs],
                max_new_tokens=max_tokens,
                temperature=0.0,
                verbose=False,
            )
            torch.cuda.synchronize()
            t_end = time.perf_counter()

            elapsed = t_end - t_start

            # Count actual output tokens
            # generate_batch returns ONLY generated text (no prompt)
            total_out_tokens = 0
            total_in_tokens = sum(prompt_lens[i % len(prompt_lens)] for i in range(bs))
            for out in outputs:
                total_out_tokens += len(tokenizer.encode(out))

            # Bench360 TPS = output_tokens / total_time
            tps = total_out_tokens / elapsed
            # Also compute prefill throughput (input tokens / time)
            prefill_tps = total_in_tokens / elapsed
            # Combined throughput
            combined_tps = (total_in_tokens + total_out_tokens) / elapsed

            results[bs] = {
                'ok': True,
                'time': elapsed,
                'tps_output': tps,
                'tps_prefill': prefill_tps,
                'tps_combined': combined_tps,
                'total_in': total_in_tokens,
                'total_out': total_out_tokens,
            }

            print(f"   ⏱️  Time:        {elapsed:.2f}s")
            print(f"   📊 Output TPS:  {tps:.1f} tok/s  (Bench360 metric)")
            print(f"   📊 Prefill TPS: {prefill_tps:.1f} tok/s")
            print(f"   📊 Combined:    {combined_tps:.1f} tok/s")
            print(f"   📊 Tokens:      {total_in_tokens} in + {total_out_tokens} out")

            # Profiling: show actual prefill vs decode time split
            if hasattr(engine, '_last_scheduler'):
                sched_stats = engine._last_scheduler.get_stats()
                pf_ms = sched_stats.get('prefill_time_ms', 0)
                dc_ms = sched_stats.get('decode_time_ms', 0)
                pf_s = pf_ms / 1000
                dc_s = dc_ms / 1000
                dc_tps = total_out_tokens / dc_s if dc_s > 0 else 0
                pf_tok_s = total_in_tokens / pf_s if pf_s > 0 else 0
                print(f"   ⏱️  Prefill:     {pf_s:.2f}s ({pf_s/elapsed*100:.0f}%) → {pf_tok_s:.0f} tok/s")
                print(f"   ⏱️  Decode:      {dc_s:.2f}s ({dc_s/elapsed*100:.0f}%) → {dc_tps:.0f} DECODE tok/s")
                stage_timing = sched_stats.get('prefill_stage_timing')
                if stage_timing:
                    stage_order = [
                        ('mlp_native_ms', 'mlp_native'),
                        ('qkv_ms', 'QKV'),
                        ('o_proj_ms', 'O'),
                        ('gate_up_ms', 'gate_up'),
                        ('down_proj_ms', 'down'),
                        ('attn_core_ms', 'attn'),
                        ('kv_write_ms', 'kv'),
                    ]
                    stage_total_ms = float(stage_timing.get('total_ms', 0.0) or 0.0)
                    if stage_total_ms <= 0.0:
                        stage_total_ms = sum(
                            float(stage_timing.get(key, 0.0) or 0.0)
                            for key, _ in stage_order
                        )
                    parts = []
                    for key, label in stage_order:
                        ms = float(stage_timing.get(key, 0.0) or 0.0)
                        if ms <= 0.0:
                            continue
                        share = (ms / stage_total_ms * 100.0) if stage_total_ms > 0 else 0.0
                        parts.append(f"{label}={ms:.0f}ms ({share:.0f}%)")
                    meta = []
                    if sched_stats.get('prefill_stage_chunks'):
                        meta.append(f"chunks={sched_stats['prefill_stage_chunks']}")
                    if sched_stats.get('prefill_stage_total_tokens'):
                        meta.append(f"tok={sched_stats['prefill_stage_total_tokens']}")
                    if sched_stats.get('prefill_stage_max_len'):
                        meta.append(f"max_len={sched_stats['prefill_stage_max_len']}")
                    if parts:
                        suffix = f" | {' | '.join(meta)}" if meta else ""
                        print(f"   🔬 Prefill stages: {' | '.join(parts)}{suffix}")
                graph_stats = sched_stats.get('decode_cuda_graphs')
                if graph_stats:
                    parts = [
                        f"enabled={int(bool(graph_stats.get('enabled', False)))}",
                        f"captures={int(graph_stats.get('captures', 0) or 0)}",
                        f"replays={int(graph_stats.get('replays', 0) or 0)}",
                        f"warmups={int(graph_stats.get('warmups', 0) or 0)}",
                        f"failures={int(graph_stats.get('failures', 0) or 0)}",
                    ]
                    if graph_stats.get('min_batch'):
                        parts.append(f"min_batch={int(graph_stats['min_batch'])}")
                    print(f"   🕸️  Decode graphs: {' | '.join(parts)}")
                prefill_graph_stats = sched_stats.get('prefill_cuda_graphs')
                if prefill_graph_stats:
                    parts = [
                        f"enabled={int(bool(prefill_graph_stats.get('enabled', False)))}",
                        f"captures={int(prefill_graph_stats.get('captures', 0) or 0)}",
                        f"replays={int(prefill_graph_stats.get('replays', 0) or 0)}",
                        f"warmups={int(prefill_graph_stats.get('warmups', 0) or 0)}",
                        f"skips={int(prefill_graph_stats.get('skips', 0) or 0)}",
                        f"failures={int(prefill_graph_stats.get('failures', 0) or 0)}",
                    ]
                    if prefill_graph_stats.get('buckets') is not None:
                        parts.append(f"buckets={int(prefill_graph_stats.get('buckets', 0) or 0)}")
                    if prefill_graph_stats.get('min_reqs'):
                        parts.append(f"min_reqs={int(prefill_graph_stats['min_reqs'])}")
                    print(f"   🧊 Prefill graphs: {' | '.join(parts)}")

            # Show sample
            sample = outputs[0][len(prompts[0]):].strip()[:120]
            print(f"   Sample: {sample}...")

        except Exception as e:
            results[bs] = {'ok': False, 'error': str(e)}
            print(f"   ❌ FAILED: {e}")
            # Reset block manager to clean orphaned sequences
            try:
                bm = engine.block_manager
                for sid in list(bm.block_tables.keys()):
                    bm.free_sequence(sid)
            except Exception:
                pass

    # ── Results table ──
    print(f"\n{'='*70}")
    print(f"  📊 BENCH360-COMPATIBLE RESULTS — MegaGemm")
    print(f"{'='*70}")
    print(f"  Model: {model_name} | GPU: {gpu_name} | {quantize or 'FP16'}")
    print(f"  Input: ~{avg_in:.0f} tokens | Output: {max_tokens} tokens")
    print(f"{'─'*70}")
    print(f"  {'Batch':>5} │ {'Time':>7} │ {'Output TPS':>11} │ {'Prefill TPS':>12} │ {'Combined':>10}")
    print(f"  {'─'*5}─┼─{'─'*7}─┼─{'─'*11}─┼─{'─'*12}─┼─{'─'*10}")

    peak_tps = 0
    peak_bs = 0
    for bs in batch_sizes:
        r = results.get(bs, {})
        if not r.get('ok', False):
            print(f"  {bs:>5} │     ERR │           — │            — │          —")
            continue
        marker = ""
        if r['tps_output'] > peak_tps:
            peak_tps = r['tps_output']
            peak_bs = bs
            marker = " 🏆"
        print(f"  {bs:>5} │ {r['time']:>6.1f}s │ {r['tps_output']:>9.1f}  │ {r['tps_prefill']:>10.1f}  │ {r['tps_combined']:>8.1f}{marker}")

    if peak_bs > 0:
        r = results[peak_bs]
        print(f"\n  🏆 Peak at batch={peak_bs}: {r['tps_output']:.1f} output TPS")

    # ── Bench360 comparison ──
    print(f"\n{'─'*70}")
    print(f"  📊 COMPARISON vs Bench360 (arXiv:2511.16682)")
    print(f"{'─'*70}")
    print(f"  Bench360 L4 results (Table 5, FP16 7B avg, B=128):")
    print(f"    SGLang:   513 ± 29 TPS  (paper Table 5)")
    print(f"    vLLM:     402 ± 21 TPS  (paper Table 5)")
    print(f"    LMDeploy: ~150-200 TPS")
    print(f"    TGI:      ~100-180 TPS")
    if peak_bs > 0:
        print(f"    MegaGemm: {peak_tps:.0f} TPS  ← you are here")
        if peak_tps > 650:
            print(f"    ✅ ABOVE all Bench360 engines!")
        elif peak_tps > 400:
            print(f"    ✅ Competitive with SGLang/vLLM")
        elif peak_tps > 200:
            print(f"    ⚠️  Below SGLang/vLLM, above LMDeploy/TGI")
        else:
            print(f"    ⚠️  Below all Bench360 engines")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Bench360-Compatible MegaGemm Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model to benchmark")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Output tokens per request (Bench360: 32)")
    parser.add_argument("--max-batch", type=int, default=128,
                        help="Max batch size to test")
    parser.add_argument("--quantize", choices=["int8", "fp8", "awq"], default=None)
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup runs")
    parser.add_argument("--shape-warmup-runs", type=int, default=0,
                        help="Untimed per-batch warmup runs before measurement")
    args = parser.parse_args()

    sizes = [s for s in [16, 32, 64, 128, 256] if s <= args.max_batch]

    run_bench360(
        model_name=args.model,
        max_tokens=args.max_tokens,
        batch_sizes=sizes,
        quantize=args.quantize,
        num_warmup=args.warmup,
        shape_warmup_runs=args.shape_warmup_runs,
    )


if __name__ == "__main__":
    main()

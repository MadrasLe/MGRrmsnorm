"""
Benchmark Continuous Batching — FP16 vs INT8 vs AWQ INT4
Testa throughput (tok/s total) com batch sizes crescentes.
Roda no Colab: %run examples/bench_continuous_batching.py
"""
import os, sys, gc, time, torch

MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen2.5-7B-Instruct")
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
MAX_NEW_TOKENS = 64
PROMPT = "Explique relatividade."


def clear_megagemm_modules():
    for m in [k for k in list(sys.modules) if k.startswith("megagemm")]:
        del sys.modules[m]


def bench_mode_batch(mode, batch_sizes):
    """Run continuous batching benchmark for a given mode."""
    gc.collect()
    torch.cuda.empty_cache()
    clear_megagemm_modules()

    from megagemm.engine import InferenceEngine

    label = {"fp16": "🔵 FP16", "int8": "🟢 INT8", "awq": "🟡 AWQ INT4"}[mode]
    print(f"\n{'='*64}")
    print(f"{label}")
    print(f"{'='*64}")

    # Setup engine
    kw = dict(max_batch_size=max(batch_sizes) + 1, max_seq_len=2048)
    model_name = MODEL
    if mode == "int8":
        kw["quantize"] = "int8"
    elif mode == "awq":
        pass

    if mode == "awq":
        model_name = MODEL.replace("-Instruct", "-Instruct-AWQ")

    try:
        engine = InferenceEngine(model_name, **kw)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"  ⚠️ OOM: {e}")
        return None

    # Warmup
    _ = engine.generate("warmup", max_new_tokens=16, temperature=0.0, repetition_penalty=1.0)
    torch.cuda.synchronize()

    results = []
    for bs in batch_sizes:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        prompts = [PROMPT] * bs

        try:
            # Timed run
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = engine.generate_batch(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            total_tokens = sum(len(engine.tokenizer.encode(o)) for o in outputs)
            tok_per_sec = total_tokens / elapsed
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

            print(f"  batch={bs:2d} | {tok_per_sec:7.1f} tok/s total | "
                  f"{tok_per_sec/bs:6.1f} tok/s/req | {elapsed:.2f}s | peak {peak_gb:.2f} GB")
            results.append((bs, tok_per_sec, tok_per_sec/bs, peak_gb))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"  batch={bs:2d} | OOM — {e}")
            gc.collect()
            torch.cuda.empty_cache()
            break  # larger batches will also OOM

    # Cleanup
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    clear_megagemm_modules()

    return mode, results


# ═══════════════════════════════════════════════
# Run benchmarks
# ═══════════════════════════════════════════════
print(f"Model: {MODEL}")
print(f"Batch sizes: {BATCH_SIZES}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")

all_results = {}
modes_str = os.environ.get("BENCH_MODES", "fp16,int8,awq")
modes = [m.strip() for m in modes_str.split(",")]
for mode in modes:
    try:
        r = bench_mode_batch(mode, BATCH_SIZES)
        if r:
            all_results[r[0]] = r[1]
    except Exception as e:
        print(f"\n  ⚠️ {mode.upper()} failed: {e}")
        gc.collect()
        torch.cuda.empty_cache()

# Summary table
print(f"\n{'═'*72}")
print(f"RESUMO — {MODEL} — Continuous Batching")
print(f"{'═'*72}")
print(f"{'Batch':>5s} | {'FP16':>12s} | {'INT8':>12s} | {'AWQ INT4':>12s}")
print(f"{'─'*5}-+-{'─'*12}-+-{'─'*12}-+-{'─'*12}")

for i, bs in enumerate(BATCH_SIZES):
    row = f"{bs:5d} |"
    for mode in ["fp16", "int8", "awq"]:
        if mode in all_results and i < len(all_results[mode]):
            _, tps, _, _ = all_results[mode][i]
            row += f" {tps:9.1f} t/s |"
        else:
            row += f"         N/A |"
    print(row)

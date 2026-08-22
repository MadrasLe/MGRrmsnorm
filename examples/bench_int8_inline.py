"""
Benchmark INT8 inline decode vs FP16 — with flat decode debug.
Roda no Colab: cola tudo numa célula.
"""
import os, sys, gc, time, statistics, torch

# ===== setup =====
# %cd /content/drive/MyDrive/MGRrmsnorm
# !pip install -e . --no-build-isolation -q 2>&1 | tail -2

MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen2.5-7B-Instruct")
PROMPT_SHOW = "Explique o que é inteligência artificial em 3 frases curtas."
PROMPT_BENCH = "Explique relatividade."
MAX_NEW_TOKENS_SHOW = 128
MAX_NEW_TOKENS_BENCH = 256
RUNS = 4


def clear_megagemm_modules():
    for m in [k for k in list(sys.modules) if k.startswith("megagemm")]:
        del sys.modules[m]


def bench_mode(mode: str):
    clear_megagemm_modules()
    from megagemm.engine import InferenceEngine

    print(f"\n{'='*60}")
    label = {"fp16": "🔵 FP16", "int8": "🟢 INT8", "awq": "🟡 AWQ INT4"}[mode]
    print(label)
    print(f"{'='*60}")

    kw = dict(max_batch_size=1, max_seq_len=4096)
    if mode == "int8":
        kw["quantize"] = "int8"
    elif mode == "awq":
        pass  # AWQ is auto-detected from model name below

    model_name = MODEL
    if mode == "awq":
        model_name = MODEL.replace("-Instruct", "-Instruct-AWQ")
    engine = InferenceEngine(model_name, **kw)

    # warmup (triggers _prepare_flat_decode)
    _ = engine.generate("warmup", max_new_tokens=64, temperature=0.0, repetition_penalty=1.0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # debug flat decode status
    model = engine.model
    flat_ready = getattr(model, '_flat_decode_ready', False)
    flat_failed = getattr(model, '_flat_decode_failed', False)
    flat_inline = getattr(model, '_flat_inline_kernels', False)
    int8_inline = getattr(model, '_flat_int8_inline', False)

    print(f"  flat_ready={flat_ready} failed={flat_failed} "
          f"inline={flat_inline} int8_inline={int8_inline}")

    # Check W8A16 GEMV availability
    try:
        from megagemm.models.llama import _HAS_FLAT_W8A16_GEMV
        print(f"  W8A16_GEMV={_HAS_FLAT_W8A16_GEMV}")
    except Exception:
        print("  W8A16_GEMV=import_failed")

    # Check W4A16 GEMV availability
    try:
        from megagemm.models.llama import _HAS_FLAT_W4A16_GEMV
        w4a16_ready = getattr(model, '_flat_w4a16_ready', False)
        print(f"  W4A16_GEMV={_HAS_FLAT_W4A16_GEMV} ready={w4a16_ready}")
    except Exception:
        print("  W4A16_GEMV=import_failed")

    # If flat decode failed, try to get the actual error
    if not flat_ready and not flat_failed:
        print("  ⚠️  flat decode never called — forcing _prepare_flat_decode()...")
        try:
            model._prepare_flat_decode()
            print(f"  → ready={model._flat_decode_ready} failed={model._flat_decode_failed}")
        except Exception as e:
            print(f"  → EXCEPTION: {e}")
            import traceback; traceback.print_exc()
    elif flat_failed:
        print("  ⚠️  flat decode FAILED — re-running with traceback...")
        model._flat_decode_ready = False
        model._flat_decode_failed = False
        try:
            model._prepare_flat_decode()
            print(f"  → ready={model._flat_decode_ready} failed={model._flat_decode_failed}")
        except Exception as e:
            print(f"  → EXCEPTION: {e}")
            import traceback; traceback.print_exc()
        # Check the silent failure path
        if model._flat_decode_failed and not model._flat_decode_ready:
            print("  → Silent failure inside try/except — adding debug...")
            model._flat_decode_ready = False
            model._flat_decode_failed = False
            # Monkey-patch to catch the actual error
            import types
            original = model._prepare_flat_decode.__func__

            def debug_prepare(self_inner):
                try:
                    # Re-import to get fresh state
                    cfg = self_inner.config
                    print(f"    _all_full_attention={self_inner._all_full_attention}")
                    print(f"    layers={len(self_inner.layers)}")
                    layer0 = self_inner.layers[0]
                    attn0 = layer0.self_attn
                    mlp0 = layer0.mlp
                    print(f"    qkv_proj type={type(attn0.qkv_proj).__name__}")
                    print(f"    o_proj type={type(attn0.o_proj).__name__}")
                    print(f"    gate_up type={type(mlp0.gate_up_proj).__name__}")
                    print(f"    down type={type(mlp0.down_proj).__name__}")
                    print(f"    awq_separate: attn={attn0._awq_separate} mlp={mlp0._awq_separate}")
                    # Try the weight extraction
                    from megagemm.models.llama import _linear_weight_bias
                    w, b = _linear_weight_bias(attn0.qkv_proj)
                    print(f"    qkv weight={w is not None} bias={b is not None}")
                    if hasattr(attn0.qkv_proj, 'weight_int8'):
                        print(f"    qkv weight_int8 shape={attn0.qkv_proj.weight_int8.shape}")
                        print(f"    qkv scale shape={attn0.qkv_proj.scale.shape}")
                except Exception as e2:
                    print(f"    DEBUG EXCEPTION: {e2}")
                    import traceback; traceback.print_exc()

            debug_prepare(model)

    try:
        print(f"  qkv_proj: {model.layers[0].self_attn.qkv_proj}")
    except Exception:
        pass

    # sample output
    out = engine.generate(
        PROMPT_SHOW, max_new_tokens=MAX_NEW_TOKENS_SHOW,
        temperature=0.0, repetition_penalty=1.0,
    )
    print(f"\n📝 Output:\n{out}\n")

    vals = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        o = engine.generate(
            PROMPT_BENCH, max_new_tokens=MAX_NEW_TOKENS_BENCH,
            temperature=0.0, repetition_penalty=1.0,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ntok = len(engine.tokenizer.encode(o, add_special_tokens=False))
        tps = ntok / dt
        vals.append(tps)
        print(f"  run {i+1}: {tps:.1f} tok/s ({ntok} toks)")

    peak = torch.cuda.max_memory_allocated() / 1e9
    curr = torch.cuda.memory_allocated() / 1e9
    avg = statistics.mean(vals[1:]) if len(vals) > 1 else statistics.mean(vals)
    print(f"  avg={avg:.1f} tok/s | VRAM_peak={peak:.2f}GB | VRAM_current={curr:.2f}GB")

    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return mode, avg, peak, curr


results = []
for mode in ["fp16", "int8", "awq"]:
    try:
        results.append(bench_mode(mode))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"\n  {mode.upper()} skipped (OOM): {e}")
        gc.collect()
        torch.cuda.empty_cache()

print(f"\n{'─'*64}")
print(f"RESUMO — {MODEL}")
print(f"{'─'*64}")
for mode, avg, peak, curr in results:
    label = {"fp16": "FP16", "int8": "INT8", "awq": "AWQ INT4"}[mode]
    print(f"{label:10s} | {avg:7.1f} tok/s | peak {peak:.2f} GB | current {curr:.2f} GB")

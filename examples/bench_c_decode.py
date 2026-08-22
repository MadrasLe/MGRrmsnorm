#!/usr/bin/env python3
"""
🏁 C Decode Loop Benchmark
============================
Python decode vs C decode vs llama.cpp
"""
import torch
import time
import sys
import os

# Configurable model: pass as CLI arg or env var
# Examples: Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-3B-Instruct
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT_TEXT = "Explain what gravity is in two sentences."
MAX_TOKENS = 32

torch.set_num_threads(os.cpu_count())


def main():
    from megagemm.engine import InferenceEngine
    from megagemm.kernels.cpu_decode import CPUDecoder

    print("=" * 70)
    print("  🏁 C Decode Loop Benchmark")
    print("=" * 70)
    print(f"  Threads: {torch.get_num_threads()}")
    print()

    # Load engine
    print("  Loading model...", flush=True)
    engine = InferenceEngine(MODEL, device='cpu', dtype=torch.float32)
    tokenizer = engine.tokenizer

    # ── 1. Python baseline ──
    print()
    print("─" * 70)
    print("  [1] Python decode (baseline)")
    engine.generate(PROMPT_TEXT, max_new_tokens=MAX_TOKENS, temperature=0.0, verbose=False)

    times = []
    out_py = None
    for _ in range(3):
        t0 = time.perf_counter()
        out_py = engine.generate(PROMPT_TEXT, max_new_tokens=MAX_TOKENS, temperature=0.0, verbose=False)
        times.append(time.perf_counter() - t0)
    avg_py = sum(times) / len(times)
    tps_py = MAX_TOKENS / avg_py
    print(f"  → {tps_py:.1f} tok/s")
    print(f"  → {out_py[:70]}...")

    # ── 2. C decode loop (INT8) ──
    print()
    print("─" * 70)
    print("  [2] C decode loop (INT8)")

    decoder = CPUDecoder(engine.model, engine.model.config, quant='int8')
    block_manager = engine.block_manager

    # Tokenize
    tokens = tokenizer.encode(PROMPT_TEXT)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    positions = torch.arange(len(tokens), dtype=torch.long).unsqueeze(0)

    # Prefill via Python
    seq_id = 0
    total_needed = len(tokens) + MAX_TOKENS + 16  # extra margin
    block_manager.allocate_sequence(seq_id, total_needed)
    logits = engine.model.prefill(input_ids, positions, block_manager, seq_id)
    first_token = logits[0, -1, :].argmax().item()
    start_pos = len(tokens)

    bt = block_manager.get_block_table_tensor([seq_id])[0].int()

    # Test single step
    print("  Single step test...", end=" ", flush=True)
    try:
        next_id, _ = decoder.decode_step(first_token, start_pos, block_manager, bt, start_pos)
        print(f"token={next_id} '{tokenizer.decode([next_id])}' ✅")
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
        return

    # Full generate via C multi-step
    print("  Full generation test...", end=" ", flush=True)
    block_manager.free_sequence(seq_id)
    block_manager.allocate_sequence(seq_id, total_needed)
    logits = engine.model.prefill(input_ids, positions, block_manager, seq_id)
    first_token = logits[0, -1, :].argmax().item()
    bt = block_manager.get_block_table_tensor([seq_id])[0].int()

    t0 = time.perf_counter()
    gen_ids = decoder.generate(
        first_token, start_pos, block_manager, bt,
        max_tokens=MAX_TOKENS, eos_token=tokenizer.eos_token_id or 151643,
    )
    dt = time.perf_counter() - t0
    tps_i8 = len(gen_ids) / dt
    out_i8 = tokenizer.decode(gen_ids)
    print(f"{tps_i8:.1f} tok/s  ({len(gen_ids)} tokens in {dt:.2f}s)")
    print(f"  → {out_i8[:70]}...")

    # ── 3. C decode loop (INT4) ──
    print()
    print("─" * 70)
    print("  [3] C decode loop (INT4)")

    decoder4 = CPUDecoder(engine.model, engine.model.config, quant='int4')

    block_manager.free_sequence(seq_id)
    block_manager.allocate_sequence(seq_id, total_needed)
    logits = engine.model.prefill(input_ids, positions, block_manager, seq_id)
    first_token = logits[0, -1, :].argmax().item()
    bt = block_manager.get_block_table_tensor([seq_id])[0].int()
    start_pos = len(tokens)

    # Single step test
    print("  Single step test...", end=" ", flush=True)
    try:
        next_id, _ = decoder4.decode_step(first_token, start_pos, block_manager, bt, start_pos)
        print(f"token={next_id} '{tokenizer.decode([next_id])}' ✅")
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
        return

    # Full generate
    print("  Full generation test...", end=" ", flush=True)
    block_manager.free_sequence(seq_id)
    block_manager.allocate_sequence(seq_id, total_needed)
    logits = engine.model.prefill(input_ids, positions, block_manager, seq_id)
    first_token = logits[0, -1, :].argmax().item()
    bt = block_manager.get_block_table_tensor([seq_id])[0].int()

    t0 = time.perf_counter()
    gen_ids4 = decoder4.generate(
        first_token, start_pos, block_manager, bt,
        max_tokens=MAX_TOKENS, eos_token=tokenizer.eos_token_id or 151643,
    )
    dt = time.perf_counter() - t0
    tps_i4 = len(gen_ids4) / dt
    out_i4 = tokenizer.decode(gen_ids4)
    print(f"{tps_i4:.1f} tok/s  ({len(gen_ids4)} tokens in {dt:.2f}s)")
    print(f"  → {out_i4[:70]}...")

    # ── 4. Batch decode ──
    print()
    print("─" * 70)
    print("  [4] Batch decode (INT8)")

    PROMPTS = [
        "Explain what gravity is in two sentences.",
        "What is the speed of light?",
        "How does photosynthesis work?",
        "What causes earthquakes?",
    ]

    batch_results = {}
    for batch_size in [1, 2, 4]:
        prompts = PROMPTS[:batch_size]

        # Prefill each sequence
        seq_ids = list(range(batch_size))
        first_tokens = []
        start_positions = []
        bts = []

        for i, prompt in enumerate(prompts):
            tok = tokenizer.encode(prompt)
            ids = torch.tensor([tok], dtype=torch.long)
            pos = torch.arange(len(tok), dtype=torch.long).unsqueeze(0)
            total = len(tok) + MAX_TOKENS + 16

            try:
                block_manager.free_sequence(seq_ids[i])
            except (KeyError, ValueError):
                pass
            block_manager.allocate_sequence(seq_ids[i], total)
            logits = engine.model.prefill(ids, pos, block_manager, seq_ids[i])
            first_tokens.append(logits[0, -1, :].argmax().item())
            start_positions.append(len(tok))
            bts.append(block_manager.get_block_table_tensor([seq_ids[i]])[0].int())

        # Run batch decode loop
        cur_tokens = list(first_tokens)
        cur_positions = list(start_positions)
        cur_seq_lens = list(start_positions)
        all_gen = [[] for _ in range(batch_size)]
        eos = tokenizer.eos_token_id or 151643

        t0 = time.perf_counter()
        for step in range(MAX_TOKENS):
            next_tokens = decoder.batch_decode_step(
                cur_tokens, cur_positions, block_manager, bts, cur_seq_lens)
            for i in range(batch_size):
                all_gen[i].append(next_tokens[i])
                cur_tokens[i] = next_tokens[i]
                cur_positions[i] += 1
                cur_seq_lens[i] += 1
        dt = time.perf_counter() - t0

        total_tokens = batch_size * MAX_TOKENS
        tps_total = total_tokens / dt
        tps_per_seq = MAX_TOKENS / dt
        batch_results[batch_size] = tps_total

        print(f"  batch={batch_size}: {tps_total:.1f} total tok/s "
              f"({tps_per_seq:.1f}/seq, {batch_size}×{MAX_TOKENS}={total_tokens} tokens in {dt:.2f}s)")
        for i in range(min(batch_size, 2)):
            text = tokenizer.decode(all_gen[i])
            print(f"    seq{i}: {text[:60]}...")

        # Cleanup
        for sid in seq_ids:
            try:
                block_manager.free_sequence(sid)
            except (KeyError, ValueError):
                pass

    # ── Summary ──
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    speedup_i8 = tps_i8 / tps_py if tps_py > 0 else 0
    speedup_i4 = tps_i4 / tps_py if tps_py > 0 else 0
    print(f"  Python:     {tps_py:>6.1f} tok/s │ {'█' * int(tps_py)}")
    print(f"  C INT8:     {tps_i8:>6.1f} tok/s │ {'█' * min(int(tps_i8), 100)}  ({speedup_i8:.2f}x)")
    print(f"  C INT4:     {tps_i4:>6.1f} tok/s │ {'█' * min(int(tps_i4), 100)}  ({speedup_i4:.2f}x)")
    for bs, tps in batch_results.items():
        label = f"Batch={bs}"
        print(f"  {label:<10}  {tps:>6.1f} tok/s │ {'█' * min(int(tps), 100)}  (total)")
    print(f"  llama.cpp:  {'75.3':>6} tok/s │ {'█' * 75}  (reference)")
    pct_i8 = tps_i8 / 75.3 * 100
    pct_i4 = tps_i4 / 75.3 * 100
    print(f"  → INT8 is {pct_i8:.0f}% of llama.cpp")
    print(f"  → INT4 is {pct_i4:.0f}% of llama.cpp")
    print("=" * 70)


if __name__ == "__main__":
    main()

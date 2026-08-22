"""
🧪 KV Cache Persistence — End-to-end Test
==========================================
Tests save/restore of KV cache with Qwen3-4B FP16.

Run on Colab with T4/L4:
    python tests/test_kv_persist.py

Author: Gabriel Yogi
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import time
import tempfile

# ═══════════════════════════════════════════════
# TEST 1: BlockManager round-trip (CPU, no model)
# ═══════════════════════════════════════════════
def _run_blockmanager_roundtrip():
    """Serialize → deserialize → verify bit-exact."""
    from megagemm.engine.kv_cache import BlockManager

    print("=" * 60)
    print("TEST 1: BlockManager Round-Trip (CPU)")
    print("=" * 60)

    bm = BlockManager(
        num_layers=4, num_blocks=32, block_size=16,
        num_kv_heads=4, head_dim=32, dtype=torch.float32, device='cpu'
    )

    # Write random data
    bm.allocate_sequence(seq_id=0, num_tokens=48)
    for layer in range(4):
        bm.write_kv(0, layer, torch.randn(48, 4, 32), torch.randn(48, 4, 32))
    bm.advance_seq_len(0, 48)

    # Serialize
    snapshot = bm.serialize_sequence(0)
    print(f"  Serialized: seq_len={snapshot['seq_len']}, "
          f"kv_data shape={snapshot['kv_data'][0].shape}")

    # Deserialize into new BlockManager
    bm2 = BlockManager(
        num_layers=4, num_blocks=32, block_size=16,
        num_kv_heads=4, head_dim=32, dtype=torch.float32, device='cpu'
    )
    bm2.deserialize_sequence(99, snapshot)

    # Verify bit-exact
    for layer in range(4):
        for i in range(3):
            orig = bm.kv_caches[layer][bm.block_tables[0][i]]
            rest = bm2.kv_caches[layer][bm2.block_tables[99][i]]
            assert torch.allclose(orig, rest), f"Layer {layer} block {i} mismatch!"

    print("  ✅ Data is bit-exact!")

    # Test file save/load
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = tmp.name
    try:
        torch.save(snapshot, path)
        loaded = torch.load(path, weights_only=False)
        assert loaded['seq_len'] == 48
        for layer in range(4):
            assert torch.allclose(snapshot['kv_data'][layer], loaded['kv_data'][layer])
    finally:
        os.remove(path)
    print("  ✅ File save/load OK!")

    # Test config validation
    bm3 = BlockManager(
        num_layers=2, num_blocks=32, block_size=16,
        num_kv_heads=4, head_dim=32, dtype=torch.float32, device='cpu'
    )
    try:
        bm3.deserialize_sequence(0, snapshot)
        print("  ❌ Should have raised ValueError!")
        return False
    except ValueError as e:
        print(f"  ✅ Config validation: {e}")

    print("  🎉 TEST 1 PASSED!\n")
    return True


# ═══════════════════════════════════════════════
# TEST 2: Full E2E with Qwen3-4B (GPU required)
# ═══════════════════════════════════════════════
def _run_e2e_qwen3():
    """
    Full end-to-end test:
    1. Load Qwen3-4B
    2. Prefill a prompt
    3. Generate 20 tokens (greedy) → save KV cache
    4. Continue generating 20 more tokens → get full output
    5. Restore KV cache from step 3
    6. Generate 20 tokens from restored → should match step 4
    """
    if not torch.cuda.is_available():
        print("⏭️  Skipping E2E test (no CUDA)\n")
        return True

    from megagemm.engine import InferenceEngine

    print("=" * 60)
    print("TEST 2: E2E KV Cache Persistence (Qwen3-4B)")
    print("=" * 60)

    # ── Load model ──
    print("  Loading model...")
    engine = InferenceEngine(
        "Qwen/Qwen3-4B",
        dtype=torch.float16,
        max_batch_size=4,
        max_seq_len=512,
    )
    print(f"  {engine}")

    prompt = "The theory of general relativity predicts that"

    # ── Generate: prefill + 40 tokens greedy ──
    print(f"\n  Prompt: '{prompt}'")
    print("  Generating 40 tokens (greedy)...")

    # Manually do prefill + decode so we can save KV mid-generation
    bos = engine.tokenizer.bos_token
    already_fmt = bos and prompt.startswith(bos)
    if not already_fmt and hasattr(engine.tokenizer, 'chat_template') and engine.tokenizer.chat_template:
        try:
            msgs = [{"role": "user", "content": prompt}]
            formatted = engine.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt
    else:
        formatted = prompt

    add_special = not (bos and formatted.startswith(bos))
    input_ids = engine.tokenizer.encode(
        formatted, return_tensors='pt', add_special_tokens=add_special
    ).to(engine.device)
    prompt_len = input_ids.shape[1]

    seq_id = 0
    engine.block_manager.allocate_sequence(seq_id, prompt_len + 50)

    # Prefill
    positions = torch.arange(prompt_len, device=engine.device).unsqueeze(0)
    with torch.inference_mode():
        logits = engine.model.prefill(
            input_ids, positions, engine.block_manager, seq_id
        )

    # Decode 20 tokens (greedy)
    generated_ids_part1 = []
    next_token = logits[:, -1, :].argmax(dim=-1).item()
    generated_ids_part1.append(next_token)

    decode_input = torch.empty(1, 1, dtype=torch.long, device=engine.device)
    decode_pos = torch.empty(1, 1, dtype=torch.long, device=engine.device)

    with torch.inference_mode():
        for step in range(19):
            decode_input.fill_(next_token)
            decode_pos.fill_(prompt_len + step)
            logits = engine.model.decode_step(
                decode_input, decode_pos,
                engine.block_manager, [seq_id],
            )
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            generated_ids_part1.append(next_token)

    text_at_20 = engine.tokenizer.decode(generated_ids_part1, skip_special_tokens=True)
    print(f"  After 20 tokens: '{text_at_20[:80]}...'")

    # ── SAVE KV CACHE at 20 tokens ──
    snapshot = engine.save_context(seq_id, text=prompt)
    kv_size_mb = sum(d.nelement() * d.element_size() for d in snapshot['kv_data']) / 1e6
    print(f"  💾 Saved KV: seq_len={snapshot['seq_len']}, size={kv_size_mb:.1f}MB")

    # ── Continue generating 20 MORE tokens (greedy) from original seq ──
    generated_ids_part2_original = []
    with torch.inference_mode():
        for step in range(20):
            decode_input.fill_(next_token)
            decode_pos.fill_(prompt_len + 20 + step)
            logits = engine.model.decode_step(
                decode_input, decode_pos,
                engine.block_manager, [seq_id],
            )
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            generated_ids_part2_original.append(next_token)

    full_text_original = engine.tokenizer.decode(
        generated_ids_part1 + generated_ids_part2_original,
        skip_special_tokens=True
    )
    print(f"  Full 40 tokens (original): '{full_text_original[:120]}...'")

    # Free original sequence
    engine.block_manager.free_sequence(seq_id)

    # ── RESTORE KV CACHE into new sequence ──
    restored_seq_id = engine.restore_context(snapshot, seq_id=1, max_new_tokens=50)
    print(f"  📂 Restored KV: seq_id={restored_seq_id}, seq_len={snapshot['seq_len']}")

    # ── Generate 20 tokens from RESTORED seq (greedy) ──
    # Must start from the last token of part1
    next_token = generated_ids_part1[-1]
    generated_ids_part2_restored = []

    with torch.inference_mode():
        for step in range(20):
            decode_input.fill_(next_token)
            decode_pos.fill_(prompt_len + 20 + step)
            logits = engine.model.decode_step(
                decode_input, decode_pos,
                engine.block_manager, [restored_seq_id],
            )
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            generated_ids_part2_restored.append(next_token)

    full_text_restored = engine.tokenizer.decode(
        generated_ids_part1 + generated_ids_part2_restored,
        skip_special_tokens=True
    )
    print(f"  Full 40 tokens (restored): '{full_text_restored[:120]}...'")

    # ── VERIFY: outputs must be identical ──
    match = generated_ids_part2_original == generated_ids_part2_restored
    if match:
        print("\n  ✅ OUTPUTS ARE BIT-EXACT! KV restore works perfectly!")
    else:
        print(f"\n  ❌ MISMATCH!")
        print(f"     Original:  {generated_ids_part2_original[:10]}...")
        print(f"     Restored:  {generated_ids_part2_restored[:10]}...")

    # ── Test file save/load ──
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        save_path = tmp.name
    try:
        engine.save_context_to_file(seq_id=restored_seq_id, path=save_path, text=prompt)
        file_size = os.path.getsize(save_path) / 1e6
        print(f"  File size: {file_size:.1f}MB")

        engine.block_manager.free_sequence(restored_seq_id)

        loaded_seq = engine.restore_context_from_file(save_path, seq_id=2, max_new_tokens=50)
        print(f"  ✅ File round-trip: saved → loaded seq_id={loaded_seq}")

        engine.block_manager.free_sequence(loaded_seq)
    finally:
        os.remove(save_path)

    print(f"\n  🎉 TEST 2 {'PASSED' if match else 'FAILED'}!\n")
    return match

# ═══════════════════════════════════════════════
# TEST 3: Embedding Extraction (GPU required)
# ═══════════════════════════════════════════════
def _run_embedding():
    """
    Test semantic embedding extraction:
    1. Similar texts → high cosine similarity
    2. Dissimilar texts → low cosine similarity
    3. Embedding auto-attached to snapshot
    """
    if not torch.cuda.is_available():
        print("⏭️  Skipping embedding test (no CUDA)\n")
        return True

    from megagemm.engine import InferenceEngine

    print("=" * 60)
    print("TEST 3: Embedding Extraction")
    print("=" * 60)

    engine = InferenceEngine(
        "Qwen/Qwen3-4B",
        dtype=torch.float16,
        max_batch_size=4,
        max_seq_len=512,
    )

    # Extract embeddings for similar and dissimilar texts
    e1 = engine.extract_embedding("What is attention in transformers?")
    e2 = engine.extract_embedding("How does the attention mechanism work?")
    e3 = engine.extract_embedding("Recipe for chocolate cake with vanilla frosting")

    sim_12 = torch.dot(e1, e2).item()
    sim_13 = torch.dot(e1, e3).item()
    sim_23 = torch.dot(e2, e3).item()

    print(f"  e1: 'attention in transformers'     shape={e1.shape}")
    print(f"  e2: 'attention mechanism work'       shape={e2.shape}")
    print(f"  e3: 'chocolate cake recipe'          shape={e3.shape}")
    print(f"  sim(e1, e2) = {sim_12:.4f}  (similar topics → should be HIGH)")
    print(f"  sim(e1, e3) = {sim_13:.4f}  (different topics → should be LOW)")
    print(f"  sim(e2, e3) = {sim_23:.4f}  (different topics → should be LOW)")

    ok = sim_12 > sim_13 and sim_12 > sim_23
    if ok:
        print("  ✅ Similar texts have higher cosine similarity!")
    else:
        print("  ❌ Similarity ordering unexpected")

    # Test embedding in snapshot
    seq_id = 0
    engine.block_manager.allocate_sequence(seq_id, 32)
    input_ids = engine.tokenizer.encode("test", return_tensors='pt').to(engine.device)
    positions = torch.arange(input_ids.shape[1], device=engine.device).unsqueeze(0)
    with torch.inference_mode():
        engine.model.prefill(input_ids, positions, engine.block_manager, seq_id)

    snapshot = engine.save_context(seq_id, text="test prompt about AI")
    engine.block_manager.free_sequence(seq_id)

    has_emb = 'embedding' in snapshot
    emb_shape = snapshot['embedding'].shape if has_emb else None
    print(f"  Snapshot has embedding: {has_emb}, shape: {emb_shape}")

    if has_emb:
        print("  ✅ Embedding auto-attached to snapshot!")
    else:
        print("  ❌ Embedding missing from snapshot")
        ok = False

    print(f"\n  🎉 TEST 3 {'PASSED' if ok else 'FAILED'}!\n")
    return ok


# ═══════════════════════════════════════════════
# Pytest entry points. The reusable runners keep the standalone CLI summary,
# while these wrappers make a returned False an actual test failure.
# ═══════════════════════════════════════════════
def test_blockmanager_roundtrip():
    assert _run_blockmanager_roundtrip()


def test_e2e_qwen3():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Qwen3 persistence integration test")
    assert _run_e2e_qwen3()


def test_embedding():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for embedding integration validation")
    assert _run_embedding()


# Main
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print("🔥 MegaGemm KV Cache Persistence Tests")
    print("=" * 60 + "\n")

    results = {}
    results['blockmanager_roundtrip'] = _run_blockmanager_roundtrip()
    results['e2e_qwen3'] = _run_e2e_qwen3()
    results['embedding'] = _run_embedding()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    all_ok = all(results.values())
    print(f"\n{'🎉 All tests passed!' if all_ok else '⚠️ Some tests failed'}")
    sys.exit(0 if all_ok else 1)

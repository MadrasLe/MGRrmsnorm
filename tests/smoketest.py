"""Quick smoketest for imports and basic logic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Test 1: kv_cache...", end=" ")
from megagemm.kv_cache import BlockManager
print("OK")

print("Test 2: paged_attention...", end=" ")
from megagemm.paged_attention import paged_attention_decode, prefill_attention
print("OK")

print("Test 3: sampling...", end=" ")
from megagemm.sampling import sample_logits
print("OK")

print("Test 4: models.llama...", end=" ")
from megagemm.models.llama import LlamaConfig, MegaGemmLlama
print("OK")

print("Test 5: models.loader...", end=" ")
from megagemm.models.loader import load_from_hf
print("OK")

print("Test 6: engine...", end=" ")
from megagemm.engine import InferenceEngine
print("OK")

print("\n--- KV Cache CPU Test ---")
import torch
bm = BlockManager(num_layers=2, num_blocks=32, block_size=16,
                  num_kv_heads=4, head_dim=32, dtype=torch.float32, device='cpu')
bm.allocate_sequence(0, 32)
k = torch.randn(32, 4, 32)
v = torch.randn(32, 4, 32)
bm.write_kv(0, 0, k, v)
bm.write_kv(0, 1, k, v)
bm.advance_seq_len(0, 32)
print(f"  BlockManager: {bm}")
print(f"  seq_len: {bm.seq_lens[0]}")
print(f"  block_table: {bm.block_tables[0]}")
bm.free_sequence(0)
print(f"  After free: {bm.num_free_blocks} free blocks")

print("\n--- Sampling CPU Test ---")
logits = torch.randn(1, 100)
tok = sample_logits(logits, temperature=0.0)
print(f"  Greedy: {tok.item()} (expected: {logits.argmax().item()})")
assert tok.item() == logits.argmax().item()

print("\n--- Model CPU Test ---")
cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                  vocab_size=256, max_position_embeddings=64)
model = MegaGemmLlama(cfg)
bm2 = BlockManager(num_layers=2, num_blocks=16, block_size=16,
                   num_kv_heads=2, head_dim=16, dtype=torch.float32, device='cpu')
bm2.allocate_sequence(0, 8)
ids = torch.randint(0, 256, (1, 8))
pos = torch.arange(8).unsqueeze(0)
with torch.no_grad():
    logits = model.prefill(ids, pos, bm2, 0)
print(f"  Prefill logits: {logits.shape} (expected: [1, 8, 256])")
assert logits.shape == (1, 8, 256)
print(f"  seq_len after prefill: {bm2.seq_lens[0]}")

# Decode step
decode_ids = torch.tensor([[logits[0, -1].argmax().item()]])
decode_pos = torch.tensor([[8]])
with torch.no_grad():
    logits2 = model.decode_step(decode_ids, decode_pos, bm2, [0])
print(f"  Decode logits: {logits2.shape} (expected: [1, 1, 256])")
assert logits2.shape == (1, 1, 256)
print(f"  seq_len after decode: {bm2.seq_lens[0]}")

print("\n🎉 All CPU tests passed!")

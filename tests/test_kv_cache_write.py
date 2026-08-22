import torch

from megagemm.engine.kv_cache import BlockManager


def test_idle_block_manager_restores_canonical_allocation_order():
    bm = BlockManager(
        num_layers=1,
        num_blocks=32,
        block_size=4,
        num_kv_heads=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )

    def allocate_batch():
        tables = []
        for seq_id in range(4):
            bm.allocate_sequence(seq_id, num_tokens=8)
            tables.append(tuple(bm.block_tables[seq_id]))
        return tuple(tables)

    first_tables = allocate_batch()
    for seq_id in range(4):
        bm.free_sequence(seq_id)

    assert bm.free_blocks == list(range(bm.num_blocks))
    assert bm._idle_free_order_resets == 1
    assert allocate_batch() == first_tables


def test_write_kv_prefill_cross_block_layout():
    torch.manual_seed(0)
    block_size = 4
    num_heads = 2
    head_dim = 3
    cur_len = 3
    num_new = 9

    bm = BlockManager(
        num_layers=1,
        num_blocks=16,
        block_size=block_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
    )
    seq_id = 11
    bm.allocate_sequence(seq_id, num_tokens=cur_len + num_new)
    bm.seq_lens[seq_id] = cur_len

    k = torch.randn(num_new, num_heads, head_dim)
    v = torch.randn(num_new, num_heads, head_dim)
    bm.write_kv(seq_id, 0, k, v)

    cache = bm.get_kv_cache(0)
    blocks = bm.block_tables[seq_id]

    for t in range(num_new):
        pos = cur_len + t
        blk = pos // block_size
        off = pos % block_size
        phys = blocks[blk]
        assert torch.allclose(cache[phys, 0, :, off, :], k[t], atol=0.0, rtol=0.0)
        assert torch.allclose(cache[phys, 1, :, off, :], v[t], atol=0.0, rtol=0.0)


def test_write_kv_single_token_fast_path():
    torch.manual_seed(0)
    block_size = 8
    num_heads = 2
    head_dim = 4
    cur_len = 5

    bm = BlockManager(
        num_layers=1,
        num_blocks=8,
        block_size=block_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
    )
    seq_id = 3
    bm.allocate_sequence(seq_id, num_tokens=cur_len + 1)
    bm.seq_lens[seq_id] = cur_len

    k = torch.randn(1, num_heads, head_dim)
    v = torch.randn(1, num_heads, head_dim)
    bm.write_kv(seq_id, 0, k, v)

    cache = bm.get_kv_cache(0)
    blocks = bm.block_tables[seq_id]
    phys = blocks[cur_len // block_size]
    off = cur_len % block_size

    assert torch.allclose(cache[phys, 0, :, off, :], k[0], atol=0.0, rtol=0.0)
    assert torch.allclose(cache[phys, 1, :, off, :], v[0], atol=0.0, rtol=0.0)


def test_sparse_kv_layers_roundtrip():
    torch.manual_seed(0)
    block_size = 4
    num_heads = 2
    head_dim = 3
    kv_layers = [0, 2]

    bm = BlockManager(
        num_layers=4,
        num_blocks=8,
        block_size=block_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
        kv_layer_indices=kv_layers,
    )
    seq_id = 5
    bm.allocate_sequence(seq_id, num_tokens=8)

    assert bm.get_kv_cache(1) is None
    assert bm.get_kv_cache(3) is None

    k = torch.randn(4, num_heads, head_dim)
    v = torch.randn(4, num_heads, head_dim)
    bm.write_kv(seq_id, 0, k, v)
    bm.advance_seq_len(seq_id, 4)

    snap = bm.serialize_sequence(seq_id)
    assert snap["kv_layer_indices"] == kv_layers
    assert snap["kv_data_by_layer"][0] is not None
    assert snap["kv_data_by_layer"][2] is not None
    assert 1 not in snap["kv_data_by_layer"]
    assert 3 not in snap["kv_data_by_layer"]

    bm2 = BlockManager(
        num_layers=4,
        num_blocks=8,
        block_size=block_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
        kv_layer_indices=kv_layers,
    )
    bm2.deserialize_sequence(9, snap)
    assert bm2.seq_lens[9] == 4
    assert bm2.get_kv_cache(1) is None

import json
import shutil
import tempfile
from pathlib import Path

import torch

from megagemm.mesh.binary_codec import (
    PinnedTensorPool,
    decode_tensor_frame,
    encode_tensor_frame,
    encode_tensor_frame_parts,
)
from megagemm.mesh.protocol import MeshEndpoint
from megagemm.mesh.shard_pipeline import ShardPipeline, ShardReplicaRouter
from megagemm.mesh.shard_model import MegaMeshShardModel
from megagemm.mesh.shard_worker import (
    MegaMeshLMHeadWorker,
    MegaMeshMLPShardWorker,
    MegaMeshShardWorker,
)
from megagemm.mesh.tensor_codec import tensor_from_payload, tensor_to_payload
from megagemm.mesh.ttp import TTPClient, TTPClientPool, TTPShardServer
from megagemm.models.llama import LlamaConfig, MegaGemmLlama


def _tiny_hf_config() -> dict:
    return {
        "model_type": "llama",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 32,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
    }


def _write_tiny_snapshot(root: Path) -> None:
    from safetensors.torch import save_file

    torch.manual_seed(0)
    hf_config = _tiny_hf_config()
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(hf_config), encoding="utf-8")

    config = LlamaConfig.from_dict(hf_config)
    model = MegaGemmLlama(config).eval()
    model.lm_head.weight = model.embed_tokens.weight
    state = model.state_dict()

    q_size = config.num_attention_heads * config.head_dim
    k_size = config.num_key_value_heads * config.head_dim
    v_size = config.num_key_value_heads * config.head_dim
    hf_state = {
        "model.embed_tokens.weight": state["embed_tokens.weight"].clone(),
        "model.norm.weight": state["norm.weight"].clone(),
    }
    for layer_idx in range(config.num_hidden_layers):
        mg_pre = f"layers.{layer_idx}"
        hf_pre = f"model.layers.{layer_idx}"
        qkv = state[f"{mg_pre}.self_attn.qkv_proj.weight"]
        hf_state[f"{hf_pre}.self_attn.q_proj.weight"] = qkv[:q_size].clone()
        hf_state[f"{hf_pre}.self_attn.k_proj.weight"] = qkv[
            q_size : q_size + k_size
        ].clone()
        hf_state[f"{hf_pre}.self_attn.v_proj.weight"] = qkv[
            q_size + k_size : q_size + k_size + v_size
        ].clone()
        hf_state[f"{hf_pre}.self_attn.o_proj.weight"] = state[
            f"{mg_pre}.self_attn.o_proj.weight"
        ].clone()

        gate_up = state[f"{mg_pre}.mlp.gate_up_proj.weight"]
        hf_state[f"{hf_pre}.mlp.gate_proj.weight"] = gate_up[
            : config.intermediate_size
        ].clone()
        hf_state[f"{hf_pre}.mlp.up_proj.weight"] = gate_up[
            config.intermediate_size :
        ].clone()
        hf_state[f"{hf_pre}.mlp.down_proj.weight"] = state[
            f"{mg_pre}.mlp.down_proj.weight"
        ].clone()
        hf_state[f"{hf_pre}.input_layernorm.weight"] = state[
            f"{mg_pre}.input_layernorm.weight"
        ].clone()
        hf_state[f"{hf_pre}.post_attention_layernorm.weight"] = state[
            f"{mg_pre}.post_attention_layernorm.weight"
        ].clone()

    save_file(hf_state, str(root / "model.safetensors"))


def test_tensor_codec_bfloat16_roundtrip():
    tensor = torch.tensor([[1.0, -2.5]], dtype=torch.bfloat16)
    decoded = tensor_from_payload(tensor_to_payload(tensor))
    assert decoded.dtype is torch.bfloat16
    assert torch.equal(decoded, tensor)


def test_binary_tensor_frame_roundtrip():
    hidden = torch.arange(24, dtype=torch.float16).view(1, 3, 8)
    positions = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    frame = encode_tensor_frame(
        {"ok": True, "seq_id": 9},
        {"hidden": hidden, "positions": positions},
    )
    meta, tensors = decode_tensor_frame(frame)
    assert meta == {"ok": True, "seq_id": 9}
    assert torch.equal(tensors["hidden"], hidden)
    assert torch.equal(tensors["positions"], positions)


def test_binary_tensor_frame_parts_roundtrip_and_pool_reuse():
    pool = PinnedTensorPool()
    tensor = torch.arange(12, dtype=torch.float16).view(1, 3, 4)
    frame = encode_tensor_frame_parts({"ok": True}, {"hidden": tensor}, pool=pool)
    payload = b"".join(bytes(part) for part in frame.iter_parts())
    frame.release()
    meta, tensors = decode_tensor_frame(payload)
    assert meta == {"ok": True}
    assert torch.equal(tensors["hidden"], tensor)


def test_ttp_server_roundtrip():
    class _Stage:
        device = "cpu"

    class _Worker:
        name = "fake"
        stage = _Stage()

        def health(self):
            return {"ok": True, "name": self.name}

        def prefill_ttp(self, meta, tensors):
            assert int(meta["seq_id"]) == 11
            hidden = tensors["positions"].to(torch.float16).unsqueeze(-1)
            return encode_tensor_frame_parts({"ok": True}, {"hidden": hidden})

        def decode_ttp(self, meta, tensors):
            return encode_tensor_frame_parts({"ok": True, "next_token": 3})

        def decode_batch_ttp(self, meta, tensors):
            seq_ids = meta["seq_ids"]
            return encode_tensor_frame_parts(
                {"ok": True, "next_tokens": [int(seq_id) for seq_id in seq_ids]}
            )

        def free(self, payload):
            return {"ok": True, "seq_id": int(payload["seq_id"])}

    server = TTPShardServer(("127.0.0.1", 0), _Worker())
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = None
    try:
        port = int(server.server_address[1])
        client = TTPClient(MeshEndpoint(f"ttp://127.0.0.1:{port}"), timeout=5)
        meta, _ = client.request("health")
        assert meta["ok"] is True
        meta, tensors = client.request(
            "prefill",
            {"seq_id": 11},
            {"positions": torch.tensor([[0, 1]], dtype=torch.long)},
        )
        assert meta["ok"] is True
        assert tensors["hidden"].shape == (1, 2, 1)
        meta, _ = client.request("decode", {"seq_id": 11}, {"positions": torch.tensor([[2]])})
        assert meta["next_token"] == 3
        meta, _ = client.request(
            "decode_batch",
            {"seq_ids": [11, 12]},
            {"positions": torch.tensor([[2], [2]])},
        )
        assert meta["next_tokens"] == [11, 12]
        meta, _ = client.request(
            "ping",
            {},
            {"payload": torch.arange(16, dtype=torch.uint8)},
        )
        assert meta["received_bytes"] == 16
        meta, _ = client.request("free", {"seq_id": 11})
        assert meta["seq_id"] == 11
    finally:
        if client is not None:
            client.close()
        server.shutdown()
        server.server_close()


def test_worker_can_probe_peer_ttp_link():
    class _Stage:
        device = "cpu"

    class _TargetWorker:
        name = "target"
        stage = _Stage()

        def health(self):
            return {"ok": True, "name": self.name}

        def free(self, payload):
            return {"ok": True, "seq_id": int(payload.get("seq_id", 1))}

    server = TTPShardServer(("127.0.0.1", 0), _TargetWorker())
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    worker = object.__new__(MegaMeshShardWorker)
    worker.name = "source"
    worker._ttp_forward_clients = {}
    try:
        result = MegaMeshShardWorker.probe_peer_ttp(
            worker,
            {
                "target": f"ttp://127.0.0.1:{int(server.server_address[1])}#target",
                "payload_bytes": 32,
                "runs": 2,
                "warmup": 0,
            },
            {},
        )
        assert result["ok"] is True
        assert result["src"] == "source"
        assert result["payload_bytes"] == 32
        assert len(result["samples_ms"]) == 2
    finally:
        for client in worker._ttp_forward_clients.values():
            client.close()
        server.shutdown()
        server.server_close()


def test_two_stage_ttp_decode_chunk_pipeline():
    class _Stage:
        device = "cpu"

    class _FirstWorker:
        name = "first"
        stage = _Stage()

        def health(self):
            return {"ok": True, "name": self.name}

        def decode_batch_ttp(self, meta, tensors):
            hidden = tensors["input_ids"].to(torch.float16).unsqueeze(-1)
            return encode_tensor_frame_parts({"ok": True}, {"hidden": hidden})

        def free(self, payload):
            return {"ok": True}

    class _LastWorker:
        name = "last"
        stage = _Stage()

        def health(self):
            return {"ok": True, "name": self.name}

        def decode_batch_ttp(self, meta, tensors):
            hidden = tensors["hidden"].reshape(-1).to(torch.long)
            next_tokens = [int(token) + 10 for token in hidden.tolist()]
            return encode_tensor_frame_parts({"ok": True, "next_tokens": next_tokens})

        def free(self, payload):
            return {"ok": True}

    server0 = TTPShardServer(("127.0.0.1", 0), _FirstWorker())
    server1 = TTPShardServer(("127.0.0.1", 0), _LastWorker())
    import threading

    thread0 = threading.Thread(target=server0.serve_forever, daemon=True)
    thread1 = threading.Thread(target=server1.serve_forever, daemon=True)
    thread0.start()
    thread1.start()
    pool = None
    try:
        stages = [
            MeshEndpoint(f"ttp://127.0.0.1:{int(server0.server_address[1])}#s0"),
            MeshEndpoint(f"ttp://127.0.0.1:{int(server1.server_address[1])}#s1"),
        ]
        pool = TTPClientPool(stages, timeout=5)
        pipeline = object.__new__(ShardPipeline)
        pipeline.stages = stages
        pipeline._ttp_pool = pool
        chunks = [
            {
                "seq_ids": [1, 2],
                "input_ids": torch.tensor([[3], [4]], dtype=torch.long),
                "positions": torch.tensor([[5], [5]], dtype=torch.long),
            },
            {
                "seq_ids": [3, 4],
                "input_ids": torch.tensor([[7], [8]], dtype=torch.long),
                "positions": torch.tensor([[5], [5]], dtype=torch.long),
            },
        ]
        assert pipeline._run_two_stage_ttp_decode_chunks(chunks) == [[13, 14], [17, 18]]
    finally:
        if pool is not None:
            pool.close()
        server0.shutdown()
        server1.shutdown()
        server0.server_close()
        server1.server_close()


def test_ttp_chain_forward_skips_coordinator_hidden_relay():
    class _Stage:
        device = "cpu"

    class _LastWorker:
        name = "last"
        stage = _Stage()

        def decode_batch_ttp(self, meta, tensors):
            hidden = tensors["hidden"].reshape(-1).to(torch.long)
            return encode_tensor_frame_parts(
                {"ok": True, "next_tokens": [int(token) + 100 for token in hidden.tolist()]}
            )

    server = TTPShardServer(("127.0.0.1", 0), _LastWorker())
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    worker = MegaMeshShardWorker.__new__(MegaMeshShardWorker)
    worker._ttp_forward_clients = {}
    worker.ttp_chain_forward_count = 0
    try:
        frame = worker._forward_ttp_result(
            op="decode_batch",
            meta={
                "seq_ids": [1, 2],
                "next_stage": f"ttp://127.0.0.1:{int(server.server_address[1])}",
            },
            positions=torch.tensor([[3], [3]], dtype=torch.long),
            result={"hidden": torch.tensor([[[5]], [[6]]], dtype=torch.float16)},
        )
        payload = b"".join(bytes(part) for part in frame.iter_parts())
        frame.release()
        meta, tensors = decode_tensor_frame(payload)
        assert meta["next_tokens"] == [105, 106]
        assert tensors == {}
        assert worker.ttp_chain_forward_count == 1
    finally:
        for client in worker._ttp_forward_clients.values():
            client.close()
        server.shutdown()
        server.server_close()


def test_lm_head_vocab_shards_reduce_to_full_argmax():
    root = Path(tempfile.mkdtemp(prefix="megamesh_lm_head_test_", dir=Path.cwd()))
    try:
        _write_tiny_snapshot(root)
        head0 = MegaMeshLMHeadWorker(
            str(root),
            vocab_start=0,
            vocab_end=16,
            dtype=torch.float32,
            device="cpu",
            name="head0",
        )
        head1 = MegaMeshLMHeadWorker(
            str(root),
            vocab_start=16,
            vocab_end=32,
            dtype=torch.float32,
            device="cpu",
            name="head1",
        )
        hidden = torch.randn(3, 1, 16, dtype=torch.float32)
        full_weight = torch.cat([head0.weight, head1.weight], dim=0)
        expected = torch.argmax(hidden[:, -1, :] @ full_weight.t(), dim=-1).tolist()

        parts0 = head0.lm_head_argmax_ttp({}, {"hidden": hidden})
        parts1 = head1.lm_head_argmax_ttp({}, {"hidden": hidden})
        payload0 = b"".join(bytes(part) for part in parts0.iter_parts())
        payload1 = b"".join(bytes(part) for part in parts1.iter_parts())
        parts0.release()
        parts1.release()
        meta0, _ = decode_tensor_frame(payload0)
        meta1, _ = decode_tensor_frame(payload1)

        reduced = []
        for idx in range(hidden.shape[0]):
            if float(meta0["logits"][idx]) >= float(meta1["logits"][idx]):
                reduced.append(int(meta0["token_ids"][idx]))
            else:
                reduced.append(int(meta1["token_ids"][idx]))
        assert reduced == [int(token) for token in expected]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_last_layer_shard_can_use_remote_lm_head_shards():
    root = Path(tempfile.mkdtemp(prefix="megamesh_remote_lm_head_test_", dir=Path.cwd()))
    servers = []
    last_worker = None
    try:
        _write_tiny_snapshot(root)
        first = MegaMeshShardModel(
            str(root),
            layer_start=0,
            layer_end=1,
            is_first=True,
            is_last=False,
            dtype=torch.float32,
            device="cpu",
            max_seq_len=32,
            num_blocks=8,
            block_size=3,
        )
        full_last = MegaMeshShardModel(
            str(root),
            layer_start=1,
            layer_end=2,
            is_first=False,
            is_last=True,
            dtype=torch.float32,
            device="cpu",
            max_seq_len=32,
            num_blocks=8,
            block_size=3,
        )
        head0 = MegaMeshLMHeadWorker(
            str(root),
            vocab_start=0,
            vocab_end=16,
            dtype=torch.float32,
            device="cpu",
            name="head0",
        )
        head1 = MegaMeshLMHeadWorker(
            str(root),
            vocab_start=16,
            vocab_end=32,
            dtype=torch.float32,
            device="cpu",
            name="head1",
        )
        import threading

        for head in (head0, head1):
            server = TTPShardServer(("127.0.0.1", 0), head)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
        endpoints = [
            f"ttp://127.0.0.1:{int(server.server_address[1])}#head{idx}"
            for idx, server in enumerate(servers)
        ]
        last_worker = MegaMeshShardWorker(
            str(root),
            layer_start=1,
            layer_end=2,
            is_first=False,
            is_last=True,
            dtype=torch.float32,
            device="cpu",
            max_seq_len=32,
            num_blocks=8,
            block_size=3,
            lm_head_shards=endpoints,
            name="last",
        )
        assert last_worker.health()["lm_head"]["mode"] == "remote-sharded"

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        positions = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        first_prefill = first.prefill(seq_id=7, input_ids=input_ids, positions=positions)
        full_prefill = full_last.prefill(
            seq_id=7,
            hidden=first_prefill["hidden"],
            positions=positions,
        )
        frame = last_worker.prefill_ttp(
            {"seq_id": 7},
            {
                "hidden": first_prefill["hidden"],
                "positions": positions,
            },
        )
        payload = b"".join(bytes(part) for part in frame.iter_parts())
        frame.release()
        remote_prefill, tensors = decode_tensor_frame(payload)
        assert tensors == {}
        assert remote_prefill["next_token"] == full_prefill["next_token"]
    finally:
        if last_worker is not None:
            for client in last_worker._lm_head_clients:
                client.close()
        for server in servers:
            server.shutdown()
            server.server_close()
        shutil.rmtree(root, ignore_errors=True)


def test_mlp_intermediate_shards_sum_to_full_mlp():
    root = Path(tempfile.mkdtemp(prefix="megamesh_mlp_test_", dir=Path.cwd()))
    try:
        _write_tiny_snapshot(root)
        local = MegaMeshShardModel(
            str(root),
            layer_start=0,
            layer_end=1,
            is_first=False,
            is_last=False,
            dtype=torch.float32,
            device="cpu",
            max_seq_len=32,
            num_blocks=8,
            block_size=3,
        )
        mlp0 = MegaMeshMLPShardWorker(
            str(root),
            layer_start=0,
            layer_end=1,
            intermediate_start=0,
            intermediate_end=16,
            dtype=torch.float32,
            device="cpu",
            name="mlp0",
        )
        mlp1 = MegaMeshMLPShardWorker(
            str(root),
            layer_start=0,
            layer_end=1,
            intermediate_start=16,
            intermediate_end=32,
            dtype=torch.float32,
            device="cpu",
            name="mlp1",
        )
        hidden = torch.randn(2, 3, 16, dtype=torch.float32)
        expected = local.model.layers[0].mlp(hidden)

        parts0 = mlp0.mlp_forward_ttp({"layer_idx": 0}, {"hidden": hidden})
        parts1 = mlp1.mlp_forward_ttp({"layer_idx": 0}, {"hidden": hidden})
        payload0 = b"".join(bytes(part) for part in parts0.iter_parts())
        payload1 = b"".join(bytes(part) for part in parts1.iter_parts())
        parts0.release()
        parts1.release()
        _, tensors0 = decode_tensor_frame(payload0)
        _, tensors1 = decode_tensor_frame(payload1)
        actual = tensors0["partial"] + tensors1["partial"]
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_layer_stage_can_use_remote_mlp_shards():
    root = Path(tempfile.mkdtemp(prefix="megamesh_remote_mlp_test_", dir=Path.cwd()))
    servers = []
    remote_worker = None
    try:
        _write_tiny_snapshot(root)
        local = MegaMeshShardModel(
            str(root),
            layer_start=1,
            layer_end=2,
            is_first=False,
            is_last=False,
            dtype=torch.float32,
            device="cpu",
            max_seq_len=32,
            num_blocks=8,
            block_size=3,
        )
        mlp0 = MegaMeshMLPShardWorker(
            str(root),
            layer_start=1,
            layer_end=2,
            intermediate_start=0,
            intermediate_end=16,
            dtype=torch.float32,
            device="cpu",
            name="mlp0",
        )
        mlp1 = MegaMeshMLPShardWorker(
            str(root),
            layer_start=1,
            layer_end=2,
            intermediate_start=16,
            intermediate_end=32,
            dtype=torch.float32,
            device="cpu",
            name="mlp1",
        )
        import threading

        for worker in (mlp0, mlp1):
            server = TTPShardServer(("127.0.0.1", 0), worker)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
        endpoints = [
            f"ttp://127.0.0.1:{int(server.server_address[1])}#mlp{idx}"
            for idx, server in enumerate(servers)
        ]
        remote_worker = MegaMeshShardWorker(
            str(root),
            layer_start=1,
            layer_end=2,
            is_first=False,
            is_last=False,
            dtype=torch.float32,
            device="cpu",
            max_seq_len=32,
            num_blocks=8,
            block_size=3,
            mlp_shards=endpoints,
            name="remote-stage",
        )
        assert remote_worker.health()["mlp"]["mode"] == "remote-sharded"

        hidden = torch.randn(1, 3, 16, dtype=torch.float32)
        positions = torch.arange(hidden.shape[1], dtype=torch.long).unsqueeze(0)
        expected = local.prefill(seq_id=3, hidden=hidden.clone(), positions=positions)["hidden"]
        frame = remote_worker.prefill_ttp(
            {"seq_id": 3},
            {"hidden": hidden, "positions": positions},
        )
        payload = b"".join(bytes(part) for part in frame.iter_parts())
        frame.release()
        meta, tensors = decode_tensor_frame(payload)
        assert meta["ok"] is True
        assert torch.allclose(tensors["hidden"], expected, atol=1e-5, rtol=1e-5)
    finally:
        if remote_worker is not None:
            for client in remote_worker._mlp_clients:
                client.close()
            if remote_worker._mlp_executor is not None:
                remote_worker._mlp_executor.shutdown(wait=True)
        for server in servers:
            server.shutdown()
            server.server_close()
        shutil.rmtree(root, ignore_errors=True)


def test_remote_chain_batch_loop_runs_inside_first_stage():
    class _Config:
        vocab_size = 1000

    class _Stage:
        device = "cpu"
        config = _Config()

        def prefill(self, seq_id, input_ids, positions):
            del seq_id, positions
            token = int(input_ids[0, -1].item()) + 1
            return {"hidden": torch.tensor([[[token]]], dtype=torch.float16)}

        def decode_batch(self, seq_ids, input_ids, positions):
            del seq_ids, positions
            return {"hidden": input_ids.to(torch.float16).unsqueeze(-1)}

    class _LastWorker:
        name = "last"

        def prefill_ttp(self, meta, tensors):
            del meta
            token = int(tensors["hidden"].reshape(-1)[0].item()) + 10
            return encode_tensor_frame_parts({"ok": True, "next_token": token})

        def decode_batch_ttp(self, meta, tensors):
            del meta
            hidden = tensors["hidden"].reshape(-1).to(torch.long)
            return encode_tensor_frame_parts(
                {"ok": True, "next_tokens": [int(token) + 10 for token in hidden.tolist()]}
            )

    server = TTPShardServer(("127.0.0.1", 0), _LastWorker())
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    worker = MegaMeshShardWorker.__new__(MegaMeshShardWorker)
    worker.name = "first"
    worker.lock = threading.Lock()
    worker.prefill_count = 0
    worker.decode_count = 0
    worker.ttp_chain_forward_count = 0
    worker._ttp_forward_clients = {}
    worker.stage = _Stage()
    try:
        frame = worker.generate_batch_chain_ttp(
            {
                "seq_ids": [1, 2],
                "next_stage": f"ttp://127.0.0.1:{int(server.server_address[1])}",
                "max_new_tokens": 3,
                "microbatch_size": 2,
                "eos_ids": [],
            },
            {
                "input_ids": torch.tensor([[1, 2], [3, 0]], dtype=torch.long),
                "prompt_lengths": torch.tensor([2, 1], dtype=torch.long),
            },
        )
        payload = b"".join(bytes(part) for part in frame.iter_parts())
        frame.release()
        meta, tensors = decode_tensor_frame(payload)
        assert tensors == {}
        assert meta["generated"] == [[13, 23, 33], [14, 24, 34]]
        assert meta["decode_steps"] == 3
        assert meta["total_decode_chunks"] == 2
        assert meta["chain_forwards"] == 4
        assert worker.prefill_count == 2
        assert worker.decode_count == 4
    finally:
        for client in worker._ttp_forward_clients.values():
            client.close()
        server.shutdown()
        server.server_close()


def test_remote_chain_batch_loop_routes_through_middle_stage():
    class _Config:
        vocab_size = 1000

    class _FirstStage:
        device = "cpu"
        config = _Config()

        def prefill(self, seq_id, input_ids, positions):
            del seq_id, positions
            return {"hidden": input_ids[:, -1:, None].to(torch.float16)}

        def decode_batch(self, seq_ids, input_ids, positions):
            del seq_ids, positions
            return {"hidden": input_ids.to(torch.float16).unsqueeze(-1)}

    class _Stage:
        device = "cpu"

    class _MiddleWorker:
        name = "middle"
        stage = _Stage()

        def __init__(self):
            self._ttp_forward_clients = {}
            self.ttp_chain_forward_count = 0

        _forward_client = MegaMeshShardWorker._forward_client
        _forward_hidden_ttp = MegaMeshShardWorker._forward_hidden_ttp
        _forward_ttp_result = MegaMeshShardWorker._forward_ttp_result
        _format_ttp_result = MegaMeshShardWorker._format_ttp_result

        def _route_from_meta(self, meta):
            del self
            return MegaMeshShardWorker._route_from_meta(meta)

        def prefill_chain_ttp(self, meta, tensors):
            hidden = tensors["hidden"] + 10
            return self._forward_ttp_result(
                op="prefill",
                meta=meta,
                positions=tensors["positions"],
                result={"hidden": hidden},
            )

        def decode_batch_chain_ttp(self, meta, tensors):
            hidden = tensors["hidden"] + 10
            return self._forward_ttp_result(
                op="decode_batch",
                meta=meta,
                positions=tensors["positions"],
                result={"hidden": hidden},
            )

    class _LastWorker:
        name = "last"

        def prefill_ttp(self, meta, tensors):
            del meta
            token = int(tensors["hidden"].reshape(-1)[0].item()) + 100
            return encode_tensor_frame_parts({"ok": True, "next_token": token})

        def decode_batch_ttp(self, meta, tensors):
            del meta
            hidden = tensors["hidden"].reshape(-1).to(torch.long)
            return encode_tensor_frame_parts(
                {"ok": True, "next_tokens": [int(token) + 100 for token in hidden.tolist()]}
            )

    middle = _MiddleWorker()
    server_mid = TTPShardServer(("127.0.0.1", 0), middle)
    server_last = TTPShardServer(("127.0.0.1", 0), _LastWorker())
    import threading

    threading.Thread(target=server_mid.serve_forever, daemon=True).start()
    threading.Thread(target=server_last.serve_forever, daemon=True).start()

    worker = MegaMeshShardWorker.__new__(MegaMeshShardWorker)
    worker.name = "first"
    worker.lock = threading.Lock()
    worker.prefill_count = 0
    worker.decode_count = 0
    worker.ttp_chain_forward_count = 0
    worker._ttp_forward_clients = {}
    worker.stage = _FirstStage()
    try:
        frame = worker.generate_batch_chain_ttp(
            {
                "seq_ids": [1, 2],
                "next_stages": [
                    f"ttp://127.0.0.1:{int(server_mid.server_address[1])}",
                    f"ttp://127.0.0.1:{int(server_last.server_address[1])}",
                ],
                "max_new_tokens": 2,
                "microbatch_size": 2,
                "eos_ids": [],
            },
            {
                "input_ids": torch.tensor([[1], [2]], dtype=torch.long),
                "prompt_lengths": torch.tensor([1, 1], dtype=torch.long),
            },
        )
        payload = b"".join(bytes(part) for part in frame.iter_parts())
        frame.release()
        meta, tensors = decode_tensor_frame(payload)
        assert tensors == {}
        assert meta["generated"] == [[111, 221], [112, 222]]
        assert meta["chain_forwards"] == 6
        assert worker.ttp_chain_forward_count == 3
        assert middle.ttp_chain_forward_count == 3
    finally:
        for client in worker._ttp_forward_clients.values():
            client.close()
        for client in middle._ttp_forward_clients.values():
            client.close()
        server_mid.shutdown()
        server_last.shutdown()
        server_mid.server_close()
        server_last.server_close()


def test_pipeline_remote_chain_batch_request():
    class _Tokenizer:
        eos_token_id = None
        pad_token_id = None

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return ",".join(str(int(token)) for token in ids)

    class _Stage:
        device = "cpu"

    class _FirstWorker:
        name = "first"
        stage = _Stage()

        def generate_batch_chain_ttp(self, meta, tensors):
            assert meta["seq_ids"] == [1, 2]
            assert meta["next_stages"] == ["ttp://127.0.0.1:1#s1"]
            assert meta["max_new_tokens"] == 2
            assert meta["microbatch_size"] == 8
            assert tensors["input_ids"].tolist() == [[10, 11], [12, 0]]
            assert tensors["prompt_lengths"].tolist() == [2, 1]
            return encode_tensor_frame_parts(
                {
                    "ok": True,
                    "generated": [[3, 4], [5]],
                    "decode_steps": 2,
                    "total_decode_chunks": 1,
                    "max_decode_chunks_per_step": 1,
                    "chain_forwards": 4,
                }
            )

        def health(self):
            return {"ok": True, "name": self.name}

        def free(self, payload):
            return {"ok": True, "seq_id": int(payload["seq_id"])}

    server = TTPShardServer(("127.0.0.1", 0), _FirstWorker())
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    pool = None
    try:
        stages = [
            MeshEndpoint(f"ttp://127.0.0.1:{int(server.server_address[1])}#s0"),
            MeshEndpoint("ttp://127.0.0.1:1#s1"),
        ]
        pool = TTPClientPool(stages[:1], timeout=5)
        pipeline = object.__new__(ShardPipeline)
        pipeline.stages = stages
        pipeline._ttp_pool = pool
        pipeline.transport = "ttp"
        pipeline.remote_chain_loop = True
        pipeline._ttp_chain_requests = 0
        pipeline.tokenizer = _Tokenizer()
        pipeline.timeout = 5
        pipeline._stage_snapshot = lambda: []
        result = pipeline._generate_batch_ttp_remote_chain(
            rows=[
                {"seq_id": 1, "token_ids": [10, 11]},
                {"seq_id": 2, "token_ids": [12]},
            ],
            max_new_tokens=2,
            microbatch_size=8,
            include_prompt=False,
            chain_start=0,
            t0=0.0,
        )
        assert result["pipeline"]["remote_chain_loop"] is True
        assert result["pipeline"]["coordinator_ttp_requests"] == 1
        assert result["pipeline"]["worker_chain_forwards"] == 4
        assert result["outputs"][0]["text"] == "3,4"
        assert result["outputs"][1]["text"] == "5"
    finally:
        if pool is not None:
            pool.close()
        server.shutdown()
        server.server_close()


def test_pipeline_remote_chain_continuous_request():
    class _Tokenizer:
        eos_token_id = None
        pad_token_id = None

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return ",".join(str(int(token)) for token in ids)

    class _Stage:
        device = "cpu"

    class _FirstWorker:
        name = "first"
        stage = _Stage()

        def generate_continuous_chain_ttp(self, meta, tensors):
            assert meta["seq_ids"] == [1, 2, 3]
            assert meta["next_stages"] == ["ttp://127.0.0.1:1#s1"]
            assert meta["max_new_tokens"] == 2
            assert meta["microbatch_size"] == 2
            assert meta["max_batch_size"] == 2
            assert tensors["input_ids"].tolist() == [[10, 11], [12, 0], [13, 0]]
            assert tensors["prompt_lengths"].tolist() == [2, 1, 1]
            return encode_tensor_frame_parts(
                {
                    "ok": True,
                    "generated": [[3, 4], [5], [6, 7]],
                    "continuous_batching": True,
                    "scheduler_steps": 3,
                    "admission_events": 2,
                    "total_prefills": 3,
                    "max_running": 2,
                    "decode_steps": 2,
                    "total_decode_chunks": 2,
                    "max_decode_chunks_per_step": 1,
                    "chain_forwards": 9,
                }
            )

        def health(self):
            return {"ok": True, "name": self.name}

        def free(self, payload):
            return {"ok": True, "seq_id": int(payload["seq_id"])}

    server = TTPShardServer(("127.0.0.1", 0), _FirstWorker())
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    pool = None
    try:
        stages = [
            MeshEndpoint(f"ttp://127.0.0.1:{int(server.server_address[1])}#s0"),
            MeshEndpoint("ttp://127.0.0.1:1#s1"),
        ]
        pool = TTPClientPool(stages[:1], timeout=5)
        pipeline = object.__new__(ShardPipeline)
        pipeline.stages = stages
        pipeline._ttp_pool = pool
        pipeline.transport = "ttp"
        pipeline.remote_chain_loop = True
        pipeline._ttp_chain_requests = 0
        pipeline.tokenizer = _Tokenizer()
        pipeline.timeout = 5
        pipeline._stage_snapshot = lambda: []
        result = pipeline._generate_continuous_ttp_remote_chain(
            rows=[
                {"seq_id": 1, "token_ids": [10, 11]},
                {"seq_id": 2, "token_ids": [12]},
                {"seq_id": 3, "token_ids": [13]},
            ],
            max_new_tokens=2,
            microbatch_size=2,
            max_batch_size=2,
            include_prompt=False,
            chain_start=0,
            t0=0.0,
        )
        assert result["continuous_batching"] is True
        assert result["pipeline"]["continuous_batching"] is True
        assert result["pipeline"]["admission_events"] == 2
        assert result["pipeline"]["max_running"] == 2
        assert result["outputs"][0]["text"] == "3,4"
        assert result["outputs"][2]["text"] == "6,7"
    finally:
        if pool is not None:
            pool.close()
        server.shutdown()
        server.server_close()


def test_shard_replica_router_preserves_prompt_order():
    class _FakePipeline:
        def __init__(self, name: str):
            self.name = name
            self.closed = False

        def health(self):
            return [{"ok": True, "name": self.name}]

        def close(self):
            self.closed = True

        def generate_batch(self, prompts, *, max_new_tokens, microbatch_size, include_prompt):
            return {
                "ok": True,
                "num_prompts": len(prompts),
                "generated_tokens": len(prompts),
                "elapsed_ms": 1.0,
                "tokens_per_second": 1000.0,
                "pipeline": {"remote_chain_loop": True},
                "stages": [{"name": self.name}],
                "outputs": [
                    {
                        "seq_id": i + 1,
                        "text": f"{self.name}:{prompt}",
                        "prompt_tokens": 1,
                        "generated_tokens": 1,
                    }
                    for i, prompt in enumerate(prompts)
                ],
            }

    router = object.__new__(ShardReplicaRouter)
    router.pipelines = [_FakePipeline("r0"), _FakePipeline("r1")]
    router.transport = "ttp"

    result = router.generate_batch(
        ["a", "b", "c", "d"],
        max_new_tokens=2,
        microbatch_size=2,
    )

    assert result["replicated_shards"] is True
    assert result["num_replicas"] == 2
    assert result["num_prompts"] == 4
    assert [row["text"] for row in result["outputs"]] == [
        "r0:a",
        "r1:b",
        "r0:c",
        "r1:d",
    ]
    assert [row["replica_index"] for row in result["outputs"]] == [0, 1, 0, 1]
    assert [row["num_prompts"] for row in result["replicas"]] == [2, 2]


def test_two_stage_layer_shard_prefill_and_decode():
    root = Path(tempfile.mkdtemp(prefix="megamesh_shard_test_", dir=Path.cwd()))
    try:
        _write_tiny_snapshot(root)
        first = MegaMeshShardModel(
            str(root),
            layer_start=0,
            layer_end=1,
            is_first=True,
            is_last=False,
            dtype=torch.float32,
            device="cpu",
            max_seq_len=32,
            num_blocks=8,
            block_size=3,
        )
        last = MegaMeshShardModel(
            str(root),
            layer_start=1,
            layer_end=2,
            is_first=False,
            is_last=True,
            dtype=torch.float32,
            device="cpu",
            max_seq_len=32,
            num_blocks=8,
            block_size=3,
        )

        assert first.fastpath_info["decode"] in {
            "cpp-full-attention-loop",
            "python-full-attention-infer",
        }
        assert last.fastpath_info["local_all_full_attention"] is True

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        positions = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        first_prefill = first.prefill(
            seq_id=7,
            input_ids=input_ids,
            positions=positions,
        )
        last_prefill = last.prefill(
            seq_id=7,
            hidden=first_prefill["hidden"],
            positions=positions,
        )
        first_prefill_2 = first.prefill(
            seq_id=8,
            input_ids=input_ids,
            positions=positions,
        )
        last_prefill_2 = last.prefill(
            seq_id=8,
            hidden=first_prefill_2["hidden"],
            positions=positions,
        )

        assert first_prefill["hidden"].shape == (1, 3, 16)
        assert isinstance(last_prefill["next_token"], int)
        assert isinstance(last_prefill_2["next_token"], int)

        batch_ids = torch.tensor(
            [[last_prefill["next_token"]], [last_prefill_2["next_token"]]],
            dtype=torch.long,
        )
        batch_pos = torch.tensor([[input_ids.shape[1]], [input_ids.shape[1]]], dtype=torch.long)
        first_batch = first.decode_batch(
            seq_ids=[7, 8],
            input_ids=batch_ids,
            positions=batch_pos,
        )
        last_batch = last.decode_batch(
            seq_ids=[7, 8],
            hidden=first_batch["hidden"],
            positions=batch_pos,
        )
        assert first_batch["hidden"].shape == (2, 1, 16)
        assert len(last_batch["next_tokens"]) == 2

        next_ids = torch.tensor([[last_batch["next_tokens"][0]]], dtype=torch.long)
        next_pos = torch.tensor([[input_ids.shape[1] + 1]], dtype=torch.long)
        first_decode = first.decode(seq_id=7, input_ids=next_ids, positions=next_pos)
        last_decode = last.decode(
            seq_id=7,
            hidden=first_decode["hidden"],
            positions=next_pos,
        )

        assert first_decode["hidden"].shape == (1, 1, 16)
        assert isinstance(last_decode["next_token"], int)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_tensor_codec_bfloat16_roundtrip()
    test_binary_tensor_frame_roundtrip()
    test_binary_tensor_frame_parts_roundtrip_and_pool_reuse()
    test_ttp_server_roundtrip()
    test_worker_can_probe_peer_ttp_link()
    test_two_stage_ttp_decode_chunk_pipeline()
    test_ttp_chain_forward_skips_coordinator_hidden_relay()
    test_lm_head_vocab_shards_reduce_to_full_argmax()
    test_last_layer_shard_can_use_remote_lm_head_shards()
    test_mlp_intermediate_shards_sum_to_full_mlp()
    test_layer_stage_can_use_remote_mlp_shards()
    test_remote_chain_batch_loop_runs_inside_first_stage()
    test_remote_chain_batch_loop_routes_through_middle_stage()
    test_pipeline_remote_chain_batch_request()
    test_pipeline_remote_chain_continuous_request()
    test_shard_replica_router_preserves_prompt_order()
    test_two_stage_layer_shard_prefill_and_decode()
    print("MegaMesh shard tests passed")

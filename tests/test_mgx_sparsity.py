import copy
import gc
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from megagemm.kernels.sparse24_gemv import (
    sparse24_cutlass_gemv,
    sparse24_cutlass_gemv_eligible,
    sparse24_triton_available,
)
from megagemm.kernels.sparse24_mma import (
    sparse24_mma_available,
    sparse24_mma_linear,
    sparse24_portable_metadata_to_ptx,
)
from megagemm.models import export_to_mgx, load_from_mgx
from megagemm.models.llama import LlamaConfig, MegaGemmLlama
from megagemm.models.sparsity import (
    expand_sparse24_payload,
    is_valid_sparse24_dense,
    normalize_sparsity_mode,
    pack_model_state_sparse24,
    pack_sparse24_weight,
    prepare_sparse24_runtime,
    unpack_sparse24_weight,
    validate_sparse24_config,
)


class _Sparse24Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(64, 64, bias=False, dtype=torch.float16)
        self.unrelated = nn.Linear(64, 64, bias=False, dtype=torch.float16)


class _Sparse24ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Sparse24Block()])


class Sparse24PackingTests(unittest.TestCase):
    def test_pack_round_trip_keeps_two_largest_magnitudes_per_group(self) -> None:
        group = torch.tensor([-1.0, 4.0, -3.0, 2.0], dtype=torch.float16)
        weight = group.repeat(64, 16)

        values, metadata = pack_sparse24_weight(weight)
        restored = unpack_sparse24_weight(values, metadata, [64, 64])

        self.assertEqual(tuple(values.shape), (64, 32))
        self.assertEqual(tuple(metadata.shape), (64, 8))
        self.assertTrue(is_valid_sparse24_dense(restored))
        expected_group = torch.tensor([0.0, 4.0, -3.0, 0.0], dtype=torch.float16)
        torch.testing.assert_close(restored[0, :4], expected_group, rtol=0, atol=0)

    def test_invalid_position_metadata_is_rejected(self) -> None:
        values = torch.ones((64, 32), dtype=torch.float16)
        metadata = torch.full((64, 8), 0xFF, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "invalid position code"):
            unpack_sparse24_weight(values, metadata, [64, 64])

    def test_portable_metadata_maps_to_ordered_ptx_nibbles(self) -> None:
        # MGX codes 0..5 represent (01, 02, 03, 12, 13, 23).
        metadata = torch.tensor([[0x10, 0x32, 0x54]], dtype=torch.uint8)
        converted = sparse24_portable_metadata_to_ptx(metadata)
        self.assertEqual(
            converted.tolist(),
            [[0x84, 0x9C, 0xED]],
        )

    def test_native_mma_matches_pruned_fp16_linear(self) -> None:
        if not torch.cuda.is_available() or not sparse24_mma_available():
            self.skipTest("CUDA + the standalone sparse24 extension are required")
        capability = torch.cuda.get_device_capability()
        if int(capability[0]) != 8:
            self.skipTest("standalone mma.sp currently targets SM80-SM89")

        torch.manual_seed(29)
        dense = torch.randn((64, 128), device="cuda", dtype=torch.float16)
        values, metadata = pack_sparse24_weight(dense)
        pruned = unpack_sparse24_weight(values, metadata, [64, 128])
        ptx_metadata = sparse24_portable_metadata_to_ptx(metadata)
        bias = torch.randn((64,), device="cuda", dtype=torch.float16)
        for rows in (1, 8, 16, 32, 64):
            x = torch.randn((rows, 128), device="cuda", dtype=torch.float16)
            expected = torch.nn.functional.linear(x, pruned, bias)
            actual = sparse24_mma_linear(x, values, ptx_metadata, bias)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_model_state_pack_and_expand(self) -> None:
        model = _Sparse24ToyModel()
        state = copy.copy(model.state_dict())
        original_unrelated = state["layers.0.unrelated.weight"]

        config = pack_model_state_sparse24(model, state)

        self.assertEqual(config["format"], "2:4")
        self.assertEqual(config["tensor_count"], 1)
        self.assertAlmostEqual(config["storage_ratio"], 9 / 16)
        self.assertNotIn("layers.0.q_proj.weight", state)
        self.assertIn("layers.0.unrelated.weight", state)
        self.assertIs(state["layers.0.unrelated.weight"], original_unrelated)
        validate_sparse24_config(config, tensor_names=set(state))

        expanded = expand_sparse24_payload(state, config)
        self.assertEqual(expanded, ["layers.0.q_proj.weight"])
        self.assertIn("layers.0.q_proj.weight", state)
        self.assertTrue(is_valid_sparse24_dense(state["layers.0.q_proj.weight"]))

    def test_cpu_runtime_is_explicit_dense_fallback(self) -> None:
        model = _Sparse24ToyModel()
        state = copy.copy(model.state_dict())
        config = pack_model_state_sparse24(model, state)
        expand_sparse24_payload(state, config)
        model.layers[0].q_proj.weight = nn.Parameter(
            state["layers.0.q_proj.weight"], requires_grad=False
        )

        stats = prepare_sparse24_runtime(model, config, device="cpu")

        self.assertFalse(stats["active"])
        self.assertEqual(stats["backend"], "dense-fallback")
        self.assertEqual(stats["reason"], "CUDA is unavailable")
        self.assertEqual(model.layers[0].q_proj.weight.layout, torch.strided)
        self.assertIn("specialized_kernel_available", stats)

    def test_specialized_kernel_policy_rejects_cpu_tensors(self) -> None:
        x = torch.randn((1, 64), dtype=torch.float16)
        values = torch.randn((64, 32), dtype=torch.float16)
        metadata = torch.zeros((64, 4), dtype=torch.int16)
        self.assertFalse(sparse24_cutlass_gemv_eligible(x, values, metadata))

    def test_cuda_specialized_kernel_matches_torch_sparse_linear(self) -> None:
        if not torch.cuda.is_available() or not sparse24_triton_available():
            self.skipTest("CUDA + Triton are required")
        cutlass_cls = getattr(torch.sparse, "SparseSemiStructuredTensorCUTLASS", None)
        if cutlass_cls is None:
            self.skipTest("PyTorch CUTLASS semi-structured backend is unavailable")

        torch.manual_seed(17)
        dense = torch.randn((64, 128), device="cuda", dtype=torch.float16)
        values, metadata = pack_sparse24_weight(dense)
        pruned = unpack_sparse24_weight(values, metadata, [64, 128])
        try:
            sparse = cutlass_cls.from_dense(pruned)
        except Exception as exc:
            self.skipTest(f"CUTLASS semi-structured backend is unsupported: {exc}")

        for rows in (1, 16):
            x = torch.randn((rows, 128), device="cuda", dtype=torch.float16)
            with torch.inference_mode():
                expected = torch.nn.functional.linear(x, sparse)
                actual = sparse24_cutlass_gemv(x, sparse.values(), sparse.meta)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_public_mode_aliases(self) -> None:
        self.assertIsNone(normalize_sparsity_mode("none"))
        self.assertEqual(normalize_sparsity_mode("2;4"), "2:4")
        with self.assertRaisesRegex(ValueError, "Unsupported MGX sparsity"):
            normalize_sparsity_mode("unstructured")

    def test_sparse24_rejects_quantized_export(self) -> None:
        hf_config = {
            "model_type": "llama",
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "vocab_size": 64,
            "max_position_embeddings": 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(hf_config), encoding="utf-8")
            with self.assertRaisesRegex(NotImplementedError, "cannot be combined"):
                export_to_mgx(
                    str(root),
                    root / "invalid.mgx",
                    dtype="fp16",
                    quantize="int8",
                    sparsity="2:4",
                )

    def test_mgx_export_load_round_trip_uses_packed_payload_and_dense_cpu_fallback(self) -> None:
        hf_config = {
            "model_type": "llama",
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "vocab_size": 64,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }
        torch.manual_seed(7)
        source_model = MegaGemmLlama(LlamaConfig.from_dict(hf_config)).half().eval()
        runtime_state = source_model.state_dict()
        qkv = runtime_state["layers.0.self_attn.qkv_proj.weight"]
        gate_up = runtime_state["layers.0.mlp.gate_up_proj.weight"]
        hf_state = {
            "model.embed_tokens.weight": runtime_state["embed_tokens.weight"].clone(),
            "model.norm.weight": runtime_state["norm.weight"].clone(),
            "lm_head.weight": runtime_state["lm_head.weight"].clone(),
            "model.layers.0.self_attn.q_proj.weight": qkv[:64].clone(),
            "model.layers.0.self_attn.k_proj.weight": qkv[64:128].clone(),
            "model.layers.0.self_attn.v_proj.weight": qkv[128:].clone(),
            "model.layers.0.self_attn.o_proj.weight": runtime_state[
                "layers.0.self_attn.o_proj.weight"
            ].clone(),
            "model.layers.0.mlp.gate_proj.weight": gate_up[:64].clone(),
            "model.layers.0.mlp.up_proj.weight": gate_up[64:].clone(),
            "model.layers.0.mlp.down_proj.weight": runtime_state[
                "layers.0.mlp.down_proj.weight"
            ].clone(),
            "model.layers.0.input_layernorm.weight": runtime_state[
                "layers.0.input_layernorm.weight"
            ].clone(),
            "model.layers.0.post_attention_layernorm.weight": runtime_state[
                "layers.0.post_attention_layernorm.weight"
            ].clone(),
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text(json.dumps(hf_config), encoding="utf-8")
            from safetensors.torch import save_file

            save_file(hf_state, str(snapshot / "model.safetensors"))
            artifact = root / "sparse24.mgx"
            info = export_to_mgx(
                str(snapshot),
                artifact,
                dtype="fp16",
                quantize="none",
                sparsity="2:4",
                export_mode="streaming",
                emit_payload_cache=True,
            )

            manifest = info["manifest"]
            self.assertEqual(manifest["sparsity"], "2:4")
            self.assertEqual(manifest["version"]["minor"], 1)
            self.assertEqual(manifest["sparsity_config"]["tensor_count"], 4)
            physical_names = {entry["name"] for entry in manifest["tensor_table"]}
            self.assertNotIn("layers.0.self_attn.qkv_proj.weight", physical_names)
            self.assertIn(
                "__mgx_sparse24__.layers.0.self_attn.qkv_proj.weight.values",
                physical_names,
            )

            loaded = load_from_mgx(
                artifact,
                device="cpu",
                dtype_override="fp16",
                prefer_payload_cache=False,
            )
            self.assertTrue(
                is_valid_sparse24_dense(loaded.layers[0].self_attn.qkv_proj.weight)
            )
            runtime = loaded.decode_runtime_stats()["mgx_sparsity_runtime"]
            self.assertFalse(runtime["active"])
            self.assertEqual(runtime["backend"], "dense-fallback")

            cached = load_from_mgx(
                artifact,
                device="cpu",
                dtype_override="fp16",
                prefer_payload_cache=True,
            )
            self.assertEqual(cached._load_timing["payload_source"], "payload_cache_streaming")
            self.assertEqual(cached._load_timing["sparse24_expanded_tensor_count"], 4)
            self.assertTrue(
                is_valid_sparse24_dense(cached.layers[0].mlp.gate_up_proj.weight)
            )
            with patch.dict(os.environ, {"MEGAGEMM_MGX_PREFER_RUNTIME_CACHE": "1"}):
                runtime_cached = load_from_mgx(
                    artifact,
                    device="cpu",
                    dtype_override="fp16",
                    prefer_payload_cache=True,
                )
            self.assertEqual(
                runtime_cached._load_timing["payload_source"],
                "payload_runtime_cache_packed",
            )
            self.assertTrue(
                is_valid_sparse24_dense(
                    runtime_cached.layers[0].self_attn.o_proj.weight
                )
            )
            # Bulk safetensors loading can keep a Windows file mapping alive for
            # as long as model parameters reference its storage.
            del cached, loaded, runtime_cached
            gc.collect()


if __name__ == "__main__":
    unittest.main()

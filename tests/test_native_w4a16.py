import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from megagemm.models import export_to_mgx, load_from_mgx
from megagemm.models.llama import LlamaConfig, MegaGemmLlama
from megagemm.models.llama import _FlatLayerWeights
from megagemm.quantization.native_w4a16 import (
    NativeW4A16Linear,
    dequantize_native_w4a16,
    quantize_model_native_w4a16,
    quantize_native_w4a16,
)


class NativeW4A16PackingTests(unittest.TestCase):
    def test_common_flat_weights_initialize_v_alias_flag(self):
        self.assertFalse(_FlatLayerWeights().v_from_k)

    def test_dense_pack_roundtrip_and_shapes(self):
        torch.manual_seed(1)
        weight = torch.randn(7, 128)
        qweight, scales, metadata = quantize_native_w4a16(weight, group_size=128)
        restored = dequantize_native_w4a16(
            qweight,
            scales,
            metadata,
            in_features=128,
            group_size=128,
            sparse24=False,
            dtype=torch.float32,
        )
        self.assertEqual(tuple(qweight.shape), (7, 64))
        self.assertEqual(tuple(scales.shape), (7, 1))
        self.assertEqual(tuple(metadata.shape), (7, 0))
        self.assertLess(float((weight - restored).abs().mean()), 0.2)

    def test_sparse_pack_has_exactly_two_values_per_quartet(self):
        weight = torch.arange(-64, 64, dtype=torch.float32).reshape(1, 128)
        qweight, scales, metadata = quantize_native_w4a16(
            weight, group_size=128, sparse24=True
        )
        restored = dequantize_native_w4a16(
            qweight,
            scales,
            metadata,
            in_features=128,
            group_size=128,
            sparse24=True,
            dtype=torch.float32,
        )
        nonzero_per_quartet = (restored.reshape(1, -1, 4) != 0).sum(dim=-1)
        self.assertTrue(bool((nonzero_per_quartet <= 2).all()))
        self.assertEqual(tuple(qweight.shape), (1, 32))
        self.assertEqual(tuple(metadata.shape), (1, 16))

    def test_cpu_forward_matches_explicit_dequantization(self):
        torch.manual_seed(2)
        linear = nn.Linear(128, 11, bias=True, dtype=torch.float32)
        quantized = NativeW4A16Linear.from_linear(
            linear, group_size=128, sparse24=True
        )
        x = torch.randn(5, 128)
        expected = torch.nn.functional.linear(x, quantized.dequantize(torch.float32), quantized.bias)
        actual = quantized(x)
        torch.testing.assert_close(actual, expected)

    def test_model_replacement_keeps_lm_head_dense(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(128, 64, bias=False)
                self.lm_head = nn.Linear(128, 32, bias=False)

        model = Toy()
        info = quantize_model_native_w4a16(model, group_size=128, sparse24=True)
        self.assertIsInstance(model.proj, NativeW4A16Linear)
        self.assertIsInstance(model.lm_head, nn.Linear)
        self.assertEqual(info["module_count"], 1)


class NativeW4A16MGXTests(unittest.TestCase):
    @staticmethod
    def _config_dict():
        return {
            "model_type": "llama",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }

    @classmethod
    def _write_snapshot(cls, path: Path) -> None:
        from safetensors.torch import save_file

        config_dict = cls._config_dict()
        model = MegaGemmLlama(LlamaConfig.from_dict(config_dict)).to(dtype=torch.float16)
        state = model.state_dict()
        cfg = model.config
        q_size = cfg.num_attention_heads * cfg.head_dim
        k_size = cfg.num_key_value_heads * cfg.head_dim
        v_size = cfg.num_key_value_heads * cfg.head_dim
        qkv = state["layers.0.self_attn.qkv_proj.weight"]
        gate_up = state["layers.0.mlp.gate_up_proj.weight"]
        hf_state = {
            "model.embed_tokens.weight": state["embed_tokens.weight"].clone(),
            "model.norm.weight": state["norm.weight"].clone(),
            "lm_head.weight": state["lm_head.weight"].clone(),
            "model.layers.0.self_attn.q_proj.weight": qkv[:q_size].clone(),
            "model.layers.0.self_attn.k_proj.weight": qkv[q_size : q_size + k_size].clone(),
            "model.layers.0.self_attn.v_proj.weight": qkv[
                q_size + k_size : q_size + k_size + v_size
            ].clone(),
            "model.layers.0.self_attn.o_proj.weight": state[
                "layers.0.self_attn.o_proj.weight"
            ].clone(),
            "model.layers.0.mlp.gate_proj.weight": gate_up[: cfg.intermediate_size].clone(),
            "model.layers.0.mlp.up_proj.weight": gate_up[cfg.intermediate_size :].clone(),
            "model.layers.0.mlp.down_proj.weight": state[
                "layers.0.mlp.down_proj.weight"
            ].clone(),
            "model.layers.0.input_layernorm.weight": state[
                "layers.0.input_layernorm.weight"
            ].clone(),
            "model.layers.0.post_attention_layernorm.weight": state[
                "layers.0.post_attention_layernorm.weight"
            ].clone(),
        }
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps(config_dict), encoding="utf-8")
        save_file(hf_state, str(path / "model.safetensors"))

    def test_dense_and_sparse_native_mgx_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot"
            self._write_snapshot(snapshot)
            for sparse24 in (False, True):
                artifact = root / ("sparse.mgx" if sparse24 else "dense.mgx")
                info = export_to_mgx(
                    str(snapshot),
                    artifact,
                    dtype="fp16",
                    quantize="native-int4",
                    sparsity="2:4" if sparse24 else "none",
                    export_mode="streaming",
                )
                manifest = info["manifest"]
                self.assertEqual(manifest["quantization"], "int4")
                self.assertEqual(
                    manifest["quantization_config"]["format"], "mgx-native-w4a16-v1"
                )
                self.assertEqual(manifest["sparsity"], "2:4" if sparse24 else "none")

                loaded = load_from_mgx(
                    artifact,
                    device="cpu",
                    dtype_override="fp16",
                    prefer_payload_cache=False,
                )
                module = loaded.layers[0].self_attn.qkv_proj
                self.assertIsInstance(module, NativeW4A16Linear)
                self.assertEqual(module.sparse24, sparse24)
                x = torch.randn(2, 32, dtype=torch.float16)
                y = module(x)
                self.assertEqual(tuple(y.shape), (2, 64))
                self.assertTrue(bool(torch.isfinite(y).all()))


if __name__ == "__main__":
    unittest.main()

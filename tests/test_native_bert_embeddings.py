import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megagemm.embeddings import EmbeddingEngine
from megagemm.embeddings.native_bert import (
    NativeBertEncoder,
    _normalize_native_bert_state_dict,
    is_native_bert_supported,
    load_native_bert_encoder,
)
from megagemm.embeddings.pooling import pool_hidden_states
from megagemm.kernels.paged_attention import packed_attention


class FakeTokenizer:
    def __init__(self):
        self.model_max_length = 32

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=None,
        return_tensors=None,
        pad_to_multiple_of=None,
        return_attention_mask=True,
    ):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        lengths = [max(1, len(text.split())) for text in texts]
        max_len = max(lengths)
        input_ids = torch.zeros(len(texts), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), max_len, dtype=torch.long)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = torch.arange(1, length + 1, dtype=torch.long)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _build_tiny_bert_checkpoint(tmpdir: str) -> str:
    from transformers import BertConfig, BertModel

    config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = BertModel(config).eval()
    model.save_pretrained(tmpdir, safe_serialization=True)
    return tmpdir


def _build_tiny_bert_mlm_checkpoint(tmpdir: str) -> str:
    from transformers import BertConfig, BertForMaskedLM

    config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = BertForMaskedLM(config).eval()
    model.save_pretrained(tmpdir, safe_serialization=True)
    return tmpdir


def test_native_bert_encoder_matches_hf_last_hidden_state():
    from transformers import BertModel

    torch.manual_seed(0)
    model_dir = _build_tiny_bert_checkpoint(tempfile.mkdtemp(prefix="native-bert-hf-"))
    hf_model = BertModel.from_pretrained(model_dir, local_files_only=True).eval()
    native_model = load_native_bert_encoder(model_dir, device="cpu")

    input_ids = torch.tensor(
        [
            [5, 7, 9, 11, 0, 0],
            [3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    with torch.inference_mode():
        hf_hidden = hf_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        native_hidden = native_model(input_ids=input_ids, attention_mask=attention_mask)

    assert hf_hidden.shape == native_hidden.shape
    assert torch.allclose(hf_hidden, native_hidden, atol=1e-5, rtol=1e-5)


def test_native_bert_encoder_loads_masked_lm_checkpoint_backbone():
    from transformers import BertForMaskedLM

    torch.manual_seed(0)
    model_dir = _build_tiny_bert_mlm_checkpoint(tempfile.mkdtemp(prefix="native-bert-mlm-"))
    hf_model = BertForMaskedLM.from_pretrained(model_dir, local_files_only=True).eval()
    native_model = load_native_bert_encoder(model_dir, device="cpu")

    input_ids = torch.tensor(
        [
            [5, 7, 9, 11, 0, 0],
            [3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    with torch.inference_mode():
        hf_hidden = hf_model.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        native_hidden = native_model(input_ids=input_ids, attention_mask=attention_mask)

    assert hf_hidden.shape == native_hidden.shape
    assert torch.allclose(hf_hidden, native_hidden, atol=1e-5, rtol=1e-5)


def test_native_bert_support_accepts_encoder_compatible_task_heads():
    fake_cfg = type(
        "Cfg",
        (),
        {
            "model_type": "bert",
            "architectures": ["BertForMaskedLM"],
            "position_embedding_type": "absolute",
        },
    )()
    assert is_native_bert_supported(fake_cfg) is True


def test_native_bert_state_dict_normalizes_gamma_beta_layernorm_keys():
    raw = {
        "bert.embeddings.LayerNorm.gamma": torch.ones(4),
        "bert.embeddings.LayerNorm.beta": torch.zeros(4),
        "bert.encoder.layer.0.attention.output.LayerNorm.gamma": torch.ones(4),
        "bert.encoder.layer.0.attention.output.LayerNorm.beta": torch.zeros(4),
    }
    normalized = _normalize_native_bert_state_dict(raw)
    assert "embeddings.LayerNorm.weight" in normalized
    assert "embeddings.LayerNorm.bias" in normalized
    assert "encoder.layer.0.attention.output.LayerNorm.weight" in normalized
    assert "encoder.layer.0.attention.output.LayerNorm.bias" in normalized


def test_native_bert_encoder_loads_gamma_beta_style_state_dict():
    from transformers import BertConfig, BertModel

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    hf_model = BertModel(config).eval()
    state_dict = {}
    for key, value in hf_model.state_dict().items():
        renamed = key.replace("LayerNorm.weight", "LayerNorm.gamma").replace(
            "LayerNorm.bias", "LayerNorm.beta"
        )
        state_dict[renamed] = value

    native_model = NativeBertEncoder.from_hf_state_dict(config, state_dict)

    input_ids = torch.tensor(
        [
            [5, 7, 9, 11, 0, 0],
            [3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    with torch.inference_mode():
        hf_hidden = hf_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        native_hidden = native_model(input_ids=input_ids, attention_mask=attention_mask)

    assert torch.allclose(hf_hidden, native_hidden, atol=1e-5, rtol=1e-5)


def test_packed_attention_noncausal_matches_reference_sdpa():
    torch.manual_seed(0)
    lengths = [3, 2]
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    total_tokens = sum(lengths)

    q = torch.randn(total_tokens, 4, 8, dtype=torch.float32)
    k = torch.randn(total_tokens, 4, 8, dtype=torch.float32)
    v = torch.randn(total_tokens, 4, 8, dtype=torch.float32)

    packed_out = packed_attention(q, k, v, cu_seqlens, causal=False)
    ref_out = torch.empty_like(q)
    boundaries = cu_seqlens.tolist()
    for idx in range(len(lengths)):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        qi = q[start:end].transpose(0, 1).unsqueeze(0)
        ki = k[start:end].transpose(0, 1).unsqueeze(0)
        vi = v[start:end].transpose(0, 1).unsqueeze(0)
        oi = torch.nn.functional.scaled_dot_product_attention(
            qi, ki, vi, is_causal=False,
        )
        ref_out[start:end] = oi.squeeze(0).transpose(0, 1)

    assert torch.allclose(packed_out, ref_out, atol=1e-5, rtol=1e-5)


def test_native_bert_padding_free_preserves_valid_tokens_and_pooled_embeddings():
    from transformers import BertModel

    torch.manual_seed(0)
    model_dir = _build_tiny_bert_checkpoint(tempfile.mkdtemp(prefix="native-bert-padding-free-"))
    hf_model = BertModel.from_pretrained(model_dir, local_files_only=True).eval()
    native_model = load_native_bert_encoder(
        model_dir,
        device="cpu",
        padding_free=True,
        padding_free_force=True,
    )

    input_ids = torch.tensor(
        [
            [5, 7, 9, 11, 0, 0],
            [3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    with torch.inference_mode():
        hf_hidden = hf_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        native_hidden = native_model(input_ids=input_ids, attention_mask=attention_mask)

    valid_mask = attention_mask.to(dtype=torch.bool)
    assert torch.allclose(hf_hidden[valid_mask], native_hidden[valid_mask], atol=1e-5, rtol=1e-5)

    hf_pooled = pool_hidden_states(hf_hidden, attention_mask, ("mean", "lasttoken"))
    native_pooled = pool_hidden_states(native_hidden, attention_mask, ("mean", "lasttoken"))
    assert torch.allclose(hf_pooled, native_pooled, atol=1e-5, rtol=1e-5)


def test_native_bert_padding_free_reuses_packed_metadata_across_layers():
    import megagemm.embeddings.native_bert as native_bert_mod
    from transformers import BertConfig

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = NativeBertEncoder(config, padding_free=True, padding_free_force=True).eval()
    input_ids = torch.tensor(
        [
            [5, 7, 9, 11, 0, 0],
            [3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    orig_prepare = native_bert_mod.prepare_packed_attention_metadata
    orig_packed_attention = native_bert_mod.packed_attention
    calls = {"prepare": 0, "packed_meta_ids": []}
    try:
        def _wrapped_prepare(cu_seqlens, head_dim):
            calls["prepare"] += 1
            return orig_prepare(cu_seqlens, head_dim)

        def _wrapped_packed_attention(
            q,
            k,
            v,
            cu_seqlens,
            scale=None,
            causal=True,
            packed_meta=None,
        ):
            calls["packed_meta_ids"].append(id(packed_meta))
            assert packed_meta is not None
            return orig_packed_attention(
                q,
                k,
                v,
                cu_seqlens,
                scale=scale,
                causal=causal,
                packed_meta=packed_meta,
            )

        native_bert_mod.prepare_packed_attention_metadata = _wrapped_prepare
        native_bert_mod.packed_attention = _wrapped_packed_attention

        with torch.inference_mode():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        native_bert_mod.prepare_packed_attention_metadata = orig_prepare
        native_bert_mod.packed_attention = orig_packed_attention

    assert calls["prepare"] == 1
    assert len(calls["packed_meta_ids"]) == config.num_hidden_layers
    assert len(set(calls["packed_meta_ids"])) == 1


def test_embedding_engine_native_backend_matches_hf_backend():
    torch.manual_seed(0)
    model_dir = _build_tiny_bert_checkpoint(tempfile.mkdtemp(prefix="native-bert-engine-"))
    tokenizer = FakeTokenizer()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
        native_engine = EmbeddingEngine(
            model_dir,
            device="cpu",
            backend="native",
            normalize=False,
            local_files_only=True,
        )
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
        hf_engine = EmbeddingEngine(
            model_dir,
            device="cpu",
            backend="hf",
            normalize=False,
            local_files_only=True,
        )

    texts = ["alpha beta gamma delta", "one two"]
    native_embeddings = native_engine.encode(texts, batch_size=2)
    hf_embeddings = hf_engine.encode(texts, batch_size=2)

    assert native_engine._runtime_backend == "native"
    assert hf_engine._runtime_backend == "hf"
    assert native_embeddings.shape == hf_embeddings.shape
    assert torch.allclose(native_embeddings, hf_embeddings, atol=1e-5, rtol=1e-5)


def main():
    test_native_bert_encoder_matches_hf_last_hidden_state()
    test_native_bert_encoder_loads_masked_lm_checkpoint_backbone()
    test_native_bert_support_accepts_encoder_compatible_task_heads()
    test_native_bert_state_dict_normalizes_gamma_beta_layernorm_keys()
    test_native_bert_encoder_loads_gamma_beta_style_state_dict()
    test_packed_attention_noncausal_matches_reference_sdpa()
    test_native_bert_padding_free_preserves_valid_tokens_and_pooled_embeddings()
    test_native_bert_padding_free_reuses_packed_metadata_across_layers()
    test_embedding_engine_native_backend_matches_hf_backend()
    print("native bert embedding tests ok")


if __name__ == "__main__":
    main()

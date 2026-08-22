import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megagemm.embeddings.engine import EmbeddingEngine
from megagemm.embeddings.formats import load_sentence_transformer_spec
from megagemm.embeddings.pooling import normalize_embeddings, pool_hidden_states


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_dense_state(path: Path, weight: torch.Tensor, bias: torch.Tensor = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"linear.weight": weight}
    if bias is not None:
        state["linear.bias"] = bias
    torch.save(state, path)


class FakeTokenizer:
    def __init__(self):
        self.calls = []
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
        return_length=False,
        add_special_tokens=True,
        **kwargs,
    ):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        self.calls.append({"texts": texts, "return_tensors": return_tensors, "return_length": return_length})

        lengths = [max(1, len(text.split())) for text in texts]
        max_len = max(lengths)
        input_ids = torch.zeros(len(texts), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), max_len, dtype=torch.long)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = torch.arange(1, length + 1, dtype=torch.long)
            attention_mask[row, :length] = 1

        result = {}
        if return_tensors == "pt":
            result["input_ids"] = input_ids
            result["attention_mask"] = attention_mask
        else:
            result["input_ids"] = [row[:length].tolist() for row, length in zip(input_ids, lengths)]
            result["attention_mask"] = [row[:length].tolist() for row, length in zip(attention_mask, lengths)]
        if return_length:
            result["length"] = lengths
        return result


class FakeModel(torch.nn.Module):
    def __init__(self, hidden_size=3):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids, attention_mask):
        base = input_ids.float()
        hidden = torch.stack([base, base + 1.0, base + 2.0], dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_pool_hidden_states_combines_modes():
    hidden = torch.tensor(
        [
            [[1.0, 10.0], [3.0, 30.0], [0.0, 0.0]],
            [[2.0, 20.0], [4.0, 40.0], [6.0, 60.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    pooled = pool_hidden_states(hidden, mask, ("mean", "lasttoken"))
    expected = torch.tensor(
        [
            [2.0, 20.0, 3.0, 30.0],
            [4.0, 40.0, 6.0, 60.0],
        ]
    )
    assert torch.allclose(pooled, expected)


def test_sentence_transformer_spec_parsing_with_dense_and_prompts():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_json(
            root / "modules.json",
            [
                {"idx": 0, "name": "0", "path": "0_Transformer", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
                {"idx": 3, "name": "3", "path": "3_Normalize", "type": "sentence_transformers.models.Normalize"},
            ],
        )
        _write_json(
            root / "config_sentence_transformers.json",
            {
                "prompts": {"query": "query: ", "document": "passage: "},
                "default_prompt_name": "query",
            },
        )
        _write_json(
            root / "1_Pooling" / "config.json",
            {
                "word_embedding_dimension": 3,
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": True,
                "pooling_mode_lasttoken": True,
            },
        )
        _write_json(
            root / "2_Dense" / "config.json",
            {
                "in_features": 6,
                "out_features": 4,
                "bias": True,
                "activation_function": "torch.nn.modules.activation.Tanh",
            },
        )
        _write_dense_state(
            root / "2_Dense" / "pytorch_model.bin",
            weight=torch.eye(4, 6),
            bias=torch.zeros(4),
        )

        spec = load_sentence_transformer_spec(str(root))
        assert spec is not None
        assert spec.transformer_module_dir == str(root / "0_Transformer")
        assert spec.pooling.modes == ("mean", "lasttoken")
        assert spec.normalize is True
        assert spec.prompts["query"] == "query: "
        assert spec.default_prompt_name == "query"
        assert len(spec.dense_layers) == 1
        assert spec.dense_layers[0].activation == "tanh"


def test_embedding_engine_encode_query_applies_prompt_dense_and_normalize():
    with tempfile.TemporaryDirectory(prefix="e5-small-") as tmpdir:
        root = Path(tmpdir)
        _write_json(
            root / "modules.json",
            [
                {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
                {"idx": 3, "name": "3", "path": "3_Normalize", "type": "sentence_transformers.models.Normalize"},
            ],
        )
        _write_json(
            root / "1_Pooling" / "config.json",
            {"word_embedding_dimension": 3, "pooling_mode_mean_tokens": True},
        )
        _write_json(
            root / "2_Dense" / "config.json",
            {"in_features": 3, "out_features": 2, "bias": True, "activation_function": "Identity"},
        )
        _write_dense_state(
            root / "2_Dense" / "pytorch_model.bin",
            weight=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            bias=torch.tensor([0.5, -0.5]),
        )

        tokenizer = FakeTokenizer()
        model = FakeModel(hidden_size=3)

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), patch(
            "transformers.AutoModel.from_pretrained", return_value=model
        ):
            engine = EmbeddingEngine(str(root), device="cpu")
            embeddings = engine.encode_query(["alpha beta", "gamma"], batch_size=2)

        assert tokenizer.calls[0]["texts"] == ["query: alpha beta", "query: gamma"]
        assert embeddings.shape == (2, 2)

        pooled = torch.tensor(
            [
                [2.0, 3.0, 4.0],  # mean of tokens 1,2,3 after query prefix
                [1.5, 2.5, 3.5],  # mean of tokens 1,2
            ]
        )
        projected = torch.stack(
            [
                torch.tensor([pooled[0, 0] + 0.5, pooled[0, 1] - 0.5]),
                torch.tensor([pooled[1, 0] + 0.5, pooled[1, 1] - 0.5]),
            ]
        )
        expected = normalize_embeddings(projected)
        assert torch.allclose(embeddings, expected, atol=1e-6, rtol=1e-6)


def test_embedding_engine_uses_transformer_subdir_and_default_prompt():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_json(
            root / "modules.json",
            [
                {"idx": 0, "name": "0", "path": "0_Transformer", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            ],
        )
        _write_json(
            root / "config_sentence_transformers.json",
            {"prompts": {"query": "prompt: "}, "default_prompt_name": "query"},
        )
        _write_json(
            root / "1_Pooling" / "config.json",
            {"word_embedding_dimension": 3, "pooling_mode_cls_token": True},
        )

        tokenizer = FakeTokenizer()
        model = FakeModel(hidden_size=3)

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as tok_patch, patch(
            "transformers.AutoModel.from_pretrained", return_value=model
        ) as model_patch:
            engine = EmbeddingEngine(str(root), device="cpu", normalize=False)
            embedding = engine.encode("hello world")

        expected_backbone = str(root / "0_Transformer")
        assert tok_patch.call_args.args[0] == expected_backbone
        assert model_patch.call_args.args[0] == expected_backbone
        assert tokenizer.calls[0]["texts"] == ["prompt: hello world"]
        assert embedding.shape == (3,)


def test_embedding_engine_token_budget_batching_splits_mixed_lengths():
    tokenizer = FakeTokenizer()
    model = FakeModel(hidden_size=3)

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), patch(
        "transformers.AutoModel.from_pretrained", return_value=model
    ):
        engine = EmbeddingEngine(
            str(Path.cwd()),
            device="cpu",
            normalize=False,
            max_batch_tokens=6,
        )
        embeddings = engine.encode(
            ["one two three", "one two", "one", "one"],
            batch_size=4,
        )

    batch_calls = [call["texts"] for call in tokenizer.calls if call["return_tensors"] == "pt"]
    assert len(batch_calls) == 2
    assert batch_calls[0] == ["one two three", "one two"]
    assert batch_calls[1] == ["one", "one"]
    assert embeddings.shape == (4, 3)
    assert len(engine._last_batch_plan) == 2
    assert engine._last_batch_plan[0]["padded_tokens"] == 6


def test_embedding_engine_native_padding_free_flag_is_forwarded():
    tokenizer = FakeTokenizer()
    native_model = FakeModel(hidden_size=3)
    fake_config = SimpleNamespace(
        hidden_size=3,
        model_type="bert",
        architectures=["BertModel"],
        position_embedding_type="absolute",
    )

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), patch(
        "transformers.AutoConfig.from_pretrained", return_value=fake_config
    ), patch(
        "megagemm.embeddings.engine.is_native_bert_supported", return_value=True
    ), patch(
        "megagemm.embeddings.engine.load_native_bert_encoder", return_value=native_model
    ) as native_loader:
        engine = EmbeddingEngine(
            str(Path.cwd()),
            device="cpu",
            backend="native",
            normalize=False,
            native_padding_free=False,
        )

    assert engine._runtime_backend == "native"
    assert native_loader.call_args.kwargs["padding_free"] is False


def main():
    test_pool_hidden_states_combines_modes()
    test_sentence_transformer_spec_parsing_with_dense_and_prompts()
    test_embedding_engine_encode_query_applies_prompt_dense_and_normalize()
    test_embedding_engine_uses_transformer_subdir_and_default_prompt()
    test_embedding_engine_token_budget_batching_splits_mixed_lengths()
    test_embedding_engine_native_padding_free_flag_is_forwarded()
    print("embedding tests ok")


if __name__ == "__main__":
    main()

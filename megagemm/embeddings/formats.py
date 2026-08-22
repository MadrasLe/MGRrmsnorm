"""
Helpers for Sentence Transformers-compatible model layouts.

This module intentionally supports the common deployment path:
- Transformer backbone
- Pooling
- Optional Dense projection(s)
- Optional Normalize
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .pooling import normalize_pooling_modes

__all__ = [
    "DenseSpec",
    "PoolingSpec",
    "SentenceTransformerSpec",
    "load_module_state_dict",
    "load_sentence_transformer_spec",
]


@dataclass(frozen=True)
class PoolingSpec:
    modes: Tuple[str, ...] = ("mean",)
    word_embedding_dimension: Optional[int] = None


@dataclass(frozen=True)
class DenseSpec:
    module_dir: str
    in_features: int
    out_features: int
    bias: bool = True
    activation: str = "identity"


@dataclass
class SentenceTransformerSpec:
    transformer_module_dir: Optional[str] = None
    pooling: Optional[PoolingSpec] = None
    dense_layers: List[DenseSpec] = field(default_factory=list)
    normalize: bool = False
    prompts: Dict[str, str] = field(default_factory=dict)
    default_prompt_name: Optional[str] = None
    unsupported_modules: List[str] = field(default_factory=list)


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_activation(name: Optional[str]) -> str:
    if not name:
        return "identity"
    key = str(name).strip().split(".")[-1].lower()
    aliases = {
        "identity": "identity",
        "relu": "relu",
        "gelu": "gelu",
        "silu": "silu",
        "swish": "silu",
        "tanh": "tanh",
    }
    return aliases.get(key, "identity")


def _load_pooling_spec(module_dir: Path) -> PoolingSpec:
    config = _read_json(module_dir / "config.json") or {}
    modes = []
    if config.get("pooling_mode"):
        modes.append(config["pooling_mode"])
    else:
        flags = {
            "pooling_mode_cls_token": "cls",
            "pooling_mode_mean_tokens": "mean",
            "pooling_mode_max_tokens": "max",
            "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len",
            "pooling_mode_weightedmean_tokens": "weightedmean",
            "pooling_mode_lasttoken": "lasttoken",
            "pooling_mode_last_token": "lasttoken",
        }
        for key, mode in flags.items():
            if config.get(key):
                modes.append(mode)
    return PoolingSpec(
        modes=normalize_pooling_modes(modes or ("mean",)),
        word_embedding_dimension=config.get("word_embedding_dimension"),
    )


def _load_dense_spec(module_dir: Path) -> DenseSpec:
    config = _read_json(module_dir / "config.json") or {}
    return DenseSpec(
        module_dir=str(module_dir),
        in_features=int(config.get("in_features", 0) or 0),
        out_features=int(config.get("out_features", 0) or 0),
        bias=bool(config.get("bias", True)),
        activation=_normalize_activation(config.get("activation_function")),
    )


def load_module_state_dict(module_dir: str) -> dict:
    path = Path(module_dir)
    safetensors_path = path / "model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Install safetensors to load Sentence Transformers dense modules: pip install safetensors"
            ) from exc
        return load_file(str(safetensors_path))

    pytorch_path = path / "pytorch_model.bin"
    if pytorch_path.exists():
        import torch

        state = torch.load(str(pytorch_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
        return state

    raise FileNotFoundError(f"No Dense module weights found in {module_dir}")


def load_sentence_transformer_spec(model_dir: str) -> Optional[SentenceTransformerSpec]:
    root = Path(model_dir)
    modules = _read_json(root / "modules.json")
    meta = _read_json(root / "config_sentence_transformers.json") or {}

    if modules is None and not meta:
        return None

    spec = SentenceTransformerSpec()
    prompts = meta.get("prompts") or {}
    if isinstance(prompts, dict):
        spec.prompts = {str(k): str(v) for k, v in prompts.items()}
    default_prompt_name = meta.get("default_prompt_name")
    if default_prompt_name is not None:
        spec.default_prompt_name = str(default_prompt_name)

    for module in modules or []:
        module_type = str(module.get("type", ""))
        short_type = module_type.rsplit(".", 1)[-1]
        relative_path = str(module.get("path", "") or "")
        module_dir = root / relative_path if relative_path else root

        if short_type == "Transformer":
            spec.transformer_module_dir = str(module_dir)
            continue
        if short_type == "Pooling":
            spec.pooling = _load_pooling_spec(module_dir)
            continue
        if short_type == "Dense":
            spec.dense_layers.append(_load_dense_spec(module_dir))
            continue
        if short_type == "Normalize":
            spec.normalize = True
            continue
        spec.unsupported_modules.append(module_type)

    return spec

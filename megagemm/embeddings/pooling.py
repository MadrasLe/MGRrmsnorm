"""
Pooling utilities for encoder embeddings.

Supports the common Sentence Transformers pooling modes:
- cls
- mean
- max
- mean_sqrt_len
- weightedmean
- lasttoken
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch

__all__ = [
    "normalize_pooling_modes",
    "pool_hidden_states",
    "normalize_embeddings",
    "pooling_output_dim",
]


_POOLING_ALIASES = {
    "cls": "cls",
    "cls_token": "cls",
    "mean": "mean",
    "mean_tokens": "mean",
    "max": "max",
    "max_tokens": "max",
    "mean_sqrt_len": "mean_sqrt_len",
    "mean_sqrt_len_tokens": "mean_sqrt_len",
    "weightedmean": "weightedmean",
    "weighted_mean": "weightedmean",
    "weightedmean_tokens": "weightedmean",
    "lasttoken": "lasttoken",
    "last_token": "lasttoken",
}


def normalize_pooling_modes(modes: Iterable[str]) -> Tuple[str, ...]:
    normalized = []
    for mode in modes:
        key = str(mode).strip().lower()
        if not key:
            continue
        mapped = _POOLING_ALIASES.get(key)
        if mapped is None:
            raise ValueError(f"Unsupported pooling mode: {mode}")
        normalized.append(mapped)
    if not normalized:
        return ("mean",)
    return tuple(normalized)


def _masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor, sqrt_len: bool = False) -> torch.Tensor:
    mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    if sqrt_len:
        denom = denom.sqrt()
    return summed / denom


def _masked_max(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(dtype=torch.bool).unsqueeze(-1)
    masked = hidden_states.masked_fill(~mask, torch.finfo(hidden_states.dtype).min)
    values = masked.max(dim=1).values
    all_pad = ~attention_mask.to(dtype=torch.bool).any(dim=1)
    if all_pad.any():
        values[all_pad] = 0
    return values


def _last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.to(dtype=torch.long).sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_idx, lengths]


def _weighted_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.to(dtype=hidden_states.dtype).cumsum(dim=1)
    weights = weights * attention_mask.to(dtype=hidden_states.dtype)
    weights = weights.unsqueeze(-1)
    summed = (hidden_states * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return summed / denom


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    modes: Iterable[str],
) -> torch.Tensor:
    normalized_modes = normalize_pooling_modes(modes)
    outputs = []
    for mode in normalized_modes:
        if mode == "cls":
            outputs.append(hidden_states[:, 0])
        elif mode == "mean":
            outputs.append(_masked_mean(hidden_states, attention_mask, sqrt_len=False))
        elif mode == "max":
            outputs.append(_masked_max(hidden_states, attention_mask))
        elif mode == "mean_sqrt_len":
            outputs.append(_masked_mean(hidden_states, attention_mask, sqrt_len=True))
        elif mode == "weightedmean":
            outputs.append(_weighted_mean(hidden_states, attention_mask))
        elif mode == "lasttoken":
            outputs.append(_last_token(hidden_states, attention_mask))
        else:
            raise ValueError(f"Unsupported pooling mode: {mode}")
    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=-1)


def normalize_embeddings(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings, p=2, dim=-1, eps=eps)


def pooling_output_dim(hidden_size: int, modes: Iterable[str]) -> int:
    return hidden_size * len(normalize_pooling_modes(modes))

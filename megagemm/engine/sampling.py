"""
🎲 Sampling Utilities for MegaGemm
-----------------------------------
Token sampling strategies for LLM inference.
"""

import torch
from typing import Optional

__all__ = ['sample_logits']


def sample_logits(
    logits: torch.Tensor,        # [batch, vocab_size]
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    past_tokens: Optional[torch.Tensor] = None,  # [batch, seq_len]
) -> torch.Tensor:
    """
    Sample next tokens from logits with various strategies.

    Args:
        logits: Raw logits [batch, vocab_size]
        temperature: Sampling temperature (0 = greedy, 1 = normal)
        top_k: Keep only top-k tokens (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        repetition_penalty: Penalize repeated tokens (1.0 = disabled)
        past_tokens: Previously generated tokens for repetition penalty

    Returns:
        next_tokens: [batch] sampled token IDs
    """
    # Repetition penalty
    if repetition_penalty != 1.0 and past_tokens is not None:
        # Fully vectorized: zero Python loops, zero GPU↔CPU syncs
        # past_tokens: [batch, seq_len], may contain -1 padding
        valid_mask = past_tokens >= 0  # [batch, seq_len]
        safe_ids = past_tokens.clamp(min=0)  # replace -1 with 0 for gather

        # Gather logits at past token positions
        gathered = torch.gather(logits, 1, safe_ids)  # [batch, seq_len]

        # Apply penalty: divide positives, multiply negatives
        penalized = torch.where(
            gathered > 0,
            gathered / repetition_penalty,
            gathered * repetition_penalty,
        )

        # Only apply where tokens are valid (not padding)
        penalized = torch.where(valid_mask, penalized, gathered)

        # Scatter back into logits
        logits.scatter_(1, safe_ids, penalized)

    # Greedy
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    # Temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Remove tokens with cumulative prob above threshold
        mask = cumulative_probs - probs > top_p
        sorted_logits[mask] = float('-inf')

        # Scatter back
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    # Sample
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

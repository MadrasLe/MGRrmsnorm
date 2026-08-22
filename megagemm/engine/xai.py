"""
🔍 XAI — Explainable AI Module for MegaGemm
---------------------------------------------
Opt-in interpretability features for LLM inference:
  - Top-K token probabilities per generation step
  - Confidence scoring (geometric mean of chosen token probs)
  - Logit Lens: per-layer hidden state projections
  - Entropy-based hallucination detection

All features are disabled by default (zero overhead when off).

Usage:
    text, report = engine.generate("Hello", xai=True, xai_top_k=5)
    report.to_json("report.json")
    report.to_txt("report.txt")
    print(report.summary())

Author: Gabriel Yogi
"""

import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


__all__ = ['TokenPrediction', 'GenerationStep', 'XAIReport']


@dataclass
class TokenPrediction:
    """A single token with its probability."""
    token_id: int
    token_str: str
    probability: float

    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "token": self.token_str,
            "probability": round(self.probability, 6),
        }


@dataclass
class GenerationStep:
    """One decode step: the chosen token + top-K alternatives + entropy."""
    position: int
    chosen: TokenPrediction
    top_k: List[TokenPrediction]
    entropy: float = 0.0  # Shannon entropy of token distribution (higher = more uncertain)
    logit_lens: Optional[Dict[int, List[TokenPrediction]]] = None

    def to_dict(self) -> dict:
        d = {
            "position": self.position,
            "chosen": self.chosen.to_dict(),
            "top_k": [t.to_dict() for t in self.top_k],
            "entropy": round(self.entropy, 6),
        }
        if self.logit_lens is not None:
            d["logit_lens"] = {
                str(layer): [t.to_dict() for t in preds]
                for layer, preds in self.logit_lens.items()
            }
        return d


# Entropy thresholds for hallucination risk classification
# Calibrated for typical LLM vocabulary sizes (32k-128k tokens)
# Low entropy (<2.0): model is confident, likely accurate
# Medium entropy (2.0-4.0): some uncertainty, may need verification
# High entropy (>4.0): model is uncertain, high hallucination risk
ENTROPY_LOW = 2.0
ENTROPY_HIGH = 4.0


def classify_hallucination_risk(
    mean_entropy: float,
    confidence: float,
) -> str:
    """
    Classify hallucination risk based on mean entropy and confidence.

    Uses a combined heuristic:
    - High entropy + low confidence → HIGH risk
    - Medium entropy or medium confidence → MEDIUM risk
    - Low entropy + high confidence → LOW risk
    """
    if mean_entropy > ENTROPY_HIGH or confidence < 0.15:
        return "HIGH"
    elif mean_entropy > ENTROPY_LOW or confidence < 0.5:
        return "MEDIUM"
    return "LOW"


@dataclass
class XAIReport:
    """
    Full interpretability report for one generation.

    Contains per-step token probabilities, confidence score,
    entropy-based hallucination detection, and optionally
    Logit Lens layer-by-layer analysis.
    """
    prompt: str
    generated_text: str
    steps: List[GenerationStep]
    confidence_score: float
    model_name: str = ""
    timestamp: str = ""
    num_layers: int = 0

    # Computed in __post_init__
    mean_entropy: float = 0.0
    hallucination_risk: str = "LOW"
    high_entropy_steps: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        # Auto-compute entropy stats
        if self.steps:
            entropies = [s.entropy for s in self.steps]
            self.mean_entropy = sum(entropies) / len(entropies)
            self.high_entropy_steps = sum(1 for e in entropies if e > ENTROPY_HIGH)
            self.hallucination_risk = classify_hallucination_risk(
                self.mean_entropy, self.confidence_score
            )

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "confidence_score": round(self.confidence_score, 6),
            "mean_entropy": round(self.mean_entropy, 6),
            "hallucination_risk": self.hallucination_risk,
            "high_entropy_steps": self.high_entropy_steps,
            "num_steps": len(self.steps),
            "num_layers": self.num_layers,
            "has_logit_lens": any(s.logit_lens is not None for s in self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, path: str, indent: int = 2) -> None:
        """Export report as formatted JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    def to_txt(self, path: str) -> None:
        """Export report as human-readable TXT file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.summary())

    def summary(self) -> str:
        """Human-readable summary of the generation."""
        lines = []
        lines.append("=" * 60)
        lines.append("🔍 MegaGemm XAI Report")
        lines.append("=" * 60)
        if self.model_name:
            lines.append(f"Model:      {self.model_name}")
        lines.append(f"Timestamp:  {self.timestamp}")
        lines.append(f"Confidence: {self.confidence_score:.4f}")
        lines.append(f"Entropy:    {self.mean_entropy:.4f} (avg)")

        # Hallucination risk with emoji
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
        emoji = risk_emoji.get(self.hallucination_risk, "⚪")
        lines.append(f"Halluc.Risk:{emoji} {self.hallucination_risk}")
        if self.high_entropy_steps > 0:
            lines.append(f"⚠️  {self.high_entropy_steps}/{len(self.steps)} steps with high entropy")
        lines.append(f"Steps:      {len(self.steps)}")
        lines.append("")
        lines.append(f"Prompt: {self.prompt}")
        lines.append(f"Output: {self.generated_text}")
        lines.append("")
        lines.append("-" * 60)
        lines.append("Token Probabilities (per step)")
        lines.append("-" * 60)

        for step in self.steps:
            chosen = step.chosen
            bar = _prob_bar(chosen.probability)
            # Entropy indicator
            if step.entropy > ENTROPY_HIGH:
                e_flag = " 🔴"
            elif step.entropy > ENTROPY_LOW:
                e_flag = " 🟡"
            else:
                e_flag = ""
            lines.append(
                f"  [{step.position:3d}] "
                f"{chosen.token_str!r:<20s} "
                f"p={chosen.probability:.4f} H={step.entropy:.2f}{e_flag} {bar}"
            )
            for alt in step.top_k:
                if alt.token_id == chosen.token_id:
                    continue  # skip the chosen one in alternatives
                lines.append(
                    f"        "
                    f"{alt.token_str!r:<20s} "
                    f"p={alt.probability:.4f}"
                )

            # Logit Lens
            if step.logit_lens:
                lines.append(f"        ── Logit Lens ──")
                # Show first, middle, last layers for brevity
                layer_ids = sorted(step.logit_lens.keys())
                show_layers = _select_representative_layers(layer_ids)
                for lid in show_layers:
                    preds = step.logit_lens[lid]
                    if preds:
                        top = preds[0]
                        lines.append(
                            f"        Layer {lid:2d}: "
                            f"{top.token_str!r:<15s} "
                            f"p={top.probability:.4f}"
                        )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


def _prob_bar(p: float, width: int = 20) -> str:
    """Visual probability bar: ████░░░░"""
    filled = int(p * width)
    return "█" * filled + "░" * (width - filled)


def _select_representative_layers(layer_ids: List[int], max_show: int = 5) -> List[int]:
    """Pick representative layers: first, last, and evenly spaced middle layers."""
    if len(layer_ids) <= max_show:
        return layer_ids

    n = len(layer_ids)
    indices = [0]
    step = (n - 1) / (max_show - 1)
    for i in range(1, max_show - 1):
        indices.append(int(i * step))
    indices.append(n - 1)

    # Remove duplicates, preserve order
    seen = set()
    result = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            result.append(layer_ids[idx])
    return result


def compute_confidence(probs: List[float]) -> float:
    """
    Compute confidence score as geometric mean of token probabilities.

    Returns 0.0 if any probability is 0, handles log-domain for stability.
    """
    if not probs:
        return 0.0

    log_sum = 0.0
    for p in probs:
        if p <= 0:
            return 0.0
        log_sum += math.log(p)

    return math.exp(log_sum / len(probs))


def extract_top_k_predictions(
    logits,         # torch.Tensor [vocab_size]
    tokenizer,
    k: int = 10,
) -> List[TokenPrediction]:
    """
    Extract top-K token predictions from logits.

    Args:
        logits: Raw logits tensor [vocab_size] (single position)
        tokenizer: HuggingFace tokenizer for id→string mapping
        k: Number of top predictions to return

    Returns:
        List of TokenPrediction sorted by probability (descending)
    """
    import torch

    probs = torch.softmax(logits.float(), dim=-1)
    k = min(k, probs.shape[-1])
    top_probs, top_ids = torch.topk(probs, k)

    predictions = []
    for i in range(k):
        tid = top_ids[i].item()
        predictions.append(TokenPrediction(
            token_id=tid,
            token_str=tokenizer.decode([tid]),
            probability=top_probs[i].item(),
        ))

    return predictions


def compute_entropy(logits) -> float:
    """
    Compute Shannon entropy of token probability distribution.

    Higher entropy = more uncertainty = higher hallucination risk.

    For reference (with vocab_size=32000):
    - Entropy ~0.5: very confident (one token dominates)
    - Entropy ~2.0: moderate confidence
    - Entropy ~4.0: uncertain (many plausible tokens)
    - Entropy ~10.3: max entropy (uniform distribution, log2(32000))

    Args:
        logits: Raw logits tensor [vocab_size]

    Returns:
        Shannon entropy in nats (base-e logarithm)
    """
    import torch

    probs = torch.softmax(logits.float(), dim=-1)
    # Clamp to avoid log(0)
    log_probs = torch.log(probs.clamp(min=1e-10))
    entropy = -(probs * log_probs).sum().item()
    return entropy

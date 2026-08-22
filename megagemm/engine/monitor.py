"""
📊 Monitor — Inference Monitoring Module for MegaGemm
------------------------------------------------------
Lightweight metrics collection for production LLM inference.
Aggregates latency, throughput, XAI quality signals, and resource stats
across requests, enabling drift detection and hallucination rate tracking.

All monitoring is opt-in via `monitor=True` on InferenceEngine.

Usage:
    engine = InferenceEngine("model", monitor=True)
    engine.generate("Hello", xai=True)

    stats = engine.get_monitor_stats()
    engine.export_monitor_log("log.jsonl")

Author: Gabriel Yogi
"""

import json
import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from collections import deque


__all__ = ['InferenceMonitor', 'RequestRecord', 'get_gpu_stats']


def get_gpu_stats() -> Dict[str, Any]:
    """
    Collect GPU resource stats (VRAM, device info).
    Returns empty dict if CUDA is not available.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {}

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        total = props.total_mem

        return {
            "gpu_name": props.name,
            "vram_used_mb": round(allocated / 1024**2, 1),
            "vram_reserved_mb": round(reserved / 1024**2, 1),
            "vram_total_mb": round(total / 1024**2, 1),
            "vram_usage_pct": round(allocated / total * 100, 1) if total > 0 else 0,
            "vram_free_mb": round((total - allocated) / 1024**2, 1),
        }
    except Exception:
        return {}


@dataclass
class RequestRecord:
    """Metrics for a single inference request."""
    timestamp: str
    prompt: str
    generated_text: str
    num_tokens_input: int
    num_tokens_output: int

    # Timing (milliseconds)
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    ttft_ms: float = 0.0        # Time to first token
    tokens_per_second: float = 0.0

    # XAI quality signals (populated when xai=True)
    confidence_score: float = -1.0   # -1 = not computed
    mean_entropy: float = -1.0
    hallucination_risk: str = ""
    high_entropy_steps: int = 0

    # Config context
    model_name: str = ""
    quantization: str = "fp16"

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "prompt": self.prompt[:100],  # Truncate for log readability
            "generated_text": self.generated_text[:100],
            "num_tokens_input": self.num_tokens_input,
            "num_tokens_output": self.num_tokens_output,
            "prefill_ms": round(self.prefill_ms, 2),
            "decode_ms": round(self.decode_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "ttft_ms": round(self.ttft_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "model_name": self.model_name,
            "quantization": self.quantization,
        }
        # Only include XAI fields if they were computed
        if self.confidence_score >= 0:
            d["confidence_score"] = round(self.confidence_score, 6)
            d["mean_entropy"] = round(self.mean_entropy, 6)
            d["hallucination_risk"] = self.hallucination_risk
            d["high_entropy_steps"] = self.high_entropy_steps
        return d


class InferenceMonitor:
    """
    Lightweight inference monitor that aggregates metrics across requests.

    Thread-safe. Collects timing, throughput, and XAI quality signals.
    Supports rolling windows for drift detection.

    Features:
    - Per-request metrics (latency, TPS, tokens, XAI quality)
    - Aggregate stats (mean, P95, P99 latencies, hallucination rate)
    - Rolling window for drift detection
    - JSONL log export
    """

    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Max records to keep in memory for rolling stats.
                         Older records are evicted (but still counted in totals).
        """
        self._records: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

        # Running totals (never evicted)
        self._total_requests = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0
        self._total_time_ms = 0.0
        self._hallucination_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

        # Start time for uptime tracking
        self._start_time = time.time()

    def record(self, rec: RequestRecord) -> None:
        """Add a request record to the monitor. Thread-safe."""
        with self._lock:
            self._records.append(rec)
            self._total_requests += 1
            self._total_tokens_in += rec.num_tokens_input
            self._total_tokens_out += rec.num_tokens_output
            self._total_time_ms += rec.total_ms
            if rec.hallucination_risk:
                self._hallucination_counts[rec.hallucination_risk] = (
                    self._hallucination_counts.get(rec.hallucination_risk, 0) + 1
                )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregated monitoring statistics.

        Returns a dict with performance, quality, and resource metrics
        computed from the rolling window of recent requests.
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return {
                "total_requests": 0,
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "message": "No requests recorded yet",
            }

        # --- Performance metrics ---
        latencies = [r.total_ms for r in records]
        ttfts = [r.ttft_ms for r in records if r.ttft_ms > 0]
        tps_list = [r.tokens_per_second for r in records if r.tokens_per_second > 0]
        prefill_times = [r.prefill_ms for r in records]
        decode_times = [r.decode_ms for r in records]

        # Percentiles
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        perf = {
            "avg_latency_ms": round(sum(latencies) / n, 2),
            "p50_latency_ms": round(latencies_sorted[n // 2], 2),
            "p95_latency_ms": round(latencies_sorted[int(n * 0.95)], 2) if n >= 20 else round(latencies_sorted[-1], 2),
            "p99_latency_ms": round(latencies_sorted[int(n * 0.99)], 2) if n >= 100 else round(latencies_sorted[-1], 2),
            "avg_prefill_ms": round(sum(prefill_times) / n, 2),
            "avg_decode_ms": round(sum(decode_times) / n, 2),
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else 0.0,
            "avg_tps": round(sum(tps_list) / len(tps_list), 2) if tps_list else 0.0,
        }

        # --- Quality metrics (XAI) ---
        xai_records = [r for r in records if r.confidence_score >= 0]

        if xai_records:
            confidences = [r.confidence_score for r in xai_records]
            entropies = [r.mean_entropy for r in xai_records]

            risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
            for r in xai_records:
                risk_counts[r.hallucination_risk] = risk_counts.get(r.hallucination_risk, 0) + 1

            total_xai = len(xai_records)
            quality = {
                "xai_enabled_requests": total_xai,
                "avg_confidence": round(sum(confidences) / total_xai, 4),
                "min_confidence": round(min(confidences), 4),
                "max_confidence": round(max(confidences), 4),
                "avg_entropy": round(sum(entropies) / total_xai, 4),
                "hallucination_rate_high": round(risk_counts.get("HIGH", 0) / total_xai, 4),
                "hallucination_rate_medium": round(risk_counts.get("MEDIUM", 0) / total_xai, 4),
                "risk_distribution": risk_counts,
            }
        else:
            quality = {"xai_enabled_requests": 0}

        # --- Throughput metrics ---
        tokens_in = [r.num_tokens_input for r in records]
        tokens_out = [r.num_tokens_output for r in records]

        throughput = {
            "total_requests": self._total_requests,
            "window_requests": len(records),
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_out": self._total_tokens_out,
            "avg_tokens_in": round(sum(tokens_in) / n, 1),
            "avg_tokens_out": round(sum(tokens_out) / n, 1),
        }

        # --- Uptime ---
        uptime = round(time.time() - self._start_time, 1)
        rps = self._total_requests / uptime if uptime > 0 else 0

        return {
            "uptime_seconds": uptime,
            "requests_per_second": round(rps, 3),
            "performance": perf,
            "quality": quality,
            "throughput": throughput,
            "gpu": get_gpu_stats(),
            "hallucination_totals": dict(self._hallucination_counts),
        }

    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get the N most recent request records as dicts."""
        with self._lock:
            records = list(self._records)
        return [r.to_dict() for r in records[-n:]]

    def export_log(self, path: str) -> int:
        """
        Export all records in the window as JSONL (one JSON per line).

        Returns the number of records exported.
        """
        with self._lock:
            records = list(self._records)

        with open(path, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + '\n')

        return len(records)

    def summary(self) -> str:
        """Human-readable monitoring dashboard."""
        stats = self.get_stats()

        if self._total_requests == 0:
            return "📊 MegaGemm Monitor — No requests yet"

        lines = []
        lines.append("=" * 60)
        lines.append("📊 MegaGemm Inference Monitor")
        lines.append("=" * 60)
        lines.append(f"Uptime:     {stats['uptime_seconds']:.0f}s")
        lines.append(f"Requests:   {stats['throughput']['total_requests']} total"
                     f" ({stats['throughput']['window_requests']} in window)")
        lines.append(f"RPS:        {stats['requests_per_second']:.3f}")

        lines.append("")
        lines.append("── Performance ──")
        perf = stats['performance']
        lines.append(f"  Latency:  avg={perf['avg_latency_ms']:.0f}ms"
                     f"  P50={perf['p50_latency_ms']:.0f}ms"
                     f"  P95={perf['p95_latency_ms']:.0f}ms")
        lines.append(f"  Prefill:  {perf['avg_prefill_ms']:.0f}ms avg")
        lines.append(f"  Decode:   {perf['avg_decode_ms']:.0f}ms avg")
        lines.append(f"  TTFT:     {perf['avg_ttft_ms']:.0f}ms avg")
        lines.append(f"  Speed:    {perf['avg_tps']:.1f} tok/s avg")

        lines.append("")
        lines.append("── Throughput ──")
        tp = stats['throughput']
        lines.append(f"  Tokens:   {tp['total_tokens_in']} in, {tp['total_tokens_out']} out")
        lines.append(f"  Avg/req:  {tp['avg_tokens_in']:.0f} in, {tp['avg_tokens_out']:.0f} out")

        quality = stats.get('quality', {})
        if quality.get('xai_enabled_requests', 0) > 0:
            lines.append("")
            lines.append("── Quality (XAI) ──")
            lines.append(f"  Confidence:  {quality['avg_confidence']:.4f} avg"
                         f"  [{quality['min_confidence']:.4f} — {quality['max_confidence']:.4f}]")
            lines.append(f"  Entropy:     {quality['avg_entropy']:.4f} avg")

            hr = quality['hallucination_rate_high']
            risk_emoji = "🟢" if hr == 0 else ("🟡" if hr < 0.2 else "🔴")
            lines.append(f"  Halluc.Rate: {risk_emoji} {hr:.1%} HIGH"
                         f"  ({quality['hallucination_rate_medium']:.1%} MEDIUM)")

            rd = quality['risk_distribution']
            lines.append(f"  Distribution: 🟢{rd.get('LOW',0)}"
                         f" 🟡{rd.get('MEDIUM',0)}"
                         f" 🔴{rd.get('HIGH',0)}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all monitoring stats."""
        with self._lock:
            self._records.clear()
            self._total_requests = 0
            self._total_tokens_in = 0
            self._total_tokens_out = 0
            self._total_time_ms = 0.0
            self._hallucination_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
            self._start_time = time.time()

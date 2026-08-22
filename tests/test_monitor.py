"""
📊 Monitor Module Tests for MegaGemm
=======================================
Unit tests for the inference monitoring features.
Runs without GPU — tests data classes, aggregation, and export.

Usage:
    python tests/test_monitor.py

Author: Gabriel Yogi
"""

import sys
import os
import json
import time
import tempfile

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from megagemm.engine.monitor import InferenceMonitor, RequestRecord


def _make_record(
    prompt="test",
    text="result",
    tokens_in=10,
    tokens_out=5,
    total_ms=100.0,
    prefill_ms=20.0,
    confidence=-1.0,
    entropy=-1.0,
    risk="",
) -> RequestRecord:
    """Helper to create test records."""
    return RequestRecord(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        prompt=prompt,
        generated_text=text,
        num_tokens_input=tokens_in,
        num_tokens_output=tokens_out,
        prefill_ms=prefill_ms,
        decode_ms=total_ms - prefill_ms,
        total_ms=total_ms,
        ttft_ms=prefill_ms,
        tokens_per_second=tokens_out / (total_ms / 1000) if total_ms > 0 else 0,
        confidence_score=confidence,
        mean_entropy=entropy,
        hallucination_risk=risk,
        model_name="test-model",
        quantization="fp16",
    )


def test_request_record():
    """Test RequestRecord creation and serialization."""
    rec = _make_record(prompt="Hello", text="World", tokens_in=5, tokens_out=3)

    assert rec.prompt == "Hello"
    assert rec.num_tokens_input == 5
    assert rec.num_tokens_output == 3

    d = rec.to_dict()
    assert "timestamp" in d
    assert d["prompt"] == "Hello"
    assert d["num_tokens_input"] == 5
    assert "confidence_score" not in d  # -1 = not computed, should be omitted

    # With XAI
    rec_xai = _make_record(confidence=0.85, entropy=1.2, risk="LOW")
    d2 = rec_xai.to_dict()
    assert d2["confidence_score"] == 0.85
    assert d2["hallucination_risk"] == "LOW"

    print("  ✅ RequestRecord: OK")


def test_monitor_empty():
    """Test monitor with no requests."""
    mon = InferenceMonitor()
    stats = mon.get_stats()

    assert stats["total_requests"] == 0
    assert "uptime_seconds" in stats

    summary = mon.summary()
    assert "No requests" in summary

    print("  ✅ Monitor (empty): OK")


def test_monitor_basic():
    """Test monitor with several requests."""
    mon = InferenceMonitor()

    # Add some requests
    for i in range(5):
        mon.record(_make_record(
            total_ms=100 + i * 20,
            prefill_ms=20 + i * 5,
            tokens_out=10 + i,
        ))

    stats = mon.get_stats()

    assert stats["throughput"]["total_requests"] == 5
    assert stats["throughput"]["window_requests"] == 5
    assert stats["performance"]["avg_latency_ms"] > 0
    assert stats["performance"]["avg_tps"] > 0
    assert stats["quality"]["xai_enabled_requests"] == 0  # No XAI data

    print("  ✅ Monitor (basic): OK")


def test_monitor_with_xai():
    """Test monitor with XAI quality signals."""
    mon = InferenceMonitor()

    # Mix of confident and uncertain requests
    mon.record(_make_record(confidence=0.9, entropy=0.8, risk="LOW"))
    mon.record(_make_record(confidence=0.7, entropy=1.5, risk="LOW"))
    mon.record(_make_record(confidence=0.3, entropy=3.5, risk="MEDIUM"))
    mon.record(_make_record(confidence=0.1, entropy=5.0, risk="HIGH"))

    stats = mon.get_stats()
    quality = stats["quality"]

    assert quality["xai_enabled_requests"] == 4
    assert 0 < quality["avg_confidence"] < 1
    assert quality["avg_entropy"] > 0
    assert quality["hallucination_rate_high"] == 0.25  # 1/4
    assert quality["risk_distribution"]["LOW"] == 2
    assert quality["risk_distribution"]["MEDIUM"] == 1
    assert quality["risk_distribution"]["HIGH"] == 1

    print("  ✅ Monitor (XAI quality): OK")


def test_monitor_summary():
    """Test human-readable summary."""
    mon = InferenceMonitor()

    for i in range(3):
        mon.record(_make_record(
            confidence=0.8 - i * 0.3,
            entropy=1.0 + i * 1.5,
            risk=["LOW", "MEDIUM", "HIGH"][i],
        ))

    summary = mon.summary()

    # Use ASCII-safe assertions (summary has Unicode emojis)
    assert "Performance" in summary
    assert "Quality" in summary
    assert "Latency" in summary
    assert "Uptime" in summary

    print("  ✅ Monitor summary: OK")


def test_monitor_export():
    """Test JSONL export."""
    mon = InferenceMonitor()

    mon.record(_make_record(prompt="Q1", text="A1"))
    mon.record(_make_record(prompt="Q2", text="A2"))

    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False, mode='w') as f:
        tmp_path = f.name

    try:
        count = mon.export_log(tmp_path)
        assert count == 2

        with open(tmp_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Each line should be valid JSON
        for line in lines:
            d = json.loads(line.strip())
            assert "timestamp" in d
            assert "prompt" in d

        print("  ✅ Monitor JSONL export: OK")
    finally:
        os.unlink(tmp_path)


def test_monitor_rolling_window():
    """Test that rolling window evicts old records."""
    mon = InferenceMonitor(window_size=5)

    # Add 10 records
    for i in range(10):
        mon.record(_make_record(prompt=f"Q{i}"))

    stats = mon.get_stats()

    # Totals should reflect all 10
    assert stats["throughput"]["total_requests"] == 10

    # Window should only have 5
    assert stats["throughput"]["window_requests"] == 5

    # Recent should show latest
    recent = mon.get_recent(3)
    assert len(recent) == 3
    assert recent[-1]["prompt"] == "Q9"

    print("  ✅ Monitor rolling window: OK")


def test_monitor_reset():
    """Test monitor reset."""
    mon = InferenceMonitor()

    mon.record(_make_record())
    mon.record(_make_record())
    assert mon.get_stats()["throughput"]["total_requests"] == 2

    mon.reset()
    assert mon.get_stats()["total_requests"] == 0

    print("  ✅ Monitor reset: OK")


def main():
    """Run all monitor tests."""
    print("=" * 60)
    print("📊 MegaGemm Monitor Module Tests")
    print("=" * 60)

    tests = [
        ("RequestRecord",       test_request_record),
        ("Monitor (empty)",     test_monitor_empty),
        ("Monitor (basic)",     test_monitor_basic),
        ("Monitor (XAI)",       test_monitor_with_xai),
        ("Monitor summary",     test_monitor_summary),
        ("JSONL export",        test_monitor_export),
        ("Rolling window",      test_monitor_rolling_window),
        ("Monitor reset",       test_monitor_reset),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print(f"Results: {passed}/{passed + failed} passed")

    if failed > 0:
        print(f"⚠️  {failed} test(s) failed!")
        return 1
    else:
        print("✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

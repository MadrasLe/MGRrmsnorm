"""
🧪 XAI Module Tests for MegaGemm
==================================
Unit tests for the XAI interpretability features.
Runs without GPU — tests data classes, export, and confidence scoring.

Usage:
    python -m pytest tests/test_xai.py -v
    python tests/test_xai.py

Author: Gabriel Yogi
"""

import sys
import os
import json
import math
import tempfile

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from megagemm.engine.xai import (
    TokenPrediction,
    GenerationStep,
    XAIReport,
    compute_confidence,
    classify_hallucination_risk,
    ENTROPY_LOW,
    ENTROPY_HIGH,
    _prob_bar,
    _select_representative_layers,
)


def test_token_prediction():
    """Test TokenPrediction dataclass."""
    tp = TokenPrediction(token_id=42, token_str="hello", probability=0.85)

    assert tp.token_id == 42
    assert tp.token_str == "hello"
    assert tp.probability == 0.85

    d = tp.to_dict()
    assert d["token_id"] == 42
    assert d["token"] == "hello"
    assert d["probability"] == 0.85

    print("  ✅ TokenPrediction: OK")


def test_generation_step():
    """Test GenerationStep: top-K ordering and structure."""
    chosen = TokenPrediction(token_id=10, token_str="world", probability=0.6)
    top_k = [
        TokenPrediction(token_id=10, token_str="world", probability=0.6),
        TokenPrediction(token_id=20, token_str="earth", probability=0.25),
        TokenPrediction(token_id=30, token_str="planet", probability=0.1),
    ]

    step = GenerationStep(position=0, chosen=chosen, top_k=top_k)

    assert step.position == 0
    assert step.chosen.token_id == 10
    assert len(step.top_k) == 3
    assert step.logit_lens is None

    # Verify probabilities sum <= 1.0
    total = sum(t.probability for t in step.top_k)
    assert total <= 1.0 + 1e-6, f"Probabilities sum to {total}"

    # Test to_dict
    d = step.to_dict()
    assert "chosen" in d
    assert "top_k" in d
    assert len(d["top_k"]) == 3
    assert "logit_lens" not in d  # None should be omitted

    print("  ✅ GenerationStep: OK")


def test_generation_step_with_logit_lens():
    """Test GenerationStep with Logit Lens data."""
    chosen = TokenPrediction(token_id=10, token_str="world", probability=0.6)
    top_k = [chosen]

    lens_data = {
        0: [TokenPrediction(5, "the", 0.3), TokenPrediction(6, "a", 0.2)],
        15: [TokenPrediction(10, "world", 0.4), TokenPrediction(7, "of", 0.1)],
        31: [TokenPrediction(10, "world", 0.8), TokenPrediction(20, "earth", 0.1)],
    }

    step = GenerationStep(position=0, chosen=chosen, top_k=top_k, logit_lens=lens_data)

    assert step.logit_lens is not None
    assert len(step.logit_lens) == 3
    assert 0 in step.logit_lens
    assert 15 in step.logit_lens
    assert 31 in step.logit_lens

    d = step.to_dict()
    assert "logit_lens" in d
    assert "0" in d["logit_lens"]  # keys are stringified
    assert "15" in d["logit_lens"]

    print("  ✅ GenerationStep with Logit Lens: OK")


def test_confidence_score():
    """Test confidence score computation (geometric mean)."""
    # All high probabilities
    probs = [0.9, 0.8, 0.85, 0.95]
    score = compute_confidence(probs)
    assert 0.0 < score < 1.0
    assert score > 0.8, f"Expected high confidence, got {score}"

    # Manual geometric mean check
    expected = math.exp(sum(math.log(p) for p in probs) / len(probs))
    assert abs(score - expected) < 1e-6

    # All low probabilities
    low_probs = [0.1, 0.05, 0.08]
    low_score = compute_confidence(low_probs)
    assert low_score < 0.1, f"Expected low confidence, got {low_score}"

    # Empty list
    assert compute_confidence([]) == 0.0

    # Zero probability
    assert compute_confidence([0.5, 0.0, 0.3]) == 0.0

    # Single item
    assert abs(compute_confidence([0.7]) - 0.7) < 1e-6

    print("  ✅ Confidence Score: OK")


def test_xai_report():
    """Test XAIReport creation and summary."""
    steps = [
        GenerationStep(
            position=0,
            chosen=TokenPrediction(10, "Hello", 0.9),
            top_k=[
                TokenPrediction(10, "Hello", 0.9),
                TokenPrediction(20, "Hi", 0.05),
            ],
        ),
        GenerationStep(
            position=1,
            chosen=TokenPrediction(30, "world", 0.85),
            top_k=[
                TokenPrediction(30, "world", 0.85),
                TokenPrediction(40, "there", 0.1),
            ],
        ),
    ]

    report = XAIReport(
        prompt="Say hello",
        generated_text="Hello world",
        steps=steps,
        confidence_score=compute_confidence([0.9, 0.85]),
        model_name="test-model",
        num_layers=32,
    )

    assert report.prompt == "Say hello"
    assert report.generated_text == "Hello world"
    assert len(report.steps) == 2
    assert report.confidence_score > 0.85
    assert report.model_name == "test-model"
    assert report.timestamp != ""

    # Test summary
    summary = report.summary()
    assert "MegaGemm XAI Report" in summary
    assert "Say hello" in summary
    assert "Hello world" in summary
    assert "Hello" in summary
    assert "Confidence" in summary

    print("  ✅ XAIReport: OK")


def test_json_export():
    """Test JSON export and reload."""
    steps = [
        GenerationStep(
            position=0,
            chosen=TokenPrediction(1, "A", 0.7),
            top_k=[
                TokenPrediction(1, "A", 0.7),
                TokenPrediction(2, "B", 0.2),
            ],
        ),
    ]

    report = XAIReport(
        prompt="Test",
        generated_text="A",
        steps=steps,
        confidence_score=0.7,
        model_name="test-model",
        num_layers=4,
    )

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        tmp_path = f.name

    try:
        report.to_json(tmp_path)

        with open(tmp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data["model"] == "test-model"
        assert data["prompt"] == "Test"
        assert data["generated_text"] == "A"
        assert data["confidence_score"] == 0.7
        assert data["num_steps"] == 1
        assert data["num_layers"] == 4
        assert len(data["steps"]) == 1

        step = data["steps"][0]
        assert step["position"] == 0
        assert step["chosen"]["token"] == "A"
        assert step["chosen"]["probability"] == 0.7
        assert len(step["top_k"]) == 2

        print("  ✅ JSON Export: OK")
    finally:
        os.unlink(tmp_path)


def test_txt_export():
    """Test TXT export readability."""
    steps = [
        GenerationStep(
            position=0,
            chosen=TokenPrediction(1, "Test", 0.95),
            top_k=[TokenPrediction(1, "Test", 0.95)],
        ),
    ]

    report = XAIReport(
        prompt="Hello",
        generated_text="Test",
        steps=steps,
        confidence_score=0.95,
    )

    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
        tmp_path = f.name

    try:
        report.to_txt(tmp_path)

        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "MegaGemm XAI Report" in content
        assert "Hello" in content
        assert "Test" in content
        assert "0.95" in content

        print("  ✅ TXT Export: OK")
    finally:
        os.unlink(tmp_path)


def test_prob_bar():
    """Test probability visualization bar."""
    bar_full = _prob_bar(1.0, width=10)
    assert bar_full == "██████████"

    bar_empty = _prob_bar(0.0, width=10)
    assert bar_empty == "░░░░░░░░░░"

    bar_half = _prob_bar(0.5, width=10)
    assert "█" in bar_half
    assert "░" in bar_half

    print("  ✅ Probability Bar: OK")


def test_representative_layers():
    """Test layer selection for Logit Lens display."""
    # Few layers — show all
    result = _select_representative_layers([0, 1, 2, 3])
    assert result == [0, 1, 2, 3]

    # Many layers — show representative subset
    layers = list(range(32))
    result = _select_representative_layers(layers, max_show=5)
    assert len(result) <= 5
    assert result[0] == 0     # first
    assert result[-1] == 31   # last

    print("  ✅ Representative Layers: OK")


def test_logit_lens_in_report():
    """Test XAI report with Logit Lens data."""
    lens_data = {
        0: [TokenPrediction(5, "the", 0.1)],
        16: [TokenPrediction(10, "cat", 0.4)],
        31: [TokenPrediction(10, "cat", 0.9)],
    }

    steps = [
        GenerationStep(
            position=0,
            chosen=TokenPrediction(10, "cat", 0.8),
            top_k=[TokenPrediction(10, "cat", 0.8)],
            logit_lens=lens_data,
        ),
    ]

    report = XAIReport(
        prompt="The animal is a",
        generated_text="cat",
        steps=steps,
        confidence_score=0.8,
        num_layers=32,
    )

    d = report.to_dict()
    assert d["has_logit_lens"] is True
    assert "logit_lens" in d["steps"][0]

    summary = report.summary()
    assert "Logit Lens" in summary
    assert "Layer" in summary

    print("  ✅ Logit Lens in Report: OK")


def test_entropy_in_step():
    """Test GenerationStep with entropy field."""
    step_low = GenerationStep(
        position=0,
        chosen=TokenPrediction(10, "Paris", 0.99),
        top_k=[TokenPrediction(10, "Paris", 0.99)],
        entropy=0.5,  # Low entropy = confident
    )

    step_high = GenerationStep(
        position=1,
        chosen=TokenPrediction(20, "maybe", 0.05),
        top_k=[TokenPrediction(20, "maybe", 0.05)],
        entropy=5.0,  # High entropy = uncertain
    )

    assert step_low.entropy == 0.5
    assert step_high.entropy == 5.0

    # Test in to_dict
    d = step_low.to_dict()
    assert "entropy" in d
    assert d["entropy"] == 0.5

    print("  ✅ Entropy in Step: OK")


def test_hallucination_risk():
    """Test hallucination risk classification."""
    # LOW: low entropy + high confidence
    assert classify_hallucination_risk(1.0, 0.9) == "LOW"
    assert classify_hallucination_risk(0.5, 0.8) == "LOW"

    # MEDIUM: moderate entropy or moderate confidence
    assert classify_hallucination_risk(3.0, 0.6) == "MEDIUM"
    assert classify_hallucination_risk(1.5, 0.3) == "MEDIUM"

    # HIGH: high entropy or very low confidence
    assert classify_hallucination_risk(5.0, 0.3) == "HIGH"
    assert classify_hallucination_risk(1.0, 0.1) == "HIGH"

    # Test auto-computed on XAIReport
    # Confident generation
    steps_confident = [
        GenerationStep(0, TokenPrediction(1, "A", 0.9), [], entropy=0.5),
        GenerationStep(1, TokenPrediction(2, "B", 0.85), [], entropy=0.7),
    ]
    report = XAIReport(
        prompt="test", generated_text="AB",
        steps=steps_confident, confidence_score=0.87,
    )
    assert report.hallucination_risk == "LOW"
    assert report.high_entropy_steps == 0
    assert report.mean_entropy < ENTROPY_LOW

    # Uncertain generation
    steps_uncertain = [
        GenerationStep(0, TokenPrediction(1, "X", 0.05), [], entropy=5.5),
        GenerationStep(1, TokenPrediction(2, "Y", 0.03), [], entropy=6.0),
    ]
    report2 = XAIReport(
        prompt="test", generated_text="XY",
        steps=steps_uncertain, confidence_score=0.04,
    )
    assert report2.hallucination_risk == "HIGH"
    assert report2.high_entropy_steps == 2
    assert report2.mean_entropy > ENTROPY_HIGH

    # Summary should show risk
    summary = report2.summary()
    assert "HIGH" in summary
    assert "high entropy" in summary

    print("  ✅ Hallucination Risk: OK")


def test_entropy_in_json():
    """Test entropy fields in JSON export."""
    steps = [
        GenerationStep(0, TokenPrediction(1, "A", 0.7), [], entropy=1.5),
    ]
    report = XAIReport(
        prompt="test", generated_text="A",
        steps=steps, confidence_score=0.7,
    )

    d = report.to_dict()
    assert "mean_entropy" in d
    assert "hallucination_risk" in d
    assert "high_entropy_steps" in d
    assert d["steps"][0]["entropy"] == 1.5
    assert d["hallucination_risk"] == "LOW"  # entropy < 2.0 and confidence > 0.5

    print("  ✅ Entropy in JSON: OK")


def main():
    """Run all XAI tests."""
    print("=" * 60)
    print("🧪 MegaGemm XAI Module Tests")
    print("=" * 60)

    tests = [
        ("TokenPrediction",         test_token_prediction),
        ("GenerationStep",          test_generation_step),
        ("GenerationStep + Lens",   test_generation_step_with_logit_lens),
        ("Confidence Score",        test_confidence_score),
        ("XAIReport",               test_xai_report),
        ("JSON Export",             test_json_export),
        ("TXT Export",              test_txt_export),
        ("Probability Bar",         test_prob_bar),
        ("Representative Layers",   test_representative_layers),
        ("Logit Lens in Report",    test_logit_lens_in_report),
        ("Entropy in Step",          test_entropy_in_step),
        ("Hallucination Risk",       test_hallucination_risk),
        ("Entropy in JSON",          test_entropy_in_json),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: FAILED — {e}")
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

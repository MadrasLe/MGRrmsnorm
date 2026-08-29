"""Full-model A/B gate for the Gemma 4 E2B PLE GELU-tanh fusion.

The two cases run in isolated Python processes because the experimental flag
is resolved when the model module is imported.  The gate uses the production
P2048/B8/BF16 L4 path, verifies generated-token digests, audits actual kernel
hits, and bases promotion on decode wall time rather than a standalone kernel
microbenchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "benchmarks" / "benchmark_inference_matrix.py"

CASES: tuple[tuple[str, str], ...] = (
    ("baseline", "0"),
    ("conditioned_gelu", "1"),
)


def _cuda_latency_samples(fn, *, warmup: int = 20, iterations: int = 100) -> list[float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
    return samples


def _kernel_preflight() -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from megagemm.kernels.swiglu import conditioned_gelu_tanh_forward

    if not torch.cuda.is_available():
        return {"correct": False, "error": "CUDA is unavailable"}
    gpu = torch.cuda.get_device_name(0)
    if "L4" not in gpu.upper():
        return {"correct": False, "error": f"expected NVIDIA L4, got {gpu}"}

    torch.manual_seed(20260829)
    gate = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    per_layer = torch.randn(
        8,
        1,
        35,
        256,
        device="cuda",
        dtype=torch.bfloat16,
    )
    condition = per_layer[:, 0, 17, :]
    reference = F.gelu(gate, approximate="tanh")
    reference.mul_(condition)

    candidate = conditioned_gelu_tanh_forward(gate, condition)
    alias = gate.clone()
    conditioned_gelu_tanh_forward(alias, condition, out=alias)
    repeat = conditioned_gelu_tanh_forward(gate, condition)
    torch.cuda.synchronize()

    diff = (candidate.float() - reference.float()).abs()
    max_abs_error = float(diff.max().item())
    mean_abs_error = float(diff.mean().item())
    cosine = float(
        F.cosine_similarity(
            candidate.float().reshape(1, -1),
            reference.float().reshape(1, -1),
        ).item()
    )
    finite = bool(torch.isfinite(candidate).all().item())
    alias_exact = bool(torch.equal(alias, candidate))
    repeat_exact = bool(torch.equal(repeat, candidate))
    correct = bool(
        finite
        and alias_exact
        and repeat_exact
        and cosine >= 0.9999
        and max_abs_error <= 0.125
    )

    candidate_out = torch.empty_like(gate)

    def baseline_fn():
        value = F.gelu(gate, approximate="tanh")
        value.mul_(condition)
        return value

    def candidate_fn():
        return conditioned_gelu_tanh_forward(
            gate,
            condition,
            out=candidate_out,
        )

    baseline_samples = _cuda_latency_samples(baseline_fn)
    candidate_samples = _cuda_latency_samples(candidate_fn)
    baseline_us = statistics.median(baseline_samples)
    candidate_us = statistics.median(candidate_samples)
    return {
        "correct": correct,
        "gpu": gpu,
        "shape": "B8/L35/PLE256/BF16",
        "condition_stride": list(condition.stride()),
        "finite": finite,
        "alias_exact": alias_exact,
        "repeat_exact": repeat_exact,
        "cosine": cosine,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "baseline_samples_us": baseline_samples,
        "candidate_samples_us": candidate_samples,
        "baseline_median_us": baseline_us,
        "candidate_median_us": candidate_us,
        "micro_speedup": baseline_us / candidate_us if candidate_us > 0.0 else 0.0,
    }


def _case_environment(enabled: str) -> dict[str, str]:
    env = os.environ.copy()
    root_text = str(ROOT)
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        root_text
        if not current_pythonpath
        else root_text + os.pathsep + current_pythonpath
    )
    env.update(
        {
            "MEGAGEMM_FLAT_DECODE": "1",
            "MEGAGEMM_BENCHMARK_TOKEN_DIGEST": "1",
            "MEGAGEMM_GEMMA4_PLE_CONDITIONED_GELU_DECODE": enabled,
            "MEGAGEMM_DECODE_CUDA_GRAPHS": "0",
            "MEGAGEMM_DECODE_PREFER_STEP": "0",
            "MEGAGEMM_REUSE_REQUEST_SCHEDULER": "0",
            "MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE": "1",
            "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_PREFILL": "1",
            "MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE": "0",
            "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE": "0",
            "MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE": "0",
            "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE": "0",
            "MEGAGEMM_PAGED_DECODE_SPLITS": "1",
            "MEGAGEMM_PAGED_DECODE_GQA2": "1",
            "MEGAGEMM_PAGED_DECODE_WARPS_H256": "2",
        }
    )
    return env


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _successful_long_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    return [
        row
        for row in rows
        if bool(row.get("ok"))
        and int(row.get("batch_size", 0) or 0) == 8
        and int(row.get("prompt_tokens_requested_per_request", 0) or 0) == 2048
    ]


def _decode_tps(row: dict[str, Any]) -> float:
    scheduler = row.get("scheduler_stats")
    if not isinstance(scheduler, dict):
        return 0.0
    decode_ms = float(scheduler.get("decode_time_ms") or 0.0)
    if decode_ms <= 0.0:
        return 0.0
    return float(row.get("generated_tokens") or 0.0) / (decode_ms / 1000.0)


def _audit(rows: list[dict[str, Any]], *, expected_enabled: bool) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["no successful P2048/B8 rows"]
    for index, row in enumerate(rows):
        stats = row.get("decode_runtime_stats")
        if not isinstance(stats, dict):
            errors.append(f"row {index}: decode_runtime_stats missing")
            continue
        enabled = bool(stats.get("gemma4_ple_conditioned_gelu_decode_enabled"))
        hits = int(stats.get("gemma4_ple_conditioned_gelu_decode_hits") or 0)
        disabled = bool(stats.get("gemma4_ple_conditioned_gelu_runtime_disabled"))
        failure = str(stats.get("gemma4_ple_conditioned_gelu_first_failure") or "")
        if enabled != expected_enabled:
            errors.append(
                f"row {index}: enabled={enabled}, expected {expected_enabled}"
            )
        if expected_enabled and hits <= 0:
            errors.append(f"row {index}: candidate recorded no kernel hits")
        if not expected_enabled and hits != 0:
            errors.append(f"row {index}: baseline unexpectedly recorded {hits} hits")
        if disabled:
            errors.append(f"row {index}: runtime path disabled itself")
        if failure:
            errors.append(f"row {index}: {failure}")
    return errors


def _case_result(name: str, case_dir: Path, *, expected_enabled: bool) -> dict[str, Any]:
    raw_files = sorted(case_dir.glob("*.jsonl"))
    if len(raw_files) != 1:
        return {
            "name": name,
            "status": "failed",
            "errors": [f"expected one JSONL artifact, found {len(raw_files)}"],
        }
    rows = _successful_long_rows(raw_files[0])
    decode_samples = [_decode_tps(row) for row in rows]
    output_samples = [float(row.get("output_tps") or 0.0) for row in rows]
    digests = [str(row.get("generated_token_digest") or "") for row in rows]
    errors = _audit(rows, expected_enabled=expected_enabled)
    if any(value <= 0.0 for value in decode_samples):
        errors.append("one or more decode wall-throughput samples are missing")
    if any(not digest for digest in digests):
        errors.append("one or more generated-token digests are missing")
    return {
        "name": name,
        "status": "ok" if not errors else "failed",
        "raw_jsonl": str(raw_files[0]),
        "decode_samples_tps": decode_samples,
        "output_samples_tps": output_samples,
        "median_decode_tps": statistics.median(decode_samples) if decode_samples else 0.0,
        "median_output_tps": statistics.median(output_samples) if output_samples else 0.0,
        "spread_ratio": (
            max(decode_samples) / min(decode_samples)
            if decode_samples and min(decode_samples) > 0.0
            else float("inf")
        ),
        "digests": digests,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gemma 4 E2B/L4 PLE conditioned-GELU full-model A/B gate"
    )
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--minimum-speedup", type=float, default=1.005)
    parser.add_argument("--maximum-spread", type=float, default=1.04)
    args = parser.parse_args()

    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3")
    if args.max_new_tokens < 32:
        raise ValueError("--max-new-tokens must be at least 32")
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n=== kernel preflight: B8/L35/PLE256/BF16, strided condition ===")
    preflight = _kernel_preflight()
    print("PREFLIGHT " + json.dumps(preflight, separators=(",", ":")))
    if not bool(preflight.get("correct")):
        payload = {
            "decision": "INVALID_KERNEL_PREFLIGHT",
            "model": args.model,
            "preflight": preflight,
            "results": [],
            "errors": [str(preflight.get("error") or "kernel correctness failed")],
        }
        decision_path = out_root / "decision.json"
        decision_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("DECISION " + json.dumps(payload, separators=(",", ":")))
        print(f"wrote {decision_path}")
        return 2

    results: list[dict[str, Any]] = []
    for name, enabled in CASES:
        case_dir = out_root / name
        case_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(MATRIX),
            "--backend",
            "megagemm",
            "--model",
            args.model,
            "--hardware-label",
            "1xl4",
            "--batch-sizes",
            "8",
            "--prompt-tokens",
            "2048",
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--repeats",
            str(args.repeats),
            "--warmup",
            str(args.warmup),
            "--out-dir",
            str(case_dir),
            "--run-id",
            f"gemma4_e2b_ple_{name}",
            "--device",
            "cuda",
            "--dtype",
            "bf16",
            "--max-seq-len",
            "2304",
            "--max-batch-size",
            "8",
            "--ignore-eos",
        ]
        print(f"\n=== {name} ===", flush=True)
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=_case_environment(enabled),
            check=False,
        )
        if completed.returncode != 0:
            results.append(
                {
                    "name": name,
                    "status": "failed",
                    "errors": [f"benchmark exited with {completed.returncode}"],
                }
            )
            continue
        results.append(
            _case_result(name, case_dir, expected_enabled=enabled == "1")
        )

    baseline = next((item for item in results if item["name"] == "baseline"), None)
    candidate = next(
        (item for item in results if item["name"] == "conditioned_gelu"), None
    )
    errors = [
        f"{item['name']}: {error}"
        for item in results
        for error in item.get("errors", ())
    ]
    if baseline is None or candidate is None:
        errors.append("baseline or candidate result missing")

    digest_match = False
    speedup = 0.0
    conservative_speedup = 0.0
    output_ratio = 0.0
    stable = False
    if baseline and candidate:
        baseline_digests = set(baseline.get("digests") or ())
        candidate_digests = set(candidate.get("digests") or ())
        digest_match = (
            len(baseline_digests) == 1
            and baseline_digests == candidate_digests
            and "" not in baseline_digests
        )
        baseline_decode = float(baseline.get("median_decode_tps") or 0.0)
        candidate_decode = float(candidate.get("median_decode_tps") or 0.0)
        speedup = candidate_decode / baseline_decode if baseline_decode > 0.0 else 0.0
        baseline_samples = baseline.get("decode_samples_tps") or []
        candidate_samples = candidate.get("decode_samples_tps") or []
        conservative_speedup = (
            min(candidate_samples) / max(baseline_samples)
            if baseline_samples and candidate_samples and max(baseline_samples) > 0.0
            else 0.0
        )
        baseline_output = float(baseline.get("median_output_tps") or 0.0)
        candidate_output = float(candidate.get("median_output_tps") or 0.0)
        output_ratio = candidate_output / baseline_output if baseline_output > 0.0 else 0.0
        stable = (
            float(baseline.get("spread_ratio") or float("inf"))
            <= args.maximum_spread
            and float(candidate.get("spread_ratio") or float("inf"))
            <= args.maximum_spread
        )

    if not digest_match:
        errors.append("generated-token digests differ between baseline and candidate")
    if not stable:
        errors.append("decode samples exceeded the allowed spread")

    all_valid = not errors and all(item.get("status") == "ok" for item in results)
    if (
        all_valid
        and speedup >= args.minimum_speedup
        and conservative_speedup > 1.0
        and output_ratio >= 0.9975
    ):
        decision = "PROMOTE_CONDITIONED_GELU"
    elif all_valid:
        decision = "KEEP_BASELINE"
    else:
        decision = "INVALID_GATE"

    payload = {
        "decision": decision,
        "model": args.model,
        "shape": "L4/BF16/P2048/B8",
        "speedup": speedup,
        "conservative_speedup": conservative_speedup,
        "output_ratio": output_ratio,
        "digest_match": digest_match,
        "stable": stable,
        "minimum_speedup": args.minimum_speedup,
        "maximum_spread": args.maximum_spread,
        "preflight": preflight,
        "results": results,
        "errors": errors,
    }
    decision_path = out_root / "decision.json"
    decision_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 96)
    print("GEMMA 4 E2B / L4 / BF16 / P2048 / B8 — PLE CONDITIONED GELU A/B")
    print("=" * 96)
    for item in results:
        print(
            f"{item['name']:<18} decode={float(item.get('median_decode_tps') or 0.0):8.2f} "
            f"output={float(item.get('median_output_tps') or 0.0):8.2f} "
            f"spread={float(item.get('spread_ratio') or 0.0):.4f} "
            f"samples={item.get('decode_samples_tps') or []}"
        )
    print("DECISION " + json.dumps(payload, separators=(",", ":")))
    print(f"wrote {decision_path}")
    return 0 if decision != "INVALID_GATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Tune the fused RMSNorm + LM-head argmax for Gemma 4 E2B on L4.

Every launch configuration runs in a fresh child process because the Triton
configuration is selected from environment variables at module import time.
The tensors use E2B's exact decode shape, while avoiding a model download.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MICROBENCH = ROOT / "benchmarks" / "run_lm_head_argmax_microbench.py"


@dataclass(frozen=True)
class LaunchConfig:
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int = 2
    baseline: bool = False

    @property
    def name(self) -> str:
        return (
            f"bn{self.block_n}_bk{self.block_k}_"
            f"w{self.num_warps}_s{self.num_stages}"
        )


CURRENT = LaunchConfig(256, 128, 4, 2, baseline=True)
CONFIGS = (
    CURRENT,
    LaunchConfig(64, 64, 4),
    LaunchConfig(64, 128, 4),
    LaunchConfig(64, 128, 8),
    LaunchConfig(128, 64, 4),
    LaunchConfig(128, 64, 8),
    LaunchConfig(128, 128, 4),
    LaunchConfig(128, 128, 8),
    LaunchConfig(256, 64, 4),
    LaunchConfig(256, 64, 8),
    LaunchConfig(256, 128, 8),
    LaunchConfig(256, 128, 4, 1),
    LaunchConfig(256, 128, 4, 3),
    LaunchConfig(512, 64, 8),
    CURRENT,
)


def _valid(row: dict[str, Any], maximum_spread: float) -> bool:
    samples = row.get("samples_us") or []
    return bool(
        row.get("error") is None
        and row.get("tokens_equal") is True
        and row.get("median_us") is not None
        and samples
        and float(row.get("spread_ratio") or float("inf")) <= maximum_spread
    )


def decide_lm_head_sweep(
    rows: list[dict[str, Any]],
    *,
    minimum_speedup: float,
    maximum_spread: float,
) -> dict[str, Any]:
    baselines = [row for row in rows if row.get("baseline")]
    valid_baselines = [row for row in baselines if _valid(row, maximum_spread)]
    baseline_samples = [
        float(sample)
        for row in valid_baselines
        for sample in row.get("samples_us") or []
    ]
    baseline_us = (
        float(statistics.median(baseline_samples)) if baseline_samples else None
    )
    baseline_stable = bool(
        len(valid_baselines) == 2
        and baseline_samples
        and max(baseline_samples) / min(baseline_samples) <= maximum_spread
    )

    candidates = [
        row
        for row in rows
        if not row.get("baseline") and _valid(row, maximum_spread)
    ]
    candidates.sort(key=lambda row: float(row["median_us"]))
    winner = candidates[0] if candidates else None
    winner_us = float(winner["median_us"]) if winner is not None else None
    speedup = (
        baseline_us / winner_us
        if baseline_us is not None and winner_us is not None and winner_us > 0.0
        else None
    )
    winner_samples = (
        [float(value) for value in winner.get("samples_us") or []]
        if winner is not None
        else []
    )
    sample_ranges_separate = bool(
        baseline_samples
        and winner_samples
        and max(winner_samples) < min(baseline_samples)
    )
    apply_change = bool(
        baseline_stable
        and speedup is not None
        and speedup >= minimum_speedup
        and sample_ranges_separate
    )
    return {
        "decision": (
            f"TEST_FULL_MODEL_{winner['name'].upper()}"
            if apply_change and winner is not None
            else "KEEP_CURRENT_LM_HEAD_CONFIG"
        ),
        "apply_change": apply_change,
        "baseline": CURRENT.name,
        "baseline_us": baseline_us,
        "baseline_stable": baseline_stable,
        "winner": winner.get("name") if winner is not None else None,
        "winner_us": winner_us,
        "speedup": speedup,
        "sample_ranges_separate": sample_ranges_separate,
        "minimum_speedup": minimum_speedup,
        "maximum_spread": maximum_spread,
        "estimated_saved_ms_per_128_decode_steps": (
            (baseline_us - winner_us) * 128.0 / 1000.0
            if baseline_us is not None and winner_us is not None
            else None
        ),
    }


def _run_case(
    config: LaunchConfig,
    *,
    case_index: int,
    args: argparse.Namespace,
    temp_root: Path,
) -> dict[str, Any]:
    label = config.name
    if config.baseline:
        label += "_baseline" if case_index == 0 else "_baseline_recheck"
    output = temp_root / f"{case_index:02d}_{label}.json"
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(ROOT)
        if not current_pythonpath
        else str(ROOT) + os.pathsep + current_pythonpath
    )
    env.update(
        {
            "MEGAGEMM_FUSED_LM_HEAD_ARGMAX_BLOCK_N": str(config.block_n),
            "MEGAGEMM_FUSED_LM_HEAD_ARGMAX_BLOCK_K": str(config.block_k),
            "MEGAGEMM_FUSED_LM_HEAD_ARGMAX_NUM_WARPS": str(config.num_warps),
            "MEGAGEMM_FUSED_LM_HEAD_ARGMAX_NUM_STAGES": str(config.num_stages),
            "MEGAGEMM_FUSED_LM_HEAD_ARGMAX_TRITON_REDUCE": "1",
        }
    )
    command = [
        sys.executable,
        str(MICROBENCH),
        "--rows", str(args.rows),
        "--hidden-size", str(args.hidden_size),
        "--vocab-size", str(args.vocab_size),
        "--dtype", args.dtype,
        "--warmup", str(args.warmups),
        "--iterations", str(args.iterations),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--eager",
        "--cases", "fused-rmsnorm",
        "--out-json", str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    row: dict[str, Any] = {
        "name": label,
        "baseline": bool(config.baseline),
        "config": asdict(config),
        "returncode": int(completed.returncode),
        "median_us": None,
        "samples_us": [],
        "spread_ratio": None,
        "tokens_equal": False,
        "error": None,
    }
    if completed.returncode != 0 or not output.exists():
        detail = (completed.stderr or completed.stdout).strip()
        row["error"] = detail[-2000:] or f"child exited {completed.returncode}"
        return row

    payload = json.loads(output.read_text(encoding="utf-8"))
    results = payload.get("results") or []
    selected = next(
        (
            item
            for item in results
            if item.get("case") == "fused_rmsnorm_lm_head_argmax"
        ),
        None,
    )
    if selected is None:
        row["error"] = "fused RMSNorm LM-head result missing"
        return row
    samples = [float(value) for value in selected.get("samples_us") or []]
    row.update(
        {
            "median_us": selected.get("median_us"),
            "samples_us": samples,
            "spread_ratio": max(samples) / min(samples) if samples else None,
            "tokens_equal": bool(selected.get("tokens_equal", False)),
            "error": selected.get("error"),
        }
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1536)
    parser.add_argument("--vocab-size", type=int, default=262144)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--minimum-speedup", type=float, default=1.03)
    parser.add_argument("--maximum-spread", type=float, default=1.05)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if (args.rows, args.hidden_size, args.vocab_size, args.dtype) != (
        8,
        1536,
        262144,
        "bf16",
    ):
        raise SystemExit(
            "this promotion gate requires Gemma 4 E2B decode shape "
            "rows=8 hidden=1536 vocab=262144 BF16"
        )

    print("Gemma 4 E2B/L4 fused RMSNorm LM-head sweep")
    print("  shape: rows=8 hidden=1536 vocab=262144 BF16")
    print("  model_download: disabled")
    print("  competing_engine_install: disabled")

    with tempfile.TemporaryDirectory(prefix="gemma4_e2b_lm_head_") as temp:
        temp_root = Path(temp)
        rows: list[dict[str, Any]] = []
        for index, config in enumerate(CONFIGS):
            print(f"\n=== {index + 1}/{len(CONFIGS)} {config.name} ===", flush=True)
            row = _run_case(
                config,
                case_index=index,
                args=args,
                temp_root=temp_root,
            )
            rows.append(row)
            print("CASE", json.dumps(row, sort_keys=True), flush=True)

    decision = decide_lm_head_sweep(
        rows,
        minimum_speedup=args.minimum_speedup,
        maximum_spread=args.maximum_spread,
    )
    payload = {
        "shape": {
            "rows": args.rows,
            "hidden_size": args.hidden_size,
            "vocab_size": args.vocab_size,
            "dtype": args.dtype,
        },
        "rows": rows,
        **decision,
    }

    print("\nRANKING")
    ranked = [row for row in rows if row.get("median_us") is not None]
    ranked.sort(key=lambda row: float(row["median_us"]))
    for index, row in enumerate(ranked, 1):
        suffix = " [current]" if row.get("baseline") else ""
        print(
            f"{index:02d}. {row['name']}: {float(row['median_us']):.3f} us"
            f" spread={float(row['spread_ratio']):.4f}{suffix}"
        )
    print("\nDECISION", json.dumps(decision, sort_keys=True))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {output}")
    return 0 if any(row.get("median_us") is not None for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())

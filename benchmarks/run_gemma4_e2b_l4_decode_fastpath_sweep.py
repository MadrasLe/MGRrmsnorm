"""Sweep the production Gemma 4 E2B/L4 decode MLP paths.

This gate intentionally uses the long-context publication shape.  Short-context
measurements can promote an MLP kernel that loses once paged attention is also
streaming the 2048-token KV working set.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "benchmarks" / "benchmark_inference_matrix.py"


CASES: tuple[tuple[str, dict[str, str | None]], ...] = (
    (
        "cublas_baseline",
        {
            "MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE": "0",
            "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE": "0",
            "MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE": "0",
            "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE": "0",
        },
    ),
    (
        "fused_gateup",
        {
            "MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE": "1",
            "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE": "1",
            "MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE": "0",
            "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE": "0",
        },
    ),
    (
        "deepfusion",
        {
            "MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE": "0",
            "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE": "0",
            "MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE": "1",
            "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE": "1",
        },
    ),
    (
        "current_policy",
        {
            "MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE": None,
            "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE": None,
            "MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE": None,
            "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE": None,
        },
    ),
)


def _case_environment(overrides: dict[str, str | None]) -> dict[str, str]:
    env = os.environ.copy()
    root_text = str(ROOT)
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        root_text if not current_pythonpath else root_text + os.pathsep + current_pythonpath
    )
    env.update(
        {
            "MEGAGEMM_FLAT_DECODE": "1",
            "MEGAGEMM_DISABLE_CUDA_RMSNORM": "1",
            "MEGAGEMM_DECODE_PREFER_STEP": "0",
            "MEGAGEMM_DECODE_CUDA_GRAPHS": "0",
            "MEGAGEMM_REUSE_REQUEST_SCHEDULER": "0",
            "MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE": "1",
            "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_PREFILL": "1",
        }
    )
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _summary_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    successful = [
        row
        for row in rows
        if str(row.get("status", "ok")).lower() == "ok"
        and int(row.get("ok_samples", 0) or 0) > 0
        and int(row.get("batch_size", 0) or 0) == 8
        and int(row.get("prompt_tokens_requested_per_request", 0) or 0) == 2048
    ]
    if len(successful) != 1:
        raise RuntimeError(
            f"expected one successful P2048/B8 summary row in {path}, got {len(successful)}"
        )
    return successful[0]


def _median(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        return 0.0
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gemma 4 E2B/L4 long-context decode fast-path sweep"
    )
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--out-root", default="/tmp/gemma4_e2b_l4_decode_fastpath")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    if args.max_new_tokens < 16:
        raise ValueError("--max-new-tokens must be at least 16 for a stable decode gate")
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for name, overrides in CASES:
        case_dir = out_root / name
        case_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"gemma4_e2b_l4_decode_{name}"
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
            run_id,
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
            env=_case_environment(overrides),
            check=False,
        )
        if completed.returncode != 0:
            results.append(
                {"name": name, "status": "failed", "returncode": completed.returncode}
            )
            continue
        summaries = sorted(case_dir.glob("*_summary.json"))
        if len(summaries) != 1:
            results.append(
                {
                    "name": name,
                    "status": "failed",
                    "error": f"expected one summary, found {len(summaries)}",
                }
            )
            continue
        row = _summary_row(summaries[0])
        results.append(
            {
                "name": name,
                "status": "ok",
                "output_tps": _median(row, "median_output_tps"),
                "decode_tps": _median(row, "median_decode_wall_tps"),
                "decode_ms": _median(row, "median_decode_time_ms"),
                "prefill_ms": _median(row, "median_prefill_time_ms"),
                "summary": str(summaries[0]),
                "overrides": overrides,
            }
        )

    successful = [item for item in results if item.get("status") == "ok"]
    if not successful:
        print(json.dumps({"decision": "NO_VALID_CASE", "results": results}, indent=2))
        return 1
    successful.sort(key=lambda item: float(item["decode_tps"]), reverse=True)
    winner = successful[0]
    current = next(
        (item for item in successful if item["name"] == "current_policy"), None
    )
    current_decode_tps = float(current["decode_tps"]) if current else 0.0

    print("\n" + "=" * 88)
    print("GEMMA 4 E2B / L4 / BF16 — LONG-CONTEXT DECODE FAST-PATH SWEEP")
    print("=" * 88)
    print(f"{'path':<20} {'decode tok/s':>14} {'decode ms':>12} {'output tok/s':>14} {'prefill ms':>12}")
    for item in successful:
        print(
            f"{item['name']:<20} {item['decode_tps']:>14.2f} "
            f"{item['decode_ms']:>12.2f} {item['output_tps']:>14.2f} "
            f"{item['prefill_ms']:>12.2f}"
        )

    gain_vs_current = (
        (float(winner["decode_tps"]) / current_decode_tps - 1.0) * 100.0
        if current_decode_tps > 0.0
        else 0.0
    )
    decision = {
        "decision": "PROMOTE_WINNER" if winner["name"] != "current_policy" else "KEEP_CURRENT",
        "winner": winner["name"],
        "winner_decode_tps": winner["decode_tps"],
        "current_decode_tps": current_decode_tps,
        "gain_vs_current_percent": gain_vs_current,
        "results": results,
    }
    decision_path = out_root / "decision.json"
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("\nDECISION " + json.dumps(decision, separators=(",", ":")))
    print(f"wrote {decision_path}")
    return 0 if len(successful) == len(CASES) else 2


if __name__ == "__main__":
    raise SystemExit(main())

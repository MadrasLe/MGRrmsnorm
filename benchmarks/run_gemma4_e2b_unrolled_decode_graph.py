"""Three-way L4 gate for a genuinely multi-token Gemma 4 E2B CUDA Graph.

The graph cases execute the same one-token model implementation and persistent
token feedback.  The graph control launches once per token; the candidate
captures up to eight dependent steps and replays the complete burst with one
host launch.  Eager multi-step remains in the matrix to verify that the first
post-capture replay also restores production token parity.
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
CASES: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "eager_multi_step",
        {
            "MEGAGEMM_DECODE_CUDA_GRAPHS": "0",
            "MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP": "0",
            "MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK": "0",
            "MEGAGEMM_DECODE_UNROLLED_GRAPH_BURST": "0",
            "MEGAGEMM_NATIVE_DECODE_GRAPH_BURST": "0",
        },
    ),
    (
        "one_step_graph",
        {
            "MEGAGEMM_DECODE_UNROLLED_GRAPH_BURST": "0",
            "MEGAGEMM_NATIVE_DECODE_GRAPH_BURST": "0",
        },
    ),
    (
        "unrolled_graph",
        {
            "MEGAGEMM_DECODE_UNROLLED_GRAPH_BURST": "1",
            "MEGAGEMM_NATIVE_DECODE_GRAPH_BURST": "0",
        },
    ),
)


def _environment(
    overrides: dict[str, str], *, reuse_scheduler: bool = False
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else str(ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    env.update(
        {
            "MEGAGEMM_FLAT_DECODE": "1",
            "MEGAGEMM_DISABLE_CUDA_RMSNORM": "1",
            "MEGAGEMM_REUSE_REQUEST_SCHEDULER": "1" if reuse_scheduler else "0",
            "MEGAGEMM_DECODE_PREFER_STEP": "0",
            "MEGAGEMM_DECODE_CUDA_GRAPHS": "1",
            "MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP": "1",
            "MEGAGEMM_DECODE_CUDA_GRAPHS_SHAPE_CACHE": "1",
            "MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE": "0",
            "MEGAGEMM_DECODE_CUDA_GRAPHS_STABLE_MAX_BLOCKS": "1",
            "MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK": "1",
            "MEGAGEMM_DECODE_GRAPH_TOKEN_BURST": "1",
            "MEGAGEMM_MULTI_STEP_BURST_BATCH": "8",
            "MEGAGEMM_BENCHMARK_TOKEN_DIGEST": "1",
            "MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE": "1",
            "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_PREFILL": "1",
            # Keep the validated E2B/B8 MLP policy identical in both processes.
            "MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE": "0",
            "MEGAGEMM_GEMMA4_FORCE_FUSED_GATEUP_USE": "0",
            "MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE": "0",
            "MEGAGEMM_GEMMA4_FORCE_DEEPFUSION_USE": "0",
        }
    )
    env.update(overrides)
    return env


def _load_rows(case_dir: Path) -> list[dict[str, Any]]:
    raw_files = sorted(case_dir.glob("*.jsonl"))
    if len(raw_files) != 1:
        raise RuntimeError(
            f"expected one raw JSONL in {case_dir}, found {len(raw_files)}"
        )
    rows = []
    for line in raw_files[0].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            bool(row.get("ok"))
            and int(row.get("batch_size", 0) or 0) == 8
            and int(row.get("prompt_tokens_requested_per_request", 0) or 0) == 2048
        ):
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no successful P2048/B8 rows in {raw_files[0]}")
    return rows


def _case_result(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    digests = {str(row.get("generated_token_digest") or "") for row in rows}
    if len(digests) != 1 or len(next(iter(digests), "")) != 64:
        raise RuntimeError(f"{name} token digest is missing or nondeterministic")
    decode_ms = []
    decode_tps = []
    output_tps = []
    graph_stats = []
    for row in rows:
        stats = row.get("scheduler_stats") or {}
        milliseconds = float(stats.get("decode_time_ms", 0.0) or 0.0)
        tokens = int(row.get("generated_tokens", 0) or 0)
        if milliseconds <= 0.0 or tokens <= 0:
            raise RuntimeError(f"{name} row lacks decode timing or tokens")
        decode_ms.append(milliseconds)
        decode_tps.append(tokens / (milliseconds / 1000.0))
        output_tps.append(float(row["output_tps"]))
        graph_stats.append(dict(stats.get("decode_cuda_graphs") or {}))

    graph_failures = sum(int(x.get("failures", 0) or 0) for x in graph_stats)
    unrolled_captures = sum(
        int(x.get("unrolled_token_burst_captures", 0) or 0) for x in graph_stats
    )
    unrolled_replays = sum(
        int(x.get("unrolled_token_burst_replays", 0) or 0) for x in graph_stats
    )
    unrolled_bursts = sum(
        int(x.get("unrolled_token_bursts", 0) or 0) for x in graph_stats
    )
    unrolled_steps = sum(
        int(x.get("unrolled_token_burst_steps", 0) or 0) for x in graph_stats
    )
    unrolled_failures = sum(
        int(x.get("unrolled_token_burst_failures", 0) or 0) for x in graph_stats
    )
    if name != "eager_multi_step" and graph_failures:
        raise RuntimeError(f"{name} had {graph_failures} one-step graph failures")
    if name == "unrolled_graph":
        # A steady-state request may reuse graphs captured by warmup.  Replays,
        # bursts and covered steps prove that measured rows used the candidate;
        # requiring a fresh capture would force its setup cost into every row.
        if unrolled_replays <= 0 or unrolled_bursts <= 0 or unrolled_steps <= 0:
            raise RuntimeError("candidate did not execute the unrolled graph")
        if unrolled_failures:
            raise RuntimeError(
                f"candidate had {unrolled_failures} unrolled capture/replay failures"
            )
    elif unrolled_bursts:
        raise RuntimeError("control unexpectedly executed the unrolled graph")

    return {
        "name": name,
        "status": "ok",
        "samples": len(rows),
        "generated_token_digest": next(iter(digests)),
        "median_decode_tps": statistics.median(decode_tps),
        "median_decode_ms": statistics.median(decode_ms),
        "median_output_tps": statistics.median(output_tps),
        "unrolled_captures": unrolled_captures,
        "unrolled_replays": unrolled_replays,
        "unrolled_bursts": unrolled_bursts,
        "unrolled_steps": unrolled_steps,
        "graph_stats": graph_stats,
    }


def _preflight() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the unrolled graph gate requires CUDA")
    device_name = torch.cuda.get_device_name()
    if "L4" not in device_name.upper():
        raise RuntimeError(f"expected NVIDIA L4, found {device_name}")
    print(f"GPU: {device_name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gemma 4 E2B/L4 one-step vs unrolled CUDA Graph gate"
    )
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--out-root", default="/tmp/gemma4_e2b_unrolled_graph")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--reuse-scheduler",
        action="store_true",
        help=(
            "reuse the request scheduler and captured graphs after warmup; "
            "without this flag the gate measures cold-request capture cost"
        ),
    )
    args = parser.parse_args()
    if args.max_new_tokens < 32:
        raise ValueError("--max-new-tokens must be at least 32")
    _preflight()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for name, overrides in CASES:
        case_dir = out_root / name
        case_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(MATRIX),
            "--backend", "megagemm",
            "--model", args.model,
            "--hardware-label", "1xl4",
            "--batch-sizes", "8",
            "--prompt-tokens", "2048",
            "--max-new-tokens", str(args.max_new_tokens),
            "--repeats", str(args.repeats),
            "--warmup", str(args.warmup),
            "--out-dir", str(case_dir),
            "--run-id", f"gemma4_e2b_{name}",
            "--device", "cuda",
            "--dtype", "bf16",
            "--max-seq-len", "2304",
            "--max-batch-size", "8",
            "--ignore-eos",
        ]
        print(f"\n=== {name} ===", flush=True)
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=_environment(overrides, reuse_scheduler=args.reuse_scheduler),
            check=False,
        )
        if completed.returncode != 0:
            results.append(
                {"name": name, "status": "failed", "returncode": completed.returncode}
            )
            continue
        try:
            results.append(_case_result(name, _load_rows(case_dir)))
        except Exception as exc:
            results.append(
                {"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )

    successful = [x for x in results if x.get("status") == "ok"]
    print("\n" + "=" * 92)
    print("GEMMA 4 E2B / L4 / BF16 — EAGER VS ONE-STEP VS UNROLLED GRAPH")
    print("=" * 92)
    print(
        f"{'path':<20} {'decode tok/s':>14} {'decode ms':>12} "
        f"{'output tok/s':>14} {'captures':>10} {'replays':>10}"
    )
    for item in successful:
        print(
            f"{item['name']:<20} {item['median_decode_tps']:>14.2f} "
            f"{item['median_decode_ms']:>12.2f} {item['median_output_tps']:>14.2f} "
            f"{item['unrolled_captures']:>10d} {item['unrolled_replays']:>10d}"
        )

    eager = next((x for x in successful if x["name"] == "eager_multi_step"), None)
    control = next((x for x in successful if x["name"] == "one_step_graph"), None)
    candidate = next((x for x in successful if x["name"] == "unrolled_graph"), None)
    decision = "INCOMPLETE"
    gain_vs_one_step = 0.0
    gain_vs_eager = 0.0
    one_step_vs_eager = 0.0
    token_parity = {
        "unrolled_matches_one_step": False,
        "one_step_matches_eager": False,
        "unrolled_matches_eager": False,
    }
    if eager and control and candidate:
        token_parity = {
            "unrolled_matches_one_step": (
                candidate["generated_token_digest"]
                == control["generated_token_digest"]
            ),
            "one_step_matches_eager": (
                control["generated_token_digest"] == eager["generated_token_digest"]
            ),
            "unrolled_matches_eager": (
                candidate["generated_token_digest"] == eager["generated_token_digest"]
            ),
        }
        gain_vs_one_step = (
            float(candidate["median_decode_tps"])
            / float(control["median_decode_tps"])
            - 1.0
        ) * 100.0
        gain_vs_eager = (
            float(candidate["median_decode_tps"])
            / float(eager["median_decode_tps"])
            - 1.0
        ) * 100.0
        one_step_vs_eager = (
            float(control["median_decode_tps"])
            / float(eager["median_decode_tps"])
            - 1.0
        ) * 100.0
        if not all(token_parity.values()):
            decision = "UNROLLED_CORRECTNESS_FAILURE"
        elif gain_vs_one_step >= 0.5 and gain_vs_eager >= 0.0:
            decision = (
                "UNROLLED_GRAPH_WINS_STEADY_STATE"
                if args.reuse_scheduler
                else "UNROLLED_GRAPH_WINS_COLD_REQUEST"
            )
        elif one_step_vs_eager >= 0.5:
            decision = "KEEP_ONE_STEP_GRAPH"
        else:
            decision = "KEEP_EAGER_MULTI_STEP"

    payload = {
        "decision": decision,
        "measurement_mode": (
            "steady_state_scheduler_reuse"
            if args.reuse_scheduler
            else "cold_request"
        ),
        "request_scheduler_reuse": bool(args.reuse_scheduler),
        "token_parity": token_parity,
        "unrolled_vs_one_step_percent": gain_vs_one_step,
        "unrolled_vs_eager_percent": gain_vs_eager,
        "one_step_vs_eager_percent": one_step_vs_eager,
        "results": results,
    }
    decision_path = out_root / "decision.json"
    decision_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nDECISION " + json.dumps(payload, separators=(",", ":")))
    print(f"wrote {decision_path}")
    if decision == "UNROLLED_CORRECTNESS_FAILURE":
        return 3
    return 0 if len(successful) == len(CASES) else 2


if __name__ == "__main__":
    raise SystemExit(main())

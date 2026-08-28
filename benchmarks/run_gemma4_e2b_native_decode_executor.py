"""A/B gate for the Gemma 4 E2B/L4 native decode burst executor.

The three subprocesses isolate process-global Triton and CUDA Graph state:

* eager_multi_step: current production multi-step Python layer/token loops;
* python_graph_burst: one-step graph with the token replay loop in Python;
* native_graph_burst: the same graph and kernels, replayed by the C++ executor.

This makes the native-vs-Python comparison attributable: both graph cases run
the identical captured device workload and differ only in burst orchestration.
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
            "MEGAGEMM_DECODE_PREFER_STEP": "0",
            "MEGAGEMM_DECODE_CUDA_GRAPHS": "0",
            "MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP": "0",
            "MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK": "0",
            "MEGAGEMM_NATIVE_DECODE_GRAPH_BURST": "0",
        },
    ),
    (
        "python_graph_burst",
        {
            "MEGAGEMM_DECODE_PREFER_STEP": "0",
            "MEGAGEMM_DECODE_CUDA_GRAPHS": "1",
            "MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP": "1",
            "MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK": "1",
            "MEGAGEMM_NATIVE_DECODE_GRAPH_BURST": "0",
        },
    ),
    (
        "native_graph_burst",
        {
            "MEGAGEMM_DECODE_PREFER_STEP": "0",
            "MEGAGEMM_DECODE_CUDA_GRAPHS": "1",
            "MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP": "1",
            "MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK": "1",
            "MEGAGEMM_NATIVE_DECODE_GRAPH_BURST": "1",
        },
    ),
)


def _environment(overrides: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(ROOT)
        if not current_pythonpath
        else str(ROOT) + os.pathsep + current_pythonpath
    )
    env.update(
        {
            "MEGAGEMM_FLAT_DECODE": "1",
            "MEGAGEMM_DISABLE_CUDA_RMSNORM": "1",
            "MEGAGEMM_REUSE_REQUEST_SCHEDULER": "0",
            "MEGAGEMM_DECODE_CUDA_GRAPHS_SHAPE_CACHE": "1",
            "MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE": "0",
            "MEGAGEMM_DECODE_CUDA_GRAPHS_STABLE_MAX_BLOCKS": "1",
            "MEGAGEMM_DECODE_GRAPH_TOKEN_BURST": "1",
            "MEGAGEMM_MULTI_STEP_BURST_BATCH": "8",
            "MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE": "1",
            "MEGAGEMM_GEMMA4_E2B_L4_SLIDING_PREFILL": "1",
            # Hold the validated B8 MLP policy fixed across all three cases.
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
    output_tps = [float(row["output_tps"]) for row in rows]
    decode_ms = []
    decode_tps = []
    graph_stats = []
    generated = []
    for row in rows:
        tokens = int(row.get("generated_tokens", 0) or 0)
        stats = row.get("scheduler_stats") or {}
        milliseconds = float(stats.get("decode_time_ms", 0.0) or 0.0)
        if milliseconds <= 0.0:
            raise RuntimeError(f"{name} row has no decode wall time")
        generated.append(tokens)
        decode_ms.append(milliseconds)
        decode_tps.append(tokens / (milliseconds / 1000.0))
        graph_stats.append(dict(stats.get("decode_cuda_graphs") or {}))

    result = {
        "name": name,
        "status": "ok",
        "samples": len(rows),
        "generated_tokens": generated,
        "median_output_tps": statistics.median(output_tps),
        "median_decode_tps": statistics.median(decode_tps),
        "median_decode_ms": statistics.median(decode_ms),
        "graph_stats": graph_stats,
    }
    failures = sum(int(stats.get("failures", 0) or 0) for stats in graph_stats)
    captures = sum(int(stats.get("captures", 0) or 0) for stats in graph_stats)
    native_bursts = sum(
        int(stats.get("native_token_bursts", 0) or 0) for stats in graph_stats
    )
    native_steps = sum(
        int(stats.get("native_token_burst_steps", 0) or 0) for stats in graph_stats
    )
    result.update(
        {
            "graph_failures": failures,
            "graph_captures": captures,
            "native_bursts": native_bursts,
            "native_steps": native_steps,
        }
    )
    if name != "eager_multi_step" and (captures <= 0 or failures != 0):
        raise RuntimeError(
            f"{name} did not establish a clean decode graph: "
            f"captures={captures}, failures={failures}"
        )
    if name == "native_graph_burst" and (native_bursts <= 0 or native_steps <= 0):
        raise RuntimeError(
            "native_graph_burst completed without exercising the C++ executor"
        )
    if name != "native_graph_burst" and native_bursts != 0:
        raise RuntimeError(f"{name} unexpectedly used the native executor")
    return result


def _native_extension_preflight() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the native decode executor gate requires CUDA")
    device_name = torch.cuda.get_device_name()
    if "L4" not in device_name.upper():
        raise RuntimeError(f"expected NVIDIA L4, found {device_name}")
    try:
        import megagemm_decode_ops
    except ImportError as exc:
        raise RuntimeError(
            "megagemm_decode_ops is not installed; run "
            "`python -m pip install -e . --no-build-isolation` first"
        ) from exc
    if not hasattr(megagemm_decode_ops, "run_cuda_graph_token_burst"):
        raise RuntimeError(
            "megagemm_decode_ops is stale and lacks run_cuda_graph_token_burst; "
            "rebuild the editable package"
        )
    print(f"GPU: {device_name}")
    print(f"native extension: {Path(megagemm_decode_ops.__file__).resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gemma 4 E2B/L4 native decode executor A/B gate"
    )
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--out-root", default="/tmp/gemma4_e2b_native_decode")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.max_new_tokens < 32:
        raise ValueError("--max-new-tokens must be at least 32 to amortize capture")
    _native_extension_preflight()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for name, overrides in CASES:
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
            f"gemma4_e2b_native_{name}",
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
            env=_environment(overrides),
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

    successful = [item for item in results if item.get("status") == "ok"]
    print("\n" + "=" * 92)
    print("GEMMA 4 E2B / L4 / BF16 — NATIVE DECODE EXECUTOR")
    print("=" * 92)
    print(
        f"{'path':<22} {'decode tok/s':>14} {'decode ms':>12} "
        f"{'output tok/s':>14} {'native bursts':>14}"
    )
    for item in successful:
        print(
            f"{item['name']:<22} {item['median_decode_tps']:>14.2f} "
            f"{item['median_decode_ms']:>12.2f} {item['median_output_tps']:>14.2f} "
            f"{item['native_bursts']:>14d}"
        )

    eager = next((x for x in successful if x["name"] == "eager_multi_step"), None)
    python_graph = next(
        (x for x in successful if x["name"] == "python_graph_burst"), None
    )
    native = next((x for x in successful if x["name"] == "native_graph_burst"), None)
    decision = "INCOMPLETE"
    native_vs_eager = 0.0
    native_vs_python_graph = 0.0
    if eager and python_graph and native:
        native_vs_eager = (
            float(native["median_decode_tps"]) / float(eager["median_decode_tps"]) - 1.0
        ) * 100.0
        native_vs_python_graph = (
            float(native["median_decode_tps"])
            / float(python_graph["median_decode_tps"])
            - 1.0
        ) * 100.0
        decision = (
            "PROMOTE_NATIVE_EXECUTOR"
            if native_vs_eager >= 2.0 and native_vs_python_graph >= 0.5
            else "KEEP_EXPERIMENTAL"
        )

    payload = {
        "decision": decision,
        "native_vs_eager_percent": native_vs_eager,
        "native_vs_python_graph_percent": native_vs_python_graph,
        "results": results,
    }
    decision_path = out_root / "decision.json"
    decision_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nDECISION " + json.dumps(payload, separators=(",", ":")))
    print(f"wrote {decision_path}")
    return 0 if len(successful) == len(CASES) else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Isolated CUDA-process diagnosis for the Gemma 4 B16 full-prefill graph.

The child gate can poison a CUDA context with an illegal address. This parent
never initializes CUDA itself: it bisects layer prefixes in fresh processes,
then runs a small feature-ablation matrix at the first failing prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "benchmarks" / "run_gemma4_b16_full_model_graph_preflight.py"
NUM_LAYERS = 30


def _last_stage(output: str) -> str:
    stage = "process_start"
    for line in output.splitlines():
        marker = "FULL_MODEL_GRAPH_STAGE "
        if line.startswith(marker):
            stage = line[len(marker) :].strip()
    return stage


def _error_tail(output: str, limit: int = 12) -> list[str]:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    return lines[-limit:]


def run_case(
    *,
    name: str,
    layer_limit: int,
    out_dir: Path,
    env_overrides: dict[str, str] | None = None,
    skip_final_projection: bool = False,
    timeout_seconds: int,
) -> dict:
    case_out = out_dir / f"{name}.json"
    command = [
        sys.executable,
        str(CHILD),
        "--replays",
        "1",
        "--layer-limit",
        str(layer_limit),
        "--out-json",
        str(case_out),
    ]
    if skip_final_projection:
        command.append("--skip-final-projection")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        env.update(env_overrides)

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
        output = completed.stdout or ""
        return_code = int(completed.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or ""
        output = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
        return_code = 124
        timed_out = True

    payload = None
    if case_out.exists():
        try:
            payload = json.loads(case_out.read_text(encoding="utf-8"))
        except Exception:
            payload = None
    last_stage = _last_stage(output)
    cuda_safe = last_stage == "replay_synchronized"
    passed = bool(
        return_code == 0
        and isinstance(payload, dict)
        and payload.get("status") == "PASS"
    )
    result = {
        "name": name,
        "layer_limit": int(layer_limit),
        "passed": passed,
        "cuda_safe": cuda_safe,
        "return_code": return_code,
        "timed_out": timed_out,
        "last_stage": last_stage,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "skip_final_projection": bool(skip_final_projection),
        "env_overrides": dict(env_overrides or {}),
        "error_tail": [] if passed else _error_tail(output),
    }
    print("FULL_MODEL_GRAPH_CASE " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_b16_full_model_graph_diagnose.json",
    )
    parser.add_argument(
        "--child-timeout-seconds",
        type=int,
        default=150,
    )
    args = parser.parse_args()

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    case_dir = out_path.parent / f"{out_path.stem}_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    timeout_seconds = max(1, int(args.child_timeout_seconds))

    print("Gemma4 B16 full-model graph isolated diagnosis")
    print("  model_download: disabled")
    print("  vllm_install: disabled")
    print("  cuda_context: one fresh child process per case")
    print("  strategy: layer-prefix bisection + targeted ablations")

    cases: list[dict] = []
    cache: dict[int, dict] = {}

    def prefix(layer_limit: int) -> dict:
        layer_limit = int(layer_limit)
        if layer_limit not in cache:
            result = run_case(
                name=f"baseline_layers_{layer_limit:02d}",
                layer_limit=layer_limit,
                out_dir=case_dir,
                timeout_seconds=timeout_seconds,
            )
            cache[layer_limit] = result
            cases.append(result)
        return cache[layer_limit]

    full = prefix(NUM_LAYERS)
    first_failing_prefix = None
    last_passing_prefix = NUM_LAYERS
    if not full["cuda_safe"]:
        low = 0
        high = NUM_LAYERS
        while high - low > 1:
            middle = (low + high) // 2
            if prefix(middle)["cuda_safe"]:
                low = middle
            else:
                high = middle
        last_passing_prefix = low
        first_failing_prefix = high

    ablations: list[dict] = []
    passing_ablations: list[str] = []
    if first_failing_prefix is not None:
        target = int(first_failing_prefix)
        ablation_specs = (
            (
                "disable_expandable_segments",
                {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"},
                False,
            ),
            (
                "skip_final_projection",
                {},
                True,
            ),
            (
                "disable_compact_route_pack",
                {"MEGAGEMM_GEMMA4_MOE_PREFILL_COMPACT_ROUTE_PACK": "0"},
                False,
            ),
            (
                "disable_implicit_causal",
                {"MEGAGEMM_GEMMA4_IMPLICIT_CAUSAL_PREFILL": "0"},
                False,
            ),
            (
                "disable_fused_attention_frontend",
                {
                    "MEGAGEMM_GEMMA4_FUSED_QKV_PREFILL": "0",
                    "MEGAGEMM_GEMMA4_FUSED_ATTN_PREP_PREFILL": "0",
                },
                False,
            ),
            (
                "disable_fused_ffn_norms",
                {
                    "MEGAGEMM_GEMMA4_FUSED_DUAL_FFN_NORM_PREFILL": "0",
                    "MEGAGEMM_GEMMA4_FUSED_ADD_FFN_NORM_PREFILL": "0",
                    "MEGAGEMM_GEMMA4_FUSED_POST_FFN_NORMS_PREFILL": "0",
                },
                False,
            ),
        )
        for name, overrides, skip_final in ablation_specs:
            result = run_case(
                name=f"{name}_layers_{target:02d}",
                layer_limit=target,
                out_dir=case_dir,
                env_overrides=overrides,
                skip_final_projection=skip_final,
                timeout_seconds=timeout_seconds,
            )
            cases.append(result)
            ablations.append(result)
            if result["cuda_safe"]:
                passing_ablations.append(name)

    payload = {
        "status": (
            "PASS"
            if full["passed"]
            else ("CUDA_SAFE_NUMERIC_FAIL" if full["cuda_safe"] else "FAULT_ISOLATED")
        ),
        "model_download": False,
        "vllm_install": False,
        "full_model_passed": bool(full["passed"]),
        "full_model_cuda_safe": bool(full["cuda_safe"]),
        "last_passing_prefix": int(last_passing_prefix),
        "first_failing_prefix": first_failing_prefix,
        "first_failing_layer_index": (
            None
            if first_failing_prefix is None
            else int(first_failing_prefix) - 1
        ),
        "first_failing_layer_type": (
            None
            if first_failing_prefix is None
            else (
                "full_attention"
                if int(first_failing_prefix) - 1 in {5, 11, 17, 23, 29}
                else "sliding_attention"
            )
        ),
        "passing_ablations": passing_ablations,
        "cases": cases,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("FULL_MODEL_GRAPH_DIAGNOSIS " + json.dumps(payload, sort_keys=True))
    print(f"wrote {out_path}")
    # A found fault is a successful diagnosis. Only infrastructure/timeouts fail
    # this parent so a poisoned child does not turn the Colab cell into a traceback.
    infrastructure_failed = bool(
        any(case["timed_out"] for case in cases)
        or not cases
    )
    return 2 if infrastructure_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

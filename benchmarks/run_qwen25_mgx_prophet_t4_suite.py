"""Run Qwen 2.5 MGX + Prophet probes on T4-class GPUs.

This runner is intentionally separate from the inference matrix. The matrix
measures fresh prefill + decode throughput, while Prophet measures how much
prompt preparation can be avoided when a compatible state already exists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROPHET_BENCH = ROOT / "benchmarks" / "benchmark_prophet.py"
DEFAULT_WORKLOAD = ROOT / "benchmarks" / "workloads" / "prophet_cache_reuse_2k.json"


def _slugify(value: str) -> str:
    slug = value.strip().replace("\\", "/").strip("/")
    slug = slug.replace("/", "--")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug)
    return slug[:120] or "model"


def _shell_join(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def _parse_modes(raw: str) -> list[str]:
    aliases = {
        "all": ["cachelike", "validated"],
        "core": ["cachelike", "validated"],
    }
    modes: list[str] = []
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        selected = aliases.get(key, [key])
        for mode in selected:
            if mode not in {"cachelike", "validated"}:
                raise SystemExit("Unknown mode {!r}; use cachelike, validated, or all.".format(mode))
            if mode not in modes:
                modes.append(mode)
    if not modes:
        raise SystemExit("No modes selected")
    return modes


def _default_mgx_path(args: argparse.Namespace) -> Path:
    suffix = args.quantize if args.quantize != "none" else args.dtype
    return Path(args.artifacts_dir) / "{}-{}.mgx".format(_slugify(args.model), suffix)


def _mode_json_path(args: argparse.Namespace, run_id: str, mode: str) -> Path:
    return Path(args.out_dir) / "{}_{}_mgx_prophet.json".format(run_id, mode)


def _mode_prophet_dir(args: argparse.Namespace, run_id: str, mode: str) -> Path:
    if args.prophet_dir:
        return Path(args.prophet_dir) / mode
    return Path(args.out_dir) / "{}_{}_prophet_library".format(run_id, mode)


def _mode_command(
    args: argparse.Namespace,
    *,
    run_id: str,
    mode: str,
    mgx_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROPHET_BENCH),
        "--model",
        args.model,
        "--mgx",
        str(mgx_path),
        "--prophet-dir",
        str(_mode_prophet_dir(args, run_id, mode)),
        "--dtype",
        args.dtype,
        "--quantize",
        args.quantize,
        "--device",
        "cuda",
        "--workload-file",
        str(args.workload_file),
        "--max-seq-len",
        str(args.max_seq_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-batch-size",
        str(args.max_batch_size),
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
        "--prefix-tokens",
        str(args.prefix_tokens),
        "--top-k",
        str(args.top_k),
        "--min-similarity",
        str(args.min_similarity),
        "--min-prefix-coverage",
        str(args.min_prefix_coverage),
        "--min-prefix-reuse-score",
        str(args.min_prefix_reuse_score),
        "--max-prefix-rollback-ratio",
        str(args.max_prefix_rollback_ratio),
        "--max-prefix-tail-ratio",
        str(args.max_prefix_tail_ratio),
        "--json-out",
        str(_mode_json_path(args, run_id, mode)),
    ]
    if args.export_if_missing:
        cmd.append("--export-if-missing")
    if args.force_export:
        cmd.append("--force-export")
    if args.reset_prophet_dir:
        cmd.append("--reset-prophet-dir")
    if args.skip_hash_check:
        cmd.append("--skip-hash-check")
    if args.mgx_emit_payload_cache:
        cmd.append("--mgx-emit-payload-cache")
    if args.mgx_payload_cache_dir:
        cmd.extend(["--mgx-payload-cache-dir", args.mgx_payload_cache_dir])
    if not args.mgx_prefer_payload_cache:
        cmd.append("--no-mgx-prefer-payload-cache")
    cmd.extend(["--mgx-export-mode", args.mgx_export_mode])

    if mode == "cachelike":
        cmd.extend(
            [
                "--validation-mode",
                "none",
                "--continuation-tokens",
                str(args.cachelike_continuation_tokens),
            ]
        )
    else:
        cmd.extend(
            [
                "--validation-mode",
                "full_prefill",
                "--validation-tokens",
                str(args.validation_tokens),
                "--continuation-tokens",
                str(args.continuation_tokens),
                "--fallback-to-prefill",
            ]
        )
    return cmd


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _write_report(args: argparse.Namespace, run_id: str, modes: list[str], mgx_path: Path) -> Path:
    report_path = Path(args.out_dir) / f"{run_id}_qwen25_mgx_prophet_t4_report.md"
    lines = [
        "# Qwen 2.5 MGX + Prophet T4 Benchmark",
        "",
        "## Configuration",
        "",
        f"- Model: `{args.model}`",
        f"- MGX artifact: `{mgx_path}`",
        f"- Dtype/quantize: `{args.dtype}` / `{args.quantize}`",
        f"- Workload: `{args.workload_file}`",
        f"- Max seq len: `{args.max_seq_len}`",
        f"- Max new tokens headroom: `{args.max_new_tokens}`",
        f"- Runs/warmup: `{args.runs}` / `{args.warmup}`",
        f"- MGX payload cache preferred: `{args.mgx_prefer_payload_cache}`",
        f"- MGX payload cache emitted/primed: `{args.mgx_emit_payload_cache}`",
        f"- Run id: `{run_id}`",
        "",
        "## Route Summary",
        "",
        "| Mode | Route | Count | Avg baseline s | Avg Prophet s | Avg speedup % |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for mode in modes:
        payload = _load_json(_mode_json_path(args, run_id, mode))
        route_summary = payload.get("route_summary") or {}
        if not route_summary:
            lines.append(f"| {mode} | no results | | | | |")
            continue
        for route, row in sorted(route_summary.items()):
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    mode,
                    route,
                    row.get("count", ""),
                    _fmt(row.get("avg_baseline_seconds")),
                    _fmt(row.get("avg_prophet_seconds")),
                    _fmt(row.get("avg_speedup_pct"), digits=2),
                )
            )

    lines.extend(
        [
            "",
            "## Query Summary",
            "",
            "| Mode | Query | Avg baseline s | Avg Prophet s | Speedup % | Reused ratio | Accepted | Restored |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for mode in modes:
        payload = _load_json(_mode_json_path(args, run_id, mode))
        for query in payload.get("queries", []) or []:
            summary = query.get("summary") or {}
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    mode,
                    query.get("label", ""),
                    _fmt(summary.get("avg_baseline_seconds")),
                    _fmt(summary.get("avg_prophet_seconds")),
                    _fmt(summary.get("avg_speedup_pct"), digits=2),
                    _fmt(summary.get("avg_reused_token_ratio_estimate"), digits=3),
                    summary.get("accepted_runs", ""),
                    summary.get("restored_runs", ""),
                )
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen 2.5 MGX + Prophet T4 probes.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--modes", default="cachelike,validated", help="cachelike, validated, or all")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--quantize", choices=("none", "int8", "int4"), default="int8")
    parser.add_argument("--out-dir", default="bench_results/qwen25_mgx_prophet_t4")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--mgx", help="Path to an existing or desired .mgx artifact.")
    parser.add_argument("--prophet-dir", help="Base directory for Prophet libraries. Each mode gets a subdirectory.")
    parser.add_argument("--workload-file", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    parser.add_argument("--min-prefix-coverage", type=float, default=0.50)
    parser.add_argument("--min-prefix-reuse-score", type=float, default=0.55)
    parser.add_argument("--max-prefix-rollback-ratio", type=float, default=0.35)
    parser.add_argument("--max-prefix-tail-ratio", type=float, default=0.50)
    parser.add_argument("--cachelike-continuation-tokens", type=int, default=0)
    parser.add_argument("--validation-tokens", type=int, default=4)
    parser.add_argument("--continuation-tokens", type=int, default=8)
    parser.add_argument("--export-if-missing", action="store_true", default=True)
    parser.add_argument("--no-export-if-missing", dest="export_if_missing", action="store_false")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--reset-prophet-dir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--mgx-prefer-payload-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mgx-payload-cache-dir")
    parser.add_argument("--mgx-emit-payload-cache", action="store_true", default=True)
    parser.add_argument("--no-mgx-emit-payload-cache", dest="mgx_emit_payload_cache", action="store_false")
    parser.add_argument("--mgx-export-mode", choices=("normal", "streaming"), default="streaming")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    modes = _parse_modes(args.modes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)
    if args.mgx_payload_cache_dir:
        Path(args.mgx_payload_cache_dir).mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or time.strftime("qwen25_mgx_prophet_%Y%m%d_%H%M%S")
    mgx_path = Path(args.mgx) if args.mgx else _default_mgx_path(args)

    print("Qwen 2.5 MGX + Prophet T4 suite")
    print(f"  run_id: {run_id}")
    print(f"  model:  {args.model}")
    print(f"  modes:  {', '.join(modes)}")
    print(f"  mgx:    {mgx_path}")
    print(f"  out:    {out_dir}")
    print(f"  workload: {args.workload_file}")

    commands = [(mode, _mode_command(args, run_id=run_id, mode=mode, mgx_path=mgx_path)) for mode in modes]
    for mode, cmd in commands:
        print()
        print(f"=== Prophet mode: {mode} ===")
        print(_shell_join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(ROOT), check=True)

    if args.dry_run:
        return 0

    report = _write_report(args, run_id, modes, mgx_path)
    print()
    print(f"Wrote combined report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

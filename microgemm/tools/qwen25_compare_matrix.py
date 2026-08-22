#!/usr/bin/env python3
"""Run a small same-session MicroGemm vs llama.cpp benchmark matrix.

The individual paired benchmark script is intentionally strict about one model
at a time. This wrapper keeps the Colab workflow tidy: run several paired cases
back-to-back, then write one rollup CSV/Markdown table with the comparison
numbers that matter.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


SCRIPT_BUILD_TAG = "qwen25_compare_matrix_v2_groupwise_quant_cases"

CASE_DEFS: dict[str, dict[str, Any]] = {
    "qwen25_05b_q8": {
        "label": "Qwen2.5 0.5B INT8 vs Q8_0",
        "preset": "qwen25_05b",
        "quant": "int8",
        "llamacpp_quant": "q8_0",
        "batch_sizes": "8",
        "prompt_tokens": "64,256",
        "batch_prompt_tokens": 64,
        "max_new_tokens": 128,
    },
    "qwen25_05b_int8g_q8": {
        "label": "Qwen2.5 0.5B INT8G128 vs Q8_0",
        "preset": "qwen25_05b",
        "quant": "int8g128",
        "llamacpp_quant": "q8_0",
        "batch_sizes": "8",
        "prompt_tokens": "64,256",
        "batch_prompt_tokens": 64,
        "max_new_tokens": 128,
    },
    "qwen25_15b_q8": {
        "label": "Qwen2.5 1.5B INT8 vs Q8_0",
        "preset": "qwen25_15b",
        "quant": "int8",
        "llamacpp_quant": "q8_0",
        "batch_sizes": "8",
        "prompt_tokens": "64,256",
        "batch_prompt_tokens": 64,
        "max_new_tokens": 64,
    },
    "qwen25_3b_q4": {
        "label": "Qwen2.5 3B INT4 vs Q4_K_M",
        "preset": "qwen25_3b",
        "quant": "int4",
        "llamacpp_quant": "q4_k_m",
        "batch_sizes": "8",
        "prompt_tokens": "64,256",
        "batch_prompt_tokens": 64,
        "max_new_tokens": 64,
    },
    "qwen25_3b_int4g_q4": {
        "label": "Qwen2.5 3B INT4G128 vs Q4_K_M",
        "preset": "qwen25_3b",
        "quant": "int4g128",
        "llamacpp_quant": "q4_k_m",
        "batch_sizes": "8",
        "prompt_tokens": "64,256",
        "batch_prompt_tokens": 64,
        "max_new_tokens": 64,
    },
    "qwen25_7b_q4": {
        "label": "Qwen2.5 7B INT4 vs Q4_K_M",
        "preset": "qwen25_7b",
        "quant": "int4",
        "llamacpp_quant": "q4_k_m",
        "batch_sizes": "8",
        "prompt_tokens": "64",
        "batch_prompt_tokens": 64,
        "max_new_tokens": 32,
    },
    "qwen25_14b_q4": {
        "label": "Qwen2.5 14B INT4 vs Q4_K_M",
        "preset": "qwen25_14b",
        "quant": "int4",
        "llamacpp_quant": "q4_k_m",
        "batch_sizes": "8",
        "prompt_tokens": "64",
        "batch_prompt_tokens": 64,
        "max_new_tokens": 16,
    },
    "mistral7b_q4": {
        "label": "Mistral 7B v0.3 INT4 vs Q4_K_M",
        "preset": "mistral7b_v03",
        "quant": "int4",
        "llamacpp_quant": "q4_k_m",
        "batch_sizes": "8",
        "prompt_tokens": "64,256",
        "batch_prompt_tokens": 64,
        "max_new_tokens": 32,
    },
    "mistral7b_int4g_q4": {
        "label": "Mistral 7B v0.3 INT4G128 vs Q4_K_M",
        "preset": "mistral7b_v03",
        "quant": "int4g128",
        "llamacpp_quant": "q4_k_m",
        "batch_sizes": "8",
        "prompt_tokens": "64,256",
        "batch_prompt_tokens": 64,
        "max_new_tokens": 32,
    },
    "llama31_8b_q4": {
        "label": "Llama 3.1 8B INT4 vs Q4_K_M",
        "preset": "llama31_8b",
        "quant": "int4",
        "llamacpp_quant": "q4_k_m",
        "batch_sizes": "8",
        "prompt_tokens": "64,256",
        "batch_prompt_tokens": 64,
        "max_new_tokens": 32,
    },
    "qwen35_9b_q8": {
        "label": "Qwen3.5 9B INT8 vs Q8_0",
        "preset": "qwen35_9b",
        "quant": "int8",
        "llamacpp_quant": "q8_0",
        "batch_sizes": "8",
        "prompt_tokens": "64,256",
        "batch_prompt_tokens": 64,
        "max_new_tokens": 128,
    },
    "qwen35_9b_int8g_q8": {
        "label": "Qwen3.5 9B INT8G128 vs Q8_0",
        "preset": "qwen35_9b",
        "quant": "int8g128",
        "llamacpp_quant": "q8_0",
        "batch_sizes": "8",
        "prompt_tokens": "64,256",
        "batch_prompt_tokens": 64,
        "max_new_tokens": 128,
    },
}

MATRICES: dict[str, list[str]] = {
    "quick": ["qwen25_05b_q8", "mistral7b_q4"],
    "standard": ["qwen25_05b_q8", "qwen25_15b_q8", "mistral7b_q4", "llama31_8b_q4", "qwen35_9b_q8"],
    "qwen25": ["qwen25_05b_q8", "qwen25_15b_q8", "qwen25_3b_q4", "qwen25_7b_q4", "qwen25_14b_q4"],
    "large": ["mistral7b_q4", "llama31_8b_q4", "qwen25_7b_q4", "qwen25_14b_q4", "qwen35_9b_q8"],
}

SUITE_BUILD_RE = re.compile(r'SUITE_BUILD_TAG\s*=\s*"([^"]+)"')


def fget(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def sget(row: dict[str, str], key: str) -> str:
    return str(row.get(key, "") or "")


def read_current_suite_build(root: Path) -> str:
    suite_path = root / "microgemm" / "tools" / "qwen25_cpu_suite.py"
    try:
        match = SUITE_BUILD_RE.search(suite_path.read_text(encoding="utf-8"))
    except OSError:
        return ""
    return match.group(1) if match else ""


def read_paired_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def collect_case_names(args: argparse.Namespace) -> list[str]:
    if args.case:
        names = args.case
    else:
        names = MATRICES[args.matrix]
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in CASE_DEFS:
            available = ", ".join(sorted(CASE_DEFS))
            raise SystemExit(f"unknown case {name!r}. Available cases: {available}")
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def build_command(
    *,
    root: Path,
    case_name: str,
    case: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
    case_out_dir: Path,
    force_builds: bool,
) -> list[str]:
    script = root / "microgemm" / "tools" / "qwen25_same_session_compare.py"
    cmd = [
        sys.executable,
        str(script),
        "--model-preset",
        str(case["preset"]),
        "--llamacpp-quant",
        str(case["llamacpp_quant"]),
        "--quant",
        str(case["quant"]),
        "--batch-sizes",
        str(case.get("batch_sizes", args.batch_sizes)),
        "--batch-prompt-tokens",
        str(case.get("batch_prompt_tokens", args.batch_prompt_tokens)),
        "--prompt-tokens",
        str(case.get("prompt_tokens", args.prompt_tokens)),
        "--max-new-tokens",
        str(case.get("max_new_tokens", args.max_new_tokens)),
        "--threads",
        str(args.threads),
        "--threads-batch",
        str(args.threads_batch or args.threads),
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
        "--out-dir",
        str(case_out_dir),
        "--run-id",
        run_id,
        "--expect-microgemm-suite-build",
        args.expect_microgemm_suite_build,
    ]
    if args.ignore_eos:
        cmd.append("--ignore-eos")
    if force_builds:
        cmd.extend(["--force-microgemm-rebuild", "--force-llamacpp-rebuild"])
    if args.force_convert:
        cmd.append("--force-convert")
    for env_override in args.microgemm_env:
        cmd.extend(["--microgemm-env", env_override])
    return cmd


def summarize_paired_row(case_name: str, case: dict[str, Any], row: dict[str, str]) -> dict[str, Any]:
    return {
        "case": case_name,
        "label": case["label"],
        "status": "ok",
        "preset": case["preset"],
        "quant": case["quant"],
        "llamacpp_quant": case["llamacpp_quant"],
        "batch_size": sget(row, "batch_size"),
        "prompt_tokens": sget(row, "prompt_tokens"),
        "max_new_tokens": sget(row, "max_new_tokens"),
        "micro_wall": fget(row, "microgemm_wall_output_tps"),
        "llama_wall": fget(row, "llamacpp_output_total_tps"),
        "wall_ratio": fget(row, "microgemm_wall_over_llamacpp_output"),
        "micro_decode": fget(row, "microgemm_decode_only_tps"),
        "llama_decode": fget(row, "llamacpp_decode_only_tps"),
        "decode_ratio": fget(row, "microgemm_decode_over_llamacpp_decode"),
        "micro_prefill": fget(row, "microgemm_prefill_tps"),
        "llama_prefill": fget(row, "llamacpp_prefill_tps"),
        "prefill_ratio": fget(row, "microgemm_prefill_over_llamacpp_prefill"),
        "micro_rope_kv_ms": fget(row, "microgemm_rope_kv_ms"),
        "micro_gate_dot_ms": fget(row, "microgemm_gate_up_dot_ms"),
        "micro_down_dot_ms": fget(row, "microgemm_down_proj_dot_ms"),
        "microgemm_suite_build": sget(row, "microgemm_suite_build"),
        "llamacpp_script_build": sget(row, "llamacpp_script_build"),
    }


def failure_row(case_name: str, case: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "case": case_name,
        "label": case["label"],
        "status": "failed",
        "preset": case["preset"],
        "quant": case["quant"],
        "llamacpp_quant": case["llamacpp_quant"],
        "error": message,
    }


def write_rollup(out_dir: Path, run_id: str, rows: list[dict[str, Any]], payload: dict[str, Any]) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_id}_comparison_matrix.json"
    csv_path = out_dir / f"{run_id}_comparison_matrix_summary.csv"
    md_path = out_dir / f"{run_id}_comparison_matrix_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_lines = [
        f"# MicroGemm vs llama.cpp comparison matrix",
        "",
        f"- script build: `{SCRIPT_BUILD_TAG}`",
        f"- run id: `{run_id}`",
        "",
        "| case | status | wall x | decode x | prefill x | micro wall | llama wall | micro decode | llama decode | rope ms | gate dot ms | down dot ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("status") != "ok":
            md_lines.append(f"| {row.get('case', '')} | failed |  |  |  |  |  |  |  |  |  |  |")
            continue
        md_lines.append(
            "| {case} | ok | {wall_ratio:.2f}x | {decode_ratio:.2f}x | {prefill_ratio:.2f}x | "
            "{micro_wall:.2f} | {llama_wall:.2f} | {micro_decode:.2f} | {llama_decode:.2f} | "
            "{micro_rope_kv_ms:.0f} | {micro_gate_dot_ms:.0f} | {micro_down_dot_ms:.0f} |".format(**row)
        )
    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, csv_path, md_path


def print_rollup(rows: list[dict[str, Any]]) -> None:
    print("\nComparison matrix rollup")
    print("  case              status     wall    decode   prefill   micro_wall  llama_wall  rope_ms  gate_ms  down_ms")
    for row in rows:
        if row.get("status") != "ok":
            print(f"  {row.get('case', ''):17s} failed")
            continue
        print(
            f"  {row['case']:17s} ok      "
            f"{row['wall_ratio']:6.2f}x "
            f"{row['decode_ratio']:7.2f}x "
            f"{row['prefill_ratio']:8.2f}x "
            f"{row['micro_wall']:10.2f} "
            f"{row['llama_wall']:10.2f} "
            f"{row['micro_rope_kv_ms']:8.0f} "
            f"{row['micro_gate_dot_ms']:8.0f} "
            f"{row['micro_down_dot_ms']:8.0f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a same-session MicroGemm vs llama.cpp comparison matrix")
    parser.add_argument("--matrix", choices=tuple(MATRICES), default="standard")
    parser.add_argument("--case", action="append", default=[], help="Run one named case. Repeat to build a custom matrix.")
    parser.add_argument(
        "--out-dir",
        default=str(Path(tempfile.gettempdir()) / "microgemm_bench_results" / "comparison_matrix"),
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--threads-batch", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompt-tokens", default="64,256")
    parser.add_argument("--batch-prompt-tokens", type=int, default=64)
    parser.add_argument("--batch-sizes", default="8")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--respect-eos", dest="ignore_eos", action="store_false", default=True)
    parser.add_argument("--force-builds-first", action="store_true")
    parser.add_argument("--force-builds-each", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--halt-on-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--microgemm-env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--expect-microgemm-suite-build", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.threads <= 0:
        args.threads = os.cpu_count() or 1
    if args.threads_batch <= 0:
        args.threads_batch = args.threads
    if not args.expect_microgemm_suite_build:
        args.expect_microgemm_suite_build = read_current_suite_build(root)
    run_id = args.run_id or time.strftime(f"{args.matrix}_matrix_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser()
    case_names = collect_case_names(args)

    print("MicroGemm comparison matrix")
    print(f"  script build:  {SCRIPT_BUILD_TAG}")
    print(f"  matrix:        {args.matrix}")
    print(f"  cases:         {','.join(case_names)}")
    print(f"  suite expect:  {args.expect_microgemm_suite_build or '(none)'}")
    print(f"  threads:       {args.threads}")
    print(f"  runs/warmup:   {args.runs}/{args.warmup}")

    rows: list[dict[str, Any]] = []
    case_outputs: dict[str, Any] = {}
    for idx, case_name in enumerate(case_names):
        case = CASE_DEFS[case_name]
        case_run_id = f"{run_id}_{case_name}"
        case_out_dir = out_dir / case_name
        force_builds = args.force_builds_each or (args.force_builds_first and idx == 0)
        cmd = build_command(
            root=root,
            case_name=case_name,
            case=case,
            args=args,
            run_id=case_run_id,
            case_out_dir=case_out_dir,
            force_builds=force_builds,
        )
        print(f"\n== case {idx + 1}/{len(case_names)}: {case_name} ==")
        print("+ " + " ".join(cmd))
        if args.dry_run:
            continue
        try:
            completed = subprocess.run(cmd, cwd=root, text=True, check=False)
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            rows.append(failure_row(case_name, case, message))
            case_outputs[case_name] = {"error": message}
            if args.halt_on_fail:
                raise
            continue
        summary_csv = case_out_dir / f"{case_run_id}_same_session_compare_summary.csv"
        paired_rows = read_paired_rows(summary_csv)
        if completed.returncode != 0 or not paired_rows:
            message = f"returncode={completed.returncode}; summary_rows={len(paired_rows)}"
            rows.append(failure_row(case_name, case, message))
            case_outputs[case_name] = {"returncode": completed.returncode, "summary_csv": str(summary_csv)}
            if args.halt_on_fail:
                raise SystemExit(message)
            continue
        for paired in paired_rows:
            rows.append(summarize_paired_row(case_name, case, paired))
        case_outputs[case_name] = {"returncode": completed.returncode, "summary_csv": str(summary_csv)}

    payload = {
        "script_build": SCRIPT_BUILD_TAG,
        "run_id": run_id,
        "matrix": args.matrix,
        "cases": case_names,
        "expect_microgemm_suite_build": args.expect_microgemm_suite_build,
        "rows": rows,
        "case_outputs": case_outputs,
    }
    json_path, csv_path, md_path = write_rollup(out_dir, run_id, rows, payload)
    print_rollup(rows)
    print("\nWrote comparison matrix outputs:")
    print(f"  json: {json_path}")
    print(f"  csv:  {csv_path}")
    print(f"  md:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a focused Qwen 3.5 benchmark suite for one single-GPU target.

Qwen 3.5 has a hybrid full-attention + linear-attention path in MegaGemm, so
this runner uses benchmarks/benchmark_qwen35.py instead of the generic
FP16/INT8/HF matrix used for Llama/Qwen 2.5. Quantized Qwen 3.5 linear-attention
loading is not wired yet.

Typical T4 run:

    python benchmarks/run_qwen35_t4_suite.py --modes core

Fast smoke test:

    python benchmarks/run_qwen35_t4_suite.py --modes smoke --model Qwen/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QWEN35_BENCH = ROOT / "benchmarks" / "benchmark_qwen35.py"
MATRIX_BENCH = ROOT / "benchmarks" / "benchmark_inference_matrix.py"


@dataclass(frozen=True)
class Mode:
    name: str
    title: str


MODES: dict[str, Mode] = {
    "compare": Mode("compare", "MegaGemm vs HF greedy compare"),
    "sweep": Mode("sweep", "MegaGemm prompt/output sweep"),
    "batch": Mode("batch", "MegaGemm continuous batching"),
    "matrix": Mode("matrix", "MegaGemm inference matrix"),
}

MODE_ALIASES = {
    "smoke": ("compare",),
    "core": ("compare", "sweep", "batch"),
    "fair": ("matrix",),
}


def shell_join(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def normalize_out_dir(out_dir: str) -> str:
    normalized = out_dir.replace("\\", "/")
    if (
        normalized.startswith("bench_results")
        and normalized != "bench_results"
        and not normalized.startswith("bench_results/")
    ):
        corrected = "bench_results/" + normalized[len("bench_results"):].lstrip("/\\")
        print(
            "Warning: --out-dir starts with 'bench_results' but has no path "
            f"separator after it; using '{corrected}' instead."
        )
        return corrected
    return out_dir


def safe_report_label(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_") or "hardware"


def parse_modes(raw: str) -> list[Mode]:
    names: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part in MODE_ALIASES:
            names.extend(MODE_ALIASES[part])
        else:
            names.append(part)
    if not names:
        raise SystemExit("No benchmark modes selected")

    modes: list[Mode] = []
    seen = set()
    for name in names:
        if name not in MODES:
            valid = sorted([*MODES.keys(), *MODE_ALIASES.keys()])
            raise SystemExit(f"Unknown mode {name!r}. Valid modes: {', '.join(valid)}")
        if name not in seen:
            modes.append(MODES[name])
            seen.add(name)
    return modes


def output_path(args: argparse.Namespace, run_id: str, mode: Mode) -> Path:
    if mode.name == "matrix":
        stem = f"{run_id}_matrix_{args.hardware_label}_{args.matrix_backend}"
        return Path(args.out_dir) / f"{stem}_summary.json"
    return Path(args.out_dir) / f"{run_id}_{mode.name}.json"


def mode_command(args: argparse.Namespace, run_id: str, mode: Mode) -> list[str]:
    out = output_path(args, run_id, mode)
    if mode.name == "matrix":
        dtype = {"float16": "fp16", "bfloat16": "bf16"}[args.dtype]
        cmd = [
            sys.executable,
            str(MATRIX_BENCH),
            "--backend",
            args.matrix_backend,
            "--model",
            args.model,
            "--hardware-label",
            args.hardware_label,
            "--batch-sizes",
            args.matrix_batch_sizes,
            "--prompt-tokens",
            args.matrix_prompt_tokens,
            "--max-new-tokens",
            str(args.matrix_max_new_tokens),
            "--repeats",
            str(args.matrix_repeats),
            "--warmup",
            str(args.warmup),
            "--out-dir",
            str(args.out_dir),
            "--run-id",
            f"{run_id}_matrix",
            "--device",
            args.device,
            "--dtype",
            dtype,
            "--max-seq-len",
            str(args.matrix_max_seq_len),
            "--max-batch-size",
            str(args.matrix_max_batch_size),
            "--num-blocks",
            str(args.matrix_num_blocks),
            "--block-size",
            str(args.matrix_block_size),
            "--kv-alloc",
            args.matrix_kv_alloc,
        ]
        if args.matrix_ignore_eos:
            cmd.append("--ignore-eos")
        if args.matrix_backend == "megagemm-prophet":
            prophet_dir = Path(args.out_dir) / f"{run_id}_matrix_prophet_library"
            cmd.extend(
                [
                    "--prophet-dir",
                    str(prophet_dir),
                    "--prophet-reset-dir",
                    "--prophet-validation-mode",
                    "none",
                    "--prophet-validation-tokens",
                    "4",
                    "--prophet-prefix-tokens",
                    "64",
                    "--prophet-top-k",
                    "3",
                    "--prophet-min-similarity",
                    "0.35",
                    "--prophet-min-prefix-coverage",
                    "0.5",
                    "--prophet-min-prefix-reuse-score",
                    "0.55",
                    "--prophet-max-prefix-rollback-ratio",
                    "0.35",
                    "--prophet-max-prefix-tail-ratio",
                    "0.5",
                    "--prophet-batch-exact-restore",
                    "--prophet-live-prefix-cache",
                ]
            )
        return cmd

    cmd = [
        sys.executable,
        str(QWEN35_BENCH),
        mode.name,
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--out",
        str(out),
    ]
    if mode.name == "compare":
        cmd.extend(
            [
                "--max-new-tokens",
                str(args.compare_max_new_tokens),
                "--num-blocks",
                str(args.compare_num_blocks),
                "--max-seq-len",
                str(args.compare_max_seq_len),
            ]
        )
        if args.skip_hf:
            cmd.append("--skip-hf")
    elif mode.name == "sweep":
        cmd.extend(
            [
                "--prompt-lengths",
                args.sweep_prompt_lengths,
                "--output-lengths",
                args.sweep_output_lengths,
                "--num-blocks",
                str(args.sweep_num_blocks),
                "--max-seq-len",
                str(args.sweep_max_seq_len),
            ]
        )
    elif mode.name == "batch":
        cmd.extend(
            [
                "--batch-sizes",
                args.batch_sizes,
                "--max-new-tokens",
                str(args.batch_max_new_tokens),
                "--num-blocks",
                str(args.batch_num_blocks),
                "--max-seq-len",
                str(args.batch_max_seq_len),
            ]
        )
    return cmd


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_report(args: argparse.Namespace, run_id: str, modes: list[Mode]) -> Path:
    hardware_label = safe_report_label(args.hardware_label)
    report = Path(args.out_dir) / f"{run_id}_qwen35_{hardware_label}_report.md"
    lines = [
        f"# Qwen 3.5 {args.hardware_label} Benchmark",
        "",
        "## Run Configuration",
        "",
        f"- Hardware label: `{args.hardware_label}`",
        f"- Model: `{args.model}`",
        f"- Device: `{args.device}`",
        f"- Dtype: `{args.dtype}`",
        f"- Runs: `{args.runs}`",
        f"- Warmup: `{args.warmup}`",
        f"- Run id: `{run_id}`",
        "",
        "## Important Methodology Note",
        "",
        "Qwen 3.5 uses MegaGemm's native hybrid full-attention + linear-attention path.",
        "The core comparison here is FP16/BF16 native performance, not INT8/AWQ.",
        "",
    ]

    compare = load_json(output_path(args, run_id, MODES["compare"]))
    if compare:
        mg = (compare.get("megagemm") or {}).get("summary") or {}
        hf = (compare.get("huggingface") or {}).get("summary") or {}
        gap = compare.get("gap") or {}
        lines.extend(
            [
                "## Compare",
                "",
                "| Backend | Prompt tok | Output tok | TTFT ms | Decode tok/s | Total tok/s | Peak VRAM MB | Model VRAM MB | KV reserved MB |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                (
                    f"| MegaGemm | {fmt(mg.get('prompt_tokens'))} | {fmt(mg.get('output_tokens'))} | "
                    f"{fmt(mg.get('ttft_ms'))} | {fmt(mg.get('decode_tps'))} | "
                    f"{fmt(mg.get('total_tps'))} | {fmt(mg.get('peak_vram_mb'))} | "
                    f"{fmt(mg.get('model_vram_mb'))} | {fmt(mg.get('kv_reserved_mb'))} |"
                ),
            ]
        )
        if hf:
            lines.append(
                f"| HF | {fmt(hf.get('prompt_tokens'))} | {fmt(hf.get('output_tokens'))} | "
                f"{fmt(hf.get('ttft_ms'))} | {fmt(hf.get('decode_tps'))} | "
                f"{fmt(hf.get('total_tps'))} | {fmt(hf.get('peak_vram_mb'))} | "
                f"{fmt(hf.get('model_vram_mb'))} |  |"
            )
        else:
            hf_error = (compare.get("huggingface") or {}).get("error")
            lines.append(f"| HF | skipped/error |  |  |  |  |  |  | `{hf_error or 'n/a'}` |")
        if gap:
            lines.extend(
                [
                    "",
                    f"- Decode TPS ratio MegaGemm/HF: `{gap.get('decode_tps_ratio_mg_over_hf')}`",
                    f"- Total TPS ratio MegaGemm/HF: `{gap.get('total_tps_ratio_mg_over_hf')}`",
                    f"- TTFT ratio MegaGemm/HF: `{gap.get('ttft_ratio_mg_over_hf')}`",
                    "",
                ]
            )

    sweep = load_json(output_path(args, run_id, MODES["sweep"]))
    if sweep:
        lines.extend(
            [
                "## Sweep",
                "",
                "| Target prompt | Actual prompt | Output tok | TTFT ms | Decode tok/s | Total tok/s | Peak VRAM MB |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sweep.get("rows", []):
            lines.append(
                f"| {row.get('target_prompt_tokens')} | {fmt(row.get('prompt_tokens'))} | "
                f"{row.get('output_tokens')} | {fmt(row.get('ttft_ms'))} | "
                f"{fmt(row.get('decode_tps'))} | {fmt(row.get('total_tps'))} | "
                f"{fmt(row.get('peak_vram_mb'))} |"
            )
        lines.append("")

    batch = load_json(output_path(args, run_id, MODES["batch"]))
    if batch:
        lines.extend(
            [
                "## Batch",
                "",
                "| Batch | Total tok/s | Per-seq tok/s | Total ms | Peak VRAM MB | Error |",
                "|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in batch.get("rows", []):
            lines.append(
                f"| {row.get('batch_size')} | {fmt(row.get('total_tps'))} | "
                f"{fmt(row.get('per_seq_tps'))} | {fmt(row.get('total_ms'))} | "
                f"{fmt(row.get('peak_vram_mb'))} | {row.get('error', '')} |"
            )
        lines.append("")

    matrix = load_json(output_path(args, run_id, MODES["matrix"]))
    if matrix:
        lines.extend(
            [
                "## Matrix",
                "",
                "| Backend | Scenario | Batch | Prompt tok/req | Max new tok/req | First tok/s | Median tok/s | Decode tok/s | OK |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in matrix.get("rows", []):
            lines.append(
                f"| {row.get('backend')} | {row.get('scenario')} | {row.get('batch_size')} | "
                f"{row.get('prompt_tokens_requested_per_request')} | {row.get('max_new_tokens_per_request')} | "
                f"{fmt(row.get('first_output_tps'))} | {fmt(row.get('median_output_tps'))} | "
                f"{fmt(row.get('median_scheduler_decode_tps') or row.get('median_decode_wall_tps'))} | "
                f"{row.get('ok_samples')}/{row.get('samples')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Files",
            "",
        ]
    )
    for mode in modes:
        lines.append(f"- {mode.title}: `{output_path(args, run_id, mode)}`")

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(
    *,
    default_hardware_label: str = "1xt4",
    default_out_dir: str = "bench_results/qwen35_t4",
    default_run_prefix: str = "qwen35_t4",
) -> int:
    parser = argparse.ArgumentParser(description="Qwen 3.5 single-GPU benchmark suite")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--modes",
        default="core",
        help="smoke, core, fair, or comma-separated compare,sweep,batch,matrix",
    )
    parser.add_argument("--hardware-label", default=default_hardware_label)
    parser.add_argument("--out-dir", default=default_out_dir)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--skip-hf", action="store_true", help="Skip HF baseline in compare mode")
    parser.add_argument("--compare-max-new-tokens", type=int, default=128)
    parser.add_argument("--compare-num-blocks", type=int, default=512)
    parser.add_argument("--compare-max-seq-len", type=int, default=1024)

    parser.add_argument("--sweep-prompt-lengths", default="128,512,1024,2048")
    parser.add_argument("--sweep-output-lengths", default="64,128")
    parser.add_argument("--sweep-num-blocks", type=int, default=0)
    parser.add_argument("--sweep-max-seq-len", type=int, default=4096)

    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--batch-max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-num-blocks", type=int, default=1024)
    parser.add_argument("--batch-max-seq-len", type=int, default=2048)

    parser.add_argument("--matrix-batch-sizes", default="8")
    parser.add_argument("--matrix-backend", default="megagemm-prophet", choices=["megagemm", "megagemm-prophet"])
    parser.add_argument("--matrix-prompt-tokens", default="2048")
    parser.add_argument("--matrix-max-new-tokens", type=int, default=128)
    parser.add_argument("--matrix-repeats", type=int, default=5)
    parser.add_argument("--matrix-max-seq-len", type=int, default=4096)
    parser.add_argument("--matrix-max-batch-size", type=int, default=8)
    parser.add_argument("--matrix-num-blocks", type=int, default=0)
    parser.add_argument("--matrix-block-size", type=int, default=128)
    parser.add_argument("--matrix-kv-alloc", default="auto", choices=["auto", "greedy"])
    parser.add_argument("--matrix-allow-eos", dest="matrix_ignore_eos", action="store_false")
    parser.set_defaults(matrix_ignore_eos=True)
    args = parser.parse_args()

    modes = parse_modes(args.modes)
    args.out_dir = normalize_out_dir(args.out_dir)
    run_id = args.run_id or f"{default_run_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    args.out_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Qwen 3.5 {args.hardware_label} benchmark suite")
    print(f"  run_id: {run_id}")
    print(f"  model:  {args.model}")
    print(f"  modes:  {', '.join(mode.name for mode in modes)}")
    print(f"  out:    {args.out_dir}")

    for mode in modes:
        cmd = mode_command(args, run_id, mode)
        print()
        print(f"=== {mode.title} ===")
        print(shell_join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(ROOT), check=True)

    if args.dry_run:
        print("\nDry run only; no report was generated from results.")
        return 0

    report = write_report(args, run_id, modes)
    print()
    print(f"Wrote combined report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a focused single-model benchmark suite for one T4-class GPU.

This is a thin orchestrator over benchmark_inference_matrix.py. It runs a
small set of comparable modes and writes a combined Markdown report.

Typical Llama 3.2 3B T4 run:

    python benchmarks/run_llama32_t4_suite.py --modes core

Fuller run with KV offload and HF sequential comparison:

    python benchmarks/run_llama32_t4_suite.py --modes full
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
MATRIX = ROOT / "benchmarks" / "benchmark_inference_matrix.py"


@dataclass(frozen=True)
class Mode:
    name: str
    title: str
    backend: str
    extra_args: tuple[str, ...] = ()


MODES: dict[str, Mode] = {
    "megagemm-fp16": Mode(
        name="megagemm-fp16",
        title="MegaGemm FP16",
        backend="megagemm",
    ),
    "megagemm-int8": Mode(
        name="megagemm-int8",
        title="MegaGemm INT8",
        backend="megagemm",
        extra_args=("--quantize", "int8"),
    ),
    "megagemm-kv-offload": Mode(
        name="megagemm-kv-offload",
        title="MegaGemm FP16 + KV offload",
        backend="megagemm",
        extra_args=("--kv-offload", "--num-cpu-blocks", "4096", "--gpu-window", "64"),
    ),
    "hf-batched": Mode(
        name="hf-batched",
        title="HF Transformers batched",
        backend="hf",
        extra_args=("--hf-mode", "batched"),
    ),
    "hf-sequential": Mode(
        name="hf-sequential",
        title="HF Transformers sequential",
        backend="hf",
        extra_args=("--hf-mode", "sequential"),
    ),
}

MODE_ALIASES = {
    "core": ("megagemm-fp16", "megagemm-int8", "hf-batched"),
    "full": (
        "megagemm-fp16",
        "megagemm-int8",
        "megagemm-kv-offload",
        "hf-batched",
        "hf-sequential",
    ),
}


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

    modes = []
    seen = set()
    for name in names:
        if name not in MODES:
            valid = sorted([*MODES.keys(), *MODE_ALIASES.keys()])
            raise SystemExit(f"Unknown mode {name!r}. Valid modes: {', '.join(valid)}")
        if name not in seen:
            modes.append(MODES[name])
            seen.add(name)
    return modes


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit(f"Expected at least one integer in {raw!r}")
    return values


def shell_join(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def run_mode(args: argparse.Namespace, mode: Mode, run_id: str) -> dict[str, Path]:
    mode_run_id = f"{run_id}_{mode.name}"
    cmd = [
        sys.executable,
        str(MATRIX),
        "--backend",
        mode.backend,
        "--model",
        args.model,
        "--hardware-label",
        args.hardware_label,
        "--batch-sizes",
        args.batch_sizes,
        "--prompt-tokens",
        args.prompt_tokens,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--out-dir",
        str(args.out_dir),
        "--run-id",
        mode_run_id,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--max-seq-len",
        str(args.max_seq_len),
        "--max-batch-size",
        str(args.max_batch_size),
        "--num-blocks",
        str(args.num_blocks),
        "--block-size",
        str(args.block_size),
        "--kv-alloc",
        args.kv_alloc,
    ]
    if args.cache_dir:
        cmd.extend(["--cache-dir", args.cache_dir])
    if args.tokenizer:
        cmd.extend(["--tokenizer", args.tokenizer])
    if args.local_files_only:
        cmd.append("--local-files-only")
    if not args.allow_eos:
        cmd.append("--ignore-eos")
    cmd.extend(mode.extra_args)

    print()
    print(f"=== {mode.title} ===")
    print(shell_join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)

    stem = f"{mode_run_id}_{args.hardware_label}_{mode.backend}"
    out_dir = Path(args.out_dir)
    return {
        "raw": out_dir / f"{stem}.jsonl",
        "summary": out_dir / f"{stem}_summary.json",
        "csv": out_dir / f"{stem}_summary.csv",
    }


def raw_peak_by_key(path: Path) -> dict[tuple[Any, ...], float]:
    peaks: dict[tuple[Any, ...], float] = {}
    if not path.exists():
        return peaks
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("ok"):
                continue
            mem = row.get("cuda_memory") or {}
            peak = mem.get("peak_allocated_gb")
            if peak is None:
                continue
            key = (
                row.get("scenario"),
                int(row.get("batch_size", 0)),
                int(row.get("prompt_tokens_requested_per_request", 0)),
                int(row.get("max_new_tokens_per_request", 0)),
                bool(row.get("kv_offload")),
            )
            peaks[key] = max(float(peak), peaks.get(key, 0.0))
    return peaks


def load_summary_rows(mode: Mode, paths: dict[str, Path]) -> list[dict[str, Any]]:
    summary_path = paths["summary"]
    if not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    peaks = raw_peak_by_key(paths["raw"])
    rows = []
    for row in payload.get("rows", []):
        key = (
            row.get("scenario"),
            int(row.get("batch_size", 0)),
            int(row.get("prompt_tokens_requested_per_request", 0)),
            int(row.get("max_new_tokens_per_request", 0)),
            bool(row.get("kv_offload")),
        )
        enriched = dict(row)
        enriched["mode"] = mode.name
        enriched["mode_title"] = mode.title
        enriched["peak_allocated_gb"] = peaks.get(key)
        rows.append(enriched)
    return rows


def write_markdown_report(
    *,
    args: argparse.Namespace,
    run_id: str,
    mode_paths: list[tuple[Mode, dict[str, Path]]],
) -> Path:
    all_rows: list[dict[str, Any]] = []
    for mode, paths in mode_paths:
        all_rows.extend(load_summary_rows(mode, paths))

    report_path = Path(args.out_dir) / f"{run_id}_{args.suite_slug}_report.md"
    lines = [
        f"# {args.suite_title} Benchmark",
        "",
        "## Run Configuration",
        "",
        f"- Model: `{args.model}`",
        f"- Hardware label: `{args.hardware_label}`",
        f"- Dtype: `{args.dtype}`",
        f"- Batch sizes: `{args.batch_sizes}`",
        f"- Prompt tokens/request: `{args.prompt_tokens}`",
        f"- Max new tokens/request: `{args.max_new_tokens}`",
        f"- Repeats: `{args.repeats}`",
        f"- Fixed decode length: `{not args.allow_eos}`",
        f"- Run id: `{run_id}`",
        "",
        "## Summary",
        "",
        "| Mode | Scenario | Batch | Prompt tok/req | Output tok/s median | Decode tok/s median | Prefill tok/s median | Peak GB | OK |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    if all_rows:
        all_rows.sort(
            key=lambda r: (
                str(r.get("mode")),
                str(r.get("scenario")),
                int(r.get("prompt_tokens_requested_per_request", 0)),
                int(r.get("batch_size", 0)),
            )
        )
        for row in all_rows:
            peak = row.get("peak_allocated_gb")
            peak_s = "" if peak is None else f"{float(peak):.2f}"
            lines.append(
                "| {mode} | {scenario} | {batch} | {prompt} | {out:.2f} | {decode:.2f} | {prefill:.2f} | {peak} | {ok}/{samples} |".format(
                    mode=row.get("mode_title"),
                    scenario=row.get("scenario"),
                    batch=row.get("batch_size"),
                    prompt=row.get("prompt_tokens_requested_per_request"),
                    out=float(row.get("median_output_tps") or 0.0),
                    decode=float(row.get("median_decode_wall_tps") or 0.0),
                    prefill=float(row.get("median_prefill_tps") or 0.0),
                    peak=peak_s,
                    ok=row.get("ok_samples"),
                    samples=row.get("samples"),
                )
            )
    else:
        lines.append("| No completed rows found | | | | | | | | |")

    lines.extend(
        [
            "",
            "## Files",
            "",
        ]
    )
    for mode, paths in mode_paths:
        lines.append(f"- {mode.title}:")
        for label, path in paths.items():
            lines.append(f"  - {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Notes To Fill In",
            "",
            "- Did any scenario hit CUDA OOM?",
            "- Which batch/context pair is the practical T4 limit?",
            "- Does INT8 save enough VRAM to justify throughput cost?",
            "- Does KV offload help capacity without hurting decode too much?",
            "- Is MegaGemm faster than HF only at batch > 1, or also single request?",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-model T4 benchmark suite")
    parser.add_argument("--suite-title", default="Llama 3.2 3B T4")
    parser.add_argument("--suite-slug", default="llama32_t4")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--hardware-label", default="1xt4")
    parser.add_argument("--modes", default="core", help="core, full, or comma-separated mode names")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--prompt-tokens", default="128,512,1024,2048")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out-dir", default="bench_results/llama32_t4")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=0, help="0 means max(--batch-sizes)")
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-alloc", default="auto", choices=["auto", "greedy"])
    parser.add_argument("--allow-eos", action="store_true", help="Allow early EOS instead of fixed decode length")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    modes = parse_modes(args.modes)
    run_id = args.run_id or f"{args.suite_slug}_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.max_batch_size <= 0:
        args.max_batch_size = max(parse_csv_ints(args.batch_sizes))
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    args.out_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{args.suite_title} benchmark suite")
    print(f"  run_id: {run_id}")
    print(f"  modes:  {', '.join(mode.name for mode in modes)}")
    print(f"  out:    {args.out_dir}")

    mode_paths: list[tuple[Mode, dict[str, Path]]] = []
    for mode in modes:
        paths = run_mode(args, mode, run_id)
        mode_paths.append((mode, paths))

    if args.dry_run:
        print("\nDry run only; no report was generated from results.")
        return 0

    report_path = write_markdown_report(args=args, run_id=run_id, mode_paths=mode_paths)
    print()
    print(f"Wrote combined report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

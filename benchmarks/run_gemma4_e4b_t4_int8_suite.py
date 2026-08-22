"""Run a focused Gemma 4 E4B INT8 benchmark suite for one T4-class GPU.

The suite combines:
  - an inference matrix over batch/context settings;
  - a decode profiler pass to expose where time is spent.

Typical T4 run:

    python benchmarks/run_gemma4_e4b_t4_int8_suite.py --modes core
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
PROFILE = ROOT / "benchmarks" / "profile_gemma4_decode.py"


@dataclass(frozen=True)
class Mode:
    name: str
    title: str


MODES: dict[str, Mode] = {
    "matrix": Mode("matrix", "MegaGemm INT8 inference matrix"),
    "profile": Mode("profile", "MegaGemm INT8 decode profile"),
}

MODE_ALIASES = {
    "core": ("matrix", "profile"),
    "matrix-only": ("matrix",),
    "profile-only": ("profile",),
}


def shell_join(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


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


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit(f"Expected at least one integer in {raw!r}")
    return values


def matrix_run_id(run_id: str) -> str:
    return f"{run_id}_matrix"


def profile_output_path(args: argparse.Namespace, run_id: str) -> Path:
    return Path(args.out_dir) / f"{run_id}_profile.json"


def matrix_paths(args: argparse.Namespace, run_id: str) -> dict[str, Path]:
    stem = f"{matrix_run_id(run_id)}_{args.hardware_label}_megagemm"
    out_dir = Path(args.out_dir)
    return {
        "raw": out_dir / f"{stem}.jsonl",
        "summary": out_dir / f"{stem}_summary.json",
        "csv": out_dir / f"{stem}_summary.csv",
    }


def matrix_command(args: argparse.Namespace, run_id: str) -> list[str]:
    cmd = [
        sys.executable,
        str(MATRIX),
        "--backend",
        "megagemm",
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
        matrix_run_id(run_id),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--quantize",
        "int8",
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
    return cmd


def profile_command(args: argparse.Namespace, run_id: str) -> list[str]:
    cmd = [
        sys.executable,
        str(PROFILE),
        "--model",
        args.model,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--quantize",
        "int8",
        "--batch-size",
        str(args.profile_batch_size),
        "--max-new-tokens",
        str(args.profile_max_new_tokens),
        "--warmup-tokens",
        str(args.profile_warmup_tokens),
        "--max-seq-len",
        str(args.profile_max_seq_len),
        "--max-batch-size",
        str(args.profile_max_batch_size),
        "--num-blocks",
        str(args.profile_num_blocks),
        "--kv-alloc",
        args.profile_kv_alloc,
        "--prompt",
        args.profile_prompt,
        "--out",
        str(profile_output_path(args, run_id)),
    ]
    return cmd


def child_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.disable_int8_dequant_reuse:
        env["MEGAGEMM_INT8_DEQUANT_REUSE"] = "0"
    if args.enable_int8_triton_fused:
        env["MEGAGEMM_INT8_TRITON_FUSED"] = "1"
    return env


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
            )
            peaks[key] = max(float(peak), peaks.get(key, 0.0))
    return peaks


def matrix_rows(args: argparse.Namespace, run_id: str) -> list[dict[str, Any]]:
    paths = matrix_paths(args, run_id)
    payload = load_json(paths["summary"])
    if not payload:
        return []
    peaks = raw_peak_by_key(paths["raw"])
    rows = []
    for row in payload.get("rows", []):
        key = (
            row.get("scenario"),
            int(row.get("batch_size", 0)),
            int(row.get("prompt_tokens_requested_per_request", 0)),
            int(row.get("max_new_tokens_per_request", 0)),
        )
        enriched = dict(row)
        enriched["peak_allocated_gb"] = peaks.get(key)
        rows.append(enriched)
    return rows


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_report(args: argparse.Namespace, run_id: str, modes: list[Mode]) -> Path:
    report = Path(args.out_dir) / f"{run_id}_gemma4_e4b_t4_int8_report.md"
    matrix = matrix_rows(args, run_id)
    profile = load_json(profile_output_path(args, run_id)) or {}
    profile_summary = profile.get("summary") or {}

    lines = [
        "# Gemma 4 E4B INT8 T4 Benchmark",
        "",
        "## Run Configuration",
        "",
        f"- Model: `{args.model}`",
        f"- Hardware label: `{args.hardware_label}`",
        f"- Device: `{args.device}`",
        f"- Activation dtype: `{args.dtype}`",
        "- Weight mode: `INT8 W8A16`",
        f"- Batch sizes: `{args.batch_sizes}`",
        f"- Prompt tokens/request: `{args.prompt_tokens}`",
        f"- Max new tokens/request: `{args.max_new_tokens}`",
        f"- Repeats: `{args.repeats}`",
        f"- Run id: `{run_id}`",
        f"- INT8 dequant reuse disabled: `{args.disable_int8_dequant_reuse}`",
        f"- INT8 Triton fused requested: `{args.enable_int8_triton_fused}`",
        "",
        "## Matrix",
        "",
        "| Scenario | Batch | Prompt tok/req | Output tok/s median | Decode tok/s median | Prefill tok/s median | Peak GB | OK |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    if matrix:
        matrix.sort(
            key=lambda row: (
                str(row.get("scenario")),
                int(row.get("prompt_tokens_requested_per_request", 0)),
                int(row.get("batch_size", 0)),
            )
        )
        for row in matrix:
            peak = row.get("peak_allocated_gb")
            lines.append(
                "| {scenario} | {batch} | {prompt} | {out:.2f} | {decode:.2f} | {prefill:.2f} | {peak} | {ok}/{samples} |".format(
                    scenario=row.get("scenario"),
                    batch=row.get("batch_size"),
                    prompt=row.get("prompt_tokens_requested_per_request"),
                    out=float(row.get("median_output_tps") or 0.0),
                    decode=float(row.get("median_decode_wall_tps") or 0.0),
                    prefill=float(row.get("median_prefill_tps") or 0.0),
                    peak="" if peak is None else f"{float(peak):.2f}",
                    ok=row.get("ok_samples"),
                    samples=row.get("samples"),
                )
            )
    else:
        lines.append("| No completed matrix rows found | | | | | | | |")

    if profile_summary:
        lines.extend(
            [
                "",
                "## Decode Profile",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| decode_total_ms | {fmt(profile_summary.get('decode_total_ms'))} |",
                f"| decode_attn_ms | {fmt(profile_summary.get('decode_attn_ms'))} |",
                f"| decode_mlp_ms | {fmt(profile_summary.get('decode_mlp_ms'))} |",
                f"| decode_ple_ms | {fmt(profile_summary.get('decode_ple_ms'))} |",
                f"| decode_lm_head_ms | {fmt(profile_summary.get('decode_lm_head_ms'))} |",
                f"| cpu_launch_ms | {fmt(profile_summary.get('cpu_launch_ms'))} |",
                f"| cuda_attn_ms | {fmt(profile_summary.get('cuda_attn_ms'))} |",
                f"| cuda_deepfusion_ms | {fmt(profile_summary.get('cuda_deepfusion_ms'))} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
        ]
    )
    if any(mode.name == "matrix" for mode in modes):
        paths = matrix_paths(args, run_id)
        for label, path in paths.items():
            lines.append(f"- Matrix {label}: `{path}`")
    if any(mode.name == "profile" for mode in modes):
        lines.append(f"- Decode profile JSON: `{profile_output_path(args, run_id)}`")

    lines.extend(
        [
            "",
            "## Notes To Fill In",
            "",
            "- Did the E4B INT8 model fit comfortably on T4?",
            "- Which batch/context row becomes the practical serving limit?",
            "- How much of decode time is attention vs MLP vs PLE?",
            "- Does long-context throughput stay stable enough for real use?",
            "- Are there any recurrent non-OOM failures in the raw JSONL?",
        ]
    )

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma 4 E4B INT8 T4 benchmark suite")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--hardware-label", default="1xt4")
    parser.add_argument("--modes", default="core", help="core, matrix-only, profile-only, or comma-separated matrix,profile")
    parser.add_argument("--out-dir", default="bench_results/gemma4_e4b_t4_int8")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--prompt-tokens", default="128,512,1024")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-batch-size", type=int, default=0, help="0 means max(--batch-sizes)")
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-alloc", default="auto", choices=["auto", "greedy"])
    parser.add_argument("--allow-eos", action="store_true")

    parser.add_argument("--profile-batch-size", type=int, default=1)
    parser.add_argument("--profile-max-new-tokens", type=int, default=64)
    parser.add_argument("--profile-warmup-tokens", type=int, default=8)
    parser.add_argument("--profile-max-seq-len", type=int, default=2048)
    parser.add_argument("--profile-max-batch-size", type=int, default=1)
    parser.add_argument("--profile-num-blocks", type=int, default=128)
    parser.add_argument("--profile-kv-alloc", default="auto", choices=["auto", "greedy"])
    parser.add_argument(
        "--profile-prompt",
        default="Explique KV cache em uma frase e depois diga por que sliding attention existe.",
    )
    parser.add_argument(
        "--keep-int8-dequant-reuse",
        dest="disable_int8_dequant_reuse",
        action="store_false",
        help="Keep the default W8A16 reusable FP16 dequant buffers. This is usually too VRAM-hungry for E4B on T4.",
    )
    parser.add_argument(
        "--disable-int8-triton-fused",
        dest="enable_int8_triton_fused",
        action="store_false",
        help="Do not request the fused Triton INT8 path before falling back to dequant mode.",
    )
    parser.set_defaults(
        disable_int8_dequant_reuse=True,
        enable_int8_triton_fused=True,
    )
    args = parser.parse_args()

    modes = parse_modes(args.modes)
    run_id = args.run_id or f"gemma4_e4b_t4_int8_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.max_batch_size <= 0:
        args.max_batch_size = max(parse_csv_ints(args.batch_sizes))

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    args.out_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Gemma 4 E4B INT8 T4 benchmark suite")
    print(f"  run_id: {run_id}")
    print(f"  model:  {args.model}")
    print(f"  modes:  {', '.join(mode.name for mode in modes)}")
    print(f"  out:    {args.out_dir}")
    print(f"  int8 dequant reuse disabled: {args.disable_int8_dequant_reuse}")
    print(f"  int8 triton fused requested: {args.enable_int8_triton_fused}")

    for mode in modes:
        if mode.name == "matrix":
            cmd = matrix_command(args, run_id)
        elif mode.name == "profile":
            cmd = profile_command(args, run_id)
        else:
            raise AssertionError(f"Unhandled mode: {mode.name}")

        print()
        print(f"=== {mode.title} ===")
        print(shell_join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(ROOT), check=True, env=child_env(args))

    if args.dry_run:
        print("\nDry run only; no report was generated from results.")
        return 0

    report_path = write_report(args, run_id, modes)
    print()
    print(f"Wrote combined report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

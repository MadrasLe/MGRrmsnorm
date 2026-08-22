"""Focused Qwen 2.5 MGX+Prophet sweep for NVIDIA L4.

This wrapper reuses the tuned matrix runner and varies only the knobs that are
most likely to move when leaving T4: paged-KV block size and, optionally, decode
warps. It ranks candidates by cache-hit output throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATRIX_RUNNER = ROOT / "benchmarks" / "run_qwen25_mgx_prophet_matrix_t4.py"
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


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise SystemExit("At least one block size is required")
    return values


def parse_warps(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip().lower()
        if not part:
            continue
        values.append(0 if part == "auto" else int(part))
    if not values:
        raise SystemExit("At least one warps value is required")
    return values


def case_name(block_size: int, warps: int) -> str:
    return f"block{block_size}_w{warps if warps > 0 else 'auto'}"


def summary_path(out_dir: Path, run_id: str, hardware_label: str) -> Path:
    stem = f"{run_id}_prophet-repeat_{hardware_label}_megagemm-prophet"
    return out_dir / f"{stem}_summary.json"


def load_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if not rows:
        return None
    return dict(rows[0])


def build_matrix_cmd(
    args: argparse.Namespace,
    *,
    run_id: str,
    out_dir: Path,
    block_size: int,
    warps: int,
    max_new_tokens: int,
    repeats: int,
    decode_timing_detail: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(MATRIX_RUNNER),
        "--modes",
        "prophet-repeat",
        "--model",
        args.model,
        "--hardware-label",
        args.hardware_label,
        "--batch-sizes",
        args.batch_sizes,
        "--prompt-tokens",
        args.prompt_tokens,
        "--max-new-tokens",
        str(max_new_tokens),
        "--repeats",
        str(repeats),
        "--warmup",
        str(args.warmup),
        "--dtype",
        args.dtype,
        "--quantize",
        args.quantize,
        "--max-seq-len",
        str(args.max_seq_len),
        "--max-batch-size",
        str(args.max_batch_size),
        "--block-size",
        str(block_size),
        "--out-dir",
        str(out_dir),
        "--run-id",
        run_id,
    ]
    if args.mgx:
        cmd.extend(["--mgx", args.mgx])
    if warps > 0:
        cmd.extend(["--paged-decode-warps", str(warps)])
    if decode_timing_detail:
        cmd.extend(["--decode-timing", "--decode-timing-detail"])
    if args.cache_dir:
        cmd.extend(["--cache-dir", args.cache_dir])
    if args.mgx_payload_cache_dir:
        cmd.extend(["--mgx-payload-cache-dir", args.mgx_payload_cache_dir])
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def run_case(
    args: argparse.Namespace,
    *,
    base_run_id: str,
    block_size: int,
    warps: int,
    max_new_tokens: int,
    repeats: int,
    decode_timing_detail: bool,
    out_dir: Path,
) -> dict[str, Any]:
    name = case_name(block_size, warps)
    case_run_id = f"{base_run_id}_{name}"
    case_out = out_dir / name
    cmd = build_matrix_cmd(
        args,
        run_id=case_run_id,
        out_dir=case_out,
        block_size=block_size,
        warps=warps,
        max_new_tokens=max_new_tokens,
        repeats=repeats,
        decode_timing_detail=decode_timing_detail,
    )

    print()
    print(f"=== L4 sweep case {name} ===")
    print(shell_join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)

    result = load_result(summary_path(case_out, case_run_id, args.hardware_label))
    return {
        "case": name,
        "block_size": block_size,
        "warps": warps if warps > 0 else "auto",
        "out_dir": str(case_out),
        "run_id": case_run_id,
        "summary_path": str(summary_path(case_out, case_run_id, args.hardware_label)),
        "result": result,
    }


def metric(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    return float(row.get("median_steady_output_tps") or row.get("median_output_tps") or 0.0)


def write_report(
    *,
    out_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    sweep_rows: list[dict[str, Any]],
    full_row: dict[str, Any] | None,
) -> Path:
    report = out_dir / f"{run_id}_qwen25_mgx_prophet_matrix_l4_report.md"
    lines = [
        "# Qwen-family MGX Prophet L4 Sweep",
        "",
        f"- Run id: `{run_id}`",
        f"- Hardware label: `{args.hardware_label}`",
        f"- Model: `{args.model}`",
        f"- MGX: `{args.mgx}`",
        f"- Batch sizes: `{args.batch_sizes}`",
        f"- Prompt tokens: `{args.prompt_tokens}`",
        f"- Sweep max new tokens: `{args.sweep_max_new_tokens}`",
        f"- Sweep repeats: `{args.sweep_repeats}`",
        f"- Full enabled: `{args.run_full}`",
        "",
        "## Sweep",
        "",
        "| Case | Block | Warps | First tok/s | Cache-hit tok/s | Decode hit tok/s | Overall tok/s | OK |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sweep_rows:
        row = item.get("result") or {}
        lines.append(
            "| {case} | {block} | {warps} | {first:.2f} | {steady:.2f} | {decode:.2f} | {overall:.2f} | {ok}/{samples} |".format(
                case=item["case"],
                block=item["block_size"],
                warps=item["warps"],
                first=float(row.get("first_output_tps") or 0.0),
                steady=float(row.get("median_steady_output_tps") or 0.0),
                decode=float(row.get("median_steady_prophet_decode_tps") or 0.0),
                overall=float(row.get("median_output_tps") or 0.0),
                ok=int(row.get("ok_samples") or 0),
                samples=int(row.get("samples") or 0),
            )
        )

    best = max(sweep_rows, key=lambda item: metric(item.get("result")), default=None)
    lines.extend(["", "## Decision", ""])
    if best and metric(best.get("result")) > 0:
        lines.append(
            f"- Best sweep case: `{best['case']}` at `{metric(best.get('result')):.2f}` cache-hit tok/s."
        )
    else:
        lines.append("- Best sweep case: unavailable; no summary rows were found.")

    if full_row:
        lines.extend(
            [
                "",
                "## Full Winner Run",
                "",
                (
                    "- Cache-hit output tok/s: "
                    f"`{float(full_row.get('median_steady_output_tps') or 0.0):.2f}`"
                ),
                (
                    "- Overall median output tok/s: "
                    f"`{float(full_row.get('median_output_tps') or 0.0):.2f}`"
                ),
                (
                    "- First output tok/s: "
                    f"`{float(full_row.get('first_output_tps') or 0.0):.2f}`"
                ),
            ]
        )

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen2.5 MGX Prophet L4 block-size sweep")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--mgx",
        default="",
        help="Optional MGX artifact path. Default lets the matrix runner derive it from --model/--dtype.",
    )
    parser.add_argument("--hardware-label", default="1xl4")
    parser.add_argument("--out-dir", default="bench_results/qwen25_mgx_fp16_l4_sweep")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--batch-sizes", default="8")
    parser.add_argument("--prompt-tokens", default="2048")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quantize", default="none", choices=["none", "int8", "fp8", "awq"])
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--block-sizes", default="32,64,128")
    parser.add_argument(
        "--warps",
        default="auto,4",
        help=(
            "Comma list: auto,4,8. L4 long-context sweeps have favored 4-warps "
            "so far, so the default tests both auto and 4."
        ),
    )
    parser.add_argument("--sweep-max-new-tokens", type=int, default=32)
    parser.add_argument("--sweep-repeats", type=int, default=2)
    parser.add_argument("--full-max-new-tokens", type=int, default=128)
    parser.add_argument("--full-repeats", type=int, default=5)
    parser.add_argument(
        "--run-full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After the sweep, run a full 128-token matrix for the best case.",
    )
    parser.add_argument(
        "--decode-timing-detail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable detailed decode timing during sweep cases.",
    )
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--mgx-payload-cache-dir", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.out_dir = normalize_out_dir(args.out_dir)
    block_sizes = parse_int_list(args.block_sizes)
    warps_values = parse_warps(args.warps)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("qwen25_mgx_prophet_l4_%Y%m%d_%H%M%S")

    print("Qwen-family MGX Prophet L4 sweep")
    print(f"  run_id: {run_id}")
    print(f"  model:  {args.model}")
    print(f"  mgx:    {args.mgx or '(auto)'}")
    print(f"  out:    {out_dir}")
    print(f"  blocks: {block_sizes}")
    print(f"  warps:  {[w if w > 0 else 'auto' for w in warps_values]}")

    sweep_rows: list[dict[str, Any]] = []
    for block_size in block_sizes:
        for warps in warps_values:
            sweep_rows.append(
                run_case(
                    args,
                    base_run_id=run_id,
                    block_size=block_size,
                    warps=warps,
                    max_new_tokens=args.sweep_max_new_tokens,
                    repeats=args.sweep_repeats,
                    decode_timing_detail=args.decode_timing_detail,
                    out_dir=out_dir,
                )
            )

    full_row = None
    if args.run_full and not args.dry_run:
        best = max(sweep_rows, key=lambda item: metric(item.get("result")), default=None)
        if best and metric(best.get("result")) > 0:
            print()
            print(f"Best L4 sweep case: {best['case']} ({metric(best.get('result')):.2f} tok/s)")
            full = run_case(
                args,
                base_run_id=f"{run_id}_full",
                block_size=int(best["block_size"]),
                warps=0 if best["warps"] == "auto" else int(best["warps"]),
                max_new_tokens=args.full_max_new_tokens,
                repeats=args.full_repeats,
                decode_timing_detail=False,
                out_dir=out_dir,
            )
            full_row = full.get("result")
        else:
            print("No successful sweep result found; skipping full run.")

    report = write_report(
        out_dir=out_dir,
        run_id=run_id,
        args=args,
        sweep_rows=sweep_rows,
        full_row=full_row,
    )
    print(f"\nWrote L4 sweep report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

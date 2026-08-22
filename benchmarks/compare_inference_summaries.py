"""Compare two benchmark_inference_matrix summary files.

Example:
    python benchmarks/compare_inference_summaries.py \
      --left "bench_results/l4_prophet_1024blocks/*prophet-warm*_summary.json" \
      --right "bench_results/l4_vllm/*_summary.json"
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


def resolve_summary(raw: str, *, label: str) -> Path:
    matches = [Path(path) for path in glob.glob(raw, recursive=True)]
    if not matches:
        path = Path(raw)
        if path.exists():
            matches = [path]
    matches = [path for path in matches if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"{label}: no summary file matched {raw!r}")
    matches.sort(key=lambda path: path.stat().st_mtime)
    return matches[-1]


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    rows = payload.get("rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    summary = payload.get("summary")
    if isinstance(summary, list):
        return [row for row in summary if isinstance(row, dict)]
    return []


def row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    prompt = row.get("prompt_tokens_requested_per_request")
    if prompt is None:
        prompt = row.get("prompt_tokens_requested")
    return (
        str(row.get("scenario") or ""),
        int(row.get("batch_size") or 0),
        int(prompt or 0),
    )


def metric(row: dict[str, Any]) -> float:
    return float(row.get("median_steady_output_tps") or row.get("median_output_tps") or 0.0)


def ok(row: dict[str, Any]) -> bool:
    return int(row.get("ok_samples") or 0) > 0


def compare(left_path: Path, right_path: Path) -> list[dict[str, Any]]:
    left_rows = {row_key(row): row for row in load_rows(left_path) if ok(row)}
    right_rows = {row_key(row): row for row in load_rows(right_path) if ok(row)}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(left_rows) & set(right_rows)):
        left_tps = metric(left_rows[key])
        right_tps = metric(right_rows[key])
        ratio = left_tps / right_tps if right_tps > 0.0 else 0.0
        rows.append(
            {
                "scenario": key[0],
                "batch_size": key[1],
                "prompt_tokens": key[2],
                "left_tok_s": left_tps,
                "right_tok_s": right_tps,
                "left_vs_right_pct": (ratio - 1.0) * 100.0 if right_tps > 0.0 else 0.0,
                "winner": "left" if left_tps > right_tps else "right",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "batch_size",
        "prompt_tokens",
        "left_tok_s",
        "right_tok_s",
        "left_vs_right_pct",
        "winner",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two inference matrix summary JSON files.")
    parser.add_argument("--left", required=True, help="Left summary path or glob, for example Prophet.")
    parser.add_argument("--right", required=True, help="Right summary path or glob, for example vLLM.")
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--csv", default="", help="Optional CSV output path.")
    args = parser.parse_args()

    left_path = resolve_summary(args.left, label=args.left_name)
    right_path = resolve_summary(args.right, label=args.right_name)
    rows = compare(left_path, right_path)

    print(f"{args.left_name}:  {left_path}")
    print(f"{args.right_name}: {right_path}")
    print()
    print(
        "scenario,batch,prompt,"
        f"{args.left_name}_tok_s,{args.right_name}_tok_s,"
        f"{args.left_name}_vs_{args.right_name}_pct,winner"
    )
    for row in rows:
        winner = args.left_name if row["winner"] == "left" else args.right_name
        print(
            f"{row['scenario']},{row['batch_size']},{row['prompt_tokens']},"
            f"{row['left_tok_s']:.2f},{row['right_tok_s']:.2f},"
            f"{row['left_vs_right_pct']:+.1f}%,{winner}"
        )

    if not rows:
        print("No matching successful rows found.")

    if args.csv:
        write_csv(Path(args.csv), rows)
        print()
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

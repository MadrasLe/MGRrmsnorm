"""Collect inference matrix summaries into one CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def flatten_row(summary_file: Path, payload: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    args = payload.get("args", {})
    system = payload.get("system", {})
    gpu = system.get("gpu", {})
    devices = gpu.get("devices") or []
    first_gpu = devices[0] if devices else {}
    return {
        "source": str(summary_file),
        "model": args.get("model"),
        "dtype": args.get("dtype"),
        "quantize": args.get("quantize") or "none",
        "backend": row.get("backend"),
        "hardware_label": row.get("hardware_label"),
        "gpu_count": gpu.get("count", 0),
        "gpu_name": first_gpu.get("name"),
        "gpu_total_gb": first_gpu.get("total_gb"),
        "scenario": row.get("scenario"),
        "batch_size": row.get("batch_size"),
        "prompt_tokens": row.get("prompt_tokens_requested_per_request"),
        "max_new_tokens": row.get("max_new_tokens_per_request"),
        "kv_offload": row.get("kv_offload"),
        "samples": row.get("samples"),
        "ok_samples": row.get("ok_samples"),
        "median_output_tps": row.get("median_output_tps"),
        "best_output_tps": row.get("best_output_tps"),
        "worst_output_tps": row.get("worst_output_tps"),
        "median_scheduler_decode_tps": row.get("median_scheduler_decode_tps"),
        "median_decode_wall_tps": row.get("median_decode_wall_tps"),
        "median_prefill_tps": row.get("median_prefill_tps"),
        "median_prefill_time_ms": row.get("median_prefill_time_ms"),
        "median_decode_time_ms": row.get("median_decode_time_ms"),
        "median_elapsed_s": row.get("median_elapsed_s"),
        "errors": "; ".join(str(err) for err in (row.get("errors") or []) if err),
    }


def collect(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            rows.append(flatten_row(path, payload, row))
    rows.sort(
        key=lambda row: (
            str(row["model"]),
            str(row["hardware_label"]),
            str(row["backend"]),
            str(row["scenario"]),
            int(row["batch_size"] or 0),
            int(row["prompt_tokens"] or 0),
        )
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="bench_results")
    parser.add_argument("--out", default="bench_results/inference_matrix_all.csv")
    args = parser.parse_args()

    rows = collect(Path(args.input_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out.write_text("", encoding="utf-8")
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

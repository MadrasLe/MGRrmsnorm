#!/usr/bin/env python3
"""Run MicroGemm kernel-env variants back-to-back in one CPU session.

Colab CPU VMs vary enough that a toggle tested in a different runtime can look
better or worse for the wrong reason. This helper runs the same MicroGemm suite
multiple times in one process, changing only selected environment variables, and
writes compact CSVs with aggregate and per-repeat paired speedups against the
first variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


SCRIPT_BUILD_TAG = "qwen25_microgemm_env_sweep_v59_phi_gatepair_off_promoted"
QUANT_CHOICES = ("int8", "int4", "int8g128", "int8g", "int4g128", "int4g")
QUANT_ALIASES = {
    "int8g": "int8g128",
    "int4g": "int4g128",
}
WROTE_CSV_RE = re.compile(r"^\s*csv:\s+(.+?)\s*$", re.MULTILINE)
WROTE_JSON_RE = re.compile(r"^\s*json:\s+(.+?)\s*$", re.MULTILINE)
LOADAVG_RE = re.compile(r"^\s*loadavg:\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\s*$", re.MULTILINE)
SUITE_BUILD_RE = re.compile(r"^\s*suite build:\s+(\S+)\s*$", re.MULTILINE)
TUNING_PRESETS = {
    "phi3_i8g_gate": {
        "model_repo": "microsoft/Phi-3-mini-4k-instruct",
        "quant": "int8g128",
        "max_seq_len": 512,
        "variants": [
            "phi_gate_pair_off:MICROGEMM_I8G_ROW_PAIR_TILE4_ALL=1,MICROGEMM_I8G_GATE_PAIR4=0",
            "gate_p4u64:MICROGEMM_I8G_ROW_PAIR_TILE4_ALL=1,MICROGEMM_I8G_GATE_PAIR4=1,MICROGEMM_I8G_GATE_PAIR4_UNROLL64=1",
            "gate_p4u128:MICROGEMM_I8G_ROW_PAIR_TILE4_ALL=1,MICROGEMM_I8G_GATE_PAIR4=1,MICROGEMM_I8G_GATE_PAIR4_UNROLL128=1",
            "gate_p8split:MICROGEMM_I8G_ROW_PAIR_TILE4_ALL=1,MICROGEMM_I8G_GATE_PAIR4=1,MICROGEMM_I8G_GATE_PAIR8_SPLITPASS=1",
            "gate_prefetch:MICROGEMM_I8G_ROW_PAIR_TILE4_ALL=1,MICROGEMM_I8G_GATE_PAIR4=1,MICROGEMM_I8G_GATE_PREFETCH=1",
        ],
    },
    "phi3_i8g_gate_dispatch": {
        "model_repo": "microsoft/Phi-3-mini-4k-instruct",
        "quant": "int8g128",
        "max_seq_len": 512,
        "variants": [
            "phi_gate_pair_off:MICROGEMM_I8G_ROW_PAIR_TILE4_ALL=1,MICROGEMM_I8G_GATE_PAIR4=0",
            "gate_pair_auto:MICROGEMM_I8G_ROW_PAIR_TILE4_ALL=1,MICROGEMM_I8G_GATE_PAIR4=1",
            "gate_bias_off:MICROGEMM_I8G_ROW_PAIR_TILE4_ALL=1,MICROGEMM_I8G_GATE8_BIASED_INPUT=0",
            "gate_a128_signed:MICROGEMM_I8G_ROW_PAIR_TILE4_ALL=1,MICROGEMM_I8G_GATE8_BIASED_INPUT=0,MICROGEMM_I8G_GATE8_ALIGNED128=1",
            "gate_legacy_tile:MICROGEMM_I8G_ROW_PAIR_TILE4_ALL=1,MICROGEMM_I8G_GATE_UP_TILE8_EXPLICIT=0",
        ],
    },
}


def parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_variant(raw: str) -> tuple[str, dict[str, str], str | None]:
    if ":" in raw:
        label, rest = raw.split(":", 1)
    else:
        label, rest = raw, ""
    label = label.strip()
    if not label:
        raise ValueError(f"empty variant label in {raw!r}")
    env: dict[str, str] = {}
    quant: str | None = None
    for part in rest.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"expected NAME=VALUE in variant {raw!r}, got {part!r}")
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"empty env name in variant {raw!r}")
        if name.lower() == "quant":
            if value not in QUANT_CHOICES:
                expected = ", ".join(QUANT_CHOICES)
                raise ValueError(f"variant quant must be one of {expected} in {raw!r}")
            quant = QUANT_ALIASES.get(value, value)
            continue
        env[name] = value
    return label, env, quant


def resolve_output_path(raw: str, cwd: Path) -> Path:
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def parse_last_written_path(output: str, regex: re.Pattern[str], cwd: Path) -> Path | None:
    matches = regex.findall(output)
    if not matches:
        return None
    return resolve_output_path(matches[-1], cwd)


def parse_loadavg(output: str) -> tuple[str, str, str]:
    match = LOADAVG_RE.search(output)
    if not match:
        return "", "", ""
    return match.group(1), match.group(2), match.group(3)


def parse_suite_build(output: str) -> str:
    matches = SUITE_BUILD_RE.findall(output)
    return matches[-1] if matches else ""


def run_streamed(cmd: list[str], *, cwd: Path, label: str, env_overrides: dict[str, str]) -> str:
    print(f"\n== {label} ==", flush=True)
    if env_overrides:
        rendered_env = " ".join(f"{key}={value}" for key, value in sorted(env_overrides.items()))
        print(f"env: {rendered_env}", flush=True)
    print("+ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.update(env_overrides)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    captured: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        captured.append(line)
        print(line, end="", flush=True)
    rc = process.wait()
    output = "".join(captured)
    if rc != 0:
        tail = output[-5000:]
        raise RuntimeError(
            f"{label} failed with exit code {rc}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"output tail:\n{tail}"
        )
    return output


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metric_float(row: dict[str, str], name: str) -> float:
    raw = row.get(name, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


METRIC_ALIASES = {
    "wall_med": ("wall_med", "wall_output_tps_median"),
    "total_wall": ("total_wall", "wall_total_tps_median"),
    "runtime_med": ("runtime_med", "runtime_output_tps_median"),
    "steady": ("steady", "steady_output_tps_median"),
    "decode_only": ("decode_only", "decode_only_output_tps_median"),
    "prefill": ("prefill", "prefill_tps_median"),
    "prefill_ms": ("prefill_ms", "prefill_ms_median"),
    "overhead": ("overhead", "harness_overhead_ms_median"),
    "engine": ("engine", "batch_profile_total_ms_median"),
    "gate_up": ("gate_up", "batch_profile_gate_up_ms_median"),
    "gate_q": ("gate_q", "batch_profile_gate_up_quant_ms_median"),
    "gate_dot": ("gate_dot", "batch_profile_gate_up_dot_ms_median"),
    "down_proj": ("down_proj", "batch_profile_down_proj_ms_median"),
    "down_q": ("down_q", "batch_profile_down_proj_quant_ms_median"),
    "down_dot": ("down_dot", "batch_profile_down_proj_dot_ms_median"),
    "lm_head": ("lm_head", "batch_profile_lm_head_ms_median"),
    "gw_i8_rowpair": ("gw_i8_rowpair", "batch_profile_groupwise_i8_row_pair_calls_median"),
    "gw_i4_rowpair": ("gw_i4_rowpair", "batch_profile_groupwise_i4_row_pair_calls_median"),
    "gw_gate_fused": ("gw_gate_fused", "batch_profile_groupwise_gate_up_fused_calls_median"),
    "gw_i8_gate_combo": ("gw_i8_gate_combo", "batch_profile_groupwise_i8_gate_safe_combined_calls_median"),
    "gw_i8_gate_combo8": ("gw_i8_gate_combo8", "batch_profile_groupwise_i8_gate_safe_combined_tile8_calls_median"),
    "gw_i8_gate8": ("gw_i8_gate8", "batch_profile_groupwise_i8_gate_tile8_calls_median"),
    "gw_i8_gate_bias": ("gw_i8_gate_bias", "batch_profile_groupwise_i8_gate_biased_calls_median"),
    "gw_i8_gate_pair": ("gw_i8_gate_pair", "batch_profile_groupwise_i8_gate_pair_calls_median"),
    "gw_i8_gate_p4u64": ("gw_i8_gate_p4u64", "batch_profile_groupwise_i8_gate_pair_unroll64_calls_median"),
    "gw_i8_gate_p4u128": ("gw_i8_gate_p4u128", "batch_profile_groupwise_i8_gate_pair_unroll128_calls_median"),
    "gw_i8_gate_p8split": ("gw_i8_gate_p8split", "batch_profile_groupwise_i8_gate_pair8_split_calls_median"),
    "gw_i8_gate_prefetch": ("gw_i8_gate_prefetch", "batch_profile_groupwise_i8_gate_prefetch_calls_median"),
    "gw_lm_rowpair": ("gw_lm_rowpair", "batch_profile_groupwise_lm_head_row_pair_calls_median"),
    "lm_stack": ("lm_stack", "batch_profile_lm_head_stack_best_calls_median"),
}


def row_value(row: dict[str, str] | None, canonical_name: str) -> str:
    if row is None:
        return ""
    for name in METRIC_ALIASES.get(canonical_name, (canonical_name,)):
        value = row.get(name, "")
        if value not in ("", None):
            return str(value)
    return ""


def row_float(row: dict[str, str] | None, canonical_name: str) -> float:
    if row is None:
        return 0.0
    for name in METRIC_ALIASES.get(canonical_name, (canonical_name,)):
        value = metric_float(row, name)
        if value != 0.0:
            return value
    return 0.0


def normalize_mode_name(mode: str) -> str:
    if mode == "continuous" or mode.startswith("continuous"):
        return "continuous"
    return mode


def row_matches_mode(row: dict[str, str], mode: str) -> bool:
    if not mode:
        return True
    wanted = normalize_mode_name(mode)
    candidates = (
        row.get("mode", ""),
        row.get("batch_mode", ""),
        row.get("effective_mode", ""),
    )
    for candidate in candidates:
        if not candidate:
            continue
        if candidate == mode or normalize_mode_name(candidate) == wanted:
            return True
    return False


def choose_rows(
    rows: list[dict[str, str]],
    *,
    batch_sizes: list[int],
    mode: str,
) -> dict[int, dict[str, str]]:
    selected: dict[int, dict[str, str]] = {}
    wanted = set(batch_sizes)
    for row in rows:
        try:
            batch = int(float(row.get("batch", row.get("batch_size", ""))))
        except ValueError:
            continue
        if batch not in wanted:
            continue
        if row.get("kind") == "single_context":
            continue
        if not row_matches_mode(row, mode):
            continue
        ok = row.get("ok", "")
        if ok in ("0", "false", "False"):
            continue
        selected[batch] = row
    return selected


def safe_speedup(numerator: float, denominator: float) -> str:
    if numerator <= 0.0 or denominator <= 0.0:
        return ""
    return f"{numerator / denominator:.3f}"


def median_text(values: list[float]) -> str:
    values = [value for value in values if value > 0.0]
    if not values:
        return ""
    return f"{statistics.median(values):.6g}"


def compact_metric_text(row: dict[str, str] | None, canonical_name: str) -> str:
    value = row_float(row, canonical_name)
    if value <= 0.0:
        return ""
    return f"{value:.6g}"


def median_float(values: list[float]) -> float:
    values = [value for value in values if value > 0.0]
    if not values:
        return 0.0
    return float(statistics.median(values))


def scheduled_variants(
    variants: list[tuple[str, dict[str, str], str]],
    repeat_idx: int,
    schedule: str,
) -> list[tuple[str, dict[str, str], str]]:
    if not variants:
        return []
    if schedule == "fixed":
        return list(variants)
    if schedule == "reverse":
        return list(variants) if (repeat_idx % 2 == 0) else list(reversed(variants))
    shift = repeat_idx % len(variants)
    return list(variants[shift:] + variants[:shift])


def build_summary_rows(
    variant_rows: list[dict[str, Any]],
    *,
    batch_sizes: list[int],
    mode: str,
) -> list[dict[str, str]]:
    out_rows: list[dict[str, str]] = []
    metric_names = (
        "wall_med",
        "total_wall",
        "runtime_med",
        "steady",
        "decode_only",
        "prefill",
        "prefill_ms",
        "overhead",
        "engine",
        "rope_kv",
        "gate_up",
        "gate_q",
        "gate_dot",
        "down_proj",
        "down_q",
        "down_dot",
        "lm_head",
        "gw_i8_rowpair",
        "gw_i4_rowpair",
        "gw_gate_fused",
        "gw_i8_gate_combo",
        "gw_i8_gate_combo8",
        "gw_i8_gate8",
        "gw_i8_gate_bias",
        "gw_i8_gate_pair",
        "gw_i8_gate_p4u64",
        "gw_i8_gate_p4u128",
        "gw_i8_gate_p8split",
        "gw_i8_gate_prefetch",
        "gw_lm_rowpair",
        "lm_stack",
    )
    variant_order: list[str] = []
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for variant in variant_rows:
        if variant["label"] not in variant_order:
            variant_order.append(variant["label"])
        by_batch = choose_rows(variant["rows"], batch_sizes=batch_sizes, mode=mode)
        for batch in batch_sizes:
            row = by_batch.get(batch)
            key = (variant["label"], batch)
            group = grouped.setdefault(
                key,
                {
                    "label": variant["label"],
                    "env_text": variant["env_text"],
                    "quant": variant["quant"],
                    "batch": batch,
                    "csvs": [],
                    "metrics": {name: [] for name in metric_names},
                    "runs": 0,
                },
            )
            group["csvs"].append(str(variant["csv"]))
            if row:
                group["runs"] += 1
                for name in metric_names:
                    value = row_float(row, name)
                    if value > 0.0:
                        group["metrics"][name].append(value)

    if not variant_order:
        return out_rows
    baseline_label = variant_order[0]
    baseline_metrics: dict[int, dict[str, float]] = {}
    for batch in batch_sizes:
        base_group = grouped.get((baseline_label, batch))
        if not base_group:
            continue
        baseline_metrics[batch] = {
            name: median_float(base_group["metrics"][name])
            for name in metric_names
        }

    for label in variant_order:
        for batch in batch_sizes:
            group = grouped.get((label, batch))
            summary: dict[str, str] = {
                "variant": label,
                "env": group["env_text"] if group else "",
                "quant": group["quant"] if group else "",
                "batch": str(batch),
                "mode": mode,
                "runs": str(group["runs"] if group else 0),
                "csv": ";".join(group["csvs"]) if group else "",
            }
            for name in metric_names:
                summary[name] = median_text(group["metrics"][name]) if group else ""
            base = baseline_metrics.get(batch, {})
            if group and base:
                summary["wall_speedup_vs_baseline"] = safe_speedup(
                    median_float(group["metrics"]["wall_med"]),
                    base.get("wall_med", 0.0),
                )
                summary["decode_speedup_vs_baseline"] = safe_speedup(
                    median_float(group["metrics"]["decode_only"]),
                    base.get("decode_only", 0.0),
                )
                summary["prefill_speedup_vs_baseline"] = safe_speedup(
                    median_float(group["metrics"]["prefill"]),
                    base.get("prefill", 0.0),
                )
            else:
                summary["wall_speedup_vs_baseline"] = ""
                summary["decode_speedup_vs_baseline"] = ""
                summary["prefill_speedup_vs_baseline"] = ""
            out_rows.append(summary)
    return out_rows


def build_paired_rows(
    variant_rows: list[dict[str, Any]],
    *,
    batch_sizes: list[int],
    mode: str,
) -> list[dict[str, str]]:
    out_rows: list[dict[str, str]] = []
    if not variant_rows:
        return out_rows

    variant_order: list[str] = []
    repeats: list[int] = []
    by_repeat_label: dict[tuple[int, str], dict[str, Any]] = {}
    for variant in variant_rows:
        label = str(variant["label"])
        repeat = int(variant["repeat"])
        if label not in variant_order:
            variant_order.append(label)
        if repeat not in repeats:
            repeats.append(repeat)
        by_repeat_label[(repeat, label)] = variant

    baseline_label = variant_order[0]
    metric_names = (
        "wall_med",
        "decode_only",
        "prefill",
        "rope_kv",
        "gate_up",
        "gate_q",
        "gate_dot",
        "down_proj",
        "down_q",
        "down_dot",
        "lm_head",
        "gw_i8_rowpair",
        "gw_i4_rowpair",
        "gw_gate_fused",
        "gw_i8_gate_combo",
        "gw_i8_gate_combo8",
        "gw_i8_gate8",
        "gw_i8_gate_bias",
        "gw_i8_gate_pair",
        "gw_i8_gate_p4u64",
        "gw_i8_gate_p4u128",
        "gw_i8_gate_p8split",
        "gw_i8_gate_prefetch",
        "gw_lm_rowpair",
        "lm_stack",
    )
    for repeat in sorted(repeats):
        base_variant = by_repeat_label.get((repeat, baseline_label))
        if not base_variant:
            continue
        base_by_batch = choose_rows(base_variant["rows"], batch_sizes=batch_sizes, mode=mode)
        for batch in batch_sizes:
            base_row = base_by_batch.get(batch)
            for label in variant_order:
                variant = by_repeat_label.get((repeat, label))
                row = None
                if variant:
                    row = choose_rows(variant["rows"], batch_sizes=batch_sizes, mode=mode).get(batch)
                paired: dict[str, str] = {
                    "repeat": str(repeat),
                    "variant": label,
                    "baseline": baseline_label,
                    "position": str(variant.get("position", "")) if variant else "",
                    "quant": str(variant["quant"]) if variant else "",
                    "batch": str(batch),
                    "mode": mode,
                    "load1": str(variant.get("load1", "")) if variant else "",
                    "load5": str(variant.get("load5", "")) if variant else "",
                    "load15": str(variant.get("load15", "")) if variant else "",
                    "csv": str(variant["csv"]) if variant else "",
                }
                for name in metric_names:
                    paired[name] = compact_metric_text(row, name)
                paired["wall_vs_baseline"] = safe_speedup(
                    row_float(row, "wall_med"),
                    row_float(base_row, "wall_med"),
                )
                paired["decode_vs_baseline"] = safe_speedup(
                    row_float(row, "decode_only"),
                    row_float(base_row, "decode_only"),
                )
                paired["prefill_vs_baseline"] = safe_speedup(
                    row_float(row, "prefill"),
                    row_float(base_row, "prefill"),
                )
                paired["gate_dot_vs_baseline"] = safe_speedup(
                    row_float(row, "gate_dot"),
                    row_float(base_row, "gate_dot"),
                )
                paired["gate_up_vs_baseline"] = safe_speedup(
                    row_float(row, "gate_up"),
                    row_float(base_row, "gate_up"),
                )
                paired["gate_q_vs_baseline"] = safe_speedup(
                    row_float(row, "gate_q"),
                    row_float(base_row, "gate_q"),
                )
                paired["rope_kv_vs_baseline"] = safe_speedup(
                    row_float(row, "rope_kv"),
                    row_float(base_row, "rope_kv"),
                )
                paired["down_proj_vs_baseline"] = safe_speedup(
                    row_float(row, "down_proj"),
                    row_float(base_row, "down_proj"),
                )
                paired["down_dot_vs_baseline"] = safe_speedup(
                    row_float(row, "down_dot"),
                    row_float(base_row, "down_dot"),
                )
                paired["lm_head_vs_baseline"] = safe_speedup(
                    row_float(row, "lm_head"),
                    row_float(base_row, "lm_head"),
                )
                out_rows.append(paired)
    return out_rows


def ratio_float(row: dict[str, str], name: str) -> float:
    try:
        return float(row.get(name, ""))
    except (TypeError, ValueError):
        return 0.0


def win_count_text(values: list[float], *, higher_is_better: bool = True) -> str:
    values = [value for value in values if value > 0.0]
    if not values:
        return ""
    if higher_is_better:
        wins = sum(1 for value in values if value > 1.0)
    else:
        wins = sum(1 for value in values if value < 1.0)
    return f"{wins}/{len(values)}"


def win_count(values: list[float], *, higher_is_better: bool = True) -> tuple[int, int]:
    values = [value for value in values if value > 0.0]
    if not values:
        return 0, 0
    if higher_is_better:
        wins = sum(1 for value in values if value > 1.0)
    else:
        wins = sum(1 for value in values if value < 1.0)
    return wins, len(values)


def classify_paired_variant(
    group: dict[str, list[float]],
    *,
    is_baseline: bool,
) -> tuple[str, str]:
    if is_baseline:
        return "baseline", "comparison anchor"

    paired_runs = len([value for value in group["decode"] if value > 0.0])
    if paired_runs < 2:
        return "inconclusive", f"only {paired_runs} paired run(s)"

    wall_med = median_float(group["wall"])
    decode_med = median_float(group["decode"])
    prefill_med = median_float(group["prefill"])
    gate_up_med = median_float(group["gate_up"])
    gate_dot_med = median_float(group["gate_dot"])
    lm_head_med = median_float(group["lm_head"])
    decode_wins, decode_total = win_count(group["decode"])
    gate_up_wins, gate_up_total = win_count(group["gate_up"], higher_is_better=False)
    gate_dot_wins, gate_dot_total = win_count(group["gate_dot"], higher_is_better=False)

    reasons = [
        f"wall={wall_med:.3f}x",
        f"decode={decode_med:.3f}x {decode_wins}/{decode_total}",
        f"prefill={prefill_med:.3f}x",
    ]
    if gate_up_med > 0.0:
        reasons.append(f"gate_up={gate_up_med:.3f}x {gate_up_wins}/{gate_up_total}")
    if gate_dot_med > 0.0:
        reasons.append(f"gate_dot={gate_dot_med:.3f}x {gate_dot_wins}/{gate_dot_total}")
    if lm_head_med > 0.0:
        reasons.append(f"lm_head={lm_head_med:.3f}x")

    decode_majority = decode_wins * 2 > decode_total
    gate_up_ok = gate_up_med <= 1.01 or gate_up_med == 0.0
    gate_dot_ok = gate_dot_med <= 1.01 or gate_dot_med == 0.0
    if decode_med >= 1.02 and wall_med >= 1.00 and decode_majority and gate_up_ok and gate_dot_ok:
        return "promote-candidate", "; ".join(reasons)

    near_wall = 0.99 <= wall_med <= 1.01
    near_decode = 0.99 <= decode_med <= 1.01
    near_prefill = prefill_med == 0.0 or 0.98 <= prefill_med <= 1.02
    lm_head_ok = lm_head_med <= 1.02 or lm_head_med == 0.0
    if near_wall and near_decode and near_prefill and gate_up_ok and gate_dot_ok and lm_head_ok:
        return "neutral", "; ".join(reasons)

    gate_up_bad = gate_up_med > 1.02 and gate_up_wins * 2 <= max(gate_up_total, 1)
    gate_dot_bad = gate_dot_med > 1.02 and gate_dot_wins * 2 <= max(gate_dot_total, 1)
    lm_head_bad = lm_head_med > 1.05
    decode_bad = decode_med < 0.99 or decode_wins * 2 < max(decode_total, 1)
    wall_bad = wall_med < 0.99
    if decode_bad or wall_bad or gate_up_bad or gate_dot_bad or lm_head_bad:
        return "reject", "; ".join(reasons)

    return "mixed", "; ".join(reasons)


def enrich_summary_with_paired_stats(
    summary_rows: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
) -> None:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    baseline_labels = [row.get("baseline", "") for row in paired_rows if row.get("baseline")]
    baseline_label = baseline_labels[0] if baseline_labels else ""
    for row in paired_rows:
        key = (row.get("variant", ""), row.get("batch", ""))
        group = grouped.setdefault(
            key,
            {
                "wall": [],
                "decode": [],
                "prefill": [],
                "rope_kv": [],
                "gate_up": [],
                "gate_q": [],
                "gate_dot": [],
                "down_proj": [],
                "down_dot": [],
                "lm_head": [],
            },
        )
        group["wall"].append(ratio_float(row, "wall_vs_baseline"))
        group["decode"].append(ratio_float(row, "decode_vs_baseline"))
        group["prefill"].append(ratio_float(row, "prefill_vs_baseline"))
        group["rope_kv"].append(ratio_float(row, "rope_kv_vs_baseline"))
        group["gate_up"].append(ratio_float(row, "gate_up_vs_baseline"))
        group["gate_q"].append(ratio_float(row, "gate_q_vs_baseline"))
        group["gate_dot"].append(ratio_float(row, "gate_dot_vs_baseline"))
        group["down_proj"].append(ratio_float(row, "down_proj_vs_baseline"))
        group["down_dot"].append(ratio_float(row, "down_dot_vs_baseline"))
        group["lm_head"].append(ratio_float(row, "lm_head_vs_baseline"))

    for row in summary_rows:
        key = (row.get("variant", ""), row.get("batch", ""))
        group = grouped.get(key)
        if not group:
            row["paired_runs"] = ""
            row["paired_wall_speedup_median"] = ""
            row["paired_decode_speedup_median"] = ""
            row["paired_prefill_speedup_median"] = ""
            row["paired_rope_kv_ratio_median"] = ""
            row["paired_gate_up_ratio_median"] = ""
            row["paired_gate_q_ratio_median"] = ""
            row["paired_gate_dot_ratio_median"] = ""
            row["paired_down_proj_ratio_median"] = ""
            row["paired_down_dot_ratio_median"] = ""
            row["paired_lm_head_ratio_median"] = ""
            row["paired_wall_wins"] = ""
            row["paired_decode_wins"] = ""
            row["paired_prefill_wins"] = ""
            row["paired_rope_kv_wins"] = ""
            row["paired_gate_up_wins"] = ""
            row["paired_gate_q_wins"] = ""
            row["paired_gate_dot_wins"] = ""
            row["paired_down_proj_wins"] = ""
            row["paired_down_dot_wins"] = ""
            row["paired_lm_head_wins"] = ""
            row["paired_verdict"] = "inconclusive"
            row["paired_verdict_reason"] = "no paired rows"
            continue
        paired_runs = len([value for value in group["wall"] if value > 0.0])
        row["paired_runs"] = str(paired_runs) if paired_runs else ""
        row["paired_wall_speedup_median"] = median_text(group["wall"])
        row["paired_decode_speedup_median"] = median_text(group["decode"])
        row["paired_prefill_speedup_median"] = median_text(group["prefill"])
        row["paired_rope_kv_ratio_median"] = median_text(group["rope_kv"])
        row["paired_gate_up_ratio_median"] = median_text(group["gate_up"])
        row["paired_gate_q_ratio_median"] = median_text(group["gate_q"])
        row["paired_gate_dot_ratio_median"] = median_text(group["gate_dot"])
        row["paired_down_proj_ratio_median"] = median_text(group["down_proj"])
        row["paired_down_dot_ratio_median"] = median_text(group["down_dot"])
        row["paired_lm_head_ratio_median"] = median_text(group["lm_head"])
        is_baseline = row.get("variant", "") == baseline_label
        if is_baseline:
            row["paired_wall_wins"] = ""
            row["paired_decode_wins"] = ""
            row["paired_prefill_wins"] = ""
            row["paired_rope_kv_wins"] = ""
            row["paired_gate_up_wins"] = ""
            row["paired_gate_q_wins"] = ""
            row["paired_gate_dot_wins"] = ""
            row["paired_down_proj_wins"] = ""
            row["paired_down_dot_wins"] = ""
            row["paired_lm_head_wins"] = ""
        else:
            row["paired_wall_wins"] = win_count_text(group["wall"])
            row["paired_decode_wins"] = win_count_text(group["decode"])
            row["paired_prefill_wins"] = win_count_text(group["prefill"])
            row["paired_rope_kv_wins"] = win_count_text(group["rope_kv"], higher_is_better=False)
            row["paired_gate_up_wins"] = win_count_text(group["gate_up"], higher_is_better=False)
            row["paired_gate_q_wins"] = win_count_text(group["gate_q"], higher_is_better=False)
            row["paired_gate_dot_wins"] = win_count_text(group["gate_dot"], higher_is_better=False)
            row["paired_down_proj_wins"] = win_count_text(group["down_proj"], higher_is_better=False)
            row["paired_down_dot_wins"] = win_count_text(group["down_dot"], higher_is_better=False)
            row["paired_lm_head_wins"] = win_count_text(group["lm_head"], higher_is_better=False)
        verdict, reason = classify_paired_variant(group, is_baseline=is_baseline)
        row["paired_verdict"] = verdict
        row["paired_verdict_reason"] = reason


def write_outputs(
    args: argparse.Namespace,
    summary_rows: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
    payload: dict[str, Any],
) -> tuple[Path, Path, Path]:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("microgemm_env_sweep_%Y%m%d_%H%M%S")
    json_path = out_dir / f"{run_id}_microgemm_env_sweep.json"
    csv_path = out_dir / f"{run_id}_microgemm_env_sweep_summary.csv"
    paired_csv_path = out_dir / f"{run_id}_microgemm_env_sweep_paired.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = [
        "variant",
        "env",
        "quant",
        "batch",
        "mode",
        "runs",
        "wall_med",
        "total_wall",
        "runtime_med",
        "steady",
        "decode_only",
        "prefill",
        "prefill_ms",
        "overhead",
        "engine",
        "rope_kv",
        "gate_up",
        "gate_q",
        "gate_dot",
        "down_proj",
        "down_q",
        "down_dot",
        "lm_head",
        "gw_i8_rowpair",
        "gw_i4_rowpair",
        "gw_gate_fused",
        "gw_i8_gate_combo",
        "gw_i8_gate_combo8",
        "gw_i8_gate8",
        "gw_i8_gate_bias",
        "gw_i8_gate_pair",
        "gw_i8_gate_p4u64",
        "gw_i8_gate_p4u128",
        "gw_i8_gate_p8split",
        "gw_i8_gate_prefetch",
        "gw_lm_rowpair",
        "lm_stack",
        "wall_speedup_vs_baseline",
        "decode_speedup_vs_baseline",
        "prefill_speedup_vs_baseline",
        "paired_runs",
        "paired_wall_speedup_median",
        "paired_decode_speedup_median",
        "paired_prefill_speedup_median",
        "paired_rope_kv_ratio_median",
        "paired_gate_up_ratio_median",
        "paired_gate_q_ratio_median",
        "paired_gate_dot_ratio_median",
        "paired_down_proj_ratio_median",
        "paired_down_dot_ratio_median",
        "paired_lm_head_ratio_median",
        "paired_wall_wins",
        "paired_decode_wins",
        "paired_prefill_wins",
        "paired_rope_kv_wins",
        "paired_gate_up_wins",
        "paired_gate_q_wins",
        "paired_gate_dot_wins",
        "paired_down_proj_wins",
        "paired_down_dot_wins",
        "paired_lm_head_wins",
        "paired_verdict",
        "paired_verdict_reason",
        "csv",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    paired_fieldnames = [
        "repeat",
        "variant",
        "baseline",
        "position",
        "quant",
        "batch",
        "mode",
        "load1",
        "load5",
        "load15",
        "wall_med",
        "decode_only",
        "prefill",
        "rope_kv",
        "gate_up",
        "gate_q",
        "gate_dot",
        "down_proj",
        "down_q",
        "down_dot",
        "lm_head",
        "gw_i8_rowpair",
        "gw_i4_rowpair",
        "gw_gate_fused",
        "gw_i8_gate_combo",
        "gw_i8_gate_combo8",
        "gw_i8_gate8",
        "gw_i8_gate_bias",
        "gw_i8_gate_pair",
        "gw_i8_gate_p4u64",
        "gw_i8_gate_p4u128",
        "gw_i8_gate_p8split",
        "gw_i8_gate_prefetch",
        "gw_lm_rowpair",
        "lm_stack",
        "wall_vs_baseline",
        "decode_vs_baseline",
        "prefill_vs_baseline",
        "rope_kv_vs_baseline",
        "gate_up_vs_baseline",
        "gate_q_vs_baseline",
        "gate_dot_vs_baseline",
        "down_proj_vs_baseline",
        "down_dot_vs_baseline",
        "lm_head_vs_baseline",
        "csv",
    ]
    with paired_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=paired_fieldnames)
        writer.writeheader()
        for row in paired_rows:
            writer.writerow(row)
    return json_path, csv_path, paired_csv_path


def print_summary(summary_rows: list[dict[str, str]]) -> None:
    print("\nMicroGemm same-session env sweep")
    print(
        "  variant             quant batch runs    wall   decode  prefill  gate_up gate_dot down_dot lm_head i8pair gatefus gcomb gcomb8  i8g8 bias8 gpair p4u64 p4u128 p8spl  gpre lmrow lmstk  wall/base  decode/base  p_dec/base gateT/base  lmT/base  dec_wins gate_w  lm_w"
    )
    for row in summary_rows:
        print(
            f"  {row['variant'][:18]:18s}"
            f" {row.get('quant', ''):>5s}"
            f" {row['batch']:>5s}"
            f" {row.get('runs', ''):>4s}"
            f" {row.get('wall_med', ''):>6s}"
            f" {row.get('decode_only', ''):>7s}"
            f" {row.get('prefill', ''):>8s}"
            f" {row.get('gate_up', ''):>8s}"
            f" {row.get('gate_dot', ''):>9s}"
            f" {row.get('down_dot', ''):>8s}"
            f" {row.get('lm_head', ''):>7s}"
            f" {row.get('gw_i8_rowpair', ''):>6s}"
            f" {row.get('gw_gate_fused', ''):>7s}"
            f" {row.get('gw_i8_gate_combo', ''):>5s}"
            f" {row.get('gw_i8_gate_combo8', ''):>6s}"
            f" {row.get('gw_i8_gate8', ''):>5s}"
            f" {row.get('gw_i8_gate_bias', ''):>5s}"
            f" {row.get('gw_i8_gate_pair', ''):>5s}"
            f" {row.get('gw_i8_gate_p4u64', ''):>5s}"
            f" {row.get('gw_i8_gate_p4u128', ''):>6s}"
            f" {row.get('gw_i8_gate_p8split', ''):>5s}"
            f" {row.get('gw_i8_gate_prefetch', ''):>5s}"
            f" {row.get('gw_lm_rowpair', ''):>5s}"
            f" {row.get('lm_stack', ''):>5s}"
            f" {row.get('wall_speedup_vs_baseline', ''):>10s}"
            f" {row.get('decode_speedup_vs_baseline', ''):>12s}"
            f" {row.get('paired_decode_speedup_median', ''):>11s}"
            f" {row.get('paired_gate_up_ratio_median', ''):>10s}"
            f" {row.get('paired_lm_head_ratio_median', ''):>9s}"
            f" {row.get('paired_decode_wins', ''):>9s}"
            f" {row.get('paired_gate_up_wins', ''):>5s}"
            f" {row.get('paired_lm_head_wins', ''):>5s}"
        )


def print_paired_summary(paired_rows: list[dict[str, str]]) -> None:
    if not paired_rows:
        return
    print("\nMicroGemm paired repeat ratios")
    print(
        "  r pos variant             batch load1    wall   decode  prefill gate_up gate_dot down_dot lm_head i8pair gatefus gcomb gcomb8  i8g8 bias8 gpair p4u64 p4u128 p8spl  gpre lmrow lmstk  wall/base  decode/base gateT/base  lmT/base"
    )
    for row in paired_rows:
        print(
            f"  {row['repeat']:>1s}"
            f" {row.get('position', ''):>3s}"
            f"  {row['variant'][:18]:18s}"
            f" {row['batch']:>5s}"
            f" {row.get('load1', ''):>5s}"
            f" {row.get('wall_med', ''):>7s}"
            f" {row.get('decode_only', ''):>7s}"
            f" {row.get('prefill', ''):>8s}"
            f" {row.get('gate_up', ''):>7s}"
            f" {row.get('gate_dot', ''):>8s}"
            f" {row.get('down_dot', ''):>8s}"
            f" {row.get('lm_head', ''):>7s}"
            f" {row.get('gw_i8_rowpair', ''):>6s}"
            f" {row.get('gw_gate_fused', ''):>7s}"
            f" {row.get('gw_i8_gate_combo', ''):>5s}"
            f" {row.get('gw_i8_gate_combo8', ''):>6s}"
            f" {row.get('gw_i8_gate8', ''):>5s}"
            f" {row.get('gw_i8_gate_bias', ''):>5s}"
            f" {row.get('gw_i8_gate_pair', ''):>5s}"
            f" {row.get('gw_i8_gate_p4u64', ''):>5s}"
            f" {row.get('gw_i8_gate_p4u128', ''):>6s}"
            f" {row.get('gw_i8_gate_p8split', ''):>5s}"
            f" {row.get('gw_i8_gate_prefetch', ''):>5s}"
            f" {row.get('gw_lm_rowpair', ''):>5s}"
            f" {row.get('lm_stack', ''):>5s}"
            f" {row.get('wall_vs_baseline', ''):>10s}"
            f" {row.get('decode_vs_baseline', ''):>12s}"
            f" {row.get('gate_up_vs_baseline', ''):>10s}"
            f" {row.get('lm_head_vs_baseline', ''):>9s}"
        )


def print_variant_verdicts(summary_rows: list[dict[str, str]]) -> None:
    verdict_rows = [row for row in summary_rows if row.get("paired_verdict")]
    if not verdict_rows:
        return
    print("\nMicroGemm variant verdicts")
    for row in verdict_rows:
        print(
            f"  {row.get('variant', '')[:18]:18s}"
            f" batch={row.get('batch', ''):>4s}"
            f" verdict={row.get('paired_verdict', ''):18s}"
            f" reason={row.get('paired_verdict_reason', '')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tuning-preset",
        choices=tuple(sorted(TUNING_PRESETS)),
        default="",
        help="Preconfigured MicroGemm env sweep, including Phi-3 int8g gate_up probe sets.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="LABEL[:NAME=VALUE[,NAME=VALUE...]]",
        help="MicroGemm variant. Use quant=int4/int8/int8g128/int4g128 for per-variant quant, plus optional env overrides. First variant is the baseline for ratios.",
    )
    parser.add_argument("--model-repo", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--cache-dir", default="/content/microgemm_qwen25_cache")
    parser.add_argument("--out-dir", default=str(Path(tempfile.gettempdir()) / "microgemm_bench_results" / "microgemm_env_sweep"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--prompt-tokens", default="64,256")
    parser.add_argument("--batch-prompt-tokens", type=int, default=64)
    parser.add_argument("--batch-sizes", default="8")
    parser.add_argument("--batch-modes", default="continuous")
    parser.add_argument("--microgemm-mode", default="continuous")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the full variant list and summarize medians per variant.")
    parser.add_argument(
        "--variant-schedule",
        choices=("rotate", "reverse", "fixed"),
        default="rotate",
        help="Variant order across repeats. rotate balances each variant across positions; reverse matches the older alternating order.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.0,
        help="Optional sleep after each variant run. Useful when Colab loadavg drifts during paired sweeps.",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--quant", choices=QUANT_CHOICES, default="int4")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--allow-slow-canary", action="store_true")
    parser.add_argument("--allow-weak-canary", action="store_true")
    parser.add_argument("--strict-canary-gate", action="store_true")
    parser.add_argument("--run-batch-canary", action="store_true")
    parser.add_argument("--force-microgemm-rebuild", action="store_true")
    parser.add_argument(
        "--expect-suite-build",
        default="",
        help="Abort if qwen25_cpu_suite.py reports a different SUITE_BUILD_TAG. Useful for catching stale Colab/Drive sources.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.quant = QUANT_ALIASES.get(args.quant, args.quant)
    preset = TUNING_PRESETS.get(args.tuning_preset, {})
    if preset:
        if not args.model_dir and args.model_repo == "Qwen/Qwen2.5-0.5B-Instruct":
            args.model_repo = str(preset["model_repo"])
        args.quant = str(preset["quant"])
        if args.max_seq_len <= 0:
            args.max_seq_len = int(preset.get("max_seq_len", 0) or 0)
    if not args.run_id:
        args.run_id = time.strftime("microgemm_env_sweep_%Y%m%d_%H%M%S")
    if args.threads <= 0:
        args.threads = os.cpu_count() or 1
    if args.repeat <= 0:
        raise SystemExit("--repeat must be >= 1")

    if args.variant:
        raw_variants = args.variant
    elif preset:
        raw_variants = list(preset["variants"])
    else:
        raw_variants = [
            "baseline",
            "gate_split:MICROGEMM_I4_GATE_TILE8_SPLIT=1",
        ]
    variants = [parse_variant(raw) for raw in raw_variants]
    batch_sizes = parse_csv_ints(args.batch_sizes)
    cwd = Path.cwd()
    root = Path(__file__).resolve().parents[2]
    cpu_script = root / "microgemm" / "tools" / "qwen25_cpu_suite.py"

    print("Qwen2.5 MicroGemm same-session env sweep")
    print(f"  script build:  {SCRIPT_BUILD_TAG}")
    print(f"  model:         {args.model_dir or args.model_repo}")
    print(f"  quant:         {args.quant}")
    print(f"  batch sizes:   {args.batch_sizes}")
    print(f"  batch prompt:  {args.batch_prompt_tokens}")
    print(f"  max new tok:   {args.max_new_tokens}")
    if args.max_seq_len > 0:
        print(f"  max seq len:   {args.max_seq_len}")
    print(f"  threads:       {args.threads}")
    if args.tuning_preset:
        print(f"  tuning preset: {args.tuning_preset}")
    print(f"  repeat:        {args.repeat}")
    print(f"  schedule:      {args.variant_schedule}")
    if args.cooldown_seconds > 0.0:
        print(f"  cooldown:      {args.cooldown_seconds:g}s")
    rendered_variants = []
    for label, _env, quant_override in variants:
        rendered_variants.append(f"{label}[{quant_override or args.quant}]")
    print("  variants:      " + ", ".join(rendered_variants))

    if args.force_microgemm_rebuild:
        build_dir = Path(args.cache_dir).expanduser() / "build" / "microgemm"
        print(f"[microgemm] removing cached build before first variant: {build_dir}", flush=True)
        shutil.rmtree(build_dir, ignore_errors=True)

    variant_rows: list[dict[str, Any]] = []
    for repeat_idx in range(args.repeat):
        repeat_suffix = f"r{repeat_idx + 1}"
        round_variants = scheduled_variants(variants, repeat_idx, args.variant_schedule)
        for position, (label, env_overrides, quant_override) in enumerate(round_variants, start=1):
            run_label = label if args.repeat == 1 else f"{label}_{repeat_suffix}"
            variant_out_dir = Path(args.out_dir).expanduser() / run_label
            variant_run_id = f"{args.run_id}_{run_label}"
            variant_quant = quant_override or args.quant
            cmd = [
                sys.executable,
                str(cpu_script),
                "--model-repo",
                args.model_repo,
                "--cache-dir",
                args.cache_dir,
                "--out-dir",
                str(variant_out_dir),
                "--run-id",
                variant_run_id,
                "--prompt-tokens",
                args.prompt_tokens,
                "--batch-prompt-tokens",
                str(args.batch_prompt_tokens),
                "--batch-sizes",
                args.batch_sizes,
                "--batch-modes",
                args.batch_modes,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--runs",
                str(args.runs),
                "--warmup",
                str(args.warmup),
                "--threads",
                str(args.threads),
                "--quant",
                variant_quant,
            ]
            if args.max_seq_len > 0:
                cmd.extend(["--max-seq-len", str(args.max_seq_len)])
            if args.model_dir:
                cmd.extend(["--model-dir", args.model_dir])
            if args.ignore_eos:
                cmd.append("--ignore-eos")
            if args.force_convert:
                cmd.append("--force-convert")
            if args.skip_download:
                cmd.append("--skip-download")
            if not args.run_batch_canary:
                cmd.append("--skip-batch-canary")
            if args.allow_slow_canary:
                cmd.append("--allow-slow-canary")
            if args.allow_weak_canary:
                cmd.append("--allow-weak-canary")
            if args.strict_canary_gate:
                cmd.append("--strict-canary-gate")

            output = run_streamed(
                cmd,
                cwd=cwd,
                label=f"microgemm variant: {run_label}",
                env_overrides=env_overrides,
            )
            csv_path = parse_last_written_path(output, WROTE_CSV_RE, cwd)
            json_path = parse_last_written_path(output, WROTE_JSON_RE, cwd)
            load1, load5, load15 = parse_loadavg(output)
            suite_build = parse_suite_build(output)
            if args.expect_suite_build and suite_build != args.expect_suite_build:
                raise RuntimeError(
                    "child suite build mismatch; the sweep is not testing the source you think it is\n"
                    f"variant: {run_label}\n"
                    f"expected: {args.expect_suite_build}\n"
                    f"got:      {suite_build or '(missing)'}\n"
                    "Note: --force-microgemm-rebuild rebuilds the cached C tree, but it does not update "
                    "the source files under /content/drive/MyDrive/MGRrmsnorm."
                )
            if csv_path is None or not csv_path.exists():
                raise RuntimeError(f"variant {run_label!r} finished, but the summary CSV path was not found")
            variant_rows.append(
                {
                    "label": label,
                    "run_label": run_label,
                    "repeat": repeat_idx + 1,
                    "position": position,
                    "env": env_overrides,
                    "quant": variant_quant,
                    "env_text": ",".join(f"{key}={value}" for key, value in sorted(env_overrides.items())),
                    "load1": load1,
                    "load5": load5,
                    "load15": load15,
                    "suite_build": suite_build,
                    "csv": csv_path,
                    "json": json_path,
                    "rows": read_csv_rows(csv_path),
                }
            )
            if args.cooldown_seconds > 0.0:
                print(f"[microgemm] cooldown {args.cooldown_seconds:g}s", flush=True)
                time.sleep(args.cooldown_seconds)

    paired_rows = build_paired_rows(variant_rows, batch_sizes=batch_sizes, mode=args.microgemm_mode)
    summary_rows = build_summary_rows(variant_rows, batch_sizes=batch_sizes, mode=args.microgemm_mode)
    enrich_summary_with_paired_stats(summary_rows, paired_rows)
    print_summary(summary_rows)
    print_paired_summary(paired_rows)
    print_variant_verdicts(summary_rows)
    payload = {
        "benchmark": "qwen25_microgemm_env_sweep",
        "script_build": SCRIPT_BUILD_TAG,
        "same_session": True,
        "run_id": args.run_id,
        "config": vars(args),
        "variants": [
            {
                "label": item["label"],
                "run_label": item["run_label"],
                "repeat": item["repeat"],
                "position": item["position"],
                "quant": item["quant"],
                "env": item["env"],
                "load1": item.get("load1", ""),
                "load5": item.get("load5", ""),
                "load15": item.get("load15", ""),
                "suite_build": item.get("suite_build", ""),
                "csv": str(item["csv"]),
                "json": str(item["json"] or ""),
            }
            for item in variant_rows
        ],
        "summary_rows": summary_rows,
        "paired_rows": paired_rows,
    }
    json_path, csv_path, paired_csv_path = write_outputs(args, summary_rows, paired_rows, payload)
    print("Wrote MicroGemm env sweep outputs:")
    print(f"  json: {json_path}")
    print(f"  csv:  {csv_path}")
    print(f"  paired_csv:  {paired_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

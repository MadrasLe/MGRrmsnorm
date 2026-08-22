#!/usr/bin/env python3
"""Run MicroGemm and llama.cpp batch benchmarks as one paired Colab session.

Colab CPU VMs vary enough that comparing a MicroGemm CSV from one runtime with
a llama.cpp CSV from another runtime is usually misleading. This wrapper runs
the MicroGemm suite first, captures the exact summary CSV it just wrote, then
runs the llama.cpp batch compare against that CSV in the same Python process.
The final paired CSV contains only same-session ratios.
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


SCRIPT_BUILD_TAG = "qwen25_same_session_compare_v29_glm4_i8g_sat_safe_preset"
QUANT_CHOICES = ("int8", "int4", "int8g128", "int8g", "int4g128", "int4g")
QUANT_ALIASES = {
    "int8g": "int8g128",
    "int4g": "int4g128",
}
DEFAULT_MODEL_REPO = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_GGUF_REPO = "hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF"
DEFAULT_GGUF_FILE = "llama-3.2-1b-instruct-q8_0.gguf"
DEFAULT_MODEL_PRESET = "llama32_1b"
MODEL_PRESETS = {
    "qwen25_05b": {
        "model_repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "gguf": {
            "q8_0": (
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "auto",
            ),
            "q4_k_m": (
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "auto",
            ),
        },
    },
    "qwen25_15b": {
        "model_repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "gguf": {
            "q8_0": (
                "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                "auto",
            ),
            "q4_k_m": (
                "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                "auto",
            ),
        },
    },
    "qwen25_3b": {
        "model_repo": "Qwen/Qwen2.5-3B-Instruct",
        "gguf": {
            "q8_0": (
                "Qwen/Qwen2.5-3B-Instruct-GGUF",
                "auto",
            ),
            "q4_k_m": (
                "Qwen/Qwen2.5-3B-Instruct-GGUF",
                "auto",
            ),
        },
    },
    "qwen25_7b": {
        "model_repo": "Qwen/Qwen2.5-7B-Instruct",
        "gguf": {
            "q8_0": (
                "Qwen/Qwen2.5-7B-Instruct-GGUF",
                "auto",
            ),
            "q4_k_m": (
                "Qwen/Qwen2.5-7B-Instruct-GGUF",
                "auto",
            ),
        },
    },
    "qwen25_14b": {
        "model_repo": "Qwen/Qwen2.5-14B-Instruct",
        "gguf": {
            "q8_0": (
                "Qwen/Qwen2.5-14B-Instruct-GGUF",
                "auto",
            ),
            "q4_k_m": (
                "Qwen/Qwen2.5-14B-Instruct-GGUF",
                "auto",
            ),
        },
    },
    "llama32_1b": {
        "model_repo": "meta-llama/Llama-3.2-1B-Instruct",
        "gguf": {
            "q8_0": (
                "hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF",
                "llama-3.2-1b-instruct-q8_0.gguf",
            ),
            "q4_k_m": (
                "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
                "llama-3.2-1b-instruct-q4_k_m.gguf",
            ),
        },
    },
    "llama31_8b": {
        # Public HF-format mirror; avoids requiring a gated Meta token for the
        # MicroGemm conversion path while preserving the Llama 3.1 8B architecture.
        "model_repo": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "gguf": {
            "q8_0": (
                "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
            ),
            "q4_k_m": (
                "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            ),
        },
    },
    "mistral7b_v03": {
        "model_repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "gguf": {
            "q8_0": (
                "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
            ),
            "q4_k_m": (
                "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            ),
        },
    },
    "qwen35_9b": {
        "model_repo": "Qwen/Qwen3.5-9B",
        "gguf": {
            "q8_0": (
                "unsloth/Qwen3.5-9B-GGUF",
                "auto",
            ),
            "q4_k_m": (
                "unsloth/Qwen3.5-9B-GGUF",
                "auto",
            ),
        },
    },
    "phi3_mini_4k": {
        "model_repo": "microsoft/Phi-3-mini-4k-instruct",
        "microgemm_env": {
            # Phi-3 has a tall fused qkv projection; all-rowpair helps qkv/o/down.
            # The gate pair4 path is slower for Phi's gate_up on the measured
            # AVX2 CPUs, so keep gate_up on the tile8 biased route.
            "MICROGEMM_I8G_ROW_PAIR_TILE4_ALL": "1",
            "MICROGEMM_I8G_GATE_PAIR4": "0",
        },
        "gguf": {
            "q8_0": (
                "reach-vb/Phi-3-mini-4k-instruct-Q8_0-GGUF",
                "auto",
            ),
            "q4_k_m": (
                "microsoft/Phi-3-mini-4k-instruct-gguf",
                "Phi-3-mini-4k-instruct-q4.gguf",
            ),
        },
    },
    "phi4_14b": {
        "model_repo": "microsoft/phi-4",
        "microgemm_env": {
            # Phi-4 keeps the Phi-3 decoder family but scales to 14B. The
            # all-rowpair route helps the tall qkv/o/down projections on AVX2,
            # while the gate pair4 route has measured worse on Phi gate_up.
            "MICROGEMM_I8G_ROW_PAIR_TILE4_ALL": "1",
            "MICROGEMM_I8G_GATE_PAIR4": "0",
        },
        "gguf": {
            "q8_0": (
                "microsoft/phi-4-gguf",
                "phi-4-Q8_0.gguf",
            ),
            "q4_k_m": (
                "microsoft/phi-4-gguf",
                "phi-4-Q4_K_S.gguf",
            ),
        },
    },
    "phi4_mini_instruct": {
        "model_repo": "microsoft/Phi-4-mini-instruct",
        "microgemm_env": {
            "MICROGEMM_I8G_ROW_PAIR_TILE4_ALL": "1",
            "MICROGEMM_I8G_GATE_PAIR4": "0",
        },
        "gguf": {
            "q8_0": (
                "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
                "auto",
            ),
            "q4_k_m": (
                "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
                "auto",
            ),
        },
    },
    "granite33_2b": {
        "model_repo": "ibm-granite/granite-3.3-2b-instruct",
        "gguf": {
            "q8_0": (
                "ibm-granite/granite-3.3-2b-instruct-GGUF",
                "auto",
            ),
            "q4_k_m": (
                "ibm-granite/granite-3.3-2b-instruct-GGUF",
                "auto",
            ),
        },
    },
    "glm4_9b_0414": {
        "model_repo": "zai-org/GLM-4-9B-0414",
        "microgemm_env": {
            # GLM-4-9B-0414 has large 4096-wide MLP projections. Keep int8g on
            # the exact/saturation-safe route for both quality and the row-pair
            # kernels that were built around that path.
            "MICROGEMM_I8G_SATURATION_SAFE": "1",
        },
        "gguf": {
            "q8_0": (
                "bartowski/THUDM_GLM-4-9B-0414-GGUF",
                "auto",
            ),
            "q4_k_m": (
                "bartowski/THUDM_GLM-4-9B-0414-GGUF",
                "auto",
            ),
        },
    },
}
WROTE_CSV_RE = re.compile(r"^\s*csv:\s+(.+?)\s*$", re.MULTILINE)
WROTE_JSON_RE = re.compile(r"^\s*json:\s+(.+?)\s*$", re.MULTILINE)
SUITE_BUILD_RE = re.compile(r"^\s*suite build:\s+(\S+)\s*$", re.MULTILINE)


def parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def planned_microgemm_max_seq_len(args: argparse.Namespace) -> int:
    if args.max_seq_len > 0:
        return int(args.max_seq_len)
    prompt_tokens = parse_csv_ints(args.prompt_tokens)
    max_prompt = max(prompt_tokens + [int(args.batch_prompt_tokens)])
    return max_prompt + int(args.max_new_tokens) + 128


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


def parse_suite_build(output: str) -> str:
    matches = SUITE_BUILD_RE.findall(output)
    return matches[-1] if matches else ""


def reject_placeholder_gguf_path(raw: str) -> None:
    if not raw:
        return
    name = Path(raw).name.upper()
    if name.startswith("SEU_") or "YOUR_" in name or "PLACEHOLDER" in name:
        raise SystemExit(
            f"--gguf-path is a placeholder, not a real file: {raw}\n"
            "Use --model-preset qwen35_9b with --llamacpp-quant q8_0/q4_k_m, "
            "or pass --gguf-repo plus --gguf-file auto."
        )


def parse_env_overrides(raw_values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"expected NAME=VALUE for --microgemm-env, got {raw!r}")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty environment variable name in --microgemm-env {raw!r}")
        env[name] = value
    return env


def run_streamed(cmd: list[str], *, cwd: Path, label: str, env_overrides: dict[str, str] | None = None) -> str:
    print(f"\n== {label} ==", flush=True)
    if env_overrides:
        rendered_env = " ".join(f"{key}={value}" for key, value in sorted(env_overrides.items()))
        print(f"env: {rendered_env}", flush=True)
    print("+ " + " ".join(cmd), flush=True)
    env = None
    if env_overrides:
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
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def fget(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        raw = row.get(key, default)
        if raw in ("", None):
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def sget(row: dict[str, Any] | None, key: str, default: str = "") -> str:
    if not row:
        return default
    value = row.get(key, default)
    return default if value is None else str(value)


def ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def collect_cpu_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {"logical_cpus": os.cpu_count() or 0}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            models: list[str] = []
            mhz: list[float] = []
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if ":" not in line:
                    continue
                key, value = [part.strip() for part in line.split(":", 1)]
                if key == "model name" and value and value not in models:
                    models.append(value)
                elif key == "cpu MHz":
                    try:
                        mhz.append(float(value))
                    except ValueError:
                        pass
            if models:
                info["model"] = models[0]
            if mhz:
                info["mhz_min"] = min(mhz)
                info["mhz_median"] = statistics.median(mhz)
                info["mhz_max"] = max(mhz)
        except OSError:
            pass
    loadavg = Path("/proc/loadavg")
    if loadavg.exists():
        try:
            parts = loadavg.read_text(encoding="utf-8").split()[:3]
            info["loadavg"] = [float(part) for part in parts]
        except (OSError, ValueError):
            pass
    cgroup_cpu = Path("/sys/fs/cgroup/cpu.max")
    if cgroup_cpu.exists():
        try:
            info["cgroup_cpu_max"] = cgroup_cpu.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return info


def print_cpu_snapshot(snapshot: dict[str, Any]) -> None:
    print("Same-session CPU snapshot")
    print(f"  logical cpus: {snapshot.get('logical_cpus', 0)}")
    if snapshot.get("model"):
        print(f"  model:        {snapshot['model']}")
    if snapshot.get("mhz_median") is not None:
        print(
            "  cpu MHz:      "
            f"min={snapshot.get('mhz_min', 0):.0f} "
            f"med={snapshot.get('mhz_median', 0):.0f} "
            f"max={snapshot.get('mhz_max', 0):.0f}"
        )
    if snapshot.get("loadavg") is not None:
        load = snapshot["loadavg"]
        print(f"  loadavg:      {load[0]:.2f},{load[1]:.2f},{load[2]:.2f}")
    if snapshot.get("cgroup_cpu_max"):
        print(f"  cgroup cpu:   {snapshot['cgroup_cpu_max']}")


def choose_microgemm_rows(
    rows: list[dict[str, str]],
    batch_sizes: list[int],
    mode: str,
) -> dict[int, dict[str, str]]:
    selected: dict[int, dict[str, str]] = {}
    for batch_size in batch_sizes:
        candidates: list[dict[str, str]] = []
        for row in rows:
            try:
                row_batch = int(float(row.get("batch_size", "0") or "0"))
            except ValueError:
                continue
            if row_batch != batch_size:
                continue
            try:
                ok_count = int(float(row.get("ok", "0") or "0"))
            except ValueError:
                ok_count = 0
            if ok_count <= 0:
                continue
            row_mode = row.get("effective_mode") or row.get("mode") or ""
            if mode != "best" and row_mode != mode and row.get("mode") != mode:
                continue
            if not row.get("runtime_output_tps_median") and not row.get("wall_output_tps_median"):
                continue
            candidates.append(row)
        if not candidates:
            continue
        if mode == "best":
            selected[batch_size] = max(candidates, key=lambda r: fget(r, "wall_output_tps_median"))
        else:
            selected[batch_size] = candidates[-1]
    return selected


def choose_llama_rows(rows: list[dict[str, str]], batch_sizes: list[int]) -> dict[int, dict[str, str]]:
    selected: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            batch_size = int(float(row.get("batch_size", "0") or "0"))
        except ValueError:
            continue
        if batch_size in batch_sizes:
            selected[batch_size] = row
    return selected


def build_paired_rows(
    micro_rows: dict[int, dict[str, str]],
    llama_rows: dict[int, dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    for batch_size in parse_csv_ints(args.batch_sizes):
        mg = micro_rows.get(batch_size)
        ll = llama_rows.get(batch_size)
        if not mg or not ll:
            continue
        micro_wall = fget(mg, "wall_output_tps_median")
        micro_runtime = fget(mg, "runtime_output_tps_median")
        micro_steady = fget(mg, "steady_output_tps_median")
        micro_decode = fget(mg, "decode_only_output_tps_median")
        micro_prefill = fget(mg, "prefill_tps_median")
        micro_total = fget(mg, "wall_total_tps_median")
        llama_output = fget(ll, "output_tps_total_median")
        llama_decode = fget(ll, "decode_tps_median")
        llama_prefill = fget(ll, "prefill_tps_median")
        llama_combined = fget(ll, "combined_tps_median")
        paired.append(
            {
                "same_session": True,
                "paired_build": SCRIPT_BUILD_TAG,
                "run_id": args.run_id,
                "model_repo": args.model_repo,
                "gguf_repo": args.gguf_repo if not args.gguf_path else "",
                "gguf_file": args.gguf_file if not args.gguf_path else Path(args.gguf_path).name,
                "quant": args.quant,
                "microgemm_suite_build": sget(mg, "suite_build"),
                "microgemm_mode": sget(mg, "effective_mode") or sget(mg, "mode"),
                "llamacpp_script_build": sget(ll, "script_build"),
                "batch_size": batch_size,
                "prompt_tokens": args.batch_prompt_tokens,
                "max_new_tokens": args.max_new_tokens,
                "threads": args.threads or (os.cpu_count() or 1),
                "microgemm_wall_output_tps": micro_wall,
                "microgemm_runtime_output_tps": micro_runtime,
                "microgemm_steady_output_tps": micro_steady,
                "microgemm_decode_only_tps": micro_decode,
                "microgemm_prefill_tps": micro_prefill,
                "microgemm_wall_total_tps": micro_total,
                "microgemm_engine_ms": fget(mg, "steady_ms_median"),
                "microgemm_overhead_ms": fget(mg, "harness_overhead_ms_median"),
                "microgemm_model_load_ms": fget(mg, "model_load_ms_median"),
                "microgemm_rope_kv_ms": fget(mg, "batch_profile_rope_kv_ms_median"),
                "microgemm_gate_up_ms": fget(mg, "batch_profile_gate_up_ms_median"),
                "microgemm_gate_up_dot_ms": fget(mg, "batch_profile_gate_up_dot_ms_median"),
                "microgemm_down_proj_ms": fget(mg, "batch_profile_down_proj_ms_median"),
                "microgemm_down_proj_dot_ms": fget(mg, "batch_profile_down_proj_dot_ms_median"),
                "microgemm_lm_head_ms": fget(mg, "batch_profile_lm_head_ms_median"),
                "llamacpp_output_total_tps": llama_output,
                "llamacpp_decode_only_tps": llama_decode,
                "llamacpp_prefill_tps": llama_prefill,
                "llamacpp_combined_total_tps": llama_combined,
                "microgemm_wall_over_llamacpp_output": ratio(micro_wall, llama_output),
                "microgemm_runtime_over_llamacpp_output": ratio(micro_runtime, llama_output),
                "microgemm_decode_over_llamacpp_decode": ratio(micro_decode, llama_decode),
                "microgemm_prefill_over_llamacpp_prefill": ratio(micro_prefill, llama_prefill),
                "microgemm_total_over_llamacpp_combined": ratio(micro_total, llama_combined),
            }
        )
    return paired


def write_paired_outputs(
    args: argparse.Namespace,
    paired_rows: list[dict[str, Any]],
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id
    json_path = out_dir / f"{run_id}_same_session_compare.json"
    csv_path = out_dir / f"{run_id}_same_session_compare_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if paired_rows:
        fieldnames: list[str] = []
        for row in paired_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(paired_rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    return json_path, csv_path


def print_paired_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No paired rows were produced. Check whether both benchmark CSVs contain the same batch sizes.")
        return
    print("\nSame-session paired comparison")
    print(
        "  batch  micro_wall  llama_output  micro/llama_wall  "
        "micro_decode  llama_decode  micro/llama_decode  micro_prefill  llama_prefill  micro_rope  micro_gate_dot  micro_down_dot"
    )
    for row in rows:
        print(
            f"  {int(row['batch_size']):5d}  "
            f"{float(row['microgemm_wall_output_tps']):10.2f}  "
            f"{float(row['llamacpp_output_total_tps']):12.2f}  "
            f"{float(row['microgemm_wall_over_llamacpp_output']):16.2f}x  "
            f"{float(row['microgemm_decode_only_tps']):12.2f}  "
            f"{float(row['llamacpp_decode_only_tps']):12.2f}  "
            f"{float(row['microgemm_decode_over_llamacpp_decode']):18.2f}x  "
            f"{float(row['microgemm_prefill_tps']):13.2f}  "
            f"{float(row['llamacpp_prefill_tps']):13.2f}  "
            f"{float(row.get('microgemm_rope_kv_ms', 0.0) or 0.0):10.0f}  "
            f"{float(row.get('microgemm_gate_up_dot_ms', 0.0) or 0.0):14.0f}  "
            f"{float(row.get('microgemm_down_proj_dot_ms', 0.0) or 0.0):14.0f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Same-session MicroGemm vs llama.cpp batch compare")
    parser.add_argument(
        "--model-preset",
        choices=tuple(MODEL_PRESETS),
        default=DEFAULT_MODEL_PRESET,
        help="Convenience preset for matching HF model and GGUF repos.",
    )
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--cache-dir", default="/content/microgemm_qwen25_cache")
    parser.add_argument("--gguf-path", default="")
    parser.add_argument(
        "--llamacpp-quant",
        choices=("q8_0", "q4_k_m"),
        default="q8_0",
        help="Convenience GGUF preset used when --gguf-path is not set.",
    )
    parser.add_argument("--gguf-repo", default=DEFAULT_GGUF_REPO)
    parser.add_argument("--gguf-file", default=DEFAULT_GGUF_FILE)
    parser.add_argument("--gguf-cache-dir", default="/content/llamacpp_qwen25_cache/gguf")
    parser.add_argument("--llama-repo", default="https://github.com/ggml-org/llama.cpp.git")
    parser.add_argument("--llama-dir", default="/content/llamacpp_qwen25_cache/llama.cpp")
    parser.add_argument("--llama-batched-bench-bin", default="")
    parser.add_argument("--build-jobs", type=int, default=0)
    parser.add_argument("--out-dir", default=str(Path(tempfile.gettempdir()) / "microgemm_bench_results" / "same_session_compare"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--prompt-tokens", default="64,256")
    parser.add_argument("--batch-prompt-tokens", type=int, default=64)
    parser.add_argument("--batch-sizes", default="8")
    parser.add_argument("--batch-modes", default="continuous")
    parser.add_argument("--microgemm-mode", default="continuous")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="MicroGemm per-request context. If omitted, planned from max(prompt_tokens,batch_prompt_tokens)+max_new_tokens+128.",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--threads-batch", type=int, default=0)
    parser.add_argument("--llama-batch-size", type=int, default=2048)
    parser.add_argument("--llama-ubatch-size", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--quant", choices=QUANT_CHOICES, default="int4")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--run-batch-canary", action="store_true", help="Run MicroGemm's extra batch canary preflight")
    parser.add_argument(
        "--skip-batch-canary",
        action="store_true",
        help="Compatibility no-op: same-session skips MicroGemm's extra batch canary unless --run-batch-canary is set.",
    )
    parser.add_argument("--strict-canary-gate", action="store_true")
    parser.add_argument("--allow-slow-canary", action="store_true")
    parser.add_argument("--allow-weak-canary", action="store_true")
    parser.add_argument("--force-microgemm-rebuild", action="store_true")
    parser.add_argument("--force-llamacpp-rebuild", action="store_true")
    parser.add_argument(
        "--microgemm-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Environment override passed only to the MicroGemm paired run. Repeat for kernel A/B toggles.",
    )
    parser.add_argument(
        "--expect-microgemm-suite-build",
        default="",
        help="Abort if the MicroGemm paired run reports a different qwen25_cpu_suite.py SUITE_BUILD_TAG.",
    )
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--prompt-shared", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.quant = QUANT_ALIASES.get(args.quant, args.quant)
    if not args.run_id:
        args.run_id = time.strftime("same_session_%Y%m%d_%H%M%S")
    if args.threads <= 0:
        args.threads = os.cpu_count() or 1
    if args.threads_batch <= 0:
        args.threads_batch = args.threads
    microgemm_max_seq_len = planned_microgemm_max_seq_len(args)
    try:
        microgemm_env = parse_env_overrides(args.microgemm_env)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    preset_microgemm_env = dict(MODEL_PRESETS[args.model_preset].get("microgemm_env", {}))
    if preset_microgemm_env:
        preset_microgemm_env.update(microgemm_env)
        microgemm_env = preset_microgemm_env
    reject_placeholder_gguf_path(args.gguf_path)
    if not args.model_dir and args.model_repo == DEFAULT_MODEL_REPO:
        args.model_repo = str(MODEL_PRESETS[args.model_preset]["model_repo"])
    if not args.gguf_path:
        default_repo, default_file = MODEL_PRESETS[DEFAULT_MODEL_PRESET]["gguf"]["q8_0"]
        if args.gguf_repo == default_repo and args.gguf_file == default_file:
            gguf_presets = MODEL_PRESETS[args.model_preset]["gguf"]
            if args.llamacpp_quant not in gguf_presets:
                available = ",".join(sorted(gguf_presets))
                raise SystemExit(
                    f"--llamacpp-quant {args.llamacpp_quant!r} is not available for "
                    f"--model-preset {args.model_preset!r}. Available: {available}"
                )
            args.gguf_repo, args.gguf_file = gguf_presets[args.llamacpp_quant]

    cwd = Path.cwd()
    root = Path(__file__).resolve().parents[2]
    cpu_script = root / "microgemm" / "tools" / "qwen25_cpu_suite.py"
    llama_script = root / "microgemm" / "tools" / "qwen25_llamacpp_batch_compare.py"
    out_dir = Path(args.out_dir).expanduser()
    micro_out_dir = out_dir / "microgemm"
    llama_out_dir = out_dir / "llamacpp"
    micro_run_id = f"{args.run_id}_microgemm"
    llama_run_id = f"{args.run_id}_llamacpp"

    print("Qwen2.5 same-session MicroGemm vs llama.cpp compare")
    print(f"  script build:  {SCRIPT_BUILD_TAG}")
    print(f"  model:         {args.model_dir or args.model_repo}")
    print(f"  gguf:          {args.gguf_path or (args.gguf_repo + '/' + args.gguf_file)}")
    print(f"  quant:         {args.quant}")
    print(f"  batch sizes:   {args.batch_sizes}")
    print(f"  batch prompt:  {args.batch_prompt_tokens}")
    print(f"  max new tok:   {args.max_new_tokens}")
    print(f"  max seq len:   {microgemm_max_seq_len}")
    print(f"  threads:       {args.threads}")
    if microgemm_env:
        rendered_env = ", ".join(f"{key}={value}" for key, value in sorted(microgemm_env.items()))
        print(f"  micro env:     {rendered_env}")
    print("  comparison:    same-session paired")
    snapshot_before = collect_cpu_snapshot()
    print_cpu_snapshot(snapshot_before)

    if args.force_microgemm_rebuild:
        build_dir = Path(args.cache_dir).expanduser() / "build" / "microgemm"
        print(f"[microgemm] removing cached build: {build_dir}", flush=True)
        shutil.rmtree(build_dir, ignore_errors=True)
    if args.force_llamacpp_rebuild:
        build_dir = Path(args.llama_dir).expanduser() / "build"
        print(f"[llama.cpp] removing cached build: {build_dir}", flush=True)
        shutil.rmtree(build_dir, ignore_errors=True)

    micro_cmd = [
        sys.executable,
        str(cpu_script),
        "--model-repo",
        args.model_repo,
        "--cache-dir",
        args.cache_dir,
        "--out-dir",
        str(micro_out_dir),
        "--run-id",
        micro_run_id,
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
        "--max-seq-len",
        str(microgemm_max_seq_len),
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
        "--threads",
        str(args.threads),
        "--quant",
        args.quant,
    ]
    if args.model_dir:
        micro_cmd.extend(["--model-dir", args.model_dir])
    if args.ignore_eos:
        micro_cmd.append("--ignore-eos")
    if args.force_convert:
        micro_cmd.append("--force-convert")
    if args.skip_download:
        micro_cmd.append("--skip-download")
    if not args.run_batch_canary:
        micro_cmd.append("--skip-batch-canary")
    if args.strict_canary_gate:
        micro_cmd.append("--strict-canary-gate")
    if args.allow_slow_canary:
        micro_cmd.append("--allow-slow-canary")
    if args.allow_weak_canary:
        micro_cmd.append("--allow-weak-canary")

    micro_output = run_streamed(
        micro_cmd,
        cwd=cwd,
        label="microgemm paired run",
        env_overrides=microgemm_env,
    )
    snapshot_after_micro = collect_cpu_snapshot()
    micro_suite_build = parse_suite_build(micro_output)
    if args.expect_microgemm_suite_build and micro_suite_build != args.expect_microgemm_suite_build:
        raise RuntimeError(
            "MicroGemm suite build mismatch; same-session comparison would use the wrong source\n"
            f"expected: {args.expect_microgemm_suite_build}\n"
            f"got:      {micro_suite_build or '(missing)'}\n"
            "Note: --force-microgemm-rebuild rebuilds the cached C tree, but it does not update "
            "the source files under /content/drive/MyDrive/MGRrmsnorm."
        )
    micro_csv = parse_last_written_path(micro_output, WROTE_CSV_RE, cwd)
    micro_json = parse_last_written_path(micro_output, WROTE_JSON_RE, cwd)
    if micro_csv is None or not micro_csv.exists():
        raise RuntimeError("MicroGemm run finished, but the summary CSV path could not be found in its output.")

    llama_cmd = [
        sys.executable,
        str(llama_script),
        "--gguf-cache-dir",
        args.gguf_cache_dir,
        "--llama-repo",
        args.llama_repo,
        "--llama-dir",
        args.llama_dir,
        "--batch-sizes",
        args.batch_sizes,
        "--prompt-tokens",
        str(args.batch_prompt_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--threads",
        str(args.threads),
        "--threads-batch",
        str(args.threads_batch),
        "--llama-batch-size",
        str(args.llama_batch_size),
        "--llama-ubatch-size",
        str(args.llama_ubatch_size),
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
        "--microgemm-csv",
        str(micro_csv),
        "--microgemm-mode",
        args.microgemm_mode,
        "--out-dir",
        str(llama_out_dir),
        "--run-id",
        llama_run_id,
    ]
    if args.gguf_path:
        llama_cmd.extend(["--gguf-path", args.gguf_path])
    else:
        llama_cmd.extend(["--gguf-repo", args.gguf_repo, "--gguf-file", args.gguf_file, "--gguf-quant", args.llamacpp_quant])
    if args.llama_batched_bench_bin:
        llama_cmd.extend(["--llama-batched-bench-bin", args.llama_batched_bench_bin])
    if args.build_jobs:
        llama_cmd.extend(["--build-jobs", str(args.build_jobs)])
    if args.flash_attn:
        llama_cmd.append("--flash-attn")
    if args.prompt_shared:
        llama_cmd.append("--prompt-shared")

    llama_output = run_streamed(llama_cmd, cwd=cwd, label="llama.cpp paired run")
    snapshot_after_llama = collect_cpu_snapshot()
    llama_csv = parse_last_written_path(llama_output, WROTE_CSV_RE, cwd)
    llama_json = parse_last_written_path(llama_output, WROTE_JSON_RE, cwd)
    if llama_csv is None or not llama_csv.exists():
        raise RuntimeError("llama.cpp run finished, but the summary CSV path could not be found in its output.")

    batch_sizes = parse_csv_ints(args.batch_sizes)
    micro_rows = choose_microgemm_rows(read_csv_rows(micro_csv), batch_sizes, args.microgemm_mode)
    llama_rows = choose_llama_rows(read_csv_rows(llama_csv), batch_sizes)
    paired_rows = build_paired_rows(micro_rows, llama_rows, args)
    print_paired_summary(paired_rows)

    payload = {
        "benchmark": "qwen25_same_session_compare",
        "script_build": SCRIPT_BUILD_TAG,
        "same_session": True,
        "run_id": args.run_id,
        "cpu_snapshot_before": snapshot_before,
        "cpu_snapshot_after_microgemm": snapshot_after_micro,
        "cpu_snapshot_after_llamacpp": snapshot_after_llama,
        "microgemm_suite_build": micro_suite_build,
        "paths": {
            "microgemm_csv": str(micro_csv),
            "microgemm_json": str(micro_json or ""),
            "llamacpp_csv": str(llama_csv),
            "llamacpp_json": str(llama_json or ""),
        },
        "config": vars(args),
        "microgemm_env": microgemm_env,
        "paired_rows": paired_rows,
    }
    json_path, csv_path = write_paired_outputs(args, paired_rows, payload)
    print("Wrote same-session paired outputs:")
    print(f"  json: {json_path}")
    print(f"  csv:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

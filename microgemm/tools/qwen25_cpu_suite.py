#!/usr/bin/env python3
"""Qwen2.5 CPU benchmark suite for the standalone MicroGemm runtime.

This harness is intentionally outside the native runtime path.  It builds the
MicroGemm C binaries when needed, downloads a Qwen2.5 HF snapshot when needed,
converts the single-file safetensors checkpoint to .mgm, then runs:

  - context/prompt sweep for one sequence;
  - native continuous batch throughput through microgemm-text batch-generate;
  - serial batch throughput;
  - concurrent batch throughput via independent microgemm-text workers.

The continuous mode is a scheduler/KV-level native batch path with batched
INT8 decode projections.  Serial and concurrent remain comparison modes for
process/worker overhead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SUITE_BUILD_TAG = "qwen25_cpu_suite_profile_v229_glm4_i8g_canary_recal"
QUANT_CHOICES = ("int8", "int4", "int8g128", "int8g", "int4g128", "int4g")
QUANT_ALIASES = {
    "int8g": "int8g128",
    "int4g": "int4g128",
}
TEXT_PROFILE_MARKER = "batch_profile_emit_version"
DECODE_PROFILE_MARKER = "microgemm_decode_batch_profile_state"
RUN_CANARY_PROMPT_TOKENS = 64
RUN_CANARY_GOOD_DECODE_TPS = 85.0
RUN_CANARY_WEAK_DECODE_TPS = 75.0
BATCH_CANARY_SIZE = 8
BATCH_CANARY_SIZES = (2, 4, 8)
BATCH_CANARY_MODE = "continuous"
BATCH_CANARY_MAX_NEW_TOKENS = 128
BATCH_CANARY_RUNS = 2
BATCH_CANARY_MAX_ATTEMPTS = 3
BATCH_CANARY_RETRY_SLEEP_S = 1.0
BATCH_CANARY_GOOD_RUNTIME_TPS = 115.0
BATCH_CANARY_WEAK_RUNTIME_TPS = 100.0
BATCH_CANARY_GOOD_RUNTIME_TPS_BY_SIZE = {
    2: 68.0,
    4: 100.0,
    8: BATCH_CANARY_GOOD_RUNTIME_TPS,
}
BATCH_CANARY_WEAK_RUNTIME_TPS_BY_SIZE = {
    2: 60.0,
    4: 90.0,
    8: BATCH_CANARY_WEAK_RUNTIME_TPS,
}
LLAMA32_1B_CANARY_PROFILE = {
    "name": "llama32_1b_cpu_avx2",
    "run_good_decode_tps": 28.0,
    "run_weak_decode_tps": 22.0,
    "batch_good_runtime_tps_by_size": {2: 20.0, 4: 28.0, 8: 29.0},
    "batch_weak_runtime_tps_by_size": {2: 17.0, 4: 23.0, 8: 24.0},
}
LLAMA31_8B_CANARY_PROFILE = {
    "name": "llama31_8b_cpu_avx2",
    "run_good_decode_tps": 5.5,
    "run_weak_decode_tps": 4.5,
    "batch_good_runtime_tps_by_size": {2: 3.0, 4: 3.5, 8: 3.8},
    "batch_weak_runtime_tps_by_size": {2: 2.5, 4: 3.0, 8: 3.2},
}
MISTRAL7B_CANARY_PROFILE = {
    "name": "mistral7b_cpu_avx2",
    "run_good_decode_tps": 5.5,
    "run_weak_decode_tps": 4.5,
    "batch_good_runtime_tps_by_size": {2: 3.0, 4: 3.5, 8: 3.8},
    "batch_weak_runtime_tps_by_size": {2: 2.5, 4: 3.0, 8: 3.2},
}
QWEN3_06B_CANARY_PROFILE = {
    "name": "qwen3_06b_cpu_avx2",
    "run_good_decode_tps": 45.0,
    "run_weak_decode_tps": 35.0,
    "batch_good_runtime_tps_by_size": {2: 18.0, 4: 25.0, 8: 30.0},
    "batch_weak_runtime_tps_by_size": {2: 14.0, 4: 20.0, 8: 24.0},
}
QWEN35_TEXT_CANARY_PROFILE = {
    "name": "qwen35_text_cpu_avx2",
    "run_good_decode_tps": 12.0,
    "run_weak_decode_tps": 8.0,
    "batch_good_runtime_tps_by_size": {2: 5.0, 4: 7.0, 8: 8.0},
    "batch_weak_runtime_tps_by_size": {2: 3.5, 4: 5.0, 8: 6.0},
}
PHI3_MINI_CANARY_PROFILE = {
    "name": "phi3_mini_cpu_avx2",
    "run_good_decode_tps": 10.0,
    "run_weak_decode_tps": 7.0,
    "batch_good_runtime_tps_by_size": {2: 6.0, 4: 9.0, 8: 10.0},
    "batch_weak_runtime_tps_by_size": {2: 4.5, 4: 7.0, 8: 8.0},
}
PHI4_14B_CANARY_PROFILE = {
    "name": "phi4_14b_cpu_avx2",
    "run_good_decode_tps": 1.4,
    "run_weak_decode_tps": 1.0,
    "batch_good_runtime_tps_by_size": {2: 1.4, 4: 1.8, 8: 2.1},
    "batch_weak_runtime_tps_by_size": {2: 1.0, 4: 1.3, 8: 1.6},
}
PHI4_MINI_CANARY_PROFILE = {
    "name": "phi4_mini_cpu_avx2",
    "run_good_decode_tps": 8.0,
    "run_weak_decode_tps": 5.5,
    "batch_good_runtime_tps_by_size": {2: 5.0, 4: 7.0, 8: 8.0},
    "batch_weak_runtime_tps_by_size": {2: 3.5, 4: 5.0, 8: 6.0},
}
GRANITE33_2B_CANARY_PROFILE = {
    "name": "granite33_2b_cpu_avx2",
    "run_good_decode_tps": 8.0,
    "run_weak_decode_tps": 6.0,
    "batch_good_runtime_tps_by_size": {2: 8.0, 4: 12.0, 8: 14.0},
    "batch_weak_runtime_tps_by_size": {2: 6.0, 4: 9.0, 8: 11.0},
}
GLM4_9B_0414_CANARY_PROFILE = {
    "name": "glm4_9b_0414_cpu_avx2",
    "run_good_decode_tps": 2.5,
    "run_weak_decode_tps": 2.0,
    "batch_good_runtime_tps_by_size": {2: 2.5, 4: 3.5, 8: 4.2},
    "batch_weak_runtime_tps_by_size": {2: 1.8, 4: 2.7, 8: 3.4},
}
DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT_SEED = (
    "Explique de forma tecnica e curta como um runtime CPU executa inferencia "
    "de um modelo de linguagem usando pesos quantizados e cache KV. "
)
BATCH_PROFILE_PHASE_KEYS = [
    "batch_profile_qkv_ms",
    "batch_profile_rope_kv_ms",
    "batch_profile_attention_ms",
    "batch_profile_o_proj_ms",
    "batch_profile_gate_up_ms",
    "batch_profile_gate_up_quant_ms",
    "batch_profile_gate_up_dot_ms",
    "batch_profile_down_proj_ms",
    "batch_profile_down_proj_quant_ms",
    "batch_profile_down_proj_dot_ms",
    "batch_profile_lm_head_ms",
    "batch_profile_alloc_ms",
]
BATCH_COMMAND_TIMING_KEYS = [
    "tokenizer_load_ms",
    "prompt_encode_ms",
    "model_open_ms",
    "model_load_ms",
    "model_cleanup_ms",
    "command_total_ms",
    "process_overhead_ms",
]


def ensure_executable(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        if not path.exists() or path.is_dir():
            return
        path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        return


def run_capture(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    exe = Path(cmd[0])
    ensure_executable(exe)
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except PermissionError as exc:
        ensure_executable(exe)
        try:
            return subprocess.run(
                cmd,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except PermissionError as retry_exc:
            raise RuntimeError(
                "permission denied while executing command after chmod +x retry\n"
                f"cmd: {' '.join(cmd)}\n"
                f"cwd: {cwd or Path.cwd()}\n"
                "If this path is on a noexec mount, copy/build MicroGemm under /content and run from there."
            ) from retry_exc


def is_colab_drive_path(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    return str(resolved).startswith("/content/drive/")


def stage_microgemm_tree(source_micro: Path, cache_dir: Path) -> Path:
    stage_micro = cache_dir / "build" / "microgemm"
    try:
        if stage_micro.resolve() == source_micro.resolve():
            return source_micro
    except OSError:
        pass

    if stage_micro.exists():
        shutil.rmtree(stage_micro)

    def ignore_build_outputs(_dirname: str, names: list[str]) -> set[str]:
        ignored = {"__pycache__", ".pytest_cache", "build", "build-host", "build-android-arm64"}
        ignored.update(
            name
            for name in names
            if (Path(_dirname) / name).is_file()
            and (
                name.endswith((".o", ".obj", ".dll", ".so", ".dylib", ".exe"))
                or name in {"microgemm", "microgemm-convert", "microgemm-text"}
            )
        )
        return ignored

    stage_micro.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_micro, stage_micro, ignore=ignore_build_outputs)
    return stage_micro


def parse_csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def collect_cpu_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {
        "logical_cpus": os.cpu_count() or 0,
    }
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
            parts = loadavg.read_text(encoding="utf-8", errors="replace").split()
            if len(parts) >= 3:
                info["loadavg_1m"] = float(parts[0])
                info["loadavg_5m"] = float(parts[1])
                info["loadavg_15m"] = float(parts[2])
        except (OSError, ValueError):
            pass
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.exists():
        try:
            info["cgroup_cpu_max"] = cpu_max.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            pass
    return info


def print_cpu_snapshot(info: dict[str, Any]) -> None:
    print("CPU snapshot")
    print(f"  logical cpus: {info.get('logical_cpus', 0)}")
    if info.get("model"):
        print(f"  model:        {info['model']}")
    if "mhz_median" in info:
        print(
            "  cpu MHz:      "
            f"min={info.get('mhz_min', 0.0):.0f} "
            f"med={info.get('mhz_median', 0.0):.0f} "
            f"max={info.get('mhz_max', 0.0):.0f}"
        )
    if "loadavg_1m" in info:
        print(
            "  loadavg:      "
            f"{info.get('loadavg_1m', 0.0):.2f},"
            f"{info.get('loadavg_5m', 0.0):.2f},"
            f"{info.get('loadavg_15m', 0.0):.2f}"
        )
    if info.get("cgroup_cpu_max"):
        print(f"  cgroup cpu:   {info['cgroup_cpu_max']}")


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "")
    if value == "":
        return default
    return value[:1].lower() not in {"0", "n", "f"}


def groupwise_gate_fused_text(quant: str) -> str:
    if groupwise_i8_saturation_safe_enabled(quant):
        specific = os.environ.get("MICROGEMM_I8G_GATE_SAFE_FUSED", "")
        if specific == "":
            specific = os.environ.get("MICROGEMM_GROUPWISE_I8_GATE_SAFE_FUSED", "")
        if specific != "":
            enabled = env_flag(
                "MICROGEMM_I8G_GATE_SAFE_FUSED",
                env_flag("MICROGEMM_GROUPWISE_I8_GATE_SAFE_FUSED", False),
            )
            return "safe(signed-dot)" if enabled and quant == "int8g128" else "off(sat-safe)"
        return "auto(sat-safe,tile4)"
    specific = os.environ.get("MICROGEMM_GROUPWISE_GATE_UP_FUSED", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_GATE_FUSED", "")
    if specific != "":
        return "on" if env_flag("MICROGEMM_GROUPWISE_GATE_UP_FUSED", env_flag("MICROGEMM_GROUPWISE_GATE_FUSED", False)) else "off"
    if quant in {"int4g128", "int8g128"}:
        return "auto(int4g/int8g,batch>=4)"
    return "off"


def groupwise_i8_gate_safe_combined_text(quant: str) -> str:
    specific = os.environ.get("MICROGEMM_I8G_GATE_SAFE_COMBINED_TILE4", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_I8G_GATE_SAFE_COMBINED", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_I8G_GATE_SAFE_COMBINED_TILE4",
            env_flag("MICROGEMM_I8G_GATE_SAFE_COMBINED", False),
        )
        return "on(sat-safe)" if enabled and quant == "int8g128" and groupwise_i8_saturation_safe_enabled(quant) else "off"
    if quant == "int8g128" and groupwise_i8_saturation_safe_enabled(quant):
        return "auto(sat-safe,batch>=4)"
    return "off"


def groupwise_i8_gate_safe_combined_tile8_text(quant: str) -> str:
    specific = os.environ.get("MICROGEMM_I8G_GATE_SAFE_COMBINED_TILE8", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_I8G_GATE_SAFE_COMBO_TILE8", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_I8G_GATE_SAFE_COMBINED_TILE8",
            env_flag("MICROGEMM_I8G_GATE_SAFE_COMBO_TILE8", False),
        )
        return "on(sat-safe)" if enabled and quant == "int8g128" and groupwise_i8_saturation_safe_enabled(quant) else "off"
    return "off(default,env-only)"


def groupwise_compact_mlp_stride_text(quant: str) -> str:
    specific = os.environ.get("MICROGEMM_GROUPWISE_COMPACT_MLP_STRIDE", "")
    if specific != "":
        enabled = env_flag("MICROGEMM_GROUPWISE_COMPACT_MLP_STRIDE", False)
        return "on" if enabled and quant == "int4g128" else "off"
    return "off"


def groupwise_i4_row_pair_text(quant: str) -> str:
    specific = os.environ.get("MICROGEMM_I4G_ROW_PAIR_TILE4", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_I4_ROW_PAIR_TILE4", "")
    if specific != "":
        enabled = env_flag("MICROGEMM_I4G_ROW_PAIR_TILE4", env_flag("MICROGEMM_GROUPWISE_I4_ROW_PAIR_TILE4", False))
        return "on(down)" if enabled and quant == "int4g128" else "off"
    if quant == "int4g128":
        return "auto(int4g,down,batch>=4)"
    return "off"


def groupwise_i8_saturation_safe_enabled(quant: str) -> bool:
    specific = os.environ.get("MICROGEMM_I8G_SATURATION_SAFE", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_I8_SATURATION_SAFE", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_I8G_SAT_SAFE", "")
    if specific == "":
        return False
    enabled = env_flag(
        "MICROGEMM_I8G_SATURATION_SAFE",
        env_flag(
            "MICROGEMM_GROUPWISE_I8_SATURATION_SAFE",
            env_flag("MICROGEMM_I8G_SAT_SAFE", False),
        ),
    )
    return enabled and quant == "int8g128"


def groupwise_i8_row_pair_text(quant: str) -> str:
    if groupwise_i8_saturation_safe_enabled(quant):
        specific = os.environ.get("MICROGEMM_I8G_SAT_SAFE_ROW_PAIR_TILE4", "")
        if specific == "":
            specific = os.environ.get("MICROGEMM_I8G_ROW_PAIR_TILE4_SAFE", "")
        if specific == "":
            specific = os.environ.get("MICROGEMM_GROUPWISE_I8_ROW_PAIR_TILE4_SAFE", "")
        if specific != "":
            enabled = env_flag(
                "MICROGEMM_I8G_SAT_SAFE_ROW_PAIR_TILE4",
                env_flag(
                    "MICROGEMM_I8G_ROW_PAIR_TILE4_SAFE",
                    env_flag("MICROGEMM_GROUPWISE_I8_ROW_PAIR_TILE4_SAFE", False),
                ),
            )
            return "on(sat-safe)" if enabled and quant == "int8g128" else "off(sat-safe)"
        return "auto(sat-safe,batch>=4)"
    all_specific = os.environ.get("MICROGEMM_I8G_ROW_PAIR_TILE4_ALL", "")
    if all_specific == "":
        all_specific = os.environ.get("MICROGEMM_GROUPWISE_I8_ROW_PAIR_TILE4_ALL", "")
    if all_specific != "":
        enabled = env_flag(
            "MICROGEMM_I8G_ROW_PAIR_TILE4_ALL",
            env_flag("MICROGEMM_GROUPWISE_I8_ROW_PAIR_TILE4_ALL", False),
        )
        return "on(all)" if enabled and quant == "int8g128" else "off"
    specific = os.environ.get("MICROGEMM_I8G_ROW_PAIR_TILE4", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_I8_ROW_PAIR_TILE4", "")
    if specific != "":
        enabled = env_flag("MICROGEMM_I8G_ROW_PAIR_TILE4", env_flag("MICROGEMM_GROUPWISE_I8_ROW_PAIR_TILE4", False))
        return "on(down)" if enabled and quant == "int8g128" else "off"
    if quant == "int8g128":
        return "auto(int8g,down,batch>=4)"
    return "off"


def groupwise_i8_saturation_safe_text(quant: str) -> str:
    if any(
        os.environ.get(name, "") != ""
        for name in (
            "MICROGEMM_I8G_SATURATION_SAFE",
            "MICROGEMM_GROUPWISE_I8_SATURATION_SAFE",
            "MICROGEMM_I8G_SAT_SAFE",
        )
    ):
        return "on" if groupwise_i8_saturation_safe_enabled(quant) else "off"
    return "off(default,env-only)"


def groupwise_i8_gate_tile8_text(quant: str) -> str:
    if groupwise_i8_saturation_safe_enabled(quant):
        return "off(sat-safe)"
    specific = os.environ.get("MICROGEMM_I8G_GATE_UP_TILE8_EXPLICIT", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_I8_GATE_UP_TILE8_EXPLICIT", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_I8G_GATE_UP_TILE8_EXPLICIT",
            env_flag("MICROGEMM_GROUPWISE_I8_GATE_UP_TILE8_EXPLICIT", False),
        )
        return "on" if enabled and quant == "int8g128" else "off"
    if quant == "int8g128":
        return "auto(int8g,batch>=8)"
    return "off"


def groupwise_i8_gate_aligned128_text(quant: str) -> str:
    if groupwise_i8_saturation_safe_enabled(quant):
        return "off(sat-safe)"
    specific = os.environ.get("MICROGEMM_I8G_GATE8_ALIGNED128", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_I8_GATE8_ALIGNED128", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_I8G_GATE8_ALIGNED128",
            env_flag("MICROGEMM_GROUPWISE_I8_GATE8_ALIGNED128", False),
        )
        return "on" if enabled and quant == "int8g128" else "off"
    return "off"


def groupwise_i8_gate_biased_input_text(quant: str) -> str:
    if groupwise_i8_saturation_safe_enabled(quant):
        return "off(sat-safe)"
    specific = os.environ.get("MICROGEMM_I8G_GATE8_BIASED_INPUT", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_I8_GATE8_BIASED_INPUT", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_I8G_GATE8_BIASED_INPUT",
            env_flag("MICROGEMM_GROUPWISE_I8_GATE8_BIASED_INPUT", False),
        )
        return "on" if enabled and quant == "int8g128" else "off"
    if quant == "int8g128":
        return "auto(int8g,batch>=8,cols%128=0)"
    return "off"


def groupwise_i8_lm_head_scores8_text(quant: str) -> str:
    if groupwise_i8_saturation_safe_enabled(quant):
        return "off(sat-safe)"
    specific = os.environ.get("MICROGEMM_I8G_LM_HEAD_SCORES8", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_I8_LM_HEAD_SCORES8", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_I8G_LM_HEAD_SCORES8",
            env_flag("MICROGEMM_GROUPWISE_I8_LM_HEAD_SCORES8", False),
        )
        return "on" if enabled and quant == "int8g128" else "off"
    if quant == "int8g128":
        return "auto(int8g,lm_head,batch>=8)"
    return "off"


def groupwise_i8_gate_prefetch_text(quant: str) -> str:
    if groupwise_i8_saturation_safe_enabled(quant):
        return "off(sat-safe)"
    specific = os.environ.get("MICROGEMM_I8G_GATE_PREFETCH", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_I8_GATE_PREFETCH", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_I8G_GATE_PREFETCH",
            env_flag("MICROGEMM_GROUPWISE_I8_GATE_PREFETCH", False),
        )
        return "on" if enabled and quant == "int8g128" else "off"
    return "off"


def groupwise_i8_gate_pair_text(quant: str) -> str:
    if groupwise_i8_saturation_safe_enabled(quant):
        return "off(sat-safe)"
    specific = os.environ.get("MICROGEMM_I8G_GATE_PAIR4", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_I8_GATE_PAIR4", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_I8G_GATE_PAIR4",
            env_flag("MICROGEMM_GROUPWISE_I8_GATE_PAIR4", False),
        )
        return "on" if enabled and quant == "int8g128" else "off"
    if quant == "int8g128":
        return "auto(int8g,batch%8=0,cols%128=0)"
    return "off"


def groupwise_i8_gate_pair_unroll64_text(quant: str) -> str:
    if groupwise_i8_saturation_safe_enabled(quant):
        return "off(sat-safe)"
    specific = os.environ.get("MICROGEMM_I8G_GATE_PAIR4_UNROLL64", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_I8_GATE_PAIR4_UNROLL64", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_I8G_GATE_PAIR4_UNROLL64",
            env_flag("MICROGEMM_GROUPWISE_I8_GATE_PAIR4_UNROLL64", False),
        )
        return "on" if enabled and quant == "int8g128" else "off"
    return "off(default,env-only)"


def groupwise_i8_gate_pair_unroll128_text(quant: str) -> str:
    if groupwise_i8_saturation_safe_enabled(quant):
        return "off(sat-safe)"
    specific = os.environ.get("MICROGEMM_I8G_GATE_PAIR4_UNROLL128", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_I8_GATE_PAIR4_UNROLL128", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_I8G_GATE_PAIR4_UNROLL128",
            env_flag("MICROGEMM_GROUPWISE_I8_GATE_PAIR4_UNROLL128", False),
        )
        return "on" if enabled and quant == "int8g128" else "off"
    return "off(default,env-only)"


def groupwise_i8_gate_pair8_split_text(quant: str) -> str:
    if groupwise_i8_saturation_safe_enabled(quant):
        return "off(sat-safe)"
    specific = os.environ.get("MICROGEMM_I8G_GATE_PAIR8_SPLITPASS", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_I8_GATE_PAIR8_SPLITPASS", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_I8G_GATE_PAIR8_SPLITPASS",
            env_flag("MICROGEMM_GROUPWISE_I8_GATE_PAIR8_SPLITPASS", False),
        )
        return "on" if enabled and quant == "int8g128" else "off"
    return "off(default,env-only)"


def groupwise_i8_lm_head_row_pair_text(quant: str) -> str:
    if groupwise_i8_saturation_safe_enabled(quant):
        specific = os.environ.get("MICROGEMM_I8G_SAT_SAFE_LM_HEAD_ROWPAIR4", "")
        if specific == "":
            specific = os.environ.get("MICROGEMM_I8G_LM_HEAD_ROWPAIR4_SAFE", "")
        if specific == "":
            specific = os.environ.get("MICROGEMM_GROUPWISE_I8_LM_HEAD_ROWPAIR4_SAFE", "")
        if specific == "":
            specific = os.environ.get("MICROGEMM_I8G_LM_HEAD_ROWPAIR4", "")
        if specific == "":
            specific = os.environ.get("MICROGEMM_GROUPWISE_I8_LM_HEAD_ROWPAIR4", "")
        if specific != "":
            enabled = (
                env_flag("MICROGEMM_I8G_SAT_SAFE_LM_HEAD_ROWPAIR4", False)
                or env_flag("MICROGEMM_I8G_LM_HEAD_ROWPAIR4_SAFE", False)
                or env_flag("MICROGEMM_GROUPWISE_I8_LM_HEAD_ROWPAIR4_SAFE", False)
                or env_flag("MICROGEMM_I8G_LM_HEAD_ROWPAIR4", False)
                or env_flag("MICROGEMM_GROUPWISE_I8_LM_HEAD_ROWPAIR4", False)
            )
            return "on(sat-safe)" if enabled and quant == "int8g128" else "off(sat-safe)"
        return "auto(sat-safe,lm_head,batch>=4)"
    specific = os.environ.get("MICROGEMM_I8G_LM_HEAD_ROWPAIR4", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_I8_LM_HEAD_ROWPAIR4", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_I8G_LM_HEAD_ROWPAIR4",
            env_flag("MICROGEMM_GROUPWISE_I8_LM_HEAD_ROWPAIR4", False),
        )
        return "on" if enabled and quant == "int8g128" else "off"
    return "off"


def lm_head_stack_best_text() -> str:
    specific = os.environ.get("MICROGEMM_LM_HEAD_STACK_BEST", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_GROUPWISE_LM_HEAD_STACK_BEST", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_LM_HEAD_STACK_BEST",
            env_flag("MICROGEMM_GROUPWISE_LM_HEAD_STACK_BEST", False),
        )
        return "on" if enabled else "off"
    return "off(default,env-only)"


def prefill_parallel_attention_text() -> str:
    specific = os.environ.get("MICROGEMM_PREFILL_PARALLEL_ATTENTION", "")
    if specific != "":
        return "on" if env_flag("MICROGEMM_PREFILL_PARALLEL_ATTENTION", False) else "off"
    return "off(default,env-only)"


def rmsnorm_prequant_text(quant: str) -> str:
    specific = os.environ.get("MICROGEMM_RMSNORM_PREQUANT", "")
    if specific != "":
        enabled = env_flag("MICROGEMM_RMSNORM_PREQUANT", False)
        return "on" if enabled and quant in {"int8", "int8g128"} else "off"
    return "off(default,env-only)"


def swiglu_down_prequant_text(quant: str) -> str:
    specific = os.environ.get("MICROGEMM_SWIGLU_DOWN_PREQUANT", "")
    if specific == "":
        specific = os.environ.get("MICROGEMM_MLP_DOWN_PREQUANT", "")
    if specific != "":
        enabled = env_flag(
            "MICROGEMM_SWIGLU_DOWN_PREQUANT",
            env_flag("MICROGEMM_MLP_DOWN_PREQUANT", False),
        )
        if enabled and quant == "int8g128":
            return "on(fused-amax)"
        return "on" if enabled and quant in {"int8", "int8g128"} else "off"
    return "off(default,env-only)"


def print_int4_kernel_toggles(quant: str) -> None:
    umbrella = os.environ.get("MICROGEMM_I4_TILE8_SPLIT", "")
    row_default = env_flag("MICROGEMM_I4_TILE8_SPLIT", True)
    gate_split_default = env_flag("MICROGEMM_I4_TILE8_SPLIT", False)
    row_pair_env = os.environ.get("MICROGEMM_I4_ROW_PAIR_TILE4", "")
    row_split = env_flag("MICROGEMM_I4_ROW_TILE8_SPLIT", row_default)
    row_pair = env_flag("MICROGEMM_I4_ROW_PAIR_TILE4", False)
    gate_split = env_flag("MICROGEMM_I4_GATE_TILE8_SPLIT", gate_split_default)
    gate_group4 = env_flag("MICROGEMM_I4_GATE_TILE8_GROUP4", False)
    gate_fused = env_flag("MICROGEMM_I4_GATE_TILE8_FUSED", False)
    groupwise_gate_fused = groupwise_gate_fused_text(quant)
    groupwise_compact_stride = groupwise_compact_mlp_stride_text(quant)
    groupwise_i4_row_pair = groupwise_i4_row_pair_text(quant)
    lm_stack_best = lm_head_stack_best_text()
    prefill_parallel_attention = prefill_parallel_attention_text()
    rmsnorm_prequant = rmsnorm_prequant_text(quant)
    swiglu_down_prequant = swiglu_down_prequant_text(quant)
    linear_delta_vec = env_flag("MICROGEMM_LINEAR_DELTA_VEC", True)
    print("Int4 kernel toggles")
    print(f"  tile8 env:     {umbrella if umbrella else '(default)'}")
    print(f"  row split:     {'on' if row_split else 'off'}")
    print(f"  row pair4:     {'off' if row_pair_env == '' else ('on' if row_pair else 'off')}")
    print(f"  gate split:    {'on' if gate_split else 'off'}")
    print(f"  gate group4:   {'on' if gate_group4 else 'off'}")
    print(f"  gate fused8:   {'on' if gate_fused else 'off'}")
    print(f"  gw gate fused: {groupwise_gate_fused}")
    print(f"  gw compact mlp:{groupwise_compact_stride}")
    print(f"  i4g row pair4: {groupwise_i4_row_pair}")
    print(f"  lm stack best: {lm_stack_best}")
    print(f"  prefill attn:  {prefill_parallel_attention}")
    print(f"  rms prequant:  {rmsnorm_prequant}")
    print(f"  swiglu preq:   {swiglu_down_prequant}")
    print(f"  delta vec:     {'on' if linear_delta_vec else 'off'}")


def print_int8_kernel_toggles(quant: str) -> None:
    gate_group4_env = os.environ.get("MICROGEMM_I8_GATE_TILE8_GROUP4", "")
    gate_group4 = env_flag("MICROGEMM_I8_GATE_TILE8_GROUP4", False)
    gate_tile4_unroll64 = env_flag("MICROGEMM_I8_GATE_TILE4_UNROLL64", False)
    gate_fused = env_flag("MICROGEMM_I8_GATE_TILE8_FUSED", False)
    groupwise_gate_fused = groupwise_gate_fused_text(quant)
    groupwise_i8_gate_safe_combined = groupwise_i8_gate_safe_combined_text(quant)
    groupwise_i8_gate_safe_combined_tile8 = groupwise_i8_gate_safe_combined_tile8_text(quant)
    groupwise_compact_stride = groupwise_compact_mlp_stride_text(quant)
    groupwise_i8_row_pair = groupwise_i8_row_pair_text(quant)
    groupwise_i8_saturation_safe = groupwise_i8_saturation_safe_text(quant)
    groupwise_i8_gate_tile8 = groupwise_i8_gate_tile8_text(quant)
    groupwise_i8_gate_aligned128 = groupwise_i8_gate_aligned128_text(quant)
    groupwise_i8_gate_biased_input = groupwise_i8_gate_biased_input_text(quant)
    groupwise_i8_gate_pair = groupwise_i8_gate_pair_text(quant)
    groupwise_i8_gate_pair_unroll64 = groupwise_i8_gate_pair_unroll64_text(quant)
    groupwise_i8_gate_pair_unroll128 = groupwise_i8_gate_pair_unroll128_text(quant)
    groupwise_i8_gate_pair8_split = groupwise_i8_gate_pair8_split_text(quant)
    groupwise_i8_gate_prefetch = groupwise_i8_gate_prefetch_text(quant)
    groupwise_i8_lm_scores8 = groupwise_i8_lm_head_scores8_text(quant)
    groupwise_i8_lm_row_pair = groupwise_i8_lm_head_row_pair_text(quant)
    lm_stack_best = lm_head_stack_best_text()
    prefill_parallel_attention = prefill_parallel_attention_text()
    rmsnorm_prequant = rmsnorm_prequant_text(quant)
    swiglu_down_prequant = swiglu_down_prequant_text(quant)
    gate_unroll64_env = os.environ.get("MICROGEMM_I8_GATE_TILE8_UNROLL64", "")
    gate_unroll64 = env_flag("MICROGEMM_I8_GATE_TILE8_UNROLL64", False)
    down_tile8 = env_flag("MICROGEMM_I8_DOWN_TILE8", False)
    linear_delta_vec = env_flag("MICROGEMM_LINEAR_DELTA_VEC", True)
    if gate_group4_env == "":
        gate_group4_text = "auto(cols>=4096)"
    else:
        gate_group4_text = "on" if gate_group4 else "off"
    if gate_unroll64_env == "":
        gate_unroll64_text = "auto(cols>=4096)"
    else:
        gate_unroll64_text = "on" if gate_unroll64 else "off"
    print("Int8 kernel toggles")
    print(f"  gate group4:   {gate_group4_text}")
    print(f"  gate tile4 u64:{'on' if gate_tile4_unroll64 else 'off'}")
    print(f"  gate fused8:   {'on' if gate_fused else 'off'}")
    print(f"  gw gate fused: {groupwise_gate_fused}")
    print(f"  i8g gate combo:{groupwise_i8_gate_safe_combined}")
    print(f"  i8g gate comb8:{groupwise_i8_gate_safe_combined_tile8}")
    print(f"  gw compact mlp:{groupwise_compact_stride}")
    print(f"  i8g gate8:     {groupwise_i8_gate_tile8}")
    print(f"  i8g gate8 a128:{groupwise_i8_gate_aligned128}")
    print(f"  i8g gate8 bias:{groupwise_i8_gate_biased_input}")
    print(f"  i8g gate pair4:{groupwise_i8_gate_pair}")
    print(f"  i8g gate p4u64:{groupwise_i8_gate_pair_unroll64}")
    print(f"  i8g gate p4u128:{groupwise_i8_gate_pair_unroll128}")
    print(f"  i8g gate p8spl:{groupwise_i8_gate_pair8_split}")
    print(f"  i8g gate pre:  {groupwise_i8_gate_prefetch}")
    print(f"  i8g lm scores8:{groupwise_i8_lm_scores8}")
    print(f"  i8g lm pair4:  {groupwise_i8_lm_row_pair}")
    print(f"  i8g row pair4: {groupwise_i8_row_pair}")
    print(f"  i8g sat safe:  {groupwise_i8_saturation_safe}")
    print(f"  lm stack best: {lm_stack_best}")
    print(f"  prefill attn:  {prefill_parallel_attention}")
    print(f"  rms prequant:  {rmsnorm_prequant}")
    print(f"  swiglu preq:   {swiglu_down_prequant}")
    print(f"  gate unroll64: {gate_unroll64_text}")
    print(f"  down tile8:    {'on' if down_tile8 else 'off'}")
    print(f"  delta vec:     {'on' if linear_delta_vec else 'off'}")


def canonical_quant(raw: str) -> str:
    return QUANT_ALIASES.get(raw, raw)


def mgm_filename_for_quant(quant: str) -> str:
    if quant == "int8":
        return "model.mgm"
    if quant == "int4":
        return "model_int4_rowsum.mgm"
    if quant == "int8g128":
        return "model_int8g128.mgm"
    if quant == "int4g128":
        return "model_int4g128.mgm"
    raise ValueError(f"unsupported quant mode: {quant}")


def infer_canary_profile(model_id: str) -> dict[str, Any]:
    key = model_id.lower()
    if "qwen3.5" in key or "qwen3_5" in key or "qwen3-5" in key or "qwen35" in key:
        return dict(QWEN35_TEXT_CANARY_PROFILE)
    if "qwen3-0.6b" in key or "qwen3_06b" in key or "qwen3-06b" in key:
        return dict(QWEN3_06B_CANARY_PROFILE)
    if "llama-3.2-1b" in key or "llama32_1b" in key:
        return dict(LLAMA32_1B_CANARY_PROFILE)
    if "llama-3.1-8b" in key or "meta-llama-3.1-8b" in key or "llama31_8b" in key:
        return dict(LLAMA31_8B_CANARY_PROFILE)
    if "mistral-7b" in key or "mistral7b" in key:
        return dict(MISTRAL7B_CANARY_PROFILE)
    if "microsoft/phi-4" in key or "phi4_14b" in key or key.endswith("phi-4"):
        return dict(PHI4_14B_CANARY_PROFILE)
    if "phi-4-mini" in key or "phi4_mini" in key or "phi4-mini" in key:
        return dict(PHI4_MINI_CANARY_PROFILE)
    if "phi-3-mini" in key or "phi3_mini" in key or "phi3-mini" in key:
        return dict(PHI3_MINI_CANARY_PROFILE)
    if "granite-3.3-2b" in key or "granite33_2b" in key:
        return dict(GRANITE33_2B_CANARY_PROFILE)
    if "glm-4-9b-0414" in key or "glm4_9b_0414" in key or "glm4-9b-0414" in key:
        return dict(GLM4_9B_0414_CANARY_PROFILE)
    return {
        "name": "qwen25_05b_cpu_avx2",
        "run_good_decode_tps": RUN_CANARY_GOOD_DECODE_TPS,
        "run_weak_decode_tps": RUN_CANARY_WEAK_DECODE_TPS,
        "batch_good_runtime_tps_by_size": dict(BATCH_CANARY_GOOD_RUNTIME_TPS_BY_SIZE),
        "batch_weak_runtime_tps_by_size": dict(BATCH_CANARY_WEAK_RUNTIME_TPS_BY_SIZE),
    }


def run_canary_thresholds(canary_profile: dict[str, Any] | None = None) -> tuple[float, float]:
    profile = canary_profile or {}
    return (
        float(profile.get("run_good_decode_tps", RUN_CANARY_GOOD_DECODE_TPS) or RUN_CANARY_GOOD_DECODE_TPS),
        float(profile.get("run_weak_decode_tps", RUN_CANARY_WEAK_DECODE_TPS) or RUN_CANARY_WEAK_DECODE_TPS),
    )


def summarize_run_canary(
    context_results: list[dict[str, Any]],
    cpu_snapshot: dict[str, Any],
    canary_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    good_threshold, weak_threshold = run_canary_thresholds(canary_profile)
    selected: dict[str, Any] | None = None
    for item in context_results:
        summary = item.get("summary", {})
        if summary.get("kind") != "single_context":
            continue
        if int(summary.get("target_prompt_tokens", 0) or 0) == RUN_CANARY_PROMPT_TOKENS:
            selected = summary
            break
        if selected is None:
            selected = summary

    if selected is None:
        return {
            "status": "missing",
            "reason": "no single-context result",
            "prompt_tokens": RUN_CANARY_PROMPT_TOKENS,
            "decode_tps": 0.0,
            "good_threshold": good_threshold,
            "weak_threshold": weak_threshold,
            "profile": (canary_profile or {}).get("name", ""),
        }

    decode_tps = float(selected.get("decode_tps", {}).get("median", 0.0) or 0.0)
    prompt_tokens = int(selected.get("target_prompt_tokens", RUN_CANARY_PROMPT_TOKENS) or 0)
    if decode_tps >= good_threshold:
        status = "good"
        reason = "single decode canary is in the comparable range"
    elif decode_tps >= weak_threshold:
        status = "weak"
        reason = "single decode canary is below the best comparable runs"
    else:
        status = "slow"
        reason = "single decode canary is too low for kernel comparisons"

    return {
        "status": status,
        "reason": reason,
        "prompt_tokens": prompt_tokens,
        "decode_tps": decode_tps,
        "good_threshold": good_threshold,
        "weak_threshold": weak_threshold,
        "profile": (canary_profile or {}).get("name", ""),
        "cpu_model": cpu_snapshot.get("model", ""),
        "cpu_mhz_median": cpu_snapshot.get("mhz_median", 0.0),
        "loadavg_1m": cpu_snapshot.get("loadavg_1m", 0.0),
    }


def print_run_canary(canary: dict[str, Any]) -> None:
    print("Run canary")
    print(
        "  single decode: "
        f"prompt={int(canary.get('prompt_tokens', 0) or 0)} "
        f"median={float(canary.get('decode_tps', 0.0) or 0.0):.2f} tok/s"
    )
    print(
        "  verdict:      "
        f"{canary.get('status', 'missing')} - {canary.get('reason', '')}"
    )
    print(
        "  thresholds:   "
        f"good>={float(canary.get('good_threshold', 0.0) or 0.0):.0f} "
        f"weak>={float(canary.get('weak_threshold', 0.0) or 0.0):.0f}"
    )
    if canary.get("profile"):
        print(f"  profile:      {canary.get('profile')}")


def write_suite_results(
    args: argparse.Namespace,
    run_id: str,
    cpu_snapshot: dict[str, Any],
    run_canary: dict[str, Any],
    batch_canary: dict[str, Any] | None,
    all_results: list[dict[str, Any]],
) -> tuple[Path, Path]:
    rows = flatten_summary_rows(all_results)
    batch_canary_status = ""
    batch_canary_runtime_tps = 0.0
    batch_canary_decode_only_tps = 0.0
    batch_canary_wall_total_tps = 0.0
    if batch_canary is not None:
        batch_summary = batch_canary.get("summary", {})
        batch_canary_status = str(batch_summary.get("canary_status", "") or "")
        batch_canary_runtime_tps = float(
            batch_summary.get("runtime_output_tps", {}).get("median", 0.0) or 0.0
        )
        batch_canary_decode_only_tps = float(
            batch_summary.get("decode_only_output_tps", {}).get("median", 0.0) or 0.0
        )
        batch_canary_wall_total_tps = float(
            batch_summary.get("wall_total_tps", {}).get("median", 0.0) or 0.0
        )
    for row in rows:
        row.setdefault("suite_build", SUITE_BUILD_TAG)
        row.setdefault("quant", getattr(args, "quant", "int8"))
        row.setdefault("run_canary_status", run_canary.get("status", ""))
        row.setdefault("run_canary_decode_tps", run_canary.get("decode_tps", 0.0))
        row.setdefault("batch_canary_status", batch_canary_status)
        row.setdefault("batch_canary_runtime_tps", batch_canary_runtime_tps)
        row.setdefault("batch_canary_decode_only_tps", batch_canary_decode_only_tps)
        row.setdefault("batch_canary_wall_total_tps", batch_canary_wall_total_tps)
    payload = {
        "benchmark": "qwen25_microgemm_cpu_suite",
        "run_id": run_id,
        "suite_build": SUITE_BUILD_TAG,
        "cpu_snapshot": cpu_snapshot,
        "run_canary": run_canary,
        "batch_canary": batch_canary,
        "config": {
            "model_repo": args.model_repo,
            "model_dir": str(getattr(args, "resolved_model_dir", "")),
            "mgm_path": str(getattr(args, "resolved_mgm_path", "")),
            "threads": args.threads,
            "prompt_tokens": parse_csv_ints(args.prompt_tokens),
            "batch_prompt_tokens": args.batch_prompt_tokens,
            "batch_sizes": parse_csv_ints(args.batch_sizes),
            "batch_modes": [m.strip() for m in args.batch_modes.split(",") if m.strip()],
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": bool(args.ignore_eos),
            "max_seq_len": args.max_seq_len,
            "kv_block_size": args.kv_block_size,
            "quant": getattr(args, "quant", "int8"),
            "allow_slow_canary": bool(args.allow_slow_canary),
            "allow_weak_canary": bool(args.allow_weak_canary),
            "strict_canary_gate": bool(getattr(args, "strict_canary_gate", False)),
            "skip_batch_canary": bool(getattr(args, "skip_batch_canary", False)),
            "canary_profile": getattr(args, "canary_profile", {}),
            "batch_canary_size": BATCH_CANARY_SIZE,
            "batch_canary_sizes": list(BATCH_CANARY_SIZES),
            "batch_canary_mode": BATCH_CANARY_MODE,
            "batch_canary_max_new_tokens": BATCH_CANARY_MAX_NEW_TOKENS,
            "batch_canary_runs": BATCH_CANARY_RUNS,
            "batch_canary_max_attempts": BATCH_CANARY_MAX_ATTEMPTS,
            "batch_canary_good_runtime_tps": BATCH_CANARY_GOOD_RUNTIME_TPS,
            "batch_canary_weak_runtime_tps": BATCH_CANARY_WEAK_RUNTIME_TPS,
            "batch_canary_good_runtime_tps_by_size": BATCH_CANARY_GOOD_RUNTIME_TPS_BY_SIZE,
            "batch_canary_weak_runtime_tps_by_size": BATCH_CANARY_WEAK_RUNTIME_TPS_BY_SIZE,
        },
        "results": all_results,
        "summary_rows": rows,
    }
    return write_outputs(Path(args.out_dir), run_id, payload, rows)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def stats(values: list[float]) -> dict[str, float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": min(clean),
        "median": statistics.median(clean),
        "mean": statistics.mean(clean),
        "p95": percentile(clean, 0.95),
        "max": max(clean),
    }


def compact_error_text(raw: Any, limit: int = 240) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def first_failure_info(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("ok"):
            continue
        info = {
            "returncode": row.get("returncode", ""),
            "stderr": row.get("stderr", ""),
            "stdout": row.get("stdout", ""),
        }
        if info["stderr"] or info["stdout"] or info["returncode"] != "":
            return info
        for worker in row.get("workers", []) or []:
            if isinstance(worker, dict) and not worker.get("ok"):
                return {
                    "returncode": worker.get("returncode", ""),
                    "stderr": worker.get("stderr", ""),
                    "stdout": worker.get("stdout", ""),
                }
    return {"returncode": "", "stderr": "", "stdout": ""}


def scale_stats(values: dict[str, Any], factor: float) -> dict[str, float]:
    return {
        "min": float(values.get("min", 0.0) or 0.0) * factor,
        "median": float(values.get("median", 0.0) or 0.0) * factor,
        "mean": float(values.get("mean", 0.0) or 0.0) * factor,
        "p95": float(values.get("p95", 0.0) or 0.0) * factor,
        "max": float(values.get("max", 0.0) or 0.0) * factor,
    }


def add_batch_total_tps(summary: dict[str, Any]) -> None:
    batch_size = int(summary.get("batch_size", 0) or 0)
    prompt_tokens = int(summary.get("target_prompt_tokens", 0) or 0)
    generated_tokens = float(summary.get("generated_tokens_median", 0.0) or 0.0)
    prompt_total_tokens = float(summary.get("prompt_tokens_median", 0.0) or 0.0)
    if prompt_total_tokens <= 0.0:
        prompt_total_tokens = float(batch_size * prompt_tokens)
    total_tokens = float(summary.get("total_tokens_median", 0.0) or 0.0)
    if total_tokens <= 0.0:
        total_tokens = prompt_total_tokens + generated_tokens
    factor = total_tokens / generated_tokens if generated_tokens > 0.0 else 0.0

    summary["prompt_tokens_total_median"] = prompt_total_tokens
    summary["total_tokens_median"] = total_tokens
    summary["total_to_output_token_ratio"] = factor
    if "wall_total_tps" not in summary:
        summary["wall_total_tps"] = scale_stats(summary.get("wall_output_tps", {}), factor)
    if "steady_total_tps" not in summary:
        summary["steady_total_tps"] = scale_stats(summary.get("steady_output_tps", {}), factor)
    if "runtime_total_tps" not in summary:
        summary["runtime_total_tps"] = scale_stats(summary.get("runtime_output_tps", {}), factor)


def run_checked(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = run_capture(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
    return result


def ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])


def ensure_text_profile_feature(micro: Path, text_bin: Path) -> None:
    text_src = micro / "src" / "microgemm_text.cpp"
    decode_src = micro / "src" / "microgemm_decode_cpu.c"
    missing_sources: list[str] = []
    if not text_src.exists() or TEXT_PROFILE_MARKER not in text_src.read_text(encoding="utf-8", errors="ignore"):
        missing_sources.append(f"{text_src} missing {TEXT_PROFILE_MARKER}")
    if not decode_src.exists() or DECODE_PROFILE_MARKER not in decode_src.read_text(encoding="utf-8", errors="ignore"):
        missing_sources.append(f"{decode_src} missing {DECODE_PROFILE_MARKER}")
    if missing_sources:
        raise SystemExit(
            "MicroGemm C sources are stale for this profiling suite.\n"
            "missing from:\n  - "
            + "\n  - ".join(missing_sources)
            + "\nSync/copy the updated microgemm/src files before rerunning."
        )
    try:
        binary = text_bin.read_bytes()
    except OSError as exc:
        raise SystemExit(f"failed to read built microgemm-text binary: {text_bin}: {exc}") from exc
    if TEXT_PROFILE_MARKER.encode("utf-8") not in binary:
        raise SystemExit(
            "Built microgemm-text does not contain the profiling marker.\n"
            f"binary: {text_bin}\n"
            f"expected marker: {TEXT_PROFILE_MARKER}\n"
            "Delete the cached build directory and rerun, or rebuild MicroGemm from the updated sources."
        )
    print(f"  C profile:    ok ({TEXT_PROFILE_MARKER})", flush=True)


def ensure_kernel_selftest(cli_bin: Path) -> None:
    result = run_capture([str(cli_bin), "kernel-selftest"], cwd=cli_bin.parent)
    if result.returncode != 0:
        raise SystemExit(
            "microgemm kernel-selftest failed.\n"
            f"cmd: {cli_bin} kernel-selftest\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
    print("  C selftest:   ok (kernel-selftest)", flush=True)


def download_model(repo_id: str, cache_dir: Path) -> Path:
    ensure_huggingface_hub()
    from huggingface_hub import snapshot_download

    local_dir = cache_dir / "hf" / repo_id.replace("/", "__")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
            "*.safetensors",
            "*.safetensors.index.json",
        ],
    )
    return Path(path).resolve()


def ensure_supported_safetensors(model_dir: Path) -> None:
    single = model_dir / "model.safetensors"
    if single.exists():
        return
    shards = sorted(model_dir.glob("*.safetensors"))
    index_files = sorted(model_dir.glob("*.safetensors.index.json"))
    if shards and index_files:
        return
    details = "\n".join(f"  - {p.name}" for p in shards[:16])
    if index_files:
        details += "\nindex files:\n" + "\n".join(f"  - {p.name}" for p in index_files)
    raise SystemExit(
        "MicroGemm converter needs either model.safetensors or sharded *.safetensors files "
        "with a *.safetensors.index.json file.\n"
        f"model_dir: {model_dir}\n"
        f"found safetensors:\n{details or '  none'}\n"
        "Rerun the suite with the updated downloader so the shard files are fetched."
    )


def ensure_microgemm_built(root: Path, cache_dir: Path | None = None) -> tuple[Path, Path, Path]:
    source_micro = root / "microgemm"
    cache_dir = cache_dir or Path(os.environ.get("MICROGEMM_QWEN25_CACHE_DIR", "/content/microgemm_qwen25_cache"))
    micro = stage_microgemm_tree(source_micro, cache_dir) if is_colab_drive_path(source_micro) else source_micro
    text_bin = micro / "microgemm-text"
    convert_bin = micro / "microgemm-convert"
    cli_bin = micro / "microgemm"
    if sys.platform == "win32":
        text_bin = micro / "microgemm-text.exe"
        convert_bin = micro / "microgemm-convert.exe"
        cli_bin = micro / "microgemm.exe"

    bins = (cli_bin, convert_bin, text_bin)
    bins_ready = all(path.exists() for path in bins)
    sources: list[Path] = []
    for dirname in ("src", "include"):
        source_root = micro / dirname
        if source_root.exists():
            sources.extend(path for path in source_root.rglob("*") if path.is_file())
    for filename in ("Makefile", "CMakeLists.txt", "build.ps1"):
        source_path = micro / filename
        if source_path.exists():
            sources.append(source_path)

    needs_build = not bins_ready
    if bins_ready and sources:
        newest_source = max(path.stat().st_mtime for path in sources)
        oldest_bin = min(path.stat().st_mtime for path in bins)
        needs_build = newest_source > oldest_bin

    if not needs_build:
        for path in bins:
            ensure_executable(path)
        ensure_text_profile_feature(micro, text_bin)
        ensure_kernel_selftest(cli_bin)
        return cli_bin, convert_bin, text_bin

    if sys.platform == "win32":
        ps1 = micro / "build.ps1"
        if not ps1.exists():
            raise SystemExit(f"missing build script: {ps1}")
        run_checked(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
            cwd=micro,
        )
    else:
        if not (micro / "Makefile").exists():
            raise SystemExit(f"missing Makefile in {micro}")
        jobs = str(max(1, os.cpu_count() or 1))
        run_checked(["make", "-j", jobs], cwd=micro)

    if not all(path.exists() for path in bins):
        raise SystemExit("MicroGemm build finished but expected binaries were not found")
    for path in bins:
        ensure_executable(path)
    ensure_text_profile_feature(micro, text_bin)
    ensure_kernel_selftest(cli_bin)
    return cli_bin, convert_bin, text_bin


def ensure_mgm(
    convert_bin: Path,
    model_dir: Path,
    mgm_path: Path,
    *,
    force: bool,
    kv_block_size: int,
    quant: str,
) -> None:
    if force and mgm_path.exists():
        mgm_path.unlink()
    if mgm_path.exists():
        return
    mgm_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(convert_bin),
        "from-dir",
        str(model_dir),
        str(mgm_path),
        "--kv-block-size",
        str(kv_block_size),
        "--quant",
        quant,
    ]
    run_checked(cmd, cwd=convert_bin.parent)


def probe_text_profile_runtime(text_bin: Path, mgm_path: Path, tokenizer_json: Path, threads: int) -> None:
    cmd = [
        str(text_bin),
        "batch-generate",
        str(mgm_path),
        str(tokenizer_json),
        "--prompt",
        "teste curto",
        "--prompt",
        "teste curto dois",
        "--max-new-tokens",
        "2",
        "--max-seq-len",
        "64",
        "--temperature",
        "0.0",
        "--top-k",
        "0",
        "--top-p",
        "1.0",
        "--ignore-eos",
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(max(1, threads))
    env["MICROGEMM_BATCH_TOTAL_THREADS"] = str(max(1, threads))
    env["MICROGEMM_BATCH_OUTER_THREADS"] = "2"
    env["MICROGEMM_BATCH_INNER_THREADS"] = str(max(1, threads // 2))
    env["MICROGEMM_BATCH_LM_HEAD_THREADS"] = str(max(1, threads))
    result = run_capture(cmd, env=env)
    if result.returncode != 0:
        raise SystemExit(
            "microgemm-text profile probe failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
    if TEXT_PROFILE_MARKER not in result.stdout:
        raise SystemExit(
            "microgemm-text profile probe did not emit the runtime marker.\n"
            f"expected marker: {TEXT_PROFILE_MARKER}\n"
            f"text binary: {text_bin}\n"
            f"stdout tail:\n{result.stdout[-3000:]}\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )
    print(f"  C runtime:    ok ({TEXT_PROFILE_MARKER})", flush=True)


def load_tokenizer(model_dir: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    except Exception:
        return None


def use_safe_ascii_prompts(model_id: str) -> bool:
    key = str(model_id or "").lower()
    return "mistral" in key or "gemma" in key or "phi" in key or "granite" in key or "glm" in key


def make_prompt(tokenizer: Any, target_tokens: int, *, safe_ascii: bool = False) -> tuple[str, int]:
    if tokenizer is None:
        repeats = max(1, target_tokens // 16)
        text = (DEFAULT_PROMPT_SEED * repeats).strip()
        return text, 0

    if safe_ascii:
        filler = " CPU MicroGemm benchmark de inferencia em lote com pesos quantizados."
        text = DEFAULT_PROMPT_SEED.strip()
        actual = len(tokenizer.encode(text, add_special_tokens=False))
        guard = 0
        while actual < target_tokens and guard < 512:
            text = f"{text}{filler}"
            actual = len(tokenizer.encode(text, add_special_tokens=False))
            guard += 1
        return text, int(actual)

    ids = tokenizer.encode(DEFAULT_PROMPT_SEED, add_special_tokens=False)
    filler_ids = tokenizer.encode(" CPU MicroGemm Qwen dois ponto cinco benchmark.", add_special_tokens=False)
    if not filler_ids:
        filler_ids = ids
    while len(ids) < target_tokens:
        ids.extend(filler_ids)
    ids = ids[:target_tokens]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    return text, int(actual)


def make_prompt_ids(tokenizer: Any, target_tokens: int) -> list[int]:
    if tokenizer is None:
        return []
    ids = tokenizer.encode(DEFAULT_PROMPT_SEED, add_special_tokens=False)
    filler_ids = tokenizer.encode(" CPU MicroGemm benchmark.", add_special_tokens=False)
    if not ids:
        ids = filler_ids[:]
    if not filler_ids:
        filler_ids = ids[:]
    while len(ids) < target_tokens:
        ids.extend(filler_ids)
    return [int(token_id) for token_id in ids[:target_tokens]]


def make_batch_prompt_ids(tokenizer: Any, target_tokens: int, batch_size: int) -> list[list[int]]:
    base_ids = make_prompt_ids(tokenizer, target_tokens)
    return [list(base_ids) for _ in range(batch_size)]


def make_batch_prompts(tokenizer: Any, target_tokens: int, batch_size: int, *, safe_ascii: bool = False) -> list[str]:
    base_prompt, _ = make_prompt(tokenizer, target_tokens, safe_ascii=safe_ascii)
    return [f"{base_prompt}\nIndice da requisicao: {idx}." for idx in range(batch_size)]


def count_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    if tokenizer is None:
        return 0
    try:
        return int(len(tokenizer.encode(prompt, add_special_tokens=False)))
    except Exception:
        return 0


def planned_max_seq_len(
    requested_max_seq_len: int,
    nominal_prompt_tokens: int,
    actual_prompt_tokens: int,
    max_new_tokens: int,
    *,
    margin: int = 64,
) -> int:
    if requested_max_seq_len > 0:
        return requested_max_seq_len
    prompt_tokens = max(int(nominal_prompt_tokens), int(actual_prompt_tokens), 1)
    return prompt_tokens + int(max_new_tokens) + int(margin)


def parse_microgemm_stdout(stdout: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    text_key: str | None = None
    text_lines: list[str] = []
    for line in stdout.splitlines():
        if line in {"generated_text:", "full_text:"}:
            if text_key is not None:
                out[text_key] = "\n".join(text_lines).strip()
            text_key = line[:-1]
            text_lines = []
            continue
        if text_key is not None:
            if ": " in line and not line.startswith(" "):
                out[text_key] = "\n".join(text_lines).strip()
                text_key = None
                text_lines = []
            else:
                text_lines.append(line)
                continue
        if ": " in line:
            key, value = line.split(": ", 1)
            out[key.strip()] = value.strip()
    if text_key is not None:
        out[text_key] = "\n".join(text_lines).strip()

    numeric_keys = {
        "prompt_token_count",
        "generated_token_count",
        "batch_size",
        "finished_request_count",
        "scheduler_iterations",
        "scheduler_outer_threads",
        "scheduler_inner_threads",
        "scheduler_lm_head_threads",
        "native_continuous_batching",
        "batched_decode",
        "batched_decode_calls",
        "batched_decode_tokens",
        "batched_lm_head",
        "batched_lm_head_calls",
        "batched_lm_head_tokens",
        "tokenizer_load_ms",
        "prompt_encode_ms",
        "model_open_ms",
        "model_load_ms",
        "model_cleanup_ms",
        "command_total_ms",
        "setup_ms",
        "prefill_ms",
        "decode_ms",
        "batch_profile_calls",
        "batch_profile_tokens",
        "batch_profile_emit_version",
        "batch_profile_total_ms",
        "batch_profile_alloc_ms",
        "batch_profile_embed_ms",
        "batch_profile_input_norm_ms",
        "batch_profile_qkv_ms",
        "batch_profile_rope_kv_ms",
        "batch_profile_attention_ms",
        "batch_profile_o_proj_ms",
        "batch_profile_post_norm_ms",
        "batch_profile_gate_up_ms",
        "batch_profile_gate_up_quant_ms",
        "batch_profile_gate_up_dot_ms",
        "batch_profile_activation_ms",
        "batch_profile_down_proj_ms",
        "batch_profile_down_proj_quant_ms",
        "batch_profile_down_proj_dot_ms",
        "batch_profile_final_norm_ms",
        "batch_profile_lm_head_ms",
        "batch_profile_groupwise_gemv_tile_calls",
        "batch_profile_groupwise_i8_row_pair_calls",
        "batch_profile_groupwise_i4_row_pair_calls",
        "batch_profile_groupwise_lm_head_argmax_calls",
        "batch_profile_lm_head_stack_best_calls",
        "batch_profile_groupwise_gate_up_fused_calls",
        "batch_profile_groupwise_i8_gate_safe_combined_calls",
        "batch_profile_groupwise_i8_gate_safe_combined_tile8_calls",
        "batch_profile_groupwise_i8_gate_tile8_calls",
        "batch_profile_groupwise_i8_gate_biased_calls",
        "batch_profile_groupwise_i8_gate_pair_calls",
        "batch_profile_groupwise_i8_gate_pair_unroll64_calls",
        "batch_profile_groupwise_i8_gate_pair_unroll128_calls",
        "batch_profile_groupwise_i8_gate_pair8_split_calls",
        "batch_profile_groupwise_i8_gate_prefetch_calls",
        "batch_profile_groupwise_lm_head_row_pair_calls",
        "batch_profile_copy_ms",
        "batch_profile_cleanup_ms",
        "total_ms",
        "loaded_model_bytes",
        "workspace_bytes",
        "kv_cache_bytes",
        "runtime_total_bytes",
        "prefill_tps",
        "decode_tps",
        "total_tps",
    }
    for key in list(out):
        if key in numeric_keys:
            try:
                if key.endswith("_count") or key.endswith("_bytes") or key in {
                    "native_continuous_batching",
                    "batched_decode",
                    "batched_decode_calls",
                    "batched_decode_tokens",
                    "batched_lm_head",
                    "batched_lm_head_calls",
                    "batched_lm_head_tokens",
                    "batch_profile_calls",
                    "batch_profile_tokens",
                    "batch_profile_emit_version",
                    "scheduler_outer_threads",
                    "scheduler_inner_threads",
                    "scheduler_lm_head_threads",
                }:
                    out[key] = int(float(out[key]))
                else:
                    out[key] = float(out[key])
            except (TypeError, ValueError):
                pass
    return out


def run_microgemm_text(
    text_bin: Path,
    mgm_path: Path,
    tokenizer_json: Path,
    prompt: str,
    *,
    prompt_ids: list[int] | None = None,
    max_new_tokens: int,
    max_seq_len: int,
    threads: int,
    seed: int,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    cmd = [
        str(text_bin),
        "generate",
        str(mgm_path),
        str(tokenizer_json),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        "0.0",
        "--top-k",
        "0",
        "--top-p",
        "1.0",
        "--seed",
        str(seed),
    ]
    if prompt_ids:
        cmd.extend(["--prompt-ids", ",".join(str(int(token_id)) for token_id in prompt_ids)])
    else:
        cmd.extend(["--prompt", prompt])
    if max_seq_len > 0:
        cmd.extend(["--max-seq-len", str(max_seq_len)])
    if ignore_eos:
        cmd.append("--ignore-eos")

    env = os.environ.copy()
    if threads > 0:
        env["OMP_NUM_THREADS"] = str(threads)
        env.setdefault("OMP_PROC_BIND", "false")

    start = time.perf_counter()
    result = run_capture(cmd, env=env)
    wall_ms = (time.perf_counter() - start) * 1000.0
    row = {
        "ok": result.returncode == 0,
        "wall_ms": wall_ms,
        "returncode": result.returncode,
    }
    if result.returncode != 0:
        row["stderr"] = result.stderr[-2000:]
        row["stdout"] = result.stdout[-1000:]
        return row
    row.update(parse_microgemm_stdout(result.stdout))
    return row


def run_microgemm_batch_text(
    text_bin: Path,
    mgm_path: Path,
    tokenizer_json: Path,
    prompts: list[str],
    *,
    prompt_id_batches: list[list[int]] | None = None,
    max_new_tokens: int,
    max_seq_len: int,
    threads: int,
    seed: int,
    ignore_eos: bool = False,
    batch_outer_threads: int | None = None,
    batch_inner_threads: int | None = None,
) -> dict[str, Any]:
    prompt_files: list[Path] = []
    try:
        cmd = [
            str(text_bin),
            "batch-generate",
            str(mgm_path),
            str(tokenizer_json),
            "--max-new-tokens",
            str(max_new_tokens),
            "--temperature",
            "0.0",
            "--top-k",
            "0",
            "--top-p",
            "1.0",
            "--seed",
            str(seed),
        ]
        if prompt_id_batches:
            for prompt_ids in prompt_id_batches:
                cmd.extend(["--prompt-ids", ",".join(str(int(token_id)) for token_id in prompt_ids)])
        else:
            inline_prompt_bytes = sum(len(prompt.encode("utf-8", errors="ignore")) for prompt in prompts)
            if inline_prompt_bytes <= 128 * 1024:
                for prompt in prompts:
                    cmd.extend(["--prompt", prompt])
            else:
                for prompt in prompts:
                    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as fh:
                        fh.write(prompt)
                        prompt_files.append(Path(fh.name))
                for prompt_file in prompt_files:
                    cmd.extend(["--prompt-file", str(prompt_file)])
        if max_seq_len > 0:
            cmd.extend(["--max-seq-len", str(max_seq_len)])
        if ignore_eos:
            cmd.append("--ignore-eos")

        env = os.environ.copy()
        if threads > 0:
            if len(prompts) > 1:
                outer_threads = batch_outer_threads if batch_outer_threads and batch_outer_threads > 0 else min(len(prompts), threads)
                outer_threads = max(1, min(len(prompts), outer_threads))
                inner_threads = batch_inner_threads if batch_inner_threads and batch_inner_threads > 0 else max(1, threads // outer_threads)
                inner_threads = max(1, inner_threads)
                env["MICROGEMM_BATCH_TOTAL_THREADS"] = str(threads)
                env["MICROGEMM_BATCH_OUTER_THREADS"] = str(outer_threads)
                env["MICROGEMM_BATCH_INNER_THREADS"] = str(inner_threads)
                env["MICROGEMM_BATCH_LM_HEAD_THREADS"] = str(threads)
                env["OMP_NUM_THREADS"] = str(threads)
                env["OMP_MAX_ACTIVE_LEVELS"] = "1"
                env["OMP_NESTED"] = "false"
            else:
                env["OMP_NUM_THREADS"] = str(threads)
            env.setdefault("OMP_PROC_BIND", "false")

        start = time.perf_counter()
        result = run_capture(cmd, env=env)
        wall_ms = (time.perf_counter() - start) * 1000.0
        row = {
            "ok": result.returncode == 0,
            "wall_ms": wall_ms,
            "returncode": result.returncode,
        }
        if result.returncode != 0:
            row["stderr"] = result.stderr[-2000:]
            row["stdout"] = result.stdout[-1000:]
            return row
        parsed = parse_microgemm_stdout(result.stdout)
        row.update(parsed)
        row["raw_profile_marker"] = 1 if TEXT_PROFILE_MARKER in result.stdout else 0
        if TEXT_PROFILE_MARKER not in parsed:
            row["stdout_tail"] = result.stdout[-2000:]
        return row
    finally:
        for prompt_file in prompt_files:
            try:
                prompt_file.unlink()
            except OSError:
                pass


def parse_continuous_split_mode(mode: str) -> tuple[str, int | None, int | None]:
    prefix = "continuous"
    if mode == prefix or not mode.startswith(prefix):
        return mode, None, None
    rest = mode[len(prefix):]
    if "x" not in rest:
        return mode, None, None
    left, right = rest.split("x", 1)
    if not left.isdigit() or not right.isdigit():
        return mode, None, None
    outer = int(left)
    inner = int(right)
    if outer <= 0 or inner <= 0:
        return mode, None, None
    return prefix, outer, inner


def aggregate_single(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("ok")]
    keys = [
        "wall_ms",
        "prefill_ms",
        "decode_ms",
        "total_ms",
        "prefill_tps",
        "decode_tps",
        "total_tps",
    ]
    out: dict[str, Any] = {"ok": len(ok), "runs": len(rows)}
    for key in keys:
        out[key] = stats([float(r.get(key, 0.0) or 0.0) for r in ok])
    if ok:
        out["prompt_token_count"] = ok[0].get("prompt_token_count", 0)
        out["generated_token_count"] = ok[0].get("generated_token_count", 0)
        out["runtime_total_bytes"] = ok[0].get("runtime_total_bytes", 0)
    else:
        failure = first_failure_info(rows)
        out["first_failure_returncode"] = failure.get("returncode", "")
        out["first_failure_stderr"] = compact_error_text(failure.get("stderr", ""))
        out["first_failure_stdout"] = compact_error_text(failure.get("stdout", ""))
    return out


def run_context_sweep(
    args: argparse.Namespace,
    text_bin: Path,
    mgm_path: Path,
    tokenizer_json: Path,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    safe_ascii = use_safe_ascii_prompts(args.model_repo or args.model_dir)
    direct_prompt_ids = safe_ascii
    for target_prompt_tokens in parse_csv_ints(args.prompt_tokens):
        prompt_ids = make_prompt_ids(tokenizer, target_prompt_tokens) if direct_prompt_ids else []
        prompt, prompt_actual = make_prompt(tokenizer, target_prompt_tokens, safe_ascii=safe_ascii)
        if prompt_ids:
            prompt_actual = len(prompt_ids)
        max_seq_len = planned_max_seq_len(
            args.max_seq_len,
            target_prompt_tokens,
            prompt_actual,
            args.max_new_tokens,
        )
        if args.warmup > 0:
            for warm_idx in range(args.warmup):
                run_microgemm_text(
                    text_bin,
                    mgm_path,
                    tokenizer_json,
                    prompt,
                    prompt_ids=prompt_ids or None,
                    max_new_tokens=min(args.max_new_tokens, 8),
                    max_seq_len=max_seq_len,
                    threads=args.threads,
                    seed=args.seed + warm_idx,
                    ignore_eos=bool(args.ignore_eos),
                )
        rows = []
        for run_idx in range(args.runs):
            rows.append(
                run_microgemm_text(
                    text_bin,
                    mgm_path,
                    tokenizer_json,
                    prompt,
                    prompt_ids=prompt_ids or None,
                    max_new_tokens=args.max_new_tokens,
                    max_seq_len=max_seq_len,
                    threads=args.threads,
                    seed=args.seed + run_idx,
                    ignore_eos=bool(args.ignore_eos),
                )
            )
        summary = aggregate_single(rows)
        summary.update(
            {
                "kind": "single_context",
                "target_prompt_tokens": target_prompt_tokens,
                "prompt_tokens_from_hf_tokenizer": prompt_actual,
                "max_seq_len": max_seq_len,
                "threads": args.threads,
            }
        )
        results.append({"summary": summary, "runs": rows})
        decode_med = summary.get("decode_tps", {}).get("median", 0.0)
        prefill_med = summary.get("prefill_tps", {}).get("median", 0.0)
        print(
            f"single prompt={target_prompt_tokens} ok={summary['ok']}/{summary['runs']} "
            f"prefill_med={prefill_med:.2f} tok/s decode_med={decode_med:.2f} tok/s",
            flush=True,
        )
        if int(summary.get("ok", 0) or 0) == 0:
            failure_note = summary.get("first_failure_stderr") or summary.get("first_failure_stdout")
            if failure_note:
                print(
                    f"  first failure: rc={summary.get('first_failure_returncode', '')} {failure_note}",
                    flush=True,
                )
        if target_prompt_tokens == RUN_CANARY_PROMPT_TOKENS and bool(getattr(args, "strict_canary_gate", False)):
            canary_status = summarize_run_canary(
                results,
                {},
                getattr(args, "canary_profile", None),
            ).get("status")
            if (
                (canary_status == "slow" and not args.allow_slow_canary)
                or (canary_status == "weak" and not args.allow_weak_canary and not args.allow_slow_canary)
            ):
                print("Strict canary gate: canary prompt is not good; skipping remaining single prompts.", flush=True)
                break
    return results


def summarize_batch_rows(rows: list[dict[str, Any]], batch_size: int, mode: str) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("ok")]
    effective_modes = sorted({str(r.get("effective_mode", mode)) for r in rows})
    profile_ms_keys = [
        "batch_profile_total_ms",
        "batch_profile_alloc_ms",
        "batch_profile_embed_ms",
        "batch_profile_input_norm_ms",
        "batch_profile_qkv_ms",
        "batch_profile_rope_kv_ms",
        "batch_profile_attention_ms",
        "batch_profile_o_proj_ms",
        "batch_profile_post_norm_ms",
        "batch_profile_gate_up_ms",
        "batch_profile_gate_up_quant_ms",
        "batch_profile_gate_up_dot_ms",
        "batch_profile_activation_ms",
        "batch_profile_down_proj_ms",
        "batch_profile_down_proj_quant_ms",
        "batch_profile_down_proj_dot_ms",
        "batch_profile_final_norm_ms",
        "batch_profile_lm_head_ms",
        "batch_profile_groupwise_gemv_tile_calls",
        "batch_profile_groupwise_i8_row_pair_calls",
        "batch_profile_groupwise_i4_row_pair_calls",
        "batch_profile_groupwise_lm_head_argmax_calls",
        "batch_profile_lm_head_stack_best_calls",
        "batch_profile_groupwise_gate_up_fused_calls",
        "batch_profile_groupwise_i8_gate_safe_combined_calls",
        "batch_profile_groupwise_i8_gate_safe_combined_tile8_calls",
        "batch_profile_groupwise_i8_gate_tile8_calls",
        "batch_profile_groupwise_i8_gate_biased_calls",
        "batch_profile_groupwise_i8_gate_pair_calls",
        "batch_profile_groupwise_i8_gate_pair_unroll64_calls",
        "batch_profile_groupwise_i8_gate_pair_unroll128_calls",
        "batch_profile_groupwise_i8_gate_pair8_split_calls",
        "batch_profile_groupwise_i8_gate_prefetch_calls",
        "batch_profile_groupwise_lm_head_row_pair_calls",
        "batch_profile_copy_ms",
        "batch_profile_cleanup_ms",
    ]
    wall_tps = [float(r.get("wall_output_tps", 0.0) or 0.0) for r in ok_rows]
    steady_tps = [float(r.get("steady_output_tps", 0.0) or 0.0) for r in ok_rows]
    runtime_tps = [float(r.get("runtime_output_tps", 0.0) or 0.0) for r in ok_rows]
    decode_only_tps = [float(r.get("decode_only_output_tps", 0.0) or 0.0) for r in ok_rows]
    wall_total_tps = [float(r.get("wall_total_tps", 0.0) or 0.0) for r in ok_rows]
    steady_total_tps = [float(r.get("steady_total_tps", 0.0) or 0.0) for r in ok_rows]
    runtime_total_tps = [float(r.get("runtime_total_tps", 0.0) or 0.0) for r in ok_rows]
    wall_ms = [float(r.get("batch_wall_ms", 0.0) or 0.0) for r in ok_rows]
    steady_ms = [float(r.get("steady_ms", 0.0) or 0.0) for r in ok_rows]
    prefill_ms = [float(r.get("prefill_ms", 0.0) or 0.0) for r in ok_rows]
    decode_ms = [float(r.get("decode_ms", 0.0) or 0.0) for r in ok_rows]
    prefill_tps = [float(r.get("prefill_tps", 0.0) or 0.0) for r in ok_rows]
    harness_overhead_ms = [float(r.get("harness_overhead_ms", 0.0) or 0.0) for r in ok_rows]
    setup_ms = [float(r.get("setup_ms", 0.0) or 0.0) for r in ok_rows if "setup_ms" in r]
    batched_decode_calls = [int(r.get("batched_decode_calls", 0) or 0) for r in ok_rows]
    batched_decode_tokens = [int(r.get("batched_decode_tokens", 0) or 0) for r in ok_rows]
    batched_lm_head_calls = [int(r.get("batched_lm_head_calls", 0) or 0) for r in ok_rows]
    batched_lm_head_tokens = [int(r.get("batched_lm_head_tokens", 0) or 0) for r in ok_rows]
    scheduler_outer_threads = [int(r.get("scheduler_outer_threads", 0) or 0) for r in ok_rows]
    scheduler_inner_threads = [int(r.get("scheduler_inner_threads", 0) or 0) for r in ok_rows]
    scheduler_lm_head_threads = [int(r.get("scheduler_lm_head_threads", 0) or 0) for r in ok_rows]
    raw_profile_marker = [int(r.get("raw_profile_marker", 0) or 0) for r in ok_rows]
    prompt_tokens = [int(r.get("prompt_tokens", 0) or 0) for r in ok_rows]
    generated = [int(r.get("generated_tokens", 0) or 0) for r in ok_rows]
    total_tokens = [int(r.get("total_tokens", 0) or 0) for r in ok_rows]
    summary = {
        "kind": "batch",
        "mode": mode,
        "effective_mode": effective_modes[0] if len(effective_modes) == 1 else ",".join(effective_modes),
        "batch_size": batch_size,
        "ok": len(ok_rows),
        "runs": len(rows),
        "prompt_tokens_median": statistics.median(prompt_tokens) if prompt_tokens else 0,
        "generated_tokens_median": statistics.median(generated) if generated else 0,
        "total_tokens_median": statistics.median(total_tokens) if total_tokens else 0,
        "setup_ms": stats(setup_ms),
        "scheduler_outer_threads": stats(scheduler_outer_threads),
        "scheduler_inner_threads": stats(scheduler_inner_threads),
        "scheduler_lm_head_threads": stats(scheduler_lm_head_threads),
        "raw_profile_marker": stats(raw_profile_marker),
        "batched_decode_calls": stats(batched_decode_calls),
        "batched_decode_tokens": stats(batched_decode_tokens),
        "batched_lm_head_calls": stats(batched_lm_head_calls),
        "batched_lm_head_tokens": stats(batched_lm_head_tokens),
        "batch_wall_ms": stats(wall_ms),
        "steady_ms": stats(steady_ms),
        "prefill_ms": stats(prefill_ms),
        "decode_ms": stats(decode_ms),
        "harness_overhead_ms": stats(harness_overhead_ms),
        "prefill_tps": stats(prefill_tps),
        "wall_output_tps": stats(wall_tps),
        "steady_output_tps": stats(steady_tps),
        "runtime_output_tps": stats(runtime_tps),
        "decode_only_output_tps": stats(decode_only_tps),
        "wall_total_tps": stats(wall_total_tps),
        "steady_total_tps": stats(steady_total_tps),
        "runtime_total_tps": stats(runtime_total_tps),
    }
    if not ok_rows:
        failure = first_failure_info(rows)
        summary["first_failure_returncode"] = failure.get("returncode", "")
        summary["first_failure_stderr"] = compact_error_text(failure.get("stderr", ""))
        summary["first_failure_stdout"] = compact_error_text(failure.get("stdout", ""))
    for key in profile_ms_keys:
        values = [float(r.get(key, 0.0) or 0.0) for r in ok_rows if key in r]
        if values:
            summary[key] = stats(values)
    for key in BATCH_COMMAND_TIMING_KEYS:
        values = [float(r.get(key, 0.0) or 0.0) for r in ok_rows if key in r]
        if values:
            summary[key] = stats(values)
    return summary


def batch_canary_thresholds(batch_size: int, canary_profile: dict[str, Any] | None = None) -> tuple[float, float]:
    profile = canary_profile or {}
    good_by_size = profile.get("batch_good_runtime_tps_by_size", BATCH_CANARY_GOOD_RUNTIME_TPS_BY_SIZE)
    weak_by_size = profile.get("batch_weak_runtime_tps_by_size", BATCH_CANARY_WEAK_RUNTIME_TPS_BY_SIZE)
    good = good_by_size.get(batch_size, BATCH_CANARY_GOOD_RUNTIME_TPS)
    weak = weak_by_size.get(batch_size, BATCH_CANARY_WEAK_RUNTIME_TPS)
    return float(good), float(weak)


def batch_profile_phase_medians(summary: dict[str, Any]) -> dict[str, float]:
    return {
        key.replace("batch_profile_", "").replace("_ms", ""): float(
            summary.get(key, {}).get("median", 0.0) or 0.0
        )
        for key in BATCH_PROFILE_PHASE_KEYS
        if key in summary
    }


def format_batch_profile_note(summary: dict[str, Any]) -> str:
    phase_medians = batch_profile_phase_medians(summary)
    if not phase_medians:
        raw_marker = int(summary.get("raw_profile_marker", {}).get("median", 0) or 0)
        return f"profile=missing raw_profile={raw_marker}"

    top_phases = sorted(phase_medians.items(), key=lambda item: item[1], reverse=True)
    top_note = ",".join(f"{phase}:{ms:.0f}ms" for phase, ms in top_phases[:3] if ms > 0.0)
    parts: list[str] = []
    if top_note:
        parts.append(f"top3={top_note}")
    gate_quant = phase_medians.get("gate_up_quant", 0.0)
    gate_dot = phase_medians.get("gate_up_dot", 0.0)
    if gate_quant or gate_dot:
        parts.append(f"gate_q={gate_quant:.0f}ms gate_dot={gate_dot:.0f}ms")
    down_quant = phase_medians.get("down_proj_quant", 0.0)
    down_dot = phase_medians.get("down_proj_dot", 0.0)
    if down_quant or down_dot:
        parts.append(f"down_q={down_quant:.0f}ms down_dot={down_dot:.0f}ms")
    gw_tile = int(summary.get("batch_profile_groupwise_gemv_tile_calls", {}).get("median", 0) or 0)
    gw_i8_rowpair = int(summary.get("batch_profile_groupwise_i8_row_pair_calls", {}).get("median", 0) or 0)
    gw_i4_rowpair = int(summary.get("batch_profile_groupwise_i4_row_pair_calls", {}).get("median", 0) or 0)
    gw_lm_argmax = int(summary.get("batch_profile_groupwise_lm_head_argmax_calls", {}).get("median", 0) or 0)
    lm_stack_best = int(summary.get("batch_profile_lm_head_stack_best_calls", {}).get("median", 0) or 0)
    gw_gate_fused = int(summary.get("batch_profile_groupwise_gate_up_fused_calls", {}).get("median", 0) or 0)
    gw_i8_gate_combo = int(summary.get("batch_profile_groupwise_i8_gate_safe_combined_calls", {}).get("median", 0) or 0)
    gw_i8_gate_combo8 = int(summary.get("batch_profile_groupwise_i8_gate_safe_combined_tile8_calls", {}).get("median", 0) or 0)
    gw_i8_gate8 = int(summary.get("batch_profile_groupwise_i8_gate_tile8_calls", {}).get("median", 0) or 0)
    gw_i8_gate_bias = int(summary.get("batch_profile_groupwise_i8_gate_biased_calls", {}).get("median", 0) or 0)
    gw_i8_gate_pair = int(summary.get("batch_profile_groupwise_i8_gate_pair_calls", {}).get("median", 0) or 0)
    gw_i8_gate_pair_unroll64 = int(summary.get("batch_profile_groupwise_i8_gate_pair_unroll64_calls", {}).get("median", 0) or 0)
    gw_i8_gate_pair_unroll128 = int(summary.get("batch_profile_groupwise_i8_gate_pair_unroll128_calls", {}).get("median", 0) or 0)
    gw_i8_gate_pair8_split = int(summary.get("batch_profile_groupwise_i8_gate_pair8_split_calls", {}).get("median", 0) or 0)
    gw_i8_gate_prefetch = int(summary.get("batch_profile_groupwise_i8_gate_prefetch_calls", {}).get("median", 0) or 0)
    gw_lm_rowpair = int(summary.get("batch_profile_groupwise_lm_head_row_pair_calls", {}).get("median", 0) or 0)
    if gw_tile or gw_i8_rowpair or gw_i4_rowpair or gw_lm_argmax or lm_stack_best or gw_gate_fused or gw_i8_gate_combo or gw_i8_gate_combo8 or gw_i8_gate8 or gw_i8_gate_bias or gw_i8_gate_pair or gw_i8_gate_pair_unroll64 or gw_i8_gate_pair_unroll128 or gw_i8_gate_pair8_split or gw_i8_gate_prefetch or gw_lm_rowpair:
        parts.append(
            f"routes=gw_tile:{gw_tile} gw_i8_rowpair:{gw_i8_rowpair} gw_i4_rowpair:{gw_i4_rowpair} "
            f"gw_lm_argmax:{gw_lm_argmax} lm_stack:{lm_stack_best} gw_gate_fused:{gw_gate_fused} "
            f"gw_i8_gate_combo:{gw_i8_gate_combo} gw_i8_gate_combo8:{gw_i8_gate_combo8} gw_i8_gate8:{gw_i8_gate8} "
            f"gw_i8_gate_bias:{gw_i8_gate_bias} gw_i8_gate_pair:{gw_i8_gate_pair} "
            f"gw_i8_gate_p4u64:{gw_i8_gate_pair_unroll64} gw_i8_gate_p4u128:{gw_i8_gate_pair_unroll128} "
            f"gw_i8_gate_p8split:{gw_i8_gate_pair8_split} gw_i8_gate_prefetch:{gw_i8_gate_prefetch} "
            f"gw_lm_rowpair:{gw_lm_rowpair}"
        )
    return " ".join(parts)


def run_batch_once(
    args: argparse.Namespace,
    text_bin: Path,
    mgm_path: Path,
    tokenizer_json: Path,
    prompts: list[str],
    *,
    prompt_id_batches: list[list[int]] | None = None,
    max_seq_len: int,
    mode: str,
    threads_per_worker: int,
    seed_base: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    effective_mode = mode
    batch_outer_threads: int | None = None
    batch_inner_threads: int | None = None
    effective_mode, batch_outer_threads, batch_inner_threads = parse_continuous_split_mode(effective_mode)
    if mode == "adaptive":
        if len(prompts) == 1:
            effective_mode = "continuous"
        elif len(prompts) <= 4:
            effective_mode = "concurrent"
        else:
            effective_mode = "continuous"

    continuous_single_fast_path = False
    if effective_mode == "continuous":
        if len(prompts) == 1:
            continuous_single_fast_path = True
            outputs = [
                run_microgemm_text(
                    text_bin,
                    mgm_path,
                    tokenizer_json,
                    prompts[0],
                    prompt_ids=prompt_id_batches[0] if prompt_id_batches else None,
                    max_new_tokens=args.max_new_tokens,
                    max_seq_len=max_seq_len,
                    threads=threads_per_worker,
                    seed=seed_base,
                    ignore_eos=bool(args.ignore_eos),
                )
            ]
        else:
            outputs = [
                run_microgemm_batch_text(
                    text_bin,
                    mgm_path,
                    tokenizer_json,
                    prompts,
                    prompt_id_batches=prompt_id_batches,
                    max_new_tokens=args.max_new_tokens,
                    max_seq_len=max_seq_len,
                    threads=threads_per_worker,
                    seed=seed_base,
                    ignore_eos=bool(args.ignore_eos),
                    batch_outer_threads=batch_outer_threads,
                    batch_inner_threads=batch_inner_threads,
                )
            ]
    elif effective_mode == "serial":
        outputs = [
            run_microgemm_text(
                text_bin,
                mgm_path,
                tokenizer_json,
                prompt,
                prompt_ids=prompt_id_batches[idx] if prompt_id_batches else None,
                max_new_tokens=args.max_new_tokens,
                max_seq_len=max_seq_len,
                threads=threads_per_worker,
                seed=seed_base + idx,
                ignore_eos=bool(args.ignore_eos),
            )
            for idx, prompt in enumerate(prompts)
        ]
    elif effective_mode == "concurrent":
        outputs = []
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [
                pool.submit(
                    run_microgemm_text,
                    text_bin,
                    mgm_path,
                    tokenizer_json,
                    prompt,
                    prompt_ids=prompt_id_batches[idx] if prompt_id_batches else None,
                    max_new_tokens=args.max_new_tokens,
                    max_seq_len=max_seq_len,
                    threads=threads_per_worker,
                    seed=seed_base + idx,
                    ignore_eos=bool(args.ignore_eos),
                )
                for idx, prompt in enumerate(prompts)
            ]
            for fut in as_completed(futures):
                outputs.append(fut.result())
    else:
        raise ValueError(f"unsupported batch mode: {mode}")

    batch_wall_ms = (time.perf_counter() - start) * 1000.0
    ok = all(o.get("ok") for o in outputs)
    prompt_tokens = sum(int(o.get("prompt_token_count", 0) or 0) for o in outputs if o.get("ok"))
    generated_tokens = sum(int(o.get("generated_token_count", 0) or 0) for o in outputs if o.get("ok"))
    total_tokens = prompt_tokens + generated_tokens
    if effective_mode in {"serial"}:
        runtime_ms = sum(float(o.get("total_ms", 0.0) or 0.0) for o in outputs if o.get("ok"))
        decode_ms = sum(float(o.get("decode_ms", 0.0) or 0.0) for o in outputs if o.get("ok"))
        setup_ms = 0.0
    elif effective_mode == "continuous":
        runtime_ms = float(outputs[0].get("total_ms", 0.0) or 0.0) if outputs else 0.0
        decode_ms = float(outputs[0].get("decode_ms", 0.0) or 0.0) if outputs else 0.0
        setup_ms = float(outputs[0].get("setup_ms", 0.0) or 0.0) if outputs else 0.0
        prefill_ms = float(outputs[0].get("prefill_ms", 0.0) or 0.0) if outputs else 0.0
    else:
        runtime_ms = max([float(o.get("total_ms", 0.0) or 0.0) for o in outputs if o.get("ok")] or [0.0])
        decode_ms = max([float(o.get("decode_ms", 0.0) or 0.0) for o in outputs if o.get("ok")] or [0.0])
        setup_ms = 0.0
        prefill_ms = max([float(o.get("prefill_ms", 0.0) or 0.0) for o in outputs if o.get("ok")] or [0.0])
    if effective_mode in {"serial"}:
        prefill_ms = sum(float(o.get("prefill_ms", 0.0) or 0.0) for o in outputs if o.get("ok"))
    batched_decode_calls = sum(int(o.get("batched_decode_calls", 0) or 0) for o in outputs)
    batched_decode_tokens = sum(int(o.get("batched_decode_tokens", 0) or 0) for o in outputs)
    batched_lm_head_calls = sum(int(o.get("batched_lm_head_calls", 0) or 0) for o in outputs)
    batched_lm_head_tokens = sum(int(o.get("batched_lm_head_tokens", 0) or 0) for o in outputs)
    scheduler_outer_threads = int(outputs[0].get("scheduler_outer_threads", 0) or 0) if outputs else 0
    scheduler_inner_threads = int(outputs[0].get("scheduler_inner_threads", 0) or 0) if outputs else 0
    scheduler_lm_head_threads = int(outputs[0].get("scheduler_lm_head_threads", 0) or 0) if outputs else 0
    steady_ms = runtime_ms + setup_ms if effective_mode == "continuous" else runtime_ms
    harness_overhead_ms = max(0.0, batch_wall_ms - steady_ms)
    command_total_ms = (
        float(outputs[0].get("command_total_ms", 0.0) or 0.0)
        if effective_mode == "continuous" and outputs
        else 0.0
    )
    process_overhead_ms = max(0.0, batch_wall_ms - command_total_ms) if command_total_ms > 0.0 else 0.0
    row = {
        "ok": ok,
        "mode": mode,
        "effective_mode": effective_mode,
        "batch_size": len(prompts),
        "native_continuous_batching": effective_mode == "continuous" and not continuous_single_fast_path,
        "continuous_single_fast_path": continuous_single_fast_path,
        "threads_per_worker": threads_per_worker,
        "setup_ms": setup_ms,
        "scheduler_outer_threads": scheduler_outer_threads,
        "scheduler_inner_threads": scheduler_inner_threads,
        "scheduler_lm_head_threads": scheduler_lm_head_threads,
        "batched_decode_calls": batched_decode_calls,
        "batched_decode_tokens": batched_decode_tokens,
        "batched_lm_head_calls": batched_lm_head_calls,
        "batched_lm_head_tokens": batched_lm_head_tokens,
        "batch_wall_ms": batch_wall_ms,
        "steady_ms": steady_ms,
        "harness_overhead_ms": harness_overhead_ms,
        "process_overhead_ms": process_overhead_ms,
        "runtime_ms": runtime_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "prefill_tps": prompt_tokens / (prefill_ms / 1000.0) if prefill_ms > 0 else 0.0,
        "wall_output_tps": generated_tokens / (batch_wall_ms / 1000.0) if batch_wall_ms > 0 else 0.0,
        "steady_output_tps": generated_tokens / (steady_ms / 1000.0) if steady_ms > 0 else 0.0,
        "runtime_output_tps": generated_tokens / (runtime_ms / 1000.0) if runtime_ms > 0 else 0.0,
        "decode_only_output_tps": generated_tokens / (decode_ms / 1000.0) if decode_ms > 0 else 0.0,
        "wall_total_tps": total_tokens / (batch_wall_ms / 1000.0) if batch_wall_ms > 0 else 0.0,
        "steady_total_tps": total_tokens / (steady_ms / 1000.0) if steady_ms > 0 else 0.0,
        "runtime_total_tps": total_tokens / (runtime_ms / 1000.0) if runtime_ms > 0 else 0.0,
        "workers": outputs,
    }
    if effective_mode == "continuous" and outputs:
        for key in [
            "raw_profile_marker",
            "batch_profile_emit_version",
            "batch_profile_calls",
            "batch_profile_tokens",
            "batch_profile_total_ms",
            "batch_profile_alloc_ms",
            "batch_profile_embed_ms",
            "batch_profile_input_norm_ms",
            "batch_profile_qkv_ms",
            "batch_profile_rope_kv_ms",
            "batch_profile_attention_ms",
            "batch_profile_o_proj_ms",
            "batch_profile_post_norm_ms",
            "batch_profile_gate_up_ms",
            "batch_profile_gate_up_quant_ms",
            "batch_profile_gate_up_dot_ms",
            "batch_profile_activation_ms",
            "batch_profile_down_proj_ms",
            "batch_profile_down_proj_quant_ms",
            "batch_profile_down_proj_dot_ms",
            "batch_profile_final_norm_ms",
            "batch_profile_lm_head_ms",
            "batch_profile_groupwise_gemv_tile_calls",
            "batch_profile_groupwise_i8_row_pair_calls",
            "batch_profile_groupwise_i4_row_pair_calls",
            "batch_profile_groupwise_lm_head_argmax_calls",
            "batch_profile_lm_head_stack_best_calls",
            "batch_profile_groupwise_gate_up_fused_calls",
            "batch_profile_groupwise_i8_gate_safe_combined_calls",
            "batch_profile_groupwise_i8_gate_safe_combined_tile8_calls",
            "batch_profile_groupwise_i8_gate_tile8_calls",
            "batch_profile_groupwise_i8_gate_biased_calls",
            "batch_profile_groupwise_i8_gate_pair_calls",
            "batch_profile_groupwise_i8_gate_pair_unroll64_calls",
            "batch_profile_groupwise_i8_gate_pair_unroll128_calls",
            "batch_profile_groupwise_i8_gate_pair8_split_calls",
            "batch_profile_groupwise_i8_gate_prefetch_calls",
            "batch_profile_groupwise_lm_head_row_pair_calls",
            "batch_profile_copy_ms",
            "batch_profile_cleanup_ms",
        ]:
            if key in outputs[0]:
                row[key] = outputs[0][key]
        for key in BATCH_COMMAND_TIMING_KEYS:
            if key in outputs[0]:
                row[key] = outputs[0][key]
    return row


def summarize_batch_canary(
    batch_canary: dict[str, Any],
    canary_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = batch_canary.get("summary", {})
    batch_size = int(summary.get("batch_size", BATCH_CANARY_SIZE) or 0)
    good_threshold, weak_threshold = batch_canary_thresholds(batch_size, canary_profile)
    runtime_tps = float(summary.get("runtime_output_tps", {}).get("median", 0.0) or 0.0)
    if not summary.get("ok"):
        status = "missing"
        reason = f"batch{batch_size} continuous canary did not produce a usable run"
    elif runtime_tps >= good_threshold:
        status = "good"
        reason = f"batch{batch_size} continuous canary is in the comparable range"
    elif runtime_tps >= weak_threshold:
        status = "weak"
        reason = f"batch{batch_size} continuous canary is below the best memory/batch runs"
    else:
        status = "slow"
        reason = f"batch{batch_size} continuous canary is too low for kernel comparisons"

    return {
        "status": status,
        "reason": reason,
        "batch_size": batch_size,
        "mode": str(summary.get("mode", BATCH_CANARY_MODE)),
        "max_new_tokens": int(summary.get("max_new_tokens", BATCH_CANARY_MAX_NEW_TOKENS) or 0),
        "attempt": int(summary.get("canary_attempt", 0) or 0),
        "max_attempts": int(summary.get("canary_max_attempts", BATCH_CANARY_MAX_ATTEMPTS) or 0),
        "runtime_tps": runtime_tps,
        "decode_only_tps": float(summary.get("decode_only_output_tps", {}).get("median", 0.0) or 0.0),
        "wall_tps": float(summary.get("wall_output_tps", {}).get("median", 0.0) or 0.0),
        "wall_total_tps": float(summary.get("wall_total_tps", {}).get("median", 0.0) or 0.0),
        "good_threshold": good_threshold,
        "weak_threshold": weak_threshold,
        "profile": (canary_profile or {}).get("name", ""),
        "profile_note": format_batch_profile_note(summary),
    }


def print_batch_canary(canary: dict[str, Any]) -> None:
    attempt = int(canary.get("attempt", 0) or 0)
    max_attempts = int(canary.get("max_attempts", 0) or 0)
    if attempt > 0 and max_attempts > 0:
        print(f"Batch canary attempt {attempt}/{max_attempts}")
    else:
        print("Batch canary")
    print(
        "  batch decode: "
        f"mode={canary.get('mode', BATCH_CANARY_MODE)} "
        f"batch={int(canary.get('batch_size', 0) or 0)} "
        f"max_new={int(canary.get('max_new_tokens', 0) or 0)} "
        f"runtime={float(canary.get('runtime_tps', 0.0) or 0.0):.2f} tok/s "
        f"decode_only={float(canary.get('decode_only_tps', 0.0) or 0.0):.2f} tok/s "
        f"wall={float(canary.get('wall_tps', 0.0) or 0.0):.2f} tok/s "
        f"total_wall={float(canary.get('wall_total_tps', 0.0) or 0.0):.2f} tok/s"
    )
    print(
        "  verdict:      "
        f"{canary.get('status', 'missing')} - {canary.get('reason', '')}"
    )
    print(
        "  thresholds:   "
        f"good>={float(canary.get('good_threshold', 0.0) or 0.0):.0f} "
        f"weak>={float(canary.get('weak_threshold', 0.0) or 0.0):.0f}"
    )
    if canary.get("profile"):
        print(f"  profile:      {canary.get('profile')}")
    profile_note = str(canary.get("profile_note", "") or "")
    if profile_note:
        print(f"  profile:      {profile_note}")


def run_batch_canary(
    args: argparse.Namespace,
    text_bin: Path,
    mgm_path: Path,
    tokenizer_json: Path,
    tokenizer: Any,
    *,
    batch_size: int = BATCH_CANARY_SIZE,
    attempt: int = 1,
) -> dict[str, Any]:
    batch_prompt_tokens = int(args.batch_prompt_tokens)
    safe_ascii = use_safe_ascii_prompts(args.model_repo or args.model_dir)
    prompt_id_batches = make_batch_prompt_ids(tokenizer, batch_prompt_tokens, batch_size) if safe_ascii else []
    prompts = make_batch_prompts(
        tokenizer,
        batch_prompt_tokens,
        batch_size,
        safe_ascii=safe_ascii,
    )
    canary_args = argparse.Namespace(**vars(args))
    canary_args.max_new_tokens = min(int(args.max_new_tokens), BATCH_CANARY_MAX_NEW_TOKENS)
    actual_batch_prompt_tokens = max(
        [len(prompt_ids) for prompt_ids in prompt_id_batches]
        or [count_prompt_tokens(tokenizer, prompt) for prompt in prompts]
        or [0]
    )
    max_seq_len = planned_max_seq_len(
        args.max_seq_len,
        batch_prompt_tokens,
        actual_batch_prompt_tokens,
        canary_args.max_new_tokens,
    )
    threads_per_worker = int(args.threads or (os.cpu_count() or 1))

    if args.warmup > 0:
        warmup_args = argparse.Namespace(**vars(canary_args))
        warmup_args.max_new_tokens = min(int(canary_args.max_new_tokens), 8)
        run_batch_once(
            warmup_args,
            text_bin,
            mgm_path,
            tokenizer_json,
            prompts,
            prompt_id_batches=prompt_id_batches or None,
            max_seq_len=max_seq_len,
            mode=BATCH_CANARY_MODE,
            threads_per_worker=threads_per_worker,
            seed_base=args.seed + 80000 + 10000 * attempt,
        )

    rows = []
    run_count = min(max(int(args.runs), 1), BATCH_CANARY_RUNS)
    for run_idx in range(run_count):
        rows.append(
            run_batch_once(
                canary_args,
                text_bin,
                mgm_path,
                tokenizer_json,
                prompts,
                prompt_id_batches=prompt_id_batches or None,
                max_seq_len=max_seq_len,
                mode=BATCH_CANARY_MODE,
                threads_per_worker=threads_per_worker,
                seed_base=args.seed + 90000 + 10000 * attempt + 1000 * run_idx,
            )
        )

    summary = summarize_batch_rows(rows, batch_size, BATCH_CANARY_MODE)
    canary_profile = getattr(args, "canary_profile", None)
    good_threshold, weak_threshold = batch_canary_thresholds(batch_size, canary_profile)
    summary.update(
        {
            "kind": "batch_canary",
            "target_prompt_tokens": batch_prompt_tokens,
            "prompt_tokens_from_hf_tokenizer": actual_batch_prompt_tokens,
            "max_seq_len": max_seq_len,
            "max_new_tokens": canary_args.max_new_tokens,
            "threads_per_worker": threads_per_worker,
            "canary_attempt": attempt,
            "canary_max_attempts": BATCH_CANARY_MAX_ATTEMPTS,
        }
    )
    add_batch_total_tps(summary)
    result = {"summary": summary, "runs": rows}
    canary = summarize_batch_canary(result, canary_profile)
    summary["canary_status"] = canary["status"]
    summary["canary_reason"] = canary["reason"]
    summary["canary_good_threshold"] = good_threshold
    summary["canary_weak_threshold"] = weak_threshold
    return result


def batch_canary_status_allowed(status: Any, args: argparse.Namespace) -> bool:
    if not bool(getattr(args, "strict_canary_gate", False)):
        return True
    return (
        status == "good"
        or (status == "weak" and args.allow_weak_canary)
        or (status == "slow" and args.allow_slow_canary)
    )


def should_gate_on_canary_status(status: Any, args: argparse.Namespace) -> bool:
    if not bool(getattr(args, "strict_canary_gate", False)):
        return False
    return (
        status in {"missing", "failed"}
        or (status == "slow" and not args.allow_slow_canary)
        or (status == "weak" and not args.allow_weak_canary and not args.allow_slow_canary)
    )


def summarize_continuous_sweep_quality(
    summary: dict[str, Any],
    canary_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    batch_size = int(summary.get("batch_size", 0) or 0)
    runtime_tps = float(summary.get("runtime_output_tps", {}).get("median", 0.0) or 0.0)
    good_threshold, weak_threshold = batch_canary_thresholds(batch_size, canary_profile)
    if runtime_tps >= good_threshold:
        status = "good"
        reason = f"continuous batch{batch_size} sweep result is in the comparable range"
    elif runtime_tps >= weak_threshold:
        status = "weak"
        reason = f"continuous batch{batch_size} sweep result fell below the good range"
    else:
        status = "slow"
        reason = f"continuous batch{batch_size} sweep result is too low for kernel comparisons"
    return {
        "status": status,
        "reason": reason,
        "runtime_tps": runtime_tps,
        "good_threshold": good_threshold,
        "weak_threshold": weak_threshold,
        "profile": (canary_profile or {}).get("name", ""),
    }


def run_batch_canary_with_retries(
    args: argparse.Namespace,
    text_bin: Path,
    mgm_path: Path,
    tokenizer_json: Path,
    tokenizer: Any,
    *,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_canary: dict[str, Any] = {}
    canary_profile = getattr(args, "canary_profile", None)

    for attempt in range(1, BATCH_CANARY_MAX_ATTEMPTS + 1):
        candidate = run_batch_canary(
            args,
            text_bin,
            mgm_path,
            tokenizer_json,
            tokenizer,
            batch_size=batch_size,
            attempt=attempt,
        )
        candidate_canary = summarize_batch_canary(candidate, canary_profile)
        print_batch_canary(candidate_canary)
        attempts.append(candidate)

        candidate_status = candidate_canary.get("status")
        if batch_canary_status_allowed(candidate_status, args):
            selected = candidate
            selected_canary = candidate_canary
            break

        if attempt < BATCH_CANARY_MAX_ATTEMPTS:
            print(
                f"Retrying batch{batch_size} canary after {candidate_status} result "
                f"({BATCH_CANARY_RETRY_SLEEP_S:.0f}s pause)...",
                flush=True,
            )
            time.sleep(BATCH_CANARY_RETRY_SLEEP_S)

    if selected is None:
        selected = max(
            attempts,
            key=lambda item: float(
                item.get("summary", {}).get("runtime_output_tps", {}).get("median", 0.0) or 0.0
            ),
        )
        selected_canary = summarize_batch_canary(selected, canary_profile)

    return selected, selected_canary, attempts


def run_batch_sweep(
    args: argparse.Namespace,
    text_bin: Path,
    mgm_path: Path,
    tokenizer_json: Path,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    batch_prompt_tokens = int(args.batch_prompt_tokens)
    modes = [m.strip() for m in args.batch_modes.split(",") if m.strip()]
    safe_ascii = use_safe_ascii_prompts(args.model_repo or args.model_dir)

    for batch_size in parse_csv_ints(args.batch_sizes):
        prompt_id_batches = make_batch_prompt_ids(tokenizer, batch_prompt_tokens, batch_size) if safe_ascii else []
        prompts = make_batch_prompts(
            tokenizer,
            batch_prompt_tokens,
            batch_size,
            safe_ascii=safe_ascii,
        )
        actual_batch_prompt_tokens = max(
            [len(prompt_ids) for prompt_ids in prompt_id_batches]
            or [count_prompt_tokens(tokenizer, prompt) for prompt in prompts]
            or [0]
        )
        max_seq_len = planned_max_seq_len(
            args.max_seq_len,
            batch_prompt_tokens,
            actual_batch_prompt_tokens,
            args.max_new_tokens,
        )

        for mode in modes:
            threads_per_worker = args.threads_per_worker
            if threads_per_worker <= 0:
                if mode == "concurrent" or (mode == "adaptive" and 1 < batch_size <= 4):
                    threads_per_worker = max(1, int(args.threads or (os.cpu_count() or 1)) // batch_size)
                else:
                    threads_per_worker = int(args.threads or (os.cpu_count() or 1))

            if args.warmup > 0:
                run_batch_once(
                    args,
                    text_bin,
                    mgm_path,
                    tokenizer_json,
                    prompts,
                    prompt_id_batches=prompt_id_batches or None,
                    max_seq_len=max_seq_len,
                    mode=mode,
                    threads_per_worker=threads_per_worker,
                    seed_base=args.seed,
                )
            rows = []
            for run_idx in range(args.runs):
                rows.append(
                    run_batch_once(
                        args,
                        text_bin,
                        mgm_path,
                        tokenizer_json,
                        prompts,
                        prompt_id_batches=prompt_id_batches or None,
                        max_seq_len=max_seq_len,
                        mode=mode,
                        threads_per_worker=threads_per_worker,
                        seed_base=args.seed + 1000 * run_idx,
                    )
                )
            summary = summarize_batch_rows(rows, batch_size, mode)
            summary.update(
                {
                    "target_prompt_tokens": batch_prompt_tokens,
                    "prompt_tokens_from_hf_tokenizer": actual_batch_prompt_tokens,
                    "max_seq_len": max_seq_len,
                    "max_new_tokens": args.max_new_tokens,
                    "threads_per_worker": threads_per_worker,
                }
            )
            add_batch_total_tps(summary)
            results.append({"summary": summary, "runs": rows})
            wall_med = summary["wall_output_tps"]["median"]
            total_wall_med = summary["wall_total_tps"]["median"]
            steady_med = summary["steady_output_tps"]["median"]
            runtime_med = summary["runtime_output_tps"]["median"]
            prefill_med = summary["prefill_tps"]["median"]
            prefill_ms_med = summary["prefill_ms"]["median"]
            decode_only_med = summary["decode_only_output_tps"]["median"]
            sweep_quality: dict[str, Any] | None = None
            if mode == BATCH_CANARY_MODE and batch_size in BATCH_CANARY_SIZES:
                sweep_quality = summarize_continuous_sweep_quality(
                    summary,
                    getattr(args, "canary_profile", None),
                )
                summary["sweep_quality_status"] = sweep_quality["status"]
                summary["sweep_quality_reason"] = sweep_quality["reason"]
                summary["sweep_quality_good_threshold"] = sweep_quality["good_threshold"]
                summary["sweep_quality_weak_threshold"] = sweep_quality["weak_threshold"]
            thread_note = f"threads/worker={threads_per_worker}"
            if summary.get("effective_mode") and summary.get("effective_mode") != mode:
                thread_note += f" effective={summary['effective_mode']}"
            if summary.get("effective_mode") == "continuous" and batch_size > 1:
                overhead_ms = summary.get("harness_overhead_ms", {}).get("median", 0.0)
                steady_ms_note = summary.get("steady_ms", {}).get("median", 0.0)
                thread_note += (
                    f" prefill={prefill_med:.2f} tok/s prefill_ms={prefill_ms_med:.0f}ms"
                    f" steady={steady_med:.2f} tok/s decode_only={decode_only_med:.2f} tok/s"
                    f" overhead={overhead_ms:.0f}ms engine={steady_ms_note:.0f}ms"
                )
                command_total_ms = summary.get("command_total_ms", {}).get("median", 0.0)
                model_open_ms = summary.get("model_open_ms", {}).get("median", 0.0)
                model_load_ms = summary.get("model_load_ms", {}).get("median", 0.0)
                model_cleanup_ms = summary.get("model_cleanup_ms", {}).get("median", 0.0)
                process_overhead_ms = summary.get("process_overhead_ms", {}).get("median", 0.0)
                tokenizer_load_ms = summary.get("tokenizer_load_ms", {}).get("median", 0.0)
                prompt_encode_ms = summary.get("prompt_encode_ms", {}).get("median", 0.0)
                if command_total_ms:
                    thread_note += (
                        f" cmd={command_total_ms:.0f}ms"
                        f" tok={tokenizer_load_ms + prompt_encode_ms:.0f}ms"
                        f" model={model_open_ms + model_load_ms:.0f}ms"
                        f" cleanup={model_cleanup_ms:.0f}ms"
                        f" proc={process_overhead_ms:.0f}ms"
                    )
                outer_threads = int(summary.get("scheduler_outer_threads", {}).get("median", 0) or 0)
                inner_threads = int(summary.get("scheduler_inner_threads", {}).get("median", 0) or 0)
                lm_threads = int(summary.get("scheduler_lm_head_threads", {}).get("median", 0) or 0)
                thread_note += f" manual={outer_threads}x{inner_threads} lm={lm_threads}"
                batch_tokens = (
                    summary.get("batched_decode_tokens", {}).get("median", 0)
                    or summary.get("batched_lm_head_tokens", {}).get("median", 0)
                )
                thread_note += f" batched_decode_tokens={int(batch_tokens)}"
                profile_note = format_batch_profile_note(summary)
                if profile_note:
                    thread_note += f" {profile_note}"
            print(
                f"batch mode={mode} batch={batch_size} ok={summary['ok']}/{summary['runs']} "
                f"wall_med={wall_med:.2f} tok/s total_wall={total_wall_med:.2f} tok/s "
                f"runtime_med={runtime_med:.2f} tok/s "
                f"{thread_note}",
                flush=True,
            )
            if int(summary.get("ok", 0) or 0) == 0:
                failure_note = summary.get("first_failure_stderr") or summary.get("first_failure_stdout")
                if failure_note:
                    print(
                        f"  first failure: rc={summary.get('first_failure_returncode', '')} {failure_note}",
                        flush=True,
                    )
            if sweep_quality is not None and not batch_canary_status_allowed(sweep_quality["status"], args):
                print(
                    "Stopping batch sweep: "
                    f"continuous batch{batch_size} is {sweep_quality['status']} "
                    f"(runtime={sweep_quality['runtime_tps']:.2f} tok/s, "
                    f"good>={sweep_quality['good_threshold']:.0f}). "
                    "Disable --strict-canary-gate, pass --allow-weak-canary, or pass --allow-slow-canary to collect sweep data.",
                    flush=True,
                )
                return results
    return results


def flatten_summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        s = item["summary"]
        row: dict[str, Any] = {
            "kind": s.get("kind"),
            "mode": s.get("mode", ""),
            "effective_mode": s.get("effective_mode", ""),
            "batch_size": s.get("batch_size", 1),
            "native_continuous_batching": s.get("effective_mode") == "continuous",
            "target_prompt_tokens": s.get("target_prompt_tokens", ""),
            "max_seq_len": s.get("max_seq_len", ""),
            "max_new_tokens": s.get("max_new_tokens", ""),
            "threads": s.get("threads", ""),
            "threads_per_worker": s.get("threads_per_worker", ""),
            "ok": s.get("ok"),
            "runs": s.get("runs"),
            "canary_status": s.get("canary_status", ""),
            "canary_good_threshold": s.get("canary_good_threshold", ""),
            "canary_weak_threshold": s.get("canary_weak_threshold", ""),
            "sweep_quality_status": s.get("sweep_quality_status", ""),
            "sweep_quality_good_threshold": s.get("sweep_quality_good_threshold", ""),
            "sweep_quality_weak_threshold": s.get("sweep_quality_weak_threshold", ""),
            "first_failure_returncode": s.get("first_failure_returncode", ""),
            "first_failure_stderr": s.get("first_failure_stderr", ""),
            "first_failure_stdout": s.get("first_failure_stdout", ""),
        }
        if s.get("kind") == "single_context":
            row.update(
                {
                    "prefill_tps_median": s.get("prefill_tps", {}).get("median", 0.0),
                    "decode_tps_median": s.get("decode_tps", {}).get("median", 0.0),
                    "total_tps_median": s.get("total_tps", {}).get("median", 0.0),
                    "wall_ms_median": s.get("wall_ms", {}).get("median", 0.0),
                    "runtime_total_bytes": s.get("runtime_total_bytes", 0),
                }
            )
        else:
            row.update(
                {
                    "wall_output_tps_median": s.get("wall_output_tps", {}).get("median", 0.0),
                    "steady_output_tps_median": s.get("steady_output_tps", {}).get("median", 0.0),
                    "runtime_output_tps_median": s.get("runtime_output_tps", {}).get("median", 0.0),
                    "decode_only_output_tps_median": s.get("decode_only_output_tps", {}).get("median", 0.0),
                    "wall_total_tps_median": s.get("wall_total_tps", {}).get("median", 0.0),
                    "steady_total_tps_median": s.get("steady_total_tps", {}).get("median", 0.0),
                    "runtime_total_tps_median": s.get("runtime_total_tps", {}).get("median", 0.0),
                    "total_to_output_token_ratio": s.get("total_to_output_token_ratio", 0.0),
                    "setup_ms_median": s.get("setup_ms", {}).get("median", 0.0),
                    "steady_ms_median": s.get("steady_ms", {}).get("median", 0.0),
                    "prefill_ms_median": s.get("prefill_ms", {}).get("median", 0.0),
                    "prefill_tps_median": s.get("prefill_tps", {}).get("median", 0.0),
                    "decode_ms_median": s.get("decode_ms", {}).get("median", 0.0),
                    "harness_overhead_ms_median": s.get("harness_overhead_ms", {}).get("median", 0.0),
                    "tokenizer_load_ms_median": s.get("tokenizer_load_ms", {}).get("median", 0.0),
                    "prompt_encode_ms_median": s.get("prompt_encode_ms", {}).get("median", 0.0),
                    "model_open_ms_median": s.get("model_open_ms", {}).get("median", 0.0),
                    "model_load_ms_median": s.get("model_load_ms", {}).get("median", 0.0),
                    "model_cleanup_ms_median": s.get("model_cleanup_ms", {}).get("median", 0.0),
                    "command_total_ms_median": s.get("command_total_ms", {}).get("median", 0.0),
                    "process_overhead_ms_median": s.get("process_overhead_ms", {}).get("median", 0.0),
                    "scheduler_outer_threads_median": s.get("scheduler_outer_threads", {}).get("median", 0.0),
                    "scheduler_inner_threads_median": s.get("scheduler_inner_threads", {}).get("median", 0.0),
                    "scheduler_lm_head_threads_median": s.get("scheduler_lm_head_threads", {}).get("median", 0.0),
                    "batched_decode_calls_median": s.get("batched_decode_calls", {}).get("median", 0.0),
                    "batched_decode_tokens_median": s.get("batched_decode_tokens", {}).get("median", 0.0),
                    "batched_lm_head_calls_median": s.get("batched_lm_head_calls", {}).get("median", 0.0),
                    "batched_lm_head_tokens_median": s.get("batched_lm_head_tokens", {}).get("median", 0.0),
                    "batch_profile_total_ms_median": s.get("batch_profile_total_ms", {}).get("median", 0.0),
                    "batch_profile_alloc_ms_median": s.get("batch_profile_alloc_ms", {}).get("median", 0.0),
                    "batch_profile_qkv_ms_median": s.get("batch_profile_qkv_ms", {}).get("median", 0.0),
                    "batch_profile_rope_kv_ms_median": s.get("batch_profile_rope_kv_ms", {}).get("median", 0.0),
                    "batch_profile_attention_ms_median": s.get("batch_profile_attention_ms", {}).get("median", 0.0),
                    "batch_profile_o_proj_ms_median": s.get("batch_profile_o_proj_ms", {}).get("median", 0.0),
                    "batch_profile_gate_up_ms_median": s.get("batch_profile_gate_up_ms", {}).get("median", 0.0),
                    "batch_profile_gate_up_quant_ms_median": s.get("batch_profile_gate_up_quant_ms", {}).get("median", 0.0),
                    "batch_profile_gate_up_dot_ms_median": s.get("batch_profile_gate_up_dot_ms", {}).get("median", 0.0),
                    "batch_profile_down_proj_ms_median": s.get("batch_profile_down_proj_ms", {}).get("median", 0.0),
                    "batch_profile_down_proj_quant_ms_median": s.get("batch_profile_down_proj_quant_ms", {}).get("median", 0.0),
                    "batch_profile_down_proj_dot_ms_median": s.get("batch_profile_down_proj_dot_ms", {}).get("median", 0.0),
                    "batch_profile_lm_head_ms_median": s.get("batch_profile_lm_head_ms", {}).get("median", 0.0),
                    "batch_profile_groupwise_gemv_tile_calls_median": s.get("batch_profile_groupwise_gemv_tile_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_i8_row_pair_calls_median": s.get("batch_profile_groupwise_i8_row_pair_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_i4_row_pair_calls_median": s.get("batch_profile_groupwise_i4_row_pair_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_lm_head_argmax_calls_median": s.get("batch_profile_groupwise_lm_head_argmax_calls", {}).get("median", 0.0),
                    "batch_profile_lm_head_stack_best_calls_median": s.get("batch_profile_lm_head_stack_best_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_gate_up_fused_calls_median": s.get("batch_profile_groupwise_gate_up_fused_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_i8_gate_safe_combined_calls_median": s.get("batch_profile_groupwise_i8_gate_safe_combined_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_i8_gate_safe_combined_tile8_calls_median": s.get("batch_profile_groupwise_i8_gate_safe_combined_tile8_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_i8_gate_tile8_calls_median": s.get("batch_profile_groupwise_i8_gate_tile8_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_i8_gate_biased_calls_median": s.get("batch_profile_groupwise_i8_gate_biased_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_i8_gate_pair_calls_median": s.get("batch_profile_groupwise_i8_gate_pair_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_i8_gate_pair_unroll64_calls_median": s.get("batch_profile_groupwise_i8_gate_pair_unroll64_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_i8_gate_pair_unroll128_calls_median": s.get("batch_profile_groupwise_i8_gate_pair_unroll128_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_i8_gate_pair8_split_calls_median": s.get("batch_profile_groupwise_i8_gate_pair8_split_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_i8_gate_prefetch_calls_median": s.get("batch_profile_groupwise_i8_gate_prefetch_calls", {}).get("median", 0.0),
                    "batch_profile_groupwise_lm_head_row_pair_calls_median": s.get("batch_profile_groupwise_lm_head_row_pair_calls", {}).get("median", 0.0),
                    "batch_wall_ms_median": s.get("batch_wall_ms", {}).get("median", 0.0),
                    "generated_tokens_median": s.get("generated_tokens_median", 0),
                    "prompt_tokens_total_median": s.get("prompt_tokens_total_median", 0),
                    "total_tokens_median": s.get("total_tokens_median", 0.0),
                }
            )
        rows.append(row)
    return rows


def write_outputs_once(out_dir: Path, run_id: str, payload: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_id}_qwen25_cpu_suite.json"
    csv_path = out_dir / f"{run_id}_qwen25_cpu_suite_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    preferred = [
        "kind",
        "mode",
        "effective_mode",
        "batch_size",
        "target_prompt_tokens",
        "max_seq_len",
        "max_new_tokens",
        "threads",
        "suite_build",
        "quant",
        "run_canary_status",
        "run_canary_decode_tps",
        "batch_canary_status",
        "batch_canary_runtime_tps",
        "batch_canary_decode_only_tps",
        "batch_canary_wall_total_tps",
        "threads_per_worker",
        "native_continuous_batching",
        "ok",
        "runs",
        "canary_status",
        "canary_good_threshold",
        "canary_weak_threshold",
        "sweep_quality_status",
        "sweep_quality_good_threshold",
        "sweep_quality_weak_threshold",
        "prefill_ms_median",
        "prefill_tps_median",
        "decode_tps_median",
        "total_tps_median",
        "wall_ms_median",
        "runtime_total_bytes",
        "wall_output_tps_median",
        "steady_output_tps_median",
        "runtime_output_tps_median",
        "decode_only_output_tps_median",
        "wall_total_tps_median",
        "steady_total_tps_median",
        "runtime_total_tps_median",
        "total_to_output_token_ratio",
        "setup_ms_median",
        "steady_ms_median",
        "decode_ms_median",
        "harness_overhead_ms_median",
        "tokenizer_load_ms_median",
        "prompt_encode_ms_median",
        "model_open_ms_median",
        "model_load_ms_median",
        "model_cleanup_ms_median",
        "command_total_ms_median",
        "process_overhead_ms_median",
        "scheduler_outer_threads_median",
        "scheduler_inner_threads_median",
        "scheduler_lm_head_threads_median",
        "batched_decode_calls_median",
        "batched_decode_tokens_median",
        "batched_lm_head_calls_median",
        "batched_lm_head_tokens_median",
        "batch_profile_total_ms_median",
        "batch_profile_alloc_ms_median",
        "batch_profile_qkv_ms_median",
        "batch_profile_rope_kv_ms_median",
        "batch_profile_attention_ms_median",
        "batch_profile_o_proj_ms_median",
        "batch_profile_gate_up_ms_median",
        "batch_profile_gate_up_quant_ms_median",
        "batch_profile_gate_up_dot_ms_median",
        "batch_profile_down_proj_ms_median",
        "batch_profile_down_proj_quant_ms_median",
        "batch_profile_down_proj_dot_ms_median",
        "batch_profile_lm_head_ms_median",
        "batch_profile_groupwise_gemv_tile_calls_median",
        "batch_profile_groupwise_i8_row_pair_calls_median",
        "batch_profile_groupwise_i4_row_pair_calls_median",
        "batch_profile_groupwise_lm_head_argmax_calls_median",
        "batch_profile_lm_head_stack_best_calls_median",
        "batch_profile_groupwise_gate_up_fused_calls_median",
        "batch_profile_groupwise_i8_gate_safe_combined_calls_median",
        "batch_profile_groupwise_i8_gate_safe_combined_tile8_calls_median",
        "batch_profile_groupwise_i8_gate_tile8_calls_median",
        "batch_profile_groupwise_i8_gate_biased_calls_median",
        "batch_profile_groupwise_i8_gate_pair_calls_median",
        "batch_profile_groupwise_i8_gate_pair_unroll64_calls_median",
        "batch_profile_groupwise_i8_gate_pair_unroll128_calls_median",
        "batch_profile_groupwise_i8_gate_pair8_split_calls_median",
        "batch_profile_groupwise_i8_gate_prefetch_calls_median",
        "batch_profile_groupwise_lm_head_row_pair_calls_median",
        "batch_wall_ms_median",
        "generated_tokens_median",
        "prompt_tokens_total_median",
        "total_tokens_median",
    ]
    seen = {key for row in rows for key in row.keys()}
    fieldnames = [key for key in preferred if key in seen]
    fieldnames.extend(sorted(seen.difference(fieldnames)))
    if not fieldnames:
        fieldnames = ["kind"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def write_outputs(out_dir: Path, run_id: str, payload: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    fallback_dir = Path(tempfile.gettempdir()) / "microgemm_bench_results" / out_dir.name
    targets = [out_dir]
    if str(fallback_dir) != str(out_dir):
        targets.append(fallback_dir)

    last_error: OSError | None = None
    for target in targets:
        try:
            return write_outputs_once(target, run_id, payload, rows)
        except OSError as exc:
            last_error = exc
            print(
                f"Warning: failed to write benchmark outputs to {target}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("failed to write benchmark outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5 CPU suite for MicroGemm")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--cache-dir", default="/content/microgemm_qwen25_cache")
    parser.add_argument("--out-dir", default="bench_results/qwen25_cpu_microgemm")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--prompt-tokens", default="64,256,512")
    parser.add_argument("--batch-prompt-tokens", type=int, default=256)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--batch-modes", default="adaptive,continuous,serial,concurrent")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ignore-eos", action="store_true", help="Benchmark fixed decode length by not stopping on EOS")
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--threads-per-worker", type=int, default=0)
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--quant", choices=QUANT_CHOICES, default="int8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--allow-slow-canary",
        action="store_true",
        help="With --strict-canary-gate, run batch sweep even when a canary is slow",
    )
    parser.add_argument(
        "--allow-weak-canary",
        action="store_true",
        help="With --strict-canary-gate, run batch sweep when a canary is weak but not slow",
    )
    parser.add_argument(
        "--strict-canary-gate",
        action="store_true",
        help="Use canary verdicts as hard gates. By default canaries are advisory and the requested sweep still runs.",
    )
    parser.add_argument(
        "--canary-gate",
        choices=("advisory", "strict"),
        default="advisory",
        help="Compatibility alias for canary gating. 'strict' behaves like --strict-canary-gate.",
    )
    parser.add_argument(
        "--canary-profile",
        default="",
        help="Optional canary profile name override, e.g. llama32_1b_cpu_avx2, llama31_8b_cpu_avx2, mistral7b_cpu_avx2, phi4_14b_cpu_avx2, granite33_2b_cpu_avx2, glm4_9b_0414_cpu_avx2.",
    )
    parser.add_argument(
        "--skip-batch-canary",
        action="store_true",
        help="Skip the extra batch canary preflight and run the requested batch sweep directly.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.quant = canonical_quant(args.quant)
    root = Path(__file__).resolve().parents[2]
    micro_dir = root / "microgemm"
    run_id = args.run_id or time.strftime("qwen25_cpu_%Y%m%d_%H%M%S")
    threads = int(args.threads or (os.cpu_count() or 1))
    args.threads = threads
    if args.canary_gate == "strict":
        args.strict_canary_gate = True
    profile_key = args.canary_profile or args.model_repo or args.model_dir
    args.canary_profile = infer_canary_profile(profile_key)
    cpu_snapshot = collect_cpu_snapshot()

    print("Qwen2.5 MicroGemm CPU suite")
    print(f"  repo root:    {root}")
    print(f"  model:        {args.model_dir or args.model_repo}")
    print(f"  threads:      {threads}")
    print(f"  prompt sweep: {args.prompt_tokens}")
    print(f"  batch sizes:  {args.batch_sizes}")
    print(f"  max new tok:  {args.max_new_tokens}")
    print(f"  ignore eos:   {bool(args.ignore_eos)}")
    print(f"  quant:        {args.quant}")
    print(f"  suite build:  {SUITE_BUILD_TAG}")
    print(f"  canary prof:  {args.canary_profile.get('name', '')}")
    print(f"  canary gate:  {'strict' if bool(args.strict_canary_gate) else 'advisory'}")
    print(f"  batch canary: {'off' if bool(args.skip_batch_canary) else 'on'}")
    print_cpu_snapshot(cpu_snapshot)
    if args.quant.startswith("int4"):
        print_int4_kernel_toggles(args.quant)
    elif args.quant.startswith("int8"):
        print_int8_kernel_toggles(args.quant)

    cli_bin, convert_bin, text_bin = ensure_microgemm_built(root, Path(args.cache_dir).resolve())
    model_dir = Path(args.model_dir).resolve() if args.model_dir else None
    if model_dir is None:
        if args.skip_download:
            raise SystemExit("--skip-download requires --model-dir")
        model_dir = download_model(args.model_repo, Path(args.cache_dir))
    model_dir = model_dir.resolve()
    ensure_supported_safetensors(model_dir)

    mgm_filename = mgm_filename_for_quant(args.quant)
    mgm_path = Path(args.cache_dir).resolve() / "mgm" / args.model_repo.replace("/", "__") / mgm_filename
    if args.model_dir:
        mgm_path = Path(args.cache_dir).resolve() / "mgm" / model_dir.name / mgm_filename
    args.resolved_model_dir = model_dir
    args.resolved_mgm_path = mgm_path
    ensure_mgm(
        convert_bin,
        model_dir,
        mgm_path,
        force=bool(args.force_convert),
        kv_block_size=int(args.kv_block_size),
        quant=args.quant,
    )

    tokenizer_json = model_dir / "tokenizer.json"
    tokenizer = load_tokenizer(model_dir)
    probe_text_profile_runtime(text_bin, mgm_path, tokenizer_json, threads)

    print("MicroGemm binaries")
    print(f"  cli:          {cli_bin}")
    print(f"  convert:      {convert_bin}")
    print(f"  text:         {text_bin}")
    print(f"  model_dir:    {model_dir}")
    print(f"  mgm:          {mgm_path}")

    print("MicroGemm capabilities")
    caps = run_checked([str(cli_bin), "capabilities"], cwd=micro_dir)
    print(caps.stdout.strip())

    print("Inspect converted model")
    inspect = run_checked([str(cli_bin), "inspect", str(mgm_path)], cwd=micro_dir)
    print("\n".join(inspect.stdout.splitlines()[:16]))

    all_results: list[dict[str, Any]] = []
    context_results = run_context_sweep(args, text_bin, mgm_path, tokenizer_json, tokenizer)
    run_canary = summarize_run_canary(context_results, cpu_snapshot, args.canary_profile)
    print_run_canary(run_canary)
    all_results.extend(context_results)
    canary_status = run_canary.get("status")
    should_skip_batch = should_gate_on_canary_status(canary_status, args)
    if should_skip_batch:
        print(
            f"Skipping batch sweep: run canary is {canary_status}. "
            "Disable --strict-canary-gate, pass --allow-weak-canary, or pass --allow-slow-canary to collect sweep data.",
            flush=True,
        )
        json_path, csv_path = write_suite_results(args, run_id, cpu_snapshot, run_canary, None, all_results)
        print("Wrote:")
        print(f"  json: {json_path}")
        print(f"  csv:  {csv_path}")
        return 0

    batch_canary_results: list[dict[str, Any]] = []
    batch_canary_result: dict[str, Any] | None = None
    batch_canary: dict[str, Any] = {}
    if bool(args.skip_batch_canary):
        print("Skipping batch canary preflight: running requested batch sweep directly.", flush=True)
    else:
        for batch_canary_size in BATCH_CANARY_SIZES:
            selected, selected_canary, attempts = run_batch_canary_with_retries(
                args,
                text_bin,
                mgm_path,
                tokenizer_json,
                tokenizer,
                batch_size=batch_canary_size,
            )
            batch_canary_results.extend(attempts)
            batch_canary_result = selected
            batch_canary = selected_canary
            if not batch_canary_status_allowed(selected_canary.get("status"), args):
                break

        if batch_canary_result is None:
            batch_canary_result = max(
                batch_canary_results,
                key=lambda item: float(
                    item.get("summary", {}).get("runtime_output_tps", {}).get("median", 0.0) or 0.0
                ),
            )
            batch_canary = summarize_batch_canary(batch_canary_result, args.canary_profile)
        all_results.extend(batch_canary_results)
        batch_canary_status = batch_canary.get("status")
        should_skip_batch = should_gate_on_canary_status(batch_canary_status, args)
        if should_skip_batch:
            print(
                "Skipping batch sweep: "
                f"batch{int(batch_canary.get('batch_size', 0) or 0)} canary is {batch_canary_status} "
                f"(runtime={float(batch_canary.get('runtime_tps', 0.0) or 0.0):.2f} tok/s, "
                f"decode_only={float(batch_canary.get('decode_only_tps', 0.0) or 0.0):.2f} tok/s, "
                f"good>={float(batch_canary.get('good_threshold', 0.0) or 0.0):.0f}). "
                "Disable --strict-canary-gate, pass --allow-weak-canary, or pass --allow-slow-canary to collect sweep data.",
                flush=True,
            )
            json_path, csv_path = write_suite_results(args, run_id, cpu_snapshot, run_canary, batch_canary_result, all_results)
            print("Wrote:")
            print(f"  json: {json_path}")
            print(f"  csv:  {csv_path}")
            return 0

    all_results.extend(run_batch_sweep(args, text_bin, mgm_path, tokenizer_json, tokenizer))

    json_path, csv_path = write_suite_results(args, run_id, cpu_snapshot, run_canary, batch_canary_result, all_results)
    print("Wrote:")
    print(f"  json: {json_path}")
    print(f"  csv:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

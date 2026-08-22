"""Run vLLM baselines for selected Qwen-family models.

This runner uses benchmark_inference_matrix.py with --backend vllm so the output
format matches the existing MegaGemm/HF benchmark reports.
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
class ModelPreset:
    key: str
    title: str
    model: str
    notes: str = ""


PRESETS: dict[str, ModelPreset] = {
    "qwen25-3b": ModelPreset(
        "qwen25-3b",
        "Qwen 2.5 3B Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Stable vLLM architecture path via Qwen2ForCausalLM.",
    ),
    "qwen25-7b": ModelPreset(
        "qwen25-7b",
        "Qwen 2.5 7B Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "L4 capacity probe for the larger Qwen2.5 dense instruct checkpoint.",
    ),
    "qwen3-4b": ModelPreset(
        "qwen3-4b",
        "Qwen 3 4B",
        "Qwen/Qwen3-4B",
        "Use as the Qwen3 text baseline closest to the Qwen 3.5 4B run.",
    ),
    "qwen3-8b": ModelPreset(
        "qwen3-8b",
        "Qwen 3 8B",
        "Qwen/Qwen3-8B",
        "L4 capacity probe for the larger Qwen3 dense checkpoint.",
    ),
    "qwen35-4b": ModelPreset(
        "qwen35-4b",
        "Qwen 3.5 4B",
        "Qwen/Qwen3.5-4B",
        "May require a very recent vLLM/Transformers stack.",
    ),
    "qwen35-9b": ModelPreset(
        "qwen35-9b",
        "Qwen 3.5 9B",
        "Qwen/Qwen3.5-9B",
        "Larger Qwen 3.5 capacity probe; prefer text-only serving when possible.",
    ),
    "qwen35-2b": ModelPreset(
        "qwen35-2b",
        "Qwen 3.5 2B",
        "Qwen/Qwen3.5-2B",
        "Smaller Qwen 3.5 capacity probe.",
    ),
    "qwen35-08b": ModelPreset(
        "qwen35-08b",
        "Qwen 3.5 0.8B",
        "Qwen/Qwen3.5-0.8B",
        "Smallest Qwen 3.5 smoke/capacity probe.",
    ),
}

ALIASES = {
    "core": ("qwen25-3b", "qwen3-4b", "qwen35-4b"),
    "qwen25-all": ("qwen25-3b", "qwen25-7b"),
    "qwen3-all": ("qwen3-4b", "qwen3-8b"),
    "l4-core": ("qwen25-3b", "qwen25-7b", "qwen3-4b", "qwen3-8b"),
    "qwen35-all": ("qwen35-08b", "qwen35-2b", "qwen35-4b", "qwen35-9b"),
    "all": (
        "qwen25-3b",
        "qwen25-7b",
        "qwen3-4b",
        "qwen3-8b",
        "qwen35-08b",
        "qwen35-2b",
        "qwen35-4b",
        "qwen35-9b",
    ),
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


def parse_models(raw: str) -> list[ModelPreset]:
    names: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part in ALIASES:
            names.extend(ALIASES[part])
        else:
            names.append(part)
    if not names:
        raise SystemExit("No models selected")

    models: list[ModelPreset] = []
    seen = set()
    for name in names:
        if name not in PRESETS:
            valid = sorted([*PRESETS.keys(), *ALIASES.keys()])
            raise SystemExit(f"Unknown model preset {name!r}. Valid: {', '.join(valid)}")
        if name not in seen:
            models.append(PRESETS[name])
            seen.add(name)
    return models


def output_stem(args: argparse.Namespace, run_id: str, preset: ModelPreset) -> str:
    return f"{run_id}_{preset.key}_{args.hardware_label}_vllm"


def output_paths(args: argparse.Namespace, run_id: str, preset: ModelPreset) -> dict[str, Path]:
    stem = output_stem(args, run_id, preset)
    out_dir = Path(args.out_dir)
    return {
        "raw": out_dir / f"{stem}.jsonl",
        "summary": out_dir / f"{stem}_summary.json",
        "csv": out_dir / f"{stem}_summary.csv",
    }


def command(args: argparse.Namespace, run_id: str, preset: ModelPreset) -> list[str]:
    cmd = [
        sys.executable,
        str(MATRIX),
        "--backend",
        "vllm",
        "--model",
        preset.model,
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
        f"{run_id}_{preset.key}",
        "--device",
        "cuda",
        "--dtype",
        args.dtype,
        "--max-seq-len",
        str(args.max_seq_len),
        "--max-batch-size",
        str(args.max_batch_size),
        "--vllm-tensor-parallel-size",
        str(args.vllm_tensor_parallel_size),
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm-max-model-len",
        str(args.vllm_max_model_len or args.max_seq_len),
    ]
    if args.cache_dir:
        cmd.extend(["--cache-dir", args.cache_dir])
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.ignore_eos:
        cmd.append("--ignore-eos")
    if args.vllm_enforce_eager:
        cmd.append("--vllm-enforce-eager")
    if args.vllm_language_model_only:
        cmd.append("--vllm-language-model-only")
    if not args.vllm_prefix_caching:
        cmd.append("--vllm-disable-prefix-caching")
    if args.vllm_disable_cudagraph_memory_profiler:
        cmd.append("--vllm-disable-cudagraph-memory-profiler")
    return cmd


def load_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    summary = paths["summary"]
    if not summary.exists():
        return []
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return payload.get("rows", [])


def write_report(args: argparse.Namespace, run_id: str, models: list[ModelPreset]) -> Path:
    hardware_label = safe_report_label(args.hardware_label)
    report = Path(args.out_dir) / f"{run_id}_qwen_vllm_{hardware_label}_report.md"
    lines = [
        f"# Qwen vLLM {args.hardware_label} Benchmark",
        "",
        "## Run Configuration",
        "",
        f"- Hardware label: `{args.hardware_label}`",
        f"- Dtype: `{args.dtype}`",
        f"- Batch sizes: `{args.batch_sizes}`",
        f"- Prompt tokens/request: `{args.prompt_tokens}`",
        f"- Max new tokens/request: `{args.max_new_tokens}`",
        f"- Repeats: `{args.repeats}`",
        f"- vLLM tensor parallel size: `{args.vllm_tensor_parallel_size}`",
        f"- vLLM GPU memory utilization: `{args.vllm_gpu_memory_utilization}`",
        f"- vLLM prefix caching: `{args.vllm_prefix_caching}`",
        f"- vLLM CUDA graph memory profiler: `{not args.vllm_disable_cudagraph_memory_profiler}`",
        f"- Run id: `{run_id}`",
        "",
        "## Summary",
        "",
        "| Model | Scenario | Batch | Prompt tok/req | Median output tok/s | OK |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for preset in models:
        rows = load_rows(output_paths(args, run_id, preset))
        if not rows:
            lines.append(f"| {preset.title} | no rows | | | | |")
            continue
        rows.sort(
            key=lambda row: (
                str(row.get("scenario")),
                int(row.get("prompt_tokens_requested_per_request", 0)),
                int(row.get("batch_size", 0)),
            )
        )
        for row in rows:
            lines.append(
                f"| {preset.title} | {row.get('scenario')} | {row.get('batch_size')} | "
                f"{row.get('prompt_tokens_requested_per_request')} | "
                f"{float(row.get('median_output_tps') or 0.0):.2f} | "
                f"{row.get('ok_samples')}/{row.get('samples')} |"
            )

    lines.extend(["", "## Files", ""])
    for preset in models:
        lines.append(f"- {preset.title} (`{preset.model}`):")
        for label, path in output_paths(args, run_id, preset).items():
            lines.append(f"  - {label}: `{path}`")
        if preset.notes:
            lines.append(f"  - note: {preset.notes}")

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(
    *,
    default_models: str = "core",
    default_hardware_label: str = "1xt4",
    default_out_dir: str = "bench_results/qwen_vllm_t4",
    default_run_prefix: str = "qwen_vllm_t4",
) -> int:
    parser = argparse.ArgumentParser(description="Qwen-family vLLM benchmark suite")
    parser.add_argument(
        "--models",
        default=default_models,
        help=(
            "core, l4-core, qwen25-all, qwen3-all, qwen35-all, all, "
            "or comma-separated preset names"
        ),
    )
    parser.add_argument("--hardware-label", default=default_hardware_label)
    parser.add_argument("--out-dir", default=default_out_dir)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--prompt-tokens", default="128,512,1024,2048")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--allow-eos", dest="ignore_eos", action="store_false")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-model-len", type=int, default=0)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-language-model-only", action="store_true")
    parser.add_argument(
        "--vllm-prefix-caching",
        action="store_true",
        help="Allow vLLM prefix caching. Disabled by default for fair repeated-prompt matrices.",
    )
    parser.add_argument("--vllm-disable-cudagraph-memory-profiler", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.out_dir = normalize_out_dir(args.out_dir)
    models = parse_models(args.models)
    run_id = args.run_id or f"{default_run_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    args.out_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Qwen vLLM {args.hardware_label} benchmark suite")
    print(f"  run_id: {run_id}")
    print(f"  models: {', '.join(model.key for model in models)}")
    print(f"  out:    {args.out_dir}")

    for preset in models:
        cmd = command(args, run_id, preset)
        print()
        print(f"=== {preset.title} ===")
        print(shell_join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(ROOT), check=True)

    if args.dry_run:
        print("\nDry run only; no report was generated from results.")
        return 0

    report = write_report(args, run_id, models)
    print()
    print(f"Wrote combined report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

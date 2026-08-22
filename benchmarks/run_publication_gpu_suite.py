"""Run a compact, same-environment MegaGemm/vLLM publication matrix.

This wrapper launches each backend in a fresh child process, keeps the workload
identical, disables vLLM prefix caching, writes the standard JSONL/JSON/CSV
artifacts, emits comparison CSV files, and packages the run directory as ZIP.
It does not install or upgrade dependencies.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "benchmarks" / "benchmark_inference_matrix.py"
COMPARE = ROOT / "benchmarks" / "compare_inference_summaries.py"


@dataclass(frozen=True)
class Variant:
    name: str
    backend: str
    dtype: str = "fp16"
    quantize: str | None = None


VARIANTS = {
    "megagemm-fp16": Variant("megagemm-fp16", "megagemm"),
    "vllm-fp16": Variant("vllm-fp16", "vllm"),
    "megagemm-bf16": Variant("megagemm-bf16", "megagemm", dtype="bf16"),
    "vllm-bf16": Variant("vllm-bf16", "vllm", dtype="bf16"),
    "megagemm-int8": Variant(
        "megagemm-int8", "megagemm", dtype="fp16", quantize="int8"
    ),
}


def parse_variants(raw: str) -> list[Variant]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise SystemExit(
            f"unknown variants: {', '.join(unknown)}; valid: {', '.join(VARIANTS)}"
        )
    if not names:
        raise SystemExit("at least one variant is required")
    return [VARIANTS[name] for name in names]


def shell_join(command: list[str]) -> str:
    return shlex.join(command)


def auto_hardware_label() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "no-gpu"
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0).lower()
        aliases = (
            ("tesla t4", "t4"),
            ("nvidia t4", "t4"),
            ("l4", "l4"),
            ("a100", "a100"),
            ("h100", "h100"),
            ("a10g", "a10g"),
            ("v100", "v100"),
            ("p100", "p100"),
        )
        suffix = next((label for needle, label in aliases if needle in name), None)
        if suffix is None:
            suffix = "".join(ch for ch in name if ch.isalnum())[:24] or "gpu"
        return f"{count}x{suffix}"
    except Exception:
        return "unknown-gpu"


def command_for(
    args: argparse.Namespace,
    variant: Variant,
    *,
    hardware_label: str,
    run_dir: Path,
    run_id: str,
) -> list[str]:
    command = [
        sys.executable,
        str(MATRIX),
        "--backend",
        variant.backend,
        "--model",
        args.model,
        "--hardware-label",
        hardware_label,
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
        str(run_dir),
        "--run-id",
        f"{run_id}_{variant.name}",
        "--device",
        "cuda",
        "--dtype",
        variant.dtype,
        "--max-seq-len",
        str(args.max_seq_len),
        "--max-batch-size",
        str(args.max_batch_size),
        "--ignore-eos",
    ]
    if variant.quantize:
        command.extend(["--quantize", variant.quantize])
    if variant.backend == "vllm":
        command.extend(
            [
                "--vllm-tensor-parallel-size",
                "1",
                "--vllm-gpu-memory-utilization",
                str(args.vllm_gpu_memory_utilization),
                "--vllm-max-model-len",
                str(args.max_seq_len),
                "--vllm-disable-prefix-caching",
            ]
        )
        if args.vllm_language_model_only:
            command.append("--vllm-language-model-only")
        if args.vllm_enforce_eager:
            command.append("--vllm-enforce-eager")
    return command


def summary_path(
    run_dir: Path,
    run_id: str,
    hardware_label: str,
    variant: Variant,
) -> Path:
    return run_dir / (
        f"{run_id}_{variant.name}_{hardware_label}_{variant.backend}_summary.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact same-environment MegaGemm/vLLM GPU matrix"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--hardware-label", default="")
    parser.add_argument(
        "--variants",
        default="megagemm-fp16,vllm-fp16",
        help=(
            "Comma-separated: megagemm-fp16,vllm-fp16,megagemm-bf16,"
            "vllm-bf16,megagemm-int8"
        ),
    )
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--prompt-tokens", default="128,512,2048")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-language-model-only", action="store_true")
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--out-dir", default="bench_results/publication_gpu")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    hardware_label = args.hardware_label or auto_hardware_label()
    run_id = args.run_id or f"publication_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir = run_dir / run_id

    commands = [
        command_for(
            args,
            variant,
            hardware_label=hardware_label,
            run_dir=run_dir,
            run_id=run_id,
        )
        for variant in variants
    ]

    print("Publication GPU suite")
    print(f"  model:    {args.model}")
    print(f"  hardware: {hardware_label}")
    print(f"  variants: {', '.join(variant.name for variant in variants)}")
    print(f"  output:   {run_dir}")
    for command in commands:
        print(f"\n{shell_join(command)}")
    if args.dry_run:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "model": args.model,
        "hardware_label": hardware_label,
        "variants": [variant.name for variant in variants],
        "args": vars(args),
        "commands": commands,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    for variant, command in zip(variants, commands):
        print(f"\n=== {variant.name} ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    for vllm in (item for item in variants if item.backend == "vllm"):
        right = summary_path(run_dir, run_id, hardware_label, vllm)
        for left in (
            item
            for item in variants
            if item.backend == "megagemm" and item.dtype == vllm.dtype
        ):
            left_path = summary_path(run_dir, run_id, hardware_label, left)
            if not left_path.exists() or not right.exists():
                continue
            csv_path = run_dir / f"compare_{left.name}_vs_{vllm.name}.csv"
            compare_command = [
                sys.executable,
                str(COMPARE),
                "--left",
                str(left_path),
                "--right",
                str(right),
                "--left-name",
                left.name,
                "--right-name",
                vllm.name,
                "--csv",
                str(csv_path),
            ]
            subprocess.run(compare_command, cwd=ROOT, check=True)

    archive_path = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    print(f"\nAttach this artifact to the benchmark report: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

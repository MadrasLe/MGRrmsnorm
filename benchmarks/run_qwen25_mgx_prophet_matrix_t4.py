"""Run Qwen 2.5 MGX and MGX+Prophet through the standard TPS matrix.

This is the apples-to-apples companion to run_qwen_vllm_t4_suite.py: it uses
benchmark_inference_matrix.py, so the output_tps summary columns are the same
as the vLLM matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MATRIX = ROOT / "benchmarks" / "benchmark_inference_matrix.py"
DEFAULT_MGX = "artifacts/Qwen--Qwen2.5-3B-Instruct-fp16.mgx"


@dataclass(frozen=True)
class Mode:
    name: str
    title: str
    backend: str
    prophet_prime_before_measure: bool = False


MODES: dict[str, Mode] = {
    "fresh": Mode(
        name="fresh",
        title="MGX INT8 fresh",
        backend="megagemm",
    ),
    "prophet-repeat": Mode(
        name="prophet-repeat",
        title="MGX INT8 + Prophet repeated prompts",
        backend="megagemm-prophet",
        prophet_prime_before_measure=False,
    ),
    "prophet-warm": Mode(
        name="prophet-warm",
        title="MGX INT8 + Prophet warm cache",
        backend="megagemm-prophet",
        prophet_prime_before_measure=True,
    ),
}

ALIASES = {
    "core": ("fresh", "prophet-repeat"),
    "all": ("fresh", "prophet-repeat", "prophet-warm"),
}


def mode_title(args: argparse.Namespace, mode: Mode) -> str:
    label = (normalize_quantize(args.quantize) or args.dtype).upper()
    return mode.title.replace("INT8", label)


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


def warn_if_known_regressed_experiments(args: argparse.Namespace) -> None:
    if not args.flat_fast_gate_up:
        print(
            "Warning: --no-flat-fast-gate-up is diagnostic only; on "
            "Qwen2.5-3B FP16 T4 batch=8 it regressed cached decode versus "
            "the known-good flat_fast_gate_up=True path."
        )
    if args.flat_fast_down:
        print(
            "Warning: --flat-fast-down is diagnostic only; on Qwen2.5-3B FP16 "
            "T4 batch=8 it regressed the MLP/down path badly."
        )
    if (
        args.fast_gemv_force_triton
        or args.fast_gemv_mode != "auto"
        or args.fast_gemv_max_rows > 4
    ):
        print(
            "Warning: fast GEMV forcing/max_rows>4 is diagnostic only; on "
            "Qwen2.5-3B FP16 T4 batch=8 it regressed flat gate_up. The "
            "known-good decode config is block_size=64, fast_gemv=auto, "
            "fast_gemv_max_rows=4, flat_fast_down=False."
        )


def parse_modes(raw: str) -> list[Mode]:
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
        raise SystemExit("No modes selected")

    modes: list[Mode] = []
    seen = set()
    for name in names:
        if name not in MODES:
            valid = sorted([*MODES.keys(), *ALIASES.keys()])
            raise SystemExit(f"Unknown mode {name!r}. Valid: {', '.join(valid)}")
        if name not in seen:
            modes.append(MODES[name])
            seen.add(name)
    return modes


def repo_path(path: str | os.PathLike[str]) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return ROOT / expanded


def normalize_quantize(name: str | None) -> str | None:
    if name is None:
        return None
    key = str(name).strip().lower()
    return None if key in {"", "none"} else key


def slugify_model_ref(model_ref: str) -> str:
    slug = model_ref.strip().replace("\\", "/").strip("/")
    if not slug:
        slug = "model"
    slug = slug.replace("/", "--")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug)
    return slug[:120]


def default_mgx_path(model_ref: str, dtype_name: str, quantize_name: str | None) -> str:
    suffix = quantize_name if quantize_name else dtype_name
    return f"artifacts/{slugify_model_ref(model_ref)}-{suffix}.mgx"


def ensure_mgx_artifact(args: argparse.Namespace) -> None:
    mgx_path = repo_path(args.mgx)
    payload_cache_dir = args.mgx_payload_cache_dir or None
    quantize_mode = normalize_quantize(args.quantize)

    if args.force_export or not mgx_path.exists():
        if not args.export_if_missing and not args.force_export:
            raise SystemExit(
                f"MGX artifact not found: {mgx_path}. "
                "Use --export-if-missing, or pass --mgx to an existing artifact."
            )

        from megagemm.models import export_to_mgx

        mgx_path.parent.mkdir(parents=True, exist_ok=True)
        reason = "force export requested" if args.force_export else "artifact missing"
        print()
        print("=" * 80)
        print("EXPORTING MGX")
        print("=" * 80)
        print(f"Reason:           {reason}")
        print(f"Model source:     {args.model}")
        print(f"MGX output:       {mgx_path}")
        print(f"DType:            {args.dtype}")
        print(f"Quantize:         {quantize_mode or 'none'}")
        print(f"Payload cache:    {args.mgx_emit_payload_cache}")
        print(f"Export mode:      {args.mgx_export_mode}")
        export_start = time.perf_counter()
        export_to_mgx(
            args.model,
            mgx_path,
            dtype=args.dtype,
            quantize=quantize_mode,
            emit_payload_cache=args.mgx_emit_payload_cache,
            payload_cache_dir=payload_cache_dir,
            export_mode=args.mgx_export_mode,
        )
        export_seconds = time.perf_counter() - export_start
        print(f"[MGX matrix] Export finished in {export_seconds:.2f}s")
        return

    if args.mgx_emit_payload_cache:
        from megagemm.models import prime_mgx_payload_cache

        print()
        print("=" * 80)
        print("PRIMING MGX PAYLOAD CACHE")
        print("=" * 80)
        print(f"MGX artifact:     {mgx_path}")
        print(f"Payload cache dir:{payload_cache_dir or '(artifact default)'}")
        prime_mgx_payload_cache(
            mgx_path,
            validate_payload_hash=not args.mgx_skip_hash_check,
            payload_cache_dir=payload_cache_dir,
        )


def output_paths(args: argparse.Namespace, run_id: str, mode: Mode) -> dict[str, Path]:
    mode_run_id = f"{run_id}_{mode.name}"
    stem = f"{mode_run_id}_{args.hardware_label}_{mode.backend}"
    out_dir = Path(args.out_dir)
    return {
        "raw": out_dir / f"{stem}.jsonl",
        "summary": out_dir / f"{stem}_summary.json",
        "csv": out_dir / f"{stem}_summary.csv",
    }


def command(args: argparse.Namespace, run_id: str, mode: Mode) -> list[str]:
    mode_run_id = f"{run_id}_{mode.name}"
    quantize_mode = normalize_quantize(args.quantize)
    cmd = [
        sys.executable,
        str(MATRIX),
        "--backend",
        mode.backend,
        "--model",
        args.mgx,
        "--tokenizer",
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
        "cuda",
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
        "--mgx-prefer-payload-cache",
    ]
    if quantize_mode:
        cmd.extend(["--quantize", quantize_mode])
    if args.cache_dir:
        cmd.extend(["--cache-dir", args.cache_dir])
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.ignore_eos:
        cmd.append("--ignore-eos")
    if args.mgx_payload_cache_dir:
        cmd.extend(["--mgx-payload-cache-dir", args.mgx_payload_cache_dir])
    if mode.backend == "megagemm-prophet":
        prophet_dir = Path(args.out_dir) / f"{mode_run_id}_prophet_library"
        cmd.extend(
            [
                "--prophet-dir",
                str(prophet_dir),
                "--prophet-reset-dir",
                "--prophet-validation-mode",
                args.prophet_validation_mode,
                "--prophet-validation-tokens",
                str(args.prophet_validation_tokens),
                "--prophet-prefix-tokens",
                str(args.prophet_prefix_tokens),
                "--prophet-top-k",
                str(args.prophet_top_k),
                "--prophet-min-similarity",
                str(args.prophet_min_similarity),
                "--prophet-min-prefix-coverage",
                str(args.prophet_min_prefix_coverage),
                "--prophet-min-prefix-reuse-score",
                str(args.prophet_min_prefix_reuse_score),
                "--prophet-max-prefix-rollback-ratio",
                str(args.prophet_max_prefix_rollback_ratio),
                "--prophet-max-prefix-tail-ratio",
                str(args.prophet_max_prefix_tail_ratio),
            ]
        )
        if args.prophet_batch_exact_restore:
            cmd.append("--prophet-batch-exact-restore")
        else:
            cmd.append("--no-prophet-batch-exact-restore")
        if args.prophet_live_prefix_cache:
            cmd.append("--prophet-live-prefix-cache")
        else:
            cmd.append("--no-prophet-live-prefix-cache")
        if args.prophet_resident_cache:
            cmd.append("--prophet-resident-cache")
        else:
            cmd.append("--no-prophet-resident-cache")
        cmd.extend(
            [
                "--prophet-resident-cache-max-entries",
                str(args.prophet_resident_cache_max_entries),
            ]
        )
        if mode.prophet_prime_before_measure:
            cmd.append("--prophet-prime-before-measure")
        if args.prophet_fallback_to_prefill:
            cmd.append("--prophet-fallback-to-prefill")
    return cmd


def command_env(args: argparse.Namespace, mode: Mode) -> dict[str, str]:
    env = os.environ.copy()
    env["MEGAGEMM_DECODE_CUDA_GRAPHS"] = "1" if args.decode_cuda_graphs else "0"
    env["MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP"] = (
        "1" if args.decode_cuda_graphs_prefer_step else "0"
    )
    env["MEGAGEMM_DECODE_CUDA_GRAPHS_SHAPE_CACHE"] = (
        "1" if args.decode_cuda_graphs_shape_cache else "0"
    )
    env["MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE"] = (
        "1" if args.decode_cuda_graphs_shared_shape_cache else "0"
    )
    env["MEGAGEMM_DECODE_CUDA_GRAPHS_STABLE_MAX_BLOCKS"] = (
        "1" if args.decode_cuda_graphs_stable_max_blocks else "0"
    )
    env["MEGAGEMM_DECODE_CUDA_GRAPHS_MIN_BATCH"] = str(args.decode_cuda_graphs_min_batch)
    env["MEGAGEMM_DECODE_CUDA_GRAPHS_LOG_LIMIT"] = str(args.decode_cuda_graphs_log_limit)
    env.setdefault("MEGAGEMM_FLAT_DEEPFUSION_DOWN", "0")
    env["MEGAGEMM_PREFILL_PREFER_PADDED"] = "1" if args.prefill_prefer_padded else "0"
    env["MEGAGEMM_PREFILL_PAD_WASTE_THRESHOLD"] = str(args.prefill_pad_waste_threshold)
    env["MEGAGEMM_PACKED_ATTN_PREFER_TRITON"] = (
        "1" if args.packed_attn_backend == "triton" else "0"
    )
    env["MEGAGEMM_PACKED_ATTN_UNIFORM_BATCH"] = (
        "1" if args.packed_attn_backend == "uniform" else "0"
    )
    env["MEGAGEMM_PACKED_ATTN_UNIFORM_MAX_SCORE_MB"] = str(
        args.packed_attn_uniform_max_score_mb
    )
    env["MEGAGEMM_PACKED_ATTN_UNIFORM_RESERVE_MB"] = str(
        args.packed_attn_uniform_reserve_mb
    )
    env["MEGAGEMM_PREFILL_TIMING"] = "1" if args.prefill_timing else "0"
    env["MEGAGEMM_PREFILL_TIMING_PRINT"] = "1" if args.prefill_timing else "0"
    decode_timing_enabled = args.decode_timing or args.decode_timing_detail
    env["MEGAGEMM_DECODE_TIMING"] = "1" if decode_timing_enabled else "0"
    env["MEGAGEMM_DECODE_TIMING_PRINT"] = "1" if decode_timing_enabled else "0"
    env["MEGAGEMM_DECODE_TIMING_DETAIL"] = "1" if args.decode_timing_detail else "0"
    env["MEGAGEMM_DECODE_SKIP_TOKEN_STORE"] = "1" if args.decode_skip_token_store else "0"
    env["MEGAGEMM_FORCE_GATE_UP_FAST"] = "1" if args.flat_fast_gate_up else "0"
    env["MEGAGEMM_FLAT_FAST_DOWN"] = "1" if args.flat_fast_down else "0"
    env["MEGAGEMM_FAST_GEMV_FORCE_TRITON"] = "1" if args.fast_gemv_force_triton else "0"
    env["MEGAGEMM_FAST_GEMV_MAX_ROWS"] = str(args.fast_gemv_max_rows)
    if args.fast_gemv_mode == "auto":
        env.pop("MEGAGEMM_FAST_GEMV_MODE", None)
    else:
        env["MEGAGEMM_FAST_GEMV_MODE"] = args.fast_gemv_mode
    env["MEGAGEMM_PREFILL_GQA_MODE"] = args.prefill_gqa_mode
    env["MEGAGEMM_NATIVE_MLP_PREFILL"] = (
        "1" if args.prefill_mlp_backend == "native" else "0"
    )
    env["MEGAGEMM_DEEPFUSION_MLP_PREFILL"] = (
        "1" if args.prefill_mlp_backend.startswith("deepfusion") else "0"
    )
    env["MEGAGEMM_DEEPFUSION_PREFILL"] = (
        "1" if args.prefill_mlp_backend.startswith("deepfusion") else "0"
    )
    env["MEGAGEMM_DEEPFUSION_PREFILL_FORCE_TRITON"] = (
        "1" if args.prefill_mlp_backend == "deepfusion-force" else "0"
    )
    env["MEGAGEMM_DEEPFUSION_PREFILL_FORCE_USE"] = (
        "1" if args.prefill_mlp_backend == "deepfusion-force" else "0"
    )
    if args.paged_decode_gqa_group > 0:
        env["MEGAGEMM_PAGED_DECODE_GQA_GROUP"] = str(args.paged_decode_gqa_group)
        env["MEGAGEMM_PAGED_DECODE_GQA2"] = (
            "1" if args.paged_decode_gqa_group == 2 else "0"
        )
    else:
        env.pop("MEGAGEMM_PAGED_DECODE_GQA_GROUP", None)
        env["MEGAGEMM_PAGED_DECODE_GQA2"] = "0"
    if args.paged_decode_warps > 0:
        env["MEGAGEMM_PAGED_DECODE_WARPS"] = str(args.paged_decode_warps)
    else:
        env.pop("MEGAGEMM_PAGED_DECODE_WARPS", None)
    if args.paged_decode_block_unroll != 0:
        env["MEGAGEMM_PAGED_DECODE_BLOCK_UNROLL"] = str(args.paged_decode_block_unroll)
    else:
        env.pop("MEGAGEMM_PAGED_DECODE_BLOCK_UNROLL", None)
    if args.paged_decode_splits > 0:
        env["MEGAGEMM_PAGED_DECODE_SPLITS"] = str(args.paged_decode_splits)
    else:
        env.pop("MEGAGEMM_PAGED_DECODE_SPLITS", None)
    if args.multi_step_burst_batch > 0:
        env["MEGAGEMM_MULTI_STEP_BURST_BATCH"] = str(args.multi_step_burst_batch)
    env["MEGAGEMM_PAGED_DECODE_LOG"] = "1" if args.paged_decode_log else "0"
    if mode.backend == "megagemm-prophet":
        env["MEGAGEMM_PROPHET_GPU_SNAPSHOT_CACHE"] = (
            "1" if args.prophet_gpu_snapshot_cache else "0"
        )
        env["MEGAGEMM_PROPHET_GPU_SNAPSHOT_CACHE_MAX_MB"] = str(
            args.prophet_gpu_snapshot_cache_max_mb
        )
    return env


def load_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    summary = paths["summary"]
    if not summary.exists():
        return []
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return payload.get("rows", [])


def write_report(args: argparse.Namespace, run_id: str, modes: list[Mode]) -> Path:
    hardware_label = safe_report_label(args.hardware_label)
    report = Path(args.out_dir) / f"{run_id}_qwen25_mgx_prophet_matrix_{hardware_label}_report.md"
    lines = [
        f"# Qwen-family MGX Prophet Matrix {args.hardware_label} Benchmark",
        "",
        "## Run Configuration",
        "",
        f"- Model/tokenizer: `{args.model}`",
        f"- MGX artifact: `{args.mgx}`",
        f"- Hardware label: `{args.hardware_label}`",
        f"- Dtype: `{args.dtype}`",
        f"- Quantize label: `{args.quantize}`",
        f"- Batch sizes: `{args.batch_sizes}`",
        f"- Prompt tokens/request: `{args.prompt_tokens}`",
        f"- Max new tokens/request: `{args.max_new_tokens}`",
        f"- Repeats: `{args.repeats}`",
        f"- Ignore EOS: `{args.ignore_eos}`",
        f"- Prophet batch exact restore: `{args.prophet_batch_exact_restore}`",
        f"- Prophet live prefix cache: `{args.prophet_live_prefix_cache}`",
        f"- Prophet resident cache: `{args.prophet_resident_cache}`",
        f"- Prophet resident cache max entries: `{args.prophet_resident_cache_max_entries}`",
        f"- Prophet GPU snapshot cache: `{args.prophet_gpu_snapshot_cache}`",
        f"- Prophet GPU snapshot cache max MB: `{args.prophet_gpu_snapshot_cache_max_mb}`",
        f"- Prefill prefer padded: `{args.prefill_prefer_padded}`",
        f"- Prefill pad waste threshold: `{args.prefill_pad_waste_threshold}`",
        f"- KV block size: `{args.block_size}`",
        f"- Packed attention backend: `{args.packed_attn_backend}`",
        f"- Packed attention uniform max score MB: `{args.packed_attn_uniform_max_score_mb}`",
        f"- Packed attention uniform reserve MB: `{args.packed_attn_uniform_reserve_mb}`",
        f"- Prefill GQA mode: `{args.prefill_gqa_mode}`",
        f"- Prefill MLP backend: `{args.prefill_mlp_backend}`",
        f"- Prefill timing: `{args.prefill_timing}`",
        f"- Decode timing: `{args.decode_timing or args.decode_timing_detail}`",
        f"- Decode timing detail: `{args.decode_timing_detail}`",
        f"- Decode skip token store: `{args.decode_skip_token_store}`",
        f"- Flat fast gate_up: `{args.flat_fast_gate_up}`",
        f"- Flat fast down: `{args.flat_fast_down}`",
        f"- Fast GEMV force Triton: `{args.fast_gemv_force_triton}`",
        f"- Fast GEMV mode: `{args.fast_gemv_mode}`",
        f"- Fast GEMV max rows: `{args.fast_gemv_max_rows}`",
        f"- Paged decode GQA group: `{args.paged_decode_gqa_group}`",
        f"- Paged decode warps override: `{args.paged_decode_warps}`",
        f"- Paged decode block unroll override: `{args.paged_decode_block_unroll}`",
        f"- Paged decode splits override: `{args.paged_decode_splits}`",
        f"- Paged decode logging: `{args.paged_decode_log}`",
        f"- Batch decode burst: `{args.multi_step_burst_batch}`",
        f"- CUDA Graph decode: `{'enabled' if args.decode_cuda_graphs else 'disabled'}`",
        f"- CUDA Graph prefer step: `{args.decode_cuda_graphs_prefer_step}`",
        f"- CUDA Graph shape cache: `{args.decode_cuda_graphs_shape_cache}`",
        f"- CUDA Graph shared shape cache: `{args.decode_cuda_graphs_shared_shape_cache}`",
        f"- CUDA Graph stable max blocks: `{args.decode_cuda_graphs_stable_max_blocks}`",
        f"- CUDA Graph min batch: `{args.decode_cuda_graphs_min_batch}`",
        f"- Run id: `{run_id}`",
        "",
        "## Summary",
        "",
        (
            "Compare `Steady output tok/s` for cache-hit baselines "
            "(MGX Prophet or vLLM prefix caching). `Median output tok/s` can "
            "include the cold miss/prime repeat."
        ),
        "",
        "| Mode | Scenario | Batch | Prompt tok/req | First output tok/s | Overall median tok/s | Cache-hit output tok/s | Cache-hit decode tok/s | Median prefill tok/s | Median decode wall tok/s | Prophet restore ms | Prophet decode ms | OK |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        rows = load_rows(output_paths(args, run_id, mode))
        if not rows:
            lines.append(f"| {mode_title(args, mode)} | no rows | | | | | | | | | |")
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
                "| {mode} | {scenario} | {batch} | {prompt} | {first:.2f} | {out:.2f} | {steady:.2f} | {steady_decode:.2f} | {prefill:.2f} | {decode:.2f} | {restore:.1f} | {prophet_decode:.1f} | {ok}/{samples} |".format(
                    mode=mode_title(args, mode),
                    scenario=row.get("scenario"),
                    batch=row.get("batch_size"),
                    prompt=row.get("prompt_tokens_requested_per_request"),
                    first=float(row.get("first_output_tps") or 0.0),
                    out=float(row.get("median_output_tps") or 0.0),
                    steady=float(row.get("median_steady_output_tps") or 0.0),
                    steady_decode=float(row.get("median_steady_prophet_decode_tps") or 0.0),
                    prefill=float(row.get("median_prefill_tps") or 0.0),
                    decode=float(row.get("median_decode_wall_tps") or 0.0),
                    restore=float(row.get("median_prophet_restore_time_ms") or 0.0),
                    prophet_decode=float(row.get("median_prophet_decode_time_ms") or 0.0),
                    ok=row.get("ok_samples"),
                    samples=row.get("samples"),
                )
            )

    lines.extend(["", "## Files", ""])
    for mode in modes:
        lines.append(f"- {mode_title(args, mode)}:")
        for label, path in output_paths(args, run_id, mode).items():
            lines.append(f"  - {label}: `{path}`")

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen 2.5 MGX Prophet matrix benchmark")
    parser.add_argument("--modes", default="core", help="core, all, or comma-separated fresh,prophet-repeat,prophet-warm")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mgx", default=DEFAULT_MGX)
    parser.add_argument("--hardware-label", default="1xt4")
    parser.add_argument("--out-dir", default="bench_results/qwen25_mgx_prophet_matrix_t4")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--prompt-tokens", default="128,512,1024,2048")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", "--runs", dest="repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quantize", default="none", choices=["none", "int8", "fp8", "awq"])
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help=(
            "Paged KV block size. For Qwen2.5-3B FP16 long-context batch=8 "
            "decode on T4, 64 cuts the attention block loop to about one "
            "quarter of the old 16-token default without increasing total KV "
            "memory in the fixed max-seq/batch setup."
        ),
    )
    parser.add_argument("--kv-alloc", default="auto", choices=["auto", "greedy"])
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--mgx-payload-cache-dir", default="")
    parser.add_argument(
        "--mgx-emit-payload-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit or prime the reusable MGX safetensors payload cache before matrix runs.",
    )
    parser.add_argument(
        "--mgx-export-mode",
        choices=["normal", "streaming"],
        default="streaming",
        help="MGX export implementation used when the artifact is missing.",
    )
    parser.add_argument("--mgx-skip-hash-check", action="store_true")
    parser.add_argument(
        "--export-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the .mgx artifact automatically if --mgx does not exist.",
    )
    parser.add_argument("--force-export", action="store_true", help="Rebuild the MGX artifact before running.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--allow-eos", dest="ignore_eos", action="store_false")
    parser.add_argument("--prophet-validation-mode", choices=["none", "full_prefill"], default="none")
    parser.add_argument("--prophet-validation-tokens", type=int, default=4)
    parser.add_argument("--prophet-fallback-to-prefill", action="store_true")
    parser.add_argument("--prophet-prefix-tokens", type=int, default=64)
    parser.add_argument("--prophet-top-k", type=int, default=3)
    parser.add_argument("--prophet-min-similarity", type=float, default=0.35)
    parser.add_argument("--prophet-min-prefix-coverage", type=float, default=0.50)
    parser.add_argument("--prophet-min-prefix-reuse-score", type=float, default=0.55)
    parser.add_argument("--prophet-max-prefix-rollback-ratio", type=float, default=0.35)
    parser.add_argument("--prophet-max-prefix-tail-ratio", type=float, default=0.50)
    parser.add_argument(
        "--prophet-batch-exact-restore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the layer-batched exact-match Prophet restore path.",
    )
    parser.add_argument(
        "--prophet-gpu-snapshot-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep Prophet snapshots hot on CUDA after writing the disk copy. "
            "This makes repeated-prompt hits closer to vLLM prefix-cache semantics "
            "by avoiding CPU->GPU KV restore copies."
        ),
    )
    parser.add_argument("--prophet-gpu-snapshot-cache-max-mb", type=int, default=2048)
    parser.add_argument(
        "--prefill-prefer-padded",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Prefer padded prefill when pad waste is under the threshold. Useful "
            "as an A/B path when packed attention falls back to per-seq SDPA."
        ),
    )
    parser.add_argument(
        "--prefill-pad-waste-threshold",
        type=float,
        default=0.03,
        help="Maximum pad waste ratio allowed when --prefill-prefer-padded is enabled.",
    )
    parser.add_argument(
        "--packed-attn-backend",
        choices=["stable", "uniform", "triton"],
        default="stable",
        help=(
            "Packed prefill attention backend. stable uses PyTorch SDPA per sequence; "
            "uniform tries one batched SDPA for uniform lengths; triton tries the "
            "custom packed kernel before SDPA fallback."
        ),
    )
    parser.add_argument(
        "--packed-attn-uniform-max-score-mb",
        type=int,
        default=4096,
        help=(
            "Soft score-memory cap for uniform packed SDPA prefill. 0 disables "
            "the score-size cap but still keeps the free-memory reserve guard."
        ),
    )
    parser.add_argument(
        "--packed-attn-uniform-reserve-mb",
        type=int,
        default=512,
        help="Free VRAM reserve kept before attempting uniform packed SDPA prefill.",
    )
    parser.add_argument(
        "--prefill-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA-event prefill stage timing logs for diagnosis.",
    )
    parser.add_argument(
        "--decode-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA-event decode stage timing logs for diagnosis.",
    )
    parser.add_argument(
        "--decode-timing-detail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Break decode timing down into flat attention/MLP sub-stages.",
    )
    parser.add_argument(
        "--decode-skip-token-store",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimental benchmark-only path: skip the GPU [batch, steps] "
            "generated-token store when decode outputs are discarded."
        ),
    )
    parser.add_argument(
        "--flat-fast-gate-up",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the fast_linear wrapper for flat decode gate_up. With the "
            "known-good T4 settings this falls back to cuBLAS for batch=8; "
            "--no-flat-fast-gate-up tests the direct torch.mm path and is "
            "known to regress the current Qwen2.5-3B FP16 T4 batch=8 target."
        ),
    )
    parser.add_argument(
        "--flat-fast-down",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic only: use fast_linear for the flat decode MLP down "
            "projection. Known to regress Qwen2.5-3B FP16 T4 batch=8."
        ),
    )
    parser.add_argument(
        "--fast-gemv-force-triton",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic only: force Triton fast_linear. Known to regress "
            "Qwen2.5-3B FP16 T4 batch=8 gate_up when combined with max_rows=8."
        ),
    )
    parser.add_argument(
        "--fast-gemv-mode",
        choices=["auto", "tile", "row", "splitk"],
        default="auto",
        help=(
            "Diagnostic only: set MEGAGEMM_FAST_GEMV_MODE. Keep auto for the "
            "known-good T4 FP16 batch=8 path."
        ),
    )
    parser.add_argument(
        "--fast-gemv-max-rows",
        type=int,
        default=4,
        help="Set MEGAGEMM_FAST_GEMV_MAX_ROWS for caller-side fast GEMV gating.",
    )
    parser.add_argument(
        "--prefill-gqa-mode",
        choices=["native", "expand"],
        default="expand",
        help=(
            "How SDPA prefill handles GQA. native uses enable_gqa=True; expand "
            "manually repeats KV heads first, which may unlock faster SDPA "
            "backends on Turing/T4."
        ),
    )
    parser.add_argument(
        "--prefill-mlp-backend",
        choices=["baseline", "deepfusion", "deepfusion-force", "native"],
        default="deepfusion-force",
        help=(
            "MLP prefill backend. baseline uses SwiGLU + cuBLAS down; deepfusion "
            "bench-gates the fused SwiGLU+down Triton path; deepfusion-force "
            "uses that path without per-layer microbench; native uses the optional "
            "rmsnorm_cuda_ops MLP prefill op when available."
        ),
    )
    parser.add_argument(
        "--prophet-live-prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep primed Prophet prefixes live and fork shared KV blocks for "
            "prefix-cache-style repeated-prompt hits."
        ),
    )
    parser.add_argument(
        "--prophet-resident-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep restored Prophet snapshots resident and fork future exact/prefix "
            "hits through MGXProphetLibrary."
        ),
    )
    parser.add_argument(
        "--prophet-resident-cache-max-entries",
        type=int,
        default=16,
        help="Maximum live Prophet resident source sequences per engine.",
    )
    parser.add_argument(
        "--paged-decode-gqa-group",
        type=int,
        default=0,
        choices=[0, 2, 4],
        help=(
            "Opt into a grouped-query paged decode kernel. 0 keeps the stable "
            "default path; 4 enables the experimental long-context Qwen/T4 path."
        ),
    )
    parser.add_argument(
        "--paged-decode-warps",
        type=int,
        default=0,
        help=(
            "Override MEGAGEMM_PAGED_DECODE_WARPS. 0 keeps the architecture-aware "
            "auto policy, which currently picks 4 warps for T4/head_dim=128."
        ),
    )
    parser.add_argument(
        "--paged-decode-block-unroll",
        type=int,
        default=0,
        choices=[-1, 0, 1, 2],
        help=(
            "Override MEGAGEMM_PAGED_DECODE_BLOCK_UNROLL. 0 keeps auto, -1/1 "
            "force no unroll, 2 forces the T4 long-context unroll-by-2 path."
        ),
    )
    parser.add_argument(
        "--paged-decode-splits",
        type=int,
        default=0,
        help="Override MEGAGEMM_PAGED_DECODE_SPLITS. 0 keeps auto.",
    )
    parser.add_argument(
        "--multi-step-burst-batch",
        type=int,
        default=128,
        help=(
            "Set MEGAGEMM_MULTI_STEP_BURST_BATCH for batched decode. The T4 "
            "Prophet suite benchmarks fixed-length --ignore-eos decode, so a "
            "long burst avoids repeated scheduler roundtrips during cached hits."
        ),
    )
    parser.add_argument(
        "--paged-decode-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print one-shot paged decode kernel path/shape logs.",
    )
    parser.add_argument(
        "--decode-cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable scheduler CUDA Graph replay for decode.",
    )
    parser.add_argument(
        "--decode-cuda-graphs-prefer-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prefer the scheduler's graph-aware decode step path when CUDA Graph decode is enabled.",
    )
    parser.add_argument(
        "--decode-cuda-graphs-shape-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse decode CUDA Graphs by batch/table shape instead of by concrete seq_ids.",
    )
    parser.add_argument(
        "--decode-cuda-graphs-shared-shape-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimentally keep shape CUDA Graphs on the BlockManager so new "
            "Schedulers can reuse them. Default off because the extra static "
            "input copies were slower on L4 in current benchmarks."
        ),
    )
    parser.add_argument(
        "--decode-cuda-graphs-stable-max-blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the allocated block-table width as the captured paged-attention "
            "loop bound so shape-cache graphs stay stable across a decode burst. "
            "The attention kernel still masks tokens beyond the real seq_len."
        ),
    )
    parser.add_argument(
        "--decode-cuda-graphs-min-batch",
        type=int,
        default=8,
        help="Minimum active decode batch eligible for CUDA Graph capture/replay.",
    )
    parser.add_argument(
        "--decode-cuda-graphs-log-limit",
        type=int,
        default=6,
        help="Maximum CUDA Graph decode diagnostic lines to print.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.mgx == DEFAULT_MGX:
        args.mgx = default_mgx_path(args.model, args.dtype, normalize_quantize(args.quantize))

    modes = parse_modes(args.modes)
    run_id = args.run_id or time.strftime("qwen25_mgx_prophet_matrix_%Y%m%d_%H%M%S")
    args.out_dir = normalize_out_dir(args.out_dir)
    warn_if_known_regressed_experiments(args)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Qwen-family MGX Prophet matrix {args.hardware_label} suite")
    print(f"  run_id: {run_id}")
    print(f"  model:  {args.model}")
    print(f"  mgx:    {args.mgx}")
    print(f"  modes:  {', '.join(mode.name for mode in modes)}")
    print(f"  out:    {args.out_dir}")
    print(
        "  prefill: "
        f"attn_backend={args.packed_attn_backend} "
        f"uniform_score_mb={args.packed_attn_uniform_max_score_mb} "
        f"reserve_mb={args.packed_attn_uniform_reserve_mb} "
        f"gqa={args.prefill_gqa_mode} "
        f"mlp={args.prefill_mlp_backend} "
        f"prefer_padded={args.prefill_prefer_padded} "
        f"timing={args.prefill_timing}"
    )
    print(
        "  decode:  "
        f"burst={args.multi_step_burst_batch} "
        f"block_size={args.block_size} "
        f"gqa_group={args.paged_decode_gqa_group} "
        f"warps={args.paged_decode_warps or 'auto'} "
        f"unroll={args.paged_decode_block_unroll or 'auto'} "
        f"splits={args.paged_decode_splits or 'auto'} "
        f"cuda_graphs={'on' if args.decode_cuda_graphs else 'off'} "
        f"cuda_graph_shape_cache={args.decode_cuda_graphs_shape_cache} "
        f"cuda_graph_shared_shape_cache={args.decode_cuda_graphs_shared_shape_cache} "
        f"cuda_graph_stable_max_blocks={args.decode_cuda_graphs_stable_max_blocks} "
        f"cuda_graph_min_batch={args.decode_cuda_graphs_min_batch} "
        f"timing={args.decode_timing or args.decode_timing_detail} "
        f"detail={args.decode_timing_detail} "
        f"skip_token_store={args.decode_skip_token_store} "
        f"flat_fast_gate_up={args.flat_fast_gate_up} "
        f"flat_fast_down={args.flat_fast_down} "
        f"fast_gemv={args.fast_gemv_mode}"
    )

    if args.dry_run:
        print("  dry-run: skipping MGX export/cache checks")
    else:
        ensure_mgx_artifact(args)

    for mode in modes:
        cmd = command(args, run_id, mode)
        print()
        print(f"=== {mode_title(args, mode)} ===")
        print(shell_join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(ROOT), env=command_env(args, mode), check=True)

    report = write_report(args, run_id, modes)
    print(f"\nWrote combined report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

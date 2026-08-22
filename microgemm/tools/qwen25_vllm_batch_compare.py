#!/usr/bin/env python3
"""Focused vLLM batch benchmark for the Qwen2.5 MicroGemm CPU suite.

This is intentionally separate from qwen25_cpu_suite.py.  It answers one
question: for the same prompt length, batch sizes, and decode length used by
the MicroGemm CPU suite, what does vLLM do?

Typical Colab usage:

    %pip install -U vllm
    !python microgemm/tools/qwen25_vllm_batch_compare.py \
        --model-repo Qwen/Qwen2.5-0.5B-Instruct \
        --batch-sizes 2,4,8 \
        --prompt-tokens 256 \
        --max-new-tokens 128 \
        --ignore-eos \
        --runs 5 \
        --warmup 1 \
        --microgemm-csv bench_results/qwen25_cpu_microgemm/qwen25_05b_cpu_xeon_v62_restore_logits_all_batches_qwen25_cpu_suite_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_BUILD_TAG = "qwen25_vllm_batch_compare_v3_explicit_cpu_device"
DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT_SEED = (
    "Explique de forma tecnica e curta como um runtime CPU executa inferencia "
    "de um modelo de linguagem usando pesos quantizados e cache KV. "
)
PROMPT_FILLER = " CPU MicroGemm Qwen dois ponto cinco benchmark."


def parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


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


def load_tokenizer(model_repo: str, *, trust_remote_code: bool, cache_dir: str = "") -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "Transformers is required for prompt shaping. Install vLLM first; "
            "it normally brings Transformers with it.\n"
            f"Import error: {exc}"
        ) from exc

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return AutoTokenizer.from_pretrained(model_repo, **kwargs)


def make_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    ids = tokenizer.encode(DEFAULT_PROMPT_SEED, add_special_tokens=False)
    filler_ids = tokenizer.encode(PROMPT_FILLER, add_special_tokens=False) or ids
    while len(ids) < target_tokens:
        ids.extend(filler_ids)
    ids = ids[:target_tokens]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    return text, int(actual)


def make_batch_prompts(tokenizer: Any, target_tokens: int, batch_size: int) -> tuple[list[str], list[int]]:
    base_prompt, _ = make_prompt(tokenizer, target_tokens)
    prompts = [f"{base_prompt}\nIndice da requisicao: {idx}." for idx in range(batch_size)]
    lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    return prompts, lengths


def sync_cuda(torch_module: Any) -> None:
    try:
        if torch_module is not None and torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
    except Exception:
        return


def device_snapshot(torch_module: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    try:
        out["vllm_version"] = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        out["vllm_version"] = ""
    if torch_module is None:
        out["torch_available"] = False
        return out
    out["torch_available"] = True
    out["torch_version"] = getattr(torch_module, "__version__", "")
    try:
        out["cuda_available"] = bool(torch_module.cuda.is_available())
        out["cuda_device_count"] = int(torch_module.cuda.device_count()) if out["cuda_available"] else 0
        devices: list[dict[str, Any]] = []
        if out["cuda_available"]:
            for idx in range(torch_module.cuda.device_count()):
                props = torch_module.cuda.get_device_properties(idx)
                devices.append(
                    {
                        "index": idx,
                        "name": torch_module.cuda.get_device_name(idx),
                        "total_memory_gb": props.total_memory / 1e9,
                        "major": props.major,
                        "minor": props.minor,
                    }
                )
        out["cuda_devices"] = devices
    except Exception as exc:
        out["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return out


def print_device_snapshot(snapshot: dict[str, Any]) -> None:
    print("Device snapshot")
    print(f"  platform:      {snapshot.get('platform', '')}")
    print(f"  python:        {snapshot.get('python', '')}")
    print(f"  vllm:          {snapshot.get('vllm_version', '') or 'not imported yet'}")
    print(f"  torch:         {snapshot.get('torch_version', '') or 'missing'}")
    if snapshot.get("cuda_available"):
        devices = snapshot.get("cuda_devices") or []
        for device in devices:
            print(
                "  cuda device:   "
                f"{device.get('index')}: {device.get('name')} "
                f"({float(device.get('total_memory_gb', 0.0)):.1f} GB)"
            )
    else:
        print("  cuda:          not available")


def import_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(f"PyTorch is required by vLLM.\nImport error: {exc}") from exc
    return torch


def import_vllm_backend() -> tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "vLLM is required for this benchmark. In Colab run:\n"
            "  %pip install -U vllm\n"
            f"Import error: {exc}"
        ) from exc
    return LLM, SamplingParams


def build_llm(args: argparse.Namespace, LLM: Any) -> Any:
    llm_kwargs: dict[str, Any] = {
        "model": args.model_repo,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max(parse_csv_ints(args.batch_sizes)),
        "trust_remote_code": bool(args.trust_remote_code),
        "enforce_eager": bool(args.enforce_eager),
        "enable_prefix_caching": bool(args.enable_prefix_caching),
        "disable_log_stats": True,
    }
    if args.device != "auto":
        llm_kwargs["device"] = args.device
    if args.download_dir:
        llm_kwargs["download_dir"] = args.download_dir
    if args.max_num_batched_tokens > 0:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    try:
        return LLM(**llm_kwargs)
    except TypeError:
        # vLLM wheels differ a lot. Retry with a conservative argument set.
        retry = dict(llm_kwargs)
        for key in (
            "disable_log_stats",
            "enable_prefix_caching",
            "trust_remote_code",
            "max_num_seqs",
            "max_num_batched_tokens",
            "download_dir",
            "device",
        ):
            retry.pop(key, None)
        return LLM(**retry)


def build_sampling_params(args: argparse.Namespace, SamplingParams: Any) -> Any:
    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_new_tokens,
        "ignore_eos": bool(args.ignore_eos),
    }
    if args.ignore_eos:
        kwargs["min_tokens"] = args.max_new_tokens
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("min_tokens", None)
        return SamplingParams(**kwargs)


def count_generated_tokens(tokenizer: Any, outputs: list[Any], fallback_tokens: int) -> int:
    total = 0
    for output in outputs:
        candidates = getattr(output, "outputs", None) or []
        if not candidates:
            continue
        first = candidates[0]
        token_ids = getattr(first, "token_ids", None)
        if token_ids is not None:
            total += len(token_ids)
            continue
        text = getattr(first, "text", "")
        if text:
            total += len(tokenizer.encode(text, add_special_tokens=False))
    return total if total > 0 else fallback_tokens


def run_vllm_once(
    llm: Any,
    sampling_params: Any,
    tokenizer: Any,
    torch_module: Any,
    prompts: list[str],
    max_new_tokens: int,
) -> dict[str, Any]:
    sync_cuda(torch_module)
    start = time.perf_counter()
    try:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    except TypeError:
        outputs = llm.generate(prompts, sampling_params)
    sync_cuda(torch_module)
    elapsed_s = time.perf_counter() - start
    generated_tokens = count_generated_tokens(tokenizer, outputs, len(prompts) * max_new_tokens)
    return {
        "ok": True,
        "elapsed_s": elapsed_s,
        "generated_tokens": generated_tokens,
        "output_tps": generated_tokens / elapsed_s if elapsed_s > 0.0 else 0.0,
    }


def aggregate_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    return {
        "ok": len(ok_rows),
        "runs": len(rows),
        "elapsed_s": stats([float(row.get("elapsed_s", 0.0) or 0.0) for row in ok_rows]),
        "generated_tokens": stats([float(row.get("generated_tokens", 0.0) or 0.0) for row in ok_rows]),
        "output_tps": stats([float(row.get("output_tps", 0.0) or 0.0) for row in ok_rows]),
    }


def load_microgemm_csv(path: str) -> dict[int, dict[str, float]]:
    if not path:
        return {}
    csv_path = Path(path)
    if not csv_path.exists():
        print(f"Warning: MicroGemm CSV not found: {csv_path}")
        return {}
    out: dict[int, dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("kind") != "batch":
                continue
            mode = row.get("effective_mode") or row.get("mode")
            if mode != "continuous":
                continue
            try:
                batch_size = int(float(row.get("batch_size", "0") or 0))
            except ValueError:
                continue
            out[batch_size] = {
                "wall_tps": float(row.get("wall_output_tps_median", "0") or 0.0),
                "runtime_tps": float(row.get("runtime_output_tps_median", "0") or 0.0),
            }
    return out


def compare_summary_rows(
    summary_rows: list[dict[str, Any]],
    microgemm: dict[int, dict[str, float]],
) -> None:
    if not microgemm:
        return
    print("Comparison vs MicroGemm continuous")
    print("  batch  vllm_wall  micro_wall  micro_runtime  vllm/micro_wall")
    for row in summary_rows:
        batch_size = int(row["batch_size"])
        mg = microgemm.get(batch_size)
        if not mg:
            continue
        vllm_tps = float(row.get("output_tps_median", 0.0) or 0.0)
        micro_wall = float(mg.get("wall_tps", 0.0) or 0.0)
        micro_runtime = float(mg.get("runtime_tps", 0.0) or 0.0)
        ratio = vllm_tps / micro_wall if micro_wall > 0.0 else 0.0
        print(
            f"  {batch_size:>5}  "
            f"{vllm_tps:>9.2f}  "
            f"{micro_wall:>10.2f}  "
            f"{micro_runtime:>13.2f}  "
            f"{ratio:>15.2f}x"
        )


def write_outputs(
    args: argparse.Namespace,
    snapshot: dict[str, Any],
    results: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"qwen25_vllm_batch_compare_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{run_id}_qwen25_vllm_batch_compare.json"
    csv_path = out_dir / f"{run_id}_qwen25_vllm_batch_compare_summary.csv"
    payload = {
        "benchmark": "qwen25_vllm_batch_compare",
        "script_build": SCRIPT_BUILD_TAG,
        "run_id": run_id,
        "device_snapshot": snapshot,
        "config": {
            "model_repo": args.model_repo,
            "batch_sizes": parse_csv_ints(args.batch_sizes),
            "prompt_tokens": args.prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": bool(args.ignore_eos),
            "runs": args.runs,
            "warmup": args.warmup,
            "dtype": args.dtype,
            "device": args.device,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": bool(args.enable_prefix_caching),
            "enforce_eager": bool(args.enforce_eager),
            "microgemm_csv": args.microgemm_csv,
        },
        "results": results,
        "summary_rows": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if summary_rows:
        fieldnames: list[str] = []
        for row in summary_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen2.5 vLLM batch 2/4/8 comparison harness")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out-dir", default="bench_results/qwen25_vllm_compare")
    parser.add_argument("--run-id", default="qwen25_05b_vllm_batch248")
    parser.add_argument("--microgemm-csv", default="")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help=(
            "vLLM target device. Use --device cpu only after installing the "
            "vLLM CPU wheel/build; the normal PyPI CUDA wheel will not become "
            "a CPU backend just because this flag is set."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--download-dir", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help=(
            "Try to continue even when torch.cuda is unavailable. This is mostly "
            "for debugging nonstandard vLLM CPU builds; normal Colab vLLM "
            "comparisons should use a GPU runtime."
        ),
    )
    args = parser.parse_args()

    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.max_model_len <= args.prompt_tokens + args.max_new_tokens:
        args.max_model_len = args.prompt_tokens + args.max_new_tokens + 64

    torch_module = import_torch()
    snapshot = device_snapshot(torch_module)

    print("Qwen2.5 vLLM batch compare")
    print(f"  script build:  {SCRIPT_BUILD_TAG}")
    print(f"  model:         {args.model_repo}")
    print(f"  batch sizes:   {args.batch_sizes}")
    print(f"  prompt tokens: {args.prompt_tokens}")
    print(f"  max new tok:   {args.max_new_tokens}")
    print(f"  ignore eos:    {bool(args.ignore_eos)}")
    print(f"  dtype:         {args.dtype}")
    print(f"  device:        {args.device}")
    print(f"  prefix cache:  {bool(args.enable_prefix_caching)}")
    print_device_snapshot(snapshot)
    if args.device == "cpu":
        args.allow_no_cuda = True
    if args.device == "cuda" and not snapshot.get("cuda_available"):
        print("Error: --device cuda was requested, but CUDA is not available.")
        return 2
    if not snapshot.get("cuda_available") and not args.allow_no_cuda:
        print(
            "Error: CUDA is not available, so this vLLM benchmark cannot produce "
            "a GPU-vLLM comparison."
        )
        print("Colab fix: Runtime > Change runtime type > GPU, restart the runtime, then reinstall vLLM.")
        print("Quick check before rerunning: nvidia-smi && python -c \"import torch; print(torch.cuda.is_available())\"")
        print("CPU-only path: install the vLLM CPU wheel/build, then rerun this script with --device cpu.")
        return 2
    if not snapshot.get("cuda_available"):
        print("Warning: continuing without CUDA because --allow-no-cuda was passed.")

    LLM, SamplingParams = import_vllm_backend()

    tokenizer = load_tokenizer(
        args.model_repo,
        trust_remote_code=bool(args.trust_remote_code),
        cache_dir=args.download_dir,
    )
    sampling_params = build_sampling_params(args, SamplingParams)

    print("Loading vLLM engine")
    t0 = time.perf_counter()
    llm = build_llm(args, LLM)
    print(f"  loaded in:     {time.perf_counter() - t0:.2f}s")

    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    microgemm_rows = load_microgemm_csv(args.microgemm_csv)

    for batch_size in parse_csv_ints(args.batch_sizes):
        prompts, prompt_lengths = make_batch_prompts(tokenizer, args.prompt_tokens, batch_size)

        for warm_idx in range(args.warmup):
            try:
                run_vllm_once(
                    llm,
                    sampling_params,
                    tokenizer,
                    torch_module,
                    prompts,
                    args.max_new_tokens,
                )
            except Exception as exc:
                print(f"warmup batch={batch_size} failed: {type(exc).__name__}: {exc}")

        run_rows: list[dict[str, Any]] = []
        for run_idx in range(args.runs):
            row: dict[str, Any]
            try:
                row = run_vllm_once(
                    llm,
                    sampling_params,
                    tokenizer,
                    torch_module,
                    prompts,
                    args.max_new_tokens,
                )
            except Exception as exc:
                row = {
                    "ok": False,
                    "elapsed_s": 0.0,
                    "generated_tokens": 0,
                    "output_tps": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            row.update(
                {
                    "run_index": run_idx + 1,
                    "batch_size": batch_size,
                    "target_prompt_tokens": args.prompt_tokens,
                    "prompt_tokens_min": min(prompt_lengths),
                    "prompt_tokens_median": statistics.median(prompt_lengths),
                    "prompt_tokens_max": max(prompt_lengths),
                    "max_new_tokens": args.max_new_tokens,
                }
            )
            run_rows.append(row)

        summary = aggregate_batch(run_rows)
        output_tps_med = summary["output_tps"]["median"]
        elapsed_med = summary["elapsed_s"]["median"]
        generated_med = summary["generated_tokens"]["median"]
        summary_row: dict[str, Any] = {
            "backend": "vllm",
            "script_build": SCRIPT_BUILD_TAG,
            "model": args.model_repo,
            "batch_size": batch_size,
            "target_prompt_tokens": args.prompt_tokens,
            "prompt_tokens_min": min(prompt_lengths),
            "prompt_tokens_median": statistics.median(prompt_lengths),
            "prompt_tokens_max": max(prompt_lengths),
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": bool(args.ignore_eos),
            "ok": summary["ok"],
            "runs": summary["runs"],
            "output_tps_median": output_tps_med,
            "output_tps_min": summary["output_tps"]["min"],
            "output_tps_mean": summary["output_tps"]["mean"],
            "output_tps_p95": summary["output_tps"]["p95"],
            "output_tps_max": summary["output_tps"]["max"],
            "elapsed_s_median": elapsed_med,
            "generated_tokens_median": generated_med,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": bool(args.enable_prefix_caching),
            "enforce_eager": bool(args.enforce_eager),
            "cuda_available": bool(snapshot.get("cuda_available")),
            "cuda_device": (
                (snapshot.get("cuda_devices") or [{}])[0].get("name", "")
                if snapshot.get("cuda_devices")
                else ""
            ),
            "vllm_version": snapshot.get("vllm_version", ""),
        }
        if batch_size in microgemm_rows:
            mg = microgemm_rows[batch_size]
            micro_wall = float(mg.get("wall_tps", 0.0) or 0.0)
            micro_runtime = float(mg.get("runtime_tps", 0.0) or 0.0)
            summary_row["microgemm_wall_tps_median"] = micro_wall
            summary_row["microgemm_runtime_tps_median"] = micro_runtime
            summary_row["vllm_over_microgemm_wall"] = output_tps_med / micro_wall if micro_wall > 0 else 0.0
            summary_row["vllm_over_microgemm_runtime"] = (
                output_tps_med / micro_runtime if micro_runtime > 0 else 0.0
            )

        summary_rows.append(summary_row)
        results.append({"summary": summary_row, "runs": run_rows})

        compare_bits = ""
        if batch_size in microgemm_rows:
            micro_wall = float(microgemm_rows[batch_size].get("wall_tps", 0.0) or 0.0)
            ratio = output_tps_med / micro_wall if micro_wall > 0.0 else 0.0
            compare_bits = f" micro_wall={micro_wall:.2f} ratio={ratio:.2f}x"
        print(
            f"vllm batch={batch_size} ok={summary['ok']}/{summary['runs']} "
            f"wall_med={output_tps_med:.2f} tok/s elapsed_med={elapsed_med:.3f}s "
            f"gen_tokens_med={generated_med:.0f}{compare_bits}",
            flush=True,
        )

    compare_summary_rows(summary_rows, microgemm_rows)
    json_path, csv_path = write_outputs(args, snapshot, results, summary_rows)
    print("Wrote:")
    print(f"  json: {json_path}")
    print(f"  csv:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

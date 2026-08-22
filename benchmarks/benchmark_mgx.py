"""
Colab-friendly benchmark for MegaGemm HF/local loading versus compiled MGX.

Measures:
  - engine init / model load time
  - first request latency (TTFT-ish end-to-end)
  - warmed decode throughput

This harness runs the HF/local path and the MGX path in isolated subprocesses,
which keeps GPU memory accounting cleaner and avoids cross-run contamination.

Examples:
    python benchmarks/benchmark_mgx.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --export-if-missing

    python benchmarks/benchmark_mgx.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --dtype bf16 \
        --quantize int8 \
        --export-if-missing \
        --json-out benchmark_mgx_qwen15b.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from megagemm.engine import InferenceEngine
from megagemm.models import export_to_mgx, inspect_mgx, prime_mgx_payload_cache

_WORKER_SENTINEL = "__MGX_BENCH_JSON__"


def _slugify_model_ref(model_ref: str) -> str:
    slug = model_ref.strip().replace("\\", "/").strip("/")
    if not slug:
        slug = "model"
    slug = slug.replace("/", "--")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug)
    return slug[:120]


def _default_mgx_path(
    model_ref: str,
    dtype_name: str,
    quantize_name: str,
    sparsity_name: str,
) -> Path:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = quantize_name if quantize_name != "none" else dtype_name
    if sparsity_name == "2:4":
        suffix += "-sparse24"
    return (out_dir / f"{_slugify_model_ref(model_ref)}-{suffix}.mgx").resolve()


def _dtype_from_name(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def _normalize_quantize_name(name: str) -> str:
    if name == "awq":
        return "int4"
    if name == "w4a16":
        return "native-int4"
    return name


def _cleanup_device(device: str) -> None:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _cuda_memory_snapshot() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "cuda_max_reserved_mb": torch.cuda.max_memory_reserved() / (1024 ** 2),
    }


def _cuda_memory_current_snapshot() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "cuda_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
    }


def _environment_metadata(device: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
    }
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        meta.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_total_memory_gb": props.total_memory / 1e9,
                "cuda_capability": f"{props.major}.{props.minor}",
            }
        )
    return meta


def _maybe_format_chat_prompt(tokenizer: Any, prompt: str) -> str:
    formatted_prompt = prompt
    bos = getattr(tokenizer, "bos_token", None)
    already_formatted = bool(bos and prompt.startswith(bos))

    if (
        not already_formatted
        and hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template
    ):
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted_prompt = prompt
    return formatted_prompt


def _measure_engine_once(
    model_ref: str,
    *,
    device: str,
    dtype: torch.dtype,
    quantize: str | None,
    prompt: str,
    first_tokens: int,
    warm_tokens: int,
    warm_runs: int,
    max_seq_len: int,
    num_blocks: int,
    block_size: int,
    max_batch_size: int,
    batch_size: int,
    kv_alloc: str,
    mgx_verify_payload: bool | None,
    mgx_prefer_payload_cache: bool,
    mgx_payload_cache_dir: str | None,
) -> dict[str, Any]:
    _cleanup_device(device)
    batch_size = max(1, int(batch_size))
    runtime_max_batch_size = max(int(max_batch_size), batch_size)
    benchmark_prompts = [prompt] * batch_size

    def _run_generation(target: InferenceEngine, token_count: int) -> None:
        if batch_size == 1:
            target.generate(
                prompt,
                max_new_tokens=token_count,
                temperature=0.0,
                repetition_penalty=1.0,
            )
            return
        target.generate_batch(
            benchmark_prompts,
            max_new_tokens=token_count,
            temperature=0.0,
            ignore_eos=True,
            verbose=False,
            decode_outputs=False,
            materialize_generated_tokens=False,
        )

    t0 = time.perf_counter()
    engine = InferenceEngine(
        model_ref,
        dtype=dtype,
        device=device,
        quantize=quantize,
        max_seq_len=max_seq_len,
        num_blocks=num_blocks,
        block_size=block_size,
        max_batch_size=runtime_max_batch_size,
        kv_alloc=kv_alloc,
        mgx_verify_payload=mgx_verify_payload,
        mgx_prefer_payload_cache=mgx_prefer_payload_cache,
        mgx_payload_cache_dir=mgx_payload_cache_dir,
    )
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    init_timing = engine.get_init_timing()
    load_peak = _cuda_memory_snapshot()
    load_current = _cuda_memory_current_snapshot()

    _, prompt_input_ids = engine._prepare_prompt_inputs(prompt)
    prompt_tokens_per_request = int(prompt_input_ids.shape[1])
    prompt_token_count = prompt_tokens_per_request * batch_size

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    first_start = time.perf_counter()
    _run_generation(engine, first_tokens)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    first_end = time.perf_counter()
    first_peak = _cuda_memory_snapshot()
    first_current = _cuda_memory_current_snapshot()
    cold_first_request_seconds = first_end - first_start

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    warm_latencies = []
    for _ in range(max(1, warm_runs)):
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        run_start = time.perf_counter()
        _run_generation(engine, warm_tokens)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        run_end = time.perf_counter()
        warm_latencies.append(run_end - run_start)

    avg_warm_latency = sum(warm_latencies) / len(warm_latencies)
    generated_tokens_per_run = warm_tokens * batch_size
    warmed_tps = generated_tokens_per_run / avg_warm_latency if avg_warm_latency > 0 else 0.0
    warm_peak = _cuda_memory_snapshot()
    warm_current = _cuda_memory_current_snapshot()
    overall_peak_allocated = max(
        load_peak.get("cuda_max_allocated_mb", 0.0),
        first_peak.get("cuda_max_allocated_mb", 0.0),
        warm_peak.get("cuda_max_allocated_mb", 0.0),
    )
    overall_peak_reserved = max(
        load_peak.get("cuda_max_reserved_mb", 0.0),
        first_peak.get("cuda_max_reserved_mb", 0.0),
        warm_peak.get("cuda_max_reserved_mb", 0.0),
    )
    runtime_model = getattr(engine, "model", None)
    decode_runtime = (
        runtime_model.decode_runtime_stats()
        if runtime_model is not None and hasattr(runtime_model, "decode_runtime_stats")
        else {}
    )
    try:
        from megagemm.quantization.native_w4a16 import native_w4a16_runtime_stats

        native_w4a16_runtime = native_w4a16_runtime_stats(runtime_model)
    except Exception:
        native_w4a16_runtime = {}
    # Do not keep the model alive through this diagnostic alias.  When
    # first_tokens > 1 the TTFT measurement intentionally destroys `engine`
    # and constructs a second cold engine below.  Retaining runtime_model here
    # otherwise keeps the complete first checkpoint on CUDA, which attempts to
    # hold two models simultaneously and OOMs for FP16 7B-class artifacts.
    runtime_model = None

    cold_first_token_seconds = None
    if first_tokens == 1:
        cold_first_token_seconds = cold_first_request_seconds
    else:
        del engine
        _cleanup_device(device)
        cold_engine = InferenceEngine(
            model_ref,
            dtype=dtype,
            device=device,
            quantize=quantize,
            max_seq_len=max_seq_len,
            num_blocks=num_blocks,
            block_size=block_size,
            max_batch_size=runtime_max_batch_size,
            kv_alloc=kv_alloc,
            mgx_verify_payload=mgx_verify_payload,
            mgx_prefer_payload_cache=mgx_prefer_payload_cache,
            mgx_payload_cache_dir=mgx_payload_cache_dir,
        )
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_start = time.perf_counter()
        _run_generation(cold_engine, 1)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        cold_first_token_seconds = time.perf_counter() - ttft_start
        del cold_engine
        _cleanup_device(device)
        engine = None

    result = {
        "model_ref": model_ref,
        "native_w4a16_runtime": native_w4a16_runtime,
        "load_seconds": t1 - t0,
        "first_request_seconds": cold_first_request_seconds,
        "cold_first_request_seconds": cold_first_request_seconds,
        "cold_first_token_seconds": cold_first_token_seconds,
        "warmed_decode_tokens_per_second": warmed_tps,
        "warmed_decode_latency_seconds": avg_warm_latency,
        "warm_runs": int(max(1, warm_runs)),
        "batch_size": batch_size,
        "prompt_token_count": int(prompt_token_count),
        "prompt_tokens_per_request": prompt_tokens_per_request,
        "generated_tokens_per_warm_run": generated_tokens_per_run,
        "first_tokens": int(first_tokens),
        "warm_tokens": int(warm_tokens),
        "dtype": str(dtype).replace("torch.", ""),
        "quantize": quantize or "none",
        "engine_init_timing": init_timing,
        "model_loader_timing": init_timing.get("model_loader_timing"),
        "decode_runtime": decode_runtime,
        "engine_reported_init_seconds": init_timing.get("total_seconds"),
        "cuda_peak_allocated_after_load_mb": load_peak.get("cuda_max_allocated_mb"),
        "cuda_peak_reserved_after_load_mb": load_peak.get("cuda_max_reserved_mb"),
        "cuda_allocated_after_load_mb": load_current.get("cuda_allocated_mb"),
        "cuda_reserved_after_load_mb": load_current.get("cuda_reserved_mb"),
        "cuda_peak_allocated_during_first_request_mb": first_peak.get("cuda_max_allocated_mb"),
        "cuda_peak_reserved_during_first_request_mb": first_peak.get("cuda_max_reserved_mb"),
        "cuda_allocated_after_first_request_mb": first_current.get("cuda_allocated_mb"),
        "cuda_reserved_after_first_request_mb": first_current.get("cuda_reserved_mb"),
        "cuda_peak_allocated_during_warm_decode_mb": warm_peak.get("cuda_max_allocated_mb"),
        "cuda_peak_reserved_during_warm_decode_mb": warm_peak.get("cuda_max_reserved_mb"),
        "cuda_allocated_after_warm_decode_mb": warm_current.get("cuda_allocated_mb"),
        "cuda_reserved_after_warm_decode_mb": warm_current.get("cuda_reserved_mb"),
        "cuda_max_allocated_mb": overall_peak_allocated,
        "cuda_max_reserved_mb": overall_peak_reserved,
    }

    del engine
    _cleanup_device(device)
    return result


def _measure_transformers_once(
    model_ref: str,
    *,
    device: str,
    dtype: torch.dtype,
    prompt: str,
    first_tokens: int,
    warm_tokens: int,
    warm_runs: int,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _cleanup_device(device)

    tokenizer_t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    tokenizer_t1 = time.perf_counter()
    formatted_prompt = _maybe_format_chat_prompt(tokenizer, prompt)

    model_t0 = time.perf_counter()
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if device == "cuda":
        model_kwargs["device_map"] = "cuda"
    hf_model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    if device != "cuda":
        hf_model.to(device)
    hf_model.eval()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    model_t1 = time.perf_counter()

    load_peak = _cuda_memory_snapshot()
    load_current = _cuda_memory_current_snapshot()

    prompt_token_count = len(tokenizer.encode(formatted_prompt, add_special_tokens=False))

    def _run_generate(model, max_new_tokens: int) -> None:
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    first_t0 = time.perf_counter()
    _run_generate(hf_model, first_tokens)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    first_t1 = time.perf_counter()
    first_peak = _cuda_memory_snapshot()
    first_current = _cuda_memory_current_snapshot()
    cold_first_request_seconds = first_t1 - first_t0

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    warm_latencies = []
    for _ in range(max(1, warm_runs)):
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        run_t0 = time.perf_counter()
        _run_generate(hf_model, warm_tokens)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        run_t1 = time.perf_counter()
        warm_latencies.append(run_t1 - run_t0)

    avg_warm_latency = sum(warm_latencies) / len(warm_latencies)
    warmed_tps = warm_tokens / avg_warm_latency if avg_warm_latency > 0 else 0.0
    warm_peak = _cuda_memory_snapshot()
    warm_current = _cuda_memory_current_snapshot()
    overall_peak_allocated = max(
        load_peak.get("cuda_max_allocated_mb", 0.0),
        first_peak.get("cuda_max_allocated_mb", 0.0),
        warm_peak.get("cuda_max_allocated_mb", 0.0),
    )
    overall_peak_reserved = max(
        load_peak.get("cuda_max_reserved_mb", 0.0),
        first_peak.get("cuda_max_reserved_mb", 0.0),
        warm_peak.get("cuda_max_reserved_mb", 0.0),
    )

    model_load_seconds = model_t1 - model_t0
    tokenizer_load_seconds = tokenizer_t1 - tokenizer_t0
    total_seconds = model_t1 - tokenizer_t0
    loader_timing = {
        "loader_kind": "transformers",
        "requested_model": model_ref,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "quantize": "none",
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "model_load_seconds": model_load_seconds,
        "total_seconds": total_seconds,
    }
    init_timing = {
        "model_ref": model_ref,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "quantize": "none",
        "model_loader_kind": "transformers",
        "model_load_seconds": model_load_seconds,
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "model_loader_timing": loader_timing,
        "total_seconds": total_seconds,
    }

    cold_first_token_seconds = None
    if first_tokens == 1:
        cold_first_token_seconds = cold_first_request_seconds
    else:
        del hf_model
        _cleanup_device(device)
        cold_model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
        if device != "cuda":
            cold_model.to(device)
        cold_model.eval()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_t0 = time.perf_counter()
        _run_generate(cold_model, 1)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        cold_first_token_seconds = time.perf_counter() - ttft_t0
        del cold_model
        _cleanup_device(device)
        hf_model = None

    result = {
        "model_ref": model_ref,
        "load_seconds": total_seconds,
        "first_request_seconds": cold_first_request_seconds,
        "cold_first_request_seconds": cold_first_request_seconds,
        "cold_first_token_seconds": cold_first_token_seconds,
        "warmed_decode_tokens_per_second": warmed_tps,
        "warmed_decode_latency_seconds": avg_warm_latency,
        "warm_runs": int(max(1, warm_runs)),
        "prompt_token_count": int(prompt_token_count),
        "first_tokens": int(first_tokens),
        "warm_tokens": int(warm_tokens),
        "dtype": str(dtype).replace("torch.", ""),
        "quantize": "none",
        "engine_init_timing": init_timing,
        "model_loader_timing": loader_timing,
        "engine_reported_init_seconds": total_seconds,
        "cuda_peak_allocated_after_load_mb": load_peak.get("cuda_max_allocated_mb"),
        "cuda_peak_reserved_after_load_mb": load_peak.get("cuda_max_reserved_mb"),
        "cuda_allocated_after_load_mb": load_current.get("cuda_allocated_mb"),
        "cuda_reserved_after_load_mb": load_current.get("cuda_reserved_mb"),
        "cuda_peak_allocated_during_first_request_mb": first_peak.get("cuda_max_allocated_mb"),
        "cuda_peak_reserved_during_first_request_mb": first_peak.get("cuda_max_reserved_mb"),
        "cuda_allocated_after_first_request_mb": first_current.get("cuda_allocated_mb"),
        "cuda_reserved_after_first_request_mb": first_current.get("cuda_reserved_mb"),
        "cuda_peak_allocated_during_warm_decode_mb": warm_peak.get("cuda_max_allocated_mb"),
        "cuda_peak_reserved_during_warm_decode_mb": warm_peak.get("cuda_max_reserved_mb"),
        "cuda_allocated_after_warm_decode_mb": warm_current.get("cuda_allocated_mb"),
        "cuda_reserved_after_warm_decode_mb": warm_current.get("cuda_reserved_mb"),
        "cuda_max_allocated_mb": overall_peak_allocated,
        "cuda_max_reserved_mb": overall_peak_reserved,
    }

    del hf_model
    del tokenizer
    _cleanup_device(device)
    return result


def _percent_delta(reference: float, candidate: float, *, higher_is_better: bool) -> float | None:
    if reference == 0:
        return None
    if higher_is_better:
        return ((candidate - reference) / reference) * 100.0
    return ((reference - candidate) / reference) * 100.0


def _build_summary(hf_stats: dict[str, Any], mgx_stats: dict[str, Any]) -> dict[str, Any]:
    cold_first_token_speedup_pct = None
    if hf_stats.get("cold_first_token_seconds") is not None and mgx_stats.get("cold_first_token_seconds") is not None:
        cold_first_token_speedup_pct = _percent_delta(
            hf_stats["cold_first_token_seconds"],
            mgx_stats["cold_first_token_seconds"],
            higher_is_better=False,
        )
    load_speedup_pct = _percent_delta(
        hf_stats["load_seconds"],
        mgx_stats["load_seconds"],
        higher_is_better=False,
    )
    first_request_speedup_pct = _percent_delta(
        hf_stats["first_request_seconds"],
        mgx_stats["first_request_seconds"],
        higher_is_better=False,
    )
    warmed_tps_speedup_pct = _percent_delta(
        hf_stats["warmed_decode_tokens_per_second"],
        mgx_stats["warmed_decode_tokens_per_second"],
        higher_is_better=True,
    )
    return {
        "load_delta_seconds": hf_stats["load_seconds"] - mgx_stats["load_seconds"],
        "cold_first_token_delta_seconds": (
            hf_stats["cold_first_token_seconds"] - mgx_stats["cold_first_token_seconds"]
            if hf_stats.get("cold_first_token_seconds") is not None and mgx_stats.get("cold_first_token_seconds") is not None
            else None
        ),
        "first_request_delta_seconds": hf_stats["first_request_seconds"] - mgx_stats["first_request_seconds"],
        "cold_first_request_delta_seconds": (
            hf_stats["cold_first_request_seconds"] - mgx_stats["cold_first_request_seconds"]
            if hf_stats.get("cold_first_request_seconds") is not None and mgx_stats.get("cold_first_request_seconds") is not None
            else None
        ),
        "warmed_decode_tps_delta": mgx_stats["warmed_decode_tokens_per_second"] - hf_stats["warmed_decode_tokens_per_second"],
        "load_speedup_pct": load_speedup_pct,
        "cold_first_token_speedup_pct": cold_first_token_speedup_pct,
        "first_request_speedup_pct": first_request_speedup_pct,
        "cold_first_request_speedup_pct": (
            _percent_delta(
                hf_stats["cold_first_request_seconds"],
                mgx_stats["cold_first_request_seconds"],
                higher_is_better=False,
            )
            if hf_stats.get("cold_first_request_seconds") is not None and mgx_stats.get("cold_first_request_seconds") is not None
            else None
        ),
        "warmed_decode_tps_speedup_pct": warmed_tps_speedup_pct,
    }


def _mb_delta(hf_stats: dict[str, Any], mgx_stats: dict[str, Any], key: str) -> float | None:
    hf_value = hf_stats.get(key)
    mgx_value = mgx_stats.get(key)
    if hf_value is None or mgx_value is None:
        return None
    return float(hf_value) - float(mgx_value)


def _build_memory_summary(hf_stats: dict[str, Any], mgx_stats: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "load_peak_allocated_delta_mb": "cuda_peak_allocated_after_load_mb",
        "load_current_allocated_delta_mb": "cuda_allocated_after_load_mb",
        "load_peak_reserved_delta_mb": "cuda_peak_reserved_after_load_mb",
        "load_current_reserved_delta_mb": "cuda_reserved_after_load_mb",
        "first_request_peak_allocated_delta_mb": "cuda_peak_allocated_during_first_request_mb",
        "first_request_current_allocated_delta_mb": "cuda_allocated_after_first_request_mb",
        "first_request_peak_reserved_delta_mb": "cuda_peak_reserved_during_first_request_mb",
        "first_request_current_reserved_delta_mb": "cuda_reserved_after_first_request_mb",
        "warm_decode_peak_allocated_delta_mb": "cuda_peak_allocated_during_warm_decode_mb",
        "warm_decode_current_allocated_delta_mb": "cuda_allocated_after_warm_decode_mb",
        "warm_decode_peak_reserved_delta_mb": "cuda_peak_reserved_during_warm_decode_mb",
        "warm_decode_current_reserved_delta_mb": "cuda_reserved_after_warm_decode_mb",
        "overall_peak_allocated_delta_mb": "cuda_max_allocated_mb",
        "overall_peak_reserved_delta_mb": "cuda_max_reserved_mb",
    }
    return {
        summary_key: _mb_delta(hf_stats, mgx_stats, stat_key)
        for summary_key, stat_key in keys.items()
    }


def _run_worker(
    *,
    label: str,
    model_ref: str,
    device: str,
    dtype_name: str,
    quantize_name: str,
    prompt: str,
    first_tokens: int,
    warm_tokens: int,
    warm_runs: int,
    max_seq_len: int,
    num_blocks: int,
    block_size: int,
    max_batch_size: int,
    batch_size: int,
    kv_alloc: str,
    baseline_kind: str,
    mgx_verify_payload: bool | None,
    mgx_prefer_payload_cache: bool,
    mgx_payload_cache_dir: str | None,
    mgx_sparse24_runtime: str = "auto",
    mgx_sparse24_kernel: str = "auto",
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(script_path),
        "--_worker",
        "--worker-label",
        label,
        "--worker-model-ref",
        model_ref,
        "--device",
        device,
        "--dtype",
        dtype_name,
        "--quantize",
        quantize_name,
        "--prompt",
        prompt,
        "--first-tokens",
        str(first_tokens),
        "--warm-tokens",
        str(warm_tokens),
        "--warm-runs",
        str(warm_runs),
        "--max-seq-len",
        str(max_seq_len),
        "--num-blocks",
        str(num_blocks),
        "--block-size",
        str(block_size),
        "--max-batch-size",
        str(max_batch_size),
        "--batch-size",
        str(batch_size),
        "--kv-alloc",
        kv_alloc,
        "--worker-baseline-kind",
        baseline_kind,
        "--mgx-verify-payload",
        "default" if mgx_verify_payload is None else ("true" if mgx_verify_payload else "false"),
    ]
    if mgx_prefer_payload_cache:
        cmd.append("--mgx-prefer-payload-cache")
    if mgx_payload_cache_dir:
        cmd.extend(["--mgx-payload-cache-dir", mgx_payload_cache_dir])
    worker_env = os.environ.copy()
    if mgx_sparse24_runtime != "auto":
        worker_env["MEGAGEMM_MGX_SPARSE24_RUNTIME"] = (
            "1" if mgx_sparse24_runtime == "on" else "0"
        )
    if mgx_sparse24_kernel != "auto":
        worker_env["MEGAGEMM_MGX_SPARSE24_KERNEL"] = mgx_sparse24_kernel
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
        env=worker_env,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"Worker benchmark failed for {label}.\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(_WORKER_SENTINEL):
            payload = json.loads(line[len(_WORKER_SENTINEL):])
    if payload is None:
        raise SystemExit(
            f"Worker benchmark for {label} did not return structured JSON.\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    return payload


def _worker_main(args: argparse.Namespace) -> int:
    quantize = None if args.quantize == "none" else args.quantize
    dtype = _dtype_from_name(args.dtype)
    if args.worker_baseline_kind == "transformers":
        stats = _measure_transformers_once(
            args.worker_model_ref,
            device=args.device,
            dtype=dtype,
            prompt=args.prompt,
            first_tokens=args.first_tokens,
            warm_tokens=args.warm_tokens,
            warm_runs=args.warm_runs,
        )
    else:
        stats = _measure_engine_once(
            args.worker_model_ref,
            device=args.device,
            dtype=dtype,
            quantize=quantize,
            prompt=args.prompt,
            first_tokens=args.first_tokens,
            warm_tokens=args.warm_tokens,
            warm_runs=args.warm_runs,
            max_seq_len=args.max_seq_len,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
            max_batch_size=args.max_batch_size,
            batch_size=args.batch_size,
            kv_alloc=args.kv_alloc,
            mgx_verify_payload=None if args.mgx_verify_payload == "default" else (args.mgx_verify_payload == "true"),
            mgx_prefer_payload_cache=bool(args.mgx_prefer_payload_cache),
            mgx_payload_cache_dir=args.mgx_payload_cache_dir,
        )
    stats["label"] = args.worker_label
    print(_WORKER_SENTINEL + json.dumps(stats, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a baseline path versus compiled MGX.")
    parser.add_argument("--model", help="HF model id or local snapshot directory")
    parser.add_argument("--mgx", help="Path to the compiled .mgx artifact")
    parser.add_argument(
        "--baseline",
        choices=["megagemm-local", "transformers"],
        default="megagemm-local",
        help=(
            "Baseline implementation to compare against MGX. "
            "'megagemm-local' loads from the HF/local snapshot through MegaGemm. "
            "'transformers' uses Hugging Face Transformers directly."
        ),
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--quantize", choices=["none", "int8", "int4", "awq", "native-int4", "w4a16"], default="none")
    parser.add_argument(
        "--sparsity",
        choices=["auto", "none", "2:4"],
        default="auto",
        help="Expected/exported MGX sparsity. 'auto' accepts an existing artifact and exports dense if missing.",
    )
    parser.add_argument(
        "--mgx-sparse24-runtime",
        choices=["auto", "on", "off"],
        default="auto",
        help="Enable or disable the 2:4 CUDA runtime in the MGX worker for A/B measurements.",
    )
    parser.add_argument(
        "--mgx-sparse24-kernel",
        choices=["auto", "native", "triton", "torch"],
        default="auto",
        help=(
            "2:4 kernel: native forces MGX's standalone FP16 mma.sp kernel; "
            "auto correctness-benchmarks native, compact Triton and PyTorch."
        ),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--prompt", default="Explain why compiled model artifacts reduce cold-start latency.")
    parser.add_argument("--first-tokens", type=int, default=32, help="Tokens for the first-request timing.")
    parser.add_argument("--warm-tokens", type=int, default=64, help="Tokens for the warmed throughput runs.")
    parser.add_argument("--warm-runs", type=int, default=3, help="Number of warmed decode runs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of concurrent prompts actually submitted to generate_batch (default: 1).",
    )
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-blocks", type=int, default=0, help="Explicit GPU KV blocks (0 keeps engine auto sizing).")
    parser.add_argument("--block-size", type=int, default=16, help="Tokens per KV cache block.")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help=(
            "Engine max batch size used by KV auto sizing. "
            "It is automatically raised to --batch-size when necessary."
        ),
    )
    parser.add_argument("--kv-alloc", choices=["auto", "greedy"], default="auto", help="KV cache allocation strategy when --num-blocks=0.")
    parser.add_argument("--mgx-skip-hash-check", action="store_true", help="Skip embedded MGX payload hash verification during load.")
    parser.add_argument("--mgx-prefer-payload-cache", action="store_true", help="Prefer a reusable extracted safetensors payload cache for MGX artifacts.")
    parser.add_argument("--mgx-payload-cache-dir", help="Optional directory for MGX payload cache files.")
    parser.add_argument("--mgx-emit-payload-cache", action="store_true", help="Emit a reusable payload cache when exporting a missing artifact.")
    parser.add_argument("--mgx-export-mode", choices=["normal", "streaming"], default="streaming", help="MGX export implementation used when --export-if-missing is active.")
    parser.add_argument("--export-if-missing", action="store_true", help="Build the .mgx artifact if it does not exist.")
    parser.add_argument("--json-out", help="Optional path to save raw benchmark results as JSON.")

    # Internal worker-only args.
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-label", help=argparse.SUPPRESS)
    parser.add_argument("--worker-model-ref", help=argparse.SUPPRESS)
    parser.add_argument("--worker-baseline-kind", choices=["megagemm-local", "transformers"], default="megagemm-local", help=argparse.SUPPRESS)
    parser.add_argument("--mgx-verify-payload", choices=["default", "true", "false"], default="default", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args._worker:
        if not args.worker_model_ref:
            raise SystemExit("--worker-model-ref is required in worker mode.")
        return _worker_main(args)

    if not args.model:
        parser.error("--model is required")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.baseline == "transformers" and args.batch_size != 1:
        parser.error("--batch-size > 1 currently requires --baseline megagemm-local")

    export_sparsity = "none" if args.sparsity == "auto" else args.sparsity
    quantize_name = _normalize_quantize_name(args.quantize)
    if quantize_name == "native-int4":
        parser.error(
            "Use benchmarks/benchmark_native_w4a16_sparse24.py for the fair native "
            "dense-INT4 versus INT4+2:4 comparison."
        )
    if export_sparsity == "2:4" and quantize_name != "none":
        parser.error("--sparsity 2:4 cannot currently be combined with --quantize INT8/INT4/AWQ")
    mgx_path = Path(args.mgx).expanduser().resolve() if args.mgx else _default_mgx_path(
        args.model,
        args.dtype,
        quantize_name,
        export_sparsity,
    )
    quantize = None if quantize_name == "none" else quantize_name

    export_seconds = None
    if not mgx_path.exists():
        if not args.export_if_missing:
            raise SystemExit(
                f"MGX artifact not found: {mgx_path}. "
                "Use --export-if-missing to compile it automatically."
            )
        print(f"[MGX benchmark] Exporting missing artifact to {mgx_path}...")
        export_start = time.perf_counter()
        export_to_mgx(
            args.model,
            mgx_path,
            dtype=args.dtype,
            quantize=quantize_name,
            sparsity=export_sparsity,
            emit_payload_cache=args.mgx_emit_payload_cache,
            payload_cache_dir=args.mgx_payload_cache_dir,
            export_mode=args.mgx_export_mode,
        )
        export_seconds = time.perf_counter() - export_start
    elif args.mgx_emit_payload_cache:
        print(f"[MGX benchmark] Priming payload cache for existing artifact {mgx_path}...")
        prime_mgx_payload_cache(
            mgx_path,
            validate_payload_hash=not args.mgx_skip_hash_check,
            payload_cache_dir=args.mgx_payload_cache_dir,
        )

    artifact_info = inspect_mgx(mgx_path, validate_payload_hash=False)
    artifact_sparsity = artifact_info["manifest"].get("sparsity", "none")
    if args.sparsity != "auto" and artifact_sparsity != args.sparsity:
        raise SystemExit(
            f"MGX artifact declares sparsity={artifact_sparsity!r}, but --sparsity={args.sparsity!r}."
        )

    baseline_label = "hf_transformers" if args.baseline == "transformers" else "hf_or_local"
    baseline_title = "HuggingFace Transformers" if args.baseline == "transformers" else "HF/local path"
    print(f"[MGX benchmark] Measuring {baseline_title} in isolated worker...")
    baseline_stats = _run_worker(
        label=baseline_label,
        model_ref=args.model,
        device=args.device,
        dtype_name=args.dtype,
        quantize_name=quantize_name,
        prompt=args.prompt,
        first_tokens=args.first_tokens,
        warm_tokens=args.warm_tokens,
        warm_runs=args.warm_runs,
        max_seq_len=args.max_seq_len,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_batch_size=args.max_batch_size,
        batch_size=args.batch_size,
        kv_alloc=args.kv_alloc,
        baseline_kind=args.baseline,
        mgx_verify_payload=None,
        mgx_prefer_payload_cache=args.mgx_prefer_payload_cache,
        mgx_payload_cache_dir=args.mgx_payload_cache_dir,
        mgx_sparse24_runtime="auto",
        mgx_sparse24_kernel="auto",
    )

    print("[MGX benchmark] Measuring MGX path in isolated worker...")
    mgx_stats = _run_worker(
        label="mgx",
        model_ref=str(mgx_path),
        device=args.device,
        dtype_name=args.dtype,
        quantize_name=quantize_name,
        prompt=args.prompt,
        first_tokens=args.first_tokens,
        warm_tokens=args.warm_tokens,
        warm_runs=args.warm_runs,
        max_seq_len=args.max_seq_len,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_batch_size=args.max_batch_size,
        batch_size=args.batch_size,
        kv_alloc=args.kv_alloc,
        baseline_kind="megagemm-local",
        mgx_verify_payload=False if args.mgx_skip_hash_check else None,
        mgx_prefer_payload_cache=args.mgx_prefer_payload_cache,
        mgx_payload_cache_dir=args.mgx_payload_cache_dir,
        mgx_sparse24_runtime=args.mgx_sparse24_runtime,
        mgx_sparse24_kernel=args.mgx_sparse24_kernel,
    )

    result = {
        "environment": _environment_metadata(args.device),
        "config": {
            "model": args.model,
            "mgx": str(mgx_path),
            "baseline": args.baseline,
            "dtype": args.dtype,
            "quantize": quantize or "none",
            "sparsity": artifact_sparsity,
            "mgx_sparse24_runtime": args.mgx_sparse24_runtime,
            "mgx_sparse24_kernel": args.mgx_sparse24_kernel,
            "baseline_quantize": "none" if args.baseline == "transformers" else (quantize or "none"),
            "prompt": args.prompt,
            "first_tokens": args.first_tokens,
            "warm_tokens": args.warm_tokens,
            "warm_runs": args.warm_runs,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "num_blocks": args.num_blocks,
            "block_size": args.block_size,
            "max_batch_size": args.max_batch_size,
            "kv_alloc": args.kv_alloc,
            "mgx_skip_hash_check": bool(args.mgx_skip_hash_check),
            "mgx_prefer_payload_cache": bool(args.mgx_prefer_payload_cache),
            "mgx_payload_cache_dir": args.mgx_payload_cache_dir,
            "mgx_emit_payload_cache": bool(args.mgx_emit_payload_cache),
            "mgx_export_mode": args.mgx_export_mode,
            "export_seconds": export_seconds,
        },
        "baseline_stats": baseline_stats,
        "mgx": mgx_stats,
        "delta": _build_summary(baseline_stats, mgx_stats),
        "memory_delta": _build_memory_summary(baseline_stats, mgx_stats),
    }

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + os.linesep, encoding="utf-8")
        print(f"[MGX benchmark] Saved JSON to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

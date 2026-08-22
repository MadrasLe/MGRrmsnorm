"""Fair single-GPU Gemma 4 MoE comparison between MegaGemm and vLLM.

Each invocation runs exactly one backend. The companion Colab shell script
launches the backends in separate processes so the 48 GiB checkpoint is never
resident twice, while both processes consume the same local model snapshot.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import glob
import io
import json
import math
import os
import site
import statistics
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VLLM_REQUEST_METRIC_FIELDS = (
    "arrival_time",
    "first_scheduled_time",
    "first_token_time",
    "last_token_time",
    "finished_time",
    "queued_ts",
    "scheduled_ts",
    "first_token_ts",
    "last_token_ts",
    "scheduler_time",
    "model_forward_time",
    "model_execute_time",
)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_snapshot() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "vram_gb": round(props.total_memory / 1024**3, 2),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def extract_vllm_request_metrics(request: Any) -> dict[str, float]:
    metrics = getattr(request, "metrics", None)
    if metrics is None:
        return {}
    result: dict[str, float] = {}
    for name in VLLM_REQUEST_METRIC_FIELDS:
        value = getattr(metrics, name, None)
        if value is None:
            continue
        try:
            result[name] = float(value)
        except (TypeError, ValueError):
            continue

    # vLLM 0.25 moved per-request phase timestamps to RequestStateStats. Its
    # arrival_time uses wall-clock time while the *_ts fields use monotonic
    # time, so they must never be subtracted from one another. Canonicalize the
    # new schema to the names consumed below, using scheduled_ts as inference
    # start to match vLLM's own TTFT and inter-token-latency definitions.
    new_schema = (
        result.get("scheduled_ts"),
        result.get("first_token_ts"),
        result.get("last_token_ts"),
    )
    if all(value is not None for value in new_schema):
        if "arrival_time" in result:
            result["frontend_arrival_time"] = result["arrival_time"]
        scheduled, first_token, last_token = new_schema
        result["arrival_time"] = float(scheduled)
        result["first_scheduled_time"] = float(scheduled)
        result["first_token_time"] = float(first_token)
        result["last_token_time"] = float(last_token)
        result["finished_time"] = float(last_token)
    return result


def validated_vllm_phase_span(
    metrics: dict[str, float],
    wall_total_ms: float,
) -> dict[str, Any]:
    required = ("arrival_time", "first_token_time", "finished_time")
    missing = [name for name in required if name not in metrics]
    invalid = {
        "valid": False,
        "prefill_ms": None,
        "decode_ms": None,
        "metric_total_ms": None,
        "wall_error_ms": None,
        "wall_error_ratio": None,
    }
    if missing:
        return {
            **invalid,
            "status": "request_metrics_unavailable",
            "reason": "missing " + ",".join(missing),
        }

    arrival = float(metrics["arrival_time"])
    first = float(metrics["first_token_time"])
    finished = float(metrics["finished_time"])
    if not all(math.isfinite(value) for value in (arrival, first, finished)):
        return {
            **invalid,
            "status": "request_metrics_invalid",
            "reason": "non-finite timestamp",
        }
    if not arrival <= first <= finished:
        return {
            **invalid,
            "status": "request_metrics_invalid",
            "reason": "timestamps are not ordered",
        }

    metric_total_ms = (finished - arrival) * 1000.0
    wall_error_ms = abs(metric_total_ms - float(wall_total_ms))
    wall_tolerance_ms = max(25.0, float(wall_total_ms) * 0.10)
    wall_error_ratio = wall_error_ms / max(float(wall_total_ms), 1e-9)
    if metric_total_ms <= 0.0 or wall_error_ms > wall_tolerance_ms:
        return {
            **invalid,
            "status": "request_metrics_inconsistent",
            "reason": (
                f"metric_total={metric_total_ms:.3f}ms wall_total={wall_total_ms:.3f}ms "
                f"tolerance={wall_tolerance_ms:.3f}ms"
            ),
            "metric_total_ms": metric_total_ms,
            "wall_error_ms": wall_error_ms,
            "wall_error_ratio": wall_error_ratio,
        }

    return {
        "valid": True,
        "status": "valid",
        "reason": "",
        "prefill_ms": (first - arrival) * 1000.0,
        "decode_ms": (finished - first) * 1000.0,
        "metric_total_ms": metric_total_ms,
        "wall_error_ms": wall_error_ms,
        "wall_error_ratio": wall_error_ratio,
    }


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(name)


def load_exact_prompt(model: str, prompt: str) -> tuple[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    formatted = prompt
    if getattr(tokenizer, "chat_template", None):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    token_ids = tokenizer.encode(formatted, add_special_tokens=False)
    return formatted, len(token_ids)


def exact_prompt_patch(engine, exact_prompt: str) -> None:
    def _prepare_prompt_inputs(prompt: str):
        del prompt
        input_ids = engine.tokenizer.encode(
            exact_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(engine.device)
        return exact_prompt, input_ids

    engine._prepare_prompt_inputs = _prepare_prompt_inputs


def token_comparison(reference: list[int], candidate: list[int]) -> dict[str, Any]:
    common = 0
    for left, right in zip(reference, candidate):
        if int(left) != int(right):
            break
        common += 1
    return {
        "exact": reference == candidate,
        "reference_tokens": len(reference),
        "candidate_tokens": len(candidate),
        "common_prefix_tokens": common,
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def median(key: str) -> float | None:
        values = [float(row[key]) for row in samples if row.get(key) is not None]
        return statistics.median(values) if len(values) == len(samples) else None

    return {
        "samples": len(samples),
        "total_ms_median": median("total_ms"),
        "output_tok_s_total_median": median("output_tok_s_total"),
        "prefill_ms_median": median("prefill_ms"),
        "decode_ms_median": median("decode_ms"),
        "decode_tok_s_median": median("decode_tok_s"),
        "decode_measurement_methods": sorted(
            {str(row["decode_measurement_method"]) for row in samples}
        ),
    }


def run_megagemm_request(engine, prompt: str, max_tokens: int) -> dict[str, Any]:
    original_eos = getattr(engine.tokenizer, "eos_token_id", None)
    engine.tokenizer.eos_token_id = None
    stream = io.StringIO()
    sync_cuda()
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(stream):
            output = engine.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                repetition_penalty=1.0,
                verbose=True,
            )
    finally:
        engine.tokenizer.eos_token_id = original_eos
    sync_cuda()
    total_ms = (time.perf_counter() - started) * 1000.0
    token_ids = [int(item) for item in engine._last_generated_ids]
    metrics = dict(engine._last_generation_metrics)
    prefill_timing = engine.model.get_last_prefill_timing()
    if len(token_ids) != max_tokens:
        raise RuntimeError(
            f"MegaGemm produced {len(token_ids)} tokens; expected {max_tokens}"
        )
    prefill_ms = float(metrics.get("prefill_ms") or 0.0)
    decode_ms = float(metrics.get("decode_ms") or 0.0)
    return {
        "total_ms": total_ms,
        "output_tokens": len(token_ids),
        "output_tok_s_total": len(token_ids) / (total_ms / 1000.0),
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_tok_s": len(token_ids) / (decode_ms / 1000.0),
        "decode_measurement_method": "engine_cuda_events",
        "token_ids": token_ids,
        "text_prefix": str(output)[:240],
        "engine_metrics": metrics,
        "prefill_timing": prefill_timing,
        "raw_log": stream.getvalue().strip(),
    }


def graph_status(engine) -> dict[str, Any]:
    states = list(engine._generate_multi_step_graph_states.values())
    failures = [str(state.get("failure") or "") for state in states if state.get("failed")]
    return {
        "captured_graphs": sum(int(state.get("graph") is not None) for state in states),
        "failed_graphs": len(failures),
        "failures": failures,
    }


def prefill_graph_status(engine) -> dict[str, Any]:
    getter = getattr(engine.model, "get_prefill_cuda_graph_store", None)
    store = getter(engine.block_manager) if callable(getter) else {}
    return {
        "enabled": bool(getattr(engine, "_prefill_cuda_graphs", False)),
        "buckets": len(store.get("buckets", {}) or {}),
        "warmups": int(store.get("warmups", 0) or 0),
        "captures": int(store.get("captures", 0) or 0),
        "capture_body_warmups": int(store.get("capture_body_warmups", 0) or 0),
        "capture_replays": int(store.get("capture_replays", 0) or 0),
        "replays": int(store.get("replays", 0) or 0),
        "failures": int(store.get("failures", 0) or 0),
        "last_failure": str(store.get("last_failure", "") or ""),
    }


def run_megagemm(args: argparse.Namespace, exact_prompt: str) -> dict[str, Any]:
    if args.prefill_timing:
        os.environ["MEGAGEMM_PREFILL_TIMING"] = "1"
        os.environ["MEGAGEMM_PREFILL_TIMING_PRINT"] = "1"
    os.environ.setdefault("MEGAGEMM_FP16_STREAMING", "1")
    os.environ.setdefault("MEGAGEMM_FLAT_DECODE", "1")
    os.environ.setdefault("MEGAGEMM_DISABLE_CUDA_RMSNORM", "1")
    os.environ.setdefault("MEGAGEMM_GENERATE_CUDA_GRAPHS", "1")
    os.environ.setdefault("MEGAGEMM_GENERATE_MULTI_STEP_CUDA_GRAPHS", "0")
    os.environ.setdefault("MEGAGEMM_GENERATE_STEP_CUDA_GRAPHS", "1")
    os.environ.setdefault("MEGAGEMM_GENERATE_GPU_TOKEN_CHAIN", "0")
    os.environ.setdefault("MEGAGEMM_GENERATE_SKIP_TOKEN_MATERIALIZATION", "0")
    os.environ.setdefault("MEGAGEMM_GENERATE_FUSED_ARGMAX_STEP", "1")
    os.environ.setdefault("MEGAGEMM_GENERATE_DIRECT_GRAPH_INPUTS", "1")

    from megagemm.engine import InferenceEngine

    engine = InferenceEngine(
        args.model,
        device="cuda",
        dtype=dtype_from_name(args.dtype),
        max_seq_len=args.max_seq_len,
        max_batch_size=1,
    )
    exact_prompt_patch(engine, exact_prompt)

    print(f"== MegaGemm warmup/capture {args.max_tokens} tokens ==")
    warmup = run_megagemm_request(engine, exact_prompt, args.max_tokens)
    print(warmup["raw_log"])
    status = graph_status(engine)
    print("MEGAGEMM_GRAPH_STATUS", json.dumps(status, sort_keys=True))
    require_decode_graph = args.max_tokens >= 3
    if status["failed_graphs"] or (
        require_decode_graph and status["captured_graphs"] <= 0
    ):
        raise RuntimeError(f"MegaGemm CUDA Graph did not capture cleanly: {status}")
    if not require_decode_graph and status["captured_graphs"] <= 0:
        print(
            "MEGAGEMM_GRAPH_STATUS decode graph not required for "
            f"prefill-only diagnostic with max_tokens={args.max_tokens}"
        )

    model = engine.model
    if not bool(getattr(model, "_flat_decode_ready", False)):
        raise RuntimeError(
            "Gemma4 flat decode did not activate: "
            f"{getattr(model, '_flat_decode_failed_reason', '')}"
        )
    if not bool(getattr(model, "_flat_is_gemma4", False)):
        raise RuntimeError("MegaGemm used a non-Gemma4 flat decode path")

    prefill_graph = prefill_graph_status(engine)
    print("MEGAGEMM_PREFILL_GRAPH_STATUS", json.dumps(prefill_graph, sort_keys=True))
    prefill_graph_required = bool(
        prefill_graph["enabled"]
        and not args.prefill_timing
        and os.environ.get("MEGAGEMM_PROFILE_PREFILL", "").strip() != "1"
    )
    if prefill_graph_required:
        if prefill_graph["failures"] or prefill_graph["warmups"] != 1:
            raise RuntimeError(
                "Gemma4 prefill graph did not complete its eager warmup: "
                f"{prefill_graph}"
            )

        print("== MegaGemm prefill graph capture (excluded from summary) ==")
        capture_probe = run_megagemm_request(engine, exact_prompt, args.max_tokens)
        capture_comparison = token_comparison(
            warmup["token_ids"], capture_probe["token_ids"]
        )
        prefill_graph = prefill_graph_status(engine)
        print("MEGAGEMM_PREFILL_GRAPH_STATUS", json.dumps(prefill_graph, sort_keys=True))
        if not capture_comparison["exact"]:
            raise RuntimeError(
                "Gemma4 prefill graph capture changed greedy output: "
                f"{capture_comparison}"
            )
        if (
            prefill_graph["failures"]
            or prefill_graph["captures"] != 1
            or prefill_graph["capture_replays"] != 1
        ):
            raise RuntimeError(
                "Gemma4 prefill graph capture/replay failed; stopping before vLLM install: "
                f"{prefill_graph}"
            )

        print("== MegaGemm prefill graph replay gate (excluded from summary) ==")
        replay_probe = run_megagemm_request(engine, exact_prompt, args.max_tokens)
        replay_comparison = token_comparison(
            warmup["token_ids"], replay_probe["token_ids"]
        )
        prefill_graph = prefill_graph_status(engine)
        print("MEGAGEMM_PREFILL_GRAPH_STATUS", json.dumps(prefill_graph, sort_keys=True))
        if not replay_comparison["exact"]:
            raise RuntimeError(
                "Gemma4 prefill graph replay changed greedy output: "
                f"{replay_comparison}"
            )
        if prefill_graph["failures"] or prefill_graph["replays"] < 1:
            raise RuntimeError(
                "Gemma4 prefill graph did not replay; stopping before vLLM install: "
                f"{prefill_graph}"
            )

    prefill_diagnostic = None
    if args.prefill_timing:
        print("== MegaGemm warm prefill diagnostic (excluded from summary) ==")
        prefill_diagnostic = run_megagemm_request(
            engine,
            exact_prompt,
            args.max_tokens,
        )
        comparison = token_comparison(
            warmup["token_ids"],
            prefill_diagnostic["token_ids"],
        )
        if not comparison["exact"]:
            raise RuntimeError(
                "MegaGemm prefill diagnostic changed greedy output: "
                f"{comparison}"
            )
        print(
            "MEGAGEMM_PREFILL_DIAGNOSTIC "
            + json.dumps(prefill_diagnostic.get("prefill_timing") or {}, sort_keys=True)
        )

        # Keep the diagnostic events out of the three official latency samples.
        import megagemm.models.llama as llama_module

        llama_module._PREFILL_TIMING = False
        model._last_prefill_timing = None

    samples = []
    for index in range(args.repeats):
        print(f"== MegaGemm measured repeat {index + 1}/{args.repeats} ==")
        row = run_megagemm_request(engine, exact_prompt, args.max_tokens)
        row["repeat"] = index + 1
        row["vs_warmup_tokens"] = token_comparison(warmup["token_ids"], row["token_ids"])
        if not row["vs_warmup_tokens"]["exact"]:
            raise RuntimeError(
                "MegaGemm graph replay changed greedy output: "
                f"{row['vs_warmup_tokens']}"
            )
        samples.append(row)
        print(
            f"MegaGemm total={row['total_ms']:.1f}ms "
            f"prefill={row['prefill_ms']:.1f}ms decode={row['decode_ms']:.1f}ms "
            f"decode={row['decode_tok_s']:.1f} tok/s"
        )
        if row.get("prefill_timing"):
            print("MEGAGEMM_PREFILL_TIMING " + json.dumps(
                row["prefill_timing"], sort_keys=True
            ))
    runtime_stats = model.decode_runtime_stats()
    optimization_status = {
        key: runtime_stats.get(key)
        for key in (
            "gemma4_a4b_segmented_prefill_layers",
            "gemma4_a4b_segmented_prefill_effective",
            "gemma4_a4b_segmented_prefill_config",
            "gemma4_fused_qkv_prefill_enabled",
            "gemma4_fused_qkv_prefill_hits",
            "gemma4_fused_qkv_prefill_skip_reason",
            "gemma4_fused_attn_prepare_enabled",
            "gemma4_fused_attn_prepare_hits",
            "gemma4_fused_attn_prepare_disabled_layers",
            "gemma4_fused_attn_prepare_skip_reason",
            "gemma4_implicit_causal_prefill_enabled",
            "gemma4_implicit_causal_prefill_batches",
            "gemma4_implicit_causal_prefill_hits",
            "gemma4_router_fused_norm_scale_hits",
            "gemma4_router_fused_topk_scale_hits",
            "gemma4_router_fused_prefill_enabled",
            "gemma4_router_fused_prefill_hits",
            "gemma4_router_fused_prefill_disabled_layers",
            "gemma4_router_fused_prefill_error",
            "gemma4_fused_dual_ffn_norm_prefill_enabled",
            "gemma4_fused_dual_ffn_norm_prefill_hits",
            "gemma4_fused_dual_ffn_norm_prefill_disabled_layers",
            "gemma4_fused_dual_ffn_norm_prefill_error",
            "gemma4_fused_add_ffn_norm_prefill_enabled",
            "gemma4_fused_add_ffn_norm_prefill_hits",
            "gemma4_fused_add_ffn_norm_prefill_disabled_layers",
            "gemma4_fused_add_ffn_norm_prefill_error",
            "gemma4_fused_post_ffn_norm_prefill_enabled",
            "gemma4_fused_post_ffn_norm_prefill_hits",
            "gemma4_fused_post_ffn_norm_prefill_disabled_layers",
            "gemma4_fused_post_ffn_norm_prefill_error",
            "gemma4_parallel_moe_prefill_enabled",
            "gemma4_parallel_moe_prefill_hits",
            "gemma4_parallel_moe_prefill_policy",
            "prefill_last_token_only_hits",
            "rmsnorm_no_weight_triton_hits",
            "qwen3_moe_segmented_prefill_total_hits",
            "qwen3_moe_segmented_prefill_async_tiles",
            "qwen3_moe_segmented_prefill_async_tiles_max_assignments",
            "qwen3_moe_segmented_prefill_async_tile_hits",
            "qwen3_moe_segmented_prefill_max_tiles",
            "qwen3_moe_segmented_prefill_partial_reduce",
            "qwen3_moe_segmented_prefill_partial_reduce_max_assignments",
            "qwen3_moe_segmented_prefill_partial_reduce_hits",
            "qwen3_moe_segmented_prefill_fixed_route_pack",
            "qwen3_moe_segmented_prefill_fixed_route_pack_hits",
            "qwen3_moe_segmented_prefill_disabled_layers",
            "qwen3_moe_segmented_prefill_first_failure",
            "gemma4_flat_fused_gateup_hits",
            "gemma4_flat_fused_gateup_runtime_disabled",
            "gemma4_flat_deepfusion_hits",
            "gemma4_parallel_moe_decode_enabled",
            "gemma4_parallel_moe_decode_hits",
            "gemma4_parallel_moe_decode_policy",
            "gemma4_fused_attn_moe_bridge_decode_enabled",
            "gemma4_fused_attn_moe_bridge_decode_hits",
            "gemma4_fused_attn_moe_router_bridge_decode_enabled",
            "gemma4_fused_attn_moe_router_bridge_decode_hits",
            "gemma4_fused_attn_moe_router_single_kernel_decode_enabled",
            "gemma4_fused_attn_moe_router_single_kernel_decode_hits",
            "gemma4_fused_post_moe_norm_residual_decode_enabled",
            "gemma4_fused_post_moe_norm_residual_decode_hits",
            "gemma4_fused_router_expert_input_norm_decode_enabled",
            "gemma4_fused_router_expert_input_norm_decode_hits",
            "fused_lm_head_argmax_checked",
            "fused_lm_head_argmax_use",
            "fused_lm_head_argmax_disabled",
            "fused_lm_head_argmax_error",
            "fused_lm_head_argmax_skip_reason",
            "fused_rmsnorm_lm_head_argmax_checked",
            "fused_rmsnorm_lm_head_argmax_use",
            "fused_rmsnorm_lm_head_argmax_disabled",
            "fused_rmsnorm_lm_head_argmax_error",
            "fused_rmsnorm_lm_head_argmax_skip_reason",
        )
    }
    print("MEGAGEMM_OPTIMIZATION_STATUS", json.dumps(optimization_status, sort_keys=True))
    profile_summary = None
    if args.profile_breakdown:
        print(f"== MegaGemm CUDA profile {args.max_tokens} tokens ==")
        profile_summary = engine.profile_decode_breakdown(
            exact_prompt,
            max_new_tokens=args.max_tokens,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
            ignore_eos=True,
        )

    return {
        "backend": "megagemm",
        "warmup": warmup,
        "samples": samples,
        "summary": summarize(samples),
        "graph_status": status,
        "prefill_graph_status": prefill_graph,
        "flat_kind": "gemma4",
        "optimization_status": optimization_status,
        "prefill_diagnostic": prefill_diagnostic,
        "profile_breakdown": profile_summary,
    }


def python_lib_roots() -> list[Path]:
    roots: list[Path] = []
    candidates: list[str | None] = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    candidates.extend(sysconfig.get_paths().get(key) for key in ("purelib", "platlib"))
    for raw in candidates:
        if raw and Path(raw).exists() and Path(raw) not in roots:
            roots.append(Path(raw))
    return roots


def preload_vllm_cuda_runtime() -> dict[str, str]:
    patterns = (
        "nvidia/**/libcudart.so.13*",
        "nvidia/**/lib/libcudart.so.13*",
        "nvidia/**/lib64/libcudart.so.13*",
        "nvidia/**/libcudart.so*",
        "nvidia/**/lib/libcudart.so*",
        "nvidia/**/lib64/libcudart.so*",
    )
    candidates: list[str] = []
    for root in python_lib_roots():
        for pattern in patterns:
            candidates.extend(glob.glob(str(root / pattern)))
    errors = []
    for candidate in sorted(set(candidates)):
        try:
            ctypes.CDLL(candidate, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            return {"status": "preloaded", "path": candidate}
        except OSError as exc:
            errors.append(str(exc))
    return {"status": "not_found", "reason": errors[-1] if errors else "no candidate"}


def run_vllm_request(llm, exact_prompt: str, max_tokens: int) -> dict[str, Any]:
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    sync_cuda()
    started = time.perf_counter()
    try:
        outputs = llm.generate([exact_prompt], params, use_tqdm=False)
    except TypeError:
        outputs = llm.generate([exact_prompt], params)
    sync_cuda()
    total_ms = (time.perf_counter() - started) * 1000.0
    request = outputs[0]
    candidate = request.outputs[0]
    token_ids = [int(item) for item in candidate.token_ids]
    if len(token_ids) != max_tokens:
        raise RuntimeError(f"vLLM produced {len(token_ids)} tokens; expected {max_tokens}")

    request_metrics = extract_vllm_request_metrics(request)
    phase = validated_vllm_phase_span(request_metrics, total_ms)
    prefill_ms = phase["prefill_ms"] if phase["valid"] else None
    decode_ms = (
        phase["decode_ms"]
        if phase["valid"] and max_tokens > 1 and phase["decode_ms"] > 0.0
        else None
    )
    decode_tokens = max_tokens - 1 if decode_ms is not None else None
    prefill_method = (
        "vllm_request_metrics_ttft"
        if prefill_ms is not None
        else str(phase["status"])
    )
    method = (
        "vllm_request_metrics_first_to_finished"
        if decode_ms is not None
        else str(phase["status"])
    )

    return {
        "total_ms": total_ms,
        "output_tokens": len(token_ids),
        "output_tok_s_total": len(token_ids) / (total_ms / 1000.0),
        "prefill_ms": prefill_ms,
        "prefill_measurement_method": prefill_method,
        "decode_ms": decode_ms,
        "decode_tokens": decode_tokens,
        "decode_tok_s": (
            decode_tokens / (decode_ms / 1000.0)
            if decode_tokens is not None and decode_ms and decode_ms > 0.0
            else None
        ),
        "decode_measurement_method": method,
        "request_metrics": request_metrics,
        "phase_metrics_status": phase["status"],
        "phase_metrics_reason": phase["reason"],
        "request_metric_total_ms": phase["metric_total_ms"],
        "request_metric_wall_error_ms": phase["wall_error_ms"],
        "request_metric_wall_error_ratio": phase["wall_error_ratio"],
        "token_ids": token_ids,
        "text_prefix": str(candidate.text)[:240],
    }


def make_vllm(args: argparse.Namespace):
    runtime = preload_vllm_cuda_runtime()
    try:
        import vllm
        from vllm import LLM
    except Exception as exc:
        raise RuntimeError(
            f"vLLM import failed: {type(exc).__name__}: {exc}; runtime={runtime}"
        ) from exc

    version = getattr(vllm, "__version__", "unknown")

    max_num_batched_tokens = max(
        int(args.max_seq_len),
        int(getattr(args, "max_num_batched_tokens", args.max_seq_len)),
    )
    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": "bfloat16" if args.dtype == "bf16" else "float16",
        "trust_remote_code": True,
        "max_model_len": args.max_seq_len,
        # This benchmark exercises the text backbone only. Without these flags
        # vLLM profiles Gemma 4's multimodal encoder and reserves a 4096-token
        # scheduler budget even though no image/video input is ever submitted.
        "language_model_only": True,
        "skip_mm_profiling": True,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": int(getattr(args, "max_batch_size", 1)),
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "enforce_eager": False,
    }
    try:
        llm = LLM(**kwargs)
    except TypeError:
        for key in (
            "trust_remote_code",
            "enable_prefix_caching",
            "disable_log_stats",
            "language_model_only",
            "skip_mm_profiling",
        ):
            kwargs.pop(key, None)
        # Older vLLM releases do not expose language_model_only and still
        # validate Gemma 4's multimodal placeholder against this budget.
        kwargs["max_num_batched_tokens"] = max(4096, max_num_batched_tokens)
        llm = LLM(**kwargs)
    return llm, runtime, version, kwargs


def run_vllm(args: argparse.Namespace, exact_prompt: str) -> dict[str, Any]:
    llm, runtime, version, kwargs = make_vllm(args)
    print(f"vLLM version: {version}")
    print(f"vLLM kwargs: {kwargs}")

    print(f"== vLLM warmup {args.max_tokens} tokens ==")
    warmup = run_vllm_request(llm, exact_prompt, args.max_tokens)
    print(
        f"vLLM warmup total={warmup['total_ms']:.1f}ms "
        f"total={warmup['output_tok_s_total']:.1f} tok/s"
    )

    # A one-token request can select a different scheduler/graph path. Only
    # timestamps from the measured long request may define its phase split.
    samples = []
    for index in range(args.repeats):
        print(f"== vLLM measured repeat {index + 1}/{args.repeats} ==")
        row = run_vllm_request(llm, exact_prompt, args.max_tokens)
        row["repeat"] = index + 1
        samples.append(row)
        if row["decode_ms"] is None:
            print(
                f"vLLM total={row['total_ms']:.1f}ms phase=n/a "
                f"status={row['phase_metrics_status']}"
            )
        else:
            print(
                f"vLLM total={row['total_ms']:.1f}ms "
                f"prefill={row['prefill_ms']:.1f}ms "
                f"decode={row['decode_ms']:.1f}ms "
                f"decode={row['decode_tok_s']:.1f} tok/s"
            )
        print("VLLM_REQUEST_METRICS " + json.dumps(row["request_metrics"], sort_keys=True))

    return {
        "backend": "vllm",
        "version": version,
        "cuda_runtime": runtime,
        "llm_kwargs": kwargs,
        "warmup": warmup,
        "samples": samples,
        "summary": summarize(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("megagemm", "vllm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--profile-breakdown", action="store_true")
    parser.add_argument("--prefill-timing", action="store_true")
    parser.add_argument(
        "--prompt",
        default="Write a compact Python Fibonacci function and explain its time complexity.",
    )
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.max_tokens < 2:
        raise ValueError("max_tokens must be >= 2")
    if args.repeats < 1:
        raise ValueError("repeats must be >= 1")

    exact_prompt, prompt_tokens = load_exact_prompt(args.model, args.prompt)
    print("Gemma 4 MoE backend benchmark")
    print(f"  backend: {args.backend}")
    print(f"  model: {args.model}")
    print(f"  dtype: {args.dtype}")
    print(f"  prompt_tokens: {prompt_tokens}")
    print(f"  output_tokens: {args.max_tokens}")
    print(f"  repeats: {args.repeats}")
    print(f"  gpu: {gpu_snapshot()}")

    if prompt_tokens + args.max_tokens > args.max_seq_len:
        raise ValueError("prompt plus output exceeds max_seq_len")

    if args.backend == "megagemm":
        result = run_megagemm(args, exact_prompt)
    else:
        result = run_vllm(args, exact_prompt)
    result.update(
        {
            "model": args.model,
            "dtype": args.dtype,
            "batch_size": 1,
            "prompt_tokens": prompt_tokens,
            "max_tokens": args.max_tokens,
            "gpu": gpu_snapshot(),
        }
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("== SUMMARY ==")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

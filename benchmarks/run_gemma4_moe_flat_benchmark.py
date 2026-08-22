"""Warm and measure Gemma 4 MoE decode in one process.

The first request intentionally absorbs Triton/kernel compilation. The second
request is the only performance result that should be compared.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megagemm.engine import InferenceEngine


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _flat_status(engine: InferenceEngine) -> dict:
    model = engine.model
    moe_layers = [layer for layer in model.layers if getattr(layer, "is_moe_layer", False)]
    expert_modules = [layer.mlp.experts for layer in moe_layers]
    return {
        "flat_ready": bool(getattr(model, "_flat_decode_ready", False)),
        "flat_failed": bool(getattr(model, "_flat_decode_failed", False)),
        "flat_reason": str(getattr(model, "_flat_decode_failed_reason", "") or ""),
        "flat_kind": "gemma4" if bool(getattr(model, "_flat_is_gemma4", False)) else "other",
        "moe_layers": len(moe_layers),
        "grouped_hits": sum(int(getattr(experts, "_grouped_decode_hits", 0)) for experts in expert_modules),
        "grouped_disabled_layers": sum(
            int(bool(getattr(experts, "_grouped_decode_disabled", False)))
            for experts in expert_modules
        ),
        "grouped_failures": [
            str(getattr(experts, "_grouped_decode_fail_reason", "") or "")
            for experts in expert_modules
            if bool(getattr(experts, "_grouped_decode_disabled", False))
        ],
    }


def _token_comparison(reference: list[int], candidate: list[int]) -> dict:
    common = 0
    for left, right in zip(reference, candidate):
        if int(left) != int(right):
            break
        common += 1
    first_reference = int(reference[common]) if common < len(reference) else None
    first_candidate = int(candidate[common]) if common < len(candidate) else None
    return {
        "exact": reference == candidate,
        "reference_tokens": len(reference),
        "candidate_tokens": len(candidate),
        "common_prefix_tokens": common,
        "first_reference_token": first_reference,
        "first_candidate_token": first_candidate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma 4 MoE flat decode benchmark")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--graph-ab",
        action="store_true",
        help="measure eager and single-step CUDA Graph decode in the same model load",
    )
    parser.add_argument(
        "--prompt",
        default="Write a compact Python Fibonacci function and explain its time complexity.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print("Gemma 4 MoE flat decode benchmark")
    print(f"  model: {args.model}")
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(f"  vram_gb: {total_bytes / 1024**3:.2f}")
    print(f"  free_gb: {free_bytes / 1024**3:.2f}")
    print(f"  dtype: {args.dtype}")
    print(f"  warmup_tokens: {args.warmup_tokens}")
    print(f"  measured_tokens: {args.max_new_tokens}")

    engine = InferenceEngine(
        args.model,
        device="cuda",
        dtype=_dtype(args.dtype),
        max_seq_len=args.max_seq_len,
        max_batch_size=1,
    )

    print("\n== WARMUP / COMPILE (do not compare this number) ==")
    engine.generate(
        args.prompt,
        max_new_tokens=args.warmup_tokens,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        verbose=True,
    )
    torch.cuda.synchronize()

    warm_status = _flat_status(engine)
    print("FLAT_STATUS_AFTER_WARMUP", json.dumps(warm_status, sort_keys=True))
    if not warm_status["flat_ready"] or warm_status["flat_kind"] != "gemma4":
        raise RuntimeError(
            "Gemma 4 flat decode did not activate: "
            f"{warm_status['flat_reason'] or 'no failure reason'}"
        )
    if warm_status["grouped_disabled_layers"]:
        raise RuntimeError(
            "Grouped expert decode disabled during warmup: "
            f"{warm_status['grouped_failures']}"
        )
    if warm_status["grouped_hits"] <= 0:
        raise RuntimeError("Grouped expert decode had zero hits during warmup")

    print("\n== EAGER MEASURED RUN ==")
    started = time.perf_counter()
    output = engine.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        verbose=True,
    )
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - started) * 1000.0
    eager_ids = list(engine._last_generated_ids)
    eager_metrics = dict(engine._last_generation_metrics)

    final_status = _flat_status(engine)
    grouped_delta = int(final_status["grouped_hits"]) - int(warm_status["grouped_hits"])
    final_status["grouped_hit_delta"] = grouped_delta
    print("FINAL_STATUS", json.dumps(final_status, sort_keys=True))
    if grouped_delta <= 0:
        raise RuntimeError("Measured run did not execute grouped expert decode")

    print(f"TOTAL_WALL_MS {total_ms:.1f}")
    print("\n== EAGER OUTPUT ==")
    print(output)

    if args.graph_ab:
        engine._generate_cuda_graphs = True
        engine._generate_multi_step_cuda_graphs = False
        engine._generate_step_cuda_graphs = True
        engine._generate_gpu_token_chain = False
        engine._generate_fused_argmax_step = True

        print("\n== CUDA GRAPH WARMUP / CAPTURE (do not compare this number) ==")
        graph_warm_output = engine.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
            verbose=True,
        )
        torch.cuda.synchronize()
        graph_warm_ids = list(engine._last_generated_ids)
        graph_states = list(engine._generate_multi_step_graph_states.values())
        graph_failures = [
            str(state.get("failure") or "")
            for state in graph_states
            if state.get("failed")
        ]
        captured_graphs = sum(int(state.get("graph") is not None) for state in graph_states)
        graph_status = {
            "captured_graphs": captured_graphs,
            "failed_graphs": len(graph_failures),
            "failures": graph_failures,
        }
        print("CUDA_GRAPH_STATUS_AFTER_WARMUP", json.dumps(graph_status, sort_keys=True))
        if graph_failures or captured_graphs <= 0:
            raise RuntimeError(f"Gemma 4 CUDA Graph capture failed: {graph_status}")
        eager_capture_comparison = _token_comparison(eager_ids, graph_warm_ids)
        print(
            "EAGER_CAPTURE_TOKEN_COMPARISON",
            json.dumps(eager_capture_comparison, sort_keys=True),
        )
        if not eager_capture_comparison["exact"]:
            raise RuntimeError(
                "CUDA Graph capture output differs from eager output: "
                f"{eager_capture_comparison}"
            )

        print("\n== CUDA GRAPH MEASURED RUN ==")
        graph_started = time.perf_counter()
        graph_output = engine.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
            verbose=True,
        )
        torch.cuda.synchronize()
        graph_total_ms = (time.perf_counter() - graph_started) * 1000.0
        graph_ids = list(engine._last_generated_ids)
        graph_metrics = dict(engine._last_generation_metrics)
        capture_replay_comparison = _token_comparison(graph_warm_ids, graph_ids)
        print(
            "CAPTURE_REPLAY_TOKEN_COMPARISON",
            json.dumps(capture_replay_comparison, sort_keys=True),
        )
        if not graph_ids:
            raise RuntimeError("CUDA Graph measured run produced no token ids")
        eager_replay_comparison = _token_comparison(eager_ids, graph_ids)
        print(
            "EAGER_GRAPH_REPLAY_TOKEN_COMPARISON",
            json.dumps(eager_replay_comparison, sort_keys=True),
        )
        if not eager_replay_comparison["exact"]:
            raise RuntimeError(
                "CUDA Graph replay output differs from eager output: "
                f"{eager_replay_comparison}"
            )
        print(f"CUDA_GRAPH_TOTAL_WALL_MS {graph_total_ms:.1f}")
        print(f"CUDA_GRAPH_TOTAL_REQUEST_SPEEDUP {total_ms / graph_total_ms:.3f}x")
        eager_decode_ms = float(eager_metrics.get("decode_ms") or 0.0)
        graph_decode_ms = float(graph_metrics.get("decode_ms") or 0.0)
        if eager_decode_ms > 0.0 and graph_decode_ms > 0.0:
            print(f"CUDA_GRAPH_DECODE_SPEEDUP {eager_decode_ms / graph_decode_ms:.3f}x")
        print("\n== CUDA GRAPH OUTPUT ==")
        print(graph_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

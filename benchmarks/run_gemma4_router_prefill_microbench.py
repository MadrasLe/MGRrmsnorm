#!/usr/bin/env python3
"""Compare the current Gemma 4 prefill router with the fused matrix kernel."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megagemm.kernels.gemma4_moe_router import gemma4_moe_prefill_router_topk
from megagemm.kernels.qwen3_moe import qwen3_moe_topk_softmax
from megagemm.kernels.rmsnorm_triton import rmsnorm_triton_scaled_no_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--out-json",
        default="bench_results/gemma4_router_prefill_a100.json",
    )
    return parser.parse_args()


def measure(call, warmup: int, iterations: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) * 1000.0 / iterations))
    return float(statistics.median(samples)), samples


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.set_grad_enabled(False)

    torch.manual_seed(41)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    rows, hidden, experts, top_k = 25, 2816, 128, 8
    hidden_states = torch.randn(rows, hidden, device=device, dtype=dtype)
    input_scale = (1.0 + 0.05 * torch.randn(hidden, device=device)).to(dtype)
    router_weight = (0.02 * torch.randn(experts, hidden, device=device)).to(dtype)
    expert_scale = (1.0 + 0.05 * torch.randn(experts, device=device)).to(dtype)
    eps = 1e-6
    output_scale = hidden ** -0.5

    baseline_workspace: dict[str, torch.Tensor] = {}
    fused_workspace: dict[str, torch.Tensor] = {}
    logits = torch.empty(rows, experts, device=device, dtype=dtype)

    def baseline():
        normalized = rmsnorm_triton_scaled_no_weight(
            hidden_states,
            input_scale,
            eps,
            output_scale,
        )
        torch.mm(normalized, router_weight.t(), out=logits)
        return qwen3_moe_topk_softmax(
            logits,
            top_k,
            workspace=baseline_workspace,
            expert_scale=expert_scale,
        )

    def fused():
        normalized = rmsnorm_triton_scaled_no_weight(
            hidden_states,
            input_scale,
            eps,
            output_scale,
        )
        return gemma4_moe_prefill_router_topk(
            normalized,
            router_weight,
            expert_scale,
            top_k,
            workspace=fused_workspace,
        )

    ref_weights, ref_experts = baseline()
    ref_weights = ref_weights.clone()
    ref_experts = ref_experts.clone()
    fused_weights, fused_experts = fused()
    torch.cuda.synchronize()
    experts_equal = bool(torch.equal(ref_experts, fused_experts))
    max_weight_error = float((ref_weights.float() - fused_weights.float()).abs().max().item())
    cosine = float(torch.nn.functional.cosine_similarity(
        ref_weights.float().reshape(1, -1),
        fused_weights.float().reshape(1, -1),
    ).item())

    rows_out = []
    for name, call in (("current", baseline), ("fused_matrix_router", fused)):
        median_us, samples = measure(call, args.warmup, args.iterations, args.repeats)
        row = {
            "case": name,
            "median_us": median_us,
            "samples_us": samples,
            "ms_per_prefill_30_layers": median_us * 30.0 / 1000.0,
            "experts_equal": experts_equal,
            "max_weight_error": max_weight_error,
            "cosine": cosine,
        }
        rows_out.append(row)
        print(json.dumps(row, sort_keys=True))

    ranking = sorted(rows_out, key=lambda item: item["median_us"])
    baseline_us = next(item["median_us"] for item in rows_out if item["case"] == "current")
    fused_us = next(item["median_us"] for item in rows_out if item["case"] == "fused_matrix_router")
    speedup = baseline_us / fused_us
    decision = "APPLY" if experts_equal and cosine >= 0.9999 and speedup >= 1.03 else "REVERT"
    summary = {
        "decision": decision,
        "speedup": speedup,
        "estimated_savings_ms_per_prefill": (baseline_us - fused_us) * 30.0 / 1000.0,
        "gpu": torch.cuda.get_device_name(0),
        "shape": {
            "rows": rows,
            "hidden": hidden,
            "experts": experts,
            "top_k": top_k,
            "dtype": "bf16",
        },
        "ranking": ranking,
    }
    print("DECISION " + json.dumps(summary, sort_keys=True))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

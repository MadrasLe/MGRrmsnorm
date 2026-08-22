"""
Microbenchmark for Qwen-style prefill MLP.

Focuses only on the dominant block:
  gate_up -> SwiGLU -> down

This lets us compare candidate implementations without scheduler / KV / decode
noise and answer a narrower question:
  "Which MLP path is actually faster for large prefill shapes on this GPU?"

Typical Colab usage:

    %cd /content/drive/MyDrive/MGRrmsnorm
    %pip install -e . --no-build-isolation -q
    !python benchmarks/benchmark_mlp_prefill.py --rows 10644 21288 42576

Qwen2.5-7B Bench360-equivalent rows:
  B=32  -> ~10644
  B=64  -> ~21288
  B=128 -> ~42576
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from megagemm.kernels.deepfusion_mlp import (
    HAS_DEEPFUSION_MLP,
    deepfusion_mlp_prefers_triton_shape,
    deepfusion_runtime_config,
    deepfusion_swiglu_down,
)
from megagemm.kernels.mlp_prefill_native import (
    HAS_NATIVE_MLP_PREFILL,
    mlp_prefill_forward_cuda,
)
from megagemm.kernels.swiglu import swiglu_forward


def _cuda_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> Tuple[float, float]:
    for _ in range(warmup):
        out = fn()
        if torch.is_tensor(out):
            del out
    torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
        if torch.is_tensor(out):
            del out

    mean = sum(times) / len(times)
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
    return mean, std


def _tflops(m_ms: float, rows: int, hidden: int, inter: int) -> float:
    if m_ms <= 0:
        return 0.0
    secs = m_ms / 1000.0
    # Rough forward FLOPs for SwiGLU MLP:
    # gate_up: 2 * rows * hidden * (2*inter)
    # down:    2 * rows * inter * hidden
    # swiglu elementwise is comparatively tiny; ignore in headline TFLOPs
    flops = 6.0 * rows * hidden * inter
    return flops / secs / 1e12


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def build_weights(
    hidden: int,
    intermediate: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    gate_up_w = torch.randn(
        2 * intermediate, hidden, generator=g, device=device, dtype=dtype
    ) / math.sqrt(hidden)
    down_w = torch.randn(
        hidden, intermediate, generator=g, device=device, dtype=dtype
    ) / math.sqrt(intermediate)
    return {
        "gate_up_w": gate_up_w.contiguous(),
        "down_w": down_w.contiguous(),
    }


def build_input(rows: int, hidden: int, dtype: torch.dtype, device: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(rows, hidden, generator=g, device=device, dtype=dtype)


def baseline_components(
    x: torch.Tensor,
    gate_up_w: torch.Tensor,
    down_w: torch.Tensor,
    warmup: int,
    iters: int,
) -> Dict[str, Tuple[float, float]]:
    inter = gate_up_w.shape[0] // 2

    def gate_up_fn():
        return F.linear(x, gate_up_w)

    gate_up = F.linear(x, gate_up_w)

    def swiglu_fn():
        return swiglu_forward(gate_up, inter)

    activated = swiglu_forward(gate_up, inter)

    def down_fn():
        return F.linear(activated, down_w)

    def full_fn():
        gate = F.linear(x, gate_up_w)
        act = swiglu_forward(gate, inter)
        return F.linear(act, down_w)

    stats = {
        "gate_up": _cuda_ms(gate_up_fn, warmup, iters),
        "swiglu": _cuda_ms(swiglu_fn, warmup, iters),
        "down": _cuda_ms(down_fn, warmup, iters),
        "full": _cuda_ms(full_fn, warmup, iters),
    }
    del gate_up, activated
    return stats


def run_benchmark(
    rows: int,
    hidden: int,
    intermediate: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    seed: int,
) -> None:
    device = "cuda"
    x = build_input(rows, hidden, dtype, device, seed)
    weights = build_weights(hidden, intermediate, dtype, device, seed + 1)
    gate_up_w = weights["gate_up_w"]
    down_w = weights["down_w"]

    print("\n" + "=" * 78)
    print(
        f"MLP Prefill Benchmark | rows={rows} hidden={hidden} "
        f"intermediate={intermediate} dtype={dtype}"
    )
    print("=" * 78)

    baseline = baseline_components(x, gate_up_w, down_w, warmup, iters)
    baseline_full_ms = baseline["full"][0]
    baseline_tail_ms = baseline["swiglu"][0] + baseline["down"][0]
    print("baseline:")
    print(
        f"  gate_up={baseline['gate_up'][0]:.2f}ms | "
        f"swiglu={baseline['swiglu'][0]:.2f}ms | "
        f"down={baseline['down'][0]:.2f}ms | "
        f"full={baseline_full_ms:.2f}ms | "
        f"~{_tflops(baseline_full_ms, rows, hidden, intermediate):.2f} TFLOP/s"
    )

    with torch.no_grad():
        gate = F.linear(x, gate_up_w)
        act = swiglu_forward(gate, intermediate)
        baseline_out = F.linear(act, down_w)
        del gate, act

    if HAS_NATIVE_MLP_PREFILL:
        x_native = x.unsqueeze(0)

        def native_fn():
            return mlp_prefill_forward_cuda(
                x_native,
                gate_up_w,
                None,
                down_w,
                None,
                intermediate,
            )

        print("native_mlp_prefill:")
        try:
            native_ms, native_std = _cuda_ms(native_fn, warmup, iters)
            native_out = native_fn().squeeze(0)
            diff = _max_abs_diff(native_out, baseline_out)
            print(
                f"  full={native_ms:.2f}ms ± {native_std:.2f} | "
                f"speedup={baseline_full_ms / native_ms:.3f}x | "
                f"diff={diff:.5f}"
            )
            del native_out
        except Exception as exc:
            print(f"  failed: {exc}")
        del x_native
    else:
        print("native_mlp_prefill:")
        print("  unavailable")

    if HAS_DEEPFUSION_MLP:
        gate_up_buf = F.linear(x, gate_up_w)
        deepfusion_uses_triton = deepfusion_mlp_prefers_triton_shape(
            intermediate,
            hidden,
            rows,
            mode="prefill",
        )

        def deepfusion_fn():
            return deepfusion_swiglu_down(
                gate_up_buf,
                down_w,
                mode="prefill",
            )

        print("deepfusion_prefill_tail:")
        try:
            deep_ms, deep_std = _cuda_ms(deepfusion_fn, warmup, iters)
            deep_out = deepfusion_fn()
            diff = _max_abs_diff(deep_out, baseline_out)
            print(
                f"  tail={deep_ms:.2f}ms ± {deep_std:.2f} | "
                f"tail_vs_baseline_tail={baseline_tail_ms / deep_ms:.3f}x | "
                f"backend={'triton' if deepfusion_uses_triton else 'fallback'} | "
                f"diff={diff:.5f}"
            )
            del deep_out
        except Exception as exc:
            print(f"  failed: {exc}")
        del gate_up_buf
    else:
        print("deepfusion_prefill_tail:")
        print("  unavailable")

    torch.cuda.synchronize()
    print(
        f"peak_mem_alloc={torch.cuda.max_memory_allocated() / 1e9:.2f}GB | "
        f"peak_mem_reserved={torch.cuda.max_memory_reserved() / 1e9:.2f}GB"
    )
    torch.cuda.reset_peak_memory_stats()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the prefill MLP hot path")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[10644, 21288, 42576],
        help="Flattened prefill rows to benchmark",
    )
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument("--intermediate", type=int, default=18944)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print(f"Native MLP prefill available: {HAS_NATIVE_MLP_PREFILL}")
    print(f"DeepFusion available: {HAS_DEEPFUSION_MLP}")
    if HAS_DEEPFUSION_MLP:
        print(f"DeepFusion config: {deepfusion_runtime_config()}")

    torch.set_grad_enabled(False)
    for rows in args.rows:
        run_benchmark(
            rows=rows,
            hidden=args.hidden,
            intermediate=args.intermediate,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

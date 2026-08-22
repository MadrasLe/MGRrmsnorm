"""Fair MGX FP16 dense versus FP16 2:4 benchmark.

No quantization backend participates in this harness.  Both artifacts originate
from the same FP16 checkpoint; the sparse artifact changes only weight pruning,
packed storage and the selected FP16 2:4 runtime kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from benchmark_mgx import (
    _build_memory_summary,
    _build_summary,
    _environment_metadata,
    _run_worker,
    _slugify_model_ref,
)
from megagemm.kernels.sparse24_mma import (
    sparse24_mma_available,
    sparse24_mma_import_error,
    sparse24_mma_linear,
    sparse24_portable_metadata_to_ptx,
)
from megagemm.models import export_to_mgx, inspect_mgx
from megagemm.models.sparsity import pack_sparse24_weight, unpack_sparse24_weight


def _parse_batch_sizes(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("batch sizes must contain positive integers")
    return list(dict.fromkeys(parsed))


def _artifact_paths(model: str, artifact_dir: Path) -> tuple[Path, Path]:
    stem = _slugify_model_ref(model)
    return (
        (artifact_dir / f"{stem}-fp16.mgx").resolve(),
        (artifact_dir / f"{stem}-fp16-sparse24.mgx").resolve(),
    )


def _ensure_artifact(
    model: str,
    path: Path,
    *,
    sparse24: bool,
    force_export: bool,
    export_mode: str,
) -> tuple[dict[str, Any], float | None]:
    export_seconds = None
    if force_export or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[FP16 2:4] Exporting {'sparse' if sparse24 else 'dense'} FP16 artifact: {path}")
        started = time.perf_counter()
        export_to_mgx(
            model,
            path,
            dtype="fp16",
            quantize="none",
            sparsity="2:4" if sparse24 else "none",
            export_mode=export_mode,
        )
        export_seconds = time.perf_counter() - started

    info = inspect_mgx(path, validate_payload_hash=False)
    manifest = info["manifest"]
    expected_sparsity = "2:4" if sparse24 else "none"
    if manifest.get("quantization", "none") != "none":
        raise SystemExit(f"Artifact {path} is quantized; this benchmark requires pure FP16.")
    if manifest.get("dtype") != "fp16":
        raise SystemExit(f"Artifact {path} is not FP16. Use --force-export to replace it.")
    if manifest.get("sparsity", "none") != expected_sparsity:
        raise SystemExit(
            f"Artifact {path} declares sparsity={manifest.get('sparsity')!r}; "
            f"expected {expected_sparsity!r}. Use --force-export."
        )
    return info, export_seconds


def _measure(
    path: Path,
    *,
    label: str,
    batch_size: int,
    args: argparse.Namespace,
    sparse24: bool,
) -> dict[str, Any]:
    return _run_worker(
        label=label,
        model_ref=str(path),
        device="cuda",
        dtype_name="fp16",
        quantize_name="none",
        prompt=args.prompt,
        first_tokens=args.first_tokens,
        warm_tokens=args.warm_tokens,
        warm_runs=args.warm_runs,
        max_seq_len=args.max_seq_len,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_batch_size=max(batch_size, args.max_batch_size),
        batch_size=batch_size,
        kv_alloc=args.kv_alloc,
        baseline_kind="megagemm-local",
        mgx_verify_payload=False if args.skip_hash_check else None,
        mgx_prefer_payload_cache=False,
        mgx_payload_cache_dir=None,
        mgx_sparse24_runtime="on" if sparse24 else "off",
        mgx_sparse24_kernel=args.kernel if sparse24 else "auto",
    )


def _validate_native_kernel() -> None:
    """Reject a build whose mma.sp register/metadata mapping is incorrect."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260820)
    dense = torch.randn((64, 128), device="cuda", dtype=torch.float16, generator=generator)
    values, portable_metadata = pack_sparse24_weight(dense)
    ptx_metadata = sparse24_portable_metadata_to_ptx(portable_metadata)
    pruned = unpack_sparse24_weight(values, portable_metadata, dense.shape)
    bias = torch.randn((64,), device="cuda", dtype=torch.float16, generator=generator)
    try:
        for rows in (1, 8, 16, 32, 64):
            x = torch.randn(
                (rows, 128),
                device="cuda",
                dtype=torch.float16,
                generator=generator,
            )
            actual = sparse24_mma_linear(x, values, ptx_metadata, bias)
            expected = torch.nn.functional.linear(x, pruned, bias)
            if not torch.allclose(actual, expected, rtol=2e-2, atol=2e-2):
                max_error = float((actual.float() - expected.float()).abs().max().item())
                raise SystemExit(
                    "Native mma.sp correctness self-test failed "
                    f"for M={rows} (max abs error={max_error:.6f})."
                )
        torch.cuda.synchronize()
    finally:
        del dense, values, portable_metadata, ptx_metadata, pruned, bias


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare pure FP16 MGX against pure FP16 MGX with structured 2:4 sparsity."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=_parse_batch_sizes("1,16,64"))
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--export-mode", choices=["normal", "streaming"], default="streaming")
    parser.add_argument("--kernel", choices=["native", "auto", "triton", "torch"], default="native")
    parser.add_argument("--prompt", default="Explain why compiled model artifacts reduce cold-start latency.")
    parser.add_argument("--first-tokens", type=int, default=8)
    parser.add_argument("--warm-tokens", type=int, default=256)
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--kv-alloc", choices=["auto", "greedy"], default="auto")
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--json-out")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    if args.first_tokens <= 0 or args.warm_tokens <= 0 or args.warm_runs <= 0:
        parser.error("token and run counts must be positive")
    if args.kernel == "native" and not sparse24_mma_available():
        raise SystemExit(
            "The standalone sparse24_cuda_ops extension is not installed. "
            "Reinstall MegaGemm from this checkout before benchmarking. "
            f"Import error: {sparse24_mma_import_error()}"
        )
    if args.kernel == "native":
        print("[FP16 2:4] Validating native mma.sp numerical correctness...")
        _validate_native_kernel()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    dense_path, sparse_path = _artifact_paths(args.model, artifact_dir)
    dense_info, dense_export_seconds = _ensure_artifact(
        args.model,
        dense_path,
        sparse24=False,
        force_export=args.force_export,
        export_mode=args.export_mode,
    )
    sparse_info, sparse_export_seconds = _ensure_artifact(
        args.model,
        sparse_path,
        sparse24=True,
        force_export=args.force_export,
        export_mode=args.export_mode,
    )

    rows = []
    for batch_size in args.batch_sizes:
        print(f"[FP16 2:4] batch={batch_size}: dense FP16 isolated worker...")
        dense = _measure(
            dense_path,
            label="fp16_dense",
            batch_size=batch_size,
            args=args,
            sparse24=False,
        )
        print(f"[FP16 2:4] batch={batch_size}: sparse FP16 isolated worker ({args.kernel})...")
        sparse = _measure(
            sparse_path,
            label="fp16_sparse24",
            batch_size=batch_size,
            args=args,
            sparse24=True,
        )
        runtime = (sparse.get("decode_runtime") or {}).get("mgx_sparsity_runtime") or {}
        if args.kernel == "native":
            dense_decode_runtime = dense.get("decode_runtime") or {}
            decode_runtime = sparse.get("decode_runtime") or {}
            requested = int(runtime.get("requested_tensor_count", 0))
            native_tensors = int(runtime.get("native_mma_tensor_count", 0))
            native_hits = int(runtime.get("native_mma_kernel_hits", 0))
            flat_native_hits = int(runtime.get("flat_decode_native_mma_hits", 0))
            native_failures = int(runtime.get("native_mma_kernel_failures", 0))
            torch_fallbacks = int(runtime.get("torch_sparse_fallback_hits", 0))
            fully_native = bool(
                requested > 0
                and native_tensors == requested
                and native_hits > 0
                and flat_native_hits > 0
                and native_failures == 0
                and torch_fallbacks == 0
                and bool(dense_decode_runtime.get("flat_decode_ready", False))
                and bool(decode_runtime.get("flat_decode_ready", False))
            )
            if not fully_native:
                raise SystemExit(
                    "Sparse worker was not fully native mma.sp; refusing a misleading FP16 "
                    "2:4 label. Runtime: " + json.dumps(runtime, default=str)
                )
        rows.append(
            {
                "batch_size": batch_size,
                "dense": dense,
                "sparse24": sparse,
                "delta": _build_summary(dense, sparse),
                "memory_delta_dense_minus_sparse_mb": _build_memory_summary(dense, sparse),
            }
        )

    print("\nFair comparison: FP16 dense vs FP16 2:4 (no INT4)")
    print("batch | dense tok/s | 2:4 tok/s | TPS delta | dense VRAM MB | 2:4 VRAM MB | mma.sp hits")
    for row in rows:
        dense = row["dense"]
        sparse = row["sparse24"]
        runtime = (sparse.get("decode_runtime") or {}).get("mgx_sparsity_runtime") or {}
        delta = row["delta"].get("warmed_decode_tps_speedup_pct")
        print(
            f"{row['batch_size']:>5} | "
            f"{dense['warmed_decode_tokens_per_second']:>11.2f} | "
            f"{sparse['warmed_decode_tokens_per_second']:>9.2f} | "
            f"{delta:>+8.2f}% | "
            f"{dense.get('cuda_allocated_after_load_mb', 0.0):>13.1f} | "
            f"{sparse.get('cuda_allocated_after_load_mb', 0.0):>11.1f} | "
            f"{int(runtime.get('native_mma_kernel_hits', 0)):>11}"
        )

    result = {
        "comparison": "mgx-fp16-dense-vs-mgx-fp16-sparse24",
        "fairness": {
            "same_source_model": True,
            "dense_dtype": "float16",
            "sparse_dtype": "float16",
            "quantization": None,
            "experimental_variable": "2:4 pruning, packed storage, and FP16 sparse kernel",
        },
        "environment": _environment_metadata("cuda"),
        "config": vars(args) | {"artifact_dir": str(artifact_dir)},
        "artifacts": {
            "dense": {
                "path": str(dense_path),
                "file_size": dense_info["file_size"],
                "export_seconds": dense_export_seconds,
            },
            "sparse24": {
                "path": str(sparse_path),
                "file_size": sparse_info["file_size"],
                "export_seconds": sparse_export_seconds,
                "sparsity_config": sparse_info["manifest"].get("sparsity_config"),
            },
        },
        "results": rows,
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    if args.print_json:
        print(payload)
    if args.json_out:
        output = Path(args.json_out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + os.linesep, encoding="utf-8")
        print(f"Saved JSON to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

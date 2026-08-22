"""Fair standalone MGX W4A16 dense versus W4A16 2:4 benchmark.

Both artifacts are quantized from the same floating-point checkpoint with the
same symmetric INT4 group quantizer. The only experimental variable is the
native 2:4 pruning, packing, and kernel route.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from benchmark_mgx import (
    _build_memory_summary,
    _build_summary,
    _environment_metadata,
    _run_worker,
    _slugify_model_ref,
)
from megagemm.models import export_to_mgx, inspect_mgx


def _parse_batch_sizes(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("batch sizes must contain positive integers")
    return list(dict.fromkeys(result))


def _artifact_paths(model: str, artifact_dir: Path) -> tuple[Path, Path]:
    stem = _slugify_model_ref(model)
    return (
        (artifact_dir / f"{stem}-native-int4.mgx").resolve(),
        (artifact_dir / f"{stem}-native-int4-sparse24.mgx").resolve(),
    )


def _ensure_artifact(
    model: str,
    path: Path,
    *,
    dtype: str,
    sparse24: bool,
    force_export: bool,
    export_mode: str,
) -> tuple[dict[str, Any], float | None]:
    export_seconds = None
    if force_export or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[native W4A16] Exporting {'2:4' if sparse24 else 'dense'} artifact: {path}")
        started = time.perf_counter()
        export_to_mgx(
            model,
            path,
            dtype=dtype,
            quantize="native-int4",
            sparsity="2:4" if sparse24 else "none",
            export_mode=export_mode,
        )
        export_seconds = time.perf_counter() - started

    info = inspect_mgx(path, validate_payload_hash=False)
    manifest = info["manifest"]
    quant_config = manifest.get("quantization_config") or {}
    expected_sparsity = "2:4" if sparse24 else "none"
    if quant_config.get("format") != "mgx-native-w4a16-v1":
        raise SystemExit(
            f"Artifact {path} is not native MGX W4A16. Use --force-export to replace it."
        )
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
    args: argparse.Namespace,
    batch_size: int,
) -> dict[str, Any]:
    return _run_worker(
        label=label,
        model_ref=str(path),
        device=args.device,
        dtype_name=args.dtype,
        # MGX already owns its quantization. Passing none avoids invoking any
        # Hugging Face/AWQ quantizer and keeps this benchmark dependency-free.
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
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare standalone MGX dense W4A16 against native W4A16 + 2:4."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=_parse_batch_sizes("1,16,64"))
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--export-mode", choices=["normal", "streaming"], default="streaming")
    parser.add_argument("--kernel", choices=["triton", "auto", "torch"], default="triton")
    parser.add_argument("--prompt", default="Explain why compiled model artifacts reduce cold-start latency.")
    parser.add_argument("--first-tokens", type=int, default=8)
    parser.add_argument("--warm-tokens", type=int, default=128)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--kv-alloc", choices=["auto", "greedy"], default="auto")
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    if args.device == "cpu" and args.kernel == "triton":
        parser.error("--kernel triton requires --device cuda")
    if args.first_tokens <= 0 or args.warm_tokens <= 0 or args.warm_runs <= 0:
        parser.error("token and run counts must be positive")

    os.environ["MEGAGEMM_NATIVE_W4A16_KERNEL"] = args.kernel
    dense_path, sparse_path = _artifact_paths(
        args.model, Path(args.artifact_dir).expanduser().resolve()
    )
    dense_info, dense_export_seconds = _ensure_artifact(
        args.model,
        dense_path,
        dtype=args.dtype,
        sparse24=False,
        force_export=args.force_export,
        export_mode=args.export_mode,
    )
    sparse_info, sparse_export_seconds = _ensure_artifact(
        args.model,
        sparse_path,
        dtype=args.dtype,
        sparse24=True,
        force_export=args.force_export,
        export_mode=args.export_mode,
    )

    rows = []
    for batch_size in args.batch_sizes:
        print(f"[native W4A16] batch={batch_size}: measuring dense in an isolated worker...")
        dense = _measure(dense_path, label="native_w4a16_dense", args=args, batch_size=batch_size)
        print(f"[native W4A16] batch={batch_size}: measuring 2:4 in an isolated worker...")
        sparse = _measure(sparse_path, label="native_w4a16_sparse24", args=args, batch_size=batch_size)
        rows.append({
            "batch_size": batch_size,
            "dense": dense,
            "sparse24": sparse,
            "delta": _build_summary(dense, sparse),
            "memory_delta_dense_minus_sparse_mb": _build_memory_summary(dense, sparse),
        })

    print("\nFair comparison: native dense INT4 vs native INT4 + 2:4")
    print("batch | dense tok/s | 2:4 tok/s | TPS delta | dense VRAM MB | 2:4 VRAM MB")
    for row in rows:
        dense = row["dense"]
        sparse = row["sparse24"]
        delta = row["delta"].get("warmed_decode_tps_speedup_pct")
        print(
            f"{row['batch_size']:>5} | "
            f"{dense['warmed_decode_tokens_per_second']:>11.2f} | "
            f"{sparse['warmed_decode_tokens_per_second']:>9.2f} | "
            f"{delta:>+8.2f}% | "
            f"{dense.get('cuda_allocated_after_load_mb', 0.0):>13.1f} | "
            f"{sparse.get('cuda_allocated_after_load_mb', 0.0):>11.1f}"
        )

    result = {
        "comparison": "mgx-native-w4a16-dense-vs-mgx-native-w4a16-sparse24",
        "fairness": {
            "same_source_model": True,
            "same_quantizer": True,
            "same_group_size": True,
            "external_int4_backend": None,
            "experimental_variable": "2:4 pruning, packed storage, and native kernel",
        },
        "environment": _environment_metadata(args.device),
        "config": vars(args) | {"artifact_dir": str(Path(args.artifact_dir).resolve())},
        "artifacts": {
            "dense": {
                "path": str(dense_path),
                "file_size": dense_info["file_size"],
                "export_seconds": dense_export_seconds,
                "quantization_config": dense_info["manifest"].get("quantization_config"),
            },
            "sparse24": {
                "path": str(sparse_path),
                "file_size": sparse_info["file_size"],
                "export_seconds": sparse_export_seconds,
                "quantization_config": sparse_info["manifest"].get("quantization_config"),
            },
        },
        "results": rows,
    }
    # argparse's parsed list and Path-like values are normalized by default=str.
    payload = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    print(payload)
    if args.json_out:
        output = Path(args.json_out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + os.linesep, encoding="utf-8")
        print(f"Saved JSON to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

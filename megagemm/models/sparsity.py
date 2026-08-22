"""Portable MGX 2:4 structured-sparsity packing and runtime activation.

The on-disk representation is backend independent: each group of four values
along the input-feature (K) dimension stores the two magnitude-selected values
plus a four-bit code identifying their positions.  At load time MegaGemm can
either reconstruct the pruned dense weights or, on a supported CUDA device,
convert eligible ``nn.Linear`` weights to PyTorch's semi-structured sparse
tensor format.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch
import torch.nn as nn

try:
    from ..kernels.sparse24_gemv import (
        sparse24_cutlass_gemv,
        sparse24_cutlass_gemv_eligible,
        sparse24_triton_available,
        sparse24_triton_max_rows,
    )
except Exception:  # pragma: no cover - the regular torch backend remains valid
    sparse24_cutlass_gemv = None
    sparse24_cutlass_gemv_eligible = None
    sparse24_triton_available = lambda: False
    sparse24_triton_max_rows = lambda: 0

try:
    from ..kernels.sparse24_mma import (
        sparse24_mma_available,
        sparse24_mma_eligible,
        sparse24_mma_import_error,
        sparse24_mma_linear,
        sparse24_mma_linear_unchecked,
        sparse24_portable_metadata_to_ptx,
    )
except Exception:  # pragma: no cover - optional native CUDA extension
    sparse24_mma_available = lambda: False
    sparse24_mma_eligible = lambda *args, **kwargs: False
    sparse24_mma_import_error = lambda: "standalone sparse24 module unavailable"
    sparse24_mma_linear = None
    sparse24_mma_linear_unchecked = None
    sparse24_portable_metadata_to_ptx = None


SPARSE24_FORMAT = "2:4"
SPARSE24_STORAGE = "mgx-2to4-values-nibbles-v1"
SPARSE24_SHAPE_MULTIPLE = 64

_SPARSE24_PROJECTION_NAMES = {
    "qkv_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_up_proj",
    "down_proj",
}

_CODE_TO_POSITIONS = torch.tensor(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
    ],
    dtype=torch.long,
)
_POSITIONS_TO_CODE = torch.full((4, 4), -1, dtype=torch.int64)
for _code, (_left, _right) in enumerate(_CODE_TO_POSITIONS.tolist()):
    _POSITIONS_TO_CODE[_left, _right] = _code


_SPARSE24_ROUTE_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def normalize_sparsity_mode(value: Optional[str]) -> Optional[str]:
    """Normalize public aliases for structured sparsity."""
    if value is None:
        return None
    key = str(value).strip().lower().replace(" ", "")
    if key in {"", "none", "dense", "off", "0"}:
        return None
    if key in {"2:4", "2;4", "24", "sparse24", "sparse-24"}:
        return SPARSE24_FORMAT
    raise ValueError(
        f"Unsupported MGX sparsity mode '{value}'. Supported modes: none, 2:4."
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        return default


def _sparse24_kernel_mode() -> str:
    value = os.getenv("MEGAGEMM_MGX_SPARSE24_KERNEL", "auto").strip().lower()
    aliases = {
        "": "auto",
        "1": "native",
        "on": "native",
        "specialized": "native",
        "mma": "native",
        "mma.sp": "native",
        "tensorcore": "native",
        "tensor-core": "native",
        "0": "torch",
        "off": "torch",
        "pytorch": "torch",
    }
    value = aliases.get(value, value)
    return value if value in {"auto", "native", "triton", "torch"} else "auto"


def sparse24_weight_is_eligible(weight: torch.Tensor) -> bool:
    """Return whether a dense weight meets PyTorch 2:4 runtime constraints."""
    return bool(
        isinstance(weight, torch.Tensor)
        and weight.layout == torch.strided
        and weight.ndim == 2
        and weight.dtype in {torch.float16, torch.bfloat16}
        and int(weight.shape[0]) > 0
        and int(weight.shape[1]) > 0
        and int(weight.shape[0]) % SPARSE24_SHAPE_MULTIPLE == 0
        and int(weight.shape[1]) % SPARSE24_SHAPE_MULTIPLE == 0
    )


def pack_sparse24_weight(
    weight: torch.Tensor,
    *,
    chunk_rows: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Magnitude-prune and pack a 2-D FP16/BF16 weight into portable 2:4 data.

    Returns ``(values, metadata)``. Values have shape ``[N, K/2]``. Metadata
    stores two four-bit position codes per byte and has shape ``[N, K/8]``.
    """
    if weight.layout != torch.strided or weight.ndim != 2:
        raise ValueError("2:4 packing requires a dense 2-D tensor")
    if weight.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("2:4 packing supports only FP16/BF16 weights")
    rows, cols = (int(weight.shape[0]), int(weight.shape[1]))
    if cols <= 0 or cols % 8 != 0:
        raise ValueError("2:4 packed metadata requires the K dimension to be divisible by 8")

    chunk_rows = max(
        1,
        int(chunk_rows or _env_int("MEGAGEMM_MGX_SPARSE24_PACK_ROWS", 256)),
    )
    groups_per_row = cols // 4
    values = torch.empty((rows, cols // 2), dtype=weight.dtype, device=weight.device)
    metadata = torch.empty((rows, groups_per_row // 2), dtype=torch.uint8, device=weight.device)
    code_lut = _POSITIONS_TO_CODE.to(device=weight.device)

    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        groups = weight[start:end].reshape(end - start, groups_per_row, 4)
        positions = torch.topk(
            groups.abs(),
            k=2,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        positions = positions.sort(dim=-1).values
        kept = torch.gather(groups, dim=-1, index=positions)
        codes = code_lut[positions[..., 0], positions[..., 1]].to(torch.uint8)
        if bool((codes > 5).any().item()):
            raise RuntimeError("Internal 2:4 position-code generation failure")

        packed_codes = codes[..., 0::2] | (codes[..., 1::2] << 4)
        values[start:end].copy_(kept.reshape(end - start, cols // 2))
        metadata[start:end].copy_(packed_codes)

    return values.contiguous(), metadata.contiguous()


def unpack_sparse24_weight(
    values: torch.Tensor,
    metadata: torch.Tensor,
    shape: list[int] | tuple[int, int],
) -> torch.Tensor:
    """Reconstruct a dense, pruned tensor from the portable MGX 2:4 format."""
    if len(shape) != 2:
        raise ValueError("2:4 logical weight shape must have two dimensions")
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0 or cols % 8 != 0:
        raise ValueError(f"Invalid 2:4 logical weight shape: {(rows, cols)}")
    expected_values = (rows, cols // 2)
    expected_metadata = (rows, cols // 8)
    if tuple(values.shape) != expected_values:
        raise ValueError(
            f"Invalid 2:4 values shape {tuple(values.shape)}; expected {expected_values}"
        )
    if tuple(metadata.shape) != expected_metadata or metadata.dtype != torch.uint8:
        raise ValueError(
            f"Invalid 2:4 metadata {tuple(metadata.shape)}/{metadata.dtype}; "
            f"expected {expected_metadata}/uint8"
        )

    low = metadata & 0x0F
    high = (metadata >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(rows, cols // 4).long()
    if bool((codes > 5).any().item()):
        raise ValueError("MGX 2:4 metadata contains an invalid position code")
    position_lut = _CODE_TO_POSITIONS.to(device=values.device)
    positions = position_lut[codes]
    kept = values.reshape(rows, cols // 4, 2)
    groups = torch.zeros((rows, cols // 4, 4), dtype=values.dtype, device=values.device)
    groups.scatter_(dim=-1, index=positions, src=kept)
    return groups.reshape(rows, cols).contiguous()


def is_valid_sparse24_dense(weight: torch.Tensor) -> bool:
    """Check that each consecutive K-axis group of four has at most two nonzeros."""
    if weight.ndim != 2 or int(weight.shape[1]) % 4 != 0:
        return False
    groups = weight.reshape(int(weight.shape[0]), -1, 4)
    return bool(((groups != 0).sum(dim=-1) <= 2).all().item())


def _physical_keys(weight_name: str) -> tuple[str, str]:
    prefix = f"__mgx_sparse24__.{weight_name}"
    return f"{prefix}.values", f"{prefix}.metadata"


def sparse24_model_weight_names(model: nn.Module) -> list[str]:
    """Collect projection weights that are semantically safe for 2:4 conversion."""
    names: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not module_name:
            continue
        if module_name.rsplit(".", 1)[-1] in _SPARSE24_PROJECTION_NAMES:
            names.append(f"{module_name}.weight")
    return names


def pack_model_state_sparse24(
    model: Optional[nn.Module],
    state: dict[str, torch.Tensor],
    *,
    weight_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Replace eligible dense Linear weights in ``state`` with packed 2:4 tensors."""
    entries: list[dict[str, Any]] = []
    original_bytes = 0
    packed_bytes = 0
    skipped_shape = 0

    if weight_names is None:
        if model is None:
            raise ValueError("model or weight_names is required for MGX 2:4 packing")
        weight_names = sparse24_model_weight_names(model)

    for weight_name in sorted(set(weight_names)):
        module_name = weight_name[: -len(".weight")] if weight_name.endswith(".weight") else ""
        if (
            not module_name
            or module_name.rsplit(".", 1)[-1] not in _SPARSE24_PROJECTION_NAMES
        ):
            continue
        weight = state.get(weight_name)
        if weight is None:
            continue
        if not sparse24_weight_is_eligible(weight):
            skipped_shape += 1
            continue

        values_key, metadata_key = _physical_keys(weight_name)
        values, metadata = pack_sparse24_weight(weight)
        original_nbytes = int(weight.numel() * weight.element_size())
        packed_nbytes = int(values.numel() * values.element_size() + metadata.numel())
        original_bytes += original_nbytes
        packed_bytes += packed_nbytes
        del state[weight_name]
        state[values_key] = values
        state[metadata_key] = metadata
        entries.append(
            {
                "weight": weight_name,
                "values": values_key,
                "metadata": metadata_key,
                "shape": [int(weight.shape[0]), int(weight.shape[1])],
                "dtype": str(weight.dtype).replace("torch.", ""),
                "original_nbytes": original_nbytes,
                "packed_nbytes": packed_nbytes,
            }
        )

    if not entries:
        raise ValueError(
            "No eligible 2:4 Linear weights were found. FP16/BF16 weight dimensions "
            "must both be positive multiples of 64."
        )

    return {
        "format": SPARSE24_FORMAT,
        "storage": SPARSE24_STORAGE,
        "axis": "input_features",
        "group_size": 4,
        "kept_values": 2,
        "pruning": "magnitude",
        "runtime": "mgx-native-fp16-mma-sp-with-safe-fallbacks",
        "dense_fallback": True,
        "shape_multiple": SPARSE24_SHAPE_MULTIPLE,
        "tensor_count": len(entries),
        "skipped_shape_count": skipped_shape,
        "original_tensor_bytes": original_bytes,
        "packed_tensor_bytes": packed_bytes,
        "storage_ratio": packed_bytes / max(1, original_bytes),
        "requires_accuracy_validation": True,
        "entries": entries,
    }


def validate_sparse24_config(
    config: Optional[dict[str, Any]],
    *,
    tensor_names: Optional[set[str]] = None,
) -> None:
    if not config:
        return
    if config.get("format") != SPARSE24_FORMAT:
        raise ValueError(f"Unsupported MGX sparsity format: {config.get('format')}")
    if config.get("storage") != SPARSE24_STORAGE:
        raise ValueError(f"Unsupported MGX 2:4 storage layout: {config.get('storage')}")
    if int(config.get("group_size", 0)) != 4 or int(config.get("kept_values", 0)) != 2:
        raise ValueError("Invalid MGX 2:4 group contract")
    entries = config.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("MGX 2:4 manifest must contain at least one tensor entry")
    seen_weights: set[str] = set()
    seen_physical_keys: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("MGX 2:4 tensor entry must be an object")
        weight_name = str(entry.get("weight") or "")
        values_key = str(entry.get("values") or "")
        metadata_key = str(entry.get("metadata") or "")
        shape = entry.get("shape")
        if not weight_name or weight_name in seen_weights:
            raise ValueError(f"Invalid or duplicate MGX 2:4 weight entry: {weight_name!r}")
        seen_weights.add(weight_name)
        if (
            not values_key
            or not metadata_key
            or values_key == metadata_key
            or values_key in seen_physical_keys
            or metadata_key in seen_physical_keys
        ):
            raise ValueError(f"Invalid or duplicate MGX 2:4 payload keys for {weight_name}")
        seen_physical_keys.update((values_key, metadata_key))
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f"Invalid MGX 2:4 shape for {weight_name}")
        rows, cols = int(shape[0]), int(shape[1])
        if (
            rows <= 0
            or cols <= 0
            or rows % SPARSE24_SHAPE_MULTIPLE
            or cols % SPARSE24_SHAPE_MULTIPLE
        ):
            raise ValueError(f"MGX 2:4 shape for {weight_name} is not 64-aligned: {(rows, cols)}")
        if entry.get("dtype") not in {"float16", "bfloat16"}:
            raise ValueError(f"Invalid MGX 2:4 dtype for {weight_name}: {entry.get('dtype')}")
        if tensor_names is not None and (
            values_key not in tensor_names or metadata_key not in tensor_names
        ):
            raise ValueError(f"MGX 2:4 payload tensors are missing for {weight_name}")


def expand_sparse24_payload(
    payload: dict[str, torch.Tensor],
    config: Optional[dict[str, Any]],
) -> list[str]:
    """Expand portable packed entries into dense pruned model-state weights."""
    if not config:
        return []
    validate_sparse24_config(config, tensor_names=set(payload.keys()))
    expanded: list[str] = []
    for entry in config["entries"]:
        weight_name = str(entry["weight"])
        if weight_name in payload:
            raise ValueError(f"MGX 2:4 payload contains duplicate logical weight {weight_name}")
        values = payload.pop(str(entry["values"]))
        metadata = payload.pop(str(entry["metadata"]))
        payload[weight_name] = unpack_sparse24_weight(values, metadata, entry["shape"])
        expanded.append(weight_name)
    return expanded


def _resolve_module(root: nn.Module, module_name: str) -> nn.Module:
    module = root
    for part in module_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _sparse24_route_key(
    x: torch.Tensor,
    values: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> tuple[Any, ...]:
    rows = int(x.numel() // max(1, int(x.shape[-1])))
    device_index = int(x.device.index) if x.device.index is not None else 0
    capability = torch.cuda.get_device_capability(x.device) if x.is_cuda else (0, 0)
    return (
        device_index,
        int(capability[0]),
        int(capability[1]),
        str(x.dtype),
        rows,
        int(values.shape[1]) * 2,
        int(values.shape[0]),
        bias is not None,
    )


def _time_cuda_callable(callable_obj, *, iterations: int) -> float:
    for _ in range(2):
        callable_obj()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(max(1, int(iterations))):
        callable_obj()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / max(1, int(iterations))


def _select_sparse24_route(
    module: nn.Linear,
    x: torch.Tensor,
    values: torch.Tensor,
    metadata: torch.Tensor,
) -> dict[str, Any]:
    """Benchmark native mma.sp, compact Triton and PyTorch once per shape."""
    key = _sparse24_route_key(x, values, module.bias)
    cached = _SPARSE24_ROUTE_CACHE.get(key)
    if cached is not None:
        return cached

    iterations = max(2, _env_int("MEGAGEMM_MGX_SPARSE24_AUTOTUNE_ITERS", 5))
    result: dict[str, Any] = {
        "route": "torch",
        "rows": int(key[4]),
        "k": int(key[5]),
        "n": int(key[6]),
    }
    try:
        expected_shape = (*x.shape[:-1], int(values.shape[0]))
        reference = torch.nn.functional.linear(x, module.weight, module.bias)
        torch.cuda.synchronize()
        torch_ms = _time_cuda_callable(
            lambda: torch.nn.functional.linear(x, module.weight, module.bias),
            iterations=iterations,
        )
        minimum_speedup = max(
            1.0,
            float(os.getenv("MEGAGEMM_MGX_SPARSE24_MIN_SPEEDUP", "1.03")),
        )
        result.update(
            {
                "torch_ms": torch_ms,
                "minimum_speedup": minimum_speedup,
                "candidate_ms": {"torch": torch_ms},
                "correct_candidates": ["torch"],
            }
        )
        best_route = "torch"
        best_ms = torch_ms

        native_values = getattr(module, "_mgx_sparse24_native_values", None)
        native_metadata = getattr(module, "_mgx_sparse24_native_meta", None)
        if (
            callable(sparse24_mma_linear)
            and isinstance(native_values, torch.Tensor)
            and isinstance(native_metadata, torch.Tensor)
            and sparse24_mma_eligible(x, native_values, native_metadata)
        ):
            native_out = torch.empty(expected_shape, device=x.device, dtype=x.dtype)
            native = sparse24_mma_linear(
                x,
                native_values,
                native_metadata,
                module.bias,
                out=native_out,
            )
            torch.cuda.synchronize()
            if torch.allclose(native, reference, rtol=2e-2, atol=2e-2):
                native_ms = _time_cuda_callable(
                    lambda: sparse24_mma_linear(
                        x,
                        native_values,
                        native_metadata,
                        module.bias,
                        out=native_out,
                    ),
                    iterations=iterations,
                )
                result["native_mma_ms"] = native_ms
                result["candidate_ms"]["native-mma-sp"] = native_ms
                result["correct_candidates"].append("native-mma-sp")
                if native_ms < best_ms:
                    best_route, best_ms = "native-mma-sp", native_ms
            else:
                result["native_mma_rejected"] = "correctness check failed"

        if sparse24_cutlass_gemv_eligible(x, values, metadata):
            triton_out = torch.empty(expected_shape, device=x.device, dtype=x.dtype)
            triton_result = sparse24_cutlass_gemv(
                x,
                values,
                metadata,
                module.bias,
                out=triton_out,
            )
            torch.cuda.synchronize()
            if torch.allclose(triton_result, reference, rtol=2e-2, atol=2e-2):
                triton_ms = _time_cuda_callable(
                    lambda: sparse24_cutlass_gemv(
                        x,
                        values,
                        metadata,
                        module.bias,
                        out=triton_out,
                    ),
                    iterations=iterations,
                )
                result["triton_ms"] = triton_ms
                # Preserve the old key for tooling that already consumes it.
                result["specialized_ms"] = triton_ms
                result["candidate_ms"]["triton"] = triton_ms
                result["correct_candidates"].append("triton")
                if triton_ms < best_ms:
                    best_route, best_ms = "triton", triton_ms
            else:
                result["triton_rejected"] = "correctness check failed"

        result["measured_speedup"] = torch_ms / max(best_ms, 1e-12)
        if best_route != "torch" and best_ms * minimum_speedup <= torch_ms:
            result["route"] = best_route
            result["reason"] = f"{best_route} won correctness-gated autotune"
        else:
            result["route"] = "torch"
            result["reason"] = "PyTorch sparse kernel won correctness-gated autotune"
    except Exception as exc:
        result["reason"] = f"autotune failed: {exc}"

    _SPARSE24_ROUTE_CACHE[key] = result
    return result


def _sparse24_linear_forward(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Bound ``nn.Linear.forward`` used by MGX's specialized FP16 2:4 path."""
    stats = getattr(module, "_mgx_sparse24_runtime_stats", None)
    specialized_disabled = bool(getattr(module, "_mgx_sparse24_triton_disabled", False))
    native_disabled = bool(getattr(module, "_mgx_sparse24_native_disabled", False))
    values = getattr(module, "_mgx_sparse24_cutlass_values", None)
    metadata = getattr(module, "_mgx_sparse24_cutlass_meta", None)
    native_values = getattr(module, "_mgx_sparse24_native_values", None)
    native_metadata = getattr(module, "_mgx_sparse24_native_meta", None)

    kernel_mode = str(stats.get("kernel_mode", "auto")) if isinstance(stats, dict) else "auto"
    route: dict[str, Any] = {"route": "torch", "reason": "fallback"}
    if kernel_mode == "native":
        route = {"route": "native-mma-sp", "reason": "native mma.sp forced"}
    elif kernel_mode == "triton":
        route = {"route": "triton", "reason": "compact Triton forced"}
    elif kernel_mode == "auto" and isinstance(values, torch.Tensor) and isinstance(metadata, torch.Tensor):
        rows = int(x.numel() // x.shape[-1])
        module_routes = getattr(module, "_mgx_sparse24_routes", None)
        route = module_routes.get(rows) if isinstance(module_routes, dict) else None
        if route is None:
            route = _select_sparse24_route(module, x, values, metadata)
            if isinstance(module_routes, dict):
                module_routes[rows] = route
            if isinstance(stats, dict):
                signature = f"m{rows}_k{int(x.shape[-1])}_n{int(values.shape[0])}"
                stats.setdefault("dispatch_routes", {})[signature] = dict(route)

    if (
        route.get("route") == "native-mma-sp"
        and not native_disabled
        and callable(sparse24_mma_linear)
        and isinstance(native_values, torch.Tensor)
        and isinstance(native_metadata, torch.Tensor)
        and sparse24_mma_eligible(x, native_values, native_metadata)
    ):
        try:
            expected_shape = (*x.shape[:-1], int(native_values.shape[0]))
            output = getattr(module, "_mgx_sparse24_output", None)
            if (
                not isinstance(output, torch.Tensor)
                or tuple(output.shape) != tuple(expected_shape)
                or output.device != x.device
                or output.dtype != x.dtype
            ):
                output = torch.empty(expected_shape, device=x.device, dtype=x.dtype)
                module._mgx_sparse24_output = output
            native_call = (
                sparse24_mma_linear_unchecked
                if callable(sparse24_mma_linear_unchecked) and x.ndim == 2
                else sparse24_mma_linear
            )
            result = native_call(
                x,
                native_values,
                native_metadata,
                module.bias,
                out=output,
            )
            if isinstance(stats, dict):
                stats["native_mma_kernel_hits"] += 1
            return result
        except Exception as exc:
            module._mgx_sparse24_native_disabled = True
            if isinstance(stats, dict):
                stats["native_mma_kernel_failures"] += 1
                failures = stats.setdefault("native_mma_failures", [])
                if isinstance(failures, list) and len(failures) < 8:
                    failures.append(
                        {
                            "shape": [
                                int(native_values.shape[0]),
                                int(native_values.shape[1]) * 2,
                            ],
                            "error": str(exc),
                        }
                    )

    can_try_specialized = bool(
        not specialized_disabled
        and callable(sparse24_cutlass_gemv)
        and callable(sparse24_cutlass_gemv_eligible)
        and isinstance(values, torch.Tensor)
        and isinstance(metadata, torch.Tensor)
        and sparse24_cutlass_gemv_eligible(x, values, metadata)
    )
    if can_try_specialized and route.get("route") == "triton":
        try:
            expected_shape = (*x.shape[:-1], int(values.shape[0]))
            output = getattr(module, "_mgx_sparse24_output", None)
            if (
                not isinstance(output, torch.Tensor)
                or tuple(output.shape) != tuple(expected_shape)
                or output.device != x.device
                or output.dtype != x.dtype
            ):
                output = torch.empty(expected_shape, device=x.device, dtype=x.dtype)
                module._mgx_sparse24_output = output
            result = sparse24_cutlass_gemv(
                x,
                values,
                metadata,
                module.bias,
                out=output,
            )
            if isinstance(stats, dict):
                stats["specialized_kernel_hits"] += 1
            return result
        except Exception as exc:
            # Compilation/backend incompatibilities are sticky per projection so
            # a production request never pays the same failed attempt repeatedly.
            module._mgx_sparse24_triton_disabled = True
            if isinstance(stats, dict):
                stats["specialized_kernel_failures"] += 1
                failures = stats.setdefault("specialized_failures", [])
                if isinstance(failures, list) and len(failures) < 8:
                    failures.append(
                        {
                            "shape": [int(values.shape[0]), int(values.shape[1]) * 2],
                            "error": str(exc),
                        }
                    )

    if isinstance(stats, dict):
        stats["torch_sparse_fallback_hits"] += 1
    return torch.nn.functional.linear(x, module.weight, module.bias)


def _install_sparse24_specialized_forward(
    module: nn.Linear,
    sparse_weight: torch.Tensor,
    stats: dict[str, Any],
    *,
    native_values: Optional[torch.Tensor] = None,
    native_metadata: Optional[torch.Tensor] = None,
) -> bool:
    """Attach zero-copy CUTLASS/native payload views and specialized forward."""
    if "CUTLASS" not in type(sparse_weight).__name__.upper():
        return False
    packed = getattr(sparse_weight, "packed", None)
    metadata = getattr(sparse_weight, "meta", None)
    if not isinstance(packed, torch.Tensor) or not isinstance(metadata, torch.Tensor):
        return False
    try:
        # ``values()`` is a view over CUTLASS' packed storage, not a duplicate.
        values = sparse_weight.values()
    except Exception:
        return False
    if values.ndim != 2 or metadata.ndim != 2 or metadata.dtype != torch.int16:
        return False

    module._mgx_sparse24_cutlass_values = values
    module._mgx_sparse24_cutlass_meta = metadata
    if isinstance(native_values, torch.Tensor) and isinstance(native_metadata, torch.Tensor):
        module._mgx_sparse24_native_values = native_values
        module._mgx_sparse24_native_meta = native_metadata
    module._mgx_sparse24_runtime_stats = stats
    module._mgx_sparse24_triton_disabled = False
    module._mgx_sparse24_native_disabled = False
    module._mgx_sparse24_routes = {}
    module.forward = _sparse24_linear_forward.__get__(module, type(module))
    return True


def prepare_sparse24_runtime(
    model: nn.Module,
    config: Optional[dict[str, Any]],
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    """Convert reconstructed dense weights to CUDA semi-structured tensors.

    Conversion is best effort. A disabled/unsupported runtime keeps the same
    pruned dense weights, which makes A/B benchmarking possible with one MGX.
    """
    requested_count = len((config or {}).get("entries") or [])
    stats: dict[str, Any] = {
        "format": (config or {}).get("format", "none"),
        "requested_tensor_count": requested_count,
        "prepared_tensor_count": 0,
        "failed_tensor_count": 0,
        "active": False,
        "backend": "dense-fallback",
        "failures": [],
        "kernel_mode": _sparse24_kernel_mode(),
        "specialized_kernel": "mgx-triton-cutlass-gemv-v1",
        "specialized_kernel_available": bool(sparse24_triton_available()),
        "specialized_max_rows": int(sparse24_triton_max_rows()),
        "specialized_tensor_count": 0,
        "specialized_kernel_hits": 0,
        "specialized_kernel_failures": 0,
        "specialized_failures": [],
        "torch_sparse_fallback_hits": 0,
        "torch_backend_counts": {},
        "dispatch_routes": {},
        "native_mma_kernel": "mgx-fp16-mma-sp-m16n8k32-v3-hybrid-flat",
        "native_mma_kernel_available": bool(sparse24_mma_available()),
        "native_mma_import_error": str(sparse24_mma_import_error()),
        "native_mma_tensor_count": 0,
        "native_mma_kernel_hits": 0,
        "native_mma_kernel_failures": 0,
        "native_mma_failures": [],
    }
    model._mgx_sparsity_config = config or {}
    model._mgx_sparsity_runtime = stats
    if not config:
        stats["reason"] = "artifact is dense"
        return stats
    if not _env_bool("MEGAGEMM_MGX_SPARSE24_RUNTIME", True):
        stats["reason"] = "disabled by MEGAGEMM_MGX_SPARSE24_RUNTIME"
        return stats

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        stats["reason"] = "CUDA is unavailable"
        return stats
    capability = torch.cuda.get_device_capability(target)
    stats["compute_capability"] = [int(capability[0]), int(capability[1])]
    if capability < (8, 0):
        stats["reason"] = f"compute capability {capability[0]}.{capability[1]} is below 8.0"
        return stats

    try:
        from torch.sparse import to_sparse_semi_structured
    except Exception as exc:
        stats["reason"] = f"PyTorch semi-structured API unavailable: {exc}"
        return stats

    start = time.perf_counter()
    kernel_mode = str(stats["kernel_mode"])
    for entry in config.get("entries") or []:
        weight_name = str(entry["weight"])
        module_name = weight_name[: -len(".weight")]
        try:
            module = _resolve_module(model, module_name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target module is {type(module).__name__}, not nn.Linear")
            dense_weight = module.weight.detach()
            if not sparse24_weight_is_eligible(dense_weight):
                raise ValueError(
                    f"runtime weight shape/dtype is ineligible: {tuple(dense_weight.shape)}/{dense_weight.dtype}"
                )
            sparse_weight = None
            native_values = None
            native_metadata = None
            portable_values = None
            if (
                kernel_mode in {"auto", "native"}
                and bool(sparse24_mma_available())
                and dense_weight.dtype == torch.float16
                and callable(sparse24_portable_metadata_to_ptx)
            ):
                portable_values, portable_metadata = pack_sparse24_weight(dense_weight)
                native_metadata = sparse24_portable_metadata_to_ptx(portable_metadata)
            # The MGX Triton GEMV consumes CUTLASS' packed values and reordered
            # metadata directly.  Construct that backend explicitly when the
            # specialized path is requested; fall back to PyTorch's public
            # selector if the concrete class is unavailable on this build/GPU.
            if kernel_mode != "torch" and (
                bool(sparse24_triton_available()) or bool(sparse24_mma_available())
            ):
                cutlass_cls = getattr(torch.sparse, "SparseSemiStructuredTensorCUTLASS", None)
                if cutlass_cls is not None:
                    try:
                        sparse_weight = cutlass_cls.from_dense(dense_weight)
                    except Exception:
                        sparse_weight = None
            if sparse_weight is None:
                sparse_weight = to_sparse_semi_structured(dense_weight)
            try:
                sparse_weight._mgx_sparse24 = True
            except Exception:
                pass
            if portable_values is not None and native_metadata is not None:
                try:
                    cutlass_values = sparse_weight.values()
                    if torch.equal(cutlass_values, portable_values):
                        # Reuse CUTLASS' packed-value storage.  Only the tiny PTX
                        # metadata view is additional; the FP16 values are never
                        # duplicated in steady-state VRAM.
                        native_values = cutlass_values
                    else:
                        native_metadata = None
                        failures = stats.setdefault("native_mma_failures", [])
                        if len(failures) < 8:
                            failures.append(
                                {
                                    "weight": weight_name,
                                    "error": "CUTLASS and portable packed value order differ",
                                }
                            )
                finally:
                    del portable_values
            module.weight = nn.Parameter(sparse_weight, requires_grad=False)
            stats["prepared_tensor_count"] += 1
            backend_name = type(module.weight).__name__
            backend_counts = stats["torch_backend_counts"]
            backend_counts[backend_name] = int(backend_counts.get(backend_name, 0)) + 1
            if kernel_mode != "torch" and _install_sparse24_specialized_forward(
                module,
                module.weight,
                stats,
                native_values=native_values,
                native_metadata=native_metadata,
            ):
                stats["specialized_tensor_count"] += 1
                if native_values is not None and native_metadata is not None:
                    stats["native_mma_tensor_count"] += 1
        except Exception as exc:
            stats["failed_tensor_count"] += 1
            if len(stats["failures"]) < 8:
                stats["failures"].append({"weight": weight_name, "error": str(exc)})

    stats["prepare_seconds"] = time.perf_counter() - start
    stats["active"] = stats["prepared_tensor_count"] > 0
    if stats["active"]:
        if stats["native_mma_tensor_count"]:
            stats["backend"] = "mgx-native-fp16-mma-sp+torch-semi-structured"
        elif stats["specialized_tensor_count"]:
            stats["backend"] = "mgx-triton-cutlass+torch-semi-structured"
        else:
            stats["backend"] = "torch-semi-structured"
        stats["reason"] = "active"
        fully_native_flat = bool(
            kernel_mode == "native"
            and requested_count > 0
            and stats["native_mma_tensor_count"] == requested_count
            and stats["prepared_tensor_count"] == requested_count
            and stats["failed_tensor_count"] == 0
        )
        stats["flat_decode_native_eligible"] = fully_native_flat
        stats["flat_decode_native_mma_hits"] = 0
        if not fully_native_flat:
            # Dense flat decode consumes transposed strided weights.  Only the
            # fully-native path below has explicit packed-value/meta support;
            # partial/torch sparse preparations retain the safe module path.
            model._flat_decode_failed = True
            model._flat_decode_ready = False
            model._flat_decode_failed_reason = (
                "MGX 2:4 flat decode requires every projection to use native mma.sp"
            )
    elif stats["failed_tensor_count"]:
        stats["reason"] = "all runtime conversions failed"
    else:
        stats["reason"] = "no eligible runtime weights"
    return stats


def is_semi_structured_weight(weight: Any) -> bool:
    if not isinstance(weight, torch.Tensor):
        return False
    if bool(getattr(weight, "_mgx_sparse24", False)):
        return True
    return "SparseSemiStructuredTensor" in type(weight).__name__

"""
MGX compiled artifact support for MegaGemm.

MGX v1 packages the already materialized MegaGemm runtime weights so serving
can skip Hugging Face snapshot parsing, weight fusion, and runtime INT8
quantization on the cold path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from .llama import LlamaConfig, MegaGemmLlama
from .loader import _get_awq_config, _is_awq_model, _load_config, load_from_hf, resolve_model_source
from .sparsity import (
    expand_sparse24_payload,
    normalize_sparsity_mode,
    pack_model_state_sparse24,
    prepare_sparse24_runtime,
    sparse24_model_weight_names,
    unpack_sparse24_weight,
    validate_sparse24_config,
)
from ..kernels.rope import precompute_freqs_cis
from ..quantization.native_w4a16 import (
    DEFAULT_GROUP_SIZE as NATIVE_W4A16_GROUP_SIZE,
    NATIVE_W4A16_FORMAT,
    NATIVE_W4A16_SPARSE24_FORMAT,
    NativeW4A16Linear,
    is_native_w4a16_config,
    native_w4a16_manifest_config,
    native_w4a16_runtime_stats,
    quantize_model_native_w4a16,
)
from ..quantization.w4a16 import QuantizedLinear
from ..quantization.w8a16 import Int8Linear

__all__ = [
    "MGXFormatError",
    "attach_session_state_to_mgx",
    "extract_session_state_from_mgx",
    "export_to_mgx",
    "inspect_mgx",
    "is_mgx_path",
    "load_from_mgx",
    "prime_mgx_payload_cache",
]


MGX_MAGIC = b"MGX1"
MGX_VERSION_MAJOR = 1
MGX_VERSION_MINOR = 1
MGX_HEADER_SIZE = 128
MGX_ALIGNMENT = 64
_HEADER_STRUCT = struct.Struct("<4sIIIIQQQQQQQQ")

MGX_SESSION_MAGIC = b"MGXS"
MGX_SESSION_VERSION_MAJOR = 1
MGX_SESSION_VERSION_MINOR = 0
MGX_SESSION_HEADER_SIZE = 64
_SESSION_HEADER_STRUCT = struct.Struct("<4sIIIIQQQQ")

_SUPPORTED_BACKENDS = {"megagemm-cuda", "megagemm-cpu", "megagemm"}
_SUPPORTED_EXPORT_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
_SUPPORTED_RUNTIME_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_AWQ_QWEIGHT_LAYOUT_LEGACY = "awq-packed-k-major-v0"
_AWQ_QWEIGHT_LAYOUT_DECODE = "awq-decode-transposed-v1"
_SUPPORTED_AWQ_QWEIGHT_LAYOUTS = {
    _AWQ_QWEIGHT_LAYOUT_LEGACY,
    _AWQ_QWEIGHT_LAYOUT_DECODE,
}
_SUPPORTED_EXPORT_MODES = {"normal", "streaming"}

_MGX_SUPPORTED_LAYER_TYPES = {"full_attention", "sliding_attention"}
_SUPPORTED_PAYLOAD_HYDRATION_MODES = {
    "auto",
    "direct_device",
    "cpu_stage_streaming",
    "cpu_stage_pinned",
    "cpu_bulk",
    "gpu_bulk",
}
_RUNTIME_PAYLOAD_CACHE_FORMAT = "mgx-runtime-packed-v1"
_RUNTIME_PAYLOAD_CACHE_VERSION = 2
_RUNTIME_PAYLOAD_CACHE_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


@dataclass(slots=True)
class _AssignmentSlot:
    module: nn.Module
    field: str
    kind: str
    clear_runtime_caches: bool
    requires_grad: bool = False


class MGXFormatError(ValueError):
    """Raised when an MGX artifact is invalid or incompatible."""


def is_mgx_path(path: str | os.PathLike[str]) -> bool:
    return str(path).lower().endswith(".mgx")


def _align_up(value: int, alignment: int = MGX_ALIGNMENT) -> int:
    remainder = value % alignment
    if remainder == 0:
        return value
    return value + (alignment - remainder)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file_set(root: Path, files: list[Path], block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(block_size)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    raise MGXFormatError(f"Unsupported dtype for MGX: {dtype}")


def _unsupported_layer_types_for_mgx(layer_types: list[str] | tuple[str, ...] | None) -> list[str]:
    if not layer_types:
        return []
    return sorted({str(layer) for layer in layer_types if str(layer) not in _MGX_SUPPORTED_LAYER_TYPES})


def _mgx_compat_flags_for_layer_types(layer_types: list[str] | tuple[str, ...] | None) -> list[str]:
    flags = [
        "decoder-only",
        "mgx-v1-no-linear-attention",
        "runtime-save-context-separate",
    ]
    if layer_types and any(str(layer) == "sliding_attention" for layer in layer_types):
        flags.append("hybrid-sliding-attention")
    else:
        flags.append("full-attention-only")
    return flags


def _normalize_export_mode(export_mode: Optional[str]) -> str:
    mode = str(export_mode or os.environ.get("MEGAGEMM_MGX_EXPORT_MODE", "streaming")).strip().lower()
    if mode not in _SUPPORTED_EXPORT_MODES:
        raise MGXFormatError(
            f"Unsupported MGX export mode '{mode}'. "
            f"Supported modes: {sorted(_SUPPORTED_EXPORT_MODES)}."
        )
    return mode


def _canonical_dtype(dtype: Optional[torch.dtype | str], *, allow_fp32: bool = True) -> torch.dtype:
    if dtype is None:
        return torch.float16
    if isinstance(dtype, torch.dtype):
        if dtype in (torch.float16, torch.bfloat16):
            return dtype
        if allow_fp32 and dtype == torch.float32:
            return dtype
        raise MGXFormatError(f"Unsupported dtype for MGX: {dtype}")
    key = str(dtype).strip().lower()
    if key in _SUPPORTED_RUNTIME_DTYPES and (allow_fp32 or key != "fp32"):
        return _SUPPORTED_RUNTIME_DTYPES[key]
    raise MGXFormatError(f"Unsupported dtype for MGX: {dtype}")


def _normalize_quantize_mode(quantize: Optional[str]) -> Optional[str]:
    if quantize in (None, "", "none"):
        return None
    key = str(quantize).strip().lower()
    if key == "awq":
        return "int4"
    if key in {"native-int4", "int4-native", "w4a16"}:
        return "native-int4"
    if key in {"int8", "int4"}:
        return key
    raise MGXFormatError(f"Unsupported MGX quantization mode: {quantize}")


def _is_native_w4_sparse_config(config: Optional[dict[str, Any]]) -> bool:
    return bool(config and config.get("format") == NATIVE_W4A16_SPARSE24_FORMAT)


def _native_w4_sparse_config(export_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": NATIVE_W4A16_SPARSE24_FORMAT,
        "storage": "signed-int4-pairs-plus-packed-position-nibbles",
        "axis": "input_features",
        "group_size": 4,
        "kept_values": 2,
        "pruning": "magnitude-before-quantization",
        "runtime": "mgx-native-triton",
        "dense_fallback": True,
        "tensor_count": int(export_meta.get("native_w4a16_module_count", 0)),
        "original_tensor_bytes": int(export_meta.get("native_w4a16_original_weight_bytes", 0)),
        "packed_tensor_bytes": int(export_meta.get("native_w4a16_packed_weight_bytes", 0)),
        "module_names": list(export_meta.get("native_w4a16_module_names", [])),
    }


def _normalize_sparsity_mode(sparsity: Optional[str]) -> Optional[str]:
    try:
        return normalize_sparsity_mode(sparsity)
    except ValueError as exc:
        raise MGXFormatError(str(exc)) from exc


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return _read_json(path)
    except Exception:
        return {}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    raise MGXFormatError(
        f"MGX session state metadata contains an unsupported JSON value of type {type(value).__name__}."
    )


def _candidate_tokenizer_files(model_dir: Path) -> list[Path]:
    names = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "vocab.txt",
        "sentencepiece.bpe.model",
        "chat_template.jinja",
    ]
    files: list[Path] = []
    for name in names:
        path = model_dir / name
        if path.exists() and path.is_file():
            files.append(path)
    return files


def _candidate_core_tokenizer_files(model_dir: Path) -> list[Path]:
    groups = [
        ["tokenizer.json"],
        ["tokenizer.model"],
        ["sentencepiece.bpe.model"],
        ["vocab.json", "merges.txt"],
        ["vocab.txt"],
    ]
    for group in groups:
        files = [model_dir / name for name in group]
        if all(path.exists() and path.is_file() for path in files):
            return files
    return []


def _read_chat_template(model_dir: Path) -> Optional[str]:
    template_path = model_dir / "chat_template.jinja"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    tokenizer_cfg = model_dir / "tokenizer_config.json"
    if tokenizer_cfg.exists():
        data = _read_json(tokenizer_cfg)
        template = data.get("chat_template")
        if template:
            return str(template)
    return None


def _compute_tokenizer_bundle_hash(model_dir: Path) -> Optional[str]:
    files = _candidate_tokenizer_files(model_dir)
    if not files:
        return None
    return _sha256_file_set(model_dir, files)


def _compute_tokenizer_hash(model_dir: Path) -> Optional[str]:
    """
    Stable tokenizer hash for MGX compatibility checks.

    The core tokenizer artifact is hashed separately from chat template metadata.
    This is less fragile than hashing the full tokenizer sidecar bundle, which may
    vary across Hugging Face cache layouts for some models.
    """
    core_files = _candidate_core_tokenizer_files(model_dir)
    if core_files:
        return _sha256_file_set(model_dir, core_files)
    return _compute_tokenizer_bundle_hash(model_dir)


def _collect_tokenizer_hashes(model_dir: Path) -> dict[str, Optional[str]]:
    core_hash = None
    core_files = _candidate_core_tokenizer_files(model_dir)
    if core_files:
        core_hash = _sha256_file_set(model_dir, core_files)
    bundle_hash = _compute_tokenizer_bundle_hash(model_dir)
    primary_hash = core_hash or bundle_hash
    primary_scheme = "core-v1" if core_hash is not None else ("bundle-v0" if bundle_hash is not None else None)
    return {
        "primary": primary_hash,
        "primary_scheme": primary_scheme,
        "core": core_hash,
        "bundle": bundle_hash,
    }


def _compute_source_model_hash(model_dir: Path) -> str:
    files = [model_dir / "config.json"]
    files.extend(sorted(model_dir.rglob("*.safetensors")))
    existing = [path for path in files if path.exists() and path.is_file()]
    if not existing:
        raise MGXFormatError(f"Cannot hash model source; no config/safetensors found in {model_dir}")
    return _sha256_file_set(model_dir, existing)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _payload_cache_dir_for_artifact(
    artifact_path: Path,
    cache_dir: Optional[str | os.PathLike[str]] = None,
) -> Path:
    if cache_dir is None:
        return artifact_path.parent / ".mgx-payload-cache"
    return Path(cache_dir).expanduser().resolve()


def _payload_cache_path_for_artifact(
    artifact_path: Path,
    manifest: dict[str, Any],
    cache_dir: Optional[str | os.PathLike[str]] = None,
) -> Path:
    payload_sha = manifest.get("tensor_payload_sha256")
    if not payload_sha:
        raise MGXFormatError("MGX manifest is missing tensor_payload_sha256; cannot derive payload cache path.")
    cache_root = _payload_cache_dir_for_artifact(artifact_path, cache_dir)
    safe_stem = artifact_path.stem.replace(" ", "_")
    return cache_root / f"{safe_stem}-{payload_sha[:16]}.safetensors"


def _payload_runtime_cache_path_for_artifact(
    artifact_path: Path,
    manifest: dict[str, Any],
    cache_dir: Optional[str | os.PathLike[str]] = None,
) -> Path:
    payload_sha = manifest.get("tensor_payload_sha256")
    if not payload_sha:
        raise MGXFormatError(
            "MGX manifest is missing tensor_payload_sha256; cannot derive runtime payload cache path."
        )
    cache_root = _payload_cache_dir_for_artifact(artifact_path, cache_dir)
    safe_stem = artifact_path.stem.replace(" ", "_")
    return cache_root / f"{safe_stem}-{payload_sha[:16]}.runtime-v1.safetensors"


def _payload_runtime_index_path_for_artifact(
    artifact_path: Path,
    manifest: dict[str, Any],
    cache_dir: Optional[str | os.PathLike[str]] = None,
) -> Path:
    payload_sha = manifest.get("tensor_payload_sha256")
    if not payload_sha:
        raise MGXFormatError(
            "MGX manifest is missing tensor_payload_sha256; cannot derive runtime payload cache index path."
        )
    cache_root = _payload_cache_dir_for_artifact(artifact_path, cache_dir)
    safe_stem = artifact_path.stem.replace(" ", "_")
    return cache_root / f"{safe_stem}-{payload_sha[:16]}.runtime-v1.json"


def _payload_cache_is_valid(path: Path, expected_size: int) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size == expected_size


def _write_payload_cache_file(payload_cache_path: Path, payload_bytes: bytes) -> None:
    payload_cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = payload_cache_path.with_suffix(payload_cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        fh.write(payload_bytes)
    os.replace(tmp_path, payload_cache_path)


def _runtime_cache_blob_name(dtype_name: str) -> str:
    return f"blob.{dtype_name.replace('.', '_')}"


def _dtype_from_runtime_cache_name(dtype_name: str) -> torch.dtype:
    try:
        return _RUNTIME_PAYLOAD_CACHE_DTYPE_MAP[str(dtype_name)]
    except KeyError as exc:
        raise MGXFormatError(f"Unsupported runtime payload cache dtype '{dtype_name}'.") from exc


def _write_runtime_payload_cache_index(path: Path, index: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp_path, path)


def _load_runtime_payload_cache_index(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise MGXFormatError(f"Invalid runtime payload cache index at {path}: expected JSON object.")
    return data


def _runtime_payload_cache_is_valid(
    payload_cache_path: Path,
    index_path: Path,
    *,
    expected_payload_sha256: Optional[str],
    expected_tensor_count: Optional[int] = None,
) -> bool:
    if not payload_cache_path.exists() or not payload_cache_path.is_file():
        return False
    if not index_path.exists() or not index_path.is_file():
        return False
    try:
        index = _load_runtime_payload_cache_index(index_path)
    except Exception:
        return False
    if index.get("format") != _RUNTIME_PAYLOAD_CACHE_FORMAT:
        return False
    if int(index.get("version", -1)) != _RUNTIME_PAYLOAD_CACHE_VERSION:
        return False
    if expected_payload_sha256 and index.get("payload_sha256") != expected_payload_sha256:
        return False
    if expected_tensor_count is not None and int(index.get("tensor_count", -1)) != int(expected_tensor_count):
        return False
    entries = index.get("entries")
    blobs = index.get("blobs")
    if not isinstance(entries, list) or not entries:
        return False
    if not isinstance(blobs, list) or not blobs:
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        offset_bytes = entry.get("offset_bytes")
        if offset_bytes is None or int(offset_bytes) % MGX_ALIGNMENT != 0:
            return False
    return True


def _build_runtime_payload_cache_from_payload_cache(
    payload_cache_path: Path,
    runtime_cache_path: Path,
    runtime_index_path: Path,
    *,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensor_table = manifest.get("tensor_table") or []
    if not tensor_table:
        raise MGXFormatError("MGX manifest is missing tensor_table; cannot build runtime payload cache.")

    totals_by_blob: dict[str, int] = {}
    dtypes_by_blob: dict[str, torch.dtype] = {}
    counts_by_blob: dict[str, int] = {}
    for entry in tensor_table:
        dtype_name = str(entry["dtype"])
        blob_name = _runtime_cache_blob_name(dtype_name)
        dtypes_by_blob[blob_name] = _dtype_from_runtime_cache_name(dtype_name)
        counts_by_blob[blob_name] = counts_by_blob.get(blob_name, 0) + 1

    blob_bytes_used = {blob_name: 0 for blob_name in dtypes_by_blob}
    for entry in tensor_table:
        dtype_name = str(entry["dtype"])
        blob_name = _runtime_cache_blob_name(dtype_name)
        dtype = dtypes_by_blob[blob_name]
        element_size = torch.empty((), dtype=dtype).element_size()
        start_byte = _align_up(blob_bytes_used[blob_name], MGX_ALIGNMENT)
        tensor_numel = int(entry["numel"])
        end_byte = start_byte + (tensor_numel * element_size)
        blob_bytes_used[blob_name] = end_byte
    for blob_name, dtype in dtypes_by_blob.items():
        element_size = torch.empty((), dtype=dtype).element_size()
        totals_by_blob[blob_name] = blob_bytes_used[blob_name] // element_size

    blob_tensors: dict[str, torch.Tensor] = {
        blob_name: torch.empty(total_numel, dtype=dtypes_by_blob[blob_name])
        for blob_name, total_numel in totals_by_blob.items()
    }
    blob_offsets_bytes = {blob_name: 0 for blob_name in blob_tensors}
    index_entries: list[dict[str, Any]] = []

    with safe_open(str(payload_cache_path), framework="pt", device="cpu") as handle:
        for entry in tensor_table:
            key = str(entry["name"])
            dtype_name = str(entry["dtype"])
            blob_name = _runtime_cache_blob_name(dtype_name)
            dtype = dtypes_by_blob[blob_name]
            element_size = torch.empty((), dtype=dtype).element_size()
            expected_numel = int(entry["numel"])
            tensor = handle.get_tensor(key)
            if int(tensor.numel()) != expected_numel:
                raise MGXFormatError(
                    f"Runtime payload cache build found tensor '{key}' with unexpected numel "
                    f"{tensor.numel()} (expected {expected_numel})."
                )
            start_byte = _align_up(blob_offsets_bytes[blob_name], MGX_ALIGNMENT)
            start = start_byte // element_size
            end = start + expected_numel
            blob_tensors[blob_name][start:end].copy_(tensor.reshape(-1))
            blob_offsets_bytes[blob_name] = start_byte + (expected_numel * element_size)
            index_entries.append(
                {
                    "key": key,
                    "blob": blob_name,
                    "offset": start,
                    "offset_bytes": start_byte,
                    "numel": expected_numel,
                    "shape": list(entry["shape"]),
                    "dtype": dtype_name,
                }
            )

    runtime_cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_runtime_cache_path = runtime_cache_path.with_suffix(runtime_cache_path.suffix + ".tmp")
    save_file(blob_tensors, str(tmp_runtime_cache_path))
    os.replace(tmp_runtime_cache_path, runtime_cache_path)

    runtime_index = {
        "format": _RUNTIME_PAYLOAD_CACHE_FORMAT,
        "version": _RUNTIME_PAYLOAD_CACHE_VERSION,
        "payload_sha256": manifest.get("tensor_payload_sha256"),
        "tensor_count": len(index_entries),
        "tensor_bytes": int(manifest.get("tensor_bytes", 0)),
        "blobs": [
            {
                "name": blob_name,
                "dtype": str(blob_tensors[blob_name].dtype).replace("torch.", ""),
                "numel": int(blob_tensors[blob_name].numel()),
                "nbytes": _tensor_num_bytes(blob_tensors[blob_name]),
                "tensor_count": int(counts_by_blob.get(blob_name, 0)),
            }
            for blob_name in sorted(blob_tensors.keys())
        ],
        "entries": index_entries,
    }
    _write_runtime_payload_cache_index(runtime_index_path, runtime_index)
    return {
        "runtime_cache_path": str(runtime_cache_path),
        "runtime_cache_index_path": str(runtime_index_path),
        "runtime_cache_tensor_count": len(index_entries),
        "runtime_cache_blob_count": len(blob_tensors),
    }


def _validate_export_request(
    config: LlamaConfig,
    hf_config: dict[str, Any],
    quantize: Optional[str],
    target_backend: str,
) -> Optional[str]:
    if target_backend not in _SUPPORTED_BACKENDS:
        raise MGXFormatError(
            f"Unsupported MGX backend '{target_backend}'. Supported backends: {sorted(_SUPPORTED_BACKENDS)}"
        )
    requested_quant = _normalize_quantize_mode(quantize)
    is_awq = _is_awq_model(hf_config)
    if is_awq:
        if requested_quant not in {None, "int4"}:
            raise NotImplementedError(
                "An AWQ source must use quantize=None/'int4'. Native W4A16 export starts "
                "from an FP16/BF16 checkpoint instead."
            )
        resolved_quant = "int4"
    else:
        if requested_quant not in {None, "int8", "int4", "native-int4"}:
            raise NotImplementedError(
                "MGX supports quantize=None, 'int8', 'int4', or 'native-int4'."
            )
        # For a floating-point checkpoint, public `int4` now resolves to the
        # standalone MGX quantizer. AWQ checkpoints retain their legacy path.
        resolved_quant = "native-int4" if requested_quant in {"int4", "native-int4"} else requested_quant
    if config.model_type == "qwen3_moe" and resolved_quant == "native-int4":
        raise NotImplementedError(
            "Native MGX W4A16 does not yet pack Qwen3 MoE expert tensors. "
            "Dense decoder models are supported in this version."
        )
    if config.model_type == "qwen3_moe" and resolved_quant not in {None, "int8", "int4"}:
        raise NotImplementedError(
            "MGX export for Qwen3 MoE currently supports quantize=None, 'int8', "
            "or AWQ/'int4' fallback artifacts only."
        )
    unsupported_layer_types = _unsupported_layer_types_for_mgx(config.layer_types)
    if unsupported_layer_types:
        raise NotImplementedError(
            "MGX v1 supports decoder-only models with "
            f"{sorted(_MGX_SUPPORTED_LAYER_TYPES)} layers only. "
            f"Unsupported layer types: {unsupported_layer_types}."
        )
    return resolved_quant


def _quantization_config_for_manifest(
    hf_config: dict[str, Any],
    quantize: Optional[str],
    state_export_meta: dict[str, Any],
) -> Optional[dict[str, Any]]:
    if quantize == "native-int4":
        return native_w4a16_manifest_config(state_export_meta)
    if quantize != "int4":
        return None
    quant_cfg = dict(_get_awq_config(hf_config))
    quant_cfg["weight_layout"] = state_export_meta.get(
        "awq_qweight_layout",
        _AWQ_QWEIGHT_LAYOUT_LEGACY,
    )
    return quant_cfg


def _awq_qweight_layout_from_quantization_config(
    quantization_config: Optional[dict[str, Any]],
) -> str:
    if is_native_w4a16_config(quantization_config):
        return "native-output-major-v1"
    if not quantization_config:
        return _AWQ_QWEIGHT_LAYOUT_LEGACY
    layout = quantization_config.get("weight_layout")
    if layout in (None, ""):
        return _AWQ_QWEIGHT_LAYOUT_LEGACY
    layout = str(layout)
    if layout not in _SUPPORTED_AWQ_QWEIGHT_LAYOUTS:
        raise MGXFormatError(
            f"Unsupported MGX INT4 qweight layout '{layout}'. "
            f"Supported layouts: {sorted(_SUPPORTED_AWQ_QWEIGHT_LAYOUTS)}"
        )
    return layout


def _apply_awq_qweight_layout(model: MegaGemmLlama, layout: str) -> None:
    if layout == "native-output-major-v1":
        return
    is_transposed = layout == _AWQ_QWEIGHT_LAYOUT_DECODE
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module._transposed = is_transposed


def _export_runtime_state(
    model: MegaGemmLlama,
    *,
    quantization: Optional[str] = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    state: dict[str, torch.Tensor] = {}
    export_meta: dict[str, Any] = {}
    exported_awq_qweights: set[str] = set()

    if quantization == "int4":
        for module_name, module in model.named_modules():
            if not isinstance(module, QuantizedLinear):
                continue
            key = f"{module_name}.qweight"
            qweight = module.qweight.detach().cpu()
            if module._transposed:
                state[key] = qweight.contiguous()
            else:
                # Persist AWQ qweights in decode-native layout so MGX loads can
                # skip the first-token transpose on the hot path.
                state[key] = qweight.t().contiguous()
            exported_awq_qweights.add(key)
        export_meta["awq_qweight_layout"] = _AWQ_QWEIGHT_LAYOUT_DECODE
    elif quantization == "native-int4":
        native_modules = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, NativeW4A16Linear)
        ]
        if not native_modules:
            raise MGXFormatError("Native W4A16 export did not contain any quantized Linear modules.")
        sparse_modes = {bool(module.sparse24) for _, module in native_modules}
        if len(sparse_modes) != 1:
            raise MGXFormatError("Native W4A16 export cannot mix dense and 2:4 modules.")
        export_meta.update({
            "native_w4a16_group_size": int(native_modules[0][1].group_size),
            "native_w4a16_sparse24": bool(native_modules[0][1].sparse24),
            "native_w4a16_module_count": len(native_modules),
            "native_w4a16_module_names": [name for name, _ in native_modules],
            "native_w4a16_original_weight_bytes": sum(
                int(module.out_features * module.in_features * 2)
                for _, module in native_modules
            ),
            "native_w4a16_packed_weight_bytes": sum(
                int(module.qweight.numel() * module.qweight.element_size())
                + int(module.metadata.numel() * module.metadata.element_size())
                + int(module.scales.numel() * module.scales.element_size())
                for _, module in native_modules
            ),
        })

    for name, tensor in model.state_dict().items():
        if name == "lm_head.weight" and model.config.tie_word_embeddings:
            continue
        if name in exported_awq_qweights:
            continue
        state[name] = tensor.detach().cpu().contiguous()
    state["__mgx__.cos_cache"] = model.cos_cache.detach().cpu().contiguous()
    state["__mgx__.sin_cache"] = model.sin_cache.detach().cpu().contiguous()
    return state, export_meta


def _build_tensor_table(state: dict[str, torch.Tensor]) -> tuple[list[dict[str, Any]], int]:
    table: list[dict[str, Any]] = []
    total_bytes = 0
    for name in sorted(state.keys()):
        tensor = state[name]
        nbytes = int(tensor.numel() * tensor.element_size())
        total_bytes += nbytes
        table.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "numel": int(tensor.numel()),
                "nbytes": nbytes,
            }
        )
    return table, total_bytes


def _encode_json_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(
        data,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")


def _write_mgx_artifact(
    output_path: Path,
    manifest: dict[str, Any],
    tensor_payload: bytes,
    *,
    prefix_payload: bytes = b"",
    session_payload: bytes = b"",
) -> None:
    manifest_bytes = _encode_json_bytes(manifest)
    manifest_offset = MGX_HEADER_SIZE
    tensor_offset = _align_up(manifest_offset + len(manifest_bytes))

    cursor = tensor_offset + len(tensor_payload)
    prefix_offset = 0
    if prefix_payload:
        prefix_offset = _align_up(cursor)
        cursor = prefix_offset + len(prefix_payload)

    session_offset = 0
    if session_payload:
        session_offset = _align_up(cursor)
        cursor = session_offset + len(session_payload)

    header = _encode_header(
        manifest_offset=manifest_offset,
        manifest_size=len(manifest_bytes),
        tensor_offset=tensor_offset,
        tensor_size=len(tensor_payload),
        prefix_offset=prefix_offset,
        prefix_size=len(prefix_payload),
        session_offset=session_offset,
        session_size=len(session_payload),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        fh.write(header)
        fh.write(manifest_bytes)

        manifest_end = manifest_offset + len(manifest_bytes)
        if tensor_offset > manifest_end:
            fh.write(b"\0" * (tensor_offset - manifest_end))
        fh.write(tensor_payload)

        if prefix_payload:
            prefix_start = fh.tell()
            if prefix_offset > prefix_start:
                fh.write(b"\0" * (prefix_offset - prefix_start))
            fh.write(prefix_payload)

        if session_payload:
            session_start = fh.tell()
            if session_offset > session_start:
                fh.write(b"\0" * (session_offset - session_start))
            fh.write(session_payload)


def _stream_file_into_handle(
    fh,
    source_path: Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    with source_path.open("rb") as src:
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            fh.write(chunk)


def _write_mgx_artifact_from_payload_path(
    output_path: Path,
    manifest: dict[str, Any],
    tensor_payload_path: Path,
    *,
    prefix_payload: bytes = b"",
    session_payload: bytes = b"",
) -> None:
    manifest_bytes = _encode_json_bytes(manifest)
    manifest_offset = MGX_HEADER_SIZE
    tensor_size = tensor_payload_path.stat().st_size
    tensor_offset = _align_up(manifest_offset + len(manifest_bytes))

    cursor = tensor_offset + tensor_size
    prefix_offset = 0
    if prefix_payload:
        prefix_offset = _align_up(cursor)
        cursor = prefix_offset + len(prefix_payload)

    session_offset = 0
    if session_payload:
        session_offset = _align_up(cursor)
        cursor = session_offset + len(session_payload)

    header = _encode_header(
        manifest_offset=manifest_offset,
        manifest_size=len(manifest_bytes),
        tensor_offset=tensor_offset,
        tensor_size=tensor_size,
        prefix_offset=prefix_offset,
        prefix_size=len(prefix_payload),
        session_offset=session_offset,
        session_size=len(session_payload),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        fh.write(header)
        fh.write(manifest_bytes)

        manifest_end = manifest_offset + len(manifest_bytes)
        if tensor_offset > manifest_end:
            fh.write(b"\0" * (tensor_offset - manifest_end))
        _stream_file_into_handle(fh, tensor_payload_path)

        if prefix_payload:
            prefix_start = fh.tell()
            if prefix_offset > prefix_start:
                fh.write(b"\0" * (prefix_offset - prefix_start))
            fh.write(prefix_payload)

        if session_payload:
            session_start = fh.tell()
            if session_offset > session_start:
                fh.write(b"\0" * (session_offset - session_start))
            fh.write(session_payload)


def _encode_session_header(
    *,
    metadata_offset: int,
    metadata_size: int,
    tensor_offset: int,
    tensor_size: int,
) -> bytes:
    raw = _SESSION_HEADER_STRUCT.pack(
        MGX_SESSION_MAGIC,
        MGX_SESSION_VERSION_MAJOR,
        MGX_SESSION_VERSION_MINOR,
        MGX_SESSION_HEADER_SIZE,
        0,
        metadata_offset,
        metadata_size,
        tensor_offset,
        tensor_size,
    )
    if len(raw) > MGX_SESSION_HEADER_SIZE:
        raise MGXFormatError("MGX session header struct is larger than the fixed session header size.")
    return raw + (b"\0" * (MGX_SESSION_HEADER_SIZE - len(raw)))


def _parse_session_header(data: bytes) -> dict[str, int]:
    if len(data) < _SESSION_HEADER_STRUCT.size:
        raise MGXFormatError("MGX session_state section is too small to contain a valid header.")

    unpacked = _SESSION_HEADER_STRUCT.unpack(data[: _SESSION_HEADER_STRUCT.size])
    magic = unpacked[0]
    if magic != MGX_SESSION_MAGIC:
        raise MGXFormatError(f"Invalid MGX session_state magic: {magic!r}")

    major = int(unpacked[1])
    minor = int(unpacked[2])
    header_size = int(unpacked[3])
    flags = int(unpacked[4])
    if major != MGX_SESSION_VERSION_MAJOR:
        raise MGXFormatError(
            f"Incompatible MGX session_state major version {major}. "
            f"Supported major version is {MGX_SESSION_VERSION_MAJOR}."
        )
    if header_size != MGX_SESSION_HEADER_SIZE:
        raise MGXFormatError(
            f"Invalid MGX session_state header size {header_size}. Expected {MGX_SESSION_HEADER_SIZE}."
        )

    return {
        "major": major,
        "minor": minor,
        "header_size": header_size,
        "flags": flags,
        "metadata_offset": int(unpacked[5]),
        "metadata_size": int(unpacked[6]),
        "tensor_offset": int(unpacked[7]),
        "tensor_size": int(unpacked[8]),
    }


def _validate_session_offsets(section_size: int, header: dict[str, int]) -> None:
    for name in ("metadata", "tensor"):
        offset = header[f"{name}_offset"]
        size = header[f"{name}_size"]
        if size < 0 or offset < 0:
            raise MGXFormatError(
                f"MGX session_state section has a negative offset/size for '{name}'."
            )
        if size == 0 and offset == 0:
            continue
        if offset < MGX_SESSION_HEADER_SIZE:
            raise MGXFormatError(
                f"MGX session_state '{name}' starts inside the fixed session header (offset={offset})."
            )
        if offset + size > section_size:
            raise MGXFormatError(
                f"MGX session_state '{name}' exceeds section size ({offset}+{size}>{section_size})."
            )


def _normalize_session_snapshot(
    snapshot: dict[str, Any],
    *,
    artifact_manifest: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    normalized_meta: dict[str, Any] = {}
    tensor_state: dict[str, torch.Tensor] = {}

    for key, value in snapshot.items():
        if key == "kv_data":
            continue
        if key == "kv_data_by_layer":
            if not isinstance(value, dict):
                raise MGXFormatError("MGX session state kv_data_by_layer must be a dict.")
            for layer_idx, layer_tensor in sorted(value.items(), key=lambda item: int(item[0])):
                tensor_state[f"kv_data_by_layer.{int(layer_idx)}"] = layer_tensor.detach().cpu().contiguous()
            continue
        if key in {"linear_conv_states", "linear_recurrent_states"}:
            if not isinstance(value, dict):
                raise MGXFormatError(f"MGX session state {key} must be a dict.")
            for layer_idx, layer_tensor in sorted(value.items(), key=lambda item: int(item[0])):
                tensor_state[f"{key}.{int(layer_idx)}"] = layer_tensor.detach().cpu().contiguous()
            continue
        if torch.is_tensor(value):
            tensor_state[key] = value.detach().cpu().contiguous()
            continue
        normalized_meta[key] = _json_safe(value)

    if "kv_data_by_layer" not in snapshot and "kv_data" in snapshot:
        kv_data = snapshot.get("kv_data") or []
        for layer_idx, layer_tensor in enumerate(kv_data):
            if layer_tensor is None:
                continue
            tensor_state[f"kv_data_by_layer.{layer_idx}"] = layer_tensor.detach().cpu().contiguous()

    if artifact_manifest is not None:
        for key in (
            "source_model_hash",
            "tokenizer_hash",
            "chat_template_hash",
            "source_model_id",
            "quantization",
            "dtype",
            "target_backend",
        ):
            value = artifact_manifest.get(key)
            if value is not None and key not in normalized_meta:
                normalized_meta[key] = _json_safe(value)

    return normalized_meta, tensor_state


def _build_session_section(
    snapshot: dict[str, Any],
    *,
    artifact_manifest: Optional[dict[str, Any]] = None,
) -> tuple[bytes, dict[str, Any]]:
    normalized_snapshot, tensor_state = _normalize_session_snapshot(
        snapshot,
        artifact_manifest=artifact_manifest,
    )

    try:
        from safetensors.torch import save as save_safetensors
    except ImportError as exc:
        raise ImportError(
            "Install safetensors to embed MGX session_state sections: pip install safetensors"
        ) from exc

    tensor_payload = save_safetensors(tensor_state)
    tensor_table, tensor_bytes = _build_tensor_table(tensor_state)
    session_manifest = {
        "format": "mgx-session-state",
        "magic": MGX_SESSION_MAGIC.decode("ascii"),
        "version": {
            "major": MGX_SESSION_VERSION_MAJOR,
            "minor": MGX_SESSION_VERSION_MINOR,
        },
        "snapshot_schema": "megagemm-save-context-v1",
        "snapshot": normalized_snapshot,
        "tensor_payload_format": "safetensors",
        "tensor_payload_sha256": _sha256_bytes(tensor_payload),
        "tensor_count": len(tensor_table),
        "tensor_bytes": tensor_bytes,
        "tensor_table": tensor_table,
    }

    metadata_bytes = _encode_json_bytes(session_manifest)
    metadata_offset = MGX_SESSION_HEADER_SIZE
    tensor_offset = _align_up(metadata_offset + len(metadata_bytes))
    padding = tensor_offset - (metadata_offset + len(metadata_bytes))
    session_header = _encode_session_header(
        metadata_offset=metadata_offset,
        metadata_size=len(metadata_bytes),
        tensor_offset=tensor_offset,
        tensor_size=len(tensor_payload),
    )
    section_bytes = session_header + metadata_bytes + (b"\0" * padding) + tensor_payload
    return section_bytes, session_manifest


def _inspect_session_state_section(
    artifact_path: Path,
    header: dict[str, int],
    *,
    validate_payload_hash: bool = False,
) -> Optional[dict[str, Any]]:
    if header["session_size"] == 0:
        return None

    with artifact_path.open("rb") as fh:
        fh.seek(header["session_offset"])
        session_header_bytes = fh.read(MGX_SESSION_HEADER_SIZE)
    if len(session_header_bytes) != MGX_SESSION_HEADER_SIZE:
        raise MGXFormatError(f"Truncated MGX session_state header in {artifact_path}.")

    session_header = _parse_session_header(session_header_bytes)
    _validate_session_offsets(header["session_size"], session_header)

    with artifact_path.open("rb") as fh:
        fh.seek(header["session_offset"] + session_header["metadata_offset"])
        metadata_bytes = fh.read(session_header["metadata_size"])
    if len(metadata_bytes) != session_header["metadata_size"]:
        raise MGXFormatError(f"Truncated MGX session_state metadata in {artifact_path}.")

    try:
        session_manifest = json.loads(metadata_bytes.decode("utf-8"))
    except Exception as exc:
        raise MGXFormatError(f"Failed to decode MGX session_state manifest from {artifact_path}: {exc}") from exc
    version = session_manifest.get("version", {})
    if session_manifest.get("magic") != MGX_SESSION_MAGIC.decode("ascii"):
        raise MGXFormatError(f"MGX session_state manifest magic mismatch in {artifact_path}.")
    if version.get("major") != MGX_SESSION_VERSION_MAJOR:
        raise MGXFormatError(
            f"MGX session_state manifest major version {version.get('major')} is not supported."
        )

    payload_sha256 = None
    if validate_payload_hash and session_header["tensor_size"] > 0:
        with artifact_path.open("rb") as fh:
            fh.seek(header["session_offset"] + session_header["tensor_offset"])
            payload = fh.read(session_header["tensor_size"])
        if len(payload) != session_header["tensor_size"]:
            raise MGXFormatError(f"Truncated MGX session_state payload in {artifact_path}.")
        payload_sha256 = _sha256_bytes(payload)
        expected = session_manifest.get("tensor_payload_sha256")
        if expected and payload_sha256 != expected:
            raise MGXFormatError(
                f"MGX session_state payload hash mismatch for {artifact_path}. "
                f"Expected {expected}, got {payload_sha256}."
            )

    return {
        "header": session_header,
        "manifest": session_manifest,
        "tensor_payload_sha256": payload_sha256,
    }


def _read_session_tensor_payload(
    artifact_path: Path,
    artifact_header: dict[str, int],
    session_header: dict[str, int],
) -> bytes:
    with artifact_path.open("rb") as fh:
        fh.seek(artifact_header["session_offset"] + session_header["tensor_offset"])
        payload = fh.read(session_header["tensor_size"])
    if len(payload) != session_header["tensor_size"]:
        raise MGXFormatError(f"Truncated MGX session_state payload in {artifact_path}.")
    return payload


def _reconstruct_session_snapshot(session_manifest: dict[str, Any], tensor_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    snapshot = copy.deepcopy(session_manifest.get("snapshot") or {})

    kv_data_by_layer: dict[int, torch.Tensor] = {}
    linear_conv_states: dict[int, torch.Tensor] = {}
    linear_recurrent_states: dict[int, torch.Tensor] = {}

    for name, tensor in tensor_state.items():
        if name.startswith("kv_data_by_layer."):
            layer_idx = int(name.split(".")[-1])
            kv_data_by_layer[layer_idx] = tensor
        elif name.startswith("linear_conv_states."):
            layer_idx = int(name.split(".")[-1])
            linear_conv_states[layer_idx] = tensor
        elif name.startswith("linear_recurrent_states."):
            layer_idx = int(name.split(".")[-1])
            linear_recurrent_states[layer_idx] = tensor
        else:
            snapshot[name] = tensor

    if kv_data_by_layer:
        snapshot["kv_data_by_layer"] = kv_data_by_layer
        num_layers = int(snapshot.get("num_layers", 0) or 0)
        if num_layers > 0:
            kv_data: list[Optional[torch.Tensor]] = [None] * num_layers
            for layer_idx, layer_tensor in kv_data_by_layer.items():
                if 0 <= layer_idx < num_layers:
                    kv_data[layer_idx] = layer_tensor
            snapshot["kv_data"] = kv_data
    else:
        snapshot.setdefault("kv_data_by_layer", {})

    snapshot["linear_conv_states"] = linear_conv_states
    snapshot["linear_recurrent_states"] = linear_recurrent_states
    return snapshot


def _build_manifest(
    *,
    model_source: str,
    source_path: Path,
    config: LlamaConfig,
    dtype: torch.dtype,
    quantize: Optional[str],
    target_backend: str,
    state: dict[str, torch.Tensor],
    payload_sha256: str,
    hf_config: dict[str, Any],
    state_export_meta: dict[str, Any],
    sparsity_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    tensor_table, tensor_bytes = _build_tensor_table(state)
    chat_template = _read_chat_template(source_path)
    model_files = sorted(source_path.rglob("*.safetensors"))

    tokenizer_hashes = _collect_tokenizer_hashes(source_path)
    tokenizer_cfg = _read_json_if_exists(source_path / "tokenizer_config.json")
    special_tokens_map = _read_json_if_exists(source_path / "special_tokens_map.json")
    tokenizer_init: dict[str, Any] = {}
    for key in (
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
        "sep_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ):
        if key in special_tokens_map:
            tokenizer_init[key] = special_tokens_map[key]
        elif key in tokenizer_cfg:
            tokenizer_init[key] = tokenizer_cfg[key]
    for key in (
        "model_max_length",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
    ):
        if key in tokenizer_cfg:
            tokenizer_init[key] = tokenizer_cfg[key]
    if chat_template is not None:
        tokenizer_init["chat_template"] = chat_template
    if (source_path / "tokenizer.json").exists():
        tokenizer_init["tokenizer_file"] = "tokenizer.json"

    manifest = {
        "format": "mgx",
        "magic": MGX_MAGIC.decode("ascii"),
        "version": {
            "major": MGX_VERSION_MAJOR,
            "minor": MGX_VERSION_MINOR,
        },
        "runtime_layout": "megagemm-runtime-v1",
        "tensor_payload_format": "safetensors",
        "source_model_id": model_source,
        "source_snapshot_path": str(source_path),
        "source_model_hash": _compute_source_model_hash(source_path),
        "source_model_files": [path.relative_to(source_path).as_posix() for path in model_files],
        "tokenizer_source_path": str(source_path),
        "tokenizer_hash": tokenizer_hashes["primary"],
        "tokenizer_hash_scheme": tokenizer_hashes["primary_scheme"],
        "tokenizer_core_hash": tokenizer_hashes["core"],
        "tokenizer_bundle_hash": tokenizer_hashes["bundle"],
        "tokenizer_init": tokenizer_init,
        "chat_template_hash": _sha256_text(chat_template),
        "architecture": config.model_type,
        "config": asdict(config),
        "dtype": _dtype_name(dtype),
        "quantization": "int4" if quantize == "native-int4" else (quantize or "none"),
        "quantization_config": _quantization_config_for_manifest(
            hf_config,
            quantize,
            state_export_meta,
        ),
        "sparsity": (
            "2:4"
            if _is_native_w4_sparse_config(sparsity_config)
            else (sparsity_config or {}).get("format", "none")
        ),
        "sparsity_config": sparsity_config,
        "target_backend": target_backend,
        "rope_config": {
            "theta": config.rope_theta,
            "rotary_dim": config.rotary_dim,
            "max_position_embeddings": config.max_position_embeddings,
            "half_rotate": bool(config.rope_half_rotate),
        },
        "compat_flags": _mgx_compat_flags_for_layer_types(config.layer_types) + (
            ["mgx-v1-native-w4a16", "mgx-v1-native-w4a16-sparse24"]
            if quantize == "native-int4" and sparsity_config
            else ["mgx-v1-native-w4a16"]
            if quantize == "native-int4"
            else
            ["mgx-v1-awq-int4", "mgx-v1-awq-decode-layout"]
            if quantize == "int4"
            else ["mgx-v1-no-awq"]
        ) + (["mgx-v1-sparse24-packed-v1"] if sparsity_config else ["mgx-v1-dense"]),
        "tensor_count": len(tensor_table),
        "tensor_bytes": tensor_bytes,
        "tensor_payload_sha256": payload_sha256,
        "tensor_table": tensor_table,
        "reserved_sections": {
            "prefix_pack": {"offset": 0, "size": 0},
            "session_state": {"offset": 0, "size": 0},
        },
    }
    return manifest


def _encode_header(
    *,
    manifest_offset: int,
    manifest_size: int,
    tensor_offset: int,
    tensor_size: int,
    prefix_offset: int = 0,
    prefix_size: int = 0,
    session_offset: int = 0,
    session_size: int = 0,
) -> bytes:
    raw = _HEADER_STRUCT.pack(
        MGX_MAGIC,
        MGX_VERSION_MAJOR,
        MGX_VERSION_MINOR,
        MGX_HEADER_SIZE,
        0,
        manifest_offset,
        manifest_size,
        tensor_offset,
        tensor_size,
        prefix_offset,
        prefix_size,
        session_offset,
        session_size,
    )
    if len(raw) > MGX_HEADER_SIZE:
        raise MGXFormatError("MGX header struct is larger than the fixed header size.")
    return raw + (b"\0" * (MGX_HEADER_SIZE - len(raw)))


def _read_header(path: Path) -> dict[str, int]:
    with path.open("rb") as fh:
        header = fh.read(MGX_HEADER_SIZE)
    if len(header) < _HEADER_STRUCT.size:
        raise MGXFormatError(f"{path} is too small to be a valid MGX artifact.")

    unpacked = _HEADER_STRUCT.unpack(header[: _HEADER_STRUCT.size])
    magic = unpacked[0]
    if magic != MGX_MAGIC:
        raise MGXFormatError(f"Invalid MGX magic in {path}: {magic!r}")

    major = int(unpacked[1])
    minor = int(unpacked[2])
    header_size = int(unpacked[3])
    flags = int(unpacked[4])
    if major != MGX_VERSION_MAJOR:
        raise MGXFormatError(
            f"Incompatible MGX major version {major}. Supported major version is {MGX_VERSION_MAJOR}."
        )
    if header_size != MGX_HEADER_SIZE:
        raise MGXFormatError(
            f"Invalid MGX header size {header_size}. Expected {MGX_HEADER_SIZE}."
        )

    sections = {
        "manifest_offset": int(unpacked[5]),
        "manifest_size": int(unpacked[6]),
        "tensor_offset": int(unpacked[7]),
        "tensor_size": int(unpacked[8]),
        "prefix_offset": int(unpacked[9]),
        "prefix_size": int(unpacked[10]),
        "session_offset": int(unpacked[11]),
        "session_size": int(unpacked[12]),
    }
    return {
        "major": major,
        "minor": minor,
        "header_size": header_size,
        "flags": flags,
        **sections,
    }


def _validate_offsets(path: Path, header: dict[str, int]) -> None:
    file_size = path.stat().st_size
    for name in ("manifest", "tensor", "prefix", "session"):
        offset = header[f"{name}_offset"]
        size = header[f"{name}_size"]
        if size < 0 or offset < 0:
            raise MGXFormatError(f"{path}: negative offset/size detected for section '{name}'.")
        if size == 0 and offset == 0:
            continue
        if offset < MGX_HEADER_SIZE:
            raise MGXFormatError(
                f"{path}: section '{name}' starts inside the fixed header (offset={offset})."
            )
        if offset + size > file_size:
            raise MGXFormatError(
                f"{path}: section '{name}' exceeds file size ({offset}+{size}>{file_size})."
            )


def _read_manifest(path: Path, header: dict[str, int]) -> dict[str, Any]:
    with path.open("rb") as fh:
        fh.seek(header["manifest_offset"])
        manifest_bytes = fh.read(header["manifest_size"])
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        raise MGXFormatError(f"Failed to decode MGX manifest from {path}: {exc}") from exc
    return manifest


def _read_tensor_payload(path: Path, header: dict[str, int]) -> bytes:
    with path.open("rb") as fh:
        fh.seek(header["tensor_offset"])
        payload = fh.read(header["tensor_size"])
    if len(payload) != header["tensor_size"]:
        raise MGXFormatError(f"Truncated tensor payload in {path}.")
    return payload


def _validate_manifest(manifest: dict[str, Any]) -> None:
    version = manifest.get("version", {})
    if version.get("major") != MGX_VERSION_MAJOR:
        raise MGXFormatError(
            f"MGX artifact major version {version.get('major')} is not supported."
        )
    if manifest.get("magic") != MGX_MAGIC.decode("ascii"):
        raise MGXFormatError("MGX manifest magic mismatch.")
    backend = manifest.get("target_backend")
    if backend not in _SUPPORTED_BACKENDS:
        raise MGXFormatError(
            f"Unsupported MGX backend '{backend}'. Supported backends: {sorted(_SUPPORTED_BACKENDS)}"
        )
    quant = manifest.get("quantization")
    if quant not in {"none", "int8", "int4"}:
        raise MGXFormatError(
            f"Unsupported MGX quantization '{quant}'. MGX supports only 'none', 'int8', and 'int4'."
        )
    if quant == "int4":
        quant_cfg = manifest.get("quantization_config") or {}
        group_size = int(quant_cfg.get("group_size", 0) or 0)
        if group_size <= 0:
            raise MGXFormatError("MGX INT4 artifact is missing a valid group_size in quantization_config.")
        if is_native_w4a16_config(quant_cfg):
            if group_size % 8:
                raise MGXFormatError("Native MGX W4A16 group_size must be divisible by 8.")
            if int(quant_cfg.get("bits", 0) or 0) != 4 or not bool(quant_cfg.get("symmetric")):
                raise MGXFormatError("Native MGX W4A16 requires symmetric 4-bit weights.")
        else:
            _awq_qweight_layout_from_quantization_config(quant_cfg)
    sparsity = manifest.get("sparsity", "none")
    sparsity_config = manifest.get("sparsity_config") or None
    if sparsity not in {"none", "2:4"}:
        raise MGXFormatError(
            f"Unsupported MGX sparsity '{sparsity}'. MGX supports only 'none' and '2:4'."
        )
    if sparsity == "2:4":
        native_sparse = quant == "int4" and is_native_w4a16_config(
            manifest.get("quantization_config") or {}
        )
        if quant != "none" and not native_sparse:
            raise MGXFormatError(
                "MGX 2:4 can be combined only with the native MGX W4A16 format."
            )
        if not sparsity_config:
            raise MGXFormatError("MGX 2:4 artifact is missing sparsity_config.")
        tensor_names = {
            str(entry.get("name"))
            for entry in manifest.get("tensor_table", [])
            if isinstance(entry, dict) and entry.get("name")
        }
        if native_sparse:
            if not _is_native_w4_sparse_config(sparsity_config):
                raise MGXFormatError("Native W4A16 2:4 artifact has an invalid sparsity format.")
            if int(sparsity_config.get("group_size", 0)) != 4 or int(
                sparsity_config.get("kept_values", 0)
            ) != 2:
                raise MGXFormatError("Native W4A16 2:4 artifact has an invalid 2-of-4 contract.")
            for module_name in sparsity_config.get("module_names", []):
                if f"{module_name}.qweight" not in tensor_names or f"{module_name}.metadata" not in tensor_names:
                    raise MGXFormatError(
                        f"Native W4A16 2:4 tensors are missing for module {module_name}."
                    )
        else:
            try:
                validate_sparse24_config(sparsity_config, tensor_names=tensor_names)
            except (TypeError, ValueError) as exc:
                raise MGXFormatError(f"Invalid MGX 2:4 sparsity configuration: {exc}") from exc
    elif sparsity_config:
        raise MGXFormatError("MGX manifest contains sparsity_config but declares sparsity='none'.")
    config_dict = manifest.get("config")
    if not isinstance(config_dict, dict):
        raise MGXFormatError("MGX manifest is missing the serialized runtime config.")
    layer_types = config_dict.get("layer_types") or []
    unsupported_layer_types = _unsupported_layer_types_for_mgx(layer_types)
    if unsupported_layer_types:
        raise MGXFormatError(
            "MGX v1 artifact contains unsupported layer types: "
            f"{unsupported_layer_types}. "
            f"Supported layer types: {sorted(_MGX_SUPPORTED_LAYER_TYPES)}."
        )


def inspect_mgx(
    path: str | os.PathLike[str],
    *,
    validate_payload_hash: bool = True,
    validate_session_hash: bool = False,
    payload_cache_dir: Optional[str | os.PathLike[str]] = None,
) -> dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    header = _read_header(artifact_path)
    _validate_offsets(artifact_path, header)
    manifest = _read_manifest(artifact_path, header)
    _validate_manifest(manifest)

    payload_sha256 = None
    if validate_payload_hash:
        payload = _read_tensor_payload(artifact_path, header)
        payload_sha256 = _sha256_bytes(payload)
        expected = manifest.get("tensor_payload_sha256")
        if expected and payload_sha256 != expected:
            raise MGXFormatError(
                f"MGX tensor payload hash mismatch for {artifact_path}. "
                f"Expected {expected}, got {payload_sha256}."
            )

    payload_cache_path = _payload_cache_path_for_artifact(
        artifact_path,
        manifest,
        cache_dir=payload_cache_dir,
    )
    runtime_cache_path = _payload_runtime_cache_path_for_artifact(
        artifact_path,
        manifest,
        cache_dir=payload_cache_dir,
    )
    runtime_index_path = _payload_runtime_index_path_for_artifact(
        artifact_path,
        manifest,
        cache_dir=payload_cache_dir,
    )
    session_state = _inspect_session_state_section(
        artifact_path,
        header,
        validate_payload_hash=validate_session_hash,
    )

    return {
        "path": str(artifact_path),
        "file_size": artifact_path.stat().st_size,
        "header": header,
        "manifest": manifest,
        "tensor_payload_sha256": payload_sha256,
        "payload_cache_path": str(payload_cache_path),
        "payload_cache_exists": _payload_cache_is_valid(payload_cache_path, header["tensor_size"]),
        "runtime_cache_path": str(runtime_cache_path),
        "runtime_cache_index_path": str(runtime_index_path),
        "runtime_cache_exists": _runtime_payload_cache_is_valid(
            runtime_cache_path,
            runtime_index_path,
            expected_payload_sha256=manifest.get("tensor_payload_sha256"),
            expected_tensor_count=manifest.get("tensor_count"),
        ),
        "session_state_present": session_state is not None,
        "session_state": session_state,
    }


def prime_mgx_payload_cache(
    path: str | os.PathLike[str],
    *,
    validate_payload_hash: bool = True,
    payload_cache_dir: Optional[str | os.PathLike[str]] = None,
) -> dict[str, Any]:
    """
    Extract the embedded safetensors payload into a reusable on-disk cache file.

    This keeps `.mgx` as the canonical single-file artifact while enabling future
    loads to reuse a standalone binary blob that `safetensors.load_file(...)` can
    mmap efficiently.
    """
    artifact = inspect_mgx(
        path,
        validate_payload_hash=False,
        payload_cache_dir=payload_cache_dir,
    )
    artifact_path = Path(artifact["path"])
    header = artifact["header"]
    manifest = artifact["manifest"]
    payload_cache_path = Path(artifact["payload_cache_path"])
    runtime_cache_path = Path(artifact["runtime_cache_path"])
    runtime_index_path = Path(artifact["runtime_cache_index_path"])
    payload_cache_hit = _payload_cache_is_valid(payload_cache_path, header["tensor_size"])
    payload_cache_written = False
    verified = False

    if not payload_cache_hit:
        payload_bytes = _read_tensor_payload(artifact_path, header)
        if validate_payload_hash:
            payload_sha256 = _sha256_bytes(payload_bytes)
            expected = manifest.get("tensor_payload_sha256")
            if expected and payload_sha256 != expected:
                raise MGXFormatError(
                    f"MGX tensor payload hash mismatch for {artifact_path}. "
                    f"Expected {expected}, got {payload_sha256}."
                )
            verified = True

        _write_payload_cache_file(payload_cache_path, payload_bytes)
        payload_cache_written = True
    runtime_cache_hit = _runtime_payload_cache_is_valid(
        runtime_cache_path,
        runtime_index_path,
        expected_payload_sha256=manifest.get("tensor_payload_sha256"),
        expected_tensor_count=manifest.get("tensor_count"),
    )
    runtime_cache_written = False
    runtime_cache_info: dict[str, Any] = {}
    if not runtime_cache_hit:
        runtime_cache_info = _build_runtime_payload_cache_from_payload_cache(
            payload_cache_path,
            runtime_cache_path,
            runtime_index_path,
            manifest=manifest,
        )
        runtime_cache_written = True
        runtime_cache_hit = False

    result = {
        "artifact_path": str(artifact_path),
        "payload_cache_path": str(payload_cache_path),
        "payload_cache_bytes": int(header["tensor_size"]),
        "payload_cache_hit": payload_cache_hit,
        "payload_cache_written": payload_cache_written,
        "verified": verified,
        "runtime_cache_path": str(runtime_cache_path),
        "runtime_cache_index_path": str(runtime_index_path),
        "runtime_cache_hit": runtime_cache_hit,
        "runtime_cache_written": runtime_cache_written,
    }
    result.update(runtime_cache_info)
    return result


def attach_session_state_to_mgx(
    path: str | os.PathLike[str],
    snapshot: dict[str, Any],
    *,
    out_path: Optional[str | os.PathLike[str]] = None,
    validate_payload_hash: bool = True,
    payload_cache_dir: Optional[str | os.PathLike[str]] = None,
) -> dict[str, Any]:
    """
    Attach a serialized runtime session snapshot into the MGX session_state section.

    The tensor payload of the compiled model is preserved byte-for-byte; only the
    outer MGX container is rewritten to add or replace the optional session_state
    section.
    """
    artifact = inspect_mgx(
        path,
        validate_payload_hash=validate_payload_hash,
        payload_cache_dir=payload_cache_dir,
    )
    artifact_path = Path(artifact["path"])
    output_path = Path(out_path).expanduser().resolve() if out_path is not None else artifact_path
    tensor_payload = _read_tensor_payload(artifact_path, artifact["header"])
    session_payload, _session_manifest = _build_session_section(
        snapshot,
        artifact_manifest=artifact["manifest"],
    )
    manifest = copy.deepcopy(artifact["manifest"])
    _write_mgx_artifact(
        output_path,
        manifest,
        tensor_payload,
        session_payload=session_payload,
    )
    return inspect_mgx(
        output_path,
        validate_payload_hash=False,
        payload_cache_dir=payload_cache_dir,
    )


def extract_session_state_from_mgx(
    path: str | os.PathLike[str],
    *,
    validate_session_hash: bool = True,
    payload_cache_dir: Optional[str | os.PathLike[str]] = None,
) -> dict[str, Any]:
    """
    Extract the optional embedded runtime session snapshot from an MGX artifact.
    """
    artifact = inspect_mgx(
        path,
        validate_payload_hash=False,
        validate_session_hash=validate_session_hash,
        payload_cache_dir=payload_cache_dir,
    )
    if not artifact["session_state_present"]:
        raise MGXFormatError(f"MGX artifact {artifact['path']} does not contain an embedded session_state section.")

    session_info = artifact["session_state"]
    session_header = session_info["header"]
    session_manifest = session_info["manifest"]
    payload_bytes = _read_session_tensor_payload(
        Path(artifact["path"]),
        artifact["header"],
        session_header,
    )

    tensor_state: dict[str, torch.Tensor] = {}
    if session_header["tensor_size"] > 0:
        try:
            from safetensors.torch import load as load_safetensors
        except ImportError as exc:
            raise ImportError(
                "Install safetensors to extract MGX session_state payloads: pip install safetensors"
            ) from exc
        tensor_state = load_safetensors(payload_bytes)

    return _reconstruct_session_snapshot(session_manifest, tensor_state)


def _replace_int8_modules_from_state(model: MegaGemmLlama, state: dict[str, torch.Tensor]) -> None:
    with torch.device("meta"):
        for layer_idx, layer in enumerate(model.layers):
            prefix = f"layers.{layer_idx}"

            qkv_key = f"{prefix}.self_attn.qkv_proj.weight_int8"
            if qkv_key in state:
                qkv_w = state[qkv_key]
                has_bias = f"{prefix}.self_attn.qkv_proj.bias" in state
                layer.self_attn.qkv_proj = Int8Linear(
                    int(qkv_w.shape[1]),
                    int(qkv_w.shape[0]),
                    bias=has_bias,
                )
            q_key = f"{prefix}.self_attn.q_proj.weight_int8"
            if q_key in state:
                q_w = state[q_key]
                has_bias = f"{prefix}.self_attn.q_proj.bias" in state
                layer.self_attn.q_proj = Int8Linear(
                    int(q_w.shape[1]),
                    int(q_w.shape[0]),
                    bias=has_bias,
                )

            k_key = f"{prefix}.self_attn.k_proj.weight_int8"
            if k_key in state:
                k_w = state[k_key]
                has_bias = f"{prefix}.self_attn.k_proj.bias" in state
                layer.self_attn.k_proj = Int8Linear(
                    int(k_w.shape[1]),
                    int(k_w.shape[0]),
                    bias=has_bias,
                )

            v_key = f"{prefix}.self_attn.v_proj.weight_int8"
            if v_key in state:
                v_w = state[v_key]
                has_bias = f"{prefix}.self_attn.v_proj.bias" in state
                layer.self_attn.v_proj = Int8Linear(
                    int(v_w.shape[1]),
                    int(v_w.shape[0]),
                    bias=has_bias,
                )

            o_key = f"{prefix}.self_attn.o_proj.weight_int8"
            if o_key in state:
                o_w = state[o_key]
                has_bias = f"{prefix}.self_attn.o_proj.bias" in state
                layer.self_attn.o_proj = Int8Linear(
                    int(o_w.shape[1]),
                    int(o_w.shape[0]),
                    bias=has_bias,
                )

            gate_up_key = f"{prefix}.mlp.gate_up_proj.weight_int8"
            if gate_up_key in state:
                gate_up_w = state[gate_up_key]
                has_bias = f"{prefix}.mlp.gate_up_proj.bias" in state
                layer.mlp.gate_up_proj = Int8Linear(
                    int(gate_up_w.shape[1]),
                    int(gate_up_w.shape[0]),
                    bias=has_bias,
                )

            down_key = f"{prefix}.mlp.down_proj.weight_int8"
            if down_key in state:
                down_w = state[down_key]
                has_bias = f"{prefix}.mlp.down_proj.bias" in state
                layer.mlp.down_proj = Int8Linear(
                    int(down_w.shape[1]),
                    int(down_w.shape[0]),
                    bias=has_bias,
                )

            _replace_qwen3_moe_int8_experts_from_state(layer, prefix, state)


def _replace_qwen3_moe_int8_experts_from_state(
    layer: nn.Module,
    prefix: str,
    state: dict[str, torch.Tensor],
) -> None:
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    if experts is None:
        return

    keys = {
        "gate_up_int8": f"{prefix}.mlp.experts.gate_up_int8",
        "gate_up_scale": f"{prefix}.mlp.experts.gate_up_scale",
        "down_int8": f"{prefix}.mlp.experts.down_int8",
        "down_scale": f"{prefix}.mlp.experts.down_scale",
    }
    if not all(key in state for key in keys.values()):
        return

    experts._parameters.pop("gate_up_proj", None)
    experts._parameters.pop("down_proj", None)
    for field, key in keys.items():
        tensor = state[key]
        experts._buffers[field] = torch.empty(
            tuple(int(dim) for dim in tensor.shape),
            device="meta",
            dtype=tensor.dtype,
        )


def _set_named_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, field = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, field, module)


def _replace_native_w4_modules(
    model: MegaGemmLlama,
    keys: set[str],
    *,
    group_size: int,
    sparse24: bool,
) -> None:
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and f"{name}.qweight" in keys:
            replacements.append((name, module))
    if not replacements:
        raise MGXFormatError("Native W4A16 payload does not match any runtime Linear modules.")
    with torch.device("meta"):
        for name, module in replacements:
            has_bias = f"{name}.bias" in keys
            quantized = NativeW4A16Linear(
                module.in_features,
                module.out_features,
                group_size=group_size,
                bias=has_bias,
                sparse24=sparse24,
            )
            _set_named_submodule(model, name, quantized)


def _replace_int4_modules_from_state(
    model: MegaGemmLlama,
    state: dict[str, torch.Tensor],
    *,
    group_size: int,
) -> None:
    with torch.device("meta"):
        for layer_idx, layer in enumerate(model.layers):
            prefix = f"layers.{layer_idx}"

            qkv_key = f"{prefix}.self_attn.qkv_proj.qweight"
            if qkv_key in state:
                has_bias = f"{prefix}.self_attn.qkv_proj.bias" in state
                layer.self_attn.qkv_proj = QuantizedLinear(
                    layer.self_attn.qkv_proj.in_features,
                    layer.self_attn.qkv_proj.out_features,
                    group_size=group_size,
                    bias=has_bias,
                )
                layer.self_attn._awq_separate = False

            o_key = f"{prefix}.self_attn.o_proj.qweight"
            if o_key in state:
                has_bias = f"{prefix}.self_attn.o_proj.bias" in state
                layer.self_attn.o_proj = QuantizedLinear(
                    layer.self_attn.o_proj.in_features,
                    layer.self_attn.o_proj.out_features,
                    group_size=group_size,
                    bias=has_bias,
                )

            gate_up_key = f"{prefix}.mlp.gate_up_proj.qweight"
            if gate_up_key in state:
                has_bias = f"{prefix}.mlp.gate_up_proj.bias" in state
                layer.mlp.gate_up_proj = QuantizedLinear(
                    layer.mlp.gate_up_proj.in_features,
                    layer.mlp.gate_up_proj.out_features,
                    group_size=group_size,
                    bias=has_bias,
                )
                layer.mlp._awq_separate = False

            down_key = f"{prefix}.mlp.down_proj.qweight"
            if down_key in state:
                has_bias = f"{prefix}.mlp.down_proj.bias" in state
                layer.mlp.down_proj = QuantizedLinear(
                    layer.mlp.down_proj.in_features,
                    layer.mlp.down_proj.out_features,
                    group_size=group_size,
                    bias=has_bias,
                )

            _replace_qwen3_moe_awq_experts_from_state(
                layer,
                prefix,
                state,
                group_size=group_size,
            )


def _replace_qwen3_moe_awq_experts_from_state(
    layer: nn.Module,
    prefix: str,
    state: dict[str, torch.Tensor],
    *,
    group_size: int,
) -> None:
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    if experts is None:
        return

    keys = {
        "gate_up_qweight": f"{prefix}.mlp.experts.gate_up_qweight",
        "gate_up_scales": f"{prefix}.mlp.experts.gate_up_scales",
        "gate_up_qzeros": f"{prefix}.mlp.experts.gate_up_qzeros",
        "down_qweight": f"{prefix}.mlp.experts.down_qweight",
        "down_scales": f"{prefix}.mlp.experts.down_scales",
        "down_qzeros": f"{prefix}.mlp.experts.down_qzeros",
    }
    if not all(key in state for key in keys.values()):
        return

    experts._parameters.pop("gate_up_proj", None)
    experts._parameters.pop("down_proj", None)
    experts.awq_group_size = int(group_size)
    for field, key in keys.items():
        tensor = state[key]
        experts._buffers[field] = torch.empty(
            tuple(int(dim) for dim in tensor.shape),
            device="meta",
            dtype=tensor.dtype,
        )


def _replace_int8_modules_from_keys(model: MegaGemmLlama, keys: set[str]) -> None:
    with torch.device("meta"):
        for layer_idx, layer in enumerate(model.layers):
            prefix = f"layers.{layer_idx}"

            qkv_key = f"{prefix}.self_attn.qkv_proj.weight_int8"
            if qkv_key in keys:
                has_bias = f"{prefix}.self_attn.qkv_proj.bias" in keys
                scale_key = f"{prefix}.self_attn.qkv_proj.scale"
                scale = None
                if scale_key in keys:
                    scale = qkv_key  # sentinel for presence only
                layer.self_attn.qkv_proj = Int8Linear(
                    layer.self_attn.qkv_proj.in_features,
                    layer.self_attn.qkv_proj.out_features,
                    bias=has_bias,
                )
            q_key = f"{prefix}.self_attn.q_proj.weight_int8"
            if q_key in keys:
                has_bias = f"{prefix}.self_attn.q_proj.bias" in keys
                out_features = int(layer.self_attn._q_proj_size)
                layer.self_attn.q_proj = Int8Linear(
                    layer.self_attn.config.hidden_size,
                    out_features,
                    bias=has_bias,
                )

            k_key = f"{prefix}.self_attn.k_proj.weight_int8"
            if k_key in keys:
                has_bias = f"{prefix}.self_attn.k_proj.bias" in keys
                out_features = int(layer.self_attn._k_size)
                layer.self_attn.k_proj = Int8Linear(
                    layer.self_attn.config.hidden_size,
                    out_features,
                    bias=has_bias,
                )

            v_key = f"{prefix}.self_attn.v_proj.weight_int8"
            if v_key in keys:
                has_bias = f"{prefix}.self_attn.v_proj.bias" in keys
                out_features = int(layer.self_attn._v_size)
                layer.self_attn.v_proj = Int8Linear(
                    layer.self_attn.config.hidden_size,
                    out_features,
                    bias=has_bias,
                )

            o_key = f"{prefix}.self_attn.o_proj.weight_int8"
            if o_key in keys:
                has_bias = f"{prefix}.self_attn.o_proj.bias" in keys
                layer.self_attn.o_proj = Int8Linear(
                    layer.self_attn.o_proj.in_features,
                    layer.self_attn.o_proj.out_features,
                    bias=has_bias,
                )

            gate_up_key = f"{prefix}.mlp.gate_up_proj.weight_int8"
            if gate_up_key in keys:
                has_bias = f"{prefix}.mlp.gate_up_proj.bias" in keys
                layer.mlp.gate_up_proj = Int8Linear(
                    layer.mlp.gate_up_proj.in_features,
                    layer.mlp.gate_up_proj.out_features,
                    bias=has_bias,
                )

            down_key = f"{prefix}.mlp.down_proj.weight_int8"
            if down_key in keys:
                has_bias = f"{prefix}.mlp.down_proj.bias" in keys
                layer.mlp.down_proj = Int8Linear(
                    layer.mlp.down_proj.in_features,
                    layer.mlp.down_proj.out_features,
                    bias=has_bias,
                )

            _replace_qwen3_moe_int8_experts_from_keys(layer, prefix, keys)


def _replace_qwen3_moe_int8_experts_from_keys(
    layer: nn.Module,
    prefix: str,
    keys: set[str],
) -> None:
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    if experts is None:
        return

    required = {
        f"{prefix}.mlp.experts.gate_up_int8",
        f"{prefix}.mlp.experts.gate_up_scale",
        f"{prefix}.mlp.experts.down_int8",
        f"{prefix}.mlp.experts.down_scale",
    }
    if not required.issubset(keys):
        return

    num_experts = int(experts.num_experts)
    hidden_dim = int(experts.hidden_dim)
    intermediate_dim = int(experts.intermediate_dim)
    experts._parameters.pop("gate_up_proj", None)
    experts._parameters.pop("down_proj", None)
    experts._buffers["gate_up_int8"] = torch.empty(
        (num_experts, 2 * intermediate_dim, hidden_dim),
        device="meta",
        dtype=torch.int8,
    )
    experts._buffers["gate_up_scale"] = torch.empty(
        (num_experts, 2 * intermediate_dim),
        device="meta",
        dtype=torch.float16,
    )
    experts._buffers["down_int8"] = torch.empty(
        (num_experts, hidden_dim, intermediate_dim),
        device="meta",
        dtype=torch.int8,
    )
    experts._buffers["down_scale"] = torch.empty(
        (num_experts, hidden_dim),
        device="meta",
        dtype=torch.float16,
    )


def _replace_int4_modules_from_keys(
    model: MegaGemmLlama,
    keys: set[str],
    *,
    group_size: int,
) -> None:
    with torch.device("meta"):
        for layer_idx, layer in enumerate(model.layers):
            prefix = f"layers.{layer_idx}"

            qkv_key = f"{prefix}.self_attn.qkv_proj.qweight"
            if qkv_key in keys:
                has_bias = f"{prefix}.self_attn.qkv_proj.bias" in keys
                layer.self_attn.qkv_proj = QuantizedLinear(
                    layer.self_attn.qkv_proj.in_features,
                    layer.self_attn.qkv_proj.out_features,
                    group_size=group_size,
                    bias=has_bias,
                )
                layer.self_attn._awq_separate = False

            o_key = f"{prefix}.self_attn.o_proj.qweight"
            if o_key in keys:
                has_bias = f"{prefix}.self_attn.o_proj.bias" in keys
                layer.self_attn.o_proj = QuantizedLinear(
                    layer.self_attn.o_proj.in_features,
                    layer.self_attn.o_proj.out_features,
                    group_size=group_size,
                    bias=has_bias,
                )

            gate_up_key = f"{prefix}.mlp.gate_up_proj.qweight"
            if gate_up_key in keys:
                has_bias = f"{prefix}.mlp.gate_up_proj.bias" in keys
                layer.mlp.gate_up_proj = QuantizedLinear(
                    layer.mlp.gate_up_proj.in_features,
                    layer.mlp.gate_up_proj.out_features,
                    group_size=group_size,
                    bias=has_bias,
                )
                layer.mlp._awq_separate = False

            down_key = f"{prefix}.mlp.down_proj.qweight"
            if down_key in keys:
                has_bias = f"{prefix}.mlp.down_proj.bias" in keys
                layer.mlp.down_proj = QuantizedLinear(
                    layer.mlp.down_proj.in_features,
                    layer.mlp.down_proj.out_features,
                    group_size=group_size,
                    bias=has_bias,
                )

            _replace_qwen3_moe_awq_experts_from_keys(
                layer,
                prefix,
                keys,
                group_size=group_size,
            )


def _replace_qwen3_moe_awq_experts_from_keys(
    layer: nn.Module,
    prefix: str,
    keys: set[str],
    *,
    group_size: int,
) -> None:
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    if experts is None:
        return

    required = {
        f"{prefix}.mlp.experts.gate_up_qweight",
        f"{prefix}.mlp.experts.gate_up_scales",
        f"{prefix}.mlp.experts.gate_up_qzeros",
        f"{prefix}.mlp.experts.down_qweight",
        f"{prefix}.mlp.experts.down_scales",
        f"{prefix}.mlp.experts.down_qzeros",
    }
    if not required.issubset(keys):
        return

    num_experts = int(experts.num_experts)
    hidden_dim = int(experts.hidden_dim)
    intermediate_dim = int(experts.intermediate_dim)
    if hidden_dim % group_size != 0 or intermediate_dim % group_size != 0:
        raise MGXFormatError(
            "Qwen3 MoE AWQ expert dimensions are not divisible by quantization_config.group_size."
        )
    if (2 * intermediate_dim) % 8 != 0 or hidden_dim % 8 != 0:
        raise MGXFormatError("Qwen3 MoE AWQ expert output dimensions must be divisible by 8.")

    experts._parameters.pop("gate_up_proj", None)
    experts._parameters.pop("down_proj", None)
    experts.awq_group_size = int(group_size)
    experts._buffers["gate_up_qweight"] = torch.empty(
        (num_experts, hidden_dim, (2 * intermediate_dim) // 8),
        device="meta",
        dtype=torch.int32,
    )
    experts._buffers["gate_up_scales"] = torch.empty(
        (num_experts, hidden_dim // group_size, 2 * intermediate_dim),
        device="meta",
        dtype=torch.float16,
    )
    experts._buffers["gate_up_qzeros"] = torch.empty(
        (num_experts, hidden_dim // group_size, (2 * intermediate_dim) // 8),
        device="meta",
        dtype=torch.int32,
    )
    experts._buffers["down_qweight"] = torch.empty(
        (num_experts, intermediate_dim, hidden_dim // 8),
        device="meta",
        dtype=torch.int32,
    )
    experts._buffers["down_scales"] = torch.empty(
        (num_experts, intermediate_dim // group_size, hidden_dim),
        device="meta",
        dtype=torch.float16,
    )
    experts._buffers["down_qzeros"] = torch.empty(
        (num_experts, intermediate_dim // group_size, hidden_dim // 8),
        device="meta",
        dtype=torch.int32,
    )


def _resolve_module_field(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    module: nn.Module = root
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module, parts[-1]


def _resolve_payload_hydration_mode(device: str | torch.device) -> str:
    requested = os.getenv("MEGAGEMM_MGX_PAYLOAD_HYDRATION", "auto").strip().lower()
    if requested not in _SUPPORTED_PAYLOAD_HYDRATION_MODES:
        requested = "auto"
    device_type = torch.device(device).type
    if requested == "gpu_bulk" and device_type != "cuda":
        return "cpu_bulk"
    if requested == "auto":
        if device_type == "cuda":
            return "gpu_bulk"
        return "direct_device"
    return requested


def _resolve_payload_profile_topk() -> int:
    raw = os.getenv("MEGAGEMM_MGX_PAYLOAD_TOPK", "8").strip()
    try:
        value = int(raw)
    except ValueError:
        return 8
    return max(0, value)


def _tensor_num_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _update_top_payload_entries(
    entries: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    metric: str,
    limit: int,
) -> None:
    if limit <= 0:
        return
    entries.append(candidate)
    entries.sort(key=lambda item: float(item.get(metric, 0.0)), reverse=True)
    del entries[limit:]


def _build_assignment_plan(model: MegaGemmLlama) -> dict[str, _AssignmentSlot]:
    plan: dict[str, _AssignmentSlot] = {}
    for name in model.state_dict().keys():
        module, field = _resolve_module_field(model, name)
        if field in module._parameters:
            existing = module._parameters[field]
            plan[name] = _AssignmentSlot(
                module=module,
                field=field,
                kind="parameter",
                clear_runtime_caches=False,
                requires_grad=bool(existing.requires_grad) if existing is not None else False,
            )
        elif field in module._buffers:
            plan[name] = _AssignmentSlot(
                module=module,
                field=field,
                kind="buffer",
                clear_runtime_caches=hasattr(module, "_clear_runtime_caches"),
            )
        else:
            plan[name] = _AssignmentSlot(
                module=module,
                field=field,
                kind="attr",
                clear_runtime_caches=False,
            )
    return plan


def _assign_tensor_to_slot(slot: _AssignmentSlot, tensor: torch.Tensor) -> None:
    module = slot.module
    if slot.kind == "parameter":
        module._parameters[slot.field] = nn.Parameter(tensor, requires_grad=slot.requires_grad)
        return
    if slot.kind == "buffer":
        module._buffers[slot.field] = tensor
        if slot.clear_runtime_caches:
            try:
                module._clear_runtime_caches()
            except Exception:
                pass
        return
    setattr(module, slot.field, tensor)


def _assign_tensor_to_model(
    model: MegaGemmLlama,
    name: str,
    tensor: torch.Tensor,
    *,
    assignment_plan: Optional[dict[str, _AssignmentSlot]] = None,
) -> None:
    if assignment_plan is not None:
        slot = assignment_plan.get(name)
        if slot is not None:
            _assign_tensor_to_slot(slot, tensor)
            return

    module, field = _resolve_module_field(model, name)
    if field in module._parameters:
        existing = module._parameters[field]
        _assign_tensor_to_slot(
            _AssignmentSlot(
                module=module,
                field=field,
                kind="parameter",
                clear_runtime_caches=False,
                requires_grad=bool(existing.requires_grad) if existing is not None else False,
            ),
            tensor,
        )
        return
    if field in module._buffers:
        _assign_tensor_to_slot(
            _AssignmentSlot(
                module=module,
                field=field,
                kind="buffer",
                clear_runtime_caches=hasattr(module, "_clear_runtime_caches"),
            ),
            tensor,
        )
        return
    _assign_tensor_to_slot(
        _AssignmentSlot(
            module=module,
            field=field,
            kind="attr",
            clear_runtime_caches=False,
        ),
        tensor,
    )


def _assign_state_tensors(
    model: MegaGemmLlama,
    state: dict[str, torch.Tensor],
    *,
    tie_word_embeddings: bool,
    assignment_plan: Optional[dict[str, _AssignmentSlot]] = None,
) -> tuple[list[str], list[str]]:
    state_keys = set(model.state_dict().keys())
    payload_keys = set(state.keys())

    unexpected = sorted(
        key for key in payload_keys
        if key not in state_keys and not (key == "lm_head.weight" and tie_word_embeddings)
    )
    for key in sorted(payload_keys):
        _assign_tensor_to_model(model, key, state[key], assignment_plan=assignment_plan)

    missing = sorted(
        key for key in state_keys
        if key not in payload_keys and not (key == "lm_head.weight" and tie_word_embeddings)
    )
    return missing, unexpected


def _retie_shared_weights(model: MegaGemmLlama, config: LlamaConfig) -> None:
    if config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight


def _materialize_runtime_meta_buffers(
    model: MegaGemmLlama,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> None:
    device = torch.device(device)
    for layer in model.layers:
        scalar = getattr(layer, "layer_scalar", None)
        if scalar is not None and scalar.device.type == "meta":
            layer.layer_scalar = torch.ones(1, device=device, dtype=dtype)
    if hasattr(model, "_refresh_gemma4_runtime_buffers"):
        model._refresh_gemma4_runtime_buffers(device=device, dtype=dtype)


def _restore_awq_scales_dtype(model: MegaGemmLlama) -> None:
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module.scales = module.scales.to(device=module.qweight.device, dtype=torch.float16)


def _materialize_model_from_payload_state(
    payload: dict[str, torch.Tensor],
    *,
    config: LlamaConfig,
    runtime_dtype: torch.dtype,
    device: str,
    timing: dict[str, object],
    quantization: str,
    quantization_config: Optional[dict[str, Any]] = None,
    sparsity_config: Optional[dict[str, Any]] = None,
    payload_device: Optional[str | torch.device] = None,
) -> MegaGemmLlama:
    native_w4 = is_native_w4a16_config(quantization_config)
    awq_qweight_layout = _awq_qweight_layout_from_quantization_config(quantization_config)
    timing["awq_qweight_layout"] = awq_qweight_layout
    target_device = torch.device(device)
    payload_device_obj = torch.device(payload_device) if payload_device is not None else torch.device("cpu")
    payload_on_target_device = payload_device_obj == target_device

    phase_start = time.perf_counter()
    try:
        expanded_sparse_weights = (
            []
            if _is_native_w4_sparse_config(sparsity_config)
            else expand_sparse24_payload(payload, sparsity_config)
        )
    except (TypeError, ValueError) as exc:
        raise MGXFormatError(f"Could not expand MGX 2:4 payload: {exc}") from exc
    timing["sparse24_expand_seconds"] = time.perf_counter() - phase_start
    timing["sparse24_expanded_tensor_count"] = len(expanded_sparse_weights)

    cos_cache = payload.pop("__mgx__.cos_cache", None)
    sin_cache = payload.pop("__mgx__.sin_cache", None)
    if config.tie_word_embeddings:
        payload.pop("lm_head.weight", None)

    phase_start = time.perf_counter()
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(runtime_dtype)
    with torch.device("meta"):
        model = MegaGemmLlama(config)
        if quantization == "int8":
            _replace_int8_modules_from_state(model, payload)
        elif quantization == "int4":
            group_size = int((quantization_config or {}).get("group_size", 0) or 0)
            if group_size <= 0:
                raise MGXFormatError("MGX INT4 artifact is missing quantization_config.group_size.")
            if native_w4:
                _replace_native_w4_modules(
                    model,
                    set(payload.keys()),
                    group_size=group_size,
                    sparse24=bool((quantization_config or {}).get("sparse24")),
                )
            else:
                _replace_int4_modules_from_state(model, payload, group_size=group_size)
    torch.set_default_dtype(prev_dtype)
    timing["model_meta_init_seconds"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    assignment_plan = _build_assignment_plan(model)
    timing["assignment_plan_build_seconds"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    missing, unexpected = _assign_state_tensors(
        model,
        payload,
        tie_word_embeddings=config.tie_word_embeddings,
        assignment_plan=assignment_plan,
    )
    if unexpected:
        raise MGXFormatError(f"Unexpected tensors found while loading MGX artifact: {unexpected}")
    if missing:
        raise MGXFormatError(
            "MGX artifact is missing required tensors: " + ", ".join(missing[:8])
        )
    timing["state_assign_seconds"] = time.perf_counter() - phase_start

    _retie_shared_weights(model, config)
    _materialize_runtime_meta_buffers(
        model,
        device=target_device if payload_on_target_device else "cpu",
        dtype=runtime_dtype,
    )
    if quantization == "int4" and not native_w4:
        _apply_awq_qweight_layout(model, awq_qweight_layout)

    if cos_cache is not None and sin_cache is not None:
        model.cos_cache = cos_cache
        model.sin_cache = sin_cache
    elif getattr(model, "cos_cache", None) is None or model.cos_cache.device.type == "meta":
        model.cos_cache, model.sin_cache = precompute_freqs_cis(
            config.rotary_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

    if payload_on_target_device:
        if quantization == "int4" and not native_w4:
            _restore_awq_scales_dtype(model)
        timing["model_to_device_seconds"] = 0.0
    else:
        phase_start = time.perf_counter()
        model = model.to(device=device, dtype=runtime_dtype)
        if quantization == "int4" and not native_w4:
            _restore_awq_scales_dtype(model)
        timing["model_to_device_seconds"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    model._move_rope_to_device(device)
    timing["rope_cache_seconds"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    timing["sparse24_runtime"] = (
        native_w4a16_runtime_stats(model)
        if _is_native_w4_sparse_config(sparsity_config)
        else prepare_sparse24_runtime(model, sparsity_config, device=device)
    )
    timing["sparse24_prepare_seconds"] = time.perf_counter() - phase_start
    model.eval()
    return model


def _load_from_mgx_payload_cache_streaming(
    payload_cache_path: Path,
    config: LlamaConfig,
    runtime_dtype: torch.dtype,
    device: str,
    timing: dict[str, object],
    quantization: str,
    quantization_config: Optional[dict[str, Any]] = None,
    sparsity_config: Optional[dict[str, Any]] = None,
) -> MegaGemmLlama:
    from safetensors import safe_open

    native_w4 = is_native_w4a16_config(quantization_config)

    phase_start = time.perf_counter()
    with safe_open(str(payload_cache_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    timing["payload_key_scan_seconds"] = time.perf_counter() - phase_start
    if not _is_native_w4_sparse_config(sparsity_config):
        try:
            validate_sparse24_config(sparsity_config, tensor_names=keys)
        except (TypeError, ValueError) as exc:
            raise MGXFormatError(f"Invalid MGX 2:4 payload cache: {exc}") from exc
    sparse_entries_by_values = {
        str(entry["values"]): entry
        for entry in (sparsity_config or {}).get("entries", [])
        if "values" in entry
    }
    sparse_metadata_keys = {
        str(entry["metadata"])
        for entry in (sparsity_config or {}).get("entries", [])
        if "metadata" in entry
    }

    expected_keys = {
        "__mgx__.cos_cache",
        "__mgx__.sin_cache",
    }
    awq_qweight_layout = _awq_qweight_layout_from_quantization_config(quantization_config)
    timing["awq_qweight_layout"] = awq_qweight_layout

    phase_start = time.perf_counter()
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(runtime_dtype)
    with torch.device("meta"):
        model = MegaGemmLlama(config)
        if quantization == "int8":
            _replace_int8_modules_from_keys(model, keys)
        elif quantization == "int4":
            group_size = int((quantization_config or {}).get("group_size", 0) or 0)
            if group_size <= 0:
                raise MGXFormatError("MGX INT4 payload cache load requires quantization_config.group_size.")
            if native_w4:
                _replace_native_w4_modules(
                    model,
                    keys,
                    group_size=group_size,
                    sparse24=bool((quantization_config or {}).get("sparse24")),
                )
            else:
                _replace_int4_modules_from_keys(model, keys, group_size=group_size)
    torch.set_default_dtype(prev_dtype)
    timing["model_meta_init_seconds"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    assignment_plan = _build_assignment_plan(model)
    timing["assignment_plan_build_seconds"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    hydration_mode = _resolve_payload_hydration_mode(device)
    profile_topk = _resolve_payload_profile_topk()
    timing["payload_hydration_mode"] = hydration_mode
    timing["payload_profile_topk_limit"] = profile_topk
    clone_seconds = 0.0
    int4_qweight_clone_count = 0
    payload_get_tensor_seconds = 0.0
    payload_device_transfer_seconds = 0.0
    payload_pin_memory_seconds = 0.0
    payload_dtype_cast_seconds = 0.0
    payload_assign_lookup_seconds = 0.0
    payload_assign_write_seconds = 0.0
    payload_special_case_seconds = 0.0
    payload_total_tensor_count = 0
    payload_float_tensor_count = 0
    payload_int8_tensor_count = 0
    payload_parameter_tensor_count = 0
    payload_buffer_tensor_count = 0
    payload_attr_tensor_count = 0
    payload_pinned_tensor_count = 0
    payload_top_get_tensor_tensors: list[dict[str, Any]] = []
    payload_top_transfer_tensors: list[dict[str, Any]] = []
    pending_transfer_events: list[tuple[str, tuple[int, ...], str, int, torch.cuda.Event, torch.cuda.Event]] = []
    device_type = torch.device(device).type
    sparse24_expand_seconds = 0.0
    sparse24_expanded_tensor_count = 0

    open_start = time.perf_counter()
    handle_device = device if hydration_mode == "direct_device" else "cpu"
    timing["payload_handle_device"] = str(handle_device)
    handle_ctx = safe_open(str(payload_cache_path), framework="pt", device=handle_device)
    handle = handle_ctx.__enter__()
    timing["payload_open_seconds"] = time.perf_counter() - open_start
    try:
        sorted_start = time.perf_counter()
        keys = sorted(handle.keys())
        timing["payload_sorted_keys_seconds"] = time.perf_counter() - sorted_start
        for key in keys:
            if key in sparse_metadata_keys:
                # Its paired values entry expands both physical tensors into the
                # logical dense-pruned weight, one layer at a time.
                continue
            if key == "lm_head.weight" and config.tie_word_embeddings:
                special_start = time.perf_counter()
                expected_keys.add(key)
                payload_special_case_seconds += time.perf_counter() - special_start
                continue
            get_tensor_start = time.perf_counter()
            tensor = handle.get_tensor(key)
            get_tensor_elapsed = time.perf_counter() - get_tensor_start
            payload_get_tensor_seconds += get_tensor_elapsed
            sparse_entry = sparse_entries_by_values.get(key)
            if sparse_entry is not None:
                metadata_start = time.perf_counter()
                metadata = handle.get_tensor(str(sparse_entry["metadata"]))
                payload_get_tensor_seconds += time.perf_counter() - metadata_start
                expand_start = time.perf_counter()
                try:
                    tensor = unpack_sparse24_weight(tensor, metadata, sparse_entry["shape"])
                except (TypeError, ValueError) as exc:
                    raise MGXFormatError(
                        f"Could not expand streamed MGX 2:4 weight {sparse_entry['weight']}: {exc}"
                    ) from exc
                sparse24_expand_seconds += time.perf_counter() - expand_start
                sparse24_expanded_tensor_count += 1
                payload_total_tensor_count += 2
                key = str(sparse_entry["weight"])
            else:
                payload_total_tensor_count += 1
            tensor_shape = tuple(int(dim) for dim in tensor.shape)
            tensor_dtype = str(tensor.dtype).replace("torch.", "")
            tensor_num_bytes = _tensor_num_bytes(tensor)
            _update_top_payload_entries(
                payload_top_get_tensor_tensors,
                {
                    "key": key,
                    "shape": list(tensor_shape),
                    "dtype": tensor_dtype,
                    "num_bytes": tensor_num_bytes,
                    "get_tensor_seconds": get_tensor_elapsed,
                },
                metric="get_tensor_seconds",
                limit=profile_topk,
            )
            if tensor.is_floating_point():
                payload_float_tensor_count += 1
            if tensor.dtype == torch.int8:
                payload_int8_tensor_count += 1
            needs_runtime_cast = (
                tensor.is_floating_point()
                and tensor.dtype != runtime_dtype
                and not (quantization == "int4" and key.endswith(".scales"))
            )
            if hydration_mode in {"cpu_stage_streaming", "cpu_stage_pinned"} and device_type == "cuda":
                non_blocking = hydration_mode == "cpu_stage_pinned"
                if hydration_mode == "cpu_stage_pinned":
                    pin_start = time.perf_counter()
                    tensor = tensor.pin_memory()
                    payload_pin_memory_seconds += time.perf_counter() - pin_start
                    payload_pinned_tensor_count += 1
                if non_blocking:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    if needs_runtime_cast:
                        tensor = tensor.to(device=device, dtype=runtime_dtype, non_blocking=True)
                    else:
                        tensor = tensor.to(device=device, non_blocking=True)
                    end_event.record()
                    pending_transfer_events.append(
                        (key, tensor_shape, tensor_dtype, tensor_num_bytes, start_event, end_event)
                    )
                else:
                    transfer_start = time.perf_counter()
                    if needs_runtime_cast:
                        tensor = tensor.to(device=device, dtype=runtime_dtype)
                    else:
                        tensor = tensor.to(device=device)
                    transfer_elapsed = time.perf_counter() - transfer_start
                    payload_device_transfer_seconds += transfer_elapsed
                    _update_top_payload_entries(
                        payload_top_transfer_tensors,
                        {
                            "key": key,
                            "shape": list(tensor_shape),
                            "dtype": tensor_dtype,
                            "num_bytes": tensor_num_bytes,
                            "device_transfer_seconds": transfer_elapsed,
                        },
                        metric="device_transfer_seconds",
                        limit=profile_topk,
                    )
                needs_runtime_cast = False
            if device == "cpu":
                clone_start = time.perf_counter()
                tensor = tensor.contiguous().clone()
                clone_seconds += time.perf_counter() - clone_start
            elif (
                quantization == "int4"
                and not native_w4
                and awq_qweight_layout != _AWQ_QWEIGHT_LAYOUT_DECODE
                and key.endswith(".qweight")
            ):
                # Re-own streamed AWQ qweight tensors on the runtime device.
                # The lazy decode transpose replaces qweight storage in-place, and
                # safetensors CUDA-backed tensors can otherwise keep the original
                # packed storage alive, effectively duplicating the full AWQ pack
                # after the first warm decode.
                clone_start = time.perf_counter()
                tensor = tensor.contiguous().clone()
                clone_seconds += time.perf_counter() - clone_start
                int4_qweight_clone_count += 1
            if needs_runtime_cast:
                cast_start = time.perf_counter()
                tensor = tensor.to(dtype=runtime_dtype)
                payload_dtype_cast_seconds += time.perf_counter() - cast_start

            if key == "__mgx__.cos_cache":
                special_start = time.perf_counter()
                model.cos_cache = tensor
                expected_keys.add("cos_cache")
                payload_special_case_seconds += time.perf_counter() - special_start
                continue
            if key == "__mgx__.sin_cache":
                special_start = time.perf_counter()
                model.sin_cache = tensor
                expected_keys.add("sin_cache")
                payload_special_case_seconds += time.perf_counter() - special_start
                continue

            lookup_start = time.perf_counter()
            slot = assignment_plan.get(key)
            if slot is None:
                module, field = _resolve_module_field(model, key)
                if field in module._parameters:
                    slot = _AssignmentSlot(
                        module=module,
                        field=field,
                        kind="parameter",
                        clear_runtime_caches=False,
                        requires_grad=bool(module._parameters[field].requires_grad)
                        if module._parameters[field] is not None else False,
                    )
                elif field in module._buffers:
                    slot = _AssignmentSlot(
                        module=module,
                        field=field,
                        kind="buffer",
                        clear_runtime_caches=hasattr(module, "_clear_runtime_caches"),
                    )
                else:
                    slot = _AssignmentSlot(
                        module=module,
                        field=field,
                        kind="attr",
                        clear_runtime_caches=False,
                    )
                assignment_plan[key] = slot
            payload_assign_lookup_seconds += time.perf_counter() - lookup_start

            if slot.kind == "parameter":
                payload_parameter_tensor_count += 1
            elif slot.kind == "buffer":
                payload_buffer_tensor_count += 1
            else:
                payload_attr_tensor_count += 1

            assign_start = time.perf_counter()
            _assign_tensor_to_slot(slot, tensor)
            payload_assign_write_seconds += time.perf_counter() - assign_start
            expected_keys.add(key)
        if pending_transfer_events:
            torch.cuda.synchronize(torch.device(device))
            for key, tensor_shape, tensor_dtype, tensor_num_bytes, start_event, end_event in pending_transfer_events:
                transfer_elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
                payload_device_transfer_seconds += transfer_elapsed
                _update_top_payload_entries(
                    payload_top_transfer_tensors,
                    {
                        "key": key,
                        "shape": list(tensor_shape),
                        "dtype": tensor_dtype,
                        "num_bytes": tensor_num_bytes,
                        "device_transfer_seconds": transfer_elapsed,
                    },
                    metric="device_transfer_seconds",
                    limit=profile_topk,
                )
    finally:
        handle_ctx.__exit__(None, None, None)
    timing["payload_stream_assign_seconds"] = time.perf_counter() - phase_start
    timing["payload_get_tensor_seconds"] = payload_get_tensor_seconds
    timing["payload_device_transfer_seconds"] = payload_device_transfer_seconds
    timing["payload_pin_memory_seconds"] = payload_pin_memory_seconds
    timing["payload_dtype_cast_seconds"] = payload_dtype_cast_seconds
    timing["payload_assign_lookup_seconds"] = payload_assign_lookup_seconds
    timing["payload_assign_write_seconds"] = payload_assign_write_seconds
    timing["payload_special_case_seconds"] = payload_special_case_seconds
    timing["payload_total_tensor_count"] = payload_total_tensor_count
    timing["payload_float_tensor_count"] = payload_float_tensor_count
    timing["payload_int8_tensor_count"] = payload_int8_tensor_count
    timing["payload_parameter_tensor_count"] = payload_parameter_tensor_count
    timing["payload_buffer_tensor_count"] = payload_buffer_tensor_count
    timing["payload_attr_tensor_count"] = payload_attr_tensor_count
    timing["payload_pinned_tensor_count"] = payload_pinned_tensor_count
    timing["payload_top_get_tensor_tensors"] = payload_top_get_tensor_tensors
    timing["payload_top_transfer_tensors"] = payload_top_transfer_tensors
    timing["sparse24_expand_seconds"] = sparse24_expand_seconds
    timing["sparse24_expanded_tensor_count"] = sparse24_expanded_tensor_count
    if clone_seconds > 0:
        timing["payload_clone_seconds"] = clone_seconds
    if int4_qweight_clone_count > 0:
        timing["payload_int4_qweight_clone_count"] = int4_qweight_clone_count

    _retie_shared_weights(model, config)
    _materialize_runtime_meta_buffers(model, device=device, dtype=runtime_dtype)
    if quantization == "int4" and not native_w4:
        _apply_awq_qweight_layout(model, awq_qweight_layout)

    state_keys = set(model.state_dict().keys())
    missing = sorted(
        key for key in state_keys
        if key not in expected_keys and not (key == "lm_head.weight" and config.tie_word_embeddings)
    )
    if missing:
        raise MGXFormatError(
            "MGX payload cache is missing required tensors: " + ", ".join(missing[:8])
        )

    if not hasattr(model, "cos_cache") or model.cos_cache is None:
        model.cos_cache, model.sin_cache = precompute_freqs_cis(
            config.rotary_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        model._move_rope_to_device(device)
    timing["state_assign_seconds"] = 0.0
    timing["model_to_device_seconds"] = 0.0
    timing["rope_cache_seconds"] = 0.0
    phase_start = time.perf_counter()
    timing["sparse24_runtime"] = (
        native_w4a16_runtime_stats(model)
        if _is_native_w4_sparse_config(sparsity_config)
        else prepare_sparse24_runtime(model, sparsity_config, device=device)
    )
    timing["sparse24_prepare_seconds"] = time.perf_counter() - phase_start
    return model


def _load_from_mgx_payload_cache_bulk(
    payload_cache_path: Path,
    config: LlamaConfig,
    runtime_dtype: torch.dtype,
    device: str,
    timing: dict[str, object],
    quantization: str,
    quantization_config: Optional[dict[str, Any]] = None,
    sparsity_config: Optional[dict[str, Any]] = None,
    *,
    load_device: str = "cpu",
    hydration_mode: str = "cpu_bulk",
) -> MegaGemmLlama:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "Install safetensors to load MGX payload caches: pip install safetensors"
        ) from exc

    timing["payload_hydration_mode"] = hydration_mode
    timing["payload_handle_device"] = str(load_device)
    timing["payload_profile_topk_limit"] = 0
    timing["payload_open_seconds"] = 0.0
    timing["payload_sorted_keys_seconds"] = 0.0
    timing["payload_get_tensor_seconds"] = 0.0
    timing["payload_device_transfer_seconds"] = 0.0
    timing["payload_pin_memory_seconds"] = 0.0
    timing["payload_assign_lookup_seconds"] = 0.0
    timing["payload_assign_write_seconds"] = 0.0
    timing["payload_special_case_seconds"] = 0.0
    timing["payload_pinned_tensor_count"] = 0
    timing["payload_top_get_tensor_tensors"] = []
    timing["payload_top_transfer_tensors"] = []

    phase_start = time.perf_counter()
    payload = load_file(str(payload_cache_path), device=load_device)
    timing["payload_bulk_load_seconds"] = time.perf_counter() - phase_start
    timing["payload_total_tensor_count"] = len(payload)
    timing["payload_float_tensor_count"] = sum(
        1 for tensor in payload.values() if tensor.is_floating_point()
    )
    timing["payload_int8_tensor_count"] = sum(
        1 for tensor in payload.values() if tensor.dtype == torch.int8
    )
    timing["payload_parameter_tensor_count"] = 0
    timing["payload_buffer_tensor_count"] = 0
    timing["payload_attr_tensor_count"] = 0
    timing["payload_stream_assign_seconds"] = 0.0

    return _materialize_model_from_payload_state(
        payload,
        config=config,
        runtime_dtype=runtime_dtype,
        device=device,
        timing=timing,
        quantization=quantization,
        quantization_config=quantization_config,
        sparsity_config=sparsity_config,
        payload_device=load_device,
    )


def _load_from_mgx_runtime_payload_cache(
    runtime_cache_path: Path,
    runtime_index_path: Path,
    config: LlamaConfig,
    runtime_dtype: torch.dtype,
    device: str,
    timing: dict[str, object],
    quantization: str,
    quantization_config: Optional[dict[str, Any]] = None,
    sparsity_config: Optional[dict[str, Any]] = None,
) -> MegaGemmLlama:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "Install safetensors to load MGX runtime payload caches: pip install safetensors"
        ) from exc

    runtime_index = _load_runtime_payload_cache_index(runtime_index_path)
    entries = runtime_index.get("entries")
    blobs_meta = runtime_index.get("blobs")
    if not isinstance(entries, list) or not isinstance(blobs_meta, list):
        raise MGXFormatError(f"Invalid runtime payload cache index at {runtime_index_path}")

    load_device = device if torch.device(device).type == "cuda" else "cpu"
    timing["payload_hydration_mode"] = "runtime_packed"
    timing["payload_handle_device"] = str(load_device)
    timing["payload_profile_topk_limit"] = 0
    timing["payload_open_seconds"] = 0.0
    timing["payload_sorted_keys_seconds"] = 0.0
    timing["payload_get_tensor_seconds"] = 0.0
    timing["payload_device_transfer_seconds"] = 0.0
    timing["payload_pin_memory_seconds"] = 0.0
    timing["payload_assign_lookup_seconds"] = 0.0
    timing["payload_assign_write_seconds"] = 0.0
    timing["payload_special_case_seconds"] = 0.0
    timing["payload_pinned_tensor_count"] = 0
    timing["payload_top_get_tensor_tensors"] = []
    timing["payload_top_transfer_tensors"] = []
    timing["payload_stream_assign_seconds"] = 0.0

    phase_start = time.perf_counter()
    packed_blobs = load_file(str(runtime_cache_path), device=load_device)
    timing["payload_bulk_load_seconds"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    payload: dict[str, torch.Tensor] = {}
    payload_float_tensor_count = 0
    payload_int8_tensor_count = 0
    payload_runtime_cast_seconds = 0.0
    for entry in entries:
        key = str(entry["key"])
        blob_name = str(entry["blob"])
        offset = int(entry["offset"])
        numel = int(entry["numel"])
        shape = tuple(int(dim) for dim in entry["shape"])
        blob = packed_blobs[blob_name]
        tensor = blob.narrow(0, offset, numel).view(shape)
        if tensor.is_floating_point():
            payload_float_tensor_count += 1
        if tensor.dtype == torch.int8:
            payload_int8_tensor_count += 1
        needs_runtime_cast = (
            tensor.is_floating_point()
            and tensor.dtype != runtime_dtype
            and not (quantization == "int4" and key.endswith(".scales"))
        )
        if needs_runtime_cast:
            cast_start = time.perf_counter()
            tensor = tensor.to(dtype=runtime_dtype)
            payload_runtime_cast_seconds += time.perf_counter() - cast_start
        payload[key] = tensor
    timing["payload_packed_view_seconds"] = time.perf_counter() - phase_start
    timing["payload_runtime_cast_seconds"] = payload_runtime_cast_seconds
    timing["payload_total_tensor_count"] = len(entries)
    timing["payload_float_tensor_count"] = payload_float_tensor_count
    timing["payload_int8_tensor_count"] = payload_int8_tensor_count
    timing["payload_parameter_tensor_count"] = 0
    timing["payload_buffer_tensor_count"] = 0
    timing["payload_attr_tensor_count"] = 0
    timing["payload_runtime_cache_blob_count"] = len(blobs_meta)

    return _materialize_model_from_payload_state(
        payload,
        config=config,
        runtime_dtype=runtime_dtype,
        device=device,
        timing=timing,
        quantization=quantization,
        quantization_config=quantization_config,
        sparsity_config=sparsity_config,
        payload_device=load_device,
    )


def export_to_mgx(
    model_source: str,
    out_path: str | os.PathLike[str],
    dtype: torch.dtype | str,
    quantize: Optional[str],
    target_backend: str = "megagemm-cuda",
    emit_payload_cache: bool = False,
    payload_cache_dir: Optional[str | os.PathLike[str]] = None,
    export_mode: Optional[str] = None,
    sparsity: Optional[str] = None,
) -> dict[str, Any]:
    """
    Compile a Hugging Face checkpoint or local snapshot into an MGX artifact.
    """
    export_dtype = _canonical_dtype(dtype, allow_fp32=False)
    quant_mode = _normalize_quantize_mode(quantize)
    sparsity_mode = _normalize_sparsity_mode(sparsity)
    export_mode = _normalize_export_mode(export_mode)

    resolved_source, source_kind = resolve_model_source(model_source)
    source_path = Path(resolved_source).resolve()
    hf_config = _load_config(str(source_path))
    config = LlamaConfig.from_dict(hf_config)
    quant_mode = _validate_export_request(config, hf_config, quant_mode, target_backend)
    if sparsity_mode is not None and quant_mode not in {None, "native-int4"}:
        raise NotImplementedError(
            "MGX 2:4 cannot be combined with INT8 or AWQ checkpoints; it supports "
            "FP16/BF16 or the standalone native W4A16 backend."
        )

    load_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_from_hf(
        str(source_path),
        dtype=export_dtype,
        device=load_device,
        quantize=None if quant_mode == "native-int4" else quant_mode,
    )
    model.eval()

    if quant_mode == "native-int4":
        try:
            native_input_dims = [
                int(module.in_features)
                for name, module in model.named_modules()
                if isinstance(module, nn.Linear) and name != "lm_head"
            ]
            native_group_size = next(
                (
                    candidate
                    for candidate in (NATIVE_W4A16_GROUP_SIZE, 64, 32, 16, 8)
                    if native_input_dims and all(dim % candidate == 0 for dim in native_input_dims)
                ),
                0,
            )
            if native_group_size <= 0:
                raise ValueError(
                    "eligible Linear input dimensions have no common W4A16 group size"
                )
            quantize_model_native_w4a16(
                model,
                group_size=native_group_size,
                sparse24=sparsity_mode == "2:4",
            )
        except ValueError as exc:
            raise MGXFormatError(f"Native W4A16 export failed: {exc}") from exc

    sparse24_weight_names = (
        sparse24_model_weight_names(model) if sparsity_mode == "2:4" else []
    )
    state, state_export_meta = _export_runtime_state(model, quantization=quant_mode)
    del model
    if load_device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    sparsity_config = None
    if sparsity_mode == "2:4" and quant_mode != "native-int4":
        try:
            sparsity_config = pack_model_state_sparse24(
                None,
                state,
                weight_names=sparse24_weight_names,
            )
        except ValueError as exc:
            raise MGXFormatError(f"MGX 2:4 export failed: {exc}") from exc
    elif sparsity_mode == "2:4":
        sparsity_config = _native_w4_sparse_config(state_export_meta)

    output_path = Path(out_path).expanduser().resolve()
    if export_mode == "normal":
        try:
            from safetensors.torch import save as save_safetensors
        except ImportError as exc:
            raise ImportError(
                "Install safetensors to export MGX artifacts: pip install safetensors"
            ) from exc

        payload = save_safetensors(state)
        payload_sha256 = _sha256_bytes(payload)
        manifest = _build_manifest(
            model_source=model_source,
            source_path=source_path,
            config=config,
            dtype=export_dtype,
            quantize=quant_mode,
            target_backend=target_backend,
            state=state,
            payload_sha256=payload_sha256,
            hf_config=hf_config,
            state_export_meta=state_export_meta,
            sparsity_config=sparsity_config,
        )
        manifest["source_kind"] = source_kind
        del state
        _write_mgx_artifact(output_path, manifest, payload)
    else:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError(
                "Install safetensors to export MGX artifacts: pip install safetensors"
            ) from exc

        payload_tmp = tempfile.NamedTemporaryFile(
            prefix=output_path.stem + "-",
            suffix=".payload.safetensors",
            dir=str(output_path.parent),
            delete=False,
        )
        payload_tmp_path = Path(payload_tmp.name)
        payload_tmp.close()
        try:
            save_file(state, str(payload_tmp_path))
            payload_sha256 = _sha256_file(payload_tmp_path)
            manifest = _build_manifest(
                model_source=model_source,
                source_path=source_path,
                config=config,
                dtype=export_dtype,
                quantize=quant_mode,
                target_backend=target_backend,
                state=state,
                payload_sha256=payload_sha256,
                hf_config=hf_config,
                state_export_meta=state_export_meta,
                sparsity_config=sparsity_config,
            )
            manifest["source_kind"] = source_kind

            # The CPU-side state dict is no longer needed once the payload file and
            # manifest have been materialized; releasing it here lowers the export
            # peak substantially for large artifacts such as Gemma 4 E2B.
            del state
            _write_mgx_artifact_from_payload_path(output_path, manifest, payload_tmp_path)
        finally:
            try:
                payload_tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    result = inspect_mgx(
        output_path,
        validate_payload_hash=False,
        payload_cache_dir=payload_cache_dir,
    )
    result["export_mode"] = export_mode
    if emit_payload_cache:
        result["payload_cache"] = prime_mgx_payload_cache(
            output_path,
            validate_payload_hash=False,
            payload_cache_dir=payload_cache_dir,
        )
    return result


def load_from_mgx(
    path: str | os.PathLike[str],
    device: str = "cuda",
    dtype_override: Optional[torch.dtype | str] = None,
    *,
    verify_payload_hash: Optional[bool] = None,
    prefer_payload_cache: Optional[bool] = None,
    payload_cache_dir: Optional[str | os.PathLike[str]] = None,
) -> MegaGemmLlama:
    """
    Load a MegaGemmLlama directly from a compiled MGX artifact.
    """
    timing: dict[str, object] = {
        "loader_kind": "mgx",
        "requested_path": str(path),
        "device": device,
    }
    total_start = time.perf_counter()

    phase_start = time.perf_counter()
    artifact = inspect_mgx(
        path,
        validate_payload_hash=False,
        payload_cache_dir=payload_cache_dir,
    )
    timing["inspect_header_manifest_seconds"] = time.perf_counter() - phase_start
    manifest = artifact["manifest"]
    config = LlamaConfig(**manifest["config"])
    runtime_dtype = _canonical_dtype(dtype_override or manifest["dtype"], allow_fp32=True)
    timing["runtime_dtype"] = str(runtime_dtype).replace("torch.", "")
    timing["quantization"] = manifest.get("quantization", "none")
    quantization_config = manifest.get("quantization_config") or None
    sparsity_config = manifest.get("sparsity_config") or None
    timing["sparsity"] = manifest.get("sparsity", "none")
    awq_qweight_layout = _awq_qweight_layout_from_quantization_config(quantization_config)
    timing["awq_qweight_layout"] = awq_qweight_layout
    verify_payload_hash = (
        _env_flag("MEGAGEMM_MGX_VERIFY_PAYLOAD", True)
        if verify_payload_hash is None
        else bool(verify_payload_hash)
    )
    prefer_payload_cache = (
        _env_flag("MEGAGEMM_MGX_PREFER_PAYLOAD_CACHE", True)
        if prefer_payload_cache is None
        else bool(prefer_payload_cache)
    )
    prefer_runtime_cache = _env_flag("MEGAGEMM_MGX_PREFER_RUNTIME_CACHE", False)
    timing["verify_payload_hash_enabled"] = verify_payload_hash
    timing["prefer_payload_cache"] = prefer_payload_cache
    timing["prefer_runtime_cache"] = prefer_runtime_cache
    payload_cache_path = Path(artifact["payload_cache_path"])
    timing["payload_cache_path"] = str(payload_cache_path)
    runtime_cache_path = Path(artifact["runtime_cache_path"])
    runtime_index_path = Path(artifact["runtime_cache_index_path"])
    timing["runtime_cache_path"] = str(runtime_cache_path)
    timing["runtime_cache_index_path"] = str(runtime_index_path)

    try:
        from safetensors.torch import load as load_safetensors
    except ImportError as exc:
        raise ImportError(
            "Install safetensors to load MGX artifacts: pip install safetensors"
        ) from exc

    payload = None
    payload_bytes = None
    expected_payload_sha256 = manifest.get("tensor_payload_sha256")
    timing["payload_hash_seconds"] = 0.0
    timing["runtime_cache_hit"] = False
    if prefer_runtime_cache and _runtime_payload_cache_is_valid(
        runtime_cache_path,
        runtime_index_path,
        expected_payload_sha256=expected_payload_sha256,
        expected_tensor_count=manifest.get("tensor_count"),
    ):
        try:
            timing["payload_source"] = "payload_runtime_cache_packed"
            timing["runtime_cache_hit"] = True
            timing["payload_cache_hit"] = True
            timing["payload_bytes"] = int(artifact["header"]["tensor_size"])
            model = _load_from_mgx_runtime_payload_cache(
                runtime_cache_path,
                runtime_index_path,
                config,
                runtime_dtype,
                device,
                timing,
                timing["quantization"],
                quantization_config,
                sparsity_config,
            )
            model.eval()
            model._mgx_manifest = manifest
            timing["total_seconds"] = time.perf_counter() - total_start
            model._load_timing = timing
            return model
        except Exception as exc:
            timing["runtime_cache_hit"] = False
            timing["runtime_cache_error"] = str(exc)

    if prefer_payload_cache and _payload_cache_is_valid(payload_cache_path, artifact["header"]["tensor_size"]):
        try:
            hydration_mode = _resolve_payload_hydration_mode(device)
            if hydration_mode == "cpu_bulk":
                timing["payload_source"] = "payload_cache_bulk"
            elif hydration_mode == "gpu_bulk":
                timing["payload_source"] = "payload_cache_gpu_bulk"
            else:
                timing["payload_source"] = "payload_cache_streaming"
            timing["payload_cache_hit"] = True
            timing["payload_bytes"] = int(artifact["header"]["tensor_size"])
            if hydration_mode in {"cpu_bulk", "gpu_bulk"}:
                model = _load_from_mgx_payload_cache_bulk(
                    payload_cache_path,
                    config,
                    runtime_dtype,
                    device,
                    timing,
                    timing["quantization"],
                    quantization_config,
                    sparsity_config,
                    load_device=device if hydration_mode == "gpu_bulk" else "cpu",
                    hydration_mode=hydration_mode,
                )
            else:
                model = _load_from_mgx_payload_cache_streaming(
                    payload_cache_path,
                    config,
                    runtime_dtype,
                    device,
                    timing,
                    timing["quantization"],
                    quantization_config,
                    sparsity_config,
                )
            model.eval()
            model._mgx_manifest = manifest
            timing["total_seconds"] = time.perf_counter() - total_start
            model._load_timing = timing
            return model
        except Exception as exc:
            timing["payload_cache_error"] = str(exc)
            payload = None

    if payload is None:
        timing["payload_source"] = "embedded"
        timing["payload_cache_hit"] = False
        phase_start = time.perf_counter()
        payload_bytes = _read_tensor_payload(Path(artifact["path"]), artifact["header"])
        timing["payload_read_seconds"] = time.perf_counter() - phase_start
        timing["payload_bytes"] = len(payload_bytes)

        if verify_payload_hash:
            phase_start = time.perf_counter()
            payload_sha256 = _sha256_bytes(payload_bytes)
            timing["payload_hash_seconds"] = time.perf_counter() - phase_start
            if expected_payload_sha256 and payload_sha256 != expected_payload_sha256:
                raise MGXFormatError(
                    f"MGX tensor payload hash mismatch for {artifact['path']}. "
                    f"Expected {expected_payload_sha256}, got {payload_sha256}."
                )
        else:
            timing["payload_hash_seconds"] = 0.0

        phase_start = time.perf_counter()
        payload = load_safetensors(payload_bytes)
        timing["payload_deserialize_seconds"] = time.perf_counter() - phase_start

    model = _materialize_model_from_payload_state(
        payload,
        config=config,
        runtime_dtype=runtime_dtype,
        device=device,
        timing=timing,
        quantization=timing["quantization"],
        quantization_config=quantization_config,
        sparsity_config=sparsity_config,
    )
    model._mgx_manifest = manifest
    timing["total_seconds"] = time.perf_counter() - total_start
    model._load_timing = timing
    return model

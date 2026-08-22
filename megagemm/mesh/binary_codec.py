"""Binary tensor frames for experimental MegaMesh shard transport.

The frame layout is intentionally small and dependency-light:

    magic[8] + header_len[u64-le] + header_json + raw tensor bytes...

The JSON header carries only metadata. Tensor payloads are contiguous raw bytes,
so this avoids the base64 expansion used by the compatibility JSON transport.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import struct
import threading
from typing import Any, Iterable, Mapping

import numpy as np
import torch


CONTENT_TYPE = "application/vnd.megamesh.tensor-frame"
MAGIC = b"MGMSHB1\0"
_PREFIX = struct.Struct("<8sQ")

_TORCH_TO_NUMPY = {
    torch.bfloat16: np.uint16,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}

_NAME_TO_TORCH = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


class PinnedTensorPool:
    """Reusable CPU pinned tensor buffers keyed by shape and dtype."""

    def __init__(self, *, enabled: bool = True, max_free_per_key: int = 8) -> None:
        self.enabled = bool(enabled)
        self.max_free_per_key = int(max_free_per_key)
        self._free: dict[tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]] = defaultdict(list)
        self._lock = threading.Lock()
        self.allocations = 0
        self.reuses = 0
        self.releases = 0
        self.pinned_allocations = 0
        self.pin_failures = 0

    def acquire(self, shape: Iterable[int], dtype: torch.dtype) -> tuple[torch.Tensor, bool]:
        key = (tuple(int(dim) for dim in shape), dtype)
        with self._lock:
            buffers = self._free.get(key)
            if buffers:
                self.reuses += 1
                return buffers.pop(), True

        pin_memory = bool(self.enabled and torch.cuda.is_available())
        try:
            tensor = torch.empty(key[0], dtype=dtype, device="cpu", pin_memory=pin_memory)
            if pin_memory:
                self.pinned_allocations += 1
        except Exception:
            self.pin_failures += 1
            tensor = torch.empty(key[0], dtype=dtype, device="cpu")
        self.allocations += 1
        return tensor, False

    def release(self, tensor: torch.Tensor) -> None:
        key = (tuple(int(dim) for dim in tensor.shape), tensor.dtype)
        with self._lock:
            buffers = self._free[key]
            if len(buffers) < self.max_free_per_key:
                buffers.append(tensor)
                self.releases += 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            free_buffers = sum(len(buffers) for buffers in self._free.values())
            free_bytes = sum(
                len(buffers) * torch.empty((), dtype=key[1]).element_size() * int(np.prod(key[0]))
                for key, buffers in self._free.items()
            )
        return {
            "enabled": self.enabled,
            "allocations": self.allocations,
            "pinned_allocations": self.pinned_allocations,
            "pin_failures": self.pin_failures,
            "reuses": self.reuses,
            "releases": self.releases,
            "free_buffers": free_buffers,
            "free_mb": round(free_bytes / 1024**2, 3),
        }


@dataclass
class TensorFrameParts:
    """A binary frame represented as sendable byte-like parts."""

    header: bytes
    chunks: list[memoryview] = field(default_factory=list)
    keepalive: list[Any] = field(default_factory=list)
    pool: PinnedTensorPool | None = None
    pooled_tensors: list[torch.Tensor] = field(default_factory=list)
    cuda_events: list[torch.cuda.Event] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return len(self.header) + sum(len(chunk) for chunk in self.chunks)

    def wait(self) -> None:
        for event in self.cuda_events:
            event.synchronize()
        self.cuda_events.clear()

    def iter_parts(self) -> Iterable[memoryview | bytes]:
        self.wait()
        yield self.header
        yield from self.chunks

    def release(self) -> None:
        if self.pool is not None:
            for tensor in self.pooled_tensors:
                self.pool.release(tensor)
        self.pooled_tensors.clear()
        self.keepalive.clear()
        self.chunks.clear()


def _tensor_to_array_view(tensor: torch.Tensor) -> tuple[np.ndarray, memoryview]:
    if tensor.dtype == torch.bfloat16:
        array = tensor.view(torch.uint16).numpy()
    else:
        array = tensor.numpy()
    return array, memoryview(array).cast("B")


def _tensor_to_part(
    name: str,
    tensor: torch.Tensor,
    *,
    pool: PinnedTensorPool | None = None,
) -> tuple[dict[str, Any], memoryview, list[Any], list[torch.Tensor], list[torch.cuda.Event]]:
    tensor = tensor.detach()
    if tensor.dtype not in _TORCH_TO_NUMPY:
        raise TypeError(f"MegaMesh binary codec does not support dtype {tensor.dtype}")

    keepalive: list[Any] = []
    pooled_tensors: list[torch.Tensor] = []
    events: list[torch.cuda.Event] = []
    source = tensor

    if source.device.type == "cuda":
        if pool is None:
            source = source.to("cpu").contiguous()
        else:
            cpu_tensor, _ = pool.acquire(source.shape, source.dtype)
            cpu_tensor.copy_(source, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(source.device))
            events.append(event)
            source = cpu_tensor
            pooled_tensors.append(cpu_tensor)
    elif source.device.type != "cpu":
        source = source.to("cpu").contiguous()
    elif not source.is_contiguous():
        source = source.contiguous()

    array, raw_view = _tensor_to_array_view(source)
    keepalive.extend([source, array])
    return (
        {
            "name": str(name),
            "shape": list(source.shape),
            "dtype": _dtype_name(source.dtype),
            "nbytes": len(raw_view),
        },
        raw_view,
        keepalive,
        pooled_tensors,
        events,
    )


def _tensor_to_bytes(name: str, tensor: torch.Tensor) -> tuple[dict[str, Any], bytes]:
    tensor = tensor.detach().to("cpu").contiguous()
    if tensor.dtype not in _TORCH_TO_NUMPY:
        raise TypeError(f"MegaMesh binary codec does not support dtype {tensor.dtype}")
    if tensor.dtype == torch.bfloat16:
        array = tensor.view(torch.uint16).numpy()
    else:
        array = tensor.numpy()
    raw = array.tobytes(order="C")
    return (
        {
            "name": str(name),
            "shape": list(tensor.shape),
            "dtype": _dtype_name(tensor.dtype),
            "nbytes": len(raw),
        },
        raw,
    )


def encode_tensor_frame_parts(
    meta: Mapping[str, Any] | None = None,
    tensors: Mapping[str, torch.Tensor] | None = None,
    *,
    pool: PinnedTensorPool | None = None,
) -> TensorFrameParts:
    """Encode metadata/tensors into sendable frame parts without concatenating payloads."""

    header = {
        "version": 1,
        "meta": dict(meta or {}),
        "tensors": [],
    }
    chunks: list[memoryview] = []
    keepalive: list[Any] = []
    pooled_tensors: list[torch.Tensor] = []
    cuda_events: list[torch.cuda.Event] = []
    for name, tensor in (tensors or {}).items():
        desc, raw_view, refs, pooled, events = _tensor_to_part(name, tensor, pool=pool)
        header["tensors"].append(desc)
        chunks.append(raw_view)
        keepalive.extend(refs)
        pooled_tensors.extend(pooled)
        cuda_events.extend(events)

    header_bytes = json.dumps(
        header,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    frame_header = _PREFIX.pack(MAGIC, len(header_bytes)) + header_bytes
    return TensorFrameParts(
        header=frame_header,
        chunks=chunks,
        keepalive=keepalive,
        pool=pool,
        pooled_tensors=pooled_tensors,
        cuda_events=cuda_events,
    )


def encode_tensor_frame(
    meta: Mapping[str, Any] | None = None,
    tensors: Mapping[str, torch.Tensor] | None = None,
) -> bytes:
    """Encode scalar metadata and named tensors into one binary frame."""

    header = {
        "version": 1,
        "meta": dict(meta or {}),
        "tensors": [],
    }
    chunks: list[bytes] = []
    for name, tensor in (tensors or {}).items():
        desc, raw = _tensor_to_bytes(name, tensor)
        header["tensors"].append(desc)
        chunks.append(raw)

    header_bytes = json.dumps(
        header,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _PREFIX.pack(MAGIC, len(header_bytes)) + header_bytes + b"".join(chunks)


def decode_tensor_frame(
    frame: bytes | bytearray | memoryview,
    *,
    device: str | torch.device = "cpu",
    copy_tensors: bool = True,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Decode a binary tensor frame into metadata and tensors."""

    if len(frame) < _PREFIX.size:
        raise ValueError("MegaMesh binary frame is too short")
    frame_view = memoryview(frame)
    magic, header_len = _PREFIX.unpack_from(frame_view, 0)
    if magic != MAGIC:
        raise ValueError("MegaMesh binary frame has invalid magic")

    header_start = _PREFIX.size
    header_end = header_start + int(header_len)
    if header_end > len(frame):
        raise ValueError("MegaMesh binary frame header is truncated")

    header = json.loads(frame_view[header_start:header_end].tobytes().decode("utf-8"))
    if int(header.get("version", 0)) != 1:
        raise ValueError(f"Unsupported MegaMesh binary frame version: {header.get('version')}")

    tensors: dict[str, torch.Tensor] = {}
    offset = header_end
    for desc in header.get("tensors", []):
        name = str(desc["name"])
        dtype_name = str(desc["dtype"])
        nbytes = int(desc["nbytes"])
        torch_dtype = _NAME_TO_TORCH.get(dtype_name)
        if torch_dtype is None:
            raise TypeError(f"MegaMesh binary codec does not support dtype {dtype_name}")

        end = offset + nbytes
        if end > len(frame):
            raise ValueError(f"MegaMesh binary frame tensor {name!r} is truncated")
        raw = frame_view[offset:end]
        offset = end

        array = np.frombuffer(raw, dtype=_TORCH_TO_NUMPY[torch_dtype])
        if copy_tensors:
            array = array.copy()
        tensor = torch.from_numpy(array).view(*[int(dim) for dim in desc["shape"]])
        if torch_dtype == torch.bfloat16:
            tensor = tensor.view(torch.bfloat16)
        tensors[name] = tensor.to(device=device, non_blocking=True)

    if offset != len(frame):
        raise ValueError("MegaMesh binary frame has trailing bytes")
    meta = header.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("MegaMesh binary frame meta must be an object")
    return meta, tensors


__all__ = [
    "CONTENT_TYPE",
    "PinnedTensorPool",
    "TensorFrameParts",
    "decode_tensor_frame",
    "encode_tensor_frame",
    "encode_tensor_frame_parts",
]

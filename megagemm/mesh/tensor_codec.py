"""JSON/base64 tensor payload codec for MegaMesh shard compatibility mode."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import torch


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


def tensor_to_payload(tensor: torch.Tensor) -> dict[str, Any]:
    """Encode a tensor as JSON-safe metadata + base64 raw bytes."""

    tensor = tensor.detach().to("cpu").contiguous()
    if tensor.dtype not in _TORCH_TO_NUMPY:
        raise TypeError(f"MegaMesh tensor codec does not support dtype {tensor.dtype}")
    if tensor.dtype is torch.bfloat16:
        array = tensor.view(torch.uint16).numpy()
    else:
        array = tensor.numpy()
    return {
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor.dtype),
        "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def tensor_from_payload(payload: dict[str, Any], *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Decode a tensor payload produced by :func:`tensor_to_payload`."""

    dtype_name = str(payload["dtype"])
    torch_dtype = _NAME_TO_TORCH.get(dtype_name)
    if torch_dtype is None:
        raise TypeError(f"MegaMesh tensor codec does not support dtype {dtype_name}")
    np_dtype = _TORCH_TO_NUMPY[torch_dtype]
    raw = base64.b64decode(str(payload["data"]).encode("ascii"))
    array = np.frombuffer(raw, dtype=np_dtype).copy()
    tensor = torch.from_numpy(array).view(*[int(dim) for dim in payload["shape"]])
    if torch_dtype is torch.bfloat16:
        tensor = tensor.view(torch.bfloat16)
    return tensor.to(device=device)


__all__ = ["tensor_from_payload", "tensor_to_payload"]

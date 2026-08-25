"""Model- and hardware-specific runtime policy selection.

The policy records measured execution choices without making benchmark runners
responsible for configuring the engine. Environment variables remain explicit
overrides for experiments and regression isolation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimePolicy:
    name: str = "generic"
    hardware: str = "generic"
    prefer_triton_rmsnorm: bool = False
    decode_prefer_step: bool = False
    reuse_request_scheduler: bool = False
    reason: str = "no model-specific measured policy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _config_int(config: Any, name: str) -> int:
    try:
        return int(getattr(config, name, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _gemma4_topology(config: Any) -> tuple[int, int, int, int]:
    return (
        _config_int(config, "num_hidden_layers"),
        _config_int(config, "hidden_size"),
        _config_int(config, "num_attention_heads"),
        _config_int(config, "num_key_value_heads"),
    )


def resolve_runtime_policy(config: Any, device_name: str = "") -> RuntimePolicy:
    """Resolve only choices backed by a model/hardware-specific measurement."""
    model_type = str(getattr(config, "model_type", "") or "").lower()
    normalized_device = str(device_name or "").upper()
    if model_type != "gemma4_text":
        return RuntimePolicy()

    topology = _gemma4_topology(config)
    if "L4" not in normalized_device:
        return RuntimePolicy(
            name="gemma4-generic",
            hardware=normalized_device or "unknown",
            reason="Gemma 4 topology recognized; no hardware-specific policy promoted",
        )

    if topology == (35, 1536, 8, 1):
        return RuntimePolicy(
            name="gemma4-e2b-l4",
            hardware="NVIDIA L4",
            prefer_triton_rmsnorm=True,
            decode_prefer_step=False,
            reuse_request_scheduler=False,
            reason=(
                "validated E2B L4 path: Triton RMSNorm and multi-step eager decode"
            ),
        )
    if topology == (42, 2560, 8, 2):
        return RuntimePolicy(
            name="gemma4-e4b-l4",
            hardware="NVIDIA L4",
            prefer_triton_rmsnorm=False,
            decode_prefer_step=True,
            reuse_request_scheduler=True,
            reason=(
                "validated E4B L4 path: native RMSNorm, decode_step, and scheduler reuse"
            ),
        )

    return RuntimePolicy(
        name="gemma4-l4-generic",
        hardware="NVIDIA L4",
        reason=f"unrecognized Gemma 4 L4 topology {topology!r}",
    )


def policy_bool(
    model: Any,
    env_name: str,
    policy_field: str,
    default: bool = False,
) -> bool:
    """Read an explicit environment override or fall back to model policy."""
    raw = os.environ.get(env_name, "").strip().lower()
    if raw:
        return raw in _TRUE_VALUES
    policy = getattr(model, "runtime_policy", None)
    return bool(getattr(policy, policy_field, default))


__all__ = ["RuntimePolicy", "policy_bool", "resolve_runtime_policy"]

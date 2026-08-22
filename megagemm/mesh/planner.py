"""Layer-stage planning helpers for MegaMesh.

This module is deliberately pure planning code. It does not import the
inference engine, kernels, scheduler, or decode path. Layer-shard execution can
consume these plans later without making replica mode depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Iterable, Sequence

from .protocol import MeshEndpoint, parse_worker_specs


@dataclass(frozen=True)
class LayerStage:
    """A contiguous layer range assigned to one MegaMesh worker."""

    stage_id: int
    layer_start: int
    layer_end: int
    worker_url: str
    name: str = ""
    weight: float = 1.0
    device: str = ""

    @property
    def num_layers(self) -> int:
        return int(self.layer_end - self.layer_start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "num_layers": self.num_layers,
            "worker_url": self.worker_url,
            "name": self.name,
            "weight": self.weight,
            "device": self.device,
        }


@dataclass(frozen=True)
class AgronNodeProfile:
    """Planning profile for one atomic MegaMesh shard node.

    ``speed`` is a relative layer-throughput score. It can be a hand-written
    weight, a measured layers/ms value, or a normalized score produced by an
    external benchmark. Higher means the node can own more layers.
    """

    name: str
    speed: float = 1.0
    max_layers: int | None = None
    fixed_ms: float = 0.0


@dataclass(frozen=True)
class AgronLinkProfile:
    """Directed TTP link profile between two shard nodes."""

    src: str
    dst: str
    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0


def _endpoint_url(endpoint: MeshEndpoint) -> str:
    return endpoint.url.strip().rstrip("/")


def _endpoint_keys(endpoint: MeshEndpoint) -> tuple[str, ...]:
    keys = [_endpoint_url(endpoint)]
    if endpoint.name:
        keys.insert(0, endpoint.name)
    try:
        keys.append(endpoint.base_url)
    except Exception:
        pass
    return tuple(dict.fromkeys(key for key in keys if key))


def _endpoint_label(endpoint: MeshEndpoint) -> str:
    return endpoint.name or _endpoint_url(endpoint)


def _node_profile_for(
    endpoint: MeshEndpoint,
    profiles: dict[str, AgronNodeProfile],
) -> AgronNodeProfile:
    for key in _endpoint_keys(endpoint):
        profile = profiles.get(key)
        if profile is not None:
            return profile
    return AgronNodeProfile(
        name=_endpoint_label(endpoint),
        speed=max(float(endpoint.weight), 1e-6),
    )


def _link_key(src: MeshEndpoint, dst: MeshEndpoint) -> tuple[str, str]:
    return (_endpoint_label(src), _endpoint_label(dst))


def _link_cost_ms(
    src: MeshEndpoint,
    dst: MeshEndpoint,
    links: dict[tuple[str, str], AgronLinkProfile],
    *,
    hidden_bytes: int,
    default_latency_ms: float,
    default_bandwidth_mbps: float,
) -> float:
    profile = None
    for src_key in _endpoint_keys(src):
        for dst_key in _endpoint_keys(dst):
            profile = links.get((src_key, dst_key))
            if profile is not None:
                break
        if profile is not None:
            break

    latency_ms = float(profile.latency_ms) if profile is not None else float(default_latency_ms)
    bandwidth_mbps = (
        float(profile.bandwidth_mbps)
        if profile is not None and float(profile.bandwidth_mbps) > 0
        else float(default_bandwidth_mbps)
    )
    if hidden_bytes <= 0 or bandwidth_mbps <= 0:
        return latency_ms
    transfer_ms = (float(hidden_bytes) / (bandwidth_mbps * 125_000.0)) * 1000.0
    return latency_ms + transfer_ms


def _bounded_compositions(
    total: int,
    limits: Sequence[int | None],
    *,
    max_candidates: int,
) -> tuple[list[list[int]], bool]:
    """Enumerate positive integer layer splits, capped for CLI responsiveness."""

    count = len(limits)
    results: list[list[int]] = []
    exhaustive = True

    def rec(idx: int, remaining: int, current: list[int]) -> None:
        nonlocal exhaustive
        if len(results) >= max_candidates:
            exhaustive = False
            return
        slots_left = count - idx
        if idx == count - 1:
            value = remaining
            limit = limits[idx]
            if value >= 1 and (limit is None or value <= int(limit)):
                results.append(current + [value])
            return
        limit = limits[idx]
        max_value = remaining - (slots_left - 1)
        if limit is not None:
            max_value = min(max_value, int(limit))
        for value in range(1, max_value + 1):
            rec(idx + 1, remaining - value, current + [value])
            if len(results) >= max_candidates:
                return

    rec(0, int(total), [])
    return results, exhaustive


def _weighted_seed_counts(total: int, profiles: Sequence[AgronNodeProfile]) -> list[int]:
    counts = _weighted_counts(total, [profile.speed for profile in profiles])
    for idx, profile in enumerate(profiles):
        if profile.max_layers is not None and counts[idx] > int(profile.max_layers):
            counts[idx] = int(profile.max_layers)
    deficit = total - sum(counts)
    while deficit > 0:
        best = None
        for idx, profile in enumerate(profiles):
            if profile.max_layers is not None and counts[idx] >= int(profile.max_layers):
                continue
            score = float(profile.speed) / max(float(counts[idx]), 1.0)
            if best is None or score > best[0]:
                best = (score, idx)
        if best is None:
            break
        counts[best[1]] += 1
        deficit -= 1
    if sum(counts) != total:
        raise ValueError("AGron node max_layers cannot cover all layers")
    return counts


def _neighbor_splits(seed: Sequence[int], profiles: Sequence[AgronNodeProfile]) -> list[list[int]]:
    """Small local search around a weighted seed when exhaustive search is too large."""

    seen = {tuple(int(x) for x in seed)}
    queue = [list(int(x) for x in seed)]
    for _ in range(max(1, len(seed) * 4)):
        base = queue.pop(0)
        for src in range(len(base)):
            if base[src] <= 1:
                continue
            for dst in range(len(base)):
                if src == dst:
                    continue
                limit = profiles[dst].max_layers
                if limit is not None and base[dst] >= int(limit):
                    continue
                candidate = list(base)
                candidate[src] -= 1
                candidate[dst] += 1
                key = tuple(candidate)
                if key in seen:
                    continue
                seen.add(key)
                queue.append(candidate)
        if not queue:
            break
    return [list(item) for item in seen]


def _score_agron_counts(
    endpoints: Sequence[MeshEndpoint],
    counts: Sequence[int],
    profiles: Sequence[AgronNodeProfile],
    links: dict[tuple[str, str], AgronLinkProfile],
    *,
    hidden_bytes: int,
    default_latency_ms: float,
    default_bandwidth_mbps: float,
    objective: str,
) -> dict[str, Any]:
    stage_compute_ms = [
        (float(profile.fixed_ms) + (float(count) / max(float(profile.speed), 1e-6)))
        for count, profile in zip(counts, profiles)
    ]
    link_ms = [
        _link_cost_ms(
            endpoints[idx],
            endpoints[idx + 1],
            links,
            hidden_bytes=hidden_bytes,
            default_latency_ms=default_latency_ms,
            default_bandwidth_mbps=default_bandwidth_mbps,
        )
        for idx in range(max(0, len(endpoints) - 1))
    ]
    serial_ms = float(sum(stage_compute_ms) + sum(link_ms))
    stage_step_ms = []
    for idx, compute in enumerate(stage_compute_ms):
        outbound = link_ms[idx] if idx < len(link_ms) else 0.0
        stage_step_ms.append(float(compute + outbound))
    pipeline_step_ms = max(stage_step_ms) if stage_step_ms else 0.0
    imbalance_ms = max(stage_compute_ms) - min(stage_compute_ms) if stage_compute_ms else 0.0
    if objective == "latency":
        score = serial_ms
    elif objective == "throughput":
        score = pipeline_step_ms
    else:
        score = pipeline_step_ms + 0.5 * imbalance_ms + 0.1 * serial_ms
    return {
        "score": round(float(score), 6),
        "serial_step_ms": round(serial_ms, 6),
        "pipeline_step_ms": round(float(pipeline_step_ms), 6),
        "imbalance_ms": round(float(imbalance_ms), 6),
        "stage_compute_ms": [round(float(value), 6) for value in stage_compute_ms],
        "link_ms": [round(float(value), 6) for value in link_ms],
    }


def _coerce_num_layers(value: Any) -> int:
    if isinstance(value, int):
        num_layers = value
    elif isinstance(value, dict):
        num_layers = int(
            value.get("num_hidden_layers")
            or value.get("n_layers")
            or value.get("num_layers")
            or 0
        )
    else:
        num_layers = int(
            getattr(value, "num_hidden_layers", None)
            or getattr(value, "n_layers", None)
            or getattr(value, "num_layers", 0)
        )
    if num_layers <= 0:
        raise ValueError("num_layers must be a positive integer or config-like object")
    return num_layers


def _coerce_endpoints(
    workers: str | Iterable[str] | Sequence[MeshEndpoint],
) -> list[MeshEndpoint]:
    if isinstance(workers, Sequence) and workers and isinstance(workers[0], MeshEndpoint):
        return list(workers)  # type: ignore[arg-type]
    return parse_worker_specs(workers)  # type: ignore[arg-type]


def _weighted_counts(total: int, weights: Sequence[float]) -> list[int]:
    """Allocate ``total`` items by largest remainder while preserving coverage."""

    if total <= 0:
        raise ValueError("total must be positive")
    if not weights:
        raise ValueError("at least one weight is required")
    if len(weights) > total:
        weights = weights[:total]

    safe_weights = [max(float(weight), 1e-6) for weight in weights]
    count = len(safe_weights)
    base = [1] * count
    remaining = total - count
    if remaining == 0:
        return base

    weight_sum = sum(safe_weights)
    raw = [remaining * weight / weight_sum for weight in safe_weights]
    floors = [int(math.floor(item)) for item in raw]
    allocated = sum(floors)
    extra = remaining - allocated

    order = sorted(
        range(count),
        key=lambda idx: (raw[idx] - floors[idx], safe_weights[idx]),
        reverse=True,
    )
    counts = [base[idx] + floors[idx] for idx in range(count)]
    for idx in order[:extra]:
        counts[idx] += 1
    return counts


def plan_layer_stages(
    num_layers: int | Any,
    workers: str | Iterable[str] | Sequence[MeshEndpoint],
    *,
    devices: Sequence[str] | None = None,
    as_dict: bool = True,
) -> list[dict[str, Any]] | list[LayerStage]:
    """
    Build a contiguous layer-stage plan for future MegaMesh layer sharding.

    Replica mode does not use this yet. The function exists so the experimental
    layer-shard path has one stable, testable planning surface without touching
    the model decode implementation.
    """

    layer_count = _coerce_num_layers(num_layers)
    endpoints = _coerce_endpoints(workers)
    active_endpoints = endpoints[: min(len(endpoints), layer_count)]
    counts = _weighted_counts(layer_count, [endpoint.weight for endpoint in active_endpoints])
    device_values = list(devices or [])

    stages: list[LayerStage] = []
    cursor = 0
    for stage_id, (endpoint, count) in enumerate(zip(active_endpoints, counts)):
        layer_start = cursor
        layer_end = cursor + count
        cursor = layer_end
        stages.append(
            LayerStage(
                stage_id=stage_id,
                layer_start=layer_start,
                layer_end=layer_end,
                worker_url=endpoint.base_url,
                name=endpoint.name,
                weight=float(endpoint.weight),
                device=device_values[stage_id] if stage_id < len(device_values) else "",
            )
        )

    if cursor != layer_count:
        raise RuntimeError("internal MegaMesh planner error: layer coverage mismatch")
    if as_dict:
        return [stage.to_dict() for stage in stages]
    return stages


def plan_agron_layer_stages(
    num_layers: int | Any,
    workers: str | Iterable[str] | Sequence[MeshEndpoint],
    *,
    node_profiles: Sequence[dict[str, Any] | AgronNodeProfile] | None = None,
    link_profiles: Sequence[dict[str, Any] | AgronLinkProfile] | None = None,
    hidden_bytes: int = 0,
    objective: str = "balanced",
    allow_reorder: bool = False,
    default_latency_ms: float = 0.0,
    default_bandwidth_mbps: float = 0.0,
    max_candidates: int = 20_000,
) -> dict[str, Any]:
    """Plan layer shards with AGron, the MegaMesh distributed mapping heuristic.

    Unlike ``plan_layer_stages``, AGron can consider directed TTP link costs and
    can optionally reorder the mesh path. It remains pure planning code: it does
    not start workers, load weights, allocate KV cache, or touch the engine.
    """

    objective = str(objective).strip().lower()
    if objective not in {"balanced", "latency", "throughput"}:
        raise ValueError("objective must be one of: balanced, latency, throughput")

    layer_count = _coerce_num_layers(num_layers)
    endpoints = _coerce_endpoints(workers)
    active_endpoints = endpoints[: min(len(endpoints), layer_count)]
    if not active_endpoints:
        raise ValueError("AGron requires at least one active endpoint")

    node_map: dict[str, AgronNodeProfile] = {}
    for raw in node_profiles or []:
        if isinstance(raw, AgronNodeProfile):
            profile = raw
        else:
            profile = AgronNodeProfile(
                name=str(raw.get("name") or raw.get("endpoint") or raw.get("url") or ""),
                speed=float(raw.get("speed", raw.get("layers_per_ms", raw.get("weight", 1.0)))),
                max_layers=(
                    int(raw["max_layers"])
                    if raw.get("max_layers") is not None
                    else None
                ),
                fixed_ms=float(raw.get("fixed_ms", 0.0)),
            )
        if not profile.name:
            raise ValueError("AGron node profiles require a name, endpoint, or url")
        node_map[profile.name] = profile

    link_map: dict[tuple[str, str], AgronLinkProfile] = {}
    for raw in link_profiles or []:
        if isinstance(raw, AgronLinkProfile):
            profile = raw
        else:
            profile = AgronLinkProfile(
                src=str(raw.get("src") or raw.get("source") or ""),
                dst=str(raw.get("dst") or raw.get("target") or ""),
                latency_ms=float(
                    raw.get(
                        "latency_ms",
                        raw.get("rtt_ms_min", raw.get("rtt_ms_avg", 0.0)),
                    )
                ),
                bandwidth_mbps=float(
                    raw.get(
                        "bandwidth_mbps",
                        raw.get("one_way_payload_mbps", raw.get("mbps", 0.0)),
                    )
                ),
            )
        if not profile.src or not profile.dst:
            raise ValueError("AGron link profiles require src and dst")
        link_map[(profile.src, profile.dst)] = profile

    orders: Iterable[tuple[MeshEndpoint, ...]]
    if allow_reorder and len(active_endpoints) > 1:
        orders = itertools.permutations(active_endpoints)
    else:
        orders = [tuple(active_endpoints)]

    best: tuple[float, tuple[MeshEndpoint, ...], list[int], dict[str, Any], bool] | None = None
    evaluated = 0
    search_exhaustive = True

    for order in orders:
        profiles = [_node_profile_for(endpoint, node_map) for endpoint in order]
        limits = [profile.max_layers for profile in profiles]
        try:
            splits, exhaustive = _bounded_compositions(
                layer_count,
                limits,
                max_candidates=max(1, int(max_candidates)),
            )
        except ValueError:
            continue
        search_exhaustive = search_exhaustive and exhaustive
        if not exhaustive:
            seed = _weighted_seed_counts(layer_count, profiles)
            splits = _neighbor_splits(seed, profiles)
        for counts in splits:
            evaluated += 1
            score = _score_agron_counts(
                order,
                counts,
                profiles,
                link_map,
                hidden_bytes=int(hidden_bytes),
                default_latency_ms=float(default_latency_ms),
                default_bandwidth_mbps=float(default_bandwidth_mbps),
                objective=objective,
            )
            candidate = (float(score["score"]), order, counts, score, exhaustive)
            if best is None or candidate[0] < best[0]:
                best = candidate

    if best is None:
        raise ValueError("AGron could not build a valid layer plan with the given profiles")

    _, order, counts, score, exhaustive = best
    profiles = [_node_profile_for(endpoint, node_map) for endpoint in order]
    stages: list[dict[str, Any]] = []
    cursor = 0
    for stage_id, (endpoint, count, profile) in enumerate(zip(order, counts, profiles)):
        layer_start = cursor
        layer_end = cursor + int(count)
        cursor = layer_end
        stage = LayerStage(
            stage_id=stage_id,
            layer_start=layer_start,
            layer_end=layer_end,
            worker_url=_endpoint_url(endpoint),
            name=_endpoint_label(endpoint),
            weight=float(endpoint.weight),
            device="",
        ).to_dict()
        stage["agron"] = {
            "speed": float(profile.speed),
            "fixed_ms": float(profile.fixed_ms),
            "max_layers": profile.max_layers,
            "estimated_compute_ms": score["stage_compute_ms"][stage_id],
        }
        if stage_id < len(score["link_ms"]):
            stage["agron"]["outbound_link_ms"] = score["link_ms"][stage_id]
        stages.append(stage)

    return {
        "planner": "agron-v0",
        "num_layers": layer_count,
        "objective": objective,
        "hidden_bytes": int(hidden_bytes),
        "allow_reorder": bool(allow_reorder),
        "evaluated_candidates": int(evaluated),
        "search_exhaustive": bool(search_exhaustive and exhaustive),
        "score": score,
        "stages": stages,
        "stages_arg": ",".join(stage["worker_url"] + f"#{stage['name']}" for stage in stages),
    }


__all__ = [
    "AgronLinkProfile",
    "AgronNodeProfile",
    "LayerStage",
    "plan_agron_layer_stages",
    "plan_layer_stages",
]

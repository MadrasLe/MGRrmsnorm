"""Small HTTP/JSON protocol for MegaMesh replica workers.

The first MegaMesh transport intentionally uses the Python standard library.
Replica mode only sends prompts and generated text over the network, so plain
HTTP is enough for early multi-pod experiments and keeps Colab/RunPod setup
simple.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable, List, Optional
from urllib import error as urlerror
from urllib import request as urlrequest


class MeshHTTPError(RuntimeError):
    """Raised when a MegaMesh worker returns an error response."""


@dataclass(frozen=True)
class MeshEndpoint:
    url: str
    weight: float = 1.0
    name: str = ""

    @property
    def base_url(self) -> str:
        url = self.url.strip()
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        return url.rstrip("/")

    @property
    def label(self) -> str:
        return self.name or self.base_url


def _parse_one_worker_spec(spec: str) -> MeshEndpoint:
    raw = spec.strip()
    if not raw:
        raise ValueError("empty worker spec")

    name = ""
    if "#" in raw:
        raw, name = raw.rsplit("#", 1)
        name = name.strip()

    weight = 1.0
    scheme_pos = raw.find("://")
    at_pos = raw.rfind("@")
    if at_pos > max(scheme_pos + 2, 0):
        maybe_weight = raw[at_pos + 1 :].strip()
        try:
            weight = float(maybe_weight)
            raw = raw[:at_pos]
        except ValueError:
            pass
    if weight <= 0:
        raise ValueError(f"worker weight must be > 0: {spec!r}")
    return MeshEndpoint(url=raw.strip(), weight=weight, name=name)


def parse_worker_specs(specs: str | Iterable[str]) -> List[MeshEndpoint]:
    """Parse worker specs like ``host:8088@2#l4`` or comma-separated lists."""

    if isinstance(specs, str):
        parts = [part for part in specs.split(",") if part.strip()]
    else:
        parts = []
        for spec in specs:
            parts.extend(part for part in str(spec).split(",") if part.strip())
    endpoints = [_parse_one_worker_spec(part) for part in parts]
    if not endpoints:
        raise ValueError("at least one MegaMesh worker endpoint is required")
    return endpoints


def request_json(
    endpoint: MeshEndpoint,
    path: str,
    payload: Optional[dict] = None,
    timeout: float = 120.0,
) -> dict:
    """Send one JSON request to a worker and return the decoded JSON object."""

    path = "/" + path.lstrip("/")
    data = None
    method = "GET"
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"

    req = urlrequest.Request(
        endpoint.base_url + path,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urlerror.HTTPError as exc:
        body = exc.read()
        try:
            parsed = json.loads(body.decode("utf-8"))
            detail = parsed.get("error") or parsed.get("message") or parsed
        except Exception:
            detail = body.decode("utf-8", errors="replace")
        raise MeshHTTPError(f"{endpoint.label} HTTP {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:
        raise MeshHTTPError(f"{endpoint.label}: {exc}") from exc

    try:
        parsed = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise MeshHTTPError(f"{endpoint.label}: invalid JSON response") from exc
    if isinstance(parsed, dict) and parsed.get("ok") is False:
        raise MeshHTTPError(f"{endpoint.label}: {parsed.get('error', parsed)}")
    if not isinstance(parsed, dict):
        raise MeshHTTPError(f"{endpoint.label}: expected JSON object response")
    return parsed


def request_binary(
    endpoint: MeshEndpoint,
    path: str,
    payload: bytes,
    timeout: float = 120.0,
    *,
    content_type: str = "application/octet-stream",
) -> bytes:
    """Send one binary request to a worker and return the raw response body."""

    path = "/" + path.lstrip("/")
    req = urlrequest.Request(
        endpoint.base_url + path,
        data=payload,
        headers={
            "Accept": content_type,
            "Content-Type": content_type,
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urlerror.HTTPError as exc:
        body = exc.read()
        try:
            parsed = json.loads(body.decode("utf-8"))
            detail = parsed.get("error") or parsed.get("message") or parsed
        except Exception:
            detail = body.decode("utf-8", errors="replace")
        raise MeshHTTPError(f"{endpoint.label} HTTP {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:
        raise MeshHTTPError(f"{endpoint.label}: {exc}") from exc

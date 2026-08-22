"""MegaMesh replica router."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .protocol import MeshEndpoint, MeshHTTPError, parse_worker_specs, request_json


class MeshRouter:
    """Weighted prompt router for full-model MegaMesh workers."""

    def __init__(
        self,
        workers: str | Iterable[str] | Sequence[MeshEndpoint],
        *,
        timeout: float = 600.0,
    ) -> None:
        if isinstance(workers, Sequence) and workers and isinstance(workers[0], MeshEndpoint):
            self.endpoints = list(workers)  # type: ignore[arg-type]
        else:
            self.endpoints = parse_worker_specs(workers)  # type: ignore[arg-type]
        self.timeout = float(timeout)

    def health(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(self.endpoints)) as pool:
            futures = {
                pool.submit(request_json, endpoint, "/health", None, self.timeout): endpoint
                for endpoint in self.endpoints
            }
            for future in as_completed(futures):
                endpoint = futures[future]
                try:
                    row = future.result()
                    row.setdefault("endpoint", endpoint.base_url)
                    row.setdefault("configured_weight", endpoint.weight)
                    rows.append(row)
                except Exception as exc:
                    rows.append(
                        {
                            "ok": False,
                            "endpoint": endpoint.base_url,
                            "name": endpoint.name,
                            "configured_weight": endpoint.weight,
                            "error": str(exc),
                        }
                    )
        rows.sort(key=lambda item: str(item.get("endpoint", "")))
        return rows

    def _assign_indexed(
        self,
        indexed_prompts: Sequence[Tuple[int, str]],
        endpoints: Optional[Sequence[MeshEndpoint]] = None,
    ) -> List[Tuple[MeshEndpoint, List[Tuple[int, str]]]]:
        endpoints = list(endpoints or self.endpoints)
        if not endpoints:
            raise ValueError("at least one MegaMesh worker endpoint is required")
        buckets: List[List[Tuple[int, str]]] = [[] for _ in endpoints]
        weights = [max(float(endpoint.weight), 1e-6) for endpoint in endpoints]
        for idx, prompt in indexed_prompts:
            best = min(range(len(buckets)), key=lambda i: len(buckets[i]) / weights[i])
            buckets[best].append((idx, prompt))
        return [
            (endpoint, bucket)
            for endpoint, bucket in zip(endpoints, buckets)
            if bucket
        ]

    def _assign(self, prompts: Sequence[str]) -> List[Tuple[MeshEndpoint, List[Tuple[int, str]]]]:
        return self._assign_indexed(list(enumerate(prompts)), self.endpoints)

    def generate_batch_with_stats(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_token_ids: Optional[List[int]] = None,
        verbose: bool = False,
        failover: bool = True,
    ) -> Tuple[List[str], Dict[str, Any]]:
        if not prompts:
            return [], {"elapsed_ms": 0.0, "generated_tokens": 0, "workers": []}

        outputs: List[Optional[str]] = [None] * len(prompts)
        assignments = self._assign(prompts)
        payload_base = {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "stop_token_ids": stop_token_ids,
            "verbose": bool(verbose),
        }

        t0 = time.perf_counter()
        worker_rows: List[Dict[str, Any]] = []
        failure_rows: List[Dict[str, Any]] = []
        unavailable: set[MeshEndpoint] = set()
        pending = assignments

        while pending:
            failed_items: List[Tuple[int, str]] = []
            with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                futures = {}
                for endpoint, bucket in pending:
                    payload = dict(payload_base)
                    payload["prompts"] = [prompt for _, prompt in bucket]
                    future = pool.submit(
                        request_json,
                        endpoint,
                        "/generate",
                        payload,
                        self.timeout,
                    )
                    futures[future] = (endpoint, bucket)

                for future in as_completed(futures):
                    endpoint, bucket = futures[future]
                    try:
                        result = future.result()
                        worker_outputs = result.get("outputs", [])
                        if len(worker_outputs) != len(bucket):
                            raise RuntimeError(
                                f"{endpoint.label} returned {len(worker_outputs)} outputs "
                                f"for {len(bucket)} prompts"
                            )
                    except Exception as exc:
                        failure_rows.append(
                            {
                                "endpoint": endpoint.base_url,
                                "name": endpoint.name or endpoint.label,
                                "configured_weight": endpoint.weight,
                                "num_prompts": len(bucket),
                                "error": str(exc),
                            }
                        )
                        failed_items.extend(bucket)
                        unavailable.add(endpoint)
                        continue

                    for (original_idx, _), text in zip(bucket, worker_outputs):
                        outputs[original_idx] = text
                    worker_rows.append(
                        {
                            "endpoint": endpoint.base_url,
                            "name": result.get("worker") or endpoint.label,
                            "configured_weight": endpoint.weight,
                            "num_prompts": len(bucket),
                            "elapsed_ms": result.get("elapsed_ms"),
                            "generated_tokens": result.get("generated_tokens"),
                            "tokens_per_second": result.get("tokens_per_second"),
                        }
                    )

            if not failed_items:
                pending = []
                continue
            if not failover:
                break

            fallback_endpoints = [
                endpoint for endpoint in self.endpoints if endpoint not in unavailable
            ]
            if not fallback_endpoints:
                break
            pending = self._assign_indexed(failed_items, fallback_endpoints)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if any(text is None for text in outputs):
            detail = "; ".join(
                f"{row.get('name')}: {row.get('error')}" for row in failure_rows[-3:]
            )
            raise MeshHTTPError(
                "MegaMesh router did not receive every output"
                + (f" ({detail})" if detail else "")
            )
        generated_tokens = sum(
            int(row.get("generated_tokens") or 0)
            for row in worker_rows
        )
        stats = {
            "elapsed_ms": round(elapsed_ms, 3),
            "generated_tokens": generated_tokens,
            "tokens_per_second": (
                round(generated_tokens / max(elapsed_ms / 1000.0, 1e-9), 3)
                if elapsed_ms > 0
                else 0.0
            ),
            "workers": sorted(worker_rows, key=lambda item: str(item.get("endpoint", ""))),
            "failures": failure_rows,
        }
        return [text for text in outputs if text is not None], stats

    def generate_batch(self, prompts: Sequence[str], **kwargs: Any) -> List[str]:
        outputs, _ = self.generate_batch_with_stats(prompts, **kwargs)
        return outputs

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self.generate_batch([prompt], **kwargs)[0]

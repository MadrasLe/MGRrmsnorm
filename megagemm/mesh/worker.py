"""MegaMesh replica worker server."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

import torch


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


class MegaMeshWorker:
    """One full-model MegaGemm worker used by replica/router mode."""

    def __init__(
        self,
        model: str,
        *,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        num_blocks: int = 0,
        max_batch_size: int = 512,
        max_seq_len: int = 4096,
        quantize: Optional[str] = None,
        cache_dir: Optional[str] = None,
        name: str = "",
        weight: float = 1.0,
    ) -> None:
        from megagemm.engine import InferenceEngine

        self.name = name or os.environ.get("MEGAMESH_WORKER_NAME") or "worker"
        self.weight = float(weight)
        self.model_name = model
        self.device = device
        self.dtype = dtype
        self.lock = threading.Lock()
        self.request_count = 0
        self.generated_tokens = 0
        self.started_at = time.time()
        self.engine = InferenceEngine(
            model,
            dtype=dtype,
            device=device,
            num_blocks=num_blocks,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            quantize=quantize,
            cache_dir=cache_dir,
        )

    def health(self) -> Dict[str, Any]:
        gpu = None
        if self.device == "cuda" and torch.cuda.is_available():
            idx = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(idx)
            gpu = {
                "name": torch.cuda.get_device_name(idx),
                "index": int(idx),
                "free_mb": round(free / 1024**2, 2),
                "total_mb": round(total / 1024**2, 2),
                "allocated_mb": round(torch.cuda.memory_allocated(idx) / 1024**2, 2),
                "reserved_mb": round(torch.cuda.memory_reserved(idx) / 1024**2, 2),
            }
        return {
            "ok": True,
            "name": self.name,
            "mode": "replica",
            "model": self.model_name,
            "device": self.device,
            "dtype": _dtype_name(self.dtype),
            "weight": self.weight,
            "request_count": self.request_count,
            "generated_tokens": self.generated_tokens,
            "uptime_s": round(time.time() - self.started_at, 2),
            "gpu": gpu,
        }

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompts = payload.get("prompts")
        if prompts is None:
            prompt = payload.get("prompt")
            if prompt is None:
                raise ValueError("payload must contain 'prompt' or 'prompts'")
            prompts = [prompt]
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("'prompts' must be a list of strings")
        if not prompts:
            return {
                "ok": True,
                "worker": self.name,
                "outputs": [],
                "elapsed_ms": 0.0,
                "generated_tokens": 0,
            }

        max_new_tokens = int(payload.get("max_new_tokens", payload.get("max_tokens", 128)))
        temperature = float(payload.get("temperature", 0.0))
        top_k = int(payload.get("top_k", 50))
        top_p = float(payload.get("top_p", 0.9))
        stop_token_ids = payload.get("stop_token_ids")
        verbose = bool(payload.get("verbose", False))

        with self.lock:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs: List[str] = self.engine.generate_batch(
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_token_ids=stop_token_ids,
                verbose=verbose,
            )
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        generated_tokens = sum(
            len(self.engine.tokenizer.encode(text, add_special_tokens=False))
            for text in outputs
        )
        self.request_count += 1
        self.generated_tokens += int(generated_tokens)
        return {
            "ok": True,
            "worker": self.name,
            "outputs": outputs,
            "elapsed_ms": round(elapsed_ms, 3),
            "generated_tokens": int(generated_tokens),
            "num_prompts": len(prompts),
            "tokens_per_second": (
                round(generated_tokens / max(elapsed_ms / 1000.0, 1e-9), 3)
                if elapsed_ms > 0
                else 0.0
            ),
        }


def _handler_for(worker: MegaMeshWorker):
    class MegaMeshHandler(BaseHTTPRequestHandler):
        server_version = "MegaMeshHTTP/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[MegaMesh:{worker.name}] {self.address_string()} - {fmt % args}")

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            if not raw:
                return {}
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("request body must be a JSON object")
            return parsed

        def do_GET(self) -> None:
            if self.path.rstrip("/") in ("", "/health", "/metrics"):
                self._send_json(200, worker.health())
            else:
                self._send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            try:
                if self.path.rstrip("/") != "/generate":
                    self._send_json(404, {"ok": False, "error": "not_found"})
                    return
                payload = self._read_json()
                self._send_json(200, worker.generate(payload))
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})

    return MegaMeshHandler


def run_worker(
    model: str,
    *,
    host: str = "0.0.0.0",
    port: int = 8088,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    num_blocks: int = 0,
    max_batch_size: int = 512,
    max_seq_len: int = 4096,
    quantize: Optional[str] = None,
    cache_dir: Optional[str] = None,
    name: str = "",
    weight: float = 1.0,
) -> None:
    worker = MegaMeshWorker(
        model,
        dtype=dtype,
        device=device,
        num_blocks=num_blocks,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        quantize=quantize,
        cache_dir=cache_dir,
        name=name,
        weight=weight,
    )
    server = ThreadingHTTPServer((host, int(port)), _handler_for(worker))
    print(
        f"[MegaMesh] {worker.name} ready on http://{host}:{port} "
        f"model={model} device={device} dtype={_dtype_name(dtype)} weight={weight}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MegaMesh] shutting down")
    finally:
        server.server_close()

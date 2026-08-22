"""TTP: persistent Tensor Transfer Protocol for MegaMesh shards.

TTP v0 is deliberately small: a long-lived TCP connection carries length-prefixed
MegaMesh binary tensor frames. It removes HTTP request/response overhead while
keeping the implementation portable for Kaggle/Colab/local machines.
"""

from __future__ import annotations

import os
import socket
import socketserver
import struct
import traceback
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import torch

from .binary_codec import (
    PinnedTensorPool,
    TensorFrameParts,
    decode_tensor_frame,
    encode_tensor_frame,
    encode_tensor_frame_parts,
)
from .protocol import MeshEndpoint


DEFAULT_MAX_FRAME_BYTES = 2 * 1024 * 1024 * 1024
_PACKET_PREFIX = struct.Struct("<Q")

if os.environ.get("MEGAGEMM_TTP_NATIVE", "1").lower() in {"0", "false", "no", "off"}:
    _ttp_native = None
    _HAS_TTP_NATIVE = False
else:
    try:
        import megagemm_ttp_native as _ttp_native

        _HAS_TTP_NATIVE = bool(getattr(_ttp_native, "SUPPORTS_TIMEOUT_SOCKETS", False))
        if not _HAS_TTP_NATIVE:
            _ttp_native = None
    except Exception:
        _ttp_native = None
        _HAS_TTP_NATIVE = False


class TTPError(RuntimeError):
    """Raised when a TTP request fails."""


def _read_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(nbytes)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("TTP connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_exact_bytearray(sock: socket.socket, nbytes: int) -> bytearray:
    buf = bytearray(int(nbytes))
    view = memoryview(buf)
    offset = 0
    while offset < nbytes:
        nread = sock.recv_into(view[offset:])
        if not nread:
            raise EOFError("TTP connection closed")
        offset += int(nread)
    return buf


def send_packet(sock: socket.socket, payload: bytes) -> None:
    """Send one length-prefixed TTP packet."""

    sock.sendall(_PACKET_PREFIX.pack(len(payload)) + payload)


def send_packet_parts(sock: socket.socket, frame: TensorFrameParts) -> None:
    """Send one length-prefixed TTP packet from reusable frame parts."""

    try:
        sock.sendall(_PACKET_PREFIX.pack(frame.nbytes))
        for part in frame.iter_parts():
            sock.sendall(part)
    finally:
        frame.release()


def recv_packet(
    sock: socket.socket,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> bytearray:
    """Receive one length-prefixed TTP packet."""

    if _ttp_native is not None:
        timeout = sock.gettimeout()
        native_timeout = -1.0 if timeout is None else float(timeout)
        return _ttp_native.recv_packet(sock.fileno(), int(max_frame_bytes), native_timeout)

    raw_len = _read_exact(sock, _PACKET_PREFIX.size)
    (length,) = _PACKET_PREFIX.unpack(raw_len)
    if length > max_frame_bytes:
        raise ValueError(f"TTP frame too large: {length} bytes")
    return _read_exact_bytearray(sock, int(length))


def _parse_ttp_address(endpoint: MeshEndpoint | str) -> tuple[str, int]:
    raw = endpoint.url if isinstance(endpoint, MeshEndpoint) else str(endpoint)
    raw = raw.strip()
    if raw.startswith(("ttp://", "tcp://", "http://", "https://")):
        parsed = urlparse(raw)
        if not parsed.hostname or parsed.port is None:
            raise ValueError(f"TTP endpoint requires host and port: {raw!r}")
        return parsed.hostname, int(parsed.port)

    if raw.count(":") != 1:
        raise ValueError(f"TTP endpoint must look like host:port or ttp://host:port: {raw!r}")
    host, port = raw.rsplit(":", 1)
    return host.strip(), int(port)


class TTPClient:
    """Persistent client for one TTP shard endpoint."""

    def __init__(self, endpoint: MeshEndpoint | str, *, timeout: float = 900.0) -> None:
        self.endpoint = endpoint
        self.host, self.port = _parse_ttp_address(endpoint)
        self.timeout = float(timeout)
        self._sock: socket.socket | None = None
        self._out_pool = PinnedTensorPool()

    def close(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _connect(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self._sock = sock
        return sock

    def request(
        self,
        op: str,
        meta: Mapping[str, Any] | None = None,
        tensors: Mapping[str, torch.Tensor] | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Send one TTP op and return decoded response metadata/tensors."""

        request_meta = {"op": str(op)}
        request_meta.update(dict(meta or {}))
        try:
            return self._request_frame(request_meta, tensors, device=device)
        except (OSError, EOFError):
            self.close()
            return self._request_frame(request_meta, tensors, device=device)

    def _request_frame(
        self,
        meta: Mapping[str, Any],
        tensors: Mapping[str, torch.Tensor] | None,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        sock = self._connect()
        frame = encode_tensor_frame_parts(meta, tensors, pool=self._out_pool)
        send_packet_parts(sock, frame)
        response = recv_packet(sock)
        meta, tensors = decode_tensor_frame(response, device=device, copy_tensors=False)
        if meta.get("ok") is False:
            raise TTPError(str(meta.get("error", meta)))
        return meta, tensors


class TTPClientPool:
    """Small persistent-connection pool keyed by shard stage."""

    def __init__(self, stages: Sequence[MeshEndpoint], *, timeout: float = 900.0) -> None:
        self.clients = {stage: TTPClient(stage, timeout=timeout) for stage in stages}

    def request(
        self,
        stage: MeshEndpoint,
        op: str,
        meta: Mapping[str, Any] | None = None,
        tensors: Mapping[str, torch.Tensor] | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        return self.clients[stage].request(op, meta, tensors, device=device)

    def close(self) -> None:
        for client in self.clients.values():
            client.close()


def ttp_runtime_info() -> dict[str, Any]:
    return {
        "native_recv": bool(_HAS_TTP_NATIVE),
        "native_version": getattr(_ttp_native, "TTP_NATIVE_VERSION", None)
        if _ttp_native is not None
        else None,
    }


class _TTPThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class TTPShardServer(_TTPThreadingServer):
    """Threaded TTP server bound to one MegaMeshShardWorker."""

    def __init__(self, server_address: tuple[str, int], worker: Any) -> None:
        self.worker = worker
        super().__init__(server_address, _TTPShardHandler)


class _TTPShardHandler(socketserver.BaseRequestHandler):
    def setup(self) -> None:
        try:
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def handle(self) -> None:
        worker = self.server.worker  # type: ignore[attr-defined]
        while True:
            try:
                payload = recv_packet(self.request)
            except EOFError:
                return
            except Exception as exc:
                try:
                    send_packet(
                        self.request,
                        encode_tensor_frame({"ok": False, "error": str(exc)}),
                    )
                except OSError:
                    return
                continue

            try:
                meta, tensors = decode_tensor_frame(payload, device="cpu", copy_tensors=False)
                op = str(meta.pop("op", ""))
                seq_id = meta.get("seq_id", None)
                if op == "prefill":
                    response = worker.prefill_ttp(meta, tensors)
                elif op == "prefill_chain":
                    response = worker.prefill_chain_ttp(meta, tensors)
                elif op == "decode":
                    response = worker.decode_ttp(meta, tensors)
                elif op == "decode_chain":
                    response = worker.decode_chain_ttp(meta, tensors)
                elif op == "decode_batch":
                    response = worker.decode_batch_ttp(meta, tensors)
                elif op == "decode_batch_chain":
                    response = worker.decode_batch_chain_ttp(meta, tensors)
                elif op == "generate_chain":
                    response = worker.generate_chain_ttp(meta, tensors)
                elif op == "generate_batch_chain":
                    response = worker.generate_batch_chain_ttp(meta, tensors)
                elif op == "generate_continuous_chain":
                    response = worker.generate_continuous_chain_ttp(meta, tensors)
                elif op == "lm_head_argmax":
                    response = worker.lm_head_argmax_ttp(meta, tensors)
                elif op == "mlp_forward":
                    response = worker.mlp_forward_ttp(meta, tensors)
                elif op == "probe_peer":
                    response = encode_tensor_frame(worker.probe_peer_ttp(meta, tensors))
                elif op == "ping":
                    response = encode_tensor_frame(
                        {
                            "ok": True,
                            "worker": getattr(worker, "name", ""),
                            "received_bytes": int(
                                sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
                            ),
                        }
                    )
                elif op == "free":
                    response = encode_tensor_frame(worker.free(meta))
                elif op == "health":
                    response = encode_tensor_frame(worker.health())
                else:
                    response = encode_tensor_frame(
                        {"ok": False, "error": f"unknown TTP op: {op!r}"}
                    )
            except Exception as exc:
                where = f"{getattr(worker, 'name', 'worker')} op={op!r} seq_id={seq_id!r}"
                tb = traceback.format_exc()
                print(f"[MegaMeshTTP:{where}] ERROR: {exc}\n{tb}", flush=True)
                response = encode_tensor_frame(
                    {
                        "ok": False,
                        "worker": getattr(worker, "name", ""),
                        "op": op,
                        "seq_id": seq_id,
                        "error": f"{where}: {exc}",
                    }
                )

            try:
                if isinstance(response, TensorFrameParts):
                    send_packet_parts(self.request, response)
                else:
                    send_packet(self.request, response)
            except OSError:
                return


__all__ = [
    "TTPClient",
    "TTPClientPool",
    "TTPError",
    "TTPShardServer",
    "recv_packet",
    "send_packet",
    "send_packet_parts",
    "ttp_runtime_info",
]

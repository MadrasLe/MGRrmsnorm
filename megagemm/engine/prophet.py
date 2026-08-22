"""
MGX Prophet - lightweight state-library and retrieval layer for MegaGemm agents.

Prophet manages a library of previously captured runtime snapshots and predicts
the best reusable state for a new prompt based on compatibility, exact/prefix
hashes, semantic embedding similarity, and token-prefix replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import torch

__all__ = ["MGXProphetLibrary"]


_SNAPSHOT_CACHE: "OrderedDict[tuple[str, int, int], dict[str, Any]]" = OrderedDict()
_SNAPSHOT_CACHE_HITS = 0
_SNAPSHOT_CACHE_MISSES = 0
_SNAPSHOT_CACHE_BYTES = 0
_RESIDENT_CACHE: "OrderedDict[tuple[int, str], dict[str, Any]]" = OrderedDict()
_RESIDENT_SEQ_COUNTER = 0


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _resident_cache_limit() -> int:
    return max(0, _env_int("MEGAGEMM_PROPHET_RESIDENT_CACHE_MAX_ENTRIES", 16))


def _resident_cache_key(engine, entry_id: str) -> tuple[int, str]:
    return (id(engine), str(entry_id))


def _new_resident_seq_id(engine) -> int:
    global _RESIDENT_SEQ_COUNTER
    while True:
        _RESIDENT_SEQ_COUNTER -= 1
        seq_id = -1_000_000_000 + _RESIDENT_SEQ_COUNTER
        if seq_id not in getattr(engine.block_manager, "block_tables", {}):
            return seq_id


def _snapshot_cache_enabled() -> bool:
    return _env_bool("MEGAGEMM_PROPHET_SNAPSHOT_CACHE", default=True)


def _snapshot_cache_limit() -> int:
    return max(0, _env_int("MEGAGEMM_PROPHET_SNAPSHOT_CACHE_MAX_ENTRIES", 32))


def _snapshot_gpu_cache_enabled() -> bool:
    return _env_bool("MEGAGEMM_PROPHET_GPU_SNAPSHOT_CACHE", default=False)


def _snapshot_gpu_cache_max_bytes() -> int:
    max_mb = max(0, _env_int("MEGAGEMM_PROPHET_GPU_SNAPSHOT_CACHE_MAX_MB", 2048))
    return max_mb * 1024 * 1024


def _snapshot_gpu_cache_device() -> Optional[torch.device]:
    if not _snapshot_gpu_cache_enabled():
        return None
    if not torch.cuda.is_available():
        return None
    raw = os.environ.get("MEGAGEMM_PROPHET_GPU_SNAPSHOT_CACHE_DEVICE", "cuda").strip() or "cuda"
    try:
        return torch.device(raw)
    except Exception:
        return torch.device("cuda")


def _snapshot_cache_key(path: str | Path) -> tuple[str, int, int]:
    snapshot_path = Path(path).expanduser()
    stat = snapshot_path.stat()
    return (str(snapshot_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _snapshot_cached_nbytes(snapshot: Optional[dict[str, Any]]) -> int:
    if not snapshot:
        return 0
    try:
        return int(snapshot.get("_snapshot_cache_nbytes", 0) or 0)
    except Exception:
        return 0


def _prepare_snapshot_for_cache(snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    Prepare a snapshot for the in-process cache.

    The on-disk snapshot stays CPU/torch.save friendly.  When explicitly enabled,
    this cache keeps a CUDA copy of the tensors so Prophet prefix hits avoid the
    slow CPU->GPU restore path and behave closer to an in-memory prefix cache.
    """
    target_device = _snapshot_gpu_cache_device()
    if target_device is None:
        cached = dict(snapshot)
        cached["_snapshot_cache_nbytes"] = 0
        cached["_snapshot_cache_device"] = None
        return cached

    tensor_memo: dict[int, torch.Tensor] = {}
    tensor_bytes = 0

    def promote(value: Any) -> Any:
        nonlocal tensor_bytes
        if torch.is_tensor(value):
            tensor_id = id(value)
            cached_tensor = tensor_memo.get(tensor_id)
            if cached_tensor is not None:
                return cached_tensor
            with torch.inference_mode(False):
                promoted = value.detach().to(target_device, non_blocking=True)
                if not promoted.is_contiguous():
                    promoted = promoted.contiguous()
            tensor_memo[tensor_id] = promoted
            tensor_bytes += _tensor_nbytes(promoted)
            return promoted
        if isinstance(value, dict):
            return {key: promote(item) for key, item in value.items()}
        if isinstance(value, list):
            return [promote(item) for item in value]
        if isinstance(value, tuple):
            return tuple(promote(item) for item in value)
        return value

    cached = promote(snapshot)
    cached["_snapshot_cache_nbytes"] = int(tensor_bytes)
    cached["_snapshot_cache_device"] = str(target_device)
    return cached


def _evict_snapshot_cache_if_needed() -> None:
    global _SNAPSHOT_CACHE_BYTES
    limit = _snapshot_cache_limit()
    max_bytes = _snapshot_gpu_cache_max_bytes() if _snapshot_gpu_cache_enabled() else 0
    while _SNAPSHOT_CACHE and (
        (limit > 0 and len(_SNAPSHOT_CACHE) > limit)
        or (max_bytes > 0 and _SNAPSHOT_CACHE_BYTES > max_bytes)
    ):
        _, evicted = _SNAPSHOT_CACHE.popitem(last=False)
        _SNAPSHOT_CACHE_BYTES -= _snapshot_cached_nbytes(evicted)
    _SNAPSHOT_CACHE_BYTES = max(0, int(_SNAPSHOT_CACHE_BYTES))


def _remember_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    global _SNAPSHOT_CACHE_BYTES
    if not _snapshot_cache_enabled():
        return
    limit = _snapshot_cache_limit()
    if limit <= 0:
        return
    key = _snapshot_cache_key(path)
    old_snapshot = _SNAPSHOT_CACHE.pop(key, None)
    if old_snapshot is not None:
        _SNAPSHOT_CACHE_BYTES -= _snapshot_cached_nbytes(old_snapshot)
    cached_snapshot = _prepare_snapshot_for_cache(snapshot)
    _SNAPSHOT_CACHE[key] = cached_snapshot
    _SNAPSHOT_CACHE_BYTES += _snapshot_cached_nbytes(cached_snapshot)
    _SNAPSHOT_CACHE.move_to_end(key)
    _evict_snapshot_cache_if_needed()


def _load_snapshot(path: str | Path) -> dict[str, Any]:
    global _SNAPSHOT_CACHE_BYTES, _SNAPSHOT_CACHE_HITS, _SNAPSHOT_CACHE_MISSES

    if not _snapshot_cache_enabled():
        _SNAPSHOT_CACHE_MISSES += 1
        return torch.load(path, weights_only=False)

    try:
        key = _snapshot_cache_key(path)
    except FileNotFoundError:
        raise

    cached = _SNAPSHOT_CACHE.get(key)
    if cached is not None:
        _SNAPSHOT_CACHE_HITS += 1
        _SNAPSHOT_CACHE.move_to_end(key)
        return cached

    _SNAPSHOT_CACHE_MISSES += 1
    snapshot = torch.load(path, weights_only=False)
    limit = _snapshot_cache_limit()
    if limit > 0:
        cached_snapshot = _prepare_snapshot_for_cache(snapshot)
        _SNAPSHOT_CACHE[key] = cached_snapshot
        _SNAPSHOT_CACHE_BYTES += _snapshot_cached_nbytes(cached_snapshot)
        _SNAPSHOT_CACHE.move_to_end(key)
        _evict_snapshot_cache_if_needed()
        cached = _SNAPSHOT_CACHE.get(key)
        if cached is not None:
            return cached
    return snapshot


def _sha256_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    raise TypeError(f"Unsupported JSON value type for MGX Prophet metadata: {type(value).__name__}")


def _now_iso8601() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _common_prefix_length(lhs: list[int], rhs: list[int]) -> int:
    count = 0
    for left, right in zip(lhs, rhs):
        if int(left) != int(right):
            break
        count += 1
    return count


def _min_prefix_tokens_for_reuse(query_token_count: int, prefix_tokens: int) -> int:
    query_token_count = max(1, int(query_token_count))
    prefix_tokens = max(1, int(prefix_tokens))
    return max(1, min(prefix_tokens, max(1, query_token_count // 2)))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _normalized_cosine(score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    return _clamp((float(score) + 1.0) / 2.0, 0.0, 1.0)


def _canonical_dtype_name(value: Any) -> Optional[str]:
    """Normalize equivalent runtime/manifest dtype spellings for matching."""
    if value is None:
        return None
    name = str(value).strip().lower().removeprefix("torch.")
    aliases = {
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
        "bf16": "bfloat16",
    }
    return aliases.get(name, name)


def _build_prefix_reuse_policy(
    match: dict[str, Any],
    *,
    query_token_count: int,
    prefix_tokens: int,
    min_prefix_reuse_score: float,
    min_prefix_coverage: float,
    max_prefix_rollback_ratio: float,
    max_prefix_tail_ratio: float,
) -> dict[str, Any]:
    query_token_count = max(1, int(query_token_count))
    common_prefix_tokens = max(0, int(match.get("common_prefix_tokens", 0) or 0))
    candidate_token_count = max(1, int(match.get("candidate_token_count", 0) or 0))
    candidate_seq_len = max(1, int(match.get("seq_len", candidate_token_count) or candidate_token_count))
    rollback_tokens = max(0, int(match.get("rollback_tokens", 0) or 0))
    tail_token_count = max(0, int(match.get("query_tail_tokens", 0) or 0))
    min_common_tokens_required = _min_prefix_tokens_for_reuse(query_token_count, prefix_tokens)

    prefix_coverage = common_prefix_tokens / query_token_count
    candidate_utilization = common_prefix_tokens / candidate_token_count
    rollback_ratio = rollback_tokens / candidate_seq_len
    tail_ratio = tail_token_count / query_token_count
    repair_ratio = (rollback_tokens + tail_token_count) / max(query_token_count, candidate_seq_len)

    base_score = (
        0.55 * prefix_coverage
        + 0.20 * candidate_utilization
        + 0.15 * (1.0 - rollback_ratio)
        + 0.10 * (1.0 - tail_ratio)
    )
    semantic_similarity = match.get("semantic_similarity")
    semantic_confidence = _normalized_cosine(semantic_similarity)
    if semantic_confidence is not None:
        prefix_reuse_score = 0.85 * base_score + 0.15 * semantic_confidence
    else:
        prefix_reuse_score = base_score

    rejected_reasons: list[str] = []
    if common_prefix_tokens < min_common_tokens_required:
        rejected_reasons.append("common_prefix_below_minimum")
    if prefix_coverage < float(min_prefix_coverage):
        rejected_reasons.append("prefix_coverage_below_threshold")
    if rollback_ratio > float(max_prefix_rollback_ratio):
        rejected_reasons.append("rollback_ratio_above_threshold")
    if tail_ratio > float(max_prefix_tail_ratio):
        rejected_reasons.append("tail_ratio_above_threshold")
    if prefix_reuse_score < float(min_prefix_reuse_score):
        rejected_reasons.append("prefix_reuse_score_below_threshold")

    return {
        "route": "token_prefix_replay",
        "accepted": not rejected_reasons,
        "rejected_reasons": rejected_reasons,
        "min_common_tokens_required": min_common_tokens_required,
        "common_prefix_tokens": common_prefix_tokens,
        "query_token_count": query_token_count,
        "candidate_token_count": candidate_token_count,
        "candidate_seq_len": candidate_seq_len,
        "rollback_tokens": rollback_tokens,
        "tail_token_count": tail_token_count,
        "prefix_coverage": prefix_coverage,
        "candidate_utilization": candidate_utilization,
        "rollback_ratio": rollback_ratio,
        "tail_ratio": tail_ratio,
        "repair_ratio": repair_ratio,
        "semantic_similarity": semantic_similarity,
        "semantic_confidence": semantic_confidence,
        "prefix_reuse_score": prefix_reuse_score,
        "min_prefix_reuse_score": float(min_prefix_reuse_score),
        "min_prefix_coverage": float(min_prefix_coverage),
        "max_prefix_rollback_ratio": float(max_prefix_rollback_ratio),
        "max_prefix_tail_ratio": float(max_prefix_tail_ratio),
    }


class MGXProphetLibrary:
    """
    Persistent state library for prompt-conditioned KV/session snapshots.

    Entries are stored as:
    - `index.json` metadata registry
    - `snapshots/<entry_id>.pt` serialized runtime snapshots
    - `embeddings/<entry_id>.pt` optional normalized query embeddings
    """

    INDEX_VERSION = 1

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.snapshots_dir = self.root_dir / "snapshots"
        self.embeddings_dir = self.root_dir / "embeddings"
        self.index_path = self.root_dir / "index.json"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def snapshot_cache_stats(cls) -> dict[str, Any]:
        return {
            "enabled": _snapshot_cache_enabled(),
            "max_entries": _snapshot_cache_limit(),
            "entries": len(_SNAPSHOT_CACHE),
            "hits": int(_SNAPSHOT_CACHE_HITS),
            "misses": int(_SNAPSHOT_CACHE_MISSES),
            "gpu_cache_enabled": _snapshot_gpu_cache_enabled(),
            "gpu_cache_max_mb": _snapshot_gpu_cache_max_bytes() / (1024 * 1024),
            "gpu_cache_device": (
                str(_snapshot_gpu_cache_device())
                if _snapshot_gpu_cache_device() is not None
                else None
            ),
            "cached_tensor_mb": _SNAPSHOT_CACHE_BYTES / (1024 * 1024),
        }

    @classmethod
    def resident_cache_stats(cls, engine=None) -> dict[str, Any]:
        records = list(_RESIDENT_CACHE.items())
        if engine is not None:
            engine_id = id(engine)
            records = [(key, value) for key, value in records if key[0] == engine_id]
        live_records = 0
        total_blocks = 0
        for _, record in records:
            engine_ref = record.get("engine_ref")
            record_engine = engine_ref() if engine_ref is not None else None
            seq_id = int(record.get("seq_id", 0) or 0)
            block_manager = getattr(record_engine, "block_manager", None)
            block_tables = getattr(block_manager, "block_tables", {})
            if seq_id in block_tables:
                live_records += 1
                total_blocks += len(block_tables.get(seq_id, []))
        return {
            "max_entries": _resident_cache_limit(),
            "entries": len(records),
            "live_entries": live_records,
            "resident_blocks": total_blocks,
        }

    # ---------------------------------------------------------------------
    # Index persistence
    # ---------------------------------------------------------------------

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {
                "format": "mgx-prophet-index",
                "version": self.INDEX_VERSION,
                "created_at": _now_iso8601(),
                "entries": [],
            }
        with self.index_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("format") != "mgx-prophet-index":
            raise ValueError(f"Invalid MGX Prophet index format in {self.index_path}")
        if int(data.get("version", 0) or 0) != self.INDEX_VERSION:
            raise ValueError(
                f"Unsupported MGX Prophet index version {data.get('version')} in {self.index_path}"
            )
        data.setdefault("entries", [])
        return data

    def _save_index(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        tmp_path = self.index_path.with_suffix(".tmp")
        with tmp_path.open("wb") as fh:
            fh.write(payload)
        tmp_path.replace(self.index_path)

    def _snapshot_path(self, entry_id: str) -> Path:
        return self.snapshots_dir / f"{entry_id}.pt"

    def _embedding_path(self, entry_id: str) -> Path:
        return self.embeddings_dir / f"{entry_id}.pt"

    def _entry_token_ids(self, entry: dict[str, Any]) -> Optional[list[int]]:
        token_ids = entry.get("token_ids")
        if token_ids is None:
            return None
        try:
            return [int(token_id) for token_id in token_ids]
        except Exception:
            return None

    def _resident_cache_supported(self, engine) -> bool:
        block_manager = getattr(engine, "block_manager", None)
        return (
            hasattr(engine, "fork_context_prefix")
            and block_manager is not None
            and type(block_manager).__name__ == "BlockManager"
        )

    def _resident_record(self, engine, entry: dict[str, Any]) -> Optional[dict[str, Any]]:
        entry_id = entry.get("entry_id")
        if not entry_id:
            return None
        key = _resident_cache_key(engine, str(entry_id))
        record = _RESIDENT_CACHE.get(key)
        if record is None:
            return None
        seq_id = int(record.get("seq_id", 0) or 0)
        if seq_id not in getattr(engine.block_manager, "block_tables", {}):
            _RESIDENT_CACHE.pop(key, None)
            return None
        _RESIDENT_CACHE.move_to_end(key)
        return record

    def _prune_resident_cache(self, engine, max_entries: int) -> None:
        max_entries = max(0, int(max_entries))
        if max_entries <= 0:
            return
        engine_id = id(engine)
        while sum(1 for key in _RESIDENT_CACHE if key[0] == engine_id) > max_entries:
            evict_key = None
            for key in _RESIDENT_CACHE:
                if key[0] == engine_id:
                    evict_key = key
                    break
            if evict_key is None:
                return
            record = _RESIDENT_CACHE.pop(evict_key, None)
            if record is None:
                continue
            seq_id = int(record.get("seq_id", 0) or 0)
            if seq_id in getattr(engine.block_manager, "block_tables", {}):
                engine.free_sequence(seq_id)

    def _ensure_resident_source(
        self,
        engine,
        entry: dict[str, Any],
        snapshot: dict[str, Any],
        *,
        max_new_tokens: int,
        max_entries: Optional[int],
    ) -> tuple[Optional[int], dict[str, Any]]:
        if not self._resident_cache_supported(engine):
            return None, {"enabled": False, "reason": "unsupported_block_manager"}
        entry_id = entry.get("entry_id")
        if not entry_id:
            return None, {"enabled": False, "reason": "missing_entry_id"}
        limit = _resident_cache_limit() if max_entries is None else max(0, int(max_entries))
        if limit <= 0:
            return None, {"enabled": False, "reason": "disabled"}

        record = self._resident_record(engine, entry)
        if record is not None:
            return int(record["seq_id"]), {
                "enabled": True,
                "hit": True,
                "entry_id": str(entry_id),
                "source_seq_id": int(record["seq_id"]),
            }

        source_seq_id = _new_resident_seq_id(engine)
        engine.restore_context(snapshot, seq_id=source_seq_id, max_new_tokens=max_new_tokens)
        key = _resident_cache_key(engine, str(entry_id))
        _RESIDENT_CACHE[key] = {
            "engine_ref": weakref.ref(engine),
            "entry_id": str(entry_id),
            "seq_id": int(source_seq_id),
            "created_at": _now_iso8601(),
            "seq_len": int(snapshot.get("seq_len", 0) or 0),
            "label": entry.get("label"),
        }
        _RESIDENT_CACHE.move_to_end(key)
        self._prune_resident_cache(engine, limit)
        return source_seq_id, {
            "enabled": True,
            "hit": False,
            "entry_id": str(entry_id),
            "source_seq_id": int(source_seq_id),
            "max_entries": limit,
        }

    def _fork_resident_source(
        self,
        engine,
        entry: dict[str, Any],
        snapshot: dict[str, Any],
        *,
        seq_id: Optional[int],
        max_new_tokens: int,
        max_entries: Optional[int],
    ) -> tuple[Optional[int], dict[str, Any]]:
        source_seq_id, info = self._ensure_resident_source(
            engine,
            entry,
            snapshot,
            max_new_tokens=max_new_tokens,
            max_entries=max_entries,
        )
        if source_seq_id is None:
            return None, info
        forked_seq_id = engine.fork_context_prefix(
            source_seq_id,
            seq_id=seq_id,
            max_new_tokens=max_new_tokens,
        )
        info = dict(info)
        info["forked"] = True
        info["seq_id"] = int(forked_seq_id)
        return forked_seq_id, info

    # ---------------------------------------------------------------------
    # Hashing and compatibility helpers
    # ---------------------------------------------------------------------

    def _hash_prefix_tokens(self, tokenizer, text: Optional[str], prefix_tokens: int) -> Optional[str]:
        if text is None:
            return None
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        tokens = [int(token_id) for token_id in tokens[:prefix_tokens]]
        payload = ",".join(str(token_id) for token_id in tokens)
        return _sha256_text(payload)

    def _engine_compatibility_fingerprint(self, engine) -> dict[str, Any]:
        manifest = getattr(engine.model, "_mgx_manifest", None) or {}
        return {
            "model_name": getattr(engine, "_model_name", None),
            "source_model_hash": manifest.get("source_model_hash"),
            "tokenizer_hash": manifest.get("tokenizer_hash"),
            "chat_template_hash": manifest.get("chat_template_hash"),
            "quantization": manifest.get("quantization"),
            "dtype": manifest.get("dtype"),
            "target_backend": manifest.get("target_backend"),
        }

    def _entry_is_compatible(self, entry: dict[str, Any], engine_fingerprint: dict[str, Any]) -> tuple[bool, str]:
        entry_source_hash = entry.get("source_model_hash")
        engine_source_hash = engine_fingerprint.get("source_model_hash")
        if entry_source_hash and engine_source_hash and entry_source_hash != engine_source_hash:
            return False, "source_model_hash"

        entry_tokenizer_hash = entry.get("tokenizer_hash")
        engine_tokenizer_hash = engine_fingerprint.get("tokenizer_hash")
        if entry_tokenizer_hash and engine_tokenizer_hash and entry_tokenizer_hash != engine_tokenizer_hash:
            return False, "tokenizer_hash"

        entry_chat_hash = entry.get("chat_template_hash")
        engine_chat_hash = engine_fingerprint.get("chat_template_hash")
        if entry_chat_hash and engine_chat_hash and entry_chat_hash != engine_chat_hash:
            return False, "chat_template_hash"

        for key in ("quantization", "dtype", "target_backend"):
            entry_value = entry.get(key)
            engine_value = engine_fingerprint.get(key)
            if key == "dtype":
                entry_value = _canonical_dtype_name(entry_value)
                engine_value = _canonical_dtype_name(engine_value)
            if entry_value and engine_value and entry_value != engine_value:
                return False, key

        return True, "compatible"

    # ---------------------------------------------------------------------
    # Entry management
    # ---------------------------------------------------------------------

    def capture(
        self,
        engine,
        seq_id: int,
        *,
        text: Optional[str] = None,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        snapshot = engine.save_context(seq_id, text=text)
        return self.record_snapshot(
            snapshot,
            label=label,
            metadata=metadata,
            tokenizer=getattr(engine, "tokenizer", None),
            model_name=getattr(engine, "_model_name", None),
        )

    def record_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tokenizer=None,
        model_name: Optional[str] = None,
        prefix_tokens: int = 16,
    ) -> dict[str, Any]:
        metadata = _json_safe(metadata or {})
        text = snapshot.get("text")
        text_hash = _sha256_text(text)
        prefix_hash = None
        if tokenizer is not None:
            try:
                prefix_hash = self._hash_prefix_tokens(tokenizer, text, prefix_tokens)
            except Exception:
                prefix_hash = None

        seed = "|".join(
            [
                snapshot.get("source_model_hash", "") or "",
                snapshot.get("tokenizer_hash", "") or "",
                snapshot.get("chat_template_hash", "") or "",
                text_hash or "",
                str(snapshot.get("seq_len", 0)),
                str(time.time_ns()),
            ]
        )
        entry_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
        snapshot_path = self._snapshot_path(entry_id)
        embedding_path = self._embedding_path(entry_id)

        torch.save(snapshot, snapshot_path)
        _remember_snapshot(snapshot_path, snapshot)
        has_embedding = torch.is_tensor(snapshot.get("embedding"))
        if has_embedding:
            torch.save(snapshot["embedding"].detach().cpu().float(), embedding_path)
        token_ids = snapshot.get("token_ids")
        if token_ids is not None:
            try:
                token_ids = [int(token_id) for token_id in token_ids]
            except Exception:
                token_ids = None

        entry = {
            "entry_id": entry_id,
            "created_at": _now_iso8601(),
            "label": label,
            "metadata": metadata,
            "snapshot_path": str(snapshot_path),
            "embedding_path": str(embedding_path) if has_embedding else None,
            "has_embedding": has_embedding,
            "seq_len": int(snapshot.get("seq_len", 0) or 0),
            "token_ids": token_ids,
            "token_count": len(token_ids) if token_ids is not None else None,
            "text": text,
            "text_hash": text_hash,
            "prefix_hash": prefix_hash,
            "model_name": snapshot.get("model_name") or model_name,
            "source_model_id": snapshot.get("source_model_id"),
            "source_model_hash": snapshot.get("source_model_hash"),
            "tokenizer_hash": snapshot.get("tokenizer_hash"),
            "chat_template_hash": snapshot.get("chat_template_hash"),
            "quantization": snapshot.get("quantization"),
            "dtype": snapshot.get("dtype"),
            "target_backend": snapshot.get("target_backend"),
        }

        index = self._load_index()
        index["entries"].append(entry)
        index["updated_at"] = _now_iso8601()
        self._save_index(index)
        return entry

    def list_entries(self) -> list[dict[str, Any]]:
        index = self._load_index()
        return list(index.get("entries", []))

    def stats(self) -> dict[str, Any]:
        entries = self.list_entries()
        with_embedding = sum(1 for entry in entries if entry.get("has_embedding"))
        return {
            "root_dir": str(self.root_dir),
            "entries": len(entries),
            "entries_with_embedding": with_embedding,
        }

    def _query_token_ids(self, engine, text: str) -> list[int]:
        _, input_ids = engine._prepare_prompt_inputs(text)
        return [int(token_id) for token_id in input_ids[0].tolist()]

    def _find_best_token_prefix_candidate(
        self,
        engine,
        query_token_ids: list[int],
        *,
        prefix_tokens: int,
        require_compatible: bool = True,
    ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        if not query_token_ids:
            return None, None

        engine_fingerprint = self._engine_compatibility_fingerprint(engine)
        min_common_tokens = _min_prefix_tokens_for_reuse(len(query_token_ids), prefix_tokens)
        best_entry = None
        best_snapshot = None
        best_key = None

        for entry in self.list_entries():
            compatible, reason = self._entry_is_compatible(entry, engine_fingerprint)
            if require_compatible and not compatible:
                continue

            snapshot_path = entry.get("snapshot_path")
            if not snapshot_path:
                continue
            snapshot_file = Path(snapshot_path)
            if not snapshot_file.exists():
                continue

            snapshot = None
            candidate_token_ids = self._entry_token_ids(entry)
            if candidate_token_ids is None:
                snapshot = _load_snapshot(snapshot_file)
                candidate_token_ids = snapshot.get("token_ids")
                if candidate_token_ids is not None:
                    candidate_token_ids = [int(token_id) for token_id in candidate_token_ids]
            if not candidate_token_ids:
                continue

            common_prefix_tokens = _common_prefix_length(query_token_ids, candidate_token_ids)
            if common_prefix_tokens < min_common_tokens:
                continue

            query_tail_tokens = len(query_token_ids) - common_prefix_tokens
            if query_tail_tokens <= 0:
                continue

            candidate_seq_len = int(entry.get("seq_len", 0) or 0)
            if snapshot is not None:
                candidate_seq_len = int(snapshot.get("seq_len", candidate_seq_len) or candidate_seq_len)
            if candidate_seq_len <= 0:
                candidate_seq_len = len(candidate_token_ids)
            rollback_tokens = max(0, candidate_seq_len - common_prefix_tokens)
            candidate = {
                **entry,
                "compatible": compatible,
                "compatibility_reason": reason,
                "route": "token_prefix_replay",
                "common_prefix_tokens": common_prefix_tokens,
                "query_token_count": len(query_token_ids),
                "candidate_token_count": len(candidate_token_ids),
                "seq_len": candidate_seq_len,
                "query_tail_tokens": query_tail_tokens,
                "rollback_tokens": rollback_tokens,
                "score": float(common_prefix_tokens),
            }
            key = (
                common_prefix_tokens,
                -query_tail_tokens,
                -rollback_tokens,
                entry.get("created_at", ""),
            )
            if best_key is None or key > best_key:
                best_entry = candidate
                best_snapshot = snapshot
                best_key = key

        if best_entry is not None and best_snapshot is None:
            best_snapshot_path = best_entry.get("snapshot_path")
            if best_snapshot_path:
                best_snapshot = _load_snapshot(best_snapshot_path)

        return best_entry, best_snapshot

    # ---------------------------------------------------------------------
    # Lookup and restore
    # ---------------------------------------------------------------------

    def lookup(
        self,
        engine,
        text: str,
        *,
        top_k: int = 3,
        min_similarity: float = 0.35,
        prefix_tokens: int = 16,
        require_compatible: bool = True,
    ) -> list[dict[str, Any]]:
        engine_fingerprint = self._engine_compatibility_fingerprint(engine)
        query_text_hash = _sha256_text(text)
        query_prefix_hash = self._hash_prefix_tokens(engine.tokenizer, text, prefix_tokens)
        query_embedding = None

        matches: list[dict[str, Any]] = []
        for entry in self.list_entries():
            compatible, reason = self._entry_is_compatible(entry, engine_fingerprint)
            if require_compatible and not compatible:
                continue

            exact_text_match = bool(entry.get("text_hash") and entry.get("text_hash") == query_text_hash)
            prefix_match = bool(entry.get("prefix_hash") and entry.get("prefix_hash") == query_prefix_hash)
            similarity = None
            if not exact_text_match and entry.get("has_embedding") and entry.get("embedding_path"):
                embedding_path = Path(entry["embedding_path"])
                if embedding_path.exists():
                    if query_embedding is None:
                        query_embedding = engine.extract_embedding(text).float()
                    stored_embedding = torch.load(embedding_path, weights_only=False).float()
                    if stored_embedding.ndim != 1:
                        stored_embedding = stored_embedding.reshape(-1)
                    denom = query_embedding.norm().item() * stored_embedding.norm().item()
                    if denom > 0:
                        similarity = float(torch.dot(query_embedding, stored_embedding).item() / denom)
                        similarity = max(-1.0, min(1.0, similarity))

            score = 0.0
            if exact_text_match:
                score = 1.0
            else:
                if similarity is not None:
                    score += similarity
                if prefix_match:
                    score += 0.15

            if not exact_text_match and not prefix_match and (similarity is None or similarity < min_similarity):
                continue

            matches.append(
                {
                    **entry,
                    "compatible": compatible,
                    "compatibility_reason": reason,
                    "exact_text_match": exact_text_match,
                    "prefix_match": prefix_match,
                    "semantic_similarity": similarity,
                    "score": score,
                }
            )

        matches.sort(
            key=lambda item: (
                1 if item.get("exact_text_match") else 0,
                float(item.get("score", 0.0)),
                1 if item.get("prefix_match") else 0,
                int(item.get("seq_len", 0)),
                item.get("created_at", ""),
            ),
            reverse=True,
        )
        return matches[: max(1, int(top_k))]

    def restore_best(
        self,
        engine,
        text: str,
        *,
        seq_id: int = None,
        max_new_tokens: int = 0,
        top_k: int = 3,
        min_similarity: float = 0.35,
        prefix_tokens: int = 16,
        require_compatible: bool = True,
    ) -> dict[str, Any]:
        matches = self.lookup(
            engine,
            text,
            top_k=top_k,
            min_similarity=min_similarity,
            prefix_tokens=prefix_tokens,
            require_compatible=require_compatible,
        )
        if not matches:
            return {
                "restored": False,
                "reason": "no_match",
                "matches": [],
            }

        best = matches[0]
        snapshot = _load_snapshot(best["snapshot_path"])
        restored_seq_id = engine.restore_context(
            snapshot,
            seq_id=seq_id,
            max_new_tokens=max_new_tokens,
        )
        return {
            "restored": True,
            "seq_id": restored_seq_id,
            "match": best,
            "matches": matches,
        }

    def restore_exact_batch(
        self,
        engine,
        texts: list[str],
        *,
        seq_ids: Optional[list[int]] = None,
        max_new_tokens: int = 0,
        top_k: int = 3,
        min_similarity: float = 0.35,
        prefix_tokens: int = 16,
        require_compatible: bool = True,
        use_resident_cache: bool = False,
        resident_cache_max_entries: Optional[int] = None,
    ) -> Optional[list[dict[str, Any]]]:
        """
        Restore a batch of exact Prophet hits with one batched KV write.

        This intentionally handles only exact text matches with pending logits.
        Prefix-replay and semantic validation still use the regular per-request
        path because they may need rollback, replay, or fallback prefill.
        """
        if not texts:
            return []

        plans: list[dict[str, Any]] = []
        snapshots: list[dict[str, Any]] = []
        for text in texts:
            matches = self.lookup(
                engine,
                text,
                top_k=top_k,
                min_similarity=min_similarity,
                prefix_tokens=prefix_tokens,
                require_compatible=require_compatible,
            )
            best = matches[0] if matches else None
            if best is None or not best.get("exact_text_match"):
                return None
            snapshot = _load_snapshot(best["snapshot_path"])
            if not torch.is_tensor(snapshot.get("pending_next_logits")):
                return None
            plans.append({"match": best, "matches": matches})
            snapshots.append(snapshot)

        if seq_ids is None:
            planned_seq_ids = [engine._next_seq_id() for _ in snapshots]
        else:
            planned_seq_ids = [int(seq_id) for seq_id in seq_ids]
        if len(planned_seq_ids) != len(snapshots):
            raise ValueError("seq_ids length must match texts length")

        if use_resident_cache:
            records: list[dict[str, Any]] = []
            forked_seq_ids: list[int] = []
            resident_failed = False
            for planned_seq_id, plan, snapshot in zip(planned_seq_ids, plans, snapshots):
                try:
                    forked_seq_id, resident_info = self._fork_resident_source(
                        engine,
                        plan["match"],
                        snapshot,
                        seq_id=planned_seq_id,
                        max_new_tokens=max_new_tokens,
                        max_entries=resident_cache_max_entries,
                    )
                except Exception as exc:
                    resident_info = {
                        "enabled": True,
                        "forked": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    forked_seq_id = None
                if forked_seq_id is None:
                    resident_failed = True
                    break
                forked_seq_ids.append(int(forked_seq_id))
                records.append(
                    {
                        "restored": True,
                        "seq_id": int(forked_seq_id),
                        "match": plan["match"],
                        "matches": plan["matches"],
                        "committed_source": "prophet_resident_fork_batch",
                        "speculative_accepted": True,
                        "reason": "exact_text_match_resident_fork_batch",
                        "decision_trace": [
                            {
                                "route": "exact_text_match_resident_batch",
                                "accepted": True,
                                "candidate_has_pending_next_logits": True,
                                "resident_cache": resident_info,
                            }
                        ],
                        "validation": {
                            "mode": "exact_batch_bypass",
                            "candidate_has_pending_next_logits": True,
                            "resident_cache": resident_info,
                        },
                    }
                )
            if not resident_failed:
                return records
            for forked_seq_id in forked_seq_ids:
                if forked_seq_id in getattr(engine.block_manager, "block_tables", {}):
                    engine.free_sequence(forked_seq_id)

        restored_seq_ids = engine.restore_contexts(
            snapshots,
            seq_ids=planned_seq_ids,
            max_new_tokens=max_new_tokens,
        )
        records: list[dict[str, Any]] = []
        for restored_seq_id, plan in zip(restored_seq_ids, plans):
            records.append(
                {
                    "restored": True,
                    "seq_id": int(restored_seq_id),
                    "match": plan["match"],
                    "matches": plan["matches"],
                    "committed_source": "prophet_exact_batch",
                    "speculative_accepted": True,
                    "reason": "exact_text_match_batch",
                    "decision_trace": [
                        {
                            "route": "exact_text_match_batch",
                            "accepted": True,
                            "candidate_has_pending_next_logits": True,
                        }
                    ],
                    "validation": {
                        "mode": "exact_batch_bypass",
                        "candidate_has_pending_next_logits": True,
                    },
                }
            )
        return records

    def restore_speculative(
        self,
        engine,
        text: str,
        *,
        seq_id: int = None,
        max_new_tokens: int = 0,
        top_k: int = 3,
        min_similarity: float = 0.35,
        prefix_tokens: int = 16,
        require_compatible: bool = True,
        validation_mode: str = "full_prefill",
        validation_tokens: int = 4,
        agreement_threshold: float = 1.0,
        fallback_to_prefill: bool = True,
        min_prefix_reuse_score: float = 0.55,
        min_prefix_coverage: float = 0.50,
        max_prefix_rollback_ratio: float = 0.35,
        max_prefix_tail_ratio: float = 0.50,
        use_resident_cache: bool = False,
        resident_cache_max_entries: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Speculatively restore the best Prophet candidate, validate it, and commit or rollback.

        Validation modes:
        - `none`: optimistic restore, no validator
        - `full_prefill`: correctness-first validator that compares a short greedy continuation
          against a freshly prefilling baseline for the incoming prompt

        Prefix replay routes are additionally gated by a recovery policy that scores:
        - shared token coverage
        - candidate utilization
        - rollback cost
        - tail replay cost
        """
        query_token_ids = self._query_token_ids(engine, text)
        decision_trace: list[dict[str, Any]] = []
        matches = self.lookup(
            engine,
            text,
            top_k=top_k,
            min_similarity=min_similarity,
            prefix_tokens=prefix_tokens,
            require_compatible=require_compatible,
        )
        best = matches[0] if matches else None
        snapshot = _load_snapshot(best["snapshot_path"]) if best is not None else None
        candidate_has_pending = bool(torch.is_tensor(snapshot.get("pending_next_logits"))) if snapshot else False

        if best is not None and best.get("exact_text_match") and candidate_has_pending:
            resident_info = None
            restored_seq_id = None
            if use_resident_cache:
                try:
                    restored_seq_id, resident_info = self._fork_resident_source(
                        engine,
                        best,
                        snapshot,
                        seq_id=seq_id,
                        max_new_tokens=max_new_tokens,
                        max_entries=resident_cache_max_entries,
                    )
                except Exception as exc:
                    resident_info = {
                        "enabled": True,
                        "forked": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    restored_seq_id = None
            if restored_seq_id is None:
                restored_seq_id = engine.restore_context(
                    snapshot,
                    seq_id=seq_id,
                    max_new_tokens=max_new_tokens,
                )
                committed_source = "prophet_exact"
                reason = "exact_text_match"
            else:
                committed_source = "prophet_resident_fork"
                reason = "exact_text_match_resident_fork"
            return {
                "restored": True,
                "seq_id": restored_seq_id,
                "match": best,
                "matches": matches,
                "committed_source": committed_source,
                "speculative_accepted": True,
                "reason": reason,
                "decision_trace": [
                    {
                        "route": "exact_text_match_resident" if resident_info and resident_info.get("forked") else "exact_text_match",
                        "accepted": True,
                        "candidate_has_pending_next_logits": True,
                        "resident_cache": resident_info,
                    }
                ],
                "validation": {
                    "mode": "exact_bypass",
                    "candidate_has_pending_next_logits": True,
                    "resident_cache": resident_info,
                },
            }

        prefix_match, prefix_snapshot = self._find_best_token_prefix_candidate(
            engine,
            query_token_ids,
            prefix_tokens=prefix_tokens,
            require_compatible=require_compatible,
        )
        if prefix_match is not None and prefix_snapshot is not None:
            prefix_policy = _build_prefix_reuse_policy(
                prefix_match,
                query_token_count=len(query_token_ids),
                prefix_tokens=prefix_tokens,
                min_prefix_reuse_score=min_prefix_reuse_score,
                min_prefix_coverage=min_prefix_coverage,
                max_prefix_rollback_ratio=max_prefix_rollback_ratio,
                max_prefix_tail_ratio=max_prefix_tail_ratio,
            )
            decision_trace.append(
                {
                    "route": "token_prefix_replay",
                    "accepted": bool(prefix_policy["accepted"]),
                    "entry_id": prefix_match.get("entry_id"),
                    "prefix_reuse_score": prefix_policy["prefix_reuse_score"],
                    "rejected_reasons": list(prefix_policy["rejected_reasons"]),
                }
            )
        else:
            prefix_policy = None

        if prefix_policy is not None and not prefix_policy["accepted"]:
            # A candidate rejected by the explicit token-reuse policy must not
            # silently re-enter through semantic validation.  Retain any other
            # independent semantic candidates instead of disabling the route
            # altogether.
            rejected_entry_id = prefix_match.get("entry_id")
            matches = [
                match
                for match in matches
                if match.get("entry_id") != rejected_entry_id
            ]
            best = matches[0] if matches else None
            snapshot = (
                _load_snapshot(best["snapshot_path"])
                if best is not None
                else None
            )
            candidate_has_pending = bool(
                torch.is_tensor(snapshot.get("pending_next_logits"))
            ) if snapshot else False

        if prefix_match is not None and prefix_snapshot is not None and prefix_policy is not None and prefix_policy["accepted"]:
            restored_seq_id = None
            try:
                common_prefix_tokens = int(prefix_match["common_prefix_tokens"])
                tail_tokens = query_token_ids[common_prefix_tokens:]
                replay_headroom = max(0, int(max_new_tokens)) + len(tail_tokens)
                resident_info = None
                if use_resident_cache:
                    try:
                        restored_seq_id, resident_info = self._fork_resident_source(
                            engine,
                            prefix_match,
                            prefix_snapshot,
                            seq_id=seq_id,
                            max_new_tokens=replay_headroom,
                            max_entries=resident_cache_max_entries,
                        )
                    except Exception as exc:
                        resident_info = {
                            "enabled": True,
                            "forked": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        restored_seq_id = None
                if restored_seq_id is None:
                    restored_seq_id = engine.restore_context(
                        prefix_snapshot,
                        seq_id=seq_id,
                        max_new_tokens=replay_headroom,
                    )
                original_seq_len = int(prefix_snapshot.get("seq_len", 0) or 0)
                if common_prefix_tokens < original_seq_len:
                    engine.truncate_context(restored_seq_id, common_prefix_tokens)

                replay_result = engine.replay_tokens_into_context(
                    restored_seq_id,
                    tail_tokens,
                )
                return {
                    "restored": True,
                    "seq_id": restored_seq_id,
                    "match": prefix_match,
                    "matches": matches,
                    "committed_source": (
                        "prophet_resident_prefix_reuse"
                        if resident_info and resident_info.get("forked")
                        else "prophet_prefix_reuse"
                    ),
                    "speculative_accepted": True,
                    "reason": "token_prefix_reuse",
                    "decision_trace": decision_trace,
                    "validation": {
                        "mode": "token_prefix_replay",
                        "policy": prefix_policy,
                        "replay_result": replay_result,
                        "restore_extra_tokens": replay_headroom,
                        "resident_cache": resident_info,
                    },
                }
            except Exception:
                if restored_seq_id is not None and restored_seq_id in engine.block_manager.block_tables:
                    engine.free_sequence(restored_seq_id)
                raise

        if best is None or snapshot is None:
            return {
                "restored": False,
                "reason": "no_match",
                "committed_source": None,
                "speculative_accepted": False,
                "matches": [],
                "decision_trace": decision_trace,
                "validation": {
                    "mode": "no_match",
                    "policy": prefix_policy,
                },
            }

        decision_trace.append(
            {
                "route": "semantic_validation",
                "accepted": None,
                "entry_id": best.get("entry_id"),
                "semantic_similarity": best.get("semantic_similarity"),
                "exact_text_match": bool(best.get("exact_text_match", False)),
                "prefix_match": bool(best.get("prefix_match", False)),
                "candidate_has_pending_next_logits": candidate_has_pending,
            }
        )

        restored_seq_id = engine.restore_context(
            snapshot,
            seq_id=seq_id,
            max_new_tokens=max_new_tokens,
        )
        candidate_has_pending = bool(torch.is_tensor(snapshot.get("pending_next_logits")))

        if validation_mode == "none":
            if candidate_has_pending:
                return {
                    "restored": True,
                    "seq_id": restored_seq_id,
                    "match": best,
                    "matches": matches,
                    "committed_source": "prophet_no_validation",
                    "speculative_accepted": True,
                    "reason": "validation_disabled",
                    "decision_trace": decision_trace,
                    "validation": {
                        "mode": "none",
                        "candidate_has_pending_next_logits": True,
                        "policy": prefix_policy,
                    },
                }
            engine.free_sequence(restored_seq_id)
            return {
                "restored": False,
                "reason": "candidate_missing_pending_next_logits",
                "committed_source": None,
                "speculative_accepted": False,
                "match": best,
                "matches": matches,
                "decision_trace": decision_trace,
                "validation": {
                    "mode": "none",
                    "candidate_has_pending_next_logits": False,
                    "policy": prefix_policy,
                },
            }

        if validation_mode != "full_prefill":
            engine.free_sequence(restored_seq_id)
            raise ValueError(
                f"Unsupported Prophet validation_mode '{validation_mode}'. "
                "Expected 'none' or 'full_prefill'."
            )

        validator_seq_id = None
        try:
            validator_seq_id = engine._next_seq_id()
            validator_info = engine.prefill_context(
                text,
                seq_id=validator_seq_id,
                max_new_tokens=max_new_tokens,
            )

            validation = {
                "mode": "full_prefill",
                "candidate_has_pending_next_logits": candidate_has_pending,
                "validator_seq_id": validator_seq_id,
                "validator_prompt_len": validator_info["prompt_len"],
                "validation_tokens_requested": max(0, int(validation_tokens)),
                "agreement_threshold": float(agreement_threshold),
                "policy": prefix_policy,
            }

            accepted = False
            candidate_result = None
            validator_result = None
            agreement = 0.0
            first_token_match = False

            if candidate_has_pending and validation_tokens > 0:
                candidate_result = engine.generate_from_context(
                    restored_seq_id,
                    max_new_tokens=validation_tokens,
                    temperature=0.0,
                    top_k=0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                )
                validator_result = engine.generate_from_context(
                    validator_seq_id,
                    max_new_tokens=validation_tokens,
                    temperature=0.0,
                    top_k=0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                )
                candidate_tokens = list(candidate_result.get("token_ids", []))
                validator_tokens = list(validator_result.get("token_ids", []))
                compare_len = min(len(candidate_tokens), len(validator_tokens))
                if compare_len > 0:
                    matched = sum(
                        1 for cand_tok, val_tok in zip(candidate_tokens[:compare_len], validator_tokens[:compare_len])
                        if int(cand_tok) == int(val_tok)
                    )
                    agreement = matched / compare_len
                    first_token_match = int(candidate_tokens[0]) == int(validator_tokens[0])
                accepted = first_token_match and agreement >= float(agreement_threshold)
                validation["candidate_result"] = {
                    "token_ids": candidate_tokens,
                    "text": candidate_result.get("text"),
                    "stopped": bool(candidate_result.get("stopped", False)),
                }
                validation["validator_result"] = {
                    "token_ids": validator_tokens,
                    "text": validator_result.get("text"),
                    "stopped": bool(validator_result.get("stopped", False)),
                }
            elif candidate_has_pending and validation_tokens <= 0:
                accepted = True
            else:
                accepted = False

            validation["agreement"] = agreement
            validation["first_token_match"] = first_token_match
            validation["accepted"] = accepted
            if decision_trace and decision_trace[-1].get("route") == "semantic_validation":
                decision_trace[-1]["accepted"] = accepted

            if accepted:
                engine.free_sequence(validator_seq_id)
                return {
                    "restored": True,
                    "seq_id": restored_seq_id,
                    "match": best,
                    "matches": matches,
                    "committed_source": "prophet_validated",
                    "speculative_accepted": True,
                    "reason": "validation_passed",
                    "decision_trace": decision_trace,
                    "validation": validation,
                }

            engine.free_sequence(restored_seq_id)
            if fallback_to_prefill:
                return {
                    "restored": True,
                    "seq_id": validator_seq_id,
                    "match": best,
                    "matches": matches,
                    "committed_source": "prefill_fallback",
                    "speculative_accepted": False,
                    "reason": (
                        "candidate_missing_pending_next_logits"
                        if not candidate_has_pending
                        else "validation_failed"
                    ),
                    "decision_trace": decision_trace,
                    "validation": validation,
                }

            engine.free_sequence(validator_seq_id)
            return {
                "restored": False,
                "seq_id": None,
                "match": best,
                "matches": matches,
                "committed_source": None,
                "speculative_accepted": False,
                "reason": (
                    "candidate_missing_pending_next_logits"
                    if not candidate_has_pending
                    else "validation_failed"
                ),
                "decision_trace": decision_trace,
                "validation": validation,
            }
        except Exception:
            if validator_seq_id is not None and validator_seq_id in engine.block_manager.block_tables:
                engine.free_sequence(validator_seq_id)
            if restored_seq_id in engine.block_manager.block_tables:
                engine.free_sequence(restored_seq_id)
            raise

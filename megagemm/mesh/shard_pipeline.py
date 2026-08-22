"""Coordinator for experimental MegaMesh layer-shard generation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Iterable, List, Sequence

import torch

from .binary_codec import CONTENT_TYPE, decode_tensor_frame, encode_tensor_frame
from .protocol import MeshEndpoint, parse_worker_specs, request_binary, request_json
from .tensor_codec import tensor_from_payload, tensor_to_payload
from .ttp import TTPClientPool


class ShardPipeline:
    """Greedy batch-1 pipeline across ordered layer-shard workers."""

    def __init__(
        self,
        stages: str | Iterable[str] | Sequence[MeshEndpoint],
        *,
        model_name: str,
        timeout: float = 900.0,
        transport: str = "binary",
        enable_thinking: bool | None = None,
        remote_chain_loop: bool = True,
    ) -> None:
        if isinstance(stages, Sequence) and stages and isinstance(stages[0], MeshEndpoint):
            self.stages = list(stages)  # type: ignore[arg-type]
        else:
            self.stages = parse_worker_specs(stages)  # type: ignore[arg-type]
        if len(self.stages) < 2:
            raise ValueError("layer-shard generation requires at least two stages")
        if not str(model_name).strip():
            raise ValueError(
                "mesh-shard generation received an empty model name. "
                "Pass --model explicitly, e.g. --model Qwen/Qwen3-14B."
            )
        self.model_name = model_name
        self.timeout = float(timeout)
        if transport not in ("ttp", "binary", "json"):
            raise ValueError("MegaMesh shard transport must be 'ttp', 'binary', or 'json'")
        self.transport = transport
        self.enable_thinking = enable_thinking
        self.remote_chain_loop = bool(remote_chain_loop)
        self._ttp_pool = TTPClientPool(self.stages, timeout=self.timeout) if transport == "ttp" else None
        self.tokenizer = self._load_tokenizer(model_name)
        self._seq_counter = 0
        self._ttp_chain_requests = 0

    @staticmethod
    def _load_tokenizer(model_name: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to use mesh-shard-generate") from exc
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def health(self) -> list[dict[str, Any]]:
        if self.transport == "ttp":
            if self._ttp_pool is None:
                raise RuntimeError("TTP client pool is not initialized")
            rows = []
            for stage in self.stages:
                row, _ = self._ttp_pool.request(stage, "health")
                row.setdefault("endpoint", stage.url)
                rows.append(row)
            return rows
        rows = []
        for stage in self.stages:
            row = request_json(stage, "/health", None, self.timeout)
            row.setdefault("endpoint", stage.base_url)
            rows.append(row)
        return rows

    def _stage_snapshot(self) -> list[dict[str, Any]]:
        try:
            health = self.health()
        except Exception as exc:
            return [{"ok": False, "error": str(exc)}]
        snapshot = []
        for row in health:
            snapshot.append(
                {
                    "name": row.get("name"),
                    "endpoint": row.get("endpoint"),
                    "layer_start": row.get("layer_start"),
                    "layer_end": row.get("layer_end"),
                    "fastpath": row.get("fastpath"),
                    "kernel_stats": row.get("kernel_stats"),
                    "lm_head": row.get("lm_head"),
                    "mlp": row.get("mlp"),
                    "ttp_runtime": row.get("ttp_runtime"),
                }
            )
        return snapshot

    def _format_prompt(self, prompt: str) -> str:
        tokenizer = self.tokenizer
        if getattr(tokenizer, "chat_template", None):
            try:
                kwargs: dict[str, Any] = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                if self.enable_thinking is not None:
                    kwargs["enable_thinking"] = self.enable_thinking
                return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
            except Exception:
                return prompt
        return prompt

    def _encode_prompt(self, prompt: str) -> list[int]:
        formatted = self._format_prompt(prompt)
        bos = getattr(self.tokenizer, "bos_token", None)
        add_special_tokens = not (bos and formatted.startswith(bos))
        return self.tokenizer.encode(formatted, add_special_tokens=add_special_tokens)

    def _next_seq_id(self) -> int:
        self._seq_counter += 1
        return self._seq_counter

    def _run_stage_chain(
        self,
        path: str,
        *,
        seq_id: int,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[int | None, torch.Tensor | None]:
        if self.transport == "ttp":
            return self._run_stage_chain_ttp(
                path,
                seq_id=seq_id,
                positions=positions,
                input_ids=input_ids,
                hidden=hidden,
            )
        if self.transport == "json":
            return self._run_stage_chain_json(
                path,
                seq_id=seq_id,
                positions=positions,
                input_ids=input_ids,
                hidden=hidden,
            )
        return self._run_stage_chain_binary(
            path,
            seq_id=seq_id,
            positions=positions,
            input_ids=input_ids,
            hidden=hidden,
        )

    def _run_stage_chain_json(
        self,
        path: str,
        *,
        seq_id: int,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[int | None, torch.Tensor | None]:
        next_token = None
        cur_hidden = hidden
        for idx, stage in enumerate(self.stages):
            payload: dict[str, Any] = {
                "seq_id": int(seq_id),
                "positions": tensor_to_payload(positions),
            }
            if idx == 0:
                if input_ids is None:
                    raise ValueError("first stage requires input_ids")
                payload["input_ids"] = tensor_to_payload(input_ids)
            else:
                if cur_hidden is None:
                    raise ValueError("intermediate stage requires hidden")
                payload["hidden"] = tensor_to_payload(cur_hidden)
            result = request_json(stage, path, payload, self.timeout)
            if "hidden" in result:
                cur_hidden = tensor_from_payload(result["hidden"], device="cpu")
            else:
                next_token = int(result["next_token"])
                cur_hidden = None
        return next_token, cur_hidden

    def _run_stage_chain_binary(
        self,
        path: str,
        *,
        seq_id: int,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[int | None, torch.Tensor | None]:
        next_token = None
        cur_hidden = hidden
        bin_path = path.rstrip("/") + ".bin"
        for idx, stage in enumerate(self.stages):
            tensors: dict[str, torch.Tensor] = {"positions": positions}
            if idx == 0:
                if input_ids is None:
                    raise ValueError("first stage requires input_ids")
                tensors["input_ids"] = input_ids
            else:
                if cur_hidden is None:
                    raise ValueError("intermediate stage requires hidden")
                tensors["hidden"] = cur_hidden

            body = encode_tensor_frame({"seq_id": int(seq_id)}, tensors)
            response = request_binary(
                stage,
                bin_path,
                body,
                self.timeout,
                content_type=CONTENT_TYPE,
            )
            meta, out_tensors = decode_tensor_frame(response, device="cpu")
            if meta.get("ok") is False:
                raise RuntimeError(str(meta.get("error", meta)))
            if "hidden" in out_tensors:
                cur_hidden = out_tensors["hidden"]
            else:
                next_token = int(meta["next_token"])
                cur_hidden = None
        return next_token, cur_hidden

    def _run_stage_chain_ttp(
        self,
        path: str,
        *,
        seq_id: int,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[int | None, torch.Tensor | None]:
        if self._ttp_pool is None:
            raise RuntimeError("TTP client pool is not initialized")
        if len(self.stages) == 2 and hidden is None:
            if input_ids is None:
                raise ValueError("first stage requires input_ids")
            op = path.strip("/")
            meta, _ = self._ttp_pool.request(
                self.stages[0],
                f"{op}_chain",
                {"seq_id": int(seq_id), "next_stage": self.stages[1].url},
                {
                    "positions": positions,
                    "input_ids": input_ids,
                },
                device="cpu",
            )
            self._ttp_chain_requests += 1
            if "next_token" in meta:
                return int(meta["next_token"]), None
            return None, None
        next_token = None
        cur_hidden = hidden
        op = path.strip("/")
        for idx, stage in enumerate(self.stages):
            tensors: dict[str, torch.Tensor] = {"positions": positions}
            if idx == 0:
                if input_ids is None:
                    raise ValueError("first stage requires input_ids")
                tensors["input_ids"] = input_ids
            else:
                if cur_hidden is None:
                    raise ValueError("intermediate stage requires hidden")
                tensors["hidden"] = cur_hidden

            meta, out_tensors = self._ttp_pool.request(
                stage,
                op,
                {"seq_id": int(seq_id)},
                tensors,
                device="cpu",
            )
            if "hidden" in out_tensors:
                cur_hidden = out_tensors["hidden"]
            else:
                next_token = int(meta["next_token"])
                cur_hidden = None
        return next_token, cur_hidden

    def _run_stage_chain_ttp_batch(
        self,
        op: str,
        *,
        seq_ids: list[int],
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[list[int] | None, torch.Tensor | None]:
        if self._ttp_pool is None:
            raise RuntimeError("TTP client pool is not initialized")
        if len(self.stages) == 2 and hidden is None:
            if input_ids is None:
                raise ValueError("first stage requires input_ids")
            meta, _ = self._ttp_pool.request(
                self.stages[0],
                f"{op}_chain",
                {
                    "seq_ids": [int(seq_id) for seq_id in seq_ids],
                    "next_stage": self.stages[1].url,
                },
                {
                    "positions": positions,
                    "input_ids": input_ids,
                },
                device="cpu",
            )
            self._ttp_chain_requests += 1
            if "next_tokens" in meta:
                return [int(token) for token in meta["next_tokens"]], None
            return None, None
        next_tokens = None
        cur_hidden = hidden
        for idx, stage in enumerate(self.stages):
            tensors: dict[str, torch.Tensor] = {"positions": positions}
            if idx == 0:
                if input_ids is None:
                    raise ValueError("first stage requires input_ids")
                tensors["input_ids"] = input_ids
            else:
                if cur_hidden is None:
                    raise ValueError("intermediate stage requires hidden")
                tensors["hidden"] = cur_hidden

            meta, out_tensors = self._ttp_pool.request(
                stage,
                op,
                {"seq_ids": [int(seq_id) for seq_id in seq_ids]},
                tensors,
                device="cpu",
            )
            if "hidden" in out_tensors:
                cur_hidden = out_tensors["hidden"]
            else:
                next_tokens = [int(token) for token in meta["next_tokens"]]
                cur_hidden = None
        return next_tokens, cur_hidden

    def _run_two_stage_ttp_decode_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> list[list[int]]:
        """Pipeline-overlap decode_batch across exactly two TTP stages."""

        if self._ttp_pool is None:
            raise RuntimeError("TTP client pool is not initialized")
        if len(self.stages) != 2:
            raise ValueError("two-stage TTP pipeline requires exactly two stages")
        if not chunks:
            return []

        stage0, stage1 = self.stages

        def run_stage0(chunk: dict[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
            return self._ttp_pool.request(
                stage0,
                "decode_batch",
                {"seq_ids": chunk["seq_ids"]},
                {
                    "positions": chunk["positions"],
                    "input_ids": chunk["input_ids"],
                },
                device="cpu",
            )

        def run_stage1(
            chunk: dict[str, Any],
            hidden: torch.Tensor,
        ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
            return self._ttp_pool.request(
                stage1,
                "decode_batch",
                {"seq_ids": chunk["seq_ids"]},
                {
                    "positions": chunk["positions"],
                    "hidden": hidden,
                },
                device="cpu",
            )

        results: list[list[int] | None] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=2) as executor:
            stage0_future = executor.submit(run_stage0, chunks[0])
            stage1_future = None
            stage1_index = -1

            for idx, chunk in enumerate(chunks):
                _, out_tensors = stage0_future.result()
                if "hidden" not in out_tensors:
                    raise RuntimeError("first shard did not return hidden during decode_batch")
                hidden = out_tensors["hidden"]

                next_stage0_future = None
                if idx + 1 < len(chunks):
                    next_stage0_future = executor.submit(run_stage0, chunks[idx + 1])

                if stage1_future is not None:
                    meta, _ = stage1_future.result()
                    results[stage1_index] = [int(token) for token in meta["next_tokens"]]

                stage1_future = executor.submit(run_stage1, chunk, hidden)
                stage1_index = idx

                if next_stage0_future is not None:
                    stage0_future = next_stage0_future

            if stage1_future is not None:
                meta, _ = stage1_future.result()
                results[stage1_index] = [int(token) for token in meta["next_tokens"]]

        return [tokens for tokens in results if tokens is not None]

    def _eos_ids(self) -> set[int]:
        eos_ids = set()
        for value in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
        ):
            if value is not None:
                if isinstance(value, (list, tuple, set)):
                    eos_ids.update(int(item) for item in value)
                else:
                    eos_ids.add(int(value))
        return eos_ids

    def _can_remote_chain_loop(self) -> bool:
        return bool(self.transport == "ttp" and self.remote_chain_loop and len(self.stages) >= 2)

    def _generate_ttp_remote_chain(
        self,
        *,
        seq_id: int,
        token_ids: list[int],
        max_new_tokens: int,
        include_prompt: bool,
        chain_start: int,
        t0: float,
    ) -> dict[str, Any]:
        if self._ttp_pool is None:
            raise RuntimeError("TTP client pool is not initialized")

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        positions = torch.arange(len(token_ids), dtype=torch.long).unsqueeze(0)
        meta, _ = self._ttp_pool.request(
            self.stages[0],
            "generate_chain",
            {
                "seq_id": int(seq_id),
                "next_stages": [stage.url for stage in self.stages[1:]],
                "max_new_tokens": int(max_new_tokens),
                "eos_ids": sorted(self._eos_ids()),
            },
            {
                "positions": positions,
                "input_ids": input_ids,
            },
            device="cpu",
        )
        self._ttp_chain_requests += 1
        generated = [int(token) for token in meta.get("generated", [])]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        decode_ids = token_ids + generated if include_prompt else generated
        text = self.tokenizer.decode(decode_ids, skip_special_tokens=True)
        return {
            "ok": True,
            "seq_id": seq_id,
            "text": text,
            "prompt_tokens": len(token_ids),
            "generated_tokens": len(generated),
            "transport": self.transport,
            "pipeline": {
                "remote_chain_loop": True,
                "coordinator_ttp_requests": self._ttp_chain_requests - chain_start,
                "worker_chain_forwards": int(meta.get("chain_forwards", 0)),
                "decode_steps": int(meta.get("decode_steps", 0)),
            },
            "ttp_chain_requests": self._ttp_chain_requests - chain_start,
            "stages": self._stage_snapshot(),
            "elapsed_ms": round(elapsed_ms, 3),
            "tokens_per_second": round(
                len(generated) / max(elapsed_ms / 1000.0, 1e-9),
                3,
            ),
        }

    def _generate_batch_ttp_remote_chain(
        self,
        *,
        rows: list[dict[str, Any]],
        max_new_tokens: int,
        microbatch_size: int,
        include_prompt: bool,
        chain_start: int,
        t0: float,
    ) -> dict[str, Any]:
        if self._ttp_pool is None:
            raise RuntimeError("TTP client pool is not initialized")

        max_prompt_len = max(len(row["token_ids"]) for row in rows)
        padded = torch.zeros((len(rows), max_prompt_len), dtype=torch.long)
        prompt_lengths = torch.empty((len(rows),), dtype=torch.long)
        for idx, row in enumerate(rows):
            token_ids = row["token_ids"]
            padded[idx, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            prompt_lengths[idx] = int(len(token_ids))

        meta, _ = self._ttp_pool.request(
            self.stages[0],
            "generate_batch_chain",
            {
                "seq_ids": [int(row["seq_id"]) for row in rows],
                "next_stages": [stage.url for stage in self.stages[1:]],
                "max_new_tokens": int(max_new_tokens),
                "microbatch_size": int(microbatch_size),
                "eos_ids": sorted(self._eos_ids()),
            },
            {
                "input_ids": padded,
                "prompt_lengths": prompt_lengths,
            },
            device="cpu",
        )
        self._ttp_chain_requests += 1
        generated_rows = meta.get("generated", [])
        if not isinstance(generated_rows, list):
            raise RuntimeError("remote chain loop did not return generated token rows")
        if len(generated_rows) != len(rows):
            raise RuntimeError(
                "remote chain loop returned "
                f"{len(generated_rows)} rows for {len(rows)} prompts"
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        outputs = []
        total_generated = 0
        for row, generated in zip(rows, generated_rows):
            generated_ids = [int(token) for token in generated]
            total_generated += len(generated_ids)
            decode_ids = (
                row["token_ids"] + generated_ids
                if include_prompt
                else generated_ids
            )
            outputs.append(
                {
                    "seq_id": row["seq_id"],
                    "text": self.tokenizer.decode(decode_ids, skip_special_tokens=True),
                    "prompt_tokens": len(row["token_ids"]),
                    "generated_tokens": len(generated_ids),
                }
            )

        return {
            "ok": True,
            "transport": self.transport,
            "microbatch_size": microbatch_size,
            "num_prompts": len(rows),
            "generated_tokens": total_generated,
            "pipeline": {
                "remote_chain_loop": True,
                "decode_steps": int(meta.get("decode_steps", 0)),
                "total_decode_chunks": int(meta.get("total_decode_chunks", 0)),
                "max_decode_chunks_per_step": int(meta.get("max_decode_chunks_per_step", 0)),
                "pipelined_decode_steps": 0,
                "ttp_chain_requests": self._ttp_chain_requests - chain_start,
                "coordinator_ttp_requests": self._ttp_chain_requests - chain_start,
                "worker_chain_forwards": int(meta.get("chain_forwards", 0)),
                "overlap_enabled": False,
            },
            "stages": self._stage_snapshot(),
            "elapsed_ms": round(elapsed_ms, 3),
            "tokens_per_second": round(
                total_generated / max(elapsed_ms / 1000.0, 1e-9),
                3,
            ),
            "outputs": outputs,
        }

    def close(self) -> None:
        if self._ttp_pool is not None:
            self._ttp_pool.close()


    def _generate_continuous_ttp_remote_chain(
        self,
        *,
        rows: list[dict[str, Any]],
        max_new_tokens: int,
        microbatch_size: int,
        max_batch_size: int,
        include_prompt: bool,
        chain_start: int,
        t0: float,
    ) -> dict[str, Any]:
        if self._ttp_pool is None:
            raise RuntimeError("TTP client pool is not initialized")

        max_prompt_len = max(len(row["token_ids"]) for row in rows)
        padded = torch.zeros((len(rows), max_prompt_len), dtype=torch.long)
        prompt_lengths = torch.empty((len(rows),), dtype=torch.long)
        for idx, row in enumerate(rows):
            token_ids = row["token_ids"]
            padded[idx, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            prompt_lengths[idx] = int(len(token_ids))

        meta, _ = self._ttp_pool.request(
            self.stages[0],
            "generate_continuous_chain",
            {
                "seq_ids": [int(row["seq_id"]) for row in rows],
                "next_stages": [stage.url for stage in self.stages[1:]],
                "max_new_tokens": int(max_new_tokens),
                "microbatch_size": int(microbatch_size),
                "max_batch_size": int(max_batch_size),
                "eos_ids": sorted(self._eos_ids()),
            },
            {
                "input_ids": padded,
                "prompt_lengths": prompt_lengths,
            },
            device="cpu",
        )
        self._ttp_chain_requests += 1
        generated_rows = meta.get("generated", [])
        if not isinstance(generated_rows, list):
            raise RuntimeError("continuous remote chain did not return generated token rows")
        if len(generated_rows) != len(rows):
            raise RuntimeError(
                "continuous remote chain returned "
                f"{len(generated_rows)} rows for {len(rows)} prompts"
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        outputs = []
        total_generated = 0
        for row, generated in zip(rows, generated_rows):
            generated_ids = [int(token) for token in generated]
            total_generated += len(generated_ids)
            decode_ids = (
                row["token_ids"] + generated_ids
                if include_prompt
                else generated_ids
            )
            outputs.append(
                {
                    "seq_id": row["seq_id"],
                    "text": self.tokenizer.decode(decode_ids, skip_special_tokens=True),
                    "prompt_tokens": len(row["token_ids"]),
                    "generated_tokens": len(generated_ids),
                }
            )

        return {
            "ok": True,
            "transport": self.transport,
            "continuous_batching": True,
            "microbatch_size": microbatch_size,
            "max_batch_size": max_batch_size,
            "num_prompts": len(rows),
            "generated_tokens": total_generated,
            "pipeline": {
                "remote_chain_loop": True,
                "continuous_batching": True,
                "scheduler_steps": int(meta.get("scheduler_steps", 0)),
                "admission_events": int(meta.get("admission_events", 0)),
                "total_prefills": int(meta.get("total_prefills", 0)),
                "max_running": int(meta.get("max_running", 0)),
                "decode_steps": int(meta.get("decode_steps", 0)),
                "total_decode_chunks": int(meta.get("total_decode_chunks", 0)),
                "max_decode_chunks_per_step": int(meta.get("max_decode_chunks_per_step", 0)),
                "pipelined_decode_steps": 0,
                "ttp_chain_requests": self._ttp_chain_requests - chain_start,
                "coordinator_ttp_requests": self._ttp_chain_requests - chain_start,
                "worker_chain_forwards": int(meta.get("chain_forwards", 0)),
                "overlap_enabled": False,
            },
            "stages": self._stage_snapshot(),
            "elapsed_ms": round(elapsed_ms, 3),
            "tokens_per_second": round(
                total_generated / max(elapsed_ms / 1000.0, 1e-9),
                3,
            ),
            "outputs": outputs,
        }

    def free(self, seq_id: int) -> None:
        if self.transport == "ttp":
            if self._ttp_pool is None:
                return
            for stage in self.stages:
                try:
                    self._ttp_pool.request(stage, "free", {"seq_id": int(seq_id)})
                except Exception:
                    pass
            return
        for stage in self.stages:
            try:
                request_json(stage, "/free", {"seq_id": int(seq_id)}, self.timeout)
            except Exception:
                pass

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        seq_id: int | None = None,
        include_prompt: bool = False,
    ) -> dict[str, Any]:
        seq_id = self._next_seq_id() if seq_id is None else int(seq_id)
        token_ids = self._encode_prompt(prompt)
        if not token_ids:
            raise ValueError("prompt produced no tokens")
        generated: List[int] = []
        chain_start = self._ttp_chain_requests
        t0 = time.perf_counter()

        try:
            if self._can_remote_chain_loop():
                return self._generate_ttp_remote_chain(
                    seq_id=seq_id,
                    token_ids=token_ids,
                    max_new_tokens=max_new_tokens,
                    include_prompt=include_prompt,
                    chain_start=chain_start,
                    t0=t0,
                )

            input_ids = torch.tensor([token_ids], dtype=torch.long)
            positions = torch.arange(len(token_ids), dtype=torch.long).unsqueeze(0)
            next_token, _ = self._run_stage_chain(
                "/prefill",
                seq_id=seq_id,
                positions=positions,
                input_ids=input_ids,
            )
            if next_token is None:
                raise RuntimeError("last shard did not return next_token during prefill")

            eos_ids = self._eos_ids()

            cur_token = int(next_token)
            for step in range(int(max_new_tokens)):
                generated.append(cur_token)
                if cur_token in eos_ids or len(generated) >= int(max_new_tokens):
                    break
                pos = len(token_ids) + step
                input_ids = torch.tensor([[cur_token]], dtype=torch.long)
                positions = torch.tensor([[pos]], dtype=torch.long)
                next_token, _ = self._run_stage_chain(
                    "/decode",
                    seq_id=seq_id,
                    positions=positions,
                    input_ids=input_ids,
                )
                if next_token is None:
                    raise RuntimeError("last shard did not return next_token during decode")
                cur_token = int(next_token)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            decode_ids = token_ids + generated if include_prompt else generated
            text = self.tokenizer.decode(decode_ids, skip_special_tokens=True)
            stage_snapshot = self._stage_snapshot()
            return {
                "ok": True,
                "seq_id": seq_id,
                "text": text,
                "prompt_tokens": len(token_ids),
                "generated_tokens": len(generated),
                "transport": self.transport,
                "ttp_chain_requests": self._ttp_chain_requests - chain_start,
                "stages": stage_snapshot,
                "elapsed_ms": round(elapsed_ms, 3),
                "tokens_per_second": round(
                    len(generated) / max(elapsed_ms / 1000.0, 1e-9),
                    3,
                ),
            }
        finally:
            self.free(seq_id)

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 64,
        microbatch_size: int = 8,
        include_prompt: bool = False,
    ) -> dict[str, Any]:
        """Generate multiple prompts with batched shard decode microsteps."""

        if self.transport != "ttp":
            raise ValueError("generate_batch currently requires --transport ttp")
        prompts = list(prompts)
        if not prompts:
            raise ValueError("generate_batch requires at least one prompt")
        microbatch_size = max(1, int(microbatch_size))
        eos_ids = self._eos_ids()
        decode_steps = 0
        total_decode_chunks = 0
        max_decode_chunks_per_step = 0
        pipelined_decode_steps = 0
        chain_start = self._ttp_chain_requests
        t0 = time.perf_counter()

        rows = []
        seq_ids: list[int] = []
        try:
            for prompt in prompts:
                seq_id = self._next_seq_id()
                seq_ids.append(seq_id)
                token_ids = self._encode_prompt(prompt)
                if not token_ids:
                    raise ValueError("prompt produced no tokens")
                rows.append(
                    {
                        "seq_id": seq_id,
                        "prompt": prompt,
                        "token_ids": token_ids,
                        "generated": [],
                        "done": False,
                    }
                )

            if self._can_remote_chain_loop():
                return self._generate_batch_ttp_remote_chain(
                    rows=rows,
                    max_new_tokens=max_new_tokens,
                    microbatch_size=microbatch_size,
                    include_prompt=include_prompt,
                    chain_start=chain_start,
                    t0=t0,
                )

            for row in rows:
                seq_id = int(row["seq_id"])
                token_ids = row["token_ids"]
                input_ids = torch.tensor([token_ids], dtype=torch.long)
                positions = torch.arange(len(token_ids), dtype=torch.long).unsqueeze(0)
                next_token, _ = self._run_stage_chain(
                    "/prefill",
                    seq_id=seq_id,
                    positions=positions,
                    input_ids=input_ids,
                )
                if next_token is None:
                    raise RuntimeError("last shard did not return next_token during prefill")
                row["cur_token"] = int(next_token)

            while True:
                ready = [
                    row
                    for row in rows
                    if not row["done"] and len(row["generated"]) < int(max_new_tokens)
                ]
                if not ready:
                    break
                decode_steps += 1

                decode_ready = []
                for row in ready:
                    cur_token = int(row["cur_token"])
                    row["generated"].append(cur_token)
                    if cur_token in eos_ids or len(row["generated"]) >= int(max_new_tokens):
                        row["done"] = True
                    else:
                        decode_ready.append(row)

                decode_chunks = []
                for start in range(0, len(decode_ready), microbatch_size):
                    chunk = decode_ready[start : start + microbatch_size]
                    chunk_seq_ids = [int(row["seq_id"]) for row in chunk]
                    chunk_tokens = [[int(row["cur_token"])] for row in chunk]
                    chunk_positions = [
                        [len(row["token_ids"]) + len(row["generated"]) - 1]
                        for row in chunk
                    ]
                    decode_chunks.append(
                        {
                            "rows": chunk,
                            "seq_ids": chunk_seq_ids,
                            "input_ids": torch.tensor(chunk_tokens, dtype=torch.long),
                            "positions": torch.tensor(chunk_positions, dtype=torch.long),
                        }
                    )

                total_decode_chunks += len(decode_chunks)
                max_decode_chunks_per_step = max(max_decode_chunks_per_step, len(decode_chunks))
                if len(self.stages) == 2 and len(decode_chunks) > 1:
                    pipelined_decode_steps += 1
                    chunk_results = self._run_two_stage_ttp_decode_chunks(decode_chunks)
                    for chunk_data, next_tokens in zip(decode_chunks, chunk_results):
                        for row, token in zip(chunk_data["rows"], next_tokens):
                            row["cur_token"] = int(token)
                else:
                    for chunk_data in decode_chunks:
                        next_tokens, _ = self._run_stage_chain_ttp_batch(
                            "decode_batch",
                            seq_ids=chunk_data["seq_ids"],
                            positions=chunk_data["positions"],
                            input_ids=chunk_data["input_ids"],
                        )
                        if next_tokens is None:
                            raise RuntimeError("last shard did not return next_tokens")
                        for row, token in zip(chunk_data["rows"], next_tokens):
                            row["cur_token"] = int(token)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            outputs = []
            total_generated = 0
            for row in rows:
                total_generated += len(row["generated"])
                decode_ids = (
                    row["token_ids"] + row["generated"]
                    if include_prompt
                    else row["generated"]
                )
                outputs.append(
                    {
                        "seq_id": row["seq_id"],
                        "text": self.tokenizer.decode(decode_ids, skip_special_tokens=True),
                        "prompt_tokens": len(row["token_ids"]),
                        "generated_tokens": len(row["generated"]),
                    }
                )
            stage_snapshot = self._stage_snapshot()
            return {
                "ok": True,
                "transport": self.transport,
                "microbatch_size": microbatch_size,
                "num_prompts": len(prompts),
                "generated_tokens": total_generated,
                "pipeline": {
                    "decode_steps": decode_steps,
                    "total_decode_chunks": total_decode_chunks,
                    "max_decode_chunks_per_step": max_decode_chunks_per_step,
                    "pipelined_decode_steps": pipelined_decode_steps,
                    "ttp_chain_requests": self._ttp_chain_requests - chain_start,
                    "overlap_enabled": bool(
                        len(self.stages) == 2 and max_decode_chunks_per_step > 1
                    ),
                },
                "stages": stage_snapshot,
                "elapsed_ms": round(elapsed_ms, 3),
                "tokens_per_second": round(
                    total_generated / max(elapsed_ms / 1000.0, 1e-9),
                    3,
                ),
                "outputs": outputs,
            }
        finally:
            for seq_id in seq_ids:
                self.free(seq_id)

    def generate_continuous(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 64,
        microbatch_size: int = 8,
        max_batch_size: int = 32,
        include_prompt: bool = False,
    ) -> dict[str, Any]:
        """Generate a prompt queue with MegaMesh shard continuous batching."""

        if self.transport != "ttp":
            raise ValueError("generate_continuous currently requires --transport ttp")
        if not self._can_remote_chain_loop():
            raise ValueError("generate_continuous requires TTP remote chain loop")
        prompts = list(prompts)
        if not prompts:
            raise ValueError("generate_continuous requires at least one prompt")
        microbatch_size = max(1, int(microbatch_size))
        max_batch_size = max(1, int(max_batch_size))
        chain_start = self._ttp_chain_requests
        t0 = time.perf_counter()

        rows = []
        seq_ids: list[int] = []
        try:
            for prompt in prompts:
                seq_id = self._next_seq_id()
                seq_ids.append(seq_id)
                token_ids = self._encode_prompt(prompt)
                if not token_ids:
                    raise ValueError("prompt produced no tokens")
                rows.append(
                    {
                        "seq_id": seq_id,
                        "prompt": prompt,
                        "token_ids": token_ids,
                    }
                )

            return self._generate_continuous_ttp_remote_chain(
                rows=rows,
                max_new_tokens=max_new_tokens,
                microbatch_size=microbatch_size,
                max_batch_size=max_batch_size,
                include_prompt=include_prompt,
                chain_start=chain_start,
                t0=t0,
            )
        finally:
            for seq_id in seq_ids:
                self.free(seq_id)


def _parse_replica_specs(replicas: str | Iterable[str]) -> list[str]:
    if isinstance(replicas, str):
        parts = replicas.split(";")
    else:
        parts = list(replicas)
    parsed = [str(part).strip() for part in parts if str(part).strip()]
    if not parsed:
        raise ValueError("at least one shard replica pipeline is required")
    return parsed


class ShardReplicaRouter:
    """Coordinator-side router over replicated MegaMesh shard pipelines."""

    def __init__(
        self,
        replicas: str | Iterable[str],
        *,
        model_name: str,
        timeout: float = 900.0,
        transport: str = "ttp",
        enable_thinking: bool | None = None,
        remote_chain_loop: bool = True,
    ) -> None:
        self.replica_specs = _parse_replica_specs(replicas)
        self.pipelines = [
            ShardPipeline(
                spec,
                model_name=model_name,
                timeout=timeout,
                transport=transport,
                enable_thinking=enable_thinking,
                remote_chain_loop=remote_chain_loop,
            )
            for spec in self.replica_specs
        ]
        self.transport = transport

    def close(self) -> None:
        for pipeline in self.pipelines:
            pipeline.close()

    def health(self) -> list[dict[str, Any]]:
        return [
            {
                "ok": True,
                "replica_index": idx,
                "stages": pipeline.health(),
            }
            for idx, pipeline in enumerate(self.pipelines)
        ]

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 64,
        microbatch_size: int = 8,
        include_prompt: bool = False,
        strategy: str = "round_robin",
    ) -> dict[str, Any]:
        if strategy not in {"round_robin", "chunk"}:
            raise ValueError("replica strategy must be 'round_robin' or 'chunk'")
        if not prompts:
            raise ValueError("generate_batch requires at least one prompt")

        n = len(self.pipelines)
        assignments: list[list[tuple[int, str]]] = [[] for _ in range(n)]
        if strategy == "round_robin":
            for idx, prompt in enumerate(prompts):
                assignments[idx % n].append((idx, prompt))
        else:
            chunk = (len(prompts) + n - 1) // n
            for idx, prompt in enumerate(prompts):
                assignments[min(idx // chunk, n - 1)].append((idx, prompt))

        t0 = time.perf_counter()
        outputs_by_index: dict[int, dict[str, Any]] = {}
        replica_results: list[dict[str, Any]] = []

        def run_one(replica_idx: int, rows: list[tuple[int, str]]) -> tuple[int, dict[str, Any] | None]:
            if not rows:
                return replica_idx, None
            result = self.pipelines[replica_idx].generate_batch(
                [prompt for _, prompt in rows],
                max_new_tokens=max_new_tokens,
                microbatch_size=microbatch_size,
                include_prompt=include_prompt,
            )
            return replica_idx, result

        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [
                executor.submit(run_one, idx, rows)
                for idx, rows in enumerate(assignments)
                if rows
            ]
            for future in futures:
                replica_idx, result = future.result()
                if result is None:
                    continue
                replica_results.append(
                    {
                        "replica_index": replica_idx,
                        "num_prompts": int(result.get("num_prompts", 0)),
                        "generated_tokens": int(result.get("generated_tokens", 0)),
                        "elapsed_ms": float(result.get("elapsed_ms", 0.0)),
                        "tokens_per_second": float(result.get("tokens_per_second", 0.0)),
                        "pipeline": result.get("pipeline"),
                        "stages": result.get("stages"),
                    }
                )
                assigned_rows = assignments[replica_idx]
                for local_idx, output in enumerate(result.get("outputs", [])):
                    global_idx = int(assigned_rows[local_idx][0])
                    row = dict(output)
                    row["replica_index"] = replica_idx
                    row["global_index"] = global_idx
                    outputs_by_index[global_idx] = row

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        outputs = [outputs_by_index[idx] for idx in range(len(prompts))]
        generated_tokens = sum(int(row.get("generated_tokens", 0)) for row in outputs)
        return {
            "ok": True,
            "transport": self.transport,
            "replicated_shards": True,
            "num_replicas": len(self.pipelines),
            "strategy": strategy,
            "microbatch_size": int(microbatch_size),
            "num_prompts": len(prompts),
            "generated_tokens": generated_tokens,
            "elapsed_ms": round(elapsed_ms, 3),
            "tokens_per_second": round(
                generated_tokens / max(elapsed_ms / 1000.0, 1e-9),
                3,
            ),
            "replicas": sorted(replica_results, key=lambda row: int(row["replica_index"])),
            "outputs": outputs,
        }


__all__ = ["ShardPipeline", "ShardReplicaRouter"]

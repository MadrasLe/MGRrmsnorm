"""
Encoder-style embedding engine with support for common Sentence Transformers layouts.

The first target is throughput-oriented batch encoding for:
- plain Hugging Face encoder models
- MiniLM / MPNet / BERT-family embedding checkpoints
- common Sentence Transformers layouts with Pooling/Dense/Normalize
"""

from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .formats import DenseSpec, load_module_state_dict, load_sentence_transformer_spec
from .native_bert import is_native_bert_supported, load_native_bert_encoder
from .pooling import normalize_embeddings, normalize_pooling_modes, pool_hidden_states

__all__ = ["EmbeddingEngine"]


def _resolve_device(device: str) -> str:
    device = str(device).strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("EmbeddingEngine: CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return device


def _resolve_dtype(dtype: Optional[Union[str, torch.dtype]], device: str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None or str(dtype).strip().lower() in {"", "auto"}:
        if device == "cuda":
            return torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
        return torch.float32

    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(dtype).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    resolved = mapping[key]
    if device == "cpu" and resolved in {torch.float16, torch.bfloat16}:
        return torch.float32
    return resolved


def _prepare_model_path(
    model_name: str,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> str:
    if os.path.isdir(model_name):
        return model_name

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_name

    return snapshot_download(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.bin",
            "*.txt",
            "*.model",
            "*.spm",
            "*.tiktoken",
            "tokenizer*",
            "vocab*",
            "merges.txt",
            "special_tokens_map.json",
        ],
    )


def _build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    return nn.Identity()


class _DenseProjection(nn.Module):
    def __init__(self, spec: DenseSpec):
        super().__init__()
        if spec.in_features <= 0 or spec.out_features <= 0:
            raise ValueError(f"Invalid DenseSpec dimensions: {spec}")
        self.linear = nn.Linear(spec.in_features, spec.out_features, bias=spec.bias)
        self.activation = _build_activation(spec.activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(hidden_states))


class EmbeddingEngine:
    """
    High-throughput encoder embedding engine for GPU and CPU.

    Supports plain Hugging Face encoder checkpoints and the common
    Sentence Transformers pipeline:
        Transformer -> Pooling -> Dense* -> Normalize?
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        cache_dir: Optional[str] = None,
        pooling: Union[str, Iterable[str]] = "auto",
        normalize: Optional[bool] = None,
        max_length: Optional[int] = None,
        trust_remote_code: bool = False,
        prompts: Optional[Dict[str, str]] = None,
        sort_by_length: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        max_batch_tokens: int = 0,
        backend: str = "auto",
        native_padding_free: Optional[bool] = None,
        native_padding_free_force: bool = False,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype, self.device)
        self.backend = str(backend).strip().lower()
        if self.backend not in {"auto", "hf", "native"}:
            raise ValueError("backend must be one of: auto, hf, native")
        self.native_padding_free = (
            True if native_padding_free is None else bool(native_padding_free)
        )
        self.native_padding_free_force = bool(native_padding_free_force)
        self.sort_by_length = sort_by_length
        self.max_batch_tokens = max(0, int(max_batch_tokens or 0))
        self.pad_to_multiple_of = (
            pad_to_multiple_of if pad_to_multiple_of is not None else (8 if self.device == "cuda" else None)
        )
        self._last_batch_plan = None
        self.local_files_only = bool(local_files_only)
        self._resolved_model_path = _prepare_model_path(
            model_name,
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
        )

        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install transformers for encoder embeddings: pip install transformers"
            ) from exc

        self._st_spec = load_sentence_transformer_spec(self._resolved_model_path)

        backbone_path = (
            self._st_spec.transformer_module_dir
            if self._st_spec is not None and self._st_spec.transformer_module_dir
            else self._resolved_model_path
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone_path,
            trust_remote_code=trust_remote_code,
            local_files_only=self.local_files_only,
        )

        try:
            hf_config = AutoConfig.from_pretrained(
                backbone_path,
                trust_remote_code=trust_remote_code,
                local_files_only=self.local_files_only,
            )
        except Exception:
            hf_config = None

        self._runtime_backend = "hf"
        native_supported = hf_config is not None and is_native_bert_supported(hf_config)
        if self.backend == "native" or (self.backend == "auto" and native_supported):
            try:
                self.model = load_native_bert_encoder(
                    backbone_path,
                    device=self.device,
                    dtype=self.dtype,
                    padding_free=self.native_padding_free,
                    padding_free_force=self.native_padding_free_force,
                )
                self._runtime_backend = "native"
            except Exception:
                if self.backend == "native":
                    raise
                model_kwargs = {"trust_remote_code": trust_remote_code, "local_files_only": self.local_files_only}
                if self.device == "cuda":
                    model_kwargs["torch_dtype"] = self.dtype
                self.model = AutoModel.from_pretrained(backbone_path, **model_kwargs)
                self.model.eval()
                self.model.to(self.device)
                if self.device == "cpu" and self.dtype != torch.float32:
                    self.model.to(dtype=self.dtype)
        else:
            model_kwargs = {"trust_remote_code": trust_remote_code, "local_files_only": self.local_files_only}
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = self.dtype
            self.model = AutoModel.from_pretrained(backbone_path, **model_kwargs)
            self.model.eval()
            self.model.to(self.device)
            if self.device == "cpu" and self.dtype != torch.float32:
                self.model.to(dtype=self.dtype)

        hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise ValueError("EmbeddingEngine requires an encoder model with config.hidden_size")

        if pooling == "auto":
            if self._st_spec and self._st_spec.pooling is not None:
                self.pooling_modes = self._st_spec.pooling.modes
            else:
                self.pooling_modes = ("mean",)
        elif isinstance(pooling, str):
            self.pooling_modes = normalize_pooling_modes(
                part.strip() for part in pooling.split(",")
            )
        else:
            self.pooling_modes = normalize_pooling_modes(pooling)

        self.projection_head = self._build_projection_head(self._st_spec)
        if self.projection_head is not None:
            self.projection_head.to(self.device)
            if self.device == "cuda":
                self.projection_head.to(dtype=self.dtype)
            self.projection_head.eval()

        if normalize is None:
            self.normalize = bool(self._st_spec.normalize) if self._st_spec is not None else False
        else:
            self.normalize = bool(normalize)

        self.max_length = max_length or self._resolve_max_length()
        self.prompts = self._merge_prompts(prompts or {})
        self.default_prompt_name = self._st_spec.default_prompt_name if self._st_spec is not None else None
        self.embedding_dim = self._infer_embedding_dim(hidden_size)
        self._unsupported_modules = list(self._st_spec.unsupported_modules) if self._st_spec is not None else []
        if self._unsupported_modules:
            joined = ", ".join(self._unsupported_modules)
            print(f"EmbeddingEngine: ignoring unsupported Sentence Transformers modules: {joined}")

    def _resolve_max_length(self) -> int:
        max_length = int(getattr(self.tokenizer, "model_max_length", 512) or 512)
        if max_length <= 0 or max_length > 1_000_000:
            return 512
        return max_length

    def _merge_prompts(self, user_prompts: Dict[str, str]) -> Dict[str, str]:
        prompts: Dict[str, str] = {}

        name = self.model_name.lower()
        if "e5" in name:
            prompts.update({
                "query": "query: ",
                "document": "passage: ",
                "passage": "passage: ",
            })
        elif "bge" in name:
            prompts.update({
                "query": "Represent this sentence for searching relevant passages: ",
                "document": "",
                "passage": "",
            })

        if self._st_spec is not None:
            prompts.update(self._st_spec.prompts)
        prompts.update({str(k): str(v) for k, v in user_prompts.items()})
        return prompts

    def _build_projection_head(self, spec) -> Optional[nn.Sequential]:
        if spec is None or not spec.dense_layers:
            return None

        layers: List[nn.Module] = []
        for dense_spec in spec.dense_layers:
            layer = _DenseProjection(dense_spec)
            state = load_module_state_dict(dense_spec.module_dir)
            weight = state.get("linear.weight", state.get("weight"))
            bias = state.get("linear.bias", state.get("bias"))
            if weight is None:
                raise KeyError(f"Dense module in {dense_spec.module_dir} is missing weight")
            layer.linear.weight.data.copy_(weight)
            if layer.linear.bias is not None and bias is not None:
                layer.linear.bias.data.copy_(bias)
            layers.append(layer)
        return nn.Sequential(*layers)

    def _infer_embedding_dim(self, hidden_size: int) -> int:
        if self.projection_head is None:
            return hidden_size * len(self.pooling_modes)
        sample_dim = hidden_size * len(self.pooling_modes)
        for module in self.projection_head:
            if isinstance(module, _DenseProjection):
                sample_dim = module.linear.out_features
        return sample_dim

    def _canonical_task(self, task: Optional[str]) -> Optional[str]:
        if task is None:
            return None
        key = str(task).strip().lower()
        aliases = {
            "doc": "document",
            "document": "document",
            "passage": "passage",
            "query": "query",
        }
        return aliases.get(key, key)

    def _resolve_prompt_template(
        self,
        task: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        if prompt is not None:
            return prompt

        canonical = self._canonical_task(task)
        if canonical is not None:
            if canonical in self.prompts:
                return self.prompts[canonical]
            if canonical == "document" and "passage" in self.prompts:
                return self.prompts["passage"]
            if canonical == "passage" and "document" in self.prompts:
                return self.prompts["document"]

        if self.default_prompt_name and self.default_prompt_name in self.prompts:
            return self.prompts[self.default_prompt_name]
        return None

    def _apply_prompt(self, text: str, template: Optional[str]) -> str:
        if template is None or template == "":
            return text
        if "{text}" in template:
            return template.format(text=text)
        return f"{template}{text}"

    def _prepare_batch(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch = {}
        for key, value in encoded.items():
            if self.device == "cuda" and value.device.type == "cpu":
                value = value.pin_memory().to(self.device, non_blocking=True)
            else:
                value = value.to(self.device)
            batch[key] = value
        return batch

    def _estimate_token_lengths(self, texts: Sequence[str]) -> List[int]:
        if not texts:
            return []
        try:
            encoded = self.tokenizer(
                list(texts),
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_length=True,
                return_attention_mask=True,
            )
            lengths = encoded.get("length")
            if lengths is not None:
                return [int(length) for length in lengths]
            masks = encoded.get("attention_mask")
            if masks is not None:
                return [int(sum(mask)) for mask in masks]
            input_ids = encoded.get("input_ids")
            if input_ids is not None:
                return [len(ids) for ids in input_ids]
        except TypeError:
            pass
        except Exception:
            pass
        return [max(1, len(text.split())) for text in texts]

    def _plan_batches(
        self,
        items: List[Tuple[int, str]],
        batch_size: int,
    ) -> List[List[Tuple[int, str]]]:
        if not items:
            self._last_batch_plan = []
            return []

        batch_size = max(1, int(batch_size))
        if self.max_batch_tokens <= 0:
            batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
            self._last_batch_plan = [
                {"count": len(batch), "padded_tokens": None, "max_tokens": None}
                for batch in batches
            ]
            return batches

        lengths = self._estimate_token_lengths([text for _, text in items])
        annotated = [
            (idx, text, length)
            for (idx, text), length in zip(items, lengths)
        ]
        if self.sort_by_length and len(annotated) > 1:
            annotated.sort(key=lambda item: (item[2], len(item[1])), reverse=True)

        batches: List[List[Tuple[int, str]]] = []
        batch_meta = []
        current: List[Tuple[int, str, int]] = []
        current_max_tokens = 0

        for item in annotated:
            proposed_max = max(current_max_tokens, item[2])
            proposed_count = len(current) + 1
            proposed_padded_tokens = proposed_max * proposed_count
            if current and (len(current) >= batch_size or proposed_padded_tokens > self.max_batch_tokens):
                batches.append([(idx, text) for idx, text, _ in current])
                batch_meta.append(
                    {
                        "count": len(current),
                        "max_tokens": current_max_tokens,
                        "padded_tokens": current_max_tokens * len(current),
                    }
                )
                current = [item]
                current_max_tokens = item[2]
            else:
                current.append(item)
                current_max_tokens = proposed_max

        if current:
            batches.append([(idx, text) for idx, text, _ in current])
            batch_meta.append(
                {
                    "count": len(current),
                    "max_tokens": current_max_tokens,
                    "padded_tokens": current_max_tokens * len(current),
                }
            )

        self._last_batch_plan = batch_meta
        return batches

    @torch.inference_mode()
    def encode(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: int = 32,
        task: Optional[str] = None,
        prompt: Optional[str] = None,
        normalize: Optional[bool] = None,
        return_numpy: bool = False,
    ):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        texts = [str(text) for text in texts]
        if not texts:
            empty = torch.empty(0, self.embedding_dim, dtype=torch.float32)
            return empty.numpy() if return_numpy else empty

        template = self._resolve_prompt_template(task=task, prompt=prompt)
        items = [(idx, self._apply_prompt(text, template)) for idx, text in enumerate(texts)]
        if self.sort_by_length and self.max_batch_tokens <= 0 and len(items) > 1:
            items.sort(key=lambda item: len(item[1]), reverse=True)

        outputs: List[Optional[torch.Tensor]] = [None] * len(items)
        do_normalize = self.normalize if normalize is None else bool(normalize)
        batches = self._plan_batches(items, batch_size=batch_size)

        for chunk in batches:
            chunk_indices = [idx for idx, _ in chunk]
            chunk_texts = [text for _, text in chunk]
            batch = self._prepare_batch(chunk_texts)
            model_out = self.model(**batch)
            if torch.is_tensor(model_out):
                hidden_states = model_out
            elif hasattr(model_out, "last_hidden_state"):
                hidden_states = model_out.last_hidden_state
            elif isinstance(model_out, (tuple, list)):
                hidden_states = model_out[0]
            else:
                raise TypeError("Embedding model output does not expose last_hidden_state")
            pooled = pool_hidden_states(hidden_states, batch["attention_mask"], self.pooling_modes)
            if self.projection_head is not None:
                pooled = self.projection_head(pooled)
            if do_normalize:
                pooled = normalize_embeddings(pooled)
            pooled = pooled.float().cpu()
            for row, original_idx in enumerate(chunk_indices):
                outputs[original_idx] = pooled[row]

        stacked = torch.stack([row for row in outputs if row is not None], dim=0)
        if single:
            result = stacked[0]
        else:
            result = stacked
        if return_numpy:
            return result.numpy()
        return result

    def encode_query(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: int = 32,
        normalize: Optional[bool] = None,
        return_numpy: bool = False,
    ):
        return self.encode(
            texts,
            batch_size=batch_size,
            task="query",
            normalize=normalize,
            return_numpy=return_numpy,
        )

    def encode_document(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: int = 32,
        normalize: Optional[bool] = None,
        return_numpy: bool = False,
    ):
        return self.encode(
            texts,
            batch_size=batch_size,
            task="document",
            normalize=normalize,
            return_numpy=return_numpy,
        )

    @torch.inference_mode()
    def benchmark(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        runs: int = 3,
        warmup: int = 1,
        task: Optional[str] = None,
    ) -> Dict[str, float]:
        texts = [str(text) for text in texts]
        if not texts:
            raise ValueError("benchmark() requires at least one text")

        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        total_tokens = sum(sum(mask) for mask in tokenized["attention_mask"])

        for _ in range(max(0, warmup)):
            self.encode(texts, batch_size=batch_size, task=task)
            if self.device == "cuda":
                torch.cuda.synchronize()

        durations = []
        for _ in range(max(1, runs)):
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.encode(texts, batch_size=batch_size, task=task)
            if self.device == "cuda":
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - t0)

        avg = sum(durations) / len(durations)
        return {
            "backend": self._runtime_backend,
            "native_padding_free": float(bool(self.native_padding_free)),
            "batch_size": float(batch_size),
            "max_batch_tokens": float(self.max_batch_tokens),
            "planned_batches": float(len(self._last_batch_plan or [])),
            "num_texts": float(len(texts)),
            "embedding_dim": float(self.embedding_dim),
            "avg_latency_ms": avg * 1000.0,
            "texts_per_second": len(texts) / avg if avg > 0 else 0.0,
            "tokens_per_second": total_tokens / avg if avg > 0 else 0.0,
        }

    def __repr__(self) -> str:
        modes = ",".join(self.pooling_modes)
        return (
            f"EmbeddingEngine(model={self.model_name!r}, "
            f"backend={self._runtime_backend}, "
            f"device={self.device}, dtype={self.dtype}, "
            f"native_padding_free={self.native_padding_free}, "
            f"pooling={modes}, normalize={self.normalize}, "
            f"max_batch_tokens={self.max_batch_tokens}, dim={self.embedding_dim})"
        )

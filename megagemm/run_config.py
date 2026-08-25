"""Declarative JSON/YAML inference configuration for MegaGemm.

This module intentionally depends only on the Python standard library while a
configuration is being parsed. PyYAML and the inference engine are imported only
when their respective paths are selected.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO


class RunConfigError(ValueError):
    """Raised when a declarative inference configuration is invalid."""


@dataclass(frozen=True)
class EngineRunConfig:
    device: str = "cuda"
    dtype: str = "fp16"
    num_blocks: int = 0
    block_size: int = 16
    max_batch_size: int = 1
    cache_dir: str | None = None
    n_gpu_layers: int = -1
    offload_dir: str | None = None
    quantize: str | None = None
    kv_offload: bool = False
    num_cpu_blocks: int = 0
    gpu_window: int = 64
    kv_alloc: str = "auto"
    max_seq_len: int = 4096
    monitor: bool = False
    dashboard: bool = False
    dashboard_port: int = 8080
    deterministic: bool = False
    seed: int = 42
    mgx_verify_payload: bool | None = None
    mgx_prefer_payload_cache: bool | None = None
    mgx_payload_cache_dir: str | None = None


@dataclass(frozen=True)
class GenerationRunConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False
    verbose: bool = False
    xai: bool = False
    xai_top_k: int = 10
    logit_lens: bool | int = False
    explicit_fields: frozenset[str] = field(default_factory=frozenset, repr=False)


@dataclass(frozen=True)
class OutputRunConfig:
    format: str = "text"
    path: Path | None = None
    include_prompt: bool = True


@dataclass(frozen=True)
class InferenceRunConfig:
    version: int
    task: str
    model: str
    engine: EngineRunConfig
    generation: GenerationRunConfig
    prompt: str | None
    prompts: tuple[str, ...]
    prompts_file: Path | None
    output: OutputRunConfig
    source_path: Path

    @property
    def is_batch(self) -> bool:
        """Whether the selected input explicitly requests the batch API."""

        return bool(self.prompts) or self.prompts_file is not None


@dataclass(frozen=True)
class InferenceRunResult:
    model: str
    prompts: tuple[str, ...]
    outputs: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    output_format: str
    output_path: Path | None


_ROOT_FIELDS = {
    "version",
    "task",
    "model",
    "engine",
    "generation",
    "prompt",
    "prompts",
    "prompts_file",
    "output",
}

_ENGINE_FIELDS = {
    "device",
    "dtype",
    "num_blocks",
    "block_size",
    "max_batch_size",
    "cache_dir",
    "n_gpu_layers",
    "offload_dir",
    "quantize",
    "kv_offload",
    "num_cpu_blocks",
    "gpu_window",
    "kv_alloc",
    "max_seq_len",
    "monitor",
    "dashboard",
    "dashboard_port",
    "deterministic",
    "seed",
    "mgx_verify_payload",
    "mgx_prefer_payload_cache",
    "mgx_payload_cache_dir",
}

_GENERATION_FIELDS = {
    "max_new_tokens",
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "stop_token_ids",
    "ignore_eos",
    "verbose",
    "xai",
    "xai_top_k",
    "logit_lens",
}

_OUTPUT_FIELDS = {"format", "path", "include_prompt"}


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RunConfigError(f"{path} must be a mapping/object")
    return dict(value)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise RunConfigError(f"{path} contains unknown field(s): {joined}")


def _bool(mapping: Mapping[str, Any], key: str, default: bool, path: str) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise RunConfigError(f"{path}.{key} must be a boolean")
    return value


def _optional_bool(
    mapping: Mapping[str, Any], key: str, default: bool | None, path: str
) -> bool | None:
    value = mapping.get(key, default)
    if value is not None and not isinstance(value, bool):
        raise RunConfigError(f"{path}.{key} must be a boolean or null")
    return value


def _int(
    mapping: Mapping[str, Any],
    key: str,
    default: int,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunConfigError(f"{path}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise RunConfigError(f"{path}.{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise RunConfigError(f"{path}.{key} must be <= {maximum}")
    return value


def _float(
    mapping: Mapping[str, Any],
    key: str,
    default: float,
    path: str,
    *,
    minimum: float | None = None,
    exclusive_minimum: bool = False,
    maximum: float | None = None,
) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunConfigError(f"{path}.{key} must be a number")
    result = float(value)
    if minimum is not None:
        if exclusive_minimum and result <= minimum:
            raise RunConfigError(f"{path}.{key} must be > {minimum}")
        if not exclusive_minimum and result < minimum:
            raise RunConfigError(f"{path}.{key} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise RunConfigError(f"{path}.{key} must be <= {maximum}")
    return result


def _string(
    mapping: Mapping[str, Any],
    key: str,
    default: str | None,
    path: str,
    *,
    allow_none: bool = False,
    nonempty: bool = False,
) -> str | None:
    value = mapping.get(key, default)
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        expected = "a string or null" if allow_none else "a string"
        raise RunConfigError(f"{path}.{key} must be {expected}")
    if nonempty and not value.strip():
        raise RunConfigError(f"{path}.{key} cannot be empty")
    return value


def _choice(value: str | None, choices: set[str | None], path: str) -> str | None:
    if value not in choices:
        rendered = ", ".join("null" if item is None else item for item in sorted(
            choices, key=lambda item: "" if item is None else item
        ))
        raise RunConfigError(f"{path} must be one of: {rendered}")
    return value


def _resolve_relative_path(value: str, config_path: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve()


def _parse_engine(raw: Any) -> EngineRunConfig:
    mapping = _mapping(raw, "engine")
    _reject_unknown(mapping, _ENGINE_FIELDS, "engine")

    device = _string(mapping, "device", "cuda", "engine", nonempty=True)
    _choice(device, {"cuda", "cpu"}, "engine.device")
    dtype = _string(mapping, "dtype", "fp16", "engine", nonempty=True)
    _choice(dtype, {"fp16", "bf16", "fp32"}, "engine.dtype")
    quantize = _string(
        mapping, "quantize", None, "engine", allow_none=True, nonempty=True
    )
    _choice(quantize, {None, "int8", "int4", "fp8", "awq"}, "engine.quantize")
    kv_alloc = _string(mapping, "kv_alloc", "auto", "engine", nonempty=True)
    _choice(kv_alloc, {"auto", "greedy"}, "engine.kv_alloc")

    return EngineRunConfig(
        device=str(device),
        dtype=str(dtype),
        num_blocks=_int(mapping, "num_blocks", 0, "engine", minimum=0),
        block_size=_int(mapping, "block_size", 16, "engine", minimum=1),
        max_batch_size=_int(mapping, "max_batch_size", 1, "engine", minimum=1),
        cache_dir=_string(mapping, "cache_dir", None, "engine", allow_none=True),
        n_gpu_layers=_int(mapping, "n_gpu_layers", -1, "engine", minimum=-1),
        offload_dir=_string(mapping, "offload_dir", None, "engine", allow_none=True),
        quantize=quantize,
        kv_offload=_bool(mapping, "kv_offload", False, "engine"),
        num_cpu_blocks=_int(mapping, "num_cpu_blocks", 0, "engine", minimum=0),
        gpu_window=_int(mapping, "gpu_window", 64, "engine", minimum=0),
        kv_alloc=str(kv_alloc),
        max_seq_len=_int(mapping, "max_seq_len", 4096, "engine", minimum=1),
        monitor=_bool(mapping, "monitor", False, "engine"),
        dashboard=_bool(mapping, "dashboard", False, "engine"),
        dashboard_port=_int(
            mapping, "dashboard_port", 8080, "engine", minimum=1, maximum=65535
        ),
        deterministic=_bool(mapping, "deterministic", False, "engine"),
        seed=_int(mapping, "seed", 42, "engine"),
        mgx_verify_payload=_optional_bool(
            mapping, "mgx_verify_payload", None, "engine"
        ),
        mgx_prefer_payload_cache=_optional_bool(
            mapping, "mgx_prefer_payload_cache", None, "engine"
        ),
        mgx_payload_cache_dir=_string(
            mapping, "mgx_payload_cache_dir", None, "engine", allow_none=True
        ),
    )


def _parse_generation(raw: Any) -> GenerationRunConfig:
    mapping = _mapping(raw, "generation")
    _reject_unknown(mapping, _GENERATION_FIELDS, "generation")

    stop_token_ids = mapping.get("stop_token_ids", [])
    if not isinstance(stop_token_ids, Sequence) or isinstance(
        stop_token_ids, (str, bytes)
    ):
        raise RunConfigError("generation.stop_token_ids must be a list of integers")
    normalized_stop_ids: list[int] = []
    for index, value in enumerate(stop_token_ids):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RunConfigError(
                f"generation.stop_token_ids[{index}] must be an integer >= 0"
            )
        normalized_stop_ids.append(value)

    logit_lens = mapping.get("logit_lens", False)
    if isinstance(logit_lens, bool):
        normalized_logit_lens: bool | int = logit_lens
    elif isinstance(logit_lens, int) and logit_lens > 0:
        normalized_logit_lens = logit_lens
    else:
        raise RunConfigError("generation.logit_lens must be a boolean or integer > 0")

    return GenerationRunConfig(
        max_new_tokens=_int(
            mapping, "max_new_tokens", 128, "generation", minimum=1
        ),
        temperature=_float(
            mapping, "temperature", 0.7, "generation", minimum=0.0
        ),
        top_k=_int(mapping, "top_k", 50, "generation", minimum=0),
        top_p=_float(
            mapping,
            "top_p",
            0.9,
            "generation",
            minimum=0.0,
            exclusive_minimum=True,
            maximum=1.0,
        ),
        repetition_penalty=_float(
            mapping,
            "repetition_penalty",
            1.1,
            "generation",
            minimum=0.0,
            exclusive_minimum=True,
        ),
        stop_token_ids=tuple(normalized_stop_ids),
        ignore_eos=_bool(mapping, "ignore_eos", False, "generation"),
        verbose=_bool(mapping, "verbose", False, "generation"),
        xai=_bool(mapping, "xai", False, "generation"),
        xai_top_k=_int(mapping, "xai_top_k", 10, "generation", minimum=1),
        logit_lens=normalized_logit_lens,
        explicit_fields=frozenset(mapping),
    )


def _parse_output(raw: Any, config_path: Path) -> OutputRunConfig:
    mapping = _mapping(raw, "output")
    _reject_unknown(mapping, _OUTPUT_FIELDS, "output")

    output_format = _string(mapping, "format", "text", "output", nonempty=True)
    _choice(output_format, {"text", "json", "jsonl"}, "output.format")
    raw_path = _string(mapping, "path", None, "output", allow_none=True)
    if raw_path == "-":
        output_path = None
    elif raw_path is None:
        output_path = None
    elif not raw_path.strip():
        raise RunConfigError("output.path cannot be empty")
    else:
        output_path = _resolve_relative_path(raw_path, config_path)

    return OutputRunConfig(
        format=str(output_format),
        path=output_path,
        include_prompt=_bool(mapping, "include_prompt", True, "output"),
    )


def _read_payload(path: Path) -> Any:
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunConfigError(f"cannot read configuration {path}: {exc}") from exc

    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RunConfigError(
                f"invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            if exc.name != "yaml":
                raise
            raise RunConfigError(
                "YAML support is optional. Install it with "
                "`pip install -e \".[config]\"` or use a .json configuration."
            ) from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise RunConfigError(f"invalid YAML in {path}: {exc}") from exc

    raise RunConfigError(
        f"unsupported configuration extension {suffix or '<none>'}; use .json, .yaml, or .yml"
    )


def load_run_config(path: str | Path) -> InferenceRunConfig:
    """Load and strictly validate a declarative inference configuration."""

    config_path = Path(path).expanduser().resolve()
    payload = _mapping(_read_payload(config_path), "configuration")
    _reject_unknown(payload, _ROOT_FIELDS, "configuration")

    version = payload.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise RunConfigError("version must be an integer")
    if version != 1:
        raise RunConfigError(f"unsupported configuration version {version}; expected 1")

    task = payload.get("task", "generate")
    if not isinstance(task, str):
        raise RunConfigError("task must be a string")
    if task != "generate":
        raise RunConfigError("task must be 'generate' in configuration version 1")

    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise RunConfigError("model must be a non-empty string")

    selected_inputs = [
        key for key in ("prompt", "prompts", "prompts_file") if key in payload
    ]
    if len(selected_inputs) != 1:
        raise RunConfigError(
            "configuration must define exactly one of: prompt, prompts, prompts_file"
        )

    prompt: str | None = None
    prompts: tuple[str, ...] = ()
    prompts_file: Path | None = None
    if selected_inputs[0] == "prompt":
        value = payload["prompt"]
        if not isinstance(value, str) or not value.strip():
            raise RunConfigError("prompt must be a non-empty string")
        prompt = value
    elif selected_inputs[0] == "prompts":
        value = payload["prompts"]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise RunConfigError("prompts must be a non-empty list of strings")
        normalized: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise RunConfigError(f"prompts[{index}] must be a non-empty string")
            normalized.append(item)
        if not normalized:
            raise RunConfigError("prompts must contain at least one string")
        prompts = tuple(normalized)
    else:
        value = payload["prompts_file"]
        if not isinstance(value, str) or not value.strip():
            raise RunConfigError("prompts_file must be a non-empty path string")
        prompts_file = _resolve_relative_path(value, config_path)

    return InferenceRunConfig(
        version=version,
        task=task,
        model=model,
        engine=_parse_engine(payload.get("engine", {})),
        generation=_parse_generation(payload.get("generation", {})),
        prompt=prompt,
        prompts=prompts,
        prompts_file=prompts_file,
        output=_parse_output(payload.get("output", {}), config_path),
        source_path=config_path,
    )


def load_prompts(config: InferenceRunConfig) -> tuple[str, ...]:
    """Resolve the configured inline prompt(s) or newline-delimited prompt file."""

    if config.prompt is not None:
        return (config.prompt,)
    if config.prompts:
        return config.prompts
    if config.prompts_file is None:
        raise RunConfigError("configuration has no prompt input")
    try:
        lines = config.prompts_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RunConfigError(
            f"cannot read prompts_file {config.prompts_file}: {exc}"
        ) from exc
    prompts = tuple(line.strip() for line in lines if line.strip())
    if not prompts:
        raise RunConfigError(f"prompts_file {config.prompts_file} contains no prompts")
    return prompts


def _validate_execution(config: InferenceRunConfig) -> None:
    generation = config.generation
    if generation.logit_lens and not generation.xai:
        raise RunConfigError("generation.logit_lens requires generation.xai: true")
    if config.is_batch:
        if "repetition_penalty" in generation.explicit_fields:
            raise RunConfigError(
                "generation.repetition_penalty is not supported by generate_batch()"
            )
        if generation.xai or generation.logit_lens:
            raise RunConfigError(
                "generation.xai and generation.logit_lens are available only with a single prompt"
            )
    elif generation.ignore_eos:
        raise RunConfigError(
            "generation.ignore_eos is available only with prompts/prompts_file batch input"
        )


def validate_run_config(config: InferenceRunConfig) -> tuple[str, ...]:
    """Validate cross-field execution rules and resolve configured prompts."""

    _validate_execution(config)
    return load_prompts(config)


def normalized_config_dict(
    config: InferenceRunConfig, *, prompt_count: int | None = None
) -> dict[str, Any]:
    """Return a JSON-serializable normalized view used by ``--dry-run``."""

    engine = config.engine
    generation = config.generation
    output = config.output
    result: dict[str, Any] = {
        "version": config.version,
        "task": config.task,
        "model": config.model,
        "engine": {
            "device": engine.device,
            "dtype": engine.dtype,
            "num_blocks": engine.num_blocks,
            "block_size": engine.block_size,
            "max_batch_size": engine.max_batch_size,
            "cache_dir": engine.cache_dir,
            "n_gpu_layers": engine.n_gpu_layers,
            "offload_dir": engine.offload_dir,
            "quantize": engine.quantize,
            "kv_offload": engine.kv_offload,
            "num_cpu_blocks": engine.num_cpu_blocks,
            "gpu_window": engine.gpu_window,
            "kv_alloc": engine.kv_alloc,
            "max_seq_len": engine.max_seq_len,
            "monitor": engine.monitor,
            "dashboard": engine.dashboard,
            "dashboard_port": engine.dashboard_port,
            "deterministic": engine.deterministic,
            "seed": engine.seed,
            "mgx_verify_payload": engine.mgx_verify_payload,
            "mgx_prefer_payload_cache": engine.mgx_prefer_payload_cache,
            "mgx_payload_cache_dir": engine.mgx_payload_cache_dir,
        },
        "generation": {
            "max_new_tokens": generation.max_new_tokens,
            "temperature": generation.temperature,
            "top_k": generation.top_k,
            "top_p": generation.top_p,
            "repetition_penalty": (
                None if config.is_batch else generation.repetition_penalty
            ),
            "stop_token_ids": list(generation.stop_token_ids),
            "ignore_eos": generation.ignore_eos,
            "verbose": generation.verbose,
            "xai": generation.xai,
            "xai_top_k": generation.xai_top_k,
            "logit_lens": generation.logit_lens,
        },
        "input": {
            "mode": "batch" if config.is_batch else "single",
            "api": "generate_batch" if config.is_batch else "generate",
            "prompt_count": prompt_count,
            "prompts_file": str(config.prompts_file) if config.prompts_file else None,
        },
        "output": {
            "format": output.format,
            "path": str(output.path) if output.path else None,
            "include_prompt": output.include_prompt,
        },
        "source_path": str(config.source_path),
    }
    return result


def _dtype_value(dtype: str):
    import torch

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]


def _engine_kwargs(config: EngineRunConfig) -> dict[str, Any]:
    return {
        "dtype": _dtype_value(config.dtype),
        "device": config.device,
        "num_blocks": config.num_blocks,
        "block_size": config.block_size,
        "max_batch_size": config.max_batch_size,
        "cache_dir": config.cache_dir,
        "n_gpu_layers": config.n_gpu_layers,
        "offload_dir": config.offload_dir,
        "quantize": config.quantize,
        "kv_offload": config.kv_offload,
        "num_cpu_blocks": config.num_cpu_blocks,
        "gpu_window": config.gpu_window,
        "kv_alloc": config.kv_alloc,
        "max_seq_len": config.max_seq_len,
        "monitor": config.monitor,
        "dashboard": config.dashboard,
        "dashboard_port": config.dashboard_port,
        "deterministic": config.deterministic,
        "seed": config.seed,
        "mgx_verify_payload": config.mgx_verify_payload,
        "mgx_prefer_payload_cache": config.mgx_prefer_payload_cache,
        "mgx_payload_cache_dir": config.mgx_payload_cache_dir,
    }


def create_inference_engine(
    model: str,
    config: EngineRunConfig,
    *,
    engine_factory: Callable[..., Any] | None = None,
):
    """Construct an inference engine from the shared typed engine configuration."""

    if engine_factory is None:
        from megagemm.engine import InferenceEngine

        engine_factory = InferenceEngine
    return engine_factory(model, **_engine_kwargs(config))


def _xai_payload(report: Any) -> Any:
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    summary = getattr(report, "summary", None)
    if callable(summary):
        return {"summary": summary()}
    return {"summary": str(report)}


def _result_rows(
    config: InferenceRunConfig,
    prompts: tuple[str, ...],
    outputs: tuple[str, ...],
    xai_payloads: tuple[Any | None, ...],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for index, (prompt, output, xai) in enumerate(
        zip(prompts, outputs, xai_payloads)
    ):
        row: dict[str, Any] = {
            "index": index,
            "model": config.model,
            "output": output,
        }
        if config.output.include_prompt:
            row["prompt"] = prompt
        if xai is not None:
            row["xai"] = xai
        rows.append(row)
    return tuple(rows)


def _render_output(
    config: InferenceRunConfig,
    rows: tuple[dict[str, Any], ...],
    output_format: str,
) -> str:
    if output_format == "jsonl":
        return "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
        )
    if output_format == "json":
        payload = {
            "version": config.version,
            "task": config.task,
            "model": config.model,
            "outputs": list(rows),
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    chunks: list[str] = []
    for row in rows:
        if len(rows) > 1:
            chunks.append(f"[{row['index']}]")
        if "prompt" in row:
            chunks.append(f"Prompt: {row['prompt']}")
        chunks.append(str(row["output"]))
        if "xai" in row:
            chunks.append(json.dumps(row["xai"], indent=2, ensure_ascii=False))
        if len(rows) > 1:
            chunks.append("")
    return "\n".join(chunks).rstrip()


def _override_output_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "-":
        return None
    return Path(path).expanduser().resolve()


def execute_run_config(
    config_or_path: InferenceRunConfig | str | Path,
    *,
    engine_factory: Callable[..., Any] | None = None,
    output_format: str | None = None,
    output_path: str | Path | None = None,
    stdout: TextIO | None = None,
) -> InferenceRunResult:
    """Execute a validated inference configuration.

    ``engine_factory`` exists primarily for embedding-free unit tests and custom
    integrations. The default imports :class:`InferenceEngine` lazily.
    """

    config = (
        config_or_path
        if isinstance(config_or_path, InferenceRunConfig)
        else load_run_config(config_or_path)
    )
    prompts = validate_run_config(config)

    selected_format = output_format or config.output.format
    if selected_format not in {"text", "json", "jsonl"}:
        raise RunConfigError("output format must be one of: text, json, jsonl")
    selected_path = (
        _override_output_path(output_path)
        if output_path is not None
        else config.output.path
    )

    engine = create_inference_engine(
        config.model,
        config.engine,
        engine_factory=engine_factory,
    )

    generation = config.generation
    xai_payloads: tuple[Any | None, ...]
    if config.is_batch:
        generated = engine.generate_batch(
            list(prompts),
            max_new_tokens=generation.max_new_tokens,
            temperature=generation.temperature,
            top_k=generation.top_k,
            top_p=generation.top_p,
            stop_token_ids=list(generation.stop_token_ids) or None,
            ignore_eos=generation.ignore_eos,
            verbose=generation.verbose,
        )
        if not isinstance(generated, Sequence) or isinstance(generated, (str, bytes)):
            raise RuntimeError("generate_batch() did not return a sequence")
        outputs = tuple(str(item) for item in generated)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"generate_batch() returned {len(outputs)} outputs for {len(prompts)} prompts"
            )
        xai_payloads = tuple(None for _ in outputs)
    else:
        generated = engine.generate(
            prompts[0],
            max_new_tokens=generation.max_new_tokens,
            temperature=generation.temperature,
            top_k=generation.top_k,
            top_p=generation.top_p,
            repetition_penalty=generation.repetition_penalty,
            stop_token_ids=list(generation.stop_token_ids) or None,
            verbose=generation.verbose,
            xai=generation.xai,
            xai_top_k=generation.xai_top_k,
            logit_lens=generation.logit_lens,
        )
        if generation.xai:
            if not isinstance(generated, tuple) or len(generated) != 2:
                raise RuntimeError("generate(xai=True) did not return (text, report)")
            text, report = generated
            outputs = (str(text),)
            xai_payloads = (_xai_payload(report),)
        else:
            outputs = (str(generated),)
            xai_payloads = (None,)

    rows = _result_rows(config, prompts, outputs, xai_payloads)
    rendered = _render_output(config, rows, selected_format)
    if selected_path is None:
        stream = stdout or sys.stdout
        stream.write(rendered)
        stream.write("\n")
    else:
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(rendered + "\n", encoding="utf-8")

    return InferenceRunResult(
        model=config.model,
        prompts=prompts,
        outputs=outputs,
        rows=rows,
        output_format=selected_format,
        output_path=selected_path,
    )


__all__ = [
    "EngineRunConfig",
    "GenerationRunConfig",
    "InferenceRunConfig",
    "InferenceRunResult",
    "OutputRunConfig",
    "RunConfigError",
    "create_inference_engine",
    "execute_run_config",
    "load_prompts",
    "load_run_config",
    "normalized_config_dict",
    "validate_run_config",
]

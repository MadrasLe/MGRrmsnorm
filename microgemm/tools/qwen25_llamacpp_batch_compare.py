#!/usr/bin/env python3
"""Focused llama.cpp batch benchmark for the Qwen2.5 MicroGemm CPU suite.

This uses llama.cpp's llama-batched-bench tool, which measures batched prompt
processing and batched token generation for parallel sequences.  For the
MicroGemm suite comparison, the useful knobs are:

  - -npp: prompt tokens per parallel request
  - -ntg: generated tokens per parallel request
  - -npl: number of parallel requests, i.e. batch size

Typical Colab usage:

    !python microgemm/tools/qwen25_llamacpp_batch_compare.py \
        --batch-sizes 2,4,8 \
        --prompt-tokens 256 \
        --max-new-tokens 128 \
        --threads 8 \
        --microgemm-csv bench_results/qwen25_cpu_microgemm/qwen25_05b_cpu_xeon_v62_restore_logits_all_batches_qwen25_cpu_suite_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_BUILD_TAG = "qwen25_llamacpp_batch_compare_v9_auto_gguf_quant"
DEFAULT_GGUF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
DEFAULT_GGUF_FILE = "qwen2.5-0.5b-instruct-q8_0.gguf"
DEFAULT_LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
SPLIT_GGUF_RE = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


def parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def resolve_path(raw: str, cwd: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def ensure_executable(path: Path) -> Path:
    if os.name != "nt" and path.exists():
        try:
            path.chmod(path.stat().st_mode | 0o111)
        except OSError:
            pass
    return path


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def stats(values: list[float]) -> dict[str, float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": min(clean),
        "median": statistics.median(clean),
        "mean": statistics.mean(clean),
        "p95": percentile(clean, 0.95),
        "max": max(clean),
    }


def run_checked(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    if cmd:
        maybe_exe = Path(cmd[0])
        if maybe_exe.exists():
            ensure_executable(maybe_exe)
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"cwd: {cwd or Path.cwd()}\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
    return result


def maybe_install_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        run_checked([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])


def split_gguf_prefix(filename: str) -> str:
    name = Path(filename).name
    match = SPLIT_GGUF_RE.match(name)
    if match:
        return match.group("prefix")
    if name.endswith(".gguf"):
        return name[: -len(".gguf")]
    return name


def find_split_gguf_files(repo_files: list[str], requested_file: str) -> list[str]:
    prefix = split_gguf_prefix(requested_file)
    parts: dict[int, str] = {}
    expected_total = 0
    for filename in repo_files:
        name = Path(filename).name
        match = SPLIT_GGUF_RE.match(name)
        if not match or match.group("prefix") != prefix:
            continue
        idx = int(match.group("idx"))
        total = int(match.group("total"))
        if expected_total and total != expected_total:
            continue
        expected_total = total
        parts[idx] = filename

    if not parts:
        return []
    missing = [idx for idx in range(1, expected_total + 1) if idx not in parts]
    if missing:
        missing_text = ",".join(str(idx) for idx in missing[:8])
        raise SystemExit(
            f"Found split GGUF shards for {prefix}, but the set is incomplete "
            f"(missing shard index(es): {missing_text})."
        )
    return [parts[idx] for idx in sorted(parts)]


def format_gguf_file_hint(repo_files: list[str], limit: int = 16) -> str:
    ggufs = [filename for filename in repo_files if filename.lower().endswith(".gguf")]
    if not ggufs:
        return "No .gguf files were listed in the repo."
    shown = "\n".join(f"  - {filename}" for filename in ggufs[:limit])
    extra = "" if len(ggufs) <= limit else f"\n  ... and {len(ggufs) - limit} more"
    return f"Available GGUF files include:\n{shown}{extra}"


def choose_auto_gguf_file(repo_files: list[str], quant: str) -> str:
    quant_l = quant.lower()
    ggufs = [filename for filename in repo_files if filename.lower().endswith(".gguf")]
    candidates = [filename for filename in ggufs if quant_l in filename.lower()]
    if not candidates:
        hint = format_gguf_file_hint(repo_files)
        raise SystemExit(f"--gguf-file auto could not find a {quant} GGUF in the repo.\n{hint}")

    def score(filename: str) -> tuple[int, int, str]:
        name = Path(filename).name.lower()
        split = SPLIT_GGUF_RE.match(Path(filename).name) is not None
        first_split = "-00001-of-" in name
        exact_suffix = name.endswith(f"-{quant_l}.gguf") or name.endswith(f"_{quant_l}.gguf")
        return (
            0 if exact_suffix else 1,
            0 if (not split or first_split) else 1,
            name,
        )

    return sorted(candidates, key=score)[0]


def reject_placeholder_gguf_path(path: Path) -> None:
    upper = path.name.upper()
    if upper.startswith("SEU_") or "YOUR_" in upper or "PLACEHOLDER" in upper:
        raise SystemExit(
            f"--gguf-path is a placeholder, not a real file: {path}\n"
            "Use --gguf-repo plus --gguf-file auto, or pass an existing .gguf path."
        )


def ensure_gguf(args: argparse.Namespace, cwd: Path) -> Path:
    if args.gguf_path:
        gguf_path = resolve_path(args.gguf_path, cwd)
        reject_placeholder_gguf_path(gguf_path)
        if not gguf_path.exists():
            raise SystemExit(f"--gguf-path not found: {gguf_path}")
        print(f"[gguf] using existing file: {gguf_path}", flush=True)
        return gguf_path

    maybe_install_huggingface_hub()
    from huggingface_hub import hf_hub_download, list_repo_files

    cache_dir = resolve_path(args.gguf_cache_dir, cwd)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.gguf_file.lower() == "auto":
        print(f"[gguf] listing {args.gguf_repo} to auto-select {args.gguf_quant}", flush=True)
        repo_files = list_repo_files(args.gguf_repo)
        args.gguf_file = choose_auto_gguf_file(repo_files, args.gguf_quant)
        print(f"[gguf] auto-selected: {args.gguf_file}", flush=True)
    requested_is_split = SPLIT_GGUF_RE.match(Path(args.gguf_file).name) is not None
    direct_error: Exception | None = None

    if not requested_is_split:
        print(f"[gguf] downloading {args.gguf_repo}/{args.gguf_file}", flush=True)
        try:
            path = hf_hub_download(
                repo_id=args.gguf_repo,
                filename=args.gguf_file,
                local_dir=str(cache_dir),
            )
            gguf_path = Path(path).resolve()
            print(f"[gguf] ready: {gguf_path}", flush=True)
            return gguf_path
        except Exception as exc:  # noqa: BLE001 - fallback handles split GGUF repos cleanly.
            direct_error = exc
            print("[gguf] single-file download failed; checking for split GGUF shards", flush=True)
    else:
        print(f"[gguf] requested split GGUF shard {args.gguf_file}; downloading full shard set", flush=True)

    try:
        repo_files = list_repo_files(args.gguf_repo)
    except Exception as exc:  # noqa: BLE001
        if direct_error is not None:
            raise RuntimeError(
                f"Could not download {args.gguf_repo}/{args.gguf_file}, "
                "and listing the repo for split GGUF shards also failed."
            ) from direct_error
        raise RuntimeError(f"Could not list GGUF repo files for {args.gguf_repo}") from exc

    split_files = find_split_gguf_files(repo_files, args.gguf_file)
    if not split_files:
        hint = format_gguf_file_hint(repo_files)
        if direct_error is not None:
            raise RuntimeError(
                f"Could not download {args.gguf_repo}/{args.gguf_file}, and no matching split GGUF "
                f"shards were found for prefix '{split_gguf_prefix(args.gguf_file)}'.\n{hint}"
            ) from direct_error
        raise SystemExit(
            f"No matching split GGUF shards were found for prefix '{split_gguf_prefix(args.gguf_file)}'.\n{hint}"
        )

    print(f"[gguf] downloading {len(split_files)} split shard(s) for {split_gguf_prefix(args.gguf_file)}", flush=True)
    first_path: Path | None = None
    for filename in split_files:
        print(f"[gguf] shard: {filename}", flush=True)
        path = hf_hub_download(
            repo_id=args.gguf_repo,
            filename=filename,
            local_dir=str(cache_dir),
        )
        if first_path is None:
            first_path = Path(path).resolve()

    if first_path is None:
        raise SystemExit("Split GGUF shard discovery returned no downloadable files")
    print(f"[gguf] ready split: {first_path} ({len(split_files)} shard(s))", flush=True)
    return first_path


def find_batched_bench(raw: str, llama_dir: Path, cwd: Path) -> Path | None:
    candidates: list[Path] = []
    if raw:
        candidates.append(resolve_path(raw, cwd))
    which = shutil.which("llama-batched-bench")
    if which:
        candidates.append(Path(which).resolve())
    candidates.extend(
        [
            cwd / "llama-batched-bench",
            cwd / ".cache" / "llama.cpp" / "build" / "bin" / "llama-batched-bench",
            cwd / "llama.cpp" / "build" / "bin" / "llama-batched-bench",
            llama_dir / "build" / "bin" / "llama-batched-bench",
            llama_dir / "build" / "tools" / "batched-bench" / "llama-batched-bench",
            llama_dir / "build" / "examples" / "batched-bench" / "llama-batched-bench",
        ]
    )
    if os.name == "nt":
        candidates.extend([Path(str(path) + ".exe") for path in list(candidates) if path.suffix != ".exe"])
    for path in candidates:
        if path.exists():
            return ensure_executable(path.resolve())
    return None


def cmake_target_exists(llama_dir: Path, target: str) -> bool:
    result = subprocess.run(
        ["cmake", "--build", "build", "--target", "help"],
        cwd=str(llama_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = f"{result.stdout}\n{result.stderr}"
    return target in text


def build_llama_cpp(args: argparse.Namespace, cwd: Path) -> Path:
    llama_dir = resolve_path(args.llama_dir, cwd)
    llama_dir.parent.mkdir(parents=True, exist_ok=True)

    existing = find_batched_bench(args.llama_batched_bench_bin, llama_dir, cwd)
    if existing is not None:
        print(f"[llama.cpp] using existing llama-batched-bench: {existing}", flush=True)
        return existing

    if not llama_dir.exists():
        print(f"[llama.cpp] cloning {args.llama_repo} -> {llama_dir}", flush=True)
        run_checked(["git", "clone", "--depth=1", args.llama_repo, str(llama_dir)])

    print("[llama.cpp] configuring CPU-only build", flush=True)
    run_checked(
        [
            "cmake",
            "-B",
            "build",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=ON",
            "-DLLAMA_BUILD_TOOLS=ON",
        ],
        cwd=llama_dir,
    )

    target = "llama-batched-bench"
    if not cmake_target_exists(llama_dir, target):
        print("[llama.cpp] target help did not list llama-batched-bench; building all examples/tools", flush=True)
        target_args: list[str] = []
    else:
        target_args = ["--target", target]

    jobs = str(max(1, int(args.build_jobs or min(4, os.cpu_count() or 2))))
    print(f"[llama.cpp] building {target or 'all'} with -j{jobs}", flush=True)
    run_checked(
        ["cmake", "--build", "build", "--config", "Release", *target_args, "-j", jobs],
        cwd=llama_dir,
    )

    built = find_batched_bench("", llama_dir, cwd)
    if built is None:
        raise SystemExit("llama.cpp build finished but llama-batched-bench was not found")
    print(f"[llama.cpp] llama-batched-bench ready: {built}", flush=True)
    return built


def platform_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count() or 0,
    }


def llama_help_text(bench_bin: Path) -> str:
    result = subprocess.run(
        [str(bench_bin), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return f"{result.stdout}\n{result.stderr}"


def help_supports(args: argparse.Namespace, *flags: str) -> bool:
    text = str(getattr(args, "llama_help_text", "") or "")
    return any(flag in text for flag in flags)


def parse_jsonl_records(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if {"pp", "tg", "pl"}.issubset(obj):
            rows.append(obj)
    return rows


def parse_markdown_records(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "----" in stripped or "PP" in stripped:
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) != 10:
            continue
        try:
            rows.append(
                {
                    "pp": int(parts[0]),
                    "tg": int(parts[1]),
                    "pl": int(parts[2]),
                    "n_kv": int(parts[3]),
                    "t_pp": float(parts[4]),
                    "speed_pp": float(parts[5]),
                    "t_tg": float(parts[6]),
                    "speed_tg": float(parts[7]),
                    "t": float(parts[8]),
                    "speed": float(parts[9]),
                }
            )
        except ValueError:
            continue
    return rows


def run_batched_bench_once(
    bench_bin: Path,
    gguf_path: Path,
    args: argparse.Namespace,
    *,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], str, str]:
    batch_sizes = parse_csv_ints(args.batch_sizes)
    ctx_size = args.ctx_size
    if ctx_size <= 0:
        ctx_size = max(batch_sizes) * (args.prompt_tokens + max_new_tokens) + 64
    llama_batch = int(args.llama_batch_size or max(512, max(batch_sizes) * args.prompt_tokens))
    llama_ubatch = int(args.llama_ubatch_size or min(512, llama_batch))

    cmd = [
        str(bench_bin),
        "-m",
        str(gguf_path),
        "-c",
        str(ctx_size),
        "-b",
        str(llama_batch),
        "-npp",
        str(args.prompt_tokens),
        "-ntg",
        str(max_new_tokens),
        "-npl",
        ",".join(str(v) for v in batch_sizes),
    ]
    if help_supports(args, "-ngl", "--gpu-layers"):
        cmd.extend(["-ngl", "0"])
    if help_supports(args, "-ub", "--ubatch-size"):
        cmd.extend(["-ub", str(llama_ubatch)])
    if help_supports(args, "--output-format"):
        cmd.extend(["--output-format", "jsonl"])
    if args.threads > 0:
        cmd.extend(["-t", str(args.threads)])
        if help_supports(args, "-tb", "--threads-batch"):
            cmd.extend(["-tb", str(args.threads_batch or args.threads)])
    if args.flash_attn and help_supports(args, "-fa", "--flash-attn"):
        cmd.append("-fa")
    if args.prompt_shared and help_supports(args, "-pps", "--prompt-shared"):
        cmd.append("-pps")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = f"{result.stdout}\n{result.stderr}"
    rows = parse_jsonl_records(combined)

    if result.returncode != 0 or not rows:
        # Older builds may not support --output-format. Retry markdown mode.
        fallback = [part for part in cmd if part not in {"--output-format", "jsonl"}]
        result = run_checked(fallback)
        combined = f"{result.stdout}\n{result.stderr}"
        rows = parse_markdown_records(combined)

    if not rows:
        raise RuntimeError(
            "llama-batched-bench produced no parseable rows\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )
    return rows, result.stdout, result.stderr


def normalize_record(record: dict[str, Any], run_index: int, args: argparse.Namespace) -> dict[str, Any]:
    batch_size = int(record.get("pl", 0) or 0)
    prompt_tokens = int(record.get("pp", 0) or 0)
    max_new_tokens = int(record.get("tg", 0) or 0)
    t_pp = float(record.get("t_pp", 0.0) or 0.0)
    t_tg = float(record.get("t_tg", 0.0) or 0.0)
    t_total = float(record.get("t", t_pp + t_tg) or 0.0)
    speed_pp = float(record.get("speed_pp", 0.0) or 0.0)
    speed_tg = float(record.get("speed_tg", 0.0) or 0.0)
    speed_total = float(record.get("speed", 0.0) or 0.0)
    generated_tokens = batch_size * max_new_tokens
    prefill_tokens = batch_size * prompt_tokens
    output_tps_total = generated_tokens / t_total if t_total > 0.0 else 0.0
    output_tps_decode = generated_tokens / t_tg if t_tg > 0.0 else 0.0
    return {
        "ok": True,
        "backend": "llamacpp",
        "run_index": run_index,
        "batch_size": batch_size,
        "target_prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": generated_tokens,
        "prefill_tokens": prefill_tokens,
        "n_kv": int(record.get("n_kv", batch_size * (prompt_tokens + max_new_tokens)) or 0),
        "prefill_s": t_pp,
        "decode_s": t_tg,
        "total_s": t_total,
        "prefill_tps": speed_pp,
        "decode_tps": speed_tg,
        "output_tps_total": output_tps_total,
        "output_tps_decode": output_tps_decode,
        "combined_tps": speed_total,
        "threads": args.threads,
        "threads_batch": args.threads_batch or args.threads,
        "prompt_shared": bool(args.prompt_shared),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ok": len([row for row in rows if row.get("ok")]),
        "runs": len(rows),
        "prefill_tps": stats([float(row.get("prefill_tps", 0.0) or 0.0) for row in rows]),
        "decode_tps": stats([float(row.get("decode_tps", 0.0) or 0.0) for row in rows]),
        "output_tps_total": stats([float(row.get("output_tps_total", 0.0) or 0.0) for row in rows]),
        "output_tps_decode": stats([float(row.get("output_tps_decode", 0.0) or 0.0) for row in rows]),
        "combined_tps": stats([float(row.get("combined_tps", 0.0) or 0.0) for row in rows]),
        "prefill_s": stats([float(row.get("prefill_s", 0.0) or 0.0) for row in rows]),
        "decode_s": stats([float(row.get("decode_s", 0.0) or 0.0) for row in rows]),
        "total_s": stats([float(row.get("total_s", 0.0) or 0.0) for row in rows]),
    }


def find_microgemm_csv(raw: str) -> Path | None:
    if raw and raw.lower() != "auto":
        path = Path(raw)
        if path.exists():
            return path
        basename_matches = sorted(Path(".").glob(f"**/{path.name}"), key=lambda p: p.stat().st_mtime, reverse=True)
        if basename_matches:
            selected = basename_matches[0]
            print(f"[microgemm] using comparison CSV by basename: {selected}", flush=True)
            return selected
        print(f"Warning: MicroGemm CSV not found: {path}")

    search_dir = Path("bench_results/qwen25_cpu_microgemm")
    if not search_dir.exists():
        return None

    patterns = [
        "*v71_gateup_tile2_dualacc*_summary.csv",
        "*v70_batch2_rowpair_gemv*_summary.csv",
        "*v69_canary_decode_only_print*_summary.csv",
        "*v68_decode_only_metric*_summary.csv",
        "*v67_batch_wall_overhead_split*_summary.csv",
        "*v62_restore_logits_all_batches*_summary.csv",
        "*v66_restore_v62_scalar_sampler*_summary.csv",
        "*v63_sampler_avx2_argmax*_summary.csv",
        "*qwen25_cpu_suite_summary.csv",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(search_dir.glob(pattern))
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    selected = candidates[0]
    print(f"[microgemm] using comparison CSV: {selected}", flush=True)
    return selected


def load_microgemm_csv(path: str, mode_selector: str = "continuous") -> dict[int, dict[str, float]]:
    if not path:
        csv_path = find_microgemm_csv("auto")
    else:
        csv_path = find_microgemm_csv(path)
    if csv_path is None:
        print("Warning: no MicroGemm CSV found for side-by-side comparison")
        return {}
    out: dict[int, dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("kind") != "batch":
                continue
            mode = row.get("effective_mode") or row.get("mode")
            requested = mode_selector.strip().lower()
            if requested not in {"best", "any"} and mode != requested and row.get("mode") != requested:
                continue
            try:
                batch_size = int(float(row.get("batch_size", "0") or 0))
            except ValueError:
                continue
            runtime_tps = float(row.get("runtime_output_tps_median", "0") or 0.0)
            candidate = {
                "mode": str(mode or row.get("mode") or ""),
                "wall_tps": float(row.get("wall_output_tps_median", "0") or 0.0),
                "steady_tps": float(row.get("steady_output_tps_median", "") or runtime_tps),
                "runtime_tps": runtime_tps,
                "decode_only_tps": float(row.get("decode_only_output_tps_median", "0") or 0.0),
                "harness_overhead_ms": float(row.get("harness_overhead_ms_median", "0") or 0.0),
            }
            if requested in {"best", "any"}:
                candidate_score = candidate["decode_only_tps"] or candidate["runtime_tps"] or candidate["wall_tps"]
                previous = out.get(batch_size)
                previous_score = 0.0 if previous is None else (
                    previous.get("decode_only_tps", 0.0)
                    or previous.get("runtime_tps", 0.0)
                    or previous.get("wall_tps", 0.0)
                )
                if previous is None or candidate_score > previous_score:
                    out[batch_size] = candidate
            else:
                out[batch_size] = candidate
    return out


def build_summary_rows(
    all_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    gguf_path: Path,
    microgemm: dict[int, dict[str, float]],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for batch_size in parse_csv_ints(args.batch_sizes):
        rows = [row for row in all_rows if int(row.get("batch_size", 0) or 0) == batch_size]
        if not rows:
            continue
        agg = aggregate(rows)
        summary: dict[str, Any] = {
            "backend": "llamacpp",
            "script_build": SCRIPT_BUILD_TAG,
            "gguf_path": str(gguf_path),
            "gguf_file": gguf_path.name,
            "batch_size": batch_size,
            "target_prompt_tokens": args.prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "threads": args.threads,
            "threads_batch": args.threads_batch or args.threads,
            "ctx_size": args.ctx_size,
            "llama_batch_size": args.llama_batch_size,
            "llama_ubatch_size": args.llama_ubatch_size,
            "prompt_shared": bool(args.prompt_shared),
            "ok": agg["ok"],
            "runs": agg["runs"],
            "prefill_tps_median": agg["prefill_tps"]["median"],
            "decode_tps_median": agg["decode_tps"]["median"],
            "output_tps_total_median": agg["output_tps_total"]["median"],
            "output_tps_decode_median": agg["output_tps_decode"]["median"],
            "combined_tps_median": agg["combined_tps"]["median"],
            "prefill_s_median": agg["prefill_s"]["median"],
            "decode_s_median": agg["decode_s"]["median"],
            "total_s_median": agg["total_s"]["median"],
            "decode_tps_min": agg["decode_tps"]["min"],
            "decode_tps_mean": agg["decode_tps"]["mean"],
            "decode_tps_p95": agg["decode_tps"]["p95"],
            "decode_tps_max": agg["decode_tps"]["max"],
        }
        if batch_size in microgemm:
            mg = microgemm[batch_size]
            micro_wall = float(mg.get("wall_tps", 0.0) or 0.0)
            micro_steady = float(mg.get("steady_tps", 0.0) or 0.0)
            micro_runtime = float(mg.get("runtime_tps", 0.0) or 0.0)
            micro_decode_only = float(mg.get("decode_only_tps", 0.0) or 0.0)
            llama_output_total = float(summary["output_tps_total_median"])
            llama_decode = float(summary["decode_tps_median"])
            summary["microgemm_wall_tps_median"] = micro_wall
            summary["microgemm_steady_tps_median"] = micro_steady
            summary["microgemm_runtime_tps_median"] = micro_runtime
            summary["microgemm_decode_only_tps_median"] = micro_decode_only
            summary["microgemm_mode"] = str(mg.get("mode", ""))
            summary["microgemm_harness_overhead_ms_median"] = float(mg.get("harness_overhead_ms", 0.0) or 0.0)
            summary["llamacpp_output_total_over_microgemm_wall"] = (
                llama_output_total / micro_wall if micro_wall else 0.0
            )
            summary["llamacpp_output_total_over_microgemm_steady"] = (
                llama_output_total / micro_steady if micro_steady else 0.0
            )
            summary["llamacpp_output_total_over_microgemm_runtime"] = (
                llama_output_total / micro_runtime if micro_runtime else 0.0
            )
            summary["llamacpp_decode_only_over_microgemm_decode_only"] = (
                llama_decode / micro_decode_only if micro_decode_only else 0.0
            )
            summary["llamacpp_decode_only_over_microgemm_wall"] = llama_decode / micro_wall if micro_wall else 0.0
        summary_rows.append(summary)
    return summary_rows


def print_comparison(summary_rows: list[dict[str, Any]], microgemm: dict[int, dict[str, float]]) -> None:
    print("llama.cpp batched-bench")
    for row in summary_rows:
        print(
            f"llama.cpp batch={int(row['batch_size'])} "
            f"ok={int(row['ok'])}/{int(row['runs'])} "
            f"output_total_med={float(row['output_tps_total_median']):.2f} tok/s "
            f"decode_only_med={float(row['decode_tps_median']):.2f} tok/s "
            f"prefill_med={float(row['prefill_tps_median']):.2f} tok/s "
            f"combined_all_tokens_med={float(row['combined_tps_median']):.2f} tok/s"
        )
    if not microgemm:
        return
    print("Comparison vs MicroGemm continuous")
    print(
        "  batch  micro_mode  llama_output_total  llama_decode_only  micro_wall  micro_steady  "
        "micro_runtime  micro_decode_only  output/micro_wall  decode/micro_decode"
    )
    for row in summary_rows:
        batch_size = int(row["batch_size"])
        mg = microgemm.get(batch_size)
        if not mg:
            continue
        llama_output_total = float(row["output_tps_total_median"])
        llama_decode = float(row["decode_tps_median"])
        micro_wall = float(mg.get("wall_tps", 0.0) or 0.0)
        micro_steady = float(mg.get("steady_tps", 0.0) or 0.0)
        micro_runtime = float(mg.get("runtime_tps", 0.0) or 0.0)
        micro_decode_only = float(mg.get("decode_only_tps", 0.0) or 0.0)
        wall_ratio = llama_output_total / micro_wall if micro_wall else 0.0
        decode_ratio = llama_decode / micro_decode_only if micro_decode_only else 0.0
        print(
            f"  {batch_size:>5}  "
            f"{str(mg.get('mode', '')):>10}  "
            f"{llama_output_total:>18.2f}  "
            f"{llama_decode:>17.2f}  "
            f"{micro_wall:>10.2f}  "
            f"{micro_steady:>12.2f}  "
            f"{micro_runtime:>13.2f}  "
            f"{micro_decode_only:>17.2f}  "
            f"{wall_ratio:>17.2f}x  "
            f"{decode_ratio:>19.2f}x"
        )


def write_outputs(
    args: argparse.Namespace,
    gguf_path: Path,
    bench_bin: Path,
    raw_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"qwen25_llamacpp_batch_compare_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{run_id}_qwen25_llamacpp_batch_compare.json"
    csv_path = out_dir / f"{run_id}_qwen25_llamacpp_batch_compare_summary.csv"

    payload = {
        "benchmark": "qwen25_llamacpp_batch_compare",
        "script_build": SCRIPT_BUILD_TAG,
        "run_id": run_id,
        "platform": platform_snapshot(),
        "config": {
            "gguf_repo": args.gguf_repo,
            "gguf_file": args.gguf_file,
            "gguf_path": str(gguf_path),
            "llama_batched_bench": str(bench_bin),
            "batch_sizes": parse_csv_ints(args.batch_sizes),
            "prompt_tokens": args.prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "threads": args.threads,
            "threads_batch": args.threads_batch or args.threads,
            "ctx_size": args.ctx_size,
            "llama_batch_size": args.llama_batch_size,
            "llama_ubatch_size": args.llama_ubatch_size,
            "prompt_shared": bool(args.prompt_shared),
            "microgemm_csv": args.microgemm_csv,
            "microgemm_mode": args.microgemm_mode,
        },
        "raw_rows": raw_rows,
        "summary_rows": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if summary_rows:
        fieldnames: list[str] = []
        for row in summary_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen2.5 llama.cpp batch 2/4/8 comparison harness")
    parser.add_argument("--gguf-path", default="")
    parser.add_argument("--gguf-repo", default=DEFAULT_GGUF_REPO)
    parser.add_argument("--gguf-file", default=DEFAULT_GGUF_FILE)
    parser.add_argument("--gguf-quant", choices=("q8_0", "q4_k_m"), default="q8_0", help="Quantization hint used when --gguf-file auto is set.")
    parser.add_argument("--gguf-cache-dir", default="/content/llamacpp_qwen25_cache/gguf")
    parser.add_argument("--llama-repo", default=DEFAULT_LLAMA_REPO)
    parser.add_argument("--llama-dir", default="/content/llamacpp_qwen25_cache/llama.cpp")
    parser.add_argument("--llama-batched-bench-bin", default="")
    parser.add_argument("--build-jobs", type=int, default=0)
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--threads-batch", type=int, default=0)
    parser.add_argument("--ctx-size", type=int, default=0)
    parser.add_argument("--llama-batch-size", type=int, default=2048)
    parser.add_argument("--llama-ubatch-size", type=int, default=512)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--prompt-shared", action="store_true")
    parser.add_argument("--microgemm-csv", default="")
    parser.add_argument("--microgemm-mode", default="continuous", help="MicroGemm mode to compare: continuous, adaptive, concurrent, serial, or best")
    parser.add_argument("--out-dir", default="bench_results/qwen25_llamacpp_compare")
    parser.add_argument("--run-id", default="qwen25_05b_llamacpp_q8_batch248")
    args = parser.parse_args()

    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.threads <= 0:
        args.threads = os.cpu_count() or 1
    if args.threads_batch <= 0:
        args.threads_batch = args.threads
    if args.ctx_size <= 0:
        max_batch = max(parse_csv_ints(args.batch_sizes))
        args.ctx_size = max_batch * (args.prompt_tokens + args.max_new_tokens) + 64

    cwd = Path.cwd()
    gguf_label = args.gguf_path if args.gguf_path else f"{args.gguf_repo}/{args.gguf_file}"
    print("Qwen2.5 llama.cpp batch compare")
    print(f"  script build:  {SCRIPT_BUILD_TAG}")
    print(f"  gguf:          {gguf_label}")
    print(f"  batch sizes:   {args.batch_sizes}")
    print(f"  prompt tokens: {args.prompt_tokens}")
    print(f"  max new tok:   {args.max_new_tokens}")
    print(f"  threads:       {args.threads}")
    print(f"  ctx size:      {args.ctx_size}")
    print(f"  llama batch:   {args.llama_batch_size}")
    print(f"  llama ubatch:  {args.llama_ubatch_size}")

    gguf_path = ensure_gguf(args, cwd)
    bench_bin = build_llama_cpp(args, cwd)
    args.llama_help_text = llama_help_text(bench_bin)
    microgemm_rows = load_microgemm_csv(args.microgemm_csv, args.microgemm_mode)

    if args.warmup:
        print(f"[bench] warmup {args.warmup}x", flush=True)
        for _ in range(args.warmup):
            run_batched_bench_once(
                bench_bin,
                gguf_path,
                args,
                max_new_tokens=min(args.max_new_tokens, 16),
            )

    raw_rows: list[dict[str, Any]] = []
    for run_idx in range(args.runs):
        print(f"[bench] run {run_idx + 1}/{args.runs}", flush=True)
        records, _, _ = run_batched_bench_once(
            bench_bin,
            gguf_path,
            args,
            max_new_tokens=args.max_new_tokens,
        )
        for record in records:
            if int(record.get("pp", 0) or 0) != args.prompt_tokens:
                continue
            if int(record.get("tg", 0) or 0) != args.max_new_tokens:
                continue
            if int(record.get("pl", 0) or 0) not in parse_csv_ints(args.batch_sizes):
                continue
            raw_rows.append(normalize_record(record, run_idx + 1, args))

    summary_rows = build_summary_rows(raw_rows, args, gguf_path, microgemm_rows)
    print_comparison(summary_rows, microgemm_rows)
    json_path, csv_path = write_outputs(args, gguf_path, bench_bin, raw_rows, summary_rows)
    print("Wrote:")
    print(f"  json: {json_path}")
    print(f"  csv:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

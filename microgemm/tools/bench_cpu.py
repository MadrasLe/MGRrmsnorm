"""
bench_cpu.py — Robust benchmark: MicroGemm vs llama.cpp (CPU)

Standalone harness that runs multiple iterations, computes statistics
(median / mean / stdev), and prints a clear comparison table.

Usage:

    python tools/bench_cpu.py \
      --model-dir /path/to/hf-model \
      --gguf-path /path/to/model.Q8_0.gguf \
      --prompt "Explain why the sky is blue." \
      --max-new-tokens 32 \
      --threads 2 \
      --runs 5

Colab one-liner (after build + convert):

    python tools/bench_cpu.py \
      --model-dir "$MODEL_DIR" \
      --gguf-path "$GGUF_PATH" \
      --llama-cli-bin .cache/llama.cpp/build/bin/llama-cli \
      --prompt "Explain quantum computing in one sentence." \
      --runs 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── llama.cpp stderr parsers ───────────────────────────────────────────

_PROMPT_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+tokens?",
    re.IGNORECASE,
)
_EVAL_RE = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+(?:runs?|tokens?)",
    re.IGNORECASE,
)
_TOTAL_RE = re.compile(r"total time\s*=\s*([\d.]+)\s*ms", re.IGNORECASE)


# ── helpers ────────────────────────────────────────────────────────────

def _resolve(raw: str, base: Path) -> Path:
    p = Path(raw)
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def _median(xs: List[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _mean(xs: List[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _stdev(xs: List[float]) -> float:
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


# ── MicroGemm runner ──────────────────────────────────────────────────

def _parse_mg_stdout(stdout: str) -> Dict[str, str]:
    """Parse key: value lines from microgemm-text stdout."""
    kv: Dict[str, str] = {}
    for line in stdout.splitlines():
        if ": " in line and not line.startswith(" "):
            key, _, value = line.partition(": ")
            kv[key.strip()] = value.strip()
    return kv


def _run_mg_once(
    text_bin: Path,
    mgm_path: Path,
    tokenizer_json: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    threads: int,
) -> Dict[str, float]:
    cmd = [
        str(text_bin), "generate",
        str(mgm_path), str(tokenizer_json),
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--temperature", str(temperature),
        "--top-k", str(top_k),
        "--top-p", str(top_p),
        "--seed", str(seed),
    ]
    env = os.environ.copy()
    if threads > 0:
        env["OMP_NUM_THREADS"] = str(threads)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    if result.returncode != 0:
        raise RuntimeError(
            f"microgemm-text failed (exit {result.returncode})\n"
            f"stderr: {result.stderr[:500]}"
        )

    kv = _parse_mg_stdout(result.stdout)

    prefill_ms = float(kv.get("prefill_ms", "0"))
    decode_ms = float(kv.get("decode_ms", "0"))
    total_ms = float(kv.get("total_ms", "0")) or wall_ms
    prompt_tokens = int(kv.get("prompt_token_count", "0"))
    gen_tokens = int(kv.get("generated_token_count", "0"))

    prefill_tps = (prompt_tokens / prefill_ms * 1000.0) if prefill_ms > 0 else 0.0
    decode_tps = (gen_tokens / decode_ms * 1000.0) if decode_ms > 0 and gen_tokens > 0 else 0.0

    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "text": kv.get("generated_text", result.stdout.strip()[:200]),
    }


# ── llama.cpp runner ──────────────────────────────────────────────────

def _run_llama_once(
    llama_cli: Path,
    gguf_path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    threads: int,
    ctx_size: int,
) -> Dict[str, float]:
    cmd = [
        str(llama_cli),
        "-m", str(gguf_path),
        "-p", prompt,
        "-n", str(max_new_tokens),
        "-ngl", "0",
        "--no-display-prompt",
        "--temp", str(temperature),
        "--top-k", str(top_k),
        "--top-p", str(top_p),
        "--seed", str(seed),
    ]
    if threads > 0:
        cmd.extend(["-t", str(threads)])
    if ctx_size > 0:
        cmd.extend(["-c", str(ctx_size)])

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    if result.returncode != 0:
        raise RuntimeError(
            f"llama-cli failed (exit {result.returncode})\n"
            f"stderr: {result.stderr[:500]}"
        )

    stderr = result.stderr

    prompt_match = _PROMPT_RE.search(stderr)
    prefill_ms = float(prompt_match.group(1)) if prompt_match else 0.0
    prompt_tokens = int(prompt_match.group(2)) if prompt_match else 0

    eval_match = _EVAL_RE.search(stderr)
    decode_ms = float(eval_match.group(1)) if eval_match else 0.0
    gen_tokens = int(eval_match.group(2)) if eval_match else 0

    total_match = _TOTAL_RE.search(stderr)
    total_ms = float(total_match.group(1)) if total_match else wall_ms

    prefill_tps = (prompt_tokens / prefill_ms * 1000.0) if prefill_ms > 0 else 0.0
    decode_tps = (gen_tokens / decode_ms * 1000.0) if decode_ms > 0 and gen_tokens > 0 else 0.0

    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "text": result.stdout.strip()[:200],
    }


# ── auto-build / auto-convert ─────────────────────────────────────────

def _ensure_mg_binaries(cwd: Path, text_bin: Path, convert_bin: Path) -> None:
    if text_bin.exists() and convert_bin.exists():
        return
    makefile = cwd / "Makefile"
    if makefile.exists() and shutil.which("make"):
        print("[setup] building MicroGemm…", flush=True)
        subprocess.run(["make"], cwd=str(cwd), check=True,
                        capture_output=True, text=True)
        if text_bin.exists() and convert_bin.exists():
            return
    raise SystemExit(
        f"MicroGemm binaries not found:\n"
        f"  text:    {text_bin}\n"
        f"  convert: {convert_bin}\n"
        f"Run `make` inside the microgemm/ folder first."
    )


def _ensure_mgm(mgm_path: Path, model_dir: Path, convert_bin: Path) -> None:
    if mgm_path.exists():
        return
    if not convert_bin.exists():
        raise SystemExit(f".mgm not found and converter missing: {convert_bin}")
    mgm_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[setup] converting model → {mgm_path}", flush=True)
    subprocess.run(
        [str(convert_bin), "from-dir", str(model_dir), str(mgm_path)],
        check=True, capture_output=True, text=True,
    )


def _find_llama_cli(hint: str, cwd: Path) -> Optional[Path]:
    if hint:
        p = _resolve(hint, cwd)
        return p if p.exists() else None
    for candidate in [
        cwd / "llama-cli",
        cwd / ".cache" / "llama.cpp" / "build" / "bin" / "llama-cli",
        cwd / "llama.cpp" / "build" / "bin" / "llama-cli",
    ]:
        if candidate.exists():
            return candidate.resolve()
    which = shutil.which("llama-cli")
    if which:
        return Path(which).resolve()
    return None


# ── statistics aggregation ─────────────────────────────────────────────

def _aggregate(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute median/mean/stdev for each numeric metric across runs."""
    if not runs:
        return {}
    keys = ["prefill_ms", "decode_ms", "total_ms", "prefill_tps", "decode_tps"]
    agg: Dict[str, Dict[str, float]] = {}
    for k in keys:
        values = [r[k] for r in runs if k in r]
        agg[k] = {
            "median": _median(values),
            "mean": _mean(values),
            "stdev": _stdev(values),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
    # Token counts from the first run (they should be identical)
    agg["prompt_tokens"] = {"median": runs[0].get("prompt_tokens", 0)}
    agg["gen_tokens"] = {"median": runs[0].get("gen_tokens", 0)}
    return agg


# ── display ────────────────────────────────────────────────────────────

def _fmt(v: float, suffix: str = "") -> str:
    if v == 0.0:
        return "—"
    return f"{v:.2f}{suffix}"


def _ratio(a: float, b: float) -> str:
    if b == 0.0 or a == 0.0:
        return "—"
    return f"{a / b:.2f}×"


def print_results(
    mg_agg: Dict[str, Dict[str, float]],
    llama_agg: Dict[str, Dict[str, float]],
    mg_text: str,
    llama_text: str,
    args: argparse.Namespace,
    runs: int,
) -> None:
    W = 86
    print()
    print("═" * W)
    print("  MicroGemm vs llama.cpp  ·  CPU Benchmark")
    print("═" * W)
    print(f"  prompt:    \"{args.prompt[:60]}{'…' if len(args.prompt)>60 else ''}\"")
    print(f"  tokens:    {int(mg_agg.get('prompt_tokens',{}).get('median',0))} prompt → "
          f"{int(mg_agg.get('gen_tokens',{}).get('median',0))} generated")
    print(f"  runs:      {runs}    threads: {args.threads or 'auto'}    "
          f"temp: {args.temperature}    quant: INT8 vs Q8_0")
    print("─" * W)
    print(f"  {'Metric':<20} {'MicroGemm':>14} {'llama.cpp':>14} "
          f"{'Ratio':>10}  {'(stdev)':>12}")
    print("─" * W)

    rows = [
        ("prefill (ms)",  "prefill_ms",  True),
        ("decode (ms)",   "decode_ms",   True),
        ("total (ms)",    "total_ms",    True),
        ("prefill (t/s)", "prefill_tps", False),
        ("decode (t/s)",  "decode_tps",  False),
    ]

    for label, key, lower_is_better in rows:
        mg_m = mg_agg.get(key, {}).get("median", 0.0)
        ll_m = llama_agg.get(key, {}).get("median", 0.0)
        mg_sd = mg_agg.get(key, {}).get("stdev", 0.0)
        ll_sd = llama_agg.get(key, {}).get("stdev", 0.0)

        # For latency: ratio = mg/llama (lower ratio = microgemm faster)
        # For throughput: ratio = mg/llama (higher ratio = microgemm faster)
        if lower_is_better:
            ratio_str = _ratio(mg_m, ll_m)
        else:
            ratio_str = _ratio(mg_m, ll_m)

        # Color indicator (emoji)
        if mg_m > 0 and ll_m > 0:
            r = mg_m / ll_m
            if lower_is_better:
                indicator = "🟢" if r < 0.95 else ("🟡" if r < 1.05 else "🔴")
            else:
                indicator = "🟢" if r > 1.05 else ("🟡" if r > 0.95 else "🔴")
        else:
            indicator = "  "

        print(f"  {label:<20} {_fmt(mg_m):>14} {_fmt(ll_m):>14} "
              f"{ratio_str:>8} {indicator} "
              f"±{_fmt(mg_sd):<5} / ±{_fmt(ll_sd):<5}")

    print("─" * W)
    print(f"  MicroGemm output: {mg_text[:80]}")
    print(f"  llama.cpp output: {llama_text[:80]}")
    print("═" * W)
    print()


def export_json(
    mg_agg: Dict, llama_agg: Dict,
    mg_runs: List, llama_runs: List,
    args: argparse.Namespace, path: str,
) -> None:
    data = {
        "benchmark": "microgemm_vs_llamacpp_cpu",
        "config": {
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "threads": args.threads,
            "runs": args.runs,
        },
        "microgemm": {"aggregate": mg_agg, "runs": mg_runs},
        "llamacpp": {"aggregate": llama_agg, "runs": llama_runs},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(f"[bench] results exported to {path}")


# ── main ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robust benchmark: MicroGemm vs llama.cpp (CPU)")

    # Paths
    p.add_argument("--model-dir", required=True,
                    help="HF model directory (config.json, tokenizer.json, model.safetensors)")
    p.add_argument("--gguf-path", required=True,
                    help="Path to the GGUF model for llama.cpp (e.g. Q8_0)")
    p.add_argument("--mgm-path", default="out/model.mgm",
                    help="Path to the .mgm file (auto-created if missing)")
    p.add_argument("--llama-cli-bin", default="",
                    help="Path to llama-cli binary (auto-detected if omitted)")
    p.add_argument("--microgemm-text-bin", default="./microgemm-text")
    p.add_argument("--microgemm-convert-bin", default="./microgemm-convert")
    p.add_argument("--tokenizer-json", default="")

    # Generation
    p.add_argument("--prompt", required=True, help="Prompt text")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Benchmark control
    p.add_argument("--threads", type=int, default=0,
                    help="CPU threads (0 = auto)")
    p.add_argument("--runs", type=int, default=5,
                    help="Number of timed iterations (default: 5)")
    p.add_argument("--warmup", type=int, default=1,
                    help="Warmup iterations before timing (default: 1)")
    p.add_argument("--ctx-size", type=int, default=512,
                    help="llama.cpp context size (-c)")

    # Output
    p.add_argument("--json", default="",
                    help="Export results to JSON file")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()

    # ── resolve paths ──────────────────────────────────────────────────
    model_dir = _resolve(args.model_dir, cwd)
    gguf_path = _resolve(args.gguf_path, cwd)
    mgm_path = _resolve(args.mgm_path, cwd)
    text_bin = _resolve(args.microgemm_text_bin, cwd)
    convert_bin = _resolve(args.microgemm_convert_bin, cwd)
    tokenizer_json = (
        _resolve(args.tokenizer_json, cwd) if args.tokenizer_json
        else model_dir / "tokenizer.json"
    )

    if not model_dir.exists():
        raise SystemExit(f"model dir not found: {model_dir}")
    if not gguf_path.exists():
        raise SystemExit(f"GGUF not found: {gguf_path}")
    if not tokenizer_json.exists():
        raise SystemExit(f"tokenizer.json not found: {tokenizer_json}")

    # ── setup ──────────────────────────────────────────────────────────
    _ensure_mg_binaries(cwd, text_bin, convert_bin)
    _ensure_mgm(mgm_path, model_dir, convert_bin)

    llama_cli = _find_llama_cli(args.llama_cli_bin, cwd)
    if llama_cli is None:
        raise SystemExit(
            "llama-cli not found. Provide --llama-cli-bin or put it in PATH.\n"
            "Build it: git clone llama.cpp && cd llama.cpp && "
            "cmake -B build -DGGML_CUDA=OFF && cmake --build build -t llama-cli"
        )
    print(f"[bench] MicroGemm text: {text_bin}")
    print(f"[bench] llama-cli:      {llama_cli}")
    print(f"[bench] model .mgm:     {mgm_path}")
    print(f"[bench] model GGUF:     {gguf_path}")
    print(f"[bench] runs: {args.runs}  warmup: {args.warmup}  threads: {args.threads or 'auto'}")
    print()

    # ── shared run kwargs ──────────────────────────────────────────────
    mg_kwargs = dict(
        text_bin=text_bin, mgm_path=mgm_path,
        tokenizer_json=tokenizer_json, prompt=args.prompt,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, seed=args.seed,
        threads=args.threads,
    )
    ll_kwargs = dict(
        llama_cli=llama_cli, gguf_path=gguf_path, prompt=args.prompt,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, seed=args.seed,
        threads=args.threads, ctx_size=args.ctx_size,
    )

    # ── warmup ─────────────────────────────────────────────────────────
    for i in range(args.warmup):
        print(f"[warmup {i+1}/{args.warmup}] MicroGemm…", end=" ", flush=True)
        _run_mg_once(**mg_kwargs)
        print("llama.cpp…", end=" ", flush=True)
        _run_llama_once(**ll_kwargs)
        print("done", flush=True)

    # ── timed runs ─────────────────────────────────────────────────────
    mg_runs: List[Dict[str, float]] = []
    ll_runs: List[Dict[str, float]] = []

    for i in range(args.runs):
        print(f"[run {i+1}/{args.runs}] ", end="", flush=True)

        print("MicroGemm…", end=" ", flush=True)
        mg_result = _run_mg_once(**mg_kwargs)
        mg_runs.append(mg_result)

        print("llama.cpp…", end=" ", flush=True)
        ll_result = _run_llama_once(**ll_kwargs)
        ll_runs.append(ll_result)

        print(f"mg={mg_result['decode_tps']:.1f} t/s  "
              f"ll={ll_result['decode_tps']:.1f} t/s", flush=True)

    # ── aggregate ──────────────────────────────────────────────────────
    mg_agg = _aggregate(mg_runs)
    ll_agg = _aggregate(ll_runs)

    mg_text = mg_runs[-1].get("text", "") if mg_runs else ""
    ll_text = ll_runs[-1].get("text", "") if ll_runs else ""

    print_results(mg_agg, ll_agg, mg_text, ll_text, args, args.runs)

    if args.json:
        export_json(mg_agg, ll_agg, mg_runs, ll_runs, args, args.json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

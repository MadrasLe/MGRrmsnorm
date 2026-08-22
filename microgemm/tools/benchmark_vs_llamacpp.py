"""
Compare MicroGemm CPU generation against llama.cpp from a standalone
`microgemm/` checkout.

This is an external benchmark harness only. It does not make the MicroGemm
runtime depend on Python.

Examples:

    cd /content/drive/MyDrive/microgemm

    python tools/benchmark_vs_llamacpp.py \
      --model-dir /path/to/hf-model-dir \
      --gguf-path /path/to/model-q8_0.gguf \
      --prompt "Explique por que o ceu parece azul em uma frase curta." \
      --max-new-tokens 32 \
      --threads 2 \
      --temperature 0.0

    python tools/benchmark_vs_llamacpp.py \
      --model-dir /path/to/hf-model-dir \
      --gguf-repo QuantFactory/SmolLM2-135M-Instruct-GGUF \
      --gguf-file SmolLM2-135M-Instruct.Q8_0.gguf \
      --prompt "Explique por que o ceu parece azul em uma frase curta." \
      --max-new-tokens 32 \
      --threads 2 \
      --temperature 0.0
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List

from benchmark_vs_hf import ensure_microgemm_binaries, ensure_mgm_exists, resolve_path, run_microgemm


PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+tokens?.*?\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s+tokens per second",
    re.IGNORECASE,
)
EVAL_RE = re.compile(
    r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+runs?.*?\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s+tokens per second",
    re.IGNORECASE,
)
TOTAL_RE = re.compile(r"total time\s*=\s*([\d.]+)\s*ms", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MicroGemm vs llama.cpp on CPU")
    parser.add_argument("--model-dir", required=True, help="HF model directory used by microgemm-convert")
    parser.add_argument("--mgm-path", default="out/model.mgm", help="Path to converted MicroGemm .mgm file")
    parser.add_argument("--prompt", required=True, help="Prompt text to benchmark")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for both runtimes")
    parser.add_argument("--microgemm-text-bin", default="./microgemm-text", help="Path to the microgemm-text binary")
    parser.add_argument("--microgemm-convert-bin", default="./microgemm-convert", help="Path to the microgemm-convert binary")
    parser.add_argument("--tokenizer-json", default="", help="Optional tokenizer.json path override")
    parser.add_argument("--gguf-path", default="", help="Path to a GGUF file for llama.cpp")
    parser.add_argument("--gguf-repo", default="", help="Optional HF repo for GGUF download")
    parser.add_argument("--gguf-file", default="", help="Optional filename inside the GGUF repo")
    parser.add_argument("--llama-cli-bin", default="", help="Path to llama-cli if already built")
    parser.add_argument("--llama-dir", default=".cache/llama.cpp", help="Directory to build/download llama.cpp into")
    parser.add_argument("--gguf-cache-dir", default=".cache/gguf", help="Directory for downloaded GGUF files")
    parser.add_argument(
        "--llama-ctx-size",
        type=int,
        default=512,
        help="Context size passed to llama.cpp (-c). Lower it if Colab kills the process.",
    )
    parser.add_argument("--warmup-new-tokens", type=int, default=8, help="Warmup generation length")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warmup runs")
    return parser.parse_args()


def ensure_gguf_exists(args: argparse.Namespace, cwd: Path) -> Path:
    if args.gguf_path:
        gguf_path = resolve_path(args.gguf_path, cwd)
        if not gguf_path.exists():
            raise SystemExit(f"--gguf-path not found: {gguf_path}")
        print(f"[llama.cpp] usando GGUF existente: {gguf_path}", flush=True)
        return gguf_path

    if not args.gguf_repo or not args.gguf_file:
        raise SystemExit("provide --gguf-path or both --gguf-repo and --gguf-file")

    from huggingface_hub import hf_hub_download

    cache_dir = resolve_path(args.gguf_cache_dir, cwd)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[llama.cpp] baixando GGUF {args.gguf_repo}/{args.gguf_file} para {cache_dir}",
        flush=True,
    )
    path = hf_hub_download(
        repo_id=args.gguf_repo,
        filename=args.gguf_file,
        local_dir=str(cache_dir),
    )
    print(f"[llama.cpp] GGUF pronto em {path}", flush=True)
    return Path(path).resolve()


def resolve_llama_cli_path(raw: str, cwd: Path) -> Path | None:
    if raw:
        path = resolve_path(raw, cwd)
        return path if path.exists() else None

    candidates = [
        cwd / "llama-cli",
        cwd / "llama.cpp" / "llama-cli",
        cwd / "llama.cpp" / "build" / "bin" / "llama-cli",
        cwd / ".cache" / "llama.cpp" / "build" / "bin" / "llama-cli",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def build_llama_cpp(llama_dir: Path) -> Path:
    llama_dir.parent.mkdir(parents=True, exist_ok=True)

    if not llama_dir.exists():
        print(f"[llama.cpp] clonando repo em {llama_dir}", flush=True)
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp.git", str(llama_dir)],
            check=True,
        )

    print("[llama.cpp] configurando build com CMake", flush=True)
    subprocess.run(
        [
            "cmake",
            "-B", "build",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
            "-DLLAMA_BUILD_EXAMPLES=ON",
            "-DLLAMA_BUILD_TOOLS=ON",
        ],
        cwd=str(llama_dir),
        check=True,
    )
    print("[llama.cpp] detectando target de CLI", flush=True)
    help_result = subprocess.run(
        ["cmake", "--build", "build", "--target", "help"],
        cwd=str(llama_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    help_text = f"{help_result.stdout}\n{help_result.stderr}"
    target = ""
    for candidate in ("llama-cli", "main"):
        if re.search(rf"(?m)^[. ]*{re.escape(candidate)}$", help_text):
            target = candidate
            break
    if not target:
        if "llama-cli" in help_text:
            target = "llama-cli"
        elif "main" in help_text:
            target = "main"
        else:
            raise SystemExit("could not find a CLI target in llama.cpp CMake targets")

    jobs = min(2, os.cpu_count() or 2)
    print(f"[llama.cpp] compilando target `{target}` com -j{jobs}", flush=True)
    subprocess.run(
        ["cmake", "--build", "build", "--config", "Release", "--target", target, "-j", str(jobs)],
        cwd=str(llama_dir),
        check=True,
    )

    candidates = [
        llama_dir / "build" / "bin" / "llama-cli",
        llama_dir / "build" / "llama-cli",
    ]
    for path in candidates:
        if path.exists():
            print(f"[llama.cpp] llama-cli pronto em {path}", flush=True)
            return path.resolve()

    raise SystemExit("llama.cpp build finished but llama-cli was not found")


def ensure_llama_cli(args: argparse.Namespace, cwd: Path) -> Path:
    cli = resolve_llama_cli_path(args.llama_cli_bin, cwd)
    if cli is not None:
        print(f"[llama.cpp] usando llama-cli existente: {cli}", flush=True)
        return cli

    llama_dir = resolve_path(args.llama_dir, cwd)
    return build_llama_cpp(llama_dir)


def parse_llama_metrics(stderr_text: str, wall_total_ms: float) -> Dict[str, str]:
    metrics: Dict[str, str] = {}

    prompt_match = PROMPT_EVAL_RE.search(stderr_text)
    if prompt_match:
        metrics["prompt_token_count"] = prompt_match.group(2)
        metrics["prefill_ms"] = f"{float(prompt_match.group(1)):.3f}"
        metrics["prefill_tps"] = f"{float(prompt_match.group(3)):.3f}"

    eval_match = EVAL_RE.search(stderr_text)
    if eval_match:
        metrics["generated_token_count"] = eval_match.group(2)
        metrics["decode_ms"] = f"{float(eval_match.group(1)):.3f}"
        metrics["decode_tps"] = f"{float(eval_match.group(3)):.3f}"

    total_match = TOTAL_RE.search(stderr_text)
    total_ms = float(total_match.group(1)) if total_match else wall_total_ms
    metrics["total_ms"] = f"{total_ms:.3f}"

    prompt_tokens = int(metrics.get("prompt_token_count", "0"))
    generated_tokens = int(metrics.get("generated_token_count", "0"))
    if total_ms > 0.0 and (prompt_tokens + generated_tokens) > 0:
        metrics["total_tps"] = f"{((prompt_tokens + generated_tokens) * 1000.0 / total_ms):.3f}"
    else:
        metrics["total_tps"] = "0.000"

    return metrics


def run_llama_cpp(
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
) -> Dict[str, str]:
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

    wall_start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall_total_ms = (time.perf_counter() - wall_start) * 1000.0
    if result.returncode != 0:
        raise RuntimeError(
            "llama.cpp failed\n"
            f"exit_code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    metrics = parse_llama_metrics(result.stderr, wall_total_ms)
    metrics["generated_text"] = result.stdout.strip()
    return metrics


def maybe_warmup(
    args: argparse.Namespace,
    microgemm_text_bin: Path,
    mgm_path: Path,
    tokenizer_json: Path,
    llama_cli: Path,
    gguf_path: Path,
) -> None:
    if args.skip_warmup or args.warmup_new_tokens <= 0:
        return

    print(f"[bench] warmup de {args.warmup_new_tokens} tokens", flush=True)
    run_microgemm(
        microgemm_text_bin=microgemm_text_bin,
        mgm_path=mgm_path,
        tokenizer_json=tokenizer_json,
        prompt=args.prompt,
        max_new_tokens=args.warmup_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        threads=args.threads,
    )
    run_llama_cpp(
        llama_cli=llama_cli,
        gguf_path=gguf_path,
        prompt=args.prompt,
        max_new_tokens=args.warmup_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        threads=args.threads,
        ctx_size=args.llama_ctx_size,
    )
    print("[bench] warmup concluido", flush=True)


def print_side_by_side(mg: Dict[str, str], llama: Dict[str, str], gguf_path: Path) -> None:
    keys = [
        "prompt_token_count",
        "generated_token_count",
        "prefill_ms",
        "decode_ms",
        "total_ms",
        "prefill_tps",
        "decode_tps",
        "total_tps",
    ]

    print("=" * 82)
    print("MicroGemm vs llama.cpp (CPU)")
    print("=" * 82)
    print(f"gguf: {gguf_path}")
    print(f"{'metric':<22} {'microgemm':>18} {'llama.cpp':>18} {'ratio mg/llama':>18}")
    print("-" * 82)
    for key in keys:
        mg_value = mg.get(key, "")
        llama_value = llama.get(key, "")
        ratio_str = ""
        try:
            mg_num = float(mg_value)
            llama_num = float(llama_value)
            ratio = (mg_num / llama_num) if llama_num != 0.0 else float("inf")
            ratio_str = f"{ratio:.3f}x"
        except ValueError:
            ratio_str = ""
        print(f"{key:<22} {mg_value:>18} {llama_value:>18} {ratio_str:>18}")
    print("-" * 82)
    print("MicroGemm text:")
    print(mg.get("generated_text", "").strip())
    print("-" * 82)
    print("llama.cpp text:")
    print(llama.get("generated_text", "").strip())
    print("=" * 82)


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    model_dir = resolve_path(args.model_dir, cwd)
    mgm_path = resolve_path(args.mgm_path, cwd)
    microgemm_text_bin = resolve_path(args.microgemm_text_bin, cwd)
    microgemm_convert_bin = resolve_path(args.microgemm_convert_bin, cwd)
    tokenizer_json = resolve_path(args.tokenizer_json, cwd) if args.tokenizer_json else (model_dir / "tokenizer.json")

    if not model_dir.exists():
        raise SystemExit(f"--model-dir not found: {model_dir}")
    if not tokenizer_json.exists():
        raise SystemExit(f"tokenizer.json not found: {tokenizer_json}")
    ensure_microgemm_binaries(
        cwd=cwd,
        microgemm_text_bin=microgemm_text_bin,
        microgemm_convert_bin=microgemm_convert_bin,
    )

    ensure_mgm_exists(
        mgm_path=mgm_path,
        model_dir=model_dir,
        microgemm_convert_bin=microgemm_convert_bin,
    )
    gguf_path = ensure_gguf_exists(args, cwd)
    llama_cli = ensure_llama_cli(args, cwd)

    maybe_warmup(
        args=args,
        microgemm_text_bin=microgemm_text_bin,
        mgm_path=mgm_path,
        tokenizer_json=tokenizer_json,
        llama_cli=llama_cli,
        gguf_path=gguf_path,
    )

    print("[bench] rodando MicroGemm", flush=True)
    mg = run_microgemm(
        microgemm_text_bin=microgemm_text_bin,
        mgm_path=mgm_path,
        tokenizer_json=tokenizer_json,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        threads=args.threads,
    )
    print("[bench] rodando llama.cpp", flush=True)
    llama = run_llama_cpp(
        llama_cli=llama_cli,
        gguf_path=gguf_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        threads=args.threads,
        ctx_size=args.llama_ctx_size,
    )
    print_side_by_side(mg, llama, gguf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

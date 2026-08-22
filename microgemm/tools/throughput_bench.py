"""
throughput_bench.py — Zero-setup throughput benchmark: MicroGemm vs llama.cpp

Usage:
    python tools/throughput_bench.py

That's it. No arguments needed. The script handles everything:
  1. Installs huggingface_hub if missing
  2. Downloads TinyLlama-1.1B-Chat-v1.0 from HuggingFace
  3. Downloads matching Q8_0 GGUF from HuggingFace
  4. Builds llama.cpp (Linux/macOS) or downloads binaries (Windows)
  5. Builds MicroGemm (make / build.ps1)
  6. Converts model to .mgm
  7. Runs the throughput benchmark

Everything is cached in microgemm/.cache/ — re-runs are instant.

Optional flags:
  --model-dir <path>    Skip download, use existing HF model dir
  --gguf-path <path>    Skip download, use existing GGUF file
  --llama-cli <path>    Skip download, use existing llama-cli
  --prompt <text>       Custom prompt (default: short question)
  --max-new-tokens N    Tokens to generate (default: 32)
  --threads N           CPU threads (default: auto)
  --runs N              Timed iterations (default: 5)
  --warmup N            Warmup iterations (default: 1)
  --json <path>         Export results to JSON
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError


# ── constants ─────────────────────────────────────────────────────────

HF_MODEL_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_GGUF_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
HF_GGUF_FILE = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
# Lock to stable b4300 to avoid recent dynamic plugin (libggml-cpu-*.so) bugs on Colab
LLAMA_GITHUB_API = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/tags/b4300"


# ── regex for llama.cpp stderr ────────────────────────────────────────

_PROMPT_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+tokens?",
    re.IGNORECASE,
)
_EVAL_RE = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+(?:runs?|tokens?)",
    re.IGNORECASE,
)

_LEADING_CHAT_TAG_RE = re.compile(
    r"^\s*(?:<\|[^|>]+?\|>|<s>|</s>|\[/?INST\]|<<SYS>>|<</SYS>>)+\s*",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════
#  SETUP: auto-download everything
# ═══════════════════════════════════════════════════════════════════════

def ensure_huggingface_hub() -> None:
    """Install huggingface_hub if not present."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("  ⚙  installing huggingface_hub...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"],
            stdout=subprocess.DEVNULL,
        )
        print("  ✓  huggingface_hub installed")


def download_hf_model(cache_dir: Path) -> Path:
    """Download HF model using huggingface_hub snapshot_download."""
    model_dir = cache_dir / "hf" / HF_MODEL_REPO.split("/")[-1]

    # Check if already downloaded (has config.json + model.safetensors)
    if (model_dir / "config.json").exists() and _has_safetensors(model_dir):
        print(f"  ✓  model already cached: {model_dir.name}")
        return model_dir

    ensure_huggingface_hub()
    from huggingface_hub import snapshot_download

    print(f"  ⬇  downloading {HF_MODEL_REPO} (~1 GB)...", flush=True)
    downloaded = snapshot_download(
        repo_id=HF_MODEL_REPO,
        local_dir=str(model_dir),
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )
    print(f"  ✓  model downloaded → {Path(downloaded).name}")
    return Path(downloaded)


def _has_safetensors(d: Path) -> bool:
    return any(d.glob("*.safetensors"))


def download_gguf(cache_dir: Path) -> Path:
    """Download Q8_0 GGUF using huggingface_hub."""
    gguf_dir = cache_dir / "gguf"
    gguf_path = gguf_dir / HF_GGUF_FILE

    if gguf_path.exists():
        print(f"  ✓  GGUF already cached: {gguf_path.name}")
        return gguf_path

    ensure_huggingface_hub()
    from huggingface_hub import hf_hub_download

    print(f"  ⬇  downloading {HF_GGUF_FILE} (~530 MB)...", flush=True)
    gguf_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_GGUF_REPO,
        filename=HF_GGUF_FILE,
        local_dir=str(gguf_dir),
    )
    print(f"  ✓  GGUF downloaded → {Path(downloaded).name}")
    return Path(downloaded)


def _get_llama_asset_pattern() -> str:
    """Determine which llama.cpp release asset to download for this platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        return "win-avx2-x64"
    elif system == "linux":
        if "aarch64" in machine or "arm64" in machine:
            return "ubuntu-aarch64"
        return "ubuntu-x64"
    elif system == "darwin":
        if "arm64" in machine or "aarch64" in machine:
            return "macos-arm64"
        return "macos-x64"
    return "ubuntu-x64"


def _download_url(url: str) -> bytes:
    """Download a URL and return bytes."""
    req = Request(url, headers={"User-Agent": "microgemm-bench/1.0"})
    with urlopen(req, timeout=120) as resp:
        return resp.read()


def download_llama_cli(cache_dir: Path) -> Path:
    """Download or compile llama-cli."""
    llama_dir = cache_dir / "llama"
    ext = ".exe" if platform.system().lower() == "windows" else ""
    cli_path = llama_dir / f"llama-cli{ext}"
    main_path = llama_dir / f"main{ext}"

    if cli_path.exists():
        print(f"  ✓  llama-cli already cached")
        return cli_path
    if main_path.exists():
        print(f"  ✓  llama-cli already cached")
        return main_path

    llama_dir.mkdir(parents=True, exist_ok=True)

    # ON LINUX/MAC: ALWAYS BUILD FROM SOURCE to avoid GLIBC and plugin bugs!
    if platform.system().lower() != "windows":
        print(f"  ⚙  fetching llama.cpp source (b4300)...", flush=True)
        src_url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b4300.zip"
        archive_bytes = _download_url(src_url)
        print(f"  ⚙  building llama-cli from source...", flush=True)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            zf.extractall(path=llama_dir.parent)

        src_dir = llama_dir.parent / "llama.cpp-b4300"

        # Configure CMake
        r = subprocess.run(["cmake", "-B", "build", "-DLLAMA_CUDA=OFF"], cwd=src_dir, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"llama.cpp CMake config failed:\n{r.stderr}")

        # Build with CMake
        r = subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j4", "-t", "llama-cli"], cwd=src_dir, capture_output=True, text=True)

        if r.returncode != 0 and ("unknown target" in r.stderr.lower() or "no rule" in r.stderr.lower()):
            # Fallback to 'main' for older b4300 if llama-cli isn't the target name yet
            r = subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j4", "-t", "main"], cwd=src_dir, capture_output=True, text=True)

        if r.returncode != 0:
            raise RuntimeError(f"llama.cpp CMake build failed:\n{r.stderr}")

        # Locate the compiled binary robustly
        src_bin = None
        bin_names = ["llama-cli", "main"]
        for p in (src_dir / "build").rglob("*"):
            if p.is_file() and p.name in bin_names and os.access(p, os.X_OK):
                src_bin = p
                break

        if not src_bin:
            raise RuntimeError(f"llama.cpp build succeeded but binary not found in {src_dir}/build\nLog:\n{r.stdout}")

        shutil.copy2(src_bin, cli_path)
        print(f"  ✓  llama-cli compiled natively")
        return cli_path

    # ON WINDOWS: Download pre-built
    pattern = _get_llama_asset_pattern()
    print(f"  ⬇  fetching llama.cpp latest release info...", flush=True)
    try:
        data = json.loads(_download_url(LLAMA_GITHUB_API))
    except Exception as e:
        raise RuntimeError(f"Could not fetch llama.cpp releases from GitHub: {e}")

    asset_url = None
    asset_name = None
    for asset in data.get("assets", []):
        name = asset["name"]
        if "cudart" in name or "sha256" in name:
            continue
        if pattern in name and name.endswith(".zip"):
            asset_url = asset["browser_download_url"]
            asset_name = name
            break

    if not asset_url:
        raise RuntimeError(f"No matching llama.cpp asset found for '{pattern}'.")

    print(f"  ⬇  downloading {asset_name}...", flush=True)
    archive_bytes = _download_url(asset_url)

    print(f"  ⚙  extracting llama.cpp binaries...", flush=True)
    extracted = False
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        for info in zf.namelist():
            if info.endswith("/"): continue
            member_path = Path(info)
            out_path = llama_dir / member_path.name
            with zf.open(info) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            if member_path.name.lower() in (f"llama-cli{ext}", f"main{ext}"):
                extracted = True

    if not extracted:
        raise RuntimeError(f"Could not find llama-cli inside {asset_name}.")

    print(f"  ✓  llama-cli downloaded")
    if cli_path.exists():
        return cli_path
    if main_path.exists():
        return main_path
    raise RuntimeError(f"Could not find extracted llama-cli binary in {llama_dir}.")


def find_or_download_llama_cli(
    hint: str, cache_dir: Path, search_dirs: List[Path]
) -> Path:
    """Try hint → common locations → PATH → download."""
    if hint:
        p = Path(hint).resolve()
        if p.exists():
            return p

    ext = ".exe" if platform.system().lower() == "windows" else ""
    for d in search_dirs:
        for candidate in [d / f"llama-cli{ext}", d / f"main{ext}", d / "llama-cli", d / "main"]:
            if candidate.exists():
                return candidate.resolve()

    w = shutil.which("llama-cli")
    if w:
        return Path(w).resolve()
    w = shutil.which("main")
    if w:
        return Path(w).resolve()

    # Auto-download
    return download_llama_cli(cache_dir)


def build_microgemm(cwd: Path) -> None:
    """Build MicroGemm using make or build.ps1."""
    system = platform.system().lower()

    if system == "windows":
        ps1 = cwd / "build.ps1"
        if ps1.exists():
            print("  ⚙  building MicroGemm (build.ps1)...", flush=True)
            try:
                subprocess.run(
                    ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
                    cwd=str(cwd), check=True, capture_output=True, text=True,
                )
                print("  ✓  MicroGemm built")
                return
            except subprocess.CalledProcessError as e:
                print(f"  ⚠  build.ps1 failed: {e.stderr[:200]}")
                print("  ⚠  you may need to run from a Developer PowerShell")
    else:
        makefile = cwd / "Makefile"
        if makefile.exists() and shutil.which("make"):
            print("  ⚙  building MicroGemm (make)...", flush=True)
            try:
                subprocess.run(
                    ["make"], cwd=str(cwd), check=True,
                    capture_output=True, text=True,
                )
                print("  ✓  MicroGemm built")
                return
            except subprocess.CalledProcessError as e:
                print(f"  ⚠  make failed: {e.stderr[:200]}")


def find_mg_binary(name: str, hint: str, search_dirs: List[Path]) -> Optional[Path]:
    """Find a MicroGemm binary."""
    ext = ".exe" if platform.system().lower() == "windows" else ""
    if hint:
        p = Path(hint).resolve()
        if p.exists():
            return p
    for d in search_dirs:
        for candidate in [d / f"{name}{ext}", d / name]:
            if candidate.exists():
                return candidate.resolve()
    w = shutil.which(name)
    return Path(w).resolve() if w else None


def _ensure_executable(path: Path) -> Path:
    """Best-effort executable bit fix for POSIX filesystems (e.g. Google Drive mounts)."""
    if platform.system().lower() == "windows":
        return path
    try:
        mode = path.stat().st_mode
        if (mode & 0o111) == 0:
            path.chmod(mode | 0o111)
    except OSError:
        pass
    return path


def convert_to_mgm(convert_bin: Path, model_dir: Path, mgm_path: Path) -> None:
    """Convert model to .mgm if not present."""
    if mgm_path.exists():
        print(f"  ✓  .mgm already cached: {mgm_path.name}")
        return
    mgm_path.parent.mkdir(parents=True, exist_ok=True)
    convert_bin = _ensure_executable(convert_bin)
    print(f"  ⚙  converting model → {mgm_path.name}...", flush=True)
    try:
        r = subprocess.run(
            [str(convert_bin), "from-dir", str(model_dir), str(mgm_path)],
            capture_output=True, text=True,
        )
    except PermissionError:
        convert_bin = _ensure_executable(convert_bin)
        r = subprocess.run(
            [str(convert_bin), "from-dir", str(model_dir), str(mgm_path)],
            capture_output=True, text=True,
        )
    if r.returncode != 0:
        raise RuntimeError(f"Conversion failed:\n  {r.stderr[:400]}")
    print(f"  ✓  conversion complete → {mgm_path.name}")


# ═══════════════════════════════════════════════════════════════════════
#  BENCHMARK: runners + stats + report
# ═══════════════════════════════════════════════════════════════════════

def run_microgemm(
    text_bin: Path, mgm_path: Path, tokenizer_json: Path,
    prompt: str, max_new_tokens: int, threads: int,
) -> Dict[str, float]:
    """Run microgemm-text once, return throughput metrics."""
    text_bin = _ensure_executable(text_bin)
    cmd = [
        str(text_bin), "generate",
        str(mgm_path), str(tokenizer_json),
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--temperature", "0", "--seed", "42",
    ]
    env = os.environ.copy()
    if threads > 0:
        env["OMP_NUM_THREADS"] = str(threads)

    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except PermissionError:
        text_bin = _ensure_executable(text_bin)
        cmd[0] = str(text_bin)
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    if r.returncode != 0:
        raise RuntimeError(
            f"microgemm-text failed (exit {r.returncode})\n{r.stderr[:400]}"
        )

    kv = {}
    for line in r.stdout.splitlines():
        if ": " in line and not line.startswith(" "):
            k, _, v = line.partition(": ")
            kv[k.strip()] = v.strip()
    generated_text = ""
    generated_match = re.search(r"generated_text:\s*\n(.*?)(?:\nfull_text:|\Z)", r.stdout, re.DOTALL)
    if generated_match:
        generated_text = generated_match.group(1).strip()

    prefill_ms = float(kv.get("prefill_ms", "0"))
    decode_ms = float(kv.get("decode_ms", "0"))
    total_ms = float(kv.get("total_ms", "0")) or wall_ms
    prompt_tok = int(kv.get("prompt_token_count", "0"))
    gen_tok = int(kv.get("generated_token_count", "0"))
    loaded_model_bytes = int(kv.get("loaded_model_bytes", "0"))
    workspace_bytes = int(kv.get("workspace_bytes", "0"))
    kv_cache_bytes = int(kv.get("kv_cache_bytes", "0"))
    runtime_total_bytes = int(kv.get("runtime_total_bytes", "0"))
    # microgemm-text reports decode_ms only for tokens generated after the
    # first token (first token comes from prefill logits).
    decode_tok = max(gen_tok - 1, 0)

    return {
        "prefill_tps": (prompt_tok / prefill_ms * 1000) if prefill_ms > 0 else 0.0,
        "decode_tps": (decode_tok / decode_ms * 1000) if decode_ms > 0 and decode_tok > 0 else 0.0,
        "total_ms": total_ms,
        "prompt_tokens": prompt_tok,
        "gen_tokens": gen_tok,
        "decode_tokens": decode_tok,
        "loaded_model_bytes": loaded_model_bytes,
        "workspace_bytes": workspace_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "runtime_total_bytes": runtime_total_bytes,
        "text": generated_text[:120],
    }


def run_llamacpp(
    llama_cli: Path, gguf_path: Path, prompt: str,
    max_new_tokens: int, threads: int, ctx_size: int, timeout_sec: int,
) -> Dict[str, float]:
    """Run llama-cli once, return throughput metrics."""
    llama_cli = _ensure_executable(llama_cli)
    cmd = [
        str(llama_cli),
        "-m", str(gguf_path),
        "-p", prompt,
        "-n", str(max_new_tokens),
        "-c", str(ctx_size) if ctx_size > 0 else "512",
        "--temp", "0",
        "--top-k", "1",
        "--top-p", "1.0",
        "--seed", "42",
        "--no-display-prompt",
    ]
    if threads > 0:
        cmd.extend(["-t", str(threads)])

    # Set LD_LIBRARY_PATH so llama-cli finds its shared libs
    env = os.environ.copy()
    llama_lib_dir = str(llama_cli.parent)
    if platform.system().lower() != "windows":
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{llama_lib_dir}:{existing}" if existing else llama_lib_dir
    else:
        existing = env.get("PATH", "")
        env["PATH"] = f"{llama_lib_dir};{existing}"

    t0 = time.perf_counter()

    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True, stdin=subprocess.DEVNULL
        )
    except PermissionError:
        llama_cli = _ensure_executable(llama_cli)
        cmd[0] = str(llama_cli)
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True, stdin=subprocess.DEVNULL
        )

    try:
        stdout_text, r_stderr = p.communicate(timeout=max(10, timeout_sec))
    except subprocess.TimeoutExpired:
        p.kill()
        stdout_text, r_stderr = p.communicate()
        raise RuntimeError(
            f"llama-cli timed out after {max(10, timeout_sec)}s\n"
            f"Args: {' '.join(cmd)}\n"
            f"Stderr:\n{r_stderr[:1000]}"
        )

    wall_ms = (time.perf_counter() - t0) * 1000.0

    if p.returncode != 0:
        raise RuntimeError(
            f"llama-cli failed (exit {p.returncode})\n{r_stderr[:1000]}"
        )

    stderr = r_stderr
    pm = _PROMPT_RE.search(stderr)
    em = _EVAL_RE.search(stderr)

    prefill_ms = float(pm.group(1)) if pm else 0.0
    prompt_tok = int(pm.group(2)) if pm else 0
    decode_ms = float(em.group(1)) if em else 0.0
    gen_tok = int(em.group(2)) if em else 0

    return {
        "prefill_tps": (prompt_tok / prefill_ms * 1000) if prefill_ms > 0 else 0.0,
        "decode_tps": (gen_tok / decode_ms * 1000) if decode_ms > 0 and gen_tok > 0 else 0.0,
        "total_ms": wall_ms,
        "prompt_tokens": prompt_tok,
        "gen_tokens": gen_tok,
        "decode_tokens": gen_tok,
        "text": stdout_text.strip()[:120],
    }


def calc_stats(values: List[float]) -> Dict[str, float]:
    """median, mean, stdev, min, max."""
    if not values:
        return {"median": 0, "mean": 0, "stdev": 0, "min": 0, "max": 0}
    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def bytes_to_mib(value: float) -> float:
    return float(value) / (1024.0 * 1024.0)


def clean_preview_text(text: str, max_len: int = 55) -> str:
    if not text:
        return ""
    s = text.replace("\r", " ").replace("\n", " ")
    # Remove common chat/control tags at the start of generation previews.
    while True:
        s2 = _LEADING_CHAT_TAG_RE.sub("", s)
        if s2 == s:
            break
        s = s2
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


def bar(value: float, max_value: float, width: int = 28) -> str:
    if max_value <= 0:
        return "░" * width
    filled = max(0, min(width, int(round(value / max_value * width))))
    return "█" * filled + "░" * (width - filled)


def print_report(
    mg_runs: List[Dict], ll_runs: List[Dict], args: argparse.Namespace,
) -> None:
    mg_decode = calc_stats([r["decode_tps"] for r in mg_runs])
    ll_decode = calc_stats([r["decode_tps"] for r in ll_runs])
    mg_prefill = calc_stats([r["prefill_tps"] for r in mg_runs])
    ll_prefill = calc_stats([r["prefill_tps"] for r in ll_runs])
    mg_total = calc_stats([r["total_ms"] for r in mg_runs])
    ll_total = calc_stats([r["total_ms"] for r in ll_runs])

    prompt_tok = mg_runs[0]["prompt_tokens"] if mg_runs else 0
    gen_tok = mg_runs[0]["gen_tokens"] if mg_runs else 0
    W = 72

    print()
    print("╔" + "═" * W + "╗")
    print("║" + "  THROUGHPUT BENCHMARK · MicroGemm vs llama.cpp (CPU)  ".center(W) + "║")
    print("╠" + "═" * W + "╣")

    p = args.prompt[:42] + "…" if len(args.prompt) > 42 else args.prompt
    thr = args.threads if args.threads > 0 else "auto"
    print(f'║  prompt: "{p}"'.ljust(W + 1) + "║")
    print(f"║  tokens: {prompt_tok} prompt → {gen_tok} decode  |  runs: {args.runs}  |  threads: {thr}".ljust(W + 1) + "║")
    print(f"║  model:  {HF_MODEL_REPO}  |  quant: INT8 vs Q8_0".ljust(W + 1) + "║")
    print("╠" + "═" * W + "╣")

    # ── decode throughput ──
    mx = max(mg_decode["median"], ll_decode["median"], 0.01)
    print("║" + "  DECODE THROUGHPUT (tokens/sec) — the metric that matters".ljust(W + 1) + "║")
    print("║" + "".ljust(W + 1) + "║")

    sd_mg = f' ±{mg_decode["stdev"]:.1f}' if mg_decode["stdev"] > 0 else ""
    sd_ll = f' ±{ll_decode["stdev"]:.1f}' if ll_decode["stdev"] > 0 else ""
    print(f'║    MicroGemm  {bar(mg_decode["median"], mx)}  {mg_decode["median"]:6.2f} t/s{sd_mg}'.ljust(W + 1) + "║")
    print(f'║    llama.cpp  {bar(ll_decode["median"], mx)}  {ll_decode["median"]:6.2f} t/s{sd_ll}'.ljust(W + 1) + "║")

    if ll_decode["median"] > 0 and mg_decode["median"] > 0:
        r = mg_decode["median"] / ll_decode["median"]
        v = f"MicroGemm is {r:.2f}× faster  🟢" if r >= 1 else f"llama.cpp is {1/r:.2f}× faster  🔴"
        print("║" + "".ljust(W + 1) + "║")
        print(f"║    → {v}".ljust(W + 1) + "║")

    print("╠" + "═" * W + "╣")

    # ── prefill throughput ──
    mx = max(mg_prefill["median"], ll_prefill["median"], 0.01)
    print("║" + "  PREFILL THROUGHPUT (tokens/sec)".ljust(W + 1) + "║")
    print("║" + "".ljust(W + 1) + "║")

    sd_mg = f' ±{mg_prefill["stdev"]:.1f}' if mg_prefill["stdev"] > 0 else ""
    sd_ll = f' ±{ll_prefill["stdev"]:.1f}' if ll_prefill["stdev"] > 0 else ""
    print(f'║    MicroGemm  {bar(mg_prefill["median"], mx)}  {mg_prefill["median"]:6.2f} t/s{sd_mg}'.ljust(W + 1) + "║")
    print(f'║    llama.cpp  {bar(ll_prefill["median"], mx)}  {ll_prefill["median"]:6.2f} t/s{sd_ll}'.ljust(W + 1) + "║")

    if ll_prefill["median"] > 0 and mg_prefill["median"] > 0:
        r = mg_prefill["median"] / ll_prefill["median"]
        v = f"MicroGemm is {r:.2f}× faster  🟢" if r >= 1 else f"llama.cpp is {1/r:.2f}× faster  🔴"
        print("║" + "".ljust(W + 1) + "║")
        print(f"║    → {v}".ljust(W + 1) + "║")

    print("╠" + "═" * W + "╣")

    # ── wall time ──
    print("║" + "  TOTAL WALL TIME (ms, lower = better)".ljust(W + 1) + "║")
    print("║" + "".ljust(W + 1) + "║")
    sd_mg = f' ±{mg_total["stdev"]:.0f}' if mg_total["stdev"] > 0 else ""
    sd_ll = f' ±{ll_total["stdev"]:.0f}' if ll_total["stdev"] > 0 else ""
    print(f'║    MicroGemm:  {mg_total["median"]:.0f} ms{sd_mg}'.ljust(W + 1) + "║")
    print(f'║    llama.cpp:  {ll_total["median"]:.0f} ms{sd_ll}'.ljust(W + 1) + "║")

    if mg_runs:
        loaded_b = mg_runs[-1].get("loaded_model_bytes", 0)
        ws_b = mg_runs[-1].get("workspace_bytes", 0)
        kv_b = mg_runs[-1].get("kv_cache_bytes", 0)
        total_b = mg_runs[-1].get("runtime_total_bytes", 0)
        if total_b > 0:
            print("║" + "".ljust(W + 1) + "║")
            print("║  MICROGEMM RUNTIME FOOTPRINT (MiB)".ljust(W + 1) + "║")
            print(f"║    model_loaded:  {bytes_to_mib(loaded_b):.2f}".ljust(W + 1) + "║")
            print(f"║    workspace:     {bytes_to_mib(ws_b):.2f}".ljust(W + 1) + "║")
            print(f"║    kv_cache:      {bytes_to_mib(kv_b):.2f}".ljust(W + 1) + "║")
            print(f"║    total:         {bytes_to_mib(total_b):.2f}".ljust(W + 1) + "║")

    print("╠" + "═" * W + "╣")

    # ── sample output ──
    mg_text = clean_preview_text(mg_runs[-1].get("text", ""))
    ll_text = clean_preview_text(ll_runs[-1].get("text", ""))
    print(f"║  mg output: {mg_text}".ljust(W + 1) + "║")
    print(f"║  ll output: {ll_text}".ljust(W + 1) + "║")

    print("╚" + "═" * W + "╝")
    print()


def export_json(
    mg_runs: List[Dict], ll_runs: List[Dict],
    args: argparse.Namespace, path: str,
) -> None:
    data = {
        "benchmark": "throughput_microgemm_vs_llamacpp",
        "model": HF_MODEL_REPO,
        "config": {
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "threads": args.threads,
            "runs": args.runs,
        },
        "microgemm": {
            "decode_tps": calc_stats([r["decode_tps"] for r in mg_runs]),
            "prefill_tps": calc_stats([r["prefill_tps"] for r in mg_runs]),
            "total_ms": calc_stats([r["total_ms"] for r in mg_runs]),
            "per_run": mg_runs,
        },
        "llamacpp": {
            "decode_tps": calc_stats([r["decode_tps"] for r in ll_runs]),
            "prefill_tps": calc_stats([r["prefill_tps"] for r in ll_runs]),
            "total_ms": calc_stats([r["total_ms"] for r in ll_runs]),
            "per_run": ll_runs,
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(f"  ✓ results → {path}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    p = argparse.ArgumentParser(
        description="Zero-setup throughput benchmark: MicroGemm vs llama.cpp (CPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-dir", default="", help="Use existing HF model dir (skip download)")
    p.add_argument("--gguf-path", default="", help="Use existing GGUF file (skip download)")
    p.add_argument("--llama-cli", default="", help="Use existing llama-cli binary (skip download)")
    p.add_argument("--microgemm-text", default="", help="Path to microgemm-text")
    p.add_argument("--microgemm-convert", default="", help="Path to microgemm-convert")
    p.add_argument(
        "--prompt",
        default="The sky is blue because ",
        help="Input text prompt"
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = auto)")
    p.add_argument("--runs", type=int, default=5, help="Timed iterations")
    p.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    p.add_argument("--ctx-size", type=int, default=512, help="llama.cpp context size")
    p.add_argument("--llama-timeout", type=int, default=120, help="llama-cli timeout in seconds")
    p.add_argument("--json", default="", help="Export results to JSON")
    args = p.parse_args()

    # ── resolve base paths ────────────────────────────────────────────
    cwd = Path.cwd()
    if platform.system().lower() != "windows":
        # Force cache to local fast disk on Colab to bypass Google Drive FUSE quota/hangs entirely
        import tempfile
        cache_dir = Path(tempfile.gettempdir()) / "microgemm_bench_cache"
    else:
        cache_dir = cwd / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │  Throughput Benchmark · MicroGemm vs llama.cpp (CPU) │")
    print("  └──────────────────────────────────────────────────────┘")
    print()

    # ── step 1: model ─────────────────────────────────────────────────
    print("  [1/6] Model", flush=True)
    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
        if not model_dir.exists():
            return _die(f"model dir not found: {model_dir}")
        print(f"  ✓  using provided model: {model_dir}")
    else:
        model_dir = download_hf_model(cache_dir)

    tokenizer_json = model_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        return _die(f"tokenizer.json not found in {model_dir}")

    # ── step 2: GGUF ──────────────────────────────────────────────────
    print("  [2/6] GGUF", flush=True)
    if args.gguf_path:
        gguf_path = Path(args.gguf_path).resolve()
        if not gguf_path.exists():
            return _die(f"GGUF not found: {gguf_path}")
        print(f"  ✓  using provided GGUF: {gguf_path.name}")
    else:
        gguf_path = download_gguf(cache_dir)

    # ── step 3: llama-cli ─────────────────────────────────────────────
    print("  [3/6] llama-cli", flush=True)
    llama_search = [
        cwd, cwd / "llama.cpp" / "build" / "bin",
    ]

    import tempfile
    # FORCE /tmp logic solely for the llama binaries to bypass Google Drive FUSE hangs
    # and properly load relative native backend .so files
    llama_cache_dir = Path(tempfile.gettempdir()) / "microgemm_llama_cache"

    try:
        llama_cli = find_or_download_llama_cli(args.llama_cli, llama_cache_dir, llama_search)
    except RuntimeError as e:
        return _die(str(e))
    llama_cli = _ensure_executable(llama_cli)
    print(f"  ✓  llama-cli: {llama_cli}")

    # ── step 4: build MicroGemm ───────────────────────────────────────
    print("  [4/6] MicroGemm binaries", flush=True)
    mg_search = [cwd, cwd / "build"]
    text_bin = find_mg_binary("microgemm-text", args.microgemm_text, mg_search)
    convert_bin = find_mg_binary("microgemm-convert", args.microgemm_convert, mg_search)

    if not text_bin or not convert_bin:
        build_microgemm(cwd)
        text_bin = text_bin or find_mg_binary("microgemm-text", "", mg_search)
        convert_bin = convert_bin or find_mg_binary("microgemm-convert", "", mg_search)

    if not text_bin:
        return _die(
            "microgemm-text not found.\n"
            "  On Linux/Mac:  cd microgemm && make\n"
            "  On Windows:    cd microgemm && .\\build.ps1  (from Developer PowerShell)"
        )
    if not convert_bin:
        return _die("microgemm-convert not found. Build MicroGemm first.")
    text_bin = _ensure_executable(text_bin)
    convert_bin = _ensure_executable(convert_bin)
    print(f"  ✓  microgemm-text:    {text_bin}")
    print(f"  ✓  microgemm-convert: {convert_bin}")

    # ── step 5: convert to .mgm ──────────────────────────────────────
    print("  [5/6] Convert to .mgm", flush=True)
    mgm_path = cache_dir / "out" / "model.mgm"
    try:
        convert_to_mgm(convert_bin, model_dir, mgm_path)
    except RuntimeError as e:
        return _die(str(e))

    # ── step 6: benchmark! ────────────────────────────────────────────
    print("  [6/6] Benchmark", flush=True)
    thr = args.threads if args.threads > 0 else "auto"
    print(f"  ✓  runs: {args.runs}  warmup: {args.warmup}  threads: {thr}")
    print()

    mg_kw = dict(
        text_bin=text_bin, mgm_path=mgm_path, tokenizer_json=tokenizer_json,
        prompt=args.prompt, max_new_tokens=args.max_new_tokens, threads=args.threads,
    )
    ll_kw = dict(
        llama_cli=llama_cli, gguf_path=gguf_path, prompt=args.prompt,
        max_new_tokens=args.max_new_tokens, threads=args.threads,
        ctx_size=args.ctx_size, timeout_sec=args.llama_timeout,
    )

    # warmup
    for i in range(args.warmup):
        print(f"  warmup {i+1}/{args.warmup}...", end=" ", flush=True)
        try:
            run_microgemm(**mg_kw)
            print("mg ✓", end="  ", flush=True)
        except RuntimeError as e:
            return _die(f"MicroGemm warmup failed:\n  {e}")
        try:
            run_llamacpp(**ll_kw)
            print("llama ✓")
        except RuntimeError as e:
            return _die(f"llama.cpp warmup failed:\n  {e}")

    # timed runs
    mg_runs: List[Dict[str, float]] = []
    ll_runs: List[Dict[str, float]] = []

    for i in range(args.runs):
        print(f"  run {i+1}/{args.runs}  ", end="", flush=True)
        mg = run_microgemm(**mg_kw)
        mg_runs.append(mg)
        print(f"mg: {mg['decode_tps']:6.2f} t/s", end="  ", flush=True)

        ll = run_llamacpp(**ll_kw)
        ll_runs.append(ll)
        print(f"llama: {ll['decode_tps']:6.2f} t/s")

    # report
    print_report(mg_runs, ll_runs, args)

    if args.json:
        export_json(mg_runs, ll_runs, args, args.json)

    return 0


def _die(msg: str) -> int:
    print(f"\n  ✗  {msg}\n", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

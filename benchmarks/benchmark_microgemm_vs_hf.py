"""
Compare MicroGemm CPU generation against Hugging Face on the same local model.

This script is intentionally outside `microgemm/` so the core project keeps its
native-only constraint. The benchmark harness may use Python + torch +
transformers purely for comparison.

Example (Colab/Linux):

    python benchmarks/benchmark_microgemm_vs_hf.py \
        --model-dir /root/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac \
        --mgm-path microgemm/out/model.mgm \
        --prompt "Explique por que o ceu parece azul em uma frase curta." \
        --max-new-tokens 32 \
        --threads 2 \
        --temperature 0.0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MicroGemm vs Hugging Face on CPU")
    parser.add_argument("--model-dir", required=True, help="Directory containing config.json, tokenizer.json, and model.safetensors")
    parser.add_argument("--mgm-path", required=True, help="Path to converted MicroGemm .mgm file")
    parser.add_argument("--prompt", required=True, help="Prompt text to benchmark")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for both Hugging Face and MicroGemm")
    parser.add_argument("--microgemm-text-bin", default="microgemm-text", help="Path to the microgemm-text binary")
    parser.add_argument("--tokenizer-json", default="", help="Optional tokenizer.json path override")
    parser.add_argument("--warmup-new-tokens", type=int, default=8, help="Warmup generation length")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warmup runs")
    return parser.parse_args()


def resolve_path(raw: str, base: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def parse_microgemm_metrics(stdout: str) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    current_key = None
    current_lines: List[str] = []

    for line in stdout.splitlines():
        if current_key in {"generated_text", "full_text"}:
            if ": " in line and not line.startswith(" "):
                metrics[current_key] = "\n".join(current_lines).rstrip()
                current_key = None
                current_lines = []
            else:
                current_lines.append(line)
                continue

        if ": " not in line:
            continue

        key, value = line.split(": ", 1)
        if key in {"generated_text", "full_text"}:
            current_key = key
            current_lines = [value]
        else:
            metrics[key] = value

    if current_key is not None:
        metrics[current_key] = "\n".join(current_lines).rstrip()
    return metrics


def run_microgemm(
    microgemm_text_bin: Path,
    mgm_path: Path,
    tokenizer_json: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    threads: int,
) -> Dict[str, str]:
    cmd = [
        str(microgemm_text_bin),
        "generate",
        str(mgm_path),
        str(tokenizer_json),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--top-p",
        str(top_p),
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    if threads > 0:
        env["OMP_NUM_THREADS"] = str(threads)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "microgemm-text failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return parse_microgemm_metrics(result.stdout)


def sample_from_logits_torch(logits, temperature: float, top_k: int, top_p: float, generator):
    import torch

    logits = logits.to(torch.float32)
    if temperature <= 0.0 or top_k == 1:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k > 0 and top_k < logits.shape[-1]:
        values, indices = torch.topk(logits, k=top_k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        candidate_indices = indices
    else:
        probs = torch.softmax(logits, dim=-1)
        candidate_indices = None

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[0] = True
        kept_probs = sorted_probs[keep]
        kept_idx = sorted_idx[keep]
        kept_probs = kept_probs / kept_probs.sum()
        sampled_local = torch.multinomial(kept_probs, 1, generator=generator)
        chosen = kept_idx[sampled_local]
    else:
        sampled_local = torch.multinomial(probs, 1, generator=generator)
        chosen = sampled_local

    if candidate_indices is not None:
        chosen = candidate_indices[chosen]
    return int(chosen.item())


def run_hf_cpu(
    model_dir: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    threads: int,
) -> Dict[str, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if threads > 0:
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(threads)
        except RuntimeError:
            pass

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    model.to("cpu")

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to("cpu")
    prompt_token_count = int(input_ids.shape[1])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    with torch.inference_mode():
        prefill_start = time.perf_counter()
        outputs = model(input_ids=input_ids, use_cache=True)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

        logits = outputs.logits[0, -1, :]
        past_key_values = outputs.past_key_values
        generated_ids: List[int] = []

        first_token = sample_from_logits_torch(logits, temperature, top_k, top_p, generator)
        generated_ids.append(first_token)
        current = torch.tensor([[first_token]], dtype=input_ids.dtype)

        decode_start = time.perf_counter()
        for _ in range(1, max_new_tokens):
            outputs = model(input_ids=current, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values
            next_token = sample_from_logits_torch(logits, temperature, top_k, top_p, generator)
            generated_ids.append(next_token)
            current = torch.tensor([[next_token]], dtype=input_ids.dtype)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0

    total_ms = prefill_ms + decode_ms
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_ids = input_ids[0].tolist() + generated_ids
    full_text = tokenizer.decode(full_ids, skip_special_tokens=True)

    prefill_tps = (prompt_token_count * 1000.0 / prefill_ms) if prefill_ms > 0 else 0.0
    decode_tps = ((len(generated_ids) - 1) * 1000.0 / decode_ms) if decode_ms > 0 and len(generated_ids) > 1 else 0.0
    total_tps = ((prompt_token_count + len(generated_ids)) * 1000.0 / total_ms) if total_ms > 0 else 0.0

    return {
        "prompt_token_count": str(prompt_token_count),
        "generated_token_count": str(len(generated_ids)),
        "temperature": f"{temperature:.4f}",
        "top_k": str(top_k),
        "top_p": f"{top_p:.4f}",
        "seed": str(seed),
        "prefill_ms": f"{prefill_ms:.3f}",
        "decode_ms": f"{decode_ms:.3f}",
        "total_ms": f"{total_ms:.3f}",
        "prefill_tps": f"{prefill_tps:.3f}",
        "decode_tps": f"{decode_tps:.3f}",
        "total_tps": f"{total_tps:.3f}",
        "generated_text": generated_text,
        "full_text": full_text,
    }


def maybe_warmup(
    args: argparse.Namespace,
    repo_root: Path,
    microgemm_text_bin: Path,
    mgm_path: Path,
    tokenizer_json: Path,
    model_dir: Path,
) -> None:
    if args.skip_warmup or args.warmup_new_tokens <= 0:
        return

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
    run_hf_cpu(
        model_dir=model_dir,
        prompt=args.prompt,
        max_new_tokens=args.warmup_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        threads=args.threads,
    )


def print_side_by_side(mg: Dict[str, str], hf: Dict[str, str]) -> None:
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

    print("=" * 78)
    print("MicroGemm vs Hugging Face (CPU)")
    print("=" * 78)
    print(f"{'metric':<22} {'microgemm':>18} {'huggingface':>18} {'ratio mg/hf':>16}")
    print("-" * 78)
    for key in keys:
        mg_value = mg.get(key, "")
        hf_value = hf.get(key, "")
        ratio_str = ""
        try:
            mg_num = float(mg_value)
            hf_num = float(hf_value)
            ratio = (mg_num / hf_num) if hf_num != 0.0 else float("inf")
            ratio_str = f"{ratio:.3f}x"
        except ValueError:
            ratio_str = ""
        print(f"{key:<22} {mg_value:>18} {hf_value:>18} {ratio_str:>16}")
    print("-" * 78)
    print("MicroGemm text:")
    print(mg.get("generated_text", "").strip())
    print("-" * 78)
    print("Hugging Face text:")
    print(hf.get("generated_text", "").strip())
    print("=" * 78)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    model_dir = resolve_path(args.model_dir, repo_root)
    mgm_path = resolve_path(args.mgm_path, repo_root)
    microgemm_text_bin = resolve_path(args.microgemm_text_bin, repo_root)
    tokenizer_json = resolve_path(args.tokenizer_json, repo_root) if args.tokenizer_json else (model_dir / "tokenizer.json")

    if not model_dir.exists():
        raise SystemExit(f"--model-dir not found: {model_dir}")
    if not mgm_path.exists():
        raise SystemExit(f"--mgm-path not found: {mgm_path}")
    if not microgemm_text_bin.exists():
        raise SystemExit(f"--microgemm-text-bin not found: {microgemm_text_bin}")
    if not tokenizer_json.exists():
        raise SystemExit(f"tokenizer.json not found: {tokenizer_json}")

    maybe_warmup(args, repo_root, microgemm_text_bin, mgm_path, tokenizer_json, model_dir)

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
    hf = run_hf_cpu(
        model_dir=model_dir,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        threads=args.threads,
    )
    print_side_by_side(mg, hf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

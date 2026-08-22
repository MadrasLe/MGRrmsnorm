"""
Compare Qwen 3.5 native vs reference backend logits on the first token.

Usage:
    python examples/compare_qwen35_backends.py --model Qwen/Qwen3.5-0.8B --prompt "Explain linear attention."
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from megagemm.engine import BlockManager
from megagemm.models.loader import load_from_hf


def topk(tokenizer, logits: torch.Tensor, k: int = 10):
    values, indices = torch.topk(logits.float(), k)
    return [
        (int(idx), repr(tokenizer.decode([int(idx)])), round(float(val), 4))
        for idx, val in zip(indices, values)
    ]


def build_block_manager(config, dtype, device):
    return BlockManager(
        num_layers=config.num_hidden_layers,
        num_blocks=128,
        block_size=16,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=dtype,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="Explique atencao linear em 4 frases.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
    positions = torch.arange(input_ids.shape[1], device=args.device).unsqueeze(0)

    native = load_from_hf(args.model, dtype=dtype, device=args.device)
    reference = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
        low_cpu_mem_usage=True,
    ).eval()

    native_bm = build_block_manager(native.config, dtype, args.device)
    native_bm.allocate_sequence(1, input_ids.shape[1] + 8)

    with torch.inference_mode():
        native_logits = native.prefill(input_ids, positions, native_bm, 1)[0, -1].float()
        reference_logits = reference(input_ids=input_ids, use_cache=True, return_dict=True).logits[0, -1].float()

    diff = (native_logits - reference_logits).abs()
    print("native_top10:", topk(tokenizer, native_logits))
    print("reference_top10:", topk(tokenizer, reference_logits))
    print("max_abs_diff:", round(float(diff.max()), 6))
    print("mean_abs_diff:", round(float(diff.mean()), 6))


if __name__ == "__main__":
    main()

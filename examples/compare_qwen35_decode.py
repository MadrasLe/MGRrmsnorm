"""
Compare one cached Qwen 3.5 native decode step against Hugging Face.

Usage:
    python examples/compare_qwen35_decode.py --model Qwen/Qwen3.5-0.8B --prompt "Explique atencao linear em 4 frases."
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


def hf_decode_step(hf_model, next_ids, prefill_out, next_positions):
    kwargs = {
        "input_ids": next_ids,
        "past_key_values": prefill_out.past_key_values,
        "use_cache": True,
        "return_dict": True,
    }
    try:
        return hf_model(
            **kwargs,
            position_ids=next_positions,
            cache_position=next_positions.view(-1),
        )
    except TypeError:
        try:
            return hf_model(**kwargs, position_ids=next_positions)
        except TypeError:
            return hf_model(**kwargs)


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
    hf = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
        low_cpu_mem_usage=True,
    ).eval()

    bm = build_block_manager(native.config, dtype, args.device)
    bm.allocate_sequence(1, input_ids.shape[1] + 8)

    with torch.inference_mode():
        native_prefill = native.prefill(input_ids, positions, bm, 1)

    with torch.no_grad():
        hf_prefill = hf(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

    prefill_diff = (native_prefill[:, -1, :].float() - hf_prefill.logits[:, -1, :].float()).abs()
    print("prefill_max_abs_diff:", round(float(prefill_diff.max().detach()), 6))
    print("prefill_mean_abs_diff:", round(float(prefill_diff.mean().detach()), 6))

    next_ids = torch.argmax(hf_prefill.logits[:, -1, :], dim=-1, keepdim=True)
    next_positions = torch.tensor([[input_ids.shape[1]]], device=args.device, dtype=torch.long)
    print("decode_input_token:", int(next_ids.item()), repr(tokenizer.decode([int(next_ids.item())])))

    with torch.inference_mode():
        native_decode = native.decode_step(next_ids, next_positions, bm, [1])

    with torch.no_grad():
        hf_decode = hf_decode_step(hf, next_ids, hf_prefill, next_positions)

    native_logits = native_decode[:, -1, :].float()
    hf_logits = hf_decode.logits[:, -1, :].float()
    decode_diff = (native_logits - hf_logits).abs()

    print("decode_native_finite:", bool(torch.isfinite(native_logits).all()))
    print("decode_hf_finite:", bool(torch.isfinite(hf_logits).all()))
    print("decode_max_abs_diff:", round(float(decode_diff.max().detach()), 6))
    print("decode_mean_abs_diff:", round(float(decode_diff.mean().detach()), 6))
    print("native_top10:", topk(tokenizer, native_logits[0]))
    print("hf_top10:", topk(tokenizer, hf_logits[0]))
    print("native_seq_len:", bm.seq_lens[1])


if __name__ == "__main__":
    main()

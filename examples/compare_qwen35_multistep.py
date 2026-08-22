"""
Compare Qwen 3.5 native decode_multi_step vs repeated decode_step.

Both paths start from the exact same prefill cache snapshot.

Usage:
    python examples/compare_qwen35_multistep.py --model Qwen/Qwen3.5-0.8B --steps 8
"""

import argparse

import torch
from transformers import AutoTokenizer

from megagemm.engine import BlockManager
from megagemm.models.loader import load_from_hf


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
    parser.add_argument(
        "--prompt",
        default="Explique linear attention em transformers, comparando com softmax attention em 4 frases curtas e tecnicas.",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
    positions = torch.arange(input_ids.shape[1], device=args.device).unsqueeze(0)

    model = load_from_hf(args.model, dtype=dtype, device=args.device)

    bm_prefill = build_block_manager(model.config, dtype, args.device)
    bm_prefill.allocate_sequence(1, input_ids.shape[1] + args.steps + 8)

    with torch.inference_mode():
        prefill_logits = model.prefill(input_ids, positions, bm_prefill, 1)

    first_token = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    first_pos = torch.tensor([[input_ids.shape[1]]], device=args.device, dtype=torch.long)
    print("first_token:", int(first_token.item()), repr(tokenizer.decode([int(first_token.item())])))

    snapshot = bm_prefill.serialize_sequence(1)

    bm_step = build_block_manager(model.config, dtype, args.device)
    bm_step.deserialize_sequence(1, snapshot, extra_tokens=args.steps + 8)

    step_input = first_token.clone()
    step_pos = first_pos.clone()
    step_tokens = []
    step_final_logits = None

    with torch.inference_mode():
        for _ in range(args.steps):
            step_final_logits = model.decode_step(step_input, step_pos, bm_step, [1])
            next_token = step_final_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            step_tokens.append(int(next_token.item()))
            step_input = next_token
            step_pos += 1

    bm_multi = build_block_manager(model.config, dtype, args.device)
    bm_multi.deserialize_sequence(1, snapshot, extra_tokens=args.steps + 8)

    with torch.inference_mode():
        multi_tokens, multi_final_logits = model.decode_multi_step(
            first_token, first_pos, bm_multi, [1], num_steps=args.steps,
        )

    multi_token_list = [int(tok) for tok in multi_tokens[0].tolist()]
    mismatch_count = sum(int(a != b) for a, b in zip(step_tokens, multi_token_list))
    final_diff = (
        step_final_logits[:, -1, :].float() - multi_final_logits[:, -1, :].float()
    ).abs()

    print("mismatch_count:", mismatch_count)
    print("step_trace:", tokenizer.decode(step_tokens, skip_special_tokens=False))
    print("multi_trace:", tokenizer.decode(multi_token_list, skip_special_tokens=False))
    print("final_max_abs_diff:", round(float(final_diff.max().detach()), 6))
    print("final_mean_abs_diff:", round(float(final_diff.mean().detach()), 6))
    print("step_seq_len:", bm_step.seq_lens[1])
    print("multi_seq_len:", bm_multi.seq_lens[1])


if __name__ == "__main__":
    main()

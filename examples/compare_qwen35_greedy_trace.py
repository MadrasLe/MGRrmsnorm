"""
Compare Qwen 3.5 native vs Hugging Face over multiple greedy decode steps.

This uses teacher forcing from the Hugging Face greedy token at each step so
both caches stay aligned and the per-step logit diff remains meaningful.

Usage:
    python examples/compare_qwen35_greedy_trace.py --model Qwen/Qwen3.5-0.8B --prompt "Explique linear attention em transformers."
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser.add_argument(
        "--prompt",
        default="Explique linear attention em transformers, comparando com softmax attention em 4 frases curtas e tecnicas.",
    )
    parser.add_argument("--steps", type=int, default=32)
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
    bm.allocate_sequence(1, input_ids.shape[1] + args.steps + 8)

    with torch.inference_mode():
        native_out = native.prefill(input_ids, positions, bm, 1)

    with torch.no_grad():
        hf_out = hf(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

    native_logits = native_out[:, -1, :].float()
    hf_logits = hf_out.logits[:, -1, :].float()
    prompt_len = input_ids.shape[1]
    mismatch_steps = 0
    teacher_tokens = []
    native_tokens = []

    for step in range(args.steps):
        diff = (native_logits - hf_logits).abs()
        native_pred = int(native_logits.argmax(dim=-1).item())
        hf_pred = int(hf_logits.argmax(dim=-1).item())
        teacher_tokens.append(hf_pred)
        native_tokens.append(native_pred)
        mismatch = native_pred != hf_pred
        mismatch_steps += int(mismatch)

        print(
            f"step={step:02d} mismatch={mismatch} "
            f"native={native_pred}:{repr(tokenizer.decode([native_pred]))} "
            f"hf={hf_pred}:{repr(tokenizer.decode([hf_pred]))} "
            f"max_abs_diff={float(diff.max().detach()):.6f} "
            f"mean_abs_diff={float(diff.mean().detach()):.6f}"
        )

        if step == args.steps - 1:
            break

        next_ids = torch.tensor([[hf_pred]], device=args.device, dtype=torch.long)
        next_positions = torch.tensor([[prompt_len + step]], device=args.device, dtype=torch.long)

        with torch.inference_mode():
            native_decode = native.decode_step(next_ids, next_positions, bm, [1])
            native_logits = native_decode[:, -1, :].float()

        with torch.no_grad():
            hf_out = hf_decode_step(hf, next_ids, hf_out, next_positions)
            hf_logits = hf_out.logits[:, -1, :].float()

    print("mismatch_steps:", mismatch_steps)
    print("native_trace:", tokenizer.decode(native_tokens, skip_special_tokens=False))
    print("hf_trace:", tokenizer.decode(teacher_tokens, skip_special_tokens=False))


if __name__ == "__main__":
    main()

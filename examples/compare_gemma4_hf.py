"""
Compare Gemma 4 text-only MegaGEMM against Hugging Face.

The script avoids holding both models in VRAM at once:
1. Load HF, record prompt/decode logits and greedy teacher tokens on CPU.
2. Delete HF and clear CUDA cache.
3. Load MegaGEMM, replay the same teacher tokens, and compare logits.

Usage:
    python examples/compare_gemma4_hf.py --model google/gemma-4-E2B-it --dtype bfloat16 --steps 8
"""

import argparse
import gc

import torch
from transformers import AutoTokenizer

from megagemm.engine import BlockManager
from megagemm.models.loader import load_from_hf


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _load_hf_model(model_name: str, dtype: torch.dtype, device: str):
    import transformers

    errors = []
    for class_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        model_cls = getattr(transformers, class_name, None)
        if model_cls is None:
            continue
        try:
            return model_cls.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device,
                low_cpu_mem_usage=True,
            ).eval()
        except Exception as exc:
            errors.append(f"{class_name}: {exc}")
    raise RuntimeError("Could not load HF Gemma 4 reference model:\n" + "\n".join(errors))


def _hf_decode_step(hf_model, next_ids, prev_out, next_positions):
    kwargs = {
        "input_ids": next_ids,
        "past_key_values": prev_out.past_key_values,
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


def _build_block_manager(config, dtype, device, total_tokens):
    kv_sources = {
        layer_idx: source_idx
        for layer_idx, source_idx in enumerate(getattr(config, "kv_share_sources", []) or [])
        if source_idx is not None
    }
    num_blocks = max(64, (total_tokens + 15) // 16 + 8)
    return BlockManager(
        num_layers=config.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=16,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=dtype,
        device=device,
        kv_layer_indices=getattr(config, "kv_cache_layer_indices", None),
        per_layer_num_kv_heads=getattr(config, "per_layer_num_kv_heads", None),
        per_layer_head_dims=getattr(config, "per_layer_head_dims", None),
        kv_layer_sources=kv_sources,
    )


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).mean().item())


def _summarize_step(tokenizer, step, native_logits, hf_logits):
    diff = (native_logits.float() - hf_logits.float()).abs()
    native_top = int(native_logits.argmax(dim=-1).item())
    hf_top = int(hf_logits.argmax(dim=-1).item())
    print(
        f"step={step:02d} top1_match={native_top == hf_top} "
        f"native={native_top}:{repr(tokenizer.decode([native_top]))} "
        f"hf={hf_top}:{repr(tokenizer.decode([hf_top]))} "
        f"max_abs={float(diff.max().item()):.6f} "
        f"mean_abs={float(diff.mean().item()):.6f} "
        f"cos={_cosine(native_logits, hf_logits):.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--prompt", default="Explique KV cache em uma frase.")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dtype = _dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = args.prompt
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
    positions = torch.arange(input_ids.shape[1], device=args.device).unsqueeze(0)

    print(f"Loading HF reference: {args.model}")
    hf_model = _load_hf_model(args.model, dtype, args.device)
    hf_logits = []
    hf_tokens = []
    with torch.inference_mode():
        hf_out = hf_model(input_ids=input_ids, use_cache=True, return_dict=True)
        logits = hf_out.logits[:, -1, :].detach().cpu()
        hf_logits.append(logits)
        for step in range(args.steps):
            token = int(logits.argmax(dim=-1).item())
            hf_tokens.append(token)
            if step == args.steps - 1:
                break
            next_ids = torch.tensor([[token]], device=args.device, dtype=torch.long)
            next_pos = torch.tensor([[input_ids.shape[1] + step]], device=args.device, dtype=torch.long)
            hf_out = _hf_decode_step(hf_model, next_ids, hf_out, next_pos)
            logits = hf_out.logits[:, -1, :].detach().cpu()
            hf_logits.append(logits)

    del hf_model, hf_out
    gc.collect()
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading MegaGEMM candidate")
    native = load_from_hf(args.model, dtype=dtype, device=args.device)
    block_manager = _build_block_manager(
        native.config,
        dtype,
        args.device,
        total_tokens=int(input_ids.shape[1]) + args.steps + 8,
    )
    block_manager.allocate_sequence(1, int(input_ids.shape[1]) + args.steps + 8)

    with torch.inference_mode():
        native_out = native.prefill(input_ids, positions, block_manager, 1)
        native_logits = native_out[:, -1, :].detach().cpu()
        _summarize_step(tokenizer, 0, native_logits, hf_logits[0])

        for step, token in enumerate(hf_tokens[:-1]):
            next_ids = torch.tensor([[token]], device=args.device, dtype=torch.long)
            next_pos = torch.tensor([[input_ids.shape[1] + step]], device=args.device, dtype=torch.long)
            native_out = native.decode_step(next_ids, next_pos, block_manager, [1])
            native_logits = native_out[:, -1, :].detach().cpu()
            _summarize_step(tokenizer, step + 1, native_logits, hf_logits[step + 1])

    print("hf_teacher_tokens:", hf_tokens)
    print("hf_teacher_text:", tokenizer.decode(hf_tokens, skip_special_tokens=False))


if __name__ == "__main__":
    main()

"""
Compare Qwen 3.5 native layer-by-layer probe logits against Hugging Face.

Usage:
    python examples/compare_qwen35_layers.py --model Qwen/Qwen3.5-0.8B --prompt "Explique atencao linear em 4 frases."
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from megagemm.engine import BlockManager
from megagemm.models.loader import load_from_hf


def _hf_probe_logits(hf_model, hidden_state, already_normed: bool = False):
    hidden_state = hidden_state.detach().clone()
    with torch.no_grad():
        if not already_normed:
            hidden_state = hf_model.model.norm(hidden_state)
        return hf_model.lm_head(hidden_state)


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

    bm = BlockManager(
        num_layers=native.config.num_hidden_layers,
        num_blocks=128,
        block_size=16,
        num_kv_heads=native.config.num_key_value_heads,
        head_dim=native.config.head_dim,
        dtype=dtype,
        device=args.device,
    )
    bm.allocate_sequence(1, input_ids.shape[1] + 8)

    with torch.inference_mode():
        native_logits, native_probes = native.prefill(
            input_ids, positions, bm, 1, logit_lens=True,
        )
        hf_out = hf(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    print("final_native_finite:", bool(torch.isfinite(native_logits).all()))
    print("final_hf_finite:", bool(torch.isfinite(hf_out.logits).all()))
    final_diff = (native_logits[:, -1, :].float() - hf_out.logits[:, -1, :].float()).abs()
    print("final_max_abs_diff:", round(float(final_diff.max().detach()), 6))
    print("final_mean_abs_diff:", round(float(final_diff.mean().detach()), 6))

    first_bad_layer = None
    num_layers = native.config.num_hidden_layers
    for layer_idx in sorted(native_probes.keys()):
        hf_hidden = hf_out.hidden_states[layer_idx + 1]
        hf_probe = _hf_probe_logits(
            hf,
            hf_hidden,
            already_normed=(layer_idx == num_layers - 1),
        )[:, -1, :].float()
        native_probe = native_probes[layer_idx].unsqueeze(0).float()

        native_finite = bool(torch.isfinite(native_probe).all())
        hf_finite = bool(torch.isfinite(hf_probe).all())
        max_abs = (
            float((native_probe - hf_probe).abs().max().detach())
            if native_finite and hf_finite else float("nan")
        )
        mean_abs = (
            float((native_probe - hf_probe).abs().mean().detach())
            if native_finite and hf_finite else float("nan")
        )

        print(
            f"layer={layer_idx:02d} native_finite={native_finite} hf_finite={hf_finite} "
            f"max_abs_diff={max_abs:.6f} mean_abs_diff={mean_abs:.6f}"
        )

        if first_bad_layer is None and (not native_finite or max_abs > 0.1):
            first_bad_layer = layer_idx

    print("first_bad_layer:", first_bad_layer)


if __name__ == "__main__":
    main()

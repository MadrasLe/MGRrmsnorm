"""
Multi-Model Smoke Test for MegaGemm Engine.
Tests generation quality across LLaMA-family models.

WARNING: LLaMA 3.x models are GATED. You need:
  1. Accept license at https://huggingface.co/meta-llama
  2. Run: huggingface-cli login

WARNING: 8B FP16 needs ~16GB VRAM. Skip if on T4 (15GB).
"""

# Standalone gated-model smoke runner; execute this file directly on a GPU.
__test__ = False

import torch
import gc, sys, os, time, traceback
sys.path.insert(0, os.getcwd())

from transformers import AutoTokenizer

def test_model(model_name, prompts, max_tokens=50, num_blocks=512):
    """Test a single model with given prompts."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    try:
        from megagemm.engine import InferenceEngine

        engine = InferenceEngine(
            model_name,
            num_blocks=num_blocks,
            block_size=16,
        )

        # Print model config
        cfg = engine.model.config
        print(f"  Model type: {cfg.model_type}")
        print(f"  Layers: {cfg.num_hidden_layers}, Hidden: {cfg.hidden_size}")
        print(f"  Heads: {cfg.num_attention_heads}, KV heads: {cfg.num_key_value_heads}")
        print(f"  Head dim: {cfg.head_dim}")
        print(f"  RoPE theta: {cfg.rope_theta}")
        print(f"  rope_half_rotate: {cfg.rope_half_rotate}")
        print(f"  attention_bias: {cfg.attention_bias}")
        print(f"  tie_word_embeddings: {cfg.tie_word_embeddings}")

        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"  VRAM: {vram:.1f}GB")

        # Format prompts with chat template
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for prompt in prompts:
            # Try using HF chat template
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback: raw prompt
                formatted = prompt

            # Generate
            output = engine.generate(
                formatted,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1,
            )

            print(f"\n  Q: {prompt}")
            print(f"  A: {output[:200]}")

        # Cleanup
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n  Status: OK")
        return True

    except Exception as e:
        print(f"\n  ERROR: {e}")
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("MegaGemm Multi-Model Smoke Test")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name()
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu} ({vram_total:.1f}GB)")
    else:
        print("No GPU! Exiting.")
        sys.exit(1)

    prompts = [
        "What is the capital of France?",
        "Explain gradient descent in one sentence.",
    ]

    results = {}

    # ── TinyLlama 1.1B ──
    results["TinyLlama-1.1B"] = test_model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        prompts, max_tokens=60, num_blocks=2048,
    )

    # ── LLaMA 3.2 1B ──
    gc.collect(); torch.cuda.empty_cache()
    results["Llama-3.2-1B"] = test_model(
        "meta-llama/Llama-3.2-1B-Instruct",
        prompts, max_tokens=60, num_blocks=2048,
    )

    # ── LLaMA 3.2 3B ──
    gc.collect(); torch.cuda.empty_cache()
    results["Llama-3.2-3B"] = test_model(
        "meta-llama/Llama-3.2-3B-Instruct",
        prompts, max_tokens=60, num_blocks=1024,
    )

    # ── LLaMA 3.1 8B (only if enough VRAM) ──
    gc.collect(); torch.cuda.empty_cache()
    if vram_total >= 18:
        results["Llama-3.1-8B"] = test_model(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            prompts, max_tokens=60, num_blocks=256,  # small KV cache to fit in VRAM
        )
    else:
        print(f"\n⚠️  Skipping Llama-3.1-8B: need 18GB+ VRAM, have {vram_total:.1f}GB")
        results["Llama-3.1-8B"] = "SKIPPED"

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = "✅" if status is True else "⚠️ SKIP" if status == "SKIPPED" else "❌"
        print(f"  {icon} {name}")
    print()

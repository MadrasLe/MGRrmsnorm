"""
🔒 Deterministic Mode — Proof of Concept
==========================================
This script PROVES deterministic mode works by:
1. Running the SAME prompt 3x WITHOUT deterministic → outputs DIFFER
2. Running the SAME prompt 3x WITH deterministic → outputs are IDENTICAL

Run on GPU with a real model:
    python examples/proof_deterministic.py
    python examples/proof_deterministic.py --model Qwen/Qwen3-4B

Author: Gabriel Yogi
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", default="Explain what gravity is in one sentence.")
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    from megagemm.engine import InferenceEngine

    print("=" * 70)
    print("🔓 PART 1: WITHOUT deterministic mode (outputs should DIFFER)")
    print("=" * 70)

    outputs_normal = []
    for i in range(args.runs):
        engine = InferenceEngine(
            args.model,
            dtype=torch.float16,
            deterministic=False,  # ← default, non-deterministic
        )
        out = engine.generate(args.prompt, max_new_tokens=args.tokens, temperature=0.7, top_k=50)
        outputs_normal.append(out)
        print(f"\n  Run {i+1}: {out[:100]}...")
        del engine
        torch.cuda.empty_cache()

    normal_all_same = all(o == outputs_normal[0] for o in outputs_normal)
    print(f"\n  All {args.runs} outputs identical? → {normal_all_same}")
    if not normal_all_same:
        print("  ✅ EXPECTED: outputs differ without deterministic mode")
    else:
        print("  ⚠️  Outputs happened to be the same (can happen with greedy-like sampling)")

    print("\n" + "=" * 70)
    print("🔒 PART 2: WITH deterministic mode (outputs should be IDENTICAL)")
    print("=" * 70)

    outputs_determ = []
    for i in range(args.runs):
        engine = InferenceEngine(
            args.model,
            dtype=torch.float16,
            deterministic=True,   # ← DETERMINISTIC
            seed=42,
        )
        out = engine.generate(args.prompt, max_new_tokens=args.tokens, temperature=0.7, top_k=50)
        outputs_determ.append(out)
        print(f"\n  Run {i+1}: {out[:100]}...")
        del engine
        torch.cuda.empty_cache()

    determ_all_same = all(o == outputs_determ[0] for o in outputs_determ)
    print(f"\n  All {args.runs} outputs identical? → {determ_all_same}")

    if determ_all_same:
        print("  ✅ PROVEN: deterministic mode guarantees bit-exact output!")
    else:
        print("  ❌ FAIL: outputs differ even with deterministic mode!")
        for i, o in enumerate(outputs_determ):
            print(f"    Run {i+1}: {repr(o[:80])}")

    # === PART 3: Greedy (temperature=0) comparison ===
    print("\n" + "=" * 70)
    print("🧊 PART 3: Greedy (temp=0) + deterministic — should be IDENTICAL")
    print("=" * 70)

    outputs_greedy = []
    for i in range(args.runs):
        engine = InferenceEngine(
            args.model,
            dtype=torch.float16,
            deterministic=True,
            seed=42,
        )
        out = engine.generate(args.prompt, max_new_tokens=args.tokens, temperature=0.0)
        outputs_greedy.append(out)
        print(f"\n  Run {i+1}: {out[:100]}...")
        del engine
        torch.cuda.empty_cache()

    greedy_all_same = all(o == outputs_greedy[0] for o in outputs_greedy)
    print(f"\n  All {args.runs} outputs identical? → {greedy_all_same}")
    if greedy_all_same:
        print("  ✅ PROVEN: greedy + deterministic = perfectly reproducible!")
    else:
        print("  ❌ FAIL!")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Non-deterministic (temp=0.7): outputs differ? {'YES ✅' if not normal_all_same else 'NO (coincidence)'}")
    print(f"  Deterministic (temp=0.7):     outputs match?  {'YES ✅' if determ_all_same else 'NO ❌'}")
    print(f"  Deterministic (greedy):       outputs match?  {'YES ✅' if greedy_all_same else 'NO ❌'}")


if __name__ == "__main__":
    main()

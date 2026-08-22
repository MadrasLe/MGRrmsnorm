"""
🔍 XAI + Monitoring Demo for MegaGemm
==========================================
Test script to validate XAI, monitoring, and dashboard features.

Usage:
    # Quick test (FP16 only)
    python examples/test_xai_demo.py

    # With Logit Lens
    python examples/test_xai_demo.py --logit-lens

    # With Monitoring dashboard
    python examples/test_xai_demo.py --monitor

    # With Live Dashboard (opens http://localhost:8080)
    python examples/test_xai_demo.py --dashboard

    # Full test (all modes)
    python examples/test_xai_demo.py --full

    # Quantized
    python examples/test_xai_demo.py --model Qwen/Qwen2.5-7B-Instruct --quantize int8

Author: Gabriel Yogi
"""

import argparse
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def run_xai_test(
    model_name: str,
    quantize: str = None,
    n_gpu_layers: int = -1,
    kv_offload: bool = False,
    logit_lens: bool = False,
    xai_top_k: int = 5,
    max_new_tokens: int = 50,
    output_dir: str = "xai_output",
    monitor: bool = False,
    dashboard: bool = False,
    dashboard_port: int = 8080,
):
    """Run XAI test with the given configuration."""
    from megagemm.engine import InferenceEngine

    # Build config description
    mode_parts = []
    if quantize:
        mode_parts.append(f"quantize={quantize}")
    if n_gpu_layers > 0:
        mode_parts.append(f"gpu_layers={n_gpu_layers}")
    if kv_offload:
        mode_parts.append("kv_offload=True")
    if logit_lens:
        mode_parts.append("logit_lens=True")
    mode_str = ", ".join(mode_parts) if mode_parts else "FP16"

    print(f"\n{'='*60}")
    print(f"🔍 XAI Test: {model_name}")
    print(f"   Mode: {mode_str}")
    print(f"{'='*60}")

    # Create engine
    engine_kwargs = {
        "model_name": model_name,
        "quantize": quantize,
    }
    if n_gpu_layers > 0:
        engine_kwargs["n_gpu_layers"] = n_gpu_layers
    if kv_offload:
        engine_kwargs["kv_offload"] = True
        engine_kwargs["num_cpu_blocks"] = 2048
        engine_kwargs["gpu_window"] = 32

    print("\n📦 Loading model...")
    t0 = time.perf_counter()
    engine = InferenceEngine(
        **engine_kwargs,
        monitor=monitor or dashboard,
        dashboard=dashboard,
        dashboard_port=dashboard_port,
    )
    t_load = time.perf_counter() - t0
    print(f"   Loaded in {t_load:.1f}s")

    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about programming.",
        "Who was the first president of the United States?",
        "What is the meaning of life?",
    ]

    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\n{'─'*60}")
        print(f"📝 Prompt {i+1}: {prompt}")
        print(f"{'─'*60}")

        t0 = time.perf_counter()
        text, report = engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            xai=True,
            xai_top_k=xai_top_k,
            logit_lens=logit_lens,
        )
        t_gen = time.perf_counter() - t0

        # Display results
        print(f"\n💬 Output: {text}")
        print(f"⏱️  Time: {t_gen:.2f}s")
        print(f"📊 Confidence: {report.confidence_score:.4f}")
        print(f"📈 Steps: {len(report.steps)}")

        # Show first 3 steps detail
        print(f"\n   Top-{xai_top_k} tokens (first 3 steps):")
        for step in report.steps[:3]:
            chosen = step.chosen
            print(f"   [{step.position:3d}] {chosen.token_str!r:<15s} p={chosen.probability:.4f}")
            for alt in step.top_k[:3]:
                if alt.token_id != chosen.token_id:
                    print(f"         {alt.token_str!r:<15s} p={alt.probability:.4f}")

            # Logit Lens
            if step.logit_lens:
                layers = sorted(step.logit_lens.keys())
                show = [layers[0], layers[len(layers)//2], layers[-1]]
                for lid in show:
                    top = step.logit_lens[lid][0]
                    print(f"         Layer {lid:2d} → {top.token_str!r:<12s} p={top.probability:.4f}")

        # Export
        safe_mode = mode_str.replace(" ", "").replace(",", "_").replace("=", "")
        json_path = os.path.join(output_dir, f"xai_test_{i+1}_{safe_mode}.json")
        txt_path = os.path.join(output_dir, f"xai_test_{i+1}_{safe_mode}.txt")

        report.to_json(json_path)
        report.to_txt(txt_path)
        print(f"\n   📄 Exported: {json_path}")
        print(f"   📄 Exported: {txt_path}")

    # Monitor summary
    if monitor or dashboard:
        print(f"\n{engine.monitor_summary()}")
        n_exported = engine.export_monitor_log(os.path.join(output_dir, "monitor_log.jsonl"))
        print(f"\n📁 Exported {n_exported} records to {output_dir}/monitor_log.jsonl")

        # Get stats as JSON too
        stats = engine.get_monitor_stats()
        stats_path = os.path.join(output_dir, "monitor_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"📁 Stats saved to {stats_path}")

    # Validate
    print(f"\n{'='*60}")
    print(f"✅ XAI test PASSED for mode: {mode_str}")
    print(f"   - All reports generated successfully")
    print(f"   - JSON/TXT exports saved to {output_dir}/")
    print(f"{'='*60}")

    # Keep alive if dashboard is running
    if dashboard:
        print(f"\n🌐 Dashboard running at http://localhost:{dashboard_port}")
        print("   Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            engine.stop_dashboard()
            print("\n📊 Dashboard stopped.")

    return True


def main():
    parser = argparse.ArgumentParser(description="MegaGemm XAI Demo")
    parser.add_argument(
        "--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--quantize", choices=["int8", None], default=None,
        help="Quantization mode"
    )
    parser.add_argument(
        "--gpu-layers", type=int, default=-1,
        help="Layers on GPU (-1=all)"
    )
    parser.add_argument(
        "--kv-offload", action="store_true",
        help="Enable KV cache CPU offloading"
    )
    parser.add_argument(
        "--logit-lens", action="store_true",
        help="Enable Logit Lens per-layer probing"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top-K tokens to capture"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50,
        help="Max tokens to generate per prompt"
    )
    parser.add_argument(
        "--output-dir", default="xai_output",
        help="Directory for output files"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full test suite (FP16 + INT8 + Logit Lens)"
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Enable inference monitoring (terminal dashboard)"
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Enable live HTML dashboard at http://localhost:8080"
    )
    parser.add_argument(
        "--dashboard-port", type=int, default=8080,
        help="Port for the live dashboard (default: 8080)"
    )

    args = parser.parse_args()

    if args.full:
        # Full test: multiple configurations
        configs = [
            {"quantize": None, "logit_lens": False},     # FP16 baseline
            {"quantize": None, "logit_lens": True},      # FP16 + Logit Lens
            {"quantize": "int8", "logit_lens": False},   # INT8
            {"quantize": "int8", "logit_lens": True},    # INT8 + Logit Lens
        ]

        print("🔥 Running FULL XAI test suite")
        print(f"   Model: {args.model}")
        print(f"   Configs: {len(configs)}")

        for cfg in configs:
            run_xai_test(
                model_name=args.model,
                xai_top_k=args.top_k,
                max_new_tokens=args.max_tokens,
                output_dir=args.output_dir,
                **cfg,
            )

        print(f"\n🎉 All {len(configs)} configurations passed!")
    else:
        # Single test
        run_xai_test(
            model_name=args.model,
            quantize=args.quantize,
            n_gpu_layers=args.gpu_layers,
            kv_offload=args.kv_offload,
            logit_lens=args.logit_lens,
            xai_top_k=args.top_k,
            max_new_tokens=args.max_tokens,
            output_dir=args.output_dir,
            monitor=args.monitor,
            dashboard=args.dashboard,
            dashboard_port=args.dashboard_port,
        )


if __name__ == "__main__":
    main()

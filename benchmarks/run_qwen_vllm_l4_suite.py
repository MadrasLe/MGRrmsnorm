"""Convenience entrypoint for Qwen-family vLLM baselines on one NVIDIA L4."""

from __future__ import annotations

from run_qwen_vllm_t4_suite import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_models="l4-core",
            default_hardware_label="1xl4",
            default_out_dir="bench_results/qwen_vllm_l4",
            default_run_prefix="qwen_vllm_l4",
        )
    )

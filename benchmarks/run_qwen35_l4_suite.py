"""Run the Qwen 3.5 MegaGemm-native benchmark suite on one NVIDIA L4."""

from __future__ import annotations

from run_qwen35_t4_suite import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_hardware_label="1xl4",
            default_out_dir="bench_results/qwen35_l4",
            default_run_prefix="qwen35_l4",
        )
    )

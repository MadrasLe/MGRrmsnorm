import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "benchmarks" / "run_gemma4_moe_vs_vllm_colab.sh"
BACKEND_BENCH = ROOT / "benchmarks" / "run_gemma4_moe_vs_vllm.py"
BATCH_HARNESS = ROOT / "benchmarks" / "run_gemma4_moe_batch_vs_vllm_colab.sh"
BATCH_BENCH = ROOT / "benchmarks" / "run_gemma4_moe_batch_vs_vllm.py"
EXPERT_PREFILL_TUNER = ROOT / "benchmarks" / "run_gemma4_expert_prefill_tune_colab.sh"
HOT_LAYER_BENCH = ROOT / "benchmarks" / "run_gemma4_hot_layer_microbench.py"
ENGINE = ROOT / "megagemm" / "engine" / "engine.py"
SCHEDULER = ROOT / "megagemm" / "engine" / "scheduler.py"
KV_CACHE = ROOT / "megagemm" / "engine" / "kv_cache.py"
LLAMA = ROOT / "megagemm" / "models" / "llama.py"
QWEN3_MOE = ROOT / "megagemm" / "kernels" / "qwen3_moe.py"
RMSNORM_TRITON = ROOT / "megagemm" / "kernels" / "rmsnorm_triton.py"
PAGED_ATTENTION = ROOT / "megagemm" / "kernels" / "paged_attention.py"
GROUPED_PREFILL = ROOT / "megagemm" / "kernels" / "gemma4_grouped_prefill.py"
GRAPH_PREFLIGHT = ROOT / "benchmarks" / "run_gemma4_b16_prefill_graph_preflight.py"
FULL_MODEL_GRAPH_PREFLIGHT = (
    ROOT / "benchmarks" / "run_gemma4_b16_full_model_graph_preflight.py"
)
FULL_MODEL_GRAPH_DIAGNOSE = (
    ROOT / "benchmarks" / "run_gemma4_b16_full_model_graph_diagnose.py"
)
FULL_MODEL_GRAPH_WRAPPER = (
    ROOT / "benchmarks" / "run_gemma4_b16_full_model_graph_preflight_colab.sh"
)
ACTIVE_LIST_GATE = (
    ROOT / "benchmarks" / "run_gemma4_active_list_early_exit_microbench.py"
)
VLLM_MOE_PARITY_GATE = (
    ROOT / "benchmarks" / "run_gemma4_vllm_moe_parity_microbench.py"
)
ROUTER_COMPACT_PACK_GATE = (
    ROOT / "benchmarks" / "run_gemma4_router_compact_pack_microbench.py"
)
ATTN_MOE_ROUTER_SINGLE_KERNEL_GATE = (
    ROOT / "benchmarks" / "run_gemma4_attn_moe_decode_bridge_microbench.py"
)
LONG_DECODE_ATTN_TUNER = (
    ROOT / "benchmarks" / "run_gemma4_long_decode_attention_tune.py"
)
LONG_DECODE_ATTN_WRAPPER = (
    ROOT / "benchmarks" / "run_gemma4_long_decode_attention_tune_colab.sh"
)
LONG_DECODE_ATTN_SHAPE_TUNER = (
    ROOT / "benchmarks" / "run_gemma4_long_decode_attention_shape_tune.py"
)
LONG_DECODE_ATTN_SHAPE_WRAPPER = (
    ROOT / "benchmarks" / "run_gemma4_long_decode_attention_shape_tune_colab.sh"
)
LONG_DECODE_ATTN_FRONTIER = (
    ROOT / "benchmarks" / "run_gemma4_long_decode_attention_frontier.py"
)
LONG_DECODE_ATTN_FRONTIER_WRAPPER = (
    ROOT / "benchmarks" / "run_gemma4_long_decode_attention_frontier_colab.sh"
)
LONG_DECODE_ATTN_VLLM_PARITY = (
    ROOT / "benchmarks" / "run_gemma4_vllm_attention_parity_microbench.py"
)
LONG_DECODE_ATTN_VLLM_PARITY_WRAPPER = (
    ROOT
    / "benchmarks"
    / "run_gemma4_long_decode_attention_vllm_parity_colab.sh"
)
LONG_CONTEXT_HARNESS = (
    ROOT / "benchmarks" / "run_gemma4_long_context_vs_vllm_colab.sh"
)
LONG_CONTEXT_BENCH = ROOT / "benchmarks" / "run_gemma4_long_context_vs_vllm.py"


class Gemma4ColabHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = HARNESS.read_text(encoding="utf-8")
        cls.benchmark = BACKEND_BENCH.read_text(encoding="utf-8")
        cls.batch_benchmark = BATCH_BENCH.read_text(encoding="utf-8")
        cls.engine = ENGINE.read_text(encoding="utf-8")
        cls.scheduler = SCHEDULER.read_text(encoding="utf-8")
        cls.kv_cache = KV_CACHE.read_text(encoding="utf-8")
        cls.llama = LLAMA.read_text(encoding="utf-8")
        cls.qwen3_moe = QWEN3_MOE.read_text(encoding="utf-8")
        cls.rmsnorm_triton = RMSNORM_TRITON.read_text(encoding="utf-8")
        cls.paged_attention = PAGED_ATTENTION.read_text(encoding="utf-8")
        cls.grouped_prefill = GROUPED_PREFILL.read_text(encoding="utf-8")
        cls.router_compact_pack_gate = ROUTER_COMPACT_PACK_GATE.read_text(
            encoding="utf-8"
        )
        cls.attn_moe_router_single_kernel_gate = (
            ATTN_MOE_ROUTER_SINGLE_KERNEL_GATE.read_text(encoding="utf-8")
        )
        cls.graph_preflight = GRAPH_PREFLIGHT.read_text(encoding="utf-8")
        cls.full_model_graph_preflight = FULL_MODEL_GRAPH_PREFLIGHT.read_text(
            encoding="utf-8"
        )
        cls.full_model_graph_diagnose = FULL_MODEL_GRAPH_DIAGNOSE.read_text(
            encoding="utf-8"
        )
        cls.full_model_graph_wrapper = FULL_MODEL_GRAPH_WRAPPER.read_text(
            encoding="utf-8"
        )

    def test_embedded_python_blocks_compile(self):
        blocks = re.findall(r"<<'PY'\r?\n(.*?)\r?\nPY", self.script, re.DOTALL)
        self.assertGreater(len(blocks), 0)
        for index, block in enumerate(blocks, start=1):
            compile(block, f"{HARNESS}:heredoc-{index}", "exec")

    def test_rmsnorm_uses_distinct_input_and_output_row_strides(self):
        kernel = self.rmsnorm_triton
        self.assertIn("x_stride_row,\n        out_stride_row,", kernel)
        self.assertIn("x_row_off = row * x_stride_row", kernel)
        self.assertIn("out_row_off = row * out_stride_row", kernel)
        self.assertIn(
            "x_2d.stride(0),\n        out_2d.stride(0),\n        eps,",
            kernel,
        )
        self.assertIn(
            "x_2d.stride(0),\n        out.stride(0),\n        eps,",
            kernel,
        )
        self.assertIn("GEMMA4_FINAL_NORM_STRIDE_PREFLIGHT", self.script)
        self.assertLess(
            self.script.index("GEMMA4_FINAL_NORM_STRIDE_PREFLIGHT"),
            self.script.index("== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES =="),
        )

    def test_gemma4_flat_decode_forwards_model_paged_attention_policy(self):
        flat_decode = self.llama[
            self.llama.index("    def _gemma4_flat_decode_layers(") :
            self.llama.index("    def _flat_decode_layers(")
        ]

        self.assertIn(
            'getattr(self.runtime_policy, "paged_decode_splits", 0)',
            flat_decode,
        )
        self.assertIn(
            'getattr(self.runtime_policy, "paged_decode_gqa2_direct", False)',
            flat_decode,
        )
        self.assertIn(
            'getattr(self.runtime_policy, "paged_decode_warps_h256", 0)',
            flat_decode,
        )
        self.assertIn(
            "split_policy_override=paged_decode_splits or None",
            flat_decode,
        )
        self.assertIn(
            "gqa2_direct_policy_enabled=paged_decode_gqa2_direct",
            flat_decode,
        )
        self.assertIn(
            "num_warps_policy_override=(",
            flat_decode,
        )

    def test_dense_post_norm_chain_is_scoped_to_the_measured_e2b_policy(self):
        setup = self.llama[
            self.llama.index("dense_post_norm_chain_requested = policy_bool(") :
            self.llama.index("self._gemma4_flat_dense_next_attn_norm_bufs = (")
        ]

        self.assertIn(
            '"MEGAGEMM_GEMMA4_DENSE_POST_NORM_CHAIN_DECODE"',
            setup,
        )
        self.assertIn('"gemma4_dense_post_norm_chain"', setup)
        self.assertIn(
            'self.runtime_policy.name == "gemma4-e2b-l4"',
            setup,
        )
        self.assertNotIn('"gemma4-e4b-l4"', setup)

    def test_dense_post_norm_chain_remains_active_under_decode_profiling(self):
        flat_decode = self.llama[
            self.llama.index("    def _gemma4_flat_decode_layers(") :
            self.llama.index("    def _hybrid_flat_decode_layers(")
        ]
        selection = flat_decode[
            flat_decode.index("dense_post_norm_chain = bool(") :
            flat_decode.index("for layer_idx, lw in enumerate")
        ]

        self.assertNotIn("timing_events is None", selection)
        self.assertIn('"dense_post_norm_chain"', flat_decode)

    def test_vllm_phase_split_requires_consistent_same_request_metrics(self):
        sys.path.insert(0, str(ROOT / "benchmarks"))
        from run_gemma4_moe_vs_vllm import validated_vllm_phase_span

        valid = validated_vllm_phase_span(
            {
                "arrival_time": 10.0,
                "first_token_time": 10.08,
                "finished_time": 11.0,
            },
            1000.0,
        )
        self.assertTrue(valid["valid"])
        self.assertAlmostEqual(valid["prefill_ms"], 80.0)
        self.assertAlmostEqual(valid["decode_ms"], 920.0)

        unavailable = validated_vllm_phase_span({}, 1000.0)
        self.assertEqual(unavailable["status"], "request_metrics_unavailable")

        inconsistent = validated_vllm_phase_span(
            {
                "arrival_time": 10.0,
                "first_token_time": 10.08,
                "finished_time": 11.0,
            },
            100.0,
        )
        self.assertEqual(inconsistent["status"], "request_metrics_inconsistent")

    def test_vllm_outputs_are_aligned_to_the_shared_prompt_manifest(self):
        sys.path.insert(0, str(ROOT / "benchmarks"))
        from run_gemma4_moe_batch_vs_vllm import align_vllm_outputs_to_prompts

        def output(prompt, tokens):
            return SimpleNamespace(
                prompt_token_ids=prompt,
                outputs=[SimpleNamespace(token_ids=tokens)],
            )

        prompts = [[10, 11], [20, 21], [30, 31]]
        matrix, alignment = align_vllm_outputs_to_prompts(
            [
                output(prompts[2], [302]),
                output(prompts[0], [102]),
                output(prompts[1], [202]),
            ],
            prompts,
        )

        self.assertEqual(matrix, [[102], [202], [302]])
        self.assertEqual(alignment["method"], "prompt_token_ids")
        self.assertTrue(alignment["reordered"])
        self.assertEqual(alignment["output_to_prompt_index"], [2, 0, 1])

    def test_vllm_warmup_requires_timing_and_token_stability(self):
        sys.path.insert(0, str(ROOT / "benchmarks"))
        from run_gemma4_moe_batch_vs_vllm import evaluate_vllm_warmup_stability

        def sample(total_ms, token=7):
            return {"total_ms": total_ms, "token_ids": [[token, 8]]}

        too_few = evaluate_vllm_warmup_stability(
            [sample(200.0), sample(100.0)]
        )
        self.assertFalse(too_few["stable"])
        self.assertEqual(too_few["reason"], "minimum_warmups_not_reached")

        timing_unstable = evaluate_vllm_warmup_stability(
            [sample(200.0), sample(100.0), sample(80.0)]
        )
        self.assertFalse(timing_unstable["stable"])
        self.assertFalse(timing_unstable["accepted"])
        self.assertEqual(timing_unstable["reason"], "last_pair_timing_unstable")

        token_unstable = evaluate_vllm_warmup_stability(
            [sample(200.0), sample(100.0), sample(101.0, token=9)]
        )
        self.assertFalse(token_unstable["stable"])
        self.assertEqual(token_unstable["reason"], "last_pair_tokens_changed")

        stable = evaluate_vllm_warmup_stability(
            [sample(200.0), sample(100.0), sample(102.0)]
        )
        self.assertTrue(stable["stable"])
        self.assertTrue(stable["accepted"])
        self.assertEqual(stable["reason"], "stable")

        exhausted = evaluate_vllm_warmup_stability(
            [
                sample(total_ms)
                for total_ms in (
                    6691.9,
                    2779.8,
                    939.0,
                    1733.2,
                    992.7,
                    1450.0,
                    940.0,
                    1320.0,
                )
            ]
        )
        self.assertFalse(exhausted["stable"])
        self.assertTrue(exhausted["accepted"])
        self.assertEqual(
            exhausted["acceptance_reason"],
            "warmup_budget_exhausted_all_tokens_exact",
        )
        self.assertTrue(exhausted["all_warmup_tokens"]["exact"])

    def test_vllm_measurements_start_only_after_adaptive_warmup_stabilizes(self):
        sys.path.insert(0, str(ROOT / "benchmarks"))
        import run_gemma4_moe_batch_vs_vllm as batch_bench

        token_ids = [[7, 8] for _ in range(16)]

        def row(total_ms):
            return {
                "total_ms": float(total_ms),
                "output_tok_s_total": 1.0,
                "prefill_ms": 10.0,
                "decode_ms": 90.0,
                "decode_tok_s": 1.0,
                "decode_measurement_method": "test",
                "phase_metrics_status": "valid",
                "phase_metrics_reason": "",
                "request_metrics_aggregate": {},
                "request_metric_total_ms": float(total_ms),
                "request_metric_wall_error_ratio": 0.0,
                "token_ids": token_ids,
            }

        responses = [
            row(6605.0),
            row(2777.0),
            row(985.0),
            row(938.0),
            row(939.0),
            row(937.0),
            row(936.0),
            row(938.0),
        ]
        prompts = [[index, index + 1] for index in range(16)]
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                batch_sizes=[16],
                max_tokens=2,
                repeats=3,
                model="local-model",
                dtype="bf16",
                out_json=str(Path(temp_dir) / "vllm.json"),
            )
            with (
                mock.patch.object(
                    batch_bench,
                    "make_vllm",
                    return_value=(object(), "cuda", "test-vllm", {}),
                ),
                mock.patch.object(
                    batch_bench,
                    "run_vllm_request",
                    side_effect=responses,
                ) as request,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = batch_bench.run_vllm(args, prompts)

        case = result["cases"]["16"]
        self.assertEqual(request.call_count, 8)
        self.assertEqual(len(case["warmups"]), 5)
        self.assertEqual(len(case["samples"]), 3)
        self.assertTrue(case["warmup_stability"]["stable"])
        self.assertTrue(case["token_stability"]["exact"])
        self.assertEqual(case["summary"]["total_ms_median"], 937.0)

    def test_vllm_025_phase_timestamps_do_not_mix_clock_domains(self):
        sys.path.insert(0, str(ROOT / "benchmarks"))
        from run_gemma4_moe_vs_vllm import (
            extract_vllm_request_metrics,
            validated_vllm_phase_span,
        )

        request = SimpleNamespace(
            metrics=SimpleNamespace(
                arrival_time=1_785_000_000.0,
                queued_ts=100.0,
                scheduled_ts=100.01,
                first_token_ts=100.08,
                last_token_ts=101.0,
            )
        )
        metrics = extract_vllm_request_metrics(request)

        self.assertEqual(metrics["frontend_arrival_time"], 1_785_000_000.0)
        self.assertEqual(metrics["arrival_time"], 100.01)
        self.assertEqual(metrics["first_token_time"], 100.08)
        self.assertEqual(metrics["finished_time"], 101.0)
        phase = validated_vllm_phase_span(metrics, 990.0)
        self.assertTrue(phase["valid"])
        self.assertAlmostEqual(phase["prefill_ms"], 70.0)
        self.assertAlmostEqual(phase["decode_ms"], 920.0)

    def test_vllm_compatibility_stack_avoids_heterogeneous_override(self):
        self.assertNotIn("gemma4_vllm_hf_overrides", self.benchmark)
        self.assertNotIn('kwargs["hf_overrides"]', self.benchmark)

    def test_reuses_qwen_snapshot_download_path(self):
        self.assertIn(
            'HARNESS_REV="gemma4-ab-qwen-snapshot-v130-pinned-vllm-transformers-stack"',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_BATCH_DETERMINISTIC="${MEGAGEMM_BATCH_DETERMINISTIC:-1}"',
            self.script,
        )
        self.assertIn("unset CUBLAS_WORKSPACE_CONFIG", self.script)
        self.assertIn("proven stable deterministic baseline", self.script)
        self.assertIn("decode graph KV recycle: OK", self.script)
        self.assertIn("capture_tables != replay_tables", self.script)
        self.assertIn("self.free_blocks.sort()", self.kv_cache)
        self.assertIn("_decode_graph_physical_rebinds", self.scheduler)
        self.assertIn('"physical_rebinds"', self.batch_benchmark)
        self.assertIn(
            'PREFILL_CUDA_GRAPH_DEFAULT=0',
            self.script,
        )
        self.assertNotIn('PREFILL_CUDA_GRAPH_DEFAULT=1', self.script)
        self.assertIn(
            'MEGAGEMM_PREFILL_CUDA_GRAPHS="${MEGAGEMM_PREFILL_CUDA_GRAPHS:-${PREFILL_CUDA_GRAPH_DEFAULT}}"',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_IMPLICIT_CAUSAL_PREFILL="${MEGAGEMM_GEMMA4_IMPLICIT_CAUSAL_PREFILL:-1}"',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_PARALLEL_MOE_PREFILL="${MEGAGEMM_GEMMA4_PARALLEL_MOE_PREFILL:-1}"',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_VECTORIZED_PREFILL_KV="${MEGAGEMM_GEMMA4_VECTORIZED_PREFILL_KV:-1}"',
            self.script,
        )
        self.assertIn("MEGAGEMM_PREFILL_CORRECTNESS_GATE", self.batch_benchmark)
        self.assertIn("MEGAGEMM_POST_KERNEL_PREFILL_CONTRACT", self.batch_benchmark)
        self.assertIn("first_token_vs_prefill_oracle", self.batch_benchmark)
        self.assertIn("MEGAGEMM_GEMMA4_FUSED_QKV_PREFILL=0", self.script)
        self.assertIn("MEGAGEMM_GEMMA4_FUSED_ATTN_PREP_PREFILL=0", self.script)
        self.assertIn(
            "MEGAGEMM_GEMMA4_PREFILL_GRAPH_FUSED_ATTN_FRONTEND=0",
            self.script,
        )
        self.assertIn(
            "run_gemma4_b16_full_model_graph_preflight.py",
            self.script,
        )
        self.assertLess(
            self.script.index("run_gemma4_b16_full_model_graph_preflight.py"),
            self.script.index("== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES =="),
        )
        self.assertIn("from huggingface_hub import snapshot_download", self.script)
        self.assertIn("snapshot_download(repo_id=repo_id, local_dir=str(local_dir), max_workers=workers)", self.script)
        self.assertIn('HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-16}"', self.script)
        self.assertNotIn('cfg.get("segmented_prefill_fixed_counts")', self.script)
        self.assertNotIn('hf download "${MODEL}"', self.script)
        self.assertNotIn("download_hf_snapshot_resilient", self.script)
        self.assertNotIn("aria2c", self.script)
        self.assertIn('"fixed_route_pack": fixed_route_pack', self.script)
        self.assertIn('"compact_route_pack": compact_route_pack', self.script)
        self.assertIn(
            'PINNED_VLLM_VERSION="${PINNED_VLLM_VERSION:-${VLLM_VERSION:-0.24.0}}"',
            self.script,
        )
        self.assertIn(
            'PINNED_TRANSFORMERS_VERSION="${PINNED_TRANSFORMERS_VERSION:-5.13.1}"',
            self.script,
        )
        self.assertIn('unset VLLM_VERSION', self.script)
        self.assertIn('"vllm==${PINNED_VLLM_VERSION}"', self.script)
        self.assertIn('"transformers==${PINNED_TRANSFORMERS_VERSION}"', self.script)
        self.assertIn(
            '"MEGAGEMM_PREFILL_CUDA_GRAPHS",\n            False,',
            self.engine,
        )

    def test_parity_gate_stages_source_off_google_drive_before_install(self):
        stage = self.script.index(
            "Staging parity source off Google Drive:"
        )
        base_install = self.script.index("== INSTALL SAME-VM RUNTIME ==")
        parity_install = self.script.index(
            "== INSTALL vLLM FOR NO-CHECKPOINT PARITY GATE =="
        )
        self.assertLess(stage, base_install)
        self.assertLess(stage, parity_install)
        self.assertIn(
            'mktemp -d /content/megagemm-gemma4-parity.XXXXXX',
            self.script,
        )
        self.assertIn(
            'cp -a "${PWD}/megagemm" "${LOCAL_STAGE_DIR}/megagemm"',
            self.script,
        )
        self.assertIn(
            'cp -a "${PWD}/benchmarks" "${LOCAL_STAGE_DIR}/benchmarks"',
            self.script,
        )
        self.assertIn(
            'export OUT_DIR="/content/bench_results/${RUN_ID}"',
            self.script,
        )
        self.assertIn(
            "exec bash benchmarks/run_gemma4_moe_vs_vllm_colab.sh",
            self.script,
        )

    def test_full_ab_rejects_low_vram_before_network_or_install(self):
        vram_guard = self.script.index("GPU PREFLIGHT FAILED:")
        https_preflight = self.script.index("== PUBLIC MODEL HTTPS PREFLIGHT ==")
        base_install = self.script.index("== INSTALL SAME-VM RUNTIME ==")
        model_download = self.script.index(
            "== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES =="
        )
        self.assertLess(vram_guard, https_preflight)
        self.assertLess(vram_guard, base_install)
        self.assertLess(vram_guard, model_download)
        self.assertIn("MIN_BF16_AB_VRAM_MIB=71680", self.script)
        self.assertIn("A100-SXM4-80GB-class GPU", self.script)
        self.assertIn("48.1 GiB checkpoint", self.script)
        self.assertIn(
            "No HTTPS preflight, package installation, checkpoint download, "
            "or GPU benchmark was started.",
            self.script,
        )

    def test_runs_direct_moe_parity_before_model_download(self):
        parity_install = self.script.index(
            "== INSTALL vLLM FOR NO-CHECKPOINT PARITY GATE =="
        )
        parity_gate = self.script.index(
            "== MEGAGEMM-vLLM FUSED-MoE PRE-DOWNLOAD GATE =="
        )
        model_download = self.script.index(
            "== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES =="
        )
        megagemm = self.script.index("== RUN MEGAGEMM")
        install_vllm = self.script.index("== INSTALL vLLM AFTER MEGAGEMM PASSES ==")
        run_vllm = self.script.index("== RUN vLLM")
        self.assertLess(parity_install, parity_gate)
        self.assertLess(parity_gate, model_download)
        self.assertLess(model_download, megagemm)
        self.assertLess(megagemm, install_vllm)
        self.assertLess(install_vllm, run_vllm)

    def test_active_list_gate_is_bounded_and_download_free(self):
        gate = ACTIVE_LIST_GATE.read_text(encoding="utf-8")
        compile(gate, str(ACTIVE_LIST_GATE), "exec")
        self.assertIn("build_v82_route", gate)
        self.assertIn("build_active_expert_route", gate)
        self.assertIn('default="8,16,32,64,90,128"', gate)
        self.assertIn('"active_experts": int(active_experts)', gate)
        self.assertIn('"singleton_experts": singleton_experts', gate)
        self.assertIn('"empty_experts": empty_experts', gate)
        self.assertIn('"minimum_speedup"', gate)
        self.assertIn('"minimum_low_active_speedup"', gate)
        self.assertIn('"maximum_regression_ratio"', gate)
        self.assertIn('"geomean_speedup"', gate)
        self.assertIn('"runtime_active_list_early_exit"', gate)
        self.assertIn("graph_warmup_replays = max(3, int(iterations) * 4)", gate)
        self.assertIn("@torch.inference_mode()", gate)
        self.assertIn(
            "active-list gate dispatch failed before CUDA graph capture",
            gate,
        )
        self.assertNotIn("snapshot_download", gate)
        self.assertNotIn("vllm", gate.lower().replace("vllm_install: disabled", ""))

    def test_vllm_moe_parity_gate_is_direct_and_checkpoint_free(self):
        gate = VLLM_MOE_PARITY_GATE.read_text(encoding="utf-8")
        compile(gate, str(VLLM_MOE_PARITY_GATE), "exec")
        self.assertIn("build_fixed_b16_route", gate)
        self.assertIn("qwen3_moe_grouped_decode", gate)
        self.assertIn(
            "from vllm.model_executor.layers.fused_moe.fused_moe "
            "import fused_experts",
            gate,
        )
        self.assertIn("activation=MoEActivation.GELU_TANH", gate)
        self.assertIn('"decision": decision_name', gate)
        self.assertIn('"PORT_VLLM_FUSED_MOE"', gate)
        self.assertIn('"MOVE_OFF_MOE"', gate)
        self.assertIn('"INVALID_PARITY_GATE"', gate)
        self.assertIn("CUDA graph vs CUDA graph", gate)
        self.assertNotIn("snapshot_download", gate)
        self.assertNotIn("huggingface_hub", gate)
        self.assertIn(
            "No checkpoint was downloaded and no full vLLM engine was started.",
            self.script,
        )

    def test_prefill_graph_must_capture_replay_and_match_before_vllm(self):
        for expected in (
            "MegaGemm prefill graph capture (excluded from summary)",
            "Gemma4 prefill graph capture changed greedy output",
            'prefill_graph["capture_replays"] != 1',
            "MegaGemm prefill graph replay gate (excluded from summary)",
            "Gemma4 prefill graph replay changed greedy output",
            "stopping before vLLM install",
        ):
            self.assertIn(expected, self.benchmark)

    def test_prefill_capture_executes_graph_before_returning_logits(self):
        engine_capture = self.engine[
            self.engine.index("def _run_single_prefill_graph_or_eager"):
            self.engine.index("def _prepare_generate_graph_capture_stream")
        ]
        scheduler_capture = self.scheduler[
            self.scheduler.index("def _capture_prefill_graph"):
            self.scheduler.index("def _advance_prefill_seq_lens")
        ]
        for source in (engine_capture, scheduler_capture):
            capture = source[source.index("graph = torch.cuda.CUDAGraph()") :]
            replay = capture.index("graph.replay()")
            publish = capture.index("store[")
            self.assertLess(replay, publish)
            self.assertIn("capture_replays", capture)

    def test_full_model_prefill_graph_gate_warms_exact_capture_body(self):
        compile(
            self.full_model_graph_preflight,
            str(FULL_MODEL_GRAPH_PREFLIGHT),
            "exec",
        )
        capture = self.scheduler[
            self.scheduler.index("def _capture_prefill_batch_graph") :
            self.scheduler.index("def _write_deferred_prefill_batch_kv")
        ]
        self.assertLess(
            capture.index("warm_result = self.model.prefill_batch_graph("),
            capture.index("graph = torch.cuda.CUDAGraph()"),
        )
        self.assertIn("capture_body_warmups", capture)
        self.assertIn("for _ in range(2):", capture)
        self.assertIn("share_identical_layer_parameters", self.full_model_graph_preflight)
        self.assertIn("NUM_LAYERS = 30", self.full_model_graph_preflight)
        self.assertIn("VOCAB_SIZE = 262144", self.full_model_graph_preflight)
        self.assertIn("defer_kv_writes=True", self.full_model_graph_preflight)
        self.assertIn("graph.replay()", self.full_model_graph_preflight)
        self.assertIn("--layer-limit", self.full_model_graph_preflight)
        self.assertIn(
            "layer_limit * ROWS * TOP_K * HIDDEN_SIZE * 4",
            self.full_model_graph_preflight,
        )
        self.assertIn("graph_safe_prefill: bool = True", self.llama)
        self.assertIn(
            "graph_safe_prefill=graph_safe_prefill",
            self.llama,
        )
        self.assertIn(
            "graph_safe_attention_frontend_layers",
            self.full_model_graph_preflight,
        )
        self.assertIn(
            "graph_fused_attention_frontend_layers",
            self.full_model_graph_preflight,
        )
        self.assertIn(
            '"reference_mode": "graph_safe_eager"',
            self.full_model_graph_preflight,
        )
        self.assertIn(
            '"attention_frontend_mode": "unfused_graph_stable_direct_kv_v105"',
            self.full_model_graph_preflight,
        )
        self.assertNotIn(
            "model.prefill_batch(",
            self.full_model_graph_preflight,
        )
        self.assertNotIn(
            "class DeferredKVCollector",
            self.full_model_graph_preflight,
        )
        self.assertIn(
            "graph_deferred_kv_max_abs_error",
            self.full_model_graph_preflight,
        )
        self.assertIn(
            "persistent_deferred_kv_buffers",
            self.full_model_graph_preflight,
        )
        self.assertIn(
            "_prefill_graph_deferred_kv_buffers",
            self.llama,
        )
        self.assertIn("prefill_kv_out=prefill_kv_out", self.llama)
        self.assertIn("k_cache.copy_(k.transpose(1, 2))", self.llama)
        self.assertIn("v_cache.copy_(v.transpose(1, 2))", self.llama)
        self.assertIn("did not write K/V directly", self.llama)
        self.assertIn("deferred_kv_storage_stable", self.scheduler)
        self.assertIn(
            "graph_safe_moe_layers",
            self.full_model_graph_preflight,
        )
        self.assertIn(
            "prefill_graph_probe_tokens = min(3, max(1, int(args.max_tokens)))",
            BATCH_BENCH.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "graph_safe=graph_safe_prefill",
            self.llama,
        )
        self.assertIn(
            "FULL_MODEL_GRAPH_STAGE capture_synchronized",
            self.full_model_graph_preflight,
        )
        self.assertNotIn("snapshot_download", self.full_model_graph_preflight)
        self.assertNotIn("huggingface_hub", self.full_model_graph_preflight)
        compile(
            self.full_model_graph_diagnose,
            str(FULL_MODEL_GRAPH_DIAGNOSE),
            "exec",
        )
        self.assertIn("subprocess.run(", self.full_model_graph_diagnose)
        self.assertIn("first_failing_prefix", self.full_model_graph_diagnose)
        self.assertIn("disable_expandable_segments", self.full_model_graph_diagnose)
        self.assertIn("disable_compact_route_pack", self.full_model_graph_diagnose)
        self.assertIn(
            "run_gemma4_b16_full_model_graph_preflight.py",
            self.full_model_graph_wrapper,
        )
        self.assertIn("--replays 3", self.full_model_graph_wrapper)
        self.assertIn("model_download: disabled", self.full_model_graph_wrapper)
        self.assertIn("vllm_install: disabled", self.full_model_graph_wrapper)

    def test_batch_matrix_is_one_bounded_same_vm_run(self):
        wrapper = BATCH_HARNESS.read_text(encoding="utf-8")
        benchmark = BATCH_BENCH.read_text(encoding="utf-8")
        self.assertIn('BATCH_SIZES="${BATCH_SIZES:-2,4,8,16}"', wrapper)
        self.assertIn(
            'PREFILL_FINITE_TRACE_ONLY="${PREFILL_FINITE_TRACE_ONLY:-0}"',
            wrapper,
        )
        self.assertIn('run_gemma4_moe_vs_vllm_colab.sh', wrapper)
        self.assertIn('BATCH_SIZES="${BATCH_SIZES:-1}"', self.script)
        self.assertIn('run_gemma4_moe_batch_vs_vllm.py', self.script)
        self.assertIn('SUPPORTED_BATCHES = (2, 4, 8, 16)', benchmark)
        self.assertIn('def run_gemma4_prefill_finite_trace(', benchmark)
        self.assertIn('--prefill-finite-trace-only', benchmark)
        self.assertIn('MEGAGEMM_PREFILL_FINITE_TRACE', benchmark)
        self.assertIn('MEGAGEMM_DECODE_CUDA_GRAPHS_PREFER_STEP', benchmark)
        self.assertIn('MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE', benchmark)
        self.assertIn('MEGAGEMM_COMPACT_KERNEL_GATE', benchmark)
        self.assertIn('MEGAGEMM_LM_HEAD_KERNEL_GATE', benchmark)
        self.assertIn('MEGAGEMM_PREFILL_KERNEL_GATE', benchmark)
        self.assertIn('MEGAGEMM_PREFILL_ATTN_MOE_BRIDGE_GATE', benchmark)
        self.assertIn('segmented prefill repeated differently', benchmark)
        self.assertNotIn('"case": "single_accumulator_l2_g8"', benchmark)
        self.assertIn('"retired_candidates": [', benchmark)
        self.assertIn('"case": "expert_major_g1"', benchmark)
        self.assertIn('"case": "vllm_m16_e128_bf16"', benchmark)
        self.assertIn('"run_id": "gemma4_moe_ab_20260724_205618"', benchmark)
        self.assertIn("def run_attention_decode_kernel_gate", benchmark)
        self.assertIn('"grouped_segmented": True', benchmark)
        self.assertIn(
            'os.environ["MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_DECODE"]',
            benchmark,
        )
        self.assertIn("def run_fused_next_attn_norm_decode_gate", benchmark)
        self.assertIn("fused_post_moe_next_attn_norm", benchmark)
        self.assertIn("enabled = bool(supported)", benchmark)
        self.assertNotIn(
            "enabled = bool(supported and runtime_supported)",
            benchmark,
        )
        self.assertIn('"case": "sliding_h256_w4_fixed_qk32"', benchmark)
        self.assertIn('"case": "sliding_h256_w4"', benchmark)
        self.assertIn('"case": "full_h512_w4"', benchmark)
        self.assertIn('"case": "gqa2_direct_w8"', benchmark)
        self.assertIn('"case": "gqa2_direct_w4"', benchmark)
        self.assertIn('"case": "generic_global_w4"', benchmark)
        self.assertIn('"run_id": "gemma4_moe_ab_20260725_001448"', benchmark)
        self.assertIn('"run_id": "gemma4_moe_ab_20260725_003420"', benchmark)
        self.assertIn("MEGAGEMM_ATTENTION_KERNEL_GATE", benchmark)
        self.assertIn("set_segmented_prefill_runtime_options", benchmark)
        self.assertIn(
            "_qwen3_moe_segmented_gate_up_single_accum_kernel",
            self.qwen3_moe,
        )
        self.assertIn(
            "_qwen3_moe_segmented_down_single_accum_partial_kernel",
            self.qwen3_moe,
        )
        self.assertIn(
            "num_pid_in_group = GROUP_SIZE_M * num_pid_n",
            self.qwen3_moe,
        )
        self.assertNotIn(
            "candidates.append(run_grouped_mm_case())",
            benchmark,
        )
        self.assertNotIn(
            "from megagemm.kernels.gemma4_grouped_prefill import",
            self.script,
        )
        self.assertIn(
            "single_accumulator_prefill_gate: retired after v76 "
            "(19 percent slower)",
            self.script,
        )
        self.assertIn(
            "compact_decode_gate: v84 active-list branch retired after "
            "measuring 1.26 percent slower",
            self.script,
        )
        self.assertIn(
            "attention_decode: v88 grouped segmented core promoted for B16 "
            "H256/GQA2 and H512/GQA8",
            self.script,
        )
        self.assertIn(
            "decode_baseline: v81 fused post-MoE layer-scalar plus "
            "next-layer RMSNorm",
            self.script,
        )
        self.assertIn(
            "decode_candidate_gate: retired after v88; grouped segmented "
            "attention is now the default",
            self.script,
        )
        self.assertIn('"case": "active_list_early_exit"', benchmark)
        self.assertIn('"run_id": "gemma4_moe_ab_20260727_031037"', benchmark)
        self.assertIn('"case": "gate_s4_down_s3"', benchmark)
        self.assertIn('"run_id": "gemma4_moe_ab_20260727_020604"', benchmark)
        self.assertIn('"full_request_gate"', benchmark)
        self.assertIn("USE_PROVEN_EXACT_BASELINE", benchmark)
        self.assertIn(
            'STOP_BEFORE_VLLM_IF_NO_DECODE_PROMOTION="${'
            'STOP_BEFORE_VLLM_IF_NO_DECODE_PROMOTION:-0}"',
            self.script,
        )
        self.assertIn(
            "BATCH_PROMOTION_ARGS=(--stop-after-no-decode-promotion)",
            self.script,
        )
        self.assertIn(
            "MEGAGEMM_EARLY_STOP_NO_DECODE_PROMOTION",
            benchmark,
        )
        self.assertIn(
            '"status": "gate_rejected"',
            benchmark,
        )
        self.assertIn(
            "NO DECODE PROMOTION (${DECODE_PROMOTION_DECISION}): "
            "stopping before vLLM install.",
            self.script,
        )
        self.assertIn('MEGAGEMM_PAGED_DECODE_GQA2=0', self.script)
        self.assertIn('MEGAGEMM_PAGED_DECODE_WARPS=0', self.script)
        self.assertIn('MEGAGEMM_PAGED_DECODE_WARPS_H256=8', self.script)
        self.assertIn('MEGAGEMM_PAGED_DECODE_WARPS_H512=4', self.script)
        self.assertIn('compact == "APPLY"', self.script)
        self.assertIn('attention == "APPLY"', self.script)
        self.assertNotIn('next_norm == "APPLY"', self.script)
        self.assertIn('[[ "${DECODE_PROMOTION_DECISION}" != APPLY:* ]]', self.script)
        self.assertIn('"MEGAGEMM_PAGED_DECODE_GQA2"', self.scheduler)
        self.assertIn('"MEGAGEMM_PAGED_DECODE_WARPS"', self.scheduler)
        self.assertIn('"MEGAGEMM_PAGED_DECODE_WARPS_H256"', self.scheduler)
        self.assertIn('"MEGAGEMM_PAGED_DECODE_WARPS_H512"', self.scheduler)
        self.assertIn(
            '"MEGAGEMM_GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE"',
            self.scheduler,
        )
        self.assertIn(
            'f"MEGAGEMM_PAGED_DECODE_WARPS_H{int(head_dim)}"',
            self.paged_attention,
        )
        self.assertNotIn("FIXED_QK_REDUCTION: tl.constexpr", self.paged_attention)
        gqa2_direct = self.paged_attention[
            self.paged_attention.index("def _use_gqa2_direct_decode") :
            self.paged_attention.index("def _use_gqa4_direct_decode")
        ]
        self.assertIn("head_dim not in (128, 256)", gqa2_direct)
        self.assertIn("_GQA2_DIRECT_DECODE_HITS += 1", self.paged_attention)
        self.assertIn("gemma4_grouped_mm_prefill", self.llama)
        self.assertIn("_gemma4_grouped_mm_route_pack_kernel", self.grouped_prefill)
        self.assertIn("_gemma4_grouped_mm_geglu_kernel", self.grouped_prefill)
        self.assertIn(
            "_gemma4_grouped_mm_topk_reduce_kernel",
            self.grouped_prefill,
        )
        self.assertIn('offs=offsets', self.grouped_prefill)
        self.assertIn(
            'MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_PREFILL="${MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_PREFILL:-1}"',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_PREFILL="${MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_PREFILL:-1}"',
            self.script,
        )
        self.assertNotIn("RETIRED_AFTER_V74", benchmark)
        self.assertIn(
            "set_gemma4_prefill_attn_moe_bridge_runtime",
            benchmark,
        )
        self.assertIn("APPLY_FUSED_ATTN_MOE_BRIDGE", benchmark)
        self.assertIn("exact_two_kernel_attn_moe_router_bridge", benchmark)
        self.assertIn('"kernel_count"] = 2', benchmark)
        self.assertIn("run_gemma4_prefill_router_kernel_gate", benchmark)
        self.assertIn("APPLY_FUSED_MATRIX_ROUTER", benchmark)
        self.assertIn("set_fused_prefill_runtime", benchmark)
        self.assertIn("MEGAGEMM_PREFILL_CANDIDATE_ROLLBACK", benchmark)
        self.assertIn("candidate_runtime_failures", benchmark)
        self.assertIn("post_contract_rollback", benchmark)
        self.assertIn(
            "normalized_router_input=bridge_router_in",
            self.llama,
        )
        self.assertIn(
            "rmsnorm_triton_attn_residual_router_bridge(",
            self.llama,
        )
        self.assertIn("baseline_stability_is_diagnostic", benchmark)
        self.assertIn('prefill_partial_reduce_layers', benchmark)
        self.assertIn('"partial_dtype"', benchmark)
        self.assertIn('"decision": "KEEP_FP32_PARTIAL"', benchmark)
        self.assertIn('required_assignments = max(batch_sizes) * 25 * 8', self.script)
        self.assertIn('PARTIAL_CACHE_MAX_ASSIGNMENTS', self.script)
        self.assertIn('"graph_workspace_bytes": graph_workspace_bytes', self.script)
        self.assertIn(
            "qwen3_moe_prepare_segmented_prefill_graph_workspace",
            self.script,
        )
        self.assertIn("B16 PREFILL GRAPH SAFETY GATE (NO MODEL DOWNLOAD)", self.script)
        self.assertIn("run_gemma4_b16_prefill_graph_preflight.py", self.script)
        self.assertLess(
            self.script.index("run_gemma4_b16_prefill_graph_preflight.py"),
            self.script.index("== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES =="),
        )
        self.assertIn("_qwen3_moe_compact_route_counts_kernel", self.qwen3_moe)
        self.assertIn("_qwen3_moe_compact_route_tile_prefix_kernel", self.qwen3_moe)
        self.assertIn("_qwen3_moe_compact_route_scatter_kernel", self.qwen3_moe)
        self.assertIn("segmented_prefill_graph_route_pack", self.qwen3_moe)
        self.assertIn(
            "qwen3_moe_segmented_prefill_graph_route_pack_layers",
            self.llama,
        )
        self.assertIn("num_tiles == max_tiles == 320", self.graph_preflight)
        self.assertIn("def run_attention_graph_case", self.graph_preflight)
        self.assertIn('"external_kv_max_abs_error"', self.graph_preflight)
        self.assertIn('(\"sliding\", 8, 256, True)', self.graph_preflight)
        self.assertIn('(\"full\", 2, 512, True)', self.graph_preflight)
        self.assertIn("@torch.inference_mode()", self.graph_preflight)
        self.assertIn(
            'raise RuntimeError("Preflight must run with autograd disabled")',
            self.graph_preflight,
        )
        self.assertIn('expert_grouped_compact', benchmark)
        self.assertIn('equal_length_distinct_prompts', benchmark)
        self.assertIn('vectorized_prefill_kv_hits', benchmark)
        self.assertNotIn('[exact_prompt] * batch_size', benchmark)
        self.assertIn('capture_run_excluded', benchmark)
        self.assertIn('batch=16 prefill graph capture', benchmark)
        self.assertIn('batch=16 prefill graph replay gate', benchmark)
        self.assertIn('MEGAGEMM_B16_PREFILL_GRAPH_GATE', benchmark)
        self.assertIn('capture_tokens_vs_eager', benchmark)
        self.assertIn('replay_tokens_vs_eager', benchmark)
        self.assertIn('measured_tokens_vs_eager', benchmark)
        self.assertIn('changed the full "', benchmark)
        self.assertIn('current_prefill_replays <= measured_prefill_replays', benchmark)
        self.assertIn('Gemma4 B16 prefill CUDA graph contract failed', benchmark)
        self.assertIn("'capture_replays': int(", self.scheduler)
        self.assertIn("def _run_prefill_batch_graph_or_eager", self.scheduler)
        self.assertIn("'kind': 'padded'", self.scheduler)
        self.assertIn("'bucket_kinds': sorted", self.scheduler)
        self.assertIn("'kv_write_mode': 'external_after_replay'", self.scheduler)
        self.assertIn("'external_kv_write_replays': int(", self.scheduler)
        self.assertIn("'deferred_kv': deferred_kv", self.scheduler)
        self.assertIn("defer_kv_writes=True", self.scheduler)
        self.assertIn("def _write_deferred_prefill_batch_kv", self.scheduler)
        self.assertIn("defer_kv_writes: bool = False", self.llama)
        self.assertIn("'workspace_refs': workspace_refs", self.scheduler)
        self.assertIn("'workspace_tensors': int(sum(", self.scheduler)
        self.assertIn("prepare_prefill_cuda_graph_workspace", self.llama)
        self.assertIn("segmented_graph_partial_out_", self.qwen3_moe)
        self.assertIn('capture_graph.get("workspace_tensors"', benchmark)
        self.assertIn('capture_graph.get("external_kv_write_replays"', benchmark)
        self.assertIn('!= ["external_after_replay"]', benchmark)
        self.assertIn("def prefill_batch_graph", self.llama)
        self.assertIn('!= ["padded"]', benchmark)
        self.assertIn('CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"', self.script)
        self.assertIn('MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST', self.script)
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EXPERT_GRID_PACK',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_COALESCED_WEIGHTS',
            self.script,
        )
        self.assertNotIn("COMPACT_SINGLETON_GEMV", self.script)
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_NUM_WARPS',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_NUM_STAGES',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM',
            self.script,
        )
        self.assertIn('BATCH_DETERMINISTIC_ARGS=(--deterministic)', self.script)
        self.assertIn('deterministic=args.deterministic', benchmark)
        self.assertIn('default=True,\n        help="enable the proven stable deterministic baseline', benchmark)
        self.assertIn('MEGAGEMM_DETERMINISM_CONTRACT', benchmark)
        self.assertIn('torch.are_deterministic_algorithms_enabled()', benchmark)
        self.assertIn('parallel_moe_enabled', benchmark)
        self.assertIn('parallel_moe_hits', benchmark)
        self.assertIn('fused_post_moe_norm_residual_enabled', benchmark)
        self.assertIn('fused_post_moe_norm_residual_hits', benchmark)
        self.assertIn('fused_expert_reduce_post_moe_enabled', benchmark)
        self.assertIn('fused_expert_reduce_post_moe_hits', benchmark)
        self.assertIn('fused_expert_reduce_post_moe_layers', benchmark)
        self.assertIn('fused_router_expert_input_norm_enabled', benchmark)
        self.assertIn('fused_router_expert_input_norm_hits', benchmark)
        self.assertIn('fork_before_router', benchmark)
        self.assertIn('batch_size in (8, 16)', benchmark)
        self.assertIn('isolated_shared_norm_buffers', benchmark)
        self.assertIn('fused_qkv_prefill_hits', benchmark)
        self.assertIn('fused_attn_prepare_hits', benchmark)
        self.assertIn('fused_attn_prepare_disabled_layers', benchmark)
        self.assertIn('Gemma4 batch fused attention prefill was not exercised', benchmark)
        self.assertIn('compact_route_pack', benchmark)
        self.assertIn('compact_route_pack_passes', benchmark)
        self.assertIn(
            'qwen3_moe_segmented_prefill_graph_route_pack_layers',
            benchmark,
        )
        self.assertIn('deterministic_route_layers', benchmark)
        self.assertIn('Gemma4 compact prefill route pack was not exercised', benchmark)
        self.assertIn('compact_expert_grid_pack', benchmark)
        self.assertIn('expert_grid_pack_layers', benchmark)
        self.assertIn('decode_kernel_tune', benchmark)
        self.assertNotIn('router_decode_tune', benchmark)
        self.assertIn('Gemma4 legacy decode router lock was not exercised', benchmark)
        self.assertIn('minimum_promotion_speedup', benchmark)
        self.assertIn('"cublas_greedy_token"', benchmark)
        self.assertNotIn('"case": "l2_grouped_grid_g8"', benchmark)
        self.assertNotIn('"case": "compact_epp2"', benchmark)
        self.assertNotIn('"case": "compact_epp4"', benchmark)
        self.assertIn('"experts_per_program": int(', benchmark)
        self.assertIn('"active_list_early_exit": bool(', benchmark)
        self.assertIn('"paired_gate_up_dot": bool(', benchmark)
        self.assertIn('"split_gate_up": bool(', benchmark)
        self.assertIn('"empty_expert_early_exit": bool(', benchmark)
        self.assertIn('"l2_grouped_grid": bool(', benchmark)
        self.assertIn('"l2_group_size": int(', benchmark)
        self.assertIn('"active_experts": active_experts', benchmark)
        self.assertIn('"singleton_experts": singleton_experts', benchmark)
        self.assertIn('"empty_experts": num_experts - active_experts', benchmark)
        self.assertIn('"selected_path": selected_path', benchmark)
        self.assertIn('selected_model_layers', benchmark)
        self.assertNotIn('"case": "aligned_active_blocks"', benchmark)
        self.assertNotIn('"case": "aligned_active_blocks_l2"', benchmark)
        self.assertNotIn('"case": "singleton_gemv"', benchmark)
        self.assertNotIn('"singleton_gemv": True', benchmark)
        self.assertIn('"minimum_promotion_speedup": 1.02', benchmark)
        self.assertIn('runtime_config_after_selection', benchmark)
        self.assertIn(
            'compact autotune config drifted immediately after selection',
            benchmark,
        )
        self.assertIn('selected compact decode config was not exercised', benchmark)
        self.assertIn('Gemma4 B16 compact autotune config drifted at runtime', benchmark)
        self.assertIn('"implicit_causal_prefill_hits"', benchmark)
        self.assertIn('"parallel_moe_prefill_hits"', benchmark)
        self.assertIn(
            'Gemma4 B16 parallel MoE prefill was not exercised',
            benchmark,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_PARALLEL_MOE_PREFILL="${MEGAGEMM_GEMMA4_PARALLEL_MOE_PREFILL:-1}"',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_BRIDGE_PREFILL',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_PREFILL=0',
            self.script,
        )
        megagemm_section = benchmark[
            benchmark.index('def run_megagemm('):benchmark.index('def run_vllm_request(')
        ]
        self.assertNotIn('probe_rows', megagemm_section)
        self.assertNotIn('prefill_probe_ms', megagemm_section)
        self.assertIn('scheduler_phase_wall_time', megagemm_section)
        self.assertIn(
            'MegaGemm greedy tokens changed across identical repeats',
            megagemm_section,
        )
        self.assertNotIn('MEGAGEMM_TOKEN_STABILITY_WARNING', megagemm_section)
        self.assertIn('write_partial_checkpoint', megagemm_section)
        self.assertIn('assignment_reference_cosine', benchmark)
        self.assertIn('router_workspace_isolated', benchmark)
        self.assertIn('token_drift_is_diagnostic', megagemm_section)
        self.assertIn('first_cross_backend_mismatch', self.script)
        self.assertIn('PERFORMANCE_AND_TOKEN_PARITY_PASS', self.script)
        self.assertIn('SHAPE_MATCHED_PERFORMANCE_ONLY', self.script)
        self.assertIn('vllm_batch_request_metrics_first_to_finished', benchmark)
        self.assertIn('validated_vllm_phase_span', benchmark)
        self.assertIn('VLLM_BATCH_REQUEST_METRICS', benchmark)
        self.assertIn('adaptive warmup', benchmark)
        self.assertIn('evaluate_vllm_warmup_stability', benchmark)
        self.assertIn('vLLM greedy tokens changed after accepted warmup', benchmark)
        self.assertIn('VLLM_WARMUP_ACCEPTANCE', benchmark)
        self.assertIn('warmup_budget_exhausted_all_tokens_exact', benchmark)
        self.assertIn('REUSE_MEGAGEMM_RESULT', self.script)
        self.assertIn('PERSISTED_MEGAGEMM_RESULT_OK', self.script)
        self.assertIn('persisted_megagemm_current_vllm', self.script)
        self.assertIn('Persisted prompt token manifest is missing', self.script)
        self.assertIn('manifest.get("contract") != result["prompt_contract"]', self.script)
        self.assertIn('COMPARISON_SCOPE="vllm_only"', self.script)
        self.assertIn('== STANDALONE vLLM RESULT ==', self.script)
        self.assertIn('MEGAGEMM_PREFILL_PROFILE_EXCLUDED', benchmark)
        self.assertIn('excluded_from_latency_summary', benchmark)
        self.assertIn('embedding_scale_contract', benchmark)
        self.assertIn('gemma4_embed_scale', self.llama)
        self.assertIn('_scale_token_embeddings', self.llama)
        self.assertIn('MEGAGEMM_TRITON_PREFILL_KV_SCATTER', self.script)
        self.assertIn('_paged_kv_cache_scatter_kernel', self.paged_attention)
        self.assertIn('paged_kv_cache_scatter', self.kv_cache)
        self.assertIn('prefill_kv_scatter', benchmark)
        self.assertNotIn('total_delta_vs_warm_batch_1_token_probe', benchmark)
        self.assertNotIn('probe_rows', benchmark[benchmark.index('def run_vllm('):])
        self.assertNotIn('total_delta_vs_warm_1_token_probe', self.benchmark)
        self.assertNotIn('== vLLM 1-token prefill probe ==', self.benchmark)
        self.assertIn('timestamps from the measured long request', self.benchmark)
        self.assertIn('vLLM request phase metrics unavailable', self.script)
        self.assertNotIn('pure graph replays changed tokens', megagemm_section)
        self.assertNotIn('warmup["token_ids"], replay["token_ids"]', benchmark)
        self.assertIn('stopping before vLLM install', self.benchmark)

    def test_rejected_grouped_mm_decode_experiment_is_retired(self):
        hot_layer = HOT_LAYER_BENCH.read_text(encoding="utf-8")
        retired_wrapper = (
            ROOT / "benchmarks" / "run_gemma4_grouped_mm_decode_tune_colab.sh"
        )

        self.assertFalse(retired_wrapper.exists())
        for source in (hot_layer, self.qwen3_moe):
            self.assertNotIn("torch_grouped_mm", source)
            self.assertNotIn("_grouped_mm", source)
            self.assertNotIn("only-grouped-mm-decode", source)

        batch_bench = BATCH_BENCH.read_text(encoding="utf-8")
        self.assertIn(
            "candidate fell through to a different decode path",
            batch_bench,
        )
        self.assertNotIn("candidate singleton telemetry mismatch", batch_bench)

    def test_batch_prompt_builder_is_distinct_and_shape_stable(self):
        benchmark_dir = str(ROOT / "benchmarks")
        sys.path.insert(0, benchmark_dir)
        try:
            import run_gemma4_moe_batch_vs_vllm as batch_bench
        finally:
            sys.path.remove(benchmark_dir)

        base_ids = [1] + list(range(2, 26))
        exact_prompt = "<bos>|" + ",".join(str(token) for token in base_ids)

        class FakeTokenizer:
            all_special_ids = [0, 1]
            bos_token = "<bos>"
            vocab_size = 128

            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return [int(token) for token in text.split("|", 1)[1].split(",")]

            @staticmethod
            def decode(token_ids, **_kwargs):
                return "<bos>|" + ",".join(str(int(token)) for token in token_ids)

            @staticmethod
            def convert_ids_to_tokens(token_id):
                return f"token_{int(token_id)}"

            @staticmethod
            def __len__():
                return 128

        fake = FakeTokenizer()
        with mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=fake,
        ):
            prompts = batch_bench.build_distinct_equal_length_prompts(
                "unused-local-model",
                exact_prompt,
                required=16,
                expected_tokens=25,
            )
            prompt_texts, prompt_token_ids = (
                batch_bench.build_distinct_equal_length_prompt_inputs(
                    "unused-local-model",
                    exact_prompt,
                    required=16,
                    expected_tokens=25,
                )
            )

        self.assertEqual(len(prompts), 16)
        self.assertEqual(len(set(prompts)), 16)
        self.assertTrue(all(len(fake.encode(prompt)) == 25 for prompt in prompts))
        self.assertEqual(prompt_texts, prompts)
        self.assertEqual(prompt_token_ids, [fake.encode(prompt) for prompt in prompts])
        contract = batch_bench.prompt_token_contract(prompt_token_ids)
        self.assertEqual(contract["rows"], 16)
        self.assertEqual(contract["tokens_per_row"], 25)
        self.assertEqual(contract["distinct_rows"], 16)
        self.assertEqual(len(contract["sha256"]), 64)

    def test_first_token_contract_uses_real_global_request_ids(self):
        benchmark_dir = str(ROOT / "benchmarks")
        sys.path.insert(0, benchmark_dir)
        try:
            import run_gemma4_moe_batch_vs_vllm as batch_bench
        finally:
            sys.path.remove(benchmark_dir)

        request_ids = [41, 73, 105]

        class FakeEngine:
            def __init__(self):
                self._last_scheduler = SimpleNamespace(_completed=[])

            def generate_batch(self, prompts, *, prefill_capture_hook, **_kwargs):
                requests = []
                captures = []
                for row, (prompt, request_id) in enumerate(
                    zip(prompts, request_ids)
                ):
                    del prompt
                    logits = batch_bench.torch.full((8,), -10.0)
                    token = row + 2
                    logits[token] = 10.0
                    req = SimpleNamespace(
                        request_id=request_id,
                        generated_ids=[token],
                    )
                    requests.append(req)
                    captures.append((req, logits))

                self._last_scheduler._completed = list(reversed(requests))
                for req, logits in reversed(captures):
                    prefill_capture_hook(req, logits)

        engine = FakeEngine()
        with mock.patch.object(batch_bench, "sync_cuda", return_value=None):
            result = batch_bench.run_megagemm_first_token_contract(
                engine,
                [[1], [2], [3]],
            )

        self.assertTrue(result["all_finite"])
        self.assertTrue(result["all_exact"])
        self.assertTrue(result["all_reference_exact"])
        self.assertEqual(result["generated_tokens"], [2, 3, 4])
        self.assertEqual(
            [row["request_id"] for row in result["details"]],
            request_ids,
        )

    def test_first_token_contract_reports_nonfinite_rows_without_raising(self):
        benchmark_dir = str(ROOT / "benchmarks")
        sys.path.insert(0, benchmark_dir)
        try:
            import run_gemma4_moe_batch_vs_vllm as batch_bench
        finally:
            sys.path.remove(benchmark_dir)

        class FakeEngine:
            def __init__(self):
                self._last_scheduler = SimpleNamespace(_completed=[])

            def generate_batch(self, prompts, *, prefill_capture_hook, **_kwargs):
                del prompts
                finite_logits = batch_bench.torch.full((8,), -10.0)
                finite_logits[2] = 10.0
                invalid_logits = batch_bench.torch.full((8,), float("nan"))
                requests = [
                    SimpleNamespace(request_id=1, generated_ids=[2]),
                    SimpleNamespace(request_id=2, generated_ids=[0]),
                ]
                self._last_scheduler._completed = requests
                prefill_capture_hook(requests[0], finite_logits)
                prefill_capture_hook(requests[1], invalid_logits)

        with mock.patch.object(batch_bench, "sync_cuda", return_value=None):
            result = batch_bench.run_megagemm_first_token_contract(
                FakeEngine(),
                [[1], [2]],
                reference_tokens=[2, 3],
                raise_on_failure=False,
            )

        self.assertFalse(result["all_finite"])
        self.assertFalse(result["all_exact"])
        self.assertFalse(result["all_reference_exact"])
        self.assertEqual(result["generated_tokens"], [2, 0])
        self.assertTrue(result["details"][0]["reference_exact"])
        self.assertFalse(result["details"][1]["reference_exact"])

    def test_prefill_correctness_gate_rolls_back_first_failing_path(self):
        benchmark_dir = str(ROOT / "benchmarks")
        sys.path.insert(0, benchmark_dir)
        try:
            import run_gemma4_moe_batch_vs_vllm as batch_bench
        finally:
            sys.path.remove(benchmark_dir)
        import megagemm.models.llama as llama_model

        engine = SimpleNamespace(
            model=SimpleNamespace(layers=[], _force_sequential_prefill=False)
        )
        calls = []

        def fake_contract_runner(
            current_engine,
            prompts,
            *,
            reference_tokens=None,
            raise_on_failure=True,
        ):
            del raise_on_failure
            sequential = bool(current_engine.model._force_sequential_prefill)
            parallel_moe = bool(llama_model._GEMMA4_PARALLEL_MOE_PREFILL)
            calls.append((sequential, parallel_moe, reference_tokens))
            tokens = [7, 8]
            finite = [True, True] if sequential or not parallel_moe else [True, False]
            details = []
            for index, (token, row_finite) in enumerate(zip(tokens, finite)):
                reference = (
                    None
                    if reference_tokens is None
                    else int(reference_tokens[index])
                )
                details.append({
                    "finite": row_finite,
                    "exact": row_finite,
                    "reference_exact": bool(
                        reference is None or (row_finite and token == reference)
                    ),
                })
            return {
                "all_finite": all(finite),
                "all_exact": all(finite),
                "all_reference_exact": all(
                    row["reference_exact"] for row in details
                ),
                "generated_tokens": tokens,
                "details": details,
            }

        with mock.patch.object(
            llama_model,
            "_GEMMA4_PARALLEL_MOE_PREFILL",
            True,
        ):
            result = batch_bench.run_gemma4_prefill_correctness_gate(
                engine,
                [[1], [2]],
                contract_runner=fake_contract_runner,
            )
            self.assertFalse(llama_model._GEMMA4_PARALLEL_MOE_PREFILL)
            self.assertFalse(engine.model._force_sequential_prefill)

        self.assertEqual(result["decision"], "APPLY_RUNTIME_ROLLBACK")
        self.assertEqual(result["selected"], "without_parallel_moe")
        self.assertEqual(
            [case["case"] for case in result["cases"]],
            ["current", "without_parallel_moe", "without_parallel_moe_recheck"],
        )
        self.assertEqual(calls[0][:2], (True, True))

    def test_prefill_finite_trace_returns_first_bad_without_reraising(self):
        benchmark_dir = str(ROOT / "benchmarks")
        sys.path.insert(0, benchmark_dir)
        try:
            import run_gemma4_moe_batch_vs_vllm as batch_bench
        finally:
            sys.path.remove(benchmark_dir)

        class FakeModel:
            def __init__(self):
                self.layers = []
                self._force_sequential_prefill = False
                self.trace = None

            def begin_gemma4_prefill_finite_trace(self, *, stop_on_nonfinite):
                self.trace = {
                    "enabled": True,
                    "stop_on_nonfinite": stop_on_nonfinite,
                    "status": "ARMED",
                    "events": [],
                    "first_bad": None,
                }

            def end_gemma4_prefill_finite_trace(self):
                self.trace["enabled"] = False
                return self.trace

        engine = SimpleNamespace(model=FakeModel())

        def fail_at_attention(_engine, _prompts, **_kwargs):
            first_bad = {
                "layer": 0,
                "stage": "attention.core_out",
                "finite_rows": [0],
                "nonfinite_rows": [1],
            }
            engine.model.trace["status"] = "NONFINITE"
            engine.model.trace["first_bad"] = first_bad
            engine.model.trace["events"].append(first_bad)
            raise RuntimeError("first nonfinite tensor")

        with mock.patch.object(
            batch_bench,
            "run_megagemm_first_token_contract",
            side_effect=fail_at_attention,
        ):
            result = batch_bench.run_gemma4_prefill_finite_trace(
                engine,
                [[1], [2]],
            )

        self.assertEqual(result["status"], "NONFINITE")
        self.assertEqual(result["first_bad"]["stage"], "attention.core_out")
        self.assertIn("RuntimeError: first nonfinite tensor", result["error"])
        self.assertFalse(engine.model._force_sequential_prefill)

    def test_parallel_shared_mlp_forks_before_router(self):
        flat_decode = self.llama[
            self.llama.index("def _gemma4_flat_decode_layers") :
            self.llama.index("def _hybrid_flat_decode_layers")
        ]
        self.assertLess(
            flat_decode.index("fork_event.record(parallel_main_stream)"),
            flat_decode.index("lw.moe_module.gate.route("),
        )
        self.assertIn('"fork_before_router": bool(', self.llama)
        self.assertIn('MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_DECODE', self.script)
        self.assertIn('MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_DECODE=0', self.script)
        self.assertNotIn('MEGAGEMM_GEMMA4_FUSED_MOE_ROUTER_DECODE:-0', self.script)
        self.assertIn('_gemma4_a100_a4b_fused_router_decode_shape', self.llama)
        self.assertIn('self._fused_decode_last_path = "fused"', self.llama)
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_NUM_STAGES:-3',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_GATE_NUM_STAGES:-3',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_DOWN_NUM_STAGES:-3',
            self.script,
        )
        self.assertIn(
            "split-pipeline gate must start from the proven 3/3 baseline",
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM=1',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT=0',
            self.script,
        )
        self.assertNotIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EXPERTS_PER_PROGRAM:-1',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_PAIRED_GATE_UP_DOT=0',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_SPLIT_GATE_UP=0',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT=0',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID=0',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_L2_GROUP_SIZE=8',
            self.script,
        )
        self.assertNotIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_STREAMING_WEIGHTS',
            self.script,
        )
        self.assertIn('MEGAGEMM_GEMMA4_BATCH_CUBLAS_LM_HEAD=1', self.script)
        self.assertIn(
            'MEGAGEMM_GEMMA4_FUSED_POST_MOE_NORM_RESIDUAL_DECODE=0',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_FUSED_ROUTER_EXPERT_INPUT_NORM_DECODE=0',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_BRIDGE_DECODE=',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_FUSED_EXPERT_REDUCE_POST_MOE_DECODE=1',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_FUSED_NEXT_ATTN_NORM_DECODE=1',
            self.script,
        )
        self.assertIn(
            "Gemma4 v81 exact next-attention RMSNorm baseline is not active",
            self.script,
        )
        self.assertIn('MEGAGEMM_DECODE_GRAPH_TOKEN_BURST=1', self.script)
        self.assertIn(
            'MEGAGEMM_GEMMA4_B16_GRAPH_TOKEN_BURST_PROVEN=1',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_B16_FUSED_SOFTCAP_ARGMAX_PROVEN=1',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_B16_PERSISTENT_TOKEN_FEEDBACK_PROVEN=1',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX=0',
            self.script,
        )
        self.assertIn(
            'MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK=0',
            self.script,
        )
        self.assertNotIn('MEGAGEMM_QWEN3_MOE_SEGMENTED_PREFILL_BF16_PARTIAL', self.script)

    def test_attn_moe_bridge_materializes_router_input_before_stream_fork(self):
        flat_decode = self.llama[
            self.llama.index("def _gemma4_flat_decode_layers") :
            self.llama.index("def _hybrid_flat_decode_layers")
        ]
        bridge = flat_decode.index(
            "rmsnorm_triton_attn_residual_router_bridge("
        )
        fork = flat_decode.index("fork_event.record(parallel_main_stream)")
        route = flat_decode.index("lw.moe_module.gate.route(")
        self.assertLess(bridge, fork)
        self.assertLess(fork, route)
        self.assertIn("normalized_router_input = bridge_router_in", flat_decode)
        self.assertIn(
            "router_out=self._gemma4_flat_router_input_bufs[layer_idx]",
            flat_decode,
        )
        self.assertIn(
            '"gemma4_fused_attn_moe_router_bridge_decode_hits"',
            self.llama,
        )
        self.assertIn(
            '"fused_attn_moe_router_bridge_decode_hits"',
            self.batch_benchmark,
        )

    def test_router_bridge_is_promoted_and_not_remeasured(self):
        self.assertIn(
            "router_bridge_gate: retired after exact v110 promotion",
            self.script,
        )
        self.assertNotIn(
            "== B16 ATTN-to-MoE/ROUTER BRIDGE GATE (NO MODEL DOWNLOAD) ==",
            self.script,
        )
        self.assertIn(
            "MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_BRIDGE_DECODE=1",
            self.script,
        )

    def test_single_kernel_router_bridge_is_promoted_and_gate_remains_opt_in(self):
        gate = self.script.index(
            "== B16 SINGLE-KERNEL ATTN-to-MoE/ROUTER BRIDGE GATE "
            "(NO MODEL DOWNLOAD) =="
        )
        download = self.script.index("== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES ==")
        self.assertLess(gate, download)
        self.assertIn(
            "MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE=1",
            self.script,
        )
        self.assertIn(
            'RUN_ATTN_MOE_ROUTER_SINGLE_KERNEL_GATE="${RUN_ATTN_MOE_ROUTER_SINGLE_KERNEL_GATE:-0}"',
            self.script,
        )
        self.assertIn(
            "router_single_kernel_gate: retired after exact v127 promotion",
            self.script,
        )
        self.assertIn("APPLY_SINGLE_KERNEL", self.script)
        self.assertIn("KEEP_TWO_KERNEL", self.script)
        self.assertIn("--minimum-speedup 1.02", self.script)
        self.assertIn("--target-gap-ms 0.706", self.script)
        self.assertIn(
            "Stopping before checkpoint download and vLLM installation.",
            self.script,
        )

        gate_source = self.attn_moe_router_single_kernel_gate
        self.assertIn("fused_two_kernel_router_bridge_recheck", gate_source)
        self.assertIn("fused_one_kernel_router_bridge", gate_source)
        self.assertIn("torch.equal", gate_source)
        self.assertIn("alias_exact", gate_source)
        self.assertIn("baseline_stability_ratio", gate_source)

        self.assertIn(
            "def rmsnorm_triton_attn_residual_router_bridge_single(",
            self.rmsnorm_triton,
        )
        self.assertIn(
            "_rmsnorm_attn_residual_router_bridge_kernel",
            self.rmsnorm_triton,
        )
        self.assertIn(
            "rmsnorm_triton_attn_residual_router_bridge_single(",
            self.llama,
        )
        self.assertIn(
            '"gemma4_fused_attn_moe_router_single_kernel_decode_hits"',
            self.llama,
        )
        self.assertIn(
            '"fused_attn_moe_router_single_kernel_decode_hits"',
            self.batch_benchmark,
        )
        self.assertIn(
            '"MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE",\n'
            '    default=True,',
            self.llama,
        )
        self.assertIn(
            "Gemma4 B16 promoted single-kernel router bridge is not active",
            self.script,
        )
        self.assertIn(
            "MegaGemm B={batch_size} decode router bridge:",
            self.script,
        )
        self.assertIn(
            "single-kernel attention-to-MoE/router bridge selection",
            self.batch_benchmark,
        )

    def test_router_compact_pack_gate_is_retired_but_remains_opt_in(self):
        gate = self.script.index(
            "== B16 ROUTER TOP-K/COMPACT-PACK GATE (NO MODEL DOWNLOAD) =="
        )
        download = self.script.index("== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES ==")
        self.assertLess(gate, download)
        self.assertIn(
            '"GEMMA4 B16 ROUTER COMPACT-PACK GATE"',
            self.script,
        )
        self.assertIn("--minimum-speedup 1.02", self.script)
        self.assertIn("--target-gap-ms 0.85", self.script)
        self.assertIn(
            "Stopping before checkpoint download and vLLM installation.",
            self.script,
        )
        self.assertIn('RUN_ROUTER_COMPACT_PACK_GATE="${RUN_ROUTER_COMPACT_PACK_GATE:-0}"', self.script)
        self.assertIn('STOP_IF_ROUTER_COMPACT_PACK_REJECTED="${STOP_IF_ROUTER_COMPACT_PACK_REJECTED:-0}"', self.script)
        self.assertIn(
            "router_compact_pack_gate: retired after v111 (12.67x slower)",
            self.script,
        )
        self.assertIn(
            "MEGAGEMM_GEMMA4_FUSED_ROUTER_COMPACT_PACK_DECODE=0",
            self.script,
        )

    def test_batch_ab_uses_one_exact_pretokenized_input_manifest(self):
        self.assertEqual(
            self.script.count(
                '--prompt-token-ids-json "${PROMPT_TOKEN_IDS_JSON}"'
            ),
            2,
        )
        self.assertIn(
            'PROMPT_TOKEN_IDS_JSON="${OUT_DIR}/prompt_token_ids.json"',
            self.script,
        )
        self.assertIn("def load_or_create_prompt_token_ids(", self.batch_benchmark)
        self.assertIn('"format": "prompt_token_ids"', self.batch_benchmark)
        self.assertIn('"prompt_token_ids": [int(token) for token in row]', self.batch_benchmark)
        self.assertIn("detokenize=False", self.batch_benchmark)
        self.assertIn("decode_outputs=False", self.batch_benchmark)
        self.assertIn("materialize_generated_tokens=True", self.batch_benchmark)
        self.assertIn("if args.profile_breakdown:", self.batch_benchmark)
        self.assertIn("engine.profile_decode_breakdown(", self.batch_benchmark)
        self.assertIn(
            "Cross-backend prompt token contract mismatch",
            self.script,
        )
        self.assertIn(
            "prompts: List[Union[str, List[int]]]",
            self.engine,
        )

    def test_router_topk_fuses_deterministic_compact_pack(self):
        self.assertIn(
            "def _qwen3_moe_topk_softmax_compact_pack_kernel(",
            self.qwen3_moe,
        )
        self.assertIn(
            "for row in tl.static_range(0, ROWS):",
            self.qwen3_moe,
        )
        self.assertIn("if route_prepacked:", self.qwen3_moe)
        self.assertIn(
            'workspace["expert_grouped_compact_route_prepacked_hits"]',
            self.qwen3_moe,
        )
        flat_decode = self.llama[
            self.llama.index("def _gemma4_flat_decode_layers") :
            self.llama.index("def _hybrid_flat_decode_layers")
        ]
        route = flat_decode.index("lw.moe_module.gate.route(")
        experts = flat_decode.index("expert_out = lw.moe_module.experts(")
        self.assertLess(route, experts)
        self.assertIn(
            "compact_route_workspace=compact_route_workspace",
            flat_decode,
        )
        self.assertIn(
            "compact_route_prepacked=compact_route_prepacked",
            flat_decode,
        )
        self.assertIn(
            '"gemma4_fused_router_compact_pack_decode_hits"',
            self.llama,
        )
        self.assertIn(
            '"gemma4_router_compact_pack_workspace_disabled_layers"',
            self.llama,
        )
        self.assertIn(
            '"expert_grouped_compact_route_prepacked_fail_reason"',
            self.llama,
        )
        self.assertIn(
            "qwen3_moe_topk_softmax_compact_pack",
            self.router_compact_pack_gate,
        )

    def test_flat_decode_fuses_post_moe_norms_and_residual_after_stream_join(self):
        flat_decode = self.llama[
            self.llama.index("def _gemma4_flat_decode_layers") :
            self.llama.index("def _hybrid_flat_decode_layers")
        ]
        join = flat_decode.index("parallel_main_stream.wait_event(parallel_join_event)")
        fused = flat_decode.index("rmsnorm_triton_pair_add_final_residual(")
        self.assertLess(join, fused)
        self.assertIn("out=residual", flat_decode)
        self.assertIn(
            '"gemma4_fused_post_moe_norm_residual_decode_hits"',
            self.llama,
        )

    def test_b16_fuses_expert_reduce_with_post_moe_chain_in_persistent_output(self):
        flat_decode = self.llama[
            self.llama.index("def _gemma4_flat_decode_layers") :
            self.llama.index("def _hybrid_flat_decode_layers")
        ]
        compact_decode = self.qwen3_moe[
            self.qwen3_moe.index("def _qwen3_moe_expert_grouped_compact_decode") :
            self.qwen3_moe.index("def _qwen3_moe_shared_route_decode")
        ]

        down = compact_decode.index(
            "_qwen3_moe_expert_grouped_compact_down_partial_kernel["
        )
        wait = compact_decode.index(
            "torch.cuda.current_stream(hidden.device).wait_event(post_moe_wait_event)"
        )
        fused = compact_decode.index(
            "_qwen3_moe_assignment_reduce_gemma4_post_kernel["
        )
        self.assertLess(down, wait)
        self.assertLess(wait, fused)
        self.assertIn("post_moe_shared=(down_out if fused_expert_reduce_post_moe else None)", flat_decode)
        self.assertIn("self._gemma4_flat_post_moe_out_bufs[layer_idx]", flat_decode)
        self.assertIn("hidden = expert_out", flat_decode)
        self.assertIn("isolated_post_moe_output_buffers", self.llama)
        self.assertIn("fused Gemma4 post-MoE cannot fall back", self.llama)
        self.assertIn(
            '"gemma4_fused_expert_reduce_post_moe_decode_hits"',
            self.llama,
        )
        batch_bench = BATCH_BENCH.read_text(encoding="utf-8")
        self.assertIn(
            "Gemma4 fused expert reduction/post-MoE chain was not exercised",
            batch_bench,
        )

    def test_b16_can_chain_post_moe_output_into_next_attention_norm(self):
        flat_decode = self.llama[
            self.llama.index("def _gemma4_flat_decode_layers") :
            self.llama.index("def _hybrid_flat_decode_layers")
        ]
        self.assertIn(
            "self._gemma4_flat_next_attn_norm_bufs[layer_idx - 1]",
            flat_decode,
        )
        self.assertIn(
            "post_moe_layer_scalar=(",
            flat_decode,
        )
        self.assertIn(
            "post_moe_next_norm_weight=(",
            flat_decode,
        )
        self.assertIn(
            "post_moe_write_next_norm=write_next_attn_norm",
            flat_decode,
        )
        self.assertIn("if not fuse_next_attn_norm:", flat_decode)
        self.assertIn("FUSE_LAYER_SCALAR: tl.constexpr", self.qwen3_moe)
        self.assertIn("WRITE_NEXT_NORM: tl.constexpr", self.qwen3_moe)
        self.assertIn(
            '"gemma4_fused_next_attn_norm_decode_hits"',
            self.llama,
        )
        benchmark = BATCH_BENCH.read_text(encoding="utf-8")
        self.assertIn(
            "USE_PROVEN_EXACT_BASELINE",
            benchmark,
        )
        self.assertIn(
            '"run_id": "gemma4_moe_ab_20260727_004118"',
            benchmark,
        )
        self.assertIn(
            '"cases": []',
            benchmark,
        )
        self.assertIn(
            "Gemma4 post-MoE/next-attention RMSNorm selection drifted",
            benchmark,
        )

    def test_flat_decode_can_share_router_and_expert_input_variance(self):
        flat_decode = self.llama[
            self.llama.index("def _gemma4_flat_decode_layers") :
            self.llama.index("def _hybrid_flat_decode_layers")
        ]
        self.assertIn(
            "rmsnorm_triton_weighted_scaled_no_weight_dual(",
            flat_decode,
        )
        self.assertIn(
            "normalized_router_input=normalized_router_input",
            flat_decode,
        )
        self.assertIn(
            '"gemma4_fused_router_expert_input_norm_decode_hits"',
            self.llama,
        )

    def test_batch_greedy_token_lm_head_is_gated_and_exercised(self):
        benchmark = BATCH_BENCH.read_text(encoding="utf-8")
        for expected in (
            '_GEMMA4_BATCH_CUBLAS_LM_HEAD',
            '_gemma4_a100_a4b_batch_cublas_lm_head_shape',
            '_gemma4_batch_cublas_lm_head_tokens',
            'gemma4_batch_cublas_lm_head_hits',
            'prefers_scheduler_greedy_token_decode',
        ):
            self.assertIn(expected, self.llama)
        for expected in (
            'def run_batch_lm_head_kernel_gate(',
            '"current_logits_cap_argmax"',
            '"cublas_greedy_token"',
            'model._decode_logits_from_hidden(hidden)',
            'return model._decode_next_token_greedy(hidden)',
            'model._gemma4_batch_cublas_lm_head_hits = 0',
            '"greedy_token_speedup_vs_current"',
            'def run_scheduler_token_burst_gate(',
            'MEGAGEMM_SCHEDULER_BURST_GATE',
            '"graph_token_burst"',
            '"fused_softcap_argmax"',
            'APPLY_FUSED_SOFTCAP_ARGMAX',
            '"first_post_capture_step_logits"',
            'priming_tokens_exact_diagnostic_only',
            'persistent_priming_tokens_exact_diagnostic_only',
            'softcap_capture_evidence',
            'batch_fused_softcap_argmax_gate_contract',
            'batch_fused_softcap_argmax_replay_exercised',
            'USE_PROVEN_FUSED_SOFTCAP_ARGMAX',
            'batch_cublas_lm_head_graph_replay',
            'Gemma4 selected batch LM-head backend was not exercised',
            'Gemma4 greedy-token CUDA graph was not exercised',
        ):
            self.assertIn(expected, benchmark)
        self.assertIn('expected_persistent_steps = expected_decode_steps', benchmark)
        self.assertIn(
            '!= result["scheduler_greedy_token_steps"]',
            benchmark,
        )
        self.assertNotIn(
            'or result["batch_fused_softcap_argmax_hits"] <= 0',
            benchmark,
        )
        self.assertIn('"minimum_promotion_speedup": 1.002', benchmark)
        self.assertIn('align_vllm_outputs_to_prompts(outputs, prompts)', benchmark)
        measured_case = benchmark.index('cases[str(batch_size)] = {')
        pending_checkpoint = benchmark.index(
            'runtime_validation={"status": "pending"}',
            measured_case,
        )
        runtime_validation = benchmark.index(
            'runtime = validate_megagemm_runtime(',
            measured_case,
        )
        self.assertLess(pending_checkpoint, runtime_validation)
        self.assertIn(
            '"MEGAGEMM_GEMMA4_BATCH_CUBLAS_LM_HEAD",\n    # Paid A100/BF16/B16 validation',
            self.llama,
        )
        self.assertIn('default=True,', self.llama)
        for expected in (
            'return_next_token=greedy_token_decode',
            "'greedy_token_steps'",
            "'batched_token_host_copies'",
            "'vectorized_input_updates'",
            "_decode_graph_token_burst_batch",
            "'token_bursts'",
            "'token_burst_steps'",
            'tokens_host.copy_(next_tokens.detach(), non_blocking=False)',
        ):
            self.assertIn(expected, self.scheduler)
        self.assertNotIn('"current_fused_rowwise"', benchmark)
        self.assertNotIn('"cublas_gemm_argmax"', benchmark)
        self.assertIn('def _decode_raw_logits_from_hidden(', self.llama)
        self.assertIn('raw_logits = self._decode_raw_logits_from_hidden(hidden)', self.llama)
        self.assertIn('logits_softcap_argmax(', self.llama)
        self.assertIn('def _decode_logits_from_hidden(', self.llama)
        self.assertIn('return self._decode_logits_from_hidden(hidden)', self.llama)

    def test_paired_gate_up_dot_is_a_complete_tunable_kernel_path(self):
        self.assertIn('PAIRED_GATE_UP_DOT: tl.constexpr', self.qwen3_moe)
        self.assertIn(
            'pair_acc = tl.zeros((BLOCK_M, BLOCK_N * 2), dtype=tl.float32)',
            self.qwen3_moe,
        )
        self.assertIn('pair_acc += tl.dot(x, pair_w, out_dtype=tl.float32)', self.qwen3_moe)
        self.assertIn('gate_acc, up_acc = tl.split(pair_acc)', self.qwen3_moe)
        self.assertIn('PAIRED_GATE_UP_DOT=bool(', self.qwen3_moe)

    def test_rejected_streaming_expert_weight_path_was_removed(self):
        benchmark = BATCH_BENCH.read_text(encoding="utf-8")
        for rejected in (
            '_CFG_EXPERT_GROUPED_COMPACT_STREAMING_WEIGHTS',
            'STREAMING_WEIGHTS: tl.constexpr',
            'cache_modifier=".cg"',
            'eviction_policy="evict_first"',
            'STREAMING_WEIGHTS=bool(',
            'expert_grouped_compact_decode_last_streaming_weights',
            'expert_grouped_compact_streaming_weights',
        ):
            self.assertNotIn(rejected, self.qwen3_moe)
        for rejected in (
            '"case": "streaming_weight_loads"',
            '"streaming_weights": True',
            'compact_streaming_weights',
        ):
            self.assertNotIn(rejected, benchmark)

    def test_split_gate_up_is_a_complete_tunable_kernel_path(self):
        self.assertIn(
            'def _qwen3_moe_expert_grouped_compact_gate_up_split_kernel(',
            self.qwen3_moe,
        )
        self.assertIn(
            'def _qwen3_moe_expert_grouped_compact_swiglu_kernel(',
            self.qwen3_moe,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_SPLIT_GATE_UP',
            self.qwen3_moe,
        )
        self.assertIn('if use_split_gate_up:', self.qwen3_moe)

    def test_empty_expert_early_exit_is_a_complete_tunable_kernel_path(self):
        self.assertIn('EMPTY_EXPERT_EARLY_EXIT: tl.constexpr', self.qwen3_moe)
        self.assertIn(
            'if EMPTY_EXPERT_EARLY_EXIT and EXPERT_GRID and EXPERTS_PER_PROGRAM == 1:',
            self.qwen3_moe,
        )
        self.assertIn('if tl.load(counts_ptr + candidate_group) <= 0:', self.qwen3_moe)
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_EMPTY_EXPERT_EARLY_EXIT',
            self.qwen3_moe,
        )
        self.assertIn('EMPTY_EXPERT_EARLY_EXIT=bool(', self.qwen3_moe)

    def test_active_list_early_exit_is_graph_safe_and_complete(self):
        self.assertIn('ACTIVE_LIST_EARLY_EXIT: tl.constexpr', self.qwen3_moe)
        self.assertIn(
            'if ACTIVE_LIST_EARLY_EXIT and ACTIVE_LIST and '
            'EXPERTS_PER_PROGRAM == 1:',
            self.qwen3_moe,
        )
        self.assertIn(
            'active_count = tl.load(unique_or_count_ptr).to(tl.int64)',
            self.qwen3_moe,
        )
        self.assertIn('if candidate_group >= active_count:', self.qwen3_moe)
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT',
            self.scheduler,
        )

    def test_l2_grouped_grid_is_a_complete_tunable_kernel_path(self):
        self.assertIn(
            'def _qwen3_moe_expert_grouped_compact_active_blocks_kernel(',
            self.qwen3_moe,
        )
        self.assertIn('rank = tl.cumsum(active.to(tl.int32), axis=0) - 1', self.qwen3_moe)
        self.assertIn('L2_GROUPED_GRID: tl.constexpr', self.qwen3_moe)
        self.assertIn('num_pid_in_group = L2_GROUP_SIZE * NUM_PID_N', self.qwen3_moe)
        self.assertIn(
            'candidate_group = first_candidate + (pid_in_group % group_size)',
            self.qwen3_moe,
        )
        self.assertIn(
            'MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_L2_GROUPED_GRID',
            self.qwen3_moe,
        )
        self.assertIn('L2_GROUPED_GRID=use_l2_grouped_grid', self.qwen3_moe)
        self.assertIn(
            'and (use_expert_grid_pack or use_active_list)',
            self.qwen3_moe,
        )


class Gemma4ExpertPrefillTunerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = EXPERT_PREFILL_TUNER.read_text(encoding="utf-8")
        cls.benchmark = HOT_LAYER_BENCH.read_text(encoding="utf-8")

    def test_embedded_python_blocks_compile(self):
        blocks = re.findall(r"<<'PY'\r?\n(.*?)\r?\nPY", self.script, re.DOTALL)
        self.assertGreater(len(blocks), 0)
        for index, block in enumerate(blocks, start=1):
            compile(block, f"{EXPERT_PREFILL_TUNER}:heredoc-{index}", "exec")

    def test_tuner_is_bounded_and_never_downloads_a_model(self):
        self.assertIn("BENCH_TIMEOUT_MIN", self.script)
        self.assertIn("timeout --foreground", self.script)
        self.assertIn("--only-expert-prefill", self.script)
        self.assertIn("--prefill-target-savings-ms 7.50", self.script)
        self.assertIn("--prefill-fixed-route-pack-only", self.script)
        self.assertNotIn("snapshot_download", self.script)
        self.assertNotIn("pip install -q -U vllm", self.script)

    def test_tuner_compares_against_the_current_production_shape(self):
        for expected in (
            '"block_m": 16',
            '"block_n": 64',
            '"block_k": 128',
            '"fused_gate_block_n": 64',
            '"warps": 8',
            '"stages": 4',
            '"fixed_route_pack": False',
            'axis="stability"',
            '"closes_full_ab_total_gap"',
        ):
            self.assertIn(expected, self.benchmark)


class Gemma4LongDecodeAttentionTunerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuner = LONG_DECODE_ATTN_TUNER.read_text(encoding="utf-8")
        cls.wrapper = LONG_DECODE_ATTN_WRAPPER.read_text(encoding="utf-8")
        cls.shape_tuner = LONG_DECODE_ATTN_SHAPE_TUNER.read_text(
            encoding="utf-8"
        )
        cls.shape_wrapper = LONG_DECODE_ATTN_SHAPE_WRAPPER.read_text(
            encoding="utf-8"
        )
        cls.frontier = LONG_DECODE_ATTN_FRONTIER.read_text(encoding="utf-8")
        cls.frontier_wrapper = LONG_DECODE_ATTN_FRONTIER_WRAPPER.read_text(
            encoding="utf-8"
        )
        cls.vllm_parity = LONG_DECODE_ATTN_VLLM_PARITY.read_text(
            encoding="utf-8"
        )
        cls.vllm_parity_wrapper = LONG_DECODE_ATTN_VLLM_PARITY_WRAPPER.read_text(
            encoding="utf-8"
        )
        cls.long_context_harness = LONG_CONTEXT_HARNESS.read_text(
            encoding="utf-8"
        )
        cls.long_context_bench = LONG_CONTEXT_BENCH.read_text(encoding="utf-8")
        cls.paged_attention = PAGED_ATTENTION.read_text(encoding="utf-8")

    def test_fresh_vm_gate_is_bounded_and_model_free(self):
        blocks = re.findall(
            r"<<'PY'\r?\n(.*?)\r?\nPY",
            self.wrapper,
            re.DOTALL,
        )
        self.assertGreater(len(blocks), 0)
        for index, block in enumerate(blocks, start=1):
            compile(block, f"{LONG_DECODE_ATTN_WRAPPER}:heredoc-{index}", "exec")
        self.assertIn('INSTALL_RUNTIME="${INSTALL_RUNTIME:-1}"', self.wrapper)
        self.assertIn("timeout --foreground", self.wrapper)
        self.assertIn("model_download: disabled", self.wrapper)
        self.assertIn("vllm_install: disabled", self.wrapper)
        self.assertNotIn("snapshot_download", self.wrapper)
        self.assertNotIn("pip install -q -U vllm", self.wrapper)

    def test_tuner_preserves_the_current_default_until_promoted(self):
        self.assertIn('segment_counts: tuple[int, ...]', self.tuner)
        self.assertIn('rows = [make_case(16)]', self.tuner)
        self.assertIn('rows.append(make_case(16, "_recheck"))', self.tuner)
        self.assertIn('"minimum_speedup"', self.tuner)
        self.assertIn('"APPLY_LONG_SEGMENTS"', self.tuner)
        self.assertIn('num_segments_override: Optional[int] = None', self.paged_attention)
        self.assertIn(
            'if num_segments not in (4, 8, 16, 32):',
            self.paged_attention,
        )

    def test_shape_gate_is_bounded_and_uses_the_promoted_baseline(self):
        blocks = re.findall(
            r"<<'PY'\r?\n(.*?)\r?\nPY",
            self.shape_wrapper,
            re.DOTALL,
        )
        self.assertGreater(len(blocks), 0)
        for index, block in enumerate(blocks, start=1):
            compile(
                block,
                f"{LONG_DECODE_ATTN_SHAPE_WRAPPER}:heredoc-{index}",
                "exec",
            )
        self.assertIn("timeout --foreground", self.shape_wrapper)
        self.assertIn("model_download: disabled", self.shape_wrapper)
        self.assertIn("vllm_install: disabled", self.shape_wrapper)
        self.assertNotIn("snapshot_download", self.shape_wrapper)
        self.assertNotIn("pip install -q -U vllm", self.shape_wrapper)
        self.assertIn('LaunchConfig(32, 32, 4, 3, 4)', self.shape_tuner)
        self.assertIn('LaunchConfig(8, 16, 4, 3, 4)', self.shape_tuner)
        self.assertIn('tile_size_override=config.tile_size', self.shape_tuner)
        self.assertIn('tile_size_override: Optional[int] = None', self.paged_attention)
        self.assertIn('_grouped_segmented_decode_tile_size(', self.paged_attention)
        self.assertIn('float(result["speedup"]) >= args.minimum_speedup', self.shape_tuner)

    def test_tile64_segment_frontier_is_bounded_and_model_free(self):
        blocks = re.findall(
            r"<<'PY'\r?\n(.*?)\r?\nPY",
            self.frontier_wrapper,
            re.DOTALL,
        )
        self.assertGreater(len(blocks), 0)
        for index, block in enumerate(blocks, start=1):
            compile(
                block,
                f"{LONG_DECODE_ATTN_FRONTIER_WRAPPER}:heredoc-{index}",
                "exec",
            )
        self.assertIn("timeout --foreground", self.frontier_wrapper)
        self.assertIn("model_download: disabled", self.frontier_wrapper)
        self.assertIn("vllm_install: disabled", self.frontier_wrapper)
        self.assertNotIn("snapshot_download", self.frontier_wrapper)
        self.assertNotIn("pip install -q -U vllm", self.frontier_wrapper)
        self.assertIn("BASELINE = LaunchConfig(32, 64, 4, 3, 4)", self.frontier)
        for segments in (4, 8, 16, 32):
            self.assertIn(f"LaunchConfig({segments}, 64, 4, 3, 4)", self.frontier)
        self.assertIn('"APPLY_SLIDING_SEGMENTS"', self.frontier)
        self.assertIn("TARGET_REMAINING_DECODE_GAP_MS", self.frontier)

    def test_long_ab_requires_the_paid_segment_and_tile_contract(self):
        self.assertIn(
            'HARNESS_REV="gemma4-long-context-ab-v37-restore-all-prefill-candidates"',
            self.long_context_harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_PREFILL=1",
            self.long_context_harness,
        )
        for expected in (
            'sliding_segments == 32',
            'full_segments == 8',
            'sliding_tile_size == 64',
            'full_tile_size == 16',
            'grouped_segmented_selected_tile_sizes',
        ):
            self.assertIn(expected, self.long_context_bench)
        self.assertIn(
            '"grouped_segmented_selected_tile_sizes"',
            self.paged_attention,
        )

    def test_long_vllm_attention_parity_is_checkpoint_free_and_current(self):
        blocks = re.findall(
            r"<<'PY'\r?\n(.*?)\r?\nPY",
            self.vllm_parity_wrapper,
            re.DOTALL,
        )
        self.assertGreater(len(blocks), 0)
        for index, block in enumerate(blocks, start=1):
            compile(
                block,
                f"{LONG_DECODE_ATTN_VLLM_PARITY_WRAPPER}:heredoc-{index}",
                "exec",
            )
        self.assertIn("timeout --foreground", self.vllm_parity_wrapper)
        self.assertIn("model_download: disabled", self.vllm_parity_wrapper)
        self.assertIn("huggingface_download: disabled", self.vllm_parity_wrapper)
        self.assertNotIn("snapshot_download", self.vllm_parity_wrapper)
        self.assertIn('"vllm==${PINNED_VLLM_VERSION}"', self.vllm_parity_wrapper)
        self.assertIn("_triton_paged_decode_grouped_segmented", self.vllm_parity)
        self.assertIn("megagemm_segments=32", self.vllm_parity)
        self.assertIn("megagemm_tile_size=64", self.vllm_parity)
        self.assertIn("megagemm_segments=8", self.vllm_parity)
        self.assertIn("megagemm_tile_size=16", self.vllm_parity)
        self.assertIn('keys = keys[-sliding_window:]', self.vllm_parity)
        self.assertIn('"PORT_VLLM_LONG_ATTENTION_CORE"', self.vllm_parity)
        self.assertIn("LONG_REMAINING_DECODE_GAP_MS = 120.58", self.vllm_parity)

if __name__ == "__main__":
    unittest.main()

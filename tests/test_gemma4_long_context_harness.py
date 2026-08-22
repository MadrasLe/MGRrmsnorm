import importlib.util
import json
import os
import re
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "benchmarks" / "run_gemma4_long_context_vs_vllm_colab.sh"
RUNNER = ROOT / "benchmarks" / "run_gemma4_long_context_vs_vllm.py"
BURST_GATE = ROOT / "benchmarks" / "run_gemma4_long_decode_burst_gate.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("gemma4_long_context_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load long-context runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_burst_gate():
    spec = importlib.util.spec_from_file_location("gemma4_long_decode_burst_gate", BURST_GATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load long decode burst gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Gemma4LongContextHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.harness = HARNESS.read_text(encoding="utf-8")
        cls.runner_source = RUNNER.read_text(encoding="utf-8")
        cls.burst_gate_source = BURST_GATE.read_text(encoding="utf-8")

    def test_embedded_python_blocks_compile(self):
        blocks = re.findall(r"<<'PY'\r?\n(.*?)\r?\nPY", self.harness, re.DOTALL)
        self.assertGreaterEqual(len(blocks), 2)
        for index, block in enumerate(blocks, start=1):
            compile(block, f"long_context_harness_block_{index}.py", "exec")

        command = re.search(r"python -c \$'(.*?)'\r?\n", self.harness)
        self.assertIsNotNone(command)
        expanded = bytes(command.group(1), "utf-8").decode("unicode_escape")
        compile(expanded, "long_context_download_command.py", "exec")

    def test_paid_work_starts_only_after_early_vram_preflight(self):
        gpu = self.harness.index("== BASE GPU AND EARLY VRAM PREFLIGHT ==")
        install = self.harness.index("== INSTALL BASE RUNTIME ==")
        download = self.harness.index("== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES ==")
        megagemm = self.harness.index("== RUN MEGAGEMM LONG-CONTEXT SWEEP ==")
        vllm = self.harness.index("== RUN vLLM LONG-CONTEXT SWEEP ==")
        self.assertLess(gpu, install)
        self.assertLess(install, download)
        self.assertLess(download, megagemm)
        self.assertLess(megagemm, vllm)
        self.assertIn("No package installation or checkpoint download was started", self.harness)
        self.assertIn("MIN_BF16_AB_VRAM_MIB=71680", self.harness)

    def test_default_matrix_is_bounded_and_actually_long(self):
        self.assertIn(
            'HARNESS_REV="gemma4-long-context-ab-v37-restore-all-prefill-candidates"',
            self.harness,
        )
        self.assertIn('CONTEXTS="${CONTEXTS:-2048}"', self.harness)
        self.assertIn('BATCH_SIZES="${BATCH_SIZES:-16}"', self.harness)
        self.assertIn('MAX_SEQ_LEN="${MAX_SEQ_LEN:-2112}"', self.harness)
        self.assertIn('MAX_TOTAL_PREFILL_TOKENS="${MAX_TOTAL_PREFILL_TOKENS:-32768}"', self.harness)
        self.assertIn('VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-0}"', self.harness)
        self.assertIn("MAX_CONTEXT + MAX_TOKENS", self.harness)
        self.assertIn("MAX_CONTEXT * MAX_BATCH", self.harness)
        self.assertIn("(full batch)", self.harness)
        self.assertIn(
            'MEGAGEMM_DETERMINISTIC_PREFILL_MAX_BATCHED_TOKENS="${MEGAGEMM_DETERMINISTIC_PREFILL_MAX_BATCHED_TOKENS:-32768}"',
            self.harness,
        )
        self.assertIn('MEGAGEMM_MIN_WARMUPS="${MEGAGEMM_MIN_WARMUPS:-3}"', self.harness)
        self.assertIn('MEGAGEMM_MAX_WARMUPS="${MEGAGEMM_MAX_WARMUPS:-8}"', self.harness)
        self.assertIn("--megagemm-required-stable-warmup-pairs", self.harness)
        self.assertIn("full token matrix + runtime topology + timing", self.harness)
        self.assertIn(
            "export MEGAGEMM_DECODE_CUDA_GRAPHS_SHARED_SHAPE_CACHE=0",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_REUSE_REQUEST_SCHEDULER=1",
            self.harness,
        )
        self.assertIn(
            "execution_policy: promoted one-chunk 32k prefill + async expert metadata + contiguous FP32 partial + burst8 decode",
            self.harness,
        )
        self.assertNotIn("--megagemm-prefill-chunk-autotune", self.harness)
        self.assertNotIn("--megagemm-prefill-chunk-gate-repeats", self.harness)
        self.assertIn("the paid loaded checkpoint gate is retired", self.harness)
        self.assertIn("export MEGAGEMM_DECODE_GRAPH_TOKEN_BURST=0", self.harness)
        self.assertIn(
            "export MEGAGEMM_GEMMA4_B16_GRAPH_TOKEN_BURST_PROVEN=0",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_B16_LONG_GRAPH_TOKEN_BURST_PROVEN=1",
            self.harness,
        )
        self.assertIn(
            "persistent_graph_feedback: hard disabled",
            self.harness,
        )
        self.assertIn(
            '"A100" in str(runtime_gpu.get("name") or "").upper()',
            self.runner_source,
        )
        self.assertIn("export MEGAGEMM_GEMMA4_LONG_SLIDING_PREFILL=1", self.harness)
        self.assertIn("export MEGAGEMM_GEMMA4_LONG_FULL_PREFILL=1", self.harness)
        self.assertIn(
            "long_attention_prepare: fused RMSNorm+RoPE+layouts promoted",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_BLOCK_M=64",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_BLOCK_N=256",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_BLOCK_K=64",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_FUSED_GATE_BLOCK_N=128",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_NUM_WARPS=4",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_NUM_STAGES=3",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_COMPACT_ROUTE_PACK=0",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_LONG_PADDED_BMM_PREFILL=0",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_PREFILL=1",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_MIN_SKEW=7.5",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_MAX_LIGHT_PADDING_RATIO=1.25",
            self.harness,
        )
        self.assertIn(
            "adaptive excluded segmented reference, exact token/topology gate",
            self.harness,
        )
        self.assertIn("run_gemma4_long_skew_segmented_prefill_microbench.py", self.harness)
        self.assertIn('RUN_LONG_SORTED_PARTIAL_GATE="${RUN_LONG_SORTED_PARTIAL_GATE:-0}"', self.harness)
        self.assertIn("STOP_IF_LONG_SORTED_PARTIAL_GATE_REJECTED", self.harness)
        self.assertIn(
            "Stopping before checkpoint download and vLLM installation",
            self.harness,
        )
        self.assertIn(
            "MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_ASYNC_TILES_MAX_ASSIGNMENTS",
            self.harness,
        )
        self.assertIn(
            "MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_SORTED_PARTIAL",
            self.harness,
        )
        self.assertIn(
            "export MEGAGEMM_GEMMA4_MOE_PREFILL_LONG_SORTED_PARTIAL=1",
            self.harness,
        )
        self.assertIn("promoted by v30 exact 1.093x gate", self.harness)
        self.assertIn(
            'RUN_DECODE_ACTIVE_LIST_FRONTIER_GATE="${RUN_DECODE_ACTIVE_LIST_FRONTIER_GATE:-0}"',
            self.harness,
        )
        self.assertIn(
            "--active-expert-profiles 8,16,32,64,90,128",
            self.harness,
        )
        active_gate = self.harness.index(
            "== CHECKPOINT-FREE B16 DECODE ACTIVE-LIST FRONTIER GATE =="
        )
        download = self.harness.index(
            "== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES =="
        )
        self.assertLess(active_gate, download)
        self.assertIn(
            "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST=1",
            self.harness,
        )
        self.assertIn(
            "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT=1",
            self.harness,
        )
        self.assertIn(
            "v32 promoted exact geomean=1.473x low-active=2.143x max-regression=0.51%",
            self.harness,
        )
        self.assertIn("--route-normalized-diagnostic", self.harness)
        self.assertIn("--route-normalized-repeats", self.harness)
        self.assertIn("--require-request-scheduler-reuse", self.harness)
        self.assertIn("--profile-prefill-stages", self.harness)
        self.assertNotIn("--profile-decode-stages", self.harness)
        self.assertIn("one compatible idle Scheduler persists", self.harness)
        self.assertIn("full LM head, one identical continuation token", self.harness)
        self.assertIn("--megagemm-determinism-auto-fallback", self.harness)

    def test_resume_is_rejected_before_paid_work_for_ephemeral_vms(self):
        self.assertIn('RESUME_MEGAGEMM="${RESUME_MEGAGEMM:-0}"', self.harness)
        self.assertNotIn('MEGAGEMM_RESUME_ARGS+=(--resume)', self.harness)
        self.assertNotIn('"${MEGAGEMM_RESUME_ARGS[@]}"', self.harness)
        rejected = self.harness.index("RESUME_MEGAGEMM is unsupported")
        install = self.harness.index("== INSTALL BASE RUNTIME ==")
        download = self.harness.index("== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES ==")
        self.assertLess(rejected, install)
        self.assertLess(rejected, download)
        self.assertIn(
            "No package installation, checkpoint download, or GPU benchmark was started",
            self.harness,
        )

    def test_vllm_version_uses_proven_loader_and_blocks_known_bad_release(self):
        self.assertIn(
            'PINNED_VLLM_VERSION="${PINNED_VLLM_VERSION:-0.24.0}"',
            self.harness,
        )
        self.assertIn(
            'PINNED_TRANSFORMERS_VERSION="${PINNED_TRANSFORMERS_VERSION:-5.13.1}"',
            self.harness,
        )
        self.assertIn('PINNED_VLLM_VERSION}" = "0.25.1"', self.harness)
        self.assertIn("256-wide parameter for a 512-wide", self.harness)
        self.assertIn('"transformers==${PINNED_TRANSFORMERS_VERSION}"', self.harness)
        self.assertIn(
            'transformers.__version__ == os.environ["PINNED_TRANSFORMERS_VERSION"]',
            self.harness,
        )

    def test_download_once_and_shared_manifest(self):
        self.assertEqual(self.harness.count("snapshot_download("), 1)
        self.assertIn("Using complete local snapshot", self.harness)
        self.assertIn("Verified {len(shards)} safetensors shard(s)", self.harness)
        self.assertIn('PROMPT_TOKEN_IDS_JSON="${OUT_DIR}/long_prompt_token_ids.json"', self.harness)
        self.assertEqual(self.harness.count('--prompt-token-ids-json "${PROMPT_TOKEN_IDS_JSON}"'), 2)
        self.assertIn('"${COMMON_ARGS[@]}"', self.harness)

    def test_burst_gate_reuses_fresh_vm_download_and_skips_vllm(self):
        download = self.harness.index("== DOWNLOAD AND VERIFY ONCE FOR BOTH ENGINES ==")
        gate = self.harness.index(
            "== RUN LOADED-CHECKPOINT LONG DECODE GPU-FEEDBACK BURST GATE =="
        )
        vllm = self.harness.index("== INSTALL vLLM AFTER MEGAGEMM COMPLETES ==")
        self.assertLess(download, gate)
        self.assertLess(gate, vllm)
        self.assertIn('RUN_LONG_DECODE_BURST_GATE_ONLY="${RUN_LONG_DECODE_BURST_GATE_ONLY:-0}"', self.harness)
        self.assertIn("vLLM installation and full backend sweeps are disabled", self.harness)
        self.assertIn("exit 0", self.harness[gate:vllm])
        self.assertEqual(self.burst_gate_source.count("InferenceEngine("), 1)
        self.assertNotIn("snapshot_download", self.burst_gate_source)
        self.assertNotIn("import vllm", self.burst_gate_source)
        self.assertIn(
            '"APPLY_LONG_DECODE_GPU_FEEDBACK_BURST"',
            self.burst_gate_source,
        )
        self.assertIn('"minimum_savings_ms"', self.burst_gate_source)
        self.assertIn(
            'MEGAGEMM_DECODE_GRAPH_PERSISTENT_TOKEN_FEEDBACK"] = "0"',
            self.burst_gate_source,
        )

    def test_burst_gate_requires_nonpersistent_gpu_feedback_contract(self):
        gate = load_burst_gate()
        stats = {
            "enabled": True,
            "prefer_step": True,
            "shape_cache": True,
            "shared_shape_cache": False,
            "failures": 0,
            "replays": 61,
            "greedy_token_shape_graphs": 1,
            "token_burst_enabled": True,
            "token_burst_size": 8,
            "token_burst_steps": 63,
            "token_bursts": 8,
            "greedy_token_steps": 63,
            "batched_token_host_copies": 8,
            "persistent_token_feedback_enabled": False,
            "persistent_token_feedback_steps": 0,
            "token_feedback_copies": 55,
            "vectorized_input_updates": 8,
            "chain_input_updates_skipped": 0,
        }
        accepted = gate.graph_contract(
            stats,
            candidate=True,
            max_tokens=64,
            burst_size=8,
        )
        self.assertTrue(accepted["exact"])
        self.assertEqual(accepted["expected_bursts"], 8)
        self.assertEqual(accepted["expected_feedback_copies"], 55)

        persistent_chain = dict(stats)
        persistent_chain["persistent_token_feedback_enabled"] = True
        persistent_chain["persistent_token_feedback_steps"] = 63
        persistent_chain["token_feedback_copies"] = 0
        persistent_chain["vectorized_input_updates"] = 1
        persistent_chain["chain_input_updates_skipped"] = 120
        rejected = gate.graph_contract(
            persistent_chain,
            candidate=True,
            max_tokens=64,
            burst_size=8,
        )
        self.assertFalse(rejected["exact"])

    def test_burst_gate_promotes_only_a_material_stable_exact_win(self):
        gate = load_burst_gate()

        def case(median_ms):
            return {
                "decode_ms_median": median_ms,
                "stability_ratio": 1.01,
                "tokens_exact": True,
                "graph_contract_exact": True,
                "softcap_contract_exact": True,
                "deterministic_moe_contract_exact": True,
            }

        promoted = gate.decide_gate(
            case(1150.0),
            case(1000.0),
            minimum_speedup=1.02,
            minimum_savings_ms=20.0,
            maximum_stability_ratio=1.03,
        )
        self.assertTrue(promoted["apply_change"])
        self.assertEqual(
            promoted["decision"],
            "APPLY_LONG_DECODE_GPU_FEEDBACK_BURST",
        )

        marginal = gate.decide_gate(
            case(1150.0),
            case(1140.0),
            minimum_speedup=1.02,
            minimum_savings_ms=20.0,
            maximum_stability_ratio=1.03,
        )
        self.assertFalse(marginal["apply_change"])
        self.assertEqual(marginal["decision"], "KEEP_ONE_STEP")

    def test_short_context_microgates_are_not_in_long_harness(self):
        self.assertNotIn("SINGLE-KERNEL ATTN-to-MoE/ROUTER BRIDGE GATE", self.harness)
        self.assertNotIn("PREFILL KERNEL GATE", self.harness)
        self.assertNotIn("VLLM MOE PARITY", self.harness)
        self.assertIn("MEGAGEMM_PREFILL_CUDA_GRAPHS=0", self.harness)
        self.assertIn("MEGAGEMM_GEMMA4_FUSED_ATTN_MOE_ROUTER_SINGLE_KERNEL_DECODE=1", self.harness)

    def test_long_prompt_builder_is_exact_and_distinct(self):
        runner = load_runner()
        rows = runner.build_long_prompt_token_rows(
            base_ids=[1, 2, 3, 4],
            filler_ids=list(range(10, 42)),
            special_ids={1},
            target_tokens=32,
            required_rows=16,
        )
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(len(row) == 32 for row in rows))
        self.assertEqual(len({tuple(row) for row in rows}), 16)
        self.assertTrue(all(row[0] == 1 for row in rows))
        self.assertTrue(all(row[-3:] == [2, 3, 4] for row in rows))

    def test_runner_preserves_partial_results_and_reports_parity_separately(self):
        self.assertIn('"status": "partial"', self.runner_source)
        self.assertIn("_write_result(out_path, result)", self.runner_source)
        self.assertIn("PERFORMANCE_AND_TOKEN_PARITY_PASS", self.runner_source)
        self.assertIn("SHAPE_MATCHED_PERFORMANCE_ONLY", self.runner_source)
        self.assertIn("prefill_tok_s", self.runner_source)
        self.assertIn("decode_tok_s_median", self.runner_source)
        self.assertIn("input_plus_output_tok_s", self.runner_source)
        self.assertIn('"capture_run_excluded": True', self.runner_source)
        self.assertIn('"token_reference_policy": "first_measured_repeat"', self.runner_source)
        self.assertIn('case_result["status"] = "failed"', self.runner_source)

    def test_capture_tokens_are_diagnostic_but_measured_repeats_are_strict(self):
        runner = load_runner()
        warmup = [[10, 20, 30], [11, 21, 31]]
        first_measured = [[10, 99, 30], [11, 98, 31]]

        contract, reference = runner.measured_token_contract(
            warmup,
            None,
            first_measured,
        )
        self.assertFalse(contract["full_tokens_vs_excluded_warmup"]["exact"])
        self.assertTrue(contract["first_token_vs_excluded_warmup"]["exact"])
        self.assertTrue(contract["steady_state_vs_first_measured"]["exact"])
        runner._raise_measured_token_failure(contract, label="test")
        with self.assertRaisesRegex(RuntimeError, "stabilized warmup"):
            runner._raise_measured_token_failure(
                contract,
                label="test",
                require_full_warmup_match=True,
            )

        changed = [[10, 100, 30], [11, 98, 31]]
        contract, _ = runner.measured_token_contract(warmup, reference, changed)
        self.assertFalse(contract["steady_state_vs_first_measured"]["exact"])
        with self.assertRaisesRegex(RuntimeError, "measured repeats"):
            runner._raise_measured_token_failure(contract, label="test")

    def test_first_generated_token_must_still_match_warmup(self):
        runner = load_runner()
        contract, _ = runner.measured_token_contract(
            [[10, 20]],
            None,
            [[12, 20]],
        )
        with self.assertRaisesRegex(RuntimeError, "first generated token"):
            runner._raise_measured_token_failure(contract, label="test")

    def test_runtime_contract_requires_ordered_moe_reduction_in_every_layer(self):
        runner = load_runner()
        runtime = {
            "gemma4_a4b_segmented_prefill_layers": 30,
            "qwen3_moe_segmented_prefill_deterministic_reduce_layers": 30,
            "qwen3_moe_segmented_prefill_atomic_reduce_layers": 0,
            "gemma4_batch_moe_decode_policy": {"enabled_layers": 30},
            "gemma4_batch_moe_decode_deterministic_reduce_layers": 30,
        }
        self.assertTrue(runner.megagemm_deterministic_moe_contract(runtime)["exact"])

        runtime["qwen3_moe_segmented_prefill_atomic_reduce_layers"] = 1
        self.assertFalse(runner.megagemm_deterministic_moe_contract(runtime)["exact"])

        runtime["qwen3_moe_segmented_prefill_atomic_reduce_layers"] = 0
        runtime["qwen3_moe_segmented_prefill_deterministic_reduce_layers"] = 0
        runtime["gemma4_long_padded_bmm_prefill_enabled"] = True
        runtime["gemma4_long_padded_bmm_prefill_last_active_layers"] = 30
        runtime["gemma4_long_padded_bmm_prefill_disabled_layers"] = 0
        contract = runner.megagemm_deterministic_moe_contract(runtime)
        self.assertTrue(contract["exact"])
        self.assertEqual(contract["prefill_backend"], "padded_bmm_fp32")

        runtime["gemma4_long_padded_bmm_prefill_last_active_layers"] = 29
        self.assertFalse(runner.megagemm_deterministic_moe_contract(runtime)["exact"])

        runtime["gemma4_long_padded_bmm_prefill_last_active_layers"] = 23
        runtime["gemma4_long_padded_bmm_prefill_disabled_layers"] = 7
        runtime["gemma4_long_padded_bmm_prefill_first_failure"] = "fallback"
        runtime["gemma4_long_padded_bmm_prefill_failures"] = [
            {"layer_idx": layer_idx, "reason": "fallback"}
            for layer_idx in range(23, 30)
        ]
        runtime["qwen3_moe_segmented_prefill_deterministic_reduce_layers"] = 7
        contract = runner.megagemm_deterministic_moe_contract(runtime)
        self.assertTrue(contract["exact"])
        self.assertEqual(contract["prefill_covered_layers"], 30)
        self.assertEqual(
            contract["prefill_backend"],
            "padded_bmm_fp32_with_segmented_deterministic_fallback",
        )

        runtime["gemma4_long_padded_bmm_prefill_last_active_layers"] = 22
        self.assertFalse(runner.megagemm_deterministic_moe_contract(runtime)["exact"])
        runtime["gemma4_long_padded_bmm_prefill_last_active_layers"] = 23
        runtime["qwen3_moe_segmented_prefill_atomic_reduce_layers"] = 1
        self.assertFalse(runner.megagemm_deterministic_moe_contract(runtime)["exact"])
        runtime["qwen3_moe_segmented_prefill_atomic_reduce_layers"] = 0

        runtime["gemma4_long_padded_bmm_prefill_last_active_layers"] = 30
        runtime["gemma4_long_padded_bmm_prefill_disabled_layers"] = 0
        runtime["qwen3_moe_segmented_prefill_deterministic_reduce_layers"] = 0
        runtime["gemma4_batch_moe_decode_deterministic_reduce_layers"] = 29
        self.assertFalse(runner.megagemm_deterministic_moe_contract(runtime)["exact"])

    def test_dominant_expert_contract_covers_all_layers_or_guarded_fallback(self):
        runner = load_runner()
        profiles = [
            {
                "layer_idx": layer_idx,
                "heavy_expert": 0,
                "heavy_count": 15_648,
                "dominant_skew": 7.640625,
                "light_padding_ratio": 1.0137,
                "capacity_ratio": 1.0129,
            }
            for layer_idx in range(30)
        ]
        runtime = {
            "gemma4_a4b_segmented_prefill_layers": 30,
            "qwen3_moe_segmented_prefill_deterministic_reduce_layers": 0,
            "qwen3_moe_segmented_prefill_atomic_reduce_layers": 0,
            "gemma4_batch_moe_decode_policy": {"enabled_layers": 30},
            "gemma4_batch_moe_decode_deterministic_reduce_layers": 30,
            "gemma4_long_dominant_expert_prefill_enabled": True,
            "gemma4_long_dominant_expert_prefill_rows": 32_768,
            "gemma4_long_dominant_expert_prefill_down_output_dtype": "fp32",
            "gemma4_long_dominant_expert_prefill_route_pack": "atomic_split",
            "gemma4_long_dominant_expert_prefill_deterministic_reduce": True,
            "gemma4_long_dominant_expert_prefill_route_pack_block": 256,
            "gemma4_long_dominant_expert_prefill_activation_block": 512,
            "gemma4_long_dominant_expert_prefill_reduce_block_n": 256,
            "gemma4_long_dominant_expert_prefill_reduce_num_warps": 4,
            "gemma4_long_dominant_expert_prefill_align_m": 16,
            "gemma4_long_dominant_expert_prefill_minimum_skew": 7.5,
            "gemma4_long_dominant_expert_prefill_max_light_padding_ratio": 1.25,
            "gemma4_long_dominant_expert_prefill_hits": 90,
            "gemma4_long_dominant_expert_prefill_last_active_layers": 30,
            "gemma4_long_dominant_expert_prefill_guard_miss_layers": 0,
            "gemma4_long_dominant_expert_prefill_guard_rejections": [],
            "gemma4_long_dominant_expert_prefill_disabled_layers": 0,
            "gemma4_long_dominant_expert_prefill_first_failure": "",
            "gemma4_long_dominant_expert_prefill_failures": [],
            "gemma4_long_dominant_expert_prefill_profiles": profiles,
        }

        routed = runner.megagemm_long_routed_expert_prefill_contract(
            runtime,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(routed["exact"])
        self.assertEqual(routed["coverage_mode"], "dominant_expert_hybrid_fp32")
        deterministic = runner.megagemm_deterministic_moe_contract(runtime)
        self.assertTrue(deterministic["exact"])
        self.assertEqual(
            deterministic["prefill_backend"],
            "dominant_expert_hybrid_fp32",
        )

        guarded = dict(runtime)
        guarded["gemma4_long_dominant_expert_prefill_last_active_layers"] = 23
        guarded["gemma4_long_dominant_expert_prefill_guard_miss_layers"] = 7
        guarded["gemma4_long_dominant_expert_prefill_guard_rejections"] = [
            {"layer_idx": layer_idx, "reason": "Dominant-expert guard: fallback"}
            for layer_idx in range(23, 30)
        ]
        guarded["gemma4_long_dominant_expert_prefill_profiles"] = profiles[:23]
        guarded["qwen3_moe_segmented_prefill_deterministic_reduce_layers"] = 7
        guarded_contract = runner.megagemm_long_routed_expert_prefill_contract(
            guarded,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(guarded_contract["exact"])
        self.assertTrue(guarded_contract["hybrid_fallback_exact"])
        self.assertTrue(runner.megagemm_deterministic_moe_contract(guarded)["exact"])

        invalid = dict(runtime)
        invalid["gemma4_long_dominant_expert_prefill_profiles"] = [
            {**profiles[0], "dominant_skew": 7.4},
            *profiles[1:],
        ]
        self.assertFalse(
            runner.megagemm_long_routed_expert_prefill_contract(
                invalid,
                batch_size=16,
                context=2048,
            )["exact"]
        )

    def test_active_list_contract_requires_requested_path_in_every_decode_layer(self):
        runner = load_runner()
        runtime = {
            "gemma4_batch_moe_decode_policy": {"enabled_layers": 30},
            "qwen3_moe_expert_grouped_compact_active_list": True,
            "qwen3_moe_expert_grouped_compact_active_list_early_exit": True,
            "gemma4_batch_moe_decode_active_list_layers": 30,
            "gemma4_batch_moe_decode_active_list_early_exit_layers": 30,
        }
        environment = {
            "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST": "1",
            "MEGAGEMM_QWEN3_MOE_EXPERT_GROUPED_COMPACT_ACTIVE_LIST_EARLY_EXIT": "1",
        }
        with patch.dict("os.environ", environment, clear=False):
            contract = runner.megagemm_compact_active_list_contract(
                runtime,
                batch_size=16,
            )
            self.assertTrue(contract["exact"])
            runtime["gemma4_batch_moe_decode_active_list_early_exit_layers"] = 29
            rejected = runner.megagemm_compact_active_list_contract(
                runtime,
                batch_size=16,
            )
            self.assertFalse(rejected["exact"])
            runtime["gemma4_batch_moe_decode_active_list_layers"] = 29
            runtime["gemma4_batch_moe_decode_policy"]["enabled_layers"] = 29
            self.assertFalse(
                runner.megagemm_compact_active_list_contract(
                    runtime,
                    batch_size=16,
                )["exact"]
            )

    def test_long_sliding_prefill_contract_is_required_only_for_gated_shapes(self):
        runner = load_runner()
        runtime = {
            "gemma4_long_sliding_prefill_enabled": True,
            "gemma4_long_sliding_prefill_hits": 50,
        }
        active = runner.megagemm_long_sliding_prefill_contract(
            runtime,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(active["expected"])
        self.assertTrue(active["exact"])

        missing = runner.megagemm_long_sliding_prefill_contract(
            {},
            batch_size=16,
            context=2048,
        )
        self.assertTrue(missing["expected"])
        self.assertFalse(missing["exact"])

        ineligible = runner.megagemm_long_sliding_prefill_contract(
            {},
            batch_size=1,
            context=2048,
        )
        self.assertFalse(ineligible["expected"])
        self.assertTrue(ineligible["exact"])

    def test_long_full_prefill_contract_is_required_only_for_gated_shapes(self):
        runner = load_runner()
        runtime = {
            "gemma4_long_full_prefill_enabled": True,
            "gemma4_long_full_prefill_hits": 10,
        }
        active = runner.megagemm_long_full_prefill_contract(
            runtime,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(active["expected"])
        self.assertTrue(active["exact"])

        missing = runner.megagemm_long_full_prefill_contract(
            {},
            batch_size=16,
            context=2048,
        )
        self.assertTrue(missing["expected"])
        self.assertFalse(missing["exact"])

        ineligible = runner.megagemm_long_full_prefill_contract(
            {},
            batch_size=1,
            context=2048,
        )
        self.assertFalse(ineligible["expected"])
        self.assertTrue(ineligible["exact"])

    def test_long_attention_prepare_contract_requires_runtime_hits(self):
        runner = load_runner()
        runtime = {
            "gemma4_fused_attn_prepare_enabled": True,
            "gemma4_fused_attn_prepare_hits": 60,
            "gemma4_fused_attn_prepare_disabled_layers": 0,
            "gemma4_fused_attn_prepare_skip_reason": "",
        }
        active = runner.megagemm_long_attention_prepare_contract(
            runtime,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(active["expected"])
        self.assertTrue(active["exact"])

        missing = runner.megagemm_long_attention_prepare_contract(
            {},
            batch_size=16,
            context=2048,
        )
        self.assertTrue(missing["expected"])
        self.assertFalse(missing["exact"])

        disabled = dict(runtime)
        disabled["gemma4_fused_attn_prepare_disabled_layers"] = 1
        self.assertFalse(
            runner.megagemm_long_attention_prepare_contract(
                disabled,
                batch_size=16,
                context=2048,
            )["exact"]
        )

        ineligible = runner.megagemm_long_attention_prepare_contract(
            {},
            batch_size=1,
            context=2048,
        )
        self.assertFalse(ineligible["expected"])
        self.assertTrue(ineligible["exact"])

    def test_prefill_chunk_plan_contract_distinguishes_two_chunks_from_one(self):
        runner = load_runner()

        def plan(token_cap, chunk_tokens, chunk_requests):
            return {
                "strategy": "batched_tokens",
                "total_prompt_tokens": 32_768,
                "num_chunks": len(chunk_tokens),
                "max_requests": 16,
                "max_batched_tokens": token_cap,
                "deterministic_moe_token_cap": token_cap,
                "chunk_prompt_tokens": chunk_tokens,
                "chunk_request_counts": chunk_requests,
            }

        baseline = runner.megagemm_prefill_chunk_plan_contract(
            plan(16_384, [16_384, 16_384], [8, 8]),
            batch_size=16,
            context=2048,
            token_cap=16_384,
        )
        candidate = runner.megagemm_prefill_chunk_plan_contract(
            plan(32_768, [32_768], [16]),
            batch_size=16,
            context=2048,
            token_cap=32_768,
        )
        wrong = runner.megagemm_prefill_chunk_plan_contract(
            plan(32_768, [16_384, 16_384], [8, 8]),
            batch_size=16,
            context=2048,
            token_cap=32_768,
        )
        self.assertTrue(baseline["exact"])
        self.assertEqual(baseline["expected"]["num_chunks"], 2)
        self.assertTrue(candidate["exact"])
        self.assertEqual(candidate["expected"]["num_chunks"], 1)
        self.assertFalse(wrong["exact"])

    def test_prefill_chunk_gate_promotes_only_an_exact_stable_material_win(self):
        runner = load_runner()

        def group(median_ms, token, *, stable=True, error=None):
            return {
                "median_ms": median_ms,
                "token_ids": [[token, token + 1] for _ in range(16)],
                "stable": stable,
                "error": error,
            }

        promoted = runner.evaluate_megagemm_prefill_chunk_gate(
            group(100.0, 7),
            group(80.0, 7),
            group(101.0, 7),
        )
        self.assertTrue(promoted["apply_change"])
        self.assertEqual(promoted["decision"], "APPLY_32768")
        self.assertEqual(promoted["selected_token_cap"], 32_768)

        token_mismatch = runner.evaluate_megagemm_prefill_chunk_gate(
            group(100.0, 7),
            group(80.0, 9),
            group(101.0, 7),
        )
        self.assertFalse(token_mismatch["apply_change"])
        self.assertEqual(token_mismatch["reason"], "candidate_tokens_changed")

        measured_v27 = runner.evaluate_megagemm_prefill_chunk_gate(
            group(1840.20838, 7),
            group(1811.881338, 7),
            group(1838.048652, 7),
        )
        self.assertTrue(measured_v27["apply_change"])
        self.assertEqual(measured_v27["decision"], "APPLY_32768")
        self.assertGreater(measured_v27["speedup"], 1.01)

        marginal = runner.evaluate_megagemm_prefill_chunk_gate(
            group(100.0, 7),
            group(99.5, 7),
            group(101.0, 7),
        )
        self.assertFalse(marginal["apply_change"])
        self.assertEqual(marginal["reason"], "candidate_speedup_below_threshold")

    def test_prefill_chunk_candidate_state_is_restored_before_baseline_recheck(self):
        runner = load_runner()
        attention = type(
            "Attention",
            (),
            {
                "_gemma4_fused_attn_prepare_disabled": False,
                "_gemma4_fused_attn_prepare_skip_reason": "",
            },
        )()
        experts = type(
            "Experts",
            (),
            {
                "_segmented_prefill_disabled": False,
                "_segmented_prefill_fail_reason": "",
                "_segmented_prefill_workspace": {"candidate": object()},
            },
        )()
        gate = type("Gate", (), {"_prefill_topk_workspaces": {32768: {}}})()
        layer = type(
            "Layer",
            (),
            {
                "self_attn": attention,
                "mlp": type("MLP", (), {"experts": experts, "gate": gate})(),
            },
        )()
        engine = type(
            "Engine",
            (),
            {"model": type("Model", (), {"layers": [layer]})()},
        )()

        state = runner._snapshot_megagemm_prefill_chunk_gate_state(engine)
        attention._gemma4_fused_attn_prepare_disabled = True
        attention._gemma4_fused_attn_prepare_skip_reason = "candidate failed"
        experts._segmented_prefill_disabled = True
        experts._segmented_prefill_fail_reason = "candidate failed"
        runner._restore_megagemm_prefill_chunk_gate_state(engine, state)

        self.assertFalse(attention._gemma4_fused_attn_prepare_disabled)
        self.assertEqual(attention._gemma4_fused_attn_prepare_skip_reason, "")
        self.assertFalse(experts._segmented_prefill_disabled)
        self.assertEqual(experts._segmented_prefill_fail_reason, "")
        self.assertEqual(experts._segmented_prefill_workspace, {})
        self.assertEqual(gate._prefill_topk_workspaces, {})

    def test_long_routed_expert_contract_requires_promoted_exact_kernel(self):
        runner = load_runner()
        runtime = {
            "gemma4_a4b_segmented_prefill_layers": 30,
            "gemma4_long_padded_bmm_prefill_enabled": True,
            "gemma4_long_padded_bmm_prefill_rows": 16_384,
            "gemma4_long_padded_bmm_prefill_down_output_dtype": "fp32",
            "gemma4_long_padded_bmm_prefill_route_pack": "argsort",
            "gemma4_long_padded_bmm_prefill_route_pack_block": 256,
            "gemma4_long_padded_bmm_prefill_max_padding_ratio": 2.0,
            "gemma4_long_padded_bmm_prefill_fused_activation": True,
            "gemma4_long_padded_bmm_prefill_activation_block": 512,
            "gemma4_long_padded_bmm_prefill_reduce_block_n": 256,
            "gemma4_long_padded_bmm_prefill_reduce_num_warps": 4,
            "gemma4_long_padded_bmm_prefill_align_m": 16,
            "gemma4_long_padded_bmm_prefill_hits": 60,
            "gemma4_long_padded_bmm_prefill_last_active_layers": 30,
            "gemma4_long_padded_bmm_prefill_disabled_layers": 0,
            "gemma4_long_padded_bmm_prefill_first_failure": "",
            "gemma4_long_padded_bmm_prefill_failures": [],
            "qwen3_moe_segmented_prefill_deterministic_reduce_layers": 0,
            "qwen3_moe_segmented_prefill_atomic_reduce_layers": 0,
        }
        active = runner.megagemm_long_routed_expert_prefill_contract(
            runtime,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(active["expected"])
        self.assertTrue(active["exact"])

        hybrid = dict(runtime)
        hybrid["gemma4_long_padded_bmm_prefill_last_active_layers"] = 23
        hybrid["gemma4_long_padded_bmm_prefill_disabled_layers"] = 7
        hybrid["gemma4_long_padded_bmm_prefill_first_failure"] = "fallback"
        hybrid["gemma4_long_padded_bmm_prefill_failures"] = [
            {"layer_idx": layer_idx, "reason": "fallback"}
            for layer_idx in range(23, 30)
        ]
        hybrid["qwen3_moe_segmented_prefill_deterministic_reduce_layers"] = 7
        hybrid_contract = runner.megagemm_long_routed_expert_prefill_contract(
            hybrid,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(hybrid_contract["exact"])
        self.assertTrue(hybrid_contract["deterministic_coverage_exact"])
        self.assertEqual(
            hybrid_contract["coverage_mode"],
            "padded_bmm_fp32_with_segmented_deterministic_fallback",
        )

        segmented = dict(runtime)
        segmented["gemma4_long_padded_bmm_prefill_hits"] = 0
        segmented["gemma4_long_padded_bmm_prefill_last_active_layers"] = 0
        segmented["gemma4_long_padded_bmm_prefill_disabled_layers"] = 30
        segmented["gemma4_long_padded_bmm_prefill_first_failure"] = "fallback"
        segmented["gemma4_long_padded_bmm_prefill_failures"] = [
            {"layer_idx": layer_idx, "reason": "fallback"}
            for layer_idx in range(30)
        ]
        segmented["qwen3_moe_segmented_prefill_deterministic_reduce_layers"] = 30
        segmented_contract = runner.megagemm_long_routed_expert_prefill_contract(
            segmented,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(segmented_contract["exact"])
        self.assertTrue(segmented_contract["segmented_only_exact"])
        self.assertEqual(
            segmented_contract["coverage_mode"],
            "segmented_deterministic_fallback",
        )

        skew_gated = {
            "gemma4_a4b_segmented_prefill_layers": 30,
            "gemma4_a4b_segmented_prefill_config": {
                "long_rows": 16_384,
                "long": {
                    "block_m": 128,
                    "block_n": 256,
                    "block_k": 64,
                    "fused_gate_block_n": 128,
                    "num_warps": 8,
                    "num_stages": 3,
                    "compact_route_pack": False,
                },
            },
            "gemma4_long_padded_bmm_prefill_enabled": False,
            "gemma4_long_padded_bmm_prefill_last_active_layers": 0,
            "gemma4_long_padded_bmm_prefill_disabled_layers": 0,
            "qwen3_moe_segmented_prefill_deterministic_reduce_layers": 30,
            "qwen3_moe_segmented_prefill_atomic_reduce_layers": 0,
        }
        skew_gated_contract = runner.megagemm_long_routed_expert_prefill_contract(
            skew_gated,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(skew_gated_contract["exact"])
        self.assertTrue(skew_gated_contract["segmented_long_config_exact"])
        self.assertEqual(
            skew_gated_contract["coverage_mode"],
            "segmented_deterministic_skew_gated",
        )

        async_requested = dict(skew_gated)
        async_requested["gemma4_a4b_segmented_prefill_config"] = {
            "long_rows": 16_384,
            "long": {
                **skew_gated["gemma4_a4b_segmented_prefill_config"]["long"],
                "async_tiles_max_assignments": 262_144,
            },
        }
        missing_async_hits = runner.megagemm_long_routed_expert_prefill_contract(
            async_requested,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(missing_async_hits["async_tile_requested"])
        self.assertFalse(missing_async_hits["async_tile_contract_exact"])
        self.assertFalse(missing_async_hits["exact"])

        async_requested["qwen3_moe_segmented_prefill_async_tile_hits"] = 30
        active_async = runner.megagemm_long_routed_expert_prefill_contract(
            async_requested,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(active_async["async_tile_contract_exact"])
        self.assertTrue(active_async["exact"])

        sorted_requested = dict(async_requested)
        sorted_requested["gemma4_a4b_segmented_prefill_config"] = {
            "long_rows": 16_384,
            "long": {
                **async_requested["gemma4_a4b_segmented_prefill_config"]["long"],
                "sorted_partial": True,
            },
        }
        missing_sorted_hits = runner.megagemm_long_routed_expert_prefill_contract(
            sorted_requested,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(missing_sorted_hits["sorted_partial_requested"])
        self.assertFalse(missing_sorted_hits["sorted_partial_contract_exact"])
        self.assertFalse(missing_sorted_hits["exact"])

        sorted_requested["qwen3_moe_segmented_prefill_sorted_partial_hits"] = 30
        sorted_requested["qwen3_moe_segmented_prefill_sorted_partial_layers"] = 30
        active_sorted = runner.megagemm_long_routed_expert_prefill_contract(
            sorted_requested,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(active_sorted["sorted_partial_contract_exact"])
        self.assertTrue(active_sorted["exact"])

        missing_layer = dict(hybrid)
        missing_layer["gemma4_long_padded_bmm_prefill_last_active_layers"] = 22
        self.assertFalse(
            runner.megagemm_long_routed_expert_prefill_contract(
                missing_layer,
                batch_size=16,
                context=2048,
            )["exact"]
        )

        atomic_fallback = dict(hybrid)
        atomic_fallback["qwen3_moe_segmented_prefill_atomic_reduce_layers"] = 1
        self.assertFalse(
            runner.megagemm_long_routed_expert_prefill_contract(
                atomic_fallback,
                batch_size=16,
                context=2048,
            )["exact"]
        )

        wrong_pack = dict(runtime)
        wrong_pack["gemma4_long_padded_bmm_prefill_route_pack"] = "atomic"
        self.assertFalse(
            runner.megagemm_long_routed_expert_prefill_contract(
                wrong_pack,
                batch_size=16,
                context=2048,
            )["exact"]
        )

        wrong_tile = dict(runtime)
        wrong_tile["gemma4_long_padded_bmm_prefill_down_output_dtype"] = "bf16"
        rejected = runner.megagemm_long_routed_expert_prefill_contract(
            wrong_tile,
            batch_size=16,
            context=2048,
        )
        self.assertTrue(rejected["expected"])
        self.assertFalse(rejected["exact"])

        ineligible = runner.megagemm_long_routed_expert_prefill_contract(
            {},
            batch_size=1,
            context=2048,
        )
        self.assertFalse(ineligible["expected"])
        self.assertTrue(ineligible["exact"])

    def test_b16_c2048_normalizer_accepts_exact_hybrid_prefill_coverage(self):
        runner = load_runner()
        failures = [
            {"layer_idx": layer_idx, "reason": "padded capacity fallback"}
            for layer_idx in range(23, 30)
        ]
        runtime = {
            "gemma4_a4b_segmented_prefill_layers": 30,
            "qwen3_moe_segmented_prefill_deterministic_reduce_layers": 7,
            "qwen3_moe_segmented_prefill_atomic_reduce_layers": 0,
            "gemma4_batch_moe_decode_policy": {"enabled_layers": 30},
            "gemma4_batch_moe_decode_deterministic_reduce_layers": 30,
            "gemma4_long_sliding_prefill_enabled": True,
            "gemma4_long_sliding_prefill_hits": 50,
            "gemma4_long_full_prefill_enabled": True,
            "gemma4_long_full_prefill_hits": 10,
            "gemma4_fused_attn_prepare_enabled": True,
            "gemma4_fused_attn_prepare_hits": 60,
            "gemma4_fused_attn_prepare_disabled_layers": 0,
            "gemma4_fused_attn_prepare_skip_reason": "",
            "paged_decode_runtime": {
                "grouped_segmented_disabled": False,
                "grouped_segmented_failure": "",
                "grouped_segmented_selected_segments": {
                    "sliding_h256_gqa2": 32,
                    "full_h512_gqa8": 8,
                },
                "grouped_segmented_selected_tile_sizes": {
                    "sliding_h256_gqa2": 64,
                    "full_h512_gqa8": 16,
                },
            },
            "gemma4_long_padded_bmm_prefill_enabled": True,
            "gemma4_long_padded_bmm_prefill_rows": 16_384,
            "gemma4_long_padded_bmm_prefill_down_output_dtype": "fp32",
            "gemma4_long_padded_bmm_prefill_route_pack": "argsort",
            "gemma4_long_padded_bmm_prefill_route_pack_block": 256,
            "gemma4_long_padded_bmm_prefill_max_padding_ratio": 2.0,
            "gemma4_long_padded_bmm_prefill_fused_activation": True,
            "gemma4_long_padded_bmm_prefill_activation_block": 512,
            "gemma4_long_padded_bmm_prefill_reduce_block_n": 256,
            "gemma4_long_padded_bmm_prefill_reduce_num_warps": 4,
            "gemma4_long_padded_bmm_prefill_align_m": 16,
            "gemma4_long_padded_bmm_prefill_hits": 46,
            "gemma4_long_padded_bmm_prefill_last_active_layers": 23,
            "gemma4_long_padded_bmm_prefill_disabled_layers": 7,
            "gemma4_long_padded_bmm_prefill_first_failure": failures[0]["reason"],
            "gemma4_long_padded_bmm_prefill_failures": failures,
        }
        row = {
            "scheduler_prefill_ms": 1000.0,
            "scheduler_decode_ms": 200.0,
            "total_ms": 1200.0,
            "decode_cuda_graphs": {
                "enabled": True,
                "token_burst_enabled": False,
                "shared_shape_cache": False,
                "failures": 0,
                "replays": 1,
                "physical_rebinds": 0,
            },
            "decode_runtime": runtime,
        }

        normalized, returned_runtime = runner.normalize_megagemm_row(
            row,
            batch_size=16,
            context=2048,
            max_tokens=64,
        )

        self.assertIs(returned_runtime["gemma4_long_padded_bmm_prefill_failures"], failures)
        self.assertTrue(normalized["deterministic_moe_contract"]["exact"])
        self.assertTrue(normalized["long_routed_expert_prefill_contract"]["exact"])
        self.assertEqual(
            normalized["long_routed_expert_prefill_contract"]["coverage_mode"],
            "padded_bmm_fp32_with_segmented_deterministic_fallback",
        )

    @staticmethod
    def _fake_megagemm_row(tokens):
        return {
            "scheduler_prefill_ms": 2.0,
            "scheduler_decode_ms": 1.0,
            "total_ms": 3.0,
            "token_ids": tokens,
            "decode_cuda_graphs": {
                "enabled": True,
                "token_burst_enabled": False,
                "shared_shape_cache": False,
                "failures": 0,
                "replays": 1,
                "physical_rebinds": 0,
            },
            "decode_runtime": {
                "gemma4_a4b_segmented_prefill_layers": 30,
                "qwen3_moe_segmented_prefill_deterministic_reduce_layers": 30,
                "qwen3_moe_segmented_prefill_atomic_reduce_layers": 0,
                "gemma4_batch_moe_decode_policy": {"enabled_layers": 30},
                "gemma4_batch_moe_decode_deterministic_reduce_layers": 30,
            },
        }

    @staticmethod
    def _fake_args(out_json):
        return Namespace(
            backend="megagemm",
            model="/content/model",
            dtype="bf16",
            contexts=[1024],
            batch_sizes=[1],
            max_seq_len=1088,
            max_tokens=3,
            max_num_batched_tokens=1024,
            warmups=1,
            megagemm_min_warmups=3,
            megagemm_max_warmups=8,
            megagemm_required_stable_warmup_pairs=2,
            megagemm_warmup_max_last_pair_ratio=1.10,
            repeats=3,
            resume=False,
            out_json=str(out_json),
        )

    def test_megagemm_sweep_stabilizes_before_measuring(self):
        runner = load_runner()
        warmups = [
            self._fake_megagemm_row([[10, 20, 496]])
            for _ in range(3)
        ]
        measured = [
            self._fake_megagemm_row([[10, 20, 496]])
            for _ in range(3)
        ]
        prompts = {1024: [[7] * 1024]}
        manifest = {
            "generator": "test",
            "cases": {"1024": {"contract": runner.prompt_token_contract(prompts[1024])}},
        }

        with tempfile.TemporaryDirectory() as raw:
            out_path = Path(raw) / "megagemm.json"
            args = self._fake_args(out_path)
            with (
                patch("megagemm.engine.InferenceEngine", return_value=object()),
                patch.object(
                    runner,
                    "run_megagemm_request",
                    side_effect=[*warmups, *measured],
                ),
                patch.object(runner, "gpu_snapshot", return_value={"name": "test GPU"}),
            ):
                result = runner.run_megagemm_sweep(args, prompts, manifest)

            case = result["cases"]["b1_c1024"]
            self.assertEqual(result["status"], "complete")
            self.assertEqual(case["status"], "complete")
            self.assertEqual(len(case["warmups"]), 3)
            self.assertTrue(case["warmup_stability"]["stable"])
            self.assertEqual(
                case["warmup_stability"]["consecutive_stable_pairs"],
                2,
            )
            self.assertTrue(case["warmup_to_first_measured"]["exact"])
            self.assertTrue(
                all(
                    row["token_contract"]["steady_state_vs_first_measured"]["exact"]
                    for row in case["samples"]
                )
            )

    def test_measured_instability_is_persisted_before_failure(self):
        runner = load_runner()
        rows = [
            self._fake_megagemm_row([[10, 20, 496]]),
            self._fake_megagemm_row([[10, 20, 496]]),
            self._fake_megagemm_row([[10, 20, 496]]),
            self._fake_megagemm_row([[10, 20, 496]]),
            self._fake_megagemm_row([[10, 20, 497]]),
        ]
        prompts = {1024: [[7] * 1024]}
        manifest = {
            "generator": "test",
            "cases": {"1024": {"contract": runner.prompt_token_contract(prompts[1024])}},
        }

        with tempfile.TemporaryDirectory() as raw:
            out_path = Path(raw) / "megagemm.json"
            args = self._fake_args(out_path)
            with (
                patch("megagemm.engine.InferenceEngine", return_value=object()),
                patch.object(runner, "run_megagemm_request", side_effect=rows),
                patch.object(runner, "gpu_snapshot", return_value={"name": "test GPU"}),
            ):
                with self.assertRaisesRegex(RuntimeError, "stabilized warmup"):
                    runner.run_megagemm_sweep(args, prompts, manifest)

            persisted = json.loads(out_path.read_text(encoding="utf-8"))
            case = persisted["cases"]["b1_c1024"]
            self.assertEqual(persisted["status"], "partial")
            self.assertEqual(case["status"], "failed")
            self.assertEqual(len(case["samples"]), 2)
            self.assertEqual(case["token_stability_failure"]["repeat"], 2)

    def test_adaptive_warmup_waits_for_tokens_and_runtime_topology(self):
        runner = load_runner()

        def row(tokens, disabled_layers):
            result = self._fake_megagemm_row(tokens)
            result["decode_runtime"][
                "gemma4_grouped_mm_prefill_disabled_layers"
            ] = disabled_layers
            return result

        rows = [
            row([[10, 20, 100]], 0),
            row([[10, 20, 496]], 1),
            row([[10, 20, 496]], 1),
            row([[10, 20, 496]], 1),
            *[row([[10, 20, 496]], 1) for _ in range(3)],
        ]
        prompts = {1024: [[7] * 1024]}
        manifest = {
            "generator": "test",
            "cases": {
                "1024": {"contract": runner.prompt_token_contract(prompts[1024])}
            },
        }

        with tempfile.TemporaryDirectory() as raw:
            out_path = Path(raw) / "megagemm.json"
            args = self._fake_args(out_path)
            with (
                patch("megagemm.engine.InferenceEngine", return_value=object()),
                patch.object(
                    runner,
                    "run_megagemm_request",
                    side_effect=rows,
                ),
                patch.object(runner, "gpu_snapshot", return_value={"name": "test GPU"}),
            ):
                result = runner.run_megagemm_sweep(args, prompts, manifest)

        case = result["cases"]["b1_c1024"]
        self.assertEqual(len(case["warmups"]), 4)
        self.assertTrue(case["warmup_stability"]["stable"])
        self.assertEqual(
            case["stabilized_runtime_topology"]["fields"][
                "gemma4_grouped_mm_prefill_disabled_layers"
            ],
            1,
        )
        self.assertTrue(
            all(row["runtime_topology_contract"]["exact"] for row in case["samples"])
        )

    def test_long_context_rejects_cross_request_decode_graph_cache(self):
        runner = load_runner()
        row = self._fake_megagemm_row([[10, 20, 496]])
        row["decode_cuda_graphs"]["shared_shape_cache"] = True

        with self.assertRaisesRegex(RuntimeError, "decode graph-step contract failed"):
            runner.normalize_megagemm_row(
                row,
                batch_size=1,
                context=1024,
                max_tokens=3,
            )

    def test_long_context_accepts_owned_scheduler_replay_without_recapture(self):
        runner = load_runner()
        row = self._fake_megagemm_row([[10, 20, 496]])
        row["decode_cuda_graphs"].update(
            {
                "request_scheduler_reuse_enabled": True,
                "request_scheduler_reused": True,
                "request_scheduler_reuse_count": 1,
                "captures": 0,
                "warmups": 0,
            }
        )

        normalized, _ = runner.normalize_megagemm_row(
            row,
            batch_size=1,
            context=1024,
            max_tokens=3,
        )

        self.assertEqual(
            normalized["decode_graph_scope"],
            "engine_persistent_scheduler_replay",
        )
        self.assertTrue(normalized["request_scheduler_reuse_contract"]["exact"])

    def test_long_context_rejects_reused_scheduler_that_recaptures(self):
        runner = load_runner()
        row = self._fake_megagemm_row([[10, 20, 496]])
        row["decode_cuda_graphs"].update(
            {
                "request_scheduler_reuse_enabled": True,
                "request_scheduler_reused": True,
                "request_scheduler_reuse_count": 1,
                "captures": 1,
                "warmups": 0,
            }
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "persistent request-Scheduler contract failed",
        ):
            runner.normalize_megagemm_row(
                row,
                batch_size=1,
                context=1024,
                max_tokens=3,
            )

    def test_long_context_rejects_graph_token_burst(self):
        runner = load_runner()
        row = self._fake_megagemm_row([[10, 20, 496]])
        row["decode_cuda_graphs"]["token_burst_enabled"] = True

        with self.assertRaisesRegex(RuntimeError, "decode graph-step contract failed"):
            runner.normalize_megagemm_row(
                row,
                batch_size=1,
                context=1024,
                max_tokens=3,
            )

    def test_paid_long_decode_burst_contract_is_exact_and_nonpersistent(self):
        runner = load_runner()
        row = self._fake_megagemm_row([[10] * 64])
        row["decode_cuda_graphs"].update(
            {
                "prefer_step": True,
                "shape_cache": True,
                "greedy_token_shape_graphs": 1,
                "token_burst_enabled": True,
                "token_burst_size": 8,
                "token_burst_steps": 63,
                "token_bursts": 8,
                "greedy_token_steps": 63,
                "batched_token_host_copies": 8,
                "persistent_token_feedback_enabled": False,
                "persistent_token_feedback_steps": 0,
                "token_feedback_copies": 55,
                "vectorized_input_updates": 8,
                "chain_input_updates_skipped": 0,
            }
        )
        row["decode_runtime"].update(
            {
                "gemma4_batch_cublas_lm_head_enabled": True,
                "gemma4_batch_cublas_lm_head_hits": 63,
                "gemma4_batch_fused_softcap_argmax_enabled": True,
                "gemma4_batch_fused_softcap_argmax_available": True,
                "gemma4_batch_fused_softcap_argmax_hits": 63,
                "gemma4_batch_fused_softcap_argmax_disabled": False,
                "gemma4_batch_fused_softcap_argmax_error": "",
            }
        )
        paid_runtime = dict(row["decode_runtime"])

        normalized, _ = runner.normalize_megagemm_row(
            row,
            batch_size=1,
            context=1024,
            max_tokens=64,
            decode_mode=runner.MEGAGEMM_DECODE_MODE_GRAPH_BURST,
        )

        self.assertEqual(
            normalized["decode_graph_scope"],
            "request_local_burst8_gpu_feedback",
        )
        self.assertTrue(normalized["long_decode_burst_contract"]["exact"])
        self.assertEqual(
            normalized["long_decode_burst_contract"]["expected_feedback_copies"],
            55,
        )

        bad_row = self._fake_megagemm_row([[10] * 64])
        bad_row["decode_cuda_graphs"] = {
            **row["decode_cuda_graphs"],
            "persistent_token_feedback_enabled": True,
            "persistent_token_feedback_steps": 63,
            "token_feedback_copies": 0,
        }
        bad_row["decode_runtime"] = paid_runtime
        with self.assertRaisesRegex(RuntimeError, "decode graph-burst contract failed"):
            runner.normalize_megagemm_row(
                bad_row,
                batch_size=1,
                context=1024,
                max_tokens=64,
                decode_mode=runner.MEGAGEMM_DECODE_MODE_GRAPH_BURST,
            )

    def test_same_burst_mode_preserves_softcap_graph_capture_evidence(self):
        runner = load_runner()
        import megagemm.models.llama as llama_model

        previous_softcap_mode = llama_model._GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX
        model = Namespace(
            _gemma4_batch_cublas_lm_head_hits=2,
            _gemma4_batch_fused_softcap_argmax_hits=2,
            _gemma4_batch_fused_softcap_argmax_disable=False,
            _gemma4_batch_fused_softcap_argmax_error="",
        )
        try:
            llama_model._GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX = True
            runner.configure_megagemm_decode_mode(
                runner.MEGAGEMM_DECODE_MODE_GRAPH_BURST,
                model,
            )
            self.assertEqual(model._gemma4_batch_cublas_lm_head_hits, 2)
            self.assertEqual(model._gemma4_batch_fused_softcap_argmax_hits, 2)

            runner.configure_megagemm_decode_mode(
                runner.MEGAGEMM_DECODE_MODE_GRAPH_STEP,
                model,
            )
            self.assertEqual(model._gemma4_batch_cublas_lm_head_hits, 0)
            self.assertEqual(model._gemma4_batch_fused_softcap_argmax_hits, 0)
        finally:
            llama_model._GEMMA4_BATCH_FUSED_SOFTCAP_ARGMAX = previous_softcap_mode

    def test_eager_decode_contract_accepts_no_graph(self):
        runner = load_runner()
        row = self._fake_megagemm_row([[10, 20, 496]])
        row["decode_cuda_graphs"] = {}

        normalized, _ = runner.normalize_megagemm_row(
            row,
            batch_size=1,
            context=1024,
            max_tokens=3,
            decode_mode=runner.MEGAGEMM_DECODE_MODE_EAGER,
        )

        self.assertEqual(normalized["decode_execution_mode"], "eager")
        self.assertEqual(normalized["decode_graph_scope"], "disabled")

    def test_b16_c2048_can_fallback_after_early_graph_token_rejection(self):
        runner = load_runner()
        modes = runner.megagemm_decode_mode_candidates(
            batch_size=16,
            context=2048,
        )
        stability = {
            "completed_warmups": 3,
            "minimum_warmups": 3,
            "reason": "last_pair_tokens_changed",
            "last_pair_runtime_topology": {"exact": True},
        }

        self.assertEqual(
            modes,
            [
                runner.MEGAGEMM_DECODE_MODE_GRAPH_STEP,
                runner.MEGAGEMM_DECODE_MODE_EAGER,
            ],
        )
        self.assertTrue(runner.should_reject_graph_step_early(stability))
        self.assertEqual(
            runner.megagemm_decode_mode_candidates(batch_size=1, context=2048),
            [runner.MEGAGEMM_DECODE_MODE_GRAPH_STEP],
        )
        self.assertEqual(
            runner.megagemm_decode_mode_candidates(
                batch_size=16,
                context=2048,
                max_tokens=64,
                long_decode_burst_proven=True,
            ),
            [
                runner.MEGAGEMM_DECODE_MODE_GRAPH_BURST,
                runner.MEGAGEMM_DECODE_MODE_GRAPH_STEP,
                runner.MEGAGEMM_DECODE_MODE_EAGER,
            ],
        )
        self.assertNotIn(
            runner.MEGAGEMM_DECODE_MODE_GRAPH_BURST,
            runner.megagemm_decode_mode_candidates(
                batch_size=16,
                context=2048,
                max_tokens=32,
                long_decode_burst_proven=True,
            ),
        )

    def test_b16_c2048_candidates_gate_dominant_hybrid_then_fall_back(self):
        runner = load_runner()
        candidates = runner.megagemm_execution_candidates(
            batch_size=16,
            context=2048,
            determinism_auto_fallback=True,
        )

        self.assertEqual(
            candidates,
            [
                {
                    "decode_mode": runner.MEGAGEMM_DECODE_MODE_GRAPH_STEP,
                    "prefill_profile": runner.MEGAGEMM_PREFILL_PROFILE_SEGMENTED,
                },
                {
                    "decode_mode": runner.MEGAGEMM_DECODE_MODE_EAGER,
                    "prefill_profile": runner.MEGAGEMM_PREFILL_PROFILE_SEGMENTED,
                },
            ],
        )

        with patch.dict(
            os.environ,
            {"MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_PREFILL": "1"},
        ):
            promoted = runner.megagemm_execution_candidates(
                batch_size=16,
                context=2048,
                max_tokens=64,
                long_decode_burst_proven=True,
                determinism_auto_fallback=True,
            )
        self.assertEqual(
            promoted,
            [
                {
                    "decode_mode": runner.MEGAGEMM_DECODE_MODE_GRAPH_BURST,
                    "prefill_profile": runner.MEGAGEMM_PREFILL_PROFILE_HYBRID,
                },
                {
                    "decode_mode": runner.MEGAGEMM_DECODE_MODE_GRAPH_BURST,
                    "prefill_profile": runner.MEGAGEMM_PREFILL_PROFILE_SEGMENTED,
                },
                {
                    "decode_mode": runner.MEGAGEMM_DECODE_MODE_GRAPH_STEP,
                    "prefill_profile": runner.MEGAGEMM_PREFILL_PROFILE_SEGMENTED,
                },
                {
                    "decode_mode": runner.MEGAGEMM_DECODE_MODE_EAGER,
                    "prefill_profile": runner.MEGAGEMM_PREFILL_PROFILE_SEGMENTED,
                },
            ],
        )

    def test_segmented_fallback_disables_every_expert_workspace(self):
        runner = load_runner()

        class Experts:
            def __init__(self):
                self._gemma4_long_dominant_expert_prefill_disabled = False
                self._gemma4_long_dominant_expert_prefill_fail_reason = ""
                self._gemma4_long_dominant_expert_prefill_last_active = True
                self._gemma4_long_dominant_expert_prefill_last_guard_reason = "stale"
                self._gemma4_long_dominant_expert_prefill_hits = 3
                self._gemma4_long_dominant_expert_prefill_assignments = 128
                self._gemma4_long_dominant_expert_prefill_guard_misses = 2
                self._gemma4_long_dominant_expert_prefill_workspace = {"buffer": 1}
                self._gemma4_long_padded_bmm_prefill_disabled = False
                self._gemma4_long_padded_bmm_prefill_fail_reason = ""
                self._gemma4_long_padded_bmm_prefill_last_active = True
                self._gemma4_long_padded_bmm_prefill_hits = 4
                self._gemma4_long_padded_bmm_prefill_assignments = 256
                self._gemma4_long_padded_bmm_prefill_workspace = {"buffer": 1}
                self._segmented_prefill_workspace = {"buffer": 1}

        class MLP:
            def __init__(self):
                self.experts = Experts()

        class Layer:
            def __init__(self):
                self.mlp = MLP()

        class Model:
            def __init__(self):
                self.layers = [Layer(), Layer()]

        model = Model()
        status = runner.force_segmented_long_prefill(model, "test fallback")

        self.assertEqual(status["disabled_layers"], 2)
        for layer in model.layers:
            experts = layer.mlp.experts
            self.assertTrue(
                experts._gemma4_long_dominant_expert_prefill_disabled
            )
            self.assertEqual(
                experts._gemma4_long_dominant_expert_prefill_fail_reason,
                "test fallback",
            )
            self.assertTrue(experts._gemma4_long_padded_bmm_prefill_disabled)
            self.assertEqual(
                experts._gemma4_long_padded_bmm_prefill_fail_reason,
                "test fallback",
            )
            self.assertFalse(experts._gemma4_long_padded_bmm_prefill_last_active)
            self.assertEqual(
                experts._gemma4_long_dominant_expert_prefill_workspace,
                {},
            )
            self.assertEqual(experts._gemma4_long_padded_bmm_prefill_workspace, {})
            self.assertEqual(experts._segmented_prefill_workspace, {})

        restored = runner.enable_guarded_padded_long_prefill(model)
        self.assertEqual(restored["enabled_layers"], 2)
        for layer in model.layers:
            experts = layer.mlp.experts
            self.assertFalse(
                experts._gemma4_long_dominant_expert_prefill_disabled
            )
            self.assertEqual(
                experts._gemma4_long_dominant_expert_prefill_fail_reason,
                "",
            )
            self.assertFalse(
                experts._gemma4_long_dominant_expert_prefill_last_active
            )
            self.assertEqual(
                experts._gemma4_long_dominant_expert_prefill_last_guard_reason,
                "",
            )
            self.assertEqual(experts._gemma4_long_dominant_expert_prefill_hits, 0)
            self.assertEqual(
                experts._gemma4_long_dominant_expert_prefill_assignments,
                0,
            )
            self.assertEqual(
                experts._gemma4_long_dominant_expert_prefill_guard_misses,
                0,
            )
            self.assertFalse(experts._gemma4_long_padded_bmm_prefill_disabled)
            self.assertEqual(experts._gemma4_long_padded_bmm_prefill_fail_reason, "")
            self.assertFalse(experts._gemma4_long_padded_bmm_prefill_last_active)
            self.assertEqual(experts._gemma4_long_padded_bmm_prefill_hits, 0)
            self.assertEqual(experts._gemma4_long_padded_bmm_prefill_assignments, 0)
            self.assertEqual(
                experts._gemma4_long_dominant_expert_prefill_workspace,
                {},
            )
            self.assertEqual(experts._gemma4_long_padded_bmm_prefill_workspace, {})
            self.assertEqual(experts._segmented_prefill_workspace, {})

    def test_padded_prefill_promotion_requires_exact_tokens_and_real_speedup(self):
        runner = load_runner()
        reference_tokens = [[10, 20, 30], [11, 21, 31]]
        reference_rows = [
            {"token_ids": reference_tokens, "prefill_ms": 125.0},
            {"token_ids": reference_tokens, "prefill_ms": 120.0},
        ]
        moe_contract = {
            "exact": True,
            "prefill_layers": 30,
            "prefill_covered_layers": 30,
            "padded_bmm_prefill_layers": 30,
        }
        candidate_rows = [
            {
                "token_ids": reference_tokens,
                "prefill_ms": prefill_ms,
                "deterministic_moe_contract": moe_contract,
            }
            for prefill_ms in (101.0, 100.0, 99.0)
        ]

        accepted = runner.evaluate_guarded_padded_prefill_promotion(
            reference_rows,
            candidate_rows,
        )
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["decision"], "APPLY_PADDED_BMM")
        self.assertEqual(accepted["padded_bmm_prefill_layers"], 30)
        self.assertGreaterEqual(accepted["speedup"], 1.02)

        modern_padded_contract = {
            **moe_contract,
            "dominant_expert_prefill_layers": 0,
            "dominant_expert_prefill_disabled_layers": 0,
        }
        modern_padded_rows = [
            {
                **row,
                "deterministic_moe_contract": modern_padded_contract,
            }
            for row in candidate_rows
        ]
        modern_padded = runner.evaluate_guarded_padded_prefill_promotion(
            reference_rows,
            modern_padded_rows,
        )
        self.assertTrue(modern_padded["accepted"])
        self.assertEqual(modern_padded["decision"], "APPLY_PADDED_BMM")
        self.assertEqual(modern_padded["candidate"], "global_padded_bmm")

        dominant_contract = {
            **moe_contract,
            "padded_bmm_prefill_layers": 0,
            "dominant_expert_prefill_layers": 30,
            "dominant_expert_prefill_disabled_layers": 0,
        }
        dominant_rows = [
            {
                **row,
                "deterministic_moe_contract": dominant_contract,
            }
            for row in candidate_rows
        ]
        dominant = runner.evaluate_guarded_padded_prefill_promotion(
            reference_rows,
            dominant_rows,
        )
        self.assertTrue(dominant["accepted"])
        self.assertEqual(
            dominant["decision"],
            "APPLY_DOMINANT_EXPERT_HYBRID",
        )
        self.assertEqual(dominant["dominant_expert_prefill_layers"], 30)

        mismatched = [dict(row) for row in candidate_rows]
        mismatched[-1] = {
            **mismatched[-1],
            "token_ids": [[10, 20, 99], [11, 21, 31]],
        }
        rejected_tokens = runner.evaluate_guarded_padded_prefill_promotion(
            reference_rows,
            mismatched,
        )
        self.assertFalse(rejected_tokens["accepted"])
        self.assertEqual(
            rejected_tokens["reason"],
            "padded_bmm_tokens_differ_from_segmented_reference",
        )

        slow = [{**row, "prefill_ms": 121.0} for row in candidate_rows]
        rejected_speed = runner.evaluate_guarded_padded_prefill_promotion(
            reference_rows,
            slow,
        )
        self.assertFalse(rejected_speed["accepted"])
        self.assertEqual(
            rejected_speed["reason"],
            "padded_bmm_prefill_speedup_below_threshold",
        )

    def test_dominant_reference_stabilizes_after_decode_graph_capture(self):
        runner = load_runner()
        capture_tokens = [[563] * 64 for _ in range(16)]
        stable_tokens = [[496] * 64 for _ in range(16)]

        class Experts:
            def __init__(self):
                self._gemma4_long_dominant_expert_prefill_disabled = False
                self._gemma4_long_dominant_expert_prefill_fail_reason = ""
                self._gemma4_long_dominant_expert_prefill_last_active = False
                self._gemma4_long_dominant_expert_prefill_last_guard_reason = ""
                self._gemma4_long_dominant_expert_prefill_hits = 0
                self._gemma4_long_dominant_expert_prefill_assignments = 0
                self._gemma4_long_dominant_expert_prefill_guard_misses = 0
                self._gemma4_long_dominant_expert_prefill_workspace = {}
                self._gemma4_long_padded_bmm_prefill_disabled = True
                self._gemma4_long_padded_bmm_prefill_fail_reason = ""
                self._gemma4_long_padded_bmm_prefill_last_active = False
                self._gemma4_long_padded_bmm_prefill_workspace = {}
                self._segmented_prefill_workspace = {}

        class Layer:
            def __init__(self):
                self.mlp = type("MLP", (), {"experts": Experts()})()

        model = type("Model", (), {"layers": [Layer() for _ in range(30)]})()
        engine = type("Engine", (), {"model": model})()
        reference_contract = {
            "exact": True,
            "prefill_backend": "segmented_deterministic",
            "prefill_layers": 30,
            "prefill_covered_layers": 30,
            "padded_bmm_prefill_layers": 0,
            "padded_bmm_prefill_disabled_layers": 0,
            "padded_bmm_prefill_failures": [],
            "dominant_expert_prefill_layers": 0,
            "dominant_expert_prefill_disabled_layers": 30,
        }
        candidate_contract = {
            **reference_contract,
            "prefill_backend": "dominant_expert_hybrid_fp32",
            "dominant_expert_prefill_layers": 30,
            "dominant_expert_prefill_disabled_layers": 0,
        }

        def normalized(tokens, prefill_ms, contract):
            return {
                "token_ids": tokens,
                "total_ms": prefill_ms + 100.0,
                "prefill_ms": prefill_ms,
                "decode_ms": 100.0,
                "output_tok_s_total": 1.0,
                "decode_tok_s": 1.0,
                "prefill_tok_s": 1.0,
                "input_plus_output_tok_s": 1.0,
                "decode_measurement_method": "test",
                "phase_metrics_status": "valid",
                "deterministic_moe_contract": contract,
            }

        reference_runtime = {
            "gemma4_long_dominant_expert_prefill_enabled": True,
            "gemma4_long_dominant_expert_prefill_last_active_layers": 0,
            "gemma4_long_dominant_expert_prefill_disabled_layers": 30,
            "qwen3_moe_segmented_prefill_deterministic_reduce_layers": 30,
            "qwen3_moe_segmented_prefill_atomic_reduce_layers": 0,
            "gemma4_fused_attn_moe_router_single_kernel_decode_enabled": True,
            "gemma4_fused_attn_moe_router_single_kernel_decode_hits": 1,
        }
        candidate_runtime = {
            **reference_runtime,
            "gemma4_long_dominant_expert_prefill_last_active_layers": 30,
            "gemma4_long_dominant_expert_prefill_disabled_layers": 0,
            "qwen3_moe_segmented_prefill_deterministic_reduce_layers": 0,
        }
        normalized_rows = [
            (normalized(capture_tokens, 1300.0, reference_contract), reference_runtime),
            *[
                (normalized(stable_tokens, 100.0, reference_contract), reference_runtime)
                for _ in range(3)
            ],
            *[
                (normalized(stable_tokens, 80.0, candidate_contract), candidate_runtime)
                for _ in range(6)
            ],
        ]
        prompts = {2048: [[7 + row] * 2048 for row in range(16)]}
        manifest = {
            "generator": "test",
            "cases": {
                "2048": {"contract": runner.prompt_token_contract(prompts[2048])}
            },
        }

        with tempfile.TemporaryDirectory() as raw:
            out_path = Path(raw) / "megagemm.json"
            args = self._fake_args(out_path)
            args.contexts = [2048]
            args.batch_sizes = [16]
            args.max_seq_len = 2112
            args.max_tokens = 64
            args.max_num_batched_tokens = 32768
            args.megagemm_determinism_auto_fallback = True
            with (
                patch.dict(
                    os.environ,
                    {
                        "MEGAGEMM_GEMMA4_B16_LONG_GRAPH_TOKEN_BURST_PROVEN": "1",
                        "MEGAGEMM_GEMMA4_MOE_LONG_DOMINANT_EXPERT_PREFILL": "1",
                    },
                ),
                patch("megagemm.engine.InferenceEngine", return_value=engine),
                patch.object(
                    runner,
                    "run_megagemm_request",
                    side_effect=[{} for _ in normalized_rows],
                ),
                patch.object(
                    runner,
                    "normalize_megagemm_row",
                    side_effect=normalized_rows,
                ),
                patch.object(
                    runner,
                    "gpu_snapshot",
                    return_value={"name": "NVIDIA A100-SXM4-80GB"},
                ),
            ):
                result = runner.run_megagemm_sweep(args, prompts, manifest)

        case = result["cases"]["b16_c2048"]
        reference = case["segmented_prefill_reference"]
        self.assertEqual(reference["status"], "stable")
        self.assertEqual(len(reference["runs"]), 4)
        self.assertTrue(reference["stability"]["stable"])
        self.assertEqual(reference["stability"]["consecutive_stable_pairs"], 2)
        self.assertFalse(
            runner.token_matrix_comparison(
                reference["runs"][0]["token_ids"],
                reference["runs"][1]["token_ids"],
            )["exact"]
        )
        self.assertEqual(case["prefill_profile"], runner.MEGAGEMM_PREFILL_PROFILE_HYBRID)
        self.assertEqual(
            case["prefill_promotion_gate"]["decision"],
            "APPLY_DOMINANT_EXPERT_HYBRID",
        )
        self.assertEqual(len(case["samples"]), 3)

    def test_paid_burst_sweep_uses_skew_gated_segmented_in_one_model_load(self):
        runner = load_runner()
        tokens = [[1000 + row, 2000 + row] for row in range(16)]

        class Experts:
            def __init__(self):
                self._gemma4_long_padded_bmm_prefill_disabled = False
                self._gemma4_long_padded_bmm_prefill_fail_reason = ""
                self._gemma4_long_padded_bmm_prefill_last_active = False
                self._gemma4_long_padded_bmm_prefill_hits = 0
                self._gemma4_long_padded_bmm_prefill_assignments = 0
                self._gemma4_long_padded_bmm_prefill_workspace = {}

        class Layer:
            def __init__(self):
                self.mlp = type("MLP", (), {"experts": Experts()})()

        model = type("Model", (), {"layers": [Layer() for _ in range(30)]})()
        engine = type("Engine", (), {"model": model})()

        def normalized_row(prefill_ms):
            return {
                "token_ids": tokens,
                "total_ms": prefill_ms + 110.0,
                "prefill_ms": prefill_ms,
                "decode_ms": 100.0,
                "output_tok_s_total": 1.0,
                "decode_tok_s": 1.0,
                "prefill_tok_s": 1.0,
                "input_plus_output_tok_s": 1.0,
                "decode_measurement_method": "test",
                "phase_metrics_status": "valid",
                "deterministic_moe_contract": {
                    "exact": True,
                    "prefill_backend": "segmented_deterministic_skew_gated",
                    "prefill_layers": 30,
                    "prefill_covered_layers": 30,
                    "padded_bmm_prefill_layers": 0,
                    "padded_bmm_prefill_disabled_layers": 0,
                    "padded_bmm_prefill_failures": [],
                },
            }

        reference_runtime = {
            "gemma4_a4b_segmented_prefill_layers": 30,
            "gemma4_a4b_segmented_prefill_config": {
                "long_rows": 16_384,
                "long": {
                    "block_m": 128,
                    "block_n": 256,
                    "block_k": 64,
                    "fused_gate_block_n": 128,
                    "num_warps": 8,
                    "num_stages": 3,
                    "compact_route_pack": False,
                },
            },
            "qwen3_moe_segmented_prefill_deterministic_reduce_layers": 30,
            "qwen3_moe_segmented_prefill_atomic_reduce_layers": 0,
            "gemma4_long_padded_bmm_prefill_enabled": False,
            "gemma4_long_padded_bmm_prefill_last_active_layers": 0,
            "gemma4_long_padded_bmm_prefill_disabled_layers": 0,
            "gemma4_fused_attn_moe_router_single_kernel_decode_enabled": True,
            "gemma4_fused_attn_moe_router_single_kernel_decode_hits": 1,
        }
        normalized = [
            (normalized_row(100.0), reference_runtime)
            for _ in range(6)
        ]
        prompts = {2048: [[7 + row] * 2048 for row in range(16)]}
        manifest = {
            "generator": "test",
            "cases": {
                "2048": {"contract": runner.prompt_token_contract(prompts[2048])}
            },
        }

        with tempfile.TemporaryDirectory() as raw:
            out_path = Path(raw) / "megagemm.json"
            args = self._fake_args(out_path)
            args.contexts = [2048]
            args.batch_sizes = [16]
            args.max_seq_len = 2112
            args.max_tokens = 64
            args.max_num_batched_tokens = 32768
            args.megagemm_determinism_auto_fallback = True
            with (
                patch.dict(
                    "os.environ",
                    {"MEGAGEMM_GEMMA4_B16_LONG_GRAPH_TOKEN_BURST_PROVEN": "1"},
                ),
                patch("megagemm.engine.InferenceEngine", return_value=engine),
                patch.object(
                    runner,
                    "run_megagemm_request",
                    side_effect=[{} for _ in normalized],
                ),
                patch.object(
                    runner,
                    "normalize_megagemm_row",
                    side_effect=normalized,
                ),
                patch.object(
                    runner,
                    "gpu_snapshot",
                    return_value={"name": "NVIDIA A100-SXM4-80GB"},
                ),
            ):
                result = runner.run_megagemm_sweep(args, prompts, manifest)

        case = result["cases"]["b16_c2048"]
        self.assertEqual(case["status"], "complete")
        self.assertEqual(
            case["decode_execution_mode"],
            runner.MEGAGEMM_DECODE_MODE_GRAPH_BURST,
        )
        self.assertEqual(
            case["prefill_profile"],
            runner.MEGAGEMM_PREFILL_PROFILE_SEGMENTED,
        )
        self.assertIsNone(case["prefill_promotion_gate"])
        self.assertIsNone(case["segmented_prefill_reference"])
        self.assertIsNone(case["prefill_fallback"])
        self.assertEqual(case["decode_mode_attempts"], [])
        self.assertEqual(len(case["samples"]), 3)

    def test_b16_c2048_sweep_falls_back_and_measures_eager(self):
        runner = load_runner()

        def tokens(last_token):
            return [[10, 20, last_token] for _ in range(16)]

        rows = [
            *[{"token_ids": tokens(value)} for value in (100, 101, 102)],
            *[{"token_ids": tokens(496)} for _ in range(6)],
        ]
        seen_modes = []

        def normalize(row, *, decode_mode, **_kwargs):
            seen_modes.append(decode_mode)
            normalized = {
                **row,
                "scheduler_prefill_ms": 2.0,
                "scheduler_decode_ms": 1.0,
                "total_ms": 3.0,
                "output_tok_s_total": 1.0,
                "prefill_ms": 2.0,
                "decode_ms": 1.0,
                "decode_tok_s": 1.0,
                "decode_measurement_method": "test",
                "decode_execution_mode": decode_mode,
                "decode_graph_scope": (
                    "disabled"
                    if decode_mode == runner.MEGAGEMM_DECODE_MODE_EAGER
                    else "request_local"
                ),
                "phase_metrics_status": "valid",
                "prefill_tok_s": 1.0,
                "input_plus_output_tok_s": 1.0,
                "deterministic_moe_contract": {
                    "padded_bmm_prefill_layers": 30,
                    "padded_bmm_prefill_disabled_layers": 0,
                    "padded_bmm_prefill_failures": [],
                },
            }
            runtime = {
                "gemma4_fused_attn_moe_router_single_kernel_decode_enabled": True,
                "gemma4_fused_attn_moe_router_single_kernel_decode_hits": 1,
            }
            return normalized, runtime

        prompts = {2048: [[7 + index] * 2048 for index in range(16)]}
        manifest = {
            "generator": "test",
            "cases": {
                "2048": {"contract": runner.prompt_token_contract(prompts[2048])}
            },
        }

        with tempfile.TemporaryDirectory() as raw:
            out_path = Path(raw) / "megagemm.json"
            args = self._fake_args(out_path)
            args.contexts = [2048]
            args.batch_sizes = [16]
            args.max_seq_len = 2112
            args.max_num_batched_tokens = 32768
            with (
                patch("megagemm.engine.InferenceEngine", return_value=object()),
                patch.object(runner, "run_megagemm_request", side_effect=rows),
                patch.object(runner, "normalize_megagemm_row", side_effect=normalize),
                patch.object(runner, "gpu_snapshot", return_value={"name": "test GPU"}),
            ):
                result = runner.run_megagemm_sweep(args, prompts, manifest)

        case = result["cases"]["b16_c2048"]
        self.assertEqual(case["status"], "complete")
        self.assertEqual(case["decode_execution_mode"], "eager")
        self.assertEqual(len(case["decode_mode_attempts"]), 1)
        self.assertTrue(
            case["decode_mode_attempts"][0]["warmup_stability"][
                "mode_rejected_early"
            ]
        )
        self.assertEqual(len(case["warmups"]), 3)
        self.assertEqual(len(case["samples"]), 3)
        self.assertEqual(
            seen_modes,
            [runner.MEGAGEMM_DECODE_MODE_GRAPH_STEP] * 3
            + [runner.MEGAGEMM_DECODE_MODE_EAGER] * 6,
        )

    def test_b16_c2048_sweep_selects_segmented_graph_before_eager(self):
        runner = load_runner()

        class Experts:
            def __init__(self):
                self._gemma4_long_padded_bmm_prefill_disabled = False
                self._gemma4_long_padded_bmm_prefill_fail_reason = ""
                self._gemma4_long_padded_bmm_prefill_last_active = True
                self._gemma4_long_padded_bmm_prefill_workspace = {"buffer": 1}

        class Layer:
            def __init__(self):
                self.mlp = type("MLP", (), {"experts": Experts()})()

        engine = type(
            "Engine",
            (),
            {"model": type("Model", (), {"layers": [Layer() for _ in range(30)]})()},
        )()

        def tokens(last_token):
            return [[10, 20, last_token] for _ in range(16)]

        rows = [{"token_ids": tokens(496)} for _ in range(6)]
        seen_modes = []

        def normalize(row, *, decode_mode, **_kwargs):
            seen_modes.append(decode_mode)
            normalized = {
                **row,
                "scheduler_prefill_ms": 2.0,
                "scheduler_decode_ms": 1.0,
                "total_ms": 3.0,
                "output_tok_s_total": 1.0,
                "prefill_ms": 2.0,
                "decode_ms": 1.0,
                "decode_tok_s": 1.0,
                "decode_measurement_method": "test",
                "decode_execution_mode": decode_mode,
                "decode_graph_scope": "request_local",
                "phase_metrics_status": "valid",
                "prefill_tok_s": 1.0,
                "input_plus_output_tok_s": 1.0,
                "deterministic_moe_contract": {
                    "padded_bmm_prefill_layers": 0,
                    "padded_bmm_prefill_disabled_layers": 30,
                    "padded_bmm_prefill_failures": [],
                },
            }
            runtime = {
                "gemma4_fused_attn_moe_router_single_kernel_decode_enabled": True,
                "gemma4_fused_attn_moe_router_single_kernel_decode_hits": 1,
            }
            return normalized, runtime

        prompts = {2048: [[7 + index] * 2048 for index in range(16)]}
        manifest = {
            "generator": "test",
            "cases": {
                "2048": {"contract": runner.prompt_token_contract(prompts[2048])}
            },
        }

        with tempfile.TemporaryDirectory() as raw:
            out_path = Path(raw) / "megagemm.json"
            args = self._fake_args(out_path)
            args.contexts = [2048]
            args.batch_sizes = [16]
            args.max_seq_len = 2112
            args.max_num_batched_tokens = 32768
            args.megagemm_determinism_auto_fallback = True
            with (
                patch("megagemm.engine.InferenceEngine", return_value=engine),
                patch.object(runner, "run_megagemm_request", side_effect=rows),
                patch.object(runner, "normalize_megagemm_row", side_effect=normalize),
                patch.object(runner, "gpu_snapshot", return_value={"name": "test GPU"}),
            ):
                result = runner.run_megagemm_sweep(args, prompts, manifest)

        case = result["cases"]["b16_c2048"]
        self.assertEqual(case["status"], "complete")
        self.assertEqual(
            case["prefill_profile"],
            runner.MEGAGEMM_PREFILL_PROFILE_SEGMENTED,
        )
        self.assertEqual(
            case["decode_execution_mode"],
            runner.MEGAGEMM_DECODE_MODE_GRAPH_STEP,
        )
        self.assertIsNone(case["prefill_fallback"])
        self.assertEqual(len(case["decode_mode_attempts"]), 0)
        self.assertEqual(
            seen_modes,
            [runner.MEGAGEMM_DECODE_MODE_GRAPH_STEP] * 6,
        )

    def test_b16_c2048_cost_guard_stops_every_unstable_profile_at_three(self):
        runner = load_runner()

        class Experts:
            def __init__(self):
                self._gemma4_long_padded_bmm_prefill_disabled = False
                self._gemma4_long_padded_bmm_prefill_fail_reason = ""
                self._gemma4_long_padded_bmm_prefill_last_active = True
                self._gemma4_long_padded_bmm_prefill_workspace = {}

        class Layer:
            def __init__(self):
                self.mlp = type("MLP", (), {"experts": Experts()})()

        engine = type(
            "Engine",
            (),
            {"model": type("Model", (), {"layers": [Layer() for _ in range(30)]})()},
        )()

        def tokens(last_token):
            return [[10, 20, last_token] for _ in range(16)]

        rows = [{"token_ids": tokens(value)} for value in range(100, 106)]

        def normalize(row, *, decode_mode, **_kwargs):
            normalized = {
                **row,
                "scheduler_prefill_ms": 2.0,
                "scheduler_decode_ms": 1.0,
                "total_ms": 3.0,
                "output_tok_s_total": 1.0,
                "prefill_ms": 2.0,
                "decode_ms": 1.0,
                "decode_tok_s": 1.0,
                "decode_measurement_method": "test",
                "decode_execution_mode": decode_mode,
                "decode_graph_scope": (
                    "disabled"
                    if decode_mode == runner.MEGAGEMM_DECODE_MODE_EAGER
                    else "request_local"
                ),
                "phase_metrics_status": "valid",
                "prefill_tok_s": 1.0,
                "input_plus_output_tok_s": 1.0,
                "deterministic_moe_contract": {
                    "padded_bmm_prefill_layers": 0,
                    "padded_bmm_prefill_disabled_layers": 30,
                    "padded_bmm_prefill_failures": [],
                },
            }
            runtime = {
                "gemma4_fused_attn_moe_router_single_kernel_decode_enabled": True,
                "gemma4_fused_attn_moe_router_single_kernel_decode_hits": 1,
            }
            return normalized, runtime

        prompts = {2048: [[7 + index] * 2048 for index in range(16)]}
        manifest = {
            "generator": "test",
            "cases": {
                "2048": {"contract": runner.prompt_token_contract(prompts[2048])}
            },
        }

        with tempfile.TemporaryDirectory() as raw:
            out_path = Path(raw) / "megagemm.json"
            args = self._fake_args(out_path)
            args.contexts = [2048]
            args.batch_sizes = [16]
            args.max_seq_len = 2112
            args.max_num_batched_tokens = 32768
            args.megagemm_determinism_auto_fallback = True
            with (
                patch("megagemm.engine.InferenceEngine", return_value=engine),
                patch.object(
                    runner,
                    "run_megagemm_request",
                    side_effect=rows,
                ) as request,
                patch.object(runner, "normalize_megagemm_row", side_effect=normalize),
                patch.object(runner, "gpu_snapshot", return_value={"name": "test GPU"}),
            ):
                with self.assertRaisesRegex(RuntimeError, "did not stabilize"):
                    runner.run_megagemm_sweep(args, prompts, manifest)

            persisted = json.loads(out_path.read_text(encoding="utf-8"))

        case = persisted["cases"]["b16_c2048"]
        self.assertEqual(request.call_count, 6)
        self.assertEqual(len(case["decode_mode_attempts"]), 2)
        self.assertTrue(
            all(
                len(attempt["warmups"]) == 3
                and attempt["warmup_stability"]["mode_rejected_early"]
                for attempt in case["decode_mode_attempts"]
            )
        )
        self.assertFalse(
            case["decode_mode_attempts"][-1]["warmup_stability"][
                "fallback_available"
            ]
        )

    def test_unstable_warmup_stops_before_measurement_and_persists_diagnostics(self):
        runner = load_runner()
        rows = [
            self._fake_megagemm_row([[10, 20, token]])
            for token in (100, 101, 102, 103)
        ]
        prompts = {1024: [[7] * 1024]}
        manifest = {
            "generator": "test",
            "cases": {
                "1024": {"contract": runner.prompt_token_contract(prompts[1024])}
            },
        }

        with tempfile.TemporaryDirectory() as raw:
            out_path = Path(raw) / "megagemm.json"
            args = self._fake_args(out_path)
            args.megagemm_max_warmups = 4
            with (
                patch("megagemm.engine.InferenceEngine", return_value=object()),
                patch.object(
                    runner,
                    "run_megagemm_request",
                    side_effect=rows,
                ),
                patch.object(runner, "gpu_snapshot", return_value={"name": "test GPU"}),
            ):
                with self.assertRaisesRegex(RuntimeError, "did not stabilize"):
                    runner.run_megagemm_sweep(args, prompts, manifest)

            persisted = json.loads(out_path.read_text(encoding="utf-8"))
            case = persisted["cases"]["b1_c1024"]
            self.assertEqual(case["status"], "failed")
            self.assertEqual(len(case["warmups"]), 4)
            self.assertEqual(case["samples"], [])
            self.assertTrue(case["warmup_stability"]["budget_exhausted"])

    def test_resume_preserves_complete_case_and_runs_only_missing_case(self):
        runner = load_runner()
        prompts = {
            1024: [[7] * 1024],
            2048: [[8] * 2048],
        }
        manifest = {
            "generator": "test",
            "cases": {
                str(context): {
                    "contract": runner.prompt_token_contract(rows),
                }
                for context, rows in prompts.items()
            },
        }
        rows = [
            *[
                self._fake_megagemm_row([[10, 20, 496]])
                for _ in range(6)
            ],
        ]

        with tempfile.TemporaryDirectory() as raw:
            out_path = Path(raw) / "megagemm.json"
            args = self._fake_args(out_path)
            args.contexts = [1024, 2048]
            args.max_seq_len = 2112
            args.max_num_batched_tokens = 2048
            args.resume = True
            with patch.object(
                runner,
                "gpu_snapshot",
                return_value={"name": "test GPU"},
            ):
                persisted = runner._base_result(args, manifest)
            persisted["cases"]["b1_c1024"] = {
                "status": "complete",
                "batch_size": 1,
                "context": 1024,
                "prompt_contract": runner.prompt_token_contract(prompts[1024]),
                "summary": {"preserved": True},
            }
            runner._write_result(out_path, persisted)

            with (
                patch("megagemm.engine.InferenceEngine", return_value=object()),
                patch.object(
                    runner,
                    "run_megagemm_request",
                    side_effect=rows,
                ) as request,
                patch.object(
                    runner,
                    "gpu_snapshot",
                    return_value={"name": "test GPU"},
                ),
            ):
                result = runner.run_megagemm_sweep(args, prompts, manifest)

            self.assertEqual(request.call_count, 6)
            self.assertTrue(result["cases"]["b1_c1024"]["summary"]["preserved"])
            self.assertEqual(result["cases"]["b1_c2048"]["status"], "complete")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(
                result["resume"]["skipped_complete_cases"],
                ["b1_c1024"],
            )
            self.assertEqual(
                result["resume"]["completed_during_resume"],
                ["b1_c2048"],
            )
            self.assertEqual(result["resume"]["pending_cases"], [])

    def test_comparison_reports_ratios_and_token_parity(self):
        runner = load_runner()
        common = {
            "status": "complete",
            "model": "/content/model",
            "dtype": "bf16",
            "contexts": [1024],
            "batch_sizes": [1],
            "max_seq_len": 1088,
            "max_tokens": 64,
            "vllm_max_num_batched_tokens": 1024,
            "prompt_contracts": {"1024": {"sha256": "same"}},
            "gpu": {"name": "test GPU"},
        }

        def case(prefill_ms, decode_tok_s, total_tok_s, tokens):
            return {
                "batch_size": 1,
                "context": 1024,
                "prompt_contract": {"sha256": "same-case"},
                "summary": {
                    "prefill_ms_median": prefill_ms,
                    "decode_tok_s_median": decode_tok_s,
                    "output_tok_s_total_median": total_tok_s,
                },
                "samples": [{"token_ids": [tokens]}],
            }

        megagemm = {
            **common,
            "backend": "megagemm",
            "cases": {"b1_c1024": case(80.0, 120.0, 100.0, [1, 2])},
        }
        vllm = {
            **common,
            "backend": "vllm",
            "version": "test",
            "cases": {"b1_c1024": case(100.0, 100.0, 80.0, [1, 3])},
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            mg_path = root / "megagemm.json"
            vl_path = root / "vllm.json"
            out_path = root / "comparison.json"
            mg_path.write_text(json.dumps(megagemm), encoding="utf-8")
            vl_path.write_text(json.dumps(vllm), encoding="utf-8")
            result = runner.compare_results(mg_path, vl_path, out_path)

        comparison = result["cases"]["b1_c1024"]
        self.assertAlmostEqual(comparison["prefill_speedup"], 1.25)
        self.assertAlmostEqual(comparison["decode_throughput_ratio"], 1.2)
        self.assertAlmostEqual(comparison["total_output_throughput_ratio"], 1.25)
        self.assertFalse(result["all_tokens_exact"])
        self.assertEqual(result["execution_scope"], "same_vm")
        self.assertEqual(result["resumed_megagemm_cases"], [])
        self.assertEqual(result["result_class"], "SHAPE_MATCHED_PERFORMANCE_ONLY")


if __name__ == "__main__":
    unittest.main()

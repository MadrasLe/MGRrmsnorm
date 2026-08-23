import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "run_publication_gpu_suite.py"


def load_runner():
    name = "publication_gpu_suite_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load publication GPU suite")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_dense_gemma4_models_select_fast_profile_automatically():
    runner = load_runner()

    assert (
        runner.resolve_megagemm_profile("auto", "google/gemma-4-E2B-it")
        == "gemma4-dense-fast"
    )
    assert (
        runner.resolve_megagemm_profile("auto", "google/gemma-4-E4B-it")
        == "gemma4-dense-fast"
    )
    assert runner.resolve_megagemm_profile("auto", "Qwen/Qwen2.5-3B") == "none"
    assert runner.resolve_megagemm_profile("none", "google/gemma-4-E4B-it") == "none"


def test_fast_profile_is_scoped_to_megagemm_child():
    runner = load_runner()

    mega_env = runner.child_environment(runner.VARIANTS["megagemm-bf16"], "gemma4-dense-fast")
    vllm_env = runner.child_environment(runner.VARIANTS["vllm-bf16"], "gemma4-dense-fast")

    assert mega_env["MEGAGEMM_FLAT_DECODE"] == "1"
    assert mega_env["MEGAGEMM_DECODE_CUDA_GRAPHS"] == "1"
    assert mega_env["MEGAGEMM_GEMMA4_FUSED_GATEUP_DECODE"] == "1"
    assert mega_env["MEGAGEMM_GEMMA4_DEEPFUSION_MLP_DECODE"] == "1"
    assert "MEGAGEMM_DECODE_CUDA_GRAPHS" not in vllm_env


def test_fast_path_audit_proves_runtime_and_reports_selected_kernels(tmp_path):
    runner = load_runner()
    raw_path = tmp_path / "mega.jsonl"
    row = {
        "ok": True,
        "decode_runtime_stats": {
            "flat_decode_ready": True,
            "flat_decode_failed": False,
            "gemma4_flat_fused_qkv_layers": 42,
            "fused_rope_attn_total_hits": 42,
            "gemma4_fused_qkv_prefill_hits": 3,
            "gemma4_fused_attn_prepare_hits": 2,
            "gemma4_batch_prefill_vectorized_kv_hits": 7,
            "gemma4_flat_fused_gateup_hits": 80,
            "gemma4_flat_deepfusion_hits": 70,
            "gemma4_fused_dual_ffn_norm_prefill_hits": 4,
            "gemma4_fused_add_ffn_norm_prefill_hits": 5,
            "gemma4_fused_post_ffn_norm_prefill_hits": 6,
            "paged_decode_runtime": {
                "generic_direct_hits": 100,
                "gqa2_direct_hits": 0,
            },
            "fused_rmsnorm_lm_head_argmax_use": True,
            "fused_lm_head_argmax_use": False,
        },
        "scheduler_stats": {
            "decode_cuda_graphs": {
                "enabled": True,
                "captures": 2,
                "replays": 30,
                "failures": 0,
                "request_scheduler_reuse_count": 3,
            }
        },
    }
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = runner.audit_gemma4_dense_fast_path(raw_path)

    assert report["status"] == "passed"
    assert report["required"]["decode_cuda_graph_replays"] == 30
    assert report["selected_kernel_counters"]["gemma4_flat_fused_qkv_layers"] == 42
    assert report["selected_kernel_counters"]["gemma4_flat_fused_gateup"] == 80
    assert report["selected_kernel_counters"]["gemma4_flat_deepfusion"] == 70
    assert report["selected_kernel_counters"]["paged_attention_generic_direct"] == 100
    assert report["selected_lm_head"]["fused_rmsnorm_lm_head_argmax"] is True


def test_fast_path_audit_rejects_eager_fallback(tmp_path):
    runner = load_runner()
    raw_path = tmp_path / "mega.jsonl"
    row = {
        "ok": True,
        "decode_runtime_stats": {
            "flat_decode_ready": False,
            "flat_decode_failed": True,
            "flat_decode_failed_reason": "unsupported shape",
        },
        "scheduler_stats": {
            "decode_cuda_graphs": {
                "enabled": True,
                "captures": 0,
                "replays": 0,
                "failures": 1,
                "last_failure": "capture failed",
            }
        },
    }
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = runner.audit_gemma4_dense_fast_path(raw_path)

    assert report["status"] == "failed"
    assert any("flat decode not ready" in error for error in report["errors"])
    assert any("zero replays" in error for error in report["errors"])
    assert any("capture failed" in error for error in report["errors"])

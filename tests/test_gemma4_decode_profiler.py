import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILER = ROOT / "benchmarks" / "profile_gemma4_decode.py"


def _load_profiler():
    spec = importlib.util.spec_from_file_location(
        "gemma4_decode_profiler",
        PROFILER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_profiler_recreates_promoted_profile_and_reports_current_paths():
    source = PROFILER.read_text(encoding="utf-8")

    assert "_configure_megagemm_profile(args.model)" in source
    assert 'os.environ["MEGAGEMM_DECODE_TIMING"] = "1"' in source
    assert "Production probe (no profiler, no timing events)" in source
    assert "Internal CUDA-event timing pass (production paths preserved)" in source
    assert "llama_runtime._DECODE_TIMING = False" in source
    assert "llama_runtime._DECODE_TIMING = True" in source
    assert "gemma4_ple_conditioned_gelu_decode_hits" in source
    assert "gemma4_cublaslt_gateup_decode_hits" in source
    assert "gemma4_dense_attn_mlp_bridge_decode_hits" in source
    assert '"grouped_segmented_hits"' in source
    assert '"attn_mlp_bridge"' in source


def test_runtime_counter_delta_includes_production_dense_and_attention_paths():
    profiler = _load_profiler()
    before = {
        "gemma4_dense_attn_mlp_bridge_decode_hits": 10,
        "gemma4_dense_post_norm_chain_decode_hits": 20,
        "paged_decode_runtime": {
            "gqa2_direct_hits": 30,
            "grouped_segmented_hits": 40,
        },
    }
    after = {
        "gemma4_dense_attn_mlp_bridge_decode_hits": 45,
        "gemma4_dense_post_norm_chain_decode_hits": 55,
        "paged_decode_runtime": {
            "gqa2_direct_hits": 58,
            "grouped_segmented_hits": 47,
        },
    }

    delta = profiler._runtime_counter_delta(before, after)

    assert delta["gemma4_dense_attn_mlp_bridge_decode_hits"] == 35
    assert delta["gemma4_dense_post_norm_chain_decode_hits"] == 35
    assert delta["paged_decode_gqa2_direct_hits"] == 28
    assert delta["paged_decode_grouped_segmented_hits"] == 7

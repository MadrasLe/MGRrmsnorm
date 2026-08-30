from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILER = ROOT / "benchmarks" / "profile_gemma4_decode.py"


def test_decode_profiler_recreates_promoted_profile_and_reports_current_paths():
    source = PROFILER.read_text(encoding="utf-8")

    assert "_configure_megagemm_profile(args.model)" in source
    assert 'os.environ["MEGAGEMM_DECODE_TIMING"] = "1"' in source
    assert "gemma4_ple_conditioned_gelu_decode_hits" in source
    assert "gemma4_cublaslt_gateup_decode_hits" in source
    assert "gemma4_dense_attn_mlp_bridge_decode_hits" in source
    assert '"grouped_segmented_hits"' in source

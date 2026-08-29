from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "megagemm" / "kernels" / "swiglu.py"
MODEL = ROOT / "megagemm" / "models" / "llama.py"
BENCHMARK = ROOT / "benchmarks" / "run_gemma4_e2b_ple_conditioned_gelu.py"
HARNESS = ROOT / "benchmarks" / "run_gemma4_e2b_ple_conditioned_gelu_colab.sh"


def _benchmark_module():
    spec = importlib.util.spec_from_file_location("ple_conditioned_gate", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kernel_fuses_exact_ple_formula_and_supports_strided_condition():
    source = KERNEL.read_text(encoding="utf-8")

    for expected in (
        "def _mg_conditioned_gelu_tanh_fwd_kernel(",
        "0.7978845608028654",
        "0.044715 * gate * gate * gate",
        ").to(tl.bfloat16)",
        "CONDITION_STRIDE_ROW",
        "def conditioned_gelu_tanh_forward(",
        "condition.stride(0)",
    ):
        assert expected in source
    assert ".contiguous()" not in source[
        source.index("def conditioned_gelu_tanh_forward(") :
    ]


def test_model_gate_is_exactly_scoped_and_falls_back_on_runtime_error():
    source = MODEL.read_text(encoding="utf-8")

    for expected in (
        'self.runtime_policy.name == "gemma4-e2b-l4"',
        "int(batch_size) == 8",
        "dtype == torch.bfloat16",
        "int(self.hidden_size_per_layer_input) == 256",
        "all(int(lw.ple_size) == 256 for lw in weights)",
        '"MEGAGEMM_GEMMA4_PLE_CONDITIONED_GELU_DECODE"',
        "ple = conditioned_gelu_tanh_forward(",
        "ple = torch.nn.functional.gelu(ple, approximate='tanh')",
        "ple.mul_(ple_condition)",
        "_gemma4_flat_ple_conditioned_gelu_runtime_disabled = True",
    ):
        assert expected in source


def test_full_model_gate_has_baseline_candidate_digest_and_runtime_audit():
    module = _benchmark_module()

    assert module.CASES == (("baseline", "0"), ("conditioned_gelu", "1"))
    baseline_env = module._case_environment("0")
    candidate_env = module._case_environment("1")
    assert baseline_env["MEGAGEMM_GEMMA4_PLE_CONDITIONED_GELU_DECODE"] == "0"
    assert candidate_env["MEGAGEMM_GEMMA4_PLE_CONDITIONED_GELU_DECODE"] == "1"
    assert candidate_env["MEGAGEMM_DECODE_CUDA_GRAPHS"] == "0"
    assert candidate_env["MEGAGEMM_DECODE_PREFER_STEP"] == "0"
    assert candidate_env["MEGAGEMM_REUSE_REQUEST_SCHEDULER"] == "0"
    assert candidate_env["MEGAGEMM_PAGED_DECODE_SPLITS"] == "1"
    assert candidate_env["MEGAGEMM_PAGED_DECODE_GQA2"] == "1"
    assert candidate_env["MEGAGEMM_PAGED_DECODE_WARPS_H256"] == "2"
    source = BENCHMARK.read_text(encoding="utf-8")
    for expected in (
        '"MEGAGEMM_BENCHMARK_TOKEN_DIGEST": "1"',
        "def _kernel_preflight()",
        'condition = per_layer[:, 0, 17, :]',
        "alias_exact",
        "repeat_exact",
        "max_abs_error <= 0.125",
        "gemma4_ple_conditioned_gelu_decode_hits",
        "gemma4_ple_conditioned_gelu_runtime_disabled",
        "conservative_speedup > 1.0",
        'decision = "PROMOTE_CONDITIONED_GELU"',
        '"--prompt-tokens",',
        '"2048",',
        '"--batch-sizes",',
        '"8",',
    ):
        assert expected in source


def test_colab_harness_uses_drive_code_without_git_or_package_mutation():
    source = HARNESS.read_text(encoding="utf-8")

    assert 'REPO="${REPO:-/content/drive/MyDrive/mg/MGRrmsnorm}"' in source
    assert 'export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"' in source
    assert "conditioned_gelu_tanh_forward" in source
    assert "git pull" not in source
    assert "pip install" not in source
    assert "bench_results/gemma4_e2b_ple_conditioned_gelu" in source

from types import SimpleNamespace

from megagemm.models.runtime_policy import policy_bool, resolve_runtime_policy


def _config(layers, hidden, q_heads, kv_heads, model_type="gemma4_text"):
    return SimpleNamespace(
        model_type=model_type,
        num_hidden_layers=layers,
        hidden_size=hidden,
        num_attention_heads=q_heads,
        num_key_value_heads=kv_heads,
    )


def test_e2b_l4_policy_preserves_measured_multi_step_triton_path():
    policy = resolve_runtime_policy(_config(35, 1536, 8, 1), "NVIDIA L4")

    assert policy.name == "gemma4-e2b-l4"
    assert policy.prefer_triton_rmsnorm is True
    assert policy.decode_prefer_step is False
    assert policy.reuse_request_scheduler is False


def test_e4b_l4_policy_preserves_measured_step_and_reuse_path():
    policy = resolve_runtime_policy(_config(42, 2560, 8, 2), "NVIDIA L4")

    assert policy.name == "gemma4-e4b-l4"
    assert policy.prefer_triton_rmsnorm is False
    assert policy.decode_prefer_step is True
    assert policy.reuse_request_scheduler is True


def test_gemma4_policy_is_not_promoted_to_unmeasured_hardware():
    policy = resolve_runtime_policy(_config(35, 1536, 8, 1), "NVIDIA A100")

    assert policy.name == "gemma4-generic"
    assert policy.decode_prefer_step is False
    assert policy.reuse_request_scheduler is False


def test_explicit_environment_flag_overrides_model_policy(monkeypatch):
    model = SimpleNamespace(
        runtime_policy=resolve_runtime_policy(
            _config(42, 2560, 8, 2), "NVIDIA L4"
        )
    )

    assert policy_bool(
        model,
        "MEGAGEMM_DECODE_PREFER_STEP",
        "decode_prefer_step",
    ) is True

    monkeypatch.setenv("MEGAGEMM_DECODE_PREFER_STEP", "0")
    assert policy_bool(
        model,
        "MEGAGEMM_DECODE_PREFER_STEP",
        "decode_prefer_step",
    ) is False

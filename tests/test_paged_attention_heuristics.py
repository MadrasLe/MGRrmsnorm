import os
from types import SimpleNamespace

import torch

import megagemm.kernels.paged_attention as paged_attention


def _clear_split_env():
    names = (
        "MEGAGEMM_PAGED_DECODE_SPLITS",
        "MEGAGEMM_PAGED_DECODE_MAX_SPLITS",
        "MEGAGEMM_PAGED_DECODE_SPLIT_MIN_BLOCKS",
        "MEGAGEMM_PAGED_DECODE_TARGET_WARPS_PER_SM",
    )
    return {name: os.environ.pop(name, None) for name in names}


def _restore_split_env(values):
    for name, value in values.items():
        if value is not None:
            os.environ[name] = value


def test_a100_long_context_auto_selects_40_splits():
    saved_env = _clear_split_env()
    old_device_info = paged_attention._cuda_device_info
    try:
        paged_attention._cuda_device_info = (
            lambda _device=None: ((8, 0), "NVIDIA A100-SXM4-80GB", 108)
        )
        actual = paged_attention._get_decode_split_count(
            1,
            8,
            136,
            num_warps=4,
            device=torch.device("cuda"),
        )
        assert actual == 40
    finally:
        paged_attention._cuda_device_info = old_device_info
        _restore_split_env(saved_env)


def test_non_a100_keeps_conservative_split_ceiling():
    saved_env = _clear_split_env()
    old_device_info = paged_attention._cuda_device_info
    try:
        paged_attention._cuda_device_info = (
            lambda _device=None: ((8, 9), "NVIDIA L4", 58)
        )
        actual = paged_attention._get_decode_split_count(
            1,
            8,
            136,
            num_warps=4,
            device=torch.device("cuda"),
        )
        assert actual == 8
    finally:
        paged_attention._cuda_device_info = old_device_info
        _restore_split_env(saved_env)


def test_model_split_policy_is_used_but_explicit_environment_wins():
    saved_env = _clear_split_env()
    old_device_info = paged_attention._cuda_device_info
    try:
        paged_attention._cuda_device_info = (
            lambda _device=None: ((8, 9), "NVIDIA L4", 58)
        )
        assert paged_attention._get_decode_split_count(
            1,
            8,
            136,
            device=torch.device("cuda"),
            policy_override=1,
        ) == 1

        os.environ["MEGAGEMM_PAGED_DECODE_SPLITS"] = "4"
        assert paged_attention._get_decode_split_count(
            1,
            8,
            136,
            device=torch.device("cuda"),
            policy_override=1,
        ) == 4
    finally:
        paged_attention._cuda_device_info = old_device_info
        _restore_split_env(saved_env)


def test_gemma4_h256_gqa2_direct_decode_is_explicit_and_unsplit_only():
    name = "MEGAGEMM_PAGED_DECODE_GQA2"
    saved_value = os.environ.get(name)
    saved_disabled = paged_attention._GQA2_DECODE_DISABLED
    try:
        os.environ[name] = "1"
        paged_attention._GQA2_DECODE_DISABLED = False
        assert paged_attention._use_gqa2_direct_decode(
            num_q_heads=16,
            num_kv_heads=8,
            head_dim=256,
            num_splits=1,
        )
        assert not paged_attention._use_gqa2_direct_decode(
            num_q_heads=16,
            num_kv_heads=8,
            head_dim=512,
            num_splits=1,
        )
        assert not paged_attention._use_gqa2_direct_decode(
            num_q_heads=16,
            num_kv_heads=8,
            head_dim=256,
            num_splits=2,
        )
    finally:
        paged_attention._GQA2_DECODE_DISABLED = saved_disabled
        if saved_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved_value


def test_gqa2_model_policy_is_used_but_explicit_environment_wins():
    name = "MEGAGEMM_PAGED_DECODE_GQA2"
    saved_value = os.environ.pop(name, None)
    saved_disabled = paged_attention._GQA2_DECODE_DISABLED
    try:
        paged_attention._GQA2_DECODE_DISABLED = False
        kwargs = {
            "num_q_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 256,
            "num_splits": 1,
        }

        assert paged_attention._use_gqa2_direct_decode(
            **kwargs,
            policy_enabled=True,
        )
        assert not paged_attention._use_gqa2_direct_decode(
            **kwargs,
            policy_enabled=False,
        )

        os.environ[name] = "0"
        assert not paged_attention._use_gqa2_direct_decode(
            **kwargs,
            policy_enabled=True,
        )

        os.environ[name] = "1"
        assert paged_attention._use_gqa2_direct_decode(
            **kwargs,
            policy_enabled=False,
        )
    finally:
        paged_attention._GQA2_DECODE_DISABLED = saved_disabled
        if saved_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved_value


def test_decode_num_warps_supports_independent_h256_and_h512_overrides():
    names = (
        "MEGAGEMM_PAGED_DECODE_WARPS",
        "MEGAGEMM_PAGED_DECODE_WARPS_H256",
        "MEGAGEMM_PAGED_DECODE_WARPS_H512",
    )
    saved_env = {name: os.environ.pop(name, None) for name in names}
    old_device_info = paged_attention._cuda_device_info
    try:
        paged_attention._cuda_device_info = (
            lambda _device=None: ((8, 0), "NVIDIA A100-SXM4-80GB", 108)
        )
        os.environ["MEGAGEMM_PAGED_DECODE_WARPS"] = "0"
        os.environ["MEGAGEMM_PAGED_DECODE_WARPS_H256"] = "4"
        os.environ["MEGAGEMM_PAGED_DECODE_WARPS_H512"] = "8"
        assert (
            paged_attention._decode_num_warps(
                256, torch.device("cuda"), num_splits=1
            )
            == 4
        )
        assert (
            paged_attention._decode_num_warps(
                512, torch.device("cuda"), num_splits=1
            )
            == 8
        )

        os.environ["MEGAGEMM_PAGED_DECODE_WARPS_H256"] = "8"
        os.environ["MEGAGEMM_PAGED_DECODE_WARPS_H512"] = "4"
        assert (
            paged_attention._decode_num_warps(
                256, torch.device("cuda"), num_splits=1
            )
            == 8
        )
        assert (
            paged_attention._decode_num_warps(
                512, torch.device("cuda"), num_splits=1
            )
            == 4
        )
    finally:
        paged_attention._cuda_device_info = old_device_info
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_gemma4_grouped_segmented_decode_policy_is_shape_selective():
    name = "MEGAGEMM_GEMMA4_GROUPED_SEGMENTED_ATTN_DECODE"
    saved_value = os.environ.get(name)
    saved_has_triton = paged_attention._HAS_TRITON
    saved_disabled = paged_attention._GROUPED_SEGMENTED_DECODE_DISABLED
    old_device_info = paged_attention._cuda_device_info

    def fake_tensor(shape, ndim):
        return SimpleNamespace(
            shape=shape,
            ndim=ndim,
            dtype=torch.bfloat16,
            is_cuda=True,
            device=torch.device("cuda"),
        )

    try:
        os.environ.pop(name, None)
        paged_attention._HAS_TRITON = True
        paged_attention._GROUPED_SEGMENTED_DECODE_DISABLED = False
        paged_attention._cuda_device_info = (
            lambda _device=None: ((8, 0), "NVIDIA A100-SXM4-80GB", 108)
        )

        block_tables = fake_tensor((16, 6), 2)
        h256_query = fake_tensor((16, 16, 256), 3)
        h256_cache = fake_tensor((96, 2, 8, 16, 256), 5)
        assert (
            paged_attention._grouped_segmented_decode_topology(
                h256_query,
                h256_cache,
                block_tables,
                sliding_window=1024,
            )
            == "sliding_h256_gqa2"
        )

        h512_query = fake_tensor((16, 16, 512), 3)
        h512_cache = fake_tensor((96, 2, 2, 16, 512), 5)
        assert (
            paged_attention._grouped_segmented_decode_topology(
                h512_query,
                h512_cache,
                block_tables,
                sliding_window=None,
            )
            == "full_h512_gqa8"
        )

        os.environ[name] = "0"
        assert (
            paged_attention._grouped_segmented_decode_topology(
                h256_query,
                h256_cache,
                block_tables,
                sliding_window=1024,
            )
            is None
        )
        os.environ.pop(name, None)

        batch8_query = fake_tensor((8, 16, 256), 3)
        assert (
            paged_attention._grouped_segmented_decode_topology(
                batch8_query,
                h256_cache,
                fake_tensor((8, 6), 2),
                sliding_window=1024,
            )
            is None
        )
    finally:
        paged_attention._HAS_TRITON = saved_has_triton
        paged_attention._GROUPED_SEGMENTED_DECODE_DISABLED = saved_disabled
        paged_attention._cuda_device_info = old_device_info
        if saved_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved_value


def test_gemma4_grouped_segmented_long_context_uses_paid_segment_counts():
    select = paged_attention._grouped_segmented_decode_num_segments

    assert select("sliding_h256_gqa2", 1008) == 16
    assert select("sliding_h256_gqa2", 1024) == 32
    assert select("full_h512_gqa8", 2032) == 16
    assert select("full_h512_gqa8", 2048) == 8


def test_gemma4_grouped_segmented_long_context_uses_paid_tile_sizes():
    select = paged_attention._grouped_segmented_decode_tile_size

    assert select("sliding_h256_gqa2", 1008) == 32
    assert select("sliding_h256_gqa2", 1024) == 64
    assert select("full_h512_gqa8", 2032) == 16
    assert select("full_h512_gqa8", 2048) == 16

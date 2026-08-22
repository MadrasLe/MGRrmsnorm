import torch

from megagemm.kernels import rmsnorm


class _FakeCudaTensor:
    def __init__(self, dtype):
        self.dtype = dtype
        self.is_cuda = True
        self.device = torch.device("cuda:0")


def test_cuda_rmsnorm_policy_allows_sm75_fp16(monkeypatch):
    monkeypatch.setattr(rmsnorm, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))

    assert rmsnorm.can_use_cuda_rmsnorm_for(_FakeCudaTensor(torch.float16)) is True


def test_cuda_rmsnorm_policy_blocks_sm75_bf16(monkeypatch):
    monkeypatch.setattr(rmsnorm, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))

    assert rmsnorm.can_use_cuda_rmsnorm_for(_FakeCudaTensor(torch.bfloat16)) is False


def test_cuda_rmsnorm_policy_allows_sm80_bf16(monkeypatch):
    monkeypatch.setattr(rmsnorm, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))

    assert rmsnorm.can_use_cuda_rmsnorm_for(_FakeCudaTensor(torch.bfloat16)) is True


def test_cuda_rmsnorm_policy_blocks_untested_bf16_arch_by_default(monkeypatch):
    monkeypatch.setattr(rmsnorm, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(rmsnorm, "_ALLOW_UNTESTED_CUDA_RMSNORM_ARCH", False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))

    assert rmsnorm.can_use_cuda_rmsnorm_for(_FakeCudaTensor(torch.bfloat16)) is False


def test_cuda_rmsnorm_policy_allows_untested_bf16_arch_when_forced(monkeypatch):
    monkeypatch.setattr(rmsnorm, "_CUDA_AVAILABLE", True)
    monkeypatch.setattr(rmsnorm, "_ALLOW_UNTESTED_CUDA_RMSNORM_ARCH", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))

    assert rmsnorm.can_use_cuda_rmsnorm_for(_FakeCudaTensor(torch.bfloat16)) is True


def test_cuda_rmsnorm_policy_blocks_offset_norm(monkeypatch):
    monkeypatch.setattr(rmsnorm, "_CUDA_AVAILABLE", True)

    assert rmsnorm.can_use_cuda_rmsnorm_for(_FakeCudaTensor(torch.float16), offset=True) is False

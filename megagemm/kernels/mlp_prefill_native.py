import torch

try:
    import rmsnorm_cuda_ops as _cuda_ops
except Exception:
    _cuda_ops = None


HAS_NATIVE_MLP_PREFILL = bool(
    _cuda_ops is not None
    and hasattr(_cuda_ops, "mlp_prefill_forward_cuda")
)


def mlp_prefill_forward_cuda(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor | None,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None,
    intermediate_size: int,
) -> torch.Tensor:
    if not HAS_NATIVE_MLP_PREFILL:
        raise RuntimeError("native MLP prefill CUDA op is unavailable")
    gate_up_bias_arg = gate_up_bias if gate_up_bias is not None else None
    down_bias_arg = down_bias if down_bias is not None else None
    return _cuda_ops.mlp_prefill_forward_cuda(
        x,
        gate_up_weight,
        gate_up_bias_arg,
        down_weight,
        down_bias_arg,
        int(intermediate_size),
    )

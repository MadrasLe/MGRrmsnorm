#pragma once

#include <torch/extension.h>

torch::Tensor swiglu_forward_cuda(torch::Tensor input, int64_t hidden_dim);

torch::Tensor mlp_prefill_forward_cuda(
    torch::Tensor input,
    torch::Tensor gate_up_weight,
    c10::optional<torch::Tensor> gate_up_bias,
    torch::Tensor down_weight,
    c10::optional<torch::Tensor> down_bias,
    int64_t intermediate_size
);

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

int64_t cublaslt_bf16_algorithm_count_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t maximum_algorithms
);

torch::Tensor cublaslt_bf16_linear_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> out,
    int64_t algorithm_index
);

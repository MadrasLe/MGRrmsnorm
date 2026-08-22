#pragma once

#include <torch/extension.h>

torch::Tensor sparse24_fp16_linear_cuda(
    torch::Tensor input,
    torch::Tensor values,
    torch::Tensor metadata,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output
);

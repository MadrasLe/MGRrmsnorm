#include <torch/extension.h>
#include "../src/rmsnorm_kernel.h"

// Check for CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> rmsnorm_forward(torch::Tensor input, torch::Tensor weight, float epsilon) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);

    int rows = input.size(0);
    int cols = input.size(1);

    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.size(0) == cols, "Weight size must match hidden size");

    auto options = input.options();
    auto output = torch::empty_like(input);
    auto inv_rms = torch::empty({rows}, options); // to save context

    bool success = rmsnorm_cuda(
        output.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        rows,
        cols,
        epsilon
    );

    TORCH_CHECK(success, "RMSNorm Forward CUDA kernel failed (check stderr for details, likely alignment or dimension issue)");

    return {output, inv_rms};
}

std::vector<torch::Tensor> rmsnorm_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_rms
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(inv_rms);

    int rows = input.size(0);
    int cols = input.size(1);

    auto grad_input = torch::empty_like(input);
    auto grad_weight = torch::empty_like(weight);

    bool success = rmsnorm_backward_cuda(
        grad_input.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        rows,
        cols
    );

    TORCH_CHECK(success, "RMSNorm Backward CUDA kernel failed (check stderr for details)");

    return {grad_input, grad_weight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
    m.def("rmsnorm_backward", &rmsnorm_backward, "RMSNorm backward (CUDA)");
}

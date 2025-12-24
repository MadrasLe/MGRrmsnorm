#include <torch/extension.h>
#include "../src/rmsnorm_kernel.h"

// CUDA includes for type casting
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Check macros
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// =============================================================================
// Forward Pass - Multi-dtype dispatch
// =============================================================================
std::vector<torch::Tensor> rmsnorm_forward(torch::Tensor input, torch::Tensor weight, float epsilon) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);

    int rows = input.size(0);
    int cols = input.size(1);

    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.size(0) == cols, "Weight size must match hidden size");

    auto dtype = input.scalar_type();
    
    // inv_rms is always FP32 for numerical stability
    auto inv_rms = torch::empty({rows}, input.options().dtype(torch::kFloat32));
    auto output = torch::empty_like(input);
    
    bool success = false;

    if (dtype == torch::kFloat32) {
        success = rmsnorm_cuda_fp32(
            output.data_ptr<float>(),
            inv_rms.data_ptr<float>(),
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            rows, cols, epsilon
        );
    } 
    else if (dtype == torch::kFloat16) {
        success = rmsnorm_cuda_fp16(
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            inv_rms.data_ptr<float>(),
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            rows, cols, epsilon
        );
    }
    else if (dtype == torch::kBFloat16) {
        success = rmsnorm_cuda_bf16(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            inv_rms.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            rows, cols, epsilon
        );
    }
    else {
        TORCH_CHECK(false, "Unsupported dtype. Use float32, float16, or bfloat16.");
    }

    TORCH_CHECK(success, "RMSNorm Forward CUDA kernel failed (check stderr for details)");

    return {output, inv_rms};
}

// =============================================================================
// Backward Pass - Multi-dtype dispatch
// =============================================================================
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

    auto dtype = input.scalar_type();
    
    auto grad_input = torch::empty_like(input);
    auto grad_weight = torch::empty_like(weight);

    bool success = false;

    if (dtype == torch::kFloat32) {
        success = rmsnorm_backward_cuda_fp32(
            grad_input.data_ptr<float>(),
            grad_weight.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            inv_rms.data_ptr<float>(),
            rows, cols
        );
    }
    else if (dtype == torch::kFloat16) {
        success = rmsnorm_backward_cuda_fp16(
            reinterpret_cast<__half*>(grad_input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(grad_weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            inv_rms.data_ptr<float>(),
            rows, cols
        );
    }
    else if (dtype == torch::kBFloat16) {
        success = rmsnorm_backward_cuda_bf16(
            reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(grad_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            inv_rms.data_ptr<float>(),
            rows, cols
        );
    }
    else {
        TORCH_CHECK(false, "Unsupported dtype for backward. Use float32, float16, or bfloat16.");
    }

    TORCH_CHECK(success, "RMSNorm Backward CUDA kernel failed (check stderr for details)");

    return {grad_input, grad_weight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA) - supports FP32/FP16/BF16");
    m.def("rmsnorm_backward", &rmsnorm_backward, "RMSNorm backward (CUDA) - supports FP32/FP16/BF16");
}

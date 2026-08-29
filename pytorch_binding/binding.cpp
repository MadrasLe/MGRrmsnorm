#include <torch/extension.h>
#include "../src/rmsnorm_kernel.h"
#include "../src/rope_kernel.h"
#include "../src/mlp_prefill_kernel.h"

// CUDA includes for type casting
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Check macros
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace {

bool device_supports_native_bf16(const torch::Tensor& tensor) {
    int device_index = tensor.get_device();
    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, device_index);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return prop.major >= 8;
}

}  // namespace

// =============================================================================
// RMSNorm Forward Pass - Multi-dtype dispatch
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
        TORCH_CHECK(
            device_supports_native_bf16(input),
            "BF16 RMSNorm CUDA path requires SM80+; use FP16 on SM75/T4."
        );
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
// RMSNorm Backward Pass - Multi-dtype dispatch
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
        TORCH_CHECK(
            device_supports_native_bf16(input),
            "BF16 RMSNorm backward CUDA path requires SM80+; use FP16 on SM75/T4."
        );
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

// =============================================================================
// RoPE Forward Pass - Multi-dtype dispatch
// =============================================================================
torch::Tensor rope_forward(
    torch::Tensor input,      // [batch, heads, seq_len, head_dim]
    torch::Tensor cos_cache,  // [max_seq_len, head_dim/2]
    torch::Tensor sin_cache,  // [max_seq_len, head_dim/2]
    c10::optional<torch::Tensor> position_ids  // [batch, seq_len] or None
) {
    CHECK_INPUT(input);
    CHECK_INPUT(cos_cache);
    CHECK_INPUT(sin_cache);

    TORCH_CHECK(input.dim() == 4, "Input must be 4D [batch, heads, seq_len, head_dim]");

    int batch = input.size(0);
    int heads = input.size(1);
    int seq_len = input.size(2);
    int head_dim = input.size(3);

    auto dtype = input.scalar_type();
    auto output = torch::empty_like(input);

    // Position IDs pointer (can be nullptr)
    const int* pos_ids_ptr = nullptr;
    if (position_ids.has_value()) {
        CHECK_INPUT(position_ids.value());
        pos_ids_ptr = position_ids.value().data_ptr<int>();
    }

    bool success = false;

    if (dtype == torch::kFloat32) {
        success = rope_forward_cuda_fp32(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            cos_cache.data_ptr<float>(),
            sin_cache.data_ptr<float>(),
            pos_ids_ptr,
            batch, heads, seq_len, head_dim
        );
    }
    else if (dtype == torch::kFloat16) {
        success = rope_forward_cuda_fp16(
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            cos_cache.data_ptr<float>(),  // cos/sin always FP32
            sin_cache.data_ptr<float>(),
            pos_ids_ptr,
            batch, heads, seq_len, head_dim
        );
    }
    else {
        TORCH_CHECK(false, "RoPE supports float32 or float16");
    }

    TORCH_CHECK(success, "RoPE Forward CUDA kernel failed");
    return output;
}

// =============================================================================
// RoPE Backward Pass - Multi-dtype dispatch
// =============================================================================
torch::Tensor rope_backward(
    torch::Tensor grad_output,
    torch::Tensor cos_cache,
    torch::Tensor sin_cache,
    c10::optional<torch::Tensor> position_ids
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(cos_cache);
    CHECK_INPUT(sin_cache);

    int batch = grad_output.size(0);
    int heads = grad_output.size(1);
    int seq_len = grad_output.size(2);
    int head_dim = grad_output.size(3);

    auto dtype = grad_output.scalar_type();
    auto grad_input = torch::empty_like(grad_output);

    const int* pos_ids_ptr = nullptr;
    if (position_ids.has_value()) {
        CHECK_INPUT(position_ids.value());
        pos_ids_ptr = position_ids.value().data_ptr<int>();
    }

    bool success = false;

    if (dtype == torch::kFloat32) {
        success = rope_backward_cuda_fp32(
            grad_input.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            cos_cache.data_ptr<float>(),
            sin_cache.data_ptr<float>(),
            pos_ids_ptr,
            batch, heads, seq_len, head_dim
        );
    }
    else if (dtype == torch::kFloat16) {
        success = rope_backward_cuda_fp16(
            reinterpret_cast<__half*>(grad_input.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
            cos_cache.data_ptr<float>(),
            sin_cache.data_ptr<float>(),
            pos_ids_ptr,
            batch, heads, seq_len, head_dim
        );
    }
    else {
        TORCH_CHECK(false, "RoPE backward supports float32 or float16");
    }

    TORCH_CHECK(success, "RoPE Backward CUDA kernel failed");
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA) - supports FP32/FP16/BF16");
    m.def("rmsnorm_backward", &rmsnorm_backward, "RMSNorm backward (CUDA) - supports FP32/FP16/BF16");
    m.def("rope_forward", &rope_forward, "RoPE forward (CUDA) - supports FP32/FP16");
    m.def("rope_backward", &rope_backward, "RoPE backward (CUDA) - supports FP32/FP16");
    m.def("swiglu_forward_cuda", &swiglu_forward_cuda, "SwiGLU forward (CUDA, FP16)");
    m.def("mlp_prefill_forward_cuda", &mlp_prefill_forward_cuda, "MLP prefill forward (CUDA, FP16)");
    m.def(
        "cublaslt_bf16_algorithm_count_cuda",
        &cublaslt_bf16_algorithm_count_cuda,
        "Return available cuBLASLt BF16 heuristics for input @ weight.T"
    );
    m.def(
        "cublaslt_bf16_linear_cuda",
        &cublaslt_bf16_linear_cuda,
        "cuBLASLt BF16 linear with an explicit heuristic index"
    );
}

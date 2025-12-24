#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// =============================================================================
// FP32 API (Original)
// =============================================================================
bool rmsnorm_cuda_fp32(
    float* out, 
    float* inv_rms, 
    const float* in, 
    const float* weight, 
    int rows, int cols, float epsilon
);

bool rmsnorm_backward_cuda_fp32(
    float* grad_input,
    float* grad_weight,
    const float* grad_output,
    const float* input,
    const float* weight,
    const float* inv_rms,
    int rows, int cols
);

// =============================================================================
// FP16 API
// =============================================================================
bool rmsnorm_cuda_fp16(
    __half* out, 
    float* inv_rms,  // Always FP32 for stability
    const __half* in, 
    const __half* weight, 
    int rows, int cols, float epsilon
);

bool rmsnorm_backward_cuda_fp16(
    __half* grad_input,
    __half* grad_weight,
    const __half* grad_output,
    const __half* input,
    const __half* weight,
    const float* inv_rms,
    int rows, int cols
);

// =============================================================================
// BF16 API
// =============================================================================
bool rmsnorm_cuda_bf16(
    __nv_bfloat16* out, 
    float* inv_rms,
    const __nv_bfloat16* in, 
    const __nv_bfloat16* weight, 
    int rows, int cols, float epsilon
);

bool rmsnorm_backward_cuda_bf16(
    __nv_bfloat16* grad_input,
    __nv_bfloat16* grad_weight,
    const __nv_bfloat16* grad_output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const float* inv_rms,
    int rows, int cols
);

// =============================================================================
// Legacy API (backward compatibility) - maps to FP32
// =============================================================================
inline bool rmsnorm_cuda(float* out, float* inv_rms, const float* in, const float* weight, int rows, int cols, float epsilon) {
    return rmsnorm_cuda_fp32(out, inv_rms, in, weight, rows, cols, epsilon);
}

inline bool rmsnorm_backward_cuda(float* grad_input, float* grad_weight, const float* grad_output, const float* input, const float* weight, const float* inv_rms, int rows, int cols) {
    return rmsnorm_backward_cuda_fp32(grad_input, grad_weight, grad_output, input, weight, inv_rms, rows, cols);
}

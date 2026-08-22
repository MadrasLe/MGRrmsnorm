#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward declarations for RoPE CUDA kernels
extern "C" {

bool rope_forward_cuda_fp32(
    float* output,
    const float* input,
    const float* cos_cache,
    const float* sin_cache,
    const int* position_ids,
    int batch, int heads, int seq_len, int head_dim
);

bool rope_forward_cuda_fp16(
    __half* output,
    const __half* input,
    const float* cos_cache,
    const float* sin_cache,
    const int* position_ids,
    int batch, int heads, int seq_len, int head_dim
);

bool rope_backward_cuda_fp32(
    float* grad_input,
    const float* grad_output,
    const float* cos_cache,
    const float* sin_cache,
    const int* position_ids,
    int batch, int heads, int seq_len, int head_dim
);

bool rope_backward_cuda_fp16(
    __half* grad_input,
    const __half* grad_output,
    const float* cos_cache,
    const float* sin_cache,
    const int* position_ids,
    int batch, int heads, int seq_len, int head_dim
);

} // extern "C"

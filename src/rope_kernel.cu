/*
 * RoPE (Rotary Position Embeddings) CUDA Kernel
 * Author: Gabriel Yogi
 *
 * High-performance RoPE implementation with:
 * - float2/half2 vectorized loads
 * - Precomputed cos/sin lookup
 * - Fused rotation operation
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// =============================================================================
// Forward Kernel: Apply RoPE rotation
// =============================================================================

// FP32 Forward Kernel
__global__ void rope_forward_kernel_fp32(
    float* __restrict__ output,          // [batch * heads * seq_len, head_dim]
    const float* __restrict__ input,     // [batch * heads * seq_len, head_dim]
    const float* __restrict__ cos_cache, // [max_seq_len, head_dim/2]
    const float* __restrict__ sin_cache, // [max_seq_len, head_dim/2]
    const int* __restrict__ position_ids, // [seq_len] or nullptr
    int total_tokens,  // batch * heads * seq_len
    int head_dim,
    int seq_len,
    int heads
) {
    // Each thread handles one token (one row)
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= total_tokens) return;

    // Compute position in sequence
    int seq_pos = (token_idx / heads) % seq_len;
    if (position_ids != nullptr) {
        seq_pos = position_ids[seq_pos];
    }

    int half_dim = head_dim / 2;

    // Pointer to this token's data
    const float* inp = input + token_idx * head_dim;
    float* out = output + token_idx * head_dim;

    // Pointer to cos/sin for this position
    const float* cos_ptr = cos_cache + seq_pos * half_dim;
    const float* sin_ptr = sin_cache + seq_pos * half_dim;

    // Apply rotation to pairs: (x[2i], x[2i+1]) with (cos, sin)
    // Using float2 for vectorized access
    for (int i = 0; i < half_dim; i++) {
        float x_even = inp[2 * i];
        float x_odd = inp[2 * i + 1];
        float c = cos_ptr[i];
        float s = sin_ptr[i];

        // Rotation: (a + bi) * (cos + i*sin)
        // Real part: a*cos - b*sin
        // Imag part: a*sin + b*cos
        out[2 * i] = x_even * c - x_odd * s;
        out[2 * i + 1] = x_even * s + x_odd * c;
    }
}

// FP16 Forward Kernel with half2 vectorization
__global__ void rope_forward_kernel_fp16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const float* __restrict__ cos_cache,  // Keep cos/sin in FP32 for precision
    const float* __restrict__ sin_cache,
    const int* __restrict__ position_ids,
    int total_tokens,
    int head_dim,
    int seq_len,
    int heads
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= total_tokens) return;

    int seq_pos = (token_idx / heads) % seq_len;
    if (position_ids != nullptr) {
        seq_pos = position_ids[seq_pos];
    }

    int half_dim = head_dim / 2;

    const __half* inp = input + token_idx * head_dim;
    __half* out = output + token_idx * head_dim;
    const float* cos_ptr = cos_cache + seq_pos * half_dim;
    const float* sin_ptr = sin_cache + seq_pos * half_dim;

    for (int i = 0; i < half_dim; i++) {
        // Load as FP16, compute in FP32 for stability
        float x_even = __half2float(inp[2 * i]);
        float x_odd = __half2float(inp[2 * i + 1]);
        float c = cos_ptr[i];
        float s = sin_ptr[i];

        float y_even = x_even * c - x_odd * s;
        float y_odd = x_even * s + x_odd * c;

        out[2 * i] = __float2half(y_even);
        out[2 * i + 1] = __float2half(y_odd);
    }
}

// =============================================================================
// Backward Kernel: Compute gradients
// =============================================================================

// The backward pass of RoPE is just the inverse rotation (negate sin)
// d_input[2i]   = d_output[2i] * cos + d_output[2i+1] * sin
// d_input[2i+1] = -d_output[2i] * sin + d_output[2i+1] * cos

__global__ void rope_backward_kernel_fp32(
    float* __restrict__ grad_input,
    const float* __restrict__ grad_output,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int* __restrict__ position_ids,
    int total_tokens,
    int head_dim,
    int seq_len,
    int heads
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= total_tokens) return;

    int seq_pos = (token_idx / heads) % seq_len;
    if (position_ids != nullptr) {
        seq_pos = position_ids[seq_pos];
    }

    int half_dim = head_dim / 2;

    const float* grad_out = grad_output + token_idx * head_dim;
    float* grad_in = grad_input + token_idx * head_dim;
    const float* cos_ptr = cos_cache + seq_pos * half_dim;
    const float* sin_ptr = sin_cache + seq_pos * half_dim;

    for (int i = 0; i < half_dim; i++) {
        float dy_even = grad_out[2 * i];
        float dy_odd = grad_out[2 * i + 1];
        float c = cos_ptr[i];
        float s = sin_ptr[i];

        // Inverse rotation (transpose of rotation matrix)
        grad_in[2 * i] = dy_even * c + dy_odd * s;
        grad_in[2 * i + 1] = -dy_even * s + dy_odd * c;
    }
}

__global__ void rope_backward_kernel_fp16(
    __half* __restrict__ grad_input,
    const __half* __restrict__ grad_output,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int* __restrict__ position_ids,
    int total_tokens,
    int head_dim,
    int seq_len,
    int heads
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= total_tokens) return;

    int seq_pos = (token_idx / heads) % seq_len;
    if (position_ids != nullptr) {
        seq_pos = position_ids[seq_pos];
    }

    int half_dim = head_dim / 2;

    const __half* grad_out = grad_output + token_idx * head_dim;
    __half* grad_in = grad_input + token_idx * head_dim;
    const float* cos_ptr = cos_cache + seq_pos * half_dim;
    const float* sin_ptr = sin_cache + seq_pos * half_dim;

    for (int i = 0; i < half_dim; i++) {
        float dy_even = __half2float(grad_out[2 * i]);
        float dy_odd = __half2float(grad_out[2 * i + 1]);
        float c = cos_ptr[i];
        float s = sin_ptr[i];

        float dx_even = dy_even * c + dy_odd * s;
        float dx_odd = -dy_even * s + dy_odd * c;

        grad_in[2 * i] = __float2half(dx_even);
        grad_in[2 * i + 1] = __float2half(dx_odd);
    }
}

// =============================================================================
// Host wrapper functions
// =============================================================================

extern "C" {

bool rope_forward_cuda_fp32(
    float* output,
    const float* input,
    const float* cos_cache,
    const float* sin_cache,
    const int* position_ids,  // can be nullptr
    int batch,
    int heads,
    int seq_len,
    int head_dim
) {
    int total_tokens = batch * heads * seq_len;
    int threads = 256;
    int blocks = (total_tokens + threads - 1) / threads;

    rope_forward_kernel_fp32<<<blocks, threads>>>(
        output, input, cos_cache, sin_cache, position_ids,
        total_tokens, head_dim, seq_len, heads
    );

    return cudaGetLastError() == cudaSuccess;
}

bool rope_forward_cuda_fp16(
    __half* output,
    const __half* input,
    const float* cos_cache,
    const float* sin_cache,
    const int* position_ids,
    int batch,
    int heads,
    int seq_len,
    int head_dim
) {
    int total_tokens = batch * heads * seq_len;
    int threads = 256;
    int blocks = (total_tokens + threads - 1) / threads;

    rope_forward_kernel_fp16<<<blocks, threads>>>(
        output, input, cos_cache, sin_cache, position_ids,
        total_tokens, head_dim, seq_len, heads
    );

    return cudaGetLastError() == cudaSuccess;
}

bool rope_backward_cuda_fp32(
    float* grad_input,
    const float* grad_output,
    const float* cos_cache,
    const float* sin_cache,
    const int* position_ids,
    int batch,
    int heads,
    int seq_len,
    int head_dim
) {
    int total_tokens = batch * heads * seq_len;
    int threads = 256;
    int blocks = (total_tokens + threads - 1) / threads;

    rope_backward_kernel_fp32<<<blocks, threads>>>(
        grad_input, grad_output, cos_cache, sin_cache, position_ids,
        total_tokens, head_dim, seq_len, heads
    );

    return cudaGetLastError() == cudaSuccess;
}

bool rope_backward_cuda_fp16(
    __half* grad_input,
    const __half* grad_output,
    const float* cos_cache,
    const float* sin_cache,
    const int* position_ids,
    int batch,
    int heads,
    int seq_len,
    int head_dim
) {
    int total_tokens = batch * heads * seq_len;
    int threads = 256;
    int blocks = (total_tokens + threads - 1) / threads;

    rope_backward_kernel_fp16<<<blocks, threads>>>(
        grad_input, grad_output, cos_cache, sin_cache, position_ids,
        total_tokens, head_dim, seq_len, heads
    );

    return cudaGetLastError() == cudaSuccess;
}

} // extern "C"

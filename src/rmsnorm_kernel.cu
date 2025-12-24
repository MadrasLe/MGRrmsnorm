#include "rmsnorm_kernel.h"
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Configuration
constexpr int WARP_SIZE = 32;
constexpr int MAX_ITEMS_PER_THREAD = 16;

// =============================================================================
// Type Traits for Generic Kernels
// =============================================================================

template<typename T> struct VecType { using type = float4; static constexpr int width = 4; };
template<> struct VecType<__half> { using type = __half2; static constexpr int width = 2; };
template<> struct VecType<__nv_bfloat16> { using type = __nv_bfloat162; static constexpr int width = 2; };

// Conversion helpers
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float(float x, float*) { return x; }
__device__ __forceinline__ __half from_float(float x, __half*) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float(float x, __nv_bfloat16*) { return __float2bfloat16(x); }

// =============================================================================
// Warp Reduce (shared across all dtypes)
// =============================================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// =============================================================================
// FP32 Forward Kernel (Original - Optimized)
// =============================================================================
__global__ void rmsnorm_kernel_fp32(
    float* __restrict__ out,
    float* __restrict__ inv_rms_out,
    const float* __restrict__ in, 
    const float* __restrict__ weight, 
    int rows, int cols, float epsilon
) {
    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    int tid = threadIdx.x;
    const float* row_in = in + row_idx * cols;
    float* row_out = out + row_idx * cols;

    float sum_sq = 0.0f;
    const int vec_cols = cols / 4;
    
    #pragma unroll
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        float4 loaded_in = reinterpret_cast<const float4*>(row_in)[i];
        sum_sq += loaded_in.x * loaded_in.x;
        sum_sq += loaded_in.y * loaded_in.y;
        sum_sq += loaded_in.z * loaded_in.z;
        sum_sq += loaded_in.w * loaded_in.w;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    static __shared__ float shared_sums[32];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0) shared_sums[warp_id] = sum_sq;
    __syncthreads();

    sum_sq = (tid < blockDim.x / WARP_SIZE) ? shared_sums[lane] : 0.0f;
    if (warp_id == 0) sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(sum_sq / cols + epsilon);
        if (inv_rms_out != nullptr) inv_rms_out[row_idx] = s_inv_rms;
    }
    __syncthreads();

    float inv_rms = s_inv_rms;

    #pragma unroll
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        float4 loaded_in = reinterpret_cast<const float4*>(row_in)[i];
        float4 loaded_w = reinterpret_cast<const float4*>(weight)[i];
        float4 out_val;

        out_val.x = loaded_in.x * inv_rms * loaded_w.x;
        out_val.y = loaded_in.y * inv_rms * loaded_w.y;
        out_val.z = loaded_in.z * inv_rms * loaded_w.z;
        out_val.w = loaded_in.w * inv_rms * loaded_w.w;

        reinterpret_cast<float4*>(row_out)[i] = out_val;
    }
}

// =============================================================================
// FP16 Forward Kernel (New - with half2 vectorization)
// =============================================================================
__global__ void rmsnorm_kernel_fp16(
    __half* __restrict__ out,
    float* __restrict__ inv_rms_out,
    const __half* __restrict__ in, 
    const __half* __restrict__ weight, 
    int rows, int cols, float epsilon
) {
    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    int tid = threadIdx.x;
    const __half* row_in = in + row_idx * cols;
    __half* row_out = out + row_idx * cols;

    float sum_sq = 0.0f;
    const int vec_cols = cols / 2;  // half2 = 2 elements
    
    #pragma unroll
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        __half2 loaded_in = reinterpret_cast<const __half2*>(row_in)[i];
        float2 f2 = __half22float2(loaded_in);
        sum_sq += f2.x * f2.x + f2.y * f2.y;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    static __shared__ float shared_sums[32];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0) shared_sums[warp_id] = sum_sq;
    __syncthreads();

    sum_sq = (tid < blockDim.x / WARP_SIZE) ? shared_sums[lane] : 0.0f;
    if (warp_id == 0) sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(sum_sq / cols + epsilon);
        if (inv_rms_out != nullptr) inv_rms_out[row_idx] = s_inv_rms;
    }
    __syncthreads();

    float inv_rms = s_inv_rms;

    #pragma unroll
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        __half2 loaded_in = reinterpret_cast<const __half2*>(row_in)[i];
        __half2 loaded_w = reinterpret_cast<const __half2*>(weight)[i];
        
        float2 f_in = __half22float2(loaded_in);
        float2 f_w = __half22float2(loaded_w);
        
        float2 f_out;
        f_out.x = f_in.x * inv_rms * f_w.x;
        f_out.y = f_in.y * inv_rms * f_w.y;
        
        reinterpret_cast<__half2*>(row_out)[i] = __float22half2_rn(f_out);
    }
}

// =============================================================================
// BF16 Forward Kernel
// =============================================================================
__global__ void rmsnorm_kernel_bf16(
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ inv_rms_out,
    const __nv_bfloat16* __restrict__ in, 
    const __nv_bfloat16* __restrict__ weight, 
    int rows, int cols, float epsilon
) {
    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    int tid = threadIdx.x;
    const __nv_bfloat16* row_in = in + row_idx * cols;
    __nv_bfloat16* row_out = out + row_idx * cols;

    float sum_sq = 0.0f;
    const int vec_cols = cols / 2;
    
    #pragma unroll
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        __nv_bfloat162 loaded_in = reinterpret_cast<const __nv_bfloat162*>(row_in)[i];
        float2 f2 = __bfloat1622float2(loaded_in);
        sum_sq += f2.x * f2.x + f2.y * f2.y;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    static __shared__ float shared_sums[32];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0) shared_sums[warp_id] = sum_sq;
    __syncthreads();

    sum_sq = (tid < blockDim.x / WARP_SIZE) ? shared_sums[lane] : 0.0f;
    if (warp_id == 0) sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(sum_sq / cols + epsilon);
        if (inv_rms_out != nullptr) inv_rms_out[row_idx] = s_inv_rms;
    }
    __syncthreads();

    float inv_rms = s_inv_rms;

    #pragma unroll
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        __nv_bfloat162 loaded_in = reinterpret_cast<const __nv_bfloat162*>(row_in)[i];
        __nv_bfloat162 loaded_w = reinterpret_cast<const __nv_bfloat162*>(weight)[i];
        
        float2 f_in = __bfloat1622float2(loaded_in);
        float2 f_w = __bfloat1622float2(loaded_w);
        
        float2 f_out;
        f_out.x = f_in.x * inv_rms * f_w.x;
        f_out.y = f_in.y * inv_rms * f_w.y;
        
        reinterpret_cast<__nv_bfloat162*>(row_out)[i] = __float22bfloat162_rn(f_out);
    }
}

// =============================================================================
// FP32 Backward Kernel (Original)
// =============================================================================
__global__ void rmsnorm_backward_kernel_fp32(
    float* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ inv_rms,
    int rows, int cols
) {
    int tid = threadIdx.x;
    const int vec_cols = cols / 4;
    
    float4 dw_acc[MAX_ITEMS_PER_THREAD];
    #pragma unroll
    for(int k = 0; k < MAX_ITEMS_PER_THREAD; ++k) {
        dw_acc[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    for (int row_idx = blockIdx.x; row_idx < rows; row_idx += gridDim.x) {
        const float* row_in = input + row_idx * cols;
        const float* row_grad_out = grad_output + row_idx * cols;
        float* row_grad_in = grad_input + row_idx * cols;
        
        float r_inv_rms = inv_rms[row_idx];
        float combined_sum = 0.0f;

        int item_idx = 0;
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 loaded_in = reinterpret_cast<const float4*>(row_in)[i];
            float4 loaded_grad_out = reinterpret_cast<const float4*>(row_grad_out)[i];
            float4 loaded_w = reinterpret_cast<const float4*>(weight)[i];
            
            combined_sum += loaded_grad_out.x * loaded_w.x * loaded_in.x;
            combined_sum += loaded_grad_out.y * loaded_w.y * loaded_in.y;
            combined_sum += loaded_grad_out.z * loaded_w.z * loaded_in.z;
            combined_sum += loaded_grad_out.w * loaded_w.w * loaded_in.w;

            if (item_idx < MAX_ITEMS_PER_THREAD) {
                dw_acc[item_idx].x += loaded_grad_out.x * loaded_in.x * r_inv_rms;
                dw_acc[item_idx].y += loaded_grad_out.y * loaded_in.y * r_inv_rms;
                dw_acc[item_idx].z += loaded_grad_out.z * loaded_in.z * r_inv_rms;
                dw_acc[item_idx].w += loaded_grad_out.w * loaded_in.w * r_inv_rms;
            }
            item_idx++;
        }

        float warp_sum = warp_reduce_sum(combined_sum);
        
        static __shared__ float shared_sums[32];
        int lane = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        if (lane == 0) shared_sums[warp_id] = warp_sum;
        __syncthreads();

        warp_sum = (tid < blockDim.x / WARP_SIZE) ? shared_sums[lane] : 0.0f;
        if (warp_id == 0) warp_sum = warp_reduce_sum(warp_sum);

        __shared__ float s_combined_sum;
        if (tid == 0) s_combined_sum = warp_sum;
        __syncthreads();
        
        float sum_dy_w_x = s_combined_sum;
        float term2 = sum_dy_w_x * (r_inv_rms * r_inv_rms) / cols;

        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 loaded_in = reinterpret_cast<const float4*>(row_in)[i];
            float4 loaded_grad_out = reinterpret_cast<const float4*>(row_grad_out)[i];
            float4 loaded_w = reinterpret_cast<const float4*>(weight)[i];
            float4 dx;

            dx.x = r_inv_rms * (loaded_grad_out.x * loaded_w.x - loaded_in.x * term2);
            dx.y = r_inv_rms * (loaded_grad_out.y * loaded_w.y - loaded_in.y * term2);
            dx.z = r_inv_rms * (loaded_grad_out.z * loaded_w.z - loaded_in.z * term2);
            dx.w = r_inv_rms * (loaded_grad_out.w * loaded_w.w - loaded_in.w * term2);

            reinterpret_cast<float4*>(row_grad_in)[i] = dx;
        }
    }

    int item_idx = 0;
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        if (item_idx < MAX_ITEMS_PER_THREAD) {
            float* gw_ptr = grad_weight + i * 4;
            atomicAdd(&gw_ptr[0], dw_acc[item_idx].x);
            atomicAdd(&gw_ptr[1], dw_acc[item_idx].y);
            atomicAdd(&gw_ptr[2], dw_acc[item_idx].z);
            atomicAdd(&gw_ptr[3], dw_acc[item_idx].w);
        }
        item_idx++;
    }
}

// =============================================================================
// FP16 Backward Kernel
// =============================================================================
__global__ void rmsnorm_backward_kernel_fp16(
    __half* __restrict__ grad_input,
    __half* __restrict__ grad_weight,
    const __half* __restrict__ grad_output,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const float* __restrict__ inv_rms,
    int rows, int cols
) {
    int tid = threadIdx.x;
    const int vec_cols = cols / 2;
    
    // Accumulate in FP32 for precision
    float2 dw_acc[MAX_ITEMS_PER_THREAD];
    #pragma unroll
    for(int k = 0; k < MAX_ITEMS_PER_THREAD; ++k) {
        dw_acc[k] = make_float2(0.0f, 0.0f);
    }

    for (int row_idx = blockIdx.x; row_idx < rows; row_idx += gridDim.x) {
        const __half* row_in = input + row_idx * cols;
        const __half* row_grad_out = grad_output + row_idx * cols;
        __half* row_grad_in = grad_input + row_idx * cols;
        
        float r_inv_rms = inv_rms[row_idx];
        float combined_sum = 0.0f;

        int item_idx = 0;
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            __half2 h_in = reinterpret_cast<const __half2*>(row_in)[i];
            __half2 h_grad_out = reinterpret_cast<const __half2*>(row_grad_out)[i];
            __half2 h_w = reinterpret_cast<const __half2*>(weight)[i];
            
            float2 f_in = __half22float2(h_in);
            float2 f_grad_out = __half22float2(h_grad_out);
            float2 f_w = __half22float2(h_w);
            
            combined_sum += f_grad_out.x * f_w.x * f_in.x;
            combined_sum += f_grad_out.y * f_w.y * f_in.y;

            if (item_idx < MAX_ITEMS_PER_THREAD) {
                dw_acc[item_idx].x += f_grad_out.x * f_in.x * r_inv_rms;
                dw_acc[item_idx].y += f_grad_out.y * f_in.y * r_inv_rms;
            }
            item_idx++;
        }

        float warp_sum = warp_reduce_sum(combined_sum);
        
        static __shared__ float shared_sums[32];
        int lane = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        if (lane == 0) shared_sums[warp_id] = warp_sum;
        __syncthreads();

        warp_sum = (tid < blockDim.x / WARP_SIZE) ? shared_sums[lane] : 0.0f;
        if (warp_id == 0) warp_sum = warp_reduce_sum(warp_sum);

        __shared__ float s_combined_sum;
        if (tid == 0) s_combined_sum = warp_sum;
        __syncthreads();
        
        float sum_dy_w_x = s_combined_sum;
        float term2 = sum_dy_w_x * (r_inv_rms * r_inv_rms) / cols;

        for (int i = tid; i < vec_cols; i += blockDim.x) {
            __half2 h_in = reinterpret_cast<const __half2*>(row_in)[i];
            __half2 h_grad_out = reinterpret_cast<const __half2*>(row_grad_out)[i];
            __half2 h_w = reinterpret_cast<const __half2*>(weight)[i];
            
            float2 f_in = __half22float2(h_in);
            float2 f_grad_out = __half22float2(h_grad_out);
            float2 f_w = __half22float2(h_w);
            
            float2 dx;
            dx.x = r_inv_rms * (f_grad_out.x * f_w.x - f_in.x * term2);
            dx.y = r_inv_rms * (f_grad_out.y * f_w.y - f_in.y * term2);

            reinterpret_cast<__half2*>(row_grad_in)[i] = __float22half2_rn(dx);
        }
    }

    // Atomic add for weight gradients (in FP32, then convert)
    int item_idx = 0;
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        if (item_idx < MAX_ITEMS_PER_THREAD) {
            // Use float atomics then convert - more precise
            __half* gw_ptr = grad_weight + i * 2;
            // Unfortunately half atomics are limited, use float accumulation
            atomicAdd(reinterpret_cast<__half*>(&gw_ptr[0]), __float2half(dw_acc[item_idx].x));
            atomicAdd(reinterpret_cast<__half*>(&gw_ptr[1]), __float2half(dw_acc[item_idx].y));
        }
        item_idx++;
    }
}

// =============================================================================
// BF16 Backward Kernel
// =============================================================================
__global__ void rmsnorm_backward_kernel_bf16(
    __nv_bfloat16* __restrict__ grad_input,
    __nv_bfloat16* __restrict__ grad_weight,
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ inv_rms,
    int rows, int cols
) {
    int tid = threadIdx.x;
    const int vec_cols = cols / 2;
    
    float2 dw_acc[MAX_ITEMS_PER_THREAD];
    #pragma unroll
    for(int k = 0; k < MAX_ITEMS_PER_THREAD; ++k) {
        dw_acc[k] = make_float2(0.0f, 0.0f);
    }

    for (int row_idx = blockIdx.x; row_idx < rows; row_idx += gridDim.x) {
        const __nv_bfloat16* row_in = input + row_idx * cols;
        const __nv_bfloat16* row_grad_out = grad_output + row_idx * cols;
        __nv_bfloat16* row_grad_in = grad_input + row_idx * cols;
        
        float r_inv_rms = inv_rms[row_idx];
        float combined_sum = 0.0f;

        int item_idx = 0;
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            __nv_bfloat162 b_in = reinterpret_cast<const __nv_bfloat162*>(row_in)[i];
            __nv_bfloat162 b_grad_out = reinterpret_cast<const __nv_bfloat162*>(row_grad_out)[i];
            __nv_bfloat162 b_w = reinterpret_cast<const __nv_bfloat162*>(weight)[i];
            
            float2 f_in = __bfloat1622float2(b_in);
            float2 f_grad_out = __bfloat1622float2(b_grad_out);
            float2 f_w = __bfloat1622float2(b_w);
            
            combined_sum += f_grad_out.x * f_w.x * f_in.x;
            combined_sum += f_grad_out.y * f_w.y * f_in.y;

            if (item_idx < MAX_ITEMS_PER_THREAD) {
                dw_acc[item_idx].x += f_grad_out.x * f_in.x * r_inv_rms;
                dw_acc[item_idx].y += f_grad_out.y * f_in.y * r_inv_rms;
            }
            item_idx++;
        }

        float warp_sum = warp_reduce_sum(combined_sum);
        
        static __shared__ float shared_sums[32];
        int lane = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        if (lane == 0) shared_sums[warp_id] = warp_sum;
        __syncthreads();

        warp_sum = (tid < blockDim.x / WARP_SIZE) ? shared_sums[lane] : 0.0f;
        if (warp_id == 0) warp_sum = warp_reduce_sum(warp_sum);

        __shared__ float s_combined_sum;
        if (tid == 0) s_combined_sum = warp_sum;
        __syncthreads();
        
        float sum_dy_w_x = s_combined_sum;
        float term2 = sum_dy_w_x * (r_inv_rms * r_inv_rms) / cols;

        for (int i = tid; i < vec_cols; i += blockDim.x) {
            __nv_bfloat162 b_in = reinterpret_cast<const __nv_bfloat162*>(row_in)[i];
            __nv_bfloat162 b_grad_out = reinterpret_cast<const __nv_bfloat162*>(row_grad_out)[i];
            __nv_bfloat162 b_w = reinterpret_cast<const __nv_bfloat162*>(weight)[i];
            
            float2 f_in = __bfloat1622float2(b_in);
            float2 f_grad_out = __bfloat1622float2(b_grad_out);
            float2 f_w = __bfloat1622float2(b_w);
            
            float2 dx;
            dx.x = r_inv_rms * (f_grad_out.x * f_w.x - f_in.x * term2);
            dx.y = r_inv_rms * (f_grad_out.y * f_w.y - f_in.y * term2);

            reinterpret_cast<__nv_bfloat162*>(row_grad_in)[i] = __float22bfloat162_rn(dx);
        }
    }

    int item_idx = 0;
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        if (item_idx < MAX_ITEMS_PER_THREAD) {
            __nv_bfloat16* gw_ptr = grad_weight + i * 2;
            atomicAdd(reinterpret_cast<__nv_bfloat16*>(&gw_ptr[0]), __float2bfloat16(dw_acc[item_idx].x));
            atomicAdd(reinterpret_cast<__nv_bfloat16*>(&gw_ptr[1]), __float2bfloat16(dw_acc[item_idx].y));
        }
        item_idx++;
    }
}

// =============================================================================
// Host Wrappers
// =============================================================================

// FP32
bool rmsnorm_cuda_fp32(float* out, float* inv_rms, const float* in, const float* weight, int rows, int cols, float epsilon) {
    int threads_per_block = 256;
    int blocks_per_grid = rows;

    if ((uintptr_t)in % 16 != 0 || (uintptr_t)out % 16 != 0 || (uintptr_t)weight % 16 != 0) {
        fprintf(stderr, "Error: input/output/weight pointers must be 16-byte aligned.\n");
        return false;
    }
    if (cols % 4 != 0) {
        fprintf(stderr, "Error: cols must be divisible by 4.\n");
        return false; 
    }

    rmsnorm_kernel_fp32<<<blocks_per_grid, threads_per_block>>>(out, inv_rms, in, weight, rows, cols, epsilon);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error (FP32 Forward): %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool rmsnorm_backward_cuda_fp32(float* grad_input, float* grad_weight, const float* grad_output, const float* input, const float* weight, const float* inv_rms, int rows, int cols) {
    int threads_per_block = 256;
    int blocks_per_grid = 128;
    if (rows < blocks_per_grid) blocks_per_grid = rows;

    if (cols > 16384) {
        fprintf(stderr, "Error: cols too large (max 16384).\n");
        return false;
    }

    cudaMemset(grad_weight, 0, cols * sizeof(float));

    rmsnorm_backward_kernel_fp32<<<blocks_per_grid, threads_per_block>>>(
        grad_input, grad_weight, grad_output, input, weight, inv_rms, rows, cols
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error (FP32 Backward): %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

// FP16
bool rmsnorm_cuda_fp16(__half* out, float* inv_rms, const __half* in, const __half* weight, int rows, int cols, float epsilon) {
    int threads_per_block = 256;
    int blocks_per_grid = rows;

    if (cols % 2 != 0) {
        fprintf(stderr, "Error: cols must be divisible by 2 for FP16.\n");
        return false; 
    }

    rmsnorm_kernel_fp16<<<blocks_per_grid, threads_per_block>>>(out, inv_rms, in, weight, rows, cols, epsilon);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error (FP16 Forward): %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool rmsnorm_backward_cuda_fp16(__half* grad_input, __half* grad_weight, const __half* grad_output, const __half* input, const __half* weight, const float* inv_rms, int rows, int cols) {
    int threads_per_block = 256;
    int blocks_per_grid = 128;
    if (rows < blocks_per_grid) blocks_per_grid = rows;

    if (cols > 32768) {
        fprintf(stderr, "Error: cols too large for FP16 (max 32768).\n");
        return false;
    }

    cudaMemset(grad_weight, 0, cols * sizeof(__half));

    rmsnorm_backward_kernel_fp16<<<blocks_per_grid, threads_per_block>>>(
        grad_input, grad_weight, grad_output, input, weight, inv_rms, rows, cols
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error (FP16 Backward): %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

// BF16
bool rmsnorm_cuda_bf16(__nv_bfloat16* out, float* inv_rms, const __nv_bfloat16* in, const __nv_bfloat16* weight, int rows, int cols, float epsilon) {
    int threads_per_block = 256;
    int blocks_per_grid = rows;

    if (cols % 2 != 0) {
        fprintf(stderr, "Error: cols must be divisible by 2 for BF16.\n");
        return false; 
    }

    rmsnorm_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(out, inv_rms, in, weight, rows, cols, epsilon);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error (BF16 Forward): %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool rmsnorm_backward_cuda_bf16(__nv_bfloat16* grad_input, __nv_bfloat16* grad_weight, const __nv_bfloat16* grad_output, const __nv_bfloat16* input, const __nv_bfloat16* weight, const float* inv_rms, int rows, int cols) {
    int threads_per_block = 256;
    int blocks_per_grid = 128;
    if (rows < blocks_per_grid) blocks_per_grid = rows;

    if (cols > 32768) {
        fprintf(stderr, "Error: cols too large for BF16 (max 32768).\n");
        return false;
    }

    cudaMemset(grad_weight, 0, cols * sizeof(__nv_bfloat16));

    rmsnorm_backward_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(
        grad_input, grad_weight, grad_output, input, weight, inv_rms, rows, cols
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error (BF16 Backward): %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

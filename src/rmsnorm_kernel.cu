#include "rmsnorm_kernel.h"
#include <cstdio>
#include <cstdint>

// Configuration
constexpr int WARP_SIZE = 32;
// Maximum number of float4 items a thread might process. 
// For cols=4096, blockDim=256 -> vec_cols=1024. 1024/256 = 4 items per thread.
constexpr int MAX_ITEMS_PER_THREAD = 16; 

// --------------------------------------------------------------------------
// CUDA Helper: Warp Reduce
// --------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// --------------------------------------------------------------------------
// CUDA Kernel: Forward
// --------------------------------------------------------------------------
__global__ void rmsnorm_kernel(
    float* __restrict__ out,
    float* __restrict__ inv_rms_out, // Added output for inv_rms
    const float* __restrict__ in, 
    const float* __restrict__ weight, 
    int rows, 
    int cols, 
    float epsilon
) {
    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    int tid = threadIdx.x;
    
    const float* row_in = in + row_idx * cols;
    float* row_out = out + row_idx * cols;

    float sum_sq = 0.0f;

    // 1. Calculate Sum of Squares
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

    if (lane == 0) {
        shared_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    sum_sq = (tid < blockDim.x / WARP_SIZE) ? shared_sums[lane] : 0.0f;
    
    if (warp_id == 0) {
        sum_sq = warp_reduce_sum(sum_sq);
    }

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(sum_sq / cols + epsilon);
        // Save inv_rms to global memory if pointer is provided
        if (inv_rms_out != nullptr) {
            inv_rms_out[row_idx] = s_inv_rms;
        }
    }
    __syncthreads();

    float inv_rms = s_inv_rms;

    // 2. Normalize and Scale
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

// --------------------------------------------------------------------------
// CUDA Kernel: Backward (Optimized with Grid-Stride Loop and Register Accumulation)
// --------------------------------------------------------------------------
__global__ void rmsnorm_backward_kernel(
    float* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ inv_rms,
    int rows,
    int cols
) {
    int tid = threadIdx.x;
    const int vec_cols = cols / 4;
    
    // Accumulator for weight gradients. 
    // Each thread handles specific columns, so we can accumulate across rows locally.
    // Note: MAX_ITEMS_PER_THREAD determines the limit on cols/blockDim.x
    float4 dw_acc[MAX_ITEMS_PER_THREAD];
    
    // Initialize accumulators
    #pragma unroll
    for(int k=0; k < MAX_ITEMS_PER_THREAD; ++k) {
        dw_acc[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Grid-Stride Loop over Rows
    for (int row_idx = blockIdx.x; row_idx < rows; row_idx += gridDim.x) {
        const float* row_in = input + row_idx * cols;
        const float* row_grad_out = grad_output + row_idx * cols;
        float* row_grad_in = grad_input + row_idx * cols;
        
        float r_inv_rms = inv_rms[row_idx];
        float combined_sum = 0.0f;

        // 1. Compute dot product for the current row
        int item_idx = 0;
        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 loaded_in = reinterpret_cast<const float4*>(row_in)[i];
            float4 loaded_grad_out = reinterpret_cast<const float4*>(row_grad_out)[i];
            float4 loaded_w = reinterpret_cast<const float4*>(weight)[i];
            
            // Term: dy * w * x
            combined_sum += loaded_grad_out.x * loaded_w.x * loaded_in.x;
            combined_sum += loaded_grad_out.y * loaded_w.y * loaded_in.y;
            combined_sum += loaded_grad_out.z * loaded_w.z * loaded_in.z;
            combined_sum += loaded_grad_out.w * loaded_w.w * loaded_in.w;

            // Accumulate gradient for weight locally: sum(dy * x * inv_rms)
            float4 dw_curr;
            dw_curr.x = loaded_grad_out.x * loaded_in.x * r_inv_rms;
            dw_curr.y = loaded_grad_out.y * loaded_in.y * r_inv_rms;
            dw_curr.z = loaded_grad_out.z * loaded_in.z * r_inv_rms;
            dw_curr.w = loaded_grad_out.w * loaded_in.w * r_inv_rms;

            if (item_idx < MAX_ITEMS_PER_THREAD) {
                dw_acc[item_idx].x += dw_curr.x;
                dw_acc[item_idx].y += dw_curr.y;
                dw_acc[item_idx].z += dw_curr.z;
                dw_acc[item_idx].w += dw_curr.w;
            }
            item_idx++;
        }

        // Reduction for combined_sum (standard warp reduce)
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

        // 2. Compute grad_input
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

    // Final: Atomic Add of Accumulated Weight Gradients
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

// --------------------------------------------------------------------------
// Host Wrappers
// --------------------------------------------------------------------------
bool rmsnorm_cuda(float* out, float* inv_rms, const float* in, const float* weight, int rows, int cols, float epsilon) {
    int threads_per_block = 256;
    int blocks_per_grid = rows;

    // Check alignment
    if ((uintptr_t)in % 16 != 0 || (uintptr_t)out % 16 != 0 || (uintptr_t)weight % 16 != 0) {
        fprintf(stderr, "Error: input/output/weight pointers must be 16-byte aligned.\n");
        return false;
    }
    if (cols % 4 != 0) {
        fprintf(stderr, "Error: cols must be divisible by 4.\n");
        return false; 
    }

    rmsnorm_kernel<<<blocks_per_grid, threads_per_block>>>(out, inv_rms, in, weight, rows, cols, epsilon);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error (Forward): %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool rmsnorm_backward_cuda(
    float* grad_input,
    float* grad_weight,
    const float* grad_output,
    const float* input,
    const float* weight,
    const float* inv_rms,
    int rows,
    int cols
) {
    int threads_per_block = 256;
    
    // Reduce grid size to limit atomic contention.
    // For example, use 128 blocks (enough to saturate most GPUs but keep atomics low).
    // Or scale with rows but clamp.
    int blocks_per_grid = 128; 
    if (rows < blocks_per_grid) blocks_per_grid = rows;

    if ((uintptr_t)grad_input % 16 != 0 || (uintptr_t)grad_output % 16 != 0 || 
        (uintptr_t)input % 16 != 0 || (uintptr_t)weight % 16 != 0) {
        fprintf(stderr, "Error: pointers must be 16-byte aligned for backward pass.\n");
        return false;
    }
    
    // Check if MAX_ITEMS_PER_THREAD is sufficient
    // Threads per block = 256. items per thread = vec_cols / 256.
    // vec_cols = cols / 4.
    // items = (cols / 4) / 256 = cols / 1024.
    // If cols > 1024 * 16 (16384), we overflow registers.
    // Add a check.
    if (cols > 16384) {
        fprintf(stderr, "Error: cols too large for current register configuration (max 16384).\n");
        return false;
    }

    // Initialize grad_weight to 0 because we use atomicAdd
    cudaMemset(grad_weight, 0, cols * sizeof(float));

    rmsnorm_backward_kernel<<<blocks_per_grid, threads_per_block>>>(
        grad_input, grad_weight, grad_output, input, weight, inv_rms, rows, cols
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error (Backward): %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

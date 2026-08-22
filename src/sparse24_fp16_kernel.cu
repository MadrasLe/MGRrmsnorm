#include "sparse24_fp16_kernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace {

constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 32;
constexpr int kSharedChunkK = 512;
constexpr int kSharedValueWords = kSharedChunkK / 4;
// 128 data words + four-word padding.  A row stride of 4 modulo 32 maps
// the eight MMA row groups onto distinct shared-memory bank quartets.
constexpr int kSharedValueStrideWords = kSharedValueWords + 4;
constexpr int kSharedMmaTiles = kSharedChunkK / kMmaK;

__device__ __forceinline__ uint32_t load_half2_or_zero(
    const __half* pointer,
    bool valid
) {
    if (!valid) {
        return 0u;
    }
    return __ldg(reinterpret_cast<const uint32_t*>(pointer));
}

__device__ __forceinline__ void mma_sp_f16_f32(
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1,
    uint32_t b2,
    uint32_t b3,
    uint32_t sparse_metadata,
    float& d0,
    float& d1,
    float& d2,
    float& d3
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
#if ((__CUDACC_VER_MAJOR__ > 12) || \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(sparse_metadata)
    );
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(sparse_metadata)
    );
#endif
#endif
}

template <int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock * 32)
void sparse24_fp16_mma_kernel(
    const __half* __restrict__ input,       // [M, K]
    const __half* __restrict__ values,      // [N, K / 2]
    const uint8_t* __restrict__ metadata,   // [N, K / 8], PTX nibbles
    const __half* __restrict__ bias,        // [N] or nullptr
    __half* __restrict__ output,            // [M, N]
    int m_rows,
    int n_rows,
    int k_cols
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
    const int thread_id = static_cast<int>(threadIdx.x);
    const int warp_id = thread_id >> 5;
    const int lane_id = thread_id & 31;
    const int group_id = lane_id >> 2;
    const int thread_in_group = lane_id & 3;
    const int n_tile = static_cast<int>(blockIdx.x) * kMmaM;
    const int m_tile = static_cast<int>(blockIdx.y) * (WarpsPerBlock * kMmaN);
    const int m_warp = m_tile + warp_id * kMmaN;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (int k_tile = 0; k_tile < k_cols; k_tile += kMmaK) {
        // Sparse A fragment for mma.sp.m16n8k32.  Registers 0/2 belong to
        // output row group_id; registers 1/3 to group_id + 8.  Each register
        // contains the two retained FP16 values from one logical quartet.
        //
        // v2 deliberately reads these fragments through the read-only cache.
        // The v1 shared-memory staging serialized every 32-wide K slice behind
        // two CTA barriers (up to 560 barriers for Qwen's down projection),
        // which reduced effective bandwidth to single-digit GB/s on L4.  The
        // direct warp loads are naturally coalesced; multiple M-warps request
        // identical cache lines and therefore reuse them through L1/L2 without
        // a software barrier.
        const int packed_col0 = thread_in_group * 2;
        const int packed_col1 = packed_col0 + 8;
        const int values_stride = k_cols / 2;
        const int values_col = k_tile / 2;
        const __half* values_row0 =
            values + (n_tile + group_id) * values_stride + values_col;
        const __half* values_row1 =
            values + (n_tile + group_id + 8) * values_stride + values_col;
        const uint32_t a0 = __ldg(reinterpret_cast<const uint32_t*>(
            values_row0 + packed_col0));
        const uint32_t a1 = __ldg(reinterpret_cast<const uint32_t*>(
            values_row1 + packed_col0));
        const uint32_t a2 = __ldg(reinterpret_cast<const uint32_t*>(
            values_row0 + packed_col1));
        const uint32_t a3 = __ldg(reinterpret_cast<const uint32_t*>(
            values_row1 + packed_col1));

        // Dense B is input.T logically.  Loading from the original row-major
        // [M,K] tensor makes each b register a naturally aligned half2.
        const int m_fragment = m_warp + group_id;
        const bool valid_m = m_fragment < m_rows;
        const __half* input_row = input + m_fragment * k_cols + k_tile;
        const int dense_col = thread_in_group * 2;
        const uint32_t b0 = load_half2_or_zero(input_row + dense_col, valid_m);
        const uint32_t b1 = load_half2_or_zero(input_row + dense_col + 8, valid_m);
        const uint32_t b2 = load_half2_or_zero(input_row + dense_col + 16, valid_m);
        const uint32_t b3 = load_half2_or_zero(input_row + dense_col + 24, valid_m);

        // For selector 0, lanes 4g and 4g+1 contribute metadata for the first
        // and second 16-wide K halves, respectively.  In each lane's uint32,
        // bits 0..15 describe output row g and bits 16..31 row g+8.  This
        // cross-row packing is required by Figure 122 of the PTX ISA; loading
        // one complete metadata row per lane would silently select the wrong
        // values even though every nibble is individually valid.
        uint32_t sparse_metadata = 0u;
        if (thread_in_group < 2) {
            const int metadata_stride = k_cols / 8;
            const int metadata_col = (k_tile / 8) + thread_in_group * 2;
            const uint8_t* metadata_row0 =
                metadata + (n_tile + group_id) * metadata_stride + metadata_col;
            const uint8_t* metadata_row1 =
                metadata + (n_tile + group_id + 8) * metadata_stride + metadata_col;
            const uint16_t low_rows =
                __ldg(reinterpret_cast<const uint16_t*>(metadata_row0));
            const uint16_t high_rows =
                __ldg(reinterpret_cast<const uint16_t*>(metadata_row1));
            sparse_metadata =
                static_cast<uint32_t>(low_rows) |
                (static_cast<uint32_t>(high_rows) << 16);
        }

        mma_sp_f16_f32(
            a0, a1, a2, a3,
            b0, b1, b2, b3,
            sparse_metadata,
            d0, d1, d2, d3
        );
    }

    const int output_n0 = n_tile + group_id;
    const int output_n1 = output_n0 + 8;
    const int output_m0 = m_warp + thread_in_group * 2;
    const int output_m1 = output_m0 + 1;
    if (bias != nullptr) {
        const float bias0 = __half2float(bias[output_n0]);
        const float bias1 = __half2float(bias[output_n1]);
        d0 += bias0;
        d1 += bias0;
        d2 += bias1;
        d3 += bias1;
    }
    if (output_m0 < m_rows) {
        output[output_m0 * n_rows + output_n0] = __float2half_rn(d0);
        output[output_m0 * n_rows + output_n1] = __float2half_rn(d2);
    }
    if (output_m1 < m_rows) {
        output[output_m1 * n_rows + output_n0] = __float2half_rn(d1);
        output[output_m1 * n_rows + output_n1] = __float2half_rn(d3);
    }
#endif
}

template <int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock * 32)
void sparse24_fp16_mma_shared_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ values,
    const uint8_t* __restrict__ metadata,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int m_rows,
    int n_rows,
    int k_cols
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
    const int thread_id = static_cast<int>(threadIdx.x);
    const int warp_id = thread_id >> 5;
    const int lane_id = thread_id & 31;
    const int group_id = lane_id >> 2;
    const int thread_in_group = lane_id & 3;
    const int n_tile = static_cast<int>(blockIdx.x) * kMmaM;
    const int m_tile = static_cast<int>(blockIdx.y) * (WarpsPerBlock * kMmaN);
    const int m_warp = m_tile + warp_id * kMmaN;
    constexpr int kThreads = WarpsPerBlock * 32;

    __shared__ __align__(16)
        uint32_t shared_values[kMmaM * kSharedValueStrideWords];
    // [K/32 tile][metadata lane].  For a fixed tile the 16 contributing
    // lanes hit 16 consecutive banks; all M-warps broadcast the same entries.
    __shared__ __align__(16)
        uint32_t shared_metadata[kSharedMmaTiles * kMmaM];

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const int global_value_stride_words = k_cols / 4;
    const int global_metadata_stride = k_cols / 8;

    for (int k_chunk = 0; k_chunk < k_cols; k_chunk += kSharedChunkK) {
        const int remaining_k = k_cols - k_chunk;
        const int chunk_k =
            remaining_k < kSharedChunkK ? remaining_k : kSharedChunkK;
        const int chunk_value_words = chunk_k / 4;
        const int value_word_count = kMmaM * chunk_value_words;
        for (int index = thread_id; index < value_word_count; index += kThreads) {
            const int local_n = index / chunk_value_words;
            const int local_word = index - local_n * chunk_value_words;
            const uint32_t* global_row = reinterpret_cast<const uint32_t*>(values) +
                (n_tile + local_n) * global_value_stride_words + k_chunk / 4;
            shared_values[local_n * kSharedValueStrideWords + local_word] =
                __ldg(global_row + local_word);
        }

        const int mma_tiles = chunk_k / kMmaK;
        const int metadata_entry_count = mma_tiles * kMmaM;
        for (int index = thread_id; index < metadata_entry_count; index += kThreads) {
            const int mma_tile = index / kMmaM;
            const int metadata_lane = index - mma_tile * kMmaM;
            const int local_group = metadata_lane >> 1;
            const int k_half = metadata_lane & 1;
            const int metadata_col =
                k_chunk / 8 + mma_tile * (kMmaK / 8) + k_half * 2;
            const uint8_t* row0 = metadata +
                (n_tile + local_group) * global_metadata_stride + metadata_col;
            const uint8_t* row1 = metadata +
                (n_tile + local_group + 8) * global_metadata_stride + metadata_col;
            const uint16_t low_rows =
                __ldg(reinterpret_cast<const uint16_t*>(row0));
            const uint16_t high_rows =
                __ldg(reinterpret_cast<const uint16_t*>(row1));
            shared_metadata[mma_tile * kMmaM + metadata_lane] =
                static_cast<uint32_t>(low_rows) |
                (static_cast<uint32_t>(high_rows) << 16);
        }
        __syncthreads();

        for (int mma_tile = 0; mma_tile < mma_tiles; ++mma_tile) {
            const int value_word = mma_tile * (kMmaK / 4) + thread_in_group;
            const int row0 = group_id;
            const int row1 = group_id + 8;
            const uint32_t a0 =
                shared_values[row0 * kSharedValueStrideWords + value_word];
            const uint32_t a1 =
                shared_values[row1 * kSharedValueStrideWords + value_word];
            const uint32_t a2 =
                shared_values[row0 * kSharedValueStrideWords + value_word + 4];
            const uint32_t a3 =
                shared_values[row1 * kSharedValueStrideWords + value_word + 4];

            const int k_tile = k_chunk + mma_tile * kMmaK;
            const int m_fragment = m_warp + group_id;
            const bool valid_m = m_fragment < m_rows;
            const __half* input_row = input + m_fragment * k_cols + k_tile;
            const int dense_col = thread_in_group * 2;
            const uint32_t b0 = load_half2_or_zero(input_row + dense_col, valid_m);
            const uint32_t b1 = load_half2_or_zero(input_row + dense_col + 8, valid_m);
            const uint32_t b2 = load_half2_or_zero(input_row + dense_col + 16, valid_m);
            const uint32_t b3 = load_half2_or_zero(input_row + dense_col + 24, valid_m);
            const uint32_t sparse_metadata =
                thread_in_group < 2
                ? shared_metadata[
                    mma_tile * kMmaM + group_id * 2 + thread_in_group]
                : 0u;

            mma_sp_f16_f32(
                a0, a1, a2, a3,
                b0, b1, b2, b3,
                sparse_metadata,
                d0, d1, d2, d3
            );
        }
        if (k_chunk + chunk_k < k_cols) {
            __syncthreads();
        }
    }

    const int output_n0 = n_tile + group_id;
    const int output_n1 = output_n0 + 8;
    const int output_m0 = m_warp + thread_in_group * 2;
    const int output_m1 = output_m0 + 1;
    if (bias != nullptr) {
        const float bias0 = __half2float(bias[output_n0]);
        const float bias1 = __half2float(bias[output_n1]);
        d0 += bias0;
        d1 += bias0;
        d2 += bias1;
        d3 += bias1;
    }
    if (output_m0 < m_rows) {
        output[output_m0 * n_rows + output_n0] = __float2half_rn(d0);
        output[output_m0 * n_rows + output_n1] = __float2half_rn(d2);
    }
    if (output_m1 < m_rows) {
        output[output_m1 * n_rows + output_n0] = __float2half_rn(d1);
        output[output_m1 * n_rows + output_n1] = __float2half_rn(d3);
    }
#endif
}

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor sparse24_fp16_linear_cuda(
    torch::Tensor input,
    torch::Tensor values,
    torch::Tensor metadata,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output
) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(values, "values");
    check_cuda_tensor(metadata, "metadata");
    TORCH_CHECK(input.dim() == 2, "input must be [M,K]");
    TORCH_CHECK(values.dim() == 2, "values must be [N,K/2]");
    TORCH_CHECK(metadata.dim() == 2, "metadata must be [N,K/8]");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16, "input must be FP16");
    TORCH_CHECK(values.scalar_type() == torch::kFloat16, "values must be FP16");
    TORCH_CHECK(metadata.scalar_type() == torch::kUInt8, "metadata must be uint8");
    TORCH_CHECK(input.device() == values.device(), "input and values must share a device");
    TORCH_CHECK(input.device() == metadata.device(), "input and metadata must share a device");

    const int64_t m_rows64 = input.size(0);
    const int64_t k_cols64 = input.size(1);
    const int64_t n_rows64 = values.size(0);
    TORCH_CHECK(m_rows64 > 0, "M must be positive");
    TORCH_CHECK(n_rows64 > 0 && n_rows64 % 64 == 0, "N must be a positive multiple of 64");
    TORCH_CHECK(k_cols64 > 0 && k_cols64 % 64 == 0, "K must be a positive multiple of 64");
    TORCH_CHECK(values.size(1) == k_cols64 / 2, "values has the wrong packed K dimension");
    TORCH_CHECK(
        metadata.size(0) == n_rows64 && metadata.size(1) == k_cols64 / 8,
        "metadata must have shape [N,K/8]"
    );
    TORCH_CHECK(
        m_rows64 <= INT32_MAX && n_rows64 <= INT32_MAX && k_cols64 <= INT32_MAX,
        "matrix dimensions exceed the standalone kernel's int32 addressing contract"
    );

    c10::cuda::CUDAGuard device_guard(input.device());
    static thread_local int cached_device = -1;
    static thread_local int cached_major = -1;
    static thread_local int cached_minor = -1;
    if (cached_device != input.get_device()) {
        cudaDeviceProp properties{};
        C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
        cached_device = input.get_device();
        cached_major = properties.major;
        cached_minor = properties.minor;
    }
    TORCH_CHECK(
        cached_major == 8,
        "standalone mma.sp FP16 kernel currently supports SM80-SM89; got sm_",
        cached_major,
        cached_minor
    );

    const __half* bias_ptr = nullptr;
    if (bias.has_value()) {
        check_cuda_tensor(bias.value(), "bias");
        TORCH_CHECK(bias.value().device() == input.device(), "bias must share input's device");
        TORCH_CHECK(bias.value().scalar_type() == torch::kFloat16, "bias must be FP16");
        TORCH_CHECK(bias.value().numel() == n_rows64, "bias must contain N elements");
        bias_ptr = reinterpret_cast<const __half*>(bias.value().data_ptr<at::Half>());
    }

    torch::Tensor result;
    if (output.has_value()) {
        check_cuda_tensor(output.value(), "output");
        TORCH_CHECK(output.value().device() == input.device(), "output must share input's device");
        TORCH_CHECK(output.value().scalar_type() == torch::kFloat16, "output must be FP16");
        TORCH_CHECK(
            output.value().dim() == 2 && output.value().size(0) == m_rows64 &&
                output.value().size(1) == n_rows64,
            "output must have shape [M,N]"
        );
        result = output.value();
    } else {
        result = torch::empty({m_rows64, n_rows64}, input.options());
    }

    const int m_rows = static_cast<int>(m_rows64);
    const int n_rows = static_cast<int>(n_rows64);
    const int k_cols = static_cast<int>(k_cols64);
    int warps = 8;
    if (m_rows <= 8) {
        warps = 1;
    } else if (m_rows <= 16) {
        warps = 2;
    } else if (m_rows <= 32) {
        warps = 4;
    }

    dim3 grid(
        static_cast<unsigned int>(n_rows / kMmaM),
        static_cast<unsigned int>((m_rows + warps * kMmaN - 1) / (warps * kMmaN))
    );
    const auto stream = at::cuda::getCurrentCUDAStream();
    const __half* input_ptr = reinterpret_cast<const __half*>(input.data_ptr<at::Half>());
    const __half* values_ptr = reinterpret_cast<const __half*>(values.data_ptr<at::Half>());
    const uint8_t* metadata_ptr = metadata.data_ptr<uint8_t>();
    __half* output_ptr = reinterpret_cast<__half*>(result.data_ptr<at::Half>());

    switch (warps) {
        case 1:
            sparse24_fp16_mma_kernel<1><<<grid, 32, 0, stream>>>(
                input_ptr, values_ptr, metadata_ptr, bias_ptr, output_ptr,
                m_rows, n_rows, k_cols);
            break;
        case 2:
            sparse24_fp16_mma_shared_kernel<2><<<grid, 64, 0, stream>>>(
                input_ptr, values_ptr, metadata_ptr, bias_ptr, output_ptr,
                m_rows, n_rows, k_cols);
            break;
        case 4:
            sparse24_fp16_mma_shared_kernel<4><<<grid, 128, 0, stream>>>(
                input_ptr, values_ptr, metadata_ptr, bias_ptr, output_ptr,
                m_rows, n_rows, k_cols);
            break;
        default:
            sparse24_fp16_mma_shared_kernel<8><<<grid, 256, 0, stream>>>(
                input_ptr, values_ptr, metadata_ptr, bias_ptr, output_ptr,
                m_rows, n_rows, k_cols);
            break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

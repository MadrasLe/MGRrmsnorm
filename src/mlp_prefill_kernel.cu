#include "mlp_prefill_kernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_CUBLASLT(expr) \
    do { \
        cublasStatus_t _status = (expr); \
        TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS, #expr " failed with status ", static_cast<int>(_status)); \
    } while (0)

namespace {

constexpr size_t kLtWorkspaceBytes = 32 * 1024 * 1024;

struct LtHandleState {
    cublasLtHandle_t handle = nullptr;
    LtHandleState() {
        CHECK_CUBLASLT(cublasLtCreate(&handle));
    }
    ~LtHandleState() {
        if (handle != nullptr) {
            cublasLtDestroy(handle);
        }
    }
};

LtHandleState& get_lt_state() {
    static LtHandleState state;
    return state;
}

struct LtAlgoKey {
    int64_t m;
    int64_t n;
    int64_t k;
    int device;
    bool operator==(const LtAlgoKey& other) const {
        return m == other.m && n == other.n && k == other.k && device == other.device;
    }
};

struct LtAlgoKeyHash {
    size_t operator()(const LtAlgoKey& key) const {
        size_t h = static_cast<size_t>(key.m);
        h = (h * 1315423911u) ^ static_cast<size_t>(key.n);
        h = (h * 1315423911u) ^ static_cast<size_t>(key.k);
        h = (h * 1315423911u) ^ static_cast<size_t>(key.device);
        return h;
    }
};

std::unordered_map<LtAlgoKey, cublasLtMatmulHeuristicResult_t, LtAlgoKeyHash>& get_algo_cache() {
    static std::unordered_map<LtAlgoKey, cublasLtMatmulHeuristicResult_t, LtAlgoKeyHash> cache;
    return cache;
}

std::mutex& get_algo_cache_mutex() {
    static std::mutex mu;
    return mu;
}

struct LtBf16PlanKey {
    int64_t m;
    int64_t n;
    int64_t k;
    int device;
    int algorithm_index;
    bool operator==(const LtBf16PlanKey& other) const {
        return m == other.m && n == other.n && k == other.k &&
            device == other.device && algorithm_index == other.algorithm_index;
    }
};

struct LtBf16PlanKeyHash {
    size_t operator()(const LtBf16PlanKey& key) const {
        size_t h = static_cast<size_t>(key.m);
        h = (h * 1315423911u) ^ static_cast<size_t>(key.n);
        h = (h * 1315423911u) ^ static_cast<size_t>(key.k);
        h = (h * 1315423911u) ^ static_cast<size_t>(key.device);
        h = (h * 1315423911u) ^ static_cast<size_t>(key.algorithm_index);
        return h;
    }
};

struct LtBf16Plan {
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic{};
    int heuristic_count = 0;

    ~LtBf16Plan() {
        if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
        if (c_layout != nullptr) cublasLtMatrixLayoutDestroy(c_layout);
        if (b_layout != nullptr) cublasLtMatrixLayoutDestroy(b_layout);
        if (a_layout != nullptr) cublasLtMatrixLayoutDestroy(a_layout);
        if (op_desc != nullptr) cublasLtMatmulDescDestroy(op_desc);
    }
};

std::unordered_map<
    LtBf16PlanKey,
    std::unique_ptr<LtBf16Plan>,
    LtBf16PlanKeyHash
>& get_bf16_plan_cache() {
    static std::unordered_map<
        LtBf16PlanKey,
        std::unique_ptr<LtBf16Plan>,
        LtBf16PlanKeyHash
    > cache;
    return cache;
}

std::mutex& get_bf16_plan_cache_mutex() {
    static std::mutex mu;
    return mu;
}

void* get_lt_workspace(int device) {
    static std::mutex mu;
    static void* workspace = nullptr;
    static int workspace_device = -1;
    static size_t workspace_bytes = 0;

    std::lock_guard<std::mutex> lock(mu);
    if (workspace == nullptr || workspace_device != device || workspace_bytes < kLtWorkspaceBytes) {
        if (workspace != nullptr && workspace_device == device) {
            cudaFree(workspace);
            workspace = nullptr;
            workspace_bytes = 0;
        }
        int prev_device = -1;
        cudaGetDevice(&prev_device);
        if (prev_device != device) {
            cudaSetDevice(device);
        }
        auto err = cudaMalloc(&workspace, kLtWorkspaceBytes);
        TORCH_CHECK(err == cudaSuccess, "cudaMalloc for cublasLt workspace failed");
        workspace_device = device;
        workspace_bytes = kLtWorkspaceBytes;
        if (prev_device != device && prev_device >= 0) {
            cudaSetDevice(prev_device);
        }
    }
    return workspace;
}

void set_row_major(cublasLtMatrixLayout_t layout) {
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
        layout,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order,
        sizeof(order)
    ));
}

std::unique_ptr<LtBf16Plan> make_bf16_plan(
    int64_t m,
    int64_t n,
    int64_t k,
    int algorithm_index
) {
    TORCH_CHECK(algorithm_index >= 0, "cuBLASLt algorithm index must be non-negative");
    auto plan = std::make_unique<LtBf16Plan>();
    cublasOperation_t trans_a = CUBLAS_OP_N;
    cublasOperation_t trans_b = CUBLAS_OP_T;
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(
        &plan->op_desc,
        CUBLAS_COMPUTE_32F,
        CUDA_R_32F
    ));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
        plan->op_desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &trans_a,
        sizeof(trans_a)
    ));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
        plan->op_desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &trans_b,
        sizeof(trans_b)
    ));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
        &plan->a_layout,
        CUDA_R_16BF,
        m,
        k,
        k
    ));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
        &plan->b_layout,
        CUDA_R_16BF,
        n,
        k,
        k
    ));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
        &plan->c_layout,
        CUDA_R_16BF,
        m,
        n,
        n
    ));
    set_row_major(plan->a_layout);
    set_row_major(plan->b_layout);
    set_row_major(plan->c_layout);
    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&plan->preference));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
        plan->preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &kLtWorkspaceBytes,
        sizeof(kLtWorkspaceBytes)
    ));

    constexpr int kMaximumHeuristics = 32;
    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(
        kMaximumHeuristics
    );
    int returned_results = 0;
    CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
        get_lt_state().handle,
        plan->op_desc,
        plan->a_layout,
        plan->b_layout,
        plan->c_layout,
        plan->c_layout,
        plan->preference,
        kMaximumHeuristics,
        heuristics.data(),
        &returned_results
    ));
    plan->heuristic_count = returned_results;
    TORCH_CHECK(
        algorithm_index < returned_results,
        "cuBLASLt BF16 algorithm index ",
        algorithm_index,
        " is unavailable; heuristic count is ",
        returned_results
    );
    plan->heuristic = heuristics[algorithm_index];
    TORCH_CHECK(
        plan->heuristic.state == CUBLAS_STATUS_SUCCESS,
        "cuBLASLt BF16 heuristic is not executable"
    );
    return plan;
}

LtBf16Plan* get_bf16_plan(
    int64_t m,
    int64_t n,
    int64_t k,
    int device,
    int algorithm_index
) {
    const LtBf16PlanKey key{m, n, k, device, algorithm_index};
    std::lock_guard<std::mutex> lock(get_bf16_plan_cache_mutex());
    auto& cache = get_bf16_plan_cache();
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(
            key,
            make_bf16_plan(m, n, k, algorithm_index)
        ).first;
    }
    return it->second.get();
}

bool use_fused_swiglu_down_kernel() {
    static int cached = -1;
    if (cached >= 0) {
        return cached == 1;
    }
    const char* raw = std::getenv("MEGAGEMM_NATIVE_MLP_FUSED_SWIGLU_DOWN");
    if (raw == nullptr) {
        cached = 0;
        return false;
    }
    if (
        std::strcmp(raw, "1") == 0 ||
        std::strcmp(raw, "true") == 0 ||
        std::strcmp(raw, "TRUE") == 0 ||
        std::strcmp(raw, "yes") == 0 ||
        std::strcmp(raw, "on") == 0
    ) {
        cached = 1;
        return true;
    }
    cached = 0;
    return false;
}

torch::Tensor cublaslt_matmul_fp16_row_major(
    torch::Tensor a,
    torch::Tensor b,
    c10::optional<torch::Tensor> bias,
    int64_t out_cols
) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(
        a.scalar_type() == torch::kFloat16 && b.scalar_type() == torch::kFloat16,
        "cublasLt matmul currently supports float16 only"
    );
    TORCH_CHECK(a.dim() == 2, "A must be 2D");
    TORCH_CHECK(b.dim() == 2, "B must be 2D");
    TORCH_CHECK(a.size(1) == b.size(1), "inner dimensions must match for A @ B^T");
    TORCH_CHECK(b.size(0) == out_cols, "out_cols must match B rows");

    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias.value().size(0) == out_cols, "bias size must match out_cols");
    }

    auto out = torch::empty({a.size(0), out_cols}, a.options());

    cublasLtHandle_t handle = get_lt_state().handle;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = out_cols;
    const int device = static_cast<int>(a.get_device());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasOperation_t trans_a = CUBLAS_OP_N;
    cublasOperation_t trans_b = CUBLAS_OP_T;
    cublasComputeType_t compute_type =
#ifdef CUBLAS_COMPUTE_32F_FAST_16F
        CUBLAS_COMPUTE_32F_FAST_16F;
#else
        CUBLAS_COMPUTE_32F;
#endif

    CHECK_CUBLASLT(cublasLtMatmulDescCreate(
        &op_desc,
        compute_type,
        CUDA_R_32F
    ));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
        op_desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &trans_a,
        sizeof(trans_a)
    ));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
        op_desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &trans_b,
        sizeof(trans_b)
    ));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16F, m, k, k));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16F, n, k, k));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_16F, m, n, n));
    set_row_major(a_layout);
    set_row_major(b_layout);
    set_row_major(c_layout);

    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &kLtWorkspaceBytes,
        sizeof(kLtWorkspaceBytes)
    ));

    LtAlgoKey algo_key{m, n, k, device};
    cublasLtMatmulHeuristicResult_t heuristic{};
    bool found_cached = false;
    {
        std::lock_guard<std::mutex> lock(get_algo_cache_mutex());
        auto& cache = get_algo_cache();
        auto it = cache.find(algo_key);
        if (it != cache.end()) {
            heuristic = it->second;
            found_cached = true;
        }
    }
    if (!found_cached) {
        int returned_results = 0;
        CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
            handle,
            op_desc,
            a_layout,
            b_layout,
            c_layout,
            c_layout,
            pref,
            1,
            &heuristic,
            &returned_results
        ));
        TORCH_CHECK(returned_results > 0, "cublasLtMatmulAlgoGetHeuristic returned no algorithm");
        std::lock_guard<std::mutex> lock(get_algo_cache_mutex());
        get_algo_cache()[algo_key] = heuristic;
    }

    void* workspace = get_lt_workspace(device);

    CHECK_CUBLASLT(cublasLtMatmul(
        handle,
        op_desc,
        &alpha,
        a.data_ptr<at::Half>(),
        a_layout,
        b.data_ptr<at::Half>(),
        b_layout,
        &beta,
        out.data_ptr<at::Half>(),
        c_layout,
        out.data_ptr<at::Half>(),
        c_layout,
        &heuristic.algo,
        workspace,
        kLtWorkspaceBytes,
        at::cuda::getCurrentCUDAStream()
    ));

    if (bias.has_value()) {
        out.add_(bias.value());
    }

    if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
    if (c_layout != nullptr) cublasLtMatrixLayoutDestroy(c_layout);
    if (b_layout != nullptr) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout != nullptr) cublasLtMatrixLayoutDestroy(a_layout);
    if (op_desc != nullptr) cublasLtMatmulDescDestroy(op_desc);

    return out;
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

template<int BLOCK_N, int BLOCK_K>
__global__ void swiglu_down_fp16_kernel(
    const __half* __restrict__ gate_up,
    const __half* __restrict__ down_weight,
    const __half* __restrict__ down_bias,
    __half* __restrict__ output,
    int rows,
    int intermediate_size,
    int out_features,
    bool has_bias
) {
    const int row = blockIdx.x;
    const int out_base = blockIdx.y * BLOCK_N;
    const int lane = threadIdx.x;
    if (row >= rows) return;

    const int out_col = out_base + lane;
    const __half* row_gate_up = gate_up + (static_cast<int64_t>(row) * intermediate_size * 2);
    const __half* row_weight = (out_col < out_features)
        ? (down_weight + static_cast<int64_t>(out_col) * intermediate_size)
        : nullptr;

    __shared__ float act_tile[BLOCK_K];
    float acc = 0.0f;

    for (int k0 = 0; k0 < intermediate_size; k0 += BLOCK_K) {
        if (lane < BLOCK_K) {
            const int k = k0 + lane;
            if (k < intermediate_size) {
                float gate = __half2float(row_gate_up[k]);
                float value = __half2float(row_gate_up[k + intermediate_size]);
                act_tile[lane] = silu_f32(gate) * value;
            } else {
                act_tile[lane] = 0.0f;
            }
        }
        __syncthreads();

        if (out_col < out_features) {
            const int remaining_k = intermediate_size - k0;
            const int valid_k = remaining_k < BLOCK_K ? remaining_k : BLOCK_K;
            #pragma unroll
            for (int kk = 0; kk < BLOCK_K; ++kk) {
                if (kk < valid_k) {
                    acc += act_tile[kk] * __half2float(row_weight[k0 + kk]);
                }
            }
        }
        __syncthreads();
    }

    if (out_col < out_features) {
        if (has_bias) {
            acc += __half2float(down_bias[out_col]);
        }
        output[static_cast<int64_t>(row) * out_features + out_col] = __float2half_rn(acc);
    }
}

__global__ void swiglu_forward_fp16_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int rows,
    int hidden_dim
) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const __half* row_in = input + (static_cast<int64_t>(row) * hidden_dim * 2);
    __half* row_out = output + (static_cast<int64_t>(row) * hidden_dim);

    for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
        float gate = __half2float(row_in[col]);
        float value = __half2float(row_in[col + hidden_dim]);
        float out = silu_f32(gate) * value;
        row_out[col] = __float2half_rn(out);
    }
}

}  // namespace


int64_t cublaslt_bf16_algorithm_count_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t maximum_algorithms
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    TORCH_CHECK(input.dim() == 2, "BF16 cuBLASLt input must be [M, K]");
    TORCH_CHECK(weight.dim() == 2, "BF16 cuBLASLt weight must be [N, K]");
    TORCH_CHECK(
        input.scalar_type() == torch::kBFloat16 &&
            weight.scalar_type() == torch::kBFloat16,
        "BF16 cuBLASLt algorithm search requires bfloat16 tensors"
    );
    TORCH_CHECK(
        input.device() == weight.device(),
        "BF16 cuBLASLt input and weight must be on the same device"
    );
    TORCH_CHECK(
        input.size(1) == weight.size(1),
        "BF16 cuBLASLt inner dimensions must match"
    );
    TORCH_CHECK(maximum_algorithms > 0, "maximum_algorithms must be positive");
    LtBf16Plan* plan = get_bf16_plan(
        input.size(0),
        weight.size(0),
        input.size(1),
        static_cast<int>(input.get_device()),
        0
    );
    return std::min<int64_t>(
        maximum_algorithms,
        static_cast<int64_t>(plan->heuristic_count)
    );
}


torch::Tensor cublaslt_bf16_linear_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> out,
    int64_t algorithm_index
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    TORCH_CHECK(input.dim() == 2, "BF16 cuBLASLt input must be [M, K]");
    TORCH_CHECK(weight.dim() == 2, "BF16 cuBLASLt weight must be [N, K]");
    TORCH_CHECK(
        input.scalar_type() == torch::kBFloat16 &&
            weight.scalar_type() == torch::kBFloat16,
        "BF16 cuBLASLt linear requires bfloat16 tensors"
    );
    TORCH_CHECK(
        input.device() == weight.device(),
        "BF16 cuBLASLt input and weight must be on the same device"
    );
    TORCH_CHECK(
        input.size(1) == weight.size(1),
        "BF16 cuBLASLt computes input @ weight.T; K dimensions must match"
    );
    TORCH_CHECK(
        algorithm_index >= 0 && algorithm_index < 32,
        "BF16 cuBLASLt algorithm index must be in [0, 31]"
    );

    const int64_t m = input.size(0);
    const int64_t n = weight.size(0);
    const int64_t k = input.size(1);
    torch::Tensor output;
    if (out.has_value()) {
        output = out.value();
        CHECK_INPUT(output);
        TORCH_CHECK(
            output.scalar_type() == torch::kBFloat16,
            "BF16 cuBLASLt output must be bfloat16"
        );
        TORCH_CHECK(
            output.device() == input.device(),
            "BF16 cuBLASLt output must be on the input device"
        );
        TORCH_CHECK(
            output.dim() == 2 && output.size(0) == m && output.size(1) == n,
            "BF16 cuBLASLt output must have shape [M, N]"
        );
    } else {
        output = torch::empty({m, n}, input.options());
    }
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
        TORCH_CHECK(
            bias.value().scalar_type() == torch::kBFloat16 &&
                bias.value().device() == input.device() &&
                bias.value().dim() == 1 && bias.value().size(0) == n,
            "BF16 cuBLASLt bias must be a contiguous bfloat16 [N] tensor"
        );
    }

    LtBf16Plan* plan = get_bf16_plan(
        m,
        n,
        k,
        static_cast<int>(input.get_device()),
        static_cast<int>(algorithm_index)
    );
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLASLT(cublasLtMatmul(
        get_lt_state().handle,
        plan->op_desc,
        &alpha,
        input.data_ptr<at::BFloat16>(),
        plan->a_layout,
        weight.data_ptr<at::BFloat16>(),
        plan->b_layout,
        &beta,
        output.data_ptr<at::BFloat16>(),
        plan->c_layout,
        output.data_ptr<at::BFloat16>(),
        plan->c_layout,
        &plan->heuristic.algo,
        get_lt_workspace(static_cast<int>(input.get_device())),
        kLtWorkspaceBytes,
        at::cuda::getCurrentCUDAStream()
    ));
    if (bias.has_value()) {
        output.add_(bias.value());
    }
    return output;
}


torch::Tensor swiglu_forward_cuda(torch::Tensor input, int64_t hidden_dim) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 2, "input must be 2D [rows, 2*hidden]");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat16,
        "swiglu_forward_cuda currently supports float16 only"
    );
    TORCH_CHECK(
        input.size(1) == hidden_dim * 2,
        "input last dim must equal 2 * hidden_dim"
    );

    auto rows = static_cast<int>(input.size(0));
    auto out = torch::empty({rows, hidden_dim}, input.options());

    constexpr int threads = 256;
    swiglu_forward_fp16_kernel<<<rows, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        rows,
        static_cast<int>(hidden_dim)
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "swiglu_forward_fp16_kernel failed");
    return out;
}


torch::Tensor mlp_prefill_forward_cuda(
    torch::Tensor input,
    torch::Tensor gate_up_weight,
    c10::optional<torch::Tensor> gate_up_bias,
    torch::Tensor down_weight,
    c10::optional<torch::Tensor> down_bias,
    int64_t intermediate_size
) {
    CHECK_INPUT(input);
    CHECK_INPUT(gate_up_weight);
    CHECK_INPUT(down_weight);
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat16,
        "mlp_prefill_forward_cuda currently supports float16 only"
    );
    TORCH_CHECK(input.dim() == 3, "input must be 3D [batch, seq, hidden]");
    TORCH_CHECK(gate_up_weight.dim() == 2, "gate_up_weight must be 2D [2I, H]");
    TORCH_CHECK(down_weight.dim() == 2, "down_weight must be 2D [H, I]");

    if (gate_up_bias.has_value()) {
        CHECK_INPUT(gate_up_bias.value());
    }
    if (down_bias.has_value()) {
        CHECK_INPUT(down_bias.value());
    }

    auto x = input.contiguous();
    auto x2d = x.view({-1, x.size(-1)});

    auto gate_up = cublaslt_matmul_fp16_row_major(
        x2d,
        gate_up_weight.contiguous(),
        gate_up_bias,
        gate_up_weight.size(0)
    );

    torch::Tensor out2d;
    if (use_fused_swiglu_down_kernel()) {
        auto gate_up_c = gate_up.contiguous();
        auto down_weight_c = down_weight.contiguous();
        c10::optional<torch::Tensor> down_bias_c = c10::nullopt;
        if (down_bias.has_value()) {
            down_bias_c = down_bias.value().contiguous();
        }

        out2d = torch::empty(
            {x2d.size(0), down_weight.size(0)},
            x2d.options()
        );
        constexpr int kBlockN = 128;
        constexpr int kBlockK = 64;
        dim3 grid(
            static_cast<unsigned int>(x2d.size(0)),
            static_cast<unsigned int>((down_weight.size(0) + kBlockN - 1) / kBlockN)
        );
        dim3 block(kBlockN);
        swiglu_down_fp16_kernel<kBlockN, kBlockK><<<
            grid,
            block,
            0,
            at::cuda::getCurrentCUDAStream()
        >>>(
            reinterpret_cast<const __half*>(gate_up_c.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(down_weight_c.data_ptr<at::Half>()),
            down_bias_c.has_value()
                ? reinterpret_cast<const __half*>(down_bias_c.value().data_ptr<at::Half>())
                : nullptr,
            reinterpret_cast<__half*>(out2d.data_ptr<at::Half>()),
            static_cast<int>(x2d.size(0)),
            static_cast<int>(intermediate_size),
            static_cast<int>(down_weight.size(0)),
            down_bias_c.has_value()
        );
        auto err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "swiglu_down_fp16_kernel failed");
    } else {
        auto activated = swiglu_forward_cuda(gate_up.contiguous(), intermediate_size);
        out2d = cublaslt_matmul_fp16_row_major(
            activated,
            down_weight.contiguous(),
            down_bias,
            down_weight.size(0)
        );
    }

    return out2d.view({x.size(0), x.size(1), down_weight.size(0)});
}

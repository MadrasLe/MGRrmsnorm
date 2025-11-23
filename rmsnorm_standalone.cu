#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <random>
#include "src/rmsnorm_kernel.h"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void rmsnorm_cpu(float* out, const float* in, const float* weight, int rows, int cols, float epsilon) {
    for (int i = 0; i < rows; ++i) {
        float sum_sq = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float x = in[i * cols + j];
            sum_sq += x * x;
        }
        float rms = 1.0f / sqrtf(sum_sq / cols + epsilon);
        
        for (int j = 0; j < cols; ++j) {
            out[i * cols + j] = in[i * cols + j] * rms * weight[j];
        }
    }
}

int main() {
    int rows = 2048;
    int cols = 4096;
    float epsilon = 1e-5f;

    size_t size_bytes = rows * cols * sizeof(float);
    size_t weight_bytes = cols * sizeof(float);

    std::vector<float> h_in(rows * cols);
    std::vector<float> h_weight(cols);
    std::vector<float> h_out_cpu(rows * cols);
    std::vector<float> h_out_gpu(rows * cols);

    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& x : h_in) x = dist(gen);
    for (auto& x : h_weight) x = dist(gen);

    float *d_in, *d_weight, *d_out, *d_inv_rms;
    CHECK_CUDA(cudaMalloc(&d_in, size_bytes));
    CHECK_CUDA(cudaMalloc(&d_weight, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, size_bytes));
    CHECK_CUDA(cudaMalloc(&d_inv_rms, rows * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice));

    std::cout << "Running CPU reference..." << std::endl;
    rmsnorm_cpu(h_out_cpu.data(), h_in.data(), h_weight.data(), rows, cols, epsilon);

    std::cout << "Running CUDA kernel..." << std::endl;
    // Call the wrapper
    bool ok = rmsnorm_cuda(d_out, d_inv_rms, d_in, d_weight, rows, cols, epsilon);
    if (!ok) {
        std::cerr << "Kernel launch failed!" << std::endl;
        return 1;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, size_bytes, cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        float diff = fabsf(h_out_cpu[i] - h_out_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Max difference between CPU and GPU: " << max_diff << std::endl;
    if (max_diff < 1e-4) {
        std::cout << "Validation PASSED" << std::endl;
    } else {
        std::cout << "Validation FAILED" << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_weight);
    cudaFree(d_out);
    cudaFree(d_inv_rms);

    return 0;
}

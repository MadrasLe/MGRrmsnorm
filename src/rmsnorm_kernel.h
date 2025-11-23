#pragma once
#include <cuda_runtime.h>

bool rmsnorm_cuda(float* out, float* inv_rms, const float* in, const float* weight, int rows, int cols, float epsilon);

bool rmsnorm_backward_cuda(
    float* grad_input,
    float* grad_weight,
    const float* grad_output,
    const float* input,
    const float* weight,
    const float* inv_rms,
    int rows,
    int cols
);

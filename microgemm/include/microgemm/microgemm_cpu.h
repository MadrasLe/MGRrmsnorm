#ifndef MICROGEMM_CPU_H
#define MICROGEMM_CPU_H

#include <stdint.h>

float microgemm_cpu_quantize_f32_to_i8(int8_t* out, const float* input, int count);
float microgemm_cpu_quantize_f32_to_biased_u8(int8_t* out, const float* input, int count);
void microgemm_cpu_rmsnorm_f32(
    float* out,
    const float* input,
    const float* weight,
    int count,
    float eps,
    int offset_weights
);

#endif

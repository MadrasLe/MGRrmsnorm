#include "microgemm/microgemm_cpu.h"
#include "microgemm/microgemm_platform.h"

#include <math.h>
#include <string.h>

#if MICROGEMM_CPU_ARM64_NEON
static float microgemm_neon_hmax_f32(float32x4_t v) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vmaxvq_f32(v);
#else
    float tmp[4];
    vst1q_f32(tmp, v);
    return fmaxf(fmaxf(tmp[0], tmp[1]), fmaxf(tmp[2], tmp[3]));
#endif
}

static float microgemm_neon_hsum_f32(float32x4_t v) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vaddvq_f32(v);
#else
    float tmp[4];
    vst1q_f32(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
}
#endif

float microgemm_cpu_quantize_f32_to_i8(int8_t* out, const float* input, int count) {
    float amax = 0.0f;
    int i = 0;

#if MICROGEMM_CPU_X86_AVX2
    __m256 vmax = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (; i + 7 < count; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        v = _mm256_andnot_ps(sign_mask, v);
        vmax = _mm256_max_ps(vmax, v);
    }

    {
        __m128 hi = _mm256_extractf128_ps(vmax, 1);
        __m128 lo = _mm256_castps256_ps128(vmax);
        __m128 m = _mm_max_ps(lo, hi);
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_movehdup_ps(m));
        amax = _mm_cvtss_f32(m);
    }
#elif MICROGEMM_CPU_ARM64_NEON
    {
        float32x4_t vmax = vdupq_n_f32(0.0f);
        for (; i + 3 < count; i += 4) {
            float32x4_t v = vabsq_f32(vld1q_f32(input + i));
            vmax = vmaxq_f32(vmax, v);
        }
        amax = microgemm_neon_hmax_f32(vmax);
    }
#endif

    for (; i < count; ++i) {
        float a = input[i] < 0.0f ? -input[i] : input[i];
        if (a > amax) {
            amax = a;
        }
    }

    if (amax == 0.0f) {
        memset(out, 0, (size_t)count);
        return 1.0f;
    }

    {
        const float scale = amax / 127.0f;
        const float inv_scale = 127.0f / amax;
        i = 0;

#if MICROGEMM_CPU_X86_AVX2
        {
            __m256 vinv = _mm256_set1_ps(inv_scale);
            for (; i + 7 < count; i += 8) {
                __m256 v = _mm256_loadu_ps(input + i);
                __m256 scaled = _mm256_mul_ps(v, vinv);
                __m256i i32 = _mm256_cvtps_epi32(scaled);
                __m128i lo16 = _mm256_castsi256_si128(i32);
                __m128i hi16 = _mm256_extracti128_si256(i32, 1);
                __m128i packed16 = _mm_packs_epi32(lo16, hi16);
                __m128i packed8 = _mm_packs_epi16(packed16, packed16);
                _mm_storel_epi64((__m128i*)(out + i), packed8);
            }
        }
#elif MICROGEMM_CPU_ARM64_NEON
        {
            float32x4_t vinv = vdupq_n_f32(inv_scale);
            float32x4_t vhalf = vdupq_n_f32(0.5f);
            float32x4_t vneg_half = vdupq_n_f32(-0.5f);
            float32x4_t vzero = vdupq_n_f32(0.0f);
            int32x4_t vmax_i32 = vdupq_n_s32(127);
            int32x4_t vmin_i32 = vdupq_n_s32(-128);
            for (; i + 7 < count; i += 8) {
                float32x4_t v0 = vmulq_f32(vld1q_f32(input + i), vinv);
                float32x4_t v1 = vmulq_f32(vld1q_f32(input + i + 4), vinv);
                uint32x4_t mask0 = vcgeq_f32(v0, vzero);
                uint32x4_t mask1 = vcgeq_f32(v1, vzero);
                v0 = vaddq_f32(v0, vbslq_f32(mask0, vhalf, vneg_half));
                v1 = vaddq_f32(v1, vbslq_f32(mask1, vhalf, vneg_half));
                int32x4_t i0 = vcvtq_s32_f32(v0);
                int32x4_t i1 = vcvtq_s32_f32(v1);
                i0 = vmaxq_s32(vmin_i32, vminq_s32(vmax_i32, i0));
                i1 = vmaxq_s32(vmin_i32, vminq_s32(vmax_i32, i1));
                int16x8_t packed16 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
                int8x8_t packed8 = vqmovn_s16(packed16);
                vst1_s8(out + i, packed8);
            }
        }
#endif

        for (; i < count; ++i) {
            float v = input[i] * inv_scale;
            int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
            if (iv > 127) {
                iv = 127;
            }
            if (iv < -128) {
                iv = -128;
            }
            out[i] = (int8_t)iv;
        }

        return scale;
    }
}

float microgemm_cpu_quantize_f32_to_biased_u8(int8_t* out, const float* input, int count) {
    float amax = 0.0f;
    int i = 0;

#if MICROGEMM_CPU_X86_AVX2
    __m256 vmax = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (; i + 7 < count; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        v = _mm256_andnot_ps(sign_mask, v);
        vmax = _mm256_max_ps(vmax, v);
    }

    {
        __m128 hi = _mm256_extractf128_ps(vmax, 1);
        __m128 lo = _mm256_castps256_ps128(vmax);
        __m128 m = _mm_max_ps(lo, hi);
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_movehdup_ps(m));
        amax = _mm_cvtss_f32(m);
    }
#elif MICROGEMM_CPU_ARM64_NEON
    {
        float32x4_t vmax = vdupq_n_f32(0.0f);
        for (; i + 3 < count; i += 4) {
            float32x4_t v = vabsq_f32(vld1q_f32(input + i));
            vmax = vmaxq_f32(vmax, v);
        }
        amax = microgemm_neon_hmax_f32(vmax);
    }
#endif

    for (; i < count; ++i) {
        float a = input[i] < 0.0f ? -input[i] : input[i];
        if (a > amax) {
            amax = a;
        }
    }

    if (amax == 0.0f) {
        memset(out, 0x80, (size_t)count);
        return 1.0f;
    }

    {
        const float scale = amax / 127.0f;
        const float inv_scale = 127.0f / amax;
        i = 0;

#if MICROGEMM_CPU_X86_AVX2
        {
            __m256 vinv = _mm256_set1_ps(inv_scale);
            __m128i vbias = _mm_set1_epi8((char)128u);
            for (; i + 7 < count; i += 8) {
                __m256 v = _mm256_loadu_ps(input + i);
                __m256 scaled = _mm256_mul_ps(v, vinv);
                __m256i i32 = _mm256_cvtps_epi32(scaled);
                __m128i lo16 = _mm256_castsi256_si128(i32);
                __m128i hi16 = _mm256_extracti128_si256(i32, 1);
                __m128i packed16 = _mm_packs_epi32(lo16, hi16);
                __m128i packed8 = _mm_packs_epi16(packed16, packed16);
                packed8 = _mm_add_epi8(packed8, vbias);
                _mm_storel_epi64((__m128i*)(out + i), packed8);
            }
        }
#elif MICROGEMM_CPU_ARM64_NEON
        {
            float32x4_t vinv = vdupq_n_f32(inv_scale);
            float32x4_t vhalf = vdupq_n_f32(0.5f);
            float32x4_t vneg_half = vdupq_n_f32(-0.5f);
            float32x4_t vzero = vdupq_n_f32(0.0f);
            int32x4_t vmax_i32 = vdupq_n_s32(127);
            int32x4_t vmin_i32 = vdupq_n_s32(-128);
            uint8x8_t vbias = vdup_n_u8(128);
            for (; i + 7 < count; i += 8) {
                float32x4_t v0 = vmulq_f32(vld1q_f32(input + i), vinv);
                float32x4_t v1 = vmulq_f32(vld1q_f32(input + i + 4), vinv);
                uint32x4_t mask0 = vcgeq_f32(v0, vzero);
                uint32x4_t mask1 = vcgeq_f32(v1, vzero);
                v0 = vaddq_f32(v0, vbslq_f32(mask0, vhalf, vneg_half));
                v1 = vaddq_f32(v1, vbslq_f32(mask1, vhalf, vneg_half));
                int32x4_t i0 = vcvtq_s32_f32(v0);
                int32x4_t i1 = vcvtq_s32_f32(v1);
                i0 = vmaxq_s32(vmin_i32, vminq_s32(vmax_i32, i0));
                i1 = vmaxq_s32(vmin_i32, vminq_s32(vmax_i32, i1));
                int16x8_t packed16 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
                int8x8_t packed8 = vqmovn_s16(packed16);
                uint8x8_t biased = vadd_u8(vreinterpret_u8_s8(packed8), vbias);
                vst1_s8(out + i, vreinterpret_s8_u8(biased));
            }
        }
#endif

        for (; i < count; ++i) {
            float v = input[i] * inv_scale;
            int iv = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
            if (iv > 127) {
                iv = 127;
            }
            if (iv < -128) {
                iv = -128;
            }
            out[i] = (int8_t)((uint8_t)(int8_t)iv + (uint8_t)128);
        }

        return scale;
    }
}

void microgemm_cpu_rmsnorm_f32(
    float* out,
    const float* input,
    const float* weight,
    int count,
    float eps,
    int offset_weights
) {
    float ss = 0.0f;
    int i = 0;

#if MICROGEMM_CPU_X86_AVX2
    {
        __m256 vss = _mm256_setzero_ps();
        for (; i + 7 < count; i += 8) {
            __m256 vx = _mm256_loadu_ps(input + i);
            vss = _mm256_add_ps(_mm256_mul_ps(vx, vx), vss);
        }
        __m128 hi = _mm256_extractf128_ps(vss, 1);
        __m128 lo = _mm256_castps256_ps128(vss);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_hadd_ps(s4, s4);
        s4 = _mm_hadd_ps(s4, s4);
        ss = _mm_cvtss_f32(s4);
    }
#elif MICROGEMM_CPU_ARM64_NEON
    {
        float32x4_t vss = vdupq_n_f32(0.0f);
        for (; i + 3 < count; i += 4) {
            float32x4_t vx = vld1q_f32(input + i);
            vss = vmlaq_f32(vss, vx, vx);
        }
        ss = microgemm_neon_hsum_f32(vss);
    }
#endif

    for (; i < count; ++i) {
        ss += input[i] * input[i];
    }

    {
        const float rms = 1.0f / sqrtf(ss / (float)count + eps);
        i = 0;

#if MICROGEMM_CPU_X86_AVX2
        {
            __m256 vrms = _mm256_set1_ps(rms);
            if (offset_weights) {
                __m256 vone = _mm256_set1_ps(1.0f);
                for (; i + 7 < count; i += 8) {
                    __m256 vx = _mm256_loadu_ps(input + i);
                    __m256 vw = _mm256_add_ps(_mm256_loadu_ps(weight + i), vone);
                    _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vw));
                }
            } else {
                for (; i + 7 < count; i += 8) {
                    __m256 vx = _mm256_loadu_ps(input + i);
                    __m256 vw = _mm256_loadu_ps(weight + i);
                    _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vw));
                }
            }
        }
#elif MICROGEMM_CPU_ARM64_NEON
        {
            float32x4_t vrms = vdupq_n_f32(rms);
            if (offset_weights) {
                float32x4_t vone = vdupq_n_f32(1.0f);
                for (; i + 3 < count; i += 4) {
                    float32x4_t vx = vld1q_f32(input + i);
                    float32x4_t vw = vaddq_f32(vld1q_f32(weight + i), vone);
                    vst1q_f32(out + i, vmulq_f32(vmulq_f32(vx, vrms), vw));
                }
            } else {
                for (; i + 3 < count; i += 4) {
                    float32x4_t vx = vld1q_f32(input + i);
                    float32x4_t vw = vld1q_f32(weight + i);
                    vst1q_f32(out + i, vmulq_f32(vmulq_f32(vx, vrms), vw));
                }
            }
        }
#endif

        for (; i < count; ++i) {
            float w = offset_weights ? (weight[i] + 1.0f) : weight[i];
            out[i] = input[i] * rms * w;
        }
    }
}

/*
 * ⚡ Optimized INT8 CPU Kernel for MegaGemm
 * ==========================================
 * W8A8: INT8 weights × INT8 activations → FP32 output
 *
 * Optimizations:
 *   1. VNNI: _mm256_dpbusd_epi32 — native int8 dot product (4x throughput)
 *   2. OpenMP: parallel over output rows
 *   3. Cache tiling: tile K to fit in L1 cache
 *   4. AVX2 fallback for CPUs without VNNI
 *   5. Scalar fallback for CPUs without AVX2
 *
 * Compile: gcc -O3 -mavx2 -mfma -mavxvnni -fopenmp -shared -fPIC \
 *               -o libcpu_gemv.so cpu_gemv.c
 *    MSVC: cl /O2 /arch:AVX2 /openmp /LD cpu_gemv.c /Fe:cpu_gemv.dll
 *
 * Author: Gabriel Yogi
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef _MSC_VER
  #include <intrin.h>
  #define EXPORT __declspec(dllexport)
  #define ALIGN32 __declspec(align(32))
#else
  #include <immintrin.h>
  #define EXPORT __attribute__((visibility("default")))
  #define ALIGN32 __attribute__((aligned(32)))
#endif

/* Cache tiling: tile K dimension to fit in L1 cache (~32KB).
 * Each tile: TILE_K int8 weights + TILE_K int8 inputs = 2 * TILE_K bytes.
 * With TILE_K=256: 512 bytes per tile pair, fits easily in L1. */
#define TILE_K 256


/* ═══════════════════════════════════════════════
 * Quantize input (activations) FP32 → INT8 on the fly
 *
 * Per-tensor absmax: scale = max(|x|) / 127
 * Returns scale factor. Output is int8.
 * ═══════════════════════════════════════════════ */
static float quantize_input_f32_to_i8(int8_t* out, const float* input, int K)
{
    /* Find absmax */
    float amax = 0.0f;
    int j = 0;

#if defined(__AVX2__)
    __m256 vmax = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (; j + 7 < K; j += 8) {
        __m256 v = _mm256_loadu_ps(input + j);
        v = _mm256_andnot_ps(sign_mask, v);  /* abs */
        vmax = _mm256_max_ps(vmax, v);
    }
    /* Horizontal max of 8 floats */
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_movehdup_ps(m));
    amax = _mm_cvtss_f32(m);
#endif
    for (; j < K; j++) {
        float a = input[j] < 0 ? -input[j] : input[j];
        if (a > amax) amax = a;
    }

    if (amax == 0.0f) { memset(out, 0, K); return 1.0f; }

    float scale = amax / 127.0f;
    float inv_scale = 127.0f / amax;

    j = 0;
#if defined(__AVX2__)
    __m256 vs = _mm256_set1_ps(inv_scale);
    for (; j + 7 < K; j += 8) {
        __m256 v = _mm256_loadu_ps(input + j);
        __m256 scaled = _mm256_mul_ps(v, vs);
        __m256i i32 = _mm256_cvtps_epi32(scaled);  /* round to nearest */
        /* Pack int32 → int16 → int8 */
        __m128i lo16 = _mm256_castsi256_si128(i32);
        __m128i hi16 = _mm256_extracti128_si256(i32, 1);
        __m128i packed16 = _mm_packs_epi32(lo16, hi16);
        __m128i packed8 = _mm_packs_epi16(packed16, packed16);
        _mm_storel_epi64((__m128i*)(out + j), packed8);
    }
#endif
    for (; j < K; j++) {
        float v = input[j] * inv_scale;
        int iv = (int)(v + (v >= 0 ? 0.5f : -0.5f));
        if (iv > 127) iv = 127;
        if (iv < -128) iv = -128;
        out[j] = (int8_t)iv;
    }

    return scale;
}


/* ═══════════════════════════════════════════════
 * VNNI path: uses _mm256_dpbusd_epi32
 *
 * dpbusd: acc += dot(uint8[4], int8[4]) for 8 lanes
 * → 32 int8 multiplies + 8 int32 accumulates per instruction
 *
 * Trick for signed × signed with dpbusd (unsigned × signed):
 *   input_u8 = input_i8 + 128
 *   result = dpbusd(input_u8, weight_i8)
 *   compensation = 128 * sum(weight_i8) per output row
 *   final = result - compensation
 * ═══════════════════════════════════════════════ */
#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)

static void gemv_vnni(
    float*        output,
    const int8_t* weights,   /* [M, K] */
    const float*  w_scales,  /* [M]    */
    const int8_t* input_i8,  /* [K]    */
    float         input_scale,
    int M, int K)
{
    /* Pre-compute compensation vector for signed→unsigned conversion.
     * For each output row i: comp[i] = 128 * sum(weights[i,:]) */

    __m256i v128 = _mm256_set1_epi8((char)128);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const int8_t* row = weights + (long long)i * K;
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i comp = _mm256_setzero_si256();  /* compensation accumulator */
        __m256i ones_u8 = _mm256_set1_epi8(1);

        int j = 0;
        for (; j + 63 < K; j += 64) {
            /* Load 32 bytes of weight (int8) and input (int8) */
            __m256i w0 = _mm256_loadu_si256((const __m256i*)(row + j));
            __m256i w1 = _mm256_loadu_si256((const __m256i*)(row + j + 32));
            __m256i x0 = _mm256_loadu_si256((const __m256i*)(input_i8 + j));
            __m256i x1 = _mm256_loadu_si256((const __m256i*)(input_i8 + j + 32));

            /* Convert input from signed to unsigned: x_u8 = x_i8 + 128 */
            __m256i xu0 = _mm256_add_epi8(x0, v128);
            __m256i xu1 = _mm256_add_epi8(x1, v128);

            /* VNNI: acc += dpbusd(unsigned_input, signed_weight) */
            acc0 = _mm256_dpbusd_epi32(acc0, xu0, w0);
            acc1 = _mm256_dpbusd_epi32(acc1, xu1, w1);

            /* Compensation: accumulate sum of weights using dpbusd with ones */
            comp = _mm256_dpbusd_epi32(comp, ones_u8, w0);
            comp = _mm256_dpbusd_epi32(comp, ones_u8, w1);
        }

        for (; j + 31 < K; j += 32) {
            __m256i w0 = _mm256_loadu_si256((const __m256i*)(row + j));
            __m256i x0 = _mm256_loadu_si256((const __m256i*)(input_i8 + j));
            __m256i xu0 = _mm256_add_epi8(x0, v128);
            acc0 = _mm256_dpbusd_epi32(acc0, xu0, w0);
            comp = _mm256_dpbusd_epi32(comp, ones_u8, w0);
        }

        /* Combine accumulators */
        __m256i total = _mm256_add_epi32(acc0, acc1);

        /* Subtract compensation: 128 * sum(weights) */
        __m256i comp_scaled = _mm256_mullo_epi32(comp, _mm256_set1_epi32(128));
        /* Wait — comp already has sum of 4-byte groups × 1. We need total scalar comp. */
        /* Actually, dpbusd(ones, w) gives sum of w per 4-byte group, accumulated in i32. */
        /* comp has 8 partial sums. We need the total. */

        /* Horizontal sum of total (8 × int32) */
        __m128i t_lo = _mm256_castsi256_si128(total);
        __m128i t_hi = _mm256_extracti128_si256(total, 1);
        __m128i t_sum = _mm_add_epi32(t_lo, t_hi);
        t_sum = _mm_hadd_epi32(t_sum, t_sum);
        t_sum = _mm_hadd_epi32(t_sum, t_sum);
        int dot_raw = _mm_cvtsi128_si32(t_sum);

        /* Horizontal sum of comp */
        __m128i c_lo = _mm256_castsi256_si128(comp);
        __m128i c_hi = _mm256_extracti128_si256(comp, 1);
        __m128i c_sum = _mm_add_epi32(c_lo, c_hi);
        c_sum = _mm_hadd_epi32(c_sum, c_sum);
        c_sum = _mm_hadd_epi32(c_sum, c_sum);
        int comp_total = _mm_cvtsi128_si32(c_sum);

        int corrected = dot_raw - 128 * comp_total;

        /* Scalar tail */
        for (; j < K; j++) {
            corrected += (int)input_i8[j] * (int)row[j];
        }

        output[i] = (float)corrected * w_scales[i] * input_scale;
    }
}

#endif /* VNNI */


/* ═══════════════════════════════════════════════
 * AVX2 path: int8→float conversion + FMA
 * With OpenMP + cache tiling
 *
 * NOTE: This path dequantizes inputs to FP32 then uses FMA.
 *   - Advantage: uses well-optimized FMA pipeline, simple code
 *   - Disadvantage: 4x more memory for input (float vs int8),
 *     extra conversion step, slightly less precise than
 *     true INT8×INT8 (VNNI path) due to floating-point rounding
 *   - Falls back here when VNNI is not available
 * ═══════════════════════════════════════════════ */
#if defined(__AVX2__) || defined(_MSC_VER)

static void gemv_avx2_tiled(
    float*        output,
    const int8_t* weights,
    const float*  w_scales,
    const int8_t* input_i8,
    float         input_scale,
    int M, int K)
{
    /* Pre-dequantize input to float — use stack for small K, heap for large */
    float* input_f;
    int heap_alloc = 0;
    if (K <= 16384) {
        input_f = (float*)alloca(K * sizeof(float));
    } else {
        input_f = (float*)malloc(K * sizeof(float));
        if (!input_f) return;
        heap_alloc = 1;
    }
    for (int j = 0; j < K; j++) {
        input_f[j] = (float)input_i8[j] * input_scale;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const int8_t* row = weights + (long long)i * K;
        float row_scale = w_scales[i];
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        /* Track total elements processed across all tiles */
        int processed = 0;

        /* Process K in tiles for cache locality */
        for (int tile = 0; tile < K; tile += TILE_K) {
            int tile_end = tile + TILE_K;
            if (tile_end > K) tile_end = K;

            for (int j = tile; j + 31 < tile_end; j += 32) {
                __m128i b0 = _mm_loadl_epi64((const __m128i*)(row + j));
                __m128i b1 = _mm_loadl_epi64((const __m128i*)(row + j + 8));
                __m128i b2 = _mm_loadl_epi64((const __m128i*)(row + j + 16));
                __m128i b3 = _mm_loadl_epi64((const __m128i*)(row + j + 24));

                __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0));
                __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1));
                __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b2));
                __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b3));

                __m256 x0 = _mm256_loadu_ps(input_f + j);
                __m256 x1 = _mm256_loadu_ps(input_f + j + 8);
                __m256 x2 = _mm256_loadu_ps(input_f + j + 16);
                __m256 x3 = _mm256_loadu_ps(input_f + j + 24);

                acc0 = _mm256_fmadd_ps(w0, x0, acc0);
                acc1 = _mm256_fmadd_ps(w1, x1, acc1);
                acc2 = _mm256_fmadd_ps(w2, x2, acc2);
                acc3 = _mm256_fmadd_ps(w3, x3, acc3);
                processed = j + 32;
            }
        }

        __m256 sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                   _mm256_add_ps(acc2, acc3));
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_hadd_ps(s4, s4);
        s4 = _mm_hadd_ps(s4, s4);
        float result = _mm_cvtss_f32(s4);

        /* Scalar tail: process elements not handled by 32-wide SIMD */
        for (int j = processed; j < K; j++) {
            result += (float)row[j] * input_f[j];
        }

        output[i] = result * row_scale;
    }

    if (heap_alloc) free(input_f);
}

#endif /* AVX2 */


/* ═══════════════════════════════════════════════
 * Scalar fallback (any CPU)
 * ═══════════════════════════════════════════════ */
static void gemv_scalar(
    float*        output,
    const int8_t* weights,
    const float*  w_scales,
    const int8_t* input_i8,
    float         input_scale,
    int M, int K)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const int8_t* row = weights + (long long)i * K;
        int acc = 0;
        for (int j = 0; j < K; j++) {
            acc += (int)row[j] * (int)input_i8[j];
        }
        output[i] = (float)acc * w_scales[i] * input_scale;
    }
}


/* ═══════════════════════════════════════════════
 * Public API: GEMV (decode — 1 token)
 *
 * input is FP32 → dynamically quantized to INT8
 * ═══════════════════════════════════════════════ */
EXPORT void megagemm_gemv_w8a32(
    float*        output,
    const int8_t* weights,
    const float*  scales,
    const float*  input,
    int M, int K)
{
    /* Dynamic quantization: FP32 input → INT8 */
    int8_t* input_i8 = (int8_t*)malloc(K);
    if (!input_i8) return;
    float input_scale = quantize_input_f32_to_i8(input_i8, input, K);

#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)
    gemv_vnni(output, weights, scales, input_i8, input_scale, M, K);
#elif defined(__AVX2__) || defined(_MSC_VER)
    gemv_avx2_tiled(output, weights, scales, input_i8, input_scale, M, K);
#else
    gemv_scalar(output, weights, scales, input_i8, input_scale, M, K);
#endif

    free(input_i8);
}


/* ═══════════════════════════════════════════════
 * Public API: GEMM (prefill — N tokens)
 *
 * NOTE: Currently dispatches N independent GEMVs.
 * Each GEMV already parallelizes over M with OpenMP.
 * For large N, the outer loop is sequential, which is
 * correct but suboptimal — a true tiled GEMM would
 * amortize memory loads across tokens.
 *
 * TODO: For N>1 prefill, implement blocked GEMM with
 * tiling over (N, M, K) for better cache utilization.
 * ═══════════════════════════════════════════════ */
EXPORT void megagemm_gemm_w8a32(
    float*        output,
    const int8_t* weights,
    const float*  scales,
    const float*  input,
    int N, int M, int K)
{
    /* For small N (typical decode), sequential is fine.
     * For large N (prefill), each GEMV uses OpenMP over M. */
    for (int n = 0; n < N; n++) {
        megagemm_gemv_w8a32(
            output + (long long)n * M,
            weights, scales,
            input + (long long)n * K,
            M, K
        );
    }
}


/* ═══════════════════════════════════════════════
 * Quantize FP32 weights → INT8 + per-row scales
 * ═══════════════════════════════════════════════ */
EXPORT void megagemm_quantize_w8(
    int8_t*      out_weights,
    float*       out_scales,
    const float* fp32_weights,
    int M, int K)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const float* row = fp32_weights + (long long)i * K;
        float amax = 0.0f;
        for (int j = 0; j < K; j++) {
            float a = row[j] < 0 ? -row[j] : row[j];
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        if (scale == 0.0f) scale = 1.0f;
        float inv_scale = 127.0f / amax;
        if (amax == 0.0f) inv_scale = 0.0f;

        out_scales[i] = scale;
        int8_t* out_row = out_weights + (long long)i * K;
        for (int j = 0; j < K; j++) {
            float v = row[j] * inv_scale;
            int iv = (int)(v + (v >= 0 ? 0.5f : -0.5f));
            if (iv > 127) iv = 127;
            if (iv < -128) iv = -128;
            out_row[j] = (int8_t)iv;
        }
    }
}


/* ═══════════════════════════════════════════════
 * Query: how many threads are available
 * ═══════════════════════════════════════════════ */
EXPORT int megagemm_get_num_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

EXPORT void megagemm_set_num_threads(int n)
{
#ifdef _OPENMP
    omp_set_num_threads(n);
#endif
}

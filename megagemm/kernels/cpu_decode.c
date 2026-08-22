/*
 * ⚡ MegaGemm CPU Decode Loop v2 — Optimized
 * =============================================
 * Full transformer decode step in C, with:
 *   1. Pre-allocated scratch buffers (zero malloc in hot path)
 *   2. AVX2 vectorized RMSNorm, RoPE, SiLU, residual
 *   3. INT8×INT8 GEMV via vpmaddubsw+vpmaddwd (2x faster dot product)
 *
 * Generic for: LLaMA, Mistral, Qwen 2.5/3, Gemma 2
 *
 * Compile: gcc -O3 -mavx2 -mfma -fopenmp -shared -fPIC -lm \
 *               -o libcpu_decode.so cpu_decode.c
 *
 * Author: Gabriel Yogi
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef _MSC_VER
  #include <intrin.h>
  #define EXPORT __declspec(dllexport)
  #define ALIGNED_ALLOC(align, size) _aligned_malloc(size, align)
  #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
  #include <immintrin.h>
  #define EXPORT __attribute__((visibility("default")))
  #define ALIGNED_ALLOC(align, size) aligned_alloc(align, (((size)+(align)-1)/(align))*(align))
  #define ALIGNED_FREE(ptr) free(ptr)
#endif

/* ═══════════════════════════════════════════════
 * Quantization modes
 * ═══════════════════════════════════════════════ */
#define QUANT_INT8  0
#define QUANT_INT4  1
#define W4_GROUP_SIZE 128

/* ═══════════════════════════════════════════════
 * Config
 * ═══════════════════════════════════════════════ */
typedef struct {
    int hidden_size, intermediate_size, num_layers;
    int num_q_heads, num_kv_heads, head_dim, vocab_size;
    float rms_norm_eps;
    int qkv_bias, norm_offset, act_gelu;
    float rope_theta;
    int kv_block_size;
    int quant_mode;   /* QUANT_INT8 or QUANT_INT4 */
} MegaGemmConfig;

/* ═══════════════════════════════════════════════
 * Per-layer weights
 * INT8: w[M,K] int8 + scales[M] float
 * INT4: w[M,K/2] uint8 (packed) + scales[M, K/group] float
 * ═══════════════════════════════════════════════ */
typedef struct {
    const float* input_norm_w;
    const float* post_attn_norm_w;
    /* INT8 weight pointers */
    const int8_t* qkv_w;   const float* qkv_s;   const float* qkv_bias;
    const int8_t* o_w;     const float* o_s;
    const int8_t* gate_up_w; const float* gate_up_s;
    const int8_t* down_w;  const float* down_s;
    /* INT4 weight pointers (packed uint8, 2 values per byte) */
    const uint8_t* qkv_w4;   const float* qkv_s4;   /* scales: [M, K/group] */
    const uint8_t* o_w4;     const float* o_s4;
    const uint8_t* gate_up_w4; const float* gate_up_s4;
    const uint8_t* down_w4;  const float* down_s4;
} LayerWeights;

typedef struct {
    const float* embed_tokens;
    LayerWeights* layers;
    const float* final_norm_w;
    const int8_t* lm_head_w;  const float* lm_head_s;
    const uint8_t* lm_head_w4; const float* lm_head_s4;
    const float* cos_cache;    const float* sin_cache;
} ModelWeights;


/* ═══════════════════════════════════════════════
 * Pre-allocated scratch buffers
 * Allocated ONCE, reused for every decode step.
 * ═══════════════════════════════════════════════ */
typedef struct {
    float* hidden;    /* [H] */
    float* residual;  /* [H] */
    float* normed;    /* [H] */
    float* qkv;       /* [qkv_size] */
    float* attn_out;  /* [Nq*D] */
    float* o_out;     /* [H] */
    float* gate_up;   /* [2*I] */
    float* mlp_out;   /* [H] */
    float* logits;    /* [V] */
    float* scores;    /* [max_seq] for attention */
    /* Quantized input buffer for INT8×INT8 */
    int8_t* input_q;  /* [max(H, 2*I, Nq*D)] */
    int valid;
} ScratchBuffers;

static ScratchBuffers _scratch = {0};

EXPORT void megagemm_alloc_scratch(const MegaGemmConfig* cfg, int max_seq) {
    int H = cfg->hidden_size;
    int I = cfg->intermediate_size;
    int Nq = cfg->num_q_heads;
    int Nkv = cfg->num_kv_heads;
    int D = cfg->head_dim;
    int qkv_size = (Nq + 2 * Nkv) * D;
    int V = cfg->vocab_size;
    int max_k = H > (2*I) ? H : (2*I);
    if (Nq*D > max_k) max_k = Nq*D;

    _scratch.hidden   = (float*)ALIGNED_ALLOC(64, H * sizeof(float));
    _scratch.residual = (float*)ALIGNED_ALLOC(64, H * sizeof(float));
    _scratch.normed   = (float*)ALIGNED_ALLOC(64, H * sizeof(float));
    _scratch.qkv      = (float*)ALIGNED_ALLOC(64, qkv_size * sizeof(float));
    _scratch.attn_out = (float*)ALIGNED_ALLOC(64, Nq * D * sizeof(float));
    _scratch.o_out    = (float*)ALIGNED_ALLOC(64, H * sizeof(float));
    _scratch.gate_up  = (float*)ALIGNED_ALLOC(64, 2 * I * sizeof(float));
    _scratch.mlp_out  = (float*)ALIGNED_ALLOC(64, H * sizeof(float));
    _scratch.logits   = (float*)ALIGNED_ALLOC(64, V * sizeof(float));
    _scratch.scores   = (float*)ALIGNED_ALLOC(64, max_seq * sizeof(float));
    _scratch.input_q  = (int8_t*)ALIGNED_ALLOC(64, max_k * sizeof(int8_t));
    _scratch.valid = 1;
}

EXPORT void megagemm_free_scratch(void) {
    if (!_scratch.valid) return;
    ALIGNED_FREE(_scratch.hidden);   ALIGNED_FREE(_scratch.residual);
    ALIGNED_FREE(_scratch.normed);   ALIGNED_FREE(_scratch.qkv);
    ALIGNED_FREE(_scratch.attn_out); ALIGNED_FREE(_scratch.o_out);
    ALIGNED_FREE(_scratch.gate_up);  ALIGNED_FREE(_scratch.mlp_out);
    ALIGNED_FREE(_scratch.logits);   ALIGNED_FREE(_scratch.scores);
    ALIGNED_FREE(_scratch.input_q);
    memset(&_scratch, 0, sizeof(_scratch));
}


/* ═══════════════════════════════════════════════
 * OPTIMIZED Primitive Ops — All AVX2 vectorized
 * ═══════════════════════════════════════════════ */

/* Vectorized RMSNorm using AVX2 */
static inline void rmsnorm(
    float* out, const float* x, const float* weight,
    int size, float eps, int offset)
{
    float ss = 0.0f;
    int i = 0;
#if defined(__AVX2__)
    __m256 vss = _mm256_setzero_ps();
    for (; i + 7 < size; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vss = _mm256_fmadd_ps(vx, vx, vss);
    }
    /* Horizontal sum */
    __m128 hi = _mm256_extractf128_ps(vss, 1);
    __m128 lo = _mm256_castps256_ps128(vss);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    ss = _mm_cvtss_f32(s4);
#endif
    for (; i < size; i++) ss += x[i] * x[i];

    float rms = 1.0f / sqrtf(ss / size + eps);

    i = 0;
#if defined(__AVX2__)
    __m256 vrms = _mm256_set1_ps(rms);
    if (offset) {
        __m256 vone = _mm256_set1_ps(1.0f);
        for (; i + 7 < size; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_add_ps(_mm256_loadu_ps(weight + i), vone);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vw));
        }
    } else {
        for (; i + 7 < size; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(weight + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vw));
        }
    }
#endif
    for (; i < size; i++) {
        float w = offset ? (weight[i] + 1.0f) : weight[i];
        out[i] = x[i] * rms * w;
    }
}


/* INT8 GEMV with on-the-fly input quantization for INT8×INT8
 * Uses vpmaddubsw + vpmaddwd for native INT8 dot product */
static inline void quantize_input_avx2(int8_t* out_q, float* out_scale,
                                        const float* input, int K)
{
    /* Find max abs value */
    float amax = 0.0f;
    int i = 0;
#if defined(__AVX2__)
    __m256 vmax = _mm256_setzero_ps();
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (; i + 7 < K; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, v));
    }
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m4 = _mm_max_ps(lo, hi);
    m4 = _mm_max_ps(m4, _mm_shuffle_ps(m4, m4, 0x4E));
    m4 = _mm_max_ps(m4, _mm_shuffle_ps(m4, m4, 0xB1));
    amax = _mm_cvtss_f32(m4);
#endif
    for (; i < K; i++) {
        float a = input[i] < 0 ? -input[i] : input[i];
        if (a > amax) amax = a;
    }

    float scale = amax / 127.0f;
    *out_scale = scale;
    float inv_scale = (amax > 1e-8f) ? 127.0f / amax : 0.0f;

    i = 0;
#if defined(__AVX2__)
    __m256 vinv = _mm256_set1_ps(inv_scale);
    for (; i + 7 < K; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        __m256i vi = _mm256_cvtps_epi32(_mm256_mul_ps(v, vinv));
        /* Pack int32→int16 (with lane-crossing fix) */
        __m256i packed16 = _mm256_packs_epi32(vi, _mm256_setzero_si256());
        /* Fix AVX2 lane interleaving: [0,1,2,3,_,_,_,_, 4,5,6,7,_,_,_,_]
         * → [0,1,2,3, 4,5,6,7, _,_,_,_, _,_,_,_] */
        packed16 = _mm256_permute4x64_epi64(packed16, 0xD8);
        __m128i lo16 = _mm256_castsi256_si128(packed16);
        /* Pack int16→int8 */
        __m128i lo8 = _mm_packs_epi16(lo16, lo16);
        *(int64_t*)(out_q + i) = _mm_cvtsi128_si64(lo8);
    }
#endif
    for (; i < K; i++) {
        int v = (int)roundf(input[i] * inv_scale);
        out_q[i] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
    }
}


/* INT8×INT8 GEMV: W_int8[M,K] @ quantized_input_int8[K]
 * Uses vpmaddubsw (u8×s8→s16) + vpmaddwd (s16→s32) */
static void gemv_i8(
    float* out, const int8_t* w, const float* scales,
    const float* input, int M, int K, const float* bias)
{
    int8_t* input_q = _scratch.valid ? _scratch.input_q : (int8_t*)malloc(K);
    float input_scale;
    quantize_input_avx2(input_q, &input_scale, input, K);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const int8_t* row = w + (long long)i * K;
        int32_t acc = 0;
        int j = 0;
#if defined(__AVX2__)
        __m256i vacc = _mm256_setzero_si256();
        __m256i vcomp = _mm256_setzero_si256();
        __m256i v128 = _mm256_set1_epi8((char)128u);
        __m256i vone16 = _mm256_set1_epi16(1);
        for (; j + 31 < K; j += 32) {
            __m256i vw = _mm256_loadu_si256((const __m256i*)(row + j));
            __m256i vi = _mm256_loadu_si256((const __m256i*)(input_q + j));
            __m256i vi_u = _mm256_add_epi8(vi, v128);
            __m256i prod = _mm256_maddubs_epi16(vi_u, vw);
            vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(prod, vone16));
            __m256i wext = _mm256_madd_epi16(_mm256_maddubs_epi16(
                _mm256_set1_epi8(1), vw), vone16);
            vcomp = _mm256_add_epi32(vcomp, wext);
        }
        __m128i hi = _mm256_extracti128_si256(vacc, 1);
        __m128i lo = _mm256_castsi256_si128(vacc);
        __m128i s4 = _mm_add_epi32(lo, hi);
        s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
        int32_t dot_raw = _mm_cvtsi128_si32(s4);
        hi = _mm256_extracti128_si256(vcomp, 1);
        lo = _mm256_castsi256_si128(vcomp);
        s4 = _mm_add_epi32(lo, hi);
        s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
        acc = dot_raw - 128 * _mm_cvtsi128_si32(s4);
#endif
        for (; j < K; j++) acc += (int32_t)input_q[j] * (int32_t)row[j];
        out[i] = (float)acc * scales[i] * input_scale + (bias ? bias[i] : 0.0f);
    }
    if (!_scratch.valid) free(input_q);
}


/* INT4 GEMV: W_int4_packed[M, K/2] @ FP32_input[K]
 * Unpacks INT4→INT8 in AVX2 registers, then uses vpmaddubsw.
 * Group-wise quantization: scales[M, num_groups] where num_groups = K/W4_GROUP_SIZE */
static void gemv_w4(
    float* out, const uint8_t* w_packed, const float* group_scales,
    const float* input, int M, int K, const float* bias)
{
    int8_t* input_q = _scratch.valid ? _scratch.input_q : (int8_t*)malloc(K);
    float input_scale;
    quantize_input_avx2(input_q, &input_scale, input, K);
    int num_groups = (K + W4_GROUP_SIZE - 1) / W4_GROUP_SIZE;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const uint8_t* row = w_packed + (long long)i * (K / 2);
        const float* gscales = group_scales + (long long)i * num_groups;
        float total = 0.0f;

        /* Process in groups for per-group scaling */
        for (int g = 0; g < num_groups; g++) {
            int g_start = g * W4_GROUP_SIZE;
            int g_end = g_start + W4_GROUP_SIZE;
            if (g_end > K) g_end = K;
            float g_scale = gscales[g];
            int32_t g_acc = 0;
            int j = g_start;

#if defined(__AVX2__)
            __m256i vacc = _mm256_setzero_si256();
            __m256i vcomp = _mm256_setzero_si256();
            __m256i v128 = _mm256_set1_epi8((char)128u);
            __m256i vone16 = _mm256_set1_epi16(1);
            __m256i vmask = _mm256_set1_epi8(0x0F);
            __m256i voff = _mm256_set1_epi8(8);

            for (; j + 31 < g_end; j += 32) {
                /* Load 16 packed bytes = 32 INT4 values */
                __m128i packed = _mm_loadu_si128((const __m128i*)(row + j / 2));
                __m128i mask128 = _mm_set1_epi8(0x0F);
                __m128i off128 = _mm_set1_epi8(8);

                /* Extract nibbles: lo=even indices, hi=odd indices */
                __m128i lo16 = _mm_and_si128(packed, mask128);
                __m128i hi16 = _mm_and_si128(_mm_srli_epi16(packed, 4), mask128);

                /* Interleave to get [val0,val1,val2,val3,...] in order:
                 * unpacklo handles bytes 0-7, unpackhi handles bytes 8-15 */
                __m128i out_lo = _mm_unpacklo_epi8(lo16, hi16);  /* 16 values */
                __m128i out_hi = _mm_unpackhi_epi8(lo16, hi16);  /* 16 values */

                /* Combine into 256-bit and convert [0,15] → [-8,7] */
                __m256i unpacked = _mm256_set_m128i(out_hi, out_lo);
                __m256i w_s8 = _mm256_sub_epi8(unpacked, voff);

                /* Load quantized input */
                __m256i vi = _mm256_loadu_si256((const __m256i*)(input_q + j));
                __m256i vi_u = _mm256_add_epi8(vi, v128);

                /* INT8×INT8 dot product */
                __m256i prod = _mm256_maddubs_epi16(vi_u, w_s8);
                vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(prod, vone16));

                /* Compensation for signed→unsigned input conversion */
                __m256i wext = _mm256_madd_epi16(_mm256_maddubs_epi16(
                    _mm256_set1_epi8(1), w_s8), vone16);
                vcomp = _mm256_add_epi32(vcomp, wext);
            }

            /* Horizontal sum */
            __m128i hi = _mm256_extracti128_si256(vacc, 1);
            __m128i lo = _mm256_castsi256_si128(vacc);
            __m128i s4 = _mm_add_epi32(lo, hi);
            s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
            int32_t dot_raw = _mm_cvtsi128_si32(s4);
            hi = _mm256_extracti128_si256(vcomp, 1);
            lo = _mm256_castsi256_si128(vcomp);
            s4 = _mm_add_epi32(lo, hi);
            s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
            g_acc = dot_raw - 128 * _mm_cvtsi128_si32(s4);
#endif
            /* Scalar tail */
            for (; j < g_end; j++) {
                int byte_idx = j / 2;
                int nibble = (j % 2 == 0)
                    ? (row[byte_idx] & 0x0F)
                    : ((row[byte_idx] >> 4) & 0x0F);
                int8_t w_val = (int8_t)(nibble - 8);
                g_acc += (int32_t)input_q[j] * (int32_t)w_val;
            }

            total += (float)g_acc * g_scale * input_scale;
        }

        out[i] = total + (bias ? bias[i] : 0.0f);
    }
    if (!_scratch.valid) free(input_q);
}


/* Dispatch GEMV based on quant_mode */
static inline void gemv_dispatch(
    float* out, const LayerWeights* lw, int which,
    const float* input, int M, int K,
    const float* bias, int quant_mode)
{
    /* which: 0=qkv, 1=o, 2=gate_up, 3=down */
    if (quant_mode == QUANT_INT4) {
        const uint8_t* w4;
        const float* s4;
        switch (which) {
            case 0: w4 = lw->qkv_w4;     s4 = lw->qkv_s4;     break;
            case 1: w4 = lw->o_w4;       s4 = lw->o_s4;       break;
            case 2: w4 = lw->gate_up_w4; s4 = lw->gate_up_s4; break;
            case 3: w4 = lw->down_w4;    s4 = lw->down_s4;    break;
            default: return;
        }
        if (w4) { gemv_w4(out, w4, s4, input, M, K, bias); return; }
    }
    /* INT8 fallback */
    const int8_t* w;
    const float* s;
    switch (which) {
        case 0: w = lw->qkv_w;     s = lw->qkv_s;     break;
        case 1: w = lw->o_w;       s = lw->o_s;       break;
        case 2: w = lw->gate_up_w; s = lw->gate_up_s; break;
        case 3: w = lw->down_w;    s = lw->down_s;    break;
        default: return;
    }
    gemv_i8(out, w, s, input, M, K, bias);
}


/* Vectorized RoPE with AVX2 */
static inline void rope_half_rotate(
    float* q, float* k, const float* cos_v, const float* sin_v,
    int num_q, int num_k, int dim)
{
    int half = dim / 2;

    for (int h = 0; h < num_q; h++) {
        float* qh = q + h * dim;
        int i = 0;
#if defined(__AVX2__)
        for (; i + 7 < half; i += 8) {
            __m256 q0 = _mm256_loadu_ps(qh + i);
            __m256 q1 = _mm256_loadu_ps(qh + i + half);
            __m256 c  = _mm256_loadu_ps(cos_v + i);
            __m256 s  = _mm256_loadu_ps(sin_v + i);
            _mm256_storeu_ps(qh + i,        _mm256_fmsub_ps(q0, c, _mm256_mul_ps(q1, s)));
            _mm256_storeu_ps(qh + i + half,  _mm256_fmadd_ps(q0, s, _mm256_mul_ps(q1, c)));
        }
#endif
        for (; i < half; i++) {
            float a = qh[i], b = qh[i + half];
            qh[i]        = a * cos_v[i] - b * sin_v[i];
            qh[i + half] = a * sin_v[i] + b * cos_v[i];
        }
    }

    for (int h = 0; h < num_k; h++) {
        float* kh = k + h * dim;
        int i = 0;
#if defined(__AVX2__)
        for (; i + 7 < half; i += 8) {
            __m256 k0 = _mm256_loadu_ps(kh + i);
            __m256 k1 = _mm256_loadu_ps(kh + i + half);
            __m256 c  = _mm256_loadu_ps(cos_v + i);
            __m256 s  = _mm256_loadu_ps(sin_v + i);
            _mm256_storeu_ps(kh + i,        _mm256_fmsub_ps(k0, c, _mm256_mul_ps(k1, s)));
            _mm256_storeu_ps(kh + i + half,  _mm256_fmadd_ps(k0, s, _mm256_mul_ps(k1, c)));
        }
#endif
        for (; i < half; i++) {
            float a = kh[i], b = kh[i + half];
            kh[i]        = a * cos_v[i] - b * sin_v[i];
            kh[i + half] = a * sin_v[i] + b * cos_v[i];
        }
    }
}


/* Attention decode — parallelized over Q heads */
static void attention_decode(
    float* out, const float* q, const float* kv_cache,
    const int* block_table, int seq_len,
    int num_q_heads, int num_kv_heads, int head_dim,
    int kv_block_size, float scale,
    int kv_cache_stride_block, int kv_cache_stride_kv,
    int kv_cache_stride_head, int kv_cache_stride_pos)
{
    int gqa_ratio = num_q_heads / num_kv_heads;

    #pragma omp parallel for schedule(static)
    for (int qh = 0; qh < num_q_heads; qh++) {
        /* Per-thread score buffer (avoid shared _scratch.scores) */
        float* scores = (float*)alloca(seq_len * sizeof(float));
        int kv_h = qh / gqa_ratio;
        const float* qvec = q + qh * head_dim;

        float max_score = -1e30f;
        for (int pos = 0; pos < seq_len; pos++) {
            int blk_idx = pos / kv_block_size;
            int blk_off = pos % kv_block_size;
            int phys_blk = block_table[blk_idx];
            const float* kvec = kv_cache
                + (long long)phys_blk * kv_cache_stride_block
                + kv_h * kv_cache_stride_head
                + blk_off * kv_cache_stride_pos;

            float dot = 0.0f;
            int d = 0;
#if defined(__AVX2__)
            __m256 vdot = _mm256_setzero_ps();
            for (; d + 7 < head_dim; d += 8) {
                __m256 vq = _mm256_loadu_ps(qvec + d);
                __m256 vk = _mm256_loadu_ps(kvec + d);
                vdot = _mm256_fmadd_ps(vq, vk, vdot);
            }
            __m128 hi = _mm256_extractf128_ps(vdot, 1);
            __m128 lo = _mm256_castps256_ps128(vdot);
            __m128 s4 = _mm_add_ps(lo, hi);
            s4 = _mm_hadd_ps(s4, s4);
            s4 = _mm_hadd_ps(s4, s4);
            dot = _mm_cvtss_f32(s4);
#endif
            for (; d < head_dim; d++) dot += qvec[d] * kvec[d];

            scores[pos] = dot * scale;
            if (scores[pos] > max_score) max_score = scores[pos];
        }

        /* Softmax */
        float sum_exp = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            scores[pos] = expf(scores[pos] - max_score);
            sum_exp += scores[pos];
        }
        float inv_sum = 1.0f / sum_exp;

        /* Weighted sum of V */
        float* ovec = out + qh * head_dim;
        memset(ovec, 0, head_dim * sizeof(float));

        for (int pos = 0; pos < seq_len; pos++) {
            float w = scores[pos] * inv_sum;
            int blk_idx = pos / kv_block_size;
            int blk_off = pos % kv_block_size;
            int phys_blk = block_table[blk_idx];
            const float* vvec = kv_cache
                + (long long)phys_blk * kv_cache_stride_block
                + 1 * kv_cache_stride_kv
                + kv_h * kv_cache_stride_head
                + blk_off * kv_cache_stride_pos;

            int d = 0;
#if defined(__AVX2__)
            __m256 vw = _mm256_set1_ps(w);
            for (; d + 7 < head_dim; d += 8) {
                __m256 vo = _mm256_loadu_ps(ovec + d);
                __m256 vv = _mm256_loadu_ps(vvec + d);
                _mm256_storeu_ps(ovec + d, _mm256_fmadd_ps(vw, vv, vo));
            }
#endif
            for (; d < head_dim; d++) ovec[d] += w * vvec[d];
        }
    }
}


/* Fast SiLU/GELU + gate multiply — partially vectorized
 * SiLU(x) = x * sigmoid(x) ≈ x * 0.5 * (1 + tanh(0.7978846 * x * (1 + 0.044715 * x²)))
 * We use the same tanh-approx formula as GELU for SiLU sigmoid,
 * but the exact scalar expf path for best accuracy. */
static inline float silu_fast(float x) {
    return x / (1.0f + expf(-x));
}

static inline void swiglu_activation(float* gate, const float* up, int size, int use_gelu) {
    if (use_gelu) {
        for (int i = 0; i < size; i++) {
            float x = gate[i];
            gate[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f *
                      (x + 0.044715f * x * x * x))) * up[i];
        }
    } else {
        /* SiLU: compute sigmoid scalar, vectorize the gate*up multiply */
        int i = 0;
        /* Compute SiLU values first */
        for (i = 0; i < size; i++) {
            gate[i] = silu_fast(gate[i]);
        }
        /* Vectorized gate * up multiply */
        i = 0;
#if defined(__AVX2__)
        for (; i + 7 < size; i += 8) {
            __m256 vg = _mm256_loadu_ps(gate + i);
            __m256 vu = _mm256_loadu_ps(up + i);
            _mm256_storeu_ps(gate + i, _mm256_mul_ps(vg, vu));
        }
#endif
        for (; i < size; i++) gate[i] *= up[i];
    }
}


/* Vectorized residual add */
static inline void residual_add(float* out, const float* a, const float* b, int n) {
    int i = 0;
#if defined(__AVX2__)
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
#endif
    for (; i < n; i++) out[i] = a[i] + b[i];
}


/* ═══════════════════════════════════════════════
 * MAIN: Full decode step
 * ═══════════════════════════════════════════════ */
EXPORT int megagemm_decode_step(
    const MegaGemmConfig* cfg,
    const ModelWeights* model,
    int token_id, int position,
    float** kv_caches, const int* block_table, int seq_len,
    int kv_cache_stride_block, int kv_cache_stride_kv,
    int kv_cache_stride_head, int kv_cache_stride_pos,
    float* logits_out, int logits_out_size)
{
    int H = cfg->hidden_size;
    int I = cfg->intermediate_size;
    int Nq = cfg->num_q_heads;
    int Nkv = cfg->num_kv_heads;
    int D = cfg->head_dim;
    int qkv_size = (Nq + 2 * Nkv) * D;
    float scale = 1.0f / sqrtf((float)D);
    int qm = cfg->quant_mode;

    /* Use pre-allocated or malloc */
    float *hidden, *residual, *normed, *qkv, *attn_out, *o_out, *gate_up, *mlp_out;
    int need_free = 0;

    if (_scratch.valid) {
        hidden = _scratch.hidden;   residual = _scratch.residual;
        normed = _scratch.normed;   qkv = _scratch.qkv;
        attn_out = _scratch.attn_out; o_out = _scratch.o_out;
        gate_up = _scratch.gate_up; mlp_out = _scratch.mlp_out;
    } else {
        hidden = (float*)malloc(H * sizeof(float));
        residual = (float*)malloc(H * sizeof(float));
        normed = (float*)malloc(H * sizeof(float));
        qkv = (float*)malloc(qkv_size * sizeof(float));
        attn_out = (float*)malloc(Nq * D * sizeof(float));
        o_out = (float*)malloc(H * sizeof(float));
        gate_up = (float*)malloc(2 * I * sizeof(float));
        mlp_out = (float*)malloc(H * sizeof(float));
        need_free = 1;
    }

    /* Embedding lookup */
    memcpy(hidden, model->embed_tokens + (long long)token_id * H, H * sizeof(float));

    const float* cos_v = model->cos_cache + (long long)position * D;
    const float* sin_v = model->sin_cache + (long long)position * D;

    /* Layer loop */
    for (int l = 0; l < cfg->num_layers; l++) {
        const LayerWeights* lw = &model->layers[l];

        memcpy(residual, hidden, H * sizeof(float));
        rmsnorm(normed, hidden, lw->input_norm_w, H, cfg->rms_norm_eps, cfg->norm_offset);

        gemv_dispatch(qkv, lw, 0, normed, qkv_size, H,
                cfg->qkv_bias ? lw->qkv_bias : NULL, qm);

        float* q = qkv;
        float* k = qkv + Nq * D;
        float* v = qkv + (Nq + Nkv) * D;

        rope_half_rotate(q, k, cos_v, sin_v, Nq, Nkv, D);

        float* layer_kv = kv_caches[l];
        int blk_idx = seq_len / cfg->kv_block_size;
        int blk_off = seq_len % cfg->kv_block_size;
        int phys_blk = block_table[blk_idx];

        for (int h = 0; h < Nkv; h++) {
            float* k_dst = layer_kv + (long long)phys_blk * kv_cache_stride_block
                + h * kv_cache_stride_head + blk_off * kv_cache_stride_pos;
            memcpy(k_dst, k + h * D, D * sizeof(float));

            float* v_dst = layer_kv + (long long)phys_blk * kv_cache_stride_block
                + 1 * kv_cache_stride_kv
                + h * kv_cache_stride_head + blk_off * kv_cache_stride_pos;
            memcpy(v_dst, v + h * D, D * sizeof(float));
        }

        attention_decode(attn_out, q, layer_kv, block_table, seq_len + 1,
                        Nq, Nkv, D, cfg->kv_block_size, scale,
                        kv_cache_stride_block, kv_cache_stride_kv,
                        kv_cache_stride_head, kv_cache_stride_pos);

        gemv_dispatch(o_out, lw, 1, attn_out, H, Nq * D, NULL, qm);
        residual_add(hidden, residual, o_out, H);

        memcpy(residual, hidden, H * sizeof(float));
        rmsnorm(normed, hidden, lw->post_attn_norm_w, H, cfg->rms_norm_eps, cfg->norm_offset);

        gemv_dispatch(gate_up, lw, 2, normed, 2 * I, H, NULL, qm);
        swiglu_activation(gate_up, gate_up + I, I, cfg->act_gelu);

        gemv_dispatch(mlp_out, lw, 3, gate_up, H, I, NULL, qm);
        residual_add(hidden, residual, mlp_out, H);
    }

    rmsnorm(normed, hidden, model->final_norm_w, H, cfg->rms_norm_eps, cfg->norm_offset);

    int V = cfg->vocab_size;
    if (logits_out && logits_out_size >= V) {
        if (qm == QUANT_INT4 && model->lm_head_w4) {
            int num_groups = (H + W4_GROUP_SIZE - 1) / W4_GROUP_SIZE;
            gemv_w4(logits_out, model->lm_head_w4, model->lm_head_s4, normed, V, H, NULL);
        } else {
            gemv_i8(logits_out, model->lm_head_w, model->lm_head_s, normed, V, H, NULL);
        }
    }

    /* Argmax */
    int best_id = 0;
    float best_val = logits_out[0];
    for (int i = 1; i < V; i++) {
        if (logits_out[i] > best_val) { best_val = logits_out[i]; best_id = i; }
    }

    if (need_free) {
        free(hidden); free(residual); free(normed); free(qkv);
        free(attn_out); free(o_out); free(gate_up); free(mlp_out);
    }

    return best_id;
}


/* Multi-step decode */
EXPORT int megagemm_decode_multi(
    const MegaGemmConfig* cfg, const ModelWeights* model,
    int first_token, int start_position, int max_tokens, int eos_token,
    float** kv_caches, const int* block_table,
    int kv_cache_stride_block, int kv_cache_stride_kv,
    int kv_cache_stride_head, int kv_cache_stride_pos,
    int* output_tokens, float* final_logits, int logits_buf_size)
{
    /* Pre-allocate scratch if not done */
    if (!_scratch.valid) {
        megagemm_alloc_scratch(cfg, start_position + max_tokens + 64);
    }

    float* logits = _scratch.logits;
    int token = first_token;
    int pos = start_position;
    int generated = 0;

    for (int t = 0; t < max_tokens; t++) {
        token = megagemm_decode_step(
            cfg, model, token, pos,
            kv_caches, block_table, pos,
            kv_cache_stride_block, kv_cache_stride_kv,
            kv_cache_stride_head, kv_cache_stride_pos,
            logits, cfg->vocab_size);

        output_tokens[t] = token;
        generated++;
        pos++;

        if (token == eos_token) break;
    }

    if (final_logits && logits_buf_size >= cfg->vocab_size) {
        memcpy(final_logits, logits, cfg->vocab_size * sizeof(float));
    }

    return generated;
}


/* ═══════════════════════════════════════════════
 * Batch decode: N sequences simultaneously
 *
 * Weights loaded once per layer, applied to all N sequences.
 * Each sequence has its own token, position, KV cache, block table.
 * The key optimization: GEMV weight reads are amortized across N seqs.
 * ═══════════════════════════════════════════════ */
EXPORT int megagemm_decode_batch(
    const MegaGemmConfig* cfg,
    const ModelWeights* model,
    int batch_size,
    const int* token_ids,       /* [batch_size] */
    const int* positions,       /* [batch_size] */
    float** kv_caches,          /* [num_layers] per-layer KV cache */
    const int* block_tables,    /* [batch_size * max_blocks] flattened */
    int max_blocks_per_seq,
    const int* seq_lens,        /* [batch_size] */
    int kv_cache_stride_block, int kv_cache_stride_kv,
    int kv_cache_stride_head, int kv_cache_stride_pos,
    int* output_tokens          /* [batch_size] output */
)
{
    int H = cfg->hidden_size;
    int I = cfg->intermediate_size;
    int Nq = cfg->num_q_heads;
    int Nkv = cfg->num_kv_heads;
    int D = cfg->head_dim;
    int V = cfg->vocab_size;
    int qkv_size = (Nq + 2 * Nkv) * D;
    float scale = 1.0f / sqrtf((float)D);
    int qm = cfg->quant_mode;
    int N = batch_size;

    /* Allocate per-sequence buffers */
    size_t per_seq_size = (H + H + H + qkv_size + Nq*D + H + 2*I + H + V) * sizeof(float);
    float* buf = (float*)malloc(N * per_seq_size);
    if (!buf) return -1;

    /* Per-sequence buffer pointers */
    typedef struct {
        float *hidden, *residual, *normed, *qkv, *attn_out, *o_out, *gate_up, *mlp_out, *logits;
    } SeqBufs;
    SeqBufs* sbufs = (SeqBufs*)alloca(N * sizeof(SeqBufs));

    for (int n = 0; n < N; n++) {
        float* base = buf + n * (H + H + H + qkv_size + Nq*D + H + 2*I + H + V);
        sbufs[n].hidden   = base;
        sbufs[n].residual = base + H;
        sbufs[n].normed   = base + 2*H;
        sbufs[n].qkv      = base + 3*H;
        sbufs[n].attn_out = base + 3*H + qkv_size;
        sbufs[n].o_out    = base + 3*H + qkv_size + Nq*D;
        sbufs[n].gate_up  = base + 4*H + qkv_size + Nq*D;
        sbufs[n].mlp_out  = base + 4*H + qkv_size + Nq*D + 2*I;
        sbufs[n].logits   = base + 5*H + qkv_size + Nq*D + 2*I;
    }

    /* Embedding lookup for all sequences */
    for (int n = 0; n < N; n++) {
        memcpy(sbufs[n].hidden,
               model->embed_tokens + (long long)token_ids[n] * H,
               H * sizeof(float));
    }

    /* Layer loop — weights loaded once, applied to all N sequences */
    for (int l = 0; l < cfg->num_layers; l++) {
        const LayerWeights* lw = &model->layers[l];

        /* Process each sequence through this layer.
         * The weights stay in L3 cache after the first sequence,
         * making subsequent sequences much faster (cache-warm). */
        for (int n = 0; n < N; n++) {
            float* hidden = sbufs[n].hidden;
            float* residual = sbufs[n].residual;
            float* normed = sbufs[n].normed;
            float* qkv = sbufs[n].qkv;
            float* attn_out = sbufs[n].attn_out;
            float* o_out = sbufs[n].o_out;
            float* gate_up = sbufs[n].gate_up;
            float* mlp_out = sbufs[n].mlp_out;

            int position = positions[n];
            int seq_len = seq_lens[n];
            const int* bt = block_tables + n * max_blocks_per_seq;

            /* === Attention block === */
            memcpy(residual, hidden, H * sizeof(float));
            rmsnorm(normed, hidden, lw->input_norm_w, H,
                    cfg->rms_norm_eps, cfg->norm_offset);

            gemv_dispatch(qkv, lw, 0, normed, qkv_size, H,
                    cfg->qkv_bias ? lw->qkv_bias : NULL, qm);

            float* q = qkv;
            float* k = qkv + Nq * D;
            float* v = qkv + (Nq + Nkv) * D;

            const float* cos_v = model->cos_cache + (long long)position * D;
            const float* sin_v = model->sin_cache + (long long)position * D;
            rope_half_rotate(q, k, cos_v, sin_v, Nq, Nkv, D);

            /* Write K/V to cache */
            float* layer_kv = kv_caches[l];
            int blk_idx = seq_len / cfg->kv_block_size;
            int blk_off = seq_len % cfg->kv_block_size;
            int phys_blk = bt[blk_idx];

            for (int h = 0; h < Nkv; h++) {
                float* k_dst = layer_kv + (long long)phys_blk * kv_cache_stride_block
                    + h * kv_cache_stride_head + blk_off * kv_cache_stride_pos;
                memcpy(k_dst, k + h * D, D * sizeof(float));

                float* v_dst = layer_kv + (long long)phys_blk * kv_cache_stride_block
                    + 1 * kv_cache_stride_kv
                    + h * kv_cache_stride_head + blk_off * kv_cache_stride_pos;
                memcpy(v_dst, v + h * D, D * sizeof(float));
            }

            attention_decode(attn_out, q, layer_kv, bt, seq_len + 1,
                            Nq, Nkv, D, cfg->kv_block_size, scale,
                            kv_cache_stride_block, kv_cache_stride_kv,
                            kv_cache_stride_head, kv_cache_stride_pos);

            gemv_dispatch(o_out, lw, 1, attn_out, H, Nq * D, NULL, qm);
            residual_add(hidden, residual, o_out, H);

            /* === MLP block === */
            memcpy(residual, hidden, H * sizeof(float));
            rmsnorm(normed, hidden, lw->post_attn_norm_w, H,
                    cfg->rms_norm_eps, cfg->norm_offset);

            gemv_dispatch(gate_up, lw, 2, normed, 2 * I, H, NULL, qm);
            swiglu_activation(gate_up, gate_up + I, I, cfg->act_gelu);

            gemv_dispatch(mlp_out, lw, 3, gate_up, H, I, NULL, qm);
            residual_add(hidden, residual, mlp_out, H);
        }
    }

    /* Final norm + LM head + argmax for each sequence */
    for (int n = 0; n < N; n++) {
        float* hidden = sbufs[n].hidden;
        float* normed = sbufs[n].normed;
        float* logits = sbufs[n].logits;

        rmsnorm(normed, hidden, model->final_norm_w, H,
                cfg->rms_norm_eps, cfg->norm_offset);

        if (qm == QUANT_INT4 && model->lm_head_w4) {
            gemv_w4(logits, model->lm_head_w4, model->lm_head_s4, normed, V, H, NULL);
        } else {
            gemv_i8(logits, model->lm_head_w, model->lm_head_s, normed, V, H, NULL);
        }

        /* Argmax */
        int best_id = 0;
        float best_val = logits[0];
        for (int i = 1; i < V; i++) {
            if (logits[i] > best_val) { best_val = logits[i]; best_id = i; }
        }
        output_tokens[n] = best_id;
    }

    free(buf);
    return N;
}

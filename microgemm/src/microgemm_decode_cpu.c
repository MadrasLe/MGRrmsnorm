#include "microgemm/microgemm_decode.h"

#include "microgemm/microgemm_cpu.h"
#include "microgemm/microgemm_platform.h"

#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline float microgemm_config_embedding_multiplier(const microgemm_config* config);
static inline float microgemm_config_residual_multiplier(const microgemm_config* config);
static inline float microgemm_config_logits_scaling(const microgemm_config* config);
static void microgemm_scale_inplace(float* values, int count, float scale);

static inline int8_t microgemm_i4_unpack_nibble(uint8_t packed, int high_nibble) {
    uint8_t nibble = high_nibble ? (uint8_t)(packed >> 4) : (uint8_t)(packed & 0x0fu);
    return (int8_t)(nibble >= 8u ? (int)nibble - 16 : (int)nibble);
}

static inline int8_t microgemm_i4_row_value(const uint8_t* row, int col) {
    return microgemm_i4_unpack_nibble(row[col >> 1], col & 1);
}

static int microgemm_parse_env_flag_value(const char* value, int default_value) {
    if (value == NULL || value[0] == '\0') {
        return default_value;
    }
    return !(value[0] == '0'
        || value[0] == 'n'
        || value[0] == 'N'
        || value[0] == 'f'
        || value[0] == 'F');
}

static int microgemm_i4_row_tile8_split_enabled(void) {
    const char* specific = getenv("MICROGEMM_I4_ROW_TILE8_SPLIT");
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 1);
    }
    return microgemm_parse_env_flag_value(getenv("MICROGEMM_I4_TILE8_SPLIT"), 1);
}

static int microgemm_i4_row_pair_tile4_enabled_for(int batch, int rows, int cols) {
    const char* specific = getenv("MICROGEMM_I4_ROW_PAIR_TILE4");
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0);
    }
    (void)batch;
    (void)rows;
    (void)cols;
    return 0;
}

static int microgemm_i4g_row_pair_tile4_enabled_for(int batch, int rows, int cols) {
    const char* specific = getenv("MICROGEMM_I4G_ROW_PAIR_TILE4");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I4_ROW_PAIR_TILE4");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 4
            && rows >= 2
            && cols > rows;
    }
    return batch >= 4 && rows >= 128 && cols > rows;
}

static int microgemm_i8g_row_pair_tile4_enabled_for(int batch, int rows, int cols) {
    const char* all_specific = getenv("MICROGEMM_I8G_ROW_PAIR_TILE4_ALL");
    const char* specific = getenv("MICROGEMM_I8G_ROW_PAIR_TILE4");
    if (all_specific == NULL || all_specific[0] == '\0') {
        all_specific = getenv("MICROGEMM_GROUPWISE_I8_ROW_PAIR_TILE4_ALL");
    }
    if (all_specific != NULL && all_specific[0] != '\0') {
        return microgemm_parse_env_flag_value(all_specific, 0)
            && batch >= 4
            && rows >= 2
            && cols > 0;
    }
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_ROW_PAIR_TILE4");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 4
            && rows >= 2
            && cols > rows;
    }
    return batch >= 4 && rows >= 128 && cols > rows;
}

static int microgemm_i8g_saturation_safe_enabled_for(int batch, int rows, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_SATURATION_SAFE");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_SATURATION_SAFE");
    }
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_I8G_SAT_SAFE");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch > 0
            && rows > 0
            && cols > 0;
    }
    (void)batch;
    (void)rows;
    (void)cols;
    return 0;
}

static int microgemm_i8g_sat_safe_row_pair_tile4_enabled_for(int batch, int rows, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_SAT_SAFE_ROW_PAIR_TILE4");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_I8G_ROW_PAIR_TILE4_SAFE");
    }
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_ROW_PAIR_TILE4_SAFE");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 4
            && rows >= 2
            && cols > 0;
    }
    return batch >= 4
        && rows >= 2
        && cols > 0;
}

static int microgemm_i8g_gate_safe_fused_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_GATE_SAFE_FUSED");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_GATE_SAFE_FUSED");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && microgemm_i8g_saturation_safe_enabled_for(batch, intermediate, cols)
            && batch > 0
            && intermediate > 0
            && cols > 0;
    }
    return microgemm_i8g_saturation_safe_enabled_for(batch, intermediate, cols)
        && batch >= 4
        && intermediate > 0
        && cols > 0;
}

static int microgemm_i8g_gate_safe_combined_tile4_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_GATE_SAFE_COMBINED_TILE4");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_I8G_GATE_SAFE_COMBINED");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && microgemm_i8g_saturation_safe_enabled_for(batch, intermediate, cols)
            && batch >= 4
            && intermediate > 0
            && cols > 0;
    }
    return microgemm_i8g_saturation_safe_enabled_for(batch, intermediate, cols)
        && batch >= 4
        && intermediate > 0
        && cols > 0;
}

static int microgemm_i8g_gate_safe_combined_tile8_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_GATE_SAFE_COMBINED_TILE8");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_I8G_GATE_SAFE_COMBO_TILE8");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && microgemm_i8g_saturation_safe_enabled_for(batch, intermediate, cols)
            && batch >= 8
            && intermediate > 0
            && cols > 0;
    }
    (void)batch;
    (void)intermediate;
    (void)cols;
    return 0;
}

static int microgemm_i8g_gate_tile8_explicit_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_GATE_UP_TILE8_EXPLICIT");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_GATE_UP_TILE8_EXPLICIT");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 8
            && intermediate >= 512
            && cols >= 512;
    }
    return batch >= 8 && intermediate >= 512 && cols >= 512;
}

static int microgemm_i8g_gate_tile8_aligned128_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_GATE8_ALIGNED128");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_GATE8_ALIGNED128");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 8
            && intermediate >= 512
            && cols >= 512
            && (cols % (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE) == 0;
    }
    return 0;
}

static int microgemm_i8g_gate_tile8_biased_input_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_GATE8_BIASED_INPUT");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_GATE8_BIASED_INPUT");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 8
            && (batch % 8) == 0
            && intermediate >= 512
            && cols >= 512
            && (cols % (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE) == 0;
    }
    return batch >= 8
        && (batch % 8) == 0
        && intermediate >= 512
        && cols >= 512
        && (cols % (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE) == 0;
}

static int microgemm_i8g_gate_prefetch_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_GATE_PREFETCH");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_GATE_PREFETCH");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 8
            && intermediate >= 512
            && cols >= 512;
    }
    (void)batch;
    (void)intermediate;
    (void)cols;
    return 0;
}

static int microgemm_i8g_gate_pair_tile4_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_GATE_PAIR4");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_GATE_PAIR4");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 8
            && (batch % 8) == 0
            && intermediate >= 512
            && cols >= 512
            && (cols % (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE) == 0;
    }
    return batch >= 8
        && (batch % 8) == 0
        && intermediate >= 512
        && cols >= 512
        && (cols % (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE) == 0;
}

static int microgemm_i8g_gate_pair_tile4_unroll64_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_GATE_PAIR4_UNROLL64");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_GATE_PAIR4_UNROLL64");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 8
            && (batch % 8) == 0
            && intermediate >= 512
            && cols >= 512
            && (cols % (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE) == 0;
    }
    (void)batch;
    (void)intermediate;
    (void)cols;
    return 0;
}

static int microgemm_i8g_gate_pair_tile4_unroll128_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_GATE_PAIR4_UNROLL128");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_GATE_PAIR4_UNROLL128");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 8
            && (batch % 8) == 0
            && intermediate >= 512
            && cols >= 512
            && (cols % (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE) == 0;
    }
    (void)batch;
    (void)intermediate;
    (void)cols;
    return 0;
}

static int microgemm_i8g_gate_pair_tile8_splitpass_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_GATE_PAIR8_SPLITPASS");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_GATE_PAIR8_SPLITPASS");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 8
            && (batch % 8) == 0
            && intermediate >= 512
            && cols >= 512
            && (cols % (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE) == 0;
    }
    (void)batch;
    (void)intermediate;
    (void)cols;
    return 0;
}

static int microgemm_i8g_lm_head_scores8_enabled_for(int batch, int rows, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_LM_HEAD_SCORES8");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_LM_HEAD_SCORES8");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 8
            && rows >= 96
            && cols >= 512;
    }
    return batch >= 8 && rows >= 96 && cols >= 512;
}

static int microgemm_i8g_lm_head_row_pair_tile4_enabled_for(int batch, int rows, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_LM_HEAD_ROWPAIR4");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_LM_HEAD_ROWPAIR4");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 4
            && rows >= 96
            && cols >= 512;
    }
    return 0;
}

static int microgemm_i8g_sat_safe_lm_head_row_pair_tile4_enabled_for(int batch, int rows, int cols) {
    const char* specific = getenv("MICROGEMM_I8G_SAT_SAFE_LM_HEAD_ROWPAIR4");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_I8G_LM_HEAD_ROWPAIR4_SAFE");
    }
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_LM_HEAD_ROWPAIR4_SAFE");
    }
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_I8G_LM_HEAD_ROWPAIR4");
    }
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_I8_LM_HEAD_ROWPAIR4");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && microgemm_i8g_saturation_safe_enabled_for(batch, rows, cols)
            && batch >= 4
            && rows >= 96
            && cols >= 512;
    }
    return microgemm_i8g_saturation_safe_enabled_for(batch, rows, cols)
        && batch >= 4
        && rows >= 96
        && cols >= 512;
}

#define MICROGEMM_LM_HEAD_STACK_BEST_LIMIT 512

static int microgemm_lm_head_stack_best_enabled_for(int batch, int thread_count) {
    const char* specific = getenv("MICROGEMM_LM_HEAD_STACK_BEST");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_LM_HEAD_STACK_BEST");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch > 0
            && thread_count > 0
            && (size_t)batch * (size_t)thread_count <= (size_t)MICROGEMM_LM_HEAD_STACK_BEST_LIMIT;
    }
    return 0;
}

static int microgemm_i4_gate_tile8_split_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I4_GATE_TILE8_SPLIT");
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0);
    }
    specific = getenv("MICROGEMM_I4_TILE8_SPLIT");
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0);
    }
    (void)batch;
    (void)intermediate;
    (void)cols;
    return 0;
}

static int microgemm_i4_gate_tile8_group4_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I4_GATE_TILE8_GROUP4");
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0);
    }
    (void)batch;
    (void)intermediate;
    (void)cols;
    return 0;
}

static int microgemm_i4_gate_tile8_fused_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I4_GATE_TILE8_FUSED");
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0);
    }
    (void)batch;
    (void)intermediate;
    (void)cols;
    return 0;
}

static int microgemm_i8_gate_tile8_group4_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8_GATE_TILE8_GROUP4");
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0);
    }
    (void)intermediate;
    return batch >= 8 && cols >= 4096;
}

static int microgemm_i8_gate_tile4_unroll64_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8_GATE_TILE4_UNROLL64");
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0);
    }
    (void)batch;
    (void)intermediate;
    (void)cols;
    return 0;
}

static int microgemm_i8_gate_tile8_fused_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8_GATE_TILE8_FUSED");
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0);
    }
    (void)batch;
    (void)intermediate;
    (void)cols;
    return 0;
}

static int microgemm_i8_gate_tile8_unroll64_enabled_for(int batch, int intermediate, int cols) {
    const char* specific = getenv("MICROGEMM_I8_GATE_TILE8_UNROLL64");
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0);
    }
    (void)intermediate;
    return batch >= 8 && cols >= 4096;
}

static int microgemm_i8_down_tile8_enabled_for(int batch, int rows, int cols) {
    const char* specific = getenv("MICROGEMM_I8_DOWN_TILE8");
    if (specific != NULL && specific[0] != '\0') {
        if (!microgemm_parse_env_flag_value(specific, 0)) {
            return 0;
        }
        return batch >= 8 && rows >= 512 && rows <= 8192 && cols > rows;
    }
    return 0;
}

static int microgemm_prefill_parallel_attention_enabled_for(
    int batch,
    int base_seq_len,
    int max_seq_len,
    int num_q_heads
) {
    const char* specific = getenv("MICROGEMM_PREFILL_PARALLEL_ATTENTION");
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && batch >= 4
            && max_seq_len > 0
            && num_q_heads > 0;
    }
    (void)batch;
    (void)base_seq_len;
    (void)max_seq_len;
    (void)num_q_heads;
    return 0;
}

static int microgemm_rmsnorm_prequant_enabled_for(uint32_t quant_mode, int batch, int width) {
    const char* specific = getenv("MICROGEMM_RMSNORM_PREQUANT");
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && (quant_mode == MICROGEMM_QUANT_INT8 || quant_mode == MICROGEMM_QUANT_INT8G128)
            && batch > 0
            && width > 0;
    }
    (void)quant_mode;
    (void)batch;
    (void)width;
    return 0;
}

static int microgemm_swiglu_down_prequant_enabled_for(uint32_t quant_mode, int batch, int intermediate) {
    const char* specific = getenv("MICROGEMM_SWIGLU_DOWN_PREQUANT");
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_MLP_DOWN_PREQUANT");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && (quant_mode == MICROGEMM_QUANT_INT8 || quant_mode == MICROGEMM_QUANT_INT8G128)
            && batch > 0
            && intermediate > 0;
    }
    (void)quant_mode;
    (void)batch;
    (void)intermediate;
    return 0;
}

static int microgemm_max_worker_threads(void) {
#ifdef _OPENMP
    int thread_count = omp_get_max_threads();
    return thread_count > 0 ? thread_count : 1;
#else
    return 1;
#endif
}

static int microgemm_groupwise_gate_up_fused_enabled_for(
    uint32_t quant_mode,
    int batch,
    int intermediate,
    int cols
) {
    const char* specific = getenv("MICROGEMM_GROUPWISE_GATE_UP_FUSED");
    if (quant_mode == MICROGEMM_QUANT_INT8G128
            && microgemm_i8g_saturation_safe_enabled_for(batch, intermediate, cols)) {
        return 0;
    }
    if (specific == NULL || specific[0] == '\0') {
        specific = getenv("MICROGEMM_GROUPWISE_GATE_FUSED");
    }
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0);
    }
    if (quant_mode == MICROGEMM_QUANT_INT4G128 || quant_mode == MICROGEMM_QUANT_INT8G128) {
        return batch >= 4 && intermediate >= 512 && cols >= 512;
    }
    (void)intermediate;
    (void)cols;
    return 0;
}

static int microgemm_groupwise_compact_mlp_stride_enabled_for(
    uint32_t quant_mode,
    int batch,
    int intermediate,
    int cols
) {
    const char* specific = getenv("MICROGEMM_GROUPWISE_COMPACT_MLP_STRIDE");
    if (specific != NULL && specific[0] != '\0') {
        return microgemm_parse_env_flag_value(specific, 0)
            && quant_mode == MICROGEMM_QUANT_INT4G128
            && batch >= 4
            && intermediate >= 512
            && cols >= 512;
    }
    (void)quant_mode;
    (void)batch;
    (void)intermediate;
    (void)cols;
    return 0;
}

static int microgemm_quant_mode_is_i4_storage(uint32_t quant_mode) {
    return quant_mode == MICROGEMM_QUANT_INT4
        || quant_mode == MICROGEMM_QUANT_INT4G128;
}

static int microgemm_quant_mode_is_i8_storage(uint32_t quant_mode) {
    return quant_mode == MICROGEMM_QUANT_INT8
        || quant_mode == MICROGEMM_QUANT_INT8G128;
}

static int microgemm_quant_mode_is_groupwise(uint32_t quant_mode) {
    return quant_mode == MICROGEMM_QUANT_INT8G128
        || quant_mode == MICROGEMM_QUANT_INT4G128;
}

static int microgemm_quant_group_count_int(int cols) {
    return (cols + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE - 1)
        / (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
}

static float microgemm_silu(float x);
static inline void microgemm_swiglu_absmax_update_rows(
    float* absmax,
    const float* out,
    int out_col,
    int row_count,
    int out_stride,
    int batch_offset,
    int tile
);

static int microgemm_linear_delta_vec_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = microgemm_parse_env_flag_value(getenv("MICROGEMM_LINEAR_DELTA_VEC"), 1);
    }
    return cached;
}

#if MICROGEMM_CPU_ARM64_NEON
static float microgemm_neon_hsum_f32(float32x4_t v) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vaddvq_f32(v);
#else
    float tmp[4];
    vst1q_f32(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
}

static int32_t microgemm_neon_hsum_s32(int32x4_t v) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vaddvq_s32(v);
#else
    int32_t tmp[4];
    vst1q_s32(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
}
#endif

#if MICROGEMM_CPU_X86_AVX2
static inline int32_t microgemm_avx2_hsum_epi32(__m256i v) {
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i s4 = _mm_add_epi32(lo, hi);
    s4 = _mm_hadd_epi32(s4, s4);
    s4 = _mm_hadd_epi32(s4, s4);
    return _mm_cvtsi128_si32(s4);
}

static inline float microgemm_avx2_hsum_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    return _mm_cvtss_f32(s4);
}

static inline int32_t microgemm_avx2_dot_i8_signed_i8_safe(
    const int8_t* row,
    const int8_t* input_q,
    int cols
) {
    __m256i vacc0 = _mm256_setzero_si256();
    __m256i vacc1 = _mm256_setzero_si256();
    int j = 0;
    int32_t acc;

    for (; j + 31 < cols; j += 32) {
        __m128i q0 = _mm_loadu_si128((const __m128i*)(input_q + j));
        __m128i q1 = _mm_loadu_si128((const __m128i*)(input_q + j + 16));
        __m128i w0 = _mm_loadu_si128((const __m128i*)(row + j));
        __m128i w1 = _mm_loadu_si128((const __m128i*)(row + j + 16));
        __m256i q0_16 = _mm256_cvtepi8_epi16(q0);
        __m256i q1_16 = _mm256_cvtepi8_epi16(q1);
        __m256i w0_16 = _mm256_cvtepi8_epi16(w0);
        __m256i w1_16 = _mm256_cvtepi8_epi16(w1);
        vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(q0_16, w0_16));
        vacc1 = _mm256_add_epi32(vacc1, _mm256_madd_epi16(q1_16, w1_16));
    }

    acc = microgemm_avx2_hsum_epi32(_mm256_add_epi32(vacc0, vacc1));
    for (; j < cols; ++j) {
        acc += (int32_t)input_q[j] * (int32_t)row[j];
    }
    return acc;
}

#if MICROGEMM_CPU_X86_FMA
#define MICROGEMM_AVX2_FMADD_PS(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else
#define MICROGEMM_AVX2_FMADD_PS(a, b, c) _mm256_add_ps((c), _mm256_mul_ps((a), (b)))
#endif

static inline __m256i microgemm_avx2_unpack_i4_32(const uint8_t* packed) {
    const __m128i bytes = _mm_loadu_si128((const __m128i*)packed);
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i sign = _mm_set1_epi8(0x08);
    __m128i low = _mm_and_si128(bytes, mask);
    __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    __m128i lo;
    __m128i hi;

    low = _mm_sub_epi8(_mm_xor_si128(low, sign), sign);
    high = _mm_sub_epi8(_mm_xor_si128(high, sign), sign);
    lo = _mm_unpacklo_epi8(low, high);
    hi = _mm_unpackhi_epi8(low, high);
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static inline int32_t microgemm_avx2_dot_i4_biased_i8(
    const uint8_t* row,
    const int8_t* input_q,
    int cols,
    int row_sum
) {
    const __m256i vone16 = _mm256_set1_epi16(1);
    __m256i vacc0 = _mm256_setzero_si256();
    __m256i vacc1 = _mm256_setzero_si256();
    int j = 0;
    int32_t acc;

    for (; j + 63 < cols; j += 64) {
        __m256i vw0 = microgemm_avx2_unpack_i4_32(row + (j >> 1));
        __m256i vw1 = microgemm_avx2_unpack_i4_32(row + ((j + 32) >> 1));
        __m256i vi0 = _mm256_loadu_si256((const __m256i*)(input_q + j));
        __m256i vi1 = _mm256_loadu_si256((const __m256i*)(input_q + j + 32));
        __m256i prod0;
        __m256i prod1;

        prod0 = _mm256_maddubs_epi16(vi0, vw0);
        prod1 = _mm256_maddubs_epi16(vi1, vw1);
        vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(prod0, vone16));
        vacc1 = _mm256_add_epi32(vacc1, _mm256_madd_epi16(prod1, vone16));
    }
    for (; j + 31 < cols; j += 32) {
        __m256i vw = microgemm_avx2_unpack_i4_32(row + (j >> 1));
        __m256i vi = _mm256_loadu_si256((const __m256i*)(input_q + j));
        __m256i prod;

        prod = _mm256_maddubs_epi16(vi, vw);
        vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(prod, vone16));
    }

    acc = microgemm_avx2_hsum_epi32(_mm256_add_epi32(vacc0, vacc1)) - 128 * row_sum;
    for (; j < cols; ++j) {
        acc += (int32_t)(uint8_t)input_q[j] * (int32_t)microgemm_i4_row_value(row, j);
    }
    return acc;
}

static inline void microgemm_avx2_i4_batch_row_tile(
    float* out,
    int out_row,
    int rows,
    const uint8_t* row,
    float row_scale,
    int row_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int tile,
    int cols,
    const float* bias
) {
    const __m256i vone16 = _mm256_set1_epi16(1);
    __m256i vacc0[8];
    __m256i vacc1[8];
    const int8_t* q_ptrs[8];
    float q_scales[8];
    int bb;
    int j = 0;

    for (bb = 0; bb < tile; ++bb) {
        vacc0[bb] = _mm256_setzero_si256();
        vacc1[bb] = _mm256_setzero_si256();
        q_ptrs[bb] = input_q + (size_t)(batch_offset + bb) * cols;
        q_scales[bb] = input_scales[batch_offset + bb];
    }

    for (; j + 63 < cols; j += 64) {
        __m256i vw0 = microgemm_avx2_unpack_i4_32(row + (j >> 1));
        __m256i vw1 = microgemm_avx2_unpack_i4_32(row + ((j + 32) >> 1));
        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = q_ptrs[bb];
            __m256i vi0 = _mm256_loadu_si256((const __m256i*)(q + j));
            __m256i vi1 = _mm256_loadu_si256((const __m256i*)(q + j + 32));
                __m256i prod0;
                __m256i prod1;

                prod0 = _mm256_maddubs_epi16(vi0, vw0);
                prod1 = _mm256_maddubs_epi16(vi1, vw1);
                vacc0[bb] = _mm256_add_epi32(vacc0[bb], _mm256_madd_epi16(prod0, vone16));
            vacc1[bb] = _mm256_add_epi32(vacc1[bb], _mm256_madd_epi16(prod1, vone16));
        }
    }
    for (; j + 31 < cols; j += 32) {
        __m256i vw = microgemm_avx2_unpack_i4_32(row + (j >> 1));
        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = q_ptrs[bb];
            __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j));
            __m256i prod;

            prod = _mm256_maddubs_epi16(vi, vw);
            vacc0[bb] = _mm256_add_epi32(vacc0[bb], _mm256_madd_epi16(prod, vone16));
        }
    }

    for (bb = 0; bb < tile; ++bb) {
        const int8_t* q = q_ptrs[bb];
        int32_t acc = microgemm_avx2_hsum_epi32(_mm256_add_epi32(vacc0[bb], vacc1[bb]))
            - 128 * row_sum;
        int tail;
        for (tail = j; tail < cols; ++tail) {
            acc += (int32_t)(uint8_t)q[tail] * (int32_t)microgemm_i4_row_value(row, tail);
        }
        out[(size_t)(batch_offset + bb) * rows + out_row] =
            (float)acc * row_scale * q_scales[bb] + (bias ? bias[out_row] : 0.0f);
    }
}

static inline void microgemm_avx2_i4_batch_row_tile8_split(
    float* out,
    int out_row,
    int rows,
    const uint8_t* row,
    float row_scale,
    int row_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    const float* bias
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const int8_t* q4 = input_q + (size_t)(batch_offset + 4) * cols;
    const int8_t* q5 = input_q + (size_t)(batch_offset + 5) * cols;
    const int8_t* q6 = input_q + (size_t)(batch_offset + 6) * cols;
    const int8_t* q7 = input_q + (size_t)(batch_offset + 7) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float qs4 = input_scales[batch_offset + 4];
    const float qs5 = input_scales[batch_offset + 5];
    const float qs6 = input_scales[batch_offset + 6];
    const float qs7 = input_scales[batch_offset + 7];
    const __m256i vone16 = _mm256_set1_epi16(1);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    __m256i acc4 = _mm256_setzero_si256();
    __m256i acc5 = _mm256_setzero_si256();
    __m256i acc6 = _mm256_setzero_si256();
    __m256i acc7 = _mm256_setzero_si256();
    const float bias_value = bias ? bias[out_row] : 0.0f;
    int j = 0;

#define MICROGEMM_I4_ROW_TILE8_ACC(IDX, ROW) do { \
        __m256i vi = _mm256_loadu_si256((const __m256i*)(q##IDX + j)); \
        __m256i prod; \
        prod = _mm256_maddubs_epi16(vi, ROW); \
        acc##IDX = _mm256_add_epi32(acc##IDX, _mm256_madd_epi16(prod, vone16)); \
    } while (0)

    for (; j + 31 < cols; j += 32) {
        __m256i vw = microgemm_avx2_unpack_i4_32(row + (j >> 1));
        MICROGEMM_I4_ROW_TILE8_ACC(0, vw);
        MICROGEMM_I4_ROW_TILE8_ACC(1, vw);
        MICROGEMM_I4_ROW_TILE8_ACC(2, vw);
        MICROGEMM_I4_ROW_TILE8_ACC(3, vw);
        MICROGEMM_I4_ROW_TILE8_ACC(4, vw);
        MICROGEMM_I4_ROW_TILE8_ACC(5, vw);
        MICROGEMM_I4_ROW_TILE8_ACC(6, vw);
        MICROGEMM_I4_ROW_TILE8_ACC(7, vw);
    }

#define MICROGEMM_I4_ROW_TILE8_STORE(IDX) do { \
        int32_t acc_i = microgemm_avx2_hsum_epi32(acc##IDX) - 128 * row_sum; \
        int tail; \
        for (tail = j; tail < cols; ++tail) { \
            acc_i += (int32_t)(uint8_t)q##IDX[tail] * (int32_t)microgemm_i4_row_value(row, tail); \
        } \
        out[(size_t)(batch_offset + (IDX)) * rows + out_row] = \
            (float)acc_i * row_scale * qs##IDX + bias_value; \
    } while (0)

    MICROGEMM_I4_ROW_TILE8_STORE(0);
    MICROGEMM_I4_ROW_TILE8_STORE(1);
    MICROGEMM_I4_ROW_TILE8_STORE(2);
    MICROGEMM_I4_ROW_TILE8_STORE(3);
    MICROGEMM_I4_ROW_TILE8_STORE(4);
    MICROGEMM_I4_ROW_TILE8_STORE(5);
    MICROGEMM_I4_ROW_TILE8_STORE(6);
    MICROGEMM_I4_ROW_TILE8_STORE(7);

#undef MICROGEMM_I4_ROW_TILE8_STORE
#undef MICROGEMM_I4_ROW_TILE8_ACC
}

static inline void microgemm_avx2_i4_batch_row_pair_tile4(
    float* out,
    int out_row,
    int rows,
    const uint8_t* row0,
    const uint8_t* row1,
    float row0_scale,
    float row1_scale,
    int row0_sum,
    int row1_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    const float* bias
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const __m256i vone16 = _mm256_set1_epi16(1);
    __m256i acc00 = _mm256_setzero_si256();
    __m256i acc01 = _mm256_setzero_si256();
    __m256i acc02 = _mm256_setzero_si256();
    __m256i acc03 = _mm256_setzero_si256();
    __m256i acc10 = _mm256_setzero_si256();
    __m256i acc11 = _mm256_setzero_si256();
    __m256i acc12 = _mm256_setzero_si256();
    __m256i acc13 = _mm256_setzero_si256();
    int j = 0;

#define MICROGEMM_I4_ROW_PAIR_TILE4_ACC(IDX) do { \
        __m256i vi = _mm256_loadu_si256((const __m256i*)(q##IDX + j)); \
        __m256i prod0 = _mm256_maddubs_epi16(vi, vw0); \
        __m256i prod1 = _mm256_maddubs_epi16(vi, vw1); \
        acc0##IDX = _mm256_add_epi32(acc0##IDX, _mm256_madd_epi16(prod0, vone16)); \
        acc1##IDX = _mm256_add_epi32(acc1##IDX, _mm256_madd_epi16(prod1, vone16)); \
    } while (0)

    for (; j + 31 < cols; j += 32) {
        __m256i vw0 = microgemm_avx2_unpack_i4_32(row0 + (j >> 1));
        __m256i vw1 = microgemm_avx2_unpack_i4_32(row1 + (j >> 1));
        MICROGEMM_I4_ROW_PAIR_TILE4_ACC(0);
        MICROGEMM_I4_ROW_PAIR_TILE4_ACC(1);
        MICROGEMM_I4_ROW_PAIR_TILE4_ACC(2);
        MICROGEMM_I4_ROW_PAIR_TILE4_ACC(3);
    }

#undef MICROGEMM_I4_ROW_PAIR_TILE4_ACC

#define MICROGEMM_I4_ROW_PAIR_TILE4_STORE(IDX) do { \
        int32_t acc0_i = microgemm_avx2_hsum_epi32(acc0##IDX) - 128 * row0_sum; \
        int32_t acc1_i = microgemm_avx2_hsum_epi32(acc1##IDX) - 128 * row1_sum; \
        int tail; \
        for (tail = j; tail < cols; ++tail) { \
            int32_t qv = (int32_t)(uint8_t)q##IDX[tail]; \
            acc0_i += qv * (int32_t)microgemm_i4_row_value(row0, tail); \
            acc1_i += qv * (int32_t)microgemm_i4_row_value(row1, tail); \
        } \
        out[(size_t)(batch_offset + (IDX)) * rows + out_row] = \
            (float)acc0_i * row0_scale * qs##IDX + (bias ? bias[out_row] : 0.0f); \
        out[(size_t)(batch_offset + (IDX)) * rows + out_row + 1] = \
            (float)acc1_i * row1_scale * qs##IDX + (bias ? bias[out_row + 1] : 0.0f); \
    } while (0)

    MICROGEMM_I4_ROW_PAIR_TILE4_STORE(0);
    MICROGEMM_I4_ROW_PAIR_TILE4_STORE(1);
    MICROGEMM_I4_ROW_PAIR_TILE4_STORE(2);
    MICROGEMM_I4_ROW_PAIR_TILE4_STORE(3);

#undef MICROGEMM_I4_ROW_PAIR_TILE4_STORE
}

static inline void microgemm_avx2_i8_batch_row_tile(
    float* out,
    int out_row,
    int rows,
    const int8_t* row,
    float row_scale,
    int row_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int tile,
    int cols,
    const float* bias
) {
    __m256i vacc0[4];
    __m256i vacc1[4];
    const int8_t* q_ptrs[4];
    float q_scales[4];
    const __m256i v128 = _mm256_set1_epi8((char)128u);
    const __m256i vone16 = _mm256_set1_epi16(1);
    int bb;
    int j = 0;

    for (bb = 0; bb < tile; ++bb) {
        vacc0[bb] = _mm256_setzero_si256();
        vacc1[bb] = _mm256_setzero_si256();
        q_ptrs[bb] = input_q + (size_t)(batch_offset + bb) * cols;
        q_scales[bb] = input_scales[batch_offset + bb];
    }

    for (; j + 63 < cols; j += 64) {
        __m256i vw0 = _mm256_loadu_si256((const __m256i*)(row + j));
        __m256i vw1 = _mm256_loadu_si256((const __m256i*)(row + j + 32));
        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = q_ptrs[bb];
            __m256i vi0 = _mm256_loadu_si256((const __m256i*)(q + j));
            __m256i vi1 = _mm256_loadu_si256((const __m256i*)(q + j + 32));
            __m256i vi0_u = _mm256_add_epi8(vi0, v128);
            __m256i vi1_u = _mm256_add_epi8(vi1, v128);
            __m256i prod0 = _mm256_maddubs_epi16(vi0_u, vw0);
            __m256i prod1 = _mm256_maddubs_epi16(vi1_u, vw1);
            vacc0[bb] = _mm256_add_epi32(vacc0[bb], _mm256_madd_epi16(prod0, vone16));
            vacc1[bb] = _mm256_add_epi32(vacc1[bb], _mm256_madd_epi16(prod1, vone16));
        }
    }

    for (; j + 31 < cols; j += 32) {
        __m256i vw = _mm256_loadu_si256((const __m256i*)(row + j));
        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = q_ptrs[bb];
            __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j));
            __m256i vi_u = _mm256_add_epi8(vi, v128);
            __m256i prod = _mm256_maddubs_epi16(vi_u, vw);
            vacc0[bb] = _mm256_add_epi32(vacc0[bb], _mm256_madd_epi16(prod, vone16));
        }
    }

    for (bb = 0; bb < tile; ++bb) {
        const int8_t* q = q_ptrs[bb];
        int32_t acc = microgemm_avx2_hsum_epi32(_mm256_add_epi32(vacc0[bb], vacc1[bb]))
            - 128 * row_sum;
        int tail;

        for (tail = j; tail < cols; ++tail) {
            acc += ((int32_t)q[tail] + 128) * (int32_t)row[tail];
        }

        out[(size_t)(batch_offset + bb) * rows + out_row] =
            (float)acc * row_scale * q_scales[bb]
            + (bias ? bias[out_row] : 0.0f);
    }
}

static inline void microgemm_avx2_i8_groupwise_batch_row_tile(
    float* out,
    int out_row,
    int rows,
    const int8_t* row,
    const float* row_scales,
    const int32_t* row_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int tile,
    int cols,
    int groups,
    const float* bias
) {
    const __m256i v128 = _mm256_set1_epi8((char)128u);
    const __m256i vone16 = _mm256_set1_epi16(1);
    float values[8];
    int bb;
    int group;

    for (bb = 0; bb < tile; ++bb) {
        values[bb] = bias ? bias[out_row] : 0.0f;
    }

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int group_sum;
        int j = 0;
        __m256i vacc[8];

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;
        group_sum = row_sums ? row_sums[group] : 0;
        if (row_sums == NULL) {
            int k;
            for (k = begin; k < end; ++k) {
                group_sum += (int)row[k];
            }
        }

        for (bb = 0; bb < tile; ++bb) {
            vacc[bb] = _mm256_setzero_si256();
        }

        for (; j + 31 < group_cols; j += 32) {
            __m256i vw = _mm256_loadu_si256((const __m256i*)(row + begin + j));
            for (bb = 0; bb < tile; ++bb) {
                const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
                __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j));
                __m256i vi_u = _mm256_add_epi8(vi, v128);
                __m256i prod = _mm256_maddubs_epi16(vi_u, vw);
                vacc[bb] = _mm256_add_epi32(vacc[bb], _mm256_madd_epi16(prod, vone16));
            }
        }

        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
            int32_t acc = microgemm_avx2_hsum_epi32(vacc[bb]) - 128 * group_sum;
            int tail;

            for (tail = j; tail < group_cols; ++tail) {
                acc += ((int32_t)q[tail] + 128) * (int32_t)row[begin + tail];
            }
            values[bb] += (float)acc * row_scales[group] * input_scales[batch_offset + bb];
        }
    }

    for (bb = 0; bb < tile; ++bb) {
        out[(size_t)(batch_offset + bb) * rows + out_row] = values[bb];
    }
}

static inline void microgemm_avx2_i8_groupwise_batch_row_tile_safe(
    float* out,
    int out_row,
    int rows,
    const int8_t* row,
    const float* row_scales,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int tile,
    int cols,
    int groups,
    const float* bias
) {
    float values[8];
    int bb;
    int group;

    for (bb = 0; bb < tile; ++bb) {
        values[bb] = bias ? bias[out_row] : 0.0f;
    }

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int j = 0;
        __m256i vacc0[8];
        __m256i vacc1[8];

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;

        for (bb = 0; bb < tile; ++bb) {
            vacc0[bb] = _mm256_setzero_si256();
            vacc1[bb] = _mm256_setzero_si256();
        }

        for (; j + 31 < group_cols; j += 32) {
            __m128i w0 = _mm_loadu_si128((const __m128i*)(row + begin + j));
            __m128i w1 = _mm_loadu_si128((const __m128i*)(row + begin + j + 16));
            __m256i w0_16 = _mm256_cvtepi8_epi16(w0);
            __m256i w1_16 = _mm256_cvtepi8_epi16(w1);
            for (bb = 0; bb < tile; ++bb) {
                const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
                __m128i q0 = _mm_loadu_si128((const __m128i*)(q + j));
                __m128i q1 = _mm_loadu_si128((const __m128i*)(q + j + 16));
                __m256i q0_16 = _mm256_cvtepi8_epi16(q0);
                __m256i q1_16 = _mm256_cvtepi8_epi16(q1);
                vacc0[bb] = _mm256_add_epi32(vacc0[bb], _mm256_madd_epi16(q0_16, w0_16));
                vacc1[bb] = _mm256_add_epi32(vacc1[bb], _mm256_madd_epi16(q1_16, w1_16));
            }
        }

        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
            int32_t acc = microgemm_avx2_hsum_epi32(_mm256_add_epi32(vacc0[bb], vacc1[bb]));
            int tail;

            for (tail = j; tail < group_cols; ++tail) {
                acc += (int32_t)q[tail] * (int32_t)row[begin + tail];
            }
            values[bb] += (float)acc * row_scales[group] * input_scales[batch_offset + bb];
        }
    }

    for (bb = 0; bb < tile; ++bb) {
        out[(size_t)(batch_offset + bb) * rows + out_row] = values[bb];
    }
}

static inline void microgemm_avx2_i8_groupwise_batch_row_tile4_safe(
    float* out,
    int out_row,
    int rows,
    const int8_t* row,
    const float* row_scales,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int groups,
    const float* bias
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float bias_value = bias ? bias[out_row] : 0.0f;
    float v0 = bias_value;
    float v1 = bias_value;
    float v2 = bias_value;
    float v3 = bias_value;
    int group;

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int j = 0;
        __m256i acc0a = _mm256_setzero_si256();
        __m256i acc0b = _mm256_setzero_si256();
        __m256i acc1a = _mm256_setzero_si256();
        __m256i acc1b = _mm256_setzero_si256();
        __m256i acc2a = _mm256_setzero_si256();
        __m256i acc2b = _mm256_setzero_si256();
        __m256i acc3a = _mm256_setzero_si256();
        __m256i acc3b = _mm256_setzero_si256();
        int32_t a0;
        int32_t a1;
        int32_t a2;
        int32_t a3;
        int tail;

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;

        for (; j + 31 < group_cols; j += 32) {
            __m128i w0 = _mm_loadu_si128((const __m128i*)(row + begin + j));
            __m128i w1 = _mm_loadu_si128((const __m128i*)(row + begin + j + 16));
            __m256i w0_16 = _mm256_cvtepi8_epi16(w0);
            __m256i w1_16 = _mm256_cvtepi8_epi16(w1);

#define MICROGEMM_I8G_SAFE_TILE4_ACC(IDX) do { \
                __m128i qv0 = _mm_loadu_si128((const __m128i*)(q##IDX + begin + j)); \
                __m128i qv1 = _mm_loadu_si128((const __m128i*)(q##IDX + begin + j + 16)); \
                __m256i qv0_16 = _mm256_cvtepi8_epi16(qv0); \
                __m256i qv1_16 = _mm256_cvtepi8_epi16(qv1); \
                acc##IDX##a = _mm256_add_epi32(acc##IDX##a, _mm256_madd_epi16(qv0_16, w0_16)); \
                acc##IDX##b = _mm256_add_epi32(acc##IDX##b, _mm256_madd_epi16(qv1_16, w1_16)); \
            } while (0)

            MICROGEMM_I8G_SAFE_TILE4_ACC(0);
            MICROGEMM_I8G_SAFE_TILE4_ACC(1);
            MICROGEMM_I8G_SAFE_TILE4_ACC(2);
            MICROGEMM_I8G_SAFE_TILE4_ACC(3);

#undef MICROGEMM_I8G_SAFE_TILE4_ACC
        }

        a0 = microgemm_avx2_hsum_epi32(_mm256_add_epi32(acc0a, acc0b));
        a1 = microgemm_avx2_hsum_epi32(_mm256_add_epi32(acc1a, acc1b));
        a2 = microgemm_avx2_hsum_epi32(_mm256_add_epi32(acc2a, acc2b));
        a3 = microgemm_avx2_hsum_epi32(_mm256_add_epi32(acc3a, acc3b));
        for (tail = j; tail < group_cols; ++tail) {
            const int w = (int)row[begin + tail];
            a0 += (int32_t)q0[begin + tail] * w;
            a1 += (int32_t)q1[begin + tail] * w;
            a2 += (int32_t)q2[begin + tail] * w;
            a3 += (int32_t)q3[begin + tail] * w;
        }
        v0 += (float)a0 * row_scales[group] * qs0;
        v1 += (float)a1 * row_scales[group] * qs1;
        v2 += (float)a2 * row_scales[group] * qs2;
        v3 += (float)a3 * row_scales[group] * qs3;
    }

    out[(size_t)(batch_offset + 0) * rows + out_row] = v0;
    out[(size_t)(batch_offset + 1) * rows + out_row] = v1;
    out[(size_t)(batch_offset + 2) * rows + out_row] = v2;
    out[(size_t)(batch_offset + 3) * rows + out_row] = v3;
}

static inline void microgemm_avx2_i8_groupwise_batch_row_scores_safe(
    float* scores,
    const int8_t* row,
    const float* row_scales,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int tile,
    int cols,
    int groups
) {
    const int8_t* q = input_q + (size_t)batch_offset * cols;
    const float* q_scales = input_scales + batch_offset;

    if (tile == 4) {
        microgemm_avx2_i8_groupwise_batch_row_tile4_safe(
            scores, 0, 1, row, row_scales, q, q_scales, 0, cols, groups, NULL
        );
        return;
    }
    microgemm_avx2_i8_groupwise_batch_row_tile_safe(
        scores, 0, 1, row, row_scales, q, q_scales, 0, tile, cols, groups, NULL
    );
}

static inline void microgemm_avx2_i8_groupwise_gate_up_tile4_safe(
    float* out,
    int out_row,
    int out_stride,
    const int8_t* gate_row,
    const int8_t* up_row,
    const float* gate_scales,
    const float* up_scales,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int groups,
    int use_gelu,
    float* thread_absmax
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float gate2 = 0.0f;
    float gate3 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
    float up2 = 0.0f;
    float up3 = 0.0f;
    int group;

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int j = 0;
        __m256i gate_acc0 = _mm256_setzero_si256();
        __m256i gate_acc1 = _mm256_setzero_si256();
        __m256i gate_acc2 = _mm256_setzero_si256();
        __m256i gate_acc3 = _mm256_setzero_si256();
        __m256i up_acc0 = _mm256_setzero_si256();
        __m256i up_acc1 = _mm256_setzero_si256();
        __m256i up_acc2 = _mm256_setzero_si256();
        __m256i up_acc3 = _mm256_setzero_si256();
        int32_t g0;
        int32_t g1;
        int32_t g2;
        int32_t g3;
        int32_t u0;
        int32_t u1;
        int32_t u2;
        int32_t u3;
        int tail;

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;

        for (; j + 15 < group_cols; j += 16) {
            const int off = begin + j;
            __m128i gate_w = _mm_loadu_si128((const __m128i*)(gate_row + off));
            __m128i up_w = _mm_loadu_si128((const __m128i*)(up_row + off));
            __m256i gate_w16 = _mm256_cvtepi8_epi16(gate_w);
            __m256i up_w16 = _mm256_cvtepi8_epi16(up_w);

#define MICROGEMM_I8G_SAFE_GATE_UP_ACC(IDX) do { \
                __m128i qv = _mm_loadu_si128((const __m128i*)(q##IDX + off)); \
                __m256i qv16 = _mm256_cvtepi8_epi16(qv); \
                gate_acc##IDX = _mm256_add_epi32(gate_acc##IDX, _mm256_madd_epi16(qv16, gate_w16)); \
                up_acc##IDX = _mm256_add_epi32(up_acc##IDX, _mm256_madd_epi16(qv16, up_w16)); \
            } while (0)

            MICROGEMM_I8G_SAFE_GATE_UP_ACC(0);
            MICROGEMM_I8G_SAFE_GATE_UP_ACC(1);
            MICROGEMM_I8G_SAFE_GATE_UP_ACC(2);
            MICROGEMM_I8G_SAFE_GATE_UP_ACC(3);

#undef MICROGEMM_I8G_SAFE_GATE_UP_ACC
        }

        g0 = microgemm_avx2_hsum_epi32(gate_acc0);
        g1 = microgemm_avx2_hsum_epi32(gate_acc1);
        g2 = microgemm_avx2_hsum_epi32(gate_acc2);
        g3 = microgemm_avx2_hsum_epi32(gate_acc3);
        u0 = microgemm_avx2_hsum_epi32(up_acc0);
        u1 = microgemm_avx2_hsum_epi32(up_acc1);
        u2 = microgemm_avx2_hsum_epi32(up_acc2);
        u3 = microgemm_avx2_hsum_epi32(up_acc3);
        for (tail = j; tail < group_cols; ++tail) {
            const int off = begin + tail;
            const int gw = (int)gate_row[off];
            const int uw = (int)up_row[off];
            const int q0v = (int)q0[off];
            const int q1v = (int)q1[off];
            const int q2v = (int)q2[off];
            const int q3v = (int)q3[off];
            g0 += q0v * gw;
            g1 += q1v * gw;
            g2 += q2v * gw;
            g3 += q3v * gw;
            u0 += q0v * uw;
            u1 += q1v * uw;
            u2 += q2v * uw;
            u3 += q3v * uw;
        }

        gate0 += (float)g0 * gate_scales[group] * qs0;
        gate1 += (float)g1 * gate_scales[group] * qs1;
        gate2 += (float)g2 * gate_scales[group] * qs2;
        gate3 += (float)g3 * gate_scales[group] * qs3;
        up0 += (float)u0 * up_scales[group] * qs0;
        up1 += (float)u1 * up_scales[group] * qs1;
        up2 += (float)u2 * up_scales[group] * qs2;
        up3 += (float)u3 * up_scales[group] * qs3;
    }

#define MICROGEMM_I8G_SAFE_GATE_UP_STORE(IDX) do { \
        float gate_value = gate##IDX; \
        float value; \
        if (use_gelu) { \
            float x = gate_value; \
            gate_value = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate_value = microgemm_silu(gate_value); \
        } \
        value = gate_value * up##IDX; \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_row] = value; \
        if (thread_absmax != NULL) { \
            float abs_value = fabsf(value); \
            if (abs_value > thread_absmax[batch_offset + (IDX)]) { \
                thread_absmax[batch_offset + (IDX)] = abs_value; \
            } \
        } \
    } while (0)

    MICROGEMM_I8G_SAFE_GATE_UP_STORE(0);
    MICROGEMM_I8G_SAFE_GATE_UP_STORE(1);
    MICROGEMM_I8G_SAFE_GATE_UP_STORE(2);
    MICROGEMM_I8G_SAFE_GATE_UP_STORE(3);

#undef MICROGEMM_I8G_SAFE_GATE_UP_STORE
}

static inline void microgemm_avx2_i8_groupwise_gate_up_tile8_safe(
    float* out,
    int out_row,
    int out_stride,
    const int8_t* gate_row,
    const int8_t* up_row,
    const float* gate_scales,
    const float* up_scales,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int groups,
    int use_gelu,
    float* thread_absmax
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const int8_t* q4 = input_q + (size_t)(batch_offset + 4) * cols;
    const int8_t* q5 = input_q + (size_t)(batch_offset + 5) * cols;
    const int8_t* q6 = input_q + (size_t)(batch_offset + 6) * cols;
    const int8_t* q7 = input_q + (size_t)(batch_offset + 7) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float qs4 = input_scales[batch_offset + 4];
    const float qs5 = input_scales[batch_offset + 5];
    const float qs6 = input_scales[batch_offset + 6];
    const float qs7 = input_scales[batch_offset + 7];
    float gate0 = 0.0f, gate1 = 0.0f, gate2 = 0.0f, gate3 = 0.0f;
    float gate4 = 0.0f, gate5 = 0.0f, gate6 = 0.0f, gate7 = 0.0f;
    float up0 = 0.0f, up1 = 0.0f, up2 = 0.0f, up3 = 0.0f;
    float up4 = 0.0f, up5 = 0.0f, up6 = 0.0f, up7 = 0.0f;
    int group;

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int j = 0;
        __m256i gate_acc0 = _mm256_setzero_si256(), gate_acc1 = _mm256_setzero_si256();
        __m256i gate_acc2 = _mm256_setzero_si256(), gate_acc3 = _mm256_setzero_si256();
        __m256i gate_acc4 = _mm256_setzero_si256(), gate_acc5 = _mm256_setzero_si256();
        __m256i gate_acc6 = _mm256_setzero_si256(), gate_acc7 = _mm256_setzero_si256();
        __m256i up_acc0 = _mm256_setzero_si256(), up_acc1 = _mm256_setzero_si256();
        __m256i up_acc2 = _mm256_setzero_si256(), up_acc3 = _mm256_setzero_si256();
        __m256i up_acc4 = _mm256_setzero_si256(), up_acc5 = _mm256_setzero_si256();
        __m256i up_acc6 = _mm256_setzero_si256(), up_acc7 = _mm256_setzero_si256();
        int32_t g0, g1, g2, g3, g4, g5, g6, g7;
        int32_t u0, u1, u2, u3, u4, u5, u6, u7;
        int tail;

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;

        for (; j + 15 < group_cols; j += 16) {
            const int off = begin + j;
            __m128i gate_w = _mm_loadu_si128((const __m128i*)(gate_row + off));
            __m128i up_w = _mm_loadu_si128((const __m128i*)(up_row + off));
            __m256i gate_w16 = _mm256_cvtepi8_epi16(gate_w);
            __m256i up_w16 = _mm256_cvtepi8_epi16(up_w);

#define MICROGEMM_I8G_SAFE_GATE_UP_ACC8(IDX) do { \
                __m128i qv = _mm_loadu_si128((const __m128i*)(q##IDX + off)); \
                __m256i qv16 = _mm256_cvtepi8_epi16(qv); \
                gate_acc##IDX = _mm256_add_epi32(gate_acc##IDX, _mm256_madd_epi16(qv16, gate_w16)); \
                up_acc##IDX = _mm256_add_epi32(up_acc##IDX, _mm256_madd_epi16(qv16, up_w16)); \
            } while (0)

            MICROGEMM_I8G_SAFE_GATE_UP_ACC8(0);
            MICROGEMM_I8G_SAFE_GATE_UP_ACC8(1);
            MICROGEMM_I8G_SAFE_GATE_UP_ACC8(2);
            MICROGEMM_I8G_SAFE_GATE_UP_ACC8(3);
            MICROGEMM_I8G_SAFE_GATE_UP_ACC8(4);
            MICROGEMM_I8G_SAFE_GATE_UP_ACC8(5);
            MICROGEMM_I8G_SAFE_GATE_UP_ACC8(6);
            MICROGEMM_I8G_SAFE_GATE_UP_ACC8(7);

#undef MICROGEMM_I8G_SAFE_GATE_UP_ACC8
        }

        g0 = microgemm_avx2_hsum_epi32(gate_acc0);
        g1 = microgemm_avx2_hsum_epi32(gate_acc1);
        g2 = microgemm_avx2_hsum_epi32(gate_acc2);
        g3 = microgemm_avx2_hsum_epi32(gate_acc3);
        g4 = microgemm_avx2_hsum_epi32(gate_acc4);
        g5 = microgemm_avx2_hsum_epi32(gate_acc5);
        g6 = microgemm_avx2_hsum_epi32(gate_acc6);
        g7 = microgemm_avx2_hsum_epi32(gate_acc7);
        u0 = microgemm_avx2_hsum_epi32(up_acc0);
        u1 = microgemm_avx2_hsum_epi32(up_acc1);
        u2 = microgemm_avx2_hsum_epi32(up_acc2);
        u3 = microgemm_avx2_hsum_epi32(up_acc3);
        u4 = microgemm_avx2_hsum_epi32(up_acc4);
        u5 = microgemm_avx2_hsum_epi32(up_acc5);
        u6 = microgemm_avx2_hsum_epi32(up_acc6);
        u7 = microgemm_avx2_hsum_epi32(up_acc7);
        for (tail = j; tail < group_cols; ++tail) {
            const int off = begin + tail;
            const int gw = (int)gate_row[off];
            const int uw = (int)up_row[off];
#define MICROGEMM_I8G_SAFE_GATE_UP_TAIL8(IDX) do { \
                const int qv = (int)q##IDX[off]; \
                g##IDX += qv * gw; \
                u##IDX += qv * uw; \
            } while (0)
            MICROGEMM_I8G_SAFE_GATE_UP_TAIL8(0);
            MICROGEMM_I8G_SAFE_GATE_UP_TAIL8(1);
            MICROGEMM_I8G_SAFE_GATE_UP_TAIL8(2);
            MICROGEMM_I8G_SAFE_GATE_UP_TAIL8(3);
            MICROGEMM_I8G_SAFE_GATE_UP_TAIL8(4);
            MICROGEMM_I8G_SAFE_GATE_UP_TAIL8(5);
            MICROGEMM_I8G_SAFE_GATE_UP_TAIL8(6);
            MICROGEMM_I8G_SAFE_GATE_UP_TAIL8(7);
#undef MICROGEMM_I8G_SAFE_GATE_UP_TAIL8
        }

        gate0 += (float)g0 * gate_scales[group] * qs0;
        gate1 += (float)g1 * gate_scales[group] * qs1;
        gate2 += (float)g2 * gate_scales[group] * qs2;
        gate3 += (float)g3 * gate_scales[group] * qs3;
        gate4 += (float)g4 * gate_scales[group] * qs4;
        gate5 += (float)g5 * gate_scales[group] * qs5;
        gate6 += (float)g6 * gate_scales[group] * qs6;
        gate7 += (float)g7 * gate_scales[group] * qs7;
        up0 += (float)u0 * up_scales[group] * qs0;
        up1 += (float)u1 * up_scales[group] * qs1;
        up2 += (float)u2 * up_scales[group] * qs2;
        up3 += (float)u3 * up_scales[group] * qs3;
        up4 += (float)u4 * up_scales[group] * qs4;
        up5 += (float)u5 * up_scales[group] * qs5;
        up6 += (float)u6 * up_scales[group] * qs6;
        up7 += (float)u7 * up_scales[group] * qs7;
    }

#define MICROGEMM_I8G_SAFE_GATE_UP_STORE8(IDX) do { \
        float gate_value = gate##IDX; \
        float value; \
        if (use_gelu) { \
            float x = gate_value; \
            gate_value = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate_value = microgemm_silu(gate_value); \
        } \
        value = gate_value * up##IDX; \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_row] = value; \
        if (thread_absmax != NULL) { \
            float abs_value = fabsf(value); \
            if (abs_value > thread_absmax[batch_offset + (IDX)]) { \
                thread_absmax[batch_offset + (IDX)] = abs_value; \
            } \
        } \
    } while (0)

    MICROGEMM_I8G_SAFE_GATE_UP_STORE8(0);
    MICROGEMM_I8G_SAFE_GATE_UP_STORE8(1);
    MICROGEMM_I8G_SAFE_GATE_UP_STORE8(2);
    MICROGEMM_I8G_SAFE_GATE_UP_STORE8(3);
    MICROGEMM_I8G_SAFE_GATE_UP_STORE8(4);
    MICROGEMM_I8G_SAFE_GATE_UP_STORE8(5);
    MICROGEMM_I8G_SAFE_GATE_UP_STORE8(6);
    MICROGEMM_I8G_SAFE_GATE_UP_STORE8(7);

#undef MICROGEMM_I8G_SAFE_GATE_UP_STORE8
}

static inline void microgemm_avx2_i8_groupwise_batch_row_pair_tile4(
    float* out,
    int out_row,
    int rows,
    const int8_t* row0,
    const int8_t* row1,
    const float* row0_scales,
    const float* row1_scales,
    const int32_t* row0_sums,
    const int32_t* row1_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int groups,
    const float* bias
) {
    const __m256i v128 = _mm256_set1_epi8((char)128u);
    const __m256i vone16 = _mm256_set1_epi16(1);
    float values0[4];
    float values1[4];
    int bb;
    int group;

    for (bb = 0; bb < 4; ++bb) {
        values0[bb] = bias ? bias[out_row] : 0.0f;
        values1[bb] = bias ? bias[out_row + 1] : 0.0f;
    }

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int row0_sum;
        int row1_sum;
        int j = 0;
        __m256i acc00 = _mm256_setzero_si256();
        __m256i acc01 = _mm256_setzero_si256();
        __m256i acc02 = _mm256_setzero_si256();
        __m256i acc03 = _mm256_setzero_si256();
        __m256i acc10 = _mm256_setzero_si256();
        __m256i acc11 = _mm256_setzero_si256();
        __m256i acc12 = _mm256_setzero_si256();
        __m256i acc13 = _mm256_setzero_si256();

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;
        row0_sum = row0_sums ? row0_sums[group] : 0;
        row1_sum = row1_sums ? row1_sums[group] : 0;
        if (row0_sums == NULL || row1_sums == NULL) {
            int k;
            row0_sum = 0;
            row1_sum = 0;
            for (k = begin; k < end; ++k) {
                row0_sum += (int)row0[k];
                row1_sum += (int)row1[k];
            }
        }

#define MICROGEMM_I8G_ROW_PAIR_TILE4_ACC(IDX) do { \
            const int8_t* q = input_q + (size_t)(batch_offset + (IDX)) * cols + begin; \
            __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j)); \
            __m256i vi_u = _mm256_add_epi8(vi, v128); \
            __m256i prod0 = _mm256_maddubs_epi16(vi_u, vw0); \
            __m256i prod1 = _mm256_maddubs_epi16(vi_u, vw1); \
            acc0##IDX = _mm256_add_epi32(acc0##IDX, _mm256_madd_epi16(prod0, vone16)); \
            acc1##IDX = _mm256_add_epi32(acc1##IDX, _mm256_madd_epi16(prod1, vone16)); \
        } while (0)

        for (; j + 31 < group_cols; j += 32) {
            __m256i vw0 = _mm256_loadu_si256((const __m256i*)(row0 + begin + j));
            __m256i vw1 = _mm256_loadu_si256((const __m256i*)(row1 + begin + j));
            MICROGEMM_I8G_ROW_PAIR_TILE4_ACC(0);
            MICROGEMM_I8G_ROW_PAIR_TILE4_ACC(1);
            MICROGEMM_I8G_ROW_PAIR_TILE4_ACC(2);
            MICROGEMM_I8G_ROW_PAIR_TILE4_ACC(3);
        }

#undef MICROGEMM_I8G_ROW_PAIR_TILE4_ACC

#define MICROGEMM_I8G_ROW_PAIR_TILE4_ACCUM(IDX) do { \
            const int8_t* q = input_q + (size_t)(batch_offset + (IDX)) * cols + begin; \
            int32_t acc0_i = microgemm_avx2_hsum_epi32(acc0##IDX) - 128 * row0_sum; \
            int32_t acc1_i = microgemm_avx2_hsum_epi32(acc1##IDX) - 128 * row1_sum; \
            int tail; \
            for (tail = j; tail < group_cols; ++tail) { \
                int32_t qv = (int32_t)q[tail] + 128; \
                acc0_i += qv * (int32_t)row0[begin + tail]; \
                acc1_i += qv * (int32_t)row1[begin + tail]; \
            } \
            values0[IDX] += (float)acc0_i * row0_scales[group] * input_scales[batch_offset + (IDX)]; \
            values1[IDX] += (float)acc1_i * row1_scales[group] * input_scales[batch_offset + (IDX)]; \
        } while (0)

        MICROGEMM_I8G_ROW_PAIR_TILE4_ACCUM(0);
        MICROGEMM_I8G_ROW_PAIR_TILE4_ACCUM(1);
        MICROGEMM_I8G_ROW_PAIR_TILE4_ACCUM(2);
        MICROGEMM_I8G_ROW_PAIR_TILE4_ACCUM(3);

#undef MICROGEMM_I8G_ROW_PAIR_TILE4_ACCUM
    }

    for (bb = 0; bb < 4; ++bb) {
        out[(size_t)(batch_offset + bb) * rows + out_row] = values0[bb];
        out[(size_t)(batch_offset + bb) * rows + out_row + 1] = values1[bb];
    }
}

static inline void microgemm_avx2_i8_groupwise_batch_row_pair_tile4_safe(
    float* out,
    int out_row,
    int rows,
    const int8_t* row0,
    const int8_t* row1,
    const float* row0_scales,
    const float* row1_scales,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int groups,
    const float* bias
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float bias0 = bias ? bias[out_row] : 0.0f;
    const float bias1 = bias ? bias[out_row + 1] : 0.0f;
    float v00 = bias0;
    float v01 = bias0;
    float v02 = bias0;
    float v03 = bias0;
    float v10 = bias1;
    float v11 = bias1;
    float v12 = bias1;
    float v13 = bias1;
    int group;

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int j = 0;
        __m256i acc00 = _mm256_setzero_si256();
        __m256i acc01 = _mm256_setzero_si256();
        __m256i acc02 = _mm256_setzero_si256();
        __m256i acc03 = _mm256_setzero_si256();
        __m256i acc10 = _mm256_setzero_si256();
        __m256i acc11 = _mm256_setzero_si256();
        __m256i acc12 = _mm256_setzero_si256();
        __m256i acc13 = _mm256_setzero_si256();
        int32_t a00;
        int32_t a01;
        int32_t a02;
        int32_t a03;
        int32_t a10;
        int32_t a11;
        int32_t a12;
        int32_t a13;
        int tail;

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;

        for (; j + 15 < group_cols; j += 16) {
            __m128i w0 = _mm_loadu_si128((const __m128i*)(row0 + begin + j));
            __m128i w1 = _mm_loadu_si128((const __m128i*)(row1 + begin + j));
            __m256i w0_16 = _mm256_cvtepi8_epi16(w0);
            __m256i w1_16 = _mm256_cvtepi8_epi16(w1);

#define MICROGEMM_I8G_SAFE_ROWPAIR_TILE4_ACC(IDX) do { \
                __m128i qv = _mm_loadu_si128((const __m128i*)(q##IDX + begin + j)); \
                __m256i qv_16 = _mm256_cvtepi8_epi16(qv); \
                acc0##IDX = _mm256_add_epi32(acc0##IDX, _mm256_madd_epi16(qv_16, w0_16)); \
                acc1##IDX = _mm256_add_epi32(acc1##IDX, _mm256_madd_epi16(qv_16, w1_16)); \
            } while (0)

            MICROGEMM_I8G_SAFE_ROWPAIR_TILE4_ACC(0);
            MICROGEMM_I8G_SAFE_ROWPAIR_TILE4_ACC(1);
            MICROGEMM_I8G_SAFE_ROWPAIR_TILE4_ACC(2);
            MICROGEMM_I8G_SAFE_ROWPAIR_TILE4_ACC(3);

#undef MICROGEMM_I8G_SAFE_ROWPAIR_TILE4_ACC
        }

        a00 = microgemm_avx2_hsum_epi32(acc00);
        a01 = microgemm_avx2_hsum_epi32(acc01);
        a02 = microgemm_avx2_hsum_epi32(acc02);
        a03 = microgemm_avx2_hsum_epi32(acc03);
        a10 = microgemm_avx2_hsum_epi32(acc10);
        a11 = microgemm_avx2_hsum_epi32(acc11);
        a12 = microgemm_avx2_hsum_epi32(acc12);
        a13 = microgemm_avx2_hsum_epi32(acc13);

        for (tail = j; tail < group_cols; ++tail) {
            const int w0 = (int)row0[begin + tail];
            const int w1 = (int)row1[begin + tail];
            a00 += (int32_t)q0[begin + tail] * w0;
            a01 += (int32_t)q1[begin + tail] * w0;
            a02 += (int32_t)q2[begin + tail] * w0;
            a03 += (int32_t)q3[begin + tail] * w0;
            a10 += (int32_t)q0[begin + tail] * w1;
            a11 += (int32_t)q1[begin + tail] * w1;
            a12 += (int32_t)q2[begin + tail] * w1;
            a13 += (int32_t)q3[begin + tail] * w1;
        }

        v00 += (float)a00 * row0_scales[group] * qs0;
        v01 += (float)a01 * row0_scales[group] * qs1;
        v02 += (float)a02 * row0_scales[group] * qs2;
        v03 += (float)a03 * row0_scales[group] * qs3;
        v10 += (float)a10 * row1_scales[group] * qs0;
        v11 += (float)a11 * row1_scales[group] * qs1;
        v12 += (float)a12 * row1_scales[group] * qs2;
        v13 += (float)a13 * row1_scales[group] * qs3;
    }

    out[(size_t)(batch_offset + 0) * rows + out_row] = v00;
    out[(size_t)(batch_offset + 1) * rows + out_row] = v01;
    out[(size_t)(batch_offset + 2) * rows + out_row] = v02;
    out[(size_t)(batch_offset + 3) * rows + out_row] = v03;
    out[(size_t)(batch_offset + 0) * rows + out_row + 1] = v10;
    out[(size_t)(batch_offset + 1) * rows + out_row + 1] = v11;
    out[(size_t)(batch_offset + 2) * rows + out_row + 1] = v12;
    out[(size_t)(batch_offset + 3) * rows + out_row + 1] = v13;
}

static inline void microgemm_avx2_i4_groupwise_batch_row_tile(
    float* out,
    int out_row,
    int rows,
    const uint8_t* row,
    const float* row_scales,
    const int32_t* row_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int tile,
    int cols,
    int groups,
    const float* bias
) {
    const __m256i vone16 = _mm256_set1_epi16(1);
    float values[8];
    int bb;
    int group;

    for (bb = 0; bb < tile; ++bb) {
        values[bb] = bias ? bias[out_row] : 0.0f;
    }

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int group_sum;
        int j = 0;
        __m256i vacc[8];

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;
        group_sum = row_sums ? row_sums[group] : 0;
        if (row_sums == NULL) {
            int k;
            for (k = begin; k < end; ++k) {
                group_sum += (int)microgemm_i4_row_value(row, k);
            }
        }

        for (bb = 0; bb < tile; ++bb) {
            vacc[bb] = _mm256_setzero_si256();
        }

        for (; j + 31 < group_cols; j += 32) {
            __m256i vw = microgemm_avx2_unpack_i4_32(row + ((begin + j) >> 1));
            for (bb = 0; bb < tile; ++bb) {
                const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
                __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j));
                __m256i prod = _mm256_maddubs_epi16(vi, vw);
                vacc[bb] = _mm256_add_epi32(vacc[bb], _mm256_madd_epi16(prod, vone16));
            }
        }

        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
            int32_t acc = microgemm_avx2_hsum_epi32(vacc[bb]) - 128 * group_sum;
            int tail;

            for (tail = j; tail < group_cols; ++tail) {
                acc += (int32_t)(uint8_t)q[tail]
                    * (int32_t)microgemm_i4_row_value(row, begin + tail);
            }
            values[bb] += (float)acc * row_scales[group] * input_scales[batch_offset + bb];
        }
    }

    for (bb = 0; bb < tile; ++bb) {
        out[(size_t)(batch_offset + bb) * rows + out_row] = values[bb];
    }
}

static inline void microgemm_avx2_i4_groupwise_batch_row_pair_tile4(
    float* out,
    int out_row,
    int rows,
    const uint8_t* row0,
    const uint8_t* row1,
    const float* row0_scales,
    const float* row1_scales,
    const int32_t* row0_sums,
    const int32_t* row1_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int groups,
    const float* bias
) {
    const __m256i vone16 = _mm256_set1_epi16(1);
    float values0[4];
    float values1[4];
    int bb;
    int group;

    for (bb = 0; bb < 4; ++bb) {
        values0[bb] = bias ? bias[out_row] : 0.0f;
        values1[bb] = bias ? bias[out_row + 1] : 0.0f;
    }

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int row0_sum;
        int row1_sum;
        int j = 0;
        __m256i acc00 = _mm256_setzero_si256();
        __m256i acc01 = _mm256_setzero_si256();
        __m256i acc02 = _mm256_setzero_si256();
        __m256i acc03 = _mm256_setzero_si256();
        __m256i acc10 = _mm256_setzero_si256();
        __m256i acc11 = _mm256_setzero_si256();
        __m256i acc12 = _mm256_setzero_si256();
        __m256i acc13 = _mm256_setzero_si256();

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;
        row0_sum = row0_sums ? row0_sums[group] : 0;
        row1_sum = row1_sums ? row1_sums[group] : 0;
        if (row0_sums == NULL || row1_sums == NULL) {
            int k;
            row0_sum = 0;
            row1_sum = 0;
            for (k = begin; k < end; ++k) {
                row0_sum += (int)microgemm_i4_row_value(row0, k);
                row1_sum += (int)microgemm_i4_row_value(row1, k);
            }
        }

#define MICROGEMM_I4G_ROW_PAIR_TILE4_ACC(IDX) do { \
            const int8_t* q = input_q + (size_t)(batch_offset + (IDX)) * cols + begin; \
            __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j)); \
            __m256i prod0 = _mm256_maddubs_epi16(vi, vw0); \
            __m256i prod1 = _mm256_maddubs_epi16(vi, vw1); \
            acc0##IDX = _mm256_add_epi32(acc0##IDX, _mm256_madd_epi16(prod0, vone16)); \
            acc1##IDX = _mm256_add_epi32(acc1##IDX, _mm256_madd_epi16(prod1, vone16)); \
        } while (0)

        for (; j + 31 < group_cols; j += 32) {
            __m256i vw0 = microgemm_avx2_unpack_i4_32(row0 + ((begin + j) >> 1));
            __m256i vw1 = microgemm_avx2_unpack_i4_32(row1 + ((begin + j) >> 1));
            MICROGEMM_I4G_ROW_PAIR_TILE4_ACC(0);
            MICROGEMM_I4G_ROW_PAIR_TILE4_ACC(1);
            MICROGEMM_I4G_ROW_PAIR_TILE4_ACC(2);
            MICROGEMM_I4G_ROW_PAIR_TILE4_ACC(3);
        }

#undef MICROGEMM_I4G_ROW_PAIR_TILE4_ACC

#define MICROGEMM_I4G_ROW_PAIR_TILE4_ACCUM(IDX) do { \
            const int8_t* q = input_q + (size_t)(batch_offset + (IDX)) * cols + begin; \
            int32_t acc0_i = microgemm_avx2_hsum_epi32(acc0##IDX) - 128 * row0_sum; \
            int32_t acc1_i = microgemm_avx2_hsum_epi32(acc1##IDX) - 128 * row1_sum; \
            int tail; \
            for (tail = j; tail < group_cols; ++tail) { \
                int32_t qv = (int32_t)(uint8_t)q[tail]; \
                acc0_i += qv * (int32_t)microgemm_i4_row_value(row0, begin + tail); \
                acc1_i += qv * (int32_t)microgemm_i4_row_value(row1, begin + tail); \
            } \
            values0[IDX] += (float)acc0_i * row0_scales[group] * input_scales[batch_offset + (IDX)]; \
            values1[IDX] += (float)acc1_i * row1_scales[group] * input_scales[batch_offset + (IDX)]; \
        } while (0)

        MICROGEMM_I4G_ROW_PAIR_TILE4_ACCUM(0);
        MICROGEMM_I4G_ROW_PAIR_TILE4_ACCUM(1);
        MICROGEMM_I4G_ROW_PAIR_TILE4_ACCUM(2);
        MICROGEMM_I4G_ROW_PAIR_TILE4_ACCUM(3);

#undef MICROGEMM_I4G_ROW_PAIR_TILE4_ACCUM
    }

    for (bb = 0; bb < 4; ++bb) {
        out[(size_t)(batch_offset + bb) * rows + out_row] = values0[bb];
        out[(size_t)(batch_offset + bb) * rows + out_row + 1] = values1[bb];
    }
}

static inline void microgemm_avx2_i8_groupwise_batch_row_scores(
    float* scores,
    const int8_t* row,
    const float* row_scales,
    const int32_t* row_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int tile,
    int cols,
    int groups
) {
    const __m256i v128 = _mm256_set1_epi8((char)128u);
    const __m256i vone16 = _mm256_set1_epi16(1);
    int bb;
    int group;

    for (bb = 0; bb < tile; ++bb) {
        scores[bb] = 0.0f;
    }

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int group_sum;
        int j = 0;
        __m256i vacc[8];

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;
        group_sum = row_sums ? row_sums[group] : 0;
        if (row_sums == NULL) {
            int k;
            for (k = begin; k < end; ++k) {
                group_sum += (int)row[k];
            }
        }

        for (bb = 0; bb < tile; ++bb) {
            vacc[bb] = _mm256_setzero_si256();
        }

        for (; j + 31 < group_cols; j += 32) {
            __m256i vw = _mm256_loadu_si256((const __m256i*)(row + begin + j));
            for (bb = 0; bb < tile; ++bb) {
                const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
                __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j));
                __m256i vi_u = _mm256_add_epi8(vi, v128);
                __m256i prod = _mm256_maddubs_epi16(vi_u, vw);
                vacc[bb] = _mm256_add_epi32(vacc[bb], _mm256_madd_epi16(prod, vone16));
            }
        }

        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
            int32_t acc = microgemm_avx2_hsum_epi32(vacc[bb]) - 128 * group_sum;
            int tail;

            for (tail = j; tail < group_cols; ++tail) {
                acc += ((int32_t)q[tail] + 128) * (int32_t)row[begin + tail];
            }
            scores[bb] += (float)acc * row_scales[group] * input_scales[batch_offset + bb];
        }
    }
}

static inline void microgemm_avx2_i8_groupwise_batch_row_scores8_explicit(
    float* scores,
    const int8_t* row,
    const float* row_scales,
    const int32_t* row_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int groups
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const int8_t* q4 = input_q + (size_t)(batch_offset + 4) * cols;
    const int8_t* q5 = input_q + (size_t)(batch_offset + 5) * cols;
    const int8_t* q6 = input_q + (size_t)(batch_offset + 6) * cols;
    const int8_t* q7 = input_q + (size_t)(batch_offset + 7) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float qs4 = input_scales[batch_offset + 4];
    const float qs5 = input_scales[batch_offset + 5];
    const float qs6 = input_scales[batch_offset + 6];
    const float qs7 = input_scales[batch_offset + 7];
    const __m256i v128 = _mm256_set1_epi8((char)128u);
    const __m256i vone16 = _mm256_set1_epi16(1);
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    float s4 = 0.0f;
    float s5 = 0.0f;
    float s6 = 0.0f;
    float s7 = 0.0f;
    int group;

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int group_sum;
        int j = 0;
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();
        __m256i acc4 = _mm256_setzero_si256();
        __m256i acc5 = _mm256_setzero_si256();
        __m256i acc6 = _mm256_setzero_si256();
        __m256i acc7 = _mm256_setzero_si256();

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;
        group_sum = row_sums ? row_sums[group] : 0;
        if (row_sums == NULL) {
            int k;
            for (k = begin; k < end; ++k) {
                group_sum += (int)row[k];
            }
        }

        for (; j + 31 < group_cols; j += 32) {
            const int off = begin + j;
            __m256i vw = _mm256_loadu_si256((const __m256i*)(row + off));
#define MICROGEMM_I8G_SCORE8_STEP(QPTR, ACC) do { \
                __m256i vi = _mm256_loadu_si256((const __m256i*)((QPTR) + off)); \
                __m256i vi_u = _mm256_add_epi8(vi, v128); \
                __m256i prod = _mm256_maddubs_epi16(vi_u, vw); \
                (ACC) = _mm256_add_epi32((ACC), _mm256_madd_epi16(prod, vone16)); \
            } while (0)
            MICROGEMM_I8G_SCORE8_STEP(q0, acc0);
            MICROGEMM_I8G_SCORE8_STEP(q1, acc1);
            MICROGEMM_I8G_SCORE8_STEP(q2, acc2);
            MICROGEMM_I8G_SCORE8_STEP(q3, acc3);
            MICROGEMM_I8G_SCORE8_STEP(q4, acc4);
            MICROGEMM_I8G_SCORE8_STEP(q5, acc5);
            MICROGEMM_I8G_SCORE8_STEP(q6, acc6);
            MICROGEMM_I8G_SCORE8_STEP(q7, acc7);
#undef MICROGEMM_I8G_SCORE8_STEP
        }

        {
            int32_t a0 = microgemm_avx2_hsum_epi32(acc0) - 128 * group_sum;
            int32_t a1 = microgemm_avx2_hsum_epi32(acc1) - 128 * group_sum;
            int32_t a2 = microgemm_avx2_hsum_epi32(acc2) - 128 * group_sum;
            int32_t a3 = microgemm_avx2_hsum_epi32(acc3) - 128 * group_sum;
            int32_t a4 = microgemm_avx2_hsum_epi32(acc4) - 128 * group_sum;
            int32_t a5 = microgemm_avx2_hsum_epi32(acc5) - 128 * group_sum;
            int32_t a6 = microgemm_avx2_hsum_epi32(acc6) - 128 * group_sum;
            int32_t a7 = microgemm_avx2_hsum_epi32(acc7) - 128 * group_sum;
            int tail;

            for (tail = j; tail < group_cols; ++tail) {
                int off = begin + tail;
                int32_t w = (int32_t)row[off];
                a0 += ((int32_t)q0[off] + 128) * w;
                a1 += ((int32_t)q1[off] + 128) * w;
                a2 += ((int32_t)q2[off] + 128) * w;
                a3 += ((int32_t)q3[off] + 128) * w;
                a4 += ((int32_t)q4[off] + 128) * w;
                a5 += ((int32_t)q5[off] + 128) * w;
                a6 += ((int32_t)q6[off] + 128) * w;
                a7 += ((int32_t)q7[off] + 128) * w;
            }

            s0 += (float)a0 * row_scales[group] * qs0;
            s1 += (float)a1 * row_scales[group] * qs1;
            s2 += (float)a2 * row_scales[group] * qs2;
            s3 += (float)a3 * row_scales[group] * qs3;
            s4 += (float)a4 * row_scales[group] * qs4;
            s5 += (float)a5 * row_scales[group] * qs5;
            s6 += (float)a6 * row_scales[group] * qs6;
            s7 += (float)a7 * row_scales[group] * qs7;
        }
    }

    scores[0] = s0;
    scores[1] = s1;
    scores[2] = s2;
    scores[3] = s3;
    scores[4] = s4;
    scores[5] = s5;
    scores[6] = s6;
    scores[7] = s7;
}

static inline void microgemm_avx2_i4_groupwise_batch_row_scores(
    float* scores,
    const uint8_t* row,
    const float* row_scales,
    const int32_t* row_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int tile,
    int cols,
    int groups
) {
    const __m256i vone16 = _mm256_set1_epi16(1);
    int bb;
    int group;

    for (bb = 0; bb < tile; ++bb) {
        scores[bb] = 0.0f;
    }

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int group_sum;
        int j = 0;
        __m256i vacc[8];

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;
        group_sum = row_sums ? row_sums[group] : 0;
        if (row_sums == NULL) {
            int k;
            for (k = begin; k < end; ++k) {
                group_sum += (int)microgemm_i4_row_value(row, k);
            }
        }

        for (bb = 0; bb < tile; ++bb) {
            vacc[bb] = _mm256_setzero_si256();
        }

        for (; j + 31 < group_cols; j += 32) {
            __m256i vw = microgemm_avx2_unpack_i4_32(row + ((begin + j) >> 1));
            for (bb = 0; bb < tile; ++bb) {
                const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
                __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j));
                __m256i prod = _mm256_maddubs_epi16(vi, vw);
                vacc[bb] = _mm256_add_epi32(vacc[bb], _mm256_madd_epi16(prod, vone16));
            }
        }

        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
            int32_t acc = microgemm_avx2_hsum_epi32(vacc[bb]) - 128 * group_sum;
            int tail;

            for (tail = j; tail < group_cols; ++tail) {
                acc += (int32_t)(uint8_t)q[tail]
                    * (int32_t)microgemm_i4_row_value(row, begin + tail);
            }
            scores[bb] += (float)acc * row_scales[group] * input_scales[batch_offset + bb];
        }
    }
}

static inline void microgemm_avx2_i8_groupwise_gate_up_tile(
    float* out,
    int out_col,
    int out_stride,
    const int8_t* gate_row,
    const int8_t* up_row,
    const float* gate_scales,
    const float* up_scales,
    const int32_t* gate_sums,
    const int32_t* up_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int tile,
    int cols,
    int groups,
    int use_gelu
) {
    const __m256i v128 = _mm256_set1_epi8((char)128u);
    const __m256i vone16 = _mm256_set1_epi16(1);
    float gate_values[8];
    float up_values[8];
    int bb;
    int group;

    for (bb = 0; bb < tile; ++bb) {
        gate_values[bb] = 0.0f;
        up_values[bb] = 0.0f;
    }

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int gate_sum;
        int up_sum;
        int j = 0;
        __m256i gate_acc[8];
        __m256i up_acc[8];

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;
        gate_sum = gate_sums ? gate_sums[group] : 0;
        up_sum = up_sums ? up_sums[group] : 0;
        if (gate_sums == NULL || up_sums == NULL) {
            int k;
            int gs = 0;
            int us = 0;
            for (k = begin; k < end; ++k) {
                gs += (int)gate_row[k];
                us += (int)up_row[k];
            }
            if (gate_sums == NULL) {
                gate_sum = gs;
            }
            if (up_sums == NULL) {
                up_sum = us;
            }
        }

        for (bb = 0; bb < tile; ++bb) {
            gate_acc[bb] = _mm256_setzero_si256();
            up_acc[bb] = _mm256_setzero_si256();
        }

        for (; j + 31 < group_cols; j += 32) {
            __m256i vw_gate = _mm256_loadu_si256((const __m256i*)(gate_row + begin + j));
            __m256i vw_up = _mm256_loadu_si256((const __m256i*)(up_row + begin + j));
            for (bb = 0; bb < tile; ++bb) {
                const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
                __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j));
                __m256i vi_u = _mm256_add_epi8(vi, v128);
                __m256i gate_prod = _mm256_maddubs_epi16(vi_u, vw_gate);
                __m256i up_prod = _mm256_maddubs_epi16(vi_u, vw_up);
                gate_acc[bb] = _mm256_add_epi32(
                    gate_acc[bb], _mm256_madd_epi16(gate_prod, vone16)
                );
                up_acc[bb] = _mm256_add_epi32(
                    up_acc[bb], _mm256_madd_epi16(up_prod, vone16)
                );
            }
        }

        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
            int32_t gate_i = microgemm_avx2_hsum_epi32(gate_acc[bb]) - 128 * gate_sum;
            int32_t up_i = microgemm_avx2_hsum_epi32(up_acc[bb]) - 128 * up_sum;
            int tail;

            for (tail = j; tail < group_cols; ++tail) {
                int32_t qv = (int32_t)q[tail] + 128;
                gate_i += qv * (int32_t)gate_row[begin + tail];
                up_i += qv * (int32_t)up_row[begin + tail];
            }
            gate_values[bb] += (float)gate_i * gate_scales[group] * input_scales[batch_offset + bb];
            up_values[bb] += (float)up_i * up_scales[group] * input_scales[batch_offset + bb];
        }
    }

    for (bb = 0; bb < tile; ++bb) {
        float gate = gate_values[bb];
        float up = up_values[bb];
        if (use_gelu) {
            float x = gate;
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        } else {
            gate = microgemm_silu(gate);
        }
        out[(size_t)(batch_offset + bb) * out_stride + out_col] = gate * up;
    }
}

static inline void microgemm_avx2_i8_groupwise_gate_up_tile8_explicit(
    float* out,
    int out_col,
    int out_stride,
    const int8_t* gate_row,
    const int8_t* up_row,
    const float* gate_scales,
    const float* up_scales,
    const int32_t* gate_sums,
    const int32_t* up_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int groups,
    int use_gelu
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const int8_t* q4 = input_q + (size_t)(batch_offset + 4) * cols;
    const int8_t* q5 = input_q + (size_t)(batch_offset + 5) * cols;
    const int8_t* q6 = input_q + (size_t)(batch_offset + 6) * cols;
    const int8_t* q7 = input_q + (size_t)(batch_offset + 7) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float qs4 = input_scales[batch_offset + 4];
    const float qs5 = input_scales[batch_offset + 5];
    const float qs6 = input_scales[batch_offset + 6];
    const float qs7 = input_scales[batch_offset + 7];
    const __m256i v128 = _mm256_set1_epi8((char)128u);
    const __m256i vone16 = _mm256_set1_epi16(1);
    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float gate2 = 0.0f;
    float gate3 = 0.0f;
    float gate4 = 0.0f;
    float gate5 = 0.0f;
    float gate6 = 0.0f;
    float gate7 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
    float up2 = 0.0f;
    float up3 = 0.0f;
    float up4 = 0.0f;
    float up5 = 0.0f;
    float up6 = 0.0f;
    float up7 = 0.0f;
    int group;

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int gate_sum;
        int up_sum;
        int j = 0;
        __m256i gate_acc0 = _mm256_setzero_si256();
        __m256i gate_acc1 = _mm256_setzero_si256();
        __m256i gate_acc2 = _mm256_setzero_si256();
        __m256i gate_acc3 = _mm256_setzero_si256();
        __m256i gate_acc4 = _mm256_setzero_si256();
        __m256i gate_acc5 = _mm256_setzero_si256();
        __m256i gate_acc6 = _mm256_setzero_si256();
        __m256i gate_acc7 = _mm256_setzero_si256();
        __m256i up_acc0 = _mm256_setzero_si256();
        __m256i up_acc1 = _mm256_setzero_si256();
        __m256i up_acc2 = _mm256_setzero_si256();
        __m256i up_acc3 = _mm256_setzero_si256();
        __m256i up_acc4 = _mm256_setzero_si256();
        __m256i up_acc5 = _mm256_setzero_si256();
        __m256i up_acc6 = _mm256_setzero_si256();
        __m256i up_acc7 = _mm256_setzero_si256();

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;
        gate_sum = gate_sums ? gate_sums[group] : 0;
        up_sum = up_sums ? up_sums[group] : 0;
        if (gate_sums == NULL || up_sums == NULL) {
            int k;
            int gs = 0;
            int us = 0;
            for (k = begin; k < end; ++k) {
                gs += (int)gate_row[k];
                us += (int)up_row[k];
            }
            if (gate_sums == NULL) {
                gate_sum = gs;
            }
            if (up_sums == NULL) {
                up_sum = us;
            }
        }

#define MICROGEMM_I8G_GATE_TILE8_ACC(IDX) do { \
            __m256i vi = _mm256_loadu_si256((const __m256i*)(q##IDX + begin + j)); \
            __m256i vi_u = _mm256_add_epi8(vi, v128); \
            __m256i gate_prod = _mm256_maddubs_epi16(vi_u, vw_gate); \
            __m256i up_prod = _mm256_maddubs_epi16(vi_u, vw_up); \
            gate_acc##IDX = _mm256_add_epi32(gate_acc##IDX, _mm256_madd_epi16(gate_prod, vone16)); \
            up_acc##IDX = _mm256_add_epi32(up_acc##IDX, _mm256_madd_epi16(up_prod, vone16)); \
        } while (0)

        for (; j + 31 < group_cols; j += 32) {
            __m256i vw_gate = _mm256_loadu_si256((const __m256i*)(gate_row + begin + j));
            __m256i vw_up = _mm256_loadu_si256((const __m256i*)(up_row + begin + j));
            MICROGEMM_I8G_GATE_TILE8_ACC(0);
            MICROGEMM_I8G_GATE_TILE8_ACC(1);
            MICROGEMM_I8G_GATE_TILE8_ACC(2);
            MICROGEMM_I8G_GATE_TILE8_ACC(3);
            MICROGEMM_I8G_GATE_TILE8_ACC(4);
            MICROGEMM_I8G_GATE_TILE8_ACC(5);
            MICROGEMM_I8G_GATE_TILE8_ACC(6);
            MICROGEMM_I8G_GATE_TILE8_ACC(7);
        }

#undef MICROGEMM_I8G_GATE_TILE8_ACC

#define MICROGEMM_I8G_GATE_TILE8_ACCUM(IDX) do { \
            int32_t gate_i = microgemm_avx2_hsum_epi32(gate_acc##IDX) - 128 * gate_sum; \
            int32_t up_i = microgemm_avx2_hsum_epi32(up_acc##IDX) - 128 * up_sum; \
            int tail; \
            for (tail = j; tail < group_cols; ++tail) { \
                int32_t qv = (int32_t)q##IDX[begin + tail] + 128; \
                gate_i += qv * (int32_t)gate_row[begin + tail]; \
                up_i += qv * (int32_t)up_row[begin + tail]; \
            } \
            gate##IDX += (float)gate_i * gate_scales[group] * qs##IDX; \
            up##IDX += (float)up_i * up_scales[group] * qs##IDX; \
        } while (0)

        MICROGEMM_I8G_GATE_TILE8_ACCUM(0);
        MICROGEMM_I8G_GATE_TILE8_ACCUM(1);
        MICROGEMM_I8G_GATE_TILE8_ACCUM(2);
        MICROGEMM_I8G_GATE_TILE8_ACCUM(3);
        MICROGEMM_I8G_GATE_TILE8_ACCUM(4);
        MICROGEMM_I8G_GATE_TILE8_ACCUM(5);
        MICROGEMM_I8G_GATE_TILE8_ACCUM(6);
        MICROGEMM_I8G_GATE_TILE8_ACCUM(7);

#undef MICROGEMM_I8G_GATE_TILE8_ACCUM
    }

#define MICROGEMM_I8G_GATE_TILE8_STORE(IDX) do { \
        float gate = gate##IDX; \
        if (use_gelu) { \
            float x = gate; \
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate = microgemm_silu(gate); \
        } \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_col] = gate * up##IDX; \
    } while (0)

    MICROGEMM_I8G_GATE_TILE8_STORE(0);
    MICROGEMM_I8G_GATE_TILE8_STORE(1);
    MICROGEMM_I8G_GATE_TILE8_STORE(2);
    MICROGEMM_I8G_GATE_TILE8_STORE(3);
    MICROGEMM_I8G_GATE_TILE8_STORE(4);
    MICROGEMM_I8G_GATE_TILE8_STORE(5);
    MICROGEMM_I8G_GATE_TILE8_STORE(6);
    MICROGEMM_I8G_GATE_TILE8_STORE(7);

#undef MICROGEMM_I8G_GATE_TILE8_STORE
}

static inline void microgemm_avx2_i8_groupwise_gate_up_tile8_aligned128(
    float* out,
    int out_col,
    int out_stride,
    const int8_t* gate_row,
    const int8_t* up_row,
    const float* gate_scales,
    const float* up_scales,
    const int32_t* gate_sums,
    const int32_t* up_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int groups,
    int use_gelu
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const int8_t* q4 = input_q + (size_t)(batch_offset + 4) * cols;
    const int8_t* q5 = input_q + (size_t)(batch_offset + 5) * cols;
    const int8_t* q6 = input_q + (size_t)(batch_offset + 6) * cols;
    const int8_t* q7 = input_q + (size_t)(batch_offset + 7) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float qs4 = input_scales[batch_offset + 4];
    const float qs5 = input_scales[batch_offset + 5];
    const float qs6 = input_scales[batch_offset + 6];
    const float qs7 = input_scales[batch_offset + 7];
    const __m256i v128 = _mm256_set1_epi8((char)128u);
    const __m256i vone16 = _mm256_set1_epi16(1);
    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float gate2 = 0.0f;
    float gate3 = 0.0f;
    float gate4 = 0.0f;
    float gate5 = 0.0f;
    float gate6 = 0.0f;
    float gate7 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
    float up2 = 0.0f;
    float up3 = 0.0f;
    float up4 = 0.0f;
    float up5 = 0.0f;
    float up6 = 0.0f;
    float up7 = 0.0f;
    int group;

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        const int gate_sum = gate_sums[group];
        const int up_sum = up_sums[group];
        int j = 0;
        __m256i gate_acc0 = _mm256_setzero_si256();
        __m256i gate_acc1 = _mm256_setzero_si256();
        __m256i gate_acc2 = _mm256_setzero_si256();
        __m256i gate_acc3 = _mm256_setzero_si256();
        __m256i gate_acc4 = _mm256_setzero_si256();
        __m256i gate_acc5 = _mm256_setzero_si256();
        __m256i gate_acc6 = _mm256_setzero_si256();
        __m256i gate_acc7 = _mm256_setzero_si256();
        __m256i up_acc0 = _mm256_setzero_si256();
        __m256i up_acc1 = _mm256_setzero_si256();
        __m256i up_acc2 = _mm256_setzero_si256();
        __m256i up_acc3 = _mm256_setzero_si256();
        __m256i up_acc4 = _mm256_setzero_si256();
        __m256i up_acc5 = _mm256_setzero_si256();
        __m256i up_acc6 = _mm256_setzero_si256();
        __m256i up_acc7 = _mm256_setzero_si256();

#define MICROGEMM_I8G_GATE_TILE8_A128_ACC(IDX) do { \
            __m256i vi = _mm256_loadu_si256((const __m256i*)(q##IDX + begin + j)); \
            __m256i vi_u = _mm256_add_epi8(vi, v128); \
            __m256i gate_prod = _mm256_maddubs_epi16(vi_u, vw_gate); \
            __m256i up_prod = _mm256_maddubs_epi16(vi_u, vw_up); \
            gate_acc##IDX = _mm256_add_epi32(gate_acc##IDX, _mm256_madd_epi16(gate_prod, vone16)); \
            up_acc##IDX = _mm256_add_epi32(up_acc##IDX, _mm256_madd_epi16(up_prod, vone16)); \
        } while (0)

        for (; j < (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE; j += 32) {
            __m256i vw_gate = _mm256_loadu_si256((const __m256i*)(gate_row + begin + j));
            __m256i vw_up = _mm256_loadu_si256((const __m256i*)(up_row + begin + j));
            MICROGEMM_I8G_GATE_TILE8_A128_ACC(0);
            MICROGEMM_I8G_GATE_TILE8_A128_ACC(1);
            MICROGEMM_I8G_GATE_TILE8_A128_ACC(2);
            MICROGEMM_I8G_GATE_TILE8_A128_ACC(3);
            MICROGEMM_I8G_GATE_TILE8_A128_ACC(4);
            MICROGEMM_I8G_GATE_TILE8_A128_ACC(5);
            MICROGEMM_I8G_GATE_TILE8_A128_ACC(6);
            MICROGEMM_I8G_GATE_TILE8_A128_ACC(7);
        }

#undef MICROGEMM_I8G_GATE_TILE8_A128_ACC

#define MICROGEMM_I8G_GATE_TILE8_A128_ACCUM(IDX) do { \
            int32_t gate_i = microgemm_avx2_hsum_epi32(gate_acc##IDX) - 128 * gate_sum; \
            int32_t up_i = microgemm_avx2_hsum_epi32(up_acc##IDX) - 128 * up_sum; \
            gate##IDX += (float)gate_i * gate_scales[group] * qs##IDX; \
            up##IDX += (float)up_i * up_scales[group] * qs##IDX; \
        } while (0)

        MICROGEMM_I8G_GATE_TILE8_A128_ACCUM(0);
        MICROGEMM_I8G_GATE_TILE8_A128_ACCUM(1);
        MICROGEMM_I8G_GATE_TILE8_A128_ACCUM(2);
        MICROGEMM_I8G_GATE_TILE8_A128_ACCUM(3);
        MICROGEMM_I8G_GATE_TILE8_A128_ACCUM(4);
        MICROGEMM_I8G_GATE_TILE8_A128_ACCUM(5);
        MICROGEMM_I8G_GATE_TILE8_A128_ACCUM(6);
        MICROGEMM_I8G_GATE_TILE8_A128_ACCUM(7);

#undef MICROGEMM_I8G_GATE_TILE8_A128_ACCUM
    }

#define MICROGEMM_I8G_GATE_TILE8_A128_STORE(IDX) do { \
        float gate = gate##IDX; \
        if (use_gelu) { \
            float x = gate; \
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate = microgemm_silu(gate); \
        } \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_col] = gate * up##IDX; \
    } while (0)

    MICROGEMM_I8G_GATE_TILE8_A128_STORE(0);
    MICROGEMM_I8G_GATE_TILE8_A128_STORE(1);
    MICROGEMM_I8G_GATE_TILE8_A128_STORE(2);
    MICROGEMM_I8G_GATE_TILE8_A128_STORE(3);
    MICROGEMM_I8G_GATE_TILE8_A128_STORE(4);
    MICROGEMM_I8G_GATE_TILE8_A128_STORE(5);
    MICROGEMM_I8G_GATE_TILE8_A128_STORE(6);
    MICROGEMM_I8G_GATE_TILE8_A128_STORE(7);

#undef MICROGEMM_I8G_GATE_TILE8_A128_STORE
}

static inline void microgemm_avx2_i8_groupwise_gate_up_tile8_biased_aligned128(
    float* out,
    int out_col,
    int out_stride,
    const int8_t* gate_row,
    const int8_t* up_row,
    const float* gate_scales,
    const float* up_scales,
    const int32_t* gate_sums,
    const int32_t* up_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int groups,
    int use_gelu
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const int8_t* q4 = input_q + (size_t)(batch_offset + 4) * cols;
    const int8_t* q5 = input_q + (size_t)(batch_offset + 5) * cols;
    const int8_t* q6 = input_q + (size_t)(batch_offset + 6) * cols;
    const int8_t* q7 = input_q + (size_t)(batch_offset + 7) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float qs4 = input_scales[batch_offset + 4];
    const float qs5 = input_scales[batch_offset + 5];
    const float qs6 = input_scales[batch_offset + 6];
    const float qs7 = input_scales[batch_offset + 7];
    const __m256i vone16 = _mm256_set1_epi16(1);
    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float gate2 = 0.0f;
    float gate3 = 0.0f;
    float gate4 = 0.0f;
    float gate5 = 0.0f;
    float gate6 = 0.0f;
    float gate7 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
    float up2 = 0.0f;
    float up3 = 0.0f;
    float up4 = 0.0f;
    float up5 = 0.0f;
    float up6 = 0.0f;
    float up7 = 0.0f;
    int group;

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        const int gate_sum = gate_sums[group];
        const int up_sum = up_sums[group];
        int j = 0;
        __m256i gate_acc0 = _mm256_setzero_si256();
        __m256i gate_acc1 = _mm256_setzero_si256();
        __m256i gate_acc2 = _mm256_setzero_si256();
        __m256i gate_acc3 = _mm256_setzero_si256();
        __m256i gate_acc4 = _mm256_setzero_si256();
        __m256i gate_acc5 = _mm256_setzero_si256();
        __m256i gate_acc6 = _mm256_setzero_si256();
        __m256i gate_acc7 = _mm256_setzero_si256();
        __m256i up_acc0 = _mm256_setzero_si256();
        __m256i up_acc1 = _mm256_setzero_si256();
        __m256i up_acc2 = _mm256_setzero_si256();
        __m256i up_acc3 = _mm256_setzero_si256();
        __m256i up_acc4 = _mm256_setzero_si256();
        __m256i up_acc5 = _mm256_setzero_si256();
        __m256i up_acc6 = _mm256_setzero_si256();
        __m256i up_acc7 = _mm256_setzero_si256();

#define MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACC(IDX) do { \
            __m256i vi_u = _mm256_loadu_si256((const __m256i*)(q##IDX + begin + j)); \
            __m256i gate_prod = _mm256_maddubs_epi16(vi_u, vw_gate); \
            __m256i up_prod = _mm256_maddubs_epi16(vi_u, vw_up); \
            gate_acc##IDX = _mm256_add_epi32(gate_acc##IDX, _mm256_madd_epi16(gate_prod, vone16)); \
            up_acc##IDX = _mm256_add_epi32(up_acc##IDX, _mm256_madd_epi16(up_prod, vone16)); \
        } while (0)

        for (; j < (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE; j += 32) {
            __m256i vw_gate = _mm256_loadu_si256((const __m256i*)(gate_row + begin + j));
            __m256i vw_up = _mm256_loadu_si256((const __m256i*)(up_row + begin + j));
            MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACC(0);
            MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACC(1);
            MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACC(2);
            MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACC(3);
            MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACC(4);
            MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACC(5);
            MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACC(6);
            MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACC(7);
        }

#undef MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACC

#define MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACCUM(IDX) do { \
            int32_t gate_i = microgemm_avx2_hsum_epi32(gate_acc##IDX) - 128 * gate_sum; \
            int32_t up_i = microgemm_avx2_hsum_epi32(up_acc##IDX) - 128 * up_sum; \
            gate##IDX += (float)gate_i * gate_scales[group] * qs##IDX; \
            up##IDX += (float)up_i * up_scales[group] * qs##IDX; \
        } while (0)

        MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACCUM(0);
        MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACCUM(1);
        MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACCUM(2);
        MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACCUM(3);
        MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACCUM(4);
        MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACCUM(5);
        MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACCUM(6);
        MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACCUM(7);

#undef MICROGEMM_I8G_GATE_TILE8_BIASED_A128_ACCUM
    }

#define MICROGEMM_I8G_GATE_TILE8_BIASED_A128_STORE(IDX) do { \
        float gate = gate##IDX; \
        if (use_gelu) { \
            float x = gate; \
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate = microgemm_silu(gate); \
        } \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_col] = gate * up##IDX; \
    } while (0)

    MICROGEMM_I8G_GATE_TILE8_BIASED_A128_STORE(0);
    MICROGEMM_I8G_GATE_TILE8_BIASED_A128_STORE(1);
    MICROGEMM_I8G_GATE_TILE8_BIASED_A128_STORE(2);
    MICROGEMM_I8G_GATE_TILE8_BIASED_A128_STORE(3);
    MICROGEMM_I8G_GATE_TILE8_BIASED_A128_STORE(4);
    MICROGEMM_I8G_GATE_TILE8_BIASED_A128_STORE(5);
    MICROGEMM_I8G_GATE_TILE8_BIASED_A128_STORE(6);
    MICROGEMM_I8G_GATE_TILE8_BIASED_A128_STORE(7);

#undef MICROGEMM_I8G_GATE_TILE8_BIASED_A128_STORE
}

static inline void microgemm_avx2_i8_groupwise_gate_up_pair_tile4_biased_aligned128(
    float* out,
    int out_col,
    int out_stride,
    const int8_t* gate0_row,
    const int8_t* up0_row,
    const int8_t* gate1_row,
    const int8_t* up1_row,
    const float* gate0_scales,
    const float* up0_scales,
    const float* gate1_scales,
    const float* up1_scales,
    const int32_t* gate0_sums,
    const int32_t* up0_sums,
    const int32_t* gate1_sums,
    const int32_t* up1_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int groups,
    int use_gelu,
    int use_unroll64,
    int use_unroll128
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const __m256i vone16 = _mm256_set1_epi16(1);
    float gate00 = 0.0f;
    float gate01 = 0.0f;
    float gate02 = 0.0f;
    float gate03 = 0.0f;
    float up00 = 0.0f;
    float up01 = 0.0f;
    float up02 = 0.0f;
    float up03 = 0.0f;
    float gate10 = 0.0f;
    float gate11 = 0.0f;
    float gate12 = 0.0f;
    float gate13 = 0.0f;
    float up10 = 0.0f;
    float up11 = 0.0f;
    float up12 = 0.0f;
    float up13 = 0.0f;
    int group;

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        const int gate0_sum = gate0_sums[group];
        const int up0_sum = up0_sums[group];
        const int gate1_sum = gate1_sums[group];
        const int up1_sum = up1_sums[group];
        int j = 0;
        __m256i gate0_acc0 = _mm256_setzero_si256();
        __m256i gate0_acc1 = _mm256_setzero_si256();
        __m256i gate0_acc2 = _mm256_setzero_si256();
        __m256i gate0_acc3 = _mm256_setzero_si256();
        __m256i up0_acc0 = _mm256_setzero_si256();
        __m256i up0_acc1 = _mm256_setzero_si256();
        __m256i up0_acc2 = _mm256_setzero_si256();
        __m256i up0_acc3 = _mm256_setzero_si256();
        __m256i gate1_acc0 = _mm256_setzero_si256();
        __m256i gate1_acc1 = _mm256_setzero_si256();
        __m256i gate1_acc2 = _mm256_setzero_si256();
        __m256i gate1_acc3 = _mm256_setzero_si256();
        __m256i up1_acc0 = _mm256_setzero_si256();
        __m256i up1_acc1 = _mm256_setzero_si256();
        __m256i up1_acc2 = _mm256_setzero_si256();
        __m256i up1_acc3 = _mm256_setzero_si256();

#define MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(IDX, OFF) do { \
            __m256i vi_u = _mm256_loadu_si256((const __m256i*)(q##IDX + begin + (OFF))); \
            __m256i gate0_prod = _mm256_maddubs_epi16(vi_u, vw_gate0); \
            __m256i up0_prod = _mm256_maddubs_epi16(vi_u, vw_up0); \
            __m256i gate1_prod = _mm256_maddubs_epi16(vi_u, vw_gate1); \
            __m256i up1_prod = _mm256_maddubs_epi16(vi_u, vw_up1); \
            gate0_acc##IDX = _mm256_add_epi32(gate0_acc##IDX, _mm256_madd_epi16(gate0_prod, vone16)); \
            up0_acc##IDX = _mm256_add_epi32(up0_acc##IDX, _mm256_madd_epi16(up0_prod, vone16)); \
            gate1_acc##IDX = _mm256_add_epi32(gate1_acc##IDX, _mm256_madd_epi16(gate1_prod, vone16)); \
            up1_acc##IDX = _mm256_add_epi32(up1_acc##IDX, _mm256_madd_epi16(up1_prod, vone16)); \
        } while (0)

        if (use_unroll128) {
            for (; j + 127 < (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE; j += 128) {
                __m256i vw_gate0 = _mm256_loadu_si256((const __m256i*)(gate0_row + begin + j));
                __m256i vw_up0 = _mm256_loadu_si256((const __m256i*)(up0_row + begin + j));
                __m256i vw_gate1 = _mm256_loadu_si256((const __m256i*)(gate1_row + begin + j));
                __m256i vw_up1 = _mm256_loadu_si256((const __m256i*)(up1_row + begin + j));
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(0, j);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(1, j);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(2, j);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(3, j);

                vw_gate0 = _mm256_loadu_si256((const __m256i*)(gate0_row + begin + j + 32));
                vw_up0 = _mm256_loadu_si256((const __m256i*)(up0_row + begin + j + 32));
                vw_gate1 = _mm256_loadu_si256((const __m256i*)(gate1_row + begin + j + 32));
                vw_up1 = _mm256_loadu_si256((const __m256i*)(up1_row + begin + j + 32));
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(0, j + 32);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(1, j + 32);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(2, j + 32);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(3, j + 32);

                vw_gate0 = _mm256_loadu_si256((const __m256i*)(gate0_row + begin + j + 64));
                vw_up0 = _mm256_loadu_si256((const __m256i*)(up0_row + begin + j + 64));
                vw_gate1 = _mm256_loadu_si256((const __m256i*)(gate1_row + begin + j + 64));
                vw_up1 = _mm256_loadu_si256((const __m256i*)(up1_row + begin + j + 64));
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(0, j + 64);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(1, j + 64);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(2, j + 64);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(3, j + 64);

                vw_gate0 = _mm256_loadu_si256((const __m256i*)(gate0_row + begin + j + 96));
                vw_up0 = _mm256_loadu_si256((const __m256i*)(up0_row + begin + j + 96));
                vw_gate1 = _mm256_loadu_si256((const __m256i*)(gate1_row + begin + j + 96));
                vw_up1 = _mm256_loadu_si256((const __m256i*)(up1_row + begin + j + 96));
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(0, j + 96);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(1, j + 96);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(2, j + 96);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(3, j + 96);
            }
        } else if (use_unroll64) {
            for (; j + 63 < (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE; j += 64) {
                __m256i vw_gate0 = _mm256_loadu_si256((const __m256i*)(gate0_row + begin + j));
                __m256i vw_up0 = _mm256_loadu_si256((const __m256i*)(up0_row + begin + j));
                __m256i vw_gate1 = _mm256_loadu_si256((const __m256i*)(gate1_row + begin + j));
                __m256i vw_up1 = _mm256_loadu_si256((const __m256i*)(up1_row + begin + j));
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(0, j);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(1, j);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(2, j);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(3, j);

                vw_gate0 = _mm256_loadu_si256((const __m256i*)(gate0_row + begin + j + 32));
                vw_up0 = _mm256_loadu_si256((const __m256i*)(up0_row + begin + j + 32));
                vw_gate1 = _mm256_loadu_si256((const __m256i*)(gate1_row + begin + j + 32));
                vw_up1 = _mm256_loadu_si256((const __m256i*)(up1_row + begin + j + 32));
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(0, j + 32);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(1, j + 32);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(2, j + 32);
                MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(3, j + 32);
            }
        }
        for (; j < (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE; j += 32) {
            __m256i vw_gate0 = _mm256_loadu_si256((const __m256i*)(gate0_row + begin + j));
            __m256i vw_up0 = _mm256_loadu_si256((const __m256i*)(up0_row + begin + j));
            __m256i vw_gate1 = _mm256_loadu_si256((const __m256i*)(gate1_row + begin + j));
            __m256i vw_up1 = _mm256_loadu_si256((const __m256i*)(up1_row + begin + j));
            MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(0, j);
            MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(1, j);
            MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(2, j);
            MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT(3, j);
        }

#undef MICROGEMM_I8G_GATE_PAIR4_BIASED_ACC_AT

#define MICROGEMM_I8G_GATE_PAIR4_BIASED_ACCUM(IDX) do { \
            int32_t gate0_i = microgemm_avx2_hsum_epi32(gate0_acc##IDX) - 128 * gate0_sum; \
            int32_t up0_i = microgemm_avx2_hsum_epi32(up0_acc##IDX) - 128 * up0_sum; \
            int32_t gate1_i = microgemm_avx2_hsum_epi32(gate1_acc##IDX) - 128 * gate1_sum; \
            int32_t up1_i = microgemm_avx2_hsum_epi32(up1_acc##IDX) - 128 * up1_sum; \
            gate0##IDX += (float)gate0_i * gate0_scales[group] * qs##IDX; \
            up0##IDX += (float)up0_i * up0_scales[group] * qs##IDX; \
            gate1##IDX += (float)gate1_i * gate1_scales[group] * qs##IDX; \
            up1##IDX += (float)up1_i * up1_scales[group] * qs##IDX; \
        } while (0)

        MICROGEMM_I8G_GATE_PAIR4_BIASED_ACCUM(0);
        MICROGEMM_I8G_GATE_PAIR4_BIASED_ACCUM(1);
        MICROGEMM_I8G_GATE_PAIR4_BIASED_ACCUM(2);
        MICROGEMM_I8G_GATE_PAIR4_BIASED_ACCUM(3);

#undef MICROGEMM_I8G_GATE_PAIR4_BIASED_ACCUM
    }

#define MICROGEMM_I8G_GATE_PAIR4_STORE0(IDX) do { \
        float gate = gate0##IDX; \
        if (use_gelu) { \
            float x = gate; \
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate = microgemm_silu(gate); \
        } \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_col] = gate * up0##IDX; \
    } while (0)

#define MICROGEMM_I8G_GATE_PAIR4_STORE1(IDX) do { \
        float gate = gate1##IDX; \
        if (use_gelu) { \
            float x = gate; \
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate = microgemm_silu(gate); \
        } \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_col + 1] = gate * up1##IDX; \
    } while (0)

    MICROGEMM_I8G_GATE_PAIR4_STORE0(0);
    MICROGEMM_I8G_GATE_PAIR4_STORE0(1);
    MICROGEMM_I8G_GATE_PAIR4_STORE0(2);
    MICROGEMM_I8G_GATE_PAIR4_STORE0(3);
    MICROGEMM_I8G_GATE_PAIR4_STORE1(0);
    MICROGEMM_I8G_GATE_PAIR4_STORE1(1);
    MICROGEMM_I8G_GATE_PAIR4_STORE1(2);
    MICROGEMM_I8G_GATE_PAIR4_STORE1(3);

#undef MICROGEMM_I8G_GATE_PAIR4_STORE1
#undef MICROGEMM_I8G_GATE_PAIR4_STORE0
}

static inline void microgemm_avx2_i8_groupwise_gate_up_pair_tile8_splitpass_biased_aligned128(
    float* out,
    int out_col,
    int out_stride,
    const int8_t* gate0_row,
    const int8_t* up0_row,
    const int8_t* gate1_row,
    const int8_t* up1_row,
    const float* gate0_scales,
    const float* up0_scales,
    const float* gate1_scales,
    const float* up1_scales,
    const int32_t* gate0_sums,
    const int32_t* up0_sums,
    const int32_t* gate1_sums,
    const int32_t* up1_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int groups,
    int use_gelu
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const int8_t* q4 = input_q + (size_t)(batch_offset + 4) * cols;
    const int8_t* q5 = input_q + (size_t)(batch_offset + 5) * cols;
    const int8_t* q6 = input_q + (size_t)(batch_offset + 6) * cols;
    const int8_t* q7 = input_q + (size_t)(batch_offset + 7) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float qs4 = input_scales[batch_offset + 4];
    const float qs5 = input_scales[batch_offset + 5];
    const float qs6 = input_scales[batch_offset + 6];
    const float qs7 = input_scales[batch_offset + 7];
    const __m256i vone16 = _mm256_set1_epi16(1);
    float gate0_values[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float gate1_values[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up0_values[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up1_values[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int group;

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int j = 0;
        __m256i row0_acc0 = _mm256_setzero_si256();
        __m256i row0_acc1 = _mm256_setzero_si256();
        __m256i row0_acc2 = _mm256_setzero_si256();
        __m256i row0_acc3 = _mm256_setzero_si256();
        __m256i row0_acc4 = _mm256_setzero_si256();
        __m256i row0_acc5 = _mm256_setzero_si256();
        __m256i row0_acc6 = _mm256_setzero_si256();
        __m256i row0_acc7 = _mm256_setzero_si256();
        __m256i row1_acc0 = _mm256_setzero_si256();
        __m256i row1_acc1 = _mm256_setzero_si256();
        __m256i row1_acc2 = _mm256_setzero_si256();
        __m256i row1_acc3 = _mm256_setzero_si256();
        __m256i row1_acc4 = _mm256_setzero_si256();
        __m256i row1_acc5 = _mm256_setzero_si256();
        __m256i row1_acc6 = _mm256_setzero_si256();
        __m256i row1_acc7 = _mm256_setzero_si256();

#define MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(IDX, OFF) do { \
            __m256i vi_u = _mm256_loadu_si256((const __m256i*)(q##IDX + begin + (OFF))); \
            __m256i row0_prod = _mm256_maddubs_epi16(vi_u, vw0); \
            __m256i row1_prod = _mm256_maddubs_epi16(vi_u, vw1); \
            row0_acc##IDX = _mm256_add_epi32(row0_acc##IDX, _mm256_madd_epi16(row0_prod, vone16)); \
            row1_acc##IDX = _mm256_add_epi32(row1_acc##IDX, _mm256_madd_epi16(row1_prod, vone16)); \
        } while (0)

        for (; j < (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE; j += 32) {
            __m256i vw0 = _mm256_loadu_si256((const __m256i*)(gate0_row + begin + j));
            __m256i vw1 = _mm256_loadu_si256((const __m256i*)(gate1_row + begin + j));
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(0, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(1, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(2, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(3, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(4, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(5, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(6, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(7, j);
        }

#undef MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT

#define MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_GATE(IDX) do { \
            int32_t row0_i = microgemm_avx2_hsum_epi32(row0_acc##IDX) - 128 * gate0_sums[group]; \
            int32_t row1_i = microgemm_avx2_hsum_epi32(row1_acc##IDX) - 128 * gate1_sums[group]; \
            gate0_values[IDX] += (float)row0_i * gate0_scales[group] * qs##IDX; \
            gate1_values[IDX] += (float)row1_i * gate1_scales[group] * qs##IDX; \
        } while (0)

        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_GATE(0);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_GATE(1);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_GATE(2);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_GATE(3);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_GATE(4);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_GATE(5);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_GATE(6);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_GATE(7);

#undef MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_GATE
    }

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int j = 0;
        __m256i row0_acc0 = _mm256_setzero_si256();
        __m256i row0_acc1 = _mm256_setzero_si256();
        __m256i row0_acc2 = _mm256_setzero_si256();
        __m256i row0_acc3 = _mm256_setzero_si256();
        __m256i row0_acc4 = _mm256_setzero_si256();
        __m256i row0_acc5 = _mm256_setzero_si256();
        __m256i row0_acc6 = _mm256_setzero_si256();
        __m256i row0_acc7 = _mm256_setzero_si256();
        __m256i row1_acc0 = _mm256_setzero_si256();
        __m256i row1_acc1 = _mm256_setzero_si256();
        __m256i row1_acc2 = _mm256_setzero_si256();
        __m256i row1_acc3 = _mm256_setzero_si256();
        __m256i row1_acc4 = _mm256_setzero_si256();
        __m256i row1_acc5 = _mm256_setzero_si256();
        __m256i row1_acc6 = _mm256_setzero_si256();
        __m256i row1_acc7 = _mm256_setzero_si256();

#define MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(IDX, OFF) do { \
            __m256i vi_u = _mm256_loadu_si256((const __m256i*)(q##IDX + begin + (OFF))); \
            __m256i row0_prod = _mm256_maddubs_epi16(vi_u, vw0); \
            __m256i row1_prod = _mm256_maddubs_epi16(vi_u, vw1); \
            row0_acc##IDX = _mm256_add_epi32(row0_acc##IDX, _mm256_madd_epi16(row0_prod, vone16)); \
            row1_acc##IDX = _mm256_add_epi32(row1_acc##IDX, _mm256_madd_epi16(row1_prod, vone16)); \
        } while (0)

        for (; j < (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE; j += 32) {
            __m256i vw0 = _mm256_loadu_si256((const __m256i*)(up0_row + begin + j));
            __m256i vw1 = _mm256_loadu_si256((const __m256i*)(up1_row + begin + j));
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(0, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(1, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(2, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(3, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(4, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(5, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(6, j);
            MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT(7, j);
        }

#undef MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACC_AT

#define MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_UP(IDX) do { \
            int32_t row0_i = microgemm_avx2_hsum_epi32(row0_acc##IDX) - 128 * up0_sums[group]; \
            int32_t row1_i = microgemm_avx2_hsum_epi32(row1_acc##IDX) - 128 * up1_sums[group]; \
            up0_values[IDX] += (float)row0_i * up0_scales[group] * qs##IDX; \
            up1_values[IDX] += (float)row1_i * up1_scales[group] * qs##IDX; \
        } while (0)

        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_UP(0);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_UP(1);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_UP(2);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_UP(3);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_UP(4);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_UP(5);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_UP(6);
        MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_UP(7);

#undef MICROGEMM_I8G_GATE_PAIR8_SPLIT_ACCUM_UP
    }

#define MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE0(IDX) do { \
        float gate = gate0_values[IDX]; \
        if (use_gelu) { \
            float x = gate; \
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate = microgemm_silu(gate); \
        } \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_col] = gate * up0_values[IDX]; \
    } while (0)

#define MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE1(IDX) do { \
        float gate = gate1_values[IDX]; \
        if (use_gelu) { \
            float x = gate; \
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate = microgemm_silu(gate); \
        } \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_col + 1] = gate * up1_values[IDX]; \
    } while (0)

    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE0(0);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE0(1);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE0(2);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE0(3);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE0(4);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE0(5);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE0(6);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE0(7);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE1(0);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE1(1);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE1(2);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE1(3);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE1(4);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE1(5);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE1(6);
    MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE1(7);

#undef MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE1
#undef MICROGEMM_I8G_GATE_PAIR8_SPLIT_STORE0
}

static inline void microgemm_avx2_i4_groupwise_gate_up_tile(
    float* out,
    int out_col,
    int out_stride,
    const uint8_t* gate_row,
    const uint8_t* up_row,
    const float* gate_scales,
    const float* up_scales,
    const int32_t* gate_sums,
    const int32_t* up_sums,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int tile,
    int cols,
    int groups,
    int use_gelu
) {
    const __m256i vone16 = _mm256_set1_epi16(1);
    float gate_values[8];
    float up_values[8];
    int bb;
    int group;

    for (bb = 0; bb < tile; ++bb) {
        gate_values[bb] = 0.0f;
        up_values[bb] = 0.0f;
    }

    for (group = 0; group < groups; ++group) {
        const int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int group_cols;
        int gate_sum;
        int up_sum;
        int j = 0;
        __m256i gate_acc[8];
        __m256i up_acc[8];

        if (end > cols) {
            end = cols;
        }
        group_cols = end - begin;
        gate_sum = gate_sums ? gate_sums[group] : 0;
        up_sum = up_sums ? up_sums[group] : 0;
        if (gate_sums == NULL || up_sums == NULL) {
            int k;
            int gs = 0;
            int us = 0;
            for (k = begin; k < end; ++k) {
                gs += (int)microgemm_i4_row_value(gate_row, k);
                us += (int)microgemm_i4_row_value(up_row, k);
            }
            if (gate_sums == NULL) {
                gate_sum = gs;
            }
            if (up_sums == NULL) {
                up_sum = us;
            }
        }

        for (bb = 0; bb < tile; ++bb) {
            gate_acc[bb] = _mm256_setzero_si256();
            up_acc[bb] = _mm256_setzero_si256();
        }

        for (; j + 31 < group_cols; j += 32) {
            __m256i vw_gate = microgemm_avx2_unpack_i4_32(gate_row + ((begin + j) >> 1));
            __m256i vw_up = microgemm_avx2_unpack_i4_32(up_row + ((begin + j) >> 1));
            for (bb = 0; bb < tile; ++bb) {
                const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
                __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j));
                __m256i gate_prod = _mm256_maddubs_epi16(vi, vw_gate);
                __m256i up_prod = _mm256_maddubs_epi16(vi, vw_up);
                gate_acc[bb] = _mm256_add_epi32(
                    gate_acc[bb], _mm256_madd_epi16(gate_prod, vone16)
                );
                up_acc[bb] = _mm256_add_epi32(
                    up_acc[bb], _mm256_madd_epi16(up_prod, vone16)
                );
            }
        }

        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = input_q + (size_t)(batch_offset + bb) * cols + begin;
            int32_t gate_i = microgemm_avx2_hsum_epi32(gate_acc[bb]) - 128 * gate_sum;
            int32_t up_i = microgemm_avx2_hsum_epi32(up_acc[bb]) - 128 * up_sum;
            int tail;

            for (tail = j; tail < group_cols; ++tail) {
                int32_t qv = (int32_t)(uint8_t)q[tail];
                gate_i += qv * (int32_t)microgemm_i4_row_value(gate_row, begin + tail);
                up_i += qv * (int32_t)microgemm_i4_row_value(up_row, begin + tail);
            }
            gate_values[bb] += (float)gate_i * gate_scales[group] * input_scales[batch_offset + bb];
            up_values[bb] += (float)up_i * up_scales[group] * input_scales[batch_offset + bb];
        }
    }

    for (bb = 0; bb < tile; ++bb) {
        float gate = gate_values[bb];
        float up = up_values[bb];
        if (use_gelu) {
            float x = gate;
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        } else {
            gate = microgemm_silu(gate);
        }
        out[(size_t)(batch_offset + bb) * out_stride + out_col] = gate * up;
    }
}

static inline void microgemm_avx2_i8_batch_row_pair_tile(
    float* out,
    int out_row,
    int rows,
    const int8_t* row0,
    const int8_t* row1,
    float row0_scale,
    float row1_scale,
    int row0_sum,
    int row1_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int tile,
    int cols,
    const float* bias
) {
    __m256i vacc0[4];
    __m256i vacc1[4];
    const int8_t* q_ptrs[4];
    float q_scales[4];
    const __m256i v128 = _mm256_set1_epi8((char)128u);
    const __m256i vone16 = _mm256_set1_epi16(1);
    int bb;
    int j = 0;

    for (bb = 0; bb < tile; ++bb) {
        vacc0[bb] = _mm256_setzero_si256();
        vacc1[bb] = _mm256_setzero_si256();
        q_ptrs[bb] = input_q + (size_t)(batch_offset + bb) * cols;
        q_scales[bb] = input_scales[batch_offset + bb];
    }

    for (; j + 31 < cols; j += 32) {
        __m256i vw0 = _mm256_loadu_si256((const __m256i*)(row0 + j));
        __m256i vw1 = _mm256_loadu_si256((const __m256i*)(row1 + j));
        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = q_ptrs[bb];
            __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j));
            __m256i vi_u = _mm256_add_epi8(vi, v128);
            __m256i prod0 = _mm256_maddubs_epi16(vi_u, vw0);
            __m256i prod1 = _mm256_maddubs_epi16(vi_u, vw1);
            vacc0[bb] = _mm256_add_epi32(vacc0[bb], _mm256_madd_epi16(prod0, vone16));
            vacc1[bb] = _mm256_add_epi32(vacc1[bb], _mm256_madd_epi16(prod1, vone16));
        }
    }

    for (bb = 0; bb < tile; ++bb) {
        const int8_t* q = q_ptrs[bb];
        int32_t acc0 = microgemm_avx2_hsum_epi32(vacc0[bb]) - 128 * row0_sum;
        int32_t acc1 = microgemm_avx2_hsum_epi32(vacc1[bb]) - 128 * row1_sum;
        int tail;

        for (tail = j; tail < cols; ++tail) {
            int32_t qv = (int32_t)q[tail] + 128;
            acc0 += qv * (int32_t)row0[tail];
            acc1 += qv * (int32_t)row1[tail];
        }

        out[(size_t)(batch_offset + bb) * rows + out_row] =
            (float)acc0 * row0_scale * q_scales[bb]
            + (bias ? bias[out_row] : 0.0f);
        out[(size_t)(batch_offset + bb) * rows + out_row + 1] =
            (float)acc1 * row1_scale * q_scales[bb]
            + (bias ? bias[out_row + 1] : 0.0f);
    }
}

static inline void microgemm_avx2_i8_batch_row_pair_tile8_split(
    float* out,
    int out_row,
    int rows,
    const int8_t* row0,
    const int8_t* row1,
    float row0_scale,
    float row1_scale,
    int row0_sum,
    int row1_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    const float* bias
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const int8_t* q4 = input_q + (size_t)(batch_offset + 4) * cols;
    const int8_t* q5 = input_q + (size_t)(batch_offset + 5) * cols;
    const int8_t* q6 = input_q + (size_t)(batch_offset + 6) * cols;
    const int8_t* q7 = input_q + (size_t)(batch_offset + 7) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float qs4 = input_scales[batch_offset + 4];
    const float qs5 = input_scales[batch_offset + 5];
    const float qs6 = input_scales[batch_offset + 6];
    const float qs7 = input_scales[batch_offset + 7];
    const __m256i v128 = _mm256_set1_epi8((char)128u);
    const __m256i vone16 = _mm256_set1_epi16(1);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    __m256i acc4 = _mm256_setzero_si256();
    __m256i acc5 = _mm256_setzero_si256();
    __m256i acc6 = _mm256_setzero_si256();
    __m256i acc7 = _mm256_setzero_si256();
    int j = 0;

#define MICROGEMM_BATCH_TILE8_ACC(IDX, ROWV) do { \
        __m256i vi = _mm256_loadu_si256((const __m256i*)(q##IDX + j)); \
        vi = _mm256_add_epi8(vi, v128); \
        acc##IDX = _mm256_add_epi32( \
            acc##IDX, \
            _mm256_madd_epi16(_mm256_maddubs_epi16(vi, ROWV), vone16) \
        ); \
    } while (0)

    for (; j + 31 < cols; j += 32) {
        __m256i vw = _mm256_loadu_si256((const __m256i*)(row0 + j));
        MICROGEMM_BATCH_TILE8_ACC(0, vw);
        MICROGEMM_BATCH_TILE8_ACC(1, vw);
        MICROGEMM_BATCH_TILE8_ACC(2, vw);
        MICROGEMM_BATCH_TILE8_ACC(3, vw);
        MICROGEMM_BATCH_TILE8_ACC(4, vw);
        MICROGEMM_BATCH_TILE8_ACC(5, vw);
        MICROGEMM_BATCH_TILE8_ACC(6, vw);
        MICROGEMM_BATCH_TILE8_ACC(7, vw);
    }

#define MICROGEMM_BATCH_TILE8_STORE_ROW(IDX, ROWPTR, ROWSUM, ROWSCALE, ROWOFF) do { \
        int32_t acc_i = microgemm_avx2_hsum_epi32(acc##IDX) - 128 * (ROWSUM); \
        int tail; \
        for (tail = j; tail < cols; ++tail) { \
            acc_i += ((int32_t)q##IDX[tail] + 128) * (int32_t)(ROWPTR)[tail]; \
        } \
        out[(size_t)(batch_offset + (IDX)) * rows + out_row + (ROWOFF)] = \
            (float)acc_i * (ROWSCALE) * qs##IDX + (bias ? bias[out_row + (ROWOFF)] : 0.0f); \
    } while (0)

    MICROGEMM_BATCH_TILE8_STORE_ROW(0, row0, row0_sum, row0_scale, 0);
    MICROGEMM_BATCH_TILE8_STORE_ROW(1, row0, row0_sum, row0_scale, 0);
    MICROGEMM_BATCH_TILE8_STORE_ROW(2, row0, row0_sum, row0_scale, 0);
    MICROGEMM_BATCH_TILE8_STORE_ROW(3, row0, row0_sum, row0_scale, 0);
    MICROGEMM_BATCH_TILE8_STORE_ROW(4, row0, row0_sum, row0_scale, 0);
    MICROGEMM_BATCH_TILE8_STORE_ROW(5, row0, row0_sum, row0_scale, 0);
    MICROGEMM_BATCH_TILE8_STORE_ROW(6, row0, row0_sum, row0_scale, 0);
    MICROGEMM_BATCH_TILE8_STORE_ROW(7, row0, row0_sum, row0_scale, 0);

    acc0 = _mm256_setzero_si256();
    acc1 = _mm256_setzero_si256();
    acc2 = _mm256_setzero_si256();
    acc3 = _mm256_setzero_si256();
    acc4 = _mm256_setzero_si256();
    acc5 = _mm256_setzero_si256();
    acc6 = _mm256_setzero_si256();
    acc7 = _mm256_setzero_si256();
    j = 0;

    for (; j + 31 < cols; j += 32) {
        __m256i vw = _mm256_loadu_si256((const __m256i*)(row1 + j));
        MICROGEMM_BATCH_TILE8_ACC(0, vw);
        MICROGEMM_BATCH_TILE8_ACC(1, vw);
        MICROGEMM_BATCH_TILE8_ACC(2, vw);
        MICROGEMM_BATCH_TILE8_ACC(3, vw);
        MICROGEMM_BATCH_TILE8_ACC(4, vw);
        MICROGEMM_BATCH_TILE8_ACC(5, vw);
        MICROGEMM_BATCH_TILE8_ACC(6, vw);
        MICROGEMM_BATCH_TILE8_ACC(7, vw);
    }

    MICROGEMM_BATCH_TILE8_STORE_ROW(0, row1, row1_sum, row1_scale, 1);
    MICROGEMM_BATCH_TILE8_STORE_ROW(1, row1, row1_sum, row1_scale, 1);
    MICROGEMM_BATCH_TILE8_STORE_ROW(2, row1, row1_sum, row1_scale, 1);
    MICROGEMM_BATCH_TILE8_STORE_ROW(3, row1, row1_sum, row1_scale, 1);
    MICROGEMM_BATCH_TILE8_STORE_ROW(4, row1, row1_sum, row1_scale, 1);
    MICROGEMM_BATCH_TILE8_STORE_ROW(5, row1, row1_sum, row1_scale, 1);
    MICROGEMM_BATCH_TILE8_STORE_ROW(6, row1, row1_sum, row1_scale, 1);
    MICROGEMM_BATCH_TILE8_STORE_ROW(7, row1, row1_sum, row1_scale, 1);

#undef MICROGEMM_BATCH_TILE8_STORE_ROW
#undef MICROGEMM_BATCH_TILE8_ACC
}

static inline void microgemm_avx2_i8_batch_row_pair_tile4_split(
    float* out,
    int out_row,
    int rows,
    const int8_t* row0,
    const int8_t* row1,
    float row0_scale,
    float row1_scale,
    int row0_sum,
    int row1_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    const float* bias
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const __m256i v128 = _mm256_set1_epi8((char)128u);
    const __m256i vone16 = _mm256_set1_epi16(1);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    int j = 0;

#define MICROGEMM_BATCH_TILE4_SPLIT_ACC(IDX, ROWV, OFF) do { \
        __m256i vi = _mm256_loadu_si256((const __m256i*)(q##IDX + j + (OFF))); \
        vi = _mm256_add_epi8(vi, v128); \
        acc##IDX = _mm256_add_epi32( \
            acc##IDX, \
            _mm256_madd_epi16(_mm256_maddubs_epi16(vi, ROWV), vone16) \
        ); \
    } while (0)

    for (; j + 63 < cols; j += 64) {
        __m256i vw0 = _mm256_loadu_si256((const __m256i*)(row0 + j));
        __m256i vw1 = _mm256_loadu_si256((const __m256i*)(row0 + j + 32));
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(0, vw0, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(1, vw0, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(2, vw0, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(3, vw0, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(0, vw1, 32);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(1, vw1, 32);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(2, vw1, 32);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(3, vw1, 32);
    }
    for (; j + 31 < cols; j += 32) {
        __m256i vw = _mm256_loadu_si256((const __m256i*)(row0 + j));
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(0, vw, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(1, vw, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(2, vw, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(3, vw, 0);
    }

#define MICROGEMM_BATCH_TILE4_SPLIT_STORE(IDX, ROWPTR, ROWSUM, ROWSCALE, ROWOFF) do { \
        int32_t acc_i = microgemm_avx2_hsum_epi32(acc##IDX) - 128 * (ROWSUM); \
        int tail; \
        for (tail = j; tail < cols; ++tail) { \
            acc_i += ((int32_t)q##IDX[tail] + 128) * (int32_t)(ROWPTR)[tail]; \
        } \
        out[(size_t)(batch_offset + (IDX)) * rows + out_row + (ROWOFF)] = \
            (float)acc_i * (ROWSCALE) * qs##IDX + (bias ? bias[out_row + (ROWOFF)] : 0.0f); \
    } while (0)

    MICROGEMM_BATCH_TILE4_SPLIT_STORE(0, row0, row0_sum, row0_scale, 0);
    MICROGEMM_BATCH_TILE4_SPLIT_STORE(1, row0, row0_sum, row0_scale, 0);
    MICROGEMM_BATCH_TILE4_SPLIT_STORE(2, row0, row0_sum, row0_scale, 0);
    MICROGEMM_BATCH_TILE4_SPLIT_STORE(3, row0, row0_sum, row0_scale, 0);

    acc0 = _mm256_setzero_si256();
    acc1 = _mm256_setzero_si256();
    acc2 = _mm256_setzero_si256();
    acc3 = _mm256_setzero_si256();
    j = 0;

    for (; j + 63 < cols; j += 64) {
        __m256i vw0 = _mm256_loadu_si256((const __m256i*)(row1 + j));
        __m256i vw1 = _mm256_loadu_si256((const __m256i*)(row1 + j + 32));
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(0, vw0, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(1, vw0, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(2, vw0, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(3, vw0, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(0, vw1, 32);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(1, vw1, 32);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(2, vw1, 32);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(3, vw1, 32);
    }
    for (; j + 31 < cols; j += 32) {
        __m256i vw = _mm256_loadu_si256((const __m256i*)(row1 + j));
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(0, vw, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(1, vw, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(2, vw, 0);
        MICROGEMM_BATCH_TILE4_SPLIT_ACC(3, vw, 0);
    }

    MICROGEMM_BATCH_TILE4_SPLIT_STORE(0, row1, row1_sum, row1_scale, 1);
    MICROGEMM_BATCH_TILE4_SPLIT_STORE(1, row1, row1_sum, row1_scale, 1);
    MICROGEMM_BATCH_TILE4_SPLIT_STORE(2, row1, row1_sum, row1_scale, 1);
    MICROGEMM_BATCH_TILE4_SPLIT_STORE(3, row1, row1_sum, row1_scale, 1);

#undef MICROGEMM_BATCH_TILE4_SPLIT_STORE
#undef MICROGEMM_BATCH_TILE4_SPLIT_ACC
}
#endif

static int32_t microgemm_dot_i8_biased_i8(
    const int8_t* row,
    const int8_t* input_q,
    int cols,
    int row_sum
) {
    int32_t acc;
    int j = 0;

#if MICROGEMM_CPU_X86_AVX2
    {
        const __m256i v128 = _mm256_set1_epi8((char)128u);
        const __m256i vone16 = _mm256_set1_epi16(1);
        __m256i vacc0 = _mm256_setzero_si256();
        __m256i vacc1 = _mm256_setzero_si256();

        for (; j + 63 < cols; j += 64) {
            __m256i vw0 = _mm256_loadu_si256((const __m256i*)(row + j));
            __m256i vw1 = _mm256_loadu_si256((const __m256i*)(row + j + 32));
            __m256i vi0 = _mm256_loadu_si256((const __m256i*)(input_q + j));
            __m256i vi1 = _mm256_loadu_si256((const __m256i*)(input_q + j + 32));
            __m256i prod0;
            __m256i prod1;

            vi0 = _mm256_add_epi8(vi0, v128);
            vi1 = _mm256_add_epi8(vi1, v128);
            prod0 = _mm256_maddubs_epi16(vi0, vw0);
            prod1 = _mm256_maddubs_epi16(vi1, vw1);
            vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(prod0, vone16));
            vacc1 = _mm256_add_epi32(vacc1, _mm256_madd_epi16(prod1, vone16));
        }
        for (; j + 31 < cols; j += 32) {
            __m256i vw = _mm256_loadu_si256((const __m256i*)(row + j));
            __m256i vi = _mm256_loadu_si256((const __m256i*)(input_q + j));
            __m256i prod;

            vi = _mm256_add_epi8(vi, v128);
            prod = _mm256_maddubs_epi16(vi, vw);
            vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(prod, vone16));
        }
        acc = microgemm_avx2_hsum_epi32(_mm256_add_epi32(vacc0, vacc1)) - 128 * row_sum;
    }
#else
    acc = -128 * row_sum;
#endif

    for (; j < cols; ++j) {
        acc += ((int32_t)input_q[j] + 128) * (int32_t)row[j];
    }
    return acc;
}

static int32_t microgemm_dot_i8_signed_i8_safe(
    const int8_t* row,
    const int8_t* input_q,
    int cols
) {
#if MICROGEMM_CPU_X86_AVX2
    return microgemm_avx2_dot_i8_signed_i8_safe(row, input_q, cols);
#else
    int32_t acc = 0;
    int j;
    for (j = 0; j < cols; ++j) {
        acc += (int32_t)input_q[j] * (int32_t)row[j];
    }
    return acc;
#endif
}

struct microgemm_decode_workspace {
    microgemm_config config;
    uint32_t max_seq_len;
    size_t scratch_bytes;

    float* hidden;
    float* residual;
    float* normed;
    float* qkv;
    float* attn_out;
    float* o_out;
    float* gate_up;
    float* mlp_out;
    float* logits;
    float* scores;
    float* linear_conv_state;
    float* linear_recurrent_state;
    float* linear_delta;
    int8_t* input_q;
};

static int microgemm_decode_batch_profile_enabled = 0;
static int microgemm_profile_next_down_proj_gemv = 0;
static microgemm_decode_batch_profile microgemm_decode_batch_profile_state;

static double microgemm_profile_now_ms(void) {
#ifdef _OPENMP
    return omp_get_wtime() * 1000.0;
#else
    return ((double)clock() * 1000.0) / (double)CLOCKS_PER_SEC;
#endif
}

static void microgemm_profile_add(double* dst, double start_ms) {
    *dst += microgemm_profile_now_ms() - start_ms;
}

void microgemm_decode_batch_profile_set_enabled(int enabled) {
    microgemm_decode_batch_profile_enabled = enabled ? 1 : 0;
}

void microgemm_decode_batch_profile_reset(void) {
    memset(&microgemm_decode_batch_profile_state, 0, sizeof(microgemm_decode_batch_profile_state));
}

void microgemm_decode_batch_profile_get(microgemm_decode_batch_profile* out_profile) {
    if (out_profile == NULL) {
        return;
    }
    *out_profile = microgemm_decode_batch_profile_state;
}

static int microgemm_decode_config_is_valid(const microgemm_config* cfg) {
    if (cfg == NULL) {
        return 0;
    }
    if (cfg->hidden_size == 0 || cfg->intermediate_size == 0 || cfg->num_layers == 0) {
        return 0;
    }
    if (cfg->num_q_heads == 0 || cfg->num_kv_heads == 0 || cfg->head_dim == 0) {
        return 0;
    }
    if ((cfg->head_dim & 1u) != 0u) {
        return 0;
    }
    if (cfg->vocab_size == 0 || cfg->kv_block_size == 0) {
        return 0;
    }
    if (cfg->num_q_heads % cfg->num_kv_heads != 0) {
        return 0;
    }
    if (cfg->rotary_dim != 0 && (cfg->rotary_dim > cfg->head_dim || (cfg->rotary_dim & 1u) != 0u)) {
        return 0;
    }
    if (cfg->architecture == MICROGEMM_ARCH_QWEN35_LIKE) {
        if (cfg->linear_key_head_dim == 0 || cfg->linear_value_head_dim == 0
                || cfg->linear_num_key_heads == 0 || cfg->linear_num_value_heads == 0
                || cfg->linear_conv_kernel_dim == 0) {
            return 0;
        }
        if (cfg->linear_num_value_heads % cfg->linear_num_key_heads != 0) {
            return 0;
        }
    }
    return 1;
}

static int microgemm_full_attn_width(const microgemm_config* cfg) {
    return (int)(cfg->attn_width != 0u ? cfg->attn_width : cfg->num_q_heads * cfg->head_dim);
}

static int microgemm_full_q_rows_decode(const microgemm_config* cfg) {
    int rows = microgemm_full_attn_width(cfg);
    if ((cfg->flags & MICROGEMM_FLAG_ATTN_OUTPUT_GATE) != 0u) {
        rows *= 2;
    }
    return rows;
}

static int microgemm_full_qkv_rows_decode(const microgemm_config* cfg) {
    return microgemm_full_q_rows_decode(cfg) + 2 * (int)cfg->num_kv_heads * (int)cfg->head_dim;
}

static int microgemm_linear_key_head_dim_decode(const microgemm_config* cfg) {
    return (int)(cfg->linear_key_head_dim != 0u ? cfg->linear_key_head_dim : cfg->head_dim);
}

static int microgemm_linear_value_head_dim_decode(const microgemm_config* cfg) {
    return (int)(cfg->linear_value_head_dim != 0u ? cfg->linear_value_head_dim : cfg->head_dim);
}

static int microgemm_linear_num_key_heads_decode(const microgemm_config* cfg) {
    return (int)(cfg->linear_num_key_heads != 0u ? cfg->linear_num_key_heads : cfg->num_q_heads);
}

static int microgemm_linear_num_value_heads_decode(const microgemm_config* cfg) {
    return (int)(cfg->linear_num_value_heads != 0u ? cfg->linear_num_value_heads : cfg->num_q_heads);
}

/* Self-biasing INT8 dots add +128 inside the kernel; prebiased MADDUBS dots do not. */
static float microgemm_quantize_activation_for_i8_self_biasing_dot(
    int8_t* out,
    const float* input,
    int count
) {
    return microgemm_cpu_quantize_f32_to_i8(out, input, count);
}

static float microgemm_quantize_activation_for_prebiased_maddubs_dot(
    int8_t* out,
    const float* input,
    int count
) {
    return microgemm_cpu_quantize_f32_to_biased_u8(out, input, count);
}

static int microgemm_linear_key_dim_decode(const microgemm_config* cfg) {
    return microgemm_linear_num_key_heads_decode(cfg) * microgemm_linear_key_head_dim_decode(cfg);
}

static int microgemm_linear_value_dim_decode(const microgemm_config* cfg) {
    return microgemm_linear_num_value_heads_decode(cfg) * microgemm_linear_value_head_dim_decode(cfg);
}

static int microgemm_linear_conv_dim_decode(const microgemm_config* cfg) {
    return 2 * microgemm_linear_key_dim_decode(cfg) + microgemm_linear_value_dim_decode(cfg);
}

static int microgemm_linear_baz_rows_decode(const microgemm_config* cfg) {
    return microgemm_linear_value_dim_decode(cfg) + 2 * microgemm_linear_num_value_heads_decode(cfg);
}

static int microgemm_has_linear_attention_layers(const microgemm_config* cfg, const microgemm_model_weights_i8* model) {
    uint32_t i;
    if (cfg == NULL || model == NULL || model->layers == NULL) {
        return 0;
    }
    for (i = 0u; i < cfg->num_layers; ++i) {
        if (model->layers[i].layer_type == MICROGEMM_LAYER_LINEAR_ATTENTION) {
            return 1;
        }
    }
    return 0;
}

static size_t microgemm_decode_compute_scratch_bytes(const microgemm_config* cfg, uint32_t max_seq_len) {
    size_t hidden_size = cfg->hidden_size;
    size_t qkv_size = (size_t)(cfg->qkv_rows != 0u ? cfg->qkv_rows : (cfg->num_q_heads + 2U * cfg->num_kv_heads) * cfg->head_dim);
    size_t attn_out_size = (size_t)(cfg->attn_width != 0u ? cfg->attn_width : cfg->num_q_heads * cfg->head_dim);
    size_t linear_conv_dim = (size_t)microgemm_linear_conv_dim_decode(cfg);
    size_t linear_value_dim = (size_t)microgemm_linear_value_dim_decode(cfg);
    size_t linear_delta_size = cfg->architecture == MICROGEMM_ARCH_QWEN35_LIKE
        ? (size_t)microgemm_linear_value_head_dim_decode(cfg)
        : 0u;
    size_t linear_state_size =
        (size_t)microgemm_linear_num_value_heads_decode(cfg)
        * (size_t)microgemm_linear_key_head_dim_decode(cfg)
        * (size_t)microgemm_linear_value_head_dim_decode(cfg);
    size_t scores_size = (size_t)cfg->num_q_heads * max_seq_len;
    size_t max_input_q = hidden_size;
    size_t total = 0;

    if ((size_t)(2U * cfg->intermediate_size) > max_input_q) {
        max_input_q = (size_t)(2U * cfg->intermediate_size);
    }
    if ((size_t)(cfg->num_q_heads * cfg->head_dim) > max_input_q) {
        max_input_q = (size_t)(cfg->num_q_heads * cfg->head_dim);
    }
    if (linear_value_dim > attn_out_size) {
        attn_out_size = linear_value_dim;
    }
    if (linear_conv_dim > max_input_q) {
        max_input_q = linear_conv_dim;
    }
    if ((size_t)microgemm_linear_baz_rows_decode(cfg) > max_input_q) {
        max_input_q = (size_t)microgemm_linear_baz_rows_decode(cfg);
    }

    total += hidden_size * sizeof(float);
    total += hidden_size * sizeof(float);
    total += hidden_size * sizeof(float);
    total += qkv_size * sizeof(float);
    total += attn_out_size * sizeof(float);
    total += hidden_size * sizeof(float);
    total += (size_t)(2U * cfg->intermediate_size) * sizeof(float);
    total += hidden_size * sizeof(float);
    total += cfg->vocab_size * sizeof(float);
    total += scores_size * sizeof(float);
    if (cfg->architecture == MICROGEMM_ARCH_QWEN35_LIKE) {
        total += (size_t)cfg->num_layers * linear_conv_dim * (size_t)cfg->linear_conv_kernel_dim * sizeof(float);
        total += (size_t)cfg->num_layers * linear_state_size * sizeof(float);
        total += linear_delta_size * sizeof(float);
    }
    total += max_input_q * sizeof(int8_t);
    return total;
}

static void microgemm_decode_workspace_clear(struct microgemm_decode_workspace* ws) {
    if (ws == NULL) {
        return;
    }
    free(ws->hidden);
    free(ws->residual);
    free(ws->normed);
    free(ws->qkv);
    free(ws->attn_out);
    free(ws->o_out);
    free(ws->gate_up);
    free(ws->mlp_out);
    free(ws->logits);
    free(ws->scores);
    free(ws->linear_conv_state);
    free(ws->linear_recurrent_state);
    free(ws->linear_delta);
    free(ws->input_q);
    memset(ws, 0, sizeof(*ws));
}

static void microgemm_residual_add(float* out, const float* a, const float* b, int count) {
    int i = 0;
#if MICROGEMM_CPU_X86_AVX2
    for (; i + 7 < count; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
#elif MICROGEMM_CPU_ARM64_NEON
    for (; i + 3 < count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
#endif
    for (; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

static void microgemm_residual_add_scaled(
    float* out,
    const float* a,
    const float* b,
    int count,
    float residual_multiplier
) {
    int i = 0;
    if (residual_multiplier == 1.0f) {
        microgemm_residual_add(out, a, b, count);
        return;
    }
#if MICROGEMM_CPU_X86_AVX2
    {
        __m256 vm = _mm256_set1_ps(residual_multiplier);
        for (; i + 7 < count; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(out + i, _mm256_fmadd_ps(vb, vm, va));
        }
    }
#elif MICROGEMM_CPU_ARM64_NEON
    {
        float32x4_t vm = vdupq_n_f32(residual_multiplier);
        for (; i + 3 < count; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            vst1q_f32(out + i, vmlaq_f32(va, vb, vm));
        }
    }
#endif
    for (; i < count; ++i) {
        out[i] = a[i] + b[i] * residual_multiplier;
    }
}

static inline float microgemm_config_embedding_multiplier(const microgemm_config* config) {
    if (config != NULL && config->embedding_multiplier > 0.0f) {
        return config->embedding_multiplier;
    }
    return 1.0f;
}

static inline float microgemm_config_residual_multiplier(const microgemm_config* config) {
    if (config != NULL && config->residual_multiplier > 0.0f) {
        return config->residual_multiplier;
    }
    return 1.0f;
}

static inline float microgemm_config_logits_scaling(const microgemm_config* config) {
    if (config != NULL && config->logits_scaling > 0.0f) {
        return config->logits_scaling;
    }
    return 1.0f;
}

static void microgemm_scale_inplace(float* values, int count, float scale) {
    int i = 0;
    if (values == NULL || count <= 0 || scale == 1.0f) {
        return;
    }
#if MICROGEMM_CPU_X86_AVX2
    {
        __m256 vs = _mm256_set1_ps(scale);
        for (; i + 7 < count; i += 8) {
            __m256 v = _mm256_loadu_ps(values + i);
            _mm256_storeu_ps(values + i, _mm256_mul_ps(v, vs));
        }
    }
#elif MICROGEMM_CPU_ARM64_NEON
    {
        float32x4_t vs = vdupq_n_f32(scale);
        for (; i + 3 < count; i += 4) {
            float32x4_t v = vld1q_f32(values + i);
            vst1q_f32(values + i, vmulq_f32(v, vs));
        }
    }
#endif
    for (; i < count; ++i) {
        values[i] *= scale;
    }
}

static float microgemm_silu(float x) {
    return x / (1.0f + expf(-x));
}

static void microgemm_swiglu_activation(float* gate, const float* up, int count, int use_gelu) {
    int i;
    if (use_gelu) {
        for (i = 0; i < count; ++i) {
            float x = gate[i];
            gate[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))) * up[i];
        }
    } else {
        for (i = 0; i < count; ++i) {
            gate[i] = microgemm_silu(gate[i]) * up[i];
        }
    }
}

#if MICROGEMM_CPU_X86_AVX2
static inline void microgemm_avx2_i8_gate_up_tile(
    float* out,
    int out_col,
    int out_stride,
    const int8_t* gate_row,
    const int8_t* up_row,
    float gate_scale,
    float up_scale,
    int gate_sum,
    int up_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int tile,
    int cols,
    int use_gelu
) {
    __m256i gate_acc[4];
    __m256i up_acc[4];
    const int8_t* q_ptrs[4];
    float q_scales[4];
    const __m256i vone16 = _mm256_set1_epi16(1);
    int bb;
    int j = 0;

    for (bb = 0; bb < tile; ++bb) {
        gate_acc[bb] = _mm256_setzero_si256();
        up_acc[bb] = _mm256_setzero_si256();
        q_ptrs[bb] = input_q + (size_t)(batch_offset + bb) * cols;
        q_scales[bb] = input_scales[batch_offset + bb];
    }

    for (; j + 31 < cols; j += 32) {
        __m256i vw_gate = _mm256_loadu_si256((const __m256i*)(gate_row + j));
        __m256i vw_up = _mm256_loadu_si256((const __m256i*)(up_row + j));
        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = q_ptrs[bb];
            __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j));
            __m256i gate_prod = _mm256_maddubs_epi16(vi, vw_gate);
            __m256i up_prod = _mm256_maddubs_epi16(vi, vw_up);
            gate_acc[bb] = _mm256_add_epi32(gate_acc[bb], _mm256_madd_epi16(gate_prod, vone16));
            up_acc[bb] = _mm256_add_epi32(up_acc[bb], _mm256_madd_epi16(up_prod, vone16));
        }
    }

    for (bb = 0; bb < tile; ++bb) {
        const int8_t* q = q_ptrs[bb];
        int32_t gate_i = microgemm_avx2_hsum_epi32(gate_acc[bb]) - 128 * gate_sum;
        int32_t up_i = microgemm_avx2_hsum_epi32(up_acc[bb]) - 128 * up_sum;
        float gate;
        float up;
        int tail;

        for (tail = j; tail < cols; ++tail) {
            int32_t qv = (int32_t)(uint8_t)q[tail];
            gate_i += qv * (int32_t)gate_row[tail];
            up_i += qv * (int32_t)up_row[tail];
        }

        gate = (float)gate_i * gate_scale * q_scales[bb];
        up = (float)up_i * up_scale * q_scales[bb];
        if (use_gelu) {
            float x = gate;
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        } else {
            gate = microgemm_silu(gate);
        }
        out[(size_t)(batch_offset + bb) * out_stride + out_col] = gate * up;
    }
}

static inline void microgemm_avx2_i8_gate_up_tile2(
    float* out,
    int out_col,
    int out_stride,
    const int8_t* gate_row,
    const int8_t* up_row,
    float gate_scale,
    float up_scale,
    int gate_sum,
    int up_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int use_gelu
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const __m256i vone16 = _mm256_set1_epi16(1);
    __m256i gate_acc0a = _mm256_setzero_si256();
    __m256i gate_acc0b = _mm256_setzero_si256();
    __m256i gate_acc1a = _mm256_setzero_si256();
    __m256i gate_acc1b = _mm256_setzero_si256();
    __m256i up_acc0a = _mm256_setzero_si256();
    __m256i up_acc0b = _mm256_setzero_si256();
    __m256i up_acc1a = _mm256_setzero_si256();
    __m256i up_acc1b = _mm256_setzero_si256();
    int j = 0;

#define MICROGEMM_GATE_UP_TILE2_ACC(CHUNK, OFF) do { \
        __m256i vw_gate = _mm256_loadu_si256((const __m256i*)(gate_row + j + (OFF))); \
        __m256i vw_up = _mm256_loadu_si256((const __m256i*)(up_row + j + (OFF))); \
        __m256i vi0 = _mm256_loadu_si256((const __m256i*)(q0 + j + (OFF))); \
        __m256i vi1 = _mm256_loadu_si256((const __m256i*)(q1 + j + (OFF))); \
        __m256i gate_prod0 = _mm256_maddubs_epi16(vi0, vw_gate); \
        __m256i up_prod0 = _mm256_maddubs_epi16(vi0, vw_up); \
        __m256i gate_prod1 = _mm256_maddubs_epi16(vi1, vw_gate); \
        __m256i up_prod1 = _mm256_maddubs_epi16(vi1, vw_up); \
        gate_acc0##CHUNK = _mm256_add_epi32(gate_acc0##CHUNK, _mm256_madd_epi16(gate_prod0, vone16)); \
        up_acc0##CHUNK = _mm256_add_epi32(up_acc0##CHUNK, _mm256_madd_epi16(up_prod0, vone16)); \
        gate_acc1##CHUNK = _mm256_add_epi32(gate_acc1##CHUNK, _mm256_madd_epi16(gate_prod1, vone16)); \
        up_acc1##CHUNK = _mm256_add_epi32(up_acc1##CHUNK, _mm256_madd_epi16(up_prod1, vone16)); \
    } while (0)

    for (; j + 63 < cols; j += 64) {
        MICROGEMM_GATE_UP_TILE2_ACC(a, 0);
        MICROGEMM_GATE_UP_TILE2_ACC(b, 32);
    }
    for (; j + 31 < cols; j += 32) {
        MICROGEMM_GATE_UP_TILE2_ACC(a, 0);
    }

#undef MICROGEMM_GATE_UP_TILE2_ACC

#define MICROGEMM_GATE_UP_TILE2_STORE(IDX) do { \
        int32_t gate_i = microgemm_avx2_hsum_epi32(_mm256_add_epi32(gate_acc##IDX##a, gate_acc##IDX##b)) - 128 * gate_sum; \
        int32_t up_i = microgemm_avx2_hsum_epi32(_mm256_add_epi32(up_acc##IDX##a, up_acc##IDX##b)) - 128 * up_sum; \
        float gate; \
        float up; \
        int tail; \
        for (tail = j; tail < cols; ++tail) { \
            int32_t qv = (int32_t)(uint8_t)q##IDX[tail]; \
            gate_i += qv * (int32_t)gate_row[tail]; \
            up_i += qv * (int32_t)up_row[tail]; \
        } \
        gate = (float)gate_i * gate_scale * qs##IDX; \
        up = (float)up_i * up_scale * qs##IDX; \
        if (use_gelu) { \
            float x = gate; \
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate = microgemm_silu(gate); \
        } \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_col] = gate * up; \
    } while (0)

    MICROGEMM_GATE_UP_TILE2_STORE(0);
    MICROGEMM_GATE_UP_TILE2_STORE(1);

#undef MICROGEMM_GATE_UP_TILE2_STORE
}

static inline void microgemm_avx2_i8_gate_up_tile4(
    float* out,
    int out_col,
    int out_stride,
    const int8_t* gate_row,
    const int8_t* up_row,
    float gate_scale,
    float up_scale,
    int gate_sum,
    int up_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int use_gelu
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const __m256i vone16 = _mm256_set1_epi16(1);
    __m256i gate_acc0 = _mm256_setzero_si256();
    __m256i gate_acc1 = _mm256_setzero_si256();
    __m256i gate_acc2 = _mm256_setzero_si256();
    __m256i gate_acc3 = _mm256_setzero_si256();
    __m256i up_acc0 = _mm256_setzero_si256();
    __m256i up_acc1 = _mm256_setzero_si256();
    __m256i up_acc2 = _mm256_setzero_si256();
    __m256i up_acc3 = _mm256_setzero_si256();
    int j = 0;
    int use_unroll64 = microgemm_i8_gate_tile4_unroll64_enabled_for(4, 0, cols);

#define MICROGEMM_GATE_UP_TILE4_ACCUM(IDX, GATE_ROW, UP_ROW, OFF) do { \
        __m256i vi = _mm256_loadu_si256((const __m256i*)(q##IDX + j + (OFF))); \
        __m256i gate_prod = _mm256_maddubs_epi16(vi, (GATE_ROW)); \
        __m256i up_prod = _mm256_maddubs_epi16(vi, (UP_ROW)); \
        gate_acc##IDX = _mm256_add_epi32(gate_acc##IDX, _mm256_madd_epi16(gate_prod, vone16)); \
        up_acc##IDX = _mm256_add_epi32(up_acc##IDX, _mm256_madd_epi16(up_prod, vone16)); \
    } while (0)

    if (use_unroll64) {
        for (; j + 63 < cols; j += 64) {
            __m256i vw_gate0 = _mm256_loadu_si256((const __m256i*)(gate_row + j));
            __m256i vw_up0 = _mm256_loadu_si256((const __m256i*)(up_row + j));
            __m256i vw_gate1 = _mm256_loadu_si256((const __m256i*)(gate_row + j + 32));
            __m256i vw_up1 = _mm256_loadu_si256((const __m256i*)(up_row + j + 32));
            MICROGEMM_GATE_UP_TILE4_ACCUM(0, vw_gate0, vw_up0, 0);
            MICROGEMM_GATE_UP_TILE4_ACCUM(1, vw_gate0, vw_up0, 0);
            MICROGEMM_GATE_UP_TILE4_ACCUM(2, vw_gate0, vw_up0, 0);
            MICROGEMM_GATE_UP_TILE4_ACCUM(3, vw_gate0, vw_up0, 0);
            MICROGEMM_GATE_UP_TILE4_ACCUM(0, vw_gate1, vw_up1, 32);
            MICROGEMM_GATE_UP_TILE4_ACCUM(1, vw_gate1, vw_up1, 32);
            MICROGEMM_GATE_UP_TILE4_ACCUM(2, vw_gate1, vw_up1, 32);
            MICROGEMM_GATE_UP_TILE4_ACCUM(3, vw_gate1, vw_up1, 32);
        }
    }

    for (; j + 31 < cols; j += 32) {
        __m256i vw_gate = _mm256_loadu_si256((const __m256i*)(gate_row + j));
        __m256i vw_up = _mm256_loadu_si256((const __m256i*)(up_row + j));
        MICROGEMM_GATE_UP_TILE4_ACCUM(0, vw_gate, vw_up, 0);
        MICROGEMM_GATE_UP_TILE4_ACCUM(1, vw_gate, vw_up, 0);
        MICROGEMM_GATE_UP_TILE4_ACCUM(2, vw_gate, vw_up, 0);
        MICROGEMM_GATE_UP_TILE4_ACCUM(3, vw_gate, vw_up, 0);
    }

#undef MICROGEMM_GATE_UP_TILE4_ACCUM

#define MICROGEMM_GATE_UP_TILE4_STORE(IDX) do { \
        int32_t gate_i = microgemm_avx2_hsum_epi32(gate_acc##IDX) - 128 * gate_sum; \
        int32_t up_i = microgemm_avx2_hsum_epi32(up_acc##IDX) - 128 * up_sum; \
        float gate; \
        float up; \
        int tail; \
        for (tail = j; tail < cols; ++tail) { \
            int32_t qv = (int32_t)(uint8_t)q##IDX[tail]; \
            gate_i += qv * (int32_t)gate_row[tail]; \
            up_i += qv * (int32_t)up_row[tail]; \
        } \
        gate = (float)gate_i * gate_scale * qs##IDX; \
        up = (float)up_i * up_scale * qs##IDX; \
        if (use_gelu) { \
            float x = gate; \
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate = microgemm_silu(gate); \
        } \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_col] = gate * up; \
    } while (0)

    MICROGEMM_GATE_UP_TILE4_STORE(0);
    MICROGEMM_GATE_UP_TILE4_STORE(1);
    MICROGEMM_GATE_UP_TILE4_STORE(2);
    MICROGEMM_GATE_UP_TILE4_STORE(3);

#undef MICROGEMM_GATE_UP_TILE4_STORE
}

static inline void microgemm_avx2_i4_gate_up_tile4(
    float* out,
    int out_col,
    int out_stride,
    const uint8_t* gate_row,
    const uint8_t* up_row,
    float gate_scale,
    float up_scale,
    int gate_sum,
    int up_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int tile,
    int cols,
    int use_gelu
) {
    const __m256i vone16 = _mm256_set1_epi16(1);
    const int8_t* q_ptrs[8];
    float q_scales[8];
    __m256i gate_acc[8];
    __m256i up_acc[8];
    int bb;
    int j = 0;

    for (bb = 0; bb < tile; ++bb) {
        q_ptrs[bb] = input_q + (size_t)(batch_offset + bb) * cols;
        q_scales[bb] = input_scales[batch_offset + bb];
        gate_acc[bb] = _mm256_setzero_si256();
        up_acc[bb] = _mm256_setzero_si256();
    }

    for (; j + 31 < cols; j += 32) {
        __m256i vw_gate = microgemm_avx2_unpack_i4_32(gate_row + (j >> 1));
        __m256i vw_up = microgemm_avx2_unpack_i4_32(up_row + (j >> 1));
        for (bb = 0; bb < tile; ++bb) {
            const int8_t* q = q_ptrs[bb];
            __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j));
            __m256i gate_prod;
            __m256i up_prod;
            gate_prod = _mm256_maddubs_epi16(vi, vw_gate);
            up_prod = _mm256_maddubs_epi16(vi, vw_up);
            gate_acc[bb] = _mm256_add_epi32(gate_acc[bb], _mm256_madd_epi16(gate_prod, vone16));
            up_acc[bb] = _mm256_add_epi32(up_acc[bb], _mm256_madd_epi16(up_prod, vone16));
        }
    }

    for (bb = 0; bb < tile; ++bb) {
        const int8_t* q = q_ptrs[bb];
        int32_t gate_i = microgemm_avx2_hsum_epi32(gate_acc[bb]) - 128 * gate_sum;
        int32_t up_i = microgemm_avx2_hsum_epi32(up_acc[bb]) - 128 * up_sum;
        float gate;
        float up;
        int tail;

        for (tail = j; tail < cols; ++tail) {
            int32_t qv = (int32_t)(uint8_t)q[tail];
            gate_i += qv * (int32_t)microgemm_i4_row_value(gate_row, tail);
            up_i += qv * (int32_t)microgemm_i4_row_value(up_row, tail);
        }

        gate = (float)gate_i * gate_scale * q_scales[bb];
        up = (float)up_i * up_scale * q_scales[bb];
        if (use_gelu) {
            float x = gate;
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        } else {
            gate = microgemm_silu(gate);
        }
        out[(size_t)(batch_offset + bb) * out_stride + out_col] = gate * up;
    }
}

static inline void microgemm_avx2_i4_gate_up_tile8_fused(
    float* out,
    int out_col,
    int out_stride,
    const uint8_t* gate_row,
    const uint8_t* up_row,
    float gate_scale,
    float up_scale,
    int gate_sum,
    int up_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int use_gelu
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const int8_t* q4 = input_q + (size_t)(batch_offset + 4) * cols;
    const int8_t* q5 = input_q + (size_t)(batch_offset + 5) * cols;
    const int8_t* q6 = input_q + (size_t)(batch_offset + 6) * cols;
    const int8_t* q7 = input_q + (size_t)(batch_offset + 7) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float qs4 = input_scales[batch_offset + 4];
    const float qs5 = input_scales[batch_offset + 5];
    const float qs6 = input_scales[batch_offset + 6];
    const float qs7 = input_scales[batch_offset + 7];
    const __m256i vone16 = _mm256_set1_epi16(1);
    __m256i gate_acc0 = _mm256_setzero_si256();
    __m256i gate_acc1 = _mm256_setzero_si256();
    __m256i gate_acc2 = _mm256_setzero_si256();
    __m256i gate_acc3 = _mm256_setzero_si256();
    __m256i gate_acc4 = _mm256_setzero_si256();
    __m256i gate_acc5 = _mm256_setzero_si256();
    __m256i gate_acc6 = _mm256_setzero_si256();
    __m256i gate_acc7 = _mm256_setzero_si256();
    __m256i up_acc0 = _mm256_setzero_si256();
    __m256i up_acc1 = _mm256_setzero_si256();
    __m256i up_acc2 = _mm256_setzero_si256();
    __m256i up_acc3 = _mm256_setzero_si256();
    __m256i up_acc4 = _mm256_setzero_si256();
    __m256i up_acc5 = _mm256_setzero_si256();
    __m256i up_acc6 = _mm256_setzero_si256();
    __m256i up_acc7 = _mm256_setzero_si256();
    int j = 0;

#define MICROGEMM_I4_GATE_UP_TILE8_FUSED_ACC(IDX) do { \
        __m256i vi = _mm256_loadu_si256((const __m256i*)(q##IDX + j)); \
        __m256i gate_prod = _mm256_maddubs_epi16(vi, vw_gate); \
        __m256i up_prod = _mm256_maddubs_epi16(vi, vw_up); \
        gate_acc##IDX = _mm256_add_epi32(gate_acc##IDX, _mm256_madd_epi16(gate_prod, vone16)); \
        up_acc##IDX = _mm256_add_epi32(up_acc##IDX, _mm256_madd_epi16(up_prod, vone16)); \
    } while (0)

    for (; j + 31 < cols; j += 32) {
        __m256i vw_gate = microgemm_avx2_unpack_i4_32(gate_row + (j >> 1));
        __m256i vw_up = microgemm_avx2_unpack_i4_32(up_row + (j >> 1));
        MICROGEMM_I4_GATE_UP_TILE8_FUSED_ACC(0);
        MICROGEMM_I4_GATE_UP_TILE8_FUSED_ACC(1);
        MICROGEMM_I4_GATE_UP_TILE8_FUSED_ACC(2);
        MICROGEMM_I4_GATE_UP_TILE8_FUSED_ACC(3);
        MICROGEMM_I4_GATE_UP_TILE8_FUSED_ACC(4);
        MICROGEMM_I4_GATE_UP_TILE8_FUSED_ACC(5);
        MICROGEMM_I4_GATE_UP_TILE8_FUSED_ACC(6);
        MICROGEMM_I4_GATE_UP_TILE8_FUSED_ACC(7);
    }

#define MICROGEMM_I4_GATE_UP_TILE8_FUSED_STORE(IDX) do { \
        int32_t gate_i = microgemm_avx2_hsum_epi32(gate_acc##IDX) - 128 * gate_sum; \
        int32_t up_i = microgemm_avx2_hsum_epi32(up_acc##IDX) - 128 * up_sum; \
        int tail; \
        float gate; \
        float up; \
        for (tail = j; tail < cols; ++tail) { \
            int32_t qv = (int32_t)(uint8_t)q##IDX[tail]; \
            gate_i += qv * (int32_t)microgemm_i4_row_value(gate_row, tail); \
            up_i += qv * (int32_t)microgemm_i4_row_value(up_row, tail); \
        } \
        gate = (float)gate_i * gate_scale * qs##IDX; \
        up = (float)up_i * up_scale * qs##IDX; \
        if (use_gelu) { \
            float x = gate; \
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate = microgemm_silu(gate); \
        } \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_col] = gate * up; \
    } while (0)

    MICROGEMM_I4_GATE_UP_TILE8_FUSED_STORE(0);
    MICROGEMM_I4_GATE_UP_TILE8_FUSED_STORE(1);
    MICROGEMM_I4_GATE_UP_TILE8_FUSED_STORE(2);
    MICROGEMM_I4_GATE_UP_TILE8_FUSED_STORE(3);
    MICROGEMM_I4_GATE_UP_TILE8_FUSED_STORE(4);
    MICROGEMM_I4_GATE_UP_TILE8_FUSED_STORE(5);
    MICROGEMM_I4_GATE_UP_TILE8_FUSED_STORE(6);
    MICROGEMM_I4_GATE_UP_TILE8_FUSED_STORE(7);

#undef MICROGEMM_I4_GATE_UP_TILE8_FUSED_STORE
#undef MICROGEMM_I4_GATE_UP_TILE8_FUSED_ACC
}

static inline void microgemm_avx2_i4_gate_up_tile8_split(
    float* out,
    int out_col,
    int out_stride,
    const uint8_t* gate_row,
    const uint8_t* up_row,
    float gate_scale,
    float up_scale,
    int gate_sum,
    int up_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int use_gelu
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const int8_t* q4 = input_q + (size_t)(batch_offset + 4) * cols;
    const int8_t* q5 = input_q + (size_t)(batch_offset + 5) * cols;
    const int8_t* q6 = input_q + (size_t)(batch_offset + 6) * cols;
    const int8_t* q7 = input_q + (size_t)(batch_offset + 7) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float qs4 = input_scales[batch_offset + 4];
    const float qs5 = input_scales[batch_offset + 5];
    const float qs6 = input_scales[batch_offset + 6];
    const float qs7 = input_scales[batch_offset + 7];
    const __m256i vone16 = _mm256_set1_epi16(1);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    __m256i acc4 = _mm256_setzero_si256();
    __m256i acc5 = _mm256_setzero_si256();
    __m256i acc6 = _mm256_setzero_si256();
    __m256i acc7 = _mm256_setzero_si256();
    float gate0;
    float gate1;
    float gate2;
    float gate3;
    float gate4;
    float gate5;
    float gate6;
    float gate7;
    int j = 0;

#define MICROGEMM_I4_GATE_UP_TILE8_ACC(IDX, ROW) do { \
        __m256i vi = _mm256_loadu_si256((const __m256i*)(q##IDX + j)); \
        __m256i prod; \
        prod = _mm256_maddubs_epi16(vi, ROW); \
        acc##IDX = _mm256_add_epi32(acc##IDX, _mm256_madd_epi16(prod, vone16)); \
    } while (0)

    for (; j + 31 < cols; j += 32) {
        __m256i vw_gate = microgemm_avx2_unpack_i4_32(gate_row + (j >> 1));
        MICROGEMM_I4_GATE_UP_TILE8_ACC(0, vw_gate);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(1, vw_gate);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(2, vw_gate);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(3, vw_gate);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(4, vw_gate);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(5, vw_gate);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(6, vw_gate);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(7, vw_gate);
    }

#define MICROGEMM_I4_GATE_UP_TILE8_FINISH_GATE(IDX) do { \
        int32_t gate_i = microgemm_avx2_hsum_epi32(acc##IDX) - 128 * gate_sum; \
        int tail; \
        for (tail = j; tail < cols; ++tail) { \
            gate_i += (int32_t)(uint8_t)q##IDX[tail] * (int32_t)microgemm_i4_row_value(gate_row, tail); \
        } \
        gate##IDX = (float)gate_i * gate_scale * qs##IDX; \
        if (use_gelu) { \
            float x = gate##IDX; \
            gate##IDX = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate##IDX = microgemm_silu(gate##IDX); \
        } \
    } while (0)

    MICROGEMM_I4_GATE_UP_TILE8_FINISH_GATE(0);
    MICROGEMM_I4_GATE_UP_TILE8_FINISH_GATE(1);
    MICROGEMM_I4_GATE_UP_TILE8_FINISH_GATE(2);
    MICROGEMM_I4_GATE_UP_TILE8_FINISH_GATE(3);
    MICROGEMM_I4_GATE_UP_TILE8_FINISH_GATE(4);
    MICROGEMM_I4_GATE_UP_TILE8_FINISH_GATE(5);
    MICROGEMM_I4_GATE_UP_TILE8_FINISH_GATE(6);
    MICROGEMM_I4_GATE_UP_TILE8_FINISH_GATE(7);

    acc0 = _mm256_setzero_si256();
    acc1 = _mm256_setzero_si256();
    acc2 = _mm256_setzero_si256();
    acc3 = _mm256_setzero_si256();
    acc4 = _mm256_setzero_si256();
    acc5 = _mm256_setzero_si256();
    acc6 = _mm256_setzero_si256();
    acc7 = _mm256_setzero_si256();
    j = 0;

    for (; j + 31 < cols; j += 32) {
        __m256i vw_up = microgemm_avx2_unpack_i4_32(up_row + (j >> 1));
        MICROGEMM_I4_GATE_UP_TILE8_ACC(0, vw_up);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(1, vw_up);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(2, vw_up);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(3, vw_up);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(4, vw_up);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(5, vw_up);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(6, vw_up);
        MICROGEMM_I4_GATE_UP_TILE8_ACC(7, vw_up);
    }

#define MICROGEMM_I4_GATE_UP_TILE8_STORE_UP(IDX) do { \
        int32_t up_i = microgemm_avx2_hsum_epi32(acc##IDX) - 128 * up_sum; \
        int tail; \
        float up; \
        for (tail = j; tail < cols; ++tail) { \
            up_i += (int32_t)(uint8_t)q##IDX[tail] * (int32_t)microgemm_i4_row_value(up_row, tail); \
        } \
        up = (float)up_i * up_scale * qs##IDX; \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_col] = gate##IDX * up; \
    } while (0)

    MICROGEMM_I4_GATE_UP_TILE8_STORE_UP(0);
    MICROGEMM_I4_GATE_UP_TILE8_STORE_UP(1);
    MICROGEMM_I4_GATE_UP_TILE8_STORE_UP(2);
    MICROGEMM_I4_GATE_UP_TILE8_STORE_UP(3);
    MICROGEMM_I4_GATE_UP_TILE8_STORE_UP(4);
    MICROGEMM_I4_GATE_UP_TILE8_STORE_UP(5);
    MICROGEMM_I4_GATE_UP_TILE8_STORE_UP(6);
    MICROGEMM_I4_GATE_UP_TILE8_STORE_UP(7);

#undef MICROGEMM_I4_GATE_UP_TILE8_STORE_UP
#undef MICROGEMM_I4_GATE_UP_TILE8_FINISH_GATE
#undef MICROGEMM_I4_GATE_UP_TILE8_ACC
}

static inline void microgemm_avx2_i8_gate_up_tile8_split(
    float* out,
    int out_col,
    int out_stride,
    const int8_t* gate_row,
    const int8_t* up_row,
    float gate_scale,
    float up_scale,
    int gate_sum,
    int up_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int use_gelu
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const int8_t* q4 = input_q + (size_t)(batch_offset + 4) * cols;
    const int8_t* q5 = input_q + (size_t)(batch_offset + 5) * cols;
    const int8_t* q6 = input_q + (size_t)(batch_offset + 6) * cols;
    const int8_t* q7 = input_q + (size_t)(batch_offset + 7) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float qs4 = input_scales[batch_offset + 4];
    const float qs5 = input_scales[batch_offset + 5];
    const float qs6 = input_scales[batch_offset + 6];
    const float qs7 = input_scales[batch_offset + 7];
    const __m256i vone16 = _mm256_set1_epi16(1);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    __m256i acc4 = _mm256_setzero_si256();
    __m256i acc5 = _mm256_setzero_si256();
    __m256i acc6 = _mm256_setzero_si256();
    __m256i acc7 = _mm256_setzero_si256();
    float gate0;
    float gate1;
    float gate2;
    float gate3;
    float gate4;
    float gate5;
    float gate6;
    float gate7;
    int j = 0;
    int use_unroll64 = microgemm_i8_gate_tile8_unroll64_enabled_for(8, 0, cols);

#define MICROGEMM_GATE_UP_TILE8_ACC(IDX, ROW) do { \
        __m256i vi = _mm256_loadu_si256((const __m256i*)(q##IDX + j)); \
        __m256i prod = _mm256_maddubs_epi16(vi, ROW); \
        acc##IDX = _mm256_add_epi32(acc##IDX, _mm256_madd_epi16(prod, vone16)); \
    } while (0)

#define MICROGEMM_GATE_UP_TILE8_ACC_OFF(IDX, ROW, OFF) do { \
        __m256i vi = _mm256_loadu_si256((const __m256i*)(q##IDX + j + (OFF))); \
        __m256i prod = _mm256_maddubs_epi16(vi, ROW); \
        acc##IDX = _mm256_add_epi32(acc##IDX, _mm256_madd_epi16(prod, vone16)); \
    } while (0)

    if (use_unroll64) {
        for (; j + 63 < cols; j += 64) {
            __m256i vw_gate0 = _mm256_loadu_si256((const __m256i*)(gate_row + j));
            __m256i vw_gate1 = _mm256_loadu_si256((const __m256i*)(gate_row + j + 32));
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(0, vw_gate0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(1, vw_gate0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(2, vw_gate0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(3, vw_gate0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(4, vw_gate0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(5, vw_gate0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(6, vw_gate0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(7, vw_gate0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(0, vw_gate1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(1, vw_gate1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(2, vw_gate1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(3, vw_gate1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(4, vw_gate1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(5, vw_gate1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(6, vw_gate1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(7, vw_gate1, 32);
        }
    }

    for (; j + 31 < cols; j += 32) {
        __m256i vw_gate = _mm256_loadu_si256((const __m256i*)(gate_row + j));
        MICROGEMM_GATE_UP_TILE8_ACC(0, vw_gate);
        MICROGEMM_GATE_UP_TILE8_ACC(1, vw_gate);
        MICROGEMM_GATE_UP_TILE8_ACC(2, vw_gate);
        MICROGEMM_GATE_UP_TILE8_ACC(3, vw_gate);
        MICROGEMM_GATE_UP_TILE8_ACC(4, vw_gate);
        MICROGEMM_GATE_UP_TILE8_ACC(5, vw_gate);
        MICROGEMM_GATE_UP_TILE8_ACC(6, vw_gate);
        MICROGEMM_GATE_UP_TILE8_ACC(7, vw_gate);
    }

#define MICROGEMM_GATE_UP_TILE8_FINISH_GATE(IDX) do { \
        int32_t gate_i = microgemm_avx2_hsum_epi32(acc##IDX) - 128 * gate_sum; \
        int tail; \
        for (tail = j; tail < cols; ++tail) { \
            gate_i += (int32_t)(uint8_t)q##IDX[tail] * (int32_t)gate_row[tail]; \
        } \
        gate##IDX = (float)gate_i * gate_scale * qs##IDX; \
        if (use_gelu) { \
            float x = gate##IDX; \
            gate##IDX = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate##IDX = microgemm_silu(gate##IDX); \
        } \
    } while (0)

    MICROGEMM_GATE_UP_TILE8_FINISH_GATE(0);
    MICROGEMM_GATE_UP_TILE8_FINISH_GATE(1);
    MICROGEMM_GATE_UP_TILE8_FINISH_GATE(2);
    MICROGEMM_GATE_UP_TILE8_FINISH_GATE(3);
    MICROGEMM_GATE_UP_TILE8_FINISH_GATE(4);
    MICROGEMM_GATE_UP_TILE8_FINISH_GATE(5);
    MICROGEMM_GATE_UP_TILE8_FINISH_GATE(6);
    MICROGEMM_GATE_UP_TILE8_FINISH_GATE(7);

    acc0 = _mm256_setzero_si256();
    acc1 = _mm256_setzero_si256();
    acc2 = _mm256_setzero_si256();
    acc3 = _mm256_setzero_si256();
    acc4 = _mm256_setzero_si256();
    acc5 = _mm256_setzero_si256();
    acc6 = _mm256_setzero_si256();
    acc7 = _mm256_setzero_si256();
    j = 0;

    if (use_unroll64) {
        for (; j + 63 < cols; j += 64) {
            __m256i vw_up0 = _mm256_loadu_si256((const __m256i*)(up_row + j));
            __m256i vw_up1 = _mm256_loadu_si256((const __m256i*)(up_row + j + 32));
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(0, vw_up0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(1, vw_up0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(2, vw_up0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(3, vw_up0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(4, vw_up0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(5, vw_up0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(6, vw_up0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(7, vw_up0, 0);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(0, vw_up1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(1, vw_up1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(2, vw_up1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(3, vw_up1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(4, vw_up1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(5, vw_up1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(6, vw_up1, 32);
            MICROGEMM_GATE_UP_TILE8_ACC_OFF(7, vw_up1, 32);
        }
    }

    for (; j + 31 < cols; j += 32) {
        __m256i vw_up = _mm256_loadu_si256((const __m256i*)(up_row + j));
        MICROGEMM_GATE_UP_TILE8_ACC(0, vw_up);
        MICROGEMM_GATE_UP_TILE8_ACC(1, vw_up);
        MICROGEMM_GATE_UP_TILE8_ACC(2, vw_up);
        MICROGEMM_GATE_UP_TILE8_ACC(3, vw_up);
        MICROGEMM_GATE_UP_TILE8_ACC(4, vw_up);
        MICROGEMM_GATE_UP_TILE8_ACC(5, vw_up);
        MICROGEMM_GATE_UP_TILE8_ACC(6, vw_up);
        MICROGEMM_GATE_UP_TILE8_ACC(7, vw_up);
    }

#define MICROGEMM_GATE_UP_TILE8_STORE_UP(IDX) do { \
        int32_t up_i = microgemm_avx2_hsum_epi32(acc##IDX) - 128 * up_sum; \
        int tail; \
        float up; \
        for (tail = j; tail < cols; ++tail) { \
            up_i += (int32_t)(uint8_t)q##IDX[tail] * (int32_t)up_row[tail]; \
        } \
        up = (float)up_i * up_scale * qs##IDX; \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_col] = gate##IDX * up; \
    } while (0)

    MICROGEMM_GATE_UP_TILE8_STORE_UP(0);
    MICROGEMM_GATE_UP_TILE8_STORE_UP(1);
    MICROGEMM_GATE_UP_TILE8_STORE_UP(2);
    MICROGEMM_GATE_UP_TILE8_STORE_UP(3);
    MICROGEMM_GATE_UP_TILE8_STORE_UP(4);
    MICROGEMM_GATE_UP_TILE8_STORE_UP(5);
    MICROGEMM_GATE_UP_TILE8_STORE_UP(6);
    MICROGEMM_GATE_UP_TILE8_STORE_UP(7);

#undef MICROGEMM_GATE_UP_TILE8_STORE_UP
#undef MICROGEMM_GATE_UP_TILE8_FINISH_GATE
#undef MICROGEMM_GATE_UP_TILE8_ACC_OFF
#undef MICROGEMM_GATE_UP_TILE8_ACC
}

static inline void microgemm_avx2_i8_gate_up_tile8_fused(
    float* out,
    int out_col,
    int out_stride,
    const int8_t* gate_row,
    const int8_t* up_row,
    float gate_scale,
    float up_scale,
    int gate_sum,
    int up_sum,
    const int8_t* input_q,
    const float* input_scales,
    int batch_offset,
    int cols,
    int use_gelu
) {
    const int8_t* q0 = input_q + (size_t)(batch_offset + 0) * cols;
    const int8_t* q1 = input_q + (size_t)(batch_offset + 1) * cols;
    const int8_t* q2 = input_q + (size_t)(batch_offset + 2) * cols;
    const int8_t* q3 = input_q + (size_t)(batch_offset + 3) * cols;
    const int8_t* q4 = input_q + (size_t)(batch_offset + 4) * cols;
    const int8_t* q5 = input_q + (size_t)(batch_offset + 5) * cols;
    const int8_t* q6 = input_q + (size_t)(batch_offset + 6) * cols;
    const int8_t* q7 = input_q + (size_t)(batch_offset + 7) * cols;
    const float qs0 = input_scales[batch_offset + 0];
    const float qs1 = input_scales[batch_offset + 1];
    const float qs2 = input_scales[batch_offset + 2];
    const float qs3 = input_scales[batch_offset + 3];
    const float qs4 = input_scales[batch_offset + 4];
    const float qs5 = input_scales[batch_offset + 5];
    const float qs6 = input_scales[batch_offset + 6];
    const float qs7 = input_scales[batch_offset + 7];
    const __m256i vone16 = _mm256_set1_epi16(1);
    __m256i gate_acc0 = _mm256_setzero_si256();
    __m256i gate_acc1 = _mm256_setzero_si256();
    __m256i gate_acc2 = _mm256_setzero_si256();
    __m256i gate_acc3 = _mm256_setzero_si256();
    __m256i gate_acc4 = _mm256_setzero_si256();
    __m256i gate_acc5 = _mm256_setzero_si256();
    __m256i gate_acc6 = _mm256_setzero_si256();
    __m256i gate_acc7 = _mm256_setzero_si256();
    __m256i up_acc0 = _mm256_setzero_si256();
    __m256i up_acc1 = _mm256_setzero_si256();
    __m256i up_acc2 = _mm256_setzero_si256();
    __m256i up_acc3 = _mm256_setzero_si256();
    __m256i up_acc4 = _mm256_setzero_si256();
    __m256i up_acc5 = _mm256_setzero_si256();
    __m256i up_acc6 = _mm256_setzero_si256();
    __m256i up_acc7 = _mm256_setzero_si256();
    int j = 0;

#define MICROGEMM_I8_GATE_UP_TILE8_FUSED_ACC(IDX) do { \
        __m256i vi = _mm256_loadu_si256((const __m256i*)(q##IDX + j)); \
        __m256i gate_prod = _mm256_maddubs_epi16(vi, vw_gate); \
        __m256i up_prod = _mm256_maddubs_epi16(vi, vw_up); \
        gate_acc##IDX = _mm256_add_epi32(gate_acc##IDX, _mm256_madd_epi16(gate_prod, vone16)); \
        up_acc##IDX = _mm256_add_epi32(up_acc##IDX, _mm256_madd_epi16(up_prod, vone16)); \
    } while (0)

    for (; j + 31 < cols; j += 32) {
        __m256i vw_gate = _mm256_loadu_si256((const __m256i*)(gate_row + j));
        __m256i vw_up = _mm256_loadu_si256((const __m256i*)(up_row + j));
        MICROGEMM_I8_GATE_UP_TILE8_FUSED_ACC(0);
        MICROGEMM_I8_GATE_UP_TILE8_FUSED_ACC(1);
        MICROGEMM_I8_GATE_UP_TILE8_FUSED_ACC(2);
        MICROGEMM_I8_GATE_UP_TILE8_FUSED_ACC(3);
        MICROGEMM_I8_GATE_UP_TILE8_FUSED_ACC(4);
        MICROGEMM_I8_GATE_UP_TILE8_FUSED_ACC(5);
        MICROGEMM_I8_GATE_UP_TILE8_FUSED_ACC(6);
        MICROGEMM_I8_GATE_UP_TILE8_FUSED_ACC(7);
    }

#undef MICROGEMM_I8_GATE_UP_TILE8_FUSED_ACC

#define MICROGEMM_I8_GATE_UP_TILE8_FUSED_STORE(IDX) do { \
        int32_t gate_i = microgemm_avx2_hsum_epi32(gate_acc##IDX) - 128 * gate_sum; \
        int32_t up_i = microgemm_avx2_hsum_epi32(up_acc##IDX) - 128 * up_sum; \
        float gate; \
        float up; \
        int tail; \
        for (tail = j; tail < cols; ++tail) { \
            int32_t qv = (int32_t)(uint8_t)q##IDX[tail]; \
            gate_i += qv * (int32_t)gate_row[tail]; \
            up_i += qv * (int32_t)up_row[tail]; \
        } \
        gate = (float)gate_i * gate_scale * qs##IDX; \
        up = (float)up_i * up_scale * qs##IDX; \
        if (use_gelu) { \
            float x = gate; \
            gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); \
        } else { \
            gate = microgemm_silu(gate); \
        } \
        out[(size_t)(batch_offset + (IDX)) * out_stride + out_col] = gate * up; \
    } while (0)

    MICROGEMM_I8_GATE_UP_TILE8_FUSED_STORE(0);
    MICROGEMM_I8_GATE_UP_TILE8_FUSED_STORE(1);
    MICROGEMM_I8_GATE_UP_TILE8_FUSED_STORE(2);
    MICROGEMM_I8_GATE_UP_TILE8_FUSED_STORE(3);
    MICROGEMM_I8_GATE_UP_TILE8_FUSED_STORE(4);
    MICROGEMM_I8_GATE_UP_TILE8_FUSED_STORE(5);
    MICROGEMM_I8_GATE_UP_TILE8_FUSED_STORE(6);
    MICROGEMM_I8_GATE_UP_TILE8_FUSED_STORE(7);

#undef MICROGEMM_I8_GATE_UP_TILE8_FUSED_STORE
}

#endif

static void microgemm_apply_rope_one(float* vec, const float* cos_v, const float* sin_v, int dim) {
    int half = dim / 2;
    int i;
    for (i = 0; i < half; ++i) {
        float a = vec[i];
        float b = vec[i + half];
        vec[i] = a * cos_v[i] - b * sin_v[i];
        vec[i + half] = a * sin_v[i] + b * cos_v[i];
    }
}

static void microgemm_apply_rope_one_interleaved(float* vec, const float* cos_v, const float* sin_v, int dim) {
    int pairs = dim / 2;
    int i;
    for (i = 0; i < pairs; ++i) {
        int j = 2 * i;
        float a = vec[j];
        float b = vec[j + 1];
        vec[j] = a * cos_v[i] - b * sin_v[i];
        vec[j + 1] = a * sin_v[i] + b * cos_v[i];
    }
}

static void microgemm_rope_half_rotate(
    float* q,
    float* k,
    const float* cos_v,
    const float* sin_v,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int h;
    for (h = 0; h < num_q_heads; ++h) {
        microgemm_apply_rope_one(q + (size_t)h * head_dim, cos_v, sin_v, head_dim);
    }
    for (h = 0; h < num_kv_heads; ++h) {
        microgemm_apply_rope_one(k + (size_t)h * head_dim, cos_v, sin_v, head_dim);
    }
}

static void microgemm_rope_half_rotate_partial(
    float* q,
    float* k,
    const float* cos_v,
    const float* sin_v,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim
) {
    int h;
    if (rotary_dim <= 0 || rotary_dim >= head_dim) {
        microgemm_rope_half_rotate(q, k, cos_v, sin_v, num_q_heads, num_kv_heads, head_dim);
        return;
    }
    for (h = 0; h < num_q_heads; ++h) {
        microgemm_apply_rope_one(q + (size_t)h * head_dim, cos_v, sin_v, rotary_dim);
    }
    for (h = 0; h < num_kv_heads; ++h) {
        microgemm_apply_rope_one(k + (size_t)h * head_dim, cos_v, sin_v, rotary_dim);
    }
}

static void microgemm_rope_rotate_partial(
    float* q,
    float* k,
    const float* cos_v,
    const float* sin_v,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    int interleaved
) {
    int h;
    int dim = (rotary_dim <= 0 || rotary_dim >= head_dim) ? head_dim : rotary_dim;

    if (!interleaved) {
        microgemm_rope_half_rotate_partial(
            q,
            k,
            cos_v,
            sin_v,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim
        );
        return;
    }

    for (h = 0; h < num_q_heads; ++h) {
        microgemm_apply_rope_one_interleaved(q + (size_t)h * head_dim, cos_v, sin_v, dim);
    }
    for (h = 0; h < num_kv_heads; ++h) {
        microgemm_apply_rope_one_interleaved(k + (size_t)h * head_dim, cos_v, sin_v, dim);
    }
}

static void microgemm_gemv_i8(
    float* out,
    const int8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int rows,
    int cols,
    const float* bias,
    int8_t* input_q
) {
    float input_scale = microgemm_quantize_activation_for_i8_self_biasing_dot(input_q, input, cols);
    int i;
    int parallel_rows = rows >= 128;

#if MICROGEMM_CPU_X86_AVX512_VNNI
    if (row_sums != NULL) {
        #pragma omp parallel for schedule(static) if(parallel_rows)
        for (i = 0; i < rows; ++i) {
            const int8_t* row = weights + (size_t)i * cols;
            int32_t acc;
            int j = 0;
            __m512i vacc0 = _mm512_setzero_si512();
            __m512i vacc1 = _mm512_setzero_si512();
            __m512i vacc2 = _mm512_setzero_si512();
            __m512i vacc3 = _mm512_setzero_si512();
            __m512i v128 = _mm512_set1_epi8((char)128u);

            for (; j + 255 < cols; j += 256) {
                __m512i vw0 = _mm512_loadu_si512((const void*)(row + j));
                __m512i vw1 = _mm512_loadu_si512((const void*)(row + j + 64));
                __m512i vw2 = _mm512_loadu_si512((const void*)(row + j + 128));
                __m512i vw3 = _mm512_loadu_si512((const void*)(row + j + 192));
                __m512i vi0 = _mm512_loadu_si512((const void*)(input_q + j));
                __m512i vi1 = _mm512_loadu_si512((const void*)(input_q + j + 64));
                __m512i vi2 = _mm512_loadu_si512((const void*)(input_q + j + 128));
                __m512i vi3 = _mm512_loadu_si512((const void*)(input_q + j + 192));
                vi0 = _mm512_add_epi8(vi0, v128);
                vi1 = _mm512_add_epi8(vi1, v128);
                vi2 = _mm512_add_epi8(vi2, v128);
                vi3 = _mm512_add_epi8(vi3, v128);
                vacc0 = _mm512_dpbusd_epi32(vacc0, vi0, vw0);
                vacc1 = _mm512_dpbusd_epi32(vacc1, vi1, vw1);
                vacc2 = _mm512_dpbusd_epi32(vacc2, vi2, vw2);
                vacc3 = _mm512_dpbusd_epi32(vacc3, vi3, vw3);
            }
            for (; j + 63 < cols; j += 64) {
                __m512i vw = _mm512_loadu_si512((const void*)(row + j));
                __m512i vi = _mm512_loadu_si512((const void*)(input_q + j));
                vi = _mm512_add_epi8(vi, v128);
                vacc0 = _mm512_dpbusd_epi32(vacc0, vi, vw);
            }
            vacc0 = _mm512_add_epi32(
                _mm512_add_epi32(vacc0, vacc1),
                _mm512_add_epi32(vacc2, vacc3)
            );
            acc = _mm512_reduce_add_epi32(vacc0) - 128 * row_sums[i];

            for (; j < cols; ++j) {
                acc += ((int32_t)input_q[j] + 128) * (int32_t)row[j];
            }
            out[i] = (float)acc * scales[i] * input_scale + (bias ? bias[i] : 0.0f);
        }
        return;
    }
#endif

#if MICROGEMM_CPU_X86_AVX_VNNI
    if (row_sums != NULL) {
        #pragma omp parallel for schedule(static) if(parallel_rows)
        for (i = 0; i < rows; ++i) {
            const int8_t* row = weights + (size_t)i * cols;
            int32_t acc = 0;
            int j = 0;
            __m256i vacc0 = _mm256_setzero_si256();
            __m256i vacc1 = _mm256_setzero_si256();
            __m256i vacc2 = _mm256_setzero_si256();
            __m256i vacc3 = _mm256_setzero_si256();
            __m256i v128 = _mm256_set1_epi8((char)128u);

            for (; j + 127 < cols; j += 128) {
                __m256i vw0 = _mm256_loadu_si256((const __m256i*)(row + j));
                __m256i vw1 = _mm256_loadu_si256((const __m256i*)(row + j + 32));
                __m256i vw2 = _mm256_loadu_si256((const __m256i*)(row + j + 64));
                __m256i vw3 = _mm256_loadu_si256((const __m256i*)(row + j + 96));
                __m256i vi0 = _mm256_loadu_si256((const __m256i*)(input_q + j));
                __m256i vi1 = _mm256_loadu_si256((const __m256i*)(input_q + j + 32));
                __m256i vi2 = _mm256_loadu_si256((const __m256i*)(input_q + j + 64));
                __m256i vi3 = _mm256_loadu_si256((const __m256i*)(input_q + j + 96));
                vi0 = _mm256_add_epi8(vi0, v128);
                vi1 = _mm256_add_epi8(vi1, v128);
                vi2 = _mm256_add_epi8(vi2, v128);
                vi3 = _mm256_add_epi8(vi3, v128);
                vacc0 = _mm256_dpbusd_epi32(vacc0, vi0, vw0);
                vacc1 = _mm256_dpbusd_epi32(vacc1, vi1, vw1);
                vacc2 = _mm256_dpbusd_epi32(vacc2, vi2, vw2);
                vacc3 = _mm256_dpbusd_epi32(vacc3, vi3, vw3);
            }
            for (; j + 31 < cols; j += 32) {
                __m256i vw = _mm256_loadu_si256((const __m256i*)(row + j));
                __m256i vi = _mm256_loadu_si256((const __m256i*)(input_q + j));
                vi = _mm256_add_epi8(vi, v128);
                vacc0 = _mm256_dpbusd_epi32(vacc0, vi, vw);
            }

            {
                __m256i vacc = _mm256_add_epi32(
                    _mm256_add_epi32(vacc0, vacc1),
                    _mm256_add_epi32(vacc2, vacc3)
                );
                __m128i hi = _mm256_extracti128_si256(vacc, 1);
                __m128i lo = _mm256_castsi256_si128(vacc);
                __m128i s4 = _mm_add_epi32(lo, hi);
                s4 = _mm_hadd_epi32(s4, s4);
                s4 = _mm_hadd_epi32(s4, s4);
                acc = _mm_cvtsi128_si32(s4) - 128 * row_sums[i];
            }

            for (; j < cols; ++j) {
                acc += ((int32_t)input_q[j] + 128) * (int32_t)row[j];
            }
            out[i] = (float)acc * scales[i] * input_scale + (bias ? bias[i] : 0.0f);
        }
        return;
    }
#endif

#if MICROGEMM_CPU_X86_AVX2
    if (row_sums != NULL) {
        #pragma omp parallel for schedule(static) if(parallel_rows)
        for (i = 0; i < rows; ++i) {
            const int8_t* row = weights + (size_t)i * cols;
            int32_t acc = 0;
            int j = 0;
            __m256i vacc0 = _mm256_setzero_si256();
            __m256i vacc1 = _mm256_setzero_si256();
            __m256i vacc2 = _mm256_setzero_si256();
            __m256i vacc3 = _mm256_setzero_si256();
            __m256i v128 = _mm256_set1_epi8((char)128u);
            __m256i vone16 = _mm256_set1_epi16(1);

            for (; j + 127 < cols; j += 128) {
                __m256i vw0 = _mm256_loadu_si256((const __m256i*)(row + j));
                __m256i vw1 = _mm256_loadu_si256((const __m256i*)(row + j + 32));
                __m256i vw2 = _mm256_loadu_si256((const __m256i*)(row + j + 64));
                __m256i vw3 = _mm256_loadu_si256((const __m256i*)(row + j + 96));
                __m256i vi0 = _mm256_loadu_si256((const __m256i*)(input_q + j));
                __m256i vi1 = _mm256_loadu_si256((const __m256i*)(input_q + j + 32));
                __m256i vi2 = _mm256_loadu_si256((const __m256i*)(input_q + j + 64));
                __m256i vi3 = _mm256_loadu_si256((const __m256i*)(input_q + j + 96));
                vi0 = _mm256_add_epi8(vi0, v128);
                vi1 = _mm256_add_epi8(vi1, v128);
                vi2 = _mm256_add_epi8(vi2, v128);
                vi3 = _mm256_add_epi8(vi3, v128);
                vacc0 = _mm256_add_epi32(
                    vacc0,
                    _mm256_madd_epi16(_mm256_maddubs_epi16(vi0, vw0), vone16)
                );
                vacc1 = _mm256_add_epi32(
                    vacc1,
                    _mm256_madd_epi16(_mm256_maddubs_epi16(vi1, vw1), vone16)
                );
                vacc2 = _mm256_add_epi32(
                    vacc2,
                    _mm256_madd_epi16(_mm256_maddubs_epi16(vi2, vw2), vone16)
                );
                vacc3 = _mm256_add_epi32(
                    vacc3,
                    _mm256_madd_epi16(_mm256_maddubs_epi16(vi3, vw3), vone16)
                );
            }
            for (; j + 31 < cols; j += 32) {
                __m256i vw = _mm256_loadu_si256((const __m256i*)(row + j));
                __m256i vi = _mm256_loadu_si256((const __m256i*)(input_q + j));
                __m256i vi_u = _mm256_add_epi8(vi, v128);
                __m256i prod = _mm256_maddubs_epi16(vi_u, vw);
                vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(prod, vone16));
            }

            {
                __m256i vacc = _mm256_add_epi32(
                    _mm256_add_epi32(vacc0, vacc1),
                    _mm256_add_epi32(vacc2, vacc3)
                );
                __m128i hi = _mm256_extracti128_si256(vacc, 1);
                __m128i lo = _mm256_castsi256_si128(vacc);
                __m128i s4 = _mm_add_epi32(lo, hi);
                s4 = _mm_hadd_epi32(s4, s4);
                s4 = _mm_hadd_epi32(s4, s4);
                acc = _mm_cvtsi128_si32(s4) - 128 * row_sums[i];
            }

            for (; j < cols; ++j) {
                acc += ((int32_t)input_q[j] + 128) * (int32_t)row[j];
            }
            out[i] = (float)acc * scales[i] * input_scale + (bias ? bias[i] : 0.0f);
        }
        return;
    }
#endif

    #pragma omp parallel for schedule(static) if(parallel_rows)
    for (i = 0; i < rows; ++i) {
        const int8_t* row = weights + (size_t)i * cols;
        int32_t acc = 0;
        int j = 0;

#if MICROGEMM_CPU_X86_AVX2
        __m256i vacc = _mm256_setzero_si256();
        __m256i vcomp = _mm256_setzero_si256();
        __m256i v128 = _mm256_set1_epi8((char)128u);
        __m256i vone16 = _mm256_set1_epi16(1);
        int use_precomputed_sum = row_sums != NULL;

        for (; j + 31 < cols; j += 32) {
            __m256i vw = _mm256_loadu_si256((const __m256i*)(row + j));
            __m256i vi = _mm256_loadu_si256((const __m256i*)(input_q + j));
            __m256i vi_u = _mm256_add_epi8(vi, v128);
            __m256i prod = _mm256_maddubs_epi16(vi_u, vw);
            vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(prod, vone16));

            if (!use_precomputed_sum) {
                __m256i wext = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_set1_epi8(1), vw),
                    vone16
                );
                vcomp = _mm256_add_epi32(vcomp, wext);
            }
        }

        {
            __m128i hi = _mm256_extracti128_si256(vacc, 1);
            __m128i lo = _mm256_castsi256_si128(vacc);
            __m128i s4 = _mm_add_epi32(lo, hi);
            s4 = _mm_hadd_epi32(s4, s4);
            s4 = _mm_hadd_epi32(s4, s4);
            acc = _mm_cvtsi128_si32(s4);

            if (use_precomputed_sum) {
                acc -= 128 * row_sums[i];
            } else {
                hi = _mm256_extracti128_si256(vcomp, 1);
                lo = _mm256_castsi256_si128(vcomp);
                s4 = _mm_add_epi32(lo, hi);
                s4 = _mm_hadd_epi32(s4, s4);
                s4 = _mm_hadd_epi32(s4, s4);
                acc -= 128 * _mm_cvtsi128_si32(s4);
            }
        }
#elif MICROGEMM_CPU_ARM64_NEON
        {
            int32x4_t vacc0 = vdupq_n_s32(0);
            int32x4_t vacc1 = vdupq_n_s32(0);
            for (; j + 15 < cols; j += 16) {
                int8x16_t vw = vld1q_s8(row + j);
                int8x16_t vi = vld1q_s8(input_q + j);
                int16x8_t prod0 = vmull_s8(vget_low_s8(vi), vget_low_s8(vw));
                int16x8_t prod1 = vmull_s8(vget_high_s8(vi), vget_high_s8(vw));
                vacc0 = vpadalq_s16(vacc0, prod0);
                vacc1 = vpadalq_s16(vacc1, prod1);
            }
            acc = microgemm_neon_hsum_s32(vaddq_s32(vacc0, vacc1));
        }
#endif

        for (; j < cols; ++j) {
#if MICROGEMM_CPU_X86_AVX2
            if (row_sums != NULL) {
                acc += ((int32_t)input_q[j] + 128) * (int32_t)row[j];
            } else {
                acc += (int32_t)input_q[j] * (int32_t)row[j];
            }
#else
            acc += (int32_t)input_q[j] * (int32_t)row[j];
#endif
        }
        out[i] = (float)acc * scales[i] * input_scale + (bias ? bias[i] : 0.0f);
    }
}

static int32_t microgemm_dot_i4_biased_i8_scalar(
    const uint8_t* row,
    const int8_t* input_q,
    int cols,
    int row_sum
) {
    int32_t acc = -128 * row_sum;
    int j;
    for (j = 0; j < cols; ++j) {
        acc += ((int32_t)input_q[j] + 128) * (int32_t)microgemm_i4_row_value(row, j);
    }
    return acc;
}

static void microgemm_gemv_i4_packed(
    float* out,
    const uint8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int rows,
    int cols,
    const float* bias,
    int8_t* input_q
) {
#if MICROGEMM_CPU_X86_AVX2
    float input_scale = microgemm_quantize_activation_for_prebiased_maddubs_dot(input_q, input, cols);
#else
    float input_scale = microgemm_quantize_activation_for_i8_self_biasing_dot(input_q, input, cols);
#endif
    size_t row_bytes = (size_t)((cols + 1) / 2);
    int parallel_rows = rows >= 128;
    int i;

    #pragma omp parallel for schedule(static) if(parallel_rows)
    for (i = 0; i < rows; ++i) {
        const uint8_t* row = weights + (size_t)i * row_bytes;
        int row_sum = row_sums ? row_sums[i] : 0;
        int32_t acc;
        if (row_sums == NULL) {
            int j;
            for (j = 0; j < cols; ++j) {
                row_sum += (int)microgemm_i4_row_value(row, j);
            }
        }
#if MICROGEMM_CPU_X86_AVX2
        acc = microgemm_avx2_dot_i4_biased_i8(row, input_q, cols, row_sum);
#else
        acc = microgemm_dot_i4_biased_i8_scalar(row, input_q, cols, row_sum);
#endif
        out[i] = (float)acc * scales[i] * input_scale + (bias ? bias[i] : 0.0f);
    }
}

static void microgemm_gemv_i8_groupwise(
    float* out,
    const int8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int rows,
    int cols,
    const float* bias,
    int8_t* input_q
) {
    const int groups = microgemm_quant_group_count_int(cols);
    float input_scale = microgemm_quantize_activation_for_i8_self_biasing_dot(input_q, input, cols);
    int parallel_rows = rows >= 128;
    int i;

    #pragma omp parallel for schedule(static) if(parallel_rows)
    for (i = 0; i < rows; ++i) {
        const int8_t* row = weights + (size_t)i * cols;
        const float* row_scales = scales + (size_t)i * groups;
        const int32_t* row_group_sums = row_sums ? row_sums + (size_t)i * groups : NULL;
        float value = bias ? bias[i] : 0.0f;
        int group;

        for (group = 0; group < groups; ++group) {
            int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            int group_cols;
            int group_sum = 0;
            int32_t acc;

            if (end > cols) {
                end = cols;
            }
            group_cols = end - begin;
            if (row_group_sums != NULL) {
                group_sum = row_group_sums[group];
            } else {
                int j;
                for (j = begin; j < end; ++j) {
                    group_sum += (int)row[j];
                }
            }
            acc = microgemm_dot_i8_biased_i8(row + begin, input_q + begin, group_cols, group_sum);
            value += (float)acc * row_scales[group] * input_scale;
        }
        out[i] = value;
    }
}

static void microgemm_gemv_i4_groupwise(
    float* out,
    const uint8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int rows,
    int cols,
    const float* bias,
    int8_t* input_q
) {
    const int groups = microgemm_quant_group_count_int(cols);
    const size_t row_bytes = (size_t)((cols + 1) / 2);
    float input_scale;
    int parallel_rows = rows >= 128;
    int i;

#if MICROGEMM_CPU_X86_AVX2
    input_scale = microgemm_quantize_activation_for_prebiased_maddubs_dot(input_q, input, cols);
#else
    input_scale = microgemm_quantize_activation_for_i8_self_biasing_dot(input_q, input, cols);
#endif

    #pragma omp parallel for schedule(static) if(parallel_rows)
    for (i = 0; i < rows; ++i) {
        const uint8_t* row = weights + (size_t)i * row_bytes;
        const float* row_scales = scales + (size_t)i * groups;
        const int32_t* row_group_sums = row_sums ? row_sums + (size_t)i * groups : NULL;
        float value = bias ? bias[i] : 0.0f;
        int group;

        for (group = 0; group < groups; ++group) {
            int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            int group_cols;
            int group_sum = 0;
            int32_t acc;

            if (end > cols) {
                end = cols;
            }
            group_cols = end - begin;
            if (row_group_sums != NULL) {
                group_sum = row_group_sums[group];
            } else {
                int j;
                for (j = begin; j < end; ++j) {
                    group_sum += (int)microgemm_i4_row_value(row, j);
                }
            }
#if MICROGEMM_CPU_X86_AVX2
            acc = microgemm_avx2_dot_i4_biased_i8(
                row + (begin >> 1), input_q + begin, group_cols, group_sum
            );
#else
            acc = microgemm_dot_i4_biased_i8_scalar(
                row + (begin >> 1), input_q + begin, group_cols, group_sum
            );
#endif
            value += (float)acc * row_scales[group] * input_scale;
        }
        out[i] = value;
    }
}

static void microgemm_gemv_quantized(
    const microgemm_config* config,
    float* out,
    const int8_t* weights_i8,
    const uint8_t* weights_i4,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int rows,
    int cols,
    const float* bias,
    int8_t* input_q
) {
    if (config != NULL && config->quant_mode == MICROGEMM_QUANT_INT8G128) {
        microgemm_gemv_i8_groupwise(
            out, weights_i8, scales, row_sums, input, rows, cols, bias, input_q
        );
        return;
    }
    if (config != NULL && config->quant_mode == MICROGEMM_QUANT_INT4G128) {
        microgemm_gemv_i4_groupwise(
            out, weights_i4, scales, row_sums, input, rows, cols, bias, input_q
        );
        return;
    }
    if (config != NULL && config->quant_mode == MICROGEMM_QUANT_INT4) {
        microgemm_gemv_i4_packed(
            out, weights_i4, scales, row_sums, input, rows, cols, bias, input_q
        );
        return;
    }
    microgemm_gemv_i8(
        out, weights_i8, scales, row_sums, input, rows, cols, bias, input_q
    );
}

static void microgemm_gemv_i8_batched_impl(
    float* out,
    const int8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int batch,
    int rows,
    int cols,
    int input_stride,
    const float* bias,
    int8_t* input_q,
    float* input_scales,
    int input_prequantized
) {
    int b;
    int i;
    int profile_down_proj = microgemm_decode_batch_profile_enabled && microgemm_profile_next_down_proj_gemv;
    double profile_phase_start = 0.0;

    if (batch <= 0) {
        return;
    }

    if (profile_down_proj) {
        profile_phase_start = microgemm_profile_now_ms();
    }

    if (!input_prequantized) {
        #pragma omp parallel for schedule(static) if(batch >= 8)
        for (b = 0; b < batch; ++b) {
            input_scales[b] = microgemm_quantize_activation_for_i8_self_biasing_dot(
                input_q + (size_t)b * cols,
                input + (size_t)b * input_stride,
                cols
            );
        }
    }

    if (profile_down_proj && !input_prequantized) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_quant_ms, profile_phase_start);
        profile_phase_start = microgemm_profile_now_ms();
    }

#if MICROGEMM_CPU_X86_AVX512_VNNI
    if (row_sums != NULL) {
        #pragma omp parallel for schedule(static) if(rows >= 96)
        for (i = 0; i < rows; ++i) {
            const int8_t* row = weights + (size_t)i * cols;
            float row_scale = scales[i];
            int row_sum = row_sums[i];

            for (b = 0; b < batch; ++b) {
                const int8_t* q = input_q + (size_t)b * cols;
                int32_t acc;
                int j = 0;
                __m512i vacc0 = _mm512_setzero_si512();
                __m512i vacc1 = _mm512_setzero_si512();
                __m512i vacc2 = _mm512_setzero_si512();
                __m512i vacc3 = _mm512_setzero_si512();
                __m512i v128 = _mm512_set1_epi8((char)128u);

                for (; j + 255 < cols; j += 256) {
                    __m512i vw0 = _mm512_loadu_si512((const void*)(row + j));
                    __m512i vw1 = _mm512_loadu_si512((const void*)(row + j + 64));
                    __m512i vw2 = _mm512_loadu_si512((const void*)(row + j + 128));
                    __m512i vw3 = _mm512_loadu_si512((const void*)(row + j + 192));
                    __m512i vi0 = _mm512_loadu_si512((const void*)(q + j));
                    __m512i vi1 = _mm512_loadu_si512((const void*)(q + j + 64));
                    __m512i vi2 = _mm512_loadu_si512((const void*)(q + j + 128));
                    __m512i vi3 = _mm512_loadu_si512((const void*)(q + j + 192));
                    vi0 = _mm512_add_epi8(vi0, v128);
                    vi1 = _mm512_add_epi8(vi1, v128);
                    vi2 = _mm512_add_epi8(vi2, v128);
                    vi3 = _mm512_add_epi8(vi3, v128);
                    vacc0 = _mm512_dpbusd_epi32(vacc0, vi0, vw0);
                    vacc1 = _mm512_dpbusd_epi32(vacc1, vi1, vw1);
                    vacc2 = _mm512_dpbusd_epi32(vacc2, vi2, vw2);
                    vacc3 = _mm512_dpbusd_epi32(vacc3, vi3, vw3);
                }
                for (; j + 63 < cols; j += 64) {
                    __m512i vw = _mm512_loadu_si512((const void*)(row + j));
                    __m512i vi = _mm512_loadu_si512((const void*)(q + j));
                    vi = _mm512_add_epi8(vi, v128);
                    vacc0 = _mm512_dpbusd_epi32(vacc0, vi, vw);
                }
                vacc0 = _mm512_add_epi32(
                    _mm512_add_epi32(vacc0, vacc1),
                    _mm512_add_epi32(vacc2, vacc3)
                );
                acc = _mm512_reduce_add_epi32(vacc0) - 128 * row_sum;

                for (; j < cols; ++j) {
                    acc += ((int32_t)q[j] + 128) * (int32_t)row[j];
                }

                out[(size_t)b * rows + i] =
                    (float)acc * row_scale * input_scales[b] + (bias ? bias[i] : 0.0f);
            }
        }
        return;
    }
#endif

#if MICROGEMM_CPU_X86_AVX_VNNI
    if (row_sums != NULL) {
        #pragma omp parallel for schedule(static) if(rows >= 96)
        for (i = 0; i < rows; ++i) {
            const int8_t* row = weights + (size_t)i * cols;
            float row_scale = scales[i];
            int row_sum = row_sums[i];

            for (b = 0; b < batch; ++b) {
                const int8_t* q = input_q + (size_t)b * cols;
                int32_t acc = 0;
                int j = 0;
                __m256i vacc0 = _mm256_setzero_si256();
                __m256i vacc1 = _mm256_setzero_si256();
                __m256i vacc2 = _mm256_setzero_si256();
                __m256i vacc3 = _mm256_setzero_si256();
                __m256i v128 = _mm256_set1_epi8((char)128u);

                for (; j + 127 < cols; j += 128) {
                    __m256i vw0 = _mm256_loadu_si256((const __m256i*)(row + j));
                    __m256i vw1 = _mm256_loadu_si256((const __m256i*)(row + j + 32));
                    __m256i vw2 = _mm256_loadu_si256((const __m256i*)(row + j + 64));
                    __m256i vw3 = _mm256_loadu_si256((const __m256i*)(row + j + 96));
                    __m256i vi0 = _mm256_loadu_si256((const __m256i*)(q + j));
                    __m256i vi1 = _mm256_loadu_si256((const __m256i*)(q + j + 32));
                    __m256i vi2 = _mm256_loadu_si256((const __m256i*)(q + j + 64));
                    __m256i vi3 = _mm256_loadu_si256((const __m256i*)(q + j + 96));
                    vi0 = _mm256_add_epi8(vi0, v128);
                    vi1 = _mm256_add_epi8(vi1, v128);
                    vi2 = _mm256_add_epi8(vi2, v128);
                    vi3 = _mm256_add_epi8(vi3, v128);
                    vacc0 = _mm256_dpbusd_epi32(vacc0, vi0, vw0);
                    vacc1 = _mm256_dpbusd_epi32(vacc1, vi1, vw1);
                    vacc2 = _mm256_dpbusd_epi32(vacc2, vi2, vw2);
                    vacc3 = _mm256_dpbusd_epi32(vacc3, vi3, vw3);
                }
                for (; j + 31 < cols; j += 32) {
                    __m256i vw = _mm256_loadu_si256((const __m256i*)(row + j));
                    __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j));
                    vi = _mm256_add_epi8(vi, v128);
                    vacc0 = _mm256_dpbusd_epi32(vacc0, vi, vw);
                }

                {
                    __m256i vacc = _mm256_add_epi32(
                        _mm256_add_epi32(vacc0, vacc1),
                        _mm256_add_epi32(vacc2, vacc3)
                    );
                    __m128i hi = _mm256_extracti128_si256(vacc, 1);
                    __m128i lo = _mm256_castsi256_si128(vacc);
                    __m128i s4 = _mm_add_epi32(lo, hi);
                    s4 = _mm_hadd_epi32(s4, s4);
                    s4 = _mm_hadd_epi32(s4, s4);
                    acc = _mm_cvtsi128_si32(s4) - 128 * row_sum;
                }

                for (; j < cols; ++j) {
                    acc += ((int32_t)q[j] + 128) * (int32_t)row[j];
                }

                out[(size_t)b * rows + i] =
                    (float)acc * row_scale * input_scales[b] + (bias ? bias[i] : 0.0f);
            }
        }
        return;
    }
#endif

#if MICROGEMM_CPU_X86_AVX2
    if (row_sums != NULL) {
        if (batch >= 8) {
            int pair_count = rows / 2;
            int use_down_tile8 = microgemm_i8_down_tile8_enabled_for(batch, rows, cols);
            #pragma omp parallel for schedule(static) if(rows >= 96)
            for (i = 0; i < pair_count; ++i) {
                int row_idx = i * 2;
                const int8_t* row0 = weights + (size_t)row_idx * cols;
                const int8_t* row1 = row0 + cols;
                int b0 = 0;

                if (rows >= 32768 || (rows >= 8192 && cols >= 2048) || use_down_tile8) {
                    for (; b0 + 7 < batch; b0 += 8) {
                        microgemm_avx2_i8_batch_row_pair_tile8_split(
                            out, row_idx, rows, row0, row1,
                            scales[row_idx], scales[row_idx + 1],
                            row_sums[row_idx], row_sums[row_idx + 1],
                            input_q, input_scales, b0, cols, bias
                        );
                    }
                } else if (rows >= 512 && cols >= 2048) {
                    for (; b0 + 3 < batch; b0 += 4) {
                        microgemm_avx2_i8_batch_row_pair_tile4_split(
                            out, row_idx, rows, row0, row1,
                            scales[row_idx], scales[row_idx + 1],
                            row_sums[row_idx], row_sums[row_idx + 1],
                            input_q, input_scales, b0, cols, bias
                        );
                    }
                }
                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i8_batch_row_pair_tile(
                        out, row_idx, rows, row0, row1,
                        scales[row_idx], scales[row_idx + 1],
                        row_sums[row_idx], row_sums[row_idx + 1],
                        input_q, input_scales, b0, 4, cols, bias
                    );
                }
                for (; b0 + 1 < batch; b0 += 2) {
                    microgemm_avx2_i8_batch_row_pair_tile(
                        out, row_idx, rows, row0, row1,
                        scales[row_idx], scales[row_idx + 1],
                        row_sums[row_idx], row_sums[row_idx + 1],
                        input_q, input_scales, b0, 2, cols, bias
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i8_batch_row_pair_tile(
                        out, row_idx, rows, row0, row1,
                        scales[row_idx], scales[row_idx + 1],
                        row_sums[row_idx], row_sums[row_idx + 1],
                        input_q, input_scales, b0, 1, cols, bias
                    );
                }
            }
            if ((rows & 1) != 0) {
                int row_idx = rows - 1;
                const int8_t* row = weights + (size_t)row_idx * cols;
                float row_scale = scales[row_idx];
                int row_sum = row_sums[row_idx];
                int b0 = 0;
                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i8_batch_row_tile(
                        out, row_idx, rows, row, row_scale, row_sum,
                        input_q, input_scales, b0, 4, cols, bias
                    );
                }
                for (; b0 + 1 < batch; b0 += 2) {
                    microgemm_avx2_i8_batch_row_tile(
                        out, row_idx, rows, row, row_scale, row_sum,
                        input_q, input_scales, b0, 2, cols, bias
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i8_batch_row_tile(
                        out, row_idx, rows, row, row_scale, row_sum,
                        input_q, input_scales, b0, 1, cols, bias
                    );
                }
            }
            if (profile_down_proj) {
                microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
            }
            return;
        }

        if (batch >= 4 && rows >= 512 && cols >= 2048) {
            int pair_count = rows / 2;
            #pragma omp parallel for schedule(static) if(rows >= 96)
            for (i = 0; i < pair_count; ++i) {
                int row_idx = i * 2;
                const int8_t* row0 = weights + (size_t)row_idx * cols;
                const int8_t* row1 = row0 + cols;
                int b0 = 0;

                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i8_batch_row_pair_tile4_split(
                        out, row_idx, rows, row0, row1,
                        scales[row_idx], scales[row_idx + 1],
                        row_sums[row_idx], row_sums[row_idx + 1],
                        input_q, input_scales, b0, cols, bias
                    );
                }
                for (; b0 + 1 < batch; b0 += 2) {
                    microgemm_avx2_i8_batch_row_pair_tile(
                        out, row_idx, rows, row0, row1,
                        scales[row_idx], scales[row_idx + 1],
                        row_sums[row_idx], row_sums[row_idx + 1],
                        input_q, input_scales, b0, 2, cols, bias
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i8_batch_row_pair_tile(
                        out, row_idx, rows, row0, row1,
                        scales[row_idx], scales[row_idx + 1],
                        row_sums[row_idx], row_sums[row_idx + 1],
                        input_q, input_scales, b0, 1, cols, bias
                    );
                }
            }
            if ((rows & 1) != 0) {
                int row_idx = rows - 1;
                const int8_t* row = weights + (size_t)row_idx * cols;
                float row_scale = scales[row_idx];
                int row_sum = row_sums[row_idx];
                int b0 = 0;
                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i8_batch_row_tile(
                        out, row_idx, rows, row, row_scale, row_sum,
                        input_q, input_scales, b0, 4, cols, bias
                    );
                }
                for (; b0 + 1 < batch; b0 += 2) {
                    microgemm_avx2_i8_batch_row_tile(
                        out, row_idx, rows, row, row_scale, row_sum,
                        input_q, input_scales, b0, 2, cols, bias
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i8_batch_row_tile(
                        out, row_idx, rows, row, row_scale, row_sum,
                        input_q, input_scales, b0, 1, cols, bias
                    );
                }
            }
            if (profile_down_proj) {
                microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
            }
            return;
        }

        if (batch >= 2 && rows >= 512) {
            int pair_count = rows / 2;
            #pragma omp parallel for schedule(static) if(rows >= 96)
            for (i = 0; i < pair_count; ++i) {
                int row_idx = i * 2;
                const int8_t* row0 = weights + (size_t)row_idx * cols;
                const int8_t* row1 = row0 + cols;
                int b0 = 0;

                for (; b0 + 1 < batch; b0 += 2) {
                    microgemm_avx2_i8_batch_row_pair_tile(
                        out, row_idx, rows, row0, row1,
                        scales[row_idx], scales[row_idx + 1],
                        row_sums[row_idx], row_sums[row_idx + 1],
                        input_q, input_scales, b0, 2, cols, bias
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i8_batch_row_pair_tile(
                        out, row_idx, rows, row0, row1,
                        scales[row_idx], scales[row_idx + 1],
                        row_sums[row_idx], row_sums[row_idx + 1],
                        input_q, input_scales, b0, 1, cols, bias
                    );
                }
            }
            if ((rows & 1) != 0) {
                int row_idx = rows - 1;
                const int8_t* row = weights + (size_t)row_idx * cols;
                float row_scale = scales[row_idx];
                int row_sum = row_sums[row_idx];
                int b0 = 0;
                for (; b0 + 1 < batch; b0 += 2) {
                    microgemm_avx2_i8_batch_row_tile(
                        out, row_idx, rows, row, row_scale, row_sum,
                        input_q, input_scales, b0, 2, cols, bias
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i8_batch_row_tile(
                        out, row_idx, rows, row, row_scale, row_sum,
                        input_q, input_scales, b0, 1, cols, bias
                    );
                }
            }
            if (profile_down_proj) {
                microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
            }
            return;
        }

        #pragma omp parallel for schedule(static) if(rows >= 96)
        for (i = 0; i < rows; ++i) {
            const int8_t* row = weights + (size_t)i * cols;
            float row_scale = scales[i];
            int row_sum = row_sums[i];
            int b0 = 0;

            for (; b0 + 3 < batch; b0 += 4) {
                microgemm_avx2_i8_batch_row_tile(
                    out, i, rows, row, row_scale, row_sum,
                    input_q, input_scales, b0, 4, cols, bias
                );
            }
            for (; b0 + 1 < batch; b0 += 2) {
                microgemm_avx2_i8_batch_row_tile(
                    out, i, rows, row, row_scale, row_sum,
                    input_q, input_scales, b0, 2, cols, bias
                );
            }
            if (b0 < batch) {
                microgemm_avx2_i8_batch_row_tile(
                    out, i, rows, row, row_scale, row_sum,
                    input_q, input_scales, b0, 1, cols, bias
                );
            }
        }
        if (profile_down_proj) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
        }
        return;
    }
#endif

    #pragma omp parallel for schedule(static) if(rows >= 96)
    for (i = 0; i < rows; ++i) {
        const int8_t* row = weights + (size_t)i * cols;
        float row_scale = scales[i];
        int row_sum = row_sums ? row_sums[i] : 0;

        for (b = 0; b < batch; ++b) {
            const int8_t* q = input_q + (size_t)b * cols;
            int32_t acc = 0;
            int j = 0;

#if MICROGEMM_CPU_X86_AVX2
            __m256i vacc = _mm256_setzero_si256();
            __m256i vcomp = _mm256_setzero_si256();
            __m256i v128 = _mm256_set1_epi8((char)128u);
            __m256i vone16 = _mm256_set1_epi16(1);
            int use_precomputed_sum = row_sums != NULL;

            for (; j + 31 < cols; j += 32) {
                __m256i vw = _mm256_loadu_si256((const __m256i*)(row + j));
                __m256i vi = _mm256_loadu_si256((const __m256i*)(q + j));
                __m256i vi_u = _mm256_add_epi8(vi, v128);
                __m256i prod = _mm256_maddubs_epi16(vi_u, vw);
                vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(prod, vone16));

                if (!use_precomputed_sum) {
                    __m256i wext = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(_mm256_set1_epi8(1), vw),
                        vone16
                    );
                    vcomp = _mm256_add_epi32(vcomp, wext);
                }
            }

            {
                __m128i hi = _mm256_extracti128_si256(vacc, 1);
                __m128i lo = _mm256_castsi256_si128(vacc);
                __m128i s4 = _mm_add_epi32(lo, hi);
                s4 = _mm_hadd_epi32(s4, s4);
                s4 = _mm_hadd_epi32(s4, s4);
                acc = _mm_cvtsi128_si32(s4);

                if (use_precomputed_sum) {
                    acc -= 128 * row_sum;
                } else {
                    hi = _mm256_extracti128_si256(vcomp, 1);
                    lo = _mm256_castsi256_si128(vcomp);
                    s4 = _mm_add_epi32(lo, hi);
                    s4 = _mm_hadd_epi32(s4, s4);
                    s4 = _mm_hadd_epi32(s4, s4);
                    acc -= 128 * _mm_cvtsi128_si32(s4);
                }
            }
#elif MICROGEMM_CPU_ARM64_NEON
            {
                int32x4_t vacc0 = vdupq_n_s32(0);
                int32x4_t vacc1 = vdupq_n_s32(0);
                for (; j + 15 < cols; j += 16) {
                    int8x16_t vw = vld1q_s8(row + j);
                    int8x16_t vi = vld1q_s8(q + j);
                    int16x8_t prod0 = vmull_s8(vget_low_s8(vi), vget_low_s8(vw));
                    int16x8_t prod1 = vmull_s8(vget_high_s8(vi), vget_high_s8(vw));
                    vacc0 = vpadalq_s16(vacc0, prod0);
                    vacc1 = vpadalq_s16(vacc1, prod1);
                }
                acc = microgemm_neon_hsum_s32(vaddq_s32(vacc0, vacc1));
            }
#endif

            for (; j < cols; ++j) {
#if MICROGEMM_CPU_X86_AVX2
                if (row_sums != NULL) {
                    acc += ((int32_t)q[j] + 128) * (int32_t)row[j];
                } else {
                    acc += (int32_t)q[j] * (int32_t)row[j];
                }
#else
                acc += (int32_t)q[j] * (int32_t)row[j];
#endif
            }

            out[(size_t)b * rows + i] =
                (float)acc * row_scale * input_scales[b] + (bias ? bias[i] : 0.0f);
        }
    }
    if (profile_down_proj) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
    }
}

static void microgemm_gemv_i8_batched(
    float* out,
    const int8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int batch,
    int rows,
    int cols,
    int input_stride,
    const float* bias,
    int8_t* input_q,
    float* input_scales
) {
    microgemm_gemv_i8_batched_impl(
        out, weights, scales, row_sums, input, batch, rows, cols,
        input_stride, bias, input_q, input_scales, 0
    );
}

static void microgemm_gemv_i8_batched_prequantized(
    float* out,
    const int8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    int batch,
    int rows,
    int cols,
    const float* bias,
    int8_t* input_q,
    float* input_scales
) {
    microgemm_gemv_i8_batched_impl(
        out, weights, scales, row_sums, NULL, batch, rows, cols,
        cols, bias, input_q, input_scales, 1
    );
}

static void microgemm_gemv_i4_packed_batched(
    float* out,
    const uint8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int batch,
    int rows,
    int cols,
    int input_stride,
    const float* bias,
    int8_t* input_q,
    float* input_scales
) {
    size_t row_bytes = (size_t)((cols + 1) / 2);
    int b;
    int i;
    int profile_down_proj = microgemm_decode_batch_profile_enabled && microgemm_profile_next_down_proj_gemv;
    double profile_phase_start = 0.0;

    if (batch <= 0) {
        return;
    }

    if (profile_down_proj) {
        profile_phase_start = microgemm_profile_now_ms();
    }

    #pragma omp parallel for schedule(static) if(batch >= 8)
    for (b = 0; b < batch; ++b) {
#if MICROGEMM_CPU_X86_AVX2
        if (row_sums != NULL) {
            input_scales[b] = microgemm_quantize_activation_for_prebiased_maddubs_dot(
                input_q + (size_t)b * cols,
                input + (size_t)b * input_stride,
                cols
            );
        } else
#endif
        input_scales[b] = microgemm_quantize_activation_for_i8_self_biasing_dot(
            input_q + (size_t)b * cols,
            input + (size_t)b * input_stride,
            cols
        );
    }

    if (profile_down_proj) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_quant_ms, profile_phase_start);
        profile_phase_start = microgemm_profile_now_ms();
    }

#if MICROGEMM_CPU_X86_AVX2
    if (row_sums != NULL) {
        int use_tile8_split = microgemm_i4_row_tile8_split_enabled();
        int use_row_pair_tile4 = microgemm_i4_row_pair_tile4_enabled_for(batch, rows, cols);
        if (use_row_pair_tile4) {
            int pair_count = rows / 2;
            #pragma omp parallel for schedule(static) if(rows >= 96)
            for (i = 0; i < pair_count; ++i) {
                int row_idx = i * 2;
                const uint8_t* row0 = weights + (size_t)row_idx * row_bytes;
                const uint8_t* row1 = row0 + row_bytes;
                int b0 = 0;

                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i4_batch_row_pair_tile4(
                        out,
                        row_idx,
                        rows,
                        row0,
                        row1,
                        scales[row_idx],
                        scales[row_idx + 1],
                        row_sums[row_idx],
                        row_sums[row_idx + 1],
                        input_q,
                        input_scales,
                        b0,
                        cols,
                        bias
                    );
                }
                for (; b0 + 1 < batch; b0 += 2) {
                    microgemm_avx2_i4_batch_row_tile(
                        out,
                        row_idx,
                        rows,
                        row0,
                        scales[row_idx],
                        row_sums[row_idx],
                        input_q,
                        input_scales,
                        b0,
                        2,
                        cols,
                        bias
                    );
                    microgemm_avx2_i4_batch_row_tile(
                        out,
                        row_idx + 1,
                        rows,
                        row1,
                        scales[row_idx + 1],
                        row_sums[row_idx + 1],
                        input_q,
                        input_scales,
                        b0,
                        2,
                        cols,
                        bias
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i4_batch_row_tile(
                        out,
                        row_idx,
                        rows,
                        row0,
                        scales[row_idx],
                        row_sums[row_idx],
                        input_q,
                        input_scales,
                        b0,
                        1,
                        cols,
                        bias
                    );
                    microgemm_avx2_i4_batch_row_tile(
                        out,
                        row_idx + 1,
                        rows,
                        row1,
                        scales[row_idx + 1],
                        row_sums[row_idx + 1],
                        input_q,
                        input_scales,
                        b0,
                        1,
                        cols,
                        bias
                    );
                }
            }
            if ((rows & 1) != 0) {
                int row_idx = rows - 1;
                const uint8_t* row = weights + (size_t)row_idx * row_bytes;
                int b0 = 0;
                for (; b0 + 7 < batch; b0 += 8) {
                    if (use_tile8_split) {
                        microgemm_avx2_i4_batch_row_tile8_split(
                            out, row_idx, rows, row, scales[row_idx], row_sums[row_idx],
                            input_q, input_scales, b0, cols, bias
                        );
                    } else {
                        microgemm_avx2_i4_batch_row_tile(
                            out, row_idx, rows, row, scales[row_idx], row_sums[row_idx],
                            input_q, input_scales, b0, 8, cols, bias
                        );
                    }
                }
                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i4_batch_row_tile(
                        out, row_idx, rows, row, scales[row_idx], row_sums[row_idx],
                        input_q, input_scales, b0, 4, cols, bias
                    );
                }
                for (; b0 + 1 < batch; b0 += 2) {
                    microgemm_avx2_i4_batch_row_tile(
                        out, row_idx, rows, row, scales[row_idx], row_sums[row_idx],
                        input_q, input_scales, b0, 2, cols, bias
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i4_batch_row_tile(
                        out, row_idx, rows, row, scales[row_idx], row_sums[row_idx],
                        input_q, input_scales, b0, 1, cols, bias
                    );
                }
            }
            if (profile_down_proj) {
                microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
            }
            return;
        }
        #pragma omp parallel for schedule(static) if(rows >= 96)
        for (i = 0; i < rows; ++i) {
            const uint8_t* row = weights + (size_t)i * row_bytes;
            float row_scale = scales[i];
            int row_sum = row_sums[i];
            int b0 = 0;

            for (; b0 + 7 < batch; b0 += 8) {
                if (use_tile8_split) {
                    microgemm_avx2_i4_batch_row_tile8_split(
                        out, i, rows, row, row_scale, row_sum,
                        input_q, input_scales, b0, cols, bias
                    );
                } else {
                    microgemm_avx2_i4_batch_row_tile(
                        out, i, rows, row, row_scale, row_sum,
                        input_q, input_scales, b0, 8, cols, bias
                    );
                }
            }
            for (; b0 + 3 < batch; b0 += 4) {
                microgemm_avx2_i4_batch_row_tile(
                    out, i, rows, row, row_scale, row_sum,
                    input_q, input_scales, b0, 4, cols, bias
                );
            }
            for (; b0 + 1 < batch; b0 += 2) {
                microgemm_avx2_i4_batch_row_tile(
                    out, i, rows, row, row_scale, row_sum,
                    input_q, input_scales, b0, 2, cols, bias
                );
            }
            if (b0 < batch) {
                microgemm_avx2_i4_batch_row_tile(
                    out, i, rows, row, row_scale, row_sum,
                    input_q, input_scales, b0, 1, cols, bias
                );
            }
        }
        if (profile_down_proj) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
        }
        return;
    }
#endif

    #pragma omp parallel for schedule(static) if(rows >= 96)
    for (i = 0; i < rows; ++i) {
        const uint8_t* row = weights + (size_t)i * row_bytes;
        float row_scale = scales[i];
        int row_sum = row_sums ? row_sums[i] : 0;
        if (row_sums == NULL) {
            int j;
            for (j = 0; j < cols; ++j) {
                row_sum += (int)microgemm_i4_row_value(row, j);
            }
        }

        for (b = 0; b < batch; ++b) {
            const int8_t* q = input_q + (size_t)b * cols;
            int32_t acc = microgemm_dot_i4_biased_i8_scalar(row, q, cols, row_sum);
            out[(size_t)b * rows + i] =
                (float)acc * row_scale * input_scales[b] + (bias ? bias[i] : 0.0f);
        }
    }
    if (profile_down_proj) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
    }
}

static void microgemm_gemv_i8_groupwise_batched_impl(
    float* out,
    const int8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int batch,
    int rows,
    int cols,
    int input_stride,
    const float* bias,
    int8_t* input_q,
    float* input_scales,
    int input_prequantized
) {
    const int groups = microgemm_quant_group_count_int(cols);
    int profile_down_proj = microgemm_decode_batch_profile_enabled && microgemm_profile_next_down_proj_gemv;
    double profile_phase_start = 0.0;
    int b;
    int i;

    if (batch <= 0) {
        return;
    }
    if (profile_down_proj) {
        profile_phase_start = microgemm_profile_now_ms();
    }
    if (!input_prequantized) {
        #pragma omp parallel for schedule(static) if(batch >= 8)
        for (b = 0; b < batch; ++b) {
            input_scales[b] = microgemm_quantize_activation_for_i8_self_biasing_dot(
                input_q + (size_t)b * cols,
                input + (size_t)b * input_stride,
                cols
            );
        }
    }
    if (profile_down_proj && !input_prequantized) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_quant_ms, profile_phase_start);
        profile_phase_start = microgemm_profile_now_ms();
    }

    if (microgemm_i8g_saturation_safe_enabled_for(batch, rows, cols)) {
#if MICROGEMM_CPU_X86_AVX2
        int use_i8g_sat_safe_row_pair_tile4 =
            microgemm_i8g_sat_safe_row_pair_tile4_enabled_for(batch, rows, cols);
        if (microgemm_decode_batch_profile_enabled) {
            microgemm_decode_batch_profile_state.groupwise_gemv_tile_calls += 1u;
            if (use_i8g_sat_safe_row_pair_tile4) {
                microgemm_decode_batch_profile_state.groupwise_i8_row_pair_calls += 1u;
            }
        }
        if (use_i8g_sat_safe_row_pair_tile4) {
            #pragma omp parallel for schedule(static) if(rows >= 96)
            for (i = 0; i < rows - 1; i += 2) {
                const int8_t* row0 = weights + (size_t)i * cols;
                const int8_t* row1 = weights + (size_t)(i + 1) * cols;
                const float* row0_scales = scales + (size_t)i * groups;
                const float* row1_scales = scales + (size_t)(i + 1) * groups;
                int b0 = 0;

                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i8_groupwise_batch_row_pair_tile4_safe(
                        out,
                        i,
                        rows,
                        row0,
                        row1,
                        row0_scales,
                        row1_scales,
                        input_q,
                        input_scales,
                        b0,
                        cols,
                        groups,
                        bias
                    );
                }
                if (b0 < batch) {
                    int tile = batch - b0;
                    microgemm_avx2_i8_groupwise_batch_row_tile_safe(
                        out,
                        i,
                        rows,
                        row0,
                        row0_scales,
                        input_q,
                        input_scales,
                        b0,
                        tile,
                        cols,
                        groups,
                        bias
                    );
                    microgemm_avx2_i8_groupwise_batch_row_tile_safe(
                        out,
                        i + 1,
                        rows,
                        row1,
                        row1_scales,
                        input_q,
                        input_scales,
                        b0,
                        tile,
                        cols,
                        groups,
                        bias
                    );
                }
            }
            if ((rows & 1) != 0) {
                const int8_t* row = weights + (size_t)(rows - 1) * cols;
                const float* row_scales = scales + (size_t)(rows - 1) * groups;
                int b0 = 0;
                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i8_groupwise_batch_row_tile4_safe(
                        out,
                        rows - 1,
                        rows,
                        row,
                        row_scales,
                        input_q,
                        input_scales,
                        b0,
                        cols,
                        groups,
                        bias
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i8_groupwise_batch_row_tile_safe(
                        out,
                        rows - 1,
                        rows,
                        row,
                        row_scales,
                        input_q,
                        input_scales,
                        b0,
                        batch - b0,
                        cols,
                        groups,
                        bias
                    );
                }
            }
        } else {
            #pragma omp parallel for schedule(static) if(rows >= 96)
            for (i = 0; i < rows; ++i) {
                const int8_t* row = weights + (size_t)i * cols;
                const float* row_scales = scales + (size_t)i * groups;
                int b0 = 0;

                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i8_groupwise_batch_row_tile4_safe(
                        out,
                        i,
                        rows,
                        row,
                        row_scales,
                        input_q,
                        input_scales,
                        b0,
                        cols,
                        groups,
                        bias
                    );
                }
                for (; b0 + 1 < batch; b0 += 2) {
                    microgemm_avx2_i8_groupwise_batch_row_tile_safe(
                        out,
                        i,
                        rows,
                        row,
                        row_scales,
                        input_q,
                        input_scales,
                        b0,
                        2,
                        cols,
                        groups,
                        bias
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i8_groupwise_batch_row_tile_safe(
                        out,
                        i,
                        rows,
                        row,
                        row_scales,
                        input_q,
                        input_scales,
                        b0,
                        1,
                        cols,
                        groups,
                        bias
                    );
                }
            }
        }
        if (profile_down_proj) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
        }
        return;
#else
        #pragma omp parallel for schedule(static) if(rows >= 96)
        for (i = 0; i < rows; ++i) {
            const int8_t* row = weights + (size_t)i * cols;
            const float* row_scales = scales + (size_t)i * groups;
            for (b = 0; b < batch; ++b) {
                const int8_t* q = input_q + (size_t)b * cols;
                float value = bias ? bias[i] : 0.0f;
                int group;
                for (group = 0; group < groups; ++group) {
                    int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
                    int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
                    int32_t acc;
                    if (end > cols) {
                        end = cols;
                    }
                    acc = microgemm_dot_i8_signed_i8_safe(row + begin, q + begin, end - begin);
                    value += (float)acc * row_scales[group] * input_scales[b];
                }
                out[(size_t)b * rows + i] = value;
            }
        }
        if (profile_down_proj) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
        }
        return;
#endif
    }

#if MICROGEMM_CPU_X86_AVX2
    if (batch >= 4 && row_sums != NULL) {
        int use_i8g_row_pair_tile4 = microgemm_i8g_row_pair_tile4_enabled_for(batch, rows, cols);
        if (use_i8g_row_pair_tile4) {
            if (microgemm_decode_batch_profile_enabled) {
                microgemm_decode_batch_profile_state.groupwise_gemv_tile_calls += 1u;
                microgemm_decode_batch_profile_state.groupwise_i8_row_pair_calls += 1u;
            }
            #pragma omp parallel for schedule(static) if(rows >= 96)
            for (i = 0; i < rows - 1; i += 2) {
                const int8_t* row0 = weights + (size_t)i * cols;
                const int8_t* row1 = weights + (size_t)(i + 1) * cols;
                const float* row0_scales = scales + (size_t)i * groups;
                const float* row1_scales = scales + (size_t)(i + 1) * groups;
                const int32_t* row0_group_sums = row_sums + (size_t)i * groups;
                const int32_t* row1_group_sums = row_sums + (size_t)(i + 1) * groups;
                int b0 = 0;

                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i8_groupwise_batch_row_pair_tile4(
                        out,
                        i,
                        rows,
                        row0,
                        row1,
                        row0_scales,
                        row1_scales,
                        row0_group_sums,
                        row1_group_sums,
                        input_q,
                        input_scales,
                        b0,
                        cols,
                        groups,
                        bias
                    );
                }
                if (b0 < batch) {
                    int tile = batch - b0;
                    microgemm_avx2_i8_groupwise_batch_row_tile(
                        out,
                        i,
                        rows,
                        row0,
                        row0_scales,
                        row0_group_sums,
                        input_q,
                        input_scales,
                        b0,
                        tile,
                        cols,
                        groups,
                        bias
                    );
                    microgemm_avx2_i8_groupwise_batch_row_tile(
                        out,
                        i + 1,
                        rows,
                        row1,
                        row1_scales,
                        row1_group_sums,
                        input_q,
                        input_scales,
                        b0,
                        tile,
                        cols,
                        groups,
                        bias
                    );
                }
            }
            if ((rows & 1) != 0) {
                const int8_t* row = weights + (size_t)(rows - 1) * cols;
                const float* row_scales = scales + (size_t)(rows - 1) * groups;
                const int32_t* row_group_sums = row_sums + (size_t)(rows - 1) * groups;
                int b0 = 0;

                for (; b0 + 7 < batch; b0 += 8) {
                    microgemm_avx2_i8_groupwise_batch_row_tile(
                        out, rows - 1, rows, row, row_scales, row_group_sums,
                        input_q, input_scales, b0, 8, cols, groups, bias
                    );
                }
                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i8_groupwise_batch_row_tile(
                        out, rows - 1, rows, row, row_scales, row_group_sums,
                        input_q, input_scales, b0, 4, cols, groups, bias
                    );
                }
                for (; b0 + 1 < batch; b0 += 2) {
                    microgemm_avx2_i8_groupwise_batch_row_tile(
                        out, rows - 1, rows, row, row_scales, row_group_sums,
                        input_q, input_scales, b0, 2, cols, groups, bias
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i8_groupwise_batch_row_tile(
                        out, rows - 1, rows, row, row_scales, row_group_sums,
                        input_q, input_scales, b0, 1, cols, groups, bias
                    );
                }
            }
            if (profile_down_proj) {
                microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
            }
            return;
        }
        if (microgemm_decode_batch_profile_enabled) {
            microgemm_decode_batch_profile_state.groupwise_gemv_tile_calls += 1u;
        }
        #pragma omp parallel for schedule(static) if(rows >= 96)
        for (i = 0; i < rows; ++i) {
            const int8_t* row = weights + (size_t)i * cols;
            const float* row_scales = scales + (size_t)i * groups;
            const int32_t* row_group_sums = row_sums + (size_t)i * groups;
            int b0 = 0;

            for (; b0 + 7 < batch; b0 += 8) {
                microgemm_avx2_i8_groupwise_batch_row_tile(
                    out,
                    i,
                    rows,
                    row,
                    row_scales,
                    row_group_sums,
                    input_q,
                    input_scales,
                    b0,
                    8,
                    cols,
                    groups,
                    bias
                );
            }
            for (; b0 + 3 < batch; b0 += 4) {
                microgemm_avx2_i8_groupwise_batch_row_tile(
                    out,
                    i,
                    rows,
                    row,
                    row_scales,
                    row_group_sums,
                    input_q,
                    input_scales,
                    b0,
                    4,
                    cols,
                    groups,
                    bias
                );
            }
            for (; b0 + 1 < batch; b0 += 2) {
                microgemm_avx2_i8_groupwise_batch_row_tile(
                    out,
                    i,
                    rows,
                    row,
                    row_scales,
                    row_group_sums,
                    input_q,
                    input_scales,
                    b0,
                    2,
                    cols,
                    groups,
                    bias
                );
            }
            if (b0 < batch) {
                microgemm_avx2_i8_groupwise_batch_row_tile(
                    out,
                    i,
                    rows,
                    row,
                    row_scales,
                    row_group_sums,
                    input_q,
                    input_scales,
                    b0,
                    1,
                    cols,
                    groups,
                    bias
                );
            }
        }
        if (profile_down_proj) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
        }
        return;
    }
#endif

    #pragma omp parallel for schedule(static) if(rows >= 96)
    for (i = 0; i < rows; ++i) {
        const int8_t* row = weights + (size_t)i * cols;
        const float* row_scales = scales + (size_t)i * groups;
        const int32_t* row_group_sums = row_sums ? row_sums + (size_t)i * groups : NULL;

        for (b = 0; b < batch; ++b) {
            const int8_t* q = input_q + (size_t)b * cols;
            float value = bias ? bias[i] : 0.0f;
            int group;

            for (group = 0; group < groups; ++group) {
                int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
                int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
                int group_cols;
                int group_sum = 0;
                int32_t acc;

                if (end > cols) {
                    end = cols;
                }
                group_cols = end - begin;
                if (row_group_sums != NULL) {
                    group_sum = row_group_sums[group];
                } else {
                    int j;
                    for (j = begin; j < end; ++j) {
                        group_sum += (int)row[j];
                    }
                }
                acc = microgemm_dot_i8_biased_i8(row + begin, q + begin, group_cols, group_sum);
                value += (float)acc * row_scales[group] * input_scales[b];
            }
            out[(size_t)b * rows + i] = value;
        }
    }
    if (profile_down_proj) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
    }
}

static void microgemm_gemv_i8_groupwise_batched(
    float* out,
    const int8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int batch,
    int rows,
    int cols,
    int input_stride,
    const float* bias,
    int8_t* input_q,
    float* input_scales
) {
    microgemm_gemv_i8_groupwise_batched_impl(
        out, weights, scales, row_sums, input, batch, rows, cols,
        input_stride, bias, input_q, input_scales, 0
    );
}

static void microgemm_gemv_i8_groupwise_batched_prequantized(
    float* out,
    const int8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    int batch,
    int rows,
    int cols,
    const float* bias,
    int8_t* input_q,
    float* input_scales
) {
    microgemm_gemv_i8_groupwise_batched_impl(
        out, weights, scales, row_sums, NULL, batch, rows, cols,
        cols, bias, input_q, input_scales, 1
    );
}

static void microgemm_gemv_i4_groupwise_batched(
    float* out,
    const uint8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int batch,
    int rows,
    int cols,
    int input_stride,
    const float* bias,
    int8_t* input_q,
    float* input_scales
) {
    const int groups = microgemm_quant_group_count_int(cols);
    const size_t row_bytes = (size_t)((cols + 1) / 2);
    int profile_down_proj = microgemm_decode_batch_profile_enabled && microgemm_profile_next_down_proj_gemv;
    double profile_phase_start = 0.0;
    int b;
    int i;

    if (batch <= 0) {
        return;
    }
    if (profile_down_proj) {
        profile_phase_start = microgemm_profile_now_ms();
    }
    #pragma omp parallel for schedule(static) if(batch >= 8)
    for (b = 0; b < batch; ++b) {
#if MICROGEMM_CPU_X86_AVX2
        input_scales[b] = microgemm_quantize_activation_for_prebiased_maddubs_dot(
            input_q + (size_t)b * cols,
            input + (size_t)b * input_stride,
            cols
        );
#else
        input_scales[b] = microgemm_quantize_activation_for_i8_self_biasing_dot(
            input_q + (size_t)b * cols,
            input + (size_t)b * input_stride,
            cols
        );
#endif
    }
    if (profile_down_proj) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_quant_ms, profile_phase_start);
        profile_phase_start = microgemm_profile_now_ms();
    }

#if MICROGEMM_CPU_X86_AVX2
    if (batch >= 4 && row_sums != NULL) {
        int use_i4g_row_pair_tile4 = microgemm_i4g_row_pair_tile4_enabled_for(batch, rows, cols);
        if (use_i4g_row_pair_tile4) {
            if (microgemm_decode_batch_profile_enabled) {
                microgemm_decode_batch_profile_state.groupwise_gemv_tile_calls += 1u;
                microgemm_decode_batch_profile_state.groupwise_i4_row_pair_calls += 1u;
            }
            #pragma omp parallel for schedule(static) if(rows >= 96)
            for (i = 0; i < rows - 1; i += 2) {
                const uint8_t* row0 = weights + (size_t)i * row_bytes;
                const uint8_t* row1 = weights + (size_t)(i + 1) * row_bytes;
                const float* row0_scales = scales + (size_t)i * groups;
                const float* row1_scales = scales + (size_t)(i + 1) * groups;
                const int32_t* row0_group_sums = row_sums + (size_t)i * groups;
                const int32_t* row1_group_sums = row_sums + (size_t)(i + 1) * groups;
                int b0 = 0;

                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i4_groupwise_batch_row_pair_tile4(
                        out,
                        i,
                        rows,
                        row0,
                        row1,
                        row0_scales,
                        row1_scales,
                        row0_group_sums,
                        row1_group_sums,
                        input_q,
                        input_scales,
                        b0,
                        cols,
                        groups,
                        bias
                    );
                }
                if (b0 < batch) {
                    int tile = batch - b0;
                    microgemm_avx2_i4_groupwise_batch_row_tile(
                        out,
                        i,
                        rows,
                        row0,
                        row0_scales,
                        row0_group_sums,
                        input_q,
                        input_scales,
                        b0,
                        tile,
                        cols,
                        groups,
                        bias
                    );
                    microgemm_avx2_i4_groupwise_batch_row_tile(
                        out,
                        i + 1,
                        rows,
                        row1,
                        row1_scales,
                        row1_group_sums,
                        input_q,
                        input_scales,
                        b0,
                        tile,
                        cols,
                        groups,
                        bias
                    );
                }
            }
            if ((rows & 1) != 0) {
                const uint8_t* row = weights + (size_t)(rows - 1) * row_bytes;
                const float* row_scales = scales + (size_t)(rows - 1) * groups;
                const int32_t* row_group_sums = row_sums + (size_t)(rows - 1) * groups;
                int b0 = 0;

                for (; b0 + 7 < batch; b0 += 8) {
                    microgemm_avx2_i4_groupwise_batch_row_tile(
                        out, rows - 1, rows, row, row_scales, row_group_sums,
                        input_q, input_scales, b0, 8, cols, groups, bias
                    );
                }
                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i4_groupwise_batch_row_tile(
                        out, rows - 1, rows, row, row_scales, row_group_sums,
                        input_q, input_scales, b0, 4, cols, groups, bias
                    );
                }
                for (; b0 + 1 < batch; b0 += 2) {
                    microgemm_avx2_i4_groupwise_batch_row_tile(
                        out, rows - 1, rows, row, row_scales, row_group_sums,
                        input_q, input_scales, b0, 2, cols, groups, bias
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i4_groupwise_batch_row_tile(
                        out, rows - 1, rows, row, row_scales, row_group_sums,
                        input_q, input_scales, b0, 1, cols, groups, bias
                    );
                }
            }
            if (profile_down_proj) {
                microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
            }
            return;
        }
        if (microgemm_decode_batch_profile_enabled) {
            microgemm_decode_batch_profile_state.groupwise_gemv_tile_calls += 1u;
        }
        #pragma omp parallel for schedule(static) if(rows >= 96)
        for (i = 0; i < rows; ++i) {
            const uint8_t* row = weights + (size_t)i * row_bytes;
            const float* row_scales = scales + (size_t)i * groups;
            const int32_t* row_group_sums = row_sums + (size_t)i * groups;
            int b0 = 0;

            for (; b0 + 7 < batch; b0 += 8) {
                microgemm_avx2_i4_groupwise_batch_row_tile(
                    out,
                    i,
                    rows,
                    row,
                    row_scales,
                    row_group_sums,
                    input_q,
                    input_scales,
                    b0,
                    8,
                    cols,
                    groups,
                    bias
                );
            }
            for (; b0 + 3 < batch; b0 += 4) {
                microgemm_avx2_i4_groupwise_batch_row_tile(
                    out,
                    i,
                    rows,
                    row,
                    row_scales,
                    row_group_sums,
                    input_q,
                    input_scales,
                    b0,
                    4,
                    cols,
                    groups,
                    bias
                );
            }
            for (; b0 + 1 < batch; b0 += 2) {
                microgemm_avx2_i4_groupwise_batch_row_tile(
                    out,
                    i,
                    rows,
                    row,
                    row_scales,
                    row_group_sums,
                    input_q,
                    input_scales,
                    b0,
                    2,
                    cols,
                    groups,
                    bias
                );
            }
            if (b0 < batch) {
                microgemm_avx2_i4_groupwise_batch_row_tile(
                    out,
                    i,
                    rows,
                    row,
                    row_scales,
                    row_group_sums,
                    input_q,
                    input_scales,
                    b0,
                    1,
                    cols,
                    groups,
                    bias
                );
            }
        }
        if (profile_down_proj) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
        }
        return;
    }
#endif

    #pragma omp parallel for schedule(static) if(rows >= 96)
    for (i = 0; i < rows; ++i) {
        const uint8_t* row = weights + (size_t)i * row_bytes;
        const float* row_scales = scales + (size_t)i * groups;
        const int32_t* row_group_sums = row_sums ? row_sums + (size_t)i * groups : NULL;

        for (b = 0; b < batch; ++b) {
            const int8_t* q = input_q + (size_t)b * cols;
            float value = bias ? bias[i] : 0.0f;
            int group;

            for (group = 0; group < groups; ++group) {
                int begin = group * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
                int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
                int group_cols;
                int group_sum = 0;
                int32_t acc;

                if (end > cols) {
                    end = cols;
                }
                group_cols = end - begin;
                if (row_group_sums != NULL) {
                    group_sum = row_group_sums[group];
                } else {
                    int j;
                    for (j = begin; j < end; ++j) {
                        group_sum += (int)microgemm_i4_row_value(row, j);
                    }
                }
#if MICROGEMM_CPU_X86_AVX2
                acc = microgemm_avx2_dot_i4_biased_i8(
                    row + (begin >> 1), q + begin, group_cols, group_sum
                );
#else
                acc = microgemm_dot_i4_biased_i8_scalar(
                    row + (begin >> 1), q + begin, group_cols, group_sum
                );
#endif
                value += (float)acc * row_scales[group] * input_scales[b];
            }
            out[(size_t)b * rows + i] = value;
        }
    }
    if (profile_down_proj) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_dot_ms, profile_phase_start);
    }
}

static void microgemm_gemv_quantized_batched(
    const microgemm_config* config,
    float* out,
    const int8_t* weights_i8,
    const uint8_t* weights_i4,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int batch,
    int rows,
    int cols,
    int input_stride,
    const float* bias,
    int8_t* input_q,
    float* input_scales
) {
    if (config != NULL && config->quant_mode == MICROGEMM_QUANT_INT8G128) {
        microgemm_gemv_i8_groupwise_batched(
            out, weights_i8, scales, row_sums, input, batch, rows, cols,
            input_stride, bias, input_q, input_scales
        );
        return;
    }
    if (config != NULL && config->quant_mode == MICROGEMM_QUANT_INT4G128) {
        microgemm_gemv_i4_groupwise_batched(
            out, weights_i4, scales, row_sums, input, batch, rows, cols,
            input_stride, bias, input_q, input_scales
        );
        return;
    }
    if (config != NULL && config->quant_mode == MICROGEMM_QUANT_INT4) {
        microgemm_gemv_i4_packed_batched(
            out, weights_i4, scales, row_sums, input, batch, rows, cols,
            input_stride, bias, input_q, input_scales
        );
        return;
    }
    microgemm_gemv_i8_batched(
        out, weights_i8, scales, row_sums, input, batch, rows, cols,
        input_stride, bias, input_q, input_scales
    );
}

static void microgemm_gemv_quantized_batched_prequantized(
    const microgemm_config* config,
    float* out,
    const int8_t* weights_i8,
    const uint8_t* weights_i4,
    const float* scales,
    const int32_t* row_sums,
    int batch,
    int rows,
    int cols,
    const float* bias,
    int8_t* input_q,
    float* input_scales
) {
    /* input_q/input_scales are signed int8 activations produced for self-biasing i8 dots. */
    if (config != NULL && config->quant_mode == MICROGEMM_QUANT_INT8G128) {
        microgemm_gemv_i8_groupwise_batched_prequantized(
            out, weights_i8, scales, row_sums, batch, rows, cols,
            bias, input_q, input_scales
        );
        return;
    }
    if (config != NULL && config->quant_mode == MICROGEMM_QUANT_INT8) {
        microgemm_gemv_i8_batched_prequantized(
            out, weights_i8, scales, row_sums, batch, rows, cols,
            bias, input_q, input_scales
        );
        return;
    }
    (void)weights_i4;
    (void)bias;
    (void)input_q;
    (void)input_scales;
    abort();
}

static inline void microgemm_argmax_update_score(
    float* best_values,
    int* best_ids,
    int batch_index,
    int token_id,
    float score
) {
    if (score > best_values[batch_index]
            || (score == best_values[batch_index] && token_id < best_ids[batch_index])) {
        best_values[batch_index] = score;
        best_ids[batch_index] = token_id;
    }
}

static microgemm_status microgemm_lm_head_argmax_i8_batched(
    int* out_token_ids,
    const int8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int batch,
    int rows,
    int cols,
    int input_stride,
    int8_t* input_q,
    float* input_scales
) {
    int b;

    if (out_token_ids == NULL || weights == NULL || scales == NULL
            || input == NULL || input_q == NULL || input_scales == NULL
            || batch <= 0 || rows <= 0 || cols <= 0) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

#if MICROGEMM_CPU_X86_AVX2
    if (row_sums != NULL) {
        int i;
        int thread_count = 1;
        int use_stack_best = 0;
        size_t best_count;
        float best_values_stack[MICROGEMM_LM_HEAD_STACK_BEST_LIMIT];
        int best_ids_stack[MICROGEMM_LM_HEAD_STACK_BEST_LIMIT];
        float* best_values;
        int* best_ids;

        #pragma omp parallel for schedule(static) if(batch >= 8)
        for (b = 0; b < batch; ++b) {
            input_scales[b] = microgemm_quantize_activation_for_i8_self_biasing_dot(
                input_q + (size_t)b * cols,
                input + (size_t)b * input_stride,
                cols
            );
        }

#ifdef _OPENMP
        thread_count = omp_get_max_threads();
        if (thread_count < 1) {
            thread_count = 1;
        }
#endif
        if ((size_t)thread_count > ((size_t)-1) / (size_t)batch) {
            return MICROGEMM_STATUS_OUT_OF_MEMORY;
        }
        best_count = (size_t)thread_count * (size_t)batch;
        use_stack_best = microgemm_lm_head_stack_best_enabled_for(batch, thread_count);
        if (use_stack_best && microgemm_decode_batch_profile_enabled) {
            microgemm_decode_batch_profile_state.lm_head_stack_best_calls += 1u;
        }
        if (use_stack_best) {
            best_values = best_values_stack;
            best_ids = best_ids_stack;
        } else {
            best_values = (float*)malloc(best_count * sizeof(float));
            best_ids = (int*)malloc(best_count * sizeof(int));
            if (best_values == NULL || best_ids == NULL) {
                free(best_values);
                free(best_ids);
                return MICROGEMM_STATUS_OUT_OF_MEMORY;
            }
        }
        {
            size_t idx;
            for (idx = 0u; idx < best_count; ++idx) {
                best_values[idx] = -FLT_MAX;
                best_ids[idx] = 0;
            }
        }

        {
            int pair_count = rows / 2;
            #pragma omp parallel for schedule(static) if(rows >= 96)
            for (i = 0; i < pair_count; ++i) {
                int row_idx = i * 2;
                const int8_t* row0 = weights + (size_t)row_idx * cols;
                const int8_t* row1 = row0 + cols;
                float row0_scale = scales[row_idx];
                float row1_scale = scales[row_idx + 1];
                int row0_sum = row_sums[row_idx];
                int row1_sum = row_sums[row_idx + 1];
                int thread_index = 0;
                float* thread_best;
                int* thread_ids;
                int b0 = 0;

#ifdef _OPENMP
                thread_index = omp_get_thread_num();
#endif
                thread_best = best_values + (size_t)thread_index * (size_t)batch;
                thread_ids = best_ids + (size_t)thread_index * (size_t)batch;

                for (; b0 + 7 < batch; b0 += 8) {
                    float scores[16];
                    int bb;
                    microgemm_avx2_i8_batch_row_pair_tile8_split(
                        scores,
                        0,
                        2,
                        row0,
                        row1,
                        row0_scale,
                        row1_scale,
                        row0_sum,
                        row1_sum,
                        input_q + (size_t)b0 * cols,
                        input_scales + b0,
                        0,
                        cols,
                        NULL
                    );
                    for (bb = 0; bb < 8; ++bb) {
                        int batch_index = b0 + bb;
                        microgemm_argmax_update_score(
                            thread_best, thread_ids, batch_index, row_idx, scores[bb * 2]
                        );
                        microgemm_argmax_update_score(
                            thread_best, thread_ids, batch_index, row_idx + 1, scores[bb * 2 + 1]
                        );
                    }
                }
                for (; b0 + 3 < batch; b0 += 4) {
                    float scores[8];
                    int bb;
                    microgemm_avx2_i8_batch_row_pair_tile4_split(
                        scores,
                        0,
                        2,
                        row0,
                        row1,
                        row0_scale,
                        row1_scale,
                        row0_sum,
                        row1_sum,
                        input_q + (size_t)b0 * cols,
                        input_scales + b0,
                        0,
                        cols,
                        NULL
                    );
                    for (bb = 0; bb < 4; ++bb) {
                        int batch_index = b0 + bb;
                        microgemm_argmax_update_score(
                            thread_best, thread_ids, batch_index, row_idx, scores[bb * 2]
                        );
                        microgemm_argmax_update_score(
                            thread_best, thread_ids, batch_index, row_idx + 1, scores[bb * 2 + 1]
                        );
                    }
                }
                for (; b0 + 1 < batch; b0 += 2) {
                    float scores[4];
                    int bb;
                    microgemm_avx2_i8_batch_row_pair_tile(
                        scores,
                        0,
                        2,
                        row0,
                        row1,
                        row0_scale,
                        row1_scale,
                        row0_sum,
                        row1_sum,
                        input_q + (size_t)b0 * cols,
                        input_scales + b0,
                        0,
                        2,
                        cols,
                        NULL
                    );
                    for (bb = 0; bb < 2; ++bb) {
                        int batch_index = b0 + bb;
                        microgemm_argmax_update_score(
                            thread_best, thread_ids, batch_index, row_idx, scores[bb * 2]
                        );
                        microgemm_argmax_update_score(
                            thread_best, thread_ids, batch_index, row_idx + 1, scores[bb * 2 + 1]
                        );
                    }
                }
                if (b0 < batch) {
                    float scores[2];
                    microgemm_avx2_i8_batch_row_pair_tile(
                        scores,
                        0,
                        2,
                        row0,
                        row1,
                        row0_scale,
                        row1_scale,
                        row0_sum,
                        row1_sum,
                        input_q + (size_t)b0 * cols,
                        input_scales + b0,
                        0,
                        1,
                        cols,
                        NULL
                    );
                    microgemm_argmax_update_score(thread_best, thread_ids, b0, row_idx, scores[0]);
                    microgemm_argmax_update_score(thread_best, thread_ids, b0, row_idx + 1, scores[1]);
                }
            }
        }

        if ((rows & 1) != 0) {
            int row_idx = rows - 1;
            const int8_t* row = weights + (size_t)row_idx * cols;
            float row_scale = scales[row_idx];
            int row_sum = row_sums[row_idx];
            int thread_index = 0;
            float* thread_best;
            int* thread_ids;
            int b0 = 0;

#ifdef _OPENMP
            thread_index = omp_get_thread_num();
#endif
            thread_best = best_values + (size_t)thread_index * (size_t)batch;
            thread_ids = best_ids + (size_t)thread_index * (size_t)batch;
            for (; b0 + 3 < batch; b0 += 4) {
                float scores[4];
                int bb;
                microgemm_avx2_i8_batch_row_tile(
                    scores,
                    0,
                    1,
                    row,
                    row_scale,
                    row_sum,
                    input_q + (size_t)b0 * cols,
                    input_scales + b0,
                    0,
                    4,
                    cols,
                    NULL
                );
                for (bb = 0; bb < 4; ++bb) {
                    microgemm_argmax_update_score(thread_best, thread_ids, b0 + bb, row_idx, scores[bb]);
                }
            }
            for (; b0 + 1 < batch; b0 += 2) {
                float scores[2];
                int bb;
                microgemm_avx2_i8_batch_row_tile(
                    scores,
                    0,
                    1,
                    row,
                    row_scale,
                    row_sum,
                    input_q + (size_t)b0 * cols,
                    input_scales + b0,
                    0,
                    2,
                    cols,
                    NULL
                );
                for (bb = 0; bb < 2; ++bb) {
                    microgemm_argmax_update_score(thread_best, thread_ids, b0 + bb, row_idx, scores[bb]);
                }
            }
            if (b0 < batch) {
                float score;
                microgemm_avx2_i8_batch_row_tile(
                    &score,
                    0,
                    1,
                    row,
                    row_scale,
                    row_sum,
                    input_q + (size_t)b0 * cols,
                    input_scales + b0,
                    0,
                    1,
                    cols,
                    NULL
                );
                microgemm_argmax_update_score(thread_best, thread_ids, b0, row_idx, score);
            }
        }

        for (b = 0; b < batch; ++b) {
            float best_value = best_values[b];
            int best_id = best_ids[b];
            int t;
            for (t = 1; t < thread_count; ++t) {
                size_t idx = (size_t)t * (size_t)batch + (size_t)b;
                float value = best_values[idx];
                int token_id = best_ids[idx];
                if (value > best_value || (value == best_value && token_id < best_id)) {
                    best_value = value;
                    best_id = token_id;
                }
            }
            out_token_ids[b] = best_id;
        }
        free(best_values);
        free(best_ids);
        return MICROGEMM_STATUS_OK;
    }
#endif

    {
        float* logits;
        size_t logits_elems;

        if ((size_t)batch > ((size_t)-1) / (size_t)rows) {
            return MICROGEMM_STATUS_OUT_OF_MEMORY;
        }
        logits_elems = (size_t)batch * (size_t)rows;
        logits = (float*)malloc(logits_elems * sizeof(float));
        if (logits == NULL) {
            return MICROGEMM_STATUS_OUT_OF_MEMORY;
        }
        microgemm_gemv_i8_batched(
            logits,
            weights,
            scales,
            row_sums,
            input,
            batch,
            rows,
            cols,
            input_stride,
            NULL,
            input_q,
            input_scales
        );
        for (b = 0; b < batch; ++b) {
            const float* row = logits + (size_t)b * rows;
            float best_value = row[0];
            int best_id = 0;
            int i;
            for (i = 1; i < rows; ++i) {
                if (row[i] > best_value) {
                    best_value = row[i];
                    best_id = i;
                }
            }
            out_token_ids[b] = best_id;
        }
        free(logits);
    }
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_lm_head_argmax_groupwise_batched(
    const microgemm_config* config,
    int* out_token_ids,
    const int8_t* weights_i8,
    const uint8_t* weights_i4,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int batch,
    int rows,
    int cols,
    int input_stride,
    int8_t* input_q,
    float* input_scales
) {
    int b;

    if (config == NULL || out_token_ids == NULL || scales == NULL
            || input == NULL || input_q == NULL || input_scales == NULL
            || batch <= 0 || rows <= 0 || cols <= 0) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (config->quant_mode == MICROGEMM_QUANT_INT8G128 && weights_i8 == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (config->quant_mode == MICROGEMM_QUANT_INT4G128 && weights_i4 == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

#if MICROGEMM_CPU_X86_AVX2
    {
        const int groups = microgemm_quant_group_count_int(cols);
        const int use_i4 = config->quant_mode == MICROGEMM_QUANT_INT4G128;
        const int use_i8_safe = !use_i4
            && microgemm_i8g_saturation_safe_enabled_for(batch, rows, cols);
        const int use_i8_safe_row_pair = !use_i4
            && use_i8_safe
            && microgemm_i8g_sat_safe_lm_head_row_pair_tile4_enabled_for(batch, rows, cols);
        const int use_i8_scores8 = !use_i4
            && !use_i8_safe
            && microgemm_i8g_lm_head_scores8_enabled_for(batch, rows, cols);
        const int use_i8_row_pair = !use_i4
            && !use_i8_safe
            && microgemm_i8g_lm_head_row_pair_tile4_enabled_for(batch, rows, cols);
        const size_t row_bytes = (size_t)((cols + 1) / 2);
        int i;
        int thread_count = 1;
        int use_stack_best = 0;
        size_t best_count;
        float best_values_stack[MICROGEMM_LM_HEAD_STACK_BEST_LIMIT];
        int best_ids_stack[MICROGEMM_LM_HEAD_STACK_BEST_LIMIT];
        float* best_values;
        int* best_ids;

        if (microgemm_decode_batch_profile_enabled) {
            microgemm_decode_batch_profile_state.groupwise_lm_head_argmax_calls += 1u;
            if (use_i8_safe_row_pair || use_i8_row_pair) {
                microgemm_decode_batch_profile_state.groupwise_lm_head_row_pair_calls += 1u;
            }
        }

        #pragma omp parallel for schedule(static) if(batch >= 8)
        for (b = 0; b < batch; ++b) {
            if (use_i4) {
                input_scales[b] = microgemm_quantize_activation_for_prebiased_maddubs_dot(
                    input_q + (size_t)b * cols,
                    input + (size_t)b * input_stride,
                    cols
                );
            } else {
                input_scales[b] = microgemm_quantize_activation_for_i8_self_biasing_dot(
                    input_q + (size_t)b * cols,
                    input + (size_t)b * input_stride,
                    cols
                );
            }
        }

#ifdef _OPENMP
        thread_count = omp_get_max_threads();
        if (thread_count < 1) {
            thread_count = 1;
        }
#endif
        if ((size_t)thread_count > ((size_t)-1) / (size_t)batch) {
            return MICROGEMM_STATUS_OUT_OF_MEMORY;
        }
        best_count = (size_t)thread_count * (size_t)batch;
        use_stack_best = microgemm_lm_head_stack_best_enabled_for(batch, thread_count);
        if (use_stack_best && microgemm_decode_batch_profile_enabled) {
            microgemm_decode_batch_profile_state.lm_head_stack_best_calls += 1u;
        }
        if (use_stack_best) {
            best_values = best_values_stack;
            best_ids = best_ids_stack;
        } else {
            best_values = (float*)malloc(best_count * sizeof(float));
            best_ids = (int*)malloc(best_count * sizeof(int));
            if (best_values == NULL || best_ids == NULL) {
                free(best_values);
                free(best_ids);
                return MICROGEMM_STATUS_OUT_OF_MEMORY;
            }
        }
        {
            size_t idx;
            for (idx = 0u; idx < best_count; ++idx) {
                best_values[idx] = -FLT_MAX;
                best_ids[idx] = 0;
            }
        }

        if (use_i8_safe_row_pair) {
            const int pair_count = rows / 2;

            #pragma omp parallel for schedule(static) if(rows >= 96)
            for (i = 0; i < pair_count; ++i) {
                const int row_idx = i * 2;
                const int8_t* row0 = weights_i8 + (size_t)row_idx * cols;
                const int8_t* row1 = row0 + cols;
                const float* row0_scales = scales + (size_t)row_idx * groups;
                const float* row1_scales = row0_scales + groups;
                int thread_index = 0;
                float* thread_best;
                int* thread_ids;
                int b0 = 0;

#ifdef _OPENMP
                thread_index = omp_get_thread_num();
#endif
                thread_best = best_values + (size_t)thread_index * (size_t)batch;
                thread_ids = best_ids + (size_t)thread_index * (size_t)batch;

                for (; b0 + 3 < batch; b0 += 4) {
                    float scores[8];
                    int bb;
                    microgemm_avx2_i8_groupwise_batch_row_pair_tile4_safe(
                        scores,
                        0,
                        2,
                        row0,
                        row1,
                        row0_scales,
                        row1_scales,
                        input_q + (size_t)b0 * cols,
                        input_scales + b0,
                        0,
                        cols,
                        groups,
                        NULL
                    );
                    for (bb = 0; bb < 4; ++bb) {
                        const int batch_index = b0 + bb;
                        microgemm_argmax_update_score(
                            thread_best, thread_ids, batch_index, row_idx, scores[bb * 2]
                        );
                        microgemm_argmax_update_score(
                            thread_best, thread_ids, batch_index, row_idx + 1, scores[bb * 2 + 1]
                        );
                    }
                }
                for (; b0 < batch; ++b0) {
                    float scores0[1];
                    float scores1[1];
                    microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                        scores0, row0, row0_scales,
                        input_q, input_scales, b0, 1, cols, groups
                    );
                    microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                        scores1, row1, row1_scales,
                        input_q, input_scales, b0, 1, cols, groups
                    );
                    microgemm_argmax_update_score(thread_best, thread_ids, b0, row_idx, scores0[0]);
                    microgemm_argmax_update_score(thread_best, thread_ids, b0, row_idx + 1, scores1[0]);
                }
            }

            if ((rows & 1) != 0) {
                const int row_idx = rows - 1;
                const int8_t* row = weights_i8 + (size_t)row_idx * cols;
                const float* row_scales = scales + (size_t)row_idx * groups;
                int thread_index = 0;
                float* thread_best;
                int* thread_ids;
                int b0 = 0;

#ifdef _OPENMP
                thread_index = omp_get_thread_num();
#endif
                thread_best = best_values + (size_t)thread_index * (size_t)batch;
                thread_ids = best_ids + (size_t)thread_index * (size_t)batch;

                for (; b0 + 3 < batch; b0 += 4) {
                    float scores[4];
                    int bb;
                    microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                        scores, row, row_scales,
                        input_q, input_scales, b0, 4, cols, groups
                    );
                    for (bb = 0; bb < 4; ++bb) {
                        microgemm_argmax_update_score(thread_best, thread_ids, b0 + bb, row_idx, scores[bb]);
                    }
                }
                for (; b0 < batch; ++b0) {
                    float score[1];
                    microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                        score, row, row_scales,
                        input_q, input_scales, b0, 1, cols, groups
                    );
                    microgemm_argmax_update_score(thread_best, thread_ids, b0, row_idx, score[0]);
                }
            }

            for (b = 0; b < batch; ++b) {
                float best_value = best_values[b];
                int best_id = best_ids[b];
                int t;
                for (t = 1; t < thread_count; ++t) {
                    size_t idx = (size_t)t * (size_t)batch + (size_t)b;
                    float value = best_values[idx];
                    int token_id = best_ids[idx];
                    if (value > best_value || (value == best_value && token_id < best_id)) {
                        best_value = value;
                        best_id = token_id;
                    }
                }
                out_token_ids[b] = best_id;
            }
            if (!use_stack_best) {
                free(best_values);
                free(best_ids);
            }
            return MICROGEMM_STATUS_OK;
        }

        if (use_i8_row_pair) {
            const int pair_count = rows / 2;

            #pragma omp parallel for schedule(static) if(rows >= 96)
            for (i = 0; i < pair_count; ++i) {
                const int row_idx = i * 2;
                const int8_t* row0 = weights_i8 + (size_t)row_idx * cols;
                const int8_t* row1 = row0 + cols;
                const float* row0_scales = scales + (size_t)row_idx * groups;
                const float* row1_scales = row0_scales + groups;
                const int32_t* row0_sums = row_sums ? row_sums + (size_t)row_idx * groups : NULL;
                const int32_t* row1_sums = row0_sums ? row0_sums + groups : NULL;
                int thread_index = 0;
                float* thread_best;
                int* thread_ids;
                int b0 = 0;

#ifdef _OPENMP
                thread_index = omp_get_thread_num();
#endif
                thread_best = best_values + (size_t)thread_index * (size_t)batch;
                thread_ids = best_ids + (size_t)thread_index * (size_t)batch;

                for (; b0 + 3 < batch; b0 += 4) {
                    float scores[8];
                    int bb;
                    microgemm_avx2_i8_groupwise_batch_row_pair_tile4(
                        scores,
                        0,
                        2,
                        row0,
                        row1,
                        row0_scales,
                        row1_scales,
                        row0_sums,
                        row1_sums,
                        input_q + (size_t)b0 * cols,
                        input_scales + b0,
                        0,
                        cols,
                        groups,
                        NULL
                    );
                    for (bb = 0; bb < 4; ++bb) {
                        const int batch_index = b0 + bb;
                        microgemm_argmax_update_score(
                            thread_best, thread_ids, batch_index, row_idx, scores[bb * 2]
                        );
                        microgemm_argmax_update_score(
                            thread_best, thread_ids, batch_index, row_idx + 1, scores[bb * 2 + 1]
                        );
                    }
                }
                for (; b0 < batch; ++b0) {
                    float scores0[1];
                    float scores1[1];
                    microgemm_avx2_i8_groupwise_batch_row_scores(
                        scores0, row0, row0_scales, row0_sums,
                        input_q, input_scales, b0, 1, cols, groups
                    );
                    microgemm_avx2_i8_groupwise_batch_row_scores(
                        scores1, row1, row1_scales, row1_sums,
                        input_q, input_scales, b0, 1, cols, groups
                    );
                    microgemm_argmax_update_score(thread_best, thread_ids, b0, row_idx, scores0[0]);
                    microgemm_argmax_update_score(thread_best, thread_ids, b0, row_idx + 1, scores1[0]);
                }
            }

            if ((rows & 1) != 0) {
                const int row_idx = rows - 1;
                const int8_t* row = weights_i8 + (size_t)row_idx * cols;
                const float* row_scales = scales + (size_t)row_idx * groups;
                const int32_t* row_group_sums = row_sums ? row_sums + (size_t)row_idx * groups : NULL;
                int thread_index = 0;
                float* thread_best;
                int* thread_ids;
                int b0 = 0;

#ifdef _OPENMP
                thread_index = omp_get_thread_num();
#endif
                thread_best = best_values + (size_t)thread_index * (size_t)batch;
                thread_ids = best_ids + (size_t)thread_index * (size_t)batch;

                for (; b0 + 7 < batch; b0 += 8) {
                    float scores[8];
                    int bb;
                    if (use_i8_scores8) {
                        microgemm_avx2_i8_groupwise_batch_row_scores8_explicit(
                            scores, row, row_scales, row_group_sums,
                            input_q, input_scales, b0, cols, groups
                        );
                    } else {
                        microgemm_avx2_i8_groupwise_batch_row_scores(
                            scores, row, row_scales, row_group_sums,
                            input_q, input_scales, b0, 8, cols, groups
                        );
                    }
                    for (bb = 0; bb < 8; ++bb) {
                        microgemm_argmax_update_score(thread_best, thread_ids, b0 + bb, row_idx, scores[bb]);
                    }
                }
                for (; b0 < batch; ++b0) {
                    float score[1];
                    microgemm_avx2_i8_groupwise_batch_row_scores(
                        score, row, row_scales, row_group_sums,
                        input_q, input_scales, b0, 1, cols, groups
                    );
                    microgemm_argmax_update_score(thread_best, thread_ids, b0, row_idx, score[0]);
                }
            }

            for (b = 0; b < batch; ++b) {
                float best_value = best_values[b];
                int best_id = best_ids[b];
                int t;
                for (t = 1; t < thread_count; ++t) {
                    size_t idx = (size_t)t * (size_t)batch + (size_t)b;
                    float value = best_values[idx];
                    int token_id = best_ids[idx];
                    if (value > best_value || (value == best_value && token_id < best_id)) {
                        best_value = value;
                        best_id = token_id;
                    }
                }
                out_token_ids[b] = best_id;
            }
            if (!use_stack_best) {
                free(best_values);
                free(best_ids);
            }
            return MICROGEMM_STATUS_OK;
        }

        #pragma omp parallel for schedule(static) if(rows >= 96)
        for (i = 0; i < rows; ++i) {
            const float* row_scales = scales + (size_t)i * groups;
            const int32_t* row_group_sums = row_sums ? row_sums + (size_t)i * groups : NULL;
            int thread_index = 0;
            float* thread_best;
            int* thread_ids;
            int b0 = 0;

#ifdef _OPENMP
            thread_index = omp_get_thread_num();
#endif
            thread_best = best_values + (size_t)thread_index * (size_t)batch;
            thread_ids = best_ids + (size_t)thread_index * (size_t)batch;

            for (; b0 + 7 < batch; b0 += 8) {
                float scores[8];
                int bb;
                if (use_i4) {
                    const uint8_t* row = weights_i4 + (size_t)i * row_bytes;
                    microgemm_avx2_i4_groupwise_batch_row_scores(
                        scores, row, row_scales, row_group_sums,
                        input_q, input_scales, b0, 8, cols, groups
                    );
                } else {
                    const int8_t* row = weights_i8 + (size_t)i * cols;
                    if (use_i8_safe) {
                        microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                            scores, row, row_scales, input_q, input_scales, b0, 4, cols, groups
                        );
                        microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                            scores + 4, row, row_scales, input_q, input_scales, b0 + 4, 4, cols, groups
                        );
                    } else if (use_i8_scores8) {
                        microgemm_avx2_i8_groupwise_batch_row_scores8_explicit(
                            scores, row, row_scales, row_group_sums,
                            input_q, input_scales, b0, cols, groups
                        );
                    } else {
                        microgemm_avx2_i8_groupwise_batch_row_scores(
                            scores, row, row_scales, row_group_sums,
                            input_q, input_scales, b0, 8, cols, groups
                        );
                    }
                }
                for (bb = 0; bb < 8; ++bb) {
                    microgemm_argmax_update_score(thread_best, thread_ids, b0 + bb, i, scores[bb]);
                }
            }
            for (; b0 + 3 < batch; b0 += 4) {
                float scores[4];
                int bb;
                if (use_i4) {
                    const uint8_t* row = weights_i4 + (size_t)i * row_bytes;
                    microgemm_avx2_i4_groupwise_batch_row_scores(
                        scores, row, row_scales, row_group_sums,
                        input_q, input_scales, b0, 4, cols, groups
                    );
                } else {
                    const int8_t* row = weights_i8 + (size_t)i * cols;
                    if (use_i8_safe) {
                        microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                            scores, row, row_scales, input_q, input_scales, b0, 4, cols, groups
                        );
                    } else {
                        microgemm_avx2_i8_groupwise_batch_row_scores(
                            scores, row, row_scales, row_group_sums,
                            input_q, input_scales, b0, 4, cols, groups
                        );
                    }
                }
                for (bb = 0; bb < 4; ++bb) {
                    microgemm_argmax_update_score(thread_best, thread_ids, b0 + bb, i, scores[bb]);
                }
            }
            for (; b0 + 1 < batch; b0 += 2) {
                float scores[2];
                int bb;
                if (use_i4) {
                    const uint8_t* row = weights_i4 + (size_t)i * row_bytes;
                    microgemm_avx2_i4_groupwise_batch_row_scores(
                        scores, row, row_scales, row_group_sums,
                        input_q, input_scales, b0, 2, cols, groups
                    );
                } else {
                    const int8_t* row = weights_i8 + (size_t)i * cols;
                    if (use_i8_safe) {
                        microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                            scores, row, row_scales, input_q, input_scales, b0, 2, cols, groups
                        );
                    } else {
                        microgemm_avx2_i8_groupwise_batch_row_scores(
                            scores, row, row_scales, row_group_sums,
                            input_q, input_scales, b0, 2, cols, groups
                        );
                    }
                }
                for (bb = 0; bb < 2; ++bb) {
                    microgemm_argmax_update_score(thread_best, thread_ids, b0 + bb, i, scores[bb]);
                }
            }
            if (b0 < batch) {
                float score[1];
                if (use_i4) {
                    const uint8_t* row = weights_i4 + (size_t)i * row_bytes;
                    microgemm_avx2_i4_groupwise_batch_row_scores(
                        score, row, row_scales, row_group_sums,
                        input_q, input_scales, b0, 1, cols, groups
                    );
                } else {
                    const int8_t* row = weights_i8 + (size_t)i * cols;
                    if (use_i8_safe) {
                        microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                            score, row, row_scales, input_q, input_scales, b0, 1, cols, groups
                        );
                    } else {
                        microgemm_avx2_i8_groupwise_batch_row_scores(
                            score, row, row_scales, row_group_sums,
                            input_q, input_scales, b0, 1, cols, groups
                        );
                    }
                }
                microgemm_argmax_update_score(thread_best, thread_ids, b0, i, score[0]);
            }
        }

        for (b = 0; b < batch; ++b) {
            float best_value = best_values[b];
            int best_id = best_ids[b];
            int t;
            for (t = 1; t < thread_count; ++t) {
                size_t idx = (size_t)t * (size_t)batch + (size_t)b;
                float value = best_values[idx];
                int token_id = best_ids[idx];
                if (value > best_value || (value == best_value && token_id < best_id)) {
                    best_value = value;
                    best_id = token_id;
                }
            }
            out_token_ids[b] = best_id;
        }
        if (!use_stack_best) {
            free(best_values);
            free(best_ids);
        }
        return MICROGEMM_STATUS_OK;
    }
#else
    {
        float* logits;
        size_t logits_elems;

        if ((size_t)batch > ((size_t)-1) / (size_t)rows) {
            return MICROGEMM_STATUS_OUT_OF_MEMORY;
        }
        logits_elems = (size_t)batch * (size_t)rows;
        logits = (float*)malloc(logits_elems * sizeof(float));
        if (logits == NULL) {
            return MICROGEMM_STATUS_OUT_OF_MEMORY;
        }
        microgemm_gemv_quantized_batched(
            config,
            logits,
            weights_i8,
            weights_i4,
            scales,
            row_sums,
            input,
            batch,
            rows,
            cols,
            input_stride,
            NULL,
            input_q,
            input_scales
        );
        for (b = 0; b < batch; ++b) {
            const float* row = logits + (size_t)b * rows;
            float best_value = row[0];
            int best_id = 0;
            int i;
            for (i = 1; i < rows; ++i) {
                if (row[i] > best_value) {
                    best_value = row[i];
                    best_id = i;
                }
            }
            out_token_ids[b] = best_id;
        }
        free(logits);
        return MICROGEMM_STATUS_OK;
    }
#endif
}

static microgemm_status microgemm_lm_head_argmax_quantized_batched(
    const microgemm_config* config,
    int* out_token_ids,
    const int8_t* weights_i8,
    const uint8_t* weights_i4,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int batch,
    int rows,
    int cols,
    int input_stride,
    int8_t* input_q,
    float* input_scales
) {
    if (config != NULL && microgemm_quant_mode_is_groupwise(config->quant_mode)) {
        return microgemm_lm_head_argmax_groupwise_batched(
            config,
            out_token_ids,
            weights_i8,
            weights_i4,
            scales,
            row_sums,
            input,
            batch,
            rows,
            cols,
            input_stride,
            input_q,
            input_scales
        );
    }

    if (config == NULL || config->quant_mode != MICROGEMM_QUANT_INT4) {
        return microgemm_lm_head_argmax_i8_batched(
            out_token_ids, weights_i8, scales, row_sums, input, batch, rows, cols,
            input_stride, input_q, input_scales
        );
    }

    {
        float* logits;
        size_t logits_elems;
        int b;

        if ((size_t)batch > ((size_t)-1) / (size_t)rows) {
            return MICROGEMM_STATUS_OUT_OF_MEMORY;
        }
        logits_elems = (size_t)batch * (size_t)rows;
        logits = (float*)malloc(logits_elems * sizeof(float));
        if (logits == NULL) {
            return MICROGEMM_STATUS_OUT_OF_MEMORY;
        }
        microgemm_gemv_i4_packed_batched(
            logits,
            weights_i4,
            scales,
            row_sums,
            input,
            batch,
            rows,
            cols,
            input_stride,
            NULL,
            input_q,
            input_scales
        );
        for (b = 0; b < batch; ++b) {
            const float* row = logits + (size_t)b * rows;
            float best_value = row[0];
            int best_id = 0;
            int i;
            for (i = 1; i < rows; ++i) {
                if (row[i] > best_value) {
                    best_value = row[i];
                    best_id = i;
                }
            }
            out_token_ids[b] = best_id;
        }
        free(logits);
    }
    return MICROGEMM_STATUS_OK;
}

static void microgemm_gate_up_swiglu_i8_batched(
    float* out,
    const int8_t* weights,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int batch,
    int intermediate,
    int cols,
    int input_stride,
    int out_stride,
    int use_gelu,
    int8_t* input_q,
    float* input_scales,
    float* swiglu_absmax_scratch,
    int swiglu_absmax_threads
) {
    int b;
    (void)swiglu_absmax_scratch;
    (void)swiglu_absmax_threads;

    if (batch <= 0) {
        return;
    }

#if MICROGEMM_CPU_X86_AVX2
    if (row_sums != NULL && batch >= 2) {
        int i;
        int profile_enabled = microgemm_decode_batch_profile_enabled;
        int use_tile8_group4 = microgemm_i8_gate_tile8_group4_enabled_for(batch, intermediate, cols);
        int use_tile8_fused = microgemm_i8_gate_tile8_fused_enabled_for(batch, intermediate, cols);
        double profile_phase_start = 0.0;

        if (profile_enabled) {
            profile_phase_start = microgemm_profile_now_ms();
        }
        for (b = 0; b < batch; ++b) {
            int8_t* q = input_q + (size_t)b * cols;
            input_scales[b] = microgemm_quantize_activation_for_prebiased_maddubs_dot(
                q,
                input + (size_t)b * input_stride,
                cols
            );
        }
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.gate_up_quant_ms, profile_phase_start);
            profile_phase_start = microgemm_profile_now_ms();
        }

        #pragma omp parallel for schedule(static) if(intermediate >= 96)
        for (i = 0; i < intermediate; ++i) {
            const int8_t* gate_row = weights + (size_t)i * cols;
            const int8_t* up_row = weights + (size_t)(intermediate + i) * cols;
            int b0 = 0;

            for (; b0 + 7 < batch; b0 += 8) {
                if (use_tile8_fused) {
                    microgemm_avx2_i8_gate_up_tile8_fused(
                        out,
                        i,
                        out_stride,
                        gate_row,
                        up_row,
                        scales[i],
                        scales[intermediate + i],
                        row_sums[i],
                        row_sums[intermediate + i],
                        input_q,
                        input_scales,
                        b0,
                        cols,
                        use_gelu
                    );
                } else if (use_tile8_group4) {
                    microgemm_avx2_i8_gate_up_tile4(
                        out,
                        i,
                        out_stride,
                        gate_row,
                        up_row,
                        scales[i],
                        scales[intermediate + i],
                        row_sums[i],
                        row_sums[intermediate + i],
                        input_q,
                        input_scales,
                        b0,
                        cols,
                        use_gelu
                    );
                    microgemm_avx2_i8_gate_up_tile4(
                        out,
                        i,
                        out_stride,
                        gate_row,
                        up_row,
                        scales[i],
                        scales[intermediate + i],
                        row_sums[i],
                        row_sums[intermediate + i],
                        input_q,
                        input_scales,
                        b0 + 4,
                        cols,
                        use_gelu
                    );
                } else {
                    microgemm_avx2_i8_gate_up_tile8_split(
                        out,
                        i,
                        out_stride,
                        gate_row,
                        up_row,
                        scales[i],
                        scales[intermediate + i],
                        row_sums[i],
                        row_sums[intermediate + i],
                        input_q,
                        input_scales,
                        b0,
                        cols,
                        use_gelu
                    );
                }
            }
            for (; b0 + 3 < batch; b0 += 4) {
                microgemm_avx2_i8_gate_up_tile4(
                    out,
                    i,
                    out_stride,
                    gate_row,
                    up_row,
                    scales[i],
                    scales[intermediate + i],
                    row_sums[i],
                    row_sums[intermediate + i],
                    input_q,
                    input_scales,
                    b0,
                    cols,
                    use_gelu
                );
            }
            for (; b0 + 1 < batch; b0 += 2) {
                microgemm_avx2_i8_gate_up_tile2(
                    out,
                    i,
                    out_stride,
                    gate_row,
                    up_row,
                    scales[i],
                    scales[intermediate + i],
                    row_sums[i],
                    row_sums[intermediate + i],
                    input_q,
                    input_scales,
                    b0,
                    cols,
                    use_gelu
                );
            }
            if (b0 < batch) {
                microgemm_avx2_i8_gate_up_tile(
                    out,
                    i,
                    out_stride,
                    gate_row,
                    up_row,
                    scales[i],
                    scales[intermediate + i],
                    row_sums[i],
                    row_sums[intermediate + i],
                    input_q,
                    input_scales,
                    b0,
                    1,
                    cols,
                    use_gelu
                );
            }
        }
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.gate_up_dot_ms, profile_phase_start);
        }
        return;
    }
#endif

    microgemm_gemv_i8_batched(
        out,
        weights,
        scales,
        row_sums,
        input,
        batch,
        2 * intermediate,
        cols,
        input_stride,
        NULL,
        input_q,
        input_scales
    );

    #pragma omp parallel for schedule(static) if(batch >= 8)
    for (b = 0; b < batch; ++b) {
        float* gate_t = out + (size_t)b * out_stride;
        microgemm_swiglu_activation(gate_t, gate_t + intermediate, intermediate, use_gelu);
    }
}

static void microgemm_gate_up_swiglu_groupwise_batched(
    const microgemm_config* config,
    float* out,
    const int8_t* weights_i8,
    const uint8_t* weights_i4,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int batch,
    int intermediate,
    int cols,
    int input_stride,
    int out_stride,
    int use_gelu,
    int8_t* input_q,
    float* input_scales,
    float* swiglu_absmax_scratch,
    int swiglu_absmax_threads
) {
    int b;

#if MICROGEMM_CPU_X86_AVX2
    if (config != NULL
            && config->quant_mode == MICROGEMM_QUANT_INT8G128
            && weights_i8 != NULL
            && scales != NULL
            && input != NULL
            && input_q != NULL
            && input_scales != NULL
            && batch > 0
            && intermediate > 0
            && cols > 0
            && microgemm_i8g_gate_safe_fused_enabled_for(batch, intermediate, cols)) {
        const int groups = microgemm_quant_group_count_int(cols);
        const int use_combined_tile4 =
            microgemm_i8g_gate_safe_combined_tile4_enabled_for(batch, intermediate, cols);
        const int use_combined_tile8 =
            microgemm_i8g_gate_safe_combined_tile8_enabled_for(batch, intermediate, cols);
        int i;
        int profile_enabled = microgemm_decode_batch_profile_enabled;
        double profile_phase_start = 0.0;

        if (swiglu_absmax_scratch != NULL && swiglu_absmax_threads > 0) {
            memset(
                swiglu_absmax_scratch,
                0,
                (size_t)swiglu_absmax_threads * (size_t)batch * sizeof(float)
            );
        }
        if (profile_enabled) {
            microgemm_decode_batch_profile_state.groupwise_gate_up_fused_calls += 1u;
            if (use_combined_tile4) {
                microgemm_decode_batch_profile_state.groupwise_i8_gate_safe_combined_calls += 1u;
            }
            if (use_combined_tile8) {
                microgemm_decode_batch_profile_state.groupwise_i8_gate_safe_combined_tile8_calls += 1u;
            }
            profile_phase_start = microgemm_profile_now_ms();
        }

        #pragma omp parallel for schedule(static) if(batch >= 8)
        for (b = 0; b < batch; ++b) {
            input_scales[b] = microgemm_quantize_activation_for_i8_self_biasing_dot(
                input_q + (size_t)b * cols,
                input + (size_t)b * input_stride,
                cols
            );
        }
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.gate_up_quant_ms, profile_phase_start);
            profile_phase_start = microgemm_profile_now_ms();
        }

        #pragma omp parallel for schedule(static) if(intermediate >= 96)
        for (i = 0; i < intermediate; ++i) {
            const int8_t* gate_row = weights_i8 + (size_t)i * cols;
            const int8_t* up_row = weights_i8 + (size_t)(intermediate + i) * cols;
            const float* gate_scales = scales + (size_t)i * groups;
            const float* up_scales = scales + (size_t)(intermediate + i) * groups;
            int thread_index = 0;
            float* thread_absmax = NULL;
            int b0 = 0;

#ifdef _OPENMP
            thread_index = omp_get_thread_num();
#endif
            if (swiglu_absmax_scratch != NULL
                    && thread_index >= 0
                    && thread_index < swiglu_absmax_threads) {
                thread_absmax = swiglu_absmax_scratch + (size_t)thread_index * batch;
            }

            for (; b0 + 7 < batch; b0 += 8) {
                if (!use_combined_tile8) {
                    break;
                }
                microgemm_avx2_i8_groupwise_gate_up_tile8_safe(
                    out,
                    i,
                    out_stride,
                    gate_row,
                    up_row,
                    gate_scales,
                    up_scales,
                    input_q,
                    input_scales,
                    b0,
                    cols,
                    groups,
                    use_gelu,
                    thread_absmax
                );
            }
            for (; b0 + 3 < batch; b0 += 4) {
                if (use_combined_tile4) {
                    microgemm_avx2_i8_groupwise_gate_up_tile4_safe(
                        out,
                        i,
                        out_stride,
                        gate_row,
                        up_row,
                        gate_scales,
                        up_scales,
                        input_q,
                        input_scales,
                        b0,
                        cols,
                        groups,
                        use_gelu,
                        thread_absmax
                    );
                    continue;
                }
                float gate_scores[4];
                float up_scores[4];
                int bb;
                microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                    gate_scores, gate_row, gate_scales,
                    input_q, input_scales, b0, 4, cols, groups
                );
                microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                    up_scores, up_row, up_scales,
                    input_q, input_scales, b0, 4, cols, groups
                );
                for (bb = 0; bb < 4; ++bb) {
                    int batch_index = b0 + bb;
                    float gate = gate_scores[bb];
                    float value;
                    if (use_gelu) {
                        float x = gate;
                        gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
                    } else {
                        gate = microgemm_silu(gate);
                    }
                    value = gate * up_scores[bb];
                    out[(size_t)batch_index * out_stride + i] = value;
                    if (thread_absmax != NULL) {
                        float abs_value = fabsf(value);
                        if (abs_value > thread_absmax[batch_index]) {
                            thread_absmax[batch_index] = abs_value;
                        }
                    }
                }
            }
            for (; b0 + 1 < batch; b0 += 2) {
                float gate_scores[2];
                float up_scores[2];
                int bb;
                microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                    gate_scores, gate_row, gate_scales,
                    input_q, input_scales, b0, 2, cols, groups
                );
                microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                    up_scores, up_row, up_scales,
                    input_q, input_scales, b0, 2, cols, groups
                );
                for (bb = 0; bb < 2; ++bb) {
                    int batch_index = b0 + bb;
                    float gate = gate_scores[bb];
                    float value;
                    if (use_gelu) {
                        float x = gate;
                        gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
                    } else {
                        gate = microgemm_silu(gate);
                    }
                    value = gate * up_scores[bb];
                    out[(size_t)batch_index * out_stride + i] = value;
                    if (thread_absmax != NULL) {
                        float abs_value = fabsf(value);
                        if (abs_value > thread_absmax[batch_index]) {
                            thread_absmax[batch_index] = abs_value;
                        }
                    }
                }
            }
            if (b0 < batch) {
                float gate_score[1];
                float up_score[1];
                float gate;
                float value;
                microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                    gate_score, gate_row, gate_scales,
                    input_q, input_scales, b0, 1, cols, groups
                );
                microgemm_avx2_i8_groupwise_batch_row_scores_safe(
                    up_score, up_row, up_scales,
                    input_q, input_scales, b0, 1, cols, groups
                );
                gate = gate_score[0];
                if (use_gelu) {
                    float x = gate;
                    gate = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
                } else {
                    gate = microgemm_silu(gate);
                }
                value = gate * up_score[0];
                out[(size_t)b0 * out_stride + i] = value;
                if (thread_absmax != NULL) {
                    float abs_value = fabsf(value);
                    if (abs_value > thread_absmax[b0]) {
                        thread_absmax[b0] = abs_value;
                    }
                }
            }
        }
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.gate_up_dot_ms, profile_phase_start);
        }
        return;
    }
#endif

#if MICROGEMM_CPU_X86_AVX2
    if (config != NULL
            && row_sums != NULL
            && batch >= 2
            && microgemm_groupwise_gate_up_fused_enabled_for(
                config->quant_mode, batch, intermediate, cols
            )) {
        const int groups = microgemm_quant_group_count_int(cols);
        const int use_i4 = config->quant_mode == MICROGEMM_QUANT_INT4G128;
        const int use_i8_gate_tile8_explicit = !use_i4
            && microgemm_i8g_gate_tile8_explicit_enabled_for(batch, intermediate, cols);
        const int use_i8_gate_tile8_aligned128 = use_i8_gate_tile8_explicit
            && row_sums != NULL
            && microgemm_i8g_gate_tile8_aligned128_enabled_for(batch, intermediate, cols);
        const int use_i8_gate_tile8_biased_input = use_i8_gate_tile8_explicit
            && row_sums != NULL
            && microgemm_i8g_gate_tile8_biased_input_enabled_for(batch, intermediate, cols);
        const int use_i8_gate_pair4_biased = !use_i4
            && use_i8_gate_tile8_biased_input
            && microgemm_i8g_gate_pair_tile4_enabled_for(batch, intermediate, cols);
        const int use_i8_gate_pair4_unroll128 = use_i8_gate_pair4_biased
            && microgemm_i8g_gate_pair_tile4_unroll128_enabled_for(batch, intermediate, cols);
        const int use_i8_gate_pair4_unroll64 = use_i8_gate_pair4_biased
            && !use_i8_gate_pair4_unroll128
            && microgemm_i8g_gate_pair_tile4_unroll64_enabled_for(batch, intermediate, cols);
        const int use_i8_gate_pair8_splitpass = use_i8_gate_pair4_biased
            && microgemm_i8g_gate_pair_tile8_splitpass_enabled_for(batch, intermediate, cols);
        const int use_i8_gate_prefetch = !use_i4
            && microgemm_i8g_gate_prefetch_enabled_for(batch, intermediate, cols);
        const size_t row_bytes = (size_t)((cols + 1) / 2);
        int i;
        int profile_enabled = microgemm_decode_batch_profile_enabled;
        double profile_phase_start = 0.0;
        if (swiglu_absmax_scratch != NULL && swiglu_absmax_threads > 0) {
            memset(
                swiglu_absmax_scratch,
                0,
                (size_t)swiglu_absmax_threads * (size_t)batch * sizeof(float)
            );
        }

        if (profile_enabled) {
            microgemm_decode_batch_profile_state.groupwise_gate_up_fused_calls += 1u;
            if (use_i8_gate_tile8_explicit) {
                microgemm_decode_batch_profile_state.groupwise_i8_gate_tile8_calls += 1u;
            }
            if (use_i8_gate_tile8_biased_input) {
                microgemm_decode_batch_profile_state.groupwise_i8_gate_biased_calls += 1u;
            }
            if (use_i8_gate_pair4_biased) {
                microgemm_decode_batch_profile_state.groupwise_i8_gate_pair_calls += 1u;
            }
            if (use_i8_gate_pair4_unroll64) {
                microgemm_decode_batch_profile_state.groupwise_i8_gate_pair_unroll64_calls += 1u;
            }
            if (use_i8_gate_pair4_unroll128) {
                microgemm_decode_batch_profile_state.groupwise_i8_gate_pair_unroll128_calls += 1u;
            }
            if (use_i8_gate_pair8_splitpass) {
                microgemm_decode_batch_profile_state.groupwise_i8_gate_pair8_split_calls += 1u;
            }
            if (use_i8_gate_prefetch) {
                microgemm_decode_batch_profile_state.groupwise_i8_gate_prefetch_calls += 1u;
            }
            profile_phase_start = microgemm_profile_now_ms();
        }
        #pragma omp parallel for schedule(static) if(batch >= 8)
        for (b = 0; b < batch; ++b) {
            if (use_i4) {
                input_scales[b] = microgemm_quantize_activation_for_prebiased_maddubs_dot(
                    input_q + (size_t)b * cols,
                    input + (size_t)b * input_stride,
                    cols
                );
            } else if (use_i8_gate_tile8_biased_input) {
                input_scales[b] = microgemm_quantize_activation_for_prebiased_maddubs_dot(
                    input_q + (size_t)b * cols,
                    input + (size_t)b * input_stride,
                    cols
                );
            } else {
                input_scales[b] = microgemm_quantize_activation_for_i8_self_biasing_dot(
                    input_q + (size_t)b * cols,
                    input + (size_t)b * input_stride,
                    cols
                );
            }
        }
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.gate_up_quant_ms, profile_phase_start);
            profile_phase_start = microgemm_profile_now_ms();
        }

        if (use_i8_gate_pair4_biased) {
            int pair_limit = intermediate & ~1;
            #pragma omp parallel for schedule(static) if(intermediate >= 96)
            for (i = 0; i < pair_limit; i += 2) {
                const int8_t* gate0_row = weights_i8 + (size_t)i * cols;
                const int8_t* gate1_row = gate0_row + cols;
                const int8_t* up0_row = weights_i8 + (size_t)(intermediate + i) * cols;
                const int8_t* up1_row = up0_row + cols;
                const float* gate0_scales = scales + (size_t)i * groups;
                const float* gate1_scales = gate0_scales + groups;
                const float* up0_scales = scales + (size_t)(intermediate + i) * groups;
                const float* up1_scales = up0_scales + groups;
                const int32_t* gate0_sums = row_sums + (size_t)i * groups;
                const int32_t* gate1_sums = gate0_sums + groups;
                const int32_t* up0_sums = row_sums + (size_t)(intermediate + i) * groups;
                const int32_t* up1_sums = up0_sums + groups;
                int b0 = 0;
                int thread_index = 0;
                float* thread_absmax = NULL;

#ifdef _OPENMP
                thread_index = omp_get_thread_num();
#endif
                if (swiglu_absmax_scratch != NULL
                        && thread_index >= 0
                        && thread_index < swiglu_absmax_threads) {
                    thread_absmax = swiglu_absmax_scratch + (size_t)thread_index * batch;
                }

                if (use_i8_gate_prefetch) {
                    const int next_i = i + 8;
                    if (next_i + 1 < pair_limit) {
                        _mm_prefetch((const char*)(weights_i8 + (size_t)next_i * cols), _MM_HINT_T0);
                        _mm_prefetch((const char*)(weights_i8 + (size_t)(next_i + 1) * cols), _MM_HINT_T0);
                        _mm_prefetch((const char*)(weights_i8 + (size_t)(intermediate + next_i) * cols), _MM_HINT_T0);
                        _mm_prefetch((const char*)(weights_i8 + (size_t)(intermediate + next_i + 1) * cols), _MM_HINT_T0);
                        _mm_prefetch((const char*)(scales + (size_t)next_i * groups), _MM_HINT_T0);
                        _mm_prefetch((const char*)(scales + (size_t)(next_i + 1) * groups), _MM_HINT_T0);
                        _mm_prefetch((const char*)(scales + (size_t)(intermediate + next_i) * groups), _MM_HINT_T0);
                        _mm_prefetch((const char*)(scales + (size_t)(intermediate + next_i + 1) * groups), _MM_HINT_T0);
                        _mm_prefetch((const char*)(row_sums + (size_t)next_i * groups), _MM_HINT_T0);
                        _mm_prefetch((const char*)(row_sums + (size_t)(next_i + 1) * groups), _MM_HINT_T0);
                        _mm_prefetch((const char*)(row_sums + (size_t)(intermediate + next_i) * groups), _MM_HINT_T0);
                        _mm_prefetch((const char*)(row_sums + (size_t)(intermediate + next_i + 1) * groups), _MM_HINT_T0);
                    }
                }

                for (; use_i8_gate_pair8_splitpass && b0 + 7 < batch; b0 += 8) {
                    microgemm_avx2_i8_groupwise_gate_up_pair_tile8_splitpass_biased_aligned128(
                        out, i, out_stride,
                        gate0_row, up0_row, gate1_row, up1_row,
                        gate0_scales, up0_scales, gate1_scales, up1_scales,
                        gate0_sums, up0_sums, gate1_sums, up1_sums,
                        input_q, input_scales, b0, cols, groups, use_gelu
                    );
                    microgemm_swiglu_absmax_update_rows(
                        thread_absmax, out, i, 2, out_stride, b0, 8
                    );
                }
                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i8_groupwise_gate_up_pair_tile4_biased_aligned128(
                        out, i, out_stride,
                        gate0_row, up0_row, gate1_row, up1_row,
                        gate0_scales, up0_scales, gate1_scales, up1_scales,
                        gate0_sums, up0_sums, gate1_sums, up1_sums,
                        input_q, input_scales, b0, cols, groups, use_gelu,
                        use_i8_gate_pair4_unroll64,
                        use_i8_gate_pair4_unroll128
                    );
                    microgemm_swiglu_absmax_update_rows(
                        thread_absmax, out, i, 2, out_stride, b0, 4
                    );
                }
            }
            if (pair_limit < intermediate) {
                const int tail_i = pair_limit;
                const int8_t* gate_row = weights_i8 + (size_t)tail_i * cols;
                const int8_t* up_row = weights_i8 + (size_t)(intermediate + tail_i) * cols;
                const float* gate_scales = scales + (size_t)tail_i * groups;
                const float* up_scales = scales + (size_t)(intermediate + tail_i) * groups;
                const int32_t* gate_sums = row_sums + (size_t)tail_i * groups;
                const int32_t* up_sums = row_sums + (size_t)(intermediate + tail_i) * groups;
                int b0 = 0;
                for (; b0 + 7 < batch; b0 += 8) {
                    microgemm_avx2_i8_groupwise_gate_up_tile8_biased_aligned128(
                        out, tail_i, out_stride, gate_row, up_row,
                        gate_scales, up_scales, gate_sums, up_sums,
                        input_q, input_scales, b0, cols, groups, use_gelu
                    );
                    microgemm_swiglu_absmax_update_rows(
                        swiglu_absmax_scratch, out, tail_i, 1, out_stride, b0, 8
                    );
                }
            }
            if (profile_enabled) {
                microgemm_profile_add(&microgemm_decode_batch_profile_state.gate_up_dot_ms, profile_phase_start);
            }
            return;
        }

        #pragma omp parallel for schedule(static) if(intermediate >= 96)
        for (i = 0; i < intermediate; ++i) {
            const float* gate_scales = scales + (size_t)i * groups;
            const float* up_scales = scales + (size_t)(intermediate + i) * groups;
            const int32_t* gate_sums = row_sums + (size_t)i * groups;
            const int32_t* up_sums = row_sums + (size_t)(intermediate + i) * groups;
            int b0 = 0;
            int thread_index = 0;
            float* thread_absmax = NULL;

#ifdef _OPENMP
            thread_index = omp_get_thread_num();
#endif
            if (swiglu_absmax_scratch != NULL
                    && thread_index >= 0
                    && thread_index < swiglu_absmax_threads) {
                thread_absmax = swiglu_absmax_scratch + (size_t)thread_index * batch;
            }

            if (use_i4) {
                const uint8_t* gate_row = weights_i4 + (size_t)i * row_bytes;
                const uint8_t* up_row = weights_i4 + (size_t)(intermediate + i) * row_bytes;
                for (; b0 + 7 < batch; b0 += 8) {
                    microgemm_avx2_i4_groupwise_gate_up_tile(
                        out, i, out_stride, gate_row, up_row,
                        gate_scales, up_scales, gate_sums, up_sums,
                        input_q, input_scales, b0, 8, cols, groups, use_gelu
                    );
                    microgemm_swiglu_absmax_update_rows(
                        thread_absmax, out, i, 1, out_stride, b0, 8
                    );
                }
                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i4_groupwise_gate_up_tile(
                        out, i, out_stride, gate_row, up_row,
                        gate_scales, up_scales, gate_sums, up_sums,
                        input_q, input_scales, b0, 4, cols, groups, use_gelu
                    );
                    microgemm_swiglu_absmax_update_rows(
                        thread_absmax, out, i, 1, out_stride, b0, 4
                    );
                }
                for (; b0 + 1 < batch; b0 += 2) {
                    microgemm_avx2_i4_groupwise_gate_up_tile(
                        out, i, out_stride, gate_row, up_row,
                        gate_scales, up_scales, gate_sums, up_sums,
                        input_q, input_scales, b0, 2, cols, groups, use_gelu
                    );
                    microgemm_swiglu_absmax_update_rows(
                        thread_absmax, out, i, 1, out_stride, b0, 2
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i4_groupwise_gate_up_tile(
                        out, i, out_stride, gate_row, up_row,
                        gate_scales, up_scales, gate_sums, up_sums,
                        input_q, input_scales, b0, 1, cols, groups, use_gelu
                    );
                    microgemm_swiglu_absmax_update_rows(
                        thread_absmax, out, i, 1, out_stride, b0, 1
                    );
                }
            } else {
                const int8_t* gate_row = weights_i8 + (size_t)i * cols;
                const int8_t* up_row = weights_i8 + (size_t)(intermediate + i) * cols;
                if (use_i8_gate_prefetch) {
                    const int next_i = i + 8;
                    if (next_i < intermediate) {
                        _mm_prefetch((const char*)(weights_i8 + (size_t)next_i * cols), _MM_HINT_T0);
                        _mm_prefetch((const char*)(weights_i8 + (size_t)(intermediate + next_i) * cols), _MM_HINT_T0);
                        _mm_prefetch((const char*)(scales + (size_t)next_i * groups), _MM_HINT_T0);
                        _mm_prefetch((const char*)(scales + (size_t)(intermediate + next_i) * groups), _MM_HINT_T0);
                        _mm_prefetch((const char*)(row_sums + (size_t)next_i * groups), _MM_HINT_T0);
                        _mm_prefetch((const char*)(row_sums + (size_t)(intermediate + next_i) * groups), _MM_HINT_T0);
                    }
                }
                for (; b0 + 7 < batch; b0 += 8) {
                    if (use_i8_gate_tile8_biased_input) {
                        microgemm_avx2_i8_groupwise_gate_up_tile8_biased_aligned128(
                            out, i, out_stride, gate_row, up_row,
                            gate_scales, up_scales, gate_sums, up_sums,
                            input_q, input_scales, b0, cols, groups, use_gelu
                        );
                    } else if (use_i8_gate_tile8_aligned128) {
                        microgemm_avx2_i8_groupwise_gate_up_tile8_aligned128(
                            out, i, out_stride, gate_row, up_row,
                            gate_scales, up_scales, gate_sums, up_sums,
                            input_q, input_scales, b0, cols, groups, use_gelu
                        );
                    } else if (use_i8_gate_tile8_explicit) {
                        microgemm_avx2_i8_groupwise_gate_up_tile8_explicit(
                            out, i, out_stride, gate_row, up_row,
                            gate_scales, up_scales, gate_sums, up_sums,
                            input_q, input_scales, b0, cols, groups, use_gelu
                        );
                    } else {
                        microgemm_avx2_i8_groupwise_gate_up_tile(
                            out, i, out_stride, gate_row, up_row,
                            gate_scales, up_scales, gate_sums, up_sums,
                            input_q, input_scales, b0, 8, cols, groups, use_gelu
                        );
                    }
                    microgemm_swiglu_absmax_update_rows(
                        thread_absmax, out, i, 1, out_stride, b0, 8
                    );
                }
                for (; b0 + 3 < batch; b0 += 4) {
                    microgemm_avx2_i8_groupwise_gate_up_tile(
                        out, i, out_stride, gate_row, up_row,
                        gate_scales, up_scales, gate_sums, up_sums,
                        input_q, input_scales, b0, 4, cols, groups, use_gelu
                    );
                    microgemm_swiglu_absmax_update_rows(
                        thread_absmax, out, i, 1, out_stride, b0, 4
                    );
                }
                for (; b0 + 1 < batch; b0 += 2) {
                    microgemm_avx2_i8_groupwise_gate_up_tile(
                        out, i, out_stride, gate_row, up_row,
                        gate_scales, up_scales, gate_sums, up_sums,
                        input_q, input_scales, b0, 2, cols, groups, use_gelu
                    );
                    microgemm_swiglu_absmax_update_rows(
                        thread_absmax, out, i, 1, out_stride, b0, 2
                    );
                }
                if (b0 < batch) {
                    microgemm_avx2_i8_groupwise_gate_up_tile(
                        out, i, out_stride, gate_row, up_row,
                        gate_scales, up_scales, gate_sums, up_sums,
                        input_q, input_scales, b0, 1, cols, groups, use_gelu
                    );
                    microgemm_swiglu_absmax_update_rows(
                        thread_absmax, out, i, 1, out_stride, b0, 1
                    );
                }
            }
        }
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.gate_up_dot_ms, profile_phase_start);
        }
        return;
    }
#endif

    microgemm_gemv_quantized_batched(
        config,
        out,
        weights_i8,
        weights_i4,
        scales,
        row_sums,
        input,
        batch,
        2 * intermediate,
        cols,
        input_stride,
        NULL,
        input_q,
        input_scales
    );

    #pragma omp parallel for schedule(static) if(batch >= 8)
    for (b = 0; b < batch; ++b) {
        float* gate_t = out + (size_t)b * out_stride;
        microgemm_swiglu_activation(gate_t, gate_t + intermediate, intermediate, use_gelu);
    }
}

static void microgemm_gate_up_swiglu_quantized_batched(
    const microgemm_config* config,
    float* out,
    const int8_t* weights_i8,
    const uint8_t* weights_i4,
    const float* scales,
    const int32_t* row_sums,
    const float* input,
    int batch,
    int intermediate,
    int cols,
    int input_stride,
    int out_stride,
    int use_gelu,
    int8_t* input_q,
    float* input_scales,
    float* swiglu_absmax_scratch,
    int swiglu_absmax_threads
) {
    int b;

    if (config != NULL && microgemm_quant_mode_is_groupwise(config->quant_mode)) {
        microgemm_gate_up_swiglu_groupwise_batched(
            config,
            out,
            weights_i8,
            weights_i4,
            scales,
            row_sums,
            input,
            batch,
            intermediate,
            cols,
            input_stride,
            out_stride,
            use_gelu,
            input_q,
            input_scales,
            swiglu_absmax_scratch,
            swiglu_absmax_threads
        );
        return;
    }

    (void)swiglu_absmax_scratch;
    (void)swiglu_absmax_threads;

    if (config == NULL || config->quant_mode != MICROGEMM_QUANT_INT4) {
        microgemm_gate_up_swiglu_i8_batched(
            out, weights_i8, scales, row_sums, input, batch, intermediate, cols,
            input_stride, out_stride, use_gelu, input_q, input_scales,
            swiglu_absmax_scratch, swiglu_absmax_threads
        );
        return;
    }

#if MICROGEMM_CPU_X86_AVX2
    if (row_sums != NULL && batch >= 2) {
        size_t row_bytes = (size_t)((cols + 1) / 2);
        int i;
        int profile_enabled = microgemm_decode_batch_profile_enabled;
        int use_tile8_split = microgemm_i4_gate_tile8_split_enabled_for(batch, intermediate, cols);
        int use_tile8_group4 = microgemm_i4_gate_tile8_group4_enabled_for(batch, intermediate, cols);
        int use_tile8_fused = microgemm_i4_gate_tile8_fused_enabled_for(batch, intermediate, cols);
        double profile_phase_start = 0.0;

        if (profile_enabled) {
            profile_phase_start = microgemm_profile_now_ms();
        }
        for (b = 0; b < batch; ++b) {
            input_scales[b] = microgemm_quantize_activation_for_prebiased_maddubs_dot(
                input_q + (size_t)b * cols,
                input + (size_t)b * input_stride,
                cols
            );
        }
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.gate_up_quant_ms, profile_phase_start);
            profile_phase_start = microgemm_profile_now_ms();
        }

        #pragma omp parallel for schedule(static) if(intermediate >= 96)
        for (i = 0; i < intermediate; ++i) {
            const uint8_t* gate_row = weights_i4 + (size_t)i * row_bytes;
            const uint8_t* up_row = weights_i4 + (size_t)(intermediate + i) * row_bytes;
            int b0 = 0;

            for (; b0 + 7 < batch; b0 += 8) {
                if (use_tile8_fused) {
                    microgemm_avx2_i4_gate_up_tile8_fused(
                        out,
                        i,
                        out_stride,
                        gate_row,
                        up_row,
                        scales[i],
                        scales[intermediate + i],
                        row_sums[i],
                        row_sums[intermediate + i],
                        input_q,
                        input_scales,
                        b0,
                        cols,
                        use_gelu
                    );
                } else if (use_tile8_split) {
                    microgemm_avx2_i4_gate_up_tile8_split(
                        out,
                        i,
                        out_stride,
                        gate_row,
                        up_row,
                        scales[i],
                        scales[intermediate + i],
                        row_sums[i],
                        row_sums[intermediate + i],
                        input_q,
                        input_scales,
                        b0,
                        cols,
                        use_gelu
                    );
                } else if (use_tile8_group4) {
                    microgemm_avx2_i4_gate_up_tile4(
                        out,
                        i,
                        out_stride,
                        gate_row,
                        up_row,
                        scales[i],
                        scales[intermediate + i],
                        row_sums[i],
                        row_sums[intermediate + i],
                        input_q,
                        input_scales,
                        b0,
                        4,
                        cols,
                        use_gelu
                    );
                    microgemm_avx2_i4_gate_up_tile4(
                        out,
                        i,
                        out_stride,
                        gate_row,
                        up_row,
                        scales[i],
                        scales[intermediate + i],
                        row_sums[i],
                        row_sums[intermediate + i],
                        input_q,
                        input_scales,
                        b0 + 4,
                        4,
                        cols,
                        use_gelu
                    );
                } else {
                    microgemm_avx2_i4_gate_up_tile4(
                        out,
                        i,
                        out_stride,
                        gate_row,
                        up_row,
                        scales[i],
                        scales[intermediate + i],
                        row_sums[i],
                        row_sums[intermediate + i],
                        input_q,
                        input_scales,
                        b0,
                        8,
                        cols,
                        use_gelu
                    );
                }
            }
            for (; b0 + 3 < batch; b0 += 4) {
                microgemm_avx2_i4_gate_up_tile4(
                    out,
                    i,
                    out_stride,
                    gate_row,
                    up_row,
                    scales[i],
                    scales[intermediate + i],
                    row_sums[i],
                    row_sums[intermediate + i],
                    input_q,
                    input_scales,
                    b0,
                    4,
                    cols,
                    use_gelu
                );
            }
            if (b0 < batch) {
                microgemm_avx2_i4_gate_up_tile4(
                    out,
                    i,
                    out_stride,
                    gate_row,
                    up_row,
                    scales[i],
                    scales[intermediate + i],
                    row_sums[i],
                    row_sums[intermediate + i],
                    input_q,
                    input_scales,
                    b0,
                    batch - b0,
                    cols,
                    use_gelu
                );
            }
        }
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.gate_up_dot_ms, profile_phase_start);
        }
        return;
    }
#endif

    microgemm_gemv_i4_packed_batched(
        out,
        weights_i4,
        scales,
        row_sums,
        input,
        batch,
        2 * intermediate,
        cols,
        input_stride,
        NULL,
        input_q,
        input_scales
    );

    #pragma omp parallel for schedule(static) if(batch >= 8)
    for (b = 0; b < batch; ++b) {
        float* gate_t = out + (size_t)b * out_stride;
        microgemm_swiglu_activation(gate_t, gate_t + intermediate, intermediate, use_gelu);
    }
}

static void microgemm_residual_add_batch(
    float* out,
    const float* a,
    const float* b,
    int batch,
    int width,
    float residual_multiplier
) {
    int t;
    #pragma omp parallel for schedule(static) if(batch >= 8)
    for (t = 0; t < batch; ++t) {
        microgemm_residual_add_scaled(
            out + (size_t)t * width,
            a + (size_t)t * width,
            b + (size_t)t * width,
            width,
            residual_multiplier
        );
    }
}

static void microgemm_rmsnorm_batch(
    float* out,
    const float* input,
    const float* weight,
    int batch,
    int width,
    float eps,
    int offset_weights
) {
    int t;
    #pragma omp parallel for schedule(static) if(batch >= 8)
    for (t = 0; t < batch; ++t) {
        microgemm_cpu_rmsnorm_f32(
            out + (size_t)t * width,
            input + (size_t)t * width,
            weight,
            width,
            eps,
            offset_weights
        );
    }
}

static void microgemm_quantize_activation_batch_for_i8_self_biasing(
    int8_t* input_q,
    float* input_scales,
    const float* input,
    int batch,
    int width,
    int input_stride
) {
    int t;
    #pragma omp parallel for schedule(static) if(batch >= 8)
    for (t = 0; t < batch; ++t) {
        input_scales[t] = microgemm_quantize_activation_for_i8_self_biasing_dot(
            input_q + (size_t)t * width,
            input + (size_t)t * input_stride,
            width
        );
    }
}

static float microgemm_quantize_activation_for_i8_self_biasing_dot_known_amax(
    int8_t* out,
    const float* input,
    int count,
    float amax
) {
    int i = 0;
    if (amax == 0.0f) {
        memset(out, 0, (size_t)count);
        return 1.0f;
    }
    {
        const float scale = amax / 127.0f;
        const float inv_scale = 127.0f / amax;
#if MICROGEMM_CPU_X86_AVX2
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

static void microgemm_quantize_activation_batch_for_i8_self_biasing_known_amax(
    int8_t* input_q,
    float* input_scales,
    const float* input,
    const float* input_absmax,
    int batch,
    int width,
    int input_stride
) {
    int t;
    #pragma omp parallel for schedule(static) if(batch >= 8)
    for (t = 0; t < batch; ++t) {
        input_scales[t] = microgemm_quantize_activation_for_i8_self_biasing_dot_known_amax(
            input_q + (size_t)t * width,
            input + (size_t)t * input_stride,
            width,
            input_absmax[t]
        );
    }
}

static inline void microgemm_swiglu_absmax_update_rows(
    float* absmax,
    const float* out,
    int out_col,
    int row_count,
    int out_stride,
    int batch_offset,
    int tile
) {
    int bb;
    int r;
    if (absmax == NULL) {
        return;
    }
    for (bb = 0; bb < tile; ++bb) {
        float best = absmax[batch_offset + bb];
        const float* base = out + (size_t)(batch_offset + bb) * out_stride + out_col;
        for (r = 0; r < row_count; ++r) {
            float v = base[r];
            float a = v < 0.0f ? -v : v;
            if (a > best) {
                best = a;
            }
        }
        absmax[batch_offset + bb] = best;
    }
}

static void microgemm_swiglu_absmax_reduce_threads(
    float* absmax_scratch,
    int thread_count,
    int batch
) {
    int b;
    int t;
    if (absmax_scratch == NULL || thread_count <= 1) {
        return;
    }
    for (b = 0; b < batch; ++b) {
        float best = absmax_scratch[b];
        for (t = 1; t < thread_count; ++t) {
            float v = absmax_scratch[(size_t)t * batch + b];
            if (v > best) {
                best = v;
            }
        }
        absmax_scratch[b] = best;
    }
}

static void microgemm_rmsnorm_batch_quantize_i8_self_biasing(
    float* out,
    int8_t* input_q,
    float* input_scales,
    const float* input,
    const float* weight,
    int batch,
    int width,
    float eps,
    int offset_weights
) {
    int t;
    #pragma omp parallel for schedule(static) if(batch >= 8)
    for (t = 0; t < batch; ++t) {
        float* row_out = out + (size_t)t * width;
        microgemm_cpu_rmsnorm_f32(
            row_out,
            input + (size_t)t * width,
            weight,
            width,
            eps,
            offset_weights
        );
        input_scales[t] = microgemm_quantize_activation_for_i8_self_biasing_dot(
            input_q + (size_t)t * width,
            row_out,
            width
        );
    }
}

static void microgemm_rmsnorm_heads_inplace(
    float* values,
    int heads,
    int head_dim,
    const float* weight,
    float eps,
    int offset_weights
) {
    int h;
    if (values == NULL || weight == NULL) {
        return;
    }
    for (h = 0; h < heads; ++h) {
        float* head = values + (size_t)h * head_dim;
        microgemm_cpu_rmsnorm_f32(head, head, weight, head_dim, eps, offset_weights);
    }
}

static void microgemm_qk_norm_inplace(
    float* q,
    float* k,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const float* q_norm_w,
    const float* k_norm_w,
    float eps,
    int offset_weights
) {
    microgemm_rmsnorm_heads_inplace(q, num_q_heads, head_dim, q_norm_w, eps, offset_weights);
    microgemm_rmsnorm_heads_inplace(k, num_kv_heads, head_dim, k_norm_w, eps, offset_weights);
}

static inline float microgemm_attention_scale(const microgemm_config* config, int head_dim) {
    float scalar = (float)head_dim;
    if (config != NULL && config->query_pre_attn_scalar > 0.0f) {
        if (config->architecture == MICROGEMM_ARCH_GRANITE_LIKE) {
            return config->query_pre_attn_scalar;
        }
        scalar = config->query_pre_attn_scalar;
    }
    return 1.0f / sqrtf(scalar);
}

static inline float microgemm_softcap_scalar(float value, float softcap) {
    if (softcap > 0.0f) {
        return softcap * tanhf(value / softcap);
    }
    return value;
}

static void microgemm_softcap_inplace(float* values, int count, float softcap) {
    int i;
    if (values == NULL || count <= 0 || softcap <= 0.0f) {
        return;
    }
    for (i = 0; i < count; ++i) {
        values[i] = microgemm_softcap_scalar(values[i], softcap);
    }
}

static void microgemm_softcap_batched(float* values, int batch, int width, float softcap) {
    int b;
    if (values == NULL || batch <= 0 || width <= 0 || softcap <= 0.0f) {
        return;
    }
    #pragma omp parallel for schedule(static) if(batch >= 8)
    for (b = 0; b < batch; ++b) {
        microgemm_softcap_inplace(values + (size_t)b * width, width, softcap);
    }
}

static void microgemm_logits_postprocess_inplace(
    const microgemm_config* config,
    float* values,
    int count
) {
    float logits_scaling = microgemm_config_logits_scaling(config);
    if (logits_scaling != 1.0f) {
        microgemm_scale_inplace(values, count, 1.0f / logits_scaling);
    }
    microgemm_softcap_inplace(
        values,
        count,
        config != NULL ? config->final_logit_softcap : 0.0f
    );
}

static void microgemm_logits_postprocess_batched(
    const microgemm_config* config,
    float* values,
    int batch,
    int width
) {
    int b;
    float logits_scaling = microgemm_config_logits_scaling(config);
    if (values == NULL || batch <= 0 || width <= 0) {
        return;
    }
    if (logits_scaling != 1.0f) {
        microgemm_scale_inplace(values, batch * width, 1.0f / logits_scaling);
    }
    if (config == NULL || config->final_logit_softcap <= 0.0f) {
        return;
    }
    #pragma omp parallel for schedule(static) if(batch >= 8)
    for (b = 0; b < batch; ++b) {
        microgemm_softcap_inplace(
            values + (size_t)b * width,
            width,
            config->final_logit_softcap
        );
    }
}

static inline float microgemm_sigmoid_scalar(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

static inline float microgemm_softplus_scalar(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return expf(x);
    }
    return log1pf(expf(x));
}

static void microgemm_l2_normalize_heads(float* values, int heads, int head_dim) {
    int h;
    for (h = 0; h < heads; ++h) {
        float* head = values + (size_t)h * head_dim;
        float ss = 0.0f;
        float inv;
        int i;
        for (i = 0; i < head_dim; ++i) {
            ss += head[i] * head[i];
        }
        inv = 1.0f / sqrtf(ss + 1.0e-6f);
        for (i = 0; i < head_dim; ++i) {
            head[i] *= inv;
        }
    }
}

static void microgemm_linear_delta_rule_update_head_legacy(
    const float* qh,
    const float* khv,
    const float* vh_value,
    float* state,
    float* out_head,
    int key_head_dim,
    int value_head_dim,
    float beta,
    float decay,
    float query_scale
) {
    int kd;
    int vd;

    for (kd = 0; kd < key_head_dim; ++kd) {
        for (vd = 0; vd < value_head_dim; ++vd) {
            state[(size_t)kd * (size_t)value_head_dim + (size_t)vd] *= decay;
        }
    }
    for (vd = 0; vd < value_head_dim; ++vd) {
        float kv_mem = 0.0f;
        float delta;
        float acc = 0.0f;
        for (kd = 0; kd < key_head_dim; ++kd) {
            kv_mem += state[(size_t)kd * (size_t)value_head_dim + (size_t)vd] * khv[kd];
        }
        delta = (vh_value[vd] - kv_mem) * beta;
        for (kd = 0; kd < key_head_dim; ++kd) {
            state[(size_t)kd * (size_t)value_head_dim + (size_t)vd] += khv[kd] * delta;
            acc += state[(size_t)kd * (size_t)value_head_dim + (size_t)vd] * qh[kd];
        }
        out_head[vd] = acc * query_scale;
    }
}

static void microgemm_linear_delta_rule_update_head(
    const float* qh,
    const float* khv,
    const float* vh_value,
    float* state,
    float* out_head,
    float* delta_scratch,
    int key_head_dim,
    int value_head_dim,
    float beta,
    float decay,
    float query_scale
) {
    int kd;
    int vd;

    if (!microgemm_linear_delta_vec_enabled() || delta_scratch == NULL) {
        microgemm_linear_delta_rule_update_head_legacy(
            qh,
            khv,
            vh_value,
            state,
            out_head,
            key_head_dim,
            value_head_dim,
            beta,
            decay,
            query_scale
        );
        return;
    }

    memset(out_head, 0, (size_t)value_head_dim * sizeof(float));
    for (kd = 0; kd < key_head_dim; ++kd) {
        float* row = state + (size_t)kd * (size_t)value_head_dim;
        float key = khv[kd];
        vd = 0;
#if MICROGEMM_CPU_X86_AVX2
        {
            __m256 vdecay = _mm256_set1_ps(decay);
            __m256 vkey = _mm256_set1_ps(key);
            for (; vd + 7 < value_head_dim; vd += 8) {
                __m256 r = _mm256_loadu_ps(row + vd);
                __m256 kv = _mm256_loadu_ps(out_head + vd);
                r = _mm256_mul_ps(r, vdecay);
                kv = MICROGEMM_AVX2_FMADD_PS(r, vkey, kv);
                _mm256_storeu_ps(row + vd, r);
                _mm256_storeu_ps(out_head + vd, kv);
            }
        }
#endif
        for (; vd < value_head_dim; ++vd) {
            row[vd] *= decay;
            out_head[vd] += row[vd] * key;
        }
    }

    vd = 0;
#if MICROGEMM_CPU_X86_AVX2
    {
        __m256 vbeta = _mm256_set1_ps(beta);
        __m256 vzero = _mm256_setzero_ps();
        for (; vd + 7 < value_head_dim; vd += 8) {
            __m256 vv = _mm256_loadu_ps(vh_value + vd);
            __m256 kv = _mm256_loadu_ps(out_head + vd);
            __m256 d = _mm256_mul_ps(_mm256_sub_ps(vv, kv), vbeta);
            _mm256_storeu_ps(delta_scratch + vd, d);
            _mm256_storeu_ps(out_head + vd, vzero);
        }
    }
#endif
    for (; vd < value_head_dim; ++vd) {
        delta_scratch[vd] = (vh_value[vd] - out_head[vd]) * beta;
        out_head[vd] = 0.0f;
    }

    for (kd = 0; kd < key_head_dim; ++kd) {
        float* row = state + (size_t)kd * (size_t)value_head_dim;
        float key = khv[kd];
        float query = qh[kd];
        vd = 0;
#if MICROGEMM_CPU_X86_AVX2
        {
            __m256 vkey = _mm256_set1_ps(key);
            __m256 vquery = _mm256_set1_ps(query);
            for (; vd + 7 < value_head_dim; vd += 8) {
                __m256 r = _mm256_loadu_ps(row + vd);
                __m256 d = _mm256_loadu_ps(delta_scratch + vd);
                __m256 acc = _mm256_loadu_ps(out_head + vd);
                r = MICROGEMM_AVX2_FMADD_PS(vkey, d, r);
                acc = MICROGEMM_AVX2_FMADD_PS(r, vquery, acc);
                _mm256_storeu_ps(row + vd, r);
                _mm256_storeu_ps(out_head + vd, acc);
            }
        }
#endif
        for (; vd < value_head_dim; ++vd) {
            row[vd] += key * delta_scratch[vd];
            out_head[vd] += row[vd] * query;
        }
    }

    vd = 0;
#if MICROGEMM_CPU_X86_AVX2
    {
        __m256 vscale = _mm256_set1_ps(query_scale);
        for (; vd + 7 < value_head_dim; vd += 8) {
            __m256 acc = _mm256_loadu_ps(out_head + vd);
            _mm256_storeu_ps(out_head + vd, _mm256_mul_ps(acc, vscale));
        }
    }
#endif
    for (; vd < value_head_dim; ++vd) {
        out_head[vd] *= query_scale;
    }
}

static microgemm_status microgemm_linear_attention_update_from_buffers(
    const microgemm_config* config,
    const microgemm_layer_weights_i8* layer,
    microgemm_decode_workspace* workspace,
    int layer_idx,
    int position,
    float* qkv,
    float* gate_up,
    float* attn_out
) {
    int key_heads = microgemm_linear_num_key_heads_decode(config);
    int value_heads = microgemm_linear_num_value_heads_decode(config);
    int key_head_dim = microgemm_linear_key_head_dim_decode(config);
    int value_head_dim = microgemm_linear_value_head_dim_decode(config);
    int key_dim = key_heads * key_head_dim;
    int value_dim = value_heads * value_head_dim;
    int conv_dim = 2 * key_dim + value_dim;
    int kernel = (int)config->linear_conv_kernel_dim;
    int groups;
    float* conv_state;
    float* recurrent_state;
    float* query;
    float* key;
    float* value;
    float* z;
    float* b;
    float* a;
    float query_scale;
    int c;
    int vh;

    if (workspace == NULL || qkv == NULL || gate_up == NULL || attn_out == NULL
            || workspace->linear_conv_state == NULL || workspace->linear_recurrent_state == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (key_heads <= 0 || value_heads <= 0 || key_head_dim <= 0 || value_head_dim <= 0
            || kernel <= 0 || (value_heads % key_heads) != 0) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    conv_state = workspace->linear_conv_state
        + (size_t)layer_idx * (size_t)conv_dim * (size_t)kernel;
    recurrent_state = workspace->linear_recurrent_state
        + (size_t)layer_idx * (size_t)value_heads * (size_t)key_head_dim * (size_t)value_head_dim;
    if (position == 0) {
        memset(conv_state, 0, (size_t)conv_dim * (size_t)kernel * sizeof(float));
        memset(
            recurrent_state,
            0,
            (size_t)value_heads * (size_t)key_head_dim * (size_t)value_head_dim * sizeof(float)
        );
    }

    for (c = 0; c < conv_dim; ++c) {
        float* state = conv_state + (size_t)c * (size_t)kernel;
        const float* weight = layer->linear_conv_w + (size_t)c * (size_t)kernel;
        float current = qkv[c];
        float sum = 0.0f;
        int j;
        for (j = 0; j + 1 < kernel; ++j) {
            state[j] = state[j + 1];
        }
        state[kernel - 1] = current;
        for (j = 0; j < kernel; ++j) {
            sum += state[j] * weight[j];
        }
        qkv[c] = microgemm_silu(sum);
    }

    query = qkv;
    key = qkv + (size_t)key_dim;
    value = qkv + (size_t)(2 * key_dim);
    z = gate_up;
    b = gate_up + (size_t)value_dim;
    a = gate_up + (size_t)value_dim + (size_t)value_heads;

    microgemm_l2_normalize_heads(query, key_heads, key_head_dim);
    microgemm_l2_normalize_heads(key, key_heads, key_head_dim);

    groups = value_heads / key_heads;
    query_scale = 1.0f / sqrtf((float)key_head_dim);
    for (vh = 0; vh < value_heads; ++vh) {
        int kh = vh / groups;
        const float* qh = query + (size_t)kh * (size_t)key_head_dim;
        const float* khv = key + (size_t)kh * (size_t)key_head_dim;
        const float* vh_value = value + (size_t)vh * (size_t)value_head_dim;
        float* state = recurrent_state
            + (size_t)vh * (size_t)key_head_dim * (size_t)value_head_dim;
        float* out_head = attn_out + (size_t)vh * (size_t)value_head_dim;
        float beta = microgemm_sigmoid_scalar(b[vh]);
        float gate = -expf(layer->linear_a_log[vh])
            * microgemm_softplus_scalar(a[vh] + layer->linear_dt_bias[vh]);
        float decay = expf(gate);
        microgemm_linear_delta_rule_update_head(
            qh,
            khv,
            vh_value,
            state,
            out_head,
            workspace->linear_delta,
            key_head_dim,
            value_head_dim,
            beta,
            decay,
            query_scale
        );
    }

    for (vh = 0; vh < value_heads; ++vh) {
        float* out_head = attn_out + (size_t)vh * (size_t)value_head_dim;
        const float* z_head = z + (size_t)vh * (size_t)value_head_dim;
        float ss = 0.0f;
        float inv;
        int vd;
        for (vd = 0; vd < value_head_dim; ++vd) {
            ss += out_head[vd] * out_head[vd];
        }
        inv = 1.0f / sqrtf(ss / (float)value_head_dim + config->rms_norm_eps);
        for (vd = 0; vd < value_head_dim; ++vd) {
            out_head[vd] = out_head[vd]
                * inv
                * layer->linear_norm_w[vd]
                * microgemm_silu(z_head[vd]);
        }
    }

    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_decode_linear_attention_layer(
    const microgemm_config* config,
    const microgemm_layer_weights_i8* layer,
    microgemm_decode_workspace* workspace,
    int layer_idx,
    int position
) {
    int H = (int)config->hidden_size;
    int key_heads = microgemm_linear_num_key_heads_decode(config);
    int value_heads = microgemm_linear_num_value_heads_decode(config);
    int key_head_dim = microgemm_linear_key_head_dim_decode(config);
    int value_head_dim = microgemm_linear_value_head_dim_decode(config);
    int key_dim = key_heads * key_head_dim;
    int value_dim = value_heads * value_head_dim;
    int conv_dim = 2 * key_dim + value_dim;
    int baz_rows = value_dim + 2 * value_heads;
    int kernel = (int)config->linear_conv_kernel_dim;
    int groups;
    float* conv_state;
    float* recurrent_state;
    float* query;
    float* key;
    float* value;
    float* z;
    float* b;
    float* a;
    float query_scale;
    int c;
    int vh;

    if (workspace->linear_conv_state == NULL || workspace->linear_recurrent_state == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (key_heads <= 0 || value_heads <= 0 || key_head_dim <= 0 || value_head_dim <= 0
            || kernel <= 0 || (value_heads % key_heads) != 0) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    conv_state = workspace->linear_conv_state
        + (size_t)layer_idx * (size_t)conv_dim * (size_t)kernel;
    recurrent_state = workspace->linear_recurrent_state
        + (size_t)layer_idx * (size_t)value_heads * (size_t)key_head_dim * (size_t)value_head_dim;
    if (position == 0) {
        memset(conv_state, 0, (size_t)conv_dim * (size_t)kernel * sizeof(float));
        memset(
            recurrent_state,
            0,
            (size_t)value_heads * (size_t)key_head_dim * (size_t)value_head_dim * sizeof(float)
        );
    }

    microgemm_gemv_quantized(
        config,
        workspace->qkv,
        layer->linear_qkv_w,
        layer->linear_qkv_w_i4,
        layer->linear_qkv_s,
        layer->linear_qkv_row_sums,
        workspace->normed,
        conv_dim,
        H,
        NULL,
        workspace->input_q
    );
    microgemm_gemv_quantized(
        config,
        workspace->gate_up,
        layer->linear_baz_w,
        layer->linear_baz_w_i4,
        layer->linear_baz_s,
        layer->linear_baz_row_sums,
        workspace->normed,
        baz_rows,
        H,
        NULL,
        workspace->input_q
    );

    for (c = 0; c < conv_dim; ++c) {
        float* state = conv_state + (size_t)c * (size_t)kernel;
        const float* weight = layer->linear_conv_w + (size_t)c * (size_t)kernel;
        float current = workspace->qkv[c];
        float sum = 0.0f;
        int j;
        for (j = 0; j + 1 < kernel; ++j) {
            state[j] = state[j + 1];
        }
        state[kernel - 1] = current;
        for (j = 0; j < kernel; ++j) {
            sum += state[j] * weight[j];
        }
        workspace->qkv[c] = microgemm_silu(sum);
    }

    query = workspace->qkv;
    key = workspace->qkv + (size_t)key_dim;
    value = workspace->qkv + (size_t)(2 * key_dim);
    z = workspace->gate_up;
    b = workspace->gate_up + (size_t)value_dim;
    a = workspace->gate_up + (size_t)value_dim + (size_t)value_heads;

    microgemm_l2_normalize_heads(query, key_heads, key_head_dim);
    microgemm_l2_normalize_heads(key, key_heads, key_head_dim);

    groups = value_heads / key_heads;
    query_scale = 1.0f / sqrtf((float)key_head_dim);
    for (vh = 0; vh < value_heads; ++vh) {
        int kh = vh / groups;
        const float* qh = query + (size_t)kh * (size_t)key_head_dim;
        const float* khv = key + (size_t)kh * (size_t)key_head_dim;
        const float* vh_value = value + (size_t)vh * (size_t)value_head_dim;
        float* state = recurrent_state
            + (size_t)vh * (size_t)key_head_dim * (size_t)value_head_dim;
        float* out_head = workspace->attn_out + (size_t)vh * (size_t)value_head_dim;
        float beta = microgemm_sigmoid_scalar(b[vh]);
        float gate = -expf(layer->linear_a_log[vh])
            * microgemm_softplus_scalar(a[vh] + layer->linear_dt_bias[vh]);
        float decay = expf(gate);
        microgemm_linear_delta_rule_update_head(
            qh,
            khv,
            vh_value,
            state,
            out_head,
            workspace->linear_delta,
            key_head_dim,
            value_head_dim,
            beta,
            decay,
            query_scale
        );
    }

    for (vh = 0; vh < value_heads; ++vh) {
        float* out_head = workspace->attn_out + (size_t)vh * (size_t)value_head_dim;
        const float* z_head = z + (size_t)vh * (size_t)value_head_dim;
        float ss = 0.0f;
        float inv;
        int vd;
        for (vd = 0; vd < value_head_dim; ++vd) {
            ss += out_head[vd] * out_head[vd];
        }
        inv = 1.0f / sqrtf(ss / (float)value_head_dim + config->rms_norm_eps);
        for (vd = 0; vd < value_head_dim; ++vd) {
            out_head[vd] = out_head[vd]
                * inv
                * layer->linear_norm_w[vd]
                * microgemm_silu(z_head[vd]);
        }
    }

    microgemm_gemv_quantized(
        config,
        workspace->o_out,
        layer->linear_out_w,
        layer->linear_out_w_i4,
        layer->linear_out_s,
        layer->linear_out_row_sums,
        workspace->attn_out,
        H,
        value_dim,
        NULL,
        workspace->input_q
    );
    return MICROGEMM_STATUS_OK;
}

static void microgemm_attention_decode_head(
    float* out,
    float* scores,
    const float* q,
    const float* kv_cache,
    const int* block_table,
    int seq_len,
    int qh,
    int gqa_ratio,
    int head_dim,
    int kv_block_size,
    float scale,
    float softcap,
    int stride_block,
    int stride_kv,
    int stride_head,
    int stride_pos
) {
    const float* qvec = q + (size_t)qh * head_dim;
    float* head_scores = scores + (size_t)qh * seq_len;
    int kv_h = qh / gqa_ratio;
    float* out_vec = out + (size_t)qh * head_dim;
    float max_score = -1e30f;
    float sum_exp = 0.0f;
    int pos;
    int d;

    for (pos = 0; pos < seq_len; ++pos) {
        int blk_idx = pos / kv_block_size;
        int blk_off = pos % kv_block_size;
        int phys_blk = block_table[blk_idx];
        const float* kvec = kv_cache
            + (size_t)phys_blk * stride_block
            + (size_t)kv_h * stride_head
            + (size_t)blk_off * stride_pos;
        float dot = 0.0f;
        d = 0;

#if MICROGEMM_CPU_X86_AVX2
        {
            __m256 vacc0 = _mm256_setzero_ps();
            __m256 vacc1 = _mm256_setzero_ps();
            __m256 vacc2 = _mm256_setzero_ps();
            __m256 vacc3 = _mm256_setzero_ps();
            for (; d + 31 < head_dim; d += 32) {
                __m256 vq0 = _mm256_loadu_ps(qvec + d);
                __m256 vk0 = _mm256_loadu_ps(kvec + d);
                __m256 vq1 = _mm256_loadu_ps(qvec + d + 8);
                __m256 vk1 = _mm256_loadu_ps(kvec + d + 8);
                __m256 vq2 = _mm256_loadu_ps(qvec + d + 16);
                __m256 vk2 = _mm256_loadu_ps(kvec + d + 16);
                __m256 vq3 = _mm256_loadu_ps(qvec + d + 24);
                __m256 vk3 = _mm256_loadu_ps(kvec + d + 24);
                vacc0 = MICROGEMM_AVX2_FMADD_PS(vq0, vk0, vacc0);
                vacc1 = MICROGEMM_AVX2_FMADD_PS(vq1, vk1, vacc1);
                vacc2 = MICROGEMM_AVX2_FMADD_PS(vq2, vk2, vacc2);
                vacc3 = MICROGEMM_AVX2_FMADD_PS(vq3, vk3, vacc3);
            }
            for (; d + 7 < head_dim; d += 8) {
                __m256 vq = _mm256_loadu_ps(qvec + d);
                __m256 vk = _mm256_loadu_ps(kvec + d);
                vacc0 = MICROGEMM_AVX2_FMADD_PS(vq, vk, vacc0);
            }
            {
                __m256 vsum = _mm256_add_ps(_mm256_add_ps(vacc0, vacc1), _mm256_add_ps(vacc2, vacc3));
                dot = microgemm_avx2_hsum_ps(vsum);
            }
        }
#elif MICROGEMM_CPU_ARM64_NEON
        {
            float32x4_t vacc = vdupq_n_f32(0.0f);
            for (; d + 3 < head_dim; d += 4) {
                float32x4_t vq = vld1q_f32(qvec + d);
                float32x4_t vk = vld1q_f32(kvec + d);
                vacc = vmlaq_f32(vacc, vq, vk);
            }
            dot = microgemm_neon_hsum_f32(vacc);
        }
#endif

        for (; d < head_dim; ++d) {
            dot += qvec[d] * kvec[d];
        }
        head_scores[pos] = microgemm_softcap_scalar(dot * scale, softcap);
        if (head_scores[pos] > max_score) {
            max_score = head_scores[pos];
        }
    }

    for (pos = 0; pos < seq_len; ++pos) {
        head_scores[pos] = expf(head_scores[pos] - max_score);
        sum_exp += head_scores[pos];
    }

    memset(out_vec, 0, (size_t)head_dim * sizeof(float));
    if (sum_exp == 0.0f) {
        return;
    }

    for (pos = 0; pos < seq_len; ++pos) {
        float w = head_scores[pos] / sum_exp;
        int blk_idx = pos / kv_block_size;
        int blk_off = pos % kv_block_size;
        int phys_blk = block_table[blk_idx];
        const float* vvec = kv_cache
            + (size_t)phys_blk * stride_block
            + (size_t)stride_kv
            + (size_t)kv_h * stride_head
            + (size_t)blk_off * stride_pos;
        d = 0;

#if MICROGEMM_CPU_X86_AVX2
        {
            __m256 vw = _mm256_set1_ps(w);
            for (d = 0; d + 7 < head_dim; d += 8) {
                __m256 vout = _mm256_loadu_ps(out_vec + d);
                __m256 vv = _mm256_loadu_ps(vvec + d);
                _mm256_storeu_ps(out_vec + d, MICROGEMM_AVX2_FMADD_PS(vw, vv, vout));
            }
        }
#elif MICROGEMM_CPU_ARM64_NEON
        {
            float32x4_t vw = vdupq_n_f32(w);
            for (d = 0; d + 3 < head_dim; d += 4) {
                float32x4_t vout = vld1q_f32(out_vec + d);
                float32x4_t vv = vld1q_f32(vvec + d);
                vst1q_f32(out_vec + d, vmlaq_f32(vout, vv, vw));
            }
        }
#endif
        for (; d < head_dim; ++d) {
            out_vec[d] += w * vvec[d];
        }
    }
}

#if MICROGEMM_CPU_X86_AVX2
static void microgemm_attention_decode_gqa_group_avx2(
    float* out,
    float* scores,
    const float* q,
    const float* kv_cache,
    const int* block_table,
    int seq_len,
    int kv_h,
    int group,
    int head_dim,
    int kv_block_size,
    float scale,
    float softcap,
    int stride_block,
    int stride_kv,
    int stride_head,
    int stride_pos
) {
    const int qh0 = kv_h * group;
    const float* q_heads[8];
    float* score_heads[8];
    float* out_heads[8];
    float max_score[8];
    float sum_exp[8];
    int r;
    int pos;

    if (group <= 0 || group > 8) {
        return;
    }

    for (r = 0; r < group; ++r) {
        q_heads[r] = q + (size_t)(qh0 + r) * head_dim;
        score_heads[r] = scores + (size_t)(qh0 + r) * seq_len;
        out_heads[r] = out + (size_t)(qh0 + r) * head_dim;
        max_score[r] = -1e30f;
        sum_exp[r] = 0.0f;
    }

    for (pos = 0; pos < seq_len; ++pos) {
        int blk_idx = pos / kv_block_size;
        int blk_off = pos % kv_block_size;
        int phys_blk = block_table[blk_idx];
        const float* kvec = kv_cache
            + (size_t)phys_blk * stride_block
            + (size_t)kv_h * stride_head
            + (size_t)blk_off * stride_pos;
        __m256 acc[8];
        int d = 0;

        for (r = 0; r < group; ++r) {
            acc[r] = _mm256_setzero_ps();
        }
        for (; d + 7 < head_dim; d += 8) {
            __m256 vk = _mm256_loadu_ps(kvec + d);
            for (r = 0; r < group; ++r) {
                __m256 vq = _mm256_loadu_ps(q_heads[r] + d);
                acc[r] = MICROGEMM_AVX2_FMADD_PS(vq, vk, acc[r]);
            }
        }
        for (r = 0; r < group; ++r) {
            float dot = microgemm_avx2_hsum_ps(acc[r]);
            int tail;
            for (tail = d; tail < head_dim; ++tail) {
                dot += q_heads[r][tail] * kvec[tail];
            }
            {
                float s = microgemm_softcap_scalar(dot * scale, softcap);
                score_heads[r][pos] = s;
                if (s > max_score[r]) {
                    max_score[r] = s;
                }
            }
        }
    }

    for (r = 0; r < group; ++r) {
        float sum = 0.0f;
        for (pos = 0; pos < seq_len; ++pos) {
            float e = expf(score_heads[r][pos] - max_score[r]);
            score_heads[r][pos] = e;
            sum += e;
        }
        sum_exp[r] = sum;
        memset(out_heads[r], 0, (size_t)head_dim * sizeof(float));
    }

    for (pos = 0; pos < seq_len; ++pos) {
        int blk_idx = pos / kv_block_size;
        int blk_off = pos % kv_block_size;
        int phys_blk = block_table[blk_idx];
        const float* vvec = kv_cache
            + (size_t)phys_blk * stride_block
            + (size_t)stride_kv
            + (size_t)kv_h * stride_head
            + (size_t)blk_off * stride_pos;
        float w[8];
        int d = 0;

        for (r = 0; r < group; ++r) {
            w[r] = sum_exp[r] != 0.0f ? score_heads[r][pos] / sum_exp[r] : 0.0f;
        }
        for (; d + 7 < head_dim; d += 8) {
            __m256 vv = _mm256_loadu_ps(vvec + d);
            for (r = 0; r < group; ++r) {
                __m256 vw = _mm256_set1_ps(w[r]);
                __m256 vout = _mm256_loadu_ps(out_heads[r] + d);
                vout = MICROGEMM_AVX2_FMADD_PS(vw, vv, vout);
                _mm256_storeu_ps(out_heads[r] + d, vout);
            }
        }
        for (r = 0; r < group; ++r) {
            int tail;
            for (tail = d; tail < head_dim; ++tail) {
                out_heads[r][tail] += w[r] * vvec[tail];
            }
        }
    }
}
#endif

static void microgemm_attention_decode(
    float* out,
    float* scores,
    const float* q,
    const float* kv_cache,
    const int* block_table,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int kv_block_size,
    float scale,
    float softcap,
    int stride_block,
    int stride_kv,
    int stride_head,
    int stride_pos
) {
    int qh;
    int gqa_ratio = num_q_heads / num_kv_heads;
    int parallel_work = num_q_heads * seq_len;
    int use_parallel_heads = parallel_work >= 256;

#if MICROGEMM_CPU_X86_AVX2
    if (gqa_ratio > 1 && gqa_ratio <= 8 && head_dim >= 32) {
        int kv_h;
        int use_parallel_groups = num_kv_heads * seq_len >= 256;
#ifdef _OPENMP
        if (omp_in_parallel()) {
            use_parallel_groups = 0;
        }
#endif
        #pragma omp parallel for schedule(static) if(use_parallel_groups)
        for (kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
            microgemm_attention_decode_gqa_group_avx2(
                out,
                scores,
                q,
                kv_cache,
                block_table,
                seq_len,
                kv_h,
                gqa_ratio,
                head_dim,
                kv_block_size,
                scale,
                softcap,
                stride_block,
                stride_kv,
                stride_head,
                stride_pos
            );
        }
        return;
    }
#endif

#ifdef _OPENMP
    if (omp_in_parallel()) {
        use_parallel_heads = 0;
    }
#endif

    #pragma omp parallel for schedule(static) if(use_parallel_heads)
    for (qh = 0; qh < num_q_heads; ++qh) {
        microgemm_attention_decode_head(
            out,
            scores,
            q,
            kv_cache,
            block_table,
            seq_len,
            qh,
            gqa_ratio,
            head_dim,
            kv_block_size,
            scale,
            softcap,
            stride_block,
            stride_kv,
            stride_head,
            stride_pos
        );
    }
}

#if MICROGEMM_CPU_X86_AVX2
static void microgemm_attention_decode_gqa7_head64(
    float* out,
    float* scores,
    const float* q,
    const float* kv_cache,
    const int* block_table,
    int seq_len,
    int kv_h,
    int kv_block_size,
    float scale,
    float softcap,
    int stride_block,
    int stride_kv,
    int stride_head,
    int stride_pos
) {
    const int group = 7;
    const int head_dim = 64;
    const int qh0 = kv_h * group;
    const float* q_heads[7];
    float* score_heads[7];
    float* out_heads[7];
    float max_score[7];
    float sum_exp[7];
    int r;
    int pos;

    for (r = 0; r < group; ++r) {
        q_heads[r] = q + (size_t)(qh0 + r) * head_dim;
        score_heads[r] = scores + (size_t)(qh0 + r) * seq_len;
        out_heads[r] = out + (size_t)(qh0 + r) * head_dim;
        max_score[r] = -1e30f;
        sum_exp[r] = 0.0f;
    }

    for (pos = 0; pos < seq_len; ++pos) {
        int blk_idx = pos / kv_block_size;
        int blk_off = pos % kv_block_size;
        int phys_blk = block_table[blk_idx];
        const float* kvec = kv_cache
            + (size_t)phys_blk * stride_block
            + (size_t)kv_h * stride_head
            + (size_t)blk_off * stride_pos;
        __m256 acc[7];
        int d;

        for (r = 0; r < group; ++r) {
            acc[r] = _mm256_setzero_ps();
        }
        for (d = 0; d < head_dim; d += 8) {
            __m256 vk = _mm256_loadu_ps(kvec + d);
            for (r = 0; r < group; ++r) {
                __m256 vq = _mm256_loadu_ps(q_heads[r] + d);
                acc[r] = MICROGEMM_AVX2_FMADD_PS(vq, vk, acc[r]);
            }
        }
        for (r = 0; r < group; ++r) {
            float s = microgemm_softcap_scalar(microgemm_avx2_hsum_ps(acc[r]) * scale, softcap);
            score_heads[r][pos] = s;
            if (s > max_score[r]) {
                max_score[r] = s;
            }
        }
    }

    for (r = 0; r < group; ++r) {
        float sum = 0.0f;
        for (pos = 0; pos < seq_len; ++pos) {
            float e = expf(score_heads[r][pos] - max_score[r]);
            score_heads[r][pos] = e;
            sum += e;
        }
        sum_exp[r] = sum;
        memset(out_heads[r], 0, (size_t)head_dim * sizeof(float));
    }

    for (pos = 0; pos < seq_len; ++pos) {
        int blk_idx = pos / kv_block_size;
        int blk_off = pos % kv_block_size;
        int phys_blk = block_table[blk_idx];
        const float* vvec = kv_cache
            + (size_t)phys_blk * stride_block
            + (size_t)stride_kv
            + (size_t)kv_h * stride_head
            + (size_t)blk_off * stride_pos;
        float w[7];
        int d;

        for (r = 0; r < group; ++r) {
            w[r] = sum_exp[r] != 0.0f ? score_heads[r][pos] / sum_exp[r] : 0.0f;
        }
        for (d = 0; d < head_dim; d += 8) {
            __m256 vv = _mm256_loadu_ps(vvec + d);
            for (r = 0; r < group; ++r) {
                __m256 vw = _mm256_set1_ps(w[r]);
                __m256 vout = _mm256_loadu_ps(out_heads[r] + d);
                vout = MICROGEMM_AVX2_FMADD_PS(vw, vv, vout);
                _mm256_storeu_ps(out_heads[r] + d, vout);
            }
        }
    }
}
#endif

static void microgemm_attention_decode_batch(
    float* out,
    microgemm_decode_workspace* const* workspaces,
    const float* q,
    const microgemm_kv_layout* const* kvs,
    int batch,
    int layer_idx,
    int q_stride,
    int out_stride,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int kv_block_size,
    float scale,
    float softcap
) {
    int work;
    int gqa_ratio = num_q_heads / num_kv_heads;
    int total_heads = batch * num_q_heads;

#if MICROGEMM_CPU_X86_AVX2
    if (gqa_ratio == 7 && head_dim == 64) {
        int total_kv_heads = batch * num_kv_heads;

        #pragma omp parallel for schedule(static) if(total_kv_heads >= 2)
        for (work = 0; work < total_kv_heads; ++work) {
            int t = work / num_kv_heads;
            int kv_h = work - t * num_kv_heads;
            const microgemm_kv_layout* kv = kvs[t];
            const float* q_t = q + (size_t)t * q_stride;
            float* out_t = out + (size_t)t * out_stride;
            float* layer_kv = kv->layer_kv[layer_idx];

            microgemm_attention_decode_gqa7_head64(
                out_t,
                workspaces[t]->scores,
                q_t,
                layer_kv,
                kv->block_table,
                kv->seq_len + 1,
                kv_h,
                kv_block_size,
                scale,
                softcap,
                kv->stride_block,
                kv->stride_kv,
                kv->stride_head,
                kv->stride_pos
            );
        }
        return;
    }
    if (gqa_ratio > 1 && gqa_ratio <= 8 && head_dim >= 32) {
        int total_kv_heads = batch * num_kv_heads;

        #pragma omp parallel for schedule(static) if(total_kv_heads >= 2)
        for (work = 0; work < total_kv_heads; ++work) {
            int t = work / num_kv_heads;
            int kv_h = work - t * num_kv_heads;
            const microgemm_kv_layout* kv = kvs[t];
            const float* q_t = q + (size_t)t * q_stride;
            float* out_t = out + (size_t)t * out_stride;
            float* layer_kv = kv->layer_kv[layer_idx];

            microgemm_attention_decode_gqa_group_avx2(
                out_t,
                workspaces[t]->scores,
                q_t,
                layer_kv,
                kv->block_table,
                kv->seq_len + 1,
                kv_h,
                gqa_ratio,
                head_dim,
                kv_block_size,
                scale,
                softcap,
                kv->stride_block,
                kv->stride_kv,
                kv->stride_head,
                kv->stride_pos
            );
        }
        return;
    }
#endif

    #pragma omp parallel for schedule(static) if(total_heads >= 2)
    for (work = 0; work < total_heads; ++work) {
        int t = work / num_q_heads;
        int qh = work - t * num_q_heads;
        const microgemm_kv_layout* kv = kvs[t];
        const float* q_t = q + (size_t)t * q_stride;
        float* out_t = out + (size_t)t * out_stride;
        float* layer_kv = kv->layer_kv[layer_idx];

        microgemm_attention_decode_head(
            out_t,
            workspaces[t]->scores,
            q_t,
            layer_kv,
            kv->block_table,
            kv->seq_len + 1,
            qh,
            gqa_ratio,
            head_dim,
            kv_block_size,
            scale,
            softcap,
            kv->stride_block,
            kv->stride_kv,
            kv->stride_head,
            kv->stride_pos
        );
    }
}

static int microgemm_argmax(const float* values, int count) {
    int best_index = 0;
    float best_value = values[0];
    int i;
    for (i = 1; i < count; ++i) {
        if (values[i] > best_value) {
            best_value = values[i];
            best_index = i;
        }
    }
    return best_index;
}

microgemm_status microgemm_decode_workspace_create(
    const microgemm_config* config,
    uint32_t max_seq_len,
    microgemm_decode_workspace** out_workspace
) {
    microgemm_decode_workspace* ws;
    size_t qkv_size;
    size_t attn_out_size;
    size_t linear_conv_state_size = 0u;
    size_t linear_recurrent_state_size = 0u;
    size_t linear_delta_size = 0u;
    size_t max_input_q;

    if (config == NULL || out_workspace == NULL || max_seq_len == 0) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (!microgemm_decode_config_is_valid(config)) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    ws = (microgemm_decode_workspace*)calloc(1, sizeof(*ws));
    if (ws == NULL) {
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }

    ws->config = *config;
    ws->max_seq_len = max_seq_len;
    ws->scratch_bytes = microgemm_decode_compute_scratch_bytes(config, max_seq_len);

    qkv_size = (size_t)(config->qkv_rows != 0u
        ? config->qkv_rows
        : (config->num_q_heads + 2U * config->num_kv_heads) * config->head_dim);
    attn_out_size = (size_t)(config->attn_width != 0u
        ? config->attn_width
        : config->num_q_heads * config->head_dim);
    if ((size_t)microgemm_linear_value_dim_decode(config) > attn_out_size) {
        attn_out_size = (size_t)microgemm_linear_value_dim_decode(config);
    }
    if (config->architecture == MICROGEMM_ARCH_QWEN35_LIKE) {
        linear_conv_state_size =
            (size_t)config->num_layers
            * (size_t)microgemm_linear_conv_dim_decode(config)
            * (size_t)config->linear_conv_kernel_dim;
        linear_recurrent_state_size =
            (size_t)config->num_layers
            * (size_t)microgemm_linear_num_value_heads_decode(config)
            * (size_t)microgemm_linear_key_head_dim_decode(config)
            * (size_t)microgemm_linear_value_head_dim_decode(config);
        linear_delta_size = (size_t)microgemm_linear_value_head_dim_decode(config);
    }
    max_input_q = config->hidden_size;
    if ((size_t)(2U * config->intermediate_size) > max_input_q) {
        max_input_q = (size_t)(2U * config->intermediate_size);
    }
    if (attn_out_size > max_input_q) {
        max_input_q = attn_out_size;
    }
    if ((size_t)microgemm_linear_conv_dim_decode(config) > max_input_q) {
        max_input_q = (size_t)microgemm_linear_conv_dim_decode(config);
    }
    if ((size_t)microgemm_linear_baz_rows_decode(config) > max_input_q) {
        max_input_q = (size_t)microgemm_linear_baz_rows_decode(config);
    }

    ws->hidden = (float*)malloc((size_t)config->hidden_size * sizeof(float));
    ws->residual = (float*)malloc((size_t)config->hidden_size * sizeof(float));
    ws->normed = (float*)malloc((size_t)config->hidden_size * sizeof(float));
    ws->qkv = (float*)malloc(qkv_size * sizeof(float));
    ws->attn_out = (float*)malloc(attn_out_size * sizeof(float));
    ws->o_out = (float*)malloc((size_t)config->hidden_size * sizeof(float));
    ws->gate_up = (float*)malloc((size_t)(2U * config->intermediate_size) * sizeof(float));
    ws->mlp_out = (float*)malloc((size_t)config->hidden_size * sizeof(float));
    ws->logits = (float*)malloc((size_t)config->vocab_size * sizeof(float));
    ws->scores = (float*)malloc((size_t)config->num_q_heads * max_seq_len * sizeof(float));
    if (linear_conv_state_size != 0u) {
        ws->linear_conv_state = (float*)calloc(linear_conv_state_size, sizeof(float));
    }
    if (linear_recurrent_state_size != 0u) {
        ws->linear_recurrent_state = (float*)calloc(linear_recurrent_state_size, sizeof(float));
    }
    if (linear_delta_size != 0u) {
        ws->linear_delta = (float*)malloc(linear_delta_size * sizeof(float));
    }
    ws->input_q = (int8_t*)malloc(max_input_q * sizeof(int8_t));

    if (ws->hidden == NULL || ws->residual == NULL || ws->normed == NULL
            || ws->qkv == NULL || ws->attn_out == NULL || ws->o_out == NULL
            || ws->gate_up == NULL || ws->mlp_out == NULL || ws->logits == NULL
            || ws->scores == NULL || ws->input_q == NULL
            || (linear_conv_state_size != 0u && ws->linear_conv_state == NULL)
            || (linear_recurrent_state_size != 0u && ws->linear_recurrent_state == NULL)
            || (linear_delta_size != 0u && ws->linear_delta == NULL)) {
        microgemm_decode_workspace_clear(ws);
        free(ws);
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }

    *out_workspace = ws;
    return MICROGEMM_STATUS_OK;
}

void microgemm_decode_workspace_destroy(microgemm_decode_workspace* workspace) {
    if (workspace == NULL) {
        return;
    }
    microgemm_decode_workspace_clear(workspace);
    free(workspace);
}

size_t microgemm_decode_workspace_bytes(const microgemm_decode_workspace* workspace) {
    return workspace ? workspace->scratch_bytes : 0;
}

static int microgemm_model_has_embedding(const microgemm_model_weights_i8* model) {
    return model != NULL
        && (model->embed_tokens != NULL
            || (model->embed_tokens_i8 != NULL && model->embed_tokens_s != NULL)
            || (model->embed_tokens_i4 != NULL && model->embed_tokens_s != NULL));
}

static int microgemm_has_quantized_weight(
    const microgemm_config* config,
    const int8_t* weights_i8,
    const uint8_t* weights_i4
) {
    if (config != NULL && microgemm_quant_mode_is_i4_storage(config->quant_mode)) {
        return weights_i4 != NULL;
    }
    return weights_i8 != NULL;
}

static int microgemm_model_has_lm_head(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model
) {
    return model != NULL
        && model->lm_head_s != NULL
        && microgemm_has_quantized_weight(config, model->lm_head_w, model->lm_head_w_i4);
}

static int microgemm_layer_has_quantized_weights(
    const microgemm_config* config,
    const microgemm_layer_weights_i8* layer
) {
    if (layer == NULL
            || layer->input_norm_w == NULL
            || layer->post_attn_norm_w == NULL
            || layer->gate_up_s == NULL
            || layer->down_s == NULL
            || !microgemm_has_quantized_weight(config, layer->gate_up_w, layer->gate_up_w_i4)
            || !microgemm_has_quantized_weight(config, layer->down_w, layer->down_w_i4)) {
        return 0;
    }
    if (layer->layer_type == MICROGEMM_LAYER_LINEAR_ATTENTION) {
        return layer->linear_qkv_s != NULL
            && layer->linear_qkv_row_sums != NULL
            && layer->linear_baz_s != NULL
            && layer->linear_baz_row_sums != NULL
            && layer->linear_conv_w != NULL
            && layer->linear_dt_bias != NULL
            && layer->linear_a_log != NULL
            && layer->linear_norm_w != NULL
            && layer->linear_out_s != NULL
            && layer->linear_out_row_sums != NULL
            && microgemm_has_quantized_weight(config, layer->linear_qkv_w, layer->linear_qkv_w_i4)
            && microgemm_has_quantized_weight(config, layer->linear_baz_w, layer->linear_baz_w_i4)
            && microgemm_has_quantized_weight(config, layer->linear_out_w, layer->linear_out_w_i4);
    }
    return layer->qkv_s != NULL
        && layer->o_s != NULL
        && microgemm_has_quantized_weight(config, layer->qkv_w, layer->qkv_w_i4)
        && microgemm_has_quantized_weight(config, layer->o_w, layer->o_w_i4);
}

static void microgemm_decode_load_embedding_row(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    int token_id,
    int hidden_size,
    float* out
) {
    if (model->embed_tokens != NULL) {
        memcpy(
            out,
            model->embed_tokens + (size_t)token_id * hidden_size,
            (size_t)hidden_size * sizeof(float)
        );
    } else if (model->embed_tokens_i8 != NULL) {
        const int8_t* row = model->embed_tokens_i8 + (size_t)token_id * hidden_size;
        int i;
        if (config != NULL && microgemm_quant_mode_is_groupwise(config->quant_mode)) {
            int groups = microgemm_quant_group_count_int(hidden_size);
            const float* scales = model->embed_tokens_s + (size_t)token_id * groups;
            for (i = 0; i < hidden_size; ++i) {
                out[i] = (float)row[i]
                    * scales[i / (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE];
            }
        } else {
            float scale = model->embed_tokens_s[token_id];
            for (i = 0; i < hidden_size; ++i) {
                out[i] = (float)row[i] * scale;
            }
        }
    } else {
        const uint8_t* row = model->embed_tokens_i4
            + (size_t)token_id * (size_t)((hidden_size + 1) / 2);
        int i;
        if (config != NULL && microgemm_quant_mode_is_groupwise(config->quant_mode)) {
            int groups = microgemm_quant_group_count_int(hidden_size);
            const float* scales = model->embed_tokens_s + (size_t)token_id * groups;
            for (i = 0; i < hidden_size; ++i) {
                out[i] = (float)microgemm_i4_row_value(row, i)
                    * scales[i / (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE];
            }
        } else {
            float scale = model->embed_tokens_s[token_id];
            for (i = 0; i < hidden_size; ++i) {
                out[i] = (float)microgemm_i4_row_value(row, i) * scale;
            }
        }
    }
    microgemm_scale_inplace(out, hidden_size, microgemm_config_embedding_multiplier(config));
}

static microgemm_status microgemm_decode_validate_common_inputs(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    const microgemm_decode_workspace* workspace,
    const microgemm_kv_layout* kv,
    float* logits_out,
    int logits_capacity,
    int wants_token_output
) {
    if (config == NULL || model == NULL || workspace == NULL || kv == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (!microgemm_decode_config_is_valid(config)) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (!microgemm_quant_mode_is_i8_storage(config->quant_mode)
            && !microgemm_quant_mode_is_i4_storage(config->quant_mode)) {
        return MICROGEMM_STATUS_UNSUPPORTED;
    }
    if (workspace->config.hidden_size != config->hidden_size
            || workspace->config.num_layers != config->num_layers
            || workspace->config.vocab_size != config->vocab_size) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (kv->seq_len < 0 || kv->seq_len >= (int)workspace->max_seq_len) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (!microgemm_model_has_embedding(model) || model->layers == NULL || model->final_norm_w == NULL
            || model->cos_cache == NULL || model->sin_cache == NULL
            || kv->layer_kv == NULL || kv->block_table == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (logits_out != NULL && logits_capacity < (int)config->vocab_size) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if ((logits_out != NULL || wants_token_output)
            && !microgemm_model_has_lm_head(config, model)) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_decode_validate_token_position(
    const microgemm_config* config,
    const microgemm_decode_workspace* workspace,
    int token_id,
    int position
) {
    if (token_id < 0 || token_id >= (int)config->vocab_size) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (position < 0 || position >= (int)config->max_position_embeddings
            || position >= (int)workspace->max_seq_len) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_decode_step_i8_impl(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* workspace,
    int token_id,
    int position,
    const microgemm_kv_layout* kv,
    float* logits_out,
    int logits_capacity,
    int* out_token_id
) {
    int H;
    int I;
    int Nq;
    int Nkv;
    int D;
    int qkv_size;
    int full_q_rows;
    int attn_width;
    int rotary_dim;
    int layer_idx;
    float* logits_target;
    const float* cos_v;
    const float* sin_v;
    float scale;
    (void)logits_capacity;

    H = (int)config->hidden_size;
    I = (int)config->intermediate_size;
    Nq = (int)config->num_q_heads;
    Nkv = (int)config->num_kv_heads;
    D = (int)config->head_dim;
    full_q_rows = microgemm_full_q_rows_decode(config);
    attn_width = microgemm_full_attn_width(config);
    qkv_size = microgemm_full_qkv_rows_decode(config);
    rotary_dim = (int)(config->rotary_dim != 0u ? config->rotary_dim : config->head_dim);
    scale = microgemm_attention_scale(config, D);
    logits_target = logits_out ? logits_out : workspace->logits;

    microgemm_decode_load_embedding_row(config, model, token_id, H, workspace->hidden);

    cos_v = model->cos_cache + (size_t)position * (D / 2);
    sin_v = model->sin_cache + (size_t)position * (D / 2);

    for (layer_idx = 0; layer_idx < (int)config->num_layers; ++layer_idx) {
        const microgemm_layer_weights_i8* layer = &model->layers[layer_idx];
        float* layer_kv;
        float* q;
        float* k;
        float* v;
        int phys_blk;
        int blk_idx;
        int blk_off;
        int h;

        if (!microgemm_layer_has_quantized_weights(config, layer)) {
            return MICROGEMM_STATUS_INVALID_ARGUMENT;
        }

        memcpy(workspace->residual, workspace->hidden, (size_t)H * sizeof(float));
        microgemm_cpu_rmsnorm_f32(
            workspace->normed,
            workspace->hidden,
            layer->input_norm_w,
            H,
            config->rms_norm_eps,
            (config->flags & MICROGEMM_FLAG_NORM_OFFSET) != 0
        );

        if (layer->layer_type == MICROGEMM_LAYER_LINEAR_ATTENTION) {
            microgemm_status linear_status = microgemm_decode_linear_attention_layer(
                config,
                layer,
                workspace,
                layer_idx,
                position
            );
            if (linear_status != MICROGEMM_STATUS_OK) {
                return linear_status;
            }
        } else {
            float* q_gate = NULL;
            int gate_i;

            microgemm_gemv_quantized(
                config,
                workspace->qkv,
                layer->qkv_w,
                layer->qkv_w_i4,
                layer->qkv_s,
                layer->qkv_row_sums,
                workspace->normed,
                qkv_size,
                H,
                (config->flags & MICROGEMM_FLAG_QKV_BIAS) ? layer->qkv_bias : NULL,
                workspace->input_q
            );

            q = workspace->qkv;
            if ((config->flags & MICROGEMM_FLAG_ATTN_OUTPUT_GATE) != 0u) {
                q_gate = workspace->qkv + (size_t)attn_width;
            }
            k = workspace->qkv + (size_t)full_q_rows;
            v = k + (size_t)Nkv * D;

            if ((config->flags & MICROGEMM_FLAG_QK_NORM) != 0u) {
                microgemm_qk_norm_inplace(
                    q,
                    k,
                    Nq,
                    Nkv,
                    D,
                    layer->q_norm_w,
                    layer->k_norm_w,
                    config->rms_norm_eps,
                    (config->flags & MICROGEMM_FLAG_NORM_OFFSET) != 0
                );
            }
            microgemm_rope_rotate_partial(
                q,
                k,
                cos_v,
                sin_v,
                Nq,
                Nkv,
                D,
                rotary_dim,
                (config->flags & MICROGEMM_FLAG_ROPE_INTERLEAVED) != 0u
            );

            layer_kv = kv->layer_kv[layer_idx];
            if (layer_kv == NULL) {
                return MICROGEMM_STATUS_INVALID_ARGUMENT;
            }

            blk_idx = kv->seq_len / (int)config->kv_block_size;
            blk_off = kv->seq_len % (int)config->kv_block_size;
            phys_blk = kv->block_table[blk_idx];

            for (h = 0; h < Nkv; ++h) {
                float* k_dst = layer_kv
                    + (size_t)phys_blk * kv->stride_block
                    + (size_t)h * kv->stride_head
                    + (size_t)blk_off * kv->stride_pos;
                float* v_dst = layer_kv
                    + (size_t)phys_blk * kv->stride_block
                    + (size_t)kv->stride_kv
                    + (size_t)h * kv->stride_head
                    + (size_t)blk_off * kv->stride_pos;
                memcpy(k_dst, k + (size_t)h * D, (size_t)D * sizeof(float));
                memcpy(v_dst, v + (size_t)h * D, (size_t)D * sizeof(float));
            }

            microgemm_attention_decode(
                workspace->attn_out,
                workspace->scores,
                q,
                layer_kv,
                kv->block_table,
                kv->seq_len + 1,
                Nq,
                Nkv,
                D,
                (int)config->kv_block_size,
                scale,
                config->attention_logit_softcap,
                kv->stride_block,
                kv->stride_kv,
                kv->stride_head,
                kv->stride_pos
            );
            if (q_gate != NULL) {
                for (gate_i = 0; gate_i < attn_width; ++gate_i) {
                    workspace->attn_out[gate_i] *= microgemm_sigmoid_scalar(q_gate[gate_i]);
                }
            }

            microgemm_gemv_quantized(
                config,
                workspace->o_out,
                layer->o_w,
                layer->o_w_i4,
                layer->o_s,
                layer->o_row_sums,
                workspace->attn_out,
                H,
                attn_width,
                NULL,
                workspace->input_q
            );
            if ((config->flags & MICROGEMM_FLAG_ATTN_OUTPUT_NORM) != 0u) {
                microgemm_cpu_rmsnorm_f32(
                    workspace->o_out,
                    workspace->o_out,
                    layer->attn_output_norm_w,
                    H,
                    config->rms_norm_eps,
                    (config->flags & MICROGEMM_FLAG_NORM_OFFSET) != 0
                );
            }
        }
        microgemm_residual_add_scaled(
            workspace->hidden,
            workspace->residual,
            workspace->o_out,
            H,
            microgemm_config_residual_multiplier(config)
        );

        memcpy(workspace->residual, workspace->hidden, (size_t)H * sizeof(float));
        microgemm_cpu_rmsnorm_f32(
            workspace->normed,
            workspace->hidden,
            layer->post_attn_norm_w,
            H,
            config->rms_norm_eps,
            (config->flags & MICROGEMM_FLAG_NORM_OFFSET) != 0
        );

        microgemm_gemv_quantized(
            config,
            workspace->gate_up,
            layer->gate_up_w,
            layer->gate_up_w_i4,
            layer->gate_up_s,
            layer->gate_up_row_sums,
            workspace->normed,
            2 * I,
            H,
            NULL,
            workspace->input_q
        );
        microgemm_swiglu_activation(
            workspace->gate_up,
            workspace->gate_up + I,
            I,
            (config->flags & MICROGEMM_FLAG_MLP_GELU) != 0
        );

        microgemm_gemv_quantized(
            config,
            workspace->mlp_out,
            layer->down_w,
            layer->down_w_i4,
            layer->down_s,
            layer->down_row_sums,
            workspace->gate_up,
            H,
            I,
            NULL,
            workspace->input_q
        );
        if ((config->flags & MICROGEMM_FLAG_MLP_OUTPUT_NORM) != 0u) {
            microgemm_cpu_rmsnorm_f32(
                workspace->mlp_out,
                workspace->mlp_out,
                layer->mlp_output_norm_w,
                H,
                config->rms_norm_eps,
                (config->flags & MICROGEMM_FLAG_NORM_OFFSET) != 0
            );
        }
        microgemm_residual_add_scaled(
            workspace->hidden,
            workspace->residual,
            workspace->mlp_out,
            H,
            microgemm_config_residual_multiplier(config)
        );
    }

    microgemm_cpu_rmsnorm_f32(
        workspace->normed,
        workspace->hidden,
        model->final_norm_w,
        H,
        config->rms_norm_eps,
        (config->flags & MICROGEMM_FLAG_NORM_OFFSET) != 0
    );

    if (logits_out != NULL || out_token_id != NULL) {
        microgemm_gemv_quantized(
            config,
            logits_target,
            model->lm_head_w,
            model->lm_head_w_i4,
            model->lm_head_s,
            model->lm_head_row_sums,
            workspace->normed,
            (int)config->vocab_size,
            H,
            NULL,
            workspace->input_q
        );
        if (logits_out != NULL) {
            microgemm_logits_postprocess_inplace(config, logits_target, (int)config->vocab_size);
        }
    }

    if (out_token_id != NULL) {
        *out_token_id = microgemm_argmax(logits_target, (int)config->vocab_size);
    }
    return MICROGEMM_STATUS_OK;
}

microgemm_status microgemm_decode_step_i8(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* workspace,
    int token_id,
    int position,
    const microgemm_kv_layout* kv,
    float* logits_out,
    int logits_capacity,
    int* out_token_id
) {
    microgemm_status status = microgemm_decode_validate_common_inputs(
        config,
        model,
        workspace,
        kv,
        logits_out,
        logits_capacity,
        out_token_id != NULL
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }

    status = microgemm_decode_validate_token_position(config, workspace, token_id, position);
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }

    return microgemm_decode_step_i8_impl(
        config,
        model,
        workspace,
        token_id,
        position,
        kv,
        logits_out,
        logits_capacity,
        out_token_id
    );
}

static microgemm_status microgemm_decode_step_i8_batch_impl(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* const* workspaces,
    const int* token_ids,
    const int* positions,
    const microgemm_kv_layout* const* kvs,
    size_t batch,
    float* const* logits_out,
    int logits_capacity,
    int* out_token_ids
) {
    int H;
    int I;
    int Nq;
    int Nkv;
    int D;
    int V;
    int qkv_size;
    int full_q_rows;
    int attn_width;
    int attn_storage_width;
    int linear_conv_dim;
    int linear_value_dim;
    int linear_baz_rows;
    int rotary_dim;
    int has_linear_layers;
    int max_input_cols;
    int batch_i;
    int layer_idx;
    int t;
    int offset_weights;
    int use_gelu;
    int outputs_are_contiguous = 1;
    float scale;
    float* hidden = NULL;
    float* residual = NULL;
    float* normed = NULL;
    float* qkv = NULL;
    float* attn_out = NULL;
    float* tmp_h = NULL;
    float* gate_up = NULL;
    float* logits_batch = NULL;
    float* logits_target = NULL;
    float* input_scales = NULL;
    float* swiglu_absmax_scratch = NULL;
    int8_t* input_q = NULL;
    size_t b;
    size_t hidden_elems;
    size_t qkv_elems;
    size_t attn_elems;
    size_t gate_up_elems;
    size_t max_input_elems;
    size_t swiglu_absmax_elems;
    size_t logits_elems = 0u;
    microgemm_status status = MICROGEMM_STATUS_OK;
    int profile_enabled = 0;
    int swiglu_down_prequant_requested = 0;
    int swiglu_absmax_threads = 1;
    double profile_total_start = 0.0;
    double profile_phase_start = 0.0;

    if (!microgemm_decode_config_is_valid(config) || model == NULL
            || workspaces == NULL || token_ids == NULL || positions == NULL
            || kvs == NULL || (logits_out == NULL && out_token_ids == NULL) || batch == 0u) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (!microgemm_quant_mode_is_i8_storage(config->quant_mode)
            && !microgemm_quant_mode_is_i4_storage(config->quant_mode)) {
        return MICROGEMM_STATUS_UNSUPPORTED;
    }
    if (!microgemm_model_has_embedding(model) || model->layers == NULL || model->final_norm_w == NULL
            || !microgemm_model_has_lm_head(config, model)
            || model->cos_cache == NULL || model->sin_cache == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (batch > (size_t)INT_MAX) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    H = (int)config->hidden_size;
    I = (int)config->intermediate_size;
    Nq = (int)config->num_q_heads;
    Nkv = (int)config->num_kv_heads;
    D = (int)config->head_dim;
    V = (int)config->vocab_size;
    qkv_size = (int)(config->qkv_rows != 0u
        ? config->qkv_rows
        : microgemm_full_qkv_rows_decode(config));
    full_q_rows = microgemm_full_q_rows_decode(config);
    attn_width = microgemm_full_attn_width(config);
    has_linear_layers = microgemm_has_linear_attention_layers(config, model);
    linear_conv_dim = has_linear_layers ? microgemm_linear_conv_dim_decode(config) : 0;
    linear_value_dim = has_linear_layers ? microgemm_linear_value_dim_decode(config) : 0;
    linear_baz_rows = has_linear_layers ? microgemm_linear_baz_rows_decode(config) : 0;
    attn_storage_width = attn_width;
    if (linear_value_dim > attn_storage_width) {
        attn_storage_width = linear_value_dim;
    }
    rotary_dim = (int)(config->rotary_dim != 0u ? config->rotary_dim : config->head_dim);
    max_input_cols = H;
    batch_i = (int)batch;
    offset_weights = (config->flags & MICROGEMM_FLAG_NORM_OFFSET) != 0;
    use_gelu = (config->flags & MICROGEMM_FLAG_MLP_GELU) != 0;
    scale = microgemm_attention_scale(config, D);
    profile_enabled = microgemm_decode_batch_profile_enabled && batch > 1u;
    if (profile_enabled) {
        profile_total_start = microgemm_profile_now_ms();
        microgemm_decode_batch_profile_state.calls += 1u;
        microgemm_decode_batch_profile_state.tokens += (uint64_t)batch;
    }

    if (logits_out != NULL && logits_capacity < V) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (attn_storage_width > max_input_cols) {
        max_input_cols = attn_storage_width;
    }
    if (I > max_input_cols) {
        max_input_cols = I;
    }
    if (linear_conv_dim > max_input_cols) {
        max_input_cols = linear_conv_dim;
    }
    if (linear_baz_rows > max_input_cols) {
        max_input_cols = linear_baz_rows;
    }

    for (b = 0u; b < batch; ++b) {
        if (workspaces[b] == NULL || kvs[b] == NULL
                || (logits_out != NULL && logits_out[b] == NULL)) {
            return MICROGEMM_STATUS_INVALID_ARGUMENT;
        }
        if (workspaces[b]->config.hidden_size != config->hidden_size
                || workspaces[b]->config.num_layers != config->num_layers
                || workspaces[b]->config.vocab_size != config->vocab_size) {
            return MICROGEMM_STATUS_INVALID_ARGUMENT;
        }
        if (kvs[b]->seq_len < 0 || kvs[b]->seq_len >= (int)workspaces[b]->max_seq_len
                || kvs[b]->layer_kv == NULL || kvs[b]->block_table == NULL) {
            return MICROGEMM_STATUS_INVALID_ARGUMENT;
        }
        status = microgemm_decode_validate_token_position(
            config,
            workspaces[b],
            token_ids[b],
            positions[b]
        );
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
        if (positions[b] != kvs[b]->seq_len) {
            return MICROGEMM_STATUS_INVALID_ARGUMENT;
        }
        if (logits_out != NULL && b > 0u) {
            uintptr_t expected =
                (uintptr_t)logits_out[0] + b * (size_t)V * sizeof(float);
            if ((uintptr_t)logits_out[b] != expected) {
                outputs_are_contiguous = 0;
            }
        }
    }

    if (batch == 1u) {
        return microgemm_decode_step_i8(
            config,
            model,
            workspaces[0],
            token_ids[0],
            positions[0],
            kvs[0],
            logits_out != NULL ? logits_out[0] : NULL,
            logits_out != NULL ? logits_capacity : 0,
            out_token_ids != NULL ? &out_token_ids[0] : NULL
        );
    }

    if (batch > ((size_t)-1) / (size_t)H
            || batch > ((size_t)-1) / (size_t)qkv_size
            || batch > ((size_t)-1) / (size_t)attn_storage_width
            || batch > ((size_t)-1) / (size_t)(2 * I)
            || batch > ((size_t)-1) / (size_t)max_input_cols
            || batch > ((size_t)-1) / (size_t)V) {
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }
    swiglu_down_prequant_requested = microgemm_swiglu_down_prequant_enabled_for(
        config->quant_mode, batch_i, I
    );
    if (swiglu_down_prequant_requested) {
        swiglu_absmax_threads = microgemm_max_worker_threads();
        if ((size_t)swiglu_absmax_threads > ((size_t)-1) / batch) {
            return MICROGEMM_STATUS_OUT_OF_MEMORY;
        }
        swiglu_absmax_elems = (size_t)swiglu_absmax_threads * batch;
    } else {
        swiglu_absmax_elems = 0u;
    }

    hidden_elems = batch * (size_t)H;
    qkv_elems = batch * (size_t)qkv_size;
    attn_elems = batch * (size_t)attn_storage_width;
    gate_up_elems = batch * (size_t)((2 * I) > linear_baz_rows ? (2 * I) : linear_baz_rows);
    max_input_elems = batch * (size_t)max_input_cols;
    logits_elems = logits_out != NULL ? batch * (size_t)V : 0u;

    if (profile_enabled) {
        profile_phase_start = microgemm_profile_now_ms();
    }
    hidden = (float*)malloc(hidden_elems * sizeof(float));
    residual = (float*)malloc(hidden_elems * sizeof(float));
    normed = (float*)malloc(hidden_elems * sizeof(float));
    qkv = (float*)malloc(qkv_elems * sizeof(float));
    attn_out = (float*)malloc(attn_elems * sizeof(float));
    tmp_h = (float*)malloc(hidden_elems * sizeof(float));
    gate_up = (float*)malloc(gate_up_elems * sizeof(float));
    logits_batch = logits_out == NULL
        ? NULL
        : (outputs_are_contiguous ? logits_out[0] : (float*)malloc(logits_elems * sizeof(float)));
    input_scales = (float*)malloc(batch * sizeof(float));
    if (swiglu_down_prequant_requested) {
        swiglu_absmax_scratch = (float*)malloc(swiglu_absmax_elems * sizeof(float));
    }
    input_q = (int8_t*)malloc(max_input_elems * sizeof(int8_t));
    if (hidden == NULL || residual == NULL || normed == NULL || qkv == NULL
            || attn_out == NULL || tmp_h == NULL || gate_up == NULL
            || (logits_out != NULL && logits_batch == NULL)
            || input_scales == NULL
            || (swiglu_down_prequant_requested && swiglu_absmax_scratch == NULL)
            || input_q == NULL) {
        status = MICROGEMM_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    if (profile_enabled) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.alloc_ms, profile_phase_start);
    }

    if (profile_enabled) {
        profile_phase_start = microgemm_profile_now_ms();
    }
    for (t = 0; t < batch_i; ++t) {
        microgemm_decode_load_embedding_row(config, model, token_ids[t], H, hidden + (size_t)t * H);
    }
    if (profile_enabled) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.embed_ms, profile_phase_start);
    }

    for (layer_idx = 0; layer_idx < (int)config->num_layers; ++layer_idx) {
        const microgemm_layer_weights_i8* layer = &model->layers[layer_idx];
        int normed_prequantized = microgemm_rmsnorm_prequant_enabled_for(config->quant_mode, batch_i, H);

        if (!microgemm_layer_has_quantized_weights(config, layer)) {
            status = MICROGEMM_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }

        memcpy(residual, hidden, hidden_elems * sizeof(float));
        if (profile_enabled) {
            profile_phase_start = microgemm_profile_now_ms();
        }
        if (normed_prequantized) {
            microgemm_rmsnorm_batch_quantize_i8_self_biasing(
                normed,
                input_q,
                input_scales,
                hidden,
                layer->input_norm_w,
                batch_i,
                H,
                config->rms_norm_eps,
                offset_weights
            );
        } else {
            microgemm_rmsnorm_batch(
                normed,
                hidden,
                layer->input_norm_w,
                batch_i,
                H,
                config->rms_norm_eps,
                offset_weights
            );
        }
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.input_norm_ms, profile_phase_start);
            profile_phase_start = microgemm_profile_now_ms();
        }
        if (layer->layer_type == MICROGEMM_LAYER_LINEAR_ATTENTION) {
            if (normed_prequantized) {
                microgemm_gemv_quantized_batched_prequantized(
                    config,
                    qkv,
                    layer->linear_qkv_w,
                    layer->linear_qkv_w_i4,
                    layer->linear_qkv_s,
                    layer->linear_qkv_row_sums,
                    batch_i,
                    linear_conv_dim,
                    H,
                    NULL,
                    input_q,
                    input_scales
                );
            } else {
                microgemm_gemv_quantized_batched(
                    config,
                    qkv,
                    layer->linear_qkv_w,
                    layer->linear_qkv_w_i4,
                    layer->linear_qkv_s,
                    layer->linear_qkv_row_sums,
                    normed,
                    batch_i,
                    linear_conv_dim,
                    H,
                    H,
                    NULL,
                    input_q,
                    input_scales
                );
            }
            if (profile_enabled) {
                microgemm_profile_add(&microgemm_decode_batch_profile_state.qkv_ms, profile_phase_start);
                profile_phase_start = microgemm_profile_now_ms();
            }
            if (normed_prequantized) {
                microgemm_gemv_quantized_batched_prequantized(
                    config,
                    gate_up,
                    layer->linear_baz_w,
                    layer->linear_baz_w_i4,
                    layer->linear_baz_s,
                    layer->linear_baz_row_sums,
                    batch_i,
                    linear_baz_rows,
                    H,
                    NULL,
                    input_q,
                    input_scales
                );
            } else {
                microgemm_gemv_quantized_batched(
                    config,
                    gate_up,
                    layer->linear_baz_w,
                    layer->linear_baz_w_i4,
                    layer->linear_baz_s,
                    layer->linear_baz_row_sums,
                    normed,
                    batch_i,
                    linear_baz_rows,
                    H,
                    H,
                    NULL,
                    input_q,
                    input_scales
                );
            }
            if (profile_enabled) {
                microgemm_profile_add(&microgemm_decode_batch_profile_state.attention_ms, profile_phase_start);
                profile_phase_start = microgemm_profile_now_ms();
            }
            {
                microgemm_status update_status = MICROGEMM_STATUS_OK;
                #pragma omp parallel for schedule(static) if(batch_i >= 2)
                for (t = 0; t < batch_i; ++t) {
                    microgemm_status local_status = microgemm_linear_attention_update_from_buffers(
                        config,
                        layer,
                        workspaces[t],
                        layer_idx,
                        positions[t],
                        qkv + (size_t)t * (size_t)linear_conv_dim,
                        gate_up + (size_t)t * (size_t)linear_baz_rows,
                        attn_out + (size_t)t * (size_t)linear_value_dim
                    );
                    if (local_status != MICROGEMM_STATUS_OK) {
                        #pragma omp critical(microgemm_linear_attention_update_status)
                        {
                            if (update_status == MICROGEMM_STATUS_OK) {
                                update_status = local_status;
                            }
                        }
                    }
                }
                if (update_status != MICROGEMM_STATUS_OK) {
                    status = update_status;
                    goto cleanup;
                }
            }
            if (profile_enabled) {
                microgemm_profile_add(&microgemm_decode_batch_profile_state.rope_kv_ms, profile_phase_start);
                profile_phase_start = microgemm_profile_now_ms();
            }
            microgemm_gemv_quantized_batched(
                config,
                tmp_h,
                layer->linear_out_w,
                layer->linear_out_w_i4,
                layer->linear_out_s,
                layer->linear_out_row_sums,
                attn_out,
                batch_i,
                H,
                linear_value_dim,
                linear_value_dim,
                NULL,
                input_q,
                input_scales
            );
        } else {
            if (normed_prequantized) {
                microgemm_gemv_quantized_batched_prequantized(
                    config,
                    qkv,
                    layer->qkv_w,
                    layer->qkv_w_i4,
                    layer->qkv_s,
                    layer->qkv_row_sums,
                    batch_i,
                    qkv_size,
                    H,
                    (config->flags & MICROGEMM_FLAG_QKV_BIAS) ? layer->qkv_bias : NULL,
                    input_q,
                    input_scales
                );
            } else {
                microgemm_gemv_quantized_batched(
                    config,
                    qkv,
                    layer->qkv_w,
                    layer->qkv_w_i4,
                    layer->qkv_s,
                    layer->qkv_row_sums,
                    normed,
                    batch_i,
                    qkv_size,
                    H,
                    H,
                    (config->flags & MICROGEMM_FLAG_QKV_BIAS) ? layer->qkv_bias : NULL,
                    input_q,
                    input_scales
                );
            }
            if (profile_enabled) {
                microgemm_profile_add(&microgemm_decode_batch_profile_state.qkv_ms, profile_phase_start);
                profile_phase_start = microgemm_profile_now_ms();
            }

            {
                microgemm_status update_status = MICROGEMM_STATUS_OK;
                #pragma omp parallel for schedule(static) if(batch_i >= 2)
                for (t = 0; t < batch_i; ++t) {
                    const microgemm_kv_layout* kv = kvs[t];
                    float* layer_kv = kv->layer_kv[layer_idx];
                    int blk_idx = kv->seq_len / (int)config->kv_block_size;
                    int blk_off = kv->seq_len % (int)config->kv_block_size;
                    int phys_blk;
                    float* q = qkv + (size_t)t * qkv_size;
                    float* q_gate = (config->flags & MICROGEMM_FLAG_ATTN_OUTPUT_GATE) != 0u
                        ? q + (size_t)attn_width
                        : NULL;
                    float* k = q + (size_t)full_q_rows;
                    float* v = k + (size_t)Nkv * D;
                    const float* cos_v = model->cos_cache + (size_t)positions[t] * (D / 2);
                    const float* sin_v = model->sin_cache + (size_t)positions[t] * (D / 2);
                    int h;
                    (void)q_gate;

                    if (layer_kv == NULL) {
                        #pragma omp critical(microgemm_full_attention_kv_status)
                        {
                            if (update_status == MICROGEMM_STATUS_OK) {
                                update_status = MICROGEMM_STATUS_INVALID_ARGUMENT;
                            }
                        }
                        continue;
                    }
                    phys_blk = kv->block_table[blk_idx];

                    if ((config->flags & MICROGEMM_FLAG_QK_NORM) != 0u) {
                        microgemm_qk_norm_inplace(
                            q,
                            k,
                            Nq,
                            Nkv,
                            D,
                            layer->q_norm_w,
                            layer->k_norm_w,
                            config->rms_norm_eps,
                            offset_weights
                        );
                    }
                    microgemm_rope_rotate_partial(
                        q,
                        k,
                        cos_v,
                        sin_v,
                        Nq,
                        Nkv,
                        D,
                        rotary_dim,
                        (config->flags & MICROGEMM_FLAG_ROPE_INTERLEAVED) != 0u
                    );
                    for (h = 0; h < Nkv; ++h) {
                        float* k_dst = layer_kv
                            + (size_t)phys_blk * kv->stride_block
                            + (size_t)h * kv->stride_head
                            + (size_t)blk_off * kv->stride_pos;
                        float* v_dst = layer_kv
                            + (size_t)phys_blk * kv->stride_block
                            + (size_t)kv->stride_kv
                            + (size_t)h * kv->stride_head
                            + (size_t)blk_off * kv->stride_pos;
                        memcpy(k_dst, k + (size_t)h * D, (size_t)D * sizeof(float));
                        memcpy(v_dst, v + (size_t)h * D, (size_t)D * sizeof(float));
                    }
                }
                if (update_status != MICROGEMM_STATUS_OK) {
                    status = update_status;
                    goto cleanup;
                }
            }
            if (profile_enabled) {
                microgemm_profile_add(&microgemm_decode_batch_profile_state.rope_kv_ms, profile_phase_start);
                profile_phase_start = microgemm_profile_now_ms();
            }

            microgemm_attention_decode_batch(
                attn_out,
                workspaces,
                qkv,
                kvs,
                batch_i,
                layer_idx,
                qkv_size,
                attn_width,
                Nq,
                Nkv,
                D,
                (int)config->kv_block_size,
                scale,
                config->attention_logit_softcap
            );
            if ((config->flags & MICROGEMM_FLAG_ATTN_OUTPUT_GATE) != 0u) {
                for (t = 0; t < batch_i; ++t) {
                    float* q = qkv + (size_t)t * qkv_size;
                    float* q_gate = q + (size_t)attn_width;
                    float* out_t = attn_out + (size_t)t * attn_width;
                    int gate_i;
                    for (gate_i = 0; gate_i < attn_width; ++gate_i) {
                        out_t[gate_i] *= microgemm_sigmoid_scalar(q_gate[gate_i]);
                    }
                }
            }
            if (profile_enabled) {
                microgemm_profile_add(&microgemm_decode_batch_profile_state.attention_ms, profile_phase_start);
                profile_phase_start = microgemm_profile_now_ms();
            }

            microgemm_gemv_quantized_batched(
                config,
                tmp_h,
                layer->o_w,
                layer->o_w_i4,
                layer->o_s,
                layer->o_row_sums,
                attn_out,
                batch_i,
                H,
                attn_width,
                attn_width,
                NULL,
                input_q,
                input_scales
            );
        }
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.o_proj_ms, profile_phase_start);
        }
        if ((config->flags & MICROGEMM_FLAG_ATTN_OUTPUT_NORM) != 0u) {
            microgemm_rmsnorm_batch(
                tmp_h,
                tmp_h,
                layer->attn_output_norm_w,
                batch_i,
                H,
                config->rms_norm_eps,
                offset_weights
            );
        }
        microgemm_residual_add_batch(
            hidden,
            residual,
            tmp_h,
            batch_i,
            H,
            microgemm_config_residual_multiplier(config)
        );

        memcpy(residual, hidden, hidden_elems * sizeof(float));
        if (profile_enabled) {
            profile_phase_start = microgemm_profile_now_ms();
        }
        microgemm_rmsnorm_batch(
            normed,
            hidden,
            layer->post_attn_norm_w,
            batch_i,
            H,
            config->rms_norm_eps,
            offset_weights
        );
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.post_norm_ms, profile_phase_start);
            profile_phase_start = microgemm_profile_now_ms();
        }
        {
            int mlp_stride = 2 * I;
            int down_prequantized = microgemm_swiglu_down_prequant_enabled_for(
                config->quant_mode, batch_i, I
            );
            int capture_swiglu_absmax = down_prequantized
                && config->quant_mode == MICROGEMM_QUANT_INT8G128
                && layer->gate_up_row_sums != NULL
                && microgemm_groupwise_gate_up_fused_enabled_for(
                    config->quant_mode, batch_i, I, H
                );
#if MICROGEMM_CPU_X86_AVX2
            if ((config->quant_mode == MICROGEMM_QUANT_INT8
                        || config->quant_mode == MICROGEMM_QUANT_INT4)
                    && layer->gate_up_row_sums != NULL && batch_i >= 2) {
                mlp_stride = I;
            } else if (microgemm_quant_mode_is_groupwise(config->quant_mode)
                    && layer->gate_up_row_sums != NULL
                    && batch_i >= 2
                    && microgemm_groupwise_gate_up_fused_enabled_for(
                        config->quant_mode, batch_i, I, H
                    )
                    && microgemm_groupwise_compact_mlp_stride_enabled_for(
                        config->quant_mode, batch_i, I, H
                    )) {
                mlp_stride = I;
            }
#endif
        microgemm_gate_up_swiglu_quantized_batched(
            config,
            gate_up,
            layer->gate_up_w,
            layer->gate_up_w_i4,
            layer->gate_up_s,
            layer->gate_up_row_sums,
            normed,
            batch_i,
            I,
            H,
            H,
            mlp_stride,
            use_gelu,
            input_q,
            input_scales,
            capture_swiglu_absmax ? swiglu_absmax_scratch : NULL,
            capture_swiglu_absmax ? swiglu_absmax_threads : 0
        );
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.gate_up_ms, profile_phase_start);
            profile_phase_start = microgemm_profile_now_ms();
        }

        if (down_prequantized) {
            if (capture_swiglu_absmax) {
                microgemm_swiglu_absmax_reduce_threads(
                    swiglu_absmax_scratch, swiglu_absmax_threads, batch_i
                );
                microgemm_quantize_activation_batch_for_i8_self_biasing_known_amax(
                    input_q,
                    input_scales,
                    gate_up,
                    swiglu_absmax_scratch,
                    batch_i,
                    I,
                    mlp_stride
                );
            } else {
                microgemm_quantize_activation_batch_for_i8_self_biasing(
                    input_q,
                    input_scales,
                    gate_up,
                    batch_i,
                    I,
                    mlp_stride
                );
            }
        }
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.activation_ms, profile_phase_start);
            profile_phase_start = microgemm_profile_now_ms();
        }

        if (profile_enabled) {
            microgemm_profile_next_down_proj_gemv = 1;
        }
        if (down_prequantized) {
            microgemm_gemv_quantized_batched_prequantized(
                config,
                tmp_h,
                layer->down_w,
                layer->down_w_i4,
                layer->down_s,
                layer->down_row_sums,
                batch_i,
                H,
                I,
                NULL,
                input_q,
                input_scales
            );
        } else {
            microgemm_gemv_quantized_batched(
                config,
                tmp_h,
                layer->down_w,
                layer->down_w_i4,
                layer->down_s,
                layer->down_row_sums,
                gate_up,
                batch_i,
                H,
                I,
                mlp_stride,
                NULL,
                input_q,
                input_scales
            );
        }
        microgemm_profile_next_down_proj_gemv = 0;
        if ((config->flags & MICROGEMM_FLAG_MLP_OUTPUT_NORM) != 0u) {
            microgemm_rmsnorm_batch(
                tmp_h,
                tmp_h,
                layer->mlp_output_norm_w,
                batch_i,
                H,
                config->rms_norm_eps,
                offset_weights
            );
        }
        }
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.down_proj_ms, profile_phase_start);
        }
        microgemm_residual_add_batch(
            hidden,
            residual,
            tmp_h,
            batch_i,
            H,
            microgemm_config_residual_multiplier(config)
        );
    }

    if (profile_enabled) {
        profile_phase_start = microgemm_profile_now_ms();
    }
    microgemm_rmsnorm_batch(
        normed,
        hidden,
        model->final_norm_w,
        batch_i,
        H,
        config->rms_norm_eps,
        offset_weights
    );
    if (profile_enabled) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.final_norm_ms, profile_phase_start);
        profile_phase_start = microgemm_profile_now_ms();
    }
    for (t = 0; t < batch_i; ++t) {
        memcpy(workspaces[t]->hidden, hidden + (size_t)t * H, (size_t)H * sizeof(float));
        memcpy(workspaces[t]->normed, normed + (size_t)t * H, (size_t)H * sizeof(float));
    }
    if (profile_enabled) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.copy_ms, profile_phase_start);
    }

    logits_target = logits_batch;
    if (profile_enabled) {
        profile_phase_start = microgemm_profile_now_ms();
    }
    if (out_token_ids != NULL && logits_out == NULL) {
        status = microgemm_lm_head_argmax_quantized_batched(
            config,
            out_token_ids,
            model->lm_head_w,
            model->lm_head_w_i4,
            model->lm_head_s,
            model->lm_head_row_sums,
            normed,
            batch_i,
            V,
            H,
            H,
            input_q,
            input_scales
        );
        if (status != MICROGEMM_STATUS_OK) {
            goto cleanup;
        }
    } else {
        microgemm_gemv_quantized_batched(
            config,
            logits_target,
            model->lm_head_w,
            model->lm_head_w_i4,
            model->lm_head_s,
            model->lm_head_row_sums,
            normed,
            batch_i,
            V,
            H,
            H,
            NULL,
            input_q,
            input_scales
        );
    }
    if (profile_enabled) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.lm_head_ms, profile_phase_start);
    }

    if (logits_out != NULL) {
        microgemm_logits_postprocess_batched(config, logits_target, batch_i, V);
    }

    if (logits_out != NULL && !outputs_are_contiguous) {
        if (profile_enabled) {
            profile_phase_start = microgemm_profile_now_ms();
        }
        for (b = 0u; b < batch; ++b) {
            memcpy(
                logits_out[b],
                logits_batch + b * (size_t)V,
                (size_t)V * sizeof(float)
            );
        }
        if (profile_enabled) {
            microgemm_profile_add(&microgemm_decode_batch_profile_state.copy_ms, profile_phase_start);
        }
    }

cleanup:
    if (profile_enabled) {
        profile_phase_start = microgemm_profile_now_ms();
    }
    free(hidden);
    free(residual);
    free(normed);
    free(qkv);
    free(attn_out);
    free(tmp_h);
    free(gate_up);
    if (logits_out != NULL && !outputs_are_contiguous) {
        free(logits_batch);
    }
    free(input_scales);
    free(swiglu_absmax_scratch);
    free(input_q);
    if (profile_enabled) {
        microgemm_profile_add(&microgemm_decode_batch_profile_state.cleanup_ms, profile_phase_start);
        microgemm_profile_add(&microgemm_decode_batch_profile_state.total_ms, profile_total_start);
    }
    return status;
}

microgemm_status microgemm_decode_step_i8_batch(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* const* workspaces,
    const int* token_ids,
    const int* positions,
    const microgemm_kv_layout* const* kvs,
    size_t batch,
    float* const* logits_out,
    int logits_capacity
) {
    return microgemm_decode_step_i8_batch_impl(
        config,
        model,
        workspaces,
        token_ids,
        positions,
        kvs,
        batch,
        logits_out,
        logits_capacity,
        NULL
    );
}

microgemm_status microgemm_decode_step_i8_batch_next_token(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* const* workspaces,
    const int* token_ids,
    const int* positions,
    const microgemm_kv_layout* const* kvs,
    size_t batch,
    int* out_token_ids
) {
    return microgemm_decode_step_i8_batch_impl(
        config,
        model,
        workspaces,
        token_ids,
        positions,
        kvs,
        batch,
        NULL,
        0,
        out_token_ids
    );
}

microgemm_status microgemm_decode_logits_i8_batch(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* const* workspaces,
    size_t batch,
    float* const* logits_out,
    int logits_capacity
) {
    int H;
    int V;
    int batch_i;
    int outputs_are_contiguous = 1;
    size_t b;
    float* normed_batch = NULL;
    float* logits_batch = NULL;
    float* logits_target = NULL;
    int8_t* input_q = NULL;
    float* input_scales = NULL;
    microgemm_status status = MICROGEMM_STATUS_OK;

    if (!microgemm_decode_config_is_valid(config) || model == NULL
            || workspaces == NULL || logits_out == NULL || batch == 0u) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (!microgemm_quant_mode_is_i8_storage(config->quant_mode)
            && !microgemm_quant_mode_is_i4_storage(config->quant_mode)) {
        return MICROGEMM_STATUS_UNSUPPORTED;
    }
    if (!microgemm_model_has_lm_head(config, model)) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (batch > (size_t)INT_MAX) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    H = (int)config->hidden_size;
    V = (int)config->vocab_size;
    batch_i = (int)batch;
    if (logits_capacity < V) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    for (b = 0u; b < batch; ++b) {
        if (workspaces[b] == NULL || logits_out[b] == NULL
                || workspaces[b]->normed == NULL) {
            return MICROGEMM_STATUS_INVALID_ARGUMENT;
        }
        if (workspaces[b]->config.hidden_size != config->hidden_size
                || workspaces[b]->config.num_layers != config->num_layers
                || workspaces[b]->config.vocab_size != config->vocab_size) {
            return MICROGEMM_STATUS_INVALID_ARGUMENT;
        }
        if (b > 0u) {
            uintptr_t expected =
                (uintptr_t)logits_out[0] + b * (size_t)V * sizeof(float);
            if ((uintptr_t)logits_out[b] != expected) {
                outputs_are_contiguous = 0;
            }
        }
    }

    if (batch == 1u) {
        microgemm_gemv_quantized(
            config,
            logits_out[0],
            model->lm_head_w,
            model->lm_head_w_i4,
            model->lm_head_s,
            model->lm_head_row_sums,
            workspaces[0]->normed,
            V,
            H,
            NULL,
            workspaces[0]->input_q
        );
        microgemm_logits_postprocess_inplace(config, logits_out[0], V);
        return MICROGEMM_STATUS_OK;
    }

    if (batch > ((size_t)-1) / (size_t)H
            || batch > ((size_t)-1) / (size_t)V) {
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }

    normed_batch = (float*)malloc(batch * (size_t)H * sizeof(float));
    logits_batch = outputs_are_contiguous
        ? logits_out[0]
        : (float*)malloc(batch * (size_t)V * sizeof(float));
    input_q = (int8_t*)malloc(batch * (size_t)H * sizeof(int8_t));
    input_scales = (float*)malloc(batch * sizeof(float));
    if (normed_batch == NULL || logits_batch == NULL
            || input_q == NULL || input_scales == NULL) {
        status = MICROGEMM_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (b = 0u; b < batch; ++b) {
        memcpy(
            normed_batch + b * (size_t)H,
            workspaces[b]->normed,
            (size_t)H * sizeof(float)
        );
    }

    logits_target = logits_batch;
    microgemm_gemv_quantized_batched(
        config,
        logits_target,
        model->lm_head_w,
        model->lm_head_w_i4,
        model->lm_head_s,
        model->lm_head_row_sums,
        normed_batch,
        batch_i,
        V,
        H,
        H,
        NULL,
        input_q,
        input_scales
    );
    microgemm_logits_postprocess_batched(config, logits_target, batch_i, V);

    if (!outputs_are_contiguous) {
        for (b = 0u; b < batch; ++b) {
            memcpy(
                logits_out[b],
                logits_batch + b * (size_t)V,
                (size_t)V * sizeof(float)
            );
        }
    }

cleanup:
    free(normed_batch);
    if (!outputs_are_contiguous) {
        free(logits_batch);
    }
    free(input_q);
    free(input_scales);
    return status;
}

static microgemm_status microgemm_decode_prefill_i8_sequential(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* workspace,
    const int* token_ids,
    size_t token_count,
    microgemm_kv_layout* kv,
    float* logits_out,
    int logits_capacity,
    int* out_token_id
) {
    size_t i;
    microgemm_status status;

    for (i = 0u; i < token_count; ++i) {
        int position = kv->seq_len;
        int is_last = ((i + 1u) == token_count);
        float* step_logits = is_last ? logits_out : NULL;
        int step_logits_capacity = is_last ? logits_capacity : 0;
        int* step_token_id = is_last ? out_token_id : NULL;

        status = microgemm_decode_validate_token_position(
            config,
            workspace,
            token_ids[i],
            position
        );
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }

        status = microgemm_decode_step_i8_impl(
            config,
            model,
            workspace,
            token_ids[i],
            position,
            kv,
            step_logits,
            step_logits_capacity,
            step_token_id
        );
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }

        kv->seq_len += 1;
    }

    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_decode_prefill_i8_batched(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* workspace,
    const int* token_ids,
    size_t token_count,
    microgemm_kv_layout* kv,
    float* logits_out,
    int* out_token_id
) {
    int H = (int)config->hidden_size;
    int I = (int)config->intermediate_size;
    int Nq = (int)config->num_q_heads;
    int Nkv = (int)config->num_kv_heads;
    int D = (int)config->head_dim;
    int qkv_size = (int)(config->qkv_rows != 0u
        ? config->qkv_rows
        : microgemm_full_qkv_rows_decode(config));
    int full_q_rows = microgemm_full_q_rows_decode(config);
    int attn_width = microgemm_full_attn_width(config);
    int linear_conv_dim = microgemm_linear_conv_dim_decode(config);
    int linear_value_dim = microgemm_linear_value_dim_decode(config);
    int linear_baz_rows = microgemm_linear_baz_rows_decode(config);
    int attn_storage_width = attn_width;
    int rotary_dim = (int)(config->rotary_dim != 0u ? config->rotary_dim : config->head_dim);
    int has_linear_layers = microgemm_has_linear_attention_layers(config, model);
    int batch = (int)token_count;
    int base_seq_len = kv->seq_len;
    int max_input_cols = H;
    int layer_idx;
    int t;
    float scale = microgemm_attention_scale(config, D);
    int offset_weights = (config->flags & MICROGEMM_FLAG_NORM_OFFSET) != 0;
    int use_gelu = (config->flags & MICROGEMM_FLAG_MLP_GELU) != 0;
    float* hidden = NULL;
    float* residual = NULL;
    float* normed = NULL;
    float* qkv = NULL;
    float* attn_out = NULL;
    float* prefill_attention_scores = NULL;
    float* tmp_h = NULL;
    float* gate_up = NULL;
    float* input_scales = NULL;
    float* swiglu_absmax_scratch = NULL;
    int8_t* input_q = NULL;
    size_t hidden_elems = token_count * (size_t)H;
    size_t qkv_elems = token_count * (size_t)qkv_size;
    size_t attn_elems;
    size_t gate_up_elems;
    size_t swiglu_absmax_elems;
    size_t prefill_attention_score_stride = 0u;
    int use_prefill_parallel_attention = 0;
    int max_attn_seq_len = base_seq_len + batch;
    int swiglu_down_prequant_requested = 0;
    int swiglu_absmax_threads = 1;
    microgemm_status status = MICROGEMM_STATUS_OK;

    if (batch <= 0) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    if (!has_linear_layers) {
        linear_conv_dim = 0;
        linear_value_dim = 0;
        linear_baz_rows = 0;
    }
    if (linear_value_dim > attn_storage_width) {
        attn_storage_width = linear_value_dim;
    }
    if (attn_storage_width > max_input_cols) {
        max_input_cols = attn_storage_width;
    }
    if (I > max_input_cols) {
        max_input_cols = I;
    }
    if (linear_conv_dim > max_input_cols) {
        max_input_cols = linear_conv_dim;
    }
    if (linear_baz_rows > max_input_cols) {
        max_input_cols = linear_baz_rows;
    }
    swiglu_down_prequant_requested = microgemm_swiglu_down_prequant_enabled_for(
        config->quant_mode, batch, I
    );
    if (swiglu_down_prequant_requested) {
        swiglu_absmax_threads = microgemm_max_worker_threads();
        if ((size_t)swiglu_absmax_threads > ((size_t)-1) / (size_t)batch) {
            return MICROGEMM_STATUS_OUT_OF_MEMORY;
        }
        swiglu_absmax_elems = (size_t)swiglu_absmax_threads * (size_t)batch;
    } else {
        swiglu_absmax_elems = 0u;
    }
    attn_elems = token_count * (size_t)attn_storage_width;
    gate_up_elems = token_count * (size_t)((2 * I) > linear_baz_rows ? (2 * I) : linear_baz_rows);

    hidden = (float*)malloc(hidden_elems * sizeof(float));
    residual = (float*)malloc(hidden_elems * sizeof(float));
    normed = (float*)malloc(hidden_elems * sizeof(float));
    qkv = (float*)malloc(qkv_elems * sizeof(float));
    attn_out = (float*)malloc(attn_elems * sizeof(float));
    tmp_h = (float*)malloc(hidden_elems * sizeof(float));
    gate_up = (float*)malloc(gate_up_elems * sizeof(float));
    input_scales = (float*)malloc((size_t)batch * sizeof(float));
    if (swiglu_down_prequant_requested) {
        swiglu_absmax_scratch = (float*)malloc(swiglu_absmax_elems * sizeof(float));
    }
    input_q = (int8_t*)malloc((size_t)batch * (size_t)max_input_cols * sizeof(int8_t));

    if (hidden == NULL || residual == NULL || normed == NULL || qkv == NULL
            || attn_out == NULL || tmp_h == NULL || gate_up == NULL
            || input_scales == NULL
            || (swiglu_down_prequant_requested && swiglu_absmax_scratch == NULL)
            || input_q == NULL) {
        status = MICROGEMM_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    if (microgemm_prefill_parallel_attention_enabled_for(batch, base_seq_len, max_attn_seq_len, Nq)
            && (size_t)max_attn_seq_len <= SIZE_MAX / (size_t)Nq) {
        const size_t max_score_elems = ((size_t)128u * 1024u * 1024u) / sizeof(float);
        prefill_attention_score_stride = (size_t)Nq * (size_t)max_attn_seq_len;
        if (prefill_attention_score_stride <= SIZE_MAX / (size_t)batch
                && prefill_attention_score_stride * (size_t)batch <= max_score_elems) {
            prefill_attention_scores = (float*)malloc(
                prefill_attention_score_stride * (size_t)batch * sizeof(float)
            );
            use_prefill_parallel_attention = prefill_attention_scores != NULL;
        }
    }

    for (t = 0; t < batch; ++t) {
        int token_id = token_ids[t];
        int position = base_seq_len + t;
        status = microgemm_decode_validate_token_position(config, workspace, token_id, position);
        if (status != MICROGEMM_STATUS_OK) {
            goto cleanup;
        }
        microgemm_decode_load_embedding_row(config, model, token_id, H, hidden + (size_t)t * H);
    }

    for (layer_idx = 0; layer_idx < (int)config->num_layers; ++layer_idx) {
        const microgemm_layer_weights_i8* layer = &model->layers[layer_idx];
        int normed_prequantized = microgemm_rmsnorm_prequant_enabled_for(config->quant_mode, (int)batch, H);
        float* layer_kv;

        if (!microgemm_layer_has_quantized_weights(config, layer)) {
            status = MICROGEMM_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }

        layer_kv = kv->layer_kv[layer_idx];
        if (layer->layer_type != MICROGEMM_LAYER_LINEAR_ATTENTION && layer_kv == NULL) {
            status = MICROGEMM_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }

        memcpy(residual, hidden, hidden_elems * sizeof(float));
        if (normed_prequantized) {
            microgemm_rmsnorm_batch_quantize_i8_self_biasing(
                normed,
                input_q,
                input_scales,
                hidden,
                layer->input_norm_w,
                batch,
                H,
                config->rms_norm_eps,
                offset_weights
            );
        } else {
            microgemm_rmsnorm_batch(
                normed,
                hidden,
                layer->input_norm_w,
                batch,
                H,
                config->rms_norm_eps,
                offset_weights
            );
        }
        if (layer->layer_type == MICROGEMM_LAYER_LINEAR_ATTENTION) {
            if (normed_prequantized) {
                microgemm_gemv_quantized_batched_prequantized(
                    config,
                    qkv,
                    layer->linear_qkv_w,
                    layer->linear_qkv_w_i4,
                    layer->linear_qkv_s,
                    layer->linear_qkv_row_sums,
                    batch,
                    linear_conv_dim,
                    H,
                    NULL,
                    input_q,
                    input_scales
                );
                microgemm_gemv_quantized_batched_prequantized(
                    config,
                    gate_up,
                    layer->linear_baz_w,
                    layer->linear_baz_w_i4,
                    layer->linear_baz_s,
                    layer->linear_baz_row_sums,
                    batch,
                    linear_baz_rows,
                    H,
                    NULL,
                    input_q,
                    input_scales
                );
            } else {
                microgemm_gemv_quantized_batched(
                    config,
                    qkv,
                    layer->linear_qkv_w,
                    layer->linear_qkv_w_i4,
                    layer->linear_qkv_s,
                    layer->linear_qkv_row_sums,
                    normed,
                    batch,
                    linear_conv_dim,
                    H,
                    H,
                    NULL,
                    input_q,
                    input_scales
                );
                microgemm_gemv_quantized_batched(
                    config,
                    gate_up,
                    layer->linear_baz_w,
                    layer->linear_baz_w_i4,
                    layer->linear_baz_s,
                    layer->linear_baz_row_sums,
                    normed,
                    batch,
                    linear_baz_rows,
                    H,
                    H,
                    NULL,
                    input_q,
                    input_scales
                );
            }

            for (t = 0; t < batch; ++t) {
                int position = base_seq_len + t;
                status = microgemm_linear_attention_update_from_buffers(
                    config,
                    layer,
                    workspace,
                    layer_idx,
                    position,
                    qkv + (size_t)t * (size_t)linear_conv_dim,
                    gate_up + (size_t)t * (size_t)linear_baz_rows,
                    attn_out + (size_t)t * (size_t)linear_value_dim
                );
                if (status != MICROGEMM_STATUS_OK) {
                    goto cleanup;
                }
            }

            microgemm_gemv_quantized_batched(
                config,
                tmp_h,
                layer->linear_out_w,
                layer->linear_out_w_i4,
                layer->linear_out_s,
                layer->linear_out_row_sums,
                attn_out,
                batch,
                H,
                linear_value_dim,
                linear_value_dim,
                NULL,
                input_q,
                input_scales
            );
        } else {
            if (normed_prequantized) {
                microgemm_gemv_quantized_batched_prequantized(
                    config,
                    qkv,
                    layer->qkv_w,
                    layer->qkv_w_i4,
                    layer->qkv_s,
                    layer->qkv_row_sums,
                    batch,
                    qkv_size,
                    H,
                    (config->flags & MICROGEMM_FLAG_QKV_BIAS) ? layer->qkv_bias : NULL,
                    input_q,
                    input_scales
                );
            } else {
                microgemm_gemv_quantized_batched(
                    config,
                    qkv,
                    layer->qkv_w,
                    layer->qkv_w_i4,
                    layer->qkv_s,
                    layer->qkv_row_sums,
                    normed,
                    batch,
                    qkv_size,
                    H,
                    H,
                    (config->flags & MICROGEMM_FLAG_QKV_BIAS) ? layer->qkv_bias : NULL,
                    input_q,
                    input_scales
                );
            }

            for (t = 0; t < batch; ++t) {
                int position = base_seq_len + t;
                int blk_idx = position / (int)config->kv_block_size;
                int blk_off = position % (int)config->kv_block_size;
                int phys_blk = kv->block_table[blk_idx];
                float* q = qkv + (size_t)t * qkv_size;
                float* k = q + (size_t)full_q_rows;
                float* v = k + (size_t)Nkv * D;
                const float* cos_v = model->cos_cache + (size_t)position * (D / 2);
                const float* sin_v = model->sin_cache + (size_t)position * (D / 2);
                int h;

                if ((config->flags & MICROGEMM_FLAG_QK_NORM) != 0u) {
                    microgemm_qk_norm_inplace(
                        q,
                        k,
                        Nq,
                        Nkv,
                        D,
                        layer->q_norm_w,
                        layer->k_norm_w,
                        config->rms_norm_eps,
                        offset_weights
                    );
                }
                microgemm_rope_rotate_partial(
                    q,
                    k,
                    cos_v,
                    sin_v,
                    Nq,
                    Nkv,
                    D,
                    rotary_dim,
                    (config->flags & MICROGEMM_FLAG_ROPE_INTERLEAVED) != 0u
                );
                for (h = 0; h < Nkv; ++h) {
                    float* k_dst = layer_kv
                        + (size_t)phys_blk * kv->stride_block
                        + (size_t)h * kv->stride_head
                        + (size_t)blk_off * kv->stride_pos;
                    float* v_dst = layer_kv
                        + (size_t)phys_blk * kv->stride_block
                        + (size_t)kv->stride_kv
                        + (size_t)h * kv->stride_head
                        + (size_t)blk_off * kv->stride_pos;
                    memcpy(k_dst, k + (size_t)h * D, (size_t)D * sizeof(float));
                    memcpy(v_dst, v + (size_t)h * D, (size_t)D * sizeof(float));
                }
            }

            if (use_prefill_parallel_attention) {
                #pragma omp parallel for schedule(dynamic, 1) if(batch >= 16)
                for (t = 0; t < batch; ++t) {
                    int seq_len_t = base_seq_len + t + 1;
                    const float* q = qkv + (size_t)t * qkv_size;
                    float* out_t = attn_out + (size_t)t * attn_width;
                    float* scores_t = prefill_attention_scores
                        + (size_t)t * prefill_attention_score_stride;

                    microgemm_attention_decode(
                        out_t,
                        scores_t,
                        q,
                        layer_kv,
                        kv->block_table,
                        seq_len_t,
                        Nq,
                        Nkv,
                        D,
                        (int)config->kv_block_size,
                        scale,
                        config->attention_logit_softcap,
                        kv->stride_block,
                        kv->stride_kv,
                        kv->stride_head,
                        kv->stride_pos
                    );
                }
            } else {
                for (t = 0; t < batch; ++t) {
                    int seq_len_t = base_seq_len + t + 1;
                    const float* q = qkv + (size_t)t * qkv_size;
                    float* out_t = attn_out + (size_t)t * attn_width;

                    microgemm_attention_decode(
                        out_t,
                        workspace->scores,
                        q,
                        layer_kv,
                        kv->block_table,
                        seq_len_t,
                        Nq,
                        Nkv,
                        D,
                        (int)config->kv_block_size,
                        scale,
                        config->attention_logit_softcap,
                        kv->stride_block,
                        kv->stride_kv,
                        kv->stride_head,
                        kv->stride_pos
                    );
                }
            }
            if ((config->flags & MICROGEMM_FLAG_ATTN_OUTPUT_GATE) != 0u) {
                for (t = 0; t < batch; ++t) {
                    float* q = qkv + (size_t)t * qkv_size;
                    float* q_gate = q + (size_t)attn_width;
                    float* out_t = attn_out + (size_t)t * attn_width;
                    int gate_i;
                    for (gate_i = 0; gate_i < attn_width; ++gate_i) {
                        out_t[gate_i] *= microgemm_sigmoid_scalar(q_gate[gate_i]);
                    }
                }
            }

            microgemm_gemv_quantized_batched(
                config,
                tmp_h,
                layer->o_w,
                layer->o_w_i4,
                layer->o_s,
                layer->o_row_sums,
                attn_out,
                batch,
                H,
                attn_width,
                attn_width,
                NULL,
                input_q,
                input_scales
            );
        }
        if ((config->flags & MICROGEMM_FLAG_ATTN_OUTPUT_NORM) != 0u) {
            microgemm_rmsnorm_batch(
                tmp_h,
                tmp_h,
                layer->attn_output_norm_w,
                batch,
                H,
                config->rms_norm_eps,
                offset_weights
            );
        }
        microgemm_residual_add_batch(
            hidden,
            residual,
            tmp_h,
            batch,
            H,
            microgemm_config_residual_multiplier(config)
        );

        memcpy(residual, hidden, hidden_elems * sizeof(float));
        microgemm_rmsnorm_batch(
            normed,
            hidden,
            layer->post_attn_norm_w,
            batch,
            H,
            config->rms_norm_eps,
            offset_weights
        );
        {
            int mlp_stride = 2 * I;
            int down_prequantized = microgemm_swiglu_down_prequant_enabled_for(
                config->quant_mode, (int)batch, I
            );
            int capture_swiglu_absmax = down_prequantized
                && config->quant_mode == MICROGEMM_QUANT_INT8G128
                && layer->gate_up_row_sums != NULL
                && microgemm_groupwise_gate_up_fused_enabled_for(
                    config->quant_mode, (int)batch, I, H
                );
#if MICROGEMM_CPU_X86_AVX2
            if ((config->quant_mode == MICROGEMM_QUANT_INT8
                        || config->quant_mode == MICROGEMM_QUANT_INT4)
                    && layer->gate_up_row_sums != NULL && batch >= 2) {
                mlp_stride = I;
            } else if (microgemm_quant_mode_is_groupwise(config->quant_mode)
                    && layer->gate_up_row_sums != NULL
                    && batch >= 2
                    && microgemm_groupwise_gate_up_fused_enabled_for(
                        config->quant_mode, (int)batch, I, H
                    )
                    && microgemm_groupwise_compact_mlp_stride_enabled_for(
                        config->quant_mode, (int)batch, I, H
                    )) {
                mlp_stride = I;
            }
#endif
        microgemm_gate_up_swiglu_quantized_batched(
            config,
            gate_up,
            layer->gate_up_w,
            layer->gate_up_w_i4,
            layer->gate_up_s,
            layer->gate_up_row_sums,
            normed,
            batch,
            I,
            H,
            H,
            mlp_stride,
            use_gelu,
            input_q,
            input_scales,
            capture_swiglu_absmax ? swiglu_absmax_scratch : NULL,
            capture_swiglu_absmax ? swiglu_absmax_threads : 0
        );

        if (down_prequantized) {
            if (capture_swiglu_absmax) {
                microgemm_swiglu_absmax_reduce_threads(
                    swiglu_absmax_scratch, swiglu_absmax_threads, batch
                );
                microgemm_quantize_activation_batch_for_i8_self_biasing_known_amax(
                    input_q,
                    input_scales,
                    gate_up,
                    swiglu_absmax_scratch,
                    batch,
                    I,
                    mlp_stride
                );
            } else {
                microgemm_quantize_activation_batch_for_i8_self_biasing(
                    input_q,
                    input_scales,
                    gate_up,
                    batch,
                    I,
                    mlp_stride
                );
            }
            microgemm_gemv_quantized_batched_prequantized(
                config,
                tmp_h,
                layer->down_w,
                layer->down_w_i4,
                layer->down_s,
                layer->down_row_sums,
                batch,
                H,
                I,
                NULL,
                input_q,
                input_scales
            );
        } else {
            microgemm_gemv_quantized_batched(
                config,
                tmp_h,
                layer->down_w,
                layer->down_w_i4,
                layer->down_s,
                layer->down_row_sums,
                gate_up,
                batch,
                H,
                I,
                mlp_stride,
                NULL,
                input_q,
                input_scales
            );
        }
        if ((config->flags & MICROGEMM_FLAG_MLP_OUTPUT_NORM) != 0u) {
            microgemm_rmsnorm_batch(
                tmp_h,
                tmp_h,
                layer->mlp_output_norm_w,
                batch,
                H,
                config->rms_norm_eps,
                offset_weights
            );
        }
        }
        microgemm_residual_add_batch(
            hidden,
            residual,
            tmp_h,
            batch,
            H,
            microgemm_config_residual_multiplier(config)
        );
    }

    if (logits_out != NULL || out_token_id != NULL) {
        const float* last_hidden = hidden + (size_t)(batch - 1) * H;
        microgemm_cpu_rmsnorm_f32(
            workspace->normed,
            last_hidden,
            model->final_norm_w,
            H,
            config->rms_norm_eps,
            offset_weights
        );
        if (out_token_id != NULL) {
            status = microgemm_lm_head_argmax_quantized_batched(
                config,
                out_token_id,
                model->lm_head_w,
                model->lm_head_w_i4,
                model->lm_head_s,
                model->lm_head_row_sums,
                workspace->normed,
                1,
                (int)config->vocab_size,
                H,
                H,
                input_q,
                input_scales
            );
            if (status != MICROGEMM_STATUS_OK) {
                goto cleanup;
            }
        }
        if (logits_out != NULL) {
            microgemm_gemv_quantized(
                config,
                logits_out,
                model->lm_head_w,
                model->lm_head_w_i4,
                model->lm_head_s,
                model->lm_head_row_sums,
                workspace->normed,
                (int)config->vocab_size,
                H,
                NULL,
                workspace->input_q
            );
            microgemm_logits_postprocess_inplace(config, logits_out, (int)config->vocab_size);
        }
    }

    kv->seq_len += batch;

cleanup:
    free(hidden);
    free(residual);
    free(normed);
    free(qkv);
    free(attn_out);
    free(prefill_attention_scores);
    free(tmp_h);
    free(gate_up);
    free(input_scales);
    free(swiglu_absmax_scratch);
    free(input_q);
    return status;
}

static microgemm_status microgemm_decode_prefill_i8_impl(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* workspace,
    const int* token_ids,
    size_t token_count,
    microgemm_kv_layout* kv,
    float* logits_out,
    int logits_capacity,
    int* out_token_id
) {
    microgemm_status status;
    int H;
    int I;
    int Nq;
    int Nkv;
    int D;
    int qkv_size;
    int attn_width;
    int attn_storage_width;
    int linear_conv_dim;
    int linear_value_dim;
    int linear_baz_rows;
    int gate_up_width;
    int has_linear_layers;
    int max_input_cols;
    size_t per_token_bytes;
    size_t estimated_bytes;

    if (token_ids == NULL || kv == NULL || token_count == 0u) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    status = microgemm_decode_validate_common_inputs(
        config,
        model,
        workspace,
        kv,
        logits_out,
        logits_capacity,
        out_token_id != NULL
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }

    if ((size_t)kv->seq_len + token_count > (size_t)workspace->max_seq_len
            || (size_t)kv->seq_len + token_count > (size_t)config->max_position_embeddings) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    if (token_count < 4u || token_count > (size_t)INT_MAX) {
        return microgemm_decode_prefill_i8_sequential(
            config,
            model,
            workspace,
            token_ids,
            token_count,
            kv,
            logits_out,
            logits_capacity,
            out_token_id
        );
    }

    H = (int)config->hidden_size;
    I = (int)config->intermediate_size;
    Nq = (int)config->num_q_heads;
    Nkv = (int)config->num_kv_heads;
    D = (int)config->head_dim;
    qkv_size = (int)(config->qkv_rows != 0u
        ? config->qkv_rows
        : microgemm_full_qkv_rows_decode(config));
    attn_width = microgemm_full_attn_width(config);
    has_linear_layers = microgemm_has_linear_attention_layers(config, model);
    linear_conv_dim = has_linear_layers ? microgemm_linear_conv_dim_decode(config) : 0;
    linear_value_dim = has_linear_layers ? microgemm_linear_value_dim_decode(config) : 0;
    linear_baz_rows = has_linear_layers ? microgemm_linear_baz_rows_decode(config) : 0;
    attn_storage_width = attn_width;
    if (linear_value_dim > attn_storage_width) {
        attn_storage_width = linear_value_dim;
    }
    gate_up_width = (2 * I) > linear_baz_rows ? (2 * I) : linear_baz_rows;
    max_input_cols = H;
    if (attn_storage_width > max_input_cols) {
        max_input_cols = attn_storage_width;
    }
    if (I > max_input_cols) {
        max_input_cols = I;
    }
    if (linear_conv_dim > max_input_cols) {
        max_input_cols = linear_conv_dim;
    }
    if (linear_baz_rows > max_input_cols) {
        max_input_cols = linear_baz_rows;
    }

    per_token_bytes =
        (size_t)(4 * H + qkv_size + attn_storage_width + gate_up_width) * sizeof(float)
        + (size_t)max_input_cols * sizeof(int8_t)
        + sizeof(float);
    if (per_token_bytes != 0u && token_count > ((size_t)-1) / per_token_bytes) {
        return microgemm_decode_prefill_i8_sequential(
            config,
            model,
            workspace,
            token_ids,
            token_count,
            kv,
            logits_out,
            logits_capacity,
            out_token_id
        );
    }
    estimated_bytes = per_token_bytes * token_count;

    if (estimated_bytes > (size_t)512 * 1024u * 1024u) {
        return microgemm_decode_prefill_i8_sequential(
            config,
            model,
            workspace,
            token_ids,
            token_count,
            kv,
            logits_out,
            logits_capacity,
            out_token_id
        );
    }

    status = microgemm_decode_prefill_i8_batched(
        config,
        model,
        workspace,
        token_ids,
        token_count,
        kv,
        logits_out,
        out_token_id
    );
    if (status == MICROGEMM_STATUS_OUT_OF_MEMORY) {
        return microgemm_decode_prefill_i8_sequential(
            config,
            model,
            workspace,
            token_ids,
            token_count,
            kv,
            logits_out,
            logits_capacity,
            out_token_id
        );
    }
    return status;
}

microgemm_status microgemm_decode_prefill_i8(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* workspace,
    const int* token_ids,
    size_t token_count,
    microgemm_kv_layout* kv,
    float* logits_out,
    int logits_capacity
) {
    return microgemm_decode_prefill_i8_impl(
        config,
        model,
        workspace,
        token_ids,
        token_count,
        kv,
        logits_out,
        logits_capacity,
        NULL
    );
}

microgemm_status microgemm_decode_prefill_i8_next_token(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* workspace,
    const int* token_ids,
    size_t token_count,
    microgemm_kv_layout* kv,
    int* out_token_id
) {
    if (out_token_id == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    return microgemm_decode_prefill_i8_impl(
        config,
        model,
        workspace,
        token_ids,
        token_count,
        kv,
        NULL,
        0,
        out_token_id
    );
}

static int microgemm_selftest_close(float a, float b, float atol, float rtol) {
    float diff = fabsf(a - b);
    float scale = fmaxf(fabsf(a), fabsf(b));
    return diff <= atol + rtol * scale;
}

static float microgemm_selftest_quantize_ref(int8_t* out, const float* input, int count) {
    float amax = 0.0f;
    int i;
    for (i = 0; i < count; ++i) {
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
        float scale = amax / 127.0f;
        float inv_scale = 127.0f / amax;
        for (i = 0; i < count; ++i) {
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

static void microgemm_selftest_rmsnorm_ref(
    float* out,
    const float* input,
    const float* weight,
    int count,
    float eps,
    int offset_weights
) {
    float ss = 0.0f;
    int i;
    for (i = 0; i < count; ++i) {
        ss += input[i] * input[i];
    }
    {
        float rms = 1.0f / sqrtf(ss / (float)count + eps);
        for (i = 0; i < count; ++i) {
            float w = offset_weights ? weight[i] + 1.0f : weight[i];
            out[i] = input[i] * rms * w;
        }
    }
}

static int microgemm_selftest_quantize(void) {
    float input[37];
    int8_t got[37];
    int8_t ref[37];
    float got_scale;
    float ref_scale;
    int i;

    for (i = 0; i < 37; ++i) {
        input[i] = ((float)((i * 17) % 29) - 14.0f) * 0.03125f + (float)(i % 3) * 0.007f;
    }

    got_scale = microgemm_cpu_quantize_f32_to_i8(got, input, 37);
    ref_scale = microgemm_selftest_quantize_ref(ref, input, 37);
    if (!microgemm_selftest_close(got_scale, ref_scale, 1e-7f, 1e-6f)) {
        return 0;
    }
    for (i = 0; i < 37; ++i) {
        if (got[i] != ref[i]) {
            return 0;
        }
    }
    return 1;
}

static int microgemm_selftest_rmsnorm(void) {
    float input[33];
    float weight[33];
    float got[33];
    float ref[33];
    int i;

    for (i = 0; i < 33; ++i) {
        input[i] = ((float)((i * 11) % 23) - 10.0f) * 0.09f;
        weight[i] = 0.75f + (float)(i % 7) * 0.03125f;
    }

    microgemm_cpu_rmsnorm_f32(got, input, weight, 33, 1e-5f, 1);
    microgemm_selftest_rmsnorm_ref(ref, input, weight, 33, 1e-5f, 1);
    for (i = 0; i < 33; ++i) {
        if (!microgemm_selftest_close(got[i], ref[i], 1e-5f, 1e-5f)) {
            return 0;
        }
    }
    return 1;
}

static int microgemm_selftest_gemv_i8(void) {
    enum { ROWS = 5, COLS = 37 };
    int8_t weights[ROWS * COLS];
    float scales[ROWS];
    float input[COLS];
    float bias[ROWS];
    int8_t input_q[COLS];
    int8_t input_q_ref[COLS];
    int32_t row_sums[ROWS];
    float got[ROWS];
    float input_scale;
    int i;
    int j;

    for (i = 0; i < ROWS; ++i) {
        int32_t row_sum = 0;
        scales[i] = 0.003f + 0.0007f * (float)i;
        bias[i] = -0.02f + 0.01f * (float)i;
        for (j = 0; j < COLS; ++j) {
            weights[i * COLS + j] = (int8_t)(((i * 19 + j * 7) % 61) - 30);
            row_sum += (int32_t)weights[i * COLS + j];
        }
        row_sums[i] = row_sum;
    }
    for (j = 0; j < COLS; ++j) {
        input[j] = ((float)((j * 13) % 31) - 15.0f) * 0.017f;
    }

    microgemm_gemv_i8(got, weights, scales, row_sums, input, ROWS, COLS, bias, input_q);
    input_scale = microgemm_selftest_quantize_ref(input_q_ref, input, COLS);

    for (i = 0; i < ROWS; ++i) {
        int32_t acc = 0;
        float ref;
        for (j = 0; j < COLS; ++j) {
            acc += (int32_t)input_q_ref[j] * (int32_t)weights[i * COLS + j];
        }
        ref = (float)acc * scales[i] * input_scale + bias[i];
        if (!microgemm_selftest_close(got[i], ref, 1e-5f, 1e-5f)) {
            return 0;
        }
    }
    return 1;
}

static int microgemm_selftest_gemv_i8_batched(void) {
    enum { MAX_BATCH = 8, ROWS = 7, COLS = 65 };
    const int batches[3] = {1, 4, 8};
    int8_t weights[ROWS * COLS];
    float scales[ROWS];
    int32_t row_sums[ROWS];
    float input[MAX_BATCH * COLS];
    float bias[ROWS];
    float got[MAX_BATCH * ROWS];
    int8_t input_q[MAX_BATCH * COLS];
    float input_scales[MAX_BATCH];
    int8_t input_q_ref[COLS];
    int bi;
    int b;
    int i;
    int j;

    for (i = 0; i < ROWS; ++i) {
        int32_t row_sum = 0;
        scales[i] = 0.0025f + 0.00031f * (float)i;
        bias[i] = -0.013f + 0.006f * (float)i;
        for (j = 0; j < COLS; ++j) {
            int value = ((i * 23 + j * 11) % 25) - 12;
            weights[i * COLS + j] = (int8_t)value;
            row_sum += value;
        }
        row_sums[i] = row_sum;
    }

    for (b = 0; b < MAX_BATCH; ++b) {
        for (j = 0; j < COLS; ++j) {
            int value = (((b + 1) * 17 + j * 13) % 41) - 20;
            input[(size_t)b * COLS + j] =
                (float)value * 0.011f + (float)((j % 5) - 2) * 0.0013f;
        }
    }

    for (bi = 0; bi < 3; ++bi) {
        int batch = batches[bi];
        memset(got, 0, sizeof(got));
        memset(input_q, 0, sizeof(input_q));
        memset(input_scales, 0, sizeof(input_scales));

        microgemm_gemv_i8_batched(
            got,
            weights,
            scales,
            row_sums,
            input,
            batch,
            ROWS,
            COLS,
            COLS,
            bias,
            input_q,
            input_scales
        );

        for (b = 0; b < batch; ++b) {
            float input_scale = microgemm_selftest_quantize_ref(
                input_q_ref,
                input + (size_t)b * COLS,
                COLS
            );
            for (i = 0; i < ROWS; ++i) {
                int32_t acc = 0;
                float ref;
                for (j = 0; j < COLS; ++j) {
                    acc += (int32_t)input_q_ref[j] * (int32_t)weights[i * COLS + j];
                }
                ref = (float)acc * scales[i] * input_scale + bias[i];
                if (!microgemm_selftest_close(
                        got[(size_t)b * ROWS + i],
                        ref,
                        2e-5f,
                        2e-5f)) {
                    return 0;
                }
            }
        }
    }

    return 1;
}

static float microgemm_selftest_i8g_dot_ref(
    const int8_t* weights,
    const float* scales,
    const int8_t* input_q,
    float input_scale,
    int cols
) {
    const int groups = microgemm_quant_group_count_int(cols);
    float value = 0.0f;
    int g;
    for (g = 0; g < groups; ++g) {
        int begin = g * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int32_t acc = 0;
        int j;
        if (end > cols) {
            end = cols;
        }
        for (j = begin; j < end; ++j) {
            acc += (int32_t)input_q[j] * (int32_t)weights[j];
        }
        value += (float)acc * scales[g] * input_scale;
    }
    return value;
}

static float microgemm_selftest_i8_dot_ref(
    const int8_t* weights,
    float scale,
    const int8_t* input_q,
    float input_scale,
    int cols
) {
    int32_t acc = 0;
    int j;
    for (j = 0; j < cols; ++j) {
        acc += (int32_t)input_q[j] * (int32_t)weights[j];
    }
    return (float)acc * scale * input_scale;
}

static uint8_t microgemm_selftest_pack_i4_value(int value) {
    if (value < -7) {
        value = -7;
    }
    if (value > 7) {
        value = 7;
    }
    return (uint8_t)((int8_t)value) & 0x0fu;
}

static void microgemm_selftest_set_i4_weight(uint8_t* row, int col, int value) {
    uint8_t packed = microgemm_selftest_pack_i4_value(value);
    int byte_index = col >> 1;
    if ((col & 1) == 0) {
        row[byte_index] = (uint8_t)((row[byte_index] & 0xf0u) | packed);
    } else {
        row[byte_index] = (uint8_t)((row[byte_index] & 0x0fu) | (uint8_t)(packed << 4));
    }
}

static float microgemm_selftest_i4g_dot_ref(
    const uint8_t* weights,
    const float* scales,
    const int8_t* input_q,
    float input_scale,
    int cols
) {
    const int groups = microgemm_quant_group_count_int(cols);
    float value = 0.0f;
    int g;
    for (g = 0; g < groups; ++g) {
        int begin = g * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
        int32_t acc = 0;
        int j;
        if (end > cols) {
            end = cols;
        }
        for (j = begin; j < end; ++j) {
            acc += (int32_t)input_q[j] * (int32_t)microgemm_i4_row_value(weights, j);
        }
        value += (float)acc * scales[g] * input_scale;
    }
    return value;
}

static int microgemm_selftest_gemv_i8_groupwise_batched(void) {
    enum { MAX_BATCH = 8, ROWS = 129, COLS = 257 };
    const int batches[4] = {1, 4, 5, 8};
    const int groups = microgemm_quant_group_count_int(COLS);
    int8_t weights[ROWS * COLS];
    float scales[ROWS * 3];
    int32_t row_sums[ROWS * 3];
    float input[MAX_BATCH * COLS];
    float bias[ROWS];
    float got[MAX_BATCH * ROWS];
    int8_t input_q[MAX_BATCH * COLS];
    float input_scales[MAX_BATCH];
    int8_t input_q_ref[COLS];
    int bi;
    int b;
    int i;
    int j;
    int g;

    for (i = 0; i < ROWS; ++i) {
        bias[i] = -0.017f + 0.0003f * (float)(i % 23);
        for (g = 0; g < groups; ++g) {
            int begin = g * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            int32_t row_sum = 0;
            if (end > COLS) {
                end = COLS;
            }
            scales[i * groups + g] = 0.0017f + 0.00011f * (float)((i + 3 * g) % 17);
            for (j = begin; j < end; ++j) {
                int value = ((i * 17 + j * 5 + g * 11) % 41) - 20;
                weights[i * COLS + j] = (int8_t)value;
                row_sum += value;
            }
            row_sums[i * groups + g] = row_sum;
        }
    }

    for (b = 0; b < MAX_BATCH; ++b) {
        for (j = 0; j < COLS; ++j) {
            int value = (((b + 5) * 19 + j * 7) % 53) - 26;
            input[(size_t)b * COLS + j] =
                (float)value * 0.009f + (float)((j % 7) - 3) * 0.0011f;
        }
    }

    for (bi = 0; bi < 4; ++bi) {
        int batch = batches[bi];
        memset(got, 0, sizeof(got));
        memset(input_q, 0, sizeof(input_q));
        memset(input_scales, 0, sizeof(input_scales));

        microgemm_gemv_i8_groupwise_batched(
            got,
            weights,
            scales,
            row_sums,
            input,
            batch,
            ROWS,
            COLS,
            COLS,
            bias,
            input_q,
            input_scales
        );

        for (b = 0; b < batch; ++b) {
            float input_scale = microgemm_selftest_quantize_ref(
                input_q_ref,
                input + (size_t)b * COLS,
                COLS
            );
            for (i = 0; i < ROWS; ++i) {
                float ref = microgemm_selftest_i8g_dot_ref(
                    weights + (size_t)i * COLS,
                    scales + (size_t)i * groups,
                    input_q_ref,
                    input_scale,
                    COLS
                ) + bias[i];
                if (!microgemm_selftest_close(
                        got[(size_t)b * ROWS + i],
                        ref,
                        1e-4f,
                        1e-4f)) {
                    return 0;
                }
            }
        }
    }

    return 1;
}

static int microgemm_selftest_gemv_quantized_batched_dispatch(void) {
    enum { MAX_BATCH = 8, ROWS = 129, COLS = 257, GROUPS = 3, ROW_BYTES = (COLS + 1) / 2 };
    const uint32_t modes[3] = {
        MICROGEMM_QUANT_INT8,
        MICROGEMM_QUANT_INT8G128,
        MICROGEMM_QUANT_INT4G128
    };
    const int batches[3] = {1, 4, 8};
    int8_t weights_i8[ROWS * COLS];
    uint8_t weights_i4[ROWS * ROW_BYTES];
    float scales_i8[ROWS];
    int32_t row_sums_i8[ROWS];
    float scales_i8g[ROWS * GROUPS];
    int32_t row_sums_i8g[ROWS * GROUPS];
    float scales_i4g[ROWS * GROUPS];
    int32_t row_sums_i4g[ROWS * GROUPS];
    float input[MAX_BATCH * COLS];
    float bias[ROWS];
    float got[MAX_BATCH * ROWS];
    int8_t input_q[MAX_BATCH * COLS];
    float input_scales[MAX_BATCH];
    int8_t input_q_ref[COLS];
    microgemm_config config;
    int mi;
    int bi;
    int b;
    int i;
    int j;
    int g;

    if (microgemm_quant_group_count_int(COLS) != GROUPS) {
        return 0;
    }

    memset(&config, 0, sizeof(config));
    memset(weights_i4, 0, sizeof(weights_i4));

    for (i = 0; i < ROWS; ++i) {
        int32_t row_sum_i8 = 0;
        bias[i] = -0.011f + 0.0002f * (float)(i % 29);
        scales_i8[i] = 0.0021f + 0.00009f * (float)(i % 31);
        for (j = 0; j < COLS; ++j) {
            int value_i8 = ((i * 17 + j * 5) % 37) - 18;
            weights_i8[i * COLS + j] = (int8_t)value_i8;
            row_sum_i8 += value_i8;
        }
        row_sums_i8[i] = row_sum_i8;

        for (g = 0; g < GROUPS; ++g) {
            int begin = g * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            int32_t sum_i8g = 0;
            int32_t sum_i4g = 0;
            if (end > COLS) {
                end = COLS;
            }
            scales_i8g[i * GROUPS + g] =
                0.0015f + 0.00008f * (float)((i + 7 * g) % 23);
            scales_i4g[i * GROUPS + g] =
                0.0032f + 0.00013f * (float)((i + 5 * g) % 19);
            for (j = begin; j < end; ++j) {
                int value_i4 = ((i * 11 + j * 3 + g * 13) % 15) - 7;
                sum_i8g += (int32_t)weights_i8[i * COLS + j];
                sum_i4g += value_i4;
                microgemm_selftest_set_i4_weight(
                    weights_i4 + (size_t)i * ROW_BYTES,
                    j,
                    value_i4
                );
            }
            row_sums_i8g[i * GROUPS + g] = sum_i8g;
            row_sums_i4g[i * GROUPS + g] = sum_i4g;
        }
    }

    for (b = 0; b < MAX_BATCH; ++b) {
        for (j = 0; j < COLS; ++j) {
            int value = (((b + 3) * 29 + j * 7) % 59) - 29;
            input[(size_t)b * COLS + j] =
                (float)value * 0.0075f + (float)((j % 11) - 5) * 0.0008f;
        }
    }

    for (mi = 0; mi < 3; ++mi) {
        uint32_t mode = modes[mi];
        const int8_t* selected_i8 = weights_i8;
        const uint8_t* selected_i4 = weights_i4;
        const float* selected_scales = scales_i8;
        const int32_t* selected_sums = row_sums_i8;
        config.quant_mode = mode;
        if (mode == MICROGEMM_QUANT_INT8G128) {
            selected_scales = scales_i8g;
            selected_sums = row_sums_i8g;
        } else if (mode == MICROGEMM_QUANT_INT4G128) {
            selected_scales = scales_i4g;
            selected_sums = row_sums_i4g;
        }

        for (bi = 0; bi < 3; ++bi) {
            int batch = batches[bi];
            memset(got, 0, sizeof(got));
            memset(input_q, 0, sizeof(input_q));
            memset(input_scales, 0, sizeof(input_scales));

            microgemm_gemv_quantized_batched(
                &config,
                got,
                selected_i8,
                selected_i4,
                selected_scales,
                selected_sums,
                input,
                batch,
                ROWS,
                COLS,
                COLS,
                bias,
                input_q,
                input_scales
            );

            for (b = 0; b < batch; ++b) {
                float input_scale = microgemm_selftest_quantize_ref(
                    input_q_ref,
                    input + (size_t)b * COLS,
                    COLS
                );
                for (i = 0; i < ROWS; ++i) {
                    float ref;
                    if (mode == MICROGEMM_QUANT_INT8) {
                        ref = microgemm_selftest_i8_dot_ref(
                            weights_i8 + (size_t)i * COLS,
                            scales_i8[i],
                            input_q_ref,
                            input_scale,
                            COLS
                        );
                    } else if (mode == MICROGEMM_QUANT_INT8G128) {
                        ref = microgemm_selftest_i8g_dot_ref(
                            weights_i8 + (size_t)i * COLS,
                            scales_i8g + (size_t)i * GROUPS,
                            input_q_ref,
                            input_scale,
                            COLS
                        );
                    } else {
                        ref = microgemm_selftest_i4g_dot_ref(
                            weights_i4 + (size_t)i * ROW_BYTES,
                            scales_i4g + (size_t)i * GROUPS,
                            input_q_ref,
                            input_scale,
                            COLS
                        );
                    }
                    ref += bias[i];
                    if (!microgemm_selftest_close(
                            got[(size_t)b * ROWS + i],
                            ref,
                            2e-4f,
                            2e-4f)) {
                        return 0;
                    }
                }
            }
        }
    }

    return 1;
}

int microgemm_kernel_i8g_saturation_probe(float* out_got, float* out_ref, float* out_abs_diff) {
    enum { BATCH = 1, ROWS = 1, COLS = 128, GROUPS = 1 };
    int8_t weights[ROWS * COLS];
    float scales[ROWS * GROUPS];
    int32_t row_sums[ROWS * GROUPS];
    float input[BATCH * COLS];
    float got[BATCH * ROWS];
    int8_t input_q[BATCH * COLS];
    float input_scales[BATCH];
    int8_t input_q_ref[COLS];
    float input_scale;
    float ref;
    float diff;
    int j;

    for (j = 0; j < COLS; ++j) {
        weights[j] = 127;
        input[j] = 1.0f;
    }
    scales[0] = 1.0f;
    row_sums[0] = 127 * COLS;
    got[0] = 0.0f;
    memset(input_q, 0, sizeof(input_q));
    memset(input_scales, 0, sizeof(input_scales));

    microgemm_gemv_i8_groupwise_batched(
        got,
        weights,
        scales,
        row_sums,
        input,
        BATCH,
        ROWS,
        COLS,
        COLS,
        NULL,
        input_q,
        input_scales
    );

    input_scale = microgemm_selftest_quantize_ref(input_q_ref, input, COLS);
    ref = microgemm_selftest_i8g_dot_ref(
        weights,
        scales,
        input_q_ref,
        input_scale,
        COLS
    );
    diff = fabsf(got[0] - ref);

    if (out_got != NULL) {
        *out_got = got[0];
    }
    if (out_ref != NULL) {
        *out_ref = ref;
    }
    if (out_abs_diff != NULL) {
        *out_abs_diff = diff;
    }
    return diff > 1.0f;
}

static int microgemm_selftest_gate_up_swiglu_i8_groupwise(void) {
    enum { MAX_BATCH = 8, INTERMEDIATE = 513, OUT_STRIDE = 2 * INTERMEDIATE, COLS = 512 };
    const int batches[2] = {4, 8};
    const int groups = microgemm_quant_group_count_int(COLS);
    const size_t row_count = (size_t)2 * INTERMEDIATE;
    const size_t weight_count = row_count * COLS;
    const size_t scale_count = row_count * groups;
    int8_t* weights = (int8_t*)malloc(weight_count * sizeof(int8_t));
    float* scales = (float*)malloc(scale_count * sizeof(float));
    int32_t* row_sums = (int32_t*)malloc(scale_count * sizeof(int32_t));
    float* input = (float*)malloc((size_t)MAX_BATCH * COLS * sizeof(float));
    float* got = (float*)malloc((size_t)MAX_BATCH * OUT_STRIDE * sizeof(float));
    int8_t* input_q = (int8_t*)malloc((size_t)MAX_BATCH * COLS * sizeof(int8_t));
    float* input_scales = (float*)malloc((size_t)MAX_BATCH * sizeof(float));
    int8_t input_q_ref[COLS];
    microgemm_config config;
    int bi;
    int b;
    int i;
    int j;
    int g;

    if (weights == NULL || scales == NULL || row_sums == NULL
            || input == NULL || got == NULL || input_q == NULL || input_scales == NULL) {
        free(weights);
        free(scales);
        free(row_sums);
        free(input);
        free(got);
        free(input_q);
        free(input_scales);
        return 0;
    }

    memset(&config, 0, sizeof(config));
    config.quant_mode = MICROGEMM_QUANT_INT8G128;

    for (i = 0; i < (int)row_count; ++i) {
        for (g = 0; g < groups; ++g) {
            int begin = g * (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            int end = begin + (int)MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            int32_t row_sum = 0;
            if (end > COLS) {
                end = COLS;
            }
            scales[(size_t)i * groups + g] =
                0.0013f + 0.00007f * (float)((i + 5 * g) % 19);
            for (j = begin; j < end; ++j) {
                int value = ((i * 13 + j * 3 + g * 7) % 39) - 19;
                weights[(size_t)i * COLS + j] = (int8_t)value;
                row_sum += value;
            }
            row_sums[(size_t)i * groups + g] = row_sum;
        }
    }

    for (b = 0; b < MAX_BATCH; ++b) {
        for (j = 0; j < COLS; ++j) {
            int value = (((b + 2) * 23 + j * 11) % 47) - 23;
            input[(size_t)b * COLS + j] =
                (float)value * 0.008f + (float)((j % 9) - 4) * 0.0009f;
        }
    }

    for (bi = 0; bi < 2; ++bi) {
        int batch = batches[bi];
        memset(got, 0, (size_t)MAX_BATCH * OUT_STRIDE * sizeof(float));
        memset(input_q, 0, (size_t)MAX_BATCH * COLS * sizeof(int8_t));
        memset(input_scales, 0, (size_t)MAX_BATCH * sizeof(float));

        microgemm_gate_up_swiglu_groupwise_batched(
            &config,
            got,
            weights,
            NULL,
            scales,
            row_sums,
            input,
            batch,
            INTERMEDIATE,
            COLS,
            COLS,
            OUT_STRIDE,
            0,
            input_q,
            input_scales,
            NULL,
            0
        );

        for (b = 0; b < batch; ++b) {
            float input_scale = microgemm_selftest_quantize_ref(
                input_q_ref,
                input + (size_t)b * COLS,
                COLS
            );
            for (i = 0; i < INTERMEDIATE; ++i) {
                float gate = microgemm_selftest_i8g_dot_ref(
                    weights + (size_t)i * COLS,
                    scales + (size_t)i * groups,
                    input_q_ref,
                    input_scale,
                    COLS
                );
                float up = microgemm_selftest_i8g_dot_ref(
                    weights + (size_t)(INTERMEDIATE + i) * COLS,
                    scales + (size_t)(INTERMEDIATE + i) * groups,
                    input_q_ref,
                    input_scale,
                    COLS
                );
                float ref = microgemm_silu(gate) * up;
                if (!microgemm_selftest_close(
                        got[(size_t)b * OUT_STRIDE + i],
                        ref,
                        2e-4f,
                        2e-4f)) {
                    free(weights);
                    free(scales);
                    free(row_sums);
                    free(input);
                    free(got);
                    free(input_q);
                    free(input_scales);
                    return 0;
                }
            }
        }
    }

    free(weights);
    free(scales);
    free(row_sums);
    free(input);
    free(got);
    free(input_q);
    free(input_scales);
    return 1;
}

static int microgemm_selftest_attention(void) {
    enum {
        NUM_Q = 2,
        NUM_KV = 1,
        HEAD_DIM = 8,
        BLOCK_SIZE = 4,
        SEQ_LEN = 5,
        NUM_BLOCKS = 2
    };
    int block_table[NUM_BLOCKS] = {0, 1};
    int stride_pos = HEAD_DIM;
    int stride_head = BLOCK_SIZE * HEAD_DIM;
    int stride_kv = NUM_KV * stride_head;
    int stride_block = 2 * stride_kv;
    float q[NUM_Q * HEAD_DIM];
    float kv_cache[NUM_BLOCKS * 2 * NUM_KV * BLOCK_SIZE * HEAD_DIM];
    float scores[NUM_Q * SEQ_LEN];
    float got[NUM_Q * HEAD_DIM];
    float ref[NUM_Q * HEAD_DIM];
    int qh;
    int d;
    int pos;

    for (d = 0; d < NUM_Q * HEAD_DIM; ++d) {
        q[d] = ((float)((d * 5) % 17) - 8.0f) * 0.025f;
    }
    for (d = 0; d < (int)(sizeof(kv_cache) / sizeof(kv_cache[0])); ++d) {
        kv_cache[d] = ((float)((d * 3) % 19) - 9.0f) * 0.018f;
    }

    microgemm_attention_decode(
        got,
        scores,
        q,
        kv_cache,
        block_table,
        SEQ_LEN,
        NUM_Q,
        NUM_KV,
        HEAD_DIM,
        BLOCK_SIZE,
        1.0f / sqrtf((float)HEAD_DIM),
        0.0f,
        stride_block,
        stride_kv,
        stride_head,
        stride_pos
    );

    for (qh = 0; qh < NUM_Q; ++qh) {
        float local_scores[SEQ_LEN];
        float max_score = -1e30f;
        float sum_exp = 0.0f;
        int kv_h = 0;
        for (pos = 0; pos < SEQ_LEN; ++pos) {
            int blk_idx = pos / BLOCK_SIZE;
            int blk_off = pos % BLOCK_SIZE;
            int phys_blk = block_table[blk_idx];
            const float* kvec = kv_cache
                + (size_t)phys_blk * stride_block
                + (size_t)kv_h * stride_head
                + (size_t)blk_off * stride_pos;
            float dot = 0.0f;
            for (d = 0; d < HEAD_DIM; ++d) {
                dot += q[qh * HEAD_DIM + d] * kvec[d];
            }
            local_scores[pos] = dot / sqrtf((float)HEAD_DIM);
            if (local_scores[pos] > max_score) {
                max_score = local_scores[pos];
            }
        }
        for (pos = 0; pos < SEQ_LEN; ++pos) {
            local_scores[pos] = expf(local_scores[pos] - max_score);
            sum_exp += local_scores[pos];
        }
        for (d = 0; d < HEAD_DIM; ++d) {
            ref[qh * HEAD_DIM + d] = 0.0f;
        }
        for (pos = 0; pos < SEQ_LEN; ++pos) {
            float w = local_scores[pos] / sum_exp;
            int blk_idx = pos / BLOCK_SIZE;
            int blk_off = pos % BLOCK_SIZE;
            int phys_blk = block_table[blk_idx];
            const float* vvec = kv_cache
                + (size_t)phys_blk * stride_block
                + (size_t)stride_kv
                + (size_t)kv_h * stride_head
                + (size_t)blk_off * stride_pos;
            for (d = 0; d < HEAD_DIM; ++d) {
                ref[qh * HEAD_DIM + d] += w * vvec[d];
            }
        }
    }

    for (d = 0; d < NUM_Q * HEAD_DIM; ++d) {
        if (!microgemm_selftest_close(got[d], ref[d], 1e-5f, 1e-5f)) {
            return 0;
        }
    }
    return 1;
}

static int microgemm_selftest_linear_delta_rule(void) {
    enum {
        KEY_DIM = 5,
        VALUE_DIM = 13
    };
    float qh[KEY_DIM];
    float khv[KEY_DIM];
    float value[VALUE_DIM];
    float state_ref[KEY_DIM * VALUE_DIM];
    float state_got[KEY_DIM * VALUE_DIM];
    float out_ref[VALUE_DIM];
    float out_got[VALUE_DIM];
    float scratch[VALUE_DIM];
    const float beta = 0.37f;
    const float decay = 0.93f;
    const float query_scale = 0.125f;
    int i;

    for (i = 0; i < KEY_DIM; ++i) {
        qh[i] = ((float)((i * 7) % 11) - 5.0f) * 0.071f;
        khv[i] = ((float)((i * 5) % 13) - 6.0f) * 0.053f;
    }
    for (i = 0; i < VALUE_DIM; ++i) {
        value[i] = ((float)((i * 3) % 17) - 8.0f) * 0.041f;
    }
    for (i = 0; i < KEY_DIM * VALUE_DIM; ++i) {
        state_ref[i] = ((float)((i * 19) % 23) - 11.0f) * 0.012f;
        state_got[i] = state_ref[i];
    }

    microgemm_linear_delta_rule_update_head_legacy(
        qh,
        khv,
        value,
        state_ref,
        out_ref,
        KEY_DIM,
        VALUE_DIM,
        beta,
        decay,
        query_scale
    );
    microgemm_linear_delta_rule_update_head(
        qh,
        khv,
        value,
        state_got,
        out_got,
        scratch,
        KEY_DIM,
        VALUE_DIM,
        beta,
        decay,
        query_scale
    );

    for (i = 0; i < VALUE_DIM; ++i) {
        if (!microgemm_selftest_close(out_got[i], out_ref[i], 1e-6f, 1e-5f)) {
            return 0;
        }
    }
    for (i = 0; i < KEY_DIM * VALUE_DIM; ++i) {
        if (!microgemm_selftest_close(state_got[i], state_ref[i], 1e-6f, 1e-5f)) {
            return 0;
        }
    }
    return 1;
}

int microgemm_kernel_selftest(void) {
    if (!microgemm_selftest_quantize()) {
        return 1;
    }
    if (!microgemm_selftest_rmsnorm()) {
        return 2;
    }
    if (!microgemm_selftest_gemv_i8()) {
        return 3;
    }
    if (!microgemm_selftest_gemv_i8_batched()) {
        return 4;
    }
    if (!microgemm_selftest_gemv_i8_groupwise_batched()) {
        return 5;
    }
    if (!microgemm_selftest_gemv_quantized_batched_dispatch()) {
        return 6;
    }
    if (!microgemm_selftest_gate_up_swiglu_i8_groupwise()) {
        return 7;
    }
    if (!microgemm_selftest_attention()) {
        return 8;
    }
    if (!microgemm_selftest_linear_delta_rule()) {
        return 9;
    }
    return 0;
}

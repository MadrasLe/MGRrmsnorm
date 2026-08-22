#ifndef MICROGEMM_DECODE_H
#define MICROGEMM_DECODE_H

#include <stddef.h>
#include <stdint.h>

#include "microgemm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct microgemm_layer_weights_i8 {
    uint32_t layer_type;

    const float* input_norm_w;
    const float* post_attn_norm_w;

    const int8_t* qkv_w;
    const uint8_t* qkv_w_i4;
    const float* qkv_s;
    const int32_t* qkv_row_sums;
    const float* qkv_bias;
    const float* q_norm_w;
    const float* k_norm_w;

    const int8_t* o_w;
    const uint8_t* o_w_i4;
    const float* o_s;
    const int32_t* o_row_sums;
    const float* attn_output_norm_w;

    const int8_t* linear_qkv_w;
    const uint8_t* linear_qkv_w_i4;
    const float* linear_qkv_s;
    const int32_t* linear_qkv_row_sums;
    const int8_t* linear_baz_w;
    const uint8_t* linear_baz_w_i4;
    const float* linear_baz_s;
    const int32_t* linear_baz_row_sums;
    const float* linear_conv_w;
    const float* linear_dt_bias;
    const float* linear_a_log;
    const float* linear_norm_w;
    const int8_t* linear_out_w;
    const uint8_t* linear_out_w_i4;
    const float* linear_out_s;
    const int32_t* linear_out_row_sums;

    const int8_t* gate_up_w;
    const uint8_t* gate_up_w_i4;
    const float* gate_up_s;
    const int32_t* gate_up_row_sums;

    const int8_t* down_w;
    const uint8_t* down_w_i4;
    const float* down_s;
    const int32_t* down_row_sums;
    const float* mlp_output_norm_w;
} microgemm_layer_weights_i8;

typedef struct microgemm_model_weights_i8 {
    const float* embed_tokens;
    const int8_t* embed_tokens_i8;
    const uint8_t* embed_tokens_i4;
    const float* embed_tokens_s;
    const microgemm_layer_weights_i8* layers;
    const float* final_norm_w;

    const int8_t* lm_head_w;
    const uint8_t* lm_head_w_i4;
    const float* lm_head_s;
    const int32_t* lm_head_row_sums;

    const float* cos_cache;
    const float* sin_cache;
} microgemm_model_weights_i8;

typedef struct microgemm_kv_layout {
    float** layer_kv;
    const int* block_table;
    int seq_len;

    int stride_block;
    int stride_kv;
    int stride_head;
    int stride_pos;
} microgemm_kv_layout;

typedef struct microgemm_decode_batch_profile {
    uint64_t calls;
    uint64_t tokens;
    double total_ms;
    double alloc_ms;
    double embed_ms;
    double input_norm_ms;
    double qkv_ms;
    double rope_kv_ms;
    double attention_ms;
    double o_proj_ms;
    double post_norm_ms;
    double gate_up_ms;
    double gate_up_quant_ms;
    double gate_up_dot_ms;
    double activation_ms;
    double down_proj_ms;
    double down_proj_quant_ms;
    double down_proj_dot_ms;
    double final_norm_ms;
    double lm_head_ms;
    double copy_ms;
    double cleanup_ms;
    uint64_t groupwise_gemv_tile_calls;
    uint64_t groupwise_i8_row_pair_calls;
    uint64_t groupwise_i4_row_pair_calls;
    uint64_t groupwise_lm_head_argmax_calls;
    uint64_t lm_head_stack_best_calls;
    uint64_t groupwise_gate_up_fused_calls;
    uint64_t groupwise_i8_gate_safe_combined_calls;
    uint64_t groupwise_i8_gate_safe_combined_tile8_calls;
    uint64_t groupwise_i8_gate_tile8_calls;
    uint64_t groupwise_i8_gate_biased_calls;
    uint64_t groupwise_i8_gate_pair_calls;
    uint64_t groupwise_i8_gate_pair_unroll64_calls;
    uint64_t groupwise_i8_gate_pair_unroll128_calls;
    uint64_t groupwise_i8_gate_pair8_split_calls;
    uint64_t groupwise_i8_gate_prefetch_calls;
    uint64_t groupwise_lm_head_row_pair_calls;
} microgemm_decode_batch_profile;

typedef struct microgemm_decode_workspace microgemm_decode_workspace;
typedef struct microgemm_loaded_model_i8 microgemm_loaded_model_i8;

microgemm_status microgemm_loaded_model_i8_create(
    const microgemm_model* model,
    microgemm_loaded_model_i8** out_loaded
);

void microgemm_loaded_model_i8_destroy(microgemm_loaded_model_i8* loaded);

const microgemm_model_weights_i8* microgemm_loaded_model_i8_weights(
    const microgemm_loaded_model_i8* loaded
);

const microgemm_config* microgemm_loaded_model_i8_config(
    const microgemm_loaded_model_i8* loaded
);

size_t microgemm_loaded_model_i8_bytes(const microgemm_loaded_model_i8* loaded);

microgemm_status microgemm_decode_workspace_create(
    const microgemm_config* config,
    uint32_t max_seq_len,
    microgemm_decode_workspace** out_workspace
);

void microgemm_decode_workspace_destroy(microgemm_decode_workspace* workspace);

size_t microgemm_decode_workspace_bytes(const microgemm_decode_workspace* workspace);

void microgemm_decode_batch_profile_set_enabled(int enabled);
void microgemm_decode_batch_profile_reset(void);
void microgemm_decode_batch_profile_get(microgemm_decode_batch_profile* out_profile);

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
);

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
);

microgemm_status microgemm_decode_step_i8_batch_next_token(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* const* workspaces,
    const int* token_ids,
    const int* positions,
    const microgemm_kv_layout* const* kvs,
    size_t batch,
    int* out_token_ids
);

microgemm_status microgemm_decode_logits_i8_batch(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* const* workspaces,
    size_t batch,
    float* const* logits_out,
    int logits_capacity
);

microgemm_status microgemm_decode_prefill_i8(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* workspace,
    const int* token_ids,
    size_t token_count,
    microgemm_kv_layout* kv,
    float* logits_out,
    int logits_capacity
);

microgemm_status microgemm_decode_prefill_i8_next_token(
    const microgemm_config* config,
    const microgemm_model_weights_i8* model,
    microgemm_decode_workspace* workspace,
    const int* token_ids,
    size_t token_count,
    microgemm_kv_layout* kv,
    int* out_token_id
);

#ifdef __cplusplus
}
#endif

#endif

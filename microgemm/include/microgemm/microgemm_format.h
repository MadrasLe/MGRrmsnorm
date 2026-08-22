#ifndef MICROGEMM_FORMAT_H
#define MICROGEMM_FORMAT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MICROGEMM_MAGIC_0 'M'
#define MICROGEMM_MAGIC_1 'G'
#define MICROGEMM_MAGIC_2 'M'
#define MICROGEMM_MAGIC_3 '1'

#define MICROGEMM_MAX_TENSOR_NAME 64
#define MICROGEMM_MAX_TENSOR_RANK 4

typedef enum microgemm_architecture {
    MICROGEMM_ARCH_UNKNOWN = 0,
    MICROGEMM_ARCH_LLAMA_LIKE = 1,
    MICROGEMM_ARCH_QWEN2_LIKE = 2,
    MICROGEMM_ARCH_MISTRAL_LIKE = 3,
    MICROGEMM_ARCH_GEMMA_LIKE = 4,
    MICROGEMM_ARCH_QWEN35_LIKE = 5,
    MICROGEMM_ARCH_PHI_LIKE = 6,
    MICROGEMM_ARCH_GRANITE_LIKE = 7,
    MICROGEMM_ARCH_GLM4_LIKE = 8
} microgemm_architecture;

typedef enum microgemm_layer_type {
    MICROGEMM_LAYER_FULL_ATTENTION = 0,
    MICROGEMM_LAYER_LINEAR_ATTENTION = 1
} microgemm_layer_type;

typedef enum microgemm_dtype {
    MICROGEMM_DTYPE_UNKNOWN = 0,
    MICROGEMM_DTYPE_F32 = 1,
    MICROGEMM_DTYPE_F16 = 2,
    MICROGEMM_DTYPE_BF16 = 3,
    MICROGEMM_DTYPE_I8 = 4,
    MICROGEMM_DTYPE_U8 = 5,
    MICROGEMM_DTYPE_I4 = 6,
    MICROGEMM_DTYPE_I32 = 7
} microgemm_dtype;

typedef enum microgemm_quant_mode {
    MICROGEMM_QUANT_NONE = 0,
    MICROGEMM_QUANT_INT8 = 1,
    MICROGEMM_QUANT_INT4 = 2,
    MICROGEMM_QUANT_INT8G128 = 3,
    MICROGEMM_QUANT_INT4G128 = 4
} microgemm_quant_mode;

#define MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE 128u

#define MICROGEMM_FLAG_QKV_BIAS   (1u << 0)
#define MICROGEMM_FLAG_NORM_OFFSET (1u << 1)
#define MICROGEMM_FLAG_MLP_GELU   (1u << 2)
#define MICROGEMM_FLAG_QK_NORM    (1u << 3)
#define MICROGEMM_FLAG_ATTN_OUTPUT_NORM (1u << 4)
#define MICROGEMM_FLAG_MLP_OUTPUT_NORM  (1u << 5)
#define MICROGEMM_FLAG_ATTN_OUTPUT_GATE (1u << 6)
#define MICROGEMM_FLAG_PARTIAL_ROPE     (1u << 7)
#define MICROGEMM_FLAG_ROPE_INTERLEAVED (1u << 8)

typedef struct microgemm_config {
    uint32_t architecture;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_layers;
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t max_position_embeddings;
    uint32_t kv_block_size;
    uint32_t quant_mode;
    uint32_t flags;
    float rms_norm_eps;
    float rope_theta;
    float attention_logit_softcap;
    float final_logit_softcap;
    float query_pre_attn_scalar;
    uint32_t qkv_rows;
    uint32_t attn_width;
    uint32_t rotary_dim;
    uint32_t linear_key_head_dim;
    uint32_t linear_value_head_dim;
    uint32_t linear_num_key_heads;
    uint32_t linear_num_value_heads;
    uint32_t linear_conv_kernel_dim;
    float embedding_multiplier;
    float residual_multiplier;
    float logits_scaling;
} microgemm_config;

typedef struct microgemm_file_header {
    uint8_t magic[4];
    uint16_t version_major;
    uint16_t version_minor;
    uint32_t header_bytes;
    uint32_t config_bytes;
    uint32_t tensor_entry_bytes;
    uint32_t reserved0;
    uint64_t tensor_count;
    uint64_t tensor_directory_offset;
    uint64_t tensor_data_offset;
} microgemm_file_header;

typedef struct microgemm_tensor_entry {
    char name[MICROGEMM_MAX_TENSOR_NAME];
    uint32_t dtype;
    uint32_t storage_flags;
    uint32_t rank;
    uint32_t reserved0;
    uint64_t dims[MICROGEMM_MAX_TENSOR_RANK];
    uint64_t offset;
    uint64_t byte_length;
} microgemm_tensor_entry;

int microgemm_format_validate_header(const microgemm_file_header* header);
const char* microgemm_format_architecture_name(uint32_t architecture);
const char* microgemm_format_dtype_name(uint32_t dtype);
const char* microgemm_format_quant_name(uint32_t quant_mode);

#ifdef __cplusplus
}
#endif

#endif

#include "microgemm/microgemm_decode.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct microgemm_loaded_model_i8 {
    microgemm_config config;
    microgemm_model_weights_i8 weights;
    microgemm_layer_weights_i8* layers;

    void** owned_allocations;
    size_t allocation_count;
    size_t allocation_capacity;
    size_t total_bytes;
};

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

static uint64_t microgemm_quant_group_count(uint64_t cols) {
    return (cols + MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE - 1u)
        / MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
}

static void microgemm_loaded_model_i8_reset(microgemm_loaded_model_i8* loaded) {
    size_t i;
    if (loaded == NULL) {
        return;
    }
    if (loaded->owned_allocations != NULL) {
        for (i = 0; i < loaded->allocation_count; ++i) {
            free(loaded->owned_allocations[i]);
        }
    }
    free(loaded->owned_allocations);
    free(loaded->layers);
    memset(loaded, 0, sizeof(*loaded));
}

static microgemm_status microgemm_push_owned_allocation(
    microgemm_loaded_model_i8* loaded,
    void* ptr,
    size_t num_bytes
) {
    if (loaded == NULL || ptr == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (loaded->allocation_count >= loaded->allocation_capacity) {
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }
    loaded->owned_allocations[loaded->allocation_count++] = ptr;
    loaded->total_bytes += num_bytes;
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_load_tensor_raw(
    const microgemm_model* model,
    microgemm_loaded_model_i8* loaded,
    const char* name,
    uint32_t expected_dtype,
    uint32_t expected_rank,
    const uint64_t* expected_dims,
    int required,
    void** out_ptr,
    size_t* out_bytes
) {
    const microgemm_tensor_entry* tensor;
    void* buf;
    uint32_t dim_idx;
    microgemm_status status;

    if (model == NULL || loaded == NULL || name == NULL || out_ptr == NULL || out_bytes == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    *out_ptr = NULL;
    *out_bytes = 0;

    tensor = microgemm_model_find_tensor(model, name);
    if (tensor == NULL) {
        return required ? MICROGEMM_STATUS_FORMAT_ERROR : MICROGEMM_STATUS_OK;
    }

    if (tensor->dtype != expected_dtype || tensor->rank != expected_rank) {
        return MICROGEMM_STATUS_FORMAT_ERROR;
    }
    for (dim_idx = 0; dim_idx < expected_rank; ++dim_idx) {
        if (tensor->dims[dim_idx] != expected_dims[dim_idx]) {
            return MICROGEMM_STATUS_FORMAT_ERROR;
        }
    }

    buf = malloc((size_t)tensor->byte_length);
    if (buf == NULL) {
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }

    status = microgemm_model_read_tensor(model, tensor, buf, (size_t)tensor->byte_length);
    if (status != MICROGEMM_STATUS_OK) {
        free(buf);
        return status;
    }

    status = microgemm_push_owned_allocation(loaded, buf, (size_t)tensor->byte_length);
    if (status != MICROGEMM_STATUS_OK) {
        free(buf);
        return status;
    }

    *out_ptr = buf;
    *out_bytes = (size_t)tensor->byte_length;
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_load_f32_1d(
    const microgemm_model* model,
    microgemm_loaded_model_i8* loaded,
    const char* name,
    uint64_t dim0,
    const float** out_ptr
) {
    const uint64_t dims[1] = {dim0};
    void* ptr = NULL;
    size_t bytes = 0;
    microgemm_status status = microgemm_load_tensor_raw(
        model, loaded, name, MICROGEMM_DTYPE_F32, 1, dims, 1, &ptr, &bytes
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }
    if (bytes != (size_t)dim0 * sizeof(float)) {
        return MICROGEMM_STATUS_FORMAT_ERROR;
    }
    *out_ptr = (const float*)ptr;
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_load_f32_2d(
    const microgemm_model* model,
    microgemm_loaded_model_i8* loaded,
    const char* name,
    uint64_t dim0,
    uint64_t dim1,
    const float** out_ptr
) {
    const uint64_t dims[2] = {dim0, dim1};
    void* ptr = NULL;
    size_t bytes = 0;
    microgemm_status status = microgemm_load_tensor_raw(
        model, loaded, name, MICROGEMM_DTYPE_F32, 2, dims, 1, &ptr, &bytes
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }
    if (bytes != (size_t)(dim0 * dim1) * sizeof(float)) {
        return MICROGEMM_STATUS_FORMAT_ERROR;
    }
    *out_ptr = (const float*)ptr;
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_load_i8_2d(
    const microgemm_model* model,
    microgemm_loaded_model_i8* loaded,
    const char* name,
    uint64_t dim0,
    uint64_t dim1,
    const int8_t** out_ptr
) {
    const uint64_t dims[2] = {dim0, dim1};
    void* ptr = NULL;
    size_t bytes = 0;
    microgemm_status status = microgemm_load_tensor_raw(
        model, loaded, name, MICROGEMM_DTYPE_I8, 2, dims, 1, &ptr, &bytes
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }
    if (bytes != (size_t)(dim0 * dim1) * sizeof(int8_t)) {
        return MICROGEMM_STATUS_FORMAT_ERROR;
    }
    *out_ptr = (const int8_t*)ptr;
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_load_optional_i32_1d(
    const microgemm_model* model,
    microgemm_loaded_model_i8* loaded,
    const char* name,
    uint64_t dim0,
    const int32_t** out_ptr
) {
    const uint64_t dims[1] = {dim0};
    void* ptr = NULL;
    size_t bytes = 0;
    microgemm_status status = microgemm_load_tensor_raw(
        model, loaded, name, MICROGEMM_DTYPE_I32, 1, dims, 0, &ptr, &bytes
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }
    if (ptr != NULL && bytes != (size_t)dim0 * sizeof(int32_t)) {
        return MICROGEMM_STATUS_FORMAT_ERROR;
    }
    *out_ptr = (const int32_t*)ptr;
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_load_optional_i32_2d(
    const microgemm_model* model,
    microgemm_loaded_model_i8* loaded,
    const char* name,
    uint64_t dim0,
    uint64_t dim1,
    const int32_t** out_ptr
) {
    const uint64_t dims[2] = {dim0, dim1};
    void* ptr = NULL;
    size_t bytes = 0;
    microgemm_status status = microgemm_load_tensor_raw(
        model, loaded, name, MICROGEMM_DTYPE_I32, 2, dims, 0, &ptr, &bytes
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }
    if (ptr != NULL && bytes != (size_t)(dim0 * dim1) * sizeof(int32_t)) {
        return MICROGEMM_STATUS_FORMAT_ERROR;
    }
    *out_ptr = (const int32_t*)ptr;
    return MICROGEMM_STATUS_OK;
}

static uint64_t microgemm_full_q_rows(const microgemm_config* cfg) {
    uint64_t rows;
    if (cfg == NULL) {
        return 0u;
    }
    rows = (uint64_t)cfg->num_q_heads * cfg->head_dim;
    if ((cfg->flags & MICROGEMM_FLAG_ATTN_OUTPUT_GATE) != 0u) {
        rows *= 2u;
    }
    return rows;
}

static uint64_t microgemm_full_qkv_rows(const microgemm_config* cfg) {
    if (cfg == NULL) {
        return 0u;
    }
    return microgemm_full_q_rows(cfg)
        + (uint64_t)(2u * cfg->num_kv_heads) * cfg->head_dim;
}

static uint32_t microgemm_linear_key_head_dim(const microgemm_config* cfg) {
    return cfg != NULL && cfg->linear_key_head_dim != 0u ? cfg->linear_key_head_dim : (cfg != NULL ? cfg->head_dim : 0u);
}

static uint32_t microgemm_linear_value_head_dim(const microgemm_config* cfg) {
    return cfg != NULL && cfg->linear_value_head_dim != 0u ? cfg->linear_value_head_dim : (cfg != NULL ? cfg->head_dim : 0u);
}

static uint32_t microgemm_linear_num_key_heads(const microgemm_config* cfg) {
    return cfg != NULL && cfg->linear_num_key_heads != 0u ? cfg->linear_num_key_heads : (cfg != NULL ? cfg->num_q_heads : 0u);
}

static uint32_t microgemm_linear_num_value_heads(const microgemm_config* cfg) {
    return cfg != NULL && cfg->linear_num_value_heads != 0u ? cfg->linear_num_value_heads : microgemm_linear_num_key_heads(cfg);
}

static uint64_t microgemm_linear_key_dim(const microgemm_config* cfg) {
    return (uint64_t)microgemm_linear_num_key_heads(cfg) * microgemm_linear_key_head_dim(cfg);
}

static uint64_t microgemm_linear_value_dim(const microgemm_config* cfg) {
    return (uint64_t)microgemm_linear_num_value_heads(cfg) * microgemm_linear_value_head_dim(cfg);
}

static uint64_t microgemm_linear_conv_dim(const microgemm_config* cfg) {
    return 2u * microgemm_linear_key_dim(cfg) + microgemm_linear_value_dim(cfg);
}

static uint64_t microgemm_linear_baz_rows(const microgemm_config* cfg) {
    return microgemm_linear_value_dim(cfg) + 2u * microgemm_linear_num_value_heads(cfg);
}

static int8_t microgemm_unpack_i4_nibble(uint8_t packed, int high_nibble) {
    uint8_t nibble = high_nibble ? (uint8_t)(packed >> 4) : (uint8_t)(packed & 0x0fu);
    return (int8_t)(nibble >= 8u ? (int)nibble - 16 : (int)nibble);
}

static int8_t microgemm_i4_row_value(const uint8_t* row, uint64_t col) {
    return microgemm_unpack_i4_nibble(row[col / 2u], (int)(col & 1u));
}

static microgemm_status microgemm_load_i4_2d(
    const microgemm_model* model,
    microgemm_loaded_model_i8* loaded,
    const char* name,
    uint64_t dim0,
    uint64_t dim1,
    const uint8_t** out_ptr
) {
    const uint64_t dims[2] = {dim0, dim1};
    void* ptr = NULL;
    size_t bytes = 0;
    uint64_t row_bytes;
    microgemm_status status = microgemm_load_tensor_raw(
        model, loaded, name, MICROGEMM_DTYPE_I4, 2, dims, 1, &ptr, &bytes
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }
    if (dim0 == 0u || dim1 == 0u) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    row_bytes = (dim1 + 1u) / 2u;
    if (dim0 > ((uint64_t)-1) / row_bytes) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (bytes != (size_t)(dim0 * row_bytes)) {
        return MICROGEMM_STATUS_FORMAT_ERROR;
    }
    *out_ptr = (const uint8_t*)ptr;
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_load_quantized_2d(
    const microgemm_model* model,
    microgemm_loaded_model_i8* loaded,
    const char* name_i8,
    const char* name_i4,
    uint64_t dim0,
    uint64_t dim1,
    const int8_t** out_i8,
    const uint8_t** out_i4
) {
    if (loaded == NULL || out_i8 == NULL || out_i4 == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    *out_i8 = NULL;
    *out_i4 = NULL;
    if (microgemm_quant_mode_is_i4_storage(loaded->config.quant_mode)) {
        return microgemm_load_i4_2d(model, loaded, name_i4, dim0, dim1, out_i4);
    }
    if (microgemm_quant_mode_is_i8_storage(loaded->config.quant_mode)) {
        return microgemm_load_i8_2d(model, loaded, name_i8, dim0, dim1, out_i8);
    }
    return MICROGEMM_STATUS_UNSUPPORTED;
}

static microgemm_status microgemm_compute_i8_row_sums(
    microgemm_loaded_model_i8* loaded,
    const int8_t* weights,
    uint64_t rows,
    uint64_t cols,
    const int32_t** out_ptr
) {
    int32_t* sums;
    uint64_t row;

    if (loaded == NULL || weights == NULL || out_ptr == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0u || cols == 0u || rows > ((uint64_t)-1) / sizeof(int32_t)) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    sums = (int32_t*)malloc((size_t)rows * sizeof(int32_t));
    if (sums == NULL) {
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }

    for (row = 0u; row < rows; ++row) {
        const int8_t* w = weights + (size_t)(row * cols);
        int64_t sum = 0;
        uint64_t col;
        for (col = 0u; col < cols; ++col) {
            sum += (int32_t)w[col];
        }
        sums[row] = (int32_t)sum;
    }

    {
        microgemm_status status = microgemm_push_owned_allocation(
            loaded,
            sums,
            (size_t)rows * sizeof(int32_t)
        );
        if (status != MICROGEMM_STATUS_OK) {
            free(sums);
            return status;
        }
    }

    *out_ptr = sums;
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_compute_i4_row_sums(
    microgemm_loaded_model_i8* loaded,
    const uint8_t* weights,
    uint64_t rows,
    uint64_t cols,
    const int32_t** out_ptr
) {
    int32_t* sums;
    uint64_t row;
    uint64_t row_bytes;

    if (loaded == NULL || weights == NULL || out_ptr == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0u || cols == 0u || rows > ((uint64_t)-1) / sizeof(int32_t)) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    row_bytes = (cols + 1u) / 2u;
    sums = (int32_t*)malloc((size_t)rows * sizeof(int32_t));
    if (sums == NULL) {
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }

    for (row = 0u; row < rows; ++row) {
        const uint8_t* w = weights + (size_t)(row * row_bytes);
        int64_t sum = 0;
        uint64_t col;
        for (col = 0u; col < cols; ++col) {
            sum += (int32_t)microgemm_i4_row_value(w, col);
        }
        sums[row] = (int32_t)sum;
    }

    {
        microgemm_status status = microgemm_push_owned_allocation(
            loaded,
            sums,
            (size_t)rows * sizeof(int32_t)
        );
        if (status != MICROGEMM_STATUS_OK) {
            free(sums);
            return status;
        }
    }

    *out_ptr = sums;
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_compute_i8_group_sums(
    microgemm_loaded_model_i8* loaded,
    const int8_t* weights,
    uint64_t rows,
    uint64_t cols,
    const int32_t** out_ptr
) {
    const uint64_t groups = microgemm_quant_group_count(cols);
    int32_t* sums;

    if (loaded == NULL || weights == NULL || out_ptr == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0u || cols == 0u || groups == 0u
            || rows > ((uint64_t)-1) / groups
            || rows * groups > ((uint64_t)-1) / sizeof(int32_t)) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    sums = (int32_t*)malloc((size_t)(rows * groups) * sizeof(int32_t));
    if (sums == NULL) {
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }

    for (uint64_t row = 0u; row < rows; ++row) {
        const int8_t* w = weights + (size_t)(row * cols);
        for (uint64_t group = 0u; group < groups; ++group) {
            const uint64_t begin = group * MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            uint64_t end = begin + MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            int64_t sum = 0;
            if (end > cols) {
                end = cols;
            }
            for (uint64_t col = begin; col < end; ++col) {
                sum += (int32_t)w[col];
            }
            sums[(size_t)(row * groups + group)] = (int32_t)sum;
        }
    }

    {
        microgemm_status status = microgemm_push_owned_allocation(
            loaded,
            sums,
            (size_t)(rows * groups) * sizeof(int32_t)
        );
        if (status != MICROGEMM_STATUS_OK) {
            free(sums);
            return status;
        }
    }

    *out_ptr = sums;
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_compute_i4_group_sums(
    microgemm_loaded_model_i8* loaded,
    const uint8_t* weights,
    uint64_t rows,
    uint64_t cols,
    const int32_t** out_ptr
) {
    const uint64_t groups = microgemm_quant_group_count(cols);
    const uint64_t row_bytes = (cols + 1u) / 2u;
    int32_t* sums;

    if (loaded == NULL || weights == NULL || out_ptr == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (rows == 0u || cols == 0u || groups == 0u
            || rows > ((uint64_t)-1) / groups
            || rows * groups > ((uint64_t)-1) / sizeof(int32_t)) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    sums = (int32_t*)malloc((size_t)(rows * groups) * sizeof(int32_t));
    if (sums == NULL) {
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }

    for (uint64_t row = 0u; row < rows; ++row) {
        const uint8_t* w = weights + (size_t)(row * row_bytes);
        for (uint64_t group = 0u; group < groups; ++group) {
            const uint64_t begin = group * MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            uint64_t end = begin + MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            int64_t sum = 0;
            if (end > cols) {
                end = cols;
            }
            for (uint64_t col = begin; col < end; ++col) {
                sum += (int32_t)microgemm_i4_row_value(w, col);
            }
            sums[(size_t)(row * groups + group)] = (int32_t)sum;
        }
    }

    {
        microgemm_status status = microgemm_push_owned_allocation(
            loaded,
            sums,
            (size_t)(rows * groups) * sizeof(int32_t)
        );
        if (status != MICROGEMM_STATUS_OK) {
            free(sums);
            return status;
        }
    }

    *out_ptr = sums;
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_compute_quantized_row_sums(
    microgemm_loaded_model_i8* loaded,
    const int8_t* weights_i8,
    const uint8_t* weights_i4,
    uint64_t rows,
    uint64_t cols,
    const int32_t** out_ptr
) {
    if (loaded == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (loaded->config.quant_mode == MICROGEMM_QUANT_INT4G128) {
        return microgemm_compute_i4_group_sums(loaded, weights_i4, rows, cols, out_ptr);
    }
    if (loaded->config.quant_mode == MICROGEMM_QUANT_INT8G128) {
        return microgemm_compute_i8_group_sums(loaded, weights_i8, rows, cols, out_ptr);
    }
    if (loaded->config.quant_mode == MICROGEMM_QUANT_INT4) {
        return microgemm_compute_i4_row_sums(loaded, weights_i4, rows, cols, out_ptr);
    }
    if (loaded->config.quant_mode == MICROGEMM_QUANT_INT8) {
        return microgemm_compute_i8_row_sums(loaded, weights_i8, rows, cols, out_ptr);
    }
    return MICROGEMM_STATUS_UNSUPPORTED;
}

static microgemm_status microgemm_load_or_compute_quantized_row_sums(
    const microgemm_model* model,
    microgemm_loaded_model_i8* loaded,
    const char* name,
    const int8_t* weights_i8,
    const uint8_t* weights_i4,
    uint64_t rows,
    uint64_t cols,
    const int32_t** out_ptr
) {
    microgemm_status status;
    if (out_ptr == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    *out_ptr = NULL;
    if (microgemm_quant_mode_is_groupwise(loaded->config.quant_mode)) {
        status = microgemm_load_optional_i32_2d(
            model, loaded, name, rows, microgemm_quant_group_count(cols), out_ptr
        );
    } else {
        status = microgemm_load_optional_i32_1d(model, loaded, name, rows, out_ptr);
    }
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }
    if (*out_ptr != NULL) {
        (void)cols;
        (void)weights_i8;
        (void)weights_i4;
        return MICROGEMM_STATUS_OK;
    }
    return microgemm_compute_quantized_row_sums(
        loaded, weights_i8, weights_i4, rows, cols, out_ptr
    );
}

static microgemm_status microgemm_load_optional_f32_1d(
    const microgemm_model* model,
    microgemm_loaded_model_i8* loaded,
    const char* name,
    uint64_t dim0,
    const float** out_ptr
) {
    const uint64_t dims[1] = {dim0};
    void* ptr = NULL;
    size_t bytes = 0;
    microgemm_status status = microgemm_load_tensor_raw(
        model, loaded, name, MICROGEMM_DTYPE_F32, 1, dims, 0, &ptr, &bytes
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }
    if (ptr != NULL && bytes != (size_t)dim0 * sizeof(float)) {
        return MICROGEMM_STATUS_FORMAT_ERROR;
    }
    *out_ptr = (const float*)ptr;
    return MICROGEMM_STATUS_OK;
}

static microgemm_status microgemm_load_quantized_scales(
    const microgemm_model* model,
    microgemm_loaded_model_i8* loaded,
    const char* name,
    uint64_t rows,
    uint64_t cols,
    const float** out_ptr
) {
    if (loaded == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (microgemm_quant_mode_is_groupwise(loaded->config.quant_mode)) {
        return microgemm_load_f32_2d(
            model,
            loaded,
            name,
            rows,
            microgemm_quant_group_count(cols),
            out_ptr
        );
    }
    return microgemm_load_f32_1d(model, loaded, name, rows, out_ptr);
}

static microgemm_status microgemm_load_layer_i8(
    const microgemm_model* model,
    microgemm_loaded_model_i8* loaded,
    uint32_t layer_idx
) {
    char name[96];
    char name_i4[96];
    microgemm_layer_weights_i8* layer = &loaded->layers[layer_idx];
    const microgemm_config* cfg = &loaded->config;
    uint64_t hidden = cfg->hidden_size;
    uint64_t inter = cfg->intermediate_size;
    uint64_t qkv_rows = microgemm_full_qkv_rows(cfg);
    uint64_t attn_cols = cfg->attn_width != 0u
        ? cfg->attn_width
        : (uint64_t)cfg->num_q_heads * cfg->head_dim;
    microgemm_status status;
    const int32_t* layer_type = NULL;

    layer->layer_type = MICROGEMM_LAYER_FULL_ATTENTION;
    snprintf(name, sizeof(name), "layers.%u.type", layer_idx);
    status = microgemm_load_optional_i32_1d(model, loaded, name, 1u, &layer_type);
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }
    if (layer_type != NULL) {
        if (layer_type[0] != MICROGEMM_LAYER_FULL_ATTENTION
                && layer_type[0] != MICROGEMM_LAYER_LINEAR_ATTENTION) {
            return MICROGEMM_STATUS_FORMAT_ERROR;
        }
        layer->layer_type = (uint32_t)layer_type[0];
    }

    snprintf(name, sizeof(name), "layers.%u.input_norm.weight", layer_idx);
    status = microgemm_load_f32_1d(model, loaded, name, hidden, &layer->input_norm_w);
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }

    snprintf(name, sizeof(name), "layers.%u.post_norm.weight", layer_idx);
    status = microgemm_load_f32_1d(model, loaded, name, hidden, &layer->post_attn_norm_w);
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }

    if (layer->layer_type == MICROGEMM_LAYER_LINEAR_ATTENTION) {
        uint64_t conv_dim = microgemm_linear_conv_dim(cfg);
        uint64_t baz_rows = microgemm_linear_baz_rows(cfg);
        uint64_t value_dim = microgemm_linear_value_dim(cfg);
        uint64_t value_head_dim = microgemm_linear_value_head_dim(cfg);
        uint64_t value_heads = microgemm_linear_num_value_heads(cfg);

        snprintf(name, sizeof(name), "layers.%u.linear_qkv.weight_i8", layer_idx);
        snprintf(name_i4, sizeof(name_i4), "layers.%u.linear_qkv.weight_i4", layer_idx);
        status = microgemm_load_quantized_2d(
            model, loaded, name, name_i4, conv_dim, hidden,
            &layer->linear_qkv_w, &layer->linear_qkv_w_i4
        );
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
        snprintf(name, sizeof(name), "layers.%u.linear_qkv.row_sum", layer_idx);
        status = microgemm_load_or_compute_quantized_row_sums(
            model, loaded, name, layer->linear_qkv_w, layer->linear_qkv_w_i4,
            conv_dim, hidden, &layer->linear_qkv_row_sums
        );
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
        snprintf(name, sizeof(name), "layers.%u.linear_qkv.scale", layer_idx);
        status = microgemm_load_quantized_scales(
            model, loaded, name, conv_dim, hidden, &layer->linear_qkv_s
        );
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }

        snprintf(name, sizeof(name), "layers.%u.linear_baz.weight_i8", layer_idx);
        snprintf(name_i4, sizeof(name_i4), "layers.%u.linear_baz.weight_i4", layer_idx);
        status = microgemm_load_quantized_2d(
            model, loaded, name, name_i4, baz_rows, hidden,
            &layer->linear_baz_w, &layer->linear_baz_w_i4
        );
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
        snprintf(name, sizeof(name), "layers.%u.linear_baz.row_sum", layer_idx);
        status = microgemm_load_or_compute_quantized_row_sums(
            model, loaded, name, layer->linear_baz_w, layer->linear_baz_w_i4,
            baz_rows, hidden, &layer->linear_baz_row_sums
        );
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
        snprintf(name, sizeof(name), "layers.%u.linear_baz.scale", layer_idx);
        status = microgemm_load_quantized_scales(
            model, loaded, name, baz_rows, hidden, &layer->linear_baz_s
        );
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }

        snprintf(name, sizeof(name), "layers.%u.linear_conv.weight", layer_idx);
        status = microgemm_load_f32_2d(
            model, loaded, name, conv_dim, cfg->linear_conv_kernel_dim, &layer->linear_conv_w
        );
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
        snprintf(name, sizeof(name), "layers.%u.linear_dt_bias", layer_idx);
        status = microgemm_load_f32_1d(model, loaded, name, value_heads, &layer->linear_dt_bias);
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
        snprintf(name, sizeof(name), "layers.%u.linear_A_log", layer_idx);
        status = microgemm_load_f32_1d(model, loaded, name, value_heads, &layer->linear_a_log);
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
        snprintf(name, sizeof(name), "layers.%u.linear_norm.weight", layer_idx);
        status = microgemm_load_f32_1d(model, loaded, name, value_head_dim, &layer->linear_norm_w);
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
        snprintf(name, sizeof(name), "layers.%u.linear_out.weight_i8", layer_idx);
        snprintf(name_i4, sizeof(name_i4), "layers.%u.linear_out.weight_i4", layer_idx);
        status = microgemm_load_quantized_2d(
            model, loaded, name, name_i4, hidden, value_dim,
            &layer->linear_out_w, &layer->linear_out_w_i4
        );
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
        snprintf(name, sizeof(name), "layers.%u.linear_out.row_sum", layer_idx);
        status = microgemm_load_or_compute_quantized_row_sums(
            model, loaded, name, layer->linear_out_w, layer->linear_out_w_i4,
            hidden, value_dim, &layer->linear_out_row_sums
        );
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
        snprintf(name, sizeof(name), "layers.%u.linear_out.scale", layer_idx);
        status = microgemm_load_quantized_scales(
            model, loaded, name, hidden, value_dim, &layer->linear_out_s
        );
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
    } else {
    snprintf(name, sizeof(name), "layers.%u.qkv.weight_i8", layer_idx);
    snprintf(name_i4, sizeof(name_i4), "layers.%u.qkv.weight_i4", layer_idx);
    status = microgemm_load_quantized_2d(
        model, loaded, name, name_i4, qkv_rows, hidden, &layer->qkv_w, &layer->qkv_w_i4
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }
    snprintf(name, sizeof(name), "layers.%u.qkv.row_sum", layer_idx);
    status = microgemm_load_or_compute_quantized_row_sums(
        model,
        loaded,
        name,
        layer->qkv_w,
        layer->qkv_w_i4,
        qkv_rows,
        hidden,
        &layer->qkv_row_sums
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }

    snprintf(name, sizeof(name), "layers.%u.qkv.scale", layer_idx);
    status = microgemm_load_quantized_scales(
        model, loaded, name, qkv_rows, hidden, &layer->qkv_s
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }

    if ((cfg->flags & MICROGEMM_FLAG_QKV_BIAS) != 0u) {
        snprintf(name, sizeof(name), "layers.%u.qkv.bias", layer_idx);
        status = microgemm_load_f32_1d(model, loaded, name, qkv_rows, &layer->qkv_bias);
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
    } else {
        snprintf(name, sizeof(name), "layers.%u.qkv.bias", layer_idx);
        status = microgemm_load_optional_f32_1d(model, loaded, name, qkv_rows, &layer->qkv_bias);
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
    }

    if ((cfg->flags & MICROGEMM_FLAG_QK_NORM) != 0u) {
        snprintf(name, sizeof(name), "layers.%u.q_norm.weight", layer_idx);
        status = microgemm_load_f32_1d(model, loaded, name, cfg->head_dim, &layer->q_norm_w);
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
        snprintf(name, sizeof(name), "layers.%u.k_norm.weight", layer_idx);
        status = microgemm_load_f32_1d(model, loaded, name, cfg->head_dim, &layer->k_norm_w);
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
    }

    snprintf(name, sizeof(name), "layers.%u.o.weight_i8", layer_idx);
    snprintf(name_i4, sizeof(name_i4), "layers.%u.o.weight_i4", layer_idx);
    status = microgemm_load_quantized_2d(
        model, loaded, name, name_i4, hidden, attn_cols, &layer->o_w, &layer->o_w_i4
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }
    snprintf(name, sizeof(name), "layers.%u.o.row_sum", layer_idx);
    status = microgemm_load_or_compute_quantized_row_sums(
        model,
        loaded,
        name,
        layer->o_w,
        layer->o_w_i4,
        hidden,
        attn_cols,
        &layer->o_row_sums
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }

    snprintf(name, sizeof(name), "layers.%u.o.scale", layer_idx);
    status = microgemm_load_quantized_scales(
        model, loaded, name, hidden, attn_cols, &layer->o_s
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }

    if ((cfg->flags & MICROGEMM_FLAG_ATTN_OUTPUT_NORM) != 0u) {
        snprintf(name, sizeof(name), "layers.%u.attn_output_norm.weight", layer_idx);
        status = microgemm_load_f32_1d(model, loaded, name, hidden, &layer->attn_output_norm_w);
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
    }
    }

    snprintf(name, sizeof(name), "layers.%u.gate_up.weight_i8", layer_idx);
    snprintf(name_i4, sizeof(name_i4), "layers.%u.gate_up.weight_i4", layer_idx);
    status = microgemm_load_quantized_2d(
        model, loaded, name, name_i4, 2u * inter, hidden,
        &layer->gate_up_w, &layer->gate_up_w_i4
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }
    snprintf(name, sizeof(name), "layers.%u.gate_up.row_sum", layer_idx);
    status = microgemm_load_or_compute_quantized_row_sums(
        model,
        loaded,
        name,
        layer->gate_up_w,
        layer->gate_up_w_i4,
        2u * inter,
        hidden,
        &layer->gate_up_row_sums
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }

    snprintf(name, sizeof(name), "layers.%u.gate_up.scale", layer_idx);
    status = microgemm_load_quantized_scales(
        model, loaded, name, 2u * inter, hidden, &layer->gate_up_s
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }

    snprintf(name, sizeof(name), "layers.%u.down.weight_i8", layer_idx);
    snprintf(name_i4, sizeof(name_i4), "layers.%u.down.weight_i4", layer_idx);
    status = microgemm_load_quantized_2d(
        model, loaded, name, name_i4, hidden, inter, &layer->down_w, &layer->down_w_i4
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }
    snprintf(name, sizeof(name), "layers.%u.down.row_sum", layer_idx);
    status = microgemm_load_or_compute_quantized_row_sums(
        model,
        loaded,
        name,
        layer->down_w,
        layer->down_w_i4,
        hidden,
        inter,
        &layer->down_row_sums
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }

    snprintf(name, sizeof(name), "layers.%u.down.scale", layer_idx);
    status = microgemm_load_quantized_scales(
        model, loaded, name, hidden, inter, &layer->down_s
    );
    if (status != MICROGEMM_STATUS_OK) {
        return status;
    }

    if ((cfg->flags & MICROGEMM_FLAG_MLP_OUTPUT_NORM) != 0u) {
        snprintf(name, sizeof(name), "layers.%u.mlp_output_norm.weight", layer_idx);
        status = microgemm_load_f32_1d(model, loaded, name, hidden, &layer->mlp_output_norm_w);
        if (status != MICROGEMM_STATUS_OK) {
            return status;
        }
    }

    return MICROGEMM_STATUS_OK;
}

microgemm_status microgemm_loaded_model_i8_create(
    const microgemm_model* model,
    microgemm_loaded_model_i8** out_loaded
) {
    microgemm_loaded_model_i8* loaded;
    const microgemm_config* cfg;
    microgemm_status status;
    uint32_t layer_idx;

    if (model == NULL || out_loaded == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    *out_loaded = NULL;

    cfg = microgemm_model_config(model);
    if (cfg == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (!microgemm_quant_mode_is_i8_storage(cfg->quant_mode)
            && !microgemm_quant_mode_is_i4_storage(cfg->quant_mode)) {
        return MICROGEMM_STATUS_UNSUPPORTED;
    }

    loaded = (microgemm_loaded_model_i8*)calloc(1, sizeof(*loaded));
    if (loaded == NULL) {
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }
    loaded->config = *cfg;
    loaded->allocation_capacity = 8u + (size_t)cfg->num_layers * 36u;
    loaded->owned_allocations = (void**)calloc(loaded->allocation_capacity, sizeof(void*));
    loaded->layers = (microgemm_layer_weights_i8*)calloc(cfg->num_layers, sizeof(*loaded->layers));
    if (loaded->owned_allocations == NULL || loaded->layers == NULL) {
        microgemm_loaded_model_i8_destroy(loaded);
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }

    if (microgemm_quant_mode_is_i4_storage(cfg->quant_mode)) {
        status = microgemm_load_i4_2d(
            model, loaded, "embed_tokens.weight_i4",
            cfg->vocab_size, cfg->hidden_size,
            &loaded->weights.embed_tokens_i4
        );
        if (status != MICROGEMM_STATUS_OK) {
            microgemm_loaded_model_i8_destroy(loaded);
            return status;
        }
        status = microgemm_load_quantized_scales(
            model, loaded, "embed_tokens.scale",
            cfg->vocab_size,
            cfg->hidden_size,
            &loaded->weights.embed_tokens_s
        );
        if (status != MICROGEMM_STATUS_OK) {
            microgemm_loaded_model_i8_destroy(loaded);
            return status;
        }
    } else if (microgemm_model_find_tensor(model, "embed_tokens.weight_i8") != NULL) {
        status = microgemm_load_i8_2d(
            model, loaded, "embed_tokens.weight_i8",
            cfg->vocab_size, cfg->hidden_size,
            &loaded->weights.embed_tokens_i8
        );
        if (status != MICROGEMM_STATUS_OK) {
            microgemm_loaded_model_i8_destroy(loaded);
            return status;
        }
        status = microgemm_load_quantized_scales(
            model, loaded, "embed_tokens.scale",
            cfg->vocab_size,
            cfg->hidden_size,
            &loaded->weights.embed_tokens_s
        );
        if (status != MICROGEMM_STATUS_OK) {
            microgemm_loaded_model_i8_destroy(loaded);
            return status;
        }
    } else {
        status = microgemm_load_f32_2d(
            model, loaded, "embed_tokens.weight",
            cfg->vocab_size, cfg->hidden_size,
            &loaded->weights.embed_tokens
        );
        if (status != MICROGEMM_STATUS_OK) {
            microgemm_loaded_model_i8_destroy(loaded);
            return status;
        }
    }

    status = microgemm_load_f32_1d(
        model, loaded, "final_norm.weight",
        cfg->hidden_size,
        &loaded->weights.final_norm_w
    );
    if (status != MICROGEMM_STATUS_OK) {
        microgemm_loaded_model_i8_destroy(loaded);
        return status;
    }

    status = microgemm_load_quantized_2d(
        model, loaded, "lm_head.weight_i8", "lm_head.weight_i4",
        cfg->vocab_size, cfg->hidden_size,
        &loaded->weights.lm_head_w,
        &loaded->weights.lm_head_w_i4
    );
    if (status != MICROGEMM_STATUS_OK) {
        microgemm_loaded_model_i8_destroy(loaded);
        return status;
    }
    status = microgemm_load_or_compute_quantized_row_sums(
        model,
        loaded,
        "lm_head.row_sum",
        loaded->weights.lm_head_w,
        loaded->weights.lm_head_w_i4,
        cfg->vocab_size,
        cfg->hidden_size,
        &loaded->weights.lm_head_row_sums
    );
    if (status != MICROGEMM_STATUS_OK) {
        microgemm_loaded_model_i8_destroy(loaded);
        return status;
    }

    status = microgemm_load_quantized_scales(
        model, loaded, "lm_head.scale",
        cfg->vocab_size,
        cfg->hidden_size,
        &loaded->weights.lm_head_s
    );
    if (status != MICROGEMM_STATUS_OK) {
        microgemm_loaded_model_i8_destroy(loaded);
        return status;
    }

    status = microgemm_load_f32_2d(
        model, loaded, "rope.cos",
        cfg->max_position_embeddings, cfg->head_dim / 2u,
        &loaded->weights.cos_cache
    );
    if (status != MICROGEMM_STATUS_OK) {
        microgemm_loaded_model_i8_destroy(loaded);
        return status;
    }

    status = microgemm_load_f32_2d(
        model, loaded, "rope.sin",
        cfg->max_position_embeddings, cfg->head_dim / 2u,
        &loaded->weights.sin_cache
    );
    if (status != MICROGEMM_STATUS_OK) {
        microgemm_loaded_model_i8_destroy(loaded);
        return status;
    }

    loaded->weights.layers = loaded->layers;
    for (layer_idx = 0; layer_idx < cfg->num_layers; ++layer_idx) {
        status = microgemm_load_layer_i8(model, loaded, layer_idx);
        if (status != MICROGEMM_STATUS_OK) {
            microgemm_loaded_model_i8_destroy(loaded);
            return status;
        }
    }

    loaded->total_bytes += sizeof(*loaded);
    loaded->total_bytes += (size_t)cfg->num_layers * sizeof(*loaded->layers);
    loaded->total_bytes += loaded->allocation_capacity * sizeof(void*);

    *out_loaded = loaded;
    return MICROGEMM_STATUS_OK;
}

void microgemm_loaded_model_i8_destroy(microgemm_loaded_model_i8* loaded) {
    if (loaded == NULL) {
        return;
    }
    microgemm_loaded_model_i8_reset(loaded);
    free(loaded);
}

const microgemm_model_weights_i8* microgemm_loaded_model_i8_weights(
    const microgemm_loaded_model_i8* loaded
) {
    return loaded ? &loaded->weights : NULL;
}

const microgemm_config* microgemm_loaded_model_i8_config(
    const microgemm_loaded_model_i8* loaded
) {
    return loaded ? &loaded->config : NULL;
}

size_t microgemm_loaded_model_i8_bytes(const microgemm_loaded_model_i8* loaded) {
    return loaded ? loaded->total_bytes : 0;
}

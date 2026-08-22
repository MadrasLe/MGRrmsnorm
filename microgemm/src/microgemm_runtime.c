#if defined(__unix__) || defined(__APPLE__)
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#endif

#include "microgemm/microgemm.h"
#include "microgemm/microgemm_platform.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <sys/types.h>
#endif

struct microgemm_model {
    char* path;
    microgemm_file_header header;
    microgemm_config config;
    microgemm_tensor_entry* tensors;
};

struct microgemm_runtime {
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
    int8_t* input_q;
};

static int microgemm_seek(FILE* fp, uint64_t offset) {
#if defined(_WIN32)
    return _fseeki64(fp, (__int64)offset, SEEK_SET);
#elif defined(__unix__) || defined(__APPLE__)
    return fseeko(fp, (off_t)offset, SEEK_SET);
#else
    if (offset > (uint64_t)LONG_MAX) {
        return -1;
    }
    return fseek(fp, (long)offset, SEEK_SET);
#endif
}

static char* microgemm_strdup_local(const char* s) {
    size_t n;
    char* out;
    if (s == NULL) {
        return NULL;
    }
    n = strlen(s) + 1;
    out = (char*)malloc(n);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, s, n);
    return out;
}

static int microgemm_config_is_valid(const microgemm_config* cfg) {
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
    if (cfg->vocab_size == 0 || cfg->max_position_embeddings == 0) {
        return 0;
    }
    if (cfg->kv_block_size == 0) {
        return 0;
    }
    return 1;
}

static size_t microgemm_compute_scratch_bytes(const microgemm_config* cfg, uint32_t max_seq_len) {
    size_t hidden_size;
    size_t qkv_size;
    size_t max_input_q;
    size_t total;

    hidden_size = cfg->hidden_size;
    qkv_size = (size_t)(cfg->num_q_heads + 2U * cfg->num_kv_heads) * cfg->head_dim;
    max_input_q = hidden_size;
    if ((size_t)(2U * cfg->intermediate_size) > max_input_q) {
        max_input_q = (size_t)(2U * cfg->intermediate_size);
    }
    if ((size_t)(cfg->num_q_heads * cfg->head_dim) > max_input_q) {
        max_input_q = (size_t)(cfg->num_q_heads * cfg->head_dim);
    }

    total = 0;
    total += hidden_size * sizeof(float);
    total += hidden_size * sizeof(float);
    total += hidden_size * sizeof(float);
    total += qkv_size * sizeof(float);
    total += (size_t)(cfg->num_q_heads * cfg->head_dim) * sizeof(float);
    total += hidden_size * sizeof(float);
    total += (size_t)(2U * cfg->intermediate_size) * sizeof(float);
    total += hidden_size * sizeof(float);
    total += cfg->vocab_size * sizeof(float);
    total += max_seq_len * sizeof(float);
    total += max_input_q * sizeof(int8_t);
    return total;
}

static void microgemm_runtime_clear(struct microgemm_runtime* rt) {
    if (rt == NULL) {
        return;
    }
    free(rt->hidden);
    free(rt->residual);
    free(rt->normed);
    free(rt->qkv);
    free(rt->attn_out);
    free(rt->o_out);
    free(rt->gate_up);
    free(rt->mlp_out);
    free(rt->logits);
    free(rt->scores);
    free(rt->input_q);
    memset(rt, 0, sizeof(*rt));
}

uint32_t microgemm_version_major(void) {
    return 0;
}

uint32_t microgemm_version_minor(void) {
    return 1;
}

const char* microgemm_status_string(microgemm_status status) {
    switch (status) {
        case MICROGEMM_STATUS_OK:
            return "ok";
        case MICROGEMM_STATUS_INVALID_ARGUMENT:
            return "invalid_argument";
        case MICROGEMM_STATUS_IO_ERROR:
            return "io_error";
        case MICROGEMM_STATUS_FORMAT_ERROR:
            return "format_error";
        case MICROGEMM_STATUS_UNSUPPORTED:
            return "unsupported";
        case MICROGEMM_STATUS_OUT_OF_MEMORY:
            return "out_of_memory";
        case MICROGEMM_STATUS_NOT_IMPLEMENTED:
            return "not_implemented";
        default:
            return "unknown";
    }
}

microgemm_status microgemm_get_backend_info(microgemm_backend_info* out_info) {
    if (out_info == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    memset(out_info, 0, sizeof(*out_info));
    out_info->abi_version = 1;
    out_info->has_scalar = 1;
    out_info->has_avx2 = MICROGEMM_CPU_X86_AVX2 ? 1u : 0u;
    out_info->has_neon = MICROGEMM_CPU_ARM64_NEON ? 1u : 0u;
    out_info->has_dotprod = MICROGEMM_CPU_ARM64_DOTPROD ? 1u : 0u;

#if MICROGEMM_CPU_ARM64_NEON
    out_info->target_arch = MICROGEMM_PLATFORM_ANDROID ? "android-arm64" : "arm64";
    out_info->backend_name = MICROGEMM_CPU_ARM64_DOTPROD ? "cpu-arm64-neon-dotprod" : "cpu-arm64-neon";
#elif MICROGEMM_CPU_X86_AVX512_VNNI
    out_info->target_arch = "x86";
    out_info->backend_name = "cpu-x86-avx512-vnni";
#elif MICROGEMM_CPU_X86_AVX_VNNI
    out_info->target_arch = "x86";
    out_info->backend_name = "cpu-x86-avx-vnni";
#elif MICROGEMM_CPU_X86_AVX2
    out_info->target_arch = "x86";
    out_info->backend_name = MICROGEMM_CPU_X86_FMA ? "cpu-x86-avx2-fma" : "cpu-x86-avx2";
#else
    out_info->target_arch = "generic";
    out_info->backend_name = "cpu-scalar";
#endif

#ifdef _OPENMP
    out_info->max_threads = (uint32_t)omp_get_max_threads();
#else
    out_info->max_threads = 1;
#endif

    return MICROGEMM_STATUS_OK;
}

microgemm_status microgemm_model_open(const char* path, microgemm_model** out_model) {
    FILE* fp = NULL;
    microgemm_model* model = NULL;
    size_t read_count;
    size_t tensor_bytes;

    if (path == NULL || out_model == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    *out_model = NULL;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return MICROGEMM_STATUS_IO_ERROR;
    }

    model = (microgemm_model*)calloc(1, sizeof(*model));
    if (model == NULL) {
        fclose(fp);
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }

    read_count = fread(&model->header, sizeof(model->header), 1, fp);
    if (read_count != 1) {
        fclose(fp);
        free(model);
        return MICROGEMM_STATUS_IO_ERROR;
    }

    if (!microgemm_format_validate_header(&model->header)) {
        fclose(fp);
        free(model);
        return MICROGEMM_STATUS_FORMAT_ERROR;
    }

    if (microgemm_seek(fp, model->header.header_bytes) != 0) {
        fclose(fp);
        free(model);
        return MICROGEMM_STATUS_IO_ERROR;
    }

    read_count = fread(&model->config, sizeof(model->config), 1, fp);
    if (read_count != 1 || !microgemm_config_is_valid(&model->config)) {
        fclose(fp);
        free(model);
        return MICROGEMM_STATUS_FORMAT_ERROR;
    }

    if (model->header.tensor_count > 0) {
        if (model->header.tensor_count > ((uint64_t)SIZE_MAX / sizeof(microgemm_tensor_entry))) {
            fclose(fp);
            free(model);
            return MICROGEMM_STATUS_FORMAT_ERROR;
        }

        if (microgemm_seek(fp, model->header.tensor_directory_offset) != 0) {
            fclose(fp);
            free(model);
            return MICROGEMM_STATUS_IO_ERROR;
        }

        tensor_bytes = (size_t)model->header.tensor_count * sizeof(microgemm_tensor_entry);
        model->tensors = (microgemm_tensor_entry*)malloc(tensor_bytes);
        if (model->tensors == NULL) {
            fclose(fp);
            free(model);
            return MICROGEMM_STATUS_OUT_OF_MEMORY;
        }

        read_count = fread(model->tensors, sizeof(microgemm_tensor_entry), (size_t)model->header.tensor_count, fp);
        if (read_count != (size_t)model->header.tensor_count) {
            fclose(fp);
            microgemm_model_close(model);
            return MICROGEMM_STATUS_IO_ERROR;
        }
    }

    model->path = microgemm_strdup_local(path);
    if (model->path == NULL) {
        fclose(fp);
        microgemm_model_close(model);
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }

    fclose(fp);
    *out_model = model;
    return MICROGEMM_STATUS_OK;
}

void microgemm_model_close(microgemm_model* model) {
    if (model == NULL) {
        return;
    }
    free(model->path);
    free(model->tensors);
    free(model);
}

const char* microgemm_model_path(const microgemm_model* model) {
    return model ? model->path : NULL;
}

const microgemm_config* microgemm_model_config(const microgemm_model* model) {
    return model ? &model->config : NULL;
}

size_t microgemm_model_tensor_count(const microgemm_model* model) {
    return model ? (size_t)model->header.tensor_count : 0;
}

const microgemm_tensor_entry* microgemm_model_tensor_at(const microgemm_model* model, size_t index) {
    if (model == NULL || index >= (size_t)model->header.tensor_count) {
        return NULL;
    }
    return &model->tensors[index];
}

const microgemm_tensor_entry* microgemm_model_find_tensor(const microgemm_model* model, const char* name) {
    size_t i;
    if (model == NULL || name == NULL) {
        return NULL;
    }
    for (i = 0; i < (size_t)model->header.tensor_count; ++i) {
        if (strncmp(model->tensors[i].name, name, MICROGEMM_MAX_TENSOR_NAME) == 0) {
            return &model->tensors[i];
        }
    }
    return NULL;
}

microgemm_status microgemm_model_read_tensor(
    const microgemm_model* model,
    const microgemm_tensor_entry* tensor,
    void* dst,
    size_t dst_bytes
) {
    FILE* fp;
    size_t read_count;

    if (model == NULL || tensor == NULL || dst == NULL) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }
    if (dst_bytes < (size_t)tensor->byte_length) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    fp = fopen(model->path, "rb");
    if (fp == NULL) {
        return MICROGEMM_STATUS_IO_ERROR;
    }
    if (microgemm_seek(fp, tensor->offset) != 0) {
        fclose(fp);
        return MICROGEMM_STATUS_IO_ERROR;
    }

    read_count = fread(dst, 1, (size_t)tensor->byte_length, fp);
    fclose(fp);
    if (read_count != (size_t)tensor->byte_length) {
        return MICROGEMM_STATUS_IO_ERROR;
    }
    return MICROGEMM_STATUS_OK;
}

microgemm_status microgemm_runtime_create(
    const microgemm_config* config,
    uint32_t max_seq_len,
    microgemm_runtime** out_runtime
) {
    microgemm_runtime* rt = NULL;
    size_t hidden_size;
    size_t qkv_size;
    size_t attn_out_size;
    size_t max_input_q;

    if (config == NULL || out_runtime == NULL || max_seq_len == 0) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    if (!microgemm_config_is_valid(config)) {
        return MICROGEMM_STATUS_INVALID_ARGUMENT;
    }

    rt = (microgemm_runtime*)calloc(1, sizeof(*rt));
    if (rt == NULL) {
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }

    rt->config = *config;
    rt->max_seq_len = max_seq_len;
    rt->scratch_bytes = microgemm_compute_scratch_bytes(config, max_seq_len);

    hidden_size = config->hidden_size;
    qkv_size = (size_t)(config->num_q_heads + 2U * config->num_kv_heads) * config->head_dim;
    attn_out_size = (size_t)(config->num_q_heads * config->head_dim);
    max_input_q = hidden_size;
    if ((size_t)(2U * config->intermediate_size) > max_input_q) {
        max_input_q = (size_t)(2U * config->intermediate_size);
    }
    if (attn_out_size > max_input_q) {
        max_input_q = attn_out_size;
    }

    rt->hidden = (float*)malloc(hidden_size * sizeof(float));
    rt->residual = (float*)malloc(hidden_size * sizeof(float));
    rt->normed = (float*)malloc(hidden_size * sizeof(float));
    rt->qkv = (float*)malloc(qkv_size * sizeof(float));
    rt->attn_out = (float*)malloc(attn_out_size * sizeof(float));
    rt->o_out = (float*)malloc(hidden_size * sizeof(float));
    rt->gate_up = (float*)malloc((size_t)(2U * config->intermediate_size) * sizeof(float));
    rt->mlp_out = (float*)malloc(hidden_size * sizeof(float));
    rt->logits = (float*)malloc(config->vocab_size * sizeof(float));
    rt->scores = (float*)malloc(max_seq_len * sizeof(float));
    rt->input_q = (int8_t*)malloc(max_input_q * sizeof(int8_t));

    if (rt->hidden == NULL || rt->residual == NULL || rt->normed == NULL
            || rt->qkv == NULL || rt->attn_out == NULL || rt->o_out == NULL
            || rt->gate_up == NULL || rt->mlp_out == NULL || rt->logits == NULL
            || rt->scores == NULL || rt->input_q == NULL) {
        microgemm_runtime_clear(rt);
        free(rt);
        return MICROGEMM_STATUS_OUT_OF_MEMORY;
    }

    *out_runtime = rt;
    return MICROGEMM_STATUS_OK;
}

void microgemm_runtime_destroy(microgemm_runtime* runtime) {
    if (runtime == NULL) {
        return;
    }
    microgemm_runtime_clear(runtime);
    free(runtime);
}

size_t microgemm_runtime_scratch_bytes(const microgemm_runtime* runtime) {
    return runtime ? runtime->scratch_bytes : 0;
}

const microgemm_config* microgemm_runtime_config(const microgemm_runtime* runtime) {
    return runtime ? &runtime->config : NULL;
}

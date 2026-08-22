#ifndef MICROGEMM_H
#define MICROGEMM_H

#include <stddef.h>
#include <stdint.h>

#include "microgemm_format.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum microgemm_status {
    MICROGEMM_STATUS_OK = 0,
    MICROGEMM_STATUS_INVALID_ARGUMENT = 1,
    MICROGEMM_STATUS_IO_ERROR = 2,
    MICROGEMM_STATUS_FORMAT_ERROR = 3,
    MICROGEMM_STATUS_UNSUPPORTED = 4,
    MICROGEMM_STATUS_OUT_OF_MEMORY = 5,
    MICROGEMM_STATUS_NOT_IMPLEMENTED = 6
} microgemm_status;

typedef struct microgemm_model microgemm_model;
typedef struct microgemm_runtime microgemm_runtime;

typedef struct microgemm_backend_info {
    uint32_t abi_version;
    const char* target_arch;
    const char* backend_name;
    uint32_t has_scalar;
    uint32_t has_avx2;
    uint32_t has_neon;
    uint32_t has_dotprod;
    uint32_t max_threads;
} microgemm_backend_info;

uint32_t microgemm_version_major(void);
uint32_t microgemm_version_minor(void);
const char* microgemm_status_string(microgemm_status status);
microgemm_status microgemm_get_backend_info(microgemm_backend_info* out_info);
int microgemm_kernel_selftest(void);
int microgemm_kernel_i8g_saturation_probe(float* out_got, float* out_ref, float* out_abs_diff);

microgemm_status microgemm_model_open(const char* path, microgemm_model** out_model);
void microgemm_model_close(microgemm_model* model);
const char* microgemm_model_path(const microgemm_model* model);
const microgemm_config* microgemm_model_config(const microgemm_model* model);
size_t microgemm_model_tensor_count(const microgemm_model* model);
const microgemm_tensor_entry* microgemm_model_tensor_at(const microgemm_model* model, size_t index);
const microgemm_tensor_entry* microgemm_model_find_tensor(const microgemm_model* model, const char* name);
microgemm_status microgemm_model_read_tensor(
    const microgemm_model* model,
    const microgemm_tensor_entry* tensor,
    void* dst,
    size_t dst_bytes
);

microgemm_status microgemm_runtime_create(
    const microgemm_config* config,
    uint32_t max_seq_len,
    microgemm_runtime** out_runtime
);
void microgemm_runtime_destroy(microgemm_runtime* runtime);
size_t microgemm_runtime_scratch_bytes(const microgemm_runtime* runtime);
const microgemm_config* microgemm_runtime_config(const microgemm_runtime* runtime);

#ifdef __cplusplus
}
#endif

#endif

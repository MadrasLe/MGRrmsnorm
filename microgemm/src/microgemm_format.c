#include "microgemm/microgemm_format.h"

int microgemm_format_validate_header(const microgemm_file_header* header) {
    if (header == NULL) {
        return 0;
    }

    if (header->magic[0] != MICROGEMM_MAGIC_0
            || header->magic[1] != MICROGEMM_MAGIC_1
            || header->magic[2] != MICROGEMM_MAGIC_2
            || header->magic[3] != MICROGEMM_MAGIC_3) {
        return 0;
    }

    if (header->version_major != 0) {
        return 0;
    }

    if (header->header_bytes < sizeof(microgemm_file_header)) {
        return 0;
    }

    if (header->config_bytes < sizeof(microgemm_config)) {
        return 0;
    }

    if (header->tensor_entry_bytes < sizeof(microgemm_tensor_entry)) {
        return 0;
    }

    if (header->tensor_directory_offset < header->header_bytes + header->config_bytes) {
        return 0;
    }

    if (header->tensor_data_offset < header->tensor_directory_offset) {
        return 0;
    }

    return 1;
}

const char* microgemm_format_architecture_name(uint32_t architecture) {
    switch (architecture) {
        case MICROGEMM_ARCH_LLAMA_LIKE:
            return "llama_like";
        case MICROGEMM_ARCH_QWEN2_LIKE:
            return "qwen2_like";
        case MICROGEMM_ARCH_MISTRAL_LIKE:
            return "mistral_like";
        case MICROGEMM_ARCH_GEMMA_LIKE:
            return "gemma_like";
        case MICROGEMM_ARCH_QWEN35_LIKE:
            return "qwen3_5_like";
        case MICROGEMM_ARCH_PHI_LIKE:
            return "phi_like";
        case MICROGEMM_ARCH_GRANITE_LIKE:
            return "granite_like";
        case MICROGEMM_ARCH_GLM4_LIKE:
            return "glm4_like";
        default:
            return "unknown";
    }
}

const char* microgemm_format_dtype_name(uint32_t dtype) {
    switch (dtype) {
        case MICROGEMM_DTYPE_F32:
            return "f32";
        case MICROGEMM_DTYPE_F16:
            return "f16";
        case MICROGEMM_DTYPE_BF16:
            return "bf16";
        case MICROGEMM_DTYPE_I8:
            return "i8";
        case MICROGEMM_DTYPE_U8:
            return "u8";
        case MICROGEMM_DTYPE_I4:
            return "i4";
        case MICROGEMM_DTYPE_I32:
            return "i32";
        default:
            return "unknown";
    }
}

const char* microgemm_format_quant_name(uint32_t quant_mode) {
    switch (quant_mode) {
        case MICROGEMM_QUANT_NONE:
            return "none";
        case MICROGEMM_QUANT_INT8:
            return "int8";
        case MICROGEMM_QUANT_INT4:
            return "int4";
        case MICROGEMM_QUANT_INT8G128:
            return "int8g128";
        case MICROGEMM_QUANT_INT4G128:
            return "int4g128";
        default:
            return "unknown";
    }
}

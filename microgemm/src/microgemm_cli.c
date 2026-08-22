#include "microgemm/microgemm.h"
#include "microgemm/microgemm_decode.h"
#include "microgemm/microgemm_platform.h"

#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct microgemm_generation_args {
    int* prompt_tokens;
    size_t prompt_count;
    uint32_t generate_count;
    uint32_t max_seq_len;
    int use_eos;
    int eos_token;
} microgemm_generation_args;

typedef struct microgemm_generation_result {
    microgemm_config config;
    size_t loaded_model_bytes;
    size_t workspace_bytes;
    size_t kv_cache_bytes;
    size_t context_tokens_used;
    int* generated_tokens;
    size_t generated_count;
} microgemm_generation_result;

static void print_usage(void) {
    puts("MicroGemm CLI");
    puts("Usage:");
    puts("  microgemm version");
    puts("  microgemm capabilities");
    puts("  microgemm kernel-selftest");
    puts("  microgemm i8g-saturation-probe");
    puts("  microgemm inspect <model.mgm>");
    puts("  microgemm runtime-dryrun --hidden-size N --intermediate-size N --layers N --q-heads N --kv-heads N --head-dim N --vocab-size N --max-seq-len N [--architecture llama_like|qwen2_like|mistral_like|gemma_like|phi_like|granite_like|glm4_like] [--kv-block-size N] [--quant none|int8|int4|int8g128|int4g128]");
    puts("  microgemm decode-smoke <model.mgm> --tokens 1,2,3 [--generate N] [--max-seq-len N] [--eos-token N]");
    puts("  microgemm generate-ids <model.mgm> --tokens 1,2,3 [--max-new-tokens N] [--max-seq-len N] [--eos-token N]");
}

static int parse_arch(const char* value, uint32_t* out) {
    if (strcmp(value, "llama_like") == 0) {
        *out = MICROGEMM_ARCH_LLAMA_LIKE;
        return 1;
    }
    if (strcmp(value, "qwen2_like") == 0) {
        *out = MICROGEMM_ARCH_QWEN2_LIKE;
        return 1;
    }
    if (strcmp(value, "mistral_like") == 0) {
        *out = MICROGEMM_ARCH_MISTRAL_LIKE;
        return 1;
    }
    if (strcmp(value, "gemma_like") == 0) {
        *out = MICROGEMM_ARCH_GEMMA_LIKE;
        return 1;
    }
    if (strcmp(value, "phi_like") == 0) {
        *out = MICROGEMM_ARCH_PHI_LIKE;
        return 1;
    }
    if (strcmp(value, "granite_like") == 0) {
        *out = MICROGEMM_ARCH_GRANITE_LIKE;
        return 1;
    }
    if (strcmp(value, "glm4_like") == 0) {
        *out = MICROGEMM_ARCH_GLM4_LIKE;
        return 1;
    }
    return 0;
}

static int parse_quant(const char* value, uint32_t* out) {
    if (strcmp(value, "none") == 0) {
        *out = MICROGEMM_QUANT_NONE;
        return 1;
    }
    if (strcmp(value, "int8") == 0) {
        *out = MICROGEMM_QUANT_INT8;
        return 1;
    }
    if (strcmp(value, "int4") == 0) {
        *out = MICROGEMM_QUANT_INT4;
        return 1;
    }
    if (strcmp(value, "int8g128") == 0 || strcmp(value, "int8g") == 0) {
        *out = MICROGEMM_QUANT_INT8G128;
        return 1;
    }
    if (strcmp(value, "int4g128") == 0 || strcmp(value, "int4g") == 0) {
        *out = MICROGEMM_QUANT_INT4G128;
        return 1;
    }
    return 0;
}

static int parse_u32_arg(const char* raw, uint32_t* out) {
    char* end_ptr;
    unsigned long value;

    if (raw == NULL || out == NULL) {
        return 0;
    }

    value = strtoul(raw, &end_ptr, 10);
    if (end_ptr == raw || *end_ptr != '\0' || value > 0xffffffffUL) {
        return 0;
    }

    *out = (uint32_t)value;
    return 1;
}

static int parse_nonnegative_int_arg(const char* raw, int* out) {
    char* end_ptr;
    long value;

    if (raw == NULL || out == NULL) {
        return 0;
    }

    value = strtol(raw, &end_ptr, 10);
    if (end_ptr == raw || *end_ptr != '\0' || value < 0 || value > INT_MAX) {
        return 0;
    }

    *out = (int)value;
    return 1;
}

static void microgemm_generation_args_clear(microgemm_generation_args* args) {
    if (args == NULL) {
        return;
    }
    free(args->prompt_tokens);
    memset(args, 0, sizeof(*args));
}

static void microgemm_generation_result_clear(microgemm_generation_result* result) {
    if (result == NULL) {
        return;
    }
    free(result->generated_tokens);
    memset(result, 0, sizeof(*result));
}

static int print_capabilities(void) {
    microgemm_backend_info info;
    microgemm_status status = microgemm_get_backend_info(&info);
    if (status != MICROGEMM_STATUS_OK) {
        fprintf(stderr, "capabilities failed: %s\n", microgemm_status_string(status));
        return 1;
    }

    printf("abi_version: %u\n", info.abi_version);
    printf("target_arch: %s\n", info.target_arch ? info.target_arch : "unknown");
    printf("backend: %s\n", info.backend_name ? info.backend_name : "unknown");
    printf("scalar: %u\n", info.has_scalar);
    printf("avx2: %u\n", info.has_avx2);
    printf("fma: %u\n", MICROGEMM_CPU_X86_FMA ? 1u : 0u);
    printf("neon: %u\n", info.has_neon);
    printf("dotprod: %u\n", info.has_dotprod);
    printf("max_threads: %u\n", info.max_threads);
    return 0;
}

static int kernel_selftest(void) {
    int code = microgemm_kernel_selftest();
    if (code == 0) {
        puts("kernel_selftest: ok");
        return 0;
    }
    fprintf(stderr, "kernel_selftest: failed (%d)\n", code);
    return 1;
}

static int i8g_saturation_probe(void) {
    float got = 0.0f;
    float ref = 0.0f;
    float abs_diff = 0.0f;
    int detected = microgemm_kernel_i8g_saturation_probe(&got, &ref, &abs_diff);
    puts(detected ? "i8g_saturation_probe: detected" : "i8g_saturation_probe: not_detected");
    printf("got: %.9g\n", got);
    printf("ref: %.9g\n", ref);
    printf("abs_diff: %.9g\n", abs_diff);
    return 0;
}

static int inspect_model(const char* path) {
    microgemm_model* model = NULL;
    microgemm_status status;
    const microgemm_config* cfg;
    size_t i;

    status = microgemm_model_open(path, &model);
    if (status != MICROGEMM_STATUS_OK) {
        fprintf(stderr, "inspect failed: %s\n", microgemm_status_string(status));
        return 1;
    }

    cfg = microgemm_model_config(model);
    printf("path: %s\n", microgemm_model_path(model));
    printf("architecture: %s\n", microgemm_format_architecture_name(cfg->architecture));
    printf("quant: %s\n", microgemm_format_quant_name(cfg->quant_mode));
    printf("flags: 0x%08x\n", cfg->flags);
    printf("attention_logit_softcap: %.8g\n", cfg->attention_logit_softcap);
    printf("final_logit_softcap: %.8g\n", cfg->final_logit_softcap);
    printf("query_pre_attn_scalar: %.8g\n", cfg->query_pre_attn_scalar);
    printf("embedding_multiplier: %.8g\n", cfg->embedding_multiplier);
    printf("residual_multiplier: %.8g\n", cfg->residual_multiplier);
    printf("logits_scaling: %.8g\n", cfg->logits_scaling);
    printf("qkv_rows: %u\n", cfg->qkv_rows);
    printf("attn_width: %u\n", cfg->attn_width);
    printf("rotary_dim: %u\n", cfg->rotary_dim);
    if (cfg->architecture == MICROGEMM_ARCH_QWEN35_LIKE) {
        printf("linear_key_head_dim: %u\n", cfg->linear_key_head_dim);
        printf("linear_value_head_dim: %u\n", cfg->linear_value_head_dim);
        printf("linear_num_key_heads: %u\n", cfg->linear_num_key_heads);
        printf("linear_num_value_heads: %u\n", cfg->linear_num_value_heads);
        printf("linear_conv_kernel_dim: %u\n", cfg->linear_conv_kernel_dim);
    }
    printf("hidden_size: %u\n", cfg->hidden_size);
    printf("intermediate_size: %u\n", cfg->intermediate_size);
    printf("layers: %u\n", cfg->num_layers);
    printf("q_heads: %u\n", cfg->num_q_heads);
    printf("kv_heads: %u\n", cfg->num_kv_heads);
    printf("head_dim: %u\n", cfg->head_dim);
    printf("vocab_size: %u\n", cfg->vocab_size);
    printf("max_position_embeddings: %u\n", cfg->max_position_embeddings);
    printf("kv_block_size: %u\n", cfg->kv_block_size);
    printf("tensor_count: %zu\n", microgemm_model_tensor_count(model));

    for (i = 0; i < microgemm_model_tensor_count(model); ++i) {
        const microgemm_tensor_entry* tensor = microgemm_model_tensor_at(model, i);
        printf("tensor[%zu]: %s dtype=%s rank=%u bytes=%llu offset=%llu\n",
            i,
            tensor->name,
            microgemm_format_dtype_name(tensor->dtype),
            tensor->rank,
            (unsigned long long)tensor->byte_length,
            (unsigned long long)tensor->offset);
    }

    microgemm_model_close(model);
    return 0;
}

static int parse_token_csv(const char* raw, int** out_tokens, size_t* out_count) {
    const char* cursor;
    int* tokens = NULL;
    size_t count = 0;
    size_t capacity = 0;

    if (raw == NULL || out_tokens == NULL || out_count == NULL) {
        return 0;
    }

    cursor = raw;
    while (*cursor != '\0') {
        char* end_ptr;
        long value;
        int* grown;

        while (*cursor != '\0' && (isspace((unsigned char)*cursor) || *cursor == ',')) {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }

        value = strtol(cursor, &end_ptr, 10);
        if (end_ptr == cursor || value < 0 || value > INT_MAX) {
            free(tokens);
            return 0;
        }

        if (count >= capacity) {
            size_t new_capacity = capacity == 0 ? 8u : capacity * 2u;
            grown = (int*)realloc(tokens, new_capacity * sizeof(int));
            if (grown == NULL) {
                free(tokens);
                return 0;
            }
            tokens = grown;
            capacity = new_capacity;
        }

        tokens[count++] = (int)value;
        cursor = end_ptr;
        while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        if (*cursor == ',') {
            ++cursor;
        } else if (*cursor != '\0') {
            free(tokens);
            return 0;
        }
    }

    if (count == 0) {
        free(tokens);
        return 0;
    }

    *out_tokens = tokens;
    *out_count = count;
    return 1;
}

static int parse_generation_args(
    int argc,
    char** argv,
    const char* command_name,
    microgemm_generation_args* out_args
) {
    int i;

    if (out_args == NULL) {
        return 0;
    }

    memset(out_args, 0, sizeof(*out_args));
    out_args->generate_count = 1;

    for (i = 0; i < argc; ++i) {
        const char* key = argv[i];
        if (strcmp(key, "--tokens") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a value for --tokens\n", command_name);
                microgemm_generation_args_clear(out_args);
                return 0;
            }
            if (out_args->prompt_tokens != NULL) {
                fprintf(stderr, "%s received --tokens more than once\n", command_name);
                microgemm_generation_args_clear(out_args);
                return 0;
            }
            if (!parse_token_csv(argv[++i], &out_args->prompt_tokens, &out_args->prompt_count)) {
                fprintf(stderr, "failed to parse --tokens; expected comma-separated token ids\n");
                microgemm_generation_args_clear(out_args);
                return 0;
            }
        } else if (strcmp(key, "--generate") == 0 || strcmp(key, "--max-new-tokens") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a value for %s\n", command_name, key);
                microgemm_generation_args_clear(out_args);
                return 0;
            }
            if (!parse_u32_arg(argv[++i], &out_args->generate_count) || out_args->generate_count == 0) {
                fprintf(stderr, "%s requires a positive integer for %s\n", command_name, key);
                microgemm_generation_args_clear(out_args);
                return 0;
            }
        } else if (strcmp(key, "--max-seq-len") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a value for --max-seq-len\n", command_name);
                microgemm_generation_args_clear(out_args);
                return 0;
            }
            if (!parse_u32_arg(argv[++i], &out_args->max_seq_len) || out_args->max_seq_len == 0) {
                fprintf(stderr, "%s requires a positive integer for --max-seq-len\n", command_name);
                microgemm_generation_args_clear(out_args);
                return 0;
            }
        } else if (strcmp(key, "--eos-token") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a value for --eos-token\n", command_name);
                microgemm_generation_args_clear(out_args);
                return 0;
            }
            if (!parse_nonnegative_int_arg(argv[++i], &out_args->eos_token)) {
                fprintf(stderr, "%s requires a non-negative integer for --eos-token\n", command_name);
                microgemm_generation_args_clear(out_args);
                return 0;
            }
            out_args->use_eos = 1;
        } else {
            fprintf(stderr, "unknown flag for %s: %s\n", command_name, key);
            microgemm_generation_args_clear(out_args);
            return 0;
        }
    }

    if (out_args->prompt_tokens == NULL || out_args->prompt_count == 0) {
        fprintf(stderr, "%s requires --tokens 1,2,3\n", command_name);
        microgemm_generation_args_clear(out_args);
        return 0;
    }

    return 1;
}

static void free_kv_layers(float** layer_kv, uint32_t num_layers) {
    uint32_t layer_idx;
    if (layer_kv == NULL) {
        return;
    }
    for (layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        free(layer_kv[layer_idx]);
    }
    free(layer_kv);
}

static int run_greedy_generation(
    const char* path,
    const microgemm_generation_args* args,
    microgemm_generation_result* out_result
) {
    microgemm_model* model = NULL;
    microgemm_loaded_model_i8* loaded = NULL;
    microgemm_decode_workspace* workspace = NULL;
    const microgemm_config* cfg;
    const microgemm_model_weights_i8* weights;
    microgemm_kv_layout kv;
    float** layer_kv = NULL;
    int* block_table = NULL;
    size_t required_seq_len;
    uint32_t effective_max_seq_len;
    uint32_t num_layers = 0;
    size_t block_count;
    size_t step_idx;
    int next_token = -1;
    int current_token = -1;
    microgemm_status status;
    int ok = 0;

    if (path == NULL || args == NULL || out_result == NULL || args->prompt_tokens == NULL
            || args->prompt_count == 0 || args->generate_count == 0) {
        return 0;
    }

    memset(out_result, 0, sizeof(*out_result));
    memset(&kv, 0, sizeof(kv));

    status = microgemm_model_open(path, &model);
    if (status != MICROGEMM_STATUS_OK) {
        fprintf(stderr, "generation open failed: %s\n", microgemm_status_string(status));
        goto cleanup;
    }

    status = microgemm_loaded_model_i8_create(model, &loaded);
    if (status != MICROGEMM_STATUS_OK) {
        fprintf(stderr, "generation load failed: %s\n", microgemm_status_string(status));
        goto cleanup;
    }

    cfg = microgemm_loaded_model_i8_config(loaded);
    weights = microgemm_loaded_model_i8_weights(loaded);
    if (cfg == NULL || weights == NULL) {
        fprintf(stderr, "generation failed: loaded model is incomplete\n");
        goto cleanup;
    }
    num_layers = cfg->num_layers;

    required_seq_len = args->prompt_count + (size_t)args->generate_count - 1u;
    effective_max_seq_len = args->max_seq_len == 0 ? (uint32_t)required_seq_len : args->max_seq_len;
    if ((size_t)effective_max_seq_len < required_seq_len) {
        fprintf(stderr, "--max-seq-len is too small for prompt + generation context\n");
        goto cleanup;
    }
    if (effective_max_seq_len > cfg->max_position_embeddings) {
        fprintf(stderr, "--max-seq-len exceeds model max_position_embeddings\n");
        goto cleanup;
    }

    status = microgemm_decode_workspace_create(cfg, effective_max_seq_len, &workspace);
    if (status != MICROGEMM_STATUS_OK) {
        fprintf(stderr, "generation workspace failed: %s\n", microgemm_status_string(status));
        goto cleanup;
    }

    block_count = ((size_t)effective_max_seq_len + cfg->kv_block_size - 1u) / cfg->kv_block_size;
    kv.stride_pos = (int)cfg->head_dim;
    kv.stride_head = (int)(cfg->kv_block_size * cfg->head_dim);
    kv.stride_kv = (int)(cfg->num_kv_heads * (uint32_t)kv.stride_head);
    kv.stride_block = 2 * kv.stride_kv;
    kv.seq_len = 0;

    block_table = (int*)malloc(block_count * sizeof(int));
    layer_kv = (float**)calloc(cfg->num_layers, sizeof(float*));
    out_result->generated_tokens = (int*)malloc((size_t)args->generate_count * sizeof(int));
    if (block_table == NULL || layer_kv == NULL || out_result->generated_tokens == NULL) {
        fprintf(stderr, "generation failed: out of memory\n");
        goto cleanup;
    }

    for (step_idx = 0; step_idx < block_count; ++step_idx) {
        block_table[step_idx] = (int)step_idx;
    }
    for (step_idx = 0; step_idx < cfg->num_layers; ++step_idx) {
        layer_kv[step_idx] = (float*)calloc(block_count * (size_t)kv.stride_block, sizeof(float));
        if (layer_kv[step_idx] == NULL) {
            fprintf(stderr, "generation failed: out of memory allocating KV cache\n");
            goto cleanup;
        }
    }

    kv.layer_kv = layer_kv;
    kv.block_table = block_table;

    for (step_idx = 0; step_idx < args->prompt_count; ++step_idx) {
        status = microgemm_decode_step_i8(
            cfg,
            weights,
            workspace,
            args->prompt_tokens[step_idx],
            kv.seq_len,
            &kv,
            NULL,
            0,
            &next_token
        );
        if (status != MICROGEMM_STATUS_OK) {
            fprintf(stderr, "generation failed during prompt prefill: %s\n", microgemm_status_string(status));
            goto cleanup;
        }
        kv.seq_len += 1;
    }

    out_result->generated_tokens[0] = next_token;
    out_result->generated_count = 1;
    current_token = next_token;

    for (step_idx = 1; step_idx < (size_t)args->generate_count; ++step_idx) {
        if (args->use_eos && current_token == args->eos_token) {
            break;
        }

        status = microgemm_decode_step_i8(
            cfg,
            weights,
            workspace,
            current_token,
            kv.seq_len,
            &kv,
            NULL,
            0,
            &next_token
        );
        if (status != MICROGEMM_STATUS_OK) {
            fprintf(stderr, "generation failed during decode: %s\n", microgemm_status_string(status));
            goto cleanup;
        }

        kv.seq_len += 1;
        out_result->generated_tokens[out_result->generated_count++] = next_token;
        current_token = next_token;
    }

    out_result->config = *cfg;
    out_result->loaded_model_bytes = microgemm_loaded_model_i8_bytes(loaded);
    out_result->workspace_bytes = microgemm_decode_workspace_bytes(workspace);
    out_result->kv_cache_bytes =
        (size_t)cfg->num_layers * block_count * (size_t)kv.stride_block * sizeof(float);
    out_result->context_tokens_used = (size_t)kv.seq_len;
    ok = 1;

cleanup:
    free(block_table);
    free_kv_layers(layer_kv, num_layers);
    microgemm_decode_workspace_destroy(workspace);
    microgemm_loaded_model_i8_destroy(loaded);
    microgemm_model_close(model);
    if (!ok) {
        microgemm_generation_result_clear(out_result);
    }
    return ok;
}

static void print_generated_tokens_csv(const int* tokens, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i) {
        printf("%s%d", i == 0 ? " " : ",", tokens[i]);
    }
    printf("\n");
}

static int decode_smoke(const char* path, int argc, char** argv) {
    microgemm_generation_args args;
    microgemm_generation_result result;
    int ok;

    memset(&args, 0, sizeof(args));
    memset(&result, 0, sizeof(result));

    ok = parse_generation_args(argc, argv, "decode-smoke", &args);
    if (!ok) {
        return 1;
    }

    ok = run_greedy_generation(path, &args, &result);
    if (!ok) {
        microgemm_generation_args_clear(&args);
        return 1;
    }

    printf("path: %s\n", path);
    printf("architecture: %s\n", microgemm_format_architecture_name(result.config.architecture));
    printf("quant: %s\n", microgemm_format_quant_name(result.config.quant_mode));
    printf("loaded_model_bytes: %zu\n", result.loaded_model_bytes);
    printf("workspace_bytes: %zu\n", result.workspace_bytes);
    printf("kv_cache_bytes: %zu\n", result.kv_cache_bytes);
    printf("runtime_total_bytes: %zu\n",
        result.loaded_model_bytes + result.workspace_bytes + result.kv_cache_bytes);
    printf("prompt_tokens: %zu\n", args.prompt_count);
    printf("context_tokens_used: %zu\n", result.context_tokens_used);
    printf("generated_tokens:");
    print_generated_tokens_csv(result.generated_tokens, result.generated_count);

    microgemm_generation_result_clear(&result);
    microgemm_generation_args_clear(&args);
    return 0;
}

static int generate_ids(const char* path, int argc, char** argv) {
    microgemm_generation_args args;
    microgemm_generation_result result;
    int ok;

    memset(&args, 0, sizeof(args));
    memset(&result, 0, sizeof(result));

    ok = parse_generation_args(argc, argv, "generate-ids", &args);
    if (!ok) {
        return 1;
    }

    ok = run_greedy_generation(path, &args, &result);
    if (!ok) {
        microgemm_generation_args_clear(&args);
        return 1;
    }

    printf("generated_token_count: %zu\n", result.generated_count);
    printf("generated_tokens:");
    print_generated_tokens_csv(result.generated_tokens, result.generated_count);

    microgemm_generation_result_clear(&result);
    microgemm_generation_args_clear(&args);
    return 0;
}

static int runtime_dryrun(int argc, char** argv) {
    microgemm_config cfg;
    microgemm_runtime* runtime = NULL;
    microgemm_status status;
    int i;
    uint32_t max_seq_len = 0;

    memset(&cfg, 0, sizeof(cfg));
    cfg.architecture = MICROGEMM_ARCH_LLAMA_LIKE;
    cfg.quant_mode = MICROGEMM_QUANT_INT8;
    cfg.kv_block_size = 16;
    cfg.max_position_embeddings = 4096;
    cfg.rms_norm_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;

    for (i = 2; i < argc; ++i) {
        const char* key = argv[i];
        const char* value;
        if (i + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", key);
            return 1;
        }
        value = argv[++i];

        if (strcmp(key, "--hidden-size") == 0) {
            cfg.hidden_size = (uint32_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "--intermediate-size") == 0) {
            cfg.intermediate_size = (uint32_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "--layers") == 0) {
            cfg.num_layers = (uint32_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "--q-heads") == 0) {
            cfg.num_q_heads = (uint32_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "--kv-heads") == 0) {
            cfg.num_kv_heads = (uint32_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "--head-dim") == 0) {
            cfg.head_dim = (uint32_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "--vocab-size") == 0) {
            cfg.vocab_size = (uint32_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "--max-seq-len") == 0) {
            max_seq_len = (uint32_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "--kv-block-size") == 0) {
            cfg.kv_block_size = (uint32_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "--architecture") == 0) {
            if (!parse_arch(value, &cfg.architecture)) {
                fprintf(stderr, "unknown architecture: %s\n", value);
                return 1;
            }
        } else if (strcmp(key, "--quant") == 0) {
            if (!parse_quant(value, &cfg.quant_mode)) {
                fprintf(stderr, "unknown quant mode: %s\n", value);
                return 1;
            }
        } else {
            fprintf(stderr, "unknown flag: %s\n", key);
            return 1;
        }
    }

    if (max_seq_len == 0) {
        fprintf(stderr, "runtime-dryrun requires --max-seq-len\n");
        return 1;
    }

    status = microgemm_runtime_create(&cfg, max_seq_len, &runtime);
    if (status != MICROGEMM_STATUS_OK) {
        fprintf(stderr, "runtime-dryrun failed: %s\n", microgemm_status_string(status));
        return 1;
    }

    printf("architecture: %s\n", microgemm_format_architecture_name(cfg.architecture));
    printf("quant: %s\n", microgemm_format_quant_name(cfg.quant_mode));
    printf("max_seq_len: %u\n", max_seq_len);
    printf("scratch_bytes: %zu\n", microgemm_runtime_scratch_bytes(runtime));

    microgemm_runtime_destroy(runtime);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    if (strcmp(argv[1], "version") == 0) {
        printf("microgemm %u.%u\n", microgemm_version_major(), microgemm_version_minor());
        return 0;
    }

    if (strcmp(argv[1], "capabilities") == 0) {
        return print_capabilities();
    }

    if (strcmp(argv[1], "kernel-selftest") == 0) {
        return kernel_selftest();
    }

    if (strcmp(argv[1], "i8g-saturation-probe") == 0) {
        return i8g_saturation_probe();
    }

    if (strcmp(argv[1], "inspect") == 0) {
        if (argc < 3) {
            fprintf(stderr, "inspect requires a path\n");
            return 1;
        }
        return inspect_model(argv[2]);
    }

    if (strcmp(argv[1], "runtime-dryrun") == 0) {
        return runtime_dryrun(argc, argv);
    }

    if (strcmp(argv[1], "decode-smoke") == 0) {
        if (argc < 4) {
            fprintf(stderr, "decode-smoke requires a model path and flags\n");
            return 1;
        }
        return decode_smoke(argv[2], argc - 3, argv + 3);
    }

    if (strcmp(argv[1], "generate-ids") == 0) {
        if (argc < 4) {
            fprintf(stderr, "generate-ids requires a model path and flags\n");
            return 1;
        }
        return generate_ids(argv[2], argc - 3, argv + 3);
    }

    print_usage();
    return 1;
}

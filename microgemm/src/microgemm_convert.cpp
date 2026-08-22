#include "microgemm/microgemm_format.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

struct microgemm_hf_config {
    std::string model_type;
    std::string hidden_act;
    std::string rope_type;

    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t num_hidden_layers = 0;
    uint32_t num_attention_heads = 0;
    uint32_t num_key_value_heads = 0;
    uint32_t head_dim = 0;
    uint32_t vocab_size = 0;
    uint32_t max_position_embeddings = 0;

    double rms_norm_eps = 0.0;
    double rope_theta = 10000.0;
    double attention_logit_softcap = 0.0;
    double final_logit_softcap = 0.0;
    double query_pre_attn_scalar = 0.0;
    double embedding_multiplier = 1.0;
    double residual_multiplier = 1.0;
    double logits_scaling = 1.0;
    double rope_factor = 1.0;
    double rope_low_freq_factor = 1.0;
    double rope_high_freq_factor = 4.0;
    double partial_rotary_factor = 0.0;
    uint32_t rope_original_max_position_embeddings = 0;
    uint32_t rotary_dim = 0;
    uint32_t full_attention_interval = 4;
    uint32_t linear_conv_kernel_dim = 0;
    uint32_t linear_key_head_dim = 0;
    uint32_t linear_value_head_dim = 0;
    uint32_t linear_num_key_heads = 0;
    uint32_t linear_num_value_heads = 0;
    std::vector<std::string> layer_types;

    bool has_rope_scaling = false;
    bool attention_bias = false;
    bool mlp_bias = false;
    bool norm_offset = false;
    bool qk_norm = false;
    bool attention_output_gate = false;
    bool attention_output_gate_set = false;
    bool rope_interleaved = false;
    bool moe = false;
};

static bool read_text_file(const char* path, std::string* out_text) {
    std::ifstream file(path, std::ios::binary);
    std::ostringstream buffer;

    if (!file || out_text == NULL) {
        return false;
    }

    buffer << file.rdbuf();
    if (!file.good() && !file.eof()) {
        return false;
    }

    *out_text = buffer.str();
    return true;
}

static size_t skip_ws(const std::string& text, size_t pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
    return pos;
}

static bool find_key_value_start(const std::string& text, const char* key, size_t* out_pos) {
    std::string needle = "\"";
    size_t key_pos;
    size_t colon_pos;

    if (key == NULL || out_pos == NULL) {
        return false;
    }

    needle += key;
    needle += "\"";
    key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        return false;
    }

    colon_pos = text.find(':', key_pos + needle.size());
    if (colon_pos == std::string::npos) {
        return false;
    }

    *out_pos = skip_ws(text, colon_pos + 1);
    return true;
}

static bool parse_json_string_at(const std::string& text, size_t pos, std::string* out_value) {
    size_t cursor;
    std::string value;

    if (out_value == NULL || pos >= text.size() || text[pos] != '"') {
        return false;
    }

    cursor = pos + 1;
    while (cursor < text.size()) {
        char ch = text[cursor];
        if (ch == '\\') {
            if (cursor + 1 >= text.size()) {
                return false;
            }
            value.push_back(text[cursor + 1]);
            cursor += 2;
            continue;
        }
        if (ch == '"') {
            *out_value = value;
            return true;
        }
        value.push_back(ch);
        ++cursor;
    }

    return false;
}

static bool parse_json_double_at(const std::string& text, size_t pos, double* out_value) {
    const char* start;
    char* end_ptr;
    double value;

    if (out_value == NULL || pos >= text.size()) {
        return false;
    }

    start = text.c_str() + pos;
    value = std::strtod(start, &end_ptr);
    if (end_ptr == start) {
        return false;
    }

    *out_value = value;
    return true;
}

static bool parse_json_u32_at(const std::string& text, size_t pos, uint32_t* out_value) {
    double value;
    if (!parse_json_double_at(text, pos, &value)) {
        return false;
    }
    if (value < 0.0 || value > static_cast<double>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    *out_value = static_cast<uint32_t>(value);
    return true;
}

static bool parse_json_u64_at(const std::string& text, size_t pos, uint64_t* out_value) {
    const char* start;
    char* end_ptr;
    unsigned long long value;

    if (out_value == NULL || pos >= text.size()) {
        return false;
    }

    start = text.c_str() + pos;
    value = std::strtoull(start, &end_ptr, 10);
    if (end_ptr == start) {
        return false;
    }

    *out_value = static_cast<uint64_t>(value);
    return true;
}

static bool parse_json_bool_at(const std::string& text, size_t pos, bool* out_value) {
    if (out_value == NULL || pos >= text.size()) {
        return false;
    }
    if (text.compare(pos, 4, "true") == 0) {
        *out_value = true;
        return true;
    }
    if (text.compare(pos, 5, "false") == 0) {
        *out_value = false;
        return true;
    }
    return false;
}

static bool get_json_string(const std::string& text, const char* key, std::string* out_value) {
    size_t pos;
    return find_key_value_start(text, key, &pos) && parse_json_string_at(text, pos, out_value);
}

static bool get_json_u32(const std::string& text, const char* key, uint32_t* out_value) {
    size_t pos;
    return find_key_value_start(text, key, &pos) && parse_json_u32_at(text, pos, out_value);
}

static bool get_json_double(const std::string& text, const char* key, double* out_value) {
    size_t pos;
    return find_key_value_start(text, key, &pos) && parse_json_double_at(text, pos, out_value);
}

static bool get_json_bool(const std::string& text, const char* key, bool* out_value) {
    size_t pos;
    return find_key_value_start(text, key, &pos) && parse_json_bool_at(text, pos, out_value);
}

static bool get_json_object(const std::string& text, const char* key, std::string* out_object) {
    size_t pos;
    size_t cursor;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;

    if (out_object == NULL || !find_key_value_start(text, key, &pos)) {
        return false;
    }
    if (text.compare(pos, 4, "null") == 0) {
        return false;
    }
    if (pos >= text.size() || text[pos] != '{') {
        return false;
    }

    for (cursor = pos; cursor < text.size(); ++cursor) {
        char ch = text[cursor];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
            continue;
        }
        if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0) {
                *out_object = text.substr(pos, cursor - pos + 1);
                return true;
            }
        }
    }

    return false;
}

static bool get_json_string_array(
    const std::string& text,
    const char* key,
    std::vector<std::string>* out_values
) {
    size_t pos;
    size_t cursor;
    std::vector<std::string> values;

    if (out_values == NULL || !find_key_value_start(text, key, &pos)) {
        return false;
    }
    if (pos >= text.size() || text[pos] != '[') {
        return false;
    }
    cursor = skip_ws(text, pos + 1);
    while (cursor < text.size() && text[cursor] != ']') {
        std::string value;
        if (!parse_json_string_at(text, cursor, &value)) {
            return false;
        }
        values.push_back(value);
        cursor = text.find('"', cursor + 1);
        if (cursor == std::string::npos) {
            return false;
        }
        cursor = text.find('"', cursor + 1);
        if (cursor == std::string::npos) {
            return false;
        }
        cursor = skip_ws(text, cursor + 1);
        if (cursor < text.size() && text[cursor] == ',') {
            cursor = skip_ws(text, cursor + 1);
            continue;
        }
        if (cursor < text.size() && text[cursor] == ']') {
            break;
        }
        return false;
    }
    if (cursor >= text.size() || text[cursor] != ']') {
        return false;
    }
    *out_values = values;
    return true;
}

static uint32_t infer_head_dim(const microgemm_hf_config& cfg) {
    if (cfg.head_dim != 0) {
        return cfg.head_dim;
    }
    if (cfg.hidden_size == 0 || cfg.num_attention_heads == 0) {
        return 0;
    }
    if (cfg.hidden_size % cfg.num_attention_heads != 0) {
        return 0;
    }
    return cfg.hidden_size / cfg.num_attention_heads;
}

static uint32_t infer_kv_heads(const microgemm_hf_config& cfg) {
    if (cfg.num_key_value_heads != 0) {
        return cfg.num_key_value_heads;
    }
    return cfg.num_attention_heads;
}

static bool is_qwen_like_model_type(const std::string& model_type) {
    return model_type == "qwen2"
        || model_type == "qwen3"
        || model_type == "qwen3_moe";
}

static bool is_qwen35_model_type(const std::string& model_type) {
    return model_type == "qwen3_5_text";
}

static bool is_gemma_like_model_type(const std::string& model_type) {
    return model_type == "gemma"
        || model_type == "gemma2"
        || model_type == "gemma3"
        || model_type == "gemma3_text";
}

static bool is_phi_like_model_type(const std::string& model_type) {
    return model_type == "phi3"
        || model_type == "phi4"
        || model_type == "phi4_mini";
}

static bool is_granite_like_model_type(const std::string& model_type) {
    return model_type == "granite";
}

static bool is_glm4_like_model_type(const std::string& model_type) {
    return model_type == "glm4";
}

static bool is_supported_dense_model_type(const std::string& model_type) {
    return model_type == "llama"
        || model_type == "mistral"
        || is_qwen_like_model_type(model_type)
        || is_qwen35_model_type(model_type)
        || is_gemma_like_model_type(model_type)
        || is_phi_like_model_type(model_type)
        || is_granite_like_model_type(model_type)
        || is_glm4_like_model_type(model_type);
}

static bool is_gelu_hidden_act(const std::string& hidden_act) {
    return hidden_act == "gelu"
        || hidden_act == "gelu_new"
        || hidden_act == "gelu_fast"
        || hidden_act == "gelu_pytorch_tanh";
}

static bool is_supported_hidden_act(const std::string& hidden_act) {
    return hidden_act == "silu" || is_gelu_hidden_act(hidden_act);
}

static uint64_t estimate_int8_payload_bytes(const microgemm_hf_config& cfg) {
    uint32_t head_dim = infer_head_dim(cfg);
    uint32_t kv_heads = infer_kv_heads(cfg);
    uint64_t hidden = cfg.hidden_size;
    uint64_t inter = cfg.intermediate_size;
    uint64_t vocab = cfg.vocab_size;
    uint64_t max_pos = cfg.max_position_embeddings;
    uint64_t qkv_rows = static_cast<uint64_t>(cfg.num_attention_heads + 2u * kv_heads) * head_dim;
    uint64_t attn_cols = static_cast<uint64_t>(cfg.num_attention_heads) * head_dim;
    uint64_t total = 0;

    total += vocab * hidden;
    total += vocab * sizeof(float);
    total += hidden * sizeof(float);
    total += vocab * hidden;
    total += vocab * sizeof(float);
    total += max_pos * (head_dim / 2u) * sizeof(float);
    total += max_pos * (head_dim / 2u) * sizeof(float);

    total += static_cast<uint64_t>(cfg.num_hidden_layers) * hidden * sizeof(float);
    total += static_cast<uint64_t>(cfg.num_hidden_layers) * hidden * sizeof(float);
    total += static_cast<uint64_t>(cfg.num_hidden_layers) * qkv_rows * hidden;
    total += static_cast<uint64_t>(cfg.num_hidden_layers) * qkv_rows * sizeof(float);
    total += static_cast<uint64_t>(cfg.num_hidden_layers) * hidden * attn_cols;
    total += static_cast<uint64_t>(cfg.num_hidden_layers) * hidden * sizeof(float);
    total += static_cast<uint64_t>(cfg.num_hidden_layers) * (2u * inter) * hidden;
    total += static_cast<uint64_t>(cfg.num_hidden_layers) * (2u * inter) * sizeof(float);
    total += static_cast<uint64_t>(cfg.num_hidden_layers) * hidden * inter;
    total += static_cast<uint64_t>(cfg.num_hidden_layers) * hidden * sizeof(float);
    return total;
}

static bool load_hf_config_required_fields(
    const std::string& text,
    microgemm_hf_config* out_cfg
) {
    return get_json_u32(text, "hidden_size", &out_cfg->hidden_size)
        && get_json_u32(text, "intermediate_size", &out_cfg->intermediate_size)
        && get_json_u32(text, "num_hidden_layers", &out_cfg->num_hidden_layers)
        && get_json_u32(text, "num_attention_heads", &out_cfg->num_attention_heads)
        && get_json_u32(text, "vocab_size", &out_cfg->vocab_size)
        && get_json_u32(text, "max_position_embeddings", &out_cfg->max_position_embeddings);
}

static void load_hf_config_optional_fields(
    const std::string& text,
    microgemm_hf_config* out_cfg
) {
    std::string rope_scaling;
    std::string rope_parameters;
    uint32_t moe_value = 0;

    if (out_cfg == NULL) {
        return;
    }

    (void)get_json_u32(text, "num_key_value_heads", &out_cfg->num_key_value_heads);
    (void)get_json_u32(text, "head_dim", &out_cfg->head_dim);
    if (!get_json_string(text, "hidden_act", &out_cfg->hidden_act)) {
        (void)get_json_string(text, "hidden_activation", &out_cfg->hidden_act);
    }
    (void)get_json_double(text, "rms_norm_eps", &out_cfg->rms_norm_eps);
    (void)get_json_double(text, "rope_theta", &out_cfg->rope_theta);
    (void)get_json_double(text, "attn_logit_softcapping", &out_cfg->attention_logit_softcap);
    (void)get_json_double(text, "attention_logit_softcap", &out_cfg->attention_logit_softcap);
    (void)get_json_double(text, "final_logit_softcapping", &out_cfg->final_logit_softcap);
    (void)get_json_double(text, "final_logit_softcap", &out_cfg->final_logit_softcap);
    (void)get_json_double(text, "query_pre_attn_scalar", &out_cfg->query_pre_attn_scalar);
    (void)get_json_double(text, "embedding_multiplier", &out_cfg->embedding_multiplier);
    (void)get_json_double(text, "residual_multiplier", &out_cfg->residual_multiplier);
    (void)get_json_double(text, "logits_scaling", &out_cfg->logits_scaling);
    if (get_json_double(text, "attention_multiplier", &out_cfg->query_pre_attn_scalar)) {
        /* Granite stores the final attention scale directly instead of sqrt(head_dim). */
    }
    (void)get_json_double(text, "partial_rotary_factor", &out_cfg->partial_rotary_factor);
    (void)get_json_u32(text, "full_attention_interval", &out_cfg->full_attention_interval);
    (void)get_json_u32(text, "linear_conv_kernel_dim", &out_cfg->linear_conv_kernel_dim);
    (void)get_json_u32(text, "linear_key_head_dim", &out_cfg->linear_key_head_dim);
    (void)get_json_u32(text, "linear_value_head_dim", &out_cfg->linear_value_head_dim);
    (void)get_json_u32(text, "linear_num_key_heads", &out_cfg->linear_num_key_heads);
    (void)get_json_u32(text, "linear_num_value_heads", &out_cfg->linear_num_value_heads);
    (void)get_json_bool(text, "attention_bias", &out_cfg->attention_bias);
    (void)get_json_bool(text, "mlp_bias", &out_cfg->mlp_bias);
    (void)get_json_bool(text, "norm_offset", &out_cfg->norm_offset);
    if (!get_json_bool(text, "qk_norm", &out_cfg->qk_norm)) {
        (void)get_json_bool(text, "use_qk_norm", &out_cfg->qk_norm);
    }
    {
        bool attention_output_gate = false;
        if (get_json_bool(text, "attention_output_gate", &attention_output_gate)
                || get_json_bool(text, "attn_output_gate", &attention_output_gate)) {
            out_cfg->attention_output_gate = attention_output_gate;
            out_cfg->attention_output_gate_set = true;
        }
    }
    (void)get_json_bool(text, "rope_interleaved", &out_cfg->rope_interleaved);
    (void)get_json_string_array(text, "layer_types", &out_cfg->layer_types);

    if (get_json_object(text, "rope_parameters", &rope_parameters)) {
        (void)get_json_double(rope_parameters, "rope_theta", &out_cfg->rope_theta);
        (void)get_json_double(rope_parameters, "partial_rotary_factor", &out_cfg->partial_rotary_factor);
        if (!get_json_string(rope_parameters, "rope_type", &out_cfg->rope_type)) {
            (void)get_json_string(rope_parameters, "type", &out_cfg->rope_type);
        }
        if (get_json_double(rope_parameters, "factor", &out_cfg->rope_factor)) {
            out_cfg->has_rope_scaling = true;
        }
        (void)get_json_double(rope_parameters, "low_freq_factor", &out_cfg->rope_low_freq_factor);
        (void)get_json_double(rope_parameters, "high_freq_factor", &out_cfg->rope_high_freq_factor);
        (void)get_json_u32(
            rope_parameters,
            "original_max_position_embeddings",
            &out_cfg->rope_original_max_position_embeddings
        );
    }

    if ((get_json_u32(text, "num_experts", &moe_value) && moe_value != 0u)
            || (get_json_u32(text, "num_local_experts", &moe_value) && moe_value != 0u)
            || (get_json_u32(text, "n_routed_experts", &moe_value) && moe_value != 0u)
            || (get_json_u32(text, "moe_intermediate_size", &moe_value) && moe_value != 0u)) {
        out_cfg->moe = true;
    }

    if (get_json_object(text, "rope_scaling", &rope_scaling)) {
        out_cfg->has_rope_scaling = true;
        if (!get_json_string(rope_scaling, "rope_type", &out_cfg->rope_type)) {
            (void)get_json_string(rope_scaling, "type", &out_cfg->rope_type);
        }
        if (!get_json_double(rope_scaling, "factor", &out_cfg->rope_factor)) {
            out_cfg->rope_factor = 1.0;
        }
        (void)get_json_double(rope_scaling, "low_freq_factor", &out_cfg->rope_low_freq_factor);
        (void)get_json_double(rope_scaling, "high_freq_factor", &out_cfg->rope_high_freq_factor);
        (void)get_json_u32(
            rope_scaling,
            "original_max_position_embeddings",
            &out_cfg->rope_original_max_position_embeddings
        );
    }
}

static bool load_hf_config(const char* path, microgemm_hf_config* out_cfg, std::string* out_error) {
    std::string text;
    std::string text_config;
    std::string nested_model_type;
    std::string top_model_type;
    const std::string* config_text = &text;

    if (out_cfg == NULL) {
        return false;
    }
    if (!read_text_file(path, &text)) {
        if (out_error != NULL) {
            *out_error = "failed to read config.json";
        }
        return false;
    }

    if (!get_json_string(text, "model_type", &top_model_type)) {
        if (out_error != NULL) {
            *out_error = "config.json is missing required keys for MicroGemm";
        }
        return false;
    }
    out_cfg->model_type = top_model_type;

    if ((top_model_type == "gemma3" || top_model_type == "qwen3_5")
            && get_json_object(text, "text_config", &text_config)) {
        config_text = &text_config;
        if (get_json_string(text_config, "model_type", &nested_model_type)) {
            out_cfg->model_type = nested_model_type;
        } else if (top_model_type == "qwen3_5") {
            out_cfg->model_type = "qwen3_5_text";
        } else {
            out_cfg->model_type = "gemma3_text";
        }
        if (!load_hf_config_required_fields(text_config, out_cfg)) {
            if (out_error != NULL) {
                *out_error = "config.json text_config is missing required keys for MicroGemm";
            }
            return false;
        }
    } else if (!load_hf_config_required_fields(text, out_cfg)) {
        if (!get_json_object(text, "text_config", &text_config)
                || !load_hf_config_required_fields(text_config, out_cfg)) {
            if (out_error != NULL) {
                *out_error = "config.json is missing required keys for MicroGemm";
            }
            return false;
        }
        config_text = &text_config;
        if (get_json_string(text_config, "model_type", &nested_model_type)) {
            out_cfg->model_type = nested_model_type;
        } else if (top_model_type == "qwen3_5") {
            out_cfg->model_type = "qwen3_5_text";
        } else if (top_model_type == "gemma3") {
            out_cfg->model_type = "gemma3_text";
        }
    }

    load_hf_config_optional_fields(text, out_cfg);
    if (config_text != &text) {
        load_hf_config_optional_fields(*config_text, out_cfg);
    }

    if (out_cfg->hidden_act.empty()) {
        out_cfg->hidden_act = "silu";
    }
    if (out_cfg->rms_norm_eps == 0.0) {
        out_cfg->rms_norm_eps = 1e-5;
    }
    if (out_cfg->num_key_value_heads == 0) {
        out_cfg->num_key_value_heads = out_cfg->num_attention_heads;
    }
    if (out_cfg->model_type == "qwen3_moe") {
        out_cfg->moe = true;
    }
    if (out_cfg->model_type == "qwen3_5_text") {
        uint32_t layer_idx;
        uint32_t head_dim = infer_head_dim(*out_cfg);
        out_cfg->qk_norm = true;
        out_cfg->norm_offset = true;
        out_cfg->attention_output_gate = true;
        if (out_cfg->linear_conv_kernel_dim == 0u) {
            out_cfg->linear_conv_kernel_dim = 4u;
        }
        if (out_cfg->linear_key_head_dim == 0u) {
            out_cfg->linear_key_head_dim = head_dim;
        }
        if (out_cfg->linear_value_head_dim == 0u) {
            out_cfg->linear_value_head_dim = head_dim;
        }
        if (out_cfg->linear_num_key_heads == 0u) {
            out_cfg->linear_num_key_heads = out_cfg->num_attention_heads;
        }
        if (out_cfg->linear_num_value_heads == 0u) {
            out_cfg->linear_num_value_heads = out_cfg->num_attention_heads * 2u;
        }
        if (out_cfg->partial_rotary_factor <= 0.0) {
            out_cfg->partial_rotary_factor = 0.25;
        }
        out_cfg->rotary_dim = static_cast<uint32_t>(
            static_cast<double>(head_dim) * out_cfg->partial_rotary_factor
        );
        if (out_cfg->rotary_dim < 2u) {
            out_cfg->rotary_dim = 2u;
        }
        if ((out_cfg->rotary_dim & 1u) != 0u) {
            out_cfg->rotary_dim -= 1u;
        }
        if (out_cfg->full_attention_interval == 0u) {
            out_cfg->full_attention_interval = 4u;
        }
        if (out_cfg->layer_types.empty() && out_cfg->num_hidden_layers != 0u) {
            out_cfg->layer_types.resize(out_cfg->num_hidden_layers);
            for (layer_idx = 0u; layer_idx < out_cfg->num_hidden_layers; ++layer_idx) {
                out_cfg->layer_types[layer_idx] =
                    ((layer_idx + 1u) % out_cfg->full_attention_interval) == 0u
                    ? "full_attention"
                    : "linear_attention";
            }
        }
    }
    if (is_gemma_like_model_type(out_cfg->model_type)) {
        out_cfg->norm_offset = true;
    }
    if (is_granite_like_model_type(out_cfg->model_type)) {
        if (out_cfg->embedding_multiplier == 0.0) {
            out_cfg->embedding_multiplier = 1.0;
        }
        if (out_cfg->residual_multiplier == 0.0) {
            out_cfg->residual_multiplier = 1.0;
        }
        if (out_cfg->logits_scaling == 0.0) {
            out_cfg->logits_scaling = 1.0;
        }
    }
    if (is_glm4_like_model_type(out_cfg->model_type)) {
        uint32_t head_dim = infer_head_dim(*out_cfg);
        double partial_factor = out_cfg->partial_rotary_factor > 0.0
            ? out_cfg->partial_rotary_factor
            : 0.5;

        out_cfg->rope_interleaved = true;
        out_cfg->rotary_dim = static_cast<uint32_t>(
            static_cast<double>(head_dim) * partial_factor
        );
        if (out_cfg->rotary_dim < 2u) {
            out_cfg->rotary_dim = 2u;
        }
        if (out_cfg->rotary_dim > head_dim) {
            out_cfg->rotary_dim = head_dim;
        }
        if ((out_cfg->rotary_dim & 1u) != 0u) {
            out_cfg->rotary_dim -= 1u;
        }
    }

    return true;
}

static void collect_support_issues(const microgemm_hf_config& cfg, std::vector<std::string>* issues) {
    uint32_t head_dim = infer_head_dim(cfg);

    if (issues == NULL) {
        return;
    }
    if (!is_supported_dense_model_type(cfg.model_type)) {
        issues->push_back("unsupported model_type");
    }
    if (!is_supported_hidden_act(cfg.hidden_act)) {
        issues->push_back("hidden_act");
    }
    if (cfg.moe) {
        issues->push_back("moe");
    }
    if (cfg.attention_bias && !is_glm4_like_model_type(cfg.model_type)) {
        issues->push_back("attention_bias");
    }
    if (cfg.mlp_bias) {
        issues->push_back("mlp_bias");
    }
    if (cfg.attention_output_gate && !is_qwen35_model_type(cfg.model_type)) {
        issues->push_back("attention_output_gate");
    }
    if (cfg.rope_interleaved && !is_glm4_like_model_type(cfg.model_type)) {
        issues->push_back("rope_interleaved");
    }
    if (head_dim == 0 || (head_dim % 2u) != 0u) {
        issues->push_back("head_dim");
    }
    if (cfg.num_attention_heads == 0 || cfg.num_key_value_heads == 0
            || (cfg.num_attention_heads % cfg.num_key_value_heads) != 0u) {
        issues->push_back("gqa_layout");
    }
    if (cfg.has_rope_scaling) {
        std::string rope_type = cfg.rope_type.empty() ? "default" : cfg.rope_type;
        if (rope_type == "llama3") {
            if (!(cfg.rope_factor > 0.0)
                    || !(cfg.rope_low_freq_factor > 0.0)
                    || !(cfg.rope_high_freq_factor > cfg.rope_low_freq_factor)
                    || cfg.rope_original_max_position_embeddings == 0u) {
                issues->push_back("rope_scaling");
            }
        } else if (rope_type == "longrope") {
            issues->push_back("rope_scaling_longrope");
        } else if (rope_type != "default" || std::fabs(cfg.rope_factor - 1.0) > 1e-8) {
            issues->push_back("rope_scaling");
        }
    }
    if (is_qwen35_model_type(cfg.model_type)) {
        uint32_t layer_idx;
        if (cfg.layer_types.size() != cfg.num_hidden_layers) {
            issues->push_back("qwen35_layer_types");
        } else {
            for (layer_idx = 0u; layer_idx < cfg.num_hidden_layers; ++layer_idx) {
                const std::string& layer_type = cfg.layer_types[layer_idx];
                if (layer_type != "full_attention" && layer_type != "linear_attention") {
                    issues->push_back("qwen35_layer_types");
                    break;
                }
            }
        }
        if (cfg.linear_conv_kernel_dim == 0u
                || cfg.linear_key_head_dim == 0u
                || cfg.linear_value_head_dim == 0u
                || cfg.linear_num_key_heads == 0u
                || cfg.linear_num_value_heads == 0u
                || (cfg.linear_num_value_heads % cfg.linear_num_key_heads) != 0u) {
            issues->push_back("qwen35_linear_attention_dims");
        }
        if (cfg.rotary_dim == 0u || cfg.rotary_dim > head_dim || (cfg.rotary_dim & 1u) != 0u) {
            issues->push_back("qwen35_rotary_dim");
        }
    }
    if (is_glm4_like_model_type(cfg.model_type)) {
        if (cfg.rotary_dim == 0u || cfg.rotary_dim > head_dim || (cfg.rotary_dim & 1u) != 0u) {
            issues->push_back("glm4_rotary_dim");
        }
    }
}

static uint32_t map_architecture(const microgemm_hf_config& cfg) {
    if (cfg.model_type == "llama") {
        return MICROGEMM_ARCH_LLAMA_LIKE;
    }
    if (is_qwen35_model_type(cfg.model_type)) {
        return MICROGEMM_ARCH_QWEN35_LIKE;
    }
    if (is_qwen_like_model_type(cfg.model_type)) {
        return MICROGEMM_ARCH_QWEN2_LIKE;
    }
    if (cfg.model_type == "mistral") {
        return MICROGEMM_ARCH_MISTRAL_LIKE;
    }
    if (is_gemma_like_model_type(cfg.model_type)) {
        return MICROGEMM_ARCH_GEMMA_LIKE;
    }
    if (is_phi_like_model_type(cfg.model_type)) {
        return MICROGEMM_ARCH_PHI_LIKE;
    }
    if (is_granite_like_model_type(cfg.model_type)) {
        return MICROGEMM_ARCH_GRANITE_LIKE;
    }
    if (is_glm4_like_model_type(cfg.model_type)) {
        return MICROGEMM_ARCH_GLM4_LIKE;
    }
    return MICROGEMM_ARCH_UNKNOWN;
}

struct microgemm_safetensor_entry {
    std::string name;
    std::string dtype;
    std::string source_path;
    std::vector<uint64_t> shape;
    uint64_t data_start = 0;
    uint64_t data_end = 0;
    uint64_t file_data_base = 0;
};

struct microgemm_safetensor_file {
    std::vector<uint8_t> bytes;
    uint64_t data_base = 0;
    std::vector<microgemm_safetensor_entry> tensors;
    mutable std::string cached_path;
    mutable std::vector<uint8_t> cached_bytes;
};

struct microgemm_output_tensor {
    std::string name;
    uint32_t dtype = MICROGEMM_DTYPE_UNKNOWN;
    uint32_t rank = 0;
    std::array<uint64_t, MICROGEMM_MAX_TENSOR_RANK> dims = {0, 0, 0, 0};
    uint64_t byte_length = 0;
    std::vector<uint8_t> bytes;
    std::string spill_path;
};

static std::string g_tensor_spill_dir;
static uint64_t g_tensor_spill_counter = 0;
static const uint64_t MICROGEMM_CONVERT_SPILL_THRESHOLD_BYTES = 16ull * 1024ull * 1024ull;

static uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    uint64_t remainder;
    if (alignment <= 1u) {
        return value;
    }
    remainder = value % alignment;
    if (remainder == 0u) {
        return value;
    }
    return value + (alignment - remainder);
}

static bool read_binary_file(const char* path, std::vector<uint8_t>* out_bytes) {
    std::ifstream file;
    std::streampos end_pos;

    if (path == NULL || out_bytes == NULL) {
        return false;
    }

    file.open(path, std::ios::binary);
    if (!file) {
        return false;
    }

    file.seekg(0, std::ios::end);
    end_pos = file.tellg();
    if (end_pos < 0) {
        return false;
    }
    out_bytes->resize(static_cast<size_t>(end_pos));
    file.seekg(0, std::ios::beg);
    if (!out_bytes->empty()) {
        file.read(reinterpret_cast<char*>(out_bytes->data()), static_cast<std::streamsize>(out_bytes->size()));
        if (!file) {
            return false;
        }
    }
    return true;
}

static void begin_tensor_spill(const char* output_path) {
    std::error_code ec;
    g_tensor_spill_counter = 0;
    g_tensor_spill_dir.clear();
    if (output_path == NULL || output_path[0] == '\0') {
        return;
    }
    g_tensor_spill_dir = std::string(output_path) + ".tensor_parts";
    std::filesystem::remove_all(std::filesystem::path(g_tensor_spill_dir), ec);
    ec.clear();
    std::filesystem::create_directories(std::filesystem::path(g_tensor_spill_dir), ec);
    if (ec) {
        g_tensor_spill_dir.clear();
    }
}

static void end_tensor_spill(void) {
    std::error_code ec;
    if (!g_tensor_spill_dir.empty()) {
        std::filesystem::remove_all(std::filesystem::path(g_tensor_spill_dir), ec);
    }
    g_tensor_spill_dir.clear();
    g_tensor_spill_counter = 0;
}

static bool write_binary_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    if (!bytes.empty()) {
        out.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
    }
    return out.good();
}

static uint64_t read_u64_le(const uint8_t* src) {
    uint64_t value = 0;
    unsigned int shift;
    for (shift = 0; shift < 64u; shift += 8u) {
        value |= static_cast<uint64_t>(src[shift / 8u]) << shift;
    }
    return value;
}

static bool parse_json_string_token(
    const std::string& text,
    size_t pos,
    std::string* out_value,
    size_t* out_next
) {
    size_t cursor;
    std::string value;

    if (out_value == NULL || out_next == NULL || pos >= text.size() || text[pos] != '"') {
        return false;
    }

    cursor = pos + 1;
    while (cursor < text.size()) {
        char ch = text[cursor];
        if (ch == '\\') {
            if (cursor + 1 >= text.size()) {
                return false;
            }
            value.push_back(text[cursor + 1]);
            cursor += 2;
            continue;
        }
        if (ch == '"') {
            *out_value = value;
            *out_next = cursor + 1;
            return true;
        }
        value.push_back(ch);
        ++cursor;
    }

    return false;
}

static bool find_json_value_end(const std::string& text, size_t pos, size_t* out_end) {
    size_t cursor;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;

    if (out_end == NULL || pos >= text.size()) {
        return false;
    }

    if (text[pos] == '{' || text[pos] == '[') {
        char open_ch = text[pos];
        char close_ch = open_ch == '{' ? '}' : ']';

        for (cursor = pos; cursor < text.size(); ++cursor) {
            char ch = text[cursor];
            if (in_string) {
                if (escaped) {
                    escaped = false;
                } else if (ch == '\\') {
                    escaped = true;
                } else if (ch == '"') {
                    in_string = false;
                }
                continue;
            }

            if (ch == '"') {
                in_string = true;
                continue;
            }
            if (ch == open_ch) {
                ++depth;
            } else if (ch == close_ch) {
                --depth;
                if (depth == 0) {
                    *out_end = cursor + 1;
                    return true;
                }
            }
        }
        return false;
    }

    if (text[pos] == '"') {
        std::string ignored;
        return parse_json_string_token(text, pos, &ignored, out_end);
    }

    cursor = pos;
    while (cursor < text.size()) {
        char ch = text[cursor];
        if (ch == ',' || ch == '}' || ch == ']' || std::isspace(static_cast<unsigned char>(ch)) != 0) {
            *out_end = cursor;
            return true;
        }
        ++cursor;
    }

    *out_end = cursor;
    return true;
}

static std::string join_path(const std::string& left, const char* right) {
    std::string result = left;
    if (result.empty()) {
        return std::string(right);
    }
    if (result.back() != '/' && result.back() != '\\') {
        result.push_back('/');
    }
    result += right;
    return result;
}

static bool parse_json_u64_array(
    const std::string& text,
    size_t pos,
    std::vector<uint64_t>* out_values,
    size_t* out_next
) {
    size_t cursor;

    if (out_values == NULL || out_next == NULL || pos >= text.size() || text[pos] != '[') {
        return false;
    }

    out_values->clear();
    cursor = skip_ws(text, pos + 1);
    if (cursor < text.size() && text[cursor] == ']') {
        *out_next = cursor + 1;
        return true;
    }

    while (cursor < text.size()) {
        uint64_t value64 = 0;
        if (!parse_json_u64_at(text, cursor, &value64)) {
            return false;
        }
        out_values->push_back(value64);
        while (cursor < text.size() && text[cursor] != ',' && text[cursor] != ']') {
            ++cursor;
        }
        if (cursor >= text.size()) {
            return false;
        }
        if (text[cursor] == ']') {
            *out_next = cursor + 1;
            return true;
        }
        cursor = skip_ws(text, cursor + 1);
    }

    return false;
}

static bool parse_safetensor_object(
    const std::string& text,
    size_t pos,
    microgemm_safetensor_entry* out_entry,
    size_t* out_next
) {
    size_t cursor;

    if (out_entry == NULL || out_next == NULL || pos >= text.size() || text[pos] != '{') {
        return false;
    }

    cursor = skip_ws(text, pos + 1);
    while (cursor < text.size() && text[cursor] != '}') {
        std::string key;
        size_t value_pos;
        size_t next_after_key;
        size_t next_after_value;
        std::string string_value;
        std::vector<uint64_t> int_values;

        if (!parse_json_string_token(text, cursor, &key, &next_after_key)) {
            return false;
        }
        value_pos = skip_ws(text, next_after_key);
        if (value_pos >= text.size() || text[value_pos] != ':') {
            return false;
        }
        value_pos = skip_ws(text, value_pos + 1);

        if (key == "dtype") {
            if (!parse_json_string_token(text, value_pos, &out_entry->dtype, &next_after_value)) {
                return false;
            }
        } else if (key == "shape") {
            if (!parse_json_u64_array(text, value_pos, &out_entry->shape, &next_after_value)) {
                return false;
            }
        } else if (key == "data_offsets") {
            if (!parse_json_u64_array(text, value_pos, &int_values, &next_after_value) || int_values.size() != 2u) {
                return false;
            }
            out_entry->data_start = int_values[0];
            out_entry->data_end = int_values[1];
        } else if (parse_json_string_token(text, value_pos, &string_value, &next_after_value)) {
            /* ignore */
        } else if (!find_json_value_end(text, value_pos, &next_after_value)) {
            return false;
        }

        cursor = skip_ws(text, next_after_value);
        if (cursor < text.size() && text[cursor] == ',') {
            cursor = skip_ws(text, cursor + 1);
        }
    }

    if (cursor >= text.size() || text[cursor] != '}') {
        return false;
    }

    *out_next = cursor + 1;
    return !out_entry->dtype.empty() && !out_entry->shape.empty() && out_entry->data_end >= out_entry->data_start;
}

static bool parse_safetensors_header(
    const std::string& header_text,
    std::vector<microgemm_safetensor_entry>* out_tensors,
    std::string* out_error
) {
    size_t cursor;

    if (out_tensors == NULL) {
        return false;
    }

    out_tensors->clear();
    cursor = skip_ws(header_text, 0);
    if (cursor >= header_text.size() || header_text[cursor] != '{') {
        if (out_error != NULL) {
            *out_error = "invalid safetensors header json";
        }
        return false;
    }

    cursor = skip_ws(header_text, cursor + 1);
    while (cursor < header_text.size() && header_text[cursor] != '}') {
        std::string key;
        size_t next_after_key;
        size_t value_pos;
        size_t next_after_value;

        if (!parse_json_string_token(header_text, cursor, &key, &next_after_key)) {
            if (out_error != NULL) {
                *out_error = "failed to parse safetensors key";
            }
            return false;
        }
        value_pos = skip_ws(header_text, next_after_key);
        if (value_pos >= header_text.size() || header_text[value_pos] != ':') {
            if (out_error != NULL) {
                *out_error = "failed to parse safetensors ':'";
            }
            return false;
        }
        value_pos = skip_ws(header_text, value_pos + 1);

        if (key == "__metadata__") {
            if (!find_json_value_end(header_text, value_pos, &next_after_value)) {
                if (out_error != NULL) {
                    *out_error = "failed to skip safetensors metadata";
                }
                return false;
            }
        } else {
            microgemm_safetensor_entry entry;
            entry.name = key;
            if (!parse_safetensor_object(header_text, value_pos, &entry, &next_after_value)) {
                if (out_error != NULL) {
                    *out_error = "failed to parse safetensors tensor object";
                }
                return false;
            }
            out_tensors->push_back(entry);
        }

        cursor = skip_ws(header_text, next_after_value);
        if (cursor < header_text.size() && header_text[cursor] == ',') {
            cursor = skip_ws(header_text, cursor + 1);
        }
    }

    if (cursor >= header_text.size() || header_text[cursor] != '}') {
        if (out_error != NULL) {
            *out_error = "unterminated safetensors header object";
        }
        return false;
    }

    return true;
}

static bool load_safetensors_file(
    const char* path,
    microgemm_safetensor_file* out_file,
    std::string* out_error
) {
    uint64_t header_len;
    uint64_t data_base;
    std::string header_text;
    size_t tensor_idx;

    if (path == NULL || out_file == NULL) {
        return false;
    }
    if (!read_binary_file(path, &out_file->bytes)) {
        if (out_error != NULL) {
            *out_error = std::string("failed to read safetensors file: ") + path;
        }
        return false;
    }
    if (out_file->bytes.size() < 8u) {
        if (out_error != NULL) {
            *out_error = "safetensors file is too small";
        }
        return false;
    }

    header_len = read_u64_le(out_file->bytes.data());
    data_base = 8u + header_len;
    if (header_len > static_cast<uint64_t>(out_file->bytes.size()) || data_base > out_file->bytes.size()) {
        if (out_error != NULL) {
            *out_error = "safetensors header length is invalid";
        }
        return false;
    }

    header_text.assign(
        reinterpret_cast<const char*>(out_file->bytes.data() + 8u),
        static_cast<size_t>(header_len)
    );
    if (!parse_safetensors_header(header_text, &out_file->tensors, out_error)) {
        return false;
    }

    out_file->data_base = data_base;
    for (tensor_idx = 0; tensor_idx < out_file->tensors.size(); ++tensor_idx) {
        microgemm_safetensor_entry& entry = out_file->tensors[tensor_idx];
        uint64_t abs_end = data_base + entry.data_end;
        if (entry.data_end < entry.data_start || abs_end > out_file->bytes.size()) {
            if (out_error != NULL) {
                *out_error = "safetensors tensor offset is invalid";
            }
            return false;
        }
        entry.source_path = path;
        entry.file_data_base = data_base;
    }

    return true;
}

static bool load_safetensors_shards(
    const std::vector<std::string>& shard_paths,
    microgemm_safetensor_file* out_file,
    std::string* out_error
) {
    std::set<std::string> names;
    size_t shard_idx;

    if (out_file == NULL || shard_paths.empty()) {
        return false;
    }

    out_file->bytes.clear();
    out_file->data_base = 0;
    out_file->tensors.clear();
    out_file->cached_path.clear();
    out_file->cached_bytes.clear();

    for (shard_idx = 0; shard_idx < shard_paths.size(); ++shard_idx) {
        microgemm_safetensor_file shard;
        size_t tensor_idx;

        if (!load_safetensors_file(shard_paths[shard_idx].c_str(), &shard, out_error)) {
            return false;
        }
        for (tensor_idx = 0; tensor_idx < shard.tensors.size(); ++tensor_idx) {
            if (!names.insert(shard.tensors[tensor_idx].name).second) {
                if (out_error != NULL) {
                    *out_error = std::string("duplicate tensor across safetensors shards: ")
                        + shard.tensors[tensor_idx].name;
                }
                return false;
            }
            out_file->tensors.push_back(std::move(shard.tensors[tensor_idx]));
        }
    }

    return true;
}

static const microgemm_safetensor_entry* find_safetensor_entry(
    const microgemm_safetensor_file& file,
    const char* name
) {
    size_t i;
    for (i = 0; i < file.tensors.size(); ++i) {
        if (file.tensors[i].name == name) {
            return &file.tensors[i];
        }
    }
    return NULL;
}

static size_t safetensor_numel(const microgemm_safetensor_entry& entry) {
    size_t i;
    size_t numel = 1u;
    for (i = 0; i < entry.shape.size(); ++i) {
        numel *= static_cast<size_t>(entry.shape[i]);
    }
    return numel;
}

static size_t safetensor_element_size(const std::string& dtype) {
    if (dtype == "F32") {
        return sizeof(float);
    }
    if (dtype == "F16" || dtype == "BF16") {
        return sizeof(uint16_t);
    }
    return 0u;
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16u;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static float f16_to_f32(uint16_t value) {
    uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16u;
    uint32_t exp = (value >> 10u) & 0x1fu;
    uint32_t mant = value & 0x03ffu;
    uint32_t bits;
    float out = 0.0f;

    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            exp = 127u - 15u + 1u;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1u;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | (exp << 23u) | (mant << 13u);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13u);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23u) | (mant << 13u);
    }

    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static bool decode_tensor_f32(
    const microgemm_safetensor_file& file,
    const microgemm_safetensor_entry& entry,
    std::vector<float>* out_values,
    std::string* out_error
) {
    const uint8_t* src;
    const std::vector<uint8_t>* source_bytes;
    uint64_t data_base;
    size_t numel;
    size_t elem_size;
    size_t expected_bytes;
    size_t i;

    if (out_values == NULL) {
        return false;
    }

    numel = safetensor_numel(entry);
    elem_size = safetensor_element_size(entry.dtype);
    expected_bytes = numel * elem_size;
    if (elem_size == 0u || expected_bytes != static_cast<size_t>(entry.data_end - entry.data_start)) {
        if (out_error != NULL) {
            *out_error = "unsupported dtype or invalid tensor byte length";
        }
        return false;
    }

    source_bytes = &file.bytes;
    data_base = file.data_base;
    if (source_bytes->empty() && !entry.source_path.empty()) {
        if (file.cached_path != entry.source_path) {
            if (!read_binary_file(entry.source_path.c_str(), &file.cached_bytes)) {
                if (out_error != NULL) {
                    *out_error = std::string("failed to read safetensors shard: ") + entry.source_path;
                }
                return false;
            }
            file.cached_path = entry.source_path;
        }
        source_bytes = &file.cached_bytes;
        data_base = entry.file_data_base;
    }
    if (source_bytes->empty() || data_base + entry.data_end > source_bytes->size()) {
        if (out_error != NULL) {
            *out_error = std::string("safetensors tensor offset is invalid for ") + entry.name;
        }
        return false;
    }

    src = source_bytes->data() + static_cast<size_t>(data_base + entry.data_start);
    out_values->resize(numel);
    if (entry.dtype == "F32") {
        std::memcpy(out_values->data(), src, expected_bytes);
        return true;
    }

    for (i = 0; i < numel; ++i) {
        uint16_t word = static_cast<uint16_t>(src[2u * i]) | (static_cast<uint16_t>(src[2u * i + 1u]) << 8u);
        (*out_values)[i] = entry.dtype == "BF16" ? bf16_to_f32(word) : f16_to_f32(word);
    }
    return true;
}

static bool expect_shape(
    const microgemm_safetensor_entry& entry,
    const std::vector<uint64_t>& expected_shape,
    std::string* out_error
) {
    size_t i;
    if (entry.shape.size() != expected_shape.size()) {
        if (out_error != NULL) {
            *out_error = std::string("tensor rank mismatch for ") + entry.name;
        }
        return false;
    }
    for (i = 0; i < expected_shape.size(); ++i) {
        if (entry.shape[i] != expected_shape[i]) {
            if (out_error != NULL) {
                *out_error = std::string("tensor shape mismatch for ") + entry.name;
            }
            return false;
        }
    }
    return true;
}

static bool quant_mode_is_i4_storage(uint32_t quant_mode) {
    return quant_mode == MICROGEMM_QUANT_INT4
        || quant_mode == MICROGEMM_QUANT_INT4G128;
}

static bool quant_mode_is_i8_storage(uint32_t quant_mode) {
    return quant_mode == MICROGEMM_QUANT_INT8
        || quant_mode == MICROGEMM_QUANT_INT8G128;
}

static bool quant_mode_is_groupwise(uint32_t quant_mode) {
    return quant_mode == MICROGEMM_QUANT_INT8G128
        || quant_mode == MICROGEMM_QUANT_INT4G128;
}

static uint64_t quant_group_count(uint64_t cols) {
    return (cols + MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE - 1u)
        / MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
}

static void quantize_rows_i8(
    const std::vector<float>& src,
    uint64_t rows,
    uint64_t cols,
    std::vector<int8_t>* out_q,
    std::vector<float>* out_scales,
    std::vector<int32_t>* out_row_sums
) {
    uint64_t row;
    uint64_t col;

    out_q->assign(static_cast<size_t>(rows * cols), 0);
    out_scales->assign(static_cast<size_t>(rows), 1.0f);
    out_row_sums->assign(static_cast<size_t>(rows), 0);

    for (row = 0; row < rows; ++row) {
        float max_abs = 0.0f;
        float scale;
        int64_t row_sum = 0;
        for (col = 0; col < cols; ++col) {
            float v = src[static_cast<size_t>(row * cols + col)];
            float abs_v = std::fabs(v);
            if (abs_v > max_abs) {
                max_abs = abs_v;
            }
        }

        scale = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
        (*out_scales)[static_cast<size_t>(row)] = scale;
        for (col = 0; col < cols; ++col) {
            long q = 0;
            if (max_abs > 0.0f) {
                q = std::lround(src[static_cast<size_t>(row * cols + col)] / scale);
                if (q < -127l) {
                    q = -127l;
                }
                if (q > 127l) {
                    q = 127l;
                }
            }
            (*out_q)[static_cast<size_t>(row * cols + col)] = static_cast<int8_t>(q);
            row_sum += q;
        }
        (*out_row_sums)[static_cast<size_t>(row)] = static_cast<int32_t>(row_sum);
    }
}

static void quantize_rows_i8_groupwise(
    const std::vector<float>& src,
    uint64_t rows,
    uint64_t cols,
    std::vector<int8_t>* out_q,
    std::vector<float>* out_scales,
    std::vector<int32_t>* out_row_sums
) {
    const uint64_t groups = quant_group_count(cols);

    out_q->assign(static_cast<size_t>(rows * cols), 0);
    out_scales->assign(static_cast<size_t>(rows * groups), 1.0f);
    out_row_sums->assign(static_cast<size_t>(rows * groups), 0);

    for (uint64_t row = 0; row < rows; ++row) {
        for (uint64_t group = 0; group < groups; ++group) {
            const uint64_t begin = group * MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            uint64_t end = begin + MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            float max_abs = 0.0f;
            float scale;
            int64_t row_sum = 0;

            if (end > cols) {
                end = cols;
            }
            for (uint64_t col = begin; col < end; ++col) {
                float v = src[static_cast<size_t>(row * cols + col)];
                float abs_v = std::fabs(v);
                if (abs_v > max_abs) {
                    max_abs = abs_v;
                }
            }

            scale = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
            (*out_scales)[static_cast<size_t>(row * groups + group)] = scale;
            for (uint64_t col = begin; col < end; ++col) {
                long q = 0;
                if (max_abs > 0.0f) {
                    q = std::lround(src[static_cast<size_t>(row * cols + col)] / scale);
                    if (q < -127l) {
                        q = -127l;
                    }
                    if (q > 127l) {
                        q = 127l;
                    }
                }
                (*out_q)[static_cast<size_t>(row * cols + col)] = static_cast<int8_t>(q);
                row_sum += q;
            }
            (*out_row_sums)[static_cast<size_t>(row * groups + group)] =
                static_cast<int32_t>(row_sum);
        }
    }
}

static uint8_t pack_i4_value(long q) {
    if (q < -7l) {
        q = -7l;
    }
    if (q > 7l) {
        q = 7l;
    }
    return static_cast<uint8_t>(static_cast<int8_t>(q)) & 0x0fu;
}

static void quantize_rows_i4(
    const std::vector<float>& src,
    uint64_t rows,
    uint64_t cols,
    std::vector<uint8_t>* out_q,
    std::vector<float>* out_scales,
    std::vector<int32_t>* out_row_sums
) {
    uint64_t row;
    uint64_t col;
    uint64_t row_bytes = (cols + 1u) / 2u;

    out_q->assign(static_cast<size_t>(rows * row_bytes), 0);
    out_scales->assign(static_cast<size_t>(rows), 1.0f);
    out_row_sums->assign(static_cast<size_t>(rows), 0);

    for (row = 0; row < rows; ++row) {
        float max_abs = 0.0f;
        float scale;
        int64_t row_sum = 0;
        for (col = 0; col < cols; ++col) {
            float v = src[static_cast<size_t>(row * cols + col)];
            float abs_v = std::fabs(v);
            if (abs_v > max_abs) {
                max_abs = abs_v;
            }
        }

        scale = max_abs > 0.0f ? (max_abs / 7.0f) : 1.0f;
        (*out_scales)[static_cast<size_t>(row)] = scale;
        for (col = 0; col < cols; ++col) {
            long q = 0;
            uint8_t packed;
            size_t byte_idx = static_cast<size_t>(row * row_bytes + col / 2u);
            if (max_abs > 0.0f) {
                q = std::lround(src[static_cast<size_t>(row * cols + col)] / scale);
            }
            if (q < -7l) {
                q = -7l;
            }
            if (q > 7l) {
                q = 7l;
            }
            packed = pack_i4_value(q);
            if ((col & 1u) == 0u) {
                (*out_q)[byte_idx] = static_cast<uint8_t>((*out_q)[byte_idx] | packed);
            } else {
                (*out_q)[byte_idx] = static_cast<uint8_t>((*out_q)[byte_idx] | (packed << 4));
            }
            row_sum += q;
        }
        (*out_row_sums)[static_cast<size_t>(row)] = static_cast<int32_t>(row_sum);
    }
}

static void quantize_rows_i4_groupwise(
    const std::vector<float>& src,
    uint64_t rows,
    uint64_t cols,
    std::vector<uint8_t>* out_q,
    std::vector<float>* out_scales,
    std::vector<int32_t>* out_row_sums
) {
    const uint64_t groups = quant_group_count(cols);
    const uint64_t row_bytes = (cols + 1u) / 2u;

    out_q->assign(static_cast<size_t>(rows * row_bytes), 0);
    out_scales->assign(static_cast<size_t>(rows * groups), 1.0f);
    out_row_sums->assign(static_cast<size_t>(rows * groups), 0);

    for (uint64_t row = 0; row < rows; ++row) {
        for (uint64_t group = 0; group < groups; ++group) {
            const uint64_t begin = group * MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            uint64_t end = begin + MICROGEMM_GROUPWISE_QUANT_GROUP_SIZE;
            float max_abs = 0.0f;
            float scale;
            int64_t row_sum = 0;

            if (end > cols) {
                end = cols;
            }
            for (uint64_t col = begin; col < end; ++col) {
                float v = src[static_cast<size_t>(row * cols + col)];
                float abs_v = std::fabs(v);
                if (abs_v > max_abs) {
                    max_abs = abs_v;
                }
            }

            scale = max_abs > 0.0f ? (max_abs / 7.0f) : 1.0f;
            (*out_scales)[static_cast<size_t>(row * groups + group)] = scale;
            for (uint64_t col = begin; col < end; ++col) {
                long q = 0;
                uint8_t packed;
                size_t byte_idx = static_cast<size_t>(row * row_bytes + col / 2u);
                if (max_abs > 0.0f) {
                    q = std::lround(src[static_cast<size_t>(row * cols + col)] / scale);
                }
                if (q < -7l) {
                    q = -7l;
                }
                if (q > 7l) {
                    q = 7l;
                }
                packed = pack_i4_value(q);
                if ((col & 1u) == 0u) {
                    (*out_q)[byte_idx] = static_cast<uint8_t>((*out_q)[byte_idx] | packed);
                } else {
                    (*out_q)[byte_idx] = static_cast<uint8_t>((*out_q)[byte_idx] | (packed << 4));
                }
                row_sum += q;
            }
            (*out_row_sums)[static_cast<size_t>(row * groups + group)] =
                static_cast<int32_t>(row_sum);
        }
    }
}

static std::vector<float> concat_rows(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& c
) {
    std::vector<float> out;
    out.reserve(a.size() + b.size() + c.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    out.insert(out.end(), c.begin(), c.end());
    return out;
}

static std::vector<float> concat_rows(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    std::vector<float> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

static std::vector<uint8_t> floats_to_bytes(const std::vector<float>& values) {
    std::vector<uint8_t> out(values.size() * sizeof(float));
    if (!out.empty()) {
        std::memcpy(out.data(), values.data(), out.size());
    }
    return out;
}

static void scale_float_values(std::vector<float>* values, float scale) {
    size_t i;
    if (values == NULL || scale == 1.0f) {
        return;
    }
    for (i = 0; i < values->size(); ++i) {
        (*values)[i] *= scale;
    }
}

static std::vector<uint8_t> i8_to_bytes(const std::vector<int8_t>& values) {
    std::vector<uint8_t> out(values.size());
    if (!out.empty()) {
        std::memcpy(out.data(), values.data(), out.size());
    }
    return out;
}

static std::vector<uint8_t> i32_to_bytes(const std::vector<int32_t>& values) {
    std::vector<uint8_t> out(values.size() * sizeof(int32_t));
    if (!out.empty()) {
        std::memcpy(out.data(), values.data(), out.size());
    }
    return out;
}

static void add_output_tensor(
    std::vector<microgemm_output_tensor>* out_tensors,
    const std::string& name,
    uint32_t dtype,
    const std::vector<uint64_t>& shape,
    std::vector<uint8_t>&& bytes
) {
    microgemm_output_tensor tensor;
    size_t i;
    uint64_t byte_length = static_cast<uint64_t>(bytes.size());

    tensor.name = name;
    tensor.dtype = dtype;
    tensor.rank = static_cast<uint32_t>(shape.size());
    for (i = 0; i < shape.size() && i < MICROGEMM_MAX_TENSOR_RANK; ++i) {
        tensor.dims[i] = shape[i];
    }
    tensor.byte_length = byte_length;
    if (!g_tensor_spill_dir.empty() && byte_length >= MICROGEMM_CONVERT_SPILL_THRESHOLD_BYTES) {
        std::ostringstream path_builder;
        path_builder << g_tensor_spill_dir << "/tensor_" << g_tensor_spill_counter++ << ".bin";
        if (write_binary_file(path_builder.str(), bytes)) {
            tensor.spill_path = path_builder.str();
            std::vector<uint8_t>().swap(bytes);
        }
    }
    if (tensor.spill_path.empty()) {
        tensor.bytes = std::move(bytes);
    }
    out_tensors->push_back(std::move(tensor));
}

static double apply_llama3_rope_scaling(double inv_freq, const microgemm_hf_config& cfg) {
    const double pi = 3.14159265358979323846264338327950288;
    double wavelen;
    double low_freq_wavelen;
    double high_freq_wavelen;
    double smooth;

    if (inv_freq <= 0.0 || cfg.rope_factor <= 0.0
            || cfg.rope_low_freq_factor <= 0.0
            || cfg.rope_high_freq_factor <= cfg.rope_low_freq_factor
            || cfg.rope_original_max_position_embeddings == 0u) {
        return inv_freq;
    }

    wavelen = 2.0 * pi / inv_freq;
    low_freq_wavelen =
        static_cast<double>(cfg.rope_original_max_position_embeddings)
        / cfg.rope_low_freq_factor;
    high_freq_wavelen =
        static_cast<double>(cfg.rope_original_max_position_embeddings)
        / cfg.rope_high_freq_factor;

    if (wavelen < high_freq_wavelen) {
        return inv_freq;
    }
    if (wavelen > low_freq_wavelen) {
        return inv_freq / cfg.rope_factor;
    }

    smooth =
        (static_cast<double>(cfg.rope_original_max_position_embeddings) / wavelen
            - cfg.rope_low_freq_factor)
        / (cfg.rope_high_freq_factor - cfg.rope_low_freq_factor);
    return (1.0 - smooth) * (inv_freq / cfg.rope_factor) + smooth * inv_freq;
}

static double rope_inv_freq_for_pair(
    const microgemm_hf_config& cfg,
    uint32_t head_dim,
    uint32_t pair_idx
) {
    double exponent = (2.0 * static_cast<double>(pair_idx)) / static_cast<double>(head_dim);
    double inv_freq = std::pow(cfg.rope_theta, -exponent);
    std::string rope_type = cfg.rope_type.empty() ? "default" : cfg.rope_type;

    if (cfg.has_rope_scaling && rope_type == "llama3") {
        inv_freq = apply_llama3_rope_scaling(inv_freq, cfg);
    }
    return inv_freq;
}

static void build_rope_cache(
    const microgemm_hf_config& cfg,
    uint32_t head_dim,
    std::vector<float>* out_cos,
    std::vector<float>* out_sin
) {
    uint32_t position;
    uint32_t pair_idx;
    uint32_t half_dim = head_dim / 2u;
    uint32_t rope_dim = head_dim;
    uint32_t rope_half_dim;

    if ((is_qwen35_model_type(cfg.model_type) || is_glm4_like_model_type(cfg.model_type))
            && cfg.rotary_dim != 0u
            && cfg.rotary_dim <= head_dim) {
        rope_dim = cfg.rotary_dim;
    }
    rope_half_dim = rope_dim / 2u;

    out_cos->resize(static_cast<size_t>(cfg.max_position_embeddings) * half_dim);
    out_sin->resize(static_cast<size_t>(cfg.max_position_embeddings) * half_dim);

    for (position = 0; position < cfg.max_position_embeddings; ++position) {
        for (pair_idx = 0; pair_idx < half_dim; ++pair_idx) {
            size_t index = static_cast<size_t>(position) * half_dim + pair_idx;
            if (pair_idx < rope_half_dim) {
                double inv_freq = rope_inv_freq_for_pair(cfg, rope_dim, pair_idx);
                double angle = static_cast<double>(position) * inv_freq;
                (*out_cos)[index] = static_cast<float>(std::cos(angle));
                (*out_sin)[index] = static_cast<float>(std::sin(angle));
            } else {
                (*out_cos)[index] = 1.0f;
                (*out_sin)[index] = 0.0f;
            }
        }
    }
}

static bool write_microgemm_file(
    const char* path,
    const microgemm_config& cfg,
    const std::vector<microgemm_output_tensor>& tensors,
    uint32_t alignment,
    std::string* out_error
) {
    microgemm_file_header header;
    std::vector<microgemm_tensor_entry> directory(tensors.size());
    uint64_t directory_offset = sizeof(microgemm_file_header) + sizeof(microgemm_config);
    uint64_t data_offset = align_up_u64(
        directory_offset + static_cast<uint64_t>(tensors.size()) * sizeof(microgemm_tensor_entry),
        alignment
    );
    uint64_t cursor = data_offset;
    size_t i;
    std::ofstream out(path, std::ios::binary);

    if (!out) {
        if (out_error != NULL) {
            *out_error = "failed to open output file";
        }
        return false;
    }

    std::memset(&header, 0, sizeof(header));
    header.magic[0] = MICROGEMM_MAGIC_0;
    header.magic[1] = MICROGEMM_MAGIC_1;
    header.magic[2] = MICROGEMM_MAGIC_2;
    header.magic[3] = MICROGEMM_MAGIC_3;
    header.version_major = 0;
    header.version_minor = 1;
    header.header_bytes = sizeof(microgemm_file_header);
    header.config_bytes = sizeof(microgemm_config);
    header.tensor_entry_bytes = sizeof(microgemm_tensor_entry);
    header.tensor_count = static_cast<uint64_t>(tensors.size());
    header.tensor_directory_offset = directory_offset;
    header.tensor_data_offset = data_offset;

    for (i = 0; i < tensors.size(); ++i) {
        microgemm_tensor_entry entry;
        std::memset(&entry, 0, sizeof(entry));
        std::snprintf(entry.name, sizeof(entry.name), "%s", tensors[i].name.c_str());
        entry.dtype = tensors[i].dtype;
        entry.rank = tensors[i].rank;
        std::memcpy(entry.dims, tensors[i].dims.data(), sizeof(entry.dims));
        cursor = align_up_u64(cursor, alignment);
        entry.offset = cursor;
        entry.byte_length = tensors[i].byte_length;
        directory[i] = entry;
        cursor += entry.byte_length;
    }

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(&cfg), sizeof(cfg));
    out.write(reinterpret_cast<const char*>(directory.data()), static_cast<std::streamsize>(directory.size() * sizeof(microgemm_tensor_entry)));
    while (static_cast<uint64_t>(out.tellp()) < data_offset) {
        out.put('\0');
    }

    for (i = 0; i < tensors.size(); ++i) {
        while (static_cast<uint64_t>(out.tellp()) < directory[i].offset) {
            out.put('\0');
        }
        if (!tensors[i].spill_path.empty()) {
            std::error_code size_ec;
            uint64_t part_size = static_cast<uint64_t>(
                std::filesystem::file_size(std::filesystem::path(tensors[i].spill_path), size_ec)
            );
            if (size_ec || part_size != tensors[i].byte_length) {
                if (out_error != NULL) {
                    *out_error = std::string("temporary tensor part has wrong size: ") + tensors[i].spill_path;
                }
                return false;
            }
            std::ifstream part(tensors[i].spill_path, std::ios::binary);
            if (!part) {
                if (out_error != NULL) {
                    *out_error = std::string("failed to read temporary tensor part: ") + tensors[i].spill_path;
                }
                return false;
            }
            out << part.rdbuf();
            if (!out.good() || part.bad()) {
                if (out_error != NULL) {
                    *out_error = std::string("failed while copying temporary tensor part: ") + tensors[i].spill_path;
                }
                return false;
            }
        } else if (!tensors[i].bytes.empty()) {
            out.write(
                reinterpret_cast<const char*>(tensors[i].bytes.data()),
                static_cast<std::streamsize>(tensors[i].bytes.size())
            );
        }
    }

    if (!out.good()) {
        if (out_error != NULL) {
            *out_error = "failed while writing output file";
        }
        return false;
    }
    return true;
}

static const microgemm_safetensor_entry* require_tensor_shape(
    const microgemm_safetensor_file& file,
    const char* name,
    const std::vector<uint64_t>& expected_shape,
    std::string* out_error
) {
    const microgemm_safetensor_entry* entry = find_safetensor_entry(file, name);
    if (entry == NULL) {
        if (out_error != NULL) {
            *out_error = std::string("missing tensor: ") + name;
        }
        return NULL;
    }
    if (!expect_shape(*entry, expected_shape, out_error)) {
        return NULL;
    }
    return entry;
}

static bool load_required_tensor(
    const microgemm_safetensor_file& file,
    const char* name,
    const std::vector<uint64_t>& expected_shape,
    std::vector<float>* out_values,
    std::string* out_error
) {
    const microgemm_safetensor_entry* entry = require_tensor_shape(file, name, expected_shape, out_error);
    if (entry == NULL) {
        return false;
    }
    return decode_tensor_f32(file, *entry, out_values, out_error);
}

static bool maybe_load_required_tensor(
    const microgemm_safetensor_file& file,
    const char* primary_name,
    const char* fallback_name,
    const std::vector<uint64_t>& expected_shape,
    std::vector<float>* out_values,
    std::string* out_error
) {
    const microgemm_safetensor_entry* entry = find_safetensor_entry(file, primary_name);
    if (entry == NULL && fallback_name != NULL) {
        entry = find_safetensor_entry(file, fallback_name);
    }
    if (entry == NULL) {
        if (out_error != NULL) {
            *out_error = std::string("missing tensor: ") + primary_name;
        }
        return false;
    }
    if (!expect_shape(*entry, expected_shape, out_error)) {
        return false;
    }
    return decode_tensor_f32(file, *entry, out_values, out_error);
}

static std::string join_tensor_name(const std::string& prefix, const char* suffix) {
    if (prefix.empty()) {
        return std::string(suffix);
    }
    return prefix + "." + suffix;
}

static std::string layer_tensor_name(
    const std::string& prefix,
    uint32_t layer_idx,
    const char* suffix
) {
    std::ostringstream oss;
    if (!prefix.empty()) {
        oss << prefix << ".";
    }
    oss << "layers." << layer_idx << "." << suffix;
    return oss.str();
}

static bool string_ends_with(const std::string& text, const char* suffix) {
    size_t text_len = text.size();
    size_t suffix_len = std::strlen(suffix);
    return text_len >= suffix_len
        && text.compare(text_len - suffix_len, suffix_len, suffix) == 0;
}

static std::string detect_hf_model_prefix(const microgemm_safetensor_file& file) {
    const char* prefixes[] = {
        "model",
        "language_model.model",
        "model.language_model.model",
        "model.language_model",
        "text_model.model",
        "model.text_model"
    };
    size_t i;

    for (i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); ++i) {
        std::string name = join_tensor_name(prefixes[i], "embed_tokens.weight");
        if (find_safetensor_entry(file, name.c_str()) != NULL) {
            return prefixes[i];
        }
    }
    return "model";
}

static std::vector<std::string> lm_head_tensor_candidates(const std::string& model_prefix) {
    std::vector<std::string> names;
    names.push_back("lm_head.weight");
    names.push_back(join_tensor_name(model_prefix, "lm_head.weight"));
    if (string_ends_with(model_prefix, ".model")) {
        std::string parent = model_prefix.substr(0, model_prefix.size() - std::strlen(".model"));
        names.push_back(join_tensor_name(parent, "lm_head.weight"));
    }
    names.push_back("language_model.lm_head.weight");
    names.push_back("model.language_model.lm_head.weight");
    names.push_back(join_tensor_name(model_prefix, "embed_tokens.weight"));
    return names;
}

static bool load_required_tensor_any(
    const microgemm_safetensor_file& file,
    const std::vector<std::string>& names,
    const std::vector<uint64_t>& expected_shape,
    std::vector<float>* out_values,
    std::string* out_error
) {
    size_t i;

    for (i = 0; i < names.size(); ++i) {
        const microgemm_safetensor_entry* entry = find_safetensor_entry(file, names[i].c_str());
        if (entry == NULL) {
            continue;
        }
        if (!expect_shape(*entry, expected_shape, out_error)) {
            return false;
        }
        return decode_tensor_f32(file, *entry, out_values, out_error);
    }

    if (out_error != NULL) {
        *out_error = names.empty()
            ? "missing tensor"
            : std::string("missing tensor: ") + names[0];
    }
    return false;
}

static bool load_required_conv1d_weight(
    const microgemm_safetensor_file& file,
    const char* name,
    uint64_t channels,
    uint64_t kernel_size,
    std::vector<float>* out_values,
    std::string* out_error
) {
    const microgemm_safetensor_entry* entry = find_safetensor_entry(file, name);

    if (entry == NULL) {
        if (out_error != NULL) {
            *out_error = std::string("missing tensor: ") + name;
        }
        return false;
    }
    if (entry->shape.size() == 2u) {
        if (!expect_shape(*entry, std::vector<uint64_t>{channels, kernel_size}, out_error)) {
            return false;
        }
        return decode_tensor_f32(file, *entry, out_values, out_error);
    }
    if (entry->shape.size() == 3u) {
        if (!expect_shape(*entry, std::vector<uint64_t>{channels, 1u, kernel_size}, out_error)) {
            return false;
        }
        return decode_tensor_f32(file, *entry, out_values, out_error);
    }
    if (out_error != NULL) {
        *out_error = std::string("tensor rank mismatch for ") + name;
    }
    return false;
}

static uint32_t qwen35_layer_type_code(const microgemm_hf_config& cfg, uint32_t layer_idx) {
    if (!is_qwen35_model_type(cfg.model_type)) {
        return MICROGEMM_LAYER_FULL_ATTENTION;
    }
    if (layer_idx < cfg.layer_types.size() && cfg.layer_types[layer_idx] == "linear_attention") {
        return MICROGEMM_LAYER_LINEAR_ATTENTION;
    }
    return MICROGEMM_LAYER_FULL_ATTENTION;
}

static microgemm_config build_runtime_config(
    const microgemm_hf_config& cfg,
    uint32_t kv_block_size,
    uint32_t quant_mode
) {
    microgemm_config out;
    std::memset(&out, 0, sizeof(out));
    out.architecture = map_architecture(cfg);
    out.hidden_size = cfg.hidden_size;
    out.intermediate_size = cfg.intermediate_size;
    out.num_layers = cfg.num_hidden_layers;
    out.num_q_heads = cfg.num_attention_heads;
    out.num_kv_heads = infer_kv_heads(cfg);
    out.head_dim = infer_head_dim(cfg);
    out.vocab_size = cfg.vocab_size;
    out.max_position_embeddings = cfg.max_position_embeddings;
    out.kv_block_size = kv_block_size;
    out.quant_mode = quant_mode;
    out.flags = 0u;
    out.rms_norm_eps = static_cast<float>(cfg.rms_norm_eps);
    out.rope_theta = static_cast<float>(cfg.rope_theta);
    out.attention_logit_softcap = static_cast<float>(cfg.attention_logit_softcap);
    out.final_logit_softcap = static_cast<float>(cfg.final_logit_softcap);
    out.query_pre_attn_scalar = static_cast<float>(cfg.query_pre_attn_scalar);
    out.embedding_multiplier = static_cast<float>(cfg.embedding_multiplier);
    out.residual_multiplier = static_cast<float>(cfg.residual_multiplier);
    out.logits_scaling = static_cast<float>(cfg.logits_scaling);
    out.attn_width = out.num_q_heads * out.head_dim;
    out.rotary_dim = cfg.rotary_dim != 0u ? cfg.rotary_dim : out.head_dim;
    out.linear_key_head_dim = cfg.linear_key_head_dim != 0u ? cfg.linear_key_head_dim : out.head_dim;
    out.linear_value_head_dim = cfg.linear_value_head_dim != 0u ? cfg.linear_value_head_dim : out.head_dim;
    out.linear_num_key_heads = cfg.linear_num_key_heads != 0u ? cfg.linear_num_key_heads : out.num_q_heads;
    out.linear_num_value_heads = cfg.linear_num_value_heads != 0u ? cfg.linear_num_value_heads : out.num_q_heads;
    out.linear_conv_kernel_dim = cfg.linear_conv_kernel_dim;
    {
        uint32_t q_rows = out.attn_width;
        uint32_t full_qkv_rows;
        uint32_t linear_qkv_rows;

        if (cfg.attention_output_gate) {
            q_rows *= 2u;
        }
        full_qkv_rows = q_rows + 2u * out.num_kv_heads * out.head_dim;
        linear_qkv_rows = is_qwen35_model_type(cfg.model_type)
            ? 2u * out.linear_num_key_heads * out.linear_key_head_dim
                + out.linear_num_value_heads * out.linear_value_head_dim
            : 0u;
        out.qkv_rows = full_qkv_rows > linear_qkv_rows ? full_qkv_rows : linear_qkv_rows;
    }
    if (cfg.norm_offset) {
        out.flags |= MICROGEMM_FLAG_NORM_OFFSET;
    }
    if (cfg.attention_bias) {
        out.flags |= MICROGEMM_FLAG_QKV_BIAS;
    }
    if (is_gelu_hidden_act(cfg.hidden_act)) {
        out.flags |= MICROGEMM_FLAG_MLP_GELU;
    }
    if (cfg.attention_output_gate) {
        out.flags |= MICROGEMM_FLAG_ATTN_OUTPUT_GATE;
    }
    if (out.rotary_dim != 0u && out.rotary_dim != out.head_dim) {
        out.flags |= MICROGEMM_FLAG_PARTIAL_ROPE;
    }
    if (cfg.rope_interleaved) {
        out.flags |= MICROGEMM_FLAG_ROPE_INTERLEAVED;
    }
    return out;
}

static void add_quantized_weight_tensors(
    std::vector<microgemm_output_tensor>* out_tensors,
    const std::string& base_name,
    const std::vector<float>& values,
    uint64_t rows,
    uint64_t cols,
    uint32_t quant_mode,
    std::vector<int8_t>* q_i8,
    std::vector<uint8_t>* q_i4,
    std::vector<float>* q_scales,
    std::vector<int32_t>* q_row_sums
) {
    const bool groupwise = quant_mode_is_groupwise(quant_mode);
    const uint64_t groups = quant_group_count(cols);

    if (quant_mode_is_i4_storage(quant_mode)) {
        if (groupwise) {
            quantize_rows_i4_groupwise(values, rows, cols, q_i4, q_scales, q_row_sums);
        } else {
            quantize_rows_i4(values, rows, cols, q_i4, q_scales, q_row_sums);
        }
        add_output_tensor(
            out_tensors,
            base_name + ".weight_i4",
            MICROGEMM_DTYPE_I4,
            std::vector<uint64_t>{rows, cols},
            std::vector<uint8_t>(*q_i4)
        );
    } else {
        if (groupwise) {
            quantize_rows_i8_groupwise(values, rows, cols, q_i8, q_scales, q_row_sums);
        } else {
            quantize_rows_i8(values, rows, cols, q_i8, q_scales, q_row_sums);
        }
        add_output_tensor(
            out_tensors,
            base_name + ".weight_i8",
            MICROGEMM_DTYPE_I8,
            std::vector<uint64_t>{rows, cols},
            i8_to_bytes(*q_i8)
        );
    }
    add_output_tensor(
        out_tensors,
        base_name + ".scale",
        MICROGEMM_DTYPE_F32,
        groupwise ? std::vector<uint64_t>{rows, groups} : std::vector<uint64_t>{rows},
        floats_to_bytes(*q_scales)
    );
    add_output_tensor(
        out_tensors,
        base_name + ".row_sum",
        MICROGEMM_DTYPE_I32,
        groupwise ? std::vector<uint64_t>{rows, groups} : std::vector<uint64_t>{rows},
        i32_to_bytes(*q_row_sums)
    );
}

static bool build_microgemm_tensors_from_safetensors(
    const microgemm_hf_config& hf_cfg,
    const microgemm_safetensor_file& safetensors,
    uint32_t kv_block_size,
    uint32_t quant_mode,
    microgemm_config* out_cfg,
    std::vector<microgemm_output_tensor>* out_tensors,
    std::string* out_error
) {
    microgemm_config cfg;
    std::vector<std::string> issues;
    uint32_t layer_idx;
    uint64_t hidden = hf_cfg.hidden_size;
    uint64_t inter = hf_cfg.intermediate_size;
    uint64_t q_heads = hf_cfg.num_attention_heads;
    uint64_t kv_heads = infer_kv_heads(hf_cfg);
    uint64_t head_dim = infer_head_dim(hf_cfg);
    uint64_t attn_width = q_heads * head_dim;
    uint64_t q_rows = attn_width * (hf_cfg.attention_output_gate ? 2u : 1u);
    uint64_t kv_rows = kv_heads * head_dim;
    uint64_t qkv_rows = q_rows + 2u * kv_rows;
    uint64_t linear_key_head_dim = hf_cfg.linear_key_head_dim != 0u ? hf_cfg.linear_key_head_dim : head_dim;
    uint64_t linear_value_head_dim = hf_cfg.linear_value_head_dim != 0u ? hf_cfg.linear_value_head_dim : head_dim;
    uint64_t linear_key_heads = hf_cfg.linear_num_key_heads != 0u ? hf_cfg.linear_num_key_heads : q_heads;
    uint64_t linear_value_heads = hf_cfg.linear_num_value_heads != 0u ? hf_cfg.linear_num_value_heads : q_heads;
    uint64_t linear_key_dim = linear_key_heads * linear_key_head_dim;
    uint64_t linear_value_dim = linear_value_heads * linear_value_head_dim;
    uint64_t linear_conv_dim = 2u * linear_key_dim + linear_value_dim;
    uint64_t linear_baz_rows = linear_value_dim + 2u * linear_value_heads;
    std::vector<float> values_a;
    std::vector<float> values_b;
    std::vector<float> values_c;
    std::vector<float> values_cat;
    std::vector<int8_t> q_i8;
    std::vector<uint8_t> q_i4;
    std::vector<float> q_scales;
    std::vector<int32_t> q_row_sums;
    std::vector<float> rope_cos;
    std::vector<float> rope_sin;
    std::string hf_prefix;
    std::string hf_name;
    std::vector<std::string> hf_candidates;
    bool has_qk_norm = false;
    bool has_pre_ffn_norm = false;
    bool has_attn_output_norm = false;
    bool has_mlp_output_norm = false;
    char name[128];

    if (out_cfg == NULL || out_tensors == NULL) {
        return false;
    }
    if (!quant_mode_is_i8_storage(quant_mode) && !quant_mode_is_i4_storage(quant_mode)) {
        if (out_error != NULL) {
            *out_error = "unsupported quant mode; expected int8, int4, int8g128/int8g, or int4g128/int4g";
        }
        return false;
    }

    collect_support_issues(hf_cfg, &issues);
    if (!issues.empty()) {
        if (out_error != NULL) {
            std::ostringstream oss;
            size_t issue_idx;
            oss << "model config is not yet supported by the native exporter: ";
            for (issue_idx = 0; issue_idx < issues.size(); ++issue_idx) {
                if (issue_idx != 0u) {
                    oss << ",";
                }
                oss << issues[issue_idx];
            }
            *out_error = oss.str();
        }
        return false;
    }

    cfg = build_runtime_config(hf_cfg, kv_block_size, quant_mode);
    out_tensors->clear();
    hf_prefix = detect_hf_model_prefix(safetensors);
    has_qk_norm =
        hf_cfg.qk_norm
        || (find_safetensor_entry(
                safetensors,
                layer_tensor_name(hf_prefix, 0u, "self_attn.q_norm.weight").c_str()
            ) != NULL
            && find_safetensor_entry(
                safetensors,
                layer_tensor_name(hf_prefix, 0u, "self_attn.k_norm.weight").c_str()
            ) != NULL);
    has_pre_ffn_norm =
        find_safetensor_entry(
            safetensors,
            layer_tensor_name(hf_prefix, 0u, "pre_feedforward_layernorm.weight").c_str()
        ) != NULL;
    has_attn_output_norm =
        (has_pre_ffn_norm
            && find_safetensor_entry(
                safetensors,
                layer_tensor_name(hf_prefix, 0u, "post_attention_layernorm.weight").c_str()
            ) != NULL)
        || (is_glm4_like_model_type(hf_cfg.model_type)
            && find_safetensor_entry(
                safetensors,
                layer_tensor_name(hf_prefix, 0u, "post_self_attn_layernorm.weight").c_str()
            ) != NULL);
    has_mlp_output_norm =
        find_safetensor_entry(
            safetensors,
            layer_tensor_name(hf_prefix, 0u, "post_feedforward_layernorm.weight").c_str()
        ) != NULL
        || (is_glm4_like_model_type(hf_cfg.model_type)
            && find_safetensor_entry(
                safetensors,
                layer_tensor_name(hf_prefix, 0u, "post_mlp_layernorm.weight").c_str()
            ) != NULL);
    if (has_qk_norm) {
        cfg.flags |= MICROGEMM_FLAG_QK_NORM;
    }
    if (has_attn_output_norm) {
        cfg.flags |= MICROGEMM_FLAG_ATTN_OUTPUT_NORM;
    }
    if (has_mlp_output_norm) {
        cfg.flags |= MICROGEMM_FLAG_MLP_OUTPUT_NORM;
    }

    if (!load_required_tensor(
            safetensors,
            join_tensor_name(hf_prefix, "embed_tokens.weight").c_str(),
            std::vector<uint64_t>{static_cast<uint64_t>(hf_cfg.vocab_size), hidden},
            &values_a,
            out_error)) {
        return false;
    }
    if (is_gemma_like_model_type(hf_cfg.model_type)) {
        scale_float_values(&values_a, std::sqrt(static_cast<float>(hidden)));
    }
    add_quantized_weight_tensors(
        out_tensors,
        "embed_tokens",
        values_a,
        static_cast<uint64_t>(hf_cfg.vocab_size),
        hidden,
        quant_mode,
        &q_i8,
        &q_i4,
        &q_scales,
        &q_row_sums
    );

    hf_candidates = lm_head_tensor_candidates(hf_prefix);
    if (!load_required_tensor_any(
            safetensors,
            hf_candidates,
            std::vector<uint64_t>{static_cast<uint64_t>(hf_cfg.vocab_size), hidden},
            &values_a,
            out_error)) {
        return false;
    }
    add_quantized_weight_tensors(
        out_tensors,
        "lm_head",
        values_a,
        static_cast<uint64_t>(hf_cfg.vocab_size),
        hidden,
        quant_mode,
        &q_i8,
        &q_i4,
        &q_scales,
        &q_row_sums
    );

    if (!load_required_tensor(
            safetensors,
            join_tensor_name(hf_prefix, "norm.weight").c_str(),
            std::vector<uint64_t>{hidden},
            &values_a,
            out_error)) {
        return false;
    }
    add_output_tensor(
        out_tensors,
        "final_norm.weight",
        MICROGEMM_DTYPE_F32,
        std::vector<uint64_t>{hidden},
        floats_to_bytes(values_a)
    );

    build_rope_cache(hf_cfg, static_cast<uint32_t>(head_dim), &rope_cos, &rope_sin);
    add_output_tensor(
        out_tensors,
        "rope.cos",
        MICROGEMM_DTYPE_F32,
        std::vector<uint64_t>{static_cast<uint64_t>(hf_cfg.max_position_embeddings), head_dim / 2u},
        floats_to_bytes(rope_cos)
    );
    add_output_tensor(
        out_tensors,
        "rope.sin",
        MICROGEMM_DTYPE_F32,
        std::vector<uint64_t>{static_cast<uint64_t>(hf_cfg.max_position_embeddings), head_dim / 2u},
        floats_to_bytes(rope_sin)
    );

    for (layer_idx = 0; layer_idx < hf_cfg.num_hidden_layers; ++layer_idx) {
        hf_name = layer_tensor_name(hf_prefix, layer_idx, "input_layernorm.weight");
        if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{hidden}, &values_a, out_error)) {
            return false;
        }
        std::snprintf(name, sizeof(name), "layers.%u.input_norm.weight", layer_idx);
        add_output_tensor(out_tensors, name, MICROGEMM_DTYPE_F32, std::vector<uint64_t>{hidden}, floats_to_bytes(values_a));

        hf_candidates.clear();
        if (is_gemma_like_model_type(hf_cfg.model_type)) {
            hf_candidates.push_back(layer_tensor_name(hf_prefix, layer_idx, "pre_feedforward_layernorm.weight"));
        }
        hf_candidates.push_back(layer_tensor_name(hf_prefix, layer_idx, "post_attention_layernorm.weight"));
        if (!load_required_tensor_any(safetensors, hf_candidates, std::vector<uint64_t>{hidden}, &values_a, out_error)) {
            return false;
        }
        std::snprintf(name, sizeof(name), "layers.%u.post_norm.weight", layer_idx);
        add_output_tensor(out_tensors, name, MICROGEMM_DTYPE_F32, std::vector<uint64_t>{hidden}, floats_to_bytes(values_a));

        {
            uint32_t layer_type = qwen35_layer_type_code(hf_cfg, layer_idx);
            std::vector<int32_t> layer_type_value(1u, static_cast<int32_t>(layer_type));

            std::snprintf(name, sizeof(name), "layers.%u.type", layer_idx);
            add_output_tensor(
                out_tensors,
                name,
                MICROGEMM_DTYPE_I32,
                std::vector<uint64_t>{1u},
                i32_to_bytes(layer_type_value)
            );

            if (layer_type == MICROGEMM_LAYER_LINEAR_ATTENTION) {
                hf_name = layer_tensor_name(hf_prefix, layer_idx, "linear_attn.in_proj_qkv.weight");
                if (!load_required_tensor(
                        safetensors,
                        hf_name.c_str(),
                        std::vector<uint64_t>{linear_conv_dim, hidden},
                        &values_a,
                        out_error)) {
                    return false;
                }
                std::snprintf(name, sizeof(name), "layers.%u.linear_qkv", layer_idx);
                add_quantized_weight_tensors(
                    out_tensors,
                    name,
                    values_a,
                    linear_conv_dim,
                    hidden,
                    quant_mode,
                    &q_i8,
                    &q_i4,
                    &q_scales,
                    &q_row_sums
                );

                hf_name = layer_tensor_name(hf_prefix, layer_idx, "linear_attn.in_proj_z.weight");
                if (!load_required_tensor(
                        safetensors,
                        hf_name.c_str(),
                        std::vector<uint64_t>{linear_value_dim, hidden},
                        &values_a,
                        out_error)) {
                    return false;
                }
                hf_name = layer_tensor_name(hf_prefix, layer_idx, "linear_attn.in_proj_b.weight");
                if (!load_required_tensor(
                        safetensors,
                        hf_name.c_str(),
                        std::vector<uint64_t>{linear_value_heads, hidden},
                        &values_b,
                        out_error)) {
                    return false;
                }
                hf_name = layer_tensor_name(hf_prefix, layer_idx, "linear_attn.in_proj_a.weight");
                if (!load_required_tensor(
                        safetensors,
                        hf_name.c_str(),
                        std::vector<uint64_t>{linear_value_heads, hidden},
                        &values_c,
                        out_error)) {
                    return false;
                }
                values_cat = concat_rows(values_a, values_b, values_c);
                std::snprintf(name, sizeof(name), "layers.%u.linear_baz", layer_idx);
                add_quantized_weight_tensors(
                    out_tensors,
                    name,
                    values_cat,
                    linear_baz_rows,
                    hidden,
                    quant_mode,
                    &q_i8,
                    &q_i4,
                    &q_scales,
                    &q_row_sums
                );

                hf_name = layer_tensor_name(hf_prefix, layer_idx, "linear_attn.conv1d.weight");
                if (!load_required_conv1d_weight(
                        safetensors,
                        hf_name.c_str(),
                        linear_conv_dim,
                        hf_cfg.linear_conv_kernel_dim,
                        &values_a,
                        out_error)) {
                    return false;
                }
                std::snprintf(name, sizeof(name), "layers.%u.linear_conv.weight", layer_idx);
                add_output_tensor(
                    out_tensors,
                    name,
                    MICROGEMM_DTYPE_F32,
                    std::vector<uint64_t>{linear_conv_dim, hf_cfg.linear_conv_kernel_dim},
                    floats_to_bytes(values_a)
                );

                hf_name = layer_tensor_name(hf_prefix, layer_idx, "linear_attn.dt_bias");
                if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{linear_value_heads}, &values_a, out_error)) {
                    return false;
                }
                std::snprintf(name, sizeof(name), "layers.%u.linear_dt_bias", layer_idx);
                add_output_tensor(out_tensors, name, MICROGEMM_DTYPE_F32, std::vector<uint64_t>{linear_value_heads}, floats_to_bytes(values_a));

                hf_name = layer_tensor_name(hf_prefix, layer_idx, "linear_attn.A_log");
                if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{linear_value_heads}, &values_a, out_error)) {
                    return false;
                }
                std::snprintf(name, sizeof(name), "layers.%u.linear_A_log", layer_idx);
                add_output_tensor(out_tensors, name, MICROGEMM_DTYPE_F32, std::vector<uint64_t>{linear_value_heads}, floats_to_bytes(values_a));

                hf_name = layer_tensor_name(hf_prefix, layer_idx, "linear_attn.norm.weight");
                if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{linear_value_head_dim}, &values_a, out_error)) {
                    return false;
                }
                std::snprintf(name, sizeof(name), "layers.%u.linear_norm.weight", layer_idx);
                add_output_tensor(out_tensors, name, MICROGEMM_DTYPE_F32, std::vector<uint64_t>{linear_value_head_dim}, floats_to_bytes(values_a));

                hf_name = layer_tensor_name(hf_prefix, layer_idx, "linear_attn.out_proj.weight");
                if (!load_required_tensor(
                        safetensors,
                        hf_name.c_str(),
                        std::vector<uint64_t>{hidden, linear_value_dim},
                        &values_a,
                        out_error)) {
                    return false;
                }
                std::snprintf(name, sizeof(name), "layers.%u.linear_out", layer_idx);
                add_quantized_weight_tensors(
                    out_tensors,
                    name,
                    values_a,
                    hidden,
                    linear_value_dim,
                    quant_mode,
                    &q_i8,
                    &q_i4,
                    &q_scales,
                    &q_row_sums
                );
            } else {
                hf_name = layer_tensor_name(hf_prefix, layer_idx, "self_attn.qkv_proj.weight");
                if (is_phi_like_model_type(hf_cfg.model_type)
                        && find_safetensor_entry(safetensors, hf_name.c_str()) != NULL) {
                    if (!load_required_tensor(
                            safetensors,
                            hf_name.c_str(),
                            std::vector<uint64_t>{qkv_rows, hidden},
                            &values_cat,
                            out_error)) {
                        return false;
                    }
                } else {
                    hf_name = layer_tensor_name(hf_prefix, layer_idx, "self_attn.q_proj.weight");
                    if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{q_rows, hidden}, &values_a, out_error)) {
                        return false;
                    }
                    hf_name = layer_tensor_name(hf_prefix, layer_idx, "self_attn.k_proj.weight");
                    if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{kv_rows, hidden}, &values_b, out_error)) {
                        return false;
                    }
                    hf_name = layer_tensor_name(hf_prefix, layer_idx, "self_attn.v_proj.weight");
                    if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{kv_rows, hidden}, &values_c, out_error)) {
                        return false;
                    }
                    values_cat = concat_rows(values_a, values_b, values_c);
                }
                std::snprintf(name, sizeof(name), "layers.%u.qkv", layer_idx);
                add_quantized_weight_tensors(
                    out_tensors,
                    name,
                    values_cat,
                    qkv_rows,
                    hidden,
                    quant_mode,
                    &q_i8,
                    &q_i4,
                    &q_scales,
                    &q_row_sums
                );

                if ((cfg.flags & MICROGEMM_FLAG_QKV_BIAS) != 0u) {
                    hf_name = layer_tensor_name(hf_prefix, layer_idx, "self_attn.qkv_proj.bias");
                    if (find_safetensor_entry(safetensors, hf_name.c_str()) != NULL) {
                        if (!load_required_tensor(
                                safetensors,
                                hf_name.c_str(),
                                std::vector<uint64_t>{qkv_rows},
                                &values_cat,
                                out_error)) {
                            return false;
                        }
                    } else {
                        hf_name = layer_tensor_name(hf_prefix, layer_idx, "self_attn.q_proj.bias");
                        if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{q_rows}, &values_a, out_error)) {
                            return false;
                        }
                        hf_name = layer_tensor_name(hf_prefix, layer_idx, "self_attn.k_proj.bias");
                        if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{kv_rows}, &values_b, out_error)) {
                            return false;
                        }
                        hf_name = layer_tensor_name(hf_prefix, layer_idx, "self_attn.v_proj.bias");
                        if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{kv_rows}, &values_c, out_error)) {
                            return false;
                        }
                        values_cat = concat_rows(values_a, values_b, values_c);
                    }
                    std::snprintf(name, sizeof(name), "layers.%u.qkv.bias", layer_idx);
                    add_output_tensor(
                        out_tensors,
                        name,
                        MICROGEMM_DTYPE_F32,
                        std::vector<uint64_t>{qkv_rows},
                        floats_to_bytes(values_cat)
                    );
                }

                if ((cfg.flags & MICROGEMM_FLAG_QK_NORM) != 0u) {
                    hf_name = layer_tensor_name(hf_prefix, layer_idx, "self_attn.q_norm.weight");
                    if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{head_dim}, &values_a, out_error)) {
                        return false;
                    }
                    std::snprintf(name, sizeof(name), "layers.%u.q_norm.weight", layer_idx);
                    add_output_tensor(out_tensors, name, MICROGEMM_DTYPE_F32, std::vector<uint64_t>{head_dim}, floats_to_bytes(values_a));

                    hf_name = layer_tensor_name(hf_prefix, layer_idx, "self_attn.k_norm.weight");
                    if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{head_dim}, &values_a, out_error)) {
                        return false;
                    }
                    std::snprintf(name, sizeof(name), "layers.%u.k_norm.weight", layer_idx);
                    add_output_tensor(out_tensors, name, MICROGEMM_DTYPE_F32, std::vector<uint64_t>{head_dim}, floats_to_bytes(values_a));
                }

                hf_name = layer_tensor_name(hf_prefix, layer_idx, "self_attn.o_proj.weight");
                if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{hidden, attn_width}, &values_a, out_error)) {
                    return false;
                }
                std::snprintf(name, sizeof(name), "layers.%u.o", layer_idx);
                add_quantized_weight_tensors(
                    out_tensors,
                    name,
                    values_a,
                    hidden,
                    attn_width,
                    quant_mode,
                    &q_i8,
                    &q_i4,
                    &q_scales,
                    &q_row_sums
                );
                if ((cfg.flags & MICROGEMM_FLAG_ATTN_OUTPUT_NORM) != 0u) {
                    hf_name = layer_tensor_name(
                        hf_prefix,
                        layer_idx,
                        is_glm4_like_model_type(hf_cfg.model_type)
                            ? "post_self_attn_layernorm.weight"
                            : "post_attention_layernorm.weight"
                    );
                    if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{hidden}, &values_a, out_error)) {
                        return false;
                    }
                    std::snprintf(name, sizeof(name), "layers.%u.attn_output_norm.weight", layer_idx);
                    add_output_tensor(out_tensors, name, MICROGEMM_DTYPE_F32, std::vector<uint64_t>{hidden}, floats_to_bytes(values_a));
                }
            }
        }

        hf_name = layer_tensor_name(hf_prefix, layer_idx, "mlp.gate_up_proj.weight");
        if (find_safetensor_entry(safetensors, hf_name.c_str()) != NULL) {
            if (!load_required_tensor(
                    safetensors,
                    hf_name.c_str(),
                    std::vector<uint64_t>{2u * inter, hidden},
                    &values_cat,
                    out_error)) {
                return false;
            }
        } else {
            hf_name = layer_tensor_name(hf_prefix, layer_idx, "mlp.gate_proj.weight");
            if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{inter, hidden}, &values_a, out_error)) {
                return false;
            }
            hf_name = layer_tensor_name(hf_prefix, layer_idx, "mlp.up_proj.weight");
            if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{inter, hidden}, &values_b, out_error)) {
                return false;
            }
            values_cat = concat_rows(values_a, values_b);
        }
        std::snprintf(name, sizeof(name), "layers.%u.gate_up", layer_idx);
        add_quantized_weight_tensors(
            out_tensors,
            name,
            values_cat,
            2u * inter,
            hidden,
            quant_mode,
            &q_i8,
            &q_i4,
            &q_scales,
            &q_row_sums
        );

        hf_name = layer_tensor_name(hf_prefix, layer_idx, "mlp.down_proj.weight");
        if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{hidden, inter}, &values_a, out_error)) {
            return false;
        }
        std::snprintf(name, sizeof(name), "layers.%u.down", layer_idx);
        add_quantized_weight_tensors(
            out_tensors,
            name,
            values_a,
            hidden,
            inter,
            quant_mode,
            &q_i8,
            &q_i4,
            &q_scales,
            &q_row_sums
        );
        if ((cfg.flags & MICROGEMM_FLAG_MLP_OUTPUT_NORM) != 0u) {
            hf_name = layer_tensor_name(
                hf_prefix,
                layer_idx,
                is_glm4_like_model_type(hf_cfg.model_type)
                    ? "post_mlp_layernorm.weight"
                    : "post_feedforward_layernorm.weight"
            );
            if (!load_required_tensor(safetensors, hf_name.c_str(), std::vector<uint64_t>{hidden}, &values_a, out_error)) {
                return false;
            }
            std::snprintf(name, sizeof(name), "layers.%u.mlp_output_norm.weight", layer_idx);
            add_output_tensor(out_tensors, name, MICROGEMM_DTYPE_F32, std::vector<uint64_t>{hidden}, floats_to_bytes(values_a));
        }
    }

    *out_cfg = cfg;
    return true;
}

static bool parse_u32_arg(const char* text, uint32_t* out_value) {
    char* end_ptr;
    unsigned long value;
    if (text == NULL || out_value == NULL) {
        return false;
    }
    value = std::strtoul(text, &end_ptr, 10);
    if (end_ptr == text || *end_ptr != '\0' || value > static_cast<unsigned long>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    *out_value = static_cast<uint32_t>(value);
    return true;
}

static bool parse_quant_arg(const char* text, uint32_t* out_value) {
    if (text == NULL || out_value == NULL) {
        return false;
    }
    if (std::strcmp(text, "int8") == 0) {
        *out_value = MICROGEMM_QUANT_INT8;
        return true;
    }
    if (std::strcmp(text, "int4") == 0) {
        *out_value = MICROGEMM_QUANT_INT4;
        return true;
    }
    if (std::strcmp(text, "int8g128") == 0 || std::strcmp(text, "int8g") == 0) {
        *out_value = MICROGEMM_QUANT_INT8G128;
        return true;
    }
    if (std::strcmp(text, "int4g128") == 0 || std::strcmp(text, "int4g") == 0) {
        *out_value = MICROGEMM_QUANT_INT4G128;
        return true;
    }
    return false;
}

static int from_files_command(
    const char* config_path,
    const char* weights_path,
    const char* output_path,
    uint32_t kv_block_size,
    uint32_t alignment,
    uint32_t quant_mode
) {
    microgemm_hf_config hf_cfg;
    microgemm_safetensor_file safetensors;
    microgemm_config out_cfg;
    std::vector<microgemm_output_tensor> out_tensors;
    std::string error;

    if (kv_block_size == 0u) {
        std::fprintf(stderr, "from-files failed: kv_block_size must be greater than zero\n");
        return 1;
    }

    if (!load_hf_config(config_path, &hf_cfg, &error)) {
        std::fprintf(stderr, "from-files failed: %s\n", error.c_str());
        return 1;
    }
    if (!load_safetensors_file(weights_path, &safetensors, &error)) {
        std::fprintf(stderr, "from-files failed: %s\n", error.c_str());
        return 1;
    }
    begin_tensor_spill(output_path);
    if (!build_microgemm_tensors_from_safetensors(hf_cfg, safetensors, kv_block_size, quant_mode, &out_cfg, &out_tensors, &error)) {
        end_tensor_spill();
        std::fprintf(stderr, "from-files failed: %s\n", error.c_str());
        return 2;
    }
    if (!write_microgemm_file(output_path, out_cfg, out_tensors, alignment, &error)) {
        end_tensor_spill();
        std::fprintf(stderr, "from-files failed: %s\n", error.c_str());
        return 1;
    }
    end_tensor_spill();

    std::printf("wrote %s with %llu tensors\n",
        output_path,
        static_cast<unsigned long long>(out_tensors.size()));
    return 0;
}

static bool path_exists(const std::string& path) {
    std::error_code ec;
    return std::filesystem::exists(std::filesystem::path(path), ec);
}

static std::vector<std::string> find_safetensors_shards(const std::string& dir) {
    std::vector<std::string> shards;
    std::error_code ec;
    std::filesystem::path root(dir);

    if (!std::filesystem::exists(root, ec) || !std::filesystem::is_directory(root, ec)) {
        return shards;
    }

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(root, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        std::filesystem::path path = entry.path();
        if (path.extension() == ".safetensors" && path.filename() != "model.safetensors") {
            shards.push_back(path.string());
        }
    }
    std::sort(shards.begin(), shards.end());
    return shards;
}

static int from_sharded_dir_command(
    const char* config_path,
    const std::vector<std::string>& shard_paths,
    const char* output_path,
    uint32_t kv_block_size,
    uint32_t alignment,
    uint32_t quant_mode
) {
    microgemm_hf_config hf_cfg;
    microgemm_safetensor_file safetensors;
    microgemm_config out_cfg;
    std::vector<microgemm_output_tensor> out_tensors;
    std::string error;

    if (kv_block_size == 0u) {
        std::fprintf(stderr, "from-dir failed: kv_block_size must be greater than zero\n");
        return 1;
    }
    if (shard_paths.empty()) {
        std::fprintf(stderr, "from-dir failed: no safetensors shards found\n");
        return 1;
    }

    if (!load_hf_config(config_path, &hf_cfg, &error)) {
        std::fprintf(stderr, "from-dir failed: %s\n", error.c_str());
        return 1;
    }
    if (!load_safetensors_shards(shard_paths, &safetensors, &error)) {
        std::fprintf(stderr, "from-dir failed: %s\n", error.c_str());
        return 1;
    }
    begin_tensor_spill(output_path);
    if (!build_microgemm_tensors_from_safetensors(hf_cfg, safetensors, kv_block_size, quant_mode, &out_cfg, &out_tensors, &error)) {
        end_tensor_spill();
        std::fprintf(stderr, "from-dir failed: %s\n", error.c_str());
        return 2;
    }
    if (!write_microgemm_file(output_path, out_cfg, out_tensors, alignment, &error)) {
        end_tensor_spill();
        std::fprintf(stderr, "from-dir failed: %s\n", error.c_str());
        return 1;
    }
    end_tensor_spill();

    std::printf("wrote %s with %llu tensors from %llu safetensors shards\n",
        output_path,
        static_cast<unsigned long long>(out_tensors.size()),
        static_cast<unsigned long long>(shard_paths.size()));
    return 0;
}

static int from_dir_command(
    const char* model_dir,
    const char* output_path,
    uint32_t kv_block_size,
    uint32_t alignment,
    uint32_t quant_mode
) {
    std::string dir = model_dir == NULL ? std::string() : std::string(model_dir);
    std::string config_path = join_path(dir, "config.json");
    std::string weights_path = join_path(dir, "model.safetensors");
    std::vector<std::string> shard_paths;

    if (path_exists(weights_path)) {
        return from_files_command(config_path.c_str(), weights_path.c_str(), output_path, kv_block_size, alignment, quant_mode);
    }

    shard_paths = find_safetensors_shards(dir);
    return from_sharded_dir_command(config_path.c_str(), shard_paths, output_path, kv_block_size, alignment, quant_mode);
}

static int inspect_config_command(const char* path) {
    microgemm_hf_config cfg;
    std::string error;
    std::vector<std::string> issues;
    uint32_t arch = MICROGEMM_ARCH_UNKNOWN;
    uint32_t head_dim = 0;

    if (!load_hf_config(path, &cfg, &error)) {
        std::fprintf(stderr, "inspect-config failed: %s\n", error.c_str());
        return 1;
    }

    collect_support_issues(cfg, &issues);
    arch = map_architecture(cfg);
    head_dim = infer_head_dim(cfg);

    std::printf("path: %s\n", path);
    std::printf("model_type: %s\n", cfg.model_type.c_str());
    std::printf("architecture: %s\n", microgemm_format_architecture_name(arch));
    std::printf("hidden_size: %u\n", cfg.hidden_size);
    std::printf("intermediate_size: %u\n", cfg.intermediate_size);
    std::printf("layers: %u\n", cfg.num_hidden_layers);
    std::printf("q_heads: %u\n", cfg.num_attention_heads);
    std::printf("kv_heads: %u\n", cfg.num_key_value_heads);
    std::printf("head_dim: %u\n", head_dim);
    std::printf("vocab_size: %u\n", cfg.vocab_size);
    std::printf("max_position_embeddings: %u\n", cfg.max_position_embeddings);
    std::printf("hidden_act: %s\n", cfg.hidden_act.c_str());
    std::printf("rms_norm_eps: %.8g\n", cfg.rms_norm_eps);
    std::printf("rope_theta: %.8g\n", cfg.rope_theta);
    std::printf("attention_logit_softcap: %.8g\n", cfg.attention_logit_softcap);
    std::printf("final_logit_softcap: %.8g\n", cfg.final_logit_softcap);
    std::printf("query_pre_attn_scalar: %.8g\n", cfg.query_pre_attn_scalar);
    std::printf("embedding_multiplier: %.8g\n", cfg.embedding_multiplier);
    std::printf("residual_multiplier: %.8g\n", cfg.residual_multiplier);
    std::printf("logits_scaling: %.8g\n", cfg.logits_scaling);
    if (is_qwen35_model_type(cfg.model_type)) {
        uint32_t linear_layers = 0u;
        uint32_t full_layers = 0u;
        for (uint32_t i = 0u; i < cfg.layer_types.size(); ++i) {
            if (cfg.layer_types[i] == "linear_attention") {
                ++linear_layers;
            } else if (cfg.layer_types[i] == "full_attention") {
                ++full_layers;
            }
        }
        std::printf("attention_output_gate: %u\n", cfg.attention_output_gate ? 1u : 0u);
        std::printf("partial_rotary_factor: %.8g\n", cfg.partial_rotary_factor);
        std::printf("rotary_dim: %u\n", cfg.rotary_dim);
        std::printf("layer_types: full=%u linear=%u\n", full_layers, linear_layers);
        std::printf("linear_key_head_dim: %u\n", cfg.linear_key_head_dim);
        std::printf("linear_value_head_dim: %u\n", cfg.linear_value_head_dim);
        std::printf("linear_num_key_heads: %u\n", cfg.linear_num_key_heads);
        std::printf("linear_num_value_heads: %u\n", cfg.linear_num_value_heads);
        std::printf("linear_conv_kernel_dim: %u\n", cfg.linear_conv_kernel_dim);
    }
    if (cfg.has_rope_scaling) {
        std::printf("rope_scaling: type=%s factor=%.8g original_max_position_embeddings=%u low_freq_factor=%.8g high_freq_factor=%.8g\n",
            cfg.rope_type.empty() ? "default" : cfg.rope_type.c_str(),
            cfg.rope_factor,
            cfg.rope_original_max_position_embeddings,
            cfg.rope_low_freq_factor,
            cfg.rope_high_freq_factor);
    } else {
        std::printf("rope_scaling: none\n");
    }
    std::printf("estimated_int8_payload_bytes: %llu\n",
        static_cast<unsigned long long>(estimate_int8_payload_bytes(cfg)));

    if (issues.empty()) {
        std::printf("native_export_support: yes\n");
        return 0;
    }

    std::printf("native_export_support: no\n");
    std::printf("unsupported_features:");
    for (size_t i = 0; i < issues.size(); ++i) {
        std::printf("%s%s", i == 0 ? " " : ",", issues[i].c_str());
    }
    std::printf("\n");
    return 2;
}

static void print_usage(void) {
    std::puts("MicroGemm Native Converter");
    std::puts("Usage:");
    std::puts("  microgemm-convert inspect-config <path-to-config.json>");
    std::puts("  microgemm-convert from-files <config.json> <model.safetensors> <output.mgm> [--kv-block-size N] [--alignment N] [--quant int8|int4|int8g128|int8g|int4g128|int4g]");
    std::puts("  microgemm-convert from-dir <model-dir> <output.mgm> [--kv-block-size N] [--alignment N] [--quant int8|int4|int8g128|int8g|int4g128|int4g]");
}

int main(int argc, char** argv) {
    uint32_t kv_block_size = 16u;
    uint32_t alignment = 64u;
    uint32_t quant_mode = MICROGEMM_QUANT_INT8;
    int argi;

    if (argc < 2) {
        print_usage();
        return 1;
    }
    if (std::strcmp(argv[1], "inspect-config") == 0) {
        if (argc < 3) {
            std::fprintf(stderr, "inspect-config requires a config.json path\n");
            return 1;
        }
        return inspect_config_command(argv[2]);
    }
    if (std::strcmp(argv[1], "from-files") == 0) {
        if (argc < 5) {
            std::fprintf(stderr, "from-files requires <config.json> <model.safetensors> <output.mgm>\n");
            return 1;
        }
        for (argi = 5; argi < argc; ++argi) {
            if (std::strcmp(argv[argi], "--kv-block-size") == 0) {
                if (argi + 1 >= argc || !parse_u32_arg(argv[++argi], &kv_block_size)) {
                    std::fprintf(stderr, "invalid --kv-block-size value\n");
                    return 1;
                }
            } else if (std::strcmp(argv[argi], "--alignment") == 0) {
                if (argi + 1 >= argc || !parse_u32_arg(argv[++argi], &alignment)) {
                    std::fprintf(stderr, "invalid --alignment value\n");
                    return 1;
                }
            } else if (std::strcmp(argv[argi], "--quant") == 0) {
                if (argi + 1 >= argc || !parse_quant_arg(argv[++argi], &quant_mode)) {
                    std::fprintf(stderr, "invalid --quant value; expected int8, int4, int8g128/int8g, or int4g128/int4g\n");
                    return 1;
                }
            } else {
                std::fprintf(stderr, "unknown option: %s\n", argv[argi]);
                return 1;
            }
        }
        return from_files_command(argv[2], argv[3], argv[4], kv_block_size, alignment, quant_mode);
    }
    if (std::strcmp(argv[1], "from-dir") == 0) {
        if (argc < 4) {
            std::fprintf(stderr, "from-dir requires <model-dir> <output.mgm>\n");
            return 1;
        }
        for (argi = 4; argi < argc; ++argi) {
            if (std::strcmp(argv[argi], "--kv-block-size") == 0) {
                if (argi + 1 >= argc || !parse_u32_arg(argv[++argi], &kv_block_size)) {
                    std::fprintf(stderr, "invalid --kv-block-size value\n");
                    return 1;
                }
            } else if (std::strcmp(argv[argi], "--alignment") == 0) {
                if (argi + 1 >= argc || !parse_u32_arg(argv[++argi], &alignment)) {
                    std::fprintf(stderr, "invalid --alignment value\n");
                    return 1;
                }
            } else if (std::strcmp(argv[argi], "--quant") == 0) {
                if (argi + 1 >= argc || !parse_quant_arg(argv[++argi], &quant_mode)) {
                    std::fprintf(stderr, "invalid --quant value; expected int8, int4, int8g128/int8g, or int4g128/int4g\n");
                    return 1;
                }
            } else {
                std::fprintf(stderr, "unknown option: %s\n", argv[argi]);
                return 1;
            }
        }
        return from_dir_command(argv[2], argv[3], kv_block_size, alignment, quant_mode);
    }

    print_usage();
    return 1;
}

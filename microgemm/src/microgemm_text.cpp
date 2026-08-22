#include "microgemm/microgemm.h"
#include "microgemm/microgemm_decode.h"

#include <array>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <climits>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct microgemm_added_token {
    int id = -1;
    std::string content;
    bool special = false;
};

struct microgemm_tokenizer {
    bool byte_level = false;
    bool add_prefix_space = false;
    bool has_sentencepiece_marker = false;
    int bos_token_id = -1;
    int eos_token_id = -1;
    int unk_token_id = -1;

    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> merge_ranks;
    std::unordered_set<int> special_ids;
    std::unordered_map<std::string, int> special_tokens_by_content;
    std::array<std::string, 256> byte_encoder;
    std::unordered_map<std::string, unsigned int> byte_decoder;
};

static bool token_looks_special_text(const std::string& token) {
    if (token.size() >= 2u && token.front() == '<' && token.back() == '>') {
        return true;
    }
    if (token.size() >= 2u && token.front() == '[' && token.back() == ']') {
        return true;
    }
    return false;
}

static bool token_has_sentencepiece_marker(const std::string& token) {
    static const char* kSentencepieceMarker = "\xE2\x96\x81"; /* U+2581 */
    return token.find(kSentencepieceMarker) != std::string::npos;
}

static std::string sentencepiece_restore_spaces(const std::string& text) {
    static const std::string kSentencepieceMarker = "\xE2\x96\x81"; /* U+2581 */
    std::string out;
    size_t pos = 0u;

    while (pos < text.size()) {
        size_t marker = text.find(kSentencepieceMarker, pos);
        if (marker == std::string::npos) {
            out.append(text, pos, std::string::npos);
            break;
        }
        out.append(text, pos, marker - pos);
        out.push_back(' ');
        pos = marker + kSentencepieceMarker.size();
    }

    return out;
}

struct microgemm_text_args {
    std::string prompt;
    std::vector<int> prompt_ids;
    uint32_t max_new_tokens = 32u;
    uint32_t max_seq_len = 0u;
    float temperature = 0.0f;
    float top_p = 1.0f;
    uint32_t top_k = 0u;
    uint64_t seed = 1337u;
    bool use_eos = false;
    int eos_token = -1;
    bool ignore_eos = false;
    bool use_bos = false;
    bool disable_auto_bos = false;
    int bos_token = -1;
    bool skip_special_tokens = true;
};

struct microgemm_generation_result_cpp {
    microgemm_config config = {};
    size_t loaded_model_bytes = 0u;
    size_t workspace_bytes = 0u;
    size_t kv_cache_bytes = 0u;
    size_t context_tokens_used = 0u;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    std::vector<int> generated_tokens;
};

struct microgemm_sampling_config {
    float temperature = 0.0f;
    float top_p = 1.0f;
    uint32_t top_k = 0u;
    uint64_t rng_state = 1337u;
};

struct microgemm_batch_text_args {
    std::vector<std::string> prompts;
    std::vector<std::vector<int> > prompt_id_batches;
    uint32_t max_new_tokens = 32u;
    uint32_t max_seq_len = 0u;
    float temperature = 0.0f;
    float top_p = 1.0f;
    uint32_t top_k = 0u;
    uint64_t seed = 1337u;
    bool use_eos = false;
    int eos_token = -1;
    bool ignore_eos = false;
    bool use_bos = false;
    bool disable_auto_bos = false;
    int bos_token = -1;
    bool skip_special_tokens = true;
};

struct microgemm_batch_sequence_state {
    std::string prompt;
    std::vector<int> prompt_ids;
    std::vector<int> generated_tokens;
    std::vector<float> logits;
    std::vector<int> block_table_storage;
    std::vector<std::vector<float> > layer_kv_storage;
    std::vector<float*> layer_kv_ptrs;
    microgemm_decode_workspace* workspace = NULL;
    microgemm_kv_layout kv = {};
    microgemm_sampling_config sampling = {};
    int current_token = -1;
    bool finished = false;
    size_t workspace_bytes = 0u;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
};

struct microgemm_batch_generation_result_cpp {
    microgemm_config config = {};
    size_t loaded_model_bytes = 0u;
    size_t workspace_bytes = 0u;
    size_t kv_cache_bytes = 0u;
    size_t generated_token_count = 0u;
    size_t finished_request_count = 0u;
    size_t scheduler_iterations = 0u;
    size_t batched_decode_calls = 0u;
    size_t batched_decode_tokens = 0u;
    size_t batched_lm_head_calls = 0u;
    size_t batched_lm_head_tokens = 0u;
    int scheduler_outer_threads = 1;
    int scheduler_inner_threads = 1;
    int scheduler_lm_head_threads = 1;
    double model_open_ms = 0.0;
    double model_load_ms = 0.0;
    double model_cleanup_ms = 0.0;
    double setup_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    microgemm_decode_batch_profile decode_profile = {};
    std::vector<microgemm_batch_sequence_state> requests;
};

static int sample_from_logits(
    const float* logits,
    int vocab_size,
    microgemm_sampling_config* sampling
);

static int parse_positive_int_prefix(const char* raw) {
    char* end_ptr;
    long value;
    if (raw == NULL || *raw == '\0') {
        return 0;
    }
    value = std::strtol(raw, &end_ptr, 10);
    if (end_ptr == raw || value <= 0 || value > INT_MAX) {
        return 0;
    }
    return static_cast<int>(value);
}

static int env_positive_int(const char* name, int fallback) {
    int value = parse_positive_int_prefix(std::getenv(name));
    return value > 0 ? value : fallback;
}

static int clamp_thread_count(int value, int minimum, int maximum) {
    if (maximum < minimum) {
        maximum = minimum;
    }
    if (value < minimum) {
        return minimum;
    }
    if (value > maximum) {
        return maximum;
    }
    return value;
}

static int default_thread_count() {
    int omp_threads = parse_positive_int_prefix(std::getenv("OMP_NUM_THREADS"));
    if (omp_threads > 0) {
        return omp_threads;
    }
    {
        unsigned int hw = std::thread::hardware_concurrency();
        if (hw > 0u && hw <= static_cast<unsigned int>(INT_MAX)) {
            return static_cast<int>(hw);
        }
    }
    return 1;
}

static void configure_openmp_thread_count(int thread_count) {
#ifdef _OPENMP
    if (thread_count > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(thread_count);
    }
#else
    (void)thread_count;
#endif
}

class microgemm_worker_pool {
public:
    explicit microgemm_worker_pool(int worker_threads)
        : worker_count_(std::max(1, worker_threads)) {
        if (worker_count_ <= 1) {
            return;
        }
        try {
            workers_.reserve(static_cast<size_t>(worker_count_));
            for (int i = 0; i < worker_count_; ++i) {
                workers_.emplace_back([this]() { worker_loop(); });
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> guard(mutex_);
                stopping_ = true;
                ++generation_;
            }
            work_cv_.notify_all();
            for (std::thread& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            throw;
        }
    }

    ~microgemm_worker_pool() {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            stopping_ = true;
            ++generation_;
        }
        work_cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template <typename Fn>
    void run(int count, int inner_openmp_threads, Fn fn) {
        if (count <= 0) {
            return;
        }

        inner_openmp_threads = std::max(1, inner_openmp_threads);
        if (worker_count_ <= 1) {
            configure_openmp_thread_count(inner_openmp_threads);
            for (int idx = 0; idx < count; ++idx) {
                fn(idx);
            }
            return;
        }

        std::function<void(int)> task = fn;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            task_ = &task;
            task_count_ = count;
            next_index_ = 0;
            completed_workers_ = 0;
            inner_openmp_threads_ = inner_openmp_threads;
            ++generation_;
        }

        work_cv_.notify_all();

        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [this]() {
                return completed_workers_ >= static_cast<int>(workers_.size());
            });
            task_ = NULL;
        }
    }

private:
    void worker_loop() {
        int observed_generation = 0;

        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            work_cv_.wait(lock, [this, observed_generation]() {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_) {
                return;
            }

            observed_generation = generation_;
            int inner_threads = inner_openmp_threads_;
            configure_openmp_thread_count(inner_threads);

            for (;;) {
                int idx = next_index_++;
                const std::function<void(int)>* task = task_;
                if (idx >= task_count_) {
                    break;
                }
                lock.unlock();
                (*task)(idx);
                lock.lock();
            }

            ++completed_workers_;
            if (completed_workers_ >= static_cast<int>(workers_.size())) {
                done_cv_.notify_one();
            }
        }
    }

    int worker_count_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    bool stopping_ = false;
    int generation_ = 0;
    int completed_workers_ = 0;
    int task_count_ = 0;
    int next_index_ = 0;
    int inner_openmp_threads_ = 1;
    const std::function<void(int)>* task_ = NULL;
};

static bool read_text_file(const char* path, std::string* out_text) {
    std::ifstream file(path, std::ios::binary);
    std::ostringstream buffer;

    if (path == NULL || out_text == NULL || !file) {
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

    *out_pos = skip_ws(text, colon_pos + 1u);
    return true;
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

    cursor = pos + 1u;
    while (cursor < text.size()) {
        char ch = text[cursor];
        if (ch == '\\') {
            if (cursor + 1u >= text.size()) {
                return false;
            }
            value.push_back(text[cursor + 1u]);
            cursor += 2u;
            continue;
        }
        if (ch == '"') {
            *out_value = value;
            *out_next = cursor + 1u;
            return true;
        }
        value.push_back(ch);
        ++cursor;
    }

    return false;
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
    if (text.compare(pos, 4u, "true") == 0) {
        *out_value = true;
        return true;
    }
    if (text.compare(pos, 5u, "false") == 0) {
        *out_value = false;
        return true;
    }
    return false;
}

static bool get_json_string(const std::string& text, const char* key, std::string* out_value) {
    size_t pos;
    return find_key_value_start(text, key, &pos) && parse_json_string_token(text, pos, out_value, &pos);
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
        } else if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0) {
                *out_object = text.substr(pos, cursor - pos + 1u);
                return true;
            }
        }
    }

    return false;
}

static bool get_json_array(const std::string& text, const char* key, std::string* out_array) {
    size_t pos;
    size_t cursor;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;

    if (out_array == NULL || !find_key_value_start(text, key, &pos)) {
        return false;
    }
    if (pos >= text.size() || text[pos] != '[') {
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
        } else if (ch == '[') {
            ++depth;
        } else if (ch == ']') {
            --depth;
            if (depth == 0) {
                *out_array = text.substr(pos, cursor - pos + 1u);
                return true;
            }
        }
    }

    return false;
}

static bool parse_string_u32_object(
    const std::string& text,
    std::unordered_map<std::string, int>* out_map
) {
    size_t cursor;

    if (out_map == NULL) {
        return false;
    }
    out_map->clear();

    cursor = skip_ws(text, 0);
    if (cursor >= text.size() || text[cursor] != '{') {
        return false;
    }

    cursor = skip_ws(text, cursor + 1u);
    while (cursor < text.size() && text[cursor] != '}') {
        std::string key;
        size_t after_key;
        size_t value_pos;
        uint64_t value = 0u;

        if (!parse_json_string_token(text, cursor, &key, &after_key)) {
            return false;
        }
        value_pos = skip_ws(text, after_key);
        if (value_pos >= text.size() || text[value_pos] != ':') {
            return false;
        }
        value_pos = skip_ws(text, value_pos + 1u);
        if (!parse_json_u64_at(text, value_pos, &value) || value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            return false;
        }
        (*out_map)[key] = static_cast<int>(value);

        while (value_pos < text.size() && text[value_pos] != ',' && text[value_pos] != '}') {
            ++value_pos;
        }
        cursor = skip_ws(text, value_pos);
        if (cursor < text.size() && text[cursor] == ',') {
            cursor = skip_ws(text, cursor + 1u);
        }
    }

    return cursor < text.size() && text[cursor] == '}';
}

static bool parse_string_array(const std::string& text, std::vector<std::string>* out_values) {
    size_t cursor;

    if (out_values == NULL) {
        return false;
    }
    out_values->clear();

    cursor = skip_ws(text, 0);
    if (cursor >= text.size() || text[cursor] != '[') {
        return false;
    }

    cursor = skip_ws(text, cursor + 1u);
    while (cursor < text.size() && text[cursor] != ']') {
        std::string value;
        size_t after_value;

        if (text[cursor] == '[') {
            std::string left;
            std::string right;
            cursor = skip_ws(text, cursor + 1u);
            if (!parse_json_string_token(text, cursor, &left, &after_value)) {
                return false;
            }
            cursor = skip_ws(text, after_value);
            if (cursor >= text.size() || text[cursor] != ',') {
                return false;
            }
            cursor = skip_ws(text, cursor + 1u);
            if (!parse_json_string_token(text, cursor, &right, &after_value)) {
                return false;
            }
            cursor = skip_ws(text, after_value);
            if (cursor >= text.size() || text[cursor] != ']') {
                return false;
            }
            value = left + " " + right;
            after_value = cursor + 1u;
        } else if (!parse_json_string_token(text, cursor, &value, &after_value)) {
            return false;
        }

        out_values->push_back(value);
        cursor = skip_ws(text, after_value);
        if (cursor < text.size() && text[cursor] == ',') {
            cursor = skip_ws(text, cursor + 1u);
        }
    }

    return cursor < text.size() && text[cursor] == ']';
}

static bool parse_added_token_object(const std::string& text, microgemm_added_token* out_token) {
    size_t cursor;

    if (out_token == NULL) {
        return false;
    }

    cursor = skip_ws(text, 0);
    if (cursor >= text.size() || text[cursor] != '{') {
        return false;
    }

    cursor = skip_ws(text, cursor + 1u);
    while (cursor < text.size() && text[cursor] != '}') {
        std::string key;
        size_t after_key;
        size_t value_pos;
        size_t after_value;

        if (!parse_json_string_token(text, cursor, &key, &after_key)) {
            return false;
        }
        value_pos = skip_ws(text, after_key);
        if (value_pos >= text.size() || text[value_pos] != ':') {
            return false;
        }
        value_pos = skip_ws(text, value_pos + 1u);

        if (key == "id") {
            uint64_t value = 0u;
            if (!parse_json_u64_at(text, value_pos, &value) || value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                return false;
            }
            out_token->id = static_cast<int>(value);
        } else if (key == "content") {
            if (!parse_json_string_token(text, value_pos, &out_token->content, &after_value)) {
                return false;
            }
            cursor = skip_ws(text, after_value);
            if (cursor < text.size() && text[cursor] == ',') {
                cursor = skip_ws(text, cursor + 1u);
            }
            continue;
        } else if (key == "special") {
            if (!parse_json_bool_at(text, value_pos, &out_token->special)) {
                return false;
            }
        }

        while (value_pos < text.size() && text[value_pos] != ',' && text[value_pos] != '}') {
            ++value_pos;
        }
        cursor = skip_ws(text, value_pos);
        if (cursor < text.size() && text[cursor] == ',') {
            cursor = skip_ws(text, cursor + 1u);
        }
    }

    return cursor < text.size() && text[cursor] == '}';
}

static bool parse_added_token_array(
    const std::string& text,
    std::vector<microgemm_added_token>* out_tokens
) {
    size_t cursor;

    if (out_tokens == NULL) {
        return false;
    }
    out_tokens->clear();

    cursor = skip_ws(text, 0);
    if (cursor >= text.size() || text[cursor] != '[') {
        return false;
    }

    cursor = skip_ws(text, cursor + 1u);
    while (cursor < text.size() && text[cursor] != ']') {
        size_t start = cursor;
        size_t depth = 0u;
        bool in_string = false;
        bool escaped = false;
        microgemm_added_token token;

        for (; cursor < text.size(); ++cursor) {
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
            } else if (ch == '{') {
                ++depth;
            } else if (ch == '}') {
                if (depth == 0u) {
                    return false;
                }
                --depth;
                if (depth == 0u) {
                    ++cursor;
                    break;
                }
            }
        }

        if (cursor > text.size()) {
            return false;
        }
        if (!parse_added_token_object(text.substr(start, cursor - start), &token)) {
            return false;
        }
        out_tokens->push_back(token);
        cursor = skip_ws(text, cursor);
        if (cursor < text.size() && text[cursor] == ',') {
            cursor = skip_ws(text, cursor + 1u);
        }
    }

    return cursor < text.size() && text[cursor] == ']';
}

static std::string codepoint_to_utf8(unsigned int cp) {
    std::string out;
    if (cp <= 0x7fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7ffu) {
        out.push_back(static_cast<char>(0xc0u | (cp >> 6u)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3fu)));
    } else if (cp <= 0xffffu) {
        out.push_back(static_cast<char>(0xe0u | (cp >> 12u)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3fu)));
    } else {
        out.push_back(static_cast<char>(0xf0u | (cp >> 18u)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3fu)));
    }
    return out;
}

static void build_byte_level_maps(microgemm_tokenizer* tokenizer) {
    std::vector<unsigned int> bs;
    std::vector<unsigned int> cs;
    unsigned int b;
    unsigned int next_cp = 256u;

    if (tokenizer == NULL) {
        return;
    }

    for (b = 33u; b <= 126u; ++b) {
        bs.push_back(b);
    }
    for (b = 161u; b <= 172u; ++b) {
        bs.push_back(b);
    }
    for (b = 174u; b <= 255u; ++b) {
        bs.push_back(b);
    }

    cs = bs;
    for (b = 0u; b <= 255u; ++b) {
        size_t idx;
        bool present = false;
        for (idx = 0; idx < bs.size(); ++idx) {
            if (bs[idx] == b) {
                present = true;
                break;
            }
        }
        if (!present) {
            bs.push_back(b);
            cs.push_back(next_cp++);
        }
    }

    for (b = 0u; b < bs.size(); ++b) {
        std::string encoded = codepoint_to_utf8(cs[b]);
        tokenizer->byte_encoder[bs[b]] = encoded;
        tokenizer->byte_decoder[encoded] = bs[b];
    }
}

static size_t utf8_char_length(unsigned char lead) {
    if ((lead & 0x80u) == 0u) {
        return 1u;
    }
    if ((lead & 0xe0u) == 0xc0u) {
        return 2u;
    }
    if ((lead & 0xf0u) == 0xe0u) {
        return 3u;
    }
    if ((lead & 0xf8u) == 0xf0u) {
        return 4u;
    }
    return 1u;
}

static std::vector<std::string> split_utf8_chars(const std::string& text) {
    std::vector<std::string> out;
    size_t pos = 0u;
    while (pos < text.size()) {
        size_t len = utf8_char_length(static_cast<unsigned char>(text[pos]));
        if (pos + len > text.size()) {
            len = 1u;
        }
        out.push_back(text.substr(pos, len));
        pos += len;
    }
    return out;
}

static bool tokenizer_add_vocab_entry(microgemm_tokenizer* tokenizer, const std::string& token, int id) {
    if (tokenizer == NULL || id < 0) {
        return false;
    }
    tokenizer->token_to_id[token] = id;
    if (token_has_sentencepiece_marker(token)) {
        tokenizer->has_sentencepiece_marker = true;
    }
    if (static_cast<size_t>(id) >= tokenizer->id_to_token.size()) {
        tokenizer->id_to_token.resize(static_cast<size_t>(id) + 1u);
    }
    tokenizer->id_to_token[static_cast<size_t>(id)] = token;
    return true;
}

static void tokenizer_infer_special_ids(microgemm_tokenizer* tokenizer) {
    static const char* bos_names[] = {"<s>", "<|begin_of_text|>", "<bos>"};
    static const char* eos_names[] = {"</s>", "<|end_of_text|>", "<eos>"};
    size_t idx;
    size_t token_idx;

    if (tokenizer == NULL) {
        return;
    }

    for (idx = 0u; idx < sizeof(bos_names) / sizeof(bos_names[0]); ++idx) {
        std::unordered_map<std::string, int>::const_iterator it = tokenizer->special_tokens_by_content.find(bos_names[idx]);
        if (it != tokenizer->special_tokens_by_content.end()) {
            tokenizer->bos_token_id = it->second;
            break;
        }
        it = tokenizer->token_to_id.find(bos_names[idx]);
        if (it != tokenizer->token_to_id.end()) {
            tokenizer->bos_token_id = it->second;
            tokenizer->special_ids.insert(it->second);
            tokenizer->special_tokens_by_content[bos_names[idx]] = it->second;
            break;
        }
    }
    for (idx = 0u; idx < sizeof(eos_names) / sizeof(eos_names[0]); ++idx) {
        std::unordered_map<std::string, int>::const_iterator it = tokenizer->special_tokens_by_content.find(eos_names[idx]);
        if (it != tokenizer->special_tokens_by_content.end()) {
            tokenizer->eos_token_id = it->second;
            break;
        }
        it = tokenizer->token_to_id.find(eos_names[idx]);
        if (it != tokenizer->token_to_id.end()) {
            tokenizer->eos_token_id = it->second;
            tokenizer->special_ids.insert(it->second);
            tokenizer->special_tokens_by_content[eos_names[idx]] = it->second;
            break;
        }
    }

    for (token_idx = 0u; token_idx < tokenizer->id_to_token.size(); ++token_idx) {
        const std::string& token = tokenizer->id_to_token[token_idx];
        if (token_looks_special_text(token)) {
            tokenizer->special_ids.insert(static_cast<int>(token_idx));
            tokenizer->special_tokens_by_content[token] = static_cast<int>(token_idx);
        }
    }
}

static bool microgemm_tokenizer_load(
    const char* tokenizer_json_path,
    microgemm_tokenizer* out_tokenizer,
    std::string* out_error
) {
    std::string text;
    std::string model_object;
    std::string vocab_object;
    std::string merges_array;
    std::string added_tokens_array;
    std::string model_type;
    std::string unk_token;
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> merges;
    std::vector<microgemm_added_token> added_tokens;
    std::unordered_map<std::string, int>::const_iterator vocab_it;
    size_t idx;

    if (tokenizer_json_path == NULL || out_tokenizer == NULL) {
        return false;
    }

    *out_tokenizer = microgemm_tokenizer();
    build_byte_level_maps(out_tokenizer);

    if (!read_text_file(tokenizer_json_path, &text)) {
        if (out_error != NULL) {
            *out_error = "failed to read tokenizer.json";
        }
        return false;
    }

    out_tokenizer->byte_level = text.find("\"ByteLevel\"") != std::string::npos;
    (void)get_json_bool(text, "add_prefix_space", &out_tokenizer->add_prefix_space);

    if (!get_json_object(text, "model", &model_object)) {
        if (out_error != NULL) {
            *out_error = "tokenizer.json is missing model object";
        }
        return false;
    }
    if (!get_json_string(model_object, "type", &model_type) || model_type != "BPE") {
        if (out_error != NULL) {
            *out_error = "only tokenizer.json model.type=BPE is supported today";
        }
        return false;
    }
    if (!get_json_object(model_object, "vocab", &vocab_object) || !parse_string_u32_object(vocab_object, &vocab)) {
        if (out_error != NULL) {
            *out_error = "failed to parse tokenizer vocab";
        }
        return false;
    }
    if (!get_json_array(model_object, "merges", &merges_array) || !parse_string_array(merges_array, &merges)) {
        if (out_error != NULL) {
            *out_error = "failed to parse tokenizer merges";
        }
        return false;
    }

    for (vocab_it = vocab.begin(); vocab_it != vocab.end(); ++vocab_it) {
        if (!tokenizer_add_vocab_entry(out_tokenizer, vocab_it->first, vocab_it->second)) {
            if (out_error != NULL) {
                *out_error = "failed to store tokenizer vocab";
            }
            return false;
        }
    }

    for (idx = 0u; idx < merges.size(); ++idx) {
        size_t split_pos = merges[idx].find(' ');
        if (split_pos == std::string::npos) {
            continue;
        }
        out_tokenizer->merge_ranks[merges[idx].substr(0u, split_pos) + "\t" + merges[idx].substr(split_pos + 1u)] =
            static_cast<int>(idx);
    }

    if (get_json_array(text, "added_tokens", &added_tokens_array) && parse_added_token_array(added_tokens_array, &added_tokens)) {
        for (idx = 0u; idx < added_tokens.size(); ++idx) {
            if (added_tokens[idx].id >= 0 && !added_tokens[idx].content.empty()) {
                (void)tokenizer_add_vocab_entry(out_tokenizer, added_tokens[idx].content, added_tokens[idx].id);
                if (added_tokens[idx].special) {
                    out_tokenizer->special_ids.insert(added_tokens[idx].id);
                    out_tokenizer->special_tokens_by_content[added_tokens[idx].content] = added_tokens[idx].id;
                }
            }
        }
    }

    if (get_json_string(model_object, "unk_token", &unk_token)) {
        std::unordered_map<std::string, int>::const_iterator it = out_tokenizer->token_to_id.find(unk_token);
        if (it != out_tokenizer->token_to_id.end()) {
            out_tokenizer->unk_token_id = it->second;
        }
    }
    tokenizer_infer_special_ids(out_tokenizer);
    return true;
}

static bool bpe_encode_piece(
    const microgemm_tokenizer& tokenizer,
    const std::string& piece,
    std::vector<int>* out_ids,
    std::string* out_error
) {
    std::vector<std::string> symbols = split_utf8_chars(piece);

    if (out_ids == NULL) {
        return false;
    }
    if (symbols.empty()) {
        return true;
    }

    while (symbols.size() > 1u) {
        int best_rank = INT_MAX;
        size_t best_pos = 0u;
        bool found = false;
        size_t i;

        for (i = 0u; i + 1u < symbols.size(); ++i) {
            std::unordered_map<std::string, int>::const_iterator it =
                tokenizer.merge_ranks.find(symbols[i] + "\t" + symbols[i + 1u]);
            if (it != tokenizer.merge_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
                found = true;
            }
        }

        if (!found) {
            break;
        }
        symbols[best_pos] += symbols[best_pos + 1u];
        symbols.erase(symbols.begin() + best_pos + 1u);
    }

    for (size_t i = 0u; i < symbols.size(); ++i) {
        std::unordered_map<std::string, int>::const_iterator it = tokenizer.token_to_id.find(symbols[i]);
        if (it == tokenizer.token_to_id.end()) {
            if (tokenizer.unk_token_id >= 0) {
                out_ids->push_back(tokenizer.unk_token_id);
                continue;
            }
            if (out_error != NULL) {
                *out_error = "tokenizer encode failed on subword piece";
            }
            return false;
        }
        out_ids->push_back(it->second);
    }

    return true;
}

static bool microgemm_tokenizer_encode(
    const microgemm_tokenizer& tokenizer,
    const std::string& prompt,
    std::vector<int>* out_ids,
    std::string* out_error
) {
    std::string working = prompt;
    std::string transformed;
    size_t idx;

    if (out_ids == NULL) {
        return false;
    }
    out_ids->clear();

    if (tokenizer.add_prefix_space && !working.empty() && working[0] != ' ') {
        working.insert(working.begin(), ' ');
    }

    if (tokenizer.byte_level) {
        for (idx = 0u; idx < working.size(); ++idx) {
            transformed += tokenizer.byte_encoder[static_cast<unsigned char>(working[idx])];
        }
    } else {
        transformed = working;
    }

    return bpe_encode_piece(tokenizer, transformed, out_ids, out_error);
}

static std::string microgemm_tokenizer_decode(
    const microgemm_tokenizer& tokenizer,
    const std::vector<int>& ids,
    bool skip_special_tokens
) {
    std::string merged;
    std::string out;
    size_t idx;

    for (idx = 0u; idx < ids.size(); ++idx) {
        int id = ids[idx];
        const std::string* token;
        if (id < 0 || static_cast<size_t>(id) >= tokenizer.id_to_token.size()) {
            continue;
        }
        token = &tokenizer.id_to_token[static_cast<size_t>(id)];
        if (skip_special_tokens
                && (tokenizer.special_ids.find(id) != tokenizer.special_ids.end()
                    || token_looks_special_text(*token))) {
            continue;
        }
        merged += *token;
    }

    if (!tokenizer.byte_level) {
        if (tokenizer.has_sentencepiece_marker) {
            return sentencepiece_restore_spaces(merged);
        }
        return merged;
    }

    idx = 0u;
    while (idx < merged.size()) {
        size_t len = utf8_char_length(static_cast<unsigned char>(merged[idx]));
        std::string ch;
        std::unordered_map<std::string, unsigned int>::const_iterator it;
        if (idx + len > merged.size()) {
            len = 1u;
        }
        ch = merged.substr(idx, len);
        it = tokenizer.byte_decoder.find(ch);
        if (it != tokenizer.byte_decoder.end()) {
            out.push_back(static_cast<char>(it->second));
        } else {
            out += ch;
        }
        idx += len;
    }

    return out;
}

static void free_kv_layers(float** layer_kv, uint32_t num_layers) {
    uint32_t layer_idx;
    if (layer_kv == NULL) {
        return;
    }
    for (layer_idx = 0u; layer_idx < num_layers; ++layer_idx) {
        std::free(layer_kv[layer_idx]);
    }
    std::free(layer_kv);
}

static bool run_greedy_generation_cpp(
    const char* model_path,
    const std::vector<int>& prompt_tokens,
    uint32_t max_new_tokens,
    uint32_t max_seq_len,
    bool use_eos,
    int eos_token,
    const microgemm_sampling_config* sampling_in,
    microgemm_generation_result_cpp* out_result
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
    size_t block_count;
    size_t step_idx;
    int next_token = -1;
    int current_token = -1;
    microgemm_status status;
    microgemm_sampling_config sampling_local;
    microgemm_sampling_config* sampling = NULL;
    std::vector<float> logits;
    bool greedy_decode = false;
    std::chrono::steady_clock::time_point total_start;
    std::chrono::steady_clock::time_point prefill_start;
    std::chrono::steady_clock::time_point decode_start;
    std::chrono::steady_clock::time_point decode_end;

    if (model_path == NULL || out_result == NULL || prompt_tokens.empty() || max_new_tokens == 0u) {
        return false;
    }

    *out_result = microgemm_generation_result_cpp();
    std::memset(&kv, 0, sizeof(kv));

    status = microgemm_model_open(model_path, &model);
    if (status != MICROGEMM_STATUS_OK) {
        std::fprintf(stderr, "generation open failed: %s\n", microgemm_status_string(status));
        goto cleanup;
    }

    status = microgemm_loaded_model_i8_create(model, &loaded);
    if (status != MICROGEMM_STATUS_OK) {
        std::fprintf(stderr, "generation load failed: %s\n", microgemm_status_string(status));
        goto cleanup;
    }

    cfg = microgemm_loaded_model_i8_config(loaded);
    weights = microgemm_loaded_model_i8_weights(loaded);
    if (cfg == NULL || weights == NULL) {
        std::fprintf(stderr, "generation failed: loaded model is incomplete\n");
        goto cleanup;
    }
    sampling_local = sampling_in != NULL ? *sampling_in : microgemm_sampling_config();
    sampling = &sampling_local;
    greedy_decode = sampling->temperature <= 0.0f || sampling->top_k == 1u;
    if (!greedy_decode) {
        logits.assign(static_cast<size_t>(cfg->vocab_size), 0.0f);
    }

    required_seq_len = prompt_tokens.size() + static_cast<size_t>(max_new_tokens) - 1u;
    effective_max_seq_len = max_seq_len == 0u ? static_cast<uint32_t>(required_seq_len) : max_seq_len;
    if (static_cast<size_t>(effective_max_seq_len) < required_seq_len) {
        std::fprintf(stderr, "--max-seq-len is too small for prompt + generation context\n");
        goto cleanup;
    }
    if (effective_max_seq_len > cfg->max_position_embeddings) {
        std::fprintf(stderr, "--max-seq-len exceeds model max_position_embeddings\n");
        goto cleanup;
    }

    status = microgemm_decode_workspace_create(cfg, effective_max_seq_len, &workspace);
    if (status != MICROGEMM_STATUS_OK) {
        std::fprintf(stderr, "generation workspace failed: %s\n", microgemm_status_string(status));
        goto cleanup;
    }

    block_count = (static_cast<size_t>(effective_max_seq_len) + cfg->kv_block_size - 1u) / cfg->kv_block_size;
    kv.stride_pos = static_cast<int>(cfg->head_dim);
    kv.stride_head = static_cast<int>(cfg->kv_block_size * cfg->head_dim);
    kv.stride_kv = static_cast<int>(cfg->num_kv_heads * static_cast<uint32_t>(kv.stride_head));
    kv.stride_block = 2 * kv.stride_kv;
    kv.seq_len = 0;

    block_table = static_cast<int*>(std::malloc(block_count * sizeof(int)));
    layer_kv = static_cast<float**>(std::calloc(cfg->num_layers, sizeof(float*)));
    if (block_table == NULL || layer_kv == NULL) {
        std::fprintf(stderr, "generation failed: out of memory\n");
        goto cleanup;
    }

    for (step_idx = 0u; step_idx < block_count; ++step_idx) {
        block_table[step_idx] = static_cast<int>(step_idx);
    }
    for (step_idx = 0u; step_idx < cfg->num_layers; ++step_idx) {
        layer_kv[step_idx] = static_cast<float*>(std::calloc(block_count * static_cast<size_t>(kv.stride_block), sizeof(float)));
        if (layer_kv[step_idx] == NULL) {
            std::fprintf(stderr, "generation failed: out of memory allocating KV cache\n");
            goto cleanup;
        }
    }

    kv.layer_kv = layer_kv;
    kv.block_table = block_table;
    total_start = std::chrono::steady_clock::now();
    prefill_start = total_start;

    if (greedy_decode) {
        status = microgemm_decode_prefill_i8_next_token(
            cfg,
            weights,
            workspace,
            prompt_tokens.data(),
            prompt_tokens.size(),
            &kv,
            &next_token
        );
    } else {
        status = microgemm_decode_prefill_i8(
            cfg,
            weights,
            workspace,
            prompt_tokens.data(),
            prompt_tokens.size(),
            &kv,
            logits.data(),
            static_cast<int>(cfg->vocab_size)
        );
    }
    if (status != MICROGEMM_STATUS_OK) {
        std::fprintf(stderr, "generation failed during prompt prefill: %s\n", microgemm_status_string(status));
        goto cleanup;
    }
    out_result->prefill_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prefill_start
    ).count();

    if (!greedy_decode) {
        next_token = sample_from_logits(logits.data(), static_cast<int>(cfg->vocab_size), sampling);
    }
    out_result->generated_tokens.push_back(next_token);
    current_token = next_token;
    decode_start = std::chrono::steady_clock::now();
    decode_end = decode_start;

    for (step_idx = 1u; step_idx < static_cast<size_t>(max_new_tokens); ++step_idx) {
        if (use_eos && current_token == eos_token) {
            break;
        }

        if (greedy_decode) {
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
        } else {
            status = microgemm_decode_step_i8(
                cfg,
                weights,
                workspace,
                current_token,
                kv.seq_len,
                &kv,
                logits.data(),
                static_cast<int>(cfg->vocab_size),
                NULL
            );
        }
        if (status != MICROGEMM_STATUS_OK) {
            std::fprintf(stderr, "generation failed during decode: %s\n", microgemm_status_string(status));
            goto cleanup;
        }

        kv.seq_len += 1;
        if (!greedy_decode) {
            next_token = sample_from_logits(logits.data(), static_cast<int>(cfg->vocab_size), sampling);
        }
        out_result->generated_tokens.push_back(next_token);
        current_token = next_token;
    }
    decode_end = std::chrono::steady_clock::now();

    out_result->config = *cfg;
    out_result->loaded_model_bytes = microgemm_loaded_model_i8_bytes(loaded);
    out_result->workspace_bytes = microgemm_decode_workspace_bytes(workspace);
    out_result->kv_cache_bytes =
        cfg->num_layers * block_count * static_cast<size_t>(kv.stride_block) * sizeof(float);
    out_result->context_tokens_used = static_cast<size_t>(kv.seq_len);
    out_result->decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    out_result->total_ms = std::chrono::duration<double, std::milli>(decode_end - total_start).count();

    std::free(block_table);
    free_kv_layers(layer_kv, cfg->num_layers);
    microgemm_decode_workspace_destroy(workspace);
    microgemm_loaded_model_i8_destroy(loaded);
    microgemm_model_close(model);
    return true;

cleanup:
    std::free(block_table);
    free_kv_layers(layer_kv, loaded != NULL && microgemm_loaded_model_i8_config(loaded) != NULL
        ? microgemm_loaded_model_i8_config(loaded)->num_layers
        : 0u);
    microgemm_decode_workspace_destroy(workspace);
    microgemm_loaded_model_i8_destroy(loaded);
    microgemm_model_close(model);
    *out_result = microgemm_generation_result_cpp();
    return false;
}

static bool initialize_batch_sequence_kv(
    const microgemm_config* cfg,
    size_t block_count,
    microgemm_batch_sequence_state* seq
) {
    size_t layer_idx;

    if (cfg == NULL || seq == NULL || block_count == 0u) {
        return false;
    }

    try {
        size_t layer_kv_elems =
            block_count * 2u * static_cast<size_t>(cfg->num_kv_heads)
            * static_cast<size_t>(cfg->kv_block_size)
            * static_cast<size_t>(cfg->head_dim);
        seq->block_table_storage.resize(block_count);
        seq->layer_kv_storage.resize(cfg->num_layers);
        seq->layer_kv_ptrs.resize(cfg->num_layers);
        for (layer_idx = 0u; layer_idx < block_count; ++layer_idx) {
            seq->block_table_storage[layer_idx] = static_cast<int>(layer_idx);
        }
        for (layer_idx = 0u; layer_idx < cfg->num_layers; ++layer_idx) {
            seq->layer_kv_storage[layer_idx].assign(layer_kv_elems, 0.0f);
            seq->layer_kv_ptrs[layer_idx] = seq->layer_kv_storage[layer_idx].data();
        }
    } catch (...) {
        return false;
    }

    seq->kv.layer_kv = seq->layer_kv_ptrs.data();
    seq->kv.block_table = seq->block_table_storage.data();
    seq->kv.seq_len = 0;
    seq->kv.stride_pos = static_cast<int>(cfg->head_dim);
    seq->kv.stride_head = static_cast<int>(cfg->kv_block_size * cfg->head_dim);
    seq->kv.stride_kv = static_cast<int>(cfg->num_kv_heads * static_cast<uint32_t>(seq->kv.stride_head));
    seq->kv.stride_block = 2 * seq->kv.stride_kv;
    return true;
}

static void destroy_batch_sequence_workspaces(
    std::vector<microgemm_batch_sequence_state>* requests
) {
    size_t request_idx;

    if (requests == NULL) {
        return;
    }
    for (request_idx = 0u; request_idx < requests->size(); ++request_idx) {
        microgemm_decode_workspace_destroy((*requests)[request_idx].workspace);
        (*requests)[request_idx].workspace = NULL;
        (*requests)[request_idx].workspace_bytes = 0u;
    }
}

static bool run_continuous_batch_generation_cpp(
    const char* model_path,
    const std::vector<std::vector<int> >& prompt_batches,
    uint32_t max_new_tokens,
    uint32_t max_seq_len,
    bool use_eos,
    int eos_token,
    const microgemm_sampling_config* sampling_in,
    microgemm_batch_generation_result_cpp* out_result
) {
    microgemm_model* model = NULL;
    microgemm_loaded_model_i8* loaded = NULL;
    microgemm_worker_pool* request_pool = NULL;
    const microgemm_config* cfg;
    const microgemm_model_weights_i8* weights;
    microgemm_status status;
    microgemm_status first_error_status = MICROGEMM_STATUS_OK;
    microgemm_sampling_config sampling_template;
    size_t request_count;
    int request_count_i;
    size_t max_prompt_tokens = 0u;
    size_t required_seq_len;
    uint32_t effective_max_seq_len;
    size_t block_count;
    size_t request_idx;
    size_t active_requests = 0u;
    int first_error_request = -1;
    int scheduler_total_threads = 1;
    int scheduler_outer_threads = 1;
    int scheduler_inner_threads = 1;
    int scheduler_lm_head_threads = 1;
    std::mutex first_error_mutex;
    std::chrono::steady_clock::time_point setup_start;
    std::chrono::steady_clock::time_point setup_end;
    std::chrono::steady_clock::time_point model_open_start;
    std::chrono::steady_clock::time_point model_open_end;
    std::chrono::steady_clock::time_point model_load_start;
    std::chrono::steady_clock::time_point model_load_end;
    std::chrono::steady_clock::time_point cleanup_start;
    std::chrono::steady_clock::time_point cleanup_end;
    std::chrono::steady_clock::time_point total_start;
    std::chrono::steady_clock::time_point prefill_start;
    std::chrono::steady_clock::time_point decode_start;
    std::chrono::steady_clock::time_point decode_end;
    std::vector<int> ready_request_indices;
    std::vector<microgemm_decode_workspace*> ready_workspaces;
    std::vector<int> ready_token_ids;
    std::vector<int> ready_positions;
    std::vector<const microgemm_kv_layout*> ready_kvs;
    std::vector<int> ready_next_token_ids;
    std::vector<float*> ready_logits;
    std::vector<float> ready_logits_storage;
    bool greedy_decode = false;

    if (model_path == NULL || out_result == NULL || prompt_batches.empty() || max_new_tokens == 0u) {
        return false;
    }
    for (request_idx = 0u; request_idx < prompt_batches.size(); ++request_idx) {
        if (prompt_batches[request_idx].empty()) {
            return false;
        }
        if (prompt_batches[request_idx].size() > max_prompt_tokens) {
            max_prompt_tokens = prompt_batches[request_idx].size();
        }
    }

    *out_result = microgemm_batch_generation_result_cpp();
    request_count = prompt_batches.size();
    if (request_count > static_cast<size_t>(INT_MAX)) {
        std::fprintf(stderr, "batch generation failed: too many requests\n");
        return false;
    }
    request_count_i = static_cast<int>(request_count);
    scheduler_total_threads = env_positive_int(
        "MICROGEMM_BATCH_TOTAL_THREADS",
        default_thread_count()
    );
    scheduler_outer_threads = env_positive_int(
        "MICROGEMM_BATCH_OUTER_THREADS",
        std::min(request_count_i, scheduler_total_threads)
    );
    scheduler_outer_threads = clamp_thread_count(scheduler_outer_threads, 1, request_count_i);
    scheduler_inner_threads = env_positive_int(
        "MICROGEMM_BATCH_INNER_THREADS",
        std::max(1, scheduler_total_threads / scheduler_outer_threads)
    );
    scheduler_inner_threads = std::max(1, scheduler_inner_threads);
    scheduler_lm_head_threads = env_positive_int(
        "MICROGEMM_BATCH_LM_HEAD_THREADS",
        scheduler_total_threads
    );
    scheduler_lm_head_threads = std::max(1, scheduler_lm_head_threads);
    try {
        request_pool = new microgemm_worker_pool(scheduler_outer_threads);
    } catch (...) {
        std::fprintf(stderr, "batch generation failed: out of memory creating worker pool\n");
        goto cleanup;
    }

    model_open_start = std::chrono::steady_clock::now();
    status = microgemm_model_open(model_path, &model);
    model_open_end = std::chrono::steady_clock::now();
    out_result->model_open_ms = std::chrono::duration<double, std::milli>(
        model_open_end - model_open_start
    ).count();
    if (status != MICROGEMM_STATUS_OK) {
        std::fprintf(stderr, "batch generation open failed: %s\n", microgemm_status_string(status));
        goto cleanup;
    }

    model_load_start = std::chrono::steady_clock::now();
    status = microgemm_loaded_model_i8_create(model, &loaded);
    model_load_end = std::chrono::steady_clock::now();
    out_result->model_load_ms = std::chrono::duration<double, std::milli>(
        model_load_end - model_load_start
    ).count();
    if (status != MICROGEMM_STATUS_OK) {
        std::fprintf(stderr, "batch generation load failed: %s\n", microgemm_status_string(status));
        goto cleanup;
    }

    cfg = microgemm_loaded_model_i8_config(loaded);
    weights = microgemm_loaded_model_i8_weights(loaded);
    if (cfg == NULL || weights == NULL) {
        std::fprintf(stderr, "batch generation failed: loaded model is incomplete\n");
        goto cleanup;
    }

    required_seq_len = max_prompt_tokens + static_cast<size_t>(max_new_tokens) - 1u;
    effective_max_seq_len = max_seq_len == 0u ? static_cast<uint32_t>(required_seq_len) : max_seq_len;
    if (static_cast<size_t>(effective_max_seq_len) < required_seq_len) {
        std::fprintf(stderr, "--max-seq-len is too small for prompt + generation context\n");
        goto cleanup;
    }
    if (effective_max_seq_len > cfg->max_position_embeddings) {
        std::fprintf(stderr, "--max-seq-len exceeds model max_position_embeddings\n");
        goto cleanup;
    }

    block_count = (static_cast<size_t>(effective_max_seq_len) + cfg->kv_block_size - 1u) / cfg->kv_block_size;
    sampling_template = sampling_in != NULL ? *sampling_in : microgemm_sampling_config();
    greedy_decode = sampling_template.temperature <= 0.0f || sampling_template.top_k == 1u;

    try {
        out_result->requests.resize(request_count);
    } catch (...) {
        std::fprintf(stderr, "batch generation failed: out of memory allocating request states\n");
        goto cleanup;
    }

    setup_start = std::chrono::steady_clock::now();

    for (request_idx = 0u; request_idx < request_count; ++request_idx) {
        microgemm_batch_sequence_state* seq = &out_result->requests[request_idx];

        seq->prompt_ids = prompt_batches[request_idx];
        seq->sampling = sampling_template;
        seq->sampling.rng_state += static_cast<uint64_t>(request_idx) * 0x9E3779B9u;
        seq->generated_tokens.reserve(max_new_tokens);
        if (!greedy_decode) {
            seq->logits.assign(static_cast<size_t>(cfg->vocab_size), 0.0f);
        }
        if (!initialize_batch_sequence_kv(cfg, block_count, seq)) {
            std::fprintf(stderr, "batch generation failed: out of memory allocating KV cache\n");
            goto cleanup;
        }
        status = microgemm_decode_workspace_create(cfg, effective_max_seq_len, &seq->workspace);
        if (status != MICROGEMM_STATUS_OK) {
            std::fprintf(stderr, "batch generation workspace failed: %s\n", microgemm_status_string(status));
            goto cleanup;
        }
        seq->workspace_bytes = microgemm_decode_workspace_bytes(seq->workspace);
    }

    setup_end = std::chrono::steady_clock::now();
    out_result->setup_ms = std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
    total_start = setup_end;
    prefill_start = total_start;

    first_error_request = -1;
    first_error_status = MICROGEMM_STATUS_OK;
    try {
        request_pool->run(
            request_count_i,
            scheduler_inner_threads,
            [&](int req_i) {
            microgemm_batch_sequence_state* seq = &out_result->requests[static_cast<size_t>(req_i)];
            std::chrono::steady_clock::time_point seq_prefill_start;
            microgemm_status local_status;
            int first_token;

            seq_prefill_start = std::chrono::steady_clock::now();
            if (greedy_decode) {
                local_status = microgemm_decode_prefill_i8_next_token(
                    cfg,
                    weights,
                    seq->workspace,
                    seq->prompt_ids.data(),
                    seq->prompt_ids.size(),
                    &seq->kv,
                    &first_token
                );
            } else {
                local_status = microgemm_decode_prefill_i8(
                    cfg,
                    weights,
                    seq->workspace,
                    seq->prompt_ids.data(),
                    seq->prompt_ids.size(),
                    &seq->kv,
                    seq->logits.data(),
                    static_cast<int>(cfg->vocab_size)
                );
            }
            seq->prefill_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - seq_prefill_start
            ).count();
            if (local_status != MICROGEMM_STATUS_OK) {
                {
                    std::lock_guard<std::mutex> guard(first_error_mutex);
                    if (first_error_request < 0) {
                        first_error_request = req_i;
                        first_error_status = local_status;
                    }
                }
                return;
            }

            if (!greedy_decode) {
                first_token = sample_from_logits(seq->logits.data(), static_cast<int>(cfg->vocab_size), &seq->sampling);
            }
            seq->generated_tokens.push_back(first_token);
            seq->current_token = first_token;
            seq->finished = seq->generated_tokens.size() >= static_cast<size_t>(max_new_tokens)
                || (use_eos && first_token == eos_token);
            }
        );
    } catch (...) {
        std::fprintf(stderr, "batch generation failed during prompt prefill: worker pool error\n");
        goto cleanup;
    }

    if (first_error_request >= 0) {
        std::fprintf(
            stderr,
            "batch generation failed during prompt prefill for request %d: %s\n",
            first_error_request,
            microgemm_status_string(first_error_status)
        );
        goto cleanup;
    }

    active_requests = 0u;
    for (request_idx = 0u; request_idx < request_count; ++request_idx) {
        if (!out_result->requests[request_idx].finished) {
            ++active_requests;
        }
    }

    out_result->prefill_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prefill_start
    ).count();

    microgemm_decode_batch_profile_reset();
    microgemm_decode_batch_profile_set_enabled(1);
    decode_start = std::chrono::steady_clock::now();
    decode_end = decode_start;
    try {
        ready_request_indices.reserve(request_count);
        ready_workspaces.reserve(request_count);
        ready_token_ids.reserve(request_count);
        ready_positions.reserve(request_count);
        ready_kvs.reserve(request_count);
        ready_next_token_ids.reserve(request_count);
        if (!greedy_decode) {
            ready_logits.reserve(request_count);
            ready_logits_storage.assign(
                request_count * static_cast<size_t>(cfg->vocab_size),
                0.0f
            );
        }
    } catch (...) {
        std::fprintf(stderr, "batch generation failed: out of memory allocating scheduler buffers\n");
        goto cleanup;
    }
    while (active_requests > 0u) {
        int finished_this_round = 0;

        ready_request_indices.clear();
        ready_workspaces.clear();
        ready_token_ids.clear();
        ready_positions.clear();
        ready_kvs.clear();
        ready_next_token_ids.clear();
        ready_logits.clear();

        ++out_result->scheduler_iterations;
        for (int req_i = 0; req_i < request_count_i; ++req_i) {
            microgemm_batch_sequence_state* seq = &out_result->requests[static_cast<size_t>(req_i)];
            size_t ready_idx;

            if (seq->finished) {
                continue;
            }
            ready_idx = ready_request_indices.size();
            ready_request_indices.push_back(req_i);
            ready_workspaces.push_back(seq->workspace);
            ready_token_ids.push_back(seq->current_token);
            ready_positions.push_back(seq->kv.seq_len);
            ready_kvs.push_back(&seq->kv);
            if (!greedy_decode) {
                ready_logits.push_back(
                    ready_logits_storage.data() + ready_idx * static_cast<size_t>(cfg->vocab_size)
                );
            }
        }

        if (ready_request_indices.empty()) {
            std::fprintf(stderr, "batch generation failed during decode: no active request made progress\n");
            goto cleanup;
        }

        {
            std::chrono::steady_clock::time_point batch_start = std::chrono::steady_clock::now();
            double batch_ms;
            configure_openmp_thread_count(scheduler_lm_head_threads);
            if (greedy_decode) {
                ready_next_token_ids.resize(ready_request_indices.size());
                status = microgemm_decode_step_i8_batch_next_token(
                    cfg,
                    weights,
                    ready_workspaces.data(),
                    ready_token_ids.data(),
                    ready_positions.data(),
                    ready_kvs.data(),
                    ready_request_indices.size(),
                    ready_next_token_ids.data()
                );
            } else {
                status = microgemm_decode_step_i8_batch(
                    cfg,
                    weights,
                    ready_workspaces.data(),
                    ready_token_ids.data(),
                    ready_positions.data(),
                    ready_kvs.data(),
                    ready_request_indices.size(),
                    ready_logits.data(),
                    static_cast<int>(cfg->vocab_size)
                );
            }
            batch_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - batch_start
            ).count();
            if (status != MICROGEMM_STATUS_OK) {
                std::fprintf(
                    stderr,
                    "batch generation failed during batched decode: %s\n",
                    microgemm_status_string(status)
                );
                goto cleanup;
            }

            out_result->batched_decode_calls += 1u;
            out_result->batched_decode_tokens += ready_request_indices.size();
            out_result->batched_lm_head_calls += 1u;
            out_result->batched_lm_head_tokens += ready_request_indices.size();
            for (size_t ready_idx = 0u; ready_idx < ready_request_indices.size(); ++ready_idx) {
                microgemm_batch_sequence_state* seq =
                    &out_result->requests[static_cast<size_t>(ready_request_indices[ready_idx])];
                int next_token;

                seq->decode_ms += batch_ms / static_cast<double>(ready_request_indices.size());
                seq->kv.seq_len += 1;
                if (greedy_decode) {
                    next_token = ready_next_token_ids[ready_idx];
                } else {
                    next_token = sample_from_logits(
                        ready_logits[ready_idx],
                        static_cast<int>(cfg->vocab_size),
                        &seq->sampling
                    );
                }
                seq->generated_tokens.push_back(next_token);
                seq->current_token = next_token;

                if (seq->generated_tokens.size() >= static_cast<size_t>(max_new_tokens)
                        || (use_eos && next_token == eos_token)) {
                    seq->finished = true;
                    ++finished_this_round;
                }
            }
        }

        active_requests -= static_cast<size_t>(finished_this_round);
    }
    decode_end = std::chrono::steady_clock::now();
    microgemm_decode_batch_profile_get(&out_result->decode_profile);
    microgemm_decode_batch_profile_set_enabled(0);

    out_result->config = *cfg;
    out_result->scheduler_outer_threads = scheduler_outer_threads;
    out_result->scheduler_inner_threads = scheduler_inner_threads;
    out_result->scheduler_lm_head_threads = scheduler_lm_head_threads;
    out_result->loaded_model_bytes = microgemm_loaded_model_i8_bytes(loaded);
    out_result->workspace_bytes = 0u;
    for (request_idx = 0u; request_idx < request_count; ++request_idx) {
        out_result->workspace_bytes += out_result->requests[request_idx].workspace_bytes;
    }
    out_result->kv_cache_bytes =
        request_count * cfg->num_layers * block_count
        * 2u
        * static_cast<size_t>(cfg->num_kv_heads)
        * static_cast<size_t>(cfg->kv_block_size)
        * static_cast<size_t>(cfg->head_dim)
        * sizeof(float);
    out_result->decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    out_result->total_ms = std::chrono::duration<double, std::milli>(decode_end - total_start).count();

    for (request_idx = 0u; request_idx < request_count; ++request_idx) {
        out_result->generated_token_count += out_result->requests[request_idx].generated_tokens.size();
        if (out_result->requests[request_idx].finished) {
            ++out_result->finished_request_count;
        }
    }

    cleanup_start = std::chrono::steady_clock::now();
    delete request_pool;
    request_pool = NULL;
    destroy_batch_sequence_workspaces(&out_result->requests);
    microgemm_loaded_model_i8_destroy(loaded);
    loaded = NULL;
    microgemm_model_close(model);
    model = NULL;
    cleanup_end = std::chrono::steady_clock::now();
    out_result->model_cleanup_ms = std::chrono::duration<double, std::milli>(
        cleanup_end - cleanup_start
    ).count();
    return true;

cleanup:
    microgemm_decode_batch_profile_set_enabled(0);
    delete request_pool;
    destroy_batch_sequence_workspaces(&out_result->requests);
    microgemm_loaded_model_i8_destroy(loaded);
    microgemm_model_close(model);
    *out_result = microgemm_batch_generation_result_cpp();
    return false;
}

static bool parse_u32_arg(const char* raw, uint32_t* out_value) {
    char* end_ptr;
    unsigned long value;

    if (raw == NULL || out_value == NULL) {
        return false;
    }

    value = std::strtoul(raw, &end_ptr, 10);
    if (end_ptr == raw || *end_ptr != '\0' || value > static_cast<unsigned long>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }

    *out_value = static_cast<uint32_t>(value);
    return true;
}

static bool parse_nonnegative_int_arg(const char* raw, int* out_value) {
    char* end_ptr;
    long value;

    if (raw == NULL || out_value == NULL) {
        return false;
    }

    value = std::strtol(raw, &end_ptr, 10);
    if (end_ptr == raw || *end_ptr != '\0' || value < 0 || value > INT_MAX) {
        return false;
    }

    *out_value = static_cast<int>(value);
    return true;
}


static bool parse_prompt_ids_arg(const char* raw, std::vector<int>* out_ids) {
    const char* cursor;
    char* end_ptr;
    long value;

    if (raw == NULL || out_ids == NULL) {
        return false;
    }
    out_ids->clear();
    cursor = raw;
    while (*cursor != '\0') {
        while (*cursor == ',' || std::isspace(static_cast<unsigned char>(*cursor)) != 0) {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        value = std::strtol(cursor, &end_ptr, 10);
        if (end_ptr == cursor || value < 0 || value > INT_MAX) {
            return false;
        }
        out_ids->push_back(static_cast<int>(value));
        cursor = end_ptr;
        while (*cursor != '\0' && *cursor != ',') {
            if (std::isspace(static_cast<unsigned char>(*cursor)) == 0) {
                return false;
            }
            ++cursor;
        }
    }
    return !out_ids->empty();
}

static bool parse_float_arg(const char* raw, float* out_value) {
    char* end_ptr;
    double value;

    if (raw == NULL || out_value == NULL) {
        return false;
    }

    value = std::strtod(raw, &end_ptr);
    if (end_ptr == raw || *end_ptr != '\0') {
        return false;
    }

    *out_value = static_cast<float>(value);
    return true;
}

static uint64_t microgemm_rng_next(uint64_t* state) {
    uint64_t x;
    if (state == NULL) {
        return 0u;
    }
    x = *state == 0u ? 0x9e3779b97f4a7c15ull : *state;
    x ^= x >> 12u;
    x ^= x << 25u;
    x ^= x >> 27u;
    *state = x;
    return x * 2685821657736338717ull;
}

static double microgemm_rng_uniform01(uint64_t* state) {
    uint64_t value = microgemm_rng_next(state);
    return (value >> 11u) * (1.0 / 9007199254740992.0);
}

static int sample_from_logits(
    const float* logits,
    int vocab_size,
    microgemm_sampling_config* sampling
) {
    struct candidate {
        int token_id;
        double logit;
        double prob;
    };

    std::vector<candidate> candidates;
    double max_logit = -std::numeric_limits<double>::infinity();
    double normalizer = 0.0;
    double cumulative = 0.0;
    double cutoff;
    int i;

    if (logits == NULL || vocab_size <= 0 || sampling == NULL) {
        return 0;
    }
    if (sampling->temperature <= 0.0f || sampling->top_k == 1u) {
        int best_id = 0;
        float best_logit = logits[0];
        for (i = 1; i < vocab_size; ++i) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    if (sampling->top_k > 0u && static_cast<size_t>(sampling->top_k) < static_cast<size_t>(vocab_size)) {
        auto min_heap_cmp = [](const candidate& a, const candidate& b) {
            return a.logit > b.logit;
        };
        std::priority_queue<candidate, std::vector<candidate>, decltype(min_heap_cmp)> heap(min_heap_cmp);
        size_t keep_count = static_cast<size_t>(sampling->top_k);

        for (i = 0; i < vocab_size; ++i) {
            double scaled = static_cast<double>(logits[i]) / static_cast<double>(sampling->temperature);
            candidate item;
            item.token_id = i;
            item.logit = scaled;
            item.prob = 0.0;
            if (heap.size() < keep_count) {
                heap.push(item);
            } else if (scaled > heap.top().logit) {
                heap.pop();
                heap.push(item);
            }
        }

        candidates.reserve(heap.size());
        while (!heap.empty()) {
            candidates.push_back(heap.top());
            heap.pop();
        }
        std::sort(candidates.begin(), candidates.end(), [](const candidate& a, const candidate& b) {
            return a.logit > b.logit;
        });
        for (size_t idx = 0u; idx < candidates.size(); ++idx) {
            if (candidates[idx].logit > max_logit) {
                max_logit = candidates[idx].logit;
            }
        }
    } else {
        candidates.reserve(static_cast<size_t>(vocab_size));
        for (i = 0; i < vocab_size; ++i) {
            double scaled = static_cast<double>(logits[i]) / static_cast<double>(sampling->temperature);
            candidate item;
            item.token_id = i;
            item.logit = scaled;
            item.prob = 0.0;
            candidates.push_back(item);
            if (scaled > max_logit) {
                max_logit = scaled;
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const candidate& a, const candidate& b) {
            return a.logit > b.logit;
        });
    }

    for (size_t idx = 0u; idx < candidates.size(); ++idx) {
        candidates[idx].prob = std::exp(candidates[idx].logit - max_logit);
        normalizer += candidates[idx].prob;
    }
    if (!(normalizer > 0.0)) {
        return candidates.empty() ? 0 : candidates.front().token_id;
    }
    for (size_t idx = 0u; idx < candidates.size(); ++idx) {
        candidates[idx].prob /= normalizer;
    }

    if (sampling->top_p > 0.0f && sampling->top_p < 1.0f) {
        std::vector<candidate> filtered;
        filtered.reserve(candidates.size());
        cumulative = 0.0;
        for (size_t idx = 0u; idx < candidates.size(); ++idx) {
            filtered.push_back(candidates[idx]);
            cumulative += candidates[idx].prob;
            if (cumulative >= static_cast<double>(sampling->top_p)) {
                break;
            }
        }
        candidates.swap(filtered);
        normalizer = 0.0;
        for (size_t idx = 0u; idx < candidates.size(); ++idx) {
            normalizer += candidates[idx].prob;
        }
        for (size_t idx = 0u; idx < candidates.size(); ++idx) {
            candidates[idx].prob /= normalizer;
        }
    }

    cutoff = microgemm_rng_uniform01(&sampling->rng_state);
    cumulative = 0.0;
    for (size_t idx = 0u; idx < candidates.size(); ++idx) {
        cumulative += candidates[idx].prob;
        if (cutoff <= cumulative || idx + 1u == candidates.size()) {
            return candidates[idx].token_id;
        }
    }

    return candidates.front().token_id;
}

static bool parse_text_args(int argc, char** argv, microgemm_text_args* out_args) {
    int i;

    if (out_args == NULL) {
        return false;
    }

    *out_args = microgemm_text_args();
    for (i = 0; i < argc; ++i) {
        const char* key = argv[i];
        if (std::strcmp(key, "--prompt") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--prompt requires a value\n");
                return false;
            }
            out_args->prompt = argv[++i];
        } else if (std::strcmp(key, "--prompt-ids") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--prompt-ids requires a comma-separated id list\n");
                return false;
            }
            if (!parse_prompt_ids_arg(argv[++i], &out_args->prompt_ids)) {
                std::fprintf(stderr, "--prompt-ids requires comma-separated non-negative integers\n");
                return false;
            }
        } else if (std::strcmp(key, "--prompt-file") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--prompt-file requires a path\n");
                return false;
            }
            if (!read_text_file(argv[++i], &out_args->prompt)) {
                std::fprintf(stderr, "failed to read prompt file\n");
                return false;
            }
        } else if (std::strcmp(key, "--max-new-tokens") == 0) {
            if (i + 1 >= argc || !parse_u32_arg(argv[++i], &out_args->max_new_tokens) || out_args->max_new_tokens == 0u) {
                std::fprintf(stderr, "--max-new-tokens requires a positive integer\n");
                return false;
            }
        } else if (std::strcmp(key, "--max-seq-len") == 0) {
            if (i + 1 >= argc || !parse_u32_arg(argv[++i], &out_args->max_seq_len) || out_args->max_seq_len == 0u) {
                std::fprintf(stderr, "--max-seq-len requires a positive integer\n");
                return false;
            }
        } else if (std::strcmp(key, "--temperature") == 0) {
            if (i + 1 >= argc || !parse_float_arg(argv[++i], &out_args->temperature) || out_args->temperature < 0.0f) {
                std::fprintf(stderr, "--temperature requires a non-negative number\n");
                return false;
            }
        } else if (std::strcmp(key, "--top-p") == 0) {
            if (i + 1 >= argc || !parse_float_arg(argv[++i], &out_args->top_p) || out_args->top_p <= 0.0f || out_args->top_p > 1.0f) {
                std::fprintf(stderr, "--top-p requires a number in (0, 1]\n");
                return false;
            }
        } else if (std::strcmp(key, "--top-k") == 0) {
            if (i + 1 >= argc || !parse_u32_arg(argv[++i], &out_args->top_k)) {
                std::fprintf(stderr, "--top-k requires a non-negative integer\n");
                return false;
            }
        } else if (std::strcmp(key, "--seed") == 0) {
            uint32_t seed32 = 0u;
            if (i + 1 >= argc || !parse_u32_arg(argv[++i], &seed32)) {
                std::fprintf(stderr, "--seed requires a non-negative integer\n");
                return false;
            }
            out_args->seed = static_cast<uint64_t>(seed32);
        } else if (std::strcmp(key, "--eos-token") == 0) {
            if (i + 1 >= argc || !parse_nonnegative_int_arg(argv[++i], &out_args->eos_token)) {
                std::fprintf(stderr, "--eos-token requires a non-negative integer\n");
                return false;
            }
            out_args->use_eos = true;
        } else if (std::strcmp(key, "--ignore-eos") == 0) {
            out_args->ignore_eos = true;
            out_args->use_eos = false;
            out_args->eos_token = -1;
        } else if (std::strcmp(key, "--bos-token") == 0) {
            if (i + 1 >= argc || !parse_nonnegative_int_arg(argv[++i], &out_args->bos_token)) {
                std::fprintf(stderr, "--bos-token requires a non-negative integer\n");
                return false;
            }
            out_args->use_bos = true;
        } else if (std::strcmp(key, "--no-bos") == 0) {
            out_args->use_bos = false;
            out_args->disable_auto_bos = true;
            out_args->bos_token = -1;
        } else if (std::strcmp(key, "--skip-special-tokens") == 0) {
            out_args->skip_special_tokens = true;
        } else if (std::strcmp(key, "--keep-special-tokens") == 0) {
            out_args->skip_special_tokens = false;
        } else {
            std::fprintf(stderr, "unknown flag: %s\n", key);
            return false;
        }
    }

    if (out_args->prompt.empty() && out_args->prompt_ids.empty()) {
        std::fprintf(stderr, "generate requires --prompt, --prompt-file, or --prompt-ids\n");
        return false;
    }

    return true;
}

static void append_nonempty_lines(const std::string& text, std::vector<std::string>* out_prompts) {
    std::istringstream input(text);
    std::string line;

    if (out_prompts == NULL) {
        return;
    }
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            out_prompts->push_back(line);
        }
    }
}

static bool parse_batch_text_args(int argc, char** argv, microgemm_batch_text_args* out_args) {
    int i;

    if (out_args == NULL) {
        return false;
    }

    *out_args = microgemm_batch_text_args();
    for (i = 0; i < argc; ++i) {
        const char* key = argv[i];
        if (std::strcmp(key, "--prompt") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--prompt requires a value\n");
                return false;
            }
            out_args->prompts.push_back(argv[++i]);
        } else if (std::strcmp(key, "--prompt-ids") == 0) {
            std::vector<int> ids;
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--prompt-ids requires a comma-separated id list\n");
                return false;
            }
            if (!parse_prompt_ids_arg(argv[++i], &ids)) {
                std::fprintf(stderr, "--prompt-ids requires comma-separated non-negative integers\n");
                return false;
            }
            out_args->prompt_id_batches.push_back(ids);
        } else if (std::strcmp(key, "--prompt-file") == 0) {
            std::string prompt;
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--prompt-file requires a path\n");
                return false;
            }
            if (!read_text_file(argv[++i], &prompt)) {
                std::fprintf(stderr, "failed to read prompt file\n");
                return false;
            }
            out_args->prompts.push_back(prompt);
        } else if (std::strcmp(key, "--prompts-file") == 0) {
            std::string prompts_text;
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--prompts-file requires a path\n");
                return false;
            }
            if (!read_text_file(argv[++i], &prompts_text)) {
                std::fprintf(stderr, "failed to read prompts file\n");
                return false;
            }
            append_nonempty_lines(prompts_text, &out_args->prompts);
        } else if (std::strcmp(key, "--max-new-tokens") == 0) {
            if (i + 1 >= argc || !parse_u32_arg(argv[++i], &out_args->max_new_tokens) || out_args->max_new_tokens == 0u) {
                std::fprintf(stderr, "--max-new-tokens requires a positive integer\n");
                return false;
            }
        } else if (std::strcmp(key, "--max-seq-len") == 0) {
            if (i + 1 >= argc || !parse_u32_arg(argv[++i], &out_args->max_seq_len) || out_args->max_seq_len == 0u) {
                std::fprintf(stderr, "--max-seq-len requires a positive integer\n");
                return false;
            }
        } else if (std::strcmp(key, "--temperature") == 0) {
            if (i + 1 >= argc || !parse_float_arg(argv[++i], &out_args->temperature) || out_args->temperature < 0.0f) {
                std::fprintf(stderr, "--temperature requires a non-negative number\n");
                return false;
            }
        } else if (std::strcmp(key, "--top-p") == 0) {
            if (i + 1 >= argc || !parse_float_arg(argv[++i], &out_args->top_p) || out_args->top_p <= 0.0f || out_args->top_p > 1.0f) {
                std::fprintf(stderr, "--top-p requires a number in (0, 1]\n");
                return false;
            }
        } else if (std::strcmp(key, "--top-k") == 0) {
            if (i + 1 >= argc || !parse_u32_arg(argv[++i], &out_args->top_k)) {
                std::fprintf(stderr, "--top-k requires a non-negative integer\n");
                return false;
            }
        } else if (std::strcmp(key, "--seed") == 0) {
            uint32_t seed32 = 0u;
            if (i + 1 >= argc || !parse_u32_arg(argv[++i], &seed32)) {
                std::fprintf(stderr, "--seed requires a non-negative integer\n");
                return false;
            }
            out_args->seed = static_cast<uint64_t>(seed32);
        } else if (std::strcmp(key, "--eos-token") == 0) {
            if (i + 1 >= argc || !parse_nonnegative_int_arg(argv[++i], &out_args->eos_token)) {
                std::fprintf(stderr, "--eos-token requires a non-negative integer\n");
                return false;
            }
            out_args->use_eos = true;
        } else if (std::strcmp(key, "--ignore-eos") == 0) {
            out_args->ignore_eos = true;
            out_args->use_eos = false;
            out_args->eos_token = -1;
        } else if (std::strcmp(key, "--bos-token") == 0) {
            if (i + 1 >= argc || !parse_nonnegative_int_arg(argv[++i], &out_args->bos_token)) {
                std::fprintf(stderr, "--bos-token requires a non-negative integer\n");
                return false;
            }
            out_args->use_bos = true;
        } else if (std::strcmp(key, "--no-bos") == 0) {
            out_args->use_bos = false;
            out_args->disable_auto_bos = true;
            out_args->bos_token = -1;
        } else if (std::strcmp(key, "--skip-special-tokens") == 0) {
            out_args->skip_special_tokens = true;
        } else if (std::strcmp(key, "--keep-special-tokens") == 0) {
            out_args->skip_special_tokens = false;
        } else {
            std::fprintf(stderr, "unknown flag: %s\n", key);
            return false;
        }
    }

    if (out_args->prompts.empty() && out_args->prompt_id_batches.empty()) {
        std::fprintf(stderr, "batch-generate requires --prompt, --prompt-file, --prompts-file, or --prompt-ids\n");
        return false;
    }
    if (!out_args->prompts.empty() && !out_args->prompt_id_batches.empty()) {
        std::fprintf(stderr, "batch-generate cannot mix text prompts and --prompt-ids in the same command\n");
        return false;
    }

    return true;
}

static void print_usage(void) {
    std::puts("MicroGemm Text CLI");
    std::puts("Usage:");
    std::puts("  microgemm-text generate <model.mgm> <tokenizer.json> --prompt \"...\" [--max-new-tokens N] [--max-seq-len N] [--temperature F] [--top-k N] [--top-p F] [--seed N] [--bos-token N] [--eos-token N] [--ignore-eos] [--no-bos] [--skip-special-tokens]");
    std::puts("  microgemm-text generate <model.mgm> <tokenizer.json> --prompt-file prompt.txt [--prompt-ids 1,2,3] [--max-new-tokens N] [--max-seq-len N] [--temperature F] [--top-k N] [--top-p F] [--seed N] [--bos-token N] [--eos-token N] [--ignore-eos] [--no-bos] [--skip-special-tokens]");
    std::puts("  microgemm-text batch-generate <model.mgm> <tokenizer.json> --prompt-file a.txt --prompt-file b.txt [--prompt-ids 1,2,3 --prompt-ids 4,5,6] [--max-new-tokens N] [--max-seq-len N] [--ignore-eos]");
    std::puts("  microgemm-text batch-generate <model.mgm> <tokenizer.json> --prompts-file prompts.txt [--max-new-tokens N] [--max-seq-len N] [--ignore-eos]");
}

static int command_generate(int argc, char** argv) {
    microgemm_tokenizer tokenizer;
    microgemm_text_args args;
    microgemm_generation_result_cpp result;
    microgemm_sampling_config sampling;
    std::vector<int> prompt_ids;
    std::vector<int> full_ids;
    std::string error;
    std::string generated_text;
    std::string full_text;
    double prefill_tps;
    double decode_tps;
    double total_tps;

    if (argc < 5) {
        std::fprintf(stderr, "generate requires <model.mgm> <tokenizer.json> and prompt flags\n");
        return 1;
    }
    if (!microgemm_tokenizer_load(argv[3], &tokenizer, &error)) {
        std::fprintf(stderr, "tokenizer load failed: %s\n", error.c_str());
        return 1;
    }
    if (!parse_text_args(argc - 4, argv + 4, &args)) {
        return 1;
    }
    if (!args.prompt_ids.empty()) {
        prompt_ids = args.prompt_ids;
    } else {
        if (!microgemm_tokenizer_encode(tokenizer, args.prompt, &prompt_ids, &error)) {
            std::fprintf(stderr, "tokenizer encode failed: %s\n", error.c_str());
            return 1;
        }
    }

    if (args.use_bos) {
        prompt_ids.insert(prompt_ids.begin(), args.bos_token);
    } else if (!args.disable_auto_bos && tokenizer.bos_token_id >= 0) {
        prompt_ids.insert(prompt_ids.begin(), tokenizer.bos_token_id);
    }
    if (!args.ignore_eos && !args.use_eos && tokenizer.eos_token_id >= 0) {
        args.use_eos = true;
        args.eos_token = tokenizer.eos_token_id;
    }
    sampling.temperature = args.temperature;
    sampling.top_k = args.top_k;
    sampling.top_p = args.top_p;
    sampling.rng_state = args.seed;

    if (!run_greedy_generation_cpp(
            argv[2],
            prompt_ids,
            args.max_new_tokens,
            args.max_seq_len,
            args.use_eos,
            args.eos_token,
            &sampling,
            &result)) {
        return 1;
    }

    full_ids = prompt_ids;
    full_ids.insert(full_ids.end(), result.generated_tokens.begin(), result.generated_tokens.end());
    generated_text = microgemm_tokenizer_decode(tokenizer, result.generated_tokens, args.skip_special_tokens);
    full_text = microgemm_tokenizer_decode(tokenizer, full_ids, args.skip_special_tokens);
    prefill_tps = result.prefill_ms > 0.0 ? (static_cast<double>(prompt_ids.size()) * 1000.0 / result.prefill_ms) : 0.0;
    decode_tps = result.decode_ms > 0.0 && result.generated_tokens.size() > 1u
        ? (static_cast<double>(result.generated_tokens.size() - 1u) * 1000.0 / result.decode_ms)
        : 0.0;
    total_tps = result.total_ms > 0.0
        ? (static_cast<double>(prompt_ids.size() + result.generated_tokens.size()) * 1000.0 / result.total_ms)
        : 0.0;

    std::printf("prompt_token_count: %zu\n", prompt_ids.size());
    std::printf("generated_token_count: %zu\n", result.generated_tokens.size());
    std::printf("temperature: %.4f\n", args.temperature);
    std::printf("top_k: %u\n", args.top_k);
    std::printf("top_p: %.4f\n", args.top_p);
    std::printf("seed: %llu\n", static_cast<unsigned long long>(args.seed));
    std::printf("prefill_ms: %.3f\n", result.prefill_ms);
    std::printf("decode_ms: %.3f\n", result.decode_ms);
    std::printf("total_ms: %.3f\n", result.total_ms);
    std::printf("loaded_model_bytes: %zu\n", result.loaded_model_bytes);
    std::printf("workspace_bytes: %zu\n", result.workspace_bytes);
    std::printf("kv_cache_bytes: %zu\n", result.kv_cache_bytes);
    std::printf("runtime_total_bytes: %zu\n",
        result.loaded_model_bytes + result.workspace_bytes + result.kv_cache_bytes);
    std::printf("prefill_tps: %.3f\n", prefill_tps);
    std::printf("decode_tps: %.3f\n", decode_tps);
    std::printf("total_tps: %.3f\n", total_tps);
    std::printf("generated_text:\n%s\n", generated_text.c_str());
    std::printf("full_text:\n%s\n", full_text.c_str());
    return 0;
}

static int command_batch_generate(int argc, char** argv) {
    microgemm_tokenizer tokenizer;
    microgemm_batch_text_args args;
    microgemm_batch_generation_result_cpp result;
    microgemm_sampling_config sampling;
    std::vector<std::vector<int> > prompt_batches;
    std::string error;
    size_t request_idx;
    size_t prompt_token_count = 0u;
    double prefill_tps;
    double decode_tps;
    double total_tps;
    double tokenizer_load_ms = 0.0;
    double prompt_encode_ms = 0.0;
    double command_total_ms = 0.0;
    std::chrono::steady_clock::time_point command_start;
    std::chrono::steady_clock::time_point tokenizer_start;
    std::chrono::steady_clock::time_point tokenizer_end;
    std::chrono::steady_clock::time_point encode_start;
    std::chrono::steady_clock::time_point encode_end;

    command_start = std::chrono::steady_clock::now();
    if (argc < 5) {
        std::fprintf(stderr, "batch-generate requires <model.mgm> <tokenizer.json> and prompt flags\n");
        return 1;
    }
    tokenizer_start = std::chrono::steady_clock::now();
    if (!microgemm_tokenizer_load(argv[3], &tokenizer, &error)) {
        std::fprintf(stderr, "tokenizer load failed: %s\n", error.c_str());
        return 1;
    }
    tokenizer_end = std::chrono::steady_clock::now();
    tokenizer_load_ms = std::chrono::duration<double, std::milli>(
        tokenizer_end - tokenizer_start
    ).count();
    if (!parse_batch_text_args(argc - 4, argv + 4, &args)) {
        return 1;
    }

    if (!args.ignore_eos && !args.use_eos && tokenizer.eos_token_id >= 0) {
        args.use_eos = true;
        args.eos_token = tokenizer.eos_token_id;
    }

    if (!args.prompt_id_batches.empty()) {
        prompt_batches = args.prompt_id_batches;
    } else {
        prompt_batches.resize(args.prompts.size());
    }
    encode_start = std::chrono::steady_clock::now();
    for (request_idx = 0u; request_idx < prompt_batches.size(); ++request_idx) {
        if (args.prompt_id_batches.empty()) {
            if (!microgemm_tokenizer_encode(tokenizer, args.prompts[request_idx], &prompt_batches[request_idx], &error)) {
                std::fprintf(stderr, "tokenizer encode failed for request %zu: %s\n", request_idx, error.c_str());
                return 1;
            }
        }
        if (args.use_bos) {
            prompt_batches[request_idx].insert(prompt_batches[request_idx].begin(), args.bos_token);
        } else if (!args.disable_auto_bos && tokenizer.bos_token_id >= 0) {
            prompt_batches[request_idx].insert(prompt_batches[request_idx].begin(), tokenizer.bos_token_id);
        }
        prompt_token_count += prompt_batches[request_idx].size();
    }
    encode_end = std::chrono::steady_clock::now();
    prompt_encode_ms = std::chrono::duration<double, std::milli>(
        encode_end - encode_start
    ).count();

    sampling.temperature = args.temperature;
    sampling.top_k = args.top_k;
    sampling.top_p = args.top_p;
    sampling.rng_state = args.seed;

    if (!run_continuous_batch_generation_cpp(
            argv[2],
            prompt_batches,
            args.max_new_tokens,
            args.max_seq_len,
            args.use_eos,
            args.eos_token,
            &sampling,
            &result)) {
        return 1;
    }
    command_total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - command_start
    ).count();

    prefill_tps = result.prefill_ms > 0.0
        ? (static_cast<double>(prompt_token_count) * 1000.0 / result.prefill_ms)
        : 0.0;
    decode_tps = result.decode_ms > 0.0
        ? (static_cast<double>(result.generated_token_count) * 1000.0 / result.decode_ms)
        : 0.0;
    total_tps = result.total_ms > 0.0
        ? (static_cast<double>(prompt_token_count + result.generated_token_count) * 1000.0 / result.total_ms)
        : 0.0;

    std::printf("batch_profile_emit_version: 1\n");
    std::printf("mode: continuous_batch\n");
    std::printf("native_continuous_batching: 1\n");
    std::printf("batch_size: %zu\n", result.requests.size());
    std::printf("prompt_token_count: %zu\n", prompt_token_count);
    std::printf("generated_token_count: %zu\n", result.generated_token_count);
    std::printf("finished_request_count: %zu\n", result.finished_request_count);
    std::printf("scheduler_iterations: %zu\n", result.scheduler_iterations);
    std::printf("scheduler_outer_threads: %d\n", result.scheduler_outer_threads);
    std::printf("scheduler_inner_threads: %d\n", result.scheduler_inner_threads);
    std::printf("scheduler_lm_head_threads: %d\n", result.scheduler_lm_head_threads);
    std::printf("batched_decode: 1\n");
    std::printf("batched_decode_calls: %zu\n", result.batched_decode_calls);
    std::printf("batched_decode_tokens: %zu\n", result.batched_decode_tokens);
    std::printf("batched_lm_head: 1\n");
    std::printf("batched_lm_head_calls: %zu\n", result.batched_lm_head_calls);
    std::printf("batched_lm_head_tokens: %zu\n", result.batched_lm_head_tokens);
    std::printf("tokenizer_load_ms: %.3f\n", tokenizer_load_ms);
    std::printf("prompt_encode_ms: %.3f\n", prompt_encode_ms);
    std::printf("model_open_ms: %.3f\n", result.model_open_ms);
    std::printf("model_load_ms: %.3f\n", result.model_load_ms);
    std::printf("model_cleanup_ms: %.3f\n", result.model_cleanup_ms);
    std::printf("command_total_ms: %.3f\n", command_total_ms);
    std::printf("setup_ms: %.3f\n", result.setup_ms);
    std::printf("prefill_ms: %.3f\n", result.prefill_ms);
    std::printf("decode_ms: %.3f\n", result.decode_ms);
    std::printf("batch_profile_calls: %llu\n",
        static_cast<unsigned long long>(result.decode_profile.calls));
    std::printf("batch_profile_tokens: %llu\n",
        static_cast<unsigned long long>(result.decode_profile.tokens));
    std::printf("batch_profile_total_ms: %.3f\n", result.decode_profile.total_ms);
    std::printf("batch_profile_alloc_ms: %.3f\n", result.decode_profile.alloc_ms);
    std::printf("batch_profile_embed_ms: %.3f\n", result.decode_profile.embed_ms);
    std::printf("batch_profile_input_norm_ms: %.3f\n", result.decode_profile.input_norm_ms);
    std::printf("batch_profile_qkv_ms: %.3f\n", result.decode_profile.qkv_ms);
    std::printf("batch_profile_rope_kv_ms: %.3f\n", result.decode_profile.rope_kv_ms);
    std::printf("batch_profile_attention_ms: %.3f\n", result.decode_profile.attention_ms);
    std::printf("batch_profile_o_proj_ms: %.3f\n", result.decode_profile.o_proj_ms);
    std::printf("batch_profile_post_norm_ms: %.3f\n", result.decode_profile.post_norm_ms);
    std::printf("batch_profile_gate_up_ms: %.3f\n", result.decode_profile.gate_up_ms);
    std::printf("batch_profile_gate_up_quant_ms: %.3f\n", result.decode_profile.gate_up_quant_ms);
    std::printf("batch_profile_gate_up_dot_ms: %.3f\n", result.decode_profile.gate_up_dot_ms);
    std::printf("batch_profile_activation_ms: %.3f\n", result.decode_profile.activation_ms);
    std::printf("batch_profile_down_proj_ms: %.3f\n", result.decode_profile.down_proj_ms);
    std::printf("batch_profile_down_proj_quant_ms: %.3f\n", result.decode_profile.down_proj_quant_ms);
    std::printf("batch_profile_down_proj_dot_ms: %.3f\n", result.decode_profile.down_proj_dot_ms);
    std::printf("batch_profile_final_norm_ms: %.3f\n", result.decode_profile.final_norm_ms);
    std::printf("batch_profile_lm_head_ms: %.3f\n", result.decode_profile.lm_head_ms);
    std::printf("batch_profile_copy_ms: %.3f\n", result.decode_profile.copy_ms);
    std::printf("batch_profile_cleanup_ms: %.3f\n", result.decode_profile.cleanup_ms);
    std::printf("batch_profile_groupwise_gemv_tile_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_gemv_tile_calls);
    std::printf("batch_profile_groupwise_i8_row_pair_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_i8_row_pair_calls);
    std::printf("batch_profile_groupwise_i4_row_pair_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_i4_row_pair_calls);
    std::printf("batch_profile_groupwise_lm_head_argmax_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_lm_head_argmax_calls);
    std::printf("batch_profile_lm_head_stack_best_calls: %llu\n",
        (unsigned long long)result.decode_profile.lm_head_stack_best_calls);
    std::printf("batch_profile_groupwise_gate_up_fused_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_gate_up_fused_calls);
    std::printf("batch_profile_groupwise_i8_gate_safe_combined_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_i8_gate_safe_combined_calls);
    std::printf("batch_profile_groupwise_i8_gate_safe_combined_tile8_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_i8_gate_safe_combined_tile8_calls);
    std::printf("batch_profile_groupwise_i8_gate_tile8_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_i8_gate_tile8_calls);
    std::printf("batch_profile_groupwise_i8_gate_biased_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_i8_gate_biased_calls);
    std::printf("batch_profile_groupwise_i8_gate_pair_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_i8_gate_pair_calls);
    std::printf("batch_profile_groupwise_i8_gate_pair_unroll64_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_i8_gate_pair_unroll64_calls);
    std::printf("batch_profile_groupwise_i8_gate_pair_unroll128_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_i8_gate_pair_unroll128_calls);
    std::printf("batch_profile_groupwise_i8_gate_pair8_split_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_i8_gate_pair8_split_calls);
    std::printf("batch_profile_groupwise_i8_gate_prefetch_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_i8_gate_prefetch_calls);
    std::printf("batch_profile_groupwise_lm_head_row_pair_calls: %llu\n",
        (unsigned long long)result.decode_profile.groupwise_lm_head_row_pair_calls);
    std::printf("total_ms: %.3f\n", result.total_ms);
    std::printf("loaded_model_bytes: %zu\n", result.loaded_model_bytes);
    std::printf("workspace_bytes: %zu\n", result.workspace_bytes);
    std::printf("kv_cache_bytes: %zu\n", result.kv_cache_bytes);
    std::printf("runtime_total_bytes: %zu\n",
        result.loaded_model_bytes + result.workspace_bytes + result.kv_cache_bytes);
    std::printf("prefill_tps: %.3f\n", prefill_tps);
    std::printf("decode_tps: %.3f\n", decode_tps);
    std::printf("total_tps: %.3f\n", total_tps);

    for (request_idx = 0u; request_idx < result.requests.size(); ++request_idx) {
        const microgemm_batch_sequence_state& request = result.requests[request_idx];
        std::printf("request_%zu_prompt_token_count: %zu\n", request_idx, request.prompt_ids.size());
        std::printf("request_%zu_generated_token_count: %zu\n", request_idx, request.generated_tokens.size());
        std::printf("request_%zu_prefill_ms: %.3f\n", request_idx, request.prefill_ms);
        std::printf("request_%zu_decode_ms: %.3f\n", request_idx, request.decode_ms);
        std::printf("request_%zu_finished: %d\n", request_idx, request.finished ? 1 : 0);
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    if (std::strcmp(argv[1], "generate") == 0) {
        return command_generate(argc, argv);
    }
    if (std::strcmp(argv[1], "batch-generate") == 0) {
        return command_batch_generate(argc, argv);
    }

    print_usage();
    return 1;
}

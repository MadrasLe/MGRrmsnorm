#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;
using at::indexing::Slice;

static inline py::object vectorcall_positional(
    PyObject* callable,
    PyObject* const* args,
    size_t nargs
) {
    PyObject* out = PyObject_Vectorcall(callable, args, nargs, nullptr);
    if (out == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(out);
}

py::object run_decode_layers_full_attention(
    const py::list& layers,
    const py::list& layer_kv_caches,
    py::object hidden,
    py::object cos,
    py::object sin,
    py::object positions,
    py::object block_table,
    py::object seq_lens,
    py::object seq_lens_kv,
    py::object decode_phys_blocks,
    py::object decode_blk_offsets,
    py::object timing_events = py::none()) {
    const py::ssize_t n = py::len(layers);
    if (py::len(layer_kv_caches) != n) {
        throw std::runtime_error("layers and layer_kv_caches must have same length");
    }

    for (py::ssize_t i = 0; i < n; ++i) {
        py::object layer = layers[i];
        py::object kv_cache = layer_kv_caches[i];

        py::object out_obj = layer.attr("decode_forward")(
            hidden,
            cos,
            sin,
            positions,
            kv_cache,
            block_table,
            seq_lens,
            seq_lens_kv,
            decode_phys_blocks,
            decode_blk_offsets,
            py::none(),
            py::none(),
            py::bool_(true),
            timing_events
        );
        py::tuple out = out_obj.cast<py::tuple>();
        if (out.size() < 1) {
            throw std::runtime_error("decode_forward must return at least hidden_states");
        }
        hidden = out[0];
    }
    return hidden;
}

py::object run_decode_fns_full_attention(
    const py::list& decode_fns,
    const py::list& layer_kv_caches,
    py::object hidden,
    py::object cos,
    py::object sin,
    py::object positions,
    py::object block_table,
    py::object seq_lens,
    py::object seq_lens_kv,
    py::object decode_phys_blocks,
    py::object decode_blk_offsets) {
    const py::ssize_t n = py::len(decode_fns);
    const py::ssize_t n_kv = static_cast<py::ssize_t>(py::len(layer_kv_caches));
    if (n_kv != n) {
        throw std::runtime_error("decode_fns and layer_kv_caches must have same length");
    }

    PyObject* decode_fns_list = decode_fns.ptr();
    PyObject* layer_kv_list = layer_kv_caches.ptr();
    PyObject* cos_ptr = cos.ptr();
    PyObject* sin_ptr = sin.ptr();
    PyObject* positions_ptr = positions.ptr();
    PyObject* block_table_ptr = block_table.ptr();
    PyObject* seq_lens_ptr = seq_lens.ptr();
    PyObject* seq_lens_kv_ptr = seq_lens_kv.ptr();
    PyObject* decode_phys_blocks_ptr = decode_phys_blocks.ptr();
    PyObject* decode_blk_offsets_ptr = decode_blk_offsets.ptr();

    for (py::ssize_t i = 0; i < n; ++i) {
        PyObject* decode_fn = PyList_GET_ITEM(decode_fns_list, i);
        PyObject* kv_cache = PyList_GET_ITEM(layer_kv_list, i);
        PyObject* args[10] = {
            hidden.ptr(),
            cos_ptr,
            sin_ptr,
            positions_ptr,
            kv_cache,
            block_table_ptr,
            seq_lens_ptr,
            seq_lens_kv_ptr,
            decode_phys_blocks_ptr,
            decode_blk_offsets_ptr,
        };
        hidden = vectorcall_positional(decode_fn, args, 10);
    }
    return hidden;
}

py::tuple run_decode_steps_full_attention(
    const py::list& decode_fns,
    const py::list& layer_kv_caches,
    py::object embed_fn,
    py::object decode_head_fn,
    py::object decode_next_token_fn,
    torch::Tensor cur_ids,              // [B, 1], mutated in-place
    torch::Tensor cur_pos,              // [B, 1], mutated in-place
    torch::Tensor block_table,          // [B, max_blocks]
    torch::Tensor seq_lens,             // [B], local tensor (not manager-owned)
    torch::Tensor blk_ids,              // [B]
    torch::Tensor decode_blk_offsets,   // [B]
    int64_t block_size,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t num_steps,
    double embed_scale
) {
    const py::ssize_t n_layers = py::len(decode_fns);
    const py::ssize_t n_kv = static_cast<py::ssize_t>(py::len(layer_kv_caches));
    if (n_kv != n_layers) {
        throw std::runtime_error("decode_fns and layer_kv_caches must have same length");
    }
    const auto bsz = cur_ids.size(0);
    const bool single_seq = (bsz == 1);
    auto all_tokens = torch::empty(
        {bsz, num_steps},
        torch::TensorOptions().dtype(torch::kLong).device(cur_ids.device())
    );
    auto batch_indices = torch::arange(
        bsz,
        torch::TensorOptions().dtype(torch::kLong).device(block_table.device())
    );
    auto cur_ids_col0 = cur_ids.select(1, 0);
    auto seq_lens_kv = seq_lens + 1;
    py::object cur_ids_obj = py::cast(cur_ids);
    py::object cos_obj = py::cast(cos);
    py::object sin_obj = py::cast(sin);
    py::object cur_pos_obj = py::cast(cur_pos);
    py::object block_table_obj = py::cast(block_table);
    py::object seq_lens_obj = py::cast(seq_lens);
    py::object seq_lens_kv_obj = py::cast(seq_lens_kv);
    py::object decode_blk_offsets_obj = py::cast(decode_blk_offsets);
    PyObject* decode_fns_list = decode_fns.ptr();
    PyObject* layer_kv_list = layer_kv_caches.ptr();
    PyObject* embed_callable = embed_fn.ptr();
    PyObject* decode_head_callable = decode_head_fn.ptr();

    py::object hidden;
    py::object final_logits = py::none();
    const bool use_fused_next_token = !decode_next_token_fn.is_none();
    PyObject* decode_next_callable = use_fused_next_token ? decode_next_token_fn.ptr() : nullptr;
    int64_t seq_len_host = 0;
    int64_t blk_host = 0;
    int64_t off_host = 0;
    if (single_seq) {
        seq_len_host = seq_lens.item<int64_t>();
        blk_host = blk_ids.item<int64_t>();
        off_host = decode_blk_offsets.item<int64_t>();
    }
    for (int64_t step = 0; step < num_steps; ++step) {
        PyObject* embed_args[1] = {cur_ids_obj.ptr()};
        hidden = vectorcall_positional(embed_callable, embed_args, 1);
        torch::Tensor hidden_t = hidden.cast<torch::Tensor>();
        if (embed_scale != 1.0) {
            hidden_t.mul_(embed_scale);
            hidden = py::cast(hidden_t);
        }

        torch::Tensor decode_phys_blocks;
        if (single_seq) {
            // Avoid extra tiny tensor ops for seq state update when B=1.
            decode_phys_blocks = block_table.index({0, blk_host}).view({1});
        } else {
            decode_phys_blocks = block_table.index({batch_indices, blk_ids});
        }
        py::object decode_phys_blocks_obj = py::cast(decode_phys_blocks);
        for (py::ssize_t i = 0; i < n_layers; ++i) {
            PyObject* decode_fn = PyList_GET_ITEM(decode_fns_list, i);
            PyObject* kv_cache = PyList_GET_ITEM(layer_kv_list, i);
            PyObject* args[10] = {
                hidden.ptr(),
                cos_obj.ptr(),
                sin_obj.ptr(),
                cur_pos_obj.ptr(),
                kv_cache,
                block_table_obj.ptr(),
                seq_lens_obj.ptr(),
                seq_lens_kv_obj.ptr(),
                decode_phys_blocks_obj.ptr(),
                decode_blk_offsets_obj.ptr(),
            };
            hidden = vectorcall_positional(decode_fn, args, 10);
        }

        torch::Tensor next_tokens;
        if (use_fused_next_token) {
            PyObject* token_args[1] = {hidden.ptr()};
            py::object tokens_obj = vectorcall_positional(decode_next_callable, token_args, 1);
            next_tokens = tokens_obj.cast<torch::Tensor>();
            final_logits = py::none();
        } else {
            PyObject* head_args[1] = {hidden.ptr()};
            py::object logits = vectorcall_positional(decode_head_callable, head_args, 1);
            torch::Tensor logits_t = logits.cast<torch::Tensor>();
            next_tokens = logits_t.index({Slice(), -1, Slice()}).argmax(-1);
            final_logits = logits;
        }
        all_tokens.select(1, step).copy_(next_tokens);
        cur_ids_col0.copy_(next_tokens);
        cur_pos.add_(1);

        if (single_seq) {
            seq_len_host += 1;
            off_host += 1;
            if (off_host == block_size) {
                off_host = 0;
                blk_host += 1;
            }
            seq_lens.fill_(seq_len_host);
            seq_lens_kv.fill_(seq_len_host + 1);
            decode_blk_offsets.fill_(off_host);
            blk_ids.fill_(blk_host);
        } else {
            seq_lens.add_(1);
            seq_lens_kv.add_(1);
            decode_blk_offsets.add_(1);
            auto wrapped = decode_blk_offsets.eq(block_size);
            blk_ids.add_(wrapped.to(blk_ids.dtype()));
            decode_blk_offsets.masked_fill_(wrapped, 0);
        }
    }

    return py::make_tuple(all_tokens, final_logits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MegaGemm decode-loop C++ helpers";
    m.def(
        "run_decode_layers_full_attention",
        &run_decode_layers_full_attention,
        py::arg("layers"),
        py::arg("layer_kv_caches"),
        py::arg("hidden"),
        py::arg("cos"),
        py::arg("sin"),
        py::arg("positions"),
        py::arg("block_table"),
        py::arg("seq_lens"),
        py::arg("seq_lens_kv"),
        py::arg("decode_phys_blocks"),
        py::arg("decode_blk_offsets"),
        py::arg("timing_events") = py::none()
    );
    m.def(
        "run_decode_fns_full_attention",
        &run_decode_fns_full_attention,
        py::arg("decode_fns"),
        py::arg("layer_kv_caches"),
        py::arg("hidden"),
        py::arg("cos"),
        py::arg("sin"),
        py::arg("positions"),
        py::arg("block_table"),
        py::arg("seq_lens"),
        py::arg("seq_lens_kv"),
        py::arg("decode_phys_blocks"),
        py::arg("decode_blk_offsets")
    );
    m.def(
        "run_decode_steps_full_attention",
        &run_decode_steps_full_attention,
        py::arg("decode_fns"),
        py::arg("layer_kv_caches"),
        py::arg("embed_fn"),
        py::arg("decode_head_fn"),
        py::arg("decode_next_token_fn") = py::none(),
        py::arg("cur_ids"),
        py::arg("cur_pos"),
        py::arg("block_table"),
        py::arg("seq_lens"),
        py::arg("blk_ids"),
        py::arg("decode_blk_offsets"),
        py::arg("block_size"),
        py::arg("cos"),
        py::arg("sin"),
        py::arg("num_steps"),
        py::arg("embed_scale")
    );
}

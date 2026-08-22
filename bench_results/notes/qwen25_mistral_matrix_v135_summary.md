# Qwen2.5 + Mistral CPU Same-Session Matrix v135

Source log: pasted Colab output from `qwen25_mistral_matrix_v135`.

Builds:

- MicroGemm suite: `qwen25_cpu_suite_profile_v135_parallel_rope_kv_batch`
- same-session wrapper: `qwen25_same_session_compare_v18_qwen25_presets`
- matrix wrapper: `qwen25_compare_matrix_v1_same_session_rollup`
- llama.cpp compare: `qwen25_llamacpp_batch_compare_v9_auto_gguf_quant`
- backend: `cpu-x86-avx2-fma`
- FMA: `1`

Run shape:

- CPU: Intel Xeon @ 2.20 GHz, 8 logical CPUs
- batch: 8
- prompt tokens for batch compare: 64
- ignore EOS: true
- runs/warmup: 3/1

## Results

| Case | Quant Pair | Max New | Micro Wall | llama.cpp Wall | Wall Ratio | Micro Decode | llama.cpp Decode | Decode Ratio | Micro Prefill | llama.cpp Prefill | Prefill Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5 0.5B | INT8 vs Q8_0 | 128 | 95.62 | 74.64 | 1.28x | 126.49 | 89.46 | 1.41x | 328.74 | 225.20 | 1.46x |
| Qwen2.5 1.5B | INT8 vs Q8_0 | 64 | 32.74 | 21.13 | 1.55x | 66.27 | 35.33 | 1.88x | 96.63 | 52.55 | 1.84x |
| Mistral 7B v0.3 | INT4 vs Q4_K_M | 32 | 5.01 | 3.71 | 1.35x | 14.69 | 8.64 | 1.70x | 16.98 | 12.93 | 1.31x |

## MicroGemm Profile Hot Spots

| Case | rope_kv ms | gate_up_dot ms | down_proj_dot ms | Main Bottleneck |
|---|---:|---:|---:|---|
| Qwen2.5 0.5B | 203 | 2354 | 1025 | `gate_up_dot` |
| Qwen2.5 1.5B | 89 | 3189 | 1563 | `gate_up_dot` |
| Mistral 7B v0.3 | 227 | 7736 | 4005 | `gate_up_dot`, then `down_proj_dot` |

## Notes

- All three comparisons are same-session paired, so the MicroGemm/llama.cpp ratios are much more trustworthy than cross-Colab comparisons.
- Qwen2.5 0.5B and 1.5B both show MicroGemm ahead in wall/output, decode-only, and prefill.
- Mistral 7B INT4 also shows a solid win against llama.cpp Q4_K_M, especially in decode.
- The v135 `parallel_rope_kv_batch` patch does not matter much for these three dense models at this context length: `rope_kv` is now small. The next meaningful CPU optimization target is the quantized MLP dot path: `gate_up_dot`, followed by `down_proj_dot`.
- The later cases were run after llama.cpp build/download activity and show high loadavg at case start. Because each case is still paired in the same session, the ratios remain useful, but repeat the matrix before publishing hard claims.

# Qwen2.5 0.5B INT8 Signed Fix Same-Session Result v184

Source log: pasted Colab output from `qwen25_05b_int8_v184_vs_llamacpp_q8_same_session`.

Builds:

- MicroGemm suite: `qwen25_cpu_suite_profile_v184_i8_batched_signed_fix`
- same-session wrapper: `qwen25_same_session_compare_v19_groupwise_quant`
- llama.cpp compare: `qwen25_llamacpp_batch_compare_v9_auto_gguf_quant`
- backend: `cpu-x86-avx2-fma`
- C selftest: `ok (kernel-selftest)`

Run shape:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- MicroGemm quant: `int8`
- llama.cpp GGUF: `Q8_0`
- CPU: AMD EPYC 7B12, 8 logical CPUs
- batch: 8
- batch prompt tokens: 64
- max new tokens: 128
- runs/warmup: 3/1
- ignore EOS: true

## Result

| Metric | MicroGemm INT8 | llama.cpp Q8_0 | Micro/llama |
|---|---:|---:|---:|
| wall/output tok/s | 154.15 | 110.49 | 1.40x |
| decode-only tok/s | 199.33 | 131.94 | 1.51x |
| prefill tok/s | 574.04 | 339.86 | 1.69x |

## MicroGemm Profile

| Profile Field | Value |
|---|---:|
| single prompt=64 prefill | 489.80 tok/s |
| single prompt=64 decode | 89.75 tok/s |
| batch runtime | 165.63 tok/s |
| batch steady | 165.33 tok/s |
| batch total wall | 244.48 tok/s |
| overhead | 451 ms |
| engine | 6194 ms |
| gate_up | 1402 ms |
| gate_up_dot | 1386 ms |
| down_proj | 866 ms |
| down_proj_dot | 646 ms |
| rope | 170 ms |

## Notes

- This run is the first same-session Qwen2.5 0.5B INT8 vs llama.cpp Q8_0 comparison after the v184 batched INT8 signed-input fix.
- INT8 per-row comparison results before v184 are invalidated by a path where `microgemm_gemv_i8_batched` could feed prebiased activations into kernels that add `+128` internally.
- The v184 run is a stronger result than the older v135 Qwen2.5 0.5B INT8 vs Q8_0 note: wall ratio improved from 1.28x to 1.40x, decode from 1.41x to 1.51x, and prefill from 1.46x to 1.69x.
- The profile is concentrated in the quantized MLP path: `gate_up_dot` is the largest recorded hotspot, followed by `down_proj_dot`.

## INT8G128 vs llama.cpp Q8_0 Same-Session Check

Source log: pasted Colab output from `qwen25_05b_int8g_v184_vs_llamacpp_q8_same_session`.

Run shape is the same as above except MicroGemm quant is `int8g128`.

| Metric | MicroGemm INT8G128 | llama.cpp Q8_0 | Micro/llama |
|---|---:|---:|---:|
| wall/output tok/s | 115.80 | 115.94 | 1.00x |
| decode-only tok/s | 157.46 | 149.44 | 1.05x |
| prefill tok/s | 321.69 | 345.44 | 0.93x |

INT8G128 is therefore a speed tie with llama.cpp Q8_0 on wall/output in this paired run: slightly faster in decode, slightly slower in prefill. Compared with the same v184 INT8 per-row result above, INT8G128 is meaningfully slower, which is expected from group metadata and per-group scale work. Its reason to exist is quality preservation versus per-row INT8, not raw TPS.

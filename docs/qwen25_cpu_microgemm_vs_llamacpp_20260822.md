# Qwen 2.5 0.5B on Intel Xeon: MicroGemm versus llama.cpp

This report records the same-session CPU comparison completed on 2026-08-22
with run ID `qwen25_05b_int8_vs_q8_20260822`. It compares the current
MicroGemm CPU suite build with llama.cpp at commit
`b21e4de74567f5eef213765c9476a843c2e43f0d`.

## Result summary

Across batches 1, 2, 4, and 8, MicroGemm reached **1.286x llama.cpp's
geometric-mean internal engine throughput**, **1.285x decode throughput**, and
**1.365x prefill throughput**. The result is strongly batch-dependent:

- batches 1-2: 1.003x engine and 0.995x decode geometric mean — parity;
- batches 4-8: 1.649x engine, 1.661x decode, and 1.465x prefill geometric mean.

| Batch | Micro wall | Micro engine | llama.cpp engine | Engine ratio | Micro decode | llama.cpp decode | Decode ratio | Micro prefill | llama.cpp prefill | Prefill ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 34.83 | 45.11 | 43.49 | 1.04x | 50.80 | 49.30 | 1.03x | 235.94 | 188.33 | 1.25x |
| 2 | 47.22 | 56.20 | 57.94 | 0.97x | 64.21 | 66.88 | 0.96x | 263.80 | 204.04 | 1.29x |
| 4 | 76.41 | 87.82 | 51.93 | **1.69x** | 110.25 | 69.78 | **1.58x** | 306.18 | 215.88 | **1.42x** |
| 8 | 100.68 | 110.62 | 68.82 | **1.61x** | 144.44 | 82.71 | **1.75x** | 328.34 | 216.97 | **1.51x** |

All throughput values are tokens/second. Ratios are MicroGemm divided by
llama.cpp and were recomputed from the displayed medians.

## Conditions

- CPU: Intel Xeon at 2.20 GHz, family 6/model 79, under KVM
- Topology visible to the process: 4 physical cores, 8 logical CPUs, one socket
- ISA: AVX2 and FMA
- Cache: 128 KiB aggregate L1d, 128 KiB aggregate L1i, 1 MiB aggregate L2,
  55 MiB shared L3 as reported by `lscpu`
- CPU frequency snapshot: 2,200 MHz minimum/median/maximum
- Initial load average: 0.61, 0.35, 0.15
- cgroup CPU setting: `max 100000`, with no quota reported
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- MicroGemm format: per-row INT8
- llama.cpp format: Q8_0 GGUF
- MicroGemm suite: `qwen25_cpu_suite_profile_v229_glm4_i8g_canary_recal`
- MicroGemm backend: `cpu-x86-avx2-fma`; C kernel self-test passed
- llama.cpp commit: `b21e4de74567f5eef213765c9476a843c2e43f0d`
- Requested prompt: 64 tokens per request
- Output: 128 tokens per request, EOS ignored
- Batch sizes: 1, 2, 4, and 8
- Threads: 8 for both backends
- Warmups: 2
- Measured repetitions: 5; all configurations completed 5/5
- Order: MicroGemm first, then llama.cpp, in the same Colab session
- llama.cpp GPU offload: disabled by the CPU harness (`-ngl 0` when supported)

## Interpreting the metrics

The primary engine ratio compares MicroGemm's `runtime_output_tps` with
llama.cpp batched-bench's `output_tps_total`. Both cover prefill plus generated
tokens while excluding model startup from the timed engine interval. Decode and
prefill are also compared like-for-like using the backends' internal timings.

MicroGemm's `wall_output_tps` includes process startup, tokenizer work, model
open/load, cleanup, and Python harness overhead. llama.cpp's reported
`output_tps_total` is an internal benchmark interval and does not include an
equivalent external process wall measurement. Their ratio therefore must not be
called a symmetric wall-versus-wall comparison. Even under this deliberately
conservative mismatch, MicroGemm reached 1.089x the four-batch geometric mean;
it lost at batches 1-2 and won at batches 4-8.

## Scaling behavior

The current result supports a specific architectural claim: MicroGemm's native
continuous batching scales substantially better in this workload once four or
more requests are active. At batch 8 it reached 144.44 decode tok/s versus
82.71 for llama.cpp, a 1.75x ratio. Its batch-8 prefill ratio was 1.51x.

The MicroGemm profile still identifies the quantized MLP as the main
bottleneck. At batch 8, `gate_up_dot` consumed about 2,386 ms and
`down_proj_dot` about 1,145 ms, while RoPE/KV work was about 121 ms. The next
optimization target remains the quantized gate/up and down-projection dot path.

## Evidence and limitations

The supplied console log contains the complete invocation, medians, CPU
snapshot, build tags, artifact paths, and llama.cpp commit. The generated
JSON/CSV files are still on the Colab Drive and are not yet present in this
repository, so this is currently console-log evidence rather than a
self-contained publication archive. Per-repeat variance cannot be audited from
the console summary alone.

The MicroGemm canary classified the single-request decode result as `slow`:
50.01 tok/s versus its profile's 75 tok/s weak and 85 tok/s good thresholds.
The paired ratios remain useful because llama.cpp ran on the same allocation,
but the absolute MicroGemm result may be below its expected performance and
should be repeated on another allocation.

The formats are both 8-bit but are not quality-identical: MicroGemm uses its
per-row INT8 layout while llama.cpp uses blockwise Q8_0. No perplexity,
token-equivalence, or quality test accompanied this throughput run.

The requested prompt length was 64 tokens. MicroGemm's text harness appends a
per-request index before tokenization, whereas llama-batched-bench uses its
synthetic 64-token workload. The raw MicroGemm summary records the actual token
count and must be checked before making an exact token-identical prefill claim.
Until then, treat the prefill result as a closely matched requested-length
comparison.

The run order was fixed rather than alternated, and the Colab source copy had
no immutable MegaGemm Git commit. A release-grade rerun should alternate
backend order, preserve both source commits, hash the converted model files,
and save raw per-repeat results.

## Narrow publication claim

On one 8-thread Intel Xeon Colab allocation, for Qwen 2.5 0.5B with MicroGemm
INT8 versus llama.cpp Q8_0, 64 requested prompt tokens and 128 fixed generated
tokens, MicroGemm matched llama.cpp's internal engine/decode throughput at
batches 1-2 and led substantially at batches 4-8. Across all four batches it
reached 1.286x geometric-mean engine throughput; at batch 8 it reached 1.61x
engine, 1.75x decode, and 1.51x prefill throughput.

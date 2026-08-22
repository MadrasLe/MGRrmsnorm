# Benchmark evidence and publication policy

MegaGemm has extensive benchmark infrastructure and dated result summaries.
This page is the canonical index for claims suitable for the project front
page. Results are project-run measurements, not independent third-party
validation.

> **Current status (August 2026):** Qwen 2.5 3B FP16 now has a current compact
> same-session L4 rerun against vLLM 0.27.1. The May 2026 T4 and Qwen 3.5
> results remain historical development evidence until they receive equivalent
> reruns.

## Claim policy

Performance claims must name:

- backend and quantization format;
- model and checkpoint;
- hardware and visible memory;
- prompt length, output length, and batch size;
- dtype and EOS policy;
- warmup/repeat count and aggregation method;
- PyTorch, CUDA, Triton, Transformers, and comparison-backend versions;
- whether prefix caching, CUDA Graphs, or state restore was active;
- whether results came from the same process/session.

Use “parity” only for the measured configuration. Do not generalize a selected
row to every model, batch size, or context length.

## GPU: selected vLLM comparisons

### Qwen 2.5 3B on one NVIDIA L4 (current compact matrix)

Hardware and stack: NVIDIA L4, Qwen 2.5 3B Instruct, FP16, 128 generated
tokens, fixed-length decode, vLLM prefix caching disabled, one warmup and three
measured repetitions. Both backends ran from the same Python environment with
PyTorch 2.13.0+cu129, Triton 3.7.1, Transformers 5.15.1, and vLLM
0.27.1+cu129.

| Batch | Prompt | MegaGemm FP16 | vLLM FP16 | MegaGemm / vLLM | Interpretation |
|---:|---:|---:|---:|---:|---|
| 1 | 128 | 37.82 | 38.81 | 97.4% | Near parity; vLLM leads by 2.6% |
| 1 | 512 | 37.39 | 38.50 | 97.1% | Near parity; vLLM leads by 2.9% |
| 1 | 2048 | 33.68 | 36.25 | 92.9% | vLLM leads by 7.1% |
| 8 | 128 | 281.20 | 294.65 | 95.4% | vLLM leads by 4.6% |
| 8 | 512 | 244.32 | 266.24 | 91.8% | vLLM leads by 8.2% |
| 8 | 2048 | 155.01 | 183.78 | 84.3% | Largest measured gap: 15.7% |

MegaGemm reached **93.1% of vLLM's unweighted geometric-mean throughput**
across these six matched rows. vLLM won every row; MegaGemm stayed within 10%
in five of six and within 3% for the two batch-1 prompts up to 512 requested
tokens.

Source: [`qwen25_l4_results_20260822.md`](qwen25_l4_results_20260822.md), with
the complete raw archive at
[`publication_20260822_145527.zip`](publication_20260822_145527.zip).

Caveats: this is a compact three-repeat matrix, not the final full sweep, and
the Colab copy had no Git metadata, so the recorded commit is `null`. The
result is current project-run evidence, not independent third-party
validation. A separate MGX+Prophet run studies exact-prefix cache hits and must
not be mixed into this uncached table.

### Qwen 2.5 3B MGX+Prophet cache reuse on one NVIDIA L4

This same-session matrix compared warm exact-prefix reuse on both sides: MGX
with Prophet exact restore/live resident cache, and vLLM 0.27.1 with prefix
caching enabled. The output length was 32 tokens, with one warmup and five
measured repetitions. Values are the suite-reported cached medians.

| Batch | Prompt | MGX+Prophet | vLLM prefix cache | Prophet / vLLM | Interpretation |
|---:|---:|---:|---:|---:|---|
| 1 | 512 | 38.44 | 38.60 | 99.6% | vLLM leads by 0.4% |
| 8 | 512 | 296.67 | 286.83 | 103.4% | Prophet leads by 3.4% |
| 1 | 2048 | 38.56 | 37.96 | 101.6% | Prophet leads by 1.6% |
| 8 | 2048 | 268.82 | 251.76 | 106.8% | Prophet leads by 6.8% |

MGX+Prophet won three of four rows and reached **102.8% of vLLM's unweighted
geometric-mean cached throughput**, a 2.8% lead in this measured matrix.

Source: [`qwen25_l4_prophet_results_20260822.md`](qwen25_l4_prophet_results_20260822.md).

Caveats: repeated prompts exercise exact-prefix reuse rather than semantic
near-match reuse. Prophet used `validation-mode none`, and the generated raw
JSONL/JSON/CSV files have not yet been copied from Colab. This is console-log
evidence and must remain separate from the uncached FP16 result.

### Historical GPU runs

Older T4, Hugging Face, quantized, and Qwen 3.5 measurements guided runtime
development, but their raw reports are not included in the current curated
publication bundle. They must be restored with their evidence or rerun before
promotion. The August uncached L4 matrix and the separate Prophet cache-reuse
matrix above are the current GPU results.

## CPU: MicroGemm versus llama.cpp

CPU comparisons refer to the standalone `microgemm/` runtime, not the generic
PyTorch CPU path in `InferenceEngine`.

### Current same-session Intel Xeon matrix

Hardware: Intel Xeon 2.20 GHz, 4 cores/8 logical CPUs, AVX2/FMA. Workload:
Qwen 2.5 0.5B, MicroGemm INT8 versus llama.cpp Q8_0, 64 requested prompt
tokens, 128 fixed generated tokens, two warmups and five repetitions.

| Batch | Micro engine | llama.cpp engine | Engine ratio | Decode ratio | Prefill ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 45.11 | 43.49 | 1.04x | 1.03x | 1.25x |
| 2 | 56.20 | 57.94 | 0.97x | 0.96x | 1.29x |
| 4 | 87.82 | 51.93 | 1.69x | 1.58x | 1.42x |
| 8 | 110.62 | 68.82 | 1.61x | 1.75x | 1.51x |

Across all four batches, MicroGemm reached **1.286x unweighted geometric-mean
engine throughput**, 1.285x decode, and 1.365x prefill. Batches 1-2 were at
engine/decode parity; across batches 4-8, MicroGemm reached 1.649x engine and
1.661x decode geometric-mean throughput.

Source: [`qwen25_cpu_microgemm_vs_llamacpp_20260822.md`](qwen25_cpu_microgemm_vs_llamacpp_20260822.md).

Caveats: the formats are not quality-identical, the MicroGemm canary reported a
slow absolute single-request result, and raw JSON/CSV artifacts have not yet
been copied from Colab. The 64-token prompt was requested on both sides, but
the MicroGemm text harness appends a request index; check its actual token count
in the raw CSV before making an exact token-identical prefill claim.

### Historical CPU evidence

The post-fix AMD EPYC v184 run recorded MicroGemm INT8/Q8_0 ratios of 1.40x
wall/output, 1.51x decode, and 1.69x prefill at batch 8. Its INT8G128/Q8_0
variant reached 1.00x wall/output, 1.05x decode, and 0.93x prefill. Source:
[`../bench_results/notes/qwen25_05b_v184_int8_signed_fix_summary.md`](../bench_results/notes/qwen25_05b_v184_int8_signed_fix_summary.md).

The older v135 Intel Qwen INT8 rows predate the signed-input fix and must not be
promoted. Its Mistral INT4/Q4_K_M row was not affected by that INT8 bug but was
recorded after elevated system load. Source:
[`../bench_results/notes/qwen25_mistral_matrix_v135_summary.md`](../bench_results/notes/qwen25_mistral_matrix_v135_summary.md).

## Reproduction entry points

- `benchmarks/benchmark_inference_matrix.py` — common inference matrix runner.
- `benchmarks/run_publication_gpu_suite.py` — compact same-environment GPU
  publication runner used by the current L4 result.
- `benchmarks/run_qwen25_mgx_prophet_vs_vllm_l4.py` — same-session MGX+Prophet
  versus vLLM prefix-cache runner.
- `benchmarks/run_qwen_vllm_t4_suite.py` — Qwen/vLLM T4 suite.
- `benchmarks/run_qwen_vllm_l4_suite.py` — Qwen/vLLM L4 suite.
- `benchmarks/run_gemma4_long_context_vs_vllm.py` — long-context Gemma comparison.
- `microgemm/tools/qwen25_compare_matrix.py` — MicroGemm CPU comparison matrix.
- `microgemm/tools/qwen25_same_session_compare.py` — paired same-session CPU
  runner used by the current Qwen 2.5 0.5B result.
- `microgemm/tools/qwen25_llamacpp_batch_compare.py` — llama.cpp paired runner.

## Before a public performance release

1. Rerun the headline GPU rows in fresh, isolated environments.
2. Disable prefix caching unless cache behavior is the subject of the test.
3. Save raw JSON/CSV, console logs, `nvidia-smi`, CPU model, thread affinity,
   package versions, commit hash, and exact command.
4. Run correctness/quality checks for every quantized configuration.
5. Separate TTFT, prefill TPS, decode TPS, end-to-end output TPS, and memory.
6. Publish failures and OOMs alongside successful rows.

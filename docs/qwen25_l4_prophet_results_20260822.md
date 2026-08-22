# Qwen 2.5 3B on NVIDIA L4: MGX+Prophet versus vLLM prefix cache

This report records the same-session cache-reuse matrix completed on
2026-08-22 with run ID
`qwen25_prophet_vs_vllm_l4_20260822_152646`. It is a separate experiment from
the uncached FP16 matrix: both sides used their cache-reuse path, and the
comparison below uses each suite's reported cached median.

## Result summary

MGX+Prophet won three of four measured configurations and reached **102.8% of
vLLM's unweighted geometric-mean cached throughput**, a **2.8% lead** over this
four-row matrix.

| Batch | Requested prompt tokens | MGX+Prophet | vLLM prefix cache | Prophet / vLLM | Result |
|---:|---:|---:|---:|---:|---|
| 1 | 512 | 38.44 tok/s | 38.60 tok/s | 99.6% | vLLM by 0.4% |
| 8 | 512 | 296.67 tok/s | 286.83 tok/s | 103.4% | Prophet by 3.4% |
| 1 | 2048 | 38.56 tok/s | 37.96 tok/s | 101.6% | Prophet by 1.6% |
| 8 | 2048 | 268.82 tok/s | 251.76 tok/s | 106.8% | Prophet by 6.8% |

The aggregate was recomputed as the geometric mean of the four
Prophet/vLLM cached-throughput ratios. It does not include export or engine
startup time.

## Conditions

- GPU: 1x NVIDIA L4 with 22.03 GiB visible to the benchmark process
- Model: `Qwen/Qwen2.5-3B-Instruct`
- MegaGemm input: FP16 MGX artifact with embedded payload cache
- vLLM: 0.27.1 on PyTorch 2.13.0+cu129
- Batch sizes: 1 and 8
- Requested prompt lengths: 512 and 2048 tokens
- Output length: 32 tokens per request
- EOS policy: ignored, producing a fixed decode length
- Maximum sequence length: 4096
- Warmup: 1
- Repetitions: 5
- Comparison statistic: suite-reported `cached_median`
- Prophet: batch exact restore, live prefix cache, resident cache, primed before
  measurement, maximum 16 resident entries
- vLLM: prefix caching enabled; FlashAttention 2, compilation, and CUDA Graphs
  enabled by backend configuration
- MegaGemm decode CUDA Graphs: reported off for this run
- Execution: both backends ran during the same Colab L4 session

## What was measured

The prompts were repeated, so this matrix measures **warm exact-prefix cache
reuse**. Prophet supports prefix and semantic lookup modes, but this result is
not evidence for semantically similar, non-identical prompts.

The first measured samples show why the cached median is the relevant statistic
for this experiment. For example, at batch 8 / prompt 2048, vLLM's first sample
was 116.54 tok/s and its cached median was 251.76 tok/s; Prophet's first sample
was 267.24 tok/s and its cached median was 268.82 tok/s. At batch 1 / prompt
2048, Prophet's first sample was 19.22 tok/s before reaching a cached median of
38.56 tok/s. Cold-fill and shape effects must not be mixed with the steady
cache-hit comparison.

## Evidence and limitations

The result was recovered from the complete console log supplied for the run.
The log records all five samples, both commands, backend versions, cache
configuration, generated-token counts, output artifact paths, and the final
same-session comparison. The generated JSONL/JSON/CSV artifacts named in the
log are not yet present in this repository, so this report is **console-log
evidence**, not yet a self-contained raw publication bundle.

Prophet was run with `--prophet-validation-mode none`. The harness verified
successful fixed-length generation and timing, but did not compare output tokens
against vLLM or validate semantic-reuse correctness. Therefore the result
supports a performance claim about this exact-prefix cache-hit workload, not an
output-equivalence or semantic-cache-quality claim.

Before treating this as release-grade evidence, copy or export the following
Colab outputs into a dated archive:

- Prophet raw JSONL, summary JSON, and summary CSV;
- vLLM raw JSONL, summary JSON, and summary CSV;
- `qwen25_prophet_vs_vllm_l4_20260822_152646_prophet_vs_vllm.csv`;
- both generated Markdown reports;
- environment manifest and an immutable Git commit;
- a correctness run with Prophet validation enabled.

## Narrow publication claim

On one NVIDIA L4 with Qwen 2.5 3B FP16, 32-token fixed decode, and repeated
512/2048-token prompts, MGX+Prophet's warm exact-prefix reuse reached 102.8% of
vLLM 0.27.1 prefix caching's geometric-mean throughput and won three of four
measured batch/context configurations. The largest measured lead was 6.8% at
batch 8 / prompt 2048; vLLM led by 0.4% at batch 1 / prompt 512.

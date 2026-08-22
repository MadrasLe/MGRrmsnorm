# MegaMesh

MegaMesh is the experimental distributed inference layer for MegaGemm.

It has two intended modes:

1. **Replica mode**: every worker loads the full model. The router sends each
   prompt or prompt batch to one worker. This is the right first mode for pods
   connected over normal network links because only prompts and generated text
   cross the wire.
2. **Layer-shard mode**: each worker owns a layer range and passes hidden states
   to the next stage. This is useful when one model does not fit on one GPU, but
   it is much more sensitive to network latency. It should cross the network only
   at coarse stage boundaries, not after every individual layer.

## Replica Quickstart

Start one worker per pod:

```bash
python -m megagemm mesh-worker \
  --model Qwen/Qwen3.5-4B \
  --device cuda \
  --host 0.0.0.0 \
  --port 8088 \
  --name t4-0 \
  --weight 1 \
  --max-seq-len 256 \
  --num-blocks 1024
```

For a faster worker, use a larger weight:

```bash
python -m megagemm mesh-worker \
  --model Qwen/Qwen3.5-4B \
  --device cuda \
  --host 0.0.0.0 \
  --port 8088 \
  --name l4-0 \
  --weight 2 \
  --max-seq-len 256 \
  --num-blocks 1024
```

Query workers:

```bash
python -m megagemm mesh-health \
  --workers http://10.0.0.10:8088@1#t4,http://10.0.0.11:8088@2#l4
```

Generate through the router:

```bash
python -m megagemm mesh-generate \
  --workers http://10.0.0.10:8088@1#t4,http://10.0.0.11:8088@2#l4 \
  --max-tokens 128 \
  "Explain Qwen 3.5 linear attention."
```

Benchmark aggregate throughput:

```bash
python benchmarks/benchmark_megamesh.py \
  --workers http://10.0.0.10:8088@1#t4,http://10.0.0.11:8088@2#l4 \
  --batch-sizes 32,64,96,128 \
  --max-new-tokens 128 \
  --warmup 1 \
  --runs 3 \
  --out megamesh_replica.json
```

## Experimental Layer-Shard Quickstart

Layer-shard mode is now available as an isolated proof-of-concept. It does not
touch `InferenceEngine`, the normal scheduler, or the replica worker path.

Use this when a model does not fit on one GPU and you want to prove that two
separate T4 workers can own different layer ranges:

```bash
MODEL="Qwen/Qwen2.5-14B-Instruct"
LAYERS=48

python -m megagemm mesh-plan \
  --num-layers "$LAYERS" \
  --workers http://127.0.0.1:8090@1#stage0,http://127.0.0.1:8091@1#stage1 \
  --devices cuda:0,cuda:1
```

Start one process per GPU. On one Kaggle machine you can bind explicitly:

```bash
python -u -m megagemm mesh-shard-worker \
  --model "$MODEL" \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8090 \
  --ttp-port 9090 \
  --name stage0 \
  --layer-start 0 \
  --layer-end 24 \
  --first-stage \
  --max-seq-len 512 \
  --num-blocks 256
```

```bash
python -u -m megagemm mesh-shard-worker \
  --model "$MODEL" \
  --device cuda:1 \
  --host 127.0.0.1 \
  --port 8091 \
  --ttp-port 9091 \
  --name stage1 \
  --layer-start 24 \
  --layer-end 48 \
  --last-stage \
  --max-seq-len 512 \
  --num-blocks 256
```

If you prefer to treat each process as if it had only one GPU, use
`CUDA_VISIBLE_DEVICES=0` for the first process and `CUDA_VISIBLE_DEVICES=1` for
the second, then pass `--device cuda` to both.

Generate through the ordered stages:

```bash
python -m megagemm mesh-shard-generate \
  --model "$MODEL" \
  --stages ttp://127.0.0.1:9090#stage0,ttp://127.0.0.1:9091#stage1 \
  --transport ttp \
  --max-tokens 32 \
  --health \
  --json \
  "Explique em uma frase o que e pipeline parallelism."
```

To avoid hard-coding layer counts, check the model config first:

```bash
python - <<'PY'
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download("Qwen/Qwen2.5-14B-Instruct", "config.json")
cfg = json.load(open(path, "r", encoding="utf-8"))
cfg = cfg.get("text_config", cfg)
print(cfg["num_hidden_layers"])
PY
```

The fastest portable shard transport is TTP:

```text
persistent TCP connection + length-prefixed binary tensor frames
```

Start workers with `--ttp-port` and generate with `--transport ttp` plus
`ttp://host:port` stage URLs.

TTP also uses a small pinned-memory transmit pool for CUDA tensors. Hidden
states leaving a shard are copied into reusable CPU pinned buffers with
`non_blocking=True`, then streamed as frame parts without concatenating the
whole tensor payload into one extra `bytes` object. Health checks expose the pool
under `ttp_out_pool`. Use `--ttp-no-pinned` to keep TTP persistent sockets but
disable pinned outbound buffers for A/B debugging.

The fallback shard transport is binary HTTP tensor frames:

```text
magic + small JSON header + raw tensor bytes
```

This removes base64 expansion and large tensor JSON parsing. It still copies
through CPU memory and is not the final GPU-direct data plane, but it is much
faster and remains portable across Kaggle, Colab, RunPod, or local machines.
Use `--transport binary` for HTTP binary fallback and `--transport json` only as
a compatibility fallback.

For throughput tests, use decode microbatching. Prefill is still run per prompt,
but decode steps are grouped across active sequences so each shard executes one
batched layer pass per microbatch:

```bash
python -m megagemm mesh-shard-generate-batch \
  --model "$MODEL" \
  --stages ttp://127.0.0.1:9090#stage0,ttp://127.0.0.1:9091#stage1 \
  --transport ttp \
  --microbatch-size 8 \
  --max-tokens 32 \
  --disable-thinking \
  --json \
  "Explique MegaMesh em uma frase." \
  "Explique pipeline parallelism em uma frase." \
  "Explique KV cache em uma frase." \
  "Explique tensor parallelism em uma frase."
```

For queue-style sustained serving tests, use MegaMesh shard continuous batching:

```bash
python -m megagemm mesh-shard-generate-continuous \
  --model "$MODEL" \
  --stages ttp://127.0.0.1:9090#stage0,ttp://127.0.0.1:9091#stage1 \
  --transport ttp \
  --max-batch-size 8 \
  --microbatch-size 8 \
  --max-tokens 32 \
  --disable-thinking \
  --json \
  --prompts-file prompts.txt
```

This mode sends one queue to the first TTP stage. Stage 0 owns the scheduler
loop: requests are admitted into a live running set up to `--max-batch-size`,
finished requests leave the set, and waiting requests are admitted into freed
slots. Decode still runs through the ordered shard chain, and the result reports:

```json
"pipeline": {
  "continuous_batching": true,
  "admission_events": 2,
  "max_running": 8
}
```

This is not a public HTTP serving loop yet. It is the isolated MegaMesh shard
runtime primitive for continuous batching over TTP.

### Replicated shard pipelines

For serving-style throughput tests, MegaMesh can route prompts across multiple
independent layer-shard pipelines. Each replica is still an ordered shard chain,
but replicas do not share KV cache or layer state with each other. This is useful
for "two 14B pipelines on four L4s": one 2-GPU shard chain handles replica A and
another 2-GPU shard chain handles replica B.

```bash
python -m megagemm mesh-shard-generate-replicas \
  --model "$MODEL" \
  --replicas "ttp://127.0.0.1:9090#a0,ttp://127.0.0.1:9091#a1;ttp://127.0.0.1:9092#b0,ttp://127.0.0.1:9093#b1" \
  --transport ttp \
  --strategy round_robin \
  --microbatch-size 4 \
  --max-tokens 32 \
  --disable-thinking \
  --json \
  --prompts-file prompts.txt
```

The `--replicas` value is semicolon-separated. Inside each replica, stages are
comma-separated and ordered exactly like `--stages` in
`mesh-shard-generate-batch`. The router runs replicas concurrently and restores
the original prompt order in the final output. This improves aggregate
throughput for multiple prompts/users, but a single prompt still uses one
replica.

### Experimental vocab-sharded `lm_head`

MegaMesh can also split the final vocabulary projection away from the last
layer shard. This is the first intra-layer sharding primitive: the last layer
stage keeps the final RMSNorm, skips local `lm_head.weight`, sends the final
hidden state to one or more TTP `lm_head` workers, and reduces each worker's
local argmax into the global greedy token.

This mode is opt-in and does not change normal layer-shard workers unless
`--lm-head-shards` is passed to the last stage.

Discover the vocab size and start two vocab shards:

```bash
VOCAB=$(python - "$MODEL" <<'PY'
import json
import os
import sys

model = sys.argv[1]
with open(os.path.join(model, "config.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg = cfg.get("text_config", cfg)
print(int(cfg["vocab_size"]))
PY
)
VOCAB_SPLIT=$((VOCAB / 2))
```

```bash
python -u -m megagemm mesh-lm-head-worker \
  --model "$MODEL" \
  --device cuda \
  --host 127.0.0.1 \
  --port 8190 \
  --ttp-port 9190 \
  --name head0 \
  --vocab-start 0 \
  --vocab-end "$VOCAB_SPLIT"
```

```bash
python -u -m megagemm mesh-lm-head-worker \
  --model "$MODEL" \
  --device cuda \
  --host 127.0.0.1 \
  --port 8191 \
  --ttp-port 9191 \
  --name head1 \
  --vocab-start "$VOCAB_SPLIT" \
  --vocab-end "$VOCAB"
```

Then attach them to the last layer stage:

```bash
python -u -m megagemm mesh-shard-worker \
  --model "$MODEL" \
  --device cuda \
  --host 127.0.0.1 \
  --port 8093 \
  --ttp-port 9093 \
  --name stage3 \
  --layer-start 50 \
  --layer-end 64 \
  --last-stage \
  --lm-head-shards ttp://127.0.0.1:9190#head0,ttp://127.0.0.1:9191#head1 \
  --max-seq-len 512 \
  --num-blocks 256
```

The final stage health reports:

```json
"lm_head": {
  "mode": "remote-sharded",
  "skip_local_lm_head": true
}
```

This is not a full tensor-parallel transformer block yet. It is a deliberately
small proof that MegaMesh can isolate a sub-layer owner with its own weights,
device, transport, and failure boundary. The next harder splits are MLP row/col
shards and attention-head shards, because those need distributed reductions
inside every transformer layer rather than one final argmax reduction.

### Experimental MLP intermediate sharding

MegaMesh also has an opt-in MLP shard primitive. A layer stage can skip local
MLP weights and attach TTP workers that own slices of the FFN intermediate
dimension. Each MLP worker loads:

```text
gate_proj[intermediate_start:intermediate_end, :]
up_proj[intermediate_start:intermediate_end, :]
down_proj[:, intermediate_start:intermediate_end]
```

At runtime the layer stage keeps attention, KV cache, norms, and residuals
local. It sends normalized MLP input to the MLP shards, sums their partial
`down_proj` outputs, and applies the residual locally. This is a real
intra-block split, but it adds a reduction inside every transformer layer, so it
is much more network-sensitive than vocab-sharded `lm_head`.

Start two MLP shards for the same layer range as a stage:

```bash
INTERMEDIATE=$(python - "$MODEL" <<'PY'
import json
import os
import sys

model = sys.argv[1]
with open(os.path.join(model, "config.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg = cfg.get("text_config", cfg)
print(int(cfg["intermediate_size"]))
PY
)
ISPLIT=$((INTERMEDIATE / 2))
```

```bash
python -u -m megagemm mesh-mlp-worker \
  --model "$MODEL" \
  --device cuda \
  --host 127.0.0.1 \
  --port 8290 \
  --ttp-port 9290 \
  --name mlp0 \
  --layer-start 48 \
  --layer-end 64 \
  --intermediate-start 0 \
  --intermediate-end "$ISPLIT"
```

```bash
python -u -m megagemm mesh-mlp-worker \
  --model "$MODEL" \
  --device cuda \
  --host 127.0.0.1 \
  --port 8291 \
  --ttp-port 9291 \
  --name mlp1 \
  --layer-start 48 \
  --layer-end 64 \
  --intermediate-start "$ISPLIT" \
  --intermediate-end "$INTERMEDIATE"
```

Attach them to the layer stage that owns the same layer range:

```bash
python -u -m megagemm mesh-shard-worker \
  --model "$MODEL" \
  --device cuda \
  --host 127.0.0.1 \
  --port 8093 \
  --ttp-port 9093 \
  --name stage3 \
  --layer-start 48 \
  --layer-end 64 \
  --last-stage \
  --mlp-shards ttp://127.0.0.1:9290#mlp0,ttp://127.0.0.1:9291#mlp1 \
  --max-seq-len 512 \
  --num-blocks 256
```

Health reports:

```json
"mlp": {
  "mode": "remote-sharded",
  "skip_local_mlp": true
}
```

The current implementation is intentionally conservative: full-attention
layers only, FP16/BF16/FP32 weights, greedy generation, and the local C++ decode
loop is disabled for stages using remote MLP because the stage now needs a TTP
callback inside each layer.

Layer-shard decode now has an isolated fastpath. If the local stage contains
only full-attention layers and the compiled `megagemm_decode_ops` extension is
available, the shard uses the same C++ layer loop as the normal engine for
decode and `decode_batch`. The fallback remains local to MegaMesh: cached layer
references, cached RoPE references, and `decode_forward_full_attn_infer` for
full-attention layers. Hybrid Qwen 3.5 stages use
`python-hybrid-linear-fused`: full-attention layers take the inference decode
path and linear-attention layers use the fused raw decode path so `GatedDeltaNet`
can combine RMSNorm with its input projection. `/health` reports the selected
path under `fastpath`.
JSON generation results also include a compact `stages` snapshot so benchmark
output shows whether the run used `cpp-full-attention-loop` or a fallback.

For Qwen chat templates that expose a thinking switch, pass `--disable-thinking`
to request answer-only output. This does not change the MegaMesh kernels, but it
keeps short benchmarks from spending the entire visible output on the model's
reasoning preamble.

Shard workers set Qwen 3.5 microbatch decode kernel defaults before importing
the model runtime:

```text
MEGAGEMM_FUSED_RMSNORM_LINEAR_MAX_ROWS=16
MEGAGEMM_FAST_GEMV_MAX_ROWS=16
MEGAGEMM_FAST_GEMV_OPS=gate_up,down,linear_attn_in,linear_attn_out
```

These defaults do not force a slower kernel. They let the existing per-shape
microbenchmark consider fused/fast projection kernels for `rows=8` and
`rows=16`; the runtime can still reject them if the T4 benchmark says PyTorch is
faster. Use `--no-qwen35-shard-kernel-tune` to disable these worker defaults or
`--qwen35-kernel-max-rows` to test a different microbatch row limit.

For two-stage TTP decode batches, MegaMesh overlaps stage execution when there
is more than one microbatch chunk in a decode step: stage0 can process chunk
`n+1` while stage1 consumes chunk `n`. This only happens when
`num_prompts > microbatch_size`. For example, 8 prompts with
`--microbatch-size 16` is one chunk and cannot pipeline; 32 prompts with
`--microbatch-size 8` creates four chunks and can reduce the stage bubble. Batch
JSON includes `pipeline.max_decode_chunks_per_step` and
`pipeline.pipelined_decode_steps` to make this visible.

For low-concurrency two-stage TTP runs, MegaMesh uses direct shard chaining:
the coordinator sends `*_chain` to stage0 with `next_stage`, stage0 computes its
hidden state and forwards it directly to stage1 over a cached TTP connection,
then returns only the final token metadata to the coordinator. This avoids the
old relay path:

```text
old: coordinator -> stage0 -> coordinator -> stage1 -> coordinator
new: coordinator -> stage0 -> stage1 -> coordinator
```

Batch JSON reports `pipeline.ttp_chain_requests`, and worker health reports
`ttp_chain_forward_count`.

For two or more TTP stages, `mesh-shard-generate` and
`mesh-shard-generate-batch` can also push the whole decode loop into the first
stage. The coordinator sends one `generate_chain`/`generate_batch_chain`
request, then stage 0 repeatedly forwards hidden states through the ordered
stage route until the generation is finished. This removes one
coordinator-to-stage0 roundtrip per decode token while keeping the optimization
isolated to MegaMesh TTP.

Batch JSON reports this as:

```json
"pipeline": {
  "remote_chain_loop": true,
  "coordinator_ttp_requests": 1,
  "worker_chain_forwards": 40
}
```

Use `--no-remote-chain-loop` to compare against the older coordinator-driven
token loop.

The TTP receive path is tuned for these small per-token hidden-state transfers:
packets are read into a mutable `bytearray`, TTP tensor decode can build CPU
tensors as views over that buffer instead of making an intermediate NumPy copy,
and worker host-to-device transfers use `non_blocking=True` where PyTorch can
take advantage of it.

If the optional `megagemm_ttp_native` C extension is built, TTP uses native
socket `recv` into a Python `bytearray` instead of looping through Python
`socket.recv_into`. The native path understands Python timeout sockets and
waits on readiness instead of surfacing transient `EAGAIN`/`EWOULDBLOCK`
responses. Health reports this as:

```json
"ttp_runtime": {"native_recv": true}
```

If the extension is unavailable, TTP falls back to the pure-Python receive path.
Set `MEGAGEMM_TTP_NATIVE=0` before starting the coordinator/workers to force
that fallback while debugging native socket behavior.

## Current Implementation Boundary

MegaMesh is intentionally isolated from the core decode path.

Implemented today:

- replica worker HTTP server (`megagemm.mesh.worker`);
- weighted prompt router (`megagemm.mesh.router`);
- JSON protocol helpers (`megagemm.mesh.protocol`);
- failover inside the router when a replica worker fails before returning;
- pure layer-stage planning (`megagemm.mesh.planner`);
- AGron distributed shard mapping and directed TTP link probing;
- experimental layer-shard workers (`megagemm.mesh.shard_worker`);
- experimental ordered shard generation (`megagemm.mesh.shard_pipeline`);
- replicated layer-shard pipeline router (`ShardReplicaRouter`);
- TTP decode microbatching (`decode_batch`);
- TTP queue-style continuous batching (`generate_continuous_chain`);
- experimental vocab-sharded `lm_head` workers;
- experimental MLP intermediate-shard workers;
- isolated shard decode fastpath using the engine C++ decode loop when possible;
- persistent TTP socket transport (`megagemm.mesh.ttp`);
- binary tensor frame codec (`megagemm.mesh.binary_codec`);
- pinned-memory outbound tensor buffer pool (`megagemm.mesh.binary_codec`);
- isolated tensor payload codec (`megagemm.mesh.tensor_codec`).

Current layer-shard limitations:

- greedy generation only;
- prefill is still per prompt, decode supports TTP microbatches;
- FP16/BF16/FP32 tensor path, no quantized shard loading yet;
- TTP/binary hidden-state transport, still CPU-mediated;
- pinned-memory pooling is outbound-only today;
- no NCCL/GPU-direct transport yet;
- no general transformer-block tensor parallelism yet;
- remote MLP sharding is a correctness/architecture primitive, not a tuned
  throughput path yet;
- no streaming tokens across the router;
- Gemma 4 text shards are intentionally rejected for now.

This boundary is important: replica mode may call `InferenceEngine.generate_batch`
inside a worker, but MegaMesh modules do not modify the model, scheduler,
attention kernels, or decode loop.

## Layer-Stage Planning

The layer planner is available as a pure helper:

```python
from megagemm.mesh import plan_layer_stages

plan = plan_layer_stages(
    40,
    "http://10.0.0.10:8088@1#t4,http://10.0.0.11:8088@2#l4",
    devices=["cuda:0", "cuda:0"],
)
```

It returns contiguous ranges such as:

```json
[
  {"stage_id": 0, "layer_start": 0, "layer_end": 14, "num_layers": 14},
  {"stage_id": 1, "layer_start": 14, "layer_end": 40, "num_layers": 26}
]
```

The planner can be used by the layer-shard runtime, but it is still pure: it only
returns a stage contract and does not load weights or allocate KV cache.

## AGron Mesh Mapping

AGron is the experimental MegaMesh mapper for distributed layer-shard
inference. It is designed for the TTP case where workers may live on separate
Kaggle, Colab, RunPod, or personal machines. It does not assume local GPU
interconnects. It treats each shard worker as an atomic node with local layers,
local KV cache, local kernel policy, and directed TTP links to other nodes.

Probe the directed TTP mesh:

```bash
python -m megagemm mesh-agron-probe \
  --stages ttp://10.0.0.10:9090@1#s0,ttp://10.0.0.11:9091@1#s1,ttp://10.0.0.12:9092@1#s2 \
  --payload-bytes 65536 \
  --runs 5 \
  --warmup 1 > agron_profile.json
```

The probe asks each source worker to send a TTP payload to each target worker,
so the measured direction is the same direction used by shard-chain hidden-state
forwarding.

Build a plan from the measured profile:

```bash
python -m megagemm mesh-agron-plan \
  --num-layers 40 \
  --profile-json agron_profile.json \
  --hidden-bytes 20480 \
  --objective balanced \
  --allow-reorder
```

`--objective balanced` favors similar per-stage work, which is normally better
for pipelined decode. `--objective latency` favors the lowest serial step time
for very low concurrency. `--objective throughput` focuses on the slowest
pipeline stage. The output includes `stages_arg`, which can be passed to
`mesh-shard-generate-batch --stages` after starting workers with the planned
layer ranges.

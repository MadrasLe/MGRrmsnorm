"""
🔥 MegaGemm CLI
================
Command-line interface for MegaGemm inference engine.

Usage:
    megagemm generate "What is gravity?" --model Qwen/Qwen3-4B
    megagemm chat --model Qwen/Qwen3-4B
    megagemm bench --model Qwen/Qwen3-4B

    # Or via python -m:
    python -m megagemm generate "Hello world" --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

Author: Gabriel Yogi
"""

import argparse
import json
import os
import sys
import time
import torch


def _set_env_default_int(name: str, value: int) -> None:
    if name not in os.environ and int(value) > 0:
        os.environ[name] = str(int(value))


def _set_env_default_csv(name: str, values: list[str]) -> None:
    if name not in os.environ:
        os.environ[name] = ",".join(values)


def cmd_export_mgx(args):
    """Compile a model snapshot into an MGX artifact."""
    from megagemm.models import export_to_mgx

    result = export_to_mgx(
        model_source=args.model,
        out_path=args.out,
        dtype=args.dtype,
        quantize=args.quantize,
        sparsity=args.sparsity,
        target_backend=args.target_backend,
        emit_payload_cache=args.emit_payload_cache,
        payload_cache_dir=args.payload_cache_dir,
        export_mode=args.export_mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_inspect_mgx(args):
    """Inspect an MGX artifact header and manifest."""
    from megagemm.models import inspect_mgx

    result = inspect_mgx(
        args.path,
        validate_payload_hash=not args.skip_hash_check,
        payload_cache_dir=args.payload_cache_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_generate(args):
    """Generate text from a prompt."""
    from megagemm.engine import InferenceEngine

    engine = InferenceEngine(
        args.model,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device=args.device,
        quantize=args.quantize,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        deterministic=args.deterministic,
        seed=args.seed,
        monitor=args.monitor,
        dashboard=args.dashboard,
        mgx_verify_payload=None if not args.mgx_skip_hash_check else False,
        mgx_prefer_payload_cache=args.mgx_prefer_payload_cache,
        mgx_payload_cache_dir=args.mgx_payload_cache_dir,
    )

    result = engine.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        verbose=True,
        xai=args.xai,
    )

    if args.xai:
        text, report = result
        print(f"\n{'='*60}")
        print(report.summary())
    else:
        text = result

    print(f"\n{text}")


def cmd_chat(args):
    """Interactive chat mode."""
    from megagemm.engine import InferenceEngine

    print(f"🔥 MegaGemm Chat — Loading {args.model}...")
    engine = InferenceEngine(
        args.model,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device=args.device,
        quantize=args.quantize,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        deterministic=args.deterministic,
        seed=args.seed,
        mgx_verify_payload=None if not args.mgx_skip_hash_check else False,
        mgx_prefer_payload_cache=args.mgx_prefer_payload_cache,
        mgx_payload_cache_dir=args.mgx_payload_cache_dir,
    )

    print(f"\n💬 Chat ready! Type 'quit' or Ctrl+C to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ('quit', 'exit', 'q'):
                break

            t0 = time.perf_counter()
            output = engine.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            t1 = time.perf_counter()

            print(f"\nAssistant: {output}")
            print(f"  ({t1-t0:.1f}s)\n")

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break


def cmd_bench(args):
    """Benchmark decode speed."""
    from megagemm.engine import InferenceEngine

    print(f"⚡ MegaGemm Benchmark — {args.model}")
    print(f"   Tokens: {args.max_tokens}, Runs: {args.runs}")
    print(f"   Quantize: {args.quantize or 'none'}, Device: {args.device}")
    print()

    engine = InferenceEngine(
        args.model,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device=args.device,
        quantize=args.quantize,
        max_seq_len=args.max_seq_len,
        mgx_verify_payload=None if not args.mgx_skip_hash_check else False,
        mgx_prefer_payload_cache=args.mgx_prefer_payload_cache,
        mgx_payload_cache_dir=args.mgx_payload_cache_dir,
    )

    prompt = args.prompt or "Explain the theory of general relativity in detail."
    speeds = []

    for i in range(args.runs):
        if args.device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        engine.generate(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=0.0,  # greedy for consistent benchmark
        )

        if args.device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        tps = args.max_tokens / (t1 - t0)
        speeds.append(tps)
        print(f"  Run {i+1}/{args.runs}: {tps:.1f} tok/s ({(t1-t0)*1000:.0f}ms)")

    avg = sum(speeds) / len(speeds)
    peak = max(speeds)
    print(f"\n📊 Results ({args.runs} runs):")
    print(f"   Average: {avg:.1f} tok/s")
    print(f"   Peak:    {peak:.1f} tok/s")
    print(f"   Model:   {args.model}")
    print(f"   Quant:   {args.quantize or 'FP16'}")
    if args.device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU:     {gpu_name} ({vram_gb:.0f}GB)")


def cmd_embed(args):
    """Generate embeddings from one or more texts."""
    from megagemm.embeddings import EmbeddingEngine

    engine = EmbeddingEngine(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        normalize=None if not args.no_normalize else False,
        max_batch_tokens=args.max_batch_tokens,
        backend=args.backend,
        native_padding_free=not args.disable_native_padding_free,
        native_padding_free_force=args.force_native_padding_free_cpu,
        local_files_only=args.local_files_only,
    )

    result = engine.encode(
        args.texts,
        batch_size=args.batch_size,
        task=args.task,
        prompt=args.prompt,
        return_numpy=True,
    )
    print(json.dumps(result.tolist(), ensure_ascii=False))


def cmd_embed_bench(args):
    """Benchmark embedding throughput."""
    from megagemm.embeddings import EmbeddingEngine

    engine = EmbeddingEngine(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        normalize=None if not args.no_normalize else False,
        max_batch_tokens=args.max_batch_tokens,
        backend=args.backend,
        native_padding_free=not args.disable_native_padding_free,
        native_padding_free_force=args.force_native_padding_free_cpu,
        local_files_only=args.local_files_only,
    )

    texts = [args.text for _ in range(args.copies)]
    stats = engine.benchmark(
        texts,
        batch_size=args.batch_size,
        runs=args.runs,
        warmup=args.warmup,
        task=args.task,
    )

    print(f"Embedding Benchmark - {args.model}")
    print(f"  Texts:      {int(stats['num_texts'])}")
    print(f"  Batch size: {int(stats['batch_size'])}")
    print(f"  Dim:        {int(stats['embedding_dim'])}")
    print(f"  Latency:    {stats['avg_latency_ms']:.1f} ms")
    print(f"  Text/s:     {stats['texts_per_second']:.1f}")
    print(f"  Tok/s:      {stats['tokens_per_second']:.1f}")


def cmd_mesh_worker(args):
    """Start a MegaMesh replica worker."""
    from megagemm.mesh.worker import run_worker

    run_worker(
        args.model,
        host=args.host,
        port=args.port,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device=args.device,
        num_blocks=args.num_blocks,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        quantize=args.quantize,
        cache_dir=args.cache_dir,
        name=args.name,
        weight=args.weight,
    )


def cmd_mesh_shard_worker(args):
    """Start one experimental MegaMesh layer-shard worker."""
    if args.disable_shard_flat_decode:
        os.environ.setdefault("MEGAGEMM_FLAT_DECODE", "0")
    if args.disable_shard_cuda_rmsnorm:
        os.environ.setdefault("MEGAGEMM_DISABLE_CUDA_RMSNORM", "1")
    if not args.no_qwen35_shard_kernel_tune:
        _set_env_default_int(
            "MEGAGEMM_FUSED_RMSNORM_LINEAR_MAX_ROWS",
            args.qwen35_kernel_max_rows,
        )
        _set_env_default_int(
            "MEGAGEMM_FAST_GEMV_MAX_ROWS",
            args.qwen35_kernel_max_rows,
        )
        _set_env_default_csv(
            "MEGAGEMM_FAST_GEMV_OPS",
            ["gate_up", "down", "linear_attn_in", "linear_attn_out"],
        )
    from megagemm.mesh.shard_worker import run_shard_worker

    if getattr(args, "quantize", None):
        raise SystemExit("mesh-shard-worker currently supports FP16/BF16 only; omit --quantize")
    run_shard_worker(
        args.model,
        host=args.host,
        port=args.port,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        is_first=args.first_stage,
        is_last=args.last_stage,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device=args.device,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_seq_len=args.max_seq_len,
        cache_dir=args.cache_dir,
        name=args.name,
        ttp_port=args.ttp_port,
        ttp_pinned=not args.ttp_no_pinned,
        lm_head_shards=args.lm_head_shards,
        mlp_shards=args.mlp_shards,
    )


def cmd_mesh_lm_head_worker(args):
    """Start one experimental MegaMesh lm_head vocab-shard worker."""
    from megagemm.mesh.shard_worker import run_lm_head_worker

    run_lm_head_worker(
        args.model,
        host=args.host,
        port=args.port,
        ttp_port=args.ttp_port,
        vocab_start=args.vocab_start,
        vocab_end=args.vocab_end,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device=args.device,
        cache_dir=args.cache_dir,
        name=args.name,
    )


def cmd_mesh_mlp_worker(args):
    """Start one experimental MegaMesh MLP intermediate-shard worker."""
    from megagemm.mesh.shard_worker import run_mlp_worker

    run_mlp_worker(
        args.model,
        host=args.host,
        port=args.port,
        ttp_port=args.ttp_port,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        intermediate_start=args.intermediate_start,
        intermediate_end=args.intermediate_end,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device=args.device,
        cache_dir=args.cache_dir,
        name=args.name,
    )


def cmd_mesh_health(args):
    """Query MegaMesh worker health."""
    from megagemm.mesh import MeshRouter

    router = MeshRouter(args.workers, timeout=args.timeout)
    print(json.dumps(router.health(), indent=2, ensure_ascii=False))


def cmd_mesh_plan(args):
    """Plan contiguous layer stages for future MegaMesh layer sharding."""
    from megagemm.mesh import plan_layer_stages

    devices = None
    if args.devices:
        devices = [part.strip() for part in args.devices.split(",") if part.strip()]
    plan = plan_layer_stages(
        args.num_layers,
        args.workers,
        devices=devices,
        as_dict=True,
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def _load_agron_profile(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        parsed = json.load(fh)
    if not isinstance(parsed, dict):
        raise SystemExit("AGron profile JSON must be an object")
    return parsed


def _workers_from_agron_profile(profile: dict) -> str:
    parts = []
    for node in profile.get("nodes", []):
        if not isinstance(node, dict):
            continue
        endpoint = str(node.get("endpoint") or node.get("url") or node.get("worker_url") or "").strip()
        if not endpoint:
            continue
        weight = float(node.get("weight", node.get("speed", 1.0)))
        name = str(node.get("name", "")).strip()
        spec = f"{endpoint}@{weight:g}"
        if name:
            spec += f"#{name}"
        parts.append(spec)
    if not parts:
        raise SystemExit("--workers is required when --profile-json has no node endpoints")
    return ",".join(parts)


def cmd_mesh_agron_plan(args):
    """Plan layer shards with AGron distributed mesh mapping."""
    from megagemm.mesh import plan_agron_layer_stages

    profile = _load_agron_profile(args.profile_json)
    workers = args.workers or _workers_from_agron_profile(profile)
    plan = plan_agron_layer_stages(
        args.num_layers,
        workers,
        node_profiles=profile.get("nodes", []),
        link_profiles=profile.get("links", []),
        hidden_bytes=args.hidden_bytes,
        objective=args.objective,
        allow_reorder=args.allow_reorder,
        default_latency_ms=args.default_latency_ms,
        default_bandwidth_mbps=args.default_bandwidth_mbps,
        max_candidates=args.max_candidates,
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def cmd_mesh_agron_probe(args):
    """Measure directed TTP links between shard workers for AGron planning."""
    from megagemm.mesh.protocol import parse_worker_specs
    from megagemm.mesh.ttp import TTPClientPool

    stages = parse_worker_specs(args.stages)
    pool = TTPClientPool(stages, timeout=args.timeout)
    try:
        health_rows = []
        for stage in stages:
            row, _ = pool.request(stage, "health")
            row = dict(row)
            row.setdefault("endpoint", stage.url)
            row.setdefault("name", stage.name or row.get("name") or stage.url)
            row.setdefault("weight", stage.weight)
            health_rows.append(row)

        links = []
        name_by_url = {
            stage.url: str(health_rows[idx].get("name") or stage.name or stage.url)
            for idx, stage in enumerate(stages)
        }
        for src_idx, src in enumerate(stages):
            for dst_idx, dst in enumerate(stages):
                if src_idx == dst_idx:
                    continue
                meta, _ = pool.request(
                    src,
                    "probe_peer",
                    {
                        "target": dst.url,
                        "payload_bytes": args.payload_bytes,
                        "runs": args.runs,
                        "warmup": args.warmup,
                    },
                )
                row = dict(meta)
                row["src"] = str(health_rows[src_idx].get("name") or src.name or src.url)
                row["dst"] = str(health_rows[dst_idx].get("name") or dst.name or dst.url)
                row["src_endpoint"] = src.url
                row["dst_endpoint"] = dst.url
                row["dst_worker"] = name_by_url.get(dst.url, dst.url)
                links.append(row)

        nodes = []
        for idx, row in enumerate(health_rows):
            nodes.append(
                {
                    "name": str(row.get("name") or stages[idx].name or stages[idx].url),
                    "endpoint": stages[idx].url,
                    "weight": float(stages[idx].weight),
                    "layer_start": row.get("layer_start"),
                    "layer_end": row.get("layer_end"),
                    "device": row.get("device"),
                    "dtype": row.get("dtype"),
                    "gpu": row.get("gpu"),
                    "fastpath": row.get("fastpath"),
                    "ttp_runtime": row.get("ttp_runtime"),
                }
            )

        result = {
            "ok": True,
            "probe": "agron-probe-v0",
            "payload_bytes": args.payload_bytes,
            "runs": args.runs,
            "warmup": args.warmup,
            "nodes": nodes,
            "links": links,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        pool.close()


def _load_mesh_prompts(args):
    prompts = list(args.prompts or [])
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts.extend(line.rstrip("\n") for line in f if line.strip())
    if not prompts:
        raise SystemExit("mesh-generate requires at least one prompt or --prompts-file")
    return prompts


def cmd_mesh_generate(args):
    """Generate through a MegaMesh replica router."""
    from megagemm.mesh import MeshRouter

    router = MeshRouter(args.workers, timeout=args.timeout)
    prompts = _load_mesh_prompts(args)
    outputs, stats = router.generate_batch_with_stats(
        prompts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        verbose=args.verbose,
    )
    if args.json:
        print(json.dumps({"outputs": outputs, "stats": stats}, indent=2, ensure_ascii=False))
        return
    for i, text in enumerate(outputs):
        print(f"\n[{i}] {text}")
    print(
        f"\n[MegaMesh] {len(outputs)} prompts | "
        f"{stats['generated_tokens']} tokens | "
        f"{stats['elapsed_ms']:.1f} ms | "
        f"{stats['tokens_per_second']:.1f} tok/s"
    )


def cmd_mesh_shard_generate(args):
    """Generate through ordered experimental MegaMesh layer shards."""
    from megagemm.mesh import ShardPipeline

    if not str(args.model).strip():
        raise SystemExit("--model is empty; pass --model Qwen/Qwen3-14B or set MODEL in this shell cell.")
    if not str(args.stages).strip():
        raise SystemExit("--stages is empty; pass the ordered TTP/HTTP shard endpoints.")
    pipeline = ShardPipeline(
        args.stages,
        model_name=args.model,
        timeout=args.timeout,
        transport=args.transport,
        enable_thinking=False if args.disable_thinking else None,
        remote_chain_loop=not args.no_remote_chain_loop,
    )
    if args.health:
        print(json.dumps(pipeline.health(), indent=2, ensure_ascii=False))
    result = pipeline.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        include_prompt=args.include_prompt,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["text"])
        print(
            f"\n[MegaMeshShard] {result['generated_tokens']} tokens | "
            f"{result['elapsed_ms']:.1f} ms | "
            f"{result['tokens_per_second']:.2f} tok/s"
        )


def cmd_mesh_shard_generate_batch(args):
    """Generate multiple prompts through ordered experimental MegaMesh layer shards."""
    from megagemm.mesh import ShardPipeline

    if not str(args.model).strip():
        raise SystemExit("--model is empty; pass --model Qwen/Qwen3-14B or set MODEL in this shell cell.")
    if not str(args.stages).strip():
        raise SystemExit("--stages is empty; pass the ordered TTP/HTTP shard endpoints.")
    prompts = list(args.prompts or [])
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as fh:
            prompts.extend(line.rstrip("\n") for line in fh if line.strip())
    if not prompts:
        raise SystemExit("mesh-shard-generate-batch requires prompts or --prompts-file")
    pipeline = ShardPipeline(
        args.stages,
        model_name=args.model,
        timeout=args.timeout,
        transport=args.transport,
        enable_thinking=False if args.disable_thinking else None,
        remote_chain_loop=not args.no_remote_chain_loop,
    )
    if args.health:
        print(json.dumps(pipeline.health(), indent=2, ensure_ascii=False))
    result = pipeline.generate_batch(
        prompts,
        max_new_tokens=args.max_tokens,
        microbatch_size=args.microbatch_size,
        include_prompt=args.include_prompt,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for idx, output in enumerate(result["outputs"]):
            print(f"\n[{idx}] {output['text']}")
        print(
            f"\n[MegaMeshShardBatch] {result['num_prompts']} prompts | "
            f"{result['generated_tokens']} tokens | "
            f"{result['elapsed_ms']:.1f} ms | "
            f"{result['tokens_per_second']:.2f} tok/s"
        )


def cmd_mesh_shard_generate_continuous(args):
    """Generate a prompt queue through MegaMesh layer shards with continuous batching."""
    from megagemm.mesh import ShardPipeline

    if not str(args.model).strip():
        raise SystemExit("--model is empty; pass --model Qwen/Qwen3-14B or set MODEL in this shell cell.")
    if not str(args.stages).strip():
        raise SystemExit("--stages is empty; pass the ordered TTP shard endpoints.")
    prompts = list(args.prompts or [])
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as fh:
            prompts.extend(line.rstrip("\n") for line in fh if line.strip())
    if not prompts:
        raise SystemExit("mesh-shard-generate-continuous requires prompts or --prompts-file")
    pipeline = ShardPipeline(
        args.stages,
        model_name=args.model,
        timeout=args.timeout,
        transport=args.transport,
        enable_thinking=False if args.disable_thinking else None,
        remote_chain_loop=not args.no_remote_chain_loop,
    )
    if args.health:
        print(json.dumps(pipeline.health(), indent=2, ensure_ascii=False))
    result = pipeline.generate_continuous(
        prompts,
        max_new_tokens=args.max_tokens,
        microbatch_size=args.microbatch_size,
        max_batch_size=args.max_batch_size,
        include_prompt=args.include_prompt,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for idx, output in enumerate(result["outputs"]):
            print(f"\n[{idx}] {output['text']}")
        print(
            f"\n[MegaMeshShardContinuous] {result['num_prompts']} prompts | "
            f"{result['generated_tokens']} tokens | "
            f"{result['elapsed_ms']:.1f} ms | "
            f"{result['tokens_per_second']:.2f} tok/s"
        )


def cmd_mesh_shard_generate_replicas(args):
    """Generate through replicated MegaMesh layer-shard pipelines."""
    from megagemm.mesh import ShardReplicaRouter

    if not str(args.model).strip():
        raise SystemExit("--model is empty; pass --model Qwen/Qwen3-14B or a local snapshot path.")
    if not str(args.replicas).strip():
        raise SystemExit("--replicas is empty; pass semicolon-separated shard pipeline replicas.")
    prompts = list(args.prompts or [])
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as fh:
            prompts.extend(line.rstrip("\n") for line in fh if line.strip())
    if not prompts:
        raise SystemExit("mesh-shard-generate-replicas requires prompts or --prompts-file")

    router = ShardReplicaRouter(
        args.replicas,
        model_name=args.model,
        timeout=args.timeout,
        transport=args.transport,
        enable_thinking=False if args.disable_thinking else None,
        remote_chain_loop=not args.no_remote_chain_loop,
    )
    try:
        if args.health:
            print(json.dumps(router.health(), indent=2, ensure_ascii=False))
        result = router.generate_batch(
            prompts,
            max_new_tokens=args.max_tokens,
            microbatch_size=args.microbatch_size,
            include_prompt=args.include_prompt,
            strategy=args.strategy,
        )
    finally:
        router.close()

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for idx, output in enumerate(result["outputs"]):
            print(f"\n[{idx}] replica={output.get('replica_index')} {output['text']}")
        print(
            f"\n[MegaMeshShardReplicas] {result['num_replicas']} replicas | "
            f"{result['num_prompts']} prompts | "
            f"{result['generated_tokens']} tokens | "
            f"{result['elapsed_ms']:.1f} ms | "
            f"{result['tokens_per_second']:.2f} tok/s"
        )


def main():
    parser = argparse.ArgumentParser(
        prog='megagemm',
        description='🔥 MegaGemm — High-performance LLM inference engine',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # === Shared arguments ===
    def add_common_args(p):
        p.add_argument('--model', '-m', required=True, help='HuggingFace model ID, local snapshot path, or .mgx artifact')
        p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device (default: cuda)')
        p.add_argument('--bf16', action='store_true', help='Use bfloat16 instead of float16')
        p.add_argument('--quantize', '-q', choices=['int8', 'int4', 'fp8', 'awq'], help='Quantization mode')
        p.add_argument('--max-seq-len', type=int, default=4096, help='Max sequence length (default: 4096)')
        p.add_argument('--deterministic', '-d', action='store_true', help='Enable deterministic mode (bit-exact output)')
        p.add_argument('--seed', type=int, default=42, help='Random seed for deterministic mode (default: 42)')
        p.add_argument('--mgx-skip-hash-check', action='store_true', help='Skip embedded payload hash verification when loading a .mgx artifact')
        p.add_argument('--mgx-prefer-payload-cache', action='store_true', help='Prefer a reusable extracted safetensors cache for .mgx artifacts when available')
        p.add_argument('--mgx-payload-cache-dir', help='Optional directory for reusable .mgx payload cache files')

    def add_sampling_args(p):
        p.add_argument('--max-tokens', '-n', type=int, default=128, help='Max tokens to generate (default: 128)')
        p.add_argument('--temperature', '-t', type=float, default=0.7, help='Sampling temperature (default: 0.7)')
        p.add_argument('--top-k', type=int, default=50, help='Top-k sampling (default: 50)')
        p.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling threshold (default: 0.9)')
        p.add_argument('--repetition-penalty', type=float, default=1.1, help='Repetition penalty (default: 1.1)')

    def add_embedding_args(p):
        p.add_argument('--model', '-m', required=True, help='HuggingFace model ID or local path')
        p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Device (default: auto)')
        p.add_argument('--dtype', default='auto', choices=['auto', 'fp32', 'fp16', 'bf16'], help='Embedding dtype')
        p.add_argument('--backend', default='auto', choices=['auto', 'hf', 'native'], help='Embedding backend (default: auto)')
        p.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
        p.add_argument('--max-batch-tokens', type=int, default=0, help='Cap padded tokens per batch; 0 disables token-budget batching')
        p.add_argument('--max-length', type=int, help='Optional max token length')
        p.add_argument('--task', choices=['query', 'document', 'passage'], help='Apply a task-specific prompt')
        p.add_argument('--prompt', help='Custom prompt prefix or template containing {text}')
        p.add_argument('--no-normalize', action='store_true', help='Disable final L2 normalization')
        p.add_argument('--disable-native-padding-free', action='store_true', help='Disable padding-free packed attention in the native embedding backend')
        p.add_argument('--force-native-padding-free-cpu', action='store_true', help='Force padding-free packed attention on CPU for debugging or equivalence checks')
        p.add_argument('--local-files-only', action='store_true', help='Use only local Hugging Face cache files')

    # === generate ===
    p_gen = subparsers.add_parser('generate', aliases=['gen', 'g'], help='Generate text from a prompt')
    p_gen.add_argument('prompt', help='Input prompt text')
    add_common_args(p_gen)
    add_sampling_args(p_gen)
    p_gen.add_argument('--max-batch-size', type=int, default=1, help='Max batch size (default: 1)')
    p_gen.add_argument('--xai', action='store_true', help='Enable XAI interpretability report')
    p_gen.add_argument('--monitor', action='store_true', help='Enable inference monitoring')
    p_gen.add_argument('--dashboard', action='store_true', help='Start live monitoring dashboard')
    p_gen.set_defaults(func=cmd_generate)

    # === chat ===
    p_chat = subparsers.add_parser('chat', aliases=['c'], help='Interactive chat mode')
    add_common_args(p_chat)
    add_sampling_args(p_chat)
    p_chat.add_argument('--max-batch-size', type=int, default=1, help='Max batch size (default: 1)')
    p_chat.set_defaults(func=cmd_chat)

    # === bench ===
    p_bench = subparsers.add_parser('bench', aliases=['benchmark', 'b'], help='Benchmark decode speed')
    add_common_args(p_bench)
    p_bench.add_argument('--max-tokens', '-n', type=int, default=100, help='Tokens per run (default: 100)')
    p_bench.add_argument('--runs', '-r', type=int, default=3, help='Number of runs (default: 3)')
    p_bench.add_argument('--prompt', help='Custom benchmark prompt')
    p_bench.set_defaults(func=cmd_bench)

    # === export-mgx ===
    p_export_mgx = subparsers.add_parser('export-mgx', help='Compile a model into an MGX artifact')
    p_export_mgx.add_argument('--model', '-m', required=True, help='HuggingFace model ID or local snapshot directory')
    p_export_mgx.add_argument('--out', '-o', required=True, help='Output .mgx file')
    p_export_mgx.add_argument('--dtype', required=True, choices=['fp16', 'bf16'], help='Artifact base dtype')
    p_export_mgx.add_argument('--quantize', default='none', choices=['none', 'int8', 'int4', 'awq', 'native-int4', 'w4a16'], help='Optional quantization mode; native-int4/w4a16 is the standalone MGX backend')
    p_export_mgx.add_argument('--sparsity', default='none', choices=['none', '2:4'], help='Magnitude-prune eligible floating-point weights, or combine with native-int4 for packed INT4 2:4')
    p_export_mgx.add_argument('--target-backend', default='megagemm-cuda', help='Target backend tag stored in the artifact')
    p_export_mgx.add_argument('--export-mode', choices=['normal', 'streaming'], default='streaming', help='MGX export implementation. normal keeps payload in memory; streaming stages it through a temp safetensors file to reduce RAM usage.')
    p_export_mgx.add_argument('--emit-payload-cache', action='store_true', help='Also emit a reusable extracted safetensors payload cache alongside the .mgx metadata')
    p_export_mgx.add_argument('--payload-cache-dir', help='Optional directory for the emitted MGX payload cache')
    p_export_mgx.set_defaults(func=cmd_export_mgx)

    # === inspect-mgx ===
    p_inspect_mgx = subparsers.add_parser('inspect-mgx', help='Inspect an MGX artifact')
    p_inspect_mgx.add_argument('path', help='Path to the .mgx artifact')
    p_inspect_mgx.add_argument('--skip-hash-check', action='store_true', help='Skip tensor payload hash validation')
    p_inspect_mgx.add_argument('--payload-cache-dir', help='Optional directory used to resolve the expected MGX payload cache location')
    p_inspect_mgx.set_defaults(func=cmd_inspect_mgx)

    # === embed ===
    p_embed = subparsers.add_parser('embed', help='Encode one or more texts into embeddings')
    p_embed.add_argument('texts', nargs='+', help='Input text(s)')
    add_embedding_args(p_embed)
    p_embed.set_defaults(func=cmd_embed)

    # === embed-bench ===
    p_embed_bench = subparsers.add_parser('embed-bench', help='Benchmark embedding throughput')
    add_embedding_args(p_embed_bench)
    p_embed_bench.add_argument('--text', default='Embedding throughput benchmark text.', help='Benchmark text')
    p_embed_bench.add_argument('--copies', type=int, default=128, help='Number of texts per run (default: 128)')
    p_embed_bench.add_argument('--runs', '-r', type=int, default=3, help='Number of measured runs (default: 3)')
    p_embed_bench.add_argument('--warmup', type=int, default=1, help='Warmup runs before measurement (default: 1)')
    p_embed_bench.set_defaults(func=cmd_embed_bench)

    # === MegaMesh replica worker/router ===
    p_mesh_worker = subparsers.add_parser('mesh-worker', help='Start a MegaMesh full-model replica worker')
    add_common_args(p_mesh_worker)
    p_mesh_worker.add_argument('--host', default='0.0.0.0', help='Bind host (default: 0.0.0.0)')
    p_mesh_worker.add_argument('--port', type=int, default=8088, help='Bind port (default: 8088)')
    p_mesh_worker.add_argument('--name', default='', help='Worker name reported to the router')
    p_mesh_worker.add_argument('--weight', type=float, default=1.0, help='Relative routing weight for this worker')
    p_mesh_worker.add_argument('--num-blocks', type=int, default=0, help='KV cache blocks (0=auto)')
    p_mesh_worker.add_argument('--max-batch-size', type=int, default=512, help='Max local continuous batch size')
    p_mesh_worker.add_argument('--cache-dir', help='Optional Hugging Face cache directory')
    p_mesh_worker.set_defaults(func=cmd_mesh_worker)

    p_mesh_shard_worker = subparsers.add_parser('mesh-shard-worker', help='Start an experimental MegaMesh layer-shard worker')
    p_mesh_shard_worker.add_argument('--model', '-m', required=True, help='HuggingFace model ID or local snapshot path')
    p_mesh_shard_worker.add_argument('--device', default='cuda', help='Device for this shard process, e.g. cuda, cuda:0, cuda:1, or cpu')
    p_mesh_shard_worker.add_argument('--bf16', action='store_true', help='Use bfloat16 instead of float16')
    p_mesh_shard_worker.add_argument('--max-seq-len', type=int, default=512, help='Max sequence length for this shard KV cache')
    p_mesh_shard_worker.add_argument('--host', default='0.0.0.0', help='Bind host (default: 0.0.0.0)')
    p_mesh_shard_worker.add_argument('--port', type=int, default=8090, help='Bind port (default: 8090)')
    p_mesh_shard_worker.add_argument('--ttp-port', type=int, default=0, help='Optional TTP bind port for persistent tensor transfer')
    p_mesh_shard_worker.add_argument('--ttp-no-pinned', action='store_true', help='Disable pinned-memory outbound buffers for TTP')
    p_mesh_shard_worker.add_argument('--name', default='', help='Shard name reported in health checks')
    p_mesh_shard_worker.add_argument('--layer-start', type=int, required=True, help='First transformer layer owned by this shard, inclusive')
    p_mesh_shard_worker.add_argument('--layer-end', type=int, required=True, help='Last transformer layer owned by this shard, exclusive')
    p_mesh_shard_worker.add_argument('--first-stage', action='store_true', help='This shard owns token embeddings')
    p_mesh_shard_worker.add_argument('--last-stage', action='store_true', help='This shard owns final norm and lm_head')
    p_mesh_shard_worker.add_argument('--lm-head-shards', default='', help='Comma-separated TTP lm_head shard endpoints; last stage will skip local lm_head and reduce remote vocab shards')
    p_mesh_shard_worker.add_argument('--mlp-shards', default='', help='Comma-separated TTP MLP intermediate-shard endpoints; this layer stage will skip local MLP weights and reduce remote partial MLP outputs')
    p_mesh_shard_worker.add_argument('--num-blocks', type=int, default=512, help='KV cache blocks local to this shard')
    p_mesh_shard_worker.add_argument('--block-size', type=int, default=16, help='KV cache block size (default: 16)')
    p_mesh_shard_worker.add_argument('--cache-dir', help='Optional Hugging Face cache directory')
    p_mesh_shard_worker.add_argument('--qwen35-kernel-max-rows', type=int, default=16, help='Default max decode rows for Qwen 3.5 shard fused kernels (default: 16)')
    p_mesh_shard_worker.add_argument('--no-qwen35-shard-kernel-tune', action='store_true', help='Do not set MegaMesh shard defaults for Qwen 3.5 microbatch kernel selection')
    p_mesh_shard_worker.add_argument('--disable-shard-flat-decode', action='store_true', help='Disable experimental flat decode inside layer-shard workers for debugging')
    p_mesh_shard_worker.add_argument('--disable-shard-cuda-rmsnorm', action='store_true', help='Disable CUDA RMSNorm extension inside layer-shard workers for debugging')
    p_mesh_shard_worker.set_defaults(func=cmd_mesh_shard_worker)

    p_mesh_lm_head_worker = subparsers.add_parser('mesh-lm-head-worker', help='Start an experimental MegaMesh lm_head vocab-shard worker')
    p_mesh_lm_head_worker.add_argument('--model', '-m', required=True, help='HuggingFace model ID or local snapshot path')
    p_mesh_lm_head_worker.add_argument('--device', default='cuda', help='Device for this lm_head process, e.g. cuda, cuda:0, or cpu')
    p_mesh_lm_head_worker.add_argument('--bf16', action='store_true', help='Use bfloat16 instead of float16')
    p_mesh_lm_head_worker.add_argument('--host', default='0.0.0.0', help='Bind host (default: 0.0.0.0)')
    p_mesh_lm_head_worker.add_argument('--port', type=int, default=8099, help='HTTP health port (default: 8099)')
    p_mesh_lm_head_worker.add_argument('--ttp-port', type=int, required=True, help='TTP bind port for lm_head argmax requests')
    p_mesh_lm_head_worker.add_argument('--name', default='', help='lm_head shard name reported in health checks')
    p_mesh_lm_head_worker.add_argument('--vocab-start', type=int, required=True, help='First vocab row owned by this lm_head shard, inclusive')
    p_mesh_lm_head_worker.add_argument('--vocab-end', type=int, required=True, help='Last vocab row owned by this lm_head shard, exclusive')
    p_mesh_lm_head_worker.add_argument('--cache-dir', help='Optional Hugging Face cache directory')
    p_mesh_lm_head_worker.set_defaults(func=cmd_mesh_lm_head_worker)

    p_mesh_mlp_worker = subparsers.add_parser('mesh-mlp-worker', help='Start an experimental MegaMesh MLP intermediate-shard worker')
    p_mesh_mlp_worker.add_argument('--model', '-m', required=True, help='HuggingFace model ID or local snapshot path')
    p_mesh_mlp_worker.add_argument('--device', default='cuda', help='Device for this MLP process, e.g. cuda, cuda:0, or cpu')
    p_mesh_mlp_worker.add_argument('--bf16', action='store_true', help='Use bfloat16 instead of float16')
    p_mesh_mlp_worker.add_argument('--host', default='0.0.0.0', help='Bind host (default: 0.0.0.0)')
    p_mesh_mlp_worker.add_argument('--port', type=int, default=8098, help='HTTP health port (default: 8098)')
    p_mesh_mlp_worker.add_argument('--ttp-port', type=int, required=True, help='TTP bind port for MLP forward requests')
    p_mesh_mlp_worker.add_argument('--name', default='', help='MLP shard name reported in health checks')
    p_mesh_mlp_worker.add_argument('--layer-start', type=int, required=True, help='First transformer layer owned by this MLP shard, inclusive')
    p_mesh_mlp_worker.add_argument('--layer-end', type=int, required=True, help='Last transformer layer owned by this MLP shard, exclusive')
    p_mesh_mlp_worker.add_argument('--intermediate-start', type=int, required=True, help='First MLP intermediate row owned by this shard, inclusive')
    p_mesh_mlp_worker.add_argument('--intermediate-end', type=int, required=True, help='Last MLP intermediate row owned by this shard, exclusive')
    p_mesh_mlp_worker.add_argument('--cache-dir', help='Optional Hugging Face cache directory')
    p_mesh_mlp_worker.set_defaults(func=cmd_mesh_mlp_worker)

    p_mesh_health = subparsers.add_parser('mesh-health', help='Query MegaMesh worker health')
    p_mesh_health.add_argument('--workers', required=True, help='Comma-separated worker URLs, optionally url@weight#name')
    p_mesh_health.add_argument('--timeout', type=float, default=30.0, help='Request timeout in seconds')
    p_mesh_health.set_defaults(func=cmd_mesh_health)

    p_mesh_plan = subparsers.add_parser('mesh-plan', help='Plan MegaMesh layer-shard stages without running inference')
    p_mesh_plan.add_argument('--num-layers', type=int, required=True, help='Total transformer layers to partition')
    p_mesh_plan.add_argument('--workers', required=True, help='Comma-separated worker URLs, optionally url@weight#name')
    p_mesh_plan.add_argument('--devices', help='Optional comma-separated device labels, one per planned stage')
    p_mesh_plan.set_defaults(func=cmd_mesh_plan)

    p_mesh_agron_plan = subparsers.add_parser('mesh-agron-plan', help='Plan MegaMesh layer shards with AGron mesh mapping')
    p_mesh_agron_plan.add_argument('--num-layers', type=int, required=True, help='Total transformer layers to partition')
    p_mesh_agron_plan.add_argument('--workers', help='Comma-separated TTP/HTTP worker URLs, optionally url@weight#name')
    p_mesh_agron_plan.add_argument('--profile-json', help='Optional AGron probe/profile JSON with nodes and directed links')
    p_mesh_agron_plan.add_argument('--hidden-bytes', type=int, default=0, help='Hidden-state bytes crossing each shard boundary per step')
    p_mesh_agron_plan.add_argument('--objective', choices=['balanced', 'latency', 'throughput'], default='balanced', help='Planning objective')
    p_mesh_agron_plan.add_argument('--allow-reorder', action='store_true', help='Allow AGron to reorder workers by measured mesh links')
    p_mesh_agron_plan.add_argument('--default-latency-ms', type=float, default=0.0, help='Fallback directed-link latency when profile is missing')
    p_mesh_agron_plan.add_argument('--default-bandwidth-mbps', type=float, default=0.0, help='Fallback directed-link bandwidth when profile is missing')
    p_mesh_agron_plan.add_argument('--max-candidates', type=int, default=20000, help='Max split candidates per worker order')
    p_mesh_agron_plan.set_defaults(func=cmd_mesh_agron_plan)

    p_mesh_agron_probe = subparsers.add_parser('mesh-agron-probe', help='Probe directed TTP links between shard workers')
    p_mesh_agron_probe.add_argument('--stages', required=True, help='Comma-separated TTP shard worker URLs, optionally url@weight#name')
    p_mesh_agron_probe.add_argument('--payload-bytes', type=int, default=65536, help='Probe payload bytes sent from source to target')
    p_mesh_agron_probe.add_argument('--runs', type=int, default=5, help='Measured probe requests per directed link')
    p_mesh_agron_probe.add_argument('--warmup', type=int, default=1, help='Warmup requests per directed link')
    p_mesh_agron_probe.add_argument('--timeout', type=float, default=30.0, help='TTP request timeout')
    p_mesh_agron_probe.set_defaults(func=cmd_mesh_agron_probe)

    p_mesh_generate = subparsers.add_parser('mesh-generate', help='Generate through MegaMesh replica workers')
    p_mesh_generate.add_argument('prompts', nargs='*', help='Prompt(s) to route')
    p_mesh_generate.add_argument('--workers', required=True, help='Comma-separated worker URLs, optionally url@weight#name')
    p_mesh_generate.add_argument('--prompts-file', help='Optional newline-delimited prompt file')
    p_mesh_generate.add_argument('--timeout', type=float, default=900.0, help='Request timeout in seconds')
    p_mesh_generate.add_argument('--json', action='store_true', help='Print JSON output')
    p_mesh_generate.add_argument('--verbose', action='store_true', help='Ask workers to print local scheduler stats')
    p_mesh_generate.add_argument('--max-tokens', '-n', type=int, default=128, help='Max tokens to generate (default: 128)')
    p_mesh_generate.add_argument('--temperature', '-t', type=float, default=0.0, help='Sampling temperature (default: 0.0)')
    p_mesh_generate.add_argument('--top-k', type=int, default=50, help='Top-k sampling (default: 50)')
    p_mesh_generate.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling threshold (default: 0.9)')
    p_mesh_generate.set_defaults(func=cmd_mesh_generate)

    p_mesh_shard_generate = subparsers.add_parser('mesh-shard-generate', help='Generate through ordered experimental MegaMesh layer shards')
    p_mesh_shard_generate.add_argument('prompt', help='Input prompt text')
    p_mesh_shard_generate.add_argument('--model', '-m', required=True, help='Tokenizer source matching the shard workers')
    p_mesh_shard_generate.add_argument('--stages', required=True, help='Comma-separated ordered shard worker URLs')
    p_mesh_shard_generate.add_argument('--max-tokens', '-n', type=int, default=64, help='Max tokens to generate (default: 64)')
    p_mesh_shard_generate.add_argument('--timeout', type=float, default=900.0, help='Request timeout in seconds')
    p_mesh_shard_generate.add_argument('--transport', choices=['ttp', 'binary', 'json'], default='binary', help='Shard tensor transport (default: binary)')
    p_mesh_shard_generate.add_argument('--json', action='store_true', help='Print JSON output')
    p_mesh_shard_generate.add_argument('--health', action='store_true', help='Print shard health before generation')
    p_mesh_shard_generate.add_argument('--disable-thinking', action='store_true', help='Ask supported chat templates to disable visible thinking output')
    p_mesh_shard_generate.add_argument('--include-prompt', action='store_true', help='Decode prompt + generated text instead of generated text only')
    p_mesh_shard_generate.add_argument('--no-remote-chain-loop', action='store_true', help='Disable two-stage TTP remote decode loop optimization')
    p_mesh_shard_generate.set_defaults(func=cmd_mesh_shard_generate)

    p_mesh_shard_generate_batch = subparsers.add_parser('mesh-shard-generate-batch', help='Generate multiple prompts with TTP decode microbatches')
    p_mesh_shard_generate_batch.add_argument('prompts', nargs='*', help='Input prompt text(s)')
    p_mesh_shard_generate_batch.add_argument('--prompts-file', help='Optional newline-delimited prompt file')
    p_mesh_shard_generate_batch.add_argument('--model', '-m', required=True, help='Tokenizer source matching the shard workers')
    p_mesh_shard_generate_batch.add_argument('--stages', required=True, help='Comma-separated ordered shard worker URLs')
    p_mesh_shard_generate_batch.add_argument('--max-tokens', '-n', type=int, default=64, help='Max tokens to generate per prompt (default: 64)')
    p_mesh_shard_generate_batch.add_argument('--microbatch-size', type=int, default=8, help='Decode microbatch size (default: 8)')
    p_mesh_shard_generate_batch.add_argument('--timeout', type=float, default=900.0, help='Request timeout in seconds')
    p_mesh_shard_generate_batch.add_argument('--transport', choices=['ttp'], default='ttp', help='Batch shard transport (default: ttp)')
    p_mesh_shard_generate_batch.add_argument('--json', action='store_true', help='Print JSON output')
    p_mesh_shard_generate_batch.add_argument('--health', action='store_true', help='Print shard health before generation')
    p_mesh_shard_generate_batch.add_argument('--disable-thinking', action='store_true', help='Ask supported chat templates to disable visible thinking output')
    p_mesh_shard_generate_batch.add_argument('--include-prompt', action='store_true', help='Decode prompt + generated text instead of generated text only')
    p_mesh_shard_generate_batch.add_argument('--no-remote-chain-loop', action='store_true', help='Disable two-stage TTP remote decode loop optimization')
    p_mesh_shard_generate_batch.set_defaults(func=cmd_mesh_shard_generate_batch)

    p_mesh_shard_generate_continuous = subparsers.add_parser('mesh-shard-generate-continuous', help='Generate a prompt queue with MegaMesh shard continuous batching')
    p_mesh_shard_generate_continuous.add_argument('prompts', nargs='*', help='Input prompt text(s)')
    p_mesh_shard_generate_continuous.add_argument('--prompts-file', help='Optional newline-delimited prompt file')
    p_mesh_shard_generate_continuous.add_argument('--model', '-m', required=True, help='Tokenizer source matching the shard workers')
    p_mesh_shard_generate_continuous.add_argument('--stages', required=True, help='Comma-separated ordered TTP shard worker URLs')
    p_mesh_shard_generate_continuous.add_argument('--max-tokens', '-n', type=int, default=64, help='Max tokens to generate per prompt (default: 64)')
    p_mesh_shard_generate_continuous.add_argument('--microbatch-size', type=int, default=8, help='Decode microbatch size (default: 8)')
    p_mesh_shard_generate_continuous.add_argument('--max-batch-size', type=int, default=32, help='Max live requests admitted at once (default: 32)')
    p_mesh_shard_generate_continuous.add_argument('--timeout', type=float, default=900.0, help='Request timeout in seconds')
    p_mesh_shard_generate_continuous.add_argument('--transport', choices=['ttp'], default='ttp', help='Continuous shard transport (default: ttp)')
    p_mesh_shard_generate_continuous.add_argument('--json', action='store_true', help='Print JSON output')
    p_mesh_shard_generate_continuous.add_argument('--health', action='store_true', help='Print shard health before generation')
    p_mesh_shard_generate_continuous.add_argument('--disable-thinking', action='store_true', help='Ask supported chat templates to disable visible thinking output')
    p_mesh_shard_generate_continuous.add_argument('--include-prompt', action='store_true', help='Decode prompt + generated text instead of generated text only')
    p_mesh_shard_generate_continuous.add_argument('--no-remote-chain-loop', action='store_true', help='Disable TTP remote decode loop optimization')
    p_mesh_shard_generate_continuous.set_defaults(func=cmd_mesh_shard_generate_continuous)

    p_mesh_shard_generate_replicas = subparsers.add_parser('mesh-shard-generate-replicas', help='Generate through replicated MegaMesh layer-shard pipelines')
    p_mesh_shard_generate_replicas.add_argument('prompts', nargs='*', help='Input prompt text(s)')
    p_mesh_shard_generate_replicas.add_argument('--prompts-file', help='Optional newline-delimited prompt file')
    p_mesh_shard_generate_replicas.add_argument('--model', '-m', required=True, help='Tokenizer source matching the shard workers')
    p_mesh_shard_generate_replicas.add_argument('--replicas', required=True, help='Semicolon-separated replicas; each replica is a comma-separated ordered shard pipeline')
    p_mesh_shard_generate_replicas.add_argument('--max-tokens', '-n', type=int, default=64, help='Max tokens to generate per prompt (default: 64)')
    p_mesh_shard_generate_replicas.add_argument('--microbatch-size', type=int, default=8, help='Decode microbatch size used inside each replica (default: 8)')
    p_mesh_shard_generate_replicas.add_argument('--strategy', choices=['round_robin', 'chunk'], default='round_robin', help='Prompt assignment strategy across replicas (default: round_robin)')
    p_mesh_shard_generate_replicas.add_argument('--timeout', type=float, default=900.0, help='Request timeout in seconds')
    p_mesh_shard_generate_replicas.add_argument('--transport', choices=['ttp'], default='ttp', help='Replicated shard transport (default: ttp)')
    p_mesh_shard_generate_replicas.add_argument('--json', action='store_true', help='Print JSON output')
    p_mesh_shard_generate_replicas.add_argument('--health', action='store_true', help='Print replica health before generation')
    p_mesh_shard_generate_replicas.add_argument('--disable-thinking', action='store_true', help='Ask supported chat templates to disable visible thinking output')
    p_mesh_shard_generate_replicas.add_argument('--include-prompt', action='store_true', help='Decode prompt + generated text instead of generated text only')
    p_mesh_shard_generate_replicas.add_argument('--no-remote-chain-loop', action='store_true', help='Disable TTP remote decode loop optimization inside each replica')
    p_mesh_shard_generate_replicas.set_defaults(func=cmd_mesh_shard_generate_replicas)

    # Parse
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    args.func(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())

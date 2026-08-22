"""
🔥 MegaGemm Stress Test — Maximum Throughput on Single GPU
============================================================
Realistic stress test with LONG contexts and high batch sizes.

- Prompts: 256-600+ tokens each (real workload)
- Output: 200 tokens per request
- KV offload to CPU RAM
- Scales batch: 8 → 16 → 32 → 64 → 128 → 256

Usage (Kaggle/Colab L4):
    python examples/stress_test.py
    python examples/stress_test.py --max-batch 512 --cpu-blocks 40000
    python examples/stress_test.py --quantize int8 --max-batch 256

Author: Gabriel Yogi
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Long, realistic prompts (256-600+ tokens each after chat template)
PROMPTS = [
    # ~300 tokens — document analysis
    """Read the following research paper abstract and provide a detailed analysis of its methodology, findings, and potential limitations:

Abstract: This study investigates the effectiveness of transformer-based language models in automated code review. We collected a dataset of 50,000 pull requests from 200 popular open-source repositories on GitHub, spanning multiple programming languages including Python, JavaScript, Java, and C++. Our approach fine-tunes a pre-trained CodeBERT model on pairs of code changes and reviewer comments. We evaluate the model's ability to identify common code quality issues such as potential bugs, style violations, performance bottlenecks, and security vulnerabilities. Our results show that the fine-tuned model achieves 78.3% precision and 65.7% recall in identifying genuine code quality issues, outperforming rule-based static analysis tools by 23% on average. We also conducted a user study with 30 professional developers who rated the model's suggestions as helpful in 71% of cases. However, the model struggles with context-dependent issues that require understanding of the broader codebase architecture. We discuss implications for integrating AI-assisted code review into existing development workflows and propose a hybrid approach combining our model with traditional static analysis. Future work will focus on incorporating repository-level context through retrieval-augmented generation techniques. The code and dataset are publicly available for reproducibility.""",

    # ~350 tokens — legal analysis
    """Analyze the following terms of service excerpt and identify potential issues for consumers, areas of ambiguity, and clauses that might be considered unfair or unenforceable:

Section 7. Limitation of Liability: TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL THE COMPANY, ITS AFFILIATES, AGENTS, DIRECTORS, EMPLOYEES, SUPPLIERS OR LICENSORS BE LIABLE FOR ANY INDIRECT, PUNITIVE, INCIDENTAL, SPECIAL, CONSEQUENTIAL OR EXEMPLARY DAMAGES, INCLUDING WITHOUT LIMITATION DAMAGES FOR LOSS OF PROFITS, GOODWILL, USE, DATA OR OTHER INTANGIBLE LOSSES, ARISING OUT OF OR RELATING TO THE USE OF, OR INABILITY TO USE, THE SERVICE. UNDER NO CIRCUMSTANCES WILL THE COMPANY BE RESPONSIBLE FOR ANY DAMAGE, LOSS OR INJURY RESULTING FROM HACKING, TAMPERING OR OTHER UNAUTHORIZED ACCESS OR USE OF THE SERVICE OR YOUR ACCOUNT OR THE INFORMATION CONTAINED THEREIN. THE COMPANY ASSUMES NO LIABILITY OR RESPONSIBILITY FOR ANY ERRORS, MISTAKES, OR INACCURACIES OF CONTENT, PERSONAL INJURY OR PROPERTY DAMAGE, OF ANY NATURE WHATSOEVER, RESULTING FROM YOUR ACCESS TO AND USE OF OUR SERVICE, ANY UNAUTHORIZED ACCESS TO OR USE OF OUR SECURE SERVERS AND ALL PERSONAL INFORMATION STORED THEREIN. THE AGGREGATE LIABILITY OF THE COMPANY SHALL NOT EXCEED THE GREATER OF ONE HUNDRED DOLLARS OR THE AMOUNT YOU PAID THE COMPANY IN THE PAST SIX MONTHS.""",

    # ~400 tokens — technical debugging
    """I'm experiencing a complex issue with my distributed training setup. Here's the full context:

Environment: 4x NVIDIA A100-80GB GPUs connected via NVLink, Ubuntu 22.04, CUDA 12.1, PyTorch 2.1.0, DeepSpeed ZeRO Stage 3 with CPU offloading enabled.

Model: Custom 13B parameter transformer with 40 layers, hidden size 5120, 40 attention heads, GQA with 8 KV heads, RoPE positional encoding with theta=500000, SwiGLU activation, RMSNorm.

Training config: Global batch size 256, micro batch size 4 per GPU, gradient accumulation steps 16, BF16 mixed precision, learning rate 2e-5 with cosine schedule, 2000 warmup steps, AdamW optimizer with ZeRO Stage 3.

The problem: Training runs fine for the first 500 steps (approximately 2 hours), achieving expected loss decrease from 2.8 to 1.9. Then suddenly at step ~500, loss spikes to NaN and all gradient norms become inf. This happens consistently across multiple runs with different random seeds. I've verified the data pipeline has no corrupted samples. The issue does not occur with ZeRO Stage 2 or with a smaller batch size of 128, but I need Stage 3 for memory efficiency and the larger batch for convergence.

I've tried: gradient clipping at 1.0, reducing learning rate to 1e-5, disabling BF16, checking for inf/nan in inputs. None of these fixes the issue.

What could be causing this and how should I debug it?""",

    # ~350 tokens — data science
    """Given the following dataset description and business requirements, design a complete machine learning pipeline and explain your choices:

Dataset: E-commerce customer behavior data with 2.5 million rows and 47 features including:
- Demographics: age, gender, location (country, state, city), income bracket, education level
- Behavioral: page views per session, average session duration, bounce rate, pages per visit, time on site, scroll depth
- Purchase history: total orders (last 30/90/365 days), average order value, total spend, return rate, coupon usage frequency, last purchase date
- Product interactions: categories viewed, items added to cart, wishlist size, product review count, search queries per session
- Marketing: email open rate, click through rate, SMS opt-in status, push notification engagement, referral source, UTM parameters
- Technical: device type, browser, operating system, screen resolution, connection speed

Target variable: Binary classification - will the customer make a purchase in the next 14 days (current conversion rate: 3.2%)

Business requirements:
1. Model must be interpretable enough to explain to non-technical stakeholders why a customer is flagged as likely to convert
2. Predictions must be generated in under 100ms for real-time website personalization
3. Model should handle the severe class imbalance (96.8% negative, 3.2% positive)
4. False positives are 3x more costly than false negatives (aggressive discounting to non-buyers wastes margin)
5. The pipeline must be automated for daily retraining with new data""",

    # ~300 tokens — creative writing critique
    """Please provide a detailed literary analysis of the following passage, examining narrative technique, thematic elements, symbolism, and prose style:

The rain had been falling for seventeen days when Maria first noticed the water pooling beneath the floorboards. It gathered in the gaps between the old pine planks like dark mercury, catching what little light filtered through the windows she had stopped opening weeks ago. She pressed her palm against the cool surface and watched her reflection fracture into a thousand versions of herself, each one slightly different from the last, as though the house were showing her all the lives she might have lived if she had made different choices at different crossroads.

The walls had begun to breathe. She was certain of it now, after nights spent pressing her ear against the plaster, listening to the deep, tidal rhythm that matched nothing in the natural world she remembered. The house had its own pulse, its own metabolism, consuming the rain and transforming it into something she couldn't name but could feel in the marrow of her bones. Her mother had warned her about old houses, about the way they absorbed the memories of their inhabitants until the distinction between building and dweller dissolved like sugar in warm water.

She found herself speaking to the rooms as though they were old friends, narrating her movements through the hallways as if the house needed to know where she was at all times.""",

    # ~350 tokens — systems design
    """Design a real-time collaborative document editing system similar to Google Docs that can support 10,000 concurrent users editing the same document. Address the following aspects in detail:

1. Conflict Resolution: How would you handle simultaneous edits to the same paragraph by multiple users? Compare and contrast Operational Transformation (OT) vs Conflict-free Replicated Data Types (CRDTs). Which would you choose and why?

2. Architecture: Design the system architecture including frontend clients, WebSocket servers, document processing layer, and persistent storage. How would you handle the connection management for 10,000 concurrent WebSocket connections? What load balancing strategy would you use?

3. Data Model: How would you structure the document data to efficiently support real-time character-by-character updates, cursor position tracking for all users, revision history with full undo/redo capability, and comment threads anchored to specific text ranges?

4. Performance: What strategies would you employ to minimize latency? Consider server proximity, delta compression for updates, batching of operations, and client-side prediction. What is your target latency from keystroke to visibility on other clients?

5. Scaling: How does your design scale horizontally? What happens when a single document becomes too popular for a single server to handle? How do you partition the document processing?

6. Offline Support: How would you handle users going offline and coming back with conflicting changes? What data structures do you need on the client side for offline editing?""",

    # ~250 tokens — medical
    """Review the following patient case and provide a differential diagnosis with reasoning:

Patient: 45-year-old female presents to the emergency department with acute onset chest pain that started 3 hours ago while at rest. The pain is described as sharp, substernal, radiating to the left shoulder and jaw. Pain intensity is 8/10. She reports associated shortness of breath, diaphoresis, and nausea.

Past Medical History: Type 2 diabetes mellitus diagnosed 8 years ago (HbA1c 7.8%), hypertension (on lisinopril 20mg daily), hyperlipidemia (on atorvastatin 40mg daily), obesity (BMI 34.2), family history of coronary artery disease (father had MI at age 52).

Current Medications: Metformin 1000mg BID, lisinopril 20mg daily, atorvastatin 40mg daily, aspirin 81mg daily.

Vital Signs: BP 165/95 mmHg, HR 110 bpm (regular), RR 22, SpO2 94% on room air, Temp 37.1°C.

Physical Exam: Anxious appearing, diaphoretic. Cardiac: tachycardic, regular rhythm, no murmurs/gallops/rubs. Lungs: bilateral basilar crackles. Abdomen: soft, non-tender. Extremities: no edema, pulses intact bilaterally.

ECG: ST elevation in leads II, III, aVF with reciprocal ST depression in I, aVL. Troponin I: 2.4 ng/mL (normal <0.04).""",

    # ~300 tokens — code review
    """Review this Python code for a rate limiter implementation and identify all bugs, performance issues, security concerns, and design problems:

```python
import time
import threading
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, client_id):
        now = time.time()
        with self.lock:
            # Remove old entries
            self.requests[client_id] = [
                t for t in self.requests[client_id]
                if now - t < self.window
            ]

            if len(self.requests[client_id]) < self.max_requests:
                self.requests[client_id].append(now)
                return True
            return False

    def get_remaining(self, client_id):
        now = time.time()
        active = [t for t in self.requests[client_id] if now - t < self.window]
        return max(0, self.max_requests - len(active))

    def reset(self, client_id):
        del self.requests[client_id]

    def cleanup(self):
        now = time.time()
        for client_id in self.requests:
            self.requests[client_id] = [
                t for t in self.requests[client_id]
                if now - t < self.window
            ]

limiter = RateLimiter(max_requests=10, window_seconds=1)
```

Focus on: thread safety issues, memory leaks, algorithmic complexity, and production readiness.""",
]


def run_stress_test(
    model_name: str,
    max_tokens: int = 200,
    batch_sizes=None,
    quantize=None,
    kv_offload: bool = True,
    num_gpu_blocks: int = 2048,
    num_cpu_blocks: int = 4096,
    gpu_window: int = 64,
):
    import torch
    from megagemm.engine import InferenceEngine

    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 64]

    print(f"\n{'='*75}")
    print(f"  🔥 MegaGemm STRESS TEST — Maximum Throughput")
    print(f"{'='*75}")
    print(f"  Model:          {model_name}")
    print(f"  Quantize:       {quantize or 'FP16'}")
    print(f"  Max tokens:     {max_tokens}")
    print(f"  KV offload:     {'✅ ON' if kv_offload else '❌ OFF'}")
    if kv_offload:
        print(f"  GPU blocks:     {num_gpu_blocks if num_gpu_blocks > 0 else 'auto'}")
        print(f"  CPU blocks:     {num_cpu_blocks if num_cpu_blocks > 0 else 'auto'}")
        print(f"  GPU window:     {gpu_window}")
    print(f"  Batch sizes:    {batch_sizes}")
    print(f"  Prompt pool:    {len(PROMPTS)} prompts")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:            {gpu} ({vram:.0f}GB)")
    print(f"{'='*75}")

    # Load engine
    print(f"\n📦 Loading {model_name}...")
    t0 = time.perf_counter()
    engine = InferenceEngine(
        model_name,
        quantize=quantize,
        kv_offload=kv_offload,
        num_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        gpu_window=gpu_window,
        max_batch_size=max(batch_sizes),
    )
    tokenizer = engine.tokenizer
    load_time = time.perf_counter() - t0
    print(f"   Loaded in {load_time:.1f}s")

    # Show prompt stats
    prompt_lens = []
    for p in PROMPTS:
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            try:
                msg = [{"role": "user", "content": p}]
                fmt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            except Exception:
                fmt = p
        else:
            fmt = p
        prompt_lens.append(len(tokenizer.encode(fmt)))
    avg_prompt = sum(prompt_lens) / len(prompt_lens)
    print(f"   Prompt lengths: min={min(prompt_lens)}, avg={avg_prompt:.0f}, max={max(prompt_lens)} tokens")
    print(f"   Context per request: ~{avg_prompt:.0f} prompt + {max_tokens} output = ~{avg_prompt + max_tokens:.0f} tokens")

    # Compute max feasible batch (paged attention needs ALL blocks on GPU)
    block_size = engine.block_manager.block_size if hasattr(engine.block_manager, 'block_size') else 16
    actual_gpu_blocks = getattr(engine.block_manager, 'num_gpu_blocks',
                                getattr(engine.block_manager, 'num_blocks', num_gpu_blocks))
    max_ctx = int(max(prompt_lens) + max_tokens)
    blocks_per_seq = (max_ctx + block_size - 1) // block_size
    max_feasible_batch = actual_gpu_blocks // blocks_per_seq
    print(f"   ⚡ Blocks/seq: {blocks_per_seq} ({max_ctx} tokens / {block_size})")
    print(f"   ⚡ Max feasible batch: {actual_gpu_blocks} GPU blocks / {blocks_per_seq} = {max_feasible_batch}")

    # Filter impossible batch sizes
    feasible_sizes = [bs for bs in batch_sizes if bs <= max_feasible_batch]
    skipped = [bs for bs in batch_sizes if bs > max_feasible_batch]
    if skipped:
        print(f"   ⚠️  Skipping batch sizes {skipped} (exceed GPU block limit)")
    batch_sizes = feasible_sizes
    if not batch_sizes:
        print(f"\n   ❌ No feasible batch sizes! Need more GPU blocks or smaller context.")
        print(f"      Try: --gpu-blocks {blocks_per_seq * 16} or --max-tokens {(num_gpu_blocks // 8) * block_size - int(avg_prompt)}")
        return

    # Warmup
    print("   🔥 Warmup (compiling Triton kernels)...")
    engine.generate("warmup", max_new_tokens=10, temperature=0.0,
                    repetition_penalty=1.0, verbose=False)
    engine.generate_batch(["warmup batch"], max_new_tokens=10,
                          temperature=0.0, verbose=False)
    engine.reset_monitor()
    torch.cuda.synchronize()
    print("   ✅ Ready!")

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   📊 VRAM: {vram_used:.1f}GB / {vram_total:.0f}GB ({vram_used/vram_total*100:.0f}%)")

    results = {}

    for bs in batch_sizes:
        prompts = (PROMPTS * ((bs // len(PROMPTS)) + 1))[:bs]

        # Count input tokens
        total_input_tokens = 0
        for p in prompts:
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                try:
                    msg = [{"role": "user", "content": p}]
                    fmt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                except Exception:
                    fmt = p
            else:
                fmt = p
            bos = tokenizer.bos_token
            add_sp = not (bos and fmt.startswith(bos))
            total_input_tokens += len(tokenizer.encode(fmt, add_special_tokens=add_sp))

        avg_ctx = total_input_tokens / bs + max_tokens
        print(f"\n{'─'*75}")
        print(f"  🚀 Batch={bs} | {total_input_tokens} input tok | ~{avg_ctx:.0f} ctx/request")
        print(f"{'─'*75}")

        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            texts = engine.generate_batch(
                prompts, max_new_tokens=max_tokens,
                temperature=0.0, verbose=False,
            )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            total_output_tokens = sum(
                len(tokenizer.encode(t, add_special_tokens=False)) for t in texts
            )
            total_tokens = total_input_tokens + total_output_tokens

            tok_s = total_tokens / elapsed
            out_tok_s = total_output_tokens / elapsed
            ms_per_prompt = elapsed / bs * 1000

            results[bs] = {
                'time': elapsed,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'total_tokens': total_tokens,
                'tok_s': tok_s,
                'out_tok_s': out_tok_s,
                'ms_per_prompt': ms_per_prompt,
                'prompts_per_hour': bs / elapsed * 3600,
                'ok': True,
            }

            print(f"   ⏱️  Time:       {elapsed:.2f}s")
            print(f"   📊 Throughput: {tok_s:,.0f} tok/s total | {out_tok_s:,.0f} tok/s decode")
            print(f"   📊 Per prompt: {ms_per_prompt:.0f}ms | {bs/elapsed:.1f} prompts/s")
            print(f"   📊 Rate:       {bs/elapsed*3600:,.0f} prompts/hour")
            print(f"   Sample output:")
            preview = texts[0].strip()[:120]
            print(f"     → {preview}...")

        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            results[bs] = {'ok': False, 'error': str(e)}

        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            print(f"   VRAM: {vram_used:.1f}GB")

    # Summary
    print(f"\n{'='*75}")
    print(f"  📊 STRESS TEST RESULTS")
    print(f"{'='*75}")
    print(f"\n  {'Batch':>5} │ {'Time':>7} │ {'Total tok/s':>12} │ {'Decode tok/s':>12} │ {'ms/prompt':>10} │ {'prompts/hr':>12}")
    print(f"  {'─'*5}─┼─{'─'*7}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*10}─┼─{'─'*12}")

    peak_tok_s = 0
    peak_bs = 0
    for bs in batch_sizes:
        r = results.get(bs, {})
        if not r.get('ok', False):
            print(f"  {bs:>5} │ {'ERR':>7} │ {'—':>12} │ {'—':>12} │ {'—':>10} │ {'—':>12}")
            continue
        if r['tok_s'] > peak_tok_s:
            peak_tok_s = r['tok_s']
            peak_bs = bs
        marker = " 🏆" if r['tok_s'] >= peak_tok_s else ""
        print(f"  {bs:>5} │ {r['time']:>6.1f}s │ {r['tok_s']:>10,.0f}  │ {r['out_tok_s']:>10,.0f}  │ {r['ms_per_prompt']:>8.0f}ms │ {r['prompts_per_hour']:>10,.0f}{marker}")

    if peak_bs > 0:
        r = results[peak_bs]
        print(f"\n  🏆 Peak at batch={peak_bs}:")
        print(f"     {r['tok_s']:,.0f} tok/s | {r['prompts_per_hour']:,.0f} prompts/hr | {r['ms_per_prompt']:.0f}ms/prompt")

    print(f"\n  📈 Scaling:")
    for bs in batch_sizes:
        r = results.get(bs, {})
        if not r.get('ok', False):
            print(f"      batch={bs:>3}: ❌ OOM/ERROR")
            continue
        bar_len = int(r['tok_s'] / max(1, peak_tok_s) * 40)
        print(f"      batch={bs:>3}: {r['tok_s']:>6,.0f} tok/s  {'█' * bar_len}")

    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser(description="MegaGemm Stress Test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--max-batch", type=int, default=256,
                        help="Max batch size to test")
    parser.add_argument("--quantize", choices=["int8", "fp8"], default=None)
    parser.add_argument("--no-offload", action="store_true")
    parser.add_argument("--gpu-blocks", type=int, default=0,
                        help="GPU KV blocks (0=auto)")
    parser.add_argument("--cpu-blocks", type=int, default=0,
                        help="CPU KV blocks (0=auto)")
    parser.add_argument("--gpu-window", type=int, default=64)
    args = parser.parse_args()

    sizes = [s for s in [16, 32, 64, 128, 256, 512, 1024] if s <= args.max_batch]

    run_stress_test(
        model_name=args.model,
        max_tokens=args.max_tokens,
        batch_sizes=sizes,
        quantize=args.quantize,
        kv_offload=not args.no_offload,
        num_gpu_blocks=args.gpu_blocks,
        num_cpu_blocks=args.cpu_blocks,
        gpu_window=args.gpu_window,
    )


if __name__ == "__main__":
    main()

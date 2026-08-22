# Contributing to MegaGemm

MegaGemm is a solo-directed experimental inference runtime with a large kernel
and benchmark surface. Contributions are welcome when they preserve
correctness, provide evidence, and keep experimental paths clearly labeled.

## Development setup

GPU inference development:

```bash
python -m pip install -e ".[inference,dev]" --no-build-isolation
```

Minimal CPU/documentation work without native compilation:

```bash
MEGAGEMM_SKIP_NATIVE=1 python -m pip install -e ".[cpu,dev]" --no-build-isolation
```

See [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) for all extras and Windows
PowerShell syntax.

## Before opening a change

1. State the model, device, dtype, and execution path affected.
2. Add or update a correctness test.
3. Run the smallest relevant CPU suite.
4. For GPU changes, compare against a trusted reference before benchmarking.
5. Record exact hardware/software versions for performance claims.
6. Keep generated benchmark output out of source control; promote only curated
   summaries with links to raw artifacts.

## Correctness before speed

Kernel changes include maximum absolute/relative error or an appropriate
task-specific metric. Quantization work separates speed/memory results from
quality validation. A faster result with an unexplained correctness change is
not accepted as a performance improvement.

## Benchmark claims

Follow [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Do not generalize a win on one
shape to every workload. Include negative results, OOMs, and regimes where the
comparison backend wins.

## Project organization

- Stable runtime code belongs under `megagemm/`.
- Native sources belong under `src/` and bindings under `pytorch_binding/`.
- Reusable benchmark runners belong under `benchmarks/` or `microgemm/tools/`.
- Generated output belongs in ignored result directories.
- Dated, curated conclusions belong under `docs/`.
- Temporary model/test artifacts must use an ignored temporary directory.

## Commit scope

Prefer changes that isolate one concern: correctness fix, kernel optimization,
model enablement, packaging, tests, or documentation. When an optimization
needs a fallback, document the selection condition and include coverage for
both paths.

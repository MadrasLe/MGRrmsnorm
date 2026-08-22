# Test suite

The test directory contains several kinds of validation. It is intentionally
broader than a single `unittest discover` run: some files are unittest/pytest
modules, some are executable assertion scripts, and some are GPU or Colab
harnesses.

Current source inventory:

- 46 Python test files;
- approximately 18.7k lines of test code;
- more than 500 functions named `test_*`;
- CPU-safe unit tests, GPU kernel tests, model integration tests, and remote
  hardware harnesses.

Counts are descriptive and will change as the project evolves.

## Test classes

| Class | Examples | Expected environment |
|---|---|---|
| Pure Python/reporting | `test_xai.py`, `test_monitor.py` | Python + base package |
| CPU/PyTorch logic | `test_deterministic.py`, `test_scheduler.py`, `test_mgx.py` | CPU PyTorch; optional packages by module |
| Kernel policy/heuristics | `test_rmsnorm_policy.py`, `test_paged_attention_heuristics.py` | Usually CPU with mocked device properties; pytest for monkeypatch fixtures |
| CUDA/Triton kernels | `test_rmsnorm.py`, `test_awq_gemm.py`, `test_qwen3_moe.py` | Compatible NVIDIA GPU and feature extras |
| Model integration | `test_inference.py`, `test_multimodel.py`, `test_qwen35.py`, `test_gemma4.py` | Model access, sufficient RAM/VRAM, inference extras |
| Hardware harnesses | `test_gemma4_colab_harness.py`, long-context harness files | Colab/Kaggle or explicitly provisioned GPU environment |

## Fast local checks

The following scripts are CPU-safe in the current tree:

```bash
python -X utf8 tests/test_xai.py
python -X utf8 tests/test_monitor.py
python -X utf8 tests/test_deterministic.py
```

The UTF-8 switch keeps emoji-bearing test output portable on Windows consoles.

For pytest-style CPU tests:

```bash
pytest -q \
  tests/test_scheduler.py \
  tests/test_paged_attention_heuristics.py \
  tests/test_rmsnorm_policy.py
```

## Broad discovery

```bash
python -m unittest discover -s tests -p "test_*.py"
```

This command does **not** represent the complete suite because top-level
pytest-style functions and standalone `main()` harnesses are not all collected
by unittest. It may also report missing optional native modules in a CPU-only
environment.

## GPU validation policy

Every published kernel-performance change should record:

- GPU name and compute capability;
- CUDA, driver, PyTorch, and Triton versions;
- dtype and tensor shape;
- reference implementation and tolerances;
- warmup/iteration count and timing method;
- correctness result before latency numbers;
- whether compilation/autotuning time was excluded.

Do not silently treat a skipped GPU test as a pass. Hardware-gated results
should be reported separately from the CPU suite.

## Consolidation roadmap

- Add pytest markers: `cpu`, `cuda`, `model`, `benchmark`, `remote`, `slow`.
- Move standalone harness behavior behind marker-aware pytest entry points.
- Publish one CPU CI command and separate scheduled GPU jobs.
- Replace string/source-inspection regressions with behavioral assertions where
  possible.
- Store benchmark performance gates outside correctness tests.

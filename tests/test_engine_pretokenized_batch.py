from types import SimpleNamespace
from unittest import mock

from megagemm.engine.engine import InferenceEngine


class _NoTokenizeTokenizer:
    eos_token_id = 2

    def encode(self, *_args, **_kwargs):
        raise AssertionError("pretokenized prompts must bypass tokenizer.encode")

    def decode(self, *_args, **_kwargs):
        raise AssertionError("decode_outputs=False must bypass tokenizer.decode")


class _FakeScheduler:
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.added = []
        self.completed = []
        self.pending = True
        _FakeScheduler.last_instance = self

    def add_request(self, *, prompt_ids, metadata, **_kwargs):
        request_id = len(self.added)
        self.added.append(
            {
                "prompt_ids": list(prompt_ids),
                "metadata": dict(metadata),
            }
        )
        self.completed.append(
            SimpleNamespace(request_id=request_id, generated_ids=[10, 11])
        )
        return request_id

    def has_pending(self):
        return self.pending

    def step(self):
        self.pending = False
        return list(self.completed)

    def get_stats(self):
        return {"total_tokens": 4, "running": 0, "waiting": 0, "completed": 2}


def test_generate_batch_accepts_exact_pretokenized_rows():
    engine = object.__new__(InferenceEngine)
    engine.model = object()
    engine.block_manager = object()
    engine.max_batch_size = 2
    engine.device = "cpu"
    engine.tokenizer = _NoTokenizeTokenizer()
    engine.kv_offload = False

    token_rows = [[1, 7, 8], [1, 9, 8]]
    with mock.patch("megagemm.engine.engine.Scheduler", _FakeScheduler):
        output = engine.generate_batch(
            token_rows,
            max_new_tokens=2,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            ignore_eos=True,
            decode_outputs=False,
            materialize_generated_tokens=True,
        )

    scheduler = _FakeScheduler.last_instance
    assert output == ["", ""]
    assert [row["prompt_ids"] for row in scheduler.added] == token_rows
    assert all(row["metadata"]["pretokenized"] for row in scheduler.added)
    assert all(row["metadata"]["prompt"] is None for row in scheduler.added)
    assert scheduler.kwargs["materialize_generated_tokens"] is True


def test_generate_batch_releases_previous_request_scheduler_before_next_one():
    engine = object.__new__(InferenceEngine)
    engine.model = object()
    engine.block_manager = object()
    engine.max_batch_size = 1
    engine.device = "cpu"
    engine.tokenizer = _NoTokenizeTokenizer()
    engine.kv_offload = False
    engine._last_scheduler = object()

    class _FreshScheduler(_FakeScheduler):
        def __init__(self, **kwargs):
            assert engine._last_scheduler is None
            super().__init__(**kwargs)

    with mock.patch("megagemm.engine.engine.Scheduler", _FreshScheduler):
        output = engine.generate_batch(
            [[1, 7, 8]],
            max_new_tokens=2,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            ignore_eos=True,
            decode_outputs=False,
            materialize_generated_tokens=True,
        )

    assert output == [""]
    assert engine._last_scheduler is _FreshScheduler.last_instance


def test_generate_batch_reuses_compatible_idle_scheduler(monkeypatch):
    monkeypatch.setenv("MEGAGEMM_REUSE_REQUEST_SCHEDULER", "1")
    engine = object.__new__(InferenceEngine)
    engine.model = object()
    engine.block_manager = object()
    engine.max_batch_size = 1
    engine.device = "cpu"
    engine.tokenizer = _NoTokenizeTokenizer()
    engine.kv_offload = False

    class _ReusableScheduler(_FakeScheduler):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.reset_calls = []

        def can_reuse_for_request(self, **kwargs):
            assert kwargs == {
                "model": engine.model,
                "block_manager": engine.block_manager,
                "max_batch_size": 1,
                "device": "cpu",
            }
            return True

        def reset_for_request(self, **kwargs):
            self.reset_calls.append(dict(kwargs))
            self.added.clear()
            self.completed.clear()
            self.pending = True

    previous = _ReusableScheduler(
        model=engine.model,
        block_manager=engine.block_manager,
        max_batch_size=1,
        device="cpu",
        prefill_capture_hook=None,
        materialize_generated_tokens=True,
    )
    engine._last_scheduler = previous

    with mock.patch(
        "megagemm.engine.engine.Scheduler",
        side_effect=AssertionError("a compatible scheduler must not be recreated"),
    ):
        output = engine.generate_batch(
            [[1, 7, 8]],
            max_new_tokens=2,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            ignore_eos=True,
            decode_outputs=False,
            materialize_generated_tokens=True,
        )

    assert output == [""]
    assert engine._last_scheduler is previous
    assert previous.reset_calls == [
        {
            "prefill_capture_hook": None,
            "materialize_generated_tokens": True,
        }
    ]

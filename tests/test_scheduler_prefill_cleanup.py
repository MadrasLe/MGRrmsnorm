import pytest
import torch

from megagemm.engine.scheduler import Scheduler


class _FailingModel:
    def prefill(self, *args, **kwargs):
        raise RuntimeError("prefill failed")

    def prefill_packed(self, *args, **kwargs):
        raise RuntimeError("packed prefill failed")


class _BlockManager:
    block_size = 16
    num_blocks = 1024

    def __init__(self):
        self.block_tables = {}

    def can_allocate(self, num_tokens):
        return True

    def allocate_sequence(self, seq_id, num_tokens=0):
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already exists")
        self.block_tables[seq_id] = [0]

    def free_sequence(self, seq_id):
        self.block_tables.pop(seq_id, None)


def test_prefill_failure_frees_single_sequence():
    blocks = _BlockManager()
    scheduler = Scheduler(_FailingModel(), blocks, max_batch_size=1, device="cpu")
    scheduler.add_request([1, 2, 3], max_new_tokens=8, temperature=0.0)

    with pytest.raises(RuntimeError, match="prefill failed"):
        scheduler.step()

    assert blocks.block_tables == {}
    assert scheduler.num_running == 0


def test_packed_prefill_failure_frees_batch_sequences():
    blocks = _BlockManager()
    scheduler = Scheduler(_FailingModel(), blocks, max_batch_size=2, device="cpu")
    scheduler.add_request([1, 2, 3], max_new_tokens=8, temperature=0.0)
    scheduler.add_request([4, 5, 6], max_new_tokens=8, temperature=0.0)

    with pytest.raises(RuntimeError, match="packed prefill failed"):
        scheduler.step()

    assert blocks.block_tables == {}
    assert scheduler.num_running == 0

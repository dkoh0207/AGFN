"""Standalone tests for the top-K persistence behaviour of FTSampling_Iterator.

No pytest in this env, so this is a plain script: run it directly, it raises on
failure and prints PASS lines on success.

    cd src && python ../tests/test_top_k_flush.py

Covers the two behaviours added when decoupling the top-K flush from
``checkpoint_every``:
  1. ``_flush_top_k`` writes the full heap to CSV and guards the SDF export on a
     Uni-Dock backend being present.
  2. ``__iter__`` flushes the heap one final time when the generator is torn down
     (exception path here; GeneratorExit follows the same finally mechanism).
"""

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.top_k_tracker import TopKTracker, REWARD_CSV, AFFINITY_CSV  # noqa: E402
from iterators.samp_iter_finetune import FTSampling_Iterator  # noqa: E402


def _populated_tracker(n=5):
    t = TopKTracker(ks=(10, 100))
    for i in range(n):
        t.add(smiles="C" * (i + 1), reward=float(i), affinity=-float(i), iteration=i)
    return t


def _bare_iter(tmpdir, tracker, reward):
    """An iterator instance with __init__ bypassed; only the attributes the
    flush path touches are set."""
    it = FTSampling_Iterator.__new__(FTSampling_Iterator)
    it.top_k_tracker = tracker
    it.top_k_dir = tmpdir
    it.reward = reward
    return it


def _csv_rows(path):
    with open(path, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # drop header


class _RewardNoUnidock:  # hasattr(reward, "unidock") -> False
    pass


class _CondRaises:
    def compute_cond_info_forward(self, n):
        raise RuntimeError("boom-from-body")


def test_flush_helper_writes_all_entries():
    with tempfile.TemporaryDirectory() as d:
        it = _bare_iter(d, _populated_tracker(5), _RewardNoUnidock())
        it._flush_top_k()
        for fn in (REWARD_CSV, AFFINITY_CSV):
            p = os.path.join(d, fn)
            assert os.path.exists(p), f"{fn} not written"
            assert _csv_rows(p) == 5, f"{fn} has {_csv_rows(p)} rows, expected 5"
    print("PASS test_flush_helper_writes_all_entries")


def test_final_flush_on_teardown():
    # The online loop body raises on its first statement. The outer try/finally
    # must still persist the heap before the exception propagates.
    with tempfile.TemporaryDirectory() as d:
        it = _bare_iter(d, _populated_tracker(3), _RewardNoUnidock())
        it.offline_data = False
        it.num_online = 4
        # __iter__ resolves top_k_dir from hps['log_dir'] (the per-run subdir) and resumes
        # from it before looping; d is empty so load() is a no-op and the 3 pre-seeded
        # molecules survive to be flushed by the finally.
        it.hps = {"num_iter": 1, "top_k_save_every": 100, "log_dir": d}
        it.cond_info_task = _CondRaises()
        raised = False
        try:
            for _ in it.__iter__():
                pass
        except RuntimeError:
            raised = True
        assert raised, "body exception should propagate out of __iter__"
        p = os.path.join(d, REWARD_CSV)
        assert os.path.exists(p), "final flush did not write CSV on teardown"
        assert _csv_rows(p) == 3, f"final flush wrote {_csv_rows(p)} rows, expected 3"
    print("PASS test_final_flush_on_teardown")


def test_generator_exit_reaches_finally_once():
    # Contract the worker-shutdown path relies on: an inner `except Exception`
    # does NOT swallow GeneratorExit, so closing mid-yield reaches the outer
    # finally exactly once.
    calls = {"n": 0}

    def gen():
        try:
            for _ in range(1000):
                try:
                    yield 1
                except Exception as e:  # must not catch GeneratorExit
                    raise e
        finally:
            calls["n"] += 1

    g = gen()
    next(g)
    g.close()
    assert calls["n"] == 1, f"finally ran {calls['n']} times, expected 1"
    print("PASS test_generator_exit_reaches_finally_once")


if __name__ == "__main__":
    test_flush_helper_writes_all_entries()
    test_final_flush_on_teardown()
    test_generator_exit_reaches_finally_once()
    print("ALL PASS")

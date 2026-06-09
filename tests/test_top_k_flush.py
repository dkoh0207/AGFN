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


def test_dedup_running_mean_and_roundtrip():
    # The same molecule sampled 3x (re-docked, so the scores drift) must collapse to ONE entry
    # whose reward/affinity are the running means; a second molecule seen once stays distinct.
    t = TopKTracker(ks=(10, 100))
    for r, a, it in [(1.0, -1.0, 5), (2.0, -2.0, 7), (3.0, -3.0, 9)]:
        t.add(smiles="CCO", reward=r, affinity=a, iteration=it)
    t.add(smiles="c1ccccc1", reward=0.5, affinity=-0.5, iteration=6)

    rew = t.snapshot()["top_by_reward"][100]
    assert len(rew) == 2, f"expected 2 unique mols, got {len(rew)}: {rew}"
    cco = next(e for e in rew if e[0] == "CCO")  # (smiles, reward_mean, aff_mean, first_iter, count)
    assert abs(cco[1] - 2.0) < 1e-9, cco          # mean(1,2,3)
    assert abs(cco[2] - (-2.0)) < 1e-9, cco        # mean(-1,-2,-3)
    assert cco[3] == 5, cco                         # discovery (first-seen) iteration
    assert cco[4] == 3, cco                         # sample count

    # Round-trip: save -> the CSV has no duplicate SMILES -> load resumes the exact means + counts.
    with tempfile.TemporaryDirectory() as d:
        t.save(d)
        with open(os.path.join(d, REWARD_CSV), newline="") as f:
            smis = [row["smiles"] for row in csv.DictReader(f)]
        assert len(smis) == len(set(smis)) == 2, f"CSV not unique: {smis}"
        t2 = TopKTracker(ks=(10, 100))
        t2.load(d)
    cco2 = next(e for e in t2.snapshot()["top_by_reward"][100] if e[0] == "CCO")
    assert abs(cco2[1] - 2.0) < 1e-9 and abs(cco2[2] - (-2.0)) < 1e-9 and cco2[4] == 3, cco2

    # Resuming and sampling again continues the cumulative mean rather than resetting the count.
    t2.add(smiles="CCO", reward=6.0, affinity=-6.0, iteration=20)
    cco3 = next(e for e in t2.snapshot()["top_by_reward"][100] if e[0] == "CCO")
    assert cco3[4] == 4, cco3                                  # 3 restored + 1 new
    assert abs(cco3[1] - (1 + 2 + 3 + 6) / 4) < 1e-9, cco3      # = 3.0
    print("PASS test_dedup_running_mean_and_roundtrip")


def test_load_pre_count_csv_collapses_duplicates():
    # Resume path for runs written before this change: the old CSV has no `count` column and one
    # row per observation, so the same molecule appears on multiple rows (the bug being fixed).
    # load() must replay those rows and collapse them into the new running mean.
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, REWARD_CSV), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("rank", "smiles", "reward", "affinity", "iteration"))  # pre-count header
            w.writerow((1, "CCO", 3.0, -3.0, 9))
            w.writerow((2, "CCO", 1.0, -1.0, 5))
            w.writerow((3, "c1ccccc1", 0.5, -0.5, 6))
        t = TopKTracker(ks=(10, 100))
        t.load(d)
    rew = t.snapshot()["top_by_reward"][100]
    assert len(rew) == 2, f"old duplicate rows should collapse to 2 mols, got {rew}"
    cco = next(e for e in rew if e[0] == "CCO")
    assert abs(cco[1] - 2.0) < 1e-9 and abs(cco[2] - (-2.0)) < 1e-9 and cco[4] == 2, cco
    print("PASS test_load_pre_count_csv_collapses_duplicates")


def test_averages_means_and_nan():
    import math

    # rewards 0..4, affinities 0..-4; <10 entries so top-10 == top-100 == all of them.
    a = _populated_tracker(5).averages()
    assert abs(a["top10_reward"] - 2.0) < 1e-9, a
    assert abs(a["top100_reward"] - 2.0) < 1e-9, a
    assert abs(a["top10_affinity"] - (-2.0)) < 1e-9, a
    assert abs(a["top100_affinity"] - (-2.0)) < 1e-9, a

    # empty tracker -> every bucket NaN
    ae = TopKTracker(ks=(10, 100)).averages()
    assert all(math.isnan(v) for v in ae.values()), ae

    # rewards present but no affinities -> reward buckets real, affinity buckets NaN
    noaff = TopKTracker(ks=(10, 100))
    for i in range(3):
        noaff.add(smiles="N" * (i + 1), reward=float(i), affinity=None, iteration=i)
    an = noaff.averages()
    assert abs(an["top10_reward"] - 1.0) < 1e-9, an  # mean(0,1,2)
    assert math.isnan(an["top10_affinity"]) and math.isnan(an["top100_affinity"]), an
    print("PASS test_averages_means_and_nan")


if __name__ == "__main__":
    test_flush_helper_writes_all_entries()
    test_final_flush_on_teardown()
    test_generator_exit_reaches_finally_once()
    test_dedup_running_mean_and_roundtrip()
    test_load_pre_count_csv_collapses_duplicates()
    test_averages_means_and_nan()
    print("ALL PASS")

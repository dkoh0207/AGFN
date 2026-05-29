"""Standalone tests for BBB batch metrics used by metrics.csv logging.

Run directly:

    cd src && python ../tests/test_bbb_metrics_logging.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from iterators.samp_iter_finetune import FTSampling_Iterator  # noqa: E402


class _Reward:
    last_bbb_probabilities = np.array([0.95, np.nan, 0.20])
    last_bbb_score_mask = np.array([True, False, True])
    last_bbb_pass_mask = np.array([True, False, False])


class _Batch:
    pass


def test_bbb_metrics_count_failures_as_nonpassing_and_exclude_from_average():
    it = FTSampling_Iterator.__new__(FTSampling_Iterator)
    it.reward = _Reward()
    metrics = it._bbb_batch_metrics()
    assert np.allclose(metrics[0], 0.575), metrics
    assert np.allclose(metrics[1], 100.0 / 3.0), metrics

    batch = _Batch()
    it._attach_bbb_metrics(batch, metrics)
    assert np.allclose(batch.avg_bbb_score, 0.575), batch.avg_bbb_score
    assert np.allclose(batch.percent_bbb_passing_threshold, 100.0 / 3.0), (
        batch.percent_bbb_passing_threshold
    )
    print("PASS test_bbb_metrics_count_failures_as_nonpassing_and_exclude_from_average")


def test_missing_bbb_state_has_no_metrics():
    it = FTSampling_Iterator.__new__(FTSampling_Iterator)
    it.reward = object()
    assert it._bbb_batch_metrics() is None
    print("PASS test_missing_bbb_state_has_no_metrics")


if __name__ == "__main__":
    test_bbb_metrics_count_failures_as_nonpassing_and_exclude_from_average()
    test_missing_bbb_state_has_no_metrics()
    print("ALL PASS")

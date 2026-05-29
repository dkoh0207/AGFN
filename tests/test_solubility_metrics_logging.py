"""Standalone tests for solubility batch metrics used by metrics.csv logging.

Run directly:

    cd src && python ../tests/test_solubility_metrics_logging.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from iterators.samp_iter_finetune import FTSampling_Iterator  # noqa: E402


class _Reward:
    last_sol_values = np.array([-3.0, np.nan, -8.0])
    last_sol_score_mask = np.array([True, False, True])
    last_sol_pass_mask = np.array([True, False, False])


class _Batch:
    pass


def test_solubility_metrics_count_failures_as_nonpassing_and_exclude_from_average():
    it = FTSampling_Iterator.__new__(FTSampling_Iterator)
    it.reward = _Reward()
    metrics = it._solubility_batch_metrics()
    # average LogS over scored rows only: (-3.0 + -8.0) / 2 = -5.5
    assert np.allclose(metrics[0], -5.5), metrics
    # one of three molecules passes the gate
    assert np.allclose(metrics[1], 100.0 / 3.0), metrics

    batch = _Batch()
    it._attach_solubility_metrics(batch, metrics)
    assert np.allclose(batch.avg_sol_logS, -5.5), batch.avg_sol_logS
    assert np.allclose(batch.percent_sol_passing_threshold, 100.0 / 3.0), (
        batch.percent_sol_passing_threshold
    )
    print("PASS test_solubility_metrics_count_failures_as_nonpassing_and_exclude_from_average")


def test_missing_solubility_state_has_no_metrics():
    it = FTSampling_Iterator.__new__(FTSampling_Iterator)
    it.reward = object()
    assert it._solubility_batch_metrics() is None
    print("PASS test_missing_solubility_state_has_no_metrics")


if __name__ == "__main__":
    test_solubility_metrics_count_failures_as_nonpassing_and_exclude_from_average()
    test_missing_solubility_state_has_no_metrics()
    print("ALL PASS")

"""Standalone tests for the shared reward-gate transition functions.

Run directly:

    cd src && python ../tests/test_thresholding.py
"""

import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from agfn.autogluon.thresholding import soft_threshold, apply_gate  # noqa: E402

FAIL = 1e-30


def test_soft_threshold_edges_and_midpoint():
    x = np.array([-7.0, -6.0, -5.0, -4.0, -3.0])
    s = soft_threshold(x, low=-6.0, high=-4.0)
    # below/at low -> 0, at/above high -> 1, midpoint -> 0.5 (smoothstep is symmetric)
    assert np.allclose(s, [0.0, 0.0, 0.5, 1.0, 1.0]), s
    print("PASS test_soft_threshold_edges_and_midpoint")


def test_soft_threshold_is_monotonic_and_flat_outside():
    x = np.linspace(-8.0, -2.0, 25)
    s = soft_threshold(x, low=-6.0, high=-4.0)
    assert np.all(np.diff(s) >= -1e-12), s
    assert s[0] == 0.0 and s[-1] == 1.0, s
    print("PASS test_soft_threshold_is_monotonic_and_flat_outside")


def test_apply_gate_soft_attenuates_smoothly():
    rewards = np.array([1.0, 1.0, 1.0, 1.0])
    values = np.array([-7.0, -5.0, -4.0, -3.0])
    mask = np.array([True, True, True, True])
    gated, factors, pass_mask = apply_gate(
        rewards, values, mask, mode="soft", low=-6.0, high=-4.0, fail_reward=FAIL)
    assert np.allclose(factors, [0.0, 0.5, 1.0, 1.0]), factors
    assert np.allclose(gated, [FAIL, 0.5, 1.0, 1.0]), gated
    assert pass_mask.tolist() == [False, False, True, True]
    print("PASS test_apply_gate_soft_attenuates_smoothly")


def test_apply_gate_clamp_caps_and_is_flat_above_threshold():
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.2, 0.91, 0.95])
    mask = np.array([True, True, True])
    gated, factors, pass_mask = apply_gate(
        rewards, values, mask, mode="clamp", threshold=0.9, fail_reward=FAIL)
    # below threshold the true probability passes through; above it is flat at 0.9
    assert np.allclose(factors, [0.2, 0.9, 0.9]), factors
    assert np.allclose(gated, [0.2, 0.9, 0.9]), gated
    assert pass_mask.tolist() == [False, True, True]
    print("PASS test_apply_gate_clamp_caps_and_is_flat_above_threshold")


def test_apply_gate_hard_steps_and_floors():
    rewards = np.array([0.8, 0.4, 0.2])
    values = np.array([0.91, 0.10, 0.90])
    mask = np.array([True, True, True])
    gated, factors, pass_mask = apply_gate(
        rewards, values, mask, mode="hard", threshold=0.9, fail_reward=FAIL)
    assert np.allclose(factors, [1.0, 0.0, 1.0]), factors
    assert np.allclose(gated, [0.8, FAIL, 0.2]), gated
    assert pass_mask.tolist() == [True, False, True]
    print("PASS test_apply_gate_hard_steps_and_floors")


def test_apply_gate_fails_closed_on_score_failures():
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.95, np.nan, 0.95])
    mask = np.array([True, False, True])
    gated, factors, pass_mask = apply_gate(
        rewards, values, mask, mode="clamp", threshold=0.9, fail_reward=FAIL)
    assert np.allclose(factors, [0.9, 0.0, 0.9]), factors
    assert np.allclose(gated, [0.9, FAIL, 0.9]), gated
    assert pass_mask.tolist() == [True, False, True]
    print("PASS test_apply_gate_fails_closed_on_score_failures")


def test_apply_gate_unknown_mode_raises():
    try:
        apply_gate(np.array([1.0]), np.array([0.5]), np.array([True]),
                   mode="bogus", threshold=0.5, fail_reward=FAIL)
    except ValueError as exc:
        assert "bogus" in str(exc), str(exc)
    else:
        raise AssertionError("apply_gate must reject unknown modes")
    print("PASS test_apply_gate_unknown_mode_raises")


if __name__ == "__main__":
    test_soft_threshold_edges_and_midpoint()
    test_soft_threshold_is_monotonic_and_flat_outside()
    test_apply_gate_soft_attenuates_smoothly()
    test_apply_gate_clamp_caps_and_is_flat_above_threshold()
    test_apply_gate_hard_steps_and_floors()
    test_apply_gate_fails_closed_on_score_failures()
    test_apply_gate_unknown_mode_raises()
    print("ALL PASS")

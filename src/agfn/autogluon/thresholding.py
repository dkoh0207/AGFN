"""Shared transition functions that turn a predictor score into a reward gate factor.

Both the BBB and solubility constraints reduce a per-molecule predictor value to a factor
``s in [0, 1]`` that multiplies the docking task reward. Three modes cover the predictor
types we use:

- ``soft``  : smoothstep over ``[low, high]`` for *raw regression* outputs (e.g. LogS) that
              need normalizing into ``[0, 1]``. Higher value = better.
- ``clamp`` : ``min(value, threshold)`` for predictors that already emit a normalized
              probability (e.g. BBB ``predict_proba``). The true score passes through below
              the threshold and saturates above it, so the sampler gets a gradient toward the
              threshold but no incentive to over-optimize past it.
- ``hard``  : a step at ``threshold`` (the original BBB floor-to-``fail_reward`` behaviour).

Rows whose scoring failed (``score_mask`` is ``False``) always collapse to factor ``0`` so
the reward fails closed.
"""

import numpy as np


def soft_threshold(x, low, high):
    """Smoothstep ``3t**2 - 2t**3`` rescaled so ``x<=low -> 0`` and ``x>=high -> 1``."""
    x = np.asarray(x, dtype=np.float64)
    s = np.zeros_like(x)
    s[x >= high] = 1.0
    mask = (x > low) & (x < high)
    t = (x[mask] - low) / (high - low)
    s[mask] = 3.0 * t**2 - 2.0 * t**3
    return s


def apply_gate(task_rewards, values, score_mask, *, mode, threshold=None,
               low=None, high=None, fail_reward):
    """Gate ``task_rewards`` by ``values`` and return ``(gated, factors, pass_mask)``.

    ``factors`` is the per-row multiplier in ``[0, 1]`` (``0`` for failed rows), ``gated`` is
    ``max(task_rewards * factors, fail_reward)``, and ``pass_mask`` flags rows that scored and
    cleared the gate's upper edge.
    """
    task_rewards = np.asarray(task_rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    score_mask = np.asarray(score_mask, dtype=bool)

    if mode == "soft":
        factors = soft_threshold(values, low, high)
        edge = high
    elif mode == "clamp":
        factors = np.minimum(values, threshold)
        edge = threshold
    elif mode == "hard":
        factors = np.where(np.isfinite(values) & (values >= threshold), 1.0, 0.0)
        edge = threshold
    else:
        raise ValueError(f"unknown gate mode {mode!r}")

    factors = np.where(score_mask, factors, 0.0)
    factors = np.nan_to_num(factors, nan=0.0)
    gated = np.maximum(task_rewards * factors, fail_reward)
    pass_mask = score_mask & np.isfinite(values) & (values >= edge)
    return gated, factors, pass_mask

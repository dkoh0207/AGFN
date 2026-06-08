"""Tests for the configurable DiversityFilter identity + fail-fast config guard.

No pytest in this env, so this is a plain script: run it directly, it raises on
failure and prints PASS lines on success.

    python tests/test_diversity_filter.py

Covers:
  1. ``DiversityFilter`` identity selector: ``scaffold`` (default, unchanged) vs
     ``canonical_smiles`` (whole-molecule dedup, the seed-run fix).
  2. The hard-cap penalty threshold (strict ``>`` ``bucket_size``).
  3. The ``max_unique`` memory cap with lowest-count eviction.
  4. ``validate_diversity_filter_config`` fail-fast on the degenerate combo
     (scaffold identity + diversity filter + a seed/frozen-core init).
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.helpers import DiversityFilter  # noqa: E402

EPS = float(np.finfo(float).eps)

# Distinct benzene derivatives -> all share the Bemis-Murcko scaffold ``c1ccccc1``.
SAME_SCAFFOLD = [
    "OCc1ccccc1", "NCc1ccccc1", "COc1ccccc1", "CCc1ccccc1", "CNc1ccccc1", "SCc1ccccc1",
    "NNc1ccccc1", "ONc1ccccc1", "FCc1ccccc1", "ClCc1ccccc1", "Cc1ccccc1", "Nc1ccccc1",
]


def _rewards(n, val=0.5):
    return torch.full((n, 1), float(val))


def _is_penalized(x):
    return float(x) < 1e-6  # truncated to ~eps


def _not_penalized(x, val=0.5):
    return abs(float(x) - val) < 1e-9


def test_canonical_smiles_mode_keeps_distinct_same_scaffold():
    """The seed-run fix: distinct molecules sharing one Murcko scaffold, each seen once,
    must NOT be penalized when keying on the whole molecule."""
    df = DiversityFilter(identity="canonical_smiles")
    df.update(SAME_SCAFFOLD)
    out = df.penalize_reward(SAME_SCAFFOLD, _rewards(len(SAME_SCAFFOLD)))
    assert all(_not_penalized(out[i]) for i in range(len(SAME_SCAFFOLD))), \
        "distinct molecules (same scaffold) must pass through un-penalized in canonical_smiles mode"


def test_default_identity_is_scaffold():
    """Default behavior unchanged: scaffold keying penalizes >bucket_size molecules that
    share a scaffold, even when each molecule is distinct."""
    df = DiversityFilter()  # default identity == scaffold, bucket_size == 10
    df.update(SAME_SCAFFOLD)  # scaffold c1ccccc1 count == 12 > 10
    out = df.penalize_reward(SAME_SCAFFOLD, _rewards(len(SAME_SCAFFOLD)))
    assert all(_is_penalized(out[i]) for i in range(len(SAME_SCAFFOLD))), \
        "scaffold mode must penalize molecules whose shared scaffold exceeds bucket_size"


def test_canonical_smiles_exact_repeat_penalized():
    df = DiversityFilter(identity="canonical_smiles", bucket_size=10)
    smi = "OCc1ccccc1"
    for _ in range(11):  # count 11 > 10
        df.update([smi])
    out = df.penalize_reward([smi], _rewards(1))
    assert _is_penalized(out[0]), "an exact molecule generated > bucket_size times must be penalized"


def test_at_threshold_not_penalized():
    df = DiversityFilter(identity="canonical_smiles", bucket_size=10)
    smi = "OCc1ccccc1"
    for _ in range(10):  # count == 10, not > 10
        df.update([smi])
    out = df.penalize_reward([smi], _rewards(1))
    assert _not_penalized(out[0]), "exactly bucket_size occurrences must not be penalized (strict >)"


def test_invalid_identity_raises():
    try:
        DiversityFilter(identity="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown identity")


def test_max_unique_evicts_lowest_count():
    """The cap keeps the dict bounded and retains over-threshold (high-count) molecules,
    evicting only the diverse count-1 ones."""
    max_unique = 5
    df = DiversityFilter(identity="canonical_smiles", bucket_size=3, max_unique=max_unique)
    hot = "OCc1ccccc1"
    for _ in range(10):  # high count -> must be retained + penalized
        df.update([hot])
    for i in range(1, 30):  # many distinct count-1 molecules -> exceed the cap
        df.update(["C" + "C" * i + "O"])

    prune_at = max_unique + max(1, max_unique // 10)
    assert len(df.bucket_history) <= prune_at, \
        f"bucket_history exceeded the cap: {len(df.bucket_history)} > {prune_at}"
    assert df.bucket_history.get(hot, 0) > df.bucket_size, "the over-threshold molecule must be retained"
    out = df.penalize_reward([hot], _rewards(1))
    assert _is_penalized(out[0]), "the over-threshold molecule must remain penalized after eviction"


def test_guard_raises_scaffold_plus_seed():
    from utils.helpers import validate_diversity_filter_config
    try:
        validate_diversity_filter_config({"diversity_filter": True}, seed_configured=True)
    except ValueError:
        return
    raise AssertionError("expected ValueError: scaffold identity + diversity filter + seed")


def test_guard_ok_canonical_smiles_plus_seed():
    from utils.helpers import validate_diversity_filter_config
    validate_diversity_filter_config(
        {"diversity_filter": True, "diversity_filter_identity": "canonical_smiles"},
        seed_configured=True,
    )  # must not raise


def test_guard_ok_scaffold_without_seed():
    from utils.helpers import validate_diversity_filter_config
    validate_diversity_filter_config({"diversity_filter": True}, seed_configured=False)  # must not raise


def test_guard_ok_filter_disabled_with_seed():
    from utils.helpers import validate_diversity_filter_config
    validate_diversity_filter_config({"diversity_filter": False}, seed_configured=True)  # must not raise


if __name__ == "__main__":
    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_"))
    for name in tests:
        getattr(mod, name)()
        print(f"PASS {name}")
    print(f"\nAll {len(tests)} diversity-filter tests passed.")

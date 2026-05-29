"""Standalone tests for the AutoGluon BBB denovo constraint.

Run directly:

    cd src && python ../tests/test_bbb_constraint.py
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "src")
DENOVO = os.path.join(SRC, "apps", "docking", "denovo")
sys.path.insert(0, SRC)
sys.path.insert(0, DENOVO)

from agfn.autogluon.bbb import AutoGluonBBBScorer  # noqa: E402
from agfn.autogluon.featurization import generate_features  # noqa: E402
from reward_dock import RewardDockFineTune  # noqa: E402


class _Predictor:
    def __init__(self):
        self.last_features = None

    def predict_proba(self, features):
        self.last_features = features
        return pd.DataFrame({0: [0.8, 0.05], 1: [0.2, 0.95]})


class _StringLabelPredictor:
    def predict_proba(self, features):
        return pd.DataFrame({"0": [0.8], "1": [0.2]})


class _DynamicPredictor:
    def __init__(self):
        self.last_features = None

    def predict_proba(self, features):
        self.last_features = features
        return pd.DataFrame({0: [0.8] * len(features), 1: [0.2] * len(features)})


class _UniDock:
    def calculate_rewards(self, smiles):
        return None, [-7.0, -6.0, -8.0], [0.8, 0.4, 0.2]


class _BBBScorer:
    def score_smiles(self, smiles):
        return np.array([0.91, 0.10, 0.90])


class _FailingRowsBBBScorer:
    def score_smiles_with_failures(self, smiles):
        return np.array([0.91, np.nan, 0.10]), np.array([True, False, True])


class _RaisingBBBScorer:
    def score_smiles(self, smiles):
        raise AssertionError("BBB scorer should not be called when constraint is disabled")


def _bare_reward(*, constraint):
    reward = RewardDockFineTune.__new__(RewardDockFineTune)
    reward.unidock = _UniDock()
    reward.bbb_constraint = constraint
    reward.bbb_threshold = 0.9
    reward.bbb_fail_reward = 1e-30
    reward.bbb_scorer = _BBBScorer() if constraint else _RaisingBBBScorer()
    reward.last_bbb_probabilities = None
    reward.last_bbb_score_mask = None
    reward.last_bbb_pass_mask = None
    return reward


def test_feature_shape_and_no_files_written():
    with tempfile.TemporaryDirectory() as tmpdir:
        before = set(os.listdir(tmpdir))
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            features = generate_features(["CCO", "c1ccccc1"])
        finally:
            os.chdir(old_cwd)
        after = set(os.listdir(tmpdir))
    assert after == before, f"feature generation wrote files: {after - before}"
    assert features.shape == (2, 167 + len(Descriptors.descList)), features.shape
    print("PASS test_feature_shape_and_no_files_written")


def test_positive_probability_selection():
    predictor = _Predictor()
    scorer = AutoGluonBBBScorer("unused", positive_class=1, predictor=predictor)
    scores = scorer.score_smiles(["CC", "O"])
    assert np.allclose(scores, [0.2, 0.95]), scores
    assert predictor.last_features.shape[0] == 2
    print("PASS test_positive_probability_selection")


def test_string_positive_probability_selection():
    scorer = AutoGluonBBBScorer("unused", positive_class=1, predictor=_StringLabelPredictor())
    scores = scorer.score_smiles(["CC"])
    assert np.allclose(scores, [0.2]), scores
    print("PASS test_string_positive_probability_selection")


def test_string_config_can_select_integer_probability_column():
    scorer = AutoGluonBBBScorer("unused", positive_class="1", predictor=_Predictor())
    scores = scorer.score_smiles(["CC", "O"])
    assert np.allclose(scores, [0.2, 0.95]), scores
    print("PASS test_string_config_can_select_integer_probability_column")


def test_invalid_bbb_feature_rows_are_not_predicted():
    predictor = _DynamicPredictor()
    scorer = AutoGluonBBBScorer("unused", positive_class=1, predictor=predictor)
    scores, score_mask = scorer.score_smiles_with_failures(["CC", "not_a_smiles"])
    assert predictor.last_features.shape[0] == 1
    assert np.allclose(scores[:1], [0.2]), scores
    assert np.isnan(scores[1]), scores
    assert score_mask.tolist() == [True, False]
    print("PASS test_invalid_bbb_feature_rows_are_not_predicted")


def test_bbb_gate_floors_failing_rewards():
    reward = _bare_reward(constraint=True)
    mols = [Chem.MolFromSmiles(s) for s in ["CC", "O", "N"]]
    flat_rewards, docking_scores = reward.task_reward("QedxSaxDock", mols)
    assert np.allclose(flat_rewards.reshape(-1), [0.8, 1e-30, 0.2]), flat_rewards
    assert np.allclose(docking_scores, [-7.0, -6.0, -8.0]), docking_scores
    assert np.allclose(reward.last_bbb_probabilities, [0.91, 0.10, 0.90])
    assert reward.last_bbb_pass_mask.tolist() == [True, False, True]
    print("PASS test_bbb_gate_floors_failing_rewards")


def test_bbb_score_failures_are_reward_failures():
    reward = _bare_reward(constraint=True)
    reward.bbb_scorer = _FailingRowsBBBScorer()
    mols = [Chem.MolFromSmiles(s) for s in ["CC", "O", "N"]]
    flat_rewards, docking_scores = reward.task_reward("QedxSaxDock", mols)
    assert np.allclose(flat_rewards.reshape(-1), [0.8, 1e-30, 1e-30]), flat_rewards
    assert np.allclose(docking_scores, [-7.0, -6.0, -8.0]), docking_scores
    assert np.allclose(reward.last_bbb_probabilities, [0.91, np.nan, 0.10], equal_nan=True)
    assert reward.last_bbb_score_mask.tolist() == [True, False, True]
    assert reward.last_bbb_pass_mask.tolist() == [True, False, False]
    print("PASS test_bbb_score_failures_are_reward_failures")


def test_bbb_clamp_mode_caps_and_is_flat_above_threshold():
    reward = _bare_reward(constraint=True)
    reward.bbb_mode = "clamp"
    mols = [Chem.MolFromSmiles(s) for s in ["CC", "O", "N"]]
    flat_rewards, _ = reward.task_reward("QedxSaxDock", mols)
    # probs [0.91, 0.10, 0.90] clamped at 0.9 -> factors [0.9, 0.10, 0.9]
    # docking rewards [0.8, 0.4, 0.2] -> [0.72, 0.04, 0.18]
    assert np.allclose(flat_rewards.reshape(-1), [0.72, 0.04, 0.18]), flat_rewards
    assert reward.last_bbb_pass_mask.tolist() == [True, False, True]
    print("PASS test_bbb_clamp_mode_caps_and_is_flat_above_threshold")


def test_missing_autogluon_raises_instead_of_silent_nan():
    """A missing autogluon dependency must abort loudly at construction.

    Regression for the bug where the lazy predictor's ImportError was swallowed by the
    per-row ``except Exception`` in ``_predict_feature_rows``, so a run with autogluon
    absent silently scored every molecule NaN (metrics logged 0.0 / nan) instead of
    crashing. We block the autogluon import regardless of whether it is installed in the
    active env.
    """
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "autogluon" or name.startswith("autogluon."):
            raise ImportError("autogluon blocked for this test")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _blocking_import
    try:
        try:
            AutoGluonBBBScorer("./nonexistent_bbb_model")
        except ImportError as exc:
            assert "autogluon" in str(exc).lower(), str(exc)
        else:
            raise AssertionError(
                "AutoGluonBBBScorer must raise ImportError when autogluon is missing, "
                "not construct successfully and degrade to NaN scores")
    finally:
        builtins.__import__ = real_import
    print("PASS test_missing_autogluon_raises_instead_of_silent_nan")


def test_default_path_does_not_call_bbb_scorer():
    reward = _bare_reward(constraint=False)
    mols = [Chem.MolFromSmiles(s) for s in ["CC", "O", "N"]]
    flat_rewards, docking_scores = reward.task_reward("QedxSaxDock", mols)
    assert np.allclose(flat_rewards.reshape(-1), [0.8, 0.4, 0.2]), flat_rewards
    assert np.allclose(docking_scores, [-7.0, -6.0, -8.0]), docking_scores
    assert reward.last_bbb_probabilities is None
    assert reward.last_bbb_pass_mask is None
    print("PASS test_default_path_does_not_call_bbb_scorer")


if __name__ == "__main__":
    test_feature_shape_and_no_files_written()
    test_positive_probability_selection()
    test_string_positive_probability_selection()
    test_string_config_can_select_integer_probability_column()
    test_invalid_bbb_feature_rows_are_not_predicted()
    test_bbb_gate_floors_failing_rewards()
    test_bbb_score_failures_are_reward_failures()
    test_bbb_clamp_mode_caps_and_is_flat_above_threshold()
    test_missing_autogluon_raises_instead_of_silent_nan()
    test_default_path_does_not_call_bbb_scorer()
    print("ALL PASS")

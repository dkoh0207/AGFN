"""Standalone tests for the AutoGluon solubility (LogS) denovo gate.

Run directly:

    cd src && python ../tests/test_solubility_constraint.py
"""

import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "src")
DENOVO = os.path.join(SRC, "apps", "docking", "denovo")
sys.path.insert(0, SRC)
sys.path.insert(0, DENOVO)

from agfn.autogluon.solubility import AutoGluonSolubilityScorer  # noqa: E402
from reward_dock import RewardDockFineTune  # noqa: E402

FAIL = 1e-30


class _RegPredictor:
    """Regression stub returning a fixed LogS per row."""

    def __init__(self, preds):
        self.preds = list(preds)
        self.last_features = None

    def predict(self, features):
        self.last_features = features
        return pd.Series(self.preds[: len(features)])


class _UniDock:
    def calculate_rewards(self, smiles):
        return None, [-7.0, -6.0, -8.0], [1.0, 1.0, 1.0]


class _SolScorer:
    """LogS values: insoluble, borderline (midpoint), soluble."""

    def score_smiles_with_failures(self, smiles):
        return np.array([-7.0, -5.0, -3.0]), np.array([True, True, True])


class _FailingSolScorer:
    def score_smiles_with_failures(self, smiles):
        return np.array([-3.0, np.nan, -3.0]), np.array([True, False, True])


class _RaisingSolScorer:
    def score_smiles_with_failures(self, smiles):
        raise AssertionError("solubility scorer must not be called when constraint is off")


def _bare_reward(*, sol_constraint, scorer=None):
    reward = RewardDockFineTune.__new__(RewardDockFineTune)
    reward.unidock = _UniDock()
    reward.bbb_constraint = False
    reward.sol_constraint = sol_constraint
    reward.sol_mode = "soft"
    reward.sol_low = -6.0
    reward.sol_high = -4.0
    reward.sol_threshold = -4.0
    reward.sol_fail_reward = FAIL
    reward.sol_scorer = (scorer or _SolScorer()) if sol_constraint else _RaisingSolScorer()
    for attr in ("last_sol_values", "last_sol_score_mask",
                 "last_sol_pass_mask", "last_sol_factors"):
        setattr(reward, attr, None)
    return reward


# --- scorer ---------------------------------------------------------------

def test_solubility_predict_returns_logs():
    predictor = _RegPredictor([-0.7, -5.2])
    scorer = AutoGluonSolubilityScorer("unused", predictor=predictor)
    scores = scorer.score_smiles(["CCO", "c1ccccc1C(=O)O"])
    assert np.allclose(scores, [-0.7, -5.2]), scores
    assert predictor.last_features.shape[0] == 2
    print("PASS test_solubility_predict_returns_logs")


def test_invalid_solubility_rows_are_not_predicted():
    predictor = _RegPredictor([-0.7])
    scorer = AutoGluonSolubilityScorer("unused", predictor=predictor)
    scores, score_mask = scorer.score_smiles_with_failures(["CCO", "not_a_smiles"])
    assert predictor.last_features.shape[0] == 1
    assert np.allclose(scores[:1], [-0.7]), scores
    assert np.isnan(scores[1]), scores
    assert score_mask.tolist() == [True, False]
    print("PASS test_invalid_solubility_rows_are_not_predicted")


def test_missing_autogluon_raises_instead_of_silent_nan():
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "autogluon" or name.startswith("autogluon."):
            raise ImportError("autogluon blocked for this test")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _blocking_import
    try:
        try:
            AutoGluonSolubilityScorer("./nonexistent_sol_model")
        except ImportError as exc:
            assert "autogluon" in str(exc).lower(), str(exc)
        else:
            raise AssertionError(
                "AutoGluonSolubilityScorer must raise ImportError when autogluon is missing")
    finally:
        builtins.__import__ = real_import
    print("PASS test_missing_autogluon_raises_instead_of_silent_nan")


# --- gate integration -----------------------------------------------------

def test_solubility_soft_gate_attenuates_rewards():
    reward = _bare_reward(sol_constraint=True)
    mols = [Chem.MolFromSmiles(s) for s in ["CC", "O", "N"]]
    flat_rewards, docking_scores = reward.task_reward("QedxSaxDock", mols)
    # LogS [-7,-5,-3] under soft[-6,-4] -> factors [0, 0.5, 1] on docking rewards [1,1,1]
    assert np.allclose(flat_rewards.reshape(-1), [FAIL, 0.5, 1.0]), flat_rewards
    assert np.allclose(docking_scores, [-7.0, -6.0, -8.0]), docking_scores
    assert np.allclose(reward.last_sol_values, [-7.0, -5.0, -3.0])
    assert reward.last_sol_pass_mask.tolist() == [False, False, True]
    print("PASS test_solubility_soft_gate_attenuates_rewards")


def test_solubility_score_failures_are_reward_failures():
    reward = _bare_reward(sol_constraint=True, scorer=_FailingSolScorer())
    mols = [Chem.MolFromSmiles(s) for s in ["CC", "O", "N"]]
    flat_rewards, _ = reward.task_reward("QedxSaxDock", mols)
    # row 1 failed scoring -> factor 0 -> floored; rows 0,2 soluble -> factor 1
    assert np.allclose(flat_rewards.reshape(-1), [1.0, FAIL, 1.0]), flat_rewards
    assert reward.last_sol_score_mask.tolist() == [True, False, True]
    print("PASS test_solubility_score_failures_are_reward_failures")


def test_default_path_does_not_call_solubility_scorer():
    reward = _bare_reward(sol_constraint=False)
    mols = [Chem.MolFromSmiles(s) for s in ["CC", "O", "N"]]
    flat_rewards, _ = reward.task_reward("QedxSaxDock", mols)
    assert np.allclose(flat_rewards.reshape(-1), [1.0, 1.0, 1.0]), flat_rewards
    assert reward.last_sol_values is None
    print("PASS test_default_path_does_not_call_solubility_scorer")


if __name__ == "__main__":
    test_solubility_predict_returns_logs()
    test_invalid_solubility_rows_are_not_predicted()
    test_missing_autogluon_raises_instead_of_silent_nan()
    test_solubility_soft_gate_attenuates_rewards()
    test_solubility_score_failures_are_reward_failures()
    test_default_path_does_not_call_solubility_scorer()
    print("ALL PASS")

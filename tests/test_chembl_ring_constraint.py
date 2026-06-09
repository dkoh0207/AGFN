"""Standalone tests for the ChEMBL ring-system rarity denovo constraint.

Run directly:

    cd src && python ../tests/test_chembl_ring_constraint.py
"""

import os
import sys

import numpy as np
from rdkit import Chem

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "src")
DENOVO = os.path.join(SRC, "apps", "docking", "denovo")
sys.path.insert(0, SRC)
sys.path.insert(0, DENOVO)

from agfn.autogluon.thresholding import apply_gate  # noqa: E402
from agfn.ring_system import RingSystemScorer, ACYCLIC_SENTINEL  # noqa: E402
from reward_dock import RewardDockFineTune  # noqa: E402

FAIL = 1e-30


class _UniDock:
    def calculate_rewards(self, smiles):
        return None, [-7.0, -6.0, -8.0], [0.8, 0.4, 0.2]


class _FakeLookup:
    """Stand-in for useful_rdkit_utils.RingSystemLookup keyed by SMILES."""

    def __init__(self, table):
        self.table = table  # {smiles: [(ring_smiles, count), ...]}

    def process_mol(self, mol):
        return self.table[Chem.MolToSmiles(mol)]


class _RingScorer:
    def __init__(self, values, mask=None):
        self.values = np.asarray(values, dtype=np.float64)
        self.mask = np.ones(len(values), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    def score_smiles_with_failures(self, smiles):
        return self.values, self.mask


class _RaisingRingScorer:
    def score_smiles_with_failures(self, smiles):
        raise AssertionError("ring scorer should not be called when constraint is disabled")


def _bare_reward(*, constraint, scorer=None):
    reward = RewardDockFineTune.__new__(RewardDockFineTune)
    reward.unidock = _UniDock()
    # task_reward also reads the BBB/solubility flags; keep those gates off for these tests.
    reward.bbb_constraint = False
    reward.sol_constraint = False
    reward.chembl_ring_constraint = constraint
    reward.chembl_ring_mode = "hard"
    reward.chembl_ring_min_count = 5
    reward.chembl_ring_low = None
    reward.chembl_ring_high = None
    reward.chembl_ring_fail_reward = FAIL
    reward.chembl_ring_scorer = scorer if scorer is not None else _RaisingRingScorer()
    reward.last_chembl_ring_values = None
    reward.last_chembl_ring_score_mask = None
    reward.last_chembl_ring_pass_mask = None
    return reward


def test_hard_gate_floors_rare_rings():
    # common (10), rare (2 < 5), acyclic (sentinel) -> factors [1, 0, 1]
    scorer = _RingScorer([10, 2, ACYCLIC_SENTINEL])
    reward = _bare_reward(constraint=True, scorer=scorer)
    mols = [Chem.MolFromSmiles(s) for s in ["c1ccccc1C", "CC", "CCO"]]
    flat_rewards, docking_scores = reward.task_reward("QedxSaxDock", mols)
    assert np.allclose(flat_rewards.reshape(-1), [0.8, FAIL, 0.2]), flat_rewards
    assert np.allclose(docking_scores, [-7.0, -6.0, -8.0]), docking_scores
    assert reward.last_chembl_ring_pass_mask.tolist() == [True, False, True]
    print("PASS test_hard_gate_floors_rare_rings")


def test_acyclic_sentinel_passes_gate():
    """Acyclic sentinel must be finite so hard mode's np.isfinite guard lets it pass."""
    assert np.isfinite(ACYCLIC_SENTINEL), ACYCLIC_SENTINEL
    scorer = _RingScorer([ACYCLIC_SENTINEL, ACYCLIC_SENTINEL, ACYCLIC_SENTINEL])
    reward = _bare_reward(constraint=True, scorer=scorer)
    mols = [Chem.MolFromSmiles(s) for s in ["CC", "CCO", "CCN"]]
    flat_rewards, _ = reward.task_reward("QedxSaxDock", mols)
    assert np.allclose(flat_rewards.reshape(-1), [0.8, 0.4, 0.2]), flat_rewards
    assert reward.last_chembl_ring_pass_mask.tolist() == [True, True, True]
    print("PASS test_acyclic_sentinel_passes_gate")


def test_ring_score_failures_are_reward_failures():
    # NaN + mask False fails closed; a finite-but-rare value (3 < 5) also fails the gate.
    scorer = _RingScorer([10, np.nan, 3], mask=[True, False, True])
    reward = _bare_reward(constraint=True, scorer=scorer)
    mols = [Chem.MolFromSmiles(s) for s in ["c1ccccc1C", "CC", "CCO"]]
    flat_rewards, _ = reward.task_reward("QedxSaxDock", mols)
    assert np.allclose(flat_rewards.reshape(-1), [0.8, FAIL, FAIL]), flat_rewards
    assert reward.last_chembl_ring_score_mask.tolist() == [True, False, True]
    assert reward.last_chembl_ring_pass_mask.tolist() == [True, False, False]
    print("PASS test_ring_score_failures_are_reward_failures")


def test_default_path_does_not_call_ring_scorer():
    reward = _bare_reward(constraint=False)
    mols = [Chem.MolFromSmiles(s) for s in ["c1ccccc1C", "CC", "CCO"]]
    flat_rewards, docking_scores = reward.task_reward("QedxSaxDock", mols)
    assert np.allclose(flat_rewards.reshape(-1), [0.8, 0.4, 0.2]), flat_rewards
    assert np.allclose(docking_scores, [-7.0, -6.0, -8.0]), docking_scores
    assert reward.last_chembl_ring_values is None
    assert reward.last_chembl_ring_pass_mask is None
    print("PASS test_default_path_does_not_call_ring_scorer")


def test_scorer_with_injected_fake_lookup():
    """RingSystemScorer min-over-rings + acyclic-sentinel logic, no data download needed."""
    table = {
        Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1CCC2CCNCC2")): [("c1ccccc1", 2568039), ("C1CCNCC1", 212367)],
        Chem.MolToSmiles(Chem.MolFromSmiles("O=C1CC2(C1)C1=CC1C2")): [("O=C1CC2(C1)CC1C=C12", 0)],
        Chem.MolToSmiles(Chem.MolFromSmiles("CC")): [],
    }
    scorer = RingSystemScorer(lookup=_FakeLookup(table))
    smis = ["c1ccccc1CCC2CCNCC2", "O=C1CC2(C1)C1=CC1C2", "CC"]
    values, mask = scorer.score_smiles_with_failures(smis)
    assert values[0] == 212367 and values[1] == 0 and values[2] == ACYCLIC_SENTINEL, values
    assert mask.tolist() == [True, True, True]
    # invalid SMILES -> NaN + fail-closed
    values, mask = scorer.score_smiles_with_failures(["not_a_smiles"])
    assert np.isnan(values[0]) and mask.tolist() == [False]
    print("PASS test_scorer_with_injected_fake_lookup")


def test_real_ring_lookup_smoke():
    """End-to-end against the real ChEMBL ring DB; skipped if the data isn't available."""
    try:
        scorer = RingSystemScorer()
    except Exception as exc:  # missing package or uncached CSV (e.g. offline)
        print(f"SKIP test_real_ring_lookup_smoke ({type(exc).__name__}: {exc})")
        return
    smis = ["c1ccccc1CCC2CCNCC2", "O=C1CC2(C1)C1=CC1C2", "CC"]
    values, mask = scorer.score_smiles_with_failures(smis)
    assert mask.all(), mask
    assert values[0] >= 5, f"common ring should be frequent: {values[0]}"
    assert values[1] < 5, f"strained ring should be rare: {values[1]}"
    assert values[2] == ACYCLIC_SENTINEL, values[2]
    factors = apply_gate(np.ones(3), values, mask, mode="hard", threshold=5, fail_reward=FAIL)[1]
    assert factors.tolist() == [1.0, 0.0, 1.0], factors
    print("PASS test_real_ring_lookup_smoke")


if __name__ == "__main__":
    test_hard_gate_floors_rare_rings()
    test_acyclic_sentinel_passes_gate()
    test_ring_score_failures_are_reward_failures()
    test_default_path_does_not_call_ring_scorer()
    test_scorer_with_injected_fake_lookup()
    test_real_ring_lookup_smoke()
    print("ALL PASS")

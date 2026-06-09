import numpy as np
import os
import sys
from rdkit import Chem
from agfn.autogluon.bbb import AutoGluonBBBScorer
from agfn.autogluon.solubility import AutoGluonSolubilityScorer
from agfn.autogluon.thresholding import apply_gate
from agfn.ring_system import RingSystemScorer
from agfn.reward import Reward

class RewardDockFineTune(Reward):
    def __init__(self, cond_range_dict, ft_cond_dict,cond_prop_var, reward_aggregation, molenv_dict_path, zinc_rad_scale, hps,gfn_samples_path ) -> None:
        super().__init__(cond_range_dict, cond_prop_var, reward_aggregation, molenv_dict_path, zinc_rad_scale, hps)
        self.hps = hps
        self.gfn_samples_path = gfn_samples_path

        # BBB gate: predict_proba is already in [0,1], so the default mode is "clamp" (cap the
        # score at the threshold) or "hard" (legacy step); "soft" is also available.
        self.bbb_constraint = bool(hps.get("bbb_constraint", False))
        self.bbb_mode = hps.get("bbb_mode", "hard")
        self.bbb_threshold = float(hps.get("bbb_threshold", 0.9))
        self.bbb_low = _opt_float(hps.get("bbb_low"))
        self.bbb_high = _opt_float(hps.get("bbb_high"))
        self.bbb_fail_reward = float(hps.get("bbb_fail_reward", 1e-30))
        self.bbb_scorer = None
        self.last_bbb_probabilities = None
        self.last_bbb_score_mask = None
        self.last_bbb_pass_mask = None
        if self.bbb_constraint:
            self.bbb_scorer = AutoGluonBBBScorer(
                hps["bbb_model_path"],
                positive_class=hps.get("bbb_positive_class", 1),
            )

        # Solubility gate: the model emits raw LogS, so the default mode is "soft" (smoothstep
        # over [sol_low, sol_high] to normalize into [0,1]). Higher LogS = more soluble.
        self.sol_constraint = bool(hps.get("sol_constraint", False))
        self.sol_mode = hps.get("sol_mode", "soft")
        self.sol_threshold = _opt_float(hps.get("sol_threshold"))
        self.sol_low = _opt_float(hps.get("sol_low"))
        self.sol_high = _opt_float(hps.get("sol_high"))
        self.sol_fail_reward = float(hps.get("sol_fail_reward", 1e-30))
        self.sol_scorer = None
        self.last_sol_values = None
        self.last_sol_score_mask = None
        self.last_sol_pass_mask = None
        self.last_sol_factors = None
        if self.sol_constraint:
            self.sol_scorer = AutoGluonSolubilityScorer(hps["sol_model_path"])

        # ChEMBL ring-system rarity gate: the scorer emits the rarest ring system's ChEMBL
        # frequency, so the default mode is "hard" (floor the reward when that frequency is below
        # chembl_ring_min_count). Acyclic molecules score +inf and always pass.
        self.chembl_ring_constraint = bool(hps.get("chembl_ring_constraint", False))
        self.chembl_ring_mode = hps.get("chembl_ring_mode", "hard")
        self.chembl_ring_min_count = float(hps.get("chembl_ring_min_count", 5))
        self.chembl_ring_low = _opt_float(hps.get("chembl_ring_low"))
        self.chembl_ring_high = _opt_float(hps.get("chembl_ring_high"))
        self.chembl_ring_fail_reward = float(hps.get("chembl_ring_fail_reward", 1e-30))
        self.chembl_ring_scorer = None
        self.last_chembl_ring_values = None
        self.last_chembl_ring_score_mask = None
        self.last_chembl_ring_pass_mask = None
        if self.chembl_ring_constraint:
            self.chembl_ring_scorer = RingSystemScorer(
                ring_file=(hps.get("chembl_ring_db_path") or None),
                ignore_stereo=bool(hps.get("chembl_ring_ignore_stereo", False)),
            )

        # Docking runs in-process via Uni-Dock (the unidock_tools API). unidock.py sits next to
        # this package in src/apps/docking/; make it importable regardless of how the driver was
        # launched.
        docking_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if docking_dir not in sys.path:
            sys.path.insert(0, docking_dir)
        from unidock import UniDockGPU
        self.unidock = UniDockGPU(
            target=hps.target_name,
            search_mode=hps.get('unidock_search_mode', 'fast'),
            num_workers=hps.get('unidock_num_workers', 1),
            **dict(hps.target_grid[hps.target_name]),
        )

    def _score_constraint(self, scorer, smiles_list):
        """Return ``(values, score_mask)`` for a gate scorer, fail-closed per row."""
        if hasattr(scorer, "score_smiles_with_failures"):
            values, score_mask = scorer.score_smiles_with_failures(smiles_list)
            return np.asarray(values, dtype=np.float64), np.asarray(score_mask, dtype=bool)
        values = np.asarray(scorer.score_smiles(smiles_list), dtype=np.float64)
        return values, np.isfinite(values)

    def task_reward(self, task, mols):
        """In-process docking via Uni-Dock. Returns (flat_rewards_task, true_task_score)."""
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        outs = self.unidock.calculate_rewards(smiles_list)
        true_task_score = np.array(outs[1])
        task_rewards = np.array(outs[2], dtype=np.float64)

        if self.bbb_constraint:
            values, score_mask = self._score_constraint(self.bbb_scorer, smiles_list)
            task_rewards, _factors, pass_mask = apply_gate(
                task_rewards, values, score_mask,
                mode=getattr(self, "bbb_mode", "hard"),
                threshold=self.bbb_threshold,
                low=getattr(self, "bbb_low", None),
                high=getattr(self, "bbb_high", None),
                fail_reward=self.bbb_fail_reward,
            )
            self.last_bbb_probabilities = values
            self.last_bbb_score_mask = score_mask
            self.last_bbb_pass_mask = pass_mask
        else:
            self.last_bbb_probabilities = None
            self.last_bbb_score_mask = None
            self.last_bbb_pass_mask = None

        if getattr(self, "sol_constraint", False):
            values, score_mask = self._score_constraint(self.sol_scorer, smiles_list)
            task_rewards, factors, pass_mask = apply_gate(
                task_rewards, values, score_mask,
                mode=getattr(self, "sol_mode", "soft"),
                threshold=getattr(self, "sol_threshold", None),
                low=getattr(self, "sol_low", None),
                high=getattr(self, "sol_high", None),
                fail_reward=self.sol_fail_reward,
            )
            self.last_sol_values = values
            self.last_sol_score_mask = score_mask
            self.last_sol_pass_mask = pass_mask
            self.last_sol_factors = factors
        else:
            self.last_sol_values = None
            self.last_sol_score_mask = None
            self.last_sol_pass_mask = None
            self.last_sol_factors = None

        if getattr(self, "chembl_ring_constraint", False):
            values, score_mask = self._score_constraint(self.chembl_ring_scorer, smiles_list)
            task_rewards, _factors, pass_mask = apply_gate(
                task_rewards, values, score_mask,
                mode=getattr(self, "chembl_ring_mode", "hard"),
                threshold=getattr(self, "chembl_ring_min_count", 5),
                low=getattr(self, "chembl_ring_low", None),
                high=getattr(self, "chembl_ring_high", None),
                fail_reward=self.chembl_ring_fail_reward,
            )
            self.last_chembl_ring_values = values
            self.last_chembl_ring_score_mask = score_mask
            self.last_chembl_ring_pass_mask = pass_mask
        else:
            self.last_chembl_ring_values = None
            self.last_chembl_ring_score_mask = None
            self.last_chembl_ring_pass_mask = None

        flat_rewards_task = np.expand_dims(task_rewards, axis=1)
        return flat_rewards_task, true_task_score


def _opt_float(value):
    return None if value is None else float(value)

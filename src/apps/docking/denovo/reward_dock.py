import numpy as np
import os
import sys
from rdkit import Chem
from agfn.reward import Reward

class RewardDockFineTune(Reward):
    def __init__(self, cond_range_dict, ft_cond_dict,cond_prop_var, reward_aggregation, molenv_dict_path, zinc_rad_scale, hps,gfn_samples_path ) -> None:
        super().__init__(cond_range_dict, cond_prop_var, reward_aggregation, molenv_dict_path, zinc_rad_scale, hps)
        self.hps = hps
        self.gfn_samples_path = gfn_samples_path

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

    def task_reward(self, task, mols):
        """In-process docking via Uni-Dock. Returns (flat_rewards_task, true_task_score)."""
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        outs = self.unidock.calculate_rewards(smiles_list)
        true_task_score = np.array(outs[1])
        flat_rewards_task = np.expand_dims(np.array(outs[2]), axis=1)
        return flat_rewards_task, true_task_score

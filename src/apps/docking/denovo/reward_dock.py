import pickle
import numpy as np
import json
import os
import sys
from rdkit import Chem
import subprocess
from agfn.reward import Reward

class RewardDockFineTune(Reward):
    def __init__(self, cond_range_dict, ft_cond_dict,cond_prop_var, reward_aggregation, molenv_dict_path, zinc_rad_scale, hps,gfn_samples_path ) -> None:
        super().__init__(cond_range_dict, cond_prop_var, reward_aggregation, molenv_dict_path, zinc_rad_scale, hps)
        self.hps = hps
        self.gfn_samples_path = gfn_samples_path
        self.vina_path = hps['vina_path']
        with open(f'./data/docking/tmp_config.json', "w") as f:
            json.dump(hps.target_grid, f, indent=4)

        # Docking backend: "vina" (QuickVina2-GPU, run as a subprocess) or
        # "unidock" (Uni-Dock, run in-process via the unidock_tools API).
        self.docking_backend = hps.get('docking_backend', 'vina')
        if self.docking_backend == 'unidock':
            # unidock.py sits next to gpuvina.py in src/apps/docking/; make it importable
            # regardless of how the driver was launched, and only when actually selected.
            docking_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if docking_dir not in sys.path:
                sys.path.insert(0, docking_dir)
            from unidock import UniDockGPU
            self.unidock = UniDockGPU(
                target=hps.target_name,
                search_mode=hps.get('unidock_search_mode', 'fast'),
                num_workers=hps.get('unidock_num_workers', 1),
                unidock_bin_dir=hps.get('unidock_bin_dir', None),
                **dict(hps.target_grid[hps.target_name]),
            )

    def vina_docking_reward(self, mols):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols] 
        outs = self.vina.calculate_rewards(smiles_list)
        return outs
    
    def _unidock_task_reward(self, mols):
        """In-process docking via Uni-Dock. Returns (flat_rewards_task, true_task_score) with
        the same contract as the Vina subprocess branch below."""
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        outs = self.unidock.calculate_rewards(smiles_list)
        true_task_score = np.array(outs[1])
        flat_rewards_task = np.expand_dims(np.array(outs[2]), axis=1)
        return flat_rewards_task, true_task_score

    def task_reward(self, task, mols):
        if self.docking_backend == 'unidock':
            return self._unidock_task_reward(mols)

        vina_docking_cmd = ["python", "./src/apps/docking/gpuvina.py", self.hps.target_name, self.gfn_samples_path, self.vina_path]
        subprocess.run(vina_docking_cmd, check=True)
        with open(f'{self.gfn_samples_path}/{self.hps.target_name}_docked.pkl','rb') as f:
            outs = pickle.load(f)
        true_task_score = np.array(outs[1]) 
        flat_rewards_task = np.expand_dims(np.array(np.array(outs[2])),axis=1)
        return flat_rewards_task, true_task_score  
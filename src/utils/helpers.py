import torch
from gflownet.envs.graph_building_env import GraphActionCategorical
import datamol as dm
import os
from fnmatch import fnmatch
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

def validate_batch( batch, trajs, ctx, model):
    #borrowed from current trunk, sampling_it.py 
    for actions, atypes in [(batch.actions, ctx.action_type_order)] + (
        [(batch.bck_actions, ctx.bck_action_type_order)]
        if hasattr(batch, "bck_actions") and hasattr(ctx, "bck_action_type_order")
        else []
    ):
        # print(actions )
        # print(atypes)
        mask_cat = GraphActionCategorical(
            batch,
            [model._action_type_to_mask(t, batch) for t in atypes],
            [model._action_type_to_key[t] for t in atypes],
            [None for _ in atypes],
        )
        masked_action_is_used = (1 - mask_cat.log_prob(actions, logprobs=mask_cat.logits))* (1 - batch.is_sink)
        num_trajs = len(trajs)
        batch_idx = torch.arange(num_trajs, device=batch.x.device).repeat_interleave(batch.traj_lens)
        first_graph_idx = torch.zeros_like(batch.traj_lens)
        torch.cumsum(batch.traj_lens[:-1], 0, out=first_graph_idx[1:])
        if masked_action_is_used.sum() != 0:
            invalid_idx = masked_action_is_used.argmax().item()
            traj_idx = batch_idx[invalid_idx].item()
            timestep = invalid_idx - first_graph_idx[traj_idx].item()
            raise ValueError("Found an action that was masked out", trajs[traj_idx]["traj"][timestep], trajs[traj_idx]["traj"])

def visualize_trajectory(traj,ctx_mol):
    mols = []
    for item in traj:
        mols.append(ctx_mol.graph_to_mol(item[0]))
    return dm.to_image(mols, mol_size=(200, 200), align=True)

def get_Hfile_paths(root):
    pattern = "*.smi.gz"
    Hfile_paths = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                Hfile_paths.append(os.path.join(path,name))
    # last_file = '/mnt/ps/home/CORP/mohit.pandey/project/files.docking.org/zinc22/2d-all/H04/H04M200.smi.gz'
    # Hfile_paths.remove(last_file)
    # Hfile_paths.append(last_file)
    print('Total Files ', len(Hfile_paths))
    return Hfile_paths

def avg_traj_len(bck_traj_batch, fwd_traj_batch ):
    avg_batch_bck_traj_len, avg_batch_fwd_traj_len, avg_batch_traj_len = 0, 0,0
    for i in range(len(bck_traj_batch)):
        avg_batch_bck_traj_len += len(bck_traj_batch[i]['bck_a'])
    avg_batch_traj_len += avg_batch_bck_traj_len

    for i in range(len(fwd_traj_batch)):
        avg_batch_fwd_traj_len += len(fwd_traj_batch[i]['bck_a'])
    avg_batch_traj_len += avg_batch_fwd_traj_len

    avg_batch_fwd_traj_len = avg_batch_fwd_traj_len/len(fwd_traj_batch)
    avg_batch_bck_traj_len = avg_batch_bck_traj_len/len(bck_traj_batch)
    avg_batch_traj_len = avg_batch_traj_len/ (len(fwd_traj_batch)+len(bck_traj_batch) )
    return avg_batch_fwd_traj_len, avg_batch_bck_traj_len, avg_batch_traj_len

def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

def get_bemis_murcko_scaffold(smiles):
    """
    Get the Bemis-Murcko scaffold: https://pubs.acs.org/doi/10.1021/jm9602928 of a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            scaffold = GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, canonical=True)
        except Exception:
            return ""
    else:
        return ""

class DiversityFilter:
    """
    Diversity Filter (REINVENT-style memory bucket) as described in
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00473-0

    An identity key is counted across the run; once a key has been generated more than
    ``bucket_size`` times, every further occurrence has its reward truncated to ~``eps``.

    ``identity`` selects what counts as "the same" molecule:
      - ``"scaffold"`` (default): the Bemis-Murcko scaffold (original behavior). This is
        degenerate under a fixed seed core -- every molecule shares one scaffold, so after
        ``bucket_size`` molecules essentially everything is penalized -- so prefer
        ``"canonical_smiles"`` for seed / frozen-core runs.
      - ``"canonical_smiles"``: the whole molecule. Callers pass canonical SMILES (the
        iterator already does, via ``Chem.MolToSmiles``), so the SMILES string is used
        directly as the key -- distinct molecules that share a scaffold stay distinct.

    ``max_unique`` bounds ``bucket_history`` (which grows by one entry per distinct key) so a
    long molecule-level run cannot exhaust memory: when the dict exceeds the cap (plus a small
    slack to amortize the prune), the lowest-count entries -- the diverse, sub-threshold
    molecules -- are evicted, retaining the high-count ones we still want to penalize.
    ``None`` disables the cap.
    """

    _IDENTITIES = ("scaffold", "canonical_smiles")

    def __init__(
        self,
        bucket_size=10,
        identity="scaffold",
        max_unique=1_000_000,
    ):
        if identity not in self._IDENTITIES:
            raise ValueError(
                f"DiversityFilter identity must be one of {self._IDENTITIES}, got {identity!r}"
            )
        # Track how many times a given identity key (scaffold or whole molecule) has appeared.
        self.bucket_history = dict()
        self.bucket_size = bucket_size
        self.identity = identity
        self.max_unique = max_unique

    def _key(self, smiles):
        """Identity key for a (canonical) SMILES: its Bemis-Murcko scaffold or the molecule
        itself, per ``self.identity``."""
        if self.identity == "scaffold":
            return get_bemis_murcko_scaffold(smiles)
        return smiles  # canonical_smiles: caller passes canonical SMILES, use it directly

    def _evict_to_cap(self):
        """Bound ``bucket_history`` to ``max_unique`` by dropping the lowest-count keys.

        A small slack margin amortizes the O(n log n) prune across many batches instead of
        running it every batch once the cap is reached. Lowest-count keys are the diverse,
        sub-threshold molecules, so evicting them never weakens the penalty.
        """
        if self.max_unique is None:
            return
        prune_at = self.max_unique + max(1, self.max_unique // 10)
        if len(self.bucket_history) > prune_at:
            kept = sorted(self.bucket_history.items(), key=lambda kv: kv[1], reverse=True)
            self.bucket_history = dict(kept[: self.max_unique])

    def update(self, smiles):
        """Update the bucket history with a sampled (or hallucinated) batch of canonical SMILES."""
        for s in smiles:
            key = self._key(s)
            self.bucket_history[key] = self.bucket_history.get(key, 0) + 1
        self._evict_to_cap()

    def penalize_reward(self, smiles, rewards):
        """Truncate the reward of any molecule whose identity key has been generated more than
        ``bucket_size`` times to ~``eps``; pass the rest through unchanged."""
        if len(smiles) > 0:
            penalized_rewards = []
            for idx, s in enumerate(smiles):
                if self.bucket_history.get(self._key(s), 0) > self.bucket_size:
                    penalized_rewards.append(torch.tensor(0.0 + np.finfo(float).eps).unsqueeze(dim=-1))
                else:
                    penalized_rewards.append(rewards[idx])
            return torch.stack(penalized_rewards)  # stacked_pen_rew
        else:
            return np.array([])


def validate_diversity_filter_config(hps, seed_configured):
    """Fail fast (before training starts) on a degenerate diversity-filter configuration.

    A scaffold-keyed diversity filter combined with a seed / frozen-core initialization is
    degenerate: every generated molecule shares the seed's Bemis-Murcko scaffold, so the
    filter truncates virtually all rewards to ~eps and the reward signal collapses (this is
    the confirmed cause of an empty top-K-by-reward hall of fame). Raise so the run crashes
    during setup instead of silently producing garbage; no-op for any compatible config.
    """
    if (seed_configured and bool(hps.get("diversity_filter", False))
            and hps.get("diversity_filter_identity", "scaffold") == "scaffold"):
        raise ValueError(
            "diversity_filter with BM-scaffold identity is incompatible with a seed / "
            "frozen-core initialization: every generated molecule shares the seed's Murcko "
            "scaffold, so the filter collapses all rewards to ~eps. Set "
            "diversity_filter_identity: canonical_smiles (molecule-level dedup) for seed runs, "
            "or disable diversity_filter."
        )
        

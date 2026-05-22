"""Running top-K tracker for generated molecules across a training run.

Maintains two bounded heaps in parallel:
  - top_by_reward: largest rewards (GFN training signal)
  - top_by_affinity: most-negative Uni-Dock docking scores (kcal/mol)

Stored entries are SMILES-only (no RDKit Mol objects) for portability.
Persisted via torch.save to keep the API consistent with other checkpoint
files in this project. Resumes from disk if a prior dump exists.
"""

import heapq
import os
from typing import Iterable, Optional

import torch


class TopKTracker:
    def __init__(self, ks: Iterable[int] = (10, 100)):
        self.ks = tuple(sorted(ks))
        self.max_k = self.ks[-1]
        self._by_reward: list = []
        self._by_affinity: list = []

    def add(self, smiles: str, reward: float, affinity: Optional[float], iteration: int):
        if not smiles or reward is None:
            return
        reward = float(reward)
        aff = None if affinity is None else float(affinity)
        rew_entry = (reward, smiles, reward, aff, iteration)
        self._push_bounded(self._by_reward, rew_entry)
        if aff is not None:
            aff_entry = (-aff, smiles, reward, aff, iteration)
            self._push_bounded(self._by_affinity, aff_entry)

    def add_batch(self, smiles_list, rewards, affinities=None, iteration: int = 0):
        if affinities is None:
            affinities = [None] * len(smiles_list)
        for s, r, a in zip(smiles_list, rewards, affinities):
            self.add(s, r, a, iteration)

    def _push_bounded(self, heap, entry):
        if len(heap) < self.max_k:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)

    def snapshot(self):
        rew_sorted = sorted(self._by_reward, key=lambda e: e[0], reverse=True)
        aff_sorted = sorted(self._by_affinity, key=lambda e: e[0], reverse=True)
        strip = lambda e: (e[1], e[2], e[3], e[4])
        return {
            "top_by_reward": {k: [strip(e) for e in rew_sorted[:k]] for k in self.ks},
            "top_by_affinity": {k: [strip(e) for e in aff_sorted[:k]] for k in self.ks},
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.snapshot(), path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        snap = torch.load(path, map_location="cpu")
        # both buckets may hold the same entries with different sort orders;
        # also, an entry with no affinity is only in top_by_reward. Merge by
        # (smiles, iteration) so each underlying molecule is pushed exactly once.
        seen = {}
        for bucket_name in ("top_by_reward", "top_by_affinity"):
            rows = snap.get(bucket_name, {}).get(self.max_k, [])
            for smiles, reward, affinity, iteration in rows:
                key = (smiles, iteration)
                if key not in seen:
                    seen[key] = (smiles, reward, affinity, iteration)
        for smiles, reward, affinity, iteration in seen.values():
            self.add(smiles, reward, affinity, iteration)

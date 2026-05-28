"""Running top-K tracker for generated molecules across a training run.

Maintains two bounded heaps in parallel:
  - top_by_reward: largest rewards (GFN training signal)
  - top_by_affinity: most-negative Uni-Dock docking scores (kcal/mol)

Stored entries are SMILES-only (no RDKit Mol objects) for portability.
Persisted as two flat CSVs (``top100_by_reward.csv`` / ``top100_by_affinity.csv``) so the
hall of fame is human-readable and easy to load elsewhere; the same CSVs double as the
resume source. A legacy ``top_k_mols.pt`` dump (the previous ``torch.save`` format) is still
read on resume if the CSVs are absent. The in-memory heaps and the add/snapshot API are
unchanged — only the on-disk format differs.
"""

import csv
import heapq
import os
from typing import Iterable, Optional

import torch

REWARD_CSV = "top100_by_reward.csv"
AFFINITY_CSV = "top100_by_affinity.csv"
LEGACY_PT = "top_k_mols.pt"
_CSV_FIELDS = ("rank", "smiles", "reward", "affinity", "iteration")


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

    def averages(self):
        """Mean reward of the top-10/top-100 (ranked by reward) and mean affinity of the
        top-10/top-100 (ranked by affinity). Returns NaN for an empty bucket, or -- for the
        affinity buckets -- when no entry has a docking score. Cheap: heaps are bounded at
        ``max_k``. Used to log monotonically-improving hall-of-fame curves during training.
        """
        snap = self.snapshot()

        def _mean(entries, idx):
            vals = [e[idx] for e in entries if e[idx] is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        return {
            "top10_reward": _mean(snap["top_by_reward"].get(10, []), 1),
            "top100_reward": _mean(snap["top_by_reward"].get(100, []), 1),
            "top10_affinity": _mean(snap["top_by_affinity"].get(10, []), 2),
            "top100_affinity": _mean(snap["top_by_affinity"].get(100, []), 2),
        }

    def save(self, out_dir: str):
        """Write the two top-``max_k`` buckets to flat CSVs in ``out_dir``."""
        os.makedirs(out_dir or ".", exist_ok=True)
        snap = self.snapshot()
        for fname, bucket in ((REWARD_CSV, "top_by_reward"), (AFFINITY_CSV, "top_by_affinity")):
            rows = snap[bucket][self.max_k]
            with open(os.path.join(out_dir, fname), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(_CSV_FIELDS)
                for rank, (smiles, reward, affinity, iteration) in enumerate(rows, start=1):
                    w.writerow([rank, smiles, reward, "" if affinity is None else affinity, iteration])

    def load(self, out_dir: str):
        """Repopulate the heaps from the CSVs in ``out_dir`` (or a legacy ``.pt`` dump)."""
        # both buckets may hold the same entries with different sort orders; also, an entry
        # with no affinity is only in top_by_reward. Merge by (smiles, iteration) so each
        # underlying molecule is pushed exactly once.
        seen = {}
        reward_csv = os.path.join(out_dir, REWARD_CSV)
        affinity_csv = os.path.join(out_dir, AFFINITY_CSV)
        if os.path.exists(reward_csv) or os.path.exists(affinity_csv):
            for path in (reward_csv, affinity_csv):
                if not os.path.exists(path):
                    continue
                with open(path, newline="") as f:
                    for row in csv.DictReader(f):
                        smiles = row["smiles"]
                        aff = row["affinity"]
                        key = (smiles, int(row["iteration"]))
                        if key not in seen:
                            seen[key] = (smiles, float(row["reward"]),
                                         None if aff == "" else float(aff), int(row["iteration"]))
        else:
            # Back-compat: resume from the previous torch.save dump if present.
            legacy = os.path.join(out_dir, LEGACY_PT)
            if not os.path.exists(legacy):
                return
            # weights_only=True is safe here: the dump is plain dict/list/tuple of str/float/int.
            snap = torch.load(legacy, map_location="cpu", weights_only=True)
            for bucket_name in ("top_by_reward", "top_by_affinity"):
                for smiles, reward, affinity, iteration in snap.get(bucket_name, {}).get(self.max_k, []):
                    key = (smiles, iteration)
                    if key not in seen:
                        seen[key] = (smiles, reward, affinity, iteration)
        for smiles, reward, affinity, iteration in seen.values():
            self.add(smiles, reward, affinity, iteration)

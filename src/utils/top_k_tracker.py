"""Running top-K tracker for generated molecules across a training run.

Keeps **one entry per unique canonical SMILES** in ``self._stats`` (a dict). Each entry holds
cumulative running statistics — mean reward (GFN training signal), mean Uni-Dock docking score /
affinity (kcal/mol), and the number of times the molecule was sampled. A molecule re-sampled (and
re-docked, with a slightly different score each time) therefore collapses to a single row whose
reward/affinity are the running means, instead of appearing at many ranks.

Ranking by reward (largest mean reward) and by affinity (most-negative mean affinity) happens at
snapshot time over the unique molecules.

Stored entries are SMILES-only (no RDKit Mol objects) for portability. Persisted as two flat CSVs
(``top100_by_reward.csv`` / ``top100_by_affinity.csv``) so the hall of fame is human-readable and
easy to load elsewhere; the same CSVs double as the resume source and carry a ``count`` column so
the running means resume exactly. Pre-``count`` CSVs and the legacy ``top_k_mols.pt`` dump (the
previous ``torch.save`` format) are still read on resume — their rows are replayed through ``add``
so any old duplicate rows collapse into the new running means.
"""

import csv
import os
from typing import Iterable, Optional

import torch

REWARD_CSV = "top100_by_reward.csv"
AFFINITY_CSV = "top100_by_affinity.csv"
LEGACY_PT = "top_k_mols.pt"
_CSV_FIELDS = ("rank", "smiles", "reward", "affinity", "iteration", "count")


class TopKTracker:
    def __init__(self, ks: Iterable[int] = (10, 100), cap: int = 20000):
        self.ks = tuple(sorted(ks))
        self.max_k = self.ks[-1]
        # canonical SMILES -> running stats. ``reward_n`` counts every observation (== the sample
        # count); ``aff_n`` counts only observations that carried an affinity, so a non-docking task
        # leaves the affinity mean undefined without skewing the reward mean. ``first_iter`` is the
        # discovery iteration ("when found"), kept stable across re-samples.
        self._stats: dict = {}
        # Soft cap on unique molecules retained, to bound memory on long runs (the previous heaps
        # were bounded at ``max_k``). Far above ``max_k`` so anything near the hall of fame survives.
        self._cap = cap

    def add(self, smiles: str, reward: float, affinity: Optional[float], iteration: int):
        if not smiles or reward is None:
            return
        st = self._stats.get(smiles)
        if st is None:
            st = {"reward_sum": 0.0, "reward_n": 0, "aff_sum": 0.0, "aff_n": 0,
                  "first_iter": iteration}
            self._stats[smiles] = st
        st["reward_sum"] += float(reward)
        st["reward_n"] += 1
        if affinity is not None:
            st["aff_sum"] += float(affinity)
            st["aff_n"] += 1
        if len(self._stats) > self._cap:
            self._prune()

    def add_batch(self, smiles_list, rewards, affinities=None, iteration: int = 0):
        if affinities is None:
            affinities = [None] * len(smiles_list)
        for s, r, a in zip(smiles_list, rewards, affinities):
            self.add(s, r, a, iteration)

    def _entries(self):
        """One ``(smiles, reward_mean, aff_mean_or_None, first_iter, count)`` tuple per unique mol."""
        out = []
        for smi, st in self._stats.items():
            if st["reward_n"] == 0:
                continue
            reward_mean = st["reward_sum"] / st["reward_n"]
            aff_mean = st["aff_sum"] / st["aff_n"] if st["aff_n"] else None
            out.append((smi, reward_mean, aff_mean, st["first_iter"], st["reward_n"]))
        return out

    def _prune(self):
        """Bound memory: keep the union of the top ``cap//2`` molecules by mean reward and by mean
        affinity. Any molecule near the top-``max_k`` under either ranking always survives, so the
        hall-of-fame running means are never disturbed; only persistently-mediocre molecules are
        dropped. Amortized cheap — runs at most once per ``cap//2`` new unique molecules.
        """
        entries = self._entries()
        keep_each = max(self.max_k, self._cap // 2)
        by_reward = sorted(entries, key=lambda e: (-e[1], e[0]))[:keep_each]
        with_aff = [e for e in entries if e[2] is not None]
        by_aff = sorted(with_aff, key=lambda e: (e[2], e[0]))[:keep_each]
        keep = {e[0] for e in by_reward} | {e[0] for e in by_aff}
        self._stats = {smi: st for smi, st in self._stats.items() if smi in keep}

    def snapshot(self):
        # Deterministic ordering: reward descending then SMILES; affinity ascending (most negative
        # = best) then SMILES. Entries are already (smiles, reward, affinity, iteration, count).
        entries = self._entries()
        rew_sorted = sorted(entries, key=lambda e: (-e[1], e[0]))
        aff_sorted = sorted([e for e in entries if e[2] is not None], key=lambda e: (e[2], e[0]))
        return {
            "top_by_reward": {k: rew_sorted[:k] for k in self.ks},
            "top_by_affinity": {k: aff_sorted[:k] for k in self.ks},
        }

    def averages(self):
        """Mean reward of the top-10/top-100 (ranked by reward) and mean affinity of the
        top-10/top-100 (ranked by affinity). Returns NaN for an empty bucket, or -- for the
        affinity buckets -- when no entry has a docking score. Used to log monotonically-improving
        hall-of-fame curves during training.
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
        """Write the two top-``max_k`` buckets (one row per unique molecule) to flat CSVs."""
        os.makedirs(out_dir or ".", exist_ok=True)
        snap = self.snapshot()
        for fname, bucket in ((REWARD_CSV, "top_by_reward"), (AFFINITY_CSV, "top_by_affinity")):
            rows = snap[bucket][self.max_k]
            with open(os.path.join(out_dir, fname), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(_CSV_FIELDS)
                for rank, (smiles, reward, affinity, iteration, count) in enumerate(rows, start=1):
                    w.writerow([rank, smiles, reward, "" if affinity is None else affinity,
                                iteration, count])

    def load(self, out_dir: str):
        """Repopulate the running stats from the CSVs in ``out_dir`` (or a legacy ``.pt`` dump)."""
        reward_csv = os.path.join(out_dir, REWARD_CSV)
        affinity_csv = os.path.join(out_dir, AFFINITY_CSV)
        if os.path.exists(reward_csv) or os.path.exists(affinity_csv):
            # A molecule appears in both CSVs (different sort orders) with identical stats; restore
            # each only once. seen_new keys the new (count) format by SMILES; seen_obs keys the
            # replayed old format by (smiles, iteration) so each original observation replays once.
            seen_new: set = set()
            seen_obs: set = set()
            for path in (reward_csv, affinity_csv):
                if not os.path.exists(path):
                    continue
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    has_count = reader.fieldnames is not None and "count" in reader.fieldnames
                    for row in reader:
                        smiles = row["smiles"]
                        if not smiles:
                            continue
                        aff = row["affinity"]
                        affinity = None if aff == "" else float(aff)
                        iteration = int(row["iteration"])
                        if has_count:
                            if smiles in seen_new:
                                continue
                            seen_new.add(smiles)
                            count = int(row["count"])
                            st = {"reward_sum": float(row["reward"]) * count, "reward_n": count,
                                  "aff_sum": 0.0, "aff_n": 0, "first_iter": iteration}
                            if affinity is not None:
                                st["aff_sum"] = affinity * count
                                st["aff_n"] = count
                            self._stats[smiles] = st
                        else:
                            key = (smiles, iteration)
                            if key in seen_obs:
                                continue
                            seen_obs.add(key)
                            self.add(smiles, float(row["reward"]), affinity, iteration)
            return
        # Back-compat: resume from the previous torch.save dump if present.
        legacy = os.path.join(out_dir, LEGACY_PT)
        if not os.path.exists(legacy):
            return
        # weights_only=True is safe here: the dump is plain dict/list/tuple of str/float/int.
        snap = torch.load(legacy, map_location="cpu", weights_only=True)
        seen_obs = set()
        for bucket_name in ("top_by_reward", "top_by_affinity"):
            for smiles, reward, affinity, iteration in snap.get(bucket_name, {}).get(self.max_k, []):
                key = (smiles, iteration)
                if key in seen_obs:
                    continue
                seen_obs.add(key)
                self.add(smiles, reward, affinity, iteration)

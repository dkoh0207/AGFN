"""ChEMBL ring-system rarity scorer for the denovo docking reward gate.

Produces, per molecule, the frequency of its *rarest* ring system in ChEMBL (via
``useful_rdkit_utils.RingSystemLookup``). Fed into the shared ``apply_gate`` helper with
``mode="hard"`` / ``threshold=5`` this reproduces the notebook's ``min_freq >= 5`` plausibility
boolean: molecules whose rarest ring system appears fewer than 5 times in ChEMBL are floored to
``fail_reward``.

Acyclic molecules have no ring systems, so they carry no rare-ring penalty -> their value is a
large finite sentinel and they always pass the gate (matching the notebook's ``default=10**9``).
The sentinel must be *finite* — ``apply_gate``'s hard mode guards on ``np.isfinite``, so a
``+inf`` sentinel would be treated as a gate failure. ChEMBL ring counts top out in the low
millions, so ``1e9`` clears any real threshold while staying finite. Only genuinely unparseable
SMILES take the fail-closed ``NaN`` path.
"""

import numpy as np
from rdkit import Chem

# Large finite "no rare ring" value for acyclic molecules; must stay np.isfinite (see above).
ACYCLIC_SENTINEL = 1e9


class RingSystemScorer:
    """Per-molecule rarest-ChEMBL-ring-system frequency, as a reward gate value.

    ``score_smiles_with_failures(smiles_list)`` returns ``(values, score_mask)``:
      - ``value`` = min ChEMBL count over the molecule's ring systems
      - acyclic molecules -> ``ACYCLIC_SENTINEL`` (no rare ring -> always passes the gate)
      - unparseable SMILES -> ``NaN`` + ``score_mask`` ``False`` (fail closed)
    """

    def __init__(self, ring_file=None, ignore_stereo=False, lookup=None):
        # Eager construction (mirrors AutoGluonBBBScorer): a missing/undownloadable ring DB
        # aborts at driver startup instead of silently degrading every molecule to NaN during
        # training. The heavy import is deferred to here so importing this module stays cheap
        # and only an *enabled* constraint requires useful_rdkit_utils. Tests inject ``lookup``.
        if lookup is not None:
            self.lookup = lookup
        else:
            from useful_rdkit_utils import RingSystemLookup
            self.lookup = RingSystemLookup(ring_file=ring_file, ignore_stereo=ignore_stereo)

    def score_smiles_with_failures(self, smiles_list):
        n = len(smiles_list)
        values = np.full(n, np.nan, dtype=np.float64)
        score_mask = np.zeros(n, dtype=bool)
        for i, smi in enumerate(smiles_list):
            # Parse a fresh mol from SMILES: process_mol tags bonds in place, so we never want
            # to hand it (or mutate) a caller-owned RDKit object.
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            rings = self.lookup.process_mol(mol)  # [(ring_smiles, chembl_count), ...]
            values[i] = min((count for _, count in rings), default=ACYCLIC_SENTINEL)
            score_mask[i] = True
        return values, score_mask

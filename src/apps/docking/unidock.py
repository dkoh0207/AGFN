"""UniDock docking backend for AGFN.

The docking engine for de novo finetuning (``task: QedxSaxDock``). It runs entirely **inside the
calling Python session** via the ``unidock_tools`` API (no separate Python interpreter, no
on-disk mol hand-off) — only the compiled ``unidock`` binary is launched as a child process
internally by ``unidock_tools``, which is intrinsic to the engine.

Adapted from the reference implementation in RxnFlow
(``src/rxnflow/tasks/utils/unidock.py``), with one AGFN-specific change: AGFN already ships
``.pdbqt`` receptors and explicit box centers/sizes (``target_grid`` in the config), so we feed
those straight to ``UniDock`` and skip the pdb->pdbqt conversion and pybel-based pocket-center
detection that RxnFlow performs.

The ``unidock`` binary ships in the same conda env as the training stack (the canonical
``agfn`` env), at ``$CONDA_PREFIX/bin/unidock``. ``unidock_tools`` resolves it via
``shutil.which`` (i.e. ``PATH``), so :func:`_ensure_env_bin_on_path` puts this interpreter's own
``$CONDA_PREFIX/bin`` on ``PATH`` before docking. That guard is needed because a Jupyter kernel
launched by absolute-path python (the standard ``ipykernel`` install) leaves the env's ``bin/``
off ``PATH`` even though its site-packages are importable — so without it ``import
unidock_tools`` succeeds but ``shutil.which("unidock")`` returns ``None``.
"""

import os
import sys
import tempfile
import multiprocessing
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule, srETKDGv3


def _ensure_env_bin_on_path() -> None:
    """Put the running interpreter's own ``$CONDA_PREFIX/bin`` on ``PATH`` (idempotent).

    ``unidock_tools`` finds the compiled ``unidock`` binary via ``shutil.which`` (which reads
    ``PATH``, not ``sys.path``) and launches it as a child process. A Jupyter kernel started by
    absolute-path python puts the env's site-packages on ``sys.path`` but leaves the env's
    ``bin/`` off ``PATH`` unless the env was activated — so the binary lookup fails even though
    ``import unidock_tools`` works. Scoped to this process and adds only this env's own ``bin``,
    so nothing leaks past the conda env boundary.
    """
    env_bin = os.path.join(sys.prefix, "bin")
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    if env_bin not in parts:
        os.environ["PATH"] = os.pathsep.join([env_bin, *parts])


def run_etkdg_func(args: Tuple[str, Path]) -> Optional[Path]:
    """Embed a single SMILES into a 3D conformer and write it as an SDF.

    Returns the SDF path on success, or ``None`` if embedding fails (so one bad ligand never
    kills the batch). Ported verbatim from RxnFlow.
    """
    param = srETKDGv3()
    param.randomSeed = 1
    param.timeout = 1  # prevent a pathological molecule from stalling the batch

    smi, sdf_path = args
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        mol = Chem.AddHs(mol)
        EmbedMolecule(mol, param)
        assert mol.GetNumConformers() > 0
        mol = Chem.RemoveHs(mol)
        with Chem.SDWriter(str(sdf_path)) as w:
            w.write(mol)
    except Exception:
        return None
    return sdf_path


def dock_smiles(
    smiles_list: List[str],
    receptor: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    search_mode: str = "fast",
    seed: int = 1,
    num_workers: int = 1,
    return_poses: bool = False,
):
    """Dock a batch of SMILES against ``receptor`` and return per-SMILES docking scores.

    Failures (bad SMILES, failed embedding, missing output) yield ``0.0``. ETKDG runs as an
    in-process loop by default; ``num_workers > 1`` opts into a multiprocessing pool.

    When ``return_poses`` is True, also returns the docked 3D poses as MOL-block strings (so the
    hall of fame can be written to SDF without re-docking). Returns ``(scores, molblocks)`` with
    ``molblocks`` aligned to ``smiles_list`` (``None`` where docking failed); the pose's
    coordinates are absolute, in the receptor's frame. The molblock is captured here, before the
    ``TemporaryDirectory`` is torn down.
    """
    # Make sure this interpreter's own $CONDA_PREFIX/bin is on PATH so unidock_tools can find the
    # `unidock` binary even when running under a Jupyter kernel that didn't activate the env.
    _ensure_env_bin_on_path()

    # Imported here (lazily) so importing this module stays cheap and any unidock_tools issue
    # surfaces at dock time with a clear traceback rather than at import.
    from unidock_tools.application.unidock_pipeline import UniDock

    num_mols = len(smiles_list)
    receptor_path = Path(receptor)
    if receptor_path.suffix.lower() == ".pdb":
        # AGFN ships pdbqt, but support a raw pdb just in case.
        from unidock_tools.application.proteinprep import pdb2pdbqt

        pdbqt_path = receptor_path.with_suffix(".pdbqt")
        if not pdbqt_path.exists():
            pdb2pdbqt(receptor_path, pdbqt_path)
        receptor_path = pdbqt_path

    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        etkdg_dir = out_dir / "etkdg"
        etkdg_dir.mkdir(parents=True)

        args = [(smi, etkdg_dir / f"{i}.sdf") for i, smi in enumerate(smiles_list)]
        if num_workers and num_workers > 1:
            with multiprocessing.Pool(num_workers) as pool:
                sdf_list = pool.map(run_etkdg_func, args)
        else:
            sdf_list = [run_etkdg_func(a) for a in args]
        sdf_list = [f for f in sdf_list if f is not None]

        if len(sdf_list) > 0:
            runner = UniDock(
                receptor_path,
                sdf_list,
                center[0], center[1], center[2],
                size[0], size[1], size[2],
                out_dir / "workdir",
            )
            runner.docking(
                out_dir / "savedir",
                num_modes=1,
                search_mode=search_mode,
                seed=seed,
            )

        scores: List[float] = []
        molblocks: List[Optional[str]] = []
        for i in range(num_mols):
            molblock = None
            try:
                docked_file = out_dir / "savedir" / f"{i}.sdf"
                docked_rdmol = list(Chem.SDMolSupplier(str(docked_file)))[0]
                assert docked_rdmol is not None
                score = float(docked_rdmol.GetProp("docking_score"))
                if return_poses:
                    molblock = Chem.MolToMolBlock(docked_rdmol)
            except Exception:
                score = 0.0
            scores.append(score)
            molblocks.append(molblock)
    if return_poses:
        return scores, molblocks
    return scores


class UniDockGPU:
    """UniDock docking backend.

    Construct from a ``target_grid`` entry, e.g.::

        UniDockGPU(target="braf", **hps.target_grid["braf"], search_mode="fast")

    where the grid entry provides ``receptor`` and ``center_x/y/z`` + ``size_x/y/z``.
    """

    def __init__(
        self,
        target: Optional[str] = None,
        receptor: Optional[str] = None,
        center_x: float = None,
        center_y: float = None,
        center_z: float = None,
        size_x: float = None,
        size_y: float = None,
        size_z: float = None,
        search_mode: str = "fast",
        reward_scale_max: float = -1.0,
        reward_scale_min: float = -10.0,
        num_workers: int = 1,
        seed: int = 1,
    ):
        if receptor is None:
            raise ValueError("UniDockGPU requires a `receptor` pdbqt/pdb path")
        self.target = target
        self.receptor = receptor
        self.center = (center_x, center_y, center_z)
        self.size = (size_x, size_y, size_z)
        self.search_mode = search_mode
        self.reward_scale_max = reward_scale_max
        self.reward_scale_min = reward_scale_min
        self.num_workers = num_workers
        self.seed = seed
        # Cache of the best (most negative) docked pose seen per canonical SMILES, so the hall of
        # fame can be written to SDF without re-docking. Keyed by the SMILES passed to
        # calculate_rewards (already canonical), so keys match TopKTracker's keys exactly.
        self.pose_index: dict = {}                # smiles -> (affinity, molblock)
        self._pose_cap = 2000                     # bound memory; >> tracker max_k (100)

    def calculate_rewards(self, smiles: List[str]) -> Tuple[List[str], List[float], List[float]]:
        """Dock ``smiles`` and return ``(smiles, affinities, rewards)``.

        Affinities are clamped to <= 0 (positive/failed scores -> 0, as in RxnFlow), then scaled
        to a reward via ``(affinity + reward_scale_min) / (reward_scale_min + reward_scale_max)
        - 1`` (defaults map an affinity of -10 -> reward 0 and -1 -> reward -1).

        Side effect: caches each successful docked pose in ``self.pose_index`` for SDF export.
        """
        affinities, molblocks = dock_smiles(
            smiles,
            self.receptor,
            self.center,
            self.size,
            search_mode=self.search_mode,
            seed=self.seed,
            num_workers=self.num_workers,
            return_poses=True,
        )
        affinities = np.array([min(a, 0.0) for a in affinities], dtype=np.float64)
        rewards = (affinities + self.reward_scale_min) / (self.reward_scale_min + self.reward_scale_max) - 1

        self._cache_poses(smiles, affinities, molblocks)

        print(
            f"UNIDOCK AFFINITIES: mean={round(float(np.mean(affinities)), 3)}, "
            f"std={round(float(np.std(affinities)), 3)}, "
            f"min={round(float(np.min(affinities)), 3)}, "
            f"max={round(float(np.max(affinities)), 3)}"
        )

        return list(smiles), list(affinities), list(rewards)

    def _cache_poses(self, smiles, affinities, molblocks):
        """Keep the most-negative-affinity pose per SMILES, bounded to ``_pose_cap`` entries."""
        for smi, aff, mb in zip(smiles, affinities, molblocks):
            if mb is None:
                continue
            aff = float(aff)
            prev = self.pose_index.get(smi)
            if prev is None or aff < prev[0]:
                self.pose_index[smi] = (aff, mb)
        if len(self.pose_index) > self._pose_cap:
            # drop the worst (least negative) affinities
            keep = sorted(self.pose_index.items(), key=lambda kv: kv[1][0])[: self._pose_cap]
            self.pose_index = dict(keep)

    def write_hall_of_fame_sdf(self, entries, path: str) -> int:
        """Write top molecules + their cached docked poses to an SDF.

        ``entries`` is a TopKTracker snapshot bucket: a list of ``(smiles, reward, affinity,
        iteration, count)`` already in rank order, one row per unique molecule. ``reward``/
        ``affinity`` are the running means and ``count`` is how many times the molecule was sampled.
        For each entry with a cached pose, write the 3D pose with ``smiles``/``docking_score``/
        ``reward``/``iteration``/``n_obs``/``rank`` as SD tags. Entries whose pose isn't cached are
        skipped. Returns the number of molecules written.
        """
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with Chem.SDWriter(str(path)) as w:
            for rank, (smiles, reward, affinity, iteration, count) in enumerate(entries, start=1):
                cached = self.pose_index.get(smiles)
                if cached is None:
                    continue
                mol = Chem.MolFromMolBlock(cached[1])
                if mol is None:
                    continue
                mol.SetProp("_Name", f"rank{rank}")
                mol.SetProp("smiles", smiles)
                mol.SetProp("rank", str(rank))
                mol.SetProp("docking_score", f"{affinity}")
                mol.SetProp("reward", f"{reward}")
                mol.SetProp("iteration", str(iteration))
                mol.SetProp("n_obs", str(count))
                w.write(mol)
                written += 1
        return written

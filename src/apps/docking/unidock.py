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

The ``unidock`` binary is installed in the same conda env as the training stack (the
``agfn-no-vina`` env), so it resolves from ``$CONDA_PREFIX/bin`` with no PATH manipulation.
"""

import tempfile
import multiprocessing
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule, srETKDGv3


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
) -> List[float]:
    """Dock a batch of SMILES against ``receptor`` and return per-SMILES docking scores.

    Failures (bad SMILES, failed embedding, missing output) yield ``0.0``. ETKDG runs as an
    in-process loop by default; ``num_workers > 1`` opts into a multiprocessing pool.
    """
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
        for i in range(num_mols):
            try:
                docked_file = out_dir / "savedir" / f"{i}.sdf"
                docked_rdmol = list(Chem.SDMolSupplier(str(docked_file)))[0]
                assert docked_rdmol is not None
                score = float(docked_rdmol.GetProp("docking_score"))
            except Exception:
                score = 0.0
            scores.append(score)
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

    def calculate_rewards(self, smiles: List[str]) -> Tuple[List[str], List[float], List[float]]:
        """Dock ``smiles`` and return ``(smiles, affinities, rewards)``.

        Affinities are clamped to <= 0 (positive/failed scores -> 0, as in RxnFlow), then scaled
        to a reward via ``(affinity + reward_scale_min) / (reward_scale_min + reward_scale_max)
        - 1`` (defaults map an affinity of -10 -> reward 0 and -1 -> reward -1).
        """
        affinities = dock_smiles(
            smiles,
            self.receptor,
            self.center,
            self.size,
            search_mode=self.search_mode,
            seed=self.seed,
            num_workers=self.num_workers,
        )
        affinities = np.array([min(a, 0.0) for a in affinities], dtype=np.float64)
        rewards = (affinities + self.reward_scale_min) / (self.reward_scale_min + self.reward_scale_max) - 1

        print(
            f"UNIDOCK AFFINITIES: mean={round(float(np.mean(affinities)), 3)}, "
            f"std={round(float(np.std(affinities)), 3)}, "
            f"min={round(float(np.min(affinities)), 3)}, "
            f"max={round(float(np.max(affinities)), 3)}"
        )

        return list(smiles), list(affinities), list(rewards)

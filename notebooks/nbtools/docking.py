"""In-process Uni-Dock redocking — same ETKDG + UniDock backend training uses (needs a GPU).

Repo imports (``unidock``, ``unidock_tools``) are lazy, so this module imports cleanly even on a
box without the docking backend; only :func:`dock_batch` requires it.
"""

from pathlib import Path

# Docking-score -> reward normalization, identical to the reward backend
# (src/apps/docking/denovo/reward_dock.py). Affinity is clamped to <= 0 first.
REWARD_SCALE_MAX = -1.0
REWARD_SCALE_MIN = -10.0


def normalize_reward(aff):
    """Map a docking score (kcal/mol, more negative = better) to the training reward scale."""
    return (aff + REWARD_SCALE_MIN) / (REWARD_SCALE_MIN + REWARD_SCALE_MAX) - 1


def dock_batch(smiles, receptor, center, size, workdir, search_mode="fast", seed=0):
    """Embed (ETKDG) and dock a list of SMILES with Uni-Dock; poses persist under ``workdir``.

    ``center`` / ``size`` are 3-tuples (Å). Returns a dict with ``smiles``, ``affinities`` (clamped
    docking scores), ``rewards`` (normalized), ``pose_paths`` (``None`` where docking failed), and
    ``save_dir`` (where the per-ligand ``<i>.sdf`` poses live, for the 3D viewer).
    """
    from rdkit import Chem
    from unidock import run_etkdg_func                       # same ETKDG embedder the reward backend uses
    from unidock_tools.application.unidock_pipeline import UniDock

    workdir = Path(workdir)
    etkdg_dir = workdir / "etkdg"
    save_dir = workdir / "poses"
    etkdg_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    sdf_inputs = [f for f in (run_etkdg_func((s, etkdg_dir / f"{i}.sdf"))
                              for i, s in enumerate(smiles)) if f is not None]
    if sdf_inputs:
        UniDock(Path(receptor), sdf_inputs, *center, *size, workdir / "workdir").docking(
            save_dir, num_modes=1, search_mode=search_mode, seed=seed,
        )

    affinities, rewards, pose_paths = [], [], []
    for i, _s in enumerate(smiles):
        pose = save_dir / f"{i}.sdf"
        try:
            m = list(Chem.SDMolSupplier(str(pose)))[0]
            aff = min(float(m.GetProp("docking_score")), 0.0)
            ok = m is not None
        except Exception:
            aff, ok = 0.0, False
        affinities.append(aff)
        rewards.append(float(normalize_reward(aff)))
        pose_paths.append(str(pose) if ok else None)

    return {"smiles": list(smiles), "affinities": affinities, "rewards": rewards,
            "pose_paths": pose_paths, "save_dir": save_dir}

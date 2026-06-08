"""Repo bootstrap + denovo config loading — the boilerplate shared by the notebooks."""

import os
import sys
from pathlib import Path

# Conditional ranges + per-property variances for the QedxSaxDock denovo task. These are the exact
# literals denovo_driver.py uses; centralized here so visualize / visualize_pocket /
# visualize_hall_of_fame don't each carry their own copy.
QEDXSAXDOCK_CONDITIONAL_RANGE = {
    "tpsa":      [[10, 200], [10, 200], 0],
    "num_rings": [[1, 5],   [1, 5],   1],
    "sas":       [[1, 5],   [1, 5],   0],
    "qed":       [[0.5, 1], [0, 1],   0],
}
COND_PROP_VAR = {"tpsa": 20, "num_rings": 1, "sas": 1, "qed": 1}

# Paths (relative to the repo root) that the notebooks' imports expect on sys.path. denovo_trainer
# and unidock are flat modules under src/apps/docking[/denovo], not package-qualified.
_SRC_PATHS = ("src", "src/apps/docking", "src/apps/docking/denovo")


def find_repo_root(marker="src/config", start=None):
    """Walk up from ``start`` (default: cwd) and return the first ancestor containing ``marker``."""
    start = Path(start or Path.cwd())
    for cand in [start, *start.parents]:
        if (cand / marker).exists():
            return cand
    raise FileNotFoundError(f"could not find repo root (no {marker!r} above {start})")


def setup_repo(marker="src/config", chdir=True, start=None):
    """Locate the repo root, ``chdir`` into it (so the configs' relative paths resolve), and put the
    repo's ``src`` trees on ``sys.path`` — exactly the wiring denovo_driver.py relies on.

    Returns the repo-root ``Path``. Idempotent: re-running won't duplicate sys.path entries.
    """
    root = find_repo_root(marker, start=start)
    if chdir:
        os.chdir(root)
    for rel in _SRC_PATHS:
        full = str((root / rel).resolve())
        if full not in sys.path:
            sys.path.insert(0, full)
    return root


def load_denovo_hps(config_path, saved_model_path=None, target_name=None):
    """Load + patch the hyperparameters for a QedxSaxDock denovo run.

    Mirrors the load cell shared by the visualization notebooks: read ``finetuning`` from the YAML
    into an EasyDict, apply the training-time patches, and pair it with the task's conditional
    ranges. ``saved_model_path`` / ``target_name`` override the YAML values when given.

    Returns ``(hps, conditional_range_dict, cond_prop_var)``.
    """
    import yaml
    from easydict import EasyDict

    with open(config_path) as f:
        hps = EasyDict(yaml.safe_load(f)).finetuning
    hps.update({"Z_learning_rate": 1e-3})

    if hps.task != "QedxSaxDock":
        raise RuntimeError(f"unsupported task: {hps.task}")
    conditional_range_dict = {k: list(v) for k, v in QEDXSAXDOCK_CONDITIONAL_RANGE.items()}
    cond_prop_var = dict(COND_PROP_VAR)

    hps.task_conditionals = False
    hps.task_rewards_only = False
    hps.update(conditional_range_dict)

    if saved_model_path is not None:
        hps.saved_model_path = saved_model_path
    if target_name is not None:
        hps.target_name = target_name
    return hps, conditional_range_dict, cond_prop_var

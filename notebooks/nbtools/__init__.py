"""nbtools — helper package for the AGFN analysis notebooks.

The four notebooks under ``notebooks/`` (``visualize.ipynb``, ``visualize_pocket.ipynb``,
``visualize_hall_of_fame.ipynb``, ``check_frozen_core.ipynb``) keep only a *parameters* cell, a
one-line bootstrap, and thin call-sites; the real workflows live here as importable functions.

Bootstrap (the only boilerplate a notebook needs)::

    import sys
    from pathlib import Path
    for _c in [Path.cwd(), *Path.cwd().parents]:        # make `import nbtools` work from anywhere
        if (_c / "nbtools").is_dir():       sys.path.insert(0, str(_c)); break
        if (_c / "notebooks" / "nbtools").is_dir(): sys.path.insert(0, str(_c / "notebooks")); break
    import nbtools
    REPO_ROOT = nbtools.setup_repo()                     # chdir to repo root + put src/ on path

Importing this package pulls in ``py3Dmol`` **first** (best effort). In the ``agfn`` conda env,
importing ``rdkit.Chem.Draw`` before ``py3Dmol`` segfaults the kernel (native-lib init order);
because every notebook does ``import nbtools`` before touching RDKit's 2D drawing, that ordering
bug is structurally impossible here. Modules that need the repo's ``src/`` packages import them
*lazily inside functions*, since ``src/`` only lands on ``sys.path`` after :func:`setup_repo` runs.
"""

# Enforce the py3Dmol-before-rdkit.Chem.Draw load order for the whole package. Best effort: when
# py3Dmol isn't installed there is no conflict to avoid (nothing loads second against it), so a
# missing import here must not break Draw-only notebooks such as check_frozen_core.
try:  # noqa: SIM105
    import py3Dmol  # noqa: F401
except Exception:  # pragma: no cover - py3Dmol optional for 2D-only notebooks
    pass

from .repo import setup_repo, load_denovo_hps  # noqa: E402

__all__ = ["setup_repo", "load_denovo_hps"]

"""Hall-of-fame artifacts: locate a run's files, plot training/top-K curves, and load the top-N.

Reads what training writes: ``top100_by_{reward,affinity}.{sdf,csv}`` (top molecules + 3D docked
poses + SD tags) plus per-run ``metrics.csv`` / ``config.json``. The ``top100_*`` files live in the
per-run subdir for new runs but in the parent log dir for older runs — :func:`resolve_run_paths`
handles both.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, AllChem


@dataclass
class RunPaths:
    log_dir: Path
    run_subdir: Path
    metrics_csv: Path
    config_json: Path
    sdf_path: Path
    csv_path: Path
    reward_csv: Path
    affinity_csv: Path
    receptor_path: Path | None
    box_center: tuple | None
    box_size: tuple | None
    target_name: str | None


def resolve_run_paths(log_dir, repo_root, rank_by="reward", verbose=True):
    """Resolve every file the hall-of-fame visualizer needs from a run's parent ``log_dir``.

    Auto-finds the latest run subfolder that has ``metrics.csv``; resolves the ``top100_*`` files
    from the run subdir first then the parent; and reads the receptor + docking box from the run's
    ``config.json`` (repo-root-relative paths made absolute). Returns a :class:`RunPaths`.
    """
    log_dir = Path(log_dir)
    repo_root = Path(repo_root)

    run_dirs = sorted([p for p in log_dir.iterdir() if p.is_dir() and (p / "metrics.csv").exists()])
    run_subdir = run_dirs[-1] if run_dirs else log_dir
    metrics_csv = run_subdir / "metrics.csv"
    config_json = run_subdir / "config.json"

    def _resolve_hof(name):
        cand = run_subdir / name
        return cand if cand.exists() else log_dir / name

    sdf_path = _resolve_hof(f"top100_by_{rank_by}.sdf")
    csv_path = _resolve_hof(f"top100_by_{rank_by}.csv")
    reward_csv = _resolve_hof("top100_by_reward.csv")
    affinity_csv = _resolve_hof("top100_by_affinity.csv")

    receptor_path = box_center = box_size = target_name = None
    if config_json.exists():
        cfg = json.loads(config_json.read_text())
        target_name = cfg.get("target_name")
        grid = cfg.get("target_grid", {}).get(target_name, {})
        if grid:
            rp = Path(grid["receptor"])
            receptor_path = rp if rp.is_absolute() else (repo_root / rp)
            box_center = (grid["center_x"], grid["center_y"], grid["center_z"])
            box_size = (grid["size_x"], grid["size_y"], grid["size_z"])

    paths = RunPaths(log_dir, run_subdir, metrics_csv, config_json, sdf_path, csv_path,
                     reward_csv, affinity_csv, receptor_path, box_center, box_size, target_name)
    if verbose:
        print(f"LOG_DIR      : {log_dir}")
        print(f"run subfolder: {run_subdir.name}")
        print(f"metrics.csv  : {'found' if metrics_csv.exists() else 'MISSING'}")
        print(f"hall of fame : {sdf_path.name} "
              f"{'found' if sdf_path.exists() else 'MISSING (run training first)'} "
              f"(dir: {sdf_path.parent.name})")
        print(f"target       : {target_name}   receptor: {receptor_path}")
        print(f"docking box  : center={box_center} size={box_size}")
    return paths


def plot_metrics(metrics_csv, smooth=1):
    """Plot a curated subset of ``metrics.csv`` columns vs ``Train_iter``, auto-skipping any column
    that isn't present (headers are matched by stripped name). Returns the DataFrame."""
    df = pd.read_csv(metrics_csv)
    df.columns = [c.strip() for c in df.columns]
    xcol = "Train_iter" if "Train_iter" in df.columns else df.columns[-1]

    wanted = [
        ("Average QedxSaxDock score online", "Docking affinity (kcal/mol)"),
        ("Average Overall Reward",           "Mean reward"),
        ("Total loss",                       "Total loss"),
        ("logZ",                             "logZ"),
        ("Percent_valid_mols",               "Valid (%)"),
        ("Percent_unique_in_batch",          "Unique in batch (%)"),
        ("Percent_novel_cumulative",         "Novel cumulative (%)"),
        ("Avg. QED onln",                    "QED"),
        ("Avg. SAS onln",                    "SAS"),
    ]
    present = [(c, t) for c, t in wanted if c in df.columns]
    ncol = 3
    nrow = (len(present) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (col, title) in zip(axes.flat, present):
        ax.axis("on")
        y = df[col].rolling(smooth, min_periods=1).mean() if smooth > 1 else df[col]
        ax.plot(df[xcol], y, lw=1.3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xcol, fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    return df


def topk_curves_from_metrics(df):
    """True top-K curves logged during training (if present). Returns ``(x, {label: series})`` or None."""
    cols = {
        "top-10 reward":   "Top-10 mean reward",
        "top-100 reward":  "Top-100 mean reward",
        "top-10 docking":  "Top-10 mean docking score",
        "top-100 docking": "Top-100 mean docking score",
    }
    if not any(c in df.columns for c in cols.values()):
        return None
    x = df["Train_iter"] if "Train_iter" in df.columns else df.index
    return x, {k: df[v] for k, v in cols.items() if v in df.columns}


def topk_curves_reconstructed(reward_csv, affinity_csv, ks=(10, 100)):
    """Approximate running top-K averages from the FINAL hall-of-fame CSVs (each row carries its
    discovery ``iteration``). Survivors only (hindsight); top-100 is a running mean until 100 exist."""
    def _curve(path, col, best):
        if not Path(path).exists():
            return None
        d = pd.read_csv(path).dropna(subset=[col, "iteration"]).sort_values("iteration")
        if d.empty:
            return None
        rows = []
        for t in sorted(d["iteration"].unique()):
            seen = d.loc[d["iteration"] <= t, col]
            for k in ks:
                topk = seen.nlargest(k) if best == "max" else seen.nsmallest(k)
                rows.append((t, k, topk.mean()))
        return pd.DataFrame(rows, columns=["iteration", "k", "mean"])
    return {
        "reward":  _curve(reward_csv,   "reward",   "max"),    # higher reward = better
        "docking": _curve(affinity_csv, "affinity", "min"),    # lower (more negative) = better
    }


def plot_topk_curves(paths):
    """Plot running top-10 / top-100 mean reward & docking score: the true logged curves when the
    run wrote them, otherwise reconstructed from the final ``top100_*`` CSVs in ``paths``."""
    fig, (axr, axd) = plt.subplots(1, 2, figsize=(12, 4))
    logged = None
    if paths.metrics_csv.exists():
        df = pd.read_csv(paths.metrics_csv)
        df.columns = [c.strip() for c in df.columns]
        logged = topk_curves_from_metrics(df)

    if logged is not None:
        x, series = logged
        for label, y in series.items():
            (axr if "reward" in label else axd).plot(x, y, lw=1.5, label=label)
        axr.set_title("Top-K mean reward (logged)")
        axd.set_title("Top-K mean docking score (logged)")
        for ax in (axr, axd):
            ax.set_xlabel("Train_iter")
    else:
        rec = topk_curves_reconstructed(paths.reward_csv, paths.affinity_csv)
        for col, ax, title in (("reward",  axr, "Top-K mean reward (reconstructed)"),
                               ("docking", axd, "Top-K mean docking score (reconstructed)")):
            cur = rec.get(col)
            if cur is None:
                ax.text(0.5, 0.5, f"no {col} data", ha="center", va="center")
                continue
            for k, g in cur.groupby("k"):
                ax.plot(g["iteration"], g["mean"], lw=1.5, marker=".", ms=3, label=f"top-{k}")
            ax.set_title(title)
            ax.set_xlabel("discovery iteration")

    axr.set_ylabel("reward  (higher = better)")
    axr.grid(alpha=0.3)
    axr.legend(fontsize=8)
    axd.set_ylabel("docking score kcal/mol  (lower = better)")
    axd.grid(alpha=0.3)
    axd.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_bbb_solubility(metrics_csv, smooth=20):
    """Plot running **BBB score** and **solubility LogS** vs ``Train_iter`` from ``metrics.csv``.

    Two side-by-side panels (BBB and LogS sit on very different scales, so they don't share a
    y-axis): each overlays the raw per-step trace (faint) with a rolling mean over ``smooth``
    steps. A panel whose column is absent is annotated and left blank, so this is safe across
    configs. Returns the DataFrame, or None if the file is missing.
    """
    if not Path(metrics_csv).exists():
        print(f"metrics.csv not found at {metrics_csv} — run training first.")
        return None
    df = pd.read_csv(metrics_csv)
    df.columns = [c.strip() for c in df.columns]
    x = df["Train_iter"] if "Train_iter" in df.columns else df.index

    panels = [
        ("Average BBB score",       "Average BBB score",       "BBB score  (higher = more permeant)",  "#0D4A70"),
        ("Average solubility LogS", "Average solubility LogS", "Solubility LogS  (higher = more soluble)", "#FF1F5B"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (col, title, ylabel, color) in zip(axes, panels):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"no '{col}' column", ha="center", va="center")
            ax.set_title(title)
            ax.axis("off")
            continue
        ax.plot(x, df[col], c=color, alpha=0.15)
        ax.plot(x, df[col].rolling(smooth, min_periods=1).mean(), c=color, lw=1.5)
        ax.set_title(title)
        ax.set_xlabel("Train_iter")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    return df


def load_hall_of_fame(sdf_path, n=None):
    """Return a list of ``(mol_with_3D_pose, props_dict)``, in file (rank) order."""
    if not Path(sdf_path).exists():
        return []
    out = []
    for m in Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True):
        if m is None:
            continue
        props = {k: m.GetProp(k) for k in m.GetPropNames()}
        out.append((m, props))
        if n is not None and len(out) >= n:
            break
    return out


def hall_of_fame_2d_grid(hof, mols_per_row=5, sub_img_size=(240, 200)):
    """Clean 2D depictions of the hall of fame, rebuilt from each record's ``smiles`` tag and
    captioned with rank / reward / docking score. Returns the grid image (or None if empty)."""
    mols2d, legends = [], []
    for m, p in hof:
        m2d = Chem.MolFromSmiles(p.get("smiles", Chem.MolToSmiles(m)))
        if m2d is None:
            continue
        AllChem.Compute2DCoords(m2d)
        mols2d.append(m2d)
        legends.append(f"#{p.get('rank', '?')}  r={float(p.get('reward', 'nan')):.2f}  "
                       f"dock={float(p.get('docking_score', 'nan')):.2f}")
    if not mols2d:
        return None
    return Draw.MolsToGridImage(mols2d, legends=legends, molsPerRow=mols_per_row,
                                subImgSize=sub_img_size, useSVG=False)


def hall_of_fame_table(paths, hof, n=None):
    """Flat table of the hall of fame: the ``top100_*.csv`` when present, else derived from the SDF
    property tags. Returns a DataFrame."""
    if paths.csv_path.exists():
        df = pd.read_csv(paths.csv_path)
        return df.head(n) if n is not None else df
    return pd.DataFrame([
        {"rank": p.get("rank"), "reward": p.get("reward"),
         "docking_score": p.get("docking_score"), "iteration": p.get("iteration"),
         "smiles": p.get("smiles")} for _, p in hof])


def topk_pt_grid(log_dir, rank_by="reward", n_show=20, mols_per_row=4, sub_img_size=(260, 200)):
    """Render the top-K molecules dumped to ``<log_dir>/top_k_mols.pt`` by the TopKTracker hook
    (a legacy artifact distinct from the ``top100_*`` CSV/SDF files). Each entry is
    ``(smiles, reward, affinity, iteration)``. Returns the grid image, or None (printing why) when
    the dump or the requested bucket is missing."""
    import torch

    top_path = Path(log_dir) / "top_k_mols.pt"
    if not top_path.exists():
        print(f"WARNING: no top-K dump at {top_path}.")
        print("Re-run training (denovo_driver.py) so the TopKTracker hook can write it.")
        return None
    top = torch.load(top_path, map_location="cpu")
    bucket_key = f"top_by_{rank_by}"
    available_ks = sorted(top.get(bucket_key, {}).keys())
    if not available_ks:
        print(f"top_k_mols.pt has no bucket {bucket_key!r}")
        return None
    bucket = top[bucket_key][max(available_ks)]
    mols, legends = [], []
    for smiles, reward, affinity, iteration in bucket[:n_show]:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            continue
        aff_str = "  n/a" if affinity is None else f"{affinity:+.2f}"
        mols.append(m)
        legends.append(f"r={reward:+.2f}  aff={aff_str}  it={iteration}")
    print(f"showing top {len(mols)} of {len(bucket)} ranked by {rank_by}")
    if not mols:
        return None
    return Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=sub_img_size,
                                legends=legends)


def plot_training_summary(df, out_svg=None, font_dirs=None, window=50, title=None):
    """Publication-style 3-panel training summary (reward / docking+composite / BBB+solubility).

    Each panel skips gracefully if its columns are absent. ``font_dirs`` (optional) registers .ttf/
    .otf fonts before plotting; ``out_svg`` (optional) saves the figure. Returns the Figure.
    """
    from matplotlib import font_manager, rcParams

    if font_dirs:
        for fd in font_dirs:
            for ff in Path(fd).rglob("*.[ot]tf"):
                font_manager.fontManager.addfont(str(ff))

    colors = {"public": "#0D4A70", "extra": "#FFC61E", "in_house": "#FF1F5B"}
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    x = df["Train_iter"] if "Train_iter" in df.columns else df.index

    def smooth_trace(ax, col, color, label, raw_alpha=0.1):
        if col not in df.columns:
            return
        ax.plot(x, df[col], c=color, alpha=raw_alpha)
        ax.plot(x, df[col].rolling(window).mean(), c=color, label=label)

    def line(ax, col, color, label):
        if col in df.columns:
            ax.plot(x, df[col], c=color, label=label)

    def merge_legends(primary, twin=None, **kw):
        handles, labels = [], []
        for ax in [primary] + ([twin] if twin else []):
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
        (twin or primary).legend(handles, labels, **kw)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    smooth_trace(axes[0], "Average Overall Reward", colors["public"], "Average Total Reward")
    line(axes[0], "Top-10 mean reward", colors["in_house"], "Top 10 Reward")
    line(axes[0], "Top-100 mean reward", colors["extra"], "Top 100 Reward")
    axes[0].set_ylabel("Reward", fontsize=14)
    axes[0].legend()

    ax1r = axes[1].twinx()
    line(axes[1], "Top-10 mean docking score", colors["in_house"], "Top 10 Docking Score")
    line(axes[1], "Top-100 mean docking score", colors["extra"], "Top 100 Docking Score")
    smooth_trace(ax1r, "Average QedxSaxDock score online", colors["public"], "Average QEDSaS x Docking")
    axes[1].set_ylabel("Docking Score", fontsize=14)
    ax1r.set_ylabel("Avg. QED x SAS x Docking", fontsize=14)
    merge_legends(axes[1], ax1r)

    ax2r = axes[2].twinx()
    smooth_trace(axes[2], "Average BBB score", colors["public"], "Average BBB Score", raw_alpha=0.2)
    smooth_trace(ax2r, "Average solubility LogS", colors["in_house"], "Average Solubility LogS", raw_alpha=0.2)
    axes[2].set_ylabel("Average BBB Score", fontsize=14)
    ax2r.set_ylabel("Average Solubility LogS", fontsize=14)
    merge_legends(axes[2], ax2r)

    for ax in axes:
        ax.grid(linestyle="--", alpha=0.3)
        ax.set_xlabel("Iteration", fontsize=14)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if out_svg:
        fig.savefig(out_svg)
    plt.show()
    return fig

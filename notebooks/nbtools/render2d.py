"""2D molecule rendering helpers (RDKit).

Safe to import ``rdkit.Chem.Draw`` at module top: importing the ``nbtools`` package loads py3Dmol
first, so the py3Dmol-before-Draw ordering is already satisfied by the time this module loads.
"""

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG


def draw_mol_with_indices(mol, highlight=None, legend="", size=(620, 460), recompute_coords=True):
    """SVG of ``mol`` with each atom labelled by its 0-based RDKit index (``atomNote``).

    ``highlight``: optional list of atom indices, or a ``{atom_idx: (r, g, b)}`` color map. The
    input ``mol`` is copied, never mutated. Returns an ``IPython.display.SVG``.
    """
    m = Chem.Mol(mol)                                   # copy so we don't mutate the caller's mol
    if recompute_coords:
        rdDepictor.Compute2DCoords(m)
    for a in m.GetAtoms():
        a.SetProp("atomNote", str(a.GetIdx()))
    hl = list(highlight) if highlight else []
    hlc = highlight if isinstance(highlight, dict) else None
    d = rdMolDraw2D.MolDraw2DSVG(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, highlightAtoms=hl, highlightAtomColors=hlc,
                                       legend=legend)
    d.FinishDrawing()
    return SVG(d.GetDrawingText())


def mol_grid(mols, legends=None, mols_per_row=4, sub_img_size=(260, 200),
             highlight_atom_lists=None, use_svg=False, save_path=None):
    """``Draw.MolsToGridImage`` wrapper that also (optionally) saves to ``save_path``.

    ``MolsToGridImage`` returns a PIL image or an IPython ``Image`` depending on the environment;
    this normalizes the save path across both. Returns the grid image object for inline display.
    """
    img = Draw.MolsToGridImage(
        list(mols), molsPerRow=mols_per_row, subImgSize=sub_img_size,
        legends=list(legends) if legends is not None else None,
        highlightAtomLists=highlight_atom_lists, useSVG=use_svg,
    )
    if save_path is not None:
        if hasattr(img, "save"):                        # PIL.Image
            img.save(str(save_path))
        elif hasattr(img, "data"):                      # IPython.display.Image
            Path(save_path).write_bytes(img.data)
    return img


def trajectory_step_grid(traj, ctx, mols_per_row=6, sub_img_size=(220, 180)):
    """Grid of a *single* trajectory's intermediate molecules (one tile per renderable step).

    ``traj`` is the ``(graph, action)`` step list (e.g. ``t["traj"]``); ``traj[t][0]`` is already
    the pre-action graph, so no env replay is needed. Returns ``(grid_image, n_steps)`` —
    ``grid_image`` is ``None`` when the trajectory has no renderable steps.
    """
    steps = []
    for g, _action in traj:
        if len(g.nodes) == 0:
            continue
        try:
            m = ctx.graph_to_mol(g)
        except Exception:
            m = None
        if m is not None:
            steps.append(m)
    if not steps:
        return None, 0
    legends = [f"t={i}" for i in range(len(steps))]
    return Draw.MolsToGridImage(steps, molsPerRow=mols_per_row, subImgSize=sub_img_size,
                                legends=legends), len(steps)


def resolve_growth(n, allowed, frozen):
    """Growth-site set implied by ``allowed_growth_atoms`` / ``frozen_atoms`` (exactly as
    ``build_frozen_seed_graph`` does). Set ONE of them; neither -> every seed atom is a growth site.
    """
    if allowed is not None and frozen is not None:
        raise ValueError("Set only ONE of ALLOWED_GROWTH_ATOMS / FROZEN_ATOMS (leave the other None).")
    if allowed is not None:
        return set(allowed)
    if frozen is not None:
        return set(range(n)) - set(frozen)
    return set(range(n))


def growth_partition_colors(n, growth):
    """Atom-color map for a frozen/growth preview: green = growth site, red = sealed-only."""
    sealed_only = set(range(n)) - set(growth)
    return {**{i: (1.00, 0.60, 0.60) for i in sealed_only},     # red  = sealed (no attachment)
            **{i: (0.60, 0.95, 0.60) for i in growth}}          # green = growth site


def smiles_to_2d(smiles_or_mol, fallback_mol=None):
    """Clean 2D depiction mol from a SMILES (or fall back to ``fallback_mol``). Returns None on
    parse failure. Used to rebuild flat 2D structures from hall-of-fame ``smiles`` tags."""
    m = Chem.MolFromSmiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    if m is None and fallback_mol is not None:
        m = Chem.MolFromSmiles(Chem.MolToSmiles(fallback_mol))
    if m is None:
        return None
    AllChem.Compute2DCoords(m)
    return m

"""3D structure / docked-pose rendering with py3Dmol, plus the pdbqt->pdb shim it needs."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import py3Dmol
from rdkit import Chem


def view_pose(receptor, ligand_sdf, w=500, h=400):
    """Receptor rainbow cartoon + ligand green-carbon sticks. ``receptor`` is a .pdbqt path."""
    with open(receptor) as f:
        rec = f.read()
    with open(ligand_sdf) as f:
        lig = f.read()
    v = py3Dmol.view(width=w, height=h)
    v.addModel(rec, "pdbqt")
    v.setStyle({"model": 0}, {"cartoon": {"color": "spectrum"}})
    v.addModel(lig, "sdf")
    v.setStyle({"model": 1}, {"stick": {"colorscheme": "greenCarbon"}})
    v.zoomTo({"model": 1})
    return v


def view_structure(path, fmt, boxes=(), ligand_resn=None,
                   bg="0x0e1116", zoom_out=1.0, w=620, h=460):
    """Cartoon view of a receptor with one or more docking boxes overlaid.

    ``boxes``: iterable of ``(center, size, color)``. Optional ligand drawn as green sticks.
    ``bg``: py3Dmol background (dark by default); ``zoom_out`` < 1 pulls the camera back so boxes
    that sit off the protein stay in frame.
    """
    txt = open(path).read()
    v = py3Dmol.view(width=w, height=h)
    v.setBackgroundColor(bg)
    v.addModel(txt, fmt)
    v.setStyle({"cartoon": {"color": "spectrum"}})
    if ligand_resn:
        v.addStyle({"resn": ligand_resn},
                   {"stick": {"colorscheme": "greenCarbon", "radius": 0.25}})
    for c, s, color in boxes:
        v.addBox({"center": {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])},
                  "dimensions": {"w": float(s[0]), "h": float(s[1]), "d": float(s[2])},
                  "color": color, "wireframe": True})
    v.zoomTo()
    if zoom_out != 1.0:
        v.zoom(zoom_out)
    return v


def pdbqt_to_pdb_text(path):
    """Minimal pdbqt -> pdb text: keep PDB columns [:66] of ATOM/HETATM, append an element guess.

    AGFN receptors are .pdbqt with a blank chain column; this yields a clean cartoon for 3Dmol.
    """
    lines = []
    for ln in open(path):
        if ln.startswith(("ATOM", "HETATM")):
            atom = ln[12:16].strip()
            elem = (atom[0] if atom else "C").upper()
            lines.append(ln[:66].rstrip("\n").ljust(66) + "          " + f"{elem:>2}" + "\n")
    lines.append("END\n")
    return "".join(lines)


@dataclass
class ReceptorPocket:
    """Pre-parsed receptor: minimal PDB text for 3Dmol + heavy-atom coords/resids for pocket calc."""
    pdb_text: str | None = None
    xyz: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    resi: list = field(default_factory=list)

    @property
    def available(self):
        return self.pdb_text is not None


def load_receptor_pocket(receptor_path):
    """Parse a .pdbqt receptor once into a :class:`ReceptorPocket` (pdb text + atom coords/resids)."""
    pocket = ReceptorPocket()
    if not (receptor_path and Path(receptor_path).exists()):
        return pocket
    pocket.pdb_text = pdbqt_to_pdb_text(receptor_path)
    xyz, resi = [], []
    for ln in open(receptor_path):
        if ln.startswith(("ATOM", "HETATM")):
            try:
                xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
                resi.append(int(ln[22:26]))
            except ValueError:
                continue
    pocket.xyz, pocket.resi = np.asarray(xyz), resi
    return pocket


def _conf_xyz(mol):
    c = mol.GetConformer()
    return np.asarray([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z]
                       for i in range(mol.GetNumAtoms())])


def pocket_residue_sel(mol, pocket, cutoff=5.0):
    """3Dmol selection ``{model, resi:[...]}`` for residues with any atom within ``cutoff`` Å of the
    ligand. Resolved in Python (not via a fragile 3Dmol ``within`` selector)."""
    if pocket.xyz.shape[0] == 0:
        return {"model": 0, "resi": []}
    lig = _conf_xyz(mol)
    min_d2 = ((pocket.xyz[:, None, :] - lig[None, :, :]) ** 2).sum(-1).min(axis=1)
    resis = sorted({pocket.resi[i] for i in np.where(min_d2 <= cutoff * cutoff)[0]})
    return {"model": 0, "resi": resis}


def render_pose_grid(hof, pocket, mols_per_row=2, cell_w=360, cell_h=320,
                     surface_opacity=0.45, pocket_dist=5.0):
    """py3Dmol grid, one molecule per cell: gray protein cartoon, cyan-carbon ligand sticks, and a
    translucent surface over the pocket residues. ``hof`` is a list of ``(mol_with_pose, props)``
    from :func:`nbtools.hall_of_fame.load_hall_of_fame`.

    The SDF stores ligand atoms at absolute coordinates in the receptor frame, so no alignment is
    needed — loading both into one scene drops the ligand straight into the pocket.
    """
    n = len(hof)
    rows = (n + mols_per_row - 1) // mols_per_row
    grid = py3Dmol.view(viewergrid=(rows, mols_per_row),
                        width=cell_w * mols_per_row, height=cell_h * rows)
    for k, (mol, props) in enumerate(hof):
        r, c = divmod(k, mols_per_row)
        if pocket.available:
            grid.addModel(pocket.pdb_text, "pdb", viewer=(r, c))
            grid.setStyle({"model": 0}, {"cartoon": {"color": "lightgray", "opacity": 0.7}},
                          viewer=(r, c))
        grid.addModel(Chem.MolToMolBlock(mol), "mol", viewer=(r, c))
        grid.setStyle({"model": -1}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.2}},
                      viewer=(r, c))
        if pocket.available:
            sel = pocket_residue_sel(mol, pocket, cutoff=pocket_dist)
            if sel["resi"]:
                grid.setStyle(sel, {"cartoon": {"color": "lightblue", "opacity": 0.5}}, viewer=(r, c))
                grid.addSurface(py3Dmol.MS, {"opacity": surface_opacity, "color": "lightblue"},
                                sel, viewer=(r, c))
        grid.addLabel(f"#{props.get('rank', '?')}  dock={props.get('docking_score', '?')}",
                      {"fontSize": 10, "backgroundColor": "black", "backgroundOpacity": 0.5,
                       "fontColor": "white"}, {"model": -1}, viewer=(r, c))
        grid.zoomTo({"model": -1}, viewer=(r, c))
        grid.zoom(0.8, viewer=(r, c))
    return grid

"""Highlight each ring system in a molecule and annotate its ChEMBL frequency.

draw_ring_systems(mol, lookup) -> (matplotlib.Figure, ring_data)
ring_data: list of (atom_indices, ring_smiles, chembl_count)
"""
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D
from useful_rdkit_utils import RingSystemLookup

_PALETTE = [(0.40,0.76,0.65),(0.99,0.55,0.38),(0.55,0.63,0.80),
            (0.91,0.54,0.76),(0.65,0.85,0.33),(1.00,0.85,0.18),
            (0.90,0.77,0.58),(0.70,0.70,0.70)]

def ring_systems_with_atoms(mol, lookup):
    """[(atom_indices, ring_smiles, chembl_count), ...]; counts aligned to lookup.process_mol."""
    rsf = lookup.ring_system_finder
    work = Chem.Mol(mol); n0 = work.GetNumAtoms()
    rsf.tag_bonds_to_preserve(work)
    cut = [b.GetIdx() for b in work.GetBonds()
           if not b.IsInRing() and not b.GetBoolProp("protected")
           and b.GetBondType() == Chem.BondType.SINGLE]
    fm = Chem.FragmentOnBonds(work, cut) if cut else work
    counts = dict(lookup.process_mol(Chem.Mol(mol)))
    out = []
    for idxs, frag in zip(Chem.GetMolFrags(fm, asMols=False, sanitizeFrags=False),
                          Chem.GetMolFrags(fm, asMols=True,  sanitizeFrags=False)):
        orig = [i for i in idxs if i < n0]
        if not any(work.GetAtomWithIdx(i).IsInRing() for i in orig):
            continue
        f = Chem.RWMol(frag)
        for a in f.GetAtoms():
            if a.GetAtomicNum() == 0: a.SetAtomicNum(1); a.SetIsotope(0)
        smi = Chem.MolToSmiles(rsf.fix_bond_stereo(Chem.RemoveAllHs(f)))
        out.append((orig, smi, counts.get(smi, 0)))
    return out

def draw_ring_systems(mol, lookup=None, size=(500, 440), rare_threshold=5):
    lookup = lookup or RingSystemLookup()
    mol = Chem.Mol(mol)
    rings = ring_systems_with_atoms(mol, lookup)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()

    hl_atoms, hl_bonds, a_cols, b_cols = [], [], {}, {}
    for k, (idxs, _s, _c) in enumerate(rings):
        col = _PALETTE[k % len(_PALETTE)]; s = set(idxs)
        for i in idxs: hl_atoms.append(i); a_cols[i] = col
        for b in mol.GetBonds():
            if b.GetBeginAtomIdx() in s and b.GetEndAtomIdx() in s:
                hl_bonds.append(b.GetIdx()); b_cols[b.GetIdx()] = col

    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().highlightBondWidthMultiplier = 12
    d.DrawMolecule(mol, highlightAtoms=hl_atoms, highlightBonds=hl_bonds,
                   highlightAtomColors=a_cols, highlightBondColors=b_cols)
    centroids = []                            # pixel coords for matplotlib overlay
    for idxs, _s, _c in rings:
        cx = sum(conf.GetAtomPosition(i).x for i in idxs) / len(idxs)
        cy = sum(conf.GetAtomPosition(i).y for i in idxs) / len(idxs)
        p = d.GetDrawCoords(Point2D(cx, cy)); centroids.append((p.x, p.y))
    d.FinishDrawing()
    img = Image.open(io.BytesIO(d.GetDrawingText()))

    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100 + 0.9), dpi=150)
    ax.imshow(img); ax.axis("off")
    for (px, py), (_i, _s, c) in zip(centroids, rings):
        ax.text(px, py, f"{c:,}", ha="center", va="center", fontsize=11, weight="bold",
                color="#111", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#444", alpha=0.85))
    handles = [mpatches.Patch(color=_PALETTE[k % len(_PALETTE)],
               label=f"{c:,}  ·  {s}" + ("  ⚠ rare" if c < rare_threshold else ""))
               for k, (_i, s, c) in enumerate(rings)]
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                  ncol=1, frameon=False, fontsize=9, handlelength=1.2,
                  title="ring system  ·  ChEMBL count")
    fig.tight_layout()
    return fig, rings

# if __name__ == "__main__":
#     lk = RingSystemLookup()
#     for name, smi in {"caffeine":"Cn1cnc2c1c(=O)n(C)c(=O)n2C",
#                       "benzene_piperidine":"c1ccccc1CCC2CCNCC2",
#                       "strained":"O=C1CC2(C1)C1=CC1C2"}.items():
#         fig, rings = draw_ring_systems(Chem.MolFromSmiles(smi), lk)
#         fig.savefig(f"{name}.png", bbox_inches="tight", dpi=150); plt.close(fig)
#         print(name, [(r[1], r[2]) for r in rings])
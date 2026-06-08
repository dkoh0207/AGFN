"""Render a GFlowNet molecular-generation trajectory as a 'growing molecule' GIF.

Atom identity is tracked by graph node label (stamped as an atom-map number that survives
``graph_to_mol``'s SMILES round-trip), so every atom is pinned to the position it occupies in the
finished molecule on a fixed canvas — the structure accretes in place without jumping or rescaling.
"""

import io
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D, Point3D
from PIL import Image as PILImage


def describe_action(act):
    """Human-readable caption for the GraphAction that produced the current state."""
    if act is None:
        return "seed"
    name = getattr(act.action, "name", str(act.action))
    if name == "AddNode":
        return f"AddNode {act.value}"
    if name == "AddEdge":
        return "AddEdge (ring)"
    if name in ("SetNodeAttr", "SetEdgeAttr"):
        kind = "atom" if name == "SetNodeAttr" else "bond"
        return f"Set {kind} {act.attr}={act.value}"
    return name  # Stop / Remove* etc.


def trajectory_to_gif(traj, ctx, out_path, frame_size=(440, 380),
                      frame_ms=650, end_hold_ms=2200,
                      highlight=True, annotate_action=True, pad=1.2):
    """Render a GFlowNet molecular-generation trajectory as a 'growing molecule' GIF.

    Each frame is the partial molecule built so far, with every atom pinned to the position it
    occupies in the finished molecule on a fixed canvas, so the structure accretes *in place*
    without jumping or zooming. Frames are captioned with the build action and the atom/bond
    changed at that step is highlighted in orange.

    traj     : list[(Graph, GraphAction)] e.g. ``valid[sel]["traj"]``; ``traj[k][0]`` is the
               pre-action graph at step k.
    ctx      : graph context (``finetuner.ctx``) providing the atom/bond conventions.
    out_path : str | Path  where to write the .gif.
    frame_ms / end_hold_ms : per-step / final-frame display time (ms).
    highlight / annotate_action : colour the atom-bond changed each step / caption the action.
    pad      : Angstrom padding around the final molecule's coordinate window.
    Returns the Path written.
    """
    out_path = Path(out_path)
    W, H = frame_size
    wildcard = getattr(ctx, "default_wildcard_replacement", "*")

    # Build a molecule from a step graph exactly as ctx.graph_to_mol does, but stamp every atom
    # with map num = node_label+1. The map numbers survive the SMILES round-trip (which reorders
    # atoms), so we can recover each drawn atom's graph-node identity. That yields a STABLE atom
    # correspondence across frames with NO substructure matching -- matching would pick an
    # arbitrary embedding and make symmetric rings (e.g. a benzene) jump between frames. Atoms
    # are added in node-label order, so node label i == RWMol atom index i before the round-trip.
    def tagged_mol(g):
        mp = Chem.RWMol()
        mp.BeginBatchEdit()
        for i in range(len(g.nodes)):
            d = g.nodes[i]
            s = d.get("fill_wildcard", d["v"])
            a = Chem.Atom(s if s is not None else wildcard)
            if "chi" in d:
                a.SetChiralTag(d["chi"])
            if "charge" in d:
                a.SetFormalCharge(d["charge"])
            if "expl_H" in d:
                a.SetNumExplicitHs(d["expl_H"])
            if "no_impl" in d:
                a.SetNoImplicit(d["no_impl"])
            a.SetAtomMapNum(i + 1)
            mp.AddAtom(a)
        for e in g.edges:
            mp.AddBond(e[0], e[1], g.edges[e].get("type", Chem.BondType.SINGLE))
        mp.CommitBatchEdit()
        try:
            Chem.SanitizeMol(mp)
            return Chem.MolFromSmiles(Chem.MolToSmiles(mp))
        except Exception:
            return None  # rare unsanitizable intermediate -> just skip that frame

    # 1) tagged mol per non-empty step graph + its node->atom map + the action that produced it
    #    (traj[j-1][1]; Stop is never a frame). Forward trajectories never remove nodes, so node
    #    sets grow monotonically -> the last renderable frame is the node-superset for the layout.
    frames_data = []
    for j, (g, _next_action) in enumerate(traj):
        if len(g.nodes) == 0:
            continue
        m = tagged_mol(g)
        if m is None:
            continue
        n2a = {a.GetAtomMapNum() - 1: a.GetIdx() for a in m.GetAtoms()}
        frames_data.append((m, n2a, traj[j - 1][1] if j >= 1 else None))
    if not frames_data:
        raise ValueError("trajectory has no renderable steps")

    # 2) Lay out the final molecule once; keep its coordinates keyed by node identity, and lock
    #    the coordinate window to its bounds so RDKit never re-centres/zooms a partial structure.
    final, final_n2a, _ = frames_data[-1]
    rdDepictor.Compute2DCoords(final)
    fc = final.GetConformer()
    node_xy = {nd: (fc.GetAtomPosition(ai).x, fc.GetAtomPosition(ai).y)
               for nd, ai in final_n2a.items()}
    xs = [p[0] for p in node_xy.values()]
    ys = [p[1] for p in node_xy.values()]
    minv = Point2D(min(xs) - pad, min(ys) - pad)
    maxv = Point2D(max(xs) + pad, max(ys) + pad)

    # 3) Pin each frame's atoms to their final positions BY NODE IDENTITY (no matching), then
    #    clear the map numbers so they don't render as ":n" labels on the drawing.
    for m, n2a, _ in frames_data:
        conf = Chem.Conformer(m.GetNumAtoms())
        conf.Set3D(False)
        for nd, ai in n2a.items():
            x, y = node_xy[nd]
            conf.SetAtomPosition(ai, Point3D(x, y, 0.0))
        m.AddConformer(conf, assignId=True)
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)

    # 4) Highlight what each step changed, via node identity: atoms whose node is new (AddNode),
    #    and bonds whose node-pair is new (AddNode / ring-closing AddEdge) or whose order changed
    #    (SetEdgeAttr, e.g. forming a double bond / aromatizing a ring).
    def changed(prev, prev_n2a, cur, cur_n2a):
        if prev is None:
            return list(range(cur.GetNumAtoms())), []
        prev_nodes = set(prev_n2a)
        new_atoms = [cur_n2a[nd] for nd in cur_n2a if nd not in prev_nodes]
        a2n_prev = {ai: nd for nd, ai in prev_n2a.items()}
        prev_bt = {}
        for b in prev.GetBonds():
            key = frozenset((a2n_prev[b.GetBeginAtomIdx()], a2n_prev[b.GetEndAtomIdx()]))
            prev_bt[key] = b.GetBondType()
        a2n_cur = {ai: nd for nd, ai in cur_n2a.items()}
        new_bonds = []
        for b in cur.GetBonds():
            key = frozenset((a2n_cur[b.GetBeginAtomIdx()], a2n_cur[b.GetEndAtomIdx()]))
            if key not in prev_bt or prev_bt[key] != b.GetBondType():
                new_bonds.append(b.GetIdx())
        return new_atoms, new_bonds

    # 5) Render every frame at the fixed scale, highlighting what changed.
    hl = (1.0, 0.55, 0.0)  # orange
    frames, prev, prev_n2a = [], None, None
    for i, (m, n2a, act) in enumerate(frames_data):
        ha, hb = (changed(prev, prev_n2a, m, n2a) if highlight else ([], []))
        d = rdMolDraw2D.MolDraw2DCairo(W, H)
        d.SetScale(W, H, minv, maxv, m)              # pin world->pixel mapping to the final window
        legend = f"t={i}   {describe_action(act)}" if annotate_action else f"t={i}"
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, m, legend=legend,
            highlightAtoms=ha, highlightBonds=hb,
            highlightAtomColors={a: hl for a in ha},
            highlightBondColors={b: hl for b in hb},
        )
        d.FinishDrawing()
        frames.append(PILImage.open(io.BytesIO(d.GetDrawingText())).convert("RGB"))
        prev, prev_n2a = m, n2a

    # 6) Stitch into a looping GIF, holding the final frame longer.
    durations = [frame_ms] * (len(frames) - 1) + [end_hold_ms]
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=2)
    return out_path

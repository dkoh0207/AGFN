"""Docking-box / pocket diagnostics — pure NumPy, no repo (``src/``) dependencies.

Parses PDB/PDBQT coordinates, reads the configured Uni-Dock box from a denovo config, and checks
(and, optionally, cross-frame-corrects) that the box actually sits on the binding site.
"""

import numpy as np


def read_atoms(path, records=("ATOM",)):
    """Parse fixed-column PDB/PDBQT coordinates. Returns ``(coords[N,3], keys[N])`` where each key
    is ``(chain, resSeq, atom_name)`` so atoms can be matched across files."""
    coords, keys = [], []
    for ln in open(path):
        if ln[:6].strip() in records:
            try:
                xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            except ValueError:
                continue
            coords.append(xyz)
            keys.append((ln[21], ln[22:27].strip(), ln[12:16].strip()))
    return np.array(coords), keys


def read_ligand(path, resn):
    """Coordinates of a single HETATM residue (e.g. the co-crystal inhibitor)."""
    coords = []
    for ln in open(path):
        if ln[:6].strip() == "HETATM" and ln[17:20].strip() == resn:
            try:
                coords.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            except ValueError:
                pass
    return np.array(coords)


def box_diag(coords, c, s):
    """How a box sits on an atom cloud: ``(#inside, fraction_inside, nearest_atom_distance)``."""
    lo, hi = c - s / 2, c + s / 2
    inside = ((coords >= lo) & (coords <= hi)).all(1)
    nearest = float(np.linalg.norm(coords - c, axis=1).min())
    return int(inside.sum()), float(inside.mean()), nearest


def kabsch(P, Q):
    """Optimal rigid transform mapping P onto Q. Returns ``(R, t, rmsd)``; ``Q ~= P @ R.T + t``."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    rmsd = float(np.sqrt((((R @ P.T).T + t - Q) ** 2).sum(1).mean()))
    return R, t, rmsd


def iou(c1, c2, s):
    """IoU of two equal-size axis-aligned boxes. (Box B's rotation is dropped for this glyph-level
    metric; its center is exact.)"""
    lo = np.maximum(c1 - s / 2, c2 - s / 2)
    hi = np.minimum(c1 + s / 2, c2 + s / 2)
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    vol = float(np.prod(s))
    return inter / (2 * vol - inter)


def read_box_from_config(config_path):
    """Read the configured Uni-Dock box exactly the way the training backend does.

    Returns ``(hps, center, size, receptor, target_name)`` with ``center``/``size`` as float arrays.
    """
    import yaml
    from easydict import EasyDict

    hps = EasyDict(yaml.safe_load(open(config_path))).finetuning
    grid = dict(hps.target_grid[hps.target_name])
    center = np.array([grid["center_x"], grid["center_y"], grid["center_z"]], float)
    size = np.array([grid["size_x"], grid["size_y"], grid["size_z"]], float)
    return hps, center, size, grid["receptor"], hps.target_name


def primary_box_check(dock_frame_path, rcsb_path, ligand_resn, center, size, verbose=True):
    """Primary check (no second structure needed): does the configured box sit on the receptor and
    enclose the reference co-crystal ligand? Everything is in the docking (RCSB) frame.

    Returns a dict with the receptor/ligand enclosure stats, ``center_to_lig``, ``iou``, the
    ligand centroid (``lig_center``, a paste-ready re-centered box), and the boolean ``ok`` verdict.
    """
    receptor, _ = read_atoms(dock_frame_path)
    lig = read_ligand(rcsb_path, ligand_resn)
    if len(lig) == 0:
        raise ValueError(f"No HETATM residue {ligand_resn!r} found in {rcsb_path}; "
                         f"set ligand_resn to the co-crystal ligand's residue name.")
    lig_center = lig.mean(0)

    rec_in, rec_frac, rec_near = box_diag(receptor, center, size)
    lig_in, lig_frac, _ = box_diag(lig, center, size)
    center_to_lig = float(np.linalg.norm(center - lig_center))
    iou_val = iou(center, lig_center, size)
    ok = lig_frac >= 0.8 and center_to_lig <= float(size.min()) / 2

    result = {"receptor": receptor, "lig": lig, "lig_center": lig_center,
              "rec_in": rec_in, "rec_frac": rec_frac, "rec_near": rec_near,
              "lig_in": lig_in, "lig_frac": lig_frac, "center_to_lig": center_to_lig,
              "iou": iou_val, "ok": ok}
    if verbose:
        print(f"configured center   : {np.round(center, 2)}   size {np.round(size, 2)}")
        print(f"receptor ({dock_frame_path.split('/')[-1]}): {len(receptor)} atoms")
        print(f"  atoms inside box  : {rec_in} ({100 * rec_frac:.1f}%)   nearest atom -> center: {rec_near:.1f} A")
        print(f"reference ligand {ligand_resn}: {len(lig)} atoms, centroid {np.round(lig_center, 1)}")
        print(f"  atoms inside box  : {lig_in}/{len(lig)} ({100 * lig_frac:.0f}% enclosed)")
        print(f"  center -> ligand centroid: {center_to_lig:.1f} A")
        print(f"  IoU(configured box, ligand-centered box): {iou_val:.2f}")
        print()
        if ok:
            print("VERDICT: OK -- the configured box encloses the reference ligand and is centered on it.")
        else:
            print("VERDICT: CHECK -- the configured box does NOT enclose the reference ligand well.")
            print("  Either the box center/size is off, or the coords are in the wrong frame.")
            print("  Try the ligand-centered box below, or run the optional BIOVIA cross-frame diagnostic.")
        print()
        print("--- paste-ready ligand-centered box (size unchanged) ---")
        print(f"  center_x: {lig_center[0]:.4f}  center_y: {lig_center[1]:.4f}  center_z: {lig_center[2]:.4f}")
    return result


def crossframe_box(biovia_path, dock_frame_path, center, size, receptor, box_C, verbose=True):
    """Optional cross-frame diagnostic: superpose a second structure (a BIOVIA-frame PDB) onto the
    docking receptor by matching atoms on ``(chain, resSeq, atom_name)``, then map the configured
    center into the docking frame (box **B**). ``box_C`` is the MLI-2 ligand centroid (ground truth).

    Returns box B (np.ndarray) — whichever of A (as-written) or B lands on C is the center truly in
    the docking frame. Returns ``None`` if the BIOVIA frame isn't usable.
    """
    B_xyz, B_keys = read_atoms(biovia_path)
    R_xyz, R_keys = read_atoms(dock_frame_path)
    r_index = {k: i for i, k in enumerate(R_keys)}
    pairs = [(B_xyz[i], R_xyz[r_index[k]]) for i, k in enumerate(B_keys) if k in r_index]
    if not pairs:
        if verbose:
            print("no atoms matched between BIOVIA and RCSB -> cannot superpose.")
        return None
    P = np.array([p for p, _ in pairs])
    Q = np.array([q for _, q in pairs])
    Rot, trans, rmsd = kabsch(P, Q)
    box_B = Rot @ center + trans

    if verbose:
        verdict = "same protein" if rmsd < 2 else "CHECK -- high RMSD"
        print(f"superposition BIOVIA -> RCSB: {len(P)} atoms matched, RMSD {rmsd:.3f} A ({verdict})")
        print()
        print(f"{'box':<38}{'center (RCSB frame)':<24}{'in-box':>8}{'->lig':>8}")
        for name, c in [("A as-written (docking uses this)", center),
                        ("B frame-corrected (A as BIOVIA)",  box_B),
                        ("C ligand-defined (MLI-2 centroid)", box_C)]:
            n_in, _, _ = box_diag(receptor, c, size)
            print(f"{name:<38}{str(np.round(c, 1)):<24}{n_in:>8}{np.linalg.norm(c - box_C):>7.1f}A")
        print()
        if np.linalg.norm(center - box_C) <= np.linalg.norm(box_B - box_C):
            print("=> box A (as-written) is closest to the MLI-2 pocket: the configured box is ALREADY in")
            print("   the docking frame -- no fix needed (box B is just the wrong-frame image here).")
        else:
            print("=> box B is closest to the MLI-2 pocket: the configured coords are in the BIOVIA frame.")
            print("   Replace center_x/y/z with box B below (size unchanged).")
        print()
        print("--- paste-ready centers (size unchanged) ---")
        for nm, c in [("frame-corrected (box B)", box_B), ("MLI-2 site (box C)", box_C)]:
            print(f"  {nm:<26} center_x: {c[0]:.4f}  center_y: {c[1]:.4f}  center_z: {c[2]:.4f}")
    return box_B

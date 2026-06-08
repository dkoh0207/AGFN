"""Frozen-core validation + visualization for fixed-seed (frozen-core) generation.

Three checks per generated molecule (see check_frozen_core.ipynb for the full rationale):
  A — the subgraph on labels ``0..N-1`` equals the seed (exact, attribute-level).
  B — ``generated_mol.HasSubstructMatch(seed_mol)`` (chemistry-level).
  C — the atoms located by reserved atom-map tag are exactly one of the seed's substructure
      matches (identity-level), across several independent rollouts.

The core is located by its reserved map number (``FROZEN_MAP_BASE + label``), which survives
``graph_to_mol``'s canonical round-trip, NOT by RDKit index. Repo helpers are imported lazily.
"""

from .sampling import sample_trajectories


def check_core_intact_graph(g, seed_graph, N):
    """Check A for one graph: the induced subgraph on labels ``0..N-1`` equals the seed exactly
    (nodes, edges, and their attribute dicts)."""
    if not all(k in g.nodes for k in range(N)):
        return False
    for k in range(N):
        if dict(g.nodes[k]) != dict(seed_graph.nodes[k]):
            return False
    seed_edges = {frozenset(e): dict(seed_graph.edges[e])
                  for e in seed_graph.edges if e[0] < N and e[1] < N}
    g_edges = {frozenset(e): dict(g.edges[e]) for e in g.edges if e[0] < N and e[1] < N}
    return seed_edges == g_edges


def check_substructure(mols, seed_mol):
    """Check B: how many molecules contain ``seed_mol`` as a substructure. Returns the pass count."""
    return sum(m is not None and m.HasSubstructMatch(seed_mol) for m in mols)


def check_labeled_core(sampler, n_samples, rounds, seed_mol, N, verbose=True):
    """Check C: re-sample ``rounds`` independent batches and, for every generated molecule, confirm
    the atoms located by reserved map number are exactly one of the seed's substructure matches.

    Returns a dict: ``total``, ``c_count`` (exact-N core located), ``c_redseed`` (located core ==
    seed substructure), ``grew_c`` (grew beyond the seed), and the ``success`` / ``failure`` lists
    of ``(tagged_mol, seed_mol)`` pairs.
    """
    from gflownet.envs.mol_building_env import frozen_core_atoms, frozen_core_index_map

    total = c_count = c_redseed = grew_c = 0
    success, failure = [], []
    for rd in range(rounds):
        trajs = sample_trajectories(sampler, n_samples, seed_graph=sampler.seed_graph,
                                    require_logprob=True)
        graphs = [t["traj"][-1][0] for t in trajs]
        for g in graphs:
            m = sampler.ctx.graph_to_mol(g, tag_frozen=True)
            core = frozen_core_atoms(m)                          # the atoms we highlight
            total += 1
            grew_c += len(g.nodes) > N
            if len(core) == N and sorted(frozen_core_index_map(m).values()) == list(range(N)):
                c_count += 1
            if set(core) in [set(mt) for mt in m.GetSubstructMatches(seed_mol)]:
                c_redseed += 1
                success.append((m, seed_mol))
            else:
                failure.append((m, seed_mol))
        if verbose:
            print(f"round {rd + 1}: cumulative {total} mols | exact-N core {c_count} | red==seed {c_redseed}")

    if verbose:
        print(f"\nCheck C across {rounds} rollouts ({total} molecules, {grew_c} grew beyond seed):")
        print(f"  highlighted core has all N seed atoms : {c_count}/{total}")
        print(f"  highlighted core == seed substructure : {c_redseed}/{total}")
    return {"total": total, "c_count": c_count, "c_redseed": c_redseed, "grew_c": grew_c,
            "success": success, "failure": failure}


def grown_core_grid(tagged_mols, N, n=6, mols_per_row=3, sub_img_size=(320, 260), save_path=None):
    """Grid of grown molecules with the frozen core highlighted (located by map number, not index).
    Core map tags are cleared for a clean depiction; legends are the canonical SMILES."""
    from gflownet.envs.mol_building_env import frozen_core_atoms, strip_frozen_maps
    from rdkit import Chem
    from .render2d import mol_grid

    show = [m for m in tagged_mols if m is not None and m.GetNumAtoms() > N][:n]
    highlights, clean, legends = [], [], []
    for m in show:
        highlights.append(frozen_core_atoms(m))
        d = Chem.Mol(m)
        for a in d.GetAtoms():
            a.SetAtomMapNum(0)                                  # clean labels, same atom order
        clean.append(d)
        legends.append(Chem.MolToSmiles(strip_frozen_maps(m)))
    return mol_grid(clean, legends=legends, mols_per_row=mols_per_row, sub_img_size=sub_img_size,
                    highlight_atom_lists=highlights, save_path=save_path)


def grown_core_svgs(tagged_mols, N, n=6, size=(360, 280)):
    """Per-molecule SVGs: highlight the core (by map number) and annotate each core atom with its
    ORIGINAL seed index (recovered from the map number), wherever the round-trip placed it."""
    from gflownet.envs.mol_building_env import frozen_core_atoms, frozen_core_index_map
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from IPython.display import SVG

    show = [m for m in tagged_mols if m is not None and m.GetNumAtoms() > N][:n]
    imgs = []
    for m in show:
        core = frozen_core_atoms(m)
        imap = frozen_core_index_map(m)                         # rdkit idx -> original seed index
        d_mol = Chem.Mol(m)
        for a in d_mol.GetAtoms():
            if a.GetIdx() in imap:
                a.SetProp("atomNote", str(imap[a.GetIdx()]))
            a.SetAtomMapNum(0)
        d = rdMolDraw2D.MolDraw2DSVG(*size)
        rdMolDraw2D.PrepareAndDrawMolecule(d, d_mol, highlightAtoms=core)
        d.FinishDrawing()
        imgs.append(SVG(d.GetDrawingText()))
    return imgs

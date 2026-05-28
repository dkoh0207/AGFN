"""Tests for the frozen-core / fixed-seed feature.

No pytest in this env, so this is a plain script: run it directly, it raises on
failure and prints PASS lines on success.

    cd src && python ../tests/test_frozen_core.py

Covers:
  1. ``build_frozen_seed_graph`` spec parsing, validation, and the graph-level
     attributes it stamps onto the seed graph.
  2. ``MolBuildingEnvContext.graph_to_Data`` action masks honour the frozen core
     (and are a byte-identical no-op when no core is set).
  3. ``GraphBuildingEnv.step`` raises on any action that would edit the frozen core.
"""

import os
import sys

import numpy as np
import torch
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gflownet.envs.mol_building_env import MolBuildingEnvContext, build_frozen_seed_graph  # noqa: E402
from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphAction, GraphActionType  # noqa: E402

ATOMS = ["Br", "C", "Cl", "F", "I", "N", "O", "S"]


def _ctx():
    return MolBuildingEnvContext(atoms=ATOMS, num_cond_dim=2, num_rw_feat=0, max_nodes=45)


def _assert_raises(fn, *, contains=None):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        if contains is not None:
            assert contains.lower() in str(e).lower(), f"expected {contains!r} in {e!r}"
        return
    raise AssertionError("expected an exception, none raised")


# --------------------------------------------------------------------------- #
# 1. build_frozen_seed_graph
# --------------------------------------------------------------------------- #
def test_helper_allowed_growth_atoms():
    g = build_frozen_seed_graph(_ctx(), "c1ccccc1", allowed_growth_atoms=[0])
    assert g.graph["frozen_seed_size"] == 6
    assert g.graph["frozen_growth_sites"] == {0}
    assert set(g.nodes) == set(range(6))


def test_helper_frozen_atoms_is_complement():
    g = build_frozen_seed_graph(_ctx(), "c1ccccc1", frozen_atoms=[0, 1, 2, 3])
    assert g.graph["frozen_seed_size"] == 6
    assert g.graph["frozen_growth_sites"] == {4, 5}


def test_helper_default_all_sites_growable():
    g = build_frozen_seed_graph(_ctx(), "c1ccccc1")
    assert g.graph["frozen_growth_sites"] == set(range(6))


def test_helper_accepts_comma_string():
    g = build_frozen_seed_graph(_ctx(), "c1ccccc1", allowed_growth_atoms="0, 2 ,4")
    assert g.graph["frozen_growth_sites"] == {0, 2, 4}


def test_helper_accepts_scalar_int():
    # A single index is a natural YAML scalar (`allowed_growth_atoms: 0`); must not crash.
    g = build_frozen_seed_graph(_ctx(), "c1ccccc1", allowed_growth_atoms=0)
    assert g.graph["frozen_growth_sites"] == {0}


def test_helper_blank_string_treated_as_unset():
    # A blank YAML value means "no restriction" (whole core frozen, all atoms grow),
    # not "all atoms frozen".
    g = build_frozen_seed_graph(_ctx(), "c1ccccc1", allowed_growth_atoms="")
    assert g.graph["frozen_growth_sites"] == set(range(6))


def test_helper_rejects_both_lists():
    _assert_raises(
        lambda: build_frozen_seed_graph(_ctx(), "c1ccccc1", allowed_growth_atoms=[0], frozen_atoms=[1]),
        contains="only one",
    )


def test_helper_rejects_out_of_range():
    _assert_raises(lambda: build_frozen_seed_graph(_ctx(), "c1ccccc1", allowed_growth_atoms=[6]), contains="range")


def test_helper_rejects_all_frozen():
    _assert_raises(
        lambda: build_frozen_seed_graph(_ctx(), "c1ccccc1", frozen_atoms=[0, 1, 2, 3, 4, 5]),
        contains="growth",
    )


def test_helper_rejects_bad_smiles():
    _assert_raises(lambda: build_frozen_seed_graph(_ctx(), "not_a_smiles"), contains="smiles")


def test_helper_rejects_unknown_atom():
    # Phosphorus is absent from ATOMS above; should be a friendly error, not a KeyError deep in featurization.
    _assert_raises(lambda: build_frozen_seed_graph(_ctx(), "CP(C)C"), contains="atom")


# --------------------------------------------------------------------------- #
# 2. graph_to_Data frozen-core masking
# --------------------------------------------------------------------------- #
def _data(g, ctx):
    return ctx.graph_to_Data(g, cond_info=torch.zeros((1, ctx.num_cond_dim)))


def test_mask_freezes_all_core_node_attrs():
    ctx = _ctx()
    g = build_frozen_seed_graph(ctx, "c1ccccc1", allowed_growth_atoms=[0])
    d = _data(g, ctx)
    assert d.set_node_attr_mask[:6].sum().item() == 0  # every core atom's attrs are frozen


def test_mask_addnode_only_at_growth_site():
    ctx = _ctx()
    g = build_frozen_seed_graph(ctx, "c1ccccc1", allowed_growth_atoms=[0])
    d = _data(g, ctx)
    assert d.add_node_mask[1:6].sum().item() == 0  # non-growth core atoms cannot receive new atoms
    assert d.add_node_mask[0].sum().item() > 0  # the growth site still has valence room


def test_mask_freezes_core_bonds():
    ctx = _ctx()
    g = build_frozen_seed_graph(ctx, "c1ccccc1", allowed_growth_atoms=[0])
    d = _data(g, ctx)
    assert d.set_edge_attr_mask.sum().item() == 0  # all 6 bonds are core bonds -> frozen
    assert d.add_edge_mask.sum().item() == 0  # no new bond may form between two core atoms


def test_mask_is_noop_without_frozen_core():
    ctx = _ctx()
    g = ctx.mol_to_graph(Chem.MolFromSmiles("c1ccccc1"))  # plain seed, no frozen attrs
    d = _data(g, ctx)
    # Unconstrained: attrs are settable and growth is allowed somewhere.
    assert d.set_node_attr_mask.sum().item() > 0
    assert d.add_node_mask.sum().item() > 0


# --------------------------------------------------------------------------- #
# 3. env.step frozen-core guards (defense-in-depth behind the masks)
# --------------------------------------------------------------------------- #
def _seed_env_graph(growth):
    ctx = _ctx()
    g = build_frozen_seed_graph(ctx, "c1ccccc1", allowed_growth_atoms=growth)
    return GraphBuildingEnv(), g


def test_step_blocks_addnode_at_frozen_atom():
    env, g = _seed_env_graph([0])
    _assert_raises(lambda: env.step(g, GraphAction(GraphActionType.AddNode, source=1, value="C")))


def test_step_allows_addnode_at_growth_site_and_propagates_spec():
    env, g = _seed_env_graph([0])
    gp = env.step(g, GraphAction(GraphActionType.AddNode, source=0, value="C"))
    assert set(gp.nodes) == set(range(7))  # new atom appended as node 6
    assert gp.has_edge(0, 6)
    assert gp.graph["frozen_seed_size"] == 6  # spec rides along via g.copy()
    assert gp.graph["frozen_growth_sites"] == {0}


def test_step_allows_growth_on_newly_added_atom():
    env, g = _seed_env_graph([0])
    gp = env.step(g, GraphAction(GraphActionType.AddNode, source=0, value="C"))  # node 6 (non-core)
    gpp = env.step(gp, GraphAction(GraphActionType.AddNode, source=6, value="C"))  # may grow freely
    assert gpp.has_edge(6, 7)


def test_step_blocks_setnodeattr_on_core_atom():
    env, g = _seed_env_graph([0])  # even atom 0 (a growth site) keeps its identity frozen
    _assert_raises(lambda: env.step(g, GraphAction(GraphActionType.SetNodeAttr, source=0, attr="charge", value=1)))


def test_step_blocks_addedge_between_core_atoms():
    env, g = _seed_env_graph([0])
    _assert_raises(lambda: env.step(g, GraphAction(GraphActionType.AddEdge, source=0, target=2)))


def test_step_blocks_setedgeattr_on_core_bond():
    env, g = _seed_env_graph([0])
    _assert_raises(
        lambda: env.step(
            g, GraphAction(GraphActionType.SetEdgeAttr, source=0, target=1, attr="type", value=Chem.rdchem.BondType.DOUBLE)
        )
    )


def test_step_unconstrained_graph_allows_addnode():
    ctx = _ctx()
    g = ctx.mol_to_graph(Chem.MolFromSmiles("c1ccccc1"))  # no frozen attrs -> guards never fire
    gp = GraphBuildingEnv().step(g, GraphAction(GraphActionType.AddNode, source=1, value="C"))
    assert gp.has_edge(1, 6)


if __name__ == "__main__":
    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_"))
    for name in tests:
        getattr(mod, name)()
        print(f"PASS {name}")
    print(f"\nAll {len(tests)} frozen-core tests passed.")

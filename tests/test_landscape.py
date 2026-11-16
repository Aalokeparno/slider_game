"""landscape.py: the static trap landscape (Unit 1). Hand values from the task spec (verified independently).

Human networks in the spec's node order: Hierarchy 0-1, 0-2, 1-3, 1-4, 2-5, 2-6; CT = Hierarchy + 3-4, 5-6;
Ring = C7 (0-1-...-6-0); Path = P7 (0-1-...-6); Star = K1,6; Full = K7.
"""
import numpy as np
import pandas as pd
import pytest

from slider import landscape as L


def graph(name):
    return L.adjacency(L.HUMAN_EDGES[name])


def trap(name, state):
    return next(r for r in L.trap_landscape(graph(name)) if r["state"] == state)


@pytest.fixture(scope="module")
def tables():
    T, G, _ = L.build_tables()
    return T, G


def test_853_graphs_all_split(tables):
    T, G = tables
    assert len(G) == 853 and G["atlas_id"].is_unique
    assert set(G["split"]) == {"train", "test"}
    assert set(T["split"]) <= {"train", "test"}


def test_human_names(tables):
    _, G = tables
    assert sorted(G.loc[G["human_name"].fillna("") != "", "human_name"]) == sorted(L.HUMAN_EDGES)
    split = pd.read_csv(L.config.GRAPH_SPLIT)          # Path 3 names (no Path) agree with the isomorphism match
    named = split[split["human"].notna()].set_index("atlas")["human"]
    assert all(G.set_index("atlas_id").loc[a, "human_name"] == n for a, n in named.items())


@pytest.mark.parametrize("name, n", [("Hierarchy", 4), ("CT", 4), ("Ring", 28), ("Path", 14), ("Star", 0), ("Full", 0)])
def test_trap_counts(name, n):
    assert len(L.trap_landscape(graph(name))) == n


def check(r, **want):
    for key, value in want.items():
        assert r[key] == pytest.approx(value), key


def test_hierarchy_trap():
    check(trap("Hierarchy", "+-+--++"), B=1, P_size=2, traps_in_P=2, n_flat_exits=0, n_gates=10, W=0.2,
          a_c=0.5, progress_share_c=0)


def test_ct_trap():
    check(trap("CT", "+-+--++"), B=1, P_size=2, n_gates=2, W=1.0, a_c=0.5)


def test_ring_trap():
    check(trap("Ring", "++++---"), B=0, P_size=42, traps_in_P=28, n_flat_exits=14, n_gates=14, W=3.0,
          a_c=1 / 3, progress_share_c=0.5)


def test_path_trap():
    check(trap("Path", "+++----"), B=0, P_size=6, n_flat_exits=2, n_gates=2, W=3.0)


def test_mirror_symmetry(tables):
    T, _ = tables
    cols = ["B", "a_c", "P_size", "n_gates"]
    for _, g in T.groupby(["atlas_id", "pair_id"]):
        assert len(g) == 2
        assert np.allclose(g[cols].iloc[0].to_numpy(float), g[cols].iloc[1].to_numpy(float))


@pytest.mark.parametrize("name, lam2, cond, mod", [
    ("Hierarchy", 0.2679, 0.2, 0.3194), ("CT", 0.2679, 0.1429, 0.3672), ("Ring", 0.7530, 0.3333, 0.2245),
    ("Star", None, 1.0, 0.0), ("Full", 7.0, None, 0.0)])
def test_globals(name, lam2, cond, mod):
    A = graph(name)
    if lam2 is not None:
        assert round(L.lam2(A), 4) == lam2
    if cond is not None:
        assert round(L.conductance(A), 4) == cond
    assert round(L.modularity(A), 4) == mod


def test_partitions_and_lamP(tables):
    _, G = tables
    assert len(L.PARTITIONS) == 877
    split = pd.read_csv(L.config.GRAPH_SPLIT).set_index("atlas")      # Path 3 |lambda| is the same quantity
    assert np.allclose(G.set_index("atlas_id")["lamP"], split.loc[G["atlas_id"], "lam"])
    assert (G.set_index("atlas_id")["n_traps"] == split.loc[G["atlas_id"], "n_fraught"]).all()   # trap = fraught

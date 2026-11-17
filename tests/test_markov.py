"""markov.py: exact escape times, trap time and committor under the logit rule (Units 2a, 2b). Values from the specs.

Graph-train graphs only: the mirror and detailed-balance checks loop over graph-train graphs, and the guard test
only checks that a graph-test graph raises before anything is computed.
Node order 0..6: Hierarchy 0-1, 0-2, 1-3, 1-4, 2-5, 2-6; CT = Hierarchy + 3-4, 5-6; Ring = C7 (i - i+1 mod 7).
"""
import numpy as np
import pandas as pd
import pytest

from slider import config
from slider import landscape as L
from slider import markov as M

HIER = L.adjacency(L.HUMAN_EDGES["Hierarchy"])
CT = L.adjacency(L.HUMAN_EDGES["CT"])
RING = L.adjacency(L.HUMAN_EDGES["Ring"])
H_TRAP = L.state_from_signs([1, -1, 1, -1, -1, 1, 1])
R_TRAP = L.state_from_signs([1, 1, 1, 1, -1, -1, -1])


@pytest.fixture(scope="module")
def train():
    graphs = L.atlas_graphs()
    split = L.load_graph_split()
    return {a: A for a, A in graphs.items() if split[a] == "train"}


@pytest.mark.parametrize("A, s, T, want", [
    (HIER, H_TRAP, 1, 51.363), (HIER, H_TRAP, 0.5, 53.359), (HIER, H_TRAP, 0.1, 85225.19),
    (CT, H_TRAP, 1, 54.473), (CT, H_TRAP, 0.1, 385433.14),
    (RING, R_TRAP, 1, 48.477), (RING, R_TRAP, 0.1, 42.000)])
def test_tau_values(A, s, T, want):
    assert M.tau(A, T)[s] == pytest.approx(want, rel=1e-4)


def test_rows_and_consensus():
    P = M.transition_matrix(HIER, 0.3)
    assert np.allclose(P.sum(axis=1), 1) and (P >= 0).all()
    assert P[L.PLUS, L.PLUS] == 1 and P[L.MINUS, L.MINUS] == 1
    t = M.tau(HIER, 0.3)
    assert t[L.PLUS] == 0 and t[L.MINUS] == 0


def test_train_only(train):
    assert len(train) == 597


def test_mirror(train):
    for A in train.values():
        for T in M.T_GRID:
            t = M.tau(A, T)
            assert np.allclose(t, t[L.PLUS ^ L.STATES], rtol=1e-9)


def test_detailed_balance(train):
    S = M.TRANSIENT
    for A in train.values():
        cut = L.basics(A)[0]
        for T in M.T_GRID:
            P = M.transition_matrix(A, T)
            pi = np.exp(-cut / T)
            F = pi[:, None] * P
            F = F[np.ix_(S, S)]
            assert np.allclose(F, F.T, rtol=1e-9, atol=0)


def test_monte_carlo():
    x = M.monte_carlo(HIER, 1, H_TRAP, 4000, seed=config.SEED)
    assert abs(x.mean() - 51.363) < 3 * x.std(ddof=1) / np.sqrt(len(x))


def test_hierarchy_slope():
    slope = (np.log(M.tau(HIER, 0.1)[H_TRAP]) - np.log(M.tau(HIER, 0.15)[H_TRAP])) / (1 / 0.1 - 1 / 0.15)
    assert abs(slope - 1.0) < 0.01


def test_guard():
    graphs = L.atlas_graphs()
    split = L.load_graph_split()
    tests = [a for a in graphs if split[a] == "test"]
    assert len(tests) == 256
    for a in tests:
        with pytest.raises(ValueError):
            M.graph_tau(a, split, graphs)


# ---------------------------------------------------------------- Unit 2b

def human_case(name, sig):
    A = L.adjacency(L.HUMAN_EDGES[name])
    dcut = L.basics(A)[1]
    return A, L.traps(dcut), L.plateau(M.state_code(sig), dcut)


@pytest.fixture(scope="module")
def tables_2b():
    G, Pl, _ = M.build_trap_time_train()
    return G, Pl


@pytest.mark.parametrize("name, sig, T, want", [
    ("Hierarchy", "+-+--++", 0.2, (238.766, 222.017, 1.0614, 209.177, 0.3878, 592.021, 0.3654, 0.3654)),
    ("Ring", "++++---", 0.2, (34.054, 24.616, 1.6094, 15.295, 0.8045, 37.601, 0.9995, 0)),
    ("Hierarchy", "+-+--++", 1, (42.999, 6.501, 1.2929, 5.028, 0.5005, 51.363, 0.4573, 0.4573)),
    ("Ring", "++++---", 1, (41.981, 23.989, 2.8411, 8.444, 0.8320, 45.075, 0.4249, 0))])
def test_2b_values(name, sig, T, want):
    A, traps, P_states = human_case(name, sig)
    r = {**M.trap_time(A, T, traps), **M.committor(A, T, P_states)}
    keys = ["Yc", "Yb", "entries", "duration", "P_hit", "Ya", "qbar", "q_up"]
    for key, value in zip(keys, want):
        assert r[key] == pytest.approx(value, rel=1e-3, abs=1e-12), key


def test_2b_plateaus():
    A, _, P_states = human_case("Hierarchy", "+-+--++")
    assert len(P_states) == 2
    A_r, _, P_r = human_case("Ring", "++++---")
    assert len(P_r) == 42
    for T in M.T_GRID:
        assert np.isnan(M.committor(A, T, P_states)["q_down"])
        r = M.committor(A_r, T, P_r)
        assert abs(r["q_up"]) < 1e-12 and r["q_down"] == pytest.approx(1)


def test_2b_tables(tables_2b):
    G, Pl = tables_2b
    split = L.load_graph_split()
    assert G["atlas_id"].nunique() == 597 and {split[a] for a in G["atlas_id"]} == {"train"}
    assert len(G) == 597 * len(M.T_GRID)
    no_trap = pd.read_csv(config.U1_GRAPHS).query("n_traps == 0")["atlas_id"]
    z = G[G["atlas_id"].isin(no_trap)]
    assert len(z) > 0 and (z["Yb"] == 0).all() and (z["entries"] == 0).all() and (z["P_hit"] == 0).all()
    assert z["Ya"].isna().all()
    assert (G["Yb"] <= G["Yc"]).all()
    u1 = pd.read_csv(config.U1_TRAPS).query("split == 'train'")
    assert len(Pl) == u1.groupby(["atlas_id", "plateau_id"]).ngroups * len(M.T_GRID)


def test_2b_monte_carlo():
    A, traps, _ = human_case("Ring", "++++---")
    h = M.monte_carlo_hit(A, 1, traps, 4000, seed=config.SEED)
    assert abs(h.mean() - 0.8320) < 3 * h.std(ddof=1) / np.sqrt(len(h))


def test_2b_guard():
    graphs = L.atlas_graphs()
    split = L.load_graph_split()
    u1 = pd.read_csv(config.U1_TRAPS)
    for a in [a for a in graphs if split[a] == "test"]:
        with pytest.raises(ValueError):
            M.graph_trap_time(a, split, graphs, u1)

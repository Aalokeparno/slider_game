"""markov.py: exact escape times under the logit rule (Unit 2a). Hand values from the task spec.

Graph-train graphs only: the mirror and detailed-balance checks loop over graph-train graphs, and the guard test
only checks that a graph-test graph raises before anything is computed.
Node order 0..6: Hierarchy 0-1, 0-2, 1-3, 1-4, 2-5, 2-6; CT = Hierarchy + 3-4, 5-6; Ring = C7 (i - i+1 mod 7).
"""
import numpy as np
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

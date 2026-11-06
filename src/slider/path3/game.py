"""Path 3: the game on the lab v09 interface, and the graph measures.

Positions are kept as integers k = 10 s (k in -10..10), so the 0.1 grid is exact.
    dt = 0.1 s, one 0.1 step per tick (1 unit/s; full crossing 2 s): lab v09 modal step and speed (interface.py).
    Agr_i = |s_i + sum of s_j over the neighbours j| / (d_i + 1), recomputed every tick.
    display "int": players read Agr as an integer percent (round half up); "raw": unrounded.
    Consensus: |sum of s| / 7 >= 0.99.
"""
import itertools

import numpy as np

from slider.models.apem import GRAPHS

DT = 0.1           # s per tick
STEP = 0.1         # one step on the slider (k = 1)
WALL = 10          # k at the walls
N = 7

HUMAN_GRAPHS = {"Hierarchy": GRAPHS["Hierarchy"], "Full": GRAPHS["Full"], "Ring": GRAPHS["Ring"],
                "Star": GRAPHS["Star"], "CT": GRAPHS["CT-Hierarchy"]}


def agreement(A, k, display="int"):
    """Agr of every player at positions k (integers, s = k / 10).
    display="int": integer percent, half up, as a fraction (e.g. 0.33); "raw": the exact value."""
    members = A.sum(axis=1) + 1
    total = np.abs(k + A @ k)                            # 10 x |neighbourhood sum|
    if display == "raw":
        return total / (10 * members)
    return np.floor(10 * total / members + 0.5 + 1e-9) / 100


def is_consensus(k):
    return abs(int(np.sum(k))) >= 0.99 * N * WALL


def corner_from_sets(plus, minus):
    """Corner (+1 / -1 per player) from 1-based player sets."""
    c = np.zeros(N, dtype=int)
    c[[i - 1 for i in plus]] = 1
    c[[i - 1 for i in minus]] = -1
    return c


def non_consensus_corners():
    """The 126 corners that are not all +1 or all -1, in a fixed order."""
    return [np.array(c) for c in itertools.product([-1, 1], repeat=N) if abs(sum(c)) < N]


def corner_is_fraught(A, c):
    """Every player has at least as many same-side neighbours as other-side neighbours."""
    return bool(np.all(c * (A @ c) >= 0))


def fraught_corners(A):
    return [c for c in non_consensus_corners() if corner_is_fraught(A, c)]


def second_eigenvalue(A):
    """|lambda| = the largest |eigenvalue| other than the eigenvalue 1 of P = D^-1 (A + I), D = diag(d_i + 1).
    P is similar to a symmetric matrix, so its eigenvalues are real; for a connected graph 1 is simple."""
    d = A.sum(axis=1) + 1
    S = (A + np.eye(len(A))) / np.sqrt(np.outer(d, d))
    ev = np.sort(np.linalg.eigvalsh(S))                  # the largest is 1
    return float(max(abs(ev[0]), abs(ev[-2])))


def fraught_now(A, k):
    """Per tick (rows of k): the sign pattern is a fraught non-consensus corner and every |s_i| >= 0.5."""
    c = np.sign(k)
    corner = np.all(c != 0, axis=1) & (np.abs(c.sum(axis=1)) < N)
    ok = np.all(c * (c @ A) >= 0, axis=1)               # A is symmetric
    return corner & ok & np.all(np.abs(k) >= 5, axis=1)

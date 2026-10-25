"""APEM simulator for the Slider Game (HW 5.2).

A run is returned as a dict like the human runs from slider_logs.py
({"A", "t", "X", "weights", "success"}), so fraught_mask and the
forward-hold % fraught from stage1_hw3.ipynb work on it unchanged.
"""
import numpy as np


def adjacency(edges, n=7):
    # 7x7 adjacency matrix from a 1-indexed edge list (no self-loops)
    A = np.zeros((n, n))
    for i, j in edges:
        A[i - 1, j - 1] = 1
        A[j - 1, i - 1] = 1
    return A


HIERARCHY = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]

GRAPHS = {
    "Full":         adjacency([(i, j) for i in range(1, 8) for j in range(i + 1, 8)]),
    "Star":         adjacency([(1, j) for j in range(2, 8)]),
    "Hierarchy":    adjacency(HIERARCHY),
    "CT-Hierarchy": adjacency(HIERARCHY + [(4, 5), (6, 7)]),
    "Ring":         adjacency([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 1)]),
}


def apem(A, a_th=0.5, s_th=0.5, mu=0.1, speed=1.0, T=300, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(A)
    deg = A.sum(axis=1)

    s = np.zeros(n)                         # positions start in the middle
    phi = rng.choice([-1.0, 1.0], size=n)   # random starting directions
    t = 0.0
    ts, X = [t], [s.copy()]                 # record (t, s)
    success = False

    while True:
        dt = rng.exponential(mu / n)        # time to the next check-in

        if t + dt > T:                      # time is up: move to T, record, stop (failed)
            s = np.clip(s + speed * (T - t) * phi, -1, 1)
            t = T
            ts.append(t)
            X.append(s.copy())
            break

        s = np.clip(s + speed * dt * phi, -1, 1)   # everyone moves
        t = t + dt

        i = rng.integers(n)                         # who checks in
        A_i = abs(s[i] + A[i] @ s) / (deg[i] + 1)   # agreement of i with its neighbourhood
        if A_i <= a_th and abs(s[i]) >= s_th:       # low agreement and far out: turn around
            phi[i] = -phi[i]

        ts.append(t)                                # record (t, s)
        X.append(s.copy())

        if np.all(s == 1) or np.all(s == -1):       # consensus: stop (success)
            success = True
            break

    return {"A": A, "t": np.array(ts), "X": np.array(X), "weights": np.ones(n), "success": success}

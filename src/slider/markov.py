"""Unit 2a: exact expected escape times under the logit rule (graph-train graphs only).

States and their order are those of landscape.py (state s = 0..127, bit i set = node i is +1).
One step: pick a node i uniformly (prob 1/7); it flips with prob p_i = 1 / (1 + exp(dcut_i / T)),
dcut_i = F_i - E_i = change in cut if i flips; otherwise nothing happens (the step still counts).
The two consensus states are absorbing. Q = P on the 126 non-consensus states;
tau = (I - Q)^-1 1 = expected number of steps to consensus (0 at consensus).
"""
import time

import numpy as np
import pandas as pd
from scipy.special import expit

from slider import config
from slider import landscape as L

T_GRID = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 5]
ABSORBING = [L.MINUS, L.PLUS]
TRANSIENT = np.array([s for s in L.STATES if s not in ABSORBING])
FLIPS = L.STATES[:, None] ^ (1 << np.arange(L.N))           # FLIPS[s, i] = flip(s, i)


def transition_matrix(A, T):
    """128 x 128 transition matrix P of the logit rule at temperature T (consensus rows absorbing)."""
    _, dcut, _, _ = L.basics(A)
    P = np.zeros((len(L.STATES), len(L.STATES)))
    P[L.STATES[:, None], FLIPS] = expit(-dcut / T) / L.N      # = (1/7) / (1 + exp(dcut / T))
    P[ABSORBING] = 0.0
    P[L.STATES, L.STATES] = 1 - P.sum(axis=1)                 # no flip (1 at consensus)
    return P


def tau(A, T, tol=1e-8):
    """Expected steps to consensus from every state (array of 128; 0 at consensus); linear solve, float64.
    Raises if the relative residual ||(I - Q) tau - 1||_2 / ||1||_2 is >= tol."""
    P = transition_matrix(A, T)
    Q = P[np.ix_(TRANSIENT, TRANSIENT)]
    M = np.eye(len(TRANSIENT)) - Q
    one = np.ones(len(TRANSIENT))
    x = np.linalg.solve(M, one)
    resid = np.linalg.norm(M @ x - one) / np.linalg.norm(one)
    if resid >= tol:
        raise RuntimeError(f"relative residual {resid:.2e} >= {tol}")
    out = np.zeros(len(L.STATES))
    out[TRANSIENT] = x
    return out


def monte_carlo(A, T, start, n_runs, seed=config.SEED):
    """Steps to consensus of n_runs simulated chains from state start (array of n_runs)."""
    rng = np.random.default_rng(seed)
    P = transition_matrix(A, T)
    cum = np.cumsum(P, axis=1)
    steps = np.zeros(n_runs, dtype=int)
    for r in range(n_runs):
        s = start
        while s not in ABSORBING:
            s = min(int(np.searchsorted(cum[s], rng.random(), side="right")), len(L.STATES) - 1)
            steps[r] += 1
    return steps


def graph_tau(atlas_id, split, graphs, T_grid=T_GRID, allow_test=False):
    """Long table (atlas_id, T, state, cut, is_trap, tau) of one atlas graph, all 128 states x T_grid.
    Raises for a graph that is not graph-train unless allow_test=True (checked before any computation)."""
    if split[atlas_id] != "train" and not allow_test:
        raise ValueError(f"atlas graph {atlas_id} is in graph-{split[atlas_id]}; Unit 2a is graph-train only")
    A = graphs[atlas_id]
    cut, dcut, _, _ = L.basics(A)
    is_trap = np.isin(L.STATES, L.traps(dcut))
    names = [L.state_string(s) for s in L.STATES]
    return pd.concat([pd.DataFrame({"atlas_id": atlas_id, "T": T, "state": names, "cut": cut,
                                    "is_trap": is_trap, "tau": tau(A, T)}) for T in T_grid], ignore_index=True)


def build_tau_train():
    """tau for every graph-train graph (597); returns (long table, seconds)."""
    t0 = time.time()
    graphs = L.atlas_graphs()
    split = L.load_graph_split()
    train = [a for a in graphs if split[a] == "train"]
    out = pd.concat([graph_tau(a, split, graphs) for a in train], ignore_index=True)
    return out, time.time() - t0


def trap_tau(tau_long):
    """One row per trap of u1_traps.csv (graph-train only) x T: atlas_id, state, B, W, T, tau."""
    traps = pd.read_csv(config.U1_TRAPS)
    traps = traps[traps["split"] == "train"]
    out = traps[["atlas_id", "state", "B", "W"]].merge(tau_long[["atlas_id", "state", "T", "tau"]],
                                                         on=["atlas_id", "state"], how="left", validate="1:m")
    assert len(out) == len(traps) * len(T_GRID) and out["tau"].notna().all()
    return out.sort_values(["atlas_id", "state", "T"], ignore_index=True)


def slope_check(trap_rows, tol=0.1):
    """Per trap: slope = [ln tau(0.1) - ln tau(0.15)] / (1/0.1 - 1/0.15), match = |slope - B| < tol,
    with falls_deeper from u1_traps.csv."""
    w = trap_rows.pivot_table(index=["atlas_id", "state", "B"], columns="T", values="tau").reset_index()
    w["slope"] = (np.log(w[0.1]) - np.log(w[0.15])) / (1 / 0.1 - 1 / 0.15)
    w["match"] = (w["slope"] - w["B"]).abs() < tol
    u1 = pd.read_csv(config.U1_TRAPS)[["atlas_id", "state", "falls_deeper", "cut", "P_size"]]
    w = w.merge(u1, on=["atlas_id", "state"], how="left", validate="1:1")
    return w[["atlas_id", "state", "B", "cut", "P_size", "falls_deeper", 0.1, 0.15, "slope", "match"]]


def summary(tau_long, trap_rows):
    """Counts (graphs, states, traps; with the splits they come from), the slope check, its failures and
    the max tau over traps per T."""
    split = L.load_graph_split()
    sl = slope_check(trap_rows)
    return {"n_graphs": tau_long["atlas_id"].nunique(),
            "n_states": len(tau_long[["atlas_id", "state"]].drop_duplicates()),
            "n_traps": len(sl),
            "splits": sorted({split[a] for a in tau_long["atlas_id"].unique()}),
            "n_match": int(sl["match"].sum()),
            "failures": sl[~sl["match"]],
            "max_tau": trap_rows.groupby("T")["tau"].max()}


def save_tables():
    """Builds and writes config.U2_TAU_TRAIN and config.U2_TRAP_TAU; returns (tau_long, trap_rows, seconds)."""
    tau_long, secs = build_tau_train()
    rows = trap_tau(tau_long)
    config.U2_DIR.mkdir(parents=True, exist_ok=True)
    tau_long.to_parquet(config.U2_TAU_TRAIN, index=False)
    rows.to_csv(config.U2_TRAP_TAU, index=False)
    return tau_long, rows, secs

"""Unit 2a: exact expected escape times under the logit rule (graph-train graphs only).

States and their order are those of landscape.py (state s = 0..127, bit i set = node i is +1).
One step: pick a node i uniformly (prob 1/7); it flips with prob p_i = 1 / (1 + exp(dcut_i / T)),
dcut_i = F_i - E_i = change in cut if i flips; otherwise nothing happens (the step still counts).
The two consensus states are absorbing. Q = P on the 126 non-consensus states;
tau = (I - Q)^-1 1 = expected number of steps to consensus (0 at consensus).

Unit 2b (mu = uniform over all 128 states; traps and plateau_id from u1_traps.csv):
    N = (I - Q)^-1; Yc = sum_s mu(s) tau(s); visits v = mu_tr N; Yb = sum over traps of v
    entries = mu(traps) + sum over non-trap, non-consensus x of v_x P(x -> traps); duration = Yb / entries
    first hit (traps and consensus absorbing): P_hit = P(absorbed in a trap), w_c = P(absorbed in c) / P_hit,
        Ya = sum_c w_c tau_c (a start in a trap is absorbed there at once)
    committor of a plateau P: q = P(consensus before P), q = 1 on consensus, 0 on P;
        qbar = sum over x in P, y not in P of P(x, y) q(y) / sum of P(x, y); split by exit down / up;
        attempts = 1 / qbar
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


def check_train(atlas_id, split, allow_test=False, reason=None):
    """Raises for a graph that is not graph-train unless allow_test=True and a reason string is given."""
    if split[atlas_id] != "train":
        if not allow_test:
            raise ValueError(f"atlas graph {atlas_id} is in graph-{split[atlas_id]}; graph-train only")
        if not reason:
            raise ValueError("allow_test=True needs a reason")


def graph_tau(atlas_id, split, graphs, T_grid=T_GRID, allow_test=False):
    """Long table (atlas_id, T, state, cut, is_trap, tau) of one atlas graph, all 128 states x T_grid.
    Raises for a graph that is not graph-train unless allow_test=True (checked before any computation)."""
    check_train(atlas_id, split, allow_test)
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


# ---------------------------------------------------------------- Unit 2b: trap time, first hit, committor

MU = 1 / len(L.STATES)                                       # uniform start over all 128 states


def state_code(string):
    """State (int) from a string of + / - (as in u1_traps.csv)."""
    return L.state_from_signs([1 if ch == "+" else -1 for ch in string])


def trap_time(A, T, trap_states, tol=1e-8):
    """Yc, Yb, entries, duration, P_hit, Ya of the graph A at temperature T, from mu (see the docstring)."""
    P = transition_matrix(A, T)
    M = np.eye(len(TRANSIENT)) - P[np.ix_(TRANSIENT, TRANSIENT)]
    Nm = np.linalg.solve(M, np.eye(len(TRANSIENT)))
    resid = np.linalg.norm(M @ Nm - np.eye(len(TRANSIENT))) / np.linalg.norm(np.eye(len(TRANSIENT)))
    if resid >= tol:
        raise RuntimeError(f"relative residual {resid:.2e} >= {tol}")
    tau_all = np.zeros(len(L.STATES))
    tau_all[TRANSIENT] = Nm.sum(axis=1)
    v = MU * Nm.sum(axis=0)                                  # mu_tr N
    is_trap = np.isin(L.STATES, trap_states)
    trap_tr = is_trap[TRANSIENT]
    other = TRANSIENT[~trap_tr]                              # non-trap, non-consensus states
    Yc = MU * tau_all.sum()
    Yb = v[trap_tr].sum()
    entries = MU * is_trap.sum() + (v[~trap_tr] * P[other][:, is_trap].sum(axis=1)).sum()
    out = {"Yc": Yc, "Yb": Yb, "entries": entries, "duration": Yb / entries if entries > 0 else np.nan,
           "P_hit": 0.0, "Ya": np.nan}
    if len(trap_states):
        traps = np.array(sorted(trap_states))
        H = np.linalg.solve(np.eye(len(other)) - P[np.ix_(other, other)], P[np.ix_(other, traps)])
        hit = MU * H.sum(axis=0) + MU                        # absorbed in c: from a non-trap start, or start at c
        out["P_hit"] = hit.sum()
        out["Ya"] = (hit / hit.sum() * tau_all[traps]).sum()
    return out


def committor(A, T, plateau_states):
    """qbar, flux_share_down, q_down, flux_share_up, q_up, attempts of one plateau (see the docstring)."""
    P = transition_matrix(A, T)
    cut = L.basics(A)[0]
    in_P = np.isin(L.STATES, list(plateau_states))
    outside = L.STATES[~in_P & ~np.isin(L.STATES, ABSORBING)]
    q = np.zeros(len(L.STATES))
    q[ABSORBING] = 1.0
    q[outside] = np.linalg.solve(np.eye(len(outside)) - P[np.ix_(outside, outside)],
                                 P[np.ix_(outside, ABSORBING)].sum(axis=1))
    X, Y = np.array(sorted(plateau_states)), L.STATES[~in_P]
    F = P[np.ix_(X, Y)]                                      # flux out of the plateau, x equally weighted
    down = cut[Y][None, :] < cut[X][:, None]
    up = cut[Y][None, :] > cut[X][:, None]
    qbar = (F * q[Y]).sum() / F.sum()
    out = {"qbar": qbar, "attempts": 1 / qbar}
    for name, mask in [("down", down), ("up", up)]:
        f = F * mask
        out[f"flux_share_{name}"] = f.sum() / F.sum()
        out[f"q_{name}"] = (f * q[Y]).sum() / f.sum() if f.sum() > 0 else np.nan
    return out


def graph_outcomes(atlas_id, split, graphs, u1, T_grid, allow_test=False, reason=None):
    """trap_time rows (atlas_id, T, Yc, Yb, entries, duration, P_hit, Ya) of one atlas graph; traps from
    u1 = u1_traps.csv. Raises for a graph that is not graph-train unless allow_test=True with a reason."""
    check_train(atlas_id, split, allow_test, reason)
    A = graphs[atlas_id]
    codes = [state_code(x) for x in u1.loc[u1["atlas_id"] == atlas_id, "state"]]
    assert sorted(codes) == sorted(L.traps(L.basics(A)[1]))
    return [{"atlas_id": atlas_id, "T": T, **trap_time(A, T, codes)} for T in T_grid]


def graph_trap_time(atlas_id, split, graphs, u1, T_grid=T_GRID, allow_test=False):
    """(graph rows, plateau rows) of one atlas graph over T_grid; u1 = u1_traps.csv.
    Raises for a graph that is not graph-train unless allow_test=True (checked before any computation)."""
    check_train(atlas_id, split, allow_test)
    A = graphs[atlas_id]
    dcut = L.basics(A)[1]
    tr = u1[u1["atlas_id"] == atlas_id]
    codes = [state_code(x) for x in tr["state"]]
    assert sorted(codes) == sorted(L.traps(dcut))
    g_rows = [{"atlas_id": atlas_id, "T": T, **trap_time(A, T, codes)} for T in T_grid]
    p_rows = []
    for pid, g in tr.groupby("plateau_id"):
        P_states = L.plateau(state_code(g["state"].iloc[0]), dcut)
        assert len(P_states) == g["P_size"].iloc[0] and g["B"].nunique() == 1 and len(g) == g["traps_in_P"].iloc[0]
        for T in T_grid:
            p_rows.append({"atlas_id": atlas_id, "plateau_id": pid, "T": T, "P_size": len(P_states),
                           "n_traps_in_P": len(g), "B": g["B"].iloc[0], **committor(A, T, P_states)})
    return g_rows, p_rows


PLATEAU_COLS = ["atlas_id", "plateau_id", "T", "P_size", "n_traps_in_P", "B", "qbar", "flux_share_down", "q_down",
                "flux_share_up", "q_up", "attempts"]


def build_trap_time_train():
    """Unit 2b tables for every graph-train graph (597); returns (graph table, plateau table, seconds)."""
    t0 = time.time()
    graphs = L.atlas_graphs()
    split = L.load_graph_split()
    u1 = pd.read_csv(config.U1_TRAPS)
    g_rows, p_rows = [], []
    for a in graphs:
        if split[a] == "train":
            g, p = graph_trap_time(a, split, graphs, u1)
            g_rows += g
            p_rows += p
    return pd.DataFrame(g_rows), pd.DataFrame(p_rows)[PLATEAU_COLS], time.time() - t0


def monte_carlo_hit(A, T, trap_states, n_runs, seed=config.SEED):
    """Fraction of n_runs chains started from mu that hit a trap before consensus (a trap start counts)."""
    rng = np.random.default_rng(seed)
    cum = np.cumsum(transition_matrix(A, T), axis=1)
    traps = set(trap_states)
    hits = np.zeros(n_runs, dtype=bool)
    for r in range(n_runs):
        s = int(rng.integers(len(L.STATES)))
        while s not in ABSORBING and s not in traps:
            s = min(int(np.searchsorted(cum[s], rng.random(), side="right")), len(L.STATES) - 1)
        hits[r] = s in traps
    return hits


def crossing_T(T, y1, y2):
    """T values where log y1 = log y2, by linear interpolation of log y1 - log y2 between grid points."""
    d = np.log(np.asarray(y1)) - np.log(np.asarray(y2))
    out = []
    for i in range(len(T) - 1):
        if d[i] == 0:
            out.append(T[i])
        elif d[i] * d[i + 1] < 0:
            out.append(T[i] + (T[i + 1] - T[i]) * d[i] / (d[i] - d[i + 1]))
    return out


def summary_2b(G, Pl):
    """Human networks (Hierarchy, Ring, CT, Path) by T, T* where Yb(Hierarchy) = Yb(Ring), qbar by B and T."""
    names = pd.read_csv(config.U1_GRAPHS)[["atlas_id", "human_name"]].dropna()
    humans = G.merge(names[names["human_name"].isin(["Hierarchy", "Ring", "CT", "Path"])], on="atlas_id")
    yb = humans.pivot_table(index="T", columns="human_name", values="Yb")
    q = Pl.groupby(["B", "T"])["qbar"]
    qbar = pd.DataFrame({"n_plateaus": q.size(), "median": q.median(), "q25": q.quantile(0.25),
                         "q75": q.quantile(0.75)})
    qbar["IQR"] = qbar["q75"] - qbar["q25"]
    return {"humans": humans.set_index(["human_name", "T"])[["Yb", "entries", "duration", "Ya", "P_hit"]],
            "T_star": crossing_T(list(yb.index), yb["Hierarchy"], yb["Ring"]),
            "qbar_by_B": qbar}


def save_tables_2b():
    """Builds and writes config.U2B_GRAPH and config.U2B_PLATEAU; returns (graph table, plateau table, seconds)."""
    G, Pl, secs = build_trap_time_train()
    G.to_csv(config.U2B_GRAPH, index=False)
    Pl.to_csv(config.U2B_PLATEAU, index=False)
    return G, Pl, secs

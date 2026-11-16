"""Unit 1: the static trap landscape of every connected 7-node graph (networkx graph_atlas_g, 853 graphs).

No human data and no dynamics: only the 128 states s in {+1, -1}^7 of a graph and single-node flips.

A state is an integer 0..127: bit i set = node i is +1. A move flips one node i.
    cut(s)       = # edges whose ends have opposite signs
    F_i, E_i     = # neighbours of i on the same / opposite side; dcut of flipping i = F_i - E_i
    consensus    = all +1 or all -1; trap = non-consensus state with F_i >= E_i for every i
    k(s)         = min(#+, #-); a progress move lowers k
    B            = min over paths from the trap to a consensus of (max cut along the path) - cut(trap)
    a_c          = min over paths to a consensus of the max, over the path's moves, of the mover's agreement
                   before its move, A_i(s) = |1 + F_i - E_i| / (1 + d_i)
    P            = bottom plateau: states reachable from the trap by flat moves (dcut = 0)
    flat exit    = state of P with a downhill move; traps_in_P = |P| - # flat exits
    gate         = (x in P, i) with y = flip_i(x) not in P, cut(y) <= h* = h0 + B, and from y a state with
                   cut < h0 reachable through states with cut <= h* outside P (cut(y) < h0 counts directly)
    W            = |P| / # gates
    falls_deeper = a trap with cut < h0 is reachable from the trap through states with cut <= h*

Graph measures: lam2 (Laplacian), conductance (exact), modularity (exact, all 877 set partitions),
lamP (largest |eigenvalue| of D~^-1 (A + I) other than one eigenvalue 1).
"""
import heapq
import itertools
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from slider import config

N = 7
STATES = np.arange(2 ** N)
PLUS, MINUS = 2 ** N - 1, 0                           # the two consensus states
SIGN = np.where((STATES[:, None] >> np.arange(N)) & 1, 1, -1)    # SIGN[s, i] = +1 / -1

HUMAN_EDGES = {
    "Hierarchy": [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
    "CT": [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 4), (5, 6)],
    "Ring": [(i, (i + 1) % N) for i in range(N)],
    "Star": [(0, i) for i in range(1, N)],
    "Full": list(itertools.combinations(range(N), 2)),
    "Path": [(i, i + 1) for i in range(N - 1)],
}


def adjacency(edges):
    A = np.zeros((N, N), dtype=int)
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    return A


def state_from_signs(signs):
    """State (int) from a sequence of +1 / -1 (node order 0..6)."""
    return sum(1 << i for i, x in enumerate(signs) if x > 0)


def state_string(s):
    return "".join("+" if x > 0 else "-" for x in SIGN[s])


def flip(s, i):
    return s ^ (1 << i)


# ---------------------------------------------------------------- per-state quantities

def basics(A):
    """cut[s], dcut[s, i] = F_i - E_i, agr[s, i] = |1 + F_i - E_i| / (1 + d_i), k[s]."""
    same = SIGN * (SIGN @ A)                          # F_i - E_i
    cut = (A.sum() - same.sum(axis=1)) // 4            # (m - sum over edges of s_i s_j) / 2
    d = A.sum(axis=1)
    agr = np.abs(1 + same) / (1 + d)
    n_plus = (SIGN > 0).sum(axis=1)
    k = np.minimum(n_plus, N - n_plus)
    return cut, same, agr, k


def traps(dcut):
    """Trap states: non-consensus, no downhill move."""
    return [s for s in STATES if s not in (PLUS, MINUS) and np.all(dcut[s] >= 0)]


def bottleneck(start, weight):
    """min over paths from start to a consensus of the max move weight; weight(s, i) -> number."""
    best = {start: -np.inf}
    heap = [(-np.inf, start)]
    while heap:
        b, s = heapq.heappop(heap)
        if s in (PLUS, MINUS):
            return b
        if b > best[s]:
            continue
        for i in range(N):
            t = flip(s, i)
            nb = max(b, weight(s, i))
            if nb < best.get(t, np.inf):
                best[t] = nb
                heapq.heappush(heap, (nb, t))
    raise RuntimeError("no path to consensus")


def reach(starts, allowed):
    """States reachable from starts by single flips, staying in the set allowed (bool array over states)."""
    seen = set(s for s in starts if allowed[s])
    todo = list(seen)
    while todo:
        s = todo.pop()
        for i in range(N):
            t = flip(s, i)
            if allowed[t] and t not in seen:
                seen.add(t)
                todo.append(t)
    return seen


def plateau(c, dcut):
    """Bottom plateau: states reachable from c by flat moves only."""
    seen, todo = {c}, [c]
    while todo:
        s = todo.pop()
        for i in range(N):
            t = flip(s, i)
            if dcut[s, i] == 0 and t not in seen:
                seen.add(t)
                todo.append(t)
    return seen


def trap_landscape(A):
    """One dict per trap of the graph A (see the module docstring for the columns)."""
    cut, dcut, agr, k = basics(A)
    trap_list = traps(dcut)
    trap_set = set(trap_list)
    rows, plateau_of = [], {}
    for c in trap_list:
        h0 = int(cut[c])
        # barrier: node-weighted bottleneck, max cut of the states visited
        top = bottleneck(c, lambda s, i: max(cut[s], cut[flip(s, i)]))
        B = int(top) - h0
        h_star = h0 + B
        a_c = float(bottleneck(c, lambda s, i: agr[s, i]))

        P = plateau(c, dcut)
        if c not in plateau_of:
            pid = len(set(plateau_of.values()))
            for x in P:
                plateau_of[x] = pid
        exits = [x for x in P if np.any(dcut[x] < 0)]

        inside = np.zeros(len(STATES), dtype=bool)
        inside[list(P)] = True
        low_outside = (cut <= h_star) & ~inside
        good = reach([s for s in STATES if low_outside[s] and cut[s] < h0], low_outside)
        n_gates = sum(1 for x in P for i in range(N) if flip(x, i) in good)

        flat_c = [i for i in range(N) if dcut[c, i] == 0]
        prog_c = sum(k[flip(c, i)] < k[c] for i in flat_c)
        flat_P = [(x, i) for x in P for i in range(N) if dcut[x, i] == 0]
        prog_P = sum(k[flip(x, i)] < k[x] for x, i in flat_P)

        below = reach([c], cut <= h_star)
        falls = any(d in below and cut[d] < h0 for d in trap_set)

        rows.append({"state": state_string(c), "code": c, "cut": h0, "k": int(k[c]), "B": B, "a_c": a_c,
                     "P_size": len(P), "traps_in_P": len(P) - len(exits), "n_flat_exits": len(exits),
                     "n_gates": n_gates, "W": len(P) / n_gates,
                     "progress_share_c": prog_c / len(flat_c) if flat_c else np.nan,
                     "progress_share_P": prog_P / len(flat_P) if flat_P else np.nan,
                     "falls_deeper": bool(falls), "plateau_id": plateau_of[c]})
    # pair_id: c and its negation share one id
    pair = {}
    for r in rows:
        key = min(r["code"], PLUS ^ r["code"])
        pair.setdefault(key, len(pair))
        r["pair_id"] = pair[key]
    return rows


# ---------------------------------------------------------------- graph measures

def lam2(A):
    L = np.diag(A.sum(axis=1)) - A
    return float(np.sort(np.linalg.eigvalsh(L))[1])


def conductance(A):
    """min over nonempty S with vol(S) <= vol(V) / 2 of cut(S) / vol(S)."""
    d = A.sum(axis=1)
    best = np.inf
    for s in range(1, 2 ** N - 1):
        inS = SIGN[s] > 0
        vol = d[inS].sum()
        if vol <= d.sum() / 2:
            best = min(best, A[inS][:, ~inS].sum() / vol)
    return float(best)


def set_partitions(n=N):
    """All set partitions of range(n) as label lists (restricted growth strings); 877 for n = 7."""
    out = [[0]]
    for _ in range(1, n):
        out = [p + [c] for p in out for c in range(max(p) + 2)]
    return np.array(out)


PARTITIONS = set_partitions()


def modularity(A):
    """Exact maximum Newman modularity over all set partitions of the nodes."""
    d = A.sum(axis=1)
    m2 = d.sum()
    Bm = (A - np.outer(d, d) / m2) / m2
    same = PARTITIONS[:, :, None] == PARTITIONS[:, None, :]
    return float(max(0.0, np.einsum("pij,ij->p", same, Bm).max()))


def lamP(A):
    """Largest |eigenvalue| of D~^-1 (A + I), D~ = diag(d_i + 1), after removing one eigenvalue equal to 1."""
    d = A.sum(axis=1) + 1
    S = (A + np.eye(N)) / np.sqrt(np.outer(d, d))      # similar to P, symmetric
    ev = np.sort(np.linalg.eigvalsh(S))                 # the largest is 1
    return float(np.abs(ev[:-1]).max())


# ---------------------------------------------------------------- all graphs

def atlas_graphs():
    """{atlas index: adjacency} for the connected 7-node graphs of graph_atlas_g (atlas node order)."""
    out = {}
    for idx, g in enumerate(nx.graph_atlas_g()):
        if g.number_of_nodes() == N and nx.is_connected(g):
            out[idx] = nx.to_numpy_array(g, nodelist=range(N), dtype=int)
    return out


def human_atlas(graphs):
    """{atlas index: human name}, by isomorphism with HUMAN_EDGES."""
    out = {}
    for name, edges in HUMAN_EDGES.items():
        H = nx.Graph(edges)
        hits = [idx for idx, A in graphs.items() if nx.is_isomorphic(nx.from_numpy_array(A), H)]
        assert len(hits) == 1, name
        out[hits[0]] = name
    return out


def load_graph_split():
    """atlas index -> 'train' / 'test' from config.GRAPH_SPLIT (read only)."""
    t = pd.read_csv(config.GRAPH_SPLIT)
    return dict(zip(t["atlas"], t["split"]))


def build_tables():
    """(traps, graphs) DataFrames for all 853 graphs; also returns the runtime in seconds."""
    t0 = time.time()
    graphs = atlas_graphs()
    assert len(graphs) == 853
    split = load_graph_split()
    assert set(split) == set(graphs)
    human = human_atlas(graphs)
    trap_rows, graph_rows = [], []
    for idx, A in graphs.items():
        info = {"atlas_id": idx, "split": split[idx], "human_name": human.get(idx, "")}
        rows = trap_landscape(A)
        for r in rows:
            trap_rows.append({**info, **r})
        T = pd.DataFrame(rows)
        agg = lambda col, f: getattr(T[col], f)() if len(T) else np.nan
        graph_rows.append({
            **info, "n_edges": int(A.sum() // 2),
            "degree_seq": ",".join(map(str, sorted(A.sum(axis=1), reverse=True))),
            "n_traps": len(T), "n_plateaus": T["plateau_id"].nunique() if len(T) else 0,
            "B_max": agg("B", "max"), "B_mean": agg("B", "mean"), "W_max": agg("W", "max"),
            "W_mean": agg("W", "mean"), "P_max": agg("P_size", "max"), "n_gates_min": agg("n_gates", "min"),
            "ac_max": agg("a_c", "max"), "ac_mean": agg("a_c", "mean"),
            "any_falls_deeper": bool(T["falls_deeper"].any()) if len(T) else np.nan,
            "lam2": lam2(A), "conductance": conductance(A), "modularity": modularity(A), "lamP": lamP(A)})
    cols = ["atlas_id", "split", "human_name", "state", "pair_id", "plateau_id", "cut", "k", "B", "a_c", "P_size",
            "traps_in_P", "n_flat_exits", "n_gates", "W", "progress_share_c", "progress_share_P", "falls_deeper"]
    return pd.DataFrame(trap_rows)[cols], pd.DataFrame(graph_rows), time.time() - t0


# ---------------------------------------------------------------- train / test balance

TRAP_BINS = ["0", "2", "4-6", "8-14", ">=16"]
BMAX_BINS = ["none", "0", "1", ">=2"]


def trap_bin(n):
    assert n % 2 == 0                                   # traps come in mirror pairs
    return "0" if n == 0 else "2" if n == 2 else "4-6" if n <= 6 else "8-14" if n <= 14 else ">=16"


def bmax_bin(b):
    return "none" if pd.isna(b) else str(int(b)) if b < 2 else ">=2"


def balance(G, max_pp=10, alpha=0.05):
    """Train vs test counts and shares per bin; chi-square p per variable (empty bins dropped);
    FLAG if a bin's share differs by > max_pp percentage points or p < alpha."""
    rows = []
    for var, binner, bins in [("n_traps", trap_bin, TRAP_BINS), ("B_max", bmax_bin, BMAX_BINS)]:
        b = G[var].map(binner)
        tab = pd.crosstab(b, G["split"]).reindex(bins, fill_value=0)[["train", "test"]]
        share = 100 * tab / tab.sum()
        diff = share["train"] - share["test"]
        p = chi2_contingency(tab[tab.sum(axis=1) > 0].to_numpy())[1]
        flag = bool((diff.abs() > max_pp).any() or p < alpha)
        for bin_ in bins:
            rows.append({"variable": var, "bin": bin_, "n_train": int(tab.loc[bin_, "train"]),
                         "n_test": int(tab.loc[bin_, "test"]), "share_train_pct": share.loc[bin_, "train"],
                         "share_test_pct": share.loc[bin_, "test"], "diff_pp": diff[bin_], "chi2_p": p,
                         "max_abs_diff_pp": diff.abs().max(), "flag": "FLAG" if flag else "ok"})
    return pd.DataFrame(rows)


def summary(T, G):
    """Counts for the report: B over traps, B_max over graphs (NaN = no trap), graphs with a walled trap
    (B >= 1), traps with falls_deeper, and the human networks' graph rows."""
    return {"n_traps": len(T),
            "B_hist": T["B"].value_counts().sort_index(),
            "B_max_hist": G["B_max"].value_counts(dropna=False).sort_index(),
            "n_graphs_walled": int((G["B_max"] >= 1).sum()),
            "n_falls_deeper": int(T["falls_deeper"].sum()),
            "humans": G[G["human_name"] != ""]}


def save_tables():
    """Builds and writes config.U1_TRAPS, U1_GRAPHS, U1_BALANCE; returns (traps, graphs, balance, seconds)."""
    T, G, secs = build_tables()
    bal = balance(G)
    config.TABLES.mkdir(parents=True, exist_ok=True)
    T.to_csv(config.U1_TRAPS, index=False)
    G.to_csv(config.U1_GRAPHS, index=False)
    bal.to_csv(config.U1_BALANCE, index=False)
    return T, G, bal, secs

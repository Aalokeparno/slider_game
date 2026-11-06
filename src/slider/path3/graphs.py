"""Path 3 graphs: the 853 connected 7-node graphs (networkx graph_atlas_g), their fraught-corner count and |lambda|,
the graph-train / graph-test split and the screening set.

Split (made once, never overwritten): 70 / 30 within each (fraught-count tercile x |lambda| tercile) cell,
seed 573; the 5 human networks are forced into graph-train. Graph-test is never used.
Terciles: cut at the 1/3 and 2/3 quantiles of the values; a value equal to a cut goes to the lower bin
(ties stay together: 330 of 853 graphs have no fraught corner, so the fraught bins are 0 / 2 / >= 4).
Screening set: the 5 human networks + 2 random graph-train graphs (not human) per cell, seed 573.
"""
import json

import networkx as nx
import numpy as np
import pandas as pd

from slider import config
from slider.path3 import game

SCREENING = config.PATH3 / "screening_set.json"


def adjacency(g):
    return nx.to_numpy_array(g, nodelist=sorted(g.nodes()))


def atlas_table():
    """One row per connected 7-node atlas graph: graph (e.g. "G1012"), atlas index, edges (0-based list),
    n_edges, n_fraught (fraught non-consensus corners), lam (|lambda|), human (name or NaN)."""
    rows = []
    for idx, g in enumerate(nx.graph_atlas_g()):
        if g.number_of_nodes() != 7 or not nx.is_connected(g):
            continue
        A = adjacency(g)
        human = [name for name, H in game.HUMAN_GRAPHS.items() if nx.is_isomorphic(g, nx.from_numpy_array(H))]
        rows.append({"graph": f"G{idx}", "atlas": idx, "edges": json.dumps(sorted(map(list, g.edges()))),
                     "n_edges": g.number_of_edges(), "n_fraught": len(game.fraught_corners(A)),
                     "lam": game.second_eigenvalue(A), "human": human[0] if human else np.nan})
    return pd.DataFrame(rows)


def tercile(x):
    q1, q2 = x.quantile([1 / 3, 2 / 3])
    return (x > q1).astype(int) + (x > q2).astype(int)


def make_split(seed=config.SEED):
    """Writes config.GRAPH_SPLIT (atlas_table + f_terc, lam_terc, cell, split); refuses to overwrite."""
    if config.GRAPH_SPLIT.exists():
        raise FileExistsError(f"{config.GRAPH_SPLIT} exists; the graph split is made once")
    t = atlas_table()
    t["f_terc"], t["lam_terc"] = tercile(t["n_fraught"]), tercile(t["lam"])
    t["cell"] = 3 * t["f_terc"] + t["lam_terc"]
    t["split"] = "train"
    rng = np.random.default_rng(seed)
    for cell in sorted(t["cell"].unique()):
        rows = t.index[t["cell"] == cell].to_numpy()
        n_test = int(round(0.3 * len(rows)))
        free = rng.permutation(rows[t.loc[rows, "human"].isna().to_numpy()])
        t.loc[free[:n_test], "split"] = "test"
    t.to_csv(config.GRAPH_SPLIT, index=False)
    return t


def load_split():
    return pd.read_csv(config.GRAPH_SPLIT)


def make_screening_set(seed=config.SEED):
    """{"graph": edges (0-based)} for the 5 human networks (by name) and 2 random graph-train graphs per cell."""
    t = load_split()
    out = {}
    for name in game.HUMAN_GRAPHS:
        out[name] = json.loads(t.loc[t["human"] == name, "edges"].iloc[0])
    rng = np.random.default_rng(seed)
    for cell in range(9):
        pool = t[(t["cell"] == cell) & (t["split"] == "train") & t["human"].isna()]
        for i in rng.choice(len(pool), 2, replace=False):
            out[pool["graph"].iloc[i]] = json.loads(pool["edges"].iloc[i])
    config.PATH3.mkdir(parents=True, exist_ok=True)
    SCREENING.write_text(json.dumps(out, indent=1))
    return out


def load_screening_set():
    """{"graph": adjacency matrix}; the human networks keep the repo's player numbering."""
    edges = json.loads(SCREENING.read_text())
    out = {}
    for name, e in edges.items():
        if name in game.HUMAN_GRAPHS:
            out[name] = game.HUMAN_GRAPHS[name]
        else:
            A = np.zeros((7, 7))
            for i, j in e:
                A[i, j] = A[j, i] = 1
            out[name] = A
    return out

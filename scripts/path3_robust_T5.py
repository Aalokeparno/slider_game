"""Path 3 run T5: robustness of S0b'' vs S0b''' (soft gate) at eps 0.75 and 1 on every graph-train graph.

Graphs: the train split of data/splits/graph_split.csv (597 graphs; test graphs are dropped before any
adjacency is built). Start A, 200 runs per graph per variant. Seeds: run r on atlas graph a uses
default_rng([5731, a, r]) in all three variants (paired).
    S0b''         eps 0.5, H 4
    S0b''' 0.75   soft gate, eps 0.75, H 4
    S0b''' 1      soft gate, eps 1, H 4
Runs are saved in chunks of 50 graphs (results/path3/blocks/T5/chunk_<k>.csv, git-ignored; existing chunks are
skipped). Then robust.analyse writes robust_T5.csv, robust_T5_summary.csv, robust_T5_decision.csv,
robust_T5_spearman.csv, robust_T5_worst10.csv and figures/path3_robust_T5.png.

Usage: python scripts/path3_robust_T5.py [runs | analyse]   (default: both)
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from slider import config
from slider.path3 import graphs, human, robust, sim, strategies

OUT = config.PATH3 / "blocks" / "T5"
CHUNK = 50
SEED = 5731
VARIANTS = {"S0b''": {"eps": 0.5, "H": 4.0}, "S0b''' 0.75": {"eps": 0.75, "H": 4.0, "soft_gate": True},
            "S0b''' 1": {"eps": 1.0, "H": 4.0, "soft_gate": True}}
DELAYS = human.load_delays()


def train_graphs():
    """The train split only: graph, atlas index, adjacency, edges, n_fraught, lam, screening (bool)."""
    t = graphs.load_split()
    t = t[t["split"] == "train"].reset_index(drop=True)
    screen = set(json.loads((config.PATH3 / "screening_set.json").read_text()))
    t["screening"] = t["graph"].isin(screen) | t["human"].notna()
    return t


def adjacency(edges):
    A = np.zeros((7, 7))
    for i, j in json.loads(edges):
        A[i, j] = A[j, i] = 1
    return A


def one(job):
    graph, atlas, edges, variant, r = job
    out = sim.run(adjacency(edges), strategies.S0b, VARIANTS[variant], np.zeros(7, dtype=int), DELAYS,
                  np.random.default_rng([SEED, atlas, r]))
    return {"graph": graph, "variant": variant, "run": r, "t_consensus": out["t_consensus"],
            "success_180": out["success_180"], "success_300": out["success_300"], "n_yields": out["n_yields"],
            "breakups": out["breakups"]}


def runs():
    t = train_graphs()
    OUT.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(t), CHUNK):
        path = OUT / f"chunk_{k // CHUNK:02d}.csv"
        if path.exists():
            continue
        part = t.iloc[k:k + CHUNK]
        todo = [(g, a, e, v, r) for g, a, e in zip(part["graph"], part["atlas"], part["edges"])
                for v in VARIANTS for r in range(200)]
        with Pool(2) as pool:
            pd.DataFrame(pool.map(one, todo, chunksize=50)).to_csv(path, index=False)
        print(f"chunk {k // CHUNK} done ({min(k + CHUNK, len(t))} of {len(t)} graphs)", flush=True)


def analyse():
    t = train_graphs()
    r = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("chunk_*.csv"))], ignore_index=True)
    assert set(r["graph"]) == set(t["graph"]) and len(r) == len(t) * 3 * 200
    robust.analyse(r, t, list(VARIANTS))


if __name__ == "__main__":
    steps = sys.argv[1:] or ["runs", "analyse"]
    if "runs" in steps:
        runs()
    if "analyse" in steps:
        analyse()

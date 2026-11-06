"""Path 3 tuning run T1: Hierarchy and Full, start A (all sliders at 0), 200 runs per setting.

Blocks (results/path3/blocks/<block>.csv; a block that exists is skipped):
    T1ref  S0a and S0b'' at defaults: the screen_A reference runs, rerun to get the new per-run metrics
           (moving_per_tick, crossings); every run must equal its row in runs.csv
    T1a    S0a_hold, H in {2, 4, 8} s
    T1b    S0b'' (eps = 0.5, H = 4), t0 in {0, 5, 10, 20} s
    T1c    S0b'', eps in {0.1, 0.25, 0.5, 1} x H in {2, 4, 8} s
Seeds as in the screen: every setting on a graph reuses the 200 screen_A seeds of its base strategy on that graph
(S0a for S0a_hold, S0b for the S0b'' settings), so settings are compared on common random numbers.
All runs -> results/path3/runs_T1.csv.

Usage: python scripts/path3_tune_T1.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from slider import config
from slider.path3 import game, human, sim, strategies

OUT = config.PATH3 / "blocks"
GRAPHS = ["Hierarchy", "Full"]
BLOCKS = {
    "T1ref": [("S0a", {}), ("S0b", {"eps": 0.5, "H": 4.0})],
    "T1a": [("S0a_hold", {"H": h}) for h in [2.0, 4.0, 8.0]],
    "T1b": [("S0b", {"eps": 0.5, "H": 4.0, "t0": t0}) for t0 in [0.0, 5.0, 10.0, 20.0]],
    "T1c": [("S0b", {"eps": e, "H": h}) for e in [0.1, 0.25, 0.5, 1.0] for h in [2.0, 4.0, 8.0]],
}
BASE = {"S0a": "S0a", "S0a_hold": "S0a", "S0b": "S0b"}
DELAYS = human.load_delays()
SCREEN = pd.read_csv(config.PATH3 / "runs.csv").query("block == 'screen_A'")


def screen_seeds(strategy, graph):
    rows = SCREEN[(SCREEN["strategy"] == BASE[strategy]) & (SCREEN["graph"] == graph)]
    assert len(rows) == 200
    return list(rows["seed"])


def one(job):
    strategy, params, graph, seed = job
    rng = np.random.default_rng([int(x) for x in seed.split("-")])
    out = sim.run(game.HUMAN_GRAPHS[graph], strategies.STRATEGIES[strategy], params, np.zeros(7, dtype=int),
                  DELAYS, rng)
    out.update(graph=graph, start="A", seed=seed)
    return out


def run_block(block):
    path = OUT / f"{block}.csv"
    if path.exists():
        print(block, "exists, skipped")
        return pd.read_csv(path)
    jobs = [(s, p, g, seed) for s, p in BLOCKS[block] for g in GRAPHS for seed in screen_seeds(s, g)]
    print(block, len(jobs), "runs", flush=True)
    with Pool(2) as pool:
        df = pd.DataFrame(pool.map(one, jobs, chunksize=20))
    df.insert(0, "block", block)
    df.to_csv(path, index=False)
    return df


def check_reference(ref):
    """Every T1ref run equals its screen_A row (same seed): t_consensus, yields, collisions, outcome."""
    m = ref.merge(SCREEN, on=["strategy", "graph", "seed"], suffixes=("", "_screen"))
    assert len(m) == len(ref) == 800
    for c in ["t_consensus", "n_yields", "n_collisions", "outcome", "frac_time_fraught"]:
        a, b = m[c], m[c + "_screen"]
        if a.dtype == object:
            same = a == b
        else:                                   # floats read back from CSV can differ in the last bit
            same = np.isclose(a, b, rtol=0, atol=1e-12) | (a.isna() & b.isna())
        assert same.all(), c
    print("T1ref: all 800 runs equal their runs.csv rows")


if __name__ == "__main__":
    parts = {b: run_block(b) for b in BLOCKS}
    check_reference(parts["T1ref"])
    pd.concat(parts.values()).to_csv(config.PATH3 / "runs_T1.csv", index=False)

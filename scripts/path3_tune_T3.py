"""Path 3 tuning run T3: soft gate for S0b'' (S0b''' = soft_gate=True: yield chance eps * (1 - Agr)).

Blocks (results/path3/blocks/<block>.csv, git-ignored; a block that exists is skipped):
    T3a  start A, all 23 screening graphs, 200 runs each: S0b'' and S0b''' (eps 0.5, H 4)
    T3b  start B, the screen_B corners and runs of S0b'' (15 graphs with fraught corners): the same two
    T3c  start A, Hierarchy and Full, 200 runs each: S0b''' with eps in {0.5, 0.75, 1}, H 4
Seeds as in the screen: every run reuses the seed (and, for start B, the corner) of an S0b'' run in runs.csv
(screen_A / screen_B), so S0b'' vs S0b''' is paired. Every S0b'' rerun must equal its runs.csv row.
Then writes results/path3/tune_T3.csv (report.t3_table) and results/figures/path3_tune_T3.png (report.t3_figure).

Usage: python scripts/path3_tune_T3.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from slider import config
from slider.path3 import graphs, human, report, sim, strategies

OUT = config.PATH3 / "blocks"
DELAYS = human.load_delays()
GRAPHS = graphs.load_screening_set()
RUNS = pd.read_csv(config.PATH3 / "runs.csv").query("strategy == 'S0b'")
PLAIN = {"eps": 0.5, "H": 4.0}
SOFT = {"eps": 0.5, "H": 4.0, "soft_gate": True}


def corner_k(label):
    return np.array([10 if c == "+" else -10 for c in label])


def jobs(block):
    """(params, graph, start, corner, seed) per run."""
    out = []
    if block == "T3a":
        ref = RUNS[RUNS["block"] == "screen_A"]
        out = [(p, g, "A", "", s) for p in [PLAIN, SOFT] for g, s in zip(ref["graph"], ref["seed"])]
    elif block == "T3b":
        ref = RUNS[RUNS["block"] == "screen_B"]
        out = [(p, g, "B", c, s) for p in [PLAIN, SOFT] for g, c, s in zip(ref["graph"], ref["corner"], ref["seed"])]
    else:
        ref = RUNS[(RUNS["block"] == "screen_A") & RUNS["graph"].isin(["Hierarchy", "Full"])]
        out = [({**SOFT, "eps": e}, g, "A", "", s) for e in [0.5, 0.75, 1.0] for g, s in zip(ref["graph"], ref["seed"])]
    return out


def one(job):
    params, graph, start, corner, seed = job
    k0 = corner_k(corner) if start == "B" else np.zeros(7, dtype=int)
    rng = np.random.default_rng([int(x) for x in seed.split("-")])
    out = sim.run(GRAPHS[graph], strategies.S0b, params, k0, DELAYS, rng)
    out.update(graph=graph, start=start, corner=corner, seed=seed,
               variant="S0b'''" if params.get("soft_gate") else "S0b''")
    return out


def run_block(block):
    path = OUT / f"{block}.csv"
    if path.exists():
        print(block, "exists, skipped")
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    todo = jobs(block)
    print(block, len(todo), "runs", flush=True)
    with Pool(2) as pool:
        df = pd.DataFrame(pool.map(one, todo, chunksize=20))
    df.insert(0, "block", block)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def check_plain(df):
    """Every S0b'' rerun equals its runs.csv row (same seed, graph, start corner)."""
    plain = df[df["variant"] == "S0b''"]
    ref = RUNS.assign(start=RUNS["block"].str[-1])[["graph", "start", "seed", "t_consensus", "n_yields", "n_collisions",
                                                    "outcome"]]
    m = plain.merge(ref, on=["graph", "start", "seed"], suffixes=("", "_ref"))
    assert len(m) == len(plain), (len(m), len(plain))
    for c in ["t_consensus", "n_yields", "n_collisions", "outcome"]:
        a, b = m[c], m[c + "_ref"]
        same = (a == b) | (a.isna() & b.isna())
        if not same.all():
            raise SystemExit(f"S0b'' rerun differs from runs.csv in {c}: {int((~same).sum())} runs")
    print(f"{len(plain)} S0b'' reruns equal runs.csv")


if __name__ == "__main__":
    parts = [run_block(b) for b in ["T3a", "T3b", "T3c"]]
    check_plain(pd.concat(parts[:2]))
    table = report.t3_table(pd.concat(parts, ignore_index=True))
    table.to_csv(config.PATH3 / "tune_T3.csv", index=False)
    order = sorted(GRAPHS, key=lambda g: GRAPHS[g].sum())        # by number of edges (sparse to dense)
    report.t3_figure(table, order, config.FIGURES / "path3_tune_T3.png")

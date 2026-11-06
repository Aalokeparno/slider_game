"""Path 3 tuning run T4: S0b''' (soft gate, H 4 s) at eps 0.75 and 1 on every screening graph.

New blocks (results/path3/blocks/<block>.csv, git-ignored; a block that exists is skipped):
    T4a  start A, all 23 screening graphs, 200 runs each, eps in {0.75, 1}
    T4b  start B, the 15 graphs with fraught corners (the screen_B corners and runs), eps in {0.75, 1}
Seeds as in T3 (the screen's S0b'' seeds and corners), so every run pairs with the T3 runs of S0b'' and
S0b''' eps 0.5, which are reused from blocks/T3a.csv and T3b.csv (not rerun).
Writes results/path3/tune_T4.csv (one row per graph x start x variant, plus pooled rows per start, with the
graph's edge count), results/path3/tune_T4_choice.csv (choice tables), results/path3/tune_T4_spearman.csv
and results/figures/path3_tune_T4.png.

Usage: python scripts/path3_tune_T4.py
"""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from slider import config
from slider.path3 import graphs, human, report, sim, strategies

OUT = config.PATH3 / "blocks"
DELAYS = human.load_delays()
GRAPHS = graphs.load_screening_set()
RUNS = pd.read_csv(config.PATH3 / "runs.csv").query("strategy == 'S0b'")
EPS = [0.75, 1.0]


def jobs(block):
    ref = RUNS[RUNS["block"] == ("screen_A" if block == "T4a" else "screen_B")]
    start = "A" if block == "T4a" else "B"
    corners = ref["corner"].fillna("") if start == "B" else [""] * len(ref)
    return [({"eps": e, "H": 4.0, "soft_gate": True}, g, start, c, s)
            for e in EPS for g, c, s in zip(ref["graph"], corners, ref["seed"])]


def one(job):
    params, graph, start, corner, seed = job
    k0 = np.array([10 if c == "+" else -10 for c in corner]) if start == "B" else np.zeros(7, dtype=int)
    out = sim.run(GRAPHS[graph], strategies.S0b, params, k0, DELAYS,
                  np.random.default_rng([int(x) for x in seed.split("-")]))
    out.update(graph=graph, start=start, corner=corner, seed=seed, variant="S0b'''")
    return out


def run_block(block):
    path = OUT / f"{block}.csv"
    if not path.exists():
        todo = jobs(block)
        print(block, len(todo), "runs", flush=True)
        with Pool(2) as pool:
            df = pd.DataFrame(pool.map(one, todo, chunksize=20))
        df.insert(0, "block", block)
        df.to_csv(path, index=False)
    else:
        print(block, "exists, skipped")
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def label(variant, eps):
    return "S0b''" if variant == "S0b''" else f"S0b''' eps {eps:g}"


def main():
    t3 = pd.concat([pd.read_csv(OUT / f"{b}.csv", keep_default_na=False, na_values=[""]) for b in ["T3a", "T3b"]])
    runs = pd.concat([t3] + [run_block(b) for b in ["T4a", "T4b"]], ignore_index=True)
    # pairing check: every graph x start has the same seeds in every variant
    runs["eps_"] = [json.loads(p)["eps"] for p in runs["params"]]
    seeds = runs.groupby(["variant", "eps_", "graph", "start"])["seed"].apply(lambda s: tuple(sorted(s)))
    for (graph, start), s in seeds.groupby(level=["graph", "start"]):
        assert s.nunique() == 1 and len(s) == 4, (graph, start)
    print("paired: every graph x start has the same seeds in all 4 variants")
    table = report.t3_table(runs.drop(columns="eps_").assign(block="T4"))
    table.insert(1, "label", [label(v, e) for v, e in zip(table["variant"], table["eps"])])
    edges = {g: int(A.sum() / 2) for g, A in GRAPHS.items()}
    table.insert(table.columns.get_loc("graph") + 1, "edges", table["graph"].map(edges))
    table = table.drop(columns="block")
    table.to_csv(config.PATH3 / "tune_T4.csv", index=False)
    choice, spearman = report.t4_choice(table)
    choice.to_csv(config.PATH3 / "tune_T4_choice.csv", index=False)
    spearman.to_csv(config.PATH3 / "tune_T4_spearman.csv", index=False)
    report.t4_figure(table, config.FIGURES / "path3_tune_T4.png")


if __name__ == "__main__":
    main()

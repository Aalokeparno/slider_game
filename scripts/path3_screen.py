"""Path 3 screening runs -> results/path3/runs.csv (one row per run).

Blocks (each saved to results/path3/blocks/<block>.csv; a block that exists is skipped, so the script resumes):
    screen_A     6 strategies (defaults; v2 S2 / S1p and the v1 variants S2_v1 / S1p_v1) x screening set,
                 all sliders at 0, 200 runs each
    screen_B     the same 6 x screening set, fraught corners (all on Hierarchy, up to 20 random
                 per other graph, seed 573), 20 runs per corner
    sweep_A/B    Hierarchy, S0b / S2 / S1p, one parameter at a time around the defaults (incl. B and Q),
                 100 runs per setting
                 (B: the start corner drawn at random from Hierarchy's 4 fraught corners in each run)
    raw_A/B      the screen blocks again with Agreement read unrounded (sensitivity)
Seeds: run j of a block uses default_rng([573, block number, j]).

Usage: python scripts/path3_screen.py [block ...]   (default: all blocks)
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from slider import config
from slider.path3 import game, graphs, human, sim, strategies

BLOCKS = ["screen_A", "screen_B", "sweep_A", "sweep_B", "raw_A", "raw_B"]
OUT = config.PATH3 / "blocks"
DEFAULTS = {"S0a": {}, "S0b": {"eps": 0.5, "H": 4.0}, "S2": {"T": 1.3, "p": 0.1, "H": 4.0, "B": 3},
            "S1p": {"kappa": 1.05, "Q": 2}, "S2_v1": {"T": 1.3, "p": 0.1, "H": 4.0}, "S1p_v1": {"kappa": 1.05}}
SWEEPS = {"S0b": {"eps": [0.05, 0.2, 0.5, 1.0], "H": [0.0, 2.0, 4.0, 8.0]},
          "S2": {"T": [1.0, 2.0, 3.0, 5.0], "p": [0.05, 0.1, 0.2, 0.5, 1.0], "H": [0.0, 2.0, 4.0, 8.0],
                 "B": [2, 3, 5]},
          "S1p": {"kappa": [1.0, 1.02, 1.05, 1.08, 1.15], "Q": [1, 2, 3]}}


def corner_label(c):
    return "".join("+" if v > 0 else "-" for v in c)


def start_corners(name, A, rng):
    corners = game.fraught_corners(A)
    if name == "Hierarchy" or len(corners) <= 20:
        return corners
    return [corners[i] for i in sorted(rng.choice(len(corners), 20, replace=False))]


def tasks(block):
    """(strategy, params, graph, start, corner or None, display, runs) per setting."""
    out = []
    display = "raw" if block.startswith("raw") else "int"
    if block.startswith(("screen", "raw")):
        rng = np.random.default_rng(config.SEED)
        for name, A in GRAPHS.items():
            corners = start_corners(name, A, rng)
            for s in DEFAULTS:
                if block.endswith("A"):
                    out.append((s, DEFAULTS[s], name, "A", None, display, 200))
                else:
                    out += [(s, DEFAULTS[s], name, "B", c, display, 20) for c in corners]
    else:
        start = block[-1]
        for s, sweep in SWEEPS.items():
            for param, values in sweep.items():
                for v in values:
                    out.append((s, {**DEFAULTS[s], param: v}, "Hierarchy", start, None, "int", 100))
    return out


def one(job):
    block_no, j, (s, params, name, start, corner, display) = job
    rng = np.random.default_rng([config.SEED, block_no, j])
    A = GRAPHS[name]
    if start == "A":
        k0 = np.zeros(7, dtype=int)
    elif corner is None:                                        # sweep B: a random Hierarchy fraught corner
        fc = game.fraught_corners(A)
        corner = fc[rng.integers(len(fc))]
        k0 = corner * 10
    else:
        k0 = corner * 10
    out = sim.run(A, strategies.STRATEGIES[s], params, k0, DELAYS, rng, display=display)
    out.update(graph=name, start=start, corner=corner_label(corner) if corner is not None else "",
               display=display, seed=f"{config.SEED}-{block_no}-{j}")
    return out


DELAYS = human.load_delays()
GRAPHS = graphs.load_screening_set()


def run_block(block):
    path = OUT / f"{block}.csv"
    if path.exists():
        print(block, "exists, skipped")
        return
    jobs, j = [], 0
    for s, params, name, start, corner, display, n in tasks(block):
        for _ in range(n):
            jobs.append((BLOCKS.index(block), j, (s, params, name, start, corner, display)))
            j += 1
    print(block, len(jobs), "runs", flush=True)
    with Pool(2) as pool:
        rows = pool.map(one, jobs, chunksize=50)
    df = pd.DataFrame(rows)
    df.insert(0, "block", block)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(blocks):
    for block in blocks:
        run_block(block)
    parts = [pd.read_csv(OUT / f"{b}.csv") for b in BLOCKS if (OUT / f"{b}.csv").exists()]
    cols = ["block", "strategy", "params", "graph", "start", "corner", "display", "seed", "success_180", "success_300",
            "t_consensus", "frac_time_fraught", "n_yields", "n_collisions", "outcome", "period"]
    pd.concat(parts)[cols].to_csv(config.PATH3 / "runs.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or BLOCKS)

"""Path 3 run S3.2: S3a+ / S3b+ (one-seed wave), S4-old / S4-new (seniority flags), S6 (backoff), S7 (fast/slow)
against S0b''' (soft gate, eps 0.75, H 4). Start A everywhere.

    pilot    50 runs per default strategy on Hierarchy and Full: seconds per run and the projected time of stages 1, 2
    stage1   the 23 screening graphs. Defaults (S0b''' rerun + the 6 new): 200 runs per graph, the screen_A seeds of
             S0b in runs.csv (the T3 / T4 seeds); the S0b''' rerun must equal blocks/T4a.csv. One-factor sweeps
             (SWEEPS): 100 runs per graph, the first 100 of those seeds. Sweeps are descriptive only.
    stage2   the 597 graph-train graphs, T5 seeds default_rng([5731, atlas, r]): S3a+, S3b+, S6, S7 200 runs per graph,
             S4-old, S4-new 100 (r < 100). S0b''' = the T5 "S0b''' 0.75" runs (blocks/T5, not rerun).
Checkpoints: one file per (stage, setting, graph) in results/path3/blocks/S32/ (git-ignored); existing files are
skipped, so an interrupted stage resumes. `collect` appends the stage-1 rows to results/path3/runs.csv (blocks
S32_screen, S32_sweep) and writes the stage-2 rows to results/path3/runs_s3_2_confirm.parquet.

Usage: python scripts/path3_s3_2.py [pilot | stage1 | stage2 | collect] ...
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from slider import config
from slider.path3 import graphs, human, s32, sim, strategies

OUT = config.PATH3 / "blocks" / "S32"
DELAYS = human.load_delays()
S0B3 = {"eps": 0.75, "H": 4.0, "soft_gate": True}
# label -> (strategy class, params, noise)
DEFAULTS = {"S0b'''": ("S0b", S0B3, {}),
            "S3a+": ("S3", {"assigned_seed": True}, {}), "S3b+": ("S3", {"assigned_seed": False}, {}),
            "S4-old": ("S4", {"mode": "old"}, {}), "S4-new": ("S4", {"mode": "new"}, {}),
            "S6": ("S6", {}, {}), "S7": ("S7", {}, {})}
STAGE2 = {"S3a+": 200, "S3b+": 200, "S6": 200, "S7": 200, "S4-old": 100, "S4-new": 100}


def sweep_settings():
    """label -> (class, params, noise): one parameter changed from a default, non-default values only."""
    out = {}

    def add(base, key, values, noise=False):
        cls, params, nz = DEFAULTS[base]
        for v in values:
            p, n = dict(params), dict(nz)
            (n if noise else p)[key] = v
            out[f"{base} | {key}={v:g}" if not isinstance(v, str) else f"{base} | {key}={v}"] = (cls, p, n)

    for base in ["S3a+", "S3b+"]:
        add(base, "pause", [0.0, 1.0])
        add(base, "check", ["off"])
        add(base, "G", [0.0, 2.0, 8.0])
        add(base, "fallback", ["S2"])
        add(base, "fidget", [0.01, 0.03, 0.1, 0.3], noise=True)
        add(base, "slip", [0.02, 0.05], noise=True)
    add("S3b+", "lam", [0.03, 0.3])
    for base in ["S4-old", "S4-new"]:
        add(base, "A", [20.0, 30.0])
    add("S6", "w_min", [0.25, 1.0])
    add("S6", "W_max", [4.0, 16.0])
    add("S6", "rand", ["off"])
    add("S7", "q", [0.0, 1 / 7, 3 / 7, 1.0])
    add("S7", "N", [3, 10])
    for label, (cls, p, n) in out.items():                 # "off" switches
        for key in ["check", "rand"]:
            if p.get(key) == "off":
                p[key] = False
    return out


SWEEPS = sweep_settings()
SETTINGS = {**DEFAULTS, **SWEEPS}


def slug(label):
    return label.replace("'", "p").replace(" | ", "__").replace("=", "-").replace("+", "plus").replace("/", "_")


def screening_seeds():
    """graph -> the 200 screen_A seeds of S0b (in run order), as in T3 / T4."""
    runs = pd.read_csv(config.PATH3 / "runs.csv", usecols=["block", "strategy", "graph", "seed"])
    ref = runs[(runs["block"] == "screen_A") & (runs["strategy"] == "S0b")]
    return {g: list(d["seed"]) for g, d in ref.groupby("graph", sort=False)}


def train_graphs():
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


def one_graph(job):
    """All runs of one (stage, setting, graph); writes the checkpoint file."""
    stage, label, graph, A, seeds, path = job
    cls, params, noise = SETTINGS[label]
    t0 = time.time()
    rows = []
    for seed in seeds:
        out = sim.run(A, strategies.STRATEGIES[cls], params, np.zeros(7, dtype=int), DELAYS,
                      np.random.default_rng(seed), **noise)
        out.update(params=json.dumps({**params, **noise}, sort_keys=True), variant=label, graph=graph, start="A",
                   seed="-".join(map(str, seed)))
        rows.append(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return stage, label, graph, len(seeds), time.time() - t0


def stage1_jobs():
    screen = graphs.load_screening_set()
    seeds = screening_seeds()
    jobs = []
    for label in SETTINGS:
        n = 200 if label in DEFAULTS else 100
        for g, A in screen.items():
            s = [[int(x) for x in v.split("-")] for v in seeds[g][:n]]
            jobs.append(("stage1", label, g, A, s, OUT / "stage1" / slug(label) / f"{g}.csv"))
    return jobs


def stage2_jobs():
    t = train_graphs()
    jobs = []
    for label, n in STAGE2.items():
        for g, a, e in zip(t["graph"], t["atlas"], t["edges"]):
            jobs.append(("stage2", label, g, adjacency(e), [[5731, int(a), r] for r in range(n)],
                         OUT / "stage2" / slug(label) / f"{g}.csv"))
    return jobs


def run_jobs(jobs):
    todo = [j for j in jobs if not j[-1].exists()]
    total = sum(len(j[4]) for j in todo)
    print(f"{len(jobs) - len(todo)} of {len(jobs)} (setting, graph) files exist; {total} runs to do", flush=True)
    done, start = 0, time.time()
    with Pool(2) as pool:
        for stage, label, graph, n, sec in pool.imap_unordered(one_graph, todo):
            done += n
            rate = done / (time.time() - start)
            print(f"{stage} {label} {graph}: {n} runs, {sec / n:.3f} s/run | {done}/{total}, "
                  f"{(total - done) / rate / 60:.0f} min left", flush=True)


def pilot():
    screen = graphs.load_screening_set()
    per_run = {}
    for label in DEFAULTS:
        cls, params, noise = DEFAULTS[label]
        t0 = time.time()
        for g in ["Hierarchy", "Full"]:
            for r in range(50):
                sim.run(screen[g], strategies.STRATEGIES[cls], params, np.zeros(7, dtype=int), DELAYS,
                        np.random.default_rng([1, r]), **noise)
        per_run[label] = (time.time() - t0) / 100
        print(f"pilot {label}: {per_run[label]:.3f} s per run (Hierarchy + Full, 50 each)", flush=True)
    s1 = sum(per_run[l] * 23 * 200 for l in DEFAULTS)
    s1 += sum(per_run[l.split(" | ")[0]] * 23 * 100 for l in SWEEPS)
    s2 = sum(per_run[l] * 597 * n for l, n in STAGE2.items())
    print(f"projected on 2 cores: stage 1 {s1 / 2 / 3600:.1f} h, stage 2 {s2 / 2 / 3600:.1f} h", flush=True)


def check_rerun():
    """The stage-1 S0b''' rerun equals blocks/T4a.csv (eps 0.75) run by run."""
    new = pd.concat([pd.read_csv(p, keep_default_na=False, na_values=[""])
                     for p in (OUT / "stage1" / slug("S0b'''")).glob("*.csv")])
    old = pd.read_csv(config.PATH3 / "blocks" / "T4a.csv", keep_default_na=False, na_values=[""])
    old = old[old["params"] == json.dumps(S0B3, sort_keys=True)]
    m = new.merge(old, on=["graph", "seed"], suffixes=("", "_old"))
    assert len(m) == len(new) == len(old) == 4600, (len(m), len(new), len(old))
    for c in ["t_consensus", "n_yields", "n_collisions", "outcome", "breakups", "yields_undone", "yields_majority"]:
        same = (m[c] == m[c + "_old"]) | (m[c].isna() & m[c + "_old"].isna())
        assert same.all(), (c, int((~same).sum()))
    print("stage-1 S0b''' rerun: 4600 runs equal blocks/T4a.csv (eps 0.75)")


def collect():
    check_rerun()
    parts = []
    for p in (OUT / "stage1").glob("*/*.csv"):
        d = pd.read_csv(p, keep_default_na=False, na_values=[""])
        d["run"] = range(len(d))                         # rows are in seed order
        parts.append(d)
    s1 = pd.concat(parts, ignore_index=True)
    s1.insert(0, "block", np.where(s1["variant"].isin(list(DEFAULTS)), "S32_screen", "S32_sweep"))
    s1["display"], s1["corner"] = "int", ""
    runs = pd.read_csv(config.PATH3 / "runs.csv", keep_default_na=False, na_values=[""])
    runs = runs[~runs["block"].isin(["S32_screen", "S32_sweep"])]
    out = pd.concat([runs, s1[[c for c in runs.columns] + [c for c in s1.columns if c not in runs.columns]]],
                    ignore_index=True)
    out.to_csv(config.PATH3 / "runs.csv", index=False)
    print(f"runs.csv: {len(runs)} earlier rows + {len(s1)} stage-1 rows")
    s2 = pd.concat([pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in (OUT / "stage2").glob("*/*.csv")],
                   ignore_index=True)
    s2["run"] = [int(s.split("-")[2]) for s in s2["seed"]]
    s2.to_parquet(config.PATH3 / "runs_s3_2_confirm.parquet", index=False)
    print(f"runs_s3_2_confirm.parquet: {len(s2)} stage-2 rows")


def analyse():
    """Writes results/path3/s3_2_stage1.csv, s3_2_stage2_summary.csv, s3_2_stage2_per_graph.csv, s3_2_decision.csv,
    s3_2_outcomes.csv and the four figures results/figures/path3_s3_2_*.png."""
    runs = pd.read_csv(config.PATH3 / "runs.csv", keep_default_na=False, na_values=[""], low_memory=False)
    s1 = runs[runs["block"].isin(["S32_screen", "S32_sweep"])].reset_index(drop=True)
    for c in ["success_180", "success_300"]:
        s1[c] = s1[c].astype(str) == "True"
    table = s32.stage1_table(s1)
    table.to_csv(config.PATH3 / "s3_2_stage1.csv", index=False)
    s32.figure_fidget(table, config.FIGURES / "path3_s3_2_fidget.png")
    s32.figure_q(table, config.FIGURES / "path3_s3_2_S7_q.png")
    s32.figure_wmax(table, config.FIGURES / "path3_s3_2_S6_Wmax.png")
    s2 = pd.read_parquet(config.PATH3 / "runs_s3_2_confirm.parquet")
    t = train_graphs()
    summ, pg, worst, dense = s32.stage2(s2, t)
    summ.to_csv(config.PATH3 / "s3_2_stage2_summary.csv", index=False)
    pg.to_csv(config.PATH3 / "s3_2_stage2_per_graph.csv", index=False)
    s32.decide(summ).to_csv(config.PATH3 / "s3_2_decision.csv", index=False)
    out = pd.crosstab(s2["variant"], s2["outcome"], normalize="index")
    out.to_csv(config.PATH3 / "s3_2_outcomes.csv")
    s32.figure_edges(pg, config.FIGURES / "path3_s3_2_edges.png")
    print(f"worst-decile set {worst.sum()} graphs, dense subset {dense.sum()} graphs")


if __name__ == "__main__":
    for step in sys.argv[1:]:
        {"pilot": pilot, "stage1": lambda: run_jobs(stage1_jobs()), "stage2": lambda: run_jobs(stage2_jobs()),
         "collect": collect, "analyse": analyse}[step]()

"""Supervisor comparison (notebooks/path1/supervisor_compare.ipynb; descriptive, no model change): the simulations
behind it. Models: RW (Agreement-blind baseline: the APEM preset with a = 1.0, s = 0, mu = 0.1), APEM (published
a = s = 0.5, mu = 0.1), final, C1 and C2 (exploratory). Scope (results/model_b_log.md entry 4): v09 + cloud rounds,
old CL excluded, for every model. Each step is resumable (finished parts are skipped).

    checks    RW, APEM: the P24 check statistics of R replicates of the L3 design without its old-CL rounds (EL,
              CL v09, CC; targets 8-10 on the L5 part), with p24_main.py's job seeds ([seed, batch, first missing
              replicate]) and saved rows -> results/tables/sup_reps_<model>.parquet. Registered: RW R = 50 seed 5201,
              APEM R = 100 seed 5101. Batch size (memory): APEM 20 as p24_main.py; RW 2 (20,000-34,000 position
              events per round copy, vs. a few hundred for final or C2).
              C1 on the CL v09 rounds: scripts/p24_noOldCL.py --preset C1 --R 200 --seed 7101.
    traj      replicates 0-19 again with the stored streams: RW, APEM the batches of their checks run (RW 0-9 of 2,
              APEM 0 of 20) on the checks design; final, C1, C2 batch 0, [main seed, 0, 0] on the full L3 design (EL
              and CC rounds kept) and [noOldCL seed, 0, 0] on its CL v09 rounds. Their check statistics must equal the saved rows exactly, else the step stops.
              Saves what the figures need (collect) -> results/tables/sup_traj_<model>.pkl
    extended  20 replicates of the checks design with the time limit x 3 (900 s lab, 540 s cloud), registered seeds
              9001-9005 (simulated one round at a time, with the streams the round gets in the whole design); one row
              per round copy: success and time to consensus (the simulator's end time). sim.HORIZON is raised to
              1000 s for this step (it sizes the time grids only).
              -> results/tables/sup_ext_<model>.parquet

    python scripts/sup_runs.py checks --model RW --R 50 --seed 5201
    python scripts/sup_runs.py traj --model final
    python scripts/sup_runs.py extended --model C2
"""
import argparse
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from slider import config, design, sim, standoffs, stuck, targets
from slider import visuals as V

sys.path.insert(0, os.path.dirname(__file__))
import p24_noOldCL                                          # noqa: E402
from p24_main import SEEDS as MAIN_SEEDS, keep              # noqa: E402

warnings.filterwarnings("ignore", message="Pandas requires version")
MODELS = {"RW": ("APEM", {"a": 1.0, "s_th": 0.0, "mu": 0.1}), "APEM": ("APEM", V.APEM_PUBLISHED),
          "final": ("final", {}), "C1": ("C1", {}), "C2": ("C2", {})}
CHECK_SEEDS = {"RW": 5201, "APEM": 5101}
CHECK_BATCH = {"RW": 2, "APEM": design.BATCH}
EXT_SEEDS = {"RW": 9001, "APEM": 9002, "final": 9003, "C1": 9004, "C2": 9005}
N_TRAJ = 20
EXT_HORIZON = 1000.0      # s: sim.HORIZON (400) only sizes the time grids; ticks before 400 s are unchanged
CL_CELLS = [("Hierarchy", "CL"), ("Full", "CL")]
KEYS = ["replicate", "target", "stat", "topology", "dataset", "group"]


def reps_path(model):
    return config.TABLES / f"sup_reps_{model}.parquet"


def traj_path(model):
    return config.TABLES / f"sup_traj_{model}.pkl"


def ext_path(model):
    return config.TABLES / f"sup_ext_{model}.parquet"


def scope(d):
    """The rounds of a design in scope: old CL dropped."""
    return d[d["interface"] != "old_cl"].reset_index(drop=True)


# ------------------------------------------------------------------ what the figures need, per set of runs

def collect(runs3, table3, runs5, table5, ifaces5):
    """From one set of rounds (humans, or one replicate), in V.CELLS (Star left out):
    km, traj    as visuals.collect (traj: the CL cells)
    rev         Agreement at reversal (target 9) as counts: topology, dataset, A, n
    first       {cell: (run, info)}: the first qualifying round (visuals.qualifies) of each CL cell
    standoffs   standoffs.standoff_table of runs5 (Full)
    episodes    stuck.episode_table of runs5 (ifaces5: one interface label per run)
    rounds5     topology, dataset, interface, file of each round of runs5 (the per-round denominators)"""
    base = V.collect(runs3, table3, runs5, traj_cells=CL_CELLS)
    rev = base["rev"].groupby(["topology", "dataset", "A"]).size().rename("n").reset_index()
    t3 = table3.reset_index(drop=True)
    first = {}
    for k, run in enumerate(runs3):
        cell = (t3["topology"][k], t3["dataset"][k])
        if cell in CL_CELLS and cell not in first and V.qualifies(run, t3["success"][k], cell[0]):
            first[cell] = (run, {"file": run["file"], "wave": run["wave"], "success": bool(t3["success"][k])})
    t5 = table5.reset_index(drop=True).assign(interface=list(ifaces5))
    cells5 = V.in_cells(t5)
    runs5 = [r for r, c in zip(runs5, cells5) if c]
    t5 = t5[cells5].reset_index(drop=True)
    return {"km": base["km"], "traj": base["traj"], "rev": rev, "first": first,
            "standoffs": standoffs.standoff_table(runs5, t5),
            "episodes": stuck.episode_table(runs5, t5, t5["interface"]),
            "rounds5": t5[["topology", "dataset", "interface", "file"]]}


def merge(parts):
    """Pool collect() outputs: parts = [(replicate, part), ...] in replicate order (a replicate may have several
    parts, e.g. its EL / CC rounds and its CL v09 rounds)."""
    traj, first = {}, {}
    for r, p in parts:
        for cell, (x, a, n) in p["traj"].items():
            acc = traj.setdefault(cell, [0.0, 0.0, 0])
            acc[0], acc[1], acc[2] = acc[0] + x, acc[1] + a, acc[2] + n
        for cell, (run, info) in p["first"].items():
            first.setdefault(cell, (run, {**info, "replicate": r}))
    cat = lambda name: pd.concat([p[name].assign(replicate=r) for r, p in parts], ignore_index=True)
    rev = cat("rev").groupby(["topology", "dataset", "A"])["n"].sum().reset_index()
    return {"km": cat("km"), "traj": traj, "rev": rev, "first": first, "standoffs": cat("standoffs"),
            "episodes": cat("episodes"), "rounds5": cat("rounds5")}


# ------------------------------------------------------------------ steps

def run_checks(model, R, seed, procs):
    if seed != CHECK_SEEDS[model]:
        raise SystemExit(f"the registered seed of {model} is {CHECK_SEEDS[model]}, not {seed}")
    preset, kw = MODELS[model]
    out = reps_path(model)
    saved = pd.read_parquet(out) if out.exists() else None
    if saved is not None and set(saved["seed"]) != {seed}:
        raise SystemExit(f"{out} holds seed {sorted(set(saved['seed']))}, not {seed}")
    have = set() if saved is None else set(saved["replicate"])
    d = scope(design.design("L3"))
    jobs = []
    batch = CHECK_BATCH[model]
    for b, first in enumerate(range(0, R, batch)):
        missing = [r for r in range(first, min(first + batch, R)) if r not in have]
        if missing:
            assert missing == list(range(missing[0], missing[-1] + 1)), "saved replicates are a prefix of each batch"
            jobs.append((d, preset, len(missing), [seed, b, missing[0]], missing[0], kw))
    todo = sum(job[2] for job in jobs)
    print(f"{model}: {len(d)} rounds, R = {R}, {len(have & set(range(R)))} saved, {todo} to run in {len(jobs)} jobs, "
          f"{procs} processes -> {out}", flush=True)
    start, ran = time.time(), 0
    with ProcessPoolExecutor(procs, max_tasks_per_child=1) as pool:
        for future in as_completed([pool.submit(design.check_job, job) for job in jobs]):
            new = keep(future.result()).assign(preset=model, seed=seed)
            saved = new if saved is None else pd.concat([saved, new], ignore_index=True)
            tmp = out.with_suffix(".tmp")
            saved.to_parquet(tmp, index=False)
            os.replace(tmp, out)
            ran += new["replicate"].nunique()
            per_rep = (time.time() - start) / ran
            print(f"{time.strftime('%H:%M:%S')}  {ran}/{todo} done, {per_rep:.1f} s per replicate, "
                  f"about {per_rep * (todo - ran) / 60:.0f} min left", flush=True)
    print(f"{model}: {saved['replicate'].nunique() if saved is not None else 0} replicates in {out}")


def same_rows(a, b):
    """The same statistics with exactly the same values (NaN = NaN)."""
    a = a.sort_values(KEYS).reset_index(drop=True)
    b = b.sort_values(KEYS).reset_index(drop=True)
    if len(a) != len(b) or not (a[KEYS].astype(str).to_numpy() == b[KEYS].astype(str).to_numpy()).all():
        return False
    va, vb = a["value"].to_numpy(float), b["value"].to_numpy(float)
    return bool(np.all((va == vb) | (np.isnan(va) & np.isnan(vb))))


def part_path(model, segment, r):
    return config.TABLES / "sup_traj_parts" / f"{model}_{segment}_r{r:02d}.pkl"


def replay(model, segment, d, rows, batches):
    """Replay the stored batches [(seed, R, first replicate), ...] of design d, check every replicate's statistics
    against the saved rows (saved_rows) and save collect() of the rounds at positions `rows` of d per replicate
    (part_path). A batch whose replicates are all saved is skipped."""
    preset, kw = MODELS[model]
    saved = saved_rows(model, segment)
    roles, keep5 = list(d["role"]), ~d["whole_second"].to_numpy()
    i3 = sorted(rows)
    i5 = [i for i in i3 if keep5[i]]
    for seed, R, first in batches:
        if all(part_path(model, segment, first + r).exists() for r in range(R)):
            continue
        rounds = design.simulate_rounds(d, preset, R=R, seed=seed, **kw)
        for r in range(R):
            runs3 = design.copy_runs(d, rounds, r, seed)
            table3 = sim.table(runs3, roles)
            runs5 = [run for run, k in zip(runs3, keep5) if k]
            st = keep(targets.check_stats(runs3, table3, runs5)).assign(replicate=first + r)
            if not same_rows(st[KEYS + ["value"]], saved.loc[saved["replicate"] == first + r, KEYS + ["value"]]):
                raise SystemExit(f"STOP: {model} {segment} replicate {first + r} (seed {seed}) differs from the saved "
                                 "statistics")
            part = collect([runs3[i] for i in i3], table3.iloc[i3].reset_index(drop=True), [runs3[i] for i in i5],
                           table3.iloc[i5].reset_index(drop=True), d["interface"].to_numpy()[i5])
            path = part_path(model, segment, first + r)
            path.parent.mkdir(exist_ok=True)
            with open(path.with_suffix(".tmp"), "wb") as f:
                pickle.dump(part, f)
            os.replace(path.with_suffix(".tmp"), path)
            for o, _ in rounds:
                o[r] = None                                     # copy r is done
            print(f"{time.strftime('%H:%M:%S')}  {model} {segment} replicate {first + r}: equal to the saved "
                  "statistics; collected", flush=True)
        del rounds


def saved_rows(model, segment):
    """The saved statistics a replay is checked against."""
    if segment == "all":
        return pd.read_parquet(reps_path(model))
    return pd.read_parquet(config.P24_REPS[model] if segment == "main" else p24_noOldCL.path(model))


def run_traj(model):
    out = traj_path(model)
    if out.exists():
        print(f"{model}: {out} exists, nothing to do")
        return
    start = time.time()
    if model in CHECK_SEEDS:                                    # RW, APEM: the checks design, its batches
        d, B = scope(design.design("L3")), CHECK_BATCH[model]
        replay(model, "all", d, range(len(d)), [([CHECK_SEEDS[model], b, B * b], B, B * b) for b in range(N_TRAJ // B)])
        segments = ["all"]
    else:                                                       # final, C1, C2: the main run, then the noOldCL run
        d3 = design.design("L3")
        replay(model, "main", d3, np.flatnonzero((d3["dataset"] != "CL").to_numpy()),
               [([MAIN_SEEDS[model], 0, 0], N_TRAJ, 0)])
        dv = p24_noOldCL.v09_cl(d3)
        replay(model, "v09", dv, range(len(dv)), [([p24_noOldCL.SEEDS[model], 0, 0], N_TRAJ, 0)])
        segments = ["main", "v09"]
    parts = []
    for r in range(N_TRAJ):
        for segment in segments:
            with open(part_path(model, segment, r), "rb") as f:
                parts.append((r, pickle.load(f)))
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(merge(parts), f)
    os.replace(tmp, out)
    for r in range(N_TRAJ):
        for segment in segments:
            part_path(model, segment, r).unlink()
    print(f"{model}: {N_TRAJ} replicates -> {out} ({(time.time() - start) / 60:.1f} min)")


def run_extended(model, seed):
    if seed != EXT_SEEDS[model]:
        raise SystemExit(f"the registered seed of {model} (extended) is {EXT_SEEDS[model]}, not {seed}")
    out = ext_path(model)
    if out.exists():
        print(f"{model}: {out} exists, nothing to do")
        return
    preset, kw = MODELS[model]
    d = scope(design.design("L3"))
    d = d.assign(T=3 * d["T"])
    sim.HORIZON = EXT_HORIZON                                   # the time grids must cover 900 s
    start = time.time()
    rows = []
    for i in range(len(d)):                                     # one round at a time (memory), whole-design streams
        sub, rounds = V.simulate_rows(d, [i], preset, N_TRAJ, seed, **kw)
        row = sub.iloc[0]
        rows += [{"topology": row["topology"], "dataset": row["dataset"], "interface": row["interface"],
                  "group": row["group"], "wave": row["wave"], "T3": row["T"], "replicate": r, "success": o["success"],
                  "end_time": o["end_time"]} for r, o in enumerate(rounds[0][0])]
        if i % 25 == 0:
            print(f"{time.strftime('%H:%M:%S')}  {model}: round {i + 1}/{len(d)}", flush=True)
    tab = pd.DataFrame(rows).assign(model=model, seed=seed)
    tab.to_parquet(out, index=False)
    print(f"{model}: {len(tab)} round copies -> {out} ({(time.time() - start) / 60:.1f} min)")


def main():
    ap = argparse.ArgumentParser(description="simulations of the supervisor comparison (resumable)")
    ap.add_argument("step", choices=["checks", "traj", "extended"])
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--R", type=int, help="checks: number of replicates")
    ap.add_argument("--seed", type=int, help="checks, extended: the registered seed")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    if a.step == "checks":
        if a.model not in CHECK_SEEDS:
            raise SystemExit("checks: RW or APEM (final, C2: saved runs; C1: scripts/p24_noOldCL.py)")
        run_checks(a.model, a.R, a.seed, a.procs)
    elif a.step == "traj":
        run_traj(a.model)
    else:
        run_extended(a.model, a.seed)


if __name__ == "__main__":
    main()

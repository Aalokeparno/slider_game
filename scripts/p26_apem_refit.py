"""P26 (results/model_b_log.md entry 5): APEM refit by a two-stage search. Baseline refit only; the simulator is
unchanged (the sim.py APEM preset with a, s_th, mu set per run). Never loads Ring/CT.

Design: design.design("L3") without its old-CL rounds (entry 4), split into two parts simulated on their own:
lab (the lab_v09 rounds: Hierarchy EL, Hierarchy CL, Full CL; 27 check statistics) with mu_lab and cloud (the cc
rounds: Full, Hierarchy, Star CC; 26 statistics) with mu_cloud; a_th and s_th are shared.

Loss of a part: sum of z^2 over its check statistics (targets.check_stats, the fast path), z = (sim - human) /
sqrt(SE_h^2 + SE_mc^2), SE_h = (human CI width) / 3.92 (p24_checks_noOldCL.csv, final rows), SE_mc = SD over the
replicates with a value / sqrt(n) (0 if n = 1). A statistic with no value in any replicate adds z^2 = MISSING_Z2 = 100 (entry 5, amendment 1).

Common random numbers: batch b of part p (0 lab, 1 cloud) in stage k has job seed [SEEDS[k], p, b] for every
setting; batch sizes are fixed per stage (BATCH).

    stage 1  6 a_th x 4 s_th x 6 mu, both parts, R = 2; per (a_th, s_th) the best mu per part
    stage 2  around the 2 best (a_th, s_th): a_th +/- 0.05, s_th +/- 0.1 (inside the box), the center's best mu
             per part with its 2 neighbours in MU; R = 3
    stage 3  the 3 best (a_th, s_th, mu_lab, mu_cloud) of stage 2, R = 50; the lowest L wins
    stage 4  the winner, R = 200, seed 5304 -> p26_apem_refit_checks.csv, p26_apem_refit.json

Outputs (results/tables/): p26_apem_search.csv (one row per part run of stages 1-3), p26_reps_stage3.parquet,
p26_reps_stage4.parquet, p26_apem_refit_checks.csv, p26_apem_refit.json. Every stage is resumable (finished part
runs and batches are skipped) and refuses to start before the stage before it is complete.

    python scripts/p26_apem_refit.py --stage all --procs 2
    python scripts/p26_apem_refit.py --stage 1 --procs 2 --smoke DIR     (tiny design: 2 rounds per cell, outputs in DIR)
"""
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from slider import config, design, targets

sys.path.insert(0, os.path.dirname(__file__))
import p24_checks as pc                                     # noqa: E402
from p24_main import keep                                   # noqa: E402

warnings.filterwarnings("ignore", message="Pandas requires version")
SEEDS = {1: 5301, 2: 5302, 3: 5303, 4: 5304}                # registered, entry 5
R = {1: 2, 2: 3, 3: 50, 4: 200}
BATCH = {1: 2, 2: 3, 3: 10, 4: 10}
A1 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
S1 = [0.3, 0.5, 0.7, 0.9]
MU = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
A_STEP, S_STEP = 0.05, 0.1
A_BOX, S_BOX = (0.3, 0.8), (0.3, 0.9)
PARTS = {"lab": ("lab_v09", 0), "cloud": ("cc", 1)}
N_STATS = {"lab": 27, "cloud": 26}
MISSING_Z2 = 100.0          # z^2 of a statistic with no value in any replicate (amendment 1)
RUN = ["part", "a_th", "s_th", "mu"]


def files(out):
    return {"search": out / "p26_apem_search.csv", 3: out / "p26_reps_stage3.parquet",
            4: out / "p26_reps_stage4.parquet", "checks": out / "p26_apem_refit_checks.csv",
            "json": out / "p26_apem_refit.json"}


def part_of(dataset):
    return "cloud" if dataset == "CC" else "lab"


def human():
    """The human values of p24_checks_noOldCL.csv (final rows): the 53 + 10c and the other non-list rows."""
    t = pd.read_csv(config.TABLES / "p24_checks_noOldCL.csv")
    t = t[(t["preset"] == "final") & (t["stat"] != pc.SD_STAT)].rename(columns={"human": "value"})
    t = t[pc.KEY + ["value", "ci_lo", "ci_hi", "ci_rule", "in_check_list"]].reset_index(drop=True)
    t["ci_rule"] = t["ci_rule"].fillna("")
    t["se_h"] = (t["ci_hi"] - t["ci_lo"]) / 3.92
    t["part"] = t["dataset"].map(part_of)
    return t


def designs(smoke):
    """{part: design}: the L3 design without old CL, per part (smoke: the first 2 rounds of each cell)."""
    d = design.design("L3")
    assert set(d["topology"]) == {"Full", "Hierarchy", "Star"}             # no Ring, no CT
    d = d[d["interface"] != "old_cl"]
    if smoke:
        d = d.groupby(["topology", "dataset"], sort=False).head(2)
    return {p: d[d["interface"] == iface].reset_index(drop=True) for p, (iface, _) in PARTS.items()}


def loss(reps, hum):
    """Part loss from the check_stats rows of all replicates of one part run. hum: that part's check-list rows.
    Returns (L, statistics without a value, per-statistic table)."""
    s = reps.groupby(pc.KEY)["value"].agg(n="count", sim="mean", sd="std").reset_index()
    x = hum.merge(s, on=pc.KEY, how="left", validate="one_to_one")
    x["n"] = x["n"].fillna(0).astype(int)
    x["se_mc"] = np.where(x["n"] > 1, x["sd"] / np.sqrt(x["n"]), 0.0)
    x["z"] = (x["sim"] - x["value"]) / np.sqrt(x["se_h"] ** 2 + x["se_mc"] ** 2)
    missing = int((x["n"] == 0).sum())
    z2 = np.where(x["n"] == 0, MISSING_Z2, x["z"] ** 2)
    return float(z2.sum()), missing, x


def batch_job(job):
    """One batch: (design, part, a_th, s_th, mu, n, seed, first replicate) -> (keep rows of check_stats, seconds)."""
    d, part, a, s, mu, n, seed, first = job
    t0 = time.time()
    rows = design.check_job((d, "APEM", n, seed, first, {"a": a, "s_th": s, "mu": mu}))
    return keep(rows).assign(part=part, a_th=a, s_th=s, mu=mu), time.time() - t0


def run_parts(stage, runs, ds, hum, out, procs):
    """Run the part runs [(part, a_th, s_th, mu), ...] of a stage (resumable). Stages 1-3: one row per finished
    part run appended to the search table. Stages 3-4: every finished batch appended to the stage's reps file."""
    f = files(out)
    done = set()
    if f["search"].exists():
        t = pd.read_csv(f["search"])
        done = {tuple(r) for r in t.loc[t["stage"] == stage, RUN].itertuples(index=False)}
    saved = pd.read_parquet(f[stage]) if stage in f and f[stage].exists() else None
    have = {} if saved is None else {k: set(g["replicate"]) for k, g in saved.groupby(RUN)}
    nb = int(np.ceil(R[stage] / BATCH[stage]))
    jobs, wanted = [], {}
    for run in runs:
        part, a, s, mu = run
        if run in done:
            continue
        got = have.get(run, set())
        for b in range(nb):
            first = b * BATCH[stage]
            reps = list(range(first, min(first + BATCH[stage], R[stage])))
            if not set(reps) <= got:
                jobs.append((ds[part], part, a, s, mu, len(reps), [SEEDS[stage], PARTS[part][1], b], first))
        wanted[run] = []
    print(f"stage {stage}: {len(runs)} part runs, {len(runs) - len(wanted)} done, {len(jobs)} batches to run "
          f"(R = {R[stage]}, batch {BATCH[stage]}, seed {SEEDS[stage]}), {procs} processes", flush=True)
    if saved is not None:                                    # earlier batches of unfinished runs
        for k, g in saved.groupby(RUN):
            if k in wanted:
                wanted[k].append(g)
    start = time.time()
    with ProcessPoolExecutor(procs, max_tasks_per_child=1) as pool:
        futures = [pool.submit(batch_job, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            rows, sec = future.result()
            rows = rows.assign(seed=SEEDS[stage], batch=int(rows["replicate"].min()) // BATCH[stage], seconds=sec)
            run = tuple(rows[RUN].iloc[0])
            if stage in (3, 4):
                saved = rows if saved is None else pd.concat([saved, rows], ignore_index=True)
                tmp = f[stage].with_suffix(".tmp")
                saved.to_parquet(tmp, index=False)
                os.replace(tmp, f[stage])
            wanted[run].append(rows)
            reps = pd.concat(wanted[run], ignore_index=True)
            if stage < 4 and reps["replicate"].nunique() == R[stage]:
                h = hum[hum["in_check_list"] & (hum["part"] == run[0])]
                L, missing, _ = loss(reps, h)
                row = pd.DataFrame([{"stage": stage, "part": run[0], "a_th": run[1], "s_th": run[2], "mu": run[3],
                                     "R": R[stage], "seed": SEEDS[stage], "L": L, "n_stats": len(h),
                                     "n_missing": missing,
                                     "seconds": round(reps.groupby("batch")["seconds"].first().sum(), 1)}])
                row.to_csv(f["search"], mode="a", header=not f["search"].exists(), index=False)
            el = time.time() - start
            print(f"{time.strftime('%H:%M:%S')}  {i}/{len(jobs)} batches  {run[0]} a {run[1]} s {run[2]} mu {run[3]}"
                  f"  {sec:.0f} s  (about {el / i * (len(jobs) - i) / 60:.0f} min left)", flush=True)


def stage_rows(out, stage, runs):
    """The search rows of a finished stage; stops if a run is missing."""
    f = files(out)["search"]
    t = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["stage"] + RUN)
    t = t[t["stage"] == stage]
    missing = set(runs) - {tuple(r) for r in t[RUN].itertuples(index=False)}
    if missing:
        raise SystemExit(f"stage {stage} is not complete: {len(missing)} of {len(runs)} part runs missing")
    return t


def best_mu(t):
    """Per (a_th, s_th): the best mu per part (ties: the smaller mu, the first in MU) and L = L_lab + L_cloud,
    sorted by L (ties: grid order)."""
    t = t.sort_values(["a_th", "s_th", "part", "mu"])
    best = t.loc[t.groupby(["a_th", "s_th", "part"])["L"].idxmin()]
    w = best.pivot(index=["a_th", "s_th"], columns="part", values=["mu", "L"])
    w.columns = [f"{v}_{p}" for v, p in w.columns]
    w = w.reset_index()
    w["L"] = w["L_lab"] + w["L_cloud"]
    return w.sort_values("L", kind="stable").reset_index(drop=True)


def grid(x, step, box):
    v = np.round(np.array([x - step, x, x + step]) / step) * step
    return [round(float(y), 2) for y in v if box[0] - 1e-9 <= y <= box[1] + 1e-9]


def triple(mu):
    i = MU.index(mu)
    j = min(max(i - 1, 0), len(MU) - 3)
    return MU[j:j + 3]


def stage1_runs():
    return [(p, a, s, mu) for a in A1 for s in S1 for mu in MU for p in PARTS]


def stage2_runs(out):
    centers = best_mu(stage_rows(out, 1, stage1_runs())).head(2)
    mus = {}
    for c in centers.itertuples():
        for a in grid(c.a_th, A_STEP, A_BOX):
            for s in grid(c.s_th, S_STEP, S_BOX):
                for p in PARTS:
                    mus.setdefault((a, s, p), set()).update(triple(getattr(c, f"mu_{p}")))
    return [(p, a, s, mu) for (a, s, p), m in sorted(mus.items()) for mu in sorted(m)], centers


def stage3_settings(out):
    """The 3 best (a_th, s_th, mu_lab, mu_cloud) of stage 2 by L_lab + L_cloud (ties: grid order)."""
    runs, _ = stage2_runs(out)
    t = stage_rows(out, 2, runs)
    lab = t[t["part"] == "lab"][["a_th", "s_th", "mu", "L"]].rename(columns={"mu": "mu_lab", "L": "L_lab"})
    cloud = t[t["part"] == "cloud"][["a_th", "s_th", "mu", "L"]].rename(columns={"mu": "mu_cloud", "L": "L_cloud"})
    c = lab.merge(cloud, on=["a_th", "s_th"]).sort_values(["a_th", "s_th", "mu_lab", "mu_cloud"])
    c["L"] = c["L_lab"] + c["L_cloud"]
    return c.sort_values("L", kind="stable").head(3).reset_index(drop=True)


def runs_of(settings):
    out = []
    for c in settings.itertuples():
        for p in PARTS:
            run = (p, c.a_th, c.s_th, getattr(c, f"mu_{p}"))
            if run not in out:
                out.append(run)
    return out


def stage3_result(out):
    """The stage-3 candidates with their R = 50 losses, the winner first (ties: the stage-2 rank)."""
    cand = stage3_settings(out).rename(columns={"L_lab": "L2_lab", "L_cloud": "L2_cloud", "L": "L2"})
    t = stage_rows(out, 3, runs_of(cand)).set_index(RUN)["L"]
    cand["L_lab"] = [t[("lab", c.a_th, c.s_th, c.mu_lab)] for c in cand.itertuples()]
    cand["L_cloud"] = [t[("cloud", c.a_th, c.s_th, c.mu_cloud)] for c in cand.itertuples()]
    cand["L"] = cand["L_lab"] + cand["L_cloud"]
    cand["stage2_rank"] = np.arange(1, len(cand) + 1)
    return cand.sort_values("L", kind="stable").reset_index(drop=True)


def finish(out, hum, smoke):
    """Stage 4 tables: the checks (like sup_checks.csv) and the json."""
    f = files(out)
    cand = stage3_result(out)
    w = cand.iloc[0]
    reps = pd.read_parquet(f[4])
    reps = pd.concat([reps[(reps["part"] == p) & (reps["mu"] == w[f"mu_{p}"])] for p in PARTS], ignore_index=True)
    assert reps.groupby("part")["replicate"].nunique().to_dict() == {p: R[4] for p in PARTS}, "stage 4 incomplete"
    h = hum[hum["dataset"].isin(set(reps["dataset"]))] if smoke else hum
    chk = pc.summarize(reps[reps["stat"] != targets.GROUP_RATE], h[pc.KEY + ["value", "ci_lo", "ci_hi", "ci_rule",
                                                                          "in_check_list"]])
    chk["gap"] = (chk["sim_mean"] - chk["value"]) / ((chk["ci_hi"] - chk["ci_lo"]) / 2)
    chk["se_h"] = (chk["ci_hi"] - chk["ci_lo"]) / 3.92
    chk["z"] = (chk["sim_mean"] - chk["value"]) / np.sqrt(chk["se_h"] ** 2 + chk["se_mc"].fillna(0) ** 2)
    wide = chk[chk["in_check_list"] | (chk["stat"] == targets.PAUSE_SHARE)].rename(columns={"value": "human"})
    wide = wide.assign(sim=wide["sim_mean"], **{"pass": wide["verdict"] == "pass"})
    wide = wide[pc.KEY + ["human", "ci_lo", "ci_hi", "ci_rule", "in_check_list", "sim", "gap", "pass", "se_mc", "z"]]
    wide = wide.sort_values(["target", "topology", "dataset", "stat"], kind="stable").reset_index(drop=True)
    wide.to_csv(f["checks"], index=False)
    parts = {}
    for p in PARTS:
        L, missing, _ = loss(reps[reps["part"] == p], h[h["in_check_list"] & (h["part"] == p)])
        parts[p] = (L, missing)
    c = wide[wide["in_check_list"]]
    res = {"model": "APEM (refit)", "entry": "results/model_b_log.md entry 5 (P26)",
           "winner": {"a_th": float(w["a_th"]), "s_th": float(w["s_th"]), "mu_lab": float(w["mu_lab"]),
                      "mu_cloud": float(w["mu_cloud"])},
           "stage3_R50": {"seed": SEEDS[3], "L": float(w["L"]), "L_lab": float(w["L_lab"]),
                          "L_cloud": float(w["L_cloud"])},
           "stage4_R200": {"seed": SEEDS[4], "L": parts["lab"][0] + parts["cloud"][0], "L_lab": parts["lab"][0],
                           "L_cloud": parts["cloud"][0], "missing": parts["lab"][1] + parts["cloud"][1],
                           "passes": int(c["pass"].sum()), "checks": len(c),
                           "primary_passes": int(c.loc[c["target"] <= 4, "pass"].sum()),
                           "primary": int((c["target"] <= 4).sum()),
                           "median_abs_gap": float(c["gap"].abs().median())},
           "stage3_candidates": json.loads(cand.to_json(orient="records"))}
    f["json"].write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res["winner"]), json.dumps(res["stage4_R200"]))
    print(f"-> {f['checks']}\n-> {f['json']}")


def main():
    ap = argparse.ArgumentParser(description="P26 APEM refit (resumable)")
    ap.add_argument("--stage", required=True, choices=["1", "2", "3", "4", "all"])
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--smoke", type=Path, help="tiny design (2 rounds per cell); outputs in this directory")
    a = ap.parse_args()
    out = a.smoke or config.TABLES
    out.mkdir(parents=True, exist_ok=True)
    ds = designs(a.smoke is not None)
    hum = human()
    if a.smoke is None:
        for p in PARTS:
            assert (hum["in_check_list"] & (hum["part"] == p)).sum() == N_STATS[p]
    print({p: len(d) for p, d in ds.items()}, "rounds ->", out, flush=True)
    for stage in ([1, 2, 3, 4] if a.stage == "all" else [int(a.stage)]):
        if stage == 1:
            runs = stage1_runs()
        elif stage == 2:
            runs, centers = stage2_runs(out)
            print("stage 2 centers:\n" + centers.to_string(index=False), flush=True)
        elif stage == 3:
            settings = stage3_settings(out)
            print("stage 3 settings:\n" + settings.to_string(index=False), flush=True)
            runs = runs_of(settings)
        else:
            w = stage3_result(out).iloc[0]
            print("winner:", w[["a_th", "s_th", "mu_lab", "mu_cloud", "L"]].to_dict(), flush=True)
            runs = [(p, w["a_th"], w["s_th"], w[f"mu_{p}"]) for p in PARTS]
        run_parts(stage, runs, ds, hum, out, a.procs)
        if stage == 4:
            finish(out, hum, a.smoke is not None)


if __name__ == "__main__":
    main()

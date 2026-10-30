"""Path A P24, Step 4 checks and triggers, evaluated once, as pre-registered.

Reads results/tables/p24_reps_<preset>.parquet (L1, L1_M3, L0; a missing file is reported as missing), writes
results/tables/p24_checks.csv and prints the report.

Per preset and statistic (the 53 check statistics, in_check_list, and 10c pause_share_pct):
    R           replicates in the file; n = replicates with a value
    sim_mean    mean over replicates; sim_sd = SD over replicates (ddof 1); se_mc = sim_sd / sqrt(n)
    human, ci_lo, ci_hi   targets.human_check_values (train + val, 95% group-bootstrap CI)
    verdict     pass = sim_mean inside [ci_lo, ci_hi], miss = outside, no value = no replicate has a value
    borderline  sim_mean within 2 se_mc of a CI edge (the verdict is still by the point)
Rows with stat = reversal_rate_between_group_sd (one per cell, L3 trigger): sim_mean = mean over replicates of
the between-group SD (ddof 1) of the per-group reversal rates; human = that SD on the L5 runs, 95% CI by group
bootstrap (targets.between_group_sd_ci: 10,000 resamples, seed 573).

Triggers (L1 decides; L0 is reported only, never used for a decision):
    Rescue  L1 misses target 5 in >= 1 Full cell AND L1_M3 passes target 5 in both Full cells
    L1b     L1 misses 10a, 10c or 2 in >= 2 of the 6 cells, for any one of the three
    L3      L1's between-group SD is below the lower bound of the human SD's CI in >= 2 of the 6 cells
    L2      no trigger

    python scripts/p24_checks.py | tee results/p24_report.txt
"""
import warnings

import numpy as np
import pandas as pd

from slider import config
from slider import design
from slider import targets

warnings.filterwarnings("ignore", message="Pandas requires version")
PRESETS = ["L1", "L1_M3", "L0"]
KEY = ["target", "stat", "topology", "dataset"]
SD_STAT = "reversal_rate_between_group_sd"
L1B = [(10, targets.CHECKS[10], "10a"), (10, targets.PAUSE_SHARE, "10c"), (2, targets.CHECKS[2], "2")]


def summarize(reps, human):
    """One row per statistic of human: R, n, sim_mean, sim_sd, se_mc, the human value and CI, verdict, borderline."""
    s = reps.groupby(KEY)["value"].agg(n="count", sim_mean="mean", sim_sd="std").reset_index()
    out = human.merge(s, on=KEY, how="left", validate="one_to_one")
    out["R"] = reps["replicate"].nunique()
    out["n"] = out["n"].fillna(0).astype(int)
    out["se_mc"] = out["sim_sd"] / np.sqrt(out["n"])
    inside = (out["sim_mean"] >= out["ci_lo"]) & (out["sim_mean"] <= out["ci_hi"])
    out["verdict"] = np.where(out["sim_mean"].isna(), "no value", np.where(inside, "pass", "miss"))
    edge = np.minimum(np.abs(out["sim_mean"] - out["ci_lo"]), np.abs(out["sim_mean"] - out["ci_hi"]))
    out["borderline"] = edge <= 2 * out["se_mc"]
    return out


def human_sd():
    """Between-group SD of the human per-group reversal rates (L5 runs) and its group-bootstrap CI, per cell."""
    runs5, _ = design.load("L5")
    g = targets.group_rates(targets.reversal_rate_runs(runs5))
    rows = []
    for (topology, dataset), c in g.groupby(["topology", "dataset"], sort=False):
        lo, hi = targets.between_group_sd_ci(c["rate"].to_numpy())
        rows.append({"target": 8, "stat": SD_STAT, "topology": topology, "dataset": dataset,
                     "value": targets.between_group_sd(c["rate"]), "ci_lo": lo, "ci_hi": hi, "in_check_list": False})
    return pd.DataFrame(rows)


def sim_sd(reps):
    """Between-group SD of the per-group reversal rates of every replicate and cell (value = that SD)."""
    g = reps[reps["stat"] == targets.GROUP_RATE]
    s = g.groupby(["replicate", "topology", "dataset"])["value"].apply(targets.between_group_sd).rename("value")
    return s.reset_index().assign(target=8, stat=SD_STAT)


def cell(r):
    return f"{r['topology']} {r['dataset']}"


def main():
    human = targets.human_check_values()
    human = human[human["in_check_list"] | (human["stat"] == targets.PAUSE_SHARE)]     # the 53 and 10c
    sd_human = human_sd()
    tables, R = {}, {}
    for preset in PRESETS:
        path = config.P24_REPS[preset]
        if not path.exists():
            continue
        reps = pd.read_parquet(path)
        R[preset] = reps["replicate"].nunique()
        main_rows = summarize(reps[reps["stat"] != targets.GROUP_RATE], human)
        sd_rows = summarize(sim_sd(reps), sd_human[KEY + ["value", "ci_lo", "ci_hi", "in_check_list"]])
        tables[preset] = pd.concat([main_rows, sd_rows], ignore_index=True).assign(preset=preset)
    if not tables:
        raise SystemExit("no p24_reps_<preset>.parquet file found")
    out = pd.concat(tables.values(), ignore_index=True)
    out = out[["preset"] + KEY + ["in_check_list", "R", "n", "sim_mean", "sim_sd", "se_mc", "value", "ci_lo", "ci_hi",
                                  "verdict", "borderline"]].rename(columns={"value": "human"})
    out.to_csv(config.P24_CHECKS, index=False)

    # ---------------------------------------------------------------- report
    print("P24 checks, Step 4 main runs")
    print("R (replicates): " + ", ".join(f"{p} {R[p]}" if p in R else f"{p} missing" for p in PRESETS))
    if (out["n"] < out["R"]).any():
        short = out[out["n"] < out["R"]]
        print("statistics with replicates without a value: " +
              "; ".join(f"{r.preset} {r.stat} {r.topology} {r.dataset} n = {r.n}" for r in short.itertuples()))
    print(f"\nPass / miss (sim mean inside the human 95% CI); * = borderline (within 2 SE_mc of a CI edge)")
    marks = {}
    for p, t in tables.items():
        t = t[t["stat"] != SD_STAT]
        marks[p] = [("pass" if v == "pass" else "MISS" if v == "miss" else "none") + ("*" if b else "")
                    for v, b in zip(t["verdict"], t["borderline"])]
    first = tables[next(iter(tables))]
    first = first[first["stat"] != SD_STAT]
    print(f"{'target':>6}  {'stat':<28} {'cell':<13} " + " ".join(f"{p:<7}" for p in PRESETS))
    for i, r in enumerate(first.itertuples()):
        name = "10c" if r.stat == targets.PAUSE_SHARE else str(r.target)
        tag = "" if r.in_check_list else "  (L1b only)"
        print(f"{name:>6}  {r.stat:<28} {r.topology + ' ' + r.dataset:<13} "
              + " ".join(f"{marks[p][i] if p in marks else '-':<7}" for p in PRESETS) + tag)
    print("\nOf the 53 check statistics:")
    for p, t in tables.items():
        c = t[t["in_check_list"]]
        print(f"  {p:<6} misses {int((c['verdict'] == 'miss').sum())}, borderline {int(c['borderline'].sum())} "
              f"(misses that are borderline {int(((c['verdict'] == 'miss') & c['borderline']).sum())}), "
              f"no value {int((c['verdict'] == 'no value').sum())}")

    print("\nTriggers (L1 decides; L0 is reported only)")
    L1 = tables.get("L1")
    if L1 is None:
        print("  no L1 runs: no trigger evaluated")
        return

    def show(t, target, stat):
        rows = t[(t["target"] == target) & (t["stat"] == stat)]
        return "; ".join(f"{cell(r)} {r['verdict']} (sim {r['sim_mean']:.4g}, CI [{r['ci_lo']:.4g}, {r['ci_hi']:.4g}])"
                         for _, r in rows.iterrows()), rows

    text, t5 = show(L1, 5, targets.CHECKS[5])
    l1_miss = int((t5["verdict"] == "miss").sum())
    print(f"  Rescue: L1 target 5: {text}")
    if "L1_M3" in tables:
        text, m5 = show(tables["L1_M3"], 5, targets.CHECKS[5])
        m3_pass = int((m5["verdict"] == "pass").sum())
        print(f"          L1_M3 target 5: {text}")
        hit = l1_miss >= 1 and m3_pass == len(m5) == 2
        print(f"          L1 misses in {l1_miss} of {len(t5)} Full cells, L1_M3 passes {m3_pass} of {len(m5)}"
              f" -> {'TRIGGERED' if hit else 'not triggered'}")
    else:
        print(f"          L1 misses in {l1_miss} of {len(t5)} Full cells; no L1_M3 runs -> not evaluated")

    hits = []
    for target, stat, name in L1B:
        rows = L1[(L1["target"] == target) & (L1["stat"] == stat)]
        missed = rows[rows["verdict"] == "miss"]
        print(f"  L1b: {name:<3} misses in {len(missed)} of {len(rows)} cells"
              + (": " + ", ".join(cell(r) for _, r in missed.iterrows()) if len(missed) else ""))
        if len(missed) >= 2:
            hits.append(name)
    print(f"       -> {'TRIGGERED (' + ', '.join(hits) + ')' if hits else 'not triggered'}")

    sd = L1[L1["stat"] == SD_STAT]
    below = sd["sim_mean"] < sd["ci_lo"]
    for _, r in sd.iterrows():
        print(f"  L3: {cell(r):<13} L1 between-group SD {r['sim_mean']:.4g} (SE_mc {r['se_mc']:.2g}); "
              f"human SD CI [{r['ci_lo']:.4g}, {r['ci_hi']:.4g}]" + ("  below" if r["sim_mean"] < r["ci_lo"] else ""))
    print(f"       below the lower bound in {int(below.sum())} of {len(sd)} cells"
          f" -> {'TRIGGERED' if below.sum() >= 2 else 'not triggered'}")
    print("  L2: no trigger")
    print(f"\nwrote {config.P24_CHECKS}")


if __name__ == "__main__":
    main()

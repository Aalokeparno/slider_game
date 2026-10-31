"""Path A P24, Step 4 checks and triggers, evaluated once, as pre-registered.

Reads results/tables/p24_reps_<preset>.parquet (L1, final, L1_M3, L0, B1; a missing file is reported as missing), writes
results/tables/p24_checks.csv and prints the report.

Per preset and statistic (the 53 check statistics, in_check_list, and 10c pause_share_pct):
    R           replicates in the file; n = replicates with a value
    sim_mean    mean over replicates; sim_sd = SD over replicates (ddof 1); se_mc = sim_sd / sqrt(n)
    human, ci_lo, ci_hi   targets.human_check_values (train + val, 95% group-bootstrap CI), with zero_width_rule
    ci_rule     "" (the bootstrap CI) or the rule that replaced a zero-width CI
    verdict     pass = sim_mean inside [ci_lo, ci_hi], miss = outside, no value = no replicate has a value
    borderline  sim_mean within 2 se_mc of a CI edge (the verdict is still by the point)
Rows with stat = reversal_rate_between_group_sd (one per cell, L3 trigger): sim_mean = mean over replicates of
the between-group SD (ddof 1) of the per-group reversal rates; human = that SD on the L5 runs, 95% CI by group
bootstrap (targets.between_group_sd_ci: 10,000 resamples, seed 573).

Zero-width human CIs (width < 1e-9; rule decided Oct 29, 2026, before any results), zero_width_rule:
    success_rate_pct at 0 or 100   Clopper-Pearson 95% over the cell's n rounds: all successes
                                   [100 * 0.025^(1/n), 100], no success [0, 100 (1 - 0.025^(1/n))]
    crossing_time_median_s (10a)   pinned by the interface: value +/- 0.05 s (half the 0.1 s lab clock step)
    any other zero-width CI stops the script (no rule for it).

final (P25, L1 + L1b + L3) is reported next to L1, with no triggers.
B1 (Model B, EXPLORATORY; results/model_b_log.md entry 1) is reported next to final and L1, with no triggers.

Triggers (L1 decides; L0 is reported only, never used for a decision), see triggers:
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
PRESETS = ["L1", "final", "B1", "L1_M3", "L0"]    # final and B1 (EXPLORATORY): reported only, no triggers
EXPLORATORY = ["B1"]
KEY = ["target", "stat", "topology", "dataset"]
SD_STAT = "reversal_rate_between_group_sd"
L1B = [(10, targets.CHECKS[10], "10a"), (10, targets.PAUSE_SHARE, "10c"), (2, targets.CHECKS[2], "2")]
ZERO = 1e-9                 # a CI narrower than this has zero width
LAB_CLOCK_HALF = 0.05       # s: half the 0.1 s lab clock step


def rounds_per_cell():
    """{(topology, dataset): number of rounds} of the L3 runs (train + val), from train_targets.csv."""
    tt = pd.read_csv(config.TRAIN_TARGETS)
    tt = tt[(tt["role"] == "train+val") & (tt["stat"] == targets.CHECKS[1])]
    return {(t, d): n for t, d, n in zip(tt["topology"], tt["dataset"], tt["n_runs"])}


def zero_width_rule(human, n_rounds):
    """The human table with every zero-width CI (width < ZERO) replaced as in the module doc, and a ci_rule column.
    n_rounds: {(topology, dataset): rounds}, for the Clopper-Pearson interval."""
    human = human.assign(ci_rule="")
    for i, r in human[(human["ci_hi"] - human["ci_lo"]).abs() < ZERO].iterrows():
        if r["stat"] == targets.CHECKS[1] and r["value"] in (0, 100):
            q = 100 * 0.025 ** (1 / n_rounds[(r["topology"], r["dataset"])])
            lo, hi = (q, 100.0) if r["value"] == 100 else (0.0, 100 - q)
            human.loc[i, ["ci_lo", "ci_hi", "ci_rule"]] = [lo, hi, "Clopper-Pearson"]
        elif r["stat"] == targets.CHECKS[10]:
            human.loc[i, ["ci_lo", "ci_hi", "ci_rule"]] = [r["value"] - LAB_CLOCK_HALF, r["value"] + LAB_CLOCK_HALF,
                                                           "lab clock +/- 0.05 s"]
        else:
            raise SystemExit(f"zero-width human CI with no rule: target {r['target']} {r['stat']} "
                             f"{r['topology']} {r['dataset']}")
    return human


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
                     "value": targets.between_group_sd(c["rate"]), "ci_lo": lo, "ci_hi": hi,
                     "in_check_list": False, "ci_rule": ""})
    return pd.DataFrame(rows)


def sim_sd(reps):
    """Between-group SD of the per-group reversal rates of every replicate and cell (value = that SD)."""
    g = reps[reps["stat"] == targets.GROUP_RATE]
    s = g.groupby(["replicate", "topology", "dataset"])["value"].apply(targets.between_group_sd).rename("value")
    return s.reset_index().assign(target=8, stat=SD_STAT)


def build_tables(reps_by_preset, human, sd_human):
    """{preset: summarize rows of the 53 + 10c (human) and of the between-group SD (sd_human)}.
    reps_by_preset: {preset: the per-replicate table p24_main.py saves}."""
    tables = {}
    for preset, reps in reps_by_preset.items():
        main_rows = summarize(reps[reps["stat"] != targets.GROUP_RATE], human)
        sd_rows = summarize(sim_sd(reps), sd_human)
        tables[preset] = pd.concat([main_rows, sd_rows], ignore_index=True).assign(preset=preset)
    return tables


def rows_of(t, target, stat):
    return t[(t["target"] == target) & (t["stat"] == stat)]


def triggers(tables):
    """The pre-registered triggers from build_tables output. Returns a dict:
    rescue        True / False, or None if there are no L1_M3 runs; rescue_l1_miss (L1 Full cells missed),
                  rescue_m3_pass (L1_M3 Full cells passed), rescue_m3_cells
    l1b           True / False; l1b_misses {"10a", "10c", "2": L1 cells missed}
    l3            True / False; l3_below (L1 cells with sim_mean < the human CI lower bound), l3_cells
    None for everything if there are no L1 runs."""
    L1 = tables.get("L1")
    if L1 is None:
        return None
    t5 = rows_of(L1, 5, targets.CHECKS[5])
    out = {"rescue_l1_miss": int((t5["verdict"] == "miss").sum())}
    if "L1_M3" in tables:
        m5 = rows_of(tables["L1_M3"], 5, targets.CHECKS[5])
        out["rescue_m3_pass"], out["rescue_m3_cells"] = int((m5["verdict"] == "pass").sum()), len(m5)
        out["rescue"] = out["rescue_l1_miss"] >= 1 and out["rescue_m3_pass"] == len(m5) == 2
    else:
        out["rescue"] = None
    out["l1b_misses"] = {name: int((rows_of(L1, target, stat)["verdict"] == "miss").sum()) for target, stat, name in L1B}
    out["l1b"] = any(n >= 2 for n in out["l1b_misses"].values())
    sd = L1[L1["stat"] == SD_STAT]
    out["l3_below"], out["l3_cells"] = int((sd["sim_mean"] < sd["ci_lo"]).sum()), len(sd)
    out["l3"] = out["l3_below"] >= 2
    return out


def cell(r):
    return f"{r['topology']} {r['dataset']}"


def report(tables, R, changed, trig):
    print("P24 checks, Step 4 main runs")
    print("EXPLORATORY (reported only, never used for a decision): " + ", ".join(EXPLORATORY))
    print("R (replicates): " + ", ".join(f"{p} {R[p]}" if p in R else f"{p} missing" for p in PRESETS))
    print("human CIs replaced by the zero-width rule: " + ("; ".join(
        f"target {r.target} {r.stat} {r.topology} {r.dataset}: {r.ci_rule} [{r.ci_lo:.4g}, {r.ci_hi:.4g}]"
        for r in changed.itertuples()) if len(changed) else "none"))
    out = pd.concat(tables.values(), ignore_index=True)
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
        tag = ("" if r.in_check_list else "  (L1b only)") + (f"  [CI: {r.ci_rule}]" if r.ci_rule else "")
        print(f"{name:>6}  {r.stat:<28} {r.topology + ' ' + r.dataset:<13} "
              + " ".join(f"{marks[p][i] if p in marks else '-':<7}" for p in PRESETS) + tag)
    print("\nOf the 53 check statistics:")
    for p, t in tables.items():
        c = t[t["in_check_list"]]
        label = p + (" (EXPLORATORY)" if p in EXPLORATORY else "")
        print(f"  {label:<6} misses {int((c['verdict'] == 'miss').sum())}, borderline {int(c['borderline'].sum())} "
              f"(misses that are borderline {int(((c['verdict'] == 'miss') & c['borderline']).sum())}), "
              f"no value {int((c['verdict'] == 'no value').sum())}")

    print("\nTriggers (L1 decides; L0, final and B1 are reported only)")
    if trig is None:
        print("  no L1 runs: no trigger evaluated")
        return
    L1 = tables["L1"]

    def show(t, target, stat):
        return "; ".join(f"{cell(r)} {r['verdict']} (sim {r['sim_mean']:.4g}, CI [{r['ci_lo']:.4g}, {r['ci_hi']:.4g}])"
                         for _, r in rows_of(t, target, stat).iterrows())

    n5 = len(rows_of(L1, 5, targets.CHECKS[5]))
    print(f"  Rescue: L1 target 5: {show(L1, 5, targets.CHECKS[5])}")
    if trig["rescue"] is None:
        print(f"          L1 misses in {trig['rescue_l1_miss']} of {n5} Full cells; no L1_M3 runs -> not evaluated")
    else:
        print(f"          L1_M3 target 5: {show(tables['L1_M3'], 5, targets.CHECKS[5])}")
        print(f"          L1 misses in {trig['rescue_l1_miss']} of {n5} Full cells, L1_M3 passes "
              f"{trig['rescue_m3_pass']} of {trig['rescue_m3_cells']} -> {'TRIGGERED' if trig['rescue'] else 'not triggered'}")

    for target, stat, name in L1B:
        rows = rows_of(L1, target, stat)
        missed = rows[rows["verdict"] == "miss"]
        print(f"  L1b: {name:<3} misses in {len(missed)} of {len(rows)} cells"
              + (": " + ", ".join(cell(r) for _, r in missed.iterrows()) if len(missed) else ""))
    fired = [name for name, n in trig["l1b_misses"].items() if n >= 2]
    print(f"       -> {'TRIGGERED (' + ', '.join(fired) + ')' if trig['l1b'] else 'not triggered'}")

    for _, r in L1[L1["stat"] == SD_STAT].iterrows():
        print(f"  L3: {cell(r):<13} L1 between-group SD {r['sim_mean']:.4g} (SE_mc {r['se_mc']:.2g}); "
              f"human SD CI [{r['ci_lo']:.4g}, {r['ci_hi']:.4g}]" + ("  below" if r["sim_mean"] < r["ci_lo"] else ""))
    print(f"       below the lower bound in {trig['l3_below']} of {trig['l3_cells']} cells"
          f" -> {'TRIGGERED' if trig['l3'] else 'not triggered'}")
    print("  L2: no trigger")
    print(f"\nwrote {config.P24_CHECKS}")


def main():
    human = targets.human_check_values()
    human = human[human["in_check_list"] | (human["stat"] == targets.PAUSE_SHARE)]     # the 53 and 10c
    human = zero_width_rule(human, rounds_per_cell())
    changed = human[human["ci_rule"] != ""]
    reps_by_preset = {p: pd.read_parquet(config.P24_REPS[p]) for p in PRESETS if config.P24_REPS[p].exists()}
    if not reps_by_preset:
        raise SystemExit("no p24_reps_<preset>.parquet file found")
    tables = build_tables(reps_by_preset, human, human_sd())
    R = {p: reps["replicate"].nunique() for p, reps in reps_by_preset.items()}
    out = pd.concat(tables.values(), ignore_index=True)
    out = out[["preset"] + KEY + ["in_check_list", "R", "n", "sim_mean", "sim_sd", "se_mc", "value", "ci_lo", "ci_hi",
                                  "ci_rule", "verdict", "borderline"]].rename(columns={"value": "human"})
    out.to_csv(config.P24_CHECKS, index=False)
    report(tables, R, changed, triggers(tables))


if __name__ == "__main__":
    main()

"""Training targets: summary statistics of the human runs, per topology x dataset x role
(train, val, and train + val pooled), each with a 95% bootstrap CI over groups (log files).

No new definitions: % fraught is fraught.fraught_percent_hold, committed splits / stalls / breakers
come from h5 (main setting: cutoff 0.9, tau 1 s), Kaplan-Meier from stats.kaplan_meier, and
end-sitting uses the rule of HW 4 part B (forward-hold weights, every player counts, |s| >= 0.9).
CIs: stats.group_bootstrap_mean for means and shares, stats.group_bootstrap (same draws) for KM
quantiles and pooled ratios; 10,000 resamples, seed 573.

Target 8 (reversal rate, reversal_rate_table) uses the P2c method of labclock.py.
Target 9 (Agreement at reversal, agreement_at_reversal_table) uses hazard.py for the at-risk time.
"""
import numpy as np
import pandas as pd

from slider import config
from slider import h5
from slider import hazard
from slider import slider_logs as sl
from slider.fraught import fraught_percent_hold, round_end
from slider.labclock import CELLS, events_table, exposure, lab_log
from slider.reversals import run_events
from slider.stats import group_bootstrap, group_bootstrap_mean, kaplan_meier, km_quantiles

ROLES = [("train", {"train"}), ("val", {"val"}), ("train+val", {"train", "val"})]
COLUMNS = ["stat", "topology", "dataset", "role", "value", "ci_lo", "ci_hi", "n_runs", "n_groups"]


def km_quantile(km_table, p):
    """Quantile p of a kaplan_meier table: first time with S <= 1 - p (the median rule of kaplan_meier)."""
    hit = km_table[km_table["survival"] <= 1 - p + 1e-9]
    return hit["time"].iloc[0] if len(hit) > 0 else np.nan


def end_sitting_time(run, t_end, cutoff=0.9):
    """(player-time at |s| >= cutoff, all player-time) of one run, forward-hold weights (HW 4 part B)."""
    s = sl.sign_correct(run)
    dt = np.diff(np.append(run["t"], t_end))
    return np.sum(dt[:, None] * (np.abs(s) >= cutoff)), np.sum(dt) * s.shape[1]


def by_group(df, *columns):
    """One tuple of arrays per group (file), in sorted order, for stats.group_bootstrap."""
    return [tuple(d[c].to_numpy(dtype=float) for c in columns) for _, d in df.groupby("file", sort=True)]


def per_run_values(runs, table):
    """One row per run: what the targets need (time to consensus, % fraught, % committed, end-sitting)."""
    sp, per_run = h5.committed_spells(runs, table, h5.MAIN_CUTOFF, h5.MAIN_TAU)
    r = table[["file", "wave", "dataset", "topology", "role", "success", "session", "duration"]].copy()
    limit = r["session"].map(config.TIME_LIMIT)
    r["ttc"] = np.where(r["success"], r["duration"], limit)             # failures censored at the time limit
    r["fraught %"] = [fraught_percent_hold(k, runs, table) for k in range(len(runs))]
    r["committed %"] = per_run["% committed"].to_numpy()                # same order as runs
    sits = [end_sitting_time(run, round_end(run["t"], table["end"][k], table["session"][k])) for k, run in enumerate(runs)]
    r["end time"], r["player time"] = [a for a, b in sits], [b for a, b in sits]
    sp["role"] = sp["file"].map(dict(zip(table["file"], table["role"])))
    return r, sp


def train_targets(runs, table, n_boot=10000, seed=config.SEED):
    """The target statistics in long format (see COLUMNS)."""
    r_all, sp_all = per_run_values(runs, table)
    rows = []
    for topology in h5.TOPOLOGIES:
        for dataset in sorted(r_all.loc[r_all["topology"] == topology, "dataset"].unique()):
            for role, roles in ROLES:
                r = r_all[(r_all["topology"] == topology) & (r_all["dataset"] == dataset) & r_all["role"].isin(roles)]
                sp = sp_all[(sp_all["topology"] == topology) & (sp_all["dataset"] == dataset) & sp_all["role"].isin(roles)]
                if len(r) == 0:
                    continue
                cell = {"topology": topology, "dataset": dataset, "role": role,
                        "n_runs": len(r), "n_groups": r["file"].nunique()}

                def add(stat, value, ci):
                    rows.append({"stat": stat, **cell, "value": value, "ci_lo": ci[0], "ci_hi": ci[1]})

                # 1 success rate
                add("success_rate_pct", 100 * r["success"].mean(),
                    group_bootstrap_mean(100 * r["success"], r["file"], n_boot, seed))

                # 2 time to consensus (KM, failures censored at the limit)
                km = kaplan_meier(r["ttc"], r["success"])[0]
                r = r.assign(event=r["success"].astype(float))
                lo, hi = group_bootstrap(by_group(r, "ttc", "event"),
                                         lambda d, e: km_quantiles(d, e, (0.5, 0.25, 0.75)), n_boot, seed)
                for j, (name, p) in enumerate([("median", 0.5), ("q1", 0.25), ("q3", 0.75)]):
                    add(f"consensus_time_km_{name}_s", km_quantile(km, p), (lo[j], hi[j]))

                # 3 mean % fraught (forward hold)
                add("fraught_pct_mean", r["fraught %"].mean(),
                    group_bootstrap_mean(r["fraught %"], r["file"], n_boot, seed))

                # 4 mean % committed split; KM median stall (all k, and by k on Full)
                add("committed_pct_mean", r["committed %"].mean(),
                    group_bootstrap_mean(r["committed %"], r["file"], n_boot, seed))
                sp = sp.assign(escape=(sp["after"] != "censored").astype(float))
                parts = [("stall_km_median_s", sp)]
                if topology == "Full":
                    parts += [(f"stall_km_median_s_k{k}", sp[sp["k"] == k]) for k in [1, 2, 3]]
                for stat, d in parts:
                    median = kaplan_meier(d["duration"], d["escape"] > 0)[1] if len(d) > 0 else np.nan
                    ci = group_bootstrap(by_group(d, "duration", "escape"),
                                         lambda x, e: km_quantiles(x, e > 0)[0], n_boot, seed)
                    add(stat, median, ci)

                # 5 minority-breaker excess (Full): (observed - expected) / n, as in h5
                if topology == "Full":
                    b = h5.breaker_summary(sp, n_boot, seed)
                    add("breaker_excess", b["(obs - exp) / n"], (b["CI low"], b["CI high"]))

                # 6 end-sitting: % of player-time at |s| >= 0.9 (pooled over runs)
                add("end_sitting_pct", 100 * r["end time"].sum() / r["player time"].sum(),
                    group_bootstrap(by_group(r, "end time", "player time"),
                                    lambda a, b: 100 * a.sum() / b.sum(), n_boot, seed))

                # 7 relapse share among committed spells that ended
                ended = sp[sp["after"] != "censored"]
                relapse = 100 * (ended["after"] == "relapse")
                add("relapse_pct", relapse.mean() if len(ended) > 0 else np.nan,
                    group_bootstrap_mean(relapse, ended["file"], n_boot, seed))
    return pd.DataFrame(rows, columns=COLUMNS)


def show(targets, topology):
    """One topology as a wide table: rows = stat, columns = dataset x role, cells "value [lo, hi]"."""
    t = targets[targets["topology"] == topology].copy()
    t["cell"] = [f"{v:.2f} [{lo:.2f}, {hi:.2f}]" for v, lo, hi in zip(t["value"], t["ci_lo"], t["ci_hi"])]
    order = list(dict.fromkeys(t["stat"]))
    wide = t.pivot_table(index="stat", columns=["dataset", "role"], values="cell", aggfunc="first")
    return wide.reindex(order).reindex(columns=[(d, r) for d in sorted(t["dataset"].unique()) for r, _ in ROLES])


# ------------------------------------------------------------- target 8: reversal rate

def reversal_rate_table(runs, m=2, tick=0.1, lab_clock_cc=True):
    """Target 8: reversals per human player-minute in each cell (topology x dataset), pooled over runs.

    Reversals: labclock.events_table (reversals.run_events with m, human players only). With
    lab_clock_cc=True, CC runs first go through labclock.lab_log with this tick; lab data is used as logged.
    Player-minutes: labclock.exposure (every human player until fraught.round_end).
    rate = all reversals / all player-minutes of the cell; 95% CI by group bootstrap
    (log files resampled, stats.group_bootstrap, 10,000 resamples, seed 573), as in P2c.
    """
    events = events_table(runs, m=m, lab_clock=lab_clock_cc, tick=tick)
    reversals = events[events["type"] == "reversal"].groupby(["file", "wave"]).size().rename("reversals")
    r = exposure(runs).groupby(["topology", "dataset", "file", "wave"], as_index=False)["minutes"].sum()
    r = r.join(reversals, on=["file", "wave"])
    r["reversals"] = r["reversals"].fillna(0)                  # runs with no reversal
    out = []
    for topology, dataset in CELLS:
        c = r[(r["topology"] == topology) & (r["dataset"] == dataset)]
        if len(c) == 0:
            continue
        per_group = [(d["reversals"].to_numpy(float), d["minutes"].to_numpy(float)) for _, d in c.groupby("file", sort=True)]
        lo, hi = group_bootstrap(per_group, lambda rev, mi: rev.sum() / mi.sum(), 10000, config.SEED)
        out.append({"topology": topology, "dataset": dataset, "groups": c["file"].nunique(), "runs": len(c),
                    "reversals": int(c["reversals"].sum()), "player_min": c["minutes"].sum(),
                    "rate": c["reversals"].sum() / c["minutes"].sum(), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out)


# ------------------------------------------------------------- target 9: Agreement at reversal

A_STATS = ["median", "q1", "q3", "pct_A_gt_0.5"]


def describe_A(x):
    """median, q1, q3 and % above 0.5 of Agreement values (NaN if there are none)."""
    if len(x) == 0:
        return np.full(4, np.nan)
    return np.append(np.percentile(x, [50, 25, 75]), 100 * np.mean(x > 0.5))


def agreement_bin(A):
    """Bin 0..9 of width 0.1 on [0, 1], left-closed; the last bin includes 1.0."""
    return np.minimum(np.floor(np.asarray(A, dtype=float) * 10 + 1e-9), 9).astype(int)


def agreement_at_reversal_table(runs, m=2, tick=0.1, b=0.25, lab_clock_cc=True):
    """Target 9: Agreement A_i (hazard.agreement, in [0, 1]) at reversals and over at-risk time, per cell.

    f_rev   A_i of the reverser at the last row before the reversal's first step (row_before of run_events, m).
    f_time  A of hazard.at_risk_samples (bins of b seconds from each player's first move to fraught.round_end).
    With lab_clock_cc=True, CC runs first go through labclock.lab_log with this tick; lab data is used as logged.
    Returns (summary, h):
      summary  long format, one row per cell, measure (f_rev, f_time) and stat (A_STATS):
               groups, runs, n (reversals or samples), value, ci_lo, ci_hi.
      h        one row per cell and A bin (width 0.1, left-closed, last bin includes 1.0): reversals,
               at_risk_min (samples x b / 60), rate = reversals / at_risk_min, ci_lo, ci_hi,
               sparse (at_risk_min < 0.5).
    95% CIs: group bootstrap (log files resampled, stats.group_bootstrap, 10,000 resamples, seed 573).
    A resample with no at-risk time in a bin has rate NaN, which group_bootstrap counts as +inf.
    """
    rev, smp, info = [], [], []
    for run in runs:
        if lab_clock_cc and run["dataset"] == "CC":
            t_lab, X_lab = lab_log(run["t"], run["X"], tick)
            run = {**run, "t": t_lab, "X": X_lab}
        cell = {"topology": sl.label_topology(run["A"]), "dataset": run["dataset"], "file": run["file"]}
        A = hazard.agreement(run)
        ev = run_events(run, m)
        ev = ev[ev["type"] == "reversal"]
        rev.append(pd.DataFrame({**cell, "A": A[ev["row_before"].to_numpy(int), ev["player"].to_numpy(int)]}))
        smp.append(hazard.at_risk_samples(run, b)[["A"]].assign(**cell))
        info.append(cell)
    rev, smp, info = pd.concat(rev), pd.concat(smp), pd.DataFrame(info)

    def rate(rv, sm):
        with np.errstate(divide="ignore", invalid="ignore"):
            return rv.reshape(-1, 10).sum(axis=0) / (sm.reshape(-1, 10).sum(axis=0) * b / 60)

    summary, h = [], []
    for topology, dataset in CELLS:
        runs_c = info[(info["topology"] == topology) & (info["dataset"] == dataset)]
        if len(runs_c) == 0:
            continue
        r = rev[(rev["topology"] == topology) & (rev["dataset"] == dataset)]
        s = smp[(smp["topology"] == topology) & (smp["dataset"] == dataset)]
        files = sorted(runs_c["file"].unique())
        per_group = [(r.loc[r["file"] == f, "A"].to_numpy(float), s.loc[s["file"] == f, "A"].to_numpy(float))
                     for f in files]
        cell = {"topology": topology, "dataset": dataset}

        # f_rev and f_time: median, q1, q3, % > 0.5
        lo, hi = group_bootstrap(per_group, lambda a, c: np.append(describe_A(a), describe_A(c)), 10000, config.SEED)
        value = np.append(describe_A(r["A"].to_numpy(float)), describe_A(s["A"].to_numpy(float)))
        for i, (measure, n) in enumerate([("f_rev", len(r)), ("f_time", len(s))]):
            for j, stat in enumerate(A_STATS):
                summary.append({**cell, "measure": measure, "groups": len(files), "runs": len(runs_c), "n": n,
                                "stat": stat, "value": value[4 * i + j], "ci_lo": lo[4 * i + j], "ci_hi": hi[4 * i + j]})

        # h(A): reversals / at-risk minutes in each A bin
        counts = [(np.bincount(agreement_bin(a), minlength=10), np.bincount(agreement_bin(c), minlength=10))
                  for a, c in per_group]
        lo, hi = group_bootstrap(counts, rate, 10000, config.SEED)
        n_rev = np.bincount(agreement_bin(r["A"]), minlength=10)
        n_smp = np.bincount(agreement_bin(s["A"]), minlength=10)
        point, minutes = rate(n_rev, n_smp), n_smp * b / 60
        for j in range(10):
            h.append({**cell, "A_lo": j / 10, "A_hi": (j + 1) / 10, "reversals": int(n_rev[j]),
                      "at_risk_min": minutes[j], "rate": point[j], "ci_lo": lo[j], "ci_hi": hi[j],
                      "sparse": minutes[j] < 0.5})
    return pd.DataFrame(summary), pd.DataFrame(h)

"""Training targets: summary statistics of the human runs, per topology x dataset x role
(train, val, and train + val pooled), each with a 95% bootstrap CI over groups (log files).

No new definitions: % fraught is fraught.fraught_percent_hold, committed splits / stalls / breakers
come from h5 (main setting: cutoff 0.9, tau 1 s), Kaplan-Meier from stats.kaplan_meier, and
end-sitting uses the rule of HW 4 part B (forward-hold weights, every player counts, |s| >= 0.9).
CIs: stats.group_bootstrap_mean for means and shares, stats.group_bootstrap (same draws) for KM
quantiles and pooled ratios; 10,000 resamples, seed 573.

Target 8 (reversal rate, reversal_rate_table) uses the P2c method of labclock.py.
Target 9 (Agreement at reversal, agreement_at_reversal_table) uses hazard.py for the at-risk time.
Target 10 (crossing time, crossing_time_table) uses crossings.py.
"""
import numpy as np
import pandas as pd

from slider import config
from slider import h5
from slider import hazard
from slider import slider_logs as sl
from slider import traps
from slider.crossings import JUMP, crossing_events
from slider.fraught import fraught_percent_hold, round_end
from slider.labclock import CELLS, events_table, exposure, lab_log
from slider.reversals import run_events, step_sizes, to_steps
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

def reversal_rate_runs(runs, m=2, tick=0.1, lab_clock_cc=True):
    """Target 8 per run: topology, dataset, file, wave, minutes (human players, labclock.exposure) and reversals."""
    events = events_table(runs, m=m, lab_clock=lab_clock_cc, tick=tick)
    reversals = events[events["type"] == "reversal"].groupby(["file", "wave"]).size().rename("reversals")
    r = exposure(runs).groupby(["topology", "dataset", "file", "wave"], as_index=False)["minutes"].sum()
    r = r.join(reversals, on=["file", "wave"])
    r["reversals"] = r["reversals"].fillna(0)                  # runs with no reversal
    return r


def reversal_rate_table(runs, m=2, tick=0.1, lab_clock_cc=True, n_boot=10000):
    """Target 8: reversals per human player-minute in each cell (topology x dataset), pooled over runs.

    Reversals: labclock.events_table (reversals.run_events with m, human players only). With
    lab_clock_cc=True, CC runs first go through labclock.lab_log with this tick; lab data is used as logged.
    Player-minutes: labclock.exposure (every human player until fraught.round_end).
    rate = all reversals / all player-minutes of the cell; 95% CI by group bootstrap
    (log files resampled, stats.group_bootstrap, n_boot = 10,000 resamples, seed 573), as in P2c.
    """
    r = reversal_rate_runs(runs, m, tick, lab_clock_cc)
    out = []
    for topology, dataset in CELLS:
        c = r[(r["topology"] == topology) & (r["dataset"] == dataset)]
        if len(c) == 0:
            continue
        per_group = [(d["reversals"].to_numpy(float), d["minutes"].to_numpy(float)) for _, d in c.groupby("file", sort=True)]
        lo, hi = group_bootstrap(per_group, lambda rev, mi: rev.sum() / mi.sum(), n_boot, config.SEED)
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


def agreement_values(runs, m=2, tick=0.1, b=0.25, lab_clock_cc=True, samples=True):
    """Target 9 inputs: (rev, smp, info). rev: A of the reverser at row_before of each reversal; smp: A of the
    at-risk samples (only if samples); info: one row per run. Each with topology, dataset, file."""
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
        if samples:
            smp.append(hazard.at_risk_samples(run, b)[["A"]].assign(**cell))
        info.append(cell)
    return pd.concat(rev), (pd.concat(smp) if samples else None), pd.DataFrame(info)


def agreement_at_reversal_table(runs, m=2, tick=0.1, b=0.25, lab_clock_cc=True, n_boot=10000):
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
    95% CIs: group bootstrap (log files resampled, stats.group_bootstrap, n_boot = 10,000 resamples, seed 573).
    A resample with no at-risk time in a bin has rate NaN, which group_bootstrap counts as +inf.
    """
    rev, smp, info = agreement_values(runs, m, tick, b, lab_clock_cc)

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
        lo, hi = group_bootstrap(per_group, lambda a, c: np.append(describe_A(a), describe_A(c)), n_boot, config.SEED)
        value = np.append(describe_A(r["A"].to_numpy(float)), describe_A(s["A"].to_numpy(float)))
        for i, (measure, n) in enumerate([("f_rev", len(r)), ("f_time", len(s))]):
            for j, stat in enumerate(A_STATS):
                summary.append({**cell, "measure": measure, "groups": len(files), "runs": len(runs_c), "n": n,
                                "stat": stat, "value": value[4 * i + j], "ci_lo": lo[4 * i + j], "ci_hi": hi[4 * i + j]})

        # h(A): reversals / at-risk minutes in each A bin
        counts = [(np.bincount(agreement_bin(a), minlength=10), np.bincount(agreement_bin(c), minlength=10))
                  for a, c in per_group]
        lo, hi = group_bootstrap(counts, rate, n_boot, config.SEED)
        n_rev = np.bincount(agreement_bin(r["A"]), minlength=10)
        n_smp = np.bincount(agreement_bin(s["A"]), minlength=10)
        point, minutes = rate(n_rev, n_smp), n_smp * b / 60
        for j in range(10):
            h.append({**cell, "A_lo": j / 10, "A_hi": (j + 1) / 10, "reversals": int(n_rev[j]),
                      "at_risk_min": minutes[j], "rate": point[j], "ci_lo": lo[j], "ci_hi": hi[j],
                      "sparse": minutes[j] < 0.5})
    return pd.DataFrame(summary), pd.DataFrame(h)


# ------------------------------------------------------------- target 10: crossing time

STATUSES = ["complete", "aborted", "returned", "unfinished"]
CROSSING_STATS = ["crossing_time_median_s", "crossing_time_q1_s", "crossing_time_q3_s", "pace_steps_per_s",
                  "pause_share_pct", "pct_with_pause", "jumps_per_player_min"]


def crossing_stats(duration, steps, moving, pause_time, n_pauses, jumps, minutes):
    """CROSSING_STATS of a set of complete, jump-free crossings (and the jumps and player-minutes of their runs)."""
    if len(duration) == 0:
        return np.append(np.full(6, np.nan), jumps.sum() / minutes.sum())
    return np.append(np.percentile(duration, [50, 25, 75]),
                     [steps.sum() / moving.sum(), 100 * pause_time.sum() / duration.sum(),
                      100 * np.mean(n_pauses > 0), jumps.sum() / minutes.sum()])


def crossing_values(runs, pause=0.3, lab_clock_cc=True):
    """Target 10 inputs: (crossings, per_run). crossings: every crossing with topology, dataset, file, wave, player
    and the crossing_events columns; per_run: topology, dataset, file, wave and jumps (raw rows)."""
    crossings, per_run = [], []
    for run in runs:
        raw_k = to_steps(sl.sign_correct(run))
        t, k = run["t"], raw_k
        if lab_clock_cc and run["dataset"] == "CC":
            t, X_lab = lab_log(run["t"], run["X"])
            k = to_steps(sl.sign_correct({**run, "X": X_lab}))
        info = {"topology": sl.label_topology(run["A"]), "dataset": run["dataset"], "file": run["file"],
                "wave": run["wave"]}
        for j in range(7):
            if j + 1 not in run["bot_players"]:
                crossings.append(crossing_events(t, k[:, j], run["t"], raw_k[:, j], pause).assign(**info, player=j))
        per_run.append({**info, "jumps": int(np.sum(step_sizes(run) >= JUMP))})
    cr = pd.concat([c for c in crossings if len(c) > 0], ignore_index=True)
    cr = cr[["topology", "dataset", "file", "wave", "player"] + [c for c in cr.columns if c not in info and c != "player"]]
    return cr, pd.DataFrame(per_run)


def crossing_time_table(runs, pause=0.3, lab_clock_cc=True, n_boot=10000):
    """Target 10: crossings from one end band to the other (crossings.crossing_events), per cell.

    With lab_clock_cc=True, CC runs first go through labclock.lab_log (tick 0.1 s); lab data is used as logged.
    has_jump and the jump rate always use the raw rows. Returns (summary, crossings):
      summary    long format, one row per cell and stat: groups, runs, stat, value, ci_lo, ci_hi.
                 Counts (no CI): started, complete, aborted, returned, unfinished, with_jump (any status),
                 used (complete, no jump), jumps (raw rows with |Δk| >= 4, reversals.step_sizes, human players)
                 and player_min (labclock.exposure).
                 With CI (CROSSING_STATS, used crossings only): 10a median, Q1, Q3 of crossing time (s);
                 10b pace = all steps / all moving time (steps/s); 10c pause share = all pause time / all
                 crossing time (%) and % of crossings with >= 1 pause; jumps per human player-minute (all rows).
      crossings  every crossing: topology, dataset, file, wave, player, then the crossing_events columns.
    95% CIs: group bootstrap (log files resampled, stats.group_bootstrap, n_boot = 10,000 resamples, seed 573).
    """
    cr, per_run = crossing_values(runs, pause, lab_clock_cc)
    minutes = exposure(runs).groupby(["file", "wave"], as_index=False)["minutes"].sum()
    per_run = per_run.merge(minutes, on=["file", "wave"])

    summary = []
    for topology, dataset in CELLS:
        rc = per_run[(per_run["topology"] == topology) & (per_run["dataset"] == dataset)]
        if len(rc) == 0:
            continue
        c = cr[(cr["topology"] == topology) & (cr["dataset"] == dataset)]
        used = c[(c["status"] == "complete") & ~c["has_jump"].astype(bool)]
        files = sorted(rc["file"].unique())
        per_group = [tuple(used.loc[used["file"] == f, col].to_numpy(float)
                           for col in ["duration", "steps", "moving", "pause_time", "n_pauses"])
                     + tuple(rc.loc[rc["file"] == f, col].to_numpy(float) for col in ["jumps", "minutes"])
                     for f in files]
        lo, hi = group_bootstrap(per_group, crossing_stats, n_boot, config.SEED)
        value = crossing_stats(*[np.concatenate(a) for a in zip(*per_group)])
        cell = {"topology": topology, "dataset": dataset, "groups": len(files), "runs": len(rc)}
        counts = {"started": len(c), **{s: int(np.sum(c["status"] == s)) for s in STATUSES},
                  "with_jump": int(c["has_jump"].sum()), "used": len(used), "jumps": int(rc["jumps"].sum()),
                  "player_min": rc["minutes"].sum()}
        summary += [{**cell, "stat": s, "value": v, "ci_lo": np.nan, "ci_hi": np.nan} for s, v in counts.items()]
        summary += [{**cell, "stat": s, "value": value[j], "ci_lo": lo[j], "ci_hi": hi[j]}
                    for j, s in enumerate(CROSSING_STATS)]
    return pd.DataFrame(summary), cr


# ------------------------------------------------------------- check statistics (fast path)

CHECKS = {1: "success_rate_pct", 2: "consensus_time_km_median_s", 3: "fraught_pct_mean", 4: "committed_pct_mean",
          5: "breaker_excess", 6: "end_sitting_pct", 7: "relapse_pct", 8: "reversal_rate",
          9: "f_rev_median", 10: "crossing_time_median_s"}
PAUSE_SHARE = "pause_share_pct"           # 10c, used only by the L1b trigger (P24)
GROUP_RATE = "reversal_rate_group"        # target 8 per group, used only by the L3 trigger (P24)
TRAP_HOPS = 11                            # optional extra (P27): traps.trap_hops on the Hierarchy cells


def median_or_limit(median, sessions):
    """Target 2 for every model (P24): the KM median, or the cell's time limit if there is none (success < 50%)."""
    return sessions.map(config.TIME_LIMIT).max() if np.isnan(median) else median


def group_rates(rates):
    """Target 8 per group: topology, dataset, group, rate (reversals / player-minutes of the group's runs in the
    cell), from reversal_rate_runs; groups in sorted order within each cell."""
    out = []
    for (topology, dataset, group), g in rates.groupby(["topology", "dataset", "file"], sort=True):
        out.append({"topology": topology, "dataset": dataset, "group": group,
                    "rate": g["reversals"].sum() / g["minutes"].sum()})
    return pd.DataFrame(out, columns=["topology", "dataset", "group", "rate"])


def between_group_sd(rates):
    """L3 trigger statistic: SD (ddof = 1) of the per-group reversal rates of one cell."""
    return np.std(np.asarray(rates, dtype=float), ddof=1)


def between_group_sd_ci(rates, n_boot=10000, seed=config.SEED):
    """95% CI of between_group_sd by group bootstrap (one value per group, in sorted group order)."""
    return group_bootstrap([(np.array([x], dtype=float),) for x in rates], between_group_sd, n_boot, seed)


def check_stats(runs3, table3, runs5, m=2, tick=0.1, pause=0.3, trap_hops=False):
    """The check statistics, one per target per cell, as point values (no bootstrap), with the same code as
    the full functions. Long format: target, stat, topology, dataset, group ("" for a cell value), value,
    in_check_list.

    targets 1-7  train_targets on (runs3, table3), all runs pooled (= its "train+val" rows for train + val runs):
                 success_rate_pct, consensus_time_km_median_s, fraught_pct_mean, committed_pct_mean,
                 breaker_excess (Full only), end_sitting_pct, relapse_pct
                 target 2: the cell's time limit if there is no KM median (success < 50%), see median_or_limit
    target 8     reversal_rate_table(runs5): rate
    target 9     agreement_at_reversal_table(runs5): f_rev median (Agreement at reversal)
    target 10    crossing_time_table(runs5): crossing_time_median_s (10a)
    That is 56 statistics; in_check_list marks the 53 of the check list: target 3 (% fraught) only in the
    Hierarchy cells (Full and Star have no fraught states, so it is structurally 0 there).
    Not in the check list (P24 triggers only):
    10c          crossing_time_table(runs5): pause_share_pct, per cell (L1b)
    8 per group  reversal_rate_table of each group's runs5: rate, one row per cell and group (L3)
    Optional (trap_hops=True, P27; not in the check list): target 11, traps.trap_hops on the Hierarchy rounds of
    runs5, per cell: trap_<n_spells | n_distinct | n_hops | any_hop>_mean. With trap_hops=False (default) the
    output is unchanged.
    """
    rows = []

    def add(target, topology, dataset, value, stat=None, group=""):
        rows.append({"target": target, "stat": stat or CHECKS[target], "topology": topology, "dataset": dataset,
                     "group": group, "value": value})

    r_all, sp_all = per_run_values(runs3, table3)
    for topology in h5.TOPOLOGIES:
        for dataset in sorted(r_all.loc[r_all["topology"] == topology, "dataset"].unique()):
            r = r_all[(r_all["topology"] == topology) & (r_all["dataset"] == dataset)]
            sp = sp_all[(sp_all["topology"] == topology) & (sp_all["dataset"] == dataset)]
            add(1, topology, dataset, 100 * r["success"].mean())
            add(2, topology, dataset, median_or_limit(km_quantile(kaplan_meier(r["ttc"], r["success"])[0], 0.5), r["session"]))
            add(3, topology, dataset, r["fraught %"].mean())
            add(4, topology, dataset, r["committed %"].mean())
            if topology == "Full":                                # h5.breaker_test: mean of minority - k/7
                ok = sp[(sp["after"] != "censored") & (sp["breaker_status"] == "ok")]
                d = ok["minority"].astype(float).to_numpy() - np.asarray(ok["k"] / 7, dtype=float)
                add(5, topology, dataset, d.mean() if len(ok) > 0 else np.nan)
            add(6, topology, dataset, 100 * r["end time"].sum() / r["player time"].sum())
            ended = sp[sp["after"] != "censored"]
            relapse = 100 * (ended["after"] == "relapse")
            add(7, topology, dataset, relapse.mean() if len(ended) > 0 else np.nan)

    rates = reversal_rate_runs(runs5, m, tick)
    per_group = group_rates(rates)
    rev, _, _ = agreement_values(runs5, m, tick, samples=False)
    cr, _ = crossing_values(runs5, pause)
    extra = []
    for topology, dataset in CELLS:
        c = rates[(rates["topology"] == topology) & (rates["dataset"] == dataset)]
        if len(c) == 0:
            continue
        add(8, topology, dataset, c["reversals"].sum() / c["minutes"].sum())
        a = rev[(rev["topology"] == topology) & (rev["dataset"] == dataset)]
        add(9, topology, dataset, describe_A(a["A"].to_numpy(float))[0])
        x = cr[(cr["topology"] == topology) & (cr["dataset"] == dataset)]
        used = x[(x["status"] == "complete") & ~x["has_jump"].astype(bool)]
        add(10, topology, dataset, np.percentile(used["duration"].to_numpy(float), [50, 25, 75])[0] if len(used) else np.nan)
        u = used.sort_values("file", kind="stable")              # crossing_time_table's group order: same sums
        extra.append((10, topology, dataset, 100 * u["pause_time"].to_numpy(float).sum() / u["duration"].to_numpy(float).sum()
                      if len(used) else np.nan, PAUSE_SHARE, ""))
        g = per_group[(per_group["topology"] == topology) & (per_group["dataset"] == dataset)]
        extra += [(8, topology, dataset, rate, GROUP_RATE, group) for group, rate in zip(g["group"], g["rate"])]
    n_main = len(rows)
    for e in extra:
        add(*e)
    if trap_hops:
        hops = traps.trap_hop_runs(runs5)
        for (topology, dataset), c in hops[hops["topology"] == "Hierarchy"].groupby(["topology", "dataset"]):
            for stat in traps.STATS:
                add(TRAP_HOPS, topology, dataset, c[stat].mean(), f"trap_{stat}_mean")
    out = pd.DataFrame(rows, columns=["target", "stat", "topology", "dataset", "group", "value"])
    out["in_check_list"] = (np.arange(len(out)) < n_main) & ((out["target"] != 3) | (out["topology"] == "Hierarchy"))
    return out


def full_check_values(runs3, table3, runs5, n_boot=10000):
    """The same statistics read from the full functions (train_targets "train+val" rows, reversal_rate_table,
    agreement_at_reversal_table, crossing_time_table; target 8 per group: reversal_rate_table of the group's
    runs), in check_stats' row order. Target 2 takes median_or_limit here too. For tests."""
    tt = train_targets(runs3, table3, n_boot=n_boot)
    tt = tt[tt["role"] == "train+val"].set_index(["stat", "topology", "dataset"])["value"]
    rr = reversal_rate_table(runs5, n_boot=n_boot).set_index(["topology", "dataset"])["rate"]
    ag = agreement_at_reversal_table(runs5, n_boot=n_boot)[0]
    ag = ag[(ag["measure"] == "f_rev") & (ag["stat"] == "median")].set_index(["topology", "dataset"])["value"]
    cs = crossing_time_table(runs5, n_boot=n_boot)[0].set_index(["stat", "topology", "dataset"])["value"]
    by_group = {}
    for f in sorted({run["file"] for run in runs5}):
        t = reversal_rate_table([run for run in runs5 if run["file"] == f], n_boot=n_boot)
        by_group.update({(top, ds, f): rate for top, ds, rate in zip(t["topology"], t["dataset"], t["rate"])})
    fast = check_stats(runs3, table3, runs5)
    values = []
    for t, s, top, ds, g in zip(fast["target"], fast["stat"], fast["topology"], fast["dataset"], fast["group"]):
        if s == GROUP_RATE:
            values.append(by_group[(top, ds, g)])
        elif t == 2:
            sessions = table3.loc[(table3["topology"] == top) & (table3["dataset"] == ds), "session"]
            values.append(median_or_limit(tt[(s, top, ds)], sessions))
        elif t <= 7:
            values.append(tt[(s, top, ds)])
        else:
            values.append({8: rr, 9: ag}[t][(top, ds)] if t < 10 else cs[(s, top, ds)])
    return fast.assign(full=values)


def human_check_values():
    """Human value and 95% CI of the 53 check statistics (in_check_list) and of 10c, from the saved tables:
    targets 1-7  train_targets.csv, "train+val" rows
    target 8     target8_reversal_rate.csv, set b_target8 (rate)
    target 9     target9_agreement_at_reversal.csv, set target9, f_rev median
    target 10    target10_crossing_time.csv, set target10: crossing_time_median_s (10a) and pause_share_pct (10c)
    Columns: target, stat, topology, dataset, value, ci_lo, ci_hi, in_check_list."""
    tt = pd.read_csv(config.TRAIN_TARGETS)
    tt = tt[tt["role"] == "train+val"]
    parts = [tt[tt["stat"] == CHECKS[t]].assign(target=t) for t in range(1, 8)]
    t8 = pd.read_csv(config.TARGET8)
    parts.append(t8[t8["set"] == "b_target8"].rename(columns={"rate": "value"}).assign(target=8, stat=CHECKS[8]))
    t9 = pd.read_csv(config.TARGET9)
    t9 = t9[(t9["set"] == "target9") & (t9["measure"] == "f_rev") & (t9["stat"] == "median")]
    parts.append(t9.assign(target=9, stat=CHECKS[9]))
    t10 = pd.read_csv(config.TARGET10)
    t10 = t10[t10["set"] == "target10"]
    parts += [t10[t10["stat"] == stat].assign(target=10) for stat in [CHECKS[10], PAUSE_SHARE]]
    out = pd.concat(parts)[["target", "stat", "topology", "dataset", "value", "ci_lo", "ci_hi"]].reset_index(drop=True)
    out["in_check_list"] = (out["stat"] != PAUSE_SHARE) & ((out["target"] != 3) | (out["topology"] == "Hierarchy"))
    return out

"""Training targets: summary statistics of the human runs, per topology x dataset x role
(train, val, and train + val pooled), each with a 95% bootstrap CI over groups (log files).

No new definitions: % fraught is fraught.fraught_percent_hold, committed splits / stalls / breakers
come from h5 (main setting: cutoff 0.9, tau 1 s), Kaplan-Meier from stats.kaplan_meier, and
end-sitting uses the rule of HW 4 part B (forward-hold weights, every player counts, |s| >= 0.9).
CIs: stats.group_bootstrap_mean for means and shares, stats.group_bootstrap (same draws) for KM
quantiles and pooled ratios; 10,000 resamples, seed 573.
"""
import numpy as np
import pandas as pd

from slider import config
from slider import h5
from slider import slider_logs as sl
from slider.fraught import fraught_percent_hold, round_end
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

"""Step 5: the one-time test of C2, APEM (published) and final on the held-out cells, as locked in
docs/step5_protocol.md (commit H_LOCK). Every rule below is the protocol's; the choices it leaves open are marked
"(script choice)".

Modes
    --dry-run   the whole pipeline on a TRAINING stand-in: Hierarchy EL as "Ring EL", Hierarchy CL v09 as
                "Ring CL v09", Hierarchy EL again as "CT-Hierarchy EL"; R = 2, seeds C2 1, APEM 2, final 3.
                No test access. Outputs go to results/step5_dryrun/ (their numbers mean nothing).
    --open      the test, once:
                (a) H_LOCK must be an ancestor of HEAD and docs/step5_protocol.md unchanged since H_LOCK;
                (b) results/test_cache/step5_runs.pkl is loaded if it exists; otherwise ONE call to
                    load_runs("behavior", roles={"test"}, allow_test=True, reason=REASON) with every keep flag on
                    (the filters are applied in (c), with slider_logs.drop_mask), saved to that file;
                (c) scope (EL and CL v09 rounds of the 3 cells), the L5 filters (design.SETS["L5"]: incomplete,
                    aborted, empty, bot), step 0 speeds (design.design_rows: the round's own rows with
                    interface.step_and_pace, else the file's modal speed), the round list with counts per cell;
                (d) C2 9101, APEM 9102, final 9103, R = 200, batches of 20 (job seed [seed, batch, first
                    replicate], as p24_main.py), resumable: each finished batch is saved under
                    results/test_cache/step5_parts/ and skipped on a rerun;
                (e) human statistics + 95% CIs, model statistics, pass / borderline, the success verdict,
                    mechanism metrics, figures.

Checks (27 = 9 targets x 3 cells): 1 success_rate_pct, 2 consensus_time_km_median_s, 3 fraught_pct_mean,
4 committed_pct_mean, 6 end_sitting_pct, 7 relapse_pct, 8 reversal rate, 9 f_rev median, 10a crossing time median.
    model value   targets.check_stats per replicate (the P24 fast path), mean over replicates; se_mc = SD / sqrt(n)
    human value   the full functions (train_targets, reversal_rate_table, agreement_at_reversal_table,
                  crossing_time_table) on the test rounds, group bootstrap 1,000 resamples, seed 573
    target 2      the KM median, or the time limit if there is none, for humans as for the models (P24
                  median_or_limit); the human CI bounds are capped at the time limit (= median_or_limit in every
                  resample; a resample with no median counts as +inf in stats.group_bootstrap) (script choice)
    zero-width human CI (< 1e-9), as P24: success at 0 or 100 -> Clopper-Pearson over the cell's rounds; 10a ->
                  +/- 0.05 s (half the lab clock step); target 2 pinned at the time limit -> +/- 0.05 s (script
                  choice: a time pinned by the interface). Any other zero-width CI is kept and flagged (P24 would
                  stop; here the test is already open, so the script reports instead) (script choice)
    verdict       pass = sim mean inside [ci_lo, ci_hi]; borderline = within 2 se_mc of an edge; "no human value"
                  if the human statistic is undefined (e.g. no reversal), "no value" if no replicate has one
    gap           (sim mean - human) / CI half-width, half-width = (ci_hi - ci_lo) / 2 (as p24_directions.py)
    roles         primary 1-4 (12), secondary 6-7 (6), descriptive 8-10a (9)
    success       C2 passes more of the 12 primary checks than APEM published (tie = no success)

Mechanism metrics (descriptive, step5_mechanism.csv):
    stuck         stuck.episode_table, type fraught / non-fraught; per cell and type: episodes, per round, median
                  duration (= escape time; an episode not broken by the round end lasts to it, as in entry 4).
                  Humans: stuck.boot_cell CIs (1,000, seed 573); models pooled over replicates
    trap hops     traps.trap_hop_table (humans with CIs; models pooled over replicates, no CI)
    Fig 5 (CT)    visuals.trajectories on the CT cell (roles checked by visuals.check_roles), humans and models
                  (all replicates pooled): number of rounds averaged; step5_fig1.png

    python scripts/step5_test.py --dry-run
    python scripts/step5_test.py --open --procs 2        (once)
"""
import argparse
import os
import pickle
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from slider import config, design, sim, stuck, targets, traps
from slider import slider_logs as sl
from slider import visuals as V
from slider.interface import moves, step_and_pace, version_tag
from slider.stats import group_bootstrap_mean

warnings.filterwarnings("ignore", message="Pandas requires version")

H_LOCK = "f547677a7a9b7ac6be262bb69c17dced5c7190fb"       # "Step 5 protocol locked"
PROTOCOL = config.ROOT / "docs" / "step5_protocol.md"
REASON = "Step 5 test opening"
MODELS = {"C2": ("C2", {}), "APEM": ("APEM", V.APEM_PUBLISHED), "final": ("final", {})}
SEEDS = {"open": {"C2": 9101, "APEM": 9102, "final": 9103}, "dry": {"C2": 1, "APEM": 2, "final": 3}}
R = {"open": 200, "dry": 2}
BATCH = 20
N_BOOT = 1000
# cell name -> (topology, dataset) of the rounds it holds
CELLS = {"open": {"Ring EL": ("Ring", "EL"), "Ring CL v09": ("Ring", "CL"), "CT-Hierarchy EL": ("CT-Hierarchy", "EL")},
         "dry": {"Ring EL": ("Hierarchy", "EL"), "Ring CL v09": ("Hierarchy", "CL"),
                 "CT-Hierarchy EL": ("Hierarchy", "EL")}}
PLANNED = {"Ring EL": (51, 17), "Ring CL v09": (39, 13), "CT-Hierarchy EL": (51, 17)}   # p27_test_design.csv
CT = "CT-Hierarchy EL"
TARGETS = [1, 2, 3, 4, 6, 7, 8, 9, 10]
ROLE = {1: "primary", 2: "primary", 3: "primary", 4: "primary", 6: "secondary", 7: "secondary",
        8: "descriptive", 9: "descriptive", 10: "descriptive"}
ZERO = 1e-9
EPISODE_COLUMNS = ["file", "wave", "dataset", "topology", "interface", "type", "start", "duration",
                   "broken", "tries", "kept", "probes", "first_try", "break_minority"]
LAB_CLOCK_HALF = 0.05
L5 = {**sl.PURPOSES["behavior"], **{k: v for k, v in design.SETS["L5"].items() if k.startswith("keep_")}}
KEEP_ALL = {k: True for k in sl.PURPOSES["behavior"]}


def paths(mode):
    """Output and cache locations of a mode."""
    if mode == "open":
        cache = config.RESULTS / "test_cache"
        return {"tables": config.TABLES, "figures": config.FIGURES, "report": config.RESULTS / "step5_report.txt",
                "runs": cache / "step5_runs.pkl", "parts": cache / "step5_parts"}
    root = config.RESULTS / "step5_dryrun"
    return {"tables": root, "figures": root, "report": root / "step5_report.txt", "runs": None,
            "parts": root / "cache"}


# ------------------------------------------------------------------ (a) lock, (b) the one test access

def check_lock():
    """H_LOCK is an ancestor of HEAD and the protocol file (working tree) is unchanged since H_LOCK."""
    git = ["git", "-C", str(config.ROOT)]
    if subprocess.run(git + ["merge-base", "--is-ancestor", H_LOCK, "HEAD"]).returncode != 0:
        raise SystemExit(f"STOP: {H_LOCK} (protocol lock) is not an ancestor of HEAD")
    if subprocess.run(git + ["diff", "--quiet", H_LOCK, "--", str(PROTOCOL)]).returncode != 0:
        raise SystemExit(f"STOP: {PROTOCOL} differs from the locked version ({H_LOCK})")


def load_test_runs(cache):
    """(runs, table) of every test round, all keep flags on. From the cache if it exists; else the ONE
    load_runs call with allow_test=True, saved to the cache."""
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    runs, table = sl.load_runs("behavior", roles={"test"}, allow_test=True, reason=REASON, **KEEP_ALL)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache.with_suffix(".tmp"), "wb") as f:
        pickle.dump((runs, table), f)
    os.replace(cache.with_suffix(".tmp"), cache)
    return runs, table


def load_rounds(mode, p):
    """(runs, table) before filters: the test rounds (open) or the training stand-in (dry run, train + val roles)."""
    if mode == "open":
        return load_test_runs(p["runs"])
    return sl.load_runs("behavior", roles={"train", "val"}, **KEEP_ALL)


# ------------------------------------------------------------------ (c) scope, filters, step 0

def select(runs, table, cells):
    """Scope (EL and CL v09 rounds of the cells' topology x dataset), then the L5 filters.
    Returns (runs, table, counts): counts per cell = rounds in scope, dropped by each filter, kept, groups."""
    actual = set(cells.values())
    in_scope = pd.Series([(top, ds) in actual and (ds != "CL" or version_tag(f) == "v09")
                          for top, ds, f in zip(table["topology"], table["dataset"], table["file"])], index=table.index)
    reasons = {"incomplete": table["end"] == "incomplete", "aborted": (table["T"] == 1) & (table["end"] == "failed"),
               "empty": table["T"] < 2, "bot": table["has_bot"], "whole_second": table["coarse_time"]}
    drop = sl.drop_mask(table, L5)
    counts = []
    for name, (top, ds) in cells.items():
        c = in_scope & (table["topology"] == top) & (table["dataset"] == ds)
        row = {"cell": name, "in_scope": int(c.sum())}
        row.update({f"drop_{k}": int((c & v).sum()) for k, v in reasons.items()})
        row["kept"] = int((c & ~drop).sum())
        row["groups"] = table.loc[c & ~drop, "file"].nunique()
        counts.append(row)
    keep = in_scope & ~drop
    idx = table.index[keep]
    return [runs[i] for i in idx], table.loc[idx].reset_index(drop=True), pd.DataFrame(counts)


def round_list(runs, d, cells):
    """One row per kept round: cell(s), file, wave, dataset, topology, interface, speed and its source."""
    own = [not np.isnan(np.prod(step_and_pace(moves(run)))) for run in runs]
    names = {v: [k for k, w in cells.items() if w == v] for v in cells.values()}
    return pd.DataFrame({"cell": [" + ".join(names[(t, ds)]) for t, ds in zip(d["topology"], d["dataset"])],
                         "file": d["group"], "wave": d["wave"], "dataset": d["dataset"], "topology": d["topology"],
                         "interface": d["interface"], "speed": d["speed"],
                         "speed_source": np.where(own, "round rows", "file modal speed")})


# ------------------------------------------------------------------ statistics of one set of rounds

def point_stats(runs, table, cells):
    """The 27 check values (targets.check_stats) as rows target, stat, topology, dataset, value."""
    actual = sorted(set(cells.values()))
    s = targets.check_stats(runs, table, runs, topologies=sorted({t for t, _ in actual}), cells=actual)
    s = s[(s["group"] == "") & s["target"].isin(TARGETS) & (s["stat"] != targets.PAUSE_SHARE)]
    return s[["target", "stat", "topology", "dataset", "value"]].reset_index(drop=True)


def mechanism_values(runs, table, ifaces, cells):
    """What the mechanism metrics need from one set of rounds: episodes, trap hops per round, CT trajectories."""
    ep = stuck.episode_table(runs, table, ifaces)
    return {"episodes": ep if len(ep) else pd.DataFrame(columns=EPISODE_COLUMNS),
            "hops": traps.trap_hop_runs(runs),
            "traj": V.trajectories(runs, table, [cells[CT]]),
            "rounds": table[["file", "dataset", "topology"]].copy()}


def job(args):
    """One batch of one model: simulate n replicates of design d together, then per replicate: point_stats and
    mechanism_values. Saved to the part file (resumable)."""
    d, model, n, seed, first, cells, part = args
    preset, kw = MODELS[model]
    rounds = design.simulate_rounds(d, preset, R=n, seed=seed, **kw)
    roles = list(d["role"])
    out = {"stats": [], "mech": []}
    for r in range(n):
        runs = design.copy_runs(d, rounds, r, seed)
        table = sim.table(runs, roles)
        out["stats"].append(point_stats(runs, table, cells).assign(replicate=first + r))
        out["mech"].append(mechanism_values(runs, table, d["interface"].to_numpy(), cells))
        for o, _ in rounds:
            o[r] = None
    with open(part.with_suffix(".tmp"), "wb") as f:
        pickle.dump(out, f)
    os.replace(part.with_suffix(".tmp"), part)
    return model, first, n


def simulate_all(mode, d, cells, p, procs):
    """(d) every batch of every model, skipping finished parts; returns {model: [part, ...]}."""
    p["parts"].mkdir(parents=True, exist_ok=True)
    jobs = []
    for model in MODELS:
        for b, first in enumerate(range(0, R[mode], BATCH)):
            part = p["parts"] / f"{model}_b{b:02d}.pkl"
            if not part.exists():
                n = min(BATCH, R[mode] - first)
                jobs.append((d, model, n, [SEEDS[mode][model], b, first], first, cells, part))
    print(f"simulation: {len(jobs)} batches to run, {procs} processes", flush=True)
    start, done = time.time(), 0
    with ProcessPoolExecutor(procs, max_tasks_per_child=1) as pool:
        for future in as_completed([pool.submit(job, j) for j in jobs]):
            model, first, n = future.result()
            done += 1
            per = (time.time() - start) / done
            print(f"{time.strftime('%H:%M:%S')}  {model} replicates {first}-{first + n - 1} done; "
                  f"about {per * (len(jobs) - done) / 60:.0f} min left", flush=True)
    parts = {}
    for model in MODELS:
        parts[model] = []
        for b in range(len(range(0, R[mode], BATCH))):
            with open(p["parts"] / f"{model}_b{b:02d}.pkl", "rb") as f:
                parts[model].append(pickle.load(f))
    return parts


# ------------------------------------------------------------------ (e) human statistics and CIs

def human_stats(runs, table, cells, n_boot=N_BOOT):
    """The 27 human values with 95% group-bootstrap CIs (full functions): target, stat, topology, dataset, value,
    ci_lo, ci_hi, n_runs."""
    actual = sorted(set(cells.values()))
    tops = sorted({t for t, _ in actual})
    tt = targets.train_targets(runs, table, n_boot=n_boot, topologies=tops,
                               roles=[("all", {"train", "val", "test"})])
    rows = [dict(target=t, stat=targets.CHECKS[t], **{k: r[k] for k in ["topology", "dataset", "value", "ci_lo",
                                                                          "ci_hi", "n_runs"]})
            for t in [1, 2, 3, 4, 6, 7] for _, r in tt[tt["stat"] == targets.CHECKS[t]].iterrows()]
    n = {(r["topology"], r["dataset"]): r["n_runs"] for r in rows}
    for _, r in targets.reversal_rate_table(runs, n_boot=n_boot, cells=actual).iterrows():
        rows.append(dict(target=8, stat=targets.CHECKS[8], topology=r["topology"], dataset=r["dataset"],
                         value=r["rate"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"]))
    ag = targets.agreement_at_reversal_table(runs, n_boot=n_boot, cells=actual)[0]
    for _, r in ag[(ag["measure"] == "f_rev") & (ag["stat"] == "median")].iterrows():
        rows.append(dict(target=9, stat=targets.CHECKS[9], topology=r["topology"], dataset=r["dataset"],
                         value=r["value"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"]))
    ct = targets.crossing_time_table(runs, n_boot=n_boot, cells=actual)[0]
    for _, r in ct[ct["stat"] == targets.CHECKS[10]].iterrows():
        rows.append(dict(target=10, stat=targets.CHECKS[10], topology=r["topology"], dataset=r["dataset"],
                         value=r["value"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"]))
    out = pd.DataFrame(rows)
    out["n_runs"] = [n[(t, ds)] for t, ds in zip(out["topology"], out["dataset"])]
    limit = config.TIME_LIMIT["lab"]                     # every test round is a lab round
    t2 = out["target"] == 2
    out.loc[t2, "value"] = out.loc[t2, "value"].fillna(limit)
    out.loc[t2, ["ci_lo", "ci_hi"]] = out.loc[t2, ["ci_lo", "ci_hi"]].clip(upper=limit)
    return out


def zero_width_rule(human):
    """P24's rule for zero-width human CIs (module doc); adds ci_rule."""
    human = human.assign(ci_rule="")
    for i, r in human[(human["ci_hi"] - human["ci_lo"]).abs() < ZERO].iterrows():
        if r["target"] == 1 and r["value"] in (0, 100):
            q = 100 * 0.025 ** (1 / r["n_runs"])
            lo, hi = (q, 100.0) if r["value"] == 100 else (0.0, 100 - q)
            human.loc[i, ["ci_lo", "ci_hi", "ci_rule"]] = [lo, hi, "Clopper-Pearson"]
        elif r["target"] == 10 or (r["target"] == 2 and r["value"] == config.TIME_LIMIT["lab"]):
            human.loc[i, ["ci_lo", "ci_hi", "ci_rule"]] = [r["value"] - LAB_CLOCK_HALF, r["value"] + LAB_CLOCK_HALF,
                                                           "pinned: +/- 0.05 s"]
        else:
            human.loc[i, "ci_rule"] = "zero width, no rule (kept)"
    return human


def by_cell(df, cells):
    """Rows keyed by (topology, dataset) -> one copy per named cell (a dry-run cell may repeat a real one)."""
    parts = [df[(df["topology"] == t) & (df["dataset"] == ds)].assign(cell=name) for name, (t, ds) in cells.items()]
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ (e) checks, scorecard, verdict

def checks(human, reps, cells):
    """One row per model x check: the human value and CI, sim mean, se_mc, verdict, borderline, gap."""
    h = by_cell(human, cells)
    out = []
    for model, rep in reps.items():
        s = by_cell(rep, cells).groupby(["cell", "target"])["value"].agg(n="count", sim_mean="mean", sim_sd="std")
        x = h.join(s, on=["cell", "target"]).assign(model=model, R=rep["replicate"].nunique())
        x["n"] = x["n"].fillna(0).astype(int)
        x["se_mc"] = x["sim_sd"] / np.sqrt(x["n"])
        inside = (x["sim_mean"] >= x["ci_lo"]) & (x["sim_mean"] <= x["ci_hi"])
        x["verdict"] = np.select([x["value"].isna() | x["ci_lo"].isna() | x["ci_hi"].isna(), x["sim_mean"].isna(),
                                  inside], ["no human value", "no value", "pass"], "miss")
        edge = np.minimum(np.abs(x["sim_mean"] - x["ci_lo"]), np.abs(x["sim_mean"] - x["ci_hi"]))
        x["borderline"] = edge <= 2 * x["se_mc"]
        with np.errstate(divide="ignore", invalid="ignore"):
            x["gap"] = (x["sim_mean"] - x["value"]) / ((x["ci_hi"] - x["ci_lo"]) / 2)
        x["role"] = x["target"].map(ROLE)
        out.append(x)
    cols = ["model", "cell", "target", "stat", "role", "R", "n", "sim_mean", "sim_sd", "se_mc", "value", "ci_lo",
            "ci_hi", "ci_rule", "n_runs", "verdict", "borderline", "gap"]
    out = pd.concat(out, ignore_index=True)
    out["_m"], out["_c"] = out["model"].map(list(MODELS).index), out["cell"].map(list(cells).index)
    out = out.sort_values(["_m", "target", "_c"], kind="stable")
    return out[cols].rename(columns={"value": "human", "n_runs": "human_rounds"}).reset_index(drop=True)


def scorecard(ch):
    """Per model: passes by role and of 27, borderline, median |gap|."""
    rows = []
    for model, c in ch.groupby("model", sort=False):
        row = {"model": model}
        for role in ["primary", "secondary", "descriptive"]:
            x = c[c["role"] == role]
            row[f"{role}_pass"], row[f"{role}_n"] = int((x["verdict"] == "pass").sum()), len(x)
        row["pass_of_27"] = int((c["verdict"] == "pass").sum())
        row["borderline"] = int(c["borderline"].sum())
        row["no_value"] = int(c["verdict"].isin(["no value", "no human value"]).sum())
        row["median_abs_gap"] = float(np.nanmedian(np.abs(c["gap"].replace([np.inf, -np.inf], np.nan))))
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(card):
    c2 = int(card.loc[card["model"] == "C2", "primary_pass"].iloc[0])
    apem = int(card.loc[card["model"] == "APEM", "primary_pass"].iloc[0])
    word = "SUCCESS" if c2 > apem else "NO SUCCESS" + (" (tie)" if c2 == apem else "")
    return f"VERDICT: {word}: C2 passes {c2} of 12 primary checks, APEM published {apem} of 12."


# ------------------------------------------------------------------ (e) mechanism metrics

def mechanism(human_mech, model_mech, cells):
    """Long table: metric, cell, who, type, stat, value, ci_lo, ci_hi."""
    rows = []

    def add(metric, cell, who, kind, stat, v, lo=np.nan, hi=np.nan):
        rows.append({"metric": metric, "cell": cell, "who": who, "type": kind, "stat": stat, "value": v,
                     "ci_lo": lo, "ci_hi": hi})

    sources = {"humans": [human_mech], **model_mech}
    for who, parts in sources.items():
        ep = pd.concat([m["episodes"] for m in parts], ignore_index=True)
        rd = pd.concat([m["rounds"] for m in parts], ignore_index=True)
        hops = pd.concat([m["hops"] for m in parts], ignore_index=True)
        for name, (top, ds) in cells.items():
            rounds = rd[(rd["topology"] == top) & (rd["dataset"] == ds)]
            for kind in stuck.TYPES[top]:
                e = ep[(ep["topology"] == top) & (ep["dataset"] == ds) & (ep["type"] == kind)]
                if who == "humans":
                    b = stuck.boot_cell(e, rounds, N_BOOT, config.SEED)
                    for stat in ["episodes", "per_round", "duration_median"]:
                        add("stuck", name, who, kind, stat, *b[stat])
                else:
                    st = stuck.episode_stats(e, len(rounds))
                    for stat in ["episodes", "per_round", "duration_median"]:
                        add("stuck", name, who, kind, stat, st[stat])
            h = hops[(hops["topology"] == top) & (hops["dataset"] == ds)]
            for stat in traps.STATS:
                if who == "humans":
                    lo, hi = group_bootstrap_mean(h[stat], h["file"], N_BOOT, config.SEED)
                    add("trap_hops", name, who, "", stat, h[stat].mean(), lo, hi)
                else:
                    add("trap_hops", name, who, "", stat, h[stat].mean())
            add("trap_hops", name, who, "", "spells_cut", h["n_cut"].sum())
        traj = [m["traj"].get(cells[CT]) for m in parts]
        add("fig5_rounds", CT, who, "", "rounds_averaged", sum(t[2] for t in traj if t))
    return pd.DataFrame(rows)


def pooled_traj(parts, cell):
    """Mean warped positions and Agreement over the qualifying rounds of all parts (None if there are none)."""
    acc = [p["traj"][cell] for p in parts if cell in p["traj"]]
    n = sum(a[2] for a in acc)
    if n == 0:
        return None
    return sum(a[0] for a in acc) / n, sum(a[1] for a in acc) / n, n


# ------------------------------------------------------------------ figures

INK, MUTED, GRID = "#222222", "#6b6b6b", "#e5e5e5"
GROUP_COLORS = {"Apex": "#2a78d6", "Left Branch": "#eb6834", "Right Branch": "#1baf7a"}
MODEL_COLORS = {"C2": "#2a78d6", "APEM": "#eb6834", "final": "#1baf7a"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_trajectories(trajs, path, title):
    """Fig 5 style, CT cell: mean position of Apex, Left Branch, Right Branch over normalized time, one panel per
    source (humans, C2, APEM, final)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    tgrid = np.arange(V.NWARP + 1) / (V.NWARP + 1)
    for ax, (who, tr) in zip(axes, trajs.items()):
        style(ax)
        if tr is None:
            ax.text(0.5, 0.5, "no qualifying round", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        else:
            x, _, n = tr
            for (name, color), g in zip(GROUP_COLORS.items(), V.group_means(x).T):
                ax.plot(tgrid, g, color=color, linewidth=2, label=name)
        ax.set_title(f"{who}" + (f" (n = {tr[2]})" if tr else ""), fontsize=10, color=INK, loc="left")
        ax.set_xlabel("normalized time", fontsize=8, color=MUTED)
    axes[0].set_ylabel("mean position", fontsize=8, color=MUTED)
    axes[0].set_ylim(-1.05, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8, ncol=3)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_gaps(ch, path, title):
    """Gap (sim mean - human) in CI half-widths per check, one row per check, one dot per model; band = the CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(dict.fromkeys(f"{t} {c}" for t, c in zip(ch["target"].map(lambda t: "10a" if t == 10 else str(t)),
                                                            ch["cell"])))
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(labels) + 1.4))
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.axvspan(-1, 1, color="#f1f1f1", zorder=0)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    y = {lab: i for i, lab in enumerate(labels)}
    offsets = {m: o for m, o in zip(MODELS, [-0.2, 0, 0.2])}
    for model, c in ch.groupby("model", sort=False):
        lab = [f"{'10a' if t == 10 else t} {cell}" for t, cell in zip(c["target"], c["cell"])]
        g = np.clip(c["gap"].to_numpy(float), -6, 6)
        ax.scatter(g, [y[l] + offsets[model] for l in lab], s=22, color=MODEL_COLORS[model], label=model, zorder=3,
                   edgecolors="white", linewidths=0.8)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(-6.3, 6.3)
    ax.set_xlabel("gap = (sim mean - human) / CI half-width (clipped at +/- 6; shaded = +/- 1 half-width)",
                  fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1), ncol=3)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main

def run(mode, procs):
    p = paths(mode)
    cells = CELLS[mode]
    if mode == "open":
        check_lock()
    p["tables"].mkdir(parents=True, exist_ok=True)
    tag = "step5" if mode == "open" else "step5_DRYRUN"
    start = time.time()

    # (b), (c)
    runs, table = load_rounds(mode, p)
    runs, table, counts = select(runs, table, cells)
    d = design.design_rows(runs, table)                                   # step 0: speeds, before simulating
    rounds = round_list(runs, d, cells)
    rounds.to_csv(p["tables"] / f"{tag}_rounds.csv", index=False)
    if mode == "open":
        counts["planned"] = counts["cell"].map(lambda c: PLANNED[c][0])
        counts["planned_groups"] = counts["cell"].map(lambda c: PLANNED[c][1])
    print(counts.to_string(index=False), flush=True)

    # (d)
    parts = simulate_all(mode, d, cells, p, procs)
    reps = {m: pd.concat([s for part in parts[m] for s in part["stats"]], ignore_index=True) for m in MODELS}
    model_mech = {m: [x for part in parts[m] for x in part["mech"]] for m in MODELS}

    # (e)
    human = zero_width_rule(human_stats(runs, table, cells))
    ch = checks(human, reps, cells)
    card = scorecard(ch)
    line = verdict(card)
    if mode == "dry":
        line = "[DRY RUN, training stand-in: numbers mean nothing] " + line
    ch.to_csv(p["tables"] / f"{tag}_checks.csv", index=False)
    card.to_csv(p["tables"] / f"{tag}_scorecard.csv", index=False)
    V.check_roles(cells[CT][0])
    human_mech = mechanism_values(runs, table, d["interface"].to_numpy(), cells)
    mech = mechanism(human_mech, model_mech, cells)
    mech.to_csv(p["tables"] / f"{tag}_mechanism.csv", index=False)
    trajs = {"humans": pooled_traj([human_mech], cells[CT]), **{m: pooled_traj(model_mech[m], cells[CT])
                                                                for m in MODELS}}
    fig_trajectories(trajs, p["figures"] / f"{tag}_fig1.png", f"{CT}: averaged trajectories (Fig 5 style)"
                     + (" - DRY RUN" if mode == "dry" else ""))
    fig_gaps(ch, p["figures"] / f"{tag}_fig2.png", "Step 5 checks: gap in human CI half-widths"
             + (" - DRY RUN" if mode == "dry" else ""))

    lines = [line, "",
             f"protocol: docs/step5_protocol.md, locked at {H_LOCK}",
             f"mode: {mode}; R = {R[mode]}; seeds " + ", ".join(f"{m} {s}" for m, s in SEEDS[mode].items()),
             "", "rounds per cell (scope EL + CL v09; L5 filters):", counts.to_string(index=False),
             f"speeds (step 0): {int((rounds['speed_source'] == 'round rows').sum())} from the round's rows, "
             f"{int((rounds['speed_source'] != 'round rows').sum())} from the file's modal speed",
             "", "scorecard:", card.to_string(index=False),
             "", "human CIs replaced or flagged by the zero-width rule: "
             + ("; ".join(f"{r.stat} {r.topology} {r.dataset}: {r.ci_rule}" for r in human[human["ci_rule"] != ""]
                          .itertuples()) or "none"),
             "", "checks (* = borderline):"]
    for r in ch.itertuples():
        lines.append(f"  {r.model:<6} {('10a' if r.target == 10 else str(r.target)):>3} {r.role:<11} {r.cell:<16} "
                     f"{r.verdict}{'*' if r.borderline else ''}  sim {r.sim_mean:.4g}  human {r.human:.4g} "
                     f"[{r.ci_lo:.4g}, {r.ci_hi:.4g}]")
    lines += ["", f"wall time: {(time.time() - start) / 60:.1f} min"]
    p["report"].write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:lines.index("scorecard:") + len(card) + 2]))
    print(f"wrote {p['report']}")


def main():
    ap = argparse.ArgumentParser(description="Step 5 test (docs/step5_protocol.md)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="training stand-in, R = 2, no test access")
    g.add_argument("--open", action="store_true", help="the one-time test")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    run("open" if a.open else "dry", a.procs)


if __name__ == "__main__":
    main()

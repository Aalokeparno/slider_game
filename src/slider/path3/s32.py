"""Path 3 run S3.2 analysis (scripts/path3_s3_2.py): stage-1 tables, stage-2 paired bootstrap, decision rule, figures.

Times: t_consensus with failures (no consensus by 300 s) as inf; quantiles use the inverted CDF (an actual run's value),
as in robust.py (T5).
Stage-2 statistics per strategy, on a set of graphs: pooled_median, pooled_p95 (over all runs), mean_p95 (mean of the
per-graph p95), worst_decile_p95 (mean per-graph p95 over the fixed T5 worst-decile set: the 60 graphs with the
highest per-graph p95 averaged over the three T5 variants), failures (runs without consensus).
Bootstrap (paired, two-level, as T5): resample graphs with replacement, then run indices within each drawn graph; the
same draws for a strategy and S0b''' (the T5 "S0b''' 0.75" runs on the same seeds; for S4, 100 runs per graph, its first
100). 1,000 resamples, seed 573; 95% CIs = 2.5 and 97.5 percentiles of the resampled differences.
Subsets: all 597 graph-train graphs; dense = top tercile of edge count (graphs.tercile: a value equal to the cut goes
to the lower bin); out of sample = the 574 graphs not in the screening set.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slider import config
from slider.path3 import graphs, report, robust

q = robust.q
STATS = ["pooled_median", "pooled_p95", "mean_p95", "worst_decile_p95", "failures"]


def times(t):
    return np.where(np.isnan(t), np.inf, t)


# ------------------------------------------------------------------ stage 1

def stage1_table(runs):
    """One row per variant x graph, plus graph = "pooled" (all runs of the variant). Default variants also get
    "(first 100)" rows (runs 0-99, the seeds of the sweeps) so a sweep setting compares with its default."""
    runs = runs.assign(minutes=np.where(runs["t_consensus"].isna(), 5.0, runs["t_consensus"] / 60),
                       wave_success=runs["success_300"] & (runs["n_yields"] == 0))
    first = runs[(runs["block"] == "S32_screen") & (runs["run"] < 100)]
    first = first.assign(variant=first["variant"] + " (first 100)")
    runs = pd.concat([runs, first], ignore_index=True)
    parts = [(k, d) for k, d in runs.groupby(["variant", "graph"], sort=False)]
    parts += [((v, "pooled"), d) for v, d in runs.groupby("variant", sort=False)]
    rows = []
    for (variant, graph), d in parts:
        t = d["t_consensus"].to_numpy()
        row = {"variant": variant, "graph": graph, "runs": len(d), "success_180": d["success_180"].mean(),
               "success_300": d["success_300"].mean(), "median_t": report.median_time(t), "p95_t": report.p95_time(t),
               "breakups_per_min": d["breakups"].sum() / d["minutes"].sum(), "yields_per_run": d["n_yields"].mean(),
               "yield_majority_share": d["yields_majority"].sum() / max(d["yields_sided"].sum(), 1),
               "yield_undone_share": d["yields_undone"].sum() / max(d["n_yields"].sum(), 1),
               "frac_time_fraught": d["frac_time_fraught"].mean()}
        for c in ["consensus", "fixed trap", "periodic", "irregular"]:
            row[c] = (d["outcome"] == c).mean()
        if d["strategy"].iloc[0] == "S3":
            row.update(wave_success=d["wave_success"].mean(), seeds_per_run=d["n_seeds"].mean(),
                       median_t_all_reached=float(np.nanmedian(d["t_all_reached"])) if d["t_all_reached"].notna().any()
                       else np.nan, reached_all_share=d["t_all_reached"].notna().mean(),
                       reversals_per_run=d["n_reversals"].mean(),
                       wrong_wall_per_run=d["n_wrong_wall"].mean(), fidgets_per_run=d["n_fidgets"].mean())
        if d["strategy"].iloc[0] in ("S6", "S7"):
            row.update(decisions_per_min=d["decisions_per_min"].mean(), mean_W=d["mean_W"].mean())
        if d["strategy"].iloc[0] == "S7":
            row.update(slow_share=d["slow_share"].mean(), switches_per_run=d["n_switches"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ stage 2

def t5_reference(t):
    """T (variants x graphs x 200) of the three T5 variants, in the order of t["graph"]."""
    r = pd.concat([pd.read_csv(p) for p in sorted((config.PATH3 / "blocks" / "T5").glob("chunk_*.csv"))])
    variants = ["S0b''", "S0b''' 0.75", "S0b''' 1"]
    x = r.pivot_table(index=["variant", "graph"], columns="run", values="t_consensus", aggfunc="first", dropna=False)
    x = x.reindex(pd.MultiIndex.from_product([variants, t["graph"]]))
    return times(x.to_numpy(float).reshape(3, len(t), 200))


def array(runs, variant, graph_order, n):
    x = runs[runs["variant"] == variant].pivot_table(index="graph", columns="run", values="t_consensus",
                                                      aggfunc="first", dropna=False)
    x = x.reindex(index=graph_order, columns=range(n))
    assert x.shape == (len(graph_order), n)
    return times(x.to_numpy(float))


def statistics(T, worst):
    """STATS for each variant of T (variants x graphs x runs); worst = columns in the fixed worst-decile set."""
    p95 = q(T, 0.95, axis=2)
    flat = T.reshape(T.shape[0], -1)
    wd = p95[:, worst].mean(axis=1) if len(worst) else np.full(T.shape[0], np.nan)
    return np.column_stack([q(flat, 0.5, axis=1), q(flat, 0.95, axis=1), p95.mean(axis=1), wd,
                            np.isinf(flat).sum(axis=1)])


def compare(T, in_worst, names, label, n_boot=1000, seed=config.SEED):
    """T: variants x graphs x runs, T[0] = S0b'''. Point values with CIs per variant and differences vs T[0]."""
    G, R = T.shape[1], T.shape[2]
    point = statistics(T, np.flatnonzero(in_worst))
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, T.shape[0], len(STATS)))
    for b in range(n_boot):
        g = rng.integers(0, G, G)
        r = rng.integers(0, R, (G, R))
        boot[b] = statistics(T[:, g[:, None], r], np.flatnonzero(in_worst[g]))
    rows = []
    for v, name in enumerate(names):
        row = {"graphs": label, "n_graphs": G, "runs_per_graph": R, "variant": name}
        for s, stat in enumerate(STATS):
            row[stat], (row[stat + "_lo"], row[stat + "_hi"]) = point[v, s], robust.ci(boot[:, v, s])
        rows.append(row)
    for v, name in enumerate(names[1:], start=1):
        row = {"graphs": label, "n_graphs": G, "runs_per_graph": R, "variant": f"{name} - {names[0]}"}
        for s, stat in enumerate(STATS):
            row[stat] = point[v, s] - point[0, s]
            row[stat + "_lo"], row[stat + "_hi"] = robust.ci(boot[:, v, s] - boot[:, 0, s])
        rows.append(row)
    return pd.DataFrame(rows)


def per_graph(runs, t, ref):
    """One row per strategy x graph: median, p95, failures, yields/run, breakups/min, edges and the median / p95
    differences vs S0b''' on the same seeds."""
    rows = []
    for variant, d in runs.groupby("variant"):
        n = int(d["run"].max()) + 1
        T = array(runs, variant, t["graph"], n)
        for g, gname in enumerate(t["graph"]):
            x = d[d["graph"] == gname]
            minutes = np.where(x["t_consensus"].isna(), 5.0, x["t_consensus"] / 60).sum()
            rows.append({"variant": variant, "graph": gname, "edges": t["n_edges"].iloc[g],
                         "n_fraught": t["n_fraught"].iloc[g], "screening": bool(t["screening"].iloc[g]), "runs": n,
                         "median_t": q(T[g], 0.5), "p95_t": q(T[g], 0.95), "failures": int(np.isinf(T[g]).sum()),
                         "d_median": q(T[g], 0.5) - q(ref[g, :n], 0.5), "d_p95": q(T[g], 0.95) - q(ref[g, :n], 0.95),
                         "yields_per_run": x["n_yields"].mean(), "breakups_per_min": x["breakups"].sum() / minutes})
    return pd.DataFrame(rows)


def stage2(runs, t):
    """Summary table (all / dense / out of sample, both run-count groups) and per-graph table."""
    T5 = t5_reference(t)
    ref = T5[1]                                                       # S0b''' eps 0.75
    worst = np.zeros(len(t), bool)
    worst[robust.worst_set(T5)] = True
    dense = (graphs.tercile(t["n_edges"]) == 2).to_numpy()
    oos = (~t["screening"]).to_numpy()
    groups = [(["S3a+", "S3b+", "S6", "S7"], 200), (["S4-old", "S4-new"], 100)]
    parts = []
    for names, n in groups:
        T = np.stack([ref[:, :n]] + [array(runs, v, t["graph"], n) for v in names])
        for label, mask in [("all train", np.ones(len(t), bool)), ("dense (top edge tercile)", dense),
                            ("not in screening set", oos)]:
            parts.append(compare(T[:, mask], worst[mask], ["S0b'''"] + names, label))
    summ = pd.concat(parts, ignore_index=True)
    return summ, per_graph(runs, t, ref), worst, dense


def decide(summ):
    """The pre-stated rule, step by step, for S3a+, S3b+, S6, S7 (defaults) vs S0b''' on all 597 graphs."""
    summ = summ[summ["runs_per_graph"] == 200]
    s = summ[(summ["graphs"] == "all train")].set_index("variant")
    ref = s.loc["S0b'''"]
    rows = []
    for name in ["S3a+", "S3b+", "S6", "S7"]:
        d, v = s.loc[f"{name} - S0b'''"], s.loc[name]
        reliable = v["failures"] <= ref["failures"]
        med = d["pooled_median_hi"] < 0
        p95 = d["pooled_p95_hi"] < 0
        dd = summ[(summ["graphs"] == "dense (top edge tercile)") & (summ["variant"] == f"{name} - S0b'''")].iloc[0]
        def ci(r, c):  # an inf point with no CI: some graphs fail in more than 5 % of runs
            if np.isinf(r[c]) and (np.isnan(r[c + "_lo"]) or np.isnan(r[c + "_hi"])):
                return "inf (no CI)"
            return f"{r[c]:+.1f} [{r[c + '_lo']:+.1f}, {r[c + '_hi']:+.1f}]"

        rows.append({"strategy": name,
                     "1 failures (S0b''' {:.0f})".format(ref["failures"]): f"{v['failures']:.0f}; diff CI "
                     f"[{d['failures_lo']:.0f}, {d['failures_hi']:.0f}] -> {'pass' if reliable else 'FAIL'}",
                     "2a pooled median diff (s)": f"{d['pooled_median']:+.1f} [{d['pooled_median_lo']:+.1f}, "
                     f"{d['pooled_median_hi']:+.1f}] -> {'below 0' if med else 'not below 0'}",
                     "2b pooled p95 diff (s)": f"{d['pooled_p95']:+.1f} [{d['pooled_p95_lo']:+.1f}, "
                     f"{d['pooled_p95_hi']:+.1f}] -> {'below 0' if p95 else 'not below 0'}",
                     "3 dense median / p95 diff (s)": ci(dd, "pooled_median") + " / " + ci(dd, "pooled_p95"),
                     "3 worst-decile p95 diff (s)": ci(d, "worst_decile_p95"),
                     "verdict": ("PROMOTE" if reliable and med and p95 else "not promoted")
                     + (" (reported separately: needs an assigned seed role)" if name == "S3a+" else "")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures

def _pooled(table, variants):
    return table[(table["graph"] == "pooled") & table["variant"].isin(variants)].set_index("variant")


def figure_fidget(table, path):
    """(i) S3a+ / S3b+ success_300, wave success and median vs the fidget rate mu (stage 1, 100 runs per graph)."""
    mus = [0.0, 0.01, 0.03, 0.1, 0.3]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=report.SURFACE)
    for base, color in [("S3a+", report.BLUE), ("S3b+", report.ORANGE)]:
        labels = [f"{base} (first 100)"] + [f"{base} | fidget={m:g}" for m in mus[1:]]
        p = _pooled(table, labels).reindex(labels)
        for ax, col in zip(axes, ["success_300", "wave_success", "median_t"]):
            y = p[col].to_numpy()
            ax.plot(range(len(mus)), y, color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=report.SURFACE, label=base)
    for ax, title in zip(axes, ["consensus by 300 s", "wave success (consensus, no yield)",
                                "median time to consensus (s)"]):
        report.style(ax)
        ax.set_xticks(range(len(mus)), [f"{m:g}" for m in mus])
        ax.set_xlabel("fidget rate μ (steps per s, waiting players)", fontsize=8, color=report.INK_2)
        ax.set_title(title, fontsize=9, color=report.INK, loc="left")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Stage 1: 23 screening graphs pooled, start A, 100 runs per graph per point "
             "(μ = 0: the first 100 default runs).", fontsize=8, color=report.INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=report.SURFACE)


def figure_q(table, path):
    """(ii) S7 median and p95 vs the slow share q (stage 1)."""
    qs = [0.0, 1 / 7, 2 / 7, 3 / 7, 1.0]
    labels = [f"S7 | q={v:g}" if v != 2 / 7 else "S7 (first 100)" for v in qs]
    p = _pooled(table, labels).reindex(labels)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=report.SURFACE)
    for col, color, name in [("median_t", report.BLUE, "median"), ("p95_t", report.ORANGE, "95th percentile")]:
        ax.plot(range(len(qs)), p[col].to_numpy(), color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=report.SURFACE, label=name)
    report.style(ax)
    ax.set_xticks(range(len(qs)), ["0", "1/7", "2/7\n(default)", "3/7", "1\n(= S6)"])
    ax.set_xlabel("q = chance a player starts slow", fontsize=8, color=report.INK_2)
    ax.set_ylabel("time to consensus (s)", fontsize=8, color=report.INK_2)
    ax.set_ylim(bottom=0)
    ax.set_title("S7 fast/slow mix: time to consensus vs q", fontsize=9, color=report.INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Stage 1: 23 screening graphs pooled, start A, 100 runs per graph per point.",
             fontsize=8, color=report.INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=report.SURFACE)


def figure_wmax(table, path):
    """(iii) S6 breakups per minute vs W_max (stage 1)."""
    ws = [4.0, 8.0, 16.0]
    labels = ["S6 | W_max=4", "S6 (first 100)", "S6 | W_max=16"]
    p = _pooled(table, labels).reindex(labels)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=report.SURFACE)
    ax.plot(range(3), p["breakups_per_min"].to_numpy(), color=report.BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=report.SURFACE)
    for x, y in enumerate(p["breakups_per_min"].to_numpy()):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8,
                    color=report.INK)
    ref = _pooled(table, ["S0b''' (first 100)"])["breakups_per_min"].iloc[0]
    ax.axhline(ref, color=report.INK_2, linewidth=1, linestyle="--")
    ax.annotate(f"S0b''' {ref:.2f}", (2, ref), textcoords="offset points", xytext=(0, 5), ha="right", fontsize=8,
                color=report.INK_2)
    report.style(ax)
    ax.set_xticks(range(3), ["4", "8 (default)", "16"])
    ax.set_xlabel("W_max (s)", fontsize=8, color=report.INK_2)
    ax.set_ylabel("breakups per minute played", fontsize=8, color=report.INK_2)
    ax.set_ylim(0, 1.2 * max(ref, p["breakups_per_min"].max()))
    ax.set_title("S6 backoff: breakups per minute vs W_max", fontsize=9, color=report.INK, loc="left")
    fig.text(0.01, 0.01, "Stage 1: 23 screening graphs pooled, start A, 100 runs per graph per point.",
             fontsize=8, color=report.INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=report.SURFACE)


def figure_edges(pg, path):
    """(iv) per-graph median difference vs S0b''' against edge count, one panel per strategy (stage 2)."""
    names = ["S3a+", "S3b+", "S6", "S7", "S4-old", "S4-new"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, facecolor=report.SURFACE)
    rng = np.random.default_rng(0)
    finite = pg["d_median"].replace([np.inf, -np.inf], np.nan)
    lo, hi = np.nanmin(finite), np.nanmax(finite)
    for ax, name in zip(axes.ravel(), names):
        d = pg[pg["variant"] == name]
        ax.scatter(d["edges"] + rng.uniform(-0.25, 0.25, len(d)), d["d_median"].clip(lo, hi), s=10,
                   color=report.BLUE, alpha=0.6, edgecolor="none")
        m = d.groupby("edges")["d_median"].median()
        ax.plot(m.index, m.to_numpy(), color=report.INK, linewidth=2, label="median over graphs")
        ax.axhline(0, color=report.INK_2, linewidth=1)
        report.style(ax)
        ax.set_title(f"{name} − S0b'''", fontsize=9, color=report.INK, loc="left")
        ax.set_xlabel("number of edges", fontsize=8, color=report.INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("per-graph Δ median time (s)", fontsize=8, color=report.INK_2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.005, "Stage 2: 597 graph-train graphs, start A, T5 seeds; S3/S6/S7 200 runs per graph, S4 100. "
             "Below 0 = faster than S0b''' on that graph.", fontsize=8, color=report.INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=report.SURFACE)

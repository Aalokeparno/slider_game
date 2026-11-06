"""Path 3 summaries of results/path3/runs.csv: rankings, efficiency gap, figures.

Median time to consensus: over all runs of a cell, a run without consensus counting as infinite
(so the median is inf when half or more of the runs fail). Same rule for the human rounds.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAMES = {"S0a": "S0a greedy", "S0b": "S0b'' noise + hold", "S2": "S2 v2 stuck timer", "S1p": "S1' v2 averaging",
         "S2_v1": "S2 v1", "S1p_v1": "S1' v1"}
ORDER = ["S0a", "S0b", "S2", "S1p", "S2_v1", "S1p_v1"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def median_time(t):
    """Median of t_consensus with NaN (no consensus) as inf."""
    return float(np.median(np.where(np.isnan(t), np.inf, t)))


def summary(runs, by):
    """Per group: runs, success_180, success_300, median_t, mean frac_time_fraught, share of each outcome class."""
    rows = []
    for key, d in runs.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row.update(runs=len(d), success_180=d["success_180"].mean(), success_300=d["success_300"].mean(),
                   median_t=median_time(d["t_consensus"].to_numpy()), frac_time_fraught=d["frac_time_fraught"].mean(),
                   yields_per_run=d["n_yields"].mean(), collisions_per_run=d["n_collisions"].mean())
        for c in ["consensus", "fixed trap", "periodic", "irregular"]:
            row[c] = (d["outcome"] == c).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def sanity(fix_tests):
    """Strategy -> "passed" / "failed sanity test", from the Hierarchy tests: S0a test 1 and S2_v1 / S1p_v1
    (v1 round, pytest), S0b / S2 / S1p from fix_tests (results/path3/fix_tests.csv)."""
    out = {"S0a": "passed", "S2_v1": "failed sanity test", "S1p_v1": "failed sanity test"}
    for s, ok in zip(fix_tests["strategy"], fix_tests["pass"]):
        out[s] = "passed" if ok else "failed sanity test"
    return out


def rankings(screen, flags):
    """Per (start, graph) and pooled over the screening set ("pooled", all runs together): summary() per strategy,
    ranked by success_300 (high first), then median_t (low first); sanity = flags[strategy]."""
    per_graph = summary(screen, ["start", "graph", "strategy"])
    pooled = summary(screen, ["start", "strategy"]).assign(graph="pooled")
    t = pd.concat([pooled, per_graph], ignore_index=True)
    t = t.sort_values(["start", "graph", "success_300", "median_t"], ascending=[True, True, False, True])
    t["rank"] = t.groupby(["start", "graph"]).cumcount() + 1
    t["sanity"] = t["strategy"].map(flags)
    first = ["start", "graph", "rank", "strategy", "sanity"]
    return t[first + [c for c in t.columns if c not in first]].reset_index(drop=True)


def efficiency_gap(screen_A, human):
    """Human median consensus time / strategy median time, start A, on the human networks with lab v09
    G-train rounds (Hierarchy, Full). > 1: the strategy is faster than the humans."""
    rows = []
    for topology, h in human.groupby("topology"):
        h_med = float(np.median(h["t_consensus"]))
        name = {"CT-Hierarchy": "CT"}.get(topology, topology)
        for s, d in screen_A[screen_A["graph"] == name].groupby("strategy"):
            m = median_time(d["t_consensus"].to_numpy())
            rows.append({"graph": name, "human_rounds": len(h), "human_median_t": h_med, "strategy": s,
                         "strategy_median_t": m, "gap": h_med / m})
    return pd.DataFrame(rows)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def sweep_figure(sweeps, path):
    """Median time vs each swept parameter (Hierarchy), start A and B; failures-as-inf medians drawn at 300 s
    with an open marker."""
    panels = [("S0b", "eps", "ε"), ("S0b", "H", "H (s)"), ("S2", "T", "T (s)"), ("S2", "p", "p (1/s)"),
              ("S2", "H", "H (s)"), ("S2", "B", "B (ticks)"), ("S1p", "kappa", "κ"), ("S1p", "Q", "Q (ticks)")]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), facecolor=SURFACE)
    for ax, (s, param, label) in zip(axes.ravel(), panels):
        style(ax)
        d = sweeps[sweeps["strategy"] == s].copy()
        d["value"] = [json.loads(p)[param] for p in d["params"]]
        for start, color in [("A", BLUE), ("B", ORANGE)]:
            x = d[d["start"] == start]
            m = x.groupby("value")["t_consensus"].apply(lambda t: median_time(t.to_numpy()))
            y = m.clip(upper=300)
            ax.plot(y.index, y.to_numpy(), color=color, linewidth=2, label=f"start {start}")
            ok = np.isfinite(m.to_numpy())
            ax.scatter(y.index[ok], y.to_numpy()[ok], color=color, s=36, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.scatter(y.index[~ok], y.to_numpy()[~ok], facecolor=SURFACE, edgecolor=color, s=36, zorder=3,
                       linewidth=1.5)
        ax.set_title(f"{NAMES[s]}: {label}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel(label, fontsize=8, color=INK_2)
        ax.set_ylabel("median time to consensus (s)", fontsize=8, color=INK_2)
        ax.set_ylim(0, 310)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.005, "Hierarchy, 100 runs per point. Open marker: half or more of the runs fail "
             "(median > 300 s, drawn at 300 s).", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig


def heatmap_figure(screen_A, order, path):
    """Strategy x screening graph, median time to consensus (start A); '>300' where the median is inf."""
    strategies = ORDER
    M = np.array([[median_time(screen_A[(screen_A["strategy"] == s) & (screen_A["graph"] == g)]["t_consensus"]
                               .to_numpy()) for g in order] for s in strategies])
    fig, ax = plt.subplots(figsize=(13, 4), facecolor=SURFACE)
    shown = np.where(np.isfinite(M), M, np.nan)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#c3c2b7")                                  # failing cells: neutral gray
    im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=np.nanmax(shown) if np.isfinite(shown).any() else 1,
                   aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            dark = np.isfinite(v) and v > 0.6 * np.nanmax(shown)
            ax.text(j, i, ">300" if not np.isfinite(v) else f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if dark else INK)
    ax.set_xticks(range(len(order)), order, rotation=60, fontsize=8, color=INK_2)
    ax.set_yticks(range(len(strategies)), [NAMES[s] for s in strategies], fontsize=8, color=INK_2)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("median time to consensus (s)", fontsize=8, color=INK_2)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    ax.set_title("Start A (all sliders at 0), 200 runs per cell; gray '>300': half or more of the runs fail",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig


# ------------------------------------------------------------------ tuning run T1

def p95_time(t):
    """95th percentile of t_consensus with NaN as inf (inverted CDF: an actual run's value, inf if > 5% fail)."""
    return float(np.quantile(np.where(np.isnan(t), np.inf, t), 0.95, method="inverted_cdf"))


def tune_table(runs):
    """One row per setting x graph of the T1 runs (block, strategy, H, t0, eps, graph): runs, success_180,
    success_300, median_t, p95_t, yields_per_run, collisions_per_run (overlapping yield pairs), moving_per_tick,
    and on Full cross_toward_share = greedy crossings toward the global majority / (toward + away), pooled over
    the runs (ties, sum of s = 0, left out)."""
    rows = []
    for (block, strategy, params, graph), d in runs.groupby(["block", "strategy", "params", "graph"], sort=False):
        p = json.loads(params)
        t = d["t_consensus"].to_numpy()
        toward, away = d["cross_toward"].sum(), d["cross_away"].sum()
        rows.append({"block": block, "strategy": strategy, "H": p.get("H", np.nan), "t0": p.get("t0", np.nan),
                     "eps": p.get("eps", np.nan), "graph": graph, "runs": len(d),
                     "success_180": d["success_180"].mean(), "success_300": d["success_300"].mean(),
                     "median_t": median_time(t), "p95_t": p95_time(t), "yields_per_run": d["n_yields"].mean(),
                     "collisions_per_run": d["n_collisions"].mean(), "moving_per_tick": d["moving_per_tick"].mean(),
                     "cross_toward_share": toward / (toward + away) if graph == "Full" else np.nan})
    return pd.DataFrame(rows)


def tune_figure(tune, path):
    """Median (solid) and 95th percentile (dashed) of the time to consensus vs each T1 parameter, Hierarchy and
    Full side by side. Rows: S0a_hold vs H, S0b'' vs t0, S0b'' vs eps (one line per H). Gray dotted: the
    reference (S0a, S0b'' at defaults). Values above 300 s (failures counted as inf) are drawn at 330 with an
    open marker."""
    TOP, HIGH = 300, 330
    colors = {2.0: BLUE, 4.0: ORANGE, 8.0: "#1baf7a"}
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), facecolor=SURFACE, sharey=True)

    def draw(ax, x, med, p95, color, label):
        for y, dash, kind in [(med, "-", "median"), (p95, "--", "95th pct")]:
            y = np.asarray(y, dtype=float)
            ok = y <= TOP
            yy = np.where(ok, y, HIGH)
            ax.plot(x, yy, dash, color=color, linewidth=2, label=f"{label}, {kind}" if label else kind)
            ax.scatter(np.asarray(x)[ok], yy[ok], color=color, s=36, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.scatter(np.asarray(x)[~ok], yy[~ok], facecolor=SURFACE, edgecolor=color, s=36, zorder=3, linewidth=1.5)

    def reference(ax, d):
        for y, dash in [(d["median_t"], ":"), (d["p95_t"], ":")]:
            ax.axhline(min(y, HIGH), color=INK_2, linestyle=dash, linewidth=1.2)

    for col, graph in enumerate(["Hierarchy", "Full"]):
        g = tune[tune["graph"] == graph]
        ref = g[g["block"] == "T1ref"].set_index("strategy")
        ax = axes[0, col]
        a = g[g["block"] == "T1a"].sort_values("H")
        draw(ax, a["H"], a["median_t"], a["p95_t"], BLUE, "")
        reference(ax, ref.loc["S0a"])
        ax.set_xlabel("H (s)", fontsize=8, color=INK_2)
        ax.set_title(f"{graph}: S0a_hold vs H (dotted: S0a)", fontsize=9, color=INK, loc="left")
        ax = axes[1, col]
        b = g[g["block"] == "T1b"].sort_values("t0")
        draw(ax, b["t0"], b["median_t"], b["p95_t"], BLUE, "")
        reference(ax, ref.loc["S0b"])
        ax.set_xlabel("t0 (s)", fontsize=8, color=INK_2)
        ax.set_title(f"{graph}: S0b'' (eps 0.5, H 4) vs t0 (dotted: S0b'' default)", fontsize=9, color=INK, loc="left")
        ax = axes[2, col]
        c = g[g["block"] == "T1c"]
        for H, d in c.groupby("H"):
            d = d.sort_values("eps")
            draw(ax, d["eps"], d["median_t"], d["p95_t"], colors[H], f"H = {H:g} s")
        ax.set_xlabel("ε", fontsize=8, color=INK_2)
        ax.set_title(f"{graph}: S0b'' vs ε, one line per H", fontsize=9, color=INK, loc="left")
    for ax in axes.ravel():
        style(ax)
        ax.set_ylim(0, 345)
        ax.set_yticks([0, 50, 100, 150, 200, 250, 300, HIGH], ["0", "50", "100", "150", "200", "250", "300", ">300"])
        ax.set_ylabel("time to consensus (s)", fontsize=8, color=INK_2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[2, 0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.text(0.01, 0.005, "Start A, 200 runs per point, common seeds (screen_A). Solid: median; dashed: 95th "
             "percentile; open marker at '>300': above 300 s (failures counted as infinite).", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig

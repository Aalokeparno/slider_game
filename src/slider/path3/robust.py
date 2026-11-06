"""Path 3 run T5 analysis: S0b'' vs S0b''' (eps 0.75, 1) on every graph-train graph.

Times: t_consensus with failures as inf. Per-graph p95 and pooled quantiles use the inverted CDF (an actual run's
value). Statistics per variant, on a set of graphs:
    pooled_median, pooled_p95   over all runs of the set
    mean_p95                    mean of the per-graph p95
    worst_decile_p95            mean per-graph p95 over the worst 10% of graphs (round(0.1 x graphs)), chosen
                                by the per-graph p95 averaged over the three variants (one set for all variants)
    max_p95 (+ graph), graphs_with_failure, failure_rate
Bootstrap (paired, two-level): resample graphs with replacement, then the 200 run indices within each drawn graph
with replacement; the same draws for all variants. The worst-decile set stays fixed (chosen once from the data):
in each resample, worst_decile_p95 is the mean per-graph p95 over the drawn graphs that belong to the set (with
their multiplicity). 1,000 resamples, seed 573; 95% CIs are the 2.5 and 97.5 percentiles.
(First version, replaced 2026-11-06 before reporting: the set was re-chosen in every resample; that biased the
resampled values upward, so the point estimate fell at or below its own CI.)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slider import config
from slider.path3 import report

STATS = ["pooled_median", "pooled_p95", "mean_p95", "worst_decile_p95"]


def q(x, p, axis=None):
    return np.quantile(x, p, axis=axis, method="inverted_cdf")


def worst_set(T):
    """Indices of the worst 10% of graphs (round(0.1 x graphs)) by per-graph p95 averaged over the variants."""
    p95 = q(T, 0.95, axis=2)
    return np.argsort(-p95.mean(axis=0), kind="stable")[:int(round(0.1 * T.shape[1]))]


def statistics(T, worst=None):
    """STATS per variant (array variants x 4) and the worst-decile columns of T used (worst_set(T) if None)."""
    p95 = q(T, 0.95, axis=2)                                         # variants x graphs
    if worst is None:
        worst = worst_set(T)
    flat = T.reshape(T.shape[0], -1)
    out = np.column_stack([q(flat, 0.5, axis=1), q(flat, 0.95, axis=1), p95.mean(axis=1), p95[:, worst].mean(axis=1)])
    return out, worst


def bootstrap(T, n_boot=1000, seed=config.SEED):
    """Resampled statistics: array n_boot x variants x 4."""
    rng = np.random.default_rng(seed)
    G = T.shape[1]
    fixed = worst_set(T)
    out = np.empty((n_boot, T.shape[0], len(STATS)))
    for b in range(n_boot):
        g = rng.integers(0, G, G)
        r = rng.integers(0, 200, (G, 200))
        out[b] = statistics(T[:, g[:, None], r], np.flatnonzero(np.isin(g, fixed)))[0]
    return out


def ci(x):
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def summary(T, graphs, variants, label):
    """Point estimates with 95% CIs per variant and paired differences, plus max p95 and failures."""
    point, worst = statistics(T)
    boot = bootstrap(T)
    p95 = q(T, 0.95, axis=2)
    rows = []
    for v, name in enumerate(variants):
        row = {"graphs": label, "n_graphs": T.shape[1], "variant": name}
        for s, stat in enumerate(STATS):
            row[stat], (row[stat + "_lo"], row[stat + "_hi"]) = point[v, s], ci(boot[:, v, s])
        m = int(np.argmax(p95[v]))
        row.update(max_p95=p95[v, m], max_p95_graph=graphs["graph"].iloc[m],
                   graphs_with_failure=int(np.isinf(T[v]).any(axis=1).sum()), failure_rate=float(np.isinf(T[v]).mean()))
        rows.append(row)
    pairs = [(1, 0), (2, 0), (2, 1)]
    for a, b in pairs:
        row = {"graphs": label, "n_graphs": T.shape[1], "variant": f"{variants[a]} - {variants[b]}"}
        for s, stat in enumerate(STATS):
            row[stat] = point[a, s] - point[b, s]
            row[stat + "_lo"], row[stat + "_hi"] = ci(boot[:, a, s] - boot[:, b, s])
        rows.append(row)
    return pd.DataFrame(rows), worst


def decide(summ, variants):
    """The pre-stated rule on the S0b''' variants (variants[1] = eps 0.75, variants[2] = eps 1)."""
    d = summ.set_index("variant").loc[f"{variants[2]} - {variants[1]}"]
    for stat, step in [("worst_decile_p95", "1: worst-decile p95"), ("pooled_p95", "2: pooled p95")]:
        lo, hi = d[stat + "_lo"], d[stat + "_hi"]
        if lo > 0 or hi < 0:
            chosen = variants[2] if d[stat] < 0 else variants[1]
            return chosen, f"step {step}: eps 1 - eps 0.75 = {d[stat]:.2f} s, CI [{lo:.2f}, {hi:.2f}] excludes 0"
    return variants[2], ("step 3 (default eps 1): neither CI excludes 0 (worst-decile "
                         f"[{d['worst_decile_p95_lo']:.2f}, {d['worst_decile_p95_hi']:.2f}], pooled p95 "
                         f"[{d['pooled_p95_lo']:.2f}, {d['pooled_p95_hi']:.2f}])")


def per_graph(T, Y, B, M, graphs, variants):
    rows = []
    for v, name in enumerate(variants):
        for g, row in graphs.reset_index(drop=True).iterrows():
            t = T[v, g]
            rows.append({"graph": row["graph"], "variant": name, "success_180": float(np.mean(t <= 180)),
                         "success_300": float(np.mean(t <= 300)), "median_t": float(q(t, 0.5)), "p95_t": float(q(t, 0.95)),
                         "yields_per_run": Y[v, g].mean(), "breakups_per_min": B[v, g].sum() / M[v, g].sum(),
                         "edges": row["n_edges"], "n_fraught": row["n_fraught"], "lam": row["lam"],
                         "screening": bool(row["screening"])})
    return pd.DataFrame(rows)


def figure(pg, variants, path):
    colors = {variants[0]: report.BLUE, variants[1]: report.ORANGE, variants[2]: "#1baf7a"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=report.SURFACE)
    ax = axes[0]
    base = pg[pg["variant"] == variants[0]].set_index("graph")["p95_t"]
    for name in variants[1:]:
        y = pg[pg["variant"] == name].set_index("graph")["p95_t"].reindex(base.index)
        ax.scatter(base, y, s=14, color=colors[name], alpha=0.7, edgecolor="none", label=name)
    top = min(330, np.nanmax(np.where(np.isfinite(pg["p95_t"]), pg["p95_t"], np.nan)) * 1.05)
    ax.plot([0, top], [0, top], color=report.INK_2, linewidth=1)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("S0b'' per-graph p95 (s)", fontsize=8, color=report.INK_2)
    ax.set_ylabel("S0b''' per-graph p95 (s)", fontsize=8, color=report.INK_2)
    ax.set_title("Per-graph p95: below the diagonal = shorter tail than S0b''", fontsize=9, color=report.INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    rng = np.random.default_rng(0)
    for k, name in enumerate(variants):
        d = pg[pg["variant"] == name]
        ax.scatter(d["edges"] + (k - 1) * 0.22 + rng.uniform(-0.08, 0.08, len(d)), d["median_t"], s=10,
                   color=colors[name], alpha=0.6, edgecolor="none", label=name)
    ax.set_xlabel("number of edges", fontsize=8, color=report.INK_2)
    ax.set_ylabel("per-graph median time to consensus (s)", fontsize=8, color=report.INK_2)
    ax.set_title("Per-graph median vs edges (offset by variant)", fontsize=9, color=report.INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        report.style(ax)
    fig.text(0.01, 0.005, "597 graph-train graphs, start A, 200 paired runs per graph and variant (seed base 5731).",
             fontsize=8, color=report.INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=report.SURFACE)


def analyse(runs, graphs, variants):
    graphs = graphs.reset_index(drop=True)
    arr = {}
    for col in ["t_consensus", "n_yields", "breakups"]:
        x = runs.pivot_table(index=["variant", "graph"], columns="run", values=col, aggfunc="first", dropna=False)
        x = x.reindex(pd.MultiIndex.from_product([variants, graphs["graph"]]))
        arr[col] = x.to_numpy(float).reshape(len(variants), len(graphs), 200)
    T = np.where(np.isnan(arr["t_consensus"]), np.inf, arr["t_consensus"])
    M = np.where(np.isinf(T), 300.0, T) / 60
    pg = per_graph(T, arr["n_yields"], arr["breakups"], M, graphs, variants)
    pg.to_csv(config.PATH3 / "robust_T5.csv", index=False)
    s_all, worst = summary(T, graphs, variants, "all train")
    oos = (~graphs["screening"]).to_numpy()
    s_oos, _ = summary(T[:, oos], graphs[oos].reset_index(drop=True), variants, "not in screening set")
    summ = pd.concat([s_all, s_oos], ignore_index=True)
    summ.to_csv(config.PATH3 / "robust_T5_summary.csv", index=False)
    chosen, why = decide(s_all, variants)
    d = s_all.set_index("variant").loc[f"{chosen} - {variants[0]}"]
    dec = pd.DataFrame([{"chosen": chosen, "rule": why, "chosen_minus_S0b''_worst_decile_p95": d["worst_decile_p95"],
                         "lo": d["worst_decile_p95_lo"], "hi": d["worst_decile_p95_hi"],
                         "beats_S0b''": bool(d["worst_decile_p95_hi"] < 0)}])
    dec.to_csv(config.PATH3 / "robust_T5_decision.csv", index=False)
    corr = []
    for name in variants:
        x = pg[pg["variant"] == name]
        corr.append({"variant": name, **{f"spearman_median_vs_{m}": x["median_t"].corr(x[m], method="spearman")
                                         for m in ["edges", "n_fraught", "lam"]}})
    pd.DataFrame(corr).to_csv(config.PATH3 / "robust_T5_spearman.csv", index=False)
    p95 = q(T, 0.95, axis=2)
    top = np.argsort(-p95.mean(axis=0), kind="stable")[:10]
    w = graphs.iloc[top][["graph", "n_edges", "n_fraught", "lam", "screening"]].reset_index(drop=True)
    for v, name in enumerate(variants):
        w[f"p95 {name}"] = p95[v, top]
        w[f"median {name}"] = q(T[v, top], 0.5, axis=1)
    w.to_csv(config.PATH3 / "robust_T5_worst10.csv", index=False)
    figure(pg, variants, config.FIGURES / "path3_robust_T5.png")
    return summ, dec

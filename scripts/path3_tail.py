"""Path 3 tail diagnosis of S0b'' (defaults eps 0.5, H 4 s), start A, replaying screen_A runs from runs.csv.

Groups: on Hierarchy and Full, the runs ranked by t_consensus (failures = inf, ties in runs.csv order):
slow = the 20 slowest, median = ranks 91-110, fast = the 20 fastest; on every other screening graph the 5%
slowest (10 of 200), pooled as slow_other. Each replay must reproduce its t_consensus and yield count exactly,
or the script stops. Outputs in results/path3/tail/: tail_runs.csv (one row per run), tail_yields.csv (one row
per yield), tail_summary.csv (mean and median of each metric per graph x group, and the pooled slow_other);
figure results/figures/path3_tail_examples.png.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slider import config
from slider.path3 import graphs, human, report, sim, strategies, tail

OUT = config.PATH3 / "tail"
PARAMS = {"eps": 0.5, "H": 4.0}
SCREEN = pd.read_csv(config.PATH3 / "runs.csv").query("block == 'screen_A' and strategy == 'S0b'")
GRAPHS = graphs.load_screening_set()
DELAYS = human.load_delays()


def groups():
    """(graph, group, rank, seed) of every run to replay; rank 1 = fastest."""
    out = []
    for graph, d in SCREEN.groupby("graph", sort=False):
        t = d["t_consensus"].fillna(np.inf).to_numpy()
        order = np.argsort(t, kind="stable")
        seeds = d["seed"].to_numpy()[order]
        if graph in ("Hierarchy", "Full"):
            picks = [("fast", range(0, 20)), ("median", range(90, 110)), ("slow", range(180, 200))]
        else:
            picks = [("slow_other", range(190, 200))]
        for group, idx in picks:
            out += [(graph, group, r + 1, seeds[r]) for r in idx]
    return out


def replay(graph, seed):
    rng = np.random.default_rng([int(x) for x in seed.split("-")])
    return sim.run(GRAPHS[graph], strategies.S0b, PARAMS, np.zeros(7, dtype=int), DELAYS, rng, keep_path=True)


def summary(runs):
    metrics = [c for c in runs.columns if c not in ("graph", "group", "rank", "seed")]
    parts = []
    for (graph, group), d in list(runs[runs["group"] != "slow_other"].groupby(["graph", "group"])) + \
            [(("pooled other", "slow_other"), runs[runs["group"] == "slow_other"])]:
        x = d[metrics].replace([np.inf], np.nan)
        row = {"graph": graph, "group": group, "runs": len(d), "failures": int(d["t_consensus"].isna().sum())}
        for m in metrics:
            row[m + "_mean"], row[m + "_median"] = x[m].mean(), x[m].median()
        parts.append(row)
    order = {"slow": 0, "median": 1, "fast": 2, "slow_other": 3}
    return pd.DataFrame(parts).sort_values(["graph", "group"], key=lambda c: c.map(order) if c.name == "group" else c)


def figure(examples, path):
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), facecolor=report.SURFACE, sharey=True)
    for ax, (title, out) in zip(axes.ravel(), examples):
        report.style(ax)
        p = out["path"] / 10
        t = np.arange(len(p)) * 0.1
        for i in range(7):
            ax.plot(t, p[:, i], color=colors[i], linewidth=1.1, label=f"P{i + 1}")
        for i, start, end in out["yields"]:
            ax.scatter(start * 0.1, p[start, i], marker="v", s=28, color=report.INK, zorder=4,
                       edgecolor=report.SURFACE, linewidth=0.8)
        ax.set_title(title, fontsize=9, color=report.INK, loc="left")
        ax.set_xlabel("time (s)", fontsize=8, color=report.INK_2)
        ax.set_ylabel("s_i", fontsize=8, color=report.INK_2)
        ax.set_ylim(-1.1, 1.1)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=7, loc="lower left", bbox_to_anchor=(0, 1.08))
    fig.text(0.01, 0.005, "S0b'' (eps 0.5, H 4 s), start A, screen_A seeds. Black triangles: yields (at the "
             "yielder's position when it decides).", fontsize=8, color=report.INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150, facecolor=report.SURFACE)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    screen = SCREEN.set_index(["graph", "seed"])
    rows, ylist, outs = [], [], {}
    for graph, group, rank, seed in groups():
        out = replay(graph, seed)
        ref = screen.loc[(graph, seed)]
        same_t = (np.isnan(out["t_consensus"]) and np.isnan(ref["t_consensus"])) or out["t_consensus"] == ref["t_consensus"]
        if not (same_t and out["n_yields"] == ref["n_yields"]):
            raise SystemExit(f"replay mismatch: {graph} {seed}: t {out['t_consensus']} vs {ref['t_consensus']}, "
                             f"yields {out['n_yields']} vs {ref['n_yields']}")
        m, y = tail.run_metrics(GRAPHS[graph], out)
        rows.append({"graph": graph, "group": group, "rank": rank, "seed": seed, **m})
        ylist.append(y.assign(graph=graph, group=group, seed=seed))
        outs[(graph, rank)] = out
    print(len(rows), "replays, all reproduce t_consensus and yield count")
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "tail_runs.csv", index=False)
    pd.concat(ylist, ignore_index=True).to_csv(OUT / "tail_yields.csv", index=False)
    summary(runs).to_csv(OUT / "tail_summary.csv", index=False)
    examples = []
    for graph in ["Hierarchy", "Full"]:
        for rank, label in [(200, "slowest"), (100, "median (rank 100)")]:
            out = outs[(graph, rank)]
            t = out["t_consensus"]
            examples.append((f"{graph}, {label}: " + (f"consensus at {t:.1f} s" if not np.isnan(t) else
                                                      "no consensus by 300 s"), out))
    figure(examples, config.FIGURES / "path3_tail_examples.png")


if __name__ == "__main__":
    main()

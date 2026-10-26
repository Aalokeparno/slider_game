"""Path 1, P2a: timing of the log rows and slider values off the 0.1 grid.

Steps: k = round(10 s), s = weights * X (reversals.to_steps, slider_logs.sign_correct).
Players are 0-based; run["bot_players"] holds 1-based seats.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slider import slider_logs as sl
from slider.reversals import step_sizes

DATASETS = ["EL", "CL", "CC"]
QUANTILES = {"min": 0, "1%": .01, "5%": .05, "25%": .25, "50%": .5, "75%": .75, "95%": .95, "99%": .99}


def row_gaps(runs):
    """One row per consecutive pair of log rows: dataset, file, wave, row (= k + 1), gap = t[k+1] - t[k],
    n_movers (human players whose k = round(10 s) changed in row k + 1).
    gap is rounded to 1 µs to drop float noise (the logs have at most 3 decimals)."""
    parts = []
    for run in runs:
        t = run["t"]
        parts.append(pd.DataFrame({"dataset": run["dataset"], "file": run["file"], "wave": run["wave"],
                                   "row": np.arange(1, len(t)), "gap": np.round(np.diff(t), 6),
                                   "n_movers": (step_sizes(run) > 0).sum(axis=1)}))   # bot seats are NaN
    return pd.concat(parts, ignore_index=True)


def off_grid(runs):
    """One row per sign-corrected value s (every row, every seat): dataset, file, wave, player, seat_type
    (human / bot), row, n_rows (rows in the run), value, off_grid (|10 s - round(10 s)| > 1e-6)."""
    parts = []
    for run in runs:
        s = sl.sign_correct(run)
        T = len(s)
        bot = np.array([j + 1 in run["bot_players"] for j in range(7)])
        values = s.ravel()                                  # row 0 players 0..6, then row 1, ...
        parts.append(pd.DataFrame({"dataset": run["dataset"], "file": run["file"], "wave": run["wave"],
                                   "player": np.tile(np.arange(7), T),
                                   "seat_type": np.where(np.tile(bot, T), "bot", "human"),
                                   "row": np.repeat(np.arange(T), 7), "n_rows": T, "value": values,
                                   "off_grid": np.abs(10 * values - np.round(10 * values)) > 1e-6}))
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ summaries

def gap_quantiles(gaps):
    """Quantiles of the gaps (s) by dataset."""
    q = gaps.groupby("dataset")["gap"].quantile(list(QUANTILES.values())).unstack().loc[DATASETS]
    q.columns = list(QUANTILES)
    return q


def plot_gap_hist(gaps, width=0.005, top=1.0):
    """Histogram of the gaps in [0, top] s, one panel per dataset."""
    edges = np.round(np.arange(0, top + width / 2, width), 6)     # rounded like the gaps, so no float noise at edges
    _, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, d in zip(axes, DATASETS):
        g = gaps.loc[gaps["dataset"] == d, "gap"]
        ax.hist(g, bins=edges, color="#4C72B0")
        ax.set_title(f"{d}: {np.mean(g <= top):.1%} of {len(g)} gaps are in [0, {top}] s", fontsize=10)
        ax.set_ylabel("rows")
    axes[-1].set_xlabel(f"gap between consecutive log rows (s), bins of {width} s")
    plt.tight_layout()
    plt.show()


def top_values(df, column, n=10, decimals=3):
    """The n most common values of column (rounded to decimals) by dataset, with their counts."""
    parts = {}
    for d in DATASETS:
        counts = df.loc[df["dataset"] == d, column].round(decimals).value_counts().head(n)
        parts[d] = pd.DataFrame({column: counts.index, "n": counts.values}, index=range(1, len(counts) + 1))
    return pd.concat(parts, axis=1)


def time_decimals(runs, max_decimals=6):
    """Number of logged times t with 0, 1, 2, ... decimals (smallest d with t * 10^d whole, up to float noise),
    by dataset. The largest d with a nonzero count is the timestamp resolution.
    Cloud t is clock time minus the clock at the start of the round, so it has the clock's decimals."""
    rows = []
    for run in runs:
        for x in run["t"]:
            d = 0
            while d < max_decimals and abs(x * 10**d - round(x * 10**d)) > 1e-3:
                d += 1
            rows.append({"dataset": run["dataset"], "decimals": d})
    rows = pd.DataFrame(rows)
    return pd.crosstab(rows["decimals"], rows["dataset"])[DATASETS]


def lab_gap_by_movers(gaps):
    """Lab only (EL, CL): number of rows and median gap (s) for rows with 1 mover vs. 2 or more."""
    lab = gaps[gaps["dataset"].isin(["EL", "CL"]) & (gaps["n_movers"] >= 1)].copy()
    lab["movers"] = np.where(lab["n_movers"] == 1, "1", ">= 2")
    return lab.groupby(["dataset", "movers"])["gap"].agg(rows="count", median_gap="median").loc[["EL", "CL"]]


def off_grid_share(og):
    """Number of values, number off grid and % off grid, by dataset and seat type."""
    out = og.groupby(["dataset", "seat_type"])["off_grid"].agg(values="count", off_grid="sum", pct="mean")
    out["pct"] = 100 * out["pct"]
    return out.loc[DATASETS]


def off_grid_position(og):
    """Off-grid values in the first row of a round, the last row and the rows in between, by dataset:
    number of values, number off grid and % off grid."""
    og = og.assign(position=np.where(og["row"] == 0, "first",
                                     np.where(og["row"] == og["n_rows"] - 1, "last", "middle")))
    out = og.groupby(["dataset", "position"])["off_grid"].agg(values="count", off_grid="sum", pct="mean")
    out["pct"] = 100 * out["pct"]
    return out.loc[DATASETS]

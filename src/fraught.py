"""Fraught-state tests and % fraught measures, moved out of the Stage 0 and Stage 1 notebooks.

The function bodies are unchanged. Functions that used the notebook variables
`runs`, `table` or `masks` now take them as arguments.
"""
import numpy as np

import slider_logs as sl
from models import apem_osf


# ---------------------------------------------------- Stage 0, HW 5 (corners)

def is_fraught(A, s):
    return bool(np.all(s * (A @ s) >= 0))


# ------------------------------------------------------ Stage 1, HW 2 and HW 3

def fraught_mask(run):
    A = run["A"]                         # 7x7, no self-loops
    s = sl.sign_correct(run)             # s = weights * X, shape T x 7
    no_zeros = np.all(s != 0, axis=1)
    mixed = ~(np.all(s > 0, axis=1) | np.all(s < 0, axis=1))
    delta = np.sign(s) * (s @ A)         # row k of s @ A is A @ s_k (A is symmetric)
    no_switch = np.all(delta >= 0, axis=1)
    return no_zeros & mixed & no_switch


# HW 2 version: lab time limit only, returns (time-weighted %, row-counted %)
def fraught_percent(i, runs, masks, table):
    t = runs[i]["t"]
    mask = masks[i]
    if table["success"][i]:
        T_end = t[-1]                  # the consensus row
    else:
        T_end = max(300, t[-1])        # lab time limit (one failed run has its last row at 300.08 s)
    dt = np.diff(np.append(t, T_end))  # how long each row lasts
    by_time = 100 * np.sum(dt * mask) / T_end
    by_rows = 100 * np.mean(mask)
    return by_time, by_rows


# method 1, forward hold: each row lasts until the next row
def fraught_percent_hold(i, runs, table):
    run = runs[i]
    t = run["t"]
    if table["success"][i]:
        T_end = t[-1]                    # the consensus row
    elif table["session"][i] == "lab":
        T_end = max(300, t[-1])          # lab time limit
    else:
        T_end = max(180, t[-1])          # cloud time limit
    dt = np.diff(np.append(t, T_end))    # how long each row lasts
    return 100 * np.sum(dt * fraught_mask(run)) / T_end


# method 2, interpolation (authors' Matlab code): join the rows with straight lines,
# take 1001 evenly spaced times from the first row to the last row
def fraught_percent_interp(i, runs, n=1000):
    run = runs[i]
    t = run["t"]
    s = sl.sign_correct(run)
    if len(t) < 2:
        return 0.0
    grid = np.linspace(t[0], t[-1], n + 1)
    s_grid = np.column_stack([np.interp(grid, t, s[:, j]) for j in range(7)])
    # weights = 1 because s_grid is already sign-corrected
    mask = fraught_mask({"A": run["A"], "weights": np.ones(7), "X": s_grid})
    return 100 * np.mean(mask)


# --------------------------------------------------- Stage 1, HW 4 (Ring traps)

def trap_sums(run, dt):
    A = run["A"]
    s = sl.sign_correct(run)
    fraught = fraught_mask(run)
    c = np.sign(s)                                                    # sign pattern of each row
    trap = np.all(s != 0, axis=1) & ~np.all(c == c[:, [0]], axis=1) & np.all(c * (c @ A) >= 0, axis=1)
    kind = np.minimum((c > 0).sum(axis=1), (c < 0).sum(axis=1))       # 2 = 5-2 split, 3 = 4-3 split

    assert np.all(trap[fraught])                                      # every fraught row is a trap row
    assert np.all(np.isin(kind[trap], [2, 3]))                        # only 5-2 and 4-3 traps on the Ring
    sums = {"T52": dt[trap & (kind == 2)].sum(), "T43": dt[trap & (kind == 3)].sum(),
            "F52": dt[fraught & (kind == 2)].sum(), "F43": dt[fraught & (kind == 3)].sum()}
    assert np.isclose(sums["F52"] + sums["F43"], dt[fraught].sum())   # fraught time is fully split
    return sums


# method 1, forward hold: each row lasts until the next row (all Ring runs here are lab runs)
def sums_hold(i, runs, table):
    run = runs[i]
    t = run["t"]
    T_end = t[-1] if table["success"][i] else max(300, t[-1])
    dt = np.diff(np.append(t, T_end))
    return trap_sums(run, dt)


# method 2, interpolation: 1001 evenly spaced times from the first row to the last, equal weights (in seconds)
def sums_interp(i, runs, n=1000):
    run = runs[i]
    t = run["t"]
    grid = np.linspace(t[0], t[-1], n + 1)
    s_grid = np.column_stack([np.interp(grid, t, sl.sign_correct(run)[:, j]) for j in range(7)])
    dt = np.full(n + 1, (t[-1] - t[0]) / (n + 1))
    return trap_sums({"A": run["A"], "weights": np.ones(7), "X": s_grid}, dt)   # weights = 1, s_grid is already sign-corrected


def trap_mask(run):
    s = sl.sign_correct(run)
    c = np.sign(s)
    return np.all(s != 0, axis=1) & ~np.all(c == c[:, [0]], axis=1) & np.all(c * (c @ run["A"]) >= 0, axis=1)   # same rule as trap_sums


# ------------------------------------------------- Stage 1, HW 5 (APEM runs)
# The HW 5 versions take a simulated run (with run["success"]) instead of a row number,
# so they are named *_sim to keep them apart from the HW 3 versions above.

# forward hold (HW 3 method 1): each row lasts until the next row, time limit 300 s (lab) or 180 s (cloud)
def fraught_percent_hold_sim(run, T_limit=300):
    t = run["t"]
    T_end = t[-1] if run["success"] else max(T_limit, t[-1])
    dt = np.diff(np.append(t, T_end))
    return 100 * np.sum(dt * fraught_mask(run)) / T_end


# authors' measure for apem.py runs: interpolate through all rows at 1001 evenly spaced times (HW 3 method 2)
def fraught_percent_interp_sim(run, n=1000):
    t = run["t"]
    grid = np.linspace(t[0], t[-1], n + 1)
    S = np.column_stack([np.interp(grid, t, run["X"][:, j]) for j in range(7)])
    return 100 * np.mean(fraught_mask({"A": run["A"], "weights": np.ones(7), "X": S}))


# authors' measure: time warp (1001 points), share of fraught states; fraught_mask = fraught.m
def fraught_percent_osf(run):
    S = apem_osf.warp(run)
    return 100 * np.mean(fraught_mask({"A": run["A"], "weights": np.ones(7), "X": S}))

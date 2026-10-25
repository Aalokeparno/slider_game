"""Fraught-state tests and % fraught measures, moved out of the Stage 0 and Stage 1 notebooks.

The function bodies are unchanged. Functions that used the notebook variables
`runs`, `table` or `masks` now take them as arguments.
"""
import numpy as np
import pandas as pd

from slider import slider_logs as sl
from slider.models import apem_osf


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


# ---------------------------------------------------------------- fraught spells

def spells(t, mask, t_end, censored, tau=1.0, labels=None, is_consensus=None):
    """Fraught spells of one round, one row per spell.

    Forward hold: row k lasts [t_k, t_k+1), the last row lasts until t_end.
    A spell is a maximal run of True rows of mask. Two spells separated by a non-mask stretch
    shorter than tau are merged, unless that stretch has a consensus row (consensus ends a spell).

    t_end      end of the round: last row time if it succeeded or has no end marker,
               max(time limit, last row time) if it failed
    censored   True if the round failed or has no end marker
    labels     one label per row, e.g. the sign pattern of s ("+-++-+-")
    is_consensus  True on consensus rows; default: the last row of a round that is not censored

    Columns: start, end, duration, after, t_to_consensus, orthants (distinct labels on the spell's
    True rows, in order of first appearance), n_hops (label changes between the spell's True rows).
    after = what comes next once the spell has ended (the next event, not the next row):
        "consensus"   a consensus row comes before any new spell
        "relapse"     a new spell starts before consensus
        "unresolved"  neither happens before the round ends
        "censored"    the spell is still open at t_end of a censored round
    t_to_consensus = consensus time - spell end if after == "consensus", else NaN.
    """
    t = np.asarray(t, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n = len(t)
    if n > 0 and t_end < t[-1]:
        raise ValueError(f"t_end = {t_end} is before the last row (t = {t[-1]})")
    if is_consensus is None:
        is_consensus = np.zeros(n, dtype=bool)
        if n > 0 and not censored:
            is_consensus[-1] = True
    is_consensus = np.asarray(is_consensus, dtype=bool)
    row_end = np.append(t[1:], t_end)          # row k lasts [t[k], row_end[k])

    # maximal runs of True rows, as [first row, last row]
    blocks = []
    k = 0
    while k < n:
        if mask[k]:
            j = k
            while j + 1 < n and mask[j + 1]:
                j += 1
            blocks.append([k, j])
            k = j + 1
        else:
            k += 1

    # merge a block into the previous one if the stretch between them is shorter than tau
    # and has no consensus row
    merged = []
    for k, j in blocks:
        if merged:
            prev_last = merged[-1][1]
            gap = t[k] - row_end[prev_last]
            if gap < tau and not is_consensus[prev_last + 1:k].any():
                merged[-1][1] = j
                continue
        merged.append([k, j])

    rows = []
    for k, j in merged:
        end = row_end[j]
        t_to_consensus = np.nan
        if j == n - 1:                                     # still open at t_end
            after = "censored" if censored else "consensus"
            if after == "consensus":
                t_to_consensus = 0.0
        else:
            after = "unresolved"
            for r in range(j + 1, n):                      # first event after the spell
                if is_consensus[r]:
                    after = "consensus"
                    t_to_consensus = t[r] - end
                    break
                if mask[r]:
                    after = "relapse"
                    break
        if labels is None:
            orthants, n_hops = None, None
        else:
            seq = [labels[r] for r in range(k, j + 1) if mask[r]]
            orthants = tuple(dict.fromkeys(seq))
            n_hops = sum(a != b for a, b in zip(seq, seq[1:]))
        rows.append({"start": t[k], "end": end, "duration": end - t[k], "after": after,
                     "t_to_consensus": t_to_consensus, "orthants": orthants, "n_hops": n_hops})
    return pd.DataFrame(rows, columns=["start", "end", "duration", "after", "t_to_consensus", "orthants", "n_hops"])

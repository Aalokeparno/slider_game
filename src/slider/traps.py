"""P27: trap hops, from one trap orthant to another, within a round (training data only).

Spells: fraught.spells of fraught.fraught_mask, forward hold, t_end = fraught.round_end, censored unless the round
succeeded, tau = 1 s (the spells default). A spell can hold more than one orthant (two fraught stretches less than
1 s apart are merged, or a slider jumps across 0 between two fraught rows), so each spell is cut wherever the
canonical orthant changes; each piece then has one orthant.

Canonical orthant of a fraught row: its sign vector (seat order) times the sign of its first entry, so a pattern
and its global negation are the same orthant (fraught rows have no zeros).

Per round: n_spells = number of pieces; n_distinct = number of distinct canonical orthants; n_hops = number of
consecutive pieces (in time order) with different orthants; n_cut = number of spells cut into more than one piece.
"""
import numpy as np
import pandas as pd

from slider import config
from slider import slider_logs as sl
from slider.fraught import fraught_mask, round_end, spells
from slider.stats import group_bootstrap_mean

STATS = ["n_spells", "n_distinct", "n_hops", "any_hop"]


def canonical(row):
    """Sign vector of a row of s times the sign of its first nonzero entry, as a tuple."""
    c = np.sign(row).astype(int)
    first = c[np.flatnonzero(c)[0]]
    return tuple(c * first)


def trap_hops(run):
    """n_spells, n_distinct, n_hops, n_cut of one round (see the module docstring)."""
    t = run["t"]
    s = sl.sign_correct(run)
    mask = fraught_mask(run)
    sp = spells(t, mask, round_end(t, run["end"], run["session"]), censored=run["end"] != "success")
    pieces = []                                         # canonical orthant of each piece, in time order
    n_cut = 0
    for k, j in zip(sp["first_row"], sp["last_row"]):
        seq = [canonical(s[r]) for r in range(k, j + 1) if mask[r]]
        assert all(np.all(s[r] != 0) for r in range(k, j + 1) if mask[r])     # fraught rows have no zeros
        cut = [o for i, o in enumerate(seq) if i == 0 or o != seq[i - 1]]     # one entry per piece
        n_cut += len(cut) > 1
        pieces += cut
    return {"n_spells": len(pieces), "n_distinct": len(set(pieces)),
            "n_hops": sum(a != b for a, b in zip(pieces, pieces[1:])), "n_cut": n_cut}


def trap_hop_runs(runs):
    """One row per round: topology, dataset, file, wave and the trap_hops values; any_hop = 100 if n_hops >= 1."""
    rows = [{"topology": sl.label_topology(run["A"]), "dataset": run["dataset"], "file": run["file"],
             "wave": run["wave"], **trap_hops(run)} for run in runs]
    out = pd.DataFrame(rows, columns=["topology", "dataset", "file", "wave", "n_spells", "n_distinct", "n_hops", "n_cut"])
    out["any_hop"] = 100.0 * (out["n_hops"] >= 1)
    return out


def trap_hop_table(runs, n_boot=1000, seed=config.SEED):
    """Per cell (topology x dataset): mean n_spells, n_distinct, n_hops and % of rounds with n_hops >= 1 (any_hop),
    each with a 95% CI by group bootstrap (log files resampled, stats.group_bootstrap_mean).
    Long format: topology, dataset, stat, value, ci_lo, ci_hi, n_runs, n_groups, n_cut (spells cut in the cell)."""
    r = trap_hop_runs(runs)
    out = []
    for (topology, dataset), c in r.groupby(["topology", "dataset"], sort=True):
        for stat in STATS:
            lo, hi = group_bootstrap_mean(c[stat], c["file"], n_boot, seed)
            out.append({"topology": topology, "dataset": dataset, "stat": stat, "value": c[stat].mean(),
                        "ci_lo": lo, "ci_hi": hi, "n_runs": len(c), "n_groups": c["file"].nunique(),
                        "n_cut": int(c["n_cut"].sum())})
    return pd.DataFrame(out)

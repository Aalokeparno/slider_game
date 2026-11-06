"""Path 3 tail diagnosis: per-run and per-yield metrics from a replayed trajectory (sim.run with keep_path).

Ticks are the states held during each tick: path[n] for n = 0 .. len(path) - 2 (as frac_time_fraught in the screen).
    fraught      game.fraught_now (fraught corner sign pattern and every |s_i| >= 0.5)
    near         |sum of s| / 7 >= 5/7 and not fraught
    mid          at least one slider with |s| < 0.9
Breakup: |sum s| / 7 reaches >= 5/7, and later drops to <= 3/7 (then it must reach 5/7 again to count another).
Yield (sim yields: player, first tick = the decision tick, first idle tick = arrival at the opposite wall):
    side         sign of the yielder's s at the decision tick (it is at its probe position, |s| = 0.9)
    on_majority  side == sign of sum of s at the decision tick (ties: sum = 0 -> NaN)
    at_fraught   fraught_now at the decision tick
    undone       the yielder is back on its old side (sign of s == side) within 10 s of the decision tick
    age_s        time since the yielder's last wall-to-wall flip (last arrival at a wall other than the wall it
                 was at before); its first arrival at a wall from 0 counts as a flip
    agr_before   the yielder's displayed Agr (integer %) at the tick before its probe (decision tick - 1)
"""
import numpy as np
import pandas as pd

from slider.path3 import game

UNDO_S = 10.0


def last_flips(k):
    """For one player's positions k (per tick): flip[n] = the last tick <= n at which it arrived at a wall other than
    the previous wall it was at (the first wall from 0 counts); -1 if none yet."""
    flip = np.full(len(k), -1)
    wall, last = 0, -1
    for n, v in enumerate(k):
        if abs(v) == game.WALL and np.sign(v) != wall:
            wall, last = np.sign(v), n
        flip[n] = last
    return flip


def breakups(path):
    """Breakups in a trajectory: |sum s| / 7 reaches >= 5/7 and later falls to <= 3/7 (it must reach 5/7 again
    before the next one counts)."""
    count, armed = 0, False
    for v in np.abs(path.sum(axis=1)) / (7 * game.WALL):
        if v >= 5 / 7 - 1e-9:
            armed = True
        elif armed and v <= 3 / 7 + 1e-9:
            count, armed = count + 1, False
    return count


def yield_flags(path, i, start, end):
    """(on_majority, undone) of one yield (see the module doc); on_majority is NaN when the sum of s is 0."""
    side = np.sign(path[start, i])
    maj = np.sign(path[start].sum())
    later = path[end:min(start + int(round(UNDO_S / game.DT)), len(path) - 1) + 1, i]
    return (float(side == maj) if maj != 0 else np.nan), bool(np.any(np.sign(later) == side))


def yield_table(A, path, yields):
    """One row per yield: player, start_s, end_s, on_majority, at_fraught, undone, age_s, agr_before."""
    fraught = game.fraught_now(A, path)
    flips = {i: last_flips(path[:, i]) for i in range(path.shape[1])}
    rows = []
    for i, start, end in yields:
        on_majority, undone = yield_flags(path, i, start, end)
        f = flips[i][start]
        rows.append({"player": i, "start_s": start * game.DT, "end_s": end * game.DT, "on_majority": on_majority,
                     "at_fraught": bool(fraught[start]), "undone": undone,
                     "age_s": (start - f) * game.DT if f >= 0 else np.nan,
                     "agr_before": float(game.agreement(A, path[max(start - 1, 0)])[i])})
    return pd.DataFrame(rows, columns=["player", "start_s", "end_s", "on_majority", "at_fraught", "undone", "age_s",
                                       "agr_before"])


def run_metrics(A, out):
    """Per-run metrics of one replayed run (sim.run output with keep_path)."""
    path = out["path"]
    held = path[:-1]
    fraught = game.fraught_now(A, held)
    level = np.abs(path.sum(axis=1)) / (7 * game.WALL)
    near = (level[:-1] >= 5 / 7) & ~fraught
    patterns = {tuple(r) for r in np.sign(held[fraught])}
    entries = int(fraught[0]) + int(np.sum(fraught[1:] & ~fraught[:-1]))
    y = yield_table(A, path, out["yields"])
    toward, away = out["cross_toward"], out["cross_away"]
    return {"t_consensus": out["t_consensus"], "frac_fraught": float(fraught.mean()), "fraught_entries": entries,
            "fraught_patterns": len(patterns), "moving_per_tick": out["moving_per_tick"],
            "frac_mid": float(np.mean(np.any(np.abs(held) < 9, axis=1))),
            "cross_toward_share": toward / (toward + away) if toward + away else np.nan,
            "max_level": float(level[:-1].max()), "breakups": breakups(path),
            "n_yields": out["n_yields"], "n_collisions": out["n_collisions"],
            "yield_on_majority": y["on_majority"].mean(), "yield_at_fraught": y["at_fraught"].mean(),
            "yield_undone": y["undone"].mean(), "yield_age_median_s": y["age_s"].median(),
            "yield_agr_before_mean": y["agr_before"].mean(),
            "time_fraught": float(fraught.mean()), "time_near": float(near.mean()),
            "time_other": float(1 - fraught.mean() - near.mean())}, y

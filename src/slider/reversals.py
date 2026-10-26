"""Path 1, P2: probes and reversals of each player's slider, and step sizes.

Everything works in integer steps k = round(10 s), with s = weights * X (slider_logs.sign_correct).
Players are 0-based (as in fraught.py); run["bot_players"] holds 1-based seats.
"""
import numpy as np
import pandas as pd

from slider import slider_logs as sl

COLUMNS = ["time", "type", "event_row", "row_before", "s_before", "dir_before", "n_steps"]
DATASETS = ["EL", "CL", "CC"]


def to_steps(s):
    """k = round(10 s) as integers."""
    return np.round(10 * np.asarray(s, dtype=float)).astype(int)


def event(t, s, row, kind, direction, n):
    return {"time": t[row], "type": kind, "event_row": row, "row_before": row - 1,
            "s_before": s[row - 1], "dir_before": direction, "n_steps": n}


def detect_events(t, s, m=2):
    """Probes and reversals of ONE player's slider, one row per event.

    t, s   all rows of the run (s in [-1, 1])
    A row with Δk = 0 is a pause: the direction carries over.
    Direction = sign of the last nonzero Δk; the first nonzero move only sets it (no event).
    A move against the direction opens a candidate at that row; a jump of |Δk| counts as |Δk| steps.
        "reversal"    the candidate reaches m steps against the direction; the direction flips
        "probe"       a move back in the old direction comes first; the direction stays
        "unresolved"  the round ends first
    Columns: time (t of the first opposite step), type, event_row, row_before (= event_row - 1),
    s_before (s at row_before), dir_before (+1 / -1), n_steps (steps against the direction when the
    event was decided: >= m for a reversal, < m for a probe or unresolved).
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    k = to_steps(s)
    events = []
    direction = 0      # sign of the last nonzero Δk (0 before the first move)
    start = None       # row where the open candidate started (None = no candidate)
    n = 0              # steps against the direction in the open candidate
    for i in range(1, len(k)):
        dk = int(k[i] - k[i - 1])
        if dk == 0:                                   # pause
            continue
        if direction == 0:                            # first move: sets the direction
            direction = int(np.sign(dk))
        elif np.sign(dk) == direction:                # move in the direction
            if start is not None:
                events.append(event(t, s, start, "probe", direction, n))
                start = None
        else:                                         # move against the direction
            if start is None:
                start, n = i, 0
            n += abs(dk)
            if n >= m:
                events.append(event(t, s, start, "reversal", direction, n))
                direction = -direction
                start = None
    if start is not None:
        events.append(event(t, s, start, "unresolved", direction, n))
    return pd.DataFrame(events, columns=COLUMNS)


def run_events(run, m=2):
    """detect_events for every player of one run; bot players are skipped.

    Columns: file, wave (together the run id), player (0-based), then the detect_events columns.
    """
    s = sl.sign_correct(run)
    rows = []
    for j in range(7):
        if j + 1 in run["bot_players"]:
            continue
        for ev in detect_events(run["t"], s[:, j], m).to_dict("records"):
            rows.append({"file": run["file"], "wave": run["wave"], "player": j, **ev})
    return pd.DataFrame(rows, columns=["file", "wave", "player"] + COLUMNS)


# ----------------------------------------------------------------- step sizes (P2a)

def step_sizes(run):
    """|Δk| of every row from the second on and every player, shape (T - 1, 7). Bot seats are NaN."""
    a = np.abs(np.diff(to_steps(sl.sign_correct(run)), axis=0)).astype(float)
    for seat in run["bot_players"]:
        a[:, seat - 1] = np.nan
    return a


def step_counts(runs, table):
    """Number of player-rows with |Δk| = 1, 2, 3, ... by dataset (human players only)."""
    sizes, datasets = [], []
    for k, run in enumerate(runs):
        a = step_sizes(run)
        moves = a[a > 0].astype(int)
        sizes += list(moves)
        datasets += [table["dataset"][k]] * len(moves)
    return pd.crosstab(pd.Series(sizes, name="|Δk|"), pd.Series(datasets, name="dataset"))[DATASETS]


def simultaneous_moves(runs, table):
    """By dataset: rows (from the second row of each run), rows where at least one human player moves
    (|Δk| >= 1), rows where more than one moves, and that last count as % of all rows and of moving rows."""
    rows = []
    for k, run in enumerate(runs):
        movers = (step_sizes(run) > 0).sum(axis=1)
        rows.append({"dataset": table["dataset"][k], "rows": len(movers),
                     "rows with a move": int(np.sum(movers >= 1)), "rows with >1 mover": int(np.sum(movers >= 2))})
    out = pd.DataFrame(rows).groupby("dataset").sum().loc[DATASETS]
    out["% of rows"] = 100 * out["rows with >1 mover"] / out["rows"]
    out["% of moving rows"] = 100 * out["rows with >1 mover"] / out["rows with a move"]
    return out

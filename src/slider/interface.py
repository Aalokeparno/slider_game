"""Path A, P7e: the slider interface of each session file (log file = group), from raw rows and human seats.

A move: a row where a human player's s changes (|Δs| > 0 against the row before; s = slider_logs.sign_correct).
A drag move: a move less than DRAG seconds after the same player's previous move in the round.
Step 4 (the simulator) reads interface_table.
"""
import re

import numpy as np
import pandas as pd

from slider import config
from slider import slider_logs as sl

DRAG = 0.3        # s: a move closer than this to the player's previous move is part of a drag
PACE_TOL = 0.3    # a round is flagged if its modal pace differs from the file's by more than this share
DATASETS = ["EL", "CL", "CC"]


def mode(values, decimals):
    """Most common value after rounding to decimals (the smallest on a tie); NaN if there are none."""
    v = np.round(np.asarray(values, dtype=float), decimals)
    if len(v) == 0:
        return np.nan
    u, n = np.unique(v, return_counts=True)
    return u[np.argmax(n)]


def version_tag(name):
    """Version tag in a log file name, e.g. "v09" in "log_v09_Fri Sep 23 ...txt"; "none" if there is none."""
    m = re.search(r"_(v\d+)_", name)
    return m.group(1) if m else "none"


def time_decimals(t):
    """Decimals needed to write every time in t (at most 6)."""
    t = np.asarray(t, dtype=float)
    for d in range(7):
        if np.all(np.abs(t - np.round(t, d)) < 1e-9):
            return d
    return 6


def moves(run):
    """Moves of the human players in one round, one row per (row, player) with |Δs| > 0.
    Columns: player (0-based), row, time, ds (= |Δs|), gap (time since the player's previous move; NaN first)."""
    s = sl.sign_correct(run)
    t = np.asarray(run["t"], dtype=float)
    parts = []
    for j in range(s.shape[1]):
        if j + 1 in run["bot_players"]:
            continue
        d = np.abs(np.diff(s[:, j]))
        rows = np.flatnonzero(d > 1e-9) + 1
        parts.append(pd.DataFrame({"player": j, "row": rows, "time": t[rows], "ds": d[rows - 1],
                                   "gap": np.diff(t[rows], prepend=np.nan)}))
    return pd.concat(parts, ignore_index=True)


def human_values(run):
    """All values s of the human seats in one round, flattened."""
    s = sl.sign_correct(run)
    return s[:, [j for j in range(s.shape[1]) if j + 1 not in run["bot_players"]]].ravel()


def step_and_pace(mv):
    """(step size, modal pace) of a set of moves.
    step size = mode of |Δs| (3 decimals); modal pace = mode over drag moves of (|Δs| / step size) / gap,
    in steps per second (1 decimal)."""
    step = mode(mv["ds"], 3)
    drag = mv[(mv["gap"] > 0) & (mv["gap"] < DRAG)]
    return step, mode(drag["ds"] / step / drag["gap"], 1)


def interface_table(runs):
    """One row per session file (group) of runs, from raw rows and human seats:
    file, dataset, group_id and split (split G), version (version_tag), rounds and rows (in runs),
    time_decimals, median_gap_s (between consecutive rows), pct_rows_multi_mover (% of rows, from the second
    of each round, where more than one human player moves), step_size, pct_on_grid (% of human values on the
    0.1 grid), modal_pace (steps/s), speed (units/s = step_size x modal_pace),
    flagged (some round's step size differs from the file's, or its modal pace by more than 30%),
    differing_rounds (which rounds, e.g. "wave 3: step 0.1") and unchecked_rounds (rounds with no move
    or no drag move, so no step size or modal pace to compare).
    """
    split = dict(pd.read_csv(config.SPLIT_G)[["group_id", "split"]].to_numpy())
    out = []
    for file in sorted({run["file"] for run in runs}):
        rounds = [run for run in runs if run["file"] == file]
        mv = [moves(run) for run in rounds]
        step, pace = step_and_pace(pd.concat(mv))
        differ, unchecked = [], []
        for run, m in zip(rounds, mv):
            round_step, round_pace = step_and_pace(m)
            if not np.isnan(round_step) and round_step != step:
                differ.append(f"wave {run['wave']}: step {round_step:g}")
            if abs(round_pace - pace) > PACE_TOL * pace:
                differ.append(f"wave {run['wave']}: pace {round_pace:g}")
            if np.isnan(round_step) or np.isnan(round_pace):
                unchecked.append(f"wave {run['wave']}")
        values = np.concatenate([human_values(run) for run in rounds])
        multi = sum(int(np.sum(m.groupby("row").size() > 1)) for m in mv)
        n_pairs = sum(len(run["t"]) - 1 for run in rounds)
        out.append({"file": file, "dataset": rounds[0]["dataset"], "group_id": file, "split": split.get(file, "none"),
                    "version": version_tag(file), "rounds": len(rounds), "rows": sum(len(run["t"]) for run in rounds),
                    "time_decimals": time_decimals(np.concatenate([run["t"] for run in rounds])),
                    "median_gap_s": np.median(np.concatenate([np.diff(run["t"]) for run in rounds])),
                    "pct_rows_multi_mover": 100 * multi / n_pairs, "step_size": step,
                    "pct_on_grid": 100 * np.mean(np.abs(10 * values - np.round(10 * values)) <= 1e-6),
                    "modal_pace": pace, "speed": step * pace, "flagged": len(differ) > 0,
                    "differing_rounds": "; ".join(differ), "unchecked_rounds": ", ".join(unchecked)})
    out = pd.DataFrame(out)
    order = out["dataset"].map({d: i for i, d in enumerate(DATASETS)})
    return out.iloc[np.lexsort((out["file"], order))].reset_index(drop=True)

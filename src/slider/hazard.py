"""Path A: at-risk time for reversal hazards.

A player is at risk of a reversal from their first move (the first row where k = round(10 s) changes;
in reversals.detect_events this move sets the direction) until fraught.round_end.
Agreement A_i is slider_logs.agreement / 100, so it lies in [0, 1].
"""
import numpy as np
import pandas as pd

from slider import slider_logs as sl
from slider.fraught import round_end
from slider.reversals import to_steps

COLUMNS = ["file", "wave", "player", "time", "row", "A"]


def agreement(run):
    """A_i of every row and player, in [0, 1]: slider_logs.agreement(A_log, s) / 100."""
    return sl.agreement(run["A_log"], sl.sign_correct(run)) / 100


def at_risk_samples(run, b=0.25):
    """Each human player's at-risk time in bins of b seconds, one row per bin.

    At-risk time: from the player's first move to fraught.round_end. Bins start at t_first + j * b
    (j = 0, 1, ... while before the round end); each stands for b seconds. Values by forward hold:
    row = the last row at or before the bin start, A = A_i at that row.
    Columns: file, wave, player (0-based), time (bin start), row, A.
    Bot players and players who never move have no rows.
    """
    t = np.asarray(run["t"], dtype=float)
    A = agreement(run)
    k = to_steps(sl.sign_correct(run))
    t_end = round_end(t, run["end"], run["session"])
    out = []
    for j in range(A.shape[1]):
        moves = np.flatnonzero(np.diff(k[:, j]) != 0)
        if j + 1 in run["bot_players"] or len(moves) == 0:
            continue
        t_first = t[moves[0] + 1]
        times = t_first + b * np.arange(int(np.ceil((t_end - t_first) / b - 1e-9)))
        rows = np.searchsorted(t, times + 1e-6, side="right") - 1
        out.append(pd.DataFrame({"file": run["file"], "wave": run["wave"], "player": j,
                                 "time": times, "row": rows, "A": A[rows, j]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLUMNS)

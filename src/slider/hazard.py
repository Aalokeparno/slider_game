"""Path A: at-risk time for reversal hazards.

A player is at risk of a reversal from their first move (the first row where k = round(10 s) changes;
in reversals.detect_events this move sets the direction) until fraught.round_end.
Agreement A_i is slider_logs.agreement / 100, so it lies in [0, 1].
build_hazard_table (P8): one row per at-risk bin, the input of Step 3.
"""
import numpy as np
import pandas as pd

from slider import interface
from slider import slider_logs as sl
from slider.crossings import BAND
from slider.fraught import round_end
from slider.labclock import lab_log
from slider.reversals import run_events, to_steps

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


# ----------------------------------------------------------------- P8: hazard table

LABELS = ["group_id", "split", "dataset", "topology", "interface", "round_speed", "round_id", "player", "t",
          "t_row", "bot_run", "incomplete"]
FEATURES = ["A", "trend", "trend_w05", "trend_w2", "trend_scaled", "abs_s", "at_end", "time_at_end",
            "time_at_end_log1p", "time_at_end_cap10", "time_since_rev", "time_since_rev_log1p", "time_since_rev_cap10"]


def hold_before(t, values, at):
    """values (one per row) at the last row strictly before each time in at; the first row's value if none."""
    rows = np.clip(np.searchsorted(t, np.asarray(at) - 1e-6, side="right") - 1, 0, None)
    return values[rows]


def bins_with_events(times, event_times, b):
    """1 for each bin [time, time + b) (times = bin starts) that holds some event time, else 0."""
    y = np.zeros(len(times), dtype=int)
    idx = np.searchsorted(times, np.asarray(event_times) + 1e-6, side="right") - 1
    ok = (idx >= 0) & (np.asarray(event_times) + 1e-6 < times[np.maximum(idx, 0)] + b)
    y[idx[ok]] = 1
    return y


def resolve_times(t, k, events, m, t_end):
    """When each event of ONE player (detect_events rows, same m) resolves: the row with the first move in the
    old direction (probe) or the row where the move reaches m steps (reversal); t_end if unresolved."""
    dk = np.diff(k, prepend=k[0])
    out = []
    for e in events.itertuples():
        if e.type == "unresolved":
            out.append(t_end)
            continue
        n = 0
        for i in range(e.event_row, len(k)):
            if dk[i] == 0:
                continue
            if np.sign(dk[i]) == e.dir_before:      # back in the old direction: a probe ends here
                break
            n += abs(dk[i])
            if n >= m:                              # m steps against it: a reversal ends here
                break
        out.append(t[i])
    return np.array(out, dtype=float)


def hazard_rows(run, b, speed, m):
    """The bins of one round for build_hazard_table (CC already on the lab clock), per player:
    (kept bins with player, t, t_row, y_rev, y_probe and the base features; dropped bins with player, t)."""
    t = np.asarray(run["t"], dtype=float)
    A = agreement(run)
    s = sl.sign_correct(run)
    k = to_steps(s)
    t_end = round_end(t, run["end"], run["session"])
    events = run_events(run, m)
    kept, dropped = [], []
    for j, smp in at_risk_samples(run, b).groupby("player", sort=True):
        times = smp["time"].to_numpy(float)
        ev = events[events["player"] == j]
        start, end = ev["time"].to_numpy(float), resolve_times(t, k[:, j], ev, m, t_end)
        is_rev = (ev["type"] == "reversal").to_numpy()
        mid = np.zeros(len(times), dtype=bool)                  # e < t <= r: the player is mid-event
        for e, r in zip(start, end):
            mid |= (times > e + 1e-6) & (times <= r + 1e-6)
        rows = np.clip(np.searchsorted(t, times - 1e-6, side="right") - 1, 0, None)   # last row before t
        band = np.sign(k[:, j]) * (np.abs(k[:, j]) >= BAND)                    # +1 / -1 in an end band, else 0
        entered = t[np.maximum.accumulate(np.where(np.r_[True, band[1:] != band[:-1]], np.arange(len(t)), 0))]
        anchors = np.r_[times[0], start[is_rev]]                    # first move, then the reversals' first steps
        resolved = np.r_[-np.inf, end[is_rev]]                      # ... and when each was resolved
        anchor = anchors[np.searchsorted(resolved, times - 1e-6, side="right") - 1]   # last resolved before t
        a, at_end = A[rows, j], band[rows] != 0
        bins = pd.DataFrame({
            "player": int(j), "t": times, "t_row": t[rows],
            "y_rev": bins_with_events(times, start[is_rev], b),
            "y_probe": bins_with_events(times, start[(ev["type"] == "probe").to_numpy()], b),
            "A": a, "trend": a - hold_before(t, A[:, j], times - 1), "trend_w05": a - hold_before(t, A[:, j], times - 0.5),
            "trend_w2": a - hold_before(t, A[:, j], times - 2),
            "trend_scaled": a - hold_before(t, A[:, j], times - 1 / speed),
            "abs_s": np.abs(s[rows, j]), "at_end": at_end.astype(int),
            "time_at_end": np.where(at_end, times - entered[rows], 0.0),
            "time_since_rev": times - anchor})
        kept.append(bins[~mid])
        dropped.append(bins.loc[mid, ["player", "t"]])
    return kept, dropped


def build_hazard_table(runs, b=0.25, m=2):
    """Step 3 input: one row per at-risk bin of a human player (at_risk_samples, bins of b seconds from the
    player's first move to fraught.round_end), without look-ahead. CC runs go through labclock.lab_log
    (tick 0.1 s); lab data as logged. Events: reversals.run_events with m (probes, reversals, unresolved).

    Steady at-risk set: an event starts at e (its first step against the player's direction) and resolves at r
    (resolve_times; the round end if unresolved). Bins with e < t <= r are dropped: the player is mid-event.
    y_rev    1 if a reversal's first step is in [t, t + b)
    y_probe  1 if a probe's first step is in [t, t + b)
    Features at the bin start t, all from the last row strictly before t (its time is t_row):
      A             Agreement of the player (agreement; every seat, bots included, enters it)
      trend         A(t-) - A((t - 1 s)-); before the round's first row, A at the first row.
                    trend_w05, trend_w2: the same over 0.5 s and 2 s; trend_scaled: over 1 / round_speed s
      abs_s         |s|;  at_end: 1 if |k| >= 9
      time_at_end   seconds since entering the current end band (reset on leaving it); 0 when not in one
      time_since_rev  seconds since the first step of the player's last reversal resolved before t,
                    or since the player's first move if there is none
      *_log1p, *_cap10  log(1 + x) and min(x, 10) of the two time features
    Labels: group_id (= file), split (split G), dataset, topology, interface (lab_v09 / old_cl / cc), round_speed,
    round_id (file#w<wave>), player (0-based), t, t_row, bot_run (a robot played in the round), incomplete.
    round_speed (units/s, raw rows, interface.py): lab = the round's own step size x modal pace; CC = the file's
    speed (interface_table); the file's speed also when the round has no step size or modal pace.
    Returns (table, dropped): dropped = the mid-event bins, with the labels but no t_row.
    """
    files = interface.interface_table(runs).set_index("file")
    kept, dropped = [], []
    for run in runs:
        f = files.loc[run["file"]]
        step, pace = interface.step_and_pace(interface.moves(run))
        speed = f["speed"] if run["dataset"] == "CC" or np.isnan(step * pace) else step * pace
        kind = "cc" if run["dataset"] == "CC" else "lab_v09" if f["version"] == "v09" else "old_cl"
        if run["dataset"] == "CC":
            t_lab, X_lab = lab_log(run["t"], run["X"])
            run = {**run, "t": t_lab, "X": X_lab}
        labels = {"group_id": run["file"], "split": f["split"], "dataset": run["dataset"],
                  "topology": sl.label_topology(run["A"]), "interface": kind, "round_speed": speed,
                  "round_id": f"{run['file']}#w{run['wave']}", "bot_run": bool(run["bot"]),
                  "incomplete": run["end"] == "incomplete"}
        k_parts, d_parts = hazard_rows(run, b, speed, m)
        kept += [p.assign(**labels) for p in k_parts]
        dropped += [p.assign(**labels) for p in d_parts]
    out = pd.concat(kept, ignore_index=True)
    for c in ["time_at_end", "time_since_rev"]:
        out[c + "_log1p"] = np.log1p(out[c])
        out[c + "_cap10"] = np.minimum(out[c], 10)
    dropped = pd.concat(dropped, ignore_index=True)
    return out[LABELS + ["y_rev", "y_probe"] + FEATURES], dropped[[c for c in LABELS if c != "t_row"]]

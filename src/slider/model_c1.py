"""Model C, C1 (EXPLORATORY): the data side. Trip durations of human seats (results/model_b_log.md, entry 2).

Positions: the logged rows, sign-corrected, with the t = 0, s = 0 row first (as motion_spells). A seat is at a wall
on a row with |s| = 1 (side = sign(s)).
    first trip   from the seat's first own step (the first row where its s changes) to its first row at a wall:
                 duration = the two rows' time difference; wall = the side reached; censored if no wall is reached
    departure    a row where the seat leaves a wall (the previous row at a wall, this one not at that wall):
                 "arrived" if its next row at a wall is the opposite wall (a wall-to-wall trip, duration = that
                 row's time - the departure row's time, pauses and wobbles included), "returned" if it is the
                 starting wall, "censored" if the round ends first
D2 (descriptive): the position just before each reversal's first step (reversals.detect_events, m = 2), in the end
band (stop_states.band_side: |s| >= 0.9, old CL |k| >= 9) or mid-slider.
"""
import numpy as np
import pandas as pd

from slider import config, interface, motion
from slider import slider_logs as sl
from slider.reversals import detect_events
from slider.stop_states import band_side

WALL = 1 - 1e-9


def seat_trips(t, s):
    """The first trip and the departures of ONE seat (t, s with the t = 0, s = 0 row first). Returns a list of dicts:
    kind ("first" / "cross"), outcome ("arrived" / "returned" / "censored"), start, duration (NaN unless arrived),
    wall (the side reached; NaN unless arrived), origin (the wall left; 0 for a first trip)."""
    wall = np.where(np.abs(s) >= WALL, np.sign(s), 0).astype(int)
    at = np.flatnonzero(wall != 0)
    out = []
    moved = np.flatnonzero(np.abs(np.diff(s)) > 1e-9) + 1
    if len(moved):
        i = moved[0]
        hit = at[at >= i]
        out.append({"kind": "first", "outcome": "arrived" if len(hit) else "censored", "start": t[i],
                    "duration": t[hit[0]] - t[i] if len(hit) else np.nan,
                    "wall": wall[hit[0]] if len(hit) else np.nan, "origin": 0})
    for i in np.flatnonzero((wall[:-1] != 0) & (wall[1:] != wall[:-1])) + 1:
        origin = wall[i - 1]
        hit = at[at >= i]
        if not len(hit):
            out.append({"kind": "cross", "outcome": "censored", "start": t[i], "duration": np.nan, "wall": np.nan,
                        "origin": origin})
            continue
        arrived = wall[hit[0]] == -origin
        out.append({"kind": "cross", "outcome": "arrived" if arrived else "returned", "start": t[i],
                    "duration": t[hit[0]] - t[i] if arrived else np.nan, "wall": wall[hit[0]] if arrived else np.nan,
                    "origin": origin})
    return out


def seat_reversals(t, s, old_cl, m=2):
    """D2: the reversals of ONE seat: start (the first step's time), s_before (the position just before it),
    end_band (1 if s_before is in an end band)."""
    ev = detect_events(t, s, m)
    rows = ev.loc[ev["type"] == "reversal", "event_row"].to_numpy()
    return [{"start": t[i], "s_before": s[i - 1], "end_band": int(band_side(s[i - 1], old_cl) != 0)} for i in rows]


def tables(runs):
    """(trips, reversals) of every human seat: seat_trips and seat_reversals rows with file, wave, dataset,
    interface, dt (motion.round_iface) and player (0-based seat)."""
    files = interface.interface_table(runs).set_index("file")
    trips, revs = [], []
    for run in runs:
        iface = motion.round_iface(run, files.loc[run["file"]])
        t = np.r_[0.0, run["t"]]
        s = np.vstack([np.zeros(len(run["weights"])), sl.sign_correct(run)])
        labels = {"file": run["file"], "wave": run["wave"], "dataset": run["dataset"],
                  "interface": iface["interface"], "dt": iface["dt"]}
        for j in range(s.shape[1]):
            if j + 1 in run["bot_players"]:
                continue
            trips += [{**labels, "player": j, **r} for r in seat_trips(t, s[:, j])]
            revs += [{**labels, "player": j, **r} for r in seat_reversals(t, s[:, j], iface["interface"] == "old_cl")]
    return pd.DataFrame(trips), pd.DataFrame(revs)


def durations(iface, path=None):
    """C1: {"first": first-trip durations, "cross": wall-to-wall durations} (s, sorted) of the arrived trips of an
    interface in pC_durations.parquet (path: another file)."""
    d = pd.read_parquet(path or config.PC_DURATIONS)
    d = d[(d["interface"] == iface) & (d["outcome"] == "arrived")]
    return {kind: np.sort(d.loc[d["kind"] == kind, "duration"].to_numpy(float)) for kind in ["first", "cross"]}


def summary(trips):
    """Per interface and kind: counts by outcome, censored share (of all), and the quartiles of the arrived durations."""
    rows = []
    for (name, kind), d in trips.groupby(["interface", "kind"], sort=False):
        a = d.loc[d["outcome"] == "arrived", "duration"]
        q = a.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_numpy()
        rows.append({"interface": name, "kind": kind, "n": len(d), "arrived": len(a),
                     "returned": int((d["outcome"] == "returned").sum()),
                     "censored": int((d["outcome"] == "censored").sum()),
                     "censored_share": (d["outcome"] == "censored").mean(),
                     "q10": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q90": q[4]})
    out = pd.DataFrame(rows)
    out["interface"] = pd.Categorical(out["interface"], motion.INTERFACES)
    return out.sort_values(["kind", "interface"], ascending=[False, True]).reset_index(drop=True)

"""Path A, target 10: crossings of a player's slider from one end band to the other.

Positions in integer steps k = round(10 s), s sign-corrected. End bands: k >= 9 and k <= -9.
Both directions count.
"""
import numpy as np
import pandas as pd

from slider.reversals import detect_events

BAND = 9      # end band: |k| >= BAND
JUMP = 4      # a jump: one raw row with |Δk| >= JUMP
COLUMNS = ["start", "end", "status", "direction", "has_jump", "duration", "moving", "pause_time", "steps", "n_pauses"]


def crossing_events(t, k, raw_t, raw_k, pause=0.3):
    """Crossings of ONE player's slider, one row per crossing.

    t, k          the rows used (lab-clocked for CC) and the player's k on them
    raw_t, raw_k  the raw rows and the player's k on them (only for has_jump)
    start   time of the first step out of an end band (the first row outside it)
    end     the crossing stops at the first of
              "complete"    the first step into the opposite band (end = arrival)
              "aborted"     a reversal away from the target (detect_events, m = 2; probes allowed);
                            end = the reversal's first step
              "returned"    a step back into the band it started from; end = that step.
                            The next step out starts a new crossing.
              "unfinished"  the round ends first (no arrival by the last row); end = last row
            (an aborted and a returned at the same row count as aborted)
    direction  +1 or -1: the side of the target band
    has_jump   some raw row in (time of the last row before start, end] has |Δk| >= 4
    Complete crossings only: duration = end - start. The gaps between the player's consecutive step rows
    (from start to end) longer than pause are pause time, the others moving time;
    steps = sum of |Δk| over the rows that close a moving gap; n_pauses = number of pause gaps.
    """
    t, k = np.asarray(t, dtype=float), np.asarray(k, dtype=int)
    raw_t, raw_k = np.asarray(raw_t, dtype=float), np.asarray(raw_k, dtype=int)
    rev = detect_events(t, k / 10, m=2)
    rev = rev[rev["type"] == "reversal"]
    raw_jump = np.append(False, np.abs(np.diff(raw_k)) >= JUMP)      # one flag per raw row
    out = []
    i = 1
    while i < len(k):
        origin = np.sign(k[i - 1])
        if not (origin * k[i - 1] >= BAND and origin * k[i] < BAND):   # no step out of a band at row i
            i += 1
            continue
        d = -origin
        arrive = np.flatnonzero(d * k[i:] >= BAND)
        back = np.flatnonzero(origin * k[i + 1:] >= BAND)
        abort = rev.loc[(rev["dir_before"] == d) & (rev["event_row"] > i), "event_row"]
        stops = {"complete": i + arrive[0] if len(arrive) else np.inf,
                 "aborted": abort.min() if len(abort) else np.inf,
                 "returned": i + 1 + back[0] if len(back) else np.inf}
        status = min(stops, key=lambda s: stops[s])                     # first in this order on a tie
        if stops[status] == np.inf:
            status, stop = "unfinished", len(k) - 1
        else:
            stop = int(stops[status])
        inside = (raw_t > t[i - 1] + 1e-6) & (raw_t <= t[stop] + 1e-6)
        row = {"start": t[i], "end": t[stop], "status": status, "direction": int(d),
               "has_jump": bool(np.any(raw_jump & inside))}
        if status == "complete":
            step_rows = i + np.flatnonzero(np.diff(k[i - 1:stop + 1]) != 0)      # own step rows, start..end
            gaps = np.diff(t[step_rows])
            is_pause = gaps > pause + 1e-6
            dk = np.abs(np.diff(k))[step_rows[1:] - 1]                            # |Δk| of the row closing each gap
            row.update({"duration": t[stop] - t[i], "moving": gaps[~is_pause].sum(),
                        "pause_time": gaps[is_pause].sum(), "steps": int(dk[~is_pause].sum()),
                        "n_pauses": int(is_pause.sum())})
        out.append(row)
        i = stop + 1
    return pd.DataFrame(out, columns=COLUMNS)

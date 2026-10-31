"""Model B, B0: where do humans stop? One row per counted moving step, with causal state labels.

Counted moving steps and stops are motion.py's F1b classification, recomputed here row by row in the same way
(motion_spells): a row with |Δs| holds k = max(1, round(|Δs| / step)) steps; it is counted if it is not
mid-event and does not land on the wall; a stop is a halt in halted time right after a counted row.
Each row is split into its k steps (positions on the line from the previous s to the new s); a stop belongs
to the last step of the row. Descriptive only.

Labels (from the past only, except outcome):
  end band   |s| >= 0.9; old CL: |k| >= 9 with k = round(10 s) (crossings.BAND, as in hazard.py)
  phase      first_trip  no end band reached yet in this round
             outbound    left an end band, no end band reached since
             in_band     the step lands in an end band (not first_trip or outbound)
  phi        first_trip: the sign of the seat's first move; outbound: away from the band it left
  progress   s * phi (NaN in_band)
  reversed   1 if a reversal (detect_events, m = 2) starts after the phase's first row and before this step
  outcome    outbound only: "completed" (reaches the opposite band), "aborted" (back to the same band),
             "censored" (the round ends first)
"""
import numpy as np
import pandas as pd

from slider import interface, motion
from slider import slider_logs as sl
from slider.crossings import BAND
from slider.fraught import round_end
from slider.hazard import resolve_times
from slider.reversals import detect_events, to_steps


def band_side(p, old_cl):
    """+1 / -1 if position p is in an end band, else 0."""
    inside = abs(to_steps(p)) >= BAND if old_cl else abs(p) >= 0.9 - 1e-9
    return int(np.sign(p)) if inside else 0


def seat_steps(t, s, dt, step, t_end, old_cl, m=2):
    """The steps of ONE human seat (t, s with the t = 0, s = 0 row first). Returns a list of dicts."""
    ds = np.diff(s)
    rows = np.flatnonzero(np.abs(ds) > 1e-9) + 1
    n_steps = np.maximum(1, np.round(np.abs(ds[rows - 1]) / step)).astype(int)
    ev = detect_events(t, s, m)
    starts = ev["event_row"].to_numpy()
    ends = resolve_times(np.arange(len(t), dtype=float), to_steps(s), ev, m, np.inf)
    rev_rows = ev.loc[ev["type"] == "reversal", "event_row"].to_numpy()

    out = []
    phase, origin, phase_row, seg, seg_first = "first_trip", 0, 0, -1, 0
    phi = np.sign(ds[rows[0] - 1]) if len(rows) else 0
    for n, (i, ks) in enumerate(zip(rows, n_steps)):
        counted = not np.any((starts <= i) & (i <= ends)) and not np.isclose(abs(s[i]), 1)
        # stop: the halt after this row is in halted time (same tests as motion_spells.halt)
        nxt = rows[n + 1] if n + 1 < len(rows) else len(t)
        g = (t[nxt] - t[i] - n_steps[n + 1] * dt) if n + 1 < len(rows) else t_end - t[i]
        stop = counted and g > dt / 2 and not np.any((starts < nxt) & (nxt <= ends))
        for u in range(ks):
            p = s[i] if u == ks - 1 else s[i - 1] + (u + 1) / ks * ds[i - 1]
            side = band_side(p, old_cl)
            if side != 0:
                if phase == "outbound":
                    for r in out[seg_first:]:
                        r["outcome"] = "completed" if side == -origin else "aborted"
                phase, origin = "in_band", side
            elif phase == "in_band":
                phase, phase_row, seg, phi, seg_first = "outbound", i, seg + 1, -origin, len(out)
            if counted:
                out.append({"row": i, "s": p, "phase": phase, "dt": dt,
                            "progress": p * phi if phase != "in_band" else np.nan,
                            "reversed": int(np.any((rev_rows > phase_row) & (rev_rows < i))),
                            "segment": seg if phase == "outbound" else -1,
                            "outcome": "censored" if phase == "outbound" else "",
                            "stop": int(stop and u == ks - 1)})
    return out


def step_table(runs, m=2):
    """One row per counted moving step of every human seat: file, wave, dataset, interface, player and the
    seat_steps columns (segment ids are per seat)."""
    files = interface.interface_table(runs).set_index("file")
    out = []
    for run in runs:
        iface = motion.round_iface(run, files.loc[run["file"]])
        t = np.r_[0.0, run["t"]]
        s = np.vstack([np.zeros(len(run["weights"])), sl.sign_correct(run)])
        t_end = round_end(run["t"], run["end"], run["session"])
        for j in range(s.shape[1]):
            if j + 1 in run["bot_players"]:
                continue
            rows = seat_steps(t, s[:, j], iface["dt"], iface["step"], t_end,
                              iface["interface"] == "old_cl", m)
            labels = {"file": run["file"], "wave": run["wave"], "dataset": run["dataset"],
                      "interface": iface["interface"], "player": j}
            out += [{**labels, **r} for r in rows]
    return pd.DataFrame(out)

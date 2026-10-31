"""Model B, B1 (EXPLORATORY): the data side. States C / E of every own step and halt, and the B1 parameters.

States (causal), from the seat's path:
    C   crossing attempt: left an end band, no reversal since, no end band reached since
        (stop_states labels: phase outbound, reversed 0)
    E   elsewhere: first trip, outbound after a reversal, in_band (not at the wall)
A halt (P17 F1b halted time: mid-slider, after the first move, not mid-event, not at the wall) gets the state of
the row that starts it. A reversal counts for the state of a row if its first step is after the phase's first row
and at or before the row (stop_states uses "before" for counted steps; a counted step never opens an event).
Halt length in ticks n = max(1, round(duration / dt)) (halts.add_ticks); resume chance per tick = resumes / halted
ticks (one geometric: halts.py with p = 1).
"""
import numpy as np
import pandas as pd

from slider import halts as halts_mod
from slider import interface, model_b, motion, stop_states
from slider import slider_logs as sl
from slider.fraught import round_end
from slider.hazard import resolve_times
from slider.reversals import detect_events, to_steps

INTERFACES = motion.INTERFACES


def seat_halts(t, s, dt, step, t_end, old_cl, m=2):
    """The halts of ONE human seat (t, s with the t = 0, s = 0 row first), as motion_spells finds them, with the
    state (C / E) of the row before each. Returns a list of dicts: start, duration, exit, state."""
    ds = np.diff(s)
    rows = np.flatnonzero(np.abs(ds) > 1e-9) + 1
    n_steps = np.maximum(1, np.round(np.abs(ds[rows - 1]) / step)).astype(int)
    ev = detect_events(t, s, m)
    starts = ev["event_row"].to_numpy()
    ends = resolve_times(np.arange(len(t), dtype=float), to_steps(s), ev, m, np.inf)
    opens = dict(zip(starts, ev["type"]))
    rev_rows = ev.loc[ev["type"] == "reversal", "event_row"].to_numpy()
    out = []
    phase, phase_row = "first_trip", 0
    for n, (i, ks) in enumerate(zip(rows, n_steps)):
        for u in range(ks):                                    # the phase, step by step (as stop_states)
            p = s[i] if u == ks - 1 else s[i - 1] + (u + 1) / ks * ds[i - 1]
            side = stop_states.band_side(p, old_cl)
            if side != 0:
                phase = "in_band"
            elif phase == "in_band":
                phase, phase_row = "outbound", i
        reversed_ = np.any((rev_rows > phase_row) & (rev_rows <= i))
        state = "C" if phase == "outbound" and not reversed_ else "E"
        nxt = rows[n + 1] if n + 1 < len(rows) else len(t)
        g = (t[nxt] - t[i] - n_steps[n + 1] * dt) if n + 1 < len(rows) else t_end - t[i]
        if g <= dt / 2 or np.any((starts < nxt) & (nxt <= ends)) or np.isclose(abs(s[i]), 1):
            continue                                           # no halt, mid-event or at the wall (motion_spells)
        exit = opens.get(nxt, "resume") if nxt < len(t) else "round_end"
        out.append({"start": t[i], "duration": g, "exit": exit, "state": state})
    return out


def halt_table(runs, m=2):
    """seat_halts of every human seat: file, wave, dataset, interface, dt, player, start, duration, exit, state,
    n (ticks), event (1 = resume)."""
    files = interface.interface_table(runs).set_index("file")
    out = []
    for run in runs:
        iface = motion.round_iface(run, files.loc[run["file"]])
        t = np.r_[0.0, run["t"]]
        s = np.vstack([np.zeros(len(run["weights"])), sl.sign_correct(run)])
        t_end = round_end(run["t"], run["end"], run["session"])
        labels = {"file": run["file"], "wave": run["wave"], "dataset": run["dataset"],
                  "interface": iface["interface"], "dt": iface["dt"]}
        for j in range(s.shape[1]):
            if j + 1 in run["bot_players"]:
                continue
            for h in seat_halts(t, s[:, j], iface["dt"], iface["step"], t_end, iface["interface"] == "old_cl", m):
                out.append({**labels, "player": j, **h})
    return halts_mod.add_ticks(pd.DataFrame(out))


def step_state(steps):
    """The B0 step table with state C / E."""
    c = (steps["phase"] == "outbound") & (steps["reversed"] == 0)
    return steps.assign(state=np.where(c, "C", "E"))


def estimates(runs):
    """Point estimates of the B1 parameters per interface, with the code of steps 0-1:
    pi, rho (beta-binomial over players with >= 3 completed crossings), q_sg (stops / steps in non-smooth
    completed crossings), q_E (stops / counted steps in state E), resume_C, resume_E, resume_all (resumes / halted
    ticks). Returns (DataFrame interface x parameter, the tables used: steps, crossings, halts)."""
    steps = step_state(stop_states.step_table(runs))
    cross = model_b.crossing_table(runs, steps)
    halts = halt_table(runs)
    players = model_b.player_shares(cross)
    rows = []
    for name in INTERFACES:
        p, c = players[players["interface"] == name], cross[cross["interface"] == name]
        st, h = steps[steps["interface"] == name], halts[halts["interface"] == name]
        pi, rho = model_b.betabinom_fit(p["smooth"].to_numpy(), p["n"].to_numpy())
        sg, e = c[c["smooth"] == 0], st[st["state"] == "E"]
        hc, he = h[h["state"] == "C"], h[h["state"] == "E"]
        rows.append({"interface": name, "pi": pi, "rho": rho, "q_sg": sg["stops"].sum() / sg["steps"].sum(),
                     "q_E": e["stop"].sum() / len(e), "resume_C": hc["event"].sum() / hc["n"].sum(),
                     "resume_E": he["event"].sum() / he["n"].sum(), "resume_all": h["event"].sum() / h["n"].sum()})
    return pd.DataFrame(rows).set_index("interface"), {"steps": steps, "cross": cross, "halts": halts}

"""Path A, P17: stop, resume and start rates of the Step 4 simulator, from each human seat's own steps.

Simulator states: halted (also the initial halt at s = 0), moving (one step per tick dt), at the wall (|s| = 1),
reversing (2 forced steps). Moving <-> halted runs on two constant rates, lam_stop and lam_resume;
the first move is the first resume.

A seat's path starts at t = 0 with s = 0 (the round clock starts at 0; lab files log a row only on a change,
so the first row can come seconds later). Own steps: rows where the seat's s changes; a change of |Δs| holds
k = round(|Δs| / step) steps (at least 1), in the file's step units.
Halt: a gap g between consecutive own steps, followed by k steps, holds a halt of g - k*dt (counted if > dt/2).
After the last own step, the gap to fraught.round_end (k = 0) is a halt the same way.
Start delay: the gap from t = 0 to the first own step, minus its k*dt (at least 0).
Events (probes, reversals) come from reversals.detect_events on k = round(10 s), m = 2. Mid-event: from the
first step against the direction until hazard.resolve_times (to the round end if unresolved). In old CL
(0.02 steps) a step back that does not change round(10 s) is not an event.

Time of a seat, from t = 0 to round_end:
    pre        before the first move (the start delay; all of it for a seat that never moves)
    mid        steps and halts inside a mid-event span
    wall       halts at |s| = 1, and steps that land on |s| = 1 (not mid-event)
    halted     halts mid-slider after the first move, not mid-event  (resume at-risk time)
    moving     the other steps, k*dt each                            (stop at-risk time)
    residual   the rest: gaps g with 0 < g - k*dt <= dt/2 (no halt) and g < k*dt (steps faster than dt)
A step that lands on the wall is wall time, not stop exposure: a halt after it is at the wall and is never a stop
(F1b, Oct 29 2026, per the locked spec "stop exposure excludes time at the wall"; old rates: p17_rates_v1.csv).
"""
import numpy as np
import pandas as pd

from slider import config
from slider import interface
from slider import slider_logs as sl
from slider.fraught import round_end
from slider.hazard import resolve_times
from slider.reversals import detect_events, to_steps

INTERFACES = ["lab_v09", "old_cl", "cc"]
OLD_CL_DT, OLD_CL_STEP = 0.1, 0.02
SEAT_COLUMNS = ["player", "total_s", "pre_s", "moving_s", "halted_s", "wall_s", "mid_s", "residual_s",
                "stops", "resumes", "moved", "start_delay", "first_dir", "nbr_side"]
HALT_COLUMNS = ["player", "start", "duration", "exit"]
RATES = {"lam_stop": ("stops", "moving_s"), "lam_resume": ("resumes", "halted_s"), "lam_start": ("moved", "pre_s")}


def round_iface(run, file_row):
    """{interface, dt, step} of one round; file_row = the file's row of interface.interface_table.
    lab_v09: dt = the round's tick, step / round speed (round speed = its own step size x modal pace, the
    file's speed if the round has none); old_cl: dt = 0.1 s, 0.02 steps; cc (raw rows): dt = 1 / file pace."""
    if run["dataset"] == "CC":
        return {"interface": "cc", "dt": 1 / file_row["modal_pace"], "step": file_row["step_size"]}
    if file_row["version"] != "v09":
        return {"interface": "old_cl", "dt": OLD_CL_DT, "step": OLD_CL_STEP}
    step, pace = interface.step_and_pace(interface.moves(run))
    speed = file_row["speed"] if np.isnan(step * pace) else step * pace
    return {"interface": "lab_v09", "dt": file_row["step_size"] / speed, "step": file_row["step_size"]}


def neighbor_side(A_row, s_row):
    """sign(sum of sign(s_j)) over the neighbors j with s_j != 0; NaN if none has moved or on a tie."""
    v = s_row[(A_row > 0) & (np.abs(s_row) > 1e-9)]
    side = np.sign(np.sum(np.sign(v)))
    return float(side) if side != 0 else np.nan


def motion_spells(run, iface, m=2):
    """Per human seat of one round: time by state, stops, resumes and the first move; and the halt spells.

    Returns (seats, halts).
    seats: one row per human seat: player (0-based), total_s (t = 0 to round_end), pre_s, moving_s, halted_s,
      wall_s, mid_s, residual_s (see the module doc), stops (halts in halted_s right after a step in moving_s),
      resumes, moved (1 if the seat moves), start_delay (NaN if it never moves), first_dir (+1 / -1),
      nbr_side (neighbor_side just before the first move; bot seats count as neighbors).
    halts: the halts in halted_s: player, start, duration, exit = "resume" (the next own step opens no event),
      "probe" / "reversal" / "unresolved" (the next own step opens that event) or "round_end".
    """
    dt, step = iface["dt"], iface["step"]
    t = np.r_[0.0, run["t"]]
    s = np.vstack([np.zeros(len(run["weights"])), sl.sign_correct(run)])
    k = to_steps(s)
    t_end = round_end(run["t"], run["end"], run["session"])
    seats, halts = [], []
    for j in range(s.shape[1]):
        if j + 1 in run["bot_players"]:
            continue
        ds = np.diff(s[:, j])
        rows = np.flatnonzero(np.abs(ds) > 1e-9) + 1                          # own steps
        n_steps = np.maximum(1, np.round(np.abs(ds[rows - 1]) / step)).astype(int)
        ev = detect_events(t, s[:, j], m)
        starts = ev["event_row"].to_numpy()
        ends = resolve_times(np.arange(len(t), dtype=float), k[:, j], ev, m, np.inf)   # rows; inf = unresolved
        opens = dict(zip(starts, ev["type"]))
        out = dict.fromkeys(["pre_s", "moving_s", "halted_s", "wall_s", "mid_s", "stops", "resumes"], 0.0)
        out.update(player=j, total_s=t_end, moved=int(len(rows) > 0), start_delay=np.nan,
                   first_dir=np.nan, nbr_side=np.nan)

        def halt(i, prev, g, moving_before):
            """Halt in the gap g after own step prev, ended by own step i (len(t) = the round end)."""
            if g <= dt / 2:
                return
            if np.any((starts < i) & (i <= ends)):
                out["mid_s"] += g
            elif np.isclose(abs(s[prev, j]), 1):
                out["wall_s"] += g
            else:
                exit = opens.get(i, "resume") if i < len(t) else "round_end"
                out["halted_s"] += g
                out["stops"] += moving_before
                out["resumes"] += exit == "resume"
                halts.append({"player": j, "start": t[prev], "duration": g, "exit": exit})

        moving_before = False                       # was the last own step in moving_s?
        for n, (i, ks) in enumerate(zip(rows, n_steps)):
            g = t[i] - t[rows[n - 1]] if n > 0 else t[i]
            if n == 0:
                out["pre_s"] = out["start_delay"] = max(g - ks * dt, 0.0)
                out["first_dir"] = float(np.sign(ds[i - 1]))
                out["nbr_side"] = neighbor_side(run["A"][j], s[i - 1])
            else:
                halt(i, rows[n - 1], g - ks * dt, moving_before)
            mid = bool(np.any((starts <= i) & (i <= ends)))
            on_wall = np.isclose(abs(s[i, j]), 1)                 # a step that lands on the wall: wall time (F1b)
            moving_before = not mid and not on_wall
            out["mid_s" if mid else "wall_s" if on_wall else "moving_s"] += ks * dt
        if len(rows) == 0:
            out["pre_s"] = t_end
        else:
            halt(len(t), rows[-1], t_end - t[rows[-1]], moving_before)
        out["residual_s"] = t_end - sum(out[c] for c in ["pre_s", "moving_s", "halted_s", "wall_s", "mid_s"])
        seats.append(out)
    return pd.DataFrame(seats, columns=SEAT_COLUMNS), pd.DataFrame(halts, columns=HALT_COLUMNS)


def motion_table(runs, m=2):
    """motion_spells for every round, with labels file (= group), wave, dataset, interface and dt.
    Returns (seats, halts)."""
    files = interface.interface_table(runs).set_index("file")
    seats, halts = [], []
    for run in runs:
        iface = round_iface(run, files.loc[run["file"]])
        labels = {"file": run["file"], "wave": run["wave"], "dataset": run["dataset"],
                  "interface": iface["interface"], "dt": iface["dt"]}
        st, ht = motion_spells(run, iface, m)
        seats.append(st.assign(**labels))
        halts.append(ht.assign(**labels))
    return pd.concat(seats, ignore_index=True), pd.concat([h for h in halts if len(h)], ignore_index=True)


def boot_ratios(df, pairs, n_boot=1000, seed=config.SEED):
    """95% CIs of sum(num) / sum(den) for each (num, den) in pairs: n_boot resamples of the groups (files),
    drawn with replacement within each dataset (EL, CL, CC), as many as in the data. Returns (low, high) arrays."""
    cols = [c for pair in pairs for c in pair]
    g = df.groupby("file")[cols].sum()
    dataset = df.groupby("file")["dataset"].first()
    strata = [np.flatnonzero(dataset.to_numpy() == d) for d in interface.DATASETS if (dataset == d).any()]
    sums = g.to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([st[rng.integers(0, len(st), len(st))] for st in strata])
        tot = sums[idx].sum(axis=0)
        boot.append([tot[2 * p] / tot[2 * p + 1] for p in range(len(pairs))])
    return np.percentile(np.array(boot), [2.5, 97.5], axis=0)


def by_interface(df):
    """(name, rows) for each interface, then "shared" (all rows)."""
    return [(name, df[df["interface"] == name]) for name in INTERFACES] + [("shared", df)]


def rates_table(runs, m=2, n_boot=1000, seed=config.SEED):
    """lam_stop = stops / moving seconds, lam_resume = resumes / halted seconds (other exits censored),
    lam_start = first moves / pre-first-move seconds; per interface (lab_v09, old_cl, cc) and shared.
    95% CI: boot_ratios. Columns: interface, rate, events, exposure_s, estimate, ci_low, ci_high, groups, seats."""
    seats, _ = motion_table(runs, m)
    out = []
    for name, d in by_interface(seats):
        low, high = boot_ratios(d, list(RATES.values()), n_boot, seed)
        for p, (rate, (num, den)) in enumerate(RATES.items()):
            out.append({"interface": name, "rate": rate, "events": int(d[num].sum()), "exposure_s": d[den].sum(),
                        "estimate": d[num].sum() / d[den].sum(), "ci_low": low[p], "ci_high": high[p],
                        "groups": d["file"].nunique(), "seats": len(d)})
    return pd.DataFrame(out)


def follow_table(seats, n_boot=1000, seed=config.SEED):
    """Among first moves with a defined neighbor majority (nbr_side not NaN): the share toward it,
    per interface and shared, with the 95% CI of boot_ratios."""
    d = seats[seats["nbr_side"].notna()].assign(defined=1)
    d = d.assign(toward=(d["first_dir"] == d["nbr_side"]).astype(int))
    out = []
    for name, x in by_interface(d):
        (low,), (high,) = boot_ratios(x, [("toward", "defined")], n_boot, seed)
        out.append({"interface": name, "first_moves": len(x), "toward": int(x["toward"].sum()),
                    "share": x["toward"].mean(), "ci_low": low, "ci_high": high, "groups": x["file"].nunique()})
    return pd.DataFrame(out)


def time_check(seats, tol=0.01):
    """Per interface and shared: seconds by state, total (t = 0 to round_end, human seats),
    the share of the total they leave out (residual) and ok = |residual share| <= tol."""
    cols = ["pre_s", "moving_s", "halted_s", "wall_s", "mid_s"]
    out = []
    for name, d in by_interface(seats):
        row = {"interface": name, **d[cols].sum().to_dict(), "total_s": d["total_s"].sum()}
        row["residual_share"] = 1 - sum(row[c] for c in cols) / row["total_s"]
        row["ok"] = abs(row["residual_share"]) <= tol
        out.append(row)
    return pd.DataFrame(out)

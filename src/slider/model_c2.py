"""Model C, C2 (EXPLORATORY): the data side, step 0 (results/model_b_log.md, entry 3).

Positions: sign-corrected s with the t = 0, s = 0 row first (as motion_spells), human seats only. An own-step row is
a row where the seat's s changes; it holds k = max(1, round(|ds| / step)) grid steps (motion.round_iface: dt, step).
    (a) first_step_gaps   from rest = a 1-step own row after a halt (gap to the previous own row - dt > dt / 2; the
                          first move counts from t = 0); gap = time to the next own row if it moves the same way
    (b) try_bins          the hazard table's G-train bins at the wall (abs_s == 1), with y_try (the seat's first
                          estimator row off the wall in [t, t + 0.25)) and g = log(1 + time since a neighbor's last step)
    (c) walk_gaps, q_w    gaps between own steps inside excursions that return to their wall
    (e) paces             the 10a crossings (complete, no jump) on the raw rows: v = |ds| / duration, f = v / round speed
    (f) wall_time, round_summary   all-at-wall time and standoffs per round
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from slider import config, interface, model_b, motion, standoffs
from slider import slider_logs as sl
from slider.crossings import crossing_events
from slider.fraught import round_end
from slider.labclock import lab_log
from slider.reversals import to_steps

WALL = 1 - 1e-9
TOL = 0.05
BIN = 0.25          # hazard.py bins
D_GRID = np.round(np.arange(0, 2.001, 0.01), 2)


def seat_paths(runs):
    """(labels, t, s) of every human seat: t and s with the t = 0, s = 0 row first; labels: file, wave, dataset,
    interface, dt, step, speed (= step / dt), player."""
    files = interface.interface_table(runs).set_index("file")
    for run in runs:
        iface = motion.round_iface(run, files.loc[run["file"]])
        t = np.r_[0.0, run["t"]]
        s = np.vstack([np.zeros(len(run["weights"])), sl.sign_correct(run)])
        labels = {"file": run["file"], "wave": run["wave"], "dataset": run["dataset"],
                  "topology": sl.label_topology(run["A"]), "interface": iface["interface"], "dt": iface["dt"],
                  "step": iface["step"], "speed": iface["step"] / iface["dt"]}
        for j in range(s.shape[1]):
            if j + 1 not in run["bot_players"]:
                yield {**labels, "player": j}, t, s[:, j]


def own_rows(s, step):
    """Own-step rows of one seat and the grid steps k each holds."""
    rows = np.flatnonzero(np.abs(np.diff(s)) > 1e-9) + 1
    return rows, np.maximum(1, np.round(np.abs(s[rows] - s[rows - 1]) / step)).astype(int)


# ------------------------------------------------------------------ (a) auto-repeat

def seat_first_gaps(t, s, dt, step):
    """(a) for ONE seat: one dict per movement from rest: start, origin (wall / mid), gap (NaN if left out),
    left_out ("" / "reversed" / "round_end")."""
    rows, k = own_rows(s, step)
    out = []
    for p, (i, ki) in enumerate(zip(rows, k)):
        prev = t[rows[p - 1]] if p > 0 else 0.0
        if ki != 1 or t[i] - prev - dt <= dt / 2:
            continue
        rec = {"start": t[i], "origin": "wall" if abs(s[i - 1]) >= WALL else "mid", "gap": np.nan, "left_out": ""}
        if p + 1 == len(rows):
            rec["left_out"] = "round_end"
        elif np.sign(s[rows[p + 1]] - s[rows[p + 1] - 1]) != np.sign(s[i] - s[i - 1]):
            rec["left_out"] = "reversed"
        else:
            rec["gap"] = t[rows[p + 1]] - t[i]
        out.append(rec)
    return out


def first_step_gaps(runs, ifaces=("lab_v09", "cc")):
    """(a): seat_first_gaps of every human seat of the given interfaces, with the seat labels."""
    out = []
    for lab, t, s in seat_paths(runs):
        if lab["interface"] in ifaces:
            out += [{**lab, **r} for r in seat_first_gaps(t, s, lab["dt"], lab["step"])]
    return pd.DataFrame(out)


def share_near(gaps, d, tol=TOL):
    """Share of gaps within +/- tol of each d."""
    g = np.asarray(gaps, float)
    return np.array([np.mean(np.abs(g - x) <= tol + 1e-9) for x in np.atleast_1d(d)])


def d_ar_rule(gaps_by_iface, tol=TOL, threshold=0.40):
    """(a) RULE: d* = argmax over D_GRID of the smallest share across interfaces (the smallest d on a tie);
    d_ar = d* if that share >= threshold, else 0. Returns (d_ar, d*, min share, {interface: (own best d, share)})."""
    shares = np.vstack([share_near(g, D_GRID, tol) for g in gaps_by_iface.values()])
    low = shares.min(axis=0)
    i = int(np.argmax(low))
    own = {name: (D_GRID[int(np.argmax(sh))], sh.max()) for name, sh in zip(gaps_by_iface, shares)}
    return (D_GRID[i] if low[i] >= threshold else 0.0), D_GRID[i], low[i], own


# ------------------------------------------------------------------ (b) try hazard

def estimator_rows(run):
    """(t, s) of the rows hazard.py reads: lab as logged, CC on the 0.1 s lab clock (no t = 0 row)."""
    if run["dataset"] == "CC":
        t, X = lab_log(run["t"], run["X"])
        return t, sl.sign_correct({**run, "X": X})
    return np.asarray(run["t"], float), sl.sign_correct(run)


def try_bins(runs, ht):
    """(b): the G-train at-wall bins of the hazard table ht (split train, abs_s == 1), with y_try and g.
    y_try = 1 if the seat's first estimator row with |s| < 1 after a row with |s| = 1 is in [t, t + 0.25).
    g = log(1 + t - the last estimator row strictly before t on which a neighbor's s changed (from the row before,
    the first row from 0)), log(1 + t) if none."""
    w = ht[(ht["split"] == "train") & ht["topology"].isin(config.TRAIN_TOPOLOGIES) & (ht["abs_s"] >= WALL)]
    by_id = {f"{r['file']}#w{r['wave']}": r for r in runs}
    out = []
    for (rid, j), b in w.groupby(["round_id", "player"], sort=False):
        run = by_id[rid]
        t, s = estimator_rows(run)
        leave = t[1:][(np.abs(s[:-1, j]) >= WALL) & (np.abs(s[1:, j]) < WALL)]
        tau = b["t"].to_numpy(float)
        nxt = np.searchsorted(leave, tau - 1e-6, side="left")              # first leave at or after tau
        y = (nxt < len(leave)) & (np.r_[leave, np.inf][nxt] < tau + BIN - 1e-6)
        nbrs = np.flatnonzero((run["A_log"][j] > 0) & (np.arange(s.shape[1]) != j))
        moved = np.any(np.abs(np.diff(np.vstack([np.zeros(s.shape[1]), s]), axis=0)[:, nbrs]) > 1e-9, axis=1)
        last = np.where(moved, t, -np.inf)
        last = np.maximum.accumulate(last)
        row = np.clip(np.searchsorted(t, tau - 1e-6, side="right") - 1, 0, None)
        since = np.where(np.isfinite(last[row]), tau - last[row], tau)
        out.append(b.assign(y_try=y.astype(int), g=np.log1p(since)))
    return pd.concat(out)


T0 = ["A", "time_at_end_log1p", "time_since_rev_log1p"]


def fit_try(bins, feats):
    """Unpenalized Logit of y_try on const, lab and feats (fit.design)."""
    from slider import fit
    return sm.Logit(bins["y_try"].astype(float), fit.design(bins, feats, False)).fit(disp=0)


def g_rule(m0, m1):
    """(b) RULE: include g iff beta_g > 0, its Wald 95% CI excludes 0 and the LR test T1 vs T0 has p < 0.01."""
    lo, hi = m1.conf_int().loc["g"]
    lr = 2 * (m1.llf - m0.llf)
    p = stats.chi2.sf(lr, 1)
    return {"beta_g": m1.params["g"], "ci_lo": lo, "ci_hi": hi, "lr": lr, "p": p,
            "include_g": bool(m1.params["g"] > 0 and lo > 0 and p < 0.01)}


# ------------------------------------------------------------------ (c) walk timing

def seat_walk_gaps(t, s, dt, step):
    """(c) for ONE seat: gap ticks n of the own steps inside excursions that leave a wall and come back to it
    (a row with k steps: its gap, then k - 1 one-tick gaps). Returns a list of n."""
    wall = np.where(np.abs(s) >= WALL, np.sign(s), 0).astype(int)
    at = np.flatnonzero(wall != 0)
    rows, k = own_rows(s, step)
    kk = dict(zip(rows, k))
    out = []
    for i in np.flatnonzero((wall[:-1] != 0) & (wall[1:] != wall[:-1])) + 1:
        hit = at[at >= i]
        if not len(hit) or wall[hit[0]] != wall[i - 1]:
            continue                                                   # not a returned excursion
        inside = rows[(rows >= i) & (rows <= hit[0])]
        out += [1] * (kk[inside[0]] - 1)                               # the departure row's extra steps
        for a, b in zip(inside[:-1], inside[1:]):
            out += [int(round((t[b] - t[a]) / dt))] + [1] * (kk[b] - 1)
    return out


def walk_gaps(runs, d_ar):
    """(c): one row per gap: seat labels, n (ticks) and D = max(1, round(d_ar / dt)) (the wait to the decision)."""
    out = []
    for lab, t, s in seat_paths(runs):
        n = seat_walk_gaps(t, s, lab["dt"], lab["step"])
        if n:
            out.append(pd.DataFrame({**lab, "n": n, "D": max(1, int(round(d_ar / lab["dt"])))}))
    return pd.concat(out, ignore_index=True)


def q_w(gaps):
    """(c) geometric MLE: gaps / (gaps + sum of max(n - D, 0))."""
    return len(gaps) / (len(gaps) + np.maximum(gaps["n"] - gaps["D"], 0).sum())


# ------------------------------------------------------------------ (e) pace

def paces(runs):
    """(e): one row per 10a crossing (complete, no jump) of a human seat on the raw rows: seat labels, duration,
    v = |s at the end row - s at the start row| / duration (units/s), f = v / round speed."""
    out = []
    for lab, t, s in seat_paths(runs):
        tt, ss = t[1:], s[1:]
        k = to_steps(ss)
        c = crossing_events(tt, k, tt, k)
        if not len(c):
            continue
        c = c[(c["status"] == "complete") & ~c["has_jump"].astype(bool)]
        for x in c.itertuples():
            a, b = np.searchsorted(tt, x.start - 1e-9), np.searchsorted(tt, x.end - 1e-9)
            v = abs(ss[b] - ss[a]) / x.duration
            out.append({**lab, "duration": x.duration, "v": v, "f": v / lab["speed"]})
    out = pd.DataFrame(out)
    out["round_speed"] = out["speed"].round(2)
    return out


# ------------------------------------------------------------------ (f) descriptive

def round_summary(run):
    """(f) for one run: all-at-wall time (all 7 seats at |k| = 10), round time (to fraught.round_end) and its
    standoffs (standoffs.round_standoffs)."""
    t = np.r_[0.0, run["t"]]
    k = np.vstack([np.zeros(7, int), to_steps(sl.sign_correct(run))])
    t_end = round_end(run["t"], run["end"], run["session"])
    dur = np.diff(np.r_[t, t_end])
    all7 = np.all(np.abs(k) == standoffs.WALL, axis=1)
    return {"wall_time": dur[all7].sum(), "time": t_end, "standoffs": standoffs.round_standoffs(t, k, t_end)}


def descriptive(runs):
    """(f) per cell (topology, dataset): rounds, all_wall_share, standoffs per round (6-1, 5-2) and median
    standoff duration (all, 6-1, 5-2)."""
    rows, so = [], []
    for run in runs:
        r = round_summary(run)
        cell = {"topology": sl.label_topology(run["A"]), "dataset": run["dataset"]}
        rows.append({**cell, "wall_time": r["wall_time"], "time": r["time"]})
        so += [{**cell, **x} for x in r["standoffs"]]
    rows, so = pd.DataFrame(rows), pd.DataFrame(so, columns=["topology", "dataset", "split", "duration"])
    key = ["topology", "dataset"]
    out = rows.groupby(key).agg(rounds=("time", "size"), wall_time=("wall_time", "sum"), time=("time", "sum"))
    out["all_wall_share"] = out["wall_time"] / out["time"]
    for split in ["6-1", "5-2"]:
        x = so[so["split"] == split]
        out[f"standoffs_{split}_per_round"] = x.groupby(key).size().reindex(out.index, fill_value=0) / out["rounds"]
        out[f"median_duration_{split}"] = x.groupby(key)["duration"].median()
    out["median_duration_all"] = so.groupby(key)["duration"].median()
    return out.drop(columns=["wall_time", "time"])

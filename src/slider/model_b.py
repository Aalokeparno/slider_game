"""Model B data checks B0b (crossing style) and B2 (try-and-keep). Descriptive only.

Same runs, human seats, steps and stops as B0 (stop_states.step_table, motion.py F1b). Rows are indexed as in
motion_spells: a row t = 0, s = 0 first, then the logged rows. Agreement A in [0, 1] (slider_logs.agreement / 100,
every seat enters it, bots too); A = 0 on the t = 0 row.
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats

from slider import config, interface, motion
from slider import slider_logs as sl
from slider.fraught import round_end
from slider.reversals import detect_events, to_steps
from slider.stop_states import band_side


def padded(run):
    """t, s and A of one run with the t = 0, s = 0 row first."""
    t = np.r_[0.0, run["t"]]
    s = np.vstack([np.zeros(len(run["weights"])), sl.sign_correct(run)])
    return t, s, sl.agreement(run["A_log"], s) / 100


def run_labels(runs):
    """{(file, wave): {dataset, interface, topology}} for every run."""
    files = interface.interface_table(runs).set_index("file")
    return {(r["file"], r["wave"]): {"dataset": r["dataset"], "topology": sl.label_topology(r["A"]),
                                     "interface": motion.round_iface(r, files.loc[r["file"]])["interface"]}
            for r in runs}


# ----------------------------------------------------------------- B0b: crossings

def crossing_table(runs, steps):
    """One row per completed crossing (B0: an outbound segment that reaches the opposite end band), from the
    B0 step table: file, wave, player, segment, dataset, interface, topology, steps, stops, smooth (no stop),
    start_row (the row whose step leaves the band) and A_start (the player's A at the row before it)."""
    by_run = {(r["file"], r["wave"]): r for r in runs}
    labels = run_labels(runs)
    c = steps[steps["outcome"] == "completed"]
    c = c.groupby(["file", "wave", "player", "segment"]).agg(first_row=("row", "min"), steps=("stop", "size"),
                                                             stops=("stop", "sum")).reset_index()
    rows = []
    for (file, wave), d in c.groupby(["file", "wave"]):
        run = by_run[(file, wave)]
        _, s, A = padded(run)
        old_cl = labels[(file, wave)]["interface"] == "old_cl"
        for x in d.itertuples():
            side = np.array([band_side(p, old_cl) for p in s[:x.first_row + 1, x.player]])
            out = (side[:-1] != 0) & (side[1:] != side[:-1])           # row i: from a band to outside it
            start = int(np.flatnonzero(out)[-1]) + 1
            rows.append({"file": file, "wave": wave, "player": x.player, "segment": x.segment,
                         **labels[(file, wave)], "steps": x.steps, "stops": x.stops, "smooth": int(x.stops == 0),
                         "start_row": start, "A_start": A[start - 1, x.player]})
    return pd.DataFrame(rows)


def player_shares(cross, min_crossings=3):
    """Per player (file, player) with >= min_crossings completed crossings: interface, dataset, n, smooth."""
    p = cross.groupby(["interface", "dataset", "file", "player"]).agg(n=("smooth", "size"), smooth=("smooth", "sum"))
    return p[p["n"] >= min_crossings].reset_index()


def sd_of_shares(smooth, player_codes, n):
    return np.std(np.bincount(player_codes, weights=smooth) / n, ddof=1)


def permutation_sd(cross, players, n_perm=1000, seed=config.SEED):
    """Observed SD of the player smooth shares (players = player_shares rows) and its permutation
    distribution: the smooth labels of these players' crossings shuffled across them, within interface."""
    key = cross.set_index(["file", "player"]).index
    keep = key.isin(pd.MultiIndex.from_frame(players[["file", "player"]]))
    c = cross[keep]
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(c[["file", "player"]]))
    n = np.bincount(codes)
    smooth = c["smooth"].to_numpy(float)
    groups = [np.flatnonzero(c["interface"].to_numpy() == name) for name in c["interface"].unique()]
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(n_perm):
        y = smooth.copy()
        for g in groups:
            y[g] = rng.permutation(y[g])
        perm.append(sd_of_shares(y, codes, n))
    return sd_of_shares(smooth, codes, n), np.array(perm)


def betabinom_fit(k, n):
    """Beta-binomial MLE: mean pi and overdispersion rho = 1 / (a + b + 1)."""
    def nll(x):
        pi, rho = 1 / (1 + np.exp(-x))
        a, b = pi * (1 - rho) / rho, (1 - pi) * (1 - rho) / rho
        return -stats.betabinom.logpmf(k, n, a, b).sum()
    x = optimize.minimize(nll, [0.0, -2.0], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8}).x
    return 1 / (1 + np.exp(-x))


def boot_groups(df, fn, n_boot=1000, seed=config.SEED):
    """95% CI of fn(resample) over n_boot resamples of the groups (files) within dataset, drawn as in
    motion.boot_ratios. fn returns an array; returns (low, high)."""
    files = df.groupby("file")["dataset"].first()
    parts = {f: d for f, d in df.groupby("file")}
    strata = [files.index[files.to_numpy() == d].to_numpy() for d in interface.DATASETS if (files == d).any()]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        draw = np.concatenate([st[rng.integers(0, len(st), len(st))] for st in strata])
        out.append(fn(pd.concat([parts[f] for f in draw], ignore_index=True)))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


# ----------------------------------------------------------------- B2: tries

def try_table(runs, m=2):
    """One row per try of a human seat: the first step of an event that detect_events (m) labels a reversal
    or a probe (unresolved events left out). Columns: file, wave, player, dataset, interface, topology, type,
    kept (1 = reversal), row, s_before (the seat's s on the row before the try), from_wall (|k| = 10 before the try: |s| = 1 on lab and CC; on old CL the
    detector's k grid puts the try a few 0.02 steps off the wall), same_row (the try row already makes m steps
    against the direction), hold (time to the next own step; to round_end if none), dA (A at the row before the
    next own step - A at the row before the try), dA_own (A with only the player's try step, neighbors at their
    pre-try positions, - A before the try), others_moved (a neighbor's s changes from the row before the try to
    the row before the next own step)."""
    labels = run_labels(runs)
    out = []
    for run in runs:
        t, s, A = padded(run)
        t_end = round_end(run["t"], run["end"], run["session"])
        deg = run["A_log"].sum(axis=1)
        for j in range(s.shape[1]):
            if j + 1 in run["bot_players"]:
                continue
            ev = detect_events(t, s[:, j], m)
            ev = ev[ev["type"] != "unresolved"]
            own = np.flatnonzero(np.abs(np.diff(s[:, j])) > 1e-9) + 1
            nbrs = np.flatnonzero((run["A_log"][j] > 0) & (np.arange(s.shape[1]) != j))
            k = to_steps(s[:, j])
            dk = np.abs(np.diff(k, prepend=0))
            for e in ev.itertuples():
                i = e.event_row
                later = own[own > i]
                nxt = later[0] if len(later) else len(t)
                hold = (t[nxt] if len(later) else t_end) - t[i]
                s_own = s[i - 1].copy()
                s_own[j] = s[i, j]
                moved = np.abs(np.diff(s[i - 1:nxt][:, nbrs], axis=0)) > 1e-9
                out.append({"file": run["file"], "wave": run["wave"], "player": j, **labels[(run["file"], run["wave"])],
                            "type": e.type, "kept": int(e.type == "reversal"), "row": i, "s_before": s[i - 1, j],
                            "from_wall": int(abs(k[i - 1]) == 10), "same_row": int(dk[i] >= m),
                            "hold": hold, "dA": A[nxt - 1, j] - A[i - 1, j],
                            "dA_own": abs(run["A_log"][j] @ s_own) / deg[j] - A[i - 1, j],
                            "others_moved": int(moved.any())})
    return pd.DataFrame(out)


def sign_label(x, tol=1e-9):
    return np.where(x > tol, "> 0", np.where(x < -tol, "< 0", "= 0"))

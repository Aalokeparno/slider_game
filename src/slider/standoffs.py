"""Model C (descriptive): who breaks a standoff on Full, and the C1 10a check (notebooks/path1/pC_checks.ipynb).

Positions: k = round(10 s), s sign-corrected, with the t = 0, k = 0 row first. A seat is at its wall on a row with
|k| = 10 (side = sign(k); old CL: |s| >= 0.95).
    standoff     a row where all 7 seats are at a wall, split k_min vs 7 - k_min (k_min = 1: 6-1, 2: 5-2), and the
                 row before is not the same configuration; the all-at-wall stretch from it must last >= 0.5 s (to
                 the first row with a seat off its wall, or the round end)
    try          the first step off its wall by a seat (the first step of an m = 2 reversal or a probe,
                 reversals.detect_events, from a row at the wall); kept = reversal, undone = probe
    the standoff runs from its start to its break (the first reversal away from a seat's standoff wall, usually a
                 kept try) or the round end; tries, kept tries and player-seconds at the wall are counted over it
    last arriver the seat(s) whose arrival at the wall made the configuration (at the start row); NaN if seats of
                 both sides arrived on that row
"""
import numpy as np
import pandas as pd

from slider import config, interface, model_c1, motion, targets
from slider import slider_logs as sl
from slider.crossings import crossing_events
from slider.fraught import round_end
from slider.reversals import detect_events, to_steps

WALL = 10
SPLITS = {1: "6-1", 2: "5-2"}
MIN_LEN = 0.5


def round_standoffs(t, k, t_end, min_len=MIN_LEN, m=2):
    """The standoffs of one round (t, k: rows x 7 seats, the t = 0 row first). One dict per standoff."""
    side = np.where(np.abs(k) == WALL, np.sign(k), 0)
    all_wall = np.all(side != 0, axis=1)
    ev = [detect_events(t, k[:, j] / 10, m) for j in range(k.shape[1])]
    dt = np.diff(np.r_[t, t_end])                                    # each row holds until the next (last: round end)
    out = []
    i = 1
    while i < len(t):
        same = all_wall[i - 1] and np.array_equal(side[i - 1], side[i])
        n_plus = int(np.sum(side[i] > 0))
        k_min = min(n_plus, 7 - n_plus)
        if not all_wall[i] or same or k_min not in SPLITS:
            i += 1
            continue
        off = np.flatnonzero(~all_wall[i:])
        stretch = (t[i + off[0]] if len(off) else t_end) - t[i]
        if stretch < min_len:
            i += off[0] if len(off) else len(t)
            continue
        sd = side[i]
        minority = sd == (1 if n_plus == k_min else -1)
        # the break: the first reversal away from a seat's standoff wall after the start
        brk = [(r, j) for j in range(7) for r, d, typ in zip(ev[j]["event_row"], ev[j]["dir_before"], ev[j]["type"])
               if typ == "reversal" and d == sd[j] and r > i]
        end_row, brk_seat = min(brk) if brk else (len(t), -1)
        # tries: events that start from the seat's own wall, in (start, break]
        tries = sorted((r, j, typ) for j in range(7) for r, typ in zip(ev[j]["event_row"], ev[j]["type"])
                       if i < r <= end_row and side[r - 1, j] == sd[j])
        rows = slice(i, end_row)
        expo = np.sum(dt[rows, None] * (side[rows] == sd), axis=0)   # seconds at its own wall, per seat
        rec = {"split": SPLITS[k_min], "k_min": k_min, "start": t[i], "stretch": stretch,
               "duration": (t[end_row] if brk else t_end) - t[i], "broken": int(bool(brk)),
               "break_minority": float(minority[brk_seat]) if brk else np.nan,
               "break_from_wall": float(side[end_row - 1, brk_seat] == sd[brk_seat]) if brk else np.nan,
               "expo_min": expo[minority].sum(), "expo_maj": expo[~minority].sum()}
        for g, mask in [("min", minority), ("maj", ~minority)]:
            mine = [typ for r, j, typ in tries if mask[j]]
            rec[f"tries_{g}"] = len(mine)
            rec[f"kept_{g}"] = sum(typ == "reversal" for typ in mine)
            rec[f"resolved_{g}"] = sum(typ != "unresolved" for typ in mine)
        if tries:
            first = [x for x in tries if x[0] == tries[0][0]]                # all tries on the first try's row
            by = {bool(minority[j]) for _, j, _ in first}
            rec["first_by"] = ("minority" if by == {True} else "majority") if len(by) == 1 else "tie"
            rec["first_time"] = t[tries[0][0]] - t[i]
            typ = first[0][2] if len(first) == 1 else "tie"
            rec["first_kept"] = {"reversal": 1.0, "probe": 0.0}.get(typ, np.nan)
        else:
            rec.update(first_by="none", first_time=np.nan, first_kept=np.nan)
        arrived = (side[i - 1] != sd)
        sides = set(minority[arrived])
        rec["last_minority"] = float(sides.pop()) if len(sides) == 1 else np.nan
        out.append(rec)
        i = end_row
    return out


def standoff_table(runs, table):
    """Standoffs of every all-human Full round of runs: round_standoffs rows with file, wave, dataset, interface
    and consensus (1 = the round ended in consensus)."""
    files = interface.interface_table(runs).set_index("file")
    out = []
    for run, (_, row) in zip(runs, table.iterrows()):
        if row["topology"] != "Full" or len(run["bot_players"]) > 0:
            continue
        iface = motion.round_iface(run, files.loc[run["file"]])
        t = np.r_[0.0, run["t"]]
        k = np.vstack([np.zeros(7, int), to_steps(sl.sign_correct(run))])
        labels = {"file": run["file"], "wave": run["wave"], "dataset": run["dataset"],
                  "interface": iface["interface"], "consensus": int(run["end"] == "success")}
        out += [{**labels, **r} for r in round_standoffs(t, k, round_end(run["t"], run["end"], run["session"]))]
    return pd.DataFrame(out)


def standoff_stats(d):
    """The standoff statistics of a set of standoff rows (dict)."""
    first = d[d["first_by"].isin(["minority", "majority"])]
    fmin = first["first_by"] == "minority"
    known = d[d["last_minority"].notna()]
    out = {"first_min_share": fmin.mean(),
           "rate_ratio": (d["tries_min"].sum() / d["expo_min"].sum()) / (d["tries_maj"].sum() / d["expo_maj"].sum()),
           "kept_first_min": first.loc[fmin, "first_kept"].mean(),
           "kept_first_maj": first.loc[~fmin, "first_kept"].mean(),
           "kept_all_min": d["kept_min"].sum() / d["resolved_min"].sum(),
           "kept_all_maj": d["kept_maj"].sum() / d["resolved_maj"].sum(),
           "break_min_share": d["break_minority"].mean(),
           "last_min_share": known["last_minority"].mean()}
    for name, v in [("last_min", 1.0), ("last_maj", 0.0)]:
        f = first[first["last_minority"] == v]
        out[f"first_min_share_{name}"] = (f["first_by"] == "minority").mean()
    out["first_time_median"] = d["first_time"].median()
    return out


def boot_cells(d, cells, n_boot=1000, seed=config.SEED):
    """standoff_stats with 95% CIs for each (name, rows) in cells: n_boot resamples of the files, drawn with
    replacement within each dataset (as motion.boot_ratios), rng(seed) per cell. Long format: cell, stat, value,
    ci_lo, ci_hi, plus the counts standoffs and groups (no CI)."""
    rows = []
    for name, c in cells:
        c = c.reset_index(drop=True)
        by_file = {f: idx.to_numpy() for f, idx in c.groupby("file").groups.items()}
        files = c.groupby("file")["dataset"].first()
        strata = [files.index[files.to_numpy() == ds].to_numpy() for ds in interface.DATASETS if (files == ds).any()]
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(n_boot):
            draw = np.concatenate([st[rng.integers(0, len(st), len(st))] for st in strata])
            with np.errstate(divide="ignore", invalid="ignore"):
                boot.append(standoff_stats(c.loc[np.concatenate([by_file[f] for f in draw])]))
        boot = pd.DataFrame(boot)
        rows += [{"cell": name, "stat": s, "value": n, "ci_lo": np.nan, "ci_hi": np.nan}
                 for s, n in [("standoffs", len(c)), ("groups", c["file"].nunique())]]
        for s, v in standoff_stats(c).items():
            with np.errstate(invalid="ignore"):
                lo, hi = np.nanpercentile(boot[s].to_numpy(float), [2.5, 97.5])
            rows.append({"cell": name, "stat": s, "value": v, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def baselines(k_min, keep_min=0.88, keep_maj=0.73):
    """Arithmetic baselines for a split with k_min minority seats, every seat at its wall trying at the same rate:
    C1 (every try kept) and C2 (kept with keep_min / keep_maj). first_min_share = k_min / 7, rate_ratio = 1, and
    break_min_share = the minority's share of the kept tries: k p_min / (k p_min + (7 - k) p_maj)."""
    share = k_min / 7
    out = {}
    for name, (pm, pj) in [("C1", (1.0, 1.0)), ("C2", (keep_min, keep_maj))]:
        out[name] = {"first_min_share": share, "rate_ratio": 1.0, "kept_all_min": pm, "kept_all_maj": pj,
                     "break_min_share": k_min * pm / (k_min * pm + (7 - k_min) * pj)}
    return out


# ------------------------------------------------------------------ C1 10a: wall-to-wall trips vs 10a crossings

def trip_parts(runs):
    """One row per arrived wall-to-wall trip (model_c1.seat_trips) of every human seat, with the 10a crossings
    (crossings.crossing_events: |k| >= 9 end bands, on the raw rows) that start and end inside it:
    topology, dataset, file, wave, player, start, duration (wall to wall), n_cross (crossings inside),
    clean (1 = exactly one crossing inside, and it is complete), core (that crossing's duration; NaN unless clean),
    lead / tail (the trip's time before / after it: wall to band exit, band entry to wall; NaN unless clean),
    core_jump (1 = that crossing has a jump)."""
    out = []
    for run in runs:
        t, k = run["t"], to_steps(sl.sign_correct(run))
        t0, s0 = np.r_[0.0, t], np.vstack([np.zeros(7), sl.sign_correct(run)])
        info = {"topology": sl.label_topology(run["A"]), "dataset": run["dataset"], "file": run["file"],
                "wave": run["wave"]}
        for j in range(7):
            if j + 1 in run["bot_players"]:
                continue
            cr = crossing_events(t, k[:, j], t, k[:, j])
            for r in model_c1.seat_trips(t0, s0[:, j]):
                if r["kind"] != "cross" or r["outcome"] != "arrived":
                    continue
                end = r["start"] + r["duration"]
                c = cr[(cr["start"] >= r["start"] - 1e-9) & (cr["end"] <= end + 1e-9)] if len(cr) else cr
                clean = len(c) == 1 and c["status"].iloc[0] == "complete"
                out.append({**info, "player": j, "start": r["start"], "duration": r["duration"], "n_cross": len(c),
                            "clean": int(clean), "core": c["duration"].iloc[0] if clean else np.nan,
                            "lead": c["start"].iloc[0] - r["start"] if clean else np.nan,
                            "tail": end - c["end"].iloc[0] if clean else np.nan,
                            "core_jump": int(c["has_jump"].iloc[0]) if clean else np.nan})
    return pd.DataFrame(out)


def crossing_times(runs):
    """Per cell (topology, dataset), over the human seats, medians (s):
        tenA            the 10a statistic: complete crossings with no jump, CC on the lab clock (targets.crossing_values)
        band_raw        the 10a crossings on the raw rows (complete, no jump)
        wall_to_wall    every arrived wall-to-wall trip (what C1 draws from, pooled per interface)
        w2w_clean       the clean trips only (one complete crossing inside)
        core_clean      the band-to-band part of the clean trips
        lead_clean, tail_clean   the rest of the clean trips: wall to band exit, band entry to wall
    plus n_tenA, n_w2w, clean_share (share of the trips that are clean) and lead_05_share (share of the clean trips
    whose lead is 0.5 s within 10 ms)."""
    lab, _ = targets.crossing_values(runs)
    raw, _ = targets.crossing_values(runs, lab_clock_cc=False)
    trips = trip_parts(runs)
    key = ["topology", "dataset"]
    used = lambda c: c[(c["status"] == "complete") & ~c["has_jump"].astype(bool)]
    clean = trips[trips["clean"] == 1]
    return pd.DataFrame({
        "tenA": used(lab).groupby(key)["duration"].median(),
        "band_raw": used(raw).groupby(key)["duration"].median(),
        "wall_to_wall": trips.groupby(key)["duration"].median(),
        "w2w_clean": clean.groupby(key)["duration"].median(),
        "core_clean": clean.groupby(key)["core"].median(),
        "lead_clean": clean.groupby(key)["lead"].median(),
        "tail_clean": clean.groupby(key)["tail"].median(),
        "n_tenA": used(lab).groupby(key).size(),
        "n_w2w": trips.groupby(key).size(),
        "clean_share": trips.groupby(key)["clean"].mean(),
        "lead_05_share": clean.assign(x=(clean["lead"] - 0.5).abs() < 0.01).groupby(key)["x"].mean()})

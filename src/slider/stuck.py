"""Track C (results/model_b_log.md entry 4, descriptive): stuck episodes, humans vs. C2
(notebooks/path1/pC2_stuck.ipynb).

Positions: k = round(10 s), s sign-corrected, with the t = 0, k = 0 row first. A seat is at a wall on a row with
|k| = 10; side = sign(k).
    episode      a row where all 7 seats are at a wall with mixed sides, and the row before is not the same
                 configuration; the all-at-wall stretch from it (to the first row with a seat off its wall, or the
                 round end) must last >= 0.5 s. It ends on the first later row with a seat at the wall opposite
                 its side at the start, or at the round end. The next episode is searched from the end row.
    type         Full, Star: the split 6-1, 5-2 or 4-3 (k_min = min(seats +, seats -)); Hierarchy: fraught or
                 non-fraught (fraught.is_fraught at the corner of the start row)
    try          an m = 2 event (reversals.detect_events) by a seat whose previous row is at its start wall, with
                 its event row in (start, end]; kept = reversal, not kept = probe (unresolved: left out of P(kept))
    breaker      the seat(s) at the opposite wall on the end row; minority = the k_min side
"""
import numpy as np
import pandas as pd

from slider import config, interface
from slider import slider_logs as sl
from slider.fraught import is_fraught, round_end
from slider.reversals import detect_events, to_steps

WALL = 10
MIN_LEN = 0.5
SPLITS = {1: "6-1", 2: "5-2", 3: "4-3"}
TYPES = {"Full": ["6-1", "5-2", "4-3"], "Star": ["6-1", "5-2", "4-3"], "Hierarchy": ["fraught", "non-fraught"]}


def round_episodes(t, k, A, topology, t_end, min_len=MIN_LEN, m=2):
    """The episodes of one round (t, k: rows x 7 seats, the t = 0 row first; A in seat order). One dict each."""
    side = np.where(np.abs(k) == WALL, np.sign(k), 0)
    all_wall = np.all(side != 0, axis=1)
    mixed = np.any(side > 0, axis=1) & np.any(side < 0, axis=1)
    ev = [detect_events(t, k[:, j] / 10, m) for j in range(7)]
    out = []
    i = 1
    while i < len(t):
        same = all_wall[i - 1] and np.array_equal(side[i - 1], side[i])
        if not (all_wall[i] and mixed[i]) or same:
            i += 1
            continue
        off = np.flatnonzero(~all_wall[i:])
        if (t[i + off[0]] if len(off) else t_end) - t[i] < min_len:
            i += off[0] if len(off) else len(t)
            continue
        sd = side[i]
        n_plus = int(np.sum(sd > 0))
        k_min = min(n_plus, 7 - n_plus)
        minority = sd == (1 if n_plus == k_min else -1)
        opp = np.flatnonzero(np.any(side[i + 1:] == -sd, axis=1))
        end_row = i + 1 + opp[0] if len(opp) else len(t)
        tries = sorted((r, typ) for j in range(7) for r, typ in zip(ev[j]["event_row"], ev[j]["type"])
                       if i < r <= end_row and side[r - 1, j] == sd[j])
        kind = ("fraught" if is_fraught(A, sd.astype(float)) else "non-fraught") if topology == "Hierarchy" \
            else SPLITS[k_min]
        rec = {"type": kind, "start": t[i], "duration": (t[end_row] if len(opp) else t_end) - t[i],
               "broken": int(len(opp) > 0), "tries": len(tries),
               "kept": sum(typ == "reversal" for _, typ in tries), "probes": sum(typ == "probe" for _, typ in tries),
               "first_try": t[tries[0][0]] - t[i] if tries else np.nan, "break_minority": np.nan}
        if len(opp):
            by = set(minority[side[end_row] == -sd])
            rec["break_minority"] = float(by.pop()) if len(by) == 1 else np.nan
        out.append(rec)
        i = end_row
    return out


def episode_table(runs, table, ifaces):
    """Episodes of every run (round_episodes rows) with file, wave, dataset, topology and interface (ifaces: one
    label per run)."""
    out = []
    for run, (_, row), iface in zip(runs, table.iterrows(), ifaces):
        t = np.r_[0.0, run["t"]]
        k = np.vstack([np.zeros(7, int), to_steps(sl.sign_correct(run))])
        labels = {"file": row["file"], "wave": row["wave"], "dataset": row["dataset"], "topology": row["topology"],
                  "interface": iface}
        out += [{**labels, **r} for r in round_episodes(t, k, run["A"], row["topology"],
                                                        round_end(run["t"], run["end"], run["session"]))]
    return pd.DataFrame(out)


def episode_stats(d, n_rounds):
    """The statistics of a set of episodes (rows of one cell) over n_rounds rounds."""
    resolved = d["kept"].sum() + d["probes"].sum()
    return {"episodes": len(d), "per_round": len(d) / n_rounds, "duration_median": d["duration"].median(),
            "first_try_median": d["first_try"].median(), "tries_median": d["tries"].median(),
            "p_kept": d["kept"].sum() / resolved if resolved > 0 else np.nan,
            "break_min_share": d["break_minority"].mean()}


def boot_cell(d, rounds, n_boot=1000, seed=config.SEED):
    """episode_stats of one cell with 95% CIs: n_boot resamples of the files, drawn with replacement within each
    dataset, rng(seed). d: the cell's episodes; rounds: one row per round of the cell (file, dataset).
    Returns {stat: (value, ci_lo, ci_hi)}."""
    d = d.reset_index(drop=True)
    per_file = rounds.groupby("file").size()
    files = rounds.groupby("file")["dataset"].first()
    strata = [files.index[files.to_numpy() == ds].to_numpy() for ds in interface.DATASETS if (files == ds).any()]
    rng = np.random.default_rng(seed)
    by_file = {f: idx.to_numpy() for f, idx in d.groupby("file").groups.items()}
    boot = []
    for _ in range(n_boot):
        draw = np.concatenate([st[rng.integers(0, len(st), len(st))] for st in strata])
        rows = np.concatenate([by_file.get(f, np.array([], dtype=int)) for f in draw])
        with np.errstate(divide="ignore", invalid="ignore"):
            boot.append(episode_stats(d.loc[rows], per_file[draw].sum()))
    boot = pd.DataFrame(boot)
    out = {}
    for s, v in episode_stats(d, len(rounds)).items():
        x = boot[s].to_numpy(float)
        x = x[~np.isnan(x)]
        out[s] = (v, *(np.percentile(x, [2.5, 97.5]) if len(x) else (np.nan, np.nan)))
    return out

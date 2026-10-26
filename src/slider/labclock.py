"""Path 1, P2c: cloud runs logged like the lab, and event rates (reversals and probes per player-minute).

The lab logs a row about every 0.1 s, only when something changed; the cloud logs every move with 1 ms
times. lab_log turns cloud rows into lab-style rows, so probes and reversals can be counted the same way.
Rates: events / human player-minutes, pooled over the runs of a cell, 95% CI by group (log file) bootstrap.
"""
import numpy as np
import pandas as pd

from slider import config
from slider import slider_logs as sl
from slider.fraught import round_end
from slider.reversals import COLUMNS, run_events, to_steps
from slider.stats import group_bootstrap

CELLS = [("Hierarchy", "EL"), ("Hierarchy", "CL"), ("Hierarchy", "CC"), ("Full", "CL"), ("Full", "CC"), ("Star", "CC")]
TYPES = ["reversal", "probe", "unresolved"]


def lab_log(t, S, tick=0.1):
    """One run's rows (times t, positions S: rows x 7) as lab-style rows. Returns (t_lab, S_lab).

    Tick times: t0 + j * tick while <= t_last, plus t_last itself.
    State at a tick = the last original row at or before it (forward hold).
    A tick is kept only if some k = round(10 s) differs from the last kept row; the first row is always kept.
    Kept times are rounded to 0.01 s.
    """
    t = np.asarray(t, dtype=float)
    S = np.asarray(S, dtype=float)
    n_ticks = int(np.floor((t[-1] - t[0]) / tick + 1e-6)) + 1
    ticks = t[0] + tick * np.arange(n_ticks)
    if ticks[-1] < t[-1] - 1e-6:
        ticks = np.append(ticks, t[-1])
    rows = np.searchsorted(t, ticks + 1e-6, side="right") - 1     # last original row at or before each tick
    k = to_steps(S[rows])
    keep = [0]
    for j in range(1, len(ticks)):
        if np.any(k[j] != k[keep[-1]]):
            keep.append(j)
    return np.round(ticks[keep], 2), S[rows[keep]]


def events_table(runs, m=2, lab_clock=True, tick=0.1):
    """run_events on every run; with lab_clock=True, CC runs go through lab_log first.
    Columns: dataset, topology, file, wave, player, then the detect_events columns."""
    rows = []
    for run in runs:
        if lab_clock and run["dataset"] == "CC":
            t_lab, X_lab = lab_log(run["t"], run["X"], tick)
            run = {**run, "t": t_lab, "X": X_lab}
        info = {"dataset": run["dataset"], "topology": sl.label_topology(run["A"])}
        for ev in run_events(run, m).to_dict("records"):
            rows.append({**info, **ev})
    return pd.DataFrame(rows, columns=["dataset", "topology", "file", "wave", "player"] + COLUMNS)


def exposure(runs):
    """Minutes of play of every human player in every run: dataset, topology, file, wave, player, minutes.
    A round lasts until fraught.round_end: the last row if it succeeded or has no end marker,
    max(time limit, last row) if it failed."""
    rows = []
    for run in runs:
        minutes = round_end(run["t"], run["end"], run["session"]) / 60
        for j in range(7):
            if j + 1 not in run["bot_players"]:
                rows.append({"dataset": run["dataset"], "topology": sl.label_topology(run["A"]),
                             "file": run["file"], "wave": run["wave"], "player": j, "minutes": minutes})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ summaries

def per_run_counts(events, expo):
    """One row per run: dataset, topology, file, wave, human player-minutes, and the number of each event type."""
    r = expo.groupby(["dataset", "topology", "file", "wave"], as_index=False)["minutes"].sum()
    counts = pd.crosstab([events["file"], events["wave"]], events["type"]).reindex(columns=TYPES, fill_value=0)
    r = r.merge(counts.reset_index(), on=["file", "wave"], how="left")
    r[TYPES] = r[TYPES].fillna(0)
    return r


def rates(events, expo, ci=True, n_boot=10000, seed=config.SEED):
    """Reversals and probes per human player per minute in each cell (topology x dataset), pooled over runs.
    With ci=True: 95% CI by group bootstrap (log files resampled, stats.group_bootstrap)."""
    r = per_run_counts(events, expo)
    out = []
    for topology, dataset in CELLS:
        c = r[(r["topology"] == topology) & (r["dataset"] == dataset)]
        row = {"topology": topology, "dataset": dataset, "groups": c["file"].nunique(), "runs": len(c),
               "player-min": c["minutes"].sum(),
               "reversals/min": c["reversal"].sum() / c["minutes"].sum(),
               "probes/min": c["probe"].sum() / c["minutes"].sum()}
        if ci:
            per_group = [(d["reversal"].to_numpy(float), d["probe"].to_numpy(float), d["minutes"].to_numpy(float))
                         for _, d in c.groupby("file", sort=True)]
            lo, hi = group_bootstrap(per_group, lambda rev, pr, mi: np.array([rev.sum(), pr.sum()]) / mi.sum(),
                                     n_boot, seed)
            row.update({"rev lo": lo[0], "rev hi": hi[0], "probe lo": lo[1], "probe hi": hi[1]})
        out.append(row)
    cols = ["groups", "runs", "player-min", "reversals/min"] + (["rev lo", "rev hi"] if ci else []) + \
           ["probes/min"] + (["probe lo", "probe hi"] if ci else [])
    return pd.DataFrame(out).set_index(["topology", "dataset"])[cols]


def compare_rates(settings, expo):
    """Point rates side by side; settings = {name: events table}."""
    return pd.concat({name: rates(ev, expo, ci=False)[["reversals/min", "probes/min"]]
                      for name, ev in settings.items()}, axis=1)


def logging_loss(raw, logged):
    """CC only, by topology (and all CC): probes and reversals counted on the raw cloud rows and on the
    lab-logged rows, and the share lost = 1 - logged / raw."""
    out = []
    for topology in ["Full", "Hierarchy", "Star", "All"]:
        row = {"topology": topology}
        for kind in ["probe", "reversal"]:
            n = []
            for ev in [raw, logged]:
                e = ev[(ev["dataset"] == "CC") & (ev["type"] == kind)]
                n.append(len(e) if topology == "All" else int(np.sum(e["topology"] == topology)))
            row.update({f"{kind}s raw": n[0], f"{kind}s logged": n[1], f"{kind}s lost": 1 - n[1] / n[0]})
        out.append(row)
    return pd.DataFrame(out).set_index("topology")


def unresolved_share(events):
    """Number of events and % that are unresolved, by cell."""
    out = []
    for topology, dataset in CELLS:
        e = events[(events["topology"] == topology) & (events["dataset"] == dataset)]
        n = int(np.sum(e["type"] == "unresolved"))
        out.append({"topology": topology, "dataset": dataset, "events": len(e), "unresolved": n,
                    "% unresolved": 100 * n / len(e)})
    return pd.DataFrame(out).set_index(["topology", "dataset"])

"""Path 1, P3: the reversal detector on APEM runs, where the true reversals and probes are known.

APEM agents move in straight lines at constant speed and only turn when they flip at a check-in,
so the rows of an apem() run hold the exact path and every turning point.
Truth: true_events on that path (distance in units). Detected: reversals.detect_events (m grid steps)
on the rows a lab or cloud log would have kept (observe). score matches the two.
"""
import numpy as np
import pandas as pd

from slider import config
from slider.labclock import lab_log
from slider.models.apem import GRAPHS, apem
from slider.reversals import detect_events, to_steps

TOPOLOGIES = ["Hierarchy", "Full", "Star"]
# detector settings: name -> (log style, tick, m); the truth for m grid steps uses m_dist = m / 10
SETTINGS = {"lab (primary)": ("lab", 0.1, 2), "cloud": ("cloud", 0.1, 2),
            "lab, tick 0.05": ("lab", 0.05, 2), "lab, m = 3": ("lab", 0.1, 3)}


def simulate(topology, n_runs=100, seed=config.SEED, **params):
    """n_runs APEM runs on one topology (apem defaults: a_th = s_th = 0.5, mu = 0.1, speed 1, T = 300 s),
    one random generator per call, with the flip log."""
    rng = np.random.default_rng(seed)
    return [apem(GRAPHS[topology], rng=rng, record_flips=True, **params) for _ in range(n_runs)]


def true_events(t, s, m_dist=0.2):
    """The detect_events state machine on ONE agent's exact path, with distance in units instead of grid steps.

    The path is piecewise linear through the rows (t, s); it turns only at flip times. A row-to-row change of 0
    is a pause (at the wall) and the direction carries over. Moves in the same direction are merged into one
    stretch, which starts at its turning point (the flip time). A stretch against the direction is
        "reversal"    if it travels >= m_dist (the direction flips)
        "probe"       if the agent turns back sooner
        "unresolved"  if the run ends first
    Columns: time (flip time), type, distance (travelled against the direction).
    """
    t = np.asarray(t, dtype=float)
    ds = np.diff(np.asarray(s, dtype=float))
    moving = np.flatnonzero(ds != 0)                  # row k -> k+1 moves
    if len(moving) == 0:
        return pd.DataFrame(columns=["time", "type", "distance"])
    sign = np.sign(ds[moving])
    first = np.r_[True, sign[1:] != sign[:-1]]        # first move of each straight stretch
    stretch_time = t[moving[first]]                   # the row where the stretch starts = the flip time
    stretch_sign = sign[first].astype(int)
    stretch_length = np.add.reduceat(np.abs(ds[moving]), np.flatnonzero(first))

    events = []
    direction = 0      # sign of the last move (0 before the first move)
    start = None       # flip time of the open candidate (None = no candidate)
    dist = 0.0         # distance against the direction in the open candidate
    for time, d, length in zip(stretch_time, stretch_sign, stretch_length):
        if direction == 0:                            # first move: sets the direction
            direction = d
        elif d == direction:                          # move in the direction
            if start is not None:
                events.append({"time": start, "type": "probe", "distance": dist})
                start = None
        else:                                         # move against the direction
            if start is None:
                start, dist = time, 0.0
            dist += length
            if dist >= m_dist - 1e-9:
                events.append({"time": start, "type": "reversal", "distance": dist})
                direction = -direction
                start = None
    if start is not None:
        events.append({"time": start, "type": "unresolved", "distance": dist})
    return pd.DataFrame(events, columns=["time", "type", "distance"])


def observe(t, S, style, tick=0.1):
    """The rows a log would keep. "lab" = labclock.lab_log(t, S, tick);
    "cloud" = the first row and every row where some round(10 s) changes, at exact times. Returns (t, S)."""
    if style == "lab":
        return lab_log(t, S, tick)
    if style == "cloud":
        k = to_steps(S)
        keep = np.r_[True, np.any(np.diff(k, axis=0) != 0, axis=1)]
        return np.asarray(t)[keep], np.asarray(S)[keep]
    raise ValueError(f"style must be 'lab' or 'cloud', not {style!r}")


def true_table(runs, m_dist=0.2):
    """true_events of every agent of every run: run (index in runs), agent (0-based), time, type."""
    rows = []
    for r, run in enumerate(runs):
        for j in range(run["X"].shape[1]):
            for ev in true_events(run["t"], run["X"][:, j], m_dist).itertuples():
                rows.append({"run": r, "agent": j, "time": ev.time, "type": ev.type})
    return pd.DataFrame(rows, columns=["run", "agent", "time", "type"])


def detected_table(runs, style, m=2, tick=0.1):
    """detect_events (m steps) on the observed rows of every agent of every run: run, agent, time, type."""
    rows = []
    for r, run in enumerate(runs):
        t_obs, S_obs = observe(run["t"], run["X"], style, tick)
        for j in range(S_obs.shape[1]):
            for ev in detect_events(t_obs, S_obs[:, j], m).itertuples():
                rows.append({"run": r, "agent": j, "time": ev.time, "type": ev.type})
    return pd.DataFrame(rows, columns=["run", "agent", "time", "type"])


# ------------------------------------------------------------------ scoring

def match(det_times, true_times, used, tol):
    """Each detected time, in order, goes to the closest true time within ±tol that is not used yet.
    true_times sorted; used = set of true indices already taken (updated here). Returns [(det index, true index)]."""
    pairs = []
    for a, x in enumerate(det_times):
        lo = np.searchsorted(true_times, x - tol - 1e-9)
        hi = np.searchsorted(true_times, x + tol + 1e-9, side="right")
        free = [b for b in range(lo, hi) if b not in used]
        if free:
            b = min(free, key=lambda b: abs(x - true_times[b]))
            used.add(b)
            pairs.append((a, b))
    return pairs


def score(true_ev, det_ev, tol=0.2):
    """Detected vs. true events, per agent (tables from true_table and detected_table).

    Reversals: each detected reversal goes to the closest unmatched true reversal within ±tol (a hit).
    recall = hits / true reversals, precision = hits / detected reversals, median error = median of
    (detected - true) time over the hits.
    Fate of each true probe: "probe" if a detected probe is matched to it (same rule), else "reversal" if a
    detected reversal that matched no true reversal is, else "nothing".
    """
    errors = []
    fate = {"probe": 0, "reversal": 0, "nothing": 0}
    det_by_agent = {key: d for key, d in det_ev.groupby(["run", "agent"])}
    for key, tr in true_ev.groupby(["run", "agent"]):
        de = det_by_agent.get(key, det_ev.iloc[0:0])
        true_rev = tr.loc[tr["type"] == "reversal", "time"].to_numpy()
        true_probe = tr.loc[tr["type"] == "probe", "time"].to_numpy()
        det_rev = de.loc[de["type"] == "reversal", "time"].to_numpy()
        det_probe = de.loc[de["type"] == "probe", "time"].to_numpy()

        pairs = match(det_rev, true_rev, set(), tol)
        errors += [det_rev[a] - true_rev[b] for a, b in pairs]

        used = set()
        fate["probe"] += len(match(det_probe, true_probe, used, tol))
        false_rev = np.delete(det_rev, [a for a, _ in pairs])
        fate["reversal"] += len(match(false_rev, true_probe, used, tol))
        fate["nothing"] += len(true_probe) - len(used)

    n_true = int(np.sum(true_ev["type"] == "reversal"))
    n_det = int(np.sum(det_ev["type"] == "reversal"))
    n_probe = int(np.sum(true_ev["type"] == "probe"))
    return {"true reversals": n_true, "detected reversals": n_det, "hits": len(errors),
            "recall": len(errors) / n_true, "precision": len(errors) / n_det,
            "median error (s)": float(np.median(errors)), "true probes": n_probe,
            "probe -> probe": fate["probe"] / n_probe, "probe -> nothing": fate["nothing"] / n_probe,
            "probe -> reversal": fate["reversal"] / n_probe}


# ------------------------------------------------------------------ tables

def run_summary(runs_by_topology):
    """Per topology: runs, % success, median run length (s), agent-minutes."""
    out = {}
    for topology, runs in runs_by_topology.items():
        length = np.array([run["t"][-1] for run in runs])
        out[topology] = {"runs": len(runs), "success %": 100 * np.mean([run["success"] for run in runs]),
                         "median length (s)": np.median(length), "agent-min": 7 * length.sum() / 60}
    return pd.DataFrame(out).T


def detector_table(runs, settings=SETTINGS, tol=0.2):
    """One row per detector setting: true reversals and probes per agent-minute, recall, precision,
    median timing error, fate of true probes, and the logged probe-to-reversal ratio."""
    agent_min = sum(7 * run["t"][-1] / 60 for run in runs)
    truth = {}
    out = {}
    for name, (style, tick, m) in settings.items():
        if m not in truth:
            truth[m] = true_table(runs, m_dist=m / 10)
        true_ev = truth[m]
        det_ev = detected_table(runs, style, m, tick)
        sc = score(true_ev, det_ev, tol)
        out[name] = {"true rev/agent-min": sc["true reversals"] / agent_min,
                     "true probes/agent-min": sc["true probes"] / agent_min,
                     "recall": sc["recall"], "precision": sc["precision"], "median error (s)": sc["median error (s)"],
                     "probe -> probe": sc["probe -> probe"], "probe -> nothing": sc["probe -> nothing"],
                     "probe -> reversal": sc["probe -> reversal"],
                     "logged probe/rev": np.sum(det_ev["type"] == "probe") / sc["detected reversals"]}
    return pd.DataFrame(out).T

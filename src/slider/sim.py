"""Path A, Step 4: a rule-pluggable simulator of the 7-player Slider Game, and a converter to the log format.

Positions are integers n in [-N, N], s = n / N (N = 10 on the 0.1 grid, 50 on the old-CL 0.02 grid); one tick
is dt = d / speed. Detector grid: k = round(10 s), as in reversals.to_steps. R replicates of one round run at
once as (R, 7) arrays; a replicate that has ended is dropped from the arrays.

Modes of a player:
    not_started  s = 0, no direction
    halted       mid-slider
    moving       one step per tick in direction phi
    at_wall      |s| = 1, pinned
    reversing    scripted: one step per tick against phi until k has moved 2 steps against phi, then
                 phi := -phi and moving; if the wall stops it first: at_wall and phi := -phi

Features, each tick, from the positions at the tick start t (hazard.build_hazard_table definitions):
    A       |sum of s over i and its neighbours| / their number (i included), in [0, 1]
    abs_s   |s|
    at_end  1 if |k| >= 9 (crossings.BAND)
    e       log(1 + time since entering the current end band) if at_end, else 0
    r       log(1 + time since the first step of the last reversal, or since the first move if none)
    lab     1 = EL / CL, 0 = CC
    trend   A(t) - A at the last tick at or before t - 1 s (A at the round start if t < 1 s); M3 only
The first move and a reversal's first step are steps of k, as in reversals.detect_events (m = 2): the first
change of k, and the first step of k against the direction. On the 0.1 grid these are the player's own steps;
on the 0.02 grid they can come a few ticks later. The reversal hazard is 0 before the first move.
Decisions (2026-10-29):
    - Old CL (0.02 grid) keeps hazard.py's definitions, not the |s| wording of the spec: at_end = |k| >= 9
      (|s| >= 0.86 there), first move = the first change of k, tau_r resets at a reversal's first step of k.
    - CC features come from the snapshot at each 0.033 s tick, as built (hazard.py computed them on CC rows
      put on the 0.1 s lab clock).

Tick j, from t = j dt to t + dt (synchronous):
    (a) features from the positions at t
    (b) one uniform u per player, by mode: not_started: u < p_start -> moving, coin-flip phi;
        halted: u < h -> reversal, else u < h + p_resume -> moving; moving: u < h -> reversal,
        else u < h + p_stop -> halted; at_wall: u < h -> reversal; reversing: no draw
    (c) moving players step +phi, reversing players -phi (a reversal takes its first step in the tick it fires);
        a player reaching |s| = 1 is at_wall
    (d) detector and time features updated with the new positions, which belong to t + dt
    (e) the round ends on consensus (see consensus) or when the next tick would pass the time limit T
Every position change is an event (time, seat, s). The round start is also recorded: every seat at s = 0 at
t = 0, like the all-zero first row the logs open with.
"""
import numpy as np
import pandas as pd

from slider import config
from slider import slider_logs as sl
from slider.crossings import BAND

MODES = ["not_started", "halted", "moving", "at_wall", "reversing"]
NOT_STARTED, HALTED, MOVING, AT_WALL, REVERSING = range(5)

# grid step d, speed (units/s; the most common file speed of each interface in interface_table.csv), lab flag,
# session (time limit in config.TIME_LIMIT), tag in the group name (so interface.version_tag reads it back)
IFACES = {
    "lab_v09": {"d": 0.1, "speed": 1.0, "lab": 1, "session": "lab", "tag": "v09"},
    "old_cl": {"d": 0.02, "speed": 0.2, "lab": 1, "session": "lab", "tag": "oldcl"},
    "cc": {"d": 0.1, "speed": 3.03, "lab": 0, "session": "cloud", "tag": "cc"},
}
LOGGERS = {"lab": ["lab_v09", "old_cl"], "whole_second": ["old_cl"], "cloud": ["cc"]}   # logger: interfaces
M = 2               # a reversal: k moves M steps against the direction (reversals.detect_events, m = 2)
BIN = 0.25          # the hazard models give h per 0.25 s bin
CONSENSUS = 0.99    # the round ends when |sum of s| / 7 >= 0.99 (see consensus)


# ------------------------------------------------------------------ rules

def per_tick(h, dt):
    """Probability per tick dt of a hazard h per 0.25 s bin."""
    return 1 - (1 - h) ** (dt / BIN)


def coefs(model):
    """{term: beta} of model ("M2'" or "M3") in results/tables/p11_coefs.csv."""
    c = pd.read_csv(config.P11_COEFS)
    c = c[c["model"] == model]
    return dict(zip(c["term"], c["beta"]))


# p11_coefs.csv term -> feature name here
TERMS = {"const": "const", "lab": "lab", "A": "A", "abs_s": "abs_s", "at_end": "at_end", "trend": "trend",
         "time_at_end_log1p": "e", "time_since_rev_log1p": "r"}


class Logistic:
    """M2' or M3: logit h = beta . features, h per 0.25 s; prob = per_tick(h, dt)."""

    def __init__(self, model):
        self.beta = {TERMS[term]: b for term, b in coefs(model).items()}
        self.trend = "trend" in self.beta

    def prob(self, f, dt):
        z = self.beta["const"] + sum(b * f[name] for name, b in self.beta.items() if name != "const")
        return per_tick(1 / (1 + np.exp(-z)), dt)


class APEM:
    """prob = 1 - exp(-dt / mu) if A <= a and abs_s >= s_th, else 0."""
    trend = False

    def __init__(self, a, s_th, mu):
        self.a, self.s_th, self.mu = a, s_th, mu

    def prob(self, f, dt):
        return np.where((f["A"] <= self.a) & (f["abs_s"] >= self.s_th), 1 - np.exp(-dt / self.mu), 0.0)


def preset(name, **apem):
    """{rule, motion, script} of a named model, to pass to simulate as **preset(name).
    L1 = (M2', L1, script), L1_M3 = (M3, L1, script), L0 = (M2', none, script), APEM = (APEM, none, no script);
    APEM needs a, s_th and mu."""
    if name == "APEM":
        return {"rule": APEM(**apem), "motion": "none", "script": False}
    model, motion = {"L1": ("M2'", "L1"), "L1_M3": ("M3", "L1"), "L0": ("M2'", "none")}[name]
    return {"rule": Logistic(model), "motion": motion, "script": True}


def rates(iface):
    """{lam_stop, lam_resume, lam_start} (per second) of an interface in results/tables/p17_rates.csv."""
    r = pd.read_csv(config.P17_RATES)
    r = r[r["interface"] == iface]
    return dict(zip(r["rate"], r["estimate"]))


# ------------------------------------------------------------------ the simulator

def consensus(s):
    """True for each row of s (sign-corrected positions, last axis = the 7 seats) with |sum of s| / 7 >= 0.99.
    fraught.round_end reads the logged end marker and has no consensus test of its own; this rule gives the
    last row of every train and val round, and no earlier row, in all three interfaces. On the 0.1 grid it
    means every slider at the same end; on the old-CL 0.02 grid one slider may be at 0.94 (as logged)."""
    s = np.asarray(s, dtype=float)
    return np.abs(s.sum(axis=-1)) >= CONSENSUS * s.shape[-1] - 1e-9


def constants(A, iface, rule, motion="L1", script=True, speed=None):
    """What every tick needs: adjacency with self-loops, grid, dt, lab flag, motion probabilities, trend lag."""
    f = IFACES[iface]
    A_log = np.asarray(A, dtype=float) + np.eye(7)
    N = int(round(1 / f["d"]))
    dt = f["d"] / (speed or f["speed"])
    if motion == "L1":
        lam = rates(iface)
        p = {name: 1 - np.exp(-lam["lam_" + name] * dt) for name in ["stop", "resume", "start"]}
    elif motion == "none":
        p = {"stop": 0.0, "resume": 0.0, "start": 0.0}
    else:
        raise ValueError(f"motion must be 'L1' or 'none', not {motion!r}")
    lag = int(np.ceil(1 / dt - 1e-9))        # ticks back to the last tick at or before t - 1 s
    return {"A_log": A_log, "deg": A_log.sum(axis=1), "N": N, "dt": dt, "lab": f["lab"], "p": p,
            "motion": motion, "script": script, "rule": rule, "lag": lag}


def to_k(n, N):
    """k = round(10 s) of s = n / N (no ties on either grid)."""
    return np.round(10 * n / N).astype(int)


def start(R, c, rng):
    """State at t = 0: every slider at 0. motion "L1": not started; "none": moving, coin-flip direction."""
    z = np.zeros((R, 7), dtype=int)
    st = {"n": z.copy(), "mode": np.full((R, 7), NOT_STARTED), "phi": z.copy(),
          "k0": z.copy(),                                    # k when the current reversal fired
          "dir": z.copy(),                                   # detector direction (0 before the first move)
          "open": np.zeros((R, 7), dtype=bool),              # a step of k against dir is pending
          "open_t": np.zeros((R, 7)), "open_n": z.copy(),    # its first step's time and steps so far
          "anchor": np.zeros((R, 7)),                        # time of the first move or the last reversal
          "band": z.copy(), "entered": np.zeros((R, 7)),    # end band (+1 / -1 / 0) and when it was entered
          "hist": np.zeros((R, c["lag"] + 1, 7))}            # A of the last lag + 1 ticks (for trend)
    if c["motion"] == "none":
        st["mode"][:] = MOVING
        st["phi"] = np.where(rng.random((R, 7)) < 0.5, 1, -1)
    return st


def features(st, j, c):
    """Features of every player at the start of tick j, from the positions at t = j dt."""
    t = j * c["dt"]
    s = st["n"] / c["N"]
    A = np.abs(s @ c["A_log"].T) / c["deg"]
    at_end = st["band"] != 0
    f = {"A": A, "abs_s": np.abs(s), "at_end": at_end.astype(float),
         "e": np.where(at_end, np.log1p(t - st["entered"]), 0.0),
         "r": np.log1p(t - st["anchor"]), "lab": float(c["lab"])}
    L = c["lag"] + 1
    st["hist"][:, j % L] = A
    if c["rule"].trend:
        f["trend"] = A - st["hist"][:, (j - c["lag"]) % L if j >= c["lag"] else 0]
    return f


def tick(st, j, c, rng):
    """Tick j of every replicate in st (changed in place). Returns (mode during the tick, step) per player."""
    n, phi, N = st["n"], st["phi"], c["N"]
    mode = st["mode"]
    t1 = (j + 1) * c["dt"]

    # (a) features; the hazard is off before the first move and while reversing
    f = features(st, j, c)
    at_risk = (st["dir"] != 0) & (mode != REVERSING) & (mode != NOT_STARTED)
    h = np.where(at_risk, c["rule"].prob(f, c["dt"]), 0.0)

    # (b) one uniform per player
    u = rng.random(n.shape)
    p = c["p"]
    rev = at_risk & (u < h)
    begin = (mode == NOT_STARTED) & (u < p["start"])
    resume = (mode == HALTED) & ~rev & (u < h + p["resume"])
    stop = (mode == MOVING) & ~rev & (u < h + p["stop"])
    phi[begin] = np.where(rng.random(int(begin.sum())) < 0.5, 1, -1)     # coin flip
    mode[begin | resume] = MOVING
    mode[stop] = HALTED
    if c["script"]:
        mode[rev] = REVERSING
        st["k0"][rev] = to_k(n[rev], N)
    else:
        phi[rev] = -phi[rev]
        mode[rev] = MOVING
    during = mode.copy()

    # (c) move
    step = np.where(mode == MOVING, phi, np.where(mode == REVERSING, -phi, 0))
    k_old = to_k(n, N)
    n += step
    k = to_k(n, N)
    wall = np.abs(n) == N
    reversing = mode == REVERSING
    flip = reversing & (((st["k0"] - k) * phi >= M) | wall)    # script done, or the wall stopped it
    phi[flip] = -phi[flip]
    mode[flip] = MOVING
    mode[(mode == MOVING) & wall] = AT_WALL

    # (d) detector (reversals.detect_events, m = 2) and end band, at t + dt
    dk = k - k_old
    sign = np.sign(dk)
    first = (dk != 0) & (st["dir"] == 0)
    st["dir"][first] = sign[first]
    st["anchor"][first] = t1
    along = (dk != 0) & ~first & (sign == st["dir"])
    st["open"][along] = False                                   # a probe ends
    against = (dk != 0) & ~first & (sign == -st["dir"])
    new = against & ~st["open"]
    st["open_t"][new] = t1
    st["open_n"][new] = 0
    st["open"] |= against
    st["open_n"] += np.where(against, np.abs(dk), 0)
    done = against & (st["open_n"] >= M)                        # a reversal: its first step is the anchor
    st["anchor"][done] = st["open_t"][done]
    st["dir"][done] = -st["dir"][done]
    st["open"][done] = False
    band = np.sign(k) * (np.abs(k) >= BAND)
    st["entered"][band != st["band"]] = t1
    st["band"] = band
    return during, step


def simulate(A, T, iface, rule, motion="L1", script=True, R=100, seed=None, speed=None):
    """R replicates of one round on adjacency A (7 x 7, seat order, no self-loops) with time limit T (s).

    iface: "lab_v09", "old_cl" or "cc" (grid, speed, lab flag, motion rates); speed overrides the interface's.
    rule: an object with prob(features, dt) and trend (True if it needs trend); motion: "L1" or "none";
    script: True = scripted reversals, False = a reversal flips phi at once. preset(name) gives the three.
    Returns one dict per replicate: events (DataFrame time, seat, s; the round start first), end_time (s),
    ticks, success (consensus before T), mode_ticks ((7, 5) ticks per seat in each of MODES), dt.
    """
    rng = np.random.default_rng(seed)
    c = constants(A, iface, rule, motion, script, speed)
    st = start(R, c, rng)
    ids = np.arange(R)                       # replicate of each row of st
    counts = np.zeros((R, 7, len(MODES)), dtype=int)
    ends = np.zeros(R, dtype=int)
    success = np.zeros(R, dtype=bool)
    ev = []                                  # (replicate, tick, seat, n) of each position change
    n_ticks = int(np.floor(T / c["dt"] + 1e-9))
    for j in range(n_ticks):
        during, step = tick(st, j, c, rng)
        counts[ids] += during[:, :, None] == np.arange(len(MODES))
        rows, seats = np.nonzero(step)
        ev.append(np.column_stack([ids[rows], np.full(len(rows), j + 1), seats, st["n"][rows, seats]]))
        over = consensus(st["n"] / c["N"])
        if over.any():
            ends[ids[over]] = j + 1
            success[ids[over]] = True
            st = {key: value[~over] for key, value in st.items()}
            ids = ids[~over]
            if len(ids) == 0:
                break
    ends[ids] = n_ticks

    ev = np.concatenate(ev) if ev else np.zeros((0, 4), dtype=int)
    ev = ev[np.argsort(ev[:, 0], kind="stable")]
    parts = np.split(ev, np.searchsorted(ev[:, 0], np.arange(1, R)))
    out = []
    for r, e in enumerate(parts):
        events = pd.DataFrame({"time": np.r_[np.zeros(7), e[:, 1] * c["dt"]],
                               "seat": np.r_[np.arange(7), e[:, 2]],
                               "s": np.r_[np.zeros(7), e[:, 3] / c["N"]]})
        out.append({"events": events, "end_time": ends[r] * c["dt"], "ticks": int(ends[r]),
                    "success": bool(success[r]), "mode_ticks": counts[r], "dt": c["dt"]})
    return out


# ------------------------------------------------------------------ to the log format

def ffill_rows(row, seat, s, n_rows):
    """Positions of each row: event (row, seat, s) sets the seat from that row on; 0 before its first event."""
    X = np.full((n_rows, 7), np.nan)
    X[row, seat] = s
    return pd.DataFrame(X).ffill().fillna(0.0).to_numpy()


def to_run(events, meta, logger, rng=None):
    """One round as slider_logs.load_runs returns its runs: sign-corrected (all weights +1), all seats human.

    events: time, seat (0-based), s of every position change (the round start too), in time order, e.g.
    simulate's events.
    meta: group (= file), dataset (EL / CL / CC), topology, wave (round id), graph (7 x 7 node adjacency, no
    self-loops), seat_node (seat i plays node seat_node[i]), interface (IFACES), time_limit (s; must be
    config.TIME_LIMIT of the interface's session), end ("success" or "failed"), seed (for the cloud logger).
    logger:
        "lab"           one row per time with any change (all events at that time), t rounded to 0.01 s
        "cloud"         one mover per row; moves at the same time go into rows 1 or 2 ms apart, in random
                        order; a time with no change (the round start) is one row; t rounded to 0.001 s
        "whole_second"  as "lab", then t cut to whole seconds (floor, as in the whole-second CL files)
    agree = slider_logs.agreement of the rows; general = their mean.
    """
    session = IFACES[meta["interface"]]["session"]
    if meta["interface"] not in LOGGERS[logger]:
        raise ValueError(f"logger {logger!r} does not log interface {meta['interface']!r}")
    if meta["time_limit"] != config.TIME_LIMIT[session]:
        raise ValueError(f"time_limit {meta['time_limit']} is not the {session} time limit {config.TIME_LIMIT[session]}")
    time = np.asarray(events["time"], dtype=float)
    seat = np.asarray(events["seat"], dtype=int)
    s = np.asarray(events["s"], dtype=float)

    if logger in ("lab", "whole_second"):
        o = np.argsort(time, kind="stable")
        time, seat, s = np.round(time[o], 2), seat[o], s[o]
        t, row = np.unique(time, return_inverse=True)
        X = ffill_rows(row, seat, s, len(t))
        if logger == "whole_second":
            t = np.floor(t + 1e-9)
    else:
        rng = rng if rng is not None else np.random.default_rng(meta.get("seed"))
        group = np.unique(time, return_inverse=True)[1]
        key = rng.random((group.max() + 1, 7))[group, seat]    # seats in random order within a time,
        o = np.lexsort((np.arange(len(time)), key, time))       # each seat's own events kept in order
        time, seat, s = time[o], seat[o], s[o]
        prev = np.zeros(len(s))                                 # the seat's value before this event
        by_seat = np.lexsort((np.arange(len(s)), seat))
        same = np.r_[False, seat[by_seat][1:] == seat[by_seat][:-1]]
        prev[by_seat[same]] = s[by_seat][:-1][same[1:]]
        group = np.unique(time, return_inverse=True)[1]
        mover = s != prev
        quiet = ~np.isin(group, group[mover])                   # times with no change: keep one row
        keep = mover | (quiet & np.r_[True, group[1:] != group[:-1]])
        time, seat, s, group, mover = time[keep], seat[keep], s[keep], group[keep], mover[keep]
        gap = rng.integers(1, 3, len(time)) * 0.001
        first = np.r_[True, group[1:] != group[:-1]]
        gap[first] = 0.0
        offset = np.cumsum(gap)
        offset -= offset[np.flatnonzero(first)][np.cumsum(first) - 1]
        t = np.round(time + offset, 3)
        X = ffill_rows(np.arange(len(t)), seat, np.where(mover, s, np.nan), len(t))

    graph = np.asarray(meta["graph"], dtype=int)
    order = list(meta["seat_node"])
    A = graph[np.ix_(order, order)]
    A_log = A + np.eye(7, dtype=int)
    agree = sl.agreement(A_log, X)
    return {"file": meta["group"], "session": session, "wave": meta["wave"], "name": meta["topology"],
            "A_log": A_log, "A": A, "weights": np.ones(7), "t": t, "X": X, "agree": agree,
            "general": agree.mean(axis=1), "end": meta["end"], "bot": False, "bot_players": set(),
            "dataset": meta["dataset"]}


def table(runs, role="train"):
    """The table slider_logs.load_runs returns with runs, for runs from to_run (no bots). role: one for all runs,
    or one per run. coarse_time is per run (all rounds of a whole-second file are whole-second)."""
    out = sl.runs_table(runs)
    out["role"] = role
    out["has_bot"] = False
    out["bot_players"] = [()] * len(out)
    out["coarse_time"] = [bool(np.all(run["t"] == np.round(run["t"]))) for run in runs]
    return out


def simulate_runs(A, topology, iface, model, R=100, seed=None, dataset=None, logger=None, role="train", **apem):
    """simulate + to_run: R simulated rounds as (runs, table), like slider_logs.load_runs.

    model: a preset name (L1, L1_M3, L0, APEM; APEM takes a, s_th, mu). T = the session's time limit.
    dataset: default CL for lab_v09 and old_cl, CC for cc. logger: default "lab" for lab interfaces, "cloud" for cc.
    Group (file) of replicate r: sim_<tag>_<topology>_<model>_<seed>_r<r>.txt, one round each (wave 1).
    """
    f = IFACES[iface]
    T = config.TIME_LIMIT[f["session"]]
    sim_seed, log_seed = np.random.SeedSequence(seed).spawn(2)
    out = simulate(A, T, iface, R=R, seed=sim_seed, **preset(model, **apem))
    log_rng = np.random.default_rng(log_seed)
    dataset = dataset or ("CC" if iface == "cc" else "CL")
    logger = logger or ("cloud" if iface == "cc" else "lab")
    runs = []
    for r, o in enumerate(out):
        meta = {"group": f"sim_{f['tag']}_{topology}_{model}_{seed}_r{r:03d}.txt", "dataset": dataset,
                "topology": topology, "wave": 1, "graph": A, "seat_node": range(7), "interface": iface,
                "time_limit": T, "end": "success" if o["success"] else "failed", "seed": seed}
        runs.append(to_run(o["events"], meta, logger, log_rng))
    return runs, table(runs, role)

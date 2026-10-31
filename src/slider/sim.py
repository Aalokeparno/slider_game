"""Path A, Step 4: a rule-pluggable simulator of the 7-player Slider Game, and a converter to the log format.

Positions are integers n in [-N, N], s = n / N (N = 10 on the 0.1 grid, 50 on the old-CL 0.02 grid); one tick
is dt = d / speed. Detector grid: k = round(10 s), as in reversals.to_steps. R replicates of one round run at
once as (R, 7) arrays; a replicate that has ended is dropped from the arrays.

Modes of a player:
    not_started  s = 0, no direction
    halted       mid-slider
    moving       one step per tick in direction phi
    at_wall      |s| = 1, pinned
    reversing    protected: one step per tick in direction phi and no draws, while the reversal is open on the
                 detector (reversals.detect_events, m = 2: from its first step of k against the direction to the
                 step of k that confirms it)

Fixes after the P23 recovery (approved Oct 29, 2026):
F1  Motion probabilities per tick as motion.py counts the P17 rates: p_stop = lam_stop dt (stops per moving
    step), p_resume = lam_resume dt (resumes per halted tick), p_start = lam_start dt / (1 + lam_start dt)
    (motion.py's start delay leaves out the start tick, so it counts starts per tick before the first step).
F2  M2' / M3 reversals are drawn per estimator bin, exactly as hazard.build_hazard_table defines bins:
    bins     0.25 s from the player's first move (the first change of k on the estimator's rows): t_first + 0.25 m
    rows     what hazard.py reads: lab interfaces, the logged rows (one per tick, times rounded to 0.01 s);
             CC, the 0.1 s lab clock (labclock.lab_log: the positions at the last tick at or before each 0.1 s)
    features at the bin start tau, positions from the last row strictly before tau, times at tau:
        A       |sum of s over i and its neighbours| / their number (i included), in [0, 1]
        abs_s   |s|;  at_end: 1 if |k| >= 9 (crossings.BAND)
        e       log(1 + tau - entry time of the current end band) if at_end, else 0
        r       log(1 + tau - the first step of the last reversal confirmed before tau, or the first move)
        lab     1 = EL / CL, 0 = CC
        trend   A - A at the last row strictly before tau - 1 s (the first row's A if none); M3 only
    a bin that starts inside an open reversal has no hazard (hazard.py drops it as mid-event)
    the bin's decisions are the ticks whose first step the estimator dates inside the bin (lab: the step's row
    time; CC: the first 0.1 s lab tick at or after it), after the first move; each fires with probability
    1 - (1 - h)^(1/n) for n decisions, so the bin fires with probability h; at most one reversal per bin.
    APEM stays per tick: 1 - exp(-dt / mu) if A <= a and abs_s >= s_th, at the positions of the tick start.
F3  A reversal flips phi at once. With script=True the player is protected (reversing) while the reversal is open
    on the detector. On the 0.1 grid the first step after the flip opens it, so a reversal is the 2-step script as
    before; on the old-CL 0.02 grid the steps before k changes are ordinary moving steps (they can stop or reverse
    again). With script=False (APEM) nothing is protected. A player reaching |s| = 1 is at_wall.
P25 additions (pre-registered final model = L1 + L1b + L3; L1 itself is unchanged):
L1b  motion "L1b": a mid-round halt (a stop) draws its type when it starts, fast with probability p; while halted
     the player resumes per tick with q_f or q_s (halts.py, results/tables/p25_halts.csv, per interface). Stops and
     the start delay keep the L1 rates. The reversal hazard still runs while halted. With p = 1 (or 0) no type is
     drawn, so p = 1 and q_f = lam_resume dt is exactly L1.
L3   offset: one value u per replicate added to logit h of every player (Logistic rules only). design.py draws
     u_g ~ N(0, sigma_g^2) once per simulated group copy (frailty.py, results/tables/p25_sigma.csv) and passes it
     to every round of that group; offset None or 0 is exactly the model without it.
Model B, B1 (EXPLORATORY, results/model_b_log.md entry 1; L1, final, L1_M3, L0 and APEM unchanged, bit-identical):
B1   motion "B1" = L1 (lam_start, coin flip, L1's resume chance lam_resume dt) with stops by state. A player is in C
     (crossing attempt) from the tick whose step leaves an end band (tick detector: |k| from >= 9 to < 9) until it
     enters an end band or the detector confirms a reversal whose first step came after that tick; else in E. On
     entering C it draws the crossing's style: smooth with probability p_i (no draw if p_i is 0 or 1). Stop chance
     per moving step: C smooth 0, C stop-and-go q_sg, E q_E (b1_params: pB1_params.csv). p_i = the Beta(a, b)
     quantile at the player's U_i (trait, one per seat; design.py draws it once per group copy), a = pi (1 - rho)
     / rho, b = (1 - pi)(1 - rho) / rho. If the parameters hold resume_C / resume_E, a halt resumes per tick with
     the one of its state at the start (not used by B1: step 0's rule chose L1's single chance).
Model C, C1 (EXPLORATORY, results/model_b_log.md entry 2; every other preset unchanged, bit-identical):
C1   motion "C1", its own tick (tick_c1): start late (L1's p_start, coin-flip wall), cross in human time, sit at the
     wall, switch by the frozen M2' hazard. A trip (the first trip: N steps from 0; a crossing: 2N steps to the
     opposite wall) draws its duration D from the human durations of the interface (model_c1.durations,
     pC_durations.parquet: first trips / wall-to-wall trips) and takes L = max(n, round(D / dt) + 1) ticks, its
     first step in the start / switch tick: by the end of trip tick i it has taken i (n - 1) // (L - 1) + 1 steps
     (ticks with a step are moving, the others halted), so the logs measure round(D / dt) dt. No stops or
     reversals in a trip. The M2' hazard runs as in F2, at risk only in bins whose estimator row before the bin
     start shows |s| = 1 (and no open reversal), and only while at_wall; a switch flips phi and starts a crossing.
Model C, C2 (EXPLORATORY, results/model_b_log.md entry 3; every other preset unchanged, bit-identical):
C2   motion "C2", its own tick (tick_c2). Start and first trip as C1. At the wall the player sits; the try hazard
     (TryLogistic: T0 or T1 of pC2_try_hazard.csv, per pC2_params.csv use_g) runs per F2 bin, at risk as in C1;
     T1's g = log(1 + tau - the last estimator row strictly before tau on which a neighbor's s changed). A try is
     one step off the wall. After every own step (a try or a walk step) in tick j the player waits for its decision
     point, tick j + D (D = max(1, round(d_ar / dt))), and looks: dA = A at the start of tick j + D - A at the start
     of tick j. It keeps with probability keep(sign(dA)) (pC2_params.csv, per interface): it moves on in the
     direction of that step to the wall, n steps at pace f (pC2_pace.parquet, the same interface and round speed;
     min(f, 1) steps per tick) over max(n, round((n - 1) / f) + 1) ticks (trip_steps), the first in tick j + D.
     Otherwise it walks: from tick j + D, a step with chance q_w per tick, up or down 50/50, then the next look.
     Reaching |s| = 1 ends a walk or a kept movement: the player sits. M2' is not used.
Not matched (measurement dating): on the 0.02 grid the estimator dates a reversal at its first step of k, up to
4 ticks after the flip; in CC the cloud logger splits one tick's moves into rows 1-2 ms apart (after the
simulation), so a lab tick that falls inside such a split sees part of the tick.

Decisions (2026-10-29): old CL keeps hazard.py's definitions (at_end = |k| >= 9, first move = first change of k,
tau_r resets at a reversal's first step of k). CC features first came from the 0.033 s tick snapshot; F2 replaced
that with the 0.1 s lab clock, as hazard.py reads CC.

Tick j, from t = j dt to t + dt (synchronous):
    (a) the reversal hazard h of every player (0 before the first move, while reversing or not started)
    (b) one uniform u per player, by mode: not_started: u < p_start -> moving, coin-flip phi;
        halted: u < h -> reversal, else u < h + p_resume -> moving; moving: u < h -> reversal,
        else u < h + p_stop -> halted; at_wall: u < h -> reversal; reversing: no draw
    (c) moving and reversing players step +phi (a reversal takes its first step in the tick it fires)
    (d) detector updated with the new positions, which belong to t + dt; the estimator's rows recorded (F2)
    (e) the round ends on consensus (see consensus) or when the next tick would pass the time limit T
Every position change is an event (time, seat, s). The round start is also recorded: every seat at s = 0 at
t = 0, like the all-zero first row the logs open with.
"""
import numpy as np
import pandas as pd
from scipy import stats

from slider import config
from slider import model_c1
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
LAB_CLOCK = 0.1     # labclock.lab_log tick: hazard.py reads CC rows on it
CONSENSUS = 0.99    # the round ends when |sum of s| / 7 >= 0.99 (see consensus)
HORIZON = 400.0     # s: the time grids below cover every time limit


# ------------------------------------------------------------------ rules

def per_tick(h, dt):
    """Probability per tick dt of a hazard h per 0.25 s bin, if h held for the whole bin."""
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
    """M2' or M3: logit h = beta . features, h per 0.25 s bin, drawn per estimator bin (F2)."""
    per_bin = True

    def __init__(self, model):
        self.beta = {TERMS[term]: b for term, b in coefs(model).items()}
        self.trend = "trend" in self.beta

    def h(self, f, u=None):
        """h per bin; u: the L3 group offset of each row (None = no offset)."""
        z = self.beta["const"] + sum(b * f[name] for name, b in self.beta.items() if name != "const")
        if u is not None:
            z = z + u
        return 1 / (1 + np.exp(-z))


class TryLogistic(Logistic):
    """C2: the try hazard T0 or T1 (results/tables/pC2_try_hazard.csv), h per 0.25 s bin, drawn per estimator bin.
    model None: the one pC2_params.csv chose (use_g: T1, else T0)."""

    def __init__(self, model=None, params=None):
        if model is None:
            p = pd.read_csv(params or config.PC2_PARAMS).set_index("parameter")["estimate"]
            model = "T1" if p["use_g"] == 1 else "T0"
        c = pd.read_csv(config.PC2_TRY)
        c = c[c["model"] == model]
        self.model = model
        self.beta = {{"g": "g"}.get(term, TERMS.get(term)): b for term, b in zip(c["term"], c["beta"])}
        self.trend = False


class APEM:
    """Per tick: prob = 1 - exp(-dt / mu) if A <= a and abs_s >= s_th, else 0."""
    per_bin = False

    def __init__(self, a, s_th, mu):
        self.a, self.s_th, self.mu = a, s_th, mu

    def prob(self, f, dt):
        return np.where((f["A"] <= self.a) & (f["abs_s"] >= self.s_th), 1 - np.exp(-dt / self.mu), 0.0)


def preset(name, **apem):
    """{rule, motion, script} of a named model, to pass to simulate as **preset(name); "final" also has sigma_g
    (design.py draws the group offsets from it and removes it before simulate).
    L1 = (M2', L1, script), L1_M3 = (M3, L1, script), L0 = (M2', none, script), APEM = (APEM, none, no script);
    final = (M2', L1b, script) + L3 with sigma_g from p25_sigma.csv (sigma_g=... overrides it).
    B1 (EXPLORATORY) = (M2', B1, script) with a player trait (trait: True; design.py draws U per seat);
    params=path reads the B1 parameters from another file (default pB1_params.csv).
    C1 (EXPLORATORY) = (M2', C1, no script): durations=path reads another duration file (default pC_durations.parquet).
    C2 (EXPLORATORY) = (TryLogistic, C2, no script): params=path reads another pC2_params.csv, model="T0" / "T1"
    overrides its choice.
    APEM needs a, s_th and mu."""
    if name == "APEM":
        return {"rule": APEM(**apem), "motion": "none", "script": False}
    if name == "final":
        return {"rule": Logistic("M2'"), "motion": "L1b", "script": True, "sigma_g": apem.get("sigma_g", sigma_g())}
    if name == "B1":
        return {"rule": Logistic("M2'"), "motion": "B1", "script": True, "trait": True, "b1": apem.get("params")}
    if name == "C1":
        return {"rule": Logistic("M2'"), "motion": "C1", "script": False, "c1": apem.get("durations")}
    if name == "C2":
        return {"rule": TryLogistic(apem.get("model"), apem.get("params")), "motion": "C2", "script": False,
                "c2": apem.get("params")}
    model, motion = {"L1": ("M2'", "L1"), "L1_M3": ("M3", "L1"), "L0": ("M2'", "none")}[name]
    return {"rule": Logistic(model), "motion": motion, "script": True}


def rates(iface):
    """{lam_stop, lam_resume, lam_start} (per second) of an interface in results/tables/p17_rates.csv."""
    r = pd.read_csv(config.P17_RATES)
    r = r[r["interface"] == iface]
    return dict(zip(r["rate"], r["estimate"]))


def halt_params(iface):
    """L1b: {p, q_f, q_s} (q per tick) of an interface in results/tables/p25_halts.csv."""
    t = pd.read_csv(config.P25_HALTS)
    t = t[t["interface"] == iface].set_index("parameter")["estimate"]
    return {k: float(t[k]) for k in ["p", "q_f", "q_s"]}


def b1_params(iface, path=None):
    """B1: {pi, rho, q_sg, q_E} (and resume_C, resume_E if step 0 chose them) of an interface in pB1_params.csv
    (path: another parameter file, e.g. config.PB1_PARAMS_V1)."""
    t = pd.read_csv(path or config.PB1_PARAMS)
    t = t[t["interface"] == iface].set_index("parameter")["estimate"]
    return {k: float(t[k]) for k in ["pi", "rho", "q_sg", "q_E", "resume_C", "resume_E"] if k in t}


def trait_p(U, pi, rho):
    """B1: smooth propensity p_i = the Beta(a, b) quantile at U, a = pi (1 - rho) / rho, b = (1 - pi)(1 - rho) / rho."""
    a, b = pi * (1 - rho) / rho, (1 - pi) * (1 - rho) / rho
    return stats.beta.ppf(U, a, b)


def draw_style(p, rng):
    """B1: smooth (True) with probability p for each entry of p; no draw if every p is 0 or 1."""
    p = np.asarray(p, dtype=float)
    if np.all((p == 0) | (p == 1)):
        return p == 1
    return rng.random(len(p)) < p


def c2_params(iface, speed, path=None, pace=None):
    """C2: {d_ar, q_w, keep_pos, keep_zero, keep_neg, pace} of an interface: pC2_params.csv (path: another file);
    pace = the f of the human crossings of the same interface and round speed (rounded to 0.01) in pC2_pace.parquet
    (pace: another file), the interface's pool if that speed has none, capped at 1."""
    t = pd.read_csv(path or config.PC2_PARAMS)
    p = dict(zip(t.loc[t["interface"] == "all", "parameter"], t.loc[t["interface"] == "all", "estimate"]))
    p.update(dict(zip(t.loc[t["interface"] == iface, "parameter"], t.loc[t["interface"] == iface, "estimate"])))
    f = pd.read_parquet(pace or config.PC2_PACE)
    f = f[f["interface"] == iface]
    same = f[np.isclose(f["round_speed"], round(speed, 2))]
    out = {k: float(p[k]) for k in ["d_ar", "q_w", "keep_pos", "keep_zero", "keep_neg"]}
    out["pace"] = np.minimum((same if len(same) else f)["f"].to_numpy(float), 1.0)
    return out


def sigma_g():
    """L3: sigma_g in results/tables/p25_sigma.csv."""
    return float(pd.read_csv(config.P25_SIGMA)["sigma_g"].iloc[0])


# ------------------------------------------------------------------ the simulator

def consensus(s):
    """True for each row of s (sign-corrected positions, last axis = the 7 seats) with |sum of s| / 7 >= 0.99.
    fraught.round_end reads the logged end marker and has no consensus test of its own; this rule gives the
    last row of every train and val round, and no earlier row, in all three interfaces. On the 0.1 grid it
    means every slider at the same end; on the old-CL 0.02 grid one slider may be at 0.94 (as logged)."""
    s = np.asarray(s, dtype=float)
    return np.abs(s.sum(axis=-1)) >= CONSENSUS * s.shape[-1] - 1e-9


def constants(A, iface, rule, motion="L1", script=True, speed=None, halts=None, b1=None, c1=None, c2=None):
    """What every tick needs: adjacency, grid, dt, lab flag, motion probabilities (F1), the L1b halt types
    (halt: {p, q_f, q_s}, or None; halts overrides p25_halts.csv), and the estimator's rows (F2).

    tick_time[k]  logged time of the positions after tick k - 1 (lab: rounded to 0.01 s; CC: to 0.001 s)
    G[r]          time of the estimator's row r (lab: tick_time; CC: the 0.1 s lab clock, rounded to 0.01 s)
    src[r]        the positions row r shows: those after tick src[r] - 1 (lab_log: last row at or before G + 1e-6)
    date[k]       when the estimator dates a step made in tick k - 1: the first row that shows it
    """
    f = IFACES[iface]
    A_log = np.asarray(A, dtype=float) + np.eye(7)
    N = int(round(1 / f["d"]))
    dt = f["d"] / (speed or f["speed"])
    halt = None
    if motion in ("L1", "L1b", "B1", "C1", "C2"):
        lam = rates(iface)
        x = {name: lam["lam_" + name] * dt for name in ["stop", "resume", "start"]}
        p = {"stop": x["stop"], "resume": x["resume"], "start": x["start"] / (1 + x["start"])}
        if motion == "L1b":
            halt = dict(halts) if halts is not None else halt_params(iface)
    elif motion == "none":
        p = {"stop": 0.0, "resume": 0.0, "start": 0.0}
    else:
        raise ValueError(f"motion must be 'L1', 'L1b', 'B1', 'C1', 'C2' or 'none', not {motion!r}")
    if motion in ("C1", "C2") and not getattr(rule, "per_bin", False):
        raise ValueError(f"{motion}: the hazard must be a per-bin rule (Logistic)")
    ks = np.arange(int(HORIZON / dt) + 2)
    if iface == "cc":
        tick_time = np.round(ks * dt, 3)
        G = np.round(np.arange(int(HORIZON / LAB_CLOCK) + 2) * LAB_CLOCK, 2)
        src = np.searchsorted(tick_time, G + 1e-6, side="right") - 1
    else:
        tick_time = G = np.round(ks * dt, 2)
        src = ks
    date = G[np.minimum(np.searchsorted(src, ks, side="left"), len(G) - 1)]
    row_dt = LAB_CLOCK if iface == "cc" else dt
    return {"A_log": A_log, "deg": A_log.sum(axis=1), "N": N, "dt": dt, "lab": f["lab"], "p": p, "halt": halt,
            "motion": motion, "script": script, "rule": rule, "per_bin": getattr(rule, "per_bin", False),
            "G": G, "src": src, "date": date, "rows": int(np.ceil(1.6 / row_dt)) + 3, "trace": None, "offset": None,
            "b1": (dict(b1) if isinstance(b1, dict) else b1_params(iface, b1)) if motion == "B1" else None,
            "c1": (dict(c1) if isinstance(c1, dict) else model_c1.durations(iface, c1)) if motion in ("C1", "C2") else None,
            "c2": (dict(c2) if isinstance(c2, dict) else c2_params(iface, f["d"] / dt, c2)) if motion == "C2" else None}


def to_k(n, N):
    """k = round(10 s) of s = n / N (no ties on either grid)."""
    return np.round(10 * n / N).astype(int)


DETECTOR = ["dir", "open", "open_t", "open_n", "anchor", "first", "band", "entered"]


def start(R, c, rng):
    """State at t = 0: every slider at 0. motion "L1": not started; "none": moving, coin-flip direction.
    Two detectors (reversals.detect_events, m = 2): "" on the ticks (the reversing script, per-tick rules) and
    "e_" on the estimator's rows (F2), with a ring buffer rb_* of the last rows' features."""
    z, nan = np.zeros((R, 7), dtype=int), np.full((R, 7), np.nan)
    st = {"id": np.arange(R), "n": z.copy(), "mode": np.full((R, 7), NOT_STARTED), "phi": z.copy(),
          "fast": np.ones((R, 7), dtype=bool)}                    # L1b: type of the current halt
    for pre in ["", "e_"]:
        st.update({pre + "dir": z.copy(), pre + "open": np.zeros((R, 7), dtype=bool), pre + "open_t": np.zeros((R, 7)),
                   pre + "open_n": z.copy(), pre + "anchor": nan.copy(), pre + "first": nan.copy(),
                   pre + "band": z.copy(), pre + "entered": np.zeros((R, 7))})
    L = c["rows"]
    st.update({"e_k": z.copy(), "rb_A": np.zeros((R, L, 7)), "rb_abs": np.zeros((R, L, 7)),
               "rb_band": np.zeros((R, L, 7), dtype=int), "rb_entered": np.zeros((R, L, 7)),
               "rb_anchor": np.full((R, L, 7), np.nan), "rb_open": np.zeros((R, L, 7), dtype=bool),
               "b_m": np.full((R, 7), -1), "b_q": np.zeros((R, 7)), "b_risk": np.zeros((R, 7), dtype=bool),
               "b_fired": np.zeros((R, 7), dtype=bool)})
    if c["motion"] == "B1":                                     # B1: crossing state, style, state of the halt
        st.update({"c_on": np.zeros((R, 7), dtype=bool), "c_start": np.zeros((R, 7)),
                   "smooth": np.zeros((R, 7), dtype=bool), "halt_C": np.zeros((R, 7), dtype=bool),
                   "p_smooth": trait_p(c["trait"], c["b1"]["pi"], c["b1"]["rho"])})
    if c["motion"] in ("C1", "C2"):                             # C1: the current trip (tick i of L, n steps; -1 = none)
        st.update({"c1_i": np.full((R, 7), -1), "c1_L": z.copy(), "c1_n": z.copy()})
    if c["motion"] == "C2":                                     # C2: the look (decision tick, A before the step, its
        st.update({"c2_wait": np.full((R, 7), -1), "c2_A0": np.zeros((R, 7)), "c2_dir": z.copy(),   # direction),
                   "c2_walk": np.zeros((R, 7), dtype=bool), "e_s": np.zeros((R, 7)),               # walking; the
                   "e_nb": np.zeros((R, 7)), "rb_nb": np.zeros((R, L, 7))})                         # neighbor steps
    if c["motion"] == "none":
        st["mode"][:] = MOVING
        st["phi"] = np.where(rng.random((R, 7)) < 0.5, 1, -1)
    return st


def detect(st, pre, dk, k, t):
    """One row of reversals.detect_events (m = 2) for every player: direction, the open reversal, the anchor of
    time_since_rev, the first move and the end band. Returns (opened, confirmed) masks."""
    sign = np.sign(dk)
    d = st[pre + "dir"]
    first = (dk != 0) & (d == 0)
    d[first] = sign[first]
    st[pre + "anchor"][first] = t
    st[pre + "first"][first] = t
    along = (dk != 0) & ~first & (sign == d)
    st[pre + "open"][along] = False                             # a probe ends
    against = (dk != 0) & ~first & (sign == -d)
    opened = against & ~st[pre + "open"]
    st[pre + "open_t"][opened] = t
    st[pre + "open_n"][opened] = 0
    st[pre + "open"] |= against
    st[pre + "open_n"] += np.where(against, np.abs(dk), 0)
    done = against & (st[pre + "open_n"] >= M)                  # a reversal: its first step is the anchor
    st[pre + "anchor"][done] = st[pre + "open_t"][done]
    d[done] = -d[done]
    st[pre + "open"][done] = False
    band = np.sign(k) * (np.abs(k) >= BAND)
    st[pre + "entered"][band != st[pre + "band"]] = t
    st[pre + "band"] = band
    return opened, done


def record_rows(st, k, c):
    """The estimator's rows that show the positions after tick k - 1 (lab: one; CC: zero or one lab tick)."""
    N, L = c["N"], c["rows"]
    s = st["n"] / N
    kk = to_k(st["n"], N)
    for row in range(np.searchsorted(c["src"], k, side="left"), np.searchsorted(c["src"], k, side="right")):
        detect(st, "e_", kk - st["e_k"], kk, c["G"][row])
        st["e_k"] = kk
        slot = row % L
        if c["motion"] == "C2":                                 # C2: time of a neighbor's last step on these rows
            moved = (np.abs(s - st["e_s"]) > 1e-9).astype(float)
            st["e_nb"] = np.where(moved @ (c["A_log"] - np.eye(7)).T > 0, c["G"][row], st["e_nb"])
            st["e_s"] = s.copy()
            st["rb_nb"][:, slot] = st["e_nb"]
        st["rb_A"][:, slot] = np.abs(s @ c["A_log"].T) / c["deg"]
        st["rb_abs"][:, slot] = np.abs(s)
        for name in ["band", "entered", "anchor", "open"]:
            st["rb_" + name][:, slot] = st["e_" + name]


def bin_hazard(st, j, c):
    """F2: per-tick reversal probability of every player in tick j from its estimator bin (0 if none or not at risk).
    A bin's hazard is set at its first decision; see the module doc."""
    G, L = c["G"], c["rows"]
    D = c["date"][j + 1]                                        # the estimator's date of a step made now
    first = st["e_first"]
    on = ~np.isnan(first) & (D > first + 1e-6)
    m = np.full(first.shape, -1)
    m[on] = np.floor((D - first[on] + 1e-6) / BIN).astype(int)
    new = on & (m != st["b_m"])
    if new.any():
        reps, players = np.nonzero(new)
        tau = first[new] + BIN * m[new]
        row = np.maximum(np.searchsorted(G, tau - 1e-6, side="right") - 1, 0)    # last row strictly before tau
        slot = row % L
        band, anchor = st["rb_band"][reps, slot, players], st["rb_anchor"][reps, slot, players]
        entered, at_end = st["rb_entered"][reps, slot, players], band != 0
        anchor = np.where(np.isnan(anchor), first[new], anchor)
        f = {"A": st["rb_A"][reps, slot, players], "abs_s": st["rb_abs"][reps, slot, players],
             "at_end": at_end.astype(float), "e": np.where(at_end, np.log1p(tau - entered), 0.0),
             "r": np.log1p(tau - anchor), "lab": float(c["lab"])}
        if c["motion"] == "C2":
            f["g"] = np.log1p(tau - st["rb_nb"][reps, slot, players])
        if c["rule"].trend:
            row1 = np.maximum(np.searchsorted(G, tau - 1 - 1e-6, side="right") - 1, 0)
            A1 = np.where(row1 == 0, 0.0, st["rb_A"][reps, row1 % L, players])  # row 0: the all-zero round start
            f["trend"] = f["A"] - A1
        h = c["rule"].h(f, None if c["offset"] is None else c["offset"][st["id"][reps]])
        date = c["date"]
        lo = np.maximum(np.searchsorted(date, tau - 1e-6, side="left"), np.searchsorted(date, first[new] + 1e-6, side="right"))
        n = np.maximum(np.searchsorted(date, tau + BIN - 1e-6, side="left") - lo, 1)
        risk = ~st["rb_open"][reps, slot, players]
        if c["motion"] in ("C1", "C2"):                         # C1, C2: at risk only when sitting at the wall
            risk &= st["rb_abs"][reps, slot, players] >= 1 - 1e-9
        st["b_m"][new], st["b_q"][new], st["b_risk"][new], st["b_fired"][new] = m[new], 1 - (1 - h) ** (1 / n), risk, False
        if c["trace"] is not None:
            c["trace"].append(pd.DataFrame({"rep": st["id"][reps], "player": players, "tau": tau, "n": n, "h": h,
                                            "risk": risk, "A": f["A"], "abs_s": f["abs_s"], "at_end": at_end.astype(int),
                                            "time_at_end": np.where(at_end, tau - entered, 0.0),
                                            "time_since_rev": tau - anchor, **({"trend": f["trend"]} if "trend" in f else {}),
                                            **({"g": f["g"]} if "g" in f else {})}))
    return np.where(on & st["b_risk"] & ~st["b_fired"], st["b_q"], 0.0)


def features(st, j, c):
    """Features of every player at the start of tick j, from the positions at t = j dt (per-tick rules: APEM)."""
    t = j * c["dt"]
    s = st["n"] / c["N"]
    at_end = st["band"] != 0
    return {"A": np.abs(s @ c["A_log"].T) / c["deg"], "abs_s": np.abs(s), "at_end": at_end.astype(float),
            "e": np.where(at_end, np.log1p(t - st["entered"]), 0.0), "r": np.log1p(t - st["anchor"]),
            "lab": float(c["lab"])}


def tick(st, j, c, rng):
    """Tick j of every replicate in st (changed in place). Returns (mode during the tick, step) per player."""
    n, phi, N = st["n"], st["phi"], c["N"]
    mode = st["mode"]
    t1 = (j + 1) * c["dt"]

    # (a) reversal hazard: off before the first move, while reversing and before the start
    can = (mode != REVERSING) & (mode != NOT_STARTED)
    if c["per_bin"]:
        h = np.where(can, bin_hazard(st, j, c), 0.0)
    else:
        h = np.where(can & (st["dir"] != 0), c["rule"].prob(features(st, j, c), c["dt"]), 0.0)

    # (b) one uniform per player
    u = rng.random(n.shape)
    p = c["p"]
    rev = u < h
    begin = (mode == NOT_STARTED) & (u < p["start"])
    halt = c["halt"]
    q = p["resume"] if halt is None else np.where(st["fast"], halt["q_f"], halt["q_s"])    # L1b: by halt type
    p_stop, b1 = p["stop"], c["b1"]
    if b1 is not None:                                          # B1: by state, crossing style and halt state
        p_stop = np.where(st["c_on"], np.where(st["smooth"], 0.0, b1["q_sg"]), b1["q_E"])
        if "resume_C" in b1:
            q = np.where(st["halt_C"], b1["resume_C"], b1["resume_E"])
    resume = (mode == HALTED) & ~rev & (u < h + q)
    stop = (mode == MOVING) & ~rev & (u < h + p_stop)
    phi[begin] = np.where(rng.random(int(begin.sum())) < 0.5, 1, -1)     # coin flip
    if halt is not None:                                        # L1b: a new halt draws its type
        st["fast"][stop] = rng.random(int(stop.sum())) < halt["p"] if 0 < halt["p"] < 1 else halt["p"] == 1
    if b1 is not None:
        st["halt_C"][stop] = st["c_on"][stop]
    mode[begin | resume] = MOVING
    mode[stop] = HALTED
    phi[rev] = -phi[rev]                                        # F3: a reversal flips phi at once
    opens = (to_k(n + phi, N) - to_k(n, N)) * st["dir"] < 0    # ... and its first step is a step of k against dir
    mode[rev] = np.where(opens[rev] & c["script"], REVERSING, MOVING)
    if c["per_bin"]:
        st["b_fired"] |= rev
        if c["trace"] is not None and rev.any():
            reps, players = np.nonzero(rev)
            c["trace"].append(pd.DataFrame({"rep": st["id"][reps], "player": players, "fire": c["date"][j + 1]}))
    during = mode.copy()

    # (c) move
    step = np.where((mode == MOVING) | (mode == REVERSING), phi, 0)
    k_old = to_k(n, N)
    n += step
    k = to_k(n, N)

    # (d) detector at t + dt: protected while the reversal is open (F3); the estimator's rows (F2)
    band_old = st["band"].copy()
    opened, done = detect(st, "", k - k_old, k, t1)
    if b1 is not None:                                          # B1: enter C (draw the style) / leave C
        enter = (band_old != 0) & (st["band"] == 0)
        st["c_on"][enter], st["c_start"][enter] = True, t1
        st["smooth"][enter] = draw_style(st["p_smooth"][enter], rng)
        st["c_on"][(st["band"] != 0) | (done & (st["open_t"] > st["c_start"] + 1e-9))] = False
    if c["script"]:
        mode[(mode == MOVING) & opened] = REVERSING
        mode[(mode == REVERSING) & done] = MOVING
    mode[((mode == MOVING) | (mode == REVERSING)) & (np.abs(n) == N)] = AT_WALL
    record_rows(st, j + 1, c)
    return during, step


def trip_steps(i, L, n):
    """C1: steps a trip of n steps over L ticks has taken by the end of its tick i (0 before tick 0): the first step
    in tick 0, the last in tick L - 1, at most one per tick (L >= n)."""
    return np.where(i >= 0, i * (n - 1) // np.maximum(L - 1, 1) + 1, 0)


def tick_c1(st, j, c, rng):
    """C1: tick j of every replicate in st (changed in place); see the module doc. Returns (mode during the tick,
    step) per player. Draws, in order: one uniform per player, the coin flips, the first-trip durations, the
    crossing durations."""
    n, phi, N, mode = st["n"], st["phi"], c["N"], st["mode"]
    sit = mode == AT_WALL
    h = np.where(sit, bin_hazard(st, j, c), 0.0)
    u = rng.random(n.shape)
    begin = (mode == NOT_STARTED) & (u < c["p"]["start"])
    switch = sit & (u < h)
    phi[begin] = np.where(rng.random(int(begin.sum())) < 0.5, 1, -1)     # coin flip
    phi[switch] = -phi[switch]
    st["b_fired"] |= switch
    if c["trace"] is not None and switch.any():
        reps, players = np.nonzero(switch)
        c["trace"].append(pd.DataFrame({"rep": st["id"][reps], "player": players, "fire": c["date"][j + 1]}))
    for go, steps, pool in [(begin, N, c["c1"]["first"]), (switch, 2 * N, c["c1"]["cross"])]:
        D = pool[rng.integers(0, len(pool), int(go.sum()))]
        st["c1_L"][go] = np.maximum(steps, np.round(D / c["dt"]).astype(int) + 1)
        st["c1_n"][go], st["c1_i"][go] = steps, 0
    i, L, m = st["c1_i"], st["c1_L"], st["c1_n"]
    trip = i >= 0
    moves = trip & (trip_steps(i, L, m) > trip_steps(i - 1, L, m))
    mode[trip] = np.where(moves[trip], MOVING, HALTED)
    during = mode.copy()
    step = np.where(moves, phi, 0)
    n += step
    i[trip] += 1
    arrived = trip & (np.abs(n) == N)
    mode[arrived], i[arrived] = AT_WALL, -1
    record_rows(st, j + 1, c)
    return during, step


def look_dA(A_now, A0):
    """C2: dA = A at the start of the decision tick - A at the start of the step's tick, and its sign (tol 1e-9)."""
    dA = A_now - A0
    return dA, np.where(dA > 1e-9, 1, np.where(dA < -1e-9, -1, 0))


def tick_c2(st, j, c, rng):
    """C2: tick j of every replicate in st (changed in place); see the module doc. Returns (mode during the tick,
    step) per player. Draws, in order: one uniform per player (start / try), the start coin flips, the first-trip
    durations, the keep uniforms (players at a decision point), the paces of new kept movements, the walk uniforms,
    the walk directions."""
    n, phi, N, mode, p2 = st["n"], st["phi"], c["N"], st["mode"], c["c2"]
    A_now = np.abs((n / N) @ c["A_log"].T) / c["deg"]           # positions at the start of tick j
    sit = mode == AT_WALL
    h = np.where(sit, bin_hazard(st, j, c), 0.0)
    u = rng.random(n.shape)
    begin = (mode == NOT_STARTED) & (u < c["p"]["start"])
    tries = sit & (u < h)
    phi[begin] = np.where(rng.random(int(begin.sum())) < 0.5, 1, -1)     # coin flip
    st["b_fired"] |= tries
    if c["trace"] is not None and tries.any():
        reps, players = np.nonzero(tries)
        c["trace"].append(pd.DataFrame({"rep": st["id"][reps], "player": players, "fire": c["date"][j + 1]}))
    D = c["c1"]["first"][rng.integers(0, len(c["c1"]["first"]), int(begin.sum()))]
    st["c1_L"][begin] = np.maximum(N, np.round(D / c["dt"]).astype(int) + 1)
    st["c1_n"][begin], st["c1_i"][begin] = N, 0

    # the look at a decision point: keep (a straight run to the wall at pace f) or walk
    decide = st["c2_wait"] == j
    dA, sign = look_dA(A_now, st["c2_A0"])
    kp = np.where(sign > 0, p2["keep_pos"], np.where(sign < 0, p2["keep_neg"], p2["keep_zero"]))
    keep = np.zeros_like(decide)
    keep[decide] = rng.random(int(decide.sum())) < kp[decide]
    if c["trace"] is not None and decide.any():
        reps, players = np.nonzero(decide)
        c["trace"].append(pd.DataFrame({"rep": st["id"][reps], "player": players, "look": j, "dA": dA[decide],
                                        "kept": keep[decide]}))
    f = p2["pace"][rng.integers(0, len(p2["pace"]), int(keep.sum()))]
    phi[keep] = st["c2_dir"][keep]
    left = N - phi[keep] * n[keep]                              # steps to the wall ahead
    st["c1_n"][keep], st["c1_i"][keep] = left, 0
    st["c1_L"][keep] = np.maximum(left, np.round((left - 1) / f).astype(int) + 1)
    st["c2_walk"][decide & ~keep] = True
    st["c2_wait"][decide] = -1

    # walk steps
    walk = st["c2_walk"]
    go = np.zeros_like(walk)
    go[walk] = rng.random(int(walk.sum())) < p2["q_w"]
    wdir = np.zeros_like(n)
    wdir[go] = np.where(rng.random(int(go.sum())) < 0.5, 1, -1)

    # move: trips and kept runs (trip_steps), tries (one step off the wall), walk steps
    i, L, m = st["c1_i"], st["c1_L"], st["c1_n"]
    trip = i >= 0
    moves = trip & (trip_steps(i, L, m) > trip_steps(i - 1, L, m))
    step = np.where(moves, phi, 0) + np.where(tries, -np.sign(n), 0) + wdir
    looked = tries | go                                         # these steps get a look
    st["c2_A0"][looked], st["c2_dir"][looked] = A_now[looked], step[looked]
    st["c2_wait"][looked] = j + max(1, int(round(p2["d_ar"] / c["dt"])))
    st["c2_walk"][go] = False
    mode[~sit & (mode != NOT_STARTED) | tries | begin] = HALTED
    mode[step != 0] = MOVING
    during = mode.copy()
    n += step
    i[trip] += 1
    arrived = (step != 0) & (np.abs(n) == N)
    mode[arrived], i[arrived], st["c2_wait"][arrived], st["c2_walk"][arrived] = AT_WALL, -1, -1, False
    record_rows(st, j + 1, c)
    return during, step


def simulate(A, T, iface, rule, motion="L1", script=True, R=100, seed=None, speed=None, trace=None, halts=None,
             offset=None, b1=None, trait=None, c1=None, c2=None):
    """R replicates of one round on adjacency A (7 x 7, seat order, no self-loops) with time limit T (s).

    iface: "lab_v09", "old_cl" or "cc" (grid, speed, lab flag, motion rates); speed overrides the interface's.
    rule: Logistic (per estimator bin, F2) or an object with prob(features, dt) (per tick); motion: "L1", "L1b"
    or "none"; script: True = scripted reversals, False = a reversal flips phi at once. preset(name) gives the three.
    halts: L1b {p, q_f, q_s} (default: p25_halts.csv). offset: L3, one value per replicate added to logit h
    (Logistic rules), or None. B1 (motion "B1"): b1 {pi, rho, q_sg, q_E[, resume_C, resume_E]}, or a parameter
    file (default: pB1_params.csv), and trait, the U of every seat, shape (R, 7) (required). C1 (motion "C1"):
    c1 {"first": durations, "cross": durations} (s), or a duration file (default: pC_durations.parquet). C2 (motion
    "C2"): c2 {d_ar, q_w, keep_pos, keep_zero, keep_neg, pace}, or a parameter file (default: pC2_params.csv).
    Returns one dict per replicate: events (DataFrame time, seat, s; the round start first), end_time (s),
    ticks, success (consensus before T), mode_ticks ((7, 5) ticks per seat in each of MODES), dt, u (the offset).
    trace: a list to collect, for checks, one DataFrame per tick with bins opened (rep, player, tau, n, h, risk and
    the features, time features raw) and one per tick with reversals fired (rep, player, fire = the step's date).
    """
    rng = np.random.default_rng(seed)
    c = constants(A, iface, rule, motion, script, speed, halts, b1, c1, c2)
    step_tick = {"C1": tick_c1, "C2": tick_c2}.get(motion, tick)
    c["trace"] = trace
    if motion == "B1":
        if trait is None or np.shape(trait) != (R, 7):
            raise ValueError("B1: trait must hold one U per seat, shape (R, 7)")
        c["trait"] = np.asarray(trait, dtype=float)
    if offset is not None:
        c["offset"] = np.asarray(offset, dtype=float)
        if not c["per_bin"] or len(c["offset"]) != R:
            raise ValueError("offset: one value per replicate, Logistic rules only")
    st = start(R, c, rng)
    ids = np.arange(R)                       # replicate of each row of st
    counts = np.zeros((R, 7, len(MODES)), dtype=int)
    ends = np.zeros(R, dtype=int)
    success = np.zeros(R, dtype=bool)
    ev = []                                  # (replicate, tick, seat, n) of each position change
    n_ticks = int(np.floor(T / c["dt"] + 1e-9))
    for j in range(n_ticks):
        during, step = step_tick(st, j, c, rng)
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
                    "success": bool(success[r]), "mode_ticks": counts[r], "dt": c["dt"],
                    "u": 0.0 if c["offset"] is None else float(c["offset"][r])})
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

    model: a preset name (L1, L1_M3, L0, final, B1, C1, APEM; APEM takes a, s_th, mu). T = the session's time limit.
    final: each replicate is its own group, so it draws its own u ~ N(0, sigma_g^2); B1: its own seats' U.
    dataset: default CL for lab_v09 and old_cl, CC for cc. logger: default "lab" for lab interfaces, "cloud" for cc.
    Group (file) of replicate r: sim_<tag>_<topology>_<model>_<seed>_r<r>.txt, one round each (wave 1).
    """
    f = IFACES[iface]
    T = config.TIME_LIMIT[f["session"]]
    ss = np.random.SeedSequence(seed)
    sim_seed, log_seed = ss.spawn(2)
    m = preset(model, **apem)
    sigma = m.pop("sigma_g", 0.0)
    offset = np.random.default_rng(ss.spawn(1)[0]).normal(0, sigma, R) if sigma > 0 else None
    if m.pop("trait", False):
        m["trait"] = np.random.default_rng(ss.spawn(1)[0]).random((R, 7))
    out = simulate(A, T, iface, R=R, seed=sim_seed, offset=offset, **m)
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

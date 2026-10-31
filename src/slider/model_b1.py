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
from scipy import optimize
from scipy.special import betaln, expit, logit, logsumexp

from slider import halts as halts_mod
from slider import config, interface, model_b, motion, stop_states
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


def estimates(runs, style="attempts"):
    """Point estimates of the B1 parameters per interface, with the code of steps 0-1:
    pi, rho, q_sg: style "attempts" (build fix 1): fit_attempts on all crossing attempts; style "completed" (v1):
    beta-binomial over players with >= 3 completed crossings and stops / steps in non-smooth completed crossings;
    q_E (stops / counted steps in state E), resume_C, resume_E, resume_all (resumes / halted
    ticks). Returns (DataFrame interface x parameter, the tables used: steps, crossings, halts)."""
    steps = step_state(stop_states.step_table(runs))
    cross = model_b.crossing_table(runs, steps)
    halts = halt_table(runs)
    players = model_b.player_shares(cross)
    rows = []
    for name in INTERFACES:
        p, c = players[players["interface"] == name], cross[cross["interface"] == name]
        st, h = steps[steps["interface"] == name], halts[halts["interface"] == name]
        sg, e = c[c["smooth"] == 0], st[st["state"] == "E"]
        if style == "completed":
            pi, rho = model_b.betabinom_fit(p["smooth"].to_numpy(), p["n"].to_numpy())
            q_sg = sg["stops"].sum() / sg["steps"].sum()
        else:
            pi, rho, q_sg = fit_attempts(attempt_table(st))
        hc, he = h[h["state"] == "C"], h[h["state"] == "E"]
        rows.append({"interface": name, "pi": pi, "rho": rho, "q_sg": q_sg,
                     "q_E": e["stop"].sum() / len(e), "resume_C": hc["event"].sum() / hc["n"].sum(),
                     "resume_E": he["event"].sum() / he["n"].sum(), "resume_all": h["event"].sum() / h["n"].sum()})
    return pd.DataFrame(rows).set_index("interface"), {"steps": steps, "cross": cross, "halts": halts}


# ----------------------------------------------------------------- build fix 1 (Oct 30): style fit per attempt

def attempt_table(steps):
    """One row per crossing attempt: the counted steps in state C of one outbound segment (completed, aborted or
    censored), with interface, dataset, file, wave, player, segment, outcome, n (C steps) and k (stops)."""
    c = step_state(steps)
    c = c[c["state"] == "C"]
    return c.groupby(["interface", "dataset", "file", "wave", "player", "segment"]).agg(
        outcome=("outcome", "first"), n=("stop", "size"), k=("stop", "sum")).reset_index()


def player_data(att):
    """Per player (file, seat), as arrays: file, k_sum and nk_sum (stops and non-stops over the attempts with a
    stop), m1 (attempts with a stop), N0 (the n of each attempt with no stop, one row per player, padded with 0)."""
    rows, n0 = [], []
    for (file, _), a in att.groupby(["file", "player"]):
        stop = a[a["k"] > 0]
        rows.append((file, stop["k"].sum(), (stop["n"] - stop["k"]).sum(), len(stop)))
        n0.append(a.loc[a["k"] == 0, "n"].to_numpy(float))
    N0 = np.zeros((len(n0), max([len(v) for v in n0] + [1])))
    has = np.zeros_like(N0, dtype=bool)
    for i, v in enumerate(n0):
        N0[i, :len(v)], has[i, :len(v)] = v, True
    file, k_sum, nk_sum, m1 = map(np.array, zip(*rows))
    return {"file": file, "k_sum": k_sum.astype(float), "nk_sum": nk_sum.astype(float), "m1": m1.astype(float),
            "N0": N0, "has": has}


def take(players, idx):
    """The players idx (with repeats) of player_data output."""
    return {k: v[idx] for k, v in players.items()}


def mixture_loglik(pi, rho, q, players):
    """Log-likelihood of the per-player mixture. Each attempt is smooth (no stop: likelihood 1 if k = 0, else 0) with
    the player's p, else stop-and-go (q^k (1 - q)^(n - k)); p ~ Beta(a, b), a = pi (1 - rho) / rho,
    b = (1 - pi)(1 - rho) / rho, integrated exactly: the product over a player's attempts with no stop,
    prod (p + (1 - p)(1 - q)^n), is a polynomial in p, written in the basis p^s (1 - p)^(M - s) (a padded attempt
    has (1 - q)^n = 1: the factor 1); attempts with stops add (1 - p)^m1."""
    if not (0 < pi < 1 and 0 < rho < 1 and 0 < q < 1):
        return -np.inf
    a, b = pi * (1 - rho) / rho, (1 - pi) * (1 - rho) / rho
    V = np.where(players["has"], (1 - q) ** players["N0"], 1.0)
    P, M = V.shape
    c = np.zeros((P, M + 1))
    c[:, 0] = 1.0
    for j in range(M):
        c[:, 1:] = c[:, 1:] * V[:, j:j + 1] + c[:, :-1]
        c[:, 0] *= V[:, j]
    s = np.arange(M + 1)
    with np.errstate(divide="ignore"):
        terms = np.log(c) + betaln(a + s, b + players["m1"][:, None] + M - s) - betaln(a, b)
    total = np.sum(players["k_sum"] * np.log(q) + players["nk_sum"] * np.log1p(-q) + logsumexp(terms, axis=1))
    return total if np.isfinite(total) else -np.inf


def fit_players(players, start=(0.0, -1.5, 0.0)):
    """Maximum likelihood (pi, rho, q_sg) of the per-player mixture (logit scales, Nelder-Mead)."""
    nll = lambda x: -mixture_loglik(*expit(x), players)
    r = optimize.minimize(nll, start, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-7, "maxiter": 4000})
    return expit(r.x)


def fit_attempts(att, start=(0.0, -1.5, 0.0)):
    """Maximum likelihood (pi, rho, q_sg) of the per-player mixture on the attempts att."""
    return fit_players(player_data(att), start)


def boot_attempts(att, n_boot=1000, seed=config.SEED):
    """Group bootstrap of fit_attempts: n_boot resamples of the files within dataset (the draws of
    motion.boot_ratios), each refit from the full-data estimate. Returns (estimate, n_boot x 3 refits)."""
    players = player_data(att)
    full = fit_players(players)
    files = att.groupby("file")["dataset"].first()
    by_file = {f: np.flatnonzero(players["file"] == f) for f in files.index}
    strata = [files.index[files.to_numpy() == d].to_numpy() for d in interface.DATASETS if (files == d).any()]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        draw = np.concatenate([st[rng.integers(0, len(st), len(st))] for st in strata])
        out.append(fit_players(take(players, np.concatenate([by_file[f] for f in draw])), logit(full)))
    return full, np.array(out)


def simulate_attempts(players_n, pi, rho, q, rng):
    """Attempts drawn from the mixture for the given players (a list of arrays: each player's attempt n):
    p ~ Beta(pi, rho) per player; each attempt smooth with p (k = 0), else k ~ Binomial(n, q)."""
    a, b = pi * (1 - rho) / rho, (1 - pi) * (1 - rho) / rho
    rows = []
    for i, n in enumerate(players_n):
        smooth = rng.random(len(n)) < rng.beta(a, b)
        k = np.where(smooth, 0, rng.binomial(n, q))
        rows.append(pd.DataFrame({"file": f"p{i}", "player": 0, "n": n, "k": k}))
    return pd.concat(rows, ignore_index=True)

"""Path A, P25, L1b (pre-registered): mid-round halts as a two-type mixture of geometric durations in ticks.

Halt spells: motion.motion_table (P17, the F1b version), the halts in halted_s (mid-slider, after the first move,
not mid-event, not at the wall). Length in ticks n = max(1, round(duration / dt)), dt = the round's tick
(CC halts are not whole ticks: the cloud logger's row times jitter). Event: exit = "resume"; censored: every
other exit (reversal, probe, unresolved, round_end), as in P17's lam_resume.

Model (q = resume probability per tick, P(halt > n ticks) = S(n)):
    S(n) = p (1 - q_f)^n + (1 - p) (1 - q_s)^n,          q_f >= q_s (fast = the type with the larger q)
    a resume after n ticks:  f(n) = S(n - 1) - S(n);  censored after n ticks: S(n)
With p = 1 it is one geometric, and the maximum likelihood q = resumes / all halt ticks (P17's lam_resume dt).
lam = q / dt, with dt = the mean tick of the interface's halts (descriptive; the simulator uses q per tick).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from slider import config
from slider import interface
from slider import motion
from slider import slider_logs as sl

PARAMS = ["p", "q_f", "q_s"]


def load_halts():
    """The G-train halt spells (motion.motion_table on load_runs("behavior"), as P17) with n and event."""
    runs, table = sl.load_runs("behavior")
    assert set(table["role"]) == {"train"}
    return add_ticks(motion.motion_table(runs)[1])


def add_ticks(halts):
    """halts with n (ticks) and event (1 = resume)."""
    return halts.assign(n=np.maximum(1, np.round(halts["duration"] / halts["dt"])).astype(int),
                        event=(halts["exit"] == "resume").astype(int))


def counts(halts):
    """(n, event, count): the halts grouped by length and outcome (the likelihood only needs these)."""
    c = halts.groupby(["n", "event"]).size().reset_index(name="count")
    return c["n"].to_numpy(float), c["event"].to_numpy(float), c["count"].to_numpy(float)


def survival(n, p, q_f, q_s):
    """S(n) = P(halt > n ticks)."""
    n = np.asarray(n, dtype=float)
    return p * (1 - q_f) ** n + (1 - p) * (1 - q_s) ** n


def loglik(n, event, w, p, q_f, q_s):
    """Log-likelihood of halts (n, event) with weights w (counts)."""
    f = p * q_f * (1 - q_f) ** (n - 1) + (1 - p) * q_s * (1 - q_s) ** (n - 1)
    return np.sum(w * np.where(event > 0, np.log(f), np.log(survival(n, p, q_f, q_s))))


def fit(n, event, w):
    """Maximum likelihood (p, q_f, q_s) on logit scales, from 3 starts around the one-type q; the best one, with
    q_f >= q_s. Returns {p, q_f, q_s, loglik, converged}."""
    q0 = np.sum(w * event) / np.sum(w * n)

    def nll(x):
        return -loglik(n, event, w, *expit(x))

    best = None
    for p0, f0, s0 in [(0.5, min(3 * q0, 0.9), q0 / 3), (0.8, min(1.5 * q0, 0.9), q0 / 5), (0.3, min(5 * q0, 0.95), q0 / 2)]:
        r = minimize(nll, logit([p0, f0, s0]), method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000, "maxfev": 20000})
        if best is None or r.fun < best.fun:
            best = r
    p, q_f, q_s = expit(best.x)
    if q_f < q_s:                                       # label the fast type
        p, q_f, q_s = 1 - p, q_s, q_f
    return {"p": p, "q_f": q_f, "q_s": q_s, "loglik": -best.fun, "converged": bool(best.success)}


def boot(halts, n_boot=1000, seed=config.SEED):
    """n_boot fits on resamples of the groups (files), drawn with replacement within each dataset, as many as in
    the data (the draws of motion.boot_ratios). Returns an (n_boot, 3) array of p, q_f, q_s."""
    per_file = {f: counts(d) for f, d in halts.groupby("file")}
    files = np.array(sorted(per_file))
    dataset = halts.groupby("file")["dataset"].first().loc[files].to_numpy()
    strata = [np.flatnonzero(dataset == d) for d in interface.DATASETS if (dataset == d).any()]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([st[rng.integers(0, len(st), len(st))] for st in strata])
        parts = [per_file[files[i]] for i in idx]
        c = pd.DataFrame({"n": np.concatenate([a for a, _, _ in parts]), "event": np.concatenate([b for _, b, _ in parts]),
                          "w": np.concatenate([w for _, _, w in parts])}).groupby(["n", "event"], as_index=False)["w"].sum()
        r = fit(c["n"].to_numpy(), c["event"].to_numpy(), c["w"].to_numpy())
        out.append([r[k] for k in PARAMS])
    return np.array(out)


def fit_table(halts, n_boot=1000, seed=config.SEED):
    """Per interface (motion.INTERFACES): p, q_f, q_s and lam_f, lam_s = q / dt (dt = the mean tick of the halts),
    with 95% group-bootstrap CIs (boot). Columns: interface, parameter, estimate, ci_low, ci_high, halts,
    resumes, groups, dt, loglik, loglik_one_type, converged."""
    rows = []
    for name in motion.INTERFACES:
        d = halts[halts["interface"] == name]
        n, event, w = counts(d)
        r = fit(n, event, w)
        q1 = np.sum(w * event) / np.sum(w * n)
        b = boot(d, n_boot, seed)
        dt = d["dt"].mean()
        lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
        est = [r["p"], r["q_f"], r["q_s"]]
        info = {"interface": name, "halts": len(d), "resumes": int(d["event"].sum()), "groups": d["file"].nunique(),
                "dt": dt, "loglik": r["loglik"], "loglik_one_type": loglik(n, event, w, 1.0, q1, q1),
                "converged": r["converged"]}
        for j, param in enumerate(PARAMS):
            rows.append({**info, "parameter": param, "estimate": est[j], "ci_low": lo[j], "ci_high": hi[j]})
        for j, param in [(1, "lam_f"), (2, "lam_s")]:
            rows.append({**info, "parameter": param, "estimate": est[j] / dt, "ci_low": lo[j] / dt, "ci_high": hi[j] / dt})
    cols = ["interface", "parameter", "estimate", "ci_low", "ci_high", "halts", "resumes", "groups", "dt", "loglik",
            "loglik_one_type", "converged"]
    return pd.DataFrame(rows)[cols]


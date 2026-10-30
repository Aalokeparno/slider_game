"""Path A, P25, L3 (pre-registered): a group effect on the reversal hazard.

    logit h = eta + u_g,   eta = the frozen M2' linear predictor (p11_coefs.csv, an offset),   u_g ~ N(0, sigma_g^2)
one u per group (group_id = log file). Only sigma_g is fit; the M2' weights do not change.

Marginal log-likelihood: sum over groups of log integral exp(l_g(u)) phi(u; 0, sigma^2) du, with
l_g(u) = sum over the group's bins of y (eta + u) - log(1 + exp(eta + u)). Each integral by adaptive Gauss-Hermite
(K nodes centred at the mode of the integrand and scaled by its curvature, as lme4's nAGQ): the groups have
thousands of bins, so the integrand is much narrower than sigma and plain Gauss-Hermite nodes would miss it.
At sigma = 0 it is sum l_g(0). 95% CI: profile likelihood (the only parameter), 2 (l_max - l(sigma)) <= 3.841;
the lower bound is 0 if l(0) is inside.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit

from slider import config
from slider import fit

M2P = ["A", "abs_s", "at_end", "time_at_end_log1p", "time_since_rev_log1p"]
CHI2_95 = 3.841458820694124
NODES = 30


def g_train():
    """The G-train hazard table (train topologies, split train), as P11 fit M2' on it."""
    ht = pd.read_parquet(config.HAZARD_TABLE)
    return ht[ht["topology"].isin(config.TRAIN_TOPOLOGIES) & (ht["split"] == "train")]


def offset(ht):
    """eta: the frozen M2' linear predictor of every bin (p11_coefs.csv)."""
    c = pd.read_csv(config.P11_COEFS)
    beta = c[c["model"] == "M2'"].set_index("term")["beta"]
    X = fit.design(ht, M2P, False)
    return X[beta.index].to_numpy(float) @ beta.to_numpy(float)


def prepare(ht, group="group_id"):
    """(eta, y, group index 0..G-1) of a hazard table."""
    codes = pd.factorize(ht[group], sort=True)[0]
    return offset(ht), ht["y_rev"].to_numpy(float), codes


def group_ll(eta, y, g, u):
    """l_g(u_g) for every group: u is one value per group."""
    z = eta + u[g]
    return np.bincount(g, weights=y * z - np.logaddexp(0, z), minlength=len(u))


def marginal_ll(sigma, eta, y, g, nodes=NODES):
    """Marginal log-likelihood at sigma (adaptive Gauss-Hermite with `nodes` nodes)."""
    G = g.max() + 1
    if sigma <= 0:
        return group_ll(eta, y, g, np.zeros(G)).sum()
    u = np.zeros(G)
    for _ in range(50):                                            # Newton for the mode of l_g(u) - u^2 / 2 sigma^2
        h = expit(eta + u[g])
        grad = np.bincount(g, weights=y - h, minlength=G) - u / sigma ** 2
        curv = np.bincount(g, weights=h * (1 - h), minlength=G) + 1 / sigma ** 2
        step = grad / curv
        u += step
        if np.max(np.abs(step)) < 1e-10:
            break
    h = expit(eta + u[g])
    s = 1 / np.sqrt(np.bincount(g, weights=h * (1 - h), minlength=G) + 1 / sigma ** 2)
    x, w = np.polynomial.hermite.hermgauss(nodes)
    terms = []
    for xk, wk in zip(x, w):
        uk = u + np.sqrt(2) * s * xk
        log_phi = -uk ** 2 / (2 * sigma ** 2) - np.log(np.sqrt(2 * np.pi) * sigma)
        terms.append(np.log(wk) + xk ** 2 + np.log(np.sqrt(2) * s) + group_ll(eta, y, g, uk) + log_phi)
    return np.logaddexp.reduce(np.array(terms), axis=0).sum()


def fit_sigma(ht, group="group_id", nodes=NODES, upper=3.0):
    """sigma_g by maximum marginal likelihood, with the profile-likelihood 95% CI.
    Returns {sigma_g, ci_low, ci_high, loglik, loglik_sigma0, lr_stat (2 (l_max - l(0))), groups, bins, reversals}."""
    eta, y, g = prepare(ht, group)

    def nll(sig):
        return -marginal_ll(sig, eta, y, g, nodes)

    r = minimize_scalar(nll, bounds=(1e-4, upper), method="bounded", options={"xatol": 1e-7})
    sig, l_max = r.x, -r.fun
    l0 = marginal_ll(0.0, eta, y, g, nodes)
    if l0 >= l_max:                                                # the maximum is at 0
        sig, l_max = 0.0, l0

    def gap(s):
        return 2 * (l_max - marginal_ll(s, eta, y, g, nodes)) - CHI2_95

    lo = 0.0 if gap(0.0) <= 0 else brentq(gap, 0.0, sig, xtol=1e-7)
    hi = brentq(gap, max(sig, 1e-4), upper, xtol=1e-7)
    return {"sigma_g": sig, "ci_low": lo, "ci_high": hi, "loglik": l_max, "loglik_sigma0": l0,
            "lr_stat": 2 * (l_max - l0), "groups": g.max() + 1, "bins": len(y), "reversals": int(y.sum()),
            "nodes": nodes}


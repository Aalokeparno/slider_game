"""Path A, P10 - P13: logistic hazard models, pre-registered in the Path A Workbook Ch. A8.2 / A8.4 / A8.6.

Unpenalized logistic regression of y_rev (or y_probe) on the hazard table (hazard.build_hazard_table), raw features.
lab = 1 if dataset in {EL, CL}, 0 if CC.
  R0: const + lab + feats                    (shared weights)
  R1: R0 + lab x each feature                (cloud weights beta_j, lab weights beta_j + gamma_j)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from slider import config

OUTCOME = "y_rev"
FEATURES = ["A", "trend", "abs_s", "at_end", "time_at_end_log1p", "time_since_rev_log1p"]   # primary run (w = 1 s)


def design(df, feats, lab_interact):
    """Design matrix: const, lab, feats, and lab:<feat> = lab x feat if lab_interact."""
    lab = df["dataset"].isin(["EL", "CL"]).astype(float)
    X = pd.DataFrame({"const": 1.0, "lab": lab}, index=df.index)
    for f in feats:
        X[f] = df[f].astype(float)
    if lab_interact:
        for f in feats:
            X["lab:" + f] = lab * df[f]
    return X


def fit(df, feats, lab_interact, outcome=OUTCOME):
    """Fit R0 (lab_interact=False) or R1 (True) of outcome (y_rev or y_probe) with statsmodels Logit (Newton).
    Returns the fitted model."""
    return sm.Logit(df[outcome].astype(float), design(df, feats, lab_interact)).fit(disp=0)


def bin_ll(model, df):
    """Log-likelihood of each bin under a fitted model: y ln h + (1 - y) ln(1 - h), y = the model's outcome."""
    names = model.model.exog_names
    feats = [n for n in names if n not in ("const", "lab") and not n.startswith("lab:")]
    X = design(df, feats, any(n.startswith("lab:") for n in names))[names]
    h = np.asarray(model.predict(X))
    y = df[model.model.endog_names].to_numpy(float)
    return y * np.log(h) + (1 - y) * np.log1p(-h)


def boot_delta(ll_big, ll_small, groups, n=10000, seed=config.SEED):
    """delta = sum(ll_big - ll_small) / n_bins and its 95% percentile CI, resampling whole groups with
    replacement (same draws as stats.group_bootstrap_mean). The fitted betas stay fixed: no refitting.
    Returns (delta, (low, high))."""
    d = pd.DataFrame({"diff": np.asarray(ll_big) - np.asarray(ll_small), "group": np.asarray(groups)})
    g = d.groupby("group")["diff"].agg(["sum", "size"])          # sorted group order
    sums, counts = np.asarray(g["sum"], dtype=float), np.asarray(g["size"], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return sums.sum() / counts.sum(), (low, high)


def rcs(x, knots):
    """Natural (restricted) cubic spline of x with knots t_1 < ... < t_k (Harrell's form): the k - 2 nonlinear
    columns; with x itself they span the spline (cubic between knots, linear below t_1 and above t_k).
    Column j: (x - t_j)+^3 - (x - t_k-1)+^3 (t_k - t_j) / (t_k - t_k-1) + (x - t_k)+^3 (t_k-1 - t_j) / (t_k - t_k-1),
    divided by (t_k - t_1)^2."""
    x, t = np.asarray(x, dtype=float), np.asarray(knots, dtype=float)
    k = len(t)

    def cube(u):
        return np.maximum(u, 0) ** 3

    cols = [cube(x - t[j]) - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
            + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]) for j in range(k - 2)]
    return np.column_stack(cols) / (t[k - 1] - t[0]) ** 2

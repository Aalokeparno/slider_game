"""fit.py on synthetic data from a known shared-weight logistic model (R0 is the true model)."""
import numpy as np
import pandas as pd
import pytest

from slider.fit import FEATURES, bin_ll, boot_delta, fit

BETA = {"const": -3.0, "lab": 0.3, "A": -2.0, "trend": -1.5, "abs_s": 0.5, "at_end": -0.4,
        "time_at_end_log1p": -0.2, "time_since_rev_log1p": -0.3}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 200_000
    df = pd.DataFrame({"dataset": rng.choice(["EL", "CL", "CC"], n), "group_id": rng.integers(0, 40, n),
                       "A": rng.uniform(0, 1, n), "trend": rng.normal(0, 0.3, n), "abs_s": rng.uniform(0, 1, n),
                       "at_end": rng.integers(0, 2, n), "time_at_end_log1p": rng.uniform(0, 5, n),
                       "time_since_rev_log1p": rng.uniform(0, 5, n)})
    lab = df["dataset"].isin(["EL", "CL"])
    eta = BETA["const"] + BETA["lab"] * lab + sum(BETA[f] * df[f] for f in FEATURES)
    df["y_rev"] = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-eta))).astype(int)
    return df, lab


def test_r0_recovers_the_betas(data):
    df, _ = data
    m = fit(df, FEATURES, lab_interact=False)
    assert m.mle_retvals["converged"]
    assert list(m.params.index) == list(BETA)
    for name, beta in BETA.items():
        assert abs(m.params[name] - beta) < 4 * m.bse[name], name


def test_r1_has_14_params_and_higher_ll(data):
    df, _ = data
    r0, r1 = fit(df, FEATURES, False), fit(df, FEATURES, True)
    assert len(r1.params) == 14 and r1.llf >= r0.llf


def test_sum_h_equals_sum_y_overall_lab_and_cloud(data):
    df, lab = data
    for interact in (False, True):
        h = fit(df, FEATURES, interact).predict()
        y = df["y_rev"].to_numpy()
        for mask in (np.ones(len(df), bool), lab.to_numpy(), ~lab.to_numpy()):
            assert h[mask].sum() == pytest.approx(y[mask].sum(), abs=1e-4)


def test_bin_ll_sums_to_the_model_ll(data):
    df, _ = data
    m = fit(df, FEATURES, True)
    assert bin_ll(m, df).sum() == pytest.approx(m.llf, rel=1e-10)


def test_another_outcome_column(data):
    df, _ = data
    d = df.assign(y_probe=df["y_rev"].to_numpy()[::-1])        # any other 0/1 column
    m = fit(d, FEATURES, False, outcome="y_probe")
    assert m.model.endog_names == "y_probe" and m.model.endog.sum() == d["y_probe"].sum()
    assert bin_ll(m, d).sum() == pytest.approx(m.llf, rel=1e-10)


def test_boot_delta_of_x_and_x_is_zero(data):
    df, _ = data
    x = bin_ll(fit(df, FEATURES, False), df)
    delta, (low, high) = boot_delta(x, x, df["group_id"], n=1000)
    assert (delta, low, high) == (0, 0, 0)

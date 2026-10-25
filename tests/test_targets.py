"""targets.py and the bootstrap helpers it uses. Train / val groups only (no test runs)."""
import numpy as np
import pandas as pd

from slider import config
from slider import slider_logs as sl
from slider import targets
from slider.stats import group_bootstrap, group_bootstrap_mean, kaplan_meier, km_quantiles


def test_km_quantiles_match_kaplan_meier():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = rng.integers(1, 60)
        d = rng.integers(1, 15, size=n).astype(float)
        e = rng.random(n) < 0.7
        table, median = kaplan_meier(d, e)
        q = km_quantiles(d, e, (0.25, 0.5, 0.75))
        expected = [targets.km_quantile(table, p) for p in (0.25, 0.5, 0.75)]
        assert np.allclose(q, expected, equal_nan=True)
        assert np.isclose(q[1], median, equal_nan=True)


def test_group_bootstrap_matches_group_bootstrap_mean():
    rng = np.random.default_rng(2)
    v, g = rng.random(50), rng.integers(0, 8, size=50)
    per_group = [(v[g == k],) for k in np.unique(g)]
    assert np.allclose(group_bootstrap(per_group, np.mean), group_bootstrap_mean(v, g))


def test_group_bootstrap_unreached_quantile_is_inf():
    per_group = [(np.array([5.0]), np.array([0.0])), (np.array([3.0]), np.array([1.0]))]
    low, high = group_bootstrap(per_group, lambda d, e: km_quantiles(d, e > 0)[0], n_boot=200)
    assert high == np.inf


def test_targets_agree_with_h5_on_train():
    runs, table = sl.load_runs("outcome", roles={"train"})
    t = targets.train_targets(runs, table, n_boot=20)
    assert set(t["role"]) == {"train", "train+val"}            # only train groups were loaded
    h5_main = pd.read_csv(config.H5_TRAIN_CHECKS)
    h5_main = h5_main[(h5_main["check"] == "main") & (h5_main["part"] == "topology")].set_index("key")
    cells = t[(t["stat"] == "committed_pct_mean") & (t["role"] == "train")]
    for topology, d in cells.groupby("topology"):
        pooled = np.sum(d["value"] * d["n_runs"]) / d["n_runs"].sum()     # runs-weighted mean over datasets
        assert np.isclose(pooled, h5_main.loc[topology, "mean % committed"])

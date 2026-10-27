"""targets.py and the bootstrap helpers it uses. Train / val groups only (no test runs)."""
import numpy as np
import pandas as pd
import pytest

from slider import config
from slider import hazard
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


def test_reversal_rate_on_a_synthetic_run():
    # player 0: k = 0, 1, 2, 3, 1, 3, 1 -> 3 reversals (m = 2); seat 7 is a bot with the same moves (not counted)
    t = [0, 10, 20, 30, 40, 50, 120]
    s0 = [0, .1, .2, .3, .1, .3, .1]
    X = np.full((len(t), 7), 0.5)
    X[:, 0] = X[:, 6] = s0
    run = {"file": "f", "wave": 1, "dataset": "CL", "session": "lab", "end": "failed", "bot_players": {7},
           "A": np.ones((7, 7), dtype=int) - np.eye(7, dtype=int), "weights": np.ones(7), "t": np.array(t), "X": X}
    r = targets.reversal_rate_table([run]).iloc[0]
    # failed lab round: lasts until the 300 s limit -> 6 human players x 5 min = 30 player-min
    assert (r["topology"], r["dataset"], r["groups"], r["runs"], r["reversals"]) == ("Full", "CL", 1, 1, 3)
    assert r["player_min"] == 30
    assert r["rate"] == r["ci_lo"] == r["ci_hi"] == 0.1


def test_agreement_at_reversal_on_a_synthetic_run():
    # player 0: k = 0, 0, 1, 2, 0, 0 -> first move at t = 20, one reversal at row 4 (t = 40);
    # players 1-6 never move (s = 0.5). Full network, so A_i = |sum of s| / 7 for every player.
    t = [0, 10, 20, 30, 40, 60]
    X = np.full((len(t), 7), 0.5)
    X[:, 0] = [0, 0, .1, .2, 0, 0]
    run = {"file": "f", "wave": 1, "dataset": "CL", "session": "lab", "end": "success", "bot_players": set(),
           "A_log": np.ones((7, 7), dtype=int), "A": np.ones((7, 7), dtype=int) - np.eye(7, dtype=int),
           "weights": np.ones(7), "t": np.array(t, dtype=float), "X": X}
    # at risk from 20 s to the round end at 60 s: 40 / 0.25 = 160 samples, all player 0
    samples = hazard.at_risk_samples(run, b=0.25)
    assert len(samples) == 160 and set(samples["player"]) == {0}
    assert samples["time"].iloc[0] == 20 and samples["time"].iloc[-1] == pytest.approx(59.75)
    summary, h = targets.agreement_at_reversal_table([run])
    f_rev = summary[(summary["measure"] == "f_rev") & (summary["stat"] == "median")].iloc[0]
    assert f_rev["n"] == 1
    assert f_rev["value"] == pytest.approx(3.2 / 7)       # row 3 (s = 0.2), not row 4 (s = 0)
    assert h["reversals"].sum() == 1 and h["at_risk_min"].sum() == pytest.approx(160 * 0.25 / 60)

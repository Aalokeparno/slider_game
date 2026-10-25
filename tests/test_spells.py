"""spells (fraught.py) and kaplan_meier (stats.py). Real runs: train runs only."""
import numpy as np
import pytest

from slider import slider_logs as sl
from slider.fraught import fraught_mask, fraught_percent_hold, spells
from slider.stats import kaplan_meier

F, T = False, True
ROUND_A = dict(t=[0, 2, 3, 7, 8, 12], mask=[F, T, T, F, T, F], t_end=12, censored=False,
               labels=["a", "b", "c", "a", "c", "z"])


def as_tuples(sp):
    return [(r.start, r.end, r.duration, r.after) for r in sp.itertuples()]


def test_round_A_tau_0():
    sp = spells(tau=0, **ROUND_A)
    assert as_tuples(sp) == [(2, 7, 5, "relapse"), (8, 12, 4, "consensus")]
    assert np.isnan(sp["t_to_consensus"][0]) and sp["t_to_consensus"][1] == 0
    assert list(sp["orthants"]) == [("b", "c"), ("c",)]
    assert list(sp["n_hops"]) == [1, 0]


def test_round_A_tau_1_5():
    sp = spells(tau=1.5, **ROUND_A)
    assert as_tuples(sp) == [(2, 12, 10, "consensus")]
    assert list(sp["orthants"]) == [("b", "c")]
    assert list(sp["n_hops"]) == [1]


def test_round_B_failed():
    sp = spells([0, 5, 250, 290], [F, F, T, T], t_end=300, censored=True)
    assert as_tuples(sp) == [(250, 300, 50, "censored")]
    assert np.isnan(sp["t_to_consensus"][0])


def test_realistic_split_rows_then_consensus():
    # fraught rows, then 3 non-fraught split rows over 2 s, then the consensus row
    sp = spells([0, 10, 11, 12, 13, 13.5, 14], [F, T, T, F, F, F, F], t_end=14, censored=False)
    assert as_tuples(sp) == [(10, 12, 2, "consensus")]
    assert sp["t_to_consensus"][0] == 2


def test_unresolved_when_round_ends_without_consensus_or_relapse():
    sp = spells([0, 5, 10, 20], [F, T, F, F], t_end=300, censored=True)
    assert as_tuples(sp) == [(5, 10, 5, "unresolved")]
    assert np.isnan(sp["t_to_consensus"][0])


def test_t_end_before_last_row_raises():
    with pytest.raises(ValueError):
        spells([0, 5, 250, 300.08], [F, F, T, T], t_end=300, censored=True)


def test_tau_0_total_equals_fraught_hold_time_on_train_runs():
    runs, table = sl.load_runs("behavior", keep_whole_second=True)     # train runs, incl. bots and no-end-marker rounds
    assert set(table["role"]) == {"train"}
    for k, run in enumerate(runs):
        t = run["t"]
        limit = 300 if table["session"][k] == "lab" else 180
        success = table["success"][k]
        # t_end for spells: success -> last row; failed -> time limit (or the last row if later); no end marker -> last row
        t_end = max(limit, t[-1]) if table["end"][k] == "failed" else t[-1]
        sp = spells(t, fraught_mask(run), t_end, censored=not success, tau=0)
        T_hold = t[-1] if success else max(limit, t[-1])                # what fraught_percent_hold divides by
        assert np.isclose(sp["duration"].sum(), fraught_percent_hold(k, runs, table) * T_hold / 100)


def test_kaplan_meier_example():
    table, median = kaplan_meier([2, 3, 3, 5], [1, 0, 1, 1])
    S = dict(zip(table["time"], table["survival"]))
    assert S[2] == pytest.approx(0.75)
    assert S[3] == pytest.approx(0.5)
    assert S[5] == pytest.approx(0.0)
    assert median == 3

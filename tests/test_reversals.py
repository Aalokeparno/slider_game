"""detect_events (reversals.py) on hand-made series."""
import pytest

from slider.reversals import detect_events

T_A = [0, .2, .4, .5, .7, .9, 1.0, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 2.0, 3.0, 5.0, 6.0, 6.2, 6.4, 6.6, 7.5, 7.8, 8.0,
       8.2, 8.4, 8.6]
S_A = [0, .1, .2, .2, .3, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1, 1, .9, .8, .7, .7, .7, .6, .5, .6, .7, .8]


def events(ev):
    return [(r.time, r.type) for r in ev.itertuples()]


def test_A_m2():
    ev = detect_events(T_A, S_A, m=2)
    assert events(ev) == [(pytest.approx(0.9), "probe"), (pytest.approx(6.0), "reversal"),
                          (pytest.approx(8.2), "reversal")]


def test_A_m1():
    ev = detect_events(T_A, S_A, m=1)
    assert events(ev) == [(pytest.approx(0.9), "reversal"), (pytest.approx(1.0), "reversal"),
                          (pytest.approx(6.0), "reversal"), (pytest.approx(8.2), "reversal")]


def test_B_probe_after_long_pause():
    ev = detect_events([0, .5, 1, 10, 10.3, 11], [.8, .9, 1, .9, 1, 1], m=2)
    assert events(ev) == [(pytest.approx(10.0), "probe")]


def test_C_unresolved_at_round_end():
    ev = detect_events([0, 1, 2], [.5, .6, .5], m=2)
    assert events(ev) == [(pytest.approx(2.0), "unresolved")]


def test_D_jump_of_3_steps_is_a_reversal():
    ev = detect_events([0, 1, 2], [.5, .6, .3], m=2)
    assert events(ev) == [(pytest.approx(2.0), "reversal")]
    assert ev["n_steps"][0] == 3

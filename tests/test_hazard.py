"""build_hazard_table (hazard.py) on a hand-made round."""
import numpy as np
import pytest

from slider.hazard import build_hazard_table

# player 0, k by row (direction + after the first move):
#   t = 0.0: 7 | 0.5: 8 first move | 1.0: 9 enters the end band (a step exactly at a bin start)
#   1.6: 8 one step back (leaves the band) | 2.1: 9 back: a probe, first step 1.6, resolved 2.1
#   2.6: 8 | 3.3: 7 two steps back: a reversal, first step 2.6, resolved 3.3 (direction now -)
#   3.6: 8 one step against - | 4.0: 8, round ends: unresolved
# player 1 moves once (0.5 -> 0.6 at t = 1.0); the others sit at 0.5 and never move.
# Bins of 0.25 s from 0.5; dropped (mid-event): 1.75, 2.0 (probe), 2.75, 3.0, 3.25 (reversal), 3.75 (unresolved).
T = np.array([0, 0.5, 1.0, 1.6, 2.1, 2.6, 3.3, 3.6, 4.0])
X = np.full((len(T), 7), 0.5)
X[:, 0] = np.array([7, 8, 9, 8, 9, 8, 7, 8, 8]) / 10
X[2:, 1] = 0.6
A = X.sum(axis=1) / 7            # Agreement of every player, by row (Full network)


def tables(player=0):
    run = {"file": "f", "wave": 1, "dataset": "CL", "session": "lab", "end": "success", "bot": False,
           "bot_players": set(), "A_log": np.ones((7, 7), dtype=int), "A": np.ones((7, 7), dtype=int) - np.eye(7, dtype=int),
           "weights": np.ones(7), "t": T, "X": X}
    h, dropped = build_hazard_table([run], b=0.25)
    h, dropped = h[h["player"] == player], dropped[dropped["player"] == player]
    return h.set_index(np.round(h["t"], 2)), list(np.round(dropped["t"], 2))


def test_outcomes_in_the_bins_of_the_first_steps():
    h, _ = tables()
    assert list(h.index) == [0.5, 0.75, 1.0, 1.25, 1.5, 2.25, 2.5, 3.5]
    assert list(h.index[h["y_rev"] == 1]) == [2.5]          # first step 2.6
    assert list(h.index[h["y_probe"] == 1]) == [1.5]        # first step 1.6


def test_a_step_at_a_bin_start_uses_the_row_before():
    h, _ = tables()
    r = h.loc[1.0]                                          # the row at t = 1.0 (k 8 -> 9) is not used
    assert r["t_row"] == 0.5 and r["abs_s"] == pytest.approx(0.8) and r["at_end"] == 0
    assert r["A"] == pytest.approx(A[1])


def test_bins_inside_a_pending_window_are_dropped():
    h, dropped = tables()
    assert dropped == [1.75, 2.0, 2.75, 3.0, 3.25, 3.75]
    assert not set(dropped) & set(h.index)


def test_probe_then_dropped_until_it_returns():
    h, dropped = tables()
    assert h.loc[1.5, "y_probe"] == 1
    assert 1.75 in dropped and 2.0 in dropped and 2.25 in h.index      # returns at 2.1


def test_time_since_rev_ignores_an_unconfirmed_reversal():
    h, _ = tables()
    assert h.loc[2.5, "time_since_rev"] == pytest.approx(2.0)   # since the first move at 0.5
    assert h.loc[3.5, "time_since_rev"] == pytest.approx(0.9)   # since 2.6, confirmed at 3.3
    h1, _ = tables(player=1)                                    # no reversal at all: since its move at 1.0
    assert np.allclose(h1["time_since_rev"], h1["t"] - 1.0)


def test_time_at_end_resets_on_leaving_the_band():
    h, _ = tables()
    assert h.loc[1.5, "time_at_end"] == pytest.approx(0.5)      # in the band since 1.0
    assert h.loc[2.25, "time_at_end"] == pytest.approx(0.15)    # left at 1.6, back since 2.1
    assert h.loc[3.5, "time_at_end"] == 0                       # out of the band


def test_trend_uses_round_start_in_the_first_second():
    h, _ = tables()
    assert h.loc[0.5, "trend"] == pytest.approx(0)              # A(0.5-) and A(-0.5) both at the first row
    assert h.loc[0.75, "trend"] == pytest.approx(A[1] - A[0])   # (t - 1 s) < 0: A at the first row
    assert h.loc[1.25, "trend"] == pytest.approx(A[2] - A[0])   # (0.25)-: row 0

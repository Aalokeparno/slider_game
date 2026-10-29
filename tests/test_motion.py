"""motion_spells (motion.py) on a hand-made lab round, dt = 0.1 s.

Player 0, s by row: t = 0.1: 0.1 (first move) | 0.2: 0.2 | 0.3: 0.3 | 0.8: 0.4 (a 0.5 s gap, 1 step)
| 1.0: 0.6 (a 0.2 s gap, 2 steps). The round succeeds at t = 1.0. The other players never move.
"""
import numpy as np
import pytest

from slider.motion import motion_spells

T = np.array([0.1, 0.2, 0.3, 0.8, 1.0])
X = np.zeros((len(T), 7))
X[:, 0] = [0.1, 0.2, 0.3, 0.4, 0.6]
IFACE = {"interface": "lab_v09", "dt": 0.1, "step": 0.1}


def spells():
    run = {"file": "f", "wave": 1, "dataset": "EL", "session": "lab", "end": "success", "bot": False,
           "bot_players": set(), "A_log": np.ones((7, 7), dtype=int), "A": np.ones((7, 7), dtype=int) - np.eye(7, dtype=int),
           "weights": np.ones(7), "t": T, "X": X}
    seats, halts = motion_spells(run, IFACE)
    return seats.set_index("player"), halts[halts["player"] == 0]


def test_a_half_second_gap_then_one_step_is_a_halt_of_0_4():
    seats, halts = spells()
    h = halts[np.isclose(halts["start"], 0.3)]
    assert len(h) == 1 and h["duration"].iloc[0] == pytest.approx(0.4) and h["exit"].iloc[0] == "resume"
    assert seats.loc[0, "stops"] == 1 and seats.loc[0, "resumes"] == 1


def test_a_0_2_second_gap_then_two_steps_is_no_halt():
    seats, halts = spells()
    assert len(halts) == 1                                  # only the halt after t = 0.3
    assert seats.loc[0, "halted_s"] == pytest.approx(0.4)


def test_moving_time_is_steps_times_dt():
    seats, _ = spells()
    assert seats.loc[0, "moving_s"] == pytest.approx(6 * 0.1)       # 1 + 1 + 1 + 1 + 2 steps
    assert seats.loc[0, "start_delay"] == pytest.approx(0.0)         # 0.1 s gap from t = 0, 1 step
    assert seats.loc[0, "residual_s"] == pytest.approx(0.0)
    assert seats.loc[1, "pre_s"] == pytest.approx(1.0) and seats.loc[1, "moved"] == 0

"""lab_log (labclock.py) on hand-made rows."""
import numpy as np
import pytest

from slider.labclock import lab_log


def rows(s0):
    """Positions of 7 players: player 0 = s0, the others fixed at 0.5."""
    S = np.full((len(s0), 7), 0.5)
    S[:, 0] = s0
    return S


def test_A_short_dip_between_ticks_vanishes():
    _, S_lab = lab_log([0, .03, .06, .25], rows([1, .9, 1, 1]))
    assert not np.any(np.isclose(S_lab[:, 0], 0.9))


def test_B_dip_held_until_the_next_tick():
    t_lab, S_lab = lab_log([0, .03, .25], rows([1, .9, 1]))
    assert list(t_lab) == pytest.approx([0, .1, .25])
    assert list(S_lab[:, 0]) == pytest.approx([1, .9, 1])

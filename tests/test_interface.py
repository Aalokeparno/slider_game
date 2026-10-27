"""interface.py on a hand-made round."""
import numpy as np
import pytest

from slider.interface import interface_table


def test_steps_of_002_every_01_s():
    # player 0 moves 0.02 every 0.1 s; the others never move
    t = np.round(np.arange(21) * 0.1, 1)
    X = np.full((len(t), 7), 0.5)
    X[:, 0] = -0.2 + 0.02 * np.arange(len(t))
    run = {"file": "f", "wave": 1, "dataset": "CL", "bot_players": set(), "weights": np.ones(7), "t": t, "X": X}
    r = interface_table([run]).iloc[0]
    assert r["step_size"] == pytest.approx(0.02)
    assert r["modal_pace"] == pytest.approx(10)
    assert r["speed"] == pytest.approx(0.2)
    assert not r["flagged"]

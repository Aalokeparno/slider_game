"""crossing_events (crossings.py) on hand-made series."""
import numpy as np
import pytest

from slider.crossings import crossing_events

# rows every 0.1 s: k = -10 until 0.9, -9 at 1.0, +1 per row to +3 at 2.2, hold to 3.0,
# +2 at 3.1, +3 at 3.2, +1 per row to +9 at 3.8
T = np.round(np.arange(39) * 0.1, 1)
K = np.array([-10] * 10 + list(range(-9, 4)) + [3] * 8 + [2, 3] + list(range(4, 10)))


def test_a_complete_crossing_with_a_pause_and_a_probe():
    c = crossing_events(T, K, T, K)
    assert len(c) == 1
    r = c.iloc[0]
    assert r["status"] == "complete" and not r["has_jump"]
    assert (r["start"], r["end"]) == (pytest.approx(1.1), pytest.approx(3.8))
    assert r["duration"] == pytest.approx(2.7)
    assert r["pause_time"] == pytest.approx(0.9) and r["moving"] == pytest.approx(1.8)
    assert r["steps"] / r["moving"] == pytest.approx(10)


def test_b_two_steps_back_is_aborted():
    k = K.copy()
    k[T == 3.1] = 1                     # back 2 steps from +3
    c = crossing_events(T, k, T, k)
    assert list(c["status"]) == ["aborted"]


def test_c_raw_jump_inside_the_crossing():
    raw_t = np.insert(T, 25, [2.45, 2.47])     # between the rows at 2.4 and 2.5
    raw_k = np.insert(K, 25, [8, 3])           # +5 steps, then back
    c = crossing_events(T, K, raw_t, raw_k)
    assert list(c["status"]) == ["complete"] and c["has_jump"].iloc[0]

"""true_events (apem_truth.py) and the APEM flip log."""
import numpy as np
import pytest

from slider.apem_truth import simulate, true_events


def test_one_agent_probe_then_reversal():
    ev = true_events([0, .5, .6, 1.0, 1.5], [0, .5, .4, .8, .3], m_dist=0.2)
    assert [(r.time, r.type) for r in ev.itertuples()] == [(pytest.approx(0.5), "probe"),
                                                         (pytest.approx(1.0), "reversal")]


def test_true_event_times_are_flip_times_on_apem():
    run = simulate("Hierarchy", n_runs=1)[0]
    flips = np.array(run["flips"])
    for j in range(7):
        ev = true_events(run["t"], run["X"][:, j])
        assert np.all(np.isin(ev["time"], flips[flips[:, 1] == j, 0]))

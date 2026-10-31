"""Model C, C1 (EXPLORATORY): the old presets stay bit-identical; trip pace; trips measured on the simulated logs
are the drawn durations; the hazard runs only at the wall; round ends and time accounting; the data-side trips."""
import json

import numpy as np
import pandas as pd
import pytest

from slider import config, fraught, model_c1, sim
from slider.models.apem import GRAPHS

import fingerprint_presets

POOL = {"first": np.array([1.23, 2.0]), "cross": np.array([3.0, 4.46])}


def test_old_presets_are_bit_identical():
    assert fingerprint_presets.fingerprints() == json.loads(fingerprint_presets.PATH.read_text())


@pytest.mark.parametrize("n, L", [(10, 10), (10, 13), (20, 47), (100, 101)])
def test_trip_pace(n, L):
    done = sim.trip_steps(np.arange(-1, L), L, n)
    steps = np.diff(done)
    assert done[0] == 0 and done[-1] == n and set(steps) <= {0, 1} and steps[0] == 1 and steps[-1] == 1


def run_of(o, iface, T, seed):
    meta = {"group": f"sim_{sim.IFACES[iface]['tag']}_c1.txt", "dataset": "CC" if iface == "cc" else "CL",
            "topology": "Hierarchy", "wave": 1, "graph": GRAPHS["Hierarchy"].astype(int), "seat_node": range(7),
            "interface": iface, "time_limit": T, "end": "success" if o["success"] else "failed", "seed": seed}
    return sim.to_run(o["events"], meta, "cloud" if iface == "cc" else "lab")


@pytest.fixture(scope="module")
def c1_rounds():
    """C1 rounds (Hierarchy, each interface, R = 15) with a two-value duration pool, their runs and traces."""
    out = {}
    for k, iface in enumerate(["lab_v09", "old_cl", "cc"]):
        T = config.TIME_LIMIT[sim.IFACES[iface]["session"]]
        trace = []
        rounds = sim.simulate(GRAPHS["Hierarchy"], T, iface, R=15, seed=k, trace=trace, **{**sim.preset("C1"), "c1": POOL})
        out[iface] = (rounds, [run_of(o, iface, T, k) for o in rounds], trace)
    return out


@pytest.mark.parametrize("iface", ["lab_v09", "old_cl", "cc"])
def test_logged_trips_are_the_drawn_durations(c1_rounds, iface):
    """Every trip ends at its wall, and the logs measure round(D / dt) dt (CC: within the 2 ms logger split)."""
    rounds, runs, _ = c1_rounds[iface]
    dt = rounds[0]["dt"]
    t = pd.DataFrame([r for run in runs for j in range(7)
                      for r in model_c1.seat_trips(np.r_[0.0, run["t"]], np.r_[0.0, run["X"][:, j]])])
    assert set(t["outcome"]) <= {"arrived", "censored"}                       # no trip turns back
    N = round(1 / sim.IFACES[iface]["d"])
    for kind, n in [("first", N), ("cross", 2 * N)]:
        want = np.maximum(np.round(POOL[kind] / dt), n - 1) * dt                # at least n steps, one per tick
        got = t.loc[(t["kind"] == kind) & (t["outcome"] == "arrived"), "duration"].to_numpy()
        assert len(got) > 0 and np.all(np.min(np.abs(got[:, None] - want[None, :]), axis=1) <= 0.006)


@pytest.mark.parametrize("iface", ["lab_v09", "old_cl", "cc"])
def test_hazard_only_at_the_wall(c1_rounds, iface):
    rounds, runs, trace = c1_rounds[iface]
    bins = pd.concat([d for d in trace if "tau" in d], ignore_index=True)
    assert bins["risk"].any() and np.all(bins.loc[bins["risk"], "abs_s"] == 1)
    assert all(o["mode_ticks"][:, sim.REVERSING].sum() == 0 for o in rounds)


@pytest.mark.parametrize("iface", ["lab_v09", "old_cl", "cc"])
def test_round_end_and_time_accounting(c1_rounds, iface):
    rounds, runs, _ = c1_rounds[iface]
    for o, run in zip(rounds, runs):
        t_end = fraught.round_end(run["t"], run["end"], run["session"])
        assert abs(t_end - o["end_time"]) <= o["dt"] + 1e-9
        assert sim.consensus(run["X"][-1]) == o["success"] and np.abs(run["X"]).max() <= 1
        ticks = o["mode_ticks"]
        assert np.all(ticks.sum(axis=1) == o["ticks"])
        steps = np.bincount(o["events"]["seat"].iloc[7:], minlength=7)
        assert np.array_equal(steps, ticks[:, sim.MOVING])


def test_seat_trips_on_a_synthetic_path():
    t = np.arange(12, dtype=float)
    s = np.array([0, 0, 0.5, 1, 1, 0.9, 1, 0.5, 0, -0.5, -1, -0.5])
    r = pd.DataFrame(model_c1.seat_trips(t, s))
    assert list(r["kind"]) == ["first", "cross", "cross", "cross"]
    assert list(r["outcome"]) == ["arrived", "returned", "arrived", "censored"]
    assert list(r["duration"].fillna(-1)) == [1.0, -1, 3.0, -1] and list(r["wall"].fillna(0)) == [1, 0, -1, 0]

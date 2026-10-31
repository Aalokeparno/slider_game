"""Model C, C2 (EXPLORATORY): the old presets stay bit-identical; keep = 1 commits every try at once; a walk that
reaches the wall returns to sitting; the look reads the positions from before the decision point; tries only at the
wall; round ends and time accounting."""
import json

import numpy as np
import pandas as pd
import pytest

from slider import config, fraught, sim
from slider.models.apem import GRAPHS

import fingerprint_presets
from test_c1 import run_of

BASE = {"d_ar": 0.0, "q_w": 0.3, "keep_pos": 0.8, "keep_zero": 0.7, "keep_neg": 0.6, "pace": np.array([1.0, 0.5])}


def test_old_presets_are_bit_identical():
    assert fingerprint_presets.fingerprints() == json.loads(fingerprint_presets.PATH.read_text())


def c2_rounds(iface, c2, R=10, seed=0):
    T = config.TIME_LIMIT[sim.IFACES[iface]["session"]]
    trace = []
    rounds = sim.simulate(GRAPHS["Hierarchy"], T, iface, R=R, seed=seed, trace=trace, **{**sim.preset("C2"), "c2": c2})
    return rounds, [run_of(o, iface, T, seed) for o in rounds], trace


def seat_paths(o):
    """{seat: (times, s) of its position changes} from a round's events (the round start dropped)."""
    e = o["events"].iloc[7:]
    return {j: (g["time"].to_numpy(), g["s"].to_numpy()) for j, g in e.groupby("seat")}


def excursions(t, s):
    """(departure time, return time or NaN, positions) of each departure from a wall until the next wall touch."""
    t, s = np.r_[0.0, t], np.r_[0.0, s]
    out, cur = [], None
    for i in range(1, len(s)):
        if abs(s[i - 1]) >= 1 - 1e-9:
            cur = [t[i], np.nan, [s[i - 1], s[i]]]
            out.append(cur)
        elif cur is not None:
            cur[2].append(s[i])
        if abs(s[i]) >= 1 - 1e-9 and cur is not None:
            cur[1] = t[i]
            cur = None
    return out


@pytest.mark.parametrize("iface", ["lab_v09", "old_cl", "cc"])
def test_keep_one_commits_every_try(iface):
    rounds, _, trace = c2_rounds(iface, {**BASE, "keep_pos": 1.0, "keep_zero": 1.0, "keep_neg": 1.0})
    looks = pd.concat([d for d in trace if "look" in d], ignore_index=True)
    assert len(looks) > 0 and looks["kept"].all()
    n_ex = 0
    for o in rounds:
        for t, s in seat_paths(o).values():
            for _, _, ex in excursions(t, s):
                d = np.sign(np.diff(ex))
                assert np.all(d == d[0])                              # one direction, no walk steps
                if abs(ex[-1]) >= 1 - 1e-9:
                    assert np.sign(ex[-1]) == -np.sign(ex[0])         # ... to the opposite wall
                n_ex += 1
    assert n_ex > 0


@pytest.mark.parametrize("iface", ["lab_v09", "cc"])
def test_walk_that_reaches_the_wall_sits_again(iface):
    rounds, runs, trace = c2_rounds(iface, {**BASE, "keep_pos": 0.0, "keep_zero": 0.0, "keep_neg": 0.0, "q_w": 1.0})
    bins = pd.concat([d for d in trace if "tau" in d], ignore_index=True)
    risk = bins[bins["risk"]]
    returned = 0
    for r, o in enumerate(rounds):
        for j, (t, s) in seat_paths(o).items():
            ex = excursions(t, s)
            for (_, back, a), (leave, _, _) in zip(ex[:-1], ex[1:]):
                if np.sign(a[-1]) == np.sign(a[0]):                   # the walk came back to its wall ...
                    returned += 1
                    x = risk[(risk["rep"] == r) & (risk["player"] == j)]
                    assert ((x["tau"] > back) & (x["tau"] < leave + 0.11)).any()   # ... sat there, at risk again
    assert returned > 0
    assert bins["risk"].any() and np.all(bins.loc[bins["risk"], "abs_s"] >= 1 - 1e-9)    # at risk only at the wall
    assert all(o["mode_ticks"][:, sim.REVERSING].sum() == 0 for o in rounds)


@pytest.mark.parametrize("iface", ["lab_v09", "old_cl", "cc"])
def test_look_uses_positions_before_the_decision_point(iface):
    rounds, _, trace = c2_rounds(iface, BASE, R=4, seed=3)
    f = sim.IFACES[iface]
    N = round(1 / f["d"])
    A_log = GRAPHS["Hierarchy"].astype(float) + np.eye(7)
    looks = pd.concat([d for d in trace if "look" in d], ignore_index=True)
    checked = 0
    for r, o in enumerate(rounds):
        e = o["events"].iloc[7:]
        tick = np.round(e["time"].to_numpy() / o["dt"]).astype(int)          # a step made in tick j has tick j + 1

        def A_at(j):                                                          # positions at the start of tick j
            s = np.zeros(7)
            for seat, x in zip(e["seat"].to_numpy()[tick <= j], e["s"].to_numpy()[tick <= j]):
                s[seat] = x
            return np.abs(A_log @ s) / A_log.sum(axis=1)

        for x in looks[looks["rep"] == r].head(30).itertuples():
            want = A_at(x.look)[x.player] - A_at(x.look - 1)[x.player]           # D = 1: the step was in tick look - 1
            assert x.dA == pytest.approx(want, abs=1e-12)
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("iface", ["lab_v09", "old_cl", "cc"])
def test_round_end_and_time_accounting(iface):
    rounds, runs, _ = c2_rounds(iface, BASE)
    for o, run in zip(rounds, runs):
        t_end = fraught.round_end(run["t"], run["end"], run["session"])
        assert abs(t_end - o["end_time"]) <= o["dt"] + 1e-9
        assert sim.consensus(run["X"][-1]) == o["success"] and np.abs(run["X"]).max() <= 1
        ticks = o["mode_ticks"]
        assert np.all(ticks.sum(axis=1) == o["ticks"])
        steps = np.bincount(o["events"]["seat"].iloc[7:], minlength=7)
        assert np.array_equal(steps, ticks[:, sim.MOVING])

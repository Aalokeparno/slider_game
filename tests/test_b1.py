"""Model B, B1 (EXPLORATORY): the old presets stay bit-identical; B1 with p_i = 0, q_sg = q_E = L1's stop chance and
L1's resume chance is L1 exactly; the state labels (data side and simulator agree); the style draw."""
import json

import numpy as np
import pytest

from slider import model_b1, sim, stop_states
from slider.models.apem import GRAPHS

import fingerprint_presets

A_FULL = GRAPHS["Full"].astype(int)


def same(a, b):
    return all(x["events"].equals(y["events"]) and np.array_equal(x["mode_ticks"], y["mode_ticks"]) for x, y in zip(a, b))


def test_old_presets_are_bit_identical():
    assert fingerprint_presets.fingerprints() == json.loads(fingerprint_presets.PATH.read_text())


@pytest.mark.parametrize("iface", ["lab_v09", "old_cl", "cc"])
def test_b1_with_no_smooth_crossings_and_l1_chances_is_exactly_l1(iface):
    rule = sim.Logistic("M2'")
    q = sim.constants(A_FULL, iface, rule)["p"]["stop"]
    l1 = sim.simulate(A_FULL, 90, iface, rule, motion="L1", R=4, seed=11)
    b1 = sim.simulate(A_FULL, 90, iface, rule, motion="B1", R=4, seed=11, trait=np.zeros((4, 7)),
                      b1={"pi": 0.4, "rho": 0.2, "q_sg": q, "q_E": q})
    assert same(l1, b1)


def test_style_draw():
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    assert list(sim.draw_style([0.0, 1.0, 1.0, 0.0], rng)) == [False, True, True, False]
    assert rng.bit_generator.state == state                       # p in {0, 1}: no draw
    assert abs(sim.draw_style(np.full(20000, 0.3), rng).mean() - 0.3) < 0.015
    p = sim.trait_p(np.random.default_rng(1).random(200000), 0.4, 0.25)
    assert abs(p.mean() - 0.4) < 0.005 and abs(p.var() - 0.4 * 0.6 * 0.25) < 0.003   # Beta: var = pi (1 - pi) rho
    assert sim.trait_p(np.array([0.0, 1.0]), 0.4, 0.25).tolist() == [0.0, 1.0]


def path(moves):
    """t, s (t = 0, s = 0 row first) of one seat on the 0.1 grid, dt = 0.1: moves = +1 / -1 steps, "h" = a 1 s halt."""
    t, s, now, pos = [0.0], [0.0], 0.0, 0
    for mv in moves:
        if mv == "h":
            now += 1.0
            continue
        now += 0.1
        pos += mv
        t.append(round(now, 2))
        s.append(pos / 10)
    return np.array(t), np.array(s)


def test_halt_states_on_a_synthetic_path():
    up, down = [+1], [-1]
    moves = (up * 5 + ["h"] + up * 4            # first trip: halt at 0.5 (E), reach 0.9
             + down * 4 + ["h"]                 # leave the band (the turn is not a reversal of the phase): C at 0.5
             + down * 14                        # reach -0.9
             + up * 4 + ["h"]                   # leave the band again: C at -0.5
             + down * 2 + ["h"]                 # reverse inside the crossing: E at -0.7
             + down)
    t, s = path(moves)
    h = model_b1.seat_halts(t, s, 0.1, 0.1, t[-1] + 0.04, False)          # round end: no halt after the last step
    assert [x["state"] for x in h] == ["E", "C", "C", "E"]
    assert np.allclose([x["start"] for x in h], [t[np.flatnonzero(np.isclose(s, v))[i]] for v, i in
                                                 [(0.5, 0), (0.5, 1), (-0.5, 1), (-0.7, 2)]])


def sim_steps(iface, trait, b1, R=8):
    T = 180 if iface == "cc" else 300                                  # the session's time limit (round_end)
    out = sim.simulate(A_FULL, T, iface, sim.Logistic("M2'"), motion="B1", R=R, seed=3, trait=trait, b1=b1)
    runs = []
    for r, o in enumerate(out):
        meta = {"group": f"sim_test_r{r}.txt", "dataset": "CC" if iface == "cc" else "CL", "topology": "Full",
                "wave": 1, "graph": A_FULL, "seat_node": range(7), "interface": iface,
                "time_limit": T, "end": "success" if o["success"] else "failed", "seed": 3}
        runs.append(sim.to_run(o["events"], meta, "cloud" if iface == "cc" else "lab"))
    return model_b1.step_state(stop_states.step_table(runs))


@pytest.mark.parametrize("iface", ["lab_v09", "old_cl", "cc"])
def test_simulated_states_are_the_data_states(iface):
    """All crossings smooth: no stop in C; no stops in E: every stop is in C (the estimator's labels)."""
    smooth = sim_steps(iface, np.ones((8, 7)), {"pi": 0.5, "rho": 0.2, "q_sg": 0.5, "q_E": 0.3})
    assert (smooth["state"] == "C").sum() > 200 and smooth.loc[smooth["state"] == "C", "stop"].sum() == 0
    assert smooth.loc[smooth["state"] == "E", "stop"].sum() > 50
    only_c = sim_steps(iface, np.zeros((8, 7)), {"pi": 0.5, "rho": 0.2, "q_sg": 0.4, "q_E": 0.0})
    assert only_c.loc[only_c["state"] == "E", "stop"].sum() == 0
    assert only_c.loc[only_c["state"] == "C", "stop"].sum() > 50


# ------------------------------------------------------------------ build fix 1: style fit per attempt

def test_mixture_likelihood_is_the_beta_integral():
    """The exact polynomial integral equals a direct numerical integral over p, for players with mixed attempts."""
    import pandas as pd
    from scipy import integrate, stats
    att = pd.DataFrame({"file": ["f1"] * 4 + ["f2"] * 3, "player": 0,
                        "n": [3, 7, 1, 5, 2, 9, 4], "k": [0, 2, 0, 0, 1, 0, 0]})
    pi, rho, q = 0.35, 0.2, 0.4
    a, b = pi * (1 - rho) / rho, (1 - pi) * (1 - rho) / rho
    direct = 0.0
    for _, x in att.groupby("file"):
        f = lambda p: np.prod(p * (x["k"] == 0) + (1 - p) * q ** x["k"] * (1 - q) ** (x["n"] - x["k"])) * stats.beta.pdf(p, a, b)
        direct += np.log(integrate.quad(f, 0, 1, epsabs=1e-13, epsrel=1e-11)[0])
    assert model_b1.mixture_loglik(pi, rho, q, model_b1.player_data(att)) == pytest.approx(direct, rel=1e-8)


def test_mixture_fit_recovers_known_parameters():
    rng = np.random.default_rng(5)
    players_n = [rng.integers(1, 15, rng.integers(1, 15)) for _ in range(800)]
    fit = model_b1.fit_attempts(model_b1.simulate_attempts(players_n, 0.4, 0.25, 0.45, rng))
    assert fit[0] == pytest.approx(0.4, abs=0.03) and fit[1] == pytest.approx(0.25, abs=0.06)
    assert fit[2] == pytest.approx(0.45, abs=0.01)

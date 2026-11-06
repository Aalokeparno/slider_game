"""Path 3 strategy simulator (src/slider/path3): game rules, graph measures, outcome classes,
the information constraint on strategies, and the four required behaviour tests.

The behaviour tests (1-3) use the human reaction / start delays (lab v09, G-train); they take a few minutes.
"""
import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from slider import config, interface
from slider.path3 import game, graphs, human, sim, strategies

H = game.HUMAN_GRAPHS["Hierarchy"]
F0 = game.corner_from_sets(plus={1, 3, 6, 7}, minus={2, 4, 5})


@pytest.fixture(scope="module")
def delays():
    return human.load_delays()


# ------------------------------------------------------------------ game rules

def test_interface_is_lab_v09():
    # lab v09 files: modal step 0.1, modal speed 1 unit/s (interface.py) -> dt = 0.1 s, one step per tick
    files = pd.read_csv(config.INTERFACE_TABLE)
    v09 = files[files["version"] == "v09"]
    assert interface.mode(v09["step_size"], 3) == pytest.approx(game.STEP)
    assert interface.mode(v09["speed"], 3) == pytest.approx(game.STEP / game.DT)


def test_agreement_and_display():
    k = np.array([10, -10, 10, -10, -10, 10, 10])          # s = k / 10
    agr = game.agreement(H, k, display="raw")
    # player 1 (index 0): itself + 2 + 3 = 1 - 1 + 1 = 1 over 3
    assert agr[0] == pytest.approx(1 / 3)
    assert game.agreement(H, k, display="int")[0] == pytest.approx(0.33)
    # half-up rounding of the percent: 2.5 % -> 3 %
    k = np.zeros(7, dtype=int)
    k[1] = 1                                               # s_2 = 0.1; player 1 has 3 members -> 3.33 %
    assert game.agreement(H, k, display="int")[0] == pytest.approx(0.03)
    k = np.zeros(7, dtype=int)
    k[0] = 1                                               # player 2 (index 1): 0.1 / 4 = 2.5 % -> 3 %
    assert game.agreement(H, k, display="int")[1] == pytest.approx(0.03)


def test_consensus_rule():
    assert game.is_consensus(np.full(7, 10))
    assert game.is_consensus(np.full(7, -10))
    k = np.full(7, 10)
    k[0] = 9
    assert not game.is_consensus(k)                        # |sum| / 7 = 0.986 < 0.99


def test_fraught_corners_of_hierarchy():
    corners = game.fraught_corners(H)
    assert len(corners) == 4
    assert any(np.array_equal(c, F0) for c in corners)
    assert len(game.non_consensus_corners()) == 126


def test_lambda_of_known_graphs():
    assert game.second_eigenvalue(game.HUMAN_GRAPHS["Full"]) == pytest.approx(0, abs=1e-9)
    assert game.second_eigenvalue(game.HUMAN_GRAPHS["Ring"]) == pytest.approx((1 + 2 * np.cos(2 * np.pi / 7)) / 3)


def test_atlas_has_853_connected_graphs():
    atlas = graphs.atlas_table()
    assert len(atlas) == 853
    names = atlas.dropna(subset=["human"])["human"]
    assert sorted(names) == sorted(game.HUMAN_GRAPHS)


# ------------------------------------------------------------------ outcome classes

def test_outcome_classes():
    n = 3001
    still = np.tile(F0 * 10, (n, 1))
    assert sim.outcome_class(still, success=False)[0] == "fixed trap"
    # one slider wiggles between 1.0 and 0.9 forever: sign pattern constant -> fixed trap
    wiggle = still.copy()
    wiggle[::2, 0] = 9
    assert sim.outcome_class(wiggle, success=False) == ("fixed trap", 0)
    # sign pattern constant in the last 30 s only, after earlier changes -> fixed trap
    late = still.copy()
    late[:-400:7, 2] = -10
    assert sim.outcome_class(late, success=False)[0] == "fixed trap"
    # two sign patterns alternating every 10 ticks -> periodic, period 2
    alt = still.copy()
    alt[(np.arange(n) // 10) % 2 == 1, 0] = -10
    assert sim.outcome_class(alt, success=False) == ("periodic", 2)
    # patterns A B A C A B B ... with no fixed period -> irregular
    rng = np.random.default_rng(0)
    irr = still.copy()
    flips = rng.integers(0, 7, size=n // 20)
    for j, i in enumerate(flips):
        irr[j * 20:(j + 1) * 20, i] *= -1
    assert sim.outcome_class(irr, success=False)[0] == "irregular"
    assert sim.outcome_class(irr, success=True)[0] == "consensus"


# ------------------------------------------------------------------ test 4: information constraint

def test_strategies_see_only_their_own_inputs(delays):
    # (a) the decide signature is exactly (t, s_i, agr_history, memory)
    for cls in strategies.STRATEGIES.values():
        params = list(inspect.signature(cls.decide).parameters)
        assert params == ["self", "t", "s_i", "agr_history", "memory"]
    # (b) the strategies module imports nothing that knows the game, the graph or the simulator
    tree = ast.parse(inspect.getsource(strategies))
    imported = {a.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                for a in node.names} | {node.module.split(".")[0] for node in ast.walk(tree)
                                        if isinstance(node, ast.ImportFrom) and node.module}
    assert imported <= {"math"}
    # (c) during a run, every call gets a float time, a float own position, the player's own Agreement
    #     history (one value per tick so far) and a memory holding only its rng, its own positions and
    #     the strategy's own entries; the strategy object holds only its parameters
    seen = []

    class Spy(strategies.S2):
        def decide(self, t, s_i, agr_history, memory):
            seen.append((t, s_i, list(agr_history), set(memory), dict(vars(self))))
            return super().decide(t, s_i, agr_history, memory)

    out = sim.run(H, Spy, {}, F0 * 10, delays, np.random.default_rng(1), t_max=20)
    assert len(seen) > 20
    for t, s_i, hist, keys, attrs in seen:
        assert isinstance(t, float) and isinstance(s_i, float)
        assert len(hist) == round(t / game.DT) + 1 and all(isinstance(a, float) for a in hist)
        assert {"rng", "s_history"} <= keys
        assert keys - {"rng", "s_history"} <= strategies.S2.MEMORY_KEYS
        assert set(attrs) <= {"eps", "H", "T", "p", "kappa", "B", "Q"}
    assert out["strategy"] == "S2"


def test_s2_v2_clock_ignores_blips():
    s2 = strategies.S2(B=3)
    agr = [0.5, 0.5, 0.6, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5]     # two blips of 1 and 2 ticks: no reset
    mem = {"s_history": [1.0] * len(agr)}
    assert s2.flat(0.8, agr, mem) == pytest.approx(0.8)
    agr += [0.7, 0.7, 0.7, 0.7]                               # a change that lasts 3 ticks: reset at its 3rd tick
    mem["s_history"] += [1.0] * 4
    assert s2.flat(1.2, agr, mem) == pytest.approx(0.1)
    # own moves never count: Agr changes only at ticks where the player moved, then stays
    mem = {"s_history": [1.0, 0.9, 0.9, 0.9, 0.9, 0.9]}
    assert s2.flat(0.5, [0.5, 0.4, 0.4, 0.4, 0.4, 0.4], mem) == pytest.approx(0.5)


# ------------------------------------------------------------------ tests 1-3: behaviour

def runs_from(cls, params, start, delays, n, seed0, t_max=300):
    return [sim.run(H, cls, params, start, delays, np.random.default_rng(seed0 + r), t_max=t_max, keep_path=True)
            for r in range(n)]


def test_1_s0a_never_leaves_F0(delays):
    signs = np.sign(F0)
    for out in runs_from(strategies.S0a, {}, F0 * 10, delays, 200, 1000):
        assert not out["success_300"]
        assert np.all(np.sign(out["path"]) == signs), "S0a changed the sign pattern"


@pytest.mark.parametrize("cls", [strategies.S0b, strategies.S2, pytest.param(strategies.S2_v1, marks=pytest.mark.xfail(
    reason="known failure (v1): S2 escapes F0 in 92% of 200 runs (bar 95%)"))])
def test_2_yield_strategies_escape_F0(cls, delays):
    outs = runs_from(cls, {}, F0 * 10, delays, 200, 2000)
    rate = np.mean([o["success_300"] for o in outs])
    print(cls.name, "success_300 from F0:", rate)
    assert rate >= 0.95, rate


@pytest.mark.parametrize("cls", [strategies.S1p, pytest.param(strategies.S1p_v1, marks=pytest.mark.xfail(
    reason="known failure (v1): S1' kappa = 1.05 reaches consensus in 3.2% of 126 corners x 20 runs (bar 95%)"))])
def test_3_s1_from_every_non_consensus_corner(cls, delays):
    ok = []
    for c, corner in enumerate(game.non_consensus_corners()):
        for r in range(20):
            out = sim.run(H, cls, {"kappa": 1.05}, corner * 10, delays, np.random.default_rng(3000 + 20 * c + r))
            ok.append(out["success_300"])
    print(cls.name, "success_300 over 126 corners x 20 runs:", np.mean(ok))
    assert np.mean(ok) >= 0.95, np.mean(ok)


def test_s1p_v2_probes_only_when_quiet(delays):
    out = sim.run(H, strategies.S1p, {"Q": 2}, F0 * 10, delays, np.random.default_rng(7), t_max=60, keep_path=True)
    path = out["path"]
    agr = np.array([game.agreement(H, k) for k in path])
    probes = sim.probe_table(H, path, out["memory"])
    assert len(probes) > 10
    for i, tick in zip(probes["player"], probes["tick"]):
        for j in [tick, tick - 1]:                          # the last Q = 2 ticks before the probe step
            assert agr[j, i] == agr[j - 1, i] and path[j, i] == path[j - 1, i]

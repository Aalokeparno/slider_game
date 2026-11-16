"""Path 3 S3.2: the new strategies S3 (one-seed wave), S4 (seniority flags), S6 (backoff), S7 (fast/slow) and the
simulator's noise model (fidget, slip; S3 only).

    test 1  S0b''' reruns equal the saved T3 / T4 screening rows and a sample of T5 rows (same seeds)
    test 2  S4 with w = 1 is S0b'''; S6 with w_min = W_max = 0 is S0b'''; S7 with q = 1 is S6 (same paths)
    test 3  S3a+ on Hierarchy from start A, no noise, 200 runs: all reach consensus, no fraught-corner entry,
            no fallback yield in >= 95% of runs (slow)
    test 4  from Hierarchy's fraught corner F0: S4-new, S6, S7 (q = 0) reach consensus in >= 95% of 200 runs (slow)
    test 5  information constraint for the new classes
plus small unit tests of each new rule.
"""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from slider import config
from slider.path3 import game, graphs, human, sim, strategies, tail

H = game.HUMAN_GRAPHS["Hierarchy"]
F0 = game.corner_from_sets(plus={1, 3, 6, 7}, minus={2, 4, 5})
S0B3 = {"eps": 0.75, "H": 4.0, "soft_gate": True}
BLOCKS = config.PATH3 / "blocks"
START_A = np.zeros(7, dtype=int)


@pytest.fixture(scope="module")
def delays():
    return human.load_delays()


def same_run(a, b):
    return np.array_equal(a["path"], b["path"]) and a["n_yields"] == b["n_yields"]


# ------------------------------------------------------------------ test 1: S0b''' reruns equal saved rows

def same_value(a, b):
    return (a == b) or (pd.isna(a) and pd.isna(b))


@pytest.mark.parametrize("block", ["T3a", "T4a"])
def test_1_s0b3_reruns_equal_saved_screening_rows(block, delays):
    path = BLOCKS / f"{block}.csv"
    if not path.exists():
        pytest.skip(f"{path} not here (git-ignored block file)")
    saved = pd.read_csv(path, keep_default_na=False, na_values=[""])
    saved = saved[saved["variant"] == "S0b'''"].sample(60, random_state=0)
    screen = graphs.load_screening_set()
    for _, row in saved.iterrows():
        rng = np.random.default_rng([int(x) for x in row["seed"].split("-")])
        out = sim.run(screen[row["graph"]], strategies.S0b, json.loads(row["params"]), START_A, delays, rng)
        for col in ["t_consensus", "n_yields", "n_collisions", "outcome", "breakups", "yields_undone"]:
            assert same_value(out[col], row[col]), (row["graph"], row["seed"], col, out[col], row[col])


def test_1_s0b3_reruns_equal_saved_T5_rows(delays):
    files = sorted((BLOCKS / "T5").glob("chunk_*.csv"))
    if not files:
        pytest.skip("T5 chunk files not here (git-ignored)")
    saved = pd.concat([pd.read_csv(p) for p in files])
    saved = saved[saved["variant"] == "S0b''' 0.75"].sample(60, random_state=0)
    split = graphs.load_split().set_index("graph")
    for _, row in saved.iterrows():
        g = split.loc[row["graph"]]
        A = np.zeros((7, 7))
        for i, j in json.loads(g["edges"]):
            A[i, j] = A[j, i] = 1
        out = sim.run(A, strategies.S0b, S0B3, START_A, delays, np.random.default_rng([5731, int(g["atlas"]), row["run"]]))
        for col in ["t_consensus", "n_yields", "breakups"]:
            assert same_value(out[col], row[col]), (row["graph"], row["run"], col, out[col], row[col])


# ------------------------------------------------------------------ test 2: limits that give back the base strategy

def pairs(delays, cls_a, params_a, cls_b, params_b, n=15):
    for name in ["Hierarchy", "Full", "Ring"]:
        A = game.HUMAN_GRAPHS[name]
        for start in [START_A, F0 * 10]:
            for r in range(n):
                a = sim.run(A, cls_a, params_a, start, delays, np.random.default_rng([9, r]), keep_path=True)
                b = sim.run(A, cls_b, params_b, start, delays, np.random.default_rng([9, r]), keep_path=True)
                yield name, r, a, b


def test_2_s4_with_weight_one_is_s0b3(delays):
    # S4-old with A = 0 and S4-new with A = inf both have w = 1 at every decision
    for params in [{"mode": "old", "A": 0.0}, {"mode": "new", "A": np.inf}]:
        for name, r, a, b in pairs(delays, strategies.S4, params, strategies.S0b, S0B3):
            assert same_run(a, b), (params, name, r)


def test_2_s6_without_wait_is_s0b3(delays):
    for name, r, a, b in pairs(delays, strategies.S6, {"w_min": 0.0, "W_max": 0.0}, strategies.S0b, S0B3):
        assert same_run(a, b), (name, r)


def test_2_s7_all_slow_is_s6(delays):
    for name, r, a, b in pairs(delays, strategies.S7, {"q": 1.0}, strategies.S6, {}):
        assert same_run(a, b), (name, r)


def test_noise_off_changes_nothing(delays):
    # fidget = slip = 0 gives the same rounds as a run without the noise arguments
    for r in range(10):
        a = sim.run(H, strategies.S3, {}, START_A, delays, np.random.default_rng(r), keep_path=True)
        b = sim.run(H, strategies.S3, {}, START_A, delays, np.random.default_rng(r), keep_path=True, fidget=0.0, slip=0.0)
        assert same_run(a, b)


# ------------------------------------------------------------------ unit tests of the new rules

def mem(s_hist, seed=0):
    return {"rng": np.random.default_rng(seed), "s_history": list(s_hist)}


def test_s4_age_counts_side_changes_only():
    p = strategies.S4(mode="old", A=10.0)
    # 0 -> probe 0.1 -> back to 0 -> -1 (first wall: not a side change) -> probe -0.9 and back
    s = [0.0, 0.1, 0.0] + [-0.1 * j for j in range(1, 11)] + [-0.9, -1.0]
    m = mem(s)
    assert p.weight(1.5, m) == 1.0                      # never changed side
    # yield from -1 to +1: the sign flips at the tick where s first is > 0
    s += [-0.9 + 0.1 * j for j in range(20)]            # -0.9 ... 1.0; s > 0 first at index len(s) + 10
    m["s_history"] = s
    flip = next(j for j in range(15, len(s)) if s[j] > 0)
    t = 0.1 * flip + 4.0                                # 4 s after the side change
    assert p.weight(t, m) == pytest.approx(0.4)
    assert strategies.S4(mode="new", A=10.0).weight(t, mem(s)) == pytest.approx(0.6)
    assert strategies.S4(mode="new", A=10.0).weight(t + 20, mem(s)) == pytest.approx(0.05)
    assert p.weight(t + 20, mem(s)) == 1.0


def test_s6_backoff_doubles_and_resets():
    p = strategies.S6(w_min=0.5, W_max=2.0)
    m = mem([1.0])
    assert p.decide(0.0, 1.0, [0.5], m) == ("step", -1)
    act = p.decide(0.1, 0.9, [0.5, 0.4], m)              # fell at a wall: no help -> W 0.5 -> 1
    assert m["W"] == 1.0 and act[1] in (1.0, -1.0)
    assert 0.5 <= act[2] - (4.0 if act[3] else 0.0) <= 1.0
    for _ in range(3):
        p.decide(0.0, 1.0, [0.5], m)
        p.decide(0.1, 0.9, [0.5, 0.4], m)
    assert m["W"] == 2.0                                 # capped at W_max
    p.decide(0.0, 1.0, [0.5], m)
    act = p.decide(0.1, 0.9, [0.5, 0.6], m)              # helps: greedy flip, W back to w_min
    assert m["W"] == 0.5 and act[:2] == ("target", -1.0) and 0.25 <= act[2] <= 0.5
    assert strategies.S6(rand=False).decide(0.0, 1.0, [1.0], mem([1.0])) == ("target", 1.0, 0.5, False)


def test_s7_fast_player_turns_slow_after_N_failed_probes():
    p = strategies.S7(q=0.0, N=3)
    m = mem([1.0])
    for j in range(3):
        assert not m.get("slow")
        p.decide(0.0, 1.0, [0.5], m)
        assert p.decide(0.1, 0.9, [0.5, 0.4], m) == ("target", 1.0, 0.0, False)   # fast: never yields
    assert m["slow"] and m["n_switch"] == 1 and m["W"] == 0.5
    p.decide(1.0, 1.0, [0.5], m)
    p.decide(1.1, 0.9, [0.5, 0.6], m)                    # a helpful probe: fast again
    assert not m["slow"] and m["slow_time"] == pytest.approx(1.1 - 0.1)


def test_s3_travel_check_reverses_once():
    p = strategies.S3(check=True)
    m = mem([0.0] * 5)
    m.update(state="FOLLOW", pause_end=0.0)
    agr = [0.0, 0.0, 0.14, 0.14, 0.14]
    assert p.decide(0.4, 0.0, agr, m)[0] == "step"
    d = m["probe"][0]
    act = p.decide(0.5, 0.1 * d, agr + [0.12], m)        # fell -> commit to the other wall
    assert act == ("target", float(-d), 4.0, False) and m["state"] == "TRAVEL"     # arrival hold G = 4
    agr += [0.12]
    agr += [0.10]
    assert p.watch(0.6, 0.0, agr, m) is None              # one fall
    agr += [0.08]
    assert p.watch(0.7, -0.1 * d, agr, m) == ("target", float(d), 4.0, False)   # two falls in a row: reverse
    assert m["n_rev"] == 1
    agr += [0.06, 0.04]
    assert p.watch(0.9, 0.0, agr, m) is None              # only once


def test_s3_grace_is_a_hold_after_every_arrival(delays):
    p = strategies.S3(G=4.0)
    m = mem([1.0] * 3)
    m["state"] = "WALL"
    p.decide(1.0, 1.0, [0.2], m)
    assert p.decide(1.1, 0.9, [0.2, 0.3], m) == ("target", -1.0, 4.0, False)    # greedy flip: hold G
    acts = set()
    for _ in range(200):
        p.decide(1.0, 1.0, [0.2], m)
        acts.add(p.decide(1.1, 0.9, [0.2, 0.1], m))
    assert acts == {("target", 1.0, 0.0, False), ("target", -1.0, 4.0, True)}   # step back: no hold; yield max(H, G)
    assert strategies.S3(G=8.0).grace(0.9, ("target", -1.0, 4.0, True)) == ("target", -1.0, 8.0, True)
    # in a run: after every arrival at a wall from elsewhere, the player does not move for G s (no decisions)
    for fallback in ["S0b", "S2"]:
        for r in range(5):
            out = sim.run(game.HUMAN_GRAPHS["Full"], strategies.S3, {"fallback": fallback}, START_A, delays,
                          np.random.default_rng(r), keep_path=True)
            path = out["path"]
            for i in range(7):
                k = path[:, i]
                for n in range(1, len(k)):
                    if abs(k[n]) == game.WALL and abs(k[n - 1]) < game.WALL and abs(k[n - 1] - k[n]) == 1 \
                            and not (n >= 2 and k[n - 2] == k[n]):        # an arrival, not a probe's step back
                        assert np.all(k[n:n + 40] == k[n]), (fallback, r, i, n)


def test_s3_waiting_player_reaches_at_one_seventh():
    p = strategies.S3(assigned_seed=True, pause=0.0)
    m = mem([0.0, 0.0])
    assert p.interrupt(0.1, 0.0, [0.0, 0.13], m) is None and m["state"] == "WAIT"
    m["s_history"].append(0.0)
    assert p.interrupt(0.2, 0.0, [0.0, 0.13, 0.14], m) == ("stay",) and m["state"] == "FOLLOW"


def test_s3b_seeding_is_cancelled_by_others_moves():
    p = strategies.S3(assigned_seed=False, lam=100.0)    # t_seed ~ Exp(mean 0.01 s)
    m = mem([0.0, 0.0])
    assert p.interrupt(0.0, 0.0, [0.0], m) is None or m["seeded"]
    m = mem([0.0, 0.0])
    m["t_seed"] = 0.15
    assert p.interrupt(0.1, 0.0, [0.0, 0.03], m) is None  # Agr changed, own slider still: cancelled
    m["s_history"] += [0.0]
    assert p.interrupt(0.2, 0.0, [0.0, 0.03, 0.03], m) is None and not m.get("seeded")


def test_slip_reverses_directions():
    rng = np.random.default_rng(0)
    assert sim.slipped(("step", 1), 0, rng, 1.0) == ("step", -1)
    assert sim.slipped(("target", 1.0, 0.0, False), 1, rng, 1.0) == ("target", -1.0, 0.0, False)
    assert sim.slipped(("target", -1.0, 4.0, True), 10, rng, 1.0) == ("target", 1.0, 4.0, False)   # stays put
    assert sim.slipped(("target", 1.0, 0.0, False), 9, rng, 1.0) == ("target", 0.8, 0.0, False)    # step back
    assert sim.slipped(("target", 1.0, 0.0, False), 10, rng, 1.0) == ("target", 1.0, 0.0, False)   # no direction
    assert sim.slipped(("step", 1), 0, rng, 0.0) == ("step", 1)


def test_fidgets_only_while_waiting(delays):
    # a player in WAIT never moves by itself, so its moves before it leaves WAIT (t_reach) are exactly the fidgets
    for r in range(5):
        out = sim.run(H, strategies.S3, {}, START_A, delays, np.random.default_rng(r), keep_path=True, fidget=2.0)
        moves = 0
        for i, m in enumerate(out["memory"]):
            n = int(round(m["t_reach"] / game.DT)) if "t_reach" in m else len(out["path"]) - 1
            moves += int(np.sum(np.diff(out["path"][:n + 1, i]) != 0))
        assert out["n_fidgets"] > 0 and moves == out["n_fidgets"]


# ------------------------------------------------------------------ tests 3-4: behaviour (slow)

@pytest.mark.slow
def test_3_s3a_on_hierarchy(delays):
    # S3 as first specified (grace = no yields, greedy flips allowed) failed this test on 2026-11-08: consensus 1.0,
    # 241 fraught-corner entries, 21% of runs without yields. The grace is now a hold (user's decision, same day).
    outs = [sim.run(H, strategies.S3, {"assigned_seed": True}, START_A, delays, np.random.default_rng([3, r]),
                    keep_path=True) for r in range(200)]
    entries = sum(tail.run_metrics(H, o)[0]["fraught_entries"] for o in outs)
    no_yield = np.mean([o["n_yields"] == 0 for o in outs])
    success = np.mean([o["success_300"] for o in outs])
    print("S3a+ Hierarchy: consensus", success, "fraught entries", entries, "runs without yields", no_yield)
    assert success == 1.0
    assert entries == 0
    assert no_yield >= 0.95


@pytest.mark.slow
@pytest.mark.parametrize("cls,params", [(strategies.S4, {"mode": "new"}), (strategies.S6, {}),
                                        (strategies.S7, {"q": 0.0})])
def test_4_escape_from_F0(cls, params, delays):
    outs = [sim.run(H, cls, params, F0 * 10, delays, np.random.default_rng([4, r])) for r in range(200)]
    rate = np.mean([o["success_300"] for o in outs])
    print(cls.name, params, "success_300 from F0:", rate)
    assert rate >= 0.95, rate


# ------------------------------------------------------------------ test 5: information constraint

NEW = [(strategies.S3, {"assigned_seed": True}, {"fidget": 0.3, "slip": 0.05}),
       (strategies.S3, {"assigned_seed": False, "fallback": "S2"}, {"fidget": 0.3, "slip": 0.05}),
       (strategies.S4, {"mode": "old"}, {}), (strategies.S4, {"mode": "new"}, {}),
       (strategies.S6, {}, {}), (strategies.S7, {"q": 0.5}, {})]


@pytest.mark.parametrize("cls,params,noise", NEW)
def test_5_new_strategies_see_only_their_own_inputs(cls, params, noise, delays):
    for method in ["decide", "interrupt", "watch"]:
        if hasattr(cls, method):
            assert list(inspect.signature(getattr(cls, method)).parameters) == ["self", "t", "s_i", "agr_history",
                                                                                "memory"]
    seen = []

    def spy(method):
        def f(self, t, s_i, agr_history, memory):
            seen.append((t, s_i, list(agr_history), set(memory), dict(vars(self))))
            return getattr(cls, method)(self, t, s_i, agr_history, memory)
        return f

    Spy = type("Spy", (cls,), {m: spy(m) for m in ["decide", "interrupt", "watch"] if hasattr(cls, m)})
    for start in [START_A, F0 * 10]:
        sim.run(H, Spy, params, start, delays, np.random.default_rng(1), t_max=40, **noise)
    assert len(seen) > 20
    allowed = set(inspect.signature(cls.__init__).parameters) | {"t0"}
    for t, s_i, hist, keys, attrs in seen:
        assert isinstance(t, float) and isinstance(s_i, float)
        assert len(hist) == round(t / game.DT) + 1 and all(isinstance(a, float) for a in hist)
        assert {"rng", "s_history"} <= keys
        assert keys - {"rng", "s_history"} <= cls.MEMORY_KEYS
        assert set(attrs) <= allowed

"""sim.py: rules and inputs, the reversing script, the log converter, round ends and time accounting.

Human runs are G-train runs from load_runs (default roles), used only as inputs; allow_test is never passed.
"""
import numpy as np
import pandas as pd
import pytest

from slider import config
from slider import fraught
from slider import hazard
from slider import interface
from slider import sim
from slider import slider_logs as sl
from slider import targets
from slider.models.apem import GRAPHS


# ------------------------------------------------------------------ inputs

def test_coefficients_are_the_p11_values():
    expected = {"M2'": {"const": -3.194, "lab": 0.058, "A": -2.107, "abs_s": 1.048, "at_end": 0.445,
                        "e": -0.377, "r": -0.209},
                "M3": {"const": -3.278, "lab": 0.092, "A": -1.736, "trend": -1.680, "abs_s": 0.969,
                       "at_end": 0.500, "e": -0.425, "r": -0.220}}
    for model, beta in expected.items():
        got = sim.Logistic(model).beta
        assert set(got) == set(beta)
        assert all(round(got[k], 3) == v for k, v in beta.items())
    assert sim.Logistic("M3").trend and not sim.Logistic("M2'").trend


def test_rates_are_the_p17_values():
    # lam_stop after F1b (Oct 29: a step onto the wall is not stop exposure); before: 3.04, 1.76, 10.35
    expected = {"lab_v09": (3.26, 1.77, 0.317), "old_cl": (1.80, 1.74, 0.080), "cc": (11.10, 2.55, 0.204)}
    for iface, (stop, resume, begin) in expected.items():
        lam = sim.rates(iface)
        assert (round(lam["lam_stop"], 2), round(lam["lam_resume"], 2), round(lam["lam_start"], 3)) == (stop, resume, begin)


def test_speeds_are_the_modal_file_speeds():
    files = pd.read_csv(config.INTERFACE_TABLE)
    kind = np.where(files["dataset"] == "CC", "cc", np.where(files["version"] == "v09", "lab_v09", "old_cl"))
    for iface, f in sim.IFACES.items():
        assert interface.mode(files.loc[kind == iface, "speed"], 3) == f["speed"]


def test_hazard_conversion():
    assert round(sim.per_tick(0.055, 0.1), 4) == 0.0224


def test_motion_probabilities_are_per_tick_as_motion_py_counts_them():
    lam = sim.rates("lab_v09")
    c = sim.constants(GRAPHS["Full"], "lab_v09", sim.Logistic("M2'"), speed=1.0)
    x = lam["lam_start"] * 0.1
    assert c["p"]["stop"] == pytest.approx(lam["lam_stop"] * 0.1)
    assert c["p"]["resume"] == pytest.approx(lam["lam_resume"] * 0.1)
    assert c["p"]["start"] == pytest.approx(x / (1 + x))


# ------------------------------------------------------------------ hand-set states

def hand_state(iface, rule, motion, script, R=1, n0=0, phi0=1, mode0=sim.MOVING, others=0.0):
    """Player 0 at n0 in mode0 with direction phi0 (already moved); the others halted at s = others, not at risk."""
    c = sim.constants(GRAPHS["Full"], iface, rule, motion, script)
    st = sim.start(R, c, np.random.default_rng(0))
    st["n"][:] = np.round(np.asarray(others) * c["N"]).astype(int)
    st["mode"][:] = sim.HALTED
    st["n"][:, 0], st["mode"][:, 0], st["phi"][:, 0], st["dir"][:, 0] = n0, mode0, phi0, phi0
    k = sim.to_k(st["n"], c["N"])
    st["band"] = np.sign(k) * (np.abs(k) >= 9)
    return c, st


def test_apem_rule():
    rule = sim.APEM(a=0.5, s_th=0.5, mu=0.5)
    p = rule.prob({"A": np.array([0.4, 0.6]), "abs_s": np.array([1.0, 1.0])}, 0.1)
    assert p[0] == pytest.approx(1 - np.exp(-0.1 / 0.5)) and p[1] == 0


@pytest.mark.parametrize("others, flips", [([1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], True),     # A of player 0 = 0.4
                                           ([1.0, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6], False)])   # A = 0.6
def test_apem_player_at_the_wall(others, flips):
    rule, R = sim.APEM(a=0.5, s_th=0.5, mu=0.5), 20000
    c, st = hand_state("lab_v09", rule, "none", False, R=R, n0=10, mode0=sim.AT_WALL, others=others)
    p = 1 - np.exp(-c["dt"] / rule.mu)
    sim.tick(st, 0, c, np.random.default_rng(1))
    left = np.mean(st["n"][:, 0] == 9)                        # flipped and stepped off the wall
    if flips:
        assert abs(left - p) < 4 * np.sqrt(p * (1 - p) / R)
    else:
        for j in range(1, 50):
            sim.tick(st, j, c, np.random.default_rng(j))
        assert left == 0 and np.all(st["n"][:, 0] == 10)


class Once:
    """Fires a reversal of player 0 in the first tick only."""
    trend = False

    def __init__(self):
        self.calls = 0

    def prob(self, f, dt):
        p = np.zeros_like(f["A"])
        p[:, 0] = 1.0 if self.calls == 0 else 0.0
        self.calls += 1
        return p


def reversal(iface, s0, phi0):
    """Player 0 moving at s0 in direction phi0 reverses in the first tick (no stops); tick until the detector
    confirms the reversal or the player is at the wall. Returns (ticks, protected ticks, final n, phi, mode, N)."""
    c, st = hand_state(iface, Once(), "none", True, n0=round(s0 * sim.IFACES[iface]["d"] ** -1), phi0=phi0)
    ticks = protected = 0
    for j in range(100):
        during, _ = sim.tick(st, j, c, np.random.default_rng(j))
        ticks += 1
        protected += during[0, 0] == sim.REVERSING
        if st["dir"][0, 0] == -phi0 or st["mode"][0, 0] == sim.AT_WALL:
            break
    return ticks, protected, st["n"][0, 0], st["phi"][0, 0], st["mode"][0, 0], c["N"]


def test_reversal_lab_is_two_ticks():
    ticks, protected, n, phi, mode, _ = reversal("lab_v09", 0.5, 1)
    assert (ticks, protected, n, phi, mode) == (2, 2, 3, -1, sim.MOVING)


@pytest.mark.parametrize("s0, n_ticks, n_protected", [(0.94, 10, 5), (0.86, 6, 6)])
def test_reversal_old_cl(s0, n_ticks, n_protected):
    """F3: from 0.94 the first 4 steps do not change k (ordinary moving steps), then 5 protected steps; from 0.86
    the first step changes k, so all 6 are protected. Both end at k = 7, moving in the new direction."""
    ticks, protected, n, phi, mode, N = reversal("old_cl", s0, 1)
    assert (ticks, protected, phi, mode) == (n_ticks, n_protected, -1, sim.MOVING)
    assert sim.to_k(np.array([n]), N)[0] == 7                 # k 9 -> 7


def test_reversal_stopped_by_the_wall():
    ticks, protected, n, phi, mode, N = reversal("lab_v09", 0.9, -1)
    assert (ticks, n, phi, mode) == (1, N, 1, sim.AT_WALL)


@pytest.mark.parametrize("iface", ["lab_v09", "old_cl", "cc"])
def test_reversal_bins_match_the_hazard_table(iface):
    """F2: the bins the simulator draws from are hazard.build_hazard_table's bins of the logged round.
    lab v09: identical features (M3, trend included), risk = kept / dropped, fired bins = y_rev bins, n decisions.
    old CL: identical features and risk (reversals are dated at their first step of k, later than the flip).
    CC: the cloud logger's 1-2 ms split can put a lab tick inside one tick's moves, so >= 99% identical."""
    T = config.TIME_LIMIT[sim.IFACES[iface]["session"]]
    for seed in [5, 6]:
        trace = []
        o = sim.simulate(GRAPHS["Hierarchy"], T, iface, R=1, seed=seed, trace=trace, **sim.preset("L1_M3"))[0]
        meta = {"group": f"sim_{sim.IFACES[iface]['tag']}_check.txt", "dataset": "CC" if iface == "cc" else "CL",
                "topology": "Hierarchy", "wave": 1, "graph": GRAPHS["Hierarchy"].astype(int), "seat_node": range(7),
                "interface": iface, "time_limit": T, "end": "success" if o["success"] else "failed", "seed": seed}
        run = sim.to_run(o["events"], meta, "cloud" if iface == "cc" else "lab")
        ht, dropped = hazard.build_hazard_table([run])
        t_end = fraught.round_end(run["t"], run["end"], run["session"])
        bins = pd.concat([d for d in trace if "tau" in d], ignore_index=True)
        bins = bins[bins["tau"] < t_end - 1e-9]                  # the table has no bin at or after the round end
        last = bins.groupby("player")["tau"].max()
        key = lambda d, col: list(zip(d["player"], np.round(d[col], 6)))
        bins["key"], ht["key"], dropped["key"] = key(bins, "tau"), key(ht, "t"), key(dropped, "t")
        ht = ht[ht["t"] <= ht["player"].map(last) + 1e-9]
        dropped = dropped[dropped["t"] <= dropped["player"].map(last) + 1e-9]
        m = bins[bins["risk"]].merge(ht, on="key", suffixes=("_sim", "_ht"))
        same = np.ones(len(m), dtype=bool)
        for f in ["A", "abs_s", "at_end", "time_at_end", "time_since_rev", "trend"]:
            same &= np.isclose(m[f + "_sim"], m[f + "_ht"], atol=1e-9, rtol=0)
        risk_ok = set(bins.loc[bins["risk"], "key"]) == set(ht["key"]) and set(bins.loc[~bins["risk"], "key"]) == set(dropped["key"])
        if iface == "cc":
            assert len(m) >= 0.99 * len(ht) and same.mean() >= 0.99
            continue
        assert risk_ok and len(m) == len(ht) and same.all()
        if iface == "lab_v09":
            fires = pd.concat([d for d in trace if "fire" in d], ignore_index=True)
            fired = {bins.loc[(bins["player"] == f.player) & (bins["tau"] <= f.fire + 1e-6), "key"].iloc[-1]
                     for f in fires.itertuples() if f.fire < t_end - 1e-9}
            assert fired == set(ht.loc[ht["y_rev"] == 1, "key"])
            dt = o["dt"]
            first = bins.groupby("player")["tau"].min()
            for b in bins.itertuples():                           # decisions per bin: steps dated in it, after the first move
                k = np.arange(1, int(T / dt) + 2)
                when = np.round(k * dt, 2)
                inside = (when >= b.tau - 1e-6) & (when < b.tau + 0.25 - 1e-6) & (when > first[b.player] + 1e-6)
                assert b.n == inside.sum()


# ------------------------------------------------------------------ round trip through to_run

@pytest.fixture(scope="module")
def train():
    runs, table = sl.load_runs("behavior")                    # G-train only
    return runs, table


def test_consensus_rule_gives_the_logged_round_ends(train):
    runs, table = train
    assert set(table["dataset"]) == {"EL", "CL", "CC"}
    for run in runs:
        hits = np.flatnonzero(sim.consensus(sl.sign_correct(run)))
        assert list(hits[:1]) == ([len(run["t"]) - 1] if run["end"] == "success" else [])


def pick(runs, table, lab, n=5):
    """n complete bot-free train runs of one session, spread over files, interfaces and topologies."""
    ok = table[(table["session"] == ("lab" if lab else "cloud")) & (table["end"] != "incomplete") & ~table["has_bot"]]
    if lab:
        old = ok["file"].map(interface.version_tag) == "none"
        idx = list(ok[~old].groupby("topology").head(2).index[:3]) + list(ok[old].groupby("topology").head(1).index[:2])
    else:
        idx = list(ok.groupby("topology").head(2).index[:n])
    return [runs[i] for i in idx], table.loc[idx].reset_index(drop=True)


def events_of(run):
    """(time, seat, s): every seat in the first row, then every change."""
    s = sl.sign_correct(run)
    changed = np.r_[np.ones((1, 7), dtype=bool), s[1:] != s[:-1]]
    rows, seats = np.nonzero(changed)
    return pd.DataFrame({"time": run["t"][rows], "seat": seats, "s": s[rows, seats]})


def rebuild(runs, table):
    out = []
    for run, top in zip(runs, table["topology"]):
        iface = "cc" if run["dataset"] == "CC" else "lab_v09" if interface.version_tag(run["file"]) == "v09" else "old_cl"
        meta = {"group": run["file"], "dataset": run["dataset"], "topology": top, "wave": run["wave"],
                "graph": run["A"], "seat_node": range(7), "interface": iface,
                "time_limit": config.TIME_LIMIT[run["session"]], "end": run["end"], "seed": 0}
        out.append(sim.to_run(events_of(run), meta, "cloud" if iface == "cc" else "lab"))
    return out, sim.table(out)


def all_targets(runs, table):
    rate = targets.reversal_rate_table(runs)
    a_sum, a_h = targets.agreement_at_reversal_table(runs)
    c_sum, crossings = targets.crossing_time_table(runs)
    return {"train_targets": targets.train_targets(runs, table, n_boot=500), "reversal_rate": rate,
            "agreement_summary": a_sum, "agreement_h": a_h, "crossing_summary": c_sum, "crossings": crossings}


def test_round_trip_lab_is_identical(train):
    runs, table = pick(*train, lab=True)
    assert set(table["dataset"]) == {"EL", "CL"} and set(table["topology"]) == {"Hierarchy", "Full"}
    new_runs, new_table = rebuild(runs, table)
    for a, b in zip(runs, new_runs):
        assert np.array_equal(a["t"], b["t"]) and np.array_equal(sl.sign_correct(a), sl.sign_correct(b))
        assert np.array_equal(a["A_log"], b["A_log"]) and a["end"] == b["end"] and a["session"] == b["session"]
    cols = ["file", "wave", "dataset", "topology", "success", "end", "duration", "T", "role", "has_bot", "coarse_time"]
    pd.testing.assert_frame_equal(table[cols], new_table[cols])
    old, new = all_targets(runs, table), all_targets(new_runs, new_table)
    for name in old:
        pd.testing.assert_frame_equal(old[name], new[name], check_exact=True, obj=name)


def test_round_trip_cloud_is_close(train):
    runs, table = pick(*train, lab=False)
    new_runs, new_table = rebuild(runs, table)
    for a, b in zip(runs, new_runs):
        s = sl.sign_correct(a)
        keep = np.r_[True, np.any(s[1:] != s[:-1], axis=1)]                     # the first row and every change
        assert len(b["t"]) == keep.sum() and np.abs(b["t"] - a["t"][keep]).max() < 0.01
        assert np.array_equal(b["X"][-1], s[-1])
        assert np.all(np.sum(np.diff(b["X"], axis=0) != 0, axis=1) == 1)          # one mover per row
    # Rows logged in the same millisecond come back 1-2 ms apart, so a lab-clock tick between them can see another
    # state: here one 0.25 s at-risk sample changes Agreement bin, which moves a sparse bin's rate by 0.6%.
    old, new = all_targets(runs, table), all_targets(new_runs, new_table)
    for name in old:
        rtol = 1e-2 if name == "agreement_h" else 1e-4
        pd.testing.assert_frame_equal(old[name], new[name], check_exact=False, rtol=rtol, atol=1e-9, obj=name)


def test_whole_second_logger_floors_times():
    events = pd.DataFrame({"time": [0.0] * 7 + [0.9, 2.95, 3.0], "seat": list(range(7)) + [0, 0, 1],
                           "s": [0.0] * 7 + [0.02, 0.04, -0.02]})
    meta = {"group": "g", "dataset": "CL", "topology": "Full", "wave": 1, "graph": GRAPHS["Full"].astype(int),
            "seat_node": range(7), "interface": "old_cl", "time_limit": 300, "end": "failed"}
    run = sim.to_run(events, meta, "whole_second")
    assert list(run["t"]) == [0, 0, 2, 3] and run["X"][2, 0] == 0.04 and run["X"][3, 1] == -0.02
    assert bool(sim.table([run])["coarse_time"].iloc[0])


# ------------------------------------------------------------------ simulated rounds

@pytest.fixture(scope="module")
def sim_rounds():
    """200 L1 rounds: Hierarchy and Full, lab (v09) and CC interfaces, 50 each; with their to_run runs."""
    out = []
    for k, (iface, top) in enumerate([(i, t) for i in ["lab_v09", "cc"] for t in ["Hierarchy", "Full"]]):
        T = config.TIME_LIMIT[sim.IFACES[iface]["session"]]
        for o in sim.simulate(GRAPHS[top], T, iface, R=50, seed=k, **sim.preset("L1")):
            meta = {"group": f"g{k}", "dataset": "CC" if iface == "cc" else "EL", "topology": top, "wave": 1,
                    "graph": GRAPHS[top].astype(int), "seat_node": range(7), "interface": iface, "time_limit": T,
                    "end": "success" if o["success"] else "failed", "seed": k}
            out.append((o, sim.to_run(o["events"], meta, "cloud" if iface == "cc" else "lab")))
    return out


def test_round_end_matches_fraught(sim_rounds):
    assert len(sim_rounds) == 200
    assert 0 < np.mean([o["success"] for o, _ in sim_rounds]) < 1
    for o, run in sim_rounds:
        t_end = fraught.round_end(run["t"], run["end"], run["session"])
        assert abs(t_end - o["end_time"]) <= o["dt"] + 1e-9
        assert sim.consensus(run["X"][-1]) == o["success"]
        assert np.abs(run["X"]).max() <= 1


def test_time_accounting(sim_rounds):
    for o, _ in sim_rounds:
        ticks = o["mode_ticks"]
        assert np.all(ticks.sum(axis=1) == o["ticks"]) and o["end_time"] == o["ticks"] * o["dt"]
        steps = np.bincount(o["events"]["seat"].iloc[7:], minlength=7)   # after the round-start record
        assert np.array_equal(steps, ticks[:, sim.MOVING] + ticks[:, sim.REVERSING])

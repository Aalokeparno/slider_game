"""P25 final model: the L1b halt mixture (halts.py, sim motion "L1b"), the L3 group effect (frailty.py, sim offset,
design.simulate_rounds) and the preset "final". Exactly the old behavior with one halt type (p = 1) and sigma_g = 0.
Fits are tested on synthetic data (known parameters)."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from slider import design
from slider import frailty
from slider import halts
from slider import motion
from slider import sim
from slider.models.apem import GRAPHS

A_FULL = GRAPHS["Full"].astype(int)


def same(a, b):
    return all(x["events"].equals(y["events"]) and np.array_equal(x["mode_ticks"], y["mode_ticks"]) for x, y in zip(a, b))


# ------------------------------------------------------------------ L1b

@pytest.mark.parametrize("iface", ["lab_v09", "old_cl", "cc"])
def test_one_halt_type_is_exactly_l1(iface):
    rule = sim.Logistic("M2'")
    c = sim.constants(A_FULL, iface, rule)
    q = c["p"]["resume"]                                               # lam_resume dt, as L1 uses it
    l1 = sim.simulate(A_FULL, 60, iface, rule, motion="L1", R=4, seed=5)
    fast = sim.simulate(A_FULL, 60, iface, rule, motion="L1b", R=4, seed=5, halts={"p": 1.0, "q_f": q, "q_s": 0.5})
    slow = sim.simulate(A_FULL, 60, iface, rule, motion="L1b", R=4, seed=5, halts={"p": 0.0, "q_f": 0.9, "q_s": q})
    assert same(l1, fast) and same(l1, slow)


def test_halt_types_are_drawn_with_probability_p():
    """q_f = 1, q_s = 0: a fast halt resumes at the next tick (unless a reversal comes first), a slow one never."""
    out = sim.simulate(A_FULL, 120, "lab_v09", sim.Logistic("M2'"), motion="L1b", R=10, seed=2,
                       halts={"p": 0.3, "q_f": 1.0, "q_s": 0.0})
    runs = []
    for r, o in enumerate(out):
        meta = {"group": f"sim_v09_test_r{r}.txt", "dataset": "CL", "topology": "Full", "wave": 1, "graph": A_FULL,
                "seat_node": range(7), "interface": "lab_v09", "time_limit": 300,
                "end": "success" if o["success"] else "failed", "seed": 2}
        runs.append(sim.to_run(o["events"], {**meta, "time_limit": 300}, "lab"))
    h = halts.add_ticks(motion.motion_table(runs)[1])
    resumed = h[h["event"] == 1]
    assert len(h) > 200
    assert (resumed["n"] == 1).all()                                   # only fast halts resume, after 1 tick
    assert abs(len(resumed) / len(h) - 0.3) < 0.06


def test_mixture_fit_recovers_known_parameters():
    rng = np.random.default_rng(1)
    p, q_f, q_s, size = 0.85, 0.35, 0.03, 40000
    fast = rng.random(size) < p
    n = rng.geometric(np.where(fast, q_f, q_s))                       # resume after n ticks (n >= 1)
    cens = rng.geometric(0.01, size)                                   # censoring after cens ticks
    d = pd.DataFrame({"n": np.minimum(n, cens), "event": (n <= cens).astype(int)})
    r = halts.fit(*halts.counts(d))
    assert r["converged"]
    assert r["p"] == pytest.approx(p, abs=0.02) and r["q_f"] == pytest.approx(q_f, abs=0.02)
    assert r["q_s"] == pytest.approx(q_s, abs=0.005)


def test_one_type_likelihood_gives_the_p17_ratio():
    """With p = 1 the maximum likelihood q is resumes / all halt ticks (P17's estimator)."""
    d = pd.DataFrame({"n": [1, 2, 2, 5, 7, 3], "event": [1, 1, 0, 1, 0, 1]})
    n, e, w = halts.counts(d)
    q_hat = e @ w / (n @ w)
    grid = np.linspace(0.05, 0.6, 2001)
    best = grid[np.argmax([halts.loglik(n, e, w, 1.0, q, q) for q in grid])]
    assert best == pytest.approx(q_hat, abs=1e-3)


# ------------------------------------------------------------------ L3

@pytest.mark.parametrize("iface", ["lab_v09", "cc"])
def test_zero_offset_is_exactly_no_offset(iface):
    rule = sim.Logistic("M2'")
    a = sim.simulate(A_FULL, 60, iface, rule, R=3, seed=8)
    b = sim.simulate(A_FULL, 60, iface, rule, R=3, seed=8, offset=np.zeros(3))
    assert same(a, b)


def test_offset_is_added_to_logit_h():
    rule, u, trace = sim.Logistic("M2'"), np.array([1.0, -1.5]), []
    out = sim.simulate(A_FULL, 60, "lab_v09", rule, R=2, seed=4, offset=u, trace=trace)
    assert [o["u"] for o in out] == [1.0, -1.5]
    b = pd.concat([t for t in trace if "tau" in t], ignore_index=True)
    beta = rule.beta
    z = (beta["const"] + beta["lab"] + beta["A"] * b["A"] + beta["abs_s"] * b["abs_s"] + beta["at_end"] * b["at_end"]
         + beta["e"] * np.where(b["at_end"] == 1, np.log1p(b["time_at_end"]), 0) + beta["r"] * np.log1p(b["time_since_rev"]))
    assert np.allclose(b["h"], expit(z + u[b["rep"]]))


def test_each_group_copy_draws_one_u():
    d3 = design.design("L3")
    small = d3[d3["group"].isin(sorted(d3["group"].unique())[:3])].reset_index(drop=True)
    rounds = design.simulate_rounds(small, "final", R=3, seed=6, sigma_g=0.5)
    u = pd.DataFrame([{"group": row.group, "r": r, "u": out[r]["u"]}
                      for row, (out, _) in zip(small.itertuples(), rounds) for r in range(3)])
    per = u.groupby(["group", "r"])["u"].nunique()
    assert (per == 1).all()                                            # every round of a group copy: the same u
    assert u.drop_duplicates(["group", "r"])["u"].nunique() == 3 * small["group"].nunique()
    zero = design.simulate_rounds(small, "final", R=3, seed=6, sigma_g=0.0)
    assert all(o["u"] == 0.0 for out, _ in zero for o in out)


def test_final_without_l3_keeps_the_round_streams():
    """sigma_g > 0 only adds an offset: the u stream is spawned after the round streams."""
    d3 = design.design("L3")
    small = d3.groupby("interface").head(1).reset_index(drop=True)
    a = design.simulate_rounds(small, "final", R=2, seed=9, sigma_g=0.0)
    b = design.simulate_rounds(small, "final", R=2, seed=9, sigma_g=1e-12)        # u ~ 0: same draws
    for (x, _), (y, _) in zip(a, b):
        assert same(x, y)


def test_sigma_fit_recovers_a_known_sigma():
    rng = np.random.default_rng(3)
    G, n, sigma = 60, 3000, 0.4
    g = np.repeat(np.arange(G), n)
    eta = rng.normal(-3.5, 0.5, G * n)
    y = (rng.random(G * n) < expit(eta + rng.normal(0, sigma, G)[g])).astype(float)
    a, b = frailty.marginal_ll(0.4, eta, y, g, 30), frailty.marginal_ll(0.4, eta, y, g, 60)
    assert a == pytest.approx(b, abs=1e-6)                             # 30 nodes are enough
    x = np.linspace(0.05, 1.0, 96)
    best = x[np.argmax([frailty.marginal_ll(s, eta, y, g) for s in x])]
    assert abs(best - sigma) < 0.15
    assert frailty.marginal_ll(0.0, eta, y, g) == pytest.approx(
        np.sum(y * eta - np.logaddexp(0, eta)))                          # sigma = 0: the offset-only model

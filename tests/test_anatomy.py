"""anatomy.py (Units 6 + 7): coarse-graining, nulls, episodes, the --open guard and the dry run.

Hierarchy in the spec's numbering (1 apex; 1-2, 1-3, 2-4, 2-5, 3-6, 3-7) is the repo's node order shifted by one
(0 apex; landscape.HUMAN_EDGES). Ring / CT human data are never loaded: the --open checks run on temporary files and
load_runs is replaced so that reaching it fails.
"""
import hashlib

import numpy as np
import pandas as pd
import pytest

from slider import anatomy as an
from slider import config
from slider import landscape as L
from slider import slider_logs as sl

HIER = L.adjacency(L.HUMAN_EDGES["Hierarchy"])
CT = L.adjacency(L.HUMAN_EDGES["CT"])
RING = L.adjacency(L.HUMAN_EDGES["Ring"])
TRAP = [1, -1, 1, -1, -1, 1, 1]


# ---------------------------------------------------------------- coarse-graining

def test_forward_hold():
    assert list(an.held_signs(np.array([[-0.1], [0], [0.1]])).ravel()) == [-1, -1, 1]       # one change
    assert list(an.held_signs(np.array([[0.1], [0], [0.1]])).ravel()) == [1, 1, 1]          # none
    assert list(an.held_signs(np.array([[0], [0], [0.2]])).ravel()) == [0, 0, 1]            # undefined until nonzero
    s = np.array([[0.6], [-0.3], [-0.5]])
    assert list(an.held_signs(s, hold=0.5).ravel()) == [1, 1, -1]                           # sensitivity variant


def path_of(rows, t):
    return an.corner_path(np.array(t, float), an.held_signs(np.array(rows, float)))


def test_tie_grouping():
    base = [1, 1, 1, 1, 1, 1, -1]
    a = base.copy(); a[0] = -1
    b = a.copy(); b[1] = -1
    # changes 0.1 s apart: one group of two movers
    _, x0, g = path_of([base, a, b], [0.0, 1.0, 1.1])
    assert len(g) == 1 and g[0]["movers"] == (0, 1) and g[0]["m"] == 2 and g[0]["x"] == an.code(base)
    # 0.2 s apart: two groups
    _, _, g = path_of([base, a, b], [0.0, 1.0, 1.2])
    assert [x["movers"] for x in g] == [(0,), (1,)]
    # a flip and its reversal within 0.1 s cancel
    _, _, g = path_of([base, a, base], [0.0, 1.0, 1.05])
    assert g == []
    # rows before all 7 signs are defined are ignored
    t0, x0, g = path_of([[0] + base[1:], base, a], [0.0, 0.5, 1.0])
    assert t0 == 0.5 and len(g) == 1


# ---------------------------------------------------------------- nulls

def test_hierarchy_kernel():
    ctx = an.context(HIER)
    x = L.state_from_signs(TRAP)
    prog = ctx.cls[x] % 2 == 1
    assert ctx.kernel(np.inf)[x][prog].sum() == pytest.approx(3 / 7)
    P = ctx.kernel(1.0)[x]
    assert ctx.role[0] == "apex"
    assert P[0] == pytest.approx(0.2643, abs=5e-5) and P[prog].sum() == pytest.approx(0.4264, abs=5e-5)


@pytest.mark.parametrize("A", [HIER, CT])
def test_exits_uphill(A):
    ctx = an.context(A)
    exits = [(x, i) for x in L.STATES[ctx.inA] for i in range(L.N) if not ctx.inA[ctx.flips[x, i]]]
    assert exits and all(ctx.dcut[x, i] > 0 for x, i in exits)
    assert "uphill" in an.structural(ctx, 1.0)["C"]


@pytest.mark.parametrize("T", [np.inf, 5.0, 1.0, 0.611009, 0.2])
def test_ring_committor_after_exits(T):
    ctx = an.context(RING)
    q = ctx.q(T)
    for x in L.STATES[ctx.inA]:
        for i in range(L.N):
            y = ctx.flips[x, i]
            if ctx.inA[y]:
                continue
            assert q[y] == (0.0 if ctx.dcut[x, i] > 0 else 1.0)
    assert "degenerate" in an.structural(ctx, T)["C"]


def test_leader_null_sums_to_one():
    ctx = an.context(HIER)
    lead = ctx.leader(1.0)
    rows = lead[ctx.inA]
    assert np.allclose(rows.sum(axis=1), 1) and (lead[~ctx.inA] >= 0).all()


# ---------------------------------------------------------------- episodes on a toy round

def toy_run(states, dt=0.5):
    X = np.array(states, float)
    return {"A": HIER, "weights": np.ones(7), "X": X, "t": np.arange(len(X)) * dt}


def test_round_events_toy():
    c = TRAP
    y1 = c.copy(); y1[3] = 1                     # uphill exit (leaf 3 joins +): cut 2
    z = c.copy(); z[1] = 1                       # uphill exit (middle 1 joins +): cut 2
    z3 = z.copy(); z3[3] = 1
    win = [1] * 7
    moves, attempts, ep, info = an.round_events(toy_run([c, y1, c, z, z3, win]))
    assert ep == {"escaped": True, "spells": 2, "attempts": 2}
    assert [a["outcome"] for a in attempts] == ["aborted", "success"]
    assert attempts[1]["climb"] == 1 and attempts[1]["barrier"] == 1          # climbs exactly B
    assert [m["first"] for m in moves] == [True, True]
    assert moves[1]["prev"] == (3,)


# ---------------------------------------------------------------- the --open guard and the dry run

@pytest.fixture
def no_load(monkeypatch):
    def fail(*args, **kw):
        raise AssertionError("load_runs reached")
    monkeypatch.setattr(sl, "load_runs", fail)


def test_open_refuses_without_hash(no_load):
    assert config.U67_PREREG_SHA is None
    with pytest.raises(SystemExit):
        an.test_pipeline("open")


def test_open_refuses_wrong_hash_or_log_entry(tmp_path, no_load):
    pre = tmp_path / "pre.md"
    pre.write_text("## Predictions\n```yaml\npredictions: []\n```\n")
    sha = hashlib.sha256(pre.read_bytes()).hexdigest()
    log = tmp_path / "log.csv"
    with pytest.raises(SystemExit):
        an.check_open(pre, "0" * 64, log)
    an.check_open(pre, sha, log)                                     # passes: right hash, no log
    with pytest.raises(AssertionError, match="load_runs reached"):
        an.test_pipeline("open", prereg=pre, log=log, sha=sha, cache=tmp_path / "c.pkl")
    pd.DataFrame([{"timestamp": "x", "caller": "scripts/u67_test.py", "reason": an.REASON, "n_test_runs": 1,
                   "kind": "full"}]).to_csv(log, index=False)
    with pytest.raises(SystemExit):
        an.check_open(pre, sha, log)


def test_predictions_and_scoring():
    tables = {"K": pd.DataFrame([{"variant": "main", "topology": "Ring", "cell": "pooled", "T": "T_hat",
                                  "stat": "share up/progress", "part": "all", "obs": 0.5, "exp": 0.3, "diff": 0.2,
                                  "diff_lo": 0.1, "diff_hi": 0.3, "structural": ""}])}
    preds = [{"id": "P1", "measure": "K", "topology": "Ring", "T": "T_hat", "stat": "share up/progress",
              "expect": "positive"},
             {"id": "P2", "measure": "K", "topology": "Ring", "T": "T_hat", "stat": "share up/progress",
              "expect": "no_deviation"},
             {"id": "P3", "measure": "K", "topology": "CT", "T": "inf", "stat": "share up/progress",
              "expect": "deviation"}]
    s = an.score(preds, tables)
    assert list(s["result"]) == ["pass", "fail", "no row"]
    assert an.predictions(config.U67_PREREG) == []                    # the draft's YAML block is empty


def test_dry_run(monkeypatch):
    real = sl.load_runs
    roles_seen = []

    def spy(*args, **kw):
        roles_seen.append(set(kw.get("roles", {"train"})))
        assert not kw.get("allow_test", False)
        return real(*args, **kw)

    monkeypatch.setattr(sl, "load_runs", spy)
    out = an.test_pipeline("dry")
    assert roles_seen == [{"train", "val"}]
    sel = out["selection"].set_index("cell")
    assert list(sel.index) == ["Ring EL", "Ring CL v09", "CT EL"]
    assert sel.loc["Ring EL", "kept"] == 51 and sel.loc["Ring CL v09", "kept"] == 39
    for k in "KMCLO":
        t = out[k]
        assert len(t) > 0 and set(t["T"]) == {"inf", "T_hat"} and set(t["variant"]) == set(an.VARIANTS)
        assert set(t["topology"]) == {"Ring", "CT"}
    assert len(out["scores"]) == 0
    for path in config.U67_DRYRUN.values():
        assert path.exists()

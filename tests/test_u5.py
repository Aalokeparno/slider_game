"""u5.py: Unit 5 scoring, the addendum (A1-A3), dry run and the --open guards. Never touches S-test: the --open checks run on a
temporary git repo and log, and compute_outcomes is replaced so any computation fails the test."""
import subprocess

import numpy as np
import pandas as pd
import pytest

from slider import config, markov
from slider import u5


def test_counts():
    c = u5.counts()
    assert c.loc["train", "trap_graphs"] == 366 and c.loc["test", "trap_graphs"] == 157
    assert c.loc["test", "walled"] == 9


def test_predictors():
    X, split = u5.predictors()
    assert list(X.columns) == u5.PREDICTORS and len(u5.PREDICTORS) == 14
    assert X.notna().all().all() and len(X) == 523
    g = pd.read_csv(config.U1_GRAPHS).set_index("atlas_id")
    a = X.index[0]
    assert X.loc[a, "n_traps"] == pytest.approx(np.log(g.loc[a, "n_traps"]))
    assert X.loc[a, "lam2"] == pytest.approx(g.loc[a, "lam2"])


def test_score_on_toy():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"good": rng.normal(size=200), "bad": rng.normal(size=200)})
    y = 2 * x["good"] + 0.1 * rng.normal(size=200)
    t, boot = u5.score(y, x * 2, x)
    t = t.set_index("predictor")
    assert t.loc["good", "rank"] == 1 and t.loc["good", "R2"] > 0.99 and t.loc["bad", "R2"] < 0
    assert t.loc["bad", "diff_hi"] < 0 and boot.shape == (u5.N_BOOT, 2)


def test_law_on_toy():
    rng = np.random.default_rng(1)
    F = pd.DataFrame({"W_mean": rng.normal(size=300), "B_max": rng.integers(0, 2, 300).astype(float)})
    F["log_dur"] = 1 + 0.5 * F["W_mean"] + 5 * F["B_max"] + 0.01 * rng.normal(size=300)
    f = u5.law_fit(F, 0.2)
    assert f["c_free"] == pytest.approx(5, abs=0.01) and f["b_fixed"] == pytest.approx(0.5, abs=0.01)


def test_dry_run_outputs(monkeypatch):
    seen = []
    real = markov.graph_outcomes

    def spy(atlas_id, split, *args, **kw):
        seen.append((atlas_id, split[atlas_id], args[-2] if len(args) > 3 else kw.get("allow_test", False)))
        return real(atlas_id, split, *args, **kw)

    monkeypatch.setattr(markov, "graph_outcomes", spy)
    rank, law, preds, add = u5.dry_run()
    fit, hold = u5.dry_split()
    assert len(fit) == 256 and len(hold) == 110 and not set(fit) & set(hold)
    assert {s for _, s, _ in seen} == {"train"} and not any(allow for _, _, allow in seen)
    assert len(rank) == 2 * 3 * 14 and set(rank["n_score"]) == {110} and set(rank["n_fit"]) == {256}
    assert len(law) == 3 and list(preds["prediction"]) == ["P1", "P2", "P3", "P4"]
    assert set(preds["result"]) <= {"pass", "fail"}
    for path in [config.U5_DRYRUN_RANKING, config.U5_DRYRUN_LAW, config.U5_DRYRUN_PREDICTIONS,
                 config.U5_DRYRUN_ADDENDUM]:
        assert path.exists()
    # addendum: A1 and A2 per outcome (log entries, log duration) x T, A3 per T
    assert len(add) == 2 * 2 * 3 + 3
    assert add.groupby("hypothesis").size().to_dict() == {"A1": 6, "A2": 6, "A3": 3}
    assert set(add["result"]) <= {"pass", "fail"} and add[["diff_lo", "diff_hi"]].notna().all().all()
    factors = pd.read_csv(config.U5_TRAIN_FACTORS)
    two = factors[factors["model"] == "two_var"].set_index("T")
    a3 = add[add["hypothesis"] == "A3"].set_index("T")
    assert all(a3.loc[T, "train_top"] == f"{two.loc[T, 'x_E']} + {two.loc[T, 'x_D']}" for T in u5.T_U5)


def test_compute_outcomes_guard():
    split = pd.read_csv(config.GRAPH_SPLIT).set_index("atlas")["split"]
    X, _ = u5.predictors()
    test_id = next(a for a in X.index if split[a] == "test")
    with pytest.raises(ValueError):
        u5.compute_outcomes([test_id])
    with pytest.raises(ValueError):                              # allow_test without a reason
        u5.compute_outcomes([test_id], allow_test=True)


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A temporary git repo with a pre-registration (commit h_lock) and an expectation (commit h_exp);
    compute_outcomes is replaced so reaching it fails."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@t")
    git(root, "config", "user.name", "t")
    prereg, exp = root / "prereg.md", root / "exp.md"
    prereg.write_text("locked\n")
    git(root, "add", "prereg.md")
    git(root, "commit", "-qm", "lock")
    h_lock = git(root, "rev-parse", "HEAD")
    exp.write_text("expectation\n")
    git(root, "add", "exp.md")
    git(root, "commit", "-qm", "exp")
    h_exp = git(root, "rev-parse", "HEAD")
    addendum, factors = root / "addendum.md", root / "factors.csv"
    addendum.write_text("addendum\n")
    factors.write_text("a,b\n1,2\n")
    git(root, "add", "addendum.md", "factors.csv")
    git(root, "commit", "-qm", "addendum")
    h_add = git(root, "rev-parse", "HEAD")

    def no_test(*args, **kw):
        raise AssertionError("compute_outcomes reached")

    monkeypatch.setattr(u5, "compute_outcomes", no_test)
    kw = dict(prereg=prereg, h_lock=h_lock, expectation=exp, h_exp=h_exp, addendum=addendum, factors=factors,
              h_add=h_add, log=tmp_path / "log.csv",
              cache=tmp_path / "cache", root=root)
    return kw


def test_open_passes_checks_on_clean_copy(repo):
    with pytest.raises(AssertionError, match="compute_outcomes reached"):      # all checks pass
        u5.open_test(**repo)


def test_open_refuses_changed_prereg(repo):
    repo["prereg"].write_text("edited\n")
    with pytest.raises(SystemExit):
        u5.open_test(**repo)


def test_open_refuses_changed_expectation(repo):
    repo["expectation"].write_text("edited\n")
    git(repo["root"], "commit", "-qam", "edit")
    with pytest.raises(SystemExit):
        u5.open_test(**repo)


def test_open_refuses_changed_addendum(repo):
    repo["addendum"].write_text("edited\n")
    with pytest.raises(SystemExit):
        u5.open_test(**repo)


def test_open_refuses_changed_factors_table(repo):
    repo["factors"].write_text("a,b\n1,3\n")
    git(repo["root"], "commit", "-qam", "edit")
    with pytest.raises(SystemExit):
        u5.open_test(**repo)


def test_open_refuses_wrong_hash(repo):
    repo["h_lock"] = "0" * 40
    with pytest.raises(SystemExit):
        u5.open_test(**repo)


def test_open_refuses_existing_log_entry(repo):
    pd.DataFrame([{"timestamp": "x", "caller": u5.CALLER, "reason": "r", "n_graphs": 157}]).to_csv(
        repo["log"], index=False)
    with pytest.raises(SystemExit):
        u5.open_test(**repo)


def test_no_open_outputs_yet():
    assert not config.U5_TEST_RANKING.exists() and not config.U5_TEST_CACHE.exists()
    assert not config.U5_TEST_ADDENDUM.exists()
    log = config.GRAPH_TEST_ACCESS_LOG
    assert not log.exists() or not pd.read_csv(log)["caller"].astype(str).str.contains("u5_test").any()

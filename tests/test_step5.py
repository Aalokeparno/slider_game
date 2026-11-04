"""Step 5 code paths (scripts/step5_test.py and the library changes it needs), without any test access.

Ring / CT rounds here are SIMULATED: training rounds' design rows with the graph swapped (seat = node)."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from slider import design, stuck
from slider import slider_logs as sl
from slider import visuals as V
from slider.models.apem import GRAPHS

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import fingerprint_step5                                    # noqa: E402
import step5_test as s5                                     # noqa: E402


def test_training_outputs_are_bit_identical():
    assert fingerprint_step5.fingerprints() == json.loads(fingerprint_step5.PATH.read_text())


def test_ct_roles_are_checked_from_the_adjacency():
    V.check_roles("Hierarchy")
    V.check_roles("CT-Hierarchy")
    with pytest.raises(AssertionError):
        V.check_roles("Ring")


@pytest.mark.parametrize("topology", ["Ring", "CT-Hierarchy"])
def test_stuck_types_on_test_topologies(topology):
    A = GRAPHS[topology].astype(int)
    fraught = {"Ring": [1, 1, 1, -1, -1, -1, 1], "CT-Hierarchy": [1, -1, 1, -1, -1, 1, 1]}[topology]
    assert stuck.is_fraught(A, np.array(fraught, dtype=float))
    for corner, kind in [(fraught, "fraught"), ([1, -1, 1, -1, 1, -1, 1], "non-fraught")]:
        k = np.array([[0] * 7, [10 * c for c in corner], [10 * c for c in corner]])
        ep = stuck.round_episodes(np.array([0.0, 1.0, 3.0]), k, A, topology, t_end=3.0)
        assert [e["type"] for e in ep] == [kind]


def test_drop_mask_gives_the_L5_run_set():
    _, all_ = sl.load_runs("behavior", roles={"train", "val"}, **s5.KEEP_ALL)
    kept = all_[~sl.drop_mask(all_, s5.L5)]
    _, l5 = design.load("L5")
    assert list(zip(kept["file"], kept["wave"])) == list(zip(l5["file"], l5["wave"]))


def test_cached_test_runs_never_call_load_runs(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("load_runs was called")
    monkeypatch.setattr(sl, "load_runs", fail)
    cache = tmp_path / "step5_runs.pkl"
    with open(cache, "wb") as f:
        pickle.dump(([{"x": 1}], pd.DataFrame({"a": [1]})), f)
    runs, table = s5.load_test_runs(cache)
    assert runs == [{"x": 1}] and list(table["a"]) == [1]


def test_the_lock_holds_at_head():
    s5.check_lock()


def standin(per_cell=3):
    """Simulated-test design: the first rounds of Hierarchy EL / CL v09 with the Ring or CT graph."""
    d = design.design("L5")
    out = []
    for name, (top, ds) in s5.CELLS["open"].items():
        x = d[(d["topology"] == "Hierarchy") & (d["dataset"] == ds) & (d["interface"] != "old_cl")].head(per_cell)
        x = x.assign(topology=top, graph=[GRAPHS[top].astype(int)] * len(x), seat_node=[tuple(range(7))] * len(x),
                     group=x["group"] + "_" + top)
        out.append(x)
    return pd.concat(out, ignore_index=True)


def test_fast_path_equals_the_full_functions_on_ring_and_ct():
    d = standin()
    runs, table = design.simulate_design(d, "L1", R=1, seed=5)
    cells = s5.CELLS["open"]
    fast = s5.by_cell(s5.point_stats(runs, table, cells), cells).set_index(["cell", "target"])["value"]
    full = s5.by_cell(s5.human_stats(runs, table, cells, n_boot=10), cells).set_index(["cell", "target"])["value"]
    assert len(fast) == len(full) == 27
    full = full.reindex(fast.index)
    assert all(a == b or (np.isnan(a) and np.isnan(b)) for a, b in zip(fast, full))


def test_checks_and_scorecard_on_a_toy_table():
    cells = {"Ring EL": ("Ring", "EL")}
    human = pd.DataFrame({"target": s5.TARGETS, "stat": "s", "topology": "Ring", "dataset": "EL",
                          "value": 10.0, "ci_lo": 8.0, "ci_hi": 12.0, "n_runs": 51}).assign(ci_rule="")
    reps = {m: pd.DataFrame({"target": np.repeat(s5.TARGETS, 2), "stat": "s", "topology": "Ring", "dataset": "EL",
                             "value": v, "replicate": np.tile([0, 1], 9)})
            for m, v in [("C2", 11.0), ("APEM", 13.0), ("final", 9.0)]}
    ch = s5.checks(human, reps, cells)
    card = s5.scorecard(ch).set_index("model")
    assert card.loc["C2", "primary_pass"] == 4 and card.loc["APEM", "primary_pass"] == 0
    assert card.loc["C2", "all_pass"] == card.loc["C2", "all_n"] == 9 and card.loc["C2", "median_abs_gap"] == 0.5
    assert s5.verdict(card.reset_index()).startswith("VERDICT: SUCCESS")


def test_zero_width_rule():
    h = pd.DataFrame({"target": [1, 1, 10, 2, 3], "stat": "s", "topology": "Ring", "dataset": "EL",
                      "value": [100.0, 50.0, 2.0, 300.0, 0.0], "ci_lo": [100.0, 50.0, 2.0, 300.0, 0.0],
                      "ci_hi": [100.0, 50.0, 2.0, 300.0, 0.0], "n_runs": 51})
    out = s5.zero_width_rule(h)
    assert out["ci_rule"].tolist() == ["Clopper-Pearson", s5.NOT_EVALUABLE, "pinned: +/- 0.05 s",
                                       "pinned: +/- 0.05 s", s5.NOT_EVALUABLE]
    assert np.isclose(out["ci_lo"][0], 100 * 0.025 ** (1 / 51)) and out["ci_hi"][2] == 2.05


def test_A2_a_zero_width_ci_without_a_rule_is_not_evaluable_for_every_model():
    """Target 3 (primary) has a zero-width human CI at 5 (no P24 rule): it leaves every count and the median |gap|."""
    cells = {"Ring EL": ("Ring", "EL")}
    lo = [8.0 if t != 3 else 5.0 for t in s5.TARGETS]
    hi = [12.0 if t != 3 else 5.0 for t in s5.TARGETS]
    human = pd.DataFrame({"target": s5.TARGETS, "stat": "s", "topology": "Ring", "dataset": "EL",
                          "value": [10.0 if t != 3 else 5.0 for t in s5.TARGETS], "ci_lo": lo, "ci_hi": hi,
                          "n_runs": 51})
    human = s5.zero_width_rule(human)
    assert human.loc[human["target"] == 3, "ci_rule"].tolist() == [s5.NOT_EVALUABLE]
    reps = {m: pd.DataFrame({"target": np.repeat(s5.TARGETS, 2), "stat": "s", "topology": "Ring", "dataset": "EL",
                             "value": [v if t != 3 else 5.0 for t in np.repeat(s5.TARGETS, 2)],
                             "replicate": np.tile([0, 1], 9)})
            for m, v in [("C2", 11.0), ("APEM", 13.0), ("final", 9.0)]}
    ch = s5.checks(human, reps, cells)
    t3 = ch[ch["target"] == 3]
    assert set(t3["verdict"]) == {s5.NOT_EVALUABLE} and t3["gap"].isna().all() and not t3["borderline"].any()
    card = s5.scorecard(ch).set_index("model")
    assert card["not_evaluable"].tolist() == [1, 1, 1]
    assert card["primary_n"].tolist() == [3, 3, 3] and card["all_n"].tolist() == [8, 8, 8]
    assert card.loc["C2", "primary_pass"] == 3 and card.loc["C2", "all_pass"] == 8
    assert card.loc["APEM", "primary_pass"] == 0 and card.loc["final", "all_pass"] == 8
    assert card.loc["C2", "median_abs_gap"] == 0.5 and card.loc["APEM", "median_abs_gap"] == 1.5
    assert s5.verdict(card.reset_index()) == ("VERDICT: SUCCESS: C2 passes 3 of 3 evaluable primary checks, "
                                              "APEM published 0 of 3.")

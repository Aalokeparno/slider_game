"""targets.check_stats (the fast path for the check statistics) equals the full target functions exactly,
on the human L3 / L5 runs (train + val; values are compared, never shown) and on one simulated replicate."""
import numpy as np
import pandas as pd

from slider import design
from slider import targets


def exactly_equal(out):
    return all((a == b) or (np.isnan(a) and np.isnan(b)) for a, b in zip(out["value"], out["full"]))


def test_fast_path_equals_the_full_functions_on_human_runs():
    runs3, table3 = design.load("L3")
    runs5, _ = design.load("L5")
    out = targets.full_check_values(runs3, table3, runs5, n_boot=10)    # the CIs are not compared
    assert len(out) == 56 and out.groupby("target").size().to_dict() == {t: 2 if t == 5 else 6 for t in range(1, 11)}
    assert out["in_check_list"].sum() == 53                              # target 3 only in the 3 Hierarchy cells
    assert set(out.loc[~out["in_check_list"], "topology"]) == {"Full", "Star"}
    assert exactly_equal(out)


def test_fast_path_equals_the_full_functions_on_a_simulated_replicate():
    d3 = design.design("L3")
    runs3, table3 = design.simulate_design(d3, "L1", R=1, seed=7)
    runs5 = [run for run, whole in zip(runs3, d3["whole_second"]) if not whole]
    out = targets.full_check_values(runs3, table3, runs5, n_boot=10)
    assert len(out) == 56 and exactly_equal(out)


def test_parallel_replicates_match_a_direct_run():
    d3 = design.design("L3")
    small = d3.groupby(["topology", "dataset", "whole_second"]).head(2).reset_index(drop=True)
    stats, _ = design.check_replicates(small, "L1", n_rep=3, seed=11, workers=2, batch=2)
    assert sorted(stats["replicate"].unique()) == [0, 1, 2]
    direct = design.check_job((small, "L1", 1, [11, 1], 2, {}))   # batch 1 holds replicate 2 alone
    pd.testing.assert_frame_equal(stats[stats["replicate"] == 2].reset_index(drop=True), direct)

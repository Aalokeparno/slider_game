"""targets.check_stats (the fast path for the check statistics) equals the full target functions exactly,
on the human L3 / L5 runs (train + val; values are compared, never shown) and on one simulated replicate.
That includes the P24 additions: 10c (pause share), target 8 per group, and target 2 = the time limit when
there is no KM median."""
import numpy as np
import pandas as pd

from slider import config
from slider import design
from slider import targets


def exactly_equal(out):
    return all((a == b) or (np.isnan(a) and np.isnan(b)) for a, b in zip(out["value"], out["full"]))


def check_shape(out, runs5):
    cells = out[out["group"] == ""]
    assert len(cells) == 62                                              # 56 + 10c in 6 cells
    assert cells.groupby("target").size().to_dict() == {t: 2 if t == 5 else 12 if t == 10 else 6 for t in range(1, 11)}
    assert out["in_check_list"].sum() == 53                              # target 3 only in the 3 Hierarchy cells
    assert not out.loc[out["stat"].isin([targets.PAUSE_SHARE, targets.GROUP_RATE]), "in_check_list"].any()
    groups = out[out["stat"] == targets.GROUP_RATE]
    assert set(groups["group"]) == {run["file"] for run in runs5}
    assert not groups.duplicated(["topology", "dataset", "group"]).any()


def test_fast_path_equals_the_full_functions_on_human_runs():
    runs3, table3 = design.load("L3")
    runs5, _ = design.load("L5")
    out = targets.full_check_values(runs3, table3, runs5, n_boot=10)    # the CIs are not compared
    check_shape(out, runs5)
    assert set(out.loc[(out["target"] == 3) & ~out["in_check_list"], "topology"]) == {"Full", "Star"}
    assert exactly_equal(out)


def test_fast_path_equals_the_full_functions_on_a_simulated_replicate():
    d3 = design.design("L3")
    runs3, table3 = design.simulate_design(d3, "L1", R=1, seed=7)
    runs5 = [run for run, whole in zip(runs3, d3["whole_second"]) if not whole]
    out = targets.full_check_values(runs3, table3, runs5, n_boot=10)
    check_shape(out, runs5)
    assert exactly_equal(out)


def test_target2_is_the_time_limit_without_a_median():
    sessions = pd.Series(["lab", "lab"])
    assert targets.median_or_limit(np.nan, sessions) == config.TIME_LIMIT["lab"]
    assert targets.median_or_limit(42.0, sessions) == 42.0
    d3 = design.design("L3")
    small = d3[(d3["topology"] == "Star") & (d3["dataset"] == "CC")].head(3).reset_index(drop=True)
    runs, table = design.simulate_design(small, "L1", R=1, seed=3)
    table["success"] = False                                            # no KM median
    out = targets.check_stats(runs, table, runs)
    assert out.loc[out["target"] == 2, "value"].tolist() == [config.TIME_LIMIT["cloud"]]


def test_between_group_sd():
    assert targets.between_group_sd([1.0, 2.0, 3.0]) == 1.0
    lo, hi = targets.between_group_sd_ci([1.0, 2.0, 3.0, 4.0], n_boot=200)
    assert 0 <= lo <= hi


def test_a_batch_equals_its_copies_from_simulate_design():
    d3 = design.design("L3")
    small = d3.groupby(["topology", "dataset", "whole_second"]).head(2).reset_index(drop=True)
    batch = design.check_job((small, "L1", 2, [5], 10, {}))
    runs, table = design.simulate_design(small, "L1", R=2, seed=[5])
    keep5 = ~small["whole_second"].to_numpy()
    for r in range(2):                                                  # copy r of every round: runs[r::2]
        runs3, table3 = runs[r::2], table.iloc[r::2].reset_index(drop=True)
        runs5 = [run for run, keep in zip(runs3, keep5) if keep]
        direct = targets.check_stats(runs3, table3, runs5).assign(replicate=10 + r)
        pd.testing.assert_frame_equal(batch[batch["replicate"] == 10 + r].reset_index(drop=True), direct)


def test_parallel_replicates_match_a_direct_run():
    d3 = design.design("L3")
    small = d3.groupby(["topology", "dataset", "whole_second"]).head(2).reset_index(drop=True)
    stats, _ = design.check_replicates(small, "L1", n_rep=3, seed=11, workers=2, batch=2)
    assert sorted(stats["replicate"].unique()) == [0, 1, 2]
    direct = design.check_job((small, "L1", 1, [11, 1], 2, {}))   # batch 1 holds replicate 2 alone
    pd.testing.assert_frame_equal(stats[stats["replicate"] == 2].reset_index(drop=True), direct)

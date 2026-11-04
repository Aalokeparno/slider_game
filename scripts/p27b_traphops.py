"""P27b: trap-hop statistics (traps.trap_hop_table) on the human L5 rounds (design.SETS["L5"]: train + val roles),
Hierarchy cells, v09 + cloud scope: Hierarchy EL, Hierarchy CL (v09 files only), Hierarchy CC.
95% CIs: group bootstrap, 1,000 resamples, seed 573.

Simulated runs are not included: the supervisor-comparison trajectory files (results/tables/sup_traj_<model>.pkl)
hold summaries and one example round per CL cell, not full runs.

    python scripts/p27b_traphops.py      -> results/tables/p27_traphops_train.csv
"""
from slider import config, design, interface, traps

runs, table = design.load("L5")
keep = [top == "Hierarchy" and (run["dataset"] != "CL" or interface.version_tag(run["file"]) == "v09")
        for run, top in zip(runs, table["topology"])]
runs = [run for run, k in zip(runs, keep) if k]
tab = traps.trap_hop_table(runs, n_boot=1000, seed=config.SEED)
tab.insert(0, "scope", "v09 + cloud (CL = v09 files)")
tab.insert(0, "source", "human L5")
tab.to_csv(config.P27_TRAPHOPS, index=False)
print(tab.drop(columns=["source", "scope"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

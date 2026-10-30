"""Path A P24, descriptive only: for every L1 statistic in results/tables/p24_checks.csv, the sim mean, the human
value and 95% CI, the direction (high = sim mean above the human value, low = below) and the gap
(sim mean - human value) in CI half-widths, hw = (ci_hi - ci_lo) / 2. Writes results/tables/p24_directions.csv.

    python scripts/p24_directions.py
"""
import numpy as np
import pandas as pd

from slider import config

c = pd.read_csv(config.P24_CHECKS)
c = c[c["preset"] == "L1"]
out = c[["target", "stat", "topology", "dataset", "in_check_list", "sim_mean", "human", "ci_lo", "ci_hi", "ci_rule",
         "verdict"]].copy()
out["ci_rule"] = out["ci_rule"].fillna("")
out["direction"] = np.where(out["sim_mean"] > out["human"], "high", np.where(out["sim_mean"] < out["human"], "low", "equal"))
out["gap_hw"] = (out["sim_mean"] - out["human"]) / ((out["ci_hi"] - out["ci_lo"]) / 2)
out.to_csv(config.P24_DIRECTIONS, index=False)
with pd.option_context("display.width", 250, "display.max_rows", 200, "display.float_format", "{:.4g}".format):
    print(out.drop(columns="in_check_list").to_string(index=False))

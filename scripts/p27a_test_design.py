"""P27a: design of the test rounds (slider_logs.test_design: settings only, no outcomes), counts per cell.

    python scripts/p27a_test_design.py      (run once; each run adds a line to results/test_access_log.csv)
"""
from slider import config
from slider import slider_logs as sl

d = sl.test_design("P27a: count test rounds per cell, v09 scope")
d.to_csv(config.P27_TEST_DESIGN, index=False)
counts = d.groupby(["topology", "dataset", "version"]).agg(rounds=("wave", "size"), groups=("group_id", "nunique"))
print(counts.to_string())

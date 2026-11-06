"""Path 3 fix round, step 4: tests 2 and 3 rerun with diagnostics (same seeds as tests/test_path3.py).

Test 2: S0b'' and S2 (v2) from F0 on Hierarchy, 200 runs each (seeds 2000 + r).
Test 3: S1p (v2), kappa = 1.05, all 126 non-consensus corners x 20 runs (seeds 3000 + 20 c + r).
Writes results/path3/fix_tests.csv (one row per test) and results/path3/s1p_probe_log.parquet (every probe).
"""
import numpy as np
import pandas as pd

from slider import config
from slider.path3 import game, human, sim, strategies

H = game.HUMAN_GRAPHS["Hierarchy"]
F0 = game.corner_from_sets(plus={1, 3, 6, 7}, minus={2, 4, 5})
DELAYS = human.load_delays()


def median_t(outs):
    return float(np.median([o["t_consensus"] if o["success_300"] else np.inf for o in outs]))


rows = []
for cls in [strategies.S0b, strategies.S2]:
    outs = [sim.run(H, cls, {}, F0 * 10, DELAYS, np.random.default_rng(2000 + r), keep_path=True) for r in range(200)]
    rate = np.mean([o["success_300"] for o in outs])
    row = {"test": "2", "strategy": cls.name, "runs": len(outs), "success_300": rate, "pass": rate >= 0.95,
           "median_t": median_t(outs)}
    if cls is strategies.S2:
        idle = sum(m.get("idle_ticks", 0) for o in outs for m in o["memory"])
        stuck = sum(m.get("stuck_ticks", 0) for o in outs for m in o["memory"])
        row["stuck_share_of_idle_ticks"] = stuck / idle
    rows.append(row)
    print(row, flush=True)

outs, probes = [], []
for c, corner in enumerate(game.non_consensus_corners()):
    for r in range(20):
        o = sim.run(H, strategies.S1p, {"kappa": 1.05}, corner * 10, DELAYS, np.random.default_rng(3000 + 20 * c + r),
                    keep_path=True)
        outs.append(o)
        probes.append(sim.probe_table(H, o["path"], o["memory"]).assign(corner=c, run=r))
probes = pd.concat(probes, ignore_index=True)
probes.to_parquet(config.PATH3 / "s1p_probe_log.parquet", index=False)
signed = probes[probes["true_sign"] != 0]
right = signed["sigma"] == signed["true_sign"]
rate = np.mean([o["success_300"] for o in outs])
row = {"test": "3", "strategy": "S1p", "runs": len(outs), "success_300": rate, "pass": rate >= 0.95,
       "median_t": median_t(outs), "probes": len(probes), "share_sum_zero": 1 - len(signed) / len(probes),
       "q": right.mean(), "q_nbr_moved": right[signed["nbr_moved"]].mean(),
       "q_nbr_still": right[~signed["nbr_moved"]].mean(), "share_nbr_moved": signed["nbr_moved"].mean(),
       "median_wait_s": probes["wait_s"].median(), "mean_wait_s": probes["wait_s"].mean()}
rows.append(row)
print(row, flush=True)
pd.DataFrame(rows).to_csv(config.PATH3 / "fix_tests.csv", index=False)

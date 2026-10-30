"""Path A P24 pilot: how many replicates R the main runs need.

L1 on the L3 / L5 design, 10 replicates, seed 573. For each of the 53 check statistics: SD over the replicates
and the human CI half-width hw = (ci_hi - ci_lo) / 2 (targets.human_check_values). R = the largest
ceil((SD / (0.1 hw))^2) over the 53 (so SE_mc = SD / sqrt(R) <= 0.1 hw), capped at 200.
Writes results/tables/p24_pilot.csv (no simulated means and no human point values).

    python scripts/p24_pilot.py
"""
import warnings

import numpy as np

from slider import config
from slider import design
from slider import targets

warnings.filterwarnings("ignore", message="Pandas requires version")
N_REP, SEED, CAP = 10, config.SEED, 200
KEY = ["target", "stat", "topology", "dataset"]

stats, per_rep = design.check_replicates(design.design("L3"), "L1", N_REP, seed=SEED, workers=2, batch=5)
sd = stats[stats["in_check_list"]].groupby(KEY)["value"].agg(sim_sd="std", n="count").reset_index()
human = targets.human_check_values()
human = human[human["in_check_list"]].assign(hw=lambda h: (h["ci_hi"] - h["ci_lo"]) / 2)[KEY + ["hw"]]
pilot = sd.merge(human, on=KEY, validate="one_to_one")
assert len(pilot) == 53
with np.errstate(divide="ignore", invalid="ignore"):          # hw = 0: inf if SD > 0, NaN if SD = 0 too
    pilot["R_needed"] = np.ceil((pilot["sim_sd"] / (0.1 * pilot["hw"])) ** 2)
pilot.to_csv(config.P24_PILOT, index=False)

top = pilot.loc[pilot["R_needed"].fillna(-1).idxmax()]
R = int(min(top["R_needed"], CAP))
zero = pilot[pilot["hw"] == 0]
for z in zero.itertuples():
    print(f"zero-width human CI: target {z.target} {z.stat} {z.topology} {z.dataset}; pilot SD {z.sim_sd:.4g}, "
          f"R_needed {z.R_needed}")
if len(zero):
    rest = pilot[pilot["hw"] > 0]
    other = rest.loc[rest["R_needed"].idxmax()]
    print(f"without the zero-width statistics: R = {int(min(other['R_needed'], CAP))}, set by target {other['target']} "
          f"{other['stat']} {other['topology']} {other['dataset']}")
print(f"pilot: {N_REP} replicates, {per_rep:.1f} s wall per replicate (2 processes, batch 5)")
print(f"replicates with a NaN value: {int((pilot['n'] < N_REP).sum())} statistics")
print(f"R = {R} (uncapped {top['R_needed']:.0f}), set by target {top['target']} {top['stat']} {top['topology']} "
      f"{top['dataset']}: SD {top['sim_sd']:.4g}, hw {top['hw']:.4g}")
print("statistics needing more than 100 replicates:", int((pilot["R_needed"] > 100).sum()))
print(pilot.sort_values("R_needed", ascending=False).head(8)[KEY + ["sim_sd", "hw", "R_needed"]].to_string(index=False))

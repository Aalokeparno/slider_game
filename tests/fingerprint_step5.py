"""Fingerprints of the training outputs that Step 5 code touches (sha256 of their CSV text), for test_step5.py:
they must stay bit-identical. Made with the code before the Step 5 test script (commit f547677):
load_runs tables (L3, L5), design.design("L3"), train_targets (L3), targets 8-10 full tables (L5), stuck episodes
(L5) and the Fig 5 trajectory sums (L3, Hierarchy and Full CL). Bootstrap tables use n_boot = 50 (speed).

    python tests/fingerprint_step5.py      (writes tests/data/step5_fingerprints.json)
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from slider import design, stuck, targets
from slider import visuals as V

PATH = Path(__file__).parent / "data" / "step5_fingerprints.json"
N_BOOT = 50


def sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p.to_csv(index=False).encode() if hasattr(p, "to_csv") else np.asarray(p, dtype=float).tobytes())
    return h.hexdigest()


def fingerprints():
    runs3, table3 = design.load("L3")
    runs5, table5 = design.load("L5")
    d3 = design.design("L3")
    d5 = design.design("L5")
    ag = targets.agreement_at_reversal_table(runs5, n_boot=N_BOOT)
    traj = V.trajectories(runs3, table3, [("Hierarchy", "CL"), ("Full", "CL")])
    return {
        "load_runs": sha(table3, table5),
        "design_L3": sha(d3.drop(columns=["graph"]), np.stack(d3["graph"].to_list())),
        "train_targets": sha(targets.train_targets(runs3, table3, n_boot=N_BOOT)),
        "target8": sha(targets.reversal_rate_table(runs5, n_boot=N_BOOT)),
        "target9": sha(*ag),
        "target10": sha(*targets.crossing_time_table(runs5, n_boot=N_BOOT)),
        "stuck": sha(stuck.episode_table(runs5, table5, d5["interface"])),
        "traj": sha(*[x for cell in sorted(traj) for x in traj[cell][:2]], [traj[c][2] for c in sorted(traj)]),
    }


if __name__ == "__main__":
    prints = fingerprints()
    PATH.write_text(json.dumps(prints, indent=1) + "\n")
    print(prints)

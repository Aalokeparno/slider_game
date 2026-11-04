"""P27b: trap-hop reference run for two frozen models, C2 (EXPLORATORY) and APEM (published a = s = 0.5, mu = 0.1),
on the Hierarchy training cells (descriptive, no model change).

Design: design.design("L5") (all-human copy), Hierarchy rounds, v09 + cloud (old CL dropped, as sup_runs.scope):
Hierarchy EL, CL v09, CC. 20 replicates per model, design.simulate_rounds(R = 20, seed = SEEDS[model]) and
design.copy_runs. Trap hops: traps.trap_hop_table on the 20 replicates pooled, the code of p27_traphops_train.csv.
Human rows (with their 95% CIs) are copied from results/tables/p27_traphops_train.csv; model rows have no CI.

Sanity check: the check statistics of these replicates are compared with stored replicates of the same model and
seed, if any exist (results/tables/*.parquet with a seed column); none hold 9104 or 9105, so it is skipped.

    python scripts/p27b_traphops_ref.py      -> results/tables/p27_traphops_ref.csv
"""
import glob
import time

import numpy as np
import pandas as pd

from slider import config, design, traps
from slider import visuals as V

SEEDS = {"C2": 9104, "APEM": 9105}       # registered (P27b); free in sup_runs.py, p24_main.py, p24_noOldCL.py, p26
MODELS = {"C2": ("C2", {}), "APEM": ("APEM", V.APEM_PUBLISHED)}
R = 20
CELLS = ["EL", "CL", "CC"]
STATS = {"n_spells": "n_spells", "n_distinct": "n_distinct", "n_hops": "n_hops", "any_hop": "pct_rounds_with_hop"}


def ref_design():
    d = design.design("L5")
    return d[(d["topology"] == "Hierarchy") & (d["interface"] != "old_cl")].reset_index(drop=True)


def wide(tab, who):
    """trap_hop_table output -> one row per cell: cell, who, rounds, the four stats (+ CIs), spells_cut."""
    out = []
    for dataset in CELLS:
        c = tab[tab["dataset"] == dataset].set_index("stat")
        row = {"cell": f"Hierarchy {dataset}" + (" v09" if dataset == "CL" else ""), "who": who,
               "rounds": int(c["n_runs"].iloc[0]), "spells_cut": int(c["n_cut"].iloc[0])}
        for stat, name in STATS.items():
            row[name] = c.loc[stat, "value"]
            row[f"{name}_ci_lo"] = c.loc[stat, "ci_lo"] if who == "humans" else np.nan
            row[f"{name}_ci_hi"] = c.loc[stat, "ci_hi"] if who == "humans" else np.nan
        out.append(row)
    return out


def stored_seeds():
    """Seeds of the stored replicate files."""
    seeds = set()
    for f in glob.glob(str(config.TABLES / "*.parquet")):
        t = pd.read_parquet(f)
        if "seed" in t.columns:
            seeds |= set(t["seed"].astype(str))
    return seeds


if __name__ == "__main__":
    d = ref_design()
    print("design rounds per cell:", d.groupby("dataset").size().to_dict())
    human = pd.read_csv(config.P27_TRAPHOPS)
    assert d.groupby("dataset").size().to_dict() == human.groupby("dataset")["n_runs"].first().to_dict()
    rows = wide(human, "humans")
    for model, (preset, kw) in MODELS.items():
        start = time.time()
        rounds = design.simulate_rounds(d, preset, R=R, seed=SEEDS[model], **kw)
        runs = [run for r in range(R) for run in design.copy_runs(d, rounds, r, SEEDS[model])]
        del rounds
        rows += wide(traps.trap_hop_table(runs), model)
        print(f"{model}: {R} replicates x {len(d)} rounds, seed {SEEDS[model]}, {time.time() - start:.0f} s", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(config.P27_TRAPHOPS_REF, index=False)
    print(out[["cell", "who", "rounds"] + list(STATS.values()) + ["spells_cut"]].to_string(index=False,
                                                                                         float_format="%.3f"))
    taken = stored_seeds() & {str(s) for s in SEEDS.values()}
    print("sanity check:", f"stored replicates with seed {sorted(taken)}" if taken else
          "skipped, no stored replicates with seed 9104 or 9105")

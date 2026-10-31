"""Path A P24, Step 4 main runs: the check statistics of R simulated replicates of the L3 design (targets 8-10 on
each replicate's L5 rounds) for one preset, saved per replicate to results/tables/p24_reps_<preset>.parquet:
the 53 check statistics, 10c and the per-group reversal rates (targets.check_stats), with preset, seed and
replicate columns.

Resumable: replicates already in the file are skipped. Replicates go in batches of design.BATCH (20): batch b
holds replicates 20 b ... 20 b + 19, and one job simulates the batch's missing replicates together with seed
[seed, b, first missing replicate]. The file is rewritten (to a temporary file, then renamed) after every job.

Pre-registered seeds: L1 1001, L1_M3 2001, L0 3001, final 4001 (P25: L1 + L1b + L3).
Registered seeds of the EXPLORATORY presets: B1 6001 (results/model_b_log.md, entry 1), C1 7001 (entry 2).

    python scripts/p24_main.py --preset L1 --R 200 --seed 1001 --procs 2
"""
import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from slider import config
from slider import design
from slider import targets

warnings.filterwarnings("ignore", message="Pandas requires version")
SEEDS = {"L1": 1001, "L1_M3": 2001, "L0": 3001, "final": 4001, "B1": 6001, "C1": 7001}      # B1, C1: EXPLORATORY


def keep(stats):
    """The rows saved per replicate: the 53 check statistics, 10c and the per-group reversal rates."""
    return stats[stats["in_check_list"] | stats["stat"].isin([targets.PAUSE_SHARE, targets.GROUP_RATE])]


def main():
    ap = argparse.ArgumentParser(description="Path A P24 main runs (resumable)")
    ap.add_argument("--preset", required=True, choices=list(SEEDS))
    ap.add_argument("--R", type=int, required=True, help="number of replicates")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    if a.seed != SEEDS[a.preset]:
        raise SystemExit(f"the pre-registered seed of {a.preset} is {SEEDS[a.preset]}, not {a.seed}")

    path = config.P24_REPS[a.preset]
    saved = pd.read_parquet(path) if path.exists() else None
    if saved is not None and set(saved["seed"]) != {a.seed}:
        raise SystemExit(f"{path} holds seed {sorted(set(saved['seed']))}, not {a.seed}")
    have = set() if saved is None else set(saved["replicate"])

    d3 = design.design("L3")
    jobs = []
    for b, first in enumerate(range(0, a.R, design.BATCH)):
        missing = [r for r in range(first, min(first + design.BATCH, a.R)) if r not in have]
        if missing:
            assert missing == list(range(missing[0], missing[-1] + 1)), "saved replicates are a prefix of each batch"
            jobs.append((d3, a.preset, len(missing), [a.seed, b, missing[0]], missing[0], {}))
    todo = sum(job[2] for job in jobs)
    print(f"{a.preset}: R = {a.R}, {len(have & set(range(a.R)))} saved, {todo} to run in {len(jobs)} jobs, "
          f"{a.procs} processes -> {path}", flush=True)

    start, ran = time.time(), 0
    with ProcessPoolExecutor(a.procs, max_tasks_per_child=1) as pool:
        for future in as_completed([pool.submit(design.check_job, job) for job in jobs]):
            new = keep(future.result()).assign(preset=a.preset, seed=a.seed)
            saved = new if saved is None else pd.concat([saved, new], ignore_index=True)
            tmp = path.with_suffix(".tmp")
            saved.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            ran += new["replicate"].nunique()
            per_rep = (time.time() - start) / ran
            print(f"{time.strftime('%H:%M:%S')}  {ran}/{todo} done, {per_rep:.1f} s per replicate, "
                  f"about {per_rep * (todo - ran) / 60:.0f} min left", flush=True)
    print(f"{a.preset}: {saved['replicate'].nunique() if saved is not None else 0} replicates in {path}")


if __name__ == "__main__":
    main()

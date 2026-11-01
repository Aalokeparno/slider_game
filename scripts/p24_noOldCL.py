"""Track C scope (results/model_b_log.md entry 4): the P24 check statistics of R simulated replicates of the L3 design
restricted to the CL v09 rounds (Full CL and Hierarchy CL without old CL; targets 8-10 on their L5 part), for
L1, final or C2, saved per replicate to results/tables/p24_reps_noOldCL_<preset>.parquet.

Same batches, job seeds and saved rows as p24_main.py ([seed, batch, first missing replicate]; resumable).
Registered seeds (entry 4): L1 1101, final 4101, C2 8101.

    python scripts/p24_noOldCL.py --preset C2 --R 200 --seed 8101 --procs 2
"""
import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from slider import config
from slider import design

sys.path.insert(0, os.path.dirname(__file__))
from p24_main import keep                                   # noqa: E402  the rows saved per replicate

warnings.filterwarnings("ignore", message="Pandas requires version")
SEEDS = {"L1": 1101, "final": 4101, "C2": 8101}             # entry 4


def path(preset):
    return config.TABLES / f"p24_reps_noOldCL_{preset}.parquet"


def v09_cl(d3):
    """The CL v09 rounds of the L3 design."""
    return d3[(d3["dataset"] == "CL") & (d3["interface"] == "lab_v09")].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="P24 check statistics on the CL v09 rounds (resumable)")
    ap.add_argument("--preset", required=True, choices=list(SEEDS))
    ap.add_argument("--R", type=int, required=True, help="number of replicates")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    if a.seed != SEEDS[a.preset]:
        raise SystemExit(f"the registered seed of {a.preset} is {SEEDS[a.preset]}, not {a.seed}")

    out = path(a.preset)
    saved = pd.read_parquet(out) if out.exists() else None
    if saved is not None and set(saved["seed"]) != {a.seed}:
        raise SystemExit(f"{out} holds seed {sorted(set(saved['seed']))}, not {a.seed}")
    have = set() if saved is None else set(saved["replicate"])

    d = v09_cl(design.design("L3"))
    jobs = []
    for b, first in enumerate(range(0, a.R, design.BATCH)):
        missing = [r for r in range(first, min(first + design.BATCH, a.R)) if r not in have]
        if missing:
            assert missing == list(range(missing[0], missing[-1] + 1)), "saved replicates are a prefix of each batch"
            jobs.append((d, a.preset, len(missing), [a.seed, b, missing[0]], missing[0], {}))
    todo = sum(job[2] for job in jobs)
    print(f"{a.preset}: {len(d)} CL v09 rounds, R = {a.R}, {len(have & set(range(a.R)))} saved, {todo} to run in "
          f"{len(jobs)} jobs, {a.procs} processes -> {out}", flush=True)

    start, ran = time.time(), 0
    with ProcessPoolExecutor(a.procs, max_tasks_per_child=1) as pool:
        for future in as_completed([pool.submit(design.check_job, job) for job in jobs]):
            new = keep(future.result()).assign(preset=a.preset, seed=a.seed)
            saved = new if saved is None else pd.concat([saved, new], ignore_index=True)
            tmp = out.with_suffix(".tmp")
            saved.to_parquet(tmp, index=False)
            os.replace(tmp, out)
            ran += new["replicate"].nunique()
            per_rep = (time.time() - start) / ran
            print(f"{time.strftime('%H:%M:%S')}  {ran}/{todo} done, {per_rep:.1f} s per replicate, "
                  f"about {per_rep * (todo - ran) / 60:.0f} min left", flush=True)
    print(f"{a.preset}: {saved['replicate'].nunique() if saved is not None else 0} replicates in {out}")


if __name__ == "__main__":
    main()

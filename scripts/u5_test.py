"""Unit 5 test (docs/u5_preregistration.md).

    --dry-run   No S-test access. Stand-in: S-train trap graphs split 70 / 30 (seed 573), fit on 70%, score on
                30%; writes results/tables/u5_dryrun_ranking.csv, u5_dryrun_law.csv, u5_dryrun_predictions.csv.
    --open      The one S-test scoring (run once, after H_lock and H_exp are committed). Refuses unless the
                pre-registration and expectation files are unchanged since U5_H_LOCK / U5_H_EXP and
                results/graph_test_access_log.csv has no u5 entry. Computes S-test outcomes via markov.py
                (allow_test with a reason), logs the access, caches outcomes in results/u5_test_cache/, and writes
                results/tables/u5_test_ranking.csv, u5_test_law.csv, u5_test_predictions.csv.
                Addendum (docs/u5_addendum.md, U5_H_ADD): --open also requires the addendum and
                u5_train_factors.csv unchanged since U5_H_ADD, and then writes u5_test_addendum.csv (A1-A3);
                --dry-run writes u5_dryrun_addendum.csv.
"""
import argparse
import time

from slider import u5


def main():
    ap = argparse.ArgumentParser(description="Unit 5 test (docs/u5_preregistration.md)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="S-train 70 / 30 stand-in, no S-test access")
    g.add_argument("--open", action="store_true", help="the one-time S-test scoring")
    a = ap.parse_args()
    start = time.time()
    rank, law, preds, add = u5.open_test() if a.open else u5.dry_run()
    print(preds.to_string(index=False))
    print(add[["hypothesis", "outcome", "T", "train_top", "test_top", "result"]].to_string(index=False))
    print(f"{'OPEN' if a.open else 'dry run'}: {len(rank)} ranking rows, {len(law)} law rows, "
          f"{time.time() - start:.1f} s")


if __name__ == "__main__":
    main()

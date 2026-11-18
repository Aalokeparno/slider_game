"""Units 6 + 7 test pipeline (docs/u67_preregistration.md).

    --dry-run   the whole pipeline with Hierarchy standing in for Ring / CT (Hierarchy EL as "Ring EL",
                Hierarchy CL v09 as "Ring CL v09", Hierarchy EL as "CT EL"); no test access.
                Outputs results/tables/u67_dryrun_*.csv (their numbers mean nothing).
    --open      the one Ring / CT run. Refuses unless docs/u67_preregistration.md exists with sha256 ==
                config.U67_PREREG_SHA and results/test_access_log.csv has no earlier u67 entry. Then one
                load_runs("behavior", roles={"test"}, allow_test=True, reason=anatomy.REASON) (it writes the access-log
                row), cached in results/test_cache/u67_runs.pkl; cells Ring EL, Ring CL v09, CT EL (L5 filters, old CL
                dropped); K, M, C, L, O at T = inf and U67_THAT; predictions from the pre-registration's YAML block
                scored; outputs results/tables/u67_test_*.csv.
"""
import argparse
import time

from slider import anatomy


def main():
    ap = argparse.ArgumentParser(description="Units 6 + 7 test pipeline (docs/u67_preregistration.md)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Hierarchy stands in for Ring / CT; no test access")
    g.add_argument("--open", action="store_true", help="the one-time Ring / CT run")
    a = ap.parse_args()
    start = time.time()
    out = anatomy.test_pipeline("open" if a.open else "dry")
    print(out["selection"].to_string(index=False))
    print(out["counts"][out["counts"]["variant"] == "main"].to_string(index=False))
    print(f"{len(out['scores'])} predictions scored")
    print(f"{'OPEN' if a.open else 'dry run'}: {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()

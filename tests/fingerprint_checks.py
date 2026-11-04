"""Fingerprint of targets.check_stats (sha256 of its keys and values) on the human L3 / L5 runs and on one simulated
replicate of a small L3 design, for test_traps.py: the check statistics must stay bit-identical. Made with the code
before P27 (commit 6fb02b0):

    python tests/fingerprint_checks.py      (writes tests/data/check_stats_fingerprint.json)
"""
import hashlib
import json
from pathlib import Path

from slider import design
from slider import targets

PATH = Path(__file__).parent / "data" / "check_stats_fingerprint.json"


def inputs():
    """{name: (runs3, table3, runs5)}: the human runs and one simulated replicate (L1, seed 7)."""
    runs3, table3 = design.load("L3")
    runs5, _ = design.load("L5")
    d3 = design.design("L3")
    small = d3.groupby(["topology", "dataset", "whole_second"]).head(3).reset_index(drop=True)
    sim3, simt3 = design.simulate_design(small, "L1", R=1, seed=7)
    sim5 = [run for run, whole in zip(sim3, small["whole_second"]) if not whole]
    return {"human": (runs3, table3, runs5), "sim": (sim3, simt3, sim5)}


def fingerprint(out):
    h = hashlib.sha256()
    h.update(out[["target", "stat", "topology", "dataset", "group", "in_check_list"]].to_csv(index=False).encode())
    h.update(out["value"].to_numpy(float).tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    prints = {name: fingerprint(targets.check_stats(*x)) for name, x in inputs().items()}
    PATH.write_text(json.dumps(prints, indent=1) + "\n")
    print(prints)

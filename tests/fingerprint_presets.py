"""Fingerprints of the frozen presets (sha256 of simulated events, mode ticks, ends and offsets), for
test_b1.py and test_c1.py: the old presets must stay bit-identical. L1, final, L1_M3, L0, APEM: made with the code
before Model B, B1 (commit 4a0813c); B1 added with the code before Model C, C1 (commit 7874a88); C1 added with the
code before Model C, C2 (commit d111a1e):

    python tests/fingerprint_presets.py      (writes tests/data/preset_fingerprints.json)
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from slider import design

PRESETS = {"L1": {}, "final": {}, "L1_M3": {}, "L0": {}, "APEM": {"a": 0.5, "s_th": 0.5, "mu": 0.1}, "B1": {}, "C1": {}}
PATH = Path(__file__).parent / "data" / "preset_fingerprints.json"


def small_design():
    """The first 2 rounds of each interface in the L3 design (old CL includes a whole-second file)."""
    d3 = design.design("L3")
    return d3.groupby("interface", sort=True).head(2).reset_index(drop=True)


def fingerprint(preset, **apem):
    h = hashlib.sha256()
    for out, _ in design.simulate_rounds(small_design(), preset, R=2, seed=77, **apem):
        for o in out:
            h.update(o["events"].to_numpy(float).tobytes())
            h.update(np.asarray(o["mode_ticks"], dtype=np.int64).tobytes())
            h.update(np.array([o["ticks"], o["success"], o["u"]], dtype=float).tobytes())
    return h.hexdigest()


def fingerprints():
    return {p: fingerprint(p, **kw) for p, kw in PRESETS.items()}


if __name__ == "__main__":
    PATH.write_text(json.dumps(fingerprints(), indent=1) + "\n")
    print(PATH.read_text())

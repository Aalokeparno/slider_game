"""scripts/p24_checks.py: the zero-width CI rule and the triggers (Rescue, L1b, L3), on SYNTHETIC tables only
(made-up numbers; never the real p24 outputs)."""
import sys

import numpy as np
import pandas as pd
import pytest

from slider import config
from slider import targets

sys.path.insert(0, str(config.ROOT / "scripts"))
import p24_checks as pc  # noqa: E402

CELLS = [("Hierarchy", "EL"), ("Hierarchy", "CL"), ("Hierarchy", "CC"), ("Full", "CL"), ("Full", "CC"), ("Star", "CC")]
STATS = [(5, targets.CHECKS[5])] + [(t, s) for t, s, _ in pc.L1B]
NAME = {targets.CHECKS[5]: "5", targets.CHECKS[10]: "10a", targets.PAUSE_SHARE: "10c", targets.CHECKS[2]: "2"}
R = 4


def human():
    """Every synthetic CI is [0, 10]; target 5 in the Full cells only."""
    rows = [{"target": t, "stat": s, "topology": top, "dataset": ds, "value": 5.0, "ci_lo": 0.0, "ci_hi": 10.0,
             "in_check_list": s != targets.PAUSE_SHARE, "ci_rule": ""}
            for t, s in STATS for top, ds in CELLS if t != 5 or top == "Full"]
    return pd.DataFrame(rows)


def sd_human():
    """Every synthetic human between-group SD CI is [1, 2]."""
    return pd.DataFrame([{"target": 8, "stat": pc.SD_STAT, "topology": top, "dataset": ds, "value": 1.5,
                          "ci_lo": 1.0, "ci_hi": 2.0, "in_check_list": False, "ci_rule": ""} for top, ds in CELLS])


def reps(misses=(), low_sd=()):
    """A per-replicate table like p24_main.py's: every statistic at 5 (inside [0, 10]) except misses
    {(name, topology, dataset)} at 20; per-group rates with between-group SD 1.5 (inside [1, 2]), or 0.39 (below 1)
    in the cells of low_sd."""
    rows = []
    for rep in range(R):
        for t, s in STATS:
            for top, ds in CELLS:
                if t == 5 and top != "Full":
                    continue
                value = 20.0 if (NAME[s], top, ds) in misses else 5.0 + 0.01 * rep
                rows.append({"target": t, "stat": s, "topology": top, "dataset": ds, "group": "", "value": value,
                             "in_check_list": s != targets.PAUSE_SHARE, "replicate": rep})
        for top, ds in CELLS:
            step = 0.3 if (top, ds) in low_sd else 1.5 / np.std([0, 1, 2, 3], ddof=1)
            for g in range(4):
                rows.append({"target": 8, "stat": targets.GROUP_RATE, "topology": top, "dataset": ds,
                             "group": f"g{g}", "value": 3 + step * g, "in_check_list": False, "replicate": rep})
    return pd.DataFrame(rows)


def run(**presets):
    return pc.triggers(pc.build_tables(presets, human(), sd_human()))


def test_nothing_fires_when_everything_passes():
    out = run(L1=reps(), L1_M3=reps(), L0=reps())
    assert out["rescue"] is False and out["rescue_l1_miss"] == 0 and out["rescue_m3_pass"] == 2
    assert out["l1b"] is False and out["l1b_misses"] == {"10a": 0, "10c": 0, "2": 0}
    assert out["l3"] is False and out["l3_below"] == 0 and out["l3_cells"] == 6


def test_rescue_fires():
    out = run(L1=reps(misses={("5", "Full", "CC")}), L1_M3=reps())
    assert out["rescue"] is True and out["rescue_l1_miss"] == 1 and out["rescue_m3_pass"] == 2


def test_rescue_does_not_fire():
    both = {("5", "Full", "CC"), ("5", "Full", "CL")}
    assert run(L1=reps(misses=both), L1_M3=reps(misses={("5", "Full", "CL")}))["rescue"] is False   # M3 misses one
    assert run(L1=reps(), L1_M3=reps())["rescue"] is False                                       # L1 misses none
    assert run(L1=reps(misses=both))["rescue"] is None                                           # no L1_M3 runs


def test_l1b_fires_for_one_statistic_in_two_cells():
    out = run(L1=reps(misses={("10c", "Full", "CC"), ("10c", "Star", "CC")}))
    assert out["l1b"] is True and out["l1b_misses"] == {"10a": 0, "10c": 2, "2": 0}
    out = run(L1=reps(misses={("2", "Hierarchy", "EL"), ("2", "Hierarchy", "CL"), ("2", "Star", "CC")}))
    assert out["l1b"] is True and out["l1b_misses"]["2"] == 3


def test_l1b_does_not_fire_for_single_misses_of_different_statistics():
    out = run(L1=reps(misses={("10a", "Full", "CC"), ("10c", "Star", "CC"), ("2", "Hierarchy", "EL")}))
    assert out["l1b"] is False and out["l1b_misses"] == {"10a": 1, "10c": 1, "2": 1}


def test_l3_fires_in_two_cells_and_not_in_one():
    out = run(L1=reps(low_sd={("Hierarchy", "EL"), ("Full", "CC")}))
    assert out["l3"] is True and out["l3_below"] == 2
    out = run(L1=reps(low_sd={("Star", "CC")}))
    assert out["l3"] is False and out["l3_below"] == 1


def test_l0_never_decides():
    bad = reps(misses={("10a", c[0], c[1]) for c in CELLS} | {("5", "Full", "CC")}, low_sd=set(CELLS))
    assert run(L1=reps(), L1_M3=reps(), L0=bad) == run(L1=reps(), L1_M3=reps())
    assert run(L1=reps(), L1_M3=reps(), final=bad) == run(L1=reps(), L1_M3=reps())          # nor does final
    assert pc.triggers(pc.build_tables({"L0": bad}, human(), sd_human())) is None             # no L1: nothing


def test_zero_width_rule():
    h = pd.DataFrame([
        {"target": 1, "stat": targets.CHECKS[1], "topology": "Star", "dataset": "CC", "value": 100.0, "ci_lo": 100.0, "ci_hi": 100.0},
        {"target": 1, "stat": targets.CHECKS[1], "topology": "Full", "dataset": "CC", "value": 0.0, "ci_lo": 0.0, "ci_hi": 0.0},
        {"target": 10, "stat": targets.CHECKS[10], "topology": "Full", "dataset": "CC", "value": 0.6, "ci_lo": 0.6, "ci_hi": 0.6 + 1e-15},
        {"target": 2, "stat": targets.CHECKS[2], "topology": "Full", "dataset": "CL", "value": 40.0, "ci_lo": 30.0, "ci_hi": 50.0}])
    out = pc.zero_width_rule(h, {("Star", "CC"): 39, ("Full", "CC"): 36})
    assert out["ci_lo"].iloc[0] == pytest.approx(100 * 0.025 ** (1 / 39)) and out["ci_hi"].iloc[0] == 100
    assert out["ci_lo"].iloc[1] == 0 and out["ci_hi"].iloc[1] == pytest.approx(100 * (1 - 0.025 ** (1 / 36)))
    assert out["ci_lo"].iloc[2] == pytest.approx(0.55) and out["ci_hi"].iloc[2] == pytest.approx(0.65)
    assert out[["ci_lo", "ci_hi"]].iloc[3].tolist() == [30.0, 50.0] and out["ci_rule"].iloc[3] == ""
    assert out["ci_rule"].tolist()[:3] == ["Clopper-Pearson", "Clopper-Pearson", "lab clock +/- 0.05 s"]
    other = pd.DataFrame([{"target": 6, "stat": targets.CHECKS[6], "topology": "Full", "dataset": "CC", "value": 5.0,
                           "ci_lo": 5.0, "ci_hi": 5.0}])
    with pytest.raises(SystemExit):
        pc.zero_width_rule(other, {})


def test_report_prints_every_trigger(capsys):
    presets = {"L1": reps(misses={("5", "Full", "CC"), ("10c", "Full", "CC"), ("10c", "Star", "CC")},
                          low_sd={("Hierarchy", "EL"), ("Full", "CC")}), "L1_M3": reps(), "L0": reps()}
    h = human()
    h.loc[0, "ci_rule"] = "Clopper-Pearson"                             # as if zero_width_rule had replaced it
    tables = pc.build_tables(presets, h, sd_human())
    changed = h.iloc[:1]
    pc.report(tables, {p: R for p in presets}, changed, pc.triggers(tables))
    text = capsys.readouterr().out
    assert "L1 misses in 1 of 2 Full cells, L1_M3 passes 2 of 2 -> TRIGGERED" in text
    assert "-> TRIGGERED (10c)" in text
    assert "below the lower bound in 2 of 6 cells -> TRIGGERED" in text
    assert "[CI: Clopper-Pearson]" in text and "L2: no trigger" in text

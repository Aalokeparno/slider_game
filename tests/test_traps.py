"""traps.py: trap hops on hand-made rounds on the Hierarchy graph (apem.GRAPHS, nodes = seats).

Node 0 is the apex, 1 and 2 its children, 3, 4 under 1 and 5, 6 under 2. A = left branch {1, 3, 4} at -1 and
the rest at +1; B = right branch {2, 5, 6} at -1. Both are fraught; so are their negations.
"""
import json

import numpy as np
import pandas as pd

from slider import slider_logs as sl
from slider import targets
from slider import traps
from slider.fraught import fraught_mask
from slider.models.apem import GRAPHS

import fingerprint_checks

GRAPH = GRAPHS["Hierarchy"].astype(int)
A = np.array([1, -1, 1, -1, -1, 1, 1], dtype=float)
B = np.array([1, 1, -1, 1, 1, -1, -1], dtype=float)
ZERO = np.zeros(7)                   # not fraught (zeros)
WIN = np.ones(7)                     # consensus


def toy(rows):
    """A successful lab round from [(time, s row), ...]; the last row should be WIN."""
    t, X = zip(*rows)
    return {"A": GRAPH, "weights": np.ones(7), "t": np.array(t, dtype=float), "X": np.array(X),
            "end": "success", "session": "lab", "dataset": "EL", "file": "toy.txt", "wave": 1}


def test_the_toy_patterns_are_fraught():
    assert sl.label_topology(GRAPH) == "Hierarchy"
    run = toy([(0, A), (1, -A), (2, B), (3, ZERO), (4, WIN)])
    assert list(fraught_mask(run)) == [True, True, True, False, False]


def test_a_no_fraught_spell():
    out = traps.trap_hops(toy([(0, ZERO), (5, WIN)]))
    assert out == {"n_spells": 0, "n_distinct": 0, "n_hops": 0, "n_cut": 0}


def test_b_two_spells_in_the_same_orthant():
    out = traps.trap_hops(toy([(0, ZERO), (1, A), (3, ZERO), (10, A), (12, ZERO), (20, WIN)]))
    assert out == {"n_spells": 2, "n_distinct": 1, "n_hops": 0, "n_cut": 0}


def test_c_an_orthant_then_its_negation():
    out = traps.trap_hops(toy([(0, ZERO), (1, A), (3, ZERO), (10, -A), (12, ZERO), (20, WIN)]))
    assert out["n_distinct"] == 1 and out["n_hops"] == 0 and out["n_spells"] == 2


def test_d_A_B_A():
    out = traps.trap_hops(toy([(0, ZERO), (1, A), (3, ZERO), (10, B), (12, ZERO), (20, A), (22, ZERO), (30, WIN)]))
    assert out == {"n_spells": 3, "n_distinct": 2, "n_hops": 2, "n_cut": 0}


def test_a_spell_with_two_orthants_is_cut():
    """A and B less than 1 s apart: fraught.spells merges them into one spell, which is cut into two pieces."""
    out = traps.trap_hops(toy([(0, ZERO), (1, A), (3, ZERO), (3.5, B), (5, ZERO), (20, WIN)]))
    assert out == {"n_spells": 2, "n_distinct": 2, "n_hops": 1, "n_cut": 1}


def test_canonical_identifies_a_pattern_with_its_negation():
    assert traps.canonical(A) == traps.canonical(-A) == (1, -1, 1, -1, -1, 1, 1)
    assert traps.canonical(np.array([0, -2.0, 3])) == (0, 1, -1)


def test_trap_hop_table_shape():
    runs = [toy([(0, ZERO), (1, A), (3, ZERO), (10, B), (12, ZERO), (20, WIN)]),
            {**toy([(0, ZERO), (5, WIN)]), "file": "other.txt"}]
    tab = traps.trap_hop_table(runs, n_boot=50)
    assert list(tab["stat"]) == traps.STATS
    assert tab.set_index("stat")["value"].to_dict() == {"n_spells": 1.0, "n_distinct": 1.0, "n_hops": 0.5,
                                                        "any_hop": 50.0}
    assert set(tab["n_runs"]) == {2} and set(tab["n_groups"]) == {2}


# ------------------------------------------------------------- fast path: optional, bit-identical by default

def test_check_stats_unchanged_and_trap_hops_only_add_rows():
    """check_stats without the flag gives the fingerprint made before P27; with it, the same rows plus target 11."""
    saved = json.loads(fingerprint_checks.PATH.read_text())
    for name, x in fingerprint_checks.inputs().items():
        default = targets.check_stats(*x)
        assert fingerprint_checks.fingerprint(default) == saved[name]
        extra = targets.check_stats(*x, trap_hops=True)
        pd.testing.assert_frame_equal(extra.iloc[:len(default)], default)
        added = extra.iloc[len(default):]
        assert set(added["target"]) == {targets.TRAP_HOPS} and not added["in_check_list"].any()
        assert set(added["topology"]) == {"Hierarchy"} and len(added) == 4 * added["dataset"].nunique()

"""design.py: run sets, seat maps, the speed rule, and simulate_design. Train / val rounds only (no test runs)."""
import numpy as np
import pandas as pd
import pytest

from slider import config
from slider import design
from slider import interface
from slider import sim


@pytest.fixture(scope="module")
def designs():
    return {name: design.design(name) for name in design.SETS}


def test_sets_hold_train_topologies_and_roles(designs):
    for name, d in designs.items():
        assert set(d["topology"]) <= config.TRAIN_TOPOLOGIES
        assert set(d["role"]) == ({"train"} if name == "G6train" else {"train", "val"})
        assert d["round_id"].is_unique


def test_L5_is_L3_minus_the_whole_second_rounds(designs):
    l3, l5 = designs["L3"], designs["L5"]
    assert set(l5["round_id"]) <= set(l3["round_id"])
    assert set(l3["round_id"]) - set(l5["round_id"]) == set(l3.loc[l3["whole_second"], "round_id"])
    assert l3["whole_second"].any() and not l5["whole_second"].any()


@pytest.mark.parametrize("name", ["L3", "G6train"])
def test_seat_map_gives_the_logged_adjacency(name, designs):
    runs, _ = design.load(name)
    for run, row in zip(runs, designs[name].itertuples()):
        assert np.array_equal(row.graph[np.ix_(row.seat_node, row.seat_node)], run["A"])


def test_speed_rule(designs):
    files = pd.read_csv(config.INTERFACE_TABLE).set_index("file")
    d = designs["L3"]
    cc = d[d["interface"] == "cc"]
    assert np.array_equal(cc["speed"].to_numpy(), files.loc[cc["group"], "speed"].to_numpy())
    whole = d[d["whole_second"]]
    assert set(whole["interface"]) == {"old_cl"} and set(whole["speed"]) == {0.2}
    rest = d[~d["whole_second"]]
    assert np.allclose(rest["d"], files.loc[rest["group"], "step_size"])
    assert np.allclose(d["dt"], d["d"] / d["speed"])


def test_speed_is_the_hazard_table_round_speed(designs):
    if not config.HAZARD_TABLE.exists():
        pytest.skip("hazard table not built")
    ht = pd.read_parquet(config.HAZARD_TABLE, columns=["round_id", "round_speed"]).drop_duplicates()
    m = designs["G6train"].merge(ht, on="round_id")
    assert len(m) >= 0.95 * len(designs["G6train"])
    assert np.allclose(m["speed"], m["round_speed"])


def test_simulate_design_keeps_each_round(designs):
    d = designs["L3"].groupby(["interface", "whole_second", "role"]).head(1).reset_index(drop=True)
    runs, table = design.simulate_design(d, "L1", R=2, seed=1)
    assert len(runs) == len(table) == 2 * len(d)
    rep = d.loc[d.index.repeat(2)].reset_index(drop=True)         # rows in run order: round, then copy
    for col in ["dataset", "role", "topology", "wave"]:
        assert list(table[col]) == list(rep[col])
    assert list(table["coarse_time"]) == list(rep["whole_second"])
    assert list(table["session"]) == ["cloud" if i == "cc" else "lab" for i in rep["interface"]]
    for run, row in zip(runs, rep.itertuples()):
        assert run["file"].endswith(row.group) and run["file"].startswith("sim")
        assert interface.version_tag(run["file"]) == interface.version_tag(row.group)
        assert np.array_equal(run["A"], row.graph[np.ix_(row.seat_node, row.seat_node)])
        if row.interface != "cc" and not row.whole_second:          # lab logger: the speed comes back exactly
            step, pace = interface.step_and_pace(interface.moves(run))
            assert step * pace == pytest.approx(row.speed)
    ends = [r["t"][-1] for r in runs]
    assert all(e <= T + 1e-9 for e, T in zip(ends, rep["T"]))
    assert all(sim.consensus(r["X"][-1]) == (r["end"] == "success") for r in runs)

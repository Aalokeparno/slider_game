"""load_runs: roles from split T and split G, and the guard on the test runs.

None of these tests passes allow_test=True, so results/test_access_log.csv is never written.
"""
import pandas as pd
import pytest

from slider import config
from slider import slider_logs as sl


@pytest.mark.parametrize("purpose", ["outcome", "behavior"])
def test_default_call_returns_no_test_runs(purpose):
    runs, table = sl.load_runs(purpose)
    assert len(runs) == len(table) > 0
    assert set(table["role"]) == {"train"}


def test_asking_for_test_without_allow_test_raises():
    with pytest.raises(sl.HeldOutError):
        sl.load_runs("outcome", roles={"test"})
    with pytest.raises(sl.HeldOutError):
        sl.load_runs("behavior", roles={"train", "val", "test"})


def test_allow_test_without_reason_raises():
    with pytest.raises(ValueError):
        sl.load_runs("outcome", roles={"test"}, allow_test=True)


@pytest.mark.parametrize("purpose", ["outcome", "behavior"])
def test_runs_with_fewer_than_2_rows_are_dropped(purpose):
    runs, table = sl.load_runs(purpose, roles={"train", "val"})
    assert table["T"].min() >= 2
    assert all(len(run["t"]) >= 2 for run in runs)


def test_keep_empty_brings_them_back():
    runs, table = sl.load_runs("behavior", roles={"train", "val"}, keep_empty=True, keep_aborted=True)
    assert (table["T"] == 0).sum() > 0 and (table["T"] == 1).sum() > 0


def test_role_counts_match_split_G():
    split = pd.read_csv(config.SPLIT_G)
    runs, table = sl.load_runs("behavior", roles={"train", "val"}, keep_aborted=True, keep_whole_second=True, keep_empty=True)

    # every run carries its group's label, and the groups per role are the groups per split
    label = dict(zip(split["group_id"], split["split"]))
    assert all(table["role"] == table["file"].map(label))
    for role in ["train", "val"]:
        assert set(table.loc[table["role"] == role, "file"]) == set(split.loc[split["split"] == role, "group_id"])
    assert table.groupby("role")["file"].nunique().to_dict() == split["split"].value_counts().to_dict()

    # and no Hierarchy / Full / Star run of a split G group is missing
    all_runs = sl.runs_table(sl.parse_all())
    expected = all_runs["file"].isin(split["group_id"]) & all_runs["topology"].isin(config.TRAIN_TOPOLOGIES)
    assert len(table) == expected.sum()

"""slider_logs.test_design: design-only access to the test rounds.

The log is written to a temporary file here, never to results/test_access_log.csv.
"""
import numpy as np
import pandas as pd
import pytest

from slider import config
from slider import slider_logs as sl


@pytest.fixture
def log(tmp_path, monkeypatch):
    path = tmp_path / "test_access_log.csv"
    monkeypatch.setattr(config, "TEST_ACCESS_LOG", path)
    return path


@pytest.fixture
def no_row_parser(monkeypatch):
    """Every row parser raises if it is called."""
    def fail(*args, **kwargs):
        raise AssertionError("the row parser was called")
    for name in ["parse_lab", "parse_cloud", "parse_all", "load_runs"]:
        monkeypatch.setattr(sl, name, fail)


def test_columns_are_exactly_the_design_columns(log, no_row_parser):
    d = sl.test_design("unit test")
    assert list(d.columns) == ["file", "group_id", "dataset", "version", "topology", "wave", "time_limit"]
    assert len(d) > 0
    assert set(d["topology"]) <= {"Ring", "CT-Hierarchy"}


def test_never_calls_the_row_parser(log, no_row_parser):
    sl.test_design("unit test")          # no_row_parser raises on any call


def test_one_log_line_per_call(log, no_row_parser):
    sl.test_design("first")
    sl.test_design("second")
    lines = pd.read_csv(log)
    assert len(lines) == 2
    assert list(lines["reason"]) == ["first", "second"]
    assert set(lines["kind"]) == {"design_only"}


def test_old_log_lines_get_kind_full(log, no_row_parser):
    old = "timestamp,caller,reason,n_test_runs\n2026-11-25T10:00:00,nb.ipynb,\"a, b\",12\n"
    log.write_text(old)
    sl.test_design("new")
    lines = log.read_text().splitlines()
    assert lines[0] == "timestamp,caller,reason,n_test_runs,kind"
    assert lines[1] == "2026-11-25T10:00:00,nb.ipynb,\"a, b\",12,full"       # old line unchanged, kind added
    assert list(pd.read_csv(log)["kind"]) == ["full", "design_only"]


def test_needs_a_reason(log, no_row_parser):
    with pytest.raises(ValueError):
        sl.test_design("")


def test_whole_second_files_match_load_runs():
    """WHOLE_SECOND_FILES = the coarse_time files of load_runs (train + val rows)."""
    _, table = sl.load_runs("behavior", roles={"train", "val"}, keep_whole_second=True)
    assert set(table.loc[table["coarse_time"], "file"]) == sl.WHOLE_SECOND_FILES


def test_settings_waves_match_the_parsed_training_rounds():
    """The k-th settings wave is the k-th played wave: same topology and time limit on every training round."""
    runs, table = sl.load_runs("behavior", roles={"train", "val"}, keep_whole_second=True, keep_empty=True,
                               keep_aborted=True)
    lab, cloud = sl.log_files()
    paths = {p.name: p for p in lab + cloud}
    waves = {name: sl.settings_waves(p) for name, p in paths.items() if name in set(table["file"])}
    for run, top in zip(runs, table["topology"]):
        rows, limit = waves[run["file"]][run["wave"] - 1]
        A_log = np.array(rows, dtype=int)
        assert sl.label_topology(A_log - np.diag(np.diag(A_log))) == top
        assert np.array_equal(A_log, run["A_log"])
        assert limit == config.TIME_LIMIT[run["session"]]

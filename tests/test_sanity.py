"""Sanity checks: the parser and the fraught measures still give the known numbers.

Run from the Tabor/ folder:  pytest
"""
import numpy as np
import pytest

from slider import slider_logs as sl
from slider.fraught import fraught_mask, fraught_percent_hold


@pytest.fixture(scope="module")
def runs():
    return sl.parse_all()


@pytest.fixture(scope="module")
def table(runs):
    return sl.runs_table(runs)


def test_718_runs_parsed(runs):
    assert len(runs) == 718


def test_recomputed_agreement_matches_logged(table):
    assert table["agree_error"].max() < 0.005


def test_cl_hierarchy_time_to_consensus(table):
    won = table[(table["dataset"] == "CL") & (table["topology"] == "Hierarchy") & table["success"]]
    assert round(won["duration"].mean(), 2) == 61.61


def test_cl_hierarchy_percent_fraught(runs, table):
    # all 63 CL Hierarchy rounds (paper number), forward hold
    hier = table.index[(table["dataset"] == "CL") & (table["topology"] == "Hierarchy")]
    mean = np.mean([fraught_percent_hold(i, runs, table) for i in hier])
    assert round(mean, 2) == 11.58


def test_no_fraught_rows_on_full(runs, table):
    full = [i for i in table.index[table["topology"] == "Full"] if len(runs[i]["t"]) > 0]
    assert sum(fraught_mask(runs[i]).sum() for i in full) == 0

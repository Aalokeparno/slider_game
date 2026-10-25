"""All paths and fixed settings for the project. Nothing else should hard-code a path."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]       # the Tabor/ folder (this file is src/slider/config.py)

# raw data: the OSF archive gvcpz, moved here unchanged
RAW = ROOT / "data" / "raw" / "osf"
OSF_CODE = RAW / "Code"
OSF_DATA = RAW / "Data"
LAB_DIR = OSF_DATA / "Data-2016-12-09"            # lab logs (2016)
CLOUD_DIR = OSF_DATA / "2018-05-31 RunsScrubbed"  # cloud logs (2018)

# processed data
PROCESSED = ROOT / "data" / "processed"
RUNS_TABLE = PROCESSED / "runs_table.csv"
RUNS_FRAUGHT = PROCESSED / "runs_fraught.csv"

# splits of the runs (split T, by topology, is TRAIN_TOPOLOGIES / TEST_TOPOLOGIES below)
SPLITS = ROOT / "data" / "splits"

# results
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
RING_TRAPS = TABLES / "ring_traps.csv"
APEM_SWEEP = TABLES / "apem_sweep.csv"
APEM_TOPOLOGIES = TABLES / "apem_topologies.csv"

# seed for new work (Stage 0 and Stage 1 notebooks keep their own seed 0)
SEED = 573

# split T (by topology): no new metrics on the test topologies before Nov 25
TRAIN_TOPOLOGIES = {"Hierarchy", "Full", "Star"}
TEST_TOPOLOGIES = {"Ring", "CT-Hierarchy"}

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
SPLITS_DIR = ROOT / "data" / "splits"
SPLIT_G = SPLITS_DIR / "split_G.csv"               # split G: by group (log file), train / val

# results
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
TEST_ACCESS_LOG = RESULTS / "test_access_log.csv"   # every load_runs call that returns test runs
RING_TRAPS = TABLES / "ring_traps.csv"
APEM_SWEEP = TABLES / "apem_sweep.csv"
APEM_TOPOLOGIES = TABLES / "apem_topologies.csv"
H5_TRAIN = TABLES / "h5_train.csv"
H5_TRAIN_CHECKS = TABLES / "h5_train_checks.csv"
H5_VAL = TABLES / "h5_val.csv"
TRAIN_TARGETS = TABLES / "train_targets.csv"
TARGET8 = TABLES / "target8_reversal_rate.csv"
TARGET9 = TABLES / "target9_agreement_at_reversal.csv"
TARGET9_HAZARD = TABLES / "target9_hazard.csv"
TARGET9_FIGURE = FIGURES / "target9_agreement_at_reversal.png"

# time limit of a round (seconds)
TIME_LIMIT = {"lab": 300, "cloud": 180}

# seed for new work (Stage 0 and Stage 1 notebooks keep their own seed 0)
SEED = 573

# split T (by topology): no new metrics on the test topologies before Nov 25
TRAIN_TOPOLOGIES = {"Hierarchy", "Full", "Star"}
TEST_TOPOLOGIES = {"Ring", "CT-Hierarchy"}

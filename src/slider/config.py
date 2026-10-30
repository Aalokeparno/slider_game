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
HAZARD_TABLE = PROCESSED / "hazard_table.parquet"
HAZARD_TABLE_S1 = PROCESSED / "hazard_table_s1.parquet"     # P13 S1: b = 0.1 s (m = 2)
HAZARD_TABLE_S2 = PROCESSED / "hazard_table_s2.parquet"     # P13 S2: b = 0.5 s (m = 2)
HAZARD_TABLE_S3 = PROCESSED / "hazard_table_s3.parquet"     # P13 S3: m = 3 (b = 0.25 s)

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
TARGET10 = TABLES / "target10_crossing_time.csv"
TARGET10_FIGURE = FIGURES / "target10_crossing_time.png"
INTERFACE_TABLE = TABLES / "interface_table.csv"
P10_DECISION1 = TABLES / "p10_decision1.csv"
P10_COEFS = TABLES / "p10_coefs.csv"
P11_2X2 = TABLES / "p11_2x2.csv"
P11_COEFS = TABLES / "p11_coefs.csv"
P11_PROBE = TABLES / "p11_probe.csv"
P12_BETA_CI = TABLES / "p12_beta_ci.csv"
P13_TABLES_CHECK = TABLES / "p13_tables_check.csv"
P13_WEIGHTS = TABLES / "p13_weights.csv"
P13_VERDICTS = TABLES / "p13_verdicts.csv"
P17_RATES = TABLES / "p17_rates.csv"
P17_RATES_V1 = TABLES / "p17_rates_v1.csv"    # before F1b (Oct 29): steps onto the wall counted as stop exposure
P17_SURVIVAL = FIGURES / "p17_survival.png"
P23_RECOVERY = TABLES / "p23_recovery.csv"
P23_RECOVERY_V1 = TABLES / "p23_recovery_v1.csv"    # first run, simulator at commit 59b7e68 (before F1-F3, F1b)
P24_PILOT = TABLES / "p24_pilot.csv"                 # pilot: SD over 10 L1 replicates, human CI half-width, R needed
P24_REPS = {preset: TABLES / f"p24_reps_{preset}.parquet" for preset in ["L1", "L1_M3", "L0"]}   # main runs
P24_CHECKS = TABLES / "p24_checks.csv"

# time limit of a round (seconds)
TIME_LIMIT = {"lab": 300, "cloud": 180}

# seed for new work (Stage 0 and Stage 1 notebooks keep their own seed 0)
SEED = 573

# split T (by topology): no new metrics on the test topologies before Nov 25
TRAIN_TOPOLOGIES = {"Hierarchy", "Full", "Star"}
TEST_TOPOLOGIES = {"Ring", "CT-Hierarchy"}

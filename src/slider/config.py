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
GRAPH_SPLIT = SPLITS_DIR / "graph_split.csv"       # Path 3: the 853 connected 7-node graphs, graph-train / graph-test

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
P24_REPS = {preset: TABLES / f"p24_reps_{preset}.parquet" for preset in ["L1", "L1_M3", "L0", "final", "B1", "C1", "C2"]}   # main runs (B1, C1, C2: exploratory)
P24_CHECKS = TABLES / "p24_checks.csv"
P24_DIRECTIONS = TABLES / "p24_directions.csv"       # descriptive: L1 sim mean vs human, direction, gap in half-widths
P25_HALTS = TABLES / "p25_halts.csv"                 # L1b: two-type halt mixture per interface (G-train)
P25_HALTS_FIGURE = FIGURES / "p25_halts.png"
P25_SIGMA = TABLES / "p25_sigma.csv"                 # L3: sigma_g of the group effect on the M2' hazard (G-train)
P25_RECOVERY = TABLES / "p25_recovery.csv"           # build check: "final" on G6train, R = 2, seed 573
PB1_HALTS = TABLES / "pB1_halts_by_state.csv"          # Model B, B1 (exploratory): resume chance by state C / E
PB1_HALTS_FIGURE = FIGURES / "pB1_halts_by_state.png"
PB1_PARAMS = TABLES / "pB1_params.csv"                 # B1 parameters per interface (G-train; build fix 1: style per attempt)
PB1_PARAMS_V1 = TABLES / "pB1_params_v1.csv"           # B1 before build fix 1 (pi, rho, q_sg from completed crossings)
PB1_RECOVERY = TABLES / "pB1_recovery.csv"             # build check: B1 on G6train, R = 2, seed 573 (after build fix 2)
PB1_RECOVERY_V1 = TABLES / "pB1_recovery_v1.csv"       # the same with pB1_params_v1.csv
PB1_GAPS = TABLES / "pB1_gaps.csv"                     # descriptive: (sim - human) / CI half-width, L1 / final / B1 (p24_checks.csv)
PC_DURATIONS = TABLES / "pC_durations.parquet"           # Model C, C1 (exploratory): human first-trip and wall-to-wall durations (G-train)
PC2_CHECKS = TABLES / "pC2_checks.csv"                 # Model C, C2 (exploratory): step-0 checks (G-train)
PC2_PARAMS = TABLES / "pC2_params.csv"                 # C2 simulator inputs: d_ar, use_g, q_w, keep by sign(dA)
PC2_TRY = TABLES / "pC2_try_hazard.csv"                # C2 try hazard T0 / T1 coefficients
PC2_PACE = TABLES / "pC2_pace.parquet"                 # C2 kept-movement pace: f per 10a crossing, by interface and round speed
PB2_TRY_KEEP = TABLES / "pB2_try_keep.csv"             # B2: P(kept) by sign(dA)
PB0B_STYLE = TABLES / "pB0b_crossing_style.csv"        # B0b: pi, rho, q_sg
MODEL_B_LOG = RESULTS / "model_b_log.md"
P27_TEST_DESIGN = TABLES / "p27_test_design.csv"           # test rounds: design only (slider_logs.test_design)
P27_TRAPHOPS = TABLES / "p27_traphops_train.csv"         # trap hops (traps.py), human L5 Hierarchy, v09 + cloud
P27_TRAPHOPS_REF = TABLES / "p27_traphops_ref.csv"         # trap hops: humans vs C2 and APEM (published), 20 replicates

# Path 2, Unit 1: static trap landscape of the 853 connected 7-node graphs (landscape.py)
U1_TRAPS = TABLES / "u1_traps.csv"
U1_GRAPHS = TABLES / "u1_graphs.csv"
U1_BALANCE = TABLES / "u1_balance.csv"

# Path 2, Unit 2a: exact escape times under the logit rule, graph-train graphs only (markov.py)
U2_DIR = RESULTS / "u2"
U2_TAU_TRAIN = U2_DIR / "tau_train.parquet"
U2_TRAP_TAU = TABLES / "u2_trap_tau_train.csv"
U2B_GRAPH = TABLES / "u2b_graph_train.csv"           # Unit 2b: trap time, entries, first hit (per graph x T)
U2B_PLATEAU = TABLES / "u2b_plateau_train.csv"       # Unit 2b: committor per plateau x T

# Path 2, Unit 5: pre-registered ranking of graph-only predictors of trap time (u5.py, scripts/u5_test.py)
U5_PREREG = ROOT / "docs" / "u5_preregistration.md"
U5_H_LOCK = "6d366bd52dbeaa5723f300e5c54f130749c785e1"      # commit of the pre-registration (committed alone)
U5_EXPECTATION = ROOT / "docs" / "u5_train_expectation.md"
U5_H_EXP = "4efa9239a8ed4ce750d7b214d623b635c73ad338"       # commit of the S-train expectation
U5_TRAIN_RANKING = TABLES / "u5_train_ranking.csv"
U5_TRAIN_LAW = TABLES / "u5_train_law.csv"
U5_DRYRUN_RANKING = TABLES / "u5_dryrun_ranking.csv"
U5_DRYRUN_LAW = TABLES / "u5_dryrun_law.csv"
U5_DRYRUN_PREDICTIONS = TABLES / "u5_dryrun_predictions.csv"
U5_TEST_RANKING = TABLES / "u5_test_ranking.csv"            # written once by scripts/u5_test.py --open
U5_TEST_LAW = TABLES / "u5_test_law.csv"
U5_TEST_PREDICTIONS = TABLES / "u5_test_predictions.csv"
U5_TEST_CACHE = RESULTS / "u5_test_cache"                   # S-test outcomes (git-ignored)
GRAPH_TEST_ACCESS_LOG = RESULTS / "graph_test_access_log.csv"   # every computation on graph-test graphs
U5_TRAIN_FACTORS = TABLES / "u5_train_factors.csv"         # addendum: log entries / log duration ranking (S-train)
U5_ADDENDUM = ROOT / "docs" / "u5_addendum.md"
U5_H_ADD = "38eb4a482595a286aa3cf538f95aff729517a9d2"       # commit of the addendum (re-locked Nov 17: A3 vs the named S-train best)
# first addendum, superseded before opening (A3 vs the S-test #1): 2614695863662a9cec1484edad7290ebbe61c0c0
U5_DRYRUN_ADDENDUM = TABLES / "u5_dryrun_addendum.csv"
U5_TEST_ADDENDUM = TABLES / "u5_test_addendum.csv"          # written once by scripts/u5_test.py --open

# Path 3 (strategy simulator)
PATH3 = RESULTS / "path3"

# time limit of a round (seconds)
TIME_LIMIT = {"lab": 300, "cloud": 180}

# seed for new work (Stage 0 and Stage 1 notebooks keep their own seed 0)
SEED = 573

# split T (by topology): no new metrics on the test topologies before Nov 25
TRAIN_TOPOLOGIES = {"Hierarchy", "Full", "Star"}
TEST_TOPOLOGIES = {"Ring", "CT-Hierarchy"}

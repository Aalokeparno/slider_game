# ELEC573 — Slider Game / fraught states (Tabor)

Analysis of the Slider Game logs from Tabor, Smith & Dankowicz, *Escaping fraught states* (PsyArXiv 2023).

## Rule

**No new metrics on the test topologies (Ring, CT-Hierarchy) before Nov 25 (2026).**
New work uses the train topologies only (Hierarchy, Full, Star). This split (split T) is `TRAIN_TOPOLOGIES` / `TEST_TOPOLOGIES` in `src/slider/config.py`.
Split G splits the groups (one group = one log file) into train / val, 1/3 val within each of EL, CL and CC: `data/splits/split_G.csv`.

## Leakage register

What has been seen on held-out data, and what was frozen at that point.

- Oct 25: H5 summaries on val seen. Path 1 candidate rule features frozen: Agreement level, Agreement trend, |s| / at-end indicator, time since reaching the end, time since last reversal.
- Oct 27: P8 hazard table (`hazard.build_hazard_table`, m = 2; train + val) built; outcome counts per cell and split seen, no model fitted. Known property, kept as is: 33 reversals in the bot-free complete subset (Hierarchy EL 6, Hierarchy CL 5, Hierarchy CC 4, Full CL 14, Full CC 3, Star CC 1) have their first step in a dropped mid-event bin (the bin starts while the player's previous probe or reversal is pending), so they are in no kept bin's `y_rev`. With 1 bin holding 2 reversals (Full CL): sum(`y_rev`) = 8,476 = target 8 (b) 8,510 − 1 − 33. See `notebooks/path1/p8_hazard_table.ipynb`.
- Oct 27: P10 Decision 1 (R0 vs R1, pre-registered in the Path A Workbook Ch. A8.2): R0 and R1 fit on G-train (Hierarchy + Full, 300,245 bins, 5,493 reversals), scored on G-val (204,891 bins, 21 groups). delta (R1 − R0) per bin: primary (trend, w = 1 s) 1.57e-4, 95% CI −1.79e-5 to 2.96e-4; sensitivity (`trend_scaled`) −4.67e-6, CI −3.12e-4 to 2.10e-4. Verdict (b) in both runs. See `notebooks/path1/p10_decision1.ipynb` (`src/slider/fit.py`, `results/tables/p10_decision1.csv`, `p10_coefs.csv`).
- Oct 27: P11 2x2 (pre-registered in the Path A Workbook Ch. A8.2 / A8.4): M1 (A, |s|, at_end), M2 (M1 + trend), M2' (M1 + e, r), M3 (all six), R0 structure, fit on G-train (all training cells, 336,073 bins, 6,073 reversals), G-val scored (225,033 bins, 21 groups). delta per bin: M2 − M1 2.09e-4 (CI −1.28e-4 to 4.74e-4, includes 0); M2' − M1 2.86e-3 (1.53e-3 to 4.75e-3, above 0); M3 − M2' 4.67e-4 (−1.21e-5 to 8.28e-4, includes 0); M3 − M2 3.12e-3 (1.82e-3 to 4.90e-3, above 0). Secondary, descriptive: the same 4 models with `y_probe`, also scored on G-val. See `notebooks/path1/p11_2x2.ipynb` (`results/tables/p11_2x2.csv`, `p11_coefs.csv`, `p11_probe.csv`).
- Oct 28: P13 sensitivities (pre-registered in the Path A Workbook Ch. A8.6): S1 b = 0.1 s, S2 b = 0.5 s, S3 m = 3 (rebuilt tables, git-ignored `data/processed/hazard_table_s1/s2/s3.parquet`), S4 no abs_s, S5 capped e / r, S6 no bot runs, S7 no old-CL rows, S8 spline in A (descriptive); M1, M2, M2' fit on G-train. G-val reused for robustness only (no choice changed, P10–P12 not refit): M2' − M1 CI above 0 in all 8, M2 − M1 CI includes 0 in all 8. See `notebooks/path1/p13_sensitivity.ipynb` (`results/tables/p13_tables_check.csv`, `p13_weights.csv`, `p13_verdicts.csv`).
- Oct 29: Step 4 simulator (`src/slider/sim.py`). To fill in the end-of-round rule, the last rows of train and val rounds (behavior runs) were checked against candidate consensus rules: |sum of s| / 7 >= 0.99 gives every logged round end and no earlier row (an interface property, no behavior metric). No simulated target compared with human targets.
- Oct 29: P24 pilot (`scripts/p24_pilot.py`, L1, 10 replicates, seed 573): only the SD over replicates and the human CI half-widths were looked at (`results/tables/p24_pilot.csv`, no simulated means, no human point values). R = 200 (cap): Star CC success rate has a zero-width human CI (R needed = inf) and the Full CC / Star CC 10a CIs have width ~1e-15; without them R is still 200 (target 2 Hierarchy CL needs 628). Smoke test of `p24_main.py` / `p24_checks.py` on L1 (R = 3, seed 1001, outputs deleted): one line of its report was seen by mistake (L1 target 5 misses in 2 of 2 Full cells at R = 3), plus the word TRIGGERED for one of L1b / L3; no code changed because of it.
- Oct 29, 21:40 CDT: P24 zero-width CI rule, **decided before results** (L1 main run started, no replicate saved yet; from the human tables only), in `scripts/p24_checks.py` (`zero_width_rule`). A human CI narrower than 1e-9: a proportion at 0 or 1 gets the Clopper-Pearson 95% interval over the cell's rounds (all successes: [0.025^(1/n), 1]); 10a pinned by the interface gets point +/- 0.05 s (half the 0.1 s lab clock step); any other zero-width CI stops the script. It changes 3 statistics: success rate Star CC (100%, n = 39) [100, 100] -> [90.97, 100]; 10a Full CC and Star CC (0.6 s) [0.6, 0.6] -> [0.55, 0.65]. Resolution check of the 53 + 10c: two more CIs are exactly one resolution step wide, not changed: f_rev median Hierarchy CC (width 0.025 = one A step 0.1 / 4) and 10a Hierarchy CC (width 0.1 = one lab clock step).
- Oct 30: P24 main runs done (L1, L1_M3, L0, R = 200; `results/p24_report.txt`, `results/tables/p24_checks.csv`): Rescue not triggered, L1b and L3 triggered, so the pre-registered final model is L1 + L1b + L3. `results/tables/p24_directions.csv` (descriptive): L1 sim mean, human value, direction and gap in CI half-widths.
- Oct 30: P25 final model (`notebooks/path1/p25_final.ipynb`), fit on G-train only. L1b (`src/slider/halts.py`, `results/tables/p25_halts.csv`, `results/figures/p25_halts.png`): two-type geometric halt mixture per interface, p = 0.915 / 0.893 / 0.947 (lab v09 / old CL / CC), q_f = 0.305 / 0.333 / 0.129, q_s = 0.0184 / 0.0276 / 0.0062 per tick. L3 (`src/slider/frailty.py`, `results/tables/p25_sigma.csv`): sigma_g = 0.238, profile 95% CI [0.185, 0.312] (39 groups; M2' frozen as offset). Simulator preset `final` (`sim.py`, `design.py`); L1 unchanged (bit-identical). Build check (G6train, R = 2, seed 573; `results/tables/p25_recovery.csv`): L3 1/1, L1b 8/9 (old CL q_s 0.0342 vs [0.0262, 0.0318]), M2' lab-v09 plain refit 3/6. Diagnosed, not fixed (`scripts/p25_diagnose.py`): the plain refit omits u_g (with the drawn u_g as offset it passes 6/6); old CL counts reversal exits as resumes on the 0.02 grid (6.1% of simulated resume exits flip direction). Proposals in the notebook.
- Oct 30: P25 build-check decisions (user): (1) M2' under L3 is checked by the refit with each group copy's drawn u_g as an offset (6/6); limitation: the human M2' was a plain fit, so adding u_g on top partly double-counts between-group heterogeneity; reported, not changed. (2) Old CL q_s: a known limitation (the k-grid blind spot, as in P23); P17 unchanged. The final model is frozen. Final main run done (R = 200, seed 4001, Oct 30 13:19-14:17); `p24_checks.py`: final misses 37 of 53 (`results/p24_report.txt`).
- Oct 30: P25 visuals (`notebooks/path1/p25_visuals.ipynb`, `src/slider/visuals.py`, `results/figures/p25_vis_fig1-5.png`), descriptive only, train + val, Hierarchy (EL, CL, CC) and Full (CL, CC): check overview, averaged normalized trajectories (port of the authors' runapem.m averaging analysis), time to consensus (KM), Agreement at reversal, example rounds. final re-simulated (replicates 0-19, seed-4001 batch 0 streams; fast-path statistics exactly equal to `p24_reps_final.parquet`, 20 of 20) and the published APEM on our interface (a = 0.5, s = 0.5, mu = 0.1; replicates 0-19, seed [5001, 0, 0]). No decisions, no refits.
- Oct 30: Model B, B0 (`notebooks/path1/pB0_stops.ipynb`, `src/slider/stop_states.py`, `results/tables/pB0_stop_by_state.csv`, `results/figures/pB0_stop_by_position.png`), descriptive only, G-train only: stop chance per counted moving step (motion.py F1b) by phase (first_trip / outbound / in_band), reversed since the phase began, outbound outcome and progress s * phi. All-steps row equals P17's lam_stop * dt (stops equal seat by seat). No decisions.
- Oct 30: Model B, B0b + B2 (`notebooks/path1/pB0b_pB2.ipynb`, `src/slider/model_b.py`, `results/tables/pB0b_crossing_style.csv`, `pB2_try_keep.csv`, `results/figures/pB0b_player_smooth.png`, `pB2_keep_by_dA.png`), descriptive only, G-train only: crossing style per player (smooth share, permutation SD test, beta-binomial pi / rho) and P(kept) of tries (detect_events m = 2 reversal vs. probe) by the sign of dA, dA_own, hold >= 0.2 s and topology. No decisions.

## Data source

OSF project **gvcpz**: <https://osf.io/gvcpz>. Download the archive (`gvcpz-osfstorage-archive`) and move it, unchanged, to `data/raw/osf/`, so that the folder holds `Code/` and `Data/`.
`data/raw/` is not in git. The first notebook unzips the two log archives in `Data/` if needed.

## Layout

```
data/
  raw/osf/            OSF archive gvcpz, unchanged (Code/ + Data/)
  processed/          runs_table.csv (HW 1), runs_fraught.csv (HW 3)
  splits/             split_G.csv: split G (group_id, dataset, train / val)
src/slider/           the package: from slider import config
  config.py           all paths, SEED, train/test topologies
  slider_logs.py      log parser
  fraught.py          fraught tests, % fraught measures, fraught spells (spells)
  stats.py            bootstrap / Wilcoxon test, summary tables, Kaplan-Meier (kaplan_meier)
  splits.py           check_cc_groups, make_split_G (python -m slider.splits)
  h5.py               HW 6 H5: committed-split spells, breakers, sensitivity grid; h5.run(roles) = frozen analysis
  targets.py          HW 6 training targets per topology x dataset x role, group-bootstrap CIs
  models/             apem.py (APEM simulator), apem_osf.py (port of the authors' Matlab code)
notebooks/
  stage0/             stage0_hw5_fraught_corners.ipynb
  stage1/             stage1_hw1 … stage1_hw5
  hw6/                h5_obdurate.ipynb (train), h5_val.ipynb (val, run once), train_targets.ipynb (train + val)
  path1/ path2/
results/
  tables/             ring_traps.csv (HW 4), apem_sweep.csv, apem_topologies.csv (HW 5), h5_train.csv, h5_train_checks.csv, h5_val.csv, train_targets.csv (HW 6)
  figures/
scripts/              p24_pilot.py, p24_main.py (resumable main runs), p24_checks.py (checks + triggers), p24_directions.py, p25_diagnose.py
tests/test_sanity.py
environment.yml       conda env "elec573"
```

All paths come from `src/slider/config.py`. Do not hard-code paths anywhere else.

## How to run

From the `Tabor/` folder, once: create the conda env (this also runs `pip install -e .`).

```bash
conda env create -f environment.yml
```

Then register it as a Jupyter kernel. The notebooks use the kernel `elec573`.

```bash
conda run -n elec573 python -m ipykernel install --user --name elec573 --display-name "Python (elec573)"
```

Every time:

```bash
conda activate elec573
```

```bash
pytest
```

Imports in notebooks: `from slider import config`, `from slider.fraught import fraught_mask`, `from slider.models.apem import apem`.

Run the notebooks in order: `stage0`, then `stage1_hw1` (writes `runs_table.csv`), `hw2`, `hw3` (writes `runs_fraught.csv`), `hw4`, `hw5` (about 45 minutes).

`pytest` checks the parser and the fraught measures against known numbers: 718 runs, recomputed Agreement within 0.005 of the log, CL Hierarchy mean time to consensus 61.61 s and mean 11.58 % fraught (forward hold), and no fraught rows on Full.

Seeds: the Stage 0 and Stage 1 notebooks use seed 0. New work uses `config.SEED` (573).

## New analyses: load runs with `load_runs`

New code gets its runs from `slider_logs.load_runs`, not from `parse_all`. The Stage 0 and Stage 1 notebooks are left as they were.

```python
from slider import slider_logs as sl
runs, table = sl.load_runs("outcome")                         # train runs only (default)
runs, table = sl.load_runs("behavior", roles={"train", "val"}, keep_whole_second=True)
```

- Role of a run: Ring / CT-Hierarchy → `test` (any group); Hierarchy / Full / Star → the group's label in `split_G.csv` (`train` / `val`); any other graph → `other` (never returned).
- `purpose="outcome"` drops aborted 1-row rounds, runs with no end marker and runs with a bot. `purpose="behavior"` drops aborted 1-row rounds and runs in whole-second files. Both drop runs with fewer than 2 logged rows (unplayed waves). Override with `keep_aborted`, `keep_incomplete`, `keep_bot`, `keep_whole_second`, `keep_empty`.
- Added columns: `role`, `has_bot`, `bot_players` (robot seats), `coarse_time` (whole-second file).
- Asking for test runs raises `HeldOutError`. With `allow_test=True` you must give a `reason`, and the call is logged in `results/test_access_log.csv`.

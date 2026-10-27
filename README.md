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

# ELEC573 — Slider Game / fraught states (Tabor)

Analysis of the Slider Game logs from Tabor, Smith & Dankowicz, *Escaping fraught states* (PsyArXiv 2023).

## Rule

**No new metrics on the test topologies (Ring, CT-Hierarchy) before Nov 25 (2026).**
New work uses the train topologies only (Hierarchy, Full, Star). This split (split T) is `TRAIN_TOPOLOGIES` / `TEST_TOPOLOGIES` in `src/slider/config.py`.
Split G splits the groups (one group = one log file) into train / val, 1/3 val within each of EL, CL and CC: `data/splits/split_G.csv`.

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
  fraught.py          fraught tests and % fraught measures
  stats.py            bootstrap / Wilcoxon test, summary tables
  splits.py           check_cc_groups, make_split_G (python -m slider.splits)
  models/             apem.py (APEM simulator), apem_osf.py (port of the authors' Matlab code)
notebooks/
  stage0/             stage0_hw5_fraught_corners.ipynb
  stage1/             stage1_hw1 … stage1_hw5
  hw6/ path1/ path2/
results/
  tables/             ring_traps.csv (HW 4), apem_sweep.csv, apem_topologies.csv (HW 5)
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

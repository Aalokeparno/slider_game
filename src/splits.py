"""Train/test split by topology (graph G), saved as data/splits/split_G.csv.

Run once (after runs_table.csv exists):  python -m splits
"""
import pandas as pd

import config


def split_G(table):
    """One row per run of runs_table.csv, with split = "train", "test" or "other" (topology in neither set)."""
    out = table[["file", "wave", "dataset", "topology"]].copy()
    out["split"] = "other"
    out.loc[out["topology"].isin(config.TRAIN_TOPOLOGIES), "split"] = "train"
    out.loc[out["topology"].isin(config.TEST_TOPOLOGIES), "split"] = "test"
    return out


if __name__ == "__main__":
    split = split_G(pd.read_csv(config.RUNS_TABLE))
    config.SPLITS.mkdir(parents=True, exist_ok=True)
    split.to_csv(config.SPLIT_G, index=False)
    print(split["split"].value_counts())

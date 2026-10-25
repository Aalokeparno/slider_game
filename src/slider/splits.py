"""Splits of the runs.

Split T (by topology) is config.TRAIN_TOPOLOGIES / config.TEST_TOPOLOGIES.
Split G (by group = one log file) is made here and saved as data/splits/split_G.csv.

Run:  python -m slider.splits
It checks the CC files first and writes split_G.csv only if no file looks like two groups.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

from slider import config
from slider import slider_logs as sl


def check_cc_groups():
    """For each CC log file: rounds, rounds per topology, how many times the demographic
    questions block appears, and whether the game numbers restart. Returns the flagged files."""
    lab, cloud = sl.log_files()      # tmp.txt is not in the list
    rows = []
    for path in cloud:
        text = path.read_text(errors="replace")
        runs = sl.parse_cloud(path)
        topologies = Counter(sl.label_topology(run["A"]) for run in runs)
        headers = text.count("<Questions>")                          # demographic questions (gender, age, played before)
        answers = text.count("What is your gender? ")                # one per player who answered
        games = [int(k) for k in re.findall(r"Starting Game (\d+)\s*$", text, re.M)]
        restarts = any(b <= a for a, b in zip(games, games[1:]))     # e.g. 1 2 ... 9 1 2 ...
        rows.append({"file": path.name, "rounds": len(runs),
                     "per topology": ", ".join(f"{t} {k}" for t, k in sorted(topologies.items())),
                     "header > 1": headers > 1, "players answered": answers,
                     "games restart": restarts, "two groups?": headers > 1 or restarts})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return list(table.loc[table["two groups?"], "file"])


def make_split_G(runs, seed=config.SEED, frac=1/3):
    """Split G: group_id = log file name. Groups in EL, CL and CC with 7 players only.
    Within each dataset, round(frac * n) groups are drawn at random as "val", the rest are "train".
    Only the file names go into the draw (no outcome). Saves config.SPLIT_G; will not overwrite it."""
    if config.SPLIT_G.exists():
        raise FileExistsError(f"{config.SPLIT_G} already exists; delete it first to make a new split")

    dataset = {}      # file -> dataset
    players = {}      # file -> set of player counts in its rounds
    for run in runs:
        dataset[run["file"]] = run["dataset"]
        players.setdefault(run["file"], set()).add(len(run["weights"]))

    rng = np.random.default_rng(seed)
    rows = []
    for ds in ["EL", "CL", "CC"]:
        groups = sorted(f for f in dataset if dataset[f] == ds and players[f] == {7} and f != "tmp.txt")
        val = set(rng.choice(groups, size=round(frac * len(groups)), replace=False))
        for f in groups:
            rows.append({"group_id": f, "dataset": ds, "split": "val" if f in val else "train"})

    split = pd.DataFrame(rows)
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split.to_csv(config.SPLIT_G, index=False)
    return split


if __name__ == "__main__":
    flagged = check_cc_groups()
    if flagged:
        print("\nThese CC files look like two groups:", flagged)
        print("split_G.csv was not written.")
    else:
        print("\nNo CC file looks like two groups.\n")
        split = make_split_G(sl.parse_all())
        print(pd.crosstab(split["dataset"], split["split"], margins=True))

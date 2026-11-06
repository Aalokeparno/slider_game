"""Where do sliders start? Start positions, first moves and spontaneous starters in the human rounds.
Read-only: raw logs are read through slider_logs.load_runs; nothing in data/ is written.

Rounds: load_runs("behavior", roles={"train", "val"}, keep_whole_second=True), i.e. Hierarchy / Full / Star,
all datasets, aborted 1-row rounds and rounds with < 2 rows dropped, bot and incomplete rounds kept.
Ring and CT-Hierarchy are test rounds and are only loaded with --include-test (logged in test_access_log.csv).

Definitions
    s          sign-corrected slider position, s = weights * X (slider_logs.sign_correct). X is logged in [-1, 1]
               already (lab: minContribution=-1, maxContribution=1; cloud: the same scale), so no unit change.
    start      the first logged row of the round (row 0); its time t[0] is reported as t_first.
    first move first row k > 0 where s_i[k] != s_i[k-1] (|change| > 1e-9); first_move_time = t[k].
               A player who never moves has NaN.
    first mover the human seat(s) with the smallest first_move_time (ties are all counted as first movers).
    spontaneous a human seat whose first move is strictly earlier than the first move of every neighbour
               (round's A; a neighbour who never moves counts as never). A neighbour moving in the same row
               is a tie, which is NOT spontaneous; ties are counted separately.
    Robot seats (CC, bot_players) keep their positions and count as neighbours, but are never first movers or
    spontaneous starters.

    python scripts/check_start_positions.py                 -> results/path3/start_positions_rounds.csv,
                                                               results/path3/start_positions_summary.csv
    python scripts/check_start_positions.py --include-test  (also Ring / CT-Hierarchy; asks load_runs for test runs)
"""
import sys

import numpy as np
import pandas as pd

from slider import config
from slider import slider_logs as sl

INCLUDE_TEST = "--include-test" in sys.argv
OUT_ROUNDS = config.PATH3 / "start_positions_rounds.csv"
OUT_SUMMARY = config.PATH3 / "start_positions_summary.csv"
ORDER = ["CL v09", "EL", "CL oldCL", "CL whole_second", "CC", "CC no-bot"]
NETWORKS = ["Hierarchy", "Full", "Ring", "CT-Hierarchy", "Star"]


def dataset_name(file, dataset):
    """CL v09 / CL oldCL / CL whole_second / EL / CC (slider_logs.file_version)."""
    version = sl.file_version(file, dataset)
    return "CL " + version if dataset == "CL" else dataset


def round_row(run, topology):
    """One row per round: start values, start agreement, first-move times, first mover(s), spontaneous starters."""
    s = sl.sign_correct(run)
    t = run["t"]
    n = s.shape[1]
    humans = [j for j in range(n) if j + 1 not in run["bot_players"]]

    first = np.full(n, np.nan)
    for j in range(n):
        moved = np.flatnonzero(np.abs(np.diff(s[:, j])) > 1e-9)
        if len(moved) > 0:
            first[j] = t[moved[0] + 1]

    first_inf = np.where(np.isnan(first), np.inf, first)
    spont, ties = [], []
    for j in humans:
        if np.isnan(first[j]):
            continue
        nb = np.flatnonzero(run["A"][j])
        earliest_nb = first_inf[nb].min()
        if first[j] < earliest_nb:
            spont.append(j + 1)
        elif first[j] == earliest_nb:
            ties.append(j + 1)

    human_first = first[humans]
    t_lead = np.nanmin(human_first) if np.any(~np.isnan(human_first)) else np.nan
    leaders = [j + 1 for j in humans if first[j] == t_lead]

    s0 = s[0]
    row = {"file": run["file"], "wave": run["wave"], "dataset": dataset_name(run["file"], run["dataset"]),
           "network": topology, "has_bot": run["bot"], "bot_seats": " ".join(str(b) for b in sorted(run["bot_players"])),
           "end": run["end"], "rows": len(t), "t_first": t[0],
           "all_abs_lt_0.05": bool(np.all(np.abs(s0) < 0.05)), "all_zero": bool(np.all(np.abs(s0) < 1e-9)),
           "all_on_grid_0.1": bool(np.all(np.abs(10 * s0 - np.round(10 * s0)) <= 1e-6)),
           "agree0_mean": run["agree"][0].mean(), "agree0_max": run["agree"][0].max(),
           "general0": run["general"][0],
           "first_mover_time": t_lead, "first_movers": " ".join(map(str, leaders)),
           "n_never_moved": int(np.sum(np.isnan(human_first))),
           "n_spontaneous": len(spont), "spontaneous_seats": " ".join(map(str, spont)),
           "n_ties_with_neighbour": len(ties)}
    for j in range(n):
        row[f"s0_{j + 1}"] = s0[j]
    for j in range(n):
        row[f"first_move_{j + 1}"] = first[j] if j in humans else np.nan
    return row


def q(x, p):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return np.percentile(x, p) if len(x) > 0 else np.nan


def summary_row(name, network, rounds):
    """Summary of a set of rounds (rows of the rounds table)."""
    s0 = rounds[[c for c in rounds.columns if c.startswith("s0_")]].to_numpy().ravel()
    s0 = s0[~np.isnan(s0)]
    ft = rounds[[c for c in rounds.columns if c.startswith("first_move_")]].to_numpy().ravel()
    n_sp = rounds["n_spontaneous"]
    return {"dataset": name, "network": network, "n_rounds": len(rounds),
            "pct_all_zero": 100 * rounds["all_zero"].mean(),
            "pct_all_abs_lt_0.05": 100 * rounds["all_abs_lt_0.05"].mean(),
            "pct_all_on_grid": 100 * rounds["all_on_grid_0.1"].mean(),
            "s0_min": s0.min(), "s0_median": np.median(s0), "s0_max": s0.max(),
            "s0_values": "; ".join(f"{v:g}: {c}" for v, c in zip(*np.unique(np.round(s0, 2), return_counts=True))),
            "agree0_max": rounds["agree0_max"].max(),
            "t_first_median": rounds["t_first"].median(), "t_first_max": rounds["t_first"].max(),
            "first_mover_median": q(rounds["first_mover_time"], 50),
            "first_mover_q25": q(rounds["first_mover_time"], 25), "first_mover_q75": q(rounds["first_mover_time"], 75),
            "all_players_median": q(ft, 50), "all_players_q25": q(ft, 25), "all_players_q75": q(ft, 75),
            "n_human_never_moved": int(rounds["n_never_moved"].sum()),
            "spont_mean": n_sp.mean(),
            "spont_0": int((n_sp == 0).sum()), "spont_1": int((n_sp == 1).sum()),
            "spont_2": int((n_sp == 2).sum()), "spont_3plus": int((n_sp >= 3).sum()),
            "ties_mean": rounds["n_ties_with_neighbour"].mean(),
            "pct_multi_first_mover": 100 * (rounds["first_movers"].str.split().str.len() > 1).mean()}


roles = {"train", "val", "test"} if INCLUDE_TEST else {"train", "val"}
if INCLUDE_TEST:
    runs, table = sl.load_runs("behavior", roles=roles, keep_whole_second=True, allow_test=True,
                               reason="Start positions check (descriptive, start of round only)")
else:
    runs, table = sl.load_runs("behavior", roles=roles, keep_whole_second=True)

rounds = pd.DataFrame([round_row(run, top) for run, top in zip(runs, table["topology"])])
rounds.insert(4, "role", table["role"].to_numpy())

summary = []
for name in ORDER:
    if name == "CC no-bot":
        part = rounds[(rounds["dataset"] == "CC") & ~rounds["has_bot"]]
    else:
        part = rounds[rounds["dataset"] == name]
    if len(part) == 0:
        continue
    summary.append(summary_row(name, "all", part))
    for net in NETWORKS:
        if (part["network"] == net).any():
            summary.append(summary_row(name, net, part[part["network"] == net]))
summary = pd.DataFrame(summary)

config.PATH3.mkdir(parents=True, exist_ok=True)
rounds.to_csv(OUT_ROUNDS, index=False)
summary.to_csv(OUT_SUMMARY, index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
fmt = lambda x: f"{x:.2f}"
print(f"{len(rounds)} rounds ({', '.join(sorted(roles))})\n")
print(summary[["dataset", "network", "n_rounds", "pct_all_zero", "pct_all_abs_lt_0.05", "pct_all_on_grid",
               "s0_min", "s0_median", "s0_max", "agree0_max", "t_first_median", "t_first_max"]]
      .to_string(index=False, float_format=fmt))
print()
print(summary[["dataset", "network", "first_mover_median", "first_mover_q25", "first_mover_q75",
               "all_players_median", "all_players_q25", "all_players_q75", "n_human_never_moved",
               "pct_multi_first_mover"]].to_string(index=False, float_format=fmt))
print()
print(summary[["dataset", "network", "spont_mean", "spont_0", "spont_1", "spont_2", "spont_3plus", "ties_mean"]]
      .to_string(index=False, float_format=fmt))
print("\ns0 values (value: count):")
for _, r in summary[summary["network"] == "all"].iterrows():
    print(f"  {r['dataset']}: {r['s0_values']}")

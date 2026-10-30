"""Path A, P25 visuals (descriptive only): humans vs. the final model and the published APEM on the training cells
Hierarchy (EL, CL, CC) and Full (CL, CC). Used by notebooks/path1/p25_visuals.ipynb.

Re-simulation uses the main-run streams: batch b of a preset holds replicates 20 b ... 20 b + 19 and is simulated
with seed [seed, b, 20 b] on the full L3 design (scripts/p24_main.py), so replicate r here is replicate r of
results/tables/p24_reps_<preset>.parquet. simulate_rows simulates only some rounds of a design with the streams
they get in the full design (design.simulate_rounds: round i uses SeedSequence(seed).spawn(len(design))[i]; the L3
u values come from the next spawned stream, over all the design's groups).

Averaging analysis: a port of the authors' runapem.m (OSF gvcpz, Code/runapem.m), as fraught_percent_interp does:
    time warp   linear interpolation of the sign-corrected positions at 1001 evenly spaced times from the first to
                the last row (nwarp = 1000); normalized time (0..1000) / 1001
    selection   Hierarchy: successful rounds with % fraught > 10 on that grid (fraught.fraught_percent_interp);
                Full: successful rounds
    flip        so that the winning side is +1 (sign of the sum of the final positions)
    branches    nodes of apem.GRAPHS (= the authors' adjacency): Apex {1}, Left Branch {2, 4, 5}, Right Branch
                {3, 6, 7} (1-based); if the right branch's mean position is below the left's, the two are swapped,
                so the Left Branch is always the one that reverses
    Agreement   |A_self s| / (degree + 1) per player (getagreelaopt), in [0, 1]
    plot        mean over rounds, then over the players of each group; the Left Branch's Agreement is negated
                (Hierarchy); Full shows only the Apex Agreement
"""
import numpy as np
import pandas as pd

from slider import config
from slider import design as dz
from slider import sim
from slider import slider_logs as sl
from slider import targets
from slider.fraught import fraught_percent_interp
from slider.models.apem import GRAPHS
from slider.stats import kaplan_meier

CELLS = [("Hierarchy", "EL"), ("Hierarchy", "CL"), ("Hierarchy", "CC"), ("Full", "CL"), ("Full", "CC")]
NWARP = 1000
APEX, LEFT, RIGHT = [0], [1, 3, 4], [2, 5, 6]                    # 0-based graph nodes
SWITCHED = [0, 2, 1, 5, 6, 3, 4]                                  # runapem.m switchedorder, 0-based
APEM_PUBLISHED = {"a": 0.5, "s_th": 0.5, "mu": 0.1}               # runapem.m: atheta, stheta, meanstep


def batch_seed(seed, b):
    """The seed of batch b in scripts/p24_main.py (a fresh run: first replicate 20 b)."""
    return [seed, b, dz.BATCH * b]


def simulate_rows(design, rows, preset, R, seed, **apem):
    """design.simulate_rounds for the rounds at positions `rows` only, each with the streams it gets in the full
    design. Returns (design.iloc[rows], rounds) for design.copy_runs."""
    model = sim.preset(preset, **apem)
    sigma = model.pop("sigma_g", 0.0)
    ss = np.random.SeedSequence(seed)
    streams = ss.spawn(len(design))
    u = None
    if sigma > 0:
        u_rng = np.random.default_rng(ss.spawn(1)[0])
        u = {g: u_rng.normal(0, sigma, R) for g in sorted(design["group"].unique())}
    sub = design.iloc[list(rows)].reset_index(drop=True)
    rounds = []
    for i, row in zip(rows, sub.itertuples()):
        sim_seed, log_seed = streams[i].spawn(2)
        A = row.graph[np.ix_(row.seat_node, row.seat_node)]
        rounds.append((sim.simulate(A, row.T, row.interface, R=R, seed=sim_seed, speed=row.speed,
                                    offset=None if u is None else u[row.group], **model),
                       np.random.default_rng(log_seed)))
    return sub, rounds


def in_cells(table):
    """Mask of the rows of a runs table in CELLS."""
    return pd.Series(list(zip(table["topology"], table["dataset"]))).isin(CELLS).to_numpy()


# ------------------------------------------------------------------ averaging analysis (runapem.m)

def node_positions(run, topology):
    """Sign-corrected positions with columns in graph-node order (design.seat_map)."""
    _, seat_node = dz.seat_map(run["A"], topology)
    s = sl.sign_correct(run)
    out = np.zeros_like(s)
    out[:, list(seat_node)] = s
    return out


def warped(run, topology):
    """(positions, Agreement) on the runapem.m grid: 1001 evenly spaced times from the first to the last row,
    linear interpolation; flipped so the winner is +1, branches swapped so the Left Branch reverses."""
    t, s = run["t"], node_positions(run, topology)
    grid = np.linspace(t[0], t[-1], NWARP + 1)
    x = np.column_stack([np.interp(grid, t, s[:, j]) for j in range(7)])
    if x[-1].sum() < 0:
        x = -x
    if x[:, RIGHT].mean() < x[:, LEFT].mean():
        x = x[:, SWITCHED]
    A_self = GRAPHS[topology].astype(float) + np.eye(7)
    agree = np.abs(x @ A_self.T) / A_self.sum(axis=1)
    return x, agree


def qualifies(run, success, topology):
    """runapem.m selection: successful; Hierarchy also % fraught > 10 (interpolated, 1001 points)."""
    if not success:
        return False
    return topology != "Hierarchy" or fraught_percent_interp(0, [run]) > 10


def group_means(x):
    """Mean over the players of Apex, Left Branch, Right Branch: an (n_times, 3) array."""
    return np.column_stack([x[:, g].mean(axis=1) for g in (APEX, LEFT, RIGHT)])


# ------------------------------------------------------------------ what the figures need, per set of runs

def trajectories(runs3, table3, cells):
    """{cell: [sum of warped positions, sum of warped Agreement, n]} over the qualifying rounds of cells."""
    t3 = table3.reset_index(drop=True)
    traj = {}
    for k, run in enumerate(runs3):
        cell = (t3["topology"][k], t3["dataset"][k])
        if cell in cells and qualifies(run, t3["success"][k], cell[0]):
            x, a = warped(run, cell[0])
            acc = traj.setdefault(cell, [0.0, 0.0, 0])
            acc[0], acc[1], acc[2] = acc[0] + x, acc[1] + a, acc[2] + 1
    return traj


def first_fraught(runs3, table3, cell):
    """(run, info) of the first round of cell with % fraught > 10 (interpolated, as in the selection), or None."""
    t3 = table3.reset_index(drop=True)
    for k, run in enumerate(runs3):
        if (t3["topology"][k], t3["dataset"][k]) == cell and fraught_percent_interp(0, [run]) > 10:
            return run, {"file": run["file"], "wave": run["wave"], "success": bool(t3["success"][k]),
                         "fraught_pct": fraught_percent_interp(0, [run])}
    return None


def collect(runs3, table3, runs5, traj_cells=(("Hierarchy", "CL"), ("Full", "CL")), keep_first=None):
    """From one set of rounds (humans, or one simulated replicate):
    km      one row per round in CELLS: topology, dataset, ttc (s; failures at the time limit), event (success)
    rev     Agreement at reversal (target 9: A at the row before the reversal), from runs5, in CELLS
    traj    trajectories of traj_cells
    first   first_fraught of keep_first (a cell), or None
    """
    t3 = table3.reset_index(drop=True)
    limit = t3["session"].map(config.TIME_LIMIT)
    km = pd.DataFrame({"topology": t3["topology"], "dataset": t3["dataset"],
                       "ttc": np.where(t3["success"], t3["duration"], limit), "event": t3["success"].astype(int)})
    km = km[in_cells(t3)]
    rev, _, _ = targets.agreement_values(runs5, samples=False)
    rev = rev[pd.Series(list(zip(rev["topology"], rev["dataset"]))).isin(CELLS).to_numpy()]
    first = first_fraught(runs3, t3, keep_first) if keep_first else None
    return {"km": km, "rev": rev[["topology", "dataset", "file", "A"]], "traj": trajectories(runs3, t3, traj_cells),
            "first": first}


def merge(parts):
    """Pool collect() outputs (e.g. over replicates)."""
    traj = {}
    for p in parts:
        for cell, (x, a, n) in p["traj"].items():
            acc = traj.setdefault(cell, [0.0, 0.0, 0])
            acc[0], acc[1], acc[2] = acc[0] + x, acc[1] + a, acc[2] + n
    return {"km": pd.concat([p["km"] for p in parts], ignore_index=True),
            "rev": pd.concat([p["rev"] for p in parts], ignore_index=True), "traj": traj}


def km_band(ttc, event):
    """Kaplan-Meier (stats.kaplan_meier) with a Greenwood 95% band: time, survival, lo, hi, and the median."""
    km, median = kaplan_meier(ttc, event)
    d, n = km["escapes"].to_numpy(float), km["at_risk"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = km["survival"].to_numpy() ** 2 * np.cumsum(np.where(n > d, d / (n * (n - d)), 0.0))
    se = np.sqrt(var)
    return km.assign(lo=np.clip(km["survival"] - 1.96 * se, 0, 1), hi=np.clip(km["survival"] + 1.96 * se, 0, 1)), median

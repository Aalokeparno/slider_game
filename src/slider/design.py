"""Path A, Step 4: the design of a simulation, one row per real round, the simulation of a design, and the
check statistics of simulated replicates (targets.check_stats, in parallel processes).

Run sets (slider_logs.load_runs; test runs are never loaded):
    L3       load_runs("outcome", roles={"train", "val"})                                   targets 1-7
    L5       load_runs("behavior", roles={"train", "val"}, keep_bot=False, keep_incomplete=False)  targets 8-10
    G6train  load_runs("behavior"): the G-train rounds of the hazard table, bot and incomplete runs included
             (they are simulated all-human)

Speed (units/s), as in hazard.build_hazard_table:
    lab rounds   the round's own speed (interface.step_and_pace: step size x modal pace), or the file's modal
                 speed if the round has none
    CC rounds    the file's modal speed (its cap)
    whole-second old-CL files: 0.2 (their timestamps give no pace)
File speeds are the P7e values in results/tables/interface_table.csv. Step d is the interface's grid step
(sim.IFACES: 0.1, old CL 0.02) and dt = d / speed.
"""
import time
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np
import pandas as pd

from slider import config
from slider import interface
from slider import sim
from slider import slider_logs as sl
from slider import targets
from slider.models.apem import GRAPHS

SETS = {
    "L3": {"purpose": "outcome", "roles": {"train", "val"}},
    "L5": {"purpose": "behavior", "roles": {"train", "val"}, "keep_bot": False, "keep_incomplete": False},
    "G6train": {"purpose": "behavior"},
}
WHOLE_SECOND_SPEED = 0.2


def load(run_set):
    """(runs, table) of a run set (SETS)."""
    return sl.load_runs(**SETS[run_set])


def seat_map(A, topology):
    """(graph, seat_node): the topology's graph (apem.GRAPHS) and a map with graph[seat_node][:, seat_node] = A."""
    graph = GRAPHS[topology].astype(int)
    match = nx.algorithms.isomorphism.GraphMatcher(nx.from_numpy_array(A), nx.from_numpy_array(graph))
    if not match.is_isomorphic():
        raise ValueError(f"the round's adjacency is not the {topology} graph")
    return graph, tuple(match.mapping[seat] for seat in range(7))


def design(run_set):
    """One row per round of a run set: group, dataset, role, topology, wave, round_id, graph, seat_node, interface,
    d, speed, dt, T (time limit), lab (1 = EL / CL), whole_second."""
    runs, table = load(run_set)
    files = pd.read_csv(config.INTERFACE_TABLE).set_index("file")
    rows = []
    for run, (_, r) in zip(runs, table.iterrows()):
        whole = bool(r["coarse_time"])
        if run["dataset"] == "CC":
            iface, speed = "cc", files.loc[run["file"], "speed"]
        else:
            iface = "lab_v09" if interface.version_tag(run["file"]) == "v09" else "old_cl"
            if whole:
                speed = WHOLE_SECOND_SPEED
            else:
                step, pace = interface.step_and_pace(interface.moves(run))
                speed = files.loc[run["file"], "speed"] if np.isnan(step * pace) else step * pace
        graph, seat_node = seat_map(run["A"], r["topology"])
        d = sim.IFACES[iface]["d"]
        rows.append({"group": run["file"], "dataset": run["dataset"], "role": r["role"], "topology": r["topology"],
                     "wave": run["wave"], "round_id": f"{run['file']}#w{run['wave']}", "graph": graph,
                     "seat_node": seat_node, "interface": iface, "d": d, "speed": speed, "dt": d / speed,
                     "T": config.TIME_LIMIT[run["session"]], "lab": int(run["dataset"] != "CC"), "whole_second": whole})
    return pd.DataFrame(rows)


def simulate_rounds(design, preset, R=1, seed=None, **apem):
    """sim.simulate of every round of a design, R copies at once (one call per round): a list, one
    (outputs, log_rng) per round. copy_runs turns copy r into runs. preset: sim.preset name (APEM takes a, s_th,
    mu; final takes sigma_g). Round i uses SeedSequence(seed).spawn(len(design))[i], split into a simulation and a
    logger stream.
    L3 (preset final, sigma_g > 0): copy r of group g draws u ~ N(0, sigma_g^2) once, from the next spawned stream
    (groups in sorted order, R values each), and every round of that group copy gets it as its offset; each
    output's "u" holds it. sigma_g = 0: no draws, no offset."""
    model = sim.preset(preset, **apem)
    sigma = model.pop("sigma_g", 0.0)
    ss = np.random.SeedSequence(seed)
    streams = ss.spawn(len(design))
    u = None
    if sigma > 0:
        u_rng = np.random.default_rng(ss.spawn(1)[0])
        u = {g: u_rng.normal(0, sigma, R) for g in sorted(design["group"].unique())}
    out = []
    for row, stream in zip(design.itertuples(), streams):
        sim_seed, log_seed = stream.spawn(2)
        A = row.graph[np.ix_(row.seat_node, row.seat_node)]
        out.append((sim.simulate(A, row.T, row.interface, R=R, seed=sim_seed, speed=row.speed,
                                 offset=None if u is None else u[row.group], **model),
                    np.random.default_rng(log_seed)))
    return out


def copy_runs(design, rounds, r, seed=None):
    """Copy r of every round of simulate_rounds as runs (like load_runs; all seats human), in design order.

    Each round keeps its dataset, topology, wave, graph, seat map, interface and T; copy r goes into group
    sim<r>_<group>, so each copy of a file holds the same rounds. Logger: whole_second for whole-second rounds,
    cloud for CC, else lab. Each round's logger stream is used in copy order: call r = 0, 1, 2, ... in turn.
    """
    runs = []
    for row, (out, log_rng) in zip(design.itertuples(), rounds):
        o = out[r]
        logger = "whole_second" if row.whole_second else "cloud" if row.interface == "cc" else "lab"
        meta = {"group": f"sim{r}_{row.group}", "dataset": row.dataset, "topology": row.topology, "wave": row.wave,
                "graph": row.graph, "seat_node": row.seat_node, "interface": row.interface, "time_limit": row.T,
                "end": "success" if o["success"] else "failed", "seed": seed}
        runs.append(sim.to_run(o["events"], meta, logger, log_rng))
    return runs


def simulate_design(design, preset, R=1, seed=None, **apem):
    """R simulated copies of every round of a design, as (runs, table) like load_runs, all seats human:
    simulate_rounds, then copy_runs for every copy. Runs in round order, the R copies of a round together.
    Each round keeps its dataset, role, topology, wave, graph, seat map, interface, speed and T."""
    rounds = simulate_rounds(design, preset, R, seed, **apem)
    copies = [copy_runs(design, rounds, r, seed) for r in range(R)]
    runs = [copies[r][i] for i in range(len(design)) for r in range(R)]
    return runs, sim.table(runs, list(np.repeat(design["role"].to_numpy(), R)))


BATCH = 20      # replicates per job: 21.9 s per replicate wall with 2 processes, 541 MB peak per process (Oct 29)


def check_job(job):
    """One batch for check_replicates: (d3, preset, R, seed, first replicate number, apem) -> check_stats rows.
    The R copies of each round are simulated at once (simulate_rounds); the runs of one copy at a time are then
    built, scored and dropped, so only the batch's position events stay in memory."""
    d3, preset, R, seed, first, apem = job
    rounds = simulate_rounds(d3, preset, R=R, seed=seed, **apem)
    keep5 = ~d3["whole_second"].to_numpy()                     # L5 = L3 without the whole-second rounds
    roles = list(d3["role"])
    out = []
    for r in range(R):
        runs3 = copy_runs(d3, rounds, r, seed)
        runs5 = [run for run, keep in zip(runs3, keep5) if keep]
        out.append(targets.check_stats(runs3, sim.table(runs3, roles), runs5).assign(replicate=first + r))
        for o, _ in rounds:
            o[r] = None                                         # copy r is done
    return pd.concat(out, ignore_index=True)


def check_replicates(d3, preset, n_rep, seed=None, workers=2, batch=BATCH, **apem):
    """targets.check_stats of n_rep simulated replicates of the L3 design d3 (targets 8-10 on each replicate's L5
    rounds), in `workers` processes, `batch` replicates per job (simulated together, one call per round).
    Batch b uses seed [seed, b]. Returns (stats with a replicate column, wall seconds per replicate)."""
    jobs = [(d3, preset, min(batch, n_rep - first), [seed or 0, b], first, apem)
            for b, first in enumerate(range(0, n_rep, batch))]
    start = time.time()
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(check_job, jobs))
    return pd.concat(parts, ignore_index=True), (time.time() - start) / n_rep

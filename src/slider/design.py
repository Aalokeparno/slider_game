"""Path A, Step 4: the design of a simulation, one row per real round, and the simulation of a design.

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
import networkx as nx
import numpy as np
import pandas as pd

from slider import config
from slider import interface
from slider import sim
from slider import slider_logs as sl
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


def simulate_design(design, preset, R=1, seed=None, **apem):
    """R simulated copies of every round of a design, as (runs, table) like load_runs, all seats human.

    Each round keeps its dataset, role, topology, wave, graph, seat map, interface, speed and T. preset: sim.preset
    name (APEM takes a, s_th, mu). Logger: whole_second for whole-second rounds, cloud for CC, else lab.
    Copy r of a round goes into group sim<r>_<group>, so each copy of a file holds the same rounds.
    """
    model = sim.preset(preset, **apem)
    streams = np.random.SeedSequence(seed).spawn(len(design))
    runs, roles = [], []
    for row, stream in zip(design.itertuples(), streams):
        sim_seed, log_seed = stream.spawn(2)
        A = row.graph[np.ix_(row.seat_node, row.seat_node)]
        out = sim.simulate(A, row.T, row.interface, R=R, seed=sim_seed, speed=row.speed, **model)
        log_rng = np.random.default_rng(log_seed)
        logger = "whole_second" if row.whole_second else "cloud" if row.interface == "cc" else "lab"
        for r, o in enumerate(out):
            meta = {"group": f"sim{r}_{row.group}", "dataset": row.dataset, "topology": row.topology, "wave": row.wave,
                    "graph": row.graph, "seat_node": row.seat_node, "interface": row.interface, "time_limit": row.T,
                    "end": "success" if o["success"] else "failed", "seed": seed}
            runs.append(sim.to_run(o["events"], meta, logger, log_rng))
            roles.append(row.role)
    return runs, sim.table(runs, roles)

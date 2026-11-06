"""Path 3 human inputs, lab v09 G-train rounds only (load_runs default roles; never test runs).

Reaction delays: the halt durations of motion.motion_spells (mid-slider halts that are not inside an event),
without the halts cut by the round end (censored). Start delays: each seat's start_delay (motion_spells).
Consensus times: purpose "outcome" rounds; a failed round counts as no consensus (inf).
"""
import numpy as np
import pandas as pd

from slider import config, interface, motion
from slider import slider_logs as sl

DELAYS = config.PATH3 / "human_delays.csv"
CONSENSUS = config.PATH3 / "human_consensus.csv"


def v09(runs, table):
    keep = [interface.version_tag(f) == "v09" for f in table["file"]]
    return [r for r, k in zip(runs, keep) if k], table[keep].reset_index(drop=True)


def build_delays():
    runs, table = v09(*sl.load_runs("behavior"))
    seats, halts = motion.motion_table(runs)
    halts = halts[halts["exit"] != "round_end"]
    out = pd.concat([pd.DataFrame({"kind": "halt", "seconds": halts["duration"], "file": halts["file"]}),
                     pd.DataFrame({"kind": "start", "seconds": seats["start_delay"].dropna(), "file": seats["file"]})])
    config.PATH3.mkdir(parents=True, exist_ok=True)
    out.to_csv(DELAYS, index=False)
    return out


def load_delays():
    """{"halt": seconds, "start": seconds} (numpy arrays); built from the logs on first use."""
    d = pd.read_csv(DELAYS) if DELAYS.exists() else build_delays()
    return {kind: d.loc[d["kind"] == kind, "seconds"].to_numpy() for kind in ["halt", "start"]}


def build_consensus():
    """One row per lab v09 G-train outcome round: file, wave, topology, t_consensus (inf if failed)."""
    runs, table = v09(*sl.load_runs("outcome"))
    t = [r["t"][-1] if r["end"] == "success" else np.inf for r in runs]
    out = pd.DataFrame({"file": table["file"], "wave": table["wave"], "topology": table["topology"],
                        "t_consensus": t})
    out.to_csv(CONSENSUS, index=False)
    return out

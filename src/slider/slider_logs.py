"""Parser for the Slider Game logs (Tabor, Smith & Dankowicz, OSF archive gvcpz).

Two log formats:
    lab   (2016)  Data-2016-12-09/log_*.txt
    cloud (2018)  2018-05-31 RunsScrubbed/Log_*.txt

Each parser returns a list of runs. One run is a dict with
    file, session, wave, name   where it came from (wave = 1, 2, ... in the order played;
                                name = topology name written in the log, if any)
    A_log    (n, n)  adjacency exactly as logged (1s on the diagonal)
    A        (n, n)  adjacency without the diagonal
    weights  (n,)    the +1/-1 Weights row
    t        (T,)    seconds since the start of the run
    X        (T, n)  raw slider positions
    agree    (T, n)  logged Agreement of each player (percent)
    general  (T,)    logged General-Success (percent)
    end      "success", "failed" or "incomplete"
    bot      True if a robot played in this run (cloud only)
"""
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from slider.config import OSF_DATA as DATA, LAB_DIR, CLOUD_DIR


def extract_data():
    """Unzip the two OSF archives next to the zip files (only if not done yet)."""
    if LAB_DIR.exists() and CLOUD_DIR.exists():
        return
    for z in DATA.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            names = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
            zf.extractall(DATA, members=names)


def new_run(path, session, wave, name, A_log, weights):
    A_log = np.array(A_log, dtype=int)
    return {
        "file": Path(path).name, "session": session, "wave": wave, "name": name,
        "A_log": A_log, "A": A_log - np.diag(np.diag(A_log)),
        "weights": np.array(weights, dtype=float),
        "t": [], "X": [], "agree": [], "general": [],
        "end": "incomplete", "bot": False,
    }


def to_arrays(runs):
    for run in runs:
        run["t"] = np.array(run["t"], dtype=float)
        run["X"] = np.array(run["X"], dtype=float)
        run["agree"] = np.array(run["agree"], dtype=float)
        run["general"] = np.array(run["general"], dtype=float)
    return runs


# ---------------------------------------------------------------- lab (2016)

def parse_lab(path):
    text = Path(path).read_text(errors="replace")
    settings, _, body = text.partition("Setting File End")

    # one (adjacency rows, weights, topology name in the XML comment) per wave, in file order.
    # The k-th "Wave Starting" uses the k-th WaveDescription. The Number="..." attribute is not
    # used because many settings files number the 9th wave "8" again.
    waves = []
    for block in re.findall(r"<WaveDescription.*?</WaveDescription>", settings, re.S):
        rows = [r.split() for r in re.findall(r"<row>(.*?)</row>", block)]
        weights = re.findall(r"<Weights>(.*?)</Weights>", block)[0].split()
        name = re.findall(r"<Adjacency>\s*<!--\s*(.*?)\s*-->", block)
        waves.append((rows, weights, name[0] if name else ""))

    runs = []
    run = None        # the run being read (None between runs)
    pending = None    # a positions row waiting for its agreement row
    for line in body.splitlines():
        if line.startswith("Wave Starting:"):
            wave = len(runs) + 1
            rows, weights, name = waves[wave - 1]
            run = new_run(path, "lab", wave, name, rows, weights)
            runs.append(run)
            pending = None
        elif run is None:
            continue
        elif line.startswith("Round"):              # "Round Success" or "Round Failed"
            run["end"] = "success" if "Success" in line else "failed"
            run = None
        elif " seconds" in line:                    # "3.04 seconds <tab> 7 positions"
            t, values = line.split(" seconds")
            pending = (float(t), values.split())
        elif line.strip() and line[0].isspace() and pending:   # "<tab> 7 agreements <tab> general"
            values = line.split()
            n = len(run["weights"])
            if len(pending[1]) == n and len(values) == n + 1:
                run["t"].append(pending[0])
                run["X"].append(pending[1])
                run["agree"].append(values[:n])
                run["general"].append(values[n])
            pending = None
    return to_arrays(runs)


# -------------------------------------------------------------- cloud (2018)

def parse_cloud(path):
    runs = []
    run = None        # the run being read (None between runs)
    t0 = 0.0          # clock time at the start of the run
    t_play = None     # time of the last "Adjusted Plays" row, waiting for its agreement row
    header = False    # True while reading the adjacency rows after "Starting Game k"
    for line in Path(path).read_text(errors="replace").splitlines():
        m = re.match(r"(\d\d):(\d\d):(\d\d\.\d+): (.*)", line)
        if not m:
            continue
        clock = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        msg = m.group(4)

        if re.fullmatch(r"Starting Game \d+", msg):
            run = new_run(path, "cloud", int(msg.split()[-1]), "", [], [])
            run["rows"] = []
            runs.append(run)
            t0, t_play, header = clock, None, True
        elif run is None:
            continue
        elif header:
            if msg.startswith("Adjacency Matrix:"):
                run["name"] = msg.split(":")[1].strip()
            elif msg.startswith("Weights:"):
                run.update(new_run(path, "cloud", run["wave"], run["name"], run.pop("rows"), msg.split()[1:]))
                header = False
            else:
                run["rows"].append(msg.split())
        elif msg == "Round Success" or msg == "Wave timed out":
            run["end"] = "success" if msg == "Round Success" else "failed"
            run = None
        elif msg.startswith("Adjusted Plays:"):     # 7 positions, robots written as (x)
            values = msg.split(":")[1]
            if "(" in values:
                run["bot"] = True
            run["X"].append(values.replace("(", "").replace(")", "").split())
            t_play = clock - t0
        elif msg.startswith("Extra: \t") and t_play is not None:   # 7 agreements + general
            values = msg.split()[1:]
            run["t"].append(t_play)
            run["agree"].append(values[:-1])
            run["general"].append(values[-1])
            t_play = None

    runs = [r for r in runs if "rows" not in r]     # drop a game whose header was cut off
    for run in runs:
        run["X"] = run["X"][:len(run["t"])]         # drop a last positions row with no agreement row
    return to_arrays(runs)


# ------------------------------------------------------ steps 1.3, 1.4, 1.5

def sign_correct(run):
    """s = weights ⊙ X"""
    return run["weights"] * run["X"]


def agreement(A_log, s):
    """Agreement_i = 100 * |sum of s over i and its neighbours| / (number of them).

    s has one row per time step. A_log has 1s on the diagonal (each player counts itself).
    """
    return 100 * np.abs(s @ A_log.T) / A_log.sum(axis=1)


# sorted degree sequence -> topology (7 players, self-loops removed)
DEGREE_SEQUENCES = {
    (6, 6, 6, 6, 6, 6, 6): "Full",
    (1, 1, 1, 1, 1, 1, 6): "Star",
    (1, 1, 1, 1, 2, 3, 3): "Hierarchy",
    (2, 2, 2, 2, 2, 3, 3): "CT-Hierarchy",
    (2, 2, 2, 2, 2, 2, 2): "Ring",
}


def label_topology(A):
    degrees = tuple(sorted(A.sum(axis=1).tolist()))
    return DEGREE_SEQUENCES.get(degrees, "other")


def dataset_label(path):
    """EL / CL / CC as selected by the OSF scripts selectdataset{EL,CL,CC}.m ('' = not selected)."""
    path = Path(path)
    if path.parent.name == CLOUD_DIR.name:
        return "CC"
    if path.parent.name != LAB_DIR.name:           # the Others/ subfolder
        return ""
    first, month, day, time = path.name.split(" ")[:4]
    day = int(day)
    if month == "Nov":
        return "EL" if day <= 16 and time not in ("161103", "161208") else ""
    if month == "Oct" and day == 31:
        return "EL"
    if "v09" not in first:                          # the first lab sessions, before version 09
        return "CL"
    if month == "Sep":
        return "CL" if time not in ("162439", "170339", "153023") else ""
    if month == "Oct":
        skip = ("154913", "155407", "164505", "155839", "162500", "163226",
                "171748", "160321", "170411", "170439", "173032", "180403")
        return "CL" if time not in skip else ""
    return ""


def log_files():
    """All log files. tmp.txt is skipped: it is a byte-for-byte copy of Log_2018-05-31_132916.txt."""
    lab = sorted(LAB_DIR.glob("log*.txt")) + sorted(LAB_DIR.glob("Others/log*.txt"))
    cloud = sorted(CLOUD_DIR.glob("Log_*.txt"))
    return lab, cloud


def parse_all():
    lab, cloud = log_files()
    runs = []
    for path in lab:
        for run in parse_lab(path):
            run["dataset"] = dataset_label(path)
            runs.append(run)
    for path in cloud:
        for run in parse_cloud(path):
            run["dataset"] = dataset_label(path)
            runs.append(run)
    return runs


def runs_table(runs):
    rows = []
    for run in runs:
        T = len(run["t"])
        if T > 0:
            error = np.abs(agreement(run["A_log"], sign_correct(run)) - run["agree"]).max()
        else:
            error = np.nan
        rows.append({
            "file": run["file"],
            "session": run["session"],
            "dataset": run["dataset"],
            "wave": run["wave"],
            "players": len(run["weights"]),
            "topology": label_topology(run["A"]),
            "success": run["end"] == "success",
            "end": run["end"],
            "duration": run["t"][-1] if T > 0 else np.nan,
            "bot": run["bot"],
            "T": T,
            "agree_error": error,
        })
    return pd.DataFrame(rows)


# ------------------------------------------- showing log lines (HW 1, step 1.1)

def find(lines, text):
    """index of the first line that contains text"""
    for i, line in enumerate(lines):
        if text in line:
            return i

def show(lines, start, count):
    for i in range(start, start + count):
        print(f"{i + 1:6d} | {lines[i]}")

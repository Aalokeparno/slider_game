"""Units 6 + 7: human escape anatomy. Human trajectories coarse-grained to corners, against the logit chain's
kernel, committor and reactive paths (nulls: uniform, T = inf, and logit at T_hat, fitted on Hierarchy).

Coarse-graining (per round, s = weights * X):
    held sign sigma_i = last nonzero sign of s_i (0 = undefined until the first nonzero value); it changes when
        s_i * sigma_i < 0 (variant hold05: only when s_i * sigma_i <= -0.5)
    corner x = sigma once all 7 are defined (earlier rows ignored); state code as landscape.py (bit i = node i +)
    group = sign changes within TIE = 0.1 s of the group's first change, applied jointly: pre-group state x,
        post-group state y, movers = nodes whose sign differs between x and y (m = number of movers; a group whose
        changes cancel is dropped)
Sets (landscape.py, on the round's own adjacency): A = union of the bottom plateaus of all traps; B = consensus.
Per round: episode = first entry into A until the next visit to B (escaped) or the round end (censored);
    spell = maximal run of states in A within the episode; attempt = group from x in A to y not in A, outcome
    success (B before A), aborted (A before B) or censored; escape = successful attempt; leader = its movers.
    Only groups inside the episode are used (moves from A after the episode are counted, not analysed).
Move labels: height of flipping i at x (dcut < 0 down, = 0 flat, > 0 up) x progress (k = min(#+, #-) decreases).
Tied groups: each mover counts with weight 1 / m at the pre-group state (variant no_ties: groups with m > 1 are
    dropped from every measure; the corner path keeps them).

Nulls at temperature T (T = inf: uniform): kernel P(i | x) ~ w_i = 1 / (1 + exp(dcut_i(x) / T)); committor
q(y) = P(B before A | y) (markov.committor_vector); leader null at a successful exit from x: P(i | x, success)
~ w_i q(y_i) over i with y_i not in A; overshoot null: Monte Carlo of the jump chain from the exit state y,
10,000 paths that reach B before A (seed 573), max cut along x, y, ..., B.

Measures (rows: variant x topology x cell (+ pooled) x T; observed, expected, observed - expected with a 95% CI
from 2,000 bootstrap resamples of the groups (log files), seed 573):
    K  moves from A: share per height x progress class; mean log-likelihood per move (expected: sum P log P)
    M  (a) the same shares for the first move of each spell and for later moves, and first - later;
       (b) P(a mover of this group was a mover of the previous group)
    C  attempts (censored excluded): success rate vs mean q(y) and Brier score (expected: mean q (1 - q)), each
       for all exits and by exit height; attempts per escaped episode vs 1 / q_bar (geometric)
    L  leader class (role x height x progress) vs the leader null
    O  share of escapes whose max climb (max cut on x, y, ..., B minus cut(x)) equals the barrier B of x's plateau
"""
import hashlib
import pickle
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

from slider import config, markov
from slider import landscape as L
from slider import slider_logs as sl
from slider.interface import version_tag

TIE = 0.1
EPS = 1e-9
N_BOOT = 2000
N_MC = 10_000
VARIANTS = {"main": {"hold": 0.0, "ties": True}, "hold05": {"hold": 0.5, "ties": True},
            "no_ties": {"hold": 0.0, "ties": False}}
HEIGHTS = ["down", "flat", "up"]
CLASSES = [f"{h}/{p}" for h in HEIGHTS for p in ["no_progress", "progress"]]       # class id = 2 h + p
TRAIN_CELLS = {"Hierarchy EL": ("Hierarchy", "EL"), "Hierarchy CL v09": ("Hierarchy", "CL"),
               "Hierarchy CC": ("Hierarchy", "CC")}
TEST_CELLS = {"open": {"Ring EL": ("Ring", "EL"), "Ring CL v09": ("Ring", "CL"), "CT EL": ("CT-Hierarchy", "EL")},
              "dry": {"Ring EL": ("Hierarchy", "EL"), "Ring CL v09": ("Hierarchy", "CL"), "CT EL": ("Hierarchy", "EL")}}
# role of a node: (degree, sorted neighbour degrees) -> name (Hierarchy, CT, Ring); other signatures keep their text
ROLE_NAMES = {(2, (3, 3)): "apex", (3, (1, 1, 2)): "middle", (1, (3,)): "leaf",
              (3, (2, 2, 2)): "middle", (2, (2, 3)): "leaf", (2, (2, 2)): "ring"}
L5 = {"keep_aborted": False, "keep_incomplete": False, "keep_bot": False, "keep_whole_second": False,
      "keep_empty": False}
KEEP_ALL = {k: True for k in sl.PURPOSES["behavior"]}


# ---------------------------------------------------------------- coarse-graining

def held_signs(s, hold=0.0):
    """sigma per row (T x n, values -1 / 0 / +1): forward hold of the last nonzero sign; with hold > 0 a sign
    changes only when s * sigma <= -hold."""
    s = np.asarray(s, dtype=float)
    sigma = np.zeros(s.shape, dtype=int)
    cur = np.zeros(s.shape[1], dtype=int)
    for r in range(len(s)):
        x = s[r]
        new = (cur == 0) & (x != 0)
        flip = (cur != 0) & ((x * cur < -EPS) if hold == 0 else (x * cur <= -hold + EPS))
        cur = np.where(new | flip, np.sign(x).astype(int), cur)
        sigma[r] = cur
    return sigma


def code(row):
    return int(sum(1 << i for i, v in enumerate(row) if v > 0))


def corner_path(t, sigma, tie=TIE):
    """(t0, x0, groups) from the first row with all signs defined; groups = list of dicts t, x, y, movers, m.
    None if the signs are never all defined."""
    full = np.flatnonzero(np.all(sigma != 0, axis=1))
    if len(full) == 0:
        return None
    r0 = full[0]
    rows = [r for r in range(r0 + 1, len(sigma)) if np.any(sigma[r] != sigma[r - 1])]
    groups, k = [], 0
    while k < len(rows):
        first = rows[k]
        last = k
        while last + 1 < len(rows) and t[rows[last + 1]] - t[first] <= tie + EPS:
            last += 1
        before, after = sigma[first - 1], sigma[rows[last]]
        movers = tuple(int(i) for i in np.flatnonzero(before != after))
        if movers:
            groups.append({"t": float(t[first]), "x": code(before), "y": code(after), "movers": movers,
                           "m": len(movers)})
        k = last + 1
    return float(t[r0]), code(sigma[r0]), groups


# ---------------------------------------------------------------- graph context and nulls

class Context:
    """Graph quantities of one adjacency: cut, dcut, k, A (union of bottom plateaus), barrier per A state,
    move class per (state, node), node roles; kernel, committor and leader nulls per T (cached)."""

    def __init__(self, A):
        self.A = np.asarray(A, dtype=int)
        self.cut, self.dcut, _, self.k = L.basics(self.A)
        self.inA = np.zeros(len(L.STATES), dtype=bool)
        self.barrier = np.full(len(L.STATES), np.nan)
        for r in L.trap_landscape(self.A):
            P = list(L.plateau(r["code"], self.dcut))
            self.inA[P] = True
            self.barrier[P] = r["B"]
        self.inB = np.isin(L.STATES, [L.PLUS, L.MINUS])
        flips = L.STATES[:, None] ^ (1 << np.arange(L.N))
        self.flips = flips
        height = np.sign(self.dcut) + 1                                  # 0 down, 1 flat, 2 up
        progress = (self.k[flips] < self.k[:, None]).astype(int)
        self.cls = 2 * height + progress
        deg = self.A.sum(axis=1)
        self.role = [ROLE_NAMES.get(sig, str(sig)) for sig in
                     [(int(deg[i]), tuple(sorted(int(deg[j]) for j in np.flatnonzero(self.A[i])))) for i in range(L.N)]]
        self._cache = {}

    def kernel(self, T):
        """P(i | x), 128 x 7."""
        if ("P", T) not in self._cache:
            w = np.full(self.dcut.shape, 0.5) if np.isinf(T) else expit(-self.dcut / T)
            self._cache[("P", T)] = w / w.sum(axis=1, keepdims=True)
        return self._cache[("P", T)]

    def q(self, T):
        if ("q", T) not in self._cache:
            self._cache[("q", T)] = markov.committor_vector(self.A, T, L.STATES[self.inA])
        return self._cache[("q", T)]

    def leader(self, T):
        """P(i | x, success) ~ w_i q(y_i) over exits i (y_i not in A); rows of 0 where no exit can succeed."""
        if ("lead", T) not in self._cache:
            v = self.kernel(T) * self.q(T)[self.flips] * ~self.inA[self.flips]
            tot = v.sum(axis=1, keepdims=True)
            self._cache[("lead", T)] = np.divide(v, tot, out=np.zeros_like(v), where=tot > 0)
        return self._cache[("lead", T)]

    def overshoot(self, y, T, n=N_MC, seed=config.SEED, max_paths=2_000_000):
        """Max cut along the jump-chain path from y (y included) to B, for n paths that reach B before A
        (fewer if max_paths are used up); array of max cuts."""
        key = ("over", y, T)
        if key not in self._cache:
            if self.inB[y]:
                self._cache[key] = np.full(n, self.cut[y])
                return self._cache[key]
            rng = np.random.default_rng(seed)
            cum = np.cumsum(self.kernel(T), axis=1)
            out, used = [], 0
            while sum(len(o) for o in out) < n and used < max_paths:
                b = 20_000
                used += b
                s = np.full(b, y)
                mx = np.full(b, self.cut[y])
                alive = np.ones(b, dtype=bool)
                while alive.any():
                    idx = np.flatnonzero(alive)
                    u = rng.random(len(idx))
                    i = np.minimum((cum[s[idx]] < u[:, None]).sum(axis=1), L.N - 1)
                    s[idx] = s[idx] ^ (1 << i)
                    mx[idx] = np.maximum(mx[idx], self.cut[s[idx]])
                    alive[idx] = ~(self.inA[s[idx]] | self.inB[s[idx]])
                out.append(mx[self.inB[s]])
            self._cache[key] = np.concatenate(out)[:n] if out else np.array([])
        return self._cache[key]


_CONTEXTS = {}


def context(A):
    key = np.asarray(A, dtype=int).tobytes()
    if key not in _CONTEXTS:
        _CONTEXTS[key] = Context(A)
    return _CONTEXTS[key]


# ---------------------------------------------------------------- events of one round

def round_events(run, hold=0.0):
    """Corner path of one round cut into the episode; returns (moves, attempts, episode, extra) where moves =
    groups from A inside the episode, attempts = groups from A to outside A, episode = one dict or None."""
    ctx = context(run["A"])
    path = corner_path(run["t"], held_signs(sl.sign_correct(run), hold))
    info = {"corner": path is not None, "groups": 0, "tied": 0, "moves_after": 0}
    if path is None:
        return [], [], None, info
    _, x0, groups = path
    info["groups"], info["tied"] = len(groups), sum(g["m"] > 1 for g in groups)
    states = [x0] + [g["y"] for g in groups]
    inA = ctx.inA[states]
    if not inA.any():
        return [], [], None, info
    start = int(np.flatnonzero(inA)[0])
    hitB = [j for j in range(start + 1, len(states)) if ctx.inB[states[j]]]
    end = hitB[0] if hitB else len(states) - 1
    escaped = bool(hitB)
    info["moves_after"] = int(sum(inA[j] for j in range(end, len(states) - 1)))
    moves, attempts = [], []
    spell = 0
    for j in range(start, end):
        if not inA[j]:
            continue
        first = j == start or not inA[j - 1]
        spell += first
        g = groups[j]
        moves.append({"x": states[j], "y": states[j + 1], "movers": g["movers"], "m": g["m"], "first": first,
                      "spell": spell, "prev": groups[j - 1]["movers"] if j > 0 else None})
        if not inA[j + 1]:
            nxt = [l for l in range(j + 1, end + 1) if inA[l] or ctx.inB[states[l]]]
            if not nxt:
                outcome, stop = "censored", None
            else:
                stop = nxt[0]
                outcome = "success" if ctx.inB[states[stop]] else "aborted"
            climb = (max(ctx.cut[states[l]] for l in range(j, stop + 1)) - ctx.cut[states[j]]
                     if outcome == "success" else np.nan)
            attempts.append({"x": states[j], "y": states[j + 1], "movers": g["movers"], "m": g["m"],
                             "outcome": outcome, "climb": climb, "barrier": ctx.barrier[states[j]]})
    episode = {"escaped": escaped, "spells": spell, "attempts": len(attempts)}
    return moves, attempts, episode, info


def select(runs, table, cells):
    """Rounds of the cells (topology x dataset; CL = v09 only, old CL dropped) after the L5 filters.
    Returns a list of (cell name, run, table row) and the counts per cell (in scope, kept, groups)."""
    drop = sl.drop_mask(table, L5)
    out, counts = [], []
    for name, (top, ds) in cells.items():
        c = (table["topology"] == top) & (table["dataset"] == ds)
        if ds == "CL":
            c &= np.array([version_tag(f) == "v09" for f in table["file"]])
        keep = c & ~drop
        out += [(name, runs[i], table.loc[i]) for i in table.index[keep]]
        counts.append({"cell": name, "in_scope": int(c.sum()), "kept": int(keep.sum()),
                       "files": table.loc[keep, "file"].nunique()})
    return out, pd.DataFrame(counts)


def topology_label(cell):
    """Row label of a cell's topology: the first word of the cell name (Hierarchy, Ring, CT)."""
    return cell.split()[0]


def event_tables(rounds, hold=0.0):
    """moves, attempts, episodes, counts over rounds = [(cell, run, row)]; every row has cell, topology, file."""
    moves, attempts, episodes, counts = [], [], [], []
    for rid, (cell, run, row) in enumerate(rounds):
        mv, at, ep, info = round_events(run, hold)
        base = {"cell": cell, "topology": topology_label(cell), "file": row["file"], "round": rid,
                "A": np.asarray(run["A"], dtype=int).tobytes()}
        moves += [{**base, **m} for m in mv]
        attempts += [{**base, **a} for a in at]
        if ep is not None:
            episodes.append({**base, **ep})
        counts.append({**base, **info, "episode": ep is not None})
    return (pd.DataFrame(moves), pd.DataFrame(attempts), pd.DataFrame(episodes), pd.DataFrame(counts))


# ---------------------------------------------------------------- per-event observed / expected values

def ctx_of(row):
    return _CONTEXTS[row["A"]]


def move_values(moves, T):
    """Per move (group): weight 1 and, per class, obs = share of its movers in the class and exp = P(class | x);
    loglik obs / exp; same-mover obs / exp (NaN without a previous group)."""
    rows = []
    for _, r in moves.iterrows():
        ctx = ctx_of(r)
        P = ctx.kernel(T)[r["x"]]
        cls = ctx.cls[r["x"]]
        v = {"file": r["file"], "w": 1.0}
        for c in range(6):
            v[f"obs_{c}"] = np.mean([cls[j] == c for j in r["movers"]])
            v[f"exp_{c}"] = P[cls == c].sum()
        v["obs_ll"] = np.mean([np.log(P[j]) for j in r["movers"]])
        v["exp_ll"] = float((P * np.log(P)).sum())
        if r["prev"] is None:
            v["w_same"], v["obs_same"], v["exp_same"] = 0.0, 0.0, 0.0
        else:
            prev = set(r["prev"])
            v["w_same"] = 1.0
            v["obs_same"] = np.mean([j in prev for j in r["movers"]])
            v["exp_same"] = P[list(prev)].sum()
        rows.append(v)
    return pd.DataFrame(rows)


def leader_classes(ctx):
    return [f"{role} {c}" for role in sorted(set(ctx.role)) for c in CLASSES]


def attempt_values(attempts, T):
    """Per attempt: obs success, q(y), exit height; per escape: leader class obs / exp, overshoot obs / exp."""
    rows = []
    for _, r in attempts.iterrows():
        ctx = ctx_of(r)
        q = ctx.q(T)
        v = {"file": r["file"], "outcome": r["outcome"], "q": q[r["y"]], "success": float(r["outcome"] == "success"),
             "up": float(ctx.cut[r["y"]] > ctx.cut[r["x"]])}
        if r["outcome"] == "success":
            lead = ctx.leader(T)[r["x"]]
            names = [f"{ctx.role[i]} {CLASSES[ctx.cls[r['x'], i]]}" for i in range(L.N)]
            v["lead_ok"] = float(lead.sum() > 0)
            v["lead_obs"] = {n: np.mean([names[j] == n for j in r["movers"]]) for n in set(names)}
            v["lead_exp"] = {n: lead[[i for i in range(L.N) if names[i] == n]].sum() for n in set(names)}
            mx = ctx.overshoot(r["y"], T)
            v["over_ok"] = float(len(mx) > 0)
            v["over_obs"] = float(r["climb"] == r["barrier"])
            v["over_exp"] = (float(np.mean(np.maximum(mx, ctx.cut[r["x"]]) - ctx.cut[r["x"]] == r["barrier"]))
                             if len(mx) else 0.0)
        rows.append(v)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- bootstrap over groups (log files)

def boot(ev, cols, fn, n_boot=N_BOOT, seed=config.SEED):
    """fn(dict col -> sum) -> (obs, exp) for the file sums of ev[cols]; returns obs, exp, diff and the 95% CI
    of diff from n_boot resamples of the files (with replacement)."""
    S = ev.groupby("file")[cols].sum()
    files = S.to_numpy(float)
    obs, exp = fn(dict(zip(cols, files.sum(axis=0))))
    idx = np.random.default_rng(seed).integers(0, len(files), (n_boot, len(files)))
    Bs = files[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        bo, be = fn(dict(zip(cols, Bs.T)))
    lo, hi = np.nanpercentile(np.asarray(bo) - np.asarray(be), [2.5, 97.5]) if len(files) else (np.nan, np.nan)
    return {"obs": obs, "exp": exp, "diff": obs - exp, "diff_lo": lo, "diff_hi": hi, "n_files": len(files)}


def ratio(num_obs, num_exp, den):
    def fn(s):
        with np.errstate(divide="ignore", invalid="ignore"):
            return s[num_obs] / s[den], s[num_exp] / s[den]
    return fn


# ---------------------------------------------------------------- measures

def measure_rows(moves, attempts, episodes, T, base):
    """K, M, C, L, O rows of one set of events (one topology x cell) at one T; base = labels for every row."""
    out = {k: [] for k in "KMCLO"}
    if len(moves):
        mv = move_values(moves, T)
        mv["first"] = moves["first"].to_numpy().astype(float)
        n = len(mv)
        for c in range(6):
            out["K"].append({**base, "stat": f"share {CLASSES[c]}", "part": "all", "n": n,
                             **boot(mv, [f"obs_{c}", f"exp_{c}", "w"], ratio(f"obs_{c}", f"exp_{c}", "w"))})
        out["K"].append({**base, "stat": "loglik", "part": "all", "n": n,
                         **boot(mv, ["obs_ll", "exp_ll", "w"], ratio("obs_ll", "exp_ll", "w"))})
        f, l = mv["first"], 1 - mv["first"]
        for c in range(6):
            e = mv[["file"]].copy()
            e["of"], e["ef"], e["wf"] = mv[f"obs_{c}"] * f, mv[f"exp_{c}"] * f, f
            e["ol"], e["el"], e["wl"] = mv[f"obs_{c}"] * l, mv[f"exp_{c}"] * l, l
            cols = ["of", "ef", "wf", "ol", "el", "wl"]
            out["M"].append({**base, "stat": f"share {CLASSES[c]}", "part": "first", "n": int(f.sum()),
                             **boot(e, cols, ratio("of", "ef", "wf"))})
            out["M"].append({**base, "stat": f"share {CLASSES[c]}", "part": "later", "n": int(l.sum()),
                             **boot(e, cols, ratio("ol", "el", "wl"))})

            def fl(s):
                with np.errstate(divide="ignore", invalid="ignore"):
                    return s["of"] / s["wf"] - s["ol"] / s["wl"], s["ef"] / s["wf"] - s["el"] / s["wl"]
            out["M"].append({**base, "stat": f"share {CLASSES[c]}", "part": "first_minus_later", "n": n,
                             **boot(e, cols, fl)})
        out["M"].append({**base, "stat": "same_mover", "part": "all", "n": int(mv["w_same"].sum()),
                         **boot(mv, ["obs_same", "exp_same", "w_same"], ratio("obs_same", "exp_same", "w_same"))})
    at = attempt_values(attempts, T) if len(attempts) else pd.DataFrame()
    done = at[at["outcome"] != "censored"] if len(at) else at
    if len(done):
        for part, mask in [("all", np.ones(len(done), bool)), ("up", done["up"] == 1), ("down", done["up"] == 0)]:
            e = done[mask].copy()
            e["w"] = 1.0
            e["brier_obs"] = (e["success"] - e["q"]) ** 2
            e["brier_exp"] = e["q"] * (1 - e["q"])
            for stat, o, x in [("success_rate", "success", "q"), ("brier", "brier_obs", "brier_exp")]:
                row = ({**base, "stat": stat, "part": part, "n": len(e), **boot(e, [o, x, "w"], ratio(o, x, "w"))}
                       if len(e) else {**base, "stat": stat, "part": part, "n": 0})
                out["C"].append(row)
        esc_ep = episodes[episodes["escaped"]] if len(episodes) else episodes
        if len(esc_ep):
            e = pd.concat([esc_ep[["file"]].assign(att=esc_ep["attempts"].astype(float), esc=1.0, qs=0.0, nd=0.0),
                           done[["file"]].assign(att=0.0, esc=0.0, qs=done["q"], nd=1.0)], ignore_index=True)

            def ape(s):
                with np.errstate(divide="ignore", invalid="ignore"):
                    return s["att"] / s["esc"], s["nd"] / s["qs"]
            out["C"].append({**base, "stat": "attempts_per_escape", "part": "all", "n": len(esc_ep),
                             **boot(e, ["att", "esc", "qs", "nd"], ape)})
    esc = at[at["outcome"] == "success"] if len(at) else at
    if len(esc):
        lead = esc[esc["lead_ok"] == 1]
        names = sorted({n for d in lead["lead_exp"] for n in d} | {n for d in lead["lead_obs"] for n in d})
        for name in names:
            e = lead[["file"]].copy()
            e["o"] = [d.get(name, 0.0) for d in lead["lead_obs"]]
            e["x"] = [d.get(name, 0.0) for d in lead["lead_exp"]]
            e["w"] = 1.0
            if e["o"].sum() > 0 or e["x"].sum() > 0:
                out["L"].append({**base, "stat": f"leader {name}", "part": "all", "n": len(e),
                                 **boot(e, ["o", "x", "w"], ratio("o", "x", "w"))})
        ov = esc[esc["over_ok"] == 1].copy()
        ov["w"] = 1.0
        if len(ov):
            out["O"].append({**base, "stat": "climb_equals_B", "part": "all", "n": len(ov),
                             **boot(ov, ["over_obs", "over_exp", "w"], ratio("over_obs", "over_exp", "w"))})
    return out


def structural(ctx, T):
    """Notes on measures that are structural (reported, not tested) for this graph."""
    exits = [(x, i) for x in L.STATES[ctx.inA] for i in range(L.N) if not ctx.inA[ctx.flips[x, i]]]
    q = ctx.q(T)
    notes = {}
    if all(ctx.dcut[x, i] > 0 for x, i in exits):
        notes["C"] = notes["L"] = "every exit from A is uphill (split by exit height is degenerate)"
    if all(q[ctx.flips[x, i]] in (0.0, 1.0) or abs(q[ctx.flips[x, i]] - round(q[ctx.flips[x, i]])) < 1e-12
           for x, i in exits):
        notes["C"] = notes["L"] = "every exit from A has q = 0 (uphill) or q = 1 (downhill): C and L degenerate"
    return notes


def analyse(rounds, T_values, variants=VARIANTS):
    """All measures and counts. rounds = [(cell, run, row)]; T_values = {label: T}. Returns {K, M, C, L, O,
    counts} DataFrames with columns variant, topology, cell, T, T_value, stat, part, obs, exp, diff, diff_lo,
    diff_hi, n, n_files, structural."""
    tables = {k: [] for k in "KMCLO"}
    counts = []
    for vname, v in variants.items():
        moves, attempts, episodes, cnt = event_tables(rounds, v["hold"])
        if not v["ties"]:
            moves = moves[moves["m"] == 1] if len(moves) else moves
            attempts = attempts[attempts["m"] == 1] if len(attempts) else attempts
        for top in sorted(cnt["topology"].unique()):
            cells = [c for c in dict.fromkeys(cnt.loc[cnt["topology"] == top, "cell"])] + ["pooled"]
            for cell in cells:
                pick = (lambda d: d[(d["topology"] == top) & ((d["cell"] == cell) | (cell == "pooled"))]
                        if len(d) else d)
                mv, at, ep, ct = pick(moves), pick(attempts), pick(episodes), pick(cnt)
                at_all = at
                counts.append({"variant": vname, "topology": top, "cell": cell, "rounds": len(ct),
                               "rounds_with_corner": int(ct["corner"].sum()), "groups": int(ct["groups"].sum()),
                               "tied_share": ct["tied"].sum() / max(ct["groups"].sum(), 1),
                               "episodes": len(ep), "escaped": int(ep["escaped"].sum()) if len(ep) else 0,
                               "spells": int(ep["spells"].sum()) if len(ep) else 0,
                               "moves_from_A": len(mv), "moves_from_A_after_episode": int(ct["moves_after"].sum()),
                               "attempts": len(at_all),
                               **{f"attempts_{o}": int((at_all["outcome"] == o).sum()) if len(at_all) else 0
                                  for o in ["success", "aborted", "censored"]},
                               "files": ct["file"].nunique()})
                As = set(ct["A"])
                for Tl, T in T_values.items():
                    notes = {}
                    for a in As:
                        for k, n in structural(_CONTEXTS[a], T).items():
                            notes[k] = n
                    base = {"variant": vname, "topology": top, "cell": cell, "T": Tl, "T_value": T}
                    rows = measure_rows(mv, at, ep, T, base)
                    for k in "KMCLO":
                        tables[k] += [{**r, "structural": notes.get(k, "")} for r in rows[k]]
    cols = ["variant", "topology", "cell", "T", "T_value", "stat", "part", "obs", "exp", "diff", "diff_lo", "diff_hi",
            "n", "n_files", "structural"]
    out = {k: pd.DataFrame(v).reindex(columns=cols) for k, v in tables.items()}
    out["counts"] = pd.DataFrame(counts)
    return out


# ---------------------------------------------------------------- T_hat

def kernel_loglik(moves, T, weights=None):
    """Sum over moves (and movers, weight 1 / m) of log P_T(i | x)."""
    total = 0.0
    for r, w in zip(moves.itertuples(), weights if weights is not None else np.ones(len(moves))):
        P = _CONTEXTS[r.A].kernel(T)[r.x]
        total += w * np.mean([np.log(P[j]) for j in r.movers])
    return total


def fit_that(moves, n_boot=N_BOOT, seed=config.SEED, bounds=(np.log(0.05), np.log(20))):
    """MLE of T for the kernel on the moves from A (log T in bounds); 95% CI from n_boot resamples of the files.
    Returns a one-row DataFrame: T_hat, ci_lo, ci_hi, n_moves, n_files, at_bound."""
    D = np.stack([_CONTEXTS[a].dcut[x] for a, x in zip(moves["A"], moves["x"])]).astype(float)   # n x 7
    M = np.zeros(D.shape)
    for r, mv in enumerate(moves["movers"]):
        M[r, list(mv)] = 1.0 / len(mv)
    files, f_idx = np.unique(moves["file"], return_inverse=True)

    def ll(logT, w):
        T = np.exp(logT)
        logw = np.log(expit(-D / T))
        logP = logw - np.log(np.exp(logw).sum(axis=1, keepdims=True))
        return float((w * (M * logP).sum(axis=1)).sum())

    def fit(w):
        return float(np.exp(minimize_scalar(lambda u: -ll(u, w), bounds=bounds, method="bounded",
                                            options={"xatol": 1e-6}).x))

    T_hat = fit(np.ones(len(D)))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        c = np.bincount(rng.integers(0, len(files), len(files)), minlength=len(files))
        bs.append(fit(c[f_idx].astype(float)))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    at_bound = bool(min(abs(np.log(T_hat) - b) for b in bounds) < 1e-3)
    return pd.DataFrame([{"T_hat": T_hat, "ci_lo": lo, "ci_hi": hi, "n_moves": len(D), "n_files": len(files),
                          "at_bound": at_bound}])


# ---------------------------------------------------------------- Hierarchy run

def hierarchy_rounds():
    """Training rounds of TRAIN_CELLS (L5 filters; train + val roles; no test runs)."""
    runs, table = sl.load_runs("behavior", roles={"train", "val"}, **KEEP_ALL)
    return select(runs, table, TRAIN_CELLS)


def hierarchy_that():
    """T_hat on the pooled Hierarchy training cells (main variant: moves from A in the episodes)."""
    rounds, _ = hierarchy_rounds()
    moves, _, _, _ = event_tables(rounds, 0.0)
    out = fit_that(moves)
    out.to_csv(config.U67_HIER_THAT, index=False)
    return out


def hierarchy_run(T_hat=None):
    """K, M, C, L, O and counts on the Hierarchy training cells at T = inf and T_hat (config.U67_THAT), all
    variants; written to config.U67_HIER (one file per table). Returns the tables and the selection counts."""
    T_hat = config.U67_THAT if T_hat is None else T_hat
    rounds, sel = hierarchy_rounds()
    out = analyse(rounds, {"inf": np.inf, "T_hat": T_hat})
    for k, path in config.U67_HIER.items():
        out[k].to_csv(path, index=False)
    return out, sel


# ---------------------------------------------------------------- test pipeline (scripts/u67_test.py)

REASON = "Units 6+7: the one pre-registered Ring/CT run (docs/u67_preregistration.md)"
SCORE_COLUMNS = ["id", "measure", "variant", "topology", "cell", "T", "stat", "part", "expect", "obs", "exp", "diff",
                 "diff_lo", "diff_hi", "deviation", "result", "note"]


def prereg_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_open(prereg=None, sha=None, log=None):
    """SystemExit unless the pre-registration exists with sha256 == sha (config.U67_PREREG_SHA) and the test
    access log has no earlier u67 entry."""
    prereg = prereg or config.U67_PREREG
    sha = sha if sha is not None else config.U67_PREREG_SHA
    log = log or config.TEST_ACCESS_LOG
    if not prereg.exists():
        raise SystemExit(f"STOP: {prereg} does not exist")
    if not sha:
        raise SystemExit("STOP: config.U67_PREREG_SHA is not set (the pre-registration is not locked)")
    if prereg_sha(prereg) != sha:
        raise SystemExit(f"STOP: sha256 of {prereg} differs from U67_PREREG_SHA")
    if log.exists():
        t = pd.read_csv(log)
        if (t["reason"].astype(str).str.contains("Units 6\\+7") | t["caller"].astype(str).str.contains("u67")).any():
            raise SystemExit(f"STOP: {log} already has a u67 entry; Ring / CT are opened once")


def predictions(prereg):
    """The YAML block of the pre-registration's Predictions section -> list of dicts."""
    import yaml
    text = prereg.read_text()
    sec = text.split("## Predictions", 1)
    if len(sec) < 2:
        return []
    m = re.search(r"```yaml\n(.*?)```", sec[1], re.S)
    if not m:
        return []
    data = yaml.safe_load(m.group(1)) or {}
    return data.get("predictions") or []


def score(preds, tables):
    """One row per prediction: the matching measure row (exactly one) and pass / fail.
    expect: deviation (CI of obs - exp excludes 0), no_deviation, positive (lo > 0), negative (hi < 0)."""
    rows = []
    for p in preds:
        t = tables[p["measure"]]
        keys = {"variant": p.get("variant", "main"), "topology": p["topology"], "cell": p.get("cell", "pooled"),
                "T": str(p["T"]), "stat": p["stat"], "part": p.get("part", "all")}
        hit = t[np.logical_and.reduce([t[k].astype(str) == str(v) for k, v in keys.items()])]
        row = {"id": p["id"], "measure": p["measure"], **keys, "expect": p["expect"]}
        if len(hit) != 1:
            rows.append({**row, "result": "no row", "note": f"{len(hit)} matching rows"})
            continue
        h = hit.iloc[0]
        dev = bool(h["diff_lo"] > 0 or h["diff_hi"] < 0)
        ok = {"deviation": dev, "no_deviation": not dev, "positive": h["diff_lo"] > 0,
              "negative": h["diff_hi"] < 0}[p["expect"]]
        rows.append({**row, "obs": h["obs"], "exp": h["exp"], "diff": h["diff"], "diff_lo": h["diff_lo"],
                     "diff_hi": h["diff_hi"], "deviation": dev, "result": "pass" if ok else "fail",
                     "note": h["structural"]})
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def test_pipeline(mode, prereg=None, log=None, cache=None, sha=None):
    """mode "dry": Hierarchy rounds stand in for Ring / CT (no test access); outputs u67_dryrun_*.
    mode "open": checks -> one load_runs(allow_test=True) -> cache -> (load_runs writes the access-log row) ->
    measures at T = inf and U67_THAT -> predictions scored; outputs u67_test_*."""
    prereg = prereg or config.U67_PREREG
    cache = cache or config.U67_TEST_CACHE
    if mode == "open":
        check_open(prereg, sha, log)
        runs, table = sl.load_runs("behavior", roles={"test"}, allow_test=True, reason=REASON, **KEEP_ALL)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump((runs, table), f)
        paths = config.U67_TEST
    else:
        runs, table = sl.load_runs("behavior", roles={"train", "val"}, **KEEP_ALL)
        paths = config.U67_DRYRUN
    rounds, sel = select(runs, table, TEST_CELLS[mode])
    out = analyse(rounds, {"inf": np.inf, "T_hat": config.U67_THAT})
    out["selection"] = sel
    out["scores"] = score(predictions(prereg), out)
    for k, path in paths.items():
        out[k].to_csv(path, index=False)
    return out

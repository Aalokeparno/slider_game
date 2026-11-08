"""Path 3 simulator: 7 strategy players on a graph, lab v09 ticks (game.py).

One tick n (time n * dt): every player acts on what it has seen up to now (its Agr history ends with Agr(n)),
all moves are applied at once, then Agr(n + 1) is computed and appended to each player's history.
A player is in one of three modes:
    decide   decide() is called this tick (after a "step" or a "stay")
    move     going to a target, one step per tick (no calls)
    wait     idle until tick `until` (first the hold, then the reaction delay); interrupt() is called on every
             idle tick after the hold, if the strategy has one
Reaction delay (whenever a player becomes idle): a human halt duration resampled (human.load_delays),
rounded to ticks, at least 1. The first decision of the round: a resampled human start delay, the same way.
The round ends at consensus (game.is_consensus) or at t_max.

S3.2 additions (all off by default, so earlier runs replay exactly):
    watch()       a strategy may define watch(t, s_i, agr_history, memory), called on every tick while it moves to a
                  target; it returns a new action or None (S3's travel check)
    seed role     a strategy with assigned_seed = True (S3a): one seat, uniform at random, gets memory["seed"] = True
    noise         fidget: a player that has only ever been idle (S3's WAIT) takes one random 0.1 step at rate `fidget`
                  per s (not undone); slip: every action with a direction is reversed with prob `slip` (slipped()).
                  Noise draws come from a generator spawned from the run's rng, so the run's other draws are unchanged.
"""
import json

import numpy as np
import pandas as pd

from slider.path3 import game, tail

DECIDE, MOVE, WAIT = 0, 1, 2
FIXED_S, CYCLE_S = 30, 120            # outcome classes: last 30 s (fixed trap), last 120 s (periodic)


def ticks(seconds):
    return max(1, int(round(seconds / game.DT)))


def slipped(act, k_i, noise, slip):
    """Slip: with prob `slip` an action with a direction goes the other way. A step: -d. A target at the other wall
    (more than one step away): the opposite wall (from a wall that is the player's own wall: no move). Any other target
    (one step, e.g. back from a probe): mirrored about the player's position, clipped. A target equal to the position
    has no direction (no draw). A yield that no longer moves is not a yield (the hold stays)."""
    if act[0] == "step":
        return ("step", -act[1]) if noise.random() < slip else act
    if act[0] != "target":
        return act
    x = int(round(10 * act[1]))
    if x == k_i or noise.random() >= slip:
        return act
    new = -x if abs(x) == game.WALL and abs(x - k_i) > 1 else int(np.clip(2 * k_i - x, -game.WALL, game.WALL))
    return ("target", new / 10, act[2], act[3] and new != k_i)


def run(A, cls, params, start_k, delays, rng, t_max=300, display="int", keep_path=False, fidget=0.0, slip=0.0):
    """One round. start_k: start positions (integers, s = k / 10). delays: {"halt": array, "start": array} in s.
    Returns the per-run row (row()); with keep_path also "path" (positions k per tick, (ticks + 1) x 7),
    "memory" (each player's memory without rng / s_history) and "yields" ((player, first tick, first idle tick))."""
    n = len(A)
    k = np.array(start_k, dtype=int)
    players = [cls(**params) for _ in range(n)]
    has_interrupt = hasattr(players[0], "interrupt")
    hist = [[float(a)] for a in game.agreement(A, k, display)]
    mem = [{"rng": np.random.default_rng(rng.integers(2**63)), "s_history": [k[i] / 10]} for i in range(n)]
    mode = [WAIT] * n
    until = [ticks(rng.choice(delays["start"])) for _ in range(n)]
    has_watch = hasattr(players[0], "watch")
    if getattr(players[0], "assigned_seed", False):
        mem[int(rng.integers(n))]["seed"] = True
    noise = rng.spawn(1)[0] if fidget or slip else None
    waiting = [True] * n                          # only idle actions so far (fidgets apply to these players)
    n_fidget = 0
    hold_end = [0] * n
    target, hold_after, yield_start = [0] * n, [0.0] * n, [None] * n
    yields = []                                   # (player, first tick, first idle tick)
    cross = {"toward": 0, "away": 0, "tie": 0}    # greedy crossings vs the global majority (sign of sum of s)
    path = [k.copy()]
    t_cons = np.nan

    def become_idle(i, start, hold):
        mode[i] = WAIT
        hold_end[i] = start + ticks(hold) if hold > 0 else start
        until[i] = hold_end[i] + ticks(rng.choice(delays["halt"]))

    def finish_move(i, idle_from):
        if yield_start[i] is not None:
            yields.append((i, yield_start[i], idle_from))
            yield_start[i] = None
        become_idle(i, idle_from, hold_after[i])

    for tick in range(int(round(t_max / game.DT))):
        t = round(tick * game.DT, 10)
        move = [0] * n
        for i in range(n):
            act = None
            if mode[i] == MOVE:
                if has_watch:
                    act = players[i].watch(t, k[i] / 10, hist[i], mem[i])
                if act is None:
                    move[i] = 1 if target[i] > k[i] else -1
                    if k[i] + move[i] == target[i]:
                        finish_move(i, tick + 1)
                    continue
            elif mode[i] == WAIT:
                if tick < until[i]:
                    if has_interrupt and tick >= hold_end[i]:
                        act = players[i].interrupt(t, k[i] / 10, hist[i], mem[i])
                    if act is None:
                        continue
                else:
                    act = players[i].decide(t, k[i] / 10, hist[i], mem[i])
            else:
                act = players[i].decide(t, k[i] / 10, hist[i], mem[i])
            if act[0] != "idle":
                waiting[i] = False
            if slip:
                act = slipped(act, k[i], noise, slip)
            if act[0] == "step":
                move[i] = act[1]
                mode[i] = DECIDE
            elif act[0] == "stay":
                mode[i] = DECIDE
            elif act[0] == "idle":
                become_idle(i, tick, 0)
            else:                                            # ("target", x, hold, is_yield)
                target[i], hold_after[i] = int(round(10 * act[1])), act[2]
                yield_start[i] = tick if act[3] else None
                if not act[3] and abs(target[i]) == game.WALL and k[i] * target[i] < 0:   # greedy crossing
                    total = int(k.sum())
                    cross["tie" if total == 0 else "toward" if total * target[i] > 0 else "away"] += 1
                if target[i] == k[i]:
                    finish_move(i, tick)
                    continue
                move[i] = 1 if target[i] > k[i] else -1
                mode[i] = MOVE
                if k[i] + move[i] == target[i]:
                    finish_move(i, tick + 1)
        if fidget:
            for i in range(n):
                if waiting[i] and move[i] == 0 and noise.random() < fidget * game.DT:
                    move[i] = 1 if noise.random() < 0.5 else -1
                    n_fidget += 1
        k = np.clip(k + np.array(move), -game.WALL, game.WALL)
        agr = game.agreement(A, k, display)
        for i in range(n):
            hist[i].append(float(agr[i]))
            mem[i]["s_history"].append(k[i] / 10)
        path.append(k)
        if game.is_consensus(k):
            t_cons = round((tick + 1) * game.DT, 10)
            break
    path = np.array(path)
    for i in range(n):                                       # a yield still on its way at the end
        if yield_start[i] is not None:
            yields.append((i, yield_start[i], len(path) - 1))
    out = row(A, path, t_cons, yields, cls.name, params, cross)
    out.update(strategy_log(cls.name, mem, path, n_fidget))
    if keep_path:
        out["path"] = path
        out["memory"] = [{key: v for key, v in m.items() if key not in ("rng", "s_history")} for m in mem]
        out["yields"] = yields                      # (player, first tick, first idle tick) per yield
    return out


def strategy_log(name, mem, path, n_fidget):
    """Per-run fields of the S3.2 strategies, from the players' memories after the round (and the path).
    S3: n_seeds, t_all_reached (when the last player left WAIT; NaN if one never did), n_followers, n_reversals
        (travel checks), n_wrong_wall (followers whose first wall after leaving WAIT is opposite to the consensus
        side; NaN without consensus), n_fidgets.
    S6 / S7: decisions_per_min (decisions per player per minute played; S7: slow decisions only), mean_W (mean W at
        those decisions). S7 also: slow_share (mean over players of the share of the round spent slow), n_switches
        (fast -> slow switches)."""
    t_end = (len(path) - 1) * game.DT
    if name == "S3":
        consensus = game.is_consensus(path[-1])
        final = np.sign(path[-1].sum())
        reach = [m.get("t_reach", np.nan) for m in mem]
        wrong = 0
        for i, m in enumerate(mem):
            if m.get("follower"):
                walls = np.flatnonzero(np.abs(path[int(round(m["t_reach"] / game.DT)):, i]) == game.WALL)
                first = path[int(round(m["t_reach"] / game.DT)) + walls[0], i] if len(walls) else 0
                wrong += int(np.sign(first) != final)
        return {"n_seeds": sum(bool(m.get("seeded")) for m in mem), "t_all_reached": float(np.max(reach)),
                "n_followers": sum(bool(m.get("follower")) for m in mem),
                "n_reversals": sum(m.get("n_rev", 0) for m in mem),
                "n_wrong_wall": wrong if consensus else np.nan, "n_fidgets": n_fidget}
    if name in ("S6", "S7"):
        n_dec = sum(m.get("n_dec", 0) for m in mem)
        out = {"decisions_per_min": n_dec / len(mem) / (t_end / 60),
               "mean_W": sum(m.get("sum_W", 0.0) for m in mem) / n_dec if n_dec else np.nan}
        if name == "S7":
            slow = [m.get("slow_time", 0.0) + (t_end - m["slow_from"] if m.get("slow") else 0.0) for m in mem]
            out.update(slow_share=float(np.mean(slow)) / t_end,
                       n_switches=sum(m.get("n_switch", 0) for m in mem))
        return out
    return {}


def probe_table(A, path, memory):
    """Probes logged by S1p / S1p_v1 (memory["probe_log"]) in one run kept with keep_path: player, tick (probe
    start), sigma, true_sign (sign of s_i + sum of the neighbours' s at the probe start; 0 if the sum is 0),
    nbr_moved (a neighbour moved during the probe tick) and wait_s (time waited for quiet)."""
    S = path @ (A + np.eye(len(A))).T
    rows = []
    for i, m in enumerate(memory):
        for tick, sigma, wait in m.get("probe_log", []):
            moved = bool(np.any((path[tick + 1] != path[tick]) & (A[i] > 0)))
            rows.append({"player": i, "tick": tick, "sigma": sigma, "true_sign": int(np.sign(S[tick, i])),
                         "nbr_moved": moved, "wait_s": wait * game.DT})
    return pd.DataFrame(rows, columns=["player", "tick", "sigma", "true_sign", "nbr_moved", "wait_s"])


def collisions(yields):
    """Pairs of yields by different players whose moves [first tick, first idle tick) overlap in time."""
    c = 0
    for a in range(len(yields)):
        for b in range(a + 1, len(yields)):
            (i, s1, e1), (j, s2, e2) = yields[a], yields[b]
            if i != j and s1 < e2 and s2 < e1:
                c += 1
    return c


def outcome_class(path, success):
    """(class, period). consensus; fixed trap = the sign pattern (sign of each s_i) is constant over the last 30 s,
    even if sliders wiggle; periodic = the sign-pattern sequence of the last 120 s (consecutive repeats merged)
    repeats with a fixed period p >= 2 (at least 2 distinct patterns), at least twice; else irregular."""
    if success:
        return "consensus", 0
    last = np.sign(path[-(int(FIXED_S / game.DT) + 1):])
    if np.all(last == last[0]):
        return "fixed trap", 0
    signs = [tuple(r) for r in np.sign(path[-(int(CYCLE_S / game.DT) + 1):])]
    seq = [signs[0]] + [b for a, b in zip(signs, signs[1:]) if a != b]
    for p in range(2, len(seq) // 2 + 1):
        if all(seq[j] == seq[j + p] for j in range(len(seq) - p)):
            return "periodic", p
    return "irregular", 0


def row(A, path, t_cons, yields, strategy, params, cross):
    """The per-run row. moving_per_tick = mean number of sliders that move in a tick (probes included).
    cross_toward / cross_away / cross_tie: greedy (non-yield) moves to the wall on the other side of 0 from the
    player's current position, by the sign of the sum of all 7 s at the decision tick (tie: the sum is 0).
    breakups (tail.breakups), breakups_per_min (per minute played: t_consensus, or t_max without consensus);
    yields_majority / yields_sided (yields by majority-side players / yields with sum of s != 0) and
    yields_undone (back on the old side within 10 s), as in tail.yield_flags; *_share = their ratios (NaN if none)."""
    success = not np.isnan(t_cons)
    cls, period = outcome_class(path, success)
    flags = [tail.yield_flags(path, i, start, end) for i, start, end in yields]
    sided = [m for m, _ in flags if not np.isnan(m)]
    n_break = tail.breakups(path)
    minutes = (len(path) - 1) * game.DT / 60
    return {"strategy": strategy, "params": json.dumps(params, sort_keys=True),
            "success_180": bool(success and t_cons <= 180), "success_300": bool(success and t_cons <= 300),
            "t_consensus": t_cons, "frac_time_fraught": float(np.mean(game.fraught_now(A, path[:-1]))),
            "n_yields": len(yields), "n_collisions": collisions(yields), "outcome": cls, "period": period,
            "moving_per_tick": float(np.mean(np.sum(path[1:] != path[:-1], axis=1))),
            "cross_toward": cross["toward"], "cross_away": cross["away"], "cross_tie": cross["tie"],
            "breakups": n_break, "breakups_per_min": n_break / minutes,
            "yields_majority": int(sum(sided)), "yields_sided": len(sided),
            "yields_undone": int(sum(u for _, u in flags)),
            "yield_majority_share": float(np.mean(sided)) if sided else np.nan,
            "yield_undone_share": float(np.mean([u for _, u in flags])) if flags else np.nan}

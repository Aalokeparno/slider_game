"""Path 3 simulator: 7 strategy players on a graph, lab v09 ticks (game.py).

One tick n (time n * dt): every player acts on what it has seen up to now (its Agr history ends with Agr(n)),
all moves are applied at once, then Agr(n + 1) is computed and appended to each player's history.
A player is in one of three modes:
    decide   decide() is called this tick
    move     going to a target, one step per tick (no calls)
    wait     idle until tick `until` (first the hold, then the reaction delay); interrupt() is called on every
             idle tick after the hold, if the strategy has one
Reaction delay (whenever a player becomes idle): a human halt duration resampled (human.load_delays),
rounded to ticks, at least 1. The first decision of the round: a resampled human start delay, the same way.
The round ends at consensus (game.is_consensus) or at t_max.
"""
import json

import numpy as np

from slider.path3 import game

DECIDE, MOVE, WAIT = 0, 1, 2
FIXED_S, CYCLE_S = 30, 120            # outcome classes: last 30 s (fixed trap), last 120 s (periodic)


def ticks(seconds):
    return max(1, int(round(seconds / game.DT)))


def run(A, cls, params, start_k, delays, rng, t_max=300, display="int", keep_path=False):
    """One round. start_k: start positions (integers, s = k / 10). delays: {"halt": array, "start": array} in s.
    Returns the per-run row (row()); with keep_path also "path" (positions k per tick, (ticks + 1) x 7)."""
    n = len(A)
    k = np.array(start_k, dtype=int)
    players = [cls(**params) for _ in range(n)]
    has_interrupt = hasattr(players[0], "interrupt")
    hist = [[float(a)] for a in game.agreement(A, k, display)]
    mem = [{"rng": np.random.default_rng(rng.integers(2**63)), "s_history": [k[i] / 10]} for i in range(n)]
    mode = [WAIT] * n
    until = [ticks(rng.choice(delays["start"])) for _ in range(n)]
    hold_end = [0] * n
    target, hold_after, yield_start = [0] * n, [0.0] * n, [None] * n
    yields = []                                   # (player, first tick, first idle tick)
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
            if mode[i] == MOVE:
                move[i] = 1 if target[i] > k[i] else -1
                if k[i] + move[i] == target[i]:
                    finish_move(i, tick + 1)
                continue
            act = None
            if mode[i] == WAIT:
                if tick < until[i]:
                    if has_interrupt and tick >= hold_end[i]:
                        act = players[i].interrupt(t, k[i] / 10, hist[i], mem[i])
                    if act is None:
                        continue
                else:
                    act = players[i].decide(t, k[i] / 10, hist[i], mem[i])
            else:
                act = players[i].decide(t, k[i] / 10, hist[i], mem[i])
            if act[0] == "step":
                move[i] = act[1]
                mode[i] = DECIDE
            elif act[0] == "idle":
                become_idle(i, tick, 0)
            else:                                            # ("target", x, hold, is_yield)
                target[i], hold_after[i] = int(round(10 * act[1])), act[2]
                yield_start[i] = tick if act[3] else None
                if target[i] == k[i]:
                    finish_move(i, tick)
                    continue
                move[i] = 1 if target[i] > k[i] else -1
                mode[i] = MOVE
                if k[i] + move[i] == target[i]:
                    finish_move(i, tick + 1)
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
    out = row(A, path, t_cons, yields, cls.name, params)
    if keep_path:
        out["path"] = path
    return out


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
    """(class, period). consensus; fixed trap = no slider change in the last 30 s; periodic = the sign-pattern
    sequence of the last 120 s (consecutive repeats merged) repeats with a fixed period p, at least twice
    (p = 1: one sign pattern held all the time while sliders still move); else irregular."""
    if success:
        return "consensus", 0
    last = path[-(int(FIXED_S / game.DT) + 1):]
    if np.all(last == last[0]):
        return "fixed trap", 0
    signs = [tuple(r) for r in np.sign(path[-(int(CYCLE_S / game.DT) + 1):])]
    seq = [signs[0]] + [b for a, b in zip(signs, signs[1:]) if a != b]
    if len(seq) == 1:
        return "periodic", 1
    for p in range(2, len(seq) // 2 + 1):
        if all(seq[j] == seq[j + p] for j in range(len(seq) - p)):
            return "periodic", p
    return "irregular", 0


def row(A, path, t_cons, yields, strategy, params):
    success = not np.isnan(t_cons)
    cls, period = outcome_class(path, success)
    return {"strategy": strategy, "params": json.dumps(params, sort_keys=True),
            "success_180": bool(success and t_cons <= 180), "success_300": bool(success and t_cons <= 300),
            "t_consensus": t_cons, "frac_time_fraught": float(np.mean(game.fraught_now(A, path[:-1]))),
            "n_yields": len(yields), "n_collisions": collisions(yields), "outcome": cls, "period": period}

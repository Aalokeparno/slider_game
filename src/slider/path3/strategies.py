"""Path 3 strategies. A strategy sees only its own inputs:

    decide(t, s_i, agr_history, memory) -> action
        t            time now (s)
        s_i          its own slider position (on the 0.1 grid)
        agr_history  its own displayed Agreement, one value per tick from t = 0 to now
        memory       its own dict: memory["rng"] (its random numbers), memory["s_history"] (its own positions,
                     one per tick) and whatever the strategy stores there

No graph, no neighbours, no IDs, no other players' positions. This module imports nothing from the game.

Actions (the simulator carries them out):
    ("step", d)                     move one 0.1 step (d = +1 / -1) this tick; decide is called again next tick
    ("idle",)                       stay; the next decision comes after a reaction delay
    ("stay",)                       stay this tick; decide is called again next tick
    ("target", x, hold, is_yield)   move to x, one step per tick; on arrival hold for `hold` s (no decisions),
                                    then idle (reaction delay); is_yield marks a yield
A strategy may also define interrupt(t, s_i, agr_history, memory), called every tick while it is idle
(not while it holds); it returns an action or None.

Probe (shared): step one 0.1 step (inward at a wall, random direction mid-slider), wait one tick, compare Agr
with the value before: rose -> "helps", fell or equal -> "doesn't help".
"""
import math

IDLE = ("idle",)
STAY = ("stay",)


def at_wall(s):
    return abs(s) >= 1 - 1e-9


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


class Strategy:
    name = ""
    MEMORY_KEYS = {"probe", "helped"}

    def start_probe(self, s_i, agr_history, memory):
        d = -sign(s_i) if at_wall(s_i) else (1 if memory["rng"].random() < 0.5 else -1)
        memory["probe"] = (d, agr_history[-1], s_i)
        return ("step", d)

    def end_probe(self, agr_history, memory):
        """(direction, Agr before, position before, rose, fell) of the probe that just ended."""
        d, before, s0 = memory.pop("probe")
        now = agr_history[-1]
        memory["helped"] = now > before
        return d, before, s0, now > before, now < before

    def greedy(self, d, s0, rose, fell):
        """S0a after a probe: helps -> the wall in that direction; mid-slider and fell -> the opposite wall;
        otherwise step back (to s0) and go idle. Returns None in the 'otherwise' case."""
        if rose:
            return ("target", float(d), 0.0, False)
        if not at_wall(s0) and fell:
            return ("target", float(-d), 0.0, False)
        return None

    def yield_action(self, s0, d):
        """Yield: move to the wall on the other side of where the player was before the probe, then hold H."""
        side = sign(s0) if s0 != 0 else -d
        return ("target", float(-side), self.H, True)


class S0a(Strategy):
    """Greedy: if Agr = 1, do nothing; else probe and follow it (Strategy.greedy)."""
    name = "S0a"

    def __init__(self):
        pass

    def decide(self, t, s_i, agr_history, memory):
        if "probe" in memory:
            d, before, s0, rose, fell = self.end_probe(agr_history, memory)
            return self.greedy(d, s0, rose, fell) or ("target", s0, 0.0, False)
        if agr_history[-1] >= 1:
            return IDLE
        return self.start_probe(s_i, agr_history, memory)


class S0a_hold(Strategy):
    """S0a with commitment, no noise: every greedy move (to a wall, including the first move from 0) ends with a
    hold of H s (no decisions, no probes), then the usual reaction delay. A step back after a failed probe is not
    a greedy move and has no hold."""
    name = "S0a_hold"

    def __init__(self, H=4.0):
        self.H = H

    def decide(self, t, s_i, agr_history, memory):
        if "probe" in memory:
            d, before, s0, rose, fell = self.end_probe(agr_history, memory)
            act = self.greedy(d, s0, rose, fell)
            if act:
                return ("target", act[1], self.H, False)
            return ("target", s0, 0.0, False)
        if agr_history[-1] >= 1:
            return IDLE
        return self.start_probe(s_i, agr_history, memory)


class S0b(Strategy):
    """S0b'': S0a, plus: if Agr < 1 and the probe does not help (the S0a 'otherwise' case), with prob eps
    yield (move to the opposite wall), then hold H. Clock phase t0: while the round clock t < t0 it never
    yields (and draws no yield random number), so it plays exactly as S0a; t0 = 0 is the original S0b''.
    Soft gate (S0b'''): with soft_gate=True the yield chance is eps * (1 - Agr) instead of eps, Agr = the displayed
    Agr on the tick before the probe; the same single random number is drawn, so runs pair with S0b''."""
    name = "S0b"

    def __init__(self, eps=0.5, H=4.0, t0=0.0, soft_gate=False):
        self.eps, self.H, self.t0, self.soft_gate = eps, H, t0, soft_gate

    def decide(self, t, s_i, agr_history, memory):
        if "probe" in memory:
            d, before, s0, rose, fell = self.end_probe(agr_history, memory)
            act = self.greedy(d, s0, rose, fell)
            if act:
                return act
            chance = self.eps * (1 - before) if self.soft_gate else self.eps
            if t >= self.t0 and before < 1 and memory["rng"].random() < chance:
                return self.yield_action(s0, d)
            return ("target", s0, 0.0, False)
        if agr_history[-1] >= 1:
            return IDLE
        return self.start_probe(s_i, agr_history, memory)


class S2_v1(Strategy):
    """S2 v1, stuck timer + yield + hold: S0a, plus a flat clock = time since Agr last changed at a tick when the
    player itself did not move. Stuck = flat >= T, Agr < 1 and the last probe did not help.
    While stuck (checked every idle tick and at each decision), yield with prob p * dt per tick, then hold H.
    memory["idle_ticks"] / ["stuck_ticks"] count the idle ticks (interrupt calls) and those spent stuck."""
    name = "S2_v1"
    MEMORY_KEYS = Strategy.MEMORY_KEYS | {"scanned", "last_change", "idle_ticks", "stuck_ticks"}

    def __init__(self, T=1.3, p=0.1, H=4.0):
        self.T, self.p, self.H = T, p, H

    def flat(self, t, agr_history, memory):
        """Seconds since the last tick j where Agr changed and the player's own position did not."""
        s_hist = memory["s_history"]
        j = memory.get("scanned", 0) + 1
        while j < len(agr_history):
            if agr_history[j] != agr_history[j - 1] and s_hist[j] == s_hist[j - 1]:
                memory["last_change"] = j
            j += 1
        memory["scanned"] = len(agr_history) - 1
        return t - 0.1 * memory.get("last_change", 0)

    def is_stuck(self, t, agr_history, memory):
        return (self.flat(t, agr_history, memory) >= self.T - 1e-9 and agr_history[-1] < 1
                and memory.get("helped") is False)

    def stuck_yield(self, t, s_i, agr_history, memory, stuck):
        """A yield, or None (one chance per tick while stuck)."""
        if stuck and memory["rng"].random() < self.p * 0.1:
            return self.yield_action(s_i, 1 if memory["rng"].random() < 0.5 else -1)
        return None

    def interrupt(self, t, s_i, agr_history, memory):
        stuck = self.is_stuck(t, agr_history, memory)
        memory["idle_ticks"] = memory.get("idle_ticks", 0) + 1
        memory["stuck_ticks"] = memory.get("stuck_ticks", 0) + stuck
        return self.stuck_yield(t, s_i, agr_history, memory, stuck)

    def decide(self, t, s_i, agr_history, memory):
        if "probe" in memory:
            d, before, s0, rose, fell = self.end_probe(agr_history, memory)
            return self.greedy(d, s0, rose, fell) or ("target", s0, 0.0, False)
        act = self.stuck_yield(t, s_i, agr_history, memory, self.is_stuck(t, agr_history, memory))
        if act:
            return act
        if agr_history[-1] >= 1:
            return IDLE
        return self.start_probe(s_i, agr_history, memory)


class S2(S2_v1):
    """S2 v2: S2_v1 with a blip-tolerant flat clock. The player keeps a reference A_ref of its displayed Agr.
    The clock resets only when the displayed Agr differs from A_ref for B consecutive ticks in which the player
    did not move (reset at the B-th such tick); then A_ref = the new value. Changes that revert sooner (a
    neighbour's out-and-back probe) are ignored. Own moves never count: at a tick where the player moved,
    A_ref takes the new value and the count starts again."""
    name = "S2"
    MEMORY_KEYS = S2_v1.MEMORY_KEYS | {"a_ref", "differ"}

    def __init__(self, T=1.3, p=0.1, H=4.0, B=3):
        super().__init__(T, p, H)
        self.B = B

    def flat(self, t, agr_history, memory):
        s_hist = memory["s_history"]
        if "a_ref" not in memory:
            memory["a_ref"], memory["differ"] = agr_history[0], 0
        j = memory.get("scanned", 0) + 1
        while j < len(agr_history):
            if s_hist[j] != s_hist[j - 1]:                  # own move
                memory["a_ref"], memory["differ"] = agr_history[j], 0
            elif agr_history[j] != memory["a_ref"]:
                memory["differ"] += 1
                if memory["differ"] >= self.B:
                    memory["last_change"] = j
                    memory["a_ref"], memory["differ"] = agr_history[j], 0
            else:
                memory["differ"] = 0
            j += 1
        memory["scanned"] = len(agr_history) - 1
        return t - 0.1 * memory.get("last_change", 0)


class S1p_v1(Strategy):
    """S1' v1 amplified averaging: probe to get sigma = the sign of the neighbourhood sum (helps -> the probe
    direction, else the opposite); target = clip(kappa * sigma * Agr, -1, 1), Agr = the value before the probe,
    put on the 0.1 grid by randomized rounding; move there, go idle.
    memory["probe_log"]: one (start tick, sigma, ticks waited for quiet) per probe (v1 never waits)."""
    name = "S1p_v1"
    MEMORY_KEYS = Strategy.MEMORY_KEYS | {"probe_tick", "probe_log", "wait_from", "wait"}

    def __init__(self, kappa=1.05):
        self.kappa = kappa

    def probe(self, t, s_i, agr_history, memory, wait):
        memory["probe_tick"], memory["wait"] = len(agr_history) - 1, wait
        return self.start_probe(s_i, agr_history, memory)

    def finish(self, agr_history, memory):
        d, before, s0, rose, fell = self.end_probe(agr_history, memory)
        sigma = d if rose else -d
        memory.setdefault("probe_log", []).append((memory.pop("probe_tick"), sigma, memory.pop("wait")))
        x = round(10 * min(1.0, max(-1.0, self.kappa * sigma * before)), 9)
        k = math.floor(x)
        if memory["rng"].random() < x - k:
            k += 1
        return ("target", k / 10, 0.0, False)

    def decide(self, t, s_i, agr_history, memory):
        if "probe" in memory:
            return self.finish(agr_history, memory)
        return self.probe(t, s_i, agr_history, memory, 0)


class S1p(S1p_v1):
    """S1' v2, quiet probe: a probe may start only if the displayed Agr has been unchanged for the last Q ticks
    while the player did not move; if not, the player stays and checks again each tick. The rest as S1p_v1."""
    name = "S1p"

    def __init__(self, kappa=1.05, Q=2):
        super().__init__(kappa)
        self.Q = Q

    def quiet(self, agr_history, memory):
        s_hist, n = memory["s_history"], len(agr_history) - 1
        if n < self.Q:
            return False
        return all(agr_history[j] == agr_history[j - 1] and s_hist[j] == s_hist[j - 1]
                   for j in range(n - self.Q + 1, n + 1))

    def decide(self, t, s_i, agr_history, memory):
        if "probe" in memory:
            return self.finish(agr_history, memory)
        now = len(agr_history) - 1
        memory.setdefault("wait_from", now)
        if not self.quiet(agr_history, memory):
            return STAY
        return self.probe(t, s_i, agr_history, memory, now - memory.pop("wait_from"))


STRATEGIES = {"S0a": S0a, "S0a_hold": S0a_hold, "S0b": S0b, "S2": S2, "S1p": S1p, "S2_v1": S2_v1, "S1p_v1": S1p_v1}

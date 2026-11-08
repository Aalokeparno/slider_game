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

    def yield_ok(self, t, memory):
        return t >= self.t0

    def weight(self, t, memory):
        """Factor on the yield chance (1 here; S4 changes it)."""
        return 1.0

    def after_probe(self, t, agr_history, memory):
        """(action, kind) when a probe ends; kind = "greedy", "yield" or "back" (step back to s0)."""
        d, before, s0, rose, fell = self.end_probe(agr_history, memory)
        act = self.greedy(d, s0, rose, fell)
        if act:
            return act, "greedy"
        chance = self.eps * (1 - before) if self.soft_gate else self.eps
        if self.yield_ok(t, memory) and before < 1 and memory["rng"].random() < chance * self.weight(t, memory):
            return self.yield_action(s0, d), "yield"
        return ("target", s0, 0.0, False), "back"

    def decide(self, t, s_i, agr_history, memory):
        if "probe" in memory:
            return self.after_probe(t, agr_history, memory)[0]
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


# ------------------------------------------------------------------ S3.2: S4, S6, S7 (on S0b''') and S3 (one-seed wave)

class S4(S0b):
    """S4 seniority flags on S0b''' (soft gate, eps 0.75, H 4). Yield chance eps * (1 - Agr) * w, where the age
    a = time since the player's own last side change. Side = the sign of its first wall; a side change = the tick
    where s first has the other sign (so probes and the first move from 0 never count). Never changed side: w = 1.
        mode "old": w = min(1, a / A)                     (A = 0: w = 1)
        mode "new": w = max(0.05, 1 - min(1, a / A))      (control; A = inf: w = 1)
    Greedy flips are not damped; the hold after a yield is unchanged."""
    name = "S4"
    MEMORY_KEYS = Strategy.MEMORY_KEYS | {"side", "flip", "seen"}

    def __init__(self, mode="old", A=10.0, eps=0.75, H=4.0, soft_gate=True):
        super().__init__(eps, H, 0.0, soft_gate)
        self.mode, self.A = mode, A

    def age(self, t, memory):
        """Seconds since the last side change, or None if the player never changed side."""
        s = memory["s_history"]
        side, j = memory.get("side", 0), memory.get("seen", 0)
        while j < len(s):
            if side == 0 and at_wall(s[j]):
                side = sign(s[j])
            elif side != 0 and sign(s[j]) == -side:
                side, memory["flip"] = -side, j
            j += 1
        memory["side"], memory["seen"] = side, j
        return t - 0.1 * memory["flip"] if "flip" in memory else None

    def weight(self, t, memory):
        a = self.age(t, memory)
        if a is None:
            return 1.0
        x = min(1.0, a / self.A) if self.A > 0 else 1.0
        return x if self.mode == "old" else max(0.05, 1 - x)


class S6(S0b):
    """S6 exponential backoff on S0b''' (soft gate, eps 0.75, H 4). Each player keeps a wait W (start w_min).
    After each decision (a probe's outcome, or "Agr = 1, do nothing") the player waits uniform [W/2, W] (rand=False:
    exactly W) before the usual reaction delay; after a yield the wait comes on top of the hold H.
    W is updated before the wait is drawn: a greedy move (the probe helped) -> W = w_min; any other probe outcome
    (yield or step back) -> W = min(2 W, W_max); "Agr = 1" leaves W as it is.
    W = 0 adds nothing and draws no random number (w_min = W_max = 0 is S0b''').
    memory["n_dec"] / ["sum_W"]: decisions and the sum of W at them (for the log)."""
    name = "S6"
    MEMORY_KEYS = Strategy.MEMORY_KEYS | {"W", "n_dec", "sum_W"}

    def __init__(self, w_min=0.5, W_max=8.0, rand=True, eps=0.75, H=4.0, soft_gate=True):
        super().__init__(eps, H, 0.0, soft_gate)
        self.w_min, self.W_max, self.rand = w_min, W_max, rand

    def wait(self, act, s_i, memory):
        W = memory["W"]
        memory["n_dec"] = memory.get("n_dec", 0) + 1
        memory["sum_W"] = memory.get("sum_W", 0.0) + W
        if W <= 0:
            return act
        extra = memory["rng"].uniform(W / 2, W) if self.rand else W
        if act[0] == "idle":
            return ("target", s_i, extra, False)
        return ("target", act[1], act[2] + extra, act[3])

    def decide(self, t, s_i, agr_history, memory):
        memory.setdefault("W", self.w_min)
        if "probe" in memory:
            act, kind = self.after_probe(t, agr_history, memory)
            memory["W"] = self.w_min if kind == "greedy" else min(2 * memory["W"], self.W_max)
            return self.wait(act, s_i, memory)
        if agr_history[-1] >= 1:
            return self.wait(IDLE, s_i, memory)
        return self.start_probe(s_i, agr_history, memory)


class S7(S6):
    """S7 fast/slow mix. At its first decision a player is slow with prob q (q >= 1 or <= 0: no random number),
    else fast. Fast = S0a (probe and follow, never yields). Slow = S6. A fast player whose probe failed (no greedy
    move) N decisions in a row becomes slow from its next decision (W = w_min) until its next helpful probe, then
    fast again. A decision with Agr = 1 breaks the run of failures. Players slow from the draw stay slow.
    memory: "slow", "switched" (slow by a switch), "fails", "slow_from" / "slow_time" (time slow, for the log),
    "n_switch" (fast -> slow switches)."""
    name = "S7"
    MEMORY_KEYS = S6.MEMORY_KEYS | {"slow", "switched", "fails", "slow_from", "slow_time", "n_switch"}

    def __init__(self, q=2 / 7, N=5, w_min=0.5, W_max=8.0, rand=True, eps=0.75, H=4.0, soft_gate=True):
        super().__init__(w_min, W_max, rand, eps, H, soft_gate)
        self.q, self.N = q, N

    def set_slow(self, t, slow, memory):
        if slow and not memory["slow"]:
            memory["slow_from"] = t
        if not slow and memory["slow"]:
            memory["slow_time"] = memory.get("slow_time", 0.0) + t - memory["slow_from"]
        memory["slow"] = slow

    def decide(self, t, s_i, agr_history, memory):
        if "slow" not in memory:
            memory["slow"] = False
            self.set_slow(0.0, self.q >= 1 or (self.q > 0 and memory["rng"].random() < self.q), memory)
        if memory["slow"]:
            probing = "probe" in memory
            act = super().decide(t, s_i, agr_history, memory)
            if probing and memory.get("switched") and memory["helped"]:
                self.set_slow(t, False, memory)
                memory["switched"], memory["fails"] = False, 0
            return act
        if "probe" in memory:
            d, before, s0, rose, fell = self.end_probe(agr_history, memory)
            act = self.greedy(d, s0, rose, fell)
            memory["fails"] = 0 if act else memory.get("fails", 0) + 1
            if memory["fails"] >= self.N:
                self.set_slow(t, True, memory)
                memory["switched"], memory["W"] = True, self.w_min
                memory["n_switch"] = memory.get("n_switch", 0) + 1
            return act or ("target", s0, 0.0, False)
        if agr_history[-1] >= 1:
            memory["fails"] = 0
            return IDLE
        return self.start_probe(s_i, agr_history, memory)


class S3(S2, S0b):
    """S3 one-seed wave (S3a+: assigned_seed=True; S3b+: assigned_seed=False). memory["state"]:
    WAIT    at 0, no own moves; checked every tick. Reached: displayed Agr >= reach (1/7, read on the integer-percent
            display as 14 %) -> FOLLOW. S3a: the simulator gives one seat memory["seed"]; at its first decision
            (after its start delay) it seeds. S3b: t_seed ~ Exp(rate lam) drawn at the first call; at t_seed it seeds
            unless cancelled (any tick where its Agr changed and its own slider did not move).
            Seeding: a random wall, move there -> WALL.
    FOLLOW  pause ~ Exp(mean `pause`) (0: none), then probe as soon as the Agr was unchanged over the last tick;
            rose -> commit to the wall in the probe direction, fell or equal -> the other wall -> TRAVEL.
    TRAVEL  moving to the committed wall. Travel check (check=True): if the Agr fell on 2 consecutive ticks within
            the first 5 ticks of travel, reverse once and commit to the other wall (watch(), called every moving tick).
    WALL    fallback S0b''' (fallback="S0b": eps, soft gate, H) or S2 v2 (fallback="S2": T, p, H, B), with an arrival
            grace held as a hold: every move to a wall that is an arrival (seed, commit, reversal, greedy flip, yield;
            not the step back after a probe) carries a hold of G s (a yield: max(H, G)), so the player makes no
            decisions and no S2 stuck-yield checks for G s after it arrives. (Changed 2026-11-08 at the user's request
            from the spec "no yields for G s, greedy flips allowed", which failed test 3.)
    Log keys: "t_reach" (time it left WAIT), "follower", "seeded", "n_rev"."""
    name = "S3"
    MEMORY_KEYS = S2.MEMORY_KEYS | {"seed", "state", "t_seed", "cancel", "wscan", "t_reach", "follower", "seeded",
                                    "pause_end", "commit", "n_rev"}

    def __init__(self, assigned_seed=True, lam=0.1, pause=0.3, check=True, G=4.0, fallback="S0b", reach=1 / 7,
                 eps=0.75, H=4.0, soft_gate=True, T=1.3, p=0.1, B=3):
        S2.__init__(self, T, p, H, B)
        self.eps, self.soft_gate, self.t0 = eps, soft_gate, 0.0
        self.assigned_seed, self.lam, self.pause, self.check, self.G = assigned_seed, lam, pause, check, G
        self.fallback, self.reach = fallback, reach

    # ---- WAIT
    def in_wait(self, t, s_i, agr_history, memory, deciding):
        s, j = memory["s_history"], memory.get("wscan", 0) + 1
        while j < len(agr_history):
            if agr_history[j] != agr_history[j - 1] and s[j] == s[j - 1]:
                memory["cancel"] = True
            j += 1
        memory["wscan"] = len(agr_history) - 1
        if agr_history[-1] >= round(100 * self.reach) / 100 - 1e-9:
            memory.update(state="FOLLOW", t_reach=t, follower=True)
            memory["pause_end"] = t + (memory["rng"].exponential(self.pause) if self.pause > 0 else 0.0)
            return STAY
        if self.assigned_seed:
            seed_now = deciding and memory.get("seed", False)
        else:
            if "t_seed" not in memory:
                memory["t_seed"] = memory["rng"].exponential(1 / self.lam)
            seed_now = t >= memory["t_seed"] and not memory.get("cancel", False)
        if seed_now:
            w = 1.0 if memory["rng"].random() < 0.5 else -1.0
            memory.update(state="WALL", t_reach=t, seeded=True)
            return ("target", w, self.G, False)
        return IDLE if deciding else None

    # ---- FOLLOW / TRAVEL
    def follow(self, t, s_i, agr_history, memory):
        if "probe" in memory:
            d, before, s0, rose, fell = self.end_probe(agr_history, memory)
            w = float(d if rose else -d)
            memory.update(state="TRAVEL", commit=(len(agr_history) - 1, w))
            return ("target", w, self.G, False)
        if t >= memory["pause_end"] - 1e-9 and agr_history[-1] == agr_history[-2]:
            return self.start_probe(s_i, agr_history, memory)
        return STAY

    def watch(self, t, s_i, agr_history, memory):
        if not self.check or memory.get("state") != "TRAVEL":
            return None
        n0, w = memory["commit"]
        m = len(agr_history) - 1 - n0
        if 2 <= m <= 5 and agr_history[-1] < agr_history[-2] < agr_history[-3]:
            memory.update(state="WALL", n_rev=1, commit=(n0, -w))
            return ("target", -w, self.G, False)
        return None

    # ---- WALL
    def grace(self, s_i, act):
        """A move to the other wall (greedy flip or yield; not a step back) gets a hold of at least G."""
        if act is not None and act[0] == "target" and at_wall(act[1]) and abs(act[1] - s_i) > 0.15:
            return ("target", act[1], max(act[2], self.G), act[3])
        return act

    def decide(self, t, s_i, agr_history, memory):
        state = memory.setdefault("state", "WAIT")
        if state == "WAIT":
            return self.in_wait(t, s_i, agr_history, memory, True)
        if state == "FOLLOW":
            return self.follow(t, s_i, agr_history, memory)
        memory["state"] = "WALL"
        base = S2 if self.fallback == "S2" else S0b
        return self.grace(s_i, base.decide(self, t, s_i, agr_history, memory))

    def interrupt(self, t, s_i, agr_history, memory):
        state = memory.setdefault("state", "WAIT")
        if state == "WAIT":
            return self.in_wait(t, s_i, agr_history, memory, False)
        if self.fallback == "S2":
            return self.grace(s_i, S2.interrupt(self, t, s_i, agr_history, memory))
        return None


STRATEGIES = {"S0a": S0a, "S0a_hold": S0a_hold, "S0b": S0b, "S2": S2, "S1p": S1p, "S2_v1": S2_v1, "S1p_v1": S1p_v1,
              "S3": S3, "S4": S4, "S6": S6, "S7": S7}

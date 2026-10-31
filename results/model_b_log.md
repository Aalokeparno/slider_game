# Model B log

One dated entry per Model B version, with its exact spec. Model B is EXPLORATORY: no pre-registered decision rests on it; the frozen Path A presets (L1, final, L1_M3, L0, APEM) are unchanged and bit-identical (`tests/test_b1.py`, `tests/data/preset_fingerprints.json`).

Data checks before B1 (descriptive, G-train only): B0 `notebooks/path1/pB0_stops.ipynb` (stop chance by state and position), B0b + B2 `notebooks/path1/pB0b_pB2.ipynb` (crossing style as a player trait; try-and-keep).

## Entry 1 — B1 (Oct 30, 2026)

### Spec (as given)

> TASK: Model B, version B1 (EXPLORATORY). Fits on G-train only; simulations on the usual
> L3/L5 designs. Never load Ring/CT. Do not change the frozen presets (L1, final, L1_M3, L0,
> APEM): new code paths only, and the old presets must stay bit-identical.
> Start results/model_b_log.md: one dated entry per Model B version, with its exact spec. B1 is entry 1.
>
> STATES (causal: what a simulator knows at that moment), per moving player:
> - C = crossing attempt: the player has left an end band (|s| >= 0.9; old CL |k| >= 9),
>   has not reversed since, and has not yet reached an end band.
> - E = elsewhere: first trip (no end band reached yet), outbound after a reversal, or
>   in_band (inside an end band but not at the wall).
> - At the wall: no stops (as before).
>
> All probabilities use the F1 convention: per-step stop chance = stops / counted moving
> steps; resume chance per tick = resumes / halted ticks.
>
> 0. DATA CHECK FIRST (G-train, same spells as P17 F1b): do halts inside crossings end faster?
>    - Label each mid-round halt by the state at its start (C or E). Per interface: resume
>      chance per tick q_C and q_E (censoring as in P17), with group-bootstrap 95% CIs
>      (1,000, stratified, seed 573), plus a KM figure of halt duration by state
>      (results/figures/pB1_halts_by_state.png).
>    - RULE (fixed now): if the q_C and q_E CIs are disjoint in >= 2 of the 3 interfaces,
>      B1 uses separate resume chances by state; otherwise one resume chance (L1's). Report which.
> 1. B1 PARAMETERS (G-train, per interface: lab v09, old CL, CC), with CIs as above:
>    - pi, rho: beta-binomial smooth share and overdispersion (already in pB0b_crossing_style.csv; reuse).
>    - q_sg: stop chance per step in non-smooth completed crossings (pB0b).
>    - q_E: stop chance per step in state E (new: stops / counted steps over first trip +
>      outbound after a reversal + in_band).
>    - resume: per step 0's rule.
>    - Save as results/tables/pB1_params.csv.
> 2. SIMULATOR (preset "B1"; base = L1 with F1-F3: M2' per bin, lam_start, coin flip; no L1b, no L3):
>    - Player trait: each simulated player draws U_i ~ Uniform(0,1) once per group copy per
>      replicate. In each round, its smooth propensity is p_i = Beta quantile at U_i, with
>      that round's interface (a = pi(1-rho)/rho, b = (1-pi)(1-rho)/rho).
>    - When a player enters state C, draw the crossing's style: smooth with probability
>      p_i, else stop-and-go. Smooth -> stop chance 0 in C; stop-and-go -> q_sg per step.
>    - In state E: stop chance q_E per step.
>    - Resume per tick: q_C or q_E by the state at the halt start (or one value, per the rule).
>    - Tests: all old presets bit-identical; with p_i = 0, q_sg = q_E = L1's per-step stop
>      chance and a single resume chance, B1 equals L1 exactly; unit tests for the state
>      labels and the style draw.
> 3. BUILD CHECK (recovery): simulate B1 on the G6train design, R = 2, seed 573. Re-estimate
>    pi, rho, q_sg, q_E and the resume chance(s) with the same code as steps 0-1.
>    PASS: each within its 95% CI. If anything fails: diagnose and propose; don't fix by changing numbers.
> 4. RUN SCRIPTS: add preset "B1" (seed 6001, registered) to scripts/p24_main.py and to
>    p24_checks.py. Report B1 next to final and L1, labeled EXPLORATORY; no triggers.
>    Smoke-test with R = 2, then delete the outputs.

### Implementation choices (made before any B1 number; full list in `notebooks/path1/pB1_model.ipynb`)

- Halts: P17's F1b halted time, found again row by row (`model_b1.seat_halts`; equal to `motion.motion_table`: 32,263 halts, same durations and exits). A halt has the state of the row that starts it; a reversal counts if its first step is after the phase's first row and at or before that row.
- Halted ticks n = max(1, round(duration / dt)) (`halts.add_ticks`). Resume chance = resumes / halted ticks; other exits censored.
- q_E = stops / counted steps in E (B0 labels: E = not (outbound and reversed = 0)). pi, rho and q_sg are read from `pB0b_crossing_style.csv`.
- Simulator: motion "B1" in `sim.py`, a new code path used only by preset "B1". C runs from the tick whose step leaves an end band (tick detector, |k| from >= 9 to < 9) until the player enters an end band or a reversal confirms whose first step came after that tick. The style is drawn at entry (no draw if p_i is 0 or 1). U is drawn per seat per group copy (`design.simulate_rounds`, a stream spawned after the round streams; `sim.simulate_runs`: per replicate). The stop chance per moving step is the parameter itself.
- Tests (`tests/test_b1.py`, 9): the five old presets bit-identical (fingerprints from commit 4a0813c, `tests/data/preset_fingerprints.json`). B1 with p_i = 0, q_sg = q_E = L1's stop chance and L1's resume chance equals L1 exactly (lab v09, old CL, CC). Halt states on a synthetic path. Simulated states equal the estimator's labels: all smooth gives no stop in C, and q_E = 0 gives no stop in E (all three interfaces). Full suite: 122 passed.

### Results

**Step 0.** Resume chance per tick by state at the halt start (`results/tables/pB1_halts_by_state.csv`, `results/figures/pB1_halts_by_state.png`):

| interface | q_C [95% CI] | q_E [95% CI] | disjoint |
|---|---|---|---|
| lab v09 | 0.1848 [0.1365, 0.2368] | 0.1613 [0.1488, 0.1793] | no |
| old CL | 0.1799 [0.1704, 0.2004] | 0.1706 [0.1497, 0.2246] | no |
| CC | 0.1198 [0.1028, 0.1350] | 0.0578 [0.0491, 0.0712] | yes |

RULE: disjoint in 1 of 3, so **B1 uses one resume chance (L1's: lam_resume dt)**.

**Step 1.** Parameters (`results/tables/pB1_params.csv`, G-train):

| interface | pi | rho | q_sg (per step) | q_E (per step) | lam_resume (1/s, L1) |
|---|---|---|---|---|---|
| lab v09 | 0.429 [0.381, 0.478] | 0.182 [0.103, 0.269] | 0.466 [0.400, 0.526] | 0.446 [0.406, 0.483] | 1.768 [1.477, 2.065] |
| old CL | 0.284 [0.167, 0.430] | 0.400 [0.213, 0.400] | 0.139 [0.112, 0.223] | 0.254 [0.252, 0.255] | 1.736 [1.560, 2.166] |
| CC | 0.378 [0.338, 0.422] | 0.416 [0.363, 0.470] | 0.491 [0.409, 0.558] | 0.503 [0.444, 0.566] | 2.549 [2.183, 2.991] |

**Step 3.** Build check (B1 on G6train, R = 2, seed 573; `results/tables/pB1_recovery.csv`): **PASS 12 of 15**. All stop and resume chances are recovered. Misses: pi lab v09 0.480 (CI to 0.478), pi CC 0.436 (CI to 0.422), rho old CL 0.173 (CI from 0.213).

Diagnosis (descriptive; nothing changed):
1. pi is fit on completed crossings, while B1 draws the style per attempt. Stop-and-go attempts are cut short more often, so completed crossings over-represent smooth ones. This happens in the simulation (completed vs. all attempts: 0.53 vs. 0.46 lab v09, 0.50 vs. 0.43 CC) and in the human data (0.44 vs. 0.35, 0.43 vs. 0.37).
2. The old CL rho CI rests on 14 players from 2 groups. A parametric refit of 14 players gives [0.107, 0.644].

Proposals (not applied):
- (a) A per-attempt player-level mixture fit of pi, rho and q_sg on all attempts.
- (b) Or calibrate pi by simulation.
- (c) A parametric CI for old CL rho, or carry it as a known limitation.

**Step 4.** Preset "B1" registered in `scripts/p24_main.py` (seed 6001) and `scripts/p24_checks.py` (reported next to L1 and final, labeled EXPLORATORY, no triggers). The smoke test (R = 2) ran; its outputs were deleted and `p24_checks.csv` restored. Seen during the smoke test: the summary line for B1 at R = 2 (33 of 53 missed) and the first rows of the pass/miss table.

R = 200 main run (not yet run):

    python scripts/p24_main.py --preset B1 --R 200 --seed 6001 --procs 2
    python scripts/p24_checks.py | tee results/p24_report.txt

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

### Build fix 1 (Oct 30, 2026): style fit per attempt (definition match, not a new version)

**Spec (as given):**

> B1 fixes (before any R = 200 run):
> 1. pi: option (a). Refit pi, rho and q_sg on ALL crossing attempts (state C segments, completed or aborted) with a per-player mixture model: each attempt is smooth (stop chance 0) or stop-and-go (q_sg per step); the player's smooth propensity ~ Beta(pi, rho); likelihood of an attempt with n counted steps and k stops: smooth only if k = 0 (likelihood 1); stop-and-go: q_sg^k (1 - q_sg)^(n - k); attempts censored by the round end are included the same way. Maximum likelihood per interface, group-bootstrap 95% CIs (1,000, stratified, seed 573). Update pB1_params.csv (keep the old values as pB1_params_v1.csv).
> 2. old CL rho: a known limitation (14 players, 2 groups); keep the estimate, add a parametric CI, and note it in the notebook and the log.
> 3. Rerun the B1 recovery once with the new parameters (G6train, R = 2, seed 573): re-estimate pi, rho and q_sg with the NEW per-attempt code. Same pass rule. Report.
> 4. Log this as a B1 build fix (definition match: fit per attempt), not a new version.

**Choices** (`notebooks/path1/pB1_fix1.ipynb`, `model_b1.attempt_table` / `fit_attempts` / `boot_attempts`):
- An attempt is the counted C steps of one outbound segment, any outcome. Players are all (group, seat) with >= 1 attempt.
- The likelihood integrates over p exactly (a polynomial in p).
- The bootstrap refits from the full-data estimate.
- Old CL rho's parametric CI: 1,000 refits of data drawn from the fitted mixture with the human players' own attempt lengths (seed 573).
- v1 is kept: `pB1_params_v1.csv` and `pB1_recovery_v1.csv`. `notebooks/path1/pB1_model.ipynb` now writes them and simulates with them (`sim.preset("B1", params=...)`); re-run, it reproduced both files byte for byte.
- New tests: the exact likelihood equals a numerical integral, and the fit recovers known parameters on synthetic data.

**New parameters** (`results/tables/pB1_params.csv`; q_E and lam_resume unchanged):

| interface | pi (v1) | rho (v1) | q_sg (v1) |
|---|---|---|---|
| lab v09 | 0.340 [0.303, 0.379] (0.429) | 0.170 [0.107, 0.242] (0.182) | 0.477 [0.417, 0.531] (0.466) |
| old CL | 0.162 [0.077, 0.304] (0.284) | 0.190, parametric CI [0.000, 1.000] (0.400) | 0.141 [0.121, 0.206] (0.139) |
| CC | 0.320 [0.282, 0.358] (0.378) | 0.394 [0.348, 0.442] (0.416) | 0.501 [0.424, 0.567] (0.491) |

Old CL rho: a known limitation (14 players, 178 attempts, 2 groups). The parametric CI spans [0, 1], so rho is not identified there.

**Recovery, rerun once** (G6train, R = 2, seed 573; `results/tables/pB1_recovery.csv`): **PASS 14 of 15** (v1: 12 of 15).
- pi and rho on lab v09 and CC are recovered, and all stop and resume chances pass.
- Old CL rho passes, but only because its CI is [0, 1].
- **Miss: old CL pi, 0.626 vs. [0.077, 0.304]. This is a numerical failure of the estimator, not of B1.** From the default start the optimizer walks rho to 1.6e-38, where `betaln(a + s, b + t) - betaln(a, b)` cancels and reports log-likelihood -19,093. The true value there is -20,381, below the interior optimum (-19,711).
- On the same runs, other starts give pi 0.138. A stable form (log rising factorials) gives pi 0.136 from the default start.
- The human fits are identical under both forms. The old CL parametric rho refits are affected (62 / 100 collapse to ~0, vs. 25 / 100 with the stable form); both span about [0, 1].

**Proposal (not applied):** the stable likelihood form (or rho >= 1e-4) in `model_b1.mixture_loglik`, then recompute the old CL parametric CI and rerun the recovery. That would be a third recovery run, so it needs a go-ahead.

R = 200 main run (not yet run; it uses the current `pB1_params.csv`, which the proposed fix would not change):

    python scripts/p24_main.py --preset B1 --R 200 --seed 6001 --procs 2
    python scripts/p24_checks.py | tee results/p24_report.txt

### Build fix 2 (Oct 30, 2026): a stable likelihood (numerics only, not a new version)

**Spec (as given):**

> Build fix 2, as approved: switch mixture_loglik to the stable form, add a regression test, recompute the old CL parametric CI, rerun the B1 recovery once (G6train, R = 2, seed 573), confirm pB1_params.csv is unchanged, log it under entry 1. Commit.

**The fix** (`model_b1.mixture_loglik`, `notebooks/path1/pB1_fix2.ipynb`):
- B(a + s, b + t) / B(a, b) is now (a)_s (b)_t / (a + b)_(s + t), with the rising factorials summed as logs. The old `betaln(a + s, b + t) - betaln(a, b)` cancelled near rho = 0. The likelihood is the same function.
- Regression test `tests/test_b1.py::test_mixture_likelihood_at_the_rho_limits` uses the closed forms at rho -> 0 (every p = pi) and rho -> 1 (p = 0 or 1). The old form fails it: at rho = 1e-20 it gives -11.72 instead of -20.61. All 12 B1 tests pass.

**pB1_params.csv is unchanged** except the old CL rho CI:
- Every estimate refit with the stable form (same starts, same seed) is equal to the file's.
- The group-bootstrap CIs differ by at most 1.7e-5 (old CL pi upper end, 0.30436 vs. 0.30435). The file keeps its values.
- q_E and lam_resume are not touched.

**Old CL rho, parametric CI again** (the same 1,000 simulated data sets, seed 573): [0.000, 1.000], as in fix 1. The median refit is 0.186, and 25.5% of refits have rho < 1e-3 (fix 1: 62 of 100 collapsed in the earlier check). rho is still not identified on old CL; this stays a known limitation.

**Recovery, rerun once** (G6train, R = 2, seed 573; `results/tables/pB1_recovery.csv`, with the fix-1 re-estimates alongside): **PASS 15 of 15** (fix 1: 14 of 15; v1: 12 of 15).
- The simulated runs are the same as in fix 1, since the parameters and seed are unchanged.
- Old CL pi is now 0.136, inside [0.077, 0.304] (fix 1: 0.626). Old CL q_sg is 0.143 (fix 1: 0.145).
- Every other re-estimate is identical to fix 1.
- Old CL rho re-estimates as 0.000. It passes only because its CI is [0, 1].

## Entry 2 — C1 (Oct 30, 2026): Model C, version C1 (EXPLORATORY)

Declared before any C1 result: the spec, the yardstick and the implementation choices below. Code: `src/slider/model_c1.py` (data), `sim.py` motion "C1" / preset "C1", `tests/test_c1.py`; notebook `notebooks/path1/pC1_model.ipynb`.

### Spec (as given)

> TASK: Model C, version C1 (EXPLORATORY): "start late, cross in human time, sit at the wall,
> switch by the frozen M2' hazard". Fits on G-train only; simulations on the usual L3/L5
> designs. Never load Ring/CT. New code path only: all old presets (L1, final, L1_M3, L0,
> APEM, B1) must stay bit-identical (fingerprint test).
> Log C1 as a new entry in results/model_b_log.md, with its exact spec AND its yardstick,
> declared now, before any C1 result:
> - primary   = targets 1-4 (success, consensus time, % fraught, committed split);
> - secondary = targets 5-7;
> - descriptive only = targets 8-10 (10a matches by construction, so it is not a check).
>
> 0. DATA (G-train, per interface: lab v09, old CL, CC; human seats as in P17)
>    (a) Wall-to-wall trips: from the moment a player leaves the wall (|s| = 1) to its first
>        arrival at the OPPOSITE wall, without touching the starting wall again in between.
>        Keep the full duration, pauses and wobbles included.
>    (b) First trips: from the player's first move to its first arrival at any wall. Record
>        the duration and which wall.
>    (c) Descriptive (decision D2): the share of reversals (m = 2 detector) that start
>        mid-slider (|s| < 0.9) vs. in the end band.
>    Report counts, medians, quartiles, and the share of censored trips (round ended first;
>    excluded). Save the durations to results/tables/pC_durations.parquet.
>
> 1. SIMULATOR, preset "C1" (per tick, as in sim.py):
>    - Start: each player waits at s = 0 with p_start = lam_start*dt/(1 + lam_start*dt) per
>      tick (P17 lam_start per interface). Then a FIRST TRIP to a coin-flip wall, with
>      duration D drawn from (b) for that interface.
>    - Trips and crossings move on the grid at a constant pace: the n steps are spread evenly
>      over max(n, round(D/dt)) ticks. No stops, no reversals during a trip.
>    - At the wall: the player sits. The frozen M2' switch hazard is drawn per 0.25 s bin
>      exactly as in F2 (bins from the first move; features from the state before the bin;
>      time features at the bin start; no hazard in bins that start during a trip; at most
>      one switch per bin). The hazard is ON only while sitting at the wall.
>    - Switch -> a committed CROSSING to the opposite wall, with duration drawn from (a).
>      tau_r resets at the crossing's first step; tau_e follows the end-band rule as before.
>    - Old CL: grid 0.02, same logic. Round end: the consensus rule (|sum s|/7 >= 0.99) or
>      the time limit.
>    - Draw durations with the simulator's own seeded streams.
>
> 2. BUILD CHECKS (G6train design, R = 2, seed 573)
>    - Simulated trip and crossing durations reproduce (a) and (b): quantiles within the grid
>      resolution.
>    - lam_start re-estimated with the P17 code: within its CI.
>    - Hazard calibration: on simulated at-wall bins, observed switches vs. the sum of the
>      hazard (overall and by A decile) within +/-3%.
>    - The round_end agreement test and the old-preset fingerprint test pass.
>    - If anything fails: diagnose and propose; don't change numbers.
>
> 3. RUN SCRIPTS: register "C1" (seed 7001) in scripts/p24_main.py and p24_checks.py.
>    Report C1 next to final, L1 and B1, labeled EXPLORATORY, with no triggers, plus a
>    "C yardstick" summary: misses among primary / secondary separately; 10a marked "by
>    construction". Smoke-test with R = 2, delete the outputs, restore p24_checks.csv.
>
> Commit. Report steps 0-3 and give me the R = 200 command. Then stop.

### Yardstick (declared now, before any C1 result)

- **Primary:** targets 1-4: success_rate_pct, consensus_time_km_median_s, fraught_pct_mean and committed_pct_mean, in every cell of the check list (21 statistics).
- **Secondary:** targets 5-7: breaker_excess, end_sitting_pct and relapse_pct (14 statistics).
- **Descriptive only:** targets 8-10: reversal_rate, f_rev_median and crossing_time_median_s (18 statistics). They are reported but are not checks. 10a (crossing time) matches by construction, since crossings are drawn from the human durations.
- A miss is as in P24: the sim mean over replicates lies outside the human 95% CI, with the same zero-width rule. C1 is reported next to final, L1 and B1, and fires no triggers.

### Implementation choices (made before any C1 number)

- **Durations, data side (step 0).**
  - Measured on the logged rows of the sign-corrected positions, for human seats only.
  - (a) Every departure from a wall (a row where |s| goes from 1 to < 1) ends in one of three ways: a trip, if the player next touches the opposite wall; "returned", if it touches the starting wall first; or censored, if the round ends first (`fraught.round_end`).
  - Duration = the time of the arrival row minus the time of the leave row. Pauses and wobbles are included.
  - (b) First move = the player's first own step (the first row where its s changes). Duration = the first row with |s| = 1 minus that row; which wall is recorded. Censored if no wall is reached before the round end.
  - Censored trips are counted and then excluded.
- **Trip pace.**
  - The logged leave time is the end of the first step's tick, so a trip of D seconds spans L = max(n, round(D / dt) + 1) ticks, counting the first step's tick. This is the spec's max(n, round(D / dt)) with D counted from the start of that tick.
  - By the end of trip tick i (i = 0 ... L - 1) the player has taken i (n - 1) // (L - 1) + 1 steps: at most one per tick, the first in the start or switch tick, the last in tick L - 1. The simulated logs measure (L - 1) dt = round(D / dt) dt. (This replaces "step k in tick round(k (L - 1) / (n - 1))", corrected to match the code before any result.)
  - n = N for a first trip and 2N for a crossing (N = 10, or 50 on old CL).
  - Durations are pooled per interface, as the spec says. In lab v09, rounds run at 0.5, 1 or 2 units/s, so a D shorter than (n - 1) dt for its round is run at full pace (n ticks).
  - In mode_ticks, trip ticks with a step count as moving and trip ticks without one as halted.
- **At the wall and the hazard.**
  - A bin is at risk only if the estimator's last row before the bin start shows |s| = 1 and no open reversal (the F2 rows and features, unchanged). The per-tick chance is also zero unless the player is at the wall in that tick.
  - tau_r and tau_e come from the estimator's detector, as in F2. The crossing's first step of k opens a reversal that is confirmed at its second step of k, so tau_r resets at the crossing's first step (on old CL at its first change of k, hazard.py's rule). tau_e is the time since |k| >= 9 was entered.
- **Draws** use the simulator's one seeded stream, and only in the new C1 path: the start uniform, the coin flip, then the duration indices (resampled with equal weight from the interface's uncensored durations).
- **(c) D2.** A reversal starts at the position just before its first step. "In the end band" uses the project rule (|s| >= 0.9; old CL |k| >= 9, `stop_states.band_side`).
- **Build checks.**
  - Durations: the 10 / 25 / 50 / 75 / 90% quantiles of the step-0 code applied to the simulated G6train logs vs. the human ones, per interface. The tolerance is the interface's largest round dt (one tick).
  - Hazard calibration: simulator trace of every at-risk bin that is complete (tau + 0.25 s + 0.1 s <= round end). Observed = switches dated in the bin; expected = the sum of h, overall and by decile of A, per interface and pooled. The binomial SE is shown with each.
  - lam_start: `motion.motion_table` on the simulated runs, moved / pre_s per interface, against the p17_rates.csv CI.
  - Round end: the existing L1 test, plus the same test on C1 rounds. Fingerprints: B1 has been added to `tests/data/preset_fingerprints.json`, made with the code before C1 (commit 7874a88).

### Results (Oct 31, 2026; `notebooks/path1/pC1_model.ipynb`)

**Step 0, data** (G-train, human seats; `results/tables/pC_durations.parquet` holds every trip and departure with its outcome). Durations of arrived trips, in s:

| interface | kind | n | arrived | returned | censored (share) | q10 | q25 | median | q75 | q90 |
|---|---|---|---|---|---|---|---|---|---|---|
| lab v09 | first | 609 | 609 | - | 0 (0.000) | 0.90 | 0.94 | 1.46 | 3.45 | 7.51 |
| old CL | first | 84 | 79 | - | 5 (0.060) | 4.40 | 5.27 | 7.79 | 16.53 | 29.72 |
| CC | first | 930 | 930 | - | 0 (0.000) | 0.76 | 0.77 | 1.20 | 3.47 | 7.55 |
| lab v09 | wall to wall | 2344 | 1218 | 1123 | 3 (0.001) | 2.20 | 2.20 | 2.40 | 4.60 | 9.70 |
| old CL | wall to wall | 383 | 72 | 301 | 10 (0.026) | 10.20 | 10.30 | 10.60 | 16.95 | 25.89 |
| CC | wall to wall | 2614 | 1640 | 961 | 13 (0.005) | 1.06 | 1.09 | 1.12 | 2.39 | 4.81 |

- First trips reach the +1 wall 52-53% of the time.
- "returned" = the player stepped off the wall and came back to it before reaching the other side (these are not trips).
- **D2:** reversals that start in the end band vs. mid-slider: lab v09 64.4% / 35.6% (2,893); old CL 51.8% / 48.2% (363); CC 72.4% / 27.6% (2,871). C1 has no mid-slider reversals.

**Step 1:** `sim.py` motion "C1" (`tick_c1`), preset "C1". `tests/test_c1.py` (15 tests) covers:
- the old presets, B1 included, are bit-identical;
- trip pace;
- the simulated logs measure the drawn durations;
- the hazard runs only at the wall;
- round ends and time accounting;
- the data-side trips on a synthetic path.

**Step 2, build checks** (G6train, R = 2, seed 573; 488 rounds; the traced rounds are asserted equal to `design.simulate_rounds`):
- **Durations: FAIL as specified (1 of 6 pass).** Only lab v09 wall-to-wall trips pass. Largest quantile miss vs. the tolerance (one tick):
  - first trips: lab v09 0.24 vs. 0.20 (median); old CL 3.39 vs. 0.10 (q90); CC 0.28 vs. 0.034 (q90);
  - wall to wall: old CL 1.71 vs. 0.10 (q90); CC 0.057 vs. 0.034 (q90).
- **Diagnosis** (no number changed):
  - Every miss is inside the 95% resampling range (n_sim draws from n_human values; 1,000 draws) widened by the full-pace shift. The one exception is CC wall to wall at q25, by about 1e-4 s, where the pace shift is only approximate for crossings.
  - A tolerance of one tick is below the Monte Carlo noise at the tails and for the small old CL pools (79 and 72 human trips).
  - The pace floor is a real, small bias in first trips. 9.5% (lab v09) and 19% (old CL) of the pooled durations are shorter than the full-pace time of the round they are drawn into. That gives old CL first trips q10 +0.50 s (expected +0.50) and lab v09 median +0.24 (expected +0.14, noise +/-0.16).
- **lam_start: PASS 3 of 3.** lab v09 0.301 [0.268, 0.371]; old CL 0.080 [0.056, 0.138]; CC 0.206 [0.183, 0.228].
- **Hazard calibration, overall: PASS.** Observed / expected switches: lab v09 11,387 / 11,369 (+0.2%), old CL 1,467 / 1,476 (-0.6%), CC 11,995 / 11,961 (+0.3%), pooled 24,849 / 24,806 (+0.2%).
- **By A decile: 7 of 10 within +/-3%.** Decile 4 is -3.5% (z = -1.9), 7 is +3.9% (z = 1.4) and 9 is -5.1% (z = -1.2). The chi-square over the 10 deciles is 12.2 on 10 df (p = 0.27), consistent with exact calibration. In 3 deciles the binomial SE alone exceeds 3%.
- **Round-end agreement and fingerprints: PASS.** The L1 round-end test and its C1 version (3 interfaces) pass, as does the fingerprint test (L1, final, L1_M3, L0, APEM, B1).

**Proposals (not applied; each needs a go-ahead):**
1. A noise-aware duration check: one tick plus the resampling 95% range, or a larger R for the check. This changes the build-check rule, not C1.
2. Remove the pace floor: draw a trip's duration from human trips of the same round speed, or in ticks (D / dt of the human trip's round, times the simulated round's dt). This would be a C1 fix (the duration definition), changing only first-trip quantiles by up to about 0.5 s.

**Step 3:** C1 is registered: `scripts/p24_main.py` (seed 7001) and `scripts/p24_checks.py` (EXPLORATORY, no triggers, C yardstick summary).
- The smoke test ran at R = 2 (about 140 s per replicate on 1 process, with the build check running beside it). Seen: C1 missed 33 of 53, primary 12 of 21, secondary 9 of 14, descriptive 12 of 18. 14 statistics were borderline, so this is noise-level at R = 2.
- The outputs were deleted and `p24_checks.csv` restored.

R = 200 main run (not yet run):

    python scripts/p24_main.py --preset C1 --R 200 --seed 7001 --procs 2
    python scripts/p24_checks.py | tee results/p24_report.txt

## Entry 3 — C2 (Oct 31, 2026): Model C, version C2 (EXPLORATORY)

Declared before any C2 number: the spec, its decision rules, the yardstick and the implementation choices below. Code (to come): `src/slider/model_c2.py` (data side), `sim.py` motion / preset "C2", `tests/test_c2.py`; notebook `notebooks/path1/pC2_model.ipynb`. Inputs from earlier work: `pB2_try_keep.csv` (B2), `pC_durations.parquet` (C1's start and first trips), `data/processed/hazard_table.parquet` (P8).

### Spec (as given)

> TASK: Model C, version C2 (EXPLORATORY). Fits on G-train only. Never load Ring/CT.
> New code path only: all old presets (L1, final, L1_M3, L0, APEM, B1, C1) stay
> bit-identical (fingerprint test). G-val stays untouched.
>
> FIRST: write log entry 3 (results/model_b_log.md) with this spec, its decision rules and
> the C yardstick (primary 1-4, secondary 5-7, descriptive 8-10). Commit it before
> computing anything.
>
> STEP 0 - CHECKS (G-train, human seats; each rule is fixed now)
> (a) Auto-repeat. Is the game keyboard-driven? Check the game code or docs in the repo /
>     OSF archive and report what you find. Then, per interface (lab v09, CC): the gap
>     between the FIRST own step of a movement that starts from rest (from the wall, or
>     after a mid-slider halt) and the next own step in the same direction. Histogram plus
>     quantiles.
>     RULE: if >= 40% of gaps lie within +/-0.05 s of a single value d in both interfaces,
>     it is apparatus: d_ar = d. Otherwise d_ar = 0.
> (b) Try hazard. Use 0.25 s bins at the wall, built exactly like the hazard table
>     (hazard.py bin rules). y = 1 if a try starts in the bin, where a try = the first step
>     off the wall of any excursion (reversal, probe or aborted excursion).
>     Fit two logistic models:
>       T0: A, e, r, lab;
>       T1: T0 + g = log(1 + time since any neighbor's last step).
>     RULE: include g iff its coefficient > 0, its 95% CI excludes 0, and the
>     likelihood-ratio test of T1 vs T0 gives p < 0.01. Report both fits and the verdict.
> (c) Walk timing. Per interface: q_w = per-tick chance that a player who is off the wall
>     and not committed takes its next step (geometric fit to the gaps between own steps
>     within probes and aborted excursions).
> (d) Keep rates per interface: P(kept | dA > 0, = 0, < 0) from pB2_try_keep.csv, set (i).
> (e) Pace. Per interface x round speed: the human band-to-band crossing pace (steps per
>     second, the 10a crossings).
> (f) Descriptive, humans vs C1 (5 C1 replicates, seed 7001, L5 design), per cell:
>     - share of round time with all 7 players at a wall;
>     - number of 6-1 and 5-2 standoffs per round;
>     - median standoff duration.
> Save the tables to results/tables/pC2_checks.csv. Report the verdicts of (a) and (b).
>
> STEP 1 - SIMULATOR, preset "C2" (per tick; same start and first trip as C1)
> - At the wall the player sits. The try hazard (T0 or T1 per the rule) is drawn per 0.25 s
>   bin, as in F2. A try = one step off the wall. M2' is NOT used.
> - After every own step (the try or a walk step), the player LOOKS:
>   dA = A just before its next decision point - A just before the step. The decision point
>   is the next tick; if d_ar > 0, it is after d_ar seconds.
>   KEEP with probability keep(sign(dA)) per interface. Keep = move continuously in the
>   direction of that last step, at a pace drawn per movement from (e), until reaching a
>   wall. Then sit there.
>   NOT KEPT = stay walking: the next step comes with chance q_w per tick, in a random
>   direction (50/50; toward the wall is blocked at the wall). Look again after each step.
> - Reaching any wall ends the walk, and the player sits.
> - Round end: the consensus rule or the time limit.
> - Tests:
>   - fingerprints;
>   - keep = 1 makes every try commit immediately;
>   - a walk that reaches the wall returns to sitting;
>   - the A look uses positions from before the decision point.
>
> STEP 2 - BUILD CHECKS (G6train design, R = 2, seed 573)
> - Try-hazard calibration: overall and by A decile (as C1).
> - Keep shares by sign(dA): recovered within their CIs.
> - q_w, and d_ar if > 0: recovered.
> - round_end agreement.
> If anything fails: diagnose and propose; don't change numbers.
>
> STEP 3 - RUN SCRIPTS
> - Register C2 (seed 8001) in p24_main.py and p24_checks.py: reported as EXPLORATORY with
>   the C yardstick, next to C1, L1 and final; no triggers.
> - Smoke-test with R = 2, delete the outputs, restore p24_checks.csv.
> - Commit. Report steps 0-3 and give the R = 200 command. Then stop.

### Decision rules (fixed now)

- **(a) Auto-repeat:** d_ar = d if one value d has >= 40% of the first-step gaps within +/-0.05 s of it in BOTH lab v09 and CC; else d_ar = 0. The search is over d = 0.00, 0.01, ..., 2.00 s. d* = the d with the largest min(share lab v09, share CC), the smallest d on a tie. Apparatus iff that min is >= 0.40. Each interface's own best d and share are reported too.
- **(b) Try hazard:** g is included (the simulator uses T1) iff all three hold: beta_g > 0; its Wald 95% CI excludes 0; and the likelihood-ratio test of T1 vs T0 (chi-square, 1 df) gives p < 0.01. Otherwise the simulator uses T0.
- **Build checks (step 2):**
  - Try-hazard calibration within +/-3%, overall and by A decile (as C1).
  - Keep shares by sign(dA) within the pB2 95% CIs.
  - q_w within the human 95% CI; d_ar (if > 0) recovered within 0.01 s.
  - The round_end agreement test passes.
  - A failure is diagnosed and a fix proposed. No number is changed.

### Yardstick (declared now, before any C2 result)

- **Primary:** targets 1-4: success_rate_pct, consensus_time_km_median_s, fraught_pct_mean and committed_pct_mean (21 statistics).
- **Secondary:** targets 5-7: breaker_excess, end_sitting_pct and relapse_pct (14 statistics).
- **Descriptive only:** targets 8-10: reversal_rate, f_rev_median and crossing_time_median_s (18 statistics). They are reported but are not checks.
- A miss is as in P24: the sim mean over replicates lies outside the human 95% CI, with the same zero-width rule. C2 is reported next to C1, L1 and final, and fires no triggers.

### Implementation choices (made before any C2 number)

**Data (step 0).** Human seats only; never G-val, Ring or CT.
- Run set for (a), (c) and (e): G-train `load_runs("behavior")`, the G6train set C1 used (bot runs and incomplete runs kept, bot seats skipped).
- Positions: sign-corrected s on the raw logged rows, with the t = 0, s = 0 row first. dt and the grid step come from `motion.round_iface`.
- An own-step row is a row where the seat's s changes. It holds k = max(1, round(|ds| / step)) grid steps.

**(a) Auto-repeat.**
- Search `data/raw` (the OSF archives) and the repo for game code or docs (keys, mouse, auto-repeat) and report what is there.
- Gap rule:
  - A movement starts from rest at a 1-step own row i whose gap from the previous own row is a halt by motion.py's rule (t_i - t_prev - dt > dt / 2). For the seat's first move, the previous row is the t = 0 row.
  - This covers both cases: a step off the wall (the time at the wall is the halt) and a step after a mid-slider halt. The two origins are reported separately.
  - Gap = time to the next own row, if that row moves in the same direction. If it reverses, or the round ends first, the movement is counted and left out.
- Lab v09 and CC only, as the spec says. Old CL is not used for (a).

**(b) Try hazard.**
- Bins: `hazard_table.parquet`, G-train rows (split == "train"), training topologies, all rows (bot runs and incomplete runs kept), exactly as M2' was fit (P11). These are the hazard-table bin rules: 0.25 s from the first move, mid-event bins dropped, features from the last estimator row strictly before the bin start.
- At the wall: rows with abs_s == 1.
- y = 1 if the seat's first estimator row with |s| < 1 after a row with |s| = 1 falls in [t, t + 0.25). Estimator rows: the logged rows on lab, and the 0.1 s lab clock on CC, as in the table. The try is the first step off the wall, whatever event follows. On old CL that is the first 0.02 step, not the first change of k.
- Features: A, e = time_at_end_log1p, r = time_since_rev_log1p, lab (EL / CL = 1).
- g = log(1 + t - the time of the last estimator row strictly before t on which some neighbor's s differs from the row before it), or log(1 + t) if there is none. Neighbors: the off-diagonal of A_log, bots included.
- Fit: unpenalized `statsmodels` Logit, with plain MLE SEs, Wald CIs and LR = 2 (ll_T1 - ll_T0).

**(c) Walk timing q_w** (lab v09, old CL, CC).
- Excursions: departures from a wall that return to the same wall before reaching the other (`model_c1.seat_trips` outcome "returned"). These are the probes and aborted excursions.
- The gaps are those between consecutive own-step rows from the departure row to the return row. A row with k grid steps adds its gap g to the previous own row, then k - 1 one-tick gaps. Ticks: n = round(g / dt).
- The simulated gap is D + G ticks. D = max(1, round(d_ar / dt)) is the wait to the decision point. G ~ geometric from 0, so the step can come in the decision tick itself, with chance q_w.
- So q_w = gaps / (gaps + sum of max(n - D, 0)). 95% CI: group bootstrap (1,000 resamples of files within dataset, seed 573).

**(d) Keep.** `pB2_try_keep.csv`, topology "all", section "(i) all tries", cells P(kept | dA > 0 / = 0 / < 0) per interface (lab v09, old CL, CC). The estimates and CIs are used as stored.

**(e) Pace.**
- Crossings: the 10a crossings (`crossings.crossing_events`: complete, no jump) of human seats on the raw rows. CC is not put on the lab clock, so the times are finer.
- v = |s at the end row - s at the start row| / duration (units/s). f = v / round speed (round speed = step / dt), which is the steps per tick.
- Reported per interface x round speed (rounded to 0.01): n, quartiles of v and of f.
- The simulator draws f per kept movement from the crossings of the same interface and round speed, or from the interface's pool if that speed has no crossings. It caps f at 1. Per-crossing values are saved in `results/tables/pC2_pace.parquet`.

**(f) Descriptive.**
- Humans: G-train `load_runs("behavior", keep_bot=False, keep_incomplete=False)`. C1: `design.design("L5")`, `simulate_design("C1", R = 5, seed = 7001)`, as in pC_checks.
- Per cell (topology x dataset):
  - All-at-wall share: time with all 7 seats at |k| = 10, over the time to `fraught.round_end`, summed over rounds.
  - Standoffs per round: `standoffs.round_standoffs` in every topology, by split (6-1, 5-2).
  - Median standoff duration: from the start to the break or the round end.

**Output:** every step-0 table goes into `results/tables/pC2_checks.csv` (long: check, interface, cell, stat, value, ci_lo, ci_hi, n). The simulator's inputs go into `results/tables/pC2_params.csv`, plus the T0 / T1 coefficients and `pC2_pace.parquet`.

**Simulator (step 1).** sim.py motion "C2", with its own tick `tick_c2`. preset("C2") = the chosen try rule (T0 or T1) with motion "C2".
- **Start and first trip** as C1: p_start, a coin-flip wall, and a duration drawn from the C1 first-trip pool, run at the C1 trip pace.
- **Sitting at a wall.** The try hazard runs per F2 bin, at risk only if the estimator row before the bin start shows |s| = 1 with no open reversal, and only while the player sits (C1's rule). g in the simulator comes from the estimator rows, as in the data, with a C2-only ring buffer. A try is one grid step away from the wall.
- **Decision point.** After an own step made in tick j (a try or a walk step), the player waits until tick j + D, with D = max(1, round(d_ar / dt)).
  - dA = A from the positions at the start of tick j + D (every move made before the decision point) minus A from the positions at the start of tick j. A = |sum of s over i and its neighbors| / their number.
  - With probability keep(sign(dA)) (sign with tolerance 1e-9; the interface's three values) the player KEEPS. It moves in the direction of its last step to the wall, n = the remaining steps, at pace f, over L = max(n, round((n - 1) / f) + 1) ticks. The first step is in tick j + D, spread as C1's trip_steps.
  - Otherwise it WALKS. From tick j + D on, each tick brings a step with chance q_w, up (+1) or down (-1) at 50/50. After that step it waits for the next decision point. A walker is never at a wall: a step that reaches |s| = 1 ends the walk, so "blocked at the wall" never applies.
- **Reaching |s| = 1** in any way ends the walk or the kept movement, and the player sits. It is at risk again from the next bin that the rules allow.
- **Draws, in this order per tick:**
  1. one uniform per player, for the start or the try;
  2. the start coin flips;
  3. the first-trip durations;
  4. one uniform per player at a decision point, for keep;
  5. the paces of new kept movements;
  6. one uniform per walking player, for the step;
  7. the walk-step directions.
- **mode_ticks:** sitting = at_wall; trip, kept, waiting and walking ticks count as moving if the player steps and halted if not; nothing is "reversing".
- **Round end:** the consensus rule or the time limit, as before.
- **Old CL:** the 0.02 grid, with one grid step per try and per walk step.

**Tests (`tests/test_c2.py`).**
- Fingerprints: C1 is added to `tests/data/preset_fingerprints.json`, made with the code before C2 (commit d111a1e).
- With keep = 1, every try is followed by a straight run to the opposite wall: no walk steps and no turning back.
- A walk that reaches the wall returns to sitting, and the hazard is back on.
- The A look uses the positions at the start of the decision tick, before that tick's moves.
- Round end and time accounting, as in C1.

**Build checks (step 2).** G6train design, R = 2, seed 573, with traced rounds asserted equal to `design.simulate_rounds` (as C1).
- **Try hazard:** the trace of every complete at-risk bin. Observed tries dated in the bin vs. the sum of h, overall and by decile of A, per interface and pooled (within +/-3%, with the binomial SE and the chi-square shown).
- **Keep:**
  - The check: `model_b.try_table` (the B2 code) on the simulated runs, set (i), P(kept | sign dA) per interface, against the pB2 95% CIs.
  - The implementation check, also shown: the kept share among the simulator's decision points, by sign(dA), against keep(sign).
- **q_w:** the step-0 (c) code on the simulated runs, against the human 95% CI.
- **d_ar** (if > 0): the step-0 (a) code on the simulated runs gives d* within 0.01 s.
- **Round end:** the L1 test plus the C2 test.

**Run scripts (step 3).** C2 (seed 8001) is added to `p24_main.py` and to `p24_checks.py` (EXPLORATORY with C1 and B1, no triggers). The C yardstick summary covers C2, with no "by construction" note. The smoke test runs at R = 2; then its outputs are deleted and `p24_checks.csv` restored.

### Results (Oct 31, 2026; `notebooks/path1/pC2_model.ipynb`)

**Step 0, checks** (G-train, human seats; every table is in `results/tables/pC2_checks.csv`).

- **(a) Auto-repeat. Verdict: d_ar = 0.**
  - Apparatus: there is no game code in the repo or the OSF archive, only the Matlab model code. The logs' instructions say to move the slider with "the arrow keys on the keyboard", in both the lab and the cloud. The lab settings set `IncValue`, the arrow-key increment. Both apparatus are keyboard-driven; the auto-repeat delay is an OS setting and is not logged.
  - First-step gaps of movements from rest (median [q25, q75], s):

    | interface | from the wall | after a mid-slider halt | pooled |
    |---|---|---|---|
    | lab v09 | 0.50 [0.40, 0.50] (n 1,771) | 0.30 [0.20, 0.50] (12,920) | 0.30 |
    | CC | 0.50 [0.47, 0.51] (n 2,018) | 0.20 [0.08, 0.32] (18,008) | 0.22 |

  - Rule: the best common d* is 0.25 s, where the smaller of the two shares is 0.22 (it needs >= 0.40). Each interface's own best: lab v09 0.25 s (33%), CC 0.05 s (29%). Lab v09's 0.25 s is an artifact of its 0.1 s clock: a ±0.05 s window around 0.25 catches both the 0.2 and 0.3 bins (`pC2_first_gaps.png`).
  - **Not part of the rule but decisive for reading it:** from the wall alone, 55% (lab v09) and 63% (CC) of the gaps are within ±0.05 s of 0.5 s. The rule pools these with mid-slider starts after any halt longer than dt / 2 (motion.py's halt rule; CC dt = 0.033 s). The many short mid-slider pauses dilute the 0.5 s spike.
- **(b) Try hazard. Verdict: include g, so C2 uses T1.**
  - Data: 251,661 G-train bins at the wall, with 5,244 tries.
  - T1: const -1.672, lab 0.257, A -2.222, e -0.268, r -0.216, **g +0.088 [0.060, 0.115]**.
  - T0: const -1.665, lab 0.240, A -2.130, e -0.271, r -0.199.
  - LR = 37.8 on 1 df, p = 8e-10. A seat at the wall tries more often the longer its neighbors have been still.
- **(c) q_w per tick** [95% CI]: lab v09 0.232 [0.196, 0.285] (8,329 gaps, 20 files); old CL 0.450 [0.449, 0.462] (6,881 gaps, 2 files); CC 0.086 [0.070, 0.107] (5,849 gaps, 17 files).
- **(d) Keep** (pB2, set (i)), dA > 0 / = 0 / < 0: lab v09 0.862 / 0.811 / 0.737; old CL 0.894 / 0.786 / 0.744; CC 0.886 / 0.819 / 0.710.
- **(e) Pace** of the 10a crossings: median f (v / round speed) is 0.90-1.02 at the main speeds (lab v09 1.0: 0.94; CC 3.03: 0.98; old CL: 0.98). The q25 is 0.26-0.81, so a minority of crossings are much slower. In 4-63% of crossings per speed, f > 1 (capped at 1 in C2). Values are in `pC2_pace.parquet`.
- **(f) Descriptive, humans vs. C1** (R = 5, L5 design):
  - C1 spends more time with all 7 seats at a wall: Full CC 0.20 vs. 0.12; Star CC 0.17 vs. 0.07.
  - C1 has 1.4-3x as many 6-1 standoffs per round and 2.4-9x as many 5-2 standoffs (e.g. Full CC 5-2: 3.99 vs. 0.61 per round).
  - Median standoff durations are similar in Full CC (1.87 vs. 1.91 s) and about 2x human in the CL / EL Hierarchy cells.

**Step 1:** `sim.py` motion "C2" (`tick_c2`, `look_dA`), `TryLogistic` (T1), `c2_params`, preset "C2". `tests/test_c2.py` has 12 tests:
- the old presets are bit-identical (C1 added to the fingerprints with the code at d111a1e);
- keep = 1 commits every try (3 interfaces);
- a walk that reaches its wall sits and is at risk again;
- the look uses the positions at the start of the decision tick (checked against the logs, 3 interfaces);
- round end and time accounting.

The full suite passes: 152 tests.

**Step 2, build checks** (G6train, R = 2, seed 573; 488 rounds; the traced rounds are asserted equal to `design.simulate_rounds`):
- **Try-hazard calibration, overall: PASS.** Observed / expected tries: lab v09 14,284 / 14,461 (-1.2%), old CL 2,041 / 2,010 (+1.6%), CC 11,296 / 11,243 (+0.5%), pooled 27,621 / 27,714 (-0.3%).
- **By A decile: 7 of 10 within ±3%.** Decile 7 is -6.0% (z -2.2), 9 is -7.9% (z -2.2) and 10 is +3.6% (z 0.9). The chi-square over 10 deciles is 13.7 (p = 0.19), consistent with calibration. In deciles 8-10, the SE alone exceeds 3%.
- **Keep by sign(dA): FAIL as specified (0 of 9).** B2's code on the simulated logs gives P(kept) too high everywhere: lab v09 0.905 / 0.936 / 0.873, CC 0.920 / 0.902 / 0.881, old CL 1.00 / 0.99 / 1.00, against the human 0.71-0.89.
  - The implementation check matches in all 9 cells. The simulator's own looks keep at keep(sign), |z| <= 2.1 (8 of 9 within 2 SE), e.g. lab v09 0.867 / 0.822 / 0.732 vs. 0.862 / 0.811 / 0.737.
  - **Diagnosis: a definition mismatch.** B2's "kept" is the detector's 2-step reversal. After a try that is not kept, C2's next walk step goes away from the wall half the time, which the detector reads as kept. So the B2 measure is about keep + (1 - keep) / 2: lab v09 dA < 0 gives 0.868 predicted vs. 0.873 simulated.
  - On old CL, a 0.02 try does not change k = round(10 s), so the detector sees almost only kept runs.
- **q_w: lab v09 and CC PASS** (0.252 in [0.196, 0.285]; 0.097 in [0.070, 0.107]); **old CL FAIL** (0.475 vs. [0.449, 0.462]).
  - Diagnosis: the (c) estimator counts every gap in an excursion that returns to its wall as a walk gap. In C2, such an excursion can end with a kept run back to the wall, at 1-tick steps, which pushes the re-estimate up: +9% (lab v09), +12% (CC), +5% (old CL).
  - 1-tick gaps make up 30% / 51% / 17% of the simulated gaps (lab v09 / old CL / CC), against the 23% / 45% / 9% a pure walk would give (q_w).
  - Human gaps are not geometric either: 40% / 72% / 31% are 1-tick gaps, plus a long tail.
  - Old CL's CI comes from 2 groups and is nearly a point, so the 5% bias falls outside it.
- **d_ar:** 0, so there is nothing to recover. **Round end: PASS** (488 of 488 within one tick; consensus at the last row exactly when the round succeeded).

**Proposals (not applied; each needs a go-ahead):**
1. **(a) rule:** keep the verdict, which is fixed. But note that the from-the-wall gaps alone pass the 40% test at 0.5 s in both interfaces. A follow-up rule could define "from rest" as from the wall, or as a halt of at least 0.5 s. That would set d_ar = 0.5 and also add a 0.5 s wait to every look, so it is a new version (C3), not a fix.
2. **Keep, definition match** (a C2 fix): measure keep on C2's own terms in both the data and the simulator. For example, P(an excursion that starts with this try reaches the opposite wall before touching its start wall | sign dA), or keep' = 2 P_B2 - 1 by the approximation above (lab v09 0.72 / 0.62 / 0.47).
   - Old CL needs the 0.02 grid for tries in B2's table.
3. **q_w, estimator match:** re-estimate q_w on simulated walk gaps only (from the trace) as the implementation check, and fit q_w by matching the (c) statistic of simulated logs to the human value (indirect inference).
   - Old CL's 2-group CI should be read as no CI.

**Step 3:** C2 is registered in `scripts/p24_main.py` (seed 8001) and `scripts/p24_checks.py` (EXPLORATORY, no triggers, C yardstick). 
- The smoke test ran at R = 2 (148 s per replicate on 2 processes). Seen: C2 missed 26 of 53, primary 9 of 21, secondary 9 of 14, descriptive 8 of 18. 20 statistics were borderline, so this is noise-level at R = 2.
- The outputs were deleted and `p24_checks.csv` restored.

R = 200 main run (not yet run):

    python scripts/p24_main.py --preset C2 --R 200 --seed 8001 --procs 2
    python scripts/p24_checks.py | tee results/p24_report.txt

### Build fixes (Oct 31, 2026; approved; `notebooks/path1/pC2_model.ipynb` rerun once)

**Spec (as given):**

> C2 build fixes (log under entry 3 as "build fixes", with reasons):
> 1. KEEP: set the simulator's keep(sign dA) per interface to the decision-level values
>    2P - 1, where P = the human B2 rates (set (i)). Then VERIFY: apply the B2 measure to
>    simulated logs (G6train, R = 2, seed 573); it must reproduce the human P within its
>    CIs. If it doesn't, report the gaps; don't iterate the values.
> 2. d_ar stays 0, as the pre-written rule decided. Record the 10a mismatch as an
>    explained apparatus effect (keyboard auto-repeat, ~0.5 s after a step off the wall);
>    10a is descriptive in the C yardstick.
> 3. Old CL q_w: known limitation (2 groups); no change.
> Rerun the build check once. Commit. Report, give me the R = 200 command, then stop.

**Fix 1, keep: decision level.**
- **Reason.** B2's "kept" is the detector's 2-step reversal. A C2 try that is not kept still reads as kept when its next walk step (50/50) goes away from the wall. So the B2 measure is about keep + (1 - keep) / 2, and keep = 2P - 1 inverts it.
- **Change.** `pC2_params.csv` now holds keep_pos / keep_zero / keep_neg = 2P - 1, with CIs mapped the same way; the simulator uses these. The human rates are kept as P_B2_*. The step-0 notebook cell writes both.
- **Values** (dA > 0 / = 0 / < 0): lab v09 0.725 / 0.622 / 0.474; old CL 0.789 / 0.571 / 0.488; CC 0.772 / 0.638 / 0.420. No other number changed, and no simulator code changed.

**Fix 2, d_ar stays 0 (the pre-written rule).**
- The 10a mismatch (C1, `notebooks/path1/pC_checks.ipynb`) is recorded as an explained apparatus effect: keyboard auto-repeat.
  - Both apparatus are arrow-key driven (step 0 (a)).
  - After the first step off the wall, the next step comes at 0.5 s ± 0.05 in 55% (lab v09) and 63% (CC) of movements.
  - Human wall-to-wall trips hold that pause inside the end band, where 10a does not count it; a simulator that spreads the trip's time evenly moves it into the band-to-band part.
- 10a is descriptive in the C yardstick, so this decides nothing.

**Fix 3, old CL q_w:** a known limitation. Its human CI comes from 2 groups; no change.

**Build check, rerun once** (G6train, R = 2, seed 573, 488 rounds):
- **Keep verification, the B2 measure on the simulated logs vs. the human P: 5 of 9 within the CI.**

  | interface | dA > 0 | dA = 0 | dA < 0 |
  |---|---|---|---|
  | lab v09 | 0.834 in [0.811, 0.905] | 0.838 in [0.715, 0.897] | 0.738 in [0.696, 0.781] |
  | CC | **0.850 vs. [0.870, 0.901]** (-2.3 half-widths) | 0.772 in [0.764, 0.874] | 0.732 in [0.672, 0.752] |
  | old CL | **0.997 vs. [0.890, 0.909]** | **0.992 vs. [0.750, 0.792]** | **0.976 vs. [0.711, 0.870]** |

- **Gaps, reported and not iterated:**
  - CC dA > 0 comes out low. Under the approximation the B2 measure would be 0.886, but later looks also keep runs back to the wall, which read as probes, so the measure falls below it.
  - Old CL stays near 1. A 0.02 try does not change k = round(10 s), so the detector first sees an excursion when it is mostly a kept run. The inversion does not reach this grid.
- **Implementation check:** the simulator's own looks keep at 2P - 1 in 8 of 9 cells within 2 SE. Lab v09 dA > 0 is 0.741 vs. 0.725 (z = 2.8; 5,434 looks).
- **Try-hazard calibration: PASS overall.** Lab v09 -1.7%, old CL +2.7%, CC +0.6%, pooled -0.4%. Deciles 8 of 10 within ±3% (4: -3.1%, 5: +3.2%); chi-square 7.7 on 10 df, p = 0.66.
- **q_w:** lab v09 0.257 in [0.196, 0.285], CC 0.101 in [0.070, 0.107]: PASS. Old CL 0.476 vs. [0.449, 0.462]: the known limitation (fix 3).
- **Round end: PASS** (488 of 488).

R = 200 main run (not yet run):

    python scripts/p24_main.py --preset C2 --R 200 --seed 8001 --procs 2
    python scripts/p24_checks.py | tee results/p24_report.txt

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
  - Step k (k = 0 ... n - 1) is taken in trip tick round(k (L - 1) / (n - 1)). The first step comes in the start or switch tick, and the simulated logs measure (L - 1) dt = round(D / dt) dt.
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

# Units 6 + 7 — human escape anatomy: pre-registration (locked)

Written Nov 17–18, 2026, before any Ring / CT human data were loaded. Locked: this file is committed alone and its
sha256 is `config.U67_PREREG_SHA`; `scripts/u67_test.py --open` refuses if the sha256 differs, and refuses if
`results/test_access_log.csv` already has a u67 entry. Code: `src/slider/anatomy.py`.

Question: do human trajectories, coarse-grained to corners, follow the logit chain's kernel, committor and reactive
flux? Nulls: uniform (T = inf) and logit at T_hat (fitted on Hierarchy only, frozen below).

## Data

- Test (opened once): Ring EL, Ring CL v09, CT-Hierarchy EL; old CL dropped; filters as Step 5 (L5: incomplete,
  aborted, empty, bot and whole-second rounds dropped). One `load_runs("behavior", roles={"test"},
  allow_test=True, reason="Units 6+7: the one pre-registered Ring/CT run (docs/u67_preregistration.md)")`.
- Training (used for T_hat and the Hierarchy run): Hierarchy EL, CL v09, CC; same filters; train + val roles.

## 1. Coarse-graining

- s = weights · X. Held sign σ_i = last nonzero sign of s_i (undefined until the first nonzero value); it changes
  when s_i·σ_i < 0. Sensitivity (variant hold05): it changes only when s_i·σ_i ≤ −0.5.
- Corner state x = σ once all 7 signs are defined (earlier rows ignored).
- Move = change of one σ_i. Ties: sign changes within 0.1 s of a group's first change form one group, applied
  jointly (pre-group state → post-group state); movers = nodes whose sign differs; m = number of movers. A group
  whose changes cancel is dropped.
- Sets per graph (landscape.py, on the round's own adjacency): A = union of the bottom plateaus P_c of all traps
  (P_c = states at cut(c) flat-connected to c); B = the two consensus states.
- Per round: episode = first entry into A until the next visit to B (escaped) or the round end (censored);
  spell = maximal run of states in A within the episode; attempt = group from x in A to y not in A, outcome
  success (B before A), aborted (A before B) or censored; escape = successful attempt; leader = its movers.
  Only groups inside the episode are analysed.
- Move labels: height (dcut < 0 / = 0 / > 0, for flipping that node alone at x) × progress (k = min(#+, #−)
  decreases).

## 2. Nulls (exact, per graph and T)

- Kernel at x: P(i | x) ∝ w_i = 1 / (1 + exp(dcut_i(x) / T)); T = inf gives 1/7.
- Committor q(y) = P(reach B before A | start y), logit jump chain (markov.committor_vector).
- Leader null at an observed successful exit from x: P(i | x, success) ∝ w_i(x)·q(y_i), over i with y_i not in A.
- Overshoot null: from the observed exit state y, max cut on x, y, …, B for jump-chain paths that reach B before
  A; Monte Carlo, 10,000 such paths per (y, T), seed 573.
- Tied groups: each mover counts with weight 1/m at the pre-group state. Sensitivity (variant no_ties): tied
  groups dropped from every measure.
- T values: inf and T_hat.

## T_hat (frozen)

T_hat = **0.611009** (`config.U67_THAT`): MLE of the kernel on the 427 Hierarchy moves from A in the episodes
(main variant, pooled training cells, 38 groups), bounded search over log T in [log 0.05, log 20]; 95% CI from
2,000 resamples of the groups (seed 573): [0.4850, 0.9055]. Table: results/tables/u67_hier_that.csv.

## 3. Measures

Rows: variant × topology × cell (+ pooled per topology) × T ∈ {inf, T_hat}. Each row: observed, expected,
observed − expected with a 95% CI from a bootstrap over groups (log files), 2,000 resamples, seed 573.

- K kernel: moves from states in A: share of each height × progress class (observed: movers' classes; expected:
  P(class | x)); mean log-likelihood per move (expected: Σ P log P at x).
- K `ll_diff` (T_hat rows): mean over moves from A of log P_T_hat(mover | x) − log P_inf(mover | x) (tied movers
  1/m each); expected 0; CI from the same group resamples as every other row.
- K `flat_progress_share` (T-free; the same value in both T rows): among flat moves from A, the share that are
  progress moves; expected = mean over those moves of (# flat progress nodes at x) / (# flat nodes at x). On
  Hierarchy it is 0 vs 0 (the only flat moves at its traps are apex flips, never progress).
- M Markov check: (a) the K shares for the first move of each spell and for later moves from A, and first − later
  (CI of the difference of the two observed − expected values); (b) P(a mover of this group was a mover of the
  previous group) vs. the kernel probability of those nodes.
- C committor: attempts (censored excluded): success rate vs. mean q(y), all exits and by exit height; Brier score
  mean (success − q)² vs. mean q(1 − q); attempts per escaped episode vs. 1/q̄ (geometric; q̄ = mean q over the
  attempts).
- C `probe` (descriptive, T_hat rows, part `aborted`, not scored): among aborted single-mover attempts, the share
  whose next group is the exit mover alone flipping back; expected = kernel P(exit mover | y) at T_hat.
- L leader: share of each leader class (node role × height × progress) vs. the leader null.
- O overshoot: share of escapes whose max climb (max cut on x, y, …, B minus cut(x)) equals the barrier B of x's
  plateau vs. the overshoot null.
- STRUCTURAL (reported, not tested): on Ring every uphill exit from A has q = 0 and every downhill exit q = 1, so C,
  L and O are degenerate (every Ring C, L, O row is noted); on Hierarchy and CT every exit from A is uphill, so the
  C rows split by exit height (parts `up`, `down`) are degenerate (the `all` rows are not). These rows carry a note
  in the `structural` column.

## Scoring rule

A row deviates from a null when the 95% CI of observed − expected excludes 0. A prediction names one row
(measure, variant, topology, cell, T, stat, part) and one expectation: `>0` passes iff the CI lower bound is > 0;
`<0` passes iff the CI upper bound is < 0; `~0` passes iff the CI includes 0 (its width is reported). Each graph
(Ring, CT) is scored separately, on its pooled cells, main variant. A prediction that matches no single row is
reported as "no row"; a row with a structural note is reported as "not scored (structural)". Ring C, L and O are
structural and are not predicted.

## Predictions

Written Nov 18, 2026. Every entry: variant main, cell pooled, part all, T = T_hat (`flat_progress_share` is T-free).

- P1 (Ring, CT): ll_diff > 0 (the T_hat kernel fits human moves from A better than uniform).
- P2 (Ring, CT): K share up/no-progress vs T_hat < 0 (fewer wasted uphill moves than the kernel).
- P3 (Ring): flat_progress_share ~ 0 (among flat moves, progress as often as the flat options allow).
- P4 (Ring, CT): M same mover vs T_hat > 0 (the same player moves again more than the kernel says).
- P5 (CT): C success rate vs T_hat < 0 (exits succeed less often than the committor says).
- P6 (CT): L leader middle up/progress vs T_hat > 0 (escapes led by a middle node's progress move).
- P7 (CT): O climb = B vs T_hat ~ 0 (escapes climb exactly the barrier as often as the reactive paths).

```yaml
predictions:
  - {id: P1-Ring, measure: K, topology: Ring, cell: pooled, T: T_hat, stat: ll_diff, part: all, expect: ">0"}
  - {id: P1-CT, measure: K, topology: CT, cell: pooled, T: T_hat, stat: ll_diff, part: all, expect: ">0"}
  - {id: P2-Ring, measure: K, topology: Ring, cell: pooled, T: T_hat, stat: share up/no_progress, part: all, expect: "<0"}
  - {id: P2-CT, measure: K, topology: CT, cell: pooled, T: T_hat, stat: share up/no_progress, part: all, expect: "<0"}
  - {id: P3-Ring, measure: K, topology: Ring, cell: pooled, T: T_hat, stat: flat_progress_share, part: all, expect: "~0"}
  - {id: P4-Ring, measure: M, topology: Ring, cell: pooled, T: T_hat, stat: same_mover, part: all, expect: ">0"}
  - {id: P4-CT, measure: M, topology: CT, cell: pooled, T: T_hat, stat: same_mover, part: all, expect: ">0"}
  - {id: P5-CT, measure: C, topology: CT, cell: pooled, T: T_hat, stat: success_rate, part: all, expect: "<0"}
  - {id: P6-CT, measure: L, topology: CT, cell: pooled, T: T_hat, stat: leader middle up/progress, part: all, expect: ">0"}
  - {id: P7-CT, measure: O, topology: CT, cell: pooled, T: T_hat, stat: climb_equals_B, part: all, expect: "~0"}
```

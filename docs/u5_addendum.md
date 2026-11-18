# Unit 5 — addendum: entries vs duration (S-train only; S-test closed)

Written by `u5.save_factors()` (notebooks/path2/u5_factors.ipynb) from S-train only, before S-test is opened. Pre-registration: docs/u5_preregistration.md (H_lock 6d366bd52dbeaa5723f300e5c54f130749c785e1); S-train expectation: docs/u5_train_expectation.md (H_exp 4efa9239a8ed4ce750d7b214d623b635c73ad338); neither is changed. Table: results/tables/u5_train_factors.csv.
Method as Unit 5: trap graphs, T = 0.2, 0.5, 1.0, the 14 predictors and transforms, one-variable OLS, R2_cv = pooled 5-fold out-of-fold R2 (seed 573). Outcomes: log entries and log duration (duration = Y_b / entries) from u2b_graph_train.csv. Families: global = {modularity, conductance, lam2, lamP}; landscape = {n_traps, n_plateaus, B_max, B_mean, W_max, W_mean, P_max, n_gates_min, progress_share_mean}; baseline = {n_edges}.

## (i) Top 5 by R2_cv — log entries

| T | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| 0.2 | lamP (0.520, global) | n_edges (0.468, baseline) | modularity (0.442, global) | conductance (0.380, global) | lam2 (0.376, global) |
| 0.5 | lamP (0.537, global) | n_edges (0.517, baseline) | modularity (0.478, global) | conductance (0.409, global) | n_traps (0.392, landscape) |
| 1.0 | n_traps (0.524, landscape) | n_edges (0.523, baseline) | lamP (0.458, global) | modularity (0.398, global) | conductance (0.341, global) |

## (i) Top 5 by R2_cv — log duration

| T | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| 0.2 | B_max (0.788, landscape) | B_mean (0.784, landscape) | modularity (0.596, global) | conductance (0.585, global) | P_max (0.383, landscape) |
| 0.5 | modularity (0.713, global) | conductance (0.676, global) | lamP (0.448, global) | B_mean (0.331, landscape) | B_max (0.320, landscape) |
| 1.0 | modularity (0.509, global) | conductance (0.477, global) | lamP (0.406, global) | n_gates_min (0.298, landscape) | lam2 (0.201, global) |

Two-variable model for log Y_b (x_E = #1 for log entries, x_D = #1 for log duration, or its #2 if the same), against the best single predictor for log Y_b (S-train, R2_cv):

| T | x_E | x_D | R2_cv two-variable | best single | R2_cv best single | CI of difference |
|---|---|---|---|---|---|---|
| 0.2 | lamP | B_max | 0.828 | modularity | 0.751 | [0.032, 0.121] |
| 0.5 | lamP | modularity | 0.742 | modularity | 0.714 | [0.010, 0.048] |
| 1.0 | n_traps | modularity | 0.796 | lamP | 0.550 | [0.173, 0.323] |

## (ii) S-train verdict on the story (description only)

Story: entries' #1 is a global measure, duration's #1 is a landscape measure.

- T = 0.2: train supports (entries #1 = lamP, global; duration #1 = B_max, landscape)
- T = 0.5: train contradicts (entries #1 = lamP, global; duration #1 = modularity, global)
- T = 1.0: train contradicts (entries #1 = n_traps, landscape; duration #1 = modularity, global)

## (iii) Confirmatory hypotheses (S-test, scored once by scripts/u5_test.py --open)

Rules as the pre-registration: lines fitted on S-train and frozen; S-test R2 = 1 - SS_res / SS_tot (SS_tot around the S-test mean); bootstrap over S-test graphs, 2000 resamples, seed 573; "A beats B" = the 95% paired-bootstrap CI of R2(A) - R2(B) excludes 0 with A higher. Each hypothesis holds only if every case listed under it passes.

- A1 (transfer). The S-train #1 predictor is not beaten on S-test by any of the other 13 predictors:
  - log entries: T = 0.2: lamP; T = 0.5: lamP; T = 1.0: n_traps
  - log duration: T = 0.2: B_max; T = 0.5: modularity; T = 1.0: modularity
- A2 (families). The S-test #1 predictor is in the same family as the S-train #1:
  - log entries: T = 0.2: global (S-train #1 lamP); T = 0.5: global (S-train #1 lamP); T = 1.0: landscape (S-train #1 n_traps)
  - log duration: T = 0.2: landscape (S-train #1 B_max); T = 0.5: global (S-train #1 modularity); T = 1.0: global (S-train #1 modularity)
- A3 (two-part model). The two-variable line log Y_b = a + b_E x_E + b_D x_D (fitted on S-train, frozen) beats the best single predictor for log Y_b on S-test, i.e. the S-test #1 of the 14 pre-registered single-predictor lines (frozen on S-train):
  - T = 0.2: lamP + B_max (S-train best single modularity); T = 0.5: lamP + modularity (S-train best single modularity); T = 1.0: n_traps + modularity (S-train best single lamP)

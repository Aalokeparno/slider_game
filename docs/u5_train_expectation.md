# Unit 5 — S-train expectation (before opening S-test)

Written by `u5.save_train()` (notebooks/path2/u5_train_ranking.ipynb) from S-train only (366 trap graphs). Pre-registration: docs/u5_preregistration.md (H_lock 6d366bd52dbeaa5723f300e5c54f130749c785e1).
Ranking by R2_cv: pooled out-of-fold R2 of 5-fold CV within S-train (seed 573). "Beats" for P3: 95% paired-bootstrap CI (2000, seed 573, over S-train graphs) of the out-of-fold R2 difference is above 0.

## Top 5 by R2_cv — log_Yb

| T | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| 0.2 | modularity (0.751) | conductance (0.706) | B_max (0.561) | B_mean (0.544) | lamP (0.485) |
| 0.5 | modularity (0.714) | conductance (0.641) | lamP (0.626) | n_edges (0.422) | lam2 (0.401) |
| 1.0 | lamP (0.550) | modularity (0.536) | conductance (0.472) | n_edges (0.437) | lam2 (0.367) |

## Top 5 by R2_cv — log_Ya

| T | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| 0.2 | B_max (0.885) | B_mean (0.882) | progress_share_mean (0.513) | conductance (0.489) | modularity (0.474) |
| 0.5 | modularity (0.791) | conductance (0.787) | lamP (0.489) | B_mean (0.464) | B_max (0.461) |
| 1.0 | n_edges (0.771) | modularity (0.750) | conductance (0.705) | lamP (0.653) | lam2 (0.562) |

## Two-term law on S-train (log duration)

| T | 1/T | c (free, S-train) | 95% CI | law fits | R2 fixed | R2 free | walled |
|---|---|---|---|---|---|---|---|
| 0.2 | 5.00 | 4.665 | [4.301, 5.049] | yes | 0.884 | 0.888 | 37 |
| 0.5 | 2.00 | 1.802 | [1.556, 2.073] | yes | 0.684 | 0.692 | 37 |
| 1.0 | 1.00 | 0.896 | [0.747, 1.068] | yes | 0.561 | 0.568 | 37 |

## Predictions

- P1: train contradicts (T = 0.2: #1 = modularity; B_max is #3)
- P2: train contradicts (T = 1.0: #1 = lamP; n_traps #6, P_max #13)
- P3: train contradicts (T=0.2: conductance beats B_max; T=0.2: conductance beats n_edges; T=0.2: modularity beats B_max; T=0.2: modularity beats n_edges; T=0.2: lamP beats n_edges; T=0.5: conductance beats n_gates_min; T=0.5: conductance beats n_edges; T=0.5: modularity beats n_gates_min; T=0.5: modularity beats n_edges; T=0.5: lamP beats n_gates_min; T=0.5: lamP beats n_edges; T=1.0: conductance beats n_traps; T=1.0: modularity beats n_traps; T=1.0: modularity beats n_edges; T=1.0: lamP beats n_traps; T=1.0: lamP beats n_edges)
- P4: train contradicts (law fits at T = 0.2: True; at T = 1.0: True)

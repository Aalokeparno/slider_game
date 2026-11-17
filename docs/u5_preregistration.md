# Unit 5 — pre-registration (locked Nov 17, 2026)

Locked choices for Unit 5: which graph-only measures predict trap time under the logit rule, scored once on
the S-test graphs (graph-test in data/splits/graph_split.csv). This file is committed alone; its commit hash is
H_lock (config.U5_H_LOCK). Nothing below changes after the lock.

1. Graphs: trap graphs only (n_traps > 0): S-train 366, S-test 157 (verify the counts; report if different).
2. T values: 0.2, 0.5, 1.0 (logit rule, uniform start over 128 corners, as in Unit 2b).
3. Outcomes: primary log Y_b (expected steps in trap states); secondary log Y_a; two-term law uses log duration
   (duration = Y_b / entries).
4. Predictors (14): n_traps, n_plateaus, B_max, B_mean, W_max, W_mean, P_max, n_gates_min, progress_share_mean,
   lam2, conductance, modularity, lamP, n_edges. (a_c excluded.) progress_share_mean = mean over the graph's
   traps of progress_share_P from u1_traps.csv, NaNs ignored; 0 if all NaN.
5. Transforms: natural log for n_traps, n_plateaus, P_max, W_max, W_mean, n_gates_min, n_edges; raw for the rest.
6. Model: for each predictor × T × outcome, OLS line log Y = a + b·x fitted on S-train, then frozen.
   Primary metric: out-of-sample R² on S-test = 1 − SS_res/SS_tot (SS_tot around the S-test mean).
   Secondary: Spearman ρ between x and log Y on S-test.
7. Uncertainty: bootstrap over S-test graphs, 2000 resamples, seed 573, frozen fits. "A beats B" ⇔ the 95%
   paired-bootstrap CI of R²(A) − R²(B) excludes 0. Report each predictor's rank and its CI vs. rank #1.
8. Two-term law (per T, outcome log duration): fixed model log dur = a + b·log W_mean + B_max/T (B coefficient
   fixed at 1/T; a, b fitted on S-train) vs. free model log dur = a + b·log W_mean + c·B_max (c fitted).
   Report: S-test R² of both (frozen train fits); c estimated by OLS on S-test with bootstrap CI (2000, seed 573).
   "Law fits" ⇔ that CI contains 1/T. Note: only 9 walled graphs in S-test.
9. Predictions (written Nov 17, before any ranking was computed):
   P1. T = 0.2: B_max ranks #1 for log Y_b.
   P2. T = 1.0: n_traps ranks #1 for log Y_b (P_max close second).
   P3. At every T, no spectral measure (lam2, conductance, modularity, lamP) beats the best local measure, and
       none beats n_edges (beats = paired CI excludes 0).
   P4. Two-term law fits at T = 0.2 and not at T = 1.0.
10. Size transfer: out. Protocol: S-test outcomes computed once by scripts/u5_test.py --open, after H_lock and
    H_exp are committed; nothing changes after opening (report only).

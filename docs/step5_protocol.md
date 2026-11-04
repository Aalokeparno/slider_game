Step 5 protocol (locked Nov 4, 2026)
- Models: C2 (primary; exploratory Track C, frozen Nov 1), APEM published (baseline; a_theta = s_theta
  = 0.5, mu = 0.1 s), final = L1 + L1b + L3 (secondary). Presets unchanged.
- Baseline rule: the APEM version with more training passes (published 16/53 vs refit 11/53).
- Scope: EL rounds and CL v09 rounds only; old CL and whole-second files removed.
- Test cells: Ring EL (51 planned rounds, 17 groups), Ring CL v09 (39, 13), CT-Hierarchy EL (51, 17).
  Outcome filters (incomplete, aborted, empty) applied exactly as in the training L5 run set.
- Step 0: each test round's slider speed read from its rows with interface.py (fallback: the file's
  modal speed), before simulating.
- Simulation: design copy of the test rounds, all-human, R = 200. Seeds C2 9101, APEM 9102, final 9103.
- Human CIs: group bootstrap, 1,000 resamples, seed 573; zero-width rule as in P24 (Clopper-Pearson for
  a proportion at 0 or 1; +/- half the resolution for a time pinned by the interface).
- Checks per cell: targets 1 success, 2 KM median time, 3 % fraught, 4 committed split, 6 end-sitting,
  7 relapse, 8 reversal rate, 9 median A at reversal, 10a median crossing time: 27 checks.
  Target 5 dropped (Full only). Roles: primary 1-4 (12 checks), secondary 6-7, descriptive 8-10a.
- Pass: simulated mean inside the human 95% CI; borderline if within 2*SE_mc of an edge.
- Success: C2 passes more of the 12 primary checks than APEM published (tie = no success).
  Also reported: passes of 27 and median |gap| (in CI half-widths) for all three models.
- Descriptive mechanism metrics (not in the success rule): stuck-episode counts and median escape time
  per type (fraught / non-fraught), trap hops (traps.py), Fig 5-style averaged trajectories for CT only.
- First mover deferred to Path 2.
- Test opened once with load_runs(..., allow_test=True, reason="Step 5 test opening").
  After opening: report only; no model change, retuning, new metric or reseeding.

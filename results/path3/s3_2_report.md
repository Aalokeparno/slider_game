# Path 3 run S3.2: one-seed wave, seniority, backoff, fast/slow mix

Branch `path3-s3-2`. Code: `src/slider/path3/strategies.py` (S3, S4, S6, S7), `src/slider/path3/sim.py` (seed role,
noise, watch hook, strategy logs), `src/slider/path3/s32.py` (analysis), `scripts/path3_s3_2.py` (runs).
Tests: `tests/test_path3_s32.py`.

## Outcome

**No strategy is promoted.** All four candidates (S3a+, S3b+, S6, S7) fail step 1 of the decision rule, reliability.
Over 119,400 paired runs on the 597 graph-train graphs, S0b‴ never fails. The candidates do:

| Strategy | Failures | Median vs S0b‴ | p95 vs S0b‴ |
|---|---|---|---|
| S3a+ | 3,144 | faster | far worse |
| S3b+ | 3,192 | faster | far worse |
| S6 | 29 | slower | worse |
| S7 | 10 | slower | worse |

So S6 and S7 would fail step 2 even without step 1.

- **S3 (one-seed wave).** S3 is about 12 s faster than S0b‴ on sparse graphs (10 edges or fewer). It is 15–20 s
  slower on dense graphs (14 or more edges), where 8% of its runs fail. The cause is the 4 s hold after every wall
  arrival (the grace-as-hold change below): it slows the S0b‴ fallback on dense graphs.
- **S6 and S7 (backoff, fast/slow mix).** These cut breakups per minute by about 45% (1.36 and 1.39 against 2.41),
  but they are 1–3 s slower at the median and 11–16 s slower at p95.
- **S4 (report only).** S4-old is close to S0b‴ but slightly worse. S4-new is clearly worse.

## 1. Change from the spec: grace is a hold

As first written, the S3 grace allowed greedy moves but no yields for G s after a wall arrival. That version **failed
test 3** on 2026-11-08. Test 3 is S3a+ on Hierarchy, start A, no noise, 200 runs.

| Test 3 measure | Required | As specified (grace = no yields) | Grace as hold (used) |
|---|---|---|---|
| Consensus | 100% | 100% | 100% |
| Fraught-corner entries | 0 | **241** | 0 |
| Runs without yields | ≥ 95% | **21%** | 100% |

The user's decision (2026-11-08): **the grace is a hold.** After any wall arrival the player makes no decisions for G s,
in both the S0b‴ and the S2 v2 fallback.

In the code, a target action to a wall more than 0.15 away gets hold `max(hold, G)`. This also applies to yields, which
now hold for `max(H, G)` = 4 s with the defaults (H = G = 4, so the yield hold itself is unchanged). The hold is
followed by the usual human reaction delay.

Everything below uses grace as hold.

## 2. Tests

All tests passed before any screening run. They were rerun on 2026-11-09 before this report: see the
[test results](#test-results-2026-11-09) at the end.

| # | Test | Result |
|---|---|---|
| 1 | S0b‴ reruns equal saved rows: T3a and T4a samples (60 each) and a T5 sample (60) | pass (exact equality) |
| 2 | S4 with w ≡ 1 = S0b‴; S7 with q = 1 = S6; S6 with w_min = W_max = 0 = S0b‴ (Hierarchy, Full, Ring; starts A and F0; 15 seeds each) | pass (exact equality) |
| 3 | S3a+ on Hierarchy, start A, no noise, 200 runs: 100% consensus, 0 fraught entries, ≥ 95% runs with 0 yields | pass with grace as hold; failed as specified (section 1) |
| 4 | From F0, S4-new, S6 and S7 (q = 0) reach consensus in ≥ 95% of 200 runs | pass |
| 5 | Information constraint: decide and interrupt see only (t, s_i, Agr history, memory), for all new classes, with and without noise | pass |
| unit | noise off changes nothing; S4 age; S6 backoff; S7 switch; S3 travel check; S3 grace hold (incl. 40 ticks with no moves after an arrival, both fallbacks); reach at 1/7; S3b cancel; slip; fidgets only while waiting | pass |

Two more checks:

- The 4,600 stage-1 S0b‴ runs equal `blocks/T4a.csv` row for row.
- S7 with q = 1 and S6 give identical stage-1 rows (sweep table), as test 2 requires.

## 3. Stage 1: 23 screening graphs, start A, screening seeds

Defaults use 200 runs per graph (4,600 runs). Times are in seconds, and failures count as inf in the quantiles.

### 3a. Pooled defaults

| variant | runs | success 180 | success 300 | median (s) | p95 (s) | breakups/min | yields/run | yield majority share | yield undone share | time fraught | irregular |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0b''' | 4600 | 0.998 | 1.000 | 24.5 | 87.6 | 2.41 | 22.9 | 0.73 | 0.66 | 0.024 | 0.000 |
| S3a+ | 4600 | 0.933 | 0.978 | 10.3 | 205.1 | 2.52 | 25.1 | 0.77 | 0.81 | 0.006 | 0.022 |
| S3b+ | 4600 | 0.931 | 0.974 | 13.8 | 214.4 | 2.53 | 27.3 | 0.76 | 0.80 | 0.018 | 0.026 |
| S4-old | 4600 | 0.997 | 1.000 | 24.7 | 99.4 | 2.26 | 19.2 | 0.74 | 0.58 | 0.027 | 0.000 |
| S4-new | 4600 | 0.970 | 0.995 | 31.0 | 152.5 | 1.86 | 24.3 | 0.73 | 0.60 | 0.045 | 0.005 |
| S6 | 4600 | 0.996 | 1.000 | 25.3 | 101.8 | 1.36 | 12.8 | 0.71 | 0.31 | 0.044 | 0.000 |
| S7 | 4600 | 0.994 | 1.000 | 27.0 | 101.6 | 1.39 | 9.9 | 0.74 | 0.27 | 0.042 | 0.000 |

S4-old, S6 and S7 success 300 rounds to 1.000 but is 0.9996, 0.9998 and 0.9996.

### 3b. Strategy logs (pooled defaults)

**S3.** Wave success means consensus with zero yields.

| variant | wave success | seeds/run | median t all reached (s) | all reached share | reversals/run | wrong walls/run |
|---|---|---|---|---|---|---|
| S3a+ | 0.562 | 1.00 | 4.7 | 1.000 | 0.09 | 1.32 |
| S3b+ | 0.479 | 1.46 | 3.6 | 1.000 | 0.12 | 1.41 |

**S6 and S7.**

| variant | decisions/min per player | mean W (s) | slow share | switches/run |
|---|---|---|---|---|
| S6 | 18.5 | 1.83 | – | – |
| S7 | 10.2 | 2.15 | 0.579 | 5.93 |

### 3c. Per graph (defaults)

Each cell shows median / p95 in seconds. success_300 is added in brackets when it is below 1. Rows are sorted by the
S0b‴ median.

| graph | S0b''' | S3a+ | S3b+ | S4-old | S4-new | S6 | S7 |
|---|---|---|---|---|---|---|---|
| Star | 19.6 / 66 | 5.7 / 10 | 5.7 / 15 | 17.9 / 65 | 17.6 / 54 | 15.8 / 49 | 20.2 / 66 |
| G331 | 20.1 / 64 | 9.1 / 73 | 9.9 / 102 | 19.6 / 57 | 19.6 / 122 | 20.1 / 66 | 22.4 / 66 |
| CT | 20.3 / 60 | 28.6 / 120 | 24.9 / 124 | 19.9 / 55 | 29.8 / 148 (0.99) | 22.9 / 80 | 24.8 / 82 |
| G474 | 20.8 / 66 | 18.5 / 119 | 25.8 / 113 | 21.5 / 68 | 27.8 / 120 | 21.2 / 78 | 22.1 / 76 |
| Hierarchy | 21.4 / 68 | 7.6 / 14 | 7.1 / 31 | 19.4 / 62 | 20.0 / 101 | 21.1 / 75 | 19.6 / 66 |
| G426 | 22.1 / 81 | 9.5 / 99 | 10.2 / 107 | 21.9 / 79 | 27.5 / 112 | 24.9 / 71 | 24.8 / 83 |
| Ring | 22.6 / 115 | 7.9 / 71 | 7.4 / 87 | 22.9 / 100 | 23.6 / 102 | 25.4 / 110 | 23.1 / 112 |
| G427 | 23.5 / 92 | 6.8 / 12 | 6.3 / 54 | 25.5 / 92 | 22.8 / 87 | 25.0 / 86 | 23.8 / 71 |
| G853 | 23.6 / 82 | 30.1 / 205 (0.98) | 31.0 / 215 (0.98) | 25.0 / 99 | 37.5 / 147 (0.99) | 25.2 / 105 | 28.1 / 83 |
| G385 | 23.7 / 72 | 9.6 / 98 | 9.4 / 97 | 20.6 / 65 | 25.2 / 99 | 21.8 / 75 | 21.7 / 70 |
| G872 | 24.9 / 85 | 8.2 / 166 | 7.7 / 166 (0.98) | 26.2 / 95 | 32.9 / 131 | 31.1 / 101 | 30.4 / 97 |
| G964 | 25.4 / 91 | 25.9 / 284 (0.95) | 37.7 / 206 (0.98) | 28.8 / 102 | 35.8 / 161 (0.99) | 27.2 / 105 | 30.6 / 87 |
| G579 | 25.8 / 98 | 10.7 / 172 (0.99) | 10.7 / 156 (0.99) | 25.8 / 87 | 34.1 / 132 | 29.4 / 92 | 26.1 / 94 |
| G1110 | 26.0 / 90 | 41.7 / 284 (0.95) | 51.1 / inf (0.92) | 27.6 / 106 | 47.7 / 186 (0.99) | 26.4 / 102 | 32.8 / 111 |
| G624 | 26.2 / 90 | 8.6 / 171 (0.99) | 10.9 / 182 (0.98) | 23.7 / 93 | 31.4 / 109 | 26.4 / 82 | 25.8 / 77 |
| G823 | 26.4 / 87 | 8.6 / 174 (0.99) | 9.3 / 218 (0.99) | 25.0 / 76 | 30.8 / 125 | 30.1 / 122 | 28.9 / 112 |
| G522 | 26.4 / 99 | 7.8 / 123 (0.99) | 8.2 / 120 | 23.2 / 85 | 24.8 / 99 | 23.1 / 78 | 22.6 / 76 |
| G666 | 26.9 / 93 | 19.0 / 154 (0.99) | 17.5 / 146 (0.98) | 24.6 / 97 | 36.9 / 141 | 22.8 / 82 | 26.9 / 80 |
| G1067 | 27.8 / 97 | 19.4 / 233 (0.97) | 22.8 / inf (0.94) | 28.2 / 116 | 47.0 / 197 (0.99) | 27.9 / 102 | 30.4 / 112 |
| G1193 | 28.3 / 91 | 42.4 / inf (0.94) | 42.9 / inf (0.94) | 30.2 / 115 (0.99) | 47.2 / 190 (0.99) | 32.0 / 107 | 39.0 / 117 |
| G1103 | 29.2 / 90 | 36.3 / inf (0.94) | 40.3 / 260 (0.95) | 32.1 / 110 (0.99) | 59.7 / 193 (0.99) | 33.8 / 128 | 33.8 / 109 |
| G1089 | 29.6 / 81 | 29.2 / inf (0.94) | 30.3 / inf (0.94) | 29.8 / 115 | 49.4 / 176 | 30.1 / 126 | 34.0 / 110 |
| Full | 37.6 / 114 | 141.3 / inf (0.81) | 128.2 / inf (0.80) | 53.5 / 159 | 87.9 / inf (0.93) | 40.1 / 177 (0.99) | 56.5 / 205 (0.99) |

### 3d. One-factor sweeps (100 runs per graph, descriptive only)

Each setting is compared with its default's first 100 runs, which use the same seeds. The sweeps do not change any
stage-2 setting.

| strategy | setting | success 300 | median (s) | p95 (s) | breakups/min | yields/run | strategy log | default, first 100 (succ / med / p95) |
|---|---|---|---|---|---|---|---|---|
| S3a+ | G=0 | 1.000 | 23.6 | 75.4 | 2.24 | 18.7 | wave 0.00, wrong walls 3.48, fidgets 0.0 | 0.975 / 9.9 / 207.4 |
| S3a+ | G=2 | 1.000 | 11.9 | 91.2 | 2.86 | 13.3 | wave 0.52, wrong walls 1.47, fidgets 0.0 | 0.975 / 9.9 / 207.4 |
| S3a+ | G=8 | 0.988 | 10.0 | 180.9 | 2.28 | 15.5 | wave 0.57, wrong walls 1.32, fidgets 0.0 | 0.975 / 9.9 / 207.4 |
| S3a+ | check=off | 0.973 | 10.5 | 212.6 | 2.49 | 25.5 | wave 0.56, wrong walls 1.33, fidgets 0.0 | 0.975 / 9.9 / 207.4 |
| S3a+ | fallback=S2 | 0.932 | 9.9 | inf | 1.49 | 3.6 | wave 0.64, wrong walls 1.10, fidgets 0.0 | 0.975 / 9.9 / 207.4 |
| S3a+ | fidget=0.01 | 0.972 | 10.0 | 212.2 | 2.48 | 25.1 | wave 0.57, wrong walls 1.26, fidgets 0.3 | 0.975 / 9.9 / 207.4 |
| S3a+ | fidget=0.03 | 0.973 | 10.0 | 212.8 | 2.47 | 25.0 | wave 0.57, wrong walls 1.24, fidgets 1.0 | 0.975 / 9.9 / 207.4 |
| S3a+ | fidget=0.1 | 0.974 | 10.0 | 212.8 | 2.42 | 24.4 | wave 0.57, wrong walls 1.21, fidgets 3.1 | 0.975 / 9.9 / 207.4 |
| S3a+ | fidget=0.3 | 0.976 | 9.7 | 197.9 | 2.41 | 23.0 | wave 0.58, wrong walls 1.24, fidgets 8.7 | 0.975 / 9.9 / 207.4 |
| S3a+ | pause=0 | 0.969 | 20.6 | 236.1 | 2.41 | 30.9 | wave 0.47, wrong walls 1.56, fidgets 0.0 | 0.975 / 9.9 / 207.4 |
| S3a+ | pause=1 | 0.980 | 10.2 | 184.5 | 2.36 | 17.4 | wave 0.69, wrong walls 0.92, fidgets 0.0 | 0.975 / 9.9 / 207.4 |
| S3a+ | slip=0.02 | 0.961 | 14.3 | 268.5 | 2.19 | 30.0 | wave 0.51, wrong walls 1.45, fidgets 0.0 | 0.975 / 9.9 / 207.4 |
| S3a+ | slip=0.05 | 0.950 | 22.8 | 295.2 | 1.90 | 34.2 | wave 0.43, wrong walls 1.70, fidgets 0.0 | 0.975 / 9.9 / 207.4 |
| S3b+ | G=0 | 1.000 | 23.4 | 75.4 | 2.33 | 20.3 | wave 0.00, wrong walls 3.22, fidgets 0.0 | 0.974 / 13.7 / 217.6 |
| S3b+ | G=2 | 1.000 | 14.5 | 96.0 | 3.03 | 15.7 | wave 0.41, wrong walls 1.71, fidgets 0.0 | 0.974 / 13.7 / 217.6 |
| S3b+ | G=8 | 0.982 | 16.6 | 200.9 | 2.29 | 17.3 | wave 0.49, wrong walls 1.45, fidgets 0.0 | 0.974 / 13.7 / 217.6 |
| S3b+ | check=off | 0.975 | 14.9 | 214.4 | 2.50 | 28.2 | wave 0.47, wrong walls 1.47, fidgets 0.0 | 0.974 / 13.7 / 217.6 |
| S3b+ | fallback=S2 | 0.933 | 11.6 | inf | 1.47 | 3.8 | wave 0.56, wrong walls 1.20, fidgets 0.0 | 0.974 / 13.7 / 217.6 |
| S3b+ | fidget=0.01 | 0.974 | 14.2 | 226.9 | 2.47 | 27.6 | wave 0.48, wrong walls 1.43, fidgets 0.2 | 0.974 / 13.7 / 217.6 |
| S3b+ | fidget=0.03 | 0.971 | 15.9 | 232.2 | 2.24 | 26.9 | wave 0.49, wrong walls 1.41, fidgets 1.6 | 0.974 / 13.7 / 217.6 |
| S3b+ | fidget=0.1 | 0.971 | 21.3 | 219.4 | 2.22 | 25.5 | wave 0.50, wrong walls 1.43, fidgets 6.5 | 0.974 / 13.7 / 217.6 |
| S3b+ | fidget=0.3 | 0.978 | 17.9 | 215.5 | 2.35 | 26.1 | wave 0.52, wrong walls 1.50, fidgets 16.6 | 0.974 / 13.7 / 217.6 |
| S3b+ | lam=0.03 | 0.977 | 17.1 | 209.2 | 2.47 | 25.8 | wave 0.53, wrong walls 1.35, fidgets 0.0 | 0.974 / 13.7 / 217.6 |
| S3b+ | lam=0.3 | 0.977 | 16.2 | 216.9 | 2.49 | 29.1 | wave 0.42, wrong walls 1.44, fidgets 0.0 | 0.974 / 13.7 / 217.6 |
| S3b+ | pause=0 | 0.971 | 18.9 | 227.3 | 2.49 | 30.7 | wave 0.43, wrong walls 1.57, fidgets 0.0 | 0.974 / 13.7 / 217.6 |
| S3b+ | pause=1 | 0.983 | 10.1 | 170.0 | 2.50 | 19.9 | wave 0.57, wrong walls 1.14, fidgets 0.0 | 0.974 / 13.7 / 217.6 |
| S3b+ | slip=0.02 | 0.965 | 17.9 | 253.9 | 2.17 | 31.1 | wave 0.43, wrong walls 1.52, fidgets 0.0 | 0.974 / 13.7 / 217.6 |
| S3b+ | slip=0.05 | 0.946 | 23.4 | inf | 1.87 | 35.0 | wave 0.38, wrong walls 1.68, fidgets 0.0 | 0.974 / 13.7 / 217.6 |
| S4-new | A=20 | 1.000 | 26.4 | 100.9 | 2.18 | 22.4 |  | 0.993 / 30.2 / 151.6 |
| S4-new | A=30 | 1.000 | 25.7 | 92.9 | 2.25 | 22.5 |  | 0.993 / 30.2 / 151.6 |
| S4-old | A=20 | 0.987 | 28.0 | 143.7 | 2.08 | 19.6 |  | 1.000 / 25.3 / 95.4 |
| S4-old | A=30 | 0.960 | 30.9 | 254.5 | 1.86 | 20.3 |  | 1.000 / 25.3 / 95.4 |
| S6 | W_max=16 | 0.997 | 25.1 | 116.9 | 1.25 | 12.7 | dec/min 18.2, mean W 2.06 | 1.000 / 25.1 / 100.9 |
| S6 | W_max=4 | 1.000 | 24.4 | 89.0 | 1.52 | 13.0 | dec/min 19.0, mean W 1.52 | 1.000 / 25.1 / 100.9 |
| S6 | rand=off | 1.000 | 24.9 | 104.8 | 1.22 | 11.7 | dec/min 17.2, mean W 1.70 | 1.000 / 25.1 / 100.9 |
| S6 | w_min=0.25 | 0.998 | 25.8 | 102.7 | 1.47 | 15.1 | dec/min 21.3, mean W 1.26 | 1.000 / 25.1 / 100.9 |
| S6 | w_min=1 | 1.000 | 25.3 | 92.9 | 1.21 | 10.2 | dec/min 15.7, mean W 2.66 | 1.000 / 25.1 / 100.9 |
| S7 | N=10 | 1.000 | 31.3 | 104.8 | 1.33 | 8.3 | dec/min 7.8, mean W 2.25, slow 0.46 | 1.000 / 26.8 / 101.9 |
| S7 | N=3 | 1.000 | 25.8 | 97.3 | 1.38 | 10.8 | dec/min 11.6, mean W 2.12, slow 0.65 | 1.000 / 26.8 / 101.9 |
| S7 | q=0 | 1.000 | 29.2 | 101.3 | 1.41 | 8.8 | dec/min 7.4, mean W 2.36, slow 0.42 | 1.000 / 26.8 / 101.9 |
| S7 | q=1/7 | 1.000 | 28.5 | 101.6 | 1.37 | 9.4 | dec/min 8.8, mean W 2.27, slow 0.50 | 1.000 / 26.8 / 101.9 |
| S7 | q=3/7 | 1.000 | 26.6 | 103.1 | 1.41 | 10.7 | dec/min 11.7, mean W 2.09, slow 0.66 | 1.000 / 26.8 / 101.9 |
| S7 | q=1 | 1.000 | 25.1 | 100.9 | 1.35 | 12.6 | dec/min 18.5, mean W 1.82, slow 1.00 | 1.000 / 26.8 / 101.9 |

Notes on the sweeps:

- **S3 hold length G.** G sets the hold, and the hold drives the S3 tail.
  - With G = 0 (no hold beyond S0b‴'s own H for yields), S3 never fails, but the wave almost never works
    (wave success 0.00, 3.5 wrong walls per run).
  - With G = 2, every run reaches consensus, the median is 12–15 s, p95 is 91–96 s, and about half the runs are
    clean waves.
  - This is only a stage-1 observation, not tested in stage 2. Stage 2 used the default G = 4, as specified.
- **S3 yield undone share.** It drops to about 0 at G = 8. A yield counts as undone only if the player is back on its
  old side within 10 s, and an 8 s hold plus the reaction delay and travel time rarely fits in 10 s.
- **S3 fallback = S2.** p95 is inf: 7% of runs fail.
- **Fidget barely matters.** A fidget rate of 0.3 per second while waiting gives 9–17 fidgets per run with almost no
  change in the result.
- **Slip hurts S3 steadily.**
- **A pause of 1 s before probing** helps both S3 versions (fewer wrong walls).
- **S4-old with a larger A gets worse.** At A = 30, success is 0.960.
- **S7 with q = 1 is exactly S6**, as test 2 requires.

### 3e. Figures

- (i) `results/figures/path3_s3_2_fidget.png`: S3 success and median against the fidget rate μ.
- (ii) `results/figures/path3_s3_2_S7_q.png`: S7 median and p95 against q.
- (iii) `results/figures/path3_s3_2_S6_Wmax.png`: S6 breakups/min against W_max, with the S0b‴ reference.
- (iv) `results/figures/path3_s3_2_edges.png`: per-graph Δmedian against edge count (stage 2).

## 4. Stage 2: 597 graph-train graphs, start A, T5 seeds (base 5731)

S3a+, S3b+, S6 and S7 use 200 runs per graph. S4-old and S4-new use 100, paired with the first 100 S0b‴ runs. The
S0b‴ reference is the existing T5 "S0b‴ 0.75" runs on the same seeds; they were not rerun.

Bootstrap settings:

- Paired, two-level: resample graphs, then runs within each graph.
- 1,000 resamples, seed 573.
- 95% percentile CIs.
- Worst decile = the fixed T5 set of 60 graphs.

"inf" means at least one graph (or the pool) has p95 = inf, i.e. more than 5% of its runs failed; no finite CI exists
then. Rows `X - S0b'''` are paired differences, with negative meaning X is faster. Failures are counts of runs with no
consensus by 300 s.

### 4a. All 597 train graphs

| variant | pooled median (s) | pooled p95 (s) | mean per-graph p95 (s) | worst-decile p95 (s) | failures |
|---|---|---|---|---|---|
| S0b''' | 25.2 [24.9, 25.5] | 85.6 [84.4, 86.9] | 83.7 [82.5, 84.9] | 101.4 [96.6, 103.9] | 0 [0, 0] |
| S3a+ | 19.2 [17.1, 22.1] | 231.6 [223.2, 240.0] | inf | inf | 3144 [2840, 3466] |
| S3b+ | 21.0 [18.6, 22.5] | 232.2 [224.2, 241.3] | inf | inf | 3192 [2890, 3514] |
| S6 | 26.3 [26.0, 26.7] | 101.8 [100.9, 102.9] | 97.9 [96.2, 99.7] | 116.0 [110.4, 122.5] | 29 [15, 46] |
| S7 | 28.5 [28.1, 29.0] | 96.8 [94.7, 98.9] | 92.4 [91.1, 94.1] | 107.4 [102.2, 114.4] | 10 [2, 21] |
| S3a+ - S0b''' | -6.0 [-8.0, -3.3] | +146.0 [138.2, 153.8] | inf | inf | +3144 [2840, 3466] |
| S3b+ - S0b''' | -4.2 [-6.5, -2.9] | +146.6 [138.8, 155.0] | inf | inf | +3192 [2890, 3514] |
| S6 - S0b''' | +1.1 [0.8, 1.4] | +16.2 [15.1, 17.3] | +14.3 [12.7, 15.8] | +14.6 [10.6, 21.8] | +29 [15, 46] |
| S7 - S0b''' | +3.3 [3.0, 3.6] | +11.2 [9.5, 13.0] | +8.8 [7.6, 10.3] | +6.0 [2.7, 13.7] | +10 [2, 21] |

S4, 100 runs per graph (report only):

| variant | pooled median (s) | pooled p95 (s) | mean per-graph p95 (s) | worst-decile p95 (s) | failures |
|---|---|---|---|---|---|
| S0b''' (first 100) | 25.2 [24.8, 25.5] | 85.2 [83.6, 86.9] | 82.2 [80.9, 83.9] | 99.5 [94.0, 103.9] | 0 [0, 0] |
| S4-old | 26.4 [26.0, 26.9] | 95.3 [92.9, 97.9] | 89.0 [87.0, 91.3] | 110.1 [102.6, 117.6] | 4 [0, 11] |
| S4-new | 34.0 [32.9, 35.1] | 156.1 [151.7, 161.3] | inf | inf | 279 [224, 338] |
| S4-old - S0b''' | +1.2 [0.8, 1.6] | +10.1 [7.9, 12.5] | +6.8 [4.8, 8.8] | +10.6 [3.9, 19.1] | +4 [0, 11] |
| S4-new - S0b''' | +8.8 [7.9, 9.8] | +70.9 [66.9, 75.8] | inf | inf | +279 [224, 338] |

### 4b. Dense subset: top edge tercile, 161 graphs

| variant | pooled median (s) | pooled p95 (s) | mean per-graph p95 (s) | worst-decile p95 (s) | failures |
|---|---|---|---|---|---|
| S0b''' | 28.4 [27.9, 29.0] | 93.5 [91.0, 95.9] | 91.9 [89.3, 93.6] | 104.1 [98.5, 107.0] | 0 [0, 0] |
| S3a+ | 44.2 [40.9, 47.0] | inf | inf | inf | 2042 [1852, 2249] |
| S3b+ | 42.2 [38.4, 44.4] | inf | inf | inf | 2079 [1888, 2270] |
| S6 | 31.3 [30.5, 32.0] | 120.0 [116.7, 123.5] | 117.6 [114.4, 120.8] | 129.1 [122.8, 136.6] | 22 [10, 37] |
| S7 | 34.5 [33.9, 35.2] | 112.2 [109.3, 115.1] | 109.7 [107.2, 112.7] | 117.9 [111.9, 126.9] | 8 [1, 19] |
| S3a+ - S0b''' | +15.8 [12.9, 18.3] | inf | inf | inf | +2042 [1852, 2249] |
| S3b+ - S0b''' | +13.8 [10.2, 15.7] | inf | inf | inf | +2079 [1888, 2270] |
| S6 - S0b''' | +2.9 [2.1, 3.6] | +26.5 [23.2, 30.1] | +25.7 [22.9, 29.1] | +25.0 [20.1, 33.2] | +22 [10, 37] |
| S7 - S0b''' | +6.1 [5.4, 6.9] | +18.7 [15.6, 21.6] | +17.9 [15.5, 21.2] | +13.8 [9.0, 24.1] | +8 [1, 19] |
| S0b''' (first 100) | 28.3 [27.7, 29.0] | 92.9 [89.9, 96.3] | 89.8 [87.1, 92.6] | 100.7 [94.9, 106.5] | 0 [0, 0] |
| S4-old | 32.4 [31.5, 33.3] | 112.8 [109.3, 117.0] | 108.7 [105.3, 113.3] | 121.5 [113.3, 133.5] | 3 [0, 10] |
| S4-new | 48.4 [46.5, 50.5] | 208.3 [199.9, 217.4] | inf | inf | 216 [168, 265] |
| S4-old - S0b''' | +4.1 [3.2, 5.0] | +19.9 [15.3, 24.2] | +18.9 [15.2, 23.8] | +20.8 [12.4, 33.5] | +3 [0, 10] |
| S4-new - S0b''' | +20.1 [18.4, 22.0] | +115.4 [106.8, 124.2] | inf | inf | +216 [168, 265] |

The worst-decile column here is the mean over those of the 60 fixed worst-decile graphs that are dense.

### 4c. Out of sample: 574 graphs not in the screening set

| variant | pooled median (s) | pooled p95 (s) | mean per-graph p95 (s) | worst-decile p95 (s) | failures |
|---|---|---|---|---|---|
| S0b''' | 25.3 [25.0, 25.6] | 85.7 [84.5, 87.0] | 83.8 [82.7, 85.0] | 101.1 [96.1, 103.2] | 0 [0, 0] |
| S3a+ | 19.7 [17.4, 22.5] | 232.3 [224.3, 241.3] | inf | inf | 3022 [2744, 3371] |
| S3b+ | 21.4 [18.9, 22.7] | 233.4 [224.9, 242.0] | inf | inf | 3099 [2797, 3415] |
| S6 | 26.4 [26.0, 26.8] | 101.8 [100.9, 102.8] | 98.1 [96.4, 99.8] | 115.2 [108.9, 122.1] | 27 [14, 45] |
| S7 | 28.6 [28.2, 29.0] | 96.7 [94.7, 98.8] | 92.5 [91.1, 94.3] | 106.6 [102.2, 113.5] | 9 [1, 20] |
| S3a+ - S0b''' | -5.6 [-7.8, -2.9] | +146.6 [138.9, 154.9] | inf | inf | +3022 [2744, 3371] |
| S3b+ - S0b''' | -3.9 [-6.2, -2.7] | +147.7 [139.8, 156.1] | inf | inf | +3099 [2797, 3415] |
| S6 - S0b''' | +1.1 [0.8, 1.5] | +16.1 [14.9, 17.2] | +14.2 [12.7, 15.8] | +14.1 [10.1, 21.8] | +27 [14, 45] |
| S7 - S0b''' | +3.3 [3.0, 3.6] | +11.0 [9.1, 12.8] | +8.6 [7.3, 10.2] | +5.5 [2.3, 13.5] | +9 [1, 20] |
| S0b''' (first 100) | 25.3 [24.9, 25.6] | 85.4 [83.8, 87.2] | 82.4 [81.0, 84.1] | 99.2 [93.9, 102.9] | 0 [0, 0] |
| S4-old | 26.5 [26.1, 26.9] | 95.4 [92.9, 97.9] | 89.1 [87.1, 90.9] | 109.7 [102.5, 116.6] | 2 [0, 7] |
| S4-new | 34.1 [33.1, 35.3] | 156.3 [152.1, 161.9] | inf | inf | 266 [214, 324] |
| S4-old - S0b''' | +1.2 [0.9, 1.6] | +10.0 [7.7, 12.3] | +6.7 [4.7, 8.4] | +10.4 [3.4, 18.2] | +2 [0, 7] |
| S4-new - S0b''' | +8.8 [8.0, 9.9] | +70.9 [67.1, 76.1] | inf | inf | +266 [214, 324] |

### 4d. Decision rule: S3a+, S3b+, S6, S7 (200 runs per graph, all 597 graphs)

| strategy | 1 reliability: failures (S0b‴ 0) | 2a pooled median diff (s) | 2b pooled p95 diff (s) | 3 dense median / p95 diff (s), report only | 3 worst-decile p95 diff (s), report only | verdict |
|---|---|---|---|---|---|---|
| S3a+ | 3144; CI [2840, 3466] → FAIL | -6.0 [-8.0, -3.3] → below 0 | +146.0 [+138.2, +153.8] → not below 0 | +15.8 [+12.9, +18.3] / inf (no CI) | inf (no CI) | not promoted (reported separately: needs an assigned seed role) |
| S3b+ | 3192; CI [2890, 3514] → FAIL | -4.2 [-6.5, -2.9] → below 0 | +146.6 [+138.8, +155.0] → not below 0 | +13.8 [+10.2, +15.7] / inf (no CI) | inf (no CI) | not promoted |
| S6 | 29; CI [15, 46] → FAIL | +1.1 [+0.8, +1.4] → not below 0 | +16.2 [+15.1, +17.3] → not below 0 | +2.9 [+2.1, +3.6] / +26.5 [+23.2, +30.1] | +14.6 [+10.6, +21.8] | not promoted |
| S7 | 10; CI [2, 21] → FAIL | +3.3 [+3.0, +3.6] → not below 0 | +11.2 [+9.5, +13.0] → not below 0 | +6.1 [+5.4, +6.9] / +18.7 [+15.6, +21.6] | +6.0 [+2.7, +13.7] | not promoted |

Reading the steps:

- **Step 1 fails for all four.** For S6 and S7 the failure CIs also exclude 0, so this is not just the literal
  "count ≤ count" reading (spec gap 20).
- **Step 2.** Only S3's median difference is below 0; its p95 difference is far above 0.
- S4 is report only: S4-old is close to S0b‴ but slightly worse on every measure, and S4-new is clearly worse.

### 4e. Outcome-class mix (stage 2, share of runs)

| variant | consensus | fixed trap | periodic | irregular |
|---|---|---|---|---|
| S0b''' | 1.0000 | 0 | 0 | 0 |
| S3a+ | 0.9737 | 0.0000 | 0 | 0.0263 |
| S3b+ | 0.9733 | 0.0000 | 0 | 0.0267 |
| S4-old | 0.9999 | 0.0000 | 0 | 0.0001 |
| S4-new | 0.9953 | 0.0000 | 0 | 0.0047 |
| S6 | 0.9998 | 0.0000 | 0 | 0.0002 |
| S7 | 0.9999 | 0.0000 | 0 | 0.0001 |

S4-new has one fixed trap in 59,700 runs. Every other failure is irregular, meaning the players keep moving until the
300 s limit.

### 4f. Where the differences come from (per graph, by edge count)

The table shows the median per-graph Δmedian against S0b‴ in seconds, with the failure rate in brackets.

| edges | graphs | S3a+ | S3b+ | S6 | S7 | S4-old | S4-new |
|---|---|---|---|---|---|---|---|
| 6–10 | 251 | -12.7 (0.4%) | -12.1 (0.5%) | +0.5 (0.01%) | +1.3 (0%) | -0.4 (0%) | +3.3 (0.05%) |
| 11–13 | 249 | +2.5 (2.9%) | +0.2 (2.8%) | +1.8 (0.01%) | +3.5 (<0.01%) | +1.7 (0.01%) | +12.2 (0.35%) |
| 14–21 | 97 | +22.2 (7.8%) | +18.9 (7.9%) | +3.1 (0.11%) | +6.6 (0.04%) | +5.0 (0.02%) | +22.8 (1.8%) |

How many of the 597 graphs each strategy is faster on:

| Strategy | Faster median | Faster p95 |
|---|---|---|
| S3a+ | 326 | 47 |
| S3b+ | 336 | 35 |
| S6 | 188 | 97 |
| S7 | 80 | 166 |
| S4-old | 215 | 215 |
| S4-new | 50 | 20 |

S6 failures are spread thinly: at most 2 runs on any one graph, on 25 graphs. S7 failures fall on 7 graphs, all but
one with 15 or more edges.

### 4g. S3 tail diagnosis (stage 1, replayed failed runs)

In every replayed S3 failure:

- the wave did reach every player, so all players were in WALL;
- the fallback (S0b‴ with the grace hold) then kept churning until 300 s, with about 200 yields, which counts as
  irregular.

The grace hold gives a 4 s pause after every wall arrival, including greedy flips from one wall to the other. On dense
graphs many flips happen, so players spend much of the time frozen, and the S0b‴ escape works more slowly than it does
on its own.

The worst graph is Full: success 0.81 (S3a+) and 0.80 (S3b+), median 141 s and 128 s, 7–9% wave success. There the seed's first move reaches all six
followers at the same moment, and their probes collide.

The G sweep (section 3d) is consistent with this account: G = 0 and G = 2 remove the failures. This is not a bug; it
follows from the grace-as-hold rule.

## 5. Spec gaps: interpretations made, listed rather than chosen silently

1. **Grace as hold** (user's decision, section 1). Any target action to a wall more than 0.15 away holds
   `max(hold, G)`, so yields hold `max(H, G)`. The same rule applies in the S2 v2 fallback.
2. **Reach threshold.** "Agr ≥ 1/7" is read on the integer percent display that players see, i.e. Agr ≥ 14%.
3. **"s still 0"** is read as "still in WAIT". A fidget step does not end WAIT.
4. **FOLLOW timing.** The FOLLOW pause (Exp, mean 0.3 s) replaces the human reaction delay for the first probe. WAIT
   is checked every tick, with no reaction delay.
5. **Travel check.**
   - It is on by default.
   - "Falls" means Agr strictly decreasing over the last three ticks, checked 2 to 5 ticks after committing to the
     wall.
   - A player reverses at most once.
6. **Seed timing.**
   - S3a: the seed acts at its first decision, after its start delay.
   - S3b: t_seed ~ Exp(λ) is measured from round start and drawn at the player's first call.
   - S3b cancel: any Agr change on a tick when the player's own s did not change.
7. **Fidget.**
   - It applies only to players whose actions so far are all idle.
   - It is a ±1 step with probability μ·dt per tick.
   - Noise uses a spawned random generator, so runs with no noise are unchanged (tested).
8. **Slip.**
   - It applies to every directional action.
   - A step reverses direction. A target wall more than one step away flips to the other wall. Any other target is
     mirrored.
   - A yield that ends up not moving is not counted as a yield.
9. **Wrong wall** is each follower's first wall after leaving WAIT, compared with the final consensus side. It is NaN
   for runs without consensus.
10. **S4 side change.** It is detected at the tick where sign(s) = −side. The side is set at the player's first wall.
11. **S4 "weight applied after the hold"** is read as: the hold is unchanged, and w scales the yield chance at later
    decisions. A player that never changed side has w = 1.
12. **S6 wait.**
    - It follows every decision, including decisions at Agr = 1.
    - W is updated before the wait is drawn.
    - W resets on any greedy move, including a mid-slider "fell" move.
    - The wait is added on top of the yield hold.
13. **S6 decisions/min** is per player.
14. **S7 roles.**
    - The fast/slow role is drawn at the first decision.
    - A switch takes effect from the next decision.
    - A decision at Agr = 1 resets the fail count.
    - Initially slow players stay slow.
    - S7 decisions/min counts slow decisions only.
15. **Stage-2 storage.** Stage-2 rows are in `runs_s3_2_confirm.parquet` (18 MB), not runs.csv, because of GitHub's
    100 MB file limit. Stage-1 rows were appended to runs.csv as blocks `S32_screen` and `S32_sweep`.
    runs.csv gained 23 columns (the S3.2 logs, `variant`, `run`). They are empty for the 81,280 earlier rows, and the
    values in those rows are unchanged (checked against the previous commit).
16. **Column names.** Sim column names are kept (`breakups`, `yield_majority_share`, `yield_undone_share`, `outcome`).
17. **Sweep comparison.** Sweeps are compared with the first 100 default runs, which use the same seeds.
18. **S4 in stage 2** is paired with the first 100 T5 S0b‴ runs per graph.
19. **Dense subset.** Top edge tercile with the T5 convention: a graph exactly at the cut goes to the lower bin. This
    gives 161 graphs.
20. **Reliability step** read literally: the strategy's failure count must be ≤ S0b‴'s count (0). The bootstrap CI of
    the difference is reported alongside.
21. **inf p95 values.** A p95 of inf (more than 5% failures on a graph or pool) gives no finite CI. It is shown as
    "inf".

## 6. Files

- Runs:
  - `results/path3/runs.csv`: stage 1, 131,100 rows added.
  - `results/path3/runs_s3_2_confirm.parquet`: stage 2, 597,000 rows.
  - Checkpoints under `results/path3/blocks/S32/`.
- Tables: `results/path3/s3_2_stage1.csv`, `s3_2_stage2_summary.csv`, `s3_2_stage2_per_graph.csv`,
  `s3_2_decision.csv`, `s3_2_outcomes.csv`.
- Figures: `results/figures/path3_s3_2_{fidget,S7_q,S6_Wmax,edges}.png`.
- To regenerate: `python scripts/path3_s3_2.py collect analyse`. The runs themselves come from
  `python scripts/path3_s3_2.py stage1 stage2`.

## Test results (2026-11-09)

`pytest tests/test_path3_s32.py` (fast): **22 passed** (17 s).

`pytest tests/test_path3_s32.py -m slow`: **4 passed** (11 s). Printed results:

| Test | Result |
|---|---|
| Test 3, S3a+ on Hierarchy | consensus 1.0, fraught entries 0, runs without yields 1.0 |
| Test 4, S4-new from F0 | success_300 = 1.0 |
| Test 4, S6 from F0 | success_300 = 1.0 |
| Test 4, S7 (q = 0) from F0 | success_300 = 1.0 |

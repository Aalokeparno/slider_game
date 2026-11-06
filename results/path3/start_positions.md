# Where sliders start (Slider Game logs)

Script: `scripts/check_start_positions.py` (read-only). Per-round table: `results/path3/start_positions_rounds.csv`
(s0 and first-move time of all 7 seats, agreement at row 0, first mover(s), spontaneous starters). Summary:
`results/path3/start_positions_summary.csv`.

Rounds: `load_runs("behavior", roles={"train", "val"}, keep_whole_second=True)`. That gives 391 rounds on
**Hierarchy, Full and Star only**. Aborted 1-row rounds and rounds with fewer than 2 rows are dropped. Bot and
incomplete rounds are kept. **Ring and CT-Hierarchy were not loaded** (see the open question at the end).

## Result in one paragraph

In **lab v09, EL and CC, every round starts with all 7 sliders at exactly 0**, and agreement is 0 at the first
logged row. That row comes 0.04–0.34 s after the round starts (one CC Star round: 1.5 s). So the start is observed
and it is the centre. Old-CL files (before v09) are different. Their log starts *after* play has begun (first row
at 2–67 s), so ~30–40% of rounds already show moved sliders at the first row. Those values do not match the
previous round's end. The settings XML gives `InitialValues 0 0 0 0 0 0 0` for every lab wave, and the cloud log
writes `InitialValues: 0.0 …` for all 221 games. So sliders are reset to 0 everywhere, and in old CL the first
moves are simply not logged. The median human makes a first move ~1 s after the start (first mover) and ~2.1–2.5 s
(all players).

## Units

The logs store X in [-1, 1] already (lab XML: `minContribution="-1" maxContribution="1"`; cloud uses the same scale).
No conversion was needed. I report **s = weights · X** (the repo's `sign_correct`). That flips the sign of some seats.
It does not change |s|, zero-ness, the grid check or move times.

## Start positions (row 0, all 7 seats)

| dataset | n rounds | % all s = 0 | % all \|s\| < 0.05 | % all on 0.1 grid | s0 min / median / max | max agreement at row 0 | t of row 0: median / max (s) |
|---|---|---|---|---|---|---|---|
| **CL v09** | 78 | 100 | 100 | 100 | 0 / 0 / 0 | 0 | 0.06 / 0.34 |
| EL | 51 | 100 | 100 | 100 | 0 / 0 / 0 | 0 | 0.06 / 0.24 |
| CL oldCL | 29 | 69.0 | 79.3 | 69.0 | -1.00 / 0 / 0.96 | 34 % | 3.39 / 67.4 |
| CL whole-second | 17 | 58.8 | 70.6 | 58.8 | -1.00 / 0 / 0.92 | 51 % | 5 / 46 |
| CC (all) | 216 | 100 | 100 | 100 | 0 / 0 / 0 | 0 | 0.04 / 1.53 |
| CC no-bot | 121 | 100 | 100 | 100 | 0 / 0 / 0 | 0 | 0.04 / 1.53 |

The same holds in each network: Hierarchy, Full, Star are all 100 % at 0 in v09 / EL / CC. In old CL:
Hierarchy 57 % / Full 80 % all-zero; in whole-second: Hierarchy 62.5 % / Full 55.6 %.

Histogram of s0 (value: count of seat-rounds):
- CL v09: 0: 546 · EL: 0: 357 · CC: 0: 1512
- CL oldCL (203): 0: 185; ±0.02: 7; 0.08: 2; -0.14, 0.26, 0.28, -0.30, 0.46, -0.54, -0.68, 0.96, -1.00: 1 each
- CL whole-second (119): 0: 102; ±0.02: 6; ±0.04–0.06: 3; ±0.48–0.66: 4; ±0.9–1.0: 4

Old-CL values off the 0.1 grid are on its 0.02 step. They are moves already made, not a different start.

## First-move time (seconds from round start; first row where s_i changes)

| dataset | first mover: median [IQR] | all human players: median [IQR] | humans who never moved | % rounds with tied first movers |
|---|---|---|---|---|
| **CL v09** | 1.05 [0.84, 1.40] | 2.24 [1.44, 3.95] | 0 | 6.4 |
| EL | 1.00 [0.72, 1.33] | 2.10 [1.34, 3.37] | 0 | 5.9 |
| CL oldCL* | 3.80 [3.45, 5.01] | 6.21 [4.46, 9.62] | 0 | 10.3 |
| CL whole-second* | 6.0 [4.0, 12.0] | 9.0 [6.0, 15.0] | 4 | 29.4 |
| CC | 1.01 [0.76, 1.39] | 2.47 [1.41, 6.11] | 58 | 0 |
| CC no-bot | 1.10 [0.77, 1.42] | 2.64 [1.48, 6.38] | 29 | 0 |

\* biased late: the early moves come before the first logged row.

By network (first mover / all players, medians):
- CL v09: Hierarchy 1.10 / 2.31; Full 1.00 / 2.15
- EL: Hierarchy 1.00 / 2.10
- CC: Hierarchy 1.01 / 2.34; Full 0.91 / 2.00; Star 1.22 / 3.55

## Spontaneous starters (first move strictly before every neighbour's first move)

| dataset | network | n | mean | 0 | 1 | 2 | 3+ |
|---|---|---|---|---|---|---|---|
| **CL v09** | all | 78 | 1.87 | 4 | 38 | 13 | 23 |
| | Hierarchy | 39 | 2.85 | 0 | 3 | 13 | 23 |
| | Full | 39 | 0.90 | 4 | 35 | 0 | 0 |
| EL | Hierarchy | 51 | 2.96 | 0 | 6 | 9 | 36 |
| CL oldCL* | Hierarchy | 14 | 3.07 | 0 | 1 | 2 | 11 |
| | Full | 15 | 0.87 | 2 | 13 | 0 | 0 |
| CL whole-second* | Hierarchy | 8 | 2.75 | 1 | 0 | 2 | 5 |
| | Full | 9 | 0.78 | 2 | 7 | 0 | 0 |
| CC | all | 216 | 2.14 | 17 | 84 | 43 | 72 |
| | Hierarchy | 74 | 2.39 | 5 | 5 | 32 | 32 |
| | Full | 71 | 0.89 | 8 | 63 | 0 | 0 |
| | Star | 71 | 3.13 | 4 | 16 | 11 | 40 |
| CC no-bot | Hierarchy | 42 | 2.67 | 2 | 1 | 18 | 21 |
| | Full | 39 | 0.97 | 1 | 38 | 0 | 0 |
| | Star | 40 | 3.55 | 1 | 9 | 5 | 25 |

How to read this: in Full, everyone is everyone's neighbour, so there is at most 1 spontaneous starter, and 0 only
when the first movers tie. In Hierarchy (4 leaves) and Star (6 leaves), leaves whose single neighbour has not yet
moved count as spontaneous. So most rounds have 2+ independent starters. Hierarchy/Star counts depend partly on the
graph's structure and should not be read only as "eagerness".

## Ambiguities (flagged, not settled)

1. **Round start = first logged row.** Lab: the first `seconds` row after `Wave Starting`; t is the logged time.
   Cloud: t = clock − time of `Starting Game k` (the parser's t0), and the first row is the first `Adjusted Plays`
   row (~0.04 s). Times are measured from these origins. In old CL, wave 1 times start at 42–67 s, so that clock
   probably includes time before the wave began.
2. **Old CL / whole-second files:** the true start is not in the log. "s at first sample" there is not the start
   position. Their first-move times and spontaneous counts are biased (moves before the first row are lost). I'd
   leave them out of any start-position claim.
3. **s vs X:** reported as sign-corrected s. Raw X gives the same zero / grid / timing results.
4. **First move** = the first change versus the previous row (|Δs| > 1e-9), not "first |s| > 0". These are the same
   here because every observed start is 0.
5. **Ties:** a player who moves in the same row as a neighbour's first move is *not* counted as spontaneous
   (`n_ties_with_neighbour` in the CSV; mean 0.17 per round in lab v09, 0.14 in EL, 0 in CC no-bot). CC logs one
   player's play per row, so humans can't tie there (the 0.08 in CC with bots comes from robot seats). Lab rows
   are tens of ms apart, so a tie means moving within one logged row of each other.
6. **Robots (CC):** 95 of 216 CC rounds have a robot seat (at some point). Robot seats keep their positions and
   count as neighbours, but are never first movers or spontaneous starters. "CC no-bot" is reported separately.
7. **"Never moved" humans** (58 in CC, 29 without bots): their first-move time is NaN. They are left out of the
   all-player quantiles, and as neighbours they count as never moving.
8. **Grid:** "on the 0.1 grid" passes trivially at 0. The old-CL interface step is 0.02, and lab v09's settings say
   `IncValue="0.05"` although logged moves are 0.1 steps (see interface_table).

## Open question: Ring and CT-Hierarchy

The request asked for a Ring / CT split. Those are the held-out test topologies. `config.py` says no new metrics on
them before Nov 25, so they were **not loaded**. The Step 5 test was already opened on 2026-11-04. Running
`python scripts/check_start_positions.py --include-test` adds them. It makes one logged `load_runs(allow_test=True)`
call. Given that every other lab and CC round starts at 0, and the settings say `InitialValues 0` for every wave, the
answer for Ring/CT start values is very likely the same. The first-move and spontaneous counts would be new metrics.

# Sealed execution-feedback construction result (hard bank v1)

One-shot sealed evaluation run 2026-11-06 under the preregistered
design in [feedback-construction-design.md](feedback-construction-design.md).
Artifact: `artifacts/construction-hard-final-20261106/report.json`.
Corpus: `fixtures/construction_hard_corpus_v1.json`, sha256
`59f31765...` (verified before inference; two generator sources
pinned and hashed in the report). Budget frozen at 0.5 CPU seconds
per task from the train-pool calibration in
`artifacts/construction-hard-calibration-20261106/`. Eval split:
300 tasks, 100 per minimal-size stratum {6, 7, 8}. Eval rows never
trained; fit and rollout-generation CPU are charged to the learned
arms. Single-threaded under the builder lock; 4,406.7 worker CPU
seconds total.

## Verdict

**FAIL. The frozen kill criterion is met on both prongs**: the
feedback-trained arm does not beat both untrained and imitation on
solves, and does not survive equal total CPU against streaming.
Per the criterion as frozen before the run, the ANN-construction
hypothesis at accessible scale is closed and is not tuned further.

| Seed | Streaming | Untrained | Imitation | Feedback |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 66/300 | 0 | 26 | 23 |
| 1 | 64/300 | 0 | 23 | 18 |
| 2 | 60/300 | 0 | 37 | 15 |
| Mean | 63.3 | 0.0 | 28.7 | 18.7 |

Feedback loses to imitation on every seed, not only on average.
Equal-CPU comparison (fit and rollout generation charged, amortized
over the actual 300-task evaluation volume): truncated mean solves
0.0044 for the learned arm vs 0.211 for streaming — streaming is
roughly 47x the learned arm at equal total CPU. `survives_equal_cpu:
false`, `kill: true`.

## Why the feedback loop never engaged

The mechanism is visible in the report's per-round records and is
the most informative part of the result:

- Round 0 is a 30-epoch teacher-imitation warm start (9,493
  examples; the imitation arm trains 100 epochs on the same data).
- Each subsequent round ran 1,200 rollouts on the training pool and
  obtained exactly **1 checker-accepted rollout** per round (seeds 0
  and 1), adding 5 new examples (9,493 → 9,498). Seed 2 obtained 0
  in round 1 and stopped (`stopped_zero_acceptance: true`); its
  final model is the round-0 fit.
- With the example set unchanged after round 1 and each round
  re-fitting from the same initial parameters, rounds 2-3 reproduce
  the round-1 model bit-for-bit (identical final hashes and final
  losses). The loop reached a fixed point after one feedback round.

So at this bank difficulty the controller solves ~1 in 1,200 of its
own training rollouts: the self-generated execution signal is too
sparse to bootstrap. The feedback arm is effectively a 30-epoch
imitation model plus five examples, and its deficit against the
100-epoch imitation arm is consistent with receiving less teacher
training, not with execution feedback actively harming it. The
hypothesis under test — that checker-scored rollouts would supply
a stronger learning signal than teacher imitation — was never
given a live signal to work with, and the kill criterion was
written to close the line precisely when that happens at this
scale.

## Costs (CPU seconds, single-threaded)

| Seed | Imitation fit | Feedback fit | Feedback rollouts | Streaming eval |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 564.8 | 758.0 | 147.6 | 157.1 |
| 1 | 631.7 | 657.9 | 147.5 | 155.3 |
| 2 | 576.3 | 175.6 | 47.8 | 156.7 |

Trained-arm eval cost ~25-26 s per seed; streaming has no fit cost.
Seed 2's lower feedback cost reflects its early stop, and it still
solves fewest. The verdict does not turn on timing noise: both
failing margins are wide and consistent across seeds.

## Stratum pattern (mean solves per seed; 100 tasks per stratum)

| Arm | Stratum 6 | Stratum 7 | Stratum 8 |
| --- | ---: | ---: | ---: |
| Streaming | 63.3 | 0 | 0 |
| Imitation | 5.3 | 0 | 23.3 |
| Feedback | 3.0 | 0 | 15.7 |
| Untrained | 0 | 0 | 0 |

The calibration anomaly replicates: the trained arms solve more of
stratum 8 than of stratum 6 and nothing at stratum 7, while
streaming solves only stratum 6. Disclosed caveat: stratum-8
identities are an enumeration-order prefix (collection stopped
after 2,900 candidates examined, per corpus metadata), so that
stratum may be structurally easier for the controller in ways the
minimal-size label does not capture. This does not affect the
verdict: feedback trails imitation within every stratum, and the
equal-CPU failure is independent of stratum composition.

## Calibration history (retained)

The first bank design (sampled-ops strata) failed calibration:
streaming solved 80% at 0.5 s/task and 100% within the 10 s cap on
the train pool, because sampled tree size is a poor proxy for the
minimal size streaming enumerates by. The bank was rebuilt on exact
minimal-size catalog strata {6, 7, 8} before any solver touched the
eval split; the failed sampled calibration is retained in the
calibration artifact directory. Second calibration (train pool,
40 tasks per stratum): at 0.5 CPU s/task, streaming 21.7%
(0.65/0.0/0.0 by stratum), imitation 50% (0.475/0.225/0.80),
untrained 0% — the smallest frozen candidate satisfying the design
rule (streaming ≤ 25%, imitation ≥ 10%). The sealed eval streaming
rate (21.1%) matches calibration closely.

## Interpretation

This closes the execution-feedback construction question as
preregistered. Three things are established by sealed evidence:

1. The controller still learns something: both trained arms beat
   the untrained control (0/300 on all seeds) on a bank where
   streaming times out on two of three strata. The signal is real
   but small (6-10% solves).
2. Execution feedback via expert iteration does not improve on
   teacher imitation here, because rollout acceptance (~0.08%) is
   far too sparse to move the training set. This is a measured
   property of the loop, not an inference from the final scores.
3. At matched total CPU the learned arms are not competitive with
   streaming enumeration (47x gap), consistent with the sealed
   construction-v1 cost-gate failure
   ([result](construction-final-results.md)) and the frozen
   transfer v2 kill
   ([result](frozen-transfer-results-v2.md)).

No further tuning of this architecture, recipe or bank is
authorized by the protocol. The 300 eval tasks are now development
data; any future construction claim needs a new sealed bank and a
new preregistered hypothesis.

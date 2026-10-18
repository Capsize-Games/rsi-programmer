# Frozen total-work comparison, 2026-10-18

The feature-statistics cache passed the rule frozen in `code-v67`: every
paired and repeated search outcome, returned graph, and shared execution
counter matched the immutable reference, and all three required learned
controls reduced pooled mean attempt CPU by at least 20% at cap 8,192.
This is a generic runtime improvement. The SNN has no demonstrated unique
advantage over the ANN controls in this experiment.

## Scope and provenance

The complete matrix contains 16,128 rows: 48 evaluation tasks, three seeds,
seven controls, four root-signature caps, two engines and two timing repeats.
The training set has 24 functions disjoint from the evaluation functions.
This reuses the known v2 corpus and its exact model weight hashes to compare
implementation cost. It does not provide another independent transfer test.
Seeds and timing repeats are not additional independent task functions.

The measured executable source is `code-v69`, commit
`eef8df202f56d936582bad36369164b13272dae7`. A data-only seed-zero checkpoint
was published as `code-v70` during the run. The manifest records that Git
revision change and unchanged source/input bytes. The prefix audit verifies
that all 5,376 published checkpoint rows match the completed raw prefix.
The code-v65 reference source and all v1/v2 inputs, scripts and result files
remain unchanged.

Protocol SHA-256:
`b5399f0ab7f90df5191a2b72c1dd3a17706bb4d64538e616e44512cbe76a1130`.
The first launch stopped before training/search on a malformed runner hash
literal. Its failure manifest/log are retained; `code-v69` corrected the
literal and added a real-input preflight regression before this full run.
No tasks, seeds, controls, caps or repeats were removed.

## Primary result

Times below are process CPU milliseconds per attempt, including pre/post
model hashes and independent observed/held/exact validation. Failed attempts
remain in the means. Setup/fit and report I/O are separate. Each mean pools
144 task-seed attempts repeated twice; success is identical between engines.
A root-signature cap is a sweep setting, not an equal total-work budget.

| Control | Reference ms | Cached ms | CPU reduction | Success at 8,192 |
| --- | ---: | ---: | ---: | ---: |
| Streaming | 16.344 | 16.662 | -1.95% | 41.67% |
| Heuristic | 218.617 | 134.803 | 38.34% | 62.50% |
| Seeded random | 297.685 | 224.360 | 24.63% | 46.53% |
| Untrained SNN | 295.477 | 217.222 | 26.48% | 45.14% |
| Trained SNN | 283.538 | 214.459 | 24.36% | 68.75% |
| MLP | 210.195 | 131.461 | 37.46% | 70.83% |
| Recurrent | 208.385 | 136.220 | 34.63% | 74.31% |

The streaming pooled reduction is negative (1.95% more CPU). Its two
repeat-specific ratios vary widely, so this does not establish a stable
streaming regression. Streaming never constructs the feature cache.

All repeat-specific optimized/reference attempt-CPU ratios at cap 8,192:

| Control | Repeat 0 | Repeat 1 | Pooled |
| --- | ---: | ---: | ---: |
| Streaming | 1.242444 | 0.812855 | 1.019462 |
| Heuristic | 0.602022 | 0.631588 | 0.616616 |
| Seeded random | 0.752717 | 0.754666 | 0.753683 |
| Untrained SNN | 0.733857 | 0.736470 | 0.735159 |
| Trained SNN | 0.745670 | 0.767451 | 0.756369 |
| MLP | 0.613077 | 0.638060 | 0.625422 |
| Recurrent | 0.624463 | 0.685236 | 0.653694 |

## Success thresholds and training cost

The frozen rule selects the first listed cap reaching each target. It
reports achieved coverage and overshoot without interpolation. These are
threshold comparisons, not exactly equal success rates. All controls first
reach 25% at cap 8,192. At 50%, streaming/random/untrained SNN require 32,768;
heuristic/trained SNN/MLP/recurrent require 8,192. Every control first reaches
75% at 32,768. The complete per-engine/per-repeat threshold table and work
vectors remain in `aggregate.json`.

The cached engine's 75% threshold points are:

| Control | Success | CPU ms | +fit/N=48 ms | +fit/N=4,800 ms |
| --- | ---: | ---: | ---: | ---: |
| Streaming | 89.58% | 44.208 | 44.208 | 44.208 |
| Heuristic | 95.83% | 148.147 | 148.147 | 148.147 |
| Seeded random | 93.06% | 242.096 | 242.096 | 242.096 |
| Untrained SNN | 87.50% | 234.882 | 234.882 | 234.882 |
| Trained SNN | 100.00% | 220.554 | 342.069 | 221.770 |
| MLP | 100.00% | 128.522 | 190.001 | 129.136 |
| Recurrent | 100.00% | 143.989 | 222.563 | 144.775 |

At the 100% target, trained SNN, MLP and recurrent reach all 144 task-seed
cases at cap 32,768. Streaming, heuristic, random and untrained SNN are
censored at that cap. Streaming's full-success CPU cost remains unknown.
MLP and recurrent reach the same 100% target at lower measured attempt and
training-amortized CPU cost than trained SNN. At the lower thresholds,
streaming costs less CPU at the listed points even though success coverage
differs. There is no measured SNN efficiency advantage here.

Common supervision generation cost 2.759960 CPU seconds. Each learned
control is charged that entire cost plus its own mean-seed construction and
fit: trained SNN 5.832681 seconds, MLP 2.950996, recurrent 3.771544. The
48/480/4,800-task amortization scenarios divide those charges by independent
task count, never by timing repeats or competing controls. Larger scenarios
extrapolate fixed measured evaluation costs; they were not additional task
runs. Control construction follows the frozen initialization order. Shared
process import/startup is recorded in the whole-run ledger without assigning
it to an individual control. Fixed heuristic/random/untrained construction
costs are recorded separately in `training.json` and are not amortized as
training.

## Work saved and work remaining

Across the 288 pooled attempts at cap 8,192, each ranked control still
materializes and scores 8,922,852 descriptors; streaming generates 1,678,230.
The cache preserves these counts and all candidate execution behavior.
For each ranked control, the reference performs 428,296,896 target-range
element reads and 428,296,896 operand-feature element reads, derived from
the pinned source. The cache records 6,912 target-range reads and 8,684,064
operand-statistics reads, with 361,836 misses and 17,924,208 cache hits.
Counterfactual `reference_*` fields are not executed work. Reference gap
scan counts are unavailable where the old engine did not expose them.
Counters are an explicit vector, not an arbitrary sum of different units.

Full-layer descriptor materialization, ordinal features, model scoring and
sorting remain. Structural changes need a separate frozen comparison and
may change search behavior. The next useful experiment is bounded candidate
selection against streaming and the cheaper ANN controls, with success and
all work charged. A larger streaming cap is needed to uncensor the 100%
endpoint before claiming a full-success efficiency win.
[Issue #81][remaining] tracks this work and remains open.

## Validation and timing boundaries

The independent stdlib auditor imports no search/training implementation.
Its final report checks all 16,128 rows and 8,064 engine pairs, exact graph
polynomials and observed/held rows, complete keys/order, shared counters,
repeat determinism, model/source hashes, aggregate costs, amortization and
the frozen optimization gate. It reports zero mismatches. Four temporary
corrupted bundles were rejected: duplicate key, forged CPU ratio, wrong
accepted graph, and uncharged training cost. Original bundle hashes stayed
unchanged. The first audit failure is retained: the auditor initially
misinterpreted task-seeded random hashes and the all-control ratio map;
its corrections and passing reports are recorded in the audit folder.

Affected verification on this executable source: 29 focused tests passed
in 0.100 seconds and `just test-fast` passed 328 tests in 222.008 seconds.
No local full/slow/retry suite was run: legacy search/author/lattice paths
were unchanged, and the new pre-evaluation API has benchmark callers.
Fast CI passed on Python 3.12 and 3.13 for `code-v69` (run 39136919234).
Full CI for that revision remained pending when this evidence was prepared.

The external wrapper measured child user+system CPU through CLI exit:
1,737.432975 seconds, including import, setup, fit, CLI-side provenance
hashes and all output I/O.
Its stage wall measurement was 1,737.267235 seconds and also includes the
observer's post-run provenance/hash work; it is not an exact CLI-only wall
measurement. The runner's internal timers recorded 1,735.369936 CPU and
1,736.163084 wall seconds through raw/summary output, excluding final
manifest serialization. Attempt/search CPU and wall boundaries are separate
and unaffected by the outer observer. No energy claim is made.

## Evidence

- `raw.jsonl`: every attempt, errors and work fields (zero row errors).
- `training.json`: supervision and exact fitted model hashes for all seeds.
- `aggregate.json`: every control/cap/repeat, thresholds and amortization.
- `manifest.json`: frozen inputs, runtime/dependency source and environment.
- `../benchmark-total-work-validation-20261018/`: commands, test logs,
  first-launch failure and complete external process ledger.
- `../benchmark-total-work-audit-20261018/`: independent auditor, reports,
  negative controls and checkpoint-prefix proof.

SHA-256 of `raw.jsonl`:
`bd134def172d853990227b5beaa42847b94ff497d45a6e888f61da593881e41e`.
SHA-256 of `training.json`:
`bd9978e24d2ac6e290e5bafb09ea7032b6a8793c0083a3235a82c248e2ba6676`.

[remaining]: https://github.com/Capsize-Games/rsi-programmer/issues/81

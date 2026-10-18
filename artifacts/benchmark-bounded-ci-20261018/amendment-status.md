# Code-v78 amendment validation record

Base publication: code-v77 commit `66b55b0` (annotated tag `code-v77`). The
code-v78 amendment applies the portable startup-test and peak metadata
changes after that publication.

The retained `portable-startup-test.patch` and
`nullable-peak-metadata.patch` files are historical candidates, not exact
representations of the final working-tree diff and not independently tested.
The startup candidate was amended with helper type hints and a valid one-epoch
training group after the first focused run exposed that zero epochs are
rejected before the expected-hash guard. The peak candidate was narrowed so
only absent `peak_live_descriptors` and `peak_live_features` become null;
pruning and scoring counters keep their existing zero semantics.

Amendment changes:

- `benchmark_bounded.attempt` now preserves absent peak measurements as null;
  it retains zero defaults for other extra counters.
- `benchmark_bounded_summary` now totals an axis only when all component
  attempts provide values. A bounded peak total sums per-attempt maxima; it is
  not a maximum across attempts.
- The startup regression checks the actual training-group call and exact
  frozen task split/caps, verifies 24 groups and 40 updates for each of three
  learned controls, locally disables neural golden-hash comparisons while
  retaining the original map assertion, and separately checks that a wrong
  expected hash is rejected with a valid one-epoch group.
- A metadata regression checks null peak fields for uninstrumented attempts,
  unchanged zero-valued streaming scoring/pruning counters, and null-preserving
  summary aggregation against numeric bounded peak totals.

Validation evidence is retained under
`artifacts/benchmark-bounded-validation-20261018/`. The first 11-test affected
run failed because the strict-guard regression supplied `epochs=0`; the log
and receipt are retained as `amendment-affected.log` and
`amendment-affected.json`. The first `just test-fast` run was interrupted after
the affected-run failure made it redundant; its partial output is retained in
`amendment-fast.log`, but the interrupted validation wrapper could not write a
receipt. The corrected rerun results follow.

## Validation results

The first attempted `tests.test_benchmark_bounded` run completed 11 tests in
7.632 seconds and failed only in the new wrong-hash guard regression: it used
zero epochs, which the helper rejects before reaching the hash check. That log
and receipt are preserved. The earlier fast run was stopped after 243 test
start lines; its log ends in `KeyboardInterrupt`. The observer was interrupted
before writing its receipt, so no exit code or timing receipt exists for that
partial run.

After changing the regression to one epoch with a valid two-label group, the
focused eight-module suite passed 53 tests. The validation receipt records
8.881 seconds CLI time (8.909 seconds including post-run source snapshot) and
`source_unchanged: true`. Then `just test-fast` passed 352 tests with 0 exit
code in 229.506 seconds CLI time (229.536 seconds including post-run source
snapshot), also with `source_unchanged: true`.

The focused and fast receipts have identical before/after SHA-256 values for
all three changed runtime/test files, so the verified source snapshots match:

- `rsi_programmer/benchmark_bounded.py` —
  `5b7aebf16176d06fcae95e1bd48faab3aaff63641f476b9ab48d588327c482b1`
- `rsi_programmer/benchmark_bounded_summary.py` —
  `e0c7fb66aea233ad5cf98d682d2a070bae7a892f1b515a8dfa57e48232860c99`
- `tests/test_benchmark_bounded.py` —
  `c6b54760585f39d468e597d9a19d50e57e6e80739e776019593b19d38e07ea65`

The changed runtime behavior is limited to reporting unavailable active-candidate peak
counters as null and preserving that unavailability in aggregate metadata. The
startup test now validates real setup inputs and fit work while avoiding a
portable test's cross-platform neural weight-byte assumption; a separate
regression still exercises the strict incorrect-hash rejection path. Search,
fit, frozen inputs, protocol, and published benchmark data were not changed.

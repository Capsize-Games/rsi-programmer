# Bounded experiment validation

The latest startup amendment and its 51 focused/350 fast passing tests are
recorded in [startup-fix.md](startup-fix.md). The earlier results below remain
as historical evidence.

The focused search/feature/benchmark contracts passed 50 tests in 0.647 s.
`just test-fast` then passed 349 tests in 219.946 s (CLI wall 223.272 s).
Both commands returned zero and all captured runtime/test/fixture/dependency
hashes were unchanged during each run. Exact commands, CPU/wall boundaries,
source digests and full logs are in `focused.*` and `fast.*`.

Four long dictionary literals were wrapped after the fast run. The two files
retain identical ASTs excluding source positions; `format-proof.json` records
the before/after hashes and equivalence assertions. No search, test or
aggregation behavior changed. The final focused run passed all 50 tests
again in 0.642 s, with unchanged source hashes. Its receipt and full output
are `final-focused.json` and `final-focused.log`.

The observer was made fully typed and now hashes its own source, rejects
existing receipt/log names, and distinguishes exact child CLI wall time from
the enclosing stage's post-run provenance work. The version used for the
initial focused/fast runs is retained as `test-observer-original.txt`.
The final focused receipt and subsequent measurement use `run_validation.py`.
No observer or memory-instrumented timing enters the benchmark cost curves.
Whole child CPU includes waited forked memory processes and final CLI I/O;
parent-only process CPU is reported separately in the benchmark manifest.

The new experimental engine covers fixed-window/quota pruning, exact feature
coordinates, local policy order, generation-order retention, cap boundary
handling, oracle rejection, invalid policy permutations and lazy streaming
delegation. Benchmark contracts cover real frozen input preflight, the
asymmetric complete grid, counterbalanced schedule, censoring, full setup
charges, output preservation and isolated native/traced memory children,
including a neural policy. Cap-one functional tests use a known corpus task;
these are contract checks, not benchmark outcomes or parameter selection.

Legacy `pre_eval.py`, `pre_eval_features.py`, search/author/lattice code and
all earlier frozen fixtures/scripts/results remain unchanged. Local slow,
retry and full suites were not run: the new API is experimental and used by
this benchmark; the legacy lanes have no new caller. No full-suite success
is claimed. Bounded performance results and independent positive/corrupted
bundle audits are pending at this implementation publication stage.

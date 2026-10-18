# Independent result audit

`result-audit.json` passes with no errors: 4,032 unique matrix cells, 1,743
returned graphs, 1,743 independently accepted rows, zero invalid graphs,
4,032 work-invariant checks, and 341 current source files matching the run
manifest. Exact graph functions, expanded tree bounds and execution on all
observed and held rows are checked without production execution helpers.
Training groups, deterministic sampling, features, counters, model freeze,
summaries and descriptive paired bootstrap are independently reconstructed.

The first audit attempt failed because of mistakes in the auditor. Its full
report and log are preserved as `result-audit-first-failure.*`. The 1,240
messages comprise:

- 72 training-field mismatches: stale selected-vector work assumptions and
  missing left/right size fields in the reconstructed records;
- one totals accumulator that overwrote rather than added group values;
- 576 comparisons of intentionally different streaming state sentinels;
- 576 heuristic step-count checks incorrectly assuming zero instead of one;
- six assumptions that streaming/random fit-work schemas match learned
  controls;
- six checks treating absent special-control state-change flags as false;
- three heuristic metadata checks making the same zero-step assumption.

Direct source and schema inspection corrected those assumptions. No runtime
source, frozen input, raw result, training report or aggregate was changed to
make the auditor pass. The first execution metadata file also retains its
original malformed literal newline suffix; the final execution metadata is
valid JSON. The successful audit took 2.907 seconds under the builder lock.

`selfcheck.py` then created five separate temporary copies of the result
bundle. In each case it updated the corresponding manifest output hash so
that rejection required a semantic or consistency check. It tested a duplicate
matrix cell, a wrong valid graph marked accepted, an altered training feature,
an altered aggregate, and internally consistent forged model-state hashes.
All five were rejected for the expected reason. `auditor-selfcheck.json`
records the errors and unchanged original bundle hash; the checks took
16.081 seconds under the same lock.

Reproduce the final checks from the repository root:

```sh
flock /tmp/rsi-programmer-builder.lock python3 \
  artifacts/benchmark-pre-eval-v2-audit-20261018/result_audit.py \
  artifacts/benchmark-pre-eval-v2-20261018
flock /tmp/rsi-programmer-builder.lock python3 \
  artifacts/benchmark-pre-eval-v2-audit-20261018/selfcheck.py
```

The result auditor compares the current runtime sources with the recorded
measurement sources. To reproduce that check after later code changes, use
the measured revision or this results tag. It does not require model fitting
or another benchmark run.

Local affected checks passed: nine focused tests and all 311 fast-lane tests
(221.000 seconds). On the measured runner's CI run `39132899183`, the fast
jobs on Python 3.12 and 3.13 passed. Both full jobs were still in progress at
publication. Slow/retry and full discovery were not run locally for the
additive benchmark change; this report does not claim a green full CI cycle.

# Counterbalanced diagnostic validation

The source freeze is `23a2ad6` / `code-v94`. This checkpoint adds the independent
replay auditor and its tests before running the comparison.

- Affected command: `.venv/bin/python -m unittest
  tests.test_counterbalanced_protocol tests.test_online_diagnostics
  tests.test_counterbalanced_audit tests.test_online_spiking
  tests.test_online_graph tests.test_online_corpus tests.test_online_audit
  tests.test_online_benchmark -v`: 46 passed in 0.189 seconds.
- `just test-fast`: 420 passed in 234.598 seconds; no skips or exclusions.
  Source hashes were unchanged; receipt and complete log are retained here.
- Commands ran serially under `/tmp/rsi-programmer-builder.lock`.
- The initial affected-test failure is retained. Its synthetic fixture set
  streaming CPU above persistent CPU while expecting the deployment gate to
  fail. Correcting streaming's fixture CPU to be cheaper resolved the
  incorrect expectation. The production gate was not relaxed. A control now
  also covers an ANN seed winning while its pooled margin loses.
- Slow/retry and local full discovery were not rerun for this additive
  diagnostic path. No existing runtime kernel or frozen benchmark changed.

Measurement, full raw replay and corruption controls are pending at this
validation checkpoint. No persistent-learning improvement is claimed.

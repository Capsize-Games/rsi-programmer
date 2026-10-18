# Observed-context adapter validation

The new frozen comparison has not been measured at this checkpoint.

- 61 affected tests passed in 0.210 seconds.
- All 435 fast tests passed in 237.242 seconds. Source hashes were unchanged;
  no tests were skipped or excluded. The receipt binds all runtime, script,
  test, fixture and workflow inputs.
- An earlier 434-test run passed before CPU-scope wording and opaque-ID
  contract clarification. Its full log and receipt are retained separately.
  The final added test verifies that episode IDs do not affect learning.
- The initial affected-test failure is retained: a synthetic audit row lacked
  the newly required diagnostic CPU field. Adding that field corrected the
  fixture without relaxing the validator.
- Checks ran serially under `/tmp/rsi-programmer-builder.lock`. Slow/retry and
  local full discovery were not rerun for this additive path. Real hub checks
  and every existing test remain intact.

The adapter's current observed examples change the learner information set.
Its fixed context projection and the existing readout/update parameters are
frozen before measuring. No performance, sensor or chip claim is made here.

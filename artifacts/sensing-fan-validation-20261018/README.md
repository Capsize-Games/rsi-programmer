# Sensing verification evidence

The prospective freeze is code-v105. Implementation `643be25` was verified
before measurement and published with verification evidence as code-v106
(`de87983`). The measured runtime, tests, protocol and workflow have not
changed since code-v106. Code-v107 publishes results, charts and evidence.
See [result and limitations](../../docs/sensing-fan-result.md).

## Final checks

- `affected-canonical-snapshot-final-receipt.json`: 56 tests in 2.546 s.
- `fast-auditor-final-receipt.json`: 521 tests in 235.269 s.
- `frozen-benchmark-receipt.json`: one 924-stream run, 277,200 windows.
- `independent-audit.json` and `independent-replay-receipt.json`: complete
  independent replay, wrapper wall 10.126 s.
- `corruption-checks.json`: eight semantic corruptions rejected, with file
  hashes rebound on disposable copies. The original bundle is unchanged.
- `descriptive-contrasts.json`: post-run comparisons and work counts from
  retained outputs, without training, model execution or threshold tuning.
- `ci-runtime-snapshot.json` and `ci-fast-evidence.json`: implementation
  source `643be25`, both 521-test fast jobs passed; both full jobs remain
  running in the snapshot. Completed-job API logs are compressed exactly.
- `publication-checks.json`: exact source/artifact/receipt bindings.

Test, model, benchmark and audit commands used the builder lock. Receipts
record the command, source-file matrix, exit code and log digest. `.log.gz`
files decompress to the exact log bytes named by `log_sha256`; no failure
text was trimmed. CI receipts additionally identify job/run/source hashes.
The initial failed attempts and intermediate passes are historical evidence,
not validation of a later source revision. The final receipts above bind the
published measured source. No tests were excluded from either final command.
No local slow/retry/full discovery ran for this additive path. Real hub
checks remain in the unchanged CI workflow. The CI heartbeat remains paused.

## Read-only reproduction

Replay all retained forecasts, parameter and boundary hashes, calibration,
alarms, metrics and gates at this source revision:

```sh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  -m scripts.audit_sensing --bundle artifacts/sensing-fan-v1-20261018
```

The two post-run helpers read the completed bundle; neither imports runtime
models or fits parameters. The plot uses an isolated Matplotlib environment
without changing the project's dependency pins. Choose fresh destinations:

```sh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  artifacts/sensing-fan-validation-20261018/describe_results.py \
  --bundle artifacts/sensing-fan-v1-20261018 --output /tmp/sensing-contrasts.json
flock /tmp/rsi-programmer-builder.lock uv run --isolated --no-project \
  --with matplotlib --python .venv/bin/python \
  artifacts/sensing-fan-validation-20261018/plot_results.py \
  --bundle artifacts/sensing-fan-v1-20261018 --output /tmp/sensing-plots
```

The plot JSON retains exact seed values, displayed transformations, summary,
manifest, data and helper-source hashes. Bars show pooled rates or mean CPU
per seed; whiskers are observed seed ranges, not confidence intervals.
All repeated clips and seeds are correlated. No learned parameter arrays,
databases, physical transition labels, hardware measurements or chip-energy
claims are included.

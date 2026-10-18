# Bounded benchmark audit

`bounded_audit.py` is a standard-library-only auditor for the frozen bounded
candidate benchmark. It reads a completed bundle with `raw.jsonl`,
`memory.jsonl`, `training.json`, `aggregate.json`, and `manifest.json`.

The auditor checks the pinned protocol, corpus, model reference, cached
baseline, feature helper, runtime snapshots, and full timed and memory grids.
It independently validates returned graph polynomials and observed/held rows,
checks reported validation flags and acceptance, compares deterministic
behavior within each engine across repeats, verifies memory rows against the
seed-zero timing rows, and recomputes work totals, matched-success costs,
training amortization, and decision fields. A semantically incorrect graph is
a rejected attempt when the validation flags and `accepted` value correctly
report that failure. It is an audit error when the row accepts it or reports
different validation flags.

For bounded rows, the auditor checks the fixed window and quota metadata,
active descriptor and feature peaks, generated/scored/skipped/executed layer
totals, the root-signature cap, and scan amplification. Frozen source hashes
and the implementation tests provide the evidence for the within-window
selection rule; the result rows do not contain per-descriptor order traces.
The memory pass is checked separately from the timing data. It reports RSS
high-water and traced Python bytes without treating either as performance
timing.

Run the result audit after the complete bundle is available:

```sh
python artifacts/benchmark-bounded-audit-20261018/bounded_audit.py \
  artifacts/benchmark-bounded-v1-<run>/ \
  --output artifacts/benchmark-bounded-audit-20261018/result-audit.json
```

`auditor-preflight.json` records the frozen protocol hash and reconciled matrix
sizes. `auditor-validation.json` records syntax, formatting, and numeric
comparison checks. `negative_controls.py` tests rejection of duplicate raw
keys, reordered rows, a paired ratio changed by 0.25, an incorrect accepted
graph, and a forged training charge. It makes temporary copies and refreshes
bundle file hashes after mutation, so the auditor must reject the corrupted
content. Run it after the completed bundle exists:

```sh
python artifacts/benchmark-bounded-audit-20261018/negative_controls.py \
  artifacts/benchmark-bounded-v1-<run>/ \
  --output artifacts/benchmark-bounded-audit-20261018/negative-controls.json
```

No complete benchmark bundle is available at this checkpoint, so the positive
result audit and the four bundle-level negative controls remain pending.

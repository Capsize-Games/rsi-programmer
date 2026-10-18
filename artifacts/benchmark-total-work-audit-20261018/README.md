# Total-work audit artifact

This folder contains the static freeze audit and an independent result auditor
for the protocol pinned at
[the total-work protocol][protocol].
The benchmark was run under the frozen protocol; the raw results are in
`../benchmark-total-work-v1-verified-20261018/`. See `freeze-audit.json` for
the hash and row-count checks completed before measurement.

The result auditor reads one completed bundle containing `raw.jsonl`,
`manifest.json`, `aggregate.json`, and `training.json`. It checks the full
two-engine, two-repeat matrix, hashed per-cell engine order, every returned
graph's exact polynomial and observed/held rows, paired and repeated behavior,
root-signature and shared-work counts, frozen model checkpoints, source hashes,
the aggregate CPU-ratio summary, and the optimization gate. It independently
reports matched success costs and training amortization. It emits a JSON report
at the requested output path and never imports search runtime code or performs
training or measurement.

The runner's internal whole-run timer covers setup through raw and aggregate
output, but excludes final manifest serialization. The external measurement
record at
[measurement.json][measurement]
reports 1,737.267 seconds of stage wall time, which includes the CLI and the
observer's post-run source hashing. The observer reads that wall value before
writing its measurement record. Its 1,737.433 seconds of child CPU comes from
`RUSAGE_CHILDREN` around the CLI subprocess and waited
descendants through exit, including final output I/O; it excludes CPU used by
the observer. These are different scopes. The runner's internal whole-run
timer excludes final manifest serialization.

Two assumptions in the first positive audit were corrected after comparison
with the frozen runner source. The runner creates a base seeded-random
checkpoint for each training seed, then reconstructs the evaluation control
with a task-derived seed (`_random_seed(seed, task_id)`). Its task-specific
weight hash therefore need not equal the shared training checkpoint hash; the
runner marks this reference comparison unavailable for `seeded_random`, while
the audit still checks each attempt's frozen state and paired/repeated search
behavior. The aggregate publishes pooled 8,192-budget CPU ratios for all seven
controls, although only trained SNN, MLP, and recurrent controls determine the
optimization gate. The auditor verifies the full ratio map and applies the
gate to those three required controls.

The initial audit launcher failure and first failed audit report are preserved
in `positive-audit.log`, `positive-audit-run2.log`, and
`positive-audit-report.json`. Corrected successful runs are recorded in
`positive-audit-report-v2.json` and `positive-audit-report-v3.json`, with
commands and timings in `positive-audit-execution.json`.

Four tampered bundle copies were used as negative controls; only their audit
reports and execution receipt remain in `negative-controls-verified-20261018/`.
The temporary raw copies were removed. The controls verified rejection of:

- a duplicate raw row key;
- a forged published CPU ratio after refreshing the aggregate hash;
- a validly shaped but wrong polynomial on a row marked accepted;
- an amortization entry that removes the learned-control training charge.

[protocol]: ../../fixtures/benchmark_total_work_v1.json
[measurement]: ../benchmark-total-work-validation-20261018/measurement.json

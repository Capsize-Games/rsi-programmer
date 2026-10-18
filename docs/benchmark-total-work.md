# Total ranking work

Issue [#80](https://github.com/Capsize-Games/rsi-programmer/issues/80) compares
the original pre-evaluation engine with cached operand feature statistics.
The protocol was published in `code-v67` before timing. It reuses v2's known
function-disjoint corpus and exact published model weights to test runtime
cost. It is not another independent generalization experiment.

The cache computes target range once per ranked search, stores each retained
operand's hit and in-range counts, and reuses those counts for candidate
features and the policy gap. It preserves all twelve feature values and
their original full-layer ordinal denominator. It does not change candidate
generation, full-layer policy ranking, execution caps, layer retention or
independent acceptance. Streaming, terminal shortcuts and zero execution
caps do not build the feature cache.

Both engines run under the same wrapper. The reference is a byte-for-byte
copy of `code-v65`'s pre-evaluation implementation. The matrix has 16,128
attempts: 48 tasks, three seeds, seven controls, four root-signature caps,
two engines and two timing repeats. Each engine pair runs adjacently in a
hash-determined order; the second repeat reverses that order. Torch uses one
intra-op thread, under the single-builder lock.

The runner records process CPU and wall time for search and for each attempt,
including model hash checks and post-search observed/held/exact verification.
Attempt timing excludes report I/O. Its internal run timer covers setup,
training, raw and summary writes but excludes final manifest serialization.
The external validation wrapper additionally records child process CPU and
wall time through CLI exit, including that final I/O. These timing boundaries
are kept separate in the evidence.

Counts remain a work vector: descriptors, root signatures, feature vectors,
operand reads, policy batches and ticks, row operations, verification and
cache scans. The original engine's feature scan counts follow from its
pinned source; missing gap counts are unavailable, not zero. Counterfactual
`reference_*` fields in the cached engine describe the original algorithm
and are not counted as executed work. Counts do not measure every machine
instruction, allocation or FLOP; CPU/wall time includes work absent from
individual counters. No energy estimate is made.

Success thresholds are 25%, 50%, 75% and 100%. For each control and engine,
the report uses the first listed cap reaching a threshold, shows the actual
coverage and overshoot, includes failures in its mean cost, and records
censoring if the threshold is unreached. It does not interpolate between
caps. The candidate cap is a sweep setting, not an equal total-work budget.

Training cost is recorded once per seed/control and kept separate from
evaluation. The 48/480/4800-task accounting scenarios charge each learned
control the full common supervision setup and its mean construction/fit
cost. They do not divide training cost among competing controls. Larger
scenario counts extrapolate the same fixed evaluation costs; they are not
measurements on additional independent tasks. Construction costs reflect
the fixed initialization order; shared process startup/import cost is
reported in the whole run and is not allocated among the controls.

The optimization rule requires exact behavior across both engines and
repeats and at least a 20% pooled paired mean attempt-CPU reduction at cap
8,192 for trained SNN, MLP and recurrent separately. All controls, caps and
repeat-specific ratios remain in the report even if that rule fails. A
generic runtime optimization does not establish an SNN advantage.

Run after the affected verification passes:

```sh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  scripts/benchmark_total_work.py --output /tmp/rsi-total-work-new-run
```

Use a new output directory. The runner rejects changed protocol, corpus,
reference implementation and model-reference bytes. Raw attempts and
training checkpoints are retained on failure. Existing v1/v2 benchmark
inputs, sources and results remain unchanged.

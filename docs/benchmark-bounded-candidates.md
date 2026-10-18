# Bounded candidate experiment

Issue #81 measures the costs that remain after the feature-statistics cache.
The protocol and immutable cached baseline were published in code-v72 before
the performance comparison. Known v2 functions and exact fitted weights are
reused for a cost experiment; this is not a new independent transfer test.

`bounded_pre_evaluation_search` lazily visits grammar-order windows of up to
256 descriptors, builds their twelve features, ranks them, and executes at
most the top 64 and the remaining signature budget. Stable policy ties use
generation order. An accepted candidate stops immediately. A full block can
be scored even when fewer than 64 root evaluations remain, so signature
caps do not bound total scoring work. Global ordinals and width denominators
refer to the complete current layer derived from retained prior operands.

Unselected candidates are permanently pruned. Only selected, executed
candidates can enter the next operand layer, and they are deduplicated and
retained in original generation order. A cap-interrupted layer is discarded.
Later widths, reachability, output graphs and success may change. The fixed
256/64 engineering limits were not selected using bounded evaluation
outcomes or timings; there is no evaluation-time parameter adjustment.
The historical default remains unchanged. This API is experimental.

The timed matrix contains 10,656 observations across all 48 evaluation
functions, three training seeds and two repeats. Full cached and bounded
engines compare heuristic, trained SNN, MLP and recurrent controls at root
caps 2,048, 8,192, 32,768 and 131,072. Streaming is a single control/engine at
those caps plus 524,288. Adjacent ranked engine pairs use a hash-derived
order, reversed on the second repeat. All cells and failures are retained.
Repeats and seeds do not create additional independent task functions.

Attempt CPU/wall covers pre/post state/weight hashes, search and independent
observed/held/exact polynomial validation. Search-only timers enclose the
engine call. Report I/O is outside attempt cost but included in whole-run
accounting. Setup and fit are charged in full to each learned standalone
control at N=48/480/4,800; the larger counts are accounting scenarios, not
measurements on more functions. Existing exact model hash checks also
constrain the unchanged training recipe and initialization order.

Memory is measured separately from timing, at seed zero/cap 32,768, in a
fresh forked child per task/control/engine. There are 432 untraced RSS cells
and 432 separate tracemalloc cells. RSS includes inherited process/model
pages and allocator behavior; both absolute highwater and peak-minus-initial
KiB are retained. Traced peaks describe Python allocations and omit some
native Torch storage. RSS from the traced pass includes tracing overhead
and is not the untraced memory measure. Every memory observation must match
its timed counterpart's graph/behavior and work. No memory-pass timing
enters the CPU curves. Active block counters exclude retained and pending
operand expressions, whose allocations remain in the memory measurements.

The report separates descriptors, scored feature vectors, executed root
signatures, quota-discarded descriptors, all scored-but-unexecuted candidates,
operand/row work and memory. Scan amplification is scored descriptors divided
by executed signatures; unlike units are never added into a scalar total.
Matched success targets are 25/50/75/100% with first listed cap, achieved
coverage, overshoot, failures included and explicit censoring. There is no
interpolation between caps or inference of a missing full-success endpoint.

The frozen cost rule requires every learned bounded control to reach 100%
and reduce N=48 total CPU by at least 20% against its full cached control at
each engine's first full-success cap. Results remain published if it fails.
The separate SNN rule requires the bounded SNN to reach 100% and cost at
least 10% less than bounded MLP, bounded recurrent and streaming, all at
observed full-success points with the full N=48 training charge. Failure or
censoring establishes no efficiency advantage. No energy claim is made.

Run only after affected verification, using a new output directory:

```sh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  scripts/benchmark_bounded_candidates.py --check-only
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  scripts/benchmark_bounded_candidates.py --output /tmp/rsi-bounded-new-run
```

The runner pins corpus/protocol/reference/helper/model bytes, hashes source
and dependencies before/after, rejects errors and incomplete grids, and
retains raw rows and training checkpoints on failure. Independent auditing
and corrupted negative controls are required before accepting the result.

## Completed comparison

The [complete results](../artifacts/benchmark-bounded-v1-verified-20261018/README.md)
pass the independent audit of all 10,656 timed rows and 864 memory rows, plus
five corrupted negative controls. The fixed 256/64 shortlist fails its cost
rule: bounded coverage stops at 45.139-47.917%, with natural exhaustion of
the pruned grammar. Streaming reaches 100% at cap 131,072 and 41.859 ms mean
attempt CPU, compared with 137.695 ms for full-cached MLP and 223.515 ms for
full-cached SNN before training charges. Memory reductions at lower coverage
do not establish a same-success efficiency improvement. Keep the generic
cache and the historical default; test a complete fallback before promoting
any future shortlist. This supplies no CPU case for SNN expansion.

## Re-auditing the measured source

The auditor checks source files relative to its own repository location,
not the shell's working directory. After a source or test amendment, export
code-v76 to a fresh directory and copy the published auditor into the same
artifact subdirectory there. Code-v76 runtime and tests are identical to the
measured code-v74. Pass the immutable full bundle by absolute path:

```sh
repo=/absolute/path/to/rsi-programmer
frozen=$(mktemp -d)
git -C "$repo" archive code-v76 | tar -x -C "$frozen"
cp "$repo/artifacts/benchmark-bounded-audit-20261018/bounded_audit.py" \
  "$frozen/artifacts/benchmark-bounded-audit-20261018/bounded_audit.py"
flock /tmp/rsi-programmer-builder.lock python3 \
  "$frozen/artifacts/benchmark-bounded-audit-20261018/bounded_audit.py" \
  "$repo/artifacts/benchmark-bounded-v1-verified-20261018" \
  --output /tmp/rsi-bounded-reaudit.json
```

Use a new output path to retain earlier evidence. The auditor uses only the
standard library. Captured dependency paths must still exist with their
recorded bytes. Do not disable source checks or point the measured-source
audit at a checkout with later runtime/test changes. The code-v77 report's
short instruction to supply an absolute auditor path is incomplete unless
the auditor is copied into the frozen export as shown here.

## Post-measurement test and metadata amendment

The portable startup regression still builds the actual 24 training groups,
passes the original frozen hash map to its test spy, and performs all 40
optimizer updates for SNN, MLP and recurrent controls. It checks learned
weights change and untrained weights stay fixed. Within that test only,
golden neural hash checks are disabled and training is checked through its
actual optimizer updates and before/after weight hashes. A separate one-epoch
regression verifies the production helper rejects a deliberately wrong hash.
The production CLI still enforces the frozen model hashes.

Future bundles report absent non-bounded `peak_live_descriptors` and
`peak_live_features` as null. Summary totals preserve null when any component
is unavailable. Known zero scoring/pruning counters stay zero. Bounded peak
work totals sum per-attempt maxima; they are not the maximum across attempts.
The auditor supports numeric and null totals and requires bounded peaks to
remain nonnegative measured integers within the frozen window. The original
auditor and receipt are retained separately. No frozen raw rows, training
records or aggregates were rewritten, and no performance rerun was needed:
these metadata assignments occur after the timed attempt returns.

Final local verification passed 53 affected tests (7.835 s unittest time;
8.881 s CLI) and 352 fast tests (226.174 s unittest time; 229.506 s CLI),
with matching source snapshots across both runs. The initial 11-test run
failed because its new guard test requested zero epochs; the redundant fast
attempt was interrupted. Both are retained with the corrected successful
runs in `artifacts/benchmark-bounded-validation-20261018`. Full/slow/retry
suites were not run locally; these amendments change benchmark reporting
and tests, with the search/training implementation unchanged.

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

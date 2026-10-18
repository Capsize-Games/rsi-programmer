# Complete-fallback shortlist benchmark

This experiment compares three execution paths over the frozen v2 arithmetic
corpus: historical streaming, full-layer cached ranking, and a cheap single-row
shortlist with complete fallback. The shortlist changes execution order. It does
not permanently prune candidates.

For each grammar-order block, the engine computes the absolute residual on
observed row zero and uses stable original-ordinal ties to select a shortlist.
The four policy-ranked controls build the unchanged twelve-feature vectors only
for selected candidates. Cheap-only mode performs no full feature assembly or
policy inference. The selected candidates execute first, followed by every
remaining candidate in grammar order, exactly once. Search stops on an
accepted graph or a declared execution cap. A complete layer is
retained in canonical generation order; an interrupted layer is discarded.
Heap and feature counts are bounded by the selected parameters, while retained
and pending layers are additional memory. Total search memory is not bounded
by the heap size.

The training-only selection recipe was frozen before it ran. It used seed zero,
all 24 training functions, two repeats, root cap 131,072, and the five shortlist
controls plus streaming. Both preregistered settings (64/4 and 256/16) had
100% independently validated training coverage for every control. The 256/16
setting had the lower mean attempt CPU across the five shortlist controls
(75.189 ms versus 112.204 ms), so the final comparison uses window 256,
shortlist size 16, and observed row zero. This is engineering parameter
selection on the known training split; it does not use evaluation functions or
establish independent generalization.

The final protocol is [benchmark_complete_shortlist_v1.json](../fixtures/benchmark_complete_shortlist_v1.json).
It pins the selection manifest, raw selection rows, training output, and the
external command receipt. That receipt is the authoritative selection setup
charge because it includes imports and final artifact writes. Its full CPU and
wall totals are charged once per standalone shortlist method at N=48, 480, and
4,800. Normal production model group generation, initialization, fitting, and
setup are charged separately. Streaming and full-cached baselines receive no
selection charge.

The evaluation matrix contains 11,808 timed rows and 960 separate memory rows.
It runs three model seeds over all 48 evaluation tasks and two repeats. Streaming
uses caps 2,048, 8,192, 32,768, 131,072, and 524,288; ranked configurations use
the first four caps. Each policy is paired across full-cached and complete-
shortlist engines, and cheap-only has its own configuration. Search uses only
the twelve observed rows. Returned graphs are checked after search against
observed rows, held rows, and exact integer polynomials. Memory instrumentation
does not supply timing data.

The overall cost decision requires all five new configurations (four ranked
controls and cheap-only) to reach full coverage. At least one must also have
N=48 selection- and production-amortized CPU at or below 90% of streaming at
streaming's first full-success cap. A per-configuration improvement requires
that configuration to reach full coverage and meet the same 90% threshold.
SNN efficiency requires SNN and every comparator to reach full coverage, with
SNN CPU at or below 90% of each comparator. These rules describe this frozen
corpus and environment; they do not establish energy savings or broad transfer.
CPU work counters can overlap and must not be added as exclusive cost
categories.

No final evaluation result is stated here until the complete frozen matrix,
independent audit, and manifest are available.

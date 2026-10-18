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

The completed matrix contains all 11,808 timed rows and 960 memory rows,
with zero row errors. The independent auditor recomputed graph acceptance,
work accounting, ordered matrix completeness, repeat/model/source checks and
all cost summaries. It passed; all six corrupted negative controls were
rejected. Two incorrect auditor assumptions about selected-versus-executed
counts and model-free controls were corrected, with the failed report retained.
The measured bundle and runtime were not changed for those corrections.

All ten configurations reach full validated coverage. Their first listed
full-success caps and mean per-task CPU costs are below. "Shortlist" means
the complete-fallback engine. "Full" means full-layer cached ranking.
N=48 and N=4,800 include the frozen selection and production charges.

| Engine/control | Cap | Attempt ms | N=48 ms | N=4,800 ms |
| --- | ---: | ---: | ---: | ---: |
| Streaming | 131,072 | 38.994 | 38.994 | 38.994 |
| Cheap-only | 131,072 | 44.001 | 1,227.184 | 55.833 |
| Full heuristic | 131,072 | 158.358 | 158.358 | 158.358 |
| Shortlist heuristic | 131,072 | 55.704 | 1,238.886 | 67.536 |
| Full SNN | 32,768 | 231.652 | 350.415 | 232.839 |
| Shortlist SNN | 131,072 | 105.211 | 1,407.157 | 118.231 |
| Full MLP | 32,768 | 137.853 | 197.807 | 138.452 |
| Shortlist MLP | 131,072 | 57.578 | 1,300.714 | 70.009 |
| Full recurrent | 32,768 | 147.697 | 224.232 | 148.463 |
| Shortlist recurrent | 131,072 | 63.451 | 1,323.168 | 76.048 |

The shortlist restores coverage and reduces CPU relative to full-layer
ranking, but every shortlist configuration is slower than streaming even
before upfront costs. Both preregistered cost gates fail. These results
support keeping streaming as the default and retaining this API as an
experimental comparison; they do not support expanding the SNN architecture.
The allocation/native-memory observations are at cap 32,768, where shortlist
coverage can be lower, so they do not establish a memory win at full success.

The known-v2 corpus and three model seeds remain the experiment's scope.
This is not an independent transfer result or an energy measurement. Final
local verification passed 75 affected and 374 fast tests; Python 3.12 and
3.13 CI each passed 374 fast tests on code-v82. Full/slow/retry suites were
not run locally, and the retained CI snapshot shows full CI still running.

Evidence is in `artifacts/benchmark-complete-shortlist-v1-20261018/`,
`artifacts/benchmark-complete-shortlist-audit-20261018/` and
`artifacts/benchmark-shortlist-validation-20261018/`. Earlier code-v83 and
code-v84 snapshots match the complete memory file and seed-zero raw prefix
byte-for-byte.

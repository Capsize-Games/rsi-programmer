# Bounded shortlist freeze

Issue #81 follows the feature-cache comparison in code-v71. This protocol
freezes one experimental method: score grammar-order windows of 256 and
execute up to the top 64 per window. Those parameters are fixed engineering
bounds, selected without bounded evaluation outcomes or timing. Full-layer
feature ordinals and width denominators refer to the bounded search's own
retained operands. Unselected candidates are pruned, so later layer widths,
reachability and success can change. No exact-equivalence claim is made.

The known v2 corpus and exact fitted weights are reused for implementation
cost measurement. This is not another independent transfer experiment.
The profile in `../benchmark-ranking-profile-20261018/` is descriptive;
its instrumented CPU/memory data do not supply performance comparisons.

The immutable cached code-v71 baseline is copied verbatim and pinned, with
its feature-statistics helper pinned too. All earlier frozen input, source
script and result bundles stay unchanged. The new protocol declares 10,656
timed cells, plus 432 isolated RSS and 432 separate tracemalloc cells.
Streaming is a single engine with caps through 131,072 and 524,288. Four
ranked controls (heuristic, trained SNN, MLP, recurrent) compare full cached
and bounded at four caps through 131,072. Every evaluation task, seed and
repeat remains in the matrix. Unreached success targets remain censored.

The cost rule requires each learned control to reach 100% success and reduce
N=48 training-amortized CPU by at least 20% against its full cached control.
All outcomes, costs, scan amplification, memory and censoring are reported
even if the rule fails. The original default search stays in place; this is
an experimental API. No architecture expansion is implied by a memory or
runtime reduction. Full setup/fit charges and all verification work remain
explicit. RSS contains inherited process/model pages; traced Python peaks
exclude some native allocations and come from separate instrumented runs.

`freeze-audit.json` records exact protocol/reference hashes and independently
counted matrix dimensions before bounded performance evaluation. The runner
and affected tests are in progress at this publication stage. They must
pass, followed by an independent result audit and corrupted controls, before
any performance conclusion. No bounded evaluation run has started.

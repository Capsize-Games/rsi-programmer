# Bounded candidate comparison results

The frozen 256-descriptor / 64-execution shortlist fails its declared cost
rule. It reduces memory but permanently prunes expressions needed by many
tasks. Extended streaming reaches 100% validated success and uses less CPU
than every full-layer ranked control on this workload. Neither the bounded
cost rule nor the separate SNN efficiency rule passes. The historical default
search is unchanged.

## Scope and freeze

The protocol was published in code-v72 before this comparison:
`fixtures/benchmark_bounded_candidates_v1.json`, SHA256
`c235994867676059311cc05ac876060da0aa1517a6e3fa53cd73b80fabd3472d`.
The executable is code-v74 (`785e8b0d006624333fe9446e16d2e78a81697d47`).
Source, fixtures, fitted model weights, dependencies and evaluation state
remained unchanged throughout the run. Data-only checkpoint publication
changed Git metadata through code-v76; both snapshots are retained.

This reuses the known v2 corpus: 24 training and 48 evaluation functions with
globally disjoint train/evaluation polynomials, three training seeds, two
timing repeats, and all predeclared cap points. It is a cost experiment on
known functions, not a new independent generalization test. There are 48
independent evaluation functions; repeated seeds and timings do not add
independent functions. No parameter was retuned after bounded evaluation.

The complete bundle has 10,656 timed rows and 864 separate memory rows.
All grid, source, environment, model-state, root-cap, active-block, repeated
behavior and memory-to-timing checks pass. All failures are retained.

## Full-success costs

Each row uses the first listed cap that reaches 100% across pooled attempts.
CPU includes pre/post hashes, search and independent observed/held/exact
validation. Training-amortized CPU charges each learned standalone control
its full setup and mean-seed model construction/fit cost. N=480 and N=4,800
are accounting scenarios, not more measured functions.

| Engine / control | First 100% cap | Attempt CPU ms | N=48 CPU ms | N=4,800 CPU ms |
| --- | ---: | ---: | ---: | ---: |
| streaming / streaming | 131,072 | 41.859 | 41.859 | 41.859 |
| full_cached / heuristic | 131,072 | 143.000 | 143.000 | 143.000 |
| full_cached / trained_snn | 32,768 | 223.515 | 340.335 | 224.683 |
| full_cached / mlp | 32,768 | 137.695 | 195.293 | 138.271 |
| full_cached / recurrent | 32,768 | 144.013 | 218.263 | 144.755 |

No bounded control reaches even 50% at any declared cap. Its full-success
cost is unobserved and is not inferred. At the maximum ranked cap:

| Bounded control | Accepted attempts | Success % | Attempt CPU ms | Unsolved termination |
| --- | ---: | ---: | ---: | --- |
| heuristic | 138/288 | 47.917 | 42.381 | no-solution-within-bound |
| trained_snn | 132/288 | 45.833 | 97.394 | no-solution-within-bound |
| mlp | 132/288 | 45.833 | 37.085 | no-solution-within-bound |
| recurrent | 130/288 | 45.139 | 49.492 | no-solution-within-bound |

All maximum-cap bounded failures exhaust their pruned grammar within tree
bound 5. Maximum executed roots are below 8,192 for each bounded control;
larger root caps cannot recover the expressions this fixed shortlist drops.
The raw threshold status `censored` means the target was not observed in the
declared sweep; these failures are natural exhaustion, not root-cap stops.

Streaming reaches 89.583% at cap 32,768, then 100% at 131,072 (41.859 CPU
ms), and remains at 100% at 524,288 (44.632 ms). The first listed full-success
point is used even though timing varies between caps. At that endpoint its
two repeat means are 38.655 and 45.063 ms. Full-cached SNN means are 227.620
and 219.411 ms; MLP means are 130.309 and 145.081 ms. These are local CPU
measurements, without confidence intervals or energy measurements.

At full success, mean executed root signatures per attempt are 13,384.583
for streaming, 6,780.632 for SNN, 7,131.653 for MLP and 6,484.118 for
recurrent. Every full-layer ranked control generates/scores 30,982.125
descriptors per attempt. Guidance reduces executed roots but its feature and
ranking overhead exceeds that saving in CPU. Signature caps do not measure
total work. All work axes and per-repeat costs are in `aggregate.json`.

## Memory and remaining profile costs

Memory uses seed 0 / cap 32,768, 48 tasks per configuration, with separate
fresh-child native RSS and tracemalloc passes. Median values are MiB:

| Configuration | Native peak RSS | RSS peak minus initial | Traced Python peak |
| --- | ---: | ---: | ---: |
| streaming / streaming | 233.023 | 3.152 | 3.870 |
| full_cached / heuristic | 263.646 | 34.568 | 27.795 |
| bounded / heuristic | 240.129 | 10.441 | 1.564 |
| full_cached / trained_snn | 300.105 | 71.201 | 27.809 |
| bounded / trained_snn | 245.197 | 15.523 | 1.415 |
| full_cached / mlp | 268.766 | 39.668 | 27.796 |
| bounded / mlp | 242.758 | 13.043 | 1.321 |
| full_cached / recurrent | 276.775 | 47.691 | 27.795 |
| bounded / recurrent | 243.396 | 13.891 | 1.633 |

The bounded active descriptor and feature blocks peak at 256, excluding
retained/pending operand expressions. Those allocations remain included in
the memory passes. Full-cached/streaming `peak_live_*` work fields are legacy
zero defaults for unavailable block instrumentation, not measurements of zero
memory. Use the separate RSS and traced values for cross-engine comparison.
RSS includes inherited model/process pages and allocator behavior. Traced
Python peaks omit some native Torch storage; traced-pass RSS is not used as
the native measurement. Lower bounded memory accompanies lower coverage.

The separate [full-layer profile](../benchmark-ranking-profile-20261018/README.md)
retains 240 cProfile files and 240 traced allocation cells. Across 48 tasks,
inclusive SNN ranking is 6.102 s and feature encoding 3.437 s; MLP ranking is
1.537 s and encoding 3.368 s. These cumulative categories overlap and must
not be added. Profiling timings are descriptive, separate from this
uninstrumented CPU comparison.

## Verification and retained failures

The [independent audit](../benchmark-bounded-audit-20261018/final-audit-v2.json)
checks every row, graph, reported work counter, schedule, charge and aggregate.
It reports zero errors, zero repeat mismatches and zero memory mismatches.
Raw rows do not record per-descriptor order traces; source hashes and the
implementation tests support the within-window selection rule.
All five corrupted controls are rejected: duplicate key, reordered rows,
paired ratio increased by 0.25, incorrect accepted graph, forged fit charge.
The [receipt](../benchmark-bounded-audit-20261018/audit-receipt.json) records
exact commands, exit codes, durations, logs and hashes.

The first auditor attempt rejected two incorrect provenance assumptions:
the nested snapshot uses the older total-work protocol hash, and data-only
checkpoint publication changes Git metadata. The correction checks that
nested hash against its actual reference and independently preserves all
runtime/fixture/test/dependency hash checks. The first failure is retained
as `final-audit.json`; auditor amendments and their timing are recorded.
A negative-harness error-substring mismatch is also retained in the receipt.
No measurement rows or summary values were changed to make the audit pass.

The earlier benchmark launch failed at training-group startup before any
fit/evaluation/memory work. It is retained separately in
`../benchmark-bounded-v1-20261018/failure.json`; code-v74 fixed the keyword
call and added a real startup integration test. Local verification then
passed 51 affected tests in 7.713 s and 350 fast tests in 226.790 s.
Full/slow/retry suites were not run locally for this benchmark-only API.

The code-v74 CI fast job failed its startup golden fitted-weight hash check
on a different Python/OS environment; another fast matrix job was cancelled.
The precise numerical cause is unproven. A test portability amendment is
pending; the strict production benchmark hash guard remains required.
This bundle is validated for its recorded local environment.

Whole CLI elapsed time is 1,382.187 s; user+system CPU for the CLI and waited
children is 1,382.959 s. This includes imports, setup, fitting, hashes, output
I/O and separate memory children. Internal parent CPU through summary is
879.981 s and memory-child CPU is 500.894 s. Final manifest serialization
is excluded from internal timers and included by the external observer.
See `../benchmark-bounded-validation-20261018/measurement-verified.json`.

Native-memory and two-seed checkpoints published as code-v75/code-v76
are byte-exact prefixes of this complete result. The proof is retained in
`../benchmark-bounded-audit-20261018/checkpoint-prefix-proof.json`.

The strict source audit must run from code-v76 (runtime/tests identical to
measured code-v74) when later source or test amendments exist. Supply this
bundle and the published auditor by absolute path to that checkout. Do not
disable source checks to audit a different executable.

## Direction

Retain streaming as the measured cost baseline. Keep the generic feature
cache; reject the permanently pruning 256/64 method as a default. A future
cheap shortlist with a complete streaming fallback must preserve expression
reachability and beat this uncensored streaming endpoint on total CPU before
promotion. This result supplies no CPU case for SNN architecture expansion.

The [CPU curves](../benchmark-bounded-plots-20261018/bounded-cost-curves.png)
show all declared cap points and full N=48 fit charges. Connected lines do
not estimate unmeasured intermediate points.

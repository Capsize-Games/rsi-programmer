# Frozen complete-shortlist comparison

This bundle completes the protocol published in code-v81, SHA256
`189fa147a11040c6a05ef173522a6146285de616996724adff49d17b5f66be0a`.
The runtime was verified and published in code-v82 before evaluation.

All 11,808 timed rows and 960 separate memory rows are retained. The manifest
reports a complete unique grid, matching memory/timed behavior, identical
repeat behavior, frozen model/source/environment snapshots, valid root caps
and shortlist work counters, and zero row errors. The outer command observer
also reports exit zero and unchanged source bytes. Its receipt and log are in
`../benchmark-shortlist-validation-20261018/measurement.*`.
The complete CLI took 1,589.295 seconds wall and 1,590.508 seconds CPU,
including memory children, imports, setup, fitting, summary and output. Those
benchmark execution totals are distinct from per-method cost curves.

The published 960-row memory snapshot and 3,936-row seed-zero timing prefix
match the final files byte-for-byte. Their earlier tagged evidence is in
`../benchmark-complete-shortlist-progress-20261018/`.

At code-v85 publication the independent audit was pending. It has now
passed, independently checking the full matrix, accepted graphs, counters,
model/source snapshots and all aggregate costs. All six corrupted negative
controls were rejected. Both the total-CPU gate and SNN-efficiency gate fail,
although all five new configurations reach full validated coverage. The
retained failed audit and its two corrected checker assumptions are documented
in `../benchmark-complete-shortlist-audit-20261018/README.md`; benchmark
bytes were unchanged. No search default changed. Detailed cost endpoints are
in `../../docs/benchmark-complete-shortlist.md` and the post-run report.

The corpus contains 24 training and 48 evaluation functions with no shared
whole polynomial function. These are the known v2 arithmetic fixtures and
three model seeds, not a new independent transfer test. Full selection CLI
costs are charged once to every standalone new method; normal production
supervision/setup and fitting are charged separately for learned controls.
Instrumented memory timings do not enter CPU cost curves. Memory observations
are at root cap 32,768 and must retain their coverage rather than implying
same-success memory comparisons.

Validation of this unchanged runtime: 75 local affected tests and 374 local
fast tests passed. Python 3.12 and 3.13 fast CI each passed all 374 tests on
code-v82. Full/slow/retry were not run locally; the retained CI snapshot shows
both full CI jobs still running.

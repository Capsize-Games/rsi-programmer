# Frozen Python and CURLEE transfer results

The trained route passes **11/18 previously unseen functions** in both
Python and CURLEE. The untrained route also passes 11/18; classical search
passes 14/18. This trial establishes transfer within the supplied grammar,
but does not establish a benefit from training or a cost advantage.

The [protocol](frozen-transfer-protocol.md), task bank and solver sources
were published before any controller ran. All three controllers finished
their choices before held execution. The original sealed arithmetic final
set remains unopened. No repairs were made from these outcomes.

## Actual target-language execution

Each candidate runs on 20 independently held arrays, including empty and
singleton inputs, length-256 boundaries and fourteen new random arrays.
Python uses the existing isolated JSON-file CLI. CURLEE uses the existing
guarded compiler and bounded processing interface. Acceptance requires all
20 cases in both languages; compilation alone does not count.

| Fixed controller | Emitted | Python accepted | CURLEE accepted | Actual cases per language |
| --- | ---: | ---: | ---: | ---: |
| Trained | 14/18 | 11/18 | 11/18 | 280 |
| Classical | 16/18 | 14/18 | 14/18 | 320 |
| Untrained | 13/18 | 11/18 | 11/18 | 260 |

All 860 Python cases and 860 CURLEE cases executed. Every emitted candidate
received all 20 cases, including incorrect candidates. Python and CURLEE
correctness agree on every recorded case. Each CURLEE campaign also
rejects a deliberately incorrect expected result. The compiler hash stays
`412d0e6ee5b8fb9e380322683863849523e1a5797e7e7dbd42bf174b138555ac`.

The [complete evidence](
../../artifacts/frozen-branch-transfer-v1-20261101/summary.json) includes
generated source, decisions, visible queries, incorrect outputs, source
hashes, compiler records and command costs. No missing candidate is counted
as an executed program.

## Costs

Both trained models were fitted afresh and reproduced their published
parameter fingerprints. Their combined fitting CPU cost was 42.406 seconds,
including supervision generation, initialization and fingerprinting. All
fitting is charged over the actual eighteen tasks.

| Controller | Search-command CPU, including fitting | Python-check CPU | CURLEE-check CPU | Total CPU | Total wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trained | 55.833 s | 3.947 s | 11.834 s | 71.614 s | 87.727 s |
| Classical | 2.884 s | 4.956 s | 13.287 s | 21.127 s | 33.817 s |
| Untrained | 11.863 s | 3.877 s | 10.860 s | 26.600 s | 36.575 s |

Whole-command CPU includes the accounting process and completed child
descendants, including isolation and resource-guard processes. The search
worker is pinned to one CPU. Each task receives one CPU second for search
and visible feedback, with a declared 0.05-second bounded-step tolerance.
No task exceeds that tolerance. These are single observations on a shared
machine, not repeated timing estimates.

The largest single-process child RSS is 510,964 KiB for trained execution,
206,524 KiB for classical and 224,004 KiB for untrained. These are not
aggregate simultaneous memory measurements. The classical search worker
itself uses 18,592 KiB and does not import Torch; its Python execution
checker imports additional shared modules.

The shared qualified library already exists. Its historical creation cost,
benchmark creation, regression testing, CI, analysis and publication are
excluded. This is a controller comparison, not an end-to-end cost claim
from an empty library. The hypothetical volume-1,000 scenario in the
summary assumes unchanged non-fit cost per request: 1.665 CPU seconds for
trained, 1.174 for classical and 1.478 for untrained. It supplies no extra
task or quality evidence and still shows no trained cost advantage.

## Failures and scope

The trained route constructs no candidate for `transfer_002`, `007`, `016`
or `017`, and emits incorrect candidates for `004`, `014` and `015`.
Classical constructs none for `016` or `017`, and emits incorrect programs
for `013` and `015`. Untrained constructs none for `007`, `008`, `010`,
`015` or `016`, and emits incorrect programs for `004` and `014`.
All these attempts remain in the reports.

Trained and classical solve different functions: trained passes `013`,
which classical misses; classical passes four that trained misses. Trained
and untrained each have two successes the other misses. The equal total
does not mean identical behavior, and these individual differences do not
establish an aggregate training benefit.

The eighteen functions were generated from the existing two-branch grammar,
two per donor group. They differ from all 2,647 curriculum functions, the
earlier curriculum and eleven development functions on the declared
24 probes. This is bounded functional separation, not a universal
inequivalence proof or a collection of eighteen independently selected
user workloads. For example, the accepted [first Python program](
../../artifacts/frozen-branch-transfer-v1-20261101/neural/transfer_000/python/clean_readings.py)
computes positive-value sum plus input length. The earlier eleven
development utilities remain the evidence for selected useful tasks.

The host mines and pairs qualified branches, supplies the loop and initial
accumulator, enumerates update variants and emits source, contracts and CLI
plumbing. The neural model fills an integer-expression hole. This does not
establish arbitrary Python/CURLEE source generation, learned abstraction
discovery, strings, records, general libraries or repository repair.

The programmer can transfer bounded executable components. Its trained
search remains the weaker part of this comparison. Further development
should address a specific programming failure and use a new frozen bank
for any subsequent final claim. Reusing this bank for diagnosis makes that
work development evidence. Increasing model capacity or claiming recursive
self-improvement is not supported by this result.

## Validation status

The preflight checkpoint passes 44 focused tests, including task separation,
source seals, lazy classical imports, timeout handling and malformed CLI
output. The full programmer fast-track regression passes **541 tests in
262.267 seconds**. Its [validation receipt](
../../artifacts/frozen-branch-transfer-validation-20261101/receipt.json)
and compressed log record the exact scope. Other local tracks and slow/retry
lanes were not rerun; remote aggregate and full CI remain separate pending
checks. The earlier execution receipt preserves its publication-time
regression status.

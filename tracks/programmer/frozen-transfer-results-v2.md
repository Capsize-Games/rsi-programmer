# Frozen branch-transfer result v2 (issue #91)

Serialized run 2026-11-05 of the sealed 54-function bank in
[frozen-transfer-protocol-v2.md](frozen-transfer-protocol-v2.md).
Artifact: `artifacts/frozen-branch-transfer-v2/` (`summary.json`,
`cost-ledger.json`, per-arm reports, `run91.log`, driver and
summary assembler kept beside them). All seven arms finished
search before any held evaluation; every arm's fixed choices were
then executed for real in isolated Python and under the guarded
CURLEE checker (compiler sha `412d0e6e...`, stable before/after,
wrong-expected control rejected in every arm) on the same 20 held
cases per task. Commands ran one at a time under the builder lock
on a loaded shared machine; costs are process CPU seconds,
single observations.

## Result

| Arm | Both backends | Total command CPU (s) |
| --- | ---: | ---: |
| classical | 38/54 | 54.7 |
| neural seed 0 | 30/54 | 108.7 |
| neural seed 1 | 28/54 | 108.3 |
| neural seed 2 | 27/54 | 108.4 |
| untrained seed 0 | 28/54 | 63.7 |
| untrained seed 1 | 30/54 | 65.3 |
| untrained seed 2 | 27/54 | 63.9 |

Python and CURLEE acceptance agree per arm in every case.
Actual executed cases: classical 800 per language, neural
640-720, untrained 660-760 (accepted programs run all 20; wrong
candidates stop early, and every count here is actual, not
assumed). Neural fresh fits cost 41.0-41.1 wall seconds per seed
inside the search command. Total measured experiment cost across
arms: 573.0 CPU seconds (search incl. fit 358.6, CURLEE 156.7,
Python evaluation 57.7). Donor-library creation from empty is a
declared exclusion with the ~50.1 s mixed-unit lower bound from
issue #92; adding it would only widen the trained arm's deficit,
since the library exists to serve the composition routes and the
classical arm uses the same qualified branches — the exclusion is
shared, and the trained-specific fit cost (~41 s/seed) is fully
charged above.

## Decision rule outcome

The kill criterion is met. Trained does not beat untrained
(means identical at 28.3/54; per-seed wins split both ways) and
does not beat classical on solves at matched total cost
(classical solves 9-11 more tasks with at most 1 task going the
other way per seed; McNemar p = 0.001-0.022 across seeds), while
costing 2.0x classical and 1.7x untrained in total measured CPU.
Per the frozen rule, **the neural arm leaves this route
permanently; the route continues classical-only.**

At n=54 the bank resolves its arms, unlike v1: classical's lead
is consistent across all three trained seeds, and trained vs
untrained is a tie in mean with wide per-task churn (symmetric
difference 8-14 tasks per seed) — training changes *which* tasks
are solved without changing *how many*.

## What this establishes

- The classical composition route transfers to a sealed,
  separation-strengthened bank at 38/54 (70%) with real
  dual-backend execution. It is a working verified synthesizer
  for this grammar family (one loop, two qualified branches,
  shared Int argument, host-supplied lowering/CLI/contracts).
- Learned guidance inside that route adds cost and nothing else
  at any seed. As in v1, and now with seeds and power, training
  benefit is not merely unestablished — the trained arm is
  indistinguishable from its untrained initialization.
- Nothing here shows general programming, learned abstraction
  discovery, or recursive self-improvement, and solver changes
  from here make this bank development evidence requiring a new
  sealed bank for any future final claim.

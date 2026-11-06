# Frozen branch-transfer evaluation v2

This evaluation freezes 54 new functions before any solver work: six
from each of the nine supplied donor groups, in the same two-branch
grammar family as v1 (identical branch library, `pattern` /
`instantiate` composition and shared-Int argument grammar). It is the
powered follow-up to the v1 bank (18 functions), built because v1
could not statistically separate its arms. The original sealed
arithmetic final set remains unopened.

## Bank and separation

The [task bank](../../fixtures/frozen_branch_transfer_v2/tasks.json),
[held inputs](../../fixtures/frozen_branch_transfer_v2/held.json),
[protocol](../../fixtures/frozen_branch_transfer_v2/protocol.json),
[hashes](../../fixtures/frozen_branch_transfer_v2/sha256.json),
[seal](../../fixtures/frozen_branch_transfer_v2/seal.json) and
[cost accounting](../../fixtures/frozen_branch_transfer_v2/cost-accounting.json)
are frozen. The worker loads visible examples and reference programs
for its task-engine callback. It never loads the held-input file or
passes reference code to a solver. All arms finish their choices
before held evaluation starts. Inputs supplied as feedback do not
overlap earlier held inputs. References are admitted only when they
execute on all declared dataset rows; this is not a universal
totality proof.

Separation is stronger than v1's 24 probes. Every v2 function differs
from all 2,647 curriculum programs, all eleven development oracle
outputs and all 18 v1 references on an extended signature suite: the
24 canonical probes, plus v1's 20 held inputs, plus 48 newly seeded
arrays (lengths 0-256, values -100-100), deduplicated to 90 probes.
Engine errors record `"ERR"`, so defined-vs-error behavior also
separates. No v2 program repeats any v1 reference or any of the
eleven development programs token-for-token. Seeded visible and held
inputs are fresh relative to v1; only the six boundary arrays
(empty, singletons, length-256 extremes) recur, by convention.

## Arms

Classical is the primary arm: composition-first branch pairing with
fragment fallback, deterministic, one run. The trained arm is the
ablation: the existing neural route fitted afresh on seeds 0, 1 and
2 with the original recipe. Seed 0 must reproduce the published
parameter fingerprint; seeds 1-2 record theirs with no prior value
to match. The untrained arm instantiates the same architecture
without fitting on seeds 0, 1 and 2. Solver changes after opening
results turn this bank into development evidence and require a new
final evaluation set.

## Budgets and execution

Every arm receives identical budgets: one CPU second per task for
search and visible-query feedback, with a declared 0.05-second
bounded-step overrun tolerance, and at most three added examples
from the fixed 24-input visible pool. Wall-clock limits derive from
the remaining CPU allowance. Over-budget choices are void.

Every arm's fixed choices are executed for real: isolated Python
execution and the existing guarded CURLEE checker on the same 20
held rows per task (six recurring boundary arrays and fourteen
newly seeded random arrays), plus a deliberately incorrect expected
result that the CURLEE campaign must reject. Incorrect, timed-out
and unconstructible programs remain failures. A process error or
malformed output stops that program's remaining cases; accepted
programs require all rows in both backends. All actual execution
counts are reported.

## Cost

Cost is end-to-end from an empty donor library: library creation,
fresh fits, search, visible oracle calls, emission, backend
execution and candidate evaluation, in CPU seconds. The
[cost-accounting file](../../fixtures/frozen_branch_transfer_v2/cost-accounting.json)
lists each stage with its command, metric and status. Integration
note (2026-11-05): the ledger now exists (`rsi_programmer/cost_ledger.py`,
`CostLedger` with `measure`/`record`/`run`/`declare_excluded`/
`write_json`/`from_legacy`, nine fixed phases). The runners do not
import it; stage costs are captured by external command timing and
converted post-run, with each stage's `ledger_phases` mapping
recorded in the cost-accounting file. Donor-library creation
remains a declared exclusion with a mixed-unit ~50.1s lower bound
until a ledger-instrumented rerun measures it (issue #92).
No hypothetical amortization volume counts as evidence.

## Decision rule

Kill criterion: if the trained arm does not beat both untrained and
classical on solves at matched total cost (or match solves at at
least 10% lower total cost), the neural arm leaves this route
permanently and it continues classical-only.

## Running it (serialized phase)

Fits and searches are timed and run serialized on a quiet machine,
one arm at a time. From the repo root:

```
PYTHONPATH=. python3 scripts/run_frozen_branch_transfer_v2.py \
  --method classical --output artifacts/frozen-branch-transfer-v2
PYTHONPATH=. python3 scripts/run_frozen_branch_transfer_v2.py \
  --method neural --seed 0 --output artifacts/frozen-branch-transfer-v2
PYTHONPATH=. python3 scripts/run_frozen_branch_transfer_v2.py \
  --method neural --seed 1 --output artifacts/frozen-branch-transfer-v2
PYTHONPATH=. python3 scripts/run_frozen_branch_transfer_v2.py \
  --method neural --seed 2 --output artifacts/frozen-branch-transfer-v2
PYTHONPATH=. python3 scripts/run_frozen_branch_transfer_v2.py \
  --method untrained --seed 0 --output artifacts/frozen-branch-transfer-v2
PYTHONPATH=. python3 scripts/run_frozen_branch_transfer_v2.py \
  --method untrained --seed 1 --output artifacts/frozen-branch-transfer-v2
PYTHONPATH=. python3 scripts/run_frozen_branch_transfer_v2.py \
  --method untrained --seed 2 --output artifacts/frozen-branch-transfer-v2
```

Then evaluate each arm (held inputs open only now):

```
PYTHONPATH=. python3 scripts/evaluate_frozen_branch_transfer_v2.py \
  --method classical --output artifacts/frozen-branch-transfer-v2
PYTHONPATH=. python3 scripts/evaluate_frozen_branch_transfer_v2.py \
  --method neural --seed 0 --output artifacts/frozen-branch-transfer-v2
... (repeat for neural seeds 1-2 and untrained seeds 0-2)
PYTHONPATH=. python3 scripts/check_frozen_branch_curlee_v2.py \
  --output artifacts/frozen-branch-transfer-v2/<arm-path>
... (repeat per arm path)
```

Bank-generation checks (smoke scale, no solvers) run anytime:

```
PYTHONPATH=. python3 -m unittest tests.test_frozen_branch_transfer_v2 -v
```

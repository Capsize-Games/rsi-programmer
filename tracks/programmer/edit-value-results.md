# Executable-edit value experiment

The learned edit-value route passes 1/6 fresh functions, matching the
untrained, classical and incumbent routes. Every route still solves 0/3
original tasks. No training benefit or broader-family transfer is measured.

The frozen array models can express all three failed development tasks.
For the supplied witnesses, the input-only model loses the correct prefix
at depths 5, 6 and 5 with beam 32; the execution-context model loses it at
10, 5 and 5. Beam 256 also loses all three supplied paths. Some losses are
from local token cutoffs; others arise from cumulative beam ranking.
The checkpoint fingerprints match the saved models. No model was refitted
for this audit. The reports are in
`artifacts/array-construction-diagnosis-20261102`.

This identifies pressure from sequence scoring and search truncation. Losing
one supplied witness does not prove all equivalent correct programs were
lost, and a small absolute sequence probability alone is not failure proof.
The controls are diagnostic labels, never training examples or donors.

The alternative scores executable token edits with a small ANN. Training
labels come from actual completed descendant executions: successful teacher
suffixes and sampled successful or failed continuations. Unfinished sampled
rollouts are recorded as censored, rather than false failure labels. The
scorer sees the visible goal, current execution and syntax/action context;
future execution outcomes are labels only. This changes the training
objective from next-token imitation to sampled continuation value. It is
not an optimal-value estimate or recursive self-improvement.

The first recipe samples up to 64 training functions, up to 32 transitions
per function and 2,048 transitions total, with a 120-CPU-second data/training
cap. It uses two sampled rollouts with at most 32 steps and 30 fit epochs.
The learned, untrained and classical execution-heuristic searches each have
5,000-action and three-CPU-second caps. The frozen execution-context
ArrayPolicy is a separate baseline using its original beam 256/top-eight
search and three-second wall cap; these budgets are not identical work
measures and cannot establish a cost advantage.

Six fresh functions are frozen in `fixtures/edit_value_v1` before the first
solver run. They differ from all 512 original teacher functions and the
three development functions on 24 declared probes. This is bounded function
separation, not universal inequivalence. Five visible examples and twenty
held cases are separate. Controller choices freeze before held cases are
parsed. Actual generated Python and guarded CURLEE execution determine
acceptance; visible-example fit is insufficient.

All tasks use the existing bounded integer-list language. Operation and
structure labels do not establish distinct language families. The three
original tasks are baseline rechecks, not retention successes: the previous
array learner solved none. The new ranker cannot establish retention of the
older scalar controller, whose unchanged historical evidence remains
separate. Strings, records, general function calls and repository repair
remain unsupported.

Seventeen initial preflight tests passed in 1.327 seconds. The original
arithmetic final set remains sealed, and memory guards and backend isolation
remain unchanged.

The first launch stopped after fitting because the harness referenced the
removed linear-model weight field. No choices or held outcomes existed.
The failure and its cost are retained. Before the first completed evaluation,
the adapter was corrected, checkpoints moved under ignored `weights/`, and
training transitions balanced across early/middle/late depths. The task bank
remains unchanged. Tests now use temporary model and compiler fixtures,
so private local artifacts are not required by CI.

## Completed result

The new ranker passes **1/6 fresh functions**, matching the untrained value
network, classical execution heuristic and frozen ArrayPolicy incumbent.
Every method still solves **0/3 original development tasks**. The new
objective establishes no solve-rate improvement or training benefit in
this single-seed bounded trial. It does not open a new language family.

| Method | Fresh accepted in both languages | Original three rechecks |
| --- | ---: | ---: |
| Learned edit value | 1/6 | 0/3 |
| Untrained edit value | 1/6 | 0/3 |
| Classical execution heuristic | 1/6 | 0/3 |
| Frozen execution-context ArrayPolicy | 1/6 | 0/3 |

The learned, untrained and classical searches construct the same
nonpositive-value filter. The incumbent constructs the positive-reading
transform that adds the current output index. It also emits an incorrect
filter selected on visible examples. Thus the equal totals cover different
functions; no learned ranker dominates the incumbent. A retrospective union
would not be evidence of a trained-model gain or a preregistered portfolio.

There are **89 actual Python cases and 89 CURLEE cases**. Each accepted
program passes all twenty cases in both languages. The incumbent's wrong
filter stops after its ninth case, the first incorrect output; its remaining
eleven cases are excluded. All cross-backend per-case verdicts agree, all
recorded source hashes match, and every qualified program rejects a wrong
expected-output control. Missing candidates remain construction failures,
with no execution claimed. See the [evidence audit](
../../artifacts/edit-value-v2-20261102/evidence-audit.json).

Training uses 64 functions and 1,988 edit transitions, of which 1,987 receive
observed labels. It records 1,105 completed successes, 3,329 completed
failures and twelve censored rollouts. Fitting takes 2.866 CPU seconds.
The complete comparison, including all controls and target executions, uses
38.969 CPU seconds and 470.395 wall seconds. The separate failed launch
costs 4.559 CPU seconds. This is not a per-method cost advantage estimate;
the incumbent's historical fitting and untimed diagnostic runs are excluded.
Agent/development time, preflight, regression testing, CI and publication are
also excluded. The checkpoint and full success/failure traces remain
available for audit. Test success covers the declared held values and
capacities, not extreme-int64 behavior or universal compiler qualification.

The [receipt](../../artifacts/edit-value-v2-20261102/receipt.json) records
all exclusions, source revision and cost scope. Nineteen affected tests pass
in 1.651 seconds. The latest full programmer fast attempt also stopped at
reserve status 75 with no completion summary. Other local lanes were not
rerun; these resource-aborted runs are not passing checks. Both remote fast
checks for the measured source revision pass; full CI remains pending.

The teacher-coverage diagnostic was also corrected: sixteen of 512 teachers
contain genuinely nested conditions; 351 contain multiple conditions,
including sequential branches. The capped-reading witness combines three
conditions, two writes and nesting, a whole structure absent from these
teachers. Broad presence of individual tokens did not establish complete
structural coverage. Witness-path pruning remains a diagnostic observation,
not proof that every equivalent solution was pruned.

This trial does not justify further parameter or beam sweeps as evidence
of progress toward general programming. Keep the verified execution and
experiment machinery. Do not promote the new learner as an improvement or
reuse these outcomes for a new final claim. The result concerns this recipe
and task set, not the impossibility of learned programming in general.

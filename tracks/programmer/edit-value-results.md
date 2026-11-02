# Executable-edit value experiment

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

Seventeen focused preflight tests pass in 1.327 seconds. The bounded
measurement is pending resource admission at this checkpoint. Memory guards,
backend isolation and the sealed original arithmetic set remain unchanged.

The first launch stopped after fitting because the harness referenced the
removed linear-model weight field. No choices or held outcomes existed.
The failure and its cost are retained. Before the first completed evaluation,
the adapter was corrected, checkpoints moved under ignored `weights/`, and
training transitions balanced across early/middle/late depths. The task bank
remains unchanged. Tests now use temporary model and compiler fixtures,
so private local artifacts are not required by CI.

# Online graph-search pilot protocol

This protocol defines a small synthetic task stream for testing whether
episode-to-episode online updates help a controller choose among typed graph
search lanes after observing executed-candidate outcomes. It does not test real
sensors, Raspberry Pi hardware, chip energy, or transfer to general code.

## Frozen corpus

The generator is `rsi_programmer/online_corpus.py`; the frozen corpus is
`fixtures/online_graph_v1.json`, generated with corpus seed `83074`. It
contains 48 tasks in three consecutive phases of 16 tasks. The root-operator
quotas are:

| Phase | add | sub | mul |
| --- | ---: | ---: | ---: |
| 0 | 12 | 2 | 2 |
| 1 | 2 | 12 | 2 |
| 2 | 2 | 2 | 12 |

The candidate grammar has 432 two-operation integer expression trees: two
binary shapes, all permutations of `x`, `y`, and one constant from `(2, 3, 4,
5)`, and all ordered inner/root choices from `add`, `sub`, and `mul`. Each root
lane contains 144 candidates. The symbolic generator expands each candidate to
an exact integer polynomial and retains target functions that have exactly one
root lane across the complete grammar. The unique-root pools contain 16 add,
36 sub, and 40 mul polynomials. The corpus uses all 16 add targets and a fixed
seeded sample of 16 targets from each other pool, so no exact target function
repeats.

The add pool is exhausted by this 48-task corpus. That limits the range of add
targets represented here. The first grammar positions of the unique-root
targets range from 1 to 120 for add, 2 to 141 for sub, and 1 to 125 for mul.
The primary per-task budget is 144 candidate executions; the secondary budget
is 432. Thus every target is reachable by exhausting its correct lane at the
primary budget, while a controller that spends work in other lanes may still
be censored. Candidate positions are grammar facts, not measured policy
results.

The corpus uses one shared set of six observed `(x, y)` inputs and 32 held
inputs drawn independently of target labels from integer pairs in `[-9, 9]`.
The two sets are disjoint, and the six observed rows distinguish each target
from every distinct polynomial in the candidate grammar. Expected outputs,
exact polynomial coefficients, root labels, and witness expressions are
present in the corpus for the runner and evaluator. The policy must receive
only the current episode's six-value state and available-lane mask; it must
never receive the corpus task object, future task data, root label, phase,
target expression, held rows, or held outputs. The fixture SHA-256 is
`a4f9af2dfc0d8c4539d700b032dcc223c82aa272f02497902024d9f371096140`.

Generate the fixture with
`.venv/bin/python scripts/generate_online_corpus.py --output
fixtures/online_graph_v1.json`.

## Episode interaction and controls

The policy begins each episode with reset transient state. Persistent weights
may carry between episodes only for the persistent-update arm. At each step,
the controller chooses an available root lane. The environment takes that
lane's next candidate in grammar order and executes it on all six observed
rows. After execution, it returns reward `1 - e`, where `e` is the mean across
rows of `min(1, abs(prediction - target) / (1 + abs(target)))`. No target
signature is computed before execution. The policy may update only after this
feedback. An exact observed-row match ends the episode. Otherwise, the episode
ends at its candidate budget or when all lanes are exhausted.

An independent evaluator then checks the candidate against held rows and
compares its exact polynomial with the answer key. The held result is recorded
outside the policy and is never returned to it. A failed held check does not
restart the episode or generate a learner update.

The frozen comparison has five arms: persistent online SNN, frozen SNN with
the same seeded initial weights, SNN updates reset at each episode boundary,
an online nonspiking policy, and fixed grammar streaming. All arms use the
same corpus, task order, candidate budgets, and typed graph interpreter. The
controller configuration is frozen at six input features, 16 hidden units,
six SNN steps, learning rate `0.05`, eligibility decay `0.0`, reward baseline
`0.05`, membrane leak beta `0.8`, threshold `1`, weight clip `5`, and update
clip `0.05`. The seeded random projection is not trained. Credit applies only
to the immediately preceding decision; there is no cross-action eligibility
trace. Model seeds are `0`, `1`, and `2`. The same frozen 48-task corpus is
used for each seed, so these are three model initializations on one task
stream, not three independent corpora. The two budget conditions use fresh
policies and are not tuned against their outcomes.

The 1,440 planned episode rows are 48 tasks × 3 model seeds × 5 arms × 2
budgets. The predeclared budgets are 144 and 432 candidate executions per
episode. At the primary budget, the correct root lane alone can reach every
target; spending executions in other lanes can still exhaust the cap before a
late target.

## Measurements and limits

Record first-lane choice before feedback on each task, held-valid acceptance,
candidate executions to a held-valid result, censored tasks, interpreter row
operations, policy decisions, online updates, process CPU time, wall time, and
whole-run peak RSS, and serialized model size. The sequential run does not
support per-arm peak-memory claims. Include failed tasks in every aggregate.
CPU time is a host measurement and is not a proxy for chip energy or
Raspberry Pi performance.

At the 144 budget, persistent SNN first-lane accuracy must exceed both frozen
and episode-reset SNN by at least 10 percentage points pooled and have a
positive margin for each model seed; its accepted-task count must be at least
each control's. Against the online nonspiking policy, persistent SNN must have
at least a 10-point pooled first-lane margin, no fewer accepted tasks, and no
greater total CPU time. The streaming comparison is a deployment gate: at
both budgets it must have no fewer accepted tasks and no greater total CPU
time than fixed grammar streaming. A failed gate is reported as a negative
result; passing this small pilot is descriptive and does not establish
generalization.

This is a prior-adaptation pilot over one finite synthetic operator-family
stream. Unique target polynomials prevent exact-function repeats, but do not
establish generalization to unseen domains or real sensing. Results that do
not consistently distinguish persistent updates from frozen, per-episode
reset, and nonspiking controls should be reported as inconclusive or negative.

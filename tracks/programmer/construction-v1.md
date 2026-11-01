# Sequential construction pilot

The corpus and acceptance rules in
[`fixtures/construction_v1.json`](../../fixtures/construction_v1.json) are
frozen before model fitting. The corpus SHA-256 is
`1771da269247c9c8a227d48050004fe215158934a14f8684d38d10e35c853735`.
The first checkpoint implements the state, corpus and independent checker;
it does not demonstrate neural program construction or a performance gain.

## Task and action contract

The pilot constructs pure integer arithmetic functions of x and y with add,
subtract and multiply. The bound is four tree operations and four graph
nodes. Reusing an intermediate still contributes its expanded tree cost to
the tree bound. Constants, branches, arrays and loops are later extensions.

The controller receives only observed input/output examples and the partial
program's observed execution. It chooses an operation, a left operand and a
right operand from live references, or explicitly chooses STOP. State edits
go through `ActionSpace`. The fixed host executor computes the new node's
observed output vector after that choice. It does not enumerate and execute
all possible next operations before asking the controller.

An observed match does not automatically stop construction. Inference must
record the controller's stop action. An explicit stop that matches observed
rows ends search; the held checker can reject it but cannot request a retry.
Premature stops, invalid edits and exhausted budgets remain reported failures.

## Data and supervision

The generator catalogs exact integer polynomial functions by minimal tree
size and retains the first syntax witness in grammar order. Hash-order quotas
select nonconstant, nonidentity targets. This canonical-witness choice and
bounded grammar are distribution and supervision biases; they are not a
general program corpus. No evaluated solver's success filters the tasks.

| Minimal tree size | Train | Development | Final |
| --- | ---: | ---: | ---: |
| 1 | 6 | 1 | 1 |
| 2 | 28 | 6 | 7 |
| 3 | 128 | 32 | 44 |
| 4 | 384 | 96 | 192 |
| Total | 546 | 135 | 244 |

The three splits are function-disjoint by exact polynomial identity.
Each task has 36 observed rows on the six-by-six grid from -3 through 2,
and 24 disjoint held inputs sampled from -6 through 6. The complete observed
grid separates polynomials in this grammar: each variable has degree at most
five at the four-operation bound. Exact identity remains separately checked.

Teacher traces replay the generator's program and its observed intermediate
execution, followed by STOP. The trace builder rejects development and final
tasks and does not read held rows. These are supervised construction labels,
not autonomous discovery of an update rule. No LLM supplies labels or actions.

## Development and final evaluation

The initial proposal is a factorized ANN controller over observed rows and
live references, with an SNN comparison using the same action interface.
The protocol records initial training and search settings. Development may
use at most three declared recipes. Every attempt must be retained, and the
selected recipe/runtime must be frozen separately before any final-task
model inference. A new final evaluation cannot be used as another tuning pass.

Controls are streaming, trained/untrained ANN, trained/untrained SNN and
random actions. All receive the same observed specification and grammar.
Report root execution caps separately from neural and feature work; caps do
not imply equal CPU. Training and supervision generation are additional
standalone costs. Initialization, scoring, validation and held/exact checks
also belong in end-to-end accounting.

The primary learning gate requires each trained ANN seed to exceed untrained
ANN and random-action acceptance by ten percentage points at cap 256.
The separate cost gate requires observed full coverage and at least a ten
percent CPU reduction against streaming, including the ANN's full setup and
fit cost amortized over 244 tasks. Missing full coverage fails that cost gate.
Learning without a cost win must be described separately. No energy result
is implied by simulation work or spike activity.

## Reproduction and next work

Generate into a new file; the generator refuses an existing output:

```sh
flock /tmp/rsi-programmer-builder.lock env OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 .venv/bin/python -m scripts.generate_construction_corpus \
  --output /tmp/construction-corpus-new.json
```

Current checks are `tests.test_construction_state` and
`tests.test_construction_corpus`. Controller implementation, training,
development selection, the frozen final runner, independent replay and cost
measurement remain outstanding at the foundation checkpoint. Python/CURLEE
backends remain later stages.

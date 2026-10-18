# Frozen pre-evaluation benchmark

The benchmark compares seven search controls on a fixed arithmetic corpus. Its
protocol and task data were committed before any search results were measured.
The run uses 12 training tasks and 24 evaluation tasks from three syntax
families: balanced trees, repeated subtrees, and skewed trees. Each task has
six observed input/output rows and four held-out rows. The train and evaluation
splits have disjoint task IDs, generating expressions, and input coordinates.

The corpus uses only `add`, `sub`, and `mul` over `x` and `y`, with a maximum
tree size of three operations. The generator enumerates larger syntax pools,
then takes a predetermined sample of 12 programs per family. It does not use
solver outcomes to select tasks. All 36 sampled tasks remain in the corpus,
including simple identities or zero-valued functions. Syntax families do not
guarantee semantic separation, and results do not cover constants, branching,
other DSLs, or larger trees.
The fixture stores each expression as generation provenance; the runner checks
that its recorded rows match the expression, then strips provenance before
search.

The frozen protocol is
[benchmark_pre_eval_v1.json](../fixtures/benchmark_pre_eval_v1.json), and the
corpus is [benchmark_pre_eval_corpus_v1.json][corpus]. The corpus generator
is [generate_benchmark_pre_eval_corpus.js][generator].
With Node.js 18 or newer, regenerate the corpus with:

```sh
node scripts/generate_benchmark_pre_eval_corpus.js
```

The runner checks the protocol, corpus, and generator SHA256 values before it
starts. If any pinned bytes differ, it refuses to run. To create one complete
run in a new output directory, use the project Python environment:

```sh
.venv/bin/python scripts/benchmark_pre_eval.py \
  --output artifacts/benchmark-pre-eval-v1
```

It uses seeds 0, 1, and 2; 40 fitting epochs; and candidate-signature budgets
of 32, 128, 512, and 2048 with tree bound three. Torch intra-op threads are
fixed at one. The learned controls are the existing `SearchPolicy`, a 12–16–1
MLP, and a 12–16 recurrent control with six ticks. All use the same 12-feature
encoding and margin objective. The SNN and MLP each have 225 trainable
parameters; the recurrent control has 497. Equal parameter counts do not
establish equal compute, capacity, or energy.

The other controls are probe-order streaming, a fixed operand heuristic, a
seeded random ordering, and an untrained SNN. Each learned model is fit only
from the 12 training tasks and is reused unchanged for every evaluation task
and budget of its seed. Group generation and labels are derived only from the
observed training rows. Search receives the six observed rows and target
values, no held-out data, and no oracle. It makes one attempt per control,
seed, task, and budget. After search returns, the runner independently checks
any returned graph against both the observed rows and the four held-out rows.
An attempt counts as accepted only if search reports a solution and both checks
pass. Failures remain in the raw results.

The output directory contains:

- `raw.jsonl`: all 2016 control/seed/task/budget attempts, returned graphs,
  search counters, validation counters, state fingerprints before and after
  each evaluation, and explicit errors.
- `training.json`: per-seed label-generation counters and duration, model
  initialization time, fit duration and loss, architecture, parameter count,
  state semantics, actual optimizer/forward/score-row work counters, and
  separate before/after weight hashes and state fingerprints. Group feature
  statistics count four operand hit/inside row passes; target range reads are
  counted separately.
- `aggregate.json`: per-seed and mean family and overall success, actual
  candidate counts, wall time and SNN steps, matched-success budgets, and the
  predeclared trained-SNN decision rule.
- `manifest.json`: input hashes, before/after environment and source manifests,
  output hashes, actual and expected row counts, and infrastructure validity.

For success comparisons, the report gives the lowest listed budget reaching
25%, 50%, 75%, and 100% mean overall accepted fraction across seeds. Targets
not reached are marked as censored at 2048. A trained-SNN advantage is
supported only if it beats the best other control by at least 10 percentage
points at
both 512 and 2048 candidate-signature evaluations, and leads in at least two
family means at both budgets. An infrastructure-invalid run cannot support
that claim. Wall time is reported as descriptive cost and is not a standalone
efficiency result. Conclusions apply only to this frozen corpus, these
controls, budgets, and recorded environment.

[corpus]: ../fixtures/benchmark_pre_eval_corpus_v1.json
[generator]: ../scripts/generate_benchmark_pre_eval_corpus.js

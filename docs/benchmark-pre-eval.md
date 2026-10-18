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

## Version 1 result

The first complete run used the published `code-v62` runtime (`5cf0998`)
and the unchanged `code-v61` inputs. It completed all 2,016 attempts in
14.095s stage wall time with no execution errors or source/dependency
changes. The independently checked accepted counts are:

| Control | Budget 32 | Budget 128 | Budget 512 | Budget 2048 |
| --- | ---: | ---: | ---: | ---: |
| Streaming | 30/72 | 33/72 | 72/72 | 72/72 |
| Heuristic | 33/72 | 39/72 | 69/72 | 72/72 |
| Seeded random | 30/72 | 35/72 | 70/72 | 72/72 |
| Untrained SNN | 28/72 | 34/72 | 62/72 | 72/72 |
| Trained SNN | 27/72 | 33/72 | 62/72 | 72/72 |
| MLP | 27/72 | 33/72 | 62/72 | 72/72 |
| Recurrent | 29/72 | 33/72 | 61/72 | 72/72 |

Each cell represents 24 fixed tasks repeated across three seeds, not 72
independent tasks. Streaming and the heuristic repeat identical outcomes
across seeds. All returned graphs passed observed and held checks; the
remaining attempts exhausted their candidate budgets. The trained SNN
does not pass the frozen advantage criterion. Equal aggregate counts do
not imply identical per-task behavior.

At budget 512, streaming uses 9,837 candidate signature evaluations for
72 accepted attempts; trained SNN uses 13,123 for 62. At matched full
acceptance, streaming uses 9,837 evaluations at budget 512, while trained
SNN uses 14,218 at budget 2048. Their recorded aggregate search times at those
matched-success settings are 0.0195s and 0.3504s, respectively. These are
small-workload timings from this environment, not general runtime or
energy claims. Training, descriptor scoring and validation costs remain
available separately in the raw bundle. Neural controls still score all
descriptors in each reached layer; moving root execution after selection
has not made descriptor scoring free.

The independent audit imports the program interpreter, not the benchmark
runner or its aggregation helpers. It verifies all 2,016 unique matrix
keys, re-executes all 1,406 returned graphs, reproduces acceptance and
summaries, checks budgets and frozen state, and validates input/output
and current source hashes. All 32 final audit checks pass. Its first
report incorrectly required streaming descriptor generation to equal
executed candidate signatures: a partial-budget stop can pull one unused
descriptor.
The corrected check permits exactly that documented lookahead. The
initial failed report is preserved; no measurement or runtime was changed
to obtain the final audit pass.

Exact polynomial expansion finds 12 distinct training functions and 19
distinct evaluation functions among 24 evaluation tasks, with one
function shared between splits. Five evaluation tasks duplicate another
evaluation function. This is a limitation of version 1's declared syntax
separation, not evidence of semantic transfer. All seven controls also
saturate at budget 2048. A separate, more discriminating protocol is
tracked in [#79](https://github.com/Capsize-Games/rsi-programmer/issues/79);
version 1 and its negative result remain frozen.

[Raw run and manifests](../artifacts/benchmark-pre-eval-v1-20261018/),
[independent audit](../artifacts/benchmark-pre-eval-v1-audit-20261018/audit.json),
[audit source](../artifacts/benchmark-pre-eval-v1-audit-20261018/audit.py),
and [stage logs and source bindings](../artifacts/pre-eval-benchmark-validation-20261018/).

[corpus]: ../fixtures/benchmark_pre_eval_corpus_v1.json
[generator]: ../scripts/generate_benchmark_pre_eval_corpus.js

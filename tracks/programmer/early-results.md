# Early construction results

Owner priority on 2026-11-01 was to establish results before expanding the
project. These bounded development trials precede the final benchmark.
They use 546 training functions and 135 function-disjoint development
functions. No final-task model inference has run.

## ANN results

The controller chooses arithmetic operations, live operand references and
STOP. Training uses generator-witness labels and observed partial-program
execution, without an LLM teacher. Inference leaves neural weights fixed.
Every counted solution passes independent observed/held execution and exact
polynomial identity checks. The grammar is limited to add/subtract/multiply
over x and y, with a four-operation expanded tree bound.

| Control | Correct / 135 | Evaluation CPU s |
| --- | ---: | ---: |
| Untrained ANN, greedy | 0 | 0.311 |
| ANN, 20 epochs, greedy | 11 | 0.480 |
| Untrained ANN, masked beam | 0 | 5.445 |
| Random actions, masked beam | 1 | 0.767 |
| ANN, 20 epochs, masked beam | 42 | 5.354 |
| ANN, 100 epochs, greedy | 24 | 0.437 |
| ANN, 100 epochs, masked beam | 78 | 4.562 |
| SNN, 20 epochs, greedy | 19 | 0.581 |
| SNN, 20 epochs, masked beam | 53 | 6.490 |
| Classical streaming | 135 | 1.033 |

All neural measurements here use parameter seed zero and the same 64-unit
architecture. Greedy takes one neural choice and rejects invalid edits.
Beam also masks illegal operands, uses the top two opcode/left/right choices,
retains eight states and caps executed roots at 256. Greedy-to-beam changes
both masks and search; the difference is not an isolated search effect.
Random and untrained controls use the same masked beam as the trained ANN.
Streaming uses a root cap of 16,384. These caps are not equal CPU budgets.

CPU encloses each recorded evaluation loop and independent acceptance, not
training or report output. The early wrappers have slightly different hash
timing boundaries. There are no repeated timing trials. Differences in the
reported timing should not be interpreted as precise speedup estimates.

The 20-epoch fit took 16.559 seconds wall / 16.004 seconds CPU, including
preparation, on 2,504 teacher states and 800 updates. It was repeated before
the beam trial and reproduced the exact trained parameter hash. The second
fit used 100 epochs from the same initialization: 4,000 updates and 61.032
seconds wall / 60.794 seconds CPU. These are separate complete fits, not an
incremental continuation. Fit costs are additional to evaluation.

| Minimal tree size | ANN beam, 20 epochs | ANN beam, 100 epochs |
| --- | ---: | ---: |
| 1 | 1/1 | 1/1 |
| 2 | 6/6 | 6/6 |
| 3 | 22/32 | 27/32 |
| 4 | 13/96 | 44/96 |

The 100-epoch beam retains 36 of the earlier 42 successes, loses six and gains
42 others. Net improvement does not imply monotonically retained solutions.
One new four-operation function it constructs is
`y + (y + (x * (x + x)))`, equivalent to `2*x*x + 2*y`. These trials used the
arithmetic interpreter. Subsequent target-language execution is recorded in
[language-results.md](language-results.md).

## What this supports

Learned construction has a measurable signal on unseen development
functions: 0 to 42 to 78 correct for the initial, 20-epoch and 100-epoch ANN
with the same masked beam. More training helps materially, especially at
four operations. The current model still misses 57/135 functions, and
streaming solves all 135 faster even before neural fitting is charged.
No efficiency or general-programmer claim is supported.

At 20 epochs, direct greedy construction also solves only 64/546 training
functions. Canonical teacher agreement is 1,890/2,504 operations,
1,529/1,958 left pointers and 1,762/1,958 right pointers. Those are teacher
metrics, not program correctness. Errors on training rollouts show that the
problem is not solely a held-function generalization deficit. They do not
identify a unique architectural cause.

Two development training recipes have run, 20 and 100 epochs, using one
parameter seed. Development results informed the longer fit. The sealed
244-function final set has not been used for inference or recipe selection.
The preregistered multi-seed final learning and cost gates have not run.
The SNN control uses the same parameter shapes and six fresh LIF ticks before
the shared pooling/pointer architecture. Its 20-epoch fit took 46.247 seconds
wall / 44.232 seconds CPU, versus 16.559 / 16.004 for ANN. It solves 53/135
with beam, versus ANN's 42/135 after 20 epochs. ANN superiority is therefore
not established for this constructor. The SNN is slower to fit and evaluate;
both trail classical streaming. A 100-epoch SNN fit has not run. The SNN
wrapper's streaming observation is 1.072 seconds CPU; timing trials and
resource-matched learning curves remain unmeasured.

The owner subsequently prioritized useful Python/CURLEE programs over
further model comparisons. See [language-results.md](language-results.md).
These arithmetic diagnostics are retained as evidence; they do not gate
language work. Learner self-edits remain a later stage.

## Evidence and reproduction

Raw per-task graphs, decisions, independent checks and source hashes are in
[`construction-early-dev-20261101`](
../../artifacts/construction-early-dev-20261101/) and the
[SNN control](../../artifacts/construction-early-snn-dev-20261101/).
Retained training weights
are local and excluded from Git. Logical neural work counters describe the
row encoder, including padding; they are not total FLOPs or device energy.

The development runner rejects final tasks and accepts only 20- or 100-epoch
fits. Use a fresh output directory:

```sh
timeout 150s flock -n /tmp/rsi-programmer-builder.lock \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.measure_construction_early --backend ann --epochs 20 \
  --output /tmp/construction-ann-dev-new
```

This runner has explicit timing boundaries in its report; it need not
reproduce earlier wrapper timing. Solver counts and parameter hashes are the
behavioral reproduction targets. A failed or timed-out run remains partial.

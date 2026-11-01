# Stateful programming development trials

The previously qualified learner solved **0 of five new tasks**. A broader
training curriculum also solved **0 of five**, and retained only five of
the six earlier utilities. That curriculum is not promoted. The preceding
[six-task result](repair-results.md) remains valid for its declared tasks;
these trials expose its limited transfer to new behavior.

The new tasks compute a maximum with a zero floor, a minimum with a zero
ceiling, the largest magnitude, the total of positive values capped at two,
and a positive count capped at two. They use the existing loop, condition,
assignment and integer-return grammar. They require state-dependent tests
or composed conditions and updates beyond the preceding counting contexts.

| Task | Frozen learner | Broader curriculum | Supplied control |
| --- | --- | --- | --- |
| Maximum with zero floor | Fail | Fail | Pass |
| Minimum with zero ceiling | Fail | Fail | Pass |
| Largest magnitude | Fail | Fail | Pass |
| Sum positive values, each capped at two | Fail | Fail | Pass |
| Positive count capped at two | Fail | Fail | Pass |

The supplied controls passed 40 held cases per task in isolated Python
execution and actual CURLEE check/VM execution: 200 cases per language.
They establish that these tasks fit the existing grammar and backends.
They are evaluator-written programs, never neural successes, training
labels or new library donors. The emitter still supplies the JSON-file
CLI, initial accumulator, contracts and source lowering.

## Frozen learner

The [task fixture](../../fixtures/stateful_tasks_v1.json) was frozen before
inference. The original 366-function, 30-epoch learner and its three
qualified condition-hole contexts were reused without updating weights.
All five target functions differ from every teacher function on the 24
declared training probes. This is a finite exclusion, not an equivalence
proof over all inputs. The sealed final task set remains unused.

Four tasks yielded no program. For the capped positive count, the learner
generated a program that counts values greater than one. It fits the three
observed examples but returns zero for `[1, -2, 0]`, whose correct result
is one. Actual Python and CURLEE execution rejected it on held behavior.
No resynthesis received those held rows.

The [executed report](
../../artifacts/stateful-programs-v1-execution-20261101/report.json)
records all choices, source hashes and Python outcomes. Its
[CURLEE report](
../../artifacts/stateful-programs-v1-execution-20261101/curlee-execution.json)
records the wrong program's 40 actual runs and the four missing programs.
The [control report](
../../artifacts/stateful-programs-v1-execution-20261101/grammar-controls/report.json)
and its [CURLEE evidence](
../../artifacts/stateful-programs-v1-execution-20261101/grammar-controls/curlee-execution.json)
identify the supplied programs separately.

An initial Python execution attempt was blocked by bubblewrap namespace
setup in the managed sandbox. Its [report](
../../artifacts/stateful-programs-v1-20261101/report.json) is retained as
infrastructure failure. An authorized execution-only retry kept the exact
chosen programs and held inputs and retained OS isolation. The resulting
0/5 outcome is based on actual execution, not that infrastructure error.
The first learner CURLEE attempt was refused by the existing memory guard;
a later admitted run completed without changing the limits.

## Broader execution curriculum

One recipe was recorded before fitting: seed zero, 30 epochs, the same
network and execution features, 1,536 sampled single-condition candidates
and 512 composed-condition candidates. These syntax combinations read no
target program. After syntax, execution, duplicate and target-function
filtering, the combined curriculum contained 1,151 teacher functions and
41,544 supervised token states. All eleven old and new target functions
were excluded on the declared probes before fitting.

Fitting took 20.979 seconds. The five new tasks kept the same three observed
examples and received no query feedback. Earlier tasks used the existing
bounded query protocol to check retention. All programming choices ended
before held arrays were generated from the new seed. Observed/query rows
were excluded from that held set. Common boundary cases are reused;
this is development evaluation, not a fresh final set.

The result was 0/5 on new tasks and 5/6 on retention in actual Python
execution and CURLEE check/VM execution. The odd counter yielded no program.
The new maximum and capped
sum both received an incorrect counting loop; the capped count received an
incorrect parity/square loop. The minimum and largest magnitude had no
program. Full choices and failures are in the [report](
../../artifacts/stateful-curriculum-v1-20261101/report.json).
The [CURLEE report](
../../artifacts/stateful-curriculum-v1-20261101/curlee-execution.json)
records 320 cases across the eight emitted programs: five accepted
retentions and three rejected new programs. A wrong-expected-result control
was rejected, and the compiler hash remained unchanged.

The [protocol](
../../artifacts/stateful-curriculum-v1-20261101/protocol.json) records the
recipe and measured source hashes. The compressed source snapshot retains
the exact measured versions; the reproduction script later received a
line-wrapping-only edit. Weights remain local and excluded from Git.
These single trials establish neither a statistical learning curve nor a
cost advantage. The earlier matched classical search control remains much
faster on its two transfer tasks.

## Query feedback without retraining

A subsequent bounded diagnostic reused both fingerprint-checked models.
Each task could receive three added examples from the existing query pool.
All 73 distinct held inputs from the preceding two trials were removed
from that pool before feedback, leaving 37 visible query inputs. Neither
model changed its weights. All choices ended before generating 40 new
random held arrays, also disjoint from observed, query and prior held rows.

Both models ended with **0/5 programs**. Feedback rejected the misleading
candidates but did not produce replacements within the declared search
limits. The reports retain every attempt and counterexample for the
[qualified model](
../../artifacts/stateful-queries-v1-20261101/qualified/report.json)
and [broader-curriculum model](
../../artifacts/stateful-queries-v1-20261101/broader/report.json).
The [protocol](
../../artifacts/stateful-queries-v1-20261101/protocol.json) records the exact
visible pool and source hash. No program survived construction, so there
were no new Python or CURLEE program executions for this diagnostic.

## Gaps exposed

The maximum and capped-sum tasks have identical first three
input/output examples. A deterministic example-only solver cannot identify
both intended functions from those examples. Better observations or query
feedback are required to distinguish them. This ambiguity does not explain
the other three failures, and adding more teacher functions did not produce
correct stateful programs.

The query diagnostic separates specification ambiguity from the remaining
construction failure: additional visible observations still yielded no
programs. The next change should target construction and transfer of
state-dependent updates. The current fragment library varies conditions
while retaining a donor's update; its three contexts cannot directly
replace the accumulator as the maximum/minimum tasks require. The capped
count may still fit the counting context with a suitable arithmetic
condition, so its failed search is not a grammar impossibility result.
Broader typed holes or loop-state execution features are candidate
experiments, not established fixes.
Another model-family comparison or a larger undirected curriculum is not
supported by this result. Strings, records, multiple mutable variables,
general file processing, arbitrary libraries and source repair remain open.

## Validation and reproduction

The 27 focused tests passed in 2.230 seconds. They cover the new reference
functions, supplied grammar controls, eleven-function exclusion, statement
construction, query repair, fragment provenance, held acceptance and track
ownership. The earlier full programmer regression passed 520 tests at
`88ca7d2`; it predates these additions and is not a full pass for this code.
Exact commands, logs and exclusions are retained in the
[validation receipt](
../../artifacts/stateful-validation-20261101/receipt.json).

Run the trials sequentially with fresh output directories:

```sh
flock /tmp/rsi-programmer-builder.lock \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.measure_stateful_programs --output /tmp/stateful-frozen-new

flock /tmp/rsi-programmer-builder.lock \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.measure_stateful_curriculum --output /tmp/stateful-fit-new

flock /tmp/rsi-programmer-builder.lock \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.measure_stateful_queries --output /tmp/stateful-queries-new

python3 /home/joe/Projects/curlee/scripts/run_limited.py \
  /usr/bin/python3 -m scripts.check_statement_curlee \
  --compiler /home/joe/Projects/curlee/build/linux-debug/curlee \
  --output /tmp/stateful-fit-new
```

Python execution requires the existing bubblewrap isolation capabilities.
For the frozen trial's control directory, use `scripts.check_stateful_curlee`
with `--output /tmp/stateful-frozen-new/grammar-controls` under the same
CURLEE guard. Compiler source remains in its local-only repository. The
binary hash remained
`412d0e6ee5b8fb9e380322683863849523e1a5797e7e7dbd42bf174b138555ac`
through the completed frozen/control checks. This does not qualify the
compiler generally or establish a source-matched compiler build.

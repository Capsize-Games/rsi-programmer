# Query repair after verified context growth

The retained-model route now accepts **8/11 development tasks** in both
Python and CURLEE: the six earlier utilities, maximum and minimum. Each
accepted program passed 40 fresh held cases per language, for **320 Python
file runs and 320 CURLEE check/VM cases**. Largest magnitude, capped
positive-value sum and capped positive count still yielded no program.

This trial repairs the minimum-retention failure in the preceding
[context-growth trial](context-growth-results.md). It preserves that failed
trial, the earlier negative experiments and the unchanged trained models.
This is bounded query-guided resynthesis with verified contexts, not
general programming or recursive self-improvement.

## What the learner changed

The maximum initially used the wrong update: it added a value when the
value exceeded the accumulator. The visible query `[2, 3, 4]` expects four,
but that candidate returned five. After adding this example, the learner
selected a replacement assignment instead of addition. The resulting
[Python maximum](
../../artifacts/statement-context-repair-v1-20261101/maximum_floor_zero/python/clean_readings.py)
uses `if value > acc: acc = value`.

Minimum needed two counterexamples. First, `[1, 2, 3]` expects zero and
rejected a candidate returning three. Next, `[3, 2, 1, 0, -1]` expects -1
and rejected the wrong summation candidate returning zero. The learner
then selected `if value <= acc: acc = value`, producing the
[qualified minimum](
../../artifacts/statement-context-repair-v1-20261101/minimum_ceiling_zero/python/clean_readings.py).
Both counterexamples came from the previously declared visible query pool;
no held input entered repair.

The library traversal and counterexample selection are classical. Neural
token scores fill conditions in contexts mined from independently checked
programs. Loop/update/return structure, initialization, lowering, contracts
and CLI handling remain supplied or copied code. The minimum/maximum
components occur inside other teacher functions in the component-preserving
curriculum. These results do not establish discovery of those algorithms.

## Protocol and independent execution

The [protocol](
../../artifacts/statement-context-repair-v1-20261101/protocol.json) was saved
before any search or feedback. The route first tries the earlier trained
controller with the old three-context library. When that finds no program,
it tries the component-trained controller with the four-context library.
Each individual search has a one-second budget; a route can call both.
No fitting or parameter update occurs during repair.

Each task starts with its original three examples and may add at most three
visible examples. The 37-input query pool excludes earlier held inputs.
The same pool and added-example cap apply to the untrained, classical and
old-library controls. Oracle-call counts differ because queries adapt to
the candidates; those calls are recorded. Maximum used one added example
and 39 oracle calls; minimum used two and 40 calls.

All four controllers finish their decisions before 40 random held arrays
are generated from seed 461047. The held rows are disjoint from queries,
observations and earlier held arrays. They have lengths 13–100 and values
-100 through 100; this trial adds no boundary-range qualification. The
sealed final set remains untouched.

| Role | Python accepted | CURLEE accepted |
| --- | ---: | ---: |
| Six earlier utilities | 6/6 | 6/6 |
| Minimum donor retention | 1/1 | 1/1 |
| Four transfer tasks | 1/4 | 1/4 |
| Total | 8/11 | 8/11 |

The [Python report](
../../artifacts/statement-context-repair-v1-20261101/report.json) retains
every repair attempt, query, choice and actual file result. The
[CURLEE report](
../../artifacts/statement-context-repair-v1-20261101/curlee-execution.json)
records every actual check/VM case. Compiler hash
`412d0e6ee5b8fb9e380322683863849523e1a5797e7e7dbd42bf174b138555ac`
remained unchanged, and the wrong-expected-result control was rejected.
All compiler execution used the existing CURLEE resource guard. Source
and earlier executions were rechecked before using the minimum as a donor.

## Controls and costs

The [control report](
../../artifacts/statement-context-repair-v1-20261101/controls.json) uses
independent held **IR interpretation only**. The neural route received
the actual target-language checks above; the other controllers did not
receive additional Python or CURLEE executions in this trial.

| Controller | Held IR accepted / 11 | Total repair/search time | Oracle calls |
| --- | ---: | ---: | ---: |
| Retained-model neural route | 8 | 5.796 s | 312 |
| Untrained, expanded library | 0 | 13.145 s | 33 |
| Classical, expanded library | 8 | 8.637 s | 309 |
| Component-trained, old library | 5 | 4.435 s | 210 |

These are single observations, not repeated timing estimates. CPU times
are recorded too. The neural route may make two searches per repair call;
the controls use one. Training, source emission, donor qualification and
backend execution are outside these component timings. Earlier fits took
5.821 and 38.019 seconds. The classical control solves the same number of
tasks, and it was much faster on maximum/minimum in the preceding trial.
There is no demonstrated matched-total-cost advantage.

## Remaining work

The three unsolved tasks involve richer combinations of conditions and
updates. The current grammar can express their supplied controls, but the
learner has not constructed them. The next bounded experiment should
compose branch fragments
from the qualified programs and test the resulting source immediately.
Examples include applying the maximum step to both a value and its
negation, or combining an accumulating step with a correction. These are
proposed search choices, not implemented capabilities or training labels.
Keep held rows out of every composition decision and retain strong
classical and untrained controls.

The supported language remains one accumulator, one bounded input loop,
conditions, integer expressions and scalar returns. The learner does not
select arbitrary libraries, manipulate strings/records, edit source
repositories or implement general file processing. Python's file CLI is
supplied; CURLEE exposes the bounded routine. Broader useful programming,
the full cost gate and bounded learner self-improvement remain open.

## Validation and reproduction

The unchanged package and test sources passed 530 programmer fast-track
tests in 270.337 seconds at `914e4e7`. The first attempt was stopped by the
memory reserve and is retained. The new measurement script was then
exercised through frozen synthesis and actual Python/CURLEE execution.
Other test lanes/tracks were not rerun, and remote CI remains pending.

```sh
flock /tmp/rsi-programmer-builder.lock \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.measure_context_repair --output /tmp/context-repair-new

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.execute_statement_trial --output /tmp/context-repair-new

python3 /home/joe/Projects/curlee/scripts/run_limited.py \
  /usr/bin/python3 -m scripts.check_statement_curlee \
  --compiler /home/joe/Projects/curlee/build/linux-debug/curlee \
  --output /tmp/context-repair-new
```

The [receipt](
../../artifacts/statement-context-repair-v1-20261101/receipt.json) retains
commands, measured source hashes, logs and exclusions. Python requires the
existing bubblewrap isolation capabilities. Weights remain local and
excluded from Git. Tags/releases remain paused under the project-specific
instruction.

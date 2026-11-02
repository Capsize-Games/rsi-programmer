# Composing verified branches

The integrated route now passes **11/11 development tasks** in Python and
CURLEE, with **40 held cases per task per language**: 440 Python-file runs
and 440 actual CURLEE check/VM cases. It retains eight previously qualified
behaviors and constructs the three remaining compositions: largest
magnitude, capped positive-value sum and capped positive count.

This is bounded library composition. The host chooses branch pairs and
enumerates accumulator add/sub variants. The trained token model chooses
a shared integer expression used in the second branch. The host supplies
the loop, copied statement structure, initialization, source lowering,
contracts and JSON-file CLI. The result does not establish arbitrary source
generation, learned abstraction discovery or recursive self-improvement.

## Generated programs

The new processing bodies use two sequential conditions inside one loop.
Their [Python sources](
../../artifacts/statement-composed-route-v1-20261101/report.json) and
corresponding CURLEE routines are recorded with exact hashes.

The [largest-magnitude program](
../../artifacts/statement-composed-route-v1-20261101/largest_magnitude/python/clean_readings.py)
applies the copied maximum step to the item and its negation:

```python
if value > acc:
    acc = value
if 0 - value > acc:
    acc = 0 - value
```

The [capped-value total](
../../artifacts/statement-composed-route-v1-20261101/positive_values_capped_at_two_total/python/clean_readings.py)
counts each positive integer once, and integers above one a second time.
For integer readings, that gives each positive item a contribution capped
at two. Its shared expression is `value + -1`.

The [capped positive counter](
../../artifacts/statement-composed-route-v1-20261101/positive_count_capped_at_two/python/clean_readings.py)
increments for a positive item, then subtracts one if `acc - 2 > 0`.
The accumulator therefore stays at or below two after each iteration.
These are different statement combinations from the supplied grammar
controls; no target witness code entered the composition generator.

## Trials and donor qualification

The first trial used eight independently checked single-branch donors from
the preceding [query-repair result](context-repair-results.md). It solved
largest magnitude and capped-value sum, but not capped counting. The
trained controller solved 2/3, the untrained controller 0/3 and classical
search 2/3 in held IR checks. The trained programs also passed 40 actual
Python and CURLEE cases each. The negative capped-count attempt remains
in the [first trial](
../../artifacts/statement-branches-v1-20261101/protocol.json).

The existing neural controller then generated a positive-count primitive.
Its function was already present in the earlier training curriculum on
the declared probes. This was donor qualification, not a new unseen-function
result or new training. The primitive passed 40 held Python and 40 CURLEE
cases before admission; see its [record](
../../artifacts/positive-counter-v1-20261101/report.json).

With that ninth donor, the trained controller solved all three compositions.
The untrained controller solved two and classical search all three in held
IR checks. The trained programs passed actual execution in both languages;
the [second trial](
../../artifacts/statement-branches-v2-20261101/protocol.json) preserves its
different capped-value algorithm and all control choices.

The shared argument grammar is a leaf or one binary operation on leaves.
The trained and untrained searches keep eight partial expressions; the
classical control interleaves all expressions by size across the same
contexts. First-donor ranking uses observed execution residuals, keeps
three donors, and prefers cloning a donor before other branch pairs.
These are supplied classical search biases. Input substitution is shared
between conditions and assignments, and every completed program must pass
the original typed parser and 32-token bound.

## Integrated route and held separation

The generic route first tries the earlier controller with its three
condition contexts, then the component-trained controller with four
contexts. If neither finds a fitting program, it tries branch composition.
It receives examples, not a task identifier. A separate engine may add at
most three counterexamples from the declared 37-input visible pool.
Largest magnitude needed none; the two capped tasks each needed one.
Both neural parameter fingerprints remain unchanged during all searches.

All controller choices finished before generating the integrated trial's
40 random held arrays from seed 461067, lengths 13–100 and values -100
through 100. Held rows were never passed to repair or composition.
Separate donor/composition trials also checked empty/singleton arrays and
length-256 boundary arrays. Qualification remains bounded; this is not an
extreme-int64 or general compiler qualification. The original sealed final
set stays unopened.

| Role | Python accepted | CURLEE accepted |
| --- | ---: | ---: |
| Six original utilities | 6/6 | 6/6 |
| Previously qualified maximum/minimum | 2/2 | 2/2 |
| Three new compositions | 3/3 | 3/3 |
| Total | 11/11 | 11/11 |

The [integrated report](
../../artifacts/statement-composed-route-v1-20261101/report.json),
[CURLEE evidence](
../../artifacts/statement-composed-route-v1-20261101/curlee-execution.json)
and [qualification records](
../../artifacts/statement-composed-route-v1-20261101/fragment-qualification.json)
retain choices, queries, donor hashes and executions. The compiler stayed
at hash
`412d0e6ee5b8fb9e380322683863849523e1a5797e7e7dbd42bf174b138555ac`.
The wrong-expected-result control was rejected. The first integrated
compiler attempt stopped at resource-guard status 75; the later complete
run used the unchanged guard after the concurrent regression finished.

## Stronger classical control and cost limits

The initial integrated comparison accepted 11/11 for the neural route,
10/11 for untrained search and 10/11 for condition-first classical search.
The classical miss was a wrong capped counter selected on its last
permitted repair round.

A retrospective stronger control tries composition before condition-only
search. It accepts **11/11** on fresh held IR rows, alongside 11/11 for the
frozen neural choices. It took 0.365 seconds across the eleven tasks,
compared with the earlier neural route's 3.116 seconds. These are separate
single observations and software snapshots, not a matched-cost timing
estimate. The [control report](
../../artifacts/statement-classical-compose-first-20261101/report.json)
retains its fresh rows, decisions and paired frozen-choice checks.

All non-primary control outcomes here are **held IR interpretation only**,
not additional Python/CURLEE executions. The neural route can make two
one-second condition searches plus a five-second composition search per
repair call; controls use one condition search plus composition. Training,
qualification, oracle calls and backend costs are additional. The stronger
control removes any claim of a solve-rate advantage on these eleven tasks.
There is no demonstrated matched-total-cost advantage.

## Validation and next work

Thirty-eight affected tests passed, including source-mutation rejection,
shared-input semantics and expression depth/scope checks. A 535-test
programmer fast-track run passed in 262.998 seconds before a final argument
shape guard was tightened. The latest focused run passes all 38 checks;
the full rerun after that guard passes all 535 tests in 267.209 seconds.
Remote CI remains pending at this checkpoint.
Other test lanes/tracks were not rerun. Exact snapshots, corrections and
logs are in the [receipt](
../../artifacts/statement-branches-validation-20261101/receipt.json).

All eleven functions have been used repeatedly for development. The next
required measurement is frozen unseen-function evaluation with matched
total-cost accounting and strong classical controls, before another
capacity increase or learner-improvement claim. The current evidence
supports a bounded programmer using supplied and mined components. Strings,
records, general libraries, source-repository repair and unrestricted
programming remain open. CURLEE still exposes a bounded processing routine;
its file-handling interface is not implemented here.

## Reproduce

The two earlier fingerprint-checked checkpoints must be present locally.
The statement-repair and return-curriculum commands recreate them when
their original recipes reproduce the recorded fingerprints. Weights stay
excluded from Git. Run from the repository root with a fresh output path:

```sh
flock /tmp/rsi-programmer-builder.lock \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.measure_context_repair \
  --branch-donor artifacts/positive-counter-v1-20261101 \
  --held-seed 461067 --output /tmp/composed-programmer-new

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.execute_statement_trial --output /tmp/composed-programmer-new

python3 /home/joe/Projects/curlee/scripts/run_limited.py \
  /usr/bin/python3 -m scripts.check_statement_curlee \
  --compiler /home/joe/Projects/curlee/build/linux-debug/curlee \
  --output /tmp/composed-programmer-new
```

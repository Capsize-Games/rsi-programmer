# Transfer from a qualified minimum program

The learner constructed a maximum program by filling a condition hole in
the previously qualified minimum. The maximum passed **40 held Python-file
cases and 40 CURLEE cases**. The retained-model route also kept all six
earlier utilities, for **7/11 accepted tasks** in both target languages.

The route failed to retain the minimum itself. It selected a different
fitting program, which failed held execution. The original minimum remains
independently qualified in the [preceding trial](component-results.md).
Across trials there are eight separately checked task programs; this
search route does not reliably reproduce all eight.

The [generated maximum CLI](
../../artifacts/statement-context-growth-v1-20261101/maximum_floor_zero/python/clean_readings.py)
contains:

```python
acc = 0
for value in items:
    if value - 0 > acc:
        acc = value
return acc
```

The learner fills only the new condition. The loop, replacement assignment,
return and stop come from the verified donor. Source lowering, initial
accumulator, contracts and JSON-file handling remain supplied code.

## Qualification and frozen choices

The prior library had three condition-hole contexts. Rechecking the new
minimum trial's actual Python/CURLEE records, source bytes and case hashes
yielded four contexts, including the new replacement-assignment context.
The expanded library retains all three prior context IDs. Qualification
is classical and exposes no held rows to the neural search API.

The primary route first tries the previous qualified neural controller
with the old library. If it finds no fitting program, it tries the
component-trained controller with the expanded library. Each search has a
one-second budget, width 32 and four next-token choices. A fallback can
therefore consume two search budgets. The models' fingerprints are checked
before and after the trial; no fitting occurs.

All tasks use their original three examples plus the same first three
inputs from the declared visible query pool. The six controller trials
finish all choices before 40 new random held arrays are generated from
seed 461043. Held arrays have lengths 13–100 and values -100 through 100.
The sealed final set remains unused.

## Actual target-language results

| Role | Python accepted | CURLEE accepted |
| --- | ---: | ---: |
| Six earlier utilities | 6/6 | 6/6 |
| Minimum donor retention | 0/1 | 0/1 |
| Four transfer tasks | 1/4 | 1/4 |
| Total | 7/11 | 7/11 |

The transfer success is maximum with a zero floor. Largest magnitude,
capped positive-value sum and capped positive count yielded no program.
Eight programs were emitted: seven correct programs and the wrong minimum.
All eight were executed on 40 cases each, giving 320 Python-file runs and
320 CURLEE check/VM cases. Only 280 cases per language belong to accepted
programs. The wrong minimum returned -125 where one held case expected
-99; it is rejected, not promoted as a donor.

The [Python report](
../../artifacts/statement-context-growth-v1-20261101/report.json),
[CURLEE report](
../../artifacts/statement-context-growth-v1-20261101/curlee-execution.json)
and [qualified library evidence](
../../artifacts/statement-context-growth-v1-20261101/fragment-qualification.json)
retain all choices and failures. The compiler stayed at hash
`412d0e6ee5b8fb9e380322683863849523e1a5797e7e7dbd42bf174b138555ac`.
The wrong-expected-result control was rejected. Compiler execution used
the existing CURLEE resource guard.

## Controls and cost limits

The [control report](
../../artifacts/statement-context-growth-v1-20261101/controls.json) uses
the independent interpreter on the same held rows. Those control outcomes
are **IR checks only**, not additional Python or CURLEE executions. The
primary route above received actual target-language checks separately.

| Controller | Held IR accepted / 11 | Maximum correct | Maximum search time |
| --- | ---: | --- | ---: |
| Previous model, expanded library | 6 | No | 170.2 ms |
| Component model, old library | 5 | No | 233.7 ms |
| Component model, expanded library | 6 | Yes | 347.5 ms |
| Untrained model, expanded library | 0 | No | 350.9 ms |
| Classical search, expanded library | 8 | Yes | 2.49 ms |
| Retained-model route | 7 | Yes | 408.1 ms |

These are single observed search timings; CPU times are also recorded.
All individual searches receive the same one-second wall limit, but the
primary route can call two controllers. Fitting, query labeling, donor
qualification, source emission and backend execution are outside these
component timings. The earlier fits took 5.821 and 38.019 seconds. This
does not pass the matched-total-cost gate. Classical search solves more
tasks and constructs maximum much faster in this trial.

The maximum needs both the new context and the component-trained model
in this comparison. That is evidence of bounded learned transfer; the
minimum component was deliberately present inside other training functions.
It is not learned abstraction discovery, algorithm invention, general
programming or recursive self-improvement.

## Remaining gap and next check

The library stores hole contexts rather than guaranteeing reuse of a
complete donor program. The current route can therefore replace a known
correct behavior with a wrong program that fits a few examples. The next
bounded check uses the existing visible query pool to reject such fitting
candidates and request resynthesis. It must preserve this failed retention
trial and keep all held inputs outside feedback. Retaining qualified whole
programs separately is another candidate change, not an implemented result.

The three other stateful tasks remain unresolved. The language still has
one accumulator, one bounded input loop and scalar integer results. There
are no new strings, records, library calls, general file processing or
arbitrary source editing capabilities.

## Validation and reproduction

Thirty-three focused tests passed in 9.922 seconds for the component/search
changes. The first broader programmer regression stopped with status 75
when system memory crossed its 5 GiB reserve. The admitted retry passed all 530
programmer fast-track tests in 270.337 seconds at `914e4e7`. The tested
package and test-source hashes still match. Other lanes/tracks were not
rerun; remote CI remains pending.

```sh
flock /tmp/rsi-programmer-builder.lock \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.measure_context_growth --output /tmp/context-growth-new

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.execute_statement_trial --output /tmp/context-growth-new

python3 /home/joe/Projects/curlee/scripts/run_limited.py \
  /usr/bin/python3 -m scripts.check_statement_curlee \
  --compiler /home/joe/Projects/curlee/build/linux-debug/curlee \
  --output /tmp/context-growth-new
```

Python requires the existing bubblewrap isolation capabilities. Weights
remain local and excluded from Git. The
[validation receipt](
../../artifacts/statement-context-growth-v1-20261101/receipt.json) records
commands, source hashes, logs, costs and exclusions. Tags/releases remain
paused under the project-specific instruction.

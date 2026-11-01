# Statement repair and verified fragment transfer

All six bounded development utilities now pass 40 held Python-file cases
and 40 CURLEE cases each. Four programs were retained from the preceding
trial. The negative counter and greater-than-two counter were constructed
by reusing a verified counting context and filling a new condition.

The generated standalone programs include the repaired
[negative counter](
../../artifacts/statement-repair-v1-20261101/negative_count/python/clean_readings.py)
and [threshold counter](
../../artifacts/statement-repair-v1-20261101/above_two_count/python/clean_readings.py).
Their core conditions are respectively `value < 0` and `value > 2`.
Each program reads an integer list from JSON and writes an integer result
to a different file. CURLEE implements the corresponding bounded array
routine; its file-handling interface remains unfinished.

## What changed

The neural weights exactly reproduce the preceding execution-context
trial's fingerprint. They were not updated during repair or evaluation.
The complete six target functions remain excluded from teacher programs on
the declared finite probe suite. This remains development work, not a final
generalization result.

Query-guided resynthesis alone retained four programs and repaired neither
failure. For the wrong negative counter, the first distinguishing query
was `[1, 2, 3] → 0`; the candidate returned `2`. That example rejected the
candidate, but the learner found no replacement within three added
examples. The threshold task likewise still had no program.

The library builder then qualified four prior donor programs against their
unchanged Python results, CURLEE execution records and source hashes. It
mined three contexts by replacing an `if` condition with a typed Boolean
hole. Identical contexts share provenance. The counting context came from
the independently checked odd counter.

Context traversal is classical. The trained policy fills the hole from
observed examples; the copied loop, update, return and stop suffix come
from the donor. This is verified-fragment reuse with neural hole filling.
It does not demonstrate neural discovery of new abstractions or a learned
retriever that outperforms classical retrieval.

Both missing conditions were found with the augmented examples and the
counting context. A separate experiment scored context order using the
existing token model. It searched the counting context later and was
slower; it is retained as a negative result rather than promoted.

## Independent execution

All program and control choices finished before generating the new held
arrays. The 39-input query pool and the 40 held arrays are disjoint. Held
cases include 36 seeded random arrays of length 13–100 and four larger
boundary arrays, including length 256. Held values are not passed to
resynthesis or fragment search.

| Stage | Accepted tasks |
| --- | ---: |
| Query-guided resynthesis alone | 4/6 |
| Add verified contexts and neural hole filling | 6/6 |
| Actual isolated Python CLI execution | 6/6, 240 cases |
| Actual CURLEE check and VM execution | 6/6, 240 cases |

The CURLEE compiler hash remained
`412d0e6ee5b8fb9e380322683863849523e1a5797e7e7dbd42bf174b138555ac`.
The wrong-expected-result control was rejected. The first compiler attempt
was refused with status 75 because available memory was below the 6 GiB
reserve. A later admitted run completed under the existing guard; no limit
was raised and no unrelated workload was stopped.

The [primary report](
../../artifacts/statement-repair-v1-20261101/report.json),
[compiler report](
../../artifacts/statement-repair-v1-20261101/curlee-execution.json),
[repair history](
../../artifacts/statement-repair-v1-20261101/repair-decisions.json)
and [qualified context provenance](
../../artifacts/statement-repair-v1-20261101/fragment-qualification.json)
retain choices, failures, query feedback, outputs and evidence hashes.

## Controls and costs

All transfer-stage controllers received the same contexts, the same six
observed examples per target and a 0.5-second wall-time budget per target.
The classical control interleaves contexts by condition-expression size.
It does not exhaust one large context before trying another.

| Transfer controller | Correct / 2 | Negative-counter time | Threshold-counter time |
| --- | ---: | ---: | ---: |
| Trained neural hole filling | 2 | 74.6 ms | 106.4 ms |
| Trained neural, scored context order | 2 | 150.4 ms | 185.5 ms |
| Untrained neural, same hole search | 0 | 304.3 ms | 254.6 ms |
| Classical breadth-first hole search | 2 | 0.85 ms | 2.03 ms |

These are single observations from the reproduction command, not repeated
timing estimates. CPU times are also recorded in its [control report](
../../artifacts/statement-repair-v1-reproduction-20261101/transfer-controls.json).
The separate initial trial is retained; its timings differ.

The policy has a learned signal against its untrained control, but the
classical controller is much faster here. Neural context scoring made
retrieval worse. The prior fit took 5.821 seconds and is additional to
search. Earlier query generation, donor qualification, library mining,
source emission and backend execution are outside this component budget.
This is not the full matched-total-cost gate, and it supports no neural
efficiency advantage. Both that gate and broader function evaluation remain
required before declaring the active goal achieved.

## Reproduce

Run sequentially with a fresh output directory:

```sh
flock -n /tmp/rsi-programmer-builder.lock \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.measure_statement_repair --output /tmp/statement-repair-new

python3 /home/joe/Projects/curlee/scripts/run_limited.py \
  /usr/bin/python3 -m scripts.check_statement_curlee \
  --compiler /home/joe/Projects/curlee/build/linux-debug/curlee \
  --output /tmp/statement-repair-new
```

The first command reuses a fingerprint-checked local checkpoint or
recreates the qualified policy from training data. Weights remain excluded
from Git. The command reproduced all six Python successes and exactly the
same program source and held inputs. Unchanged CURLEE execution evidence
was reused rather than rerunning the duplicate programs.

Seventeen focused checks passed in 0.220 seconds. They exercise bounded
query feedback, scope-preserving fragment extraction, changed-source and
false-result rejection, nonempty held evidence, no-library termination and
track ownership. The broader fast lane ran 517 checks with one provenance
error: an invalid ancestor `.git` marker under `/tmp` was mistaken for a
repository. Actual Git worktree discovery fixes that error; all seven
affected provenance checks pass. The complete rerun is pending. Logs,
source hashes and exclusions are in the
[validation receipt](
../../artifacts/statement-repair-validation-20261101/receipt.json).

The supported language still has one accumulator, one bounded input loop,
nested conditions and scalar integer returns. The emitter supplies CLI
plumbing and language lowering. Arbitrary libraries, strings, records,
general source editing and recursive learner improvement remain open.
The six-task development result does not establish a general Python or
CURLEE programmer. The sealed arithmetic final set remains unused.

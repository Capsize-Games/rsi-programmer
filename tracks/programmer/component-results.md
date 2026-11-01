# Transfer of stateful loop components

A component-preserving execution curriculum constructed a new minimum
program. Its Python CLI and CURLEE routine each passed **40 held cases**.
The new model solved **1/5 stateful tasks** and retained **4/6 earlier
utilities**. It remains separate from the earlier six-task qualified model.

The [generated Python program](
../../artifacts/statement-return-v1-20261101/minimum_ceiling_zero/python/clean_readings.py)
reads an integer-list JSON file and writes its minimum with a zero ceiling.
Its processing body is:

```python
acc = 0
for value in items:
    if acc > value:
        acc = value
return acc
```

The neural controller chose the condition and assignment tokens within a
loop-body hole mined from four independently qualified prior programs.
The donor supplies the loop and return; the emitter supplies initialization,
source lowering, contracts and CLI plumbing. The corresponding minimum
component occurs in other training functions. This is component transfer,
not invention of the minimum algorithm or arbitrary source generation.

## Retained negative trials

The preceding [stateful trials](stateful-results.md) solved 0/5 new tasks.
Replacing a condition hole with a complete loop-body hole also solved 0/5
with the qualified, broader-curriculum and untrained controllers. The
original condition-hole control likewise solved 0/5 on the same six
observed examples per task.

A retrospective diagnostic scored the supplied grammar controls' token
paths. For the qualified model, the `acc` operand in a direct item-versus-
accumulator comparison ranked eleventh, outside the four-token expansion
limit. That establishes loss of that reference path, not loss of every
equivalent program. Expanding every legal token, at beam widths 32 and 128
with a one-second search budget, still produced no programs. A wider search
alone did not resolve this failure.

The three negative trial directories retain all controllers and choices:

- [Original loop-body trial](
  ../../artifacts/statement-bodies-v1-20261101/protocol.json)
- [All legal tokens, width 32](
  ../../artifacts/statement-bodies-all32-20261101/protocol.json)
- [All legal tokens, width 128](
  ../../artifacts/statement-bodies-all128-20261101/protocol.json)

No controller emitted a program in those trials, so they required no new
Python or CURLEE executions. Their compressed measured-source snapshots
preserve the pre-formatting source versions. The subsequent source changes
preserve the original four-choice default.

## Curriculum change and scope

Neither the original 366-function curriculum nor the broader 1,151-function
curriculum retained a direct `value`-versus-`acc` comparison with a
`set value` body. Whole-function exclusion removed useful components along
with the complete requested behavior.

The new generator keeps the previous candidates, adds an output-offset
variant, and enumerates a generic grid of leaf comparisons, leaf assignments
and six return expressions. It reads no target names or supplied witness
code. The independent interpreter labels executions. All eleven complete
target functions are then excluded on the same 24 training probes.
Minimum and maximum components remain present inside other complete
functions, such as a reduction followed by an output transform. This
deliberate bias must accompany any transfer claim; finite probe exclusion
is not a general function-equivalence proof.

After filtering, the curriculum contained 2,647 functions and 97,410
supervised token states. A single seed-zero, 30-epoch fit took 38.019 seconds.
The network and execution features were unchanged. Search used a width-128
beam, every legal next token and a one-second budget per task.

Each task used its three original examples plus the first three inputs from
the previously declared visible query pool. These extra inputs were the
same across controllers and tasks. No held rows entered fitting or search.
All choices finished before generating 40 random held arrays from seed
461041, with lengths 13–100 and values from -100 through 100. This remains
development evaluation; the sealed final set was not opened.

| New stateful task | Python | CURLEE |
| --- | --- | --- |
| Maximum with zero floor | No program | No program |
| Minimum with zero ceiling | 40/40 | 40/40 |
| Largest magnitude | No program | No program |
| Sum positive values, each capped at two | No program | No program |
| Positive count capped at two | No program | No program |

The four accepted retention programs are positive sum, negative count,
positive-square sum and greater-than-two count. Even sum and odd count
yielded no program. All five emitted programs passed 200 Python-file cases
and 200 CURLEE check/VM cases in total. Retention losses prevent replacing
the previous qualified model with this one.

The [fit and Python report](
../../artifacts/statement-return-v1-20261101/report.json),
[CURLEE report](
../../artifacts/statement-return-v1-20261101/curlee-execution.json)
and [protocol](
../../artifacts/statement-return-v1-20261101/protocol.json) retain all
attempts, costs and source hashes. The compiler hash remained
`412d0e6ee5b8fb9e380322683863849523e1a5797e7e7dbd42bf174b138555ac`.
The wrong-expected-result control was rejected. Compiler execution used
the existing local resource guard. An initial Python execution command
used the system interpreter, which lacked the existing Torch dependency;
the corrected execution-only command used the project environment. There
was no retraining or held-guided resynthesis.

## Next capability check

The minimum has now been independently qualified as a donor. An
[observed-only transfer probe](
../../artifacts/statement-return-v1-20261101/maximum-transfer-development-probe.json)
filled its condition hole with `value - 0 > acc`, producing a maximum
candidate that fits six examples. The subsequent
[context-growth trial](context-growth-results.md) checks this candidate
on 40 held Python and CURLEE cases each. That transfer passes, while its
minimum-retention attempt fails.

That trial freezes the expanded library and all controller choices before
checking new held rows against old-library, untrained and classical
controls. Earlier failed trials, retention losses and the classical cost
advantage remain visible.
The present result does not demonstrate learned abstraction discovery,
a full matched-total-cost advantage, general programming or recursive
self-improvement.

## Reproduce

Use fresh output directories and run these commands sequentially:

```sh
flock /tmp/rsi-programmer-builder.lock \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.measure_return_curriculum --output /tmp/return-trial-new

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.execute_statement_trial --output /tmp/return-trial-new

python3 /home/joe/Projects/curlee/scripts/run_limited.py \
  /usr/bin/python3 -m scripts.check_statement_curlee \
  --compiler /home/joe/Projects/curlee/build/linux-debug/curlee \
  --output /tmp/return-trial-new
```

Python requires the existing bubblewrap isolation capabilities. Weights
remain local and excluded from Git. Thirty-three focused tests passed;
the broader regression stopped with status 75 when available memory fell
below its 5 GiB reserve. The admitted retry passed all
530 programmer fast-track tests in 270.337 seconds. Remote CI remains
pending. The
[validation receipt](
../../artifacts/statement-component-validation-20261101/receipt.json)
records exact commands, logs and exclusions. This checkpoint does not
qualify arbitrary Python isolation, the compiler generally, strings,
records, unrestricted libraries or general source repair.

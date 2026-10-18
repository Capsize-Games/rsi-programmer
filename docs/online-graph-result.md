# Online graph-learning pilot result

The prototype updates its readout from real executed-graph outcomes without
an LLM in the learning loop. Its first frozen experiment does not show
useful cross-task persistence: all three predeclared outcome gates fail.
Implementation and this completed experiment are tracked in
[#83](https://github.com/Capsize-Games/rsi-programmer/issues/83).

The corpus, protocol and gates were published in `code-v89` before any
full policy measurement. The measured runtime is `code-v90`, commit
`5cdb9b5782a37bf0c32bb15d22098399f3ce12df`. Its fixed input hash is
`a4f9af2dfc0d8c4539d700b032dcc223c82aa272f02497902024d9f371096140`.
All 1,440 planned episode rows completed. Source/runtime/test/input hashes
remained unchanged during the run. No failed tasks or model seeds were
removed, and no recipe was retuned after observing results.

## Primary budget: 144 candidate executions

Each arm below repeats the same 48 distinct functions with three model
seeds. The 144 attempts are not 144 independent task draws. CPU includes
setup/reset, graph execution, policy updates, trace capture and independent
checking; report compression and provenance hashing are outside this cost.
Shared Python/package import startup is also excluded. The online policy
kernel uses the standard library; process RSS includes package imports.
All failed attempts remain in the totals.

| Control | Accepted /144 | First lane correct /144 | Candidates | CPU s |
| --- | ---: | ---: | ---: | ---: |
| Persistent SNN | 56 | 36 | 15,818 | 1.607 |
| Frozen SNN | 67 | 49 | 14,733 | 1.276 |
| Episode-reset SNN | 68 | 41 | 14,402 | 1.427 |
| Online nonspiking | 45 | 42 | 16,863 | 1.727 |
| Grammar streaming | 69 | 48 | 14,739 | 0.327 |

Persistent SNN accepted 19, 20 and 17 tasks across the three seeds; frozen
SNN accepted 22, 21 and 24, and reset SNN accepted 22, 23 and 23.
Persistent SNN has more accepted tasks and lower measured CPU time than
the nonspiking control here, but fails the required first-choice margin
and performs worse
than the frozen/reset controls. It does not pass the adaptation gate.

Its acceptance across the three phases is 29/48, 12/48 and 15/48 repeated
task-seed attempts. Correct first choices fall from 24/48 to 6/48 and 6/48.
Carryover from earlier conditions may be harmful; the phase data alone do
not establish the cause.

## Complete-grammar budget: 432 candidate executions

Every control accepts all 144 task-seed attempts at this budget.

| Control | Candidates | CPU s |
| --- | ---: | ---: |
| Persistent SNN | 27,300 | 2.720 |
| Frozen SNN | 25,049 | 2.220 |
| Episode-reset SNN | 21,560 | 2.136 |
| Online nonspiking | 26,418 | 2.701 |
| Grammar streaming | 25,128 | 0.626 |

A descriptive secondary observation is that task-local learning with a
reset between tasks uses 14.2% fewer candidate executions than streaming
at full acceptance. Each seed has fewer executions: 7,421, 7,305 and 6,834
versus streaming's 8,376 each. Its CPU cost is nevertheless 3.41 times
streaming's. Fewer executed candidates do not establish a total-work,
device-energy or chip-performance advantage. Persistent SNN uses more
candidate executions and CPU than streaming.

## Verification and scope

The independent auditor replayed all 1,440 episodes, including policy
choices, executed graphs, rewards, features, weight-hash chains, masks,
budgets, work counters, held answers, exact polynomial identity, phase
summaries and gate decisions. It passed. Six corrupted copies of the full
bundle were rejected after their raw checksums were recomputed: altered
features, reward, graph, missing row, work counter and omitted source file.
This tests semantic/provenance checks beyond a checksum comparison.

Local validation passed 35 focused tests in 0.122 seconds and all 409 fast
tests in 236.972 seconds, under the shared builder lock with source hashes
unchanged. The initial red audit-fixture log is retained. Adding its
missing reset-time field corrected the fixture without relaxing the check.
No tests were excluded or skipped. Slow/retry and local full discovery
were not rerun for the additive online path. The existing deterministic
runtime and historical frozen benchmarks remain unchanged.

The new runtime's [CI run](https://github.com/Capsize-Games/rsi-programmer/actions/runs/39160627937)
is pending at this publication checkpoint. It is separate from the
`code-v87` authentication acceptance run tracked in #44. No pending full
job is credited as passing.

This experiment learns only the small readout; the encoder is fixed.
The baseline starts at zero and uses the protocol's 0.05 update rate.
Inputs describe previous action/error and within-task progress, not task
content or real sensors. It is one synthetic operator-prior stream, not
general code synthesis, physical sensing, chip export or an energy test.
Whole-run RSS is a cumulative process maximum, not per-control memory.

The next diagnostic in
[#84](https://github.com/Capsize-Games/rsi-programmer/issues/84) counterbalances
phase order before changing learning parameters. It also calls for an
explicit sensor-feedback and hardware-learning contract. The supplied
paper assessment is in [online-learning-direction.md](online-learning-direction.md).

## Reproduction

From the repository root, use an output directory that does not exist:

```sh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  -m scripts.benchmark_online_graph --output /tmp/rsi-online-new
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  -m scripts.audit_online_graph --bundle /tmp/rsi-online-new
```

The measured bundle is `artifacts/online-graph-v1-20261018/`. It includes
raw compressed episodes, summaries, source manifest, audit, negative-control
report and its reproduction script. Validation logs are in
`artifacts/online-graph-validation-20261018/`.

# Counterbalanced online-learning diagnosis

The original persistent-learning deficit is partly order dependent. Across
all six orders, persistent SNN accepts fewer tasks than frozen and task-reset
SNN, but it does not lose in every order. All three frozen outcome gates fail.

The schedule was frozen in `23a2ad6` / `code-v94` before measurement. The
measured and independently audited source is `f4b97fd` / `code-v95`.
Schedule SHA-256:
`0ffbd7eadac93aa63b8df68d9ba250cd4da5a55a178e8339f9505d81638af3f1`.
The original 48-function corpus and learning recipe remain unchanged.

## Primary budget: 144 candidates

Each arm below contains 864 repeated task/order/seed attempts: the same
48 functions in six orders with three model seeds. These are not 864
independent tasks. CPU includes setup/reset, cold diagnostics, policy,
graph execution, updates and independent acceptance; compression and
provenance hashing are excluded.

| Arm | Accepted /864 | First lane correct /864 | Candidates | CPU s |
| --- | ---: | ---: | ---: | ---: |
| Persistent SNN | 378 | 286 | 91,864 | 9.036 |
| Frozen SNN | 395 | 286 | 88,115 | 7.558 |
| Task-reset SNN | 415 | 294 | 86,144 | 8.466 |
| Online nonspiking | 361 | 295 | 94,651 | 9.731 |
| Grammar streaming | 414 | 288 | 88,434 | 2.000 |

| Phase order | Persistent accepted | Frozen accepted | Reset accepted |
| --- | ---: | ---: | ---: |
| 0-1-2 | 56 | 67 | 68 |
| 0-2-1 | 55 | 65 | 69 |
| 1-0-2 | 65 | 65 | 70 |
| 1-2-0 | 73 | 67 | 69 |
| 2-0-1 | 60 | 66 | 70 |
| 2-1-0 | 69 | 65 | 69 |

Persistent SNN falls below reset acceptance in four orders, wins one and
ties one. Its pooled first-choice count equals frozen and is below reset.
The frozen exploratory follow-up trigger passes; the stricter descriptive
label of underperformance in every order fails. This is neither statistical
significance nor proof that carryover, dense reward, missing context or
update exposure caused the deficit. Task content is paired; RNG offsets
are recorded because different episode lengths change later random draws.

## Complete grammar: 432 candidates

Every arm accepts all 864 attempts. Persistent SNN uses 148,472 candidates,
frozen 151,270, reset 129,317, nonspiking 149,967 and streaming 150,768.
Reset uses 14.2% fewer candidates than streaming and has fewer candidates
in all 18 order/seed runs. Its CPU is 12.865 versus 3.616 seconds, 3.56
times streaming. Persistent saves 1.5% pooled candidates, wins that work
comparison in 11/18 runs, and uses 14.608 seconds CPU, 4.04 times streaming.
These secondary candidate-count observations do not establish a total-work
or hardware-energy advantage. All adaptation, SNN-specific and streaming
gates remain false.

## Verification and next experiment

All 8,640 episodes completed with stable source hashes. The independent
auditor replayed every choice, candidate, observed reward, update, weight
chain, mask, stopping decision, held check and polynomial identity. It also
verified matrix completeness, phase-position metadata, cold diagnostics,
RNG offsets, summaries and CPU/gate calculations.

Seven semantic/provenance controls were rejected: four row-level changes
(entropy, reward, phase position and RNG offset), an omitted source pin,
altered summary CPU, and a full bundle missing its last row after its raw
checksum was recomputed. The original order exactly matches all 1,440
pilot rows in candidate traces, acceptance and weight hashes, excluding
timing and the new diagnostic fields.

Descriptively, failed episodes contribute 76.2% of updates; accepted
episodes average 57.9 candidates versus 144 for failures. Of all updates,
99.6% follow a nonexact candidate. Wrong-root candidates receive positive
raw reward 92.2% of the time; positive raw reward is not the same as a
positive baseline-centered update. Mean cold maximum probability rises
from 0.635 in the first phase position to 0.802 and 0.866. These observations
leave reward alignment, update exposure and confidence carryover as
mechanism hypotheses, not established causes. Exact diagnostics and baseline
reconstruction are retained in `diagnosis.json`.

The policy still receives the same initial six features on every task and
no current observed examples. A separate, prospectively frozen experiment
will provide only those six observed input-output triples to an additive
context adapter. This changes its information set; any improvement cannot
be attributed to a learning-rule change or to feedback-only adaptation.
The original kernel, pilot, frozen corpus and results remain intact.

Local validation: 46 affected tests passed in 0.190 seconds and all 420
fast tests passed in 234.598 seconds, with source hashes unchanged. No skips
or exclusions. All work ran under the builder lock. Slow/retry and local full
discovery were not rerun for this additive diagnostic. The measured source's
[CI run](https://github.com/Capsize-Games/rsi-programmer/actions/runs/39164398208)
is separate from the code-v87 and code-v90 runs; pending full jobs are not
credited as passing.

Reproduce from `code-v95`, using a new output directory:

```sh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  -m scripts.benchmark_online_counterbalanced --output /tmp/rsi-cb-new
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  -m scripts.audit_online_counterbalanced --bundle /tmp/rsi-cb-new
```

The bundle is `artifacts/online-counterbalanced-v1-20261018/`. Exact source
pins and raw SHA-256 are in its manifest. Later source additions require a
checkout of the measured source for complete-path-set provenance checks.
Issue [#84](https://github.com/Capsize-Games/rsi-programmer/issues/84) remains
open for mechanism work and the sensor-feedback/hardware contracts. No real
sensor, chip, deployment or energy benefit has been measured.

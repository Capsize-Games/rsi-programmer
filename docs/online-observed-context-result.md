# Observed-example context result

Giving the policy the current observed examples did not resolve the
persistent-learning regression. All adaptation, SNN-specific and streaming
gates fail in every phase order. The adapter remains an opt-in experiment.

The protocol was frozen in `ff9a0f3` / `code-v97` before measurement:
SHA-256 `d8897dc237ceed1a8781dd2dd828b4507842bbdf53473558edffa928fe6485fb`.
The corpus, six phase orders, three seeds, budgets and learning rule were
preserved. All four neural controls received the same fixed observed-context
projection; grammar streaming remained context free. All 8,640 episodes
completed with stable execution-source hashes.

## Outcomes

Each arm contains 864 repeated task/order/seed attempts per budget: the same
48 functions in six orders and three seeds. These are not independent task
draws. CPU includes setup/reset, context preparation, cold diagnostics,
execution, policy updates and held checking; output compression, provenance
hashing and import startup are excluded. The separately reported fixed-code
capacity study is included in summed experiment CPU, not per-arm gates.

| Budget | Arm | Accepted /864 | First lane correct | Candidates | CPU s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 144 | Persistent SNN | 390 | 290 | 91,141 | 9.399 |
| 144 | Frozen SNN | 395 | 286 | 88,115 | 8.031 |
| 144 | Task-reset SNN | 425 | 266 | 86,232 | 9.010 |
| 144 | Online nonspiking | 364 | 306 | 94,490 | 9.902 |
| 144 | Grammar streaming | 414 | 288 | 88,434 | 1.938 |
| 432 | Persistent SNN | 864 | 275 | 146,739 | 15.033 |
| 432 | Frozen SNN | 864 | 284 | 151,270 | 13.805 |
| 432 | Task-reset SNN | 864 | 303 | 130,170 | 13.660 |
| 432 | Online nonspiking | 864 | 282 | 149,036 | 15.793 |
| 432 | Grammar streaming | 864 | 288 | 150,768 | 3.507 |

The earlier no-context persistent arm accepted 378 attempts, versus 390
here, and made 286 correct first choices versus 290. This is a descriptive
comparison between separate runs on the same repeated functions. The new
arm still trails frozen acceptance by five, reset by 35, and streaming by
24. Its first-choice margin over frozen is 0.46 percentage points, far below
the frozen 10-point gate. All six orders fail every outcome gate.

At complete-grammar budget, persistent SNN saves 2.7% pooled candidates
against streaming but costs 4.29 times its CPU. Reset saves 13.7% candidates
but costs 3.89 times its CPU. The original no-context reset observation
(14.2% fewer candidates) remains valid for its own source. No SNN-specific,
total-work or hardware-energy advantage is established here.

The exploratory diagnosis flag is false because persistent first-choice
count now exceeds frozen. That flag is not an acceptance gate: it does not
mean the acceptance deficit or persistent-learning problem has been fixed.

## Fixed-encoder capacity

The predeclared cold-code diagnostic found 10, 6 and 8 distinct spike-rate
vectors over the 48 contexts for seeds 0, 1 and 2. The continuous control
produced 48 distinct vectors for each seed. Counts round rates to 12 decimal
places, as frozen before measurement. No seed, projection or scale was
changed after observing this result.

This demonstrates cold-code aliasing under this fixed representation. It
does not prove the adaptive policy cannot use context, identify a cause for
the regression, or bound its performance when weights and search history
change. Dense reward, variable update exposure and encoding remain separate
mechanism questions. Adding a larger memory module is not supported by this
comparison; a further mechanism test needs its own frozen matched controls.

## Verification and auditor correction

The first audit failed before replay because its protocol guard expected an
obsolete metadata key. Commit `be79fe9` / `code-v98` fixes the guard to the
actual frozen answer-metadata flag and opaque episode-ID contract, and adds
a regression test loading the actual pinned fixture.

The measured runtime, protocol, raw episodes, manifest and numerical
summaries were not rewritten or remeasured. Corrected verification checks
the immutable `code-v97` source archive and separately requires every current
execution-source hash to match, except the auditor script and its test.
Both changed analysis files are hashed separately in `auditor-repair.json`.
That receipt records full replay of all 8,640 episodes, context preparation,
policy choices, feedback, updates, hashes, budgets, held checks, polynomial
identity, capacity codes, CPU accounting and all gates. It passed.

Eight corruption controls were rejected after an unaltered first-row
positive control: input hash, projection hash, context work, cold probability,
context CPU, capacity code, charged summary CPU, and an omitted phase order.
Capacity and raw checksums were recomputed for the corresponding corrupted
copies, so those rejections test semantics beyond checksum agreement.

Final local validation passed 62 affected tests in 0.204 seconds and all
436 fast tests in 235.965 seconds, with stable source hashes and no skips or
exclusions. Earlier logs and the failed audit are retained. Tests and audits
ran serially under `/tmp/rsi-programmer-builder.lock`. Slow/retry and local
full discovery were not rerun for this additive path. Real hub checks and
all existing tests remain intact. The corrected source's
[CI run](https://github.com/Capsize-Games/rsi-programmer/actions/runs/39166387866)
is separate from the measured runtime's
[run](https://github.com/Capsize-Games/rsi-programmer/actions/runs/39165989858);
pending jobs are not credited as passing.

## Reproduction and remaining scope

For a fresh comparison from the corrected source, use a new output folder:

```sh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  -m scripts.benchmark_online_observed_context --output /tmp/rsi-context-new
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  -m scripts.audit_online_observed_context --bundle /tmp/rsi-context-new
```

The published bundle is `artifacts/online-observed-context-v1-20261018/`.
To verify those original measurements with the corrected auditor, extract
only the measured sources, then use the retained proof wrapper:

```sh
mkdir /tmp/rsi-code97-source
git archive code-v97 rsi_programmer tests scripts fixtures \
  requirements.txt justfile .github/workflows/ci.yml \
  | tar -x -C /tmp/rsi-code97-source
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  artifacts/online-observed-context-v1-20261018/verify-repaired-audit.py \
  "$PWD" /tmp/rsi-code97-source \
  "$PWD/artifacts/online-observed-context-v1-20261018"
```

The implemented observation contract is exactly six current integer
`(x, y, target)` triples. The targets are observed task specifications;
they are not unlabeled sensor events. Opaque episode IDs only guard boundary
calls and never enter encoding or learning. A physical calibration workload
would need available reference targets or another measurable outcome signal.
No sensor, chip, integer hardware implementation or device-energy measurement
has been established. Issue
[#84](https://github.com/Capsize-Games/rsi-programmer/issues/84) remains open
for those contracts and the unresolved learning mechanisms.

# Audit amendments and validation

This report describes code-v45. See the [audit follow-up](audit-followup.md)
for subsequent fixes and validation.

Final evidence is published as [code-v45](https://github.com/Capsize-Games/rsi-programmer/releases/tag/code-v45).
The validated runtime/test inputs are unchanged from implementation
commit [`0fcf868`](https://github.com/Capsize-Games/rsi-programmer/commit/0fcf8689b9e26a872230b89741902aecd98eba48).
These amendments retain the optimizer rebinding introduced in `5c6701b`.
They correct independent scoring, feature encoding, acceptance, work
accounting, calibration, and experiment persistence. The issue table below
distinguishes fixes from remaining research and implementation limits.
The [historical adversarial review](audit-review-5c6701b.md) contains
issue-by-issue reproductions and the three formula checks at `5c6701b`.

## Validation

| Command | Result | Verdict | Log |
|---|---|---|---|
| `just test-slow` | Ran 64 tests in 14217.561s | OK | [slow-final.log](../artifacts/validation-20261017/slow-final.log) |
| `just test-retry` | Ran 3 tests in 662.235s | OK | [retry-final.log](../artifacts/validation-20261017/retry-final.log) |
| `just test-fast` | Ran 222 tests in 276.362s | OK | [fast-final.log](../artifacts/validation-20261017/fast-final.log) |

[Validation manifest](../artifacts/validation-20261017/manifest.json) records the unchanged input hash and environment. Absolute local paths in logs are replaced by placeholders; the original logs remain in the local audit directory. The local spikeforge checkout was clean at `2981b28105a0d9a8ddcb76ae4e21a4358a1d96f6`.

Both fast-matrix jobs now install through `just setup` with the absolute SpikeForge checkout root as a positional argument. This fixes the literal `SPIKEFORGE_DIR=...` path passed by the fast job and makes the dependency available to scrubbed subprocesses in the full job. Fresh GitHub runners passed setup and the fast suite (Python 3.12: 222 tests in 329.066s; Python 3.13: 222 tests in 422.996s), including sandbox and proxy CLI regressions. These jobs ran commit `0fcf868`, whose 96 runtime/test/fixture inputs match the final local validation hash. No redundant local fresh-env setup check was run. [Python 3.12 setup/test log](../artifacts/validation-20261017/ci-fast-312-public.log), [Python 3.13 setup/test log](../artifacts/validation-20261017/ci-fast-313-public.log), and [CI manifest](../artifacts/validation-20261017/ci-matrix-validation.json). The mirrored CI logs include only setup and test sections, with runner paths replaced and trailing whitespace removed; checkout and cleanup sections are excluded. These records cover the fast matrix; the automatic full jobs have their own status on the linked workflow run.

The slow tests now compare candidate-signature work across search paths.
The operand suite's old 9-of-24 win set used incompatible `tried` units
and the earlier feature encoding, so it is retained as historical
evidence rather than enforced as a current baseline. Current tests keep
oracle checks, operand-feature variance and readout-dependence checks,
and report all 24 comparisons. Zero-budget fallback now must stop with
zero examinations. Passing these contracts does not establish a speedup.

[The first slow attempt](../artifacts/validation-20261017/slow-attempt-1.log) exposed test setup that treated skipped
children as measured candidates. That setup now selects measured entries
and separately asserts that skipped fitness stays unset. That attempt was interrupted before completion. [A second attempt](../artifacts/validation-20261017/slow-attempt-2.log) was
interrupted to fix imputation of unmeasured loop gains. [A third attempt](../artifacts/validation-20261017/slow-attempt-3.log) was interrupted after
57 reported passes when a read-through found repeated-run ledger IDs in
the next scale path. Focused regressions reproduced both caller failures;
PolicyLoop and ScaleUp now create a fresh namespace for every invocation.
None of the interrupted attempts is counted as a passing lane.
The added same-instance repeat-run tests failed against the preceding
runtime with two `LineageError` exceptions ([red log](../artifacts/validation-20261017/repeat-run-red-confirmed.log),
[command](../artifacts/validation-20261017/repeat-run-red-confirmed.command)).
After the caller fix, all 14 integrity tests passed in 0.827s
([green log](../artifacts/validation-20261017/repeat-run-integrity-confirmed.log),
[command](../artifacts/validation-20261017/repeat-run-integrity-confirmed.command)).
No tests were excluded from the final local lanes, which ran sequentially.
No separate local full `unittest discover` run was made. A five-second
profiler sample during the third interrupted attempt identified repeated
stash maximum scans and full-graph validation as costs. The final local
lanes were run without that profiler. Scored backtracking now sorts once
per layer; differential regressions preserve the preceding behavior.

## Issue disposition at code-v45 (historical)

This table records code-v45 and is retained as historical evidence.
Later fixes and current validation are in the [audit follow-up](audit-followup.md);
the statuses and implementation limits below do not describe the current runtime.

At code-v45, all 21 issues had been reviewed. Fifteen were closed, including
#51, which was already closed before those amendments. Six remained open:
#53, #54, #56, #65, #66, and #70.

| Issue | Amendment | Remaining limitation |
|---|---|---|
| [#51](https://github.com/Capsize-Games/rsi-programmer/issues/51) | Rebind all three optimizers; preserve recipient Adam options, optional donor options, and lattice freezes; actual fit-after-restore coverage. | Restore resets Adam moments and counters; it copies children rather than resuming exact training. |
| [#52](https://github.com/Capsize-Games/rsi-programmer/issues/52) | Independent ModularPolicy drives start fresh; continuation is explicit; within-drive feedback remains. | Sequence callers must request `continue_state=True`. |
| [#53](https://github.com/Capsize-Games/rsi-programmer/issues/53) | Separate signature, operand, examination, and commit counts; enforce labeled budgets; prune returned graphs; include retries. | Beam budgets cap commits, not signature work; eager layer materialization and costly backtracking/full-graph validation remain; historical gains artifacts were not regenerated and the probe tuple omits its examination count. Open. |
| [#54](https://github.com/Capsize-Games/rsi-programmer/issues/54) | Document lower-layer incompleteness and record task-generator selection filtering. | Equivalence merging can erase held-correct alternatives; fixtures remain conditioned on the enumerator. Open. |
| [#55](https://github.com/Capsize-Games/rsi-programmer/issues/55) | Use `layer-ordinal-v1` across training and inference; serialize classical bonuses and apply them once. | Untagged old snapshots have an unverified feature contract; retrain for verified current encoding. |
| [#56](https://github.com/Capsize-Games/rsi-programmer/issues/56) | Describe the six aggregate head losses, constant-bit squared error, and fitted STOP accurately. | Mixed-class margins remain 1.0 on rates in [0,1], with loss floor ln(2). Open. |
| [#57](https://github.com/Capsize-Games/rsi-programmer/issues/57) | Iterate feedback-aware calibration and report final rates, convergence, and post-fit drift. | Bounded calibration can fail to converge; reports expose that result. |
| [#58](https://github.com/Capsize-Games/rsi-programmer/issues/58) | Split whole groups before fit/tuning; report calibration joint commit accuracy and separate evaluation metrics; preserve threshold ties. | Split independence is by contrastive group, not task or seed. |
| [#59](https://github.com/Capsize-Games/rsi-programmer/issues/59) | State iid paired-task bootstrap assumptions, require minimum evidence, reject invalid numerics, average repeat generations per task. | SelfGate's score-only trainer interface cannot measure safety failures or token spend. |
| [#60](https://github.com/Capsize-Games/rsi-programmer/issues/60) | Preserve measured refused gains and exclude invalid evidence; label outcome association as descriptive; flag singleton agency; reject weight-dimension mismatches. | These summaries establish no causal autonomy or singleton convergence. |
| [#61](https://github.com/Capsize-Games/rsi-programmer/issues/61) | Restrict the default CLI to five consumed trainer fields. | Generic callers retain the wider alphabet; consumed fields need not change an eager-work score. |
| [#62](https://github.com/Capsize-Games/rsi-programmer/issues/62) | Skipped children have null fitness and a separate provisional parent score; measured selection excludes them. | Consumers of the archive must distinguish measured from skipped records. |
| [#63](https://github.com/Capsize-Games/rsi-programmer/issues/63) | Retain raw components alongside feasibility gating; normalize recovery per task; check held correctness; flag each family's regression. | The documented hard selection gate still removes partial ordering among infeasible candidates. |
| [#64](https://github.com/Capsize-Games/rsi-programmer/issues/64) | Label task/family training-row adaptation and fingerprint supported behavior settings and author weights. | Refined evaluation is task-adapted, not zero-adaptation transfer. |
| [#65](https://github.com/Capsize-Games/rsi-programmer/issues/65) | Distinguish own-channel row objectives from the opcode argmax diagnostic; document scalar operand scoring and recurrent STOP influence. | No direct opcode-choice supervision or objective-matched per-head ranking report was added. Open. |
| [#66](https://github.com/Capsize-Games/rsi-programmer/issues/66) | State canonical-trajectory imitation semantics; exclude exact task solvers from rich negatives. | Alternative completing intermediates may still be negative; no bias-bounding ablation was run. Open. |
| [#67](https://github.com/Capsize-Games/rsi-programmer/issues/67) | Reject malformed words/trailing material, normalize VM underflow and oracle failures, require Int final outputs, allow typed Bool intermediates explicitly. | In-range reference-word modulo resolution and zero-initialized VM slots remain ISA rules. |
| [#68](https://github.com/Capsize-Games/rsi-programmer/issues/68) | Use configured seed offsets; select training groups by task ID and verify the mapping. | Future changes to trace granularity must satisfy the explicit mapping contract. |
| [#69](https://github.com/Capsize-Games/rsi-programmer/issues/69) | Use insert-only history, per-invocation caller IDs, exact inserted digests, digest readback checks, and serialized cross-thread access. | This is an experiment ledger, not an authority or capability-security database. |
| [#70](https://github.com/Capsize-Games/rsi-programmer/issues/70) | Support named classical-bonus ablations and test the exact-flag mechanism separately from neural features. | No paired real author/beam ablation of exact_flag and gap was measured. Open. |
| [#71](https://github.com/Capsize-Games/rsi-programmer/issues/71) | Recount rebuilt expression operations consistently, including select children. | No downstream historical feature corruption was established by the audit. |

The missed held-out-acceptance and NaN-promotion defects are also corrected.
Default and scale selection independently check returned programs, and
headless oracle failures are reported distinctly from accepted programs.

## Independent rank measurement

Mean first-solver position within the solving layer (lower is better), averaged over seeds 0, 1, and 2:

| Family | Tasks | Probe order | Exact random | Untrained | 5 epochs | 40 epochs | 50 epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
| shadow | 4 | 640.75 | 912.60 | 848.58 | 125.42 | 20.92 | 17.25 |
| novel | 3 | 828.00 | 521.43 | 700.89 | 166.11 | 350.56 | 390.11 |
| plain OOD | 8 | 482.50 | 830.29 | 894.71 | 483.08 | 801.46 | 831.12 |

At 50 epochs, shadow ordering improves strongly. Novel ordering beats the exact random expectation in the three-seed mean; plain OOD is approximately random and depends on seed and checkpoint. These measurements do not support a general claim of worse-than-random transfer.

All 180 first-solver positions also pass the held-out expectations, and no complete training-row table overlaps between shadow and either transfer family.

| Family | Guided candidate signatures | Eager probe signatures | Streaming signatures | Guided / streaming |
|---|---:|---:|---:|---:|
| shadow | 14673 | 14673 | 4573 | 3.21 |
| novel | 8685 | 8685 | 3663 | 2.37 |
| plain OOD | 23478 | 23478 | 7061 | 3.33 |

Direct instrumentation checks generation order, a fixed uniform permutation, untrained seed 0, and trained seed 2 at epoch 50. Their candidate signature counts are identical on every task. Guided search also evaluates operand vectors in addition to the candidate counts above.

[Raw positions](../artifacts/ranker-20261017/positions.csv), [summary](../artifacts/ranker-20261017/summary.json), [work instrumentation](../artifacts/ranker-20261017/work.json), and [source and fixture manifest](../artifacts/ranker-20261017/manifest.json) come from `scripts/measure_ranker.py` on the final validated source. CSV line endings are normalized to LF; values are unchanged. The manifest training-fixture path is repository-relative.

Measurement source SHA-256: `0da3421684649a88bcf013841f8a75fda08771708188e6834550ea8e5afdaa60`.

The recipe fits shadow final-layer train-output-equality groups, distinct
from `train_author`'s canonical-trajectory imitation. It uses three seeds,
all 4 shadow, 3 novel, and 8 plain OOD tasks, and fixed 5/40/50
epoch checkpoints. Exact random expectation is `(N + 1) / (K + 1)` for
`K` valid candidates in a uniform permutation of `N` candidates.

Candidate signatures are root-vector evaluations, excluding recursive
subtree calls, neural work, graph validation, and stash operations. The shipped enumerator builds each full
layer before checking it; the independent streaming comparator stops on
the first generated fit. These are distinct comparators. No controlled
end-to-end runtime advantage is claimed.

Small enumerator-filtered families and three seeds do not establish
population-wide transfer performance. Canonical imitation and task-adapted
author results do not establish neural authorship of candidate generation
or exact evaluation.

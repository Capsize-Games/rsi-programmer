# Audit progress checkpoint

Snapshot: 2026-10-18T06:42:49.236645+00:00. The 96 runtime/test/fixture inputs still match
implementation [`0fcf868`](https://github.com/Capsize-Games/rsi-programmer/commit/0fcf8689b9e26a872230b89741902aecd98eba48).
SHA-256: `2a6b6ebbaabbd95e3ed77f467cc359ebf798fb9907794612db423b5a25b79d94`.

## Slow validation passed

`just test-slow`: **64 tests passed in 14,217.561s**. The completed lane
includes the shared-instance scale `run()` test that repeats evolution,
both scale acceptance paths, and the transfer checks.
[Exact log](../artifacts/validation-slow-20261018/slow-final.log) and
[source/timing manifest](../artifacts/validation-slow-20261018/manifest.json).

`just test-retry` started after the successful slow lane and is currently
running. The independent three-seed measurement on the final sources
follows retry. Neither pending result is claimed complete. The automatic
full CI jobs are separate from the completed fast matrix; no separate
local full discovery run was invoked.

## Completed evidence

- [Historical adversarial review at 5c6701b](audit-review-5c6701b.md): all 21 issues examined, with reproductions, formula checks, and a three-seed baseline experiment. Eleven issues were confirmed as filed and ten were overstated; all retained a narrower reproduced observation.
- [Local fast log](../artifacts/validation-checkpoint-20261018/fast-final.log): 222 tests passed in 276.362s.
- [Python 3.12 CI fast log](../artifacts/validation-checkpoint-20261018/ci-fast-312-public.log): 222 tests passed in 329.066s.
- [Python 3.13 CI fast log](../artifacts/validation-checkpoint-20261018/ci-fast-313-public.log): 222 tests passed in 422.996s.
- The two new same-instance repeat-run tests first reproduced two `LineageError` exceptions ([red log](../artifacts/validation-checkpoint-20261018/repeat-run-red-confirmed.log)); after the caller fix, 14 integrity tests passed in 0.827s ([green log](../artifacts/validation-checkpoint-20261018/repeat-run-integrity-confirmed.log)). Fitting and fitness are small deterministic stubs in those focused tests; the ledger and high-level run methods execute normally.


## Earlier partial runs

The [previous checkpoint](../artifacts/validation-checkpoint-20261018/manifest.json)
records the 06:19 UTC snapshot with 58 partial slow passes. Its partial
log and three interrupted attempts remain available in that artifact
folder; none was counted as a passing lane. The historical baseline
measurement describes 5c6701b and is separate from the pending final run.

## Issue disposition

All issues [#51](https://github.com/Capsize-Games/rsi-programmer/issues/51)–[#71](https://github.com/Capsize-Games/rsi-programmer/issues/71) have checkpoint updates. Fifteen are closed, including
[#51](https://github.com/Capsize-Games/rsi-programmer/issues/51), which was already closed before these amendments. Six remain open:
[#53](https://github.com/Capsize-Games/rsi-programmer/issues/53), [#54](https://github.com/Capsize-Games/rsi-programmer/issues/54), [#56](https://github.com/Capsize-Games/rsi-programmer/issues/56), [#65](https://github.com/Capsize-Games/rsi-programmer/issues/65), [#66](https://github.com/Capsize-Games/rsi-programmer/issues/66), and [#70](https://github.com/Capsize-Games/rsi-programmer/issues/70). Those remaining limits are not claimed fixed.

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

# Sealed construction-v1 final result (issue #89)

One-shot sealed evaluation run 2026-11-05 under the preregistered
gates in [construction-v1.md](construction-v1.md). Artifact:
`artifacts/construction-final-v1-20261105/report.json` (plus run
logs). Frozen receipt:
`artifacts/construction-early-dev-20261101/epochs-100.json`
(sha256 recorded in the report; the 100-epoch recipe selected on
the development split). Corpus hash `1771da26...` and the three
pinned generator sources verified before inference. Seed 0's fresh
fit reproduced the development fit bit-for-bit (initial and final
parameter hashes identical), so the frozen recipe is what ran.

## Verdict

**Overall: FAIL.** Learning gate passed on every seed; cost gate
failed on every seed. Per the protocol, learning without a cost
win is described separately, and this failure does not authorize
scaling or a new final-set tuning pass.

| Seed | Trained @256 | Untrained @256 | Random @256 | Margin (min) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 94/244 | 0 | 2 | +0.377 |
| 1 | 107/244 | 2 | 1 | +0.430 |
| 2 | 119/244 | 0 | 3 | +0.475 |

Learning gate (each seed beats untrained and random by >= 0.10 at
cap 256): **passed**, by 3.8x-4.9x the required margin. Seed
variance is material (94 to 119 solves) and final training loss
does not rank seeds by solves (losses 0.256/0.040/0.343 vs solves
94/107/119).

Cost gate (full coverage at <= 0.90x streaming per-task CPU with
fit amortized over 244): **failed on missing full coverage.**
Trained beam plateaus at its cap-256 count at every higher cap
(the stop action ends search; extra root budget changes nothing),
while streaming reaches 244/244 at cap 16384 in 1.90 CPU seconds
total. Fit alone cost 55.8-57.1 CPU seconds per seed. There is no
cap at which the compared configurations both cover fully, so no
ratio exists to pass.

Streaming sweep for reference: 7/244 @64, 29 @256, 56 @1024,
210 @4096, 244 @16384. Whole run: 327.4 worker CPU seconds,
391,080 KiB peak RSS, single-threaded under the builder lock on a
loaded shared machine (figures are process CPU, single
observation; the verdict does not turn on timing noise because
the cost gate fails on coverage, not on the ratio).

## Interpretation

This closes the construction-v1 question as preregistered. The
ANN constructor genuinely learns: on a sealed, function-disjoint
set it constructs 39-49% of programs against ~0% for untrained
and random controls, on all three seeds. It is also decisively
not competitive with trivial enumeration, which solves everything
for less CPU than one fit. Both halves replicate the development
pattern (78/135 vs streaming 135/135) at sealed scale, so the
development numbers were not split-specific luck in either
direction. No SNN arms were run: the gates are ANN-only, SNN
reporting was dropped with the SNN-advantage line in the
2026-11-05 audit, and this omission is recorded here rather than
silently absorbed.

The 244 functions are now development data. Any future
construction claim needs a new sealed bank.

## Failed first invocation (retained)

The first invocation on 2026-11-05 crashed before any model
inference: the new runner read `splits["final"]`, but the frozen
corpus names the sealed split `"eval"` (generator `QUOTAS`). No
task was evaluated and no result was observed; the empty output
directory was removed and the traceback kept as
`run89-failed-attempt.log` in the artifact directory. The fix
(90c663e) renamed the split via a `FINAL_SPLIT` constant derived
from the pinned generator source, added a regression test tying
the name to `QUOTAS`, and the corrected invocation above is the
single one-shot run. No recipe, gate or budget changed between
invocations.

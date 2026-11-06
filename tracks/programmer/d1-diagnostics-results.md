# D1 diagnostics result (development data, 2026-11-06)

Training-set size on shape-broad tasks, per the diagnostic plan
in [hypothesis-survey-2026-11-06.md](hypothesis-survey-2026-11-06.md)
and [d0-diagnostics-results.md](d0-diagnostics-results.md).
Artifact: `artifacts/construction-d1-diagnostics-20261106/`
(`diagnose.py`, `results.json`, `run.log`). Nested shape-broad
train pools (1,200 vs 4,800 tasks, even over strata 6/7/8;
level 8 block-balanced), one disjoint unseen broad eval set
(300 tasks), everything disjoint from the sealed hard corpus.
Two fits, seed 0, 100 epochs. Development evidence only —
design input for a preregistered test, not a capability claim.

## Result: data scale is a live lever

Unseen shape-broad eval, 300 tasks (100 per stratum):

| Arm | Solves | Rate | Eval CPU s | Fit CPU s |
| --- | ---: | ---: | ---: | ---: |
| 1,200 tasks @ width 8 | 2 | 0.7% | 25.9 | 425 |
| 1,200 tasks @ width 32 | 12 | 4.0% | 96.0 | 425 |
| 4,800 tasks @ width 8 | 24 | 8.0% | 25.0 | 1,758 |
| 4,800 tasks @ width 32 | 51 | 17.0% | 89.9 | 1,758 |
| Streaming @ 0.5 s/task | 42 | 14.0% | 165.1 | 0 |

- 4x training data gives 12x solves at matched width 8
  (2 → 24), spread across all three strata (13/6/5 at width 8;
  29/14/8 at width 32). This clears the decision rule set for
  D1 (material = ≥3x relative) by a wide margin.
- At width 32 the 4,800-task model solves more unseen broad
  tasks than streaming (51 vs 42) at lower evaluation CPU
  (89.9 vs 165.1 s) — the first such observation on an unbiased
  task distribution in this project. With fit charged at this
  300-task volume the learned total (1,848 s) still loses to
  streaming (165 s) by 11x; nothing here overturns any sealed
  verdict.
- Marginal economics on this dev set now favor the learned arm
  (width 8: 0.96 solves per marginal CPU second vs streaming's
  0.254), giving a retrospective break-even near ~7,600 tasks
  at width 8 (~4,800 at width 32). Retrospective arithmetic on
  dev data: motivation for a preregistered design, not a win.
- Training loss remains a poor proxy: 1.158 → 1.061 while
  solves went 2 → 24.

## Secondary answers

- **Width on unseen tasks (hypothesis C headroom):** width 32
  lifts solves substantially (2→12 small, 24→51 large), so
  search truncation loses real solves on unseen tasks too. But
  plain wider beam captures much of it; an execution-guided
  decoding variant would have to beat width-matched baselines.
  C stays secondary to the scaling line.
- **Feedback density (hypothesis B):** unseen-task acceptance
  equals the solve rates above (8-17% for the large model), so
  rollout harvesting is mechanically viable on broad tasks at
  this scale. It remains dominated while teacher traces are
  free and data scaling works: B's only candidate edge is
  on-policy state distribution. Parked behind scaling.
- **Small-pool broad baseline:** 2/300 confirms D0's picture —
  the historical 1,200-task recipe is near-zero on broad
  distributions; earlier double-digit rates were distribution
  artifacts (prefix stratum 8).

## Caveats

Single seed; dev eval n=300 (the 51-vs-42 margin over streaming
is within ~1-2 sigma and per-task rows were not saved, so no
paired analysis); the eval set will be reused for D2 design
measurements and must not serve any final claim. Fit cost
scales ~linearly with pool size (425 s at 1,200; 1,758 s at
4,800), so the scaling question is exactly whether solve growth
outruns the linear fit cost at a declared volume.

## Next: D2 third scaling point, then preregistration

One more dev fit at 19,200 tasks (nested superset; the same 300
eval identities stay excluded from training), widths 8/32,
per-task rows saved. Three scaling points (1.2k/4.8k/19.2k)
then fix the sealed design: train-pool size, beam width,
declared evaluation volume, streaming operating point, and a
kill criterion symmetric to previous tests — if the learned arm
does not beat streaming on solves at equal total CPU at the
declared volume on a fresh shape-broad sealed bank, the
data-scaling route closes and the exhaustion verdict follows.

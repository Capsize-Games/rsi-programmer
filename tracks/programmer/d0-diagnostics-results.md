# D0 diagnostics result (development data, 2026-11-06)

Diagnostics designed in
[hypothesis-survey-2026-11-06.md](hypothesis-survey-2026-11-06.md)
to gate three candidate hypotheses. Artifact:
`artifacts/construction-d0-diagnostics-20261106/` (`diagnose.py`,
`results.json`, `run.log`). One fresh imitation fit (seed 0, 100
epochs, hard-corpus train pool, 506 CPU s). The fit reproduced
the sealed run's seed-0 model bit-for-bit (final hash
`04649380...`, final loss 0.7593051791191101), so every number
below describes the same controller that scored 26/300 on the
sealed eval.

## D0-b: hypothesis A's premise is void

Unseen stratum-8 identities, two sets of ~60, learned arm at beam
width 8, streaming at the sealed 0.5 CPU s/task:

| Set | Learned | Streaming |
| --- | ---: | ---: |
| Prefix-unseen (corpus's own pool region) | 12/60 (20%) | 0/60 |
| Shape-broad (even over left-subtree sizes 0-7) | 1/64 (1.6%) | 0/64 |

The corpus's whole stratum-8 pool (2,264 identities) came from
the first 2,900 enumeration triples: witnesses of the form
`x <op> (earliest level-7 nodes)`. Train and eval stratum-8 tasks
were identity-disjoint but drawn from that one shape family. The
controller generalizes within the family (20%, matching the
sealed 23.3/100 stratum-8 mean) and collapses outside it (1.6%).

Consequence: the sealed learned rate (9.6%) was carried by this
stratum. On a shape-broad bank the learned rate would be roughly
(5.3% + 0% + 1.6%) / 3 ≈ 2.3%, at which the learned arm loses to
streaming even on marginal cost (0.023/0.086 ≈ 0.27 solves per
marginal CPU second vs streaming's 0.41), before fit is charged.
No break-even volume exists. **Hypothesis A (amortization at
deployment volume) is dead on its premise; the bank is not
built.** This is the cheap kill D0 existed to deliver.

## D0-a: acceptance density was a recipe artifact, on train tasks

Rollout acceptance of the 100-epoch model on its 1,200 training
tasks: stratum 6: 178/400 (44.5%), 7: 101/400 (25.3%), 8:
289/400 (72.3%). Final-graph observed-match is all-or-nothing
(the count matching all observed rows equals the accepted count
at every stratum); there is no partial-credit gradient in final
graphs.

Contrast with the sealed feedback loop's ~1/1,200: that loop's
rounds each refit for only 30 epochs from reset parameters. The
sealed verdict stands for the protocol as preregistered, but the
sparsity it measured was substantially a property of the
30-epoch round recipe, not of expert iteration as such.

Caveat that limits the forward implication: D0-a measures
acceptance on tasks the model was trained on (47% overall vs
8.7% on sealed eval for this same model — a wide memorization
gap). The density relevant to feedback is acceptance on tasks
without teacher traces, which D0-b and the sealed strata put at
~20% within a trained shape family and ~0-2% outside it. Teacher
traces are in any case free from the catalog for any identity,
so a feedback variant must beat matched-compute imitation
through on-policy state distribution alone. Hypothesis B is
therefore re-scoped, not voided: mechanically viable, narrow
expected edge, low prior.

## D0-c: width buys solves at worse efficiency (train sample)

120 train tasks (40/stratum): width 8 solves 60 in 9.9 CPU s,
width 32 solves 82 in 35.1 s, width 128 solves 89 in 122.9 s.
Solves are lost to search truncation on train tasks, with most
of the recoverable gain by width 32. This does not measure
unseen-task headroom; at 1/64 on shape-broad tasks (width 8),
hypothesis C's ceiling there looks low regardless. C stays
unscoped pending an unseen-task width measurement (folded into
D1 below).

## Streaming caps: 0.5 s was a fair operating point

One 2 s pass over the same 120 train tasks, fractions derived
per budget. Stratum 6: 12% @0.25 s, 62% @0.5 s, 82% @1 s, 100%
@2 s. Stratum 7: 0% until 2 s (15%). Stratum 8: 0% at all caps.
Mixed-bank CPU per solve: 2.22 @0.5 s (sealed run: 2.21), 3.06
@1 s, 3.94 @2 s. Raising streaming's cap buys stratum-6/7 solves
at worse efficiency; the sealed 0.5 s budget did not
disadvantage the classical arm.

## What the diagnostics jointly establish

The binding constraint is shape-level generalization, now
measured at three points on complete or broad sampling: 5.3%
(stratum 6, sealed), 0% (stratum 7, sealed), 1.6% (stratum 8
broad, D0-b) — against 47% acceptance on training tasks. Search
width and feedback density are secondary to that gap.

One standard lever has never been varied in this construction
line: training-set size (fixed at 1,200 tasks from pools of
32k/183k identities at strata 6/7). The next diagnostic (D1,
development data) varies it: fit on 1,200 vs 4,800 shape-broad
tasks, evaluate both on unseen shape-broad sets at widths 8 and
32 (which also settles C's unseen-task headroom), and measure
rollout acceptance on unseen broad tasks (B's true density).
Decision rules are stated in the survey's sequencing section as
amended by the D1 card: material broad-generalization gains
from data scale keep a sealed data-scaling test alive; flat
results across data scale, width and acceptance complete the
exhaustion evidence.

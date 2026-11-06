# D2 diagnostic result (development data, 2026-11-06)

Third scaling point planned in
[d1-diagnostics-results.md](d1-diagnostics-results.md).
Artifact: `artifacts/construction-d2-diagnostics-20261106/`
(`diagnose.py`, `results.json` with per-task rows, `run.log`).
One seed-0 fit on 19,200 shape-broad tasks (nested superset of
the D1 pools at strata 6/7; stratum 8 resampled block-balanced
with the D1 eval identities excluded), evaluated on the same
300 unseen broad dev tasks as D1. Fit: 9,246 CPU s, 150,246
teacher steps, final loss 1.249. Development evidence for
designing the preregistered test; no capability claim.

## The scaling curve (unseen broad eval, 300 tasks)

| Train tasks | @ width 8 | @ width 32 | Fit CPU s |
| ---: | ---: | ---: | ---: |
| 1,200 | 2 (0.7%) | 12 (4.0%) | 425 |
| 4,800 | 24 (8.0%) | 51 (17.0%) | 1,758 |
| 19,200 | 62 (20.7%) | 105 (35.0%) | 9,246 |
| Streaming @0.5 s | — | 50 (16.7%) | 0 |

Solve growth is superlinear in data at both widths over this
range, while fit cost grows ~linearly-to-superlinearly (the
19,200 fit cost 5.3x the 4,800 fit for 4.0x the steps). At
width 32 the 19,200 model doubles streaming's solves (105 vs
50) at lower eval CPU (87.7 vs 159.3 s). Paired on identical
tasks: learned-only 70 vs streaming-only 15 discordant pairs at
width 32 (decisive asymmetry); at width 8, 40 vs 28 (mild).

One measurement caveat: streaming solved 42 on these tasks in
D1 and 50 here. Its 0.5 CPU-second cap makes boundary tasks
flip with machine conditions; treat single streaming counts as
having several-task jitter at n=300. The paired comparisons
above use this run's rows consistently.

## Operating-point arithmetic (retrospective, dev rates)

Solves at equal total CPU, fit amortized over declared volume
N, using dev rates (streaming marginal 0.314 solves/CPU s from
this run):

| Recipe | Break-even N vs streaming | Solves/CPU s at N=12,000 |
| --- | ---: | ---: |
| 4,800 @w8 | ~10,300 | 0.31 |
| 4,800 @w32 | ~7,300 | 0.38 |
| 19,200 @w8 | ~16,000 | 0.31 |
| 19,200 @w32 | ~11,200 | 0.33 |
| Streaming | — | 0.314 |

The 4,800-task width-32 point dominates at moderate volumes:
its fit is 5x cheaper than the 19,200 fit while its solve rate
is half. The 19,200 recipe only becomes the better operating
point at volumes well above 12,000 (and width 8 only above
~16,000 against width 32 at equal fit). All projections are
single-seed dev estimates with streaming-jitter sensitivity;
they choose the sealed design, they do not prejudge it.

## Decision: preregister the sealed data-scaling test

Chosen operating point, frozen from this dev evidence: 4,800
training tasks, beam width 32, declared eval volume N=12,000 on
a fresh shape-broad sealed bank (identities disjoint from every
prior corpus and dev pool), 3 seeds, fit charged once per seed,
streaming and untrained controls, verdict on solves at equal
total CPU at N=12,000. Kill criterion: if the learned arm does
not beat streaming under that comparison, the data-scaling
route closes at accessible scale — with feedback, amortization,
ranking and fragment routes already closed, that outcome
completes the exhaustion evidence. No retest at a larger
volume. The volume is declared from the deployment story (one
large synthesis batch per fitted constructor) inside the
dev-estimated break-even range, and the downside is accepted
in advance. Design doc:
[scaling-sealed-design.md](scaling-sealed-design.md).

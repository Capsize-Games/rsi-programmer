# Sealed data-scaling result (shape-broad bank v1)

One-shot sealed evaluation run 2026-11-06 under the preregistered
design in [scaling-sealed-design.md](scaling-sealed-design.md).
Artifact: `artifacts/construction-scaling-final-20261106/report.json`
(plus run logs). Corpus:
`fixtures/construction_scaling_corpus_v1.json`, sha256
`2bff2fd8...` (verified before inference; two generator sources
pinned and hashed in the report). Bank: 4,800 train / 12,000 eval
tasks, shape-broad strata {6, 7, 8} (4,000 eval per stratum),
exclusions covering the v1 corpus, the sealed hard corpus, and
all D1/D2 development identities (23,498 identities). Recipe:
teacher imitation, 100 epochs, beam width 32, seeds 0/1/2.
Streaming control at 0.5 CPU s/task, evaluated once. Declared
volume 12,000. Total worker CPU 31,706.8 s (8.8 h),
single-threaded under the builder lock.

## Verdict

**FAIL. The frozen kill criterion is met**: the data-scaled
learned arm does not beat streaming on solves at equal total CPU
at the declared volume. Per the criterion, the data-scaling
route closes at accessible scale — no retest at larger volume,
wider beam, or larger train pool.

Credited solves at the common budget B* = 4,992.5 CPU s (the
minimum arm total; fit charged before eval rows, truncation in
the shared seeded task order):

| Arm | Credited solves |
| --- | ---: |
| Learned (mean of 3 seeds) | 1,626.7 |
| Streaming | 1,880.0 |
| Untrained (mean) | 0.0 |

The sanity gate passes (learned ≫ untrained), so the run is a
valid learning test. The failure is also visible without
truncation: raw solves over all 12,000 tasks are learned mean
1,662.3 vs streaming 2,293 — streaming solves more tasks in
absolute terms at this volume, not only per CPU second.

## Per-seed record

| Seed | Fit CPU s | Learned solves | Eval CPU s | Final loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1,437.4 | 2,025/12,000 | 3,639.7 | 1.031 |
| 1 | 1,459.3 | 1,532/12,000 | 3,674.3 | 1.322 |
| 2 | 1,338.4 | 1,430/12,000 | 3,654.1 | 1.328 |
| Streaming | 0 | 2,293/12,000 | 6,130.0 | — |
| Untrained (mean) | 0 | 0/12,000 | ~3,456 | — |

Seed variance is large (1,430-2,025 solves; the best seed still
trails streaming's raw 2,293). Final training loss again does
not rank seeds by solves.

## Strata and pairing (mean solves per seed; 4,000 per stratum)

| Arm | Stratum 6 | Stratum 7 | Stratum 8 |
| --- | ---: | ---: | ---: |
| Learned | 1,122.3 | 401.0 | 139.0 |
| Streaming | 2,293.0 | 0 | 0 |
| Untrained | 0 | 0 | 0 |

The arms remain complementary: streaming owns stratum 6 (~57%
of it) and nothing deeper; the learned arm is the only route
that solves strata 7/8 at all (540 mean solves there). Paired on
identical tasks, learned-only vs streaming-only discordant
solves are 1,180 vs 1,448 (seed 0), 887 vs 1,648 (seed 1), and
836 vs 1,699 (seed 2): the learned arm's exclusive territory is
real but smaller than streaming's at this training scale.

## Why the dev projection missed

D1/D2 dev arithmetic projected roughly +20% credited margin for
this operating point. Three measured differences explain the
miss, none of them protocol violations:

1. **Streaming is stronger on this bank's stratum 6.** Sealed:
   2,293/4,000 stratum-6 solves (57%). Dev observations put
   streaming stratum-6 at 42-50 of 300 mixed tasks with all
   solves in stratum 6; the 12,000-task bank gives it far more
   stratum-6 volume to harvest at the same 0.5 s cap.
2. **Learned eval cost grew.** Sealed learned eval is
   ~0.305 s/task at width 32 vs ~0.30 on the D1 dev set — close
   — but raw solve rate at width 32 is 13.9% mean vs 17.0% on
   the D1/D2 dev eval, at the same 4,800-task recipe. Dev-eval
   reuse across D1/D2 design choices likely carried mild
   adaptive optimism.
3. **Seed variance.** The dev projection was single-seed
   (seed 0, which is also the strongest sealed seed at 2,025);
   seeds 1/2 land at 1,532/1,430.

The result doc for D1/D2 flagged each of these risks in advance
(single seed, dev-set reuse, streaming cap jitter); the sealed
run is the arbiter and it says FAIL.

## Failed first invocation (retained)

The first invocation crashed before any model inference:
the runner creates its output directory with `exist_ok=False`
and the launcher had pre-created it (`FileExistsError`). No task
was evaluated and no result was observed; the empty directory
was removed, the traceback retained as
`runscaling-failed-attempt.log` in the artifact directory, and
the corrected invocation above is the single one-shot run. No
recipe, gate or budget changed between invocations.

## Interpretation

This closes the data-scaling question as preregistered. What
the full arc (D0-D2 diagnostics plus this sealed run)
establishes:

1. Training-set size is a genuine lever on shape-level
   generalization (dev curve 2 → 24 → 62 solves at width 8 for
   1.2k/4.8k/19.2k tasks), and the learned arm reaches task
   strata enumeration cannot touch at the frozen cap. This is
   real capability growth, measured three times.
2. At the declared deployment volume, that growth does not
   convert into a win over streaming enumeration at equal total
   CPU. Streaming's stratum-6 harvest at scale plus the fit
   charge exceeds the learned arm's broader-but-thinner
   coverage.
3. Every preregistered route to an ANN construction advantage
   in this repository has now been tested under sealed or
   powered conditions and closed on matched cost: imitation
   (construction-v1), learned ranking, neural fragment
   composition, execution feedback / expert iteration,
   amortization at volume (premise voided by D0), and now
   data-scaled imitation. The remaining untested items from
   the 2026-11-06 survey — on-policy feedback at the new
   acceptance density and execution-guided decoding — were
   assessed in the diagnostics as dominated or secondary
   (teacher traces are free; width 32 already captures most
   decoding headroom) and were explicitly parked behind the
   scaling route, not left as open decisive tests.

The 12,000 eval tasks are now development data. Any future
construction claim needs a new sealed bank and a new
preregistered hypothesis; the frozen criterion forbids
re-testing this route at a larger volume, wider beam, or
larger train pool.

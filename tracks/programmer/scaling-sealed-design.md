# Sealed data-scaling test: design (preregistered 2026-11-06)

Question left open by the D0-D2 diagnostics
([D0](d0-diagnostics-results.md), [D1](d1-diagnostics-results.md),
[D2](d2-diagnostics-results.md)): training-set size moves
unseen-task generalization strongly on shape-broad dev data.
Does a data-scaled imitation constructor beat streaming
enumeration on solves at equal total CPU, at a declared
deployment volume, on a fresh sealed bank? This doc freezes
the test before any sealed-bank task exists.

## Frozen operating point (chosen from dev evidence, above)

- Train pool: 4,800 tasks (1,600 per stratum 6/7/8), teacher
  imitation, 100 epochs, seeds 0/1/2.
- Decoder: beam search, width 32, tree-op bound 8, execution
  backstop 2,000,000 (the frozen D1/D2 settings).
- Streaming control: `pre_evaluation_search`, 0.5 CPU s/task
  (D0 showed this cap near-optimal for streaming CPU/solve).
- Untrained control: same beam, fresh policy per seed.
- Declared volume: N = 12,000 sealed eval tasks (4,000 per
  stratum), one seeded task order shared by all arms.

The 4,800/width-32 point is chosen because dev arithmetic
(D2 doc) gives it the smallest break-even volume (~7,300
tasks) of the measured operating points; N=12,000 is declared
as one large synthesis batch per fitted constructor, inside
the dev-estimated break-even range. Both numbers are frozen
now; the downside risk is accepted in advance.

## Bank

Fresh corpus, generated and hash-frozen before any fit:

- Strata: exact minimal tree-op counts {6, 7, 8} over
  add/sub/mul on x/y, as in the hard bank.
- Shape-broad sampling: strata 6/7 digest-ordered from the
  complete catalog levels; stratum 8 by rejection sampling
  balanced across left-subtree sizes 0-7 (the D1/D2 method),
  never enumeration-prefix.
- Exclusions by polynomial identity: the v1 corpus, the sealed
  hard corpus, and every D1/D2 development pool identity
  (train and eval), re-derived deterministically.
- Splits: train 4,800 / eval 12,000, identity-disjoint.
  Eval rows are never read before the sealed run; corpus and
  generator-source hashes are verified in the runner before
  inference, as in prior sealed runs.

## Cost accounting and decision rule

Per-row CPU seconds recorded for every arm; fit CPU charged
to the learned arm before its first eval row. Let B* be the
minimum total CPU across the compared arms. Each arm is
credited with solves among rows whose cumulative CPU (fit
first, where applicable) does not exceed B*, in the shared
seeded task order. No extrapolation beyond measured rows.

- Primary verdict: mean credited solves over the 3 learned
  seeds vs credited streaming solves (streaming run once: it
  is a deterministic enumerator; its only seed-dependence is
  CPU-cap jitter, which the result doc must caveat using the
  D1/D2 observation of 42 vs 50 solves on identical tasks).
- Sanity gate: credited learned solves must also exceed the
  untrained control's; failure means the run is invalid as a
  learning test and counts as FAIL.
- Secondary reporting (not decision-bearing): raw solves,
  solves per total CPU, per-stratum solves, fit/eval CPU
  split, seed spread.

## Kill criterion (verbatim for the runner)

"if the data-scaled learned arm does not beat streaming on
solves at equal total CPU (fit charged, truncated at the
common budget, declared volume 12,000) on the sealed
shape-broad bank, the data-scaling route closes at accessible
scale — no retest at larger volume, wider beam, or larger
train pool."

A PASS is the goal's payoff criterion (learned route beats
the best classical control at matched total cost on a sealed
bank) for arithmetic construction; it claims nothing about
Python/CURLEE program capability. A FAIL, with feedback,
amortization, ranking and fragment routes already closed,
completes the exhaustion evidence for ANN construction at
accessible scale in this repository.

## Build plan

- `rsi_programmer/construction_broad_corpus.py`: broad sampler
  + corpus generation with all exclusions; tests for
  minimality, disjointness (incl. vs D1/D2 dev identities),
  quotas, determinism.
- `rsi_programmer/construction_scaling_final.py` + runner
  script: bundle/hash verification, frozen-config checks,
  arms, per-row CPU records, truncation decision embedding
  the kill criterion verbatim; fixture
  `fixtures/construction_scaling_v1.json` frozen with the
  corpus hash before the run.
- Tests registered in `tracks/manifest.json` and the justfile
  fast lane; affected lane green before the run.
- One serialized run under the builder lock, single-threaded.
  Estimated ~7-8 CPU-hours (fits ~5,300 s, learned eval
  ~10,800 s, untrained ~3,200 s, streaming ~6,400 s).
- Result doc + AGENTS.md checkpoint + kanban closure with the
  numbers, win or lose; eval tasks become development data
  afterwards either way.

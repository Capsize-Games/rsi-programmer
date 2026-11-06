# Execution-feedback construction: design and frozen decisions

Status: architecture frozen for implementation, 2026-11-06.
Owner direction: keep pursuing ANN programming, testing the one
combination never tried in this repo — learning from execution
outcomes on a bank where enumeration does not saturate. Context:
[project audit](../../docs/project-audit-2026-11-05.md),
[sealed final result](construction-final-results.md) (imitation
ANN learns — 94-119/244 sealed vs ~0 untrained — and loses on
cost because streaming is nearly free at four operations).

## Hypothesis and kill criterion

Hypothesis: a construction ANN trained on its own
checker-accepted rollouts (expert iteration), on tasks deep
enough that streaming enumeration cannot saturate within budget,
beats untrained and imitation-trained controllers, and beats
streaming at equal total CPU.

Kill criterion (applies to the decisive run): if the
feedback-trained arm does not beat **both** untrained and
imitation arms on solves, **or** does not beat streaming at
equal total CPU (fit and rollout generation charged, amortized
over the actual evaluation volume), the ANN-construction
hypothesis at accessible scale is closed — not tuned further.

## Non-goals

No grammar broadening beyond depth (same operators/terminals),
no learned abstractions/fragments, no architecture search, no
SNN arm, no pretrained models, no changes to sealed v1 assets.
RECURRENT/MLP rankers and statement routes are untouched.

## Components and interfaces

All new code follows repo conventions (79 cols, full type hints,
one class per file, files <= 250 lines, functions <= 20 lines,
tests registered in `tracks/manifest.json`).

### 1. Hard bank generator v2 (worker W1)

- New module `rsi_programmer/construction_hard_corpus.py` +
  `scripts/generate_construction_hard_corpus.py` (refuses
  existing output, like v1).
- Samples random expression trees with exact sampled tree-op
  counts in strata {5, 6, 7} over operators add/sub/mul and
  terminals x/y (no constants, as v1). Stratum = sampled op
  count, recorded per task as `generator.sampled_ops`; minimal
  op count is **not** claimed (v1 catalog minimality does not
  scale; say so in the fixture scope text).
- Dedupes and splits by exact polynomial identity
  (`online_graph.graph_polynomial`), excluding constant/identity
  targets and any polynomial present in the v1 corpus
  (`fixtures/construction_corpus_v1.json`, all splits) — reading
  v1 *train/dev* polynomials only for exclusion is not possible
  without loading the file; loading the corpus file for identity
  exclusion is permitted, but no v1 eval *task rows* may be used
  as training or selection data beyond identity exclusion, and
  the exclusion count per split is recorded.
- Output schema identical to v1 corpus: top-level `splits`
  with keys `train`/`dev`/`eval`; each task `{id, split,
  observed (36 rows, grid -3..2), held (24 rows from -6..6
  excluding observed), max_tree_ops, generator:{program,
  polynomial, sampled_ops}}`. Quotas frozen in the fixture
  protocol: train 1200, dev 200, eval 300, split evenly across
  strata as far as the sampled pool allows; shortfalls recorded,
  never silently rebalanced after seeing solver results.
- Protocol fixture `fixtures/construction_hard_v1.json` mirrors
  v1 (corpus sha, source provenance hashes, evaluation block:
  model seeds [0,1,2], per-task CPU budget set at calibration,
  beam width 8, top-k 2; decision_rules with the kill criterion
  above verbatim). The eval split is sealed by the same
  hash-check discipline as `construction_final.py`: solvers load
  it only in the one evaluation pass, after all arms' training
  is frozen.

### 2. Rollout collector (worker W2)

- New module `rsi_programmer/construction_rollouts.py`:
  `collect_rollouts(policy, tasks, bound, budget, width=8,
  seed=0) -> list[Rollout]` running `beam_construct` per train
  task, scoring each returned graph with `check_graph`
  (observed rows, the task's held rows, its polynomial, bound)
  — training-split data only; assert `split == "train"`.
- `Rollout` record (dataclass or plain dict, match codebase
  style): task id, actions, accepted flag, observed-match
  fraction of the final graph's output vector, executions,
  process CPU seconds for the whole collection call.
- `trajectory_examples(rollouts, tasks_by_id)`: replays each
  accepted rollout's actions from the initial state via
  `ConstructionState.apply`, yielding `(state, action)` pairs in
  the exact shape `teacher_steps` produces. States must be the
  pre-action states; a replay whose final graph differs from
  the rollout graph raises.
- Tests: accepted synthetic rollout replays to identical graph;
  unaccepted rollouts yield no examples; non-train tasks refused.

### 3. Feedback training loop (worker W3)

- Refactor minimally: extract the example-fitting core of
  `fit_constructor` into `fit_on_examples(policy, examples,
  epochs, batch_size, lr)` in `construction_training.py`;
  `fit_constructor` keeps its signature and delegates, so all
  existing callers, hashes behaviour for the v1 recipe, and
  tests are unchanged. (`fit_on_examples` must not weaken the
  train-only guard: it takes examples, and its callers own split
  discipline; document that.)
- New module `rsi_programmer/construction_feedback.py`:
  `fit_with_feedback(policy, tasks, rounds, epochs_per_round,
  rollout_budget, bound, seed)` implementing expert iteration:
  round 0 = imitation fit on teacher steps; each later round
  collects rollouts with the current policy, unions accepted
  trajectory examples with teacher examples (both fully
  weighted; dedupe not required), refits **from the same
  initialisation** (fresh `ConstructionPolicy(seed=...)` weights
  via a new instance per round is forbidden — refit the passed
  policy only if a `refit_from_init` helper resets parameters
  deterministically. Worker chooses one, documents it, tests
  determinism by fingerprint).
  Returns per-round records: rollout counts, accepted counts,
  example counts, rollout CPU, fit CPU (from `fit_on_examples`),
  fingerprints. Total training charge = sum of all rounds'
  rollout + fit CPU.
- A round with zero accepted rollouts stops iteration and is
  recorded as such (no silent fallback to teacher-only refit
  presented as feedback).
- Tests: one tiny end-to-end feedback fit on synthetic train
  tasks changes the fingerprint; round records sum costs;
  zero-acceptance stop path (policy stubbed untrained on an
  impossible-by-bound task set is acceptable if honest).

### 4. Evaluation protocol + runner (worker W4)

- New `rsi_programmer/construction_hard_final.py` +
  `scripts/run_construction_hard_final.py`, patterned on
  `construction_final.py` / its runner (hash checks before
  inference, frozen-receipt argument, refuse existing output,
  weight fingerprints unchanged by evaluation).
- Arms: `streaming` (pre_evaluation_search with the bank bound),
  `untrained`, `imitation`, `feedback`. Learned arms: seeds
  0,1,2, fresh fits inside the run; imitation = `fit_constructor`
  at the frozen epoch count; feedback = `fit_with_feedback` at
  frozen rounds/epochs/budget from the protocol fixture.
- Per-task budget is identical **CPU seconds** for every arm
  (reuse `cpu_budget` ITIMER pattern from the ranking work);
  root-execution counts reported separately and never used as
  the budget. Acceptance by `check_graph` at the bank bound.
- Report: per-arm solves + CPU (fit, rollout generation,
  inference separately), equal-total-CPU truncation comparison
  vs streaming (as in `cpu_matched_verdict`), per-stratum solves,
  and the kill-criterion verdict computed from frozen rule text.
- The runner refuses to run if corpus/protocol hashes drift, and
  refuses a second run into an existing output directory.

## Bank v2 amendment (2026-11-06, after calibration)

Calibration of the sampled-ops bank failed its design rule at
every budget: streaming solves 80% at 0.5 s and 100% within the
10 s cap (sampled trees collapse to low-complexity polynomials;
`artifacts/construction-hard-calibration-20261106/`). The bank
therefore switches strata from sampled tree-op count to **exact
minimal tree-op count**, the intrinsic property the v1 catalog
used: enumerate polynomial identities level by level (exact,
generator-side, solver-free), strata {6, 7, 8}, bound 8, witness
= first-appearance catalog node. Catalog growth measured:
39,093 cumulative identities through size 6 in 1.7 CPU s. Level 8
collection may early-stop after collecting a recorded multiple
of its quota (first appearance at level 8 is minimal regardless
of enumeration order, provided the seen set is complete below
8); any shortfall is recorded, never silently rebalanced. The
failed sampled bank and its calibration are retained as evidence;
no solver ever ran on its eval split.

## Calibration (parent, serialized, before freezing budgets)

On a train-pool sample only (never eval rows): measure streaming
solve fraction vs per-task CPU budgets {0.5, 1, 2, 5} s per
stratum; pick the smallest budget where streaming solves <= 25%
overall while imitation beam (v1 recipe scaled to the bound)
solves >= 10% — headroom both ways. If no budget separates them,
deepen strata within bound 8 or declare the bank design failed
before freezing; either way the calibration table is published
in the result doc. Budgets then freeze into the protocol fixture
before any arm is trained for the decisive run.

## Cost and reporting rules

Everything charged: rollout generation is training cost, not
free data. Amortization only over the actual evaluation volume.
Single observations labelled as such; load at run time recorded.
Failures, zero-acceptance rounds, and the calibration table are
retained in the result doc regardless of outcome. Solver changes
after the decisive run turn this bank into development evidence.

## Interfaces between workers (frozen)

- Corpus JSON schema = v1 shape (above). W2-W4 code against it,
  not against W1 internals.
- `(ConstructionState, Action)` example pairs are the currency
  between W2 and W3, identical to `teacher_steps` output.
- W4 consumes `fit_with_feedback`'s return records for cost
  fields by documented keys; W3 freezes those keys in its module
  docstring before W4 codes against them (worker order in the
  workflow makes W3's docstring available to the parent at
  integration; mismatches are integration fixes, not redesign).

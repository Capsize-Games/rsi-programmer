# RSI outer-loop family tally, 2026-11-05 (issue #94)

Read-only audit of the family the main audit parked: the original
RSI/SearchPolicy outer loop (`policy`, `author`, `lauthor`,
`lattice`, `evolve`, `scale`, `self_gate`, `headless`, `promote`),
the `probe.TASK` line, the `rsi_*.json` fixtures,
`artifacts/ranker-20261017`, `artifacts/review-5c6701b`,
`artifacts/validation-20261017`, the `followup-*` dirs,
`docs/gains/`, and `architecture.md` / `programmingplan.md`
past line 120. No training or runs were done for this tally.

Headline answer: no trained-beats-best-classical-at-matched-cost
result hides in this family. What looks like a win in rank
(position) units evaporates under matched signature-work cost.
Details per family below.

## A. Ranker ordering (`policy.py`, `scripts/measure_ranker.py`)

- Task set: `rsi_shadow_tasks.json` (n=4, seen),
  `rsi_novel_tasks.json` (n=3, task-unseen),
  `rsi_ood_plain.json` (n=8, plain OOD). Fitted only on shadow.
- Trained (50 epochs, 3-seed mean first-solver position, lower
  is better) vs controls:
  shadow 17.25 vs probe-order 640.75 vs exact-random 912.60;
  novel 390.11 vs probe 828.00 vs random 521.43;
  plain OOD 831.12 vs probe 482.50 vs random 830.29.
- Cost basis: position counts only. Matched-cost instrumentation
  (`work.json`) finds candidate-signature counts IDENTICAL across
  generation order, uniform permutation, untrained and trained
  rankers on all 15 tasks. Family-total guided/streaming ratios:
  3.21x shadow, 2.37x novel, 3.33x plain OOD — guided does MORE
  signature work, plus operand signatures and neural scoring.
  Instrumented wall per task is also slower guided (e.g. 0.114s
  vs 0.013s on `square_plus`).
- Verdict: ordering effect is real on shadow and novel (beats
  random expectation in position units); it is NOT a
  matched-cost win — the best classical comparator (streaming
  early-stop) uses less work. No end-to-end runtime advantage is
  claimed by the source docs either.
- Artifacts: `artifacts/ranker-20261017/{positions.csv,
  summary.json,work.json,manifest.json}`,
  `artifacts/followup-measurement-20261018/ranker/` (repeat),
  `docs/audit-amendments.md` ("Independent rank measurement"),
  `docs/audit-followup.md` code-v48 section.

## B. Author/beam greedy acceptance (`author.py`, issues #66/#70)

- Task set: 4 shadow + 3 novel tasks, seeds 0/1/2, canonical
  and reachability labels, 420 raw rows.
- Trained (full bonuses + neural) vs no-neural control, paired:
  across 84 pairs full wins 29, loses 3, ties 52 on acceptance
  (train+held correctness required). Canonical novel beam:
  full 6/9 vs no-neural 3/9; canonical novel author: 0/9 both.
- Cost basis: author budgets count generated triples, beam
  budgets count committed nodes. The follow-up states explicitly
  these are NOT equal-work comparisons. The no-neural control
  keeps the fitted classical bonuses (exact_flag/open_gap), so
  this is a neural-ordering contribution, not trained vs
  classical at matched cost. 70/84 bonus-ablation pairs are null
  contrasts (five of six fits selected zero bonus weight).
- Verdict: conditional neural ordering contribution on tiny
  fixtures; no matched-cost classical comparison exists, so no
  hidden cost win. Candidate generation and signature
  evaluation remain classical throughout.
- Artifacts: `artifacts/followup-measurement-20261018/
  author-ablation-v1.json`, `author-analysis.json`,
  `author-integrity-audit.json`,
  `artifacts/followup-measurement-repaired-20261018/` (same set
  + dependency bracket).

## C. Lattice / lattice-author (`lattice.py`, `lauthor.py`)

- Task set: exercised through unit/integration tests only.
- Trained vs best control: no evaluable artifact. No
  trained-vs-classical comparison artifact exists in any
  `artifacts/` dir for lattice search.
- Verdict: no evaluable artifact.
- Related tests (not evidence of advantage):
  `tests/test_rsi_lattice_author.py`,
  `tests/test_rsi_lattice_train.py`,
  `tests/test_review_lattice.py`.

## D. Outer loop: evolve / scale / headless / self_gate / promote

- `evolve.py` (PolicyLoop), `scale.py`/`pool.py` (ScaleUp),
  `headless.py` (HeadlessLoop), `self_gate.py` (ledger),
  `promote.py` (PromotionGate): orchestration, persistence and
  gating, not competing solvers. No trained-vs-classical
  comparison artifact exists for any of them.
- Closest load-bearing check:
  `test_evolve_and_accept_author_paths` requires 8 per-task beam
  records and archive growth; it verifies acceptance SCOPE is
  retained, explicitly not improved solve rates
  (`docs/audit-followup.md` code-v56).
- The NaN-promotion and held-out-acceptance defects found in
  `review-5c6701b` are corrected per `audit-amendments.md`;
  the corrections remove false positives, not a hidden win.
- Verdict: no evaluable artifact for any trained-vs-classical
  claim. Nothing here contradicts the main audit.
- Artifacts (functional/validation only):
  `artifacts/followup-scale-checkpoint-20261018/`,
  `artifacts/followup-scale-evolve-checkpoint-20261018/`,
  `artifacts/followup-slow-pass-20261018/slow.log`,
  `artifacts/followup-final-validation-20261018/`.

## E. Gains v12–v29 (`docs/gains/gains.csv`)

- Kept DROPPED as evidence, per its own README. Reason,
  confirmed by inspection: `enum_tried` is constant (2935 twin/
  plain, 125930 full-ir) across all releases, so the rising
  solve columns cannot be work comparisons — they track grammar/
  host changes (full-ir narrow solve stays 0.250 throughout).
  Counters are path-specific `tried` units, not comparable
  signature work; comparable units are not reconstructible from
  the CSV (no per-path budget/unit labels, no seeds).
- Verdict: no evaluable artifact; correctly excluded.
- Check: `tests/test_gains.py` passes (1 passed), confirming
  only CSV well-formedness, not evidentiary value.

## F. Probe TASK line, fixtures, front-doc claims

- `rsi_programmer/probe.py:41`: `TASK` is hard-coded to the
  `square_plus` string; the probe tuple omits its examination
  count (open limitation, `audit-amendments.md` #53 row).
  Single-task probe, not a benchmark.
- Fixtures: there are 14 `fixtures/rsi_*.json` files, not 13
  (branch 2, const 2, gen_novel 12, gen_train 24, hard 4, novel
  3, ood_plain 8, ood_tasks 8, ood_twin 8, ood_v2 8, plain_hard
  12, plain_more 12, plain_train 24, shadow 4). Only shadow /
  novel / ood_plain have ranker artifacts; the rest are
  evaluated solely inside author/beam or benchmark lanes.
- `architecture.md` past line 120 is consistent with this
  tally: its work-counter section (lines ~292-308) already
  states `signature_evaluations` excludes recursive calls and
  neural work and "is not a total CPU-operation or runtime
  measurement", and its gains paragraph forbids reading the
  charts as current work comparisons.
- `programmingplan.md` W3/W4 table plus "Last measured" block
  (lines ~39-47: operand top-1 gains, beam 10/24 to 11/24,
  transfer 3/3) is self-marked stale ("do not quote as current
  without re-running"); weights w1/w2 era, superseded by the
  amendment/follow-up measurements above.

## Followup-* directory classification

Measurement: `followup-measurement-20261018/` (+`-repaired`),
`followup-validation-20261018/`,
`followup-generator-20261018/`. Integration/CI:
`followup-ci-20261018/`, `followup-integration-20261018/`,
`followup-final-validation-20261018/`,
`ci-full-followup-20261018/`. Fixes with red+green pairs:
`followup-hub-fix-20261018/`, `followup-rich-fix-20261018/`.
Partial-lane snapshots (not passes):
`followup-scale-checkpoint-20261018/`,
`followup-scale-evolve-checkpoint-20261018/`,
`followup-slow-failure-20261018/`,
`followup-slow-pass-20261018/`. Load-bearing for this tally:
the two measurement dirs and `followup-validation-20261018/
ranker-dependency-bracket.json`.

## Flagged contradictions with the main audit doc

1. Main audit coverage gap says "13 `rsi_*.json` fixtures" —
   count is 14 (listed in F). Minor.
2. Main audit Park says of this family "No claim either way
   until someone tallies" — this tally nowcloses that gap with
   a verdict (no hidden matched-cost win), corroborating rather
   than contradicting the bottom line.
3. No contradiction on substance: the audit's "evidence exists
   only in the wrong budget unit" (pursue 2) matches the
   ranker/author cost-basis findings exactly.

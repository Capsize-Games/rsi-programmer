# Cost-lever sealed epic (draft)

Question: does the LLM proposer route (python-first gating
+ near-miss repair) pass the frozen bar on unseen tasks?

## Bar (frozen on epic open)

Same three prongs, M=3, via `evaluate_bar`. Prong (ii)
uses measured per-(arm,task) CPU persisted in the
summary (`cpu_by_task`); even-spread totals are
reported for comparison only and never decide.
PROCEED needs all three prongs; else KILL, no retest
without a new epic.

## Evidence chain

- Motivation only: sequential dev run 2026-11-26, exact
  ratio 1.161 (reference, not a verdict).
- Decision: one-shot sealed run, 90 sealed tasks, all
  arms, dual-backend acceptance. Sealed rows never enter
  repair/query feedback.
- Runner must persist `cpu_by_task` so the ratio is
  recomputable from the artifact.

## Scope

Dev bank for calibration only; sealed bank stays sealed
until the one-shot. No protocol/bar edits after open.

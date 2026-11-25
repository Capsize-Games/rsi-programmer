# Shard-3 notes (inversion-dev-20261126)

Tasks: `llm_dev_010` (A6), `llm_dev_021` (A7), `llm_dev_039` (A8),
`llm_dev_045` (S1), `llm_dev_083` (S3).
Confirmed against `fixtures/llm_bank_v1/tasks.json`:
all five are pool `dev`, interfaces pair/pair/pair/list/list.
No bank-mapping discrepancy. Sealed pool never loaded
(runner asserts `llm_dev_*` ids and `dev` pool).

Procedure: exact copy of the prior no-gate shard runner
(`artifacts/inversion-dev-20261125/shard-5/run_shard5.py`),
only `SHARD` tuple and docstring changed.
Machine stats record-only, no pre-wait, no polling.
Every CURLEE invocation went through the repo resource guard
inside `LlmJudge`; refusals stand as timeout verdicts, no retry.

## Outcome

- Final accepts per arm: a/b/c/d/e = 0/0/0/0/0 (all lists empty).
- Visible accepts: a=3, b=c=d=e=0.
  Which 3 tasks arm (a) visibly accepted is NOT recoverable from
  persisted outputs (per-record candidate texts and verdict details
  are not written by the runner; only aggregate counts persist).
- Library: 0 entries, 10 skips, all
  "held-accepted judge_final verdict required".
- CURLEE check-pass rate (final stage): 0.0.
- Frozen-bar decision on this shard: KILL
  (no deep reach, undefined cost ratio, counts a/b/c = 0/0/0).
  Shard-level verdict only; global verdict belongs to the owner
  aggregating all six shards.
- Rejection taxonomy (final): a/b/c = 1 python_mismatch + 4 timeout
  each; d = 1 python_mismatch + 2 timeout (d's two statement tasks
  are not_applicable, finalized 23 of 25 records); e = 5
  python_mismatch (seeded random deck).

## Guard refusals

25 timeout verdict events in persisted records
(visible 11 + final 14; 14 distinct final-stage arm/task pairs).
All are attributed to resource-guard rc75 refusal:
whole-run wall is ~21 s while one genuine CURLEE timeout costs
30 s, and all 146 judge backend invocations total 18.9 s wall
(mean ~0.13 s). Repair-intermediate verdicts are not persisted,
so 25 is a lower bound on refusal events. Heavy contention with
the five sibling shards sharing the single CURLEE build slot is
the expected cause, per the brief.

## Files

- `run_shard3.py` — runner script copy (this shard's SHARD tuple).
- `summary.json` — full runner aggregate (counts, costs, ledgers,
  taxonomy, library, decision).
- `inversion-dev-results.md` — runner report.
- `shard_summary.json` — per-arm accept lists, CPU seconds,
  guard_refusals, protocol_sha256, machine stats.
- `protocol_sha256.txt` — `c1ee97b7...e80023c7edb15`
  (shard-scoped protocol; differs from shard-5's hash only by the
  task_ids list).
- `machine_start.json` / `machine_end.json` — record-only stats.
- `judge-work/` — sandbox outputs and generated CURLEE sources.

No commits, no pushes, no checkouts touched outside this worktree.

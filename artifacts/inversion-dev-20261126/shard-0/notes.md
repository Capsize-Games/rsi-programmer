# Shard-0 notes (2026-11-26, NEW dev trial, main 0f29072 as-is)

Tasks: llm_dev_004 (A6/pair), llm_dev_022 (A7/pair),
llm_dev_030 (A8/pair), llm_dev_049 (S1/list),
llm_dev_073 (S2/list). Confirmed against
fixtures/llm_bank_v1/tasks.json: ids, families,
interfaces and pool=dev all match the assignment;
no bank-mapping discrepancy.

Procedure: prior no-gate shard pattern copied to
run_shard0.py (SHARD tuple + docstring only; judge,
repair, protocol, bar untouched). Machine stats
record-only, no pre-wait, no polling. Every CURLEE
invocation went through the repo resource guard
inside LlmJudge (scripts/run_limited.py present).
No retry or wait on refusal.

Outcome: 0/5 final accepts on ALL arms (a/b/c/d/e).
Shard-scoped frozen bar: KILL (best arm none;
prongs all false). This is a shard slice, not the
official verdict; aggregation over shards decides.

Guard refusals: 30 timeout verdicts total (16
visible-stage + 14 final-stage across arms a-d).
Every one is a fast guard refusal (rc75), NOT a
genuine 30 s CURLEE timeout: total run wall was
21.6 s (uptime 679390.6 -> 679412.19) with judge
wall 18.9 s, so no single 30 s timeout could have
elapsed; all runs were sequential in one process.
Recorded as timeout verdicts per the hard rules;
no retry, no waiting.

Python-first gating observed: arm e shows 5/5
python_mismatch at visible with zero CURLEE-stage
verdicts, i.e. CURLEE was never invoked for those
candidates. Arm d has no grammar for statement
tasks (049, 073 recorded not_applicable by design).

Visible-stage note: arm b visibly accepted 1 task
(count 1) but it held-rejected (final python
mismatch); per-task visible-accept ids are not
persisted by the runner's summary, so the task id
is unresolved (see supplement). Repair conversion
b: 0 converted of 3 round-0 rejections. Library:
0 entries (nothing held-accepted to qualify).

Sealed pool untouched: runner asserts every id
starts with llm_dev_ and loads only dev splits;
held rows opened identically for all arms only at
final. No commit/push made; outputs only under
artifacts/inversion-dev-20261126/shard-0/ in this
worktree. No LLM teacher involved: proposer is the
deterministic no-gate fit procedure from the
shard-5 pattern (enumeration/deck scan over
visible + revealed rows only).

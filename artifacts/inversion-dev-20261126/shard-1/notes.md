# Shard-1 notes — NEW dev trial 2026-11-26 (main 0f29072)

Cost-lever measurement, not a KILL-epic retest. Five arms (a/b/c/d/e)
on dev tasks llm_dev_002 (A6/pair), llm_dev_020 (A7/pair),
llm_dev_042 (A8/pair), llm_dev_048 (S1/list), llm_dev_076 (S3/list).
Mapping confirmed against fixtures/llm_bank_v1/tasks.json (all pool=dev).
Current main code used as-is; judge, repair, protocol and bar untouched.

## Procedure

- Runner: run_shard1.py in this dir (copied from the
  artifacts/inversion-dev-20261125/shard-1 pattern, same SHARD tuple;
  docstring updated for this trial; appended a read-only enrichment
  block that adds accept lists / refusal counts / machine stats to
  summary.json after the run).
- No-gate: machine stats record-only, no pre-wait/polling. Every CURLEE
  call ran through the repo resource guard inside LlmJudge; rc75
  refusals recorded as timeout verdicts, no retry, no slot-waiting.
- Held rows: dev held opened identically for all arms at finalize only;
  sealed pool never loaded (dev-only ids asserted by DevDiagnosticRunner).

## Outcome (all arms 0 accepts — contention-confounded)

- Visible accepts a/b/c/d/e: 0/0/0/0/0. Final accepts: 0/0/0/0/0.
- Frozen-bar decision: KILL (best arm a; no deep reach; cost ratio
  undefined — no shared solves; counts a/b/c = 0/0/0).
- guard_refusals: 6 (rc75 command entries). Final/visible taxonomy:
  a/b/c each {python_mismatch: 3, timeout: 2}; d {python_mismatch: 1};
  e {python_mismatch: 5}. The 6 timeouts == the 6 refusals.
- CURLEE stage files show only llm_dev_048/076 (statement tasks)
  reached CURLEE at visible stage — i.e. the two timeouts per LLM arm
  are 048 + 076, refused by the guard under 6-shard contention.
  Graph tasks 002/020/042 failed in Python (python-first gating skipped
  CURLEE for them: zero curlee cost, as designed).
- Repair conversion b: 0 of 5 round-0 rejections. Library: 0 entries
  (nothing held-accepted; all 10 a/b candidates skipped with
  "held-accepted judge_final verdict required").
- CPU seconds a/b/c/d/e: 1.252 / 5.190 / 5.417 / 0.724 / 0.496.
- protocol_sha256: d9a61b16c555161401c8eec73e32290e38b6100f92e3da4f7a96c518de666b06
  (same as the 2026-11-25 shard-1 protocol hash — same task set).
- Machine: MemAvailable 9603428 kB (start) -> 7602672 kB (end);
  uptime 679431.14 s -> 679451.46 s (~20 s wall).

## Caveat for the cross-shard aggregation

- Prior pre-lever shard-1 (2026-11-25, same 5 tasks) scored
  a/b/c = 2/4/4 visible+final. This trial's 0/0/0 coincides with 6
  guard refusals knocking out exactly the two statement tasks that
  previously carried accepts (076 was in the prior deep-reach set).
  So this shard measures contention, not lever effect: do NOT read
  0/0/0 as "levers cost 2-4 accepts". The lever signal that IS clean:
  python-first gating held (3 graph tasks/arm skipped CURLEE after
  Python failure); the near-miss repair gate is unexercised here
  (no executable-but-wrong verdicts survived to test it — repair
  converted 0/5 and timeouts are non-near-miss stops).
- curlee_check_pass_rate_final: null (no finals to check).

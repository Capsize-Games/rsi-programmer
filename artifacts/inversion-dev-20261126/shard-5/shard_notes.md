# Shard-5 notes (inversion-dev-20261126, NEW trial with cost levers)

- Worktree: `/home/joe/Projects/rsi-programmer/.muse/worktrees/subagent-v2-01a21377-48e4-7731-9bf2-395ffcb7a325-01a214c9-fc03-75a3-9935-d042f2fa0669`
- Main commit (code as-is, unmodified): `0f29072` (python-first judge gating + near-miss repair gate).
- Runner: `run_shard5.py` (copied from `artifacts/inversion-dev-20261125/shard-5/run_shard5.py`;
  only change is SHARD order set to the assigned order).
- Protocol sha256: `63ec0a41f97a80ec5bc9ff82cbaf1d7d80f8f226fa93c7744369b20d7a3767bc`
- Bank mapping confirmed against `fixtures/llm_bank_v1/tasks.json`: all five ids
  present, all `dev` pool — `llm_dev_065` S2/list, `llm_dev_068` S2/list,
  `llm_dev_072` S2/list, `llm_dev_075` S3/list, `llm_dev_081` S3/list.
  No mapping difference; order used: 065, 068, 072, 075, 081.
- Sealed bank: `held.json` was read once with hash-pin verification and filtered
  to the five `llm_dev_*` keys before runner construction; the runner asserts
  dev-only ids and refuses any other key. No sealed row entered prompts, repair,
  retrieval, logs, or tuning. No held row entered repair/query feedback beyond
  the dev protocol (visible/query only pre-final; dev held opened identically
  for all arms at final).
- Hard rules kept: no pre-wait/polling (machine stats record-only); every CURLEE
  invocation ran through the judge resource guard; no retry loops; no
  commit/push; nothing outside this worktree touched (guard/compiler binaries
  used read/execute-only via the judge).

## Per-arm FINAL accepts (dual-backend, task-id lists)

- a: [`llm_dev_065`, `llm_dev_068`, `llm_dev_072`, `llm_dev_075`, `llm_dev_081`] (5/5)
- b: [`llm_dev_065`, `llm_dev_068`, `llm_dev_072`, `llm_dev_075`, `llm_dev_081`] (5/5)
- c: [`llm_dev_065`, `llm_dev_068`, `llm_dev_072`, `llm_dev_075`, `llm_dev_081`] (5/5)
- d: [] (0/5; statement tasks have no enumeration grammar → not_applicable)
- e: [] (0/5; final taxonomy `python_mismatch` × 5)

Visible-stage accepts for reference: a 0/5, b 0/5, c 4/5, d 0/5, e 0/5.
(The single c visible-reject task id is not recoverable from persisted
evidence — per-record verdicts are in-memory only — so it is unresolved.)

## CPU seconds (per-arm record totals)

- a 3.154, b 2.904, c 3.829, d 0.000, e 0.253.
- Judge ledger backend_execution wall 95.5 s; whole-run wall 95.7 s
  (uptime 679404.97 → 679500.68).

## Guard refusals

- Confirm-stage visible timeouts: a 5, b 5, c 1 → 11 events attributed to
  resource-guard refusals (rc75), recorded as timeout verdicts per the hard
  rules. Attribution is inferred, not directly observed: lite verdicts drop
  the detail string ("resource guard refused job" vs genuine timeout) and
  per-command exit codes are not persisted. Basis: (1) total serial run wall
  was 95.7 s, so eleven 30 s CURLEE timeouts (330 s) or eleven 5 s python
  timeouts (55 s alongside ~38+ successful CURLEE check/run invocations plus
  all final-stage python runs) do not fit the measured 95.5 s backend wall —
  the timeouts were fast failures; (2) the identical candidate texts were
  accepted at final through the same python sandbox on 40 held rows, ruling
  out code-side slowness; (3) six shards share one CURLEE slot, so lock
  contention is the expected refusal mode. Repair-internal round timeouts
  (e.g. b round-0 rejections × 5, repair conversion 0/5 under the near-miss
  gate which stops on timeout) are NOT counted — 11 is a lower bound.
- Corroborating: fragment library built 0 entries (all 10 a/b qualifications
  failed `judge_final` during the contention window with
  "held-accepted judge_final verdict required"), arm c ran retrieval against
  an empty library (retrieval delta 0), yet finalize accepted 15/15 a/b/c —
  the codes were correct; the visible-stage rejections were environmental.
- Run wall-clock ordering vs the other five parallel shards is unknown from
  this shard's evidence; no waiting or polling was performed.

## Machine stats (record-only)

- start: uptime 679404.97 s, MemAvailable 10346784 kB
- end: uptime 679500.68 s, MemAvailable 7719448 kB (above the 6 GiB guard floor)

## Shard verdict (development evidence only, 5 tasks)

- KILL-pattern on this shard: best arm b, deep reach true (streaming scores 0
  on statement tasks), cost ratio undefined (no shared solves), delta false
  (a/b/c 5/5/5 — repair and retrieval add nothing identical-task).
- Repair conversion b vs a: 0 of 5. Retrieval delta c vs b: 0.

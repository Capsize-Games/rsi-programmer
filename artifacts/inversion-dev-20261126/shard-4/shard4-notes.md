# Shard-4 notes (inversion-dev-20261126)

Shard tasks: `llm_dev_009` (A6), `llm_dev_018` (A7), `llm_dev_031`
(A8), `llm_dev_047` (S1), `llm_dev_062` (S2). Bank mapping confirmed
against `fixtures/llm_bank_v1/tasks.json` before running: all five
are `pool: dev` with the families/interfaces above. No bank
discrepancy, so no mapping override was needed.

Procedure: exact copy of the shard-5 no-gate runner
(`artifacts/inversion-dev-20261125/shard-5/run_shard5.py`), only the
`SHARD` tuple and docstring changed. Saved here as `run_shard4.py`.
Main commit `0f29072` code used as-is; judge, repair, protocol and bar
unmodified. Sealed pool never loaded (dev-only ids asserted in code).
No waiting, polling or retries; every CURLEE call went through the
judge resource guard.

## Result (all five arms, dual-backend final acceptance)

Per-arm final accepts (task-id lists): a=[], b=[], c=[], d=[], e=[].
Visible accepts likewise all zero. Shard verdict: KILL (no deep
reach, undefined cost ratio, counts a/b/c=0/0/0).

Per-arm CPU seconds (judge + proposer, from `summary.json`):
a=2.259, b=8.360, c=8.437, d=0.502, e=0.400.

Rejection taxonomy (final): a/b/c each `python_mismatch x4,
timeout x1`; d `python_mismatch x2` (plus one no-visible-match and
two statement-task not-applicable); e `python_mismatch x5`.
Visible-stage taxonomy is identical. Repair converted 0/5 round-0
rejections; retrieval delta c-vs-b is 0; library stayed empty (no
held-accepted a/b programs to qualify); CURLEE check-pass rate 0.0.

## Timeouts / guard refusals

Timeout verdicts: 1 task in each of arms a/b/c at both visible and
final stage (6 timeout verdicts total across the shard). The frozen
runner stores lite verdicts only, so per-case exit codes are not in
the retained artifacts and the exact task id plus rc75-vs-sandbox
attribution cannot be recovered post-hoc without re-execution.
Bounding evidence: whole-shard wall was ~24 s (uptime 679410.45 to
679434.68), which excludes genuine 30 s CURLEE subprocess timeouts;
the judge maps guard rc75 refusals to `timeout`, and six shards share
one CURLEE build slot, so contention refusal is the likely cause.
Recorded as timeout verdicts per the hard rules; `guard_refusals`
is therefore reported as inseparable-from-timeout (6 timeout
verdicts, 0 proven otherwise), not as a separately counted figure.

## Machine stats (record only)

Start: uptime 679410.45 s, MemAvailable 10417088 kB. End: uptime
679434.68 s, MemAvailable 9799152 kB. Full records in
`machine_start.json` / `machine_end.json`.

## Outputs in this directory

`summary.json` (runner aggregate), `inversion-dev-results.md`
(report), `protocol_sha256.txt`
(`608c63da9e0e55911e3c19a3e5ac8ad39f5f7f94079df206526672694f186e96`),
`run_shard4.py` (runner copy), `machine_start/end.json`,
`judge-work/`.

# Shard-2 notes (2026-11-26 NEW trial, cost levers at main 0f29072)

Scope: dev-only re-trial measuring python-first judge gating +
near-miss repair gate. Current main code used as-is; no judge /
repair / protocol / bar edits. Sealed bank never loaded (runner
asserts every id starts with `llm_dev_`; held filtered to shard).

Shard tasks (confirmed against fixtures/llm_bank_v1/tasks.json,
mapping matches, nothing differs): llm_dev_011 (A6 pair),
llm_dev_016 (A7 pair), llm_dev_043 (A8 pair), llm_dev_050 (S1
list), llm_dev_088 (S3 list).

Procedure: no-gate (machine stats record-only, no pre-wait, no
polling). No commit/push; outputs stay in this worktree only.

## Results (dual-backend final acceptance)

Per-arm final accepts: a=[], b=[], c=[], d=[], e=[] (0/5 each).
Visible accepts also 0/5 on every arm. Bar verdict KILL
(no deep reach, no shared solves, b-vs-a / c-vs-b deltas 0).
Library: 0 entries (nothing held-accepted; all 10 a/b qualify
attempts skipped for lack of held acceptance).

Per-arm CPU seconds (record cpu_seconds): a=1.139, b=5.554,
c=5.484, d=0.873, e=0.413. Repair conversion b: 0/5 round-0
rejections (rate 0.0). Retrieval delta c-vs-b: 0.

Final taxonomy: a/b/c = 4 python_mismatch + 1 timeout each;
d = 2 python_mismatch (3rd graph task: no visible match within
budget, no candidate); e = 5 python_mismatch. Visible
taxonomy: a/b/c = 3 python_mismatch + 2 timeout each.

## Guard refusals: 9 (all recorded as timeout verdicts)

CURLEE guard active (judge routes every CURLEE check/run through
/home/joe/Projects/curlee/scripts/run_limited.py, rc75 on lock
refusal; six shards share one CURLEE build slot). Evidence that
all 9 timeouts are refusals, not genuine timeouts:

- Whole-run wall 19.98 s (machine_start/end uptime delta);
  a genuine CURLEE timeout is 30 s per call, impossible here.
- Judge backend_execution ledger: 111 samples, max 0.066 s CPU,
  4.25 s total wall -- no 5 s python-sandbox timeout and no
  genuine CURLEE compile/run anywhere in the run.
- CURLEE-stage entry proven by written harness files
  (llm_dev_050 *_final.curlee, plus visible-stage files for
  _011/_043/_050/_088): python-first gating passed python, then
  CURLEE was refused. Final-stage refusals: 3 (a/b/c on
  llm_dev_050, which passed python on held). Visible-stage
  refusals: 6 (2 each on a/b/c). No retry, no waiting, per
  hard rules.

## Machine stats (record only)

Start: uptime 679409.34 s, MemAvailable 10435284 kB.
End: uptime 679429.32 s, MemAvailable 9317028 kB.

## Artefacts in this dir

run_shard2.py (pattern copied from
artifacts/inversion-dev-20261125/shard-5/run_shard5.py with SHARD
replaced), summary.json, inversion-dev-results.md,
protocol_sha256.txt (d97655a871c494757bb312a430809901a98f5413ac43889bba0976589ace1520),
machine_start.json, machine_end.json, judge-work/ (temp execution
files only).

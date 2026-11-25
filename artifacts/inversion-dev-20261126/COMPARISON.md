# Cost-lever trial vs frozen verdict (dev, 30 tasks)

New-trial code: main `0f29072` (python-first judge gating,
near-miss repair gate). Frozen reference: merged 2026-11-25
verdict a/b/c/d/e = 19/28/28/4/0, cost ratio 5.605 vs M=3.

## Headline: trial is contention-confounded, not a verdict

Six shards ran in parallel against one CURLEE build slot.
Shards 0-4 lost nearly everything to rc75 guard refusals
(30/6/9/25+/6 timeout verdicts) and scored 0/0/0/0/0 final.
Merged totals (5/5/5/0/0, ratio undefined, KILL) describe
the contention, not the levers. Only shard-5 has clean
finals and is comparable.

## Shard-5 vs shard-5 (same tasks, clean finals both runs)

Tasks 065/068/072/075/081. Accepts identical: 5/5/5/0/0.

| Arm | Frozen CPU s | New CPU s | Change |
| --- | ---: | ---: | --- |
| a | 3.930 | 3.154 | -20% |
| b | 3.901 | 2.904 | -26% |
| c | 3.927 | 3.829 | -2% |
| e | 3.297 | 0.253 | -92% |

Arm e never accepts; its frozen 3.3 s was pure CURLEE spend
on Python-failing candidates, now skipped. Arms b/c keep
every accept at lower cost. Caveat: the new run recorded
11 visible-stage guard refusals (finals unaffected);
the frozen run had none — repair paths may differ, and
repair conversion was 0/5 in both.

## What this does not establish

No new bar verdict: 25 of 30 tasks have no real
evaluations. The near-miss gate was unexercised (no
executable-but-wrong verdicts survived contention). The
sealed bank stays untouched. A verdict-grade comparison
needs a contention-free full run (sequential or
slot-separated), which is a new-epic decision.

## Sequential lever run (30/30, zero refusals)

`artifacts/inversion-dev-20261126-sequential/full/`:
single process, 1,250 s wall, 0 timeouts, 0 guard
refusals. Accepts a/b/c/d/e = 17/25/25/4/0 (frozen:
19/28/28/4/0). Streaming holds the same 4 shallow
tasks. Deep reach 13 streaming-0 tasks (frozen: 16;
016/018/088 lost, cause unattributed — near-miss gate
and proposer sampling both consistent). Repair
converted 8/11 round-0 rejects; retrieval delta 0.

Cost, same frozen bar function, measured per-task CPU
(`DevDiagnosticRunner.summary`, in-process records —
method-verified in code; per-task table not persisted,
only the ratio): **1.161 vs M=3 → PROCEED reference**.
Even-spread on the same data gives 7.470 — a 6.4x
approximation error. The frozen 5.605 KILL used that
same approximation, so it cannot be trusted as deciding
evidence either; only this exact measurement exists,
and it passes. Per-accept corroboration: c 1.423 s vs
d 1.191 s (ratio 1.195).

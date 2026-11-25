# RSI Inversion dev-diagnostic results

Protocol sha256: `5433a5c48ad780954c8aeca1f0a6526914fb9eb0845901bd31e7f33a5025eb17`
Card: `bce2d353-1794-4466-b747-fee7d0a1a04`, bank: `llm_bank_v1`
Model: `spark-1.3` (Spark 1.3 supervised background workers)
Tasks: 30 dev (5 per family over A6/A7/A8/S1/S2/S3)
Held opened identically: True

## Verdict: PROCEED

- deep reach: c holds ['llm_dev_031', 'llm_dev_039', 'llm_dev_042', 'llm_dev_043', 'llm_dev_062', 'llm_dev_065', 'llm_dev_068', 'llm_dev_072', 'llm_dev_073', 'llm_dev_075', 'llm_dev_076', 'llm_dev_081', 'llm_dev_083']
- cost/accepted ratio 1.161 vs M=3
- counts a/b/c=17/25/25

## Final accepts per arm/family

| family | a | b | c | d | e |
| --- | --- | --- | --- | --- | --- |
| A6 | 2 | 5 | 5 | 1 | 0 |
| A7 | 2 | 2 | 2 | 2 | 0 |
| A8 | 1 | 5 | 5 | 1 | 0 |
| S1 | 4 | 4 | 4 | 0 | 0 |
| S2 | 4 | 5 | 5 | 0 | 0 |
| S3 | 4 | 4 | 4 | 0 | 0 |

## Cost per accepted (CPU seconds)

| arm | total CPU | accepts | per accepted | marginal |
| --- | --- | --- | --- | --- |
| a | 17.016 | 17 | 1.001 | n/a |
| b | 38.554 | 25 | 1.542 | 2.692 |
| c | 35.583 | 25 | 1.423 | n/a |
| d | 4.764 | 4 | 1.191 | n/a |
| e | 2.197 | 0 | n/a | n/a |

## CURLEE check-pass rate (final stage)

0.5461538461538461

## Repair conversion (b vs a)

converted 8 of 11 round-0 rejections (rate 0.7272727272727273)

## Retrieval delta (c vs b, identical tasks)

delta 0; c-only []; b-only []

## Rejection taxonomy (final)

- a: {'python_mismatch': 13}
- b: {'python_mismatch': 5}
- c: {'python_mismatch': 5}
- d: {'python_mismatch': 6}
- e: {'python_mismatch': 30}

## Paired same-task discordance

best arm c vs streaming: both ['llm_dev_010', 'llm_dev_021', 'llm_dev_022', 'llm_dev_030']; llm-only ['llm_dev_002', 'llm_dev_004', 'llm_dev_009', 'llm_dev_011', 'llm_dev_031', 'llm_dev_039', 'llm_dev_042', 'llm_dev_043', 'llm_dev_045', 'llm_dev_047', 'llm_dev_048', 'llm_dev_050', 'llm_dev_062', 'llm_dev_065', 'llm_dev_068', 'llm_dev_072', 'llm_dev_073', 'llm_dev_075', 'llm_dev_076', 'llm_dev_081', 'llm_dev_083']; streaming-only {'best_arm': 'c', 'both': ['llm_dev_010', 'llm_dev_021', 'llm_dev_022', 'llm_dev_030'], 'llm_only': ['llm_dev_002', 'llm_dev_004', 'llm_dev_009', 'llm_dev_011', 'llm_dev_031', 'llm_dev_039', 'llm_dev_042', 'llm_dev_043', 'llm_dev_045', 'llm_dev_047', 'llm_dev_048', 'llm_dev_050', 'llm_dev_062', 'llm_dev_065', 'llm_dev_068', 'llm_dev_072', 'llm_dev_073', 'llm_dev_075', 'llm_dev_076', 'llm_dev_081', 'llm_dev_083'], 'streaming_only': [], 'neither': ['llm_dev_016', 'llm_dev_018', 'llm_dev_020', 'llm_dev_049', 'llm_dev_088']}['streaming_only']; neither ['llm_dev_016', 'llm_dev_018', 'llm_dev_020', 'llm_dev_049', 'llm_dev_088']

## Honesty notes

- Single run, one seed: development evidence only, not a capability claim.
- Dev held rows are tunable; the sealed pool was never loaded (dev-only ids asserted in code).
- Costs: proposer CPU is harness wall, tokens worker-reported, every judge phase ledgered per arm; exclusions recorded in the summary ledgers.

## Full sequential run (30 tasks, one process)

Tasks: 30 dev, frozen order (A6/A7/A8/S1/S2/S3 x 5). Mapping confirmed against fixtures/llm_bank_v1/tasks.json: union of the six 20261125 shard task lists equals the frozen 30-task set; all pool=dev.
Protocol sha256: `5433a5c48ad780954c8aeca1f0a6526914fb9eb0845901bd31e7f33a5025eb17` (task-subset hash; content frozen).
Code: main `2f34a912158a419848bd7e25022c18fa79c4aa6e` as-is (python-first judge gating + near-miss repair gate); judge/repair/protocol/bar untouched.

### Per-arm FINAL accepts (dual-backend)

- a: ['llm_dev_004', 'llm_dev_010', 'llm_dev_021', 'llm_dev_022', 'llm_dev_030', 'llm_dev_045', 'llm_dev_047', 'llm_dev_048', 'llm_dev_050', 'llm_dev_065', 'llm_dev_068', 'llm_dev_072', 'llm_dev_073', 'llm_dev_075', 'llm_dev_076', 'llm_dev_081', 'llm_dev_083'] (17/30)
- b: ['llm_dev_002', 'llm_dev_004', 'llm_dev_009', 'llm_dev_010', 'llm_dev_011', 'llm_dev_021', 'llm_dev_022', 'llm_dev_030', 'llm_dev_031', 'llm_dev_039', 'llm_dev_042', 'llm_dev_043', 'llm_dev_045', 'llm_dev_047', 'llm_dev_048', 'llm_dev_050', 'llm_dev_062', 'llm_dev_065', 'llm_dev_068', 'llm_dev_072', 'llm_dev_073', 'llm_dev_075', 'llm_dev_076', 'llm_dev_081', 'llm_dev_083'] (25/30)
- c: ['llm_dev_002', 'llm_dev_004', 'llm_dev_009', 'llm_dev_010', 'llm_dev_011', 'llm_dev_021', 'llm_dev_022', 'llm_dev_030', 'llm_dev_031', 'llm_dev_039', 'llm_dev_042', 'llm_dev_043', 'llm_dev_045', 'llm_dev_047', 'llm_dev_048', 'llm_dev_050', 'llm_dev_062', 'llm_dev_065', 'llm_dev_068', 'llm_dev_072', 'llm_dev_073', 'llm_dev_075', 'llm_dev_076', 'llm_dev_081', 'llm_dev_083'] (25/30)
- d: ['llm_dev_010', 'llm_dev_021', 'llm_dev_022', 'llm_dev_030'] (4/30)
- e: [] (0/30)

### CPU seconds (per-arm record totals)

a 17.016, b 38.554, c 35.583, d 4.764, e 2.197.

### Guard refusals

visible timeouts 0, final timeouts 0; total 0 timeout verdicts (upper bound on rc75 refusals; detail string not persisted). No waiting or polling was performed; machine stats are record-only.
Machine start uptime 680136.07 s, load [1.56, 2.08, 2.14], MemAvailable 9754528 kB; end uptime 681386.15 s, load [1.47, 1.81, 1.95], MemAvailable 9562576 kB.

### Cost ratio (same even-spread way)

Shared tasks (best vs streaming): llm_dev_010, llm_dev_021, llm_dev_022, llm_dev_030.
Exact per-task ratio from the frozen bar: 1.1611593755628198. Even-spread ratio (arm totals, shared count cancels): 7.4697458314546665. Frozen bar multiple M=3.

### Reference bar outcome

Frozen pre-lever verdict (merged 30-task, no levers): a/b/c/d/e = 19/28/28/4/0, cost ratio 5.605 vs M=3 -> KILL (deep PASS, cost FAIL, delta PASS).

This sequential lever run under the SAME even-spread accounting: a/b/c/d/e = 17/25/25/4/0, even-spread ratio 7.470 vs M=3 -> cost prong FAILS (deep PASS with 13 streaming-0 tasks, delta PASS with b_only 8) -> reference-bar outcome KILL, same shape as frozen and further from the multiple. The frozen bar function on true per-task data instead returns PROCEED (exact ratio 1.161): the merge approximation overstates LLM cost because streaming CPU concentrates on the 4 shared shallow tasks while LLM CPU spreads over 13 deep accepts where streaming spends ~0. Both figures are recorded; the even-spread one is the frozen-comparable verdict input, the exact one is what the coded bar computes on real per-task data. Single run, one seed: neither is a capability claim, and the sealed bank stays untouched.

### Vs contention-confounded parallel trial

20261126 parallel run (same lever code, six shards on one CURLEE slot): shards 0-4 lost to rc75 refusals and scored 0/0/0/0/0; merged 5/5/5/0/0 with undefined ratio is a contention record, not a verdict. Only its shard-5 (same 5 tasks, clean finals) is comparable: identical 5/5/5/0/0 accepts at -20/-26/-2/-92% CPU (a/b/c/e). This sequential run replaces that caveat with contention-free finals on all 30 tasks; compare its accepts and even-spread ratio directly against the frozen 19/28/28/4/0 and 5.605 figures above.

### Honesty notes

- Single run, one seed: development evidence only, not a capability claim.
- Dev held rows tunable; sealed pool never loaded (dev-only ids asserted in code).
- No commit/push; nothing outside this worktree touched (binaries read/execute only via the judge).
- Check-pass rate 0.546 needs its denominator: all 71 dual-backend accepts passed the CURLEE check stage; the 59 rejected finals carry no check commands because python-first gating skips CURLEE on python failures (arm e: 30/30 python_mismatch, total CPU 2.197 vs frozen 15.7). The frozen 1.0 came from CURLEE checks that ran and exited 0 even on wrong programs.
- Repair gate was exercised here (unlike the parallel trial): 8 conversions of 11 round-0 rejections. Retrieval added nothing (c==b identical sets, 25-entry library built from dev-held-accepted a/b programs). Three fewer b/c accepts than frozen (25 vs 28; missing 016/018/020/049/088) is consistent with the near-miss gate stopping non-executable rounds early, but that mechanism is unattributed without frozen per-task sets: recorded as an observed difference, not a proven cause.

# RSI sealed one-shot results (cost lever)

Protocol sha256: `88265f50d80b7f6e1b9412d6fe45d22e42042879c5d3e5fa3e4feda708c7c482`
Design: `tracks/programmer/lever-sealed-design.md`
Model: `spark-1.3` (Spark 1.3 supervised background workers)
Tasks: 90 sealed (15 per family over A6/A7/A8/S1/S2/S3)
Held opened identically: True
Resource guard active: True

## Verdict: PROCEED

- deep reach: c holds ['llm_sealed_020', 'llm_sealed_022', 'llm_sealed_023', 'llm_sealed_028', 'llm_sealed_029', 'llm_sealed_034', 'llm_sealed_036', 'llm_sealed_037', 'llm_sealed_039', 'llm_sealed_040', 'llm_sealed_041', 'llm_sealed_043', 'llm_sealed_060', 'llm_sealed_061', 'llm_sealed_063', 'llm_sealed_064', 'llm_sealed_065', 'llm_sealed_066', 'llm_sealed_067', 'llm_sealed_068', 'llm_sealed_069', 'llm_sealed_070', 'llm_sealed_071', 'llm_sealed_072', 'llm_sealed_073', 'llm_sealed_074', 'llm_sealed_075', 'llm_sealed_076', 'llm_sealed_077', 'llm_sealed_078', 'llm_sealed_079', 'llm_sealed_080', 'llm_sealed_081', 'llm_sealed_082', 'llm_sealed_083', 'llm_sealed_084', 'llm_sealed_085', 'llm_sealed_086', 'llm_sealed_087', 'llm_sealed_088', 'llm_sealed_089']
- cost/accepted ratio 1.108 vs M=3
- counts a/b/c=46/73/73

## Final accepts per arm/family

| family | a | b | c | d | e |
| --- | --- | --- | --- | --- |
| A6 | 8 | 15 | 15 | 4 | 0 |
| A7 | 5 | 9 | 9 | 4 | 0 |
| A8 | 1 | 8 | 8 | 1 | 0 |
| S1 | 12 | 12 | 12 | 0 | 0 |
| S2 | 11 | 14 | 14 | 0 | 0 |
| S3 | 9 | 15 | 15 | 0 | 0 |

## Cost per accepted (CPU seconds)

| arm | total CPU | accepts | per accepted | marginal |
| --- | --- | --- | --- | --- |
| a | 68.283 | 46 | 1.484 | n/a |
| b | 120.032 | 73 | 1.644 | 1.917 |
| c | 109.306 | 73 | 1.497 | n/a |
| d | 10.667 | 9 | 1.185 | n/a |
| e | 6.444 | 0 | n/a | n/a |

## CURLEE check-pass rate (final stage)

0.5089974293059126

## Repair conversion (b vs a)

converted 28 of 41 round-0 rejections (rate 0.6829268292682927)

## Retrieval delta (c vs b, identical tasks)

delta 0; c-only []; b-only []

## Rejection taxonomy (final)

- a: {'python_mismatch': 44}
- b: {'python_mismatch': 17}
- c: {'python_mismatch': 17}
- d: {'python_mismatch': 20}
- e: {'python_mismatch': 90}

## Paired same-task discordance

best arm c vs streaming: both ['llm_sealed_000', 'llm_sealed_003', 'llm_sealed_009', 'llm_sealed_014', 'llm_sealed_017', 'llm_sealed_024', 'llm_sealed_025', 'llm_sealed_026', 'llm_sealed_032']; llm-only ['llm_sealed_001', 'llm_sealed_002', 'llm_sealed_004', 'llm_sealed_005', 'llm_sealed_006', 'llm_sealed_007', 'llm_sealed_008', 'llm_sealed_010', 'llm_sealed_011', 'llm_sealed_012', 'llm_sealed_013', 'llm_sealed_020', 'llm_sealed_022', 'llm_sealed_023', 'llm_sealed_028', 'llm_sealed_029', 'llm_sealed_034', 'llm_sealed_036', 'llm_sealed_037', 'llm_sealed_039', 'llm_sealed_040', 'llm_sealed_041', 'llm_sealed_043', 'llm_sealed_045', 'llm_sealed_046', 'llm_sealed_047', 'llm_sealed_049', 'llm_sealed_050', 'llm_sealed_051', 'llm_sealed_052', 'llm_sealed_053', 'llm_sealed_054', 'llm_sealed_056', 'llm_sealed_057', 'llm_sealed_059', 'llm_sealed_060', 'llm_sealed_061', 'llm_sealed_063', 'llm_sealed_064', 'llm_sealed_065', 'llm_sealed_066', 'llm_sealed_067', 'llm_sealed_068', 'llm_sealed_069', 'llm_sealed_070', 'llm_sealed_071', 'llm_sealed_072', 'llm_sealed_073', 'llm_sealed_074', 'llm_sealed_075', 'llm_sealed_076', 'llm_sealed_077', 'llm_sealed_078', 'llm_sealed_079', 'llm_sealed_080', 'llm_sealed_081', 'llm_sealed_082', 'llm_sealed_083', 'llm_sealed_084', 'llm_sealed_085', 'llm_sealed_086', 'llm_sealed_087', 'llm_sealed_088', 'llm_sealed_089']; streaming-only {'best_arm': 'c', 'both': ['llm_sealed_000', 'llm_sealed_003', 'llm_sealed_009', 'llm_sealed_014', 'llm_sealed_017', 'llm_sealed_024', 'llm_sealed_025', 'llm_sealed_026', 'llm_sealed_032'], 'llm_only': ['llm_sealed_001', 'llm_sealed_002', 'llm_sealed_004', 'llm_sealed_005', 'llm_sealed_006', 'llm_sealed_007', 'llm_sealed_008', 'llm_sealed_010', 'llm_sealed_011', 'llm_sealed_012', 'llm_sealed_013', 'llm_sealed_020', 'llm_sealed_022', 'llm_sealed_023', 'llm_sealed_028', 'llm_sealed_029', 'llm_sealed_034', 'llm_sealed_036', 'llm_sealed_037', 'llm_sealed_039', 'llm_sealed_040', 'llm_sealed_041', 'llm_sealed_043', 'llm_sealed_045', 'llm_sealed_046', 'llm_sealed_047', 'llm_sealed_049', 'llm_sealed_050', 'llm_sealed_051', 'llm_sealed_052', 'llm_sealed_053', 'llm_sealed_054', 'llm_sealed_056', 'llm_sealed_057', 'llm_sealed_059', 'llm_sealed_060', 'llm_sealed_061', 'llm_sealed_063', 'llm_sealed_064', 'llm_sealed_065', 'llm_sealed_066', 'llm_sealed_067', 'llm_sealed_068', 'llm_sealed_069', 'llm_sealed_070', 'llm_sealed_071', 'llm_sealed_072', 'llm_sealed_073', 'llm_sealed_074', 'llm_sealed_075', 'llm_sealed_076', 'llm_sealed_077', 'llm_sealed_078', 'llm_sealed_079', 'llm_sealed_080', 'llm_sealed_081', 'llm_sealed_082', 'llm_sealed_083', 'llm_sealed_084', 'llm_sealed_085', 'llm_sealed_086', 'llm_sealed_087', 'llm_sealed_088', 'llm_sealed_089'], 'streaming_only': [], 'neither': ['llm_sealed_015', 'llm_sealed_016', 'llm_sealed_018', 'llm_sealed_019', 'llm_sealed_021', 'llm_sealed_027', 'llm_sealed_030', 'llm_sealed_031', 'llm_sealed_033', 'llm_sealed_035', 'llm_sealed_038', 'llm_sealed_042', 'llm_sealed_044', 'llm_sealed_048', 'llm_sealed_055', 'llm_sealed_058', 'llm_sealed_062']}['streaming_only']; neither ['llm_sealed_015', 'llm_sealed_016', 'llm_sealed_018', 'llm_sealed_019', 'llm_sealed_021', 'llm_sealed_027', 'llm_sealed_030', 'llm_sealed_031', 'llm_sealed_033', 'llm_sealed_035', 'llm_sealed_038', 'llm_sealed_042', 'llm_sealed_044', 'llm_sealed_048', 'llm_sealed_055', 'llm_sealed_058', 'llm_sealed_062']

## Honesty notes

- Single run, one seed: sealed evidence, not a capability claim.
- Sealed held rows opened only through judge_final, identically for every arm; repair/query feedback never saw held rows (held scope raises inside the repo code).
- Dev pool never loaded: any llm_dev_ key in any split aborts before execution.
- Costs: proposer CPU is harness wall, tokens worker-reported, every judge phase ledgered per arm; exclusions recorded in the summary ledgers.

## Sealed one-shot (90 tasks, one process)

Tasks: 90 sealed, frozen bank order (A6/A7/A8/S1/S2/S3 x 15). Bank splits carry no llm_dev_ key (asserted before execution); all held rows pool=sealed.
Protocol sha256: `88265f50d80b7f6e1b9412d6fe45d22e42042879c5d3e5fa3e4feda708c7c482` (sealed-run record; canonical JSON of the frozen sealed parameters above).
Code: main `8c1d70d3e9f60ef9b20f01cc0fd421b738e2aa36` as-is; judge/repair/protocol/bar modules unmodified (imported, not subclassed).

### Per-arm FINAL accepts (dual-backend)

90-task sealed pool:

- a: ['llm_sealed_000', 'llm_sealed_003', 'llm_sealed_005', 'llm_sealed_007', 'llm_sealed_009', 'llm_sealed_011', 'llm_sealed_013', 'llm_sealed_014', 'llm_sealed_017', 'llm_sealed_024', 'llm_sealed_025', 'llm_sealed_026', 'llm_sealed_028', 'llm_sealed_032', 'llm_sealed_045', 'llm_sealed_046', 'llm_sealed_047', 'llm_sealed_049', 'llm_sealed_050', 'llm_sealed_051', 'llm_sealed_052', 'llm_sealed_053', 'llm_sealed_054', 'llm_sealed_056', 'llm_sealed_057', 'llm_sealed_059', 'llm_sealed_060', 'llm_sealed_061', 'llm_sealed_063', 'llm_sealed_064', 'llm_sealed_065', 'llm_sealed_066', 'llm_sealed_067', 'llm_sealed_068', 'llm_sealed_069', 'llm_sealed_073', 'llm_sealed_074', 'llm_sealed_076', 'llm_sealed_078', 'llm_sealed_079', 'llm_sealed_080', 'llm_sealed_081', 'llm_sealed_084', 'llm_sealed_085', 'llm_sealed_086', 'llm_sealed_089'] (46/90)
- b: ['llm_sealed_000', 'llm_sealed_001', 'llm_sealed_002', 'llm_sealed_003', 'llm_sealed_004', 'llm_sealed_005', 'llm_sealed_006', 'llm_sealed_007', 'llm_sealed_008', 'llm_sealed_009', 'llm_sealed_010', 'llm_sealed_011', 'llm_sealed_012', 'llm_sealed_013', 'llm_sealed_014', 'llm_sealed_017', 'llm_sealed_020', 'llm_sealed_022', 'llm_sealed_023', 'llm_sealed_024', 'llm_sealed_025', 'llm_sealed_026', 'llm_sealed_028', 'llm_sealed_029', 'llm_sealed_032', 'llm_sealed_034', 'llm_sealed_036', 'llm_sealed_037', 'llm_sealed_039', 'llm_sealed_040', 'llm_sealed_041', 'llm_sealed_043', 'llm_sealed_045', 'llm_sealed_046', 'llm_sealed_047', 'llm_sealed_049', 'llm_sealed_050', 'llm_sealed_051', 'llm_sealed_052', 'llm_sealed_053', 'llm_sealed_054', 'llm_sealed_056', 'llm_sealed_057', 'llm_sealed_059', 'llm_sealed_060', 'llm_sealed_061', 'llm_sealed_063', 'llm_sealed_064', 'llm_sealed_065', 'llm_sealed_066', 'llm_sealed_067', 'llm_sealed_068', 'llm_sealed_069', 'llm_sealed_070', 'llm_sealed_071', 'llm_sealed_072', 'llm_sealed_073', 'llm_sealed_074', 'llm_sealed_075', 'llm_sealed_076', 'llm_sealed_077', 'llm_sealed_078', 'llm_sealed_079', 'llm_sealed_080', 'llm_sealed_081', 'llm_sealed_082', 'llm_sealed_083', 'llm_sealed_084', 'llm_sealed_085', 'llm_sealed_086', 'llm_sealed_087', 'llm_sealed_088', 'llm_sealed_089'] (73/90)
- c: ['llm_sealed_000', 'llm_sealed_001', 'llm_sealed_002', 'llm_sealed_003', 'llm_sealed_004', 'llm_sealed_005', 'llm_sealed_006', 'llm_sealed_007', 'llm_sealed_008', 'llm_sealed_009', 'llm_sealed_010', 'llm_sealed_011', 'llm_sealed_012', 'llm_sealed_013', 'llm_sealed_014', 'llm_sealed_017', 'llm_sealed_020', 'llm_sealed_022', 'llm_sealed_023', 'llm_sealed_024', 'llm_sealed_025', 'llm_sealed_026', 'llm_sealed_028', 'llm_sealed_029', 'llm_sealed_032', 'llm_sealed_034', 'llm_sealed_036', 'llm_sealed_037', 'llm_sealed_039', 'llm_sealed_040', 'llm_sealed_041', 'llm_sealed_043', 'llm_sealed_045', 'llm_sealed_046', 'llm_sealed_047', 'llm_sealed_049', 'llm_sealed_050', 'llm_sealed_051', 'llm_sealed_052', 'llm_sealed_053', 'llm_sealed_054', 'llm_sealed_056', 'llm_sealed_057', 'llm_sealed_059', 'llm_sealed_060', 'llm_sealed_061', 'llm_sealed_063', 'llm_sealed_064', 'llm_sealed_065', 'llm_sealed_066', 'llm_sealed_067', 'llm_sealed_068', 'llm_sealed_069', 'llm_sealed_070', 'llm_sealed_071', 'llm_sealed_072', 'llm_sealed_073', 'llm_sealed_074', 'llm_sealed_075', 'llm_sealed_076', 'llm_sealed_077', 'llm_sealed_078', 'llm_sealed_079', 'llm_sealed_080', 'llm_sealed_081', 'llm_sealed_082', 'llm_sealed_083', 'llm_sealed_084', 'llm_sealed_085', 'llm_sealed_086', 'llm_sealed_087', 'llm_sealed_088', 'llm_sealed_089'] (73/90)
- d: ['llm_sealed_000', 'llm_sealed_003', 'llm_sealed_009', 'llm_sealed_014', 'llm_sealed_017', 'llm_sealed_024', 'llm_sealed_025', 'llm_sealed_026', 'llm_sealed_032'] (9/90)
- e: [] (0/90)

### CPU seconds (per-arm record totals)

a 68.283, b 120.032, c 109.306, d 10.667, e 6.444.

### Guard refusals

guard active True; visible timeouts 4, final timeouts 0; total 4 timeout verdicts (upper bound on rc75 refusals; detail string not persisted). No waiting or polling was performed; machine stats are record-only.
Machine start uptime 683538.99 s, load [2.21, 2.36, 1.75], MemAvailable 10303648 kB; end uptime 687103.35 s, load [1.07, 1.18, 1.53], MemAvailable 10626104 kB.

### Cost ratio (frozen bar vs comparison)

Shared tasks (best vs streaming): llm_sealed_000, llm_sealed_003, llm_sealed_009, llm_sealed_014, llm_sealed_017, llm_sealed_024, llm_sealed_025, llm_sealed_026, llm_sealed_032.
Exact per-task ratio from the frozen bar (cpu_by_task, shared solves only): 1.107631430144453. Even-spread ratio (arm totals, shared count cancels) for comparison only: 10.247125084615638. Frozen bar multiple M=3.

### Frozen-bar prongs

deep reach: ['llm_sealed_020', 'llm_sealed_022', 'llm_sealed_023', 'llm_sealed_028', 'llm_sealed_029', 'llm_sealed_034', 'llm_sealed_036', 'llm_sealed_037', 'llm_sealed_039', 'llm_sealed_040', 'llm_sealed_041', 'llm_sealed_043', 'llm_sealed_060', 'llm_sealed_061', 'llm_sealed_063', 'llm_sealed_064', 'llm_sealed_065', 'llm_sealed_066', 'llm_sealed_067', 'llm_sealed_068', 'llm_sealed_069', 'llm_sealed_070', 'llm_sealed_071', 'llm_sealed_072', 'llm_sealed_073', 'llm_sealed_074', 'llm_sealed_075', 'llm_sealed_076', 'llm_sealed_077', 'llm_sealed_078', 'llm_sealed_079', 'llm_sealed_080', 'llm_sealed_081', 'llm_sealed_082', 'llm_sealed_083', 'llm_sealed_084', 'llm_sealed_085', 'llm_sealed_086', 'llm_sealed_087', 'llm_sealed_088', 'llm_sealed_089']; counts {'a': 46, 'b': 73, 'c': 73}; verdict PROCEED.

### Honesty notes

- Single run, one seed: sealed one-shot evidence only, not a capability claim.
- Sealed held rows never entered repair or query feedback (judge_final only; any held-scope access outside final aborts).
- No commit/push; nothing outside this worktree touched (binaries read/execute only via the judge).

# RSI Inversion dev-diagnostic results

Protocol sha256: `c1ee97b7f972a37898560bb94b3fa21833ccc820e5bf9cb9d8ee80023c7edb15`
Card: `bce2d353-1794-4466-b747-fee7d0a1a04`, bank: `llm_bank_v1`
Model: `spark-1.3` (Spark 1.3 supervised background workers)
Tasks: 5 dev (5 per family over A6/A7/A8/S1/S2/S3)
Held opened identically: True

## Verdict: PROCEED

- deep reach: c holds ['llm_dev_039', 'llm_dev_083']
- cost/accepted ratio 1.124 vs M=3
- counts a/b/c=4/5/5

## Final accepts per arm/family

| family | a | b | c | d | e |
| --- | --- | --- | --- | --- | --- |
| A6 | 1 | 1 | 1 | 1 | 0 |
| A7 | 1 | 1 | 1 | 1 | 0 |
| A8 | 0 | 1 | 1 | 0 | 0 |
| S1 | 1 | 1 | 1 | 0 | 0 |
| S3 | 1 | 1 | 1 | 0 | 0 |

## Cost per accepted (CPU seconds)

| arm | total CPU | accepts | per accepted | marginal |
| --- | --- | --- | --- | --- |
| a | 2.739 | 4 | 0.685 | n/a |
| b | 3.484 | 5 | 0.697 | 0.745 |
| c | 2.836 | 5 | 0.567 | n/a |
| d | 1.220 | 2 | 0.610 | n/a |
| e | 2.392 | 0 | n/a | n/a |

## CURLEE check-pass rate (final stage)

1.0

## Repair conversion (b vs a)

converted 1 of 1 round-0 rejections (rate 1.0)

## Retrieval delta (c vs b, identical tasks)

delta 0; c-only []; b-only []

## Rejection taxonomy (final)

- a: {'python_mismatch': 1}
- b: no rejections
- c: no rejections
- d: {'python_mismatch': 1}
- e: {'python_mismatch': 5}

## Paired same-task discordance

best arm c vs streaming: both ['llm_dev_010', 'llm_dev_021']; llm-only ['llm_dev_039', 'llm_dev_045', 'llm_dev_083']; streaming-only {'best_arm': 'c', 'both': ['llm_dev_010', 'llm_dev_021'], 'llm_only': ['llm_dev_039', 'llm_dev_045', 'llm_dev_083'], 'streaming_only': [], 'neither': []}['streaming_only']; neither []

## Honesty notes

- Single run, one seed: development evidence only, not a capability claim.
- Dev held rows are tunable; the sealed pool was never loaded (dev-only ids asserted in code).
- Costs: proposer CPU is harness wall, tokens worker-reported, every judge phase ledgered per arm; exclusions recorded in the summary ledgers.


## Guard refusals (rc75 timeouts)

uses_guard: True

total: 0; per arm: {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}

## Control-validity caveat

Arm-d streaming control reaches only ~2.6% into the size-6 enumeration layer, so d=0 on A6+ tasks is by construction, not a capability signal.

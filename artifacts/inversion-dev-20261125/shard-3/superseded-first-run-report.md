# RSI Inversion dev-diagnostic results

Protocol sha256: `c1ee97b7f972a37898560bb94b3fa21833ccc820e5bf9cb9d8ee80023c7edb15`
Card: `bce2d353-1794-4466-b747-fee7d0a1a04`, bank: `llm_bank_v1`
Model: `spark-1.3` (Spark 1.3 supervised background workers)
Tasks: 5 dev (5 per family over A6/A7/A8/S1/S2/S3)
Held opened identically: True

## Verdict: KILL

- no deep reach: a adds nothing where streaming scores 0
- cost ratio undefined: no shared solves or zero streaming cost
- counts a/b/c=0/0/0

## Final accepts per arm/family

| family | a | b | c | d | e |
| --- | --- | --- | --- | --- | --- |
| A6 | 0 | 0 | 0 | 0 | 0 |
| A7 | 0 | 0 | 0 | 0 | 0 |
| A8 | 0 | 0 | 0 | 0 | 0 |
| S1 | 0 | 0 | 0 | 0 | 0 |
| S3 | 0 | 0 | 0 | 0 | 0 |

## Cost per accepted (CPU seconds)

| arm | total CPU | accepts | per accepted | marginal |
| --- | --- | --- | --- | --- |
| a | 0.335 | 0 | n/a | n/a |
| b | 3.857 | 0 | n/a | n/a |
| c | 3.816 | 0 | n/a | n/a |
| d | 0.232 | 0 | n/a | n/a |
| e | 0.213 | 0 | n/a | n/a |

## CURLEE check-pass rate (final stage)

0.0

## Repair conversion (b vs a)

converted 0 of 5 round-0 rejections (rate 0.0)

## Retrieval delta (c vs b, identical tasks)

delta 0; c-only []; b-only []

## Rejection taxonomy (final)

- a: {'timeout': 5}
- b: {'timeout': 5}
- c: {'timeout': 5}
- d: {'timeout': 3}
- e: {'timeout': 5}

## Paired same-task discordance

best arm a vs streaming: both []; llm-only []; streaming-only {'best_arm': 'a', 'both': [], 'llm_only': [], 'streaming_only': [], 'neither': ['llm_dev_010', 'llm_dev_021', 'llm_dev_039', 'llm_dev_045', 'llm_dev_083']}['streaming_only']; neither ['llm_dev_010', 'llm_dev_021', 'llm_dev_039', 'llm_dev_045', 'llm_dev_083']

## Honesty notes

- Single run, one seed: development evidence only, not a capability claim.
- Dev held rows are tunable; the sealed pool was never loaded (dev-only ids asserted in code).
- Costs: proposer CPU is harness wall, tokens worker-reported, every judge phase ledgered per arm; exclusions recorded in the summary ledgers.


## Guard refusals (rc75 timeouts)

uses_guard: True

total: 23; per arm: {'a': 5, 'b': 5, 'c': 5, 'd': 3, 'e': 5}

## Control-validity caveat

Arm-d streaming control reaches only ~2.6% into the size-6 enumeration layer, so d=0 on A6+ tasks is by construction, not a capability signal.

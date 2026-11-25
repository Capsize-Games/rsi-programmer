# RSI Inversion dev-diagnostic results

Protocol sha256: `d8754a641e94dc9eeda0ed796ffd6f5982d561e5dd56bd5f19bf4af67cf1c7d8`
Card: `bce2d353-1794-4466-b747-fee7d0a1a04`, bank: `llm_bank_v1`
Model: `spark-1.3` (Spark 1.3 supervised background workers)
Tasks: 5 dev (5 per family over A6/A7/A8/S1/S2/S3)
Held opened identically: True

## Verdict: PROCEED

- deep reach: c holds ['llm_dev_073']
- cost/accepted ratio 0.826 vs M=3
- counts a/b/c=2/4/4

## Final accepts per arm/family

| family | a | b | c | d | e |
| --- | --- | --- | --- | --- | --- |
| A6 | 0 | 0 | 0 | 0 | 0 |
| A7 | 0 | 1 | 1 | 1 | 0 |
| A8 | 0 | 1 | 1 | 1 | 0 |
| S1 | 1 | 1 | 1 | 0 | 0 |
| S2 | 1 | 1 | 1 | 0 | 0 |

## Cost per accepted (CPU seconds)

| arm | total CPU | accepts | per accepted | marginal |
| --- | --- | --- | --- | --- |
| a | 2.376 | 2 | 1.188 | n/a |
| b | 2.517 | 4 | 0.629 | 0.071 |
| c | 2.395 | 4 | 0.599 | n/a |
| d | 1.312 | 2 | 0.656 | n/a |
| e | 2.434 | 0 | n/a | n/a |

## CURLEE check-pass rate (final stage)

0.8636363636363636

## Repair conversion (b vs a)

converted 2 of 3 round-0 rejections (rate 0.6666666666666666)

## Retrieval delta (c vs b, identical tasks)

delta 0; c-only []; b-only []

## Rejection taxonomy (final)

- a: {'budget': 1, 'python_mismatch': 2}
- b: {'budget': 1}
- c: {'budget': 1}
- d: no rejections
- e: {'python_mismatch': 5}

## Paired same-task discordance

best arm c vs streaming: both ['llm_dev_022', 'llm_dev_030']; llm-only ['llm_dev_049', 'llm_dev_073']; streaming-only {'best_arm': 'c', 'both': ['llm_dev_022', 'llm_dev_030'], 'llm_only': ['llm_dev_049', 'llm_dev_073'], 'streaming_only': [], 'neither': ['llm_dev_004']}['streaming_only']; neither ['llm_dev_004']

## Honesty notes

- Single run, one seed: development evidence only, not a capability claim.
- Dev held rows are tunable; the sealed pool was never loaded (dev-only ids asserted in code).
- Costs: proposer CPU is harness wall, tokens worker-reported, every judge phase ledgered per arm; exclusions recorded in the summary ledgers.


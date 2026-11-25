# RSI Inversion dev-diagnostic results

Protocol sha256: `d9a61b16c555161401c8eec73e32290e38b6100f92e3da4f7a96c518de666b06`
Card: `bce2d353-1794-4466-b747-fee7d0a1a04`, bank: `llm_bank_v1`
Model: `spark-1.3` (Spark 1.3 supervised background workers)
Tasks: 5 dev (5 per family over A6/A7/A8/S1/S2/S3)
Held opened identically: True

## Verdict: KILL

- deep reach: c holds ['llm_dev_042', 'llm_dev_076']
- cost ratio undefined: no shared solves or zero streaming cost
- counts a/b/c=2/4/4

## Final accepts per arm/family

| family | a | b | c | d | e |
| --- | --- | --- | --- | --- | --- |
| A6 | 0 | 1 | 1 | 0 | 0 |
| A7 | 0 | 0 | 0 | 0 | 0 |
| A8 | 0 | 1 | 1 | 0 | 0 |
| S1 | 1 | 1 | 1 | 0 | 0 |
| S3 | 1 | 1 | 1 | 0 | 0 |

## Cost per accepted (CPU seconds)

| arm | total CPU | accepts | per accepted | marginal |
| --- | --- | --- | --- | --- |
| a | 3.550 | 2 | 1.775 | n/a |
| b | 7.659 | 4 | 1.915 | 2.055 |
| c | 6.217 | 4 | 1.554 | n/a |
| d | 0.965 | 0 | n/a | n/a |
| e | 2.575 | 0 | n/a | n/a |

## CURLEE check-pass rate (final stage)

1.0

## Repair conversion (b vs a)

converted 2 of 3 round-0 rejections (rate 0.6666666666666666)

## Retrieval delta (c vs b, identical tasks)

delta 0; c-only []; b-only []

## Rejection taxonomy (final)

- a: {'python_mismatch': 3}
- b: {'python_mismatch': 1}
- c: {'python_mismatch': 1}
- d: {'python_mismatch': 1}
- e: {'python_mismatch': 5}

## Paired same-task discordance

best arm c vs streaming: both []; llm-only ['llm_dev_002', 'llm_dev_042', 'llm_dev_048', 'llm_dev_076']; streaming-only {'best_arm': 'c', 'both': [], 'llm_only': ['llm_dev_002', 'llm_dev_042', 'llm_dev_048', 'llm_dev_076'], 'streaming_only': [], 'neither': ['llm_dev_020']}['streaming_only']; neither ['llm_dev_020']

## Honesty notes

- Single run, one seed: development evidence only, not a capability claim.
- Dev held rows are tunable; the sealed pool was never loaded (dev-only ids asserted in code).
- Costs: proposer CPU is harness wall, tokens worker-reported, every judge phase ledgered per arm; exclusions recorded in the summary ledgers.


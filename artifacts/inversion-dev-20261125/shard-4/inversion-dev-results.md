# RSI Inversion dev-diagnostic results

Protocol sha256: `608c63da9e0e55911e3c19a3e5ac8ad39f5f7f94079df206526672694f186e96`
Card: `bce2d353-1794-4466-b747-fee7d0a1a04`, bank: `llm_bank_v1`
Model: `spark-1.3` (Spark 1.3 supervised background workers)
Tasks: 5 dev (5 per family over A6/A7/A8/S1/S2/S3)
Held opened identically: True

## Verdict: KILL

- deep reach: a holds ['llm_dev_018', 'llm_dev_031', 'llm_dev_062']
- cost ratio undefined: no shared solves or zero streaming cost
- counts a/b/c=5/5/5

## Final accepts per arm/family

| family | a | b | c | d | e |
| --- | --- | --- | --- | --- | --- |
| A6 | 1 | 1 | 1 | 0 | 0 |
| A7 | 1 | 1 | 1 | 0 | 0 |
| A8 | 1 | 1 | 1 | 0 | 0 |
| S1 | 1 | 1 | 1 | 0 | 0 |
| S2 | 1 | 1 | 1 | 0 | 0 |

## Cost per accepted (CPU seconds)

| arm | total CPU | accepts | per accepted | marginal |
| --- | --- | --- | --- | --- |
| a | 2.751 | 5 | 0.550 | n/a |
| b | 2.763 | 5 | 0.553 | n/a |
| c | 2.775 | 5 | 0.555 | n/a |
| d | 1.094 | 0 | n/a | n/a |
| e | 2.533 | 0 | n/a | n/a |

## CURLEE check-pass rate (final stage)

1.0

## Repair conversion (b vs a)

converted 0 of 0 round-0 rejections (rate None)

## Retrieval delta (c vs b, identical tasks)

delta 0; c-only []; b-only []

## Rejection taxonomy (final)

- a: no rejections
- b: no rejections
- c: no rejections
- d: {'python_mismatch': 2}
- e: {'python_mismatch': 5}

## Paired same-task discordance

best arm a vs streaming: both []; llm-only ['llm_dev_009', 'llm_dev_018', 'llm_dev_031', 'llm_dev_047', 'llm_dev_062']; streaming-only {'best_arm': 'a', 'both': [], 'llm_only': ['llm_dev_009', 'llm_dev_018', 'llm_dev_031', 'llm_dev_047', 'llm_dev_062'], 'streaming_only': [], 'neither': []}['streaming_only']; neither []

## Honesty notes

- Single run, one seed: development evidence only, not a capability claim.
- Dev held rows are tunable; the sealed pool was never loaded (dev-only ids asserted in code).
- Costs: proposer CPU is harness wall, tokens worker-reported, every judge phase ledgered per arm; exclusions recorded in the summary ledgers.


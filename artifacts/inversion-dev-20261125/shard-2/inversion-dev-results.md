# RSI Inversion dev-diagnostic results

Protocol sha256: `d97655a871c494757bb312a430809901a98f5413ac43889bba0976589ace1520`
Card: `bce2d353-1794-4466-b747-fee7d0a1a04`, bank: `llm_bank_v1`
Model: `spark-1.3` (Spark 1.3 supervised background workers)
Tasks: 5 dev (5 per family over A6/A7/A8/S1/S2/S3)
Held opened identically: True

## Verdict: KILL

- deep reach: c holds ['llm_dev_016', 'llm_dev_043', 'llm_dev_088']
- cost ratio undefined: no shared solves or zero streaming cost
- counts a/b/c=1/5/5

## Final accepts per arm/family

| family | a | b | c | d | e |
| --- | --- | --- | --- | --- | --- |
| A6 | 0 | 1 | 1 | 0 | 0 |
| A7 | 0 | 1 | 1 | 0 | 0 |
| A8 | 0 | 1 | 1 | 0 | 0 |
| S1 | 1 | 1 | 1 | 0 | 0 |
| S3 | 0 | 1 | 1 | 0 | 0 |

## Cost per accepted (CPU seconds)

| arm | total CPU | accepts | per accepted | marginal |
| --- | --- | --- | --- | --- |
| a | 2.621 | 1 | 2.621 | n/a |
| b | 19.332 | 5 | 3.866 | 4.178 |
| c | 15.518 | 5 | 3.104 | n/a |
| d | 1.416 | 0 | n/a | n/a |
| e | 2.476 | 0 | n/a | n/a |

## CURLEE check-pass rate (final stage)

1.0

## Repair conversion (b vs a)

converted 4 of 4 round-0 rejections (rate 1.0)

## Retrieval delta (c vs b, identical tasks)

delta 0; c-only []; b-only []

## Rejection taxonomy (final)

- a: {'python_mismatch': 4}
- b: no rejections
- c: no rejections
- d: {'python_mismatch': 2}
- e: {'python_mismatch': 5}

## Paired same-task discordance

best arm c vs streaming: both []; llm-only ['llm_dev_011', 'llm_dev_016', 'llm_dev_043', 'llm_dev_050', 'llm_dev_088']; streaming-only {'best_arm': 'c', 'both': [], 'llm_only': ['llm_dev_011', 'llm_dev_016', 'llm_dev_043', 'llm_dev_050', 'llm_dev_088'], 'streaming_only': [], 'neither': []}['streaming_only']; neither []

## Honesty notes

- Single run, one seed: development evidence only, not a capability claim.
- Dev held rows are tunable; the sealed pool was never loaded (dev-only ids asserted in code).
- Costs: proposer CPU is harness wall, tokens worker-reported, every judge phase ledgered per arm; exclusions recorded in the summary ledgers.


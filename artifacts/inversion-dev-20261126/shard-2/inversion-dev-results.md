# RSI Inversion dev-diagnostic results

Protocol sha256: `d97655a871c494757bb312a430809901a98f5413ac43889bba0976589ace1520`
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
| a | 1.139 | 0 | n/a | n/a |
| b | 5.554 | 0 | n/a | n/a |
| c | 5.484 | 0 | n/a | n/a |
| d | 0.873 | 0 | n/a | n/a |
| e | 0.413 | 0 | n/a | n/a |

## CURLEE check-pass rate (final stage)

0.0

## Repair conversion (b vs a)

converted 0 of 5 round-0 rejections (rate 0.0)

## Retrieval delta (c vs b, identical tasks)

delta 0; c-only []; b-only []

## Rejection taxonomy (final)

- a: {'python_mismatch': 4, 'timeout': 1}
- b: {'python_mismatch': 4, 'timeout': 1}
- c: {'python_mismatch': 4, 'timeout': 1}
- d: {'python_mismatch': 2}
- e: {'python_mismatch': 5}

## Paired same-task discordance

best arm a vs streaming: both []; llm-only []; streaming-only {'best_arm': 'a', 'both': [], 'llm_only': [], 'streaming_only': [], 'neither': ['llm_dev_011', 'llm_dev_016', 'llm_dev_043', 'llm_dev_050', 'llm_dev_088']}['streaming_only']; neither ['llm_dev_011', 'llm_dev_016', 'llm_dev_043', 'llm_dev_050', 'llm_dev_088']

## Honesty notes

- Single run, one seed: development evidence only, not a capability claim.
- Dev held rows are tunable; the sealed pool was never loaded (dev-only ids asserted in code).
- Costs: proposer CPU is harness wall, tokens worker-reported, every judge phase ledgered per arm; exclusions recorded in the summary ledgers.


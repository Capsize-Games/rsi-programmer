# RSI Inversion dev-diagnostic results

Protocol sha256: `63ec0a41f97a80ec5bc9ff82cbaf1d7d80f8f226fa93c7744369b20d7a3767bc`
Card: `bce2d353-1794-4466-b747-fee7d0a1a04`, bank: `llm_bank_v1`
Model: `spark-1.3` (Spark 1.3 supervised background workers)
Tasks: 5 dev (5 per family over A6/A7/A8/S1/S2/S3)
Held opened identically: True

## Verdict: KILL

- deep reach: b holds ['llm_dev_065', 'llm_dev_068', 'llm_dev_072', 'llm_dev_075', 'llm_dev_081']
- cost ratio undefined: no shared solves or zero streaming cost
- counts a/b/c=5/5/5

## Final accepts per arm/family

| family | a | b | c | d | e |
| --- | --- | --- | --- | --- | --- |
| S2 | 3 | 3 | 3 | 0 | 0 |
| S3 | 2 | 2 | 2 | 0 | 0 |

## Cost per accepted (CPU seconds)

| arm | total CPU | accepts | per accepted | marginal |
| --- | --- | --- | --- | --- |
| a | 3.154 | 5 | 0.631 | n/a |
| b | 2.904 | 5 | 0.581 | n/a |
| c | 3.829 | 5 | 0.766 | n/a |
| d | 0.000 | 0 | n/a | n/a |
| e | 0.253 | 0 | n/a | n/a |

## CURLEE check-pass rate (final stage)

0.25

## Repair conversion (b vs a)

converted 0 of 5 round-0 rejections (rate 0.0)

## Retrieval delta (c vs b, identical tasks)

delta 0; c-only []; b-only []

## Rejection taxonomy (final)

- a: no rejections
- b: no rejections
- c: no rejections
- d: no rejections
- e: {'python_mismatch': 5}

## Paired same-task discordance

best arm b vs streaming: both []; llm-only ['llm_dev_065', 'llm_dev_068', 'llm_dev_072', 'llm_dev_075', 'llm_dev_081']; streaming-only {'best_arm': 'b', 'both': [], 'llm_only': ['llm_dev_065', 'llm_dev_068', 'llm_dev_072', 'llm_dev_075', 'llm_dev_081'], 'streaming_only': [], 'neither': []}['streaming_only']; neither []

## Honesty notes

- Single run, one seed: development evidence only, not a capability claim.
- Dev held rows are tunable; the sealed pool was never loaded (dev-only ids asserted in code).
- Costs: proposer CPU is harness wall, tokens worker-reported, every judge phase ledgered per arm; exclusions recorded in the summary ledgers.


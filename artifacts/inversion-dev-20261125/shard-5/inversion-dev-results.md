# RSI Inversion dev-diagnostic results

Protocol sha256: `550a92d743d43e1c278d686897e90991def32648c5438dc5f013e06a88ca4362`
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
| a | 3.930 | 5 | 0.786 | n/a |
| b | 3.901 | 5 | 0.780 | n/a |
| c | 3.927 | 5 | 0.785 | n/a |
| d | 0.000 | 0 | n/a | n/a |
| e | 3.297 | 0 | n/a | n/a |

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
- d: no rejections
- e: {'python_mismatch': 5}

## Paired same-task discordance

best arm b vs streaming: both []; llm-only ['llm_dev_065', 'llm_dev_068', 'llm_dev_072', 'llm_dev_075', 'llm_dev_081']; streaming-only {'best_arm': 'b', 'both': [], 'llm_only': ['llm_dev_065', 'llm_dev_068', 'llm_dev_072', 'llm_dev_075', 'llm_dev_081'], 'streaming_only': [], 'neither': []}['streaming_only']; neither []

## Honesty notes

- Single run, one seed: development evidence only, not a capability claim.
- Dev held rows are tunable; the sealed pool was never loaded (dev-only ids asserted in code).
- Costs: proposer CPU is harness wall, tokens worker-reported, every judge phase ledgered per arm; exclusions recorded in the summary ledgers.

## Shard-5 runner notes (merge inputs)

- Procedure: no-gate rerun pattern from shard-0/1 reruns (machine stats recorded only, never a launch gate; no pre-waiting or idle polling). Every CURLEE invocation ran through the repo resource guard inside `LlmJudge`.
- Protocol sha256: `550a92d743d43e1c278d686897e90991def32648c5438dc5f013e06a88ca4362` (also in `protocol_sha256.txt`; machine records in `machine_start.json` / `machine_end.json`).
- Shard mapping confirmed from `fixtures/llm_bank_v1/tasks.json` and the prior shard-5 record: `llm_dev_068` (S2), `llm_dev_065` (S2), `llm_dev_072` (S2), `llm_dev_081` (S3), `llm_dev_075` (S3) — matches the assigned 068/065/072/081/075 set.
- Per-arm final solves (task-id deltas in `summary.json`): a/b/c accept all 5 tasks (`llm_dev_065`, `llm_dev_068`, `llm_dev_072`, `llm_dev_075`, `llm_dev_081`); d accepts 0; e accepts 0. CPU seconds: a 3.930, b 3.901, c 3.927, d 0.000, e 3.297.
- Guard refusals: 0 — taxonomy (visible and final) contains no `timeout` entries on any arm; CURLEE check-pass rate 1.0 on the final stage.
- Control-validity caveat: arm-d streaming enumeration has no statement grammar (`not_applicable` on all five list-interface tasks) and reaches only ~2.6% into the size-6 enumeration layer, so d=0 by construction; the cost-ratio prong is undefined (no shared solves) and the frozen bar verdict is KILL.


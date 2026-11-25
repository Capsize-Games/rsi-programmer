# Inversion dev-diagnostic result (frozen bar)

Six-shard dev run 2026-11-25 under the frozen DevProtocol in
[inversion-design.md](inversion-design.md)
(`rsi_programmer/llm_dev_protocol.py`, model `spark-1.3`).
Bank: `fixtures/llm_bank_v1` (90 dev tasks; 30 used here, 5 per
shard). Every accept below is a dual-backend accept: the program
ran the held rows in actual Python AND actual CURLEE via the
judge (`rsi_programmer/llm_judge.py`). Learner-reported scores
prove nothing and are not counted.

## Verdict

**KILL. The frozen bar is not met**: the best LLM arm reaches
deep tasks streaming cannot, and repair/retrieval show strict
deltas, but cost per shared accept exceeds the frozen multiple.

Merged over 30/30 frozen tasks (merge:
`/tmp/merge_dev.py`, per-arm sets reconstructed from
discordance + delta lists and cross-checked against reported
counts; `evaluate_bar` recomputed in the merge):

| Arm | Final accepts | CPU s (total) |
| --- | ---: | ---: |
| a (ANN, no repair) | 19 | 18.0 |
| b (ANN + R=3 repair) | 28 | 39.7 |
| c (b + retrieval) | 28 | 33.7 |
| d (streaming control) | 4 | 6.0 |
| e (untrained) | 0 | 15.7 |

Prongs: (i) deep reach PASS — arm c holds 16 deep-family
tasks streaming scores zero on (039, 042, 043, 062, 065, 068,
072, 073, 075, 076, 081, 083, 016, 018, 031, 088);
(ii) cost FAIL — 5.605 per shared accept vs M=3 over the 4
shared tasks (010, 021, 022, 030); (iii) delta PASS —
b-vs-a strict (b_only 039 among others; counts 19/28/28).
Sanity gate passes (b/c 28 ≫ untrained 0), so this is a valid
learning test. Per the frozen protocol the inversion route
closes on this dev evidence: no Phase 7 sealed test and no
retest without a new epic.

## Per-shard record

| Shard | Tasks | a/b/c/d/e | Guard refusals | Protocol sha |
| --- | --- | ---: | ---: | --- |
| 0 (rerun) | 004/022/030/049/073 | 2/4/4/2/0 | 0 | d8754a64... |
| 1 (rerun) | 002/020/042/048/076 | 2/4/4/0/0 | 0 | d9a61b16... |
| 2 (old, clean) | 011/016/043/050/088 | 1/5/5/0/0 | 0 | d97655a8... |
| 3 (re-rerun) | 010/021/039/045/083 | 4/5/5/2/0 | 0 | c1ee97b7... |
| 4 (old, clean) | 009/018/031/047/062 | 5/5/5/0/0 | 0 | 608c63da... |
| 5 (rerun) | 065/068/072/075/081 | 5/5/5/0/0 | 0 | 550a92d7... |

Protocol content frozen; hashes differ per shard only by task
subset. Shard 0/1 reruns replace the old timeout-confounded
runs. Old clean shards 2/4 kept. Shard summaries live in the
retained isolated worktrees under
`artifacts/inversion-dev-20261125/shard-N/` (summary.json,
inversion-dev-results.md, run script, machine stats).

## Caveats (recorded, not hand-waved)

- Shard-3 first rerun was fully degenerate: 23 rc75 guard
  refusals while the sibling shard-5 job held the CURLEE
  build-slot flock, 0/0/0/0/0 with zero backend evaluations.
  It carries no capability signal and is superseded by the
  identical-protocol re-rerun (0 refusals) merged above. The
  first-run summaries are retained in their worktree.
- Arm d is no longer vacuous at merged level: streaming
  solves 4 shallow tasks (010/A6, 021/A7, 022/A7, 030/A8),
  so prong (ii) is well-defined. The earlier per-shard
  d=0 readings stand but do not affect the merged bar.
- Merge-time cost uses even-spread CPU per task (shard
  summaries carry arm totals only); the ratio is exact under
  that flagged approximation, and the gap (5.6 vs 3.0) is
  far outside approximation noise.
- Shard-0 carries 3 harness budget timeouts (a/b/c 1 each),
  recorded as non-accepts. Arm-e taxonomy is all
  `python_mismatch` (30/30): proposals that never ran.
- The sealed 90-task bank is untouched; no held row entered
  repair/query feedback. All failures, refusals, and
  provenance are retained.

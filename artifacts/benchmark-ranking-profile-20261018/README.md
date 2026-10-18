# Full-layer ranking profile

This profile records the remaining costs around the frozen full-layer search
for issue #81. It uses the 48 evaluation tasks from benchmark v2, seed 0, the
32,768 candidate signature cap, and controls in this order: streaming,
heuristic, trained SNN, MLP, recurrent. The search still stops under its normal
solution and tree-bound rules. Evaluation results were used only to describe
runtime and allocation costs; they were not used to select parameters.

The runner used `benchmark_total_work._make_models_with_cpu` to build and fit
the seed-0 controls with the v2 training recipe and checked each available
expected post-fit weight hash against
`artifacts/benchmark-pre-eval-v2-20261018/training.json`. The heuristic,
trained SNN, MLP, recurrent, and seeded-random hashes matched. The script
profiles five controls; seeded-random is not part of the profile matrix.
Training-group construction and model construction/fit are recorded
separately in `training.json` and `manifest.json`.

Each task/control search ran once under cProfile, then once in a separate
tracemalloc pass. The 240 cProfile files are in `cprofile/`. `raw.jsonl` has
240 cProfile records and 240 memory records, including each search's returned
work counters and the profile's top function records. `profile_summary.json`
aggregates the main cost categories by control. Tracemalloc timings are
instrumented and are not performance measurements; peak values describe
Python-traced allocations, not process RSS.

The category boundaries are descriptive. `own_cpu_s` is the additive
cProfile accounting within each function category. Cumulative time includes
called functions and overlaps across call boundaries, so cumulative category
totals must not be added. Descriptor generation includes `policy.layer_triples`
for ranked searches and `_iter_triples` for streaming. Feature assembly
includes cached operand statistics and feature encoding. Model inference
includes ranking and scoring entry points; its cumulative time includes
backend calls. Sorting and retention are attributed where cProfile exposes
those functions. Work outside these named boundaries remains in total
profiled CPU.

Across 48 tasks, cProfile attributed the following additive own CPU totals;
they are profiler accounting, not uninstrumented runtime comparisons:

| Control | Total s | Features s | Model s | Descriptors s | Sort s | Retain s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Streaming | 4.893 | 0 | 0 | 1.369 | 0 | 0.067 |
| Heuristic | 11.423 | 2.962 | 0.215 | 1.981 | 0.147 | 0.234 |
| Trained SNN | 15.389 | 3.140 | 1.615 | 1.729 | 0.179 | 0.236 |
| MLP | 11.079 | 3.076 | 0.073 | 1.928 | 0.212 | 0.235 |
| Recurrent | 11.132 | 3.076 | 0.112 | 1.613 | 0.214 | 0.236 |

| Control | Median traced peak bytes |
| --- | ---: |
| Streaming | 3,897,848 |
| Heuristic | 28,920,027 |
| Trained SNN | 28,929,847 |
| MLP | 28,900,997 |
| Recurrent | 28,920,372 |

These named categories do not account for all profile CPU. Feature assembly,
model inference, sorting, and retention coexist with search orchestration,
signature construction, graph checks, Python allocation, and other calls.
Peak memory varies by task; `profile_summary.json` includes maximum peaks.

The profile also reports each high-level stage entrypoint once per task.
`pre_eval._rank` includes the policy call, and `FeatureStatistics.encode`
includes its nested operand-stat lookups. These cumulative totals overlap with
their nested functions and with category totals; they are shown as inclusive
stage measurements, not added to the table above.

| Control | Inclusive `_rank` s | Inclusive feature `encode` s |
| --- | ---: | ---: |
| Streaming | 0 | 0 |
| Heuristic | 1.629 | 3.257 |
| Trained SNN | 6.102 | 3.437 |
| MLP | 1.537 | 3.368 |
| Recurrent | 1.982 | 3.367 |

## Provenance and execution

The profiling command was:

```sh
flock -x /tmp/rsi-programmer-builder.lock \
  .venv/bin/python scripts/profile_benchmark_ranking.py \
  --output artifacts/benchmark-ranking-profile-20261018
```

`manifest.json` records the environment, source and input hashes, seed-0 fit
costs, and completed row counts. All active search, feature, policy, model,
training, protocol, corpus, and existing v2 training-reference hashes matched
when checked after the run. One concurrent edit to
`rsi_programmer/bounded_pre_eval.py` changed during profiling; this script did
not import or use that experimental module. The outer exec session reported
143 while detaching from the long-running process; the process continued and
wrote a complete manifest with 240 profile rows, 240 memory rows, and all 240
profile files. The final artifact integrity check is recorded in
`execution.json`.

The following bounded-search files appeared after the source snapshot and are
not included in its hash list: `benchmarks/reference/pre_eval_code_v71.py`,
`fixtures/benchmark_bounded_candidates_v1.json`,
`rsi_programmer/benchmark_bounded.py`, and
`rsi_programmer/benchmark_bounded_summary.py`, and
`scripts/benchmark_bounded_candidates.py`. They were not imported by the
profile runner. The captured hash for `rsi_programmer/bounded_pre_eval.py` is
retained as run provenance; its later hash change is reported in the manifest.

The source hash in the manifest is the script version used for the run. After
profiling, the category attribution was corrected to classify
`policy.layer_triples` as descriptor generation. The saved cProfile data did
not change; profile records and the summary were regenerated from those files
using the corrected category boundary and the two stage entrypoints above.
`execution.json` records both script hashes.

The parent independently rechecked row/profile counts, artifact hashes and
all captured source hashes with `verify_profile.py`; its exit was zero.
`parent-integrity.json` retains that result and the two documented changed
files. No search, fit or profile work ran during this check.

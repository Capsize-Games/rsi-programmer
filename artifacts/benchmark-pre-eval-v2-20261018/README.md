# Frozen pre-evaluation benchmark v2 results

The larger benchmark supports useful learned guidance on these tasks. It
does not meet the predeclared criterion for an SNN advantage. At 8,192
candidate signatures, trained SNN accepts 68.75%, streaming 41.67%, MLP
70.83%, and the nonspiking recurrent control 74.31%. At 32,768, all three
learned controls accept every task-seed combination.

The inputs were published before fitting or measurement in `b33ac13`
(`code-v64`). The measured runner is `b03ed18` (`code-v65`). The protocol and
corpus hashes are recorded in `manifest.json`; all frozen v1 files remain
unchanged. The v2 corpus has 24 training and 48 evaluation functions with no
exact function overlap, and zero overlap with v1. All 72 functions and all
1,728 input coordinate pairs are distinct. Shared operators, subexpressions
and algebraic motifs remain possible.

## Accepted results

Each cell counts accepted attempts out of 144: 48 fixed evaluation tasks
times three seeds. Deterministic controls repeat the same 48 outcomes across
seeds. These are not 144 independent tasks. The descriptive uncertainty
calculation averages seeds within each task before resampling 48 tasks,
stratified across the three families.

| Control | 512 | 2,048 | 8,192 | 32,768 |
| --- | ---: | ---: | ---: | ---: |
| Streaming | 0 | 27 | 60 | 129 |
| Heuristic | 0 | 33 | 90 | 138 |
| Seeded random | 0 | 23 | 67 | 134 |
| Untrained SNN | 0 | 26 | 65 | 126 |
| Trained SNN | 0 | 31 | 99 | 144 |
| MLP | 0 | 30 | 102 | 144 |
| Recurrent | 0 | 24 | 107 | 144 |

The frozen SNN rule fails at both primary budgets. Its margin against the
best other control is -1.39 percentage points at 2,048 and -5.56 points at
8,192, rather than the required +10 points. It exceeds the best other
control in zero family means at each primary budget; ties do not qualify.

At 8,192, descriptive paired-task 95% bootstrap intervals for trained SNN
minus each comparison are:

| Comparison | Difference, percentage points | Interval |
| --- | ---: | ---: |
| Streaming | +27.08 | +14.58 to +39.58 |
| Heuristic | +6.25 | -5.56 to +17.36 |
| Seeded random | +22.22 | +11.81 to +33.33 |
| Untrained SNN | +23.61 | +13.89 to +33.33 |
| MLP | -2.08 | -6.94 to +2.78 |
| Recurrent | -5.56 | -12.50 to +0.69 |

The rule and intervals describe this fixed corpus and recipe. They do not
establish unrestricted transfer, a consistent ANN superiority claim, or an
energy advantage. The earlier small-sample claim that a trained ranker is
worse than random on transfer does not describe this v2 result: at 8,192,
trained SNN exceeds seeded random by 22.22 percentage points.

## Work and matched success

At the same 8,192 cap, totals across the 144 task-seed attempts are:

| Control | Accepted | Root signatures | Descriptors | Search seconds |
| --- | ---: | ---: | ---: | ---: |
| Streaming | 60 | 839,031 | 839,115 | 2.219 |
| Heuristic | 90 | 844,401 | 4,461,426 | 27.502 |
| Seeded random | 67 | 870,527 | 4,461,426 | 38.815 |
| Untrained SNN | 65 | 863,364 | 4,461,426 | 37.368 |
| Trained SNN | 99 | 811,988 | 4,461,426 | 38.218 |
| MLP | 102 | 814,048 | 4,461,426 | 24.937 |
| Recurrent | 107 | 803,485 | 4,461,426 | 26.860 |

Ranked controls build and score complete reached layers, even at partial
execution budgets. Their descriptor and feature cost remains substantial.
Streaming can generate one unused lookahead descriptor per partial attempt.
The signature cap measures candidate execution, not total compute or wall
time; the table also has different success coverage between controls.

At matched full success, trained SNN, MLP and recurrent all first reach 100%
at the listed budget 32,768. Their actual root totals are respectively
976,411, 1,026,958 and 933,713; their search times are 37.370, 26.961 and
27.641 seconds. Streaming has not reached 100% at the largest frozen budget,
so there is no observed streaming cost at matched full success. All matched
targets, per-family results and per-seed results are in `aggregate.json`.

Training setup built all 24 groups successfully in 2.623 seconds. There were
721,314 descriptors in their full solving layers, 100 positive and 721,214
negative labels, and 6,244 selected feature rows. Each learned control and
seed used exactly forty optimizer updates and the same groups. Per-seed SNN
fit time was 2.879–2.935 seconds; MLP 0.187–0.192; recurrent 0.978–0.991.
Training and independent post-search verification costs are separate from
the search seconds above. Wall time is descriptive for this recorded CPU
environment and run order, not a hardware-independent efficiency measure.

## Integrity and scope

The run completed all 4,032 cells in 480.265 seconds including its validation
wrapper. All 1,743 returned graphs pass observed rows, held rows and exact
polynomial equivalence; the remaining 2,289 cells are retained failures.
The runner reports no infrastructure errors, a complete unique grid, frozen
policy weights/state, and unchanged before/after source manifests. A second
validation wrapper also confirms unchanged runtime, test, script, fixture
and complete imported SpikeForge Python bytes with no source exclusions.

The independent companion auditor reconstructs training groups and sampling,
checks every returned DAG and its expanded tree bound, checks the full raw
grid and work invariants, compares 341 source files, and recomputes all
summaries and bootstrap intervals. Its evidence lives in
`../benchmark-pre-eval-v2-audit-20261018/`. Its first attempt found errors in
the auditor's accounting and schema assumptions; that failed report, log and
execution metadata are preserved. The benchmark and raw data were not
changed to repair the auditor.

Affected verification: nine focused checks pass; `just test-fast` passes 311
tests in 221.000 seconds. Slow/retry and full local discovery were not run
for this additive benchmark change. Full CI status is tracked separately.

This benchmark distinguishes the controls before saturation and retains the
v1 negative result. Generic learned ranking merits further cost-controlled
experiments. This result does not support promoting SNN-specific architectural
expansion on a claimed SNN advantage. Keep classical and nonspiking controls
in that work, and address full-layer ranking costs before making a search
efficiency claim. Any changed recipe or corpus needs a separately frozen
experiment; do not replace this run with a selected favorable subset.

Reproduce from the recorded runner with:

```sh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  scripts/benchmark_pre_eval_v2.py --output /tmp/rsi-benchmark-v2-new-run
```

Use a new destination. The runner refuses a pre-existing directory or changed
frozen input bytes. Raw rows, training details, summary and provenance are
all committed beside this file.

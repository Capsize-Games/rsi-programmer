# Pre-evaluation benchmark v2

Version 2 compares seven fixed search controls on a frozen corpus of 24
training tasks and 48 evaluation tasks. The three structural families are
balanced, repeated-subtree, and skewed expressions. Each task has 12 observed
rows and 12 held-out rows. The corpus generator enumerates integer polynomial
expressions without calling the solver; corpus validation checks exact
functions and split integrity before a run.

The runner verifies the protocol, corpus, and generator hashes before training.
It fits the three learned controls on training tasks only, with three fixed
seeds, 40 epochs, and a learning rate of 0.05. For each training task it
enumerates candidates through the first layer that solves the observed rows,
labels that complete layer, then selects at most 16 positive and 256 negative
examples using the frozen hash recipe. Feature ordinals and denominators refer
to the complete generated layer. The training report records group sizes,
class counts, selected ordinals and hashes, selected vectors, operation counts,
setup time, and any unsolved task.

The evaluation matrix uses candidate scan budgets of 512, 2,048, 8,192, and
32,768. Search receives only the 12 observed input pairs and targets, a common
tree-operation bound of five, one attempt, and no oracle. Held-out answers and
the witness function are used only after search. Acceptance requires a
structurally valid returned graph, exact execution on observed and held-out
rows, and exact equality between the returned DAG polynomial and the frozen
witness polynomial.

Raw JSONL rows retain the search result, graph, validation work, state and
weight fingerprints, errors, and elapsed time for every attempted cell.
Training and aggregate reports accompany a manifest with source hashes,
environment details, matrix completeness, and infrastructure errors. The
predeclared rule compares trained SNN success with the best other control at
2,048 and 8,192 candidates, requiring a margin of 0.10 at both budgets and a
higher success fraction in at least two of the three families. Matched-success
budgets are reported at 0.25, 0.50, 0.75, and 1.00. Paired task bootstrap
intervals average across training seeds and resample tasks within family; they
are descriptive and do not alter the decision rule.

Run with `python scripts/benchmark_pre_eval_v2.py --output PATH`. The output
directory must not already exist. A nonzero exit indicates a failed or
incomplete run; inspect `manifest.json`, `raw.jsonl`, and `training.json` for
the recorded evidence. Runtime and operation counts are reported directly;
the benchmark does not estimate CPU energy.

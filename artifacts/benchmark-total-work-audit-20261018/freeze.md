# Total ranking work: frozen comparison

Issue: [#80](https://github.com/Capsize-Games/rsi-programmer/issues/80).

This checkpoint publishes the comparison before any new timing run. Protocol
SHA256: `b5399f0ab7f90df5191a2b72c1dd3a17706bb4d64538e616e44512cbe76a1130`.
The original pre-evaluation source is preserved verbatim from `code-v65` in
`benchmarks/reference/pre_eval_code_v65.py`, SHA256
`e5af5fe565f24a3bb08d97c3bf2e7545fbd2a34e83ee99786c2cf95a0b0c49d4`.

The experiment reuses the frozen v2 corpus, recipe and published post-fit
weight hashes. It measures all 48 evaluation tasks, three seeds, four root
signature caps and seven policy controls with both engines and two timing
repeats: 16,128 rows. Engine order is counterbalanced within each cell using
the protocol's hash recipe; no task or recipe is selected by new outcomes.
The v2 dataset is already known. This is an implementation-cost experiment,
not a new independent generalization test.

The planned fix caches target range and operand hit/range counts, retaining
the exact feature values, ordinal denominators, ranking and execution
semantics. The original source runs unchanged through the same timing and
verification wrapper as the optimized source.

Primary timing uses process CPU; wall time is also recorded. Both search and
the complete attempt, including model hash and independent verification
work, are measured. Report I/O and setup/training are separate, with whole-run
cost also recorded. Work counts retain their own units; no weighted scalar
sum is presented as total compute.

At success thresholds 25%, 50%, 75% and 100%, report the first qualifying
listed budget, attained coverage, failures, censoring, cost and training
amortization scenarios. Root caps are sweep settings and do not equalize
total work. The optimization gate requires exact behavior and execution-work
equivalence plus at least 20% pooled paired attempt-CPU reduction at 8,192
for trained SNN, MLP and recurrent separately. Publish all controls, budgets
and repeat-specific results regardless of that gate.

At this checkpoint, dependency byte hashes and the matrix dimensions were
checked statically. Implementation tests, the fast lane, measurement and
independent result verification remain pending. No new performance result
is claimed. All previous frozen benchmark files and results are preserved.

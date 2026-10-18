# Version 2 freeze checkpoint

Issue: [#79](https://github.com/Capsize-Games/rsi-programmer/issues/79).

This checkpoint publishes the inputs and decision rule before any version-2
model fitting or solver measurement. It preserves all version-1 files.

Frozen SHA256 values:

| Input | SHA256 |
| --- | --- |
| Protocol | `cf83dc077dc7960d4e576d898d842484655ac2c61d0e36128da6cda5ae8e7015` |
| Corpus | `a8fd6add90b34da732da8dc96f2c0fafcabc2f017a17cdef4b31303eec4ba921` |
| Generator script | `30cf448a5762710958df1ca96d7631b442e32638dd9236f63e297018d3076b31` |
| Generator module | `13c49ca629ce51e766ffdd3615a925a601ede4c9cff9a713710d159698f6526f` |

The corpus has 24 training and 48 evaluation tasks, with eight and sixteen
tasks per syntax family. Exact polynomial expansion establishes 72 distinct
functions. All 1,728 input coordinate pairs are distinct. Each task has twelve
observed and twelve held rows. Witnesses contain four or five tree operations;
all tasks receive the same search bound of five. An independent enumeration
of all 2,318 trees with at most three operations checks the algebraic floor.

Selection uses syntax, canonical integer polynomials, fixed hashes and quotas.
It does not use solver outcomes. Whole-function disjointness permits shared
operators and subexpressions. The family definitions, sampling, difficulty,
rows and budgets differ from version 1, so a difference in results cannot be
attributed to a single one of those changes.

The fixed matrix contains 4,032 attempts: three seeds, 48 tasks, four budgets
(512, 2,048, 8,192, 32,768), and seven controls. Training uses forty epochs,
Adam at 0.05, and deterministic per-task caps of sixteen positive and 256
negative examples, retaining original layer feature denominators. The rule
requires a trained-SNN success margin of at least 0.10 over the best other
control at both 2,048 and 8,192 signatures, and higher success in at least two
families at each budget. Paired task bootstrap intervals are descriptive.

Verification at freeze:

- `flock /tmp/rsi-programmer-builder.lock .venv/bin/python -m unittest
  tests.test_benchmark_v2_corpus -v`: two tests passed in 0.006 seconds.
- Deterministic regeneration produced byte-identical corpus bytes; command,
  exit status, duration and hashes are in `generator-reproduction.json`.
- `audit.py` independently reconstructs the candidate pools, hash selection,
  coordinates, exact functions, rows and algebraic floor; see
  `input-audit.json` for its result.

The runner, its focused tests, the updated fast lane, model fitting and the
4,032-attempt measurement remain pending at this checkpoint. No performance
or SNN-advantage conclusion is available from this freeze.

# Seed 0 progress checkpoint

Seed 0 completed all 5,376 fixed cells: 48 tasks, four caps, seven controls,
two engines and two timing repeats. The checkpoint includes every seed-0
raw row in `seed0.raw.jsonl.gz`, plus its uncompressed/compressed hashes and
training/final-state metadata in `checkpoint.json`.

Runtime attestations report no search errors, pair mismatches or policy
mutations. The expected seed-0 grid and published model references match;
post-seed model states remain frozen. Executable sources match `code-v69`.
The other two seeds and the independent result audit are pending. This is
not a final performance result or an independently audited optimization
claim.

This data-only checkpoint is published during the active run. It changes
the Git revision recorded at the end of the run without changing runtime
sources. The final before/after source manifests must still match.

Decompress with `gzip -dc seed0.raw.jsonl.gz`. The final bundle will contain
all three seeds and can be checked against this seed-0 prefix hash.

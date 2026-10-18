# Startup amendment

The initial code-v73 CLI launch stopped before group construction, fitting,
search or memory measurement. The wrapper returned 2; `measurement.json`
and `measurement.log` retain the command and exception, with unchanged
source hashes. The failure manifest is retained in
`../benchmark-bounded-v1-20261018/failure.json`. It is not a measured result.

`build_training_groups` requires a training-task list and keyword-only caps;
the runner incorrectly passed the entire corpus and protocol positionally.
The amended call uses only `corpus["splits"]["train"]`, frozen positive and
negative caps, and the frozen sampling seed. Protocol bytes are unchanged.

The new real-startup regression runs all 24 training groups, fits seed-zero
models, checks their exact historical fitted weight hashes and writes the
training checkpoint. Only the first memory call is mocked to stop before
any evaluation search. It checks that no raw/memory result file exists.
This tests the startup integration missed by the original preflight tests.

All 51 focused tests pass in 7.713 s with the amendment. The amended fast
suite also passes all 350 tests; exact duration and output are retained in
`startup-fix-fast.log` and its JSON receipt. Both source snapshots match.
The independent static auditor and four corrupted-bundle controls are ready;
positive result auditing remains pending a complete benchmark bundle.

The corrected run uses a new output directory:
`artifacts/benchmark-bounded-v1-verified-20261018`. The original failed output
and observer evidence remain unchanged. No local slow/retry/full suite was
run, and no full-suite success is claimed. The new API remains experimental.

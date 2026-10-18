# Runner validation checkpoint

The frozen inputs were pushed in `code-v64` before any v2 training or solver
measurement. This checkpoint adds the v2 runner, exact post-search function
verification, deterministic capped training groups, descriptive paired-task
bootstrap, and fast-lane coverage.

- Focused checks: nine tests passed in 0.012 seconds.
- `just test-fast`: 311 tests passed in 221.000 seconds; wrapper wall time
  224.440 seconds. See `fast.log` for the unittest duration and full output.
- Both runs recorded identical before/after hashes for all runtime and test
  Python, benchmark scripts, JSON fixtures, requirements, the justfile, and
  the complete imported SpikeForge Python tree. No source exclusions.

No existing runtime source or v1 benchmark source/input/result was edited.
The slow and retry lanes were not run locally for this additive benchmark
change; the full local suite was not run. Full CI remains a separate check.

The 4,032-attempt experiment and independent result audit are pending. This
checkpoint establishes runner validation, not a performance result.

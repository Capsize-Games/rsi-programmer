# Cache and total-work runner verification

The comparison inputs were pushed before timing in `c56a5ee` (`code-v67`).
This checkpoint implements cached operand features and the paired CPU/wall
measurement runner without modifying the frozen protocol or original engine.

- Focused affected checks: 28 tests pass in 0.053 seconds.
- `just test-fast`: 327 tests pass in 221.637 seconds, 225.132 seconds
  including the external validation wrapper.
- Both checks recorded unchanged hashes for all runtime/test Python,
  benchmark scripts, JSON fixtures, reference Python, historical model
  reference, requirements, justfile and full imported SpikeForge Python.

The new tests compare exact features, gap, graphs and deterministic work
against the preserved original engine; check streaming/zero-cap/terminal
shortcuts; verify actual cache scans; verify timer boundaries, independent
acceptance, source provenance and baseline drift rejection; and check
matched-success censoring and full-setup/mean-seed training amortization.

The slow/retry lanes were not run locally. The changed pre-evaluation API is
used by the benchmark scripts; the older search/author/lattice paths remain
unchanged. Full local discovery was not run. Full CI remains a separate check.

No new performance comparison has run at this checkpoint. The 16,128-row
experiment and independent result audit are pending. Test success does not
establish the frozen CPU improvement threshold.

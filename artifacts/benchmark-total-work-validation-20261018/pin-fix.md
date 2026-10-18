# Protocol pin correction

The first launch from `code-v68` stopped before training or search because
the runner's SHA256 literal omitted one character from the frozen protocol
hash. The protocol and other frozen bytes were unchanged. The integrity
guard refused the run; its failed output manifest is retained in
`../benchmark-total-work-v1-20261018/` and the external execution ledger is
`first-launch-failure.json` with its log alongside it.

The literal is corrected. A new regression test executes all actual frozen
input checks through the start of group generation, substituting only the
group builder so that no training or performance sample is taken.

- 29 affected focused tests passed in 0.100 seconds.
- `just test-fast` passed 328 tests in 222.008 seconds, 225.461 seconds
  including its wrapper.
- Source/dependency/reference hashes were unchanged within both runs.
- `git diff --check` passed. A trailing blank line in the new feature helper
  was removed; its behavior is unchanged.

These results supersede the earlier 327-test checkpoint for the measured
runner. No protocol, data, model recipe or performance rule was adjusted.
The complete comparison will use a new output directory so the failed launch
is preserved. Slow/retry and full discovery were not run locally.

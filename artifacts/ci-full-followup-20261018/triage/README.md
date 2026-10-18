# CI full-suite failure triage

## Evidence reviewed

- Failed workflow run: `39142352272`, source commit `785e8b0` (code-v77).
- Full job: `full (3.13)`, job `117719172635`, completed failure.
- Full-job log is retained as
  `run39142352272-job117719172635.log.gz` (compressed SHA256
  `e91872554c309c67bb9062f0e27156de62af3cfb24413819732677e027961dfb`).
  Decompression reproduces the 77,613-byte source log exactly (SHA256
  `829f8a9f51466c1dc3916175be0b3b67021ec45d02c43c202c202e1fb449c575`).
- Fast `3.13` job: `fast (3.13)`, job `117719172997`, completed failure.
  The archived log copied from the diagnostic handoff is
  `run39142352272-job117719172997.log.gz` (compressed SHA256
  `8f815de11d81a89fa7173779941e72390127597eb9b9aed5e59a87b318c2fd92`;
  decompressed SHA256
  `f9eae9bbd714602845acf1363fa8ae8a9ce335094de9e5804c1c93decd552a71`).
  The GitHub CLI returned an empty log for this job, so this artifact preserves
  the previously captured failure output rather than claiming a fresh download.
- The initial triage was read-only and did not run tests or benchmarks. After
  identifying the auth failure mechanism, a workflow-only fix was made in
  `.github/workflows/ci.yml`; no runtime or test source was changed.

## Results

The fast-job startup hash failure is fixed by code-v78 (`4ea5491`). The old
`test_real_training_startup_stops_before_any_evaluation` compared a fresh
fit's neural weight hashes to cross-environment golden hashes. The v78 test
spy first asserts the runner supplied the pinned map, then relaxes only the
test invocation's neural expected hashes to `None`. It still runs training
and asserts 40 optimizer updates and changed weights. The production hash
guard in `benchmark_total_work.py` remains strict whenever an expected hash
is supplied. Code-v82 adds the shortlist startup test with the same
test-only portability treatment. The v77 fast-job failure is therefore
covered by v78; the v77 full-job failure below is not.

The full job completed 417 tests in 8383.351 seconds and failed only
`tests.test_hub_verify.HubVerifyTests.test_missing_hub_listing_recorded`.
The assertion at `tests/test_hub_verify.py:41` expected
`res["github"] == "PASS"`; the actual value was `FAIL`. The test output does
not include `res["detail"]`, so the exact failing GitHub operation cannot be
recovered from this run.

The source shows that `verify_model()` sets GitHub to PASS only after release
asset lookup/download and byte-hash equality (`scripts/verify_hub.py:118-133`).
Any failure in those steps returns early before querying the model hub. The
failure is therefore in the GitHub release verification path, not the missing
hub-listing check. The “use a new output directory” JSON line earlier in the
log is from an expected rejection test and is not a test failure.

The workflow at `.github/workflows/ci.yml` runs full unittest discovery on
main pushes, including `test_hub_verify`; the failed run step did not map
`GITHUB_TOKEN` or `GH_TOKEN` into the shell environment. `verify_hub.py:38-41`
uses `gh api`, then `gh release download` at lines 63-69. GitHub CLI's local
`gh help environment` documents `GH_TOKEN`/`GITHUB_TOKEN` as its API auth
environment. An isolated, read-only reproduction with both variables unset
returned exit 4 and this message: `To get started with GitHub CLI, please run:
gh auth login. Alternatively, populate the GH_TOKEN environment variable
with a GitHub API authentication token.` This establishes the missing-auth
mechanism; the CI log omits `res["detail"]`, so it does not identify whether
the early return was specifically release-list auth, download auth, or a
separate asset failure.

The isolated command, exit status, stdout, and stderr are retained as
`gh-unauth.command`, `gh-unauth.exit`, `gh-unauth.stdout`, and
`gh-unauth.stderr` in this directory. No credential was used or written.

The workflow now gives only the full-discovery step `GH_TOKEN:
${{ github.token }}` and restricts the full job token to `contents: read`.
This is the token already provided by the workflow, scoped to the one step
that invokes the GitHub CLI. It does not change `verify_hub.py` or weaken the
live integration test.

## CI order and smallest follow-up

The `fast` matrix runs `just test-fast` on Python 3.12 and 3.13. The full
matrix runs only on pushes to `main` and executes
`.venv/bin/python -m unittest discover -s tests -t .` on both Python versions.
`just test-slow` includes `tests.test_hub_verify`. It does not include
`tests.test_hub_release_lookup`, which is in `just test-fast`; CI's full
discovery includes both modules regardless of the local lane recipe.

Smallest affected local command is
`.venv/bin/python -m unittest
tests.test_hub_verify.HubVerifyTests.test_missing_hub_listing_recorded`.
For the external test plus deterministic release-lookup controls, run
`.venv/bin/python -m unittest tests.test_hub_release_lookup
tests.test_hub_verify.HubVerifyTests.test_missing_hub_listing_recorded`.
Do not run local full discovery (the repository documents roughly 3.7 h).

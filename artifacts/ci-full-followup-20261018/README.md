# Full CI follow-up

The completed code-v77 Python 3.13 full job ran 417 tests in 8,383.351 seconds
and failed only `test_missing_hub_listing_recorded`: GitHub asset verification
returned FAIL where the integration test requires PASS. The verifier invokes
`gh api` and `gh release download`; the full test step did not supply a token.
The CI log omits the verifier detail, so the precise subprocess error there
is inferred from the missing environment and the reproductions below.

The workflow-only fix restricts the full job to `contents: read` and supplies
`${{ github.token }}` as `GH_TOKEN` only to the unittest step. This follows
[GitHub's documented CLI workflow setup](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-github-cli).
No verifier, test, learning, search or benchmark implementation changed.

`probe_gh_auth.py` runs the actual failing test with a fresh empty GH config,
all token variables and GH_HOST/GH_REPO removed, and GITHUB_ACTIONS=true. It
fails exactly as in CI. Supplying an existing local credential only as the
child's GH_TOKEN then passes all six hub/release-lookup tests, including real
release download/checksum verification and corrupted-weight rejection.
No credential value is stored. This local smoke does not mint or directly
test the GitHub Actions token; corrected CI remains the final integration gate.

`auth-probe.json`, both child logs, and `auth-reproduction-corrected.*` retain
the exact commands, outcomes, durations and unchanged source snapshots,
including workflow YAML. The outer probe exits zero only if the expected
negative fails and all six authenticated tests pass. An initial probe left
GH_HOST set, allowing a local OS-keyring fallback; its script, output and
failed observer receipt are retained rather than counted as a reproduction.

Earlier fast-job neural-byte portability errors were fixed in code-v78.
Current runtime/test sources already passed all 374 fast tests on Python
3.12 and 3.13 at code-v82. Those results do not establish a corrected full
CI pass. No local full/slow/retry lane was run for this workflow-only change.
Full matrices at code-v78, code-v80 and code-v82 remain in progress in the
retained snapshots and still use the old workflow without GH_TOKEN.

`older/` and `code82/` hold source-specific job metadata, safe unavailable-log
responses and source comparisons. `triage/` retains compressed failed-job
logs, exact hashes and the four-line patch rationale. GitHub's full matrix
cancels its sibling after a full-job failure; the observed full cancellation
was not caused by the separate fast-matrix failure. No job was canceled or
rerun by this follow-up.

The corrected workflow was pushed as code-v87, commit 59ef1ab. Its corrected
CI run is 39154146448. Both fast jobs passed all 374 tests: Python 3.12 in
346.461 seconds and Python 3.13 in 238.441 seconds. Compressed logs, exact
job metadata and byte hashes are in `code87/`; its full jobs remain pending
in the retained snapshot.

Local `just test-fast` subsequently passed all 374 tests in 237.352 seconds
(240.766 seconds CLI). `fast-ci-auth.*` and `local-verification.json` prove
that the same 456 source/dependency/workflow files were used before and
after both the authenticated affected checks and the fast suite, with no
source exclusions. `static-check.json` records AST and 79-column checks on
the three artifact scripts. No local full/slow/retry lane was run.

Issue #44 stays open until corrected full jobs have actual verdicts. A
background follow-up checks every fifteen minutes, remains quiet while
unchanged, and continues the authorized fix/test/publish workflow on any
new failure. It stops after full verification, publication and issue updates.
No pending run is a pass.

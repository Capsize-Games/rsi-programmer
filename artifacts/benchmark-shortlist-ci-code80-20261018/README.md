# Code-v80 fast CI evidence

Both fast CI jobs for commit `f61ca3ed33f07c1edcbaf3974be55ab2e95ba4b0`
completed successfully in workflow run `39147846343`:

| Job | Result | Tests | unittest time |
| --- | --- | ---: | ---: |
| fast (3.12), job 117735225196 | success | 367 | 443.901 s |
| fast (3.13), job 117735225454 | success | 367 | 459.066 s |

Each full uncompressed job log ends with `Ran 367 tests ...` and `OK`. Deterministic `gzip -n` logs and JSON API job records are retained beside
this file. Decompressed log bytes match the retrieved originals exactly; the receipt records all byte hashes and the
exact command exits. GitHub's CLI log view did not return logs while the
overall workflow was still running, so the completed-job logs were retrieved
from the Actions job logs API.

The retained run/job-list snapshot showed both full-suite jobs still in
progress when captured. Full CI therefore remains pending in this evidence;
these fast results apply to code-v80 only, not later uncommitted benchmark
runner or protocol changes.

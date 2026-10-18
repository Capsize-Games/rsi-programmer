# Code-v82 full CI status

The latest retained API snapshot at 2026-10-18 18:19:23 UTC is for commit
`6b1559986f3e240219661383a8f3f84bf1fb1557` and workflow run `39149596790`.
Both full jobs were still running, each in the complete unittest discovery
step, after about 52 minutes:

| Job | Python | State | Active command | Elapsed command |
| --- | --- | --- | --- | ---: |
| 117740330754 | 3.13 | in progress | `python -m unittest discover -s tests -t .` | 3,154 s |
| 117740331011 | 3.12 | in progress | `python -m unittest discover -s tests -t .` | 3,155 s |

Setup steps completed successfully. No failing step or check-run annotation
was reported in the available metadata. The workflow has no declared test
count or timeout, so its configuration does not support a defensible ETA.
Full CI remains pending; these observations are not pass results.

The job logs could not be obtained while the jobs were active. GitHub's
supported `gh run view ... --log` command reported that logs become available
when the job completes. The Actions job-log endpoint returned HTTP 404 and
Azure `BlobNotFound` for both jobs; response SHA256 values are retained in the
receipt, while raw XML is omitted to avoid retaining transient request
identifiers. Check-run annotations were available and empty for both jobs.

The workflow declares one full unittest discovery per Python 3.12 and 3.13
matrix job in `.github/workflows/ci.yml`. Fast CI for this same code-v82 commit
passed 374 tests in 464.011 s and 483.614 s; those fast-job measurements do
not predict the full-suite completion time.

See [receipt.json](receipt.json) for job metadata, timestamps, exact queries,
API and error-body hashes, and the full pending-state record. Initial and
latest job snapshots are retained; unchanged intermediate polls were
discarded.

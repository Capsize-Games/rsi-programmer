# Fast CI results for code-v78

Workflow run `39145562287` tested source commit `4ea5491c0a53f960efa5e6c0889c84d3e8b6f423`. Both requested fast matrix jobs passed:

| Job | Result | Test output | Elapsed test time |
| --- | --- | --- | ---: |
| `117728565002` (`fast (3.13)`) | success | 352 tests, `OK` | 455.570 s |
| `117728565064` (`fast (3.12)`) | success | 352 tests, `OK` | 450.121 s |

The exact GitHub API job responses and complete job logs are retained in each `job-<id>/` directory. `run.json` is the workflow response captured while the two full matrix jobs were still running; this artifact makes no claim about their completion. `execution.json` records retrieval commands, test-result evidence, and SHA256 hashes.

Complete logs are stored as deterministic gzip files. Decompressed bytes match
the original API responses exactly, including whitespace. Both compressed and
original-byte SHA256 values are recorded in `execution.json`.

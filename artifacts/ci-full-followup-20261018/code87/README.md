# Code-v87 CI evidence

Workflow run [39154146448](https://github.com/Capsize-Games/rsi-programmer/actions/runs/39154146448) tested commit `59ef1abe06a5a9d0d8e9813252f05c902fdfc729` (code-v87). Evidence was collected through GitHub REST endpoints with `gh api`; no local tests, builds, or runtime changes were made.

At the 2026-10-18 18:33:15 UTC snapshot, both fast jobs had passed and both full-suite jobs remained in progress. Fast jobs:

- Python 3.12, job `117753752743`: `just test-fast` succeeded; log reports 374 tests in 346.461 seconds.
- Python 3.13, job `117753752861`: `just test-fast` succeeded; log reports 374 tests in 238.441 seconds.

Full jobs `117753752836` (3.12) and `117753752870` (3.13) were still running unittest discovery. No full-suite result is claimed here. Job log retrieval was available from `GET /actions/jobs/{job_id}/logs` after each fast job completed, although `gh run view --job --log` remained gated while the overall run was active.

The initial run/job records are `run-initial.json` and `jobs-initial.json`. The status snapshot after both fast jobs completed is `run-fast-complete.json` and `jobs-fast-complete.json`; `job-fast-312.json` and `job-fast-313.json` are the exact job records. Fast logs are stored compressed only. For each compressed file, gzip decompression was byte-compared with the captured REST response before the temporary uncompressed copy was deleted. The original byte count and SHA-256 below refer to the uncompressed log response.

| File | Bytes | SHA-256 |
|---|---:|---|
| `fast-312.log.gz` | 24,583 | `a6a83edd39e1b7563e77714ff6de2575382ad2cfd4d22effc40020d6a382eeb9` |
| `fast-313.log.gz` | 24,516 | `947a2a3de08f2ae212492d80fd7a99da74c6f14a8498d389ba0654e56833e2f1` |

The underlying uncompressed log response sizes and hashes were, respectively, 114,207 bytes / `ce5b56079840280254b128e2b8a0d36086eacb5930b80cb427244dbe9d76e093` and 114,209 bytes / `6bc006804b8e548bf1e3bf194c0bc0e89b7434940231a62049f0d00b41b9e7f0`.

REST snapshots were saved as returned by GitHub:

- `run-initial.json` SHA-256 `94dd3d33c0d68567e763098e6bfcba617bd2017eb1319a7bea33102a1ccb6b54`
- `jobs-initial.json` SHA-256 `b52ba1b7a5e393908f412adea72778bb668fb78918d00314d3ebee3ab6680565`
- `run-fast-complete.json` SHA-256 `94dd3d33c0d68567e763098e6bfcba617bd2017eb1319a7bea33102a1ccb6b54`
- `jobs-fast-complete.json` SHA-256 `fa2888ba31de309d459556d7aead8377f3b57746fa1f79f61c41c4b7126b26f3`

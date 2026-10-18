# Complete-shortlist measurement checkpoint

All 960 isolated memory rows are complete: 480 native RSS and 480
allocation-tracing rows, with zero recorded row errors and 960 unique keys.
The deterministic gzip snapshot decompresses to the exact source bytes;
`memory-phase-receipt.json` records hashes and source/protocol identity.
Instrumented durations do not supply CPU comparison curves.

The 11,808-row timed matrix is still running. This checkpoint makes no
coverage, cost or efficiency decision. The full matrix, final source/model
checks, independent audit and six corruption controls remain pending.
The snapshot will be compared byte-for-byte with the final memory file.

Code-v82 passed all 374 fast tests on both Python 3.12 and 3.13 in CI.
The retained CI snapshots show full CI still running. Local verification
passed 75 affected and 374 fast tests; full/slow/retry were not run locally.

Seed zero subsequently completed all 3,936 scheduled timed rows with zero
recorded row errors and unique timed keys. `seed0-timed-prefix.jsonl.gz`
preserves its exact byte prefix, with hashes in the companion receipt.
Seeds one and two, the final matrix/audit and all cost decisions remain
pending at that checkpoint. The prefix will be compared to final raw data.

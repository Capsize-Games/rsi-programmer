# Online runtime fast CI verification

Both fast jobs passed in run
[39160627937](https://github.com/Capsize-Games/rsi-programmer/actions/runs/39160627937),
at source `5cdb9b5782a37bf0c32bb15d22098399f3ce12df` (`code-v90`).

| Python | Job | Tests | Unittest seconds | Result |
| --- | ---: | ---: | ---: | --- |
| 3.12 | 117772980247 | 409 | 459.645 | OK |
| 3.13 | 117772980160 | 409 | 469.247 | OK |

The exact compressed logs, completed job metadata, run snapshot and hash
receipt are retained here. Independent verification matched the compressed
and decompressed hashes, test counts, durations, and both checkout SHAs.
Each log records the RSI source above and Spikeforge source
`2981b28105a0d9a8ddcb76ae4e21a4358a1d96f6`.

The measured runtime, tests, scripts, fixtures, requirements, justfile and
workflow match the current source; subsequent commits contain evidence and
documentation. The receipt also checks the source-file hashes retained by
the completed online benchmark. No local tests were run for this
evidence-only checkpoint.

Both full jobs remain in progress in unittest discovery. Their results
are pending. #83's fast matrix is verified; its full matrix is pending.
The separate authentication acceptance run `39154146448` at `code-v87`
still has both full jobs pending, so #44 remains open and #82's full-CI
note remains unresolved. No pending or canceled job counts as passing.

The published ranking and persistent-adaptation conclusions are unchanged.
The task-reset SNN's secondary reduction in candidate executions remains a
limited observation with greater host CPU cost, not a chip-energy result.

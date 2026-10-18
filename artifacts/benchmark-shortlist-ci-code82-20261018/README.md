# Code-v82 CI evidence

Workflow run `39149596790` tests commit
`6b1559986f3e240219661383a8f3f84bf1fb1557`. Both fast jobs completed
successfully:

| Job | Result | Tests | unittest time |
| --- | --- | ---: | ---: |
| fast (3.12), job 117740331048 | success | 374 | 464.011 s |
| fast (3.13), job 117740330958 | success | 374 | 483.614 s |

Each retained full log ends with `Ran 374 tests ...` and `OK`. The exact job
JSON and deterministic `gzip -n` logs are stored beside this
file. [receipt.json](receipt.json) records commands, exits, byte hashes, test
counts, and durations.

The final run/job snapshot shows full (3.12) and full (3.13) still in
progress. Full CI is pending in this evidence. These results apply to the
pinned code-v82 commit only. Compressed logs were verified byte-for-byte
against their original raw responses before removing redundant plain copies.

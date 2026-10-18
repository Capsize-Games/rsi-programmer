# Complete-shortlist post-run report

The reporter completed successfully against the audited run directory
`artifacts/benchmark-complete-shortlist-v1-20261018`. It selected each
configuration's first complete, uncensored full-success point and verified
the matching raw rows and work totals against the aggregate summary.

CPU figures below are milliseconds per attempt for the selected point and
amortized per independent task at N=48/480/4800. Work is per attempt:
cheap row operations / root calls / policy-scored features / duplicate full
descriptor evaluations / descriptor replays. Duplicate evaluations count
repeated evaluation of an identical descriptor; distinct expressions with
the same output remain distinct candidates.

| Engine | Control | Cap | Attempt CPU ms | N48 / N480 / N4800 ms | Work per attempt |
| --- | --- | ---: | ---: | ---: | --- |
| streaming | streaming | 131072 | 38.994 | 38.994 / 38.994 / 38.994 | 0 / 13385 / 0 / 0 / 0 |
| cheap_only | cheap_only | 131072 | 44.001 | 1227.184 / 162.320 / 55.833 | 13524 / 13269 / 0 / 0 / 0 |
| full_cached | heuristic | 131072 | 158.358 | 158.358 / 158.358 / 158.358 | 0 / 9783 / 30982 / 0 / 0 |
| complete_shortlist | heuristic | 131072 | 55.704 | 1238.886 / 174.022 / 67.536 | 13524 / 13273 / 879 / 0 / 0 |
| full_cached | trained_snn | 32768 | 231.652 | 350.415 / 243.528 / 232.839 | 0 / 6781 / 30982 / 0 / 0 |
| complete_shortlist | trained_snn | 131072 | 105.211 | 1407.157 / 235.406 / 118.231 | 13524 / 13273 / 879 / 0 / 0 |
| full_cached | mlp | 32768 | 137.853 | 197.807 / 143.848 / 138.452 | 0 / 7132 / 30982 / 0 / 0 |
| complete_shortlist | mlp | 131072 | 57.578 | 1300.714 / 181.892 / 70.009 | 13524 / 13275 / 879 / 0 / 0 |
| full_cached | recurrent | 32768 | 147.697 | 224.232 / 155.351 / 148.463 | 0 / 6484 / 30982 / 0 / 0 |
| complete_shortlist | recurrent | 131072 | 63.451 | 1323.168 / 189.423 / 76.048 | 13524 / 13275 / 879 / 0 / 0 |

The memory measurement grid is complete: 960 RSS/tracemalloc rows at fixed
cap 32768. Search acceptance at that cap was streaming 43/48, cheap_only
43/48, full_cached heuristic 46/48, complete_shortlist heuristic 43/48,
full_cached trained_snn 48/48, complete_shortlist trained_snn 43/48,
full_cached mlp 48/48, complete_shortlist mlp 43/48, full_cached recurrent
48/48, and complete_shortlist recurrent 43/48. These counts describe grid
search outcomes; they do not claim a memory win at matched success. The grid
does not establish a memory curve across root caps.

The corrected reporter command exited 0 in 0.568 seconds. Its exact command,
script hash, input bundle hashes, and output hashes are recorded in
`run_receipt.json`. `report.json` contains the machine-readable table and
memory coverage statement; `report.txt` is the text rendering.

The superseded first report and receipt are retained as `first-report.*` and
`first-run-receipt.json` for traceability; the corrected report uses precise
duplicate-evaluation terminology and separates measurement-grid coverage
from search acceptance.

The report helper is artifact-only and imports no search engine. It does not
change benchmark data, implementation, protocol, or decision gates.

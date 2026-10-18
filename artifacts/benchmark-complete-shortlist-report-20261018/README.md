# Post-run report helper

`report.py` is a standard-library-only formatter for the completed shortlist
benchmark output directory. It reads `aggregate.json`, `training.json`,
`raw.jsonl`, and `memory.jsonl`; it imports no runtime engine code and does
not alter benchmark results or decision gates.

Run it only after the main benchmark completes and its audit approves the
result:

```sh
python artifacts/benchmark-complete-shortlist-report-20261018/report.py \
  artifacts/benchmark-complete-shortlist-v1
```

The report uses the summary's first observed full-success point for each of
the ten configurations, verifies the matching raw cell is complete and
uncensored, and checks summary work totals against raw observations. It shows
mean attempt CPU, N=48/480/4800 amortized CPU, and per-attempt cheap row
operations, root calls, policy-scored features, duplicate-signature
evaluations, and descriptor replays. Its memory statement is limited to the
fixed 32768 cap and requires all 960 RSS/tracemalloc rows.

Status: prepared before benchmark completion; not executed or validated
against a completed result directory.

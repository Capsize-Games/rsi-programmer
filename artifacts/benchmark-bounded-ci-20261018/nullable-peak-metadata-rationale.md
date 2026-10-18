# Nullable peak-counter metadata candidate

In `benchmark_bounded.attempt`, each `EXTRA_COUNTS` value currently uses
`row["search"].get(key, 0)`. The bounded engine reports peak live descriptor
and feature counts, but the cached and streaming search results do not expose
those measurements. The fallback therefore turns “not measured” into a numeric
zero. The same wrapper currently reports streaming `scored_unexecuted_descriptors`
as zero despite not having a scored-descriptor count.

In `benchmark_bounded_summary.summarize`, work totals use
`sum(r["work"].get(k) or 0 ...)`, which collapses both missing and null
observations to zero. The candidate uses `None` for absent counters, computes
streaming scored-unexecuted only when applicable, and sums a counter only when
every row in that point has a value. Thus bounded peak values remain numeric;
uncaptured cached/streaming peaks stay null in row and aggregate metadata.

The added regression exercises the attempt wrapper with missing peak fields,
then verifies that summary totals keep streaming peaks null while summing
bounded peak observations. It also verifies the cached attempt's
scored-unexecuted count remains computed when scored descriptors are present.
No search implementation, timing path, frozen data, protocol, or benchmark
result is changed by this candidate. Existing raw/aggregate artifacts retain
their original recorded zeros; this patch only affects a future run using the
new metadata behavior.

This patch is separate from the portable-startup-test candidate and is
unapplied and untested.

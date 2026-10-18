# Nullable peak-counter metadata candidate

`benchmark_bounded.attempt` currently maps every absent extra counter to zero.
That invents a measured zero for `peak_live_descriptors` and
`peak_live_features` when cached or streaming results do not report these
measurements. The candidate uses `None` only for those two absent peak fields.
Other extras retain zero defaults; in particular streaming reports zero
scored descriptors, zero skipped descriptors, and zero scored-unexecuted
candidates. The cached/full path's scored-unexecuted calculation is unchanged.

`benchmark_bounded_summary.summarize` currently turns missing and null values
into zero while summing. The candidate moves a small helper to module scope and
returns `None` for an axis if any attempt lacks a value; otherwise it sums all
values. This keeps unavailable peak observations null and preserves numeric
zero counters. A bounded peak total is the sum of the per-attempt maxima, not a
maximum across the whole set of attempts; this follows the existing
per-axis work-total aggregation shape.

The regression uses `_attempt`'s result structure (`search`, `work`, and
`root_calls`) with streaming's zero `scored_descriptors`. It checks that only
missing peak fields become null, skipped and scored-unexecuted remain zero,
and summary aggregation preserves null streaming peaks while summing bounded
per-attempt peak values. The patch changes no search implementation, timing
path, frozen data, protocol, or benchmark result. Historical raw and aggregate
artifacts retain their recorded values.

This separate patch is unapplied and untested.

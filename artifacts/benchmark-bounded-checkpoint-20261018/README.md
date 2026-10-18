# Native memory checkpoint

This data-only checkpoint retains every one of the completed 432 native RSS
observations at seed zero/cap 32,768: all 48 tasks, streaming, and both ranked
engines for all four controls. No task is removed. The compressed JSONL is an
exact byte-preserving copy of the first memory pass, with deterministic gzip
metadata. The traced pass and timing matrix are still in progress.

The parent checked the unique native-memory grid, frozen per-attempt model
states/weight references, absence of errors and bounded active-block peaks.
These are checkpoint integrity checks, not the final independent graph,
behavior, cost or complete-benchmark audit. Memory-to-timing comparisons
await the full timed matrix. No performance conclusion is made.

The auditor was strengthened by static review before timed data became
available: it now checks the actual ordered JSONL sequence and adjacent
engine pairs, as well as each row's order label and the complete grid. A
fifth corrupted control swaps two raw rows. Old and new auditor hashes and
the reason are retained in the audit folder's source-amendment receipt.
Runtime, training, protocol, candidate method and search sources are
unchanged; this publication adds retained data and independent checks only.

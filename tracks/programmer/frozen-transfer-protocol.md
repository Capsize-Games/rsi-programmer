# Frozen branch-transfer evaluation

This evaluation freezes 18 new functions before any controller runs. Two
functions come from each of the nine supplied donor groups. Selection uses
grammar enumeration and execution, without neural scores or solve results.
All selected functions differ from the 2,647 teacher functions, the earlier
training recipe and the eleven development functions on the declared 24
probes. This evaluates the existing two-branch grammar, not unrestricted
programming. The original sealed arithmetic set remains unopened.

The [task bank](../../fixtures/frozen_branch_transfer_v1/tasks.json),
[held inputs](../../fixtures/frozen_branch_transfer_v1/held.json) and
[protocol](../../fixtures/frozen_branch_transfer_v1/protocol.json) have
recorded hashes. The worker loads visible examples and reference programs
for its task-engine callback. It never loads the held-input file or passes
reference code to a solver. All three controllers must finish choices
before held evaluation starts. Inputs supplied as feedback do not overlap
earlier held inputs. References are admitted only when they execute on all
declared dataset rows; this is not a universal totality proof.

The fixed controllers are the current neural route, composition-first
classical search and an untrained controller. Each task receives one CPU
second for search and visible-query feedback, with a declared 0.05-second
bounded-step overrun tolerance. Search calls also have wall-clock limits
derived from remaining CPU allowance. Each task starts with three examples
and can add three from a fixed 24-input visible pool. The neural route uses
the existing condition-first order; the classical control uses its stronger
composition-first order. The argument grammar, beam and donor ranking stay
fixed. Sources are sealed before measurements.

Both neural models are fitted afresh using their original seeds and
recipes, and must reproduce their published parameter fingerprints.
Supervision generation, initialization, fitting and fingerprint checking
are timed. Setup, donor qualification, search, visible oracle calls and
actual Python/CURLEE execution are also measured. Process CPU includes
completed children; worker peak RSS is reported separately. All fitting
cost is charged over the actual 18 tasks. A volume-1,000 amortization
scenario is reported only as a hypothetical, with no extra quality credit.

The qualified donor library is a common existing input. Its historical
creation cost is outside this controller comparison; therefore this is
not the complete end-to-end cost gate from an empty library. No cost
advantage may be inferred from search timings alone. A result must report
solve rate, actual checking costs and exclusions alongside any comparison.

Each emitted candidate is checked by isolated Python execution and the
existing guarded CURLEE checker on 20 held rows: six recurring boundary
arrays and fourteen newly seeded random arrays. Incorrect, timed-out and
unconstructible programs remain failures. A process error or malformed
output stops that program's remaining cases; accepted programs require
all rows. All actual execution counts are reported. Fitting or solver
changes after opening results turn this bank into development evidence
and require a new final evaluation set.

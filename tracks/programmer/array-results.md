# Constructing list-producing programs

The first development trial constructs **0/3 programs** after a 6.471-second
fit. The untrained control also constructs none. The three supplied backend
controls pass **20 Python and 20 CURLEE cases each**, independently from the
learner. These controls are not neural successes or library donors.

The tasks are clipping negative readings to zero, discarding nonpositive
readings while capping retained values at two, and extracting magnitudes
of negative readings. They require list output rather than a scalar
summary. The new typed grammar constructs a loop, conditions, integer
expressions, append operations and a list return. It permits two append
nodes, nested conditions and one input loop. Input capacity is 256 and
output capacity 512. CLI plumbing, initialization, lowering and contracts
remain supplied code.

The engine generates 512 distinct teacher functions. All three complete
development functions are excluded on 24 fixed probes; their components
may occur in other functions. The ANN learns individual next-token choices
from 19,966 training states over 30 epochs. It receives input/output lists
and its syntax prefix, without an LLM teacher or a complete target witness.
The initial search keeps 32 prefixes and at most four token choices per
step. Its three-second limit is a cap; each unsuccessful request finishes
in about 0.02–0.03 seconds. A larger budget alone does not explore more
than this fixed beam.

All neural and untrained choices finish before generating held rows. The
[initial negative trial](
../../artifacts/array-statements-v1-20261101/report.json) preserves its
decisions and source snapshot. The first combined backend job stopped at
resource-guard status 75. A direct guarded compiler diagnostic also
rejected the supplied output bound and loop invariant. Partial case records
were not fully persisted by that initial checker, so none are claimed as
passed.

The backend now supplies the structural invariant `outlen <= 2 * i`, using
the grammar's two-write limit. Python execution and guarded CURLEE checks
run separately, and case records are journaled as they finish. The
[complete backend qualification](
../../artifacts/array-statements-v1-backend-v2-20261101/receipt.json) reuses
the unchanged learner choices and held rows, with no retraining or repair.
All 60 Python and 60 CURLEE control cases pass, the deliberately wrong
expected output is rejected, and the compiler hash remains unchanged.
The control result covers the declared tests, not universal integer or
compiler qualification.

## Bounded follow-ups

Execution-context features also construct **0/3**, alongside 0/3 untrained.
The model observes a fixed completion of each syntax prefix executed on
visible inputs, its output list compared with the visible target, and an
execution-validity flag. Completion reads no target. The fit uses the same
512 functions, 19,966 states and 30 epochs, and takes 9.443 seconds.
The [trial](
../../artifacts/array-statements-context-v1-20261101/report.json) preserves
its decisions and exact source snapshot.

A broader search keeps 256 prefixes and eight token choices per step,
instead of 32 and four. Fresh fitting reproduces the execution-context
model's trained and untrained fingerprints. It still constructs **0/3**
for each controller. The fit takes 9.465 seconds; its full worker takes
16.047 seconds wall time. The [broader trial](
../../artifacts/array-statements-broad-search-v1-20261101/report.json)
preserves the negative choices. No target-language learner program runs
because none is emitted. Common controls are not rerun for this unchanged
backend; this is an explicit exclusion, not additional execution evidence.

| Development recipe | Trained constructed | Untrained constructed | Fit wall time |
| --- | ---: | ---: | ---: |
| Input/output features, beam 32 | 0/3 | 0/3 | 6.471 s |
| Execution context, beam 32 | 0/3 | 0/3 | 9.443 s |
| Same parameters, beam 256 | 0/3 | 0/3 | 9.465 s |

The execution-context trial also qualifies a fourth supplied control that
appends each input twice. All four controls pass 20 cases per language,
including 512 outputs from 256 inputs: **80 Python and 80 CURLEE cases**.
This extends backend boundary evidence; it is not a fourth neural task or
a donor. The earlier sixty-case qualification is a separate run.

Direct token construction remains the gap. Partial execution features and
this wider beam are insufficient under these recipes. Reusing separately
qualified neural fragments or changing the structural curriculum are
possible next checks, not established fixes. The previously frozen
transfer bank is not used for repair or tuning, and no supplied control is
promoted into the neural library.

Fourteen focused tests pass, covering functional exclusion, the two-write
bound, array semantics, prefix execution and track ownership. The full
programmer fast regression was interrupted by the reserve guard when
available memory fell below 5 GiB. It has no passing completion summary;
the partial log is retained in `artifacts/array-statements-validation-20261101`.
Remote CI, other local tracks and slow/retry lanes remain separate checks.

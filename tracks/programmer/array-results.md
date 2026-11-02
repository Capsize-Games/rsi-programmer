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

This trial exposes a construction gap. The initial policy does not observe
execution of its partial program. A bounded follow-up will test that
feedback while preserving this failed trial. This is development work;
the previously frozen transfer bank is not used for repair or tuning.

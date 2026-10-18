# Train-only shortlist parameter selection

This protocol chooses between two engineering settings for the complete
fallback shortlist: a window of 64 descriptors with a shortlist of 4, or a
window of 256 with a shortlist of 16. It uses only the 24 training functions
from the frozen v2 corpus. No evaluation search or evaluation result is part
of this selection.

The harness builds the frozen v2 training groups, fits the seed-0 trained SNN,
MLP, and recurrent controls for the original 40 epochs, and keeps the
heuristic fixed. It checks the fitted weight hashes against the existing v2
training record. It then runs two repeats on every training function for
streaming and for five shortlist modes: cheap-only, heuristic, trained SNN,
MLP, and recurrent. Both shortlist settings use observed training row 0 for
the cheap residual and preserve the complete grammar-order fallback.

A result counts as covered only when the returned graph passes independent
graph validation, all 12 observed rows, and exact polynomial comparison with
that training task's declared generator program. Held-out rows and evaluation
tasks are not used by search or validation. The harness requires each of the
five shortlist modes to cover at least as many of the 48 task-repeat attempts
as streaming at root-signature cap 131072. Among eligible settings it chooses
the lower mean process CPU time across all five modes, all training tasks, and
both repeats. An exact tie goes to the smaller window. If no setting meets
the rule, the run records no selection; the parameters and rule must not be
changed after seeing results.

The selection run records model/group setup, every search counter and raw
attempt result, per-attempt CPU and wall time, environment, source hashes
before and after, and a process timer around setup, searches, validation, and
per-attempt raw-log writes. A command-level execution receipt records the
complete CLI CPU and wall totals, including imports and final artifact writes;
the later total-cost comparison uses that receipt. Those shared selection
costs are charged once per standalone shortlist method at N=48, N=480, and
N=4800. The normal production model setup and fitting charge remains
separate.

The frozen selection recipe is in
`fixtures/benchmark_shortlist_selection_v1.json` and the train-only harness is
`scripts/select_shortlist_method.py`.

Status at code-v80: the preregistration and harness are published before
training or selection search. Run only with the shared builder lock held;
the final evaluation protocol will be frozen after this train-only choice.

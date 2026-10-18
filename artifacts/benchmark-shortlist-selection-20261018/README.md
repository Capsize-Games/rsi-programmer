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

The frozen selection ran after code-v80 publication with the shared builder
lock held. It completed 528/528 attempts with zero errors, a valid complete
matrix, frozen models, identical repeated behavior, and unchanged source
snapshots. Both candidate settings were eligible: each of their five controls
covered all 48 task-repeat attempts, matching streaming's 48/48.

The selected setting is window 256 and shortlist size 16. Its mean attempt
CPU across the five controls was 75.189 ms, compared with 112.204 ms for 64/4.
At 256/16 the individual means were 52.303 ms for cheap-only, 65.043 ms for
the heuristic, 123.805 ms for trained SNN, 63.228 ms for MLP, and 71.563 ms
for recurrent. Streaming averaged 48.764 ms. These measurements choose
parameters only; they are not evaluation results.

The full selection output is in
`artifacts/benchmark-shortlist-selection-run-20261018/`. The external command
receipt is in `artifacts/benchmark-shortlist-validation-20261018/selection.json`:
exit status 0, child CPU 56.792751 s, CLI wall 55.770510 s. The external receipt
is charged by the final comparison because the selector's internal timer
excludes final manifest and selection JSON writes. The final comparison
protocol pins the manifest, raw rows, training output, and external receipt.

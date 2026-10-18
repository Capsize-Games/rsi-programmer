# Online plastic encoder comparison

This frozen comparison tests whether online updates to the six-feature
encoder and observed-context projection improve root-lane choices on the
existing synthetic graph stream. The corresponding protocol is
[`online_plastic_encoder_v1.json`](../fixtures/online_plastic_encoder_v1.json),
SHA-256 `8fd37e1cd6904ab28dae47d48baf95beab68e0819b5a554d920c3d63aa1ebbf9`.

The corpus and task order are unchanged from the counterbalanced diagnosis:
48 tasks, all six phase-order permutations, three model seeds, and budgets
144 and 432. This produces 13,824 episodes. The same 48 task functions recur
across orders, seeds, and budgets; they are not independent samples, and the
experiment does not test transfer to unseen functions.

## Policy and feedback

The policy receives the six observed integer `(x, y, target)` triples for the
current task, the existing six action/history/progress features, and the
available root-lane mask. It does not receive task identity, phase or schedule
position, root label, held rows, or the symbolic polynomial. The environment
executes each selected graph on the observed rows before returning the
existing reward. The independent held checker remains outside the policy.

The base readout update, reward, action sampler, seeded initialization, and
policy RNG are retained. In plastic arms, the existing seeded 16-by-6 encoder
and 18-by-16 observed-context projection become trainable. They start with the
same values as the matched fixed-encoder controls; the policy draws no extra
random numbers for their initialization. The context projection is refreshed
from the current six observed triples at each decision.

Plasticity uses a six-step, current-decision surrogate eligibility. For each
choice, the derivative history starts at zero and covers the six forward
steps for that choice only. The actual membrane state continues through those
steps and resets at the episode boundary. There is no delayed reward credit
across choices, and this is not an exact e-prop implementation. With
`u_t = 0.8*m_(t-1) + input_t`, the binary-LIF surrogate is
`psi(u) = 0.3*max(0, 1-abs(u-1))`; the continuous control uses the derivative
of `tanh(u)`. The frozen fixture specifies the eligibility recurrence and
clipping. Encoder learning rate is 0.01, with update clipping at 0.05 and
weight clipping at 5.0. These values are fixed before the run.

The eight arms are the four readout-only or nonspiking controls, fixed-order
streaming, and three encoder-plastic conditions: persistent binary SNN,
per-task-reset binary SNN, and persistent continuous-leaky control. All
neural arms use the same policy implementation. Fresh policies are created
for every order, seed, arm, and budget. Persistent arms retain learned model
weights and reward baseline within one order; transient membrane and pending
credit clear at episode boundaries. Reset arms restore seeded readout and
encoder state before each task without rewinding the action RNG.

## Retention diagnostic

At stream boundaries 0, 16, 32, and 48, the runner probes all 48 task contexts
in canonical corpus order using a boundary-only read-only policy copy. Each
probe starts with a zero membrane and the initial feature vector, computes
rates and masked root-lane probabilities without sampling or feedback, and
records a source state hash before and after the sweep. The root label is
joined only after the probe to report root-label probability and argmax
accuracy.

These measurements describe cold root-lane preference retention. They do
not measure retention of complete programs or held-out solutions. Probe
copying and execution costs are included in the per-arm total CPU comparison;
search-only CPU is also reported separately. Streaming has no neural policy
to probe, so it receives no fabricated retention values.

## Gates and limits

At budget 144, plastic SNN must exceed the persistent fixed-encoder SNN by at
least 10 percentage points in first-lane accuracy in every order, with a
positive margin for every seed, and accept at least as many tasks as both the
fixed-encoder SNN and per-task-reset plastic SNN. It must also exceed the
plastic continuous-leaky control by 10 points, match or exceed its acceptance,
and use no more total CPU. At both budgets, it must match or exceed streaming
acceptance and use no more total CPU in each order. Every order is reported;
pooled results cannot conceal a failed order.

The report includes accepted tasks, first-lane choices, candidate counts,
updates, readout and full-model hashes, probe measurements, search-only CPU,
and probe-charged total CPU. Full-model hashes include both encoder
projections and readout; readout hashes are retained separately to compare
the unchanged control arms with the prior observed-context bundle. Whole-run
peak RSS is a process-wide maximum, not a per-arm memory estimate.

This is a software diagnostic on one repeated synthetic task stream. It does
not establish generalization to new functions, learning from real sensors,
Raspberry Pi performance, chip behavior, or energy use. Those claims require
separate hardware and sensing measurements.

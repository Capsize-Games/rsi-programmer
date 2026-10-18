# Counterbalanced online graph-search diagnostic

This frozen follow-up replays the existing synthetic task stream under every
ordering of its three operator-family phases. It tests whether the previous
phase-order pattern is consistent across these six schedules. It does not
change the learner, establish causality, test physical sensing, or measure
hardware energy.

## Frozen inputs and schedule

The task and answer source remains
[`online_graph_v1.json`](../fixtures/online_graph_v1.json), SHA-256
`a4f9af2dfc0d8c4539d700b032dcc223c82aa272f02497902024d9f371096140`.
The schedule is frozen in
[`online_counterbalanced_v1.json`](../fixtures/online_counterbalanced_v1.json),
SHA-256 `0ffbd7eadac93aa63b8df68d9ba250cd4da5a55a178e8339f9505d81638af3f1`.
That schedule lists all six permutations of phases `0`, `1`, and `2`. Within
each phase, the original order and task IDs are preserved. No targets, inputs,
held answers, or task assignments are generated or modified.

For each phase order, model seed (`0`, `1`, `2`), arm, and budget (`144`,
`432`), start a fresh policy. The persistent SNN carries weights between
episodes within that order only. The frozen SNN keeps its seeded initial
weights; the episode-reset SNN resets learning at each task; the online
nonspiking policy and grammar-streaming control retain their existing
behavior. No state carries between phase orders or budgets. Keep the existing
five arms, learner parameters, random seeds, candidate grammar, evaluator,
budgets, and update rules fixed.

The planned matrix has 6 orders × 3 model seeds × 5 arms × 2 budgets × 48
tasks = 8,640 episode rows. Replaying a task in multiple orders is repeated
measurement of that same task, not a new independent task. Write every
attempt, including failures and censored episodes, to the raw results. Bind
the run to this schedule, the unchanged source corpus, and the unchanged
runtime through hashes. The policy receives only the existing episode state
and lane mask; it receives no task ID, phase, root family, schedule position,
answer key, or held data.

## Analysis and gates

Pair outcomes by task ID within each model seed, arm, and budget. This pairs
the same task content; it does not align random draws across orders. A policy
that takes a different number of decisions can reach a different decision
RNG offset. Record `rng_draw_offset_before` and `rng_draw_offset_after` on
each episode row; these are cumulative choose-call counts.
Report the original root-operator family, the position of its phase in the
presented order, and the task's ordinal within that phase as separate
dimensions. Show per-order and per-seed counts, acceptance, first-lane
accuracy, candidate work, and full CPU cost. Do not combine a phase-position
contrast with a root-family contrast or treat the repeated rows as
independent samples.

At the frozen diagnostic points, record logits, probabilities, entropy,
weight norm, clipped-weight count, and reward baseline from a cold policy
probe. The probe copies membrane state for the existing six-step drive. It
must not alter policy input, choices, RNG state, weights, or training state.
Its CPU cost is included in attempt accounting.

At budget 144, the adaptation gate must pass separately in all six phase
orders: persistent SNN must exceed both frozen and episode-reset SNN in
first-lane accuracy by at least 10 percentage points pooled over the 144
task-seed attempts in that order, have a positive first-lane margin for each
seed, and accept at least as many tasks as each control. The SNN-specific
gate also applies in every order: persistent SNN must exceed online
nonspiking first-lane accuracy by at least 10 percentage points, accept at
least as many tasks, and use no more total CPU. At both budgets and in every
order, the streaming deployment gate requires at least as many accepted
tasks and no more total CPU than grammar streaming. Report each order's gate
result; a pooled pass cannot hide a failed order.

The diagnostic labels underperformance *consistent across orders* only if,
at budget 144, persistent SNN has both fewer accepted tasks and fewer correct
first-lane choices than frozen SNN and episode-reset SNN in all six orders.
This is a descriptive label, not a prerequisite for further investigation.
An exploratory mechanism follow-up may be considered if the pooled budget-
144 results show fewer persistent-SNN accepted tasks and no more correct
first-lane choices than both frozen and episode-reset SNN, and persistent
SNN has fewer accepted tasks than episode-reset SNN in at least four of the
six orders. This trigger is exploratory evidence, not statistical
significance or proof of a phase-order effect. It does not establish that
carryover caused underperformance. Do not change this protocol or tune
learning parameters based on its outcomes. Any mechanism test requires its
own frozen design.

Retain all failures in summaries and denominators. Report CPU as host process
time including policy setup/reset, graph execution, online update, and held
checker work, using the existing runner's accounting. A sequential run may
report whole-process peak RSS only; it cannot support per-arm peak-memory
claims. CPU and RSS are not chip-energy or Raspberry Pi performance measures.
The result remains a bounded diagnostic on one 48-function synthetic stream,
not evidence of transfer to general code or real sensors.

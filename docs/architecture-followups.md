# Architecture follow-ups

The next experiment moves policy selection before candidate execution.
Typed program execution and independent acceptance checking remain the
foundation. The existing eager search remains a reference implementation.

## Implementation scope

- [#72: pre-evaluation selection](https://github.com/Capsize-Games/rsi-programmer/issues/72)
  adds lazy candidate evaluation, cached operand results and a budget on
  actual candidate evaluations. Proposal features may use the training
  specification and previously computed operands, but not the proposed
  candidate's result.
- [#73: frozen benchmark](https://github.com/Capsize-Games/rsi-programmer/issues/73)
  compares streaming, heuristic, random, untrained SNN, trained SNN, MLP
  and recurrent nonspiking controls. The protocol and corpus must be
  published before solver measurements. Evaluation answers cannot guide
  search or retry decisions.

## code-v60: pre-evaluation selection verified

`pre_evaluation_search` is an additive path in `rsi_programmer/pre_eval.py`.
It supports arithmetic operations and optional constant terminals. It
computes root vectors from cached operands, ranks before evaluating roots,
and preserves generation-order retention after complete layers. It
supports an optional oracle for callers that require retry; the frozen
benchmark will use no oracle feedback. A training match alone is not an
accepted held-out solution.

The fixed heuristic, seeded random, MLP and nonspiking recurrent controls
are implemented. MLP and SNN each have 225 parameters; recurrent has 497.
Equal parameter counts do not establish equal capacity or compute.
Recurrent control state is fresh per candidate drive, not task memory.

Sequential validation under `/tmp/rsi-programmer-builder.lock` passed:

- Focused selection/control checks: **15 tests in 0.442s; OK**.
- Complete `just test-fast`: **285 tests in 224.482s; OK**.

There were no exclusions or skips in those runs. The manifests bind
unchanged search, control, legacy runtime/test and fixture inputs before
and after each command. Benchmark files under construction are excluded
from this checkpoint's input scope and were not imported by these tests.
Slow, retry and full discovery were not rerun: this change adds a path and
controls without changing the existing eager runtime. The frozen benchmark
remains in progress; no new performance result is claimed here.

[Focused log](../artifacts/pre-eval-validation-20261018/focused.log),
[fast log](../artifacts/pre-eval-validation-20261018/fast.log),
and [fast input manifest](../artifacts/pre-eval-validation-20261018/fast.json).

## Conditional experiments

- [#74: episode state and structured actions](https://github.com/Capsize-Games/rsi-programmer/issues/74)
  defines task identity, resets, branch rollback and action contracts
  before testing persistent controllers.
- [#75: controller decomposition](https://github.com/Capsize-Games/rsi-programmer/issues/75)
  compares proposer, evaluator and executive modules with a single
  controller after the initial benchmark exists.
- [#76: reusable abstractions](https://github.com/Capsize-Games/rsi-programmer/issues/76)
  starts with explicit typed program storage and compares retrieval
  methods with their cost included.

## Deferred research

- [#77: SNN subsystem replacement](https://github.com/Capsize-Games/rsi-programmer/issues/77)
  requires a measured bottleneck and a replacement contract before
  changing deterministic execution, storage or checking.
- [#78: representation capacity](https://github.com/Capsize-Games/rsi-programmer/issues/78)
  requires evidence of capacity limits or interference before adding
  feature alignment or representation-analysis infrastructure.

## Decision criterion

Judge the initial experiment by independently checked evaluation success
at fixed candidate work and by work at matched success. Report policy,
operand, validation and elapsed-time costs separately; candidate counts
alone do not establish a total-compute improvement. Keep the predeclared
recipes, unsuccessful tasks and negative results. Repeated seeds do not
create independent tasks, and a neural contribution does not establish a
specific benefit from spiking dynamics.

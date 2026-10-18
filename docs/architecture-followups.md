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

Both implementations are in progress. No tests or benchmark results are
claimed by this planning checkpoint. The code-v58 validation evidence
applies to the preceding runtime only.

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

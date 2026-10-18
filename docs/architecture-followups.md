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

## code-v62: frozen benchmark runner verified

The protocol, corpus and generator were published in `code-v61`
(`f5c9b85`) before solver measurements. Their hashes remain unchanged.
There are 12 training tasks, 24 evaluation tasks, three training seeds,
four candidate-evaluation budgets and seven controls: 2,016 evaluation
attempts in the complete matrix. Every sampled task is retained; the
generator first enumerates larger syntax pools and samples from them.

The runner accepts only the pinned inputs. It records actual fitting
forward/update work, separate setup and fit costs, independent observed
and held-out checks, returned graphs, all failures, state and weight
hashes, and source/dependency provenance. Missing combinations, source or
environment drift, evaluation-time state changes and execution errors
invalidate the run. Held-out answers never guide search or retries.

Sequential validation on the complete benchmark implementation passed:

- Focused checks: **32 tests in 0.550s; OK**.
- Complete `just test-fast`: **302 tests in 222.978s; OK**.
- Corpus regeneration in a temporary directory reproduces the exact
  frozen bytes using Node.js v22.23.3; published inputs were untouched.

No tests were skipped or excluded from these commands. The manifests
cover all runtime/test Python, scripts, JSON fixtures, requirements,
justfile and the full imported SpikeForge Python source tree, unchanged
before and after each command. Slow, retry and full discovery were not
rerun; the preceding eager paths remain unchanged. Full measurement and
its independent result audit are pending at this checkpoint; #73 stays
open and no performance advantage is claimed.

[Protocol and usage](benchmark-pre-eval.md),
[focused log](../artifacts/pre-eval-benchmark-validation-20261018/focused.log),
[fast log](../artifacts/pre-eval-benchmark-validation-20261018/fast.log),
[fast provenance](../artifacts/pre-eval-benchmark-validation-20261018/fast.json),
and [generator check](../artifacts/pre-eval-benchmark-validation-20261018/generator.log).

## code-v63: complete measurement and independent audit

The first fixed benchmark completes all **2,016 attempts** on the
`code-v62` runtime without execution errors. All 32 independent audit
checks pass, including re-execution of every returned graph (1,406),
complete matrix coverage, exact summaries, budgets, frozen weights and
source/dependency hashes. The initial verifier failure from an incorrect
streaming descriptor-count assumption is retained and explained in the
[result report](benchmark-pre-eval.md#version-1-result).

There is **no supported trained-SNN advantage** under the frozen rule.
At budget 512, streaming accepts 72/72 repeated task-seed attempts;
trained SNN, untrained SNN and MLP each accept 62/72. Every control accepts
72/72 by budget 2048. At matched full acceptance the trained SNN uses
more candidate signature evaluations and recorded search time than
streaming. This result covers 24 evaluation tasks and three seeds in the
small arithmetic DSL, not general SNN capability or broad transfer.

#72 and #73 are complete implementation/evaluation tasks. The next
benchmark follow-up is [#79](https://github.com/Capsize-Games/rsi-programmer/issues/79):
semantic separation and workloads that remain discriminating at the
declared budgets. Exact polynomial expansion finds 19 distinct functions
among the 24 evaluation tasks and one function shared with training.
Do not promote the controller, decomposition or memory experiments on
this result; #74–#76 remain conditional and #77–#78 remain deferred.

Local validation remains **302 fast tests OK in 222.978s**, plus the
32 focused checks and exact corpus regeneration. Slow, retry and local
full discovery were not rerun for the additive path. Current code-v62 CI
fast jobs pass on Python 3.12 and 3.13; both full jobs remain running at
the recorded observation and are not credited as passing.
The older code-v53 full Python 3.13 job finished with one live hub
verification failure in 337 tests; its Python 3.12 sibling was cancelled.
The venv fix is present, but [#44](https://github.com/Capsize-Games/rsi-programmer/issues/44)
remains open for a completely green full-CI cycle. No cause for that hub
failure is established by the assertion alone.

[Complete raw measurement](../artifacts/benchmark-pre-eval-v1-20261018/),
[final independent audit](../artifacts/benchmark-pre-eval-v1-audit-20261018/audit.json),
[measurement log](../artifacts/pre-eval-benchmark-validation-20261018/benchmark.log),
and [dated CI status](../artifacts/pre-eval-benchmark-validation-20261018/ci-observation.json).

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

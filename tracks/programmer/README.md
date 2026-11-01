# Programmer

This is the primary development track as of 2026-11-01.

The primary outcome is a learner that writes useful programs in Python and
CURLEE. The [stateful follow-up](stateful-results.md) solves 0/5 new tasks;
one broader curriculum also solves 0/5 and retains 5/6 earlier utilities.
Supplied controls pass all five in Python and CURLEE. The preceding
[fragment-transfer result](repair-results.md) repairs two
failures and retains four programs, with 40 held cases each in both target
languages. The preceding [statement result](statement-results.md) constructs
loops, branches, updates and returns for four of six development tasks.
The earlier [utility result](language-results.md)
composes supplied list operations. Python reads and writes JSON files;
CURLEE has a bounded processing interface. General source generation and
program repair remain open.

The objective is a neural controller that constructs executable programs
from a task specification, learns from training execution and feedback, and
reuses prior solutions on new tasks. Independent execution and acceptance
remain outside the learner. Comparison with SNNs or enumeration is a
secondary diagnostic rather than a gate on useful language work.

## Existing foundation

- `construction_*` and `target_source.py`: sequential arithmetic decisions,
  training, target-language lowering and independently checked executions.
- `utility_policy.py`, `utility_ops.py` and `utility_source.py`: learned
  selection from 13 supplied list operations, plus target-code emission.
- `statement_*`: typed imperative syntax, independent execution, observed
  prefix execution, neural token choices and direct target-code lowering.
  It now also includes bounded query feedback, qualified context mining,
  neural hole filling and a classical transfer control. Classical search
  wins the current small transfer comparison; neural scoring is not a
  prerequisite for source correctness.
- `probe.py` and `actions.py`: typed expression graphs, validation and exact
  interpretation of the supported arithmetic/Boolean language.
- `policy.py`, `modules.py`, `author.py`, `lattice.py`, `lauthor.py` and the
  search modules: learned ranking/filtering and graph-commit experiments.
- `decoder.py`: a standalone spike-to-edit boundary, not wired into search.
- `online_*`: graph-execution learning experiments and matched controls.
- `benchmark_*`, frozen fixtures and auditors: independent result and cost
  checks, including unsuccessful comparisons.
- `edits.py`, `self_gate.py`, `loop_trainer.py` and `scripts/rsi_loop.py`:
  bounded configuration trials and lineage, without demonstrated recursive
  improvement.

The existing search generators, interpreter and acceptance checks are
classical code. The principal SNN-specific and total-cost gates have not
passed. See the [complete shortlist result](
../../docs/benchmark-complete-shortlist.md) and [encoder-learning result](
../../docs/online-plastic-encoder-result.md).

## Capability roadmap

Bounded query feedback also produced no programs for the five stateful
tasks. The next change should target construction and transfer of
state-dependent updates. Retain all failed attempts and the qualified
earlier model; broader typed holes and loop-state features remain proposed
experiments.
The stages below describe the broader capability progression. They do not
require repeating model-family comparisons before useful-program trials.

1. **Learn to construct programs.** Define a sequential typed action API:
   choose an operation, select operands from live references, extend the
   program, and stop. Add an encoder for actual observed input/output rows
   and partial-program state. Start with an ANN controller and retain a
   matched SNN control; changing the existing ranker's activation alone is
   not this controller. Search can explore its choices, but reports must
   distinguish learned actions from enumeration and fallback.
2. **Establish learning on new compositions.** Build a training curriculum
   with recorded generator bias, training traces and execution feedback.
   Separate training, tuning and final evaluation by program function and
   composition. Generate supervision from the task engine rather than an
   LLM. Include an untrained control and record search effort. Retain prior
   model-family and classical-search comparisons as separate diagnostics.
3. **Add Python output.** Lower the common typed representation to a
   restricted Python AST and compare its execution with the independent
   interpreter. Initially use pure functions with bounded inputs. Expanding
   to assignments, arrays and loops requires explicit semantics and tests.
   The current arithmetic sandbox is not an arbitrary-Python sandbox; any
   generated Python execution needs a separately validated OS isolation and
   resource contract.
4. **Add CURLEE output.** Emit source for the compiler's supported pure
   function fragment. Keep task contracts evaluator-owned, run the checker,
   and independently execute test cases on the selected VM/compiler
   revision. Record verification and execution separately. Align integer
   range and overflow semantics; Python's unbounded integers are not an
   automatic match for CURLEE runtime arithmetic.
5. **Learn reusable fragments and repairs.** Store verified fragments with
   provenance and measure transfer to unseen compositions against ordinary
   retrieval. Add bounded repairs using training-visible diagnostics and
   counterexamples. Final held outcomes must not guide repair retries.
6. **Reopen self-improvement.** Only after the programmer is useful, test
   learned proposals that change the learner and survive independent
   evaluation on fresh tasks. Current configuration tuning, hand-written
   proposer rules, and descriptive autonomy fields are insufficient evidence.

The restricted construction policy, arithmetic language backends and first
list-composition utility are implemented. The utility emitter supplies loops,
branches, arrays and their contracts; the learner chooses the operations.
Learning arbitrary control flow and the later stages remain work.
A deployed LLM teacher or pretrained coding model
would change the research question and requires an explicit owner decision.

## Acceptance and stopping rules

- A learned program must pass independent correctness checks. Structural
  validity, training loss and compiler success alone are insufficient.
- The primary acceptance is generated Python and CURLEE code executing
  correctly on independently held behavior tests. Cost and model comparisons
  remain separate diagnostics. The earlier frozen construction cost gate is
  preserved for that experiment, rather than blocking language work.
- Report fitting, supervision generation, search, neural scoring, checking,
  and actual memory/CPU separately. Charge training at a declared usage
  volume; an amortization scenario is not additional task evidence.
- Freeze final evaluation before tuning. If it is reused for diagnosis,
  label the work retrospective and acquire new final tasks.
- Preserve failed experiments and diagnose them before adding capacity or
  self-editing. Target-language work follows the owner's corrected priority.

Initial candidate workloads are small arithmetic/branch functions, bounded
array transforms, range checks and bit-field operations. Their exact scope
and supported semantics must be frozen in the first controller protocol.
General repository maintenance, arbitrary packages, device drivers and
unrestricted natural-language requests are outside the first milestone.

## CURLEE integration evidence

Read-only inspection on 2026-11-01 used the local CURLEE checkout's
`docs/Stability-and-Supported-Fragment.md`, `docs/Verification-Scope.md` and
`docs/Running-Programs.md`, plus the function-contract and bounded-loop
examples. It documents functions, assignments, branches, loops, fixed-size
arrays, contracts, structured diagnostics and a fuel-bounded VM. Those
features support the restricted backend. Its later execution checks are
recorded in [language-results.md](language-results.md); they do not qualify
the compiler generally.

The compiler is pre-alpha and its verification scope records limitations.
A proof concerns its modeled contract, not arbitrary task intent, compiler
correctness, external function bodies or hardware. In particular, the
documented integer/runtime correspondence limits require qualification.
Use local, source-pinned compiler inputs; do not copy compiler source or
private training data into this repository.

## Research basis

[DeepCoder](https://www.microsoft.com/en-us/research/publication/deepcoder-learning-write-programs/)
demonstrates neural guidance for bounded program synthesis.
[DreamCoder](https://arxiv.org/abs/2006.08381) combines neural search with
learned reusable symbolic abstractions.
[AlphaDev](https://www.nature.com/articles/s41586-023-06004-9) demonstrates
reinforcement learning for narrowly specified algorithm construction.
These results support a bounded research experiment; they do not establish
that this repository can become a general programmer or reproduce their
results with its current models and resources. No upstream implementation
was adopted or executed for this split.

## Commands

```sh
just test-programmer          # fast checks
just test-programmer all      # includes slow and retry lanes
just train                   # existing SNN ranker reproduction
```

`just train` still trains the existing ranker. It does not train the planned
ANN construction controller. The [original plan](../../programmingplan.md)
is retained as historical design context.

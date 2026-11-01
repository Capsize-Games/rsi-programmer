# Programmer

This is the primary development track as of 2026-11-01.

The objective is a neural controller that constructs executable programs
from a task specification, learns from training execution and feedback, and
reuses prior solutions on new tasks. Independent execution and acceptance
remain outside the learner. Python and CURLEE are planned output targets;
the current implementation does not synthesize or execute general programs
in either language.

## Existing foundation

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

## Development sequence

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
   LLM. Compare trained/untrained ANN, SNN, and classical controls under the
   same information and end-to-end budgets.
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

These are planned stages, not implemented capabilities. Stage 1 is the next
implementation milestone. A deployed LLM teacher or pretrained coding model
would change the research question and requires an explicit owner decision.

## Acceptance and stopping rules

- A learned program must pass independent correctness checks. Structural
  validity, training loss and compiler success alone are insufficient.
- The first controller must show better held-task solve rate at equal total
  cost, or lower total cost at equal correct coverage, against classical
  search. Freeze the required margin, uncertainty method and resource cap
  before measuring the final tasks.
- Report fitting, supervision generation, search, neural scoring, checking,
  and actual memory/CPU separately. Charge training at a declared usage
  volume; an amortization scenario is not additional task evidence.
- Freeze final evaluation before tuning. If it is reused for diagnosis,
  label the work retrospective and acquire new final tasks.
- If the first bounded experiment fails its gate, retain it and diagnose or
  stop that approach before adding capacity, languages or self-editing.

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
features make a restricted backend plausible; no backend or compiler
qualification was performed for this track split.

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

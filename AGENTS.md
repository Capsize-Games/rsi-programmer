# rsi-programmer

BSD-3-Clause software, copyright Capsize LLC (see [LICENSE](LICENSE)).

## Commands

- `just setup` — create `.venv`, install requirements, configure a local
  spikeforge source checkout (pass its absolute checkout root as the
  positional argument: `just setup /path/to/spikeforge`).
- `just test` — full unittest suite, run from the repo root.
- `just train` — canonical training run (seed 0, 50 epochs, writes
  `weights/`).

## Conventions

- One class per file, full type hints, 79-column lines. When a limit is
  reached, split the module instead of growing past it.
- `rsi_programmer/lineage.py` is the only persistence: a sqlite record of
  missions and intent entries. It replaces the rover mission stack the
  training loops were extracted from; keep training runnable without it
  gaining mission/authority semantics.
- The engine decides success: a synthesised program counts only when the
  independent checker (`probe.validate_graph`) holds on held-out rows.
  Learner-reported scores alone prove nothing.
- Never commit `weights/*.pt`, lineage databases, or `.venv/`.

## Research tracks

Owner direction (2026-11-01): keep three tracks in this repository and
concentrate new development on the original programmer objective. The current
scope is in `tracks/README.md`; the staged plan is in
`tracks/programmer/README.md`.

- Programmer owns typed program construction, search, graph-execution
  learning, independent checking and bounded learner-improvement research.
  Use the current ANN controller for useful-program trials. Retain completed
  SNN comparisons as evidence; a spiking architecture is not a prerequisite
  for success and another model-family comparison is not the current task.
  A restricted arithmetic construction controller is implemented; its early
  development results do not establish a cost advantage. Restricted Python
  and CURLEE source backends and their first execution checks are implemented.
- Sensing preserves timed sensor/audio prediction and adaptation. Sound
  Companion preserves the native supervised sound-learning application.
  Keep their outcomes separate from programmer capability claims.
- Runtime paths and historical evidence remain stable during this logical
  split. Update `tracks/manifest.json` when adding test modules. Run the
  affected tracks for shared changes; focused track passes do not replace
  existing aggregate or release gates.
- The deployed learner must make its own programming choices and learn
  without an LLM teacher. A pretrained coding model would change that
  research scope and requires an explicit owner decision.
- Python and CURLEE execution require independent backend qualification.
  The arithmetic graph sandbox does not establish arbitrary-Python
  isolation, and a CURLEE proof covers only its declared modeled property.

Owner priority (2026-11-01): establish results early. Use bounded development
trials and report actual solve rates and costs before expanding infrastructure.
Retain failed and negative trials. Do not spend
hours on setup while deferring the first learning measurement. The final task
set stays sealed while development choices are made.

Owner correction (2026-11-01): the primary outcome is actual programming in
Python and CURLEE. Do not substitute ANN/SNN comparisons or arithmetic-IR
benchmarks for generated target-language code that checks and executes.
Model comparisons and speed against enumeration are secondary diagnostics,
not prerequisites for target-language work. Preserve their results. Report
the supported language constructs explicitly; arithmetic-only source does
not establish branches, loops, arrays, libraries or general program repair.

Further owner clarification (2026-11-01): demonstrate real programs that
perform useful tasks. Arithmetic wrappers do not meet that requirement.
The first utility experiment learns a composition from 13 integer-list
operations and emits a standalone Python JSON-file CLI plus a bounded CURLEE
processing routine. The emitter supplies primitive implementations, loop
contracts and CLI plumbing. Describe this as library-composition synthesis;
do not claim the learner invented those implementations or can write
arbitrary source. Prioritize additional useful task results over model-family
comparisons or extended setup. Preserve failures and independent execution
checks. CURLEE source remains in its local-only repository; use its existing
resource guard for compiler execution and never modify or publish it here.

Implementation checkpoint (2026-11-01): `statement_*` now constructs a typed
statement tree from individual loop, condition, update and return tokens.
The execution-context development recipe has four of six tasks accepted by
actual Python and CURLEE runs; see `tracks/programmer/statement-results.md`.
Keep the two failures and earlier negative trials visible. Its CLI plumbing,
initial accumulator and source lowering remain supplied code. Do not claim
arbitrary programming, formal compiler qualification or a cost advantage.
The next capability checks are bounded counterexample-guided resynthesis,
new statement structures and useful fragment transfer. Keep held evaluation
rows outside repair/query feedback and preserve the sealed final set.

Subsequent implementation checkpoint (2026-11-01): bounded query-only
resynthesis repaired neither failure. Three condition contexts mined from
four independently qualified programs allowed neural hole filling to repair
both tasks; all six programs passed 40 held Python and CURLEE cases each.
See `tracks/programmer/repair-results.md`. Donor retention and transfer tasks
must remain separately identified. Context mining is classical; neural
context scoring was slower, and the classical hole-search control was much
faster. Do not claim a cost advantage, general programming, learned
abstraction discovery or recursive self-improvement from these results.
Keep actual held rows out of feedback. Check source and recorded executions
before promoting donors; preserve provenance, failed trials and controls.

## Audit checkpoints

Branch-composition checkpoint (2026-11-01): the integrated route accepts
all 11 development tasks, with 40 actual held Python and CURLEE cases per
task. A host-supplied grammar copies two qualified branches and enumerates
add/sub variants; neural token scores choose a shared Int argument. A
positive counter already present in training was separately qualified as
a donor; do not count it as unseen-function success. See
`tracks/programmer/branch-results.md`. The initial condition-first
classical control misses one task; the stronger composition-first control
accepts all 11 on fresh IR rows and is faster in this small experiment.
Controls are held-IR only, while the primary programs have actual target
executions. Do not claim neural superiority, matched-total-cost advantage,
general programming, learned abstraction discovery or recursive
self-improvement. These eleven tasks have been repeatedly used in
development. Next freeze unseen-function evaluation and total-cost
accounting before expanding capacity. Keep the existing sealed set,
all negative trials, donor provenance and held/query separation intact.

Context-repair checkpoint (2026-11-01): bounded visible-query resynthesis
repairs minimum retention and retains maximum and the earlier six
utilities. All eight pass 40 held Python and CURLEE cases each: 8/11 tasks
accepted. See `tracks/programmer/context-repair-results.md`. Three tasks
remain unsolved. Models stay fixed during repair; this is conditioning on
counterexamples, not neural weight adaptation or recursive improvement.
Classical and untrained controls are held-IR checks only; classical also
accepts 8/11. Do not claim a matched-total-cost advantage. The unchanged
package/test sources pass 530 programmer fast-track tests; other lanes
and tracks were not rerun, and remote CI remains pending. Next test bounded
composition of qualified branch fragments, preserving all failures,
provenance, controls and separation of held data from feedback.

Context-growth checkpoint (2026-11-01): condition filling in the qualified
minimum context constructs maximum, with 40 held Python and CURLEE cases
passing. The retained-model route keeps six old utilities and transfers
maximum, but selects a wrong minimum program: 7/11 accepted. The original
minimum remains separately qualified; never promote the wrong attempt.
See `tracks/programmer/context-growth-results.md`. Controls are held-IR
checks only; primary programs have actual target-language execution.
Classical search accepts 8/11 and constructs maximum much faster. The
next bounded check is visible-query repair of donor retention, preserving
this failure and keeping every held row outside feedback.

Component-transfer checkpoint (2026-11-01): loop-body contexts and wider
search alone still produced 0/5 new programs. A component-preserving
curriculum produced the minimum-with-zero-ceiling program, with 40 actual
Python and CURLEE cases passing. It solves 1/5 new tasks and retains 4/6
earlier utilities; do not replace the previously qualified model. Its
algorithmic components occur inside other teacher functions. This is
component transfer, not algorithm invention or general programming. See
`tracks/programmer/component-results.md`. The new minimum is a qualified
donor; a subsequent maximum candidate fits observed examples only and
requires held target-language execution. Retain all failures, controls,
provenance and the sealed final set.

Stateful development checkpoint (2026-11-01): the qualified frozen learner
solved 0/5 new stateful tasks. Supplied grammar controls passed 40 Python
and CURLEE cases per task; those controls are never neural successes or
library donors. One broader 1,151-function execution curriculum still
solved 0/5 and retained only 5/6 earlier utilities, so do not promote it.
See `tracks/programmer/stateful-results.md`. Preserve the failures. The
maximum and capped-sum tasks share identical initial examples; separate
that ambiguity from construction failure using bounded visible queries.
Keep held rows outside feedback and leave the sealed final set untouched.
The subsequent three-query trial produced no programs for either model.
Its 37-input query pool excluded all 73 prior held inputs. Construction and
transfer of state-dependent updates remain the next gap; broader typed
holes and loop-state features are proposals, not established fixes.

Frozen transfer checkpoint (2026-11-01): the preregistered 18-function bank
accepts 11/18 trained, 14/18 classical and 11/18 untrained programs in both
Python and CURLEE. Every emitted program ran on all 20 held cases per
language. Full measured controller CPU costs are 71.614, 21.127 and 26.600
seconds respectively, including fresh fitting where applicable. Historical
creation of the shared donor library is excluded. See
`tracks/programmer/frozen-transfer-results.md`. No training or cost benefit
is established. Preserve all failures and the original sealed arithmetic
final set. Any solver changes informed by this bank make it development
evidence and require a new frozen bank for subsequent final claims.

List-output development checkpoint (2026-11-01): the first array-statement
ANN constructs 0/3 tasks, as does its untrained control. Three supplied
controls pass 20 Python and CURLEE cases each after qualifying a structural
output-bound invariant. These controls are never neural successes or
donors. The initial guard stop and compiler rejection are retained. See
`tracks/programmer/array-results.md`. The grammar supports one input loop,
conditions and at most two append nodes; source lowering, CLI plumbing and
contracts remain supplied. Partial-program execution features and a wider
search also construct 0/3 for both trained and untrained controllers. The
broader trial reproduces the preceding parameters; retain all three
failures. A fourth supplied two-append control passes 20 cases in both
languages, including 512 outputs from 256 inputs. Qualified neural fragment
transfer or a structural curriculum change remain proposals. No supplied
control is a neural donor.

Owner direction (2026-11-02): measure progress toward general programming
through broader unseen task families, transfer across new combinations,
retention and reduced task-specific host assistance. A growing count of
hand-shaped special cases is insufficient. Do not send Mattermost messages.

Unattended campaign checkpoint (2026-11-02): the first runner campaign
completed both development recipes at 0/3 and stopped NO_MEASURED_GAIN
without opening its audit. See `tracks/programmer/campaign-runner.md`.
The current adapter only supports bounded integer-list programs. Automatic
curriculum invention, fragment-library growth and broader types remain
unimplemented; do not claim this runner establishes generality growth.
Its separate supplied-control checker smoke passes two actual Python and
CURLEE cases, including 512 outputs from 256 inputs, and rejects a wrong
expected output. This is never a neural success or donor. Resource-aborted
full regressions remain incomplete. Preserve checkpoint/source hashes,
sealed audit separation, runtime weight exclusions and memory guards.

When publication is authorized, publish completed fixes and review evidence
as the work progresses, including during long test runs. Push meaningful
checkpoints and tag published snapshots. Record the exact completed tests,
partial runs, and pending checks in each checkpoint. Update issue status
with the evidence available; keep unresolved findings open. Publish again
at validation milestones instead of holding all evidence for the final run.

Owner update (2026-10-18): commit and push completed changes without waiting
for remote CI. Run affected local checks and state pending CI accurately;
fix actual failures without weakening tests. Pause new tags and releases
until the owner asks to resume them.

## Sound Companion

Owner correction: the creature's purpose is to learn and remember sounds
with the SNN. The SNN must choose its action label; keep the previous
example matcher as a comparison control. Do not replace neural learning
with example matching or route actions through the control. A separate
rejection guard may suppress an unsupported neural choice, and must be
described as such. Preserve evaluation-only recordings, retain old taught
examples for replay, and compare both backends on the same fresh sounds.

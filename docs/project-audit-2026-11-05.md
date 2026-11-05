# Project audit, 2026-11-05

Whole-project audit at HEAD `16d27a0` (2026-11-02). Read-only: no tests,
training or program executions were rerun. Every result document in
`tracks/programmer/` was opened, numbers were spot-checked against the
JSON artifacts they cite, and the sensing, sound-companion, benchmark
and online result documents were tallied head-to-head. Claims that rest
on a single document or a self-reported field are marked as such in
the coverage section. This document records what the experiments show,
what is worth pursuing, parking or dropping, and — for each pursue
line — what it actually gets us in the end.

## Bottom line

The repository is a disciplined measurement of a negative result.
Training reliably beats untrained and random controls. It never beats
the best classical comparator at matched cost, in any experiment
family, on any track. That includes the only two non-adaptive tests
(the frozen 18-function bank and the fresh 6-function edit-value
bank). No experiment establishes neural superiority, a matched-total
cost advantage, learned abstraction discovery, algorithm invention,
general programming, or recursive self-improvement. The result
documents themselves say this at every stage; the audit found no
hidden positive and no supplied control ever counted as a neural
success or promoted as a donor.

The measurement machinery is the most valuable engineering in the
repo: sealed and hashed task banks, the `qualify()` donor gate,
dual-backend (Python and CURLEE) execution with wrong-expected
controls, actual-case counting, and retained failures. That machinery
is why the negative result is trustworthy. It is surrounded by
per-trial single-use stacks, three parallel IR/interpreter routes,
at least five different success deciders, and stale front documents.

## The stated destination

The programmer track's objective (`tracks/programmer/README.md:48-52`)
is "a neural controller that constructs executable programs from a
task specification, learns from training execution and feedback, and
reuses prior solutions on new tasks", with independent execution and
acceptance outside the learner. The primary outcome is a learner that
writes useful programs in Python and CURLEE. The six-stage roadmap
ends with self-improvement, explicitly gated: "Only after the
programmer is useful, test learned proposals that change the learner
and survive independent evaluation on fresh tasks." A deployed LLM
teacher or pretrained coding model would change the research
question and needs an explicit owner decision.

Sensing is a separate preserved workload: online prediction,
adaptation, retention and change detection on timed measurements.
Sound Companion is a separate preserved application: a teachable
sound creature whose SNN chooses the label, with the example matcher
as control only. Neither track's results are programmer evidence,
and that separation was observed in practice.

None of the three pursue lines below builds the stated destination
directly. Each one is a decisive test of whether a single candidate
component deserves to be a brick in it. The final section answers
the overall "building towards" question in those terms.

## What was run

Three waves: pre-November arithmetic search/ranking benchmarks and
the sensing/sound work (2026-10-18/19); the full target-language
chain on 2026-11-01; the campaign runner and edit-value trials on
2026-11-02.

### Early arithmetic construction (135 development functions)

The only adequately powered learning comparison. Trained ANN
(100 epochs) beam search solves 78/135, ANN-20 solves 42, SNN-20
solves 53, untrained 0, random 1. Classical streaming enumeration
solves 135/135 in 1.033 s — faster than every neural arm before
fitting is counted (ANN-100 fit alone is ~61 s). The 78 solved
programs genuinely execute in Python and CURLEE (4,680 cases per
language); the 57 unsolved stay failures. Learning signal vs weak
controls: real. Advantage over trivial classical search: none.
Caps were unequal (256 roots vs 16,384).

### Benchmark ranking (pre-eval, total-work, bounded, shortlist, online)

Pre-eval v2 at a fixed 8192 signature cap is the repo's only
solves-more-than-streaming result: trained SNN 99/144 vs streaming
60 and untrained 65, across 3 seeds on function-disjoint tasks.
But MLP scores 102 and a recurrent model 107, and at that cap the
SNN search takes ~17x streaming's time. It establishes generic
learned ranking, not an SNN advantage and not a cost advantage.
Pre-eval v1, total-work, bounded and shortlist all fail their
preregistered cost/SNN gates. Online persistence actively hurts
(performance degrades across phases); the plastic encoder buys
+1 acceptance at ~3x cost. Candidate-count savings never convert
to CPU wins.

### Utility library-composition

A standalone Python JSON-file CLI (100/100 files, both languages)
and 5/8 follow-up tasks. The learner's whole contribution is
ordering 3 of 13 host-implemented integer-list operations; the
emitter supplies 100% of the source (implementations, loops,
contracts, CLI). No classical comparator was ever run for this arm.
This is library-composition synthesis, exactly as the docs describe
it — useful engineering, no learning evidence beyond selection.

### Statement construction and the repair/branch chain

Direct token-by-token construction of loop/condition/update/return
programs peaked at 4/6 tasks (execution-context features, 30
epochs), and is fragile: 100 epochs scores 0/6, losing a task the
30-epoch static model had. Single seed throughout. Every later
stage progressively replaced learner choices with host machinery —
classically mined donor contexts, neural hole filling, component
curricula, visible-query repair, branch pairing with add/sub
enumeration — climbing 6/6, then 7/11, 8/11, and 11/11 on the same
repeatedly reused development set. At 11/11 the neural net chooses
one shared Int expression inside classically composed branches. A
retrospective composition-first classical control also reaches
11/11 in about a ninth of the time. The chain is adaptive
development progress, not a generalization estimate.

### Frozen transfer bank (18 functions)

The strongest non-adaptive test in the repo. Frozen before solver
work, seal and hashes verified, every emitted program actually
executed on 20 held cases in both languages. Trained 11/18,
untrained 11/18, classical 14/18. Full measured CPU: trained
71.614 s (including 42.406 s fresh fit), untrained 26.600 s,
classical 21.127 s. Historical donor-library creation is excluded,
so the trained total is understated. Verdict recorded at the time
and confirmed here: transfer within the grammar is real; benefit
from training is not established. With n=18 and discordant pairs
1-vs-4, McNemar p is about 0.375 — this bank cannot statistically
separate its own arms either.

### Stateful, component, array, campaign, edit-value

Uniformly negative. Stateful frozen learner 0/5, and 0/5 again
after a 1,151-function curriculum (which also lost one retained
utility and was correctly not promoted). Component curriculum 1/5
new with retention dropping to 4/6. List-output direct
construction 0/3 across three recipes including beam 256, equal to
untrained 0/3; it has never worked once. The unattended campaign
ran both recipes at 0/3 and stopped `NO_MEASURED_GAIN` without
opening its 6-function audit. Edit-value ranking tied at 1/6 fresh
functions for learned, untrained, classical and incumbent routes,
0/3 on the originals. Supplied grammar controls pass throughout
and were never donors.

### Sensing

An honest, complete negative, frozen since 2026-10-18. The fan
vibration detector passes every frozen gate in 0/3 seeds (plastic
SNN recall 15/198, 7.58%; MSE 0.070092 vs EMA 0.029432). Live and
continual microphone work shows the adaptive SNN beating only its
own frozen self, losing to ANN/EMA, and paying in probe retention
(+28%). No personal-microphone or independently-logged physical
change-detection result exists.

### Sound Companion

Exact accuracy tie with the classical matcher on generated sounds
(36/36, then 48/48, then 48/48 after reload, 3 seeds), at far
higher cost (fit 6.912 s vs matcher insertion 0.00810 s; decision
2.07 ms vs 0.81 ms). The SNN-chooses-label contract is real and
regression-guarded, but as evidence of neural benefit it is zero.
No fresh personal-recording result exists.

## The decisive comparisons

| Line | Best learned result | Best control | Verdict |
| --- | --- | --- | --- |
| Early arithmetic | ANN 78/135 | streaming 135/135, faster | sealed test only |
| Benchmark ranking | 99/144 at cap | MLP 102; ~17x time | pursue, ANN only |
| Utility composition | CLI 100/100; 5/8 | never run | park |
| Statement direct | 4/6 | untrained 0/6 only | park |
| Repair/branch stack | 11/11 dev set | classical 11/11, faster | pursue classical |
| Frozen transfer | trained 11/18 | classical 14/18 | decisive negative |
| Stateful/component | 0/5; 1/5 | supplied 5/5 | drop variants |
| Array/campaign/edit | 0/3 repeatedly; 1/6 | all arms tied | drop |
| Sensing | recall 15/198 | EMA better, faster | park track |
| Sound Companion | 48/48 | matcher 48/48, cheaper | park track |

Two patterns explain most rows. First, the learner's contribution
shrank over time — full token construction, then condition-hole
filling, then one shared expression — while host grammar, donor
mining, traversal, lowering and plumbing grew. In the branch and
frozen programs the learner fills roughly 1 of 8 process lines;
in the utility CLI, 0 of 43 lines of source. Second, every
comparison that kept the trained arm honest (fresh fit charged,
frozen bank, actual execution for controls) landed at trained
equal to untrained and behind classical.

## Pursue

Only three lines still contain a decisive, pre-registrable test
that could change the verdict. Each entry states the next
experiment, the kill criterion, what success would actually get
us, what failure gets us, and what the line can never deliver.
Run one at a time; each is designed to close a question, not to
open another sweep.

### 1. Classical branch/fragment synthesis, neural arm as ablation

Next experiment: one new frozen bank of at least 48 functions,
frozen before solver work, at least 3 seeds for any learned arm,
identical CPU and oracle budgets per arm, actual Python and
CURLEE execution for controls as well as primaries, and
end-to-end cost including donor-library creation from empty.
Kill criterion: if the trained arm does not beat both untrained
and classical on solves at matched total cost (or match solves
at at least 10% lower total cost), the neural arm leaves this
route permanently and it continues classical-only.

**What it gets us if it works.** Two different prizes, depending
on which arm wins. If classical wins again on a powered bank, we
own a verified synthesis tool: a pipeline that composes qualified
fragments into new Python and CURLEE routines nobody hand-wrote
for those tasks, with actual execution evidence, provenance and
rejection of wrong programs. Its growth mode is a qualified donor
library that widens the set of composable unseen functions; its
honest progress metric is unseen-combination coverage gained per
unit of new host code. That is a real, demonstrable artifact —
the only route in the repo with non-adaptive dual-backend
transfer (classical 14/18). If the neural ablation unexpectedly
wins at matched total cost, we get something rarer: the first
evidence that learned guidance adds value inside the verified
pipeline, which would justify widening the grammar under the
same frozen protocol.

**What it gets us if it fails.** A powered, non-adaptive answer
for this route instead of a suggestive n=18 tie, and a clean
stop on neural-guided composition in this grammar.

**What it never gets us.** By itself, this is not the stated
destination. Pursued to its classical end, the agency stays in
host enumeration, mining and pairing; nothing learns to program,
and nothing improves itself. Each new program family (new types,
nested loops, strings, arrays of records) needs new host grammar
and lowering work, so the ceiling is set by host effort, not by
accumulated learning. Continuing it as the main line is therefore
a scope decision: it trades the RSI premise for a trustworthy
classical synthesizer plus a measurement harness other work could
reuse.

### 2. Generic learned ranking, CPU-matched (explicitly not SNN)

Next experiment: pre-eval-style ranking with MLP/recurrent
rankers, budgets defined in CPU seconds rather than candidate or
signature caps, fit cost included and amortized over a declared
volume, at least 48 fresh function-disjoint tasks, against
streaming, heuristic, random and untrained arms at the same total
CPU. Kill criterion: at equal total CPU the learned ranker solves
no more than streaming/heuristic, or its solve advantage needs
more than 2x streaming's time. A cap-based win that loses on time
is not a win.

**What it gets us if it works.** The one brick whose value grows
with ambition. A ranker does not write programs; it tells a
classical enumerator where to look first. On today's tiny search
spaces that is nearly worthless because brute force is already
trivial (streaming clears 135/135 in a second). But search spaces
explode combinatorially with program size, constants, types and
control flow, and enumeration dies there. A learned prior that
still pays for its own inference at equal CPU is exactly the
component that could keep enumerative synthesis alive at program
sizes where it currently suffocates — the DeepCoder/DreamCoder
role the roadmap cites. Success buys reachable program size, and
the component is reusable inside any future synthesizer, including
the classical route above.

**What it gets us if it fails.** Closure on the repo's oldest
positive-looking signal. Pre-eval v2's cap win (99/144 vs
streaming 60) would be retired as an artifact of the wrong budget
unit, and "learned guidance helps search" would have no remaining
evidence here.

**What it never gets us.** A programmer. Ranking assumes a host
grammar, a host candidate generator and a host checker. It cannot
invent constructs, propose abstractions or write source outside
the enumerator's space. Even a total win leaves us owning a
search accelerator, with program construction, fragment reuse
and self-improvement all still unbuilt.

### 3. One-shot sealed arithmetic closure test

Next experiment: run the preregistered `construction-v1` gates
exactly as written, once, on the sealed 244-function final set —
seeds 0, 1 and 2; trained must beat untrained and random by at
least 10 percentage points at cap 256 per seed; and full coverage
must cost at least 10% less CPU than streaming with fit amortized
over 244. Kill criterion: either gate fails and the
arithmetic-construction learning-plus-cost claim is closed as
falsified, with no adjusted re-runs.

**What it gets us if it works.** Surprise, and a reason to reopen
one closed drawer. A pass would be the first multi-seed,
sealed-set evidence that learned construction beats trivial
enumeration on cost — contradicting every development result —
and would justify a fresh sealed bank and replication. Even then
it proves something narrow: tiny add/sub/mul trees over two
inputs, abundantly supervised by teacher traces.

**What it gets us if it fails.** The expected and more useful
outcome: epistemic closure. The repo's original claim gets a
verdict from its own preregistered protocol instead of indefinite
development-set inference, the 244 functions stop being an excuse
to keep the construction line on life support, and the negative
result becomes cleanly publishable. It is also the cheapest of
the three tests. Recommended sequencing: run this first.

**What it never gets us.** Capability. This test buys an answer,
not a component. It consumes the sealed set one-shot; afterwards
those functions are development data and any future claim needs a
new sealed bank. Passing would not show loops, branches, list
output, transfer or self-improvement.

## Park

Preserved, no new trials, each with its unpark condition and the
endpoint that condition would serve.

- **Statement direct token construction (scalar).** Real 4/6,
  single seed, retention-fragile, superseded by the branch route.
  Unpark only as a multi-seed replication inside a pursue test
  above. Endpoint if it ever held up: the learner (not donors)
  choosing whole loop programs — the closest thing yet to the
  stated objective, which is exactly why it needs seeds before
  any belief.
- **List-output/array capability goal.** The goal is legitimate —
  most useful programs return data, not one integer. Every
  current recipe is dead (see Drop). Unpark only with a
  preregistered new structural hypothesis (for example qualified
  fragment transfer into array bodies), never another parameter
  sweep. Endpoint: the first type expansion, and the test of
  whether the scalar results were about programming or about a
  one-accumulator toy.
- **Utility library-composition.** Works as engineering, proves
  nothing about learning, no classical comparator. Unpark only
  if host fraction becomes a measured, decreasing quantity —
  that decrease, not more solved tasks, would be the progress
  toward a learner that needs less scaffolding over time.
- **Sensing.** Unpark only with independently logged physical
  changes and separate held sessions. Endpoint is its own: a
  calibrated change detector, judged against EMA/ANN — not
  programmer evidence.
- **Sound Companion.** Unpark only with fresh personal recordings
  across stages/sessions/restart with unknown controls. Endpoint
  is its own: a teachable creature whose SNN genuinely beats
  matching on real sounds, or an honest tie kept for the
  owner-mandated SNN-chooses design without capability claims.
- **Original RSI/SearchPolicy outer loop, ranker and gains
  families.** Not tallied by this audit — parked for lack of
  evidence, not on a verdict. `docs/gains/gains.csv` is dropped
  as evidence per its own README ("not comparable units"). No
  claim either way until someone tallies this family against its
  artifacts.

## Drop

- **SNN-specific advantage claims everywhere** (programmer
  ranking, sensing detector, sound). Tested repeatedly, with
  multiple seeds where it counted; lost or tied to ANN, EMA, the
  matcher and classical search every time. There is no remaining
  hypothesis, only sunk architecture preference.
- **Online persistent/plastic-encoder line.** Persistence degrades
  across phases, all gates fail in 0/6 orders, candidate savings
  never convert to CPU wins.
- **Current array direct-construction recipes, campaign parameter
  recipes, and edit-value ranking as implemented.** 0/3 across
  six-plus attempts, a tied 1/6, `NO_MEASURED_GAIN`, and the
  result doc's own instruction not to continue parameter sweeps.
- **Neural hole-filling/context-scoring as a cost or solve-rate
  claim.** Classical BFS fills the same holes 50–88x faster;
  neural context ordering was worse than the order it replaced.
- **Stateful frozen-learner transfer and the broader-curriculum
  variants tried.** 0/5 twice; the second attempt also damaged
  retention.

## So what are we building towards?

The stated destination is a learner that constructs useful Python
and CURLEE programs from specifications, reuses its prior
solutions on new tasks, and only then earns the right to attempt
verified self-improvement. Measured against that destination, the
three pursue lines are not three roads to it. They are triage on
three candidate bricks, each currently failing:

1. **A trustworthy synthesis substrate** (pursue 1) — generation
   inside a sealed, checked, dual-executed pipeline with a
   growing library of qualified fragments. This brick exists and
   works, but today its intelligence is classical host code.
2. **Learned search guidance** (pursue 2) — a prior that makes
   enumeration tractable as program spaces grow. Evidence exists
   only in the wrong budget unit.
3. **Learned direct construction** (pursue 3, and the parked
   statement line) — the learner choosing program structure
   itself. Evidence is a fragile 4/6 and a sealed verdict never
   taken.

The end-state these bricks could compose into is concrete: a
system that proposes programs from a learned prior, disposes of
them by independent execution, accumulates the survivors as
qualified fragments, and needs progressively less host assistance
per new task family. That last clause is the metric that would
show we are facing the right way — host fraction falling while
unseen-family coverage rises. The current trajectory is the
opposite: coverage rose along the dev chain only as host fraction
rose (from full token construction to one chosen expression), and
when host help was held constant on a frozen bank, trained equalled
untrained. Nothing in the data shows a flywheel starting to turn.

Honest endpoints by scenario:

- **All three tests fail.** The project has established that this
  approach — small ANN/SNN controllers imitating teacher traces
  on tiny grammars — does not lead toward the destination. What
  remains is still worth having: a rigorous negative result, a
  verified classical composition pipeline, and a measurement
  harness (sealing, donor qualification, dual-backend checking)
  that any successor approach should inherit. Continuing toward
  the destination after that requires a new hypothesis, not more
  trials — for example learning from execution feedback rather
  than teacher imitation, or a substantially different scale or
  representation. That is an owner decision, and under the
  current rules so is any pretrained-model route.
- **Ranking wins CPU-matched.** We own the accelerator brick. The
  rational next building phase is enumerative synthesis at
  program sizes currently unreachable, with the ranker inside —
  still a host-grammar system, but one where learning demonstrably
  buys reach.
- **The neural ablation wins inside the classical route.** The
  first evidence that learning adds value where it matters
  (unseen functions, real backends, total cost). That would
  justify grammar widening under the frozen protocol and reopen
  direct-construction work with seeds.
- **Only the classical arm wins.** We own a verified synthesizer,
  not a learner. Worth keeping as tooling and as the control
  every future learned arm must beat — but calling it progress
  toward RSI would be a relabelling, and the charters would need
  to change to say so.

In one sentence: the pursue list does not build the destination;
it finds out, with three cheap decisive tests, whether any load-
bearing part of the current path to it is real — before another
hour is spent on trial infrastructure for lines the data has
already closed.

## Fixes before any new claim

Methodology. Keep: seal+hash+fingerprint protocol, `qualify()`
donor gate, dual-backend acceptance with wrong-expected controls,
actual-case counting, retention of failures and classical wins.
Fix: at least 3 seeds on the programmer chain (seed variance is
currently unmeasured); frozen banks large enough to resolve the
comparison; identical search and oracle-call budgets per arm
(mid-chain trials gave neural two searches vs one, and 312 vs 33
oracle calls); actual target-language execution for controls, not
IR-only; no retrospective control strengthening on the evaluation
set; one fixed held set instead of freshly seeded held arrays per
trial; a stated stronger separation than "differs on 24 probes";
and no curricula that embed the target components while calling
recombination transfer.

Cost accounting. No end-to-end number exists in the repo. Every
future comparison must include, from empty: donor-library
creation, fits, mining and qualification, oracle labelling,
emission, backend execution and candidate evaluation. Budgets in
CPU seconds, not candidate/signature caps. Repeat timings — most
recorded figures are single observations on a shared machine.
Retire hypothetical volume-1000 amortization as evidence.

Substrate. Correct `AGENTS.md:22-24`: `probe.validate_graph`
(`rsi_programmer/probe.py:335`) is purely structural and takes no
rows; name the actual semantic checkers (there are at least
five). Correct `AGENTS.md:18-21`: lineage is the only sqlite
store, not the only persistence, and its schema carries
authority/fence terms. Consolidate to one semantic acceptance
gate and one IR stack; freeze new per-trial policy/search module
families (53 modules have at most one non-test importer; in the
last 40 commits manifest/justfile/doc churn exceeded any source
module). Refresh or date-stamp the root README (stops at 11/11;
no frozen, array, campaign or edit-value results), the
construction-v1 "remain outstanding" text, roadmap-as-future,
and the six inconsistent test counts (147/270/285/530/541 vs 581
current fast test definitions). Resolve tracked
`weights/rsi-search-policy-w1.pt`/`w2.pt` against the never-commit
rule. Preserve as substrate: the probe IR core (minus hard-coded
TASK/run), sandbox runner, statement interpreter,
validate-then-emit, track manifest harness, lineage digest
read-back.

## Coverage gaps and unverified items

- Read-only audit: HEAD test status, fingerprint reproduction
  and held-disjointness recomputation are unverified. Last
  recorded track passes are 2026-11-01 receipts (131 sensing,
  87 companion); remote CI was pending at every checkpoint.
- Sealed 244 non-use rests on self-reported flags
  (`original_arithmetic_final_opened: false`) and one runner
  split-guard; no seal/access-log artifact exists. Weights are
  mostly git-excluded and hash-referenced only (except the
  tracked w1/w2 above).
- The CURLEE binary (sha `412d0e6e…`, constant across artifacts)
  and its local-only repo were never inspected; mid-chain CURLEE
  case counts rest on single report JSONs re-quoted across docs.
  CURLEE output is always a bounded routine, never file I/O.
- Single-source tallies: all pre-November benchmark numbers,
  sensing/sound detail beyond spot-checked headlines, host
  fractions (4 generated files line-counted), LOC figures
  (65,992 total Python lines; 27,071 in `rsi_programmer/`).
- Coverage gap: the original RSI/SearchPolicy outer-loop family
  (`evolve`/`scale`/`headless`/promotion, 13 `rsi_*.json`
  fixtures, `ranker-20261017`, gains v12–v29), most of
  `architecture.md` (1,409 lines) and `programmingplan.md`
  (739 lines) beyond their heads, `docs/audit-*` bodies, the
  benchmark protocol documents, and roughly 30 checkpoint
  artifact directories classified by name only. Park verdicts
  on those lines mean "no audited tally", not "examined and
  failed".
- Open items never resolved in-repo: sensing 30-minute
  prospective comparison (October present-tense text, no
  follow-up), the campaign's unopened 6-function audit, the
  bounded-benchmark third seed ("still running", issue #81 per
  its checkpoint README), personal-microphone accuracy, and any
  hardware/energy measurement.

## Recommended sequence

1. Run the sealed arithmetic closure test (cheapest, decisive,
   preregistered).
2. Run the CPU-matched ranking test (the only positive-looking
   signal, in the right budget unit this time).
3. Only then build the new 48-function frozen bank for the
   composition route — its design should inherit whatever 1 and
   2 teach about budgets and seeds.
4. Apply the substrate corrections (AGENTS.md, README, test
   counts, tracked weights) alongside, since future readers will
   trust those files first.
5. Stop everything in Drop now; keep Park code runnable but
   untouched. If all three tests fail, the next move is an owner
   decision on a new hypothesis, not another trial.

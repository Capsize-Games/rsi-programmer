# Hypothesis survey after the execution-feedback closure (2026-11-06)

State of play after the sealed execution-feedback result
([result](feedback-construction-results.md)). Every decisive test
the 2026-11-05 audit designed has now been run, plus the
owner-directed execution-feedback test. All closed against the
neural arm on matched cost:

| Line | Sealed/powered result | Status |
| --- | --- | --- |
| Construction-v1 imitation (audit #3) | Learning gate pass (94-119/244 vs ~0), cost gate fail (streaming 244/244) | Closed |
| CPU-matched ranking (audit #2) | Trained = untrained; streaming ~3.7x cheaper | Closed in substance |
| Fragment transfer, neural arm (audit #1) | Frozen v2: classical 38/54 vs trained 27-30 (McNemar p 0.001-0.022) | Neural arm killed; route classical-only |
| Execution feedback / expert iteration | Feedback 18.7 vs imitation 28.7 mean solves; ~47x equal-CPU deficit; rollout acceptance ~1/1,200 | Closed per frozen criterion |
| SNN advantage, online, array, edit-value, campaign | Repeated ties/zeros (audit Drop list) | Dropped, not reopened |

## The three measured failure modes

Any new hypothesis must state which of these it addresses and how:

1. **Sparse self-generated signal.** On minimal-size strata 6-8
   the trained controller accepts ~1 in 1,200 of its own training
   rollouts. Self-improvement loops starve at this difficulty.
2. **Fit cost vs enumeration economics.** One imitation fit costs
   ~590 CPU s; learned eval is cheap (0.086 s/task) but solves
   9.6%, while streaming eval is 0.521 s/task and solves 21.1%.
   At the volumes tested so far (54-300 tasks), fit cost dominates
   and streaming wins at equal total CPU by large factors.
3. **Imitation plateaus below classical coverage.** Training beats
   untrained decisively (0/300) — the signal is real — but solves
   concentrate in a subset (hard bank: stratum 8 only), and extra
   search budget does not help (construction-v1: the beam plateaus;
   the stop action ends search).

## Candidate hypotheses not yet falsified

### A. Amortization at deployment volume (addresses failure mode 2)

Every cost verdict so far amortized fit over 54-300 tasks. The
learned arm's marginal economics are better than streaming's
(0.096 solves / 0.086 s = 1.12 solves per marginal CPU second vs
0.211 / 0.521 = 0.41), so a break-even volume exists if rates
hold. Retrospective arithmetic on the sealed hard-bank report
(hypothesis generation, not evidence): break-even at ~928 total
CPU seconds, i.e. ~3,900 tasks for the imitation arm, with
streaming at its 0.5 s cap (a cap sweep over the report's per-row
times shows 0.5 s is near-optimal for streaming CPU-per-solve
among caps ≤ 0.5 s: 2.21 s/solve vs 2.29 at 0.4 s).

Test shape: fresh sealed bank at a preregistered deployment
volume (order 6,000 tasks), fresh fit per seed charged once, both
arms at operating points frozen from development data, verdict =
solves at equal total CPU. Kill criterion: if the learned arm
does not win at that volume, the amortization defense closes
permanently — no larger-volume retest.

Principal risk: the stratum-8 enumeration-prefix caveat. Learned
solves concentrate exactly where the corpus metadata says
sampling may be artifactually easy. A fresh bank must sample
identities without prefix bias, and diagnostic D0-b (below) must
first show the learned stratum-8 rate survives non-prefix
sampling on development data. If it does not, hypothesis A's
premise is void and the bank is not built.

### B. Curriculum expert iteration (addresses failure mode 1)

The closed feedback test started rollouts at strata 6-8, where
acceptance is ~0. Expert iteration from strata where acceptance
is non-trivial, expanding difficulty as acceptance grows, is a
different mechanism, not a parameter tweak of the closed test.
Key unknown: the acceptance-by-stratum curve (diagnostic D0-a).
If acceptance is ~0 at every stratum above trivial sizes, B is
dead before it is built. If a dense-signal regime exists, B is
preregistered with its own kill criterion: curriculum feedback
must beat matched-epoch imitation on a sealed bank at matched
total CPU, else feedback in all forms closes.

### C. Execution-guided decoding (addresses failure mode 3)

At inference, execute partial programs on the observed rows and
use the partial-match structure to rerank or prune the beam,
instead of ranking on policy scores alone. The array-track
diagnosis found witness paths pruned by token/beam ranking,
which is the failure this targets. Key unknown: teacher-forced
rank of the correct next action under the trained policy on
development tasks (diagnostic D0-c). If correct actions are
typically outside any plausible beam, decoding changes cannot
rescue them and C is unpromising. The classical control for C is
the same execution-guided search driven by an untrained policy,
so the ANN's contribution stays isolated.

### Not candidates

- **Larger architecture / more capacity.** Final training loss
  does not rank seeds by solves (construction-v1), and nothing
  measured identifies capacity as the binding constraint. Adding
  capacity also worsens failure mode 2. Parked unless a
  diagnostic shows representation-limited errors.
- **Reopening closed lines** (ranking, neural fragment arm,
  array variants, edit-value, SNN): no new evidence; the audit
  Drop verdicts stand.
- **Pretrained/LLM teacher**: outside research scope without an
  explicit owner decision.

## Subgoal D0: diagnostics on development data

One fresh imitation fit (seed-0 recipe, hard-corpus train pool,
~600 CPU s, weights saved this time) supports three measurements.
All on development data; nothing sealed is consumed.

- **D0-a Acceptance by stratum.** Rollouts on the train pool,
  acceptance reported per stratum 6/7/8 (the sealed run recorded
  only the 1/1,200 aggregate). Decides whether B has a regime to
  start from within this corpus; if all ~0, B additionally needs
  low-stratum tasks and is re-scoped or dropped.
- **D0-b Non-prefix stratum 8.** Generate ~100 stratum-8
  identities by non-prefix sampling (seeded reservoir over the
  catalog stream), disjoint from the corpus; measure fitted-
  controller and streaming solve rates. If the learned rate
  collapses relative to the sealed 23.3/100 stratum-8 mean,
  hypothesis A's premise is void.
- **D0-c Teacher-forced action rank.** Along teacher trajectories
  for development tasks, the rank of the correct next action
  under the fitted policy. Sizes the headroom for C.

D0 also sweeps streaming caps (0.25/0.5/1/2 s) on development
tasks to freeze hypothesis A's control operating point.

## Sequencing

1. D0 diagnostics (development data, serialized).
2. Preregister A if D0-b sustains its premise; preregister B if
   D0-a finds a dense-signal regime; scope C from D0-c.
3. Each preregistered subgoal gets its own design doc, frozen
   kill criterion, fresh sealed bank, kanban cards and a
   serialized decisive run — the same discipline as the closed
   tests.

If D0 voids A and B and shows no headroom for C, that is itself
the finding: the remaining plausible space at accessible scale
is exhausted, and the goal's terminal verdict (not achievable
here, with evidence) is reported instead of further trials.

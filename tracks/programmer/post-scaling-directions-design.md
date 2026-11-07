# Post-scaling directions: architecture (2026-11-06)

Context: the sealed data-scaling test failed its frozen kill
criterion ([result](scaling-sealed-results.md)); every
preregistered ANN-construction route at accessible scale is
closed. External literature (reviewed 2026-11-06) suggests four
directions that change a different variable than the closed
tests did. This document architects them as separately
preregistered subgoals with sequencing and kill criteria. It
authorizes no run by itself; each direction starts with a
development diagnostic and only proceeds to a sealed test if
its diagnostic clears the stated bar.

Measured failure modes referenced below (from the D0-D2
diagnostics and sealed results): (1) sparse self-generated
signal in top-down construction; (2) fit cost vs enumeration
economics; (3) solves lost to beam truncation; (4) shape-level
generalization growing with training-set size but not enough
at the declared volume.

## Direction 1: distribution-based search (Heap Search swap)

- **Hypothesis.** The beam decoder wastes the policy's
  distribution. Best-first enumeration ordered by program
  probability (Heap Search; Fijalkow et al., AAAI) extracts
  more solves per CPU second from the *same* trained model.
- **Addresses:** failure mode 3, then 2.
- **Architecture.** A new search module that enumerates
  expression trees in decreasing policy-probability order
  (priority queue over partial trees, probability = product of
  action probabilities under the existing `ConstructionPolicy`)
  with the frozen execution backstop, checked by the existing
  independent checker. No training change.
- **Diagnostic (dev data only).** Reproduce the D1 fits
  (1,200/4,800-task recipes, weights saved this time) and
  compare beam w8/w32 vs heap search on the D1/D2 300-task
  broad dev eval: solves per eval CPU second per arm.
- **Proceed bar.** Heap improves solves/CPU-s by >= 25% over
  the better beam arm on dev data; otherwise the decoder
  hypothesis is recorded as closed without a sealed run.
- **Sealed test (if bar clears).** Fresh shape-broad bank,
  frozen recipe from the diagnostic, verdict = credited solves
  at equal total CPU vs streaming at a declared volume, kill
  criterion symmetric to the scaling test.

## Direction 2: corpus-guided library learning (Stitch-style)

- **Hypothesis.** Checker-verified programs already produced
  by this project share reusable substructure. Corpus-guided
  top-down library synthesis (Stitch; Bowers et al.) extracts
  it orders of magnitude faster than DreamCoder-style
  compression, shrinking effective search depth for all arms.
- **Addresses:** search depth (modes 2/3 indirectly). This is
  a classical direction; any neural contribution is downstream.
- **Architecture.** Assemble the verified corpus (accepted
  programs from sealed/dev artifacts, provenance recorded per
  program). Evaluate an external-tool adoption (license and
  integration cost checked first) against a minimal in-repo
  top-down compression implementation; the comparison itself
  is a deliverable. Learned abstractions become new catalog
  primitives; measure held-task solve-rate and search-cost
  deltas with and without the library, classical arms first.
- **Diagnostic.** Compression ratio and abstraction reuse on
  the verified corpus; then D1-style broad dev eval with the
  library added to the grammar (streaming + learned arms).
- **Proceed bar.** Dev solves at matched CPU improve >= 15%
  for at least one arm, with the library's own construction
  cost charged; otherwise recorded as "insufficient repeated
  structure" (AbstractBeam's stated limitation) and closed.
- **Sealed test (if bar clears).** Fresh bank; library frozen
  before the run; library construction cost charged to every
  arm that uses it.

## Direction 3: bottom-up on-policy combiner (CrossBeam-style)

- **Hypothesis.** The failure was the training signal, not
  learning itself. A policy that chooses how to combine
  enumerated subprograms, conditioned on partial executions
  and search history and trained on-policy from its own
  searches (CrossBeam; ICLR 2022), receives dense signal on
  every search instead of binary trajectory acceptance.
- **Addresses:** failure mode 1 directly.
- **Architecture.** Bottom-up enumerative backbone over the
  existing catalog (subprograms keyed by execution signature
  on observed rows); a combiner policy scoring candidate
  combinations from subprogram execution vectors + search
  history; on-policy data extraction from training searches
  (correct combinations as positives); independent checker
  acceptance unchanged. New modules; catalog/checker reused.
- **Diagnostic.** On dev pools: acceptance density of on-policy
  training data (vs the ~1/1,200 trajectory baseline), then
  broad dev eval solves/CPU vs the D1 imitation arms.
- **Proceed bar.** Dev solves/CPU beats the matched-data
  imitation arm; otherwise closed as the second and final
  feedback-form test.
- **Sealed test (if bar clears).** Symmetric to Direction 1's:
  fresh bank, declared volume, credited solves at equal total
  CPU with all training/search costs charged.

## Direction 4: solver guidance outside programming (owner-gated)

- **Hypothesis.** The ANN's demonstrated strength (a learned
  prior that beats untrained decisively) pays where the host
  search space is too large to enumerate naively: guidance
  inside a mature solver (e.g. GNN-guided SAT branching;
  RLAF, 2025) rather than program authorship.
- **Status.** Domain change. Requires an explicit owner
  decision on target domain and solver before any design
  work. No diagnostic is authorized by this document.
- **Architecture sketch (for the decision).** Pick one solver
  with a measurable branching heuristic and a benchmark suite
  with headroom; train a guidance model from solver feedback;
  verdict = solve-time at matched total CPU vs the solver's
  native heuristic, preregistered identically to prior tests.

## Sequencing and shared rules

1. Direction 1 first (cheapest; reuses existing fits/banks).
2. Direction 2 in parallel (independent; classical).
3. Direction 3 only after Direction 1's diagnostic reports,
   so decoder and training-signal effects are not confounded.
4. Direction 4 waits for the owner.
5. Every direction: development diagnostic first with the
   proceed bar above; sealed claims only from a fresh bank
   generated after preregistration; fit/search/library costs
   always charged; failures retained and documented; tests
   registered in `tracks/manifest.json` and the justfile lane;
   timed runs serialized under the builder lock.
6. A direction that fails its diagnostic bar is closed in its
   result note without a sealed run. Two or more sealed
   failures among Directions 1-3 re-establish the exhaustion
   verdict for the construction line as a whole.

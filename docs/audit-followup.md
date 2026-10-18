# Audit follow-up

## code-v47: accounting, objectives, and supervision

`just test-fast`: **Ran 258 tests in 223.871s; OK**, on an isolated tree over
`d8f3877`. No fast tests were excluded. The 102 runtime/test/fixture
inputs have SHA-256 `d45d2fac4964c4d2415f851c11c38db226962551629ea7b3bebdb525ff19f0af`.
The slow and retry lanes on these inputs and the paired author/beam
measurement are separate pending checks at this checkpoint. Earlier
code-v45 slow/retry results are not counted as results for this source.

- **#53:** `enumerate_programs_report` exposes signature work, retained
  root-eligible examinations, and the raw match ordinal separately.
  Boolean comparisons count as work without being Int-root checks.
  The tuple API is retained. Guided search has an optional signature-work
  cap, independent of its scan budget, and preserves the full-layer
  feature denominator when a cap truncates a layer. Historical gains
  figures are explicitly archived with incompatible old counter units.
  Eager layers, neural work, and graph-validation costs remain; no runtime
  or signature-saving claim is made from ranked examination counts.
- **#54 follow-up:** guided search can retain multiple lower-layer
  equivalents under a cap. It reports retained/dropped alternatives and
  truncation, and defaults to a 10,000-signature cap in that mode.
  A diagonal-training counterexample now recovers `(x+y)^2` on held rows
  when two representatives are kept. This is bounded recovery, not a
  completeness claim. The generator fix was already published in code-v46.
- **#56:** mixed-class lattice heads use the attainable one-spike hinge
  `mean(max(0, 1/steps - (positive_rate - negative_rate)))`.
  Its floor is zero and separated pairs stop supplying saturation
  gradients. Constant targets still use squared error. Six aggregate
  head losses retain mean-plus-lambda-times-max; EMIT fits membership
  and STOP its complement, decoded by their calibrated rate difference.
- **#65:** modular reports measure positive-versus-negative preference
  for the five actual row scores used by the objective, with ties worth
  one half and eligible-group counts. Opcode argmax and left-signature
  top-1 remain explicitly diagnostic, with deprecated compatibility
  aliases. These heads are described as candidate-row scorers.
- **#66:** author, modular, lattice, and rich-negative labels use backward
  reachability from every within-bound target in a retained train-only
  signature graph, with positives indexed by tree size. A capped partial
  graph supplies no negatives. Skipped tasks and evidence are reported;
  all four shadow tasks fit within the 50,000-candidate cap. Canonical
  labels remain an explicit comparison mode. The training CLI records
  the actual label mode, grouped feature-label count, and evidence.

The first completed integration fast run failed one existing lattice
accuracy check: bit 0 and word-exact accuracy were 0.208. Its random
whole-group split calibrated only on add and evaluated only on sub,
leaving no positive bit-0 calibration examples. Whole-group opcode
stratification now preserves train/calibration/evaluation support when
possible and reports incomplete sparse support. Threshold calibration
covers every discrete firing-rate cut, including always-on/off, using
only calibration data. The accuracy assertion was retained. The targeted
training check then reported four bit accuracies of 1.0 and word-exact
1.0 on **313 evaluation rows**, with **2,265 calibration rows**. The prior
random split used 2,229 evaluation rows; these are different evaluation
sets, so those accuracy values are not a controlled quality comparison.

The first integration attempt was interrupted after review found a
signature-cap denominator regression. A second, already superseded gate
was interrupted after the completed run exposed the lattice failure.
Neither interrupted run is a pass. The final complete fast gate includes
regressions for those corrections and measurement provenance guards.

[Passing fast log](../artifacts/followup-integration-20261018/fast.log),
[original complete failure](../artifacts/followup-integration-20261018/prior-fast-failure.log),
[denominator red regression](../artifacts/followup-integration-20261018/signature-budget-red.log),
[first interrupted attempt](../artifacts/followup-integration-20261018/interrupted-first.log),
[superseded interrupted attempt](../artifacts/followup-integration-20261018/interrupted-superseded.log),
and [source/log manifest](../artifacts/followup-integration-20261018/manifest.json).

#70 remains pending the real paired experiment. Its harness prespecifies
three seeds, canonical/reachability supervision, five neural/bonus
settings, both greedy author and beam, and all four shadow plus three
novel tasks: 420 result rows. It independently checks train and held
outputs and rejects source, fixture, or SpikeForge mutations. Author
and beam retain different budget units; no equal-work claim is made.

## code-v46: independent task acceptance (#54)

New task generation accepts the generating expression's own train and
held-out semantics. It never asks the evaluated enumerator to solve the
task in the default mode. Each task records its generating program as
`generator_witness`, its selection filters, and whether search filtering
was applied. Families record `generation_acceptance`.

`search_filter=True`, or `scripts/make_tasks.py --legacy-search-filter`,
explicitly restores the old enumerator-and-held-confirmation selection
rule. Existing frozen fixtures were not regenerated. Their historical
selection bias remains relevant to measurements made on them.

The ordinary search still merges lower-layer train-equivalent programs.
It remains incomplete for held-out-correct alternatives, as documented
in architecture Sections 2.3 and 3. This meets #54's acceptance option
of documenting incompleteness; it does not establish complete search.
Generation also still filters nonconstant/nonidentity targets and
requested features, so these are selected synthetic tasks.

Validation on an isolated tree containing only this patch over
`239c892`: **Ran 226 tests in 217.545s; OK** (`just test-fast`). All fast tests ran,
including four new generator/CLI checks and the historical fixture
regeneration check. No new slow or retry lane was run for this
generator-only change. The tests verify default acceptance even when
calling the enumerator would raise, witness agreement on train and held
rows, legacy opt-in, deterministic families, and CLI forwarding.

The first fast attempt failed one historical-regeneration test because
it still assumed the old default. That test now explicitly requests
legacy filtering and compares all historical fields, excluding only
additive witness/selection metadata. The original failure is preserved.

[Passing fast log](../artifacts/followup-generator-20261018/fast.log),
[first failure](../artifacts/followup-generator-20261018/first-fast-failure.log),
[focused checks](../artifacts/followup-generator-20261018/focused.log),
[red regression](../artifacts/followup-generator-20261018/red.log), and
[source/log manifest](../artifacts/followup-generator-20261018/manifest.json).
Logs replace local paths and remove trailing whitespace; test values
are unchanged. The manifest binds the passing run to 97 source inputs
with SHA-256 `57497514ca691d6c49afe10a7cd87ea8f2918c69345dd1ea9f17f9579c7f79aa`.

At the code-v46 checkpoint, #53, #56, #65, #66, and #70 were
being handled separately. Their newer
changes and measurements are not included in this generator checkpoint.
The [code-v45 report](audit-amendments.md) retains the preceding full
validation and three-seed rank measurement at that release.

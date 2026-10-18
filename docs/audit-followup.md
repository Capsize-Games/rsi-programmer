# Audit follow-up

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

#53, #56, #65, #66, and #70 are being handled separately. Their newer
changes and measurements are not included in this generator checkpoint.
The [code-v45 report](audit-amendments.md) retains the preceding full
validation and three-seed rank measurement at that release.

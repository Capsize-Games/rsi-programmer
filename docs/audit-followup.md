# Audit follow-up

## code-v53: paginated asset lookup and fresh measurement audit

The restarted slow lane immediately reported a hub-verification failure.
The isolated original test reproduced it: **1 test in 18.385s, one
failure** (`github=FAIL`). The weights asset was on release 52, beyond
`gh release list --limit 50`. Lookup now paginates all release records
and asset names in one API traversal, preserving newest matching-asset
selection and failing on API or malformed-data errors. Byte comparison,
corruption detection and the original live assertions are unchanged.

Four lookup regressions plus both original live checks passed:
**6 tests in 3.015s; OK**. The fresh complete `just test-fast` passed:
**Ran 270 tests in 222.641s; OK**, with no exclusions. Its 105 input files have
SHA-256 `476d213151b3998477739267dee834bf4ac604cae8e41c81870fdd7cba3c8e73` before/after. The prior red slow attempt was
interrupted after **73.898s** once the failure was visible; it is not a
complete lane or a pass, and retry did not run. The new complete
slow/retry lanes are queued sequentially on the passing source.
#66 stays open pending those integration results.

Fresh measurements on the repaired rich runtime completed before that
unrelated hub lookup amendment: **420 author/beam rows in 72.235s stage
wall** and **180 ranker rows in 13.228s stage wall**. The native author
artifact records 71.254s measured wall; the stage timer also covers
startup and work outside the native timer. These measurements retain their original 104-input hash
`9fc2dbdc1c651ddbbfdc3e88677741d2760dc1789867dc636046f613b2742ef9`.
Every measured implementation file, both measurement scripts and all
fixtures are byte-identical in the new 105-input gate. The manifests
record these separate scopes rather than assigning the gate hash to
an earlier measurement.

Independent data audits find complete coverage, source/fixture hashes,
paired rows, score variants and summaries consistent. Author outcomes,
hit counts, work counters, weights and verdicts reproduce code-v48.
Full versus no-neural remains **29 wins, 3 losses, 52 ties** in 84
paired settings. Reachability versus canonical still changes shadow
author 6/12 to 9/12, shadow beam 8/12 to 10/12, novel author 0/9 to
0/9, and novel beam 6/9 to 1/9. Modes are separately fitted; the novel
regression remains. Seventy of 84 named-bonus pairs have zero fitted
weights. The gap bonus additionally supplies a common within-layer
offset, so its author/beam ablation has no candidate-specific signal.

The measurement script executes train and held-out checks when each
graph is available. Raw rows serialize hit counts and outcome bits,
without graph bodies; the later independent audit verifies acceptance
arithmetic and does not re-execute all returned programs. Conditional
neural-ordering effects are supported on these selected tasks. A unique
benefit from spiking dynamics and broad transfer remain unestablished.

Epoch-50 ranker means also reproduce: trained/probe/random positions
are **17.25/640.75/912.60 shadow**, **390.11/828/521.43 novel**, and
**831.125/482.5/830.29 plain OOD**. All four checkpoints and three seeds
are retained. Guided root-candidate signature counts remain invariant
across four rankers: **46,836 versus 15,297** streaming early-stop calls.
Family-total ratios are **3.209x, 2.371x and 3.325x** respectively.
These counts exclude operand/recursive/neural/validation work and do
not establish elapsed-time gains. External before/after observations
bracket ranker measurement with unchanged clean SpikeForge sources;
that dependency evidence is separate from its native manifest.

[Hub fix logs and manifest](../artifacts/followup-hub-fix-20261018/manifest.json),
[raw author data](../artifacts/followup-measurement-repaired-20261018/author-ablation-v1.json),
[paired analysis](../artifacts/followup-measurement-repaired-20261018/author-analysis.json),
[author integrity audit](../artifacts/followup-measurement-repaired-20261018/author-integrity-audit.json),
[ranker integrity audit](../artifacts/followup-measurement-repaired-20261018/ranker-integrity-audit.json),
and [measurement/source manifest](../artifacts/followup-measurement-repaired-20261018/manifest.json).

## code-v52: rich-surrogate caller repair; fast verified

Capped reachability evidence no longer produces positive-only rich
training groups or one-item selection lists. `family_pairs` omits
truncated, unsolved, or undersized tasks and records each reason.
For a complete pool of N >= 2 negatives, selection takes
`min(750, max(1, N // 2))` and training takes the disjoint remainder.
The real 1,500-pair pool still splits 750/750. Empty or singleton
selection raises an error; scale preparation also requires actual
mixed-label hard and rich fitting groups. The old two-value `hard_neg`
API remains compatible, with a separate evidence-reporting helper.

Scale reports include per-task reachability caps, completion status,
materialized known-negative counts, skipped reasons, and fitting and
selection group counts. Negative counts describe the materialized pool
capped at 1,500, not every possible eligible negative. Three hard
tasks remain capped and omitted from surrogate fitting and selection;
only `sub_square_diff` supplies complete hard evidence. Consequently
`hard_benefit` and rich fitting have one-task coverage. Search acceptance
still evaluates all four shadow and all four hard tasks.

The original real-fixture fidelity assertion was retained and passed
with the seven new regressions: **8 tests in 27.750s; OK** on the
repaired runtime. A full fast attempt then found an older held-failure
test mock supplying a singleton selection: **266 tests in 221.283s,
one error**. That fixture now contains a negative and a positive;
its held-failure correctness and recovery assertions are unchanged.
The affected modules passed **22 tests in 0.439s** after that adjustment.

The fresh complete `just test-fast` passed: **Ran 266 tests in 222.071s; OK**.
No fast tests were excluded. All 104 runtime/test/fixture inputs have
SHA-256 `9fc2dbdc1c651ddbbfdc3e88677741d2760dc1789867dc636046f613b2742ef9` before and after the run. Measurements and complete
slow/retry lanes are being rerun sequentially on this exact frozen
source. #66 stays open pending those integration gates. Code-v51's
complete failed slow run remains published as failed evidence.

[Passing fast log](../artifacts/followup-rich-fix-20261018/fast.log),
[first fast failure](../artifacts/followup-rich-fix-20261018/prior-fast-failure.log),
[affected-module checks](../artifacts/followup-rich-fix-20261018/focused-held-failure.log),
[original fidelity regression](../artifacts/followup-rich-fix-20261018/focused-real-fidelity.log),
and [source/log manifest](../artifacts/followup-rich-fix-20261018/manifest.json).

## code-v51: complete slow-lane failure; #66 reopened

The complete affected slow lane finished **64 tests in 3502.573s**,
with **63 passes and one failure**. No slow test was excluded. The
pipeline stopped on that failure and **retry did not run**. Source
before/after remained the same 102-file hash as code-v48 through
code-v50: `6111b2f0ed9cd031c668726350afbe0ece077b9b9ea0409a5e7e6ee5bf4e3a5c`.
This checkpoint preserves the red result and does not include a fix.

`test_rsi_scale.ScaleUpTests.test_surrogate_fidelity_recorded` failed
its original `all(len(g) > 1 for g in self.groups)` assertion. The
rich negative builder correctly refuses unknown labels when the
full-bound reachability graph is capped. Its caller still created a
positive-only training group and a one-item selection list for each
such task. Training ignores these groups, but selection credited the
only item as a perfect result, inflating `hard_benefit`.

The concrete probe found three truncated tasks: `deep_shift_square`
(bound 6), `mixed_depth7` and `distractor_cube` (bound 7). Each reachability probe stopped
with 46,821 candidate evaluations under the 50,000 cap and supplied
no negatives. `sub_square_diff` (bound 5) completed at 46,821 and
supplied 1,500 proven negatives, split 750/750. Thus this is a missed
caller contract, not a reason to relabel unknown candidates negative
or remove the existing fidelity assertion.

#66 is reopened. The pending repair omits capped or insufficient
groups from both training and selection, carries per-task evidence
into scale reports, and rejects vacuous selection. Focused checks
and a new complete fast/slow/retry sequence must verify the repair.
Earlier fast and measurement evidence remains tied to its original
source; it does not turn this failed integration run green.

[Complete failure log](../artifacts/followup-slow-failure-20261018/slow.log),
[diagnostic output](../artifacts/followup-slow-failure-20261018/diagnostic.log),
[reproduction script](../artifacts/followup-slow-failure-20261018/diagnose_hard_surrogates.py),
and [source/command/log manifest](../artifacts/followup-slow-failure-20261018/manifest.json).

## code-v50: why the gap ablation is inert in author/beam

Code review explains the code-v48 gap result beyond the five zero
fitted weights. `author._score_order` computes one gap and node count
for a whole layer, then `author_scores` adds those same weighted
values to every candidate score. These two extras supply a common
offset, with no candidate-specific ordering information. Beam stashes
are bucketed by size, and each pass recreates the buckets, so their
backtracking does not compare offsets across layers or passes either.

Gap itself need not stay at one: partial target matches in committed
signatures can lower it. Its uniform use within a compared candidate
set is what makes this additive author/beam ablation structurally
inert. Thus the 84/84 unchanged gap outcomes are expected from the
implemented score comparisons, including the one fit with a nonzero
gap weight. They do not estimate the effect of a candidate-varying gap
feature. The exact-flag and conditional neural comparisons remain as
reported. The modular network's nonlinear gap input is a different use.

Evidence at this source: `rsi_programmer/author.py` lines 27-43 define
gap and additive scoring; lines 376-386 share the gap/node count;
lines 505-515 compare one size bucket; lines 568-573 recreate and fill
the stash each pass. The architecture now states this scope.

This is a documentation correction backed by code review. Runtime,
test and fixture inputs remain identical to code-v48. CI fast remains
259 tests green on Python 3.12 and 3.13. Local slow scale acceptance
and queued retry are still pending; no new completed lane is claimed.

## code-v49: CI fast matrix and provenance checkpoint

Both code-v48 CI fast jobs passed on the same 102 validation inputs:
Python 3.12 **259 tests OK in 433.091s**, Python 3.13 **259 tests OK in
332.495s**. [Completed job metadata and links](../artifacts/followup-validation-20261018/ci-fast.json)
identify the tested commit. The automatic full CI jobs remain in progress
at this checkpoint; they are not counted as passes.

At **2026-10-18T09:27:36.372561+00:00**, the local complete slow lane has
**55 reported passes and 0 reported failures**. This is a partial log,
not a completed lane result. Retry remains queued sequentially. The
slow lane includes the expensive scale acceptance searches; the earlier
code-v45 lane took 14,217.561s, which is historical timing only.
No runtime, test or fixture input changed for this evidence checkpoint.

An independent read-only audit recomputed all 180 ranker rows, all 12
summary means, the exact random expectations, and the 15 task work
comparisons from the published data. Native ranker manifests do not
record imported dependency sources. A separate package-content
observation before and after that run brackets it with the same
290-file SpikeForge hash and clean Git commit. This is external
provenance evidence, not a claim that the native manifest contains it.

[Partial slow log](../artifacts/followup-validation-20261018/slow-partial.log),
[ranker audit](../artifacts/followup-validation-20261018/ranker-integrity-audit.json),
[dependency bracket](../artifacts/followup-validation-20261018/ranker-dependency-bracket.json),
and [dated stage/output manifest](../artifacts/followup-validation-20261018/manifest.json).

## code-v48: paired ablations and independent rank measurement

The measurement script now uses repository-relative fixture names in raw
rows as well as its manifest. A regression checks the serialized row
reference. The source-matched full fast lane passed **259 tests in
227.655s**, with no exclusions. Its 102 input files have SHA-256
`6111b2f0ed9cd031c668726350afbe0ece077b9b9ea0409a5e7e6ee5bf4e3a5c`.
The complete affected slow lane is running; retry follows sequentially.
Neither is counted as passed here. Earlier code-v45 results are historical.

### Author and beam (#70, #66)

The prespecified run used seeds 0, 1, 2, 100 training epochs, 12 bonus
refinement rounds, top-k/beam width 3, budget 128 and no fallback. Six
models were fitted on the four shadow tasks, using canonical or retained
reachability labels. Evaluation covered all four shadow and three novel
tasks, both search paths, and full/no-exact/no-gap/neither/no-neural
settings: **420 raw rows**, collected in **70.296s**. Independent train
and held-row interpretation determines acceptance. The two Luna reviews
check coverage, pairing, counts, weights and provenance; they did not
rerun training. All 336 bonus/neural and 210 supervision comparisons
have partners. Source, fixtures and imported SpikeForge bytes were
unchanged across measurement; its separate 290-file dependency hash
and clean Git commit are in the raw artifact.

Accepted results below require both train and held correctness. Each
denominator repeats the same task family across three seeds; these are
not independent task samples. Shadow is seen-task held-out evaluation;
novel tasks were not used for fitting or bonus selection.

| Labels | Family | Path | Full | No exact | No gap | Neither | No neural |
|---|---|---|---:|---:|---:|---:|---:|
| canonical | shadow | author | 6/12 | 5/12 | 6/12 | 5/12 | 1/12 |
| canonical | shadow | beam | 8/12 | 8/12 | 8/12 | 8/12 | 6/12 |
| canonical | novel | author | 0/9 | 0/9 | 0/9 | 0/9 | 1/9 |
| canonical | novel | beam | 6/9 | 6/9 | 6/9 | 6/9 | 3/9 |
| reachability | shadow | author | 9/12 | 9/12 | 9/12 | 9/12 | 0/12 |
| reachability | shadow | beam | 10/12 | 10/12 | 10/12 | 10/12 | 3/12 |
| reachability | novel | author | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 |
| reachability | novel | beam | 1/9 | 1/9 | 1/9 | 1/9 | 0/9 |

Only seed 0/canonical fitted nonzero exact/gap weights: exact 0.066947,
gap -0.114816. The other five fits selected zero for every bonus.
Consequently 70 of 84 pairs for each named bonus ablation are null
contrasts. Exact ablation changes one shadow author acceptance; gap
changes none. This run does not establish a general exact/gap effect.
Full versus no-neural holds the fitted full bonuses fixed: across 84
pairs, full wins 29, loses 3 and ties 52 on acceptance. This supports a
conditional neural ordering contribution on these small fixtures.
The no-neural control is paired with full bonuses, not a full factorial
of neural removal and every bonus ablation.

Reachability versus canonical improves shadow author from 6 to 9 and
beam from 8 to 10, but novel beam regresses from 6 to 1; novel author
remains zero. These are separately fitted/refined pipelines, not a
controlled label-only effect. Corrected train-graph labels do not
guarantee better transfer. The regression remains in the published data.

Candidate generation and signature evaluation remain classical. Author
budgets count generated triples; beam budgets count committed nodes.
Work counters are reported, but these are not equal-work comparisons.
Within-path ranking can change later expansions and their work; it
does not eliminate eager signature computation inside an expansion.
Wall times are descriptive for this host. Small selected frozen fixture
families and three seeds do not establish broad generalization.

### Solving-layer rank and actual signature work (#53)

A separate independent enumerator/encoder ran all three seeds at
epochs 0, 5, 40 and 50 on shadow, novel and plain-OOD: **180 rows**,
in **13.575s** stage wall time. This uses final-layer equality groups,
not the author-label training recipe above. The epoch-50 means are:

| Family | Tasks | Trained position | Probe order | Exact random expectation |
|---|---:|---:|---:|---:|
| shadow | 4 | 17.25 | 640.75 | 912.60 |
| novel | 3 | 390.11 | 828.00 | 521.43 |
| plain OOD | 8 | 831.12 | 482.50 | 830.29 |

All checkpoints and task rows remain published. The original blanket
"worse than random on transfer" conclusion does not survive this
recipe: novel beats random, while plain OOD is slightly worse at epoch
50. Other epochs differ. The author experiment also shows recipe
dependence; these results do not select a universally superior recipe.

Direct production instrumentation finds candidate-signature counts
identical across four rankers on each of all 15 tasks. Summed guided
signature work divided by summed streaming early-stop work is **3.209x
shadow, 2.371x novel, 3.325x plain OOD**. These are ratios of family
totals, not means of per-task ratios. Guided search also computes
operand signatures and neural scores. Good ordering does not produce
a signature-work win for the eager guided implementation.

[Raw author/beam data](../artifacts/followup-measurement-20261018/author-ablation-v1.json),
[paired summaries](../artifacts/followup-measurement-20261018/author-analysis.json),
[independent integrity checks](../artifacts/followup-measurement-20261018/author-integrity-audit.json),
[reproducible analyzer](../artifacts/followup-measurement-20261018/analyze_author_ablation.py),
[ranker task rows](../artifacts/followup-measurement-20261018/ranker/positions.csv),
[instrumented work](../artifacts/followup-measurement-20261018/ranker/work.json),
[fast log](../artifacts/followup-measurement-20261018/fast.log),
[partial slow log](../artifacts/followup-measurement-20261018/slow-partial.log),
and [source/stage/output manifest](../artifacts/followup-measurement-20261018/manifest.json).

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

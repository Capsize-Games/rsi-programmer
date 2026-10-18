# Adversarial review — rsi-programmer — 2026-10-17

> Historical baseline: this report and its reproductions describe repository commit `5c6701bbd9d0b2bba4bcd4022f0069a12f29c3cc` and architecture revision `e4d7210`. Follow-up fixes and later measurements are documented separately in [audit-amendments.md](audit-amendments.md); they do not replace the historical results below.

## Evidence scope

Repository HEAD: 5c6701bbd9d0b2bba4bcd4022f0069a12f29c3cc.
Architecture revision: e4d7210. Dependency revisions and fixture hashes are in the [source manifest](../artifacts/review-5c6701b/source-manifest.json).
GitHub issue bodies #51–#71 and all their comments were retrieved. No GitHub content was changed.
The earlier /tmp measurement script and raw data were not read or reused.
All source paths below are relative to the named repository revision.
Reproductions and outputs are in `../artifacts/review-5c6701b/`. Stubbed calls are identified below; they isolate production control flow and are not end-to-end training runs.

Portable reproduction setup (run from the current repository
checkout, which contains the published artifacts):

```bash
cd /path/to/rsi-programmer
SOURCE=$PWD
PY="$SOURCE/.venv/bin/python"
R="$SOURCE/artifacts/review-5c6701b"
CHECKOUT=$(mktemp -d)
git archive 5c6701bbd9d0b2bba4bcd4022f0069a12f29c3cc | tar -x -C "$CHECKOUT"
export PYTHONPATH="$CHECKOUT"
export REVIEW_REPO="$CHECKOUT"
r() { "$PY" "$R/repros.py" "$@"; }
d() { "$PY" "$R/deeper.py" "$@"; }
r 51 52
r 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
r extras
d packs
d formula
d feedback
d sandbox_lineage
d regressions
export REVIEW_OUTPUT=$(mktemp -d)
"$PY" "$R/measure.py" > "$REVIEW_OUTPUT/measurement.log" 2>&1
```

The unchanged `.venv` in the starting repository supplies dependencies.
`PYTHONPATH` and `REVIEW_REPO` direct source imports to the archived revision.
The measurement rerun writes to a temporary directory, leaving the published
historical `measurement.log` unchanged.

For the pre-fix #51/#52 comparison, reuse the same root `PY` and artifact
directory, but point imports at the archived `e4d7210` source:

```bash
PRE_FIX=$(mktemp -d)
git archive e4d7210 | tar -x -C "$PRE_FIX"
REVIEW_REPO="$PRE_FIX" PYTHONPATH="$PRE_FIX" r 51 52
```

`repros-current.log` preserves an early failed case-63 harness invocation
that omitted the task ID; the corrected isolated case-63 result is retained
in `repros-extra.log`. The original logs are copied with only machine-local
path strings redacted. Reproduction setup changes no recorded outputs.

## Severity-ranked findings


1. [P1, missed by audit] The default trainer credits programs that fail held-out validation.
   Claim attacked: architecture.md section 9's universal acceptance-independence claim, and the default loop's held-score contract.
   Verdict: CONFIRMED defect. loop_trainer.py:71–74 passes only training rows to guided_search and scores out['solved']; it never calls oracle_holds. The v2 family is a held-out task family, but its per-task held_out rows still need independent validation.
   Repro: r extras
   Output: E-oracle {"oracle_holds": false, "score": [1.966552222704002]}
   This is a real search on a small legal fixture: train x=y with target x+y; generation order returns x+x, which fails held row (1,2,3). The actual _solves path awards a positive score.
   Required change: validate the returned graph against train+held expectations before scoring. If retry is desired, provide a post-hoc oracle callback too. This repro does not establish that a particular historical v2 solve was invalid.

2. [P1, missed by audit] PromotionGate fails open on NaN scores.
   Claim attacked: promotion requires a positive lower bound and a minimum measured gain.
   Verdict: CONFIRMED defect. promote.py:46–47 propagates NaN; promote.py:66–70 checks lower<=0 and gain<minimum, both false for NaN.
   Repro: r extras
   Output: E-nan {"verdict": {"bootstrap_lower": NaN, "mean_gain": NaN, "promote": true, "reasons": [], "safety_failures": 0, "tokens_cap": 100000, "tokens_used": 0, "wall_cap_s": 1800.0, "wall_s": 0}}
   Required change: reject non-finite scores, derived statistics, and spend values before comparison. No special caller or malicious payload is needed to reach this public numerical API.

3. [P1] #53's incompatible work counters survive; its general transfer verdict does not.
   Verdict for ISSUE: CONFIRMED. Verdict for the broader measurement generalization: OVERSTATED.
   Evidence: probe.py:119–132 builds the full layer, then checks matches at :168–173. search.py:105–112 computes every candidate signature and operand features before ranking; :79–90 counts only scans. :138–146 never enforces budget. author.py:223 counts triples; :299/:315 count commits.
   Repro: r 53
   Output: {"budget": 1, "enumerator_tried": 12, "guided_tried": 4, "nodes": 13, "solved": true}
   The order policy here is generation order, explaining 4 instead of the issue's trained/untrained example of 3. The budget violation and dead nodes remain.
   Independent measurement command:
     "$PY" "$R/measure.py" > "$REVIEW_OUTPUT/measurement.log" 2>&1
   Exact outputs, summaries, seed-level rows, and direct signature instrumentation are in [measurement.log](../artifacts/review-5c6701b/measurement.log), [positions.csv](../artifacts/review-5c6701b/positions.csv), [position-summary.json](../artifacts/review-5c6701b/position-summary.json), and [signature-work.json](../artifacts/review-5c6701b/signature-work.json). Their source and sanitized output hashes are recorded in the [portable manifest](../artifacts/review-5c6701b/manifest.json).
   Required change: separate signature computation, scanned candidates, and committed nodes. Enforce a real budget. Limit the transfer statement to a specified recipe and seed set; do not infer a universal transfer failure from one seed.

4. [P1] #58's threshold-validation slice was fitted already.
   Verdict: CONFIRMED.
   Evidence: ltrain.py:174–181 packs/fits all groups, then calls refine. lprecision.py:114 discards the returned train split and keeps validation rows from those same groups.
   Repro: r 58
   Output: {"also_fitted": 2, "validation_rows": 2}
   This spies on the production train_lattice ordering with a six-row fixture; fitting and calibration are stubbed, and the actual split identifies the two overlapping rows. The source path makes the same overlap unavoidable on production groups.
   Required change: split before calibration/fitting and label threshold-tuning performance accurately. Use a further untouched evaluation set for an unbiased final performance estimate.
   Secondary repro: d packs
   Output: sweep_tie {"b0_after": 0.5, "st_after": 0.0}
   Existing non-grid thresholds .55/.1 are replaced when all scores tie. However, the docstring at lprecision.py:82 promises retention of the DEFAULTS, not arbitrary current values; that part of the issue paraphrases the promise too broadly.

5. [P1] #62's unmeasured child fitness can win selection.
   Verdict: CONFIRMED.
   Evidence: evolve.py:199–203 copies parent fitness into skipped children and persists it; :154–162 and :232 select without excluding skipped entries.
   Repro: r 62
   Output excerpt: good measured efficiency=0.5, bad screen_skipped=true with inherited efficiency=1.0; "selected": "bad".
   The screen, evaluator, and persistence are stubbed; the production _evaluate_pop and _fitkey execute unchanged. Selection of an unmeasured entry is sufficient to establish the defect; this does not pretend to measure the stub child's actual performance.
   Required change: unevaluated fitness must remain provisional and be excluded from measured bests/elite selection, or inherited-fitness racing must be an explicit, separately reported policy.

6. [P1] #69's lineage history and returned digest can be wrong.
   Verdict: CONFIRMED.
   Evidence: lineage.py:95 INSERT OR REPLACE; self_gate.py:48–53 restarts deterministic ids and returns the last lexicographically ordered row. evolve.py:173 and pool.py:49 also repeat ids across runs.
   Repro: r 69
   Output: {"kinds": ["third", "second"], "returned_second_digest": true, "returned_written_digest": false, "rows": 2}
   Two first-run records become two records after a new run overwrites record 0000. The new verdict returns the old record 0001 digest.
   Required change: run-scoped identities, insert-only history, and retrieval of the exact inserted record. Thread locks do not protect against intentional key replacement.

7. [P1 historical, fixed] #51's orphan optimizer was real.
   Verdict: CONFIRMED at e4d7210; fixed for the reported binding defect at 5c6701b.
   Evidence at e4d7210: policy.py:248, modules.py:225, lattice.py:185 replace the module without replacing its optimizer. Current rebindings: policy.py:254, modules.py:241, lattice.py:190.
   Repro: pre-fix command above; then r 51 on current HEAD.
   Before output: all three networks bound=false, moved=0.
   After output: SearchPolicy bound=true moved=4; ModularPolicy bound=true moved=12; BitLattice bound=true moved=10; recipient lr=0.06 in each case.
   These are actual backward/optimizer steps. No change is needed for the narrow fixed claim. An exact-training-resume claim would require optimizer-state and option serialization; see commit review below.

8. [P2] #52 remains true for ModularPolicy, but its existing test does not establish a need for implicit cross-call state.
   Verdict: CONFIRMED, partially fixed.
   Evidence: modules.py:90–111 consumes a previous batch's state when its size happens to match. Current policy.py:132 and lattice.py:101 start fresh. modules.py:141–145 now uses the single completed drive, fixing the discarded double-drive behavior.
   Repro: r 52
   Output: SearchPolicy repeat_equal=true/history_equal=true; ModularPolicy false/false; BitLattice true/true.
   Repro: d feedback
   Output: intra_drive_feedback {"head_rates_change": true, "seed": 0}
   tests/test_modular.py:71–86 explicitly demands priming, so preserving the test explains the partial fix. It does not show that different candidate rows or task batches should share state merely because their counts match. Delayed feedback can change outputs within one fresh drive, as the second repro demonstrates on random feature rows.
   Required change: make temporal state explicit for sequence callers, and use fresh state in independent scoring/training comparisons. Replace the priming-only contract test with tests of intended within-drive feedback and explicit continuation.
   Note: the fixed one-hot probe in r 52 showed no visible fresh-drive feedback effect; d feedback uses a different, recorded input. Neither result is hidden.

9. [P2] #55's encoding and bonus contracts differ across paths.
   Verdict: CONFIRMED.
   Evidence: policy.py:75–78/:212–216 uses total trace length; search.py:111 uses budget; author.py:69–71 increments candidate position but :210–212 uses a layer constant; ltrain.py:125 fixes gap=1 while lauthor.py:75–82 uses committed-set gap; policy.py:151–155 adds the rank-only bonus and :236–257 does not serialize it.
   Repro: r 55
   Output: train_frac_last=[1.0,0.0]; infer_frac_last=[1.2e-08,0.999999988]; training_gap=[1.0]; bonus_roundtrip_equal=false.
   Required change: define one deployable encoding contract and explicitly carry non-neural ranking parameters. The diagnostic trace denominator in the independent measurement is not a deployable fix: its solving-layer total is known only after doing the relevant search work.

10. [P2] #57's calibration report can be invalidated by its final calibration step.
    Verdict: CONFIRMED for drift; the universal non-monotonicity implication remains a possibility, not a demonstrated property of a shipped weight set.
    Evidence: ltrain.py:76–99 calibrates bit biases before struct; lattice.py:80–82 feeds struct output back into the encoder.
    Repro: r 57
    Output: maximum post-calibration bit-rate drift for seeds 0/1/2 = 0.0305555463 / 0.0472221971 / 0.0611111224. Seed 2 b3 was reported 0.4888888896 but ends at 0.4277777672, outside the declared 0.45–0.55 band.
    These are real drives and real bias calibration, on the deterministic 60-row random matrix in the repro.
    Required change: report all heads again after calibration finishes and iterate or control feedback. A claim that a particular struct rate curve is non-monotone additionally needs its bias sweep; I did not demonstrate that subclaim.

11. [P2] #56 gets important parts of the lattice loss mechanism wrong.
    Verdict: OVERSTATED.
    Evidence: author.py:98–102 partitions contrastive groups by opcode. ltrain.py:35–40 uses squared regression when the target is constant. train_lattice consumes those same-op groups at :174–175.
    Repro: d packs; r 56
    Output: real_lattice_packs {"packs":15,"mixed_bit_terms":0,"total_bit_terms":60}; perfect mixed-class margin loss=0.6931471825 with rate_gradient=[-0.5,0.5]; stop_weight_grad_norm=0.0381961986.
    Thus ALL 60 production bit terms on shadow use squared error, not the claimed full-range margin. The two membership-based struct terms do use the margin. The max is still over six pack-aggregate losses, not per-word errors; STOP is still fitted despite the calibration prose.
    ATan's derivative is strictly positive at finite membrane offsets (e.g. 0.0010121863 at offset 10). A rate of exactly 0 or 1 over six steps does not imply an infinite membrane offset or a zero surrogate derivative. The claim that saturation itself makes it vanish is false.
    Required change: narrow the margin-floor argument to mixed-class terms, correct the gradient claim, and either rename the aggregate max or implement the intended per-word objective. Correct the STOP calibration description or freeze it deliberately.

12. [P2] #59 identifies a sampling-unit problem but mixes it with ordinary bootstrap behavior and policy choices.
    Verdict: OVERSTATED.
    Evidence: promote.py:33–47 resamples scalar paired differences; self_gate.py:82–94 supplies per-task held scores from one seed and zeros for safety/tokens; headless.py:190–208 repeats tasks across generations and includes all copies.
    Repro: r 59
    Output: singleton gain/lower=0.1/0.1, promote=true; lower([0,1])=0.0, lower([0,1]*50)=0.42; SelfGate safety/tokens=[0,0].
    The actual callers do not implement a seed-cluster bootstrap, and repeating correlated observations narrows this interval. But cluster ids are not mathematically necessary if an API accepts already-aggregated cluster-level scalars. A degenerate one-observation percentile bootstrap is expected behavior of that estimator; a minimum sample policy needs an explicit contract. Bias correction and cluster resampling are separate choices.
    Required change: define and enforce the sampling unit at callers/API, refuse inadequate independent evidence under a stated minimum, and document which spend/safety clauses the trainer interface can actually measure.

13. [P2] #67's rejection-boundary claim survives, with several intentional behaviors mixed in.
    Verdict: CONFIRMED for escaped exceptions, unchecked operand words/trailing material, and Bool validation gap.
    Evidence: words.py:39–41 wraps references; :69–70 stops decoding immediately; bytecode.py:121 pops STORE without a check, and :148–150 executes outside the try; probe.py:278–283 checks output range but :289–290 separately requires Int.
    Repro: r 67
    Output: decode([0,999,-3,8,4242]) -> add(y,y); Bool output validates; binary_holds STORE -> IndexError: pop from empty list; ADD -> ValueError: stack underflow; HALT -> ValueError: empty stack at HALT; undefined FETCH -> 0; a stream with two HALTs disassembles successfully.
    The unhandled-VM-exception defect is broader than STORE: correctly raised VM ValueErrors also escape binary_holds. Reference wrapping itself is explicitly documented in _resolve; undefined slots and extra stack values need a specified VM contract before every such behavior is called a vulnerability.
    Required change: catch VM execution errors, consistently validate the accepted word alphabet, and state whether validate_graph guarantees an Int program or only a well-typed graph.

14. [P2, missed by audit] Malformed JSON graph data can crash the sandbox prescreen.
    Claim attacked: malformed candidates are rejected at the validator boundary.
    Verdict: CONFIRMED defect.
    Evidence: probe.py:255 uses an untrusted node['op'] as a dict key; sandbox.py:141–146 catches only ValueError.
    Repro: r extras
    Output: E-validation {"outcome":"TypeError: unhashable type: 'list'"}
    The graph's op=[] is JSON data; run_candidate raises instead of returning a rejection verdict.
    Required change: validate operation names' types before membership tests and normalize schema failures to the documented rejection path. This demonstrates a crash, not arbitrary code execution.

15. [P2] #68 ignores user-supplied seed values.
    Verdict: CONFIRMED.
    Evidence: proxy.py:150–153 adds k from range(len(seeds)), not each configured seed.
    Repro: r 68
    Output: {"configured":[10,20,30],"observed":[100,101,102],"reported":3} for candidate seed 100.
    _trained and gain evaluation are spies; production screen runs unchanged.
    Required change: use and report the configured offsets/values. The subset/group-granularity objection is prospective: the current one-group-per-task behavior holds and is not itself a reproduced current failure.

16. [P2] #60's caller inputs are degenerate, but eta-squared is not near one for every possible positive-vs-zero group.
    Verdict: OVERSTATED.
    Evidence: scripts/rsi_loop.py:138 imputes refused gains to zero and :150–167 groups by outcome and singleton cycle. autonomy.py:129–132 truncates distances with zip.
    Repro: r 60
    Output: singleton-group ratio=1.0; promoted=[0.05,100], refused=[0,0] ratio=0.3338; distance([0],[0,100])=0.0. Singleton agency is start on the first generation, stable thereafter, with diversity/spread=0.
    Required change: distinguish a valid descriptive eta-squared computation from an invalid causal interpretation of outcome-selected groups. Record real refused gains, reject mismatched weight dimensions, and flag singleton summaries. The claim that eta is almost one by construction needs narrower assumptions on within-group variance.

17. [P2] #61's inert edits exist and are already explicitly documented.
    Verdict: OVERSTATED as an undisclosed implementation defect; CONFIRMED descriptive count.
    Evidence: loop_trainer.py:14–17 expressly says evolve/population fields, margin and beta are inert. :88–96 consumes five fields. edits.py:19–31 still exposes all eleven fields.
    Repro: r 61
    Output: consumed=[hidden,joint_epochs,joint_lambda,lr,steps]; unused=[beta,evolve_budget_s,generations,margin,pop,sigma]; inert_edits=13; total_edits=23; documented=true.
    Trainer dependencies are stubbed to record actual configuration reads; the unmodified control flow is exercised.
    Required change: filter this trainer's proposal alphabet or implement the extra parameters to avoid guaranteed wasted trials. Calling this an accidental undocumented omission is refuted by the module header. Requiring every active parameter always to change the final score is also too strong for finite/discrete measurements.

18. [P2] #65's head/metric mismatch exists, but STOP is indirectly trained and the claimed function-class equivalence is unsupported.
    Verdict: OVERSTATED.
    Evidence: modules.py:182–207 trains own-channel and row-ranking margins; mgroups.py:108–111 measures within-row opcode argmax; modules.py:84–85 feeds the struct head back into the encoder.
    Repro: r 65; d feedback
    Output: own-op truths=[0,1], argmax=[1,0], all off-own direct output gradients=0; direct STOP output gradients=[0,0]. On the REAL recurrent network, STOP weight gradient norm=0.0863681510.
    The direct loss-output gradient probe uses synthetic head outputs to isolate the objective; the indirect gradient probe uses the actual six-step network. STOP affects later encoder/head outputs through feedback even though it is absent from the explicit score sum. Operand channels are indeed scalar candidate scorers. A recurrent multi-head system with auxiliary losses is not proved equivalent to a wider scalar head simply because final scores sum.
    Required change: align the named metrics with objectives or add classification/choice supervision. Rewrite 'STOP neither trained nor read' to distinguish direct score/loss use from recurrent influence. Supply an actual function-class proof before asserting equivalence.

19. [P2] #70's author attribution is confounded, but exact_flag is outside the SearchPolicy SNN and inference gap is not a current-layer answer bit.
    Verdict: OVERSTATED.
    Evidence: author.py:32–38 computes the flag but slices f[:DIM] for policy.scores; only the external four weights see it. author.py:208 and lauthor.py:75 compute inference gap on already COMMITTED signatures. Modular training's mgroups.py:63 uses the full current layer. SearchPolicy.rank ignores its gap argument.
    Repro: r 70
    Output: zero_snn_exact_weight_solves=true; nn_features=12; extra_features=4; gap_before_commit=1.0; gap_with_solver=0.0.
    This proves a classical extra weight can solve the small author example with zero neural scores. It does not prove the plain guided ranker receives the answer flag. Moreover, normal author/lattice search returns when it commits a target, so gap=0 generally ends search rather than providing the location of a new candidate solution.
    Required change: ablate the external bonus and define the scope of authorship attribution. A stronger claim of direct SNN answer leakage would require a code path that actually passes exact_flag to the neural input; the current slicing prevents it. Candidate generation and signature work remain classical in all the measured guided paths.

20. [P3] #63 treats a documented hard gate as accidental partial-credit loss and overgeneralizes the recovery scale.
    Verdict: OVERSTATED.
    Evidence: evolve.py:78 explicitly calls fitness gated; :90–93 zeros failed entries by design. :62–63 divides a task difference by family budget. scale.py:247–252 uses lexicographic regression.
    Repro: r 63
    Output: half_vs_none_identical=true; recovery_for_2x_cost=0.99; gating_documented=true.
    An all-or-nothing feasibility gate is a legitimate selection rule, albeit one that loses gradient among infeasible candidates. Stable sorting also means ties are not necessarily random as claimed. A family denominator gives weak per-task penalties, but 'approximately 1 whenever nonzero' is not universal: guided cost is not budget-enforced, and the formula can also exceed 1 for faster noisy searches.
    Required change: if partial-credit optimization is desired, specify it separately from feasibility. Declare recovery's normalization and bounds, and report per-family regression rather than letting one family's improvement mask another's loss.

21. [P3] #64 conflates task-train-row adaptation with held-out leakage and overstates the hash contract.
    Verdict: OVERSTATED.
    Evidence: ood.py:144–154 passes only task training rows to refinement; :166–168 makes it conditional; :182–183 reports refined=true. weights_hash at :33–40 promises only sorted neural-weight bytes. headless.py:89–98 deliberately trains/refines on its configured family. scale.py:170–181 reuses tasks for EXTRA fitting and acceptance.
    Repro: r 64
    Output: refined_ids=[t], evaluated_ids=[t], held_rows_in_refine=false; behavior_change_covered_by_hash=false after adding a rank bonus.
    Refinement/evaluation are spies recording production argument routing. This is task-level transductive adaptation, so it cannot be sold as zero-adaptation transfer. It is not held-out-label leakage, and the refined flag already discloses adaptation in the OOD report. The hash does exactly the narrow thing its docstring says; it is not an all-behavior fingerprint.
    Required change: distinguish frozen-neural/refined-extra results from zero-adaptation results. To sustain the stronger bug claim, show a contract requiring no task-level adaptation or show held labels reaching the fitter. Neither is established by the cited path.

22. [P3] #66's canonical imitation labels penalize alternatives, but they do not violate their stated membership definition.
    Verdict: OVERSTATED as a correctness bug; CONFIRMED training bias.
    Evidence: author.py:39–48 collects one solution's signatures, then :68–72 labels membership. mgroups.py:53/:74–77 and lattice's contrastive groups inherit this rule.
    Repro: r 66
    Output: {"alternative_intermediate_label":0,"canonical_signature_count":1,"valid_alternative":true}.
    For target x-y and bound 3, the intermediate x-x is negative even though x+((x-x)-y) is a valid alternative three-op program. This is consistent with imitation of a canonical trajectory; it does not establish an incorrect implementation of a 'can complete to any solution' contract, because that is not the implemented label definition.
    Required change: explicitly separate imitation membership from completability, or change the training objective and measure the effect. This issue concerns training labels, whereas #54 concerns destructive search-state merging; they are related but not duplicates.

23. [P3] #54's held-out incompleteness is real, but 'no path can revisit them' is false for final-layer guided candidates.
    Verdict: OVERSTATED.
    Evidence: probe.py:125–132 deduplicates before returning a solver; search.py:105–107 retains raw triples separately from deduplicated future members, and :79–90 scans those raw triples. tasks.py:243–250 filters generated tasks through the evaluated enumerator.
    Repro: r 54
    Output: enum_held=2; guided_with_oracle_recovers=true; lower_layer_collision_solved=false for required x+2*y within bound 2.
    On diagonal train rows, add(x,x) and add(x,y) collide. Guided search with a held oracle can nevertheless accept the final-layer add(x,y). For train target 3*x and held row (1,2) expecting 5, erasing the lower-layer y+y/x+y representatives prevents the correct x+2*y program within bound 2. Thus the stronger lower-layer failure survives refutation.
    Required change: narrow the universal wording; document or retain lower-layer alternatives if held-correct completeness is required. Record task-generator survivorship filtering. For each measured task in the independent three-family experiment, all final-layer train solvers also held, so this specific collision did not drive those measured positions.

24. [P3] #71's size metadata is wrong after rebuilding.
    Verdict: CONFIRMED, contained impact as the issue itself notes.
    Evidence: tasks.py:122–125 adds an extra operation for non-leaf children on top of part.size; the generator's ordinary convention at :73 is left.size+right.size+1.
    Repro: r 71
    Output: {"recount":5,"true_size":3}.
    Required change: use the same size convention in _size_of and test independent recounts, including select children. I did not find a current path passing these private generator Expr objects into feature encoding: fixture output serializes rows/formulas and search rebuilds its own expressions. The issue's downstream feature-impact statement is conditional, not observed impact.

## Commit 5c6701b review


- The complete four-file diff was inspected. Rebuilding Adam fixes stale parameter binding in all three classes and preserves the RECIPIENT optimizer's effective learning rate. Constructor-supported betas/eps/weight decay remain defaults. No production caller was found setting non-default optimizer groups/options.
- It does not preserve training continuation: Adam moments and step counters reset. r extras reports state_before=4, state_after=0, lr=.07, manually configured weight_decay=.1 becomes 0, betas=(.8,.9) become (.9,.999). This is compatible with the issue's explicitly allowed fresh-child optimizer fix, but it is not an exact checkpoint-resume protocol. Snapshot files never contained optimizer state. Do not invent a historical promise of moment preservation.
- Restoring a frozen lattice also loses requires_grad locks: r extras gives frozen_before=6, frozen_after=0. This behavior predates the fix because restore always rebuilt a module. It matters if snapshot/restore is used as a full training-state checkpoint; current train_lattice deliberately trains jointly. Define or extend the snapshot contract before relying on freeze preservation.
- SearchPolicy and BitLattice retain within-drive recurrence; they stop inheriting another call's temporal state. Lattice fitting/calibration already calls reset per pack/measurement. The fast lane's lattice training and OOD tests pass under the change. Snapshot temporal state is still serialized but no longer affects these two scorers; that semantic change is intentional and should be stated in the snapshot contract.
- ModularPolicy's single-drive fallback fix uses the caller's gap directly and is correct. Its implicit state reuse remains a distinct defect; the priming test is not evidence that independent candidate batches should share state.
- The new tests are not tautologies: one performs actual fit-after-restore and requires moved weights; the other exercises repeated and intervening drives. Pre-fix independent probes fail both contracts. However, both committed tests cover only SearchPolicy, live in the slow-lane module, and do not test Adam continuation, the other two classes, or explicit independent-drive OOD similarity.
- The independent turnover method from tests/test_rsi_evolve.py:128 was run directly and passed; its expensive class-level evolution setup was not invoked.

## Independent measurement


Protocol: CPU torch 2.8.0+cpu, one torch thread, SearchPolicy hidden=16/steps=6/lr=.05, seeds 0/1/2. Fit only shadow final-layer trace groups. Fixed checkpoints: 5, 40, 50 epochs, plus untrained. Continued checkpoint training is equivalent to successive epochs of the same deterministic fit. Three complete families: shadow (4), novel (3), rsi_ood_plain (8). Expression levels, compositional train/held signatures, candidate order, features, and solver positions were rebuilt independently; no audit measurement script was imported. The feature reimplementation is checked numerically against encode. Random is the EXACT expected earliest solver position (N+1)/(m+1), including all m valid candidates, not a noisy permutation draw. Inference denominator is the shipped 10^9 budget unless marked trace. Raw per-task and per-seed rows are in positions.csv.

Mean first-solver position, lower better; averaging tasks and then three seeds:
                  probe    random exact    untrained     5 epochs    40 epochs    50 epochs
  shadow          640.75     912.60           786.33         189.50       46.50        34.42
  novel           828.00     521.43           820.89         149.78      596.67       755.11
  plain OOD       482.50     830.29          1198.83         470.38      711.58       769.88

Novel seed means:
  5 epochs:   311.67, 35.67, 102.00 — all better than random expectation.
  40 epochs:  829.67, 197.67, 762.67 — mixed.
  50 epochs:  621.33, 840.67, 803.33 — all worse than random expectation.

With diagnostic trace-length normalization, 50-epoch means become shadow 15.00, novel 214.67, plain OOD 792.21. This changes the transfer verdict on novel. It demonstrates the encoding confound; the diagnostic denominator uses information from the solving layer and must not be presented as an efficiency improvement without paying its computation cost.

All 360 tested position records had identical train-match and held-correct first-solver positions; every retained final-layer train match in these three families held. No complete ordered training-row/target table was identical between shadow and either transfer family. This check does not claim that individual input rows or encoded feature values never overlap. The measured plain ranker receives 12 features and no exact_flag. Target-derived operand statistics are available specification features, not held-label leakage. Generated fixture selection is still conditioned on the enumerator (issue #54), so these small families do not justify population-wide claims.

The original probe-order and untrained seed-0 figures independently reproduce: shadow 640.75 / 833.75, novel 828 / 739. The comment's 'modest' recipe does not state an epoch count; I did not claim to reproduce its exact trained numbers 22/1225.33. Its random values 1298.9/459.3 are not the exact uniform-random expectations above; a finite random draw can differ. The original one-seed result may be true for its recipe, but its transfer conclusion does not survive as a recipe-independent claim.

Direct instrumentation (new candidate signatures; guided operand-feature signatures counted separately):
                  guided     streaming    ratio    guided operand signatures
  shadow           14673        4573       3.2086           29346
  novel             8685        3663       2.3710           17370
  plain OOD        23478        7061       3.3250           46956

Guided ranking happens after _intern computes the full candidate layer. The instrumented guided run uses generation order; changing rank cannot reduce this pre-ranking work. Streaming is independently implemented with the same per-(op,train-signature) retained lower levels and stops at the first generated fit. Recursive signature-call totals including feature operands are 235581/35465, 134793/28883, 371406/54173 for guided/streaming respectively. These are invocation counts, not equal-cost primitive operations. Instrumented wall times are saved, but are not a controlled end-to-end neural-search benchmark. '3–4x regardless of family/task' is too broad; 'no candidate-signature saving with this eager guided implementation' survives.

## Three formula checks

Command: d formula

1. Margin loss: M = mean_{p,n} softplus(1-(s_p-s_n)) matches policy.py:167–170.
   architecture.md:462–463 omits averaging from its displayed derivative. For an individual pair gap g, dM/dg = -sigmoid(1-g)/(|P||N|). With four zero gaps: implementation -0.1827646494 vs displayed -0.7310585786. If the author meant derivative of the unaveraged pair term, that term must be named instead of M.
2. Lattice pack loss: mean(six head losses)+lambda*max(six head losses), averaged over packs, matches ltrain.py:142–155. Numeric code=0.6881872416, independent=0.6881872478 (difference 6.28e-9). The constant-target fallback is squared error; in production same-op packs it applies to every bit head. The main formula is correct; the saturation interpretation and bit-margin generalization are not.
3. Promotion bootstrap: resample n paired differences with replacement, average each sample, sort 2000 means and select index floor(.05*2000)=100. This matches promote.py:33–47 and the architecture's explicitly specified order statistic. Independent seeded result=0.0, implementation=0.0 on differences [-.1,.2,.4]. A quantile symbol alone can imply interpolation, but the text explicitly defines the discrete index, so no transcription error is established there. There is no cluster resampling at the current task-level callers.

## Additional requested file checks and limits


- probe.py validation: Bool-output gap (#67) plus new unhashable-op crash above. No arbitrary code execution was demonstrated.
- search.py interning: it pre-materializes the whole layer, retains deduplicated representatives for future layers, and appends the accepted action again. The tiny solve's 13-node result demonstrates #53's dead-node cost. I found no separate invalid-reference interning failure in these checks.
- author.py stash/backtrack: overflow candidates are retained and deduplicated against committed signatures at pop. The termination comparison counts progress during backtracking, where new lower-layer commits require a rescan; bottom-up scan commits already feed higher layers in the current pass. The explicit MAX_PASSES=3 cap (:318/:339) means this is a bounded repeated sweep, not an unconditional fixpoint/completeness guarantee. No additional concrete default-width termination failure was established; do not turn an unproved suspicion into a new issue.
- bytecode.py: malformed execution errors escape beyond STORE, as shown above. Wrap16 arithmetic for supported add/sub/mul graphs was not refuted.
- sandbox.py: d sandbox_lineage returns sandbox_host_access status=ok when the public run_command hook reads a harmless sentinel outside cwd. It uses the owner's OS identity and only establishes a cwd/environment/resource boundary, not filesystem capability isolation. Candidate DAGs cannot issue file or shell instructions, so this is not evidence of a DAG sandbox escape. Supervisor interpretation at :153 and headless oracle checks execute outside child resource limits; I did not run an unbounded resource-exhaustion attack. No claims of such an exploit are made.
- lineage.py concurrency: r extras returns ProgrammingError when the same store is called from a worker thread, despite its mutex; sqlite3.connect at :54 uses default thread affinity. Current inspected callers are sequential, so no production concurrency failure beyond the already established duplicate-id overwrites is claimed. d sandbox_lineage also demonstrates an intent for a nonexistent mission is stored (rows=1); treat this as absent relational validation, not mission/authority security semantics, consistent with repository instructions.

## Verification

- just test-fast: 147 tests OK in 308.044s; complete output in [test-fast.log](../artifacts/review-5c6701b/test-fast.log).
- The two new regression tests: 2 OK in 0.410s, run separately; regressions.log.
- The individual turnover method passed without its class setup.
- Reproduction scripts cover all 21 issues; no issue is classified from source reading alone. Repro 63 initially had a harness-only missing-id KeyError; it was corrected, and its successful output is in repros-extra.log. The failed initial output remains visible in repros-current.log.
- The full ~3.7h suite, full slow lane, and retry lane were NOT run. No human testing is claimed.
- The source repository remains clean and unchanged. All scripts/data/report are in the separate review directory.

## Required closing decisions

(1) Of #51–#71: 11 CONFIRMED as filed, 10 OVERSTATED, 0 fully REFUTED, 0 UNVERIFIABLE. All 21 retain at least a reproducible narrow observation; that is not agreement with every mechanism, severity implication, or proposed fix. #51 is historical/fixed; #52 is partially fixed. Confirmed: 51,52,53,55,57,58,62,67,68,69,71. Overstated: 54,56,59,60,61,63,64,65,66,70.
(2) AMEND 5c6701b, preserving its fixes. Do not revert the optimizer binding or stateless SearchPolicy/BitLattice scoring. Resolve ModularPolicy's implicit state contract, extend real regression coverage to all three classes and independent OOD drives, and document fresh-optimizer versus full-resume semantics. Adam reset alone is not evidence this particular child-spawn fix should be reverted.
(3) An SNN contribution to candidate ORDERING survives, including transfer for the five-epoch recipe across three seeds. A general 'worse than random on transfer' conclusion does not. No candidate-signature WORK advantage survives for eager guided search; its neural scores are applied after the classical candidate signatures have already been computed.

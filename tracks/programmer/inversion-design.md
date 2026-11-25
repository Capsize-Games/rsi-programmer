# RSI Inversion design (Phase 0 freeze, 2026-11-25)

Status: frozen preregistration for epic RSI Inversion (LLM proposer,
RSI judge). No proposer code, bank, or LLM calls exist at this freeze.
Owner answers recorded here: proposer is Spark 1.3 background workers
supervised by the harness; bank scale is 90 dev + 90 sealed with rows
3 visible / 24 query / 40 held; JOE helper pick is deferred to Phase 8
entry on a Phase 7 PASS plus a separate go-ahead.

## Data flow

Task spec -> ProposerAdapter (Spark 1.3 workers) -> CandidateStore
(append-only, all proposals retained) -> structural gate
(parse + `probe.validate_graph` + `statement_syntax`) -> visible
judge (`SandboxRunner` sandboxed Python + `curlee check`/`run`,
`probe.signature` dedup) -> bounded repair (visible/query only) ->
sealed held acceptance (Python + CURLEE, judge code pinned) ->
`statement_fragment_library.qualify()` -> versioned library ->
retrieval (deterministic top-k, provenance-logged).

The judge decides. Proposer scores alone prove nothing.

## Module mapping (reuse unchanged unless noted)

- `construction_check.check_graph`: learner-independent gate, called
  only after construction; returns valid/observed/held/polynomial
  flags. Reused unchanged.
- `probe.validate_graph` / `interpret` / `signature`: structural-only
  validation, allowlisted evaluation, per-row execution tuples and
  `(op, sig)` dedup. Reused unchanged.
- `sandbox.SandboxRunner`: token denylist (oracle/held/ledger/verifier
  names), 64-node / 64 KiB caps, disposable RLIMIT subprocess,
  supervisor-side `interpret` recompute, `lineage_hash`. Unchanged.
- `headless.HeadlessLoop`: train-pair sandbox run, oracle held check,
  JSONL trace, PromotionGate. Unchanged.
- `benchmark_v2_runner`: `validate_graph_exact` + `load_frozen_inputs`
  hash-pin pattern (protocol/corpus/generator hashes). Reused.
- `statement_fragment_library.qualify()` (+ branch/body variants):
  donor gate, compiler-stability + byte-match + full coverage. Unchanged.
- `cost_ledger.CostLedger`: CPU phases plus new proposer token/CPU
  fields (Phase 3 extension, same 79-col/one-class rules).
- `rsi_programmer/proposer.py` (`EditProposer`): pre-LLM diagnostic
  ranker, retained as reference only; Phase 2 adds a new vendor-neutral
  adapter module without weakening the judge.
- `target_source` / `statement_source` / `utility_source`: emitters
  accept validated structures only, never arbitrary text. Unchanged.
- `campaign_worker.guarded_execute` + `scripts/run_limited.py`: bounded
  execution precedent (45 s timeout, `--fuel`, rc75 refusal). Reused.

## V1 task scope (frozen)

IN: bounded integer-list `process(input: &[Int; 256], count: Int)
-> Int` via `statement_source.curlee_source` (scalar return,
`0 <= count <= 256`) plus arithmetic-graph scope via
`target_source.emit_function` (add/sub/mul identities, strata
A6/A7/A8; statement shapes S1 loop+accumulate, S2 loop+branch, S3
loop+branch+update). Python unbounded ints vs CURLEE ints: every
acceptance states its i64/range assumption.

OUT: array/append scope (0/3 precedent, excluded), arbitrary
Python/CURLEE syntax beyond proven emitters, new control-flow or
array forms, JOE build path (Phase 8 pilot only), CURLEE source
copies into this repo, unsandboxed execution.

## Contamination model (frozen)

Split by task identity: visible rows (in prompt), query rows (bounded
repair feedback only, revealed one batch at a time after visible
pass, per-task cap, disjoint from held), held rows (never in
prompts, repair, retrieval, logs, or tuning). Sealed-pool held data
is file-guarded until Phase 7. Any sealed content in a prompt,
message, context, log, or tuning decision burns the pool and forces
regeneration. Old banks (300-task hard, 12k scaling eval, 11 reused
dev functions) are development data and are exclusion-listed.

## Cost and verdict (frozen shape)

`CostLedger` phases plus proposer token/CPU per proposal. A solve is
held-accepted only (both backends). Verdicts compare credited solves
at equal total CPU (fit/tokens charged, truncation at the smaller
arm total, shared seeded order) against streaming enumeration
(`pre_eval` policy None) plus a human reference at declared volume.

## Arms (frozen before sealed results)

- (a) Spark 1.3 single-shot.
- (b) Spark 1.3 + bounded repair (R=3).
- (c) Spark 1.3 + repair + retrieval.
- (d) Streaming enumeration control.
- (e) Untrained/classical where the grammar permits.
No arm is added after sealed results are opened.

## Bank v1 (frozen spec, generated only after this doc)

Home: `fixtures/llm_bank_v1/` with `protocol/tasks/visible/query/
held.json`. 180 tasks (90 dev + 90 sealed, identity-disjoint); per
pool 30 tasks x 3 strata (A6/A7/A8 arithmetic + S1/S2/S3 statement
on the scalar interface). Rows per task: 3 visible, 24 query (fixed
pool, disjoint from held), 40 held (6 boundary + 34 seeded, both
backends, range stated). Exclusions verified disjoint before
freezing: v1/hard/scaling corpora (incl. 23,498 identities), D1/D2
pools, 18-function bank, 2,647 teachers, 512 array teachers + 3 dev
tasks, 11 statement dev functions. Pinning: task/row sha256 plus
generator-source and CURLEE binary hashes, verified on load. Sealed
held file-guarded until Phase 7; solvers finish before held opens.

## Proposer interface and gate (Phase 2 contract)

`ProposerAdapter.propose(spec) -> Proposal{model, prompt_sha256,
params, candidate_text, token_usage, cpu}` where spec carries only
visible rows plus optional retrieved qualified fragments or visible
counterexamples. `CandidateStore` is append-only. Structural gate
runs before any execution; sandbox prescreen enforces token/node/
byte caps with supervisor-side recompute and `lineage_hash`.
`probe.signature` dedups on `(op, sig)`.

## Repair and library bounds

Repair: R=3 max, visible/query only, per-task query cap,
stop on accept/round-cap/cost-cap/repeated-hash loop; every round
stored with provenance; held-access path blocked by interface plus
test; contamination audit logs revealed rows and asserts disjoint
from held. Library: entry only on held acceptance with byte-match,
compiler before==after, wrong-expected rejected, full coverage;
provenance (source/spec/arm/model/prompt-hash/artifact/compiler/
coverage); deterministic top-k retrieval with shown-ids logged;
frozen version hash; construction cost charged to arm (c).

## Phase 6 PROCEED / Phase 7 PASS (frozen)

Phase 6 PROCEED iff (i) best LLM arm accepts tasks in deep families
where streaming scores ~0, AND (ii) cost/accepted within frozen
multiple M (M=3), AND (iii) repair (b vs a) or retrieval (c vs b)
shows a positive identical-task delta; else KILL: write
`inversion-dev-results.md`, cancel Phase 7. Phase 7 PASS iff the
winning LLM arm beats streaming on credited solves at equal total
CPU at declared volume AND clears the frozen deep-family rate; FAIL
on either prong closes the line with no same-epic retest. Crashed
runs retry only with identical protocol plus retained failure log.

## Safety contract

Sandbox-first: no LLM text executes outside `SandboxRunner` +
`run_limited.py` (fuel/time caps, rc75 refusal). CURLEE compiler is
local-only and pinned by path + sha256 before/after every qualify;
never copied into this repo. JOE is not in the build path; Phase 8
is one pure helper, human merge/edit/reject, on PASS + go-ahead.

## Inventory audit (Phase 0 evidence)

- CURLEE binary: `/home/joe/Projects/curlee/build/linux-debug/curlee`
  sha256 `9697bc76461c703303cbe5506399553e16ad3ccbc808a6388de82
  891a4466832` (measured 2026-11-25).
- Baseline: `just test-programmer` (fast lane) recorded on card
  `689e3429` at Phase 0 close; manifest/justfile delta stated there.
- Manifest: no new modules at this freeze; bank/generator/runner
  modules register in `tracks/manifest.json` and the justfile lane
  when Phase 1 lands.

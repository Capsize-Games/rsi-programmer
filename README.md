# rsi-programmer

A spiking neural network that learns to write programs. A small SNN
search-control policy (linear 12-feature input, hidden-16 leaky
integrate-and-fire layer, scalar readout; 229 parameters) ranks
candidate program-construction choices while a typed-DAG enumerator
builds arithmetic expressions. The policy trains on frozen
search traces with a pairwise margin loss, plus same-op contrastive
pairs that force it to use operand features.

## Measured behaviour

Headline numbers are author-path only: programs the SNN authors
with no enumerator rescue. Enumerator-backed figures sit in a
separate column, never the headline.

- Author path, shadow family: beam authors 2 of 4 (solve 0.50,
  cost 487 vs enumerator baseline 14673) with refined author
  weights; misses report `no-authored-solution`, never a rescue.
- Author path, scale acceptance (evolved policy, zero author
  weights): beam authors 0 of 4 shadow and 0 of 4 hard tasks;
  every verdict carries its path.
- Enumerator path (debug comparison only): exhaustive search
  solves 4 of 4 shadow at cost 14673, and guides novel transfer
  on three brand-new rules (158/122/952 tried vs 939/939/6807).
- Legacy mixed-path figures (enumerator rescue included, kept
  for continuity, not comparable): beam wins 11 of 24, retry
  loop clearing distractor near-misses, operand top-1
  0.267/0.133/0.200 across seeds (chance 0.060).
- Known limits: the beam still loses tasks where narrowing drops
  true paths. Training continues; see `scripts/train.py`.
- Organization metrics (expression-only, describe the process,
  never a standing): autonomy ratio 0.000 on a 1-generation
  elite archive (single lineage, nothing to attribute);
  partition divergence 0.000000 on the W1 feedforward monolith
  (no cross edges to sever) vs 0.208 on the driven modular
  policy (fb_fc severed, seeds 0/1/2: 0.208/0.056/0.069);
  agency profile converges as fitness spread collapses.
  See `rsi_programmer/autonomy.py`.
- Structural OOD (expression-only, v2.4.0 §9.1: no gradient
  placement): expanded plain training (24 tasks) lifted broad
  to 3/8 mirrored plain and twin (narrow 1/8); blindness swung
  past orthogonal to -0.267 (narrow 0.985; 1.0 = blind to the
  reversal). Full-IR boundary: 2 solves a side via degenerate
  branches (constant condition; out-of-range comparison over
  rows -4..4); v2 fixture rejects them at generation.
  Enumerator from scratch solves 8/8 plain (mean tried 2935)
  and 8/8 full-IR (125930). See `rsi_programmer/ood.py`.

## Layout

- `rsi_programmer/` — the library: `policy` (SNN),
  `probe` (typed-DAG synthesiser and independent checker),
  `author`/`evolve`/`pool` (training operators),
  `scale`/`headless` (acceptance loops), `sandbox`/`promote`
  (execution gate), `traces`/`rich`/`actions` (data and features),
  `lineage` (sqlite record of training runs).
- `fixtures/` — frozen task families: W1 shadow (training), hard,
  and novel (held-out transfer).
- `scripts/train.py` — canonical training run; writes weights and a
  manifest with the full topology spec and provenance into
  `weights/`.
- `tests/` — unittest suite; the engine or independent checker
  decides every acceptance, never the learner's own scores.

## Run it

```sh
just setup          # .venv + requirements + spikeforge source pin
just test           # full suite from the repo root
just train          # seed 0, 50 epochs -> weights/rsi-search-policy-w1.*
```

`spikeforge` is a source dependency (Capsize-Games/spikeforge); `just
setup` pins a local checkout. Torch CPU wheels come from the
PyTorch index (see `requirements.txt`).

## Weights

Released weights ship as GitHub release assets (weights, manifest,
and license together), reproducible with `python scripts/train.py
--seed 0 --epochs 50` from this repo. Spikeforge hub listing is
tracked but not yet published: `python scripts/verify_hub.py`
currently reports GitHub PASS and hub FAIL (`no published model`)
for both weights generations. Publishing to the hub is a release
blocker before any hub-distribution claim returns here.

## License

BSD-3-Clause, copyright Capsize LLC; see [LICENSE](LICENSE).

# rsi-programmer

A spiking neural network that learns to write programs. A small SNN
search-control policy (linear 12-feature input, hidden-16 leaky
integrate-and-fire layer, scalar readout; 229 parameters) ranks
candidate program-construction choices while a typed-DAG enumerator
builds arithmetic expressions. The policy trains on frozen
search traces with a pairwise margin loss, plus same-op contrastive
pairs that force it to use operand features.

## Measured behaviour

- A frozen policy trained only on the W1 shadow family guides search
  on three brand-new rules it never trained on, beating the plain
  enumerator on all three (158/122/952 tried vs 939/939/6807).
- A backtracking beam author lifts wins to 11 of 24 tasks with
  SNN attribution measured and zero flips, and an oracle-rejection
  retry loop solves the remaining distractor near-misses.
- Known limits: operand top-1 precision is 0.267/0.133/0.200 across
  seeds, and the beam still loses tasks where narrowing drops true
  paths. Training continues; see `scripts/train.py`.

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

Released weights are distributed through the spikeforge model hub as
`reference/rsi-search-policy-w1` (weights, manifest, and license
together), reproducible with `python scripts/train.py --seed 0
--epochs 50` from this repo.

## License

BSD-3-Clause, copyright Capsize LLC; see [LICENSE](LICENSE).

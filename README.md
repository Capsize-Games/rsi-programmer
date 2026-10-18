# rsi-programmer

This project studies a spiking neural network (SNN) as a ranking policy
for arithmetic program synthesis. The SNN ranks candidate operations;
classical search assembles selected steps into typed program graphs and
computes their outputs on task rows. A separate interpreter checks
accepted programs against the frozen training and held-out expectations.

The repository includes a trace-trained `SearchPolicy`, author and beam
search, a modular policy, a bit-slice lattice, task-family generators,
and promotion/evaluation loops. The implementation and its limits are
described in [architecture.md](architecture.md). Historical performance
figures in earlier reports used older feature encodings and work
counters; they should not be read as measurements of the current code.

The [audit follow-up](docs/audit-followup.md) records subsequent fixes.
The [architecture follow-ups](docs/architecture-followups.md) track the
pre-evaluation search experiment and its evaluation gates.
The [code-v45 amendments and validation](docs/audit-amendments.md) retain
that release's exact test results and three-seed measurement.
The [historical adversarial
review](docs/audit-review-5c6701b.md) includes reproductions for issues
#51–#71 and the baseline measurement at `5c6701b`.

## Layout

- `rsi_programmer/` — policies, search, typed-DAG validation,
  training, task fixtures, and evaluation utilities.
- `fixtures/` — frozen task families used for training and evaluation.
- `scripts/train.py` — canonical ranker training entry point.
- `tests/` — unit and integration checks for search, training, and
  independent acceptance.

## Run it

```sh
just setup          # create .venv and install project dependencies
just test           # run the full test suite
just train          # seed 0, 50 epochs; writes weights/
```

`spikeforge` is a source dependency (Capsize-Games/spikeforge); `just
setup` uses a local checkout. For another location, pass its absolute
checkout root as a positional argument: `just setup /path/to/spikeforge`.
Torch CPU wheels come from the PyTorch index (see `requirements.txt`).

For lazy evaluation, import `pre_evaluation_search` from
`rsi_programmer.pre_eval`. It accepts the same training-row specification
and ranking interface as guided search, but `budget` limits candidate
signature evaluations. Pass `policy=None` for streaming generation order.
Policies rank candidate descriptors using cached operand results before
the candidates are executed. Descriptor scoring still has a cost, which
the result reports separately. A returned training match requires an
independent held-out check before it counts as an accepted solution.

## Training output

`scripts/train.py` writes a weight file and a JSON manifest to the
selected output directory (the default is `weights/`).

For independent first-solver-position measurements, run:

```sh
.venv/bin/python scripts/measure_ranker.py --out /tmp/ranker-results
```

This fits shadow final-layer trace groups with three seeds and reports
untrained, 5-, 40-, and 50-epoch positions on shadow, novel, and plain OOD
fixtures. It uses the exact expected random position and writes per-task
data, source/fixture hashes, and signature instrumentation. Signature
counters measure root vector evaluations; runtime and neural work are
outside that metric. Its early-stop streaming comparator differs from
the probe enumerator, which finishes each layer before checking it.

## License

BSD-3-Clause, copyright Capsize LLC; see [LICENSE](LICENSE).

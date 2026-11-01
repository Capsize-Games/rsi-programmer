# rsi-programmer

The primary development track is neural program synthesis: a learner that
constructs executable programs and improves from independently checked
training experience. The
[programmer roadmap](tracks/programmer/README.md) defines the milestones.
The [stateful follow-up](tracks/programmer/stateful-results.md) solved
0/5 new tasks, including maximum and capped-count programs. A broader
curriculum did not improve that result and retained only 5/6 earlier tasks.
Supplied control programs passed all five tasks in Python and CURLEE;
the learner's generalization is the current limitation.
The preceding [verified-fragment trial](tracks/programmer/repair-results.md)
retains four statement-generated utilities and repairs two more. All six
pass 40 held cases each in Python and CURLEE. The learner constructs new
conditions within contexts mined from previously checked programs; the
emitter supplies syntax lowering and file handling. General programming
and a neural cost advantage remain unproven. The preceding
[statement-construction trial](tracks/programmer/statement-results.md)
retains its failures and development results.
The earlier [list-composition result](tracks/programmer/language-results.md)
is retained separately.

The work is separated into three [research tracks](tracks/README.md):

| Track | Current focus | Checks |
| --- | --- | --- |
| [Programmer](tracks/programmer/README.md) | Primary; generated Python and CURLEE code that executes correctly | `just test-programmer` |
| [Sensing](tracks/sensing/README.md) | Preserved timed prediction and adaptation experiments | `just test-sensing` |
| [Sound Companion](tracks/sound-companion/README.md) | Preserved native supervised sound-learning app | `just test-companion` |

`just tracks` lists the declared scopes. Track separation preserves existing
runtime import paths, fixtures, evidence and aggregate test commands.

Earlier experiments study a spiking neural network (SNN) as a ranking policy
for arithmetic program synthesis. Those policies rank candidate operations;
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

The [frozen pre-evaluation benchmark](docs/benchmark-pre-eval.md) compares
streaming, heuristic, random and neural controls on a separately generated
corpus. Its protocol and corpus were published in `code-v61` before any
solver measurements. Run the complete fixed matrix into a new directory:

```sh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  scripts/benchmark_pre_eval.py --output /tmp/pre-eval-results
```

The runner preserves all task/seed/budget attempts, independently checks
returned programs, and rejects changed frozen inputs. Candidate work,
policy overhead, validation and training costs are reported separately.

## License

BSD-3-Clause, copyright Capsize LLC; see [LICENSE](LICENSE).

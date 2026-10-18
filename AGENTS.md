# rsi-programmer

BSD-3-Clause software, copyright Capsize LLC (see [LICENSE](LICENSE)).

## Commands

- `just setup` — create `.venv`, install requirements, configure a local
  spikeforge source checkout (pass its absolute checkout root as the
  positional argument: `just setup /path/to/spikeforge`).
- `just test` — full unittest suite, run from the repo root.
- `just train` — canonical training run (seed 0, 50 epochs, writes
  `weights/`).

## Conventions

- One class per file, full type hints, 79-column lines. When a limit is
  reached, split the module instead of growing past it.
- `rsi_programmer/lineage.py` is the only persistence: a sqlite record of
  missions and intent entries. It replaces the rover mission stack the
  training loops were extracted from; keep training runnable without it
  gaining mission/authority semantics.
- The engine decides success: a synthesised program counts only when the
  independent checker (`probe.validate_graph`) holds on held-out rows.
  Learner-reported scores alone prove nothing.
- Never commit `weights/*.pt`, lineage databases, or `.venv/`.

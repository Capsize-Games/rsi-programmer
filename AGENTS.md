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

## Audit checkpoints

When publication is authorized, publish completed fixes and review evidence
as the work progresses, including during long test runs. Push meaningful
checkpoints and tag published snapshots. Record the exact completed tests,
partial runs, and pending checks in each checkpoint. Update issue status
with the evidence available; keep unresolved findings open. Publish again
at validation milestones instead of holding all evidence for the final run.

Owner update (2026-10-18): commit and push completed changes without waiting
for remote CI. Run affected local checks and state pending CI accurately;
fix actual failures without weakening tests. Pause new tags and releases
until the owner asks to resume them.

## Sound Companion

Owner correction: the creature's purpose is to learn and remember sounds
with the SNN. The SNN must choose its action label; keep the previous
example matcher as a comparison control. Do not replace neural learning
with example matching or route actions through the control. A separate
rejection guard may suppress an unsupported neural choice, and must be
described as such. Preserve evaluation-only recordings, retain old taught
examples for replay, and compare both backends on the same fresh sounds.

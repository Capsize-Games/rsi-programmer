# Research tracks

Owner direction, 2026-11-01: keep three tracks in this repository and make
the original programmer objective the primary development focus.

| Track | Status | Work |
| --- | --- | --- |
| [Programmer](programmer/README.md) | Primary | Learned construction of executable programs, independent checking, and eventually verified improvements to the learner. |
| [Sensing](sensing/README.md) | Preserved | Online prediction, adaptation, retention, and change detection on timed measurements. |
| [Sound Companion](sound-companion/README.md) | Preserved | Explicitly taught sound recognition, saved memory, and native creature actions. |

Preserved tracks remain runnable and retain their evidence. New experiments
should serve the programmer track unless the owner changes priorities.
Cross-track changes must identify the affected tracks and run their checks.

## Commands

Run from the repository root:

```sh
just tracks
just test-programmer          # programmer fast lane
just test-programmer slow     # expensive programmer integration checks
just test-programmer retry    # expensive retry checks
just test-programmer all      # all programmer lanes, sequentially
just test-sensing
just test-companion           # includes native-window checks under Xvfb
```

The track runner acquires the shared builder lock, uses one Torch/OpenMP
thread, and returns exit status 75 if another RSI job holds the lock. A
refused command has not run its tests. The companion uses `.venv-desktop`;
the other tracks use `.venv`.

To inspect the exact scope without importing a model or starting tests:

```sh
python3 -m scripts.tracks test programmer --lane all --list
```

Existing `just test-fast`, `just test-slow`, `just test-retry`, `just test`,
and `just test-sound` remain aggregate or historical gates. A focused track
pass does not substitute for an affected aggregate gate. The fast aggregate
also checks the track manifest for missing tests and accidental overlap.

## Source ownership and shared dependencies

The explicit test ownership is in [manifest.json](manifest.json). Every
discovered test module belongs to a track. `tests.test_audio_capture` is
deliberately shared by sensing and the companion.

Runtime modules retain their existing import paths. Programmer owns the
typed graph engine, search, policies, graph-learning experiments, benchmarks,
and promotion/lineage infrastructure. Sensing owns `sensing_*` and `audio_*`;
the companion owns `sound_*` and its launcher/installer. Physical capture
in `audio_capture.py` is a shared dependency. `sound_memory_fake.py` is a
companion test helper.

Historical artifact and fixture paths remain the source of evidence for
their recorded revisions. A new track name does not change an experiment's
scope or turn a negative result into a positive one. This separation adds
ownership and entry points; physical package extraction is a later step if
dependency isolation warrants it.

# Sound Companion

A native Linux desktop creature that learns short sounds from microphone
examples. Each taught sound can wake it, put it to sleep, make it dance,
or start a one-minute timer. Sound recognition uses local example matching;
this demo does not use an SNN or an LLM.

Run from the repository root:

```bash
just sound
```

## Teach and test

1. Click **Enable microphone** and stay quiet during the brief background
   calibration. Capture uses the current physical microphone input.
2. Enter a sound name, choose an action, and click **Record example**.
   Wait for **Make sound now**, then make a short, distinct sound.
   Record three examples for that name. Try a whistle, a clap, or a pair
   of taps; deliberately similar sounds may be rejected as ambiguous.
3. Select that name under **Test recognition**, click **Record test sound**,
   and make a fresh example after the cue. The window shows the expected
   and recognized names and counts correct tests. Test recordings do not
   add training examples, fire actions, or become correction candidates.
4. Click **Start listening** to enable the assigned actions. For a live
   recognition mistake, select its correct name in **Sound memory** and
   click **Correct last sound**. This explicitly adds the recent live
   example to the selected name; it does not repeat the mistaken action.
5. Teach another name, then test the older names again. Close and reopen
   the window and repeat the checks to verify that saved memory reloads.

**Pause listening** pauses recognition actions while keeping the microphone
available for teaching and tests. Close the window to release the microphone.
No distinct onset within five seconds cancels a recording. Keep sounds
within roughly one second after the cue, and use approximately the same
microphone position and background conditions while teaching and testing.

The microphone must hear an onset at least 5 dB above its calibrated
background. If it reports no distinct sound, try a sharper sound or move
closer. The input level is measured amplitude, not recognition confidence.
The timer counts down for 60 seconds and rings the desktop bell at zero.
The other actions animate this application; there are no arbitrary shell
commands or external app controls in this version.

## Memory

By default, feature examples and test summaries are saved to
`~/.local/state/rsi-programmer/sound-companion/memory.json` with owner-only
file permissions. Raw audio is held briefly in memory and is not saved.
**Forget sound** deletes only the selected name. Adding a new name retains
older examples; recognition can still become ambiguous when names sound
alike. A corrupt saved file is preserved and further writes are blocked.

The recognizer compares 1.2-second clips using overlapping frequency
filters and temporal features. It retains at most 20 examples per name.
It rejects quiet clips, distant examples and close competing labels.
Acceptance radii come from the taught examples. A match is not a calibrated
probability, and successful synthetic checks do not establish accuracy on
your microphone. Fresh recordings in the test panel provide that check.

## Setup and checks

The desktop environment uses the system Python's Tk library. The existing
research `.venv` is unchanged. Its bundled Tk aborts when constructing even
a text-entry widget on this machine, matching the upstream
[standalone-Python Tk issue](
https://github.com/astral-sh/python-build-standalone/issues/146).

```bash
just setup-sound /absolute/path/to/spikeforge
just test-sound
```

Setup requires system Python with Tk, `uv`, a local Spikeforge checkout,
and PulseAudio/PipeWire's `pactl` and `parec`. `test-sound` uses Xvfb for
isolated native-window checks. The pinned runtime requirements are the
same as the research environment. No model weights or sound memory belong
in Git.

The independent synthetic learn/add/restart check runs with:

```bash
flock /tmp/rsi-programmer-builder.lock \
  .venv-desktop/bin/python -m scripts.check_sound_learning_demo
```

It creates a temporary memory, teaches three generated sound families,
tests fresh seeded variations, adds a fourth, rechecks the old sounds,
reloads memory, and checks all four. It reports failed and unknown matches
alongside successes. These generated clips are not recordings of a user.

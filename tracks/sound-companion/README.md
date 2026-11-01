# Sound Companion

This track preserves the native teachable sound creature. Programmer is the
current development priority. Sound Companion remains runnable as a separate
supervised recognition application.

## Scope and learning contract

The `sound_*` modules, native window, launcher, installer and sound comparison
scripts belong here. Physical capture in `audio_capture.py` is shared with
sensing. Use `.venv-desktop` with system Tk and an isolated Xvfb display for
window tests.

The SNN chooses the recognized label. A distance support guard can reject
that choice; the old example matcher is a comparison control and cannot
replace the SNN's label or trigger a fallback action. Explicit teaching
refits both neural layers from the configured seed using all retained
teaching examples. Saved parameters reload without fitting.

Keep fresh tests evaluation-only. Retain old taught examples for replay and
compare both backends on identical queries. A cumulative test count across
changing models is historical feedback, not a fixed-model accuracy estimate.

## Evidence and remaining work

The [generated comparison](../../docs/sound-snn-comparison.md) demonstrates
neural classification, replay retention and reload equality on its generated
sounds. The matcher ties accuracy and costs less in that comparison. It does
not establish personal microphone accuracy or chip efficiency.

The next recognition milestone requires fresh personal recordings across
teaching stages, independent sessions and restart, with unknown controls,
model-version-specific results and unchanged evaluation examples.

## Commands

```sh
just sound
just test-companion
```

`just test-sound` remains available with the same desktop test scope. See
[the app guide](../../docs/sound-companion.md) for teaching, correction,
memory and microphone behavior. Starting tests does not enable capture.

# Sensing

This track preserves the timed sensing and continual prediction experiments.
Programmer is the current development priority. Sensing remains a separate
workload with its own feedback, retention and deployment criteria.

## Scope

- `sensing_*`: timestamped prediction, change detection, controls, data and
  recorded fan benchmarks.
- `audio_*`: physical microphone capture, six-band prediction, replay memory,
  session boundaries, sealed probes and private recording sessions.
- `scripts/fan_microphone.py`, sensing/audio measurement scripts and auditors.
- The corresponding frozen fixtures, recordings, artifacts and result docs.

`audio_capture.py` is also used by Sound Companion. Changes to capture need
both tracks' affected checks. The `online_*` graph-execution experiments
belong to the programmer track because their workload is program search.

## Evidence and remaining work

The [vibration pilot](../../docs/sensing-fan-result.md) fails its detector
gates. The [original microphone run](
../../docs/live-fan-microphone-30-minute-result.md) improves forecasting
against a frozen SNN but loses retention and trails ANN/EMA live prediction.
[Replay-memory diagnostics](../../docs/audio-continual-retrospective-result.md)
reduce some retention loss while adding work and a prediction tradeoff.

Useful change detection still needs independently logged physical changes,
separate held recording sessions, calibrated sequence-level alarm rates,
and a fixed positive control. Chip precision, memory, update placement and
end-to-end power/latency remain unverified. A stopped partial recording is
not a completed prospective comparison.

## Commands

```sh
just test-sensing
```

This command runs the declared sensing checks under the shared builder lock.
It does not start a microphone session. Existing capture commands and private
state remain documented in [the microphone guide](
../../docs/live-fan-microphone.md) and [memory comparison protocol](
../../docs/audio-continual-comparison.md).

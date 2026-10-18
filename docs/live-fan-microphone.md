# Live fan microphone learning

The live runner predicts the next 0.2-second frame's six frequency-band
energies from a physical PulseAudio input. It compares an adapting SNN with
the same trained SNN held fixed, a continuous neural model, a five-second
exponential average and persistence. It uses the existing scalar predictor
in six independent bands; it does not use an LLM.

## Start, inspect and stop

Run these commands from the repository root, with the project environment:

```sh
.venv/bin/python -m scripts.fan_microphone start
.venv/bin/python -m scripts.fan_microphone status
.venv/bin/python -m scripts.fan_microphone stop
```

`start` uses the default physical input, refuses output-monitor sources and
leaves source selection and volume unchanged. `--source NAME` selects a
physical input explicitly. The detached process continues after the shell
exits. `--minutes N` limits monitoring to N minutes of captured samples,
after the initial 80-second fit/calibration/probe sequence. Omitting it runs
until stopped or capture fails. A new start creates a new comparison;
resuming a stopped learner is not implemented.

`status --json` returns the full aggregate report, including the latest
signal level, strongest FFT bin, completed minute blocks and reference
fingerprint. `stop` sends the owned process a termination request; use
`status` again to confirm it has stopped.

## Fixed comparison

The recipe was frozen in
[`audio_fan_live_v1.json`](../fixtures/audio_fan_live_v1.json) before model
execution. The first 30 seconds fit the normalization and train the neural
models once. A copy of that trained SNN becomes the fixed reference. The
next 30 seconds calibrate each arm's alarm threshold on disposable copies.
The following 20 seconds form a sealed probe, excluded from training.

During monitoring, each arm predicts before receiving the next target.
Only the adaptive neural arms update. Each completed minute reports paired
next-frame mean squared error (MSE) and observed alarms. The first block
excludes five seconds of transient warmup; later blocks use all 60 seconds.
At monitoring start and each minute, read-only copies predict the same
sealed probe, allowing retention to be compared on unchanged inputs.
These repeated probe checks are correlated observations from one recording.

The fixed reference, fit-only normalization and thresholds stay unchanged.
Positive relative improvement means lower MSE than the fixed SNN; negative
values mean higher MSE. Comparisons with the continuous model and simple
predictors remain visible even if they outperform the SNN.

These are prediction measurements. There are no independently logged fan
fault labels, so this run cannot establish fault-detection accuracy or
false-alarm ground truth. A strongest FFT bin can confirm sound is present,
but does not identify its source as a fan.

## Local data and verification

The default private directory is
`~/.local/state/rsi-programmer/microphone/`. Session directories use mode
0700; status, feature/forecast traces and parameter checkpoints use 0600.
Raw PCM is discarded after feature extraction. Feature traces can still
describe the acoustic environment and stay outside Git alongside weights.
Aggregate publication excludes the microphone serial and private paths.

The runner holds `/tmp/rsi-programmer-builder.lock` while capturing and
learning. Stop it before running other tests, model jobs or benchmarks.
At monitor start and minutes 1, 2, 4, 8, and so on, it independently audits
the flushed trace's normalization,
forecast causality, baseline recurrences, errors, alarms and aggregate
arithmetic under the same lock. Doubling the interval limits repeated
full-trace work. This is an arithmetic audit of recorded
forecasts, not independent neural-network replay or audio attribution.

To plot a saved aggregate status with an isolated plotting dependency:

```sh
uv run --no-project --with matplotlib --python .venv/bin/python \
  python -m scripts.plot_audio_progress /path/to/status.json \
  --output-dir /new/output/directory
```

The chart distinguishes live minute-block errors from repeated sealed-probe
errors. Its accompanying JSON contains the plotted aggregate values.

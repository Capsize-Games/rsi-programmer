# Prediction inversion diagnostic

Negating a chart percentage changes its meaning, without changing the
forecasts. We also tested a different operation: replacing each forecast
`p` with `1 - p` while keeping its target unchanged.

The diagnostic was specified before execution in
`fixtures/audio_output_inversion_v1.json`. It uses the preserved original
first 30 monitoring minutes and checkpoint-30 probe. These recordings had
already been examined, so this is retrospective evidence.

| Adaptive SNN measurement | Original MSE | Inverted-prediction MSE | Change |
|---|---:|---:|---:|
| First 30 live minutes | 0.012324833 | 0.016726975 | 35.718% worse |
| Sealed probe at checkpoint 30 | 0.015021563 | 0.013607905 | 9.411% better |

The inverted probe prediction still has 16.253% higher error than the
original fixed SNN's probe MSE of 0.011705402. Thus output inversion helps
one retained recording while substantially hurting the ongoing recording;
it does not resolve adaptation and retention together.

All five original predictors were scored with the same transformation.
The corresponding ANN changes are 45.683% worse live and 7.524% better on
the probe. Exact control results are retained in
[`result.json`](../artifacts/audio-inversion-20261018/result.json).

As an algebraic check, complementing both prediction and target preserves
squared error for every arm. A waveform polarity flip would also leave the
current magnitude/power features unchanged; it is a separate operation.

The live calculation used the saved, causal forecasts and 53,850 scored
targets. Probe predictions were reconstructed from checkpoint-30 parameter
copies, with learning disabled, the original fit-only normalization and
the same 25-frame warmup. Original live and probe MSEs reproduce the
published report within floating-point tolerance. Parameter hashes remain
unchanged, and the 450 probe targets never enter training.

The diagnostic ran serially under the builder lock in 1.424 seconds.
Its compressed log, source fingerprints and execution receipt are under
`artifacts/audio-inversion-20261018/`. No fault-detection accuracy or
prospective improvement is established. The separate full-learning,
readout-only and training-memory comparison retains its frozen recipe;
this diagnostic does not select a transform using its future probe.

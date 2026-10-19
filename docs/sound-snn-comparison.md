# SNN sound learning and retention

The creature now uses a trained SNN to choose its sound action label.
Both the input synapses and class readout learn from explicitly labeled
teaching recordings. The previous example matcher receives the same
examples and fresh test clips as a comparison control. It cannot substitute
its label for the SNN's decision.

## Frozen generated comparison

The protocol was committed in `d2f0a21` before measurement. Each of three
SNN parameter seeds received three examples per family. The same 12 fresh
generated queries per family were used for every model and retention stage.
The generator varies pitch, gain, timing and noise. No query enters the
training set. The fixture's canonical SHA-256 is
`1f0be06f30f3010b1cd75d10fab4c8c676ab137c56f1e2dd046cfe4111839b83`.

| Stage | SNN, each seed | Previous matcher, each run |
| --- | ---: | ---: |
| Three taught sounds | 36/36 correct | 36/36 correct |
| Add fourth sound | 48/48 correct | 48/48 correct |
| Reload saved memory | 48/48 correct | 48/48 correct |

All three seeds produced those counts. The older three sounds kept their
36/36 score after adding the fourth. Saved example fingerprints were
retained, and reloading restored identical neural parameter hashes and
outcomes without fitting. All stage queries left parameters unchanged.
Before teaching, both backends rejected all 48 future-family queries.

The raw SNN choices also scored every known-family query correctly; the
support guard did not repair classification mistakes. Final fits changed
all 1,344 input parameters and 252–260 readout parameters relative to their
seeded initialization. The final training pass emitted 4,235–4,472 binary
spikes. Those spike counts describe one training pass, not lifetime activity.

This demonstrates learned neural classification and saved retention on
these generated sounds. Accuracy ties the matcher; it does not establish
an SNN advantage. There are 48 distinct generated query clips, reused
across seeds and stages, rather than hundreds of independent recordings.
Personal microphone accuracy still needs fresh recordings in the app.

## Unknown sounds and work

The SNN is a closed-set classifier. Before chime was taught, its raw output
assigned all 12 chime queries to existing labels in each seed. The support
guard rejected them. After all four sounds were taught, raw neural outputs
also assigned generated noise and the two far-frequency tones to known
labels. The guard rejected all nine such decisions across the three seeds;
silence was rejected before neural inference. The original matcher rejected
the same controls. This does not establish open-world unknown detection.

At four classes, measured mean decision time was **2.07 ms for SNN plus
guard** and **0.81 ms for the matcher**. Each denominator is 156 calls:
48 family queries plus four controls, repeated across three seeds. Times
exclude feature extraction and paired-record persistence; the SNN time
includes support checking. These are sequential wall-clock observations,
without repeated timing trials or a chip-energy measurement.

The 36 explicit teaching events across all seeds spent 6.912 seconds fitting
the SNN, and 7.351 seconds in the complete paired teaching calls. The
matcher's recorded insertion work totaled 0.00810 seconds; that insertion
figure excludes the combined model's persistence overhead. SNN training is
therefore an additional cost here. No speed or energy gain is claimed.

An initial plot divided inference totals by known queries while those
totals also included controls. That draft remains labeled in
`draft-wrong-timing-denominator/`. The final plot uses 156 timed decisions;
accuracy counts and saved raw timing totals did not change.

## Implementation and verification

The model runs 12 temporal steps through 64 leaky integrate-and-fire
neurons. Its learned readout consumes binary spike rates. Teaching uses
100 full-batch Adam epochs, a 0.025 learning rate, class-balanced loss and
surrogate gradients. Each teaching event refits from the configured seed
on all retained taught examples. Retention uses that replay memory; it is
not memory without rehearsal. Reopening loads trained weights directly.

Neural choices pass through a separate distance support guard. It only
rejects unsupported choices and never falls back to the matcher. A
regression forces an incorrect SNN choice while the matcher is right and
verifies that the matcher cannot rescue the action.

The affected desktop checks passed: 86 tests in 6.921 seconds, including
real Tk controls under isolated Xvfb, actual-SNN controller routing,
parameter changes in both synapse stages, persistence, migration,
transaction rollback, evaluation isolation and background fitting.
Independent aggregate verification checked 1,708 numeric assertions and
rejected three separately corrupted in-memory reports. The figure's 18
numeric values and denominators were checked against the saved report.
No full suite or remote CI was awaited; no tags or releases were created.

Evidence is in
[`artifacts/sound-snn-validation-20261019/`](
../artifacts/sound-snn-validation-20261019/).
The runtime source is scoped in the measurement receipt. Its only subsequent
model-source edit wrapped an error statement; identical AST hashes are
retained in `model-source-equivalence.json`. UI and controller changes do
not change this measured backend. No personal audio, feature traces or
learned parameter arrays are published.

# First sensing pilot: fan vibration changes

The first behavior is change detection from one vibration channel.
Recognition of motion/sounds and adaptive device control remain later
stages. This pilot tests a causal sensing/update loop using recorded data
before choosing or connecting a physical sensor and learning chip.

The authoritative, prospective protocol is
[`sensing_fan_v1.json`](../fixtures/sensing_fan_v1.json). Publish its freeze
before executing models or measuring held outcomes. Run it once, retain
negative results, and do not select another axis, threshold or parameter
recipe after inspecting held results.

## Data and scope

The [UCI Accelerometer dataset](https://archive.ics.uci.edu/dataset/846/accelerometer)
contains measurements from a physical fan/accelerometer rig, with three
blade-weight arrangements and 17 operating speeds. The CSV has 153,000
rows, grouped into one 3,000-row recording per arrangement/speed. Its
collection description specifies 20 ms sampling; timestamps here are
inferred from that description, not measured timestamp fields.

The first channel is `x`, fixed before outcomes. We do not use the paper's
extrapolated failure-time labels. The three configurations describe blade
weight arrangements; they do not establish healthy/faulty truth. Source
recordings contain no observed physical transitions between conditions.

Official archive SHA-256:
`00557d031124defb8cebc6ae62cb1833e4cadde546c6c2b1997a8925d92816b0`.
CSV SHA-256:
`296a94eaf40643e1188012a4ee37d0b2c397bbc04decd39cfbe9952b8c841674`.
Original files and attribution are retained under
[`data/sensing_fan_v1`](../data/sensing_fan_v1/SOURCE.txt).

Dataset citation: Scalabrini Sampaio, G., Rabello de Aguiar Vallim Filho, A.,
Santos de Silva, L., and Augusto da Silva, L. (2019), *Accelerometer*, UCI
Machine Learning Repository, DOI
[10.24432/C5Q61V](https://doi.org/10.24432/C5Q61V), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The original CSV is unmodified; channel selection, windowing,
normalization and synthetic joins are the transformations in this pilot.

## Chronology and controls

Take non-overlapping 10-sample windows and emit RMS only at each window's
end. Normalize with median and MAD fitted only on the designated training
prefixes, using the exact bounded transform in the fixture. No labels,
configuration/speed IDs, recording identity or future samples enter a
predictor.

Training uses configuration 1 at six prescribed speeds, first 30 seconds
of each recording, one pass. Calibration uses the following 15 seconds
from those recordings. Held tests use all three configurations at the
other eleven speeds. This is a held-operating-speed diagnostic from one
fan, not independent-device or independent-session validation.

For each held speed, construct two 60-second change streams from 15-second
blocks: A/B/A/C and A/C/A/B, where A is configuration 1 and B/C are 2/3.
The return to A uses a different source quarter. Also construct two
same-configuration sham streams using reordered quarters. Source offsets
are fixed in the fixture. Joins are synthetic; a successful detector has
not demonstrated detection of real physical transitions.

The seven arms are plastic SNN, readout-only SNN, frozen trained SNN,
plastic continuous control, persistence, EMA, and normalized LMS. Three
seeds initialize neural parameters; deterministic baselines repeat for
paired comparisons. There are 44 streams per arm/seed, or 924 rows, with
300 feature windows per stream. Reused clips, paired orders and seeds are
correlated repetitions, not new recordings.

Neural predictors use twelve units at three time scales and consume actual
elapsed nominal seconds. Incoming synapses and readout learn from the next
observed feature; the corresponding prediction must already exist before
that feature arrives. The readout-only and frozen controls isolate which
parameter updates matter. This local surrogate rule is not full recurrent
backpropagation or a verified on-chip implementation.

Every stream starts from a fresh copy of the arm's trained snapshot.
Temporal state and online parameter updates continue across joins. They
reset only at stream boundaries. Calibration copies use the same learning
flags but discard their updates before held evaluation.

## Alarms and acceptance

The surprise score is absolute error of the previous prediction. Record
the score and alarm before updating with the newly arrived feature.
Thresholds use one prescribed 99.5th-percentile calibration operating
point, separately for each arm/seed. No held-data sweep is allowed. A
five-window refractory interval limits repeated alarms. The first five
windows of each stream are unscored warmup for every arm.

The evaluator joins change labels afterward. Match at most one alarm to
each of the five windows beginning at a change; all other alarms are false
positives. Report event recall, delay, and sham false alarms per stable
minute, along with errors, per-speed results and model/work hashes.

The practical gate requires at least 80% event recall and at most one sham
false alarm per minute in every seed. The adaptation and SNN-specific
gates additionally require a ten-point recall improvement over frozen SNN
and the plastic continuous control, respectively, without more sham false
alarms. The competitive gate compares with eligible cheap baselines.
These are frozen descriptive pilot gates, not statistical independence
claims. Device energy, latency, chip fit and on-chip learning remain
unmeasured.

Keep preprocessing, model/detector processing, provenance/compression and
whole-process costs identifiable. CPU is software evidence only. All
model/test/benchmark execution uses the shared builder lock. Preserve the
existing graph benchmarks, historical conclusions and independent checks.

## Execution

The runtime entry point and the independent replay checker run from the
repository root. Use a new output directory; the runner refuses to replace
an existing measurement.

```sh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  -m scripts.benchmark_sensing --output artifacts/sensing-fan-v1-20261018
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  -m scripts.audit_sensing --bundle artifacts/sensing-fan-v1-20261018 \
  --output artifacts/sensing-fan-validation-20261018/independent-audit.json
```

The manifest binds the complete source-file matrix and exact output hashes.
Replay requires those source bytes. Boundary snapshots are hashed; learned
parameter arrays are not included in the published measurement files.

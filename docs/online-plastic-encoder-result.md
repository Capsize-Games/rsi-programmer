# Online encoder-learning result

Learning the incoming encoder weights did not meet the frozen improvement
gates. At the primary budget, the persistent plastic SNN accepted 391 of
864 repeated attempts versus 390 for the readout-only SNN. It chose the
correct first root lane 294 times versus 290, but used 3.10 times the
search CPU. These small count differences do not establish useful
persistent adaptation or an SNN-specific advantage.

## Scope and implementation

The protocol was published in code-v101 before model tests or measurement.
Its SHA-256 remains
`8fd37e1cd6904ab28dae47d48baf95beab68e0819b5a554d920c3d63aa1ebbf9`.
The measured runtime is code-v102,
`29bcb3dd822bffd1a89b936f7fc6f54c1ee19442`. All 202 measured source inputs
were unchanged during the run. The original corpus, schedule, earlier
protocols and measurements were preserved.

The matrix contains eight arms, six phase orders, three seeds and two
budgets: 13,824 episodes. Each arm and budget has 864 attempts on the same
48 known functions. These repetitions are not independent tasks, and no
unseen-function transfer is measured. Policies start fresh for each
order/seed/arm/budget. All arms receive the same task specification and use
the same graph generator and independent acceptance checker.

The opt-in plastic policy learns both the six-feature history projection
and the 18-value observed-context projection, alongside its readout.
Neuron count, six simulation ticks, membrane decay, spike threshold,
initial matrices, action sampler and readout rule remain matched. The
encoder rule uses current-decision surrogate traces and the pre-update
readout. It truncates derivative history at each choice; it does not
implement delayed-return or full recurrent credit assignment. See the
[protocol](benchmark-online-plastic-encoder.md) and
[current architecture](../architecture.md#11-additive-online-execution-controller).

## Primary budget: 144 candidates

All counts below are out of 864 repeated attempts per arm.

| Arm | Accepted | Correct first lane | Candidates executed | Search CPU, s | Total CPU, s |
|---|---:|---:|---:|---:|---:|
| Readout SNN | 390 | 290 | 91,141 | 10.066 | 12.425 |
| Frozen SNN | 395 | 286 | 88,115 | 8.596 | 10.883 |
| Reset readout SNN | 425 | 266 | 86,232 | 9.679 | 12.022 |
| Readout continuous control | 364 | 306 | 94,490 | 10.628 | 12.993 |
| Streaming | 414 | 288 | 88,434 | 1.978 | 1.978 |
| Plastic SNN | 391 | 294 | 91,002 | 31.173 | 34.590 |
| Reset plastic SNN | 425 | 266 | 86,232 | 28.848 | 32.139 |
| Plastic continuous control | 375 | 305 | 93,304 | 33.858 | 37.478 |

Relative to readout-only SNN, plastic SNN adds one acceptance (+0.12
percentage points) and four correct first lanes (+0.46 points), executes
139 fewer candidates, and costs 3.10 times search CPU or 2.78 times total
CPU. The changes vary in sign across seeds and phase orders.

Relative to the matched plastic continuous control, it accepts 16 more
attempts (+1.85 points), chooses the correct first lane 11 fewer times
(-1.27 points), and uses 7.7% less total CPU. Those mixed descriptive
differences do not pass the frozen SNN-specific gate. Streaming accepts
23 more attempts; plastic SNN uses 17.49 times streaming's total CPU.

## Full grammar: 432 candidates

Every arm accepts all 864 attempts. This saturated acceptance measure
cannot establish a learning improvement.

| Arm | Correct first lane | Candidates executed | Search CPU, s | Total CPU, s |
|---|---:|---:|---:|---:|
| Readout SNN | 275 | 146,739 | 16.220 | 18.580 |
| Frozen SNN | 284 | 151,270 | 14.813 | 17.087 |
| Reset readout SNN | 303 | 130,170 | 14.290 | 16.634 |
| Readout continuous control | 282 | 149,036 | 16.859 | 19.234 |
| Streaming | 288 | 150,768 | 3.430 | 3.430 |
| Plastic SNN | 283 | 145,748 | 49.192 | 52.601 |
| Reset plastic SNN | 297 | 129,272 | 42.714 | 45.997 |
| Plastic continuous control | 294 | 147,594 | 52.743 | 56.375 |

Plastic SNN executes 3.3% fewer candidates than streaming but costs 15.34
times total CPU, or 14.34 times search CPU. Relative to readout-only SNN,
it saves 991 candidates and adds eight correct first lanes, while using
3.03 times search CPU. It still chooses the correct first lane 11 fewer
times than the plastic continuous control.

Search CPU includes setup, reset, context preparation, execution, held
checking and explicitly timed state-hash diagnostics. Total CPU also
charges all read-only retention copies and probes. The full process used
442.028 CPU seconds and 442.107 wall seconds, including output compression
and other provenance work outside the per-arm timers. Peak process RSS
was 354,964 KiB; it is not a per-arm memory estimate. CPU measurements
describe this instrumented Python implementation, not device energy or a
diagnostic-free production implementation.

## Frozen gates and retention

The plastic-versus-readout improvement gate passes in 0/6 orders, the
SNN-specific gate in 0/6, and the streaming deployment gate in 0/6. The
auxiliary acceptance comparison with reset plastic SNN passes in 2/6
orders. Every all-order gate is false. Pooled counts do not override these
order-specific failures.

The 1,008 retention sweeps contain 48,384 cold context probes. They measure
root-lane preferences, not complete-program or held-out-solution retention.
At the primary budget (144), using only the earlier phases, which end at
task 16 or 32, plastic SNN's
mean probability for the earlier phase's correct root falls from 0.5241
at its phase endpoint to 0.3212 at the final boundary. The mean decline is
0.2029, with declines in all six orders. Readout-only SNN declines from
0.5249 to 0.3175, a mean decline of 0.2074. The regression remains.

At that budget, across the 48 contexts, plastic SNN's mean distinct cold
rate-code count
changes from 8.000 initially to 7.778 finally. Seed means change from
10/6/8 to 8.833/5.833/8.667. The fixed controls remain at 10/6/8; both
continuous controls retain 48 unique codes. Code counts are rounded-vector
diagnostics, not evidence about representation usefulness or causation.

## Verification and reproduction

The validation receipts and initial failures are retained in
[`online-plastic-encoder-validation-20261018`](../artifacts/online-plastic-encoder-validation-20261018/README.md).
The independent full replay passed all 13,824 episodes and all 1,008
retention sweeps, including all 48,384 cold probes. It independently
recomputes actions, execution outcomes, plastic updates, state hashes,
retention probabilities and summary gates. Exact legacy-control comparison
also passed all 8,640 old-control rows against the prior observed-context
bundle: actions, predictions, rewards, readout hashes, RNG offsets,
acceptance and semantic work counters match. Timings, new model hashes and
new plasticity counters are explicitly excluded from that old-source
equivalence claim. Receipts are retained with the evidence bundle.

The final local runs passed 91 affected tests in 0.382 seconds and all 465
fast tests in 234.550 seconds, without skips or exclusions. Tests include
independent PyTorch surrogate-autograd checks of both incoming-weight
updates, continuous finite differences, snapshots, resets, frozen-control
equivalence, evaluator purity and exact independent reference trajectories.
Nine scoped corruptions were rejected. Earlier checker failures and one
interrupted partial fast run are retained; only the final runs are passing
verdicts. All checks and measurements used the shared builder lock.

Remote CI for the measured code-v102 source is run
[39170132104](https://github.com/Capsize-Games/rsi-programmer/actions/runs/39170132104).
Both fast Python 3.12/3.13 jobs have passed; both full jobs were still
running at evidence publication. Slow/retry and local full discovery were
not rerun for this additive path. This evidence-only publication does not
supply a separate runtime or full-CI verdict.

```sh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  -m scripts.benchmark_online_plastic --output /tmp/rsi-plastic-fresh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  -m scripts.audit_online_plastic_encoder \
  --bundle artifacts/online-plastic-encoder-v1-20261018
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  -m scripts.compare_online_plastic_controls \
  --old-bundle artifacts/online-observed-context-v1-20261018 \
  --new-bundle artifacts/online-plastic-encoder-v1-20261018
```

The plot and its aggregated inputs are in
[`online-plastic-encoder-plots-20261018`](../artifacts/online-plastic-encoder-plots-20261018/online-plastic-encoder.json).
The raw bundle pins source, corpus, protocol, schedule and compressed-row
hashes. No learned weights or databases are included.

## Direction

Keep encoder plasticity opt-in as a tested experimental capability.
Do not promote it as an improvement to default search or evidence for chip
deployment. Useful persistent adaptation, causal encoding/reward diagnosis,
a real sensing/reference-feedback workload and a hardware precision,
learning and memory contract remain open in #84. The result supports
defining those contracts before scaling the network or adding memory.

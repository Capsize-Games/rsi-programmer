# Encoder-learning validation

The prospective protocol was published as code-v101 before model tests or outcome measurements. The pinned fixture remains unchanged. The implementation checkpoint is code-v102; the main benchmark and complete independent replay are pending at that checkpoint.

Final checks: 91 affected tests passed in 0.382 s; 465 fast tests passed in 234.550 s. No skips or exclusions. Source manifests contain 202 inputs and verify stable sources across each final run. `premeasurement-source-proof.json` binds the final source to the fast receipt. Tests and controls ran serially under `/tmp/rsi-programmer-builder.lock`.

The independent gradient test uses PyTorch autograd with hard forward spikes and an explicit surrogate backward derivative; it checks both history and context projection updates with nonzero membrane state and a masked action set. A continuous-control test uses finite differences. The independent reference policy does not call runtime choice or update methods.

The retained `affected-initial` run had two checker failures (reference floating-point order and the context-work key set); `affected-corrected` had the remaining second context-work comparison failure. These were corrected in the auditor without weakening snapshot equality. `affected-final` passes all 91 tests. `reference-drift-diagnostic.log` preserves the one locked multi-mode diagnostic. The first `fast` run was interrupted after 21.016 seconds because it was started before the remaining checker failure was noticed; no full-suite verdict is inferred from it. `fast-final` passes all 465 tests.

`negative-controls-final.json` records nine scoped corruptions and two valid semantic prefixes. It checks predictions, reward, encoder hash, context counters, retention rates/probabilities/source hash/neural work, and omitted source input. The valid prefixes reach the expected incomplete-matrix EOF; they are not full-bundle acceptance. The initial harness error is retained in `negative-controls-errors.log` and was corrected by handling that expected EOF. Measurement files are not used or modified by these controls.

The source/test commands are recorded exactly in their JSON receipts. `checkpoint-runner.py` retains the source-hash wrapper. For a fresh fast run, use a new receipt label:

```sh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  artifacts/online-plastic-encoder-validation-20261018/checkpoint-runner.py \
  fast-new /tmp/rsi-plastic-new-validation -- just test-fast
flock /tmp/rsi-programmer-builder.lock env PYTHONPATH=. .venv/bin/python \
  artifacts/online-plastic-encoder-validation-20261018/negative-controls.py
```

Slow/retry and local full discovery were not rerun for this additive mode. Existing tests and real hub checks remain intact. New source CI needs its own verdict; older source passes cannot verify this checkpoint. Frozen historical results are preserved, and no performance gain is claimed by software tests.

## Completed measurement and replay

The code-v102 runtime completed all 13,824 episodes in 442.107 seconds
wall / 442.028 seconds process CPU. Independent replay passed all episodes
and all 1,008 retention groups (48,384 cold probes). `audit-main.json`
records the frozen-input and raw-row hashes; its error log is empty.
`control-equivalence.json` verifies all 8,640 legacy-control rows against
the earlier context-conditioned bundle, excluding timing and the new
model/plasticity fields listed in that receipt.

`final-source-proof.json` binds all 202 measurement inputs to the final
fast receipt, current checkout and archived code-v102 source. Runtime,
tests, scripts, fixtures, dependencies and workflow are unchanged in the
evidence publication. The result is in
[online-plastic-encoder-result.md](../../docs/online-plastic-encoder-result.md).
None of the three principal gates passes in any order; the auxiliary
coverage comparison with reset plastic SNN passes in two orders only.

`ci-evidence-snapshot.json` retains the measured source's current CI state.
Both fast jobs passed: Python 3.12 ran 465 tests in 457.131 seconds and
Python 3.13 ran 465 in 486.895 seconds. Exact completed job logs are gzip
compressed; raw and compressed hashes are in `ci-fast-log-receipt.json`.
Both full jobs were still in progress at this snapshot. These CI results
belong to code-v102, not to earlier runtime or auditor scopes.

`retention-code-counts.json` independently recounts unique rounded cold
rate vectors at the primary budget's initial and final boundaries.
The retained Luna analysis notes describe preferences and code counts;
their pre-audit status statements are historical, not the final verdict.

The plot helper originally expected a nonexistent order-level task count.
It was corrected to validate each pooled arm's 144 tasks and three
48-task seed records. Initial figure layout also needed footer spacing.
These were downstream plot corrections only; source and measurements did
not change. The final PNG/SVG were inspected, and the helper's aggregates
match the independently reviewed summary counts and ratios.

The plot was generated with isolated uv tooling, without adding a project
dependency. Reproduce it in a new output directory:

```sh
flock /tmp/rsi-programmer-builder.lock uv run --isolated --no-project \
  --with matplotlib --python .venv/bin/python \
  artifacts/online-plastic-encoder-validation-20261018/plot-results.py \
  --bundle artifacts/online-plastic-encoder-v1-20261018 \
  --output /tmp/rsi-plastic-plot-fresh
```

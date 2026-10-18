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

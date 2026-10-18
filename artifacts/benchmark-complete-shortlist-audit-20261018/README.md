# Complete-shortlist independent audit

The scripts in this folder audit the frozen complete-fallback shortlist protocol and its result bundle without importing the project runner, solver, trainer, or summary module. The main auditor verifies pinned inputs, the full ordered timed and memory schedules, row uniqueness, model/source snapshots, independently evaluated graph acceptance, repeat behavior, work-counter partitions, memory child matching, and a full recomputation of the published cost summaries and decisions.

The frozen experiment covers the declared v2 task corpus and controls only. It does not establish transfer to another task distribution, total-memory bounds, or energy efficiency. The active descriptor/cheap-score block and selected feature vectors have counters and protocol caps; retained and pending expression layers remain additional storage. An interrupted layer may be discarded when the candidate cap is reached.

`duplicate_signature_evaluations == 0` means an identical descriptor identity was not executed again. It does not mean distinct expressions cannot produce equal observed-output vectors. Those collisions are handled by the canonical retention counters. `descriptor_replays == 0` means a descriptor was not generated again; it does not mean a retained expression is visited only once as an operand.

The train-only selector audit was run read-only, without the builder lock, while the lock was free. Exact command:

```sh
python artifacts/benchmark-complete-shortlist-audit-20261018/selection_audit.py --output artifacts/benchmark-complete-shortlist-audit-20261018/selection-audit.json
```

It exited 0 in approximately 0.2 seconds. It checked 528 selector rows and independently reproduced the selected 256/16 configuration, all-control full coverage, and external selection charge of 56,792,751,000 CPU ns and 55,770,510,319 wall ns. See `selection-audit.json` and `selection-audit.sha256`.

The metadata-only preflight passed before evaluation measurement. It reads only pinned fixture, corpus and selector evidence; it does not inspect the evaluation bundle. It confirmed the declared 11,808 timed rows and 960 isolated-memory rows. The exact commands, exits, durations, logs and final script hashes are in `audit-window-receipt.json`; the selector and preflight outputs are pinned in `selection-audit.sha256` and the receipt.

```sh
flock /tmp/rsi-programmer-builder.lock python artifacts/benchmark-complete-shortlist-audit-20261018/audit_window_runner.py
```

The main bundle audit and corruption controls remain pending benchmark completion and serial lock availability. Hold the lock for the audit and controls:

```sh
flock /tmp/rsi-programmer-builder.lock bash -c '
  python artifacts/benchmark-complete-shortlist-audit-20261018/complete_shortlist_audit.py artifacts/benchmark-complete-shortlist-v1 --output artifacts/benchmark-complete-shortlist-audit-20261018/main-audit.json &&
  python artifacts/benchmark-complete-shortlist-audit-20261018/negative_controls.py artifacts/benchmark-complete-shortlist-v1 --output artifacts/benchmark-complete-shortlist-audit-20261018/negative-controls.json
'
```

The negative controls operate on temporary copies, refresh the bundle file hashes, and must be rejected by semantic checks. No audit output in this folder currently represents a completed evaluation audit.

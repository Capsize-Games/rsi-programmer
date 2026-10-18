# Complete-shortlist independent audit

The scripts in this folder audit the frozen complete-fallback shortlist protocol and its result bundle without importing the project runner, solver, trainer, or summary module. The main auditor verifies pinned inputs, the full ordered timed and memory schedules, row uniqueness, model/source snapshots, independently evaluated graph acceptance, repeat behavior, work-counter partitions, memory child matching, and a full recomputation of the published cost summaries and decisions.

The frozen experiment covers the declared v2 task corpus and controls only. It does not establish transfer to another task distribution, total-memory bounds, or energy efficiency. The active descriptor/cheap-score block and selected feature vectors have counters and protocol caps; retained and pending expression layers remain additional storage. An interrupted layer may be discarded when the candidate cap is reached.

`duplicate_signature_evaluations == 0` means an identical descriptor identity was not executed again. It does not mean distinct expressions cannot produce equal observed-output vectors. Those collisions are handled by the canonical retention counters. `descriptor_replays == 0` means a descriptor was not generated again; it does not mean a retained expression is visited only once as an operand.

The first train-only selector audit was run read-only, without the builder lock while it was free. It was rerun under the authorized lock window after formatting; the successful exact command and duration are in `audit-window-receipt.json`.

```sh
python artifacts/benchmark-complete-shortlist-audit-20261018/selection_audit.py --output artifacts/benchmark-complete-shortlist-audit-20261018/selection-audit.json
```

The final selector audit exited 0 in 0.158164 seconds. It checked 528 selector rows and independently reproduced the selected 256/16 configuration, all-control full coverage, and external selection charge of 56,792,751,000 CPU ns and 55,770,510,319 wall ns. See `selection-audit.json` and `selection-audit.sha256`.

The metadata-only preflight passed before evaluation measurement. It reads only pinned fixture, corpus and selector evidence; it does not inspect the evaluation bundle. It confirmed the declared 11,808 timed rows and 960 isolated-memory rows. The exact commands, exits, durations, logs and final script hashes are in `audit-window-receipt.json`; hashes for the selector/preflight reports and scripts are in `selection-audit.sha256` and `audit-window.sha256`.

```sh
flock /tmp/rsi-programmer-builder.lock python artifacts/benchmark-complete-shortlist-audit-20261018/audit_window_runner.py
```

The complete shortlist benchmark is published. The independent auditor passed all 11,808 timed rows and 960 isolated-memory rows. It independently recomputed the cost summaries and found the cost gate unsupported; all three seeds retained full required-task success for the SNN comparison. These results cover only the frozen known-v2 corpus and do not establish transfer to another task distribution, total-memory bounds, or energy efficiency.

The first post-measurement audit attempt was retained and failed because of two auditor assumptions: it expected policy scoring to stop at the execution cap, although the frozen engine scores the full selected quota before cap-limited execution; and it treated the protocol's `none` model markers as absent training records, although `streaming` has an explicit no-fit record while `cheap_only` has none. The corrected auditor now independently checks `scored_descriptors == shortlist_descriptors` for policy engines, permits execution to stop partway through that set, requires the exact explicit no-fit streaming state, and requires the cheap-only record to be absent. The untouched first failure and corrected report are both retained.

All six corruption controls passed against temporary copies: duplicate raw key, reordered rows, wrong accepted graph, forged paired ratio (+0.25), forged selection charge, and forged cheap-counter partition. The source benchmark bundle was not modified by these controls.

The final audit and controls ran serially while holding the builder lock:

```sh
flock /tmp/rsi-programmer-builder.lock bash -c '
  python artifacts/benchmark-complete-shortlist-audit-20261018/complete_shortlist_audit.py artifacts/benchmark-complete-shortlist-v1-20261018 --output artifacts/benchmark-complete-shortlist-audit-20261018/main-audit-final-pass.json &&
  python artifacts/benchmark-complete-shortlist-audit-20261018/negative_controls.py artifacts/benchmark-complete-shortlist-v1-20261018 --output artifacts/benchmark-complete-shortlist-audit-20261018/negative-controls-final.json
'
```

The final main audit exited 0 in 1.602 seconds; the six controls exited 0 in 17.076 seconds. See `main-audit-final-pass.json`, `negative-controls-final.json`, their `.log`, `.exit`, and `.time` receipts, and the retained initial failure files. Final artifact-script formatting check passed (`4 files already formatted`).

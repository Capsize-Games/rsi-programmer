# Re-audit from the frozen code-v76 source tree

The benchmark executable was code-v74; data-only publication advanced Git
through code-v76, whose runtime and tests match code-v74. The v78 portability
change makes absent non-bounded peak measurements null. Re-audit the published
benchmark against the code-v76 source snapshot instead of the later working
tree. The audit scripts import only Python's standard library. Their repository
root is the directory two levels above the copied auditor, so keep the script
at the exact path below.

The prepared export is `/tmp/rsi-programmer-audit-code-v76`; it was created
from `git archive code-v76`. The published benchmark bundle remains at
`/home/joe/Projects/rsi-programmer/artifacts/benchmark-bounded-v1-verified-20261018`.
Run each command through the shared builder lock:

```sh
flock /tmp/rsi-programmer-builder.lock python -I -B \
  /tmp/rsi-programmer-audit-code-v76/artifacts/benchmark-bounded-audit-20261018/nullable_work_regression.py \
  /home/joe/Projects/rsi-programmer/artifacts/benchmark-bounded-v1-verified-20261018

flock /tmp/rsi-programmer-builder.lock python -I -B \
  /tmp/rsi-programmer-audit-code-v76/artifacts/benchmark-bounded-audit-20261018/bounded_audit.py \
  /home/joe/Projects/rsi-programmer/artifacts/benchmark-bounded-v1-verified-20261018 \
  --output /home/joe/Projects/rsi-programmer/artifacts/benchmark-bounded-audit-20261018/final-audit-nullable.json

flock /tmp/rsi-programmer-builder.lock python -I -B \
  /tmp/rsi-programmer-audit-code-v76/artifacts/benchmark-bounded-audit-20261018/negative_controls.py \
  /home/joe/Projects/rsi-programmer/artifacts/benchmark-bounded-v1-verified-20261018 \
  --output /home/joe/Projects/rsi-programmer/artifacts/benchmark-bounded-audit-20261018/negative-controls-nullable.json
```

`-I` prevents imports from the current code-v78 checkout or `PYTHONPATH`;
`-B` avoids bytecode writes. `bounded_audit.py` resolves `ROOT` from its own
path under the code-v76 export. The bundle argument and report paths are
absolute, so the working directory does not affect input selection. Keep
these nullable-amendment outputs separate from the original code-v77 receipt.

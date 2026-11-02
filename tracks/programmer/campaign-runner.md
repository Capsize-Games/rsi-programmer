# Unattended programming campaigns

The campaign supervisor runs bounded development recipes, records emitted
programs and execution evidence, and freezes a successful recipe before a
separate audit. Its current worker supports the existing bounded integer-list
statement language. Strings, records, general functions and repository repair
remain unsupported. A successful run within this family cannot establish
movement across programming families.

Run from the repository root:

```sh
.venv/bin/python -m scripts.programmer_campaign start \
  artifacts/campaign-initial-20261102 \
  --config campaigns/initial-list-probe.json
.venv/bin/python -m scripts.programmer_campaign status \
  artifacts/campaign-initial-20261102
.venv/bin/python -m scripts.programmer_campaign stop \
  artifacts/campaign-initial-20261102
.venv/bin/python -m scripts.programmer_campaign resume \
  artifacts/campaign-initial-20261102
```

The initial configuration has two development recipes and six audit functions.
Their signatures differ from the 512 training functions and three development
functions on the declared 24 probes. This is within-family separation, not a
universal inequivalence proof. Audit task IDs alone do not establish novelty.
The configured run has a 30-minute wall budget and at most two development
trials. A development run with no independently accepted programs stops without
opening the final audit.

The learner receives example pairs and constructs typed tokens. Both audit
conditions finish their choices before held rows are used for execution.
Python uses the existing Bubblewrap isolation. Every CURLEE invocation uses
its unchanged local resource guard. Qualification requires correct executions
and a rejected deliberately incorrect expected output. Compiler source is
neither changed nor copied into this repository.

The supervisor shares the RSI builder lock, admits heavy work only with at
least 9 GiB available, and terminates its worker group below 5 GiB. Admission
waits and worker timeouts are bounded. Resource interruption is recorded
separately from programming failure. Source and configuration hashes are
checked on resume and before stages. Completed receipts are preserved.
Runtime model files are excluded under `weights/campaigns/`.

Read `status.json`, `events.jsonl`, per-trial `result.json` and worker logs in
the run directory. Choices, source, execution records, checkpoint hashes and
costs are retained. Status reports distinguish resource blockage,
infrastructure errors, no measured gain and local gain. The current worker
always reports its family limit and does not claim cross-family generality;
retention across broader capabilities is not yet measured.

This version selects among registered recipes. Automatic curriculum invention,
automatic fragment-library growth and new language semantics are not
implemented. Repeated runs cannot overcome those representation limits.

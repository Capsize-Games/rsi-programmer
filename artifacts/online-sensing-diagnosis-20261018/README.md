# Sensing direction diagnosis

These are post-result diagnostics of the frozen code-v102 runtime and
code-v103 evidence. They do not train or tune a model, alter a benchmark,
or establish a gain on a new workload. Three Luna reviewers examined
architecture, task relevance and counterarguments; the retained numeric
analyses were checked again by the coordinating agent.

`cold_alias.py` reads saved retention rates. Across all 252 expected
primary-budget initial/final groups, exact-float and rounded-12 code
groupings agree. The initial arbitrary-lookup majority-root bound for the
plastic SNN is 32/48, 27/48 and 29/48 for seeds 0, 1 and 2. At the final
boundary, the mean bound is 60.53%, versus 61.11% initially. This is a bound
on root classification from those particular cold codes only. It is not
attained accuracy, a bound on later search, or a limit on another encoder.
Root labels are used for analysis only. Unique continuous codes do not
establish better learning.

`reward_alignment.py` verifies frozen corpus/source hashes and enumerates
all 432 candidates for each of the 48 tasks. It checks the uniquely
solvable root lane against exact target polynomials. On 9/48 tasks, a
non-solving lane has a higher full-lane average immediate reward than the
solving lane. Counts of alternative lanes beating the true lane overlap:
add 9, sub 5, mul 1 across all tasks. These are not per-true-root counts.
The averages include candidates the real early-stopping learner may never
see, so this is evidence of possible reward misalignment, not a causal
explanation of the measured losses.

`verify_snapshot.py` checks all 202 measured files against both the current
checkout and their measured source archive, verifies the episode SHA, and
recounts saved solutions and encoder updates. It does not instantiate a
model. No observed solution was rejected by the held/polynomial checker.
Primary plastic SNN updated its encoder on 91,002 attempts, with changed
encoder hashes on all 864 task rows. Parameter changes do not establish
useful learning.

Reproduction, from the repository root:

```sh
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  artifacts/online-sensing-diagnosis-20261018/cold_alias.py \
  --output /tmp/rsi-cold-check.json
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  artifacts/online-sensing-diagnosis-20261018/reward_alignment.py \
  --output /tmp/rsi-reward-check.json
flock /tmp/rsi-programmer-builder.lock .venv/bin/python \
  artifacts/online-sensing-diagnosis-20261018/verify_snapshot.py
```

The parent reruns reproduced both JSON results exactly. Source and frozen
data remain unchanged. No training, fast/slow/retry tests or local full
discovery were run for this artifact/documentation stage; the prior
runtime tests and source-specific CI evidence retain their original scope.

Recommendation and primary literature links are in
[online-learning-direction.md](../../docs/online-learning-direction.md#diagnosis-after-the-encoder-learning-comparison).
Change the primary task to a real timed sensing loop with observable
feedback; retain graph search as a secondary integration check. The next
sensor, workload and hardware contract are not yet selected or measured.

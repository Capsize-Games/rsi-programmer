# Historical gains measurements

These CSV rows and charts describe releases code-v12 through code-v29.
They retain the original measured values. They use the feature encodings,
acceptance paths, and path-specific `tried` counters of those releases.
The counters were not comparable units of signature work across paths;
these files do not establish a current search-efficiency improvement.

Current comparisons report `signature_evaluations`, operand evaluations,
ranked examinations, and commits separately. Candidate-signature counts
exclude recursive subtree calls, neural work, graph validation, and stash
operations. They are not end-to-end runtime measurements.

The [code-v45 measurements](../audit-amendments.md) contain independent
three-seed first-solver positions and direct signature instrumentation.
The [subsequent audit fixes](../audit-followup.md) record later changes
and their validation. No historical chart was silently recalculated or
presented as a measurement of the newer implementation.

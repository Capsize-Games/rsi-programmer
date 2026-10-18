# Two-seed and complete-memory checkpoint

This data-only snapshot retains the exact first 7,104 timed JSONL rows:
all cells for seeds zero and one, every task/control/cap/engine/repeat.
It also retains all 864 native/traced memory rows. The third training seed
is still running. No rows are selected using outcomes, and no performance
conclusion is made before the complete independent audit.

The parent checked unique prefix keys, identical repeated behavior, frozen
weights/states, zero row errors and completed model-state checks for both
seeds. These are checkpoint checks, not the final independent graph, work,
memory-to-timing, complete-grid, cost or corrupted-control audit. Raw prefix
and compressed hashes are retained so the final bundle can be compared.
The earlier native-RSS checkpoint stays unchanged.

Measured executable source remains code-v74. Source diff against that
revision is empty; subsequent publication commits add artifacts and
independent validation only. The CI startup test's environment-specific
historical-weight comparison is being amended separately. Its failure and
candidate test-only fix are retained; it has not changed this run or the
benchmark's strict model-reference guard. The full run and final audit
remain pending, and issue #81 remains open.

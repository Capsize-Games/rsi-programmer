# CPU-matched learned-ranking result (issue #90)

Full 864-cell run of the frozen protocol in
[benchmark-cpu-matched-protocol.md](benchmark-cpu-matched-protocol.md)
on 2026-11-05. Artifact:
`artifacts/benchmark-cpu-matched-v1/` (`raw.jsonl` 864 cells,
`aggregate.json`, `manifest.json`, `run90.log`). Machine load was
~5 during the run; all figures are process CPU seconds, which is
the protocol's unit, but they are single observations.

## Numbers

Every arm solved every task (48/48 per seed, seeds 0-2):

| Arm | Solves | Charged CPU/seed (s) | Fit CPU/seed (s) |
| --- | ---: | ---: | ---: |
| streaming | 48.0 | 1.906 | 0 |
| trained MLP | 48.0 | 7.025 | 0.19-0.22 |
| untrained MLP | 48.0 | 7.069 | 0 |
| heuristic | 48.0 | 7.341 | 0 |
| trained recurrent | 48.0 | 8.497 | 1.03-1.12 |
| seeded random | 48.0 | 12.046 | 0 |

Equal-total-CPU truncation (frozen task order, averaged over
seeds): against streaming's 1.906 s total, trained MLP reaches
0.299 solved fraction and trained recurrent 0.278, vs streaming
1.0 — `kill_vs_streaming` for both. Against heuristic, trained
MLP reaches 0.993 vs a truncated heuristic 0.958 (no kill);
trained recurrent 0.889 vs 1.0 (kill). The frozen `decide()`
requires kills against both baselines, so the mechanical verdict
is `CONTINUE` (`close_line: false`, recurrent arm closed).

## What this does and does not establish

- **Ceiling effect.** With a 10 CPU-second cell budget and a
  131072-candidate backstop, every arm including random solves
  everything. The bank as generated is too easy to discriminate
  solve rates; all discrimination comes from the CPU-truncation
  comparison. This is a validity limit of the executed test, not
  a solver result.
- **Against streaming there is no ambiguity.** At equal total
  CPU both learned arms solve roughly 0.3 of what streaming
  solves; unbounded, streaming reaches the same 48/48 at 3.7x
  less CPU than trained MLP, fit included. The pre-eval v2
  cap-based win (99/144 vs streaming 60 at a signature cap) does
  not survive CPU matching.
- **Trained equals untrained.** Trained and untrained MLP tie on
  solves and charged CPU (7.025 vs 7.069 s). No training benefit
  is visible in this matrix.
- **The surviving margin is a truncation artifact.** Trained
  MLP's non-kill against heuristic rests on 0.993 vs 0.958
  truncated fractions (about one to two tasks per seed, in frozen
  task order) between two arms that both solve everything.
  Heuristic is not the best classical comparator here; streaming
  is, and it is not close.

Mechanical verdict: **CONTINUE**, driven solely by that artifact.
Substantive verdict: **no CPU-matched learned-ranking advantage
is established**; the recurrent arm is closed by its own frozen
rule. Re-running with tighter budgets or harder tasks would be a
new protocol chosen after seeing results, so it is not done under
this issue. Any successor needs tasks where streaming does not
saturate and an explicit owner decision to spend on it.

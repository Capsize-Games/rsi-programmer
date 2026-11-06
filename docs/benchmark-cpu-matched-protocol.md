# CPU-matched learned-ranking protocol (Pursue 2, issue #90)

Frozen-before-measurement follow-up to `benchmark-pre-eval-v2`: the
same ranking question, but budgets are defined in CPU seconds rather
than candidate or signature caps, fit cost is charged and amortized
over the declared volume, and there is NO SNN arm.

## Kill criterion (verbatim)

at equal total CPU the learned ranker solves no more than
streaming/heuristic, or its solve advantage needs more than 2x
streaming's time => close line

A cap win that loses on time is not a win.

## Design

- Bank: 72 fresh integer-polynomial functions (24 train, 48 eval),
  same enumerator and selection code path as pre-eval v2 with fresh
  seeds, minus the 102 exact functions already banked in v1/v2.
  Overlap is zero by construction and re-verified by the runner.
- Arms (6): streaming, heuristic, seeded random, untrained MLP,
  trained MLP, trained recurrent. No SNN arm by owner direction.
- Seeds 0, 1, 2 for every learned arm; models are frozen after fit.
- Every arm gets an identical 10.0 CPU-second budget per cell,
  enforced by a process CPU timer; over-budget cells fail, never
  retry. A 131072 candidate backstop is a non-binding safety net.
- Fit CPU is measured per learned arm and seed, charged once, and
  amortized as fit_cpu/48 on each of that seed's eval cells. The
  declared volume is the actual evaluation volume (144 cells).
- Comparison: per seed, accumulate charged CPU in frozen task
  order, truncate the higher-total arm at the lower arm's total,
  compare accepted fractions, average over seeds.

## Frozen hashes

- Protocol: `fixtures/benchmark_cpu_matched_v1.json`
  `a661eddce10671bea4666185d1c6e646ebb3dd5072a8da159b396ace6eb43a14`
- Corpus: `fixtures/benchmark_cpu_matched_corpus_v1.json`
  `494e5d04bee656afb090a06245a2cc45c1a048dddfa21bdeac2f1c0738a8fd6d`
- Pinned in `rsi_programmer/cpu_matched_frozen.py`; the runner
  rejects any byte drift in protocol, corpus or generators.

## Full run (serialized phase)

```sh
python scripts/benchmark_cpu_matched.py --output artifacts/benchmark-cpu-matched-v1
```

Estimated wall time: about 5 minutes single-threaded
(torch threads=1; 864 cells at ~0.1-0.6 s each plus ~37 s of
measured fits across 3 seeds, from pre-freeze probes).

## Smoke validation (this issue, tiny subset)

```sh
python scripts/benchmark_cpu_matched.py --output /tmp/cm_smoke --smoke
```

One seed, two tasks, 12 cells, exit 0, verdict machinery runs.
Not evidence: on two universally solved tasks every arm ties at
equal CPU, so the mechanical verdict is CLOSE by construction.

#!/usr/bin/env python3
"""Unattended self-edit cycles: propose, fence, promote, record.

Each cycle takes the next candidate trainer edit in seeded order,
runs it through the fenced full-eval gate (proxy screening only
ever applies when a proven edit-screener is registered; otherwise
the record says unproven and evaluation is full), then promotes or
rolls back with lineage on every outcome. Stops on exhausted
budgets or two consecutive non-improving full-evals. Dry runs plan
without touching trainers or promoting anything.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

from rsi_programmer.edits import DEFAULT_CONFIG, all_edits
from rsi_programmer.edits import describe
from rsi_programmer.self_gate import SelfGate

Scores = dict[str, list[float]]
Trainer = Callable[[dict[str, Any], int, str], Scores]


class RsiLoop:
    """Sequential self-edit driver with stop rules."""

    def __init__(self, workdir: str, ledger: str, trace: str,
                 cycles: int = 3, wall_s: float = 600.0,
                 seed: int = 0, dry_run: bool = False) -> None:
        """Bind fence, ledger, trace path, and stop budgets."""
        if cycles < 1 or wall_s <= 0:
            raise ValueError("cycles and wall must be positive")
        self._gate = SelfGate(workdir, ledger,
                              mission="rsi-loop", budget=cycles)
        self._trace = trace
        self._cycles = cycles
        self._wall_s = wall_s
        self._seed = seed
        self._dry = dry_run
        self._start = time.perf_counter()
        self._records: list[dict[str, Any]] = []

    def close(self) -> None:
        """Flush the gate ledger handle."""
        self._gate.close()

    def _emit(self, record: dict[str, Any]) -> None:
        """Append one JSONL cycle record."""
        self._records.append(record)
        with open(self._trace, "a") as handle:
            handle.write(json.dumps(record, sort_keys=True)
                         + "\n")

    def _over(self) -> bool:
        """Wall budget exhausted."""
        return time.perf_counter() - self._start >= self._wall_s

    def run(self, config: dict[str, Any],
            trainer: Trainer) -> dict[str, Any]:
        """Drive cycles until a stop rule fires; report."""
        rng = random.Random(self._seed)
        pool = all_edits(dict(config))
        rng.shuffle(pool)
        current = dict(config)
        misses = 0
        done = 0
        promoted = 0
        for cycle in range(self._cycles):
            if self._over() or not pool:
                break
            edit = pool.pop(0)
            if self._dry:
                self._emit({"cycle": cycle, "edit": edit,
                            "diff": describe(edit, current),
                            "proxy": "unproven",
                            "decision": "dry-run",
                            "wall_s": 0.0})
                done += 1
                continue
            out = self._gate.propose(current, edit, trainer,
                                     seed=self._seed + cycle)
            if out["decision"] == "promote":
                current = out["config"]
                misses = 0
                promoted += 1
            else:
                misses += 1
            self._emit({"cycle": cycle, "edit": edit,
                        "diff": out["reason"],
                        "proxy": "unproven",
                        "decision": out["decision"],
                        "lineage": out["lineage"],
                        "mean_gain": out.get("mean_gain", 0.0),
                        "wall_s": 0.0})
            done += 1
            if misses >= 2:
                break
        return {"status": "complete" if done else "stopped",
                "cycles": done, "config": current,
                "promotions": promoted,
                "records": self._records}


def main(argv: list[str] | None = None) -> int:
    """Parse flags and run the loop with the default trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--wall-s", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workdir", default="/tmp/rsi-loop")
    parser.add_argument("--ledger",
                        default="/tmp/rsi-loop/loop.db")
    parser.add_argument("--trace",
                        default="/tmp/rsi-loop/cycles.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    loop = RsiLoop(args.workdir, args.ledger, args.trace,
                   cycles=args.cycles, wall_s=args.wall_s,
                   seed=args.seed, dry_run=args.dry_run)
    try:
        if args.dry_run:
            report = loop.run(dict(DEFAULT_CONFIG),
                              _refusing_trainer)
        else:
            from rsi_programmer.loop_trainer import train_scores
            report = loop.run(dict(DEFAULT_CONFIG),
                              train_scores)
    finally:
        loop.close()
    print(json.dumps({k: v for k, v in report.items()
                      if k != "records"}, indent=2,
                     sort_keys=True))
    return 0


def _refusing_trainer(config: dict[str, Any], seed: int,
                       workdir: str) -> Scores:
    """Dry-run trainer: must never be called."""
    raise AssertionError("dry run must not train")


if __name__ == "__main__":
    raise SystemExit(main())

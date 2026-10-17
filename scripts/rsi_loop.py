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

from rsi_programmer.autonomy import agency_profile, autonomy_ratio
from rsi_programmer.edits import DEFAULT_CONFIG, all_edits
from rsi_programmer.edits import describe
from rsi_programmer.proposer import EditProposer
from rsi_programmer.self_gate import SelfGate

Scores = dict[str, list[float]]
Trainer = Callable[[dict[str, Any], int, str], Scores]


class RsiLoop:
    """Sequential self-edit driver with stop rules."""

    def __init__(self, workdir: str, ledger: str, trace: str,
                 cycles: int = 3, wall_s: float = 600.0,
                 seed: int = 0, dry_run: bool = False,
                 proposer: EditProposer | None = None,
                 diagnostics: dict[str, Any] | None = None
                 ) -> None:
        """Bind fence, ledger, trace path, and stop budgets.

        Without a proposer the pool walks the alphabet in
        seeded order (the enumerated baseline, unchanged);
        with one, cycles spend in diagnostic rank order and
        each record carries its provenance.
        """
        if cycles < 1 or wall_s <= 0:
            raise ValueError("cycles and wall must be positive")
        self._gate = SelfGate(workdir, ledger,
                              mission="rsi-loop", budget=cycles)
        self._trace = trace
        self._cycles = cycles
        self._wall_s = wall_s
        self._seed = seed
        self._dry = dry_run
        self._proposer = proposer
        self._diag = dict(diagnostics) if diagnostics else {}
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

    def _pool(self, config: dict[str, Any]
              ) -> tuple[list[dict[str, Any]],
                         list[dict[str, Any]]]:
        """Candidate edits plus parallel provenance records.

        Authored order from the proposer when bound, else the
        seeded alphabet walk. Provenance slots align with the
        pool; baseline slots carry the seeded-order rule.
        """
        if self._proposer is None:
            rng = random.Random(self._seed)
            pool = all_edits(dict(config))
            rng.shuffle(pool)
            prov: list[dict[str, Any]] = [
                {"rule": "seeded-order",
                 "signal": "seed=%d" % self._seed}
                for _ in pool]
            return pool, prov
        rows = self._proposer.rank(dict(config), self._diag)
        pool = [row["edit"] for row in rows]
        prov = [row["provenance"] for row in rows]
        return pool, prov

    def run(self, config: dict[str, Any],
            trainer: Trainer) -> dict[str, Any]:
        """Drive cycles until a stop rule fires; report."""
        pool, prov = self._pool(config)
        current = dict(config)
        misses = 0
        done = 0
        promoted = 0
        for cycle in range(self._cycles):
            if self._over() or not pool:
                break
            edit = pool.pop(0)
            origin = prov.pop(0)
            authored = self._proposer is not None
            if self._dry:
                self._emit({"cycle": cycle, "edit": edit,
                            "diff": describe(edit, current),
                            "proxy": "unproven",
                            "decision": "dry-run",
                            "authored": authored,
                            "provenance": origin,
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
                        "authored": authored,
                        "provenance": origin,
                        "lineage": out["lineage"],
                        "mean_gain": out.get("mean_gain", 0.0),
                        "wall_s": 0.0})
            done += 1
            if misses >= 2:
                break
        return {"status": "complete" if done else "stopped",
                "cycles": done, "config": current,
                "promotions": promoted,
                "authored_promotions": sum(
                    1 for r in self._records
                    if r["decision"] == "promote"
                    and r["authored"]),
                "autonomy": autonomy_ratio(
                    {"promoted": [r["mean_gain"] for r in
                                  self._records
                                  if r["decision"] == "promote"
                                  and "mean_gain" in r],
                     "refused": [r["mean_gain"] for r in
                                 self._records
                                 if r["decision"] == "refuse"
                                 and "mean_gain" in r]}),
                "agency": agency_profile(
                    [{"generation": r["cycle"],
                      "weights": [r.get("mean_gain", 0.0)],
                      "fitness": {"correctness": 1.0
                                  if r["decision"] == "promote"
                                  else 0.0}}
                     for r in self._records
                     if r["decision"] in ("promote",
                                          "refuse")]),
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
